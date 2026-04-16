#!/usr/bin/env python3
"""
Synthetic Engine B simulator.

This does not validate a real model. It validates the experimental logic before
spending time on the host machine:

- A target fact lives in a distant context sentence.
- A distractor may live near the tail.
- The "answer cloud" collapses across generation steps.
- Dynamic support should keep the target while discarding most filler.
- Local-tail support should fail distant retrieval.

The simulator is intentionally lightweight and uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


Vector = List[float]


@dataclass
class SimCase:
    name: str
    total_sentences: int
    target_index: int
    distractor_index: int | None
    local_tail: int


@dataclass
class StepResult:
    step: int
    target_kept: bool
    distractor_kept: bool
    kept_fraction: float
    support_agrees: bool
    confidence: float


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(max(1e-12, dot(a, a)))


def normalize(a: Vector) -> Vector:
    n = norm(a)
    return [x / n for x in a]


def add(*vectors: Vector) -> Vector:
    return [sum(items) for items in zip(*vectors)]


def scale(a: Vector, s: float) -> Vector:
    return [x * s for x in a]


def random_unit(rng: random.Random, dim: int) -> Vector:
    return normalize([rng.gauss(0.0, 1.0) for _ in range(dim)])


def noisy_around(rng: random.Random, center: Vector, noise: float) -> Vector:
    return normalize([x + rng.gauss(0.0, noise) for x in center])


def build_world(
    rng: random.Random,
    case: SimCase,
    dim: int,
    distractor_similarity: float,
) -> Tuple[List[Vector], Vector, Vector | None]:
    target = random_unit(rng, dim)
    distractor = None
    if case.distractor_index is not None:
        unrelated = random_unit(rng, dim)
        distractor = normalize(
            add(scale(target, distractor_similarity), scale(unrelated, 1.0 - distractor_similarity))
        )

    sentences = [random_unit(rng, dim) for _ in range(case.total_sentences)]
    sentences[case.target_index] = noisy_around(rng, target, noise=0.08)
    if case.distractor_index is not None and distractor is not None:
        sentences[case.distractor_index] = noisy_around(rng, distractor, noise=0.08)
    return sentences, target, distractor


def collapse_query(
    rng: random.Random,
    target: Vector,
    distractor: Vector | None,
    step: int,
    n_steps: int,
    noise_start: float,
    distractor_pull: float,
) -> Tuple[Vector, float]:
    progress = step / max(1, n_steps - 1)
    noise = noise_start * (1.0 - progress)
    confidence = 0.35 + 0.60 * progress

    components = [scale(target, 0.55 + 0.40 * progress)]
    if distractor is not None:
        components.append(scale(distractor, distractor_pull * (1.0 - progress)))
    components.append(scale(random_unit(rng, len(target)), noise))
    return normalize(add(*components)), confidence


def select_support(
    sentences: Sequence[Vector],
    query: Vector,
    keep_ratio: float,
    local_tail: int,
    support_mode: str,
    support_mass: float,
    support_temperature: float,
    min_keep: int,
) -> List[int]:
    n = len(sentences)
    max_keep = max(min_keep, math.ceil(n * keep_ratio))
    scored = sorted(
        ((i, dot(query, sent)) for i, sent in enumerate(sentences)),
        key=lambda item: item[1],
        reverse=True,
    )
    if support_mode == "ratio":
        keep_n = max_keep
    else:
        max_score = scored[0][1]
        weights = [
            math.exp((score - max_score) / max(1e-6, support_temperature))
            for _index, score in scored
        ]
        total_weight = max(1e-12, sum(weights))
        cumulative = 0.0
        keep_n = 0
        for weight in weights:
            cumulative += weight / total_weight
            keep_n += 1
            if keep_n >= min_keep and cumulative >= support_mass:
                break
            if keep_n >= max_keep:
                break
        keep_n = min(max_keep, max(min_keep, keep_n))

    keep = {i for i, _score in scored[:keep_n]}
    keep.update(range(max(0, n - local_tail), n))
    return sorted(keep)


def run_case(
    seed: int,
    case: SimCase,
    keep_ratio: float,
    dim: int,
    n_steps: int,
    distractor_similarity: float,
    distractor_pull: float,
    noise_start: float,
    support_mode: str,
    support_mass: float,
    support_temperature: float,
    min_keep: int,
) -> List[StepResult]:
    rng = random.Random(seed)
    sentences, target, distractor = build_world(
        rng=rng,
        case=case,
        dim=dim,
        distractor_similarity=distractor_similarity,
    )

    results: List[StepResult] = []
    for step in range(n_steps):
        query, confidence = collapse_query(
            rng=rng,
            target=target,
            distractor=distractor,
            step=step,
            n_steps=n_steps,
            noise_start=noise_start,
            distractor_pull=distractor_pull,
        )
        support = select_support(
            sentences=sentences,
            query=query,
            keep_ratio=keep_ratio,
            local_tail=case.local_tail,
            support_mode=support_mode,
            support_mass=support_mass,
            support_temperature=support_temperature,
            min_keep=min_keep,
        )
        target_kept = case.target_index in support
        distractor_kept = case.distractor_index in support if case.distractor_index is not None else False

        # In this synthetic world, the reduced support agrees with the full
        # oracle when the target fact survives and either the cloud has enough
        # confidence or the distractor was not retained.
        support_agrees = target_kept and (confidence >= 0.65 or not distractor_kept)
        results.append(
            StepResult(
                step=step,
                target_kept=target_kept,
                distractor_kept=distractor_kept,
                kept_fraction=len(support) / len(sentences),
                support_agrees=support_agrees,
                confidence=confidence,
            )
        )
    return results


def summarize(results: Sequence[StepResult]) -> Dict[str, float]:
    n = max(1, len(results))
    return {
        "support_agree": sum(r.support_agrees for r in results) / n,
        "target_kept": sum(r.target_kept for r in results) / n,
        "distractor_kept": sum(r.distractor_kept for r in results) / n,
        "avg_kept_fraction": sum(r.kept_fraction for r in results) / n,
        "d_conf": results[-1].confidence - results[0].confidence if len(results) > 1 else 0.0,
        "d_support": results[0].kept_fraction - results[-1].kept_fraction if len(results) > 1 else 0.0,
    }


def mean_dict(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(dicts)
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def fmt(value: float) -> str:
    return f"{value:6.2%}" if 0.0 <= value <= 1.0 else f"{value:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--keep_ratios", default="0.01,0.02,0.03,0.05,0.08,0.10")
    parser.add_argument("--distractor_similarity", type=float, default=0.72)
    parser.add_argument("--distractor_pull", type=float, default=0.35)
    parser.add_argument("--noise_start", type=float, default=0.65)
    parser.add_argument("--support_mode", choices=["ratio", "mass"], default="mass")
    parser.add_argument("--support_mass", type=float, default=0.80)
    parser.add_argument("--support_temperature", type=float, default=0.04)
    parser.add_argument("--min_keep", type=int, default=1)
    args = parser.parse_args()

    cases = [
        SimCase("local", total_sentences=160, target_index=156, distractor_index=None, local_tail=4),
        SimCase("far", total_sentences=220, target_index=55, distractor_index=None, local_tail=4),
        SimCase("adversarial", total_sentences=240, target_index=70, distractor_index=220, local_tail=4),
    ]
    ratios = [float(x) for x in args.keep_ratios.split(",") if x]

    print("Synthetic Engine B signal check")
    print("This validates harness logic only; it does not validate a real model.")
    print()
    header = (
        f"{'case':>12} {'keep':>6} {'agree':>8} {'target':>8} "
        f"{'distr':>8} {'kept':>8} {'d_conf':>8} {'d_sup':>8}"
    )
    print(header)
    print("-" * len(header))

    for case in cases:
        for ratio in ratios:
            summaries = []
            for seed in range(args.seeds):
                results = run_case(
                    seed=seed,
                    case=case,
                    keep_ratio=ratio,
                    dim=args.dim,
                    n_steps=args.steps,
                    distractor_similarity=args.distractor_similarity,
                    distractor_pull=args.distractor_pull,
                    noise_start=args.noise_start,
                    support_mode=args.support_mode,
                    support_mass=args.support_mass,
                    support_temperature=args.support_temperature,
                    min_keep=args.min_keep,
                )
                summaries.append(summarize(results))
            mean = mean_dict(summaries)
            print(
                f"{case.name:>12} {ratio:6.2%} "
                f"{fmt(mean['support_agree']):>8} "
                f"{fmt(mean['target_kept']):>8} "
                f"{fmt(mean['distractor_kept']):>8} "
                f"{fmt(mean['avg_kept_fraction']):>8} "
                f"{mean['d_conf']:+8.3f} "
                f"{mean['d_support']:+8.3f}"
            )


if __name__ == "__main__":
    main()
