#!/usr/bin/env python3
"""
Cheap invariant simulations for Engine B.

These tests do not validate a model. They validate whether our experimental
logic is aimed at the real object: a low-dimensional boundary state whose
geometry should survive coordinate changes and control memory support.

The five invariants:

1. Support contraction: memory support shrinks as confidence rises.
2. Phase crossing: support widens during ambiguity and collapses afterward.
3. Basis robustness: the signal survives coordinate-preserving transforms and
   low-dimensional random projections.
4. Causal wedge: deleting outside support preserves behavior; deleting support
   breaks behavior.
5. A+B coupling: Engine A confidence should control Engine B support mass.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from synthetic_engine_b import (
    SimCase,
    StepResult,
    Vector,
    build_world,
    collapse_query,
    dot,
    mean_dict,
    normalize,
    run_case,
    select_support,
    summarize,
)


@dataclass
class InvariantResult:
    name: str
    passed: bool
    metrics: Dict[str, float]
    meaning: str
    next_if_fail: str


def pct(value: float) -> str:
    return f"{value:6.2%}"


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


def signed_permute(vec: Vector, permutation: Sequence[int], signs: Sequence[int]) -> Vector:
    return [signs[i] * vec[permutation[i]] for i in range(len(vec))]


def random_projection_matrix(rng: random.Random, in_dim: int, out_dim: int) -> List[Vector]:
    scale = 1.0 / math.sqrt(out_dim)
    return [[rng.gauss(0.0, scale) for _ in range(in_dim)] for _ in range(out_dim)]


def project(vec: Vector, matrix: Sequence[Vector]) -> Vector:
    return normalize([sum(weight * value for weight, value in zip(row, vec)) for row in matrix])


def confidence_to_support_mass(confidence: float) -> float:
    """Wide memory while uncertain, narrow memory after cloud collapse."""
    return max(0.52, min(0.96, 1.02 - 0.48 * confidence))


def run_coupled_case(
    seed: int,
    case: SimCase,
    dim: int,
    n_steps: int,
    keep_ratio: float,
    distractor_similarity: float,
    distractor_pull: float,
    noise_start: float,
    support_temperature: float,
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
        support_mass = confidence_to_support_mass(confidence)
        support = select_support(
            sentences=sentences,
            query=query,
            keep_ratio=keep_ratio,
            local_tail=case.local_tail,
            support_mode="mass",
            support_mass=support_mass,
            support_temperature=support_temperature,
            min_keep=1,
        )
        target_kept = case.target_index in support
        distractor_kept = case.distractor_index in support if case.distractor_index is not None else False
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


def support_contraction(seeds: int, dim: int) -> InvariantResult:
    case = SimCase("far", total_sentences=220, target_index=55, distractor_index=None, local_tail=4)
    summaries = [
        summarize(
            run_case(
                seed=seed,
                case=case,
                keep_ratio=0.10,
                dim=dim,
                n_steps=6,
                distractor_similarity=0.55,
                distractor_pull=0.0,
                noise_start=0.65,
                support_mode="mass",
                support_mass=0.80,
                support_temperature=0.04,
                min_keep=1,
            )
        )
        for seed in range(seeds)
    ]
    metrics = mean_dict(summaries)
    passed = (
        metrics["support_agree"] >= 0.95
        and metrics["target_kept"] >= 0.95
        and metrics["avg_kept_fraction"] <= 0.05
        and metrics["d_conf"] > 0.0
        and metrics["d_support"] >= 0.0
    )
    return InvariantResult(
        name="support contraction",
        passed=passed,
        metrics=metrics,
        meaning="As the answer cloud sharpens, mass-mode support should stay tiny and shrink or stabilize.",
        next_if_fail="Use mass support instead of fixed ratio, then lower support_temperature.",
    )


def phase_crossing(seeds: int, dim: int) -> InvariantResult:
    case = SimCase("adversarial", total_sentences=240, target_index=70, distractor_index=220, local_tail=4)
    early_both = []
    early_distractor = []
    late_target_only = []
    distractor_drop = []
    early_fraction = []
    late_fraction = []

    for seed in range(seeds):
        results = run_coupled_case(
            seed=seed,
            case=case,
            dim=dim,
            n_steps=7,
            keep_ratio=0.08,
            distractor_similarity=0.55,
            distractor_pull=0.60,
            noise_start=0.15,
            support_temperature=0.06,
        )
        first = results[0]
        last = results[-1]
        early_both.append(float(first.target_kept and first.distractor_kept))
        early_distractor.append(float(first.distractor_kept))
        late_target_only.append(float(last.target_kept and not last.distractor_kept))
        distractor_drop.append(float(first.distractor_kept and not last.distractor_kept))
        early_fraction.append(first.kept_fraction)
        late_fraction.append(last.kept_fraction)

    metrics = {
        "early_both": mean(early_both),
        "early_distractor": mean(early_distractor),
        "late_target_only": mean(late_target_only),
        "distractor_drop": mean(distractor_drop),
        "early_kept_fraction": mean(early_fraction),
        "late_kept_fraction": mean(late_fraction),
        "d_support": mean(early_fraction) - mean(late_fraction),
    }
    passed = (
        metrics["early_both"] >= 0.80
        and metrics["late_target_only"] >= 0.95
        and metrics["distractor_drop"] >= 0.90
        and metrics["d_support"] > 0.0
    )
    return InvariantResult(
        name="phase crossing",
        passed=passed,
        metrics=metrics,
        meaning="The selector should preserve superposition near ambiguity, then drop the distractor after collapse.",
        next_if_fail="Tie support_mass to confidence or add explicit query-conditioned scoring.",
    )


def basis_robustness(seeds: int, dim: int, projection_dim: int) -> InvariantResult:
    case = SimCase("far", total_sentences=220, target_index=55, distractor_index=None, local_tail=4)
    signed_same = []
    projected_target = []
    projected_jaccard = []

    for seed in range(seeds):
        rng = random.Random(seed)
        sentences, target, distractor = build_world(
            rng=rng,
            case=case,
            dim=dim,
            distractor_similarity=0.0,
        )
        query, _confidence = collapse_query(
            rng=rng,
            target=target,
            distractor=distractor,
            step=5,
            n_steps=6,
            noise_start=0.65,
            distractor_pull=0.0,
        )
        original = set(
            select_support(
                sentences=sentences,
                query=query,
                keep_ratio=0.05,
                local_tail=case.local_tail,
                support_mode="mass",
                support_mass=0.80,
                support_temperature=0.04,
                min_keep=1,
            )
        )

        permutation = list(range(dim))
        rng.shuffle(permutation)
        signs = [-1 if rng.random() < 0.5 else 1 for _ in range(dim)]
        permuted_sentences = [signed_permute(sentence, permutation, signs) for sentence in sentences]
        permuted_query = signed_permute(query, permutation, signs)
        permuted = set(
            select_support(
                sentences=permuted_sentences,
                query=permuted_query,
                keep_ratio=0.05,
                local_tail=case.local_tail,
                support_mode="mass",
                support_mass=0.80,
                support_temperature=0.04,
                min_keep=1,
            )
        )
        signed_same.append(float(permuted == original))

        matrix = random_projection_matrix(rng, in_dim=dim, out_dim=projection_dim)
        projected_sentences = [project(sentence, matrix) for sentence in sentences]
        projected_query = project(query, matrix)
        projected = set(
            select_support(
                sentences=projected_sentences,
                query=projected_query,
                keep_ratio=0.05,
                local_tail=case.local_tail,
                support_mode="mass",
                support_mass=0.80,
                support_temperature=0.04,
                min_keep=1,
            )
        )
        projected_target.append(float(case.target_index in projected))
        projected_jaccard.append(len(original & projected) / max(1, len(original | projected)))

    metrics = {
        "signed_permutation_same": mean(signed_same),
        "projected_target_kept": mean(projected_target),
        "projected_support_jaccard": mean(projected_jaccard),
    }
    passed = (
        metrics["signed_permutation_same"] == 1.0
        and metrics["projected_target_kept"] >= 0.95
        and metrics["projected_support_jaccard"] >= 0.90
    )
    return InvariantResult(
        name="basis robustness",
        passed=passed,
        metrics=metrics,
        meaning="A real manifold signal should survive coordinate changes; labels can rotate, support should not vanish.",
        next_if_fail="Replace coordinate-specific features with dot-product, phase, or causal-delta invariants.",
    )


def winner(
    query: Vector,
    sentences: Sequence[Vector],
    target_index: int,
    distractor_index: int | None,
    available: Set[int],
) -> str:
    best_index = max(available, key=lambda index: dot(query, sentences[index]))
    if best_index == target_index:
        return "target"
    if distractor_index is not None and best_index == distractor_index:
        return "distractor"
    return "filler"


def causal_wedge(seeds: int, dim: int) -> InvariantResult:
    case = SimCase("adversarial", total_sentences=240, target_index=70, distractor_index=220, local_tail=4)
    full_target = []
    outside_delete_preserves = []
    inside_delete_breaks = []
    target_kept = []
    kept_fraction = []

    for seed in range(seeds):
        rng = random.Random(seed)
        sentences, target, distractor = build_world(
            rng=rng,
            case=case,
            dim=dim,
            distractor_similarity=0.55,
        )
        query, _confidence = collapse_query(
            rng=rng,
            target=target,
            distractor=distractor,
            step=5,
            n_steps=6,
            noise_start=0.25,
            distractor_pull=0.60,
        )
        all_indices = set(range(len(sentences)))
        support = set(
            select_support(
                sentences=sentences,
                query=query,
                keep_ratio=0.05,
                local_tail=case.local_tail,
                support_mode="mass",
                support_mass=0.80,
                support_temperature=0.04,
                min_keep=1,
            )
        )
        complement = all_indices - support
        full = winner(query, sentences, case.target_index, case.distractor_index, all_indices)
        outside_deleted = winner(query, sentences, case.target_index, case.distractor_index, support)
        inside_deleted = winner(query, sentences, case.target_index, case.distractor_index, complement)

        full_target.append(float(full == "target"))
        outside_delete_preserves.append(float(outside_deleted == full))
        inside_delete_breaks.append(float(inside_deleted != full))
        target_kept.append(float(case.target_index in support))
        kept_fraction.append(len(support) / len(sentences))

    metrics = {
        "full_prefers_target": mean(full_target),
        "outside_delete_preserves": mean(outside_delete_preserves),
        "inside_delete_breaks": mean(inside_delete_breaks),
        "target_kept": mean(target_kept),
        "avg_kept_fraction": mean(kept_fraction),
    }
    passed = (
        metrics["full_prefers_target"] >= 0.95
        and metrics["outside_delete_preserves"] >= 0.95
        and metrics["inside_delete_breaks"] >= 0.95
        and metrics["avg_kept_fraction"] <= 0.05
    )
    return InvariantResult(
        name="causal wedge",
        passed=passed,
        metrics=metrics,
        meaning="The selected support should behave like an entanglement wedge: outside deletion is safe, inside deletion is not.",
        next_if_fail="Move from similarity support to a true causal deletion oracle.",
    )


def coupled_ab(seeds: int, dim: int) -> InvariantResult:
    case = SimCase("adversarial", total_sentences=240, target_index=70, distractor_index=220, local_tail=4)
    fixed_summaries = []
    coupled_summaries = []
    for seed in range(seeds):
        fixed_summaries.append(
            summarize(
                run_case(
                    seed=seed,
                    case=case,
                    keep_ratio=0.08,
                    dim=dim,
                    n_steps=7,
                    distractor_similarity=0.55,
                    distractor_pull=0.60,
                    noise_start=0.15,
                    support_mode="mass",
                    support_mass=0.90,
                    support_temperature=0.06,
                    min_keep=1,
                )
            )
        )
        coupled_summaries.append(
            summarize(
                run_coupled_case(
                    seed=seed,
                    case=case,
                    dim=dim,
                    n_steps=7,
                    keep_ratio=0.08,
                    distractor_similarity=0.55,
                    distractor_pull=0.60,
                    noise_start=0.15,
                    support_temperature=0.06,
                )
            )
        )

    fixed = mean_dict(fixed_summaries)
    coupled = mean_dict(coupled_summaries)
    metrics = {
        "fixed_agree": fixed["support_agree"],
        "coupled_agree": coupled["support_agree"],
        "fixed_kept_fraction": fixed["avg_kept_fraction"],
        "coupled_kept_fraction": coupled["avg_kept_fraction"],
        "fixed_distractor_kept": fixed["distractor_kept"],
        "coupled_distractor_kept": coupled["distractor_kept"],
        "coupled_d_support": coupled["d_support"],
    }
    passed = (
        metrics["coupled_agree"] > metrics["fixed_agree"] + 0.05
        and metrics["coupled_kept_fraction"] < metrics["fixed_kept_fraction"]
        and metrics["coupled_distractor_kept"] < metrics["fixed_distractor_kept"]
        and metrics["coupled_d_support"] > 0.0
    )
    return InvariantResult(
        name="A+B coupling",
        passed=passed,
        metrics=metrics,
        meaning="Compute confidence should shrink or widen memory support instead of using a fixed KV ratio.",
        next_if_fail="Retune confidence_to_support_mass or add a separate ambiguity detector.",
    )


def print_result(result: InvariantResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"{status:>4}  {result.name}")
    for key, value in result.metrics.items():
        if 0.0 <= value <= 1.0:
            rendered = pct(value)
        else:
            rendered = f"{value:+.4f}"
        print(f"      {key:>28}: {rendered}")
    print(f"      meaning: {result.meaning}")
    if not result.passed:
        print(f"      next: {result.next_if_fail}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--projection_dim", type=int, default=12)
    args = parser.parse_args()

    results = [
        support_contraction(seeds=args.seeds, dim=args.dim),
        phase_crossing(seeds=args.seeds, dim=args.dim),
        basis_robustness(seeds=args.seeds, dim=args.dim, projection_dim=args.projection_dim),
        causal_wedge(seeds=args.seeds, dim=args.dim),
        coupled_ab(seeds=args.seeds, dim=args.dim),
    ]

    print("Engine B invariant simulations")
    print("These are toy-world checks for the test logic, not model validation.")
    print()
    for result in results:
        print_result(result)

    passed = sum(result.passed for result in results)
    print(f"summary: {passed}/{len(results)} invariants passed")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
