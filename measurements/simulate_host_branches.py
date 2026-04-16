#!/usr/bin/env python3
"""
Simulate host decision-tree branches before spending model time.

This is not a model test. It is a decision-logic test:

- Fixture mode creates one synthetic metrics packet for each diagnosis branch.
- Grid mode sweeps synthetic Engine B parameters to find fragile regions.
- All mode does both.

Use this when changing the host tree or support policy. If the simulator routes
obvious failures incorrectly, the host handoff will waste time.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from synthetic_engine_b import SimCase, mean_dict, run_case, summarize


@dataclass
class Metrics:
    name: str
    case: str
    baseline_found: bool
    local_tail_found: bool
    support_found: bool
    support_agree: float
    avg_token_frac: float
    target_kept: float
    distractor_kept: float
    d_conf: float
    d_support: float
    answer_contains_expected: bool = False
    answer_is_tail_distractor: bool = False
    causal_sparse: Optional[bool] = None
    support_rerun_pass: Optional[bool] = None
    kv_mask_pass: Optional[bool] = None
    kv_mask_first_token_pass: Optional[bool] = None


@dataclass
class Route:
    label: str
    branch: str
    next_change: str
    meaning: str


def route(metrics: Metrics) -> Route:
    """Mirror the host decision tree in executable form."""
    if not metrics.baseline_found:
        if metrics.answer_contains_expected:
            return Route(
                "parser/vocab",
                "Tier 0 baseline",
                "Relax answer parser and inspect decoded tokens.",
                "The model likely produced the answer but extraction marked it wrong.",
            )
        if metrics.answer_is_tail_distractor:
            return Route(
                "recency overwrite",
                "Tier 0 baseline",
                "Strengthen prompt or move/add distractors after baseline is stable.",
                "The full model is choosing recent memory over the queried fact.",
            )
        return Route(
            "baseline amnesia",
            "Tier 0 baseline",
            "Lower context, increase max_new_tokens, or use a stronger model.",
            "The model/prompt cannot retrieve the fact; do not test Engine B yet.",
        )

    if metrics.case == "local" and not metrics.local_tail_found:
        return Route(
            "context glue loss",
            "Tier 0 local-tail",
            "Increase local_tail_sentences and inspect prompt formatting.",
            "Even local evidence is not enough; formatting or grammar glue is broken.",
        )

    if metrics.case == "far" and metrics.local_tail_found:
        return Route(
            "far too easy",
            "Tier 0 local-tail",
            "Move target earlier, increase context, or add stronger tail distractors.",
            "Recency-only memory solved the far case, so it is not a valid Engine B test.",
        )

    if metrics.causal_sparse is False:
        return Route(
            "dense causal memory",
            "Tier 4 causal oracle",
            "Try easier deterministic tasks or a different model; reject this regime if still dense.",
            "The useful memory is not sparse enough.",
        )

    if metrics.causal_sparse is True and metrics.support_rerun_pass is False:
        return Route(
            "proxy failure",
            "Tier 4 causal oracle",
            "Train or fit a selector to imitate causal support.",
            "Engine B signal exists, but the cheap proxy is wrong.",
        )

    if metrics.support_rerun_pass and metrics.kv_mask_pass is False:
        if metrics.kv_mask_first_token_pass:
            return Route(
                "cache drift",
                "Tier 5 KV mask",
                "Check per-step cache indices and update semantics.",
                "KV masking works initially then diverges.",
            )
        return Route(
            "position failure",
            "Tier 5 KV mask",
            "Check RoPE positions, attention-mask shape, and original cache indices.",
            "Support rerun works but in-place KV masking does not.",
        )

    if metrics.support_found and metrics.support_agree < 0.75:
        return Route(
            "semantic answer / token mismatch",
            "Tier 1 support",
            "Compare KL/top-5 and raise support_mass to 0.90.",
            "The answer survives but next-token equivalence is unstable.",
        )

    if metrics.target_kept < 0.50 and not metrics.support_found:
        if metrics.distractor_kept >= 0.50:
            return Route(
                "lexical capture",
                "Tier 1 support",
                "Add question-token conditioned scoring or move to causal oracle.",
                "The lens keeps lexical distractors instead of the queried fact.",
            )
        return Route(
            "support blindness",
            "Tier 1 support",
            "Sweep keep_ratio/support_mass/lens, then causal oracle.",
            "The proxy cannot see the relevant memory.",
        )

    if 0.50 <= metrics.target_kept < 0.80 and not metrics.support_found:
        if metrics.case == "adversarial" and metrics.distractor_kept >= 0.25:
            return Route(
                "partial phase crossing",
                "Tier 3 adversarial",
                "Try confidence-coupled support mass, then query-conditioned scoring.",
                "The selector partially sees the target but has not collapsed away the distractor.",
            )
        return Route(
            "weak support",
            "Tier 1 support",
            "Increase support_mass/keep_ratio or switch to multi-scale support.",
            "The target is sometimes visible, but the support lens is not stable.",
        )

    if metrics.target_kept >= 0.80 and not metrics.support_found:
        if metrics.case == "adversarial" and metrics.distractor_kept >= 0.30:
            return Route(
                "ambiguity preserved",
                "Tier 3 adversarial",
                "Lower support_mass/temperature or add query-conditioned scoring.",
                "The support kept both target and distractor, preserving superposition.",
            )
        return Route(
            "context glue loss",
            "Tier 1 support",
            "Keep neighbor sentences, add position labels, or use token spans.",
            "Evidence is retained but the reduced prompt lost structure needed to answer.",
        )

    if metrics.avg_token_frac > 0.20:
        return Route(
            "support too diffuse",
            "Tier 1 support",
            "Lower support_mass/temperature and move to token-span support.",
            "The support lens is not sharp enough to save memory.",
        )

    if metrics.d_conf <= 0.0:
        return Route(
            "collapse absent",
            "Tier 1 collapse",
            "Use a more deterministic completion prompt or inspect per-token confidence.",
            "The answer cloud is not becoming more deterministic.",
        )

    if metrics.d_support < -0.005:
        return Route(
            "support expansion",
            "Tier 1 collapse",
            "Lower support_mass/temperature and verify support_mode=mass.",
            "Memory support grows as generation unfolds.",
        )

    if (
        0.75 <= metrics.support_agree < 0.90
        and metrics.target_kept >= 0.75
        and metrics.avg_token_frac <= 0.10
    ):
        return Route(
            "borderline signal",
            "Tier 1 support",
            "Increase seeds/repeats, raise support_mass slightly, then rerun far.",
            "The signal is close but not strong enough to advance.",
        )

    if (
        metrics.support_found
        and metrics.support_agree >= 0.90
        and metrics.avg_token_frac <= 0.10
        and metrics.target_kept >= 0.90
        and metrics.d_conf > 0.0
    ):
        return Route(
            "first-order signal",
            "Tier 1 support",
            "Sweep lower keep caps, increase context, run adversarial, then causal oracle.",
            "Dynamic support is preserving behavior with small memory.",
        )

    return Route(
        "ambiguous",
        "manual review",
        "Inspect generated text and per-step support.",
        "Metrics do not match a clean branch.",
    )


def fixtures() -> List[Tuple[Metrics, str]]:
    """One fixture for each intended label."""
    return [
        (
            Metrics(
                name="parser_vocab",
                case="far",
                baseline_found=False,
                local_tail_found=False,
                support_found=False,
                support_agree=0.0,
                avg_token_frac=1.0,
                target_kept=1.0,
                distractor_kept=0.0,
                d_conf=0.0,
                d_support=0.0,
                answer_contains_expected=True,
            ),
            "parser/vocab",
        ),
        (
            Metrics("baseline_amnesia", "far", False, False, False, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            "baseline amnesia",
        ),
        (
            Metrics(
                "recency_overwrite",
                "far",
                False,
                True,
                False,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                answer_is_tail_distractor=True,
            ),
            "recency overwrite",
        ),
        (
            Metrics("local_tail_local_fail", "local", True, False, False, 0.0, 0.05, 1.0, 0.0, 0.2, 0.0),
            "context glue loss",
        ),
        (
            Metrics("far_too_easy", "far", True, True, True, 0.95, 0.03, 1.0, 0.0, 0.4, 0.0),
            "far too easy",
        ),
        (
            Metrics("answer_correct_low_agreement", "far", True, False, True, 0.40, 0.03, 1.0, 0.0, 0.5, 0.0),
            "semantic answer / token mismatch",
        ),
        (
            Metrics("support_blindness", "far", True, False, False, 0.10, 0.03, 0.0, 0.0, 0.5, 0.0),
            "support blindness",
        ),
        (
            Metrics("lexical_capture", "adversarial", True, False, False, 0.10, 0.03, 0.0, 1.0, 0.5, 0.0),
            "lexical capture",
        ),
        (
            Metrics("weak_support", "far", True, False, False, 0.55, 0.03, 0.65, 0.0, 0.5, 0.0),
            "weak support",
        ),
        (
            Metrics("partial_phase_crossing", "adversarial", True, False, False, 0.70, 0.03, 0.75, 0.35, 0.5, 0.0),
            "partial phase crossing",
        ),
        (
            Metrics("target_kept_answer_fails", "far", True, False, False, 0.20, 0.03, 1.0, 0.0, 0.5, 0.0),
            "context glue loss",
        ),
        (
            Metrics("ambiguity_preserved", "adversarial", True, False, False, 0.65, 0.03, 1.0, 0.80, 0.5, 0.0),
            "ambiguity preserved",
        ),
        (
            Metrics("support_too_diffuse", "far", True, False, True, 0.95, 0.35, 1.0, 0.0, 0.5, 0.0),
            "support too diffuse",
        ),
        (
            Metrics("collapse_absent", "far", True, False, True, 0.95, 0.03, 1.0, 0.0, -0.1, 0.0),
            "collapse absent",
        ),
        (
            Metrics("support_expansion", "far", True, False, True, 0.95, 0.03, 1.0, 0.0, 0.4, -0.05),
            "support expansion",
        ),
        (
            Metrics("dense_causal_memory", "far", True, False, False, 0.20, 0.80, 1.0, 0.0, 0.4, 0.0, causal_sparse=False),
            "dense causal memory",
        ),
        (
            Metrics("proxy_failure", "far", True, False, False, 0.20, 0.03, 0.0, 0.0, 0.4, 0.0, causal_sparse=True, support_rerun_pass=False),
            "proxy failure",
        ),
        (
            Metrics(
                "position_failure",
                "far",
                True,
                False,
                True,
                0.95,
                0.03,
                1.0,
                0.0,
                0.4,
                0.0,
                causal_sparse=True,
                support_rerun_pass=True,
                kv_mask_pass=False,
                kv_mask_first_token_pass=False,
            ),
            "position failure",
        ),
        (
            Metrics(
                "cache_drift",
                "far",
                True,
                False,
                True,
                0.95,
                0.03,
                1.0,
                0.0,
                0.4,
                0.0,
                causal_sparse=True,
                support_rerun_pass=True,
                kv_mask_pass=False,
                kv_mask_first_token_pass=True,
            ),
            "cache drift",
        ),
        (
            Metrics("borderline_signal", "far", True, False, True, 0.83, 0.03, 0.83, 0.0, 0.5, 0.0),
            "borderline signal",
        ),
        (
            Metrics("first_order_signal", "far", True, False, True, 0.96, 0.03, 1.0, 0.0, 0.5, 0.01),
            "first-order signal",
        ),
    ]


def run_fixtures() -> bool:
    print("Branch fixture simulation")
    print("-" * 96)
    print(f"{'scenario':>28} {'expected':>30} {'actual':>30} {'ok':>4}")
    ok_all = True
    for metrics, expected in fixtures():
        actual = route(metrics)
        ok = actual.label == expected
        ok_all = ok_all and ok
        print(f"{metrics.name:>28} {expected:>30} {actual.label:>30} {str(ok):>4}")
        if not ok:
            print(f"  branch={actual.branch} next={actual.next_change}")
    print()
    return ok_all


def metrics_from_synthetic(
    case: SimCase,
    keep_ratio: float,
    support_mass: float,
    support_temperature: float,
    distractor_similarity: float,
    distractor_pull: float,
    noise_start: float,
    seeds: int,
    dim: int,
) -> Metrics:
    summaries = []
    for seed in range(seeds):
        summaries.append(
            summarize(
                run_case(
                    seed=seed,
                    case=case,
                    keep_ratio=keep_ratio,
                    dim=dim,
                    n_steps=6,
                    distractor_similarity=distractor_similarity,
                    distractor_pull=distractor_pull,
                    noise_start=noise_start,
                    support_mode="mass",
                    support_mass=support_mass,
                    support_temperature=support_temperature,
                    min_keep=1,
                )
            )
        )
    mean = mean_dict(summaries)
    # Baseline/local-tail are assumed good in this synthetic sweep. We route the
    # support branch only.
    return Metrics(
        name=(
            f"{case.name}:kr={keep_ratio:.3f}:mass={support_mass:.2f}:"
            f"temp={support_temperature:.3f}:sim={distractor_similarity:.2f}:"
            f"pull={distractor_pull:.2f}:noise={noise_start:.2f}"
        ),
        case=case.name,
        baseline_found=True,
        local_tail_found=False if case.name != "local" else True,
        support_found=mean["support_agree"] >= 0.80,
        support_agree=mean["support_agree"],
        avg_token_frac=mean["avg_kept_fraction"],
        target_kept=mean["target_kept"],
        distractor_kept=mean["distractor_kept"],
        d_conf=mean["d_conf"],
        d_support=mean["d_support"],
    )


def run_grid(seeds: int, limit_print: int, profile: str, dim: int) -> Dict[str, int]:
    print("Synthetic parameter grid")
    print("-" * 96)
    cases = [
        SimCase("far", total_sentences=220, target_index=55, distractor_index=None, local_tail=4),
        SimCase("adversarial", total_sentences=240, target_index=70, distractor_index=220, local_tail=4),
    ]
    if profile == "quick":
        keep_ratios = [0.01, 0.03, 0.05]
        support_masses = [0.70, 0.80]
        temperatures = [0.02, 0.04]
        distractor_sims = [0.72, 0.88]
        distractor_pulls = [0.10, 0.60]
        noises = [0.25, 0.65]
    else:
        keep_ratios = [0.005, 0.01, 0.02, 0.03, 0.05]
        support_masses = [0.60, 0.70, 0.80, 0.90]
        temperatures = [0.02, 0.04, 0.08]
        distractor_sims = [0.55, 0.72, 0.88]
        distractor_pulls = [0.10, 0.35, 0.60]
        noises = [0.25, 0.65, 0.95]

    counts: Dict[str, int] = {}
    interesting: List[Tuple[Metrics, Route]] = []
    total = 0
    for case, keep_ratio, mass, temp, sim, pull, noise in itertools.product(
        cases,
        keep_ratios,
        support_masses,
        temperatures,
        distractor_sims,
        distractor_pulls,
        noises,
    ):
        if case.name == "far" and (sim != distractor_sims[0] or pull != distractor_pulls[0]):
            continue
        metrics = metrics_from_synthetic(
            case=case,
            keep_ratio=keep_ratio,
            support_mass=mass,
            support_temperature=temp,
            distractor_similarity=sim,
            distractor_pull=pull,
            noise_start=noise,
            seeds=seeds,
            dim=dim,
        )
        actual = route(metrics)
        counts[actual.label] = counts.get(actual.label, 0) + 1
        total += 1
        if actual.label != "first-order signal" and len(interesting) < limit_print:
            interesting.append((metrics, actual))

    print(f"grid_runs={total} seeds_per_run={seeds} profile={profile} dim={dim}")
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{label:>32}: {count}")
    print()

    if interesting:
        print("First non-success examples")
        print("-" * 96)
        for metrics, actual in interesting:
            print(
                f"{actual.label:>30} agree={metrics.support_agree:5.1%} "
                f"kept={metrics.avg_token_frac:5.1%} target={metrics.target_kept:5.1%} "
                f"distr={metrics.distractor_kept:5.1%} d_conf={metrics.d_conf:+.3f} "
                f"d_sup={metrics.d_support:+.3f} :: {metrics.name}"
            )
            print(f"  next: {actual.next_change}")
        print()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixtures", "grid", "all"], default="all")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--limit_print", type=int, default=16)
    parser.add_argument("--grid_profile", choices=["quick", "broad"], default="quick")
    parser.add_argument("--dim", type=int, default=12)
    args = parser.parse_args()

    ok = True
    if args.mode in {"fixtures", "all"}:
        ok = run_fixtures() and ok
    if args.mode in {"grid", "all"}:
        run_grid(
            seeds=args.seeds,
            limit_print=args.limit_print,
            profile=args.grid_profile,
            dim=args.dim,
        )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
