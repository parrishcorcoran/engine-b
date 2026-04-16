# Invariant Test Doctrine

The dimension is real. The names of the axes are provisional.

That distinction matters. A PCA axis can rotate, a feature can be renamed, and a
proxy can fail, but the low-dimensional boundary state should leave invariant
traces. Engine B should therefore test invariants before we spend host time on
true KV masking.

## The Object

The working claim:

> Transformer inference exposes a low-dimensional boundary state whose geometry
> predicts compute need and memory need better than raw token count, layer count,
> attention age, or KV-cache history.

This makes Engine B a dynamic causal-support problem, not a cache-eviction
problem. H2O asks what mattered historically. Engine B asks what remains inside
the current observable wavefunction.

## The Five Invariants

| Invariant | What Should Happen | Failure Means |
|---|---|---|
| Support contraction | As confidence rises, retained support shrinks or stabilizes without losing the target. | Fixed-ratio masking is hiding the signal, or the selector is measuring confusion instead of collapse. |
| Phase crossing | During ambiguity, support widens to preserve competing facts; after disambiguation, the distractor drops. | The lens is lexical, not query-conditioned, or the support mass schedule is wrong. |
| Basis robustness | Coordinate-preserving transforms and low-dimensional projections should preserve target support. | The signal is feature-coordinate artifact rather than manifold geometry. |
| Causal wedge | Deleting outside selected support preserves behavior; deleting selected support breaks behavior. | Similarity support is not causal; move to deletion oracle. |
| A+B coupling | Engine A confidence should control Engine B memory mass. Sharp cloud means smaller support; uncertain cloud means wider support. | Compute and memory gates are being tuned as separate knobs when they are the same boundary state. |

## Local Command

Run this before any host model test:

```bash
python measurements/invariant_simulations.py --seeds 100
```

Pass condition:

- All five invariants pass in the toy world.

If this fails, fix the harness or support schedule before running the Z8.

## Host Translation

For the real model, each invariant becomes a measurable host test:

| Toy invariant | Real-model metric |
|---|---|
| Support contraction | `d_conf > 0` and `d_support >= 0` while `support_agree` remains high. |
| Phase crossing | In adversarial cases, both target and distractor can survive early, but distractor support should fall after the answer disambiguates. |
| Basis robustness | Compare final-state, trajectory, projected, and causal-delta lenses; the target should survive across lenses. |
| Causal wedge | Sentence or token-span deletion oracle: outside deletion preserves logits, inside deletion changes logits. |
| A+B coupling | High early-exit confidence should predict lower retained KV fraction at matched agreement. |

## Design Rule

Do not optimize a fixed 3 percent mask. Optimize the collapse law.

A deterministic token may need less than 1 percent support. A genuinely
superposed token may need 10 percent support. The win is not that the ratio is
always tiny; the win is that the memory wedge is dynamic, causal, and tied to
the current boundary state.
