# 14D Engine Physics Map

This note maps the measured boundary-layer dimensions from
`orbit-inference/notebooks/dimension_map.md` onto the dual-engine hypothesis.

Important calibration:

- The original 14D map is an empirical PCA/feature basis, not a literal list of
  independent physical coordinates.
- Cross-model TwoNN later sharpened the claim: the clean per-sequence local
  manifold is about 7 dimensions. The practical 14D basis appears to include
  about 7 per-token difficulty dimensions plus about 7 cross-token/context
  structure dimensions.
- That split is exactly why Engine A and Engine B should be tested separately:
  Engine A reads cloud sharpness to skip compute; Engine B reads trajectory and
  resonance to keep only causal memory.

## Engine A: Compute / Sharpness Dimensions

These are the first seven per-token dimensions. They explain most of the gate
signal in the original compute-skip experiments.

| Dim | Measured anchors | Inference meaning | Physics lens | Known solution / model | Test implication |
|---:|---|---|---|---|---|
| 1 | `layer_7`, `layer_1`, `layer_5` | Layer dynamics | Black-hole radial depth | Ryu-Takayanagi minimal surface / quasinormal ringdown | If hidden state stabilizes by mid-layer, exit early. Treat layer depth as radial bulk reconstruction. |
| 2 | `layer_9`, `hnorm_0`, `cluster_0` | State location | Electron orbital | Hydrogen stationary states / eigenshells | Use cluster-conditioned bottlenecks. Some manifold regions should need less width. |
| 3 | `content_conf`, `logit_gap`, `sup_1` | Distribution sharpness | Electron-cloud collapse | Born rule / spectral gap | High top-1 mass or large logit gap means the cloud has collapsed; use CALM exit and sparse vocab. |
| 4 | `treuse_2`, `top10_cov`, `sup_1` | Lexical predictability | Resonant standing wave | Bound-state recurrence / cavity modes | Repeated lexical modes should need less compute and less memory. This dimension can also feed Engine B. |
| 5 | `cluster_1`, `layer_5`, `vel_0` | Cluster stability | Boundary layer | Prandtl-Blasius laminar layer | Stable clusters are laminar. Turbulent regions need full attention and full layers. |
| 6 | `cluster_1`, `sup_0`, `mom_0` | Superposition clarity | Quantum mixture purity | Density-matrix purity / decoherence | If candidate states are already separated, lower precision/rank is safe. |
| 7 | `mom_0`, `treuse_2`, `logit_gap` | Distribution shape | Wavepacket moments | Gaussian/coherent-state wavepacket | Use distribution skew/kurtosis to choose truncated softmax and confidence thresholds. |

## Engine B: Memory / Trajectory Dimensions

These are the seven cross-token/context dimensions. In the compute-gate PCA they
were smaller, but for KV replacement they may be the main signal.

| Dim | Measured anchors | Inference meaning | Physics lens | Known solution / model | Test implication |
|---:|---|---|---|---|---|
| 8 | `sup_0` | Candidate diversity | Electron-cloud orbital spread | Hydrogen orbital degeneracy / spherical harmonics | Keep semantic clusters, not individual lexical heavy hitters. Diversity says how wide the memory cloud must be. |
| 9 | `agreement_count` | Head consensus | Holographic redundancy | Quantum error-correcting code / entanglement wedge reconstruction | If independent heads agree, the boundary code is redundant and memory can shrink aggressively. |
| 10 | `treuse`, `nbr` | Temporal locality | Wavepacket autocorrelation | Born-Oppenheimer separation / perturbation theory | If the state barely moved, update memory as a delta instead of carrying the whole KV cloud. |
| 11 | `nbr`, `vel` with opposite signs | Stability tension | Shear/turbulence boundary | Landau-Zener transition / laminar-turbulent transition | A near-but-fast-moving state is dangerous. Keep wider support when locality and velocity disagree. |
| 12 | `logit_gap`, `mom`, `vel` | Decision clarity | Measurement separability | Stern-Gerlach separation / decoherence | Apply the 97% mask only after the decision axis separates. Before that, preserve competing supports. |
| 13 | `rg_2`, `hnorm`, `layer_9` | Scale coherence | Holographic RG | Wilsonian RG / AdS radial flow | Require token, sentence, and paragraph support to agree. If scales disagree, the selector is not causal yet. |
| 14 | `layer_5`, `cluster_1`, `rg_2` | Depth-width coupling | Minimal surface area | Ryu-Takayanagi entanglement wedge | Couple Engine A and B: early layer convergence should permit both fewer layers and a smaller memory wedge. |

## What This Narrows

The most promising Engine B path is not H2O-style historical importance. H2O is
mostly dimension 4/5: repeated lexical and attention-heavy tokens. Engine B
should target dimensions 8-14, especially 10, 11, 13, and 14:

- D10 says memory should be updated by state overlap, not age.
- D11 says the mask must widen near trajectory crossings, even if tokens look
  locally similar.
- D13 says support should survive across scales; a token-only match is not
  enough if sentence/paragraph resonance disagrees.
- D14 says compute confidence and memory sparsity should be coupled, not tuned
  as separate knobs.

The immediate host tests should therefore compare four lenses:

1. Final-state cosine support.
2. Trajectory-delta support.
3. Multi-scale support, scoring token span, sentence, and paragraph together.
4. A coupled A+B lens where high Engine A confidence lowers Engine B support
   mass, and low Engine A confidence widens memory.

## Faults To Watch Before The Host

- PCA axes rotate. A dimension label is a useful coordinate, not a law of
  nature.
- The first seven dimensions are not purely compute-only. D4 and D5 are obvious
  bridges into memory.
- The second seven dimensions were weak for compute skipping, but may be strong
  for retrieval. Do not reject them using the old skip metric.
- Fixed top-k masking is physically wrong. A collapsed cloud should need tiny
  support; a superposed cloud should keep more support. Use probability mass or
  phase coherence, not a static ratio.
- If target and distractor both survive, that is not necessarily failure. It may
  mean the selector preserved the correct superposition and the question lens
  did not collapse it.
