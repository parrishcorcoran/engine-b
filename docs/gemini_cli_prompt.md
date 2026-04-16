# Gemini CLI Handoff Prompt

Use this prompt on the HP Z8 G4.

```text
You are working in the `engine-b` repo.

Goal:
Test whether dynamic boundary-selected memory can replace most of the KV cache
without losing next-token behavior. Do not treat this as H2O. H2O keeps static
heavy hitters. Engine B should dynamically select the current resonant support
as the answer cloud collapses.

Read:
- README.md
- docs/fault_model.md
- docs/test_ladder.md
- measurements/holographic_smoke_suite.py

Work style:
- Start with the fastest tests.
- Run one variable at a time.
- If a test misses, try another lens or granularity quickly.
- Do not jump to CUDA or invasive KV masking until the causal oracle shows
  sparse support.
- Keep a log of commands, results, and interpretation.

First tests:
1. Verify baseline and local-tail controls.
2. Run the far case with 3% dynamic support.
3. Sweep support ratios: 1%, 2%, 3%, 5%, 8%, 10%.
4. Compare `--screen_lens final` vs `--screen_lens trajectory`.
5. Run adversarial recency only after far shows signal.

Before host-model tests:
- Run `python measurements/synthetic_engine_b.py --seeds 100`.
- Confirm the simulator has positive `d_conf`.
- Prefer `--support_mode mass`; fixed-ratio support cannot prove contraction.

Metrics to watch:
- baseline found
- local_tail found
- support_set found
- support agree
- avg_token_frac
- target_kept
- distractor_kept
- d_conf
- d_support

Interpretation:
- If support_set beats local_tail, that is evidence for dynamic support instead
  of recency cache.
- If support agreement is high at low retained fraction, Engine B has signal.
- If causal oracle support is sparse but cosine support fails, the proxy is bad
  but the physics signal is alive.
- If causal oracle support is dense, Engine B likely fails for that task/model.

Next implementation if Tier 1 has signal:
Build a causal deletion oracle:
- For each generation step, run full context.
- Delete each sentence or chunk one at a time.
- Rerun and measure KL/logit delta against full context.
- Rank support by causal delta.
- Measure how much context can be removed while preserving full logits.
```
