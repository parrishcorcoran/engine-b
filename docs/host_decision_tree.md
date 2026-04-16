# Host Decision Tree

Use this after each host run. The goal is fast iteration: if a test passes,
move deeper; if it fails, change one thing and retry.

## Start

Run the synthetic pre-check locally on the host:

```bash
python measurements/synthetic_engine_b.py --seeds 100 --support_mode mass
```

Expected:

- `far` has high `agree`.
- `far` has low `kept`.
- `d_conf` is positive.
- `d_sup` is positive or at least non-negative.

If this fails, fix the harness before loading a model.

## Tier 0: Baseline Controls

Command:

```bash
python measurements/holographic_smoke_suite.py \
  --model /path/to/model \
  --only_case local,far,adversarial \
  --context_tokens 2200 \
  --max_new_tokens 6 \
  --screen_keep_ratio 0.03 \
  --support_mode mass
```

### If Baseline Fails

Symptoms:

- `baseline found=No`.

Next tests:

1. Increase `--max_new_tokens` to `12`.
2. Lower `--context_tokens` to `1200`.
3. Try `--only_case local`.
4. Use a stronger/chat-tuned model.
5. Stop Engine B testing until baseline retrieval works.

Meaning:

- The task/model/prompt is broken. Do not interpret support failures.

### If Local-Tail Control Fails On Local

Symptoms:

- `local_tail found=No` on `local`.

Next tests:

1. Increase `--local_tail_sentences` from `4` to `8`.
2. Increase `--max_new_tokens` to `12`.
3. Inspect generated answer formatting.
4. Lower context length.

Meaning:

- Prompt construction or answer parsing is probably bad.

### If Local-Tail Passes Far

Symptoms:

- `local_tail found=Yes` on `far`.

Next tests:

1. Move the target earlier in the prompt.
2. Increase context length.
3. Add stronger tail distractors.

Meaning:

- The far case is too easy. It is not proving dynamic memory.

## Tier 1: Dynamic Support Signal

Command:

```bash
python measurements/holographic_smoke_suite.py \
  --model /path/to/model \
  --only_case far \
  --context_tokens 2200 \
  --screen_keep_ratio 0.03 \
  --support_mode mass \
  --screen_lens trajectory
```

### If It Passes Strongly

Symptoms:

- `support_set found=Yes`.
- `support agree` is high.
- `avg_token_frac` is less than `0.10`.
- `target_kept` is high.

Next tests:

1. Sweep lower keep caps: `0.005`, `0.01`, `0.02`, `0.03`.
2. Increase context: `4096`, then `8192` if model supports it.
3. Run `adversarial`.
4. Build the causal deletion oracle.

Meaning:

- Engine B has first-order signal. Now find the memory knee and adversarial limits.

### If Answer Is Correct But Support Agreement Is Low

Symptoms:

- `support_set found=Yes`.
- `support agree` is low.

Next tests:

1. Compare distributions, not argmax only.
2. Increase `--support_mass` from `0.80` to `0.90`.
3. Increase `--screen_keep_ratio` from `0.03` to `0.05`.
4. Try `--screen_lens final`.

Meaning:

- Retrieval survives, but next-token equivalence is not stable yet. The answer
  task may be too coarse.

### If Target Is Not Kept

Symptoms:

- `target_kept` is low.
- `support_set found=No`.

Next tests:

1. Increase `--screen_keep_ratio`: `0.05`, `0.08`, `0.10`.
2. Increase `--support_mass`: `0.90`, `0.95`.
3. Compare `--screen_lens final` vs `trajectory`.
4. Try a query-biased lens: score sentences against final question tokens only.
5. Move to causal deletion oracle if none work.

Meaning:

- Current proxy does not identify relevant memory. This does not kill Engine B
  unless the causal oracle also fails.

### If Target Is Kept But Answer Fails

Symptoms:

- `target_kept` is high.
- `support_set found=No`.

Next tests:

1. Increase `--local_tail_sentences` to `8` or `12`.
2. Keep neighboring sentences around the target.
3. Preserve original sentence order and add position labels.
4. Try token-span support instead of sentence support.

Meaning:

- The support has evidence but lost context glue, position cues, or grammar.

### If Support Fraction Is Too High

Symptoms:

- `avg_token_frac` is greater than `0.20`.

Next tests:

1. Lower `--support_mass`: `0.70`, `0.60`.
2. Lower `--support_temperature`: `0.03`, `0.02`, `0.01`.
3. Lower `--screen_keep_ratio`.
4. Move to token-span support.

Meaning:

- Support is too diffuse. The lens is not sharp enough.

### If `d_conf` Is Negative

Symptoms:

- `d_conf < 0`.

Next tests:

1. Increase `--max_new_tokens`.
2. Use a deterministic answer prompt.
3. Use a known-format completion like `Password:`.
4. Test a sentence continuation task instead of needle retrieval.

Meaning:

- The answer cloud is not collapsing under this task. Engine B may still work,
  but this test does not express the physics.

### If `d_support` Is Negative

Symptoms:

- `d_support < 0`.

Next tests:

1. Lower `--support_mass`.
2. Lower `--support_temperature`.
3. Verify `--support_mode mass` is being used.
4. Inspect per-step support sizes.

Meaning:

- Memory demand is growing as answer unfolds, or support policy is selecting by
  uncertainty rather than collapse.

## Tier 2: Lens Search

Run:

```bash
for lens in final trajectory; do
  python measurements/holographic_smoke_suite.py \
    --model /path/to/model \
    --only_case far \
    --context_tokens 2200 \
    --screen_keep_ratio 0.03 \
    --support_mode mass \
    --screen_lens "$lens"
done
```

### If Final Beats Trajectory

Next tests:

1. Reduce trajectory delta weight.
2. Use final hidden state only for query, trajectory only for tie-breaks.
3. Add question-token pooling.

Meaning:

- Rotation signal is noisy in this model/layer choice.

### If Trajectory Beats Final

Next tests:

1. Sweep exit layers: `8`, `12`, `16`, `20`, `24`.
2. Add a three-layer trajectory if hidden taps are cheap.
3. Test adversarial.

Meaning:

- Boundary motion matters. This supports the orbital lens.

### If Both Fail

Next tests:

1. Move to causal deletion oracle.
2. Try token-span granularity.
3. Try semantic clause granularity.
4. Try attention-output similarity if attention weights are available.

Meaning:

- Cosine support is probably the wrong proxy.

## Tier 3: Adversarial Recency

Command:

```bash
python measurements/holographic_smoke_suite.py \
  --model /path/to/model \
  --only_case adversarial \
  --context_tokens 2200 \
  --screen_keep_ratio 0.03 \
  --support_mode mass
```

### If Adversarial Passes

Next tests:

1. Increase distractor similarity.
2. Add multiple tail distractors.
3. Increase context length.
4. Move to causal deletion oracle.

Meaning:

- Strong evidence that support is not just recency.

### If Distractor Stays And Target Drops

Next tests:

1. Add question-token conditioned scoring.
2. Penalize generic password-like sentences not matching the question subject.
3. Use causal deletion oracle to learn the real support.

Meaning:

- The support lens is lexical. Need query-conditioned support.

### If Both Target And Distractor Stay

Next tests:

1. Lower `--support_mass`.
2. Lower temperature.
3. Move to token spans.

Meaning:

- Support is not sharp enough.

## Tier 4: Causal Deletion Oracle

Build this only if Tier 1 has any signal or if all proxy lenses fail.

For each step:

1. Run full context and record logits.
2. Delete one sentence or span.
3. Rerun.
4. Score deletion by KL divergence or full-top-token logit drop.
5. Keep high-impact spans only.
6. Rerun on retained spans and compare to full logits.

### If Causal Oracle Passes

Symptoms:

- Sparse support preserves logits.
- Target fact has high causal delta.
- Filler has low causal delta.

Next tests:

1. Train/fit a cheap proxy to imitate the oracle.
2. Move from sentence spans to token spans.
3. Implement true KV mask with original positions preserved.

Meaning:

- Engine B is real; the remaining problem is learning the selector cheaply.

### If Causal Oracle Fails

Symptoms:

- Support is dense.
- Many filler chunks have meaningful causal delta.

Next tests:

1. Try easier deterministic tasks.
2. Try a different model.
3. Try longer answer prefixes where collapse is stronger.
4. If still dense, Engine B likely fails for this model/task.

Meaning:

- The useful memory is not sparse enough in this regime.

## Tier 5: True KV Mask

Only after causal oracle passes.

Requirements:

- Preserve original positions.
- Preserve RoPE phase.
- Mask attention logits in-place.
- Do not rewrite prompt.
- Compare logits step by step.

### If True KV Mask Passes

Next tests:

1. Measure accuracy at 90%, 95%, 97%, 99% retained-distribution fidelity.
2. Measure wall-clock and memory.
3. Replace oracle with learned selector.

### If True KV Mask Fails But Oracle Passed

Next tests:

1. Check RoPE positions.
2. Check mask shape and cache indexing.
3. Check local window preservation.
4. Compare masked logits against support-rerun logits.

Meaning:

- Implementation bug, not necessarily hypothesis failure.

## Summary Routing

Use this routing:

- Baseline fails: fix prompt/model.
- Local-tail passes far: make task harder.
- Target not kept: lens/proxy failure.
- Target kept but answer fails: context glue/position failure.
- Support high and agreement high: move to adversarial or causal oracle.
- Causal oracle sparse: Engine B signal is real.
- Causal oracle dense: Engine B fails in this regime.
