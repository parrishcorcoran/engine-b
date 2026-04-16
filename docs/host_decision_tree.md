# Host Decision Tree

Use this after each host run. The goal is fast iteration: if a test passes,
move deeper; if it fails, change one thing and retry.

## Start

Run the synthetic pre-check locally on the host:

```bash
python measurements/invariant_simulations.py --seeds 100
python measurements/synthetic_engine_b.py --seeds 100 --support_mode mass
```

Expected:

- All five invariant simulations pass.
- `far` has high `agree`.
- `far` has low `kept`.
- `d_conf` is positive.
- `d_sup` is positive or at least non-negative.

If this fails, fix the harness before loading a model. The invariant simulator
is the cheaper gate: it checks support contraction, phase crossing, basis
robustness, causal wedge behavior, and A+B coupling before any model time is
spent.

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

Diagnosis:

- If the generated text contains the right answer but `found=No`, it is an
  answer parser or vocabulary formatting issue.
- If the model outputs a synonym, extra punctuation, or tokenization artifact,
  relax the parser before changing Engine B.
- If the model answers with a tail fact instead of the target fact, it is
  prompt amnesia or recency overwrite.
- If the model refuses or rambles, the prompt is not deterministic enough.

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

Diagnosis:

- If `baseline` passes but `local_tail` fails on `local`, the local-tail window
  is too short or the prompt lost grammar glue.
- If the right fact is inside the local-tail support but the model misses it,
  the issue is not memory selection; it is prompt formatting or answer parsing.

### If Local-Tail Passes Far

Symptoms:

- `local_tail found=Yes` on `far`.

Next tests:

1. Move the target earlier in the prompt.
2. Increase context length.
3. Add stronger tail distractors.

Meaning:

- The far case is too easy. It is not proving dynamic memory.

Diagnosis:

- If local-tail passes because the target is accidentally near the tail, move
  the target earlier.
- If local-tail passes because filler repeats the answer pattern too strongly,
  diversify filler and add distractors.

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

Diagnosis:

- This does not yet prove true KV masking. It proves that a small dynamic
  boundary support can preserve behavior under prompt rerun.
- The next risk is that prompt rerun is hiding a RoPE/position problem that will
  appear in true KV masking.

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

Diagnosis:

- If exact answer is right but agreement is low, the model may be semantically
  stable but token-distribution unstable.
- Check KL or top-5 agreement before rejecting the signal.
- If disagreement mostly happens after the answer token, the memory mechanism
  may work and the suffix formatting is the problem.

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

Diagnosis:

- If the target never appears in top support, the resonance lens is blind.
- If target appears in top scores early but drops later, the answer cloud is
  rotating away from the evidence too soon.
- If target is missed only when distractors exist, this is lexical capture.
- If target is missed only at long context, this may be position decay or
  long-context amnesia.

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

Diagnosis:

- If adding neighboring sentences fixes it, the failure is context glue.
- If adding sentence indices fixes it, the failure is positional addressing.
- If token-span support fixes it, sentence averaging was too coarse.
- If nothing fixes it but full prompt works, the model needs distributed context
  rather than a single fact line.

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

Diagnosis:

- If low temperature still keeps many sentences, the scores are flat.
- Flat scores mean the boundary readout is not separating relevant memory.
- Move to causal deletion oracle to see if true causal support is sparse.

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

Diagnosis:

- If `d_conf` stays flat, the task may be a retrieval jump rather than a
  sentence-completion collapse.
- If `d_conf` drops after the answer token, increase `max_new_tokens` carefully
  and inspect per-token confidence.
- If `d_conf` is negative only for adversarial, the distractor is keeping the
  cloud in superposition.

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

Diagnosis:

- If support grows while confidence rises, mass temperature is too high or
  support mass is too high.
- If support grows while confidence falls, the model is genuinely uncertain.
- If fixed ratio mode is active, `d_support` is not meaningful.

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

Diagnosis:

- Try a later exit layer; early layers may not contain the right trajectory.
- Try final-only if the model has weak orbital rotation at the chosen depth.

### If Trajectory Beats Final

Next tests:

1. Sweep exit layers: `8`, `12`, `16`, `20`, `24`.
2. Add a three-layer trajectory if hidden taps are cheap.
3. Test adversarial.

Meaning:

- Boundary motion matters. This supports the orbital lens.

Diagnosis:

- If trajectory helps far but hurts adversarial, the trajectory signal is real
  but not query-conditioned enough.

### If Both Fail

Next tests:

1. Move to causal deletion oracle.
2. Try token-span granularity.
3. Try semantic clause granularity.
4. Try attention-output similarity if attention weights are available.

Meaning:

- Cosine support is probably the wrong proxy.

Diagnosis:

- If both lenses fail but baseline is strong, do not reject Engine B yet.
- Move to causal deletion oracle to separate proxy failure from hypothesis
  failure.

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

Diagnosis:

- Now increase distractor pressure until it breaks. The break pattern tells us
  what the selector is actually using.

### If Distractor Stays And Target Drops

Next tests:

1. Add question-token conditioned scoring.
2. Penalize generic password-like sentences not matching the question subject.
3. Use causal deletion oracle to learn the real support.

Meaning:

- The support lens is lexical. Need query-conditioned support.

Diagnosis:

- If the distractor has "password" and the target has "vault password", the
  selector may be matching the generic word rather than the requested entity.
- Add query-conditioned scoring from question tokens or causal oracle labels.

### If Both Target And Distractor Stay

Next tests:

1. Lower `--support_mass`.
2. Lower temperature.
3. Move to token spans.

Meaning:

- Support is not sharp enough.

Diagnosis:

- If both survive but answer is correct, the selector is safe but wasteful.
- If both survive and answer is wrong, the selector preserved ambiguity.

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

Diagnosis:

- If causal support is sparse and cosine support failed, the proxy is the
  problem, not the physics.
- If causal support is sparse only after answer prefix grows, dynamic support is
  real but needs generation-step conditioning.

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

Diagnosis:

- If causal support is dense for filler but sparse for deterministic prompts,
  Engine B is task-dependent rather than false.
- If causal support is dense even on deterministic completions, this model may
  distribute memory too broadly for cache deletion.

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

Diagnosis:

- If support rerun works but KV mask fails, suspect positions, RoPE phase,
  attention-mask shape, or cache index mismatch.
- If KV mask fails only after several generated tokens, suspect cache update
  drift rather than support selection.

## Summary Routing

Use this routing:

- Baseline fails: fix prompt/model.
- Local-tail passes far: make task harder.
- Target not kept: lens/proxy failure.
- Target kept but answer fails: context glue/position failure.
- Support high and agreement high: move to adversarial or causal oracle.
- Causal oracle sparse: Engine B signal is real.
- Causal oracle dense: Engine B fails in this regime.

## Failure Taxonomy

Use this table to name failures consistently.

| Failure | Symptom | Likely Cause | Next Move |
|---|---|---|---|
| Parser/vocab failure | Text contains answer but `found=No` | Answer extraction too strict | Relax parser, inspect decoded tokens |
| Baseline amnesia | Full prompt answers wrong fact | Model/prompt cannot retrieve | Lower context, strengthen prompt, stronger model |
| Recency overwrite | Tail distractor wins | Model prefers recent memory | Add adversarial controls, query-conditioned support |
| Lexical capture | Generic password line wins | Selector matches words not intent | Add question-token conditioning |
| Ambiguity preserved | Target and distractor both survive | Selector keeps superposition alive | Lower mass/temp or query-condition |
| Partial phase crossing | Target partly survives, distractor partly survives | Selector sees the crossing but does not collapse it | Try confidence-coupled support mass |
| Weak support | Target survives intermittently | Lens is close but unstable | Raise support mass or use multi-scale support |
| Support blindness | Target not kept | Proxy cannot see causal memory | Try lens sweep, then causal oracle |
| Borderline signal | Agreement and target survival are 75-89% | Signal exists but is unstable | Increase repeats, raise mass slightly |
| Context glue loss | Target kept but answer wrong | Support lacks neighboring syntax/context | Keep neighbors, add position labels |
| Position failure | Rerun works but KV mask fails | RoPE/original positions not preserved | Preserve positions, compare logits per step |
| Cache drift | KV mask works first token then fails | Cache update mismatch | Check per-step cache indices |
| Dense causal memory | Causal oracle needs many chunks | Memory not sparse for task/model | Try easier task/model or reject regime |
| Collapse absent | `d_conf <= 0` | Cloud not becoming deterministic | Use deterministic completion prompt |

## Result Log Template

Use this after each run:

```text
Run:
Command:
Model:
Case:

Metrics:
- baseline found:
- local_tail found:
- support_set found:
- support agree:
- avg_token_frac:
- target_kept:
- distractor_kept:
- d_conf:
- d_support:
- answer:

Diagnosis label:
One of: parser/vocab, baseline amnesia, recency overwrite, lexical capture,
ambiguity preserved, partial phase crossing, weak support, support blindness,
borderline signal, context glue loss, position failure, cache drift, dense
causal memory, collapse absent, first-order signal.

Interpretation:

Next single change:
```
