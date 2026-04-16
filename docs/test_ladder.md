# Engine B Test Ladder

This ladder is designed for rapid-fire testing on a high-RAM workstation. Start
with the cheapest probes and only move down the ladder when the signal survives.

The guiding question is:

> Can dynamic boundary-selected memory replace most of the KV cache while
> preserving the full model's next-token behavior?

The target pattern is:

- High next-token agreement with full context.
- Low retained memory fraction.
- Target evidence stays in support.
- Distractors drop out when they stop mattering.
- Confidence rises as the answer unfolds.
- Support shrinks or stabilizes as confidence rises.

## Tier 0: Sanity And Baselines

Runtime: minutes.

### 0.1 Baseline Needle

Run the full model on local, far, and adversarial cases. Do not test Engine B
unless baseline retrieval works.

```bash
python measurements/holographic_smoke_suite.py \
  --model /path/to/model \
  --only_case local,far,adversarial \
  --context_tokens 2200 \
  --max_new_tokens 6 \
  --screen_keep_ratio 0.03 \
  --support_mode mass
```

Pass:

- `baseline found=Yes` on all selected cases.

Fail interpretation:

- The prompt/model pair is not suitable. Lower context length, increase
  `max_new_tokens`, or use a stronger model before testing memory.

### 0.2 Local-Tail Negative Control

Check whether a dumb recency-only support can solve the task.

Pass:

- `local_tail` passes `local`.
- `local_tail` fails `far`.

Fail interpretation:

- If `local_tail` passes `far`, the test is too easy or the needle is too near.
- If `local_tail` fails `local`, prompt formatting or model strength is broken.

## Tier 1: Dynamic Support Signal

Runtime: minutes to tens of minutes.

### 1.1 Sentence Support, 3 Percent

This is the current easiest Engine B test. It dynamically rebuilds the support
set per generated token.

```bash
python measurements/holographic_smoke_suite.py \
  --model /path/to/model \
  --only_case far \
  --context_tokens 2200 \
  --screen_keep_ratio 0.03 \
  --support_mode mass \
  --local_tail_sentences 4 \
  --screen_lens trajectory
```

Pass:

- `support_set found=Yes`.
- `support agree` is high against full context.
- `avg_token_frac` is low.
- `target_kept` is high.

Signal:

- If `support_set` beats `local_tail`, we have evidence for dynamic support,
  not H2O-style recency retention.

### 1.2 Sweep Keep Ratios

Find the knee of the memory curve.

```bash
for r in 0.01 0.02 0.03 0.05 0.08 0.10; do
  python measurements/holographic_smoke_suite.py \
    --model /path/to/model \
    --only_case far \
    --context_tokens 2200 \
    --screen_keep_ratio "$r" \
    --support_mode mass \
    --local_tail_sentences 4
done
```

Pass:

- Agreement remains usable below 10 percent.
- A clear knee appears where support suddenly becomes sufficient.

Fail interpretation:

- If only high ratios work, the support lens is too weak or the model needs
  finer token-level support.

### 1.3 Lens Comparison

Compare the simple final-state lens against the trajectory lens.

```bash
for lens in final trajectory; do
  python measurements/holographic_smoke_suite.py \
    --model /path/to/model \
    --only_case far,adversarial \
    --context_tokens 2200 \
    --screen_keep_ratio 0.03 \
    --support_mode mass \
    --screen_lens "$lens"
done
```

Pass:

- `trajectory` should beat or match `final`, especially on adversarial cases.

Fail interpretation:

- If `final` wins, the trajectory formula is wrong.
- If both fail, sentence averaging is probably too coarse.

## Tier 2: Collapse Curve

Runtime: tens of minutes.

### 2.1 Determinism Increases During Answer

Run longer answer completions and inspect `d_conf`.

```bash
python measurements/holographic_smoke_suite.py \
  --model /path/to/model \
  --only_case far,adversarial \
  --context_tokens 2200 \
  --max_new_tokens 12 \
  --screen_keep_ratio 0.03 \
  --support_mode mass
```

Pass:

- `d_conf` is positive or non-negative on successful answers.
- `d_support` is positive or stable.

Fail interpretation:

- If confidence does not rise, the answer is not collapsing under this prompt.
- If support grows while confidence rises, the support policy is selecting
  memory by confusion rather than resolution.

### 2.2 Adversarial Recency

Use the adversarial case to verify that dynamic support is not just recent
memory with a physics name.

Pass:

- `target_kept` remains high.
- `distractor_kept` drops or is not necessary for correct answer.
- `support_set found=Yes` while `local_tail found=No`.

Fail interpretation:

- If distractors stay and targets drop, the lens is too lexical.

## Tier 3: Causal Oracle

Runtime: hours.

This is the most honest next experiment before real KV masking.

For each generated token:

1. Run full context and record next-token logits.
2. Delete one sentence or chunk.
3. Rerun and measure delta on the full-context top token logit or KL.
4. Rank chunks by causal delta.
5. Keep only chunks above threshold or top-K chunks.
6. Measure whether reduced support preserves the full logits.

Pass:

- The causal oracle keeps a tiny support set with high next-token agreement.
- The target fact has high causal delta on retrieval tokens.
- Irrelevant filler has near-zero delta.

Fail interpretation:

- If causal support is dense, Engine B is not viable for that task.
- If causal support is sparse but cosine support fails, the proxy is bad but
  the physics signal is real.

## Tier 4: Token-Span Oracle

Runtime: hours to overnight.

Move from sentence deletion to token spans. This is closer to true KV pruning.

Suggested spans:

- 8 tokens.
- 16 tokens.
- 32 tokens.
- Semantic clauses if a parser is available.

Pass:

- Token-span causal support is much smaller than sentence support.
- Similar accuracy at lower retained fraction.

Fail interpretation:

- If token spans do not improve the curve, support is likely semantic and
  coarse, not token-local.

## Tier 5: True KV Mask With Original Positions

Runtime: overnight or longer.

Only do this after Tier 3 or 4 shows sparse causal support.

Requirements:

- Preserve original token positions.
- Preserve RoPE phase.
- Mask attention logits in-place.
- Do not rewrite the prompt.
- Compare full logits vs masked logits at every generation step.

Pass:

- 90-97 percent next-token agreement with less than 10 percent retained KV.
- Needle accuracy preserved.
- Perplexity degradation stays small.

Fail interpretation:

- If oracle works but true mask fails, the implementation is wrong.
- If oracle fails, the hypothesis fails for that model/task.

## Rapid-Fire Loop

Use this loop while iterating:

1. Pick one case.
2. Run baseline and local-tail.
3. Sweep one variable only.
4. Record `support agree`, `avg_token_frac`, `target_kept`, `d_conf`,
   `d_support`.
5. Change the lens if support fails.
6. Change granularity if lens fails.
7. Move to causal oracle before writing any CUDA or invasive KV patch.

## Best Next Bets

Try these first:

1. Increase `local_tail_sentences` from 4 to 8 while keeping support at 3%.
2. Compare `trajectory` vs `final`.
3. Sweep 1-10% keep ratio on `far`.
4. Run `adversarial` only after `far` works.
5. Build the causal deletion oracle if sentence support has any signal at all.

## Local Pre-Host Simulation

Before using the Z8, run the stdlib-only invariant simulator:

```bash
python measurements/invariant_simulations.py --seeds 100
```

This checks the five cheap invariants: support contraction, phase crossing,
basis robustness, causal wedge behavior, and A+B coupling.

Then run the broader synthetic support sweep:

```bash
python measurements/synthetic_engine_b.py --seeds 100
```

This does not validate the model. It validates whether the support policy can
show the intended shape in a toy electron-cloud world:

- `target` should stay high.
- `agree` should beat local recency on far-context cases.
- `d_conf` should be positive.
- `d_sup` should be positive in `mass` mode when the support cloud contracts.

If `d_sup` is zero, the policy is not dynamically shrinking memory.
