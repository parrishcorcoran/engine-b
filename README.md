# Engine B

Dynamic boundary-memory experiments for replacing wasteful KV cache retention.

## Core Claim

Autoregressive inference keeps too much memory. As a sentence writes itself,
the next-token distribution should become more deterministic, so the model
should need a smaller and sharper support set, not an ever-growing KV cache.

Engine B tests whether we can dynamically keep only the memory that still
matters while preserving the full model's next-token behavior.

This is not H2O. H2O keeps historical heavy hitters. Engine B asks a different
question: at each token, what support is still causally relevant to the
collapsing answer cloud?

## Physics Lens

- **Electron cloud:** possible continuations begin as a broad probability cloud.
  As local syntax and semantics resolve, the cloud collapses.
- **Holographic boundary:** the useful memory may live on a thin dynamic
  support surface, not in the full context bulk.
- **Trajectory:** support should be selected from the current direction of
  collapse, not from static token importance.

## Current Smoke Test

The first harness is intentionally slow but falsifiable:

```bash
python measurements/holographic_smoke_suite.py \
  --model /path/to/llama-or-qwen-model \
  --context_tokens 2200 \
  --screen_keep_ratio 0.03 \
  --local_tail_sentences 4
```

It runs:

1. A full-context baseline.
2. A local-tail control that should fail distant retrieval.
3. A dynamic support-set rerun that rebuilds the prompt per token from only
   resonant context.
4. A logical early-exit path for Engine A.
5. A dual Engine A + Engine B run.

## Key Metrics

- `support agree`: whether the reduced support predicts the same next token as
  the full prompt.
- `avg_token_frac`: how much context support remains.
- `target_kept`: whether the true evidence stayed in the support set.
- `d_conf`: whether the next-token cloud became more deterministic.
- `d_support`: whether the support cloud contracted.

The target behavior is high support agreement with a small retained fraction,
especially as `d_conf` rises and `d_support` stays positive.

## Roadmap

1. Sentence-level dynamic support oracle.
2. Token-span dynamic support oracle.
3. Causal ablation oracle using next-token logit deltas.
4. Learned support selector that imitates the oracle.
5. True in-place KV masking with original positions and RoPE phases preserved.

The current harness is step 1: prove or falsify the signal before optimizing
the mechanism.
