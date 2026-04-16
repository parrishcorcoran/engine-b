# Fault Model

Engine B should be tested from easiest to hardest. Each stage should fail for a
specific reason before we move to the next mechanism.

## What We Are Not Claiming Yet

- Prompt rewriting is not identical to in-place KV pruning.
- Sentence-level support is too coarse for the final mechanism.
- Cosine resonance is a proxy, not causal proof.
- Logical early exit proves a decision rule, not wall-clock speedup.

## Why Start Here Anyway

Direct KV monkey-patching can fail because of implementation artifacts:

- RoPE phases can shift if positions are not preserved.
- KV cache entries can become inconsistent with modified hidden states.
- Token-level masking can look bad because the mask is wrong, not because the
  boundary-memory hypothesis is wrong.

The support-set rerun avoids those artifacts. It asks the cleaner first
question: can a small dynamic memory support preserve next-token behavior at
all?

## Expected Failure Modes

1. `local_tail` passes local but fails far-context retrieval.
2. `support_set` fails far retrieval if resonance is just recency in disguise.
3. `support_set` fails adversarial retrieval if it is too lexical.
4. `support agree` falls while `avg_token_frac` is low if support selection is
   deleting real evidence.
5. `d_conf` does not rise if the answer cloud is not actually collapsing.
6. `d_support` does not rise if memory demand is not shrinking.

## Success Pattern

The first real signal is not speed. It is:

- `support agree` stays high.
- `avg_token_frac` is low.
- `target_kept` remains high.
- `d_conf` is positive.
- `d_support` is positive.

That pattern says the model's useful memory is dynamically smaller than the
full KV cache, which is the Engine B claim.
