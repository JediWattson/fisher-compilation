# Expanded evaluation of the frozen Qwen3 fits

This experiment measures useful performance of the original Qwen3-1.7B and
the two [existing single-block fits](qwen3-mlx-first-fit.md). All 28 decoder
layers execute. Only layer 13 has fitted replacements, and no weights were
changed for this evaluation. Observed next-token loss and task accuracy are
the primary measurements; matching every original-model choice is secondary.

The joint attention-plus-MLP fit preserved similar next-token loss on 65,456
text targets but reduced accuracy on the 96-question task panel from 68 to
49 correct. A second forward path reproduced the main regression. This gives
a concrete reason to improve task retention before expanding the MLP fit
across more layers; lower teacher agreement alone was not sufficient evidence.

## Results

The [text summary](../artifacts/qwen3/expanded_text_summary.json) retains
per-article scores and resampling intervals. The
[complete result record](../artifacts/qwen3/expanded_evaluation_results.json)
includes task predictions, paired changes, artifact receipts, resources, and
the secondary full-prompt execution check. Raw passages remain local.

| Text context | Exact NLL | Attention NLL change | Attention + MLP NLL change |
| --- | ---: | ---: | ---: |
| 512 tokens | 3.06741 | +0.00359 | -0.00777 |
| 2,048 tokens | 2.88030 | +0.00153 | -0.00151 |

Lower NLL is better. The joint changes correspond to perplexity changes of
-0.77% and -0.15%. Its paired article-bootstrap 95% intervals include zero:
[-0.01610, +0.00059] at 512 tokens and [-0.01595, +0.01252] at 2,048. These
results do not establish an improvement or formal equivalence. Attention-only
intervals were [+0.00027, +0.00698] and [-0.00251, +0.00476].

Joint teacher top-choice agreement was 87.21% and 87.15%, compared with
95.00% and 94.71% for attention-only. Joint observed-token accuracy decreased
from 45.41% to 45.00% at 512 and from 46.64% to 46.09% at 2,048. The much
larger effect appeared in the task panel:

| Correct constrained answers | Exact | Attention only | Attention + MLP |
| --- | ---: | ---: | ---: |
| BoolQ reading | 24/32 | 23/32 | 10/32 |
| PIQA commonsense | 18/32 | 17/32 | 18/32 |
| ARC-Easy science | 26/32 | 26/32 | 21/32 |
| Total | **68/96** | **66/96** | **49/96** |

The joint fit turned 24 originally correct answers into wrong answers and
five originally wrong answers into correct ones, a net loss of 19. Attention
alone had three regressions and one gain. The same questions, prompts, choice
order, scoring, and underlying pretrained checkpoint were used for every model.

BoolQ's selected panel is imbalanced: 27 Yes and five No answers. Always
choosing Yes would score 27/32, so the absolute scores must not be read as a
representative BoolQ benchmark. The joint fit shifts toward No, choosing it
27 times versus 11 for the original model. Yes recall falls from 20/27 to
5/27; No recall changes from 4/5 to 5/5. Even descriptive balanced accuracy
falls from 77.0% to 59.3%. This is an observed answer-decision regression on
these examples, not proof that every aspect of reading ability deteriorated.

The unrestricted first-token choice was outside the offered letters for
0 exact, 2 attention-only, and 14 joint examples; each such token decoded to
`The`. Constrained scoring does not measure the resulting free continuation.
All joint BoolQ examples retained more than 99.8% probability on the offered
letters, so restricting the vocabulary does not explain the BoolQ decline.

## Secondary execution check

After observing the task loss, the same 96 frozen prompts were checked using
a complete prompt forward pass instead of a cached prefix followed by its
last token. No weights, prompts, or selection rules were changed.

| Correct answers | Original cached-prefix path | Full-prompt path |
| --- | ---: | ---: |
| Exact | 68/96 | 68/96 |
| Attention only | 66/96 | 67/96 |
| Attention + MLP | 49/96 | 50/96 |

The BF16 execution paths are not numerically identical: 1, 3, and 1 answer
choices changed, respectively, with maximum offered-letter logit differences
of 2.75, 1.50, and 1.875. This check preserves the initial scores rather than
substituting the more favorable path. BoolQ remains 24/32, 23/32, and 10/32
in both paths, and the joint science score remains 21/32. The main regression
therefore survives the execution check.

The next experiment should test task retention during MLP fitting, using new
training/development data and keeping these assessment examples out of fitting
and hyperparameter selection. A broader calibration mixture and a downstream
objective are hypotheses to test, not remedies established by these results.
The observed peak MLX allocation across the expanded runs was about 4.92 GB.
The existing 71.46 MiB joint parameter saving is unchanged.

## Evaluation design

The [protocol](../artifacts/qwen3/expanded_evaluation_protocol.json) fixes the
original and fitted artifact hashes, source revisions, precision, selection,
metrics, and limitations. The supplementary receipt note records when package
and script metadata were finalized relative to reference execution.

The [text manifest](../artifacts/qwen3/expanded_corpus_manifest.json) records
80 fresh WikiText-2 articles, all distinct from each other and the 56 articles
used in the first experiment. Each contributes a deterministic midpoint
window, beginning after the article's first token:

| Panel | Articles | Context tokens | Scored next-token targets | Source |
| --- | ---: | ---: | ---: | --- |
| Medium | 64 | 512 | 32,704 | 28 validation + 36 test articles |
| Longer | 16 | 2,048 | 32,752 | Validation articles |

WikiText-2 has only 44 fresh validation articles after excluding the first
experiment. The additional test articles were therefore consumed for this
frozen assessment, with no fitting or selection based on their scores.
Preparation also rejects exact containment of any previous 128-token window
inside a new window. The panels use different articles, so their differences
do not isolate the effect of context length.

The [task manifest](../artifacts/qwen3/expanded_tasks_manifest.json) freezes
the first 32 validation examples from each of BoolQ (reading), PIQA (physical
commonsense), and ARC-Easy (science). Exact repository revisions and file
hashes are retained. Qwen's chat template disables thinking and asks for one
answer letter. The score is the highest logit among offered letters; every
letter is verified to be a single, context-stable continuation token. Prompts
range from 47 to 321 tokens and are not truncated. This is a custom 96-question
panel, not a standard benchmark score or a free-generation test. Whether the
original model saw any source data during pretraining is unknown.

## Measurement and verification

Models run separately, retaining their BF16 deployment weights, with
`MLX_ENABLE_TF32=0`. Text evaluation projects the final hidden states in chunks
of at most 128 positions, storing only original-model target NLL and top
choices. This avoids retaining a full vocabulary distribution for every token;
expanded text KL is therefore unavailable, not zero. The original dense KL
measurements remain in the first-fit report.

On one previous diagnostic window, compact and dense scoring matched all 127
top choices with zero target-NLL difference for both the exact and joint-fit
models, using chunk sizes 128 and 31. The 36 CPU regression tests cover data
selection, scoring, masks, reference identity, compact projection, task labels,
and paired task outcomes. Both expanded evaluation paths then run on the
actual pretrained MLX models.

Text NLL uses natural logarithms and token-weighted averages. Paired bootstrap
intervals resample entire articles, retaining all dependent tokens together:
5,000 resamples with seed 20261018. These describe stability within a fixed
panel; article selection was not a random sample of every possible workload.
The summary verifies artifact and corpus identities plus per-article counts
and NLL deltas against per-token records before resampling.

## Reproduce

Use the pinned tokenizer and original/fitted artifacts from the first-fit
report. All large weights and raw data remain in ignored `.local-runs/`.
The preparation steps require a local tokenizer and a Parquet reader; task
preparation uses DuckDB. Evaluation requires MLX, without DuckDB or a tokenizer.

```sh
snapshot="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
run=.local-runs/qwen3-mlx-expanded-eval
base=.local-runs/qwen3-mlx-first-fit

hf download Salesforce/wikitext wikitext-2-raw-v1/test-00000-of-00001.parquet \
  --type dataset --revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --local-dir "$run/source"
python scripts/prepare_qwen3_expanded_corpus.py --tokenizer "$snapshot"
python scripts/evaluate_qwen3_mlx_tasks.py prepare --download --tokenizer "$snapshot"

for variant in exact attention-only attention-and-mlp; do
  for length in 512 2048; do
    count=64
    if [ "$length" = 2048 ]; then count=16; fi
    phase=evaluate
    if [ "$variant" = exact ]; then phase=reference; fi
    MLX_ENABLE_TF32=0 python scripts/evaluate_qwen3_mlx_fit.py \
      --phase "$phase" --artifact "$base/$variant" \
      --corpus "$run/corpus/assessment_$length.jsonl" \
      --calibration-corpus "$base/corpus/fit.jsonl" \
      --reference-dir "$run/reference-$length" --reference-mode compact \
      --head-chunk-size 128 --report "$run/$variant-$length.json" \
      --max-examples "$count" --max-length "$length" --max-tokens 32768
  done
done

MLX_ENABLE_TF32=0 python scripts/evaluate_qwen3_mlx_tasks.py evaluate \
  --artifact "$base/exact" --output "$run/exact-tasks.json"
for variant in attention-only attention-and-mlp; do
  MLX_ENABLE_TF32=0 python scripts/evaluate_qwen3_mlx_tasks.py evaluate \
    --artifact "$base/$variant" --reference "$run/exact-tasks.json" \
    --output "$run/$variant-tasks.json"
done
python scripts/summarize_qwen3_expanded_text.py

for variant in exact attention-only attention-and-mlp; do
  MLX_ENABLE_TF32=0 python scripts/verify_qwen3_task_prefill.py \
    --report "$run/$variant-tasks.json" \
    --output "$run/$variant-task-prefill-verified.json"
done
```

Reference directories and task report paths must be new. Source files can be
reused offline; omit task `--download` and add corpus `--offline` when the
required verified local files and API pages are present. Timing and peak MLX
allocation are observations from these runs, not throughput benchmarks or
guarantees about whole-system memory.
