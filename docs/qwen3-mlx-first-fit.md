# First pretrained Qwen3 MLX fit

On 2026-10-18, two bounded fits of Qwen3-1.7B completed on MLX. Reducing
attention in one decoder block preserved close teacher fidelity. Adding a
chart MLP saved substantially more weights but changed more of the teacher's
predictions, despite nearly unchanged next-token loss on the small assessment
panel. This establishes a reproducible fitting baseline, not qualification of
a compressed model for general use.

The implementation was committed as `e5d1223` before this experiment. The
[frozen protocol](../artifacts/qwen3/first_fit_protocol.json),
[corpus manifest](../artifacts/qwen3/first_fit_corpus_manifest.json), and
[full results](../artifacts/qwen3/first_fit_results.json) record revisions,
hashes, per-article metrics, parameter counts, and measured resources.

## Setup

- Dense Qwen/Qwen3-1.7B, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; 1,720,574,976 original scalars.
- Only decoder layer 13 (zero-based) of 28 was changed. Attention input/output
  ranks were 1024/512. The second candidate added four MLP charts of rank 16,
  with generator rank 17, fitted on the attention-reduced trajectory.
- Relative ridge 0.05; MLX float32 moments/linear algebra; BF16 deployment
  weights; `MLX_ENABLE_TF32=0`. Uniform calibration weights, no empirical
  Fisher weighting, and no trajectory optimization.
- WikiText-2 raw revision `b08601e04326c79dfdd32d625aee71d232d685c3`:
  32 training articles for fitting, eight different training articles for
  diagnostics, and 16 validation articles for assessment. Each contributes
  its first 128 tokens, without added special tokens. Article titles and
  complete token windows are disjoint across all three groups.
- The calibration has 4,096 tokens. Each evaluation window scores 127 observed
  next-token targets: 1,016 diagnostic targets and 2,032 assessment targets.
  The test split remains unused. Both candidates were declared before fitting;
  no ranks, regularization, or optimization were tuned against assessment.
- The exact BF16 MLX runtime supplies the teacher distribution. Reference and
  candidate phases run in separate processes. Log probabilities are float32;
  token-weighted metric accumulation uses NumPy float64.

## Results

NLL and KL use natural logarithms, in nats per scored token. Teacher agreement
compares the most likely tokens; observed-token accuracy compares predictions
with the actual next tokens in the corpus.

| Validation assessment, 2,032 targets | Exact teacher | Attention only | Attention + MLP |
| --- | ---: | ---: | ---: |
| Whole-model scalars removed | 0 | 4,194,304 (0.244%) | 37,467,000 (2.178%) |
| BF16 parameter storage saved | 0 | 8.00 MiB | 71.46 MiB |
| Teacher top-choice agreement | 100% | 96.90% | 89.22% |
| Teacher-to-candidate KL | 0 | 0.00635 | 0.06304 |
| Observed next-token NLL | 2.78752 | 2.78978 | 2.78495 |
| NLL change from teacher | 0 | +0.00226 | -0.00257 |
| Observed-token top-choice accuracy | 50.54% | 50.54% | 50.15% |

The separate diagnostic panel showed the same fidelity ordering:

| Training diagnostics, 1,016 targets | Attention only | Attention + MLP |
| --- | ---: | ---: |
| Teacher top-choice agreement | 96.26% | 89.86% |
| Teacher-to-candidate KL | 0.00704 | 0.05812 |
| NLL change from teacher | +0.00897 | -0.00881 |

Adding the MLP fit increases assessment KL by about tenfold and drops teacher
agreement by 7.68 percentage points. Its eight fewer correct next-token
predictions correspond to a 0.39 percentage-point accuracy decrease. The
slightly lower average NLL does not establish improved language modeling:
it measures probability assigned to observed tokens, while KL and agreement
measure fidelity to the entire teacher distribution. These small, contiguous
article windows do not establish generalization beyond the panel.

On calibration, attention context-output relative error was 0.197 and MLP
output relative error was 0.459. These local reconstruction errors are not
language-model quality metrics. The recorded peak MLX allocation was 3.44 GB
for the attention fit and 3.83 GB for the subsequent MLP fit (decimal GB).
Each fit command took about five seconds, including loading, captures,
solving, and saving. These are single-run observations, not throughput
benchmarks or guarantees about whole-system memory.

These initial results alone did not establish that lower teacher agreement
harmed useful performance. The baseline artifacts were kept fixed for a
broader evaluation before choosing further fitting changes.

Follow-up: the [expanded evaluation](qwen3-mlx-expanded-evaluation.md) tested
useful performance before attempting further fitting. It found a
reading/science task regression despite similar text NLL, establishing a
concrete task-retention target for further fitting. Free generation and
controlled timing remain untested.

## Reproduce

Run from the repository root with the MLX and Transformers dependencies
installed. The pinned local Hugging Face snapshot must already exist. All
weights, token rows, and teacher distributions remain under ignored
`.local-runs/`; only provenance, evaluation code, and reports are committed.

```sh
snapshot="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
run=.local-runs/qwen3-mlx-first-fit
python scripts/prepare_qwen3_fit_corpus.py --download --tokenizer "$snapshot"
# Use --offline instead of --download to reuse the verified local source.

MLX_ENABLE_TF32=0 python -m fisher_graph.mlx_qwen3_pipeline convert \
  --hf-snapshot "$snapshot" --output "$run/exact"

MLX_ENABLE_TF32=0 python -m fisher_graph.mlx_qwen3_pipeline fit \
  --artifact "$run/exact" --corpus "$run/corpus/fit.jsonl" \
  --layers 13 --components attention \
  --attention-input-rank 1024 --attention-output-rank 512 \
  --relative-ridge 0.05 --max-examples 32 --max-length 128 \
  --max-tokens 4096 --solver mlx --output "$run/attention-only"

MLX_ENABLE_TF32=0 python -m fisher_graph.mlx_qwen3_pipeline fit \
  --artifact "$run/attention-only" --corpus "$run/corpus/fit.jsonl" \
  --layers 13 --components mlp \
  --chart-count 4 --chart-rank 16 --generator-rank 17 \
  --relative-ridge 0.05 --max-examples 32 --max-length 128 \
  --max-tokens 4096 --solver mlx --output "$run/attention-and-mlp"
```

The reference directory must be new/empty. Run each phase separately to load
one model at a time:

```sh
MLX_ENABLE_TF32=0 python scripts/evaluate_qwen3_mlx_fit.py \
  --phase reference --artifact "$run/exact" \
  --corpus "$run/corpus/assessment.jsonl" \
  --calibration-corpus "$run/corpus/fit.jsonl" \
  --reference-dir "$run/reference-assessment" \
  --report "$run/reference-assessment.json" \
  --max-examples 16 --max-length 128 --max-tokens 2048

for candidate in attention-only attention-and-mlp; do
  MLX_ENABLE_TF32=0 python scripts/evaluate_qwen3_mlx_fit.py \
    --phase evaluate --artifact "$run/$candidate" \
    --corpus "$run/corpus/assessment.jsonl" \
    --calibration-corpus "$run/corpus/fit.jsonl" \
    --reference-dir "$run/reference-assessment" \
    --report "$run/$candidate-assessment.json" \
    --max-examples 16 --max-length 128 --max-tokens 2048
done
```

Repeat the evaluation commands with `diagnostic` in place of `assessment`,
`--max-examples 8`, and `--max-tokens 1024` for the diagnostic panel. Per-token
metrics stay next to each local report; their hashes and per-article summaries
are retained in the committed results. Seven CPU regression tests cover the
evaluator's scoring, masking, corpus identity, and reference validation; the
reported experiment also exercised both phases on the actual pretrained MLX
runtime.
