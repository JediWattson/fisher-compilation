# Qwen3 on MLX: local conversion, fitting, and generation

`fisher-graph-qwen3-mlx` runs dense text-only Qwen3 from existing local files on
Apple Silicon. It converts a Hugging Face safetensors snapshot directly into an
MLX artifact, captures selected layer boundaries, fits reduced attention and
chart MLPs, and generates through the saved runtime with a KV cache. It does not
download weights or tokenizer files. Conversion does not instantiate a second
PyTorch model.

The Python environment needs the project's `mlx` and `qwen` extras. Once the
package is installed, the examples below use the console entrypoint. The
equivalent invocation is `python -m fisher_graph.mlx_qwen3_pipeline`.

## Precision and memory

Launch library scripts with `MLX_ENABLE_TF32=0` **before importing MLX**. On
supported Apple hardware, MLX can otherwise use reduced precision for float32
matrix multiplication. See the [MLX precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html).
The CLI sets this variable to `0` if it is absent; it preserves an explicit
value and records it in reports. Setting the variable after MLX has initialized
does not establish the same precision guarantee.

The default `--solver mlx` uses MLX float32 capture/statistics and MLX CPU linear
algebra for factorization and solves. It is not an all-GPU or float64 fitting
path. `--solver numpy_float64` explicitly selects host NumPy float64
factorizations; PCA still factors moments accumulated in MLX float32, while
anchored ridge accumulates and solves in NumPy float64. Fitted coefficients are
cast to the runtime's dtype. The per-fit reports state these boundaries.
Adam keeps float32 master coefficients, but evaluates and exports updates in
each parameter's original dtype. BF16 fitting therefore preserves BF16 hidden
states and KV caches rather than silently doubling their storage.

For a 16 GB machine, start with one layer, checkpoint precision, and the default
calibration limits: 16 sequences, 128 tokens per sequence, and 2,048 physical
tokens total. Batches in a JSONL row are split into single-sequence forwards.
Only the requested boundary captures are retained, and logits are omitted
during capture. These bounds limit activation storage; they are not a measured
peak-memory guarantee for every rank, model, or optimization setting.

Optional trajectory optimization keeps the original runtime as a teacher.
Unchanged parameter arrays are shared, while replaced source arrays remain
needed by the teacher. `terminal_mse` avoids the full-vocabulary teacher targets
required by `teacher_kl`. Leave trajectory optimization disabled for an initial
memory check. Conversion, fitting, and generation are separate processes in
the command examples.

## Convert an existing snapshot

Replace the local directories in these commands with your existing files:

```sh
MLX_ENABLE_TF32=0 fisher-graph-qwen3-mlx convert \
  --hf-snapshot /absolute/path/to/Qwen3-1.7B/snapshot \
  --tokenizer /absolute/path/to/Qwen3-1.7B/snapshot \
  --dtype preserve \
  --output artifacts/qwen3-mlx-exact
```

The loader supports a local safetensors file or sharded safetensors index. The
current dense Qwen3 path requires SiLU, full attention, and default or linear
RoPE scaling; unsupported architecture variants are rejected. `--dtype` may
also be `bfloat16`, `float16`, or `float32`. Preserving the checkpoint's dtype is
the memory-conscious default. Conversion itself does not measure parity.

Each conversion or fit writes:

- `weights.safetensors`: the runtime arrays;
- `mlx_qwen3.json`: layer semantics and source/fit provenance;
- `pipeline_report.json`: operation, precision settings, parameter counts,
  local source identity, and fit results when applicable.

Tokenizer files are not copied into the artifact. Conversion can record their
local path, but generation always takes an explicit `--tokenizer`. Output must
be empty unless `--overwrite` is supplied. Fit output must differ from its
input artifact, preserving a usable source artifact.

## Prepare local calibration rows

The corpus is JSONL: one JSON value per line. Text requires an explicit local
tokenizer directory:

```jsonl
{"text": "A triangle has three sides."}
"The capital of France is Paris."
```

Already tokenized examples do not require a tokenizer. Token IDs must belong
to the source model's vocabulary. Optional masks and positions must align with
the IDs:

```jsonl
{"input_ids": [101, 102, 103, 104]}
{"input_ids": [[0, 105, 106, 107], [108, 109, 110, 111]], "attention_mask": [[0, 1, 1, 1], [1, 1, 1, 1]], "position_ids": [[0, 0, 1, 2], [0, 1, 2, 3]]}
```

The sample numbers illustrate the format, not a meaningful calibration set.
Supply representative sequences and retain a separate evaluation corpus.
Rows are truncated to the requested limits, padding is excluded from fitting
captures, and sequences need at least two valid tokens. The token budget counts
physical tokens, including padding. If positions are omitted, the MLX runtime
counts valid tokens; provide explicit positions when comparing against a
teacher with a different padding convention. The corpus's SHA-256 is recorded;
its text and token rows are not included in fit reports.

## Fit selected layers

```sh
MLX_ENABLE_TF32=0 fisher-graph-qwen3-mlx fit \
  --artifact artifacts/qwen3-mlx-exact \
  --corpus /absolute/path/to/calibration.jsonl \
  --tokenizer /absolute/path/to/Qwen3-1.7B/snapshot \
  --layers 0 \
  --components both \
  --attention-input-rank 256 \
  --attention-output-rank 64 \
  --chart-count 4 \
  --chart-rank 16 \
  --max-examples 16 \
  --max-length 128 \
  --max-tokens 2048 \
  --solver mlx \
  --output artifacts/qwen3-mlx-layer0
```

Layers are zero-based. `--layers 0,2,4-7` selects a set in increasing order;
`--layers all` explicitly selects every layer. `--components attention` or
`--components mlp` fits one component type. Set ranks appropriate to the model;
the fitter rejects ranks larger than the corresponding width. The default
attention ranks target Qwen3-1.7B, not the tiny models used in unit tests.

For each selected layer, the pipeline captures and reduces its attention,
then captures and fits its MLP using the current partially fitted runtime.
Later layers therefore see the changed upstream trajectory. Attention fitting
currently requires an exact source attention block; select unfitted attention
layers when continuing from an artifact that already contains reductions.
Unselected layers remain in their existing form.

Reduced attention preserves Qwen's per-head Q/K normalization, RoPE, grouped
KV semantics, and identity operator-output normalization. Chart fitting uses
an affine term plus local chart residuals in the MLP output space.
`--relative-ridge` controls regression regularization. These CLI fits use
uniform calibration row weights; they do not automatically estimate Fisher
weights. The API accepts explicit row weights and provides separate empirical
Fisher routines.

To include a short terminal-state correction, add these options to the same
fit invocation:

```sh
--trajectory-steps 20 --trajectory-objective terminal_mse --learning-rate 0.001
```

This optimizes selected newly fitted coefficient arrays against the original
input artifact on the calibration batches. Chart geometry and input bases
remain fixed. `--anchor-weight` adds a coefficient-distance penalty; without
an explicitly supplied Fisher diagonal in the Python API, it is an isotropic
anchor, not a Fisher-weighted EWC prior. The CLI also offers `teacher_kl`, which
has higher vocabulary-related memory use. Optimization reports are calibration
measurements, not held-out qualification.

## Generate from the artifact

```sh
MLX_ENABLE_TF32=0 fisher-graph-qwen3-mlx generate \
  --artifact artifacts/qwen3-mlx-layer0 \
  --tokenizer /absolute/path/to/Qwen3-1.7B/snapshot \
  --prompt "A triangle has" \
  --max-new-tokens 32 \
  --report artifacts/qwen3-mlx-layer0/completion.json
```

This prints a greedy continuation, using one prefill followed by cached decode.
The default prompt limit is 512 tokens. Generation stops at EOS unless
`--ignore-eos` is supplied. Add `--chat` to apply the local tokenizer's chat
template with thinking disabled; otherwise the prompt is treated as raw
completion text. The optional report includes generated tokens and completion
text. Its one-run elapsed time is not a throughput benchmark.

To inspect per-token chart routing and counterfactual ablation pushes for any
saved MLX artifact:

```sh
MLX_ENABLE_TF32=0 fisher-graph-qwen3-mlx xray \
  --artifact artifacts/qwen3-mlx-layer0 \
  --tokenizer /absolute/path/to/Qwen3-1.7B/snapshot \
  --prompt "The capital of France is"
```

To launch an interactive freeform prompt REPL with live layer/chart muting
(`/mute attn <L>`, `/mute mlp <L>`, `/mute chart <L> <C>`), `/xray` toggling,
`/preset`, and optional side-by-side exact comparison:

```sh
MLX_ENABLE_TF32=0 fisher-graph-qwen3-mlx interactive \
  --artifact .local-runs/qwen3-ewc-refine/joint_waterfill_286mib \
  --tokenizer /absolute/path/to/Qwen3-1.7B/snapshot
```

Any exact, partial-fit, or full-fit `MLXQwen3` runtime can also be bridged
directly into the standalone 59-event PyTorch executor (`runtime.to_torch()`)
without allocating a Hugging Face model, supporting `JointGraphDecodeSession`
and round-trip conversion via `MLXQwen3.from_torch(executor)`.

## Python API

Start the process with `MLX_ENABLE_TF32=0`, then:

```python
import mlx.core as mx
from fisher_graph.mlx_qwen3 import MLXQwen3
from fisher_graph.mlx_qwen3_fit import (
    capture_runtime_rows, fit_reduced_attention, fit_chart_mlp,
)

runtime = MLXQwen3.load("artifacts/qwen3-mlx-exact")
batches = [{"input_ids": mx.array([[101, 102, 103, 104]])}]
rows = capture_runtime_rows(runtime, batches, [
    "attention.0.normalized", "attention.0.context",
])
result = fit_reduced_attention(runtime, 0, rows, input_rank=256, output_rank=64)
runtime = result.runtime
rows = capture_runtime_rows(runtime, batches, ["mlp.0.normalized", "mlp.0.output"])
result = fit_chart_mlp(runtime, 0, rows, chart_count=4, chart_rank=16)
runtime = result.runtime
runtime.save("artifacts/qwen3-mlx-api-fit")
output = runtime.prefill(batches[0]["input_ids"])
next_token = mx.argmax(output.logits[:, -1, :], axis=-1)[:, None]
output = runtime.decode(next_token, output.cache)
```

Fit functions return `MLXFitResult(runtime, updates, report)` and preserve the
input runtime. `capture_empirical_token_fisher` estimates activation score
gradient second moments at named capture sites; `capture_parameter_fisher`
estimates parameter diagonals. Both use bounded next-token objectives and
report the estimator's scope. A parameter diagonal can be passed as
`fisher_diagonal` to `optimize_trajectory`; the selected parameter names must
match that diagonal's shapes. This is empirical Fisher from supplied targets,
not an expectation over all model-generated labels.

## What has been validated

Tiny real Hugging Face Qwen3 tests cover native projection gradients, live
normalization fingerprints, full-rank reduced attention, MLX forward/cache
parity, masked captures, local fitting, empirical Fisher, optimization, artifact
round trips, and the conversion/fit/generation CLI. These tests do not establish
compression quality for a pretrained 1.7B model.

The focused validation run passed 118 tests: 72 Qwen/reference regressions and
46 tests on actual Metal covering runtime parity, fitting, and the CLI.

The local pretrained smoke report at
[`artifacts/qwen3/mlx_validation.json`](../artifacts/qwen3/mlx_validation.json)
compares Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` in float32
against Hugging Face PyTorch eager execution. With MLX 0.32.0 and
`MLX_ENABLE_TF32=0`, two short prompts plus four teacher-forced cached steps
matched all 10 next-token decisions. Maximum valid-token logit error was
`2.36e-4` at prefill and at most `1.04e-4` during those decode steps. This is a
bounded numerical smoke test of the exact runtime. It is not a broad quality
evaluation, fitted-model qualification, or speedup claim.

The script runs the teacher and MLX stages in separate processes to avoid
simultaneously holding both full models:

```sh
python scripts/validate_qwen3_mlx.py teacher \
  --snapshot /absolute/path/to/Qwen3-1.7B/snapshot \
  --reference /tmp/qwen3-teacher.npz
MLX_ENABLE_TF32=0 python scripts/validate_qwen3_mlx.py mlx \
  --snapshot /absolute/path/to/Qwen3-1.7B/snapshot \
  --reference /tmp/qwen3-teacher.npz \
  --report /tmp/qwen3-mlx-validation.json
```

The [first pretrained single-block fit](qwen3-mlx-first-fit.md) now records
attention-only and attention-plus-MLP fits on 4,096 calibration tokens, with
separate diagnostic and validation panels. It includes teacher fidelity,
observed next-token loss, parameter savings, and measured MLX allocation.
This remains a bounded first experiment; broad task quality, end-to-end speed
gains, and peak memory guarantees have not been established.

The [expanded evaluation](qwen3-mlx-expanded-evaluation.md) scores 65,456
fresh text targets at 512/2,048-token contexts and 96 constrained task
questions. Similar text NLL concealed a joint-fit task regression, reproduced
with a second execution path. These results distinguish useful task retention
from agreement with the original model's token choices.

The [layer-13 EWC retention and causal localization study](qwen3-mlx-ewc-retention.md)
traces that regression to chart-router flattening at width 2048, unanchored
ridge shrinkage, and five dormant chat-boundary SwiGLU channels (`4172, 4971,
751, 4206, 287`). Anchoring `fit_chart_mlp` to the pretrained SwiGLU
Euler-Jacobian prior (`0.5 * E[J(x)]`) with centered intercepts and
inverse-norm weights raises task retention from 49/96 to 57/96 on the same
32 raw WikiText passages and to 63/96 when those training passages are
additionally wrapped in the Qwen3 chat template (`88.41%` 512-token agreement).
