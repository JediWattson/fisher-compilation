# Opt-in Gemma 3 270M Fisher rung

This is the first external-model scaling rung. It is designed to exercise the
generic adapter and streaming Fisher interfaces on a real text decoder
without checking the model or its cache into this repository. The committed
suite validates this plumbing with synthetic models; no gated Gemma checkpoint
or live Gemma analysis result is committed or claimed.

It is intentionally narrow:

- model: `google/gemma-3-270m`;
- text-only causal prefill;
- one selected decoder layer;
- that layer's residual input and output boundaries;
- sequences capped at 128 tokens by default;
- a rank-32 Frequent Directions sketch by default;
- no fine-tuning, weight updates, graph fitting, or compilation claim.

The checkpoint is gated on Hugging Face. Accept Google's Gemma usage terms on
the [official model page](https://huggingface.co/google/gemma-3-270m), then
authenticate using Hugging Face's normal local credential flow. The command
does not accept or print a token.

## Run it

Use a Python environment supported by your installed PyTorch and Transformers
versions:

```bash
pip install -e ".[dev,gemma]"
fisher-graph-gemma-fisher --check-paths-only
hf auth login

fisher-graph-gemma-fisher \
  --model google/gemma-3-270m \
  --prompts examples/gemma3_prompts.txt \
  --layer-index 0 \
  --max-length 128 \
  --rank 32 \
  --sketch-rows 64 \
  --output .local-runs/gemma-3-270m/layer-0-fisher.pt
```

The preflight resolves the effective Hub, assets, Xet, and token paths and
rejects the command if any is inside either the active Git checkout or the
installed package's source checkout. It exits without importing Transformers
or loading a model, so run it before authentication. The Fisher command repeats
the same validation before every model load.

Omitting both `--prompts` and `--prompt` uses a small built-in smoke set. An
explicit file or inline prompt set that contains only blank text is rejected
instead of silently using that fallback. A successful run on the smoke set
checks the live integration path only; it is not representative calibration
data and cannot support a compilation or quality claim. For a real study,
provide a frozen, representative prompt file, record its digest, and reserve
separate validation and test prompts.

Useful variations:

```bash
# Reuse a previously downloaded model without network access.
fisher-graph-gemma-fisher --local-files-only

# Put the cache at an explicit external location.
fisher-graph-gemma-fisher \
  --cache-dir /absolute/path/outside/the/repository/huggingface

# Analyze another isolated boundary pair.
fisher-graph-gemma-fisher --layer-index 6 --max-length 256
```

Without `--output`, the command derives an ignored path from both the model ID
and layer index, such as
`.local-runs/google--gemma-3-270m/layer-6-streaming-fisher.pt`, so different
layers do not overwrite one another.

`--device auto` selects CUDA first, then Apple MPS, then CPU. You can request a
specific PyTorch device such as `cpu`, `mps`, `cuda`, or `cuda:1`.
`--dtype` accepts `auto`, `float32`, `float16`, or `bfloat16`; `auto` delegates
weight dtype selection to Transformers. Regardless of model storage dtype,
the summed NLL casts FP16/BF16 logits to FP32 for cross-entropy and then
backpropagates through that cast. Fisher sketch accumulation remains CPU
FP64 by default.

The command uses eager attention for differentiable, inspectable calibration.
It freezes every model parameter, replaces the selected layer input with an
equal-valued detached leaf, and differentiates only the suffix. The source
weights are never updated. Prompt tokenization is also streamed: only one
tokenization minibatch is moved to the analysis device at a time.

## What is saved

The `.pt` output contains:

- the pooled activation mean at the selected layer input and output;
- leading Fisher sketch eigenvalues and eigenvectors;
- the exact sum of squared score-gradient norms and exact pooled Fisher trace;
- sequence and valid-position counts;
- model ID, requested revision, resolved Hub commit when available, and a
  configuration digest;
- the exact score, pooling, rank, and sketch policies.

The sibling `.json` file contains the tensor-free report. Both outputs state
that they contain no model weights. The command never calls
`save_pretrained()` and never serializes the model `state_dict`.

The default `.local-runs/` location is ignored. Derived Fisher artifacts may
be published intentionally later, but they should pass provenance and
validation review first.

## What “Fisher” means here

For sequence \(i\), the score is summed next-token negative log likelihood
\(L_i\). At valid residual position \(t\), the row is

\[
g_{i,t} = \frac{\partial L_i}{\partial a_{i,t}}.
\]

The project pools those rows into a shared-width score-gradient second moment:

\[
F_{\mathrm{pooled}}
=
\frac{1}{\sum_i T_i}
\sum_i \sum_{t=1}^{T_i}
g_{i,t} g_{i,t}^{\mathsf T}.
\]

We call this the width-pooled activation Fisher. It is useful because one
\(D\)-wide mode basis can be reused at every token position. It is not the
full sequence Fisher in \(T_iD\) coordinates: cross-position blocks are
intentionally dropped, and token positions are pooling rows rather than
independent conventional Fisher examples.

This normalization weights longer sequences more heavily because they supply
more valid rows. Summed NLL also means an early activation can receive
gradient from multiple later predictions. Artifacts therefore record
`score_reduction=sum`, `normalizer=valid_activation_positions`, and
`scope=width_pooled`; spectra from different length mixtures should not be
treated as length-neutral.

Frequent Directions stores bounded
\(O(\text{sketch rows}\times\text{hidden width})\) state rather than all
activation/gradient rows or a dense width-squared matrix. Its returned modal
eigenvalues are conservative sketch approximations. The reported total Fisher
trace is exact, while `retained_trace_fraction` is sketch energy divided by
that exact trace—not yet an exact replay measurement of energy captured by
the returned subspace.

## Repository safeguards

Model acquisition is allowed only through an external Hugging Face cache.
Before importing Transformers or downloading anything, the command resolves
both the active Git worktree and the package's source worktree, then rejects
any Hub, assets, Xet, or credential path inside either one. This includes
paths configured through `HF_HOME`, `HF_HUB_CACHE`, `HF_ASSETS_CACHE`,
`HF_XET_CACHE`, `HF_TOKEN_PATH`, and legacy cache variables.

As defense in depth, the checkout also:

- ignores common model/cache directories and weight formats;
- has a tracked-file audit for standard model roots and weight filenames;
- uses `trust_remote_code=False`;
- requests safetensors weights;
- keeps Transformers optional, so ordinary package imports and tests do not
  download or require Gemma.

The official repository currently lists the weight payload at roughly 536 MB,
plus tokenizer assets, and requires accepting the Gemma license. See the
[official file listing](https://huggingface.co/google/gemma-3-270m/tree/main)
and [Transformers Gemma 3 documentation](https://huggingface.co/docs/transformers/model_doc/gemma3).

## Next gate

A successful opt-in run establishes real-model activation access and
bounded-memory mode extraction for its recorded model revision and prompt
set. The committed code and synthetic tests alone do not establish that, and
neither they nor the one-layer Fisher artifact prove that a graph executor can
replace the layer. The next scientific gate is:

1. freeze representative calibration, validation, and test prompt splits;
2. rerun the sketch and measure modal stability against rank and prompt mix;
3. replay validation prompts to measure the exact Rayleigh energy captured by
   the sketched subspace;
4. collect detached input/output boundary pairs for the same layer;
5. fit one variable-length causal modal executor;
6. require local boundary, end-to-end NLL, sequence-length, and fallback gates
   before replacing that layer in the mixed runtime.
