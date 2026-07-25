# Opt-in Gemma 3 270M Fisher, trajectory, and sufficiency rungs

These are the first external-model scaling rungs. They exercise the generic
adapter, streaming Fisher, split-stability, exact held-out Rayleigh replay,
multi-boundary modal transport, and joint full-width modal-sufficiency
interfaces on a real text decoder without checking the model or its cache into
this repository. The committed suite validates this plumbing with synthetic
models; no checkpoint or live result artifact is committed, and no rank,
quality, or compilation result is accepted or claimed.

It is intentionally narrow:

- model: `google/gemma-3-270m`;
- text-only causal prefill;
- either one selected decoder layer or one contiguous diagnostic block;
- canonical residual boundaries without duplicate output/input aliases;
- sequences capped at 128 tokens by default;
- a bounded Frequent Directions sketch and nested rank sweep;
- principal-angle comparison across calibration halves;
- exact frozen-basis Fisher-energy replay on validation;
- calibration-fit, validation-frozen activation transports plus row-local and
  exact-logical-lag reverse-gradient predictors for adjacent boundaries and
  the whole block endpoint;
- a full-width calibration Fisher basis at each selected layer output and a
  joint keep-top-\(k\) sufficiency curve on validation;
- a reserved test split that none of these analysis commands model-evaluate;
- no fine-tuning, weight updates, graph fitting, or compilation claim.

The checkpoint is gated on Hugging Face. Accept Google's Gemma usage terms on
the [official model page](https://huggingface.co/google/gemma-3-270m), then
authenticate using Hugging Face's normal local credential flow. The command
does not accept or print a token.

## Run the integration smoke test

Use a Python environment supported by your installed PyTorch and Transformers
versions:

```bash
pip install -e ".[dev,gemma]"
fisher-graph-gemma-fisher --check-paths-only
hf auth login

fisher-graph-gemma-fisher \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
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

## Run split stability and exact validation replay

Once the smoke path works, run the second analysis rung with a pinned model
revision:

```bash
fisher-graph-gemma-stability --check-paths-only

fisher-graph-gemma-stability \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --layer-index 0 \
  --max-length 128 \
  --ranks 8 16 24 32 48 64 96 128 \
  --sketch-rows 256 \
  --device cpu \
  --dtype float32
```

The example split contains 16 calibration-A, 16 calibration-B, 16 validation,
and 16 test prompts. It is frozen and category-matched so the mechanics are
reproducible, but its declared status is
`diagnostic_only_not_representative`: the prompts are short, their token
lengths are not rigorously stratified, and their topical templates are
strongly matched. The loader rejects exact-string overlap, but it cannot prove
semantic independence. A result on this file can expose rank or prompt
sensitivity; it cannot establish that a Gemma layer is ready to compile.

The command performs three calibration collections: A, B, and their combined
A+B set. At each requested prefix rank \(k\), it reports the normalized
principal-angle overlap

\[
S_k(U,V)=\frac{\lVert U_k^{\mathsf T}V_k\rVert_F^2}{k}.
\]

This score is one for identical subspaces and is invariant to eigenvector sign
flips and rotations within a tied eigenspace. It is therefore a better
stability measure than correlating individual eigenvector coordinates.

It then makes one transient pass over validation gradients and measures each
frozen basis with

\[
R_k =
\frac{\sum_g \lVert U_k^{\mathsf T}g\rVert^2}
     {\sum_g \lVert g\rVert^2}.
\]

That ratio is the exact width-pooled validation Fisher energy captured by the
first \(k\) modes. “Exact” describes this streamed fixed-basis calculation,
not the approximate basis, population representativeness, model quality, or
compilability. The reserved test prompts are parsed, exact-duplicate checked,
and hashed, but never tokenized, sent through Gemma, scored, or used to select
a rank.

The CLI produces a diagnostic rank curve; it does not define acceptance
thresholds or approve a rank. Its report explicitly records
`acceptance_thresholds_defined=false`, `quality_validation_claim=false`,
`compilation_claim=false`, and `test_split_evaluated=false`.

Without `--output`, the stability command writes to an ignored model/layer
path such as
`.local-runs/google--gemma-3-270m/layer-0-fisher-stability.pt`, with a
tensor-free JSON report beside it.

## Run the multi-layer modal-trajectory diagnostic

The third rung asks the question that an isolated input/output experiment
cannot answer: can one important computation persist through several native
layers while its local coordinate basis changes?

```bash
fisher-graph-gemma-trajectory --check-paths-only

fisher-graph-gemma-trajectory \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --start-layer 4 \
  --end-layer 6 \
  --max-length 128 \
  --ranks 8 16 24 32 48 64 96 128 \
  --sketch-rows 256 \
  --causal-lags 0 1 4 \
  --causal-relative-ridge 0.01 \
  --device cpu \
  --dtype float32
```

Layers 4–6 deliberately cross sliding → global → sliding attention. The
adapter plans these four unique boundaries:

```text
layer.4.input
layer.4.output  = layer.5.input in the baseline forward
layer.5.output  = layer.6.input in the baseline forward
layer.6.output
```

Only `layer.4.input` is detached. One complete forward and one backward per
sequence therefore produce all four activation/score-gradient row pairs while
discarding the autograd prefix before the block. Later input aliases are not
collected again.

For adjacent boundaries \(i\) and \(j\), the report keeps direct principal
angles separate from paired transport. Its exact cross-Rayleigh measurement is

\[
R_{i\rightarrow j,k}
=
\frac{\sum_g \lVert U_{i,k}^{\mathsf T}g_j\rVert^2}
     {\sum_g \lVert g_j\rVert^2},
\]

and its transfer ratio divides this by the target basis's own held-out
capture. A ratio near one means the source subspace captures about as much
target sensitivity as the target's own basis. Low transfer is drift, not
automatically rotation.

To test rotation, the command replays calibration A+B through the frozen
Fisher bases and stores only modal sums, two \(k\times k\) Gram matrices, and
one \(k\times k\) cross-moment per edge. It fits a whitened orthogonal
Procrustes map and evaluates that frozen map on validation. Two row kinds are
kept distinct:

- centered activation transport follows the forward boundary order;
- uncentered score-gradient transport follows the natural reverse direction,
  downstream boundary → upstream boundary.

The validation report also compares the learned map to an identity-coordinate
baseline. That baseline, and the reported gain over it, depend on the chosen
signs and ordered coordinates of the two Fisher bases; they are diagnostics,
not gauge-invariant evidence. The frozen-map prediction error itself remains
well defined when bases and maps transform together. For centered activations
it reports the calibration-mean-baseline \(R^2\)-style explained fraction.
For uncentered score gradients, the baseline is identically zero, so the
corresponding quantity is
**zero-baseline explained energy**

\[
E_0
=
1 -
\frac{\sum_t \lVert z^{(g)}_{\mathrm{target},t}
      -\widehat z^{(g)}_{\mathrm{target},t}\rVert^2}
     {\sum_t \lVert z^{(g)}_{\mathrm{target},t}\rVert^2},
\]

where \(z^{(g)}\) denotes the projected score-gradient coordinates. This is
not conventional mean-centered \(R^2\). A negative value means the frozen map
has more squared error than predicting zero projected modal gradient. The
report also includes normalized RMSE, cosine, centered activation canonical
correlations, uncentered through-origin gradient canonical correlations,
direct depth overlap, cross-Rayleigh transfer, and
eigenvalue-weighted Fisher similarity. High activation \(R^2\) by itself is
not a Fisher-trajectory result: residual connections can preserve activations
even when task sensitivity changes.

The reverse-gradient diagnostic pairs \(g_{\mathrm{downstream},t}\) with
\(g_{\mathrm{upstream},t}\) at the same position. A causal transformer has
off-diagonal Jacobian blocks as well: an upstream position \(s\) can receive
reverse-mode contributions from downstream positions \(t>s\). Those
cross-position terms are not inputs to this row-local map. Low
zero-baseline explained energy therefore rejects only a fixed same-position
reverse-gradient map; it does not reject a sequence-aware causal modal
executor that mixes the visible positions.

The trajectory command now adds a bounded sequence-aware diagnostic for that
specific omission. In frozen Fisher coordinates it fits

\[
\widehat z^{(g)}_{\mathrm{up},s}
=
\sum_{\delta=0}^{L}
z^{(g)}_{\mathrm{down},s+\delta} W_\delta .
\]

This is the reverse-mode triangular direction: forward causality lets an
earlier activation affect later outputs, so the gradient at the earlier
boundary can depend on gradient rows at the same and later downstream
positions. It is not an anti-causal forward executor.

`--causal-lags 0 1 4` requests three nested maximum-lag fits: \(L=0\),
\(L=1\), and \(L=4\). Every fit uses the same configured
`--causal-relative-ridge` scale. The \(L=0\) fit is an independently solved
row-local ridge model, so `gain_over_lag_zero_explained_fraction` compares a
larger ridge model to a fair nested ridge baseline rather than to the
orthogonal Procrustes map. A maximum-lag \(L\) model includes every exact lag
\(0,\ldots,L\), not only the numbers listed on the command line.

The matching is by original logical position, not by compacted row index.
For example, valid positions `[0, 2, 3]` provide exact lag-1 pairs only between
2 and 3; rows 0 and 2 are not treated as neighbors merely because they are
adjacent after masking. Padding, missing positions, and variable sequence
lengths therefore cannot silently create false lag pairs. Each artifact
records the per-lag pair counts and a digest of the sequence position
schedules.

The operator is also clipped to the segment's finite structural visibility.
For a segment made only of sliding-attention layers, the composed visibility
is

\[
1+\sum_{\ell}(\mathrm{window}_{\ell}-1).
\]

Lags at or beyond that visibility are excluded. If any layer in the segment
has global visibility, the structural window is unbounded. The trajectory
protocol applies this calculation to every adjacent boundary pair and also to
the first-boundary → final-boundary endpoint, so it measures both local
cross-token predictability and the aggregate block relationship.

Rows remain transient. For maximum rank \(k\) and maximum lag \(L\), the
estimator retains a feature Gram of shape
`[(L + 1)k, (L + 1)k]`, a feature/target cross-moment of shape
`[(L + 1)k, k]`, and a target Gram of shape `[k, k]`, all accumulated on CPU
in FP64. Storage is independent of calibration sequence count and sequence
length. Rank prefixes and shorter lag windows are sliced from those shared
sufficient statistics, frozen on calibration A+B, and evaluated from a
separate validation replay.

This operator is deliberately a predictive gradient diagnostic. The
\(W_\delta\) values are joint regression coefficients in pooled Fisher
coordinates; correlations among lagged rows, discarded modes, nonlinear
context dependence, and ridge regularization prevent them from identifying
the transformer's individual Jacobian blocks. Good held-out prediction would
support the hypothesis that a compact cross-token modal relationship exists.
It would not be a fitted forward graph executor, an authenticated layer
replacement, or proof of compilation.

The causal curves are currently marked `descriptive_only=true` with no
acceptance threshold. They do not alter the existing `diagnostic_v1` block
classification.

The tracked `diagnostic_v1` profile is intentionally conservative. At a
decision rank, a boundary must have at least 0.60 exact validation capture,
0.20 chance-adjusted A/B overlap, no worse than a 2:1 A/B trace ratio, and no
single validation prompt above 20% of trace. When ranks 96 and 128 are both
available, both must pass. These are diagnostic classification rules, not
representative acceptance criteria.

### Developer-local results

A local CPU run of the earlier row-local trajectory artifact at the pinned
revision completed that path without evaluating test and classified the block
`inconclusive_basis_not_identifiable`. At rank 128:

| Boundary | A/B overlap | Adjusted overlap | Exact validation capture |
|---|---:|---:|---:|
| `layer.4.input` | 0.491 | 0.364 | 0.616 |
| `layer.4.output` | 0.499 | 0.374 | 0.632 |
| `layer.5.output` | 0.519 | 0.399 | 0.686 |
| `layer.6.output` | 0.551 | 0.438 | 0.779 |

The first two boundaries captured only 0.532 and 0.549 at rank 96, so they
failed the two-rank identifiability rule even though they crossed 0.60 at rank
128.

At rank 128, adjacent cross-Rayleigh transfer ratios were approximately:

| Edge | Forward transfer | Reverse transfer | Gradient-map validation \(E_0\) (zero-baseline explained energy) | Activation-map validation \(R^2\) |
|---|---:|---:|---:|---:|
| layer 4 sliding | 0.754 | 0.712 | -0.306 | 0.709 |
| layer 5 global | 0.889 | 0.903 | 0.190 | 0.825 |
| layer 6 sliding | 0.907 | 0.912 | 0.344 | 0.820 |

The later boundaries increasingly preserve each other's Fisher energy, but
the simple same-position reverse-gradient maps do not generalize strongly enough
to call the changes predictable row-local rotations. In particular, those
values do not test the omitted causal cross-position Jacobian mixing and
therefore do not rule out a sequence-aware modal executor. Meanwhile activation
maps perform well, which is consistent with the residual stream remaining
predictable. This supports neither an all-or-nothing rejection nor a
compilation claim. It says the current row-local trajectory model is too weak
or the rank/prompt protocol is insufficient.

The pinned version-2 rerun then evaluated the exact-logical-lag ridge
diagnostic at rank 128 with `--causal-relative-ridge 0.01`. These are
zero-baseline explained-energy values. Edge labels follow the forward block;
gradient prediction runs in reverse.

Calibration fit:

| Forward block edge | Lag 0 \(E_0\) | Lag 1 \(E_0\) | Lag 4 \(E_0\) |
|---|---:|---:|---:|
| `layer.4.input -> layer.4.output` | 0.719 | 0.856 | 0.983 |
| `layer.4.output -> layer.5.output` | 0.789 | 0.865 | 0.981 |
| `layer.5.output -> layer.6.output` | 0.797 | 0.868 | 0.982 |
| `layer.4.input -> layer.6.output` | 0.598 | 0.763 | 0.972 |

Frozen validation:

| Forward block edge | Lag 0 \(E_0\) | Lag 1 \(E_0\) | Lag 4 \(E_0\) |
|---|---:|---:|---:|
| `layer.4.input -> layer.4.output` | 0.091 | -0.026 | -0.720 |
| `layer.4.output -> layer.5.output` | 0.391 | 0.246 | -0.389 |
| `layer.5.output -> layer.6.output` | 0.490 | 0.391 | -0.107 |
| `layer.4.input -> layer.6.output` | -0.190 | -0.412 | -1.404 |

The lag-0 column is not expected to reproduce the earlier gradient-map
column. The earlier map is a whitened orthogonal Procrustes transform; the new
baseline is an unconstrained homogeneous ridge regression. Lag 0 exists to
isolate the value of adding future-position features within the same ridge
family.

The pattern is much more informative than any single number:

- calibration fit improves monotonically as future-lag features are added;
- no edge improves on its independently refit lag-0 baseline on validation;
- lag 4 reaches roughly 0.97–0.98 calibration explained energy but is
  negative on validation for every edge;
- the lag-4 feature condition numbers are approximately
  \(4.75\times10^5\)–\(6.98\times10^5\), indicating a highly collinear,
  ill-conditioned regression problem.

This is severe overfit under the current exact-lag protocol. At rank 128 the
lag-4 map has \(5\times128^2=81{,}920\) coefficients per edge, the bundled
prompt split is small and template-matched, and one shared \(W_\delta\) must
describe all positions and contexts. The held-out result therefore rejects a
stationary homogeneous exact-logical-lag ridge predictor at this prompt set,
rank, lag budget, and ridge value. It does not show that causal
cross-position computation is absent, and it does not reject lower-rank,
factorized, more strongly regularized, nonlinear, or context-conditioned
sequence-aware executors.

The generated `.pt` and `.json` files remain under ignored `.local-runs/`.
The strict-loaded version-2 tensor artifact was approximately 56 MB and its
JSON report approximately 987 KB. The repository records the command,
protocol, and interpretation, not the live derived tensors; the older
version-1 row-local artifact still strict-loads.

## Run the joint full-width modal-sufficiency diagnostic

The fourth rung asks a different question from modal transport: if the native
transformer continues to run, how many leading Fisher coordinates at its
selected layer outputs are sufficient to preserve validation behavior? It
builds a full-width Fisher basis for every selected output from calibration
A+B, freezes those bases and their pooled activation means, and then evaluates
a descending retained-rank curve.

```bash
fisher-graph-gemma-ablation --check-paths-only

fisher-graph-gemma-ablation \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --start-layer 4 \
  --end-layer 6 \
  --max-length 128 \
  --sketch-rows 641 \
  --retained-ranks 640 512 384 256 192 128 96 64 32 0 \
  --include-single-sites \
  --device cpu \
  --dtype float32
```

For this layer range, the selected sites are `layer.4.output`,
`layer.5.output`, and `layer.6.output`; the block input is not part of the
joint projection.

For hidden width \(D=640\), retained rank \(k\), selected layer output
\(\ell\), and valid logical position \(t\), the intervention is

\[
a'_{\ell,t}(k)
=
\mu_\ell
+
\left((a_{\ell,t}-\mu_\ell)U_{\ell,:k}\right)
U_{\ell,:k}^{\mathsf T}.
\]

This is a top-down **sufficiency** curve. It keeps the leading \(k\) Fisher
modes and removes the lowest \(D-k\); it does not remove the leading \(k\)
modes. The primary experiment installs the same retained-rank policy jointly
at every selected native layer output. Because the edits occur inside one
forward pass, later selected layers receive the already projected state from
earlier layers. `--include-single-sites` additionally evaluates each selected
output alone as a localization control; omit it for the primary joint curve
only.

Only valid positions are projected. Padding and other invalid rows remain
unchanged, and every projection is centered on a pooled mean estimated from
calibration A+B—not validation. The source model remains frozen throughout:
this command performs no fine-tuning, retraining, or executor fitting.

Rank 640 is a mandatory full-rank identity gate. Its projected activations,
logits, and NLL must match the unmodified validation forward within the
declared numerical tolerance before any lower-rank result is interpreted. At
the other extreme, rank 0 replaces every valid selected-output activation
with its pooled calibration mean. That point is a deliberately destructive
reference; it does not represent a zero-width runnable model.

The curve is evaluated on validation, and the reserved test split remains
parse/validate/hash-only. A lower retained rank that preserves NLL would show
that the corresponding leading subspaces are sufficient under this particular
joint projection. It would not yet prove that those modes can be generated
without the discarded computation: the source transformer still executes all
native layers at full width before each projection. It therefore makes no
executor, storage, arithmetic, latency, or compilation claim. Those claims
require a fitted replacement followed by local equivalence, end-to-end NLL,
variable-sequence, causality, fallback, and runtime performance gates.

### Developer-local sufficiency result

The strict-loaded coarse joint run passed its full-rank identity gate. The
unmodified validation baseline was 4.271092 NLL per supervised token, and the
rank-640 projected run differed by only
\(-4.28\times10^{-7}\) NLL/token with perfect top-1 agreement. Selected points
from the predeclared coarse curve were:

| Joint retained rank | Delta NLL/token | Interpretation |
|---:|---:|---|
| 640 | -0.000000428 | Full-rank identity gate passed |
| 512 | +3.641870 | Top-1 agreement 0.2241 |
| 384 | +2.954244 | Large degradation |
| 128 | +3.902732 | Large degradation |
| 0 | +4.369668 | Calibration-mean replacement |

Rank 512 still retained approximately 99.97% of calibration Fisher trace at
each selected output—99.970% to 99.977% across the three sites—yet raised
NLL/token by about 85% and preserved only 78 of 348 baseline top-1 token choices.
All 16 validation prompts had positive per-prompt NLL deltas, so this failure
was not caused by one prompt outlier. The coarse curve is not monotone—for
example, rank 384 is less damaging than rank 512—which is another warning
that these are finite, interacting interventions rather than an additive
importance accounting.

After inspecting that coarse curve, a separate **posthoc validation
refinement** tested ranks immediately below full width and added single-site
localization. Joint rank 639, which removes only the final Fisher mode at each
selected output, increased NLL/token by 1.783649 and reduced top-1 agreement
to 0.3621. At the same rank, the one-site-at-a-time NLL/token deltas were:

| Posthoc rank-639 site | Delta NLL/token |
|---|---:|
| `layer.4.output` | +0.086312 |
| `layer.5.output` | +1.093586 |
| `layer.6.output` | +1.148882 |

Those values localize large finite sensitivity to the nominal Fisher tail,
especially at layers 5 and 6, but they were chosen after observing validation
and therefore cannot serve as a predeclared acceptance curve or an unbiased
rank-selection result.

The accompanying activation-amplitude diagnostic resolves the apparent
contradiction between “almost all Fisher trace retained” and “behavior
destroyed.” Tail modes have enormous centered activation RMS on both
calibration and validation. Representative values are about 4,257 for the
last mode at `layer.5.output`, 8,915 for the last mode at
`layer.6.output`, and 5,163 for zero-based mode 638 at
`layer.4.output`. Their Fisher eigenvalues are tiny because an activation
Fisher eigenvalue measures local squared score-gradient projection at the
native activation distribution. It does not multiply that local sensitivity
by the size of the finite displacement made by mean replacement.

Deleting one of these high-amplitude coordinates moves the residual stream
far from its native manifold. Subsequent RMSNorm is nonlinear in the whole
residual vector, so the edit can change the scale and direction seen by many
otherwise retained coordinates. The local Fisher ordering alone does not
capture that finite off-manifold effect.

One more **ad hoc, read-only posthoc diagnostic** multiplied each Fisher
eigenvalue by that mode's calibration activation variance and reordered the
modes by the resulting score. Its joint results were:

| Retained rank | Delta NLL/token | Top-1 agreement |
|---:|---:|---:|
| 639 | -0.00123 | 0.922 |
| 638 | +0.01224 | 0.825 |
| 636 | +0.05486 | 0.747 |
| 632 | +0.4513 | 0.575 |

This is sharply better than native Fisher ordering at rank 639
(+1.78365 NLL/token and 0.362 top-1 agreement), which supports the narrower
diagnosis that finite importance depends on activation displacement as well
as local Fisher sensitivity. A separate per-token norm-preserving projection
did not rescue native Fisher ordering and usually worsened it. Neither
diagnostic was artifacted or predeclared, so neither is confirmatory or a
valid basis for selecting a rank. At most, the reordering motivates a
pre-registered variance-weighted or generalized-Fisher basis experiment.
Removing only one dimension at each site is also not meaningful compression.

Under the native eigenvalue-only keep-top-\(k\) ordering and joint projection
protocol, only rank 640 preserved validation behavior. The primary result
therefore provides no viable compression rank. It does not establish that
modal compression is impossible: an amplitude-aware criterion, constrained
reconstruction, or a trained executor is a different hypothesis and would
need fresh, predeclared validation. The corpus here is only 16 short,
template-matched validation prompts, so the result is a strong failure of
this diagnostic protocol rather than a population-wide model claim. The
coarse, singleton, and posthoc-refinement artifacts remain ignored and
uncommitted; the ad hoc diagnostics were not artifacted; model weights were
frozen and the reserved test prompts were never model-evaluated.

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

The stability artifact contains three calibration collections and the exact
validation Rayleigh results for all three frozen bases. Its JSON rank curve
places split overlap, sketch fractions, eigengaps, and exact held-out
fractions side by side. Protocol metadata binds the run to a tokenizer
configuration digest, exact tokenized-stream digests, and per-example
serialized-input and padding-independent token-content hashes plus valid and
supervised token counts. Ordered source-prompt hashes prove that the combined
calibration stream is A+B and that no reserved-test prompt was tokenized. The
reserved test split has normalized prompt hashes only. The artifact still
contains no model weights, tokenizer files, prompt text, or executor state.

The trajectory artifact adds canonical block boundaries, split and full-depth
geometry, bounded modal sufficient statistics, calibration-frozen transports,
exact own- and cross-Rayleigh accounting, held-out transport residuals, and
per-prompt scalar and per-mode influence ledgers. Format version 2 additionally
stores exact-logical-lag causal moments, rank/lag-prefix ridge maps, structural
visibility, lag-pair accounting, and their frozen validation evaluations.
The current writer emits version 2. The strict loader remains compatible with
version 1 row-local artifacts; it validates their original field set without
inventing causal results, while version 2 requires and recomputes the causal
payload.

The strict loader recomputes the saved geometry and transport scores, rebinds
every frozen transport and evaluation to the named calibration-full Fisher
bases, and checks each rank-prefix ledger coordinate against the aggregate
replay. For version 2 it also rebuilds every causal ridge prefix from the saved
calibration moments and reevaluates the frozen maps from validation moments.
The tensor artifact stores the canonical SHA-256 of its sibling JSON report,
while the JSON report commits to the complete scientific tensor payload. The
loader rebuilds every scalar curve and classification from the strict-loaded
payload. This two-way binding rejects a one-file edit or stale swap; preventing
a coordinated rewrite of both files would require an external signature. The
large modal matrices stay in the ignored tensor artifact; the JSON report
contains scalar curves and classifications rather than duplicating those
matrices.

The modal-sufficiency artifact contains the full-width calibration Fisher
bases and pooled activation means for the selected layer outputs, the
unmodified validation baseline, the full-rank identity-gate result, and the
joint retained-rank curve. When requested, it also contains the single-site
localization curves. Protocol metadata records that only valid positions were
projected, all sites at a joint curve point used the same retained rank, model
weights stayed frozen, and test remained hash-only. Its tensor and JSON
outputs contain derived analysis data, never pretrained weights, prompt text,
tokenizer files, or executor state.

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
that exact calibration trace. It is not an exact statement about the frozen
subspace on unseen text. The stability command's validation \(R_k\) is the
separate exact held-out measurement, still conditional on the selected prompt
distribution and layer.

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

A successful opt-in smoke run establishes real-model activation access and
bounded-memory mode extraction for its recorded revision and prompt set. The
stability and trajectory CLIs now implement split comparison, exact validation
replay, multi-boundary geometry, row-local transport, and frozen exact-lag
reverse-causal prediction. The ablation CLI adds a full-width, jointly applied
keep-top-\(k\) sufficiency curve with a full-rank identity gate. The bundled
diagnostic fixture is not the representative evidence needed to pass the
compilation gate. It is short and template-matched, its bases remain
inconclusive at the decision ranks, and its rank-128 lagged ridge maps overfit
calibration without a held-out cross-position gain. The strict-loaded
sufficiency run passed full-rank identity, but every lower tested rank under
native Fisher ordering failed the validation-NLL gate. The ad hoc
variance-weighted rank-639 result is only a hypothesis-generating hint, saves
just one dimension per site, and was not artifacted or confirmatory. The
reserved test split remains model-unevaluated. None of these commands proves
that a graph executor can replace a layer or block.

The next scientific work is:

1. freeze a larger, representative and token-length-stratified calibration,
   validation, and reserved test corpus;
2. use the recorded per-example IDs, hashes, token counts, and per-prompt
   Fisher influence; add explicit truncation and length buckets, then inspect
   content and length groups for outlier domination;
3. rerun the nested rank and sketch-capacity curves, including wider ranks
   where validation capture is still rising;
4. pre-register and pass split-overlap, exact held-out capture, trace-balance,
   and per-bucket gates without model-evaluating test;
5. pre-register an amplitude-aware or generalized-Fisher criterion, including
   its exact score or basis construction, native-Fisher comparator, rank grid,
   and validation thresholds; require both full-rank identity and a
   predeclared validation-NLL tolerance before treating any lower-rank
   subspace as sufficient;
6. rerun the exact-lag diagnostic across predeclared lower ranks, lag budgets,
   and regularization values on the larger corpus; require a held-out gain over
   the independently refit lag-0 ridge baseline before considering a
   factorized or context-conditioned causal transport;
7. only if the boundaries are identifiable and a retained rank passes the NLL
   gate, fit one variable-length joint causal executor with optional internal
   trajectory losses; the projection curve alone is not a compression claim;
8. require local boundary, internal modal, end-to-end NLL, sequence-length,
   causal-leakage, fallback, storage, arithmetic, and latency gates before
   replacing that block in the mixed runtime;
9. lock the rank, executor, and thresholds, then model-evaluate the reserved
   test split exactly once without further selection.
