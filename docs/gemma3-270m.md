# Opt-in Gemma 3 270M Fisher, codec, executor, and projection rungs

These are the first external-model scaling rungs. They exercise the generic
adapter, streaming Fisher, split-stability, exact held-out Rayleigh replay,
multi-boundary modal transport, and joint full-width modal-sufficiency
interfaces on a real text decoder without checking the model or its cache into
this repository. The activation-aware follow-up also exercises split-safe
codec selection, bounded true forward JVPs, and causal weighted factorization.
The next rung fits a residual-separated gated causal graph against a real
three-layer block. A fresh projection-only rung then asks where behavior
recovers when the true block delta is reconstructed in nested prefixes of the
selected output decoder. The committed suite validates this plumbing with
synthetic models; no checkpoint or live tensor artifact is committed.
Developer-local measurements are reported below, but neither follow-up
produced a viable compression and no Gemma speed result is accepted.

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
- streamed activation covariance, native/variance/generalized codecs, a
  calibration-B rank/family lock, and locked-only validation;
- a bounded signed forward-JVP lag probe and synthetic weighted causal-prefix
  factor;
- a fresh four-way gated block-delta fit with an exact residual bypass,
  state-conditioned positive-lag experts, and locked-only validation;
- a second source-disjoint four-way projection ladder with calibration-B
  selection, one locked validation intervention, and calibration A/test
  reserved hash-only;
- a reserved test split that none of these analysis commands model-evaluate;
- no fine-tuning, weight updates, installed Gemma graph, or compilation claim.

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

## Run activation-aware codec selection and weighted forward edges

The next rung pre-registers the amplitude-aware alternatives and restores a
four-way split:

```bash
fisher-graph-gemma-weighted-jacobian \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --start-layer 4 \
  --end-layer 6 \
  --max-length 128 \
  --retained-ranks 632 636 638 639 640 \
  --sketch-rows 641 \
  --generalized-regularization 1e-3:1e-6 \
  --generalized-regularization 1e-2:1e-5 \
  --selection-nll-atol 0.05 \
  --selection-top1-min 0.95 \
  --jacobian-max-sequences 4 \
  --jacobian-modes 4 \
  --jacobian-max-lag 4 \
  --jacobian-factor-rank 2 \
  --device cpu \
  --dtype float32
```

The split ownership is:

- calibration A fits complete Fisher estimates, activation covariance, and
  every predeclared codec;
- calibration B evaluates the canonical rank-then-family schedule, requires
  every codec family to pass its own full-width behavioral identity, and then
  locks the first reduced candidate with absolute aggregate delta NLL/token at
  most 0.05 and top-1 agreement at least 0.95;
- validation evaluates only the locked candidate, the locked family's
  full-width identity, and the native full-width identity, deduplicating the
  two identity executions when the locked family is native;
- test is validated and hashed, but never tokenized or model-evaluated.

The codec families are:

1. native Fisher eigenvalue order;
2. the same Fisher vectors reordered by eigenvalue times activation variance;
3. generalized dual codecs from
   \(C_{\rm reg}^{1/2}F_{\rm reg}C_{\rm reg}^{1/2}\), with explicit absolute
   activation/Fisher floors.

The strict-loaded pinned run selected
`generalized_fisher.reg_01.joint.rank_636`, using activation floor 0.01 and
Fisher floor \(10^{-5}\):

| Split | Delta NLL/token | Top-1 agreement |
|---|---:|---:|
| Calibration B selection | -0.003316 | 0.9643 |
| Locked validation | +0.010285 | 0.9626 |
| Locked-family validation identity | \(-6.58\times10^{-8}\) | 1.0000 |
| Native validation identity | \(-6.58\times10^{-8}\) | 1.0000 |

All four codec families also passed their separate calibration-B full-width
identity controls. At rank 636, the native and variance-weighted codecs changed
calibration-B NLL/token by about +2.928 with 0.2887 top-1 agreement. The weaker
generalized floor changed it by +0.003392 with 0.9494 agreement and narrowly
missed the top-1 gate; the stronger generalized floor produced the selected
-0.003316 and 0.9643 result. This is the useful distinction between the two
implemented approaches: reordering fixed Fisher vectors did not fix the tail,
while changing to an activation-aware dual subspace did.

The validation baseline was 4.271092 NLL/token over 348 supervised tokens.
Nine of 16 prompts had positive NLL deltas and seven had negative deltas; the
per-prompt range was -0.02280 to +0.05344 NLL/token. Seven prompts had exact
top-1 agreement, while the minimum per-prompt agreement was 0.8947. The
predeclared selection gates are aggregate, so those per-prompt values are
important follow-up diagnostics rather than gate failures.

The tensor artifact strict-loads under format version 2 and binds to scientific
payload SHA-256
`c6bd8d666c75fd60ec461936ff17c631f2ce62b25915bbd6adcb2cebffda1c11`.
The validation prompts are isolated from calibration A/B inside this protocol,
but earlier exploratory experiments in this repository used the same prompt
file. This result is therefore a controlled replication on reused validation,
not a globally untouched confirmatory result. The reserved test split remains
unevaluated.

Calibration A contains only 366 valid rows for residual width 640. Its
empirical Fisher therefore has a nullspace of at least 274 dimensions.
Variance weighting cannot order that tail because
\(0\times\operatorname{Var}(z_i)=0\); in the live artifact, its final eight
columns remained the native final eight and its high-rank behavior was
identical to native Fisher. The generalized Fisher floor supplies sensitivity
inside that unidentified tail, allowing activation covariance to choose its
directions. This is why the result is promising but regularization-dependent:
a larger, more representative calibration set must test whether the same
subspace and floor choice remain stable.

### Forward-JVP and merge pilot

After calibration B locks the codec, the optional pilot replays only a bounded
calibration-A prefix. It perturbs one input codec decoder column at one valid
position, executes the real frozen layers 4–6 under forward-mode autodiff, and
projects the tangent through the output codec encoder. Signed mean and RMS
edges are pooled by exact logical lag; RMS is never used as an executable
weight.

With four sequences, a 4-by-4 projected modal slice, and lags 0–4, the run made
320 JVP calls. It found:

| Quantity | Result |
|---|---:|
| Temporal-window coverage inside the projected slice | 99.7702% |
| Future-position causal leakage | 0 |
| Aggregate projected energy in stationary signed lag mean | 95.6683% |
| Aggregate projected within-lag/context variation | 4.3317% |
| Positive-lag energy in stationary signed lag mean | 34.9292% |
| Positive-lag within-lag/context variation | 65.0708% |

The aggregate number is dominated by the same-position path:

| Logical lag | Projected energy | Constant signed mean | Context/position variation |
|---:|---:|---:|---:|
| 0 | 2083.3681 | 96.0304% | 3.9696% |
| 1 | 7.1768 | 45.3284% | 54.6716% |
| 2 | 1.8949 | 30.4113% | 69.5887% |
| 3 | 1.9809 | 19.2349% | 80.7651% |
| 4 | 1.3663 | 9.3255% | 90.6745% |

These values cover only four input and four output directions out of width
640; 99.7702% is lag-window coverage within that slice, not captured energy of
the full block Jacobian. The fractions are also specific to the locked
generalized coordinate gauge and are not comparable across codec families.
Four calibration prompts cannot establish held-out stationarity. The evidence
now favors separating the lag-zero/residual-like path and testing a
state-conditioned or small causal-mixture transport for positive lags, rather
than treating one constant lag map as the default executor.

The signed lag mean is expanded into a synthetic \(T=L+1=5\) causal Toeplitz
reference. The runner transforms pooled activation covariance and output
Fisher into the locked modal coordinates and independently factors each
output prefix. Rank two retained 97.3013% of that chosen weighted operator
energy:

| Uniform prefix rank | Weighted energy retained | Factor coefficients/MACs | Ratio to unshared dense |
|---:|---:|---:|---:|
| 1 | 85.2134% | 80 | 0.3333 |
| 2 | 97.3013% | 160 | 0.6667 |
| 3 | 99.9898% | 240 | 1.0000 |
| 4 | 100.0000% | 320 | 1.3333 |

The denominator is an explicitly unshared 240-coefficient dense causal
operator. A natural five-lag shared map stores only 80 edge coefficients, so
the rank-two factor is not a storage reduction against the sensible shared
baseline. The factor count also omits codecs, masks, routing, memory traffic,
and the original transformer. Moreover, 99.8305% of the synthetic weighted
reference energy is at lag zero, so the 97.3013% rank-two curve mostly measures
same-position transport rather than cross-token mergeability. It provides
exact SVD/tail accounting for a rank-two approximation and analytic MAC counts,
not exact reconstruction and not a Gemma parameter, FLOP, storage, or latency
claim.

Rank 636 removes four of 640 coordinates at each of three intervention sites:
0.625% of each residual width and 12 coordinate slots in the joint
intervention. The source layers still execute at full width. The result
therefore establishes a viable activation-aware node-selection direction, not
useful compression by itself. It motivated the real block-output predictor
below, with an explicit same-position path and a gated positive-lag mixture.
The weighted equations and generic interfaces are documented in
[`weighted-jacobian-compilation.md`](weighted-jacobian-compilation.md).

## Run the residual-separated gated block-output experiment

The next command fits the executable hypothesis suggested by the JVP pilot:

```bash
fisher-graph-gemma-gated-executor \
  --weighted-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_gated_executor_prompts.json \
  --max-length 128 \
  --retained-ranks 320 480 \
  --expert-counts 1 2 \
  --expert-ranks 16 \
  --router-widths 16 \
  --max-positive-lags none \
  --fit-steps 100 \
  --device cpu \
  --dtype float32
```

The weighted artifact supplies the already selected input/output codecs and
their provenance, not model weights. The new prompt fixture has 16
source-disjoint prompts in each split. Calibration A fits every candidate for
one fixed 100-step schedule; calibration B applies the predeclared fidelity
and resource gates and locks one candidate; validation evaluates only that
lock; test is parsed and hashed but not tokenized or model-evaluated.

For input residual \(h_{\rm in}\), retained rank \(r\), and output codec
decoder \(D_{\rm out}\), the intervention is

\[
\widehat h_{\rm out}
=h_{\rm in}
+G\!\left(
(h_{\rm in}-\mu_{\rm in})E_{\rm in,:r}
\right)D_{\rm out,:r}^{\mathsf T}.
\]

The raw residual is therefore an exact bypass. \(G\) predicts only the
layers-4–6 block delta. It has an independent same-position affine path and
one or two rank-16 positive-lag experts. A width-16 router conditions each
legal edge on query state, source state, and the logarithm of relative logical
lag. No expert reads the current or a future position through the cross-token
path, and no parameter shape depends on sequence length.

The seven conjunctive gates require retained fraction, stored-coefficient
ratio, and analytic-MAC ratio at most 0.75; block-delta NRMSE at most 0.20;
block-delta cosine at least 0.95; absolute delta NLL/token at most 0.05; and
top-1 agreement at least 0.95. If no configuration passes on calibration B,
the protocol still locks the best direct diagnostic fallback but labels it
nonviable before validation.

### Developer-local gated-executor result

No rank-320 or rank-480 candidate passed selection. The diagnostic fallback
locked
`rank_320.experts_2.expert_rank_16.router_16.lag_all`. Its one validation
evaluation was:

| Validation quantity | Result | Required gate |
|---|---:|---:|
| Block-delta NRMSE | 0.823388 | at most 0.20 |
| Block-delta cosine | 0.605518 | at least 0.95 |
| Delta NLL/token | +7.015665 | absolute value at most 0.05 |
| Top-1 agreement | 0.07381 | at least 0.95 |
| Stored coefficients / source block parameters | 3.2518% | at most 75% |
| Analytic MACs / source block analytic MACs | 3.2290% | at most 75% |

This cleanly separates a resource result from a quality result. The locked
runtime state uses about 2.17 MB in FP32 and only 3.2518% as many stored
coefficients as the exact 16,720,896 parameters in source layers 4–6.
Matched-shape analytic MACs are only 3.2290% of the source comparison. But the
executor does not approximate the block closely enough and destroys language
behavior, so these ratios are not a viable compression result.

The stored-coefficient count includes the retained input mean and encoder,
output decoder, and graph parameters. The graph count includes all experts in
the soft mixture. The analytic MAC comparison includes codec encode/decode,
the complete routed graph, source linear projections, and source QK/AV dot
products on the same causal validation edges. It excludes normalization,
nonlinear activations, softmax, RoPE, masking, additions, memory traffic, and
kernel overhead. It is therefore neither a FLOP-complete accounting nor a
speed measurement.

The rank-320 target-informed, per-token least-squares reference projected the
*true* block delta into the same output decoder span. Its validation direct
NRMSE was 0.055995 with cosine 0.998431, but intervention delta NLL/token was
still +6.342280 and top-1 agreement only 0.088095. Because this calculation
uses the true target delta, it is not an inference-time executor. It is also
not a behavioral upper bound: it minimizes Euclidean error rather than NLL or
top-1 disagreement. It does show that even a small residual-space error in
this rank-320 output subspace can land in directions that later layers amplify
severely.

The no-op intervention and full-width codec delta round trip passed. Model
weights remained frozen, the new fixture was hash-disjoint from the source
artifact, and the generic executor's causality, padding, variable-length, and
strict-loading controls pass. This makes the failure scientifically useful:
it is not explained by a broken hook, codec identity, weight mutation, prompt
reuse, or simple causal-mask error.

The result rejects this pinned rank-320/480, one-seed, fixed residual-MSE
protocol. It is not evidence that causal routing never helps, that a different
subspace or behavior-aware objective cannot work, or that modal compression
is impossible. See [`gated-executor.md`](gated-executor.md) for the full graph
and claim boundary.

Without `--output`, this command writes the ignored local files:

```text
.local-runs/google--gemma-3-270m/layers-4-6-gated-executor.pt
.local-runs/google--gemma-3-270m/layers-4-6-gated-executor.json
```

The roughly 16 MB `.pt` is a diagnostic artifact containing all four fitted
candidate states and audit data. It should not be confused with the
approximately 2.17 MB state needed by the one locked FP32 candidate. Model and
tokenizer files stay in the external Hugging Face cache.

## Run the target-informed projection-only behavioral rank ladder

The gated result mixes two possible failures: the output span may discard
behaviorally necessary directions, and the fitted graph may fail to generate
the coordinates that remain. This follow-up removes graph fitting from the
question. For native block input \(h_{\rm in}\), output \(h_{\rm out}\), and
the first \(r\) columns \(D_r\) of the locked generalized decoder, it computes
independently at every valid token

\[
z_r^*=\arg\min_z
\left\lVert
(h_{\rm out}-h_{\rm in})-zD_r^{\mathsf T}
\right\rVert_2^2,
\qquad
\widehat h_{\rm out}=h_{\rm in}+z_r^*D_r^{\mathsf T}.
\]

It therefore consumes the true native target delta. This is a representation
diagnostic, not an inference-time executor or behavioral upper bound:
least-squares projection optimizes Euclidean error rather than NLL or top-1
agreement.

```bash
fisher-graph-gemma-projection-ladder \
  --weighted-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt \
  --gated-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-gated-executor.pt \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_projection_ladder_prompts.json \
  --max-length 128 \
  --retained-ranks \
    480 512 544 576 592 608 616 624 632 636 638 639 640 \
  --selection-nll-atol 0.05 \
  --selection-top1-min 0.95 \
  --max-meaningful-retained-fraction 0.75 \
  --device cpu \
  --dtype float32
```

Both predecessor artifacts strict-load before model access. The 64 new prompt
hashes are disjoint from both predecessors. Calibration A and reserved test
are parsed and hashed only. Calibration B sees the full rank curve and locks
the smallest reduced rank passing both aggregate behavior gates. Direct
NRMSE/cosine never influence the lock. Full width must first pass independent
least-squares and codec round-trip identity controls; failure stops the run
before validation. Validation then sees exactly one locked-rank intervention
per batch.

### Developer-local projection-ladder result

The calibration-B curve covered 16 prompts and 457 supervised tokens:

| Rank | Retained | Removed | Direct delta NRMSE | Direct delta cosine | Delta NLL/token | Top-1 | Pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 480 | 75.000% | 160 | 0.043091 | 0.999071 | +5.511701 | 0.1860 | no |
| 512 | 80.000% | 128 | 0.036511 | 0.999333 | +5.090366 | 0.2385 | no |
| 544 | 85.000% | 96 | 0.031881 | 0.999492 | +1.969828 | 0.3917 | no |
| 576 | 90.000% | 64 | 0.029541 | 0.999564 | +1.600023 | 0.4354 | no |
| 592 | 92.500% | 48 | 0.027734 | 0.999615 | +1.115439 | 0.4376 | no |
| 608 | 95.000% | 32 | 0.024942 | 0.999689 | +0.838136 | 0.4836 | no |
| 616 | 96.250% | 24 | 0.023160 | 0.999732 | +0.705360 | 0.5098 | no |
| 624 | 97.500% | 16 | 0.019913 | 0.999802 | +0.348489 | 0.6105 | no |
| 632 | 98.750% | 8 | 0.017229 | 0.999852 | +0.293920 | 0.6783 | no |
| 636 | 99.375% | 4 | 0.007384 | 0.999973 | +0.014626 | 0.8993 | no |
| 638 | 99.688% | 2 | 0.005928 | 0.999982 | +0.011915 | 0.8972 | no |
| 639 | 99.844% | 1 | 0.003633 | 0.999993 | -0.003372 | 0.9431 | no |
| 640 | 100.000% | 0 | approximately 0 | 1.000000 | approximately 0 | 1.0000 | yes |

No reduced rank passed. Ranks 636, 638, and 639 satisfy the NLL gate but miss
top-1. Rank 639 is the closest: it agrees on 431 of 457 tokens, while the 0.95
gate requires at least 435. It preserves approximately 99.99868% of direct
block-delta energy, yet removing that final prefix direction changes 26 token
argmaxes. Top-1 also falls slightly from rank 636 to 638 even as direct
reconstruction improves. Nested projection MSE is monotone; downstream
behavior need not be.

The required fallback therefore locked rank 640, with
`selection_failed=true`. Its one validation evaluation covered 16 prompts and
447 supervised tokens:

| Locked validation quantity | Result |
|---|---:|
| Direct block-delta NRMSE | \(1.2353\times10^{-12}\) |
| Direct block-delta cosine | 1.000000 |
| Delta NLL/token | \(+2.7309\times10^{-7}\) |
| Top-1 agreement | 1.000000 |

The independent calibration-B FP32 codec round trip had block-delta NRMSE
\(1.8654\times10^{-6}\) and exact top-1 behavior; its FP64 mathematical
control was at numerical identity. These controls, the frozen-model guard,
prompt exclusion, and strict aggregate reconstruction make a broken hook,
codec, or numerical identity an implausible explanation for the reduced-rank
failure.

This result does **not** say that 640 dimensions are intrinsically necessary.
It rejects nested prefix truncation of one locked generalized-Fisher decoder
on this fresh, small diagnostic fixture. It did not search arbitrary
rank-\(r\) subspaces, reorder decoder columns by downstream behavior, or train
a predictor. Validation confirms only the rank-640 fallback identity; no
reduced rank was evaluated there. Test remains untouched. Only rank 480 meets
the predeclared 0.75 retained-fraction target, and it fails behavior by a wide
margin. Because source layers 4–6 still execute and the true target delta is
used, no parameter, MAC, storage, compression, or latency claim follows.

The ignored local outputs are:

```text
.local-runs/google--gemma-3-270m/layers-4-6-projection-ladder.pt
.local-runs/google--gemma-3-270m/layers-4-6-projection-ladder.json
```

The developer-local tensor artifact is about 6.5 MB and its tensor-free JSON
report about 875 KB. Scientific payload SHA-256:
`c98be0bd937bb0031480f3ce4912df57c54b671f291d6c9b1412f45af265b9bf`.
They contain the derived output codec and audit data, but no pretrained model
weights, prompt text, tokenizer state, or reserved-test evaluation.

## Run the codimension-one tail-rotation diagnostic

The projection ladder rejected the first 639 columns of the selected decoder,
not every possible 639-dimensional hyperplane. This follow-up asks whether the
failure is the *ordering* of the final decoder directions.

Let \(U\in\mathbb{R}^{640\times32}\) be an orthonormal basis for the Euclidean
complement of the decoder's first 608 columns. On calibration A, the runner
collects the downstream pseudo-top-1 score gradient \(g\) and native block
delta \(\delta=h_{\rm out}-h_{\rm in}\) at every valid position, then forms

\[
F=\frac{1}{N}\sum (U^{\mathsf T}g)(U^{\mathsf T}g)^{\mathsf T},
\qquad
C=\frac{1}{N}\sum (U^{\mathsf T}\delta)
                   (U^{\mathsf T}\delta)^{\mathsf T},
\]

\[
M=\frac{1}{2}\frac{F}{\operatorname{tr}F}
 +\frac{1}{2}\frac{C}{\operatorname{tr}C}.
\]

The rotated candidate omits \(q_*=Uv_{\min}(M)\) and projects the *true*
native block delta with \(I-q_*q_*^{\mathsf T}\). The controls are the
rank-639 source-codec prefix, whose omitted normal is solved from the decoder,
and full-width identity. Ground-truth NLL score Fisher is recorded as a
non-fitting control. Calibration-B direct reconstruction metrics are
diagnostic only; aggregate absolute delta NLL/token and top-1 agreement alone
lock rotated, then codec-prefix, then identity. Validation evaluates exactly
one locked reduced candidate, or is never tokenized if neither reduced
candidate passes.

The completed expanded-A objective-regret run was:

```bash
fisher-graph-gemma-codimension-rotation \
  --projection-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-projection-ladder.pt \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits \
    examples/gemma3_codimension_rotation_expanded_a_prompts.json \
  --max-length 128 \
  --tokenization-batch-size 4 \
  --tail-width 32 \
  --selection-nll-atol 0.05 \
  --selection-top1-min 0.95 \
  --identity-nll-atol 0.00001 \
  --minimum-split-half-alignment 0.8 \
  --minimum-relative-eigengap 0.001 \
  --stability-policy split_half_objective_regret \
  --minimum-split-half-operator-cosine 0.99 \
  --maximum-split-half-relative-regret 0.10 \
  --max-meaningful-retained-fraction 0.75 \
  --device cpu \
  --dtype float32 \
  --output \
    .local-runs/google--gemma-3-270m/layers-4-6-codimension-rotation.pt
```

### Why stability is objective regret, not direction agreement

The first preregistered run used 16 calibration-A prompts and required
split-half omitted-normal alignment of at least 0.8. Its alignment was only
0.1457, so the runner stopped before calibration B. Calibration A was then
expanded to 64 prompts while the exact 16-prompt calibration-B, validation,
and test splits remained prompt-for-prompt unchanged. Direction alignment fell
again, to 0.039824. That attempt also stopped before B.

Those failures establish that the *identity of one minimum eigenvector* is not
stable. They do not establish that the fitted low-cost objective is unstable.
In a shallow or nearly degenerate bottom eigenspace, a small perturbation can
rotate the minimizing vector sharply while leaving the operator and the cost
of the pooled candidate nearly unchanged. Because the experiment needs a
low-cost direction, not recovery of a privileged physical axis, the expanded
protocol adopted two A-only objective checks before B was ever inspected.

For split-half operators \(M_0,M_1\), it requires normalized Frobenius cosine

\[
\frac{\langle M_0,M_1\rangle_F}
{\lVert M_0\rVert_F\lVert M_1\rVert_F}\geq 0.99
\]

and, for the pooled candidate \(v_*\), worst split-half relative regret

\[
\max_i
\frac{v_*^{\mathsf T}M_iv_*-\lambda_{\min}(M_i)}
{\operatorname{tr}(M_i)/32}
\leq 0.10.
\]

The pooled operator must also retain relative minimum eigengap at least
0.001. Exact split-direction alignment remains recorded but does not gate the
expanded protocol. The 64-prompt fit passed with operator cosine 0.994295,
worst relative regret 0.062854, and pooled relative eigengap 0.006218. The
two half regrets were 0.062854 and 0.015513. This is evidence that the
objective surface and the chosen candidate's objective value are stable
enough under the declared thresholds; it is not evidence that the omitted
direction itself is identifiable.

The expanded fixture contains the original 16 A prompts plus 48 new A prompts.
Its B, validation, and test arrays exactly equal the original fixture. Their
ordered normalized hashes remained:

| Split | Prompts | Ordered normalized SHA-256 |
|---|---:|---|
| Calibration A | 64 | `da05f80d72197e277542ad5f7d0211ca0f0e1462a40daf0f68533bdce8ae6d1e` |
| Calibration B | 16 | `7f21cb8641bc5898fdcaa2e4c88b57c844afab0c1e7082c3dae938d635d22eb5` |
| Validation | 16 | `5bc16895f3bf42922d24bfeeb940619dcb69e3fd1e923f160f88ba012cb846a4` |
| Test | 16 | `1b740bbc091551a44b3364b5fa3b9b0d56e1fd9d93b44d20d9f295aecf6dcccc` |

### Developer-local codimension-one result

Calibration A covered 64 sequences, 1,931 valid positions, and 1,867
supervised tokens. The rotated direction Pareto-dominated the codec-prefix
normal on both fitting terms:

| Calibration-A quantity | Rotated normal | Codec-prefix normal |
|---|---:|---:|
| Pseudo-top-1 Fisher trace fraction | 0.015750 | 0.024026 |
| Block-delta moment trace fraction | 0.001336 | 0.023465 |
| Combined objective | 0.008543 | 0.023745 |
| Ground-truth NLL Fisher trace fraction, control only | 0.021358 | 0.023176 |

The absolute alignment between the rotated and codec-prefix normals was
0.046074, so this is materially a different rank-639 hyperplane rather than a
numerical replay of the failed prefix.

Calibration B covered 16 sequences, 461 valid positions, and 445 supervised
tokens. The behavior gates were absolute delta NLL/token at most 0.05 and
top-1 agreement at least 0.95:

| Calibration-B candidate | Direct delta NRMSE | Direct delta cosine | Delta NLL/token | Top-1 | Gate |
|---|---:|---:|---:|---:|:---:|
| Rotated rank 639 | 0.000870 | 0.999999622 | +0.000382 | 438/445 = 0.984270 | pass |
| Codec-prefix rank 639 | 0.003792 | 0.999992810 | +0.016279 | 414/445 = 0.930337 | fail |
| Rank-640 identity | 0 | 1.000000000 | 0 | 445/445 = 1.000000 | pass |

The codec-prefix control passed NLL but missed top-1, while the preferred
rotated candidate passed both gates and locked before validation. Validation
then evaluated only that rotated rank-639 intervention over 16 sequences, 475
valid positions, and 459 supervised tokens:

| Locked validation quantity | Result |
|---|---:|
| Direct block-delta NRMSE | 0.000941 |
| Direct block-delta cosine | 0.999999557 |
| Baseline NLL/token | 5.314410 |
| Projected NLL/token | 5.314726 |
| Delta NLL/token | +0.000316 |
| Top-1 agreement | 455/459 = 0.991285 |

Both validation behavior gates passed. Together with calibration-A Pareto
dominance and the same-rank codec-prefix failure on B, this supports the
artifact's narrow classification:

- `basis_ordering_supported=true`;
- `rank_639_fidelity_viable=true`;
- `meaningful_rank_compression=false`.

The first claim means the previously failed final prefix direction was the
problem for this frozen model, layers 4–6, and prompt protocol: another shared
codimension-one hyperplane preserved aggregate behavior on both B and locked
validation. The second is a representation result. It says the *true native
block delta* can be projected into that 639-dimensional hyperplane with the
reported fidelity.

It is not yet a compression or execution result. Rank 639 retains 99.84375%
of the 640-dimensional boundary and exceeds the declared 75% meaningful-rank
threshold. The original layers still execute at full width, their native
delta is required before projection, and no graph generates the retained
representation. No model parameters were removed, no executable layer was
installed, and no parameter, FLOP, MAC, storage, latency, or kernel-speed
reduction follows. The artifact explicitly sets inference executor,
behavioral-upper-bound, compression, and parameter/MAC-speed claims false.

Reserved test remained untouched: it was parsed and hash-validated, but it is
absent from the tokenized streams and was never model-evaluated. Model weights
remained frozen and are not present in either output. The strict artifact
binds to model commit
`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`, the negative source projection
artifact file SHA-256
`f938224a5526b610116c8e01f8cf8491b5f578a64e6dd0fd0a5278f4c4175631`,
and expanded fixture SHA-256
`5189d90dfdbffd1ac004a61d9f8358a0b23e393473e8f3b29b1fe85c8ef857a2`.

The ignored outputs are:

| Local output | Exact size | File SHA-256 |
|---|---:|---|
| `layers-4-6-codimension-rotation.pt` | 6,967,064 bytes | `ef80592808b1eed71e66d1d3eeeeff0596731afa7accfc6b51ac7a444ac78a84` |
| `layers-4-6-codimension-rotation.json` | 360,892 bytes | `8dcbde487f369d0b2a15128e3272c496ed84140b920f9724646041e0a5a1acc6` |

The `.pt` scientific payload digest is
`7a119e9eaf0e4b5c34945411d211953a84ba1f2f6c8ba82372b071b2579f4ae5`.
Its stored canonical JSON-report digest is
`33e0b9ae94d385827e890ebf0d64df9fc1692d6b3b370c013c8519bcae1571d8`;
that canonical digest intentionally differs from the pretty-printed JSON
file's byte hash above. The tensor output is about 6.64 MiB and the JSON
report about 352.4 KiB. They contain derived codec, projector, moments,
aggregate ledgers, and provenance, but no pretrained weights, prompt text,
tokenizer state, executor, or test evaluation.

## Run the true rotated-span grouped executor

The codimension-one result above is a representation result: projecting the
*true* layers 4–6 block delta into one locked rank-639 span preserves behavior.
It does not establish that a smaller graph can generate that delta from the
layer-4 input. This follow-up makes that distinction executable. Its student
path is

```text
native layers 0–3
  -> full-width layer-4 input
  -> one grouped causal executor
  -> native layers 7–17 and LM head
```

Native layers 4, 5, and 6 are absent from the student path. The execution audit
recorded zero calls to each of them and zero source-block calls in total.

Let \(q\in\mathbb{R}^{640}\) be the locked unit normal from the successful
codimension-one artifact. The runner constructs
\(B\in\mathbb{R}^{640\times639}\) deterministically in FP64: a Householder
reflection maps the largest-magnitude coordinate vector to the canonicalized
normal \(q\), and removing that coordinate's reflected column yields

\[
B^{\mathsf T}B=I_{639},
\qquad
BB^{\mathsf T}=I_{640}-qq^{\mathsf T}.
\]

The full 640-dimensional boundary remains the executor input because the
predecessor justified only an output constraint. At each position the
replacement computes

\[
x_t=\frac{h_t-\mu}{\sigma},
\qquad
z_t=G_\theta(x_{\leq t})\in\mathbb{R}^{639},
\qquad
\widehat h_t=h_t+Bz_t.
\]

The grouped causal graph \(G_\theta\) has a learned 640-to-639 same-position
affine path plus two shared rank-16 positive-lag experts. A tanh router of
width 16 conditions those experts on query, key, and relative lag. Positive
lag edges read only earlier logical positions; padding and future positions
are masked. There is no identity skip in modal space because the input and
output widths differ. The fixed basis decode enforces
\(q^{\mathsf T}(\widehat h_t-h_t)=0\) by construction, rather than with a loss
penalty.

The completed developer-local run was:

```bash
fisher-graph-gemma-rotated-span-executor \
  --rotation-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-codimension-rotation.pt \
  --model-id google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_rotated_span_executor_prompts.json \
  --max-length 128 \
  --tokenization-batch-size 2 \
  --expert-count 2 \
  --expert-rank 16 \
  --router-width 16 \
  --modal-warmup-steps 100 \
  --modal-warmup-learning-rate 0.001 \
  --train-steps 64 \
  --train-positions-per-sequence 4 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --gradient-clip-norm 1.0 \
  --ridge-regularization 0.001 \
  --ground-truth-weight 1.0 \
  --teacher-kl-weight 1.0 \
  --selection-nll-atol 0.05 \
  --selection-top1-min 0.95 \
  --selection-teacher-kl-max 0.05 \
  --selection-p90-abs-nll-max 0.10 \
  --selection-p10-top1-min 0.90 \
  --max-stored-coefficient-ratio 0.75 \
  --max-analytic-mac-ratio 0.75 \
  --seed 7301 \
  --device cpu \
  --dtype float32 \
  --output \
    .local-runs/google--gemma-3-270m/layers-4-6-rotated-span-executor.pt
```

### Frozen training and selection protocol

The prompt fixture is a lexically fresh 64/16/16/16 calibration-A,
calibration-B, validation, and reserved-test split. Its 112 exact prompt
hashes are disjoint from the projection, weighted-Jacobian, gated-executor,
and codimension-rotation fixtures. The four roles deliberately repeat broad
prompt templates, however, so this run measures paraphrase interpolation and
does **not** establish unseen-template-family generalization. A positive
generalization claim would require a wholly new fixture with explicit family
IDs assigned disjointly by role. Only calibration A can update the executor:

1. compute full-width input normalization and ridge-initialize the
   same-position 640-to-639 map from 1,871 valid A positions;
2. run exactly 100 AdamW warm-up steps on projected modal block-delta MSE;
3. run exactly 64 downstream steps on equal-weight ground-truth cross entropy
   plus teacher-logit KL, using four selected positions per sequence;
4. retain the final fixed step, with no early stopping or calibration-B
   checkpoint selection.

Model parameters remain frozen, and no source-model gradients were observed.
The executor passed the future-slot causality and
batched-versus-single-with-the-same-padding structural probes before
selection. The latter removes peer batch rows but does not trim padding, so
it is not yet a variable-length padding-invariance result.

Calibration B then evaluates one frozen executor against five strict gates:
absolute aggregate delta NLL/token at most 0.05, aggregate top-1 agreement at
least 0.95, teacher KL/token at most 0.05, per-prompt p90 absolute delta
NLL/token at most 0.10, and per-prompt p10 top-1 agreement at least 0.90.
Every gate must pass. The accounting gates independently require stored
coefficient and analytic-MAC ratios at most 0.75.

Two controls separate possible failure causes. The **rotated-span oracle**
projects the true native block delta through \(BB^{\mathsf T}\); it is not an
inference executor because it runs the source block. Its purpose is to
reconfirm that the locked span still has behavioral headroom on the
exact-hash-disjoint B
split. The **identity block skip** adds no block delta and tests whether the
suffix can simply tolerate removal of layers 4–6.

### Developer-local rotated-span executor result

Calibration B covered 16 sequences, 440 valid positions, and 424 supervised
tokens:

| Calibration-B candidate | Delta NLL/token | Top-1 agreement | Teacher KL/token | Per-prompt p90 absolute delta NLL | Per-prompt p10 top-1 | Behavior gate |
|---|---:|---:|---:|---:|---:|:---:|
| Grouped executor | +0.0610639914 | 0.653301887 | 0.337807072 | 0.431963603 | 0.523809524 | fail |
| True-delta rotated-span oracle | +0.000659043 | 0.988207547 | 0.000240363 | 0.008938129 | 0.961538462 | pass |
| Identity block skip | +9.725194769 | 0.018867925 | 10.112346343 | 13.342923945 | 0 | fail |

The learned executor failed all five behavior gates. Its direct block-delta
diagnostic was NRMSE 0.060264189 with cosine 0.998185757. The high cosine is
not contradictory: cosine measures aggregate direction, while a roughly 6%
relative delta error can still contain coordinated errors in sensitive
coordinates and positions. Here that apparently small geometric error changed
almost 35% of B next-token argmaxes, raised teacher KL to 0.337807/token, and
produced a wide per-prompt error tail. For this boundary, 6% delta NRMSE is
behaviorally catastrophic.

The oracle is the decisive control. It passed on the same hash-disjoint B
inputs with
delta NLL/token +0.000659043, top-1 agreement 0.988207547, and teacher
KL/token 0.000240363. Therefore this run does **not** overturn the
codimension-one finding. The locked rotated span remains viable when supplied
the correct projected delta; the present grouped generator is not precise
enough to produce it.

The executor nevertheless satisfied the mechanical replacement and span
contracts:

- learned executor parameters: 471,057;
- fixed runtime coefficients for the normal, basis, and input normalization:
  410,880;
- total stored runtime coefficients: 881,937, or 5.27446% of the 16,720,896
  source-block parameters;
- ideal sparse analytic MAC ratio: 5.22358% of the source layers 4–6 estimate
  on the recorded B lengths;
- native source-block calls in the student path: zero;
- relative out-of-span delta energy: \(3.1313\times10^{-18}\).

Those resource gates show that the proposed graph is structurally small, not
that it is a viable compressed replacement. The MAC estimate counts ideal
sparse mathematical operations, excludes normalization, nonlinearities,
softmax, additions, memory traffic, and kernel overhead, and is not a latency
or kernel-speed measurement. More importantly, resource savings are
scientifically irrelevant when the replacement fails behavior.

The current behavioral evaluator also materializes full
batch-by-sequence-by-vocabulary logits and computes KL over them. Its metric
memory is therefore \(O(BSV)\); at Gemma's large vocabulary, longer sequences
should use tokenization batch size 1 until the evaluator projects and reduces
supervised rows in chunks. Before any positive replacement claim, the control
set must additionally replay the exact native block boundary through the
suffix and require numerical identity, rather than relying only on the
near-identity rotated oracle.

Because calibration B failed, the runner stopped. Validation and test were
neither tokenized nor model-evaluated; no result was selected using them.
Consequently the artifact sets fidelity-viable replacement, parameter
reduction, analytic-MAC reduction, validation success, and latency/kernel
speed claims to false. This is a useful negative executor result, not a viable
compression result.

This calibration B is now consumed. Do not tune the architecture or loss and
rerun against it: any follow-up generator needs a new B/validation/test
fixture, with prompt-family IDs kept wholly disjoint across roles. The runner
also refuses to overwrite an existing tensor or JSON output.

The ignored outputs and bindings are:

| Local output | Exact size | File SHA-256 |
|---|---:|---|
| `layers-4-6-rotated-span-executor.pt` | 1,998,017 bytes | `991413892ad067fb4b2baea91370607cc6f97720826f64680e1931613d1bf750` |
| `layers-4-6-rotated-span-executor.json` | 237,530 bytes | `a9e2900d4a0025ebb28c9716b00be5cd7448390084eef9c12e90a96f028b3e42` |

Scientific payload SHA-256:
`7acd46ef7c1870c0e04f91cb7bd19e62fe0db8c70d7fd512201e023ef8a9b726`.
Stored canonical report SHA-256:
`115d9b190a0de6af15ccc8befeaa8e0a5244cc919a81fc18e0245b32ee916fc9`.
Executor execution fingerprint:
`7ebccde85d6515ae7ca550b55f14a71c6aa9fbb802dae43299f0808b079409c0`.
The tensor artifact contains the derived executor weights and fixed runtime
state, but no pretrained model weights, prompt text, or tokenizer state.

## Run the Fisher-aware merged-tail supermode oracle

The grouped executor failed to *generate* the viable rotated block delta, but
that does not answer whether the low-ranked part of the viable representation
itself can be merged. This follow-up isolates that question before another
generator is trained.

Let \(T\in\mathbb{R}^{640\times32}\) be the authenticated tail basis from the
successful codimension-one artifact and \(n\in\mathbb{R}^{640}\) its locked
omitted normal. A deterministic
\(R\in\mathbb{R}^{32\times31}\) spans the Euclidean complement of
\(T^{\mathsf T}n\). Fresh calibration A refits two moments inside the 32-wide
tail:

\[
F=\mathbb{E}[g g^{\mathsf T}],
\qquad
C=\mathbb{E}[\delta\delta^{\mathsf T}],
\]

where \(g\) is the downstream pseudo-top-1 score gradient and \(\delta\) is the
native layers 4–6 block delta in tail coordinates. The merge diagonalizes the
unregularized generalized-Fisher problem in the 31-dimensional surviving
space,

\[
F_R=R^{\mathsf T}FR,
\qquad
C_R=R^{\mathsf T}CR,
\]

and obtains a paired encoder/decoder ordered by the eigenvalues of
\(C_R^{1/2}F_RC_R^{1/2}\). For a native block delta \(d\), rank \(q\) preserves
the 608-dimensional complement of \(T\), encodes
\((dT)R\), keeps the first \(q\) generalized-Fisher coordinates, and decodes
them back through \(R^{\mathsf T}T^{\mathsf T}\). Candidate total rank is
\(608+q\).

This makes the endpoints explicit:

- \(q=0\), total rank 608: discard the entire surviving tail;
- \(q=31\), total rank 639: reproduce the authenticated codimension-one span;
- total rank 640: replay the native block boundary as an identity control.

The experiment is a target-informed representation oracle. It runs the native
block to obtain \(d\), and then runs the native suffix for every candidate. It
is not a graph executor and makes no parameter, FLOP, storage, latency, or
kernel-speed claim.

The completed command was:

```bash
fisher-graph-gemma-merged-supermodes \
  --rotation-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-codimension-rotation.pt \
  --model-id google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_merged_supermode_oracle_prompts.json \
  --family-manifest \
    examples/gemma3_merged_supermode_oracle_prompt_families.json \
  --max-length 128 \
  --tokenization-batch-size 2 \
  --supermode-ranks 0,1,2,4,8,16,24,28,30,31 \
  --selection-nll-atol 0.05 \
  --selection-top1-min 0.95 \
  --selection-teacher-kl-max 0.05 \
  --selection-p90-abs-nll-max 0.10 \
  --selection-p10-top1-min 0.90 \
  --identity-nll-atol 0.000001 \
  --minimum-subspace-stability 0.90 \
  --device cpu \
  --dtype float32 \
  --output \
    .local-runs/google--gemma-3-270m/layers-4-6-merged-supermode-oracle.pt
```

### Frozen split and authentication protocol

The fixture contains 64 calibration-A, 16 calibration-B, 16 validation, and 16
reserved-test prompts. A companion manifest assigns eight domain/template
families only to A and four different families to each of B, validation, and
test. No family suffix crosses roles. All 112 exact prompt hashes are disjoint
from every earlier Gemma fixture in the repository. The roles still use
similar broad task forms—short explanations, diagnostics, arithmetic, and
imperative protocols—so this is task-form-matched interpolation, not unseen
task-family generalization.

Fixture SHA-256:
`7076a630a286607f130e8060e62b1f8a17214a2ec8a80bcb1c36cd85e650d938`.
Family-manifest SHA-256:
`25880200ae1974a89a40b5d6c369d2707757b6336a5c8754a2ceafbeaa4a83d1`.

Only A can fit or order supermodes. Its two alternating-sequence halves also
fit independent codecs. A candidate is called stable when the mean squared
canonical correlation between the two retained decoder subspaces is at least
0.90. Calibration B evaluates the frozen rank schedule once against five
behavior gates:

1. absolute aggregate delta NLL/token at most 0.05;
2. aggregate top-1 agreement at least 0.95;
3. teacher KL/token at most 0.05;
4. per-prompt p90 absolute delta NLL/token at most 0.10;
5. per-prompt p10 top-1 agreement at least 0.90.

Before any merged rank can lock, native identity must pass at \(10^{-6}\), and
the \(q=31\) endpoint must agree with the authenticated predecessor within an
absolute \(10^{-5}\) at the block boundary and pass all behavior gates. If
those controls pass, the smallest stable \(q<31\) passing all five B gates is
locked. Validation then evaluates only that candidate. Reserved test is always
parse-, validate-, and hash-only in this experiment.

The merged artifact requires the predecessor tensor as a sibling trust anchor.
Its strict loader verifies that file's SHA-256 through the predecessor's own
strict loader, binds source payload/report/model/block/lock metadata, and
requires bitwise-equal tail-basis and locked-normal endpoints. It also
reconstructs A moments from split halves, recomputes the merge, stability,
spectrum, B ledgers, gates, controls, and lock, and binds every behavior/direct
row to the exact ordered tokenized stream. Adversarial tests re-sign forged
payloads and reports; the loader rejects altered protocol, source, execution,
split moments, row provenance, selection status, and an orthogonal substituted
normal.

### Developer-local merged-supermode result

Calibration A covered 64 sequences, 1,786 valid positions, and 1,722
supervised positions. Both 31-dimensional moments were numerically
full-support: the surviving score-Fisher minimum eigenvalue was
\(2.7061\times10^{-6}\), the delta-moment minimum eigenvalue was 32.0311, and
the full generalized codec's identity residual was
\(3.57\times10^{-15}\).

The spectrum is highly concentrated. One supermode carries 95.3124% of the
factorized weighted objective, 16 carry 98.8437%, 24 carry 99.5705%, and 28
carry 99.8345%. That objective concentration is not behavioral fidelity:
calibration B is still decisive.

Calibration B covered 16 sequences, 454 valid positions, and 438 supervised
positions:

| \(q\) | Total rank | A weighted fraction | A stable | Delta NLL/token | Top-1 | Teacher KL/token | Prompt p90 abs NLL | Prompt p10 top-1 | Five behavior gates |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 0 | 608 | 0 | yes | +0.575058 | 0.563927 | 0.699384 | 0.894724 | 0.458333 | fail |
| 1 | 609 | 0.953124 | yes | -0.008498 | 0.929224 | 0.013213 | 0.076081 | 0.809524 | fail |
| 2 | 610 | 0.962465 | yes | -0.002625 | 0.931507 | 0.012036 | 0.059428 | 0.809524 | fail |
| 4 | 612 | 0.969081 | yes | -0.004244 | 0.920091 | 0.010109 | 0.069243 | 0.809524 | fail |
| 8 | 616 | 0.977254 | no | -0.001431 | 0.940639 | 0.008016 | 0.058382 | 0.821429 | fail |
| 16 | 624 | 0.988437 | no | -0.003917 | 0.952055 | 0.004556 | 0.021937 | 0.857143 | fail |
| 24 | 632 | 0.995705 | no | +0.000777 | 0.954338 | 0.002129 | 0.021742 | 0.892857 | fail |
| 28 | 636 | 0.998345 | yes | +0.001807 | 0.977169 | 0.000958 | 0.016726 | 0.925926 | **pass** |
| 30 | 638 | 0.999498 | yes | +0.001611 | 0.979452 | 0.000440 | 0.011514 | 0.928571 | **pass** |
| 31 | 639 | 1.000000 | yes | +0.001422 | 0.988584 | 0.000242 | 0.010249 | 0.958333 | **pass** |
| identity | 640 | — | yes | 0 | 1.000000 | 0 | 0 | 1.000000 | **pass** |

This is substantially stronger than the rank-608 prefix-only endpoint in this
sweep and is suggestive relative to the earlier prefix experiments, which used
different B fixtures and therefore are not controlled same-split baselines.
Rank 636 was the smallest candidate satisfying the A stability rule and all
five B behavior gates. Rank 624 already passed aggregate top-1, but its
per-prompt p10 was only 0.857143 and it was unstable. Rank 632 was a near miss
at p10 0.892857 and was also unstable. The results show why aggregate NLL or
weighted-energy retention alone is insufficient.

There is a second stability warning. The preregistered gate uses the *mean*
squared canonical correlation. At \(q=28\) that mean was 0.929272, but the
minimum canonical correlation was only 0.095594; at \(q=30\), the corresponding
values were 0.966673 and 0.013625. Most of each large subspace is stable while
at least one direction is not. A future protocol should preregister both mean
and worst-direction criteria rather than interpreting the current boolean as
uniform subspace stability.

### Why the run still failed closed

No merged candidate was locked and validation was not touched, because the
rank-639 endpoint-equivalence control failed:

| Control | Recorded result | Required | Outcome |
|---|---:|---:|:---:|
| Native identity maximum logit error | 0 | \(\le 10^{-6}\) plus behavior gates | pass |
| Rank-639 behavior | all five gates pass | all five gates pass | pass |
| Rank-639 versus predecessor maximum boundary error | 0.00048828125 | \(\le 10^{-5}\) | **fail** |

The failure is numerical, not a different subspace. The authenticated endpoint
normal and tail basis are identical, and both operations represent
\(d-(d^{\mathsf T}n)n\). The source projector evaluated that expression
directly. The original full-rank merge passed through tail projection,
31-dimensional encode/decode, and tail reconstruction in float32. On these
large boundaries that algebraically redundant chain accumulated one-ULP-scale
error. The recorded boundary-output RMS was 437.254, so the maximum absolute
error was about \(1.12\times10^{-6}\) of that RMS, but the preregistered control
was absolute and therefore correctly failed.

After preserving the completed artifact, the implementation was fixed so
\(q=31\) dispatches the exact authenticated one-normal formula. A regression
uses dense 640-wide float32 values in the thousands and requires
`torch.equal` for both delta and masked-output projection. This prevents the
same false endpoint failure in a future run. It does **not** retroactively
change the B decision, and the consumed B split must not be rerun.
Projection semantics are state-versioned: this completed format-1 merge still
strict-loads and replays the historical factorized endpoint that produced the
recorded 0.00048828125 audit, while new format-2 merges declare and use
`authenticated_one_normal`. Strict loading therefore does not silently migrate
the immutable artifact to post-run execution semantics.

The scientific conclusion is therefore narrower than the tempting row in the
table:

- calibration B provides promising evidence that the 31-coordinate surviving
  tail can be represented by 28 generalized-Fisher supermodes;
- the preregistered experiment did not validate that candidate because a
  required control failed;
- rank 636 removes only four of 640 representation dimensions, or 0.625%;
- the oracle still computes the native block delta, so even a validated
  positive result would not itself reduce parameters, FLOPs, storage, or
  latency.

A confirmatory run needs a wholly fresh family-disjoint selection/validation/
test protocol, the corrected bitwise endpoint preflight at representative
activation magnitudes, and a preregistered worst-direction stability rule.
Only after that representation result validates would it make sense to train a
generator for the locked merged coordinates.

Validation and test from this run were never tokenized or model-evaluated. The
strict-loaded artifact records selection failure with reason `controls_failed`
and sets every compression, parameter, analytic-MAC, latency, and
merged-representation viability claim to false.

The ignored outputs are:

| Local output | Exact size | File SHA-256 |
|---|---:|---|
| `layers-4-6-merged-supermode-oracle.pt` | 506,237 bytes | `0da63f157e39fb9f7dc2ca5472cec5d45e9c40c75d79e6cde3b8cd4fcea62157` |
| `layers-4-6-merged-supermode-oracle.json` | 803,877 bytes | `fb0d339b540b3cd3957e30c2e57111c6ff6a539095bb04c2c34c7216d7918655` |

Scientific payload SHA-256:
`dfed3c5f3b48bbf9f46f5a69a7896de4c4f780d6842309ba97f600b768dd6647`.
Stored canonical report SHA-256:
`20546c1a129537a275a88e714c897dfd52a5ff2f9b1471d8317e6d9ef8e924c3`.
The tensor artifact contains derived moments, the merged codec, scalar
evaluation ledgers, and provenance, but no pretrained weights, prompt text, or
tokenizer state.

## Useful command variations

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

The weighted-Jacobian artifact adds calibration-A activation covariance and
all codec states, the calibration-B candidate ledger and deterministic lock,
every family's calibration-B full-rank control, the locked validation result
with locked-family and native identities, and optional bounded
forward-JVP/factor states. It SHA-binds the JVP endpoints to the exact locked
input/output codecs. Its strict loader regenerates every codec from saved
Fisher/covariance state, recomputes selection, all identity gates, per-lag
probe accounting, the synthetic weighted factor, and the sibling JSON report.
The report labels the projected modal slice and factor's dense denominator and
explicitly disables full-Jacobian, behavioral, variable-length, compression,
and runtime claims.

The gated-executor artifact adds all predeclared fitted candidate states, the
calibration-B gate ledger and deterministic lock, the one locked validation
evaluation, direct block-output and downstream behavioral metrics,
per-example/per-length tails, rank-conditioned target-informed references,
resource accounting, identity controls, and frozen-model/prompt provenance.
Its strict loader reconstructs each graph from weights-only state, recomputes
the selection and validation gates, and cross-checks the sibling JSON. It
contains codec and executor state but no source-model weights, tokenizer
state, prompt text, or reserved-test evaluation.

The projection-ladder artifact strict-binds both of those predecessors, the
fresh prompt hashes, the nested rank schedule, the calibration-B direct and
behavior curves, full-width controls, deterministic identity fallback, and
one locked validation evaluation. Its loader recomputes every aggregate from
per-example evidence, the nested-error condition, behavior gates, lock, and
scientific status before cross-checking the sibling JSON. It contains the
derived output codec but no executor state, source-model weights, tokenizer
state, prompt text, or reserved-test evaluation.

The rotated-span executor artifact reconstructs the graph and deterministic
basis, verifies its fingerprint, and rebinds the model, block geometry, and
executor architecture. Its strict loader rebuilds behavior and direct
aggregates from per-example evidence; recomputes B and conditional validation
gates, status, span/structural checks, and resource arithmetic; binds every
tokenized stream to the exact prompt-hash role; and cross-checks the sibling
JSON report. The source parameter and MAC denominators are provenance-bound
and arithmetically checked offline, not independently reproduced without
loading the source model. The artifact contains executor state but no source
weights, prompt text, tokenizer state, or reserved-test evaluation.

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
keep-top-\(k\) sufficiency curve with a full-rank identity gate. The
weighted-Jacobian CLI now adds split-safe activation-aware codec selection,
true bounded forward JVPs, and a signed weighted-factor reference. The gated
CLI has now executed the proposed residual-separated, state-conditioned graph
against held-out block outputs and downstream behavior. The projection and
codimension-one CLIs separate prefix ordering from arbitrary hyperplane
viability, and the rotated-span CLI now executes a source-independent grouped
replacement constrained to the successful hyperplane.

The bundled diagnostic fixture is still not the representative evidence
needed to pass the compilation gate. It is short and template-matched, its
original Fisher bases were inconclusive at the decision ranks, and its
rank-128 reverse-lag ridge maps overfit calibration without a held-out
cross-position gain. The strict-loaded generalized codec did produce the
first positive locked result—joint rank 636 remained within the same aggregate
NLL/top-1 bounds on validation—but it removes only 0.625% of each residual
width and depends on explicit floors inside an A-only Fisher nullspace. The
forward edge pilot covers only four modes and four calibration prompts. The
follow-up gated executor did execute against fresh held-out block outputs, but
every rank-320/480 candidate failed selection and the locked diagnostic
fallback failed validation badly. Even the target-informed rank-320
least-squares reference preserved raw block-delta energy while destroying
behavior, showing that Euclidean block MSE is the wrong acceptance target for
that subspace. The later codimension-one rotation found a rank-639 hyperplane
whose true-delta projection remained behaviorally viable. The grouped
replacement then skipped all three native layers and met its coefficient,
analytic-MAC, causality, and span contracts, but failed calibration B despite
the true-delta oracle passing there. That isolates the present failure to
generator precision rather than the locked output span. Validation and test
were not tokenized after that B failure. The later merged-tail oracle found
that ranks 636 and 638 pass all five behavior gates on a wholly new prompt- and
domain-family-disjoint, task-form-matched B split, which is the first evidence
that more than the single rotated direction can be removed. Its required
rank-639 endpoint control then failed on an absolute float32 roundoff threshold
before validation, so the result remains unvalidated and removes only 4/640
dimensions at best. No viable graph executor can yet replace this layer range.

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
5. rerun the generalized-Fisher family with predeclared absolute or
   scale-normalized floors and enough calibration rows to reduce the
   width-640 nullspace; require rank/floor stability, full-rank identity,
   aggregate gates, and new per-example/per-length bounds;
6. repeat the merged-tail oracle on wholly fresh family-disjoint splits with
   its now-bitwise rank-639 endpoint, and preregister both mean and minimum
   canonical-correlation stability gates; require rank 636 or lower to validate
   before treating merged coordinates as a generator target;
7. fit with a combined local and downstream-logit or KL objective, include an
   explicit same-position-only baseline, and attribute any improvement
   specifically to positive-lag routing before widening expert capacity;
8. require local boundary, internal modal, end-to-end NLL, sequence-length,
   causal-leakage, fallback, storage, arithmetic, and latency gates before
   replacing that block in the mixed runtime;
9. lock the rank, executor, and thresholds, then model-evaluate the reserved
   test split exactly once without further selection.
