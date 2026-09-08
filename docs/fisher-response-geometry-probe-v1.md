# Fisher response geometry probe v1

`fisher_response_geometry_probe_v1` is a bounded, synthetic qualification of
response-based charts. Its question is whether regions with similar aligned
derivative actions predict perturbations and residual updates better than
equal-capacity activation-only charts and a global affine model. Synthetic
results qualify the mathematics and implementation; they do not establish
that useful response charts exist in Gemma.

The current target is the normalization response after one attention message
has been formed. Live attention routing is a separate future experiment. This
diagnostic does not change model weights, install an operator, add recurrence,
modify the compiler, or authorize a compression, speed, or fluid-dynamics
claim. The terms “rotation” and “gain” below refer only to specified linear
maps and metrics.

## Execution and data boundary

The model-free entry point is:

```bash
python src/fisher_response_geometry_probe_v1.py synthetic --output /tmp/fisher-response-geometry-v1-001
```

Use a fresh output directory. The model-facing branch is a closed gate: this
runner has no live model, tokenizer, corpus reader, or capture capability.
An authorization flag is not a substitute for independent data or a separately
reviewed future runner.

`python src/fisher_response_geometry_probe_v1.py inspect` inspects the frozen
configuration without importing the tensor runtime. The `model-facing`
subcommand refuses execution with a nonzero exit status before tensor import
or data access. The standalone runner loads the numerical module directly,
without initializing the repository's broader package entry points.

The current corpus disposition remains exactly
`development_complete_independent_data_unavailable`. The existing
[P0 access boundary](fisher-compact-message-completion-probe-v2.md#current-p0-contract)
and its strict metadata-only eligibility policy remain authoritative. This
probe cannot reopen the exhausted reserve, reuse an exposed or rejected panel,
inspect protected corpus payloads, or turn synthetic qualification into a
scientific verdict. A future independent-data campaign requires new eligible
data, its own campaign identity, frozen decisions and separately authorized
model-facing execution. It cannot resume or rewrite the preserved P0 attempt.

The numerical implementation is
[`fisher_response_geometry.py`](../src/fisher_graph/fisher_response_geometry.py).
The standalone runner is
[`fisher_response_geometry_probe_v1.py`](../src/fisher_response_geometry_probe_v1.py).
This document specifies the mathematics and scope. The runner's frozen
configuration and report schema specify numeric ranks, sample sizes, seeds,
radii, tolerances and decision thresholds. Actual commands and test results
must be recorded from execution; no model result is implied by this contract.

The frozen profile is
[`examples/fisher_response_geometry_probe_v1.json`](../examples/fisher_response_geometry_probe_v1.json),
the model-free gate and validation rules are in
[`fisher_response_geometry_protocol.py`](../src/fisher_graph/fisher_response_geometry_protocol.py),
and the report contract is
[`fisher_response_geometry_probe_v1_report.schema.json`](schemas/fisher_response_geometry_probe_v1_report.schema.json).
The output contains `frozen_config.json`, `report.json` and
`artifact_manifest.json`. The manifest records hashes and byte counts for the
config/report and an explicit allowlist of implementation and contract source
files. Its domain-separated canonical configuration hash is reproducibility
bookkeeping; it is not corpus independence evidence or campaign authorization.

## One declared mutation and its tensors

The existing
[canonical capture sites](../src/fisher_graph/gemma3_fisher_local_chart_capture.py)
and [layer-4 contract](fisher-local-chart-probe.md#exact-gemma-3-layer-4-boundary)
identify this teacher computation in the pinned `google/gemma-3-270m` model.
Vectors below use column notation; repository tensors have trailing feature
axes and use `F.linear(m, W_O)`.

| Symbol | Repository boundary | Sequence shape | One valid target token |
| --- | --- | --- | --- |
| `h` | `layer.4.input`, before attention input RMSNorm | `[batch, length, 640]` | `h_t` in `R^640` |
| `m` | `layer.4.attention.context`, after attention mixing and before `o_proj` | `[batch, length, 1024]` | Four concatenated 256-wide head contexts |
| `q` | `layer.4.attention.operator_output` | `[batch, length, 640]` | `q_t = W_O m_t` |
| `Phi` | `layer.4.attention.delta` | `[batch, length, 640]` | `N_A(q_t)` |
| `T` | `layer.4.post_attention` | `[batch, length, 640]` | `h_t + Phi(h_t,m_t)` |

`W_O` has shape `[640, 1024]` and the pinned projection is bias-free. The
attention message contains four query-head contexts; the upstream attention
uses one key/value head with grouped-query repetition. The native attention
grammar and dimension checks are implemented by
[`Gemma3AttentionGenerator`](../src/fisher_graph/gemma3_attention_generator.py).
The pinned architecture records four query heads, one key/value head,
256-wide heads, Q/K RMSNorm, float32 softmax and no attention logit softcap in
[`gemma3_layer_agnostic_whole_model_compression_protocol.py`](../src/fisher_graph/gemma3_layer_agnostic_whole_model_compression_protocol.py).
Those architecture references do not extend this probe beyond attention.

For fixed model coefficients and one valid row:

```text
T(h, m) = h + Phi(h, m)
Phi(h, m) = N_A(W_O m)
N_A(q) = diag(1 + w_A) q / sqrt(mean(q^2) + epsilon)
```

Here `w_A` has width 640 and the pinned epsilon is `1e-6`. Native Gemma
computes RMSNorm in float32 and casts its result once to the input dtype.
The float64 mathematical reference evaluates the same real-valued formula in
float64. These are separately labeled numeric paths, not a claim of bitwise
native model equivalence. Current generated-input cast checks use projection
in the input dtype, normalization in fp32, multiplication by `1+w` after
normalization, and the final cast back to the projection dtype. Autodiff
differentiates this arithmetic graph through casts; this is distinct from
finite differences of the quantized function at arbitrarily small steps.

The frozen float64 analytic-JVP tolerances are `atol=1e-10`, `rtol=1e-9`;
the radial identity uses `atol=1e-10`. The entire declared finite-difference
sweep is reported, with at least one float64 relative error at most `1e-6`
required. Separately declared synthetic cast-JVP tolerances are:

| Projection/output dtype | JVP absolute tolerance | JVP relative tolerance |
| --- | ---: | ---: |
| fp32 | `2e-5` | `2e-4` |
| fp16 | `0.02` | `0.03` |
| bfloat16 | `0.15` | `0.15` |

These compare the cast computation with float64 mathematics on the same
quantized inputs and weights. Native-dtype finite differences remain a
reported step sweep, because rounding can dominate their smallest steps.
They supply no target-model qualification. The independent unit tests also
check the actual fp32-normalization multiplication order exactly on fixtures.

The pure mathematical fixtures keep `h` and `m` independent. For a message
intervention, hold `h`, coefficients, epsilon, token identity and sequence
context fixed and vary only the declared message coordinates. The existing
cache-free prefill contract fixes query/key masks, logical positions, sequence
lengths and ordering, RoPE settings and attention window. Layer 4 is sliding
causal with window 512 and RoPE theta 10,000. A future execution must retain
valid-row masking, causal support and the original sequence context; it must
not infer unrestricted attention from flattened token rows. Current analysis
uses valid rows only. Padding is outside the sampled state distribution.

This mutation ends at `layer.4.post_attention`. Historical `X4` is the later
pre-feedforward-normalized state and historical `H4` is `layer.4.output` after
the MLP residual. Neither is an alias for this mutation's output. The existing
all-stage `capture_layer4_sequence` also observes MLP sites and differentiates
the suffix; it must not be called implicitly by this attention-only probe.

## Partial responses, offsets and the normalization reference

With independent state and message, the partial derivatives are:

```text
K(h, m) = partial Phi / partial h = 0
R(h, m) = partial Phi / partial m = J_N(W_O m) W_O
J_fixed = partial T / partial h with m fixed = I
```

Writing `d=640`, `q=W_O m`, `s=sqrt(q^T q/d + epsilon)` and
`L=diag(1+w_A)` gives the exact real-valued derivative:

```text
J_N(q) = (L / s) [I - q q^T / (d s^2)]
R(m) v = (L / s) [W_O v - q (q^T W_O v) / (d s^2)]
R(m) m = (epsilon / s^2) Phi(m)
```

Equivalently, in the requested notation `y=W_O m` and
`Gamma=diag(1+weight)`:

```text
R(m) = Gamma [I/s - yy^T/(d s^3)] W_O
R(m) m = epsilon Gamma y / s^3
```

The projection-only submap `m -> q` has constant Jacobian `W_O`. Including
post-attention RMSNorm makes the response depend on `m` through its projected
norm and direction. It does not introduce a separate dependence on `h` or
measure upstream routing. A successful current chart comparison must
therefore be reported as normalization-response qualification. The analytic
RMSNorm reference is an accuracy oracle and a check that the charts are not
being credited with discovering unknown teacher structure.

The radial identity is also a strong offset check. At zero epsilon the radial
derivative vanishes, although the residual mutation need not vanish. At the
pinned positive epsilon the radial response is small when `s^2` is large
relative to epsilon. The uncentered approximation `Phi(m) = R(m) m` is thus
incorrect even when the derivative itself is exact.

The required anchored affine form is:

```text
Phi(h, m) ~= b_C + K_C (h - h_C) + R_C (m - m_C)
b_C = Phi(h_C, m_C)
```

Current normalization fixtures must recover `K_C=0`. Generic bilinear and
affine fixtures additionally exercise nonzero state derivatives, the state
offset, and the distinction between a derivative operator and a residual
predictor. An approximate offset is allowed only when its error is measured
separately from response projection, operator approximation and Taylor error.

The different-offset adversary is `Phi(h,m)=m+3h-h^3` at `h=-1` and `h=1`.
Both points have `K=0` and `R=1`, but their updates differ by four. A chart
anchored at the first point must reject the second under its frozen support
radius. Support examines the full state and message displacements, including
directions omitted by `U` and `Q`; projected coordinates alone cannot establish
locality.

`SharedGemmaRMSNormScale.from_total` in
[`gemma3_fisher_local_chart_capture.py`](../src/fisher_graph/gemma3_fisher_local_chart_capture.py)
deliberately detaches the complete branch output to make a fixed realized
scale for additive component reconstruction. Its `gemma_rmsnorm` wrapper is a
value-reconstruction helper, not the derivative function for this diagnostic.
Differentiating it omits the rank-one normalization term above. It is left
unchanged. The exact attention generator's `_rms_norm` retains the graph and
provides the existing native arithmetic convention for later qualification.

## Shared projection and coefficient coordinates

Freeze the output metric `F_*` and every basis from training data only. Let
`G` have shape `[d,k]`, `U` have shape `[d,s]`, and `Q` have shape `[p,r]`,
with `p=1024` for the declared message. `U^T U=I` and `Q^T Q=I`. No chart may
fit an independently rotated basis. The output coordinate map is:

```text
H_G = G^T F_* G
D_G = H_G^dagger G^T F_*
P_G = G D_G
```

Prefer `H_G=I`. A singular metric is a seminorm: directions in its nullspace
have zero Fisher cost, which does not mean they are absent or harmless.
Restrict Fisher interpretations to the preregistered supported subspace, or
use the frozen regularization rule there. Never silently invert `F_*` or
inflate its nullspace into measured support. Verify basis independence,
conditioning, `D_G G=I` on supported coordinates and projector idempotence.
Always report ordinary Euclidean error alongside Fisher-weighted error.

The existing
[rank-64 carrier](fisher-local-chart-probe.md#shared-rank-64-output-fisher-carrier)
is a truncated empirical activation-space Fisher/VJP metric at
`layer.4.output`. It is not a full 640-dimensional model Fisher. Using it for
the earlier attention residual would supply a common downstream-sensitivity
surrogate, not the exact local Fisher at that seam. Its original
training/provenance restrictions still apply; current synthetic commands do
not open the carrier. Synthetic PSD metrics are labeled synthetic metrics.
Neither scalar row salience weights nor parameter-space curvature may be
substituted and described as a full output activation metric.

For each chart:

```text
beta_C = D_G b_C
B_C = D_G K_C U
A_C = D_G R_C Q
z = U^T (h - h_C)
a = Q^T (m - m_C)
alpha = beta_C + B_C z + A_C a
Delta_h_hat = G alpha
h_next_hat = h + Delta_h_hat
```

`G` contains analysis/reconstruction directions. An executable repository
modal generator additionally owns an operator, its invocation and boundary
contracts, sequence semantics and artifact provenance. A column of `G` is
not itself such a generator, and this coefficient calculation does not install
one.

Response signatures use the shared-coordinate `A_i` and `B_i`. Distances must
respect the declared output coefficient metric `H_G` and the input metrics;
raw matrix entries in arbitrary nonorthonormal coordinates are not invariant
response comparisons. Equivalent sign changes and orthogonal basis changes
must preserve reconstructed predictions, errors and aligned comparisons.
Any relative scaling between state and message signature blocks is frozen
before evaluation. The normalization target has a zero state-response block.

Record output response outside `span(G)`, input residuals outside `span(U)`
and `span(Q)`, and independent direction probes that did not fit these bases.
A small projected matrix cannot establish low rank of the full response.
Chart membership also needs a training-derived support rule around the
anchor. Report occupancy, rejection and nearest-support diagnostics; do not
use response similarity to imply unlimited finite-radius validity.

## Gains, mixing and distinct metrics

For a square residual-to-residual derivative `J`, define the square coefficient
map `C_J=D_G JG` and leakage `JG-G C_J`. This is distinct from the generally
rectangular chart matrix `B_C=D_G K_C U`. If `H_G` is not the identity, compute
gains from the whitened supported map:

```text
C_white = H_G^(1/2) C_J H_G^(-1/2)
C_white = U_s Sigma V_s^T
O_white = U_s V_s^T
P_white = V_s Sigma V_s^T
C_white = O_white P_white
```

SVDs and matrix powers operate on the supported positive-definite coordinate
space. The singular values are directional gains for the specified metric.
`O_white` is orthogonal mixing in whitened coordinates and may include a
reflection; its action in original coefficient coordinates is conjugated by
the whitening maps. Rank-deficient maps have nonunique orthogonal extensions,
so a reported polar factor is not a uniquely identified physical rotation.
For `J_fixed=I`, the supported projected map is the identity and leakage is
zero, up to numerical tolerance. This is a qualification check, not a dynamic
finding.

Rectangular state or message responses require separate declared input and
output metrics. With coefficient map `A`, compare the supported singular
values of `H_out^(1/2) A H_in^(-1/2)`. In the default Euclidean input convention,
`H_in=I` because `U` or `Q` is orthonormal, while `H_out=H_G`. Singular input
or output metrics require explicit support restriction. A rectangular SVD
does not define a square residual mixing operator. Its gains also cannot be
compared across unit or metric changes without their coordinate maps.
The implementation returns supported eigen-coordinate whitening, which is
orthogonally equivalent to symmetric square-root whitening. A rectangular or
distinct-metric map is labeled a partial isometry and receives no reflection
claim. A null input with measurable output makes finite full-seminorm gain
undefined; supported gains alone do not conceal this case. Metric symmetry
and null-response checks use relative tolerances invariant to global metric
scaling.

Off-diagonal coefficients are not all rotation. Fisher basis directions are
not Jacobian eigenvectors. Eigenvalues alone miss non-normal amplification,
and symmetric/skew parts of a finite-step map do not establish fluid strain,
vorticity, a nonlinear manifold or stable recurrence.

## Separate future attention dynamics

If a future experiment recomputes the attention message from the sequence
state, write `m=M(H; context)` and:

```text
P = D_H M
J_total = I + R P
```

The general independent-input formula is `I+K+R P`; `K=0` at the seam above.
`P` includes changes in both routing probabilities and values, along with
input RMSNorm, Q/K/V projections, Q/K RMSNorm and RoPE. A control that freezes
attention probabilities while recomputing values measures the value path
only. It is useful as a labeled control and cannot stand for the total
attention derivative. Freezing the completed message instead yields
`J_fixed=I`.

For one sequence of length `L`, the full state derivative has shape
`[L*d,L*d]`. A target-token block is `J_tt=S_t J_total E_t`, where `E_t`
injects a direction only at input token `t` and `S_t` selects output token
`t`. All other input residual rows stay fixed, but attention must still
recompute the target's query, key and value effects. Its effects on later
output tokens are absent from `J_tt` and require separate cross-token
accounting. A target-token block must not be described as the full-sequence
map. Basis dimensions and leakage definitions must match the selected map.

The exact generator's `forward_delta` and existing
[`torch.func.jvp` suffix conventions](../src/fisher_graph/gemma3_l3_l4_h4_suffix_jvp_runtime.py)
provide implementation references for a later runner. The suffix runtime has
a different derivative target and is not reused as though it measured this
attention mutation. Such future work remains outside the current closed
runner.

## Smallest falsifiable comparison and reporting

The frozen synthetic profile uses these settings:

| Setting | Value |
| --- | --- |
| Seed | 1729 |
| Synthetic residual/message widths | `d=6`, `p=8` |
| Shared output/state/message basis ranks | `k=3`, `s=2`, `r=3` |
| Chart count for charted arms | 3 |
| Independent training/evaluation families | 12 / 6 |
| Rows per family | 8 |
| RMSNorm epsilon | `1e-6` |
| Central finite-difference steps | `1e-2, 1e-3, 1e-4, 1e-5, 1e-6` |
| Finite perturbation radii | `0.001, 0.01, 0.1` |
| Supported-metric relative tolerance | `1e-12` |
| Support radius multiplier | `1.5` |
| Paired bootstrap replicates | 256 |
| Maximum K-means iterations | 32 |

The small synthetic widths exercise the actual mathematical seam without
pretending to be captured 640/1024-wide Gemma activations. Use the same
synthetic ranks, data partition, chart count for charted arms, local affine
family, support policy and declared fit budget for:

1. One global anchored affine response model, with an additional global
   least-squares affine control to avoid relying on a weak single-anchor
   comparator.
2. Activation-only charts with the same local affine response model.
3. Response charts discovered from aligned projected derivative signatures.
4. Shuffled assignments with comparable training occupancy.

The global arm has one chart by definition; store and report its smaller
parameter count. Feature availability and fit-budget differences must be
explicit. Every chart anchor, metric, basis, router, support threshold and
hyperparameter is training-derived. Synthetic test partitions have independent
generated sampling units. A future corpus campaign must split and resample
at the protocol's independent prompt/family unit, never treat correlated token
rows as independent evidence, and register its acceptance rules before final
evaluation access.

Current Taylor banks use training medoids as anchors, their exact analytic
offsets `b_C`, `A_C=D_G R(anchor) Q`, and `B_C=0`. This keeps offset
approximation error exactly zero for these banks; the report still distinguishes
that component from projected-offset loss and finite-update error. A fitted
global least-squares control is identified separately from a derivative bank.
Activation clustering and the inexpensive routers share the same observable
features `[h/h_scale, m/m_scale]`, with both scales fitted on training data.
Response discovery clusters aligned `A_i` signatures; its zero `B_i` block
carries no extra information. Each discovered label's router is its centroid
in that same activation feature space.

Shuffled training labels are a permutation of the response labels, preserving
training occupancy exactly. Random evaluation labels are an explicitly
labeled diagnostic; the shuffled arm also has its own frozen activation
router. Support uses training-derived full-state/message distances and radii.
Reports keep unfiltered extrapolation error separate from support-accepted
error and coverage. Independent response probes also measure directions not
used to choose `Q` or `G`.

Label exact per-state derivatives and exact analytic normalization as oracle
references. Response-oracle assignments may establish whether useful groups
exist. They cannot establish cheap runtime routing. Separately evaluate the
frozen router using only the declared available activation features, against
oracle assignments and held-out residual prediction. It must not receive
teacher Jacobians, target residuals or evaluation labels. A router using the
completed message depends on the original attention computation; account for
that cost. An activation baseline given less message information than the
response arm measures that information difference as well as chart geometry.

Runtime numeric storage counts shared bases once and includes every anchor,
offset, operator, radius, router center and feature scale. Shared bases retain
one storage allocation across charts and are read-only by convention. The
report separately records the metric factor, output coordinate map and
response-discovery centers used offline. Tensor payload bytes exclude Python
object overhead, peak workspace and allocator costs; actual report/config file
bytes are in the artifact manifest. Arithmetic counts include coefficient
projection/reconstruction, source-feature distances, support checks and the
common residual skip addition. The completed attention message is still a
required input whose real cost is unmeasured. Single-pass synthetic CPU timings
include the router/support path and all exact/fitted comparators, and cannot
support hardware speed or compression claims.

Reports separate these questions:

| Question | Required evidence |
| --- | --- |
| Derivative correctness | Analytic/dense/JVP agreement; central finite differences across a step-size sweep; declared precision |
| Response-subspace adequacy | Fisher and Euclidean output leakage; omitted input directions; independent probe directions |
| Chart stability | Occupancy, within-chart signature variation, support/rejection rates and frozen assignment behavior |
| Finite-update fidelity | Projected derivative error, multiple-radius Taylor error, full residual-update error and separate offset approximation error |
| Router feasibility | Observable features, routed versus oracle prediction, assignment behavior and all required teacher work |
| Runtime and storage | Metric/bases, anchors/offsets, operators, router and support storage; arithmetic and prerequisite computation |

Predictions must improve held-out residual/response errors over activation-only
and global baselines under frozen criteria to support even a bounded useful
chart finding. Report paired uncertainty at the independent sampling unit.
Cluster separation, retained Fisher energy, an interesting polar factor or an
accurate infinitesimal derivative alone does not satisfy that criterion.
Synthetic numerical pass/fail is never an independent-model acceptance result.
The current synthetic mechanical qualification does not require response
charts to win. It requires correct arithmetic, comparisons and disclosures;
an unfavorable comparison is retained as a measured synthetic outcome.

Deterministic qualification covers constant affine and state-conditioned
bilinear maps, identity, orthogonal rotation, anisotropic scale, shear,
singular Fisher support, equivalent bases, substantial out-of-span response,
the normalization radial identity and fixed-message versus recomputed-message
chain rules. Guard tests must show the synthetic route cannot load a model,
tokenize or open protected data, and that the model-facing gate fails closed.
Finite differences are a complementary precision-sensitive check, not an
unqualified ground truth. Negative outcomes, including trivial normalization
structure, projection-dominated error, poor finite-radius fidelity or an
unusable router, are valid results.

## Execution record

Development verification on 2026-09-08 used:

```bash
.venv/bin/python -m pytest -q tests/test_fisher_response_geometry.py tests/test_fisher_response_geometry_cli.py tests/test_fisher_local_chart.py tests/test_gemma3_fisher_local_chart_capture.py
.venv/bin/python src/fisher_response_geometry_probe_v1.py model-facing
.venv/bin/python src/fisher_response_geometry_probe_v1.py synthetic --output artifacts/research/fisher-response-geometry-probe-v1-synthetic
```

The focused and existing model-free regression selection passed **83 tests**.
The model-facing check exited **2**, as required, with
`development_complete_independent_data_unavailable`. CLI subprocess tests deny
model/package imports and protected file reads; the integrated synthetic run
passes under those guards. Invalid flags, stale output paths, inconsistent
numerical pass flags and unsupported evidence claims are rejected. Numerical
failure returns nonzero after preserving a valid diagnostic report; malformed
measurements cannot publish a successful report.

The frozen generated fixture gives these numerical results:

| Check | Measured value |
| --- | ---: |
| Float64 analytic versus autodiff JVP, maximum absolute error | `5.55e-17` |
| Radial identity, maximum absolute error | `2.11e-16` |
| Best relative central-difference error across the full sweep | `2.21e-11` at step `1e-5` |
| True radial response norm | `1.094e-5` |
| Adversarial frozen-scale radial response norm | `2.893` |

All nine registered derivative/cast gates passed. The exact normalization
reference has zero reconstruction error by construction. Relative
Fisher-weighted residual-update errors on all 48 generated evaluation rows are:

| Method | Relative error |
| --- | ---: |
| Exact analytic normalization | `0` |
| Exact normalization projected into shared `G` | `0.02955` |
| Activation charts with frozen activation router | `0.13629` |
| Global fitted least-squares affine control | `0.17912` |
| Shuffled charts with frozen activation router | `0.32591` |
| Global anchored Taylor-affine control | `0.88944` |
| Response charts with frozen activation router | `0.89266` |

These all-row errors include labeled diagnostic extrapolations. Response
charts reject `10/48` rows (`20.83%`) on support; the report separately retains
accepted-row error, oracle-assignment error and normalized support distances.
The response-minus-activation mean squared Fisher error is `+55.69`, with a
paired generated-family bootstrap interval `[+9.66, +102.35]`. It is an
unfavorable result in this fixture, not a finding about Gemma or attention.
Independent message probes also expose substantial output-span loss
(`0.5630` Euclidean relative response error), despite much smaller projected
coordinates. No hyperparameters were changed to manufacture a chart win.

The machine-readable output and hashes are retained in
[`artifacts/research/fisher-response-geometry-probe-v1-synthetic`](../artifacts/research/fisher-response-geometry-probe-v1-synthetic/report.json).
Only the new kernel, protocol, standalone runner, configuration, contract,
schema, two test files and synthetic output artifacts are added. Existing
execution, attribution and P0 source files are unchanged.

Skipped checks are target-model captures, protected-data validation, live
attention derivatives, GPU/runtime qualification, compiler/deployment tests and
the broad repository scientific suite. The current implementation is complete
for synthetic normalization qualification. Independent eligible data, a
separately scoped live-attention contract/runner and its model-facing
authorization remain prerequisites to any later scientific campaign.
