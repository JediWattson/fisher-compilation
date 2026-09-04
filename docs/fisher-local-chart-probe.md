# Stage-aware Fisher local-chart probe

## Scientific questions

The probe answers three ordered questions.  Each answer is reported
independently; a failure at a later rung does not erase an earlier result.

1. Does a stage-aware local atlas under the shared layer-4-output empirical
   Fisher metric capture held-out attention-head and MLP-probe-block flows
   better than equal-rank Euclidean PCA controls?
2. With identical affine-generator capacity, are those flows more predictable
   in the Fisher coordinates?
3. Separately, can permitted causal source features predict typed chart
   destinations better than source-chart and stage-global frequency controls?

This is a layer-4 diagnostic experiment.  It is not a compression result and
it does not install an overlay in the default compiler or executor.

## Exact Gemma 3 layer-4 boundary

The pinned `google/gemma-3-270m` decoder block executes two residual stages:

```text
h = layer.4.input
q_A = o_proj(attention(...))
v_A = post_attention_layernorm(q_A)
h_A = h + v_A = layer.4.post_attention

x_M = pre_feedforward_layernorm(h_A)
q_M = down_proj(act(gate_proj(x_M)) * up_proj(x_M))
v_M = post_feedforward_layernorm(q_M)
h_+ = h_A + v_M = layer.4.output
```

Historical `X4` is `x_M`, not `h_A`; historical `H4` is `h_+`.  The atlas
state cloud is exactly `{h, h_A, h_+}`.  Attention generators are sourced at
`h`; MLP generators are sourced at `h_A`.

For either branch, Gemma RMSNorm computes in float32

```text
D(q) = diag((1 + weight.float())
            * rsqrt(mean(q.float() ** 2) + epsilon))
```

and casts once to the input dtype.  The pinned epsilon is `1e-6` and the
learned scale uses unit-offset parameterization.  The probe computes `D_A`
from the complete `q_A` and `D_M` from the complete `q_M`; it never normalizes
a head or block independently.

For attention head `j`, with the 256-wide head slice of the 1024-wide
pre-`o_proj` context,

```text
q_A,j = context_j @ W_O,j.T
v_A,j = D_A(q_A) q_A,j
```

For MLP probe block `r`, with exhaustive intermediate-channel assignment
`I_r`,

```text
q_M,r = down_input[..., I_r] @ W_down[:, I_r].T
v_M,r = D_M(q_M) q_M,r
```

Diagnostics accumulate in float64 and require closure of the two
pre-normalization sums, both realized branch sums, all three residual
identities, and the complete component sum.  Component GEMM reduction order
can prevent bitwise equality, so tolerances are recorded with observed
absolute and relative errors.

Two deterministic, exhaustive four-way MLP partitions are run: contiguous
quarters and seed-bound interleaved quarters.  They are probe partitions, not
discovered functional generators.  Material disagreement is reported as
`MLP conclusions are partition-sensitive`.

## Shared rank-64 output-Fisher carrier

The authoritative source is the Calibration-A Fisher section of
`.local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt` at
`layer.4.output`.  A minimal sidecar extracts only its top-64 eigensystem and
source metadata; no Calibration-B result, codec selection, or validation
metric is copied or consulted.

The sidecar represents the existing rank-64 empirical Fisher/VJP carrier

```text
F_64 = Q diag(lambda) Q.T
```

at `layer.4.output`.  It is not the exact full 640-dimensional Fisher matrix.
Using it for `h`, `h_A`, `v_A`, and `v_M` gives a common downstream-sensitivity
metric, not exact local Fisher geometry at the earlier boundaries.

Regularization remains inside `span(Q)`.  For local covariance `C_i`,

```text
C_Q,i = Q.T C_i Q
Lambda_reg = diag(lambda) + ridge I_64
S_Q,i = sqrt(Lambda_reg) C_Q,i sqrt(Lambda_reg)
B_i = Q invsqrt(Lambda_reg) U_i,k
z_i(s) = U_i,k.T sqrt(Lambda_reg) Q.T (s - mu_i)
P_i^F(s - mu_i) = B_i z_i(s)
```

The implementation never materializes a 640-by-640 Fisher matrix.  It checks
`B_i.T F_64,reg B_i = I`, projector idempotence, carrier-span containment, and
dense equivalence on a small synthetic problem.  Euclidean directions outside
`span(Q)` appear only in Euclidean controls; no such direction is called
Fisher-supported.

## Fixed neighborhoods and equal-capacity controls

Each A-fit token contributes the state-only centroid `(h + h_A + h_+) / 3`
to a deterministic K-means seed cloud. Its gate centroids `g_i` assign every
row in the full three-stage state cloud. Each local chart origin `mu_i` is the
mean of the full-cloud A-fit rows assigned by those gates; it is deliberately
distinct from `g_i`. This prevents a stage-specific outlier from creating a
chart with no attention or MLP source rows without biasing the chart covariance
around a centroid from a different cloud. The gate centroids, full-cloud
assignments, and local chart origins are receipt-bound, frozen, and reused by
every basis method and partition control:

Flow covariances are stage-aligned: attention-head and attention-total flows
are attached only to their `h` source rows, MLP-probe-block and MLP-total
flows only to their `h_A` source rows, and the auxiliary complete-layer flow
to `h`. Terminal `h_+` rows contribute to the state cloud but carry no
invented outgoing flow.

- B0: one global state-only Euclidean PCA basis broadcast across the fixed
  `K` neighborhoods;
- B1: local state-only Euclidean PCA;
- B2: local state plus total attention, MLP, and complete-flow Euclidean PCA;
- B3: local state plus attention-head and MLP-probe-block flow Euclidean PCA;
- B4: local state plus component-flow atlas under the shared output-Fisher
  carrier.

Every arm has the same `K`, coordinate rank `k`, A-fit rows, initial
assignments, affine family, ridge grid, router architecture, feature budget,
training budget, constant-mean control, and selection protocol.  Geometry,
affine prediction, and routing improvements are reported separately as
`delta_tangent`, `delta_affine`, and `delta_router`.

The attention and teacher-forced MLP Euclidean comparator arms are chosen
from B0--B3 on A-selection and frozen in the candidate before A-guard opens;
the guard cannot be used to pick the most favorable control. Tangent capture
uses the preregistered B4-versus-B3 comparison.

Chart membership combines a normal-space reconstruction residual with
tangential support:

```text
e_perp = ||(I - P_i)(s - mu_i)||^2_metric / (sigma_perp_i^2 + eps)
e_parallel = z_i(s).T inv(Sigma_z,i) z_i(s)
```

The support radius is selected only on A-selection.  A point far along the
tangent plane is rejected even when its normal-space reconstruction residual
is zero.  Reports separate inside/outside-support performance, nearest-support
distance, out-of-support rate, and `h_A -> h_hat_A` composed support drift.

## Generators, composition, and causal context

For every arm and component, the fixed family is affine in chart coordinates:

```text
y_hat_i,r = a_i,r + J_i,r z_i(source)
v_hat_i,r = decode_i(y_hat_i,r)
```

The same ridge grid and output-coordinate rank are used everywhere.  Both the
constant mean and affine result are retained.  Attention has two primary
conditions:

- source-only: `g_A,j(z(h))`;
- causal-context: `g_A,j(z(h), c_t,j)`.

The fixed primary context contains attention entropy, top-1 mass, top-4 mass,
and causal relative-position mean and standard deviation.  It contains no
future token, destination state, branch target, or full-dimensional identity
copy.  Attention probabilities and summaries are accepted only by generator
and router APIs, never geometry fitting.  Any compact transported-value sketch
is separately labeled a context-sufficiency upper-bound ablation.

MLP evaluation has two noninterchangeable paths:

```text
h_hat_+^teacher = h_A + sum_r g_M,r(phi(h_A))
h_hat_A = h + sum_j g_A,j(phi(h), causal_context)
h_hat_+^composed = h_hat_A + sum_r g_M,r(phi(h_hat_A))
```

Teacher-forced MLP failure is evidence against the selected local affine MLP
representation.  Teacher success with composed failure indicates attention
error or support drift; it is not by itself evidence of nonlinear MLP
dynamics.

## Typed chart graph

Teacher destinations label two distinct stages:

```text
C_i^input --attention--> C_j^intermediate --MLP--> C_k^output
```

The attention router uses only `z(h)` and the permitted causal summary.  The
MLP router uses only `z(h_A)`.  A separate affine softmax is fitted for each
fixed source chart and stage, and its probabilities normalize only over that
stage's destination charts.  Source-chart destination frequency and
stage-global destination frequency are the controls.  Attention and MLP
cross-entropy, top-1, entropy, support, sparsity, and split drift are never
merged.

## Split, selection, freeze, and guard protocol

The existing prompt-free three-role Calibration-A corpus supplies disjoint
A-fit, A-selection, and A-guard sequences.  Roles are separated before
tokenization or token-row expansion.  They are disjoint by both exact sequence
identity and family identity.  Every expanded row retains its sequence ID and
logical position, and bootstrap resampling uses those sequence IDs rather than
token rows.  The split commitment binds the corpus artifact, tokenizer
contract, all three role manifests, and all three role-input file hashes.

Calibration B is explicitly prohibited from fit, selection, candidate freeze,
guard evaluation, and result serialization.  The only B-derived value retained
by the prompt-free split commitment is the already-frozen assessment-manifest
digest used as a deny-list identity; no B prompt, family payload, metric,
selection, or result is opened or copied.

Only A-fit estimates the carrier-independent geometry and model coefficients.
Only A-selection selects ridge/support settings and the one candidate bundle.
The complete bundle, source hashes, coefficients, hyperparameters, selection
transcript, and report schema are content-addressed and frozen before any
guard claim.  Guard preflight reloads and authenticates the bundle and runs a
non-guard dry evaluation before an owner-only, manifest-global, atomic claim.
Only then is A-guard JSON parsed.  A post-claim failure burns the guard and is
not retried.  The frozen bundle is evaluated exactly once.

The frozen weights-only bundle has exactly these top-level sections:

```text
baselines                 # exact B0, B1, B2, B3, B4 arms
selected_hyperparameters  # selected arm plus frozen ridge settings
generators
routers
support
selection_transcript
bindings
report_schema
```

Every baseline records the same fixed-geometry receipt, which covers the gate
centroids, full-cloud assignments, chart origins, and chart seed, plus the same
serialized capacity signature. Cross-arm and cross-partition equality and the
receipt itself are recomputed after candidate reload. `bindings` includes the
protocol, split-manifest, Fisher
carrier, corpus-artifact, and prompt-free A-guard-manifest SHA-256 identities;
the envelope repeats the protocol, split, and carrier identities.  Nested
bundle keys and string values are scanned as well as top-level fields, so a B
role/result or guard payload cannot be hidden inside a transcript or model
state.  The prompt-free A-guard manifest digest is the sole permitted guard
identity in the pre-claim bundle.

The protocol source-file hashes are captured immediately after a fresh
interpreter imports the implementation, checked again before freeze, and
rechecked before the guard claim. Freeze writes a `.pt` payload and canonical
sidecar exclusively, then reloads
the payload with PyTorch's weights-only loader and verifies both its logical
content digest and exact file digest.  Guard preflight independently derives
the split from the supplied corpus, checks the corpus and guard bindings, runs
the non-guard dry evaluation, rejects any in-memory mutation, and strictly
reloads the same bytes again.  It then invokes the existing durable
manifest-global ledger.  The returned receipt must authenticate and bind the
same protocol, guard manifest, and candidate digest before guard text can be
materialized.

The single guard result at the configured result path is canonical JSON written
with exclusive creation.  It binds the candidate file and logical digests,
protocol, split, corpus, carrier, guard manifest, and durable claim receipt.  A
second write at that path is refused.  An invalid claim adapter, corrupt guard
file, evaluation failure after claim, or result-write failure does not
authorize another guard claim or evaluation.

Confidence intervals resample whole sequence IDs.  Tokens are never bootstrap
units.

## Evaluation and verdicts

For A-fit, A-selection, and the one frozen A-guard run, artifacts report:

- Euclidean and shared-Fisher reconstruction of `h`, `h_A`, and `h_+`, plus
  rank curves and support;
- Euclidean and Fisher tangent capture for each head, each probe block, total
  attention, total MLP, and the full layer flow;
- constant and affine generator errors, direct totals, component sums,
  source-only/context attention, and teacher/composed MLP;
- one-step stage Euclidean/Fisher error, cosine, relative norm, and support
  drift;
- typed router metrics and both frequency controls;
- an opt-in `layer.4.output` diagnostic overlay with delta NLL, KL, and top-1
  agreement;
- chart/generator/router parameters, serialized bytes by dtype, peak host and
  device memory, forward count, and capture/fit/evaluation time.

Six verdicts remain independent: Fisher tangent capture, affine generators
(attention and MLP separately), attention context sufficiency, graph routing,
composition, and full-sequence fidelity.  No count is described as realized
model compression.

Recurrent generator sharing is out of scope.  It should be proposed only
after held-out vector fields agree over overlapping chart support, not from
mean-vector cosine similarity.

## Reproduction phases

The runnable entry point is `python -m
fisher_graph.gemma3_fisher_local_chart_probe`.  Its final `commands.md` records
resolved paths and arguments.  The intended phase order is:

```bash
.venv/bin/python -m fisher_graph.gemma3_fisher_local_chart_probe inspect
.venv/bin/python -m fisher_graph.gemma3_fisher_local_chart_probe extract-carrier
.venv/bin/python -m fisher_graph.gemma3_fisher_local_chart_probe smoke
.venv/bin/python -m fisher_graph.gemma3_fisher_local_chart_probe fit-select
.venv/bin/python -m fisher_graph.gemma3_fisher_local_chart_probe guard
```

`run` executes the same order, but the explicit phases are preferred because
the irreversible guard claim is visible and separately preflighted.
