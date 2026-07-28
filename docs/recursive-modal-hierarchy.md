# Recursive modal hierarchy

The recursive hierarchy rung turns a flat generator graph into a graph whose
nodes may themselves contain generator graphs. Its governing rule is:

> At level \(l+1\), the coordinates are Fisher-weighted modes of the causal
> connectivity exposed by level \(l\).

This is a stricter construction than clustering nodes by similar ablation
scores. A parent has to preserve the complete signed, affine, multi-port
boundary behavior of its children. It may have several outputs, including
intermediate outputs consumed before the end of the interval.

## The recursive object

```text
level 0: parameter modes
             ↓ generated computation
level 1: modal generators
             ↓ direct signed connectivity
level 2: connectivity modes / supergenerators
             ↓ direct signed connectivity
level 3: modes of supergenerator connectivity
             ↓
           repeat
```

The implementation has three model-agnostic layers:

- `modal_connectivity_modes.py` defines causal boundary transfers, message
  covariance and activation Fisher, and balanced restriction/prolongation
  factors.
- `modal_graph_hierarchy.py` defines signed direct-Jacobian component graphs,
  complete causal cuts, the modal encoder/core/decoder boundary, measured
  mode-to-mode interaction cores, recursively reusable hierarchical
  generators, and a separate parameter-sharing relation.
- `modal_graph_hierarchy_executor.py` defines source, candidate, shadow, and
  validation-only expansion execution with separate active and resident
  accounting.

The existing `ModalGeneratorGraphPlan` remains the authenticated leaf IR. The
hierarchy wraps it; it does not widen or weaken that artifact's v1 invariants.

## Exact multi-port contraction

For a selected causal interval, collect every edge entering or leaving the
interval as a boundary port. For a linearized child DAG,

\[
z=b+A_Gz+Bx,\qquad y=d+Oz+Dx,
\]

strict causality makes \(A_G\) block lower triangular. Its exact external
transfer is

\[
H=D+O(I-A_G)^{-1}B,\qquad
c=d+O(I-A_G)^{-1}b.
\]

The reference implementation obtains the same result by symbolic topological
execution. It keeps signs throughout. This is necessary for structures such
as a diamond where two individually large paths cancel at a later output.
Using response RMS as an executable edge would destroy that cancellation.

Graph-level boundary offsets also have explicit cut ownership. When several
readouts contribute to one boundary output, extraction assigns that output's
offset to the first canonical readout and assigns zero to the remaining
readout contributions. The extracted-child digest binds that deterministic
choice. This preserves the offset exactly once instead of either dropping it
or duplicating it across surfaced outputs.

A contraction is valid only when:

- its children form a contiguous causal interval;
- every child belongs to exactly one causal parent;
- every internal edge is retained;
- every crossing edge is surfaced exactly once;
- every boundary offset has one deterministic readout owner;
- every boundary width and causal stage matches;
- every executable edge is a direct Jacobian or an exact linear map; and
- the complete immediate child graph remains available as an exact fallback.

Noncontiguous relations are not causal parents. They can be
`ModalParameterSharingFamily` values, which nominate shared storage or shared
bases while preserving separate causal locations and executions.

## Connectivity modes

Let output port \(o\) have exact signed transfer \(H_o\). Its factor contains
only inputs available no later than that output. Later inputs are absent from
the schema rather than stored as zero blocks.

With block-local input covariance \(C_o\) and output activation Fisher \(F_o\),

\[
M_o=F_o^{1/2}H_oC_o^{1/2}=U\Sigma V^\top.
\]

This is an explicit v1 metric limitation, not an inference from per-port
statistics. `MessageMoments` contains one covariance and one Fisher matrix per
port; it does not contain cross-port covariance or cross-output Fisher blocks.
Consequently:

- a causal prefix with multiple input ports fails closed unless the caller
  explicitly declares its cross-port covariance to be zero;
- multiple outputs are factored independently, which is a block-diagonal
  output-Fisher approximation; and
- `ModalConnectivityDecomposition` records both assumptions in its
  authenticated payload.

Correlated modal inputs require measured joint covariance before this
approximation can be removed. Likewise, a whole-core Fisher objective requires
the cross-output Fisher blocks rather than a sum of independent per-output
quadratics.

For retained rank \(r\), the balanced higher-level maps are

\[
R_o=\Sigma_r^{1/2}V_r^\top C_o^{\dagger/2},
\qquad
P_o=F_o^{\dagger/2}U_r\Sigma_r^{1/2}.
\]

The candidate computes

\[
m_o=R_o(x-\mu),\qquad
\widehat y_o=\bar y_o+P_om_o.
\]

`R` restricts the fine boundary messages into connectivity modes. `P`
prolongs those modes back to the exact output interface expected by the rest
of the transformer.

Runtime lowering never materializes the dense product \(PR\). A mode expansion
has three distinct pieces:

```text
fine boundary inputs
        ↓ encoder R
modal input coordinates
        ↓ recursive modal core
modal output coordinates
        ↓ decoder P
fine boundary output
```

The fine-compatible graph executes \(R\) followed by \(P\). The separate
`encoder_graph` stops after \(R\), and `prolong_modal_outputs` is the \(P\)
adapter back to the interface expected by the uncompiled model. Both modal
boundaries are typed graph ports with authenticated moments, Fisher, reduction
ID, and sample count.

The initially emitted `recursive_graph` is deliberately only a typed handoff:

```text
modal input coordinates
        ↓ zero-cost identity aliases
modal output coordinates
```

This identity core proves that the next rung literally consumes the preceding
modal coordinates and moments. It is an interface scaffold, not evidence of a
new interaction structure and not by itself a meaningful higher-level
compression.

Identity components, injections, mode edges, and readouts are authenticated
aliases. They store no dense identity tensors and count as zero stored scalars
and zero linear MACs. All public graph and factor execution paths
reauthenticate their tensors before use.

The encoder's balanced gauge gives its modal coordinates a particularly clean
metric:

\[
\operatorname{Cov}(m_o)=\operatorname{Fisher}(m_o)
=\operatorname{diag}(\sigma_1,\ldots,\sigma_r).
\]

Therefore the encoder-output salience of mode \(k\) is \(\sigma_k^2\). A real
next-rung core additionally requires signed mode-to-mode interactions. For a
fine direct Jacobian \(J_{ji}:y_i\rightarrow x_j\), the projected interaction
is

\[
K_{ji}=R_j^{(\mathrm{target\ block})}J_{ji}P_i.
\]

`projected_connection` returns a proof-carrying `ProjectedModalConnection`.
That artifact retains the exact expansion, upstream and downstream factors,
fine Jacobian, direct-Jacobian evidence digest, endpoints, and projected
matrix. Validation recomputes \(R_jJ_{ji}P_i\) before lowering it to the graph
IR, so changing the matrix or reusing it with another expansion fails closed.

That projection is not yet an exact internal-edge compiler. A parent boundary
transfer has already composed its internal child edges. Its downstream factor
therefore does not normally expose an internal target port, and adding the
same interaction to modes of the composed transfer would double-count it.
Exact replacement requires an edge-torn or node-local factorization: remove
the fine edge from the downstream base path, expose its source and target
ports, then restore it exactly once through \(K_{ji}\). Parallel fine paths
that lower to the same modal endpoints additionally need an authenticated
aggregation rule; the current graph rejects duplicate endpoint edges.

`MeasuredModalCore` binds fresh input and output `MessageMoments` collected on
the exact interacting graph. Those moments may have arbitrary PSD covariance
and Fisher matrices and must not be copied from the balanced encoder output or
assumed diagonal, equal, or unchanged. The artifact is analysis-only and
explicitly grants no source-replacement authority. Only after edge tearing,
joint modal measurement, and validation can such a core become a replacement
candidate.

The optimal measured weighted error is explicit:

\[
\left\|F_o^{1/2}(H_o-\widehat H_o)C_o^{1/2}\right\|_F^2
=\sum_{k>r}\sigma_k^2.
\]

At full supported rank, the factor reconstructs \(P_FH_oP_C\). If covariance
or Fisher is singular, it makes no claim about the unmeasured null
directions.

## Reversible execution

The reference executor has four intentionally different modes:

| mode | returned path | active linear MACs | purpose |
| --- | --- | ---: | --- |
| `source` | exact immediate child graph | source | local baseline |
| `candidate` | retained connectivity modes | candidate | logical candidate measurement |
| `shadow` | exact immediate child graph | source + candidate + Fisher error metric | measure candidate without changing behavior |
| `adaptive_validation` | candidate or exact immediate child | source + candidate + Fisher error metric | validate a local expansion threshold |

`shadow` is immediate-child-authoritative even when the candidate is poor. A
threshold crossing expands only to that child. It does not silently mix in a
correction.

The immediate child fallback remains resident in v1. Accounting consequently
reports:

- candidate stored scalars and candidate active MACs;
- source learned parameters, source MACs, and source storage;
- total resident source-plus-candidate bytes; and
- the additional output-Fisher bytes required by validation modes; and
- logical candidate deltas separately from physical deployment claims.

This fallback is not transitive. At a higher rung, an exact modal child may
already contain information loss from a lower-rung candidate; falling back to
that child cannot recover the original fine leaf. The artifact and accounting
therefore deny transitive-leaf-fallback authority. While only the immediate
fallback is resident, the hierarchy makes no deployed storage-reduction,
latency, or end-to-end reversibility claim.

## Prepared execution and the latency crossover

The reference executor is a proof and validation surface, not a hot runtime.
Every public execution re-authenticates the hierarchy, decomposition, moments,
and source graph. Its canonical tensors live on CPU in float64, and execution
converts them to the request device and dtype. Shadow and adaptive modes also
run both paths and compute a dense Fisher error metric. Their wall time should
not be interpreted as the cost of a compiled modal graph.

`PreparedTorchHierarchyRuntime` and `PreparedMLXHierarchyRuntime` now separate
that load-time proof boundary from timed execution. They:

1. validate the source decomposition once;
2. independently copy source and candidate tensors into one execution dtype
   and device;
3. fold centering into
   \(b_o=\bar y_o-P_oR_o\bar x_o\);
4. retain only \(R_o\), \(P_o\), and \(b_o\) for the compact candidate; and
5. expose exact dense source, materialized dense candidate, and factorized
   candidate controls over the same canonical inputs.

The materialized candidate is essential. It computes the same truncated
operator as the factorized candidate but uses one dense \(P_oR_o\) matrix.
Comparing those two paths isolates execution geometry from changed numerical
values.

For output \(o\), let \(p_o\) be its legal input-prefix width, \(q_o\) its
output width, and \(r_o\) its retained rank. The factorized candidate costs

\[
C=\sum_o r_o(p_o+q_o)
\]

linear MACs per row, versus the composed dense boundary cost

\[
D=\sum_o p_oq_o.
\]

It has an arithmetic advantage only when \(C<D\). For one square
\(d\)-wide boundary, this means \(r<d/2\). The prepared candidate state stores

\[
K=\sum_o\left[r_o(p_o+q_o)+q_o\right]
\]

scalars: the two factors and folded output bias. At width 640, the resulting
shape-only ladder is:

| rank | dense MAC fraction | candidate-state fraction | synthetic retained weighted energy |
| ---: | ---: | ---: | ---: |
| 80 | 25.00% | 25.12% | 68.44% |
| 160 | 50.00% | 50.08% | 90.04% |
| 256 | 80.00% | 80.03% | 97.51% |
| 320 | 100.00% | 100.00% | 99.02% |

The energy column comes from the benchmark's artificial geometric spectrum.
It is useful for testing rate-curve accounting, but it is **not** Gemma
Fisher retention, NLL retention, or downstream accuracy.

The checked Apple M5/MLX 0.32 report compares row counts from 1 through 2,048.
A standalone boundary call synchronizes the GPU after every invocation, so
fixed synchronization and dispatch cost hide most low-rank gains. An
18-stage dependency chain instead feeds each output into the next prepared
stage and synchronizes only at the outside, approximating how several
generators sit inside one lazy model traversal.

Selected median speedups of factorized candidate over the rotating exact
dense source control are:

| runtime | rows | rank 80 | rank 160 | rank 256 | rank 320 |
| --- | ---: | ---: | ---: | ---: | ---: |
| one-thread CPU | 1 | 1.04x | 0.95x | 0.86x | 0.83x |
| one-thread CPU | 512 | 2.20x | 1.63x | 1.14x | 0.98x |
| MLX, sync each boundary | 1 | 1.05x | 1.02x | 0.98x | 0.98x |
| MLX, sync each boundary | 2,048 | 1.28x | 1.12x | 0.99x | 0.89x |
| MLX, 18 stages per sync | 1 | 0.93x | 0.92x | 0.86x | 0.85x |
| MLX, 18 stages per sync | 512 | 1.28x | 1.23x | 1.12x | 0.78x |
| MLX, 18 stages per sync | 2,048 | 1.33x | 1.15x | 0.99x | 0.85x |

This is a real local kernel-level crossover, but still not an end-to-end
model speedup. The measured factors are synthetic, the current Gemma hierarchy
is only a nomination, and the validation-safe executor still retains and runs
its fallback where required. The report is committed as
[`width640_prepared_benchmark.json`](../artifacts/hierarchy_speed/width640_prepared_benchmark.json)
with its generated
[`Markdown table`](../artifacts/hierarchy_speed/width640_prepared_benchmark.md).
It can be reproduced with:

```bash
fisher-graph-benchmark-hierarchy \
  --input-width 640 \
  --output-width 640 \
  --retained-ranks 80 160 256 320 \
  --row-counts 1 8 128 512 2048 \
  --backend both \
  --mlx-chain-depths 1 18 \
  --torch-repeats 9 \
  --torch-minimum-block-seconds 0.1 \
  --torch-warmup-iterations 30 \
  --torch-minimum-warmup-seconds 0.25 \
  --mlx-rounds 9 \
  --mlx-warmup-calls 20 \
  --mlx-minimum-warmup-seconds 0.25 \
  --mlx-iterations-per-round 50 \
  --output artifacts/hierarchy_speed/width640_prepared_benchmark.json
```

## Current Gemma hierarchy nomination

The strict live causal-map artifact has scientific digest
`1a25859340cd4772730fc631cdd7d7b859dda73c81d2447bed33c025d1e73afa`.
The analysis-only adapter produces nomination digest
`49a334abdd5e6e09e1fdb77cc5d823d651ad0df6e8fe4f7e43888f56eed62ffc`.

The nominated level contains:

| relation | result |
| --- | ---: |
| level-0 generator children | 18 |
| level-1 causal parents | 17 |
| L3/L4 internal finite-response relation | 1 |
| surfaced finite-response relations | 152 |
| nonlocal sharing families | 1 |

The only multi-child causal parent is L3–L4. L12/L15 remains two singleton
causal parents plus one nonlocal sharing-family reference.

This is a meaningful structural map, not yet an executable Gemma contraction.
The current directed values are finite upstream-suppression responses. They
nominate L3–L4, but they are neither direct Jacobians nor activation Fisher
over generator messages. The adapter rejects requests to reinterpret them as
such and grants no merge, prune, route, compile, execute, or mutation
authority.

## First live L3/L4 measurement rung

`gemma3_l3_l4_hierarchy_experiment.py` advances the structural nomination to
development-only local evidence on the frozen full-stack refit. It:

1. retains the L3 normalized MLP input as an autograd leaf while keeping every
   source-model and generator parameter frozen;
2. streams joint activation covariance and valid-position score-gradient
   Fisher induced by summed prompt NLL at the L3 input/output and L4
   input/output, including measured L3-output/L4-input cross blocks;
3. Fisher-balances the two exact affine generators into local restriction and
   prolongation factors;
4. checks that replaying L3's generated residual through the intervening
   native boundary exactly reproduces the observed ordinary L4 input;
5. evaluates both the literal-zero topology tear and the prompt-conditioned L4
   reference produced by the fitted mean L3 residual; and
6. linearizes around that mean-source reference, then uses exact randomized
   JVP probes to fit a signed matrix for each nonnegative logical lag on
   content-disjoint probe prompts.

The distinction between the ordinary and torn L4 inputs is essential:

```text
topology diagnostic = f_prompt(0)
centered edge base  = f_prompt(mean_L3_source)
ordinary L4 input  = f_prompt(native_L3_source)
candidate modal L4 = R4(centered edge base - mean_x4) + K * m3
```

Here `m3` is centered, so `m3 = 0` corresponds to the fitted mean L3 source.
That makes `f_prompt(mean_L3_source)`, not `f_prompt(0)`, the compatible
execution reference. The literal-zero path remains valuable topology evidence,
but using it directly would omit the large zero-to-mean affine response.
Adding the modal edge to the ordinary input would instead restore the varying
L3 interaction a second time.

`EdgeTornModalPairBoundaryContract` therefore makes both wrong states
inexpressible in the artifact declaration: the base must explicitly be named
as a `mean_source_reference_torn_base` bound to the L3 mean, and the artifact
grants neither ordinary-path nor source-replacement authority. That declaration
does not prove the provenance of a runtime tensor.
`CausalModalPairPlan` and its prepared runtime provide dense-control,
factorized, and staged execution for a supplied reference base, but remain
analysis infrastructure rather than a Gemma executor. The plan binder now
authenticates both factors, the mean-source linearization point, logical
positions, validity mask, and exact JVP artifact. It cannot authenticate the
external function that produced the supplied reference-base tensor. A plain
tensor name is not live provider provenance.

Run the development measurement with:

```bash
fisher-graph-gemma-l3-l4-hierarchy-dev \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
```

The command writes a tensor artifact and source-safe JSON report under the
ignored `.local-runs/` tree. The report includes raw joint-moment summaries,
factor spectra, prompt-local edge diagnostics, logical-lag energy, and a
rank/parameter/MAC curve. That resource curve is **shape-only accounting**:
the local analysis plan still calls the frozen transformer boundary to obtain
its mean-source reference base.

The first full development comparison used 40 fit sequences and four
content-disjoint probe sequences, with eight exact randomized JVP directions
per probe and logical lags 0 through 4. The split was not family-disjoint. Each
row below is the mean across the same four probes:

| retained rank | source / target reconstruction error | in-sample JVP residual | oracle-base pair vs local-control cosine | oracle-base pair vs local-control relative error | pair parameter fraction | whole-model parameter fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.291 / 0.229 | 0.323 | 0.763 | 1.187 | 0.114 | 0.99315 |
| 128 | 0.125 / 0.131 | 0.257 | 0.600 | 1.771 | 0.251 | 0.99421 |

The parameter columns use the corrected runtime-consistent count for all four
stored means. The two ignored reports below predate that correction and omit
two width-640 mean vectors, so their original candidate count is lower by
1,280 scalars.

The rank-128 run is the useful negative control. It reconstructs substantially
more of both generators and lowers the in-sample directional residual, yet
every finite pair-output probe gets worse. The denser modal state exposes more
of the finite displacement that a single stationary first-order edge does not
model. Increasing rank alone is therefore not the next rung.

The prepared factorized and dense pair executors agree within
`6.68e-6`, and the stage-3 factor binding agrees exactly. Those are
implementation controls, not fidelity results. The rank-64 pair still has
relative error above 1, so its error norm exceeds the local factor-control
output norm. Positive logical lags carry about 21.6% of rank-64 kernel energy
and 17.1% at rank 128, which supports nonlocal fan-out but not accurate
transport.

The two local reports remain ignored artifacts. Their report digests are
`4a6e2437711f77af0123fd8fd3c8f35bb557f36623da6ef3272bb7f665ddd016`
at rank 64 and
`313c3af50a260cf30477e4c41e2700f07cba62cab518ab6895dbde7a6280a672`
at rank 128.

Interpret the Fisher summaries narrowly. Gradients at these adjacent sites are
derived from the same scalar sequence NLL and are chain-related, so a near-one
normalized cross-Fisher block is evidence against an independent/block-
diagonal approximation, not semantic equivalence or replacement fidelity.
The report also separates empirical rank upper bounds from support ranks after
isotropic metric regularization, and explicitly marks weighted modal energy as
not being a fidelity metric.

This rung proves that joint Fisher/covariance and signed causal transport can
be measured on the frozen compiled stack, bound into a prompt-local analysis
plan, and executed consistently against a frozen-boundary oracle. It does
**not** prove that one prompt-independent edge is adequate, choose a deployment
rank, execute a replacement Gemma model, preserve cached decoding, pass
downstream fidelity, compress the deployed model, or improve latency.

## Prompt-free spectral mapping rung

`gemma3_l3_l4_spectral_mapping_experiment.py` measures the same frozen
L3-to-L4 boundary without loading a tokenizer, prompt text, or token IDs. The
Fisher bases and means remain data-conditioned upstream artifacts. The new
run constructs a deterministic internal reference by inverting the stored L3
normalized-input mean through the live unit-offset RMSNorm, applies controlled
modal impulses around the stored L3 output mean, runs only the L3
post-feedforward norm and exact L4 attention prefix, and projects the result
into the frozen L4 modal basis.

The full development run used all 64 source modes, a 40-position causal
sequence, impulse origins 0 and 8, logical lags 0 through 31, and a 64-point
rFFT. It measured both `0.05σ` local central secants and `1σ` operating
central secants:

| measurement | `0.05σ` local | `1σ` operating |
|---|---:|---:|
| unweighted joint rank at 90% / 95% / 99% spectral energy | `11 / 15 / 24` | `11 / 15 / 24` |
| source-σ-weighted joint rank at 90% / 95% / 99% energy | `11 / 18 / 34` | `12 / 18 / 35` |
| energy after logical lag 4 | `10.72%` | `10.98%` |
| even/nonlinear residual relative to odd response | `0.00714` | `0.13707` |
| mean / minimum two-origin spectral similarity | `0.672 / 0.310` | `0.679 / 0.294` |
| connected components at similarity `0.90` | `62` | `62` |

The local and operating source signatures have mean cosine `0.99956` and
minimum cosine `0.99014`. Their low-rank topology is therefore stable across
amplitude even though the full-`1σ` response has a material nonlinear
component. Source-mode standard deviations span about `103x`, so the weighted
ranks—not the more aggressive unweighted ranks—are the compression-facing
numbers. Pairwise clustering is sparse: the only non-singleton components are
modes `(15, 16)` and `(29, 58)`. The lower joint rank is consequently a
distributed subspace result, not evidence that most modes can be deleted or
merged pairwise.

The lag profile is causal and long-tailed: lag 0 carries `64.52%` of local
energy, lags 0 through 4 carry `89.58%`, and lags 0 through 8 carry `95.04%`.
The zero precausal response validates the mask path. The substantial
origin dependence rejects a shift-invariant-convolution interpretation and
points toward a position- or state-conditioned spectral generator family.

As a descriptive cross-estimator check, the origin-pooled local finite secant
has cosine `0.791` to the existing mean prompt-probe JVP kernel over lags 0
through 4, but relative difference `3.749`. These estimators use different
reference states and amplitudes; the comparison is neither ground truth nor
an accuracy validation.

The analysis executed 642 L4 attention prefixes and no L3 or L4 MLP bodies.
Its counted linear work is `44.17B` MACs, excluding attention-score products,
softmax, normalization, RoPE, and elementwise work. That is experiment cost,
not inference cost or a speedup. The ignored tensor artifact is `49,812,769`
bytes with SHA-256
`b88201f33210c8cd5be0d28f144b1963de84f51ba065959391aad3ce496c59d3`;
the source-safe JSON report-payload SHA-256 is
`00ac238e9867f001aa8b0926f2f05087b151d39e0cacf6f92d6192c50ac56165`.

## Position-conditioned spectral executor rung

The follow-up measurement moved away from the causal boundary. It used all 64
source modes, origins `8, 16, 24, 32, 40`, 32 causal lag bins, and a 64-point
rFFT. Across these interior origins, mean source-signature similarity was
`0.9832` locally (`0.9827` at `1σ`), compared with `0.672` in the earlier
origin-0/8 run. This changes the interpretation: the map is not stationary,
but most of the apparent instability came from origin 0. A small
position-conditioned family is a plausible representation of the interior.

`conditional_spectral_generator.py` compiles that family without reading any
non-fit response during factorization. For source scale vector \(\sigma\), it
factors

\[
B_o[s,\ell,t]=\sigma_s H_o[s,\ell,t]
\approx U_s C_o[\ell]U_t^\top .
\]

The real source and target bases come from Parseval-correct one-sided rFFT
unfoldings over declared fit origins only. Runtime input coordinates are
standardized by \(\sigma\), projected through \(U_s\), transported through a
piecewise-linear source-position core, accumulated causally by lag, and
decoded through \(U_t\) once per target row. A dense materialization path is
retained as an algebraic control. The prepared runtime supports variable
logical positions and masks; source origins cannot extrapolate beyond the fit
knots, while causal target rows may extend past the last knot.

The model-specific protocol preregistered these roles:

| role | origins | authority |
|---|---|---|
| fit knots | `8, 24, 40` | shared bases and knot cores |
| opened selection | `16, 32` | rank and correction choice |
| fresh assessment | `20` | no fitting or architecture changes |

The linear ladder selected source/target ranks `20×18`, the smallest declared
pair with worst selection relative error at most `0.20` and worst cosine at
least `0.98`:

| metric | origin 16 | origin 32 | fresh origin 20 |
|---|---:|---:|---:|
| local weighted relative error | `0.19766` | `0.17916` | `0.18962` |
| local cosine | `0.98028` | `0.98382` | `0.98186` |

The finite correction is deliberately narrower than a general quadratic
edge. The available single-mode `+/-` probes identify only diagonal
self-curvature. The correction therefore uses

\[
\phi_s(m)=(m_s/\sigma_s)^2
\]

with its own shared source/target bases and no cross-mode products, linear
term, or bias. It is zero at the reference and has zero Jacobian there. A
`4×4` factor retained `0.84599` of fit even-response energy and failed the
preregistered `0.85` gate without rounding. The next `4×6` factor retained
`0.85597` and passed:

| metric | origin 16 | origin 32 | fresh origin 20 |
|---|---:|---:|---:|
| linear-only finite relative error | `0.23742` | `0.21458` | `0.22775` |
| corrected finite relative error | `0.20929` | `0.18939` | `0.20065` |
| relative error reduction | `11.85%` | `11.74%` | `11.90%` |
| corrected finite cosine | `0.97844` | `0.98231` | `0.98013` |

The candidate was frozen before the origin-20 tensor existed. Assessment
strictly loaded its candidate and measurement hashes, performed no refit, and
did not expose a fitting API.

### Resource interpretation

The linear factor stores `36,992` coefficients and the diagonal-square factor
stores `2,944`, for `39,936` total. A deduplicated prepared runtime additionally
needs the 64 shared source scales: `40,000` floats, or `160,000` bytes in
float32. A matched dense two-branch family at three knots, 32 lags, and
`64×64` modal width contains `786,432` coefficients, so the factor state is
`5.078%` of that edge-only comparator (`94.922%` fewer).

For an illustrative 72-target traversal whose 33 emitting source rows are
restricted to the supported origins 8 through 40, there are 1,056 admitted
causal pairs. Once the position cores are prepared, the fused factorized
branches require `566,784` modal matrix MACs versus `8,650,752` for the
matched two-dense-branch control (`6.55%`). This analytic count is not measured
latency or full-model compute. It excludes core interpolation, normalization
and squaring, accumulation, masks, memory traffic, kernel launches, the
missing base provider, and every surrounding native model operation.

The ignored artifacts are bound as follows:

- interior measurement tensor:
  `a80b9ce1a5e433724e74cb7c29143d18442805a7b05fcb419ede6ad1e23686b3`;
- frozen candidate tensor:
  `9be7c0345acfaef8d77c273b1b69e3d83c930b807fd33756f955b5eef3fe2d2a`;
- candidate payload:
  `ce0649eb1d4559524243e8ad7b10dd9482dea31ca9ace35bde4e8568f2f49abc`;
- fresh origin-20 measurement tensor:
  `b31047899eb29a4f20efc69f2b55e9aa343cecd69260c90696db523e8a923987`;
- assessment payload:
  `ea42a293e4d5f4c1a6ef68b0a60826a14bc61b0e5e8ac373171d4a331d43d671`.

These are modal-edge development artifacts outside Git. They contain no
tokenizer, prompt text, token IDs, or source model state.

## Mixed-mode falsification rung

The axis-only measurement cannot identify cross-mode curvature. The next
assessment therefore froze a chord panel before loading the live model:

- source origin `28`, sequence length `60`, and causal lags `0–31`;
- 24 canonical unordered pairs spanning 16 modes;
- radial standardized magnitudes `0.5` and `1.0`, with each chord component
  equal to \(\rho\sigma_i/\sqrt{2}\);
- four signs `(++,+-,-+,--)`; and
- matching `(+,-)` singleton controls for every participating mode.

The first 12 pairs use exactly the top eight rows by
\(\sum_r U_{q,ir}^2\) leverage in the frozen diagonal-square source basis.
The other 12 are a balanced rank-coverage control. Every selected mode has
degree three. The pair order, radii, signs, source grid, candidate and
hierarchy hashes, decision thresholds, and runner code hashes were fixed
before the origin-28 response was opened.

For one shared measured zero response \(Y_{00}\), define

\[
D_{ab}=Y(a u+b v)-Y_{00},\quad
D_{a0}=Y(a u)-Y_{00},\quad
D_{0b}=Y(b v)-Y_{00}.
\]

The complete interaction is

\[
I_{ab}=D_{ab}-D_{a0}-D_{0b}.
\]

This singleton subtraction is necessary: the odd-odd Walsh component \(C_{11}\)
alone can miss constant, odd-even, and even-odd interaction. The generic
artifact reconstructs all four parities \(C_{00},C_{10},C_{01},C_{11}\) and
checks the decomposition exactly.

At the preregistered operating radius, the result was decisive:

| metric | `0.5σ` diagnostic | `1σ` decision |
|---|---:|---:|
| frozen corrected relative error | `0.14606` | `0.18634` |
| frozen corrected cosine | `0.98934` | `0.98339` |
| full nonadditivity \(e_{\mathrm{add}}=\lVert I\rVert/\lVert D\rVert\) | `0.05338` | `0.11266` |
| truth-leaking full-interaction oracle error gain | `7.07%` | `23.10%` |
| \(C_{11}\) share of interaction energy | `93.51%` | `80.74%` |
| \(C_{11}\) share of full response energy | `0.266%` | `1.025%` |

The support band required pooled nonadditivity below `5%`, each family below
`7.5%`, no reliable pair at or above `10%`, and oracle gain below `5%`.
Material failure required pooled nonadditivity or oracle gain at least `10%`,
either family at least `15%`, or a reliable energetic pair at least `20%`.
The run crossed three material gates:

- pooled nonadditivity was `11.27%`;
- oracle error gain was `23.10%`; and
- pairs `(0,2)` and `(1,2)` reached `21.87%` and `20.85%`.

The stress family reached `14.17%` nonadditivity and `31.88%` oracle gain,
while the rank-coverage family reached only `3.89%` and `3.30%`. The
interaction is therefore concentrated around modes emphasized by the frozen
diagonal branch rather than spread uniformly across the rank.

The frozen candidate's own nonadditivity was only
`2.28e-7` relative to its response, so the result is not leakage from a
hidden cross term in the candidate. The measured zero response and the
start/end repeat difference were both exactly zero. Candidate, model,
protocol, and source-code hashes were unchanged across the run.

### What the failure nominates

The result falsifies the cross-free linear-plus-diagonal architecture on this
specific panel, but it also identifies a narrow repair. Scaling the `0.5σ`
\(C_{11}\) response by four predicts the `1σ` \(C_{11}\) response with cosine
`0.99560` and relative error `0.10395`. Those pass the frozen bilinear gates
of cosine at least `0.95` and scale defect at most `0.25`; the `80.74%`
interaction-energy share also passes the `75%` parity gate.

A low-rank bilinear chord generator is therefore the next justified branch.
It should use an explicit channel for every one of the 28 unordered products
among the eight nominated sensitive modes. Withholding one of those pairs
would leave its channel unidentified, so selection is origin-disjoint rather
than pair-disjoint: fit at origins `8/24/40`, select at `16/32`, and assess the
already-frozen branch on fresh mixed directions at origin `20`. Origin `28`
is architecture-development evidence only and is never reused for fitting.
The remaining `19.26%` non-`C11` interaction stays explicit: if it remains
material after the bilinear branch, it requires a conditional residual or
path-integrated JVP rather than being silently attributed to a Hessian.

The assessment made 259 structural-map calls plus one baseline attention
prefix. Its partial analytic live-model accounting is `26.832B` MACs,
excluding attention-score/value matmuls, normalization, RoPE, softmax,
elementwise work, and memory traffic. The frozen candidate controls used
`27.132M` factorized modal MACs across 256 calls. These are experiment costs,
not deployed inference costs or latency measurements.

The ignored output bindings are:

- tensor file:
  `d76feac5e13a7e8f8f8d76bac97926f1d66131059337c987cfc43fb72d15f56b`;
- logical artifact:
  `37e79c582c9e83f3c45182f924a148a032b1b7baefc9bb08452feaf20bb6761c`;
- source-safe report:
  `931596c3889fe80c822c8620ca2ea9351751a98e93c3a49f4edce1713650ef3d`;
- frozen protocol:
  `c82dceb96ac3e6dbd400cceaf00700df026ecb754ae427de5a071044e8d8c8d8`;
- generic mixed-interaction artifact:
  `d1446f89ed57820287371df3735808b479486624204436ca8ae24a159fe5277a`.

The tensor file is 30,248,885 bytes and remains under `.local-runs/`. The JSON
contains aggregate metrics and hashes but no response tensors, prompt text,
token IDs, tokenizer, or model state.

## Bilinear modal-generator rung

The nominated repair is now implemented, compiled, and independently assessed.
It does not learn an arbitrary nonlinear map. The feature ABI is the complete,
authenticated upper triangle among the eight nominated modes
`(0,1,2,7,15,28,42,43)`:

\[
z_i=\Delta m_i/\sigma_i,\qquad
\phi_{ij}=2z_i z_j,\quad i<j.
\]

There are exactly 28 lexicographically ordered feature channels. A chord whose
two components are
\(\operatorname{sign}_i\rho\sigma_i/\sqrt{2}\) and
\(\operatorname{sign}_j\rho\sigma_j/\sqrt{2}\) therefore emits
\(\operatorname{sign}_i\operatorname{sign}_j\rho^2\) in its matching channel.
A singleton axis and every declared control pair emit exact zero. The prepared
runtime authenticates this feature map before composing it with a
position-conditioned spectral generator.

The compiler used a positional firewall:

- fit origins `8/24/40`;
- selection origins `16/32`;
- assessment origin `20`;
- sequence length `72`, causal lags `0–31`, radii `0.5σ/1σ`, and all four
  chord signs; and
- six separate negative-control pairs in selection and six new controls in
  assessment.

Every candidate rank was fit from the same fit panel. The selection responses
were opened only after the complete rate ladder existed, and the assessment
command authenticated the candidate file, source-safe report, model,
hierarchy, base candidate, feature map, selected plan, compile evidence, and
runner code before opening origin `20`. Assessment exposes no fitting path and
does not reuse origin `28`.

The smallest candidate satisfying every frozen gate was the rank-`8×8`
spectral plan:

| plan | stored coefficients | selection error | selection cosine | \(C_{11}\) error | passes |
|---|---:|---:|---:|---:|:---:|
| no bilinear branch | `0` | `0.20726` | `0.97987` | `1.00000` | no |
| rank `4×6` | `2,800` | `0.17142` | `0.98679` | `0.34436` | no |
| rank `8×8` | `6,880` | `0.16852` | `0.98729` | `0.23339` | yes |
| rank `12×12` | `14,928` | `0.16721` | `0.98751` | `0.16154` | yes |
| rank `28×64` dense | `172,032` | `0.16625` | `0.98767` | `0.07140` | yes |

The selected branch reduced pooled selection error by `18.69%`, recovered
`93.92%` of the measured \(C_{11}\) oracle headroom, and reached \(C_{11}\)
cosine `0.97330`. The direct dense branch improves full-response error by only
another `0.00227`; most of the useful interaction correction is therefore
present at the first passing compact rank.

The sealed origin-20 assessment also passed:

| metric | frozen base | base + bilinear |
|---|---:|---:|
| full mixed relative error | `0.20901` | `0.16937` |
| full mixed cosine | — | `0.98710` |
| relative error reduction | — | `18.96%` |
| \(C_{11}\) relative error | `1.00000` | `0.22976` |
| \(C_{11}\) cosine | — | `0.97406` |
| \(C_{11}\) oracle recovery | `0%` | `94.10%` |

The truth retained the expected quadratic scaling (`0.99565` cosine,
`0.10294` scale defect). Assessment negative controls had pooled leakage
`0.05622` against the strict `<0.075` gate and worst reliable-pair leakage
`0.08223` against `<0.15`; all six were reliable. The branch and the base
candidate both produced exact-zero \(C_{11}\) on their declared structural
controls. The executed float32 prepared graph matched the analytic branch to
`1.47e-7` relative error with maximum absolute difference `2.58e-4`.

### Bilinear resource interpretation

The selected branch stores `6,880` coefficients, `3.999%` of the matched
`172,032`-coefficient dense bilinear family (`96.001%` fewer). Adding it to the
existing `39,936`-coefficient linear-plus-diagonal executor gives `46,816`
edge coefficients. The corresponding matched dense three-branch family has
`958,464`, so the complete modal edge state is `4.884%` of that comparator
(`95.116%` fewer).

This accounting is edge-local. It excludes the reference/base provider, Gemma
weights, embeddings, attention, normalization, and the language-model head.
It is not a whole-model parameter reduction.

For one active source row, the selected bilinear transport uses `39,136`
factorized linear MACs versus `57,344` for its direct dense convolution
(`31.75%` fewer), before counting the 28 feature products, normalization,
position interpolation, accumulation, masks, memory traffic, and launches.
Across all three branches, the analogous factorized linear count is `163,552`
versus `319,488` for the matched dense main convolution (`48.81%` fewer).
These are analytic operator counts, not measured latency or end-to-end Gemma
compute.

The live compile made `1,231` structural evaluations plus one baseline
prefix; assessment made `275` structural evaluations plus one baseline
prefix. Their deliberately partial source-model accounting totals
`186.774B` MACs and excludes attention score/value matmuls, normalization,
RoPE, softmax, elementwise work, memory traffic, and the compiled branch. That
number describes the experiment used to collect evidence, not deployed
inference.

Reproduction is deliberately split into compilation and sealed assessment:

```bash
fisher-graph-gemma-l3-l4-bilinear-spectral-dev compile \
  --device cpu \
  --dtype float32

fisher-graph-gemma-l3-l4-bilinear-spectral-dev assess \
  --candidate \
    .local-runs/google--gemma-3-270m/modal-generator-l3-l4-bilinear-spectral-executor-dev-v1.pt \
  --candidate-file-sha256 \
    631006014eaf092a27a72d2918ab61d144fe925896a4ccb812094e10d1200cf7 \
  --candidate-report-sha256 \
    856d116f687fcde936e447d8f14053e74fa9ebf3a6996a60c527cec2e541a37a \
  --device cpu \
  --dtype float32
```

The ignored artifacts are bound as follows:

- candidate file:
  `631006014eaf092a27a72d2918ab61d144fe925896a4ccb812094e10d1200cf7`;
- candidate logical artifact:
  `660830a57acda7777756d5053556c4bf185cffb3302cda55f11d7c605cfdefaa`;
- compile report:
  `856d116f687fcde936e447d8f14053e74fa9ebf3a6996a60c527cec2e541a37a`;
- compile evidence file:
  `26e6683ab2de2fec6ef80c36d41b5aed62e72c4eb3fd6e75db38022f1af4bad5`;
- selected spectral plan:
  `51aebc23ed730a6512686e7586581a25e055d076099f9970ab39b9a6d2f8acb7`;
- assessment file:
  `7252086053899895716765f1531221a31559aef26f6971cc1fa7d5aab2b8de5b`;
- assessment logical artifact:
  `26540cb0b44044673ab58673780be6046ffe091cca7b26e7d9e0d7b12c14dae8`;
- assessment report:
  `6963ba73b71d178e66c58bbcdaf9d1ca9feffb51ce1ad062599b55bdd3f753ab`.

The positive claim is narrow but new: a compact bilinear generator transports
all 28 known sensitive-mode edge identities across held-out interior
positions without refitting. This is position generalization for known edges,
not unseen-pair generalization, a full Hessian, a prompt-conditioned
replacement, downstream fidelity, or model compression.

## Prompt-blind state-conditioned reference provider

The next rung removes the fixed mean-reference tensor from the modal-delta
experiment and asks whether a causal provider can predict the 64-mode L4
reference state from the current L3 modal state. It is prompt-blind only after
the Fisher basis package has been frozen: the provider protocol loads no
prompt text, token IDs, tokenizer, natural activation rows, score-gradient
rows, or prompt-local kernel, but the upstream basis was originally estimated
from prompts.

The frozen synthetic protocol separates three roles:

- 80 Rademacher, AR(1), and axis probes fit every candidate;
- 32 seed-disjoint Rademacher and AR(1) probes select the smallest passing
  candidate; and
- 88 sparse, chirp, axis, radial-collision, and null-collision probes are
  available only to a one-shot assessment command.

V2 preserves all 80 fit identities and all 88 assessment identities from v1,
while sharing zero selection hashes and zero direction seeds with v1. The
assessment panel has a protocol-independent identity so changing fit or
selection metadata cannot reopen it.

### Source-normalized causal generator

The v1 executor chose an expert independently for each causal pair, then
summed every earlier source contribution:

\[
y_t = y_t^{\mathrm{local}}+
\sum_{s<t}\sum_e p(e\mid t,s)V_eU_ex_s.
\]

Its largest errors appeared after synthetic source activity began, especially
for long active suffixes and doubled radial scale. V2 makes the smallest
structural change consistent with attention-like causal transport:

\[
y_t = y_t^{\mathrm{local}}+
\sum_{s<t}p(s\mid t)
\sum_e p(e\mid t,s)V_eU_ex_s.
\]

One learned projection of the existing pair-hidden state produces a scalar
source score. A masked softmax normalizes those scores only across eligible
earlier positions. Empty rows remain exact zero, future positions have zero
forward and gradient influence, sparse padding matches compact execution, and
the legacy summed-source artifact remains loadable. At router width 16 this
adds exactly 16 stored scalars per candidate.

### Fresh selection result

Five of six rate points passed the unchanged selection gates:

| candidate | stored scalars | Fisher error | cosine | max p90 | worst family | passes |
|---|---:|---:|---:|---:|---:|:---:|
| dense `64→64` | `15,046` | `0.03336` | `0.99945` | `0.17051` | `0.03385` | yes |
| spectral `8→8` | `910` | `0.08263` | `0.99659` | `0.28775` | `0.09940` | **yes, selected** |
| spectral `16→16` | `2,422` | `0.06060` | `0.99817` | `0.38113` | `0.06814` | no: p90 |
| spectral `24→24` | `3,886` | `0.03845` | `0.99927` | `0.19883` | `0.04141` | yes |
| spectral `32→32` | `5,606` | `0.05279` | `0.99861` | `0.34165` | `0.05539` | yes |
| spectral `48→48` | `9,814` | `0.04555` | `0.99896` | `0.31216` | `0.04772` | yes |

The rank-8 p90 fell from `0.83766` in v1 to `0.28775` in v2. Matching-rank
p90 improvements ranged from `38.76%` to `65.65%`; delayed-onset, doubled
radial-scale, and long-active-suffix groups also improved strongly. Because
v2 intentionally uses fresh direction seeds, this is a geometry-matched
replication rather than an identical-input ablation.

The selected provider stores `93.95%` fewer scalars than the full-width
provider. Including feature encoding and output scaling, its declared ideal
provider MAC savings versus that full-width provider are `87.16%`, `79.54%`,
`71.28%`, and `58.65%` at lengths `32`, `72`, `128`, and `256`. These counts
exclude activation, softmax, masking, additions, memory traffic, launches, and
all surrounding Gemma operations. The full-width comparator also retains
rank-16 experts; it is not a literal dense edge matrix. These are
provider-relative analytic counts, not full-model FLOPs or latency.

### Sealed assessment: fidelity passed, panel control failed

The one-shot command authenticated the compiled file, report, code bundle,
basis, source model, protocol, assessment-panel identity, metric gauge, frozen
selection, and selected rank-8 plan before recording its irreversible claim
and materializing any assessment target. It performed no refit or reselection.

Every candidate-fidelity and structural gate passed:

| assessment metric | result |
|---|---:|
| Fisher-weighted relative error | `0.05900` |
| reference cosine | `0.998261` |
| reduction versus constant / position-only | `47.80%` / `47.92%` |
| maximum per-probe p90 error | `0.28970` |
| worst family relative error | `0.09502` |
| prepared float32 parity error | `3.27e-8` |
| support fraction | `1.0` |
| causality / padding / repeat violations | `0 / 0 / 0` |

Family errors were axis `0.03236`, chirp `0.09502`, null collision `0.01972`,
radial collision `0.03785`, and sparse `0.04673`.

The preregistered composite result is nevertheless **failed** because its
teacher-panel identifiability gate did not pass. For two measured,
Fisher-weighted target variants \(Z_i,Z_j\), the gate computes

\[
d(i,j)=
\frac{\lVert Z_i-Z_j\rVert_2}
{\max((\lVert Z_i\rVert_2+\lVert Z_j\rVert_2)/2,10^{-12})},
\]

keeps the minimum pairwise value inside each collision group, then requires
the minimum across every group to be at least `0.01`. The metric contains no
candidate prediction.

| collision family | groups at or above `0.01` | observed range |
|---|---:|---:|
| radial scale | `4 / 4` | `0.01516–0.03652` |
| axis sign | `0 / 8` | `0.000105–0.002329` |
| gain-null coordinate | `0 / 4` | `0.00000924–0.0000531` |

The global minimum was `9.2393e-6` for null mode 63. Thus the provider did not
fail to predict collision targets; rather, the frozen teacher panel did not
demonstrate a 1% effect for every tagged variable. Radial residual magnitude
is identifiable under this construct. Axis-sign effects are small relative to
the full-sequence target norm, and the exact RMSNorm gain-null coordinate is
nearly invariant downstream.

The result remains permanently recorded as a composite-control failure. Its
independent positive evidence is narrower: a frozen 910-scalar provider
generalized across all five sealed synthetic families on every prediction and
execution-structure gate.

Reproduction remains split between compilation and the already-consumed
one-shot assessment:

```bash
fisher-graph-gemma-l3-l4-reference-provider-dev compile \
  --device cpu \
  --dtype float32

fisher-graph-gemma-l3-l4-reference-provider-dev assess \
  --candidate \
    .local-runs/google--gemma-3-270m/modal-generator-l3-l4-reference-provider-dev-v2.pt \
  --candidate-file-sha256 \
    37bd6fbda9b3660777f0388561e4e8d7d1a28e3958bcb98c69ca302cd1f77ae1 \
  --candidate-report-sha256 \
    1e14518f915821aa7448b6f4799e322e2451074b3030ba4107c6a2a0924be4d9 \
  --device cpu \
  --dtype float32
```

The ignored outputs are bound by:

- compiled tensor:
  `37bd6fbda9b3660777f0388561e4e8d7d1a28e3958bcb98c69ca302cd1f77ae1`;
- compiled logical artifact:
  `973bab7c72d456247a535137fd3bbfa8fd064b4710718dc905dea94963144f46`;
- compiled report:
  `1e14518f915821aa7448b6f4799e322e2451074b3030ba4107c6a2a0924be4d9`;
- selected plan:
  `7ab42890daece95eeedbf08ba0e5727f2bccfd7be20e00a4e404539cd1bf9cee`;
- assessment tensor:
  `a4175def42020f1b13a370e7ee9308dcc2be3b3439960987418573ba4379b2dd`;
- assessment logical artifact:
  `21500080aed580e91b605a6fdd01984dcc41676c0dea96a7813ee0ec4a8cc57d`;
- assessment report:
  `613856ec39a7d0cac21cc6e41a155a4609c73ea05e4daa01ccf1affe26153b6e`;
- assessment panel specification:
  `c690e9f85f5629ab2701fc5db487aea1404864256f5fe24034e35143047af102`;
- assessment score:
  `510e7f406e1e9fb18f33a3b24cda90aee44ed5a6508ccd6a8c577016549c82ae`;
- frozen basis payload:
  `b2217153911436673f2ff7475c658c928112e802f5999619393287d2b0803c01`;
- v2 synthetic protocol:
  `82b6d07830c3410a89f24233fc0d2ddfb0f3c1972739b6fe55144183485b3fb3`;
  and
- v2 training protocol:
  `4eb3bc860539683802355bd156dd59ff6007e4de86c1b98558f51d45b798fbaf`.

## Next validation gate

The v2 assessment panel is consumed and must not be rerun or rescued by
lowering its threshold. The next rung must preregister a genuinely fresh v3
panel and split the overloaded collision decision into:

1. teacher-construct sensitivity gates for variables expected to matter;
2. teacher-construct invariance ceilings for intended null controls;
3. a panel-inconclusive state for effects too weak to test;
4. candidate contrast recovery on sufficiently identified groups, comparing
   \((\hat T_i-\hat T_j)\) directly with \((T_i-T_j)\); and
5. fresh modes, positions, lengths, seeds, hashes, and a new one-shot ledger
   identity.

If that frozen provider gate passes, the rank-8 provider can be composed with
the frozen linear, diagonal, and bilinear branches in one self-contained
L3→L4 graph with transitive native fallback. The following rung is then
source-authoritative shadow execution on a family-disjoint natural-prompt
split, scoring NLL, full-vocabulary KL, and top-1 agreement. Resident storage,
active compute, and measured latency remain downstream of that fidelity gate.

Parallel-path aggregation must also be authenticated before a later rung
admits multiple fine edges with the same modal endpoints. Until those gates
pass, the provider is strong synthetic evidence rather than an executable
Gemma replacement or a compression result.
