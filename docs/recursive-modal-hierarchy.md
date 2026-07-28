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

## Next validation gate

The next rung must:

1. validate the JVP fit on held-out directions;
2. replace or augment the stationary first-order edge with a
   finite-displacement correction, such as a prompt-conditioned residual or a
   path-integrated Jacobian;
3. compile and authenticate a provider for the prompt-conditioned mean-source
   L4 reference base;
4. execute that self-contained graph and collect fresh moments on its exact
   runtime path;
5. choose and freeze both architecture and rank on a disjoint selection split;
6. add a transitive fallback to the original fine leaf;
7. run source-authoritative shadow execution on a fresh family-disjoint
   assessment split; and
8. replace logical shape accounting with measured fidelity, resident storage,
   active compute, and latency results.

Parallel-path aggregation must also be authenticated before a later rung
admits multiple fine edges with the same modal endpoints. Until those gates
pass, L3/L4 is a measured development hypothesis, not an executable Gemma
replacement or a compression result.
