# Fisher Graph Compilation

An instrumentable transformer compiler research project.

The central question is whether frozen transformer computation can be
re-expressed as a smaller causal graph:

```text
weights
  ↓
Fisher coupling
  ↓
parameter clusters
  ↓
computational modes
  ↓
modal generators
  ↓
generator interaction graph
  ↓
inference by graph traversal
```

The repository contains a verified end-to-end toy implementation, a
source-free Gemma layer executor, a full Gemma MLP-stack generator baseline,
and a recursive L3→L4 hierarchy with a prompt-blind state-conditioned
reference-provider experiment and fresh sealed contrast assessment. It is a
research compiler, not a production compression library.

[![Research ladder from the verified toy executor to the prompt-blind reference-provider fidelity result](docs/images/research-ladder.svg)](docs/images/research-ladder.svg)

## Current finding

The authenticated function-preserving width control removed the initial-state
confound from the prior rank-16/rank-64 comparison. It exactly replayed D3's
rank-16 cold start and embedded the same observable function into a
gradient-open rank-64 arm. Initial observable error was exactly `0`; the
maximum provider-chart JVP absolute error was `8.88e-16`; and the added
rank-64 decoder, encoder, and executor paths passed their preregistered
gradient-openness checks.

| arm | stored scalars / canonical MACs | ordinary error / cosine | null | radial pass / macro error | signed pass / macro / worst error | weighted training objective |
|---|---:|---:|---:|---:|---:|---:|
| rank 16 D3 replay | `4,276 / 1,315,072` | `0.00769406 / 0.99997058` | `24/24` | `16/16 / 0.083419` | `3/8 / 0.380298 / 0.738002` | `0.222771` |
| rank 64 matched lift | `19,012 / 3,190,528` | `0.00652994 / 0.99997878` | `24/24` | `16/16 / 0.071013` | `3/8 / 0.386891 / 0.775428` | `0.167569` |

Both arms passed all `12/12` ordinary, `24/24` exact-null, and `16/16`
radial checks. Rank 64 improved the smooth fit measures—ordinary error,
radial macro error, and the weighted training objective—but it passed only
the same three signed identities as rank 16. Its signed macro and worst error
were also slightly worse. The valid formal outcome is `primary_both_fail`;
the conditional replication did not run because only a rank-16-fail /
rank-64-pass result could open it.

This is now a causal negative result for outer width under the matched
600-step fit budget: quadrupling outer rank alone did not recover categorical
signed fidelity. The rank-64 arm stores `4.446×` as many scalars and requires
`2.426×` the declared canonical MACs, so it is also not a compression
candidate. The result authorizes the preregistered expert/core control, which
will test capacity inside the conditional executor rather than widening the
outer modal packer. It authorizes no compressed-width ladder, C3, held-out
generalization, full-model replacement, compression, or wall-clock speed
claim.

The durable external result receipt is logical artifact
`9e07c7208b3b690a8024bd809a0d80c2842145cfa73e655bb737e5497913ce47`,
tensor file
`5a3c8de7bd6731a78904a14c488648f6641d6b3cbe96167438f633b65f9104c5`,
and report
`6aacad6f05e3b43bbeba62b6ce7ae35897af6af60d53f9dfa96eec951ad6965f`.
The protocol binding is
`c3ad81c84d41108839b5fcab13e3b5d47d99a55ae9a9223c3f116edb6b457597`
and the code bundle is
`5c314fff7959f659257911ca0190605ea4ef41c556bd18a27108acb48d2545a4`.
The tensor artifact and tensor-free JSON report remain ignored under
`.local-runs/`; the receipt recorded here is the durable trust root.

```bash
fisher-graph-gemma-l3-l4-function-preserving-width-dev describe

fisher-graph-gemma-l3-l4-function-preserving-width-dev run \
  --device cpu \
  --dtype float32
```

### Prior rung: contrast-packed C2 provider

The new C2 contrast-packed provider rung tested a genuine dense modal
bottleneck:

```text
all 64 source modes
  → learned 64→r encoder
  → causal rank-r executor
  → learned r→64 decoder
  → all 64 target modes
```

This is not prefix deletion: every rank-8, rank-16, and rank-32 candidate can
mix nonadjacent input modes and reconstruct every output mode. The exact
gain-null coordinate is omitted structurally. A disjoint held-out development
selection panel was opened only after all three candidates were fitted and
frozen.

The first C1 calibration pilot failed closed: at its maximum tested amplitude
of `2`, no rank band reached the unchanged `0.02` effect floor, so neither C1
fit nor C1 selection was opened. Fresh C2 pilot identities and a preregistered
`2/4/6/8/12` grid retained the same gates; the smallest eligible global
amplitude was `8`.

Before selection was materialized, an implementation audit also found that
endpoint subtraction was not the required hidden-to-provider-chart tangent.
The final fit was rerun with the exact midpoint chart JVP; the stale endpoint
approximation was never used to score selection.

No candidate passed the combined gate:

| latent rank | stored scalars | reduction vs prior dense-64 provider | canonical MACs, `B=1`, `S=128` | ordinary gate | null / radial / signed passes |
|---:|---:|---:|---:|---|---:|
| 8 | `1,980` | `86.84%` | `893,216` | pass | `24/24`, `12/16`, `0/7` |
| 16 | `4,276` | `71.58%` | `1,315,072` | pass | `24/24`, `13/16`, `0/7` |
| 32 | `8,676` | `42.34%` | `1,874,688` | pass | `24/24`, `7/16`, `0/7` |

Rank 8 and rank 16 use `52.35%` and `29.85%` fewer declared canonical MACs
than rank 32. All three candidates passed ordinary full-target fidelity,
support, causality, padding, repeatability, and prepared-execution gates, and
all three preserved the structurally exact gain null. Radial recovery was
partial and signed recovery failed for every teacher-qualified pair, so the
formal outcome is `no_candidate_passed_combined_gates`.

The curve is nonmonotonic: rank 16 was the best radial candidate, while rank
32 was worse despite having more latent capacity. The fitted loss explains
one likely cause. After applying the declared weights, the final nonpointwise
contrast contribution was only `0.950716`, `0.425144`, and `0.507412` against
pointwise contributions of `26,394.44`, `12,828.99`, and `12,016.03`.
Pointwise fidelity therefore occupied `99.9958–99.9967%` of the final
objective. The next development rung should rebalance or stage contrast
optimization before changing the packing hypothesis.

This is held-out **development selection**, not V4 and not a validation
result. It makes no prompt, natural-language NLL, full-model replacement,
whole-model compression, or latency claim. The synthetic provider panels were
prompt-free, but the frozen upstream Fisher basis remains prompt-derived.

[![C2 contrast-packed provider development rate curve, ordinary fidelity, and contrast recovery](docs/images/reference-provider-contrast-packed-development.svg)](docs/images/reference-provider-contrast-packed-development.svg)

### Frozen V2/V3 baseline

The exact frozen 910-scalar rank-8 reference provider passed every ordinary
fidelity, support, and structural gate again on a fresh 48-probe V3 panel, but
did **not** pass the new contrast assessment. Radial sensitivity was fully
identified and failed candidate recovery; intended-null controls were fully
valid but failed on five of twelve pairs; signed sensitivity identified only
one of four pairs and did not cover the discarded rank stratum. The formal
one-shot outcome is therefore `panel_inconclusive_sensitivity`, not a provider
pass.

This is a useful narrowing. The provider retains strong ordinary synthetic
fidelity, while the new difference-level tests show that matching absolute
targets is not enough to preserve small, meaningful changes. No prompt text,
token IDs, tokenizer, natural activation rows, or prompt-local kernels were
used by the provider.

The v1 executor mixed every eligible earlier source by summation. Its errors
accumulated after source activity began, and no rank passed the per-probe tail
gate. V2 adds one learned source score to the shared causal pair state and
factorizes each edge as

\[
p(\text{source}\mid\text{query})\,
p(\text{expert}\mid\text{query},\text{source}).
\]

The source probabilities are masked and normalized only over eligible earlier
positions. This adds 16 scalars to each candidate, preserves exact causality
and padding behavior, and leaves the original summed-source executor available
for legacy artifacts.

All 80 fit probes were held fixed. Selection used 32 genuinely fresh probes
with zero shared hashes or direction seeds versus v1. The smallest passing
candidate was `spectral-r08-t08`:

| metric | fresh selection | sealed assessment |
|---|---:|---:|
| Fisher-weighted relative error | `0.08263` | `0.05900` |
| Reference cosine | `0.99659` | `0.99826` |
| Maximum per-probe p90 error | `0.28775` | `0.28970` |
| Worst family relative error | `0.09940` | `0.09502` |
| Error reduction vs constant | `58.11%` | `47.80%` |
| Error reduction vs position-only | `58.12%` | `47.92%` |

The provider stores 910 scalars versus 15,046 for the dense-64 provider
(`93.95%` fewer). Under the declared ideal mathematical accounting, it uses
`87.16%`, `79.54%`, `71.28%`, and `58.65%` fewer provider MACs at sequence
lengths `32`, `72`, `128`, and `256`. Those counts exclude activation,
softmax, masking, additions, memory traffic, and surrounding Gemma work; they
are neither whole-model compression nor latency measurements.

The preregistered composite assessment is still formally **failed**. Its only
false gate was a teacher-panel identifiability control, not a prediction
metric: the minimum true-target difference across collision groups was
`9.24e-6` against a required `0.01`. All four radial-scale groups cleared the
threshold, while all eight axis-sign and all four gain-null groups did not.
The clean conclusion is therefore:

- sealed prompt-blind provider fidelity is positive across sparse, chirp,
  axis, radial-collision, and null-collision families;
- the current collision panel cannot establish a 1% downstream effect for
  every tagged variable; and
- prompt-independent basis discovery, natural-prompt transfer, NLL,
  whole-model replacement, compression, and speed remain unproven.

The upstream Fisher basis is itself prompt-derived. “Prompt-blind” here means
provider relation mapping after that basis was frozen, not prompt-independent
basis discovery.

### What the backward trace found

An authenticated retrospective diagnostic reexecuted all 40 collision
endpoints, reproduced every target hash from the consumed assessment, analyzed
all 32 unordered pairs, and marked the exact 16 group-minimum gate witnesses.
It then traced each finite contrast through the manifold lift, L4 attention
prefix, residual merge, pre-FF normalization, and Fisher-weighted target using
midpoint JVPs and contrast-aligned VJPs.

The result localizes the **test failure before candidate tracking**:

- all 12 below-threshold witnesses were numerically resolved, but their true
  teacher contrasts were still smaller than the frozen `0.01` gate;
- mean-reference injection was a dominant relative-contrast dilution on
  `10 / 16` witnesses, including two radial witnesses that nevertheless
  passed, so it is a shared property rather than a sufficient failure cause;
- pre-FF normalization was the only observation exclusive to failed
  witnesses, appearing on all `4 / 4` gain-null groups;
- the axis groups had no failure-exclusive checkpoint bottleneck—their small
  effect remained distributed through the teacher path;
- no failed witness showed residual/attention cancellation or a retained
  Fisher-subspace miss; the retained 64 modes captured at least `99.9988%` of
  the full Fisher-weighted contrast energy; and
- JVP/VJP adjoint error was at most `1.77e-6`, with zero response before the
  changed source position.

This does not rehabilitate v2 or show candidate contrast recovery: candidate
predictions never enter the collision metric. The diagnostic consumed only the
already-opened panel, made no refit or reselection, and cannot become compiler
input. Any repair still requires a genuinely fresh sealed v3 assessment.

[![Retrospective collision attenuation trace showing teacher contrast ranges and localized observations](docs/images/reference-provider-collision-attenuation.svg)](docs/images/reference-provider-collision-attenuation.svg)

### What the fresh V3 assessment found

V3 authenticated the exact frozen `spectral-r08-t08` artifact, its selected
plan, controls, training protocol, Fisher basis, source model, scoring code,
and all 48 new probe identities before creating an irreversible claim. Only
after that claim did it materialize 16 ordinary-fidelity probes and 32
contrast probes spanning 12 groups and 24 preregistered pairs. There was no
refit, reselection, parameter change, retry, or threshold override.

Ordinary full-target fidelity remained positive:

| metric | fresh V3 result |
|---|---:|
| Fisher-weighted relative error | `0.06773` |
| Reference cosine | `0.99772` |
| Maximum per-probe p90 error | `0.29138` |
| Worst family relative error | `0.08704` |
| Error reduction vs constant / position-only | `45.61%` / `45.34%` |
| Support / prepared parity | `1.0` / `3.26e-8` |
| Causality / padding / repeat violations | `0 / 0 / 0` |

The contrast result was materially different:

| family | teacher-qualified | candidate passes | result |
|---|---:|---:|---|
| Radial sensitivity | `8 / 8` | `0 / 8` | candidate failure; macro relative error `0.9300`, minimum cosine `0.4342` |
| Signed sensitivity | `1 / 4` | `0 / 1` | panel-inconclusive; discarded-stratum sensitivity was not established |
| Intended null | `12 / 12` | `7 / 12` | candidate failure; maximum null effect/error upper bounds `0.02533 / 0.02533` against `0.01` ceilings |

The signed family makes the whole panel inconclusive before a clean composite
candidate-failure verdict can be issued, but it does not erase the identified
radial and null failures. Weak teacher contrasts never entered candidate
relative metrics, and intended nulls never entered direction metrics.

The ignored result is bound by implementation bundle
`af06c779c18bf9bc860ca4683ed37c93a0954f090411c544b8062ddfa29086a0`,
protocol
`65959324d2815621a1d6420bdb4d41a9db74c4214205088da9545088bc19ce03`,
panel
`919126906cc6f07074d76599843504ea81462485e8f93ee6d35c71732979249e`,
logical artifact
`60e83fa843e4a2878f597f0f924e736d83b4165b2bdbb3bd40aab0ca24905594`,
and report
`df4562f976ae903fc89d6d299b4cb3fbd771f99b28e28717d545d9fdb48f0392`.
The local tensor/report artifacts contain no prompts, token IDs, model state,
provider parameters, or raw teacher/candidate tensors and remain ignored.

[![Fresh V3 assessment showing ordinary fidelity passing while radial and null contrast behavior fails](docs/images/reference-provider-v3-assessment.svg)](docs/images/reference-provider-v3-assessment.svg)

### What this builds on

The compact bilinear modal-generator branch passed its earlier frozen,
no-refit Gemma assessment.

The preceding mixed-mode rung had falsified a cross-free linear-plus-diagonal
executor: at a fresh origin, measured mode interaction accounted for `11.27%`
of response norm and a truth-leaking interaction oracle reduced error by
`23.10%`. Crucially, `80.74%` of that interaction energy lived in the
quadratically scaling odd-odd component \(C_{11}\). That made an explicit
off-diagonal product branch a testable repair rather than an unconstrained
nonlinear fit.

The compiler now maps the eight nominated sensitive modes to all 28 canonical
products

\[
\phi_{ij}=2(\Delta m_i/\sigma_i)(\Delta m_j/\sigma_j),\qquad i<j,
\]

then transports those features through a position-conditioned causal spectral
generator. It measured fit origins `8/24/40`, selected rank only at origins
`16/32`, sealed the resulting candidate, and opened origin `20` only in the
separate assessment command. Origin `28`, used to nominate the architecture,
was never reused for fitting.

The smallest passing plan was rank `8×8`:

- `6,880` stored coefficients versus `172,032` for the matched dense bilinear
  family—`96.00%` fewer;
- selection error `0.20726 → 0.16852`, an `18.69%` reduction, at cosine
  `0.98729`;
- fresh origin-20 error `0.20901 → 0.16937`, an `18.96%` reduction, at cosine
  `0.98710`;
- fresh \(C_{11}\) relative error `0.22976` and cosine `0.97406`; and
- `94.10%` recovery of the measured \(C_{11}\) oracle headroom on the sealed
  assessment.

The rank curve is strongly compressible: the direct dense branch reaches
selection error `0.16625`, only `0.00227` below the selected rank-`8×8`
branch. Six fresh control pairs passed the frozen structural-control gates
(pooled leakage `0.0562`; worst reliable pair `0.0822`), the branch is exactly
zero on singleton axes, and the prepared float32 graph matched the analytic
implementation to `1.47e-7` relative error on assessment.

Combined with the existing linear and diagonal branches, the compiled edge
stores `46,816` coefficients versus `958,464` for the matched dense
three-branch family (`95.12%` fewer). This is the first positive no-refit
evidence that a compact generator can transport known mixed-mode edges across
positions. It is not yet whole-model compression: the prompt-conditioned
reference provider and surrounding Gemma weights are excluded, only known
pair identities were tested, and no NLL, task-accuracy, or latency claim has
been made.

[![Frozen bilinear spectral assessment showing coefficient compaction and held-out-origin fidelity](docs/images/bilinear-spectral-assessment.svg)](docs/images/bilinear-spectral-assessment.svg)

The earlier rank-only failure that led to the conditional and bilinear
branches remains useful context:

[![Rank 64 versus rank 128 diagnostic showing better reconstruction but worse finite transport](docs/images/l3-l4-rank-diagnostic.svg)](docs/images/l3-l4-rank-diagnostic.svg)

The figures are generated from the committed
[`source-safe research summary`](artifacts/research/current_research_summary_v1.json),
which binds the underlying report digests without committing prompts, token
IDs, model weights, or tensor artifacts. Tests reject stale SVGs.

The next experiment is therefore:

1. preserve both consumed panels and their outcomes—do not rerun V2 or V3,
   weaken their thresholds, or refit against their targets;
2. use the opened V3 result only to localize why the frozen provider misses
   radial differences and leaks intended-null changes;
3. revise the provider architecture on fit/selection data and strengthen the
   signed-sensitivity construction so both rank strata become testable;
4. freeze that new candidate and preregister a genuinely fresh V4 panel with
   new modes, positions, lengths, seeds, hashes, and one-shot identity;
5. only if V4 passes, compose the provider with the linear, diagonal, and
   bilinear branches and run source-authoritative shadow execution on a
   family-disjoint natural-prompt split, scoring NLL, full-vocabulary KL, and
   top-1 agreement; and
6. only after downstream fidelity passes, measure resident storage, active
   compute, and end-to-end latency.

This work is described in
[`docs/recursive-modal-hierarchy.md`](docs/recursive-modal-hierarchy.md).

## What has actually worked

| Rung | Resource result | Fidelity result | Claim |
|---|---|---|---|
| Toy V2 whole-span executor | `31.60%` fewer deployment parameters; ideal matrix work is `60.70%` of native | Exact fresh-validation task behavior; reserved executor test remains sealed | Validation-backed structural compression on the toy task |
| Toy fused executor | Dense fused path executes `35.3%` of the replaced blocks' reference multiplies | Exact validation argmax; NLL delta `1.86e-7` | Verified runtime reference |
| Gemma structured layer | No compression; native-shaped layer is retained | Validation block NRMSE `9.21e-7`, top-1 `1.0`, zero source-layer calls | Replacement interface and activation-only fitting proven |
| Gemma 18-generator MLP stack | `20.90%` logical whole-model parameters saved; `79.17%` native MLP matrix MACs removed | After trajectory refit: delta NLL `+0.149649`, native top-1 `81.02%` | Open-development rate/distortion point, not accepted compression |
| Gemma L3→L4 hierarchy, rank 64 | Pair state is `11.4%` of the flat pair; nominal saving is only `0.685%` of the flat-generator whole model and excludes the reference provider | Local-control cosine `0.763`, relative error `1.187` | Analysis only; finite transport fails |
| Gemma L3→L4 spectral map, rank 64 | Source-σ-weighted ranks are `11 / 18 / 34` at `90% / 95% / 99%` energy; no deployed reduction | Local-to-`1σ` mean cosine `0.9996`; two-origin mean similarity `0.672` | Prompt-free fixed-reference analysis only; position-conditioned |
| Gemma conditional spectral modal-delta executor | `39,936` edge coefficients versus `786,432` for a matched dense two-branch family (`94.92%` fewer); provider and model excluded | Fresh origin-20 local cosine `0.9819`; diagonal correction reduces finite error `0.2278 → 0.2006` | Prompt-free fixed-reference interior interpolation only; no-refit assessment |
| Gemma mixed-mode chord assessment | No deployed reduction; frozen candidate unchanged | Fresh origin-28 error `0.1863`, cosine `0.9834`; cross nonadditivity `11.27%`; interaction-oracle gain `23.10%` | Diagonal-only correction materially falsified; compact bilinear branch nominated |
| Gemma bilinear modal-generator executor | Bilinear branch stores `6,880` coefficients versus `172,032` dense (`96.00%` fewer); all three edge branches store `46,816` versus `958,464` matched dense (`95.12%` fewer) | Fresh origin-20 error `0.2090 → 0.1694` (`18.96%` reduction), cosine `0.9871`; recovers `94.10%` of \(C_{11}\) oracle headroom | Positive no-refit mixed-mode edge transport; fixed-reference and known-pair scope only |
| Gemma prompt-blind reference provider V2/V3 | Rank 8 stores `910` scalars versus `15,046` for the full-width provider (`93.95%` fewer); provider-only ideal MAC savings are sequence-dependent | Fresh-V3 ordinary error `0.0677`, cosine `0.9977`, p90 `0.2914`; all ordinary fidelity/structure gates passed | Radial and intended-null contrast recovery failed; signed sensitivity was underpowered, so the formal V3 outcome is panel-inconclusive |
| Gemma C2 contrast-packed provider development | Ranks `8/16/32` store `1,980/4,276/8,676` scalars (`86.84%/71.58%/42.34%` below the prior dense-64 component); canonical rank-8/rank-16 MACs are `52.35%/29.85%` below rank 32 | Every rank passed ordinary fidelity and `24/24` exact-null pairs; radial passes were `12/16`, `13/16`, `7/16`, while signed passes were `0/7` at every rank | Held-out development selection only; no candidate passed, V4 remains unopened |
| Gemma rank-16 objective-balance diagnostic | Same `4,276`-scalar candidate form; no new resource or deployment claim | Unit-RMS treatments passed `12/12` ordinary, `24/24` null, and `16/16` radial fit checks, but only `2–3/8` signed checks | Fit-only diagnostic; global loss scale is not the sole blocker and C3 remains unopened |
| Gemma rank-64 capacity control | `19,012` stored scalars and `3,190,528` canonical MACs versus rank 16's `4,276` and `1,315,072`; no reduction claim | Descriptively `12/12` ordinary, `24/24` null, `16/16` radial, and `3/8` signed; ordinary error `0.00672074`, cosine `0.99997745` | Invalid comparison: initial pointwise share missed the frozen balance gate, so no capacity conclusion, replication, width ladder, or C3 is authorized |
| Gemma function-preserving width control | Rank 64 uses `19,012` scalars / `3,190,528` canonical MACs versus rank 16's `4,276` / `1,315,072` (`4.446×` storage and `2.426×` MACs); no reduction claim | Valid matched start: both passed ordinary, null, and radial gates but only the same `3/8` signed identities; rank 64 improved ordinary error `0.00769406 → 0.00652994` while signed macro error changed `0.380298 → 0.386891` | Outer width alone is insufficient under the matched fit budget; expert/core control authorized, with no replication, width ladder, C3, or compression claim |

There are three important distinctions:

- The toy system proves that Fisher modes can become a real graph executor.
- The exact Gemma layer proves that the model adapter and replacement boundary
  can reproduce a native layer without calling it.
- The Gemma compression and hierarchy rungs have not yet met a
  source-authoritative downstream-fidelity gate.

## Current Gemma baseline

The flat full-stack compiler replaces all 18 Gemma MLPs with rank-640 affine
generators while leaving attention, normalization, embeddings, and the
language-model head native.

The sequential compiled-trajectory refit reduced the original generated
stack's excess NLL by `57.1%`, from `+0.348476` to `+0.149649`, at the same
logical resource budget. A prepared CPU runtime then measured:

| Path | Factorized speedup | Fused speedup |
|---|---:|---:|
| Prefill, context 32–256 | `1.42–1.62x` | `1.50–1.73x` |
| Cached decode, context 32–256 | `1.21–1.24x` | `1.26–1.28x` |

Those are real batch-one PyTorch/CPU timings on an open-development prompt,
not a GPU result, confidence interval, or downstream-quality-qualified
deployment. The fused path is also a distinct float32 rate/distortion point:
precomposing its factors changes operation order and worsens end-to-end
fidelity.

See
[`docs/modal-generator-compiler.md`](docs/modal-generator-compiler.md) and the
committed
[`source-safe runtime report`](artifacts/gemma3_runtime/full_model_runtime_analysis_dev_v1.json).

## L3→L4 evidence, interpreted narrowly

The live hierarchy rung measures:

- joint activation covariance across the L3 output and L4 input;
- valid-position score-gradient Fisher induced by summed prompt NLL;
- Fisher-balanced restriction and prolongation factors;
- a literal-zero topology tear and a mean-source execution reference;
- signed causal JVP kernels for logical lags 0 through 4; and
- dense, factorized, and staged execution of the bound prompt-local pair plan.

The observed activation cross-coupling is strong, and about `21.6%` of the
rank-64 kernel energy lies at positive logical lags. That supports real
fan-out beyond a same-token map. The near-one cross-Fisher value is
chain-related because both sites derive from the same sequence NLL; it is not
semantic equivalence or replacement fidelity.

The reference-base tensor is still produced by the frozen transformer
boundary. The runtime authenticates the factors, mean, positions, mask, and
JVP artifact, but it cannot yet authenticate that external provider. Pair
parameter and MAC reductions are therefore shape-only opportunities, not
achieved model compression or speedups.

## Verified toy reference

The toy transformer is intentionally small enough to inspect completely. It
includes activation capture, empirical activation-space Fisher matrices,
compute modes, interventions, position-conditioned execution, conditional
completion, independently compiled layers, algebraic fusion, lazy
instrumentation, a packed triangular reference, and an optional MLX/Metal
lowering.

The clean V2 whole-span executor replaces all three source blocks, makes zero
native-block calls, and preserves exact behavior on 246 fresh validation
contexts. Its compiled deployment contains 19,064 parameters versus 27,872
native (`31.60%` fewer), and its ideal complete matrix work is `60.70%` of
native. The reserved 250-context executor test remains hash-only, and the task
is narrow query-sparse associative recall—not language modeling. See the
[clean V2 protocol](docs/conditional-computation.md#v2-clean-expanded-task-replication).

[![Authenticated toy optimization summary comparing arithmetic, CPU latency, and resident storage](docs/images/fused-executor-optimization.svg)](docs/images/fused-executor-optimization.svg)

The figure is regenerated from
[`artifacts/associative_recall/fused_executor_report.json`](artifacts/associative_recall/fused_executor_report.json);
the test suite rejects it if the committed SVG becomes stale. The packed
triangular implementation is a measured PyTorch reference, not the default
backend or a generally faster GPU kernel.

Detailed toy reports and replayable tensors live under
[`artifacts/associative_recall/`](artifacts/associative_recall/). The previous
full README, including the complete experiment chronology and command
catalog, is preserved as [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## Architecture

The code is organized around explicit compiler boundaries:

1. **Model adapters** describe layers, activation sites, parameters, masks,
   logical positions, cache semantics, dtype, and device policy.
2. **Instrumentation** captures selected activations and score gradients
   without changing source weights.
3. **Analysis** builds Fisher/covariance moments, modes, causal JVP edges, and
   guarded rate/distortion measurements.
4. **Compilation** lowers authenticated modes and edges into graph plans with
   explicit means, restrictions, prolongations, causal schedules, and
   provenance.
5. **Execution** runs dense controls, factorized graphs, staged sessions, or
   source fallbacks through a common replacement boundary.
6. **Validation** separates fit, selection, assessment, and reserved roles and
   fails closed before stronger claims.

The complete interface and scaling design is in
[`docs/compiler-architecture.md`](docs/compiler-architecture.md).

## Quick start

The core repository requires Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest
fisher-graph-experiment
fisher-graph-verify
```

Regenerate the committed research figures:

```bash
fisher-graph-plot-research
fisher-graph-plot-optimizations
```

Gemma experiments are opt-in. Accept the model license on
[Hugging Face](https://huggingface.co/google/gemma-3-270m), keep the model in
an external cache, and install the adapter dependencies:

```bash
pip install -e ".[dev,gemma]"
hf auth login
```

The current hierarchy command requires the frozen full-stack base/refit
artifacts described in the recursive-hierarchy documentation:

```bash
fisher-graph-gemma-l3-l4-hierarchy-dev \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1

fisher-graph-gemma-l3-l4-spectral-dev \
  --sequence-length 72 \
  --all-source-modes \
  --impulse-positions 8,16,24,32,40 \
  --max-lag 31 \
  --fft-length 64

fisher-graph-gemma-l3-l4-conditional-spectral-dev compile

fisher-graph-gemma-l3-l4-mixed-mode-dev

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

fisher-graph-gemma-l3-l4-contrast-provider-dev describe

fisher-graph-gemma-l3-l4-contrast-provider-dev compile \
  --device cpu \
  --dtype float32

fisher-graph-gemma-l3-l4-objective-balance-dev describe

fisher-graph-gemma-l3-l4-objective-balance-dev run \
  --device cpu \
  --dtype float32
```

No Gemma weights or local tensor artifacts are committed. Development runs
default to the ignored `.local-runs/` tree.

## Documentation

### Start here

- [Compiler architecture](docs/compiler-architecture.md)
- [Recursive modal hierarchy](docs/recursive-modal-hierarchy.md)
- [Full research log and command archive](RESEARCH_LOG.md)

### Current generator research

- [Modal-generator compiler](docs/modal-generator-compiler.md)
- [Generator causal fingerprints](docs/generator-causal-fingerprints.md)
- [Generator causal map](docs/generator-causal-map.md)

### Compression and conditional-compute experiments

- [Structured Gemma layer executor](docs/structured-layer-executor.md)
- [Dense supermode compaction](docs/dense-supermode-compaction.md)
- [Cross-block selective bundling](docs/cross-block-selective-bundling.md)
- [Fisher-need conditional computation](docs/conditional-computation.md)

### Earlier Gemma foundations

- [Gemma 3 270M experiment archive](docs/gemma3-270m.md)
- [Weighted-Jacobian compilation](docs/weighted-jacobian-compilation.md)
- [Residual-separated gated executor](docs/gated-executor.md)

## Scientific claim boundaries

The repository uses deliberately narrow status language:

- **Verified reference** means the committed toy artifact passed its declared
  equivalence and replay controls.
- **Parity** means a candidate reproduces a source boundary but may save
  nothing.
- **Open development** means the data helped choose the next experiment and
  cannot serve as fresh confirmation.
- **Shape-only** means parameter or MAC counts omit an uncompiled provider,
  router, kernel, or other required runtime work.
- **Rejected** means the tested candidate failed its declared gate; it does not
  prove the entire method impossible.
- **Measured latency** is always scoped to the reported device, backend, shape,
  batch, and timing protocol.

Validation, reserved test data, model weights, prompt text, token IDs, and
large tensor artifacts are not silently promoted into committed evidence.
Source-safe reports retain aggregates, hashes, provenance, and claim
boundaries.

## Repository layout

```text
src/fisher_graph/        compiler, analysis, runtimes, and experiment CLIs
tests/                   unit, artifact, replay, and stale-figure checks
docs/                    architecture and experiment-specific protocols
docs/images/             deterministic source-backed SVG summaries
artifacts/               committed toy artifacts and source-safe reports
examples/                prompt/split scaffolding and toy examples
.local-runs/             ignored local Gemma tensors and reports
```

The project is currently testing whether nonlinear conditional transport can
turn measured cross-layer modal structure into a self-contained, faithful
Gemma graph. That is the gate before another rank ladder or compression claim.
