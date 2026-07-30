# Residual-guided progressive compilation

The progressive compiler treats an approximate modal graph as iteration zero,
not as an all-or-nothing replacement. It repeatedly maps the remaining
source-to-candidate residual, proposes a bounded structural repair, measures
the result, and advances only the accepted candidate head. Once fidelity
passes, the same controller changes from repair to compaction.

The implementation is split between:

- `fisher_graph.compiler.progressive`, the model-independent state machine;
- `fisher_graph.gemma3_l3_l4_progressive_compilation`, the prompt-blind
  binding for the current Gemma L3/L4 rank-64 seed; and
- `fisher_graph.gemma3_l3_l4_progressive_worker`, the A-only Gemma worker
  which materializes panels, executes source-authoritative measurements, and
  maps complete-boundary residuals; and
- `fisher_graph.gemma3_l3_l4_two_head_lowerer`, the candidate-bound X4/H4
  causal residual fitter and one-forward prefill executor.

## Development and assessment boundaries

Repeated development uses three pairwise family-disjoint Calibration-A
roles:

| Role | Reusable? | Purpose |
|---|---:|---|
| `calibration_a_fit` | yes | NLL-VJP-aligned residual mapping and parameter fitting |
| `calibration_a_selection` | yes | choose among preregistered repair or compaction proposals |
| `calibration_a_guard` | one manifest-global claim | final veto after the complete challenger lineage is frozen |

The selection split is intentionally named as such. Repeatedly selecting
against it makes it development data, even though it is family-disjoint from
fit. The final A guard is supplied only after no more graph mutations can
occur, and its evaluation binds the already-frozen challenger receipt. A
failed guard terminates the campaign; the runner does not fall through to the
second-best selection candidate. The Gemma worker wraps this callback in a
durable claim-first ledger keyed by the A-guard manifest. A new protocol or
process cannot silently reuse the same guard.

Calibration B is not a fourth callback. Its frozen manifest SHA is registered
in the progressive protocol only so every A role can reject it. A passing A
handoff contains no B loader, prompt, example identity, manifest, observation,
or evaluation result.

```text
Calibration-A fit
    ↓
residual map
    ↓
repair proposals ──→ A-selection evaluation ──→ accepted active head
    ↑                                                │
    └──────────────── repeat while fidelity fails ───┘

accepted head passes fidelity
    ↓
compaction proposals ──→ A-selection evaluation ──→ smaller active head
    ↑                                                   │
    └──────────────── repeat while Pareto-positive ─────┘

freeze complete lineage
    ↓
one A-guard veto
    ↓
development-only frozen handoff
    ↓
separate candidate-bound Calibration-B protocol
```

## Acceptance policy

Protocol v2 makes two claims explicit instead of folding incompatible metrics
into one maximum:

1. **Candidate execution fidelity** is the deployment-behavior gate. It uses
   actual candidate logits: absolute NLL change, KL, aggregate top-1
   agreement, adverse-tail NLL change, and adverse-tail top-1 agreement.
2. **Structural/modal fidelity** remains fully measured and reported:
   operator NRMSE, X4 boundary geometry, projection capacity, projection
   oracle behavior, and exact-carrier oracle behavior. These characterize the
   ancestor mapping and are not changed by both residual heads, so they cannot
   honestly veto a behavior-only candidate class. A passing result would be
   called `candidate_execution_fidelity_only`, never full structural
   equivalence.

The forced Gemma staging transition is separately preregistered. X4 must
improve both aggregate and worst-family boundary relative error by at least
2%, preserve all 18 implementation-frozen diagnostic axes exactly, fit the
declared X4 site/rank count, and remain inside the full resource budget.
Candidate behavior and the two boundary-cosine axes may temporarily regress
because H4 is the required next step. Accepting X4 creates a state-machine
debt: it cannot compact, freeze, or open the guard. The next iteration must
remeasure X4 and propose the declared H4 site/rank count. H4 is judged against
the pre-X4 selection anchor, not the temporarily degraded X4 child.

Outside a staged transition, repair uses normalized worst-axis execution
burden. A repair must reduce that burden by the frozen minimum and cannot move
another execution axis beyond its allowed regression envelope.

After every execution-fidelity axis passes, a compaction is eligible only when:

1. every fidelity gate remains passed;
2. learned parameters, runtime parameter bytes, and logical MACs/token are
   all non-increasing; and
3. at least one of those resource axes strictly decreases.

The deterministic repair tie-break is fidelity burden, MACs, parameters,
bytes, then candidate ID. The compaction tie-break is MACs, parameters, bytes,
fidelity burden, then candidate ID.

The result retains every proposal, built candidate, and scalar selection
evaluation, including dominated and rejected points. This is the raw
rate-distortion archive; the accepted lineage is a view over it, not a
replacement for it.

## Full resource accounting

Every proposal and built candidate carries the same exact resource footprint.
Each resource axis is divided into:

- compiled graph work;
- retained source-model work; and
- support work such as carrier transforms, routing, lookup, or normalization.

The controller charges all three. Every receipt also binds the candidate
execution, accounting artifact, parameter/compute scopes, runtime and dtype,
and sequence scope. A source island is therefore a legitimate temporary
repair, but it is never free. Hard budgets limit both total cost and the
retained-source fraction, and incomparable scopes fail closed.
The protocol pins the seed's complete resource-footprint receipt, so changing
even one baseline cost or scope creates a different campaign rather than a
quietly revised iteration zero.

Resource receipts also have a `cost_complete` bit plus canonical incomplete
reasons. A candidate with omitted router operations, fallback execution, or
another unknown dimension is rejected before it is built or evaluated. This
does not make caller-supplied accounting authoritative; each model-specific
builder still needs to recompute these numbers and the accounting-artifact
hash from its immutable executable.

## Immutable lineage

The seed artifact, execution, runtime-binding, and resource-footprint hashes
are frozen into the protocol and checked against the seed candidate. Each
residual map binds the active candidate receipt and fit manifest. Each
mutation proposal binds:

- the current active candidate artifact and receipt;
- the exact residual-map receipt;
- the selected residual ranks;
- a mutation recipe SHA; and
- the complete proposed resource footprint.

The built candidate must bind that proposal, preserve a contiguous iteration
number, produce a new artifact hash, and match the proposal's resources.
Selection measurements bind the candidate, selection manifest, protocol, and
resource footprint. Iteration receipts form an active-head chain; a rejected
iteration cannot be followed by another transition, and stale or reordered
heads fail closed.

The current implementation recomputes the complete archived transition chain
before emitting a handoff: protocol-pinned seed, phase choice from the active
head's fidelity, legal loop termination, residual maps, proposal membership,
built candidates, complete manifest/family coverage, deterministic selection,
and the guard-bound frozen challenger. The handoff exposes the accepted
candidate's runtime-binding hash directly for the next candidate-bound
protocol. Before
parallel or distributed campaigns, these immutable in-memory receipts should
be backed by an append-only compare-and-swap session ledger so two workers
cannot publish different children of the same accepted head.

## Gemma iteration zero

`make_gemma3_l3_l4_progressive_protocol` reads only prompt-blind identities
from the existing frozen shadow protocol. It binds:

- source model
  `7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9`;
- rank-64 seed artifact
  `b3e011d8067ff3538888851c476fba03c57f4e9f172f923c20fdd90ac0799f84`;
- factorized refit execution
  `911f9869077be1fec2f8610f2f2cbe4c5c6e01a8d632573bec52f2fcc12d1df9`;
- the complete legacy runtime binding plus basis, plan, tensor-file, live
  execution, and shadow-protocol lineage; and
- forbidden Calibration-B manifest
  `986ee9da505fb056853f4fc7ed4f5eee6e9313f0419f2ca9ebc54e0df8607bdd`.

The numerical targets reuse all existing frozen behavioral, boundary,
full-width projection-capacity, worst-family, coverage, signal, and
carrier-completeness thresholds. Only actual candidate behavior is an
execution-qualification gate; the other axes remain visible structural
diagnostics. The caller must supply authenticated
seed resource accounting, source-baseline resource totals, and three new
A-only manifests with pairwise-disjoint families.

The legacy one-shot executor remains frozen to the failed rank-64 candidate.
This is intentional: a progressive winner must not inherit that candidate's
protocol identity or development evidence. The next integration boundary is
an instance-bound shadow protocol/runtime whose payload includes the accepted
A transcript and the new candidate, basis, plan, source-model, and execution
hashes. The host-global B ledger should remain keyed by the B manifest so only
one eventual winner can consume it.

## Implemented Gemma worker rung

`Gemma3L3L4ProgressiveWorker` now implements the safe first half of the
model-specific data plane:

1. it requires exactly three A-only panels whose examples, inputs, manifests,
   and families do not overlap;
2. it rejects the registered Calibration-B manifest and never receives a B
   loader;
3. its legacy executable runs the authenticated rank-64 shadow plus projection
   and exact-X4 carrier oracles for source-authoritative selection metrics;
4. on A-fit only, it additionally captures the native and candidate
   `layer.4.output` boundary, differentiates native next-token NLL with respect
   to that boundary, and verifies that exact H4 replacement restores native
   logits;
5. it builds family/example-balanced residual covariances, tilts each row by
   `1 + cosine(residual, NLL gradient)^2`, canonicalizes the leading residual
   PCA directions, and records post-hoc activation-gradient-Gram Rayleigh
   scores before emitting scalar/hash-bound `ResidualMap` receipts; and
6. it requires an external durable claim-first authority before opening the
   final A guard.

This does **not** turn the legacy runtime into a compiled deployment. Its
three shadow passes, two X4 oracle passes, extra fit-only boundary/NLL-gradient
passes, native fallback, and retained source model are measurement costs.
The seed resource receipt should therefore remain `cost_complete=False`.
The concrete legacy executable enforces the canonical incomplete reasons
`multi_pass_shadow_measurement`, `native_boundary_fallback`, and
`no_one_pass_serving_executable`; caller-supplied "complete" seed totals are
rejected.
The generic field named `jvp_gain` is explicitly zero in this rung: the worker
has measured bounded NLL-VJP alignment and activation-gradient-Gram coupling,
not a Fisher eigendecomposition or held-out JVP directional transport.

The campaign now has a separate frozen three-way A fit/selection/guard corpus.
Fit and selection are reusable adaptive-development roles. The original pilot
guard was rotated after an out-of-band audit read; the replacement is
entropy-generated, prompt-private, committed by hash, and still unclaimed.
Calibration B was not repartitioned or opened.

## Implemented candidate-bound X4/H4 rung

The worker intentionally raises
`GemmaMutationLoweringUnavailableError` if proposal or build callbacks are
used without a registered lowerer.
`GemmaL3L4TwoHeadMutationLowerer` now supplies that candidate-bound
implementation:

1. the legacy A-fit probe exports an authenticated one-pass bridge and records
   full L3 source modes, logical positions, support masks, native and same-pass
   candidate X4/H4 boundaries, and native X4/H4 NLL gradients; after X4 is
   accepted, an optional fit-only pass detaches its realized H4 boundary,
   measures the candidate-conditioned H4 NLL VJP, and proves the detached
   execution is bitwise-identical to an ordinary candidate pass;
2. the worker ranks separate X4 and H4 residual-PCA directions using the
   bounded NLL-VJP alignment tilt above and reports post-hoc
   activation-gradient-Gram scores, using the candidate tangent after X4
   whenever it is present;
3. the lowerer builds exact logical-lag designs, applies equal
   family/example/row mass, standardizes columns, and fits positive-ridge
   homogeneous kernels in CPU float64;
4. immutable head artifacts contain an orthonormal decoder, executable
   `[lag, source_mode, head_mode]` kernel, and, for the optional realized-state
   H4 arm, an authenticated `[head_mode, head_mode]` state kernel;
5. the executable bridge performs X3 capture, Y3 clamp, base graph/X4 repair,
   and H4 repair in exactly one prefill forward, passes the immutable
   post-X4/pre-H4 activation to the H4 provider, and has no native-X4
   fallback;
6. child observations still use the legacy source/oracle paths for
   source-authoritative A-selection metrics, but those diagnostic passes are
   not misreported as serving cost; and
7. resource receipts enumerate retained model parameters/bytes/linear MACs,
   bridge tensors and modal MAC upper bounds, residual-head tensors and MAC
   upper bounds, and integer support state.

The campaign schedule is deliberately sequential and protocol-enforced. The
first iteration proposes only X4. If it passes its stage-local gate, the worker
remeasures that exact child, maps its post-X4 H4 residual, and only then
proposes H4. Fitting both heads blindly from the seed would double-count the
nonlinear X4-to-H4 response. After X4 is accepted, the H4 target is conditioned
by remeasurement. The baseline H4 head reads only archived L3 source modes.
The optional realized-state arm also projects current post-X4,
pre-H4-correction state through the existing output decoder and applies a
pointwise state kernel:

```text
q_t = h_t D^T
delta_h_t = [sum_l s_(t-l) K_l + q_t A] D
```

`D` is the rank-`r` output decoder and `A` is an `r × r` state kernel. The
local term reads no future position and runs before its own correction is
applied, so it creates neither a causal violation nor a feedback loop.

The implementation has synthetic causal/ridge recovery tests, a state-only
positive control that requires the new feature, strict state serialization
and activation non-mutation checks, and future-position invariance coverage.
An authenticated tiny-Gemma integration builds X4, remeasures, builds H4,
reproduces the live correction analytically, and verifies that each prefill
execution uses one model forward. This is an integrity-heavy research
executor, not a latency-ready serving runtime:
full-model and tensor fingerprints run around execution, cached autoregressive
decode is not implemented, and the declared linear/modal MAC scope excludes
hashing, Python dispatch, device transfers, temporary memory, and wall-clock
latency. It does not by itself report a real-model fidelity, compression, or
speed win. The executable child remains an overlay on the retained factorized
Gemma carrier. The family-disjoint three-way A campaign has now run; its
result is described below. Calibration B remains unopened throughout.

## Adaptive staged-v2 Gemma result

The CPU/float32 campaign used 8 fit prompts across 4 families, 4 reusable
selection prompts across 4 disjoint families, and a fresh 4-family guard that
would be claim-opened only after selection qualification. It bound the first
pilot report and transcript as adaptive lineage.

| candidate | abs. ΔNLL | KL | top-1 | p90 abs. ΔNLL | params | MACs/token |
|---|---:|---:|---:|---:|---:|---:|
| seed | 0.0937 | 1.3092 | 0.4189 | 0.4333 | 212,479,744* | 212,226,880* |
| X4 rank-8 | 0.1450 | 1.2833 | 0.3962 | 0.3894 | 212,501,248 | 212,248,384 |
| X4 + H4 rank-8 (L3-only input) | 0.3778 | 0.9644 | 0.4792 | 0.7927 | 212,522,752 | 212,269,888 |

`*` The seed is a multi-pass measurement and is marked cost-incomplete.

X4 improved aggregate boundary relative error from 7.1341 to 6.2910 (11.8%)
and worst-family error from 8.4218 to 8.0722 (4.2%), satisfying its staged
gate. H4 reduced the worst execution burden from 26.18 at the seed to 19.29,
principally through better KL and top-1 agreement, but it regressed absolute
NLL change from 0.0937 to 0.3778 and p90 NLL change from 0.4333 to 0.7927.
Those regressions exceeded the preregistered per-axis envelope, so H4 was
rejected.

The terminal status is `stalled_fidelity`. The final active candidate is X4
only; there is no frozen challenger, guard evaluation, handoff, compression
claim, structural-equivalence claim, or latency claim. The replacement guard
remains unopened and unclaimed. Against raw Gemma, the X4 executable accounts
for 20.74% fewer parameters, 20.58% fewer parameter bytes, and 20.82% fewer
logical MACs/token, but nearly all of that reduction is inherited from the
factorized carrier. The rank-8 X4 head itself adds 21,504 parameters/MACs and
172,032 runtime bytes.

## NLL-VJP and realized-state H4 controls

The first two loss-aware H4 rungs have now been tested against a fresh
ordinary-ridge control. The source-NLL arm keeps directions fixed and changes
only the offline coefficient objective. The candidate-H4 arm performs one
additional fit-only candidate execution, uses the accepted-X4 H4 VJP in both
residual direction mapping and the coefficient metric, and therefore tests a
fully fresh tangent pipeline. These three tangent-control arms retain rank 8,
lag count 32, family/example weighting, serving head shape, parameters, bytes,
and MACs. The fourth arm below holds those controls fixed but intentionally
adds one realized-H4 input feature block and its exact storage/MAC cost.
For H4 residual
`E_i(C) = R_i - (A_i C) D` and the normalized source-native NLL VJP `u_i`,
or candidate-conditioned NLL VJP, the bounded solve minimizes

```text
sum_i w_i [ ||E_i(C)||² + (u_iᵀ E_i(C))² ] + ridge
```

The per-row metric is `I + u_i u_iᵀ`, so its eigenvalues remain between one
and two. Raw gradient magnitude cannot overwhelm family/example balancing.
The 16,384-coefficient quadratic is solved by deterministic matrix-free
conjugate gradient initialized from ordinary ridge; no dense 2 GiB vectorized
Hessian is materialized.

All CPU/float32 campaigns used the same 8-prompt fit and 4-prompt reusable
selection panels. They were selection-only processes: the generic controller
now terminates an eligible development search at `ready_for_guard_claim`, and
a separate finalizer is the only API that accepts a guard callback. The Gemma
worker was constructed without a guard authority or provider. A checksum
preflight had touched the previous private guard without exposing prompt text,
so that guard was conservatively retired before either run. The replacement
prompt-free corpus artifact is
`55015297b5f06006ac0a03fbb3fa38a15b4d6815625fe35b76ad4a52e3aa066b`;
its fresh guard manifest is
`53dc29646354beed29ceec9969a820f661d0499c2aad25b5bbcea361f1bbc3b9`.
It remains unopened and unclaimed.

| H4 mapping / objective / input | fit linearized-NLL RMSE | fit normalized-direction RMSE | fit projected-residual RMSE | selection abs. ΔNLL | selection KL | top-1 | p90 abs. ΔNLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| hidden-residual ridge / L3 | 2.822196 | 2.937411 | 0.000533 | 0.377753 | 0.964429 | 0.479245 | 0.792749 |
| source-NLL-VJP metric / L3 | 2.821386 | 2.936489 | 0.018397 | 0.377860 | 0.964445 | 0.479245 | 0.792909 |
| candidate-H4-VJP remap + metric / L3 | 2.720673 | 3.337574 | 0.021149 | 0.377742 | 0.964426 | 0.479245 | 0.792648 |
| candidate-H4-VJP + realized-H4 decoder modes | 2.720673 | 3.337573 | 0.021148 | 0.389134 | 0.948135 | 0.467925 | 0.799620 |

The source loss metric reduced its own first-order fit errors by only
`0.0287%` and `0.0314%`; projected hidden-residual RMSE became `34.52×`
ridge, and selection became slightly worse. The candidate-conditioned map
produced a much clearer `3.60%` reduction in linearized-NLL RMSE, showing that
the fresh tangent changes the optimization meaningfully. It did not solve the
actual task: normalized-direction RMSE worsened `13.62%`, hidden error became
`39.69×` ridge, and selection changes were only `-0.0000111` absolute ΔNLL,
`-0.0000035` KL, `-0.0001013` p90 ΔNLL, and exactly zero top-1 change. All
H4 candidates used exactly `212,522,752` accounted parameters and
`212,269,888` logical MACs/token, and all were rejected with
`stalled_fidelity`; X4 remains the active candidate.

The tangent result rules out stale VJP provenance as the main blocker: the
candidate pass was authenticated, its VJP matched finite-displacement tests,
the post-X4 mapper consumed it, and the fit objective responded. The next
campaign tested the narrower input-sufficiency hypothesis. It reused the
rank-8 output decoder as an input encoder for current pre-correction H4 and
added one `8 × 8` pointwise state kernel. This charged 64 parameters, 512
runtime bytes, and `640 × 8 + 8² = 5,184` logical MACs/token. The full rejected
child therefore accounted for 212,522,816 parameters, 851,876,904 runtime
bytes, and 212,275,072 logical MACs/token.

The feature was active, but its incremental fit contribution was negligible:
projected-residual RMSE improved only `0.00154%`, and both NLL fit errors
changed by less than `0.000003%`. Held-out behavior was mixed and worse
overall. KL improved `1.69%` and worst-prompt top-1 improved by one token, but
absolute ΔNLL worsened `3.02%`, p90 ΔNLL worsened `0.88%`, and aggregate top-1
fell from `127/265` to `124/265`. All five execution axes still failed, the
H4 child was rejected, and the terminal candidate remained X4-only with
`stalled_fidelity`.

This did not cleanly prove that realized H4 contained no useful signal. The
baseline had only 479 fit rows but `32 × 64 = 2,048` lagged-L3 columns; adding
eight state columns could not expand the training row space and mainly changed
the ridge parameterization. Reusing the output decoder also restricted the
input observation to the same eight directions used to emit a correction.
The fit-only diagnostic below now resolves that identifiability question.

The ridge-control report is bound by report hash
`1260de04f6d88b4e89a924ed001d675119b82b8b7466d72fb042d44d32cb8ab9`
and transcript
`7bcf1a42f47d522b85c3c724164f0fa0d6e0b55aab3f4dab8d50d1fbac2ab94a`.
The source-NLL-VJP report is bound by report hash
`39474b29896f203dc76e9d2e67dab18e830787aeb66e40271bd4609e50ca61d6`
and transcript
`b05b7583febd3172c258c45bf990e11be3114f54f9cc59cd9e26cb663ecec40d`.
The candidate-H4-VJP report is bound by report hash
`5930b64655add7d90564487a47dbce6d2b55db3590766e042cb21ba6a6c5af21`
and transcript
`eedee74d1b418725c46d5772735f7632e8a9d0594ebee8e80c3518cb5f44d49d`.
The realized-H4 input report is bound by report hash
`d0c27cccca0c19a6db8f84e39a1d3b8146d337bba07ce6492c2fb0d13ebf3975`,
transcript
`79b8638287884b1ad505b377a3f69b76307e555124bb4c9a141699bca73c0bb1`,
and report-file hash
`e116d347acba9167c0b099f66d63b498494521a5ea56fe76ff4d3a4b9f6db500`.

## Fit-only H4 incremental-signal diagnostic

`gemma3_l3_l4_h4_incremental_signal_diagnostic` is deliberately not another
campaign. It accepts the prompt-free corpus artifact, the A-fit role, and the
already accepted X4 report/candidate. It has no selection-input option, no
guard-input option, no guard authority, and no Calibration-B loader. The
published report contains scalar summaries and tensor hashes, not prompts,
tokens, activations, gradients, coefficients, or model weights.

For each lag count `L ∈ {1, 2, 4, 8, 16, 32}`, the diagnostic builds the
causal rank-64 L3 design and compares:

- the matched L3-only output-rank-8 control;
- realized H4 projected through the reused output decoder; and
- independent residual-H4 SVD encoders at input ranks 8, 16, and 32.

The four A-fit families form outer leave-one-family-out evaluation folds.
Within each three-family outer training set, another leave-one-family-out loop
fits the L3 nuisance maps and produces genuinely out-of-family H4 residuals.
The independent encoder and state kernel are derived only from those
cross-fitted residuals. A passing cell must add its complete requested
numerical rank in every outer fold, improve macro held-family
linearized-NLL-residual RMSE by at least 2%, win at least three families,
regress no family by more than 2%, keep the two secondary metrics within 2%,
and retain nontrivial residual H4 energy.

The runtime-compatible refit, if one qualifies, folds the L3-predictable H4
term back into the lag kernel:

```text
K_stored = B - C_Z A
delta_h_t = [x_t K_stored + h_t E^T A] D
```

`E` is an independent `input_rank × 640` state encoder, `A` is an
`input_rank × 8` state kernel, and `D` remains the output-rank-8 decoder.
This preserves pointwise H4 conditioning and one-forward causality without
storing a full-width nuisance map. No runtime artifact is emitted unless the
cross-fitted gate passes.

The real CPU/float32 fit run contained 479 affected rows across four families:

| lag | L3 columns | full rank / rows | outer-fold ranks | minimum unused outer row dimension |
|---:|---:|---:|---:|---:|
| 1 | 64 | 64 / 479 | 64 | 288 |
| 2 | 128 | 128 / 479 | 128 | 224 |
| 4 | 256 | 256 / 479 | 216 | 136 |
| 8 | 512 | 320 / 479 | 240 | 112 |
| 16 | 1,024 | 384 / 479 | 288 | 64 |
| 32 | 2,048 | 479 / 479 | 352–363 / 352–363 | 0 |

This confirms the prior lag-32 confound exactly: every outer design is full
row rank and every conditioned lag-32 cell adds numerical rank zero. At lags
1 through 16, all requested independent ranks are numerically identifiable.

Lag 4 was the only clear effect-size region:

| H4 input | macro linearized-NLL improvement | family wins | worst-family improvement | macro projected-residual improvement | params | MACs/token |
|---|---:|---:|---:|---:|---:|---:|
| reused output decoder, r8 | 1.8470% | 3/4 | -2.5943% | 14.4219% | 7,232 | 12,352 |
| independent residual H4, r8 | 1.5820% | 3/4 | -1.7062% | 14.4955% | 12,352 | 12,352 |
| independent residual H4, r16 | 2.7215% | 3/4 | -4.3171% | 19.4441% | 17,536 | 17,536 |
| independent residual H4, r32 | 3.7845% | 3/4 | -2.4054% | 31.6901% | 27,904 | 27,904 |

The rank-8 independent arm satisfied the family-regression bound but missed
the 2% macro threshold. Ranks 16 and 32 cleared the macro threshold and won
three families, but the held-out `progressive-fit-evidence-a` family regressed
by 4.3171% and 2.4054%, respectively. At rank 32 the other three changes were
`+2.2167%`, `+0.3113%`, and `+12.6713%`, so the macro gain was not a uniform
transfer effect.

The terminal diagnostic status is `no_crossfit_incremental_signal`: no cell
passed every gate, no winning recipe was emitted, and selection was not
opened. This is stronger than the earlier joint-ridge result in both
directions. It shows that independent H4 directions can add identifiable
held-family signal around lag 4, while also showing that the current
four-family fit panel does not support a stable transferable edge. It does
not authorize a nonlinear gate, model-level candidate, compression claim, or
latency claim.

The next valid experiment is to freeze a larger and more varied A-fit family
panel, rerun this exact grid and gate once, and only then precommit a fresh A2
selection panel for a single locked winner. Reusing the existing selection
panel or relaxing the worst-family gate after seeing these results would turn
the near miss into post-hoc optimization.

Run the fit-only diagnostic with:

```bash
fisher-graph-gemma-l3-l4-h4-incremental-signal-fit \
  --corpus-artifact \
    .local-runs/google--gemma-3-270m/progressive-a-loss-v3.corpus.json \
  --fit-input \
    .local-runs/google--gemma-3-270m/progressive-a-pilot-v1.fit.json \
  --accepted-x4-report \
    .local-runs/google--gemma-3-270m/progressive-a-h4-projected-state-v6.campaign.json \
  --accepted-x4-candidate \
    .local-runs/google--gemma-3-270m/progressive-a-h4-projected-state-v6.campaign.candidate.pt \
  --accepted-x4-candidate-sha256 \
    e52642de222a1b415fb2fa33266ac89d7621538052d7db86d722c8edb6fd122b \
  --device cpu \
  --dtype float32 \
  --output-rank 8 \
  --lag-counts 1 2 4 8 16 32 \
  --input-ranks 8 16 32 \
  --ridge 1e-6
```

The source-safe report is bound by report hash
`57f79eb3bde8f3eaddaaf93e1fabe1c71325dc39e2f0db675c3837f735be2641`,
analysis hash
`22b3d97be38e6505e08d378813731b7a6c1f7b5e3191a979444cf1803d0b701a`,
and file hash
`103d2c7cc04f16769d845c75d10c81a8889c155638fdf9527478645aa83fc0b8`.

## Expanded eight-family replication

The next run replaced only `calibration_a_fit` with 16 new prompts across
eight new families, two prompts per family. The prompt-free parent artifact
was authenticated first. The replacement builder then read only the new fit
JSON and copied the parent's exact selection and guard preclaim views. It
accepted no protected role paths. Corpus-wide identity, tokenizer binding,
forbidden Calibration-B identity, protected manifest hashes, protected
role-file hashes, and guard-ledger namespace all remained unchanged.

The replacement lineage is:

- parent corpus:
  `55015297b5f06006ac0a03fbb3fa38a15b4d6815625fe35b76ad4a52e3aa066b`;
- replacement corpus:
  `e4804338dbc3e76a84bf0483526ac9bab4e5f8aeaa86a32283832fed25f4b766`;
- replacement fit manifest:
  `75ae2045fd16de3128e3eef3a0177422bf2bad87a7bbd83ffeeb80ba2c1aac1d`;
- preserved selection manifest:
  `f0d561339b6255b6a942ee657580fa54da7f4d1ec96f502b70b4d8c7a7c29f4e`;
- preserved guard manifest:
  `53dc29646354beed29ceec9969a820f661d0499c2aad25b5bbcea361f1bbc3b9`.

The v2 diagnostic keeps outer leave-one-family-out evaluation. Within each
outer training set it uses three deterministic family-blocked inner folds,
so every encoder and state kernel is still built from out-of-family H4
residuals without the quadratic number of inner fits. It releases one lag's
private tensors before constructing the next and would recompute only a
selected lag for a final recipe. The family-win requirement is
`ceil(0.75 × family_count)`, or 6/8 here. The exact lag/rank grid, 2% macro
threshold, 2% worst-family limit, secondary-metric bounds, numerical-rank
gate, output rank, VJP objective, and ridge remained fixed.

The expanded panel produced 1,008 affected rows:

| lag | L3 columns | full rank / rows | outer-fold ranks | minimum unused outer row dimension |
|---:|---:|---:|---:|---:|
| 1 | 64 | 64 / 1,008 | 64 | 816 |
| 2 | 128 | 128 / 1,008 | 128 | 752 |
| 4 | 256 | 256 / 1,008 | 256 | 624 |
| 8 | 512 | 512 / 1,008 | 512 | 368 |
| 16 | 1,024 | 768 / 1,008 | 672 | 208 |
| 32 | 2,048 | 1,008 / 1,008 | 880–885 / 880–885 | 0 |

The larger panel makes ranks 1 through 16 substantially more identifiable;
lag 32 still saturates every outer row space and adds rank zero. The original
lag-4 effect did not replicate. The strongest expanded-panel cells were:

| H4 input | macro linearized-NLL improvement | family wins | worst-family improvement | projected-residual improvement | params | MACs/token |
|---|---:|---:|---:|---:|---:|---:|
| lag 4, independent r32 | -0.5944% | 3/8 | -5.0696% | -3.5149% | 27,904 | 27,904 |
| lag 8, independent r16 | 2.3517% | 7/8 | -6.0157% | 7.4814% | 19,584 | 19,584 |
| lag 8, independent r32 | 2.6752% | 6/8 | -9.0215% | 18.5396% | 29,952 | 29,952 |
| lag 16, independent r32 | 4.3793% | 7/8 | -3.0001% | 19.0182% | 34,048 | 34,048 |

The lag-16/r32 arm is the cleanest near miss. It improved seven families by
`+7.6715%`, `+1.2534%`, `+6.0712%`, `+4.8499%`, `+8.0840%`, `+7.0352%`,
and `+1.4136%`; only `progressive-fit-v2-constraint-propagation-s` regressed,
by `3.0001%`. Macro normalized-direction RMSE also improved `2.9581%`, so
the cell passed every preregistered gate except the 2% worst-family bound.

The terminal result nevertheless remains
`no_crossfit_incremental_signal`. Zero of 24 cells qualified, no winning
recipe was emitted, and selection and guard remained unopened. The
replication supports a transferable realized-H4 contribution more strongly
than the four-family panel, but it rejects lag 4 as a stable architecture and
still does not authorize a deployed head, nonlinear router, compression
claim, or latency claim.

That result preregistered the next fit-only rung: lock lag 16 and input rank
32, then test scalar damping between the matched L3-only baseline and the full
residual-H4 increment. The rung did not reopen lag/rank search, use the
existing selection panel, or introduce a nonlinear router.

Freeze and run the expanded fit role with:

```bash
fisher-graph-gemma-l3-l4-progressive-a-expand-fit

fisher-graph-gemma-l3-l4-h4-incremental-signal-fit \
  --corpus-artifact \
    .local-runs/google--gemma-3-270m/progressive-a-fit-expanded-v1.corpus.json \
  --fit-input \
    .local-runs/google--gemma-3-270m/progressive-a-fit-expanded-v1.fit.json \
  --accepted-x4-report \
    .local-runs/google--gemma-3-270m/progressive-a-h4-projected-state-v6.campaign.json \
  --accepted-x4-candidate \
    .local-runs/google--gemma-3-270m/progressive-a-h4-projected-state-v6.campaign.candidate.pt \
  --accepted-x4-candidate-sha256 \
    e52642de222a1b415fb2fa33266ac89d7621538052d7db86d722c8edb6fd122b \
  --expected-fit-example-count 16 \
  --expected-fit-family-count 8 \
  --device cpu \
  --dtype float32 \
  --output-rank 8 \
  --lag-counts 1 2 4 8 16 32 \
  --input-ranks 8 16 32 \
  --ridge 1e-6 \
  --output \
    .local-runs/google--gemma-3-270m/progressive-a-h4-incremental-signal-fit-expanded-v2.report.json
```

The expanded report is bound by report hash
`0dbccb00cc17995fe458a7eea6083ca030726c88b5b7df5884a0ae34087a107d`,
analysis hash
`cff5ce92530d32c84537bd7b2450eb0b70d15fa215c2d04563f4947522cd6cef`,
and file hash
`91d38b7ee2abd2693a0855e4fd10d082812d70e78cfee05e04ecd27c918ca584`.

## Fixed-head damping result

The damping runner authenticates the expanded-v2 report as a hypothesis
declaration, recollects all 16 model traces, and refits the fixed head. It
reuses no prior trace rows or coefficient tensors. Before scoring any damped
arm, alpha 1 must reproduce the source lag-16/r32 cell exactly: all 16 trace
hashes, the 1,008-row geometry, decoder hash, per-family encoder and state
kernel hashes, scalar metrics, gates, stability, and resources must match.
The real run passed that audit.

For lag design `x`, realized H4 `h`, independent encoder `E`, L3 nuisance map
`C_Z`, state kernel `A`, baseline kernel `B`, and output decoder `D`, the
cross-fitted diagnostic evaluates:

```text
incremental_h4 = (h E^T - x C_Z) A
prediction(alpha) = x B + alpha * incremental_h4
```

The fold head is fit once. Alpha does not refit `E`, `A`, `B`, or `C_Z`.
For a runtime-compatible refit the scale folds into the stored kernels:

```text
A_alpha = alpha A
K_alpha = B - C_Z A_alpha
delta_h = [x K_alpha + h E^T A_alpha] D
```

The preregistered positive ladder was `0.25/0.5/0.75/1.0`; alpha 0 is the
matched L3-only baseline. Every alpha used the unchanged v2 gate:
macro linearized-NLL improvement at least 2%, at least 6/8 family wins, no
family regression worse than 2%, both macro secondary metrics within 2%,
full rank 32 in every fold, and nontrivial residual-H4 energy.

| alpha | macro linearized-NLL improvement | family wins | worst-family improvement | projected-residual improvement | normalized-direction improvement | result |
|---:|---:|---:|---:|---:|---:|:---|
| 0.25 | 1.8400% | 7/8 | -0.0158% | 7.5220% | 1.6971% | macro gate failed |
| **0.50** | **3.1975%** | **7/8** | **-0.5272%** | **13.4237%** | **2.7690%** | **eligible, selected** |
| 0.75 | 4.0494% | 7/8 | -1.5267% | 17.3528% | 3.1926% | eligible |
| 1.00 | 4.3793% | 7/8 | -3.0001% | 19.0182% | 2.9581% | worst-family gate failed |

The fixed rule selected the smallest eligible alpha, `0.5`. The same
constraint-propagation family remains the lone linearized-NLL loss, but its
regression shrank from `3.0001%` to `0.5272%`; its projected and normalized
secondary metrics improved `17.1701%` and `0.5341%`. The result status is
`fit_only_damping_recipe_frozen`, with recipe hash
`0869aaa5bc246f4e9d73f79d3d8e3b87c6f752a7b7b384cbd6e4c4a7d5aae3a5`.

Every nonzero alpha has the same runtime shape: 34,048 head parameters,
272,384 float64 runtime bytes, and 34,048 logical MACs/token. Damping is a
robustness improvement, not a parameter or compute reduction by itself.
The report remains scalar/hash-only; it contains no activation rows,
gradients, coefficient tensors, or model weights.

Run the locked rung with:

```bash
fisher-graph-gemma-l3-l4-h4-damping-fit \
  --hypothesis-report \
    .local-runs/google--gemma-3-270m/progressive-a-h4-incremental-signal-fit-expanded-v2.report.json \
  --hypothesis-report-sha256 \
    0dbccb00cc17995fe458a7eea6083ca030726c88b5b7df5884a0ae34087a107d \
  --hypothesis-report-file-sha256 \
    91d38b7ee2abd2693a0855e4fd10d082812d70e78cfee05e04ecd27c918ca584 \
  --accepted-x4-candidate-sha256 \
    e52642de222a1b415fb2fa33266ac89d7621538052d7db86d722c8edb6fd122b
```

The damping report is bound by report hash
`dc85bb184b88a394d89d6e907ae496a3e920a011f9b7b7fb7e4f6b9a7d8e7a65`,
analysis hash
`1e01682390c135cd5616f966aa66fead3306bf785d88f5775a9ab6a1a4d439fd`,
and file hash
`1ddd80255c014d23a598ad4ec4543218a6437a39a6af3f50697ed98ed64fd94b`.

This passed the fit-only robustness rung, not deployment. At that point,
selection and guard remained unopened. The next authorized stage was to
deterministically
materialize the alpha-0.5 recipe, verify its tensor hashes, and compare that
single head against its authenticated alpha-0 baseline on a fresh
family-disjoint finite-NLL selection panel. Only a selection pass would
justify the later guard and model-level compilation steps.

## Deterministic H4 executor materialization

That authorized stage is complete. The materializer recollected the 16
expanded-fit traces from the authenticated factorized Gemma runtime,
recomputed the locked coefficients, and required exact agreement with the
decoder, independent H4 encoder, state-kernel, and stored lag-kernel hashes
in the fit-only damping report. It emitted exactly two tensor-only children
of the accepted X4 artifact:

```text
matched alpha 0:
    delta_h = x B D

challenger alpha 0.5:
    delta_h = [x (B - C_Z A_alpha) + h E^T A_alpha] D
    A_alpha = 0.5 A
```

The alpha-0 arm is not the accepted-X4-only context. It includes the matched
all-row lag-only H4 baseline `B`, which is required for the paired causal
comparison. The accepted-X4-only artifact remains a third deployment-context
control with no H4 head.

| executable | H4 conditioning | H4 parameters | logical H4 MACs/token | artifact hash |
|---|---|---:|---:|---|
| accepted X4 only | none | 0 | 0 | `f783026b…c13b02` |
| matched alpha 0 + B | L3 lag design | 13,312 | 13,312 | `38579f15…34c273` |
| alpha 0.5 challenger | L3 lag design + independent realized H4 state | 34,048 | 34,048 | `cbc3481c…a0896` |

The materialized alpha-0.5 head stores a `32 × 640` independent state
encoder, a `32 × 8` state kernel, a `16 × 64 × 8` lag kernel, and the shared
rank-8 output decoder. Its extra `20,736` parameters and MACs over alpha 0
come from the independent realized-H4 path. Damping itself does not create a
resource saving; the challenger is approximately `2.56×` the H4 head size of
the matched control.

On the all-row fit recollection, alpha 0.5 reduced projected-residual RMSE
from `61.063745` to `58.368759`, but normalized-direction RMSE changed from
`3.235888` to `3.236042` and linearized-NLL-residual RMSE changed from
`2.690616` to `2.690751`. Those all-row values are reproduction diagnostics,
not selection evidence. The selection question remained whether the
cross-fitted robustness signal survived as real finite-NLL behavior.

The materialization is bound by:

- report hash
  `27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94`;
- report-file hash
  `7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20`;
- alpha-0 tensor-file hash
  `a382bb42711c0c11cefc2922dab21c393106d7e41cc84c5f576458b859bb6948`;
- alpha-0.5 tensor-file hash
  `3b20c912fb9f3ac8fd68c45bbc8deb0363095d3b28d00873f6f34906f77988f0`.

Reproduce the materialization, before any selection panel is opened, with:

```bash
fisher-graph-gemma-l3-l4-h4-damping-materialize \
  --damping-report-sha256 \
    dc85bb184b88a394d89d6e907ae496a3e920a011f9b7b7fb7e4f6b9a7d8e7a65 \
  --damping-report-file-sha256 \
    1ddd80255c014d23a598ad4ec4543218a6437a39a6af3f50697ed98ed64fd94b \
  --accepted-x4-candidate-sha256 \
    e52642de222a1b415fb2fa33266ac89d7621538052d7db86d722c8edb6fd122b
```

## Fresh finite-NLL selection result

The executable pair was evaluated once on a new panel containing 16 prompts,
two prompts in each of eight family-disjoint capabilities. The prompt-free
panel artifact was frozen before any candidate output was observed:

- panel artifact:
  `a45e1b5eb0a08e70613dbad9f23b6401fd8a7f70bbe55b7308a63482bddaa8dc`;
- panel file:
  `449a478ac3ffacdba23b8a0cf2bf4890f4ab5ee960cc51f21adef6c61c22676c`;
- manifest:
  `ff0ac9ffa98ad558d1279d9cb2e2983a91a02c9ba7ea70c62de271dcb10c7c35`;
- membership receipt:
  `8e144b603c59bfc6d4e0af71df8aa66c1ad321b7e64f01ff336bb95c3675366e`.

The runner made exactly four forwards per example:

```text
direct factorized source
accepted X4 only
accepted X4 + matched alpha-0 B
accepted X4 + alpha-0.5 independent H4
```

That is 64 forwards total. The source output was shared only within the three
metrics for the same prompt. Full logits were measured and released
arm-by-arm; the report retains only per-example scalar sufficient statistics
and tensor hashes. A path-independent, identity-keyed `O_EXCL` claim was
written before the first prompt-text read, so copies, hard links, symlinks,
alternate input paths, and alternate report paths cannot create another
authorized opening.

For prompt `i`, the paired primary error is:

```text
e_i = abs(NLL_candidate_i - NLL_source_i) / supervised_tokens_i
```

Each family score is the mean of its two prompt errors. Taking the absolute
value per prompt is important: opposite signed NLL deltas cannot cancel
inside a family. Alpha 0.5 had to improve the eight-family macro by at least
2%, win at least 6/8 families, and regress no family by more than 2%.
KL, top-1 disagreement, prompt-p90 NLL error, and prompt-p10 top-1
disagreement could not regress by more than 2%. Qualification also required
the challenger to pass the established absolute source-fidelity gates.

### Absolute source fidelity

The source grid contained 964 supervised tokens. None of the three
executables was close to the direct factorized source:

| arm | absolute aggregate ΔNLL/token | KL/token | aggregate top-1 agreement | prompt p90 absolute ΔNLL | prompt p10 top-1 | absolute gate |
|---|---:|---:|---:|---:|---:|:---:|
| accepted X4 only | 0.222797 | 1.286448 | 45.85% | 1.030822 | 29.63% | fail |
| matched alpha 0 + B | 0.532629 | 1.396646 | 41.29% | 1.388934 | 29.55% | fail |
| alpha 0.5 challenger | 0.408207 | 1.276282 | 43.05% | 0.968875 | 29.55% | fail |
| required | ≤0.05 | ≤0.05 | ≥95% | ≤0.10 | ≥90% | — |

The accepted X4-only context was already far outside the absolute gate. The
H4 selection was therefore attempting to improve a parent whose full output
distribution did not preserve this source boundary. Alpha 0.5 slightly
reduced KL relative to accepted X4, but worsened aggregate ΔNLL and top-1
agreement while adding 34,048 head parameters and MACs/token.

### Paired alpha-0.5 effect

Against the matched alpha-0 baseline, the challenger reduced the
family-macro mean prompt error from `0.604685` to `0.524278`, a `13.2973%`
improvement, and won exactly six families:

| family | alpha-0 mean prompt error | alpha-0.5 mean prompt error | relative change |
|---|---:|---:|---:|
| algorithm execution | 0.451902 | 0.385234 | +14.75% |
| evidence attribution | 0.175722 | 0.066747 | +62.02% |
| formal validity | 0.673243 | 0.591797 | +12.10% |
| rule exceptions | 0.440358 | 0.360474 | +18.14% |
| structured extraction | 1.300116 | 1.203081 | +7.46% |
| symbolic equivalence | 0.464468 | 0.533716 | **-14.91%** |
| unit consistency | 0.180488 | 0.233940 | **-29.62%** |
| verbal entailment | 1.151181 | 0.819234 | +28.84% |

All four paired secondary metrics improved or stayed flat: family-macro KL
improved `8.61%`, top-1 disagreement improved `2.51%`, prompt-p90 absolute
NLL error improved `30.24%`, and prompt-p10 top-1 disagreement was unchanged.
The paired gate nevertheless failed because its worst family regressed
`29.62%`, not at most `2%`.

The terminal decision is:

```text
paired_gate_passed = false
challenger_absolute_gate_passed = false
qualified = false
```

This is not a null result. The six wins, macro improvement, and secondary
improvements show that the independent realized-H4 path carries a useful
direction beyond the lag-only baseline. It is also not a near deployment
pass. The effect remains sharply family-dependent, and all absolute
distribution-fidelity measures miss their thresholds by large margins.

The one-shot report is bound by:

- campaign report hash
  `63bcc5f6b03cee408164583a01109023aeb2352e3a4fa15e23ddbc2d7b842f35`;
- nested finite-NLL report hash
  `3d842a558d5fb31b5ed99623c2527867a3b6936dd55e427c88c5da4939fbb044`;
- report-file hash
  `43accd933ea8fce333d056abe7d197a8dd7178049a4c8c9d2a8c9ff2a539ad14`;
- durable claim hash
  `5a8ca2c277b0d32b954d24a85270876b354fc4b02ba8c2eaf8dffdce8e0cd653`.

The historical one-shot invocation was:

```bash
fisher-graph-gemma-l3-l4-h4-damping-selection \
  --new-selection-panel-artifact-sha256 \
    a45e1b5eb0a08e70613dbad9f23b6401fd8a7f70bbe55b7308a63482bddaa8dc \
  --new-selection-panel-file-sha256 \
    449a478ac3ffacdba23b8a0cf2bf4890f4ab5ee960cc51f21adef6c61c22676c \
  --materialization-report-sha256 \
    27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94 \
  --materialization-report-file-sha256 \
    7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20 \
  --accepted-x4-candidate-sha256 \
    e52642de222a1b415fb2fa33266ac89d7621538052d7db86d722c8edb6fd122b
```

That command is no longer runnable for this panel: the durable identity claim
has consumed it. Guard evaluation is not authorized.

The next valid research rung must return to development data. The immediate
problem is not merely choosing a better scalar alpha; the accepted X4 parent
already lacks finite-NLL fidelity, and the H4 correction cannot repair that
upstream distortion uniformly. A revised compiler should optimize a joint
X4/H4 or wider multi-layer finite-displacement objective, then earn a
completely new family-disjoint selection panel. The opened panel may be used
to understand this rejection, but it cannot validate the revised candidate.

## Reusable X4/H4 factorial attribution

The first post-rejection development rung is complete. It did not reopen the
consumed selection panel, any guard, or Calibration-B. It used only the
reusable expanded Calibration-A fit role that produced the materialized H4
heads. Consequently, this result can attribute local behavior and guide the
next fit, but it has no promotion or generalization authority.

The current one-pass bridge always clamps Y3 and applies its decoded base X4
delta. Therefore `x4_head = none` means the **base compiled bridge**, not
native X4. A separate direct factorized-model forward remains the source
authority. The executable factorial is:

```text
                         H4 none       H4 lag B       H4 independent state
base compiled bridge       ●              ●                    ●
accepted X4 repair         ●              ●                    ●

+ one direct factorized-model source per example
```

The runner made seven forwards per example over the 16-example,
eight-family expanded-fit panel: one direct source and six factorial cells.
That is exactly 112 forwards and 96 candidate observations. It captured
native X4 and H4 once per example, authenticated one post-Y3-clamp bridge
reference across all six cells, required each X4 result to remain invariant
across its three H4 levels, and reduced every output and activation
comparison immediately to scalar sufficient statistics and tensor hashes.
The persisted report contains no prompt text, token IDs, logits,
activations, coefficient tensors, or model weights.

The primary final-output error is the token-weighted sum of each prompt's
absolute NLL displacement:

```text
E = sum_i abs(NLL_candidate_i - NLL_source_i)
    / sum_i supervised_tokens_i
```

That definition prevents positive and negative prompt errors from canceling.
The signed ΔNLL channel is retained separately for attribution. Boundary
RMSE values use the exact causal target-affected rows.

| X4 / H4 arm | auxiliary params | logical MACs/token | prompt-absolute ΔNLL/token | signed ΔNLL/token | KL/token | top-1 | X4 RMSE | H4 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base / none | 403,328 | 217,920 | 0.395952 | -0.352429 | 1.304834 | 41.14% | 0.562206 | 32.0043 |
| base / lag `B` | 416,640 | 231,232 | 0.300136 | -0.193064 | 1.002118 | 45.03% | 0.562206 | 25.4643 |
| base / independent state | 437,376 | 251,968 | 0.302431 | -0.189192 | 1.001871 | 44.94% | 0.562206 | 24.1366 |
| accepted X4 / none | 424,832 | 239,424 | 0.288211 | -0.217717 | 1.275279 | 40.88% | 0.457586 | 34.9819 |
| accepted X4 / lag `B` | 438,144 | 252,736 | 0.265847 | -0.021640 | 0.891692 | 45.72% | 0.457586 | 14.5398 |
| accepted X4 / independent state | 458,880 | 273,472 | 0.265434 | -0.017090 | 0.892949 | 45.38% | 0.457586 | 14.4005 |

Here, “auxiliary” covers the compiled bridge and selected residual heads.
The factorized source model is still retained. These counts are not a
whole-model compression result.

### What the factorial isolates

The accepted X4 repair is doing useful work. With H4 disabled, it reduced
prompt-absolute NLL error from `0.395952` to `0.288211`, a `27.21%`
improvement, and reduced X4 RMSE from `0.562206` to `0.457586`, an `18.61%`
improvement. It improved seven of eight fit families. It is not sufficient
by itself: KL remained `1.275279`, top-1 agreement remained `40.88%`, and
one family regressed sharply.

The lag-only `B` head is also doing useful work. On the base bridge it
reduced prompt error by `24.20%`; on the accepted-X4 parent it reduced prompt
error by `7.76%`, KL by `30.08%`, and H4 RMSE by `58.44%`. The smaller final
NLL increment after accepted X4 is a positive-error interaction of
`+0.073452`: the two repairs have overlapping or diminishing final-output
benefit, even though their H4-boundary interaction is strongly favorable.
Accepted X4 plus `B` remains family-dependent, winning only five of eight
families against accepted X4 alone.

The independent-state H4 path is not earning its marginal cost in this
configuration. Relative to lag `B`:

- under the base bridge, it improved H4 RMSE by `5.21%` but worsened
  prompt-absolute NLL error by `0.76%`;
- under accepted X4, it improved H4 RMSE by only `0.96%` and
  prompt-absolute NLL error by only `0.16%`;
- it won only four of eight accepted-X4 fit families;
- KL worsened from `0.891692` to `0.892949`, and top-1 agreement fell from
  `45.72%` to `45.38%`;
- it added 20,736 parameters and logical MACs per token over lag `B`.

This resolves the apparent contradiction in the earlier H4 evidence.
Independent realized-H4 state does contain a direction that reduces a
projected or Euclidean residual. That does not mean it improves the
source-authoritative behavior enough to justify a larger head.

### The optimization target is the next bottleneck

The accepted-X4 plus H4 arms have signed aggregate ΔNLL values near zero
(`-0.021640` and `-0.017090`) while their prompt-absolute errors remain near
`0.266`. The H4 corrections are largely centering the average NLL bias;
large per-prompt over- and under-corrections remain and cancel in the signed
aggregate. The same mismatch appears at the activation boundary:
independent state can lower H4 RMSE without lowering finite-NLL error.

The next iterative compiler should therefore start from the cost-aware
accepted-X4 plus lag-`B` parent and add one residual repair at a time. Each
loop should:

1. collect source-authoritative per-prompt residual/JVP information on
   reusable development folds;
2. fit a small conditional residual edge against prompt-absolute NLL, KL,
   and top-1 behavior rather than H4 Euclidean error alone;
3. replay this six-cell attribution locally to check whether the new edge
   supplies independent benefit instead of duplicating X4 or `B`;
4. retain the edge only when family-macro error improves, enough families
   win, the worst family stays bounded, and its parameter/MAC cost is
   justified;
5. freeze a completely new family-disjoint selection protocol only after
   the iterative development curve reaches the established absolute gates.

A useful next form is a prompt/position-conditioned gate over the observable
accepted-X4 plus lag-`B` state and existing source modes, with a small
residual edge trained through downstream NLL geometry. The current result
does not justify simply increasing the global independent-state rank.

The development report is bound by:

- logical report hash
  `9fde13193b3ee915845247de3821e84ec84dbae368a2e3554e4c9423927878ed`;
- report-file hash
  `2e9fc09804af10199739ad6c14df658a6a81550e4e14b5eeb9170c52d5a809ab`.

Reproduce this fit-only diagnostic with:

```bash
fisher-graph-gemma-l3-l4-x4-h4-factorial-dev \
  --materialization-report-sha256 \
    27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94 \
  --materialization-report-file-sha256 \
    7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20 \
  --accepted-x4-candidate-sha256 \
    e52642de222a1b415fb2fa33266ac89d7621538052d7db86d722c8edb6fd122b
```

## Iteration 1: causal position scaling over lag B

The first executable compiler loop kept the cost-aware accepted-X4 plus
lag-`B` stack frozen and tested one intentionally tiny repair. At the H4
boundary it multiplied the existing lag-`B` correction by
`1 + theta[position_bin]`, where the fixed causal bins were `0–3`, `4–7`,
`8–15`, and `16+`. A token's bin depends only on its current logical
position, never final sequence length, token identity, prompt family, or
future occupancy. `theta = 0` is the exact parent.

For each of the 16 reusable expanded-fit prompts, the runner made a direct
factorized-source pass and an accepted-X4 plus lag-`B` NLL-VJP pass. It
reduced the latter to the prompt's signed NLL displacement `d_i` and four
derivatives `J_i`. Each of eight leave-one-family-out folds then solved the
fixed family-balanced ridge objective

```text
min_theta sum_i w_i (d_i + J_i theta)^2 + 1e-6 ||theta||^2
```

with elementwise `theta` bounded to `[-0.5, 0.5]`. The second phase reran the
direct source and evaluated the corresponding out-of-family provider.
Consequently the campaign made exactly 64 model forwards:

```text
16 source + 16 parent VJP + 16 fresh source + 16 OOF candidate
```

No selection, guard, assessment, or Calibration-B role was opened. The
persisted report retains scalar sufficient statistics, four-value fold
coefficients, and hashes only. Its validator independently reruns every fold
from the exact 14 authenticated training records, binds each exact OOF
observation to its held-family provider and execution, replays the resource
receipt, and requires an authenticated full-data provider receipt before any
successful candidate can be retained.

### Exact out-of-family result

| metric | frozen parent | four-bin candidate | relative result |
|---|---:|---:|---:|
| family-macro mean prompt-absolute ΔNLL/token | 0.268343 | 0.299215 | -11.50% |
| family-macro KL/token | 0.891598 | 0.911324 | -2.21% |
| family-macro top-1 agreement | 45.72% | 44.90% | -0.82 points |
| prompt p90 absolute ΔNLL/token | 0.404262 | 0.421663 | -4.30% |
| prompt p10 top-1 agreement | 40.00% | 37.68% | -2.32 points |
| strict family wins | — | 2/8 | required 6/8 |
| worst-family relative improvement | — | -70.64% | required at least -2% |

All resource gates passed: the edge has four scalar slots, at most 640
logical MACs/token, references rather than duplicates lag `B`, and preserves
one serving-model forward. Using the factorial's auxiliary accounting, that
would move the stack from 438,144 to 438,148 parameters and from 252,736 to
253,376 logical MACs/token. Behavior failed, so the candidate was not
retained and no full-fit child became the next parent.

### What failed

This rung separates local linearization error from representation error. The
OOF parent-point prediction and exact finite candidate had:

- correlation `0.998966`;
- RMSE `0.021381` ΔNLL/token;
- `100%` sign agreement;
- predicted prompt-absolute error `0.298132`;
- exact prompt-absolute error `0.299215`.

Both the prediction and the real model said the candidate would be worse
than the parent. A midpoint or path-integrated Jacobian is therefore not the
next bottleneck.

The position feature was also effectively only two-dimensional. All
target-affected rows began at logical position 8, so bins `0–3` and `4–7`
were unsupported and stayed at `theta = 0` in every fold. The supported
coefficients varied substantially by held family; one fold even reversed
the sign of the `8–15` adjustment. Training-fold linearized RMSE improved
only slightly, and that relationship did not transfer to held families.

The result rejects only **family-invariant position-dependent rescaling of
lag `B`**. It does not reject iterative compilation, a prompt/state
conditioned edge, a new H4 direction, or joint X4/H4 repair. The next
preregistered candidate should use causal prefix statistics and observable
compiled state to select a small residual direction. It should keep the same
LOFO and exact-finite retention protocol.

The report is bound by:

- collection hash
  `dee8864210f6047e3f05b67515a5f77b54a1ccbe332de5b9eb96519eca33714c`;
- logical hash
  `1e1e284d354dd6048406b99a335bc2065e6767e706b8e781791bb1fd365c49ca`;
- file hash
  `4ace545b0dea88aeebbbe7e8ddd57a89ff986e56a1817ab4500ad44fa056afb3`.

Reproduce it with:

```bash
fisher-graph-gemma-l3-l4-iterative-residual-dev \
  --materialization-report-sha256 \
    27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94 \
  --materialization-report-file-sha256 \
    7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20 \
  --factorial-report-sha256 \
    9fde13193b3ee915845247de3821e84ec84dbae368a2e3554e4c9423927878ed \
  --factorial-report-file-sha256 \
    2e9fc09804af10199739ad6c14df658a6a81550e4e14b5eeb9170c52d5a809ab
```
