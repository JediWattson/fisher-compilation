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

## Iteration 2: causal top-two modal state routing

Iteration 2 replaces the failed global position feature with observable
compiled state while leaving the accepted-X4 plus lag-`B` parent frozen. It
selects lag-`B` output modes 0 and 1, the two modes with the largest
lag-kernel Frobenius norms (`9.307384` and `3.937768`). For their parent modal
values `m_0(t)` and `m_1(t)`, it advances two scalar accumulators:

```text
a_t = m_0(t) / 9.307384
b_t = m_1(t) / 3.937768
N_t = N_(t-1) + a_t - b_t
Z_t = Z_(t-1) + |a_t| + |b_t|
g_t = 0 if Z_t = 0 else N_t / Z_t
```

The learned child is one row-major `2 x 2` matrix `Theta`. It changes only
the two selected modal values,

```text
[m'_0(t), m'_1(t)] =
  [m_0(t), m_1(t)] + g_t [m_0(t), m_1(t)] Theta,
```

then reuses the parent's existing modal decoder exactly once. `Theta = 0`
is exactly the frozen parent. `N_t` and `Z_t` are explicit chunkable
generation carry, so the same provider can continue across incremental
decode chunks after the upstream parent executor supplies each modal chunk
using its own lag-history cache. The route reads no prompt IDs, token IDs,
family labels, final sequence length, or future positions.

The fit used the same 16-prompt, eight-family leave-one-family-out protocol
as Iteration 1. Each parent-point NLL-VJP was reduced to four route-edge
derivatives. Every fold solved the family-balanced ridge objective with
ridge `1e-6`, then projected the fitted matrix to operator norm at most
`0.25`. The two-phase exact-finite campaign again made exactly 64 model
forwards:

```text
16 source + 16 parent VJP + 16 fresh source + 16 OOF candidate
```

### Exact out-of-family result

| metric | frozen parent | state router | relative result |
|---|---:|---:|---:|
| family-macro mean prompt-absolute ΔNLL/token | 0.268343 | 0.265900 | +0.91% |
| family-macro KL/token | 0.891598 | 0.894444 | -0.32% |
| family-macro top-1 agreement | 45.72% | 45.54% | -0.17 points |
| aggregate top-1 agreement | 45.72% (529/1,157) | 45.55% (527/1,157) | -2 matches |
| prompt p90 absolute ΔNLL/token | 0.404262 | 0.393319 | +2.71% |
| prompt p90 top-1 disagreement | 60.00% | 60.00% | unchanged |
| strict family wins | — | 5/8 | required 6/8 |
| worst-family relative improvement | — | -6.43% | required at least -2% |

The macro non-regression condition used for retention and every secondary
regression bound passed. The candidate nevertheless failed the
preregistered behavior gate because it won only five families and its worst
held family exceeded the allowed regression. It was therefore **not
retained**; no full-fit provider was created, deployment was not authorized,
and the accepted-X4 plus lag-`B` stack remains iteration zero.

The family breakdown shows that the global route is useful but not
uniformly useful:

| held family | parent error | router error | relative improvement | strict win |
|---|---:|---:|---:|:---:|
| budget allocation | 0.522789 | 0.524300 | -0.29% | no |
| constraint propagation | 0.335958 | 0.353450 | -5.21% | no |
| counterfactual isolation | 0.192740 | 0.185374 | +3.82% | yes |
| hierarchical composition | 0.258471 | 0.251244 | +2.80% | yes |
| reference frame | 0.184966 | 0.184873 | +0.05% | yes |
| state invariant | 0.130012 | 0.138370 | -6.43% | no |
| temporal dependency | 0.137643 | 0.128397 | +6.72% | yes |
| uncertainty update | 0.384163 | 0.361194 | +5.98% | yes |

Here, error is each family's mean per-prompt absolute signed ΔNLL/token.
The small positive reference-frame change counts as a strict numerical win,
while the state-invariant family determines the worst-family failure.

### Scientific, linearization, and resource evidence

The state feature was genuinely exercised rather than passing through a
degenerate design:

- all eight weighted fold designs had full rank four;
- all four directed route edges were supported in all eight folds;
- the 16 prompts contributed 1,008 active rows;
- family-macro balance-feature standard deviation was `0.154656`, above the
  preregistered `0.05` floor;
- the selected pair carried `87.145993%` of measured lag-`B` modal energy,
  above the `50%` floor.

The parent-point model also predicted the finite displacement extremely
well. Predicted-versus-exact ΔNLL had correlation `0.999985`, RMSE
`0.002107` ΔNLL/token, `100%` sign agreement, and worst absolute error
`0.007926`. It predicted a `1.20%` macro improvement; exact execution
delivered `0.91%`. Thus the failure is not evidence that the Jacobian
linearization broke. It is evidence that one family-invariant `2 x 2`
rotation does not generalize evenly across the eight behaviors.

Every fold hit the `0.25` operator-norm trust boundary; the unprojected
operator norms ranged from `11.2273` to `29.8436`. That is useful pressure
evidence, but simply raising the bound is not the safest next rung because
the current bounded route already harms three families. The better next
candidate is a tiny causal regime or expert split keyed by the same
observable balance state, allowing different rotations without introducing
prompt identity.

All resource gates passed. Relative to the frozen parent, the route adds:

- four learned float scalars;
- two derived prepared constants;
- two runtime-state floats per sequence;
- six linear MACs/token and at most five nonlinear scalar operations/token;
- no duplicated parent head and no additional serving-model forward.

This is the first positive family-disjoint iterative result, but it remains a
development near-miss rather than a retained compiler improvement or a
compression qualification.

The report is bound by:

- collection hash
  `f7ea40fd6bb5695ed9da5f21d8e8d279a8e257e00fb8b4c7bd204718b0c17b8c`;
- logical hash
  `2836d9bde0d39a5b1acbaab7d34fca69c5ad7eab9a1632a63900565eb8ff2207`;
- file hash
  `0a269c7f6336bccb601a868c157cea4dd4dce2d413a55ae7531471679d9f45f1`.

Reproduce it with:

```bash
fisher-graph-gemma-l3-l4-iterative-state-router-dev \
  --materialization-report-sha256 \
    27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94 \
  --materialization-report-file-sha256 \
    7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20 \
  --factorial-report-sha256 \
    9fde13193b3ee915845247de3821e84ec84dbae368a2e3554e4c9423927878ed \
  --factorial-report-file-sha256 \
    2e9fc09804af10199739ad6c14df658a6a81550e4e14b5eeb9170c52d5a809ab \
  --prior-iteration-report-sha256 \
    1e1e284d354dd6048406b99a335bc2065e6767e706b8e781791bb1fd365c49ca \
  --prior-iteration-report-file-sha256 \
    4ace545b0dea88aeebbbe7e8ddd57a89ff986e56a1817ab4500ad44fa056afb3 \
  --prior-iteration-collection-sha256 \
    dee8864210f6047e3f05b67515a5f77b54a1ccbe332de5b9eb96519eca33714c
```

## Iteration 3: sign-dispatched causal state experts

Iteration 3 tests the smallest conditional extension suggested by the
Iteration 2 near-miss. It does **not** stack the rejected router onto the
model. The executable parent remains the frozen accepted-X4 plus lag-`B`
iteration-zero artifact; the Iteration 2 report is an immutable rejected
prerequisite. The candidate reuses the same top-two modal balance `g_t`, but
selects one of two independently fitted `2 x 2` route matrices:

```text
expert(t) = negative     if g_t < 0
            nonnegative otherwise

[m'_0(t), m'_1(t)] =
  [m_0(t), m_1(t)]
  + g_t [m_0(t), m_1(t)] Theta_expert(t)
```

Only the selected expert is evaluated for a row. Both matrices equal zero at
the exact frozen parent, and both reuse the parent's decoder. The same
two-float causal carry `(N_t, Z_t)` supports incremental chunks when the
upstream parent supplies lag-aware modal chunks; dispatch uses no prompt ID,
family label, final sequence length, or future token.

The repository does not yet provide that upstream cached-modal producer or a
chunk-order cursor. The standard correction ABI is therefore the
full-sequence/prefill path; token-at-a-time graph integration must own the
parent lag cache and enforce ordered, nonduplicated chunks. The live campaign
does not claim end-to-end incremental-executor qualification.

Each prompt's parent-point NLL-VJP is reduced to eight derivatives: four
directed modal edges for the negative regime and four for the nonnegative
regime. Each leave-one-family-out fit uses family-balanced ridge `1e-6` and
projects each expert independently to operator norm at most `0.25`. The
campaign keeps the same 16-prompt, eight-family, two-phase protocol and makes
exactly 64 model forwards:

```text
16 source + 16 parent VJP + 16 fresh source + 16 OOF candidate
```

### Exact out-of-family result

| metric | frozen parent | sign experts | relative result |
|---|---:|---:|---:|
| family-macro mean prompt-absolute ΔNLL/token | 0.268343 | 0.269441 | -0.41% |
| family-macro KL/token | 0.891598 | 0.895068 | -0.39% |
| family-macro top-1 agreement | 45.72% | 45.36% | -0.35 points |
| aggregate top-1 agreement | 45.72% (529/1,157) | 45.38% (525/1,157) | -4 matches |
| prompt p90 absolute ΔNLL/token | 0.404262 | 0.397478 | +1.68% |
| prompt p90 top-1 disagreement | 60.00% | 60.00% | unchanged |
| strict family wins | — | 4/8 | required 6/8 |
| worst-family relative improvement | — | -3.35% | required at least -2% |

Every secondary non-regression bound passed, but the primary candidate did
not improve macro error, won only four families, and exceeded the allowed
worst-family regression. It was therefore **not retained**. No full-fit
provider was created, deployment was not authorized, and the compiler parent
remains iteration zero. The report also leaves both
`absolute_fidelity_gates_passed` and `ready_for_new_selection` false.

The family result is:

| held family | parent error | sign-expert error | relative improvement | strict win |
|---|---:|---:|---:|:---:|
| budget allocation | 0.522789 | 0.528449 | -1.08% | no |
| constraint propagation | 0.335958 | 0.346563 | -3.16% | no |
| counterfactual isolation | 0.192740 | 0.198781 | -3.13% | no |
| hierarchical composition | 0.258471 | 0.258338 | +0.05% | yes |
| reference frame | 0.184966 | 0.191157 | -3.35% | no |
| state invariant | 0.130012 | 0.126032 | +3.06% | yes |
| temporal dependency | 0.137643 | 0.131435 | +4.51% | yes |
| uncertainty update | 0.384163 | 0.374773 | +2.44% | yes |

Error is each family's mean per-prompt absolute signed ΔNLL/token, matching
the retention comparison.

### What changed relative to Iteration 2

| result | Iteration 2 shared route | Iteration 3 sign experts |
|---|---:|---:|
| family-macro relative improvement | +0.91% | -0.41% |
| strict family wins | 5/8 | 4/8 |
| worst family | -6.43% (state invariant) | -3.35% (reference frame) |
| state-invariant family | -6.43% | +3.06% |
| counterfactual-isolation family | +3.82% | -3.13% |
| temporal-dependency family | +6.72% | +4.51% |
| uncertainty-update family | +5.98% | +2.44% |

The conditional split repaired the conspicuous state-invariant failure and
reduced the magnitude of the worst regression. It did not preserve the
shared router's broader gains: counterfactual isolation reversed sign,
reference frame became the worst family, hierarchical composition became
nearly neutral, and the temporal and uncertainty wins shrank. This is
evidence that the sign of `g_t` changes which residuals are helped, but not
that it supplies a generally correct routing partition.

### Scientific, linearization, and resource evidence

The candidate failed behavior despite a healthy, exercised design:

- all eight folds had weighted design rank eight, with rank four in each
  expert;
- all eight directed expert-route edges were supported in every fold;
- the 16 prompts contributed 1,008 active rows;
- the negative and nonnegative regimes received family-macro active-row
  fractions of `21.6727%` and `78.3273%`, both above the `10%` floor;
- family-macro balance-feature standard deviation was `0.154656`;
- the top-two modes carried `87.145993%` of measured lag-`B` modal energy.

All scientific gates therefore passed. Both experts in all eight folds hit
the `0.25` operator-norm boundary. Before projection, negative-expert norms
ranged from `19.0552` to `40.9399` and nonnegative-expert norms from
`7.0942` to `20.5814`. This says both regimes contain strong fit pressure; it
does not establish that a larger trust region would generalize.

The parent-point model again tracked exact finite execution closely:

- predicted-versus-exact correlation: `0.999990`;
- RMSE: `0.001729` ΔNLL/token;
- sign agreement: `100%`;
- worst absolute prediction error: `0.006218` ΔNLL/token;
- predicted macro change: `-0.16%`;
- exact macro change: `-0.41%`.

The linearization predicted the regression rather than hiding an improvement.
The failure is therefore not a finite-displacement surprise and does not
motivate a path-integrated Jacobian by itself.

All resource gates passed. Relative to the frozen parent, the candidate adds:

- eight learned float scalars, versus four in Iteration 2;
- two derived prepared constants;
- two runtime-state floats per sequence;
- six linear MACs/token, unchanged from Iteration 2;
- at most six nonlinear scalar operations/token, versus five in Iteration 2;
- no duplicated parent head and no additional serving-model forward.

This remains a tiny conditional edge, not a compression or speed
qualification. Passing support, linearization, and resource gates shows that
the experiment was meaningful; it does not override the failed
family-disjoint behavior result.

### Conservative next evidence rung

A direct fold comparison identifies estimation variance before it identifies
a need for another state variable. Median weighted-design condition number
rose from `69.9` in Iteration 2 to `555.7` in Iteration 3. Mean pairwise
coefficient cosine fell from `0.972` to `0.686`, even though all formal
rank/support gates passed. Both experts then saturated their trust bounds.
The hard sign split therefore doubled fit dimension while assigning only
`21.67%` of rows to the negative branch; its independent directions were
substantially less stable across held families.

The next preregistered candidate should pool the conditional structure rather
than add another expert or threshold. The observed fold matrices mostly lie
in the two-scalar conformal subspace

```text
C(a, b) = [[a, -b],
           [b,  a]]
```

so a four-scalar continuous route is:

```text
C(g) = C(a0, b0) + g C(a1, b1)
delta_top(t) = g_t modal_top2(t) C(g_t)
```

`C(a0, b0)` is the shared direction and `C(a1, b1)` is a continuously
interpolated contrast. Setting the contrast to zero recovers the observed
Iteration 2 conformal route. Every prompt contributes to all four
coordinates—there is no 21/79 data split—and the fit remains linear in its
four learned scalars through `g` and `g^2` features. Bounding `C(-1)` and
`C(+1)` to operator norm `0.25` bounds every intermediate `C(g)` because
`g` lies in `[-1, 1]`.

This is the evidence-supported next rung: same learned-scalar count as
Iteration 2, continuous conditionality, and stronger partial pooling. Hard
magnitude thresholds are not yet justified by the retained statistics. If
this pooled model remains unstable or fails family-disjoint behavior, the
compiler should then test a new causal statistic or stop this branch rather
than grow more independent experts.

The report is bound by:

- collection hash
  `9789d185cca7001a399a02366b928cebfdc4d01bb0df22d8fa0f4a8ae4cfd1d0`;
- logical hash
  `c6642c16fd2620ad057b9fafcb53e21c02f29dd4dfb2f73c9e9aaf9e3f6d05a2`;
- file hash
  `0c0bcdd2c83e89a5bfd69bda8816da27e2834b0750dcc1792aaaeec9b609c6aa`.

The report contains scalar sufficient statistics and hashes only; it retains
no prompts, token IDs, logits, activations, gradients, or model weights.
Reproduce it with:

```bash
fisher-graph-gemma-l3-l4-iterative-state-experts-dev \
  --materialization-report-sha256 \
    27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94 \
  --materialization-report-file-sha256 \
    7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20 \
  --factorial-report-sha256 \
    9fde13193b3ee915845247de3821e84ec84dbae368a2e3554e4c9423927878ed \
  --factorial-report-file-sha256 \
    2e9fc09804af10199739ad6c14df658a6a81550e4e14b5eeb9170c52d5a809ab \
  --prior-iteration-report-sha256 \
    2836d9bde0d39a5b1acbaab7d34fca69c5ad7eab9a1632a63900565eb8ff2207 \
  --prior-iteration-report-file-sha256 \
    0a269c7f6336bccb601a868c157cea4dd4dce2d413a55ae7531471679d9f45f1 \
  --prior-iteration-collection-sha256 \
    f7ea40fd6bb5695ed9da5f21d8e8d279a8e257e00fb8b4c7bd204718b0c17b8c
```

## Iteration 4: pooled affine conformal routing

Iteration 4 executed the pooled candidate preregistered after the unstable
sign-expert result. The rejected Iteration 3 report is an immutable immediate
prerequisite, but its provider is not stacked into execution. The candidate
again starts from the accepted-X4 plus source-only lag-`B` iteration-zero
parent.

For continuous causal balance `g_t` and the selected top-two parent modes
`m_t`, the route is

```text
C(a, b) = [[a, -b],
           [b,  a]]

C(g_t) = C(a0, b0) + g_t C(a1, b1)
delta_top(t) = (g_t m_t) C(g_t)
```

The learned coordinate order is:

```text
[shared_real, shared_imag, contrast_real, contrast_imag]
```

This produces a four-dimensional parent-point Jacobian from shared `g m`
and contrast `g² m` features. Every prompt contributes to every coordinate,
unlike the 21/79 row split in Iteration 3. Setting the contrast pair to zero
recovers a shared conformal version of Iteration 2; setting all four
coordinates to zero exactly recovers the parent.

The fitter uses family-balanced ridge regression. It globally scales all four
coordinates when either endpoint exceeds operator norm `0.25`. Because

```text
C(g) = ((1 - g) / 2) C(-1) + ((1 + g) / 2) C(+1)
```

for `g ∈ [-1, 1]`, bounding both endpoints bounds every intermediate route.
Global radial scaling also preserves unsupported coordinates at exactly zero.

The live protocol remained the frozen 16-example, 8-family A-fit campaign:
16 source forwards and 16 parent VJP forwards in phase A, followed by 16
fresh source forwards and 16 family-disjoint candidate forwards in phase B.
That is exactly 64 Gemma model forwards. The campaign retained scalar
sufficient statistics and hashes only.

### Live decision

The candidate was **not retained**.

| metric | frozen parent | affine conformal route | relative result |
|---|---:|---:|---:|
| family-macro mean prompt-absolute ΔNLL/token | `0.268343` | `0.267479` | `+0.3217%` |
| strict family wins | — | `6/8` | passes |
| worst held family | — | state invariant | `-3.3200%` |
| per-prompt p90 absolute ΔNLL/token | `0.404262` | `0.393626` | `+2.6309%` |
| family-macro source-to-candidate KL/token | `0.891598` | `0.891961` | `-0.0407%` |
| family-macro top-1 disagreement | `0.542834` | `0.544595` | `-0.3246%` |

The macro, six-win, KL, disagreement, and p90 gates all passed. The sole
behavior failure was the preregistered rule that no held family may regress
more than `2%`.

| held family | relative improvement | result |
|---|---:|---|
| budget allocation | `+0.5792%` | win |
| constraint propagation | `-0.3964%` | loss within floor |
| counterfactual isolation | `+0.4767%` | win |
| hierarchical composition | `+0.8647%` | win |
| reference frame | `+0.3701%` | win |
| state invariant | `-3.3200%` | rejects candidate |
| temporal dependency | `+0.4252%` | win |
| uncertainty update | `+1.3284%` | win |

This is broader than Iteration 2 and materially more stable than Iteration 3,
but its macro gain is smaller than Iteration 2 and the same state-invariant
family remains outside the safety floor.

### What the state-invariant failure means

The held state-invariant family contains two prompts with opposite signed
parent errors:

| prompt receipt | parent signed ΔNLL/token | exact candidate | absolute-error result |
|---|---:|---:|---:|
| `8c6307…` | `+0.041768` | `+0.050919` | `-21.91%` |
| `8ea088…` | `-0.218256` | `-0.217738` | `+0.2376%` |

The same held-family route moved both signed errors upward. That is correct
for the second prompt, because its error was negative, but wrong for the
first. The larger prompt dominates the baseline magnitude, yet the first
prompt's `0.009151` increase is large enough to move family mean absolute
error from `0.130012` to `0.134328`.

The parent-point Jacobian predicted these two finite steps as `+0.008693` and
`+0.000380`; exact execution produced `+0.009151` and `+0.000518`. The
failure is therefore not a step that crossed a nonlinear valley. It is a
held-out conditional-direction error already visible at the parent point.

The prior sign-expert records provide a useful but now exploratory clue. The
first prompt had `39/64` negative-balance active rows, while the second had
only `1/64`. Iteration 3 could give the second prompt a much stronger
nonnegative-regime correction, which repaired the family, but its independent
experts were unstable across the other families. Iteration 4 stabilized the
fit by pooling, while also weakening the positive-balance endpoint enough to
lose that repair.

### Scientific and estimator-stability evidence

Every preregistered scientific gate passed:

- all eight weighted designs had rank four;
- all four conformal coordinates were supported in every fold;
- no fold coefficient vector had zero norm;
- family-macro balance-feature standard deviation was `0.154656`;
- the selected modes carried `87.1460%` of measured lag-`B` modal energy;
- median normal-equation condition number was `25.3466`, below the `100`
  ceiling;
- mean cosine across all 28 unordered fold-coefficient pairs was `0.984287`,
  above the `0.90` floor.

The stability comparison is:

| iteration | learned coordinates | median condition | mean fold cosine |
|---|---:|---:|---:|
| 2: shared `2 × 2` route | 4 | `69.9` | `0.972` |
| 3: independent sign experts | 8 | `555.7` | `0.686` |
| 4: affine conformal route | 4 | `25.35` | `0.984` |

This validates the partial-pooling hypothesis: conformal structure plus
continuous balance produced the best-conditioned and most cross-fold-stable
estimator in the branch. The remaining rejection is not an estimation-rank
failure.

All eight fits reached the `g=-1` endpoint bound of `0.25`. Their `g=+1`
endpoint norms ranged from `0.0595` to `0.0914`. Before projection, endpoint
norms ranged from `9.476` to `19.664` at `g=-1` and `2.871` to `6.445` at
`g=+1`; global trust scales ranged from `0.0127` to `0.0264`. There is strong
fit pressure, especially on negative-balance states, but the held-family
failure does not justify widening the trust region: its harmful direction
was already predicted correctly by the linearization.

The complete parent-point diagnostic was:

- predicted-versus-exact correlation: `0.999999904`;
- RMSE: `0.000163738` ΔNLL/token;
- sign agreement: `100%`;
- worst absolute prediction error: `0.000458541` ΔNLL/token;
- predicted macro improvement: `+0.3511%`;
- exact macro improvement: `+0.3217%`.

### Exact marginal resource receipt

All resource gates passed. Relative to the reused parent, the route adds:

- four learned float scalars;
- two derived prepared float constants;
- six prepared float scalars in total;
- two runtime-state floats per sequence;
- eight linear MACs/token;
- four linear accumulator scalar operations/token for causal balance state;
- at most five nonlinear scalar operations/token;
- one zero-denominator comparison/token;
- one reused parent-decoder invocation and one serving-model forward;
- no duplicated parent head.

These are exact logical receipts for the candidate edge. They do not by
themselves establish wall-clock speedup or full-model compression, and the
rejected provider has no retained full fit.

### Conservative next rung

The experiment answered the immediate question: smooth partial pooling fixes
the sign experts' estimator instability and reaches the required family-win
breadth. It does not supply enough causal state to choose the correction
direction for both state-invariant prompts.

A plausible next candidate is a partially pooled conformal route conditioned
on a second causal sequence statistic such as cumulative negative-balance
occupancy. That statistic is available online from the same modal stream and
could distinguish the observed `39/64` versus `1/64` regimes without prompt
IDs or semantic labels. It should not be implemented as another pair of
fully independent experts.

That hypothesis is now post hoc with respect to the current A-fit panel.
Testing it again on these same 16 examples would turn the failed held-family
receipt into feature-selection leakage. A confirmatory Iteration 5 therefore
needs a newly frozen development panel, or an explicitly nested inner
feature-selection split with untouched outer families, before any result can
count as new evidence.

The Iteration 4 report is bound by:

- collection hash
  `4958b0f64094758c13740c5ffa11fac41652af2dc6e01232c3b80c32155fde5c`;
- logical hash
  `48c708e8b717610b4ec018f9c129494c08d9027b6f05afef612ab79d91e938f1`;
- file hash
  `bad81101b0e07d75cca23ed85143d0f3d0c8649423fa84fdbefef08e55afd541`.

Reproduce it with:

```bash
fisher-graph-gemma-l3-l4-iterative-conformal-route-dev \
  --materialization-report-sha256 \
    27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94 \
  --materialization-report-file-sha256 \
    7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20 \
  --factorial-report-sha256 \
    9fde13193b3ee915845247de3821e84ec84dbae368a2e3554e4c9423927878ed \
  --factorial-report-file-sha256 \
    2e9fc09804af10199739ad6c14df658a6a81550e4e14b5eeb9170c52d5a809ab \
  --prior-iteration-report-sha256 \
    c6642c16fd2620ad057b9fafcb53e21c02f29dd4dfb2f73c9e9aaf9e3f6d05a2 \
  --prior-iteration-report-file-sha256 \
    0c0bcdd2c83e89a5bfd69bda8816da27e2834b0750dcc1792aaaeec9b609c6aa \
  --prior-iteration-collection-sha256 \
    9789d185cca7001a399a02366b928cebfdc4d01bb0df22d8fa0f4a8ae4cfd1d0
```

## Iteration 5 development screen: centered occupancy

Iteration 5 implemented the second causal statistic proposed after the
Iteration 4 state-invariant failure. It does not add a hard expert or a
prompt-conditioned branch. It augments the existing cumulative modal balance
`g_t` with one centered history variable `o_t`:

```text
z_t = +1 if g_t < 0 else -1

cumulative:
  S_t = S_(t-1) + z_t
  W_t = W_(t-1) + 1

EW:
  S_t = rho S_(t-1) + z_t
  W_t = rho W_(t-1) + 1
  rho = 2^(-1/16)

o_t = S_t / W_t
```

The current active token updates both `g_t` and `o_t` before routing.
Padding updates neither state. The route is

```text
C(g_t, o_t) = C(a0, b0) + g_t C(ag, bg) + o_t C(ao, bo)
delta_top(t) = (g_t m_t) C(g_t, o_t)
```

Its learned-coordinate order is:

```text
[
  shared_real,
  shared_imag,
  balance_contrast_real,
  balance_contrast_imag,
  occupancy_contrast_real,
  occupancy_contrast_imag,
]
```

One parent NLL VJP produces the two arms' shared first four coordinates and
their distinct occupancy pair. The fit is family balanced, has no intercept
or centering, and standardizes each training-only column:

```text
scale_j = sqrt(sum_i w_i x_ij^2)
Z_ij = X_ij / scale_j
beta = solve(Z^T W Z + 1e-6 I, Z^T W y)
theta_j = beta_j / scale_j
```

All six coefficients receive one global radial projection. The maximum
conformal operator norm is checked at all four `(g,o)` corners. Since the
route is affine on the square `[-1,1]^2`, the corner bound of `0.25` bounds
every interior controller state.

### Staged protocol and safe stop

The protocol has a 32-forward reusable development phase followed, only for
a scientifically supported arm, by a 64-forward fresh phase:

1. 16 direct source forwards;
2. 16 accepted-X4 plus lag-`B` parent VJP forwards;
3. eight leave-one-family-out fits for each occupancy arm;
4. select the passing arm with the lower predicted family-macro absolute
   delta NLL, with cumulative winning an exact tie;
5. fit and authenticate both full providers;
6. create a durable claim and open the new panel once; and
7. for each of 16 fresh prompts, run source, parent, cumulative, and EW
   forwards.

The real run stopped at step 4. Neither arm passed the frozen development
stability gates, so steps 5 through 7 were not used to make a result. The
fresh role input and prompt-free public artifact were prepared, but no durable
claim exists and the protocol did not open the prompt source. The old private
payload was later exposed during a local boundary audit, so it is permanently
disqualified from confirmatory use even though no campaign claim was created.

The development phase was replayed once solely to surface the same aggregate
gate values after the initial fail-closed exception. Each replay executed the
same 32-forward deterministic development protocol. No route, threshold, or
selection rule was changed between them.

### Results

| diagnostic | cumulative occupancy | EW occupancy |
|---|---:|---:|
| predicted family-macro mean absolute ΔNLL/token | `0.267742857` | `0.268012303` |
| family-macro occupancy standard deviation | `0.242362879` | `0.313540413` |
| raw median normal condition number | `418.261` | `456.395` |
| standardized median normal condition number | `282.512` | `303.341` |
| mean cosine over 28 fold-coefficient pairs | `0.749239` | `0.740604` |
| all six coordinates supported in all folds | yes | yes |
| all raw and standardized fold ranks equal six | yes | yes |
| scientific development gate | fail | fail |

Shared diagnostics were:

- family-macro balance standard deviation: `0.154655747`;
- top-two modal energy fraction: `0.871459933`;
- negative-balance active rows: `218`;
- nonnegative-balance active rows: `790`;
- both occupancy signs observed: yes.

This directly answers the concern about a fraction tending toward zero.
Neither occupancy signal was degenerate: the EW statistic was more variable,
not less, and cumulative occupancy also cleared the `0.05` variation floor
comfortably. The history coordinate added measurable information.

The failure was estimator identifiability. Column scaling removed unit and
magnitude imbalance, but cannot remove near-collinearity among the six
prompt-level Jacobian directions. Both standardized condition numbers
exceeded the preregistered `100` ceiling, and both fold cosines missed the
`0.90` stability floor by a wide margin. The cumulative arm had a marginally
better predicted objective, but it was ineligible; choosing it anyway would
use the development result to waive the safety rule.

The exact marginal resource envelopes, had either arm passed, were:

| resource | cumulative | EW |
|---|---:|---:|
| learned float scalars | 6 | 6 |
| prepared float scalars | 8 | 9 |
| runtime-state floats/sequence | 4 | 4 |
| marginal linear MACs/token | 10 | 10 |
| linear accumulator operations/token | 6 | 8 |
| explicit state multiplications/token | 0 | 2 |
| nonlinear scalar operations/token | at most 6 | at most 6 |
| parent decoder invocations/token | 1 | 1 |

These are logical edge receipts, not latency or compression claims. No
full-fit provider was retained and no deployment was authorized.

The prepared prompt-free panel is bound by:

- selection plan hash
  `512ea7e6b311ec7739e2198736a8f71da12034be8f1408c047220522d5a3913d`;
- panel artifact hash
  `1196c8680c985a322c2c6680293a7d826113fab6d178a9d0a218f0cf52ac42df`;
- panel file hash
  `356214833d14031d3b043899d8b658a8469d8fb1bbc117055c4e168418942aaa`;
- manifest hash
  `bbb1e35628e08b9b4e78a8bc43f3ed31454a72d6f6e8e67c5ce22cca73c050d`;
- membership receipt
  `f57a3f5b258af6a119ae0ed6ff432ed3a9078f0e7a682c1686c84bd13e3204a3`.

The ignored local files contain the private role input and public artifact.
They are not committed, and there is intentionally no selection-claim or
Iteration 5 final-report hash. The prepared artifact cannot authorize a later
candidate: it is bound to the old direct-fit recipe, and the corresponding
private payload is no longer blinded.

### Fold-local residualization result

The result does not justify weakening the condition or fold-stability gates.
It says that appending `g o m` to `g m` and `g² m` creates a poorly
identified six-coordinate basis on this development panel.

The first conservative repair was implemented as a development-only fit
coordinate change. For every arm and held-family fold, only the 14 training
prompts define:

```text
X = [B O]
A = (sqrt(W) B)^+ sqrt(W) O
R = O - B A
X_residual = [B R]
```

`B` contains the four shared/balance Jacobian columns and `O` contains the
two occupancy columns. The minimum-norm projection uses a scaled float64 SVD
and family-balanced training weights. The weighted cross-correlation
`B^T W R` is checked at `1e-10`; held-family rows never enter `A`, the column
scales, or the ridge fit.

If ridge in the residual coordinates returns
`gamma = [gamma_B, gamma_O]`, the result maps back into the original runtime
basis:

```text
theta_B = gamma_B - A gamma_O
theta_O = gamma_O
```

Therefore `X theta = X_residual gamma` exactly. The unchanged global
four-corner radial projection is applied after map-back and scales `gamma`
by the same scalar, preserving the identity. Unsupported residual columns
become exact-zero occupancy coefficients; the fitter never restores the raw
occupancy column as a fallback.

The runtime still evaluates `C0 + g Cg + o Co`. The projection `A` is not
deployed, so the resource envelope remains exactly the direct Iteration-5
envelope: six learned floats, four state floats per sequence, and 10 logical
linear MACs per token.

Before the replay, the development screen froze a `5%` minimum retained
occupancy-energy gate in addition to the existing rank, support, condition,
fold-cosine, variation, energy, and sign gates. The authenticated 32-forward
replay produced:

| diagnostic | cumulative | EW, half-life 16 |
|---|---:|---:|
| direct standardized median condition | `282.511779` | `303.340548` |
| residualized standardized median condition | `16.907425` | `16.907425` |
| residualized/direct condition ratio | `0.05985` | `0.05574` |
| median retained occupancy energy | `0.036315` | `0.034207` |
| minimum retained occupancy energy | `0.016290` | `0.015617` |
| mapped-runtime fold cosine | `0.749210` | `0.740578` |
| predicted family-macro absolute ΔNLL/token | `0.267742847` | `0.268012320` |
| development gate | fail | fail |

Every residualized fold retained raw and standardized rank six, both
occupancy coordinates remained numerically supported, and weighted
orthogonality held to at most `1.54e-14`. Numerically, the repair succeeded:
condition fell well below `100`.

Scientifically, it did not. Only about `3.4%` to `3.6%` of occupancy energy
was independent of the existing controller, with worst-fold values near
`1.6%`. The mapped coefficients—the coordinates the executor would actually
serve—had essentially the same unstable cosine as before. The predicted
objective changed by only `1e-8`. Residualization removed duplicated signal
but did not uncover a stable new compute direction.

No arm was selected, no full provider was retained, and no fresh boundary was
crossed. The prompt-free report is bound by:

- logical report hash
  `3ea83a89db0fe4f9f73f727783acddde3292a26a151e8d7057b8a6a4db6c1cbf`;
- ignored local report file hash
  `a7b6f483c4dd6376ab7a61a36cd991e620e11fe5b935996ad32bc86fcb96ac1f`.

### Next identifiable rung

The clean next candidate is a separately preregistered one-dimensional
residual-SVD controller with a canonical sign and orientation. That asks
whether the small two-column remainder contains one stable common direction,
rather than allowing two fold-dependent coordinates to rotate. It must be a
new development rung, not an adaptive fallback inside this result.

If that candidate passes reusable development, confirmation still requires a
new recipe commitment and a newly generated blinded panel. The old panel is
both plan-incompatible and disqualified by audit exposure.

## Exact token-loss Fisher development rung

The prompt-level occupancy experiments reduce one summed-NLL VJP to one
Jacobian row. That is an exact first-order prompt derivative, but token
contributions can cancel before the fit sees them. The new rung retains the
loss-token geometry long enough to form an empirical Fisher in the frozen
route-coordinate basis.

For supervised loss token `t` and declared route coordinate `k`, the bridge
computes

```text
Q[t, k] = d loss_t / d alpha_k
```

from an exact per-token H4 VJP. The shared tangent bank contains the eight
unique coordinates:

```text
[
  shared_real,
  shared_imag,
  balance_contrast_real,
  balance_contrast_imag,
  cumulative_occupancy_contrast_real,
  cumulative_occupancy_contrast_imag,
  ew_occupancy_contrast_real,
  ew_occupancy_contrast_imag,
]
```

The cumulative arm reads indices `[0,1,2,3,4,5]`; the EW arm reads
`[0,1,2,3,6,7]`. As a mandatory checksum, summing the token rows and dividing
by the supervised-token count must reproduce each arm's existing
six-coordinate summed-NLL prompt Jacobian. Inactive padding contributes
nothing, and gradient energy at an active future position is a causal
leakage error.

For compensation target `z_t = source_nll_t - parent_nll_t`, each prompt with
`N` supervised tokens is reduced to:

```text
A  = Q^T Q / N
b  = Q^T z / N
c  = z^T z / N
mu = sum_t Q[t] / N
```

`A` is the exact token-loss empirical Fisher pulled back into the route
coordinates. The prompt Fisher record retains only scalar sufficient
statistics and authenticated hashes; it does not retain prompt text, token
IDs, logits, activations, model weights, raw H4 gradients, or the
compensation-target vector. The development artifact separately retains the
reduced eight-coordinate token rows so their Fisher moments can be replayed.

### Fixed collection and validation cost

The reusable panel remains the 16-prompt, eight-family A-fit set. Collection
per prompt is:

1. one direct source forward for the finite-NLL authority;
2. one accepted-X4 plus lag-`B` parent forward with the H4 autograd graph;
3. exact token-loss VJPs in cotangent chunks of eight.

The complete development collection therefore uses exactly 32 model
forwards: 16 source plus 16 parent-VJP forwards. If prompt `i` has `N_i`
supervised tokens, its retained graph uses `ceil(N_i / 8)` batched backward
calls, and the campaign total is
`sum_i ceil(N_i / 8)`. Chunking changes peak cotangent/VJP memory and backward
call count, not the exact `Q` rows.

Fitting is ridge-free standardized minimum-norm least squares over prompt
sufficient statistics. The weighting hierarchy is deliberate:

```text
equal mass per training family
  -> equal mass per prompt within that family
    -> equal mass per supervised token within that prompt
```

Every validation fold holds out an entire family. No random token split is
permitted, so the larger number of token rows cannot turn correlated tokens
from one prompt into nominally independent held-out examples. The frozen
gates cover rank, standardized condition, fold-coefficient cosine,
family-macro held RMSE improvement, family wins, worst-family regression,
and occupancy energy incremental to the four shared/balance coordinates.

### Coupling is not direction

The reported graph uses normalized off-diagonal entries of the
family-balanced Fisher. It is symmetric by construction:

```text
F[i, j] = F[j, i]
```

These values nominate stable co-sensitive pairs among tangents whose causal
runtime direction was already declared. They do not infer that generator
`i` causes generator `j`, or the reverse. A passing development arm may only
advance to held-family finite-displacement validation and explicit
JVP/intervention orientation. Fisher coupling alone cannot authorize graph
traversal, merging, pruning, or compilation.

### Current boundary

This is a development-only mapping rung, not a compiled model result. It uses
zero candidate forwards and zero fresh-panel forwards; no selection panel is
referenced or opened. It creates no provider and adds no serving parameters
or serving MACs. Consequently it currently establishes neither compression
nor inference speedup, even if its reusable-development gates pass.

### Exact A-fit result

The authenticated local run completed over all `16` reusable prompts and
`1,157` supervised loss tokens:

| metric | cumulative | EW |
| --- | ---: | ---: |
| full Fisher rank in every LOFO fold | `6/6` | `6/6` |
| median standardized condition | `38.120` | `38.702` |
| minimum incremental occupancy Fisher energy | `9.349%` | `9.122%` |
| stable Fisher couplings | `6` | `6` |
| mean pairwise fold-coefficient cosine | `0.565` | `0.607` |
| family-macro held RMSE change | `-0.573%` | `-0.542%` |
| held-family wins | `2/8` | `2/8` |
| worst-family RMSE change | `-2.271%` | `-2.166%` |

The collection used the frozen resource envelope exactly:

```text
16 source forwards
+ 16 retained parent token-VJP forwards
= 32 total model forwards

153 batched backward calls at chunk size 8
0 candidate forwards
0 fresh forwards
```

The mapping result is positive. Prompt aggregation had previously left only
roughly `1.5%` to `3.6%` independent occupancy energy; exact loss-token
Fisher raises the worst fold/coordinate above `9%`. Both arms recover six
couplings with the same sign in all eight training-fold maps. Structurally,
the six-node graph is two disconnected dense triangles:

```text
shared_real --- balance_real --- occupancy_real
     \__________________________________/

shared_imag --- balance_imag --- occupancy_imag
     \__________________________________/
```

The strongest edges are balance-to-occupancy (`-0.925` cumulative,
`-0.927` EW), followed by shared-real to occupancy-real (about `-0.74`) and
shared-real to balance-real (`+0.734`). This is a real, family-stable
co-sensitivity map; it is still symmetric and therefore not a causal
orientation.

The mutation result is negative. A single global six-coordinate coefficient
vector does not transfer across prompt families. The shared coefficients
keep a consistent negative sign, but balance and especially occupancy
coefficients rotate across folds; the occupancy-real coefficient is positive
in only `4/8` folds in both arms. A post-run diagnostic view confirms the
separation:

```text
shared-only: coefficient cosine 0.955, macro RMSE change -0.004%
base four:   coefficient cosine 0.871, macro RMSE change -0.195%
full six:    coefficient cosine 0.565/0.607,
             macro RMSE change -0.573%/-0.542%
```

Thus exact token evidence fixed the observability/cancellation problem but
did not justify a global linear executor. No arm passed, no provider was
compiled, and no finite-displacement or fresh-panel claim is authorized.
The next development candidate should freeze the six stable couplings and
test a very small causal token-conditioned router or coefficient expert over
them, still trained and evaluated by whole held families. It must not relax
the failed gates or treat tokens as independent validation examples.

The ignored report is
`.local-runs/google--gemma-3-270m/progressive-a-iterative-token-loss-fisher-dev-v1.report.json`,
with logical hash
`6ffaf61639626b47101324573fff646de187f45212b29c88f236749bb2beb65b`
and file hash
`d80d78580102168c031a200472bd8c0259a264f2e4d8fc269a5a73b1ccd363b9`.

The CLI is:

```bash
fisher-graph-gemma-l3-l4-iterative-token-fisher-dev \
  --materialization-report-sha256 <logical-sha256> \
  --materialization-report-file-sha256 <file-sha256> \
  --factorial-report-sha256 <logical-sha256> \
  --factorial-report-file-sha256 <file-sha256>
```

## Partially pooled token-Fisher corrective screen

The exact-token result separated a stable Fisher map from an unstable
six-coordinate mutation fit. The next adaptive-development screen asks the
narrowest corrective question answerable from its retained prompt moments:
can strong partial pooling preserve a shared conformal trunk while retaining
only transferable balance/occupancy deviations?

This is not a new runtime feature. The six-coordinate occupancy route was
already token conditioned:

```text
a_t = a0 + g_t ag + o_t ao
b_t = b0 + g_t bg + o_t bo
C_t = [[a_t, -b_t],
       [b_t,  a_t]]
```

The first two coefficients `(a0,b0)` are the shared trunk. The last four are
the balance and occupancy deviations. The corrective fit standardizes the
six exact-token Fisher coordinates inside each training fold, gives the
shared pair a fixed `1e-6` ridge, and selects one common deviation ridge from

```text
{0.1, 1, 10, infinity}
```

`infinity` is the exact shared-only control: all four conditional deviations
are zero. Every outer held-family fold selects its ridge with a second,
seven-way family-held-out loop over the remaining training families. The
selection rule chooses the strongest ridge within one standard error of the
best inner mean RMSE ratio. Projection onto the existing `0.25` four-corner
operator trust bound happens before every inner and outer score. No held
family participates in standardization, ridge selection, or projection.

The conditional comparison is materiality-gated rather than merely
sign-gated. It must improve family-macro RMSE over shared-only by at least
`0.5%`, and a family counts toward the required `5/8` incremental wins only
after at least `0.1%` improvement. An exactly shared-only negative control
therefore cannot pass because of floating-point dust.

Cumulative occupancy was fixed as the primary arm. EW is sensitivity-only
and cannot rescue a failed primary. The screen reused the 16 authenticated
prompt sufficient-statistic records and performed zero model forwards,
backward calls, candidate executions, or fresh-panel reads.

### Result

Both arms selected `infinity` in every one of the eight outer folds. The
conservative selector therefore removed every balance and occupancy
deviation:

| metric | cumulative primary | EW sensitivity |
| --- | ---: | ---: |
| folds selecting a conditional deviation | `0/8` | `0/8` |
| family-macro held RMSE improvement | `+0.1726%` | `+0.1726%` |
| held-family wins | `6/8` | `6/8` |
| worst-family RMSE change | `-0.4854%` | `-0.4854%` |
| mean fold-coefficient cosine | `0.9549` | `0.9549` |
| median standardized condition | `38.120` | `38.702` |
| incremental improvement over shared-only | `0%` | `0%` |

This is a useful stabilization result, but it is not a corrective-router
success. Relative to the unregularized exact-token fit, coefficient cosine
rose from `0.565/0.607` to `0.955`, family wins rose from `2/8` to `6/8`,
and the worst regression contracted from roughly `-2.2%` to `-0.49%`.
However, all of that came from falling back to the two-coordinate shared
route. The primary still missed the preregistered `2%` macro improvement
floor, and conditional behavior provided exactly no incremental benefit.

The finding is therefore sharper than “use more regularization”:

> the balance and occupancy coordinates are real Fisher directions, but the
> current balance/occupancy state variables do not predict a transferable
> correction over them.

No provider, finite-displacement run, graph traversal, causal orientation, or
fresh confirmation is authorized. This panel was also used to generate the
partial-pooling hypothesis, so the result is adaptive development rather
than independent confirmation.

The next rung must freeze a genuinely new causal runtime feature before
collecting new family-disjoint data. A compact candidate is a fixed
channel-factored Fisher basis `U` with a four-column design

```text
R_t = [Q_t U, o_t Q_t U]
```

or an equally small current-token modal-innovation controller. Unlike the
present shrinkage screen, either candidate requires a new token collection:
the existing artifact retains `Q^T z`, but not the tokenwise target and state
products needed for new columns. Its basis, feature formula, regularization,
trust proof, and gates must be hashed before that collection.

Run the replay-only screen with:

```bash
fisher-graph-gemma-l3-l4-iterative-fisher-corrective-dev \
  --token-fisher-report-sha256 \
    6ffaf61639626b47101324573fff646de187f45212b29c88f236749bb2beb65b \
  --token-fisher-report-file-sha256 \
    d80d78580102168c031a200472bd8c0259a264f2e4d8fc269a5a73b1ccd363b9
```

The ignored corrective report has logical hash
`0d8274891904014c41f9263bb5899f9e50b4f3caf17dfb9e4279a2eb2585ed47`
and file hash
`7219444fc568ad002d317190313f3b19acce2ecfb28d9e557ed985d20e6bd0f1`.

## Fixed-basis causal generator-innovation screen

The next rung froze the generator map before opening any new prompt. It
factorized the prior six-coordinate token Fisher into a channel-factored
`6 × 2` basis `U`. The basis retained `79.20%` of source Fisher trace and was
stable under source-family deletion: the minimum leave-one-family-out basis
cosines were `0.9987` and `0.9982`.

The new runtime feature used only the frozen parent's two leading modal
values and earlier active positions. For active token `t`,

```text
x_t = parent_top2_t / frozen_positive_scales
p_t = exponentially_weighted_mean(x_<t), half-life 16
h_t = softsign(x_t - p_t)
```

The prior is emitted before the current row updates the state. Padding emits
zero and leaves the three-float per-sequence carry unchanged. Whole-sequence
and chunked evaluation are exactly equivalent. The four generator
coordinates are

```text
R_t = [
  (Q_t U)_real,
  (Q_t U)_imag,
  h_t,real (Q_t U)_real,
  h_t,imag (Q_t U)_imag
]
```

The multiplication by `h_t` occurs on each activation-position tangent
before contraction with the token-loss gradient. Multiplying a
prompt-aggregated derivative afterward would not be the same experiment.

The collection panel contained 16 new prompts in eight new semantic
families, two prompts per family. Its prompt and family identities were
disjoint from the data used for the parent fit, Fisher basis, prior occupancy
experiments, and sealed Calibration B. The worker retained only exact Q6/R4
prompt sufficient statistics, aggregate feature receipts, and hashes—no
prompt text, token IDs, logits, activation rows, gradient rows, or feature
rows.

Four arms were compared in nested family leave-one-out:

1. zero correction, called the parent baseline;
2. the prior two-coordinate legacy shared fit;
3. the fixed-`U` two-coordinate static generator fit;
4. the fixed-`U` four-coordinate conditional generator fit.

Each outer held family was excluded from normalization, fitting, projection,
and ridge selection. A second family-held-out loop selected the conditional
ridge from `{0.1, 1, 10, infinity}` using the one-standard-error rule.
`infinity` is the exact static-generator control. All fits used equal family,
then equal prompt, then equal token weighting and the frozen `0.25`
16-corner operator trust bound.

### Result

The map transferred, but the proposed conditional feature did not:

| metric | result | preregistered requirement |
| --- | ---: | ---: |
| fixed-basis Fisher trace coverage on the new panel | `83.488%` | at least `50%` |
| macro RMSE, parent baseline | `1.991958` | control |
| macro RMSE, legacy shared | `1.967239` | control |
| macro RMSE, static generator | `1.971997` | control |
| macro RMSE, conditional generator | `1.971997` | candidate |
| conditional improvement over parent | `+1.002%` | at least `+2%` |
| conditional improvement over legacy shared | `-0.242%` | at least `0%` |
| conditional improvement over static generator | `0%` | at least `+0.5%` |
| parent-baseline family wins | `8/8` | at least `6/8` |
| material static-generator family wins | `0/8` | at least `5/8` |
| folds with active conditional coefficients | `0/8` | at least `5/8` |
| worst-family change versus parent | `+0.256%` | no worse than `-2%` |
| median standardized condition | `4.065` | at most `100` |
| minimum residual conditional-design energy | `74.44%` | at least `5%` |

Every outer fold selected `infinity`. The finite ridge candidates became
steadily better as they were shrunk toward the static model, and the exact
static model had the best mean inner-family ratio. This is stronger evidence
than “the conditional columns were numerically broken”: all designs had rank
four, conditioning was good, and the conditional columns retained substantial
energy beyond the shared span. They simply did not provide transferable
target prediction under the frozen feature.

There is one concrete diagnostic clue. Across the 16 prompts, the
per-prompt mean absolute bounded innovation averaged `0.958` for the real
channel and `0.967` for the imaginary channel; even the smallest per-prompt
means were `0.873` and `0.901`. A softsign feature this close to magnitude
one is behaving mostly like a sign bit. The frozen positive scales therefore
appear mismatched to the raw innovation magnitude. The report intentionally
retains no raw rows, so this result alone cannot establish that a different
temperature would succeed.

The formal decision is to stop. No finite displacement, provider, graph
traversal, runtime, fidelity, or compression claim is authorized. The run
used 1,184 supervised tokens, 16 source forwards, 16 retained-parent
token-VJP forwards, 154 backward calls, and zero candidate or finite
forwards.

The scientifically clean next experiment is innovation v2:

1. use already-open development data to characterize raw innovation scale
   without fitting to semantic labels or loss outcomes;
2. freeze either robust per-channel temperatures or a bounded
   direction-plus-magnitude feature;
3. hash the feature, trust proof, ridge policy, and gates; and
4. collect another family-disjoint panel for the nested screen.

This preserves the prompt-blind generator map. Prompts remain an evaluation
instrument rather than the source of generator semantics.

Run the local development diagnostic with:

```bash
fisher-graph-gemma-l3-l4-generator-innovation-dev \
  --generator-panel-receipt-sha256 \
    b49b8f96125777cf0f245917fc7b9dd146b3841d4cd8f5ed27fa943a437bc2f2 \
  --generator-panel-receipt-file-sha256 \
    37b858d36fc5c0627a739ddec247859323047f661642c74b958afc7aab2c3714 \
  --generator-private-role-input-file-sha256 \
    5b0a8f0a7eafc1def5c7fb068a659fc8411ea59f9a624b3e1258b3d570c16000 \
  --materialization-report-sha256 \
    27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94 \
  --materialization-report-file-sha256 \
    7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20 \
  --factorial-report-sha256 \
    9fde13193b3ee915845247de3821e84ec84dbae368a2e3554e4c9423927878ed \
  --factorial-report-file-sha256 \
    2e9fc09804af10199739ad6c14df658a6a81550e4e14b5eeb9170c52d5a809ab
```

The ignored local report has logical hash
`d72ee86d652a7d0d1d1a77c570df67e8f7dcd23c31fccd16d0e92819f5e0855a`
and file hash
`d12c5a5716d43dfee32b01c5a59ec67abb34b9ee863e8a3a8f75c5dddda5018e`.

## Innovation-v2 scale and memory diagnostic

Innovation v2 used the same already-open `16`-prompt, eight-family panel as a
development diagnostic. It did not reinterpret that panel as fresh
confirmation. The experiment was split into three sealed stages:

1. an activation-only scale pass with no source authority, targets, losses,
   gradients, or candidate outputs;
2. a replay-only plan freeze that calibrated a fixed candidate bank; and
3. one target/VJP pass that reproduced every pre-target feature hash before
   contracting a shared Q6 tangent bank into all candidates.

The scale pass measured four prompt-blind feature sources: current-token
innovation and exponentially weighted priors with half-lives `4`, `16`, and
`64` active positions. For each source and channel, the temperature was the
median across prompts of the prompt median absolute raw innovation. Each
source then contributed multipliers `{0.5, 1, 2}`. The exact
EW16/unit-temperature v1 feature remained a scored control but could not enter
adaptive selection.

| source | real temperature at multiplier 1 | imaginary temperature at multiplier 1 |
|---|---:|---:|
| current only | `74.543444` | `81.595570` |
| EW4 | `83.555544` | `82.345802` |
| EW16 | `76.408218` | `82.399963` |
| EW64 | `77.023519` | `82.889817` |

The result confirms a real v1 scaling defect. The exact v1 control had
prompt-balanced `q90 |h| = 0.99636/0.99654` and central fractions
`0.08250/0.08581`. It failed the target-blind feature-health gate. All 12
calibrated variants passed: their prompt-balanced `q90 |h|` values ranged
from `0.62533` to `0.88186`, and their central fractions ranged from
`0.67538` to `0.94088`. Thus the calibrated bank genuinely tested graded
pedal positions rather than twelve nearly binary copies.

The target pass fit every candidate and ridge in nested whole-family
leave-one-out. The one-standard-error order was frozen as exact static first,
then larger ridge, then candidate simplicity. Static was represented exactly
once, and all candidates shared the same two static columns, compensation
target, fixed basis, tangent build, and gradient contraction.

### Result

| arm | family-macro RMSE | improvement vs parent | selected conditional folds |
|---|---:|---:|---:|
| parent | `1.991958` | control | n/a |
| legacy shared | `1.967239` | `+1.241%` | n/a |
| static fixed-U | `1.971997` | `+1.002%` | `0/8` |
| exact v1 portfolio | `1.971997` | `+1.002%` | `0/8` |
| scaled-L16 portfolio | `1.971997` | `+1.002%` | `0/8` |
| current-only portfolio | `1.971997` | `+1.002%` | `0/8` |
| full L4/L16/L64 portfolio | `1.971997` | `+1.002%` | `0/8` |

All eight outer folds chose static in all three adaptive portfolios. More
importantly, static was also the minimum-mean candidate in every inner
family screen. No finite fixed arm beat it:

| fixed-arm summary | relative change vs static |
|---|---:|
| closest finite arm: exact v1, ridge 10 | `-0.0188%` |
| best calibrated arm: EW64 × 0.5, ridge 10 | `-0.0283%` |
| mean across finite arms, ridge 10 | `-0.0363%` |
| mean across finite arms, ridge 1 | `-0.1699%` |
| mean across finite arms, ridge 0.1 | `-0.2719%` |

Negative values are degradations. Every finite fit had a materially active
conditional coefficient, but increasing its influence made held-family error
worse. The diagnostic therefore rejects all three proposed explanations:

- **scale rescue:** failed; calibrated L16 improved over neither v1 nor
  static;
- **memory rescue:** failed; the full temporal grid improved `0%` over
  scaled L16 and selected a non-L16 feature in `0/8` folds; and
- **temporal value:** failed; the full temporal grid improved `0%` over the
  current-only control.

No recipe was nominated. The conclusion is narrower than “causal state can
never help”: it says that this fixed-basis, globally linear, two-channel
softsign controller does not provide transferable correction through scalar
temperature changes or the tested memory lengths. The next useful analysis
should inspect why the conditional target relation changes across prompts or
families—most plausibly with richer direction-plus-magnitude interactions or
a token-conditioned corrective edge—rather than extending the same scalar
temperature/window grid.

The scale pass used `16` parent forwards and no losses or gradients. The
target pass used `16` source-authority forwards, `16` retained-parent
token-VJP forwards, `154` backward calls, one Q6 tangent-bank build and one
gradient contraction per example, and zero candidate, finite-displacement,
or provider forwards. The adaptive geometry gates remain explicitly
unevaluated, but the performance and attribution gates already failed.
Finite validation and provider compilation remain closed.

The ignored local artifacts are:

| artifact | logical hash | file hash |
|---|---|---|
| scale receipt | `2620f0c4041ff8b10624e55419a18513a21a94742007f989d919d3c02c00cd5a` | `114678671a730d1e917d8602d0f55579ee325685d8c33144baa356cbea5ae090` |
| scale development report | `ec92cb82d7dbd818dabf3b17df360283a0e1b6d396577c94527131cc2fcb94b3` | `d66163b6a09167e123383f2b070f0ebf34f2e768adbb0655277763640aabd704` |
| candidate plan | `7409121af103638f62ae0ae238da6ea22ac093bd447c649f323b3cd6b77e2db7` | `9446e62d880af68c6067616a306c729177e0caa85911187f6dbcddcb6ae8a65a` |
| target development report | `712b449817b48e0cf9c1de90837845bc99f2e2a4c2dd0421c6631111f7f00d85` | `ffe8591c7183220fe5cc3cf5d6eec3c927951948797423ede519351a941315df` |

Run the three sealed stages with:

```bash
fisher-graph-gemma-l3-l4-generator-innovation-v2 scale [frozen lineage flags]
fisher-graph-gemma-l3-l4-generator-innovation-v2 plan [scale and v1 hashes]
fisher-graph-gemma-l3-l4-generator-innovation-v2 target [plan, scale, and v1 hashes]
```

Each publisher is write-once, distinguishes logical hashes from file hashes,
and performs full replay before publication. All four final on-disk artifacts
also pass load-time validation and replay.

## Complete-H4 token-Fisher tail rate curve

The complete-H4 identity audit established a boundary at which replacing the
native H4 state recovers native logits bitwise. The next experiments froze the
authenticated D320 one-pass carrier, defined the omitted field as

\[
E=(I-D^\top D)(H4_{native}-H4_{graph}),
\]

and fitted the full 320-dimensional complement in whole-family leave-one-out.
The first ladder ordered those fitted tail axes by support-only endpoint token
Fisher score and executed actual cast-once finite corrections at
`K = 8, 16, 32, 64, 320`.

The finite result was much stronger than the endpoint tangent suggested. At
K64, the family-macro absolute NLL gap fell from `0.53748` to `0.06418`, an
`88.06%` improvement. Geometry already passed, and ordinary/support/core
aggregate delta NLL and KL were inside their limits. Top-1, prompt robustness,
and the 13-token causal-tail ledger still failed, however, so K64 was a near
miss rather than a passing compiler point. The full K320 sentinel recovered H4
and logits bitwise.

Because that first run left a failing K64 lower bracket and a passing K320
sentinel, the adaptive same-A follow-up fixed
`K = 64, 96, 128, 160, 192, 256, 320` before execution. It required rank-64 and
rank-320 observation receipts to reproduce the parent exactly.

| tail K | family-macro absolute NLL gap | improvement from D320 | full NRMSE | tail NRMSE | all gates |
|---:|---:|---:|---:|---:|:---:|
| 64 | `0.06418` | `88.06%` | `0.01782` | `0.03736` | no |
| 96 | `0.05327` | `90.09%` | `0.01381` | `0.03044` | no |
| 128 | `0.03624` | `93.26%` | `0.01095` | `0.02444` | no |
| 160 | `0.02648` | `95.07%` | `0.00874` | `0.01851` | no |
| 192 | `0.02418` | `95.50%` | `0.00677` | `0.01377` | no |
| 256 | `0.01285` | `97.61%` | `0.00362` | `0.00688` | yes |
| 320 | `0` | `100%` | `0` | `0` | yes |

K256 is the first **tested** sub-sentinel rank that clears all four behavioral
ledgers, all per-prompt robustness checks, and all full/core/tail geometry
checks. Its ordinary delta-NLL/KL/top-1 is
`+0.000563 / 0.003640 / 96.885%`; complete-H4-support is
`+0.000653 / 0.004220 / 96.389%`; causal-tail is
`-0.021682 / 0.004401 / 100%`. All eight families improve.

This is a valid capacity result, but not yet useful compression. D320 plus K256
retains 576 of 640 H4 directions, or 90% of this diagnostic span. More
importantly, the Fisher order is almost the residual-PCA order: foldwise
Spearman correlation is about `0.995`. Endpoint linear RMSE improves only about
29% even when the finite K320 correction is exact. The current axiswise
Fisher score therefore does not identify a substantially denser basis and the
endpoint derivative does not describe the whole displacement.

The next rung addresses both limitations together:

1. use exact per-token `KL(native || candidate)` H4 VJPs so selection targets
   the KL and top-1 failure directly;
2. form signed off-diagonal operators in `null(D320)` so a selected direction
   may combine many PCA axes rather than merely reorder them;
3. compare against a same-teacher-KL fixed-PCA/diagonal control; and
4. if endpoint derivatives do not close the K320 displacement, replace them
   with a fixed Gauss-Legendre path-integrated VJP before considering
   conditional residual edges.

The adaptive report is intentionally ignored by Git:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-lofo-adaptive-expanded-ladder-a-fit16-dev-v2.json
```

Its logical report hash is
`26010938b5b81dbce9e05607acd46e5b9e0beea1d981edbd91d5e841365799fa`
and its file hash is
`e7736b60084c8e5bbb83f44cc77613e09242848230137def3afa862162284721`.
The rank grid was selected after inspecting the first A16 result, D320 itself
was fitted on all A16 families, and each finite correction instantiates the
held native tail. Accordingly this remains truth-leaking same-A hypothesis
evidence. It does not open a fresh confirmation panel, authorize serving, or
support a model-compression or latency claim.

## Candidate-conditioned K64 gain refits V3, V4, and V5

The next three rungs asked whether the failing K64 tail could be repaired by
changing the gains on its **existing 64 directions**, rather than adding more
directions. All three rungs operated at the realized cast-once K64 candidate.
For
each outer held family they split the other seven families into disjoint fit
and tune prompt roles, recollected exact token
`KL(native || candidate)` VJPs with respect to the realized H4 state, selected
only from exact finite candidate executions, and then evaluated the selected
arm once on the untouched outer family. The held family never selected its own
step.

### V3: residual-Gauss–Newton abstention

V3 fitted one damped residual-Gauss–Newton gain direction for the
half-expected squared token teacher KL and tested
`alpha = 0, 0.25, 0.5, 1`. This was explicitly not a mean-KL natural gradient
or exact generalized Gauss–Newton claim. Exact tune execution rejected every
positive step in every outer fold: all `8 / 8` folds selected the alpha-zero
unit K64 fallback. The final selected arm was therefore exactly the unit arm:

- support family-macro teacher KL: `0.0493263792 → 0.0493263792`;
- held-family improvements: `0 / 8`;
- ordinary aggregate/family-macro top-1: `91.8367% / 91.8096%`;
- support aggregate/family-macro top-1: `90.5355% / 90.4940%`; and
- graph-core aggregate/family-macro top-1: `90.5063% / 90.4705%`.

This is a useful fail-closed result: finite execution prevented a locally
predicted direction from being reported as a gain. The classification is
`candidate_conditioned_k64_gain_refit_not_supported_same_a`. The ignored
write-once report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-gain-refit-lofo-a-fit16-dev-v3.json
```

Its logical report hash is
`01405ea80856ed4ba2cd78126e7b318f9475f4f65d48aa4ea21d02654e7fe937`
and its file hash is
`178fafdd9554650a0d90bddc5ce485e252dcd476a395c0423931fbe50202c388`.

### V4: mean-KL OPG direction does not transfer

V4 reused one authenticated 56-bank VJP collection to reproduce the V3
residual artifacts exactly and to fit a second direction without additional
backward calls. The primary direction was family-equal mean-teacher-KL descent
preconditioned by the full `64 x 64` empirical outer-product-of-gradients
matrix plus the frozen relevance regularizer. It was not labeled a Hessian,
natural gradient, exact GGN, or Gauss–Newton solve. Exact tune execution tested
`alpha = 0, 0.125, 0.25, 0.5, 1`. A reversed residual direction tested
`beta = 0, 0.125, 0.25, 0.5` as a diagnostic control only; it could not make
the primary arm pass.

The smaller mean-KL grid found a positive tune step in `3 / 8` outer folds,
but none generalized to the corresponding held family:

| held-family result | unit K64 | selected mean-KL OPG | selected reverse residual |
|---|---:|---:|---:|
| family-macro support teacher KL | `0.0493263792` | `0.0493795549` | `0.0493263792` |
| relative KL improvement | — | `-0.1078%` (`+0.1078%` regression) | `0%` |
| families improved | — | `0 / 8` | `0 / 8` |
| folds selecting a positive tune step | — | `3 / 8` | `0 / 8` |

Top-1 stayed inside the preregistered retention envelope, but it did not turn
the KL result into a win. Aggregate ordinary top-1 changed
`91.8367% → 91.7293%`, complete-H4-support changed
`90.5355% → 90.4110%`, graph-core changed `90.5063% → 90.3797%`, and
the 13-token causal-tail ledger stayed at `92.3077%`. All deciding aggregate
and family-macro top-1 values remained above `90%`, and every regression was
below the one-percentage-point tolerance. Geometry also remained report-only:
neither top-1 retention nor geometry can substitute for the failed held-family
KL gate.

The exact campaign used `632` model forwards and `494` backward calls. The
second direction added zero backward calls; the count includes parent
recollection, the shared candidate-gradient bank, eight exact tune candidates
per prompt/fold, and the final unit/mean/reverse executions. Its classification
is `tested_mean_KL_OPG_direction_not_supported_same_a`. The ignored write-once
report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-gain-refit-lofo-a-fit16-dev-v4.json
```

Its logical report hash is
`2ace5239314d5497e1c50ef17e3820ab041e99d5c19fe9d8443b0d9505f248c2`
and its file hash is
`5cad2c81694d9a0122ffe60df1c2bc1222395ddccc661b3e0751e2d78904ed50`.

### V5: the `1 / 64` microstep fits tune data but not held families

V5 closed the most immediate ambiguity in V4: perhaps `alpha = 0.125` was
simply too large. It authenticated and reproduced V4's eight refits and all 56
gradient receipts, reused V4's 56 unit tune observations without reexecuting
them, and executed exactly `+1 / 64` and `-1 / 64` in each tune cell. The
negative arm was a central-slope sign control: it could veto a positive step,
but it was never selectable, never executed on the held family, and could not
independently authorize the primary result. A fold selected `+1 / 64` only if
the central slope was negative, the exact positive execution cleared the
absolute/relative improvement floor, at least four of seven tune families were
nonworse, and every tune family stayed inside the five-percent cap.

That smaller step did work on the data that selected it. Six of eight outer
folds selected the positive microstep; obsidian and shell retained the unit
fallback because their positive improvement did not clear the preregistered
floor. Across the eight tune-fold summaries, the selected arm changed
family-equal teacher KL from `0.0198052518` to `0.0197967478`, a `0.04294%`
improvement. The held-family result went the other way:

| V5 held-family result | authenticated unit K64 | selected unit or `+1 / 64` |
|---|---:|---:|
| family-macro support teacher KL | `0.0493263792` | `0.0493284360` |
| relative KL improvement | — | `-0.00417%` (`+0.00417%` regression) |
| families improved | — | `2 / 8` |
| folds selecting the positive tune step | — | `6 / 8` |

Cave and kiln improved on their held prompts; alpine, reed, sundial, and varve
worsened; obsidian and shell were unchanged because they selected unit. The
worst-family five-percent cap passed, but the preregistered `6 / 8` held-family
improvement and two-percent family-macro KL gates failed. Behavioral retention
was not the blocker: selected ordinary/support/core aggregate top-1 was
`91.9441% / 90.6600% / 90.5063%`, all deciding aggregate and family-macro
values stayed above `90%` without a material regression, and geometry passed.

The fixed campaign used exactly `264` model forwards and `494` backward calls:
`48 / 109` for parent recollection, `64 / 385` for the reproduced gradient
bank, 120 tune forwards for the two signed finite candidates, and 32 final
forwards for one frozen selected candidate per prompt. Its classification is
`symmetric_microstep_static_cross_family_transfer_blocker_same_a`. The ignored
write-once report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-gain-microstep-lofo-a-fit16-dev-v5.json
```

Its logical report hash is
`dc87205ee91c0e854155de7f27acf7aacd14a90a2e16b32f9288d843dc911459`
and its file hash is
`488edc027f3265a624762af7f0ad6ec0ca9e7ff81bba469862a5ef0fdc72427b`.

The useful conclusion is stronger than another negative grid point. V4 left
open a coarse-step explanation; V5 shows that a direction can have the right
finite sign on six tune folds and still fail the outer-family boundary at a
`1 / 64` displacement. Repeating smaller static interpolation is therefore
not the next justified rung. The smallest live hypothesis is that the 64 gains
must depend on the current H4 state rather than remain one constant vector for
the whole fold.

### V6: state is useful, but it does not replace the global gain pedal

V6 tested that hypothesis without executing a finite state-gated candidate.
For every outer held-family fold it projected each pre-gate bridge-base H4 row
onto the first four frozen K64 Fisher directions, standardized those four
coordinates using fit-family-equal statistics, and retained the row axis in
the authenticated unit-point teacher-KL pullback. The resulting field was

```text
a_r = 1 + tanh(z_r @ w)
g_rk = 1 + (1 / 64) * a_r * (g_v4,k - 1)
```

so `w = 0` exactly reproduced the executed V5 positive arm. A separately and
identically trained one-scalar amplitude control supplied the critical test:
does row state add more local value than simply pressing the whole gain
direction harder? Each outer fold contained seven nested held-inner-family
fits, and every inner codec was recomputed using only its six training
families.

The analytic screen was healthy in every way except that attribution test:

| V6 analytic check | observed | preregistered gate |
| --- | ---: | ---: |
| feature/design identifiability | all full and inner fits passed; maximum full design condition `3.075`, maximum inner condition `3.461` | rank `4`, condition at most `100`, all fits |
| residual conditional Fisher fraction | `0.8181–0.9774`; `8/8` folds at least `5%` | at least `6/8` folds |
| full state fit non-noop | `8/8` | at least `6/8` |
| negative held-inner state derivative | `50/56`; every fold had at least `5/7` | at least `42/56` and `4/7` in at least `6/8` folds |
| median inner/full raw-slope cosine | `0.9505–0.9980`; `8/8` folds passed | at least `0.90` in `6/8` folds |
| state macro beats scalar macro | `0/8` folds; only `9/56` individual cells | at least `6/8` folds |

Across all 56 held-inner cells, the mean predicted state increment was
`-4.3586e-05`, while the scalar control was more favorable at
`-7.5433e-05`. The state field therefore carried real, stable, mostly
scalar-orthogonal signal, but the no-bias four-parameter field could not
replace the global amplitude shift. V6 is classified
`state_not_better_than_scalar`; it does not authorize finite state-gain
validation.

The run used exactly `112` model forwards and `494` backward calls:
`48 / 109` for parent recollection and `64 / 385` for the shared row-resolved
unit-candidate bank. Analytic fitting added no model work, and finite
state-candidate forwards remained zero. All eight V5 positive carriers were
hash-bound; six were V5-selected positive arms, while two were explicitly
reported counterfactual positive carriers for folds where V5 selected unit.

The ignored report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-state-gain-capacity-lofo-a-fit16-dev-v6.json
```

Its logical report hash is
`001738439b4c2bdd052f6220283103c592b96742b27f105778584e2529a758eb`
and its file hash is
`95e49e10f802121436f081f381c8622653e79767fc2ff4d265b4c976b8193a28`.

This result opens one narrower analytic question, not a finite campaign. The
four state terms were forced to compete with a scalar they could not use. The
smallest fair follow-up is a direct five-parameter `u + z_r @ w` fit with one
joint OPG solve and one joint trust region, compared against the same scalar
control. That asks whether global amplitude and conditional texture cooperate
after their cross-coupling is modeled. It must preserve the V6 thresholds and
still stop before finite execution unless the joint field wins.

### V7: joint global pedal plus local state passes the analytic screen

V7 made that exact five-parameter comparison. It fitted the logit field

```text
ell_r = u + z_r @ w
```

with one family-equal uncentered OPG system over `[s, J]`, including the
scalar/state cross block, and one joint RMS trust region. The scalar control
was the exact V6 one-parameter fit, reconstructed bit-for-bit rather than
refitted under a different objective. Each of the eight outer folds retained
the seven nested inner-family screens, and every runtime direction was
authenticated in both the V4 exact-runtime hash domain and the V6 canonical
codec hash domain.

The joint model passed all six preregistered scientific gates:

- all 64 full and inner fits were identifiable;
- all eight outer folds retained conditional residual energy;
- all eight full joint fits were non-noop;
- 56 / 56 inner derivatives were negative, with all eight local fold gates
  passing;
- the joint inner-family macro beat the exact scalar in 7 / 8 folds; and
- the inner/full state-slope cosine gate passed in 7 / 8 folds.

Aggregated over the 56 inner cells, the exact scalar predicted derivative was
`-7.5433157e-05`; the joint derivative was `-7.8620698e-05`, an additional
`-3.1875413e-06`, or `4.226%` more favorable derivative magnitude. The joint
intercept alone was slightly worse than the scalar. The apparent win came
from the conditional state contribution (`-4.0909409e-06`) compensating for
that intercept difference. This was the intended “global pedal plus local
phrasing” result.

V7 therefore opened one finite validation, but did not itself execute a
candidate or authorize a provider. It used exactly `112` model forwards and
`494` backward calls. Its classification is
`joint_capacity_supported_for_finite_validation`. The ignored report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-joint-state-gain-capacity-lofo-a-fit16-dev-v7.json
```

Its logical report hash is
`816a1b7fe25f02d2b17dcb7e8cd9a57105c94dab1e3644941650b5a111cf789a`
and its file hash is
`8822529ca2526fe73157a4497e199cadb6594e6c5e8597625f0158965e16b0b6`.

### V8: the analytic state advantage reverses at finite execution

V8 froze V5 positive `1 / 64`, the exact V6 scalar, and the V7 joint field,
then executed all three arms on both prompts of every outer-held family. It
performed no tune selection, refit, damping search, fallback, or per-family
routing. Twelve V5 positive observations were replayed exactly; the remaining
four positive arms were explicitly labeled counterfactual carriers. The V4
unit remained a pinned, nonexecuted reference.

All 12 integrity gates and all approximate top-1 safety gates passed. The
finite teacher-KL ordering nevertheless reversed the V7 analytic conclusion:

| frozen arm | family-equal support teacher KL | relation to V7 joint |
|---|---:|---:|
| pinned V4 unit | `0.0493263792` | better by `6.9461e-06` |
| V5 static plus | `0.0493284613` | better by `4.8639e-06` |
| V6 exact scalar | `0.0493322826` | better by `1.0427e-06` |
| V7 joint | `0.0493333253` | candidate |

The joint arm improved only `2 / 8` families against the scalar and `3 / 8`
against static plus. Its relative changes were `-0.00211%` and `-0.00986%`,
respectively, where negative means regression. Both worst-family five-percent
caps passed, so this was a small stable failure rather than an unstable
explosion. The unchanged two-percent macro and `6 / 8` breadth gates failed
against both controls.

NLL agrees with that attribution. The joint endpoint mean NLL was
`1.1700e-05` worse than the scalar, and ordinary mean NLL was `9.7560e-06`
worse. Joint lost to scalar on both NLL measures in all eight families. All
three arms still closed roughly `88%` of the D320-to-native NLL gap; that
recovery is inherited from the K64 carrier, not evidence for the state field.
Top-1 remained safe: ordinary and support matched the unit at `91.8367%` and
`90.5355%`, while graph-core top-1 was `90.3797%`, only `0.1266` percentage
points below unit.

A descriptive post-run attribution check makes the failure especially clear.
Across the eight folds, V7's predicted joint-minus-scalar derivative and V8's
finite joint-minus-scalar KL change had Pearson correlation `-0.512`, Spearman
correlation `-0.643`, and sign agreement in only `1 / 8` folds. V7 measured
the state relation at the realized unit-gain K64 reference, while V8 asked for
the finite scalar-to-joint displacement. V8 therefore classifies the result
as `analytic_to_finite_attribution_failure_same_a`; it does not open fresh
confirmation.

The campaign used exactly `176` model forwards and `494` backward calls. It
accounted for `5,299` candidate support-row executions,
`2,170,470,400` analysis-only D320 projection MACs, and `434,094,080`
analysis-only K64 projection MACs. These are diagnostic operations, not a
serving-kernel or latency measurement. The ignored report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-joint-state-gain-held-finite-lofo-a-fit16-dev-v8.json
```

Its logical report hash is
`622fa8ce212ea625837754a69e8e24f609b554ad35a8567bd8653f19dd44f305`
and its file hash is
`9a1a5e925b86592831bbc26164fc4df6a6dd778d2f4b7563733277c0dba231ac`.

The next rung is not a post-hoc sign flip on the inspected held rows. It must
trace the actual cast-once scalar-to-joint H4 path, evaluate the derivative at
the scalar endpoint, and determine whether the reversal comes from the V7
reference location or curvature along that finite displacement. Only then can
a scalar-referenced refit or a fit-only symmetric finite secant be justified.

### V9: the scalar endpoint restores the finite ordering

V9 performed that attribution without fitting or selecting another candidate.
It first authenticated the exact V8 file and logical hashes, reproduced the V7
analytic lineage, and replayed all 16 scalar and joint endpoint token-KL hashes.
For each held prompt it then collected fresh tokenwise H4 VJPs at the held-unit
reference, at the realized V6 scalar endpoint, and at four fixed Gauss-Legendre
nodes on the actual cast-once scalar-to-joint H4 displacement. The joint
endpoint was independently executed to supply the finite
`KL_joint - KL_scalar` target.

The additive family-equal ledger is:

| stage | family-equal joint-minus-scalar teacher-KL prediction |
|---|---:|
| V7 nested inner-CV analytic convention | `-3.1875413e-06` |
| held prompt, full-seven fit, V7 convention | `+2.0269265e-06` |
| held-unit gradient with the actual realized displacement | `+1.0053765e-06` |
| realized scalar-endpoint tangent | `+1.0597609e-06` |
| four-node GL4 path integral | `+1.0640484e-06` |
| finite joint-minus-scalar KL | `+1.0426915e-06` |

This localizes the reversal at family-macro scale. The large changes happen
while transferring from the nested inner-CV/unit convention to the held prompt
and its actual finite displacement. Moving the gradient reference from the
off-path held-unit point to the scalar endpoint adds only `5.4384e-08` to the
macro; integrating the gradient from the scalar endpoint over the entire
scalar-to-joint path adds just `4.2875e-09` to that same macro. This makes
large aggregate curvature unlikely, but V9 did not publish tokenwise
`P - T_scalar` RMS, so cancellation can still hide local transport.

At the eight-family level, both the scalar tangent and the GL4 integral agree
with the finite direction in `8 / 8` families. For GL4 versus the finite
change, Pearson correlation is `0.99616`, Spearman is `1.0`, and cosine is
`0.99607`. The tokenwise sign agreement rate is `91.23%`, and GL4 predicts the
same two improving families observed by finite execution.

The strict closure result is deliberately reported as unresolved. Overall
relative RMSE is `8.90%`, above the preregistered `5%` maximum. Four of eight
families clear the `10%` family threshold; family relative RMSE ranges from
`5.36%` to `19.94%`. The overall cosine gate passes, all 11 integrity gates
pass, and the additive ledger closes exactly, but those facts do not override
the failed magnitude gates. The casted path is a dtype staircase, so the report
also makes no exact continuous-FTC claim.

The ignored write-once report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-joint-state-gain-scalar-joint-path-gl4-attribution-lofo-a-fit16-dev-v9.json
```

Its logical report hash is
`00d51133dd579a26198f763d6c21b0c89d47e834c6bdac64e91d583e2858055e`
and its file hash is
`24343d0d4bc633d59a436ebeed95787ea7fdc83c42540b35d63a21b5d1063ed6`.
The campaign used exactly `240` model forwards, `1,148` backward chunks, and
`8,575` candidate support-row executions. It is classified
`scalar_joint_path_closure_unresolved_same_a`.

One numerical ambiguity must be closed before that refit. Both the V9 VJP
objective and the independent endpoint helper perform `log_softmax`,
probability multiplication, and the 262,144-way vocabulary reduction in
float32 whenever the logits are float32. The finite target is then the
difference between teacher-KL values near `5e-2`, while its token changes are
only around `1e-5`. The near-zero-mean `1.389e-6` closure RMSE is therefore
plausibly an objective-arithmetic floor rather than model curvature.

The immediate no-fit rung is an exact objective-precision replay: reproduce
the same endpoint and four-node H4/logit hashes, change only the selected-token
teacher-KL arithmetic to float64, and publish tokenwise `P64 - T64` as well as
`D64 - P64` and `D64 - D32`. It retains the frozen `5% / 10% / 0.99` closure
gates. If that replay closes, the scalar-endpoint nested-LOFO refit is earned;
if it does not, its token transport term distinguishes live-dtype rounding
from genuine local curvature. Repeating denser quadrature or fitting before
this precision control is not justified. V9 still reuses inspected
Calibration A, held native tails, and native teacher logits, so it opens
neither fresh confirmation nor serving, mutation, deployment, speed, or
compression claims.

### V10: objective precision is real, but it is not the missing closure

V10 executed that exact no-fit control. It authenticated the V9 file and
logical hashes, replayed all 16 scalar/joint endpoints and all 64 fixed GL4
nodes, and required their H4, provider, execution, legacy float32 KL, and
finite-delta hashes to remain unchanged. The only numerical policy change was
to promote the selected teacher and candidate logits to float64 before every
`log_softmax`, probability product, and vocabulary reduction. The primary
finite target was computed directly as
`sum(p_teacher64 * (logp_scalar64 - logp_joint64))`, rather than by subtracting
two nearly equal scalar KL totals.

Every one of the 13 integrity gates passed, including independent bitwise
recomputation of each float64 objective and a cancellation-resistant direct
endpoint cross-check. The strict scientific closure still did not pass:

| V10 objective-precision check | observed |
|---|---:|
| finite D64 RMS | `1.5575905e-05` |
| GL4 P64 versus finite D64 relative RMSE | `8.6734%` |
| GL4 P64 versus finite D64 cosine | `0.996262` |
| families at or below `10%` relative RMSE | `4 / 8` |
| path transport P64−T64 relative RMSE | `0.06221%` |
| path transport P64−T64 cosine | `0.999999805` |
| endpoint precision D64−D32 relative RMSE | `2.00395%` |

The path-transport RMS is `9.6904e-09`, versus `1.3510e-06` total closure
RMS. By the reverse triangle inequality, the scalar-tangent-to-finite term is
therefore at least `1.3413e-06`: ordinary curvature along this four-node path
cannot be the material missing term, and denser quadrature is not earned.
Endpoint arithmetic matters—the D64−D32 RMS is `3.1213e-07`, more than 32
times the transport RMS—but it is still 4.33 times smaller than the total
closure miss. V10 consequently treats objective precision as the dominant
measured secondary signal, not as a unique or sufficient cause.

The ignored write-once report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-joint-state-objective-precision-gl4-lofo-a-fit16-dev-v10.json
```

Its logical report hash is
`36468470728bb941933ada75b74497108d0c163dc23688b91375462ac96fc77c`
and its file hash is
`48afceefdf9c468a91f07b2004ed9052586e99d41dba5f4c7223b6e4ae638e79`.
The campaign used exactly `224` model forwards, `1,039` backward chunks, and
`7,756` candidate support-row executions. It is classified
`small_path_transport_live_dtype_or_finite_rounding_supported_same_a`.

This result does not authorize the scalar-endpoint refit yet. The smallest
falsifiable no-fit follow-up is a post-H4 suffix/discrete-path audit: hold the
realized float32 scalar and joint H4 endpoints fixed, reproduce the live
float32 suffix exactly, and compare it with a precision-promoted suffix or an
equivalently locked finite secant audit. That separates the differentiable
straight-through tangent from the actual cast/forward staircase without
searching a denser quadrature rule or using the held outcome to tune a fit.
Only after that boundary is localized should a live-finite secant refit be
opened.

### V11-V13 post-H4 adjoint localization

V11 first removed the remaining path-location ambiguity by replaying the
exact post-H4 suffix at all 64 V10 GL4 nodes. The input was the float64 path
point, cast exactly once to the live float32 H4 dtype; layers 5 through 17,
the final projection, and float64 selected-token teacher-KL objective were
then identical to the V10 computation. Every primal H4, logit, and token-KL
vector replayed bitwise.

The resulting forward-mode suffix JVP and the published reverse-mode Fisher
contraction were extremely close, but missed the frozen adjoint gate:

| V11 check | observed |
|---|---:|
| JVP versus VJP relative RMSE | `0.0002564459` |
| JVP versus VJP absolute RMSE | `3.9481e-9` |
| JVP finite-D64 closure relative RMSE | `8.67545%` |
| VJP finite-D64 closure relative RMSE | `8.67339%` |

V11 therefore classified the result as `suffix_adjoint_ambiguity_same_a`.
The same-function adjoint difference was roughly 338 times smaller than the
finite-displacement closure error, but it was still larger than the
preregistered `1e-4` threshold and had to be localized before any correction
fit.

V12 tested whether that gap came from the way the already-collected V10
float32 gradient bank was promoted and contracted. It replayed five fixed
orders without changing the gradient source:

| V12 stage | JVP relative RMSE | JVP absolute RMSE |
|---|---:|---:|
| `P_v10` | `0.000256445859` | `3.9480774e-9` |
| `P64_node` | `0.000256445859` | `3.9480774e-9` |
| `P_dir` | `0.000256445846` | `3.9480772e-9` |
| `P_prod` | `0.000256444067` | `3.9480498e-9` |
| `P_live` | `0.000256442898` | `3.9480318e-9` |

The largest measured boundary movement, `P_live - P_prod`, had RMS
`1.0112e-12`: only about `0.026%` of the residual `3.9480e-9` mismatch.
Every family still missed `1e-4`, and the finite-correction branch remained
closed. V12's historical classification string is
`unresolved_forward_reverse_ad_kernel_mismatch`; the report does not claim
that a particular internal kernel was causally identified.

V13 then replaced inference about the old bank with a fresh reverse-mode
measurement of the exact V11 function. For each node it called one
`torch.func.vjp(..., has_aux=True)`, applied canonical one-hot output
cotangents in chunks of eight through `vmap`, hashed each transient full and
support H4 cotangent, proved the displacement direction was exactly zero
outside support, and contracted only the support rows in float64. The first
native result was retained as node 1; no warm-up result was discarded.

All 12 integrity gates passed. The complete authenticated resource ledger is:

| V13 resource | count |
|---|---:|
| native suffix VJP primals | `64` |
| suffix segment calls | `832` |
| logit projections / H4 casts | `64 / 64` |
| canonical token cotangent rows | `3,212` |
| vmap pullback chunks | `436` |
| dense output-cotangent coordinates | `174,292` |
| streamed full H4 gradient coordinates | `130,048,000` |
| support contraction products | `113,602,560` |
| outside-support gradient coordinates observed | `16,445,440` |

These are protocol counts, not FLOPs, latency, or full-model backward counts.
No raw gradient, H4, logit, JVP, VJP, token ID, or prompt tensor is serialized.

The scientific result is decisive:

| V13 comparison | symmetric relative RMSE | absolute RMSE |
|---|---:|---:|
| native VJP versus V11 JVP, nodewise | `0.000256354214` | `3.9466648e-9` |
| native VJP versus V11 JVP, GL4 integrated | `0.000256445859` | `3.9480774e-9` |
| native VJP versus V10 `P_v10`, integrated | `4.6799e-16` | `7.2049e-21` |
| native VJP versus V12 `P64_node`, integrated | `7.9281e-17` | `1.2206e-21` |

All four GL4-node relative errors remain near `2.56e-4`. The eight integrated
family errors span `1.08886e-4` to `3.83460e-4`, so the overall and
every-family `1e-4` gates both fail. The fixed
`P_v10 → P64_node → P_dir → P_prod → P_live → J64_suffix → N64_native_vjp`
telescope closes with maximum residual `2.12e-22`.

This rules out the old V10 gradient extraction or execution path as the
material source: a fresh native reverse pass reproduces it essentially
exactly. The residual is now localized to the finite-precision
forward-versus-reverse treatment of the same cast-once float32 suffix (or an
equivalent nondifferentiable-boundary convention), not to Fisher-bank
provenance or contraction order. V13 is therefore honestly `passed: false`
and classified
`persistent_same_suffix_forward_reverse_ad_or_nondifferentiable_boundary_ambiguity`.

The ignored write-once V13 report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-token-fisher-k64-candidate-joint-state-suffix-native-vjp-gl4-lofo-a-fit16-dev-v13.json
```

Its logical report hash is
`dc88b406aa1f10d417c3cd5a1f96de8e276b8e7e9e1673515d76504bde8888ac`,
comparison hash is
`9d7f9d6db8fd1504c9b6421a3fec62cde7a0e8352ec46e5edd998b748fd1390e`,
runtime receipt-set hash is
`11a76be7232bb48c565c5a920860d46b01b92cc0d37cc2dfe85d0265cabd4206`,
resource receipt-set hash is
`e69228854626be9cf7937f348e6ac155261e5e99c86dcb41fdc60f5cceb36a1e`,
and file hash is
`991a3f3ff30684b6a96191c008c75d9ff0da3737e3606b5dae6a5abb65b26251`.

The operational next rung should not keep changing contraction arithmetic:
V13 has exhausted that explanation. For Fisher compilation, reverse-mode VJP
is the native gradient quantity and can be declared the local derivative in a
new, prospectively frozen protocol. Its forward/reverse discrepancy must be
treated as a measured live-float32 numerical envelope and confirmed on a
fresh family panel rather than post-hoc relabeling V13 as a pass. Separately,
the much larger `8.67%` finite-displacement miss still requires held-out
finite correction or a conditional residual edge. An optional promoted-suffix
control can identify the exact floating-point primitive, but it is no longer
a blocker to gradient-source provenance and cannot itself establish finite
executor fidelity. V14 below executes that held-out finite-residual-provider
rung.

### V3–V13 scientific boundary

These are negative diagnostics, not failed deployment attempts. The outer
fit/tune/held split is real, and each K64 tail basis/order excludes its held
family, but the broader experiment is still reused Calibration-A hypothesis
work. The frozen D320 carrier contains information from all A16 families, the
parent outcomes were already inspected, and every finite correction uses the
current prompt's held native H4 tail and native teacher logits. That oracle
truth leakage means this is not an end-to-end family-disjoint executor and
cannot run at inference without the source model.

V6 added the identically trained scalar-gain control, but no fixed-seed
gain-permutation or full-versus-diagonal preconditioner control has been run.
The result therefore still does not establish mode-specific value or isolate
the full OPG preconditioner as the cause. It also produced no selected global
gain vector, finite state provider, serving parameter count, serving MAC
count, model mutation, fresh confirmation, compression, deployment, speed, or
latency result. What it establishes is narrower: under the tested same-A
protocol, neither the V3 residual direction, the V4 mean-KL OPG direction, nor
V5's fixed `1 / 64` microstep provided a safe static K64 gain repair that
transferred across the outer held-family boundary; V6 additionally showed
that a stable state-only field did not beat the scalar amplitude control.
Other preconditioners, joint scalar/state or multi-step fits, positive steps
smaller than `1 / 64`, and strict interior steps between `1 / 64` and `1 / 8`
remain untested.

### V14 autonomous complete-H4 full-suffix screen

V14 converted the V13 provenance result into an operational inference
candidate class. Native H4 and the reverse VJP of supervised NLL were admitted
only while building private fit traces. At runtime, every correction provider
read only the authenticated one-pass prefix, its causal source modes and masks,
and the realized pre-correction H4 state. Native H4, teacher/source logits,
targets, and gradients were not provider inputs.

The opened A16 panel contains 16 prompts in eight families. V14 used eight
outer leave-one-family-out folds. For each held family, both the residual PCA
decoder and the causal ridge provider were refit on the other seven families;
the held family was excluded from both. Four recipes were frozen:

| recipe | rank / lags / objective | incremental provider scalars | float64 bytes | logical MACs/token |
|---|---|---:|---:|---:|
| K64 hidden | `64 / 8 / hidden residual` | `77,888` | `623,104` | `118,784` |
| K256 hidden | `256 / 8 / hidden residual` | `360,704` | `2,885,632` | `524,288` |
| K256 Fisher | `256 / 8 / reverse-VJP weighted` | `360,704` | `2,885,632` | `524,288` |
| K320 Fisher | `320 / 8 / reverse-VJP weighted` | `471,360` | `3,770,880` | `675,840` |

These are incremental provider-only counts. They exclude the retained Gemma
parameters, graph bridge, downstream suffix, final normalization, and LM head,
so they are not end-to-end parameter or FLOP reductions.

Every held prompt was then run through the real factorized Gemma carrier. The
candidate H4 continued through untouched layers 5–17, final normalization, and
the full 262,144-way LM head. Source outputs remained authoritative and
candidate logits were used only for NLL, source-to-candidate KL, and top-1
shadow metrics.

| arm | ordinary ΔNLL / KL / top-1 | complete-H4 support ΔNLL / KL / top-1 | graph core ΔNLL / KL / top-1 |
|---|---|---|---|
| base graph | `+2.56889 / 2.88209 / 42.43%` | `+2.97838 / 3.34150 / 33.25%` | `+3.01021 / 3.38089 / 32.66%` |
| K64 hidden | `+2.34791 / 2.61163 / 45.11%` | `+2.72217 / 3.02793 / 36.36%` | `+2.74115 / 3.05643 / 36.20%` |
| K256 hidden | `+1.29240 / 1.37474 / 57.47%` | `+1.49842 / 1.59387 / 50.68%` | `+1.50062 / 1.60128 / 50.76%` |
| K256 Fisher | `+1.16923 / 1.25289 / 59.72%` | `+1.35561 / 1.45261 / 53.30%` | `+1.35617 / 1.45827 / 53.42%` |
| K320 Fisher | `+1.01993 / 1.10038 / 63.69%` | `+1.18251 / 1.27579 / 57.91%` | `+1.18667 / 1.28208 / 58.10%` |

The capacity trend is strong and monotonic. Relative to the base graph, K320
reduces ordinary ΔNLL by `60.30%`, KL by `61.82%`, and raises top-1 by `21.27`
points. At fixed K256, reverse-VJP weighting reduces ΔNLL by another `9.53%`,
KL by `8.86%`, and adds `2.26` top-1 points. K256-to-K320 Fisher scaling adds
another `3.97` ordinary top-1 points. The sundial family is nevertheless the
worst K320 ordinary family on all five frozen metrics, so the miss is not just
an aggregate-weighting artifact.

No arm is close to the established qualification gates:

| frozen gate | limit | best K320 observation |
|---|---:|---:|
| absolute aggregate ΔNLL/token | `≤ 0.05` | `1.01993` |
| source→candidate KL/token | `≤ 0.05` | `1.10038` |
| aggregate top-1 agreement | `≥ 0.95` | `0.63695` |
| prompt-p90 absolute ΔNLL/token | `≤ 0.10` | `1.60130` |
| prompt-p10 top-1 agreement | `≥ 0.90` | `0.52500` |

The run completed exactly 128 full-model forwards, 16 reverse-VJP backward
traversals, 32 outer-fold provider fits, and 64 causal off-support execution
checks. Ordinary, complete-support, and graph-core ledgers each covered all
eight families; the sparse 13-token causal tail covered four families and was
reported as non-gating. All integrity checks passed.

The result is therefore `passed: false`, classified
`autonomous_complete_h4_oof_recipes_insufficient`. The conditional publisher
fit and writes an authenticated provider sidecar only after an OOF pass; no
recipe passed, so no full-panel provider or sidecar exists. The fresh guard and
Calibration B were not opened.

The ignored write-once report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-complete-h4-autonomous-residual-outer-lofo-a-fit16-dev-v14.json
```

Its logical report hash is
`01803d62e106de05acafcd000308ae2f861f2be9c6bb879fd2d7f4c9e611f906`
and its file hash is
`fc78f37790898dd3acbded89f6da8a2fa9ee466217d1eba3f123e570859384ad`.
No prompt text, token IDs, logits, H4 activations, gradients, or model weights
are serialized.

This is a real full-vocabulary/full-suffix behavioral test, but it replaces
one complete-H4 correction boundary. Layer 4 still executes, source-model
parameters remain resident, and the other 17 layers have not been compiled.
It is therefore not whole-model compilation, layer deletion, compression,
serving, latency, or speed evidence.

The smallest bounded next discriminator is one K640, lag-8,
reverse-VJP-weighted outer-LOFO capacity sentinel with the same ridge and
metrics. Its provider geometry would use `1,147,520` incremental scalars and
at most `1,556,480` logical MACs/token. If full H4 output span remains far from
the gates, PCA rank truncation is no longer the main bottleneck and the next
work should enlarge or nonlinearize the conditional residual map rather than
continue the rank ladder. If K640 closes the gap, compression can restart by
distilling that ceiling downward.

### V15 full-span autonomous capacity sentinel

V15 ran that one preregistered capacity control without searching another
rank, lag, ridge, objective, or recipe. It retained the V14 A16 panel, exact
eight outer leave-one-family-out folds, lag count `8`, ridge `1e-4`, and
reverse-VJP row weighting. The only intended change was K320 to the full H4
rank K640. Each fold therefore fit one provider with `1,147,520` float64
coefficients and an upper bound of `1,556,480` logical matrix MACs/token.
Those provider-only counts exclude retained Gemma, the graph bridge, full
suffix, final normalization, and LM head.

The full output span improves every loss family, but does not approach the
absolute gates:

| ordinary outer-LOFO metric | V14 K320 | V15 K640 | change |
|---|---:|---:|---:|
| absolute delta NLL/token | `1.019928565` | `0.818250897` | `-19.7737%` |
| source-to-candidate KL/token | `1.100382289` | `0.925721323` | `-15.8728%` |
| top-1 agreement | `63.69495%` | `66.70247%` | `+3.00752` points |
| prompt-p90 absolute delta NLL | `1.60130` | `1.43010` | lower |
| prompt-p10 top-1 | `52.50%` | `55.00%` | `+2.50` points |

Delta NLL and KL improve in all eight families; top-1 improves in seven. The
family delta-NLL reductions range from `4.435%` for obsidian to `47.126%` for
alpine, with a `21.917%` median. The broader support ledgers agree: complete
H4 moves to `0.948682 / 1.073283 / 61.3948%`, and graph core to
`0.951218 / 1.080389 / 61.5190%` for delta NLL, KL, and top-1.

That gain costs `2.4345x` the K320 coefficients and `2.3030x` its logical
MACs/token. More importantly, ordinary K640 still exceeds each `0.05` loss
limit by more than 16 times and remains `28.30` top-1 points below `95%`.
Full-span PCA capacity is therefore useful but insufficient. The result does
not uniquely prove that nonlinearity is the missing ingredient rather than
different features, lags, regularization, or more independent fit data; it
does justify testing one bounded conditional map before another rank ladder.

The run completed exactly 80 full-model forwards, 16 reverse-VJP traversals,
eight fold providers, and 16 causal off-support checks. All 819 complete-H4
rows were covered, every fold trained on at least 703 support rows, and the
source-free runtime and no-leakage checks passed. No arm passed, so no
full-panel provider or sidecar was written and the guard and Calibration B
remained closed. Its classification is
`autonomous_complete_h4_k640_oof_capacity_ceiling_insufficient`.

The ignored write-once report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-complete-h4-autonomous-residual-k640-capacity-outer-lofo-a-fit16-dev-v15.json
```

Its logical report hash is
`8518dab697e78ecb210a1fb99e173f486f8939893b081ba0d28d970c82e86ff4`
and its file hash is
`4c6033385f278437849da770c18058720c6e3dcb075e3d0844c631480994de18`.
The historical write-once artifact predates the later explicit prerequisite
receipt. The current V15 runner now checks the exact V14 path, logical hash,
file hash, failed classification, null candidate, and closed guard/B before
model load; that protocol hardening did not rewrite the historical result.

### V16 Fisher-conditioned two-coordinate square

V16 then ported the bounded conditional-map idea into the source-free
complete-H4 ABI. Its fixed parent is the V14 K256/lag-8/reverse-VJP provider.
Two exactly parameter-matched rank-16 children differ only in how their two
fit-only coordinate targets are defined:

1. the leading two empirical reverse-VJP Fisher axes in parent modal space;
2. the leading two centered activation-PCA axes as a non-Fisher control.

At serving time neither child reads gradients, native H4, targets, logits,
coordinate axes, or family IDs. The fixed parent predicts a modal correction
`p`; a learned affine router produces `u = pW + b`; and each coordinate is
bounded into the open square with `c = u / (s + |u|)`. The child then applies

```text
p* = p + [c1 p, c2 p, c1 c2 p] L R
```

before the parent decoder performs one float64-to-live-dtype boundary cast.
The three conditional blocks define a multi-affine operator over the square.
Projecting all four corners to spectral norm at most `0.25` therefore bounds
the pointwise conditional correction amplitude throughout the interior. It
does not certify the full nonlinear Jacobian or a Lipschitz constant because
the coordinates themselves depend on `p`.

Each child adds exactly `16,900` float64 coefficients and `16,896` logical
matrix MACs/token to the K256 parent, for totals of `377,604` and `541,184`.
The count is coefficient/MAC accounting only: retained Gemma, device-transfer
policy, integrity hashing, workspaces, bridge, and suffix are excluded.

After the preliminary fit-geometry-only execution exposed a control gap, an
additional fail-closed qualification was frozen before the corrected
write-once rerun. This remains the same opened development panel, not fresh
confirmation. The qualification separates fit predictability from
serving-time generalization.
On every outer fold and for both objectives, each fit-only bounded target had
to achieve R-squared at least `0.01`. The fitted provider was then replayed on
the two unseen sequences from the held family using only source modes,
logical positions, masks, and base H4. With equal mass per sequence and equal
mass per supported row within each sequence, those actual runtime coordinates
had to satisfy all of:

- covariance eigenvalue ratio lambda2/lambda1 at least `0.01`;
- absolute coordinate correlation at most `0.99`; and
- at least `0.01` residual second-coordinate energy after projecting it on
  the first coordinate.

Both arms pass both qualifications on every fold. Fisher's minimum fit-target
R-squared is `0.890091`. On held families its worst eigenvalue ratio is
`0.375454`, maximum absolute correlation is `0.428233`, and minimum residual
second-coordinate energy is `0.816617`. PCA's held-family values are
`0.481206`, `0.317924`, and `0.898925`, respectively. Every coordinate replay
is authenticated by its provider, sequence, coordinate, row-weight, and
geometry hashes. The negative fidelity result therefore cannot be attributed
to the router collapsing back to a line or point on fit or unseen families.

The full-vocabulary outer-LOFO result is nevertheless an effective tie:

| ordinary metric | K256 parent | Fisher square | PCA square |
|---|---:|---:|---:|
| delta NLL/token | `1.169234512` | `1.169296175` | `1.169187970` |
| source-to-candidate KL/token | `1.252893922` | `1.252883646` | `1.252845862` |
| top-1 agreement | `59.72073%` | `59.72073%` | `59.72073%` |
| family-macro absolute delta NLL | `1.156531725` | `1.156592874` | `1.156481787` |
| family-macro KL | `1.239356692` | `1.239347425` | `1.239304913` |

Fisher improves family loss in only one of eight folds, has no top-1 gain,
does not beat the PCA control, and microscopically regresses complete-H4 and
graph-core delta NLL. Every absolute and material mechanism gate fails except
the worst-family regression floor and top-1 non-regression against PCA.

The fit diagnostics explain why this is not yet a strong test of nonlinear
capacity. The unconstrained rank-16 Fisher operators require four-corner
projection scales of only `0.0003956–0.0006275`; PCA scales are similarly
`0.0004569–0.0006679`. After that global radial projection, Fisher improves
its own weighted fit residual RMSE by only `0.0186–0.0290%`, and PCA by
`0.0293–0.0450%`. The router is genuinely two-dimensional, but the inherited
pointwise `0.25` trust region permits almost none of the fitted residual.

The run completed exactly 112 full-model forwards, 16 reverse-VJP
traversals, 24 outer-fold provider fits, 16 held-family coordinate
diagnostics, eight parameter-match checks, and 48 causal off-support
executions. All 24 fold-ownership receipts, ledgers, hashes, and
immutable-input checks passed. No full-panel provider was fit, no sidecar was
written, and guard/B stayed closed. The report is honestly `passed: false`
and classified
`fisher_square_absolute_fidelity_insufficient`.

The ignored write-once report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-complete-h4-fisher-square-r16-k256-outer-lofo-held-runtime-geometry-a-fit16-dev-v16.json
```

Its logical report hash is
`fa6d89d49cd2b041c361a50efeb8b606d6fa0b72be74c83235950a9cdd7ef2ff`
and its file hash is
`14a2eb93cda810cd68ff859c4731f0991daa65084b3917eb8442537e0b54ad31`.
The earlier report without `held-runtime-geometry` in its filename is
preserved as a preliminary fit-geometry-only artifact and is not the
authoritative V16 result.

This rejects the tested globally projected conditional operator, not the
broader idea of nonlinear or conditional compilation. The next controlled
development rung should preserve the same Fisher/PCA coordinates and
outer-LOFO comparison while changing only the residual trust treatment. Two
bounded candidates are justified: solve the regression under the pointwise
operator constraint instead of fitting unconstrained then shrinking it
radially, or calibrate an additive decoded-H4 residual budget from fit-only
parent/residual scale and validate that budget on held families. A post-hoc
relaxation of `0.25` on the same outcome would not be confirmation; any bound
ladder belongs to a newly frozen development protocol.

### V17/V18 pointwise-bounded Fisher pedal

The next rung kept the V16 parent, coordinates, rank, opened A16 panel, and
outer leave-one-family-out ownership fixed while changing only how the
conditional direction is bounded and applied. For parent modal correction
`p`, V16 coordinates `c1,c2`, and rank-16 raw direction

```text
q = [c1 p, c2 p, c1 c2 p] L R
```

the runtime now computes

```text
b = q min(1, 0.25 ||p|| / ||q||)
a = clamp(bias + [c1,c2,c1c2] weight, 0, 1)
p* = p + a b
```

with explicit zero handling. The clip never amplifies `q`, and every row has
the exact modal-amplitude certificate `||a b|| <= 0.25 ||p||`. This is not a
decoded-H4 Jacobian or Lipschitz certificate.

Five fixed arms separate direction quality from conditional timing:

1. the unchanged K256 reverse-VJP parent;
2. Fisher direction with unit pedal;
3. the same Fisher direction with its fit-optimal constant pedal;
4. the same Fisher direction with a learned conditional pedal; and
5. an exactly parameter-matched activation-PCA conditional pedal.

The three Fisher children share authenticated bitwise-identical router,
scale, and direction tensors on every fold. All four children have identical
serving geometry: `377,608` prepared float scalars and `541,187` logical
matrix MACs/token including the parent, an increment of `16,904 / 16,899`.
Elementwise norm, clipping, and scaling operations, retained Gemma, the
bridge, and the downstream suffix are excluded from that matrix-MAC count.

The fit first clips the modal residual target into the same rowwise trust
ball. Its analytic scalar target is `<residual,b>/||b||²`; supported rows are
weighted by the existing fit weight times `||b||²`. The conditional pedal
fits `[c1,c2,c1c2]` with an unpenalized intercept and ridge-penalized centered
slopes. The constant control is the clipped weighted analytic mean, and the
unit control is exactly one. Qualification measures pedal variation only on
direction-energy-supported rows, both in fit and on held families, so
variation on negligible directions cannot qualify as conditional compute.

The first write-once V17 execution exposed a report-aggregation defect. Unit
and constant replay tensors were exactly one, but equal-sequence floating
weights produced summaries as low as `0.9999999999999996`; exact scalar-mode
gates then reported false held failures. That report is preserved rather than
rewritten:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-complete-h4-fisher-pedal-r16-k256-outer-lofo-a-fit16-dev-v17.json
```

Its logical hash is
`f59ed94966ebda89296ac1ef26ead49f464b410ac9fb8221f9e44cdfa105d1b7`
and file hash is
`1439ffe8439e47d37d8e610d0094912cfa86e006558a222cfd10b087c037c401`.
Its classification is not a scientific result. V18 preregisters and
authenticates that receipt, derives unit/constant summaries from the exact
serving scalar, and divides conditional ordinary and effective moments by
their realized float64 mass. V18 reproduces every V17 fidelity tensor exactly
while removing only the false scalar-control failures.

The authoritative V18 ordinary result is:

| arm | delta NLL/token | family-macro absolute delta NLL | KL/token | top-1 |
|---|---:|---:|---:|---:|
| K256 parent | `1.169234512` | `1.156531725` | `1.252893922` | `59.72073%` |
| Fisher unit pedal | `1.252916829` | `1.238813275` | `1.302652140` | `58.96885%` |
| Fisher constant pedal | `1.252916829` | `1.238813275` | `1.302652140` | `58.96885%` |
| Fisher conditional pedal | `1.241269976` | `1.227125494` | `1.290571496` | `59.07626%` |
| PCA conditional pedal | `1.198116944` | `1.184406662` | `1.312808147` | `59.93555%` |

The conditional computation is genuine. Every Fisher fit and held fold passes
the effective-variation gates. On held families its direction-energy-weighted
pedal mean spans `0.793695–0.999101`, standard deviation
`0.004252–0.282596`, and minimum `0.257215–0.971880`. It improves
family-macro absolute delta NLL by `0.943466%` against the identical
unit/constant direction baseline and wins `8/8` families; KL improves
`0.947532%` and top-1 rises `0.107411` points. The loss gain narrowly misses
the preregistered `1%` mechanism threshold.

The larger comparison remains negative. Against the parent, Fisher
conditional worsens macro absolute delta NLL by `6.103920%`, macro KL by
`3.029933%`, and aggregate top-1 by `0.644468` points. It wins only `2/8`
families, and its worst family regresses `17.373094%`. Complete-H4-support and
graph-core loss, KL, and top-1 all regress. Fisher also loses to matched PCA
on absolute delta NLL and top-1, although its KL is lower.

The fit explains the near-saturated behavior. Fisher's unclipped analytic
pedal mean is `1.17305–1.23942`, and `66.70–78.75%` of fit weight is clipped
from outside `[0,1]` into that interval. The constant optimum nevertheless
equals the unit pedal on every fold. The ridge conditional emits some values
below one, but is microscopically worse than the constant fit RMSE on all
eight Fisher folds; PCA has the same failure on seven of eight. These are
small algorithmic near-ties, not the V17 summation defect. The amplitude bound
itself passes everywhere, but the combined pointwise/fit qualification
correctly fails.

V18 completed exactly 144 full-model forwards, 16 reverse-VJP traversals, 40
outer fits, 32 held runtime diagnostics, 40 ownership receipts, and 80 causal
off-support checks. Coordinate geometry passes, but pointwise qualification,
mechanism support, every absolute fidelity scope, and candidate readiness
fail. No full-panel fit ran, no provider sidecar was written, and guard,
Calibration B, serving, compression, and speed claims remain closed. The
classification is `fisher_pedal_pointwise_trust_insufficient`.

The authoritative report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-complete-h4-fisher-pedal-r16-k256-outer-lofo-a-fit16-dev-v18.json
```

Its logical report hash is
`93a757c2efb5388000536a259eb5e654ece9691a3ba016d9404fec152a65c641`
and its file hash is
`95394ee0a643e13d959eff5a5d1180643ef59cc4c5d1d56652e5480229f27c81`.

This rejects the tested affine pedal, not conditional computation itself.
The most informative next change is not a looser post-hoc trust bound. The
analytic target is predominantly above one, while the deployed pedal cannot
amplify the direction. A newly frozen rung should change the direction fit or
jointly optimize direction and bounded pedal against the finite downstream
objective so that the conditional degree of freedom controls useful signed
amplitude rather than mostly suppressing an underpowered or misoriented
direction.

### V19 finite teacher-KL joint direction and sigmoid pedal

V19 executed that next hypothesis without changing the opened A16 panel,
outer leave-one-family-out ownership, K256 parent, Fisher/PCA coordinates,
rank-16 child budget, or pointwise `0.25` trust certificate. It replaced the
analytic direction/pedal fit with finite downstream authority. For each outer
fold and each Fisher/PCA initializer, it:

1. reconstructed the exact pinned V18 parent and conditional start artifacts;
2. initialized a balanced rank-16 factorization of twice the V18 direction
   product;
3. initialized sigmoid pedal slopes and intercept to zero, giving pedal
   `0.5` and the V18 direction amplitude before pointwise clipping;
4. jointly updated both direction factors and all four pedal parameters with
   exactly four full-batch float64 Adam steps;
5. scored checkpoints `0–4` by exact float64, 262,144-way
   `KL(source || candidate)` through the complete live suffix; and
6. froze the earliest exact minimum before exposing either held-family prompt
   to evaluation.

The optimizer used learning rates `1e-3` for direction factors and `2.5e-2`
for pedal parameters, Adam betas `0.9 / 0.999`, epsilon `1e-8`, no weight
decay, and no gradient clipping. The local modal-head contraction used a
straight-through derivative at the one cast boundary, while every checkpoint
score came from actual finite execution. A fold capability exposed teacher
rows for exactly 14 training prompts across seven families. Held rows were
cached during the authenticated collection pass but could not be named or
consumed by that fold's optimizer.

Six resource-matched outer arms separated initialization, direction, and
pedal effects:

1. the exact V18 K256 parent;
2. the exact V18 Fisher conditional start;
3. the finite-joint Fisher direction with unit pedal;
4. the same direction with learned intercept but zero slopes;
5. the selected finite-joint Fisher conditional; and
6. the selected finite-joint activation-PCA conditional control.

All children retained `377,608` prepared float scalars and `541,187` logical
provider matrix MACs/token, increments of `16,904 / 16,899` over the parent.
Those counts exclude retained Gemma, the bridge and suffix, and elementwise
sigmoid, norm, and clamp work.

The finite optimizer did not descend. Mean exact training-KL checkpoint
curves were:

| initializer | checkpoint 0 | checkpoint 1 | checkpoint 2 | checkpoint 3 | checkpoint 4 |
|---|---:|---:|---:|---:|---:|
| Fisher | `0.002443840` | `0.007811080` | `0.003442031` | `0.004511100` | `0.003764976` |
| PCA | `0.002446397` | `0.007579887` | `0.003837339` | `0.004509496` | `0.003852728` |

Every one of the eight Fisher folds and eight PCA folds selected checkpoint
0. The first Adam update made the mean objective about `3.20× / 3.10×`
worse, and no later checkpoint recovered below initialization. Consequently:

- `0/16` optimizations strictly improved checkpoint 0;
- `0/16` passed fit qualification;
- no selected direction product or beta vector changed;
- all selected direction products retained numerical rank 16; and
- every selected conditional pedal remained exactly `0.5`, with standard
  deviation and range both zero on fit and held direction-supported rows.

The finite rollback authority therefore worked, but V19 did not learn a
conditional function. Fisher conditional and Fisher intercept are
behaviorally identical in every ledger and family. Pointwise trust still
passes everywhere: bounded direction reaches at most about `0.25 ||p||`, and
the half pedal emits at most about `0.125 ||p||`.

The held outer result is:

| arm | delta NLL/token | family-macro absolute delta NLL | family-macro KL/token | aggregate top-1 |
|---|---:|---:|---:|---:|
| K256 parent | `1.169234512` | `1.156531725` | `1.239356692` | `59.72073%` |
| V18 Fisher start | `1.241269976` | `1.227125494` | `1.276908376` | `59.07626%` |
| Fisher finite-joint unit | `1.268964350` | `1.254938567` | `1.306326430` | `59.29108%` |
| Fisher finite-joint intercept | `1.199943681` | `1.186557783` | `1.252299004` | `59.72073%` |
| Fisher finite-joint conditional | `1.199943681` | `1.186557783` | `1.252299004` | `59.72073%` |
| PCA finite-joint conditional | `1.162872378` | `1.149900409` | `1.245964223` | `60.36520%` |

The half-strength Fisher initializer is better than the V18 start: macro
absolute delta NLL improves `3.305914%`, macro KL improves `1.927262%`,
top-1 rises `0.644468` points, and `7/8` families win. It also beats the
unit control in `7/8` families. Those gains come entirely from the fixed
checkpoint-0 initialization, not finite joint learning.

Against the parent, Fisher instead worsens macro absolute delta NLL by
`2.596216%` and macro KL by `1.044277%`, ties aggregate top-1, wins only
`2/8` families, and has an `8.252107%` worst family regression. It improves
neither metric against its identical intercept and wins `0/8` families.
Fisher also loses all three aggregate comparisons to PCA. The PCA initializer
nominally improves parent macro absolute delta NLL by `0.573379%` and top-1
by `0.644468` points, but worsens macro KL by `0.533142%`, wins only `4/8`
families, and remains far outside every absolute gate.

Fifteen of 27 outer checks fail. Failures include every-fold optimization,
macro training improvement, nonconstant Fisher pedal, both required parent
improvements, parent top-1 gain, parent family robustness, intercept
materiality, every Fisher-versus-PCA comparison, support/core no-regression,
and ordinary/support/core absolute fidelity. Twelve checks still pass,
including exact V18 geometry inheritance, all pointwise trust checks, and the
Fisher improvements over V18 start and unit.

V19 completed exactly:

- `1,280` full-model forwards;
- `912` full-suffix backward traversals;
- `896` additional local-head autograd contractions;
- `1,808` total `autograd.grad` calls;
- 40 conceptual outer provider fits; and
- `96/96` causal off-support checks.

All 16 fold capabilities authenticated 14 examples, seven families, and five
accesses per example, with the held pair excluded. The V18 prerequisite file
and logical hashes revalidate; every parent/Fisher/PCA start fold artifact
matches V18, and base, parent, and Fisher-start fidelity replay byte-for-byte.
No full-panel refit ran, no provider sidecar or staging residue exists, and
guard, Calibration B, serving, compression, speed, and end-to-end FLOP claims
remain closed.

The authoritative report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-complete-h4-finite-joint-pedal-r16-k256-outer-lofo-a-fit16-dev-v19.json
```

Its logical report hash is
`4f0439858b7e636ae648aa12d3cdb6837350510f10b520ab1c09e69074417d46`
and its file hash is
`b29e45590c3085c18ba9ad516a3bf508d34a83c57a622f8069035d3e457a9a1e`.
The file is 1,237,051 bytes, mode `0600`, scalar/hash-only, and write-once.
Its classification is `finite_joint_pedal_outer_fidelity_insufficient`.

This is an authoritative negative result for the frozen optimizer, not a
proof that finite joint fitting or conditional computation is impossible.
The next justified experiment is a nested, fit-only logarithmic finite
microstep ladder around checkpoint 0. It should compare direction-only,
pedal-only, and joint update paths, select scale without the untouched outer
family, and retain checkpoint 0 as rollback. That cleanly distinguishes an
oversized update from a non-generalizing fit or an invalid local update
direction. Another four-step Adam run at the current learning rates is not
supported.

### V20a finite Fisher microstep preflight

V20a executed the preregistered positive logarithmic ladder around V19
checkpoint 0 for direction-only, pedal-only, and joint paths. A lexically
first joint-only sentinel had to beat both checkpoint 0 and its matched
negative mirror before the full eight-fold, all-path matrix could open. The
sentinel passed at `alpha = 0.1`: teacher KL improved
`0.002474969 → 0.002365943` (`4.4051%`), while the `alpha = -0.1` mirror
worsened to `0.002727379`.

The expanded fit-only result is:

| capability-excluded family | selected path | alpha | checkpoint-zero KL | selected-positive KL | relative improvement |
|---|---|---:|---:|---:|---:|
| alpine-fir-ring-density | direction-only | `0.1` | `0.002474969` | `0.002365912` | `4.4064%` |
| cave-pearl-layering | direction-only | `0.1` | `0.002371762` | `0.002269313` | `4.3195%` |
| kiln-brick-thermal-face | joint | `0.1` | `0.002380088` | `0.002281821` | `4.1287%` |
| obsidian-hydration-rim | joint | `0.1` | `0.002372149` | `0.002296167` | `3.2031%` |
| reed-boat-fiber-strain | direction-only | `0.1` | `0.002632962` | `0.002525382` | `4.0859%` |
| shell-midden-stratigraphy | joint | `0.1` | `0.002503776` | `0.002408958` | `3.7870%` |
| sundial-gnomon-survey | direction-only | `0.1` | `0.002319047` | `0.002222630` | `4.1576%` |
| varve-lamination | direction-only | `0.1` | `0.002495965` | `0.002397118` | `3.9603%` |

The family-equal macro objective improved
`0.0024438397 → 0.0023459125`, or `4.007104%`, clearing the frozen `1%`
materiality gate. Every fold changed live H4 and full-vocabulary logits, beat
checkpoint 0 beyond the numerical floor, and beat the exact matched negative
step. Five folds preferred direction-only, three preferred the joint update,
and no fold preferred pedal-only. The shared `alpha = 0.1` winner is strong
evidence that the V19 direction carries a genuine correction while its first
full Adam step overshot.

V20a completed exactly:

- `2,622` full-model forwards;
- `128` full-suffix backward traversals;
- `112` local-head autograd contractions;
- `240` total autograd gradient calls;
- `168` positive nonzero candidate executions; and
- nine matched negative executions.

The final scalar/hash-only report is 1,715,080 bytes, mode `0600`, regular,
single-link, write-once, and reconstructs exactly from its protected
report-ready checkpoint. Its logical report hash is
`255ba898d823d983bf1f3122796032f5001f204760b49fddc168fb13311aa84e`;
the file hash is
`318e05f1643df19a674dbc0f36f7da05c65204b9e1e0561f5e37a6273dd355da`.
No candidate, provider sidecar, lock, or staging file was emitted.

This is a fit-only optimizer preflight, not outer-family fidelity. The named
family is capability-excluded and never scored; the other seven families form
that fold's selection objective. The classification
`finite_microstep_preflight_passed_for_nested_validation` therefore
authorizes only V20b: choose path and scale through inner family splits, freeze
the choice, and score the untouched outer family once. Fresh-guard,
Calibration-B, serving, compression, speed, and end-to-end parameter/FLOP
claims remain closed.

### V20b true nested finite Fisher microstep validation

V20b executed the authorized `8 x 7` nested design without exposing an outer
family to its selector. Each unordered pair of excluded families shared one
fresh six-family checkpoint-zero/first-Adam endpoint fit, giving 28 physical
fits for 56 directed inner roles. Every directed role scored the complete
three-path by seven-alpha positive grid. For each outer family, one policy was
then selected by the family-equal macro objective across its seven inner
families, and the exact signed negative of that policy was scored in all seven
roles.

All eight selectors chose the same macro-optimal policy: joint direction and
pedal at `alpha = 1.0`. The complete result is:

| outer family excluded from selection | baseline macro | selected positive | negative mirror | improvement | positive wins | mirror wins | worst family improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| alpine-fir-ring-density | `1.228563` | `1.226332` | `1.233287` | `0.1816%` | `5/7` | `5/7` | `-0.3065%` |
| cave-pearl-layering | `1.253128` | `1.249331` | `1.260490` | `0.3030%` | `4/7` | `4/7` | `-0.9962%` |
| kiln-brick-thermal-face | `1.167452` | `1.164547` | `1.172765` | `0.2488%` | `4/7` | `4/7` | `-0.7980%` |
| obsidian-hydration-rim | `1.074490` | `1.072787` | `1.078250` | `0.1585%` | `5/7` | `5/7` | `-0.5011%` |
| reed-boat-fiber-strain | `1.214340` | `1.213248` | `1.217492` | `0.0899%` | `3/7` | `5/7` | `-0.4850%` |
| shell-midden-stratigraphy | `1.130485` | `1.124438` | `1.138943` | `0.5349%` | `5/7` | `5/7` | `-0.3526%` |
| sundial-gnomon-survey | `1.169309` | `1.164320` | `1.179116` | `0.4267%` | `4/7` | `5/7` | `-0.6158%` |
| varve-lamination | `1.359045` | `1.355068` | `1.364899` | `0.2926%` | `4/7` | `4/7` | `-0.4198%` |

This is a signed signal, but it is neither material nor family-consistent.
The positive macro improved in `8/8` selectors and beat its negative mirror
in `8/8`; mean positive improvement was `0.2795%`, the negative mirror was
`0.5068%` worse than baseline on average, and mean signed separation was
`0.7863%`. Finite, pointwise-trust, rank, execution-change, and the `2%`
worst-family bound passed everywhere. But no selector reached the frozen `1%`
materiality gate, positive wins totaled only `34/56`, mirror wins totaled only
`37/56`, and no selector reached either required `6/7` count.

The failure is not an artifact of selecting macro gain ahead of family wins.
No candidate in the complete 21-arm positive grid satisfied both `1%`
materiality and `6/7` wins. Pedal-only `alpha = 1.0` was the consistency
frontier, winning `49/56` roles with worst regression only `0.0970%`, but its
mean role improvement was just `0.0574%`. Joint `alpha = 1.0` supplied more
gain and less consistency. Family aggregation is also structured rather than
random-looking: alpine-fir and obsidian improve in `6/7` incoming roles,
whereas varve improves in only `1/7` and cave-pearl in `3/7`.

The preregistered selection gate therefore failed before outer scoring. The
authenticated lock records `outer_schedule_authorized = false`, contains zero
outer refits, and issued zero outer capabilities. No outer score, held-family
fidelity result, candidate, or provider exists. The exact completed work is:

- `2,944` full-model forwards;
- `352` full-suffix backward traversals;
- `336` local-head autograd contractions;
- `688` total gradient calls;
- 28 physical shared fits, saving 28 reciprocal duplicate fits;
- `1,176` positive candidate scores, 56 baselines, and 56 signed mirrors; and
- `2,912` authenticated teacher accesses, each bound to H4 and full-vocabulary
  logits hashes.

The phase receipt reconstructs those accesses as `336` pair-training,
`2,464` inner-positive, and `112` inner-mirror accesses, with zero outer-refit
and outer-score accesses. Resume and reconstruction overhead are exactly zero.
All 39 V20b artifacts are regular, single-link, owner-only `0600`,
scalar/hash-only files. The authoritative report is:

```text
.local-runs/google--gemma-3-270m/modal-generator-l3-l4-complete-h4-finite-microstep-nested-validation-r16-k256-a-fit16-dev-v20b.json
```

Its logical report hash is
`bb45e535074608c5feb877fbceb3342809d872f41ff1776851be656de1b0403b`;
its file hash is
`42060cc4f4dffbb11ea1203518138a27b46bcd4f483623d4d7874da083a97214`.
The fail-closed selection-lock logical/file hashes are
`3b2e6c51337cb3ccdc6dd5c35c038ab83da8c8946dc711777b7333987b740cb5`
and
`49de97cd56ba5024f53b32525e3d7ad45d3e1315365f422b4aa5b4dac731d8e7`.
The classification is `nested_inner_selection_failed`.

The result does not justify lowering the gates or reading the sealed outer
rows. All 24 path-by-outer curves reached their best observed value at the
largest tested alpha, so the smallest next diagnostic is a separately
preregistered wider positive-and-matched-negative ladder under the same
nested capability barriers. It should retain both the higher-gain joint path
and the more-consistent pedal-only path. If scale raises the pedal path above
materiality while preserving family wins, amplitude was the missing piece. If
joint gain rises while consistency deteriorates, the next model must use a
prompt-blind activation/generator-conditioned trust gate rather than family
labels. If neither closes the gate, this local microstep line should stop.
Fresh-guard, Calibration-B, serving, compression, speed, and end-to-end
parameter/FLOP claims remain closed.
