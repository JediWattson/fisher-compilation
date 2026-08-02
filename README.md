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
reference-provider experiment, fresh sealed contrast assessment, and
autonomous linear and Fisher-conditioned complete-H4 residual shadows over
Gemma's full downstream suffix and vocabulary. It is a research compiler,
not a production compression library.

[![Research ladder from the verified toy executor to the prompt-blind reference-provider fidelity result](docs/images/research-ladder.svg)](docs/images/research-ladder.svg)

## Current finding

V14–V19 now separate five questions at the source-free complete-H4 boundary:
whether the linear carrier benefits from more rank, whether a bounded
two-coordinate conditional map is genuinely two-dimensional, whether
Fisher-derived coordinates outperform an exactly parameter-matched
activation-PCA control, and whether a pointwise-bounded per-token confidence
pedal can rescue that map, and whether jointly fitting its direction and pedal
against the exact finite downstream objective repairs the analytic fit. All
arms use the same opened
16-prompt/eight-family panel, outer leave-one-family-out fitting, real Gemma
suffix, and 262,144-way language-model head. Native H4 and reverse-VJP
gradients remain fit-only.

| OOF arm, ordinary tokens | incremental provider params / MACs per token | ΔNLL/token | source→candidate KL/token | top-1 |
|---|---:|---:|---:|---:|
| K256 reverse-VJP parent | `360,704 / 524,288` | `+1.16923` | `1.25289` | `59.72%` |
| K256 + rank-16 Fisher square | `377,604 / 541,184` | `+1.16930` | `1.25288` | `59.72%` |
| K256 + rank-16 PCA square | `377,604 / 541,184` | `+1.16919` | `1.25285` | `59.72%` |
| K256 + rank-16 Fisher unit/constant pedal | `377,608 / 541,187` | `+1.25292` | `1.30265` | `58.97%` |
| K256 + rank-16 Fisher conditional pedal | `377,608 / 541,187` | `+1.24127` | `1.29057` | `59.08%` |
| K256 + rank-16 PCA conditional pedal | `377,608 / 541,187` | `+1.19812` | `1.31281` | `59.94%` |
| V19 Fisher finite-joint unit | `377,608 / 541,187` | `+1.26896` | `1.31974` | `59.29%` |
| V19 Fisher finite-joint conditional/intercept, checkpoint 0 | `377,608 / 541,187` | `+1.19994` | `1.26568` | `59.72%` |
| V19 PCA finite-joint conditional, checkpoint 0 | `377,608 / 541,187` | `+1.16287` | `1.26057` | `60.37%` |
| K320 reverse-VJP | `471,360 / 675,840` | `+1.01993` | `1.10038` | `63.69%` |
| K640 full-span reverse-VJP | `1,147,520 / 1,556,480` | `+0.81825` | `0.92572` | `66.70%` |

The K640 ceiling is a broad but expensive linear gain: versus K320 it reduces
ordinary ΔNLL by `19.77%`, KL by `15.87%`, and adds `3.01` top-1 points;
loss improves in all eight families and top-1 in seven. It still remains far
outside the frozen `0.05 / 0.05 / 95%` fidelity gates while using `2.43×`
the provider scalars and `2.30×` the logical MACs/token.

The V16 nonlinear result is more diagnostic than positive. Both Fisher and
PCA routers pass the preregistered fit-predictability and held-family runtime
geometry checks on every outer fold. Fisher's minimum fit-target R² is
`0.890`; on the two unseen sequences in each fold, its worst second/first
covariance-eigenvalue ratio is `0.375`, maximum absolute correlation is
`0.428`, and minimum residual second-coordinate energy is `0.817`. The square
therefore did not secretly collapse to a line on held families. However,
the inherited four-corner `0.25` pointwise operator bound projects each
Fisher conditional fit by a factor of only `0.000396–0.000628`; its in-fit
residual-RMSE gain is just `0.0186–0.0290%`. Fisher then wins ordinary family
loss in only `1/8` folds and is microscopically worse than both the parent and
the matched PCA control. This rejects the tested bounded parameterization,
not nonlinear conditioning in general.

V18 replaced V16's global operator shrinkage with a rowwise direction
`b = q min(1, 0.25 ||p|| / ||q||)` and a learned source-free pedal
`a = clamp(bias + [c1,c2,c1c2] weight, 0, 1)`. The direction clip never
suppresses an already-small `q`, and every emitted modal delta obeys
`||a b|| <= 0.25 ||p||`. The pedal genuinely varies on every fit and
held-family fold.
Fisher's held direction-energy-weighted pedal mean spans `0.7937–0.9991`,
its standard deviation spans `0.00425–0.28260`, and its minimum reaches
`0.2572`.

That conditionality is real but not sufficient. The fit-optimal Fisher
constant saturates at `1`, making the constant and unit controls identical.
The learned Fisher pedal improves family-macro absolute delta NLL by `0.943%`
against that control and wins all eight families, narrowly missing the frozen
`1%` materiality gate. Against the parent it instead worsens macro absolute
delta NLL by `6.10%`, macro KL by `3.03%`, and aggregate top-1 by `0.644`
points; it wins only `2/8` families. The matched PCA pedal also has lower macro
absolute delta NLL (`1.18441` versus `1.22713`) and higher top-1 (`59.94%`
versus `59.08%`). Fisher's analytic pedal targets average `1.173–1.239`, with
`66.70–78.75%` of fit weight outside `[0,1]` and clipped to that interval, so
this rung mostly learns small suppressions of an over-requested direction
rather than a selective on/off rescue.

The corrected V18 result is classified
`fisher_pedal_pointwise_trust_insufficient`: Fisher conditional is
microscopically worse than its constant on the fit objective in all eight
folds, every absolute fidelity scope fails, and no provider is selected.
The preliminary V17 receipt is preserved but is not scientific evidence: its
held unit/constant summaries used exact scalar gates on floating weighted
sums such as `0.9999999999999996`. V18 derives scalar-control summaries from
their exact serving values and conditional summaries from realized float64
mass, reproducing every V17 fidelity tensor exactly while removing those
false failures.

V19 then performed the proposed finite-objective joint fit. It initialized a
rank-16 direction from twice the V18 direction product, used a sigmoid pedal
initialized to `0.5`, and jointly updated the direction factors and pedal
parameters with four fixed full-batch Adam steps. Checkpoints `0–4` were
scored by the exact float64, full-vocabulary `KL(source || candidate)` through
the real suffix, with checkpoint 0 retained as rollback authority.

The rollback won on every fold. All eight Fisher fits and all eight PCA fits
selected checkpoint 0; none improved its training objective, changed its
selected direction or beta vector, or produced a nonconstant pedal. Fisher's
mean checkpoint curve was
`0.00244384 → 0.00781108 → 0.00344203 → 0.00451110 → 0.00376498`;
PCA's was
`0.00244640 → 0.00757989 → 0.00383734 → 0.00450950 → 0.00385273`.
The first fixed Adam update therefore made the finite fit objective about
`3.20× / 3.10×` worse, and later checkpoints never recovered below the
initializer. The selected Fisher conditional pedal stayed exactly `0.5`, so
it is behaviorally identical to the intercept control. V19 did not learn a
conditional correction.

The half-strength Fisher initializer is still informative: against the V18
start it improves family-macro absolute delta NLL by `3.306%`, macro KL by
`1.927%`, and top-1 by `0.644` points, winning `7/8` families. Against the
K256 parent it instead worsens macro absolute delta NLL by `2.596%`, macro KL
by `1.044%`, ties top-1, and wins only `2/8`; the worst family regression is
`8.252%`. PCA checkpoint 0 nominally improves parent macro absolute delta NLL
by `0.573%` and top-1 by `0.644` points, but worsens macro KL by `0.533%` and
wins only `4/8` families. All ordinary, complete-H4-support, and graph-core
absolute gates remain far outside threshold.

The V19 classification is
`finite_joint_pedal_outer_fidelity_insufficient`. V20a then ran the controlled
fit-only logarithmic microstep ladder around checkpoint 0. All eight
capability-excluded folds selected a positive `alpha = 0.1` correction and
passed the signed mirror guard: five selected direction-only and three
selected joint direction-plus-pedal updates. Family-equal fit KL fell from
`0.00244384` to `0.00234591`, a `4.0071%` macro improvement; per-fold
improvements ranged from `3.2031%` to `4.4064%`, while every matched negative
step was worse. This isolates overshoot in the frozen V19 optimizer and shows
that its learned Fisher direction contains a repeatable finite corrective
signal.

V20a is deliberately not held-family scoring. The excluded family is removed
from capability access and objective selection, but the remaining seven
families form that fold's fit objective. Its classification is
`finite_microstep_preflight_passed_for_nested_validation`, authorizing only a
nested V20b experiment that selects path and scale on inner families before
scoring an untouched outer family once. Held fidelity, a fresh guard,
Calibration B, provider materialization, serving, compression, speed, and
end-to-end parameter/FLOP claims remain closed.

V15 used exactly 80 full-model forwards, 16 VJP traversals, eight fold fits,
and 16 off-support checks. V16 used 112 forwards, 16 VJP traversals, 24 fold
fits, 16 held-family coordinate replays, and 48 off-support checks. V18 used
144 forwards, 16 VJP traversals, 40 outer fits, 32 held runtime diagnostics,
and 80 causal checks. V19 used 1,280 full-model forwards, 912 full-suffix
backward traversals, 896 additional local-head contractions, 40 conceptual
outer fits, and 96 causal checks. V20a used 2,622 full-model forwards, 128
full-suffix backward traversals, 112 local-head contractions, 168 positive
candidate executions, and nine matched negative executions. All integrity,
leakage, V18 replay, and
source-free runtime checks passed. None selected or serialized a provider;
the fresh guard and Calibration B remain closed.
These are full-vocabulary/full-suffix shadows
from one H4 boundary, not layer deletion, whole-model compilation,
compression, speed, or serving evidence. See the
[V14–V20a record](docs/progressive-compilation.md#v14-autonomous-complete-h4-full-suffix-screen)
for hashes, controls, and claim boundaries.

### Earlier iterative-generator finding

The latest development rung separated two possible explanations for the
failed causal innovation controller: a badly scaled “pedal,” or the wrong
temporal memory. A target-blind first pass over the already-open `16 × 8`
development panel found robust raw innovation scales of roughly `75–84`, not
the unit temperature used by v1:

| feature source | robust real scale | robust imaginary scale |
|---|---:|---:|
| current token only | `74.543` | `81.596` |
| EW half-life 4 | `83.556` | `82.346` |
| EW half-life 16 | `76.408` | `82.400` |
| EW half-life 64 | `77.024` | `82.890` |

That confirms the original feature really was saturated. Its prompt-balanced
`q90 |h|` was `0.996/0.997`, with only `8.25%/8.58%` of values in the useful
central interval. All 12 target-blind calibrated variants passed the frozen
health gate: their `q90 |h|` values were `0.625–0.882`, with
`67.54–94.09%` central occupancy.

Repairing the pedal did **not** repair held-family transfer. Nested
family-held-out selection chose the exact static fallback in every fold for
the scaled-L16, current-only, and full L4/L16/L64 portfolios. All three
therefore reproduced the static macro RMSE `1.971997`: `1.002%` better than
the parent (`1.991958`), but `0.242%` worse than the legacy shared-coordinate
control (`1.967239`). Scale rescue, memory rescue, and temporal value all
failed with `0/8` active selected folds, and no v2 recipe was nominated.

This was not merely the one-standard-error rule being conservative. Static
had the best mean inner-family score in all eight folds, and every finite
fixed arm was worse out of family. The closest finite arm was the exact v1
control at ridge `10`, which was still `0.0188%` worse than static. The best
calibrated arm, EW64 at scale multiplier `0.5` and ridge `10`, was `0.0283%`
worse. Mean degradation grew from `0.0363%` at ridge `10` to `0.1699%` at
ridge `1` and `0.2719%` at ridge `0.1`. In the music analogy: we repaired the
pedal travel, but every attempt to turn the effect up made the held-out mix
worse. The missing ingredient is not scalar scale or one of these three
memory lengths.

The sealed diagnostic used 16 activation-only scale forwards, then 16 source
forwards, 16 retained-parent token-VJP forwards, and 154 backward calls. It
shared one tangent bank and gradient contraction across all 13 controls and
made zero candidate, finite-displacement, or provider forwards. This remains
development-only evidence; no finite, fidelity, runtime, or compression claim
is authorized. See the
[progressive compilation report](docs/progressive-compilation.md#innovation-v2-scale-and-memory-diagnostic)
for the frozen protocol, artifact hashes, and full result.

The fixed alpha-0.5 H4 generator has now been rejected by its fresh
family-disjoint finite-NLL selection gate. The one-shot panel contained 16
new prompts across eight new families. It compared the accepted X4 parent,
the matched lag-only alpha-0 baseline, and the frozen alpha-0.5 challenger
against the same direct factorized-model source outputs.

| arm | H4 params / MACs per token | absolute aggregate ΔNLL/token | source→candidate KL/token | top-1 agreement | prompt p90 absolute ΔNLL | absolute gate |
|---|---:|---:|---:|---:|---:|:---:|
| accepted X4 only | `0 / 0` | `0.222797` | `1.286448` | `45.85%` | `1.030822` | fail |
| matched alpha 0 + B | `13,312 / 13,312` | `0.532629` | `1.396646` | `41.29%` | `1.388934` | fail |
| alpha 0.5 challenger | `34,048 / 34,048` | `0.408207` | `1.276282` | `43.05%` | `0.968875` | fail |

Relative to the matched alpha-0 control, alpha 0.5 reduced the family-macro
mean prompt error by `13.30%` and won exactly `6/8` families. That confirms
that realized H4 state contains useful incremental signal. It did not
generalize uniformly: symbolic equivalence regressed `14.91%` and unit
consistency regressed `29.62%`, far outside the preregistered `2%`
worst-family limit. The challenger also failed all five absolute source
fidelity gates. The accepted X4-only context remained better on aggregate
ΔNLL and top-1 agreement while requiring no H4 head.

The formal result is therefore `qualified: false`. Guard evaluation stays
closed, and there is no model-level compression, deployment, or latency
claim. The durable report is bound by logical hash
`63bcc5f6b03cee408164583a01109023aeb2352e3a4fa15e23ddbc2d7b842f35`,
finite-NLL hash
`3d842a558d5fb31b5ed99623c2527867a3b6936dd55e427c88c5da4939fbb044`,
and file hash
`43accd933ea8fce333d056abe7d197a8dd7178049a4c8c9d2a8c9ff2a539ad14`.

The follow-up reusable-fit attribution now separates the two compiled
boundaries instead of treating the rejected stack as one unit. It crosses
the base bridge versus the accepted X4 repair with no H4 head, the lag-only
`B` head, and the independent-state H4 head. One direct factorized-model
source plus six factorial cells over 16 examples produced exactly 112
forwards and 96 scalar/hash-only comparisons:

| X4 / H4 arm | compiled auxiliary params | logical MACs/token | prompt-absolute ΔNLL/token | KL/token | top-1 | X4 RMSE | H4 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| base / none | 403,328 | 217,920 | 0.395952 | 1.304834 | 41.14% | 0.562206 | 32.0043 |
| base / lag `B` | 416,640 | 231,232 | 0.300136 | 1.002118 | 45.03% | 0.562206 | 25.4643 |
| base / independent state | 437,376 | 251,968 | 0.302431 | 1.001871 | 44.94% | 0.562206 | 24.1366 |
| accepted X4 / none | 424,832 | 239,424 | 0.288211 | 1.275279 | 40.88% | 0.457586 | 34.9819 |
| accepted X4 / lag `B` | 438,144 | 252,736 | 0.265847 | 0.891692 | 45.72% | 0.457586 | 14.5398 |
| accepted X4 / independent state | 458,880 | 273,472 | 0.265434 | 0.892949 | 45.38% | 0.457586 | 14.4005 |

This identifies two real but overlapping repairs. The accepted X4 head
reduced prompt-absolute NLL error by `27.21%` and X4 RMSE by `18.61%`
relative to the base bridge. Adding lag `B` to that accepted parent reduced
prompt error another `7.76%` and H4 RMSE by `58.44%`. The independent-state
path then reduced H4 RMSE only another `0.96%` and prompt error only `0.16%`,
while adding 20,736 parameters and MACs over `B`; it also slightly worsened
KL and top-1 agreement and won only four of eight fit families.

The important diagnosis is objective mismatch, not absence of signal.
Accepted X4 plus either H4 head brings signed aggregate ΔNLL close to zero,
but prompt-absolute error stays near `0.266`: positive and negative prompt
errors are canceling. Euclidean H4 alignment can improve dramatically
without comparable finite-NLL fidelity. The next iterative compiler rung
should therefore retain accepted X4, use lag `B` as the cost-aware H4
baseline, and fit the next residual directly against per-prompt
source-authoritative behavior rather than adding more independent H4
capacity. This factorial is reusable-fit diagnosis only—not held-out
generalization, qualification, compression, or speed evidence. Its report
is bound by logical hash
`9fde13193b3ee915845247de3821e84ec84dbae368a2e3554e4c9423927878ed`
and file hash
`2e9fc09804af10199739ad6c14df658a6a81550e4e14b5eeb9170c52d5a809ab`.

The first explicit residual-boost iteration is also complete. It froze the
accepted-X4 plus lag-`B` parent, fit four causal logical-position scales with
eight leave-one-family-out folds, and evaluated each prompt with the provider
that never saw its family. The live campaign made exactly 64 model forwards
and added only four scalar slots plus at most 640 logical MACs/token. The
candidate was **not retained**: family-macro mean prompt-absolute NLL error
rose from `0.268343` to `0.299215` (`-11.50%` relative improvement), only
`2/8` families won, and the worst family regressed `70.64%`. KL worsened
`2.21%`, top-1 agreement fell from `45.72%` to `44.90%`, prompt p90 error
rose `4.30%`, and prompt p10 top-1 agreement fell from `40.00%` to `37.68%`.

This failure is unusually informative. The parent-point Jacobian predicted
the exact candidate with correlation `0.998966`, RMSE `0.021381`, and
`100%` sign agreement; its predicted prompt error was already `11.10%`
worse. The finite displacement did not invalidate the linear model. Instead,
position-only lag-`B` amplitudes failed to generalize across families. The
first two buckets were outside target support in every fold, while the two
supported scales changed sign or magnitude by held family. The frozen parent
therefore remains iteration zero. The next candidate should introduce a
small causal prompt/state-conditioned **direction**, not more global
position damping or a path-integrated Jacobian. The development report is
replay-bound to every fold fit, OOF provider/execution/observation, and
resource receipt. Its collection hash is
`dee8864210f6047e3f05b67515a5f77b54a1ccbe332de5b9eb96519eca33714c`,
logical report hash is
`1e1e284d354dd6048406b99a335bc2065e6767e706b8e781791bb1fd365c49ca`,
and file hash is
`4ace545b0dea88aeebbbe7e8ddd57a89ff986e56a1817ab4500ad44fa056afb3`.

The second iteration implements the requested prompt/state-conditioned
correction without prompt IDs. It reads the accepted lag-`B` head's two
strongest modal responses, maintains a two-scalar causal running balance,
and applies one bounded `2 × 2` modal route before reusing the parent's
decoder. The child has four learned scalars, two derived constants, two
runtime-state scalars per sequence, and six marginal linear MACs/token. Its
zero matrix is exactly the accepted parent. Its explicit carry supports
incremental generation when the upstream parent executor supplies modal
chunks using its own lag cache.

This candidate produced the first positive family-disjoint iterative result,
but it was conservatively **not retained**. Family-macro mean prompt-absolute
ΔNLL improved `0.268343 → 0.265900` (`+0.91%`), prompt p90 error improved
`2.71%`, and five of eight families won. The preregistered gate required six
wins and no family worse than `-2%`; the worst family regressed `6.43%`.
Every scientific gate passed: all eight fold designs had rank four, every
route edge was supported, family-macro balance-feature standard deviation
was `0.1547`, and the selected modes carried `87.15%` of measured lag-`B`
modal energy. The parent-point prediction remained excellent
(`r = 0.999985`, RMSE `0.002107` ΔNLL/token, `100%` sign agreement), so the
remaining problem is conditional generalization rather than Jacobian error.
All eight fits reached the `0.25` operator-norm trust boundary.

The frozen parent therefore remains iteration zero, but this is a materially
stronger result than position scaling: observable causal modal state does
carry useful corrective information. The next candidate should split this
shared rotation into a tiny causal regime/expert route, rather than add
prompt identifiers or simply enlarge the same global step. The iteration-two
report is bound by collection hash
`f7ea40fd6bb5695ed9da5f21d8e8d279a8e257e00fb8b4c7bd204718b0c17b8c`,
logical hash
`2836d9bde0d39a5b1acbaab7d34fca69c5ad7eab9a1632a63900565eb8ff2207`,
and file hash
`0a269c7f6336bccb601a868c157cea4dd4dce2d413a55ae7531471679d9f45f1`.

The third iteration tested that exact conditional split. It keeps the same
causal balance state and frozen iteration-zero parent, but dispatches each
active row to one of two independently bounded `2 × 2` routes according to
whether the balance is negative or nonnegative. Only the selected expert is
evaluated. The edge adds eight learned scalars, two derived constants, two
runtime-state scalars per sequence, six marginal linear MACs/token, and at
most six nonlinear scalar operations/token. Its all-zero matrices are
exactly the parent.

The live sign-expert result was **not retained**. Family-macro mean
prompt-absolute ΔNLL changed `0.268343 → 0.269441` (`-0.41%`), only `4/8`
families won, and the worst family regressed `3.35%`. This is weaker overall
than Iteration 2's `+0.91%`, `5/8` wins, and `-6.43%` worst family, although
the original state-invariant failure was repaired from `-6.43%` to `+3.06%`.
That repair came with lost gains: counterfactual isolation moved from
`+3.82%` to `-3.13%`, reference frame from `+0.05%` to `-3.35%`, and the
temporal and uncertainty wins became smaller.

The split itself was real rather than degenerate. Every fold had rank eight,
all eight expert-route edges were supported, and the negative/nonnegative
regimes received family-macro active-row fractions of `21.67%`/`78.33%`.
All scientific and resource gates passed. Linearization was also excellent
(`r = 0.999990`, RMSE `0.001729` ΔNLL/token, `100%` sign agreement) and
predicted a small regression, so this is not a missed nonlinear gain. Both
experts hit their `0.25` trust bound in every fold.

The parent therefore remains iteration zero; there is no retained full-fit
provider or deployment authorization. A direct fold comparison explains why
the extra experts hurt: median design conditioning rose from `69.9` to
`555.7`, while mean pairwise coefficient cosine fell from `0.972` to `0.686`.
The next preregistered candidate should therefore pool rather than multiply
experts: a four-scalar conformal route affine in the continuous balance,
with both endpoints trust-bounded. This preserves conditionality, returns to
Iteration 2's capacity, and avoids starving a hard sign branch. The
iteration-three report is bound by collection hash
`9789d185cca7001a399a02366b928cebfdc4d01bb0df22d8fa0f4a8ae4cfd1d0`,
logical hash
`c6642c16fd2620ad057b9fafcb53e21c02f29dd4dfb2f73c9e9aaf9e3f6d05a2`,
and file hash
`0c0bcdd2c83e89a5bfd69bda8816da27e2834b0750dcc1792aaaeec9b609c6aa`.

Iteration 4 implemented that pooled route:
`C(g) = C(a0, b0) + g C(a1, b1)`, with
`delta_top(t) = g_t modal_top2(t) C(g_t)`. It uses the same four learned
scalars as Iteration 2, but spends two on a shared conformal transform and
two on a continuous balance-dependent contrast. A single global radial
projection bounds both `g=-1` and `g=+1` endpoints at operator norm `0.25`,
which bounds every intermediate state.

The live result was the strongest broad generalization result in this
iteration branch, but it was still **not retained**:

| iteration | learned scalars | marginal linear MACs/token | macro ΔNLL improvement | family wins | worst family | median condition | fold cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2: shared `2 × 2` route | 4 | 6 | `+0.91%` | `5/8` | `-6.43%` | `69.9` | `0.972` |
| 3: sign experts | 8 | 6 | `-0.41%` | `4/8` | `-3.35%` | `555.7` | `0.686` |
| 4: affine conformal route | 4 | 8 | `+0.32%` | `6/8` | `-3.32%` | `25.35` | `0.984` |

Iteration 4 passed the required six-family win count and every scientific
and resource gate. It failed only the preregistered worst-family floor:
state-invariant prompts regressed `3.32%`, below the allowed `-2%`. The
parent-point prediction was essentially exact (`r = 0.9999999`, RMSE
`0.000164` ΔNLL/token, `100%` sign agreement), so this is not a nonlinear
overshoot. The stable pooled direction is still missing a causal variable
that distinguishes two state-invariant prompts requiring opposite
corrections.

The frozen parent therefore remains iteration zero. Any next conditional
feature should be preregistered and tested on a new frozen development panel,
because the current A-fit result has now exposed which family and prompts
fail. The Iteration 4 report is bound by collection hash
`4958b0f64094758c13740c5ffa11fac41652af2dc6e01232c3b80c32155fde5c`,
logical hash
`48c708e8b717610b4ec018f9c129494c08d9027b6f05afef612ab79d91e938f1`,
and file hash
`bad81101b0e07d75cca23ed85143d0f3d0c8649423fa84fdbefef08e55afd541`.

Iteration 5 screened the proposed second causal statistic before spending a
fresh panel. It retained balance `g`, added centered negative-balance
occupancy `o`, and fit the six-scalar route
`C(g,o) = C0 + g Cg + o Co`. Cumulative occupancy and a 16-token-half-life
EW occupancy shared one parent VJP per development prompt. Both routes used
family-balanced, column-standardized ridge and one radial trust projection
over all four `(g,o)` corners.

The screening stopped safely before fresh selection:

| occupancy arm | predicted macro absolute ΔNLL/token | occupancy std | standardized condition | fold cosine |
|---|---:|---:|---:|---:|
| cumulative | `0.267743` | `0.242363` | `282.51` | `0.7492` |
| EW, half-life 16 | `0.268012` | `0.313540` | `303.34` | `0.7406` |

The concern that a running fraction might collapse toward zero was not the
failure. Both arms saw both signs (`218` negative-balance and `790`
nonnegative-balance rows), all six coordinates were supported, every fold
had rank six, and the selected top modes still carried `87.1460%` of parent
modal energy. The failure was identifiability: even after column
standardization, median condition exceeded the frozen `100` ceiling, and
mean fold-direction cosine stayed below the `0.90` floor. Neither arm was
selected, the durable claim was never created, and the new 16-example panel
remains unopened.

This narrows the next move: do not lower the stability gates or merely add
more occupancy variants.

The first identifiability repair is now complete. For each arm and each
14-prompt LOFO training fold, it split the Jacobian into the existing four
columns `B` and occupancy pair `O`, fit the weighted projection
`A = (sqrt(W) B)^+ sqrt(W) O`, and trained on `[B, O - B A]`. The result was
then mapped exactly back to the unchanged runtime coefficients
`theta = [gamma_B - A gamma_O, gamma_O]`. `A` is fit-only metadata: deployed
parameter count, state, MACs, and route semantics do not change.

| occupancy arm | direct condition | residualized condition | retained occupancy energy, median / minimum | mapped fold cosine | predicted macro absolute ΔNLL/token |
|---|---:|---:|---:|---:|---:|
| cumulative | `282.51` | `16.91` | `3.63% / 1.63%` | `0.7492` | `0.267743` |
| EW, half-life 16 | `303.34` | `16.91` | `3.42% / 1.56%` | `0.7406` | `0.268012` |

This separates numerical conditioning from scientific identifiability.
Residualization worked algebraically and reduced condition by about `94%`,
but it removed mostly duplicated signal. The small independent remainder
missed the preregistered `5%` retained-energy floor, the common deployed
direction remained far below the `0.90` cosine gate, and predicted NLL was
unchanged to roughly `1e-8`. Neither arm passed. The prompt-free development
report has logical hash
`3ea83a89db0fe4f9f73f727783acddde3292a26a151e8d7057b8a6a4db6c1cbf`
and ignored local file hash
`a7b6f483c4dd6376ab7a61a36cd991e620e11fe5b935996ad32bc86fcb96ac1f`.

No fresh claim was created. The previously prepared panel is also no longer
eligible for confirmation: the residualized recipe differs from its frozen
plan, and its private payload was exposed during a local boundary audit.
Any future confirmation must use a newly frozen recipe and a newly blinded
panel. A separate one-dimensional residual-SVD controller is the clean next
development candidate; it must not be introduced as an adaptive fallback
inside this failed run.

### Exact token-loss Fisher rung

The next development rung now keeps the cancellation information that the
prompt-level fits discarded. For every supervised loss token it computes the
exact directional row

```text
Q[t, k] = d token_nll[t] / d route_coordinate[k]
```

over the eight unique cumulative/EW occupancy tangents. The mean of those
rows must exactly replay both older six-coordinate prompt Jacobians. Each
prompt is then reduced to `Q^T Q / N`, its target cross-moment, target second
moment, and mean score; raw token IDs, logits, activations, and gradients are
not retained in the prompt Fisher record.

The fixed 16-prompt A-fit collection uses 16 source forwards plus 16 retained
parent-VJP forwards: exactly 32 model forwards. Each retained graph is
traversed in chunks of eight loss-token cotangents, so a prompt with `N`
supervised tokens uses `ceil(N / 8)` batched backward calls. Validation holds
out whole prompt families. Families have equal mass, prompts have equal mass
within a family, and tokens are never treated as independent split units.

The resulting Fisher coupling is symmetric. An off-diagonal says that two
already-declared causal tangents are co-sensitive; it cannot determine a new
causal arrow. Any stable candidate still needs held-family finite-displacement
and JVP/intervention orientation before it can become an executor edge.

This rung is development-only. It performs no candidate or fresh-panel
forwards, compiles no provider, and makes no parameter, MAC, latency, or
compression claim. Run its local diagnostic with:

```bash
fisher-graph-gemma-l3-l4-iterative-token-fisher-dev --help
```

The exact A-fit run covered `1,157` supervised loss tokens in `32` model
forwards and `153` batched backward calls. It resolved the earlier
observability problem: both six-coordinate Fishers were full rank, median
standardized condition was about `38`, every occupancy coordinate retained
at least `9.12%` Fisher energy beyond the shared/balance span, and both arms
produced the same six stable couplings across all eight LOFO folds. Those
couplings form two dense three-node components, one real and one imaginary.

It did **not** solve the mutation fit. Cumulative and EW macro held-family
RMSE changed by `-0.573%` and `-0.542%`; each won only `2/8` families, and
fold-coefficient cosine was only `0.565` and `0.607`. Neither arm passed, so
no provider or runtime claim was produced. The result cleanly separates a
successful Fisher map from a failed single-global-coefficient executor. The
next development hypothesis is a low-capacity causal token-conditioned
coefficient/router fit over the frozen coupling map, not compilation of the
current coefficients.

The ignored local report has logical hash
`6ffaf61639626b47101324573fff646de187f45212b29c88f236749bb2beb65b`
and file hash
`d80d78580102168c031a200472bd8c0259a264f2e4d8fc269a5a73b1ccd363b9`.

### Partially pooled corrective screen

The follow-up freezes the six stable Fisher couplings and tests whether
nested family-held-out shrinkage can stabilize the already token-conditioned
balance/occupancy route. It keeps the two shared conformal coefficients and
selects one common ridge for the four conditional deviations from
`{0.1, 1, 10, infinity}`. The exact shared-only fallback (`infinity`) is a
first-class option, and ridge selection happens inside every outer training
fold. Conditional success also has explicit materiality floors: at least
`0.5%` family-macro improvement over shared-only, with a family counted as an
incremental win only after at least `0.1%` improvement. This prevents
floating-point dust from authorizing a controller.

All eight cumulative-primary folds selected shared-only. That raised
fold-coefficient cosine from `0.565` to `0.955`, produced `6/8` family wins,
and limited the worst regression to `-0.485%`, but macro held RMSE improved
only `0.173%` and conditional behavior added exactly `0%` over the shared
control. EW produced the same decision. The screen therefore rejects the
current balance/occupancy variables as transferable corrective signals; it
does not reject the stable Fisher map.

This replay used zero new model forwards and remains adaptive development.
No provider, causal edge orientation, graph traversal, or runtime claim was
authorized. The next experiment needs a newly frozen causal feature and new
family-disjoint token collection. See the
[progressive compilation report](docs/progressive-compilation.md#partially-pooled-token-fisher-corrective-screen)
for the nested protocol, gates, and exact hashes.

Run the local replay with:

```bash
fisher-graph-gemma-l3-l4-iterative-fisher-corrective-dev \
  --token-fisher-report-sha256 <logical-sha256> \
  --token-fisher-report-file-sha256 <file-sha256>
```

### Parallel capacity-control rung

The authenticated function-preserving expert-count control held outer rank
64, expert rank 64, router width 16, and the complete D3 fit contract fixed
while raising the routed expert count from two to four. Every parent expert
was split into an active child and a dormant child. Duplicating its router
logit halves each child's probability; doubling the active child's output and
zeroing the dormant child's output preserves the parent contribution. The
primary observable and provider-chart JVP therefore matched at initialization
to maximum absolute errors of `1.78e-15`, with exactly zero weighted-objective
difference. The dormant paths also passed the preregistered two-step
gradient-openness checks.

| arm | stored scalars / canonical MACs | ordinary error / cosine | null | radial pass / macro error | signed pass / macro / worst error | weighted training objective |
|---|---:|---:|---:|---:|---:|---:|
| 2-expert exact source replay | `31,492 / 5,555,776` | `0.00621631 / 0.99998116` | `24/24` | `16/16 / 0.063077` | `3/8 / 0.383903 / 0.752043` | `0.164106` |
| 4-expert matched lift | `48,166 / 8,985,792` | `0.00555688 / 0.99998460` | `24/24` | `16/16 / 0.054820` | `3/8 / 0.362307 / 0.668409` | `0.144633` |

Both arms passed all `12/12` ordinary, `24/24` exact-null, and `16/16`
radial checks. Four experts improved ordinary error by `10.61%`, radial macro
error by `13.09%`, signed macro error by `5.63%`, signed worst error by
`11.12%`, and the weighted training objective by `11.87%`. Those are real
continuous improvements, but they did not recover another categorical
identity: both arms passed only `base_01`, `base_04`, and `base_06` of the
eight signed checks. The maximum ordinary per-probe p90 error also worsened by
`4.53%`.

The valid formal outcome is `primary_both_fail`. The conditional replication
did not run because only a valid 2-expert-fail / 4-expert-pass primary result
could open it. Four experts cost `52.95%` more stored scalars and `61.74%`
more declared canonical MACs than two. This is therefore a causal negative
result for four routed experts under the matched 600-step budget, not
compression or speed evidence. It authorizes only a separately preregistered
eight-expert full-count oracle. It authorizes no descending expert-count
ladder, C3, held-out generalization, full-model replacement, compression,
rank reduction, or wall-clock speed claim.

The durable external result receipt is logical artifact
`8c07a30129f2bb7c5e704e54ffc7e23fc947a27367d164490002e26aa699a015`,
tensor file
`1dda9cdae257a18155c49a8daac90ef13401d7928a1a506a5d3027e2b35ebf4f`,
and report
`cef24e40718ef2c6983d3fd08f45a1e5b5f87e2a2b07f710bc297570726d0723`.
The protocol binding is
`84c423f4f4b3020ff07d2340379707586c51f706b046edf96e4a0a95adf8c6bc`
and the code bundle is
`e6a22bb29f468c9f8ab02fd308e6eb648cd9da09f95b3ed041a7aa364b62b127`;
the executable protocol was preregistered in commit `0f3166d` and its exact
source-replay finalization was corrected and refrozen in commit `d69024c`
before the accepted run. The tensor artifact and tensor-free JSON report
remain ignored under `.local-runs/`; the receipt recorded here is the durable
trust root.

```bash
fisher-graph-gemma-l3-l4-function-preserving-expert-count-dev describe

fisher-graph-gemma-l3-l4-function-preserving-expert-count-dev run \
  --device cpu \
  --dtype float32
```

### Parallel compiler rung: graph-organized global SVD

The graph-Fourier result is now separated into the jobs it actually performs
well. A rank-45 global SVD supplies the numerical compression basis, while the
fit-only signed Fisher/GFA graph organizes those retained generators into
four routable packs. Turning on all packs reproduces the same SVD operator;
the graph changes only generator layout and conditional execution.

The frozen Gemma plan uses packs `8 / 8 / 8 / 21`. Its deployment-form edge
state contains `279,744` float coefficients versus `393,216` dense fit-knot
coefficients (`28.86%` fewer). For a complete 32-lag response with
interpolated cores cached, all-on rank-45 execution uses `72.51%` of the dense
linear MACs.

The routing curve was corrected for an important C2 data confound: exact-zero
padding is about 91% of the raw rows and is excluded before rate scoring. On
the 1,132 nonzero development-selection directions at held-out response
origin 32:

| organization / retained bound mass | mean active rank | cached-core MAC fraction | error vs dense measured response |
|---|---:|---:|---:|
| all-on global SVD | `45.00` | `72.51%` | `3.179%` |
| signed-GFA packs, `95%` | `43.78` | `70.61%` | `3.206%` |
| contiguous SVD packs, `95%` | `43.14` | `69.60%` | `4.148%` |
| signed-GFA packs, `90%` | `30.33` | `49.58%` | `12.936%` |
| contiguous SVD packs, `90%` | `30.61` | `50.02%` | `10.724%` |

This is a useful but mixed result. The graph organization gives the better
high-fidelity point against the contiguous control, but the contiguous
control wins at the more aggressive threshold. Eight size-matched random
controls also remain in the artifact. The router uses conservative
operator-norm certificates and reuses the source projection, but its cost is
excluded from the displayed cached-core MAC ratios. A later float-execution
audit found that the exact-arithmetic certificate does not yet bound the
implemented accumulation tightly enough, so routing is disabled in the
locked one-shot rung.

The opened C2 split makes this development evidence, not fresh confirmation.
A subsequent source-authoritative shadow did measure one Calibration-A
natural prompt. It rejected the current edge: the all-on modal error was
`4.8208`, a true 64-mode projection oracle still had `0.9741` full-width
error, and exact X4 injection still changed NLL by `+2.0121` per token because
the residual carrier was incomplete. This is development diagnosis, not
held-out confirmation, whole-block replacement, whole-model compression,
wall-clock latency, or GPU speed. The ignored graph artifact is bound by
logical hash
`b3e011d8067ff3538888851c476fba03c57f4e9f172f923c20fdd90ac0799f84`,
tensor file
`d77a60532b660160413331ceddbe8d970c2828d53ff5788642250ff3c5d49fa1`,
and report
`5c958c54fbcd55239cc1f5943dcb1bf138bbd4116233783bf7020e1023f4998a`.

The one-shot path is now fail-closed around that diagnosis. Its only supported
Calibration-B entry point preflights the exact runtime, live adapter, and
locally loaded Gemma tokenizer; atomically claims the manifest in one fixed
per-user host ledger; and only then loads each of the 96 prompts by identity.
It owns tokenization, the required `3 + 1 + 1` source/oracle forwards per
prompt, streaming evaluation, and terminal receipt creation. No independent
held-out issuer, evaluator, report callback, or caller-supplied observation
can produce a supported success receipt. The tokenizer check binds its backend
program, full token-to-ID vocabulary, added/special tokens, and library
versions—not only its name and configuration.

Per-example receipts bind the prompt identity, exact token tensors,
model/executor fingerprints, causal grid, both oracle interventions, and every
tensor being scored. The internal evaluator requires the complete frozen
96-example panel, all valid next-token boundaries, the real `262,144`-token
Gemma output vocabulary, and unique receipts while streaming scalar
statistics. Only the scalar report and immutable terminal receipt escape.
Hashes are reproducible integrity/audit receipts, not hostile in-process
attestation, and the host ledger is not a cross-machine authority. The
transaction has not been invoked here, so Calibration B remains unopened.

That failed rank-64 candidate can now be used as iteration zero of a
residual-guided progressive compiler rather than treated as an all-or-nothing
answer. The model-independent controller repeats fit-only residual mapping,
typed graph mutations, and family-disjoint A-selection measurements. Failing
candidates enter a repair phase; candidates inside the fidelity envelope
enter a Pareto compaction phase. Every accepted transition is parent-bound and
charges compiled, support, and retained-source parameters, bytes, and logical
MACs. Declared-incomplete or scope-incomparable accounting cannot authorize a
candidate.

The Gemma-specific worker and first candidate-bound lowerer are now present.
The worker materializes three
strict A-only panels, streams source-authoritative seed/projection/carrier
metrics, and maps distinct residuals at `layer.4.mlp.normalized_input` and the
complete `layer.4.output` boundary. A single native autograd pass differentiates
next-token NLL with respect to both X4 and H4; the private A-fit archive keeps
full L3 source modes, logical positions, masks, native/candidate boundaries,
and gradients while scalar receipts retain only hashes and ranked geometry.
After X4 is accepted, a separate fit-only pass can replay that exact
candidate, detach its realized H4 as the autograd leaf, and measure the
candidate-conditioned H4 NLL VJP. The bridge requires this gradient pass to
produce a bitwise-identical execution artifact to the ordinary candidate
pass, and it never retains model-parameter gradients.

The mapper uses family/example-balanced residual PCA with a bounded NLL-VJP
alignment tilt and post-hoc activation-gradient-Gram scores. It uses the
native tangent at the seed and the candidate-conditioned tangent after X4;
this is not a Fisher eigendecomposition or held-out JVP validation. The lowerer fits
homogeneous causal finite-displacement ridge kernels and builds immutable X4
or H4 repair heads. Its executable bridge owns one Gemma prefill forward: Y3
is clamped, the rank-64 base graph and X4 head run at X4, the H4 head runs
later in the same nonlinear carrier, and unsupported rows preserve the
same-pass reference rather than a hidden native-X4 fallback. Joint X4+H4
candidates are constructed sequentially—remeasure after X4, then fit H4—so
the H4 target reflects the first head. The baseline H4 repair reads only L3
source modes. The optional realized-state arm also reads the immutable
post-X4, pre-H4-correction activation in the same forward, projects it through
the existing H4 decoder, and mixes those coordinates through an authenticated
`rank × rank` state kernel. Exact scoped parameters, bytes, and linear/modal
MAC upper bounds include the retained Gemma carrier.

This is a tested executable overlay, not a qualified compression result: it retains
the factorized Gemma model and adds bridge/head state. It is also an
integrity-heavy research prefill path, not a latency result: full-model and
tensor hashing, Python dispatch, transfers, temporary memory, cached
autoregressive decode, and wall-clock speed are outside the declared
linear/modal MAC scope. The legacy multi-pass shadow remains the
source-authoritative evaluator.

Repeated selection does not spend the final guard. The implemented campaign
uses separate pairwise family-disjoint `calibration_a_fit`,
`calibration_a_selection`, and `calibration_a_guard` roles. A guard-incapable
development runner can freeze an eligible challenger at
`ready_for_guard_claim`; only a separate finalizer accepts the one-shot guard
callback. In that mode the Gemma worker receives no guard authority, provider,
or materialized guard panel. The claim-gated path still requires an external
durable claim-first authority, and only a guard-passing, budget-compliant
result can emit a candidate-binding handoff. The existing Calibration-B
manifest is registered only as a forbidden identity and remains unopened.
Because the legacy one-shot runtime is bound to the old failed candidate, a
new progressive winner will require a candidate-bound shadow protocol/runtime
before it can consume that final one-shot.

The first real CPU campaign is now recorded as adaptive development. It
preregistered a forced X4 → remeasure → H4 transition, separated actual
candidate-output qualification from ancestor/modal diagnostics, and bound the
earlier pilot transcript as lineage. X4 reduced aggregate and worst-family
boundary relative error by 11.8% and 4.2%, so it was accepted as the staging
step. The original L3-only H4 candidate improved KL and aggregate top-1
agreement but badly regressed absolute and tail NLL, so it was rejected
against the pre-X4 anchor:

| selection candidate | abs. ΔNLL | KL | top-1 agreement | p90 abs. ΔNLL |
|---|---:|---:|---:|---:|
| rank-64 seed | 0.0937 | 1.3092 | 0.4189 | 0.4333 |
| X4 rank-8 | 0.1450 | 1.2833 | 0.3962 | 0.3894 |
| X4 + H4 rank-8 (L3-only input) | 0.3778 | 0.9644 | 0.4792 | 0.7927 |

The result is `stalled_fidelity`: no challenger or handoff was emitted, and
the rotated manifest-global guard remains unopened and unclaimed. The X4
runtime accounts for 212.50M parameters and 212.25M logical MACs/token,
20.74% and 20.82% below raw Gemma respectively, but almost all of that saving
comes from the retained factorized carrier—not from the residual head. The
first loss-aware follow-up has also run. It replaces only the H4 coefficient
fit with the bounded metric `I + u uᵀ`, where `u` is the normalized
source-native NLL gradient; deployed shape and cost remain unchanged.

| H4 fit / input | fit linearized-NLL RMSE | fit normalized-direction RMSE | fit hidden RMSE | selection abs. ΔNLL | selection KL | top-1 | p90 abs. ΔNLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| hidden-residual ridge / L3 | 2.822196 | 2.937411 | 0.000533 | 0.377753 | 0.964429 | 0.479245 | 0.792749 |
| source-NLL-VJP metric / L3 | 2.821386 | 2.936489 | 0.018397 | 0.377860 | 0.964445 | 0.479245 | 0.792909 |
| candidate-H4-VJP remap + metric / L3 | 2.720673 | 3.337574 | 0.021149 | 0.377742 | 0.964426 | 0.479245 | 0.792648 |
| candidate-H4-VJP + realized-H4 decoder modes | 2.720673 | 3.337573 | 0.021148 | 0.389134 | 0.948135 | 0.467925 | 0.799620 |

The candidate-tangent-conditioned arm uses the post-X4 VJP in both residual
direction mapping and the bounded coefficient metric. It reduced fit
linearized-NLL RMSE by `3.60%` versus ridge, but normalized-direction RMSE
worsened `13.62%` and hidden error became `39.69×` ridge. Its held-out
changes were microscopic: absolute ΔNLL improved by `0.0000111`, KL by
`0.0000035`, p90 ΔNLL by `0.0001013`, and top-1 did not change. The H4 child
therefore failed every execution-fidelity axis and was rejected with
`stalled_fidelity`; X4 remains active.

The realized-state arm then held rank, lag count, VJP objective, corpus, and
one-forward execution fixed. For current pre-correction H4 state `h_t`, output
decoder `D`, lagged L3 modes `s`, and state kernel `A`, it executes
`q_t = h_t Dᵀ` and
`Δh_t = [Σ_l s_(t-l) K_l + q_t A] D`. The pointwise term reads no future
position and introduces no feedback loop. At rank 8 it adds exactly 64
parameters, 512 runtime bytes, and 5,184 logical MACs/token.

That feature was active but did not add meaningful fit power: hidden RMSE
improved only `0.00154%`, while the two NLL fit errors changed by less than
`0.000003%`. Held-out KL improved `1.69%`, but absolute ΔNLL worsened `3.02%`,
p90 ΔNLL worsened `0.88%`, and top-1 fell from `127/265` to `124/265`.
Every execution-fidelity axis still failed, the H4 child was rejected, and X4
remains active with `stalled_fidelity`.

The fit-only incremental-signal rung has now isolated that confound. It
collected the accepted-X4 trace once, then swept lags `1/2/4/8/16/32` and
independent H4 input ranks `8/16/32` under nested leave-one-family-out
residualization. Its CLI accepts neither a selection nor guard input. The
lag-32 design did exactly saturate every outer-fold row space, so all lag-32
H4 cells added numerical rank zero. Lags 1 through 16 left row-space capacity
and every requested H4 rank was identifiable.

Lag 4 showed the clearest real signal, but not a stable enough one:

| lag-4 H4 input | macro linearized-NLL improvement | family wins | worst family | projected-residual improvement | head params | head MACs/token |
|---|---:|---:|---:|---:|---:|---:|
| reused output decoder, r8 | 1.847% | 3/4 | -2.594% | 14.422% | 7,232 | 12,352 |
| independent H4, r8 | 1.582% | 3/4 | -1.706% | 14.495% | 12,352 | 12,352 |
| independent H4, r16 | 2.722% | 3/4 | -4.317% | 19.444% | 17,536 | 17,536 |
| independent H4, r32 | 3.785% | 3/4 | -2.405% | 31.690% | 27,904 | 27,904 |

The r8 arm stayed inside the 2% worst-family bound but missed the required 2%
macro gain. Ranks 16 and 32 cleared the macro threshold but failed the
worst-family bound. The terminal result is therefore
`no_crossfit_incremental_signal`: zero of 24 cells qualified, no selection
panel was opened, and no head was deployed. This is developmental evidence
that realized H4 contains useful incremental directions around lag 4, but
with family-dependent transfer—not yet a compression, latency, or nonlinear
gating result. The report is bound by logical hash
`57f79eb3bde8f3eaddaaf93e1fabe1c71325dc39e2f0db675c3837f735be2641`
and file hash
`103d2c7cc04f16769d845c75d10c81a8889c155638fdf9527478645aa83fc0b8`.

That larger replication has now run on 16 new prompts across 8 new families
and 1,008 affected rows. The fit role was replaced without opening either
protected role; the selection and guard preclaim views remain exactly
unchanged. The 3/4 consistency rule generalized to 6/8 wins. Lag 4 did not
replicate, but the effect reappeared at larger causal context:

| independent H4 input | macro linearized-NLL improvement | family wins | worst family | projected-residual improvement | head params/MACs |
|---|---:|---:|---:|---:|---:|
| lag 4, r32 | -0.594% | 3/8 | -5.070% | -3.515% | 27,904 |
| lag 8, r16 | 2.352% | 7/8 | -6.016% | 7.481% | 19,584 |
| lag 8, r32 | 2.675% | 6/8 | -9.022% | 18.540% | 29,952 |
| lag 16, r32 | 4.379% | 7/8 | -3.000% | 19.018% | 34,048 |

The lag-16/r32 arm cleared the macro, win-count, rank, and secondary-metric
gates, but its one losing family exceeded the 2% regression limit. The result
therefore remains `no_crossfit_incremental_signal`: zero of 24 cells
qualified, no selection panel was opened, and no runtime head was emitted.
This is stronger evidence that realized H4 contains transferable incremental
signal, but it also rejects the earlier claim that lag 4 is a stable fixed
architecture.

The next isolated rung locked that single lag-16/r32 head and evaluated only
the preregistered residual-H4 scales `0.25/0.5/0.75/1.0`. The encoder and
state kernel were fit once per family fold; alpha only scaled the fixed
incremental prediction. Alpha 0 remained the matched L3-only control, and
alpha 1 exactly reproduced the expanded source cell, including every scalar,
encoder hash, and state-kernel hash.

| fixed residual-H4 scale | macro linearized-NLL improvement | family wins | worst family | projected-residual improvement | normalized-direction improvement | eligible |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.25 | 1.840% | 7/8 | -0.016% | 7.522% | 1.697% | no |
| **0.50** | **3.198%** | **7/8** | **-0.527%** | **13.424%** | **2.769%** | **yes** |
| 0.75 | 4.049% | 7/8 | -1.527% | 17.353% | 3.193% | yes |
| 1.00 | 4.379% | 7/8 | -3.000% | 19.018% | 2.958% | no |

The fixed rule selected the smallest passing value, alpha `0.5`. This is a
real fit-only robustness pass: it retained more than the required 2% macro
signal while moving the lone family regression comfortably inside the 2%
bound. It froze a tensor-hash-only recipe with 34,048 parameters and 34,048
logical MACs/token. Damping changes neither storage nor compute for a nonzero
head because the scale folds into the kernels offline.

That fit-only result authorized one fresh, family-disjoint finite-NLL
selection of the single alpha-0.5 head. The recipe was deterministically
materialized as two authenticated executors: a `13,312`-parameter/MAC
lag-only alpha-0 control and a `34,048`-parameter/MAC independent-state
alpha-0.5 challenger. The materialization report is bound by logical hash
`27944f6e35cbf8e7828af56ef2589df486d55ec5654a92487438d077f3d01c94`
and file hash
`7dcde44cd0b01f0e810bc034d5f9a05ec395c95c77b2ba94e4f4d53970236a20`.

The fresh selection has now run once. Alpha 0.5 improved the paired
family-macro mean prompt error by `13.30%` and won `6/8` families, but its
worst family regressed `29.62%` and it failed every absolute source-fidelity
gate. Selection therefore rejected the head and guard remains unopened. This
is evidence for a useful but unstable incremental H4 direction, not a
model-level compression, latency, or downstream-fidelity claim.

The damping report is bound by logical hash
`dc85bb184b88a394d89d6e907ae496a3e920a011f9b7b7fb7e4f6b9a7d8e7a65`,
analysis hash
`1e01682390c135cd5616f966aa66fead3306bf785d88f5775a9ab6a1a4d439fd`,
and file hash
`1ddd80255c014d23a598ad4ec4543218a6437a39a6af3f50697ed98ed64fd94b`.

The expanded source report is bound by logical hash
`0dbccb00cc17995fe458a7eea6083ca030726c88b5b7df5884a0ae34087a107d`
and file hash
`91d38b7ee2abd2693a0855e4fd10d082812d70e78cfee05e04ecd27c918ca584`.
The replacement guard remains unopened and unclaimed. The one-shot selection
report is bound by logical hash
`63bcc5f6b03cee408164583a01109023aeb2352e3a4fa15e23ddbc2d7b842f35`
and file hash
`43accd933ea8fce333d056abe7d197a8dd7178049a4c8c9d2a8c9ff2a539ad14`.

The controller, Gemma seed binding, acceptance rules, and next executable rung
are described in
[`docs/progressive-compilation.md`](docs/progressive-compilation.md).

```bash
fisher-graph-gemma-l3-l4-graph-organized-svd-dev
```

The factorization, certificate, accounting, and full curve are described in
[`docs/graph-organized-svd.md`](docs/graph-organized-svd.md).

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
| Gemma phase-aware source-mode GFA | No deployed reduction; phase-aware low graph bands `0:8` / `0:16` contain `48.09%` / `60.32%` of local response energy versus `9.24%` / `21.40%` for the phase-blind magnitude control | Local phase-aware graph ranks are `45 / 52 / 62` versus `57 / 61 / 63` for the control; local-to-`1σ` low-8 projector overlap is `0.9995` | Same-artifact pooled source-response diagnostic only; no directed transfer, held-out prediction, executor, compression, or speed claim |
| Gemma fit-only graph-wavelet map | The analytic rank-45 plan payload is `283,456` float64 scalars (`29.38%` below the `401,408`-scalar full-rank plan; `287,936` with wavelet metadata) but misses fidelity; rank 52 is `326,912` (`18.56%`; `331,392` standalone) and misses the 20% plan-payload gate | Rank-52 fit-disjoint development-selection error is `0.15090`, cosine `0.98856`, and mean effective support `1.64` modes versus `15.64` for fit-energy GFA; GFA error is `0.12649` and SVD is `0.04091` | At rank 52, signed topology beats magnitude, native, permuted, and all eight random controls; no rank passes fidelity, topology, plan-payload, GFA, SVD, and compute gates, and full rank ties the random bases; localized mapping evidence only |
| Gemma graph-wavelet supermode mutation | Rank 45 folds 15 genuine `2→1` merges plus four singleton prunes into the same `283,456`-coefficient plan (`29.38%` below rank 64), with no separate runtime merge transform | Fit-disjoint selection error is `0.18422`, cosine `0.98289`, and squared-error recovery over equal-rank GOMP pruning is `18.70%`; recovery is positive at both selection origins | Dense loadings beat the same one-hot actions and all four permuted-topology controls, but fit-energy GFA (`0.17258`) and SVD (`0.05055`) remain better; open-development structural nominee only, with compute, NLL, and model-compression gates closed |
| Gemma grouped graph-wavelet local SVD | Every rank-45 arm has the same `283,456` coefficients and `2,268,184` prepared bytes (`29.38%` below rank 64); graph partitions and local mixers are folded out of the runtime | Signed local-SVD errors are `0.13256 / 0.16044 / 0.17904` for max block widths `16 / 8 / 4`; the controlled width-8 arm has cosine `0.98705`, beats its one-hot control and all four permuted-topology controls, and recovers `24.15%` of pair-supermode squared error | Multiway local response synthesis works substantially better than pair merging; cluster GFA loses at every matched setting and global SVD (`0.05055`) remains the ceiling. Opened fixed-reference development only; compute, NLL, and model-compression gates remain closed |
| Gemma signed-g8 graph-wavelet confirmation | Frozen rank 45 stores `283,456` coefficients and `2,268,184` prepared bytes (`29.38%` below rank 64); no whole-model or speed claim | On fresh prompt-free origins, error/cosine is `0.16059 / 0.98702`, better than all 63 random partitions and signed GFA (`0.17218`), with `p=0.015625` and `11.73%` median-null SSE recovery; the `6/8` group-win result misses the frozen `7/8` gate. On 16 reused Calibration-A prompts, incremental factorized-model shadow fidelity fails: `ΔNLL/token +2.72583`, KL `3.01776`, top-1 `40.49%`, target-modal error `5.5104` | The graph partition is meaningful in first-order response space, but the fixed-reference linear carrier does not reproduce real token-conditioned states. Calibration-B and held-out splits remain sealed; candidate serving and compression remain unauthorized |
| Gemma rank-45 three-basis A shadow | Signed local SVD, signed GFA, and global SVD are exactly size-matched at `283,456` coefficients and `2,268,184` prepared bytes; this diagnostic makes no deployment-saving claim | All three fail ordinary and affected gates (`000`). Signed GFA has the lowest all-token delta NLL/KL (`+2.51838 / 2.80428`) and affected delta NLL/KL (`+2.95037 / 3.29210`), while global SVD has slightly higher top-1 (`42.43% / 32.66%`). The gains over local SVD are small, and global SVD does not axiswise dominate the graph arms | Corrected V2 A-fit-only classification is `no_rank45_basis_viable_attribution_inconclusive`: basis choice alone does not repair the executor, but rank-45 capacity cannot yet be separated from the shared fixed-reference carrier. Factorized-refit source only; no held-out, serving, compression, compute, or speed claim |
| Gemma rank-64/X4 A-only ladder | Rank 64 stores `401,408` coefficients and prepares `3,211,800` bytes, `41.61% / 41.60%` more than rank 45; it is a capacity control, not compression | Rank 64 worsens the rank-45 global arm (`+2.76865` NLL, `3.05804` KL, `41.03%` top-1). The true target-64 projection improves to `+2.00137 / 2.35277 / 46.51%`, but exact native normalized-X4 on the clamped carrier still reaches only `+1.95224 / 2.31611 / 45.01%`; all ordinary and affected gates fail (`000`) | `exact_x4_continuation_invalid` means the normalized-MLP-input intervention is not a complete residual-state boundary, so upstream capacity, projection, and generator attribution remains invalid. The subsequent complete-H4 audit resolves that boundary ambiguity; Calibration-B and held-out splits remain sealed |
| Gemma complete-H4 A-only identity audit | Diagnostic only: one model and tokenizer, six authenticated forwards per prompt, 96 forwards total; no deployment reduction | The corrected rank-64 and partial exact-X4 replays match V2 exactly. Injecting native `layer.4.output` recovers every full logit tensor bitwise across all 16 prompts: delta NLL `0`, KL `0`, top-1 `100%` in both ordinary and affected views (`11`) | `complete_h4_identity_validated`: H4 is a valid complete-state attribution boundary and the prior continuation error is in layer 4 or earlier. The incomplete H4 differs on 819 valid rows, including 17 beyond the graph's finite-lag target mask on 4 prompts. Learned-H4 reconstruction is next; no serving, compression, compute, latency, or speed claim |
| Gemma complete-H4 rank-64 A-only projection | Rank 64 stores `40,960` basis coefficients (`163,840` bytes at float32) and costs `81,920` projection MACs per support row; the diagnostic used 144 forwards and 16 backwards and makes no model-reduction claim | It retains `99.21%` of row-weighted correction energy, but full/core/tail NRMSE is `0.09094 / 0.09092 / 0.30359`. Ordinary fidelity is `+0.05531` delta NLL, `0.08040` KL, and `85.39%` top-1; the complete-H4-support view is `+0.06413 / 0.09322 / 83.06%` | `11000 / rank64_h4_projection_insufficient`: identity and exact causal-support integrity pass, while geometry and both behavioral ledgers fail. This recovers most of the partial-X4 error, but rank 64—especially its causal tail—is not sufficient, so the learned generator and all held-out panels remain closed |
| Gemma complete-H4 two-basis rank ladder | Each rank-192 arm stores `122,880` basis coefficients (`491,520` bytes at float32) and costs `245,760` projection MACs per support row; the complete eight-arm diagnostic used 272 forwards, 16 backwards, and `1,006,387,200` logical projection MACs, with no deployment-saving claim | The tilted rank-192 arm reaches full/core/tail NRMSE `0.00992 / 0.00991 / 0.08779`; pooled ordinary fidelity is `+0.00506` delta NLL, `0.00360` KL, and `96.35%` top-1, while complete-support fidelity is `+0.00587 / 0.00418 / 95.77%`. The unweighted arm is numerically indistinguishable | No arm passes the strict all-strata gate (`11100000` at every rank): the 17-row causal tail misses pooled and two family NRMSE gates, shell/sundial miss family top-1 gates, and one obsidian tail token misses the family NLL gate. Fisher tilting does not improve rank efficiency; learned generators, held-out panels, serving, and compression remain closed |
| Gemma complete-H4 D320 + token-Fisher tail ladder | The first fully passing tested arm retains `320 + 256 = 576` of the `640` H4 directions (`90%` of this diagnostic span). Its ideal two-sided projection work is `737,280` MACs per support row versus `819,200` for the exact 640-span sentinel; these are capacity measurements, not serving parameters or measured latency | K256 improves the D320 family-macro absolute NLL gap by `97.61%`. Ordinary delta-NLL/KL/top-1 is `+0.00056 / 0.00364 / 96.89%`; complete-support is `+0.00065 / 0.00422 / 96.39%`; causal-tail is `-0.02168 / 0.00440 / 100%`. Full/core/tail NRMSE is `0.00362 / 0.00362 / 0.00688`; all established aggregate, prompt-robustness, and geometry gates pass | `adaptive_same_a_smallest_tail_rank_256_cleared_established_gates`: rank 64 and rank 320 reproduce the parent run exactly, all eight families improve, and K320 is bitwise native. The rank grid was chosen after seeing the first run, D320 itself contains all-A information, and held native tails instantiate every finite correction. This is authenticated hypothesis evidence only—not fresh confirmation, a deployable provider, or a compression claim. Endpoint linear prediction improves only about `29%` even at full rank, motivating the teacher-KL signed-joint/path-integrated rung |
| Gemma candidate-conditioned K64 gains V3–V10 | No serving artifact, parameter reduction, or deployable MAC saving was produced. V10 used exactly `224` model forwards / `1,039` backwards and accounted for `7,756` candidate support-row executions; its D320/K64 projection MAC counts are analysis-only | V3 abstained; V4's selected coarse steps and V5's `1/64` microsteps did not transfer. V6 found stable state signal but lost to the scalar control. V7's joint global-plus-state field analytically beat that scalar in `7/8` folds, while V8 finite execution reversed it. V9 localized the reversal by tracing the actual cast-once scalar→joint H4 path. V10 replayed the identical 16 endpoints and 64 GL4 nodes with float64 teacher-KL arithmetic: path transport is only `0.0622%` of finite-D64 RMS, while D64−D32 endpoint precision is `2.004%` | Float64 objective arithmetic improves strict closure only from `8.90%` to `8.67%`; cosine remains strong at `0.9963`, but only `4/8` families clear the frozen `10%` family gate. Higher-order quadrature is therefore not earned, and objective precision is a real secondary signal rather than a complete explanation. The active no-fit rung audits the post-H4 live suffix/discrete cast path before any scalar-endpoint refit. This is not serving or compression. See the [V3–V10 record](docs/progressive-compilation.md#candidate-conditioned-k64-gain-refits-v3-v4-and-v5) |
| Gemma post-H4 adjoint localization V11–V13 | Diagnostic only: V13 adds 64 suffix forwards, 832 segment calls, 436 chunked pullbacks, and 113,602,560 support contraction products; no serving resources are claimed | Native reverse-mode VJP reproduces V10 `P_v10` at `4.68e-16` relative RMSE, but differs from the exact same-suffix forward JVP by `2.5645e-4` integrated and `2.5635e-4` nodewise. All eight families miss the frozen `1e-4` gate; integrity and the fixed telescope pass | The original Fisher gradient source is vindicated. The residual is a live-float32 forward/reverse numerical-boundary envelope, far smaller than the open `8.67%` finite closure miss. V13 is an authenticated negative diagnostic, not compression; reverse VJP can be the Fisher-local derivative in a newly preregistered operational rung, while finite fidelity still requires a separate held-out correction |
| Gemma autonomous complete-H4 residual V14 | Incremental provider-only size/work is `77,888 / 118,784` params/MACs per token at K64, `360,704 / 524,288` at K256, and `471,360 / 675,840` at K320. The A16 screen used 128 full-model forwards and 16 VJP traversals; retained Gemma and bridge/suffix costs are excluded | Outer-LOFO ordinary ΔNLL/KL/top-1 improves monotonically from the base graph's `+2.56889 / 2.88209 / 42.43%` to K320 Fisher's `+1.01993 / 1.10038 / 63.69%`. K320 reduces the two losses by `60.30% / 61.82%`; matched-rank Fisher weighting adds `9.53% / 8.86%` and `2.26` top-1 points over hidden-only K256 | The serving ABI is genuinely source-free at H4, but every K64–K320 recipe remains far outside the `0.05 / 0.05 / 95%` gates. No provider was selected, no sidecar was emitted, and guard/B stayed closed. This is one-boundary full-vocabulary/full-suffix shadow evidence—not whole-model compilation, layer deletion, compression, or speed. A single K640 LOFO capacity ceiling is the bounded next discriminator |
| Gemma autonomous K640 capacity sentinel V15 | Full-span provider-only size/work is `1,147,520` scalars / `1,556,480` logical MACs per token (`2.43× / 2.30×` K320); retained Gemma and bridge/suffix costs are excluded | Outer-LOFO ordinary ΔNLL/KL/top-1 reaches `+0.81825 / 0.92572 / 66.70%`, improving K320 by `19.77% / 15.87% / 3.01` points. Loss improves in `8/8` families and top-1 in `7/8` | Full H4 span helps broadly but remains far outside every absolute gate. PCA rank truncation matters but cannot explain the remaining miss; no provider, guard, or B opening, and no compression or speed claim |
| Gemma Fisher-square conditional residual V16 | Each K256+rank-16 child uses `377,604` scalars / `541,184` logical MACs per token, exactly matched between Fisher and PCA and only `16,900 / 16,896` above the parent. The screen used 112 forwards, 16 VJPs, and 24 fold fits | Parent/Fisher/PCA ordinary ΔNLL is `1.16923 / 1.16930 / 1.16919`; KL is `1.25289 / 1.25288 / 1.25285`; all three top-1 values are `59.72%`. Both routers pass fit and held-family 2-D geometry gates, but Fisher wins family loss in only `1/8` | The inherited `0.25` pointwise operator certificate projects Fisher fits to `0.000396–0.000628` of their unconstrained amplitude, leaving only `0.0186–0.0290%` in-fit RMSE gain. The tested square is rejected without blaming coordinate collapse; constrained or scale-calibrated residual fitting is the next hypothesis, not deployment or compression |
| Gemma bounded Fisher-pedal conditional residual V18 | Every matched child uses `377,608` scalars / `541,187` logical matrix MACs per token, only four scalars and three MACs above V16. The screen used 144 forwards, 16 VJPs, 40 outer fits, 32 held diagnostics, and 80 causal checks | Parent/unit-or-constant/Fisher-conditional/PCA-conditional ordinary ΔNLL is `1.16923 / 1.25292 / 1.24127 / 1.19812`. Fisher conditional varies on all fit and held folds and improves its constant in `8/8` families, but the macro gain is only `0.943%`; it remains `6.10%` worse than the parent and loses to PCA on absolute ΔNLL | The rowwise `0.25` amplitude certificate works without global suppression, but analytic pedal targets mostly exceed one and the learned pedal stays near full-on. Classification is `fisher_pedal_pointwise_trust_insufficient`; no provider, serving, compression, or speed claim. V17 is retained only as an invalid floating-aggregation receipt; V18 is authoritative |
| Gemma finite teacher-KL joint direction/pedal V19 | Every matched child remains `377,608` scalars / `541,187` provider matrix MACs per token. The exact screen used 1,280 forwards, 912 suffix backwards, 896 local contractions, 40 conceptual outer fits, and 96 causal checks | All 16 Fisher/PCA fits selected checkpoint 0. Fisher's mean exact-KL curve was `.00244384 → .00781108 → .00344203 → .00451110 → .00376498`; its selected half-strength initializer improves V18 start macro absolute ΔNLL by `3.306%` but remains `2.596%` worse than the parent and identical to the intercept. PCA checkpoint 0 is `0.573%` better than parent on macro absolute ΔNLL but `0.533%` worse on KL | `finite_joint_pedal_outer_fidelity_insufficient`: finite rollback worked, but no update descended, no direction/beta change was selected, and every pedal stayed exactly `0.5`. No full refit, sidecar, serving, compression, or speed claim. A nested finite microstep ladder is the next justified optimizer diagnostic |
| Gemma finite Fisher microstep V20a | Fit-only preflight: 2,622 full-model forwards, 128 full-suffix backwards, 112 local-head contractions, 168 positive candidates, and nine signed mirrors. Candidate and provider sidecar remain null | All `8/8` capability-excluded folds passed and selected `alpha = 0.1`; direction-only won `5/8`, joint won `3/8`, and pedal-only won `0/8`. Family-equal fit KL improved `0.00244384 → 0.00234591` (`4.0071%`), with every fold improving `3.2031–4.4064%` and every negative mirror worsening | `finite_microstep_preflight_passed_for_nested_validation`: V19's direction is useful but its full update overshot. This authorizes nested family-disjoint V20b only; no held-fidelity, fresh-guard, Calibration-B, serving, compression, speed, parameter, or FLOP claim |
| Gemma fit-only signed-GFA rate curve | Rank 45 stores `283,456` coefficients versus `393,216` dense fit knots (`27.91%` fewer); cached-core linear MACs are `20.67%` lower, but the current uncached interpolation path performs `2.20×` the dense kernel-application multiplies | Frozen-origin selection error `0.1900`, worst cosine `0.9810`; the same-budget SVD error is `0.0506` and every signed-GFA cutoff loses to SVD | The signed graph beats magnitude, native-prefix, permuted, and eight random controls, but does not pass the controlled compression gate; organization/fidelity evidence only |
| Gemma graph-organized global SVD | Rank-45 deployment-form edge state is `279,744` versus `393,216` dense coefficients (`28.86%` fewer); all-on cached-core MACs are `72.51%` of dense, and 95%-bound routing lowers this to `70.61%` | On nonzero C2 selection directions, all-on measured-response error is `0.03179`; signed 95%-bound routing is `0.03206` at mean active rank `43.78` | Executable hybrid and conditional rate curve; opened synthetic development data, router cost excluded, no NLL, latency, whole-block, or whole-model claim |
| Gemma graph-organized one-shot shadow | No deployment saving claimed; the candidate runtime needs three source-model passes and the full qualification observation needs two additional oracle passes | On one Calibration-A development prompt, all-on modal error is `4.8208` with cosine `0.5404` and `ΔNLL/token +3.0853`; the true rank-64 projection oracle still has `0.9741` full-width error, and exact X4 injection still has `ΔNLL/token +2.0121` | Strong fail-closed shadow harness; current edge rejected for target-subspace capacity and residual-carrier incompleteness, with deployment and routing unauthorized |
| Gemma conditional spectral modal-delta executor | `39,936` edge coefficients versus `786,432` for a matched dense two-branch family (`94.92%` fewer); provider and model excluded | Fresh origin-20 local cosine `0.9819`; diagonal correction reduces finite error `0.2278 → 0.2006` | Prompt-free fixed-reference interior interpolation only; no-refit assessment |
| Gemma mixed-mode chord assessment | No deployed reduction; frozen candidate unchanged | Fresh origin-28 error `0.1863`, cosine `0.9834`; cross nonadditivity `11.27%`; interaction-oracle gain `23.10%` | Diagonal-only correction materially falsified; compact bilinear branch nominated |
| Gemma bilinear modal-generator executor | Bilinear branch stores `6,880` coefficients versus `172,032` dense (`96.00%` fewer); all three edge branches store `46,816` versus `958,464` matched dense (`95.12%` fewer) | Fresh origin-20 error `0.2090 → 0.1694` (`18.96%` reduction), cosine `0.9871`; recovers `94.10%` of \(C_{11}\) oracle headroom | Positive no-refit mixed-mode edge transport; fixed-reference and known-pair scope only |
| Gemma prompt-blind reference provider V2/V3 | Rank 8 stores `910` scalars versus `15,046` for the full-width provider (`93.95%` fewer); provider-only ideal MAC savings are sequence-dependent | Fresh-V3 ordinary error `0.0677`, cosine `0.9977`, p90 `0.2914`; all ordinary fidelity/structure gates passed | Radial and intended-null contrast recovery failed; signed sensitivity was underpowered, so the formal V3 outcome is panel-inconclusive |
| Gemma C2 contrast-packed provider development | Ranks `8/16/32` store `1,980/4,276/8,676` scalars (`86.84%/71.58%/42.34%` below the prior dense-64 component); canonical rank-8/rank-16 MACs are `52.35%/29.85%` below rank 32 | Every rank passed ordinary fidelity and `24/24` exact-null pairs; radial passes were `12/16`, `13/16`, `7/16`, while signed passes were `0/7` at every rank | Held-out development selection only; no candidate passed, V4 remains unopened |
| Gemma rank-16 objective-balance diagnostic | Same `4,276`-scalar candidate form; no new resource or deployment claim | Unit-RMS treatments passed `12/12` ordinary, `24/24` null, and `16/16` radial fit checks, but only `2–3/8` signed checks | Fit-only diagnostic; global loss scale is not the sole blocker and C3 remains unopened |
| Gemma rank-64 capacity control | `19,012` stored scalars and `3,190,528` canonical MACs versus rank 16's `4,276` and `1,315,072`; no reduction claim | Descriptively `12/12` ordinary, `24/24` null, `16/16` radial, and `3/8` signed; ordinary error `0.00672074`, cosine `0.99997745` | Invalid comparison: initial pointwise share missed the frozen balance gate, so no capacity conclusion, replication, width ladder, or C3 is authorized |
| Gemma function-preserving width control | Rank 64 uses `19,012` scalars / `3,190,528` canonical MACs versus rank 16's `4,276` / `1,315,072` (`4.446×` storage and `2.426×` MACs); no reduction claim | Valid matched start: both passed ordinary, null, and radial gates but only the same `3/8` signed identities; rank 64 improved ordinary error `0.00769406 → 0.00652994` while signed macro error changed `0.380298 → 0.386891` | Outer width alone is insufficient under the matched fit budget; expert/core control authorized, with no replication, width ladder, C3, or compression claim |
| Gemma function-preserving expert-rank control | Expert rank 64 uses `31,492` scalars / `5,555,776` canonical MACs versus expert rank 16's `19,012` / `3,190,528` (`65.64%` more storage and `74.13%` more MACs); no reduction claim | Valid matched start: both passed ordinary, null, and radial gates but only the same `3/8` signed identities; expert rank 64 improved ordinary error `0.00652994 → 0.00621631` and signed macro error `0.386891 → 0.383903` | Inner expert rank alone is insufficient under the matched fit budget; expert-count control authorized, with no replication, descending rank ladder, C3, compression, or speed claim |
| Gemma function-preserving expert-count control | Four experts use `48,166` scalars / `8,985,792` canonical MACs versus two experts' `31,492` / `5,555,776` (`52.95%` more storage and `61.74%` more MACs); no reduction claim | Valid matched start: both passed ordinary, null, and radial gates but only the same `3/8` signed identities; four experts improved ordinary error `0.00621631 → 0.00555688`, signed macro error `0.383903 → 0.362307`, and the weighted objective `0.164106 → 0.144633` | Four experts are insufficient under the matched fit budget; only a separately preregistered eight-expert full-count oracle is authorized, with no replication, descending count ladder, C3, compression, or speed claim |

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

fisher-graph-gemma-l3-l4-phase-graph-spectral-dev describe
fisher-graph-gemma-l3-l4-phase-graph-spectral-dev analyze

fisher-graph-gemma-l3-l4-graph-wavelet-dev describe
fisher-graph-gemma-l3-l4-graph-wavelet-dev analyze

fisher-graph-gemma-l3-l4-graph-wavelet-supermode-dev describe
fisher-graph-gemma-l3-l4-graph-wavelet-supermode-dev analyze

fisher-graph-gemma-l3-l4-graph-wavelet-grouped-dev describe
fisher-graph-gemma-l3-l4-graph-wavelet-grouped-dev analyze

fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-freeze
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-null-bundle
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-confirm
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-shadow-dev
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-shadow-bases-dev
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-rank64-oracle-dev
fisher-graph-gemma-l3-l4-complete-h4-identity-a-dev
fisher-graph-gemma-l3-l4-complete-h4-projection-a-dev
fisher-graph-gemma-l3-l4-complete-h4-basis-rank-ladder-a-dev
fisher-graph-gemma-l3-l4-complete-h4-autonomous-residual-v14-a-dev
fisher-graph-gemma-l3-l4-complete-h4-autonomous-k640-v15-a-dev
fisher-graph-gemma-l3-l4-complete-h4-fisher-square-v16-a-dev
fisher-graph-gemma-l3-l4-complete-h4-fisher-pedal-v18-a-dev
fisher-graph-gemma-l3-l4-complete-h4-finite-joint-pedal-v19-a-dev

fisher-graph-gemma-l3-l4-graph-organized-svd-dev

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
- [Fit-only graph-wavelet mapping](docs/graph-wavelet-mapping.md)
- [Graph-wavelet supermode mutation](docs/graph-wavelet-supermode-mutation.md)
- [Grouped graph-wavelet basis comparison](docs/graph-wavelet-grouped-comparison.md)
- [Signed-g8 graph-wavelet confirmation](docs/graph-wavelet-signed-g8-confirmation.md)
- [Graph-organized global SVD](docs/graph-organized-svd.md)

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
