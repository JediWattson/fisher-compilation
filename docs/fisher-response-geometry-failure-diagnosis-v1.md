# Fisher response geometry failure diagnosis v1

This is a synthetic diagnosis of the already examined
[`fisher_response_geometry_probe_v1`](fisher-response-geometry-probe-v1.md)
fixture. It asks whether the observed response-chart errors arise from chart
assignment, the available frozen predictors, limited support, or projection
and intercept limitations. It does not change the chart design or implement
a proposed remedy. These measurements are additional analysis of the same
fixture, not independent confirmation of a hypothesis.

## Preserved baseline and execution boundary

All original source, configuration, fitted parameters and output artifacts
remain unchanged. The original report is
[`artifacts/research/fisher-response-geometry-probe-v1-synthetic/report.json`](../artifacts/research/fisher-response-geometry-probe-v1-synthetic/report.json).
New diagnosis output goes to a fresh directory:

```bash
python src/fisher_response_geometry_failure_diagnosis_v1.py synthetic --output /tmp/fisher-response-geometry-failure-diagnosis-v1-001
```

The new standalone runner is
[`fisher_response_geometry_failure_diagnosis_v1.py`](../src/fisher_response_geometry_failure_diagnosis_v1.py).
Its `inspect` command reads metadata only. Its `model-facing` command refuses
execution and preserves
`development_complete_independent_data_unavailable`. Neither this runner nor
the replay/diagnosis path may load a target model, tokenize, open a protected
corpus, run live attention, consume an independent panel, or change the
[existing P0 restrictions](fisher-compact-message-completion-probe-v2.md#current-p0-contract).

[`fisher_response_geometry_replay.py`](../src/fisher_graph/fisher_response_geometry_replay.py)
authenticates the saved baseline and reconstructs only missing generated
observations through the pinned original random-number sequence. It reuses
the saved split, metric, bases, coefficients, anchors, discovery centers,
support radii, router centers and feature scales. Reconstructing omitted
synthetic observations is not permission to refit a basis, rerun chart
discovery, choose new anchors, fit coefficients, retrain routing, alter
support, add examples, or increase capacity. Existing stored parameters remain
authoritative. Replay discloses every reconstructed quantity, authenticates
the relevant saved identities, and reports any reproduction discrepancy.
It verifies all eight pinned original source identities before importing the
original numerical kernel. Recovered memberships use nearest frozen centers;
response membership requires exact signatures for classification only, with
no center fitting. Training occupancy, anchors, support and saved evaluation
metrics authenticate the replay. Projection applies stored `D_G` directly,
without a new metric solve or basis fit.

The original files have these pinned byte identities:

| Original artifact | SHA-256 |
| --- | --- |
| `report.json` | `b600bc66c260ca6722fc3b7b66340bda9093c268dda4ebbde058e36dda23d5a6` |
| `frozen_config.json` | `b06bbf156555f178b823a7cff2c89642521dda37d774dd59fdce796b16ade3d3` |
| `artifact_manifest.json` | `ac1697472a775b3b7f0185db3916ece918a31411e7ad4f43d8aa4e55ebbb805c` |

These hashes are source/artifact identity checks, not independent-data
authorization. The separate execution record states whether verification
actually passed.

The required baseline comparison retains the original aggregate Fisher-relative
absolute-residual-update error over **all 48 evaluation rows**, including
unsupported extrapolations:

| Frozen chart family | Runtime router | Direct discovery assignment |
| --- | ---: | ---: |
| Response | `0.89266` | `0.88431` |
| Activation | `0.13629` | `0.13629` |

These are the supplied rounded reproduction targets, not newly measured
values. Exact saved numbers and the declared replay tolerance govern the
authentication check. Do not substitute a supported-only average or change
the denominator to make these values reproduce. The execution record below
will contain the observed values, identities and discrepancies, if any.

## Three targets and the actual frozen predictors

The original normalization seam remains
`Phi(m)=N(W_O m)`, with independent `h` and `m`, `K=0`, and `R(m)=D_m Phi(m)`.
Use the same fixed output metric `F_*`, coordinate map `D_G`, and bases
`G,U,Q` as the baseline; write `P_G=G D_G`. The stored chart predictor is:

```text
Phi_hat_C(h,m) = G [beta_C + B_C U^T(h-h_C) + A_C Q^T(m-m_C)]
```

The normalization banks have stored `B_C=0`. Validate that property instead
of replacing a nonzero stored operator silently. Teacher offsets or per-state
teacher derivatives must not replace stored `beta_C` or `A_C` in a prediction.

For each example `i`, use its existing v1 unit message probe `q_i`. There is
one stored/reconstructed probe per evaluation row in this fixture. The three
target-specific comparisons are:

| Target | Exact target | Frozen chart prediction |
| --- | --- | --- |
| Derivative action | `R(m_i) q_i` | `G A_C Q^T q_i` |
| Finite response, separately for radius `rho` | `Phi(m_i+rho q_i)-Phi(m_i)` | `Phi_hat_C(h_i,m_i+rho q_i)-Phi_hat_C(h_i,m_i)` |
| Absolute residual update | `Phi(m_i)` | `Phi_hat_C(h_i,m_i)` |

Use the unchanged radii `0.001`, `0.01`, and `0.1`. Finite responses keep
both the chart and `h_i` fixed between endpoints. Their affine predictor
simplifies to `rho G A_C Q^T q_i`; this equality is an implementation check,
not a license to substitute an exact derivative.

[`fisher_response_geometry_failure_diagnosis.py`](../src/fisher_graph/fisher_response_geometry_failure_diagnosis.py)
compares three assignment methods within each of the two families:

- **Runtime:** the original activation-feature router.
- **Direct discovery:** the original frozen discovery assignment; response
  discovery uses privileged derivative signatures and is not a runtime router.
- **Best-supported-prediction oracle:** choose the eligible frozen predictor
  with minimum per-example squared Fisher error for this particular target.

For multiple probes, the oracle objective is the fixed arithmetic mean of
their squared Fisher errors within the example. Eligibility must hold for
every required probe/endpoint. Select one chart per example and target, not
one chart per coordinate or probe. With the fixture's single probe, the mean
is its one squared error. Each finite radius has a separate target oracle.
Different targets can choose different charts. Record ties deterministically
under the declared rule.

Compute Euclidean errors using exactly these Fisher-selected assignments.
There is no separately optimized Euclidean oracle in the primary tables.
Every supported oracle uses evaluation targets and is privileged diagnostic
information, never a learned router or a deployable prediction mechanism.

## Eligibility, abstention and common comparison rows

A derivative or absolute-update chart is eligible when its original support
rule accepts the base example. For a finite response the **same chart** must
accept both base and perturbed endpoints at the relevant radius. Support is
checked using the full original state/message coordinates, including
directions omitted by the compact bases.

For every family, assignment method and target/radius, distinguish:

1. No chart in the family supports the example.
2. Some chart supports it, but the runtime or direct-discovery assignment
   selects an unsupported chart.
3. The selected chart supports it.

The supported oracle abstains in case 1. Its assignment and prediction/error
are missing, not zero, and no nearest-chart fallback fills the gap. Coverage
and unsupported-selection counts remain visible even when a comparison uses
only eligible rows.

Use these explicitly identified row sets separately for each target/radius:

| Row set | Definition and interpretation |
| --- | --- |
| `all_rows` | All 48 original rows. Runtime/direct results preserve unrestricted extrapolation and the original baseline aggregation. An abstaining oracle is not inserted into this table as though it predicted every row. |
| `common_selected_support` | Intersection across both families: each family's runtime, direct and oracle assignments are all eligible. All six methods are compared on this identical mask. |
| `common_any_family_support` | Each family has at least one eligible chart. This is an intersection across families, not a union. Both supported oracles can be compared on this common mask; runtime/direct predictions remain explicitly unchecked wherever their selected chart is unsupported. |

For every table, provide row identities, the Boolean mask, row count, probe
count, coverage and its Euclidean/Fisher normalization denominators. Runtime,
discovery, oracle and reference rows compared within a table use identical
examples and denominators, including cross-family comparisons. A reduced
mask is not an improved all-row result. A zero-row mask yields unavailable
errors, not zeros or an undocumented change of population.

For target vectors `y_ij`, predictions `y_hat_ij`, and mask `S`, define the
aggregate squared sums over all selected rows and their probes. Every row has
the same fixed number of probes; all current targets use one probe per row.
Only the per-example oracle selection objective uses a mean over probes.

```text
numerator_F   = sum_{i in S} sum_j [(y_hat_ij-y_ij)^T F_* (y_hat_ij-y_ij)]
denominator_F = sum_{i in S} sum_j [y_ij^T F_* y_ij]
relative_F    = sqrt(numerator_F / max(denominator_F, 1e-30))
```

Euclidean aggregation replaces `F_*` with the identity and retains the same
assignments and mask. The `1e-30` denominator floor is inherited from the
original aggregation; record the raw target energy and the effective floored
denominator separately. With one probe per row this is precisely the original
sum-based relative-error aggregation. Empty masks still produce missing
errors rather than zero. For fixed probe counts, averaging the probe terms
would leave the relative ratio unchanged but would alter the reported raw
denominators; this report uses the sums above.

## Exact references and the intercept diagnosis

The exact target itself is the unprojected accuracy reference; its error is
zero by construction. The **unchanged exact projected-normalization
reference** for the absolute update is `P_G Phi(m_i)`. Apply the same fixed
projection to each other target:

```text
projected derivative reference = P_G R(m_i) q_i
projected finite reference = P_G [Phi(m_i+rho q_i)-Phi(m_i)]
```

These references use the full original probe, not its `Q Q^T` truncation.
They isolate output projection loss. The old compact per-state Taylor
reference `P_G R(m_i) Q Q^T (rho q_i)`, if retained, stays separately labeled:
it also truncates the input direction and approximates a finite response.
None of these exact references may replace a chart's actual predictor.
Report exact and projected reference errors beside the chart errors on each
common mask using its same target normalization.

Measure both intercepts explicitly:

```text
exact local tangent intercept: c(m) = Phi(m) - R(m)m
stored compact intercept: c_tilde_C = G beta_C - G A_C Q^T m_C
```

The compact intercept is calculated from stored chart parameters and checked
against the implemented predictor. It need not equal the projected exact
intercept at the anchor: input truncation and the actual stored coefficients
can change it. For the zero-state-derivative banks,
`Phi_hat_C(m)=c_tilde_C + G A_C Q^T m`. Inspect within frozen discovery groups
the disagreements in input locations, exact outputs, exact tangent
intercepts, compact-intercept-induced predictions and observed errors.
Teacher outputs/intercepts are diagnosis-only and never enter a chart
signature, router feature, fit or support rule.

The deterministic algebraic check is:

```text
R(-m) = R(m)
Phi(-m) = -Phi(m)
```

It shows that equal response signatures can in principle combine incompatible
locations and offsets. Separately inspect actual within-group example pairs
and their error association. Synthetic sign-flip checks do not add negative
examples to training or evaluation. Algebraic symmetry, pairwise association
and a causal explanation of this run's failure are three different claims.
Do not assert the third from either of the first two. Separately measured
projection, assignment and intercept errors are not assumed to add linearly.

## Reporting and stop condition

Tables identify chart family, assignment, target/radius, Fisher and Euclidean
error, evaluated row count, support coverage and unsupported selections.
Distinguish a useful predictor missed by discovery, discovery missed by the
router, inaccurate best eligible predictors, insufficient common support,
different performance across derivative/finite/absolute targets, and remaining
projection loss. Mixed or unresolved findings are allowed.

The execution record must include observed baseline reproduction, commands,
tests actually run, changed files, source/output identities, reconstruction
disclosures and unavailable measurements. Recommend at most one next
modification after inspecting the measurements. Do not implement it here.

## Execution record

Completed on 2026-09-08. The final output is separate from the original:

- [Human-readable comparisons](../artifacts/research/fisher-response-geometry-failure-diagnosis-v1/diagnosis.md).
- [Complete measurements](../artifacts/research/fisher-response-geometry-failure-diagnosis-v1/diagnosis.json), including every row mask, denominator, eligibility matrix and assignment.
- [Regenerated original inputs](../artifacts/research/fisher-response-geometry-failure-diagnosis-v1/replay_inputs.json).
- [Artifact and source identities](../artifacts/research/fisher-response-geometry-failure-diagnosis-v1/artifact_manifest.json), including all SHA-256 hashes, byte counts, command arguments and runtime versions.

### Reproduction and preservation

All **172 numerical comparisons reproduced**, with maximum absolute difference
`1.7763568394002505e-15`, below the declared `1e-12` absolute and `1e-10`
relative tolerances. The four requested all-row Fisher errors reproduce as:

| Family | Runtime | Direct discovery | Rows |
| --- | ---: | ---: | ---: |
| Response | 0.8926561273956386 | 0.8843130519102650 | 48 |
| Activation | 0.1362931496408758 | 0.1362931496408758 | 48 |

Their common squared-target denominators are `3434.9557057572297` in the
Fisher metric and `422.3896176882639` in Euclidean coordinates. The original
three artifact files and eight authenticated sources retain their pinned
byte identities. Preservation tests also check original file modification
times. No original source, report, fitted parameter, support rule, feature,
router or configuration was edited.

The original report stored fitted tensors but omitted generated input rows
and probe directions. Only those missing observations were reconstructed
from the pinned random draw schedule. The saved parameters were loaded
directly; no clustering, basis solve, chart fitting or router training ran.
Reconstructed rows, family identities and probes are now recorded explicitly
in `replay_inputs.json`. This is a replay of the examined fixture.

### Measured outcome

All 48 rows have at least one eligible chart in each family, at every tested
target/radius. The following is the shared 48-row comparison. Each cell is
aggregate Fisher-relative error; the full report also supplies the Euclidean
error under the same Fisher-selected assignments.

| Target | Radius | Response runtime | Response discovery | Response supported oracle | Activation, all three assignments | Exact projected reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Absolute update | — | 0.892656 | 0.884313 | 0.887349 | 0.136293 | 0.029547 |
| Derivative action | — | 1.011945 | 0.676147 | 0.876441 | 0.945532 | 0.194811 |
| Finite response | 0.001 | 1.011858 | 0.676219 | 0.876368 | 0.945529 | 0.194831 |
| Finite response | 0.01 | 1.011075 | 0.676887 | 0.875718 | 0.945495 | 0.195009 |
| Finite response | 0.1 | 1.003203 | 0.685281 | 0.869557 | 0.945324 | 0.196451 |

The unprojected exact analytic reference has zero error by construction. The
projected references above apply `P_G` to each full exact target without
truncating the input to `Q`. Projection accounts for a nonzero part of the
remaining discrepancy; these relative error norms are not additive.

The support categories below are identical across the five target/radius
panels. A and B remain separate; neither is silently scored as oracle zero.

| Family | Assignment | A: no eligible chart | B: eligible chart exists, selected chart unsupported | C: selected chart supported |
| --- | --- | ---: | ---: | ---: |
| Response | Runtime | 0 | 10 | 38 |
| Response | Direct discovery | 0 | 3 | 45 |
| Response | Supported oracle | 0 | 0 | 48 |
| Activation | Runtime | 0 | 0 | 48 |
| Activation | Direct discovery | 0 | 0 | 48 |
| Activation | Supported oracle | 0 | 0 | 48 |

The direct response assignment can outperform the supported oracle because
it extrapolates outside the oracle's eligible set on three rows. On the
stricter common mask where all six selections are supported, 38 rows remain.
Response's best eligible absolute-update error is `0.986956`, versus
activation's `0.067475`. For derivative action these become `0.885274` and
`0.744703`, respectively. Thus choosing among the frozen supported response
predictors does not rescue their absolute-update failure. Routing and support
selection also affect derivative/finite-response scores; neither the oracle
nor derivative correctness establishes accurate shared-chart predictions.

### Offset and group evidence

Response discovery's group sizes are `83, 1, 12`; activation discovery's are
`32, 32, 32`. In response group 0, `1632/3403` training output pairs have
negative cosine, with minimum `-0.996401`. None of the activation groups has
a negative-cosine output pair. The report retains pairwise distance summaries
for inputs, outputs, exact tangent intercepts, pointwise compact tangent
intercepts and predictions from the actual frozen compact charts.

An example from response group 0, training rows 13 and 75, has derivative
signature distance `0.170667`, output cosine `-0.987737`, message distance
`6.420451` and exact-intercept distance `5.811287`. This pair was selected
descriptively as the closest derivative-signature pair among the group's
negative output-cosine pairs. It is not a held-out selection or a threshold.

Sixteen evaluation rows have negative output cosine to their direct-response
anchor. Their mean squared Fisher prediction error is `140.469595`, compared
with `13.707960` for the other 32 rows. Actual compact-intercept mismatch and
update error have Pearson correlation `0.975320`; activation also has a high
correlation, `0.964688`. These statistics share target information and do not
isolate a causal error component.

The full analytic Jacobian sign-symmetry check uses all eight standard-basis
message directions at every one of the 48 evaluation rows. Its maximum
`R(-m)-R(m)` error is zero; the odd-output check is also zero. The original
single-probe action check is retained with its coverage labeled. Rewriting
the implemented predictor using its stored compact intercept agrees within
`1.7763568394002505e-15`. The algebraic sign ambiguity and observed group
disagreements are consistent with an offset problem, but do not establish
that it uniquely caused this run's failure.

### One proposed next modification

In a separate experiment, add a training-only compact-intercept compatibility
constraint to response-chart discovery while keeping chart count and basis
ranks fixed. This modification was **not implemented or validated** here.

### Verification and changed files

The final focused suite passed **122 tests**: 47 new diagnosis/replay/CLI
tests plus 75 existing numerical, CLI, synthetic attention-generator and
capture-helper tests. It includes the same-Jacobian/different-offset fixture,
frozen-scale adversary, endpoint support, per-example multi-probe oracle,
abstention, shared denominators, parameter-only predictions and failed-replay
reporting. Guards reject target-model/tokenizer imports and protected payload
access. An initial guard incorrectly classified Torch's installed
`torch/utils/data` Python module as a dataset; its module-source exclusion was
corrected, and the full suite then passed. No protected payload was opened.

Commands executed from the repository root:

```sh
.venv/bin/python -m pytest -q -o addopts='' \
  tests/test_fisher_response_geometry_failure_diagnosis.py \
  tests/test_fisher_response_geometry_replay.py \
  tests/test_fisher_response_geometry_failure_diagnosis_cli.py \
  tests/test_fisher_response_geometry.py \
  tests/test_fisher_response_geometry_cli.py \
  tests/test_gemma3_attention_generator.py \
  tests/test_gemma3_fisher_local_chart_capture.py

.venv/bin/python src/fisher_response_geometry_failure_diagnosis_v1.py inspect \
  > /tmp/fisher-response-failure-inspect.json

.venv/bin/python src/fisher_response_geometry_failure_diagnosis_v1.py synthetic \
  --output /tmp/fisher-response-failure-diagnosis-review-20260908

.venv/bin/python src/fisher_response_geometry_failure_diagnosis_v1.py synthetic \
  --output artifacts/research/fisher-response-geometry-failure-diagnosis-v1
```

Changed files for this follow-up are all new:

- `src/fisher_response_geometry_failure_diagnosis_v1.py`
- `src/fisher_graph/fisher_response_geometry_replay.py`
- `src/fisher_graph/fisher_response_geometry_failure_diagnosis.py`
- `tests/test_fisher_response_geometry_replay.py`
- `tests/test_fisher_response_geometry_failure_diagnosis.py`
- `tests/test_fisher_response_geometry_failure_diagnosis_cli.py`
- This document and the four files in the separate final output directory.

No target model, protected corpus, tokenizer, live-attention experiment or
independent-data campaign was used. Model-facing results, a causal test of a
changed design and independent confirmation remain unavailable. P0 remains
`development_complete_independent_data_unavailable`.
