# Fisher response geometry intercept compatibility v1

This is one controlled development ablation on the synthetic fixture already
examined by the [original probe](fisher-response-geometry-probe-v1.md) and
[failure diagnosis](fisher-response-geometry-failure-diagnosis-v1.md). It tests
whether a soft penalty for incompatible compact affine intercepts improves
response-chart predictions. The outcome may be unfavorable. No second
modification, parameter sweep or larger experiment is authorized here.

## Preserved inputs and the sole change

The original probe's three artifacts and eight bound sources, plus the
diagnosis's four artifacts and seven bound sources, are immutable inputs.
Their byte identities at source checkpoint
`95f8e348764209cb284ad8cc2b44cbbab0c1d219` are authenticated before numerical
execution and again before final publication. The existing replay reconstructs
only the original generated observations and probes; it does not change the
fixed metric, geometry or frozen control banks. The saved diagnosis input
artifact and original replay reproduction provide additional checks on those
observations.

The following saved all-48-row absolute-update Fisher-relative errors remain
baseline references, with unsupported extrapolations included where applicable:

| Original family and assignment | Rounded saved error |
| --- | ---: |
| Response runtime | `0.892656` |
| Response direct discovery | `0.884313` |
| Response supported prediction oracle | `0.887349` |
| Activation, all three assignments | `0.136293` |

Exact saved values and the original reproduction tolerances (`atol=1e-12`,
`rtol=1e-10`) govern equivalence, not agreement after rounding.

The only new discovery objective is the fixed `lambda=1` signature below.
Keep the same generated examples, split, seed, probes, three charts, fixed
metric and bases, clustering algorithm/initialization/ordering/iteration
budget, training-medoid rule, anchored Taylor construction, support policy,
radius multiplier, runtime features, centroid router, targets and finite
radii. New memberships necessarily rebuild medoids, Taylor coefficients,
support radii and router centroids by the original training-only rules.
Neither the original response nor activation banks are rebuilt for the main
comparison.

The candidate must not introduce balanced occupancy, a sign classifier, a
stronger router, learned offsets, least-squares charts, extra anchors, extra
charts, higher ranks or new training examples. Necessary rebuilding under new
memberships is distinct from changing the predictor family.

## Compact-intercept discovery feature

The unchanged normalization seam is `Phi(m)=N(W_O m)`, with independent state
and message and zero state-response term. Use stored `D_G`, `G` and `Q`.
For every training row, compute its own pointwise compact Taylor quantities:

```text
phi_i  = Phi(m_i)
A_i    = D_G R(m_i) Q
beta_i = D_G phi_i
a_i    = Q^T m_i
b_i    = beta_i - A_i a_i
c_tilde_i = G b_i
```

This is not the exact full-space tangent intercept `Phi(m_i)-R(m_i)m_i`.
Input and output truncation generally make them different. It is also not the
intercept of the row's old assigned chart: borrowing that offset would carry
the old grouping into the feature being tested.

Let `x_i` be the original aligned response signature, with its original
coordinate convention, and let:

```text
H_G = G^T F_* G
z_i = H_G^(1/2) b_i
```

Supported metric rules remain unchanged. Euclidean distances in `z` equal
Fisher distances between reconstructed pointwise compact intercepts. The
response and intercept blocks use one centered training scale each, not
per-example normalization.

For either vector block `v_i`, use float64 and scale-safe norm calculations:

```text
mu = mean_i v_i
reference_scale = sqrt(mean_i ||v_i||^2)
centered_scale = sqrt(mean_i ||v_i-mu||^2)
relative_constant_tolerance = 1e-12
```

If `reference_scale` is exactly zero, drop the identically zero block. Otherwise
drop it only if `centered_scale/reference_scale <= 1e-12`. A dropped block
contributes zero and records its reason. A retained block uses the stored
mean and centered scale. There is no absolute `max(1, scale)` floor. Safely
representable global rescaling must preserve drop decisions and normalized
pairwise distances. Nonfinite values or failed numerical calculations are
errors, not evidence of a constant block.

The new signature is:

```text
signature_i = [
    (x_i-mu_x)/s_x,
    sqrt(lambda) (z_i-mu_z)/s_z
]
lambda = 1
```

Each retained block has unit mean squared centered row norm before clustering,
giving equal total block energy. This is a soft compatibility penalty; it
does not guarantee a maximum within-chart disagreement or erase magnitude
and sign by normalizing each row independently. Save the training means,
reference/centered scales, tolerance, drop decisions and reasons. Evaluation
reuses them exactly and cannot reclassify a block or recompute a scale.

For every rebuilt chart, retain the original anchored predictor and verify
its actual stored-intercept form:

```text
Phi_hat_C(h,m) = G [beta_C + A_C Q^T(m-m_C)]
c_tilde_C = G beta_C - G A_C Q^T m_C
Phi_hat_C(h,m) = c_tilde_C + G A_C Q^T m
```

Stored `B_C` remains zero. Current exact teacher outputs and derivatives
cannot replace the stored chart offset or response when making predictions.

## Rebuilding control and freeze boundary

The `lambda=0` control executes original response-only clustering with the
original seed, ordering, initialization and iteration budget. It bypasses
the new centered/block-normalized signature and then passes the memberships
through the same chart-building procedure as the candidate. It must not just
return a saved bank.

Verify training memberships, discovery centers, medoid identities, anchors,
coefficients, support radii/eligibility, router centers/assignments,
per-example predictions and aggregate metrics against the saved response
bank. A single consistent chart-label permutation may reconcile the objects;
independent relabeling per check is forbidden. If this control does not
reproduce, publish the discrepancy and stop before scoring the candidate.

The frozen configuration is
[`examples/fisher_response_geometry_intercept_compatibility_v1.json`](../examples/fisher_response_geometry_intercept_compatibility_v1.json).
It pins `lambda=1`, control `lambda=0`, tolerance `1e-12`, all original
dimensions/counts/settings and the comparison definitions. There are 96
training rows from 12 generated families and 48 evaluation rows from six
generated families; these observations are unchanged.

The runner performs this order:

1. Authenticate both saved baselines and the new source/configuration set.
2. Replay the original inputs and verify the original metrics.
3. Rebuild and verify `lambda=0`.
4. Fit the sole candidate from training inputs only.
5. Save `frozen_config.json`, `lambda_zero_control.json`, `candidate.json`,
   `block_scales.json` and `freeze_manifest.json`, binding their exact bytes,
   baseline identities and source identities.
6. Read back and authenticate the frozen files, load the serialized candidate
   without refitting, then score that candidate.
7. Publish `evaluation.json`, `evaluation.md` and `artifact_manifest.json` in
   the same fresh output directory, with identities and execution metadata.

Control evaluation against the old saved baseline is permitted before this
candidate freeze. New candidate evaluation is not. Failure artifacts preserve
discrepancies and already frozen files; no failing control or corrupt frozen
artifact can publish candidate performance. Hashes establish byte identity,
not independent data or scientific authority.

## Runtime and target-informed diagnostics

The rebuilt runtime router still uses only the original scaled observable
features `[h/h_scale, m/m_scale]` and the original centroid procedure. Its
features contain no current teacher output, derivative, intercept or target.
The completed message `m` still depends on the original attention computation;
this downstream synthetic ablation does not replace attention.

New direct-discovery assignment evaluates the new signature with exact
evaluation responses and outputs, using frozen training transforms. It is a
target-informed diagnostic, not a deployable router or evidence of cheap
routability. Exact evaluation quantities cannot update training objects,
enter runtime features, select a weight or another candidate, or replace
stored predictors. Keep target-specific supported prediction oracles separate.

Compare the frozen activation bank, frozen original response bank and new
intercept-compatible response bank under runtime, direct discovery and the
supported prediction oracle. Targets remain derivative action, absolute
update and finite responses at `0.001`, `0.01`, `0.1`, with the original probe
directions. Each oracle selects one eligible chart per example/target/radius
by mean squared Fisher error over that example's probes. Use the same
Fisher-selected assignments for Euclidean error. One chart must support both
finite endpoints; no endpoint switching or unsupported oracle fallback is
allowed.

The exact reference is the full target. Its projected counterpart is `P_G`
times that target without input-`Q` truncation. Thus the projected finite
reference is `P_G[Phi(m+delta)-Phi(m)]`, not a compact Taylor approximation.
Every comparison uses actual stored predictors and the previous denominator
semantics: squared sums over selected rows/probes, reporting raw target
energies and the effective `1e-30` floor separately.

## Fixed masks, coverage and conclusions

The **primary mechanism comparison** is absolute-update supported-oracle
error on rows where every one of the three families has at least one eligible
chart. This common-availability mask does not require runtime/direct choices
to be supported. Each finite radius has its own endpoint eligibility. All
three oracle scores on a panel use identical rows and target denominators.

Keep common-selected-support across all nine assignments as a secondary
panel. Load the exact legacy 38 row identities from the saved diagnosis;
never reconstruct them from candidate support or routing. If the candidate
oracle abstains on any legacy row, its full-38 aggregate is unavailable. Any
reduced legacy panel has a separate label, explicit IDs/count/denominator,
and scores every compared method on those same reduced rows. Missing errors
are never zeros or undocumented extrapolations.

The **primary runtime comparison** is absolute-update runtime error on the
original 48 rows, preserving unsupported extrapolations and disclosing their
counts. The practical benchmark is the frozen activation router, including
coverage and runtime/storage requirements. Distinguish no available chart
from selection of an unsupported chart when another is eligible. Verify the
oracle lower-bound property only against eligible predictors on the same
rows, target and denominator.

Report all masks and row identities, per-target coverage, occupancy,
within-group compact-intercept dispersion, opposing output directions,
anchor disagreements and support changes. Smaller dispersion alone is not a
predictive success because the objective optimizes it. Improved predictions
would support this specific modification on this fixture, not uniquely prove
offsets caused the previous error: memberships, medoids and support may all
change together. Reduced common coverage qualifies the mechanism conclusion;
it cannot establish unconditional improvement.

Report derivative and finite-response regressions even if absolute updates
improve. Improvement over original response charts that still falls short of
activation charts is partial improvement. Runtime-resident numeric parameters,
training-only signature storage and privileged discovery metadata are counted
separately. These are synthetic arithmetic/storage measurements, not deployed
compression or speedup evidence.

## Implementation and execution

The new implementation consists of the
[fit module](../src/fisher_graph/fisher_response_geometry_intercept_compatibility.py),
[evaluation module](../src/fisher_graph/fisher_response_geometry_intercept_compatibility_evaluation.py),
[stdlib protocol](../src/fisher_graph/fisher_response_geometry_intercept_compatibility_protocol.py)
and [standalone CLI](../src/fisher_response_geometry_intercept_compatibility_v1.py).
Run from the repository root with a fresh output directory:

```bash
python src/fisher_response_geometry_intercept_compatibility_v1.py synthetic --output /tmp/fisher-response-intercept-compatibility-v1-001
```

`inspect` authenticates metadata without tensor imports. `model-facing`
returns exit status 2 before loading replay or tensor code and preserves
`development_complete_independent_data_unavailable`. No target model,
tokenizer, protected corpus, live attention, new data or independent campaign
is accessed. Original deployed execution and the old artifacts remain
unchanged.

Focused tests cover compact-intercept truncation algebra, stored predictor
equivalence, rebuilt `lambda=0`, training-only access/scaling, constant and
zero blocks, safe unit rescaling, consistent metrics/bases, distinguishable
offsets under equal responses, unchanged support/oracle/denominator semantics,
freeze-before-evaluation, corruption rejection and closed model/data gates.
Completion does not require the performance hypothesis to pass.

## Execution record

Completed on 2026-09-08. The result is a substantial absolute-update gain
over the original response bank, with partial runtime usefulness and
regressions in the secondary runtime Fisher metrics.

### Artifacts and reproduction

The separate final directory is
[`artifacts/research/fisher-response-geometry-intercept-compatibility-v1`](../artifacts/research/fisher-response-geometry-intercept-compatibility-v1).
It contains [matched tables](../artifacts/research/fisher-response-geometry-intercept-compatibility-v1/evaluation.md),
[complete measurements](../artifacts/research/fisher-response-geometry-intercept-compatibility-v1/evaluation.json),
the frozen configuration, rebuilt control, fitted candidate, block scales,
freeze manifest and final artifact manifest. The manifests record exact byte
hashes, source identities, command arguments and runtime versions.

The original replay reproduced 172 numerical comparisons, with maximum
absolute discrepancy `1.7763568394002505e-15`. The genuinely rebuilt
`lambda=0` control passed **358 checks**, including 45 aggregate metric
comparisons; its maximum recorded numerical difference was **zero**. One
consistent chart-label permutation applied throughout. The evaluator also
passed all 120 comparisons against the unchanged saved diagnosis and all
150 eligible-comparator oracle lower-bound checks.

The sole candidate's domain-separated artifact identity is
`aabc3c4971f4b788e73b8c64f88b520104f7104eac914cdd01f8577e8d3625ee`.
The guarded integration run and final artifact run reproduce this same
candidate; they are not different candidates or additional data. All original
probe and diagnosis files and their bound source identities remain unchanged.

Both signature blocks were retained. Their saved centered scales are
`s_x=1.1735954750966129` and `s_z=7.521831855311678`; reference scales are
`1.5310448322864039` and `8.051309539828159`. Their centered/reference ratios
are `0.7665324034594142` and `0.9342370726280905`, respectively. The normalized
blocks have mean squared row norms `1.0` and `1.0000000000000002`.
Neither evaluation statistics nor outcomes changed these training decisions.

### Absolute-update results

Every row below uses the original **48 rows** and the same target-energy
denominators: Fisher `3434.9557057572297`, Euclidean `422.3896176882639`.
All three families have eligible charts on all 48 rows, so common
availability also retains the full population.

| Family | Assignment | Fisher-relative error | Euclidean-relative error | Supported selections |
| --- | --- | ---: | ---: | ---: |
| Original response | Runtime | 0.892656 | 1.223846 | 38/48 |
| Original response | Direct discovery | 0.884313 | 1.222536 | 45/48 |
| Original response | Supported oracle | 0.887349 | 1.225257 | 48/48 |
| Intercept-compatible response | Runtime | 0.269648 | 0.378379 | 48/48 |
| Intercept-compatible response | Target-informed direct discovery | 0.098601 | 0.329921 | 48/48 |
| Intercept-compatible response | Supported oracle | 0.091452 | 0.328261 | 48/48 |
| Frozen activation | All three assignments | 0.136293 | 0.313175 | 48/48 |

The primary mechanism error fell by approximately **89.7%**. The primary
runtime error fell by approximately **69.8%**, but remains **1.9784 times**
the activation runtime error. The new bank has useful eligible predictors
that the unchanged observable-feature router does not always select. Its
target-informed discovery scores do not establish cheap runtime routability.
The new oracle's Fisher score beats the activation oracle on these 48 rows,
while its Euclidean score remains higher; the metric distinction matters.

The exact output-projected update reference remains `0.029547` Fisher error
and `0.186200` Euclidean error. It uses the same `P_G` and full exact target,
with no input-`Q` truncation. Error components are not an additive budget.

The legacy mask retains its exact 38 saved row identities. The new runtime
and oracle update errors there are `0.263031` and `0.073241`, versus original
response `0.989099` and `0.986956`, and activation `0.067475` for both.
The strict common-selected-support mask also retains those 38 rows in this
run. No oracle abstains and no legacy row was dropped. Tests separately
exercise the required unavailable full-mask result when abstention occurs.

### Secondary outcomes and group evidence

| Target | Radius | Original runtime Fisher error | New runtime Fisher error | Original supported-oracle Fisher error | New supported-oracle Fisher error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Derivative action | — | 1.011945 | 1.091190 | 0.876441 | 0.652319 |
| Finite response | 0.001 | 1.011858 | 1.091084 | 0.876368 | 0.652437 |
| Finite response | 0.01 | 1.011075 | 1.090131 | 0.875718 | 0.653511 |
| Finite response | 0.1 | 1.003203 | 1.080219 | 0.869557 | 0.666171 |

Runtime Fisher fidelity regresses on every secondary target despite better
supported-oracle scores. Euclidean runtime derivative error improves from
`0.920335` to `0.901393`; all Euclidean finite-response values are retained
in the full tables. Absolute-update improvement is not a universal response
fidelity improvement.

Training occupancy changes from `83,1,12` to `58,6,32`; activation remains
`32,32,32`. New training medoid row indices are `41,5,13`, compared with
`52,5,79` originally. The original response bank has 1,632 negative-cosine
within-group output pairs; the candidate has zero. Population-weighted
within-group squared Fisher compact-intercept dispersion falls from
`53.453851` to `1.534812` (activation: `1.423554`). Stored-intercept and
anchored predictor forms agree within `2.220446049250313e-15`.

All ten previously unsupported runtime selections and all three previously
unsupported discovery selections become supported, with no lost support,
at every evaluated radius. No family loses any-chart availability. These
are measured changes from the original support policy after rebuilding
memberships and medoids; the support rule itself was not changed.

Smaller dispersion and removal of opposing outputs are consistent with the
intended objective. They are not independent predictive evidence because
the objective explicitly uses intercept compatibility. The prediction gains
support this modification on this examined fixture, while changing
memberships, anchors and support together prevents unique causal attribution
to offsets. No next modification was implemented or queued.

### Cost and verification

All three banks retain **200 runtime float64 scalars (1,600 tensor bytes)**,
69 logical predictor multiply-accumulates per row, 42 router-distance
dimensions and 14 support-distance dimensions. These include the original
zero-state-operator representation and its operations; no runtime
optimization was introduced. Candidate training signatures contain 1,152
scalars versus the original response signatures' 864. Privileged discovery
centers, transforms, metric/teacher tensors, other training arrays and
serialized metadata are accounted for separately in the JSON and manifests.
These counts are not a hardware speed or compression result.

The combined focused suite passed **194 tests**: 72 new tests and 122 existing
tests. It includes full guarded execution, pre-evaluation freeze/reload,
failed-control stop, corruption rejection, scale-safe normalization,
one-permutation reconstruction, runtime teacher-access guards, immutable
legacy masks, abstention, shared denominators and oracle lower bounds.
No model or protected payload was opened.

Commands executed from the repository root:

```sh
.venv/bin/python -m pytest -q -o addopts='' \
  tests/test_fisher_response_geometry_intercept_compatibility.py \
  tests/test_fisher_response_geometry_intercept_compatibility_evaluation.py \
  tests/test_fisher_response_geometry_intercept_compatibility_cli.py \
  tests/test_fisher_response_geometry_failure_diagnosis.py \
  tests/test_fisher_response_geometry_replay.py \
  tests/test_fisher_response_geometry_failure_diagnosis_cli.py \
  tests/test_fisher_response_geometry.py \
  tests/test_fisher_response_geometry_cli.py \
  tests/test_gemma3_attention_generator.py \
  tests/test_gemma3_fisher_local_chart_capture.py

.venv/bin/python src/fisher_response_geometry_intercept_compatibility_v1.py synthetic \
  --output artifacts/research/fisher-response-geometry-intercept-compatibility-v1
```

The first guarded run is preserved at
`/private/tmp/fisher-intercept-compatibility-cli-qa/test_full_ablation_freezes_bef0/intercept-compatibility`.
Changed files are the four new implementation modules, the new configuration,
this document, the three new test files listed above and the eight files in
the final artifact directory. Existing files were not edited.

P0 remains `development_complete_independent_data_unavailable`. Target-model
results, independent confirmation, unique causal attribution and hardware
performance remain unavailable. The experiment stops after this report.
