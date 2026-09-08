# Synthetic normalization-response failure diagnosis

Additional diagnostic replay of the already examined fixture. No refitting, new model evidence or deployed change.

The original report and source identities were checked before and after execution. Missing input tensors and probe directions were reconstructed from the pinned draw schedule; fitted parameters were loaded unchanged.

## Preserved all-row absolute-update baseline

These rows retain all original evaluation examples, including unsupported extrapolations, and the original aggregate Fisher-relative error definition.

| Family | Assignment | Fisher relative error | Euclidean relative error | Rows |
| --- | --- | ---: | ---: | ---: |
| response_charts | runtime_router | 0.892656 | 1.223846 | 48 |
| response_charts | direct_discovery | 0.884313 | 1.222536 | 48 |
| activation_charts | runtime_router | 0.136293 | 0.313175 | 48 |
| activation_charts | direct_discovery | 0.136293 | 0.313175 | 48 |

## Main comparison

For each target/radius, every row below uses the same intersection of rows with at least one eligible chart in each family. Runtime/discovery selections may still extrapolate; the unsupported-selection column discloses those rows. The oracle abstains outside its support. Euclidean errors use the Fisher-selected assignments.

| Target | Radius | Family | Assignment | Fisher relative error | Euclidean relative error | Rows | Selected support, full fixture | Unsupported selected, scored rows |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | response_charts | runtime_router | 0.892656 | 1.223846 | 48 | 38/48 | 10 |
| residual_update | — | response_charts | direct_discovery | 0.884313 | 1.222536 | 48 | 45/48 | 3 |
| residual_update | — | response_charts | best_supported_oracle | 0.887349 | 1.225257 | 48 | 48/48 | 0 |
| residual_update | — | activation_charts | runtime_router | 0.136293 | 0.313175 | 48 | 48/48 | 0 |
| residual_update | — | activation_charts | direct_discovery | 0.136293 | 0.313175 | 48 | 48/48 | 0 |
| residual_update | — | activation_charts | best_supported_oracle | 0.136293 | 0.313175 | 48 | 48/48 | 0 |
| derivative_action | — | response_charts | runtime_router | 1.011945 | 0.920335 | 48 | 38/48 | 10 |
| derivative_action | — | response_charts | direct_discovery | 0.676147 | 0.876175 | 48 | 45/48 | 3 |
| derivative_action | — | response_charts | best_supported_oracle | 0.876441 | 0.918629 | 48 | 48/48 | 0 |
| derivative_action | — | activation_charts | runtime_router | 0.945532 | 0.876126 | 48 | 48/48 | 0 |
| derivative_action | — | activation_charts | direct_discovery | 0.945532 | 0.876126 | 48 | 48/48 | 0 |
| derivative_action | — | activation_charts | best_supported_oracle | 0.945532 | 0.876126 | 48 | 48/48 | 0 |
| finite_response | 0.001 | response_charts | runtime_router | 1.011858 | 0.920290 | 48 | 38/48 | 10 |
| finite_response | 0.001 | response_charts | direct_discovery | 0.676219 | 0.876160 | 48 | 45/48 | 3 |
| finite_response | 0.001 | response_charts | best_supported_oracle | 0.876368 | 0.918590 | 48 | 48/48 | 0 |
| finite_response | 0.001 | activation_charts | runtime_router | 0.945529 | 0.876117 | 48 | 48/48 | 0 |
| finite_response | 0.001 | activation_charts | direct_discovery | 0.945529 | 0.876117 | 48 | 48/48 | 0 |
| finite_response | 0.001 | activation_charts | best_supported_oracle | 0.945529 | 0.876117 | 48 | 48/48 | 0 |
| finite_response | 0.01 | response_charts | runtime_router | 1.011075 | 0.919888 | 48 | 38/48 | 10 |
| finite_response | 0.01 | response_charts | direct_discovery | 0.676887 | 0.876027 | 48 | 45/48 | 3 |
| finite_response | 0.01 | response_charts | best_supported_oracle | 0.875718 | 0.918241 | 48 | 48/48 | 0 |
| finite_response | 0.01 | activation_charts | runtime_router | 0.945495 | 0.876037 | 48 | 48/48 | 0 |
| finite_response | 0.01 | activation_charts | direct_discovery | 0.945495 | 0.876037 | 48 | 48/48 | 0 |
| finite_response | 0.01 | activation_charts | best_supported_oracle | 0.945495 | 0.876037 | 48 | 48/48 | 0 |
| finite_response | 0.1 | response_charts | runtime_router | 1.003203 | 0.916094 | 48 | 38/48 | 10 |
| finite_response | 0.1 | response_charts | direct_discovery | 0.685281 | 0.874990 | 48 | 45/48 | 3 |
| finite_response | 0.1 | response_charts | best_supported_oracle | 0.869557 | 0.914985 | 48 | 48/48 | 0 |
| finite_response | 0.1 | activation_charts | runtime_router | 0.945324 | 0.875286 | 48 | 48/48 | 0 |
| finite_response | 0.1 | activation_charts | direct_discovery | 0.945324 | 0.875286 | 48 | 48/48 | 0 |
| finite_response | 0.1 | activation_charts | best_supported_oracle | 0.945324 | 0.875286 | 48 | 48/48 | 0 |

## Exact projected reference

This is `P_G` applied to the exact target, without input-basis truncation. The unprojected analytic reference has zero error by definition.

| Target | Radius | Fisher relative error | Euclidean relative error | Rows |
| --- | --- | ---: | ---: | ---: |
| residual_update | — | 0.029547 | 0.186200 | 48 |
| derivative_action | — | 0.194811 | 0.562953 | 48 |
| finite_response | 0.001 | 0.194831 | 0.562952 | 48 |
| finite_response | 0.01 | 0.195009 | 0.562937 | 48 |
| finite_response | 0.1 | 0.196451 | 0.562553 | 48 |

## Strict common selected support

This smaller secondary mask requires every runtime, discovery and oracle selection in both families to support the row. Conditioning can hide selection failures, so this panel does not replace the main table.

| Target | Radius | Family | Assignment | Fisher relative error | Euclidean relative error | Rows |
| --- | --- | --- | --- | ---: | ---: | ---: |
| residual_update | — | response_charts | runtime_router | 0.989099 | 1.360814 | 38 |
| residual_update | — | response_charts | direct_discovery | 0.987027 | 1.359062 | 38 |
| residual_update | — | response_charts | best_supported_oracle | 0.986956 | 1.357895 | 38 |
| residual_update | — | activation_charts | runtime_router | 0.067475 | 0.221167 | 38 |
| residual_update | — | activation_charts | direct_discovery | 0.067475 | 0.221167 | 38 |
| residual_update | — | activation_charts | best_supported_oracle | 0.067475 | 0.221167 | 38 |
| derivative_action | — | response_charts | runtime_router | 0.932439 | 0.951598 | 38 |
| derivative_action | — | response_charts | direct_discovery | 0.885274 | 0.942247 | 38 |
| derivative_action | — | response_charts | best_supported_oracle | 0.885274 | 0.942247 | 38 |
| derivative_action | — | activation_charts | runtime_router | 0.744703 | 0.857860 | 38 |
| derivative_action | — | activation_charts | direct_discovery | 0.744703 | 0.857860 | 38 |
| derivative_action | — | activation_charts | best_supported_oracle | 0.744703 | 0.857860 | 38 |
| finite_response | 0.001 | response_charts | runtime_router | 0.932343 | 0.951572 | 38 |
| finite_response | 0.001 | response_charts | direct_discovery | 0.885199 | 0.942222 | 38 |
| finite_response | 0.001 | response_charts | best_supported_oracle | 0.885199 | 0.942222 | 38 |
| finite_response | 0.001 | activation_charts | runtime_router | 0.744748 | 0.857869 | 38 |
| finite_response | 0.001 | activation_charts | direct_discovery | 0.744748 | 0.857869 | 38 |
| finite_response | 0.001 | activation_charts | best_supported_oracle | 0.744748 | 0.857869 | 38 |
| finite_response | 0.01 | response_charts | runtime_router | 0.931490 | 0.951338 | 38 |
| finite_response | 0.01 | response_charts | direct_discovery | 0.884523 | 0.942001 | 38 |
| finite_response | 0.01 | response_charts | best_supported_oracle | 0.884523 | 0.942001 | 38 |
| finite_response | 0.01 | activation_charts | runtime_router | 0.745164 | 0.857945 | 38 |
| finite_response | 0.01 | activation_charts | direct_discovery | 0.745164 | 0.857945 | 38 |
| finite_response | 0.01 | activation_charts | best_supported_oracle | 0.745164 | 0.857945 | 38 |
| finite_response | 0.1 | response_charts | runtime_router | 0.923219 | 0.949199 | 38 |
| finite_response | 0.1 | response_charts | direct_discovery | 0.878188 | 0.939982 | 38 |
| finite_response | 0.1 | response_charts | best_supported_oracle | 0.878188 | 0.939982 | 38 |
| finite_response | 0.1 | activation_charts | runtime_router | 0.750355 | 0.858777 | 38 |
| finite_response | 0.1 | activation_charts | direct_discovery | 0.750355 | 0.858777 | 38 |
| finite_response | 0.1 | activation_charts | best_supported_oracle | 0.750355 | 0.858777 | 38 |

## Support and group evidence

- response_charts: 16 evaluation rows have a negative output cosine to their direct-assignment anchor. Compact-intercept mismatch/error correlation is 0.975320; this is descriptive and shares target information.
  - Chart 0: 83 fit rows, 1632/3403 within-group output pairs have negative cosine; minimum cosine -0.996401.
  - Chart 1: 1 fit rows, 0/0 within-group output pairs have negative cosine; minimum cosine missing.
  - Chart 2: 12 fit rows, 0/66 within-group output pairs have negative cosine; minimum cosine 0.155783.
- activation_charts: 0 evaluation rows have a negative output cosine to their direct-assignment anchor. Compact-intercept mismatch/error correlation is 0.964688; this is descriptive and shares target information.
  - Chart 0: 32 fit rows, 0/496 within-group output pairs have negative cosine; minimum cosine 0.155783.
  - Chart 1: 32 fit rows, 0/496 within-group output pairs have negative cosine; minimum cosine 0.874947.
  - Chart 2: 32 fit rows, 0/496 within-group output pairs have negative cosine; minimum cosine 0.711994.

## Interpretation and limitations

On the same 48 common-availability rows, response update error is 0.892656 with runtime routing, 0.884313 with discovery assignment, and 0.887349 with the supported target oracle. Activation's supported oracle error is 0.136293. Routing changes alone cannot rescue these frozen response predictors on the examined fixture.

The stricter common-selected-support panel retains 38 rows. Response's best eligible update error is 0.986956, versus 0.067475 for activation. The update failure persists when every compared selection is supported.

Of 48 fixture rows, 0 have no eligible response chart and 0 have no eligible activation chart. Response runtime routing makes 10 unsupported selections despite available support, and discovery makes 3. An extrapolating direct assignment can score better than the supported oracle because its choice is outside that oracle's eligible set. This does not violate the oracle's supported lower-bound property.

For the original derivative probes, response runtime/discovery/supported-oracle errors are 1.011945, 0.676147, and 0.876441. Derivative selection is more sensitive to routing and support than absolute-update selection. The finite-response panels at the 3 frozen radii follow a similar pattern; all remain substantially imperfect shared-chart predictions despite correct analytic derivatives.

Response chart 0 contains 83 training rows, including 1632/3403 output pairs with negative cosine (minimum -0.996401). There are 16 evaluation rows with negative output cosine to the discovery-selected anchor. These are observed incompatible outputs within the fitted groups, consistent with the even-Jacobian/odd-output ambiguity. They do not establish a unique causal explanation.

Compact-intercept mismatch and update error have descriptive Pearson correlation 0.975320. Both statistics use the same target, so correlation is not an independent causal test. Projection contributes error, as shown by the exact projected reference; the difference between component norms is not an additive error budget.

The JSON report preserves all-row extrapolation panels, masks, row indices, denominators, chart assignments, eligibility matrices, no-chart versus wrong-selection counts, and reproduction differences. Error components are not assumed additive.

No target model, protected corpus, tokenizer, live attention or independent-data campaign was used. P0 remains `development_complete_independent_data_unavailable`.

One proposed next modification (not implemented): In a separate experiment, add a training-only compact-intercept compatibility constraint to response-chart discovery while keeping the chart count and basis ranks fixed. This diagnostic does not implement or validate that change.
