# Synthetic intercept-compatibility ablation

One lambda=1 discovery candidate after a successful lambda=0 rebuilding control. All original banks, bases, examples and runtime feature definitions remain frozen. This is a development ablation on the already examined fixture.

## Primary results

The rebuilt lambda=0 control reproduced the saved baseline before the lambda=1 candidate was evaluated. This is one development ablation on the already examined synthetic fixture.

On 48 common-availability rows, the new absolute-update supported-oracle error is 0.091452, versus 0.887349 for the original response bank (lower_error). The mask requires at least one eligible chart in every family and is independent of runtime selection.

On all 48 original rows, new runtime absolute-update error is 0.269648, versus 0.892656 originally (lower_error). These all-row runtime predictions include any unsupported extrapolations.

The frozen activation runtime benchmark is 0.136293; the new-to-activation error ratio is 1.978438. New runtime selections are supported on 48/48 rows, compared with 48/48 for activation. The full storage and runtime ledger remains part of this comparison.

The runtime gain over the original response bank remains partial: its error is still above the frozen activation benchmark. This is not evidence for replacing activation charts.

Secondary derivative_action at radius None: runtime 1.091190 versus 1.011945 (higher_error); supported oracle 0.652319 versus 0.876441 (lower_error).

Secondary finite_response at radius 0.001: runtime 1.091084 versus 1.011858 (higher_error); supported oracle 0.652437 versus 0.876368 (lower_error).

Secondary finite_response at radius 0.01: runtime 1.090131 versus 1.011075 (higher_error); supported oracle 0.653511 versus 0.875718 (lower_error).

Secondary finite_response at radius 0.1: runtime 1.080219 versus 1.003203 (higher_error); supported oracle 0.666171 versus 0.869557 (lower_error).

Within-group intercept dispersion is an optimized training quantity. Prediction improvements, if observed, support this discovery modification on this fixture; they do not establish offsets as the unique cause because memberships, medoids, support, and routing labels change together. No additional modification was implemented.

## Original 48-row population

Runtime and direct discovery can extrapolate where their selected chart lacks support. A supported-oracle abstention makes a fixed-mask aggregate unavailable. Euclidean scores use the Fisher-selected assignments. Requested and evaluated counts differ for unavailable aggregates.

| Target | Radius | Family | Assignment | Fisher error | Euclidean error | Requested rows | Evaluated rows | Supported selections on full 48 | No eligible chart on full 48 | Unsupported choice despite available chart on full 48 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | activation_charts | runtime_router | 0.136293 | 0.313175 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | direct_discovery | 0.136293 | 0.313175 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | best_supported_oracle | 0.136293 | 0.313175 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | response_charts | runtime_router | 0.892656 | 1.223846 | 48 | 48 | 38/48 | 0 | 10 |
| residual_update | — | response_charts | direct_discovery | 0.884313 | 1.222536 | 48 | 48 | 45/48 | 0 | 3 |
| residual_update | — | response_charts | best_supported_oracle | 0.887349 | 1.225257 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | runtime_router | 0.269648 | 0.378379 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | direct_discovery | 0.098601 | 0.329921 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | best_supported_oracle | 0.091452 | 0.328261 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | runtime_router | 0.945532 | 0.876126 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | direct_discovery | 0.945532 | 0.876126 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | best_supported_oracle | 0.945532 | 0.876126 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | response_charts | runtime_router | 1.011945 | 0.920335 | 48 | 48 | 38/48 | 0 | 10 |
| derivative_action | — | response_charts | direct_discovery | 0.676147 | 0.876175 | 48 | 48 | 45/48 | 0 | 3 |
| derivative_action | — | response_charts | best_supported_oracle | 0.876441 | 0.918629 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | runtime_router | 1.091190 | 0.901393 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | direct_discovery | 0.784916 | 0.848223 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | best_supported_oracle | 0.652319 | 0.855676 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | runtime_router | 0.945529 | 0.876117 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | direct_discovery | 0.945529 | 0.876117 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | best_supported_oracle | 0.945529 | 0.876117 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | response_charts | runtime_router | 1.011858 | 0.920290 | 48 | 48 | 38/48 | 0 | 10 |
| finite_response | 0.001 | response_charts | direct_discovery | 0.676219 | 0.876160 | 48 | 48 | 45/48 | 0 | 3 |
| finite_response | 0.001 | response_charts | best_supported_oracle | 0.876368 | 0.918590 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | runtime_router | 1.091084 | 0.901347 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | direct_discovery | 0.784998 | 0.848219 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | best_supported_oracle | 0.652437 | 0.855671 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | runtime_router | 0.945495 | 0.876037 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | direct_discovery | 0.945495 | 0.876037 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | best_supported_oracle | 0.945495 | 0.876037 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | response_charts | runtime_router | 1.011075 | 0.919888 | 48 | 48 | 38/48 | 0 | 10 |
| finite_response | 0.01 | response_charts | direct_discovery | 0.676887 | 0.876027 | 48 | 48 | 45/48 | 0 | 3 |
| finite_response | 0.01 | response_charts | best_supported_oracle | 0.875718 | 0.918241 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | runtime_router | 1.090131 | 0.900937 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | direct_discovery | 0.785747 | 0.848181 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | best_supported_oracle | 0.653511 | 0.855629 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | runtime_router | 0.945324 | 0.875286 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | direct_discovery | 0.945324 | 0.875286 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | best_supported_oracle | 0.945324 | 0.875286 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | response_charts | runtime_router | 1.003203 | 0.916094 | 48 | 48 | 38/48 | 0 | 10 |
| finite_response | 0.1 | response_charts | direct_discovery | 0.685281 | 0.874990 | 48 | 48 | 45/48 | 0 | 3 |
| finite_response | 0.1 | response_charts | best_supported_oracle | 0.869557 | 0.914985 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | runtime_router | 1.080219 | 0.897005 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | direct_discovery | 0.794173 | 0.847946 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | best_supported_oracle | 0.666171 | 0.858110 | 48 | 48 | 48/48 | 0 | 0 |

| Target | Radius | Exact output-projected Fisher error | Exact output-projected Euclidean error | Rows | Fisher target energy | Euclidean target energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | 0.029547 | 0.186200 | 48 | 3434.9557057572297 | 422.3896176882639 |
| derivative_action | — | 0.194811 | 0.562953 | 48 | 63.10706984135741 | 45.24827536325982 |
| finite_response | 0.001 | 0.194831 | 0.562952 | 48 | 6.310197249702589e-05 | 4.525133943434381e-05 |
| finite_response | 0.01 | 0.195009 | 0.562937 | 48 | 0.00630569112943331 | 0.004527840799386111 |
| finite_response | 0.1 | 0.196451 | 0.562553 | 48 | 0.6269613106898992 | 0.45497304160800006 |

Every method in this panel uses these same target energies and the exact row IDs recorded in `evaluation.json`. The inherited effective denominator is `max(target energy, 1e-30)`. The analytic reference is zero error where a mask is nonempty. The output-projected reference applies `P_G` to the full target without input-Q truncation; component errors are not additive.

## Primary oracle comparison: common availability

Runtime and direct discovery can extrapolate where their selected chart lacks support. A supported-oracle abstention makes a fixed-mask aggregate unavailable. Euclidean scores use the Fisher-selected assignments. Requested and evaluated counts differ for unavailable aggregates.

| Target | Radius | Family | Assignment | Fisher error | Euclidean error | Requested rows | Evaluated rows | Supported selections on full 48 | No eligible chart on full 48 | Unsupported choice despite available chart on full 48 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | activation_charts | runtime_router | 0.136293 | 0.313175 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | direct_discovery | 0.136293 | 0.313175 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | best_supported_oracle | 0.136293 | 0.313175 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | response_charts | runtime_router | 0.892656 | 1.223846 | 48 | 48 | 38/48 | 0 | 10 |
| residual_update | — | response_charts | direct_discovery | 0.884313 | 1.222536 | 48 | 48 | 45/48 | 0 | 3 |
| residual_update | — | response_charts | best_supported_oracle | 0.887349 | 1.225257 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | runtime_router | 0.269648 | 0.378379 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | direct_discovery | 0.098601 | 0.329921 | 48 | 48 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | best_supported_oracle | 0.091452 | 0.328261 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | runtime_router | 0.945532 | 0.876126 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | direct_discovery | 0.945532 | 0.876126 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | best_supported_oracle | 0.945532 | 0.876126 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | response_charts | runtime_router | 1.011945 | 0.920335 | 48 | 48 | 38/48 | 0 | 10 |
| derivative_action | — | response_charts | direct_discovery | 0.676147 | 0.876175 | 48 | 48 | 45/48 | 0 | 3 |
| derivative_action | — | response_charts | best_supported_oracle | 0.876441 | 0.918629 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | runtime_router | 1.091190 | 0.901393 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | direct_discovery | 0.784916 | 0.848223 | 48 | 48 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | best_supported_oracle | 0.652319 | 0.855676 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | runtime_router | 0.945529 | 0.876117 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | direct_discovery | 0.945529 | 0.876117 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | best_supported_oracle | 0.945529 | 0.876117 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | response_charts | runtime_router | 1.011858 | 0.920290 | 48 | 48 | 38/48 | 0 | 10 |
| finite_response | 0.001 | response_charts | direct_discovery | 0.676219 | 0.876160 | 48 | 48 | 45/48 | 0 | 3 |
| finite_response | 0.001 | response_charts | best_supported_oracle | 0.876368 | 0.918590 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | runtime_router | 1.091084 | 0.901347 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | direct_discovery | 0.784998 | 0.848219 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | best_supported_oracle | 0.652437 | 0.855671 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | runtime_router | 0.945495 | 0.876037 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | direct_discovery | 0.945495 | 0.876037 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | best_supported_oracle | 0.945495 | 0.876037 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | response_charts | runtime_router | 1.011075 | 0.919888 | 48 | 48 | 38/48 | 0 | 10 |
| finite_response | 0.01 | response_charts | direct_discovery | 0.676887 | 0.876027 | 48 | 48 | 45/48 | 0 | 3 |
| finite_response | 0.01 | response_charts | best_supported_oracle | 0.875718 | 0.918241 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | runtime_router | 1.090131 | 0.900937 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | direct_discovery | 0.785747 | 0.848181 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | best_supported_oracle | 0.653511 | 0.855629 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | runtime_router | 0.945324 | 0.875286 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | direct_discovery | 0.945324 | 0.875286 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | best_supported_oracle | 0.945324 | 0.875286 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | response_charts | runtime_router | 1.003203 | 0.916094 | 48 | 48 | 38/48 | 0 | 10 |
| finite_response | 0.1 | response_charts | direct_discovery | 0.685281 | 0.874990 | 48 | 48 | 45/48 | 0 | 3 |
| finite_response | 0.1 | response_charts | best_supported_oracle | 0.869557 | 0.914985 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | runtime_router | 1.080219 | 0.897005 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | direct_discovery | 0.794173 | 0.847946 | 48 | 48 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | best_supported_oracle | 0.666171 | 0.858110 | 48 | 48 | 48/48 | 0 | 0 |

| Target | Radius | Exact output-projected Fisher error | Exact output-projected Euclidean error | Rows | Fisher target energy | Euclidean target energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | 0.029547 | 0.186200 | 48 | 3434.9557057572297 | 422.3896176882639 |
| derivative_action | — | 0.194811 | 0.562953 | 48 | 63.10706984135741 | 45.24827536325982 |
| finite_response | 0.001 | 0.194831 | 0.562952 | 48 | 6.310197249702589e-05 | 4.525133943434381e-05 |
| finite_response | 0.01 | 0.195009 | 0.562937 | 48 | 0.00630569112943331 | 0.004527840799386111 |
| finite_response | 0.1 | 0.196451 | 0.562553 | 48 | 0.6269613106898992 | 0.45497304160800006 |

Every method in this panel uses these same target energies and the exact row IDs recorded in `evaluation.json`. The inherited effective denominator is `max(target energy, 1e-30)`. The analytic reference is zero error where a mask is nonempty. The output-projected reference applies `P_G` to the full target without input-Q truncation; component errors are not additive.

## Secondary comparison: common selected support

Runtime and direct discovery can extrapolate where their selected chart lacks support. A supported-oracle abstention makes a fixed-mask aggregate unavailable. Euclidean scores use the Fisher-selected assignments. Requested and evaluated counts differ for unavailable aggregates.

| Target | Radius | Family | Assignment | Fisher error | Euclidean error | Requested rows | Evaluated rows | Supported selections on full 48 | No eligible chart on full 48 | Unsupported choice despite available chart on full 48 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | activation_charts | runtime_router | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | direct_discovery | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | best_supported_oracle | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | response_charts | runtime_router | 0.989099 | 1.360814 | 38 | 38 | 38/48 | 0 | 10 |
| residual_update | — | response_charts | direct_discovery | 0.987027 | 1.359062 | 38 | 38 | 45/48 | 0 | 3 |
| residual_update | — | response_charts | best_supported_oracle | 0.986956 | 1.357895 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | runtime_router | 0.263031 | 0.335571 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | direct_discovery | 0.073241 | 0.255848 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | best_supported_oracle | 0.073241 | 0.255848 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | runtime_router | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | direct_discovery | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | best_supported_oracle | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | response_charts | runtime_router | 0.932439 | 0.951598 | 38 | 38 | 38/48 | 0 | 10 |
| derivative_action | — | response_charts | direct_discovery | 0.885274 | 0.942247 | 38 | 38 | 45/48 | 0 | 3 |
| derivative_action | — | response_charts | best_supported_oracle | 0.885274 | 0.942247 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | runtime_router | 1.192374 | 0.910603 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | direct_discovery | 0.880432 | 0.842563 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | best_supported_oracle | 0.750768 | 0.859067 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | runtime_router | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | direct_discovery | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | best_supported_oracle | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | response_charts | runtime_router | 0.932343 | 0.951572 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.001 | response_charts | direct_discovery | 0.885199 | 0.942222 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.001 | response_charts | best_supported_oracle | 0.885199 | 0.942222 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | runtime_router | 1.192189 | 0.910578 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | direct_discovery | 0.880183 | 0.842540 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | best_supported_oracle | 0.750796 | 0.859073 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | runtime_router | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | direct_discovery | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | best_supported_oracle | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | response_charts | runtime_router | 0.931490 | 0.951338 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.01 | response_charts | direct_discovery | 0.884523 | 0.942001 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.01 | response_charts | best_supported_oracle | 0.884523 | 0.942001 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | runtime_router | 1.190511 | 0.910348 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | direct_discovery | 0.877938 | 0.842339 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | best_supported_oracle | 0.751052 | 0.859119 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | runtime_router | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | direct_discovery | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | best_supported_oracle | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | response_charts | runtime_router | 0.923219 | 0.949199 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.1 | response_charts | direct_discovery | 0.878188 | 0.939982 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.1 | response_charts | best_supported_oracle | 0.878188 | 0.939982 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | runtime_router | 1.172864 | 0.908146 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | direct_discovery | 0.855338 | 0.840421 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | best_supported_oracle | 0.754373 | 0.859612 | 38 | 38 | 48/48 | 0 | 0 |

| Target | Radius | Exact output-projected Fisher error | Exact output-projected Euclidean error | Rows | Fisher target energy | Euclidean target energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | 0.018000 | 0.121590 | 38 | 2735.951713176521 | 328.99003787202207 |
| derivative_action | — | 0.232876 | 0.548331 | 38 | 19.04276398540887 | 20.45299025732592 |
| finite_response | 0.001 | 0.232875 | 0.548310 | 38 | 1.904414835557414e-05 | 2.0455082303259456e-05 |
| finite_response | 0.01 | 0.232860 | 0.548124 | 38 | 0.0019056999623735073 | 0.0020473818973214836 |
| finite_response | 0.1 | 0.232375 | 0.546232 | 38 | 0.19225171056031026 | 0.20651548558791386 |

Every method in this panel uses these same target energies and the exact row IDs recorded in `evaluation.json`. The inherited effective denominator is `max(target energy, 1e-30)`. The analytic reference is zero error where a mask is nonempty. The output-projected reference applies `P_G` to the full target without input-Q truncation; component errors are not additive.

## Fixed legacy 38 rows

Runtime and direct discovery can extrapolate where their selected chart lacks support. A supported-oracle abstention makes a fixed-mask aggregate unavailable. Euclidean scores use the Fisher-selected assignments. Requested and evaluated counts differ for unavailable aggregates.

| Target | Radius | Family | Assignment | Fisher error | Euclidean error | Requested rows | Evaluated rows | Supported selections on full 48 | No eligible chart on full 48 | Unsupported choice despite available chart on full 48 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | activation_charts | runtime_router | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | direct_discovery | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | best_supported_oracle | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | response_charts | runtime_router | 0.989099 | 1.360814 | 38 | 38 | 38/48 | 0 | 10 |
| residual_update | — | response_charts | direct_discovery | 0.987027 | 1.359062 | 38 | 38 | 45/48 | 0 | 3 |
| residual_update | — | response_charts | best_supported_oracle | 0.986956 | 1.357895 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | runtime_router | 0.263031 | 0.335571 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | direct_discovery | 0.073241 | 0.255848 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | best_supported_oracle | 0.073241 | 0.255848 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | runtime_router | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | direct_discovery | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | best_supported_oracle | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | response_charts | runtime_router | 0.932439 | 0.951598 | 38 | 38 | 38/48 | 0 | 10 |
| derivative_action | — | response_charts | direct_discovery | 0.885274 | 0.942247 | 38 | 38 | 45/48 | 0 | 3 |
| derivative_action | — | response_charts | best_supported_oracle | 0.885274 | 0.942247 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | runtime_router | 1.192374 | 0.910603 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | direct_discovery | 0.880432 | 0.842563 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | best_supported_oracle | 0.750768 | 0.859067 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | runtime_router | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | direct_discovery | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | best_supported_oracle | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | response_charts | runtime_router | 0.932343 | 0.951572 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.001 | response_charts | direct_discovery | 0.885199 | 0.942222 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.001 | response_charts | best_supported_oracle | 0.885199 | 0.942222 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | runtime_router | 1.192189 | 0.910578 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | direct_discovery | 0.880183 | 0.842540 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | best_supported_oracle | 0.750796 | 0.859073 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | runtime_router | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | direct_discovery | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | best_supported_oracle | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | response_charts | runtime_router | 0.931490 | 0.951338 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.01 | response_charts | direct_discovery | 0.884523 | 0.942001 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.01 | response_charts | best_supported_oracle | 0.884523 | 0.942001 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | runtime_router | 1.190511 | 0.910348 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | direct_discovery | 0.877938 | 0.842339 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | best_supported_oracle | 0.751052 | 0.859119 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | runtime_router | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | direct_discovery | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | best_supported_oracle | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | response_charts | runtime_router | 0.923219 | 0.949199 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.1 | response_charts | direct_discovery | 0.878188 | 0.939982 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.1 | response_charts | best_supported_oracle | 0.878188 | 0.939982 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | runtime_router | 1.172864 | 0.908146 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | direct_discovery | 0.855338 | 0.840421 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | best_supported_oracle | 0.754373 | 0.859612 | 38 | 38 | 48/48 | 0 | 0 |

| Target | Radius | Exact output-projected Fisher error | Exact output-projected Euclidean error | Rows | Fisher target energy | Euclidean target energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | 0.018000 | 0.121590 | 38 | 2735.951713176521 | 328.99003787202207 |
| derivative_action | — | 0.232876 | 0.548331 | 38 | 19.04276398540887 | 20.45299025732592 |
| finite_response | 0.001 | 0.232875 | 0.548310 | 38 | 1.904414835557414e-05 | 2.0455082303259456e-05 |
| finite_response | 0.01 | 0.232860 | 0.548124 | 38 | 0.0019056999623735073 | 0.0020473818973214836 |
| finite_response | 0.1 | 0.232375 | 0.546232 | 38 | 0.19225171056031026 | 0.20651548558791386 |

Every method in this panel uses these same target energies and the exact row IDs recorded in `evaluation.json`. The inherited effective denominator is `max(target energy, 1e-30)`. The analytic reference is zero error where a mask is nonempty. The output-projected reference applies `P_G` to the full target without input-Q truncation; component errors are not additive.

## Separately labeled legacy intersection with common availability

Runtime and direct discovery can extrapolate where their selected chart lacks support. A supported-oracle abstention makes a fixed-mask aggregate unavailable. Euclidean scores use the Fisher-selected assignments. Requested and evaluated counts differ for unavailable aggregates.

| Target | Radius | Family | Assignment | Fisher error | Euclidean error | Requested rows | Evaluated rows | Supported selections on full 48 | No eligible chart on full 48 | Unsupported choice despite available chart on full 48 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | activation_charts | runtime_router | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | direct_discovery | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | activation_charts | best_supported_oracle | 0.067475 | 0.221167 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | response_charts | runtime_router | 0.989099 | 1.360814 | 38 | 38 | 38/48 | 0 | 10 |
| residual_update | — | response_charts | direct_discovery | 0.987027 | 1.359062 | 38 | 38 | 45/48 | 0 | 3 |
| residual_update | — | response_charts | best_supported_oracle | 0.986956 | 1.357895 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | runtime_router | 0.263031 | 0.335571 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | direct_discovery | 0.073241 | 0.255848 | 38 | 38 | 48/48 | 0 | 0 |
| residual_update | — | intercept_compatible_response_charts | best_supported_oracle | 0.073241 | 0.255848 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | runtime_router | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | direct_discovery | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | activation_charts | best_supported_oracle | 0.744703 | 0.857860 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | response_charts | runtime_router | 0.932439 | 0.951598 | 38 | 38 | 38/48 | 0 | 10 |
| derivative_action | — | response_charts | direct_discovery | 0.885274 | 0.942247 | 38 | 38 | 45/48 | 0 | 3 |
| derivative_action | — | response_charts | best_supported_oracle | 0.885274 | 0.942247 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | runtime_router | 1.192374 | 0.910603 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | direct_discovery | 0.880432 | 0.842563 | 38 | 38 | 48/48 | 0 | 0 |
| derivative_action | — | intercept_compatible_response_charts | best_supported_oracle | 0.750768 | 0.859067 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | runtime_router | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | direct_discovery | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | activation_charts | best_supported_oracle | 0.744748 | 0.857869 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | response_charts | runtime_router | 0.932343 | 0.951572 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.001 | response_charts | direct_discovery | 0.885199 | 0.942222 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.001 | response_charts | best_supported_oracle | 0.885199 | 0.942222 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | runtime_router | 1.192189 | 0.910578 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | direct_discovery | 0.880183 | 0.842540 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.001 | intercept_compatible_response_charts | best_supported_oracle | 0.750796 | 0.859073 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | runtime_router | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | direct_discovery | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | activation_charts | best_supported_oracle | 0.745164 | 0.857945 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | response_charts | runtime_router | 0.931490 | 0.951338 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.01 | response_charts | direct_discovery | 0.884523 | 0.942001 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.01 | response_charts | best_supported_oracle | 0.884523 | 0.942001 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | runtime_router | 1.190511 | 0.910348 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | direct_discovery | 0.877938 | 0.842339 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.01 | intercept_compatible_response_charts | best_supported_oracle | 0.751052 | 0.859119 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | runtime_router | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | direct_discovery | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | activation_charts | best_supported_oracle | 0.750355 | 0.858777 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | response_charts | runtime_router | 0.923219 | 0.949199 | 38 | 38 | 38/48 | 0 | 10 |
| finite_response | 0.1 | response_charts | direct_discovery | 0.878188 | 0.939982 | 38 | 38 | 45/48 | 0 | 3 |
| finite_response | 0.1 | response_charts | best_supported_oracle | 0.878188 | 0.939982 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | runtime_router | 1.172864 | 0.908146 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | direct_discovery | 0.855338 | 0.840421 | 38 | 38 | 48/48 | 0 | 0 |
| finite_response | 0.1 | intercept_compatible_response_charts | best_supported_oracle | 0.754373 | 0.859612 | 38 | 38 | 48/48 | 0 | 0 |

| Target | Radius | Exact output-projected Fisher error | Exact output-projected Euclidean error | Rows | Fisher target energy | Euclidean target energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | 0.018000 | 0.121590 | 38 | 2735.951713176521 | 328.99003787202207 |
| derivative_action | — | 0.232876 | 0.548331 | 38 | 19.04276398540887 | 20.45299025732592 |
| finite_response | 0.001 | 0.232875 | 0.548310 | 38 | 1.904414835557414e-05 | 2.0455082303259456e-05 |
| finite_response | 0.01 | 0.232860 | 0.548124 | 38 | 0.0019056999623735073 | 0.0020473818973214836 |
| finite_response | 0.1 | 0.232375 | 0.546232 | 38 | 0.19225171056031026 | 0.20651548558791386 |

Every method in this panel uses these same target energies and the exact row IDs recorded in `evaluation.json`. The inherited effective denominator is `max(target energy, 1e-30)`. The analytic reference is zero error where a mask is nonempty. The output-projected reference applies `P_G` to the full target without input-Q truncation; component errors are not additive.

## Group and support diagnostics

| Family | Chart | Training rows | Opposing output pairs | Minimum output cosine | Mean squared Fisher compact-intercept dispersion |
| --- | ---: | ---: | ---: | ---: | ---: |
| activation_charts | 0 | 32 | 0/496 | 0.155783 | 2.338522 |
| activation_charts | 1 | 32 | 0/496 | 0.874947 | 0.294421 |
| activation_charts | 2 | 32 | 0/496 | 0.711994 | 1.637721 |
| response_charts | 0 | 83 | 1632/3403 | -0.996401 | 61.446665 |
| response_charts | 1 | 1 | 0/0 | unavailable | 0.000000 |
| response_charts | 2 | 12 | 0/66 | 0.155783 | 2.624707 |
| intercept_compatible_response_charts | 0 | 58 | 0/1653 | 0.404962 | 1.311709 |
| intercept_compatible_response_charts | 1 | 6 | 0/15 | 0.155783 | 3.142617 |
| intercept_compatible_response_charts | 2 | 32 | 0/496 | 0.711994 | 1.637721 |

The JSON also retains actual stored intercepts, anchor mismatches, per-row support changes, output/prediction distance summaries and membership contingency. Smaller training dispersion alone is not predictive success.

## Storage and runtime requirements

| Family | Runtime float64 scalars | Runtime tensor bytes | Predictor multiply-accumulates per row | Training signature scalars |
| --- | ---: | ---: | ---: | ---: |
| activation_charts | 200 | 1600 | 69 | 1344 |
| response_charts | 200 | 1600 | 69 | 864 |
| intercept_compatible_response_charts | 200 | 1600 | 69 | 1152 |

Runtime counts include shared bases, anchored operators and offsets, anchors, support radii, router centroids and original feature scales. Teacher outputs, derivatives, pointwise intercepts, block normalization statistics and discovery centers for target-informed assignment belong to training or privileged diagnostics. Their separate counts are in the JSON. These are tensor storage and logical-operation counts, not measured Python memory or hardware speed.

## Evidence boundary

No target model, tokenizer, protected corpus, live attention, new data or independent-data campaign was used. P0 remains `development_complete_independent_data_unavailable`. No second modification follows automatically.
