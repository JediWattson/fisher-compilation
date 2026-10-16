# Synthetic Pullback-Fisher Input Chart and EWC Parameter Consolidation Ablation

## Summary

- **Original reproduction status:** `reproduced`
- **Lambda-zero rebuild control status:** `reproduced`
- **Original Euclidean-PCA Q retained pullback Fisher fraction:** `0.336701`
- **Pullback-Fisher Q_pb retained pullback Fisher fraction:** `0.950464`
- **Candidate training occupancy:** `[32, 32, 32]`

## All 48 Evaluation Rows (`all_rows`)

| Target | Radius | Family | Fisher (runtime) | Fisher (direct) | Fisher (oracle) | Euclidean (runtime) | Support | Runtime=Direct |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | activation_charts | 0.136293 | 0.136293 | 0.136293 | 0.313175 | 48/48 | 1.000 |
| residual_update | — | response_charts | 0.892656 | 0.884313 | 0.887349 | 1.223846 | 38/48 | 0.750 |
| residual_update | — | intercept_compatible_response_charts | 0.269648 | 0.098601 | 0.091452 | 0.378379 | 48/48 | 0.812 |
| residual_update | — | fixed_q_ewc_response_charts | 0.114031 | 0.090781 | 0.089179 | 0.292030 | 46/48 | 0.875 |
| residual_update | — | pullback_ewc_response_charts | 0.079460 | 0.079460 | 0.079460 | 0.205869 | 48/48 | 1.000 |
| derivative_action | — | activation_charts | 0.945532 | 0.945532 | 0.945532 | 0.876126 | 48/48 | 1.000 |
| derivative_action | — | response_charts | 1.011945 | 0.676147 | 0.876441 | 0.920335 | 38/48 | 0.750 |
| derivative_action | — | intercept_compatible_response_charts | 1.091190 | 0.784916 | 0.652319 | 0.901393 | 48/48 | 0.812 |
| derivative_action | — | fixed_q_ewc_response_charts | 0.813037 | 0.769566 | 0.755772 | 0.854753 | 46/48 | 0.875 |
| derivative_action | — | pullback_ewc_response_charts | 0.715604 | 0.715604 | 0.715604 | 0.652435 | 48/48 | 1.000 |
| finite_response | 0.001 | activation_charts | 0.945529 | 0.945529 | 0.945529 | 0.876117 | 48/48 | 1.000 |
| finite_response | 0.001 | response_charts | 1.011858 | 0.676219 | 0.876368 | 0.920290 | 38/48 | 0.750 |
| finite_response | 0.001 | intercept_compatible_response_charts | 1.091084 | 0.784998 | 0.652437 | 0.901347 | 48/48 | 0.812 |
| finite_response | 0.001 | fixed_q_ewc_response_charts | 0.812950 | 0.769492 | 0.755654 | 0.854723 | 46/48 | 0.875 |
| finite_response | 0.001 | pullback_ewc_response_charts | 0.715543 | 0.715543 | 0.715543 | 0.652405 | 48/48 | 1.000 |
| finite_response | 0.01 | activation_charts | 0.945495 | 0.945495 | 0.945495 | 0.876037 | 48/48 | 1.000 |
| finite_response | 0.01 | response_charts | 1.011075 | 0.676887 | 0.875718 | 0.919888 | 38/48 | 0.750 |
| finite_response | 0.01 | intercept_compatible_response_charts | 1.090131 | 0.785747 | 0.653511 | 0.900937 | 48/48 | 0.812 |
| finite_response | 0.01 | fixed_q_ewc_response_charts | 0.812162 | 0.768828 | 0.754596 | 0.854452 | 46/48 | 0.875 |
| finite_response | 0.01 | pullback_ewc_response_charts | 0.714999 | 0.714999 | 0.714999 | 0.652132 | 48/48 | 1.000 |
| finite_response | 0.1 | activation_charts | 0.945324 | 0.945324 | 0.945324 | 0.875286 | 48/48 | 1.000 |
| finite_response | 0.1 | response_charts | 1.003203 | 0.685281 | 0.869557 | 0.916094 | 38/48 | 0.750 |
| finite_response | 0.1 | intercept_compatible_response_charts | 1.080219 | 0.794173 | 0.666171 | 0.897005 | 48/48 | 0.812 |
| finite_response | 0.1 | fixed_q_ewc_response_charts | 0.804540 | 0.762623 | 0.744517 | 0.851869 | 46/48 | 0.875 |
| finite_response | 0.1 | pullback_ewc_response_charts | 0.710318 | 0.710318 | 0.710318 | 0.649612 | 48/48 | 1.000 |

## Common Selected-Support 38-Row Mask (`legacy_38`)

| Target | Radius | Family | Fisher (runtime) | Fisher (direct) | Fisher (oracle) | Euclidean (runtime) | Support | Runtime=Direct |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_update | — | activation_charts | 0.067475 | 0.067475 | 0.067475 | 0.221167 | 48/48 | 1.000 |
| residual_update | — | response_charts | 0.989099 | 0.987027 | 0.986956 | 1.360814 | 38/48 | 0.750 |
| residual_update | — | intercept_compatible_response_charts | 0.263031 | 0.073241 | 0.073241 | 0.335571 | 48/48 | 0.812 |
| residual_update | — | fixed_q_ewc_response_charts | 0.094867 | 0.061065 | 0.060177 | 0.227423 | 46/48 | 0.875 |
| residual_update | — | pullback_ewc_response_charts | 0.041704 | 0.041704 | 0.041704 | 0.130973 | 48/48 | 1.000 |
| derivative_action | — | activation_charts | 0.744703 | 0.744703 | 0.744703 | 0.857860 | 48/48 | 1.000 |
| derivative_action | — | response_charts | 0.932439 | 0.885274 | 0.885274 | 0.951598 | 38/48 | 0.750 |
| derivative_action | — | intercept_compatible_response_charts | 1.192374 | 0.880432 | 0.750768 | 0.910603 | 48/48 | 0.812 |
| derivative_action | — | fixed_q_ewc_response_charts | 0.806615 | 0.725902 | 0.725902 | 0.862368 | 46/48 | 0.875 |
| derivative_action | — | pullback_ewc_response_charts | 0.418076 | 0.418076 | 0.418076 | 0.595332 | 48/48 | 1.000 |
| finite_response | 0.001 | activation_charts | 0.744748 | 0.744748 | 0.744748 | 0.857869 | 48/48 | 1.000 |
| finite_response | 0.001 | response_charts | 0.932343 | 0.885199 | 0.885199 | 0.951572 | 38/48 | 0.750 |
| finite_response | 0.001 | intercept_compatible_response_charts | 1.192189 | 0.880183 | 0.750796 | 0.910578 | 48/48 | 0.812 |
| finite_response | 0.001 | fixed_q_ewc_response_charts | 0.806521 | 0.725794 | 0.725794 | 0.862360 | 46/48 | 0.875 |
| finite_response | 0.001 | pullback_ewc_response_charts | 0.418087 | 0.418087 | 0.418087 | 0.595292 | 48/48 | 1.000 |
| finite_response | 0.01 | activation_charts | 0.745164 | 0.745164 | 0.745164 | 0.857945 | 48/48 | 1.000 |
| finite_response | 0.01 | response_charts | 0.931490 | 0.884523 | 0.884523 | 0.951338 | 38/48 | 0.750 |
| finite_response | 0.01 | intercept_compatible_response_charts | 1.190511 | 0.877938 | 0.751052 | 0.910348 | 48/48 | 0.812 |
| finite_response | 0.01 | fixed_q_ewc_response_charts | 0.805683 | 0.724827 | 0.724827 | 0.862295 | 46/48 | 0.875 |
| finite_response | 0.01 | pullback_ewc_response_charts | 0.418216 | 0.418216 | 0.418216 | 0.594941 | 48/48 | 1.000 |
| finite_response | 0.1 | activation_charts | 0.750355 | 0.750355 | 0.750355 | 0.858777 | 48/48 | 1.000 |
| finite_response | 0.1 | response_charts | 0.923219 | 0.878188 | 0.878188 | 0.949199 | 38/48 | 0.750 |
| finite_response | 0.1 | intercept_compatible_response_charts | 1.172864 | 0.855338 | 0.754373 | 0.908146 | 48/48 | 0.812 |
| finite_response | 0.1 | fixed_q_ewc_response_charts | 0.797691 | 0.715900 | 0.715900 | 0.861748 | 46/48 | 0.875 |
| finite_response | 0.1 | pullback_ewc_response_charts | 0.422474 | 0.422474 | 0.422474 | 0.591794 | 48/48 | 1.000 |

