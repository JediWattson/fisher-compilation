# Conditional Modal Completion

The transformer checkpoint stayed frozen. Only deterministic ridge
maps from retained to discarded Fisher coordinates were fitted.

## Validation ablations

| System | Accuracy | Paired | Hard NLL | Teacher KL |
|---|---:|---:|---:|---:|
| Frozen teacher | 100.000% | 100.000% | 0.049155 | 0 |
| Input truncation only | 99.682% | 99.363% | 0.060951 | 0.0101768 |
| Input completion only | 100.000% | 100.000% | 0.049155 | 3.37795e-08 |
| Output truncation only | 100.000% | 100.000% | 0.049734 | 8.45035e-05 |
| Output completion only | 100.000% | 100.000% | 0.049168 | 6.22192e-07 |
| Both truncations | 99.682% | 99.363% | 0.061377 | 0.0102536 |
| Fit-set mean-tail control | 99.682% | 99.363% | 0.061415 | 0.0103148 |
| Both learned completions | 100.000% | 100.000% | 0.049168 | 6.24681e-07 |
| Full-basis oracle round trip | 100.000% | 100.000% | 0.049155 | 1.77456e-08 |
| Standalone modal graph | 100.000% | 100.000% | 0.050430 | 0.000959512 |
| Standalone modal graph + output completion | 100.000% | 100.000% | 0.049903 | 0.000869844 |

## Exploratory test ablations

| System | Accuracy | Paired | Hard NLL | Teacher KL |
|---|---:|---:|---:|---:|
| Frozen teacher | 100.000% | 100.000% | 0.048382 | 0 |
| Input truncation only | 99.204% | 98.408% | 0.079320 | 0.0280684 |
| Input completion only | 100.000% | 100.000% | 0.048382 | 3.10961e-08 |
| Output truncation only | 100.000% | 100.000% | 0.048927 | 7.87558e-05 |
| Output completion only | 100.000% | 100.000% | 0.048396 | 6.0199e-07 |
| Both truncations | 99.204% | 98.408% | 0.078740 | 0.0272189 |
| Fit-set mean-tail control | 99.204% | 98.408% | 0.079429 | 0.0278989 |
| Both learned completions | 100.000% | 100.000% | 0.048396 | 6.05758e-07 |
| Full-basis oracle round trip | 100.000% | 100.000% | 0.048382 | 1.59684e-08 |
| Standalone modal graph | 100.000% | 100.000% | 0.049455 | 0.000851936 |
| Standalone modal graph + output completion | 100.000% | 100.000% | 0.048890 | 0.000750618 |

## Locked bridge

- Input map: shared_local_linear 27 -> 5
- Output map: position_local_linear 25 -> 7
- Ridge: 0.0001
- Learned completion parameters: 1631
- Incremental multiplies versus zero-tail bottleneck: 5552
- Completed standalone-graph multiply ratio: 43.899%

On validation, input-tail completion reached R-squared 0.999999989; output-tail completion reached 0.994710. Both learned bridges beat zero-tail and fit-set mean-tail controls.

The selected pair restored the frozen-layer interface without
changing any teacher weight. The output bridge also improved the
already standalone modal executor, showing that the completion is
useful beyond the diagnostic bottleneck.

These results remain exploratory: the validation split supplied the
saved Fisher basis and the test split had been inspected in earlier
work. This is conditional prediction of redundant tail coordinates,
not guaranteed recovery of arbitrary discarded information.
