# Conditional Modal Completion

The transformer checkpoint stayed frozen. Only deterministic ridge
maps from retained to discarded Fisher coordinates were fitted.

## Validation ablations

| System | Accuracy | Paired | Hard NLL | Teacher KL |
|---|---:|---:|---:|---:|
| Frozen teacher | 100.000% | 100.000% | 0.049155 | 0 |
| Input truncation only | 100.000% | 100.000% | 0.049734 | 8.45016e-05 |
| Input completion only | 100.000% | 100.000% | 0.049475 | 3.31241e-05 |
| Output truncation only | 100.000% | 100.000% | 0.050552 | 0.000250499 |
| Output completion only | 100.000% | 100.000% | 0.048932 | 5.39861e-05 |
| Both truncations | 100.000% | 100.000% | 0.050984 | 0.000326821 |
| Fit-set mean-tail control | 100.000% | 100.000% | 0.050956 | 0.000326338 |
| Both learned completions | 100.000% | 100.000% | 0.049026 | 7.33468e-05 |
| Full-basis oracle round trip | 100.000% | 100.000% | 0.049155 | 1.93605e-08 |
| Standalone modal graph | 100.000% | 100.000% | 0.048313 | 0.00165013 |
| Standalone modal graph + output completion | 100.000% | 100.000% | 0.046722 | 0.00156801 |

## Exploratory test ablations

| System | Accuracy | Paired | Hard NLL | Teacher KL |
|---|---:|---:|---:|---:|
| Frozen teacher | 100.000% | 100.000% | 0.048382 | 0 |
| Input truncation only | 100.000% | 100.000% | 0.048927 | 7.87551e-05 |
| Input completion only | 100.000% | 100.000% | 0.048699 | 2.98459e-05 |
| Output truncation only | 100.000% | 100.000% | 0.049642 | 0.000253724 |
| Output completion only | 100.000% | 100.000% | 0.048068 | 4.91041e-05 |
| Both truncations | 100.000% | 100.000% | 0.050065 | 0.000329147 |
| Fit-set mean-tail control | 100.000% | 100.000% | 0.050032 | 0.000328025 |
| Both learned completions | 100.000% | 100.000% | 0.048175 | 6.64559e-05 |
| Full-basis oracle round trip | 100.000% | 100.000% | 0.048382 | 1.53305e-08 |
| Standalone modal graph | 100.000% | 100.000% | 0.048361 | 0.00157807 |
| Standalone modal graph + output completion | 100.000% | 100.000% | 0.046785 | 0.00143648 |

## Locked bridge

- Input map: shared_local_linear 25 -> 7
- Output map: position_local_linear 19 -> 13
- Ridge: 0.0001
- Learned completion parameters: 2311
- Incremental multiplies versus zero-tail bottleneck: 8496
- Completed standalone-graph multiply ratio: 60.053%

On validation, input-tail completion reached R-squared 0.906955901; output-tail completion reached 0.924580. Both learned bridges beat zero-tail and fit-set mean-tail controls.

The selected pair restored the frozen-layer interface without
changing any teacher weight. The output bridge also improved the
already standalone modal executor, showing that the completion is
useful beyond the diagnostic bottleneck.

These results remain exploratory: the validation split supplied the
saved Fisher basis and the test split had been inspected in earlier
work. This is conditional prediction of redundant tail coordinates,
not guaranteed recovery of arbitrary discarded information.
