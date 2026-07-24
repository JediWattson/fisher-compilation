# Position-Conditioned Modal Executor

## Exploratory test behavior

| System | Answer accuracy | Paired accuracy | Hard NLL |
|---|---:|---:|---:|
| Original transformer | 100.000% | 100.000% | 0.048382 |
| Transformer with modal bottlenecks | 99.204% | 98.408% | 0.078740 |
| Standalone causal affine graph | 59.076% | 23.885% | 1.891171 |
| Standalone causal nonlinear graph | 100.000% | 100.000% | 0.049455 |

## Selected modal graph

- Replaced layer: 0
- Retained input/output modes: 27/25
- Routing width: 12
- Best distillation step: 2000
- Learned parameters: 14360
- Explicit graph edges: 14064
- Estimated multiplies per sequence: 27376
- Original block estimated multiplies: 69632
- Estimated multiply ratio: 39.315%

The dense surrogate is causal by construction. At each output
position it reads only retained Fisher modes from that position
and earlier positions, passes them through a small GELU routing
bank, predicts retained output modes, and reconstructs the
residual stream using validation-derived position means. The
routing features are learned features, not Fisher eigenmodes.

The bottleneck row still runs the original transformer block and
therefore measures compression loss. The affine and nonlinear
rows remove that block entirely and measure executor fidelity.

This tested affine baseline reached substantial aggregate
activation R-squared but did not preserve associative-recall
behavior; the tested nonlinear graph did. This comparison does
not prove that every successful executor must be nonlinear.

Routing width was selected on the validation/Fisher split before
the final test evaluation. Width 12 was the smallest passing
candidate for this one-initialization, 2,000-step search. That is not a
claim of minimum possible capacity. However, this repository's
test split was inspected during earlier exploratory work, so
these numbers are evidence for this checkpoint rather than a
fresh confirmatory result.
