# Position-Conditioned Modal Executor

## Exploratory test behavior

| System | Answer accuracy | Paired accuracy | Hard NLL |
|---|---:|---:|---:|
| Original transformer | 100.000% | 100.000% | 0.048382 |
| Transformer with modal bottlenecks | 100.000% | 100.000% | 0.050065 |
| Standalone causal affine graph | 48.089% | 7.006% | 1.172144 |
| Standalone causal nonlinear graph | 100.000% | 100.000% | 0.048361 |

## Selected modal graph

- Replaced layer: 1
- Retained input/output modes: 25/19
- Routing width: 24
- Best distillation step: 2000
- Learned parameters: 25592
- Explicit graph edges: 25248
- Estimated multiplies per sequence: 36512
- Original block estimated multiplies: 69632
- Estimated multiply ratio: 52.436%

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
the final test evaluation. Width 24 was the smallest passing
candidate for this one-initialization, 2,000-step search. That is not a
claim of minimum possible capacity. However, this repository's
test split was inspected during earlier exploratory work, so
these numbers are evidence for this checkpoint rather than a
fresh confirmatory result.
