# Associative Recall Fisher Build

## Trained model

- Selected checkpoint step: 1100
- Validation answer accuracy: 100.000%
- Validation paired-context accuracy: 100.000%
- Test answer accuracy: 100.000%
- Test paired-context accuracy: 100.000%
- Mean correct-answer probability: 95.208%

## Width-pooled Fisher modes

| Activation | Fisher trace | Effective rank | k90 | k95 | k99 |
|---|---:|---:|---:|---:|---:|
| `layer.0.input` | 3.394788e-04 | 16.368 | 15 | 19 | 27 |
| `layer.0.post_attention` | 2.067538e-04 | 15.864 | 15 | 18 | 26 |
| `layer.0.output` | 1.818517e-04 | 15.275 | 14 | 18 | 25 |
| `layer.1.post_attention` | 1.099359e-04 | 10.094 | 10 | 13 | 21 |
| `layer.1.output` | 9.463919e-05 | 8.554 | 9 | 12 | 19 |
| `final_norm` | 3.798961e-04 | 7.739 | 7 | 8 | 8 |

Each basis diagonalizes the full 32 x 32 empirical Fisher
constructed from hard-target, summed-NLL activation score
gradients on the validation/Fisher split. Token positions are
pooled as observations, producing modes reusable across positions.

## Position-coupled modal computation

| Layer | Modes in/out | Descriptive R2 | Jacobian samples |
|---|---:|---:|---:|
| 0 | 27/25 | 0.737423 | 24 |
| 1 | 25/19 | 0.941812 | 24 |

The saved modal Jacobians have axes
`[output_position, output_mode, input_position, input_mode]`.
Both signed means and RMS magnitudes are stored. The affine
transitions are descriptive dataset fits and are not claimed as
causal executors.
