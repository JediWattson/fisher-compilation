# Frozen Two-Layer Modal Composition

Both transformer blocks were replaced by independently fitted modal
graph executors. No transformer weight was updated, and layer 1 was
trained only against its frozen teacher on matching clean inputs.

## Behavior

| Split | System | Answer accuracy | Paired accuracy | Hard NLL | KL vs teacher |
|---|---|---:|---:|---:|---:|
| Validation | teacher | 100.000% | 100.000% | 0.049155 | 0.000000 |
| Validation | layer 0 completed | 100.000% | 100.000% | 0.049903 | 0.000870 |
| Validation | layer 1 completed | 100.000% | 100.000% | 0.046722 | 0.001568 |
| Validation | both completed | 100.000% | 100.000% | 0.048757 | 0.002911 |
| Exploratory test | teacher | 100.000% | 100.000% | 0.048382 | 0.000000 |
| Exploratory test | layer 0 completed | 100.000% | 100.000% | 0.048890 | 0.000751 |
| Exploratory test | layer 1 completed | 100.000% | 100.000% | 0.046785 | 0.001436 |
| Exploratory test | both completed | 100.000% | 100.000% | 0.049286 | 0.003061 |

## Same-input composition contract

The critical comparison holds the input to layer 1 fixed:
`B1(E0(h))` versus `E1(E0(h))`. On validation its suffix KL is 0.002538, and its Fisher-weighted layer-output RMS error is 0.003285. This avoids mistaking
downstream cancellation of layer-0 error for layer-1 fidelity.

The raw upstream/local error cosine is -0.168536; the Fisher-weighted cosine is -0.206607.

## Compute estimate

- Both original blocks: 139264 multiplies
- Both completed modal graphs: 72384 multiplies
- Compiled/original ratio: 51.976%

These are block-only scalar-multiply estimates. Learned
parameters and stored buffers are reported separately in JSON.

This remains an exploratory single-checkpoint result because the
test split was inspected during earlier development.
