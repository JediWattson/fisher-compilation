# Fused Two-Layer Modal Executor

The two locked completed modal layers were algebraically folded into
a seven-tensor fast runtime. Normal inference keeps only those
coefficients resident. The first activation capture or intervention
loads the existing logical modal artifacts as a verified sidecar;
later instrumented calls reuse that cache, and explicit eviction
returns the runtime to its default footprint.

## Equivalence

| Split | System | Answer accuracy | Paired accuracy | Hard NLL |
|---|---|---:|---:|---:|
| Validation | teacher | 100.000% | 100.000% | 0.049155 |
| Validation | unfused | 100.000% | 100.000% | 0.048757 |
| Validation | monolithic | 100.000% | 100.000% | 0.048757 |
| Validation | lazy | 100.000% | 100.000% | 0.048757 |
| Exploratory test | teacher | 100.000% | 100.000% | 0.048382 |
| Exploratory test | unfused | 100.000% | 100.000% | 0.049286 |
| Exploratory test | monolithic | 100.000% | 100.000% | 0.049286 |
| Exploratory test | lazy | 100.000% | 100.000% | 0.049286 |

The validation equivalence gate passed before test evaluation:

- Exact argmax predictions: True
- Absolute NLL delta: 1.863e-07
- Mean answer KL: 5.631e-11
- Maximum answer-logit difference: 2.723e-04
- Lazy and monolithic fast logits bit-exact: True

## Arithmetic

| Runtime accounting | Scalar multiplies | Original ratio |
|---|---:|---:|
| Original two blocks | 139264 | 100.000% |
| Unfused logical modal stack | 72384 | 51.976% |
| Current fused dense path | 49152 | 35.294% |
| Triangular nonzero fused path | 30336 | 21.783% |

The current PyTorch `einsum` path executes dense kernels that
contain causal zeros. The triangular number is the available
arithmetic after a backend specializes those causal regions.

## CPU latency

| Batch | Teacher us | Unfused us | Monolithic us | Lazy us | Lazy vs unfused | Lazy/monolithic latency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 108.566 | 173.011 | 54.645 | 54.785 | 3.158x | 1.003x |
| 8 | 200.643 | 235.553 | 67.853 | 67.949 | 3.467x | 1.001x |
| 64 | 751.302 | 358.892 | 109.984 | 108.006 | 3.323x | 0.982x |
| 256 | 2734.323 | 617.775 | 230.706 | 227.514 | 2.715x | 0.986x |

No hard latency threshold was applied. Across the four batch
sizes, the geometric-mean lazy/monolithic latency ratio was
0.9930x (positive regression means the lazy wrapper was slower).

## Resident tensor storage

| State | Bytes |
|---|---:|
| Monolithic fused full runtime | 713920 |
| Lazy full runtime, default | 205952 |
| Logical sidecar after first instrumentation | 203648 |
| Lazy full runtime, sidecar loaded | 409600 |

On disk, the compact runtime can be deployed by itself for
uninstrumented inference. An instrumentable bundle also carries
the four existing modal source artifacts:

| Artifact files | Bytes |
|---|---:|
| Compact lazy runtime | 206893 |
| Four logical sidecar source files | 244724 |
| Compact runtime plus sidecars | 451617 |
| Monolithic fused artifact | 721471 |

## Instrumentation contract

- Default dispatch: seven-tensor forward_fast with exact cross-layer modal bypass
- Trace dispatch: load the verified four-artifact logical sidecar on first capture or intervention, then reuse it
- Sidecar loaded exactly once: True
- Repeated instrumentation reused the cache: True
- Explicit eviction released sidecar tensors: True
- Trace names equal the unfused runtime: True
- Fast-to-traced maximum logit difference: 2.333e-04

Benchmark timings are hardware- and backend-specific. Arithmetic
counts cover only the two replaced blocks; timings cover the
complete model forward. This remains an exploratory
single-checkpoint result because the test split was inspected
during earlier development.
