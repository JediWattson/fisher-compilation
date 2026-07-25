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
| Packed triangular reference | 30336 | 21.783% |

The authenticated dense paths execute kernels that contain
causal zeros. The separate packed triangular reference uses
packed causal-pair PyTorch contractions that execute only the
lower-triangular position pairs; its wall-clock behavior is
measured separately below.

## CPU latency

| Batch | Teacher us | Unfused us | Monolithic us | Lazy us | Lazy vs unfused | Lazy/monolithic latency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 114.198 | 190.748 | 57.049 | 55.976 | 3.408x | 0.981x |
| 8 | 205.205 | 242.431 | 67.965 | 68.461 | 3.541x | 1.007x |
| 64 | 817.487 | 364.664 | 108.473 | 107.841 | 3.381x | 0.994x |
| 256 | 2904.923 | 615.545 | 236.815 | 235.440 | 2.614x | 0.994x |

No hard latency threshold was applied. Across the four batch
sizes, the geometric-mean lazy/monolithic latency ratio was
0.9942x (positive regression means the lazy wrapper was slower).

## Packed triangular reference benchmark

This is an ephemeral runtime derived in memory from the
authenticated lazy artifact. It is neither a serialized artifact
nor the default backend, it updates no weights, and it does not
change the authenticated dense runtime ABI. Validation and the
separate five-system benchmark use only `validation_fisher`; the
test split is not used.

- Implementation: packed_triangular_prefix_v1
- Validation gate passed: True
- Exact lazy/triangular argmax predictions: True
- Absolute lazy/triangular NLL delta: 3.725e-09
- Mean lazy-to-triangular answer KL: 4.940e-14
- Maximum lazy/triangular answer-logit difference: 1.013e-05
- Packed causal position pairs: 36
- Packed fast-state tensor bytes: 125120
- Source lazy sidecar loads before/after: 0/0
- Source lazy sidecar bytes read after benchmark: 0

| Batch | Teacher us | Unfused us | Monolithic us | Lazy us | Triangular us | Triangular vs lazy | Triangular vs unfused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120.468 | 189.765 | 57.326 | 57.362 | 55.558 | 1.032x | 3.416x |
| 8 | 202.747 | 233.638 | 68.453 | 67.494 | 106.630 | 0.633x | 2.191x |
| 64 | 742.211 | 356.055 | 110.095 | 110.239 | 249.445 | 0.442x | 1.427x |
| 256 | 2463.926 | 596.503 | 218.807 | 218.363 | 434.046 | 0.503x | 1.374x |

Across the four batch sizes, the geometric-mean triangular
speedup was 0.6174x versus lazy and 1.9574x versus unfused. Speedups above 1 mean triangular was faster;
values below 1 mean it was slower. No hard latency threshold was
applied.

## Resident tensor storage

| State | Bytes |
|---|---:|
| Monolithic fused full runtime | 713920 |
| Lazy full runtime, default | 205952 |
| Logical sidecar after first instrumentation | 203648 |
| Lazy full runtime, sidecar loaded | 409600 |
| Packed triangular full model | 131264 |

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
