# MLX/Metal packed modal-stack benchmark

This format-1 report compares three stack-only MLX executions on one GPU. It is an exploratory accelerator measurement. It is not a task-validation report, a serialized runtime, or a change to the authenticated default backend.

- Device: Apple M5
- MLX: 0.32.0
- Dtype: float32
- Output-equivalence gate: passed
- Test split used: no
- Weights updated: no

## Steady-state stack latency

| Batch | Dense compiled | Packed compiled | Packed Metal | Metal vs dense | Metal vs packed |
|---:|---:|---:|---:|---:|---:|
| 1 | 359.525 us | 1068.050 us | 392.740 us | 0.915x | 2.719x |
| 8 | 393.054 us | 1004.162 us | 360.469 us | 1.090x | 2.786x |
| 64 | 387.539 us | 937.733 us | 382.745 us | 1.013x | 2.450x |
| 256 | 402.594 us | 906.611 us | 395.130 us | 1.019x | 2.294x |

Each timed call creates a fresh lazy result, evaluates it, and synchronizes the GPU. 9 measurement rounds rotate system order. First-observed calls are retained in JSON but are not process-isolated cold-start measurements.

## Runtime structure

- Causal pairs stored: 36
- MLX stack state: 124,544 bytes
- Custom kernel: `fisher_packed_causal_gelu`
- Custom kernel accumulation: FP32, safe math, no atomics
- Activation capture/differentiation fallback: ordinary MLX graph
- Activation Fisher oracle: authenticated PyTorch instrumentation path

The custom kernel removes gathered-pair temporaries and indexed reduction. At this toy size, MLX's dense kernels remain highly competitive despite executing structural zeros, so no hard latency gate is applied.
