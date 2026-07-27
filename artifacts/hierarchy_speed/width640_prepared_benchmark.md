# Prepared hierarchy speed probe

This is a synthetic single-boundary kernel benchmark at `640 -> 640` width. It is not a Gemma quality or end-to-end latency result.

Recorded GPU: `Apple M5` with MLX 0.32.0.

| Backend | Rank | Rows | Retained Fisher energy | Candidate state vs dense | Ideal MAC speedup | Factorized vs source | Factorized vs dense candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlx_sync_1 | 80 | 1 | 68.441% | 25.117% | 4.000x | 1.046x | 1.043x |
| mlx_sync_1 | 80 | 8 | 68.441% | 25.117% | 4.000x | 1.002x | 0.993x |
| mlx_sync_1 | 80 | 128 | 68.441% | 25.117% | 4.000x | 1.006x | 1.005x |
| mlx_sync_1 | 80 | 512 | 68.441% | 25.117% | 4.000x | 1.073x | 1.157x |
| mlx_sync_1 | 80 | 2048 | 68.441% | 25.117% | 4.000x | 1.279x | 1.292x |
| mlx_sync_18 | 80 | 1 | 68.441% | 25.117% | 4.000x | 0.933x | 0.933x |
| mlx_sync_18 | 80 | 8 | 68.441% | 25.117% | 4.000x | 1.053x | 1.056x |
| mlx_sync_18 | 80 | 128 | 68.441% | 25.117% | 4.000x | 1.149x | 1.099x |
| mlx_sync_18 | 80 | 512 | 68.441% | 25.117% | 4.000x | 1.277x | 1.273x |
| mlx_sync_18 | 80 | 2048 | 68.441% | 25.117% | 4.000x | 1.334x | 1.333x |
| torch | 80 | 1 | 68.441% | 25.117% | 4.000x | 1.035x | 1.034x |
| torch | 80 | 8 | 68.441% | 25.117% | 4.000x | 2.376x | 2.380x |
| torch | 80 | 128 | 68.441% | 25.117% | 4.000x | 1.974x | 1.975x |
| torch | 80 | 512 | 68.441% | 25.117% | 4.000x | 2.196x | 2.197x |
| torch | 80 | 2048 | 68.441% | 25.117% | 4.000x | 1.971x | 1.968x |
| mlx_sync_1 | 160 | 1 | 90.045% | 50.078% | 2.000x | 1.021x | 1.010x |
| mlx_sync_1 | 160 | 8 | 90.045% | 50.078% | 2.000x | 1.011x | 1.006x |
| mlx_sync_1 | 160 | 128 | 90.045% | 50.078% | 2.000x | 1.010x | 1.005x |
| mlx_sync_1 | 160 | 512 | 90.045% | 50.078% | 2.000x | 1.004x | 1.101x |
| mlx_sync_1 | 160 | 2048 | 90.045% | 50.078% | 2.000x | 1.120x | 1.127x |
| mlx_sync_18 | 160 | 1 | 90.045% | 50.078% | 2.000x | 0.924x | 0.927x |
| mlx_sync_18 | 160 | 8 | 90.045% | 50.078% | 2.000x | 0.976x | 0.965x |
| mlx_sync_18 | 160 | 128 | 90.045% | 50.078% | 2.000x | 1.003x | 1.003x |
| mlx_sync_18 | 160 | 512 | 90.045% | 50.078% | 2.000x | 1.225x | 1.224x |
| mlx_sync_18 | 160 | 2048 | 90.045% | 50.078% | 2.000x | 1.154x | 1.156x |
| torch | 160 | 1 | 90.045% | 50.078% | 2.000x | 0.946x | 0.946x |
| torch | 160 | 8 | 90.045% | 50.078% | 2.000x | 1.643x | 1.649x |
| torch | 160 | 128 | 90.045% | 50.078% | 2.000x | 1.503x | 1.497x |
| torch | 160 | 512 | 90.045% | 50.078% | 2.000x | 1.627x | 1.629x |
| torch | 160 | 2048 | 90.045% | 50.078% | 2.000x | 1.438x | 1.439x |
| mlx_sync_1 | 256 | 1 | 97.512% | 80.031% | 1.250x | 0.981x | 1.004x |
| mlx_sync_1 | 256 | 8 | 97.512% | 80.031% | 1.250x | 0.983x | 0.986x |
| mlx_sync_1 | 256 | 128 | 97.512% | 80.031% | 1.250x | 0.968x | 0.966x |
| mlx_sync_1 | 256 | 512 | 97.512% | 80.031% | 1.250x | 0.990x | 0.986x |
| mlx_sync_1 | 256 | 2048 | 97.512% | 80.031% | 1.250x | 0.989x | 0.987x |
| mlx_sync_18 | 256 | 1 | 97.512% | 80.031% | 1.250x | 0.859x | 0.859x |
| mlx_sync_18 | 256 | 8 | 97.512% | 80.031% | 1.250x | 0.910x | 0.903x |
| mlx_sync_18 | 256 | 128 | 97.512% | 80.031% | 1.250x | 0.950x | 0.932x |
| mlx_sync_18 | 256 | 512 | 97.512% | 80.031% | 1.250x | 1.122x | 1.118x |
| mlx_sync_18 | 256 | 2048 | 97.512% | 80.031% | 1.250x | 0.987x | 0.987x |
| torch | 256 | 1 | 97.512% | 80.031% | 1.250x | 0.861x | 0.863x |
| torch | 256 | 8 | 97.512% | 80.031% | 1.250x | 1.096x | 1.096x |
| torch | 256 | 128 | 97.512% | 80.031% | 1.250x | 1.083x | 1.084x |
| torch | 256 | 512 | 97.512% | 80.031% | 1.250x | 1.139x | 1.140x |
| torch | 256 | 2048 | 97.512% | 80.031% | 1.250x | 1.028x | 1.026x |
| mlx_sync_1 | 320 | 1 | 99.017% | 100.000% | 1.000x | 0.985x | 0.986x |
| mlx_sync_1 | 320 | 8 | 99.017% | 100.000% | 1.000x | 0.942x | 0.941x |
| mlx_sync_1 | 320 | 128 | 99.017% | 100.000% | 1.000x | 0.927x | 0.930x |
| mlx_sync_1 | 320 | 512 | 99.017% | 100.000% | 1.000x | 0.918x | 0.906x |
| mlx_sync_1 | 320 | 2048 | 99.017% | 100.000% | 1.000x | 0.890x | 0.891x |
| mlx_sync_18 | 320 | 1 | 99.017% | 100.000% | 1.000x | 0.845x | 0.831x |
| mlx_sync_18 | 320 | 8 | 99.017% | 100.000% | 1.000x | 0.676x | 0.673x |
| mlx_sync_18 | 320 | 128 | 99.017% | 100.000% | 1.000x | 0.673x | 0.680x |
| mlx_sync_18 | 320 | 512 | 99.017% | 100.000% | 1.000x | 0.782x | 0.782x |
| mlx_sync_18 | 320 | 2048 | 99.017% | 100.000% | 1.000x | 0.845x | 0.844x |
| torch | 320 | 1 | 99.017% | 100.000% | 1.000x | 0.828x | 0.828x |
| torch | 320 | 8 | 99.017% | 100.000% | 1.000x | 0.998x | 0.993x |
| torch | 320 | 128 | 99.017% | 100.000% | 1.000x | 0.966x | 0.965x |
| torch | 320 | 512 | 99.017% | 100.000% | 1.000x | 0.975x | 0.974x |
| torch | 320 | 2048 | 99.017% | 100.000% | 1.000x | 0.865x | 0.864x |

Ratios above `1.0x` mean the factorized candidate was faster. The dense-candidate control represents the same truncated operator as the factorized path, so their ratio isolates low-rank execution geometry.

`mlx_sync_1` synchronizes each standalone boundary call. `mlx_sync_18` executes an 18-stage dependency chain and synchronizes only at the outer traversal; its recorded timings are normalized per stage.

The benchmark removes proof verification, artifact hashing, per-call dtype/device copies, fallback execution, and Fisher error measurement from the timed hot path. Those remain load-time or validation concerns, not free runtime work.
