# Layer-13 EWC retention and causal neuron localization on Qwen3-1.7B

This follow-up investigates the [expanded evaluation](qwen3-mlx-expanded-evaluation.md)
regression where replacing layer 13's MLP reduced accuracy on the 96-question
task panel from 68/96 (exact) and 66/96 (attention-only) to 49/96, despite
nearly unchanged next-token NLL on 65,456 WikiText targets.

All candidates below retain the exact same layer-13 architecture and parameter
budget as `attention-and-mlp` (37,467,000 scalars removed; 71.46 MiB BF16
storage saved). Complete hashes and per-task metrics are recorded in
[`artifacts/qwen3/ewc_retention_results.json`](../artifacts/qwen3/ewc_retention_results.json).

## 1. Root-cause diagnosis via graph and neuron inspection

Separating the fitted layer-13 MLP into its global linear term
(`x @ affine_weight.T + affine_bias`) and its four local rank-16 charts
(`chart_mode_decoders`) exposed three interacting causes of the regression:

1. **High-dimensional chart router flattening (`d = 2048`).**
   With `chart_distance_scales` set to the root-mean-square 2048-dimensional
   distance to each k-means center (`~17.6–23.0`, so `scale^2 ~ 440`),
   concentration of measure made `-0.5 * ||x - c||^2 / scale^2` nearly constant
   across charts (`membership` standard deviation `~0.033`). As a result,
   `99.02%` of the chart feature energy lay inside the linear column space of
   `[x, 1]`. After subtracting `affine_features @ affine`, the four charts
   reduced calibration relative error by only `0.0007` (`0.4594 -> 0.4587`) and
   held-out diagnostic error by `0.0000` (`0.8254 -> 0.8254`).
2. **Unanchored ridge shrinkage and intercept penalization.**
   Fitting an unconstrained `2049 x 2048` matrix on `4,096` correlated WikiText
   tokens (`effective rank = 4.1`) with `anchor=None` shrank the 453+
   low-variance input directions toward zero and ridge-penalized `affine_bias`
   toward zero.
3. **Dormant chat-boundary SwiGLU neurons (`channels 4172, 4971, 751, 4206, 287`).**
   Position-level causal interventions on `mlp.13.output` across BoolQ prompts
   showed that restoring the exact layer-13 MLP output at **only the final
   token** (`offset = -1`, `\n\n` after `<|im_start|>assistant\n<think>\n\n</think>\n\n`)
   shifted `logit(A) - logit(B)` by `+9.75` logit units, and restoring the last
   15 formatting/choice tokens shifted mean `logit(A) - logit(B)` from `-7.30`
   to `+1.27` (versus `+7.00` for `attention-only`).
   Inspecting the 6,144 SwiGLU channels at `offset = -1` showed that the top
   five active neurons (`4172, 4971, 751, 4206, 287`, firing at up to `15.36`,
   `77x` the median channel `0.198`) sit at ranks `4998, 2564, 3823, 5242, 2918`
   out of `6144` on raw WikiText-2 (`~0.33` RMS) because `fit.jsonl` contains
   zero `<|im_start|>`, `<|im_end|>`, `<think>`, or `</think>` tokens.

## 2. Results across EWC arms

We compared post-fit trajectory optimization (`scripts/refine_qwen3_ewc.py`)
against in-fit operator-pullback EWC anchoring inside `fit_chart_mlp`:

- **SwiGLU Euler-Jacobian prior (`W_Euler`).** Because SwiGLU is approximately
  homogeneous of degree 2, Euler's theorem gives the expected linear operator
  prior `W_Euler = 0.5 * E[J(x)]` computed directly from the layer's source
  `gate_weight`, `up_weight`, and `down_weight` before removal.
- **Centered intercept & inverse-norm weights.** Centering `(x, y)` avoids
  ridge-shrinking `affine_bias`, and clipped inverse target-norm weights
  (`clip(median(||y||^2) / ||y_i||^2, 0.05, 5.0)`) bound outlier-token leverage.

| Variant (Layer 13, 71.46 MiB saved) | 512-Token Top-1 Agree (`32,704`) | 512-Token NLL Change | Task Total (`96`) | BoolQ (`32`) | PIQA (`32`) | ARC-Easy (`32`) | Task Teacher Agree | Task Label KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact teacher (0 MiB saved) | 100.00% | `0.00000` | **68/96** | 24 | 18 | 26 | 100.00% | `0.000` |
| Attention-only (8.00 MiB saved) | 95.00% | `+0.00359` | **66/96** | 23 | 17 | 26 | 90.62% | `0.211` |
| First-fit joint (`ridge=0.05`, zero anchor) | 87.21% | `-0.00777` | **49/96** | 10 | 18 | 21 | 68.75% | `2.304` |
| Post-fit `unanchored_a0` (`teacher_kl`, 48 steps) | 86.68% | `-0.03216` | **49/96** | 10 | 18 | 21 | 68.75% | `2.009` |
| Post-fit `isotropic_anchor_a1e+07` (`terminal_mse`) | 87.07% | `+0.00039` | **50/96** | 10 | 19 | 21 | 67.71% | `2.365` |
| Post-fit `fisher_ewc_a1e+07` (`terminal_mse`) | 87.09% | `-0.00069` | **51/96** | 10 | 20 | 21 | 68.75% | `2.360` |
| In-fit `jacobian_anchor_r0.5` (uncentered bias) | **88.44%** | `-0.01439` | **55/96** | 14 | 20 | 21 | 72.92% | `2.480` |
| In-fit `euler_centered_r0.5` (same 32 raw WikiText rows) | **88.39%** | `-0.01364` | **57/96** | 14 | 20 | **23** | **75.00%** | `2.529` |
| In-fit `euler_chatmix_r0.35` (same 32 WikiText rows + chat wrapper) | **88.41%** | `-0.01283` | **63/96** | **20** | **21** | 22 | **76.04%** | **`1.117`** |

### Key takeaways

1. **Unanchored post-fit trajectory optimization overfits calibration.**
   `unanchored_a0` reduced calibration `teacher_kl` by `9x` (`0.01749 -> 0.00190`),
   but held-out diagnostic KL worsened (`0.05812 -> 0.06488`), 512-token top-1
   agreement dropped (`87.21% -> 86.68%`), and task accuracy stayed at `49/96`.
   Adding the measured parameter Fisher diagonal (`fisher_ewc_a1e+07`) prevented
   the text degradation and edged tasks to `51/96` (`50/96` for isotropic).
2. **In-fit Euler-Jacobian EWC anchoring (`euler_centered_r0.5`) improves both
   held-out text and tasks on the exact same 32 calibration rows.**
   Held-out diagnostic MLP reconstruction error dropped from `0.8254` to
   `0.7363`, 512-token top-1 agreement rose from `87.21%` to `88.39%`, and task
   accuracy rose from `49/96` to `57/96` (`43/64` vs. exact `44/64` on
   PIQA + ARC-Easy).
3. **Exposing calibration to the tokenizer's chat wrapper (`euler_chatmix_r0.35`)
   recovers BoolQ (`10/32 -> 20/32`) and reaches `63/96` overall.**
   Using only the existing 32 training-split WikiText-2 passages from
   `fit.jsonl` wrapped in Qwen3's chat/choice template woke up the dormant
   chat-boundary channels in layer 13, cutting task label KL in half
   (`2.304 -> 1.117`) while preserving `88.41%` 512-token text agreement.
