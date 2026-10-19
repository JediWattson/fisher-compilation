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

## 3. Scaling across multiple blocks (`285.85 MiB` saved)

Using the Euler-Jacobian EWC anchor and the chat-wrapped training calibration
passages, we evaluated three multi-block compression arms sequentially along
the changed trajectory:

- `attn_4blocks_7_13_19_25`: reduced attention (`1024/512`) across 4 distributed
  blocks (`7, 13, 19, 25`), removing `16,777,216` scalars (`32.00 MiB` saved).
- `joint_4blocks_7_13_19_25`: reduced attention (`1024/512`) + Euler-anchored
  chart MLP (`4` charts, `rank 16`, `ridge 0.35`) across blocks `7, 13, 19, 25`,
  removing `149,868,000` scalars (`285.85 MiB` saved, `8.71%` of the 1.72B model).
- `attn_all28`: reduced attention (`1024/512`) across all 28 blocks (`0..27`)
  with frozen MLPs, removing `117,440,512` scalars (`224.00 MiB` saved).

| Multi-block Variant | BF16 Storage Saved | 512-Token Top-1 Agree (`32,704`) | 512-Token NLL Change | Task Total (`96`) | BoolQ (`32`) | PIQA (`32`) | ARC-Easy (`32`) | Task Teacher Agree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `attn_4blocks_7_13_19_25` | `32.00 MiB` | **90.23%** | `+0.03536` | **70/96** | **24** (100% agree) | 20 | **26** (100% agree) | **95.83%** |
| `joint_4blocks_7_13_19_25` | **`285.85 MiB`** | 69.70% | `+0.28106` | **64/96** | **21** | 19 | **24** | **82.29%** |
| `attn_all28` (uniform `1024/512`) | `224.00 MiB` | 63.07% | `+1.08999` | **53/96** | 17 | 20 | 16 | 60.42% |
| `attn_body24_3_to_26` (`3..26` `1024/512`) | `192.00 MiB` | 76.68% | `+0.14174` | **63/96** | 18 | 20 | **25** | 78.12% |
| `attn_waterfill_224mib` (cov KKT water-fill) | `224.00 MiB` | 70.28% | `+0.30833` | **70/96** | **25** | **22** | 23 | 76.04% |
| **`attn_pullback_wf_224mib`** (pullback KKT, `r=0.05`) | **`224.00 MiB`** | **73.68%** | **`+0.12273`** | **67/96** | **25** | 19 | 23 | **80.21%** |
| **`attn_pullback_wf_r0.25_224mib`** (pullback KKT, `r=0.25`) | **`224.00 MiB`** | **75.60%** | **`+0.09495`** | **61/96** | 19 | 19 | 23 | **79.17%** |

Sequential trajectory adaptation across the 4-block joint slice
(`joint_4blocks_7_13_19_25`, `64/96` tasks, `82.29%` teacher agreement) slightly
outperformed the single-block fit (`63/96`, `76.04%` agreement) while saving
`4x` as much memory (`285.85 MiB`), because later fitted chart MLPs (`19, 25`)
absorb upstream trajectory drift from earlier blocks (`7, 13`). By contrast,
reducing all 28 attention blocks uniformly with `(1024, 512)` (`attn_all28`,
`224.00 MiB` saved) accumulated `+1.09` NLL drift and dropped to `53/96` tasks.

## 4. Generalized Fisher Rate–Distortion Water-Filling (`allocate_waterfill_attention_ranks`)

Tracing the residual stream layer-by-layer showed that `mlp.2` creates a
`13,376`-amplitude massive activation on Token 0 (`||h_3[0]|| = 15,815.5` vs.
`28.2` median on tokens `1..127`) that serves as the model's attention sink
through Layer 26 before `mlp.27` cancels it. Measuring the dimensionless
Pullback-Fisher energy $\Omega_l = \mathbb{E}[a_l^T F_l a_l]$ across all 28
layers ranked Layers `0, 1, 2` as the top three layers (`9.51, 1.36, 1.19` vs.
`0.10–0.20` in Layers `21..27`).

Solving the KKT Lagrangian water-filling allocation
(`allocate_waterfill_attention_ranks`) at the **exact same `117,440,512`-scalar
(`224.00 MiB`) reduction** as `attn_all28`:
- Automatically left Layer 2 completely exact (`(2048, 2048)`), preserved
  unfactored exact $W_O$ (`r_out = 2048`) on the high-$\Omega_l$ early layers
  (`0..7, 9..11`), and compressed the low-$\Omega_l$ late layers (`21..27`)
  more deeply (`r_in ~ 384–544, r_out ~ 448–608`).
- Predicted a **`3.45x` reduction in theoretical Pullback-Fisher distortion**
  (`0.6101 -> 0.1771`), which matched the **measured `3.54x` reduction in
  512-token held-out NLL drift** (`+1.0900 -> +0.3083` across 32,704 targets)
  and raised task accuracy from **`53/96` to `70/96`** (`+17` questions, with
  task target NLL cut from `2.883` to `1.310`).

