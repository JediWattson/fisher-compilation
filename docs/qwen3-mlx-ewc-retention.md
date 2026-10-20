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
| `joint_4blocks_7_13_19_25` (hand-picked `7,13,19,25`) | `285.85 MiB` | 69.70% | `+0.28106` | **64/96** | 21 | 19 | **24** | **82.29%** |
| **`joint_waterfill_286mib`** (joint KKT water-fill) | **`285.85 MiB`** | **73.47%** | **`+0.22875`** | **70/96** | **27** | **20** | 23 | 78.12% |
| `attn_all28` (uniform `1024/512`) | `224.00 MiB` | 63.07% | `+1.08999` | **53/96** | 17 | 20 | 16 | 60.42% |
| `attn_body24_3_to_26` (`3..26` `1024/512`) | `192.00 MiB` | 76.68% | `+0.14174` | **63/96** | 18 | 20 | **25** | 78.12% |
| `attn_waterfill_224mib` (cov KKT water-fill) | `224.00 MiB` | 70.28% | `+0.30833` | **70/96** | **25** | **22** | 23 | 76.04% |
| **`attn_pullback_wf_224mib`** (pullback KKT, `r=0.05`) | **`224.00 MiB`** | **73.68%** | **`+0.12273`** | **67/96** | **25** | 19 | 23 | **80.21%** |
| **`attn_pullback_wf_r0.25_224mib`** (pullback KKT, `r=0.25`) | **`224.00 MiB`** | **75.60%** | **`+0.09495`** | **61/96** | 19 | 19 | 23 | **79.17%** |
| **`joint_waterfill_512mib`** (8 chart-MLP + L27 attn) | **`512.20 MiB`** | **61.07%** | **`+0.61101`** | **67/96** | **28** | 17 | 22 | **75.00%** |
| **`joint_waterfill_768mib`** (6 chart-MLP + all-28 attn `384`) | **`769.46 MiB`** | 43.78% | `+1.50073` | **58/96** | 27 | 19 | 12 | 48.96% |

### Budget-frontier finding (`allocate_waterfill_joint_ranks`)

Sweeping the joint KKT budget at fixed parameterization (`4` charts, `rank 16`,
`min_rank=384` attention floor, marginal-consistent MLP selection):

| Budget removed | MLPs charted | Attn layers touched | Task | Text agree | NLL $\Delta$ | Weights | Body MACs |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `285.85 MiB` | `13,15,18,25` | `5` (`r_in` 704–960) | **70/96** | **73.47%** | `+0.229` | `2.926 GiB` | `89.4%` |
| `512.20 MiB` | `8,11,13,15,18,21,23,25` | `1` (L27 `r_in=981`) | **67/96** | **61.07%** | `+0.611` | `2.705 GiB` | `80.9%` |
| `769.46 MiB` | `13,15,18,21,23,25` | **all 28** (`r_in=384` floor) | **58/96** | 43.78% | `+1.501` | `2.453 GiB` | `78.6%` |

Two structural findings:

1. **The allocator's marginal-consistency pass works as designed** — at 768 MiB it
   *rejected* `mlp.0`/`mlp.27` (Ω = 5.59/3.34) even though raw distortion-per-scalar
   ranking would have included them, because their cost exceeded the attention
   water-filling threshold μ. Without the interleave fix the schedule was degenerate.
2. **The fidelity knee is ≈512 MiB for this parameterization.** Past it, attention
   ranks hit the `384` floor on all 28 layers including the stem (`L0–2`), which
   empirically destroys the token-0 sink as predicted by earlier layer-tracing
   (ARC-Easy collapses to `12/32`, teacher agreement `48.96%`). The quadratic
   distortion model is not trustworthy below ~`rank/8`; deeper compression needs a
   *different parameterization* (lexical factorization frees ~`443 MiB` without
   touching the stem; trajectory-target fitting; cross-block transports).

### Layer-vitality scan (causal mute-impact, `scripts/layer_vitality_scan.py`)

Muting `attention.{l}.delta` / `mlp.{l}.delta` one at a time and measuring
next-token KL on 8 prompts (mean KL, `exact` vs `wf286` vs `wf768`):

| Layer | `attention` exact → wf286 → wf768 | `mlp` exact → wf286 → wf768 |
| --- | --- | --- |
| `0` | `20.16 → 17.00 →` **`8.71`** | `12.58 → 9.93 →` **`5.57`** |
| `2` | `0.12 → 0.08 → 0.06` | `10.93 → 9.18 →` **`8.02`** |
| `6` | `0.15 → 0.12 →` **`0.03`** | `0.14 → 0.19 → 0.07` |
| `13` | `0.05 → 0.02 →` **`0.00`** | `0.09 → 0.04* → 0.01*` |
| `21` | `0.03 → 0.03 →` **`0.01`** | `0.13 → 0.14 → 0.02*` |
| `26` | `0.02 → 0.03 → 0.04` | `0.21 → 0.45† → 0.22` |
| `27` | `0.03 → 0.04 → 0.05` | `0.18 → 0.27† →` **`0.50†`** |

`*` chart-MLP layer (286: `13,15,18,25`; 768: `13,15,18,21,23,25`)
`†` impact *grows* vs exact — downstream exact MLPs are recruited to compensate
for upstream chart approximations (modal redistribution).

Two falsified/clarified hypotheses:

1. **The token-0 sink does NOT die at 768 MiB** (`max_abs` `12,352` exact →
   `13,128` wf768 — slightly *larger*). Stem `r_in=384` starves the *queries/keys*
   the sink feeds, not the sink itself.
2. **The actual 768 failure mode is global attention-bandwidth collapse**: every
   layer's mute-impact falls `3–10×` below exact (mid/late attention `0.004–0.06`
   vs `0.03–0.15`), i.e. the attention blocks emit near-noise the residual stream
   mostly ignores — the model degenerates to *embeddings + weakened MLPs +
   terminal block*, which produces the calibration-surface repetition seen in chat.

### Fully-modal MLP stack experiment (`modal_mlp_*`)

Testing the modal-graph hypothesis directly — *exact attention skeleton + chart
MLPs instead of SwiGLU*:

| Variant | MLPs charted | Scalars removed | Task | Text agree | NLL $\Delta$ | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `modal_mlp_all28` | all 28 | `931.6M` (`1,777 MiB`) | 51/96† | 24.50% | `+4.731` | token-0 sink **dead** (`max_abs` `20` vs `12,352`): `mlp.2` chart `rel_err=0.846` |
| `modal_mlp_body24` | `3..26` (boundary `0,1,2,27` exact) | `798.5M` (`1,523 MiB`) | 56/96† | 33.73% | `+2.280` | body charts carry `~1/10` of a layer's function; 4 boundary MLPs hyper-recruited (`mlp.27` mute-impact `0.557` vs `0.269` on 286) |

`†` Both predict `"A"` for all 96 questions (`prediction_label_counts={"A":96}`):
the task scores are the degenerate A-count of the panel, not real accuracy.

**Verdict**: a fully-modal MLP stack does not hold fidelity at `4` charts /
`rank 16` (~30–40% per-layer relative error each — `24` sequential ~40%
approximations compound multiplicatively through the residual carrier). The
working `286/512` builds succeeded *because* interleaved exact MLPs absorbed
the error — the modal blocks need surviving dense neighbors to recruit.

### Gemma-parity retry: + transports + terminal refit (`modal_*_v2`, `modal_all28_seq`)

The `modal_mlp_*` failure lacked the two mechanisms that made Gemma3-270M's
fully-modal stack work — cross-block transport edges and terminal refit — so
`scripts/augment_modal_transports.py` was built to add them: one
`attention.l → mlp.l` edge per chart, ridge-fitted so that
`Σ_c membership_c·(x_a @ T_c) @ mode_dec_c` closes the local gap
`teacher_mlp_output − student_chart_output` on the student trajectory
(trajectory-target fitting toward the teacher's residual correction), plus a
closed-form refit of the last MLP against `teacher_terminal − pre_state`.

| Variant | Fitting | Task | Text agree | NLL $\Delta$ | Token-0 sink |
| --- | --- | ---: | ---: | ---: | ---: |
| `modal_body24_v2` | parallel (all edges vs pre-transport trajectory) | 55/96† | 29.76% | `+4.31` | `11,904` (alive) |
| `modal_all28_v2` | parallel | 56/96† | 5.16% | `+10.82` | `21,120` (**revived**, overshooting `12,352`) |
| `modal_all28_seq` | sequential (each edge vs post-transport trajectory) | 56/96† | 1.13% | `+9.90` | `9,024` (revived, well-calibrated) |

`†` Still degenerate constant predictors (`{"A":95+}`).

Per-layer local-residual closure is strong on paper — `attn.2→mlp.2` closed
`892 → 13` (1.5%) in sequential mode, literally learning to re-emit the sink
signature — but it is **globally unstable**: each of the 112 edges maximally
closes its own gap, and membership-gated rank-17 injections accumulate
residual-stream energy the teacher never had (`terminal` RMS `54`,
`max|x| 1,631` vs typical ~1). Top-1 stays plausible-but-generic (`' the'`):
direction roughly sane, magnitude blown.

**Verdict**: Gemma's mechanisms do not rescue the Qwen3 modal stack at this
parameterization — not because transports can't carry critical signatures
(the sink revival proves they can), but because per-layer closed-form fitting
has no global consistency constraint. Gemma's transports worked because they
were **jointly fitted against terminal loss** in the whole-model experiment.
The remaining levers are (a) joint gradient refinement over all params
against terminal KL — real training rather than closed-form compilation — or
(b) denser charts (`8–16` charts at `rank 32–64` for ~10% per-layer error) so
that less correction is needed in the first place.

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

