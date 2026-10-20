# Qwen3 compression review — 2026-10-20

Reviewed HEAD `77f7ac347128d45b792013dcec45be9917b6aa44`, recent
October 18–20 changes, implementation paths, and saved local evaluation reports.
This is an evidence review and proposed experiment, not a new model result.

The evidence does not establish that training-scale data is the only remaining
lever. There are coverage, target-definition, metric, and experimental-control
issues to resolve first. Keep `joint_waterfill_286mib` as the development
reference; neither its small task panel nor the newer failures qualify a
general compression claim.

## Findings that change the interpretation

### 1. The documented refinement slices never reach chat batches

`scripts/build_joint_waterfill_qwen3.py::_chatmix_batches` returns all original
batches followed by all wrapped batches. With 32 originals, indices 0–31 are
plain text and 32–63 are chat. `scripts/joint_refine_modal.py` slices this
concatenated list by `batch_offset`.

Consequently, the documented boundary-model sessions covering 0–23 contain no
chat examples; even covering 0–31 would not reach chat. Executing the actual
helper with tagged inputs reproduced this ordering. This does not mean the
initial bottleneck build lacked chat: that builder consumes the full list.

Fix: explicitly stratify plain/chat examples in every optimization step, with
coverage counters and persistent optimizer state. Accumulate gradients across
microbatches to avoid retaining all forward graphs simultaneously. Log actual
batch identities and counts rather than describing the input file as mixed.

The generated chat examples also always append `A`, and the four-choice
template can contain multiple statements from the passage. Use unambiguous
synthetic questions with balanced/permuted option positions. Teacher-KL uses
teacher distributions, so appending A is not equivalent to a hard A target;
it still creates narrow context coverage. This is a hypothesis to test, not
proof that the template caused the collapse.

### 2. The boundary-exact model stops being boundary-exact during refinement

`scripts/joint_refine_modal.py::_selected` selects every gate/up/down weight,
including layers 0, 1, 2, and 27. Calling that selector with representative
parameter names confirms all 12 boundary arrays are selected.

Fix: select compressed layers explicitly and assert boundary tensor identity
before/after training. Unfreezing the boundaries can be a separate experimental
arm. The current run preserves boundary architecture, not frozen boundary
functions. Freezing them also removes roughly 151M scalars from optimizer state.

### 3. Bottleneck and chart errors use different units and precision

`build_bottleneck_swiglu.py:102` reports normalized squared error, before BF16
export. `mlx_qwen3_fit.py:629` reports its square root, after deployment-precision
chart evaluation. Thus bottleneck `0.06` means approximately `0.245` relative
L2 error, and `0.40` means approximately `0.632`.

Fix: report both NMSE and relative L2, evaluate the exported runtime, and separate
fit rows from untouched diagnostic rows. The claimed "~0.4 per layer" is not a
consistent measurement across the two representations.

### 4. The bottleneck target mixes two distinct learning problems

The builder captures all teacher outputs once, then fits each layer using the
progressively changed student's inputs. For a residual mutation M:

    current target:        M_teacher(h_teacher)
    local operator target: M_teacher(h_student)
    state repair target:   h_teacher_after - h_student_before

Here M includes the relevant normalization and residual-delta semantics.
The current target is a legitimate teacher-feature regression target, but it is
neither a shared-input operator test nor an explicit residual-state repair.
Adding an old teacher delta to an already displaced student state does not
restore the teacher state.

Fix: compare the latter two targets on one layer at an identical parameter
budget, and report both local operator fidelity and downstream token response.
For state repair, target a precisely specified boundary; do not confuse raw
MLP output, residual delta, and terminal-normalized state.

### 5. The anchor does not demonstrate retention across slices

Each invocation recomputes observed-token diagonal Fisher on its current slice;
`optimize_trajectory` anchors to the current input checkpoint and initializes
fresh Adam state. It does not persist a fixed reference/precision across
sessions. Stable initial loss on a *new* slice is not a forgetting test:
forgetting requires re-evaluating previously learned slices.

Fix: retain a fixed, independently measured anchor if testing EWC; maintain a
previous-slice loss matrix and separately log KL, anchor penalty, gradient norms,
and their relative contributions. A coefficient such as 100 has no portable
meaning when the penalty is divided by parameter count.

Observed-label empirical Fisher is also not the conditional categorical Fisher
of the model's output distribution. This distinction is documented in
[Kunstner et al.](https://papers.nips.cc/paper_files/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html).
Use the latter for a distributional sensitivity diagnosis; do not infer
functional preservation from a diagonal parameter penalty alone.

### 6. Several causal conclusions exceed their measurements

- Testing 4×16 and 8×32 charts does not establish impossibility at every density.
  Input-dependent memberships are nonlinear; nearly constant memberships can
  make a particular fit effectively affine. Measure router entropy, membership
  variation, and feature rank before attributing failure to the whole class.
- Local relative errors do not simply multiply across residual layers.
  Downstream Jacobians, cross-layer reinforcement, and cancellation determine
  the final distortion.
- A mute-impact decrease measures sensitivity of the *current model* to that
  mute. It does not alone prove attention output is noise or identify a unique
  cause. Compare shared-state operator replacement and restoration interventions.
- `layer_vitality_scan.py:69` takes the maximum over every position, despite
  naming the result `token0_sink_max_abs`. Index position zero explicitly and
  report other positions separately. Matching one maximum amplitude does not
  establish matching sink direction, attention routing, or behavior.

The saved boundary-refined evaluation reports 37.10% text top-1 agreement,
NLL increase +2.3377 on 32,704 targets, and A for all 96 task questions. Its
56/96 task score equals the panel's A-label count. These are existing saved
measurements inspected in this review, not freshly rerun inference.

## The mathematical experiment to run next

Use one reference trajectory and introduce replacement strengths alpha_l at
explicit residual mutation boundaries. At each live input state, interpolate
between native and surrogate mutations; this defines exact endpoints.
For target position t, let:

    v_l,t = d z_t / d alpha_l at alpha = 0
    F_t   = diag(p_t) - p_t p_t^T
    G_lm  = mean_t [v_l,t^T F_t v_m,t]

Then, locally:

    KL(p_reference || p_alpha) ≈ 1/2 alpha^T G alpha

Calculate each product without a vocabulary-square matrix:

    v_l^T F v_m = sum_k p_k (v_l,k - E_p[v_l]) (v_m,k - E_p[v_m])

The diagonal measures single-replacement damage; off-diagonal entries measure
reinforcement or cancellation. Preserve token-position detail, especially chat
boundaries, rather than averaging away rare sensitive states.

Start with one layer, two adjacent layers, and two separated layers. Compare
the quadratic prediction to exact finite KL at alpha 0, 1/4, 1/2, and 1.
Use a float32 derivative path with TF32 disabled; separately measure deployed
BF16 endpoints. Check endpoint parity and future-to-past causal zeros.

Interpretation:

| Result | Justified next change |
|---|---|
| One replacement already causes large token Fisher damage | Improve neuron selection, target, or retained width locally |
| Singles are safe, pairs have large positive cross terms | Allocate compression jointly across interacting layers |
| Small-alpha predictions hold but fail at full replacement | Use smaller steps and remeasure on the changed trajectory |
| Shared-input fit is good but changed-state fit fails | Fit on student states and constrain downstream response |
| Corrected optimization fits calibration but fails untouched contexts | Broader data becomes an evidenced next lever |

## Candidate solutions after the diagnosis

1. **Progressive, interaction-aware compression.** Start from exact or wf286,
   introduce one replacement at a time, accept using finite token-distribution
   distortion, and refresh sensitivities after acceptance. A uniform jump from
   6,144 to 512 channels removes 91.7% of every selected MLP's intermediate
   channels; the same width need not be appropriate everywhere.
2. **Function-sensitive channel selection.** Current importance is activation
   second moment times the squared Euclidean norm of the down column. Compare
   it with downstream pullback-Fisher damage of removing that channel, then
   account for redundancy among selected channels. Gradient-informed structural
   selection has precedent in [LLM-Pruner](https://arxiv.org/abs/2305.11627);
   its results do not validate this Qwen3 implementation or compression ratio.
3. **A nonlinear core plus a small residual correction.** Preserve selected
   SwiGLU channels and fit a low-rank/chart correction to the remaining
   shared-input error. Select the correction with downstream response geometry,
   keep total parameter cost fixed, and compare against spending the same
   budget on additional SwiGLU channels.

Prioritize the coverage/control fixes and bounded response experiment before
another 24-layer refit. These proposals remain hypotheses until measured.

## Verification performed

- Reviewed current source and saved local evaluation JSON.
- Executed the actual batch helper and parameter selector with lightweight
  tagged inputs, without loading model weights.
- Focused test selection: 7 passed; 37 MLX/Metal-dependent tests skipped because
  Metal was unavailable in this execution environment.
- No model retraining, new full-model inference, or model artifact modification.
- Preserved the pre-existing untracked compiled-cost report.
