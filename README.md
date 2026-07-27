# Fisher Graph Transformer

A small decoder-only transformer built to expose its computation, derive
activation-space Fisher modes, and replace transformer blocks with graph
executors.

This repository now contains a complete reference run:

- a trained two-pair associative-recall transformer;
- explicit activation and gradient capture;
- full 32 x 32 empirical Fisher matrices at six residual-stream boundaries;
- eigendecomposed, reusable width-wise compute modes;
- activation-covariance-aware native, variance-weighted, and generalized
  Fisher codecs with distinct dual encoder/decoder bases;
- a position-conditioned Fisher-mode intervention "equalizer";
- a groupwise Fisher/output-aware dense-supermode compiler that rewrites
  \(K\) gated-MLP units as \(R<K\) ordinary contiguous units, omits every
  analysis codec from deployment, and records raw rate-distortion points;
- held-out necessity, control, localization, and sufficiency sweeps;
- position-coupled modal layer Jacobians;
- a bounded true-forward-JVP probe plus causal Fisher-weighted prefix SVD and
  signed factor executor;
- a residual-separated, variable-length gated causal modal executor with an
  explicit same-position path and state-conditioned positive-lag experts;
- a Fisher-need conditional-budget routing milestone with route-specific mode
  masks, hard token grouping, static/position/shuffle controls, and an
  explicit native-output-oracle boundary;
- a variable-layout, variable-length routing gate plus a source-free
  hard-routed block-executor interface that is deliberately left unfitted when
  the gate fails;
- a query-sparse full-transformer-span gate with causal-prefix routing,
  pointwise/metadata/shuffle controls, exact native-versus-graph accounting,
  and fail-closed model-fitting escalation;
- a source-independent static mini-transformer progression from the
  exploratory rank-14 result to a clean rank-24 V2 executor that replaces the
  complete three-block span, passes five-seed selection and fresh validation,
  records zero source calls, and separates ideal from dense-reference work;
- a position-conditioned modal bottleneck and standalone causal modal executor;
- causal conditional completion of discarded boundary modes;
- independently compiled layer-0 and layer-1 executors composed into a frozen
  two-layer modal runtime;
- a compact seven-tensor fused executor whose inspectable logical graph loads
  lazily for activation capture or interventions, with measured end-to-end
  CPU speedups;
- a packed causal-pair triangular reference derived from that authenticated
  runtime, with validation-gated arithmetic, storage, and latency measurements;
- an optional MLX lowering plus a custom Metal kernel that consumes packed
  causal rows directly on Apple Silicon, with a same-GPU benchmark;
- a trainable, sequence-length-independent causal modal executor for dynamic
  prefill, with explicit padding, logical-position, and visibility guards;
- a nonmutating mixed runtime that dispatches each manifested segment to a
  compiled executor or its source fallback;
- mixed-length boundary distillation utilities with separate selection
  metrics, plus a typed bridge that computes Fisher gradients at source or
  compiled activation sites;
- reproducible checkpoint, split, training, Fisher, intervention, and executor
  artifacts.

The checked build reached 100% validation and test accuracy. Its summarized
results are in
[`artifacts/associative_recall/fisher_report.md`](artifacts/associative_recall/fisher_report.md)
and
[`artifacts/associative_recall/intervention_report.md`](artifacts/associative_recall/intervention_report.md).
The layer-replacement result is in
[`artifacts/associative_recall/modal_executor_report.md`](artifacts/associative_recall/modal_executor_report.md).
The boundary-recovery result is in
[`artifacts/associative_recall/modal_completion_report.md`](artifacts/associative_recall/modal_completion_report.md).
The independently compiled layer-1 result and frozen two-layer composition are
in
[`artifacts/associative_recall/modal_executor_layer_1_report.md`](artifacts/associative_recall/modal_executor_layer_1_report.md),
[`artifacts/associative_recall/modal_completion_layer_1_report.md`](artifacts/associative_recall/modal_completion_layer_1_report.md),
and
[`artifacts/associative_recall/modal_composition_report.md`](artifacts/associative_recall/modal_composition_report.md).
The fused runtime and benchmark are in
[`artifacts/associative_recall/fused_executor_report.md`](artifacts/associative_recall/fused_executor_report.md).
The separate Apple-Silicon accelerator measurement is in
[`artifacts/associative_recall/mlx_metal_benchmark.md`](artifacts/associative_recall/mlx_metal_benchmark.md).

Separately, the repository contains an opt-in text-only Gemma 3 adapter,
bounded-memory Fisher collection, split-stability plus exact held-out
Rayleigh replay, multi-boundary modal-trajectory tooling, and an exact-logical-
lag reverse-causal gradient predictor. The external rung now also includes
joint full-width sufficiency curves, activation-aware codec selection, bounded
true forward JVPs, a weighted causal-factor reference, and a split-safe gated
block-output experiment. A fresh target-informed projection-only rank ladder
then isolates the representation from executor fitting. Those two follow-ups
are negative for their tested protocols: the gated graph has small resource
accounting but poor fidelity, and no generalized-decoder prefix from rank 480
through 639 passed both behavior gates. A later codimension-one sensitivity
rotation did recover behavioral fidelity at rank 639, but it removes only one
of 640 directions. A true source-independent executor was then trained inside
that span and did skip the native layers, but it failed every calibration-B
fidelity gate. A final Fisher-aware representation oracle found that total
ranks 636 and 638 passed its fresh prompt- and domain-family-disjoint,
task-form-matched B behavior gates, but a float32 endpoint-control false
negative stopped the preregistered run before validation. It is not part of the
completed toy reference run: no checkpoint or live tensor artifact is
committed, and no viable external-model compression or speed result is claimed
here.

The conditional-computation follow-up asks whether a cheap causal router can
spend a different modal budget and route-specific mode subset on each token.
Its first rung deliberately projects a native layer output, so it can isolate
conditional representation value before a specialist generator bank is
trained; it is not yet a layer replacement or a FLOP/latency result. The
strict-loaded fixed-format layer-0 run retained 100% validation accuracy with
11.535 active modes per token, but a position-only schedule nearly matched it.

The new variable-format middle-block gate removes that shortcut. Its Fisher
teacher preserved 100% calibration-B behavior, and one aggressive schedule
showed statistically significant hidden-state signal beyond the strongest
metadata control. But no learned policy passed behavior and compute gates at
the same time: the canonical router averaged 17.053 modes versus a passing
static rank 9, while the cheapest teacher-compatible schedule cut ideal
router-plus-selective-projection MACs to 0.749x static but reduced learned paired
accuracy to 45.83%. The run therefore failed closed before compiler validation
or model-level generator fitting. A source-free hard-routed executor interface
is implemented and tested, but remains unfitted. The Fisher-need teacher,
six-role A-basis/A-mask/A-router/B/validation/test protocol, controls, exact
results, and blocked Gemma escalation are described in
[`docs/conditional-computation.md`](docs/conditional-computation.md).

The full-span follow-up targets all three blocks together and routes only the
one demanded answer row while allowing the router to read every causal-prefix
row. Its A-selected Fisher teacher again preserved 100% calibration-B
behavior, but averaged 17.156 modes versus a passing static rank 14. The
learned causal router averaged 14.302 modes, narrowly missed the p90 NLL gate
(`0.101778 > 0.10`), and cost `1.49479x` the static projection after its causal
state and ideal selective projection were counted. Its NLL advantage over the
pointwise embedding ablation was only `0.002975`, with a semantic-context
bootstrap interval crossing zero. The full-span gate therefore also failed
closed: validation/test were untouched and no graph was fitted. The command is
gate-only even on a pass; fitting remains a separate step. Small hypothetical
shared graphs have large analytic headroom versus three native blocks, but
those envelopes are explicitly untrained and do not establish compression.

The next static-generator experiment tests that headroom directly. It excludes
all 120 semantic contexts consumed by the routing predecessor, fits graph
weights on 512 predecessor-fresh contexts, selects checkpoints and architecture
on two disjoint 128-context roles, and leaves a fourth 128-context calibration role
unopened *within the artifacted run* until the rank, architecture, seed, and
checkpoint are frozen. A post-run provenance audit found that an earlier
interactive prototype had already inspected 83 of those 128 nominal
calibration contexts, so this panel is exploratory rather than clean
confirmation. A one-layer width-32 control stayed near chance in all three
seeds. The two-layer width-16 graph passed the strong selection gate in only
one seed; the two-layer width-32 graph passed in two of three and was selected.

On nominal calibration C, that source-independent rank-14 graph made zero
calls to all three native blocks and retained 100% answer, paired-context, and
minimum-stratum accuracy. Its NLL was 0.058821 versus 0.049715 native
(`+0.009105`), teacher KL was 0.005243, p90 absolute per-example delta NLL was
0.024399, and top-1 agreement was exact. Thus every preregistered strong
behavior gate passed. The joint gate still failed for two reasons:
semantic-context bootstrap resampling put the 95% upper bound on mean NLL
degradation at 0.010323, just above the locked 0.010000 limit, and the
provenance audit found the prior-probe overlap. Official validation and test
therefore remained untouched. Only 75 train contexts are now clean relative
to the predecessor, interactive probe, and artifacted run; consuming them
needs a separately frozen confirmation protocol.

The deployable executor stores 19,118 floating coefficients versus 25,632
parameters in the replaced source span (`0.745865x`). Ideal valid-prefix
matrix work, including the shared answer head, is `0.711426x` native. The
current dense PyTorch reference trunk is much less sparse: its issued matrix
shapes are estimated at `0.964746x` native. These are coefficient and MAC
counts, not a latency or kernel-speed measurement. The result establishes
that the missing second relational stage can generate a high-fidelity
whole-span answer delta inside the Fisher subspace. Because the panel is
exploratory and its bootstrap gate also failed, it is not yet a
validation-backed compression result.

The clean V2 replication expands the source task from eight to ten keys and
values, then excludes every semantic context that appeared anywhere in the
original task. This leaves 1,986 fresh train contexts, 246 fresh validation
contexts, and 250 fresh test contexts. The development projection ladder
rejected rank 14 and 18 despite exact top-1 behavior: their NLL degradations
were +0.047595 and +0.020099, while rank 24 reduced the degradation to
+0.001793 and passed every ceiling gate.

The frozen executor has three causal layers, hidden width 24, four heads, and
feed-forward width 48 behind a rank-24 Fisher decoder. It trains for at most
3,200 steps with modal-MSE/cross-entropy/teacher-KL weights
0.05/0.25/4.0, no label smoothing, and a 0.995 learned-parameter EMA.
Checkpoints are ranked after strong/minimum pass by teacher KL, p90 absolute
NLL error, absolute mean NLL degradation, and a later-step tie-break. All five
declared seeds passed the strong selection gate.

On 256-context confirmatory calibration B, the replacement was exact across
answer, paired-context, layout, length, query, order, new-key/new-value, and
identity strata. Its NLL was 0.045733 versus 0.049742 native
(`-0.004009`), teacher KL was 0.003536, and p90 absolute delta NLL was
0.013739. The 10,000-sample semantic-context bootstrap interval for mean NLL
degradation was [-0.004472, -0.003568]. On the one allowed evaluation of 246
fresh validation contexts, behavior remained exact: NLL was 0.045992 versus
0.050580 native (`-0.004588`), KL was 0.004506, p90 was 0.015021, and the
bootstrap interval was [-0.005271, -0.003967].

The direct and reloaded executor made zero calls to all three source blocks.
Artifact replay, direct-versus-boundary execution, future invariance, and
rank-24 span-membership audits passed. The executor stores 16,824 runtime
coefficients, `0.656367x` the replaced span. Including the shared model shell,
the compiled deployment is 19,064 parameters versus 27,872 native
(`0.683984x`). Ideal valid-prefix complete MACs are `0.607037x` native; the
dense-reference shape estimate is `0.834063x`.

Fresh executor test remains hash-only and was not evaluated. The source-model
training checkpoint separately records a prior native test evaluation on its
ordinary 405-context split: 100% accuracy and NLL 0.050810. That disclosure is
not an executor test result. V2 is therefore a validation-backed structural
compression result for this query-sparse associative-recall task, not a
general language-model result. Neither MAC estimate is a measured latency,
energy, or sparse-kernel speedup.

## Optimization summary

[![Three-panel optimization summary comparing arithmetic, CPU latency, and resident tensor storage](docs/images/fused-executor-optimization.svg)](docs/images/fused-executor-optimization.svg)

The committed SVG is generated directly from the authenticated fused-executor
report. Its source hash is embedded in the image, and the test suite rejects a
stale figure if the benchmark JSON changes. The packed triangular line is a
measured in-memory PyTorch reference. It is not the serialized default backend
or an MLX/Metal kernel.

## Reproduce the build

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

fisher-graph-experiment
fisher-graph-conditional-rank
fisher-graph-variable-associative-train \
  --output .local-runs/variable-associative/checkpoint.pt
fisher-graph-variable-conditional \
  --checkpoint .local-runs/variable-associative/checkpoint.pt \
  --output .local-runs/variable-associative/layer-1-variable-conditional.pt
fisher-graph-variable-full-span \
  --checkpoint .local-runs/variable-associative/checkpoint.pt \
  --output .local-runs/variable-associative/full-span-conditional.pt
fisher-graph-variable-static-full-span \
  --checkpoint .local-runs/variable-associative/checkpoint.pt \
  --predecessor .local-runs/variable-associative/full-span-conditional.pt \
  --output .local-runs/variable-associative/static-transformer-full-span.pt
fisher-graph-variable-associative-train \
  --n-keys 10 \
  --n-values 10 \
  --split-seed 26071 \
  --output .local-runs/variable-associative-v2/checkpoint.pt
fisher-graph-variable-static-full-span-v2 \
  --checkpoint .local-runs/variable-associative-v2/checkpoint.pt \
  --hypothesis-artifact \
    .local-runs/variable-associative/static-transformer-full-span.pt \
  --output \
    .local-runs/variable-associative-v2/static-transformer-full-span-v2.pt
fisher-graph-intervene
fisher-graph-modal-executor
fisher-graph-modal-completion
fisher-graph-modal-executor --layer-index 1 --routing-widths 4 6 8 12 16 24
fisher-graph-modal-completion --layer-index 1
fisher-graph-modal-compose
fisher-graph-fuse
fisher-graph-plot-optimizations
fisher-graph-verify
python -m pytest -W error
```

The V2 entry point is fail-closed: its scientific recipe is fixed, existing
outputs are never overwritten, and exclusive receipts guard calibration and
validation access.

On Apple Silicon, install the optional MLX backend and run its separate
same-device benchmark with:

```bash
pip install -e ".[dev,mlx]"
fisher-graph-benchmark-mlx \
  --output artifacts/associative_recall/mlx_metal_benchmark.json
```

To try the external Gemma 3 270M scaling rung, first accept the model's
[Gemma license on Hugging Face](https://huggingface.co/google/gemma-3-270m).
The model remains in an external Hugging Face cache and is never copied into
this repository:

```bash
pip install -e ".[dev,gemma]"
fisher-graph-gemma-fisher --check-paths-only
hf auth login

fisher-graph-gemma-fisher \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --prompts examples/gemma3_prompts.txt \
  --layer-index 0 \
  --max-length 128 \
  --rank 32 \
  --sketch-rows 64 \
  --output .local-runs/gemma-3-270m/layer-0-fisher.pt
```

After that smoke path works, the next analysis-only command uses the frozen
prompt file's two calibration halves, validation split, and reserved test
split:

```bash
fisher-graph-gemma-stability \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --layer-index 0 \
  --max-length 128 \
  --ranks 8 16 24 32 48 64 96 128 \
  --sketch-rows 256 \
  --device cpu \
  --dtype float32
```

The bundled 64-prompt split is diagnostic scaffolding, not representative
language data. The command compares calibration A and B with principal angles,
builds a combined A+B basis, measures exact Fisher energy on validation in one
streaming replay, and deliberately does not evaluate the test prompts.

The next diagnostic captures every unique residual boundary around Gemma
layers 4–6, spanning sliding, global, and sliding attention. It tests whether
important subspaces stay fixed, rotate predictably, or drift in a way that a
small modal transport cannot reproduce:

```bash
fisher-graph-gemma-trajectory \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --start-layer 4 \
  --end-layer 6 \
  --max-length 128 \
  --ranks 8 16 24 32 48 64 96 128 \
  --sketch-rows 256 \
  --causal-lags 0 1 4 \
  --causal-relative-ridge 0.01 \
  --device cpu \
  --dtype float32
```

This command fits only small modal-coordinate transports on calibration A+B
and evaluates those frozen maps on validation. In addition to the original
same-position Procrustes diagnostic, it now fits the reverse-causal modal
predictor

\[
\widehat g_{\mathrm{up},s}
=
\sum_{\delta=0}^{L} g_{\mathrm{down},s+\delta} W_\delta .
\]

The requested `--causal-lags` are nested maximum-lag windows. Lag 0 is an
independently refit row-local ridge baseline; lags 1 and 4 ask whether adding
exact downstream logical positions improves held-out zero-baseline explained
energy. Missing or masked positions are not compressed into false neighbors,
and lags outside the finite attention visibility of the analyzed segment are
zeroed. The diagnostic runs on every adjacent boundary pair and on the whole
block endpoint, while retaining only bounded modal sufficient statistics.

It still does not fit a graph executor. The tracked diagnostic profile
requires boundary identifiability at both ranks 96 and 128 before calling a
low-overlap edge a rotation. A developer-local run of the earlier row-local
artifact classified the block `inconclusive_basis_not_identifiable`: the two
earliest boundaries cleared the rank-128 capture floor but not the rank-96
floor. Activation transports generalized substantially better than
same-position score-gradient transports, so that evidence is compatible with
residual-state persistence but does not establish a reusable Fisher-mode
trajectory.

The strict-loaded version-2 causal rerun also found no held-out cross-position
win at rank 128 with relative ridge 0.01. Every lag-1 score was below its
independently refit lag-0 ridge baseline, and every lag-4 score was negative,
even though calibration explained energy rose monotonically toward
0.97–0.98. Lag-4 feature condition numbers were approximately
\(4.75\times10^5\)–\(6.98\times10^5\). That is severe overfit for this prompt
set, rank, ridge, and stationary exact-lag model; it is not a rejection of
sequence-aware executors in general.

Even if a future protocol explains held-out gradients, its jointly fitted
coefficients would be predictive regression weights—not identified
per-position Jacobian blocks—and it would not be a forward executor or
compilation proof. The live report remains ignored and is not committed.

The next analysis-only command asks a simpler causal question: how much of the
frozen model's validation behavior survives when every selected layer output is
restricted to its leading Fisher coordinates?

```bash
fisher-graph-gemma-ablation \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --start-layer 4 \
  --end-layer 6 \
  --max-length 128 \
  --sketch-rows 641 \
  --retained-ranks 640 512 384 256 192 128 96 64 32 0 \
  --include-single-sites \
  --device cpu \
  --dtype float32
```

This is a top-down **sufficiency** curve, not a leading-mode ablation. For a
retained rank \(k\), it keeps the top \(k\) coordinates of a full 640-wide
calibration Fisher basis and removes the lowest \(640-k\). The primary curve
applies that projection jointly at the selected layer outputs; the optional
single-site curves are localization controls. Only valid positions are
projected, centered on the pooled calibration activation mean. Model weights
remain frozen, and validation is the only model-evaluated held-out split.

Rank 640 must pass a full-rank identity gate against the untouched model before
any lower-rank point is interpreted. Rank 0 is the opposite extreme: it
replaces every valid selected-output activation with its calibration mean. A
low-rank point that preserves NLL would support representational sufficiency
under this intervention, but it would not yet establish an executable
compression. The source transformer still computes every layer at full width;
an executor, local equivalence, variable-length behavior, and runtime gates
would still have to pass.

The strict-loaded developer run passed the identity gate but found no
lower-rank sufficiency under eigenvalue-only ordering. Baseline validation NLL
was 4.271092 per supervised token:

| Joint retained rank | Delta NLL/token | Interpretation |
|---:|---:|---|
| 640 | -0.000000428 | Full-rank numerical identity |
| 512 | +3.641870 | Top-1 agreement 0.2241 despite retaining about 99.97% of calibration Fisher trace |
| 384 | +2.954244 | Severe degradation remains |
| 128 | +3.902732 | Severe degradation remains |
| 0 | +4.369668 | Calibration-mean replacement |

All 16 validation prompts worsened at rank 512. An explicitly posthoc
validation refinement found that removing even the final mode at every site
(joint rank 639) increased NLL/token by 1.783649 and reduced top-1 agreement
to 0.3621. The rank-639 single-site deltas were +0.086312 at
`layer.4.output`, +1.093586 at `layer.5.output`, and +1.148882 at
`layer.6.output`. Because that refinement was chosen after seeing the coarse
curve, it is exploratory rather than acceptance evidence.

The activation-amplitude diagnostic explains why Fisher trace was misleading
here. Low-Fisher tail coordinates carried enormous centered activation RMS on
both calibration and validation—for example, roughly 4,257 in the final
layer-5 mode, 8,915 in the final layer-6 mode, and 5,163 in zero-based
layer-4 mode 638. A Fisher eigenvalue measures local score-gradient energy at
the native activations; it does not bound the finite effect of deleting a
huge-amplitude coordinate and sending the residual stream through RMSNorm.

An additional ad hoc, read-only validation diagnostic ranked modes by Fisher
eigenvalue times calibration modal variance. Under that crude
displacement-aware score, joint rank 639 changed NLL/token by -0.00123 with
0.922 top-1 agreement; ranks 638 and 636 changed it by +0.01224 and +0.05486
with 0.825 and 0.747 agreement, and rank 632 changed it by +0.4513. For
comparison, native Fisher ordering at rank 639 gave +1.78365 and 0.362
agreement. Per-token norm-preserving projection did not rescue the native
ordering and usually made it worse. These checks were posthoc, were not
artifacted, and are not confirmatory evidence. They only motivate a
pre-registered variance-weighted or generalized-Fisher experiment; preserving
behavior while removing one dimension per site is not meaningful compression.

Under the native eigenvalue-only curve, only rank 640 preserved behavior, so
that ordering provides no compression candidate on this small,
template-matched diagnostic corpus. This does not reject amplitude-aware or
otherwise constrained modal approaches. All derived primary artifacts remain
ignored and uncommitted, model weights stayed frozen, and the reserved test
split was not evaluated.

The activation-aware follow-up turns that posthoc observation into a
split-safe experiment:

```bash
fisher-graph-gemma-weighted-jacobian \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --start-layer 4 \
  --end-layer 6 \
  --max-length 128 \
  --retained-ranks 632 636 638 639 640 \
  --sketch-rows 641 \
  --generalized-regularization 1e-3:1e-6 \
  --generalized-regularization 1e-2:1e-5 \
  --selection-nll-atol 0.05 \
  --selection-top1-min 0.95 \
  --jacobian-max-sequences 4 \
  --jacobian-modes 4 \
  --jacobian-max-lag 4 \
  --jacobian-factor-rank 2 \
  --device cpu \
  --dtype float32
```

Calibration A alone fits activation covariance, full-width Fisher estimates,
and native, variance-weighted, and generalized codecs. Calibration B locks
the lowest-rank predeclared candidate satisfying both aggregate gates, but
only after every codec family passes its own full-width behavioral identity
control. Validation evaluates only that locked candidate, the locked family's
full-width identity, and the native full-width identity; the two identities
are deduplicated for a native lock. The reserved test split remains
parse-and-hash-only.

The strict-loaded local run locked the stronger generalized regularization
pair at joint rank 636. On calibration B it changed NLL/token by -0.003316
with 0.9643 top-1 agreement. On the protocol validation split it changed
NLL/token by +0.010285 with 0.9626 top-1 agreement. Both the locked
generalized-family and native validation identities changed NLL/token by only
\(-6.58\times10^{-8}\) with exact top-1 agreement. That validates the
non-orthogonal encoder/decoder path itself and isolates the rank-636 effect to
dropping four modes rather than a faulty coordinate round trip.

This explains why the simpler variance score did not reproduce the earlier
posthoc result under the stricter split: calibration A contains 366 valid
rows at width 640, so its Fisher has a nullspace of at least 274 dimensions.
Multiplying a zero Fisher eigenvalue by activation variance still gives zero,
and the last eight variance-ordered columns remained the native last eight.
The generalized codec's explicit Fisher floor lets activation covariance
organize that otherwise unidentified tail. The result is therefore
regularization-dependent and must be retested on larger calibration data.

The expanded forward-JVP pilot used four calibration-A sequences and a
4-by-4 projected slice of the locked modal coordinates at exact lags 0–4. It
observed zero future-position leakage and found 99.77% temporal-window
coverage *inside that slice*. The aggregate split was 95.67% stationary signed
lag mean versus 4.33% within-lag variation, but lag zero dominated it. After
excluding lag zero, only 34.93% of positive-lag energy was explained by a
constant mean and 65.07% varied by position or context. The stationary
fractions fell from 96.03% at lag zero to 45.33%, 30.41%, 19.23%, and 9.33%
at lags one through four. This motivated the fresh executor experiment below:
keep the same-position path separate and use a small state-conditioned causal
mixture for positive lags. These JVP values are neither held-out edge results
nor full-width Jacobian-energy measurements.

On the synthetic five-position Toeplitz reference, rank-two prefix factors
retained 97.30% of the chosen weighted operator energy and used 160 MACs
versus 240 for an explicitly unshared dense causal map. A natural lag-shared
map stores only 80 edge coefficients, however, while the factors store 160;
99.83% of this synthetic weighted energy was also at lag zero. Accordingly,
this is exact SVD/tail accounting for a rank-two approximation—not a parameter,
FLOP, storage, latency, or Gemma-runtime compression result. Rank 636 removes
only four of 640 coordinates at each of three sites (0.625% per site), and the
validation corpus is only 16 short template-matched prompts. Those prompts
were reused by earlier exploratory iterations, so this run is a controlled
replication rather than fresh confirmatory validation; the reserved test split
is still unevaluated. See
[`docs/weighted-jacobian-compilation.md`](docs/weighted-jacobian-compilation.md)
for the equations and remaining acceptance gates.

The follow-up fits that proposed executor against the real frozen layers 4–6
block on a new four-way prompt fixture:

```bash
fisher-graph-gemma-gated-executor \
  --weighted-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_gated_executor_prompts.json \
  --retained-ranks 320 480 \
  --expert-counts 1 2 \
  --expert-ranks 16 \
  --router-widths 16 \
  --max-positive-lags none \
  --fit-steps 100 \
  --device cpu \
  --dtype float32
```

The raw input residual is an exact bypass. The graph predicts only the block
delta in the retained output-decoder subspace, using an independent
same-position affine map plus low-rank positive-lag experts whose router sees
the query state, source state, and relative logical lag. Calibration A fits a
fixed 100-step schedule, calibration B locks one predeclared candidate, and
validation evaluates that lock once. Test remains parse-and-hash-only.

No rank-320 or rank-480 candidate passed selection. The required diagnostic
fallback locked rank 320 with two rank-16 experts and a width-16 router. Its
validation result was:

| Validation quantity | Result | Required gate |
|---|---:|---:|
| Block-delta NRMSE | 0.823388 | at most 0.20 |
| Block-delta cosine | 0.605518 | at least 0.95 |
| Delta NLL/token | +7.015665 | absolute value at most 0.05 |
| Top-1 agreement | 0.07381 | at least 0.95 |
| Stored coefficients / source block parameters | 3.2518% | at most 75% |
| Analytic MACs / source block analytic MACs | 3.2290% | at most 75% |

The low resource ratios are real accounting wins, but they do **not** make
this a viable compression: the quality gates fail by large margins. More
importantly, the rank-320 target-informed, per-token least-squares
output-subspace reference reached direct block-delta NRMSE 0.055995, yet its
intervention still changed NLL/token by +6.342280 and retained only 0.088095
top-1 agreement. This reference uses the true target delta, so it is neither
an inference-time executor nor a behavioral upper bound. It nevertheless
isolates a key failure: a small raw residual error in this codec/subspace can
be amplified catastrophically downstream. It is not enough to optimize
Euclidean block-output MSE.

The no-op intervention, full-width codec delta round trip, frozen-model guard,
prompt-disjointness, and structural causality/padding controls passed. The
experiment therefore rejects this rank-320/480, one-seed, fixed-MSE protocol;
it does not show that gated causal edges never help, that larger or differently
trained executors cannot work, or that modal compression is impossible. The
analytic MAC count also excludes nonlinearities, softmax, masking, additions,
and memory traffic, so it is not a kernel-speed measurement. See
[`docs/gated-executor.md`](docs/gated-executor.md) for the architecture,
protocol, and interpretation.

The projection-only follow-up removes executor-fit quality from that question.
At every token it uses the *true* native block delta and computes its
least-squares reconstruction in nested prefixes of the locked generalized
output decoder:

```bash
fisher-graph-gemma-projection-ladder \
  --weighted-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt \
  --gated-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-gated-executor.pt \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_projection_ladder_prompts.json \
  --device cpu \
  --dtype float32
```

No reduced rank passed calibration B's predeclared
`abs(delta NLL/token) <= 0.05` and top-1 agreement `>= 0.95` gates. The near
miss at rank 639 had delta NLL/token -0.003372 and 0.9431 top-1 agreement even
though its direct block-delta NRMSE was only 0.003633. The protocol therefore
locked rank 640 identity, which independently reached delta NLL/token
+0.000000273 and exact top-1 agreement on validation. Validation did not see
rank 639 or any other reduced candidate, and reserved test was never
tokenized or model-evaluated.

This is a strong negative result for prefix truncation of this particular
decoder span, not for every rank-\(r\) subspace. The calculation consumes the
native target delta and still runs source layers 4–6, so it is neither an
inference executor nor a parameter, MAC, storage, or latency result. See
[`docs/gemma3-270m.md`](docs/gemma3-270m.md#run-the-target-informed-projection-only-behavioral-rank-ladder)
for the full curve and claim boundary.

The preregistered codimension-one discriminator then tests whether the
rank-639 failure was caused by the codec's coordinate ordering:

```bash
fisher-graph-gemma-codimension-rotation \
  --projection-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-projection-ladder.pt \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits \
    examples/gemma3_codimension_rotation_expanded_a_prompts.json \
  --tail-width 32 \
  --stability-policy split_half_objective_regret \
  --device cpu \
  --dtype float32
```

Expanded calibration A passed its objective-regret stability gate: the pooled
direction's worst split-half regret was 6.2854% against a maximum of 10%, the
operator cosine was 0.9943 against a 0.99 minimum, and the relative eigengap
was 0.00622 against a 0.001 minimum. On calibration B, the rotated candidate
at rank 639 passed with delta NLL/token
+0.000382 and 0.9843 top-1 agreement. The preregistered source codec-prefix
rank-639 control failed with +0.016279 and 0.9303. Only the rotated candidate
was locked for validation, where it reached delta NLL/token +0.000316, 0.9913
top-1 agreement, block-delta NRMSE 0.0009415, and block-delta cosine
0.99999956. Reserved test remained parse-and-hash-only.

This supports a narrow basis-ordering result at codimension one. Retaining
639/640 dimensions is not meaningful compression, and the target-informed
projection still runs the native block. It is not an inference executor or a
parameter, MAC, storage, latency, or speed result. See the
[`Gemma 3 analysis`](docs/gemma3-270m.md) for the full protocol and caveats.

The next experiment turns that viable rotated span into a true grouped
replacement. The
[`implementation`](src/fisher_graph/gemma3_rotated_span_executor_experiment.py)
and exact-hash-disjoint
[`prompt fixture`](examples/gemma3_rotated_span_executor_prompts.json) can be
run with:

```bash
fisher-graph-gemma-rotated-span-executor \
  --rotation-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-codimension-rotation.pt \
  --model-id google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_rotated_span_executor_prompts.json \
  --device cpu \
  --dtype float32
```

This is an actual student execution path—native prefix, grouped causal
executor, native suffix—not another projection oracle. Instrumentation
recorded zero calls to native Gemma layers 4–6 in that path. The executor uses
471,057 learned parameters plus 410,880 fixed rotated-span coefficients:
881,937 stored coefficients, or 5.27446% of the source block's 16,720,896
parameters. Its analytic MAC ratio is 5.22358%.

Those resource ratios did not produce a viable replacement. On
exact-hash-disjoint calibration B, the executor reached delta NLL/token
+0.061064, 0.653302 top-1
agreement, and teacher KL/token 0.337807. Its direct block-delta diagnostics
were much closer—NRMSE 0.060264 and cosine 0.998186—but every predeclared
behavior-fidelity gate still failed. On the same split, the target-informed
rotated oracle passed with delta NLL/token +0.000659, 0.988208 top-1
agreement, and teacher KL/token 0.000240. The span therefore remained viable;
the learned executor did not approximate it precisely enough for downstream
behavior.

Because selection failed, neither validation nor reserved test was
model-evaluated. The parameter and MAC counts are diagnostic accounting only:
they do not support a compression, latency, or kernel-speed claim. See
[`the detailed protocol and result`](docs/gemma3-270m.md#run-the-true-rotated-span-grouped-executor)
for the full claim boundary. The four split roles share broad prompt-template
families, so this negative run is a paraphrase-interpolation test rather than
an unseen-family generalization test. Calibration B is consumed: do not tune
and rerun this fixture. A changed graph or loss needs wholly new
B/validation/test families. The runner refuses to overwrite an existing
artifact, and this FP64-audited reference path currently supports CPU or CUDA,
not MPS.

The Fisher-aware merged-tail follow-up asks whether the 31 surviving
low-ranked coordinates inside that rotated span can be combined into fewer
supermodes before attempting another generator:

```bash
fisher-graph-gemma-merged-supermodes \
  --rotation-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-codimension-rotation.pt \
  --model-id google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_merged_supermode_oracle_prompts.json \
  --family-manifest \
    examples/gemma3_merged_supermode_oracle_prompt_families.json \
  --supermode-ranks 0,1,2,4,8,16,24,28,30,31 \
  --device cpu \
  --dtype float32
```

This is deliberately an oracle, not a replacement executor: it reads the true
native layers 4–6 delta, preserves the 608-dimensional codec prefix, and
reconstructs the surviving tail through an A-fitted generalized-Fisher codec.
Total candidate rank is \(608+q\), with rank 639 as the authenticated rotated
span and rank 640 as native identity.

The fresh prompt- and domain-family-disjoint, task-form-matched calibration-B
sweep found real but narrow evidence for merging. Rank 636 (`q=28`) was the
smallest stable candidate passing all five behavior gates: delta NLL/token
+0.001807, 0.97717 top-1 agreement, 0.000958 teacher KL/token, 0.01673
per-prompt p90 absolute delta NLL, and 0.92593 per-prompt p10 top-1. Rank 638
also passed. Rank 624 passed aggregate top-1 but failed the per-prompt tail and
split-half stability gates; rank 632 missed the p10 gate at 0.89286 and was
unstable.

The preregistered run nevertheless failed closed before validation. Its
rank-639 endpoint agreed mathematically with the authenticated predecessor and
passed every behavior gate, but the original multi-matrix float32 replay
differed from the one-normal source projector by a maximum absolute
0.00048828125, above the predeclared \(10^{-5}\) equivalence tolerance. That is
only about \(1.12\times10^{-6}\) of the recorded boundary-output RMS, but the
control was absolute, so it failed. Native identity was exact. The full-rank
path now dispatches the authenticated one-normal formula bit-for-bit, with a
large-amplitude regression test. Projection semantics are versioned so the
completed format-1 artifact still replays its historical factorized endpoint;
new format-2 merges use the corrected path. The B artifact is not rewritten or
rerun.

Consequently q=28 is promising calibration-B evidence, not a validated
compression result. It removes only four of 640 representation dimensions
(0.625%), still consumes the native block, and establishes no parameter, FLOP,
storage, latency, or kernel-speed saving. Validation and test were never
tokenized or model-evaluated. This B split is consumed; any confirmatory run
needs a new family-disjoint protocol. The
[`detailed Gemma analysis`](docs/gemma3-270m.md#run-the-fisher-aware-merged-tail-supermode-oracle)
records the complete curve, numerical-control postmortem, strict predecessor
binding, and claim boundary.

The next implemented rung removes compression as a confound and asks whether
one source-independent graph can reproduce a single Gemma layer at the full
640-wide residual boundary:

```bash
fisher-graph-gemma-full-width-layer \
  --model-id google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --layer-index 4 \
  --prompt-splits /absolute/path/full-width-prompts.json \
  --family-manifest /absolute/path/full-width-families.json \
  --max-length 256 \
  --device cpu \
  --dtype float32
```

Calibration A computes the complete width-by-width, width-pooled empirical
ground-truth CE score-sensitivity matrix and trains both a small causal
transformer and a storage-matched attention-disabled control with its full
quadratic boundary metric plus suffix CE/KL distillation. This is not the
expected model Fisher and does not contain cross-position Fisher blocks.
Calibration B must pass behavior, local block-delta, exact native-boundary
replay, source-call, structural, and resource gates before validation is
tokenized; test remains hash-only. The prompt and family files are
intentionally not bundled because every real run needs a newly frozen,
representative, cross-role-family-disjoint corpus. No qualifying live Gemma
result is claimed yet. Candidate/source ratios are live-run diagnostics; the
self-contained loader rebuilds candidate counts and source denominators from
recorded token lengths plus an exact source-geometry manifest. It does not
reload Gemma to remeasure that manifest, so it does not promote the ratios
into parameter- or MAC-reduction claims. See the
[`full protocol`](docs/gemma3-270m.md#run-the-full-width-single-layer-replacement).

The model-aware follow-up replaces that generic mini-transformer with the
repo-owned Gemma-shaped executor. Format 5 identifies the full-width Gemma
operators directly from calibration-A activation pairs:

```bash
fisher-graph-gemma-structured-layer \
  --model-id google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --layer-index 4 \
  --prompt-splits /absolute/path/structured-prompts.json \
  --family-manifest /absolute/path/structured-families.json \
  --corpus-audit /absolute/path/structured-corpus-audit.json \
  --max-length 256 \
  --tokenization-batch-size 4 \
  --operator-bootstrap \
  --device cpu \
  --dtype float32 \
  --output /absolute/path/layer-4-structured-v6.pt \
  --calibration-b-ledger-dir /absolute/path/heldout-ledger
```

The bootstrap deterministically retains 8,192 calibration-A token rows and
recovers Q/K/V/O, gate/up/down, Q/K norm, and four residual RMSNorm
coefficients with active-support ridge or coordinate least squares. The
fitter receives activations and a destination executor, not a source module
or source parameter, and performs no optimizer or suffix-distillation
updates. The containing experiment still runs the source model to capture
those activations and bind model provenance, so "activation-only" describes
the compiler interface, not an assertion that the source model is never
loaded.

The full-width format-5 layer-4 v6 parent passed both fresh calibration B and
validation at near-numerical precision:

| Split | Block-delta NRMSE / cosine | Delta NLL/token | Teacher KL/token | Top-1 |
|---|---:|---:|---:|---:|
| Calibration B | `9.137e-7` / `0.999999999999583` | `-1.942e-8` | `-1.762e-9` | `1.0` |
| Validation | `9.213e-7` / `0.999999999999576` | `-3.137e-8` | `3.129e-9` | `1.0` |

The attention-disabled control failed strongly (`0.652747` block NRMSE,
`0.535714` top-1), while the strict-reloaded executor made zero source-layer
calls. Test remains sealed. This establishes source-free execution and
single-layer fidelity for one native-shaped Gemma layer; it is not itself a
compression result. See the
[`structured executor protocol`](docs/structured-layer-executor.md#learned-single-layer-fidelity-runner).

The first compression rung ranks each of the 2,048 complete gated-MLP units by
the calibration-A mean of `(activation * ground-truth-CE score-gradient)^2`,
keeps 1,536 paired gate/up rows and down columns, and refits only the
down-projection from activation targets. On 60,054 valid A rows it retained
`96.4940%` of that score and passed the A gate: block NRMSE was `0.015281`,
cosine `0.999883`, full-output NRMSE `0.006165`, and per-prompt
p50/p90/worst block NRMSE was `0.015234/0.016619/0.019486`.

```bash
fisher-graph-gemma-structured-mlp-build \
  --parent-artifact /absolute/path/format-5-v6-parent.pt \
  --prompt-splits /absolute/path/v6-prompts.json \
  --family-manifest /absolute/path/v6-families.json \
  --corpus-audit /absolute/path/v6-corpus-audit.json \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --device cpu \
  --dtype float32 \
  --output /absolute/path/mlp-1536-a.pt

fisher-graph-gemma-structured-mlp-heldout \
  --candidate-artifact /absolute/path/mlp-1536-a.pt \
  --prompt-splits /absolute/path/fresh-v7-prompts.json \
  --family-manifest /absolute/path/fresh-v7-families.json \
  --corpus-audit /absolute/path/fresh-v7-corpus-audit.json \
  --calibration-b-ledger-dir /absolute/path/compression-b-ledger \
  --device cpu \
  --dtype float32 \
  --output /absolute/path/mlp-1536-v7-heldout.pt
```

The heldout command must use a fresh ledger separate from the parent's ledger.
Once it claims calibration B, never rerun that consumed split—even if the
process later fails.

The candidate removes 983,040 parameters: 5,573,632 becomes 4,590,592
(`17.6373%` less for the layer). Its three MLP linear maps require 2,949,120
instead of 3,932,160 MACs per valid token (`25%` less MLP-linear work).
Including unchanged attention and norms, the fresh v7 stream analytically
measured `17.0296%` fewer complete-layer MACs for its sequence lengths.
These are exact structural/accounting savings for the rejected candidate,
not a quality-qualified deployment claim.

On the single allowed, source-corpus-disjoint v7 calibration-B evaluation,
the A-selected candidate was rejected. Block NRMSE rose to `0.071745`
(required at most `0.02`) and cosine fell to `0.997428` (required at least
`0.999`); the feed-forward branch NRMSE was `0.064834`, although the unchanged
attention branch remained near exact at `8.484e-7`. Delta NLL/token
(`-0.003317`), teacher KL/token (`0.016452`), per-prompt p90 absolute NLL
(`0.042397`), and p10 top-1 (`0.911765`) passed their gates, but aggregate
top-1 agreement was only `0.935116` against the `0.95` minimum. Compression
validation was therefore never tokenized and test remains sealed.

This selector is a diagonal, per-token Fisher/Taylor proxy, not a full Fisher
matrix: it does not model off-diagonal unit coupling, cross-token blocks, or
the interaction produced by removing many units together. The v7 rejection
means this particular one-shot `2048 -> 1536` rule is not a validated
compression method. It supports no whole-model quality, measured latency,
energy, fused-kernel, or model-level compression claim.

The new dense-supermode rung addresses that missing off-diagonal interaction
without retaining sparse slots or runtime references. It chooses a group of
\(K\) modes, derives a Fisher/output-aware dual coordinate system, rotates the
retained subspace toward rank-revealing native pivots, and trains \(R<K\)
actual gated generators. Every nonpooled unit remains exact. The deployment
executor contains only a normal dense MLP of width \(W-K+R\); the analysis
encoder, decoder, source pool, and indices are absent from executable state.
An outer experiment bundle may retain plan metadata for audit.

Its deterministic six-to-two synthetic fixture reaches roughly
`8.08e-7` MLP-output NRMSE, while redundancy-blind equal-width
diagonal-Fisher deletion plus down refit has `0.370075` NRMSE. A
structure-aware oracle deletion that keeps one representative from each
duplicate family is also nearly exact (`1.66e-7`). The test therefore proves
nonlinear grouped synthesis, strict artifact replay, exact parameter/MAC
removal, and a failure mode for scalar ranking on a constructed redundant
system; it does not establish superiority over the best pruning rule.

The first Gemma development rung then compiled a 512-to-384 pool inside the
2,048-wide layer-4 MLP, producing an ordinary 1,920-wide runtime and removing
245,760 parameters and linear MACs/token for \(d=640\). On the reused,
nonconfirmatory v9 A-guard, direct dense synthesis failed with `0.049127`
block NRMSE versus `0.021758` for equal-width diagonal-Fisher deletion and
`0.015359` for a new equal-width native-pivot pruning control. The dense
candidate was therefore rejected without a tensor artifact, and no fresh
heldout role was opened. The same analysis nevertheless improved
structure-aware pruning by 29.4% over diagonal deletion at identical resource
cost; that control passed ordinary gates but missed the stricter `0.015`
development margin by `0.000359`. See
[`dense supermode compaction`](docs/dense-supermode-compaction.md).

Compression quality is now represented as a raw rate-distortion curve rather
than a universal 100%-fidelity gate. The curve retains dominated candidates
and can compute Pareto views over parameters, runtime bytes, logical MACs, or
measured latency against downstream score, NLL, KL, top-1 agreement, or
operator NRMSE. It binds points to one evaluation split, task suite, candidate
fingerprint, resource scope, dtype/runtime, and—when applicable—latency
protocol so unlike measurements cannot share a frontier. A representative
Gemma development runner now freezes the candidate and both equal-width
controls before its reused A-guard. A scientific result still requires a newly
frozen protocol and a genuinely fresh family-disjoint guard.

The whole-model cross-block follow-up also reached a decisive boundary. It
found one development pair, layer-6 MLP unit 1202 to layer-15 unit 651, and
implemented it as a directed merged supermode: compute the earlier generator
once, carry its signed scalar through native layers 7–14, remove the later
gate/up rows, and retain the later decoder. The physical executor is bit-exact
to its activation oracle and removes exactly 1,280 learned parameters plus
1,279 net arithmetic MACs per valid token after its scalar multiply.

The preregistered 64-prompt fresh-family guard rejected the pair. The merge was
worse than deletion at the consumer MLP, layer-15 output, logits, and
native-teacher KL in every one of eight new families and all four length
bands. Fresh-data correlation fell to `-0.12935`, and the best retrospective
no-intercept scale explained only `2.49%` of deletion error. This proves the
merge/executor mechanism but not a viable compression edge; calibration B,
validation, and test remain unopened. See the
[`cross-block guard postmortem`](docs/cross-block-selective-bundling.md#physically-merged-executor-and-fresh-family-guard).

The full-model follow-up removes the earlier executor bottleneck. All 36,864
MLP coordinates across all 18 blocks are eligible, every layer pair may
propose an edge, roots may fan out without a quota, and every qualifying
consumer can be compiled into one native Gemma prefill. A sparse top-eight
proxy neighborhood remains an explicitly recorded search approximation;
materializing all 641,728,512 cross-layer pairs would not be a practical
analysis artifact.

On A-fit positions 0–39, the all-mode proxy scan shortlisted 10,645 edges.
Exact Fisher/activation replay still qualified only the original
layer-6:1202 to layer-15:651 edge. On disjoint A-fit positions 40–79, its
unweighted fit scale produced final-logit NRMSE `0.002891`, teacher KL/token
`0.00005786`, top-1 agreement `99.6569%`, and delta NLL/token `-0.000526`.
Matched deletion had `0.010538` NRMSE, `0.00022291` KL, `99.8235%` top-1, and
`-0.000950` delta NLL/token. The merge therefore improved continuous logit
fidelity on this same-family development slice while losing some discrete
top-1 agreement. It does not overturn the earlier fresh-family rejection.
Most importantly, unrestricted eligibility found no additional strict edges
and still removed only 1,280 parameters (`0.000477%` of the model). The
current limitation is discovery/generalization, not the ability to execute a
multi-edge full-model graph. Calibration-A guard, B, validation, and test were
not opened by this development run.

```bash
fisher-graph-gemma-full-model-merge-dev
```

The new modal-generator compiler implements the complete typed, checksummed
recipe from natural MLP weight groups through prompt-conditioned Fisher
coupling, cross-layer parameter clusters, exact per-layer fragments, affine
computational-mode bases, coordinate generators, causal generator interactions,
and incremental graph traversal. See
[`docs/modal-generator-compiler.md`](docs/modal-generator-compiler.md).
Checksums authenticate artifact contents and declared lineage, but the
numerical extraction and split membership used in this development rung are
caller-declared and self-attested; they do not independently authenticate
dataset membership or disjointness.

Its first live Gemma development rung selected a 54-channel fragment in
layer 17 from a 64-cluster whole-model Fisher fit. A predeclared rank-32
computational basis retained `99.1153%` of centered Fisher-weighted energy and
reconstructed the caller-declared, overlap-checked development rows at
`0.029680` weighted NRMSE. A
rank-16 generator predicted those 32 coordinates at `0.201444` weighted NRMSE
and `0.979509` weighted cosine. Primary graph lowering replaced 103,680 native
parameters with a 31,904-parameter graph node: 71,776 net parameters
(`69.23%` of the fragment, `0.0268%` of the model) and 72,448 logical linear
MACs per token (`69.88%` of the fragment) were removed. The graph executes
31,232 matrix MACs plus 672 elementwise additions per valid token. A separately
reported static fusion of that isolated node uses 21,120 parameters, 20,480
matrix MACs, and 640 bias additions per token, raising the local storage
reduction to `79.63%` and matrix-MAC reduction to `80.25%`; fusion is not the
primary graph result.

Across 10,200 supervised development positions drawn from 10,240 valid
tokens, generated execution had NLL `2.816920` versus native `2.819802`,
native-to-generated KL/token `0.001819`, and `97.8137%` native top-1
agreement. Matched deletion had NLL `2.893494`, KL/token `0.172642`, and
`83.8529%` top-1 agreement. This is strong evidence
that the learned generator is compensating rather than merely hiding a
low-impact deletion. The graph has one node and no possible interaction edge,
so it validates traversal and physical compaction but not yet learned fan-out
or fan-in. It remains a single-fragment, same-family, development-only result:
no calibration-A guard, calibration B, validation, or test data was opened,
and the logical MAC count is not a measured kernel-latency claim. A physical
end-to-end mean-only control is also still required.

```bash
fisher-graph-gemma-modal-generator-dev \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
```

The next development rung compiled four distinct-layer fragments at layers
10, 11, 16, and 17, then allowed only causal fan-in from the first three
nodes to the terminal layer-17 node. The original 40-prompt development
evaluation export was deterministically split before fitting: 20 prompts
selected node/edge details and the other 20 assessed the frozen graph. All
three candidate edges were selected.

On the untouched open-development assessment half, the interacting graph
improved over the exact same four nodes with no edges:

| condition | NLL/token | delta NLL | native KL/token | native top-1 |
| --- | ---: | ---: | ---: | ---: |
| interacting graph | 2.834824 | +0.010909 | 0.038655 | 91.8824% |
| identical edgeless graph | 2.839795 | +0.015880 | 0.040344 | 91.3333% |
| matched deletion | 3.551284 | +0.727369 | 0.675984 | 66.4314% |
| native | 2.823914 | 0 | 0 | 100% |

This is the first direct evidence here that learned cross-layer modal
messages add model-level fidelity rather than merely adding graph machinery.
The effect is real on this slice but small: edges recover `0.004971`
NLL/token, reduce KL by `0.001688`, and add `0.549` percentage points of
native top-1 agreement relative to the edgeless control.

The graph replaces 476,160 native fragment parameters/MACs per token with
130,784 parameters and 128,000 matrix MACs, saving `72.53%` of local storage
and `73.12%` of local matrix work. That is only `0.1288%` of whole-model
parameters because this rung replaces 248 channels across four MLPs. The
three edges cost 3,168 parameters and 3,072 MACs per token. This remains
same-family, self-attested, open-development evidence—not a heldout
compression claim—and the strongest edge used an unregularized,
poorly-scaled interaction matrix. State whitening/ridge stability and a
fresh family-disjoint guard are required next. See the
[`multi-fragment terminal fan-in result`](docs/modal-generator-compiler.md#multi-fragment-terminal-fan-in-development-rung).

```bash
fisher-graph-gemma-modal-generator-multifragment-dev \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
```

The breadth follow-up selected one top-Fisher fragment in every one of the 18
MLP layers. Its separately fused edgeless path replaced 2,271,360 native
parameters with 380,160 generator parameters, saving `83.26%` locally but
only `0.7054%` of the whole model. On assessment20 it reached NLL `2.839176`
versus `2.823914` native and `85.7647%` native top-1 agreement. Twelve
regularized terminal fan-in edges slightly improved KL and top-1, but worsened
NLL to `2.846954`, so the exhaustive rung kept the first full-stack graph
edgeless.

That exhaustive development run aggregates all 64 Fisher fragments in each
layer before fitting modes. It physically replaces all 2,048 MLP channels in
all 18 blocks—36,864 channels and 70,778,880 native MLP parameters—with one
rank-640 residual generator per block. Attention, embeddings, normalization,
and the language-model head remain native; this is the full native MLP stack,
not the whole transformer.

| condition | NLL/token | delta NLL | native KL/token | native top-1 |
| --- | ---: | ---: | ---: | ---: |
| native | 2.823987 | 0 | 0 | 100% |
| generated full MLP stack | 3.172463 | +0.348476 | 0.456653 | 74.0588% |
| matched deletion | 13.902236 | +11.078248 | 11.108717 | 0.2353% |

The generators therefore recover `96.85%` of deletion's NLL penalty and
`95.89%` of its KL penalty, showing that they learned substantial computation
rather than merely identifying expendable MLPs. They still miss the current
fidelity target: perplexity rises from `16.84` to `23.87`, and `74.06%`
native top-1 agreement is not a downstream-accuracy result.

The logical deployable candidate has 212,076,416 parameters: 14,757,120
generator parameters plus 197,319,296 retained native non-MLP parameters. It
saves 56,021,760 parameters (`79.15%` of the MLP stack, `20.90%` of the whole
model). MLP linear work falls from 70,778,880 to 14,745,600 matrix MACs per
valid token, a `79.17%` local reduction, plus 11,520 bias additions. These are
logical counts, not a measured kernel-latency claim. The live experiment keeps
the source and compiled overlay resident together, while the ignored
452 MB tensor artifact also stores float64 analysis curves; neither is the
packed deployment footprint.

Local generator fits remain strong—selection weighted NRMSE ranges from
`0.1146` to `0.2979`, with weighted cosine from `0.9553` to `0.9934`—so the
full-stack loss is evidence of compounded trajectory shift: each generator
was fit on native layer inputs but receives earlier generated states at
runtime. The next diagnostic is a frozen prefix/suffix replacement ladder,
followed by compiled-trajectory refitting or causal correction edges. This
remains same-family, self-attested open-development evidence and is not a
compression or heldout claim.

```bash
fisher-graph-gemma-full-mlp-stack-dev \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
```

The analysis reports contain only pooled activation means/covariances, derived
Fisher modes and codecs, exact trace accounting, bounded transport/JVP/factor
state or scalar evaluation curves, and provenance. The strict cross-block
candidate round-trip is the exception: it materializes the two affected
candidate MLPs, including cloned pretrained tensors, exclusively under the
ignored `.local-runs/` tree and must never be committed. The gated artifact
defaults to the same ignored local tree at
`.local-runs/google--gemma-3-270m/layers-4-6-gated-executor.pt`, with a
tensor-free JSON report beside it. The locked candidate's deployable FP32
state accounts for about 2.17 MB; the roughly 16 MB diagnostic tensor artifact
is larger because it retains all four fitted candidate states and audit
payloads. Model files remain in the external Hugging Face cache. The
trajectory writer now emits
artifact format version 2 with the causal payload; its strict loader still
accepts version-1 row-local artifacts without synthesizing causal results. A
successful opt-in smoke run checks the live integration path; the committed
synthetic tests check the adapter and analysis contracts. Neither establishes
compilability or model quality. See
[`docs/gemma3-270m.md`](docs/gemma3-270m.md) for cache safeguards, precise
Fisher semantics, device options, and the next validation gate.

The equivalent module commands are:

```bash
python -m fisher_graph.associative_experiment
python -m fisher_graph.associative_conditional_rank_experiment
python -m fisher_graph.variable_associative_training
python -m fisher_graph.variable_conditional_experiment
python -m fisher_graph.variable_full_span_experiment
python -m fisher_graph.variable_static_full_span_experiment
python -m fisher_graph.variable_static_full_span_v2_experiment
python -m fisher_graph.intervention_experiment
python -m fisher_graph.modal_executor_experiment
python -m fisher_graph.modal_completion_experiment
python -m fisher_graph.modal_executor_experiment --layer-index 1 --routing-widths 4 6 8 12 16 24
python -m fisher_graph.modal_completion_experiment --layer-index 1
python -m fisher_graph.modal_composition_experiment
python -m fisher_graph.fused_executor_experiment
python -m fisher_graph.mlx_benchmark \
  --output artifacts/associative_recall/mlx_metal_benchmark.json
python -m fisher_graph.gemma3_experiment \
  --prompts examples/gemma3_prompts.txt
python -m fisher_graph.gemma3_stability_experiment \
  --prompt-splits examples/gemma3_stability_prompts.json
python -m fisher_graph.gemma3_trajectory_experiment \
  --prompt-splits examples/gemma3_stability_prompts.json
python -m fisher_graph.gemma3_ablation_experiment \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --retained-ranks 640 512 384 256 192 128 96 64 32 0
python -m fisher_graph.gemma3_weighted_jacobian_experiment \
  --prompt-splits examples/gemma3_stability_prompts.json \
  --retained-ranks 632 636 638 639 640
python -m fisher_graph.gemma3_gated_executor_experiment \
  --weighted-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt \
  --prompt-splits examples/gemma3_gated_executor_prompts.json
python -m fisher_graph.gemma3_projection_ladder_experiment \
  --weighted-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt \
  --gated-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-gated-executor.pt \
  --prompt-splits examples/gemma3_projection_ladder_prompts.json
python -m fisher_graph.gemma3_full_width_single_layer_experiment \
  --prompt-splits /absolute/path/full-width-prompts.json \
  --family-manifest /absolute/path/full-width-families.json
python -m fisher_graph.gemma3_modal_generator_dev_experiment \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
python -m fisher_graph.gemma3_modal_generator_multifragment_dev_experiment \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
python -m fisher_graph.gemma3_full_mlp_stack_dev_experiment \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
python -m fisher_graph.optimization_figure
python -m fisher_graph.verify artifacts/associative_recall
```

The first command trains and builds the Fisher artifact. The intervention
command evaluates deterministic Fisher-mode controls, and the modal-executor
command fits candidate graph widths on training activations, selects one using
validation behavior, saves the standalone replacement, and only then evaluates
the test split. The modal-completion command freezes that model and fits small
ridge bridges from retained modes to discarded modes. The layer-1 invocations
repeat the same frozen-teacher procedure at the
`layer.0.output -> layer.1.output` boundary. The composition command then loads
the locked artifacts, applies the validation gate, and evaluates their
two-layer runtime without fitting or changing any weight.
The fusion command then folds that locked runtime algebraically, saves both
the backward-compatible monolithic artifact and the compact lazy artifact,
reloads them, applies a stricter numerical equivalence gate, exercises the
lazy instrumentation lifecycle, and benchmarks the teacher, logical modal,
monolithic fused, and compact lazy systems. It also derives the packed
triangular candidate without fitting any weight, validation-gates it against
the lazy runtime, and runs a separate same-round five-system benchmark without
using the test split. It regenerates the authenticated runtime manifest after
writing the final benchmark report.
The optional MLX command leaves that manifest and the CPU fused report
unchanged. It derives the packed state in memory, checks that the source
instrumentation sidecar remains unloaded, validates the exact outputs being
timed, and compares dense compiled MLX, ordinary packed compiled MLX, and the
custom packed Metal kernel on one GPU.
The optional Gemma commands form a separate analysis-only rung. They freeze
the source model, differentiate each sequence independently at selected
residual boundaries, and stream the resulting rows through bounded Frequent
Directions sketches. The stability command compares independently estimated
subspaces and streams held-out gradients through frozen mode bases. The
trajectory command plans nonduplicated boundaries for a contiguous block,
measures adjacent Fisher geometry and cross-Rayleigh transfer, fits
bounded-memory activation and reverse score-gradient transports on calibration,
then evaluates those frozen maps on validation. The reverse-causal comparison
uses exact logical-position lags, nested finite-lag windows, and the
structurally valid visibility of each adjacent or block-endpoint segment. Its
lag-0 ridge member is the row-local baseline for measuring the gain from later
positions. The uncentered gradient score is zero-baseline explained energy
rather than statistical \(R^2\). The ablation command instead extracts
full-width calibration bases and projects the selected layer outputs jointly
through descending keep-top-\(k\) prefixes on validation, with optional
one-site-at-a-time localization. The weighted-Jacobian command instead fits
activation-aware codecs on calibration A, selects a joint codec/rank on
calibration B, evaluates that locked choice on validation, and optionally
measures true forward JVP edges before applying the generic weighted causal
factorizer. The resulting Toeplitz factor is a bounded stationary reference,
not an installed Gemma executor. The gated-executor command then fits
residual-separated, state-conditioned causal candidates on calibration A,
locks on calibration B, and intervenes once on validation. Its current
rank-320/480 result is explicitly nonviable. The projection-ladder command
strict-loads both predecessors, leaves calibration A and test hash-only,
selects only from the calibration-B rank curve, and evaluates exactly one
locked rank on validation. Its current rank-640 identity fallback is also
explicitly nonviable as compression. None of these commands changes a runtime
manifest.

## Compiler interfaces and scaling boundary

The full-width Fisher/modal analysis path no longer depends directly on
`ToyTransformer.forward` or on a concrete layer list. It is split into five
contracts (the older diagonal-Fisher helper remains toy-specific):

- `ModelAdapter` describes stable activation sites, heterogeneous native
  layers, replaceable segments, sequence capabilities, masks, logical
  positions, cache positions, and source-model fingerprints.
- `CalibrationBatch` carries model inputs, targets, and valid activation
  positions. A calibration stream may contain batches with different sequence
  lengths.
- `ScoreObjective` defines the differentiable scalar whose activation
  gradients form the Fisher sample. `CausalLanguageModelNLL` is the built-in
  hard-target summed-NLL objective.
- `InstrumentedModel` is the smaller Fisher-facing protocol. An
  `InstrumentedModelBinding` combines a mixed runtime with explicit
  `ActivationSite` metadata for backend-native modal taps.
- `RuntimeManifest` describes compiled segments and their guards, source
  layers, fast tensors, lazily loaded instrumentation resources, validation
  state, fallback policy, and byte-level provenance.

`Gemma3CausalLMAdapter` supplies the same prefill-facing contracts for a
text-only Hugging Face Gemma 3 causal LM. `collect_streaming_fisher_modes`
preserves the existing summed-NLL, per-sequence score definition while
replacing retained calibration matrices with
`O(sketch_rows * hidden_width)` state.
`iter_activation_score_gradient_rows` exposes one transient sequence at a time
for validation or other bounded-memory analyses. `compare_fisher_subspaces`
and `StreamingRayleighEnergyEstimator` provide sign/rotation-invariant
split-stability and exact frozen-basis replay metrics.
`ModelAdapter.plan_layer_block` produces a canonical `LayerBlockBoundaryPlan`
that omits later input aliases, while `StreamingModalTransportEstimator`
retains only modal sums and \(k\times k\) moments. Its frozen Procrustes map is
evaluated without retaining held-out rows.
`StreamingCausalModalTransportEstimator` adds a sequence-scoped reverse
gradient model over exact logical lags. It streams a feature Gram, a
feature/target cross-moment, and a target Gram in FP64; storage depends on
modal rank and maximum lag, not on the number or length of calibration
sequences. The lag-0 prefix is fit separately as the nested ridge baseline,
and larger prefixes can use downstream positions \(s+\delta\) only when that
logical position exists and is structurally visible. This can test whether
the cross-token omission explains a weak row-local map. It does not recover
individual Jacobian blocks and cannot by itself authorize a sequence-aware
modal executor. Cache-aware decode and an authenticated Gemma graph
replacement remain later gates.

`StreamingActivationCovariance` supplies the matching activation second
moment. `LinearActivationCodec` represents either an orthonormal Fisher basis
or a generalized dual encoder/decoder satisfying full-width identity.
`collect_block_causal_lag_jacobian` then excites decoder columns and projects
true source-block JVPs through output encoders, preserving signed exact-lag
edges and separately accounting for causal leakage, omitted past energy, and
within-lag variation both per lag and with lag zero excluded.
`factor_causal_weighted_jacobian` factors every causal
output prefix independently under block-local activation covariance and
output Fisher metrics. Its executor has no future-position parameter slots
and exposes exact SVD-tail, per-edge/per-lag energy, coefficient, and MAC
accounting.
These APIs implement the generic node-and-edge reference; behavioral
completion, variable-length lowering, and authenticated Gemma replacement are
still separate gates.

The current model is exposed through `ToyTransformerAdapter`. Fisher
collection and modal Jacobian extraction use the generic adapter path, and
modal replacement is installed through an atomic segment context that always
restores the source model:

```python
from fisher_graph import (
    CalibrationBatch,
    CausalLanguageModelNLL,
    ToyTransformerAdapter,
    collect_adapter_score_gradients,
)

adapter = ToyTransformerAdapter(model)
batch = CalibrationBatch(
    model_inputs={"input_ids": tokens, "attention_mask": valid},
    targets=targets,
    valid_positions=valid,
)
scores = collect_adapter_score_gradients(
    adapter,
    [batch],
    activation_names=adapter.default_fisher_sites,
    score_objective=CausalLanguageModelNLL(),
)

segment = adapter.segments[0]
with adapter.replaced_segments({segment.id: compiled_executor}):
    compiled_logits = adapter.forward({"input_ids": tokens}).logits
```

The checked fused artifact is indexed by a canonical JSON manifest at
[`artifacts/associative_recall/runtime_manifest.json`](artifacts/associative_recall/runtime_manifest.json).
Loading the manifest does not deserialize tensor bundles. A resource is
exposed as bytes or a seekable streaming handle only after its contained
relative path, file type, byte length, and SHA-256 have been verified.

The checked fused artifact is still the fixed-position numerical oracle: it
stores an explicit eight-position axis, advertises exactly length eight, and
has not been converted into the new dynamic executor. The new
`VariableLengthCausalModalExecutor` instead uses shared Fisher projections and
relative-position causal state, so its parameter shapes do not depend on
sequence length. `MixedSegmentDispatcher` can execute dynamic and source
segments in one nonmutating plan, while `MixedModelRuntime` composes embedding,
that plan, and the source head. Its capability matcher distinguishes caller
input provenance, masks, query/key lengths, query/key position relationships,
logical-position domains, visibility, cache mode, dtype, device, and layout;
unknown facts fail closed.

The dynamic path is an instrumentable **prefill-only training scaffold**, not
yet a compiled Gemma runtime. It must be fit against real source-segment
boundary pairs and pass untouched-length and end-to-end quality gates before
an artifact is accepted. The current synthetic fitter test demonstrates
same-architecture distillation only. Global equal-query/key execution uses a
token-by-token Python recurrence; sliding windows or distinct query/key
semantics currently use a dense quadratic path. Cached/chunked prefill, decode,
an efficient sliding-window scan, backend-neutral symbolic IR, and a real
source-fitted dynamic artifact remain future milestones.

Runtime state authorization is strict by default: source tensors, live
adapter execution options, and both fast/inspectable compiled executors are
matched to compiler-time fingerprints and checked again at every compiled
boundary.
The normalized sequence context is snapshotted and must remain unchanged for
the whole request. Large deployments may choose `trusted_immutable` only when
an authenticated loader enforces model/executor immutability outside the
Python runtime; request sequence immutability is still checked.

The full target architecture, including symbolic sequence semantics, cache
ownership, validation gates, and a staged Gemma 3 integration, is in
[`docs/compiler-architecture.md`](docs/compiler-architecture.md).

## Learned task

Every sequence contains two key/value pairs and asks the model to retrieve one
of them:

```text
BOS  key0 value0  key1 value1  QUERY queried_key  ANSWER
```

Only the final position is supervised. Both query variants for a context stay
in the same grouped split:

| Split | Contexts | Sequences | Purpose |
|---|---:|---:|---|
| Train | 2,508 | 5,016 | Parameter learning |
| Validation/Fisher | 314 | 628 | Model selection and Fisher build |
| Test | 314 | 628 | Final baseline evaluation |

Training uses label smoothing, while evaluation and Fisher collection use hard
observed targets. This keeps a correct model from saturating so completely
that its empirical score gradients collapse.

## Activation capture

Instrumentation is explicit: there are no global hooks or singleton stores.

```python
import torch
import torch.nn.functional as F

from fisher_graph import ToyTransformer, TransformerConfig

model = ToyTransformer(TransformerConfig())
tokens = torch.randint(0, model.config.vocab_size, (2, 8))
output = model(tokens, capture_activations=True)

trace = output.activations
print(trace.names)
print(trace["layer.0.attention.probabilities"].shape)

targets = torch.randint(0, model.config.vocab_size, (2, 8))
loss = F.cross_entropy(output.logits.flatten(0, 1), targets.flatten())
loss.backward()
print(trace.gradients()["layer.0.output"])
```

Captured tensors remain attached to autograd and retain gradients by default.
`ActivationTrace.detached()` produces CPU snapshots safe for longer-term
storage.

## Fisher compute modes

For a named activation boundary with width \(D\), the build collects one
hard-target, summed-NLL score gradient per valid sequence position:

\[
g_{i,t} =
\frac{\partial[-\log p(y_i\mid x_i)]}{\partial a_{i,t}}
\in \mathbb{R}^{D}.
\]

It constructs the full width-pooled empirical Fisher:

\[
F_a = \frac{1}{M}\sum_{i,t} g_{i,t}g_{i,t}^{\mathsf T},
\]

then diagonalizes it:

\[
F_a = U_a\Lambda_a U_a^{\mathsf T}.
\]

Each column of \(U_a\) is a Fisher sensitivity mode. Modal coordinates and
reconstruction are:

\[
z=(a-\mu_a)U_a,\qquad
\hat a=zU_a^{\mathsf T}+\mu_a.
\]

```python
from fisher_graph import load_fisher_build

bases, transitions, jacobians, metadata = load_fisher_build(
    "artifacts/associative_recall/fisher_modes.pt"
)
basis = bases["layer.1.output"]

print(basis.eigenvalues)
print(basis.modes_for_fraction(0.95))
modal_coordinates = basis.project(some_activation, modes=12)
reconstructed = basis.reconstruct(modal_coordinates)
```

The width-pooled definition intentionally drops cross-position gradient
covariance so one basis can be reused at every token. It is not the same object
as a flattened, sequence-specific Fisher over \(T D\) coordinates.

More precisely, this is a width-pooled empirical activation score-gradient
second moment. Token positions are pooling rows, not independent conventional
Fisher examples. With variable lengths, normalization by all valid positions
gives longer sequences more weight; summed NLL also lets early activations
accumulate effects from later supervised predictions. Streaming artifacts
record those policies, so spectra from different length mixtures are not
presented as length-neutral.

The artifact stores both a pooled activation mean and validation/Fisher
position means. Projection and reconstruction above use the pooled mean;
causal interventions default to the position-conditioned means so muting a
mode does not replace, for example, a value-token component with an average
drawn from unrelated sequence roles. A modal graph bottleneck should pass
`centering="position"` to both `project` and `reconstruct` so it matches the
intervention and sufficiency semantics.

## Fisher-mode intervention equalizer

A Fisher basis can be treated like an equalizer whose bands are rotated
activation directions rather than audio frequencies. For examples \(i\),
positions \(t\), a selected set of modes \(S\), and suppression strength
\(s\), the intervention is

\[
a'_{i,t}
=
a_{i,t}
-
s\left((a_{i,t}-\mu_{a,t})U_{a,S}\right)U_{a,S}^{\mathsf T},
\]

where \(\mu_{a,t}\) is estimated only from the validation/Fisher split.
Strength \(s=0\) is a no-op, while \(s=1\) replaces the selected modal
coordinates with their position-conditioned mean. Unselected orthogonal
coordinates are preserved.

```python
from fisher_graph import (
    FisherModeSuppression,
    load_fisher_build,
    top_mode_indices,
)

bases, _, _, _ = load_fisher_build(
    "artifacts/associative_recall/fisher_modes.pt"
)
basis = bases["layer.0.output"]
equalizer = FisherModeSuppression(
    basis=basis,
    mode_indices=top_mode_indices(basis, 8),
    suppression_fraction=0.25,
)

intervened = model(
    tokens,
    activation_interventions={basis.activation_name: equalizer},
)
```

The experiment compares top-, bottom-, and random-mode groups across several
counts and strengths. It also matches controls to the top intervention's RMS
activation displacement, scans one token position at a time, and runs the
complementary sufficiency test: keep the top \(k\) modes by muting all modes
below them.

At the primary replacement boundary, `layer.0.output`, the designated primary
cell mutes the top eight modes by 25%. The held-out results are:

| Control | Delta hard NLL | Interpretation |
|---|---:|---|
| Top eight, fixed 25% | 0.009075 | Designated top-mode effect |
| Bottom eight, fixed 25% | 0.000058 | Same count and strength |
| 100 random sets, fixed 25% | [0.000518, 0.006565] | 95% interval; empirical \(p=0.0099\) |
| Bottom eight, energy matched | 0.000228 | RMS displacement matched to top |
| 100 random sets, energy matched | [0.001428, 0.013689] | 95% interval; empirical \(p=0.1980\) |

The fixed-strength result places the top group above random eight-mode subsets
of the same Fisher eigenbasis, and its paired-context NLL effect exceeds the
bottom group with a 95% bootstrap interval of `[0.007675, 0.010526]`. After
matching perturbation energy, with matching strengths calibrated on
validation/Fisher activations, the top-minus-bottom interval remains positive
(`[0.007477, 0.010381]`), but the top group is not distinguished from random
mode subsets at conventional significance. The cautious conclusion is
therefore that the Fisher ordering identifies causally important directions
and strongly separates them from the low-Fisher tail, while some of the
fixed-strength top-versus-random advantage is explained by how much activation
energy those directions carry. Because these controls sample subsets of the
Fisher basis, they do not compare that basis against arbitrary rotated
directions. This is evidence for useful modal structure, not yet proof that
Fisher modes are the unique equal-energy computational basis.

Accuracy begins at 100%, so hard NLL, correct-token probability, output KL,
and logit margin are recorded before accuracy moves. The single-mode
Fisher-rank versus NLL-effect Spearman correlation at `layer.0.output` is
0.8963. Mode activation RMS also correlates with both Fisher rank (0.7397) and
NLL effect (0.6998), so that single-mode correlation is descriptive rather
than an energy-normalized sensitivity result. A position scan places the
largest early-boundary effects at the two value-token positions, consistent
with the learned retrieval task, though all results here describe one trained
checkpoint.

### Modal sufficiency

Keeping only leading modes gives a direct compression test. At the primary
boundary:

| Leading modes retained | Fisher retained | Test accuracy | Hard NLL |
|---:|---:|---:|---:|
| 14 | 90.254% | 99.841% | 0.064689 |
| 18 | 95.593% | 100.000% | 0.050775 |
| 25 | 99.054% | 100.000% | 0.048927 |
| 32 | 100.000% | 100.000% | 0.048382 |

At `layer.1.output`, 12 leading modes retain 95.592% of Fisher information,
100% accuracy, and hard NLL 0.061001; 19 modes reduce NLL to 0.049642. Equal
numbers of trailing modes perform far worse. These boundary-wise results make
the leading subspaces plausible graph state spaces, but they do not yet show
that simultaneous input/output bottlenecks or a replacement executor preserve
the computation.

## Modal computation maps

Fisher modes identify task-sensitive directions; they do not by themselves
prove a causal computation graph. The build therefore also extracts each
layer's sample-local Jacobian with directional derivatives and projects it into
the Fisher bases:

\[
J_{\text{mode}} =
U_{\text{out}}^{\mathsf T} J U_{\text{in}}.
\]

Attention mixes positions, so the artifact preserves the full axes:

```text
[output_position, output_mode, input_position, input_mode]
```

Both the signed mean and RMS magnitude are saved. RMS prevents
context-dependent edges with opposite signs from disappearing through
averaging.

The Fisher artifact also contains position-coupled affine transition fits.
These remain descriptive summaries. The executable affine baseline below is
fit separately with a strictly causal mask and evaluated as an actual layer
replacement.

## Graph replacement boundary

Every transformer layer implements:

```python
LayerExecutor.forward(
    hidden_states,
    *,
    attention_mask,
    trace,
    prefix,
) -> hidden_states
```

An ordinary block can already be converted into an exactly equivalent DAG:

```python
from fisher_graph import GraphLayerExecutor

block = model.layers[0]
graph = GraphLayerExecutor.from_transformer_block(block)
model.replace_layer(0, graph)
```

That swap preserves logits and activation names. The modal executor uses the
same boundary without changing embeddings, the language-model head, or the
second transformer layer:

```python
from fisher_graph import load_position_modal_executor

executor, config, metadata = load_position_modal_executor(
    "artifacts/associative_recall/modal_executor.pt"
)
model.replace_layer(metadata["layer_index"], executor)
```

## Position-conditioned modal executor

Layer 0 can now be removed completely and replaced by a dense causal nonlinear
surrogate. For each fixed sequence position \(t\), it computes

\[
\begin{aligned}
z_t &=
  (a_t-\mu^{\mathrm{in}}_t)U^{\mathrm{in}}_{:K},\\
\tilde z_t &= z_t/\sigma^{\mathrm{in}}_t,\\
h_t &=
  \operatorname{GELU}\!\left(
    W_t[\tilde z_0;\ldots;\tilde z_t]+b_t
  \right),\\
q_t &= \sigma^{\mathrm{out}}_t\odot(V_t h_t+c_t),\\
\hat a_t &=
  \mu^{\mathrm{out}}_t+
  q_t(U^{\mathrm{out}}_{:L})^{\mathsf T}.
\end{aligned}
\]

The means come from the validation/Fisher activation bases. The scales are
estimated from training examples. The input \(z_t\) and output \(q_t\)
coordinates are Fisher modes; the hidden \(h_t\) routing features are ordinary
learned features, not additional Fisher eigenmodes. Each \(h_t\) reads only
positions \(0\) through \(t\), so future tokens cannot affect an earlier
output. Separate position parameters are deliberate here: the eight sequence
positions have fixed semantic roles, such as key, value, query, and answer.
The saved modal Jacobians were not used to prune this first executor.

The build evaluates four different systems so compression and execution are
not conflated:

| System | What runs at layer 0 | Test accuracy | Paired accuracy | Hard NLL |
|---|---|---:|---:|---:|
| Original | Transformer block | 100.000% | 100.000% | 0.048382 |
| Modal bottleneck | 27-mode input, original block, 25-mode output | 99.204% | 98.408% | 0.078740 |
| Causal affine graph | Standalone linear position-mode map | 59.076% | 23.885% | 1.891171 |
| Causal nonlinear graph | Standalone 27 -> 12 -> 25 modal graph | 100.000% | 100.000% | 0.049455 |

Widths 4, 6, 8, and 12 were fit only from training activations for one
initialization and a 2,000-step budget. The smallest tested candidate satisfying
the predeclared validation gate—at least 99.5% answer accuracy, at least 99%
paired-context accuracy, and no more than 0.01 NLL over baseline—was width 12.
This is a selection result under that budget, not proof that 12 is the minimum
possible capacity. Width 8 came close but did not pass, so the saved build does
not depend on a favorable initialization.

The tested causal affine graph explained 67.8% of centered validation
activation variance but failed behaviorally. The nonlinear graph explained
91.1% and preserved every test answer. This shows that aggregate activation
fit alone is not a sufficient executor acceptance criterion; it does not prove
that all possible successful executors must be nonlinear.

The selected graph has 14,360 learned parameters and 14,064 scalar
connections. Its analytic multiply estimate, including modal projection and
reconstruction, is 27,376 per sequence versus 69,632 for the original block,
or 39.3%. The graph has more position-specific parameters than the original
weight-sharing block (8,544), so this is a compute extraction result rather
than a parameter-compression result. The operation count is an estimate, not a
wall-clock benchmark.

The executor exposes `layer.0.modal.input`, `layer.0.modal.hidden`, and
`layer.0.modal.output` through the normal `ActivationTrace`, so the routing
nodes can be measured or intervened on just like transformer activations. It
currently targets the fixed, unpadded eight-token task.

The test split was not used for fitting or width selection. It had, however,
already been inspected during earlier work in this exploratory repository, so
the result should be treated as a successful single-checkpoint construction,
not a fresh confirmatory estimate. The next scientific step is to freeze this
procedure and repeat it across new seeds and untouched test splits, then prune
modal connections while enforcing the same behavioral validation gate.

## Conditional modal completion

Truncated reconstruction normally fills every discarded Fisher coordinate
with zero deviation from its position mean. Conditional completion instead
predicts those tail coordinates before returning to the full residual basis:

\[
\hat z^{\mathrm{tail}}_t
=
z^{\mathrm{kept}}_t A_t+b_t,\qquad
\hat a_t
=
\mu_t+
[z^{\mathrm{kept}}_t,\hat z^{\mathrm{tail}}_t]U^{\mathsf T}.
\]

All transformer weights remain frozen. The build fits four deterministic
ridge candidates from training activations and selects the smallest candidate
passing a validation behavior gate. The selected input bridge shares one
\(27\rightarrow5\) weight matrix across positions with position-specific
biases; the output bridge uses position-specific \(25\rightarrow7\) maps.
Both are strictly position-local and therefore causal.

```python
from fisher_graph import (
    PositionConditionedCompletedModalGraphExecutor,
    load_position_modal_completion,
    load_position_modal_executor,
)

executor, _, _ = load_position_modal_executor(
    "artifacts/associative_recall/modal_executor.pt"
)
output_completion, _, _ = load_position_modal_completion(
    "artifacts/associative_recall/modal_completion_output.pt"
)
model.replace_layer(
    0,
    PositionConditionedCompletedModalGraphExecutor(
        executor,
        output_completion,
    ),
)
```

| System | Test accuracy | Paired accuracy | Hard NLL |
|---|---:|---:|---:|
| Frozen teacher | 100.000% | 100.000% | 0.048382 |
| Both boundaries truncated with zero tails | 99.204% | 98.408% | 0.078740 |
| Fit-set mean-tail control | 99.204% | 98.408% | 0.079429 |
| Both learned completions | 100.000% | 100.000% | 0.048396 |
| Standalone modal graph | 100.000% | 100.000% | 0.049455 |
| Modal graph plus output completion | 100.000% | 100.000% | 0.048890 |

On validation, the input bridge recovers the five discarded coordinates with
\(R^2=0.999999989\); the output bridge reaches \(R^2=0.994710\). Together they
raise full layer-output activation \(R^2\) from 0.909438 for zero filling to
0.999613. They also beat a fit-set, position-specific mean-tail predictor,
showing that the retained coordinates carry example-specific information
about the discarded coordinates.

The two bridges contain 1,631 learned parameters. Around the original block,
completion is an interface-recovery diagnostic and adds computation. Attached
only to the standalone graph's output, it raises the analytic multiply estimate
from 27,376 to 30,568, or 43.9% of the original block estimate, while improving
the next layer's input distribution.

The completion taps
`layer.0.modal.input_completion.tail`,
`layer.0.modal.input_completion.coordinates`,
`layer.0.modal.output_completion.tail`, and
`layer.0.modal.output_completion.coordinates` are instrumentable and
intervenable. The result demonstrates conditional recovery of redundant tail
coordinates; it does not imply that arbitrary discarded information can be
reconstructed.

## Frozen layer-1 compilation and two-layer composition

Layer 1 was compiled independently at the
`layer.0.output -> layer.1.output` boundary using the same procedure as layer
0. Its selected standalone executor retains 25 input modes, uses 24 causal
routing features, and predicts 19 output modes. It reads the exact input given
to the frozen teacher layer and learns that layer's output on the same input;
it is not trained to cancel an upstream executor's errors. The standalone
layer-1 graph reached 100.000% answer and paired accuracy on validation and the
exploratory test, with hard NLL 0.048313 and 0.048361 respectively.

Layer 1 also has conditional completion bridges. The input diagnostic predicts
7 discarded modes from 25 retained modes, while the runtime output bridge
predicts 13 discarded modes from 19 retained modes. With output completion,
the standalone layer-1 replacement retained 100.000% answer and paired
accuracy, with hard NLL 0.046722 on validation and 0.046785 on the exploratory
test.

The runtime rule is intentionally narrower than the completion experiment:
each standalone graph is wrapped with its **output completion only**. That
bridge reconstructs the full 32-dimensional residual stream expected by the
next compiled layer or the language-model head. The saved input-completion
bridge is a diagnostic for asking whether retained input coordinates can
restore the interface around the original frozen transformer block; it is not
inserted before a standalone graph during composition.

The two locked completed executors compose successfully:

| Split | System | Answer accuracy | Paired accuracy | Hard NLL | KL vs teacher |
|---|---|---:|---:|---:|---:|
| Validation | Frozen teacher | 100.000% | 100.000% | 0.049155 | 0.000000 |
| Validation | Both completed graphs | 100.000% | 100.000% | 0.048757 | 0.002911 |
| Exploratory test | Frozen teacher | 100.000% | 100.000% | 0.048382 | 0.000000 |
| Exploratory test | Both completed graphs | 100.000% | 100.000% | 0.049286 | 0.003061 |

Composition is checked with a same-input contract, not only end-to-end
answers. Let \(E_0\) and \(E_1\) be the completed executors and \(B_1\) the
original frozen second transformer block. The local comparison is

\[
B_1(E_0(h)) \quad\text{versus}\quad E_1(E_0(h)).
\]

Holding `E0(h)` fixed on both sides isolates layer-1 fidelity after the
upstream distribution shift. On validation, this comparison has suffix KL
0.002538 and Fisher-weighted layer-output RMS error 0.003285. It prevents
downstream cancellation from being mistaken for a faithful local
replacement.

No transformer weight was fine-tuned or retrained, the two executors were not
co-trained, and layer 1 did not need robustification on compiled layer-0
outputs. Both executors were fitted against their pristine frozen teacher
layers on matching clean inputs, then locked before composition validation and
test. If a future checkpoint fails the same-input gate, robustification should
still train only the downstream executor on shifted inputs with the frozen
teacher block evaluated on those exact same inputs; it should not train against
clean teacher outputs or update neighboring transformer layers.

The completed layer-0 and layer-1 graphs have analytic estimates of 30,568 and
41,816 scalar multiplies per sequence. Together that is 72,384 versus 139,264
for the two original transformer blocks, or 51.976%. These are block-only
analytic multiply estimates, not wall-clock measurements.

This is still an exploratory single-checkpoint construction. Widths and
completion topologies were validation-selected, the Fisher basis uses the
validation split, and the test split was inspected during earlier development.
The result shows that frozen independent composition works for this checkpoint;
it is not yet a seed-replicated confirmatory result.

## Algebraically fused modal executor

The logical modal stack is useful for inspection, but it materializes every
intermediate step: Fisher coordinates, normalization, routing features, output
coordinates, completion tails, a full 32-dimensional decode, and then the next
layer's re-encoding. Most of those steps are affine maps separated by only two
GELUs. They can therefore be composed into three runtime contractions:

\[
\begin{aligned}
h^0_t &=
  \operatorname{GELU}\!\left(
    \sum_{s\le t}(x_s-\mu_s)K^0_{t,s}+b^0_t
  \right),\\
h^1_t &=
  \operatorname{GELU}\!\left(
    \sum_{s\le t}h^0_sM_{t,s}+b^1_t
  \right),\\
y_t &= h^1_tF_t+c_t.
\end{aligned}
\]

Input centering stays explicit to avoid unstable bias cancellation. The
cross-layer bridge is enabled only when layer 0's output basis and position
means exactly equal layer 1's input basis and means. That identity holds for
the checked artifacts, so the fast path does not decode layer 0 to 32 residual
dimensions merely to project it back into layer 1's 25 retained coordinates.
No coefficient is fitted or updated during fusion.

```python
from fisher_graph import (
    FusedToyTransformer,
    PackedTriangularFusedTwoLayerModalStack,
    load_lazy_fused_modal_stack,
)
from fisher_graph.training import load_checkpoint

teacher, _ = load_checkpoint(
    "artifacts/associative_recall/checkpoint.pt"
)
stack, config, metadata = load_lazy_fused_modal_stack(
    "artifacts/associative_recall/fused_modal_runtime.pt"
)
runtime = FusedToyTransformer.from_teacher(teacher, stack).eval()

fast_output = runtime(tokens)
print(stack.instrumentation_status())  # sidecar is still unloaded

packed_stack = PackedTriangularFusedTwoLayerModalStack.from_lazy(stack)
packed_runtime = FusedToyTransformer.from_teacher(
    teacher, packed_stack
).eval()
packed_output = packed_runtime(tokens)

traced_output = runtime(tokens, capture_activations=True)
print(traced_output.activations["layer.0.modal.hidden"])
print(stack.instrumentation_status())  # loaded and cached

stack.evict_instrumentation()
```

An ordinary forward touches only seven registered fast tensors and never opens
the instrumentation files. The first activation capture or intervention reads
the four existing per-layer executor/completion artifacts, checks their sizes
and SHA-256 hashes, validates their checkpoint/Fisher/teacher provenance,
re-folds them to prove that they derive the resident fast tensors, and caches
the accepted logical layers. Later instrumented calls reuse that cache. An
explicit eviction releases it, and a dtype/device conversion evicts it
automatically so that a later trace reloads it in the new runtime format.

The packed triangular candidate copies only the 36 legal lower-triangular
position pairs from the two causal kernels and retains the position-local
decoder. It is algebraically equivalent for finite inputs, but floating-point
reduction order means it is not bit-identical to the dense path. The candidate
does not carry instrumentation sidecars: capture and intervention continue to
use the authenticated lazy runtime shown above.

Missing or corrupt sidecars therefore disable only capture/intervention:
ordinary inference continues to use the already-resident fast tensors. The
logical path preserves the same tap names and bit-exact traced tensors as the
unfused executor.

The validation gate was applied after saving and reloading the artifact:

| Equivalence check | Result |
|---|---:|
| Answer and paired accuracy | Exactly equal |
| Argmax predictions | Exactly equal |
| Absolute hard-NLL delta | \(1.863\times10^{-7}\) |
| Mean unfused-to-fused answer KL | \(5.631\times10^{-11}\) |
| Maximum answer-logit difference | \(2.723\times10^{-4}\) |

The exploratory test also retained 100.000% answer and paired accuracy, with
hard NLL 0.049286 for both logical and fused modal runtimes at the displayed
precision.

Block-only scalar-multiply accounting is:

| Runtime | Multiplies | Ratio to original blocks |
|---|---:|---:|
| Two original transformer blocks | 139,264 | 100.000% |
| Logical completed modal stack | 72,384 | 51.976% |
| Current fused dense path | 49,152 | 35.294% |
| Packed triangular reference | 30,336 | 21.783% |

The authenticated executor continues to use dense `einsum` kernels containing
structural causal zeros. The packed PyTorch reference executes the 30,336
nonzero multiplies using gathered causal pairs and indexed reduction. Arithmetic
reduction alone is not treated as a latency claim.

The full fixed-length model was benchmarked on the recorded arm64 CPU with one
PyTorch intra-op thread, inference mode, adaptive iterations, nine rotating
measurement rounds, and input construction outside the timed region:

| Batch | Teacher | Logical modal | Monolithic fused | Compact lazy | Lazy vs logical |
|---:|---:|---:|---:|---:|---:|
| 1 | 114.198 us | 190.748 us | 57.049 us | 55.976 us | 3.408x |
| 8 | 205.205 us | 242.431 us | 67.965 us | 68.461 us | 3.541x |
| 64 | 817.487 us | 364.664 us | 108.473 us | 107.841 us | 3.381x |
| 256 | 2,904.923 us | 615.545 us | 236.815 us | 235.440 us | 2.614x |

Across these batches, the compact/monolithic geometric-mean latency ratio was
0.994x. No hard latency gate was used.

The triangular candidate was measured in a separate five-system cohort so its
ratios use dense and packed measurements from the same rotating rounds:

| Batch | Compact lazy | Packed triangular | Triangular vs lazy | Triangular vs logical |
|---:|---:|---:|---:|---:|
| 1 | 57.362 us | 55.558 us | 1.032x | 3.416x |
| 8 | 67.494 us | 106.630 us | 0.633x | 2.191x |
| 64 | 110.239 us | 249.445 us | 0.442x | 1.427x |
| 256 | 218.363 us | 434.046 us | 0.503x | 1.374x |

The packed reference wins narrowly at batch 1, but its geometric-mean speedup
versus dense lazy execution is only 0.617x: the current gather, temporary
pair-output, and `index_add` overhead outweigh the 38.3% multiply reduction at
larger batches. It still beats the unfused logical runtime at every measured
batch, with a 1.957x geometric-mean speedup. This crossover is the reason the
dense lazy runtime remains the authenticated default.

### Experimental MLX/Metal lowering

`fisher_graph.mlx_executor` copies the seven floating-point packed tensors
once into MLX-owned arrays. The fast Metal stage launches one thread per
`(batch, target position, output feature)`, calculates the target-major packed
offset \(t(t+1)/2+s\), and reads only sources \(s\le t\). It accumulates in
FP32, adds the target bias, and applies GELU before writing. It does not create
the PyTorch reference's gathered-pair tensor, run `index_add`, or use atomics.

The runtime has three selectable execution modes:

| Mode | Purpose |
|---|---|
| `eager` | Ordinary MLX graph for inspection and differentiation |
| `compiled` | The same ordinary graph wrapped in `mx.compile` |
| `metal` | Two custom packed causal kernels inside a compiled fast path |

Activation capture deliberately routes through the ordinary MLX graph. The
authenticated PyTorch instrumentation path remains the activation-Fisher
oracle, and MLX activation interventions are not implemented. Converted state
is immutable after construction, public state exports do not alias the
compiled arrays, token indices are bounds-checked, and the Metal launch rejects
shapes that exceed its current 32-bit flattened-index contract.

The complete 628-example validation split retained exact argmax predictions
and 100% answer/paired accuracy for all three MLX modes. Against the PyTorch
packed full model, hard-NLL deltas were \(1.942\times10^{-4}\) for ordinary
MLX and \(1.967\times10^{-4}\) for Metal; maximum answer-logit differences
were 0.02060 and 0.00680 respectively. These are cross-framework float32
differences, not bit-exact equivalence.

The separate Apple M5 report measures only the two-layer modal stack. Each
timed call creates a fresh lazy output, evaluates it, and synchronizes the GPU;
nine rounds rotate system order:

| Batch | Dense compiled | Packed compiled | Packed Metal | Metal vs dense | Metal vs packed |
|---:|---:|---:|---:|---:|---:|
| 1 | 359.525 us | 1,068.050 us | 392.740 us | 0.915x | 2.719x |
| 8 | 393.054 us | 1,004.162 us | 360.469 us | 1.090x | 2.786x |
| 64 | 387.539 us | 937.733 us | 382.745 us | 1.013x | 2.450x |
| 256 | 402.594 us | 906.611 us | 395.130 us | 1.019x | 2.294x |

The custom kernel is 2.555x faster than the ordinary packed MLX graph by
geometric mean, proving that direct triangular scheduling removes the
gather/reduction implementation penalty. It is only 1.007x faster than dense
MLX by geometric mean: at this tiny fixed shape, optimized dense kernels and
launch overhead almost exactly offset the 38.3% arithmetic reduction. There
is therefore no hard latency gate, and Metal is not the default backend.

The compact runtime changes the storage tradeoff:

| Resident state | Bytes |
|---|---:|
| Backward-compatible monolithic full runtime | 713,920 |
| Compact full runtime during ordinary inference | 205,952 |
| Logical sidecar added after instrumentation | 203,648 |
| Compact full runtime with sidecar cached | 409,600 |
| Derived packed triangular full model | 131,264 |

The ordinary-inference footprint is 71.2% below the monolithic runtime. Even
after instrumentation is loaded it remains 42.6% below it, because the compact
runtime does not retain the monolithic artifact's redundant logical and folded
copies. These are resident tensor bytes for the complete model shell; the four
sidecar files occupy 244,724 bytes on disk and are read only on first
instrumentation. The compact artifact plus those reusable sidecars occupies
451,617 bytes on disk, 37.4% below the 721,471-byte monolithic artifact.
The derived packed fast stack occupies 125,120 tensor bytes instead of 199,808,
a 37.4% reduction; with the model shell it is 36.3% below the compact dense
runtime and 81.6% below the monolithic runtime. That figure excludes an
instrumentation sidecar because the candidate is deliberately fast-only.

These timings are backend- and hardware-specific. The executor is also
deliberately specialized to the fixed, unpadded eight-token task. As with the
composition result, this is an exploratory result for one checkpoint rather
than a seed-replicated claim.

## Saved artifacts

`artifacts/associative_recall/` contains:

- `checkpoint.pt`: model configuration, selected weights, metrics, seeds, and
  exact split IDs;
- `split_manifest.json`: grouped split membership and hashes;
- `training_metrics.jsonl`: evaluation history;
- `fisher_modes.pt`: Fisher matrices, pooled and position-conditioned means,
  spectra, eigenvectors, transitions, modal Jacobians, and checkpoint hash;
- `fisher_report.json`: machine-readable build diagnostics;
- `fisher_report.md`: concise human-readable build results;
- `intervention_report.json`: full machine-readable necessity, control,
  energy-matching, position, bootstrap, and sufficiency results;
- `intervention_report.md`: concise human-readable intervention findings;
- `intervention_results.csv`: flat rows for analysis and plotting;
- `modal_executor.pt`: portable position-conditioned nonlinear graph replacing
  layer 0;
- `modal_executor_report.json`: fit provenance, validation width selection,
  activation fit, behavior, and size diagnostics;
- `modal_executor_report.md`: concise layer-replacement findings;
- `modal_completion_input.pt`: shared input-tail completion bridge;
- `modal_completion_output.pt`: position-local output-tail completion bridge;
- `modal_completion_report.json`: fit protocol, selection, local recovery,
  behavior, hashes, and compute accounting;
- `modal_completion_report.md`: concise conditional-completion findings;
- `modal_executor_layer_1.pt`: portable nonlinear graph replacing layer 1;
- `modal_executor_layer_1_report.json` and
  `modal_executor_layer_1_report.md`: layer-1 selection, behavior, and compute
  diagnostics;
- `modal_completion_layer_1_input.pt` and
  `modal_completion_layer_1_output.pt`: layer-1 input diagnostic and runtime
  output-completion bridge;
- `modal_completion_layer_1_report.json` and
  `modal_completion_layer_1_report.md`: layer-1 completion findings;
- `modal_composition_report.json` and `modal_composition_report.md`: locked
  two-layer validation/test contracts, provenance, and aggregate compute
  accounting;
- `fused_modal_stack.pt`: frozen weights-only logical and algebraically folded
  two-layer runtime retained for backward compatibility;
- `fused_modal_runtime.pt`: compact seven-tensor fast runtime with verified
  lazy references to the four existing logical executor/completion sidecars;
- `fused_executor_report.json` and `fused_executor_report.md`: numerical
  equivalence gate, lazy dispatch/cache/eviction contract, arithmetic, storage,
  provenance, the original four-system benchmark, and a validation-only
  five-system packed-triangular benchmark bound to the lazy artifact.
- `mlx_metal_benchmark.json` and `mlx_metal_benchmark.md`: separate
  same-device Apple-Silicon stack measurements for dense compiled MLX,
  ordinary packed MLX, and the custom packed Metal kernel; these are not part
  of the authenticated runtime manifest.

Layer 0 keeps the original unsuffixed filenames for backward compatibility.
For layer \(i>0\), executor artifacts use
`modal_executor_layer_<i>.pt` and
`modal_executor_layer_<i>_report.{json,md}`; completion artifacts use
`modal_completion_layer_<i>_{input,output}.pt` and
`modal_completion_layer_<i>_report.{json,md}`.

`fisher-graph-verify` reloads the checkpoint and artifacts independently,
re-evaluates the trained model, checks the checkpoint hash, verifies split
identity, and tests PSD spectra, eigenvector orthogonality, Fisher
reconstruction, modal round trips, tensor finiteness, executor hashes,
standalone replacement for both layers, causal-prefix invariance, both layers'
completion bridges, and saved metrics. The frozen composition workflow also
checks the locked runtime hashes, exact layer-boundary identity, same-input
layer-1 contract, absence of teacher mutation, and saved two-layer behavior.
The fused verification reloads both fused artifacts, checks exact modal
boundary compatibility, causal kernels, the compact artifact's seven-key state
and sidecar manifest, and zero-load fast execution. It recomputes
logical-versus-fused validation and test behavior, verifies one authenticated
lazy load, cache reuse, intervention dispatch, and eviction, validates
arithmetic and storage accounting, and treats benchmark timings as positive
hardware observations rather than exact reproducibility targets. For format-3
reports it independently reconstructs the packed candidate, verifies all 36
causal pairs and its 125,120-byte state, recomputes validation equivalence,
proves zero sidecar access and no test usage, and checks every five-system
timing and derived ratio. Format-2 reports remain supported.
The test suite additionally checks position-conditioned and fractional
suppression, downstream propagation, autograd, invalid interventions, and
standard-versus-graph executor equivalence under the same intervention.
