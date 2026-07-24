# Fisher Graph Transformer

A small decoder-only transformer built to expose its computation, derive
activation-space Fisher modes, and replace transformer blocks with graph
executors.

This repository now contains a complete reference run:

- a trained two-pair associative-recall transformer;
- explicit activation and gradient capture;
- full 32 x 32 empirical Fisher matrices at six residual-stream boundaries;
- eigendecomposed, reusable width-wise compute modes;
- a position-conditioned Fisher-mode intervention "equalizer";
- held-out necessity, control, localization, and sufficiency sweeps;
- position-coupled modal layer Jacobians;
- a position-conditioned modal bottleneck and standalone causal modal executor;
- causal conditional completion of discarded boundary modes;
- independently compiled layer-0 and layer-1 executors composed into a frozen
  two-layer modal runtime;
- a compact seven-tensor fused executor whose inspectable logical graph loads
  lazily for activation capture or interventions, with measured end-to-end
  CPU speedups;
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

## Optimization summary

[![Three-panel optimization summary comparing arithmetic, CPU latency, and resident tensor storage](docs/images/fused-executor-optimization.svg)](docs/images/fused-executor-optimization.svg)

The committed SVG is generated directly from the authenticated fused-executor
report. Its source hash is embedded in the image, and the test suite rejects a
stale figure if the benchmark JSON changes. The triangular arithmetic bar is an
available backend specialization, not a measured speedup.

## Reproduce the build

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

fisher-graph-experiment
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

The equivalent module commands are:

```bash
python -m fisher_graph.associative_experiment
python -m fisher_graph.intervention_experiment
python -m fisher_graph.modal_executor_experiment
python -m fisher_graph.modal_completion_experiment
python -m fisher_graph.modal_executor_experiment --layer-index 1 --routing-widths 4 6 8 12 16 24
python -m fisher_graph.modal_completion_experiment --layer-index 1
python -m fisher_graph.modal_composition_experiment
python -m fisher_graph.fused_executor_experiment
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
monolithic fused, and compact lazy systems. It regenerates the authenticated
runtime manifest after writing the final benchmark report.

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
from fisher_graph import FusedToyTransformer, load_lazy_fused_modal_stack
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
| Causal-triangular nonzero opportunity | 30,336 | 21.783% |

The current PyTorch executor uses dense `einsum` kernels containing structural
causal zeros, so 49,152 is the executed contraction count. The 30,336 figure
is an available next optimization for a triangular or sparse specialized
backend, not a speedup already claimed.

The full fixed-length model was benchmarked on the recorded arm64 CPU with one
PyTorch intra-op thread, inference mode, adaptive iterations, nine rotating
measurement rounds, and input construction outside the timed region:

| Batch | Teacher | Logical modal | Monolithic fused | Compact lazy | Lazy vs logical |
|---:|---:|---:|---:|---:|---:|
| 1 | 108.566 us | 173.011 us | 54.645 us | 54.785 us | 3.158x |
| 8 | 200.643 us | 235.553 us | 67.853 us | 67.949 us | 3.467x |
| 64 | 751.302 us | 358.892 us | 109.984 us | 108.006 us | 3.323x |
| 256 | 2,734.323 us | 617.775 us | 230.706 us | 227.514 us | 2.715x |

Across these batches, the compact/monolithic geometric-mean latency ratio was
0.993x; the maximum observed ratio was 1.003x. No hard latency gate was used.

The compact runtime changes the storage tradeoff:

| Resident state | Bytes |
|---|---:|
| Backward-compatible monolithic full runtime | 713,920 |
| Compact full runtime during ordinary inference | 205,952 |
| Logical sidecar added after instrumentation | 203,648 |
| Compact full runtime with sidecar cached | 409,600 |

The ordinary-inference footprint is 71.2% below the monolithic runtime. Even
after instrumentation is loaded it remains 42.6% below it, because the compact
runtime does not retain the monolithic artifact's redundant logical and folded
copies. These are resident tensor bytes for the complete model shell; the four
sidecar files occupy 244,724 bytes on disk and are read only on first
instrumentation. The compact artifact plus those reusable sidecars occupies
451,617 bytes on disk, 37.4% below the 721,471-byte monolithic artifact.

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
  provenance, and four-system steady-state CPU benchmark.

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
hardware observations rather than exact reproducibility targets.
The test suite additionally checks position-conditioned and fractional
suppression, downstream propagation, autograd, invalid interventions, and
standard-versus-graph executor equivalence under the same intervention.
