# Structured layer executor

The structured layer rung separates two parts of Fisher compilation that were
previously conflated:

- Fisher matrices and Jacobians describe activation sensitivity and transport.
- Model semantics describe the operator grammar that produces those
  activations.

The generic compiler remains responsible for calibration roles, Fisher
estimation, Jacobian transport, modal decomposition, graph construction,
artifact provenance, and fidelity gates. A model adapter now attaches a
portable `TransformerLayerSemantics` record to each structured `LayerSpec`.
That record contains:

- the ordered residual stages;
- normalized-input, raw-operator, normalized-delta, and residual-output
  activation sites for each stage;
- branch and Q/K normalization semantics;
- attention projection bias, dropout, and logit softcap behavior;
- gated feed-forward topology, activation, and intermediate width.

`AttentionSpec` continues to own query/KV head topology, head dimension, query
scaling, causal visibility, sliding-window size, RoPE, and cache policy. These
records contain no source modules or source tensors and are included in the
adapter semantic fingerprint.

## Gemma 3 layer grammar

For the pinned Gemma 3 270M checkpoint, one decoder layer is:

```text
attention_state =
  input
  + post_attention_rmsnorm(
      grouped_query_rope_attention(input_rmsnorm(input))
    )

output =
  attention_state
  + post_feed_forward_rmsnorm(
      gated_gelu_mlp(pre_feed_forward_rmsnorm(attention_state))
    )
```

The adapter records the checkpoint's exact width-640 structure:

- four query heads and one KV head;
- head dimension 256 and query scale 1/16;
- separate Q/K Gemma RMSNorm;
- bias-free Q/K/V/O projections;
- layer-local sliding or global visibility and RoPE policy;
- four residual-width Gemma RMSNorm modules;
- bias-free gated GELU-tanh MLP with intermediate width 2048.

Gemma RMSNorm is represented explicitly as FP32 RMS computation followed by
`1 + weight` scaling and a cast back to the input dtype.

## Repo-owned executor

`StructuredTransformerLayerExecutor` implements that grammar without
instantiating, retaining, or calling a Transformers layer. It supports:

- grouped-query attention and KV-head expansion;
- Q/K RMSNorm before RoPE;
- default or linear-scaled RoPE from `SequenceContext.logical_positions`;
- global or sliding causal masks from tensor order during cache-free prefill;
- FP32 attention softmax;
- optional attention-logit softcap;
- sandwich Gemma RMSNorm residual branches;
- gated GELU, GELU, or SiLU feed-forward operators;
- invalid-row passthrough;
- exact logical parameter/MAC accounting;
- source-free artifact round-trip and execution fingerprints;
- a storage-matched attention-output-disabled control.

The current contract is prefill-only with equal query/key lengths and no
cache. Packed or gapped `position_ids` require an explicit attention mask;
without one, the adapter and executor fail closed because native Gemma assigns
additional packed-sequence meaning to position resets. Decode, KV-cache
updates, chunked prefill, multimodal masks, and backend kernel claims remain
out of scope.

## Operator-parity control

A test-only weight-transplant method can copy a native Gemma layer into the
repo-owned implementation. This is an implementation control, not a compiled
candidate. A transplanted executor is marked contaminated and refuses artifact
serialization.

The automated tiny-Gemma tests cover sliding and global layers, GQA, padding,
gapped nonzero RoPE positions, tensor-order causal visibility, future
exclusion, sliding-window exclusion, and boundary parity. A local check
against the pinned cached 270M checkpoint measured exact float32 equality at
layer 4:

| Check | Maximum absolute error |
|---|---:|
| Ordinary model forward vs segmented adapter logits | 0 |
| Native layer 4 vs transplanted structured boundary | 0 |
| Native suffix logits vs transplanted structured suffix logits | 0 |

Both layer implementations contained 5,573,632 parameters. This establishes
operator and adapter parity only. It is neither a learned replacement nor a
compression result.

## Structured distillation

`capture_structured_layer_targets` records detached native targets for:

1. normalized attention input;
2. raw attention output;
3. post-attention normalized delta;
4. post-attention residual;
5. normalized feed-forward input;
6. raw feed-forward output;
7. post-feed-forward normalized delta;
8. final layer output.

`estimate_structured_layer_scales` establishes a calibration-A coordinate
scale for every target before fitting. It starts with per-coordinate RMS but,
by default, floors every coordinate at the median RMS of its own stage (and
at the absolute `1e-4` floor). That robust lower bound prevents an exactly
dormant native coordinate from assigning effectively unbounded loss to an
ordinary random-student error. The coefficient is configurable from zero to
one and is recorded with the effective scales and calibration-A valid-row
count.

Both those scales and the Fisher quadratic are bound to the layer ID, output
site, source-segment fingerprint, and calibration-split digest.
`StructuredOutputFisherMetric.from_raw_fisher` converts a raw
activation-space Fisher matrix into the standardized coordinates used by the
loss, validates symmetry and positive semidefiniteness once, and removes
tolerated numerical negative eigenvalues.
`structured_layer_distillation_loss` then supervises each target separately.
This closes the earlier loophole where suffix CE/KL could reward a useful
intervention that did not transcribe the native layer.

## Learned single-layer fidelity runner

`fisher-graph-gemma-structured-layer` connects those pieces to the existing
four-role Gemma protocol:

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

The runner retains the earlier optimizer-based format-3/4 path, but the
successful parent uses format 5. It selects the lowest-SHA 8,192 valid
calibration-A token rows before activation capture, then identifies Q/K/V/O,
gate/up/down, Q/K norm, and four residual RMSNorm coefficient tensors with
deterministic active-support ridge or coordinate least squares. The
calibration-A feed-forward input has exact codimension one; the preregistered
active-support policy permits at most that one-dimensional structural
nullspace and solves only on the numerically active eigenspace. There are no
optimizer, local-warmup, or suffix-distillation updates.

The bootstrap API receives compact activation pairs and a destination
executor. It is never passed a source module or source parameter. The
containing experiment necessarily executes the source model to capture those
activations and bind source/model provenance; therefore "activation-only"
describes the compiler boundary, not the entire experimental process.

Calibration B remains a one-shot lock. Validation is not tokenized unless the
strict-reloaded primary executor passes behavior, final block-delta,
attention-delta, feed-forward-delta, native-replay, ordinary-vs-segmented
parity, and zero-source-call gates. Test remains parse-and-hash-only.

The format-5 artifact contains two source-free executor states, authenticated
operator-bootstrap and compact-row reports, the activation Fisher and
structured scales used by the surrounding audit, the exact closed-form
recipe, and aggregate calibration-A/heldout audits. It contains no source
model state dict, prompt text, tokenizer state, teacher logits, captured
teacher activations, or retained bootstrap sufficient statistics. The strict
loader recomputes coefficient fingerprints, source-free execution
fingerprints, gates, accounting, status, and the tensor-free JSON report.

The attention-output-disabled candidate is a diagnostic control. Parameter
and logical-MAC ratios are also diagnostic: the primary executor deliberately
uses native shapes and is expected to remain near one source layer in both
measures. Neither the control nor those ratios can block this fidelity rung.

### Development probe

An ignored, non-promotional layer-4 probe used four calibration-A prompts, two
calibration-B prompts, and only 30 optimizer updates. It was useful for
checking loss conditioning, not fidelity:

| Scale lower bound | First/last local loss | B NLL delta/token | B teacher KL/token | B top-1 agreement |
|---|---:|---:|---:|---:|
| absolute `1e-4` only | 8,592,019 / 7,822,680 | +0.516 | 0.526 | 64.1% |
| full stage median | 7.09 / 2.42 | +0.307 | 0.410 | 69.2% |

The median floor removed the dormant-coordinate optimizer pathology and
improved held-out behavior, but the candidate still failed the strict
calibration-B branch and behavior gates: block-delta NRMSE remained about
1.00. Validation therefore stayed un-tokenized. Ordinary-vs-segmented native
parity, native boundary replay, and source-call audits were exact; the student
made zero source-layer calls. Parameter and logical-MAC ratios were both 1.0,
as expected for this source-shaped fidelity rung. This tiny probe is not
evidence of general viability; the next meaningful run needs a fresh,
representative corpus and the preregistered fixed training schedule.

### Preregistered 270M representative result

The first full layer-4 representative attempt used the pinned
`google/gemma-3-270m` revision, 280 calibration-A prompts and 64 prompts in
each heldout role, 32 role-disjoint families, 56,210 supervised A tokens, and
the fixed 400-step local plus 2,800-step suffix-distillation schedule. The
corpus had zero exact prompt overlap with prior local corpora and covered four
length buckets, but it was deterministic synthetic templated text rather than
a naturalistic language sample.

The format-3 artifact strict-reloaded successfully and proved that the
executor path was real: ordinary/segmented native parity and native boundary
replay had zero error, both students made zero source-layer calls, and the
structural causal/padding probes passed. Calibration B nevertheless failed
every fidelity gate:

| Calibration-B metric | Required | Structured executor | Attention-disabled control |
|---|---:|---:|---:|
| Absolute delta NLL/token | at most 0.05 | 0.147163 | 0.149890 |
| Teacher KL/token | at most 0.05 | 0.079517 | 0.123664 |
| Aggregate top-1 agreement | at least 0.95 | 0.865246 | 0.845996 |
| Per-prompt p90 absolute delta NLL | at most 0.10 | 0.601329 | 0.650907 |
| Per-prompt p10 top-1 agreement | at least 0.90 | 0.703704 | 0.760000 |
| Block-delta NRMSE | at most 0.02 | 0.996820 | 0.998268 |
| Block-delta cosine | at least 0.999 | 0.088936 | 0.069126 |
| Attention-delta NRMSE / cosine | 0.02 / 0.999 | 0.996684 / 0.095439 | 1.000000 / 0 |
| Feed-forward-delta NRMSE / cosine | 0.02 / 0.999 | 0.998219 / 0.071312 | 0.998464 / 0.065109 |

The negative NLL delta is not a fidelity success: it means the replacement
changed ground-truth likelihood while its KL, predictions, and especially
native branch deltas remained far from the teacher. Validation and test were
never tokenized, and compression was not started.

An A-side postmortem reproduced the same approximately 0.997 direct NRMSE, so
this was not ordinary B-only overfitting. The source layer's four residual
RMSNorm weights reach 162, 221, 51.75, and 608, while the zero-initialized
student weights remained below 0.95 after 3,200 updates at `3e-4`. At the same
time, two coordinates carry 97.75% of attention-delta scale energy and 99.19%
of feed-forward-delta scale energy. Per-coordinate standardization made
missing those enormous channels look like only two ordinary coordinate
errors even though raw delta energy was nearly absent.

Format 4 encoded the activation-derived RMSNorm initialization, mixed
coordinate/global-energy loss, and an A-side direct-fidelity preflight. Format
5 supersedes that optimizer repair for the successful parent: it directly
identifies all seven linear operators and all six normalization vectors from
calibration-A activation pairs, uses zero optimizer steps, then requires the
same strict A-side round-trip and direct/branch gates before calibration B.

### Format-5 activation-only v6 result

The full-width layer-4 v6 run used the same pinned
`google/gemma-3-270m` revision, 280 calibration-A prompts, and 64 prompts in
each heldout role. The compiler used 8,192 deterministically selected A rows.
The strict-loaded source-free executor passed calibration B and the one
allowed validation evaluation:

| Metric | Calibration B | Validation |
|---|---:|---:|
| Block-delta NRMSE | `9.137013e-7` | `9.212765e-7` |
| Block-delta cosine | `0.999999999999583` | `0.999999999999576` |
| Full-output NRMSE | `3.794613e-7` | `3.819461e-7` |
| Delta NLL/token | `-1.941651e-8` | `-3.136979e-8` |
| Teacher KL/token | `-1.762323e-9` | `3.128625e-9` |
| Aggregate top-1 agreement | `1.0` | `1.0` |
| Per-prompt block NRMSE p90 | `1.033516e-6` | `1.019390e-6` |

All four token-length buckets passed. The attention-disabled negative control
was clearly separated on calibration B: block NRMSE was `0.652747`, cosine
was `0.785139`, delta NLL/token was `+0.721448`, teacher KL/token was
`0.846907`, and top-1 agreement was `0.535714`. The direct and reloaded
primary executor made zero calls to native layer 4. Test remains sealed.

This is the stronger parent result needed to start compression: one
native-shaped Gemma layer can be reconstructed from activation observations
and executed source-free at essentially numerical precision. Because width
and operator geometry are unchanged, it provides no parameter or compute
reduction by itself and says nothing yet about consecutive-layer or
whole-model stability.

## Compression ladder

Format 5 supplies the required full-width fidelity parent. Fidelity and
compression remain separate decisions:

```text
source-shaped structured fidelity
  -> shrink MLP width
  -> shrink attention projection/head geometry
  -> introduce Fisher/modal residual rank
  -> compile consecutive layer blocks
  -> lower and benchmark fused kernels
```

No width rung may start from a parent that failed strict representative
calibration-B and validation gates. The format-5 v6 parent passed that
requirement. Its calibration A may construct and select a compression
candidate, but the width rung cannot reuse the parent's already opened
calibration B. It therefore receives a new, exclusive
compression-B/validation/test protocol.

For Gemma 3 270M layer 4, the parameter decomposition is:

| Component | Parameters |
|---|---:|
| Attention linears | 1,638,400 |
| Gated MLP | 3,932,160 |
| Six RMSNorm vectors | 3,072 |
| Total | 5,573,632 |

The MLP owns about 71% of the layer parameters, so the first interpretable
compression ladder is `2048 -> 1536 -> 1024 -> 768 -> 512` while attention,
RoPE, residual width, and fidelity gates stay fixed.

### First `2048 -> 1536` Fisher/Taylor rung

The A-only builder cuts the native graph at the layer-4 MLP down-projection
input \(z\), substitutes an equal-value detached leaf, and differentiates the
summed hard-target causal NLL through the frozen suffix. For unit \(j\), it
ranks the calibration-A mean

\[
s_j = \mathbb{E}_{\text{valid rows}}
      \left[\left(z_j\frac{\partial \mathcal{L}}{\partial z_j}\right)^2\right].
\]

Stable top-k selection keeps 1,536 complete units, slices matching gate/up
rows and down-projection columns, and activation-only ridge refits only the
new down projection. Attention tensors are byte-identical before and after
the refit. The source model is frozen and has no parameter gradients during
score collection.

This quantity is deliberately only a **diagonal per-token Fisher/Taylor
proxy**, not a full Fisher matrix. Each row's gradient includes the real
causal suffix effect, but the selector discards off-diagonal coupling between
MLP units, cross-token Fisher blocks, and the higher-order interaction from
removing 512 units simultaneously. "Retained score fraction" must therefore
not be read as retained full Fisher information.

Build the candidate from the format-5 parent and its v6 calibration A, then
evaluate it with a separate fresh-v7 ledger:

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

The heldout ledger namespace is independent of the parent fidelity ledger.
Its claim is irreversible and precedes B tokenization. Never rerun a consumed
compression-B split, including after a post-claim process failure.

On 60,054 valid v6 calibration-A rows, the selected units retained
`0.964939976` of this proxy score. The down-projection refit reduced its
operator NRMSE from `0.166783` to `0.038493`, and the strict-loaded compressed
candidate passed the preregistered A fidelity gate:

| Calibration-A metric | Result |
|---|---:|
| Block-delta NRMSE / cosine | `0.015280716` / `0.999883245` |
| Full-output NRMSE | `0.006165204` |
| Per-prompt block NRMSE p50 / p90 / worst | `0.015233670` / `0.016619463` / `0.019486219` |
| Attention-delta NRMSE | `6.171338e-7` |
| Feed-forward-delta NRMSE | `0.013724820` |
| Source-layer calls | `0` |

The candidate has real structural savings:

| Resource | Source | Compressed | Reduction |
|---|---:|---:|---:|
| Complete layer parameters | 5,573,632 | 4,590,592 | 983,040 (`17.6373%`) |
| MLP-linear MACs per valid token | 3,932,160 | 2,949,120 | 983,040 (`25%`) |
| MLP-linear FLOPs at two FLOPs/MAC | 7,864,320 | 5,898,240 | 1,966,080 (`25%`) |

Those A results authorized one heldout attempt; they did not establish
compression success.

### Fresh v7 calibration-B rejection

The one-shot compression evaluator strict-loaded the A candidate, statically
verified a new v7 corpus with zero family overlap with v6, claimed its
exclusive compression-B ledger immediately before tokenization, and evaluated
64 prompts spanning all four length buckets. It rejected the candidate:

| Calibration-B metric | Gate | Result |
|---|---:|---:|
| Block-delta NRMSE | at most `0.02` | `0.071745022` |
| Block-delta cosine | at least `0.999` | `0.997428291` |
| Feed-forward-delta NRMSE | at most `0.02` | `0.064834090` |
| Feed-forward-delta cosine | at least `0.999` | `0.997896080` |
| Attention-delta NRMSE | at most `0.02` | `8.484155e-7` |
| Delta NLL/token | absolute value at most `0.05` | `-0.003316517` |
| Teacher KL/token | at most `0.05` | `0.016451581` |
| Aggregate top-1 agreement | at least `0.95` | `0.935116258` |
| Per-prompt p90 absolute delta NLL | at most `0.10` | `0.042396637` |
| Per-prompt p10 top-1 agreement | at least `0.90` | `0.911764706` |

Per-prompt block NRMSE was `0.074881` at p50, `0.082763` at p90, and
`0.085971` at worst. The unchanged attention branch remained essentially
exact, isolating the failure to the compressed MLP rather than the graph
replacement machinery. Ordinary-versus-segmented native parity, native
boundary replay, and zero-source-call execution also passed.

For the 10,945 valid v7 tokens, exact analytic accounting measured
52,421,079,040 compressed complete-layer MACs versus 63,180,451,840 source
MACs, a `17.0296%` reduction for those sequence lengths. Parameter reduction
remained `17.6373%`. Because fidelity failed, these are properties of a
rejected candidate, not an accepted deployment.

Compression validation was never tokenized or evaluated, test remains sealed,
and the consumed v7 calibration-B split cannot be retried. The result rejects
this specific diagonal-score `2048 -> 1536` rule. It does not prove that MLP
width compression is impossible, but it supports no whole-model compression,
measured-latency, energy, fused-kernel, or model-level quality claim.
