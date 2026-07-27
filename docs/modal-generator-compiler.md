# Modal-generator compiler

The target architecture is a compiler, not an interpreter layered over the
native transformer:

```text
weights
  -> grouped Fisher coupling
  -> parameter clusters
  -> computational modes
  -> modal generators
  -> generator-interaction graph
  -> source-free inference by graph traversal
```

Each arrow has a separate artifact and a separate validation obligation.
Passing an earlier stage does not authorize a later one.

## 1. Weights become natural parameter groups

For a bias-free gated MLP,

\[
z_j(h)=\operatorname{SiLU}(w^{gate}_j h)(w^{up}_j h),
\qquad
\Delta h=\sum_j w^{down}_j z_j(h).
\]

The natural scalar-channel parameter group \(j\) owns the matching gate row,
up row, and down column:

\[
\theta_j =
\left\{
w^{gate}_j,\,
w^{up}_j,\,
w^{down}_j
\right\}.
\]

This grouping starts from the weight topology but does not copy model weights
into an analysis report.

## 2. Prompt-conditioned grouped Fisher coupling

Introduce a virtual multiplicative gate \(\alpha_j\) on channel \(z_j\). For
one independently differentiated prompt \(x\),

\[
r_j(x)
=
\left.\frac{\partial \operatorname{NLL}(x)}
{\partial \alpha_j}\right|_{\alpha_j=1}
=
\sum_t z_{t,j}
\frac{\partial \operatorname{NLL}(x)}{\partial z_{t,j}}.
\]

The prompt contribution to the empirical Fisher matrix in this grouped
coordinate system is

\[
F(x)=r(x)r(x)^\top.
\]

Across a fit split, the coupling is represented implicitly by the
prompt-by-group matrix \(R\):

\[
F_{\mathrm{fit}}=R^\top R.
\]

The implementation must not materialize the full
\(36{,}864\times36{,}864\) matrix. Chunked products and column clustering use
the same coupling exactly.

This is a virtual-gate pullback Fisher over natural MLP parameter groups. It is
not the raw per-parameter Fisher matrix.

## 3. Fisher coupling becomes parameter clusters

Modes whose columns in \(R\) have similar directions are coupled across
prompts. Axial clustering treats \(R_{\cdot i}\) and
\(-R_{\cdot i}\) as the same Fisher axis while retaining their relative
orientation.

For a coordinate cluster \(C\), its prompt signature is exact:

\[
s_C(x)
=
\operatorname{tr}(P_C F(x)P_C)
=
\sum_{j\in C}r_j(x)^2.
\]

Cluster membership is only a discovery result. It does not prove that the
members are interchangeable or that they can be deleted.

One Fisher cluster may span several blocks. Before replacement, the compiler
therefore lowers it into authenticated per-layer fragments. A fragment is the
exact list of native channel indices at one MLP, together with its gate-row,
up-row, down-column parameter count and Fisher mass. This is the object a
physical executor is allowed to remove; cross-layer cluster membership alone
does not authorize deleting an entire global cluster at one site.

## 4. Parameter clusters become computational modes

The native residual contribution of cluster \(C\) is

\[
\Delta h_C(x,t)
=
\sum_{j\in C} w^{down}_j z_j(x,t).
\]

A Fisher-row-weighted, fit-only rank-\(q\) affine output basis \(D_C\) with
mean \(\mu_C\) defines compact computational coordinates

\[
m_C(x,t)=D_C^\top(\Delta h_C(x,t)-\mu_C),
\qquad
\widehat{\Delta h}_C=D_Cm_C+\mu_C.
\]

The rank ladder is a rate-distortion curve. A cluster may need one mode,
several modes, or its complete native width. “One Fisher cluster” does not
mean “one scalar computational mode.”

## 5. Modal generators replace native cluster computation

A modal generator must produce \(m_C\) from values that remain available
after compilation:

\[
\widehat m_C
=
G_C(h,\text{causal state},\text{incoming modal messages}).
\]

It may not consume the native \(z_j\) values that it is intended to replace.
The first implementation uses deterministic reduced-rank linear generators:

\[
\widehat m_C=A_C^\top h+b_C,
\qquad
\widehat{\Delta h}_C=D_C\widehat m_C.
\]

The executable graph node state is \(\widehat m_C\) itself—not an arbitrary
private regression factor. The compiler may fuse \(G_C\) with \(D_C\) for a
conventional dense MLP overlay, but the authenticated graph form keeps this
modal-coordinate boundary explicit so interaction edges have a stable meaning.

Later rungs may use a shared input trunk, nonlinear generators, or conditional
execution. Every rung must count its parameters, multiply-accumulates,
carried state, and routing cost.

Generator fitting is performed against a frozen source model. It does not
fine-tune or mutate the source transformer.

## 6. Generator interactions become a causal graph

A directed edge \(u\rightarrow v\) carries a learned message between compact
modal states:

\[
m_v
=
G_v(h_v)
+
J_{u\rightarrow v}m_u.
\]

Multiple outgoing edges provide fan-out; multiple incoming edges provide
fan-in. Edges must follow the authenticated topological and causal order.
Their matrices are learned weights and therefore count toward storage and
runtime work.

The graph can span modes within one MLP, several transformer blocks, or an
eventual whole-model compiled region. Cross-block edges cannot read a future
state.

## 7. Inference is graph traversal

At runtime the executor:

1. accepts only declared boundary inputs and carried modal state;
2. traverses generator nodes in topological order;
3. accumulates incoming modal messages;
4. generates compact modal coordinates;
5. decodes and sums residual contributions at declared output boundaries;
6. releases modal state after its final consumer.

The all-at-once executor is useful for offline replay. An incremental session
performs the same traversal as transformer layers produce their boundary
inputs, and releases a state after its last fan-out consumer. Removed native
channels are absent from this path. Uncompiled channels may remain in a hybrid
MLP overlay; a whole source MLP is absent only when every one of its parameter
groups has a validated replacement. Instrumentation may optionally retain node
states and edge messages, but the default execution path does not.

## Validation ladder

The first external-model experiment remains development-only:

1. collect exact prompt-conditioned grouped scores on calibration-A fit;
2. freeze Fisher clusters;
3. collect native cluster contributions and cheap generator inputs;
4. fit a predeclared generator-rank ladder;
5. evaluate held-out development prompts;
6. build a generator graph only from clusters that beat matched deletion;
7. compare graph traversal with native execution using residual NRMSE,
   logit cosine, KL per token, NLL, and top-1 agreement;
8. report total model parameters, linear MACs per token, graph state width,
   and worst-family behavior.

Calibration-A guard, calibration B, validation, and test remain closed until
the complete source-free graph passes the development gate. A successful
cluster or generator fit alone is not a model-compression result.

## First live Gemma development rung

The first end-to-end development run used the pinned
`google/gemma-3-270m` revision
`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. Both fit and evaluation prompts
came from calibration-A fit-only exports. The provided manifests declared
prompt identities and source indices with no overlap, but all eight semantic
families overlapped. Numerical extraction and split membership are
caller-declared and self-attested: checksums detect artifact mutation, but do
not independently authenticate source membership or prove real-world split
disjointness. This makes the run useful for implementation and rate-curve
decisions, not for a generalization claim.

Whole-model grouped Fisher clustering selected a 54-channel layer-17
fragment. The computational-mode ladder showed:

| mode rank | eval weighted NRMSE | retained centered energy |
| ---: | ---: | ---: |
| 1 | 0.204166 | 54.1937% |
| 4 | 0.118731 | 84.4779% |
| 8 | 0.084964 | 92.4902% |
| 16 | 0.057959 | 96.4906% |
| 32 | 0.029680 | 99.1153% |
| 64 | 0.000000 | 100.0000% |

The predeclared rank-32 basis was then held fixed. Its generator ladder
predicted those coordinates from the layer input:

| generator rank | eval weighted NRMSE | eval weighted cosine | parameters |
| ---: | ---: | ---: | ---: |
| 1 | 0.705545 | 0.708666 | 704 |
| 4 | 0.406388 | 0.913702 | 2,720 |
| 8 | 0.289559 | 0.957165 | 5,408 |
| 16 | 0.201444 | 0.979509 | 10,784 |
| 32 | 0.127392 | 0.991877 | 21,536 |

The predeclared rank-16 generator was first lowered into the executable graph
form. The graph keeps its 32-coordinate computational state visible for later
edges and stores 31,904 parameters. It physically replaces 103,680 native
gate/up/down parameters, yielding 71,776 net stored parameters saved
(`69.23%` inside the selected fragment and `0.0268%` of the 268.1M-parameter
model). Its 31,232 logical MACs per valid token replace 103,680 native logical
MACs, saving 72,448 (`69.88%`) before any interaction edges are added. Bias and
accumulation work is reported separately: this one-node graph performs 672
elementwise additions per valid token.

Coordinate-space prediction error is not the final residual error: after the
rank-32 basis decodes the selected rank-16 generator, its development
Fisher-weighted residual NRMSE is `0.064897`.

The same isolated node can be algebraically fused into a conventional residual
matrix after graph optimization. That comparison stores 21,120 parameters and
saves 82,560 (`79.63%`) inside the fragment, or `0.0308%` of the full model.
It performs 20,480 matrix MACs plus 640 bias additions per valid token, saving
83,200 matrix MACs (`80.25%`) versus the native fragment.
The difference is the explicit price of preserving a modal interface that
future nodes can fan out from or fan in to; it is not extra fidelity.

On 10,200 supervised development positions drawn from 10,240 valid tokens,
the primary incremental graph traversal produced:

| condition | NLL/token | delta NLL | native KL/token | native top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| generated | 2.816920 | -0.002882 | 0.001819 | 97.8137% |
| matched deletion | 2.893494 | +0.073693 | 0.172642 | 83.8529% |
| native | 2.819802 | 0 | 0 | 100% |

The negative generated delta NLL is not evidence that the compiled model is
better than the source; on this development slice it is small enough to be
sampling variation or a mild regularization effect. The important comparison
is that generated execution stays much closer to native than deleting exactly
the same channels. No measured latency claim is made. The next scientific
gate is a multi-fragment interaction graph frozen before a genuinely fresh,
family-disjoint guard.

Matched deletion is also not the strongest possible trivial control. The
fitted affine mean alone reaches `0.288028` evaluation weighted NRMSE at the
fragment-output level, while the rank-32 computational basis reaches
`0.029680`. A physical end-to-end mean-only replacement must still be added
before attributing the full deletion gap to input-conditioned generation.
Finally, the full analysis `.pt` retains rate ladders and audit state and is
much larger than the deployable weights; parameter savings describe a
stripped runtime artifact, not the development artifact's file size.

This first graph contains one node and therefore no interaction edge. It
validates checksummed lowering, physical source compaction, coordinate-state
lifetime, traversal, and resource accounting. It does not yet validate learned
fan-out or fan-in; that requires compiling multiple fragments together.

## Multi-fragment terminal fan-in development rung

The next rung reused the strict single-fragment v3 artifact's fit-only prompt
trace, parameter catalog, grouped Fisher, cluster plan, and fragment plan.
It did not recompute or select clusters from the evaluation export. From the
fit Fisher ordering it selected the top four eligible fragments on distinct
layers:

| causal layer | cluster | native channels | native parameters/MACs |
| ---: | ---: | ---: | ---: |
| 10 | 28 | 72 | 138,240 |
| 11 | 28 | 76 | 145,920 |
| 16 | 0 | 46 | 88,320 |
| 17 | 0 | 54 | 103,680 |
| **total** |  | **248** | **476,160** |

Every node used the predeclared rank-32 computational basis and rank-16
coordinate generator. Mode-rate ladders were clipped to each fragment's
structural rank, so layers 16 and 17 stop at rank 32 rather than inventing a
rank-64 numerical null-space completion.

The source evaluation export contained 40 prompts. A content-hashed,
deterministic rule split it into:

- 20 open-development selection prompts used for node curves, ridge choice,
  and greedy edge acceptance; and
- 20 disjoint open-development assessment prompts that were not evaluated
  until the graph was frozen.

The artifact authenticates the source export, raw partition plan, both
partition memberships, tokenized content hashes, and their exact union. Prompt
text is not stored. This is still caller-declared, self-attested
calibration-A development data. The two halves overlap in semantic families,
so the assessment is not a fresh-family guard.

### Physical interaction fitting

The edgeless four-node overlay was executed before fitting edges. Temporary
compact-MLP pre-hooks captured:

1. each generator's actual modal state on the shifted compiled trajectory; and
2. the removed native gate/up/down contribution recomputed at that exact
   shifted normalized input, then encoded in the frozen target basis.

Rows were aligned by exact `(prompt SHA-256, logical position)` keys. The
interaction target was the remaining layer-17 coordinate residual. Restricting
all candidates to the terminal node makes the fitted source states invariant
to edge acceptance:

```text
layer 10 state ─┐
layer 11 state ─┼─> layer 17 modal residual
layer 16 state ─┘
```

Greedy Fisher-weighted selection accepted every candidate:

| accepted order | edge | ridge | selection weighted-NRMSE improvement |
| ---: | --- | ---: | ---: |
| 1 | layer 16 → layer 17 | 0 | 0.153661 |
| 2 | layer 10 → layer 17 | 0.0001 | 0.028802 |
| 3 | layer 11 → layer 17 | 0.0001 | 0.013093 |

The cumulative selection weighted NRMSE fell from `1.000000` with no messages
to `0.804444` with all three. These numbers selected the graph; they are not
the final quality measurement.

### Frozen assessment result

The final comparison held node weights and physical replacement scope
identical between the interacting and edgeless graphs:

| condition | NLL/token | delta NLL | native KL/token | native top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| native | 2.823914 | 0 | 0 | 100% |
| interacting graph | 2.834824 | +0.010909 | 0.038655 | 91.8824% |
| identical edgeless graph | 2.839795 | +0.015880 | 0.040344 | 91.3333% |
| dense-fused edgeless | 2.839795 | +0.015880 | 0.040344 | 91.3333% |
| matched deletion | 3.551284 | +0.727369 | 0.675984 | 66.4314% |

The assessment contains 5,100 supervised positions and 5,120 valid tokens.
Relative to the edgeless graph, interactions recover `0.004971` NLL/token,
reduce native KL by `0.001688` (`4.18%`), and improve native top-1 agreement
by `0.549` percentage points. They reduce the edgeless NLL penalty by `31.30%`.
The dense-fused and edgeless graph controls agree within a maximum absolute
supervised-logit difference of `7.25e-5` under a recorded `1e-4`,
absolute-only float32 tolerance.

This establishes that cross-layer modal messages add measurable fidelity on
an assessment partition that did not select them. It does not establish
generalization beyond the same prompt families.

### Resource result

| physical candidate | replacement parameters | matrix MACs/token | local parameter savings | local MAC savings |
| --- | ---: | ---: | ---: | ---: |
| interacting graph | 130,784 | 128,000 | 72.53% | 73.12% |
| identical edgeless graph | 127,616 | 124,928 | 73.20% | 73.76% |
| dense-fused edgeless | 84,480 | 81,920 | 82.26% | 82.79% |

The three interaction matrices add 3,168 parameters and 3,072 matrix MACs per
token. The interacting graph also performs 2,880 separately reported
elementwise additions per token. Its 345,376 saved parameters are only
`0.1288%` of the 268,098,176-parameter source model; breadth, not local rate,
is now the dominant compression limitation. No kernel-latency claim is made.

The result artifact remains ignored at:

```text
.local-runs/google--gemma-3-270m/
  modal-generator-multifragment-fanin-dev-v1.{pt,json}
```

Its scientific payload digest is
`682e68278cdd1d56a6bca2d4b427d3c868fa861c6d6b65da25086bc2016f17e0`.

### What this rung does not settle

The layer-16 edge selected ridge zero and has a very large coefficient scale
(maximum absolute entry about `4.19e7`). The corresponding source modal
coordinates are small enough that runtime messages remain useful, and the
edge improves both selection and assessment behavior, but the coefficient
scale signals poor conditioning. It is unsafe to treat this graph as
quantization-ready or robust under distribution shift.

The immediate stability rung should whiten or variance-normalize node states,
exclude zero ridge, and compare full-rank with low-rank interaction messages.
Only after the edge advantage survives that frozen sensitivity check should
the graph be opened on a genuinely fresh, family-disjoint guard. Calibration
B, validation, and test remain unopened.

## All-layer breadth development rung

The first breadth experiment selected one top-Fisher fragment on each of the
18 native Gemma MLP layers. It compiled 1,183 channels and 2,271,360 native
MLP parameters. This was deliberately a coverage test, not exhaustive MLP
replacement.

| condition | NLL/token | delta NLL | native KL/token | native top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| native | 2.823914 | 0 | 0 | 100% |
| 12-edge terminal fan-in graph | 2.846954 | +0.023039 | 0.121558 | 85.8235% |
| identical edgeless graph | 2.839176 | +0.015261 | 0.122093 | 85.7647% |
| dense-fused edgeless | 2.839176 | +0.015261 | 0.122093 | 85.7647% |
| matched deletion | 5.431380 | +2.607466 | 2.372614 | 44.7451% |

All 12 accepted edges used nonzero ridge and had bounded coefficient scales.
They slightly improved KL and top-1 agreement over the edgeless graph, but
worsened NLL. The exhaustive rung therefore used the simpler edgeless
executor so a weak interaction fit could not obscure the layer-replacement
question.

| physical candidate | replacement parameters | matrix MACs/token | local parameter savings | whole-model parameter savings |
| --- | ---: | ---: | ---: | ---: |
| interacting graph | 586,944 | 574,464 | 74.16% | 0.6283% |
| identical edgeless graph | 574,272 | 562,176 | 74.72% | 0.6330% |
| dense-fused edgeless | 380,160 | 368,640 | 83.26% | 0.7054% |

This resolved the breadth bottleneck only structurally: one fragment per layer
still touches too little of the source model to produce material whole-model
compression.

The ignored artifact is:

```text
.local-runs/google--gemma-3-270m/
  modal-generator-all-layer-fanin-dev-v1.{pt,json}
```

Its scientific payload digest is
`23075f88fd926b1de5fd82c37b873e2f5cd5a8d980dcec31281890d625c45842`.

## Exhaustive full native MLP-stack development rung

The exhaustive runner aggregates the 64 Fisher-cluster fragments on each
layer into one authenticated layer superfragment before fitting
computational modes. The resulting plan proves exact, disjoint coverage of:

- 18 transformer MLP layers;
- 1,152 source Fisher fragments;
- all 36,864 intermediate channels; and
- 70,778,880 native gate/up/down parameters.

This is a full **MLP-stack** replacement. It is not a whole-transformer
replacement: embeddings, attention, normalization, and the language-model
head remain native.

### Frozen protocol

Fit40 and a deterministic selection20 partition were materialized first. One
native gradient replay per split collected aligned normalized MLP inputs,
complete MLP residual contributions, and summed Fisher row weights for all 18
layers. There were 5,711 fit rows and 5,120 selection rows per layer.

Each layer used:

- a predeclared rank-640 computational-mode basis;
- a predeclared rank-640 coordinate generator;
- ridge `1e-6`; and
- a dense residual plan that is executable without a singular-fragment
  lowering.

Rank 640 is the complete residual width, so the mode basis reconstructs every
native layer contribution at numerical precision. It intentionally removes
rank truncation as an explanation for failure. The remaining approximation is
the affine generator from normalized layer input to the native nonlinear MLP
residual.

After all 18 generators and the physical executor were frozen, the other 20
prompts were tokenized as open-development assessment data. Generated and
matched-deletion conditions replaced the exact same 18 MLPs; deletion merely
skipped generator work.

### Full-stack assessment

| condition | NLL/token | delta NLL | perplexity | native KL/token | native top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 2.823987 | 0 | 16.84 | 0 | 100% |
| generated full MLP stack | 3.172463 | +0.348476 | 23.87 | 0.456653 | 74.0588% |
| matched deletion | 13.902236 | +11.078248 | 1,090,597 | 11.108717 | 0.2353% |

The assessment contains 5,100 supervised positions and 5,120 valid tokens.
Relative to matched deletion, the generators recover `96.85%` of the NLL
penalty, `95.89%` of the native-to-candidate KL, and 73.82 percentage points
of native top-1 agreement. That is strong evidence that the fitted generators
carry real MLP computation.

It is not yet sufficient fidelity for compression. The generated model's
perplexity is `1.417x` native, and native top-1 agreement is not downstream
task accuracy. No calibration B, family-disjoint guard, validation, test, or
downstream benchmark has been opened.

### Storage and logical compute

| quantity | native/source | generated candidate | savings |
| --- | ---: | ---: | ---: |
| whole-model learned parameters | 268,098,176 | 212,076,416 | 56,021,760 (`20.90%`) |
| replaced MLP-stack parameters | 70,778,880 | 14,757,120 | 56,021,760 (`79.15%`) |
| MLP matrix MACs / valid token | 70,778,880 | 14,745,600 | 56,033,280 (`79.17%`) |
| generator bias additions / valid token | 0 | 11,520 | -11,520 |

Across assessment20, generated execution performs 75,497,472,000 generator
matrix MACs in place of 362,387,865,600 native MLP matrix MACs. Attention,
the vocabulary head, and other retained work are not included in this local
MLP comparison. These are algebraic operation counts, not measured latency or
a custom-kernel result.

The logical candidate excludes the removed native MLP stack. The development
process does not: it keeps all 268,098,176 source parameters resident beside
14,757,120 compiled parameters so it can compare native and generated
conditions safely. The 452 MB `.pt` artifact is also not a packed model; it
stores duplicated float64 analysis curves and authenticated bases while
excluding all source weights.

At rank 640, the runtime's two square generator factors can be multiplied into
one affine `640 x 640` matrix per layer without changing the mathematical
function. That derived optimization would use 7,384,320 generator parameters
and 7,372,800 MLP matrix MACs per token, raising the potential whole-model
parameter reduction to `23.65%`. It has not yet been implemented or checked
for float32 execution equivalence, so it is not part of the measured result.

### What the full-stack loss localizes

The local selection fits are much stronger than the end-to-end result:

- weighted NRMSE ranges from `0.1146` to `0.2979` across layers; and
- weighted cosine ranges from `0.9553` to `0.9934`.

Each generator was fit against a native layer input. In the simultaneous
18-layer executor, layer \(k\) instead receives states already shifted by
generators \(0,\ldots,k-1\). The gap between good native-trajectory fits and
weaker end-to-end fidelity therefore points to compounded trajectory shift,
not loss in the full-rank computational-mode basis.

The next controlled diagnostic is a frozen replacement trajectory:

1. assess native-prefix/generated-suffix and generated-prefix/native-suffix
   ladders without refitting;
2. locate the first layers where end-to-end NLL and KL accelerate;
3. recollect teachers at the physically compiled prefix trajectory;
4. compare sequentially refit generators with causal Jacobian/message
   corrections; and
5. only then descend a rank ladder and open a fresh family-disjoint guard.

The ignored exhaustive artifact is:

```text
.local-runs/google--gemma-3-270m/
  modal-generator-full-mlp-stack-dev-v1.{pt,json}
```

Its scientific payload digest is
`babed58e93ff09bd65a7ce0062eb8e1f657672f3cc8bcf4e9fb03f446a48f5ec`.
