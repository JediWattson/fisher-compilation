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
