# Dense supermode compaction

The dense-supermode rung targets a different kind of compression from
codebooks, quantization, sparse pruning, or the earlier pairwise bundler. It
rewrites a group of \(K\) native gated-MLP units as \(R<K\) newly synthesized
units and emits ordinary, physically smaller dense matrices.

For a native gated MLP,

\[
z=\phi(Gh)\odot Uh,\qquad y=Dz,
\]

choose a pool \(S\) containing \(K\) coordinates. Units outside the pool are
copied exactly. The compiler learns \(R\) new coordinates

\[
q_\theta(h)=\phi(\widehat Gh)\odot\widehat Uh
\]

and replaces the pool contribution with

\[
\widehat y_S=\widehat Dq_\theta(h).
\]

The deployed gate and up matrices have width \(W-K+R\), and the deployed down
matrix has the same reduced width. There is no sparse mask, source-width
slot, codebook lookup, or runtime reconstruction of the original pool.

## Coordinate plan

Calibration A supplies:

- normalized MLP inputs \(h\);
- native pre-down activations \(z\);
- suffix score gradients \(s=\partial L/\partial z\); and
- the native down matrix \(D\).

For the selected pool, the planner forms the uncentered activation moment

\[
C=\mathbb E[z_Sz_S^\top].
\]

Uncentered coordinates are intentional: the current bias-free gated
generator cannot independently reproduce a centered affine offset.

The output metric is

\[
M_o=D_S^\top D_S,
\]

and the arbitrary-residual Fisher metric is

\[
M_f=\mathbb E[s_Ss_S^\top].
\]

The Fisher moment is based on score gradients, not
\((z\odot s)(z\odot s)^\top\). The latter is appropriate for multiplicative
deletion masks, whereas this compiler must score a general reconstruction
residual.

Identity, output, and Fisher metrics are normalized by their energy against
\(C\), combined, and passed through the repository's generalized-Fisher
codec. Its retained dual matrices are

\[
E\in\mathbb R^{K\times R},\qquad
B\in\mathbb R^{K\times R}.
\]

With row-vector conventions,

\[
q^*=z_SE,\qquad \widehat z_S=q^*B^\top.
\]

The retained subspace is rotated, without changing \(EB^\top\), toward a
deterministic set of rank-revealing native pivots. This matters because a
linear reconstruction subspace is rotation-invariant but gated-generator
realizability is not: one arbitrary spectral rotation can turn simple
native-like coordinates into difficult mixtures.

The separable generalized metric is an initializer. It is not treated as the
exact sampled Fisher loss because activations and score gradients can be
correlated.

## Direct generator fit

Only standalone copies of the \(R\) new gate/up rows enter the optimizer.
Neither the frozen parent nor the candidate module is optimizer-owned. The
fit combines:

\[
L_{\text{latent}}
=
\operatorname{NMSE}(q_\theta,z_SE),
\]

\[
L_{\text{output}}
=
\operatorname{NMSE}
\left(
q_\theta B^\top D_S^\top,
z_S D_S^\top
\right),
\]

and the sampled first-order contraction

\[
L_{\text{Fisher}}
=
\operatorname{NMSE}
\left(
q_{\theta,t}(s_{S,t}B)^\top,
s_{S,t}z_{S,t}^\top
\right).
\]

This last term is a token-local contraction. It does not claim to reproduce a
whole-sequence loss change with cross-token Jacobian terms.

The deployed supermode down columns remain exactly \(D_SB\); they are not
refit after generator training. Freezing that map keeps the output and Fisher
objectives tied to the decoder that actually ships. Every singleton down
column also remains exact.

The initializer, deterministic fixed-stride full-fit snapshots, and terminal
iterate compete on the complete fit objective. The earliest best checkpoint
ships; a held-out guard never chooses or updates it.

The analysis encoder \(E\), decoder \(B\), calibration rows, source pool
indices, and native pool weights are omitted from the executable state. A
scientific experiment bundle may retain hashed plan metadata and index records
for audit, but the strict executor does not load or store them. At runtime it
issues only:

```text
dense gate [W-K+R, d]
dense up   [W-K+R, d]
dense down [d, W-K+R]
```

## Resource accounting

For a bias-free gated MLP with residual width \(d\), the exact reduction is

\[
\Delta P=\Delta\operatorname{MAC/token}=3d(K-R).
\]

A proposed first Gemma-sized rung pools \(K=512\) of the 2,048 MLP units and
replaces them with \(R=384\) supermodes. The other 1,536 units remain exact,
so the runtime width is 1,920. With \(d=640\), that removes:

- 245,760 learned parameters;
- 245,760 linear MACs per valid token;
- 128 activation elements and gated products per valid token; and
- 6.25% of that MLP's gate/up/down linear work.

These are shape-based parameter and arithmetic counts. They are not a
measured latency or kernel-speed result.

## Deterministic synthetic proof

The committed test fixture contains six native units arranged as two
functionally independent activation families. Each family has three
duplicate generators, while the individual down columns remain distinct.
The score ranking deliberately makes the two highest-ranked units come from
the same family.

One observed deterministic GELU-tanh run compacting the complete
\(6\rightarrow2\) pool produced:

| Candidate | MLP-output NRMSE | Block-output NRMSE |
|---|---:|---:|
| Dense two-supermode compiler | \(8.08\times10^{-7}\) | \(1.41\times10^{-6}\) |
| Equal-width diagonal-Fisher deletion + down refit | 0.370075 | 0.289417 |
| Equal-width oracle family deletion + down refit | \(1.66\times10^{-7}\) | \(1.27\times10^{-7}\) |

The dense generator's three-term calibration objective fell from
`0.00226428` to `1.4503e-12`. Its artifact contains 2-by-8 gate/up matrices
and an 8-by-2 down matrix, removes 96 parameters and 96 linear MACs/token,
strictly reloads, and leaves the parent fingerprint, tensors, gradients,
storage, and global RNG unchanged.

The diagonal-Fisher control deliberately ranks two duplicates from the same
family, so it is blind to the redundancy that the dense codec sees. The
structure-aware oracle keeps one unit from each family and is also nearly
exact. The fixture therefore proves physical \(K\rightarrow R\) execution,
direct nonlinear coordinate synthesis, and a concrete failure mode for scalar
importance ranking. It does **not** prove that dense merging beats the best
equal-width pruning rule, nor that a natural-language model contains equally
clean grouped redundancy.

## Rate-distortion evaluation

Compiler correctness and model quality are separate:

- The strict artifact must execute the intended reduced graph exactly.
- The reduced graph does not need 100% native-model fidelity.

The rate-distortion container keeps every raw candidate, including dominated
points, and can project Pareto frontiers over:

- learned parameters;
- runtime parameter bytes;
- logical MACs/token; or
- measured latency when a real measurement exists;

against downstream score, NLL, teacher KL, top-1 agreement, or operator
NRMSE.

Points are comparable only when they share the same evaluation identity,
split hash, task suite, and relevant resource scope. Parameter and byte
frontiers require the same parameter scope; MAC frontiers require the same
compute scope. Latency additionally requires a shared runtime, hardware, and
benchmark protocol. Dtype is always disclosed but may differ, which permits a
properly matched quantized candidate to share the same curve. These rules
prevent a one-layer parameter count from being placed on the same axis as a
whole-model quantization result, or MLP-only MACs from being presented as
full-model compute.

The representative comparison for every \(R\) must include equal-width
diagonal-Fisher deletion and a stronger structure- or output-aware pruning
control on the same fit rows, with each down-refit policy stated explicitly.
In the synthetic table both pruning controls receive a down refit while the
dense decoder remains frozen, so the controls are advantaged rather than
silently handicapped. A dense supermode point is promising only if it improves
the fresh, family-disjoint quality curve at the same resource rate. The guard
may evaluate the frozen candidate but may not rotate, refit, or select it.

## Gemma 3 270M development result

The first real-model rung used the source-free layer-4 executor, all 256
authenticated `structured-strong-v9` A-fit prompts, and the frozen
\(512\rightarrow384\) configuration above. It compared three physically
1,920-wide executors:

| Candidate | Block NRMSE | Block cosine | Top-1 agreement | Delta NLL/token | Teacher KL/token |
|---|---:|---:|---:|---:|---:|
| Direct dense supermodes | 0.049127 | 0.998794 | 0.982586 | +0.003414 | 0.001436 |
| Diagonal-Fisher deletion + down refit | 0.021758 | 0.999764 | 0.991776 | +0.000403 | 0.000373 |
| Native-pivot pruning + down refit | 0.015359 | 0.999882 | 0.993251 | -0.000202 | 0.000241 |

The guard was the already-consumed v9 A-guard partition. Its 256 prompts and
eight families are disjoint from A-fit, but reuse makes this a development
diagnostic rather than fresh confirmation. Calibration B, validation, and test
remained unopened.

The direct dense candidate failed the frozen burn filter. Its block NRMSE was
above the `0.015` margin and was 2.26 times the diagonal-deletion error and
3.20 times the native-pivot error. Its feed-forward branch NRMSE was
`0.041234`. The runner therefore wrote JSON only and did not preserve a tensor
checkpoint.

This is a failure of the tested direct synthesis, not of the physical
compaction machinery. The generalized codec retained `99.9109%` of its
weighted spectrum, but one gated generator per retained coordinate did not
realize that linear subspace well: on A-fit its selected-pool output NRMSE was
`0.421880`, latent NRMSE was `0.830279`, and sampled Fisher-contraction NRMSE
was `0.475147` after all 256 fixed optimizer steps. A linear combination of
many GELU-gated native units is not generally another single GELU-gated unit.

The unexpectedly useful result is the native-pivot control. The dense plan's
rank-revealing source coordinates plus an actual-feature down refit passed all
ordinary behavior, branch, direct-fidelity, and execution gates. It missed the
stricter `0.015` development margin by `0.000359` (about 2.4%), while improving
block NRMSE by 29.4% over diagonal-Fisher deletion at identical width. This is
promising structure-aware pruning evidence, but it is not a fresh held-out
compression result and does not authorize calibration B.

All three reduced executors have the same exact shape savings:

- 5,573,632 to 5,327,872 layer parameters, a 4.4093% reduction;
- 3,932,160 to 3,686,400 MLP linear MACs per valid token, a 6.25% reduction;
- 4.2176% fewer analytic complete-layer MACs on the guard sequence mix; and
- no measured latency or kernel-speed claim.

The most direct next development control is to down-refit the dense
candidate's terminal projection from its actual synthesized runtime features,
just as both pruning controls do. That tests dense merge-and-compensate
separately from the failed requirement that the frozen analytic decoder
\(D_SB\) remain optimal after nonlinear generator approximation.
