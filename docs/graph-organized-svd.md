# Graph-organized SVD executor

This rung separates two jobs that the earlier graph-Fourier experiment had
asked one basis to do:

- global SVD compresses the measured linear operator; and
- the fit-only signed graph organizes retained SVD generators into packs that
  an executor can route conditionally.

The distinction matters. On the frozen Gemma L3→L4 response, the signed graph
basis was more meaningful than phase-blind, native-prefix, permuted, and
random controls, but it was a much worse numerical compression basis than
global SVD. The hybrid keeps the globally optimal rank-45 weights and uses the
graph only as a map over those weights.

## Factorization

For weighted source response \(H_{o,\ell}\) at source origin \(o\) and causal
lag \(\ell\), the compiler first fits the ordinary global-SVD plan

\[
H_{o,\ell}\approx U C_{o,\ell}Q^\mathsf{T}.
\]

The target rank is the complete 64-mode target width, so \(Q\) can be folded
into the causal core once:

\[
\widetilde C_{o,\ell}=C_{o,\ell}Q^\mathsf{T},\qquad
H_{o,\ell}\approx U\widetilde C_{o,\ell}.
\]

Let \(G\) be the signed graph-Laplacian eigenbasis fitted only from origins
8, 24, and 40. Each retained SVD generator \(U_j\) receives a graph-frequency
signature

\[
e_{f,j}=(G_f^\mathsf{T}U_j)^2.
\]

The compiler sums this energy inside tied-safe graph-frequency bands and
assigns each generator to its largest-mass band. For the frozen rank-45 plan,
the bands `0:8`, `8:16`, `16:32`, and `32:64` produce packs of
`8 / 8 / 8 / 21` generators.

Packing is only a permutation. The source basis and corresponding causal-core
rows are reordered together. With every pack enabled, the hybrid therefore
reconstructs the same rank-45 SVD operator; no coefficient is refitted,
deleted, averaged, or quantized.

## Certified reference router

For pack \(p\), fit knot \(k\), and lag \(\ell\), the plan stores an upward
inflated operator-norm certificate

\[
\beta_{k,\ell,p}\ge
\left\|\widetilde C_{k,\ell,p}\right\|_2.
\]

Linear interpolation of these endpoint bounds remains an upper bound because
the matrix norm is convex. For standardized source row \(x\), latent pack
\(z_p=x^\mathsf{T}U_p\), and source origin \(o\), the router score is

\[
s_p(x,o)=
\|z_p\|_2
\sqrt{\sum_\ell\beta_{o,\ell,p}^2}.
\]

The displayed product is implemented as multiplication between the two
factors. Retaining packs in descending score order gives a deterministic
reference policy. More importantly, the sum of omitted scores certifies the
norm of that source row's omitted full lag-by-target response. The percentage
passed to the router is a fraction of conservative bound mass, not a promised
percentage of true output energy.

The prepared float64 executor computes the source projection once, reuses it
for routing and transport, and caches interpolated pack cores by logical
origin within a call. Explicit external masks remain available for learned or
hardware-specific routers.

## Frozen Gemma development curve

The current experiment uses the rank-45/full-target SVD plan and evaluates
four-pack routing on the prompt-blind synthetic C2 directions. The upstream
Fisher basis is still prompt-derived. C2 contains substantial exact-zero
padding, so zero-norm rows are removed before rate scoring; the curve uses
1,132 nonzero fit rows and 1,132 nonzero selection rows.

The deployment-form coefficient accounting is:

| Representation | Float coefficients | Reduction vs dense fit knots |
|---|---:|---:|
| Three dense measured fit knots | `393,216` | — |
| Graph-organized rank-45 SVD plus certificates | `279,744` | `28.86%` |
| Full research artifact including graph diagnostics | `287,156` | `26.97%` |

For one complete 32-lag response, dense application uses 131,072 linear MACs.
With interpolated cores cached, the all-on rank-45 executor uses 95,040
linear MACs, or `72.51%` of dense. Router work, interpolation on a cache miss,
memory traffic, and surrounding Gemma computation are not included in that
ratio.

On the held-out response origin 32:

| Organization / bound fraction | Mean active rank | Cached-core MAC fraction | Error vs dense measured response |
|---|---:|---:|---:|
| Global SVD, all packs | `45.00` | `72.51%` | `3.179%` |
| Signed-GFA packs, `95%` | `43.78` | `70.61%` | `3.206%` |
| Contiguous SVD packs, `95%` | `43.14` | `69.60%` | `4.148%` |
| Signed-GFA packs, `90%` | `30.33` | `49.58%` | `12.936%` |
| Contiguous SVD packs, `90%` | `30.61` | `50.02%` | `10.724%` |

The high-fidelity signed point is real but modest: it removes about 1.22 of
45 ranks on average and reduces the cached-core MAC ratio by 1.90 percentage
points versus all-on SVD while adding little measured-response error. At the
more aggressive 90% threshold, conditional compute becomes substantial, but
fidelity degrades and contiguous SVD packing is better. Eight size-matched
random controls are retained in the machine-readable report; the signed
organization is useful, but it is not uniformly optimal.

## Claim boundary

This establishes an executable hybrid representation with exact all-on
equivalence, strict provenance, conservative omission certificates, matched
controls, real coefficient reduction for the measured edge, and a conditional
rate curve.

It does not establish natural-prompt transfer, NLL or task accuracy,
whole-block or whole-model replacement, end-to-end compression, GPU speed, or
wall-clock latency. The next fidelity rung is a family-disjoint shadow run of
the compiled edge inside the model; the next routing rung is a finer pack or
component-level policy tested at equal active-rank budgets.
