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

This remains a reference policy, not an authorized locked-runtime path. A
later audit found that its exact-arithmetic bound inflation did not cover the
floating-point accumulation path closely enough, and the measured compute
accounting omitted part of the routing work. The one-shot runtime therefore
rejects routed execution until those two issues are repaired and retested.

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

## Natural one-shot shadow lock-down

The next rung put the all-on edge inside the factorized Gemma source model.
This was a development-only diagnostic on one authenticated
`structured-strong-v9` Calibration-A prompt, not a held-out result. The
corrected runtime:

- removes only the modeled 64-mode contribution from the L3 MLP output and
  leaves the 576-mode complement intact;
- predicts every valid L4 target causally reachable from an eligible source,
  including targets beyond the source-knot interval;
- decodes target modes with a checked right inverse of `R4[:64]`, never `P4`;
- authenticates the raw-model lineage and the live factorized model
  separately; and
- keeps native boundaries and logits authoritative while candidate outputs
  remain metrics-only.

The prompt had 44 valid rows, 43 next-token labels, 33 eligible source rows,
and 36 affected target rows (`81.82%` target coverage). The whole-prompt
development measurements were:

| Path | Boundary relative error | Boundary cosine | ΔNLL/token | Source→candidate KL/token | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| Frozen all-on graph | `4.8208` | `0.5404` | `+3.0853` | `3.3077` | `0.3953` |
| True 64-mode delta plus authenticated dual | `0.9741` full-width | `0.2261` full-width | `+2.2416` | `2.7701` | `0.4651` |
| Exact full-width X4 on the clamped reference carrier | exact X4 injection | exact X4 injection | `+2.0121` | `2.5847` | `0.4651` |

These three rows isolate two independent blockers.

First, the current target subspace does not contain the natural displacement.
Even an oracle given the true 64 target modes cannot reconstruct the
full-width delta. The target-rank ladder makes the capacity boundary clear:

| Target rank | Full-width delta relative error | Cosine |
|---:|---:|---:|
| 32 | `0.9905` | `0.1375` |
| 64 | `0.9741` | `0.2261` |
| 128 | `0.9435` | `0.3315` |
| 256 | `0.8648` | `0.5022` |
| 384 | `0.7582` | `0.6520` |
| 512 | `0.5765` | `0.8171` |
| 640 | approximately `0` | `1.0000` |

Second, X4 alone is not a complete replacement boundary. Removing the L3
contribution also changes the residual stream that carries the L4 MLP output
forward. Supplying the exact native X4 on that clamped residual carrier still
does not reproduce native logits. A better target must therefore restore or
compile the carrier as well as the normalized-input boundary.

The frozen qualification protocol now fails closed on four independent
requirements: target-modal fidelity, full-width projection capacity, carrier
completeness, and source-authoritative NLL/KL/top-1 fidelity. Behavioral gates
use only causally affected supervised tokens so unchanged prefixes cannot
dilute a failure. Passing those gates can qualify only a partial shadow;
deployment, routing, standalone replacement, parameter reduction, and latency
claims remain unauthorized while a source-model reference pass is required.

The execution evidence is also fail-closed. The only supported Calibration-B
entry point statically authenticates the protocol, runtime, live adapter, and
internally loaded local Gemma tokenizer before consuming the manifest. The
tokenizer contract includes the backend program, complete token-to-ID map,
added/special tokens, and implementation versions. After an atomic claim, the
transaction loads each frozen prompt identity exactly once and owns the exact
`3 + 1 + 1` order: native/reference/candidate source passes, followed by the
rank-64 projection oracle and exact-X4 carrier oracle. Per-example tensor
observations stay internal; only a scalar report and terminal receipt escape.
A chained receipt commits to:

- the canonical prompt identity and the exact tensor-valued model inputs;
- the raw model, live factorized model, adapter execution, graph, basis, and
  plan identities;
- the causal execution grid and complete next-token boundary set;
- the shadow result, both oracle results, and both injected X4 tensors; and
- all 13 tensor payloads and shapes that the evaluator actually scores.

The frozen evaluator accepts only the full 96-example, eight-family
Calibration-B manifest. It rejects prompt or model-input replay, family
relabeling, boundary subsets, gapped sequences, routed execution, and logits
whose width is not Gemma's frozen `262,144` vocabulary. Full-vocabulary
behavioral metrics use scalar-only streaming accumulators, so the evaluator
does not retain the panel's logits in memory.

One-shot state is kept outside the pure runtime and evaluator. The only
supported ledger API is the fused transaction above. It uses an atomic private
`O_EXCL` claim keyed only by the manifest at one fixed per-user host state
path, independent of a checkout or installation, before invoking any prompt
loader. Changing the candidate, protocol, or worktree cannot mint another use
of the same Calibration-B manifest for that user on that host. Success
requires the privately streamed evaluator report and writes one hash-bound
receipt; failure writes one sanitized error receipt, and a crash leaves the
role consumed. No public report callback, held-out per-example issuer, or
held-out evaluator exists. These hashes provide reproducible integrity and
audit provenance, not cryptographic attestation against arbitrary malicious
code in the same Python process, and the local ledger is not a cross-machine
authority. This transaction has not been run, so Calibration B remains
unopened.

A generic bounded radial correction is available for a future candidate. It
adds a low-rank finite-displacement residual after the graph map, is exactly
zero at the reference, preserves the graph JVP there, and is bound to all-on
execution. It is deliberately not attached to this 64-mode candidate: a
post-map correction cannot recover directions absent from the target subspace
or replace the missing residual carrier.

## Claim boundary

This establishes an executable hybrid representation with exact all-on
equivalence, strict provenance, conservative omission certificates, matched
controls, real coefficient reduction for the measured edge, and a conditional
rate curve. It also now establishes a strict source-authoritative one-shot
shadow harness that rejects the current natural-prompt candidate for two
separately measured structural reasons.

It does not establish natural-prompt transfer, held-out NLL or task accuracy,
whole-block or whole-model replacement, end-to-end compression, GPU speed, or
wall-clock latency. The next architecture rung is a wider, carrier-aware edge.
Only after its projection and carrier oracles pass should finite-displacement
correction, family-disjoint shadow qualification, or conditional routing be
reopened.
