# Graph-wavelet mapping

This rung asks whether the fit-only signed Fisher graph can provide a source
basis that is both useful and local. It is a different construction from the
two spectral tools already used in the repository:

- the FFT resolves variation along the causal-lag axis;
- graph Fourier analysis (GFA) resolves a graph signal into global
  Laplacian eigenvectors; and
- graph wavelets resolve a graph signal jointly by graph-frequency scale and
  source-mode center.

The last point is the new part. A graph-Fourier vector can touch every source
mode. A graph-wavelet atom is centered on one source mode and has a scale that
controls how far its influence spreads through the fitted graph. The resulting
dictionary is redundant rather than orthonormal, so it must be compiled and
accounted for carefully.

This was a tensor-only development experiment. It loaded one already
authenticated five-origin L3→L4 response artifact and performed zero model
loads, token loads, prompt loads, forward passes, or new measurements.

## Parseval wavelet frame

Let the fit-only symmetric positive-semidefinite graph Laplacian be

\[
L=V\Lambda V^\mathsf{T}.
\]

The current frame freezes four diffusion scales,

\[
s=(0.5,1,2,4),
\]

and normalizes the Laplacian eigenvalues by their maximum. If
\(h_s(\lambda)=\exp(-s\lambda)\), the unnormalized windows are

\[
\begin{aligned}
r_0(\lambda)&=h_4(\lambda),\\
r_1(\lambda)&=h_2(\lambda)-h_4(\lambda),\\
r_2(\lambda)&=h_1(\lambda)-h_2(\lambda),\\
r_3(\lambda)&=h_{0.5}(\lambda)-h_1(\lambda),\\
r_4(\lambda)&=1-h_{0.5}(\lambda).
\end{aligned}
\]

Each graph frequency is normalized independently:

\[
g_f(\lambda)=
\frac{r_f(\lambda)}
{\sqrt{\sum_q r_q(\lambda)^2}}.
\]

Therefore

\[
\sum_f g_f(\lambda)^2=1
\quad\text{and}\quad
\sum_f g_f(L)^2=I.
\]

For filter \(f\) and source-mode center \(n\), the centered atom is

\[
\psi_{f,n}=g_f(L)e_n.
\]

Analysis and synthesis are

\[
C_f=g_f(L)X,
\qquad
\widehat X=\sum_f g_f(L)C_f.
\]

The spectral partition makes the complete analysis/synthesis path a Parseval
tight frame. With 64 source modes and five filters (one scaling filter plus
four band-pass filters), however, it contains 320 center-scale atoms. Exact
reconstruction does not itself imply compression.

This construction is also distinct from applying an FFT to a mode index. Its
notion of neighborhood comes from the fitted signed graph, not an arbitrary
linear ordering of the modes.

## Fit-only GOMP-to-QR compilation

The authenticated response tensor has shape

```text
[source mode, origin, causal lag, target mode] = [64, 5, 32, 64]
```

and is weighted by the frozen source-mode standard deviations. Origins
8, 24, and 40 are the only fit origins. Their lag, target, and origin axes are
flattened into the grouped fit signal \(Y_\mathrm{fit}\); the source-mode axis
remains the graph-signal axis.

Every raw atom is unit-normalized before selection. Starting with
\(R_0=Y_\mathrm{fit}\), simultaneous group orthogonal matching pursuit (GOMP)
selects

\[
j_t=\arg\max_j
\left\|d_j^\mathsf{T}R_{t-1}\right\|_F^2.
\]

The Frobenius norm groups all fit origins, causal lags, and target modes. After
each selection, the selected raw atoms are orthogonalized into \(Q_t\), and
the residual is recomputed as

\[
R_t=(I-Q_tQ_t^\mathsf{T})Y_\mathrm{fit}.
\]

For a requested rank \(K\), the compiled source projector is

\[
\Pi_K=Q_KQ_K^\mathsf{T}.
\]

The compiler freezes the center-scale order, selected atoms, QR basis, scale
schedule, graph identity, and fit-source identity before evaluating the
selection origins. It fails closed if a selected atom does not increase the
numerical rank, if QR novelty falls below the fixed `1e-10` dependency floor,
or if the rank and orthogonality checks fail. Raw selected-dictionary
condition numbers are recorded for diagnosis, but the frozen protocol does
not impose a separate maximum-condition gate.

## Leakage boundary

The development split is fixed:

- graph fit, wavelet construction, GOMP selection, QR, and all fit-adaptive
  controls use only origins `8 / 24 / 40`;
- origins `16 / 32` are fit-disjoint development selection origins and are
  read only after each candidate is frozen; and
- origin `20` is not part of this experiment, but it has already been opened
  by an earlier rung and is historical development evidence rather than a
  fresh confirmation boundary.

The signed and magnitude graphs themselves are fitted only after indexing the
three fit origins. The diffusion scales are predeclared. Signed-GFA
fit-energy ordering, native fit-energy ordering, fit SVD, and random-control
ordering likewise use only the fit tensor.

Changing the filters, scale schedule, atom-selection rule, numerical gates, or
rank after reading origins 16/32 creates a new development candidate. It must
be frozen again before any assessment-origin evaluation.

## Matched controls

The prototype compared equal source-subspace ranks against:

- the signed-GFA low-frequency prefix;
- the stronger signed-GFA fit-energy ordering;
- a phase-blind magnitude-graph wavelet dictionary;
- a conjugated node permutation of the signed graph, with the response modes
  left in their original order;
- native source-mode prefixes and fit-energy ordering;
- eight seeded random-Haar bases, including fit-energy ordering; and
- fit-only SVD, the optimal rank-\(K\) projector for fit reconstruction.

The graph permutation preserves the signed graph's spectrum and wavelet
localization statistics while destroying its alignment to the response
modes. It is therefore a particularly important control for the claim that
the signed topology matters.

## Independent projection diagnostic

The following table came from the pre-run projection prototype. It is useful
diagnostic evidence, but it is not a rate row authenticated by the compact
executor-form publication below.

Each cell below is `fit relative error / selection relative error`, measured
after orthogonal projection on the source-mode axis. “Selection” aggregates
origins 16 and 32. These are structural response-projection metrics, not
executor-output or model-behavior metrics.

| Rank \(K\) | Signed wavelet GOMP | Signed-GFA prefix | Fit SVD | Native prefix |
|---:|---:|---:|---:|---:|
| 8 | `0.578794 / 0.570016` | `0.530823 / 0.528047` | `0.347939 / 0.340745` | `0.911330 / 0.911531` |
| 16 | `0.463970 / 0.454631` | `0.433773 / 0.430253` | `0.217904 / 0.208980` | `0.832269 / 0.833040` |
| 32 | `0.313215 / 0.307198` | `0.287784 / 0.285240` | `0.092417 / 0.085150` | `0.570458 / 0.568636` |
| 45 | `0.206643 / 0.201932` | `0.189291 / 0.187347` | `0.042741 / 0.038458` | `0.387275 / 0.383706` |
| 52 | `0.151024 / 0.147472` | `0.142841 / 0.141292` | `0.026871 / 0.024178` | `0.271784 / 0.268910` |
| 64 | `0.000000 / 0.000000` | `0.000000 / 0.000000` | `0.000000 / 0.000000` | `0.000000 / 0.000000` |

The signed-wavelet selection cosines at the same ranks are
`0.821633 / 0.890680 / 0.951646 / 0.979400 / 0.989066 / 1.000000`.
The two selection origins behave similarly; at rank 32 their separate
relative errors are `0.311575` and `0.302628`.

The signed topology passes the control test. At rank 32:

| Fit-only source organization | Selection relative error |
|---|---:|
| Signed wavelet GOMP | `0.307198` |
| Conjugated-permutation signed wavelet GOMP | `0.428431` |
| Magnitude wavelet GOMP | `0.438227` |
| Native fit-energy modes | `0.434191` |
| Random-Haar fit-energy median, eight seeds | `0.491563` |

The random-control range at rank 32 is `0.480711–0.518287`. Thus the signed
result is not explained by an arbitrary localized dictionary or by
fit-adaptive mode selection.

The fidelity comparison nevertheless rejects the stable wavelet candidate.
The signed-GFA fit-energy control is stronger than its prefix and reaches
selection errors `0.261551 / 0.169614 / 0.122274` at ranks 32, 45, and 52.
Stable wavelet GOMP loses to both signed-GFA orderings at every non-full rank,
and fit SVD remains substantially better.

## Locality and numerical stability

For a vector \(v\), effective node support is

\[
\operatorname{support}_{\mathrm{eff}}(v)=
\frac{\|v\|_2^4}{\|v\|_4^4}.
\]

It ranges from 1 for a one-node vector to 64 for a perfectly uniform vector.
The table reports means over the selected raw atoms and the corresponding QR
columns. Scale counts are ordered as
`scaling / bandpass_00 / bandpass_01 / bandpass_02 / bandpass_03`.

| Rank | Selected scale counts | Raw-atom effective support | QR effective support | Raw selected-matrix condition |
|---:|---:|---:|---:|---:|
| 8 | `3 / 5 / 0 / 0 / 0` | `4.34` | `4.35` | `4.44` |
| 16 | `3 / 11 / 2 / 0 / 0` | `2.68` | `2.72` | `7.16` |
| 32 | `3 / 26 / 3 / 0 / 0` | `1.87` | `1.94` | `8.34` |
| 45 | `3 / 32 / 9 / 1 / 0` | `1.63` | `1.71` | `9.60` |
| 52 | `3 / 32 / 14 / 3 / 0` | `1.54` | `1.64` | `10.0` |
| 64 | `3 / 34 / 16 / 3 / 8` | `1.45` | `1.60` | `25.1` |

For comparison, the signed-GFA prefix has mean effective support `23.71` at
rank 8 and `15.77` at rank 52. The wavelet basis is therefore dramatically
more local. The stable GOMP path also retains full numerical rank at every
cutoff, has minimum incremental QR novelty `0.799`, and has maximum measured
QR orthogonality error `1.4e-15`.

The scale counts reveal a limitation as well as a benefit: the compiler uses
mostly highly local `bandpass_00` atoms rather than a balanced multiscale
mixture. The result behaves more like graph-informed local mode selection
than a rich hierarchy of scales.

### Rejected OLS sensitivity

This was also an independent pre-run diagnostic rather than a row in the
authenticated executor-form publication. A residual-normalized
orthogonal-least-squares sensitivity appeared to beat signed-GFA fit-energy
ordering from rank 32 onward. Its selection errors were `0.245038 / 0.145038
/ 0.099833` at ranks 32, 45, and 52.

That apparent gain is not accepted. The raw selected-atom condition numbers
rose to `2.21e5 / 6.46e5 / 7.43e5`, while minimum QR novelty fell below
`3e-4`. Orthogonalization turned local raw atoms into much denser difference
directions: at rank 32, mean effective support changed from `5.32` for the raw
atoms to `10.55` for \(Q\). The sensitivity was extracting useful directions
through near-cancellation, not demonstrating a stable sparse wavelet
executor. Storing the dense \(Q\) could make the projection numerically safe,
but would surrender much of the locality and packing motivation.

The accepted development conclusion therefore uses strict, stable GOMP only.

## Authenticated executor-form result

The landed runner lowers every frozen source basis through the existing
fit-knot conditional spectral executor with full target rank 64. Unlike the
projection table above, these rows include interpolation from fit origins
`8 / 24 / 40` to selection origins `16 / 32`.

| Rank | Selection error | Selection cosine | Analytic compiled-plan float64 scalars | Gate reduction versus 401,408-scalar full-rank plan |
|---:|---:|---:|---:|---:|
| 8 | `0.570306` | `0.821433` | `53,760` | `86.61%` |
| 16 | `0.455178` | `0.890401` | `103,424` | `74.23%` |
| 32 | `0.308557` | `0.951210` | `202,752` | `49.49%` |
| 45 | `0.204312` | `0.978912` | `283,456` | `29.38%` |
| 52 | `0.150901` | `0.988556` | `326,912` | `18.56%` |
| 64 | `0.033135` | `0.999459` | `401,408` | `0.00%` |

The gate denominator is the matched full-rank plan, which includes the
source and target bases as well as the `393,216`-scalar dense fit-knot core.
Relative to that core alone, the rank-45, rank-52, and rank-64 plan payloads
are `27.91%`, `16.86%`, and `-2.08%`; those are descriptive comparisons, not
the frozen gate. The signed-wavelet compiler also carries `4,480` float64
scalars of eigensystem and filter metadata, so its standalone totals are
`287,936`, `331,392`, and `405,888` at those three ranks. The gate uses the
plan-only full-rank baseline, while candidate ordering uses the standalone
compiler-plus-plan total.

This exposes the central result cleanly:

- ranks through 45 clear the frozen 20% analytic plan-payload reduction
  threshold, but none clears the `error <= 0.20` and `cosine >= 0.98`
  fidelity gate;
- ranks 52 and 64 clear the fidelity gate, but neither clears the
  plan-payload threshold; and
- no tested rank clears both gates.

Rank 52 is the closest fidelity-passing point that still has a nominal
plan-payload reduction, but it is not the frozen localized structural
nominee: only full rank 64 passes that combined locality gate. At rank 52 the
stronger fit-energy ordered GFA control reaches `0.126492` error, and fit-only
SVD reaches `0.040911`. The eight fit-energy ordered random-Haar controls
have median error `0.254063` at rank 52, so the signed wavelet still wins all
eight topology-null trials there. At full rank, all complete bases tie.

The runner deliberately leaves the compute gate closed. It records analytic
state size, but this experiment did not execute a matched runtime sequence or
measure latency. Dense-\(Q\), direct wavelet, and sparse/local kernels have
different execution costs and cannot inherit a speed claim from source rank.

## Rate and resource accounting

Rank \(K\) retains `K / 64` source directions: `12.5% / 25% / 50% / 70.31% /
81.25% / 100%` at the tested cutoffs. This is a source-subspace rate, not a
parameter-compression ratio.

A complete accounting must distinguish at least three representations:

1. the raw tight frame has 320 center-scale coefficient groups and is larger
   than the native 64-mode representation;
2. the compiled QR representation stores or reconstructs a \(64\times K\)
   basis plus the downstream causal/target core; and
3. an index-driven localized executor must also count its graph or
   eigensystem, scale metadata, selected center-scale indices,
   orthogonalization transform, and graph-filter work.

If the runtime stores dense \(Q\), its source projection has the same dense
shape as any other rank-\(K\) basis. If it executes selected local atoms
directly, the measured locality could become useful, but the actual sparse
kernel, memory traffic, and reconstruction transform must be implemented and
counted. The present support numbers do not establish a FLOP or latency
reduction.

## Artifact receipt

The ignored write-once publication is
`.local-runs/google--gemma-3-270m/modal-generator-l3-l4-graph-wavelet-dev-v2.pt`.
The earlier v1 receipt remains as historical local evidence. V2 leaves every
compact numerical tensor, conditional-plan hash, metric, and gate outcome
unchanged; it replaces one overbroad transform-accounting label with
method-specific semantics and adds stricter replay consistency checks.

| Receipt | SHA-256 |
|---|---|
| Logical candidate | `7659dbbc2547f0222f4ee8eb28b587b4d05a5dce0fa403f1801975083b8914f2` |
| Tensor file | `cba8148ae17aceac95e975a4593335b96cc011977d65285d43d091454ade5d3d` |
| Report payload | `3d81cec7d34e97872266ef0c019257c55c5ff231a776f6cfb9426d8d70fb858c` |
| Source tensor | `a80b9ce1a5e433724e74cb7c29143d18442805a7b05fcb419ede6ad1e23686b3` |
| Source report payload | `a3330bcd75c637811be62dd33a53b6a2329edf66d586b73f157176400462e7b5` |
| Response binding | `71a8ec2b5e108256a96c81c1cf5855280054828816e20d28233d5ce8796c28cb` |
| Fit-only graph | `855a047ef20ca3e11a105d7d62752575381ce6eeccd7d12bf98b72dc43067730` |
| Signed wavelet frame | `edac3ba54408c16ab27b205108a706ac8a7cec2993a03075654303b761cdb35e` |
| Signed fit-only GOMP path | `cfa9968b22922abbcf534348192e5626dc22e261af8f0d2839de5484beb75d02` |
| Rank-52 conditional plan | `2eb192709a5da3ae20df4ece547ef28a29f8887f21094e7876a9c3a8d7891d57` |

The compact tensor file is `496,661` bytes. It stores signed and magnitude
eigenpairs plus spectral filter multipliers, but no raw response tensor,
dense per-scale filter matrices, model state, prompt text, token IDs, or
compiled plan tensors. Its resource receipt records one authenticated
`85,246,234`-byte source artifact and zero model loads, tokenizer loads,
prompt reads, token reads, model forwards, or new response measurements.

## Claim boundary

This experiment provides fit-disjoint, open-development evidence that the
fit-only signed Fisher topology contains real structural information and that
graph-wavelet compilation can produce a dramatically more localized source
basis than GFA. It also shows on this development split that the numerically
stable wavelet path does not beat signed GFA or fit SVD on structural
projection fidelity.

It does not establish a complete transfer executor, natural-displacement
fidelity, finite-displacement correction, model-output equivalence, NLL or
downstream-task retention, parameter compression, whole-layer or whole-model
replacement, conditional routing, GPU acceleration, kernel speed, wall-clock
latency, or deployment readiness. The next accepted rung would need a frozen
sparse/local execution representation, matched compute accounting, and a
separately protected two-origin assessment without refitting before any of
those claims can be reopened.
