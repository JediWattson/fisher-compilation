# Graph-wavelet grouped basis comparison

This rung asks a narrower question than the pair-supermode experiment:

> If the wavelet map tells us which parent modes belong together, does a
> multiway rotation inside those neighborhoods recover substantially more
> fidelity than pairwise merging or pruning?

The answer on the opened Gemma L3→L4 fixed-reference panel is **yes for
block-local SVD, but no for the tested local graph-Fourier basis**.

This remains structural-response development evidence. It is not an NLL,
natural-prompt, whole-layer, whole-model, latency, or accepted compression
result.

## Frozen measurement boundary

The runner reconstructs the authenticated rank-64 signed graph-wavelet GOMP
basis \(Q_{64}\). Origins 8, 24, and 40 are the only values used to:

1. reconstruct the signed and magnitude response graphs;
2. reconstruct and authenticate \(Q_{64}\);
3. create every graph partition;
4. fit every grouped basis and leave-one-fit-origin-out replay;
5. refit every full-target conditional executor plan; and
6. compute every fit metric.

Only after all 23 plans are frozen does the runner read origins 16 and 32.
Changing only those two origins leaves every partition, basis, plan hash, and
fit row unchanged.

## Topology partition

The parent-coordinate interaction matrix uses the same locality signal as the
pair-supermode rung:

\[
T = |Q|^\mathsf{T}
    |\operatorname{offdiag}(L)|
    |Q|.
\]

The partitioner performs deterministic balanced dyadic average linkage. The
top-8 neighborhoods prioritize the first pairings, and every accepted merge
must have positive topology coupling. The tested schedules are:

| Groups | Maximum block width | Cross-block rotations |
|---:|---:|---|
| 4 | 16 | forbidden |
| 8 | 8 | forbidden |
| 16 | 4 | forbidden |

A one-group partition is not allowed to call itself local. The unrestricted
SVD endpoint remains a separately labeled control.

## Two wavelet-derived bases

For a topology block \(C\), let

\[
A_C = Q_C^\mathsf{T}Y_{\mathrm{fit}},
\]

where \(Y_{\mathrm{fit}}\) is the source-σ-weighted fit response.

### Local SVD

The local-SVD arm diagonalizes the response Gram inside each block:

\[
A_CA_C^\mathsf{T} = V_C\Lambda_CV_C^\mathsf{T},
\qquad
B_C = Q_CV_C.
\]

All block components are globally ordered by their fit eigenvalue, and the
first 45 are retained. This is the best rank-45 response projector subject to
the fixed block partition. It can mix many parent modes inside a block but
cannot create a loading across blocks.

### Cluster GFA

The cluster-GFA arm instead diagonalizes the block-restricted projected graph
Laplacian, maps those local graph-Fourier vectors through \(Q_C\), and orders
them by fit-response energy.

This is a useful falsification arm: it tests whether graph frequency itself is
the right local coordinate system, rather than merely using the graph to
define where response-derived rotations are allowed.

## Rank-45 result

All rows use the same full target rank, three knots, 32 lags, interpolation
rule, and conditional executor. Lower is better for relative error.

| Method | Selection error | Cosine |
|---|---:|---:|
| Global fit SVD ceiling | **0.05055** | 0.99873 |
| Signed local SVD, 4 groups × 16 | **0.13256** | 0.99118 |
| Magnitude local SVD, 4 groups × 16 | 0.13565 | 0.99076 |
| Signed local SVD, 8 groups × 8 | **0.16044** | 0.98705 |
| Magnitude local SVD, 8 groups × 8 | 0.16712 | 0.98594 |
| Signed fit-energy GFA | 0.17258 | 0.98500 |
| Signed cluster GFA, 4 groups × 16 | 0.17599 | 0.98440 |
| Signed local SVD, 16 groups × 4 | 0.17904 | 0.98385 |
| Pair supermode | 0.18422 | 0.98289 |
| GOMP / diagonal pruning | 0.20431 | 0.97891 |
| Signed 8-group one-hot control | 0.21500 | 0.97662 |

The full locality curve is:

| Topology | Max block | Local SVD | Cluster GFA |
|---|---:|---:|---:|
| Signed | 16 | **0.13256** | 0.17599 |
| Signed | 8 | **0.16044** | 0.19297 |
| Signed | 4 | **0.17904** | 0.19256 |
| Magnitude | 16 | **0.13565** | 0.27286 |
| Magnitude | 8 | **0.16712** | 0.26485 |
| Magnitude | 4 | **0.18821** | 0.23067 |

That monotone local-SVD trend is the important new result: wider local blocks
recover more fidelity, while stronger locality gives up some of that recovery.
The wavelet graph is functioning as a constraint on a dense local response
factorization.

## Controls

The preregistered 8-group signed arm is the most controlled structural
nominee:

- selection error `0.16044`, cosine `0.98705`;
- rank allocation `[7, 6, 6, 7, 5, 6, 3, 5]`;
- minimum leave-one-fit-origin-out rank-45 projector overlap `0.97590`;
- one-hot control error `0.21500`;
- four permuted-topology controls span `0.16890–0.17304`;
- the native signed partition beats all four permuted controls;
- 28 of its 45 retained columns have at least three parent loadings with
  squared loading at least `0.10`.

In squared-error terms, that arm recovers:

- `13.57%` versus equal-rank signed fit-energy GFA;
- `24.15%` versus pair supermodes;
- `38.34%` versus GOMP pruning; and
- `44.31%` versus its matched one-hot group-allocation control.

The best opened row, signed 4-group local SVD, recovers `40.996%` of the
signed-GFA squared error and `48.218%` of the pair-supermode squared error.
It was selected after looking across the development schedule, so it is not a
confirmation result.

Signed topology is better than magnitude topology at the matched settings in
this implementation, but the report deliberately does not make a general
signed-topology-superiority claim. A fresh size-matched random-partition panel
would still be required for that.

## Storage and compute

Every rank-45 row folds its local rotations directly into the source basis.
Partitions, \(Q_{64}\), and block mixers are not runtime payloads.

| Quantity | Rank 45 | Rank 64 | Reduction |
|---|---:|---:|---:|
| Compiled plan coefficients | 283,456 | 401,408 | 29.38% |
| Prepared runtime bytes | 2,268,184 | 3,211,800 | 29.38% |

The new methods therefore have the same stored plan and analytic executor work
as every other rank-45 basis. This experiment did not benchmark a fused kernel
or end-to-end model execution, so the compute and speed gates remain closed.

## Interpretation

The result separates three ideas that were previously tangled together:

1. **Wavelet topology is useful as a map.** It gives bounded neighborhoods in
   which modes may interact.
2. **Pairwise merging is too restrictive.** Joint block rotations recover much
   more fidelity than disjoint `2→1` actions.
3. **Graph frequency is not automatically the best payload basis.** The tested
   cluster GFA loses to response-derived local SVD at every matched setting.

The current graph is therefore best viewed as a sparse *permission structure*
for dense local synthesis, not as a complete executor basis by itself.

## Run

```bash
fisher-graph-gemma-l3-l4-graph-wavelet-grouped-dev describe
fisher-graph-gemma-l3-l4-graph-wavelet-grouped-dev analyze
```

The ignored local artifact is written to:

```text
.local-runs/google--gemma-3-270m/
  modal-generator-l3-l4-graph-wavelet-grouped-comparison-dev-v1.pt
  modal-generator-l3-l4-graph-wavelet-grouped-comparison-dev-v1.json
```

Authenticated local receipt:

| Object | SHA-256 |
|---|---|
| Logical candidate | `2169782723cf11307fa3f86adc20700f67bbbff2c8491f3f984134ac2963eab8` |
| Tensor file | `7e9e22fb5bb2125f1089511f3e8f0c4d541bb70976a929e82f7ce01f88fab28c` |
| Report payload | `bc6913b1772ed6967199e0ef3052336adbe5a78b7015fb79397484d29e029424` |

## Confirmation outcome

The conservative signed eight-group arm was subsequently frozen and tested on
fresh prompt-free origins 12, 28, and 36 against 63 preregistered size-matched
random partitions. It beat every random partition in pooled squared error and
the signed-GFA reference, but won only `6/8` native groups against a required
`7/8`. A diagnostic source-authoritative shadow on 16 already-consumed
Calibration-A prompts then failed decisively (`ΔNLL/token +2.72583`, KL
`3.01776`, top-1 agreement `40.49%`).

The full protocol, results, and next diagnostic are documented in
[Signed-g8 graph-wavelet confirmation](graph-wavelet-signed-g8-confirmation.md).
