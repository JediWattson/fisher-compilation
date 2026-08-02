# Graph-wavelet supermode mutation

This rung asks a narrower question than the graph-wavelet mapping experiment:

> At the same compiled rank, can two graph-wavelet directions be packed into
> one dense direction more faithfully than ordinary mode deletion?

The answer on the current five-origin Gemma 3 L3-to-L4 structural response is
promising development evidence, but it is not yet a model-compression or
speed result.

## Frozen input boundary

The experiment consumes the already-measured
`local_central_odd_tangent` response. It performs no model forward, prompt
load, tokenization, or new response measurement.

- Fit origins: `8`, `24`, `40`
- Development-selection origins: `16`, `32`
- Source modes: `64`
- Target modes: `64`
- Target rank: full rank `64`
- Candidate source ranks: `45` through `52`

Every graph, parent basis, merge action, loading, source basis, target basis,
and causal core is frozen using the three fit origins before either selection
origin is read. These five origins are development data; a later claim needs
new response-unopened origins.

## From a wavelet map to a mutation

Let the full fit-only signed graph-wavelet GOMP basis be

\[
Q\in\mathbb{R}^{64\times64},
\qquad Q^\mathsf{T}Q=I.
\]

Flatten the source-scale-weighted fit response over origin, lag, and target:

\[
Y\in\mathbb{R}^{64\times N},
\qquad A=Q^\mathsf{T}Y.
\]

For parent directions \(i\) and \(j\), form their response Gram block:

\[
G_{ij}
=
A_{\{i,j\}}A_{\{i,j\}}^\mathsf{T}.
\]

The best one-dimensional direction inside that two-dimensional span is the
leading eigenvector \(v_+\) of \(G_{ij}\):

\[
q_{ij}
=
\begin{bmatrix}q_i&q_j\end{bmatrix}v_+.
\]

This is the dense supermode. Its fit loss is the smaller eigenvalue of
\(G_{ij}\). The existing conditional spectral fitter then recomputes the
target decoder and causal knot cores around the folded basis. That refit is
the Jacobian/least-squares compensation step; it uses fit origins only.

Because accepted pairs have disjoint endpoints and every loading has unit
norm, merged columns remain mutually orthonormal. No extra merge transform is
executed at runtime: \(Qv_+\) is folded into the stored source basis.

## Preventing a merge from masquerading as deletion

A low rank-one loss alone is insufficient. Pairing one strong direction with
one nearly unused direction can produce a loading that is effectively
one-hot. The compiler therefore accepts a pair only when:

- each parent contributes at least `0.10` of squared loading;
- its loading has at least `0.90` absolute cosine in every leave-one-fit-origin
  out fold;
- graph-local variants place the pair in the undirected union of each
  endpoint's top `8` graph-interaction neighbors.

The `0.10` loading floor implies an effective contributor count above `1.2`
for every accepted two-parent supermode.

Eligible merge actions and ordinary singleton-prune actions compete in one
deterministic cost ordering:

- merge cost: the pair's smaller response-Gram eigenvalue;
- prune cost: the parent coordinate's fit response energy.

The compiler takes endpoint-disjoint actions until it reaches the requested
rank. It is therefore allowed to merge where bundling is cheaper and prune
where it is not.

## Matched controls

All compared executors have the same source rank, full target rank, three
knots, 32 lags, interpolation rule, and compensated core refit.

- `graph_local_merge`: primary dense merge-plus-prune path.
- `response_only_merge`: identical merge logic without graph eligibility.
- `graph_local_one_hot`: the primary action schedule with every merge forced
  to keep only its stronger endpoint.
- `signed_graph_wavelet_gomp`: ordinary nested wavelet pruning.
- `diagonal_fisher_prune`: top parent coordinates by fit response energy.
- `permuted_graph_local_*`: graph-local selection after deterministic topology
  permutations.
- signed graph-Fourier prefix and fit-energy controls.
- fit-only SVD prefix as the unconstrained linear headroom reference.

The one-hot arm isolates the value of dense loadings. The permuted arms
isolate the value of the learned graph. GOMP and diagonal pruning test whether
the action scheduler merely found a better delete order.

## Current open-development result

The strict loading floor leaves the rank-45 primary path with 15 genuine
two-parent merges and four singleton prunes.

| Rank-45 method | Selection relative error | Cosine |
|---|---:|---:|
| Graph-local supermode | **0.1842** | **0.9829** |
| Response-only supermode | 0.1852 | 0.9827 |
| Signed graph-Fourier prefix | 0.1900 | 0.9818 |
| Median permuted topology | 0.1934 | about 0.981 |
| Wavelet GOMP prune | 0.2043 | 0.9789 |
| Same actions, one-hot loadings | 0.2145 | 0.9767 |
| Graph-Fourier fit-energy | 0.1726 | 0.9850 |
| Fit-only SVD | 0.0506 | 0.9987 |

Against equal-rank GOMP pruning, the primary path recovers `18.7%` of squared
selection error. Recovery is positive at both selection origins:

- origin `16`: about `17.6%`;
- origin `32`: about `19.9%`.

That is evidence for meaningful dense bundling:

1. forcing the same merges to one-hot loadings makes the result worse;
2. destroying the graph topology gives back a substantial part of the gain;
3. ordinary rank-45 pruning misses the frozen fidelity gate, while the merged
   path passes it.

It is not a fully controlled winner. The graph-Fourier fit-energy and SVD
controls remain more faithful at the same rank. The result establishes that
localized merging adds information beyond deletion, not that this particular
pair compiler is globally optimal.

## Storage and compute accounting

For source rank \(R\), the folded conditional plan stores

\[
P(R)
=
64R + 64^2 + 3\cdot32\cdot R\cdot64
=
4096 + 6208R
\]

float64 coefficients.

At rank `45`, that is `283,456` coefficients versus `401,408` at rank `64`,
a `29.38%` coefficient reduction. The complete prepared runtime payload also
stores 64 source-normalization scalars and three knot positions: `2,268,184`
bytes at rank `45` versus `3,211,800` bytes at rank `64`, a `29.38%`
reduction. The storage gate uses those complete prepared-runtime byte counts.
Merge and pruning arms at the same rank have identical runtime storage and
analytic operation counts. Pair indices and loadings are analysis/compiler
metadata; executing or storing an additional parent-basis-to-merge transform
would invalidate the comparison.

No matched executor has been timed in this rung, so the compute gate remains
fail-closed. Lower analytic rank is not yet a latency claim.

## Reproducing the development artifact

```bash
fisher-graph-gemma-l3-l4-graph-wavelet-supermode-dev describe
fisher-graph-gemma-l3-l4-graph-wavelet-supermode-dev analyze
```

The authenticated local result is deliberately ignored by Git:

```text
.local-runs/google--gemma-3-270m/
  modal-generator-l3-l4-graph-wavelet-supermode-dev-v1.pt
  modal-generator-l3-l4-graph-wavelet-supermode-dev-v1.json
```

Its receipts are:

- logical candidate:
  `21a47a96efa83fc907c2d50fe06adc651a48ae7472cc57e6fe6b047bdc18515d`;
- tensor file:
  `5fa6a68f0068e32e4e8a78f4f90e6e64403c3e6fe7f8a632cd88be9fadef24b8`;
- report:
  `c06c5e70dae8a09a2a76f72d964f5e91b57335d825c089e5252bf428b4cd6248`.

The run fit and evaluated 96 matched conditional plans. It reconstructed and
authenticated the pushed v2 parent basis, loaded no model or tokenizer,
performed no model forward, and made no new response measurement.

## Next validation boundary

The immediate next step is to freeze one development nominee and its
equal-rank pruning control, then measure two genuinely response-unopened
origins. A fresh pass requires:

- relative error at most `0.20`;
- cosine at least `0.98`;
- at least `5%` squared-error recovery over pruning in aggregate;
- positive recovery at each fresh origin;
- at least one genuine multi-parent merge;
- at least `20%` compiled-plan reduction.

Only after that structural assessment should the folded plan be placed in a
physical model shadow run for NLL and downstream-task retention. Model
parameter compression and wall-clock speed remain unclaimed until those
tests exist.
