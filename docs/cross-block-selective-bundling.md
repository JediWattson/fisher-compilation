# Whole-model selective mode bundling

The single-layer pseudo-unit experiment established an important negative
result: low individual Fisher score does not imply that two units are
redundant. The next compiler rung therefore searches for redundancy directly,
across every MLP block and every Fisher rank, and permits zero merges when the
evidence is weak.

This rung separates three questions that must not be conflated:

1. Is a mode active or influential on only a small part of calibration data?
2. Do two modes have similar downstream influence?
3. Can one concrete executor coordinate replace both modes with acceptably
   small error?

The whole-model graph answers the first two questions. Only a compiled window
and a fresh guard evaluation can answer the third.

## Mode nodes and signed influence

One node identifies one activated pre-down-projection MLP coordinate:

```text
(layer id, layer ordinal, native unit index)
```

For example \(q\), valid token \(t\), mode activation \(z_{mqt}\), and a
shared final score \(S_q\), the discovery signature is

\[
s_{mqt}
=
z_{mqt}
\frac{\partial S_q}{\partial z_{mqt}}.
\]

Every site is differentiated against the same final score, so this scalar is
already reverse-transported through the complete native suffix. It is also
unchanged by reciprocal coordinate rescaling: multiplying \(z_m\) by a
nonzero constant divides its score gradient by the same constant.

The sign matters:

- correlation near \(+1\) is a redundancy candidate;
- correlation near \(-1\) is a cancellation candidate;
- correlation near zero is not a static bundle candidate.

Both row-local and per-sequence scopes are recorded. The sequence signature

\[
s_{mq}=\sum_{t\in\operatorname{valid}(q)}s_{mqt}
\]

preserves the cross-token terms in an empirical per-example Fisher quantity.
The row-local statistic remains useful for token density and causal
diagnostics. The two are labeled separately and are never presented as the
same Fisher.

## Density is not similarity

The implemented first pass measures valid-row concentration. For nonnegative
row energy \(e_{mqt}=s_{mqt}^{2}\), effective support density is

\[
D_m
=
\frac{\left(\sum_{q,t} e_{mqt}\right)^2}
{N\sum_{q,t} e_{mqt}^{2}},
\]

A mode whose energy is spread uniformly across the \(N\) valid token rows has
density one. A mode concentrated on one row has density \(1/N\). The same
calculation is applied to activation energy \(z^2\). This is an effective
participation ratio, not a count of exactly nonzero activations. Exact
shortlist replay separately retains per-sequence signed-influence
correlations, including cross-token terms; it does not relabel the row-density
statistic as per-example density.

Low density only says that a mode is concentrated. It does not show that
another mode can replace it. A rare mode may be essential, and two rare modes
may activate on completely different examples. Static bundling additionally
requires positive signed influence correlation, balanced Fisher energy,
coactivity, low joint-rank loss, and eventually a passing direct executor.

For an exact shortlisted pair, the signed contribution Gram is

\[
K
=
\frac{1}{Q}
\begin{bmatrix}s_i&s_j\end{bmatrix}^{\mathsf T}
\begin{bmatrix}s_i&s_j\end{bmatrix}.
\]

The discovery report keeps separate:

\[
\rho_F
=
\frac{K_{ij}}{\sqrt{K_{ii}K_{jj}}},
\qquad
\tau_F
=
\frac{\lambda_{\min}(K)}{\operatorname{tr}(K)},
\qquad
b_F
=
\frac{\min(K_{ii},K_{jj})}{\max(K_{ii},K_{jj})}.
\]

Here \(\rho_F\) measures signed similarity, \(\tau_F\) is a rank-one
reference, and \(b_F\) prevents a negligible mode paired with a large mode
from being mislabeled as a balanced bundle. As in the same-layer v3 plan,
rank-one tail energy is a reference, not a bound on the residual of a
particular executor.

## Bounded-memory whole-model search

Gemma 3 270M has 18 MLP blocks with 2,048 activated coordinates each, for
36,864 mode nodes. A dense all-node graph would contain roughly 679 million
unordered pairs; even after excluding same-block pairs, the cross-block graph
would contain about 642 million edges. Retaining every activation and
gradient row would require tens of gigabytes.

Discovery therefore uses two fit-only streaming passes:

1. Capture every MLP down-input site while cutting autograd once at the first
   transformer input. Update fixed-size deterministic sketches, Fisher energy,
   and density statistics, then discard the sequence rows.
2. Select a bounded low-density pool from every layer, search sketch-nearest
   neighbors across all Fisher ranks and blocks, freeze and hash that sparse
   shortlist, then replay fit data to accumulate exact moments only for those
   edges.

The first pass is a shortlist mechanism. The second pass still does not
authorize execution. Similar scalar score influence can hide a rank-two
vector effect or describe a serial causal lineage in which the later mode is
created by the earlier one.

Shortlisted edges are classified rather than forced into a perfect matching:

| Evidence | Discovery label | Interpretation |
|---|---|---|
| Positive influence similarity, coactivity, low joint rank, and low density | `static_merge_hypothesis` | May advance to intervention |
| Similar sequence direction but weak row coactivity | `noncoactive` | Conditional shared-slot lead only |
| Negative influence correlation | `negative_correlation` | Cancellation lead only |
| Missing endpoint energy | `zero_energy` | Deletion audit only |
| Strongly unequal endpoint energy | `energy_imbalanced` | Not a balanced bundle |
| Similar influence but non-rank-one activations | `activation_not_rank_one` | Needs a richer/window rewrite |
| A family fold misses the correlation threshold | `fold_unstable` | Rejected as unstable |
| Either endpoint exceeds a frozen density bound | `endpoint_density_too_high` | Rejected as dense |
| Exact similarity misses its threshold | `proxy_only_dissimilar` | Rejected as a sketch false positive |

Only threshold-clearing, endpoint-disjoint static hypotheses enter a proposed
bundle plan. Zero selected edges is a valid outcome.

## Development-only whole-model scan

The implementation was exercised against the pinned local
`google/gemma-3-270m` revision
`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. These are development results
from v9 calibration-A fit only. Guard, calibration B, validation, and test
were neither tokenized nor evaluated, and the local reports and tensor
artifacts remain under the ignored `.local-runs/` directory.

| Scan | Fit prompts | Valid rows | Low-density pool | Exact edges | Static hypotheses | Endpoint-disjoint |
|---|---:|---:|---:|---:|---:|---:|
| short/medium smoke | 16, two per family | 947 | 1,152 | 2,387 | 4 | 3 |
| all-length development | 40, five per family | 5,711 | 1,728 | 2,870 | 1 | 1 |

The broader scan retained the same native-coordinate pair seen in the smoke
scan:

```text
layer.6  unit 1202  ->  layer.15  unit 651
Fisher rank 11          Fisher rank 36
```

Its exact fit-side row and sequence signed-influence correlations were
`0.9281` and `0.9342`. Four family-disjoint folds ranged from `0.8957` to
`0.9744`; activation correlation was `0.9287`, coactivity was `0.9407`, and
the activation rank-one tail fraction was `0.0249`. Maximum endpoint
effective-support density was `0.00885` for activation energy and `0.00642`
for signed-influence energy. These density values describe concentration,
not literal zero-valued sparsity.

The other three smoke hypotheses retained high signed-influence similarity
on the broader data, but their activation correlation fell to
`0.804`–`0.833` and their rank-one tail fraction rose to `0.072`–`0.098`.
They were therefore rejected as `activation_not_rank_one`. This is a useful
negative result: downstream scalar influence can look redundant even when
the two generator coordinates cannot be represented by one scalar.

The survivor spans layers 6 through 15, so its smallest honest executable
test is a ten-block contiguous window. The discovery artifact itself remains
only a hypothesis: similar score influence may reflect serial causal lineage.
If a later compiled executor authorizes this carry edge, the Gemma residual
width of 640 implies only 1,280 saved parameters and 1,280 linear MACs per
valid token. That is about `0.000477%` of the 268,098,176-parameter model, so
one edge alone cannot constitute material model compression.

The proposal-only planner maps this result to exactly one unresolved window,
`layer.6` through `layer.15`. It deliberately records
`consumer_decoder_scale=None` and grants no intervention, compilation,
execution, guard, or calibration-B authority. The generic source-free stack
executor can realize a scale-authorized carry over contiguous structured
parents, but the current Gemma artifacts do not yet provide authenticated
source-free parents for every layer in this ten-block window.

## Development-only native replacement oracle

The direct intervention rung is now implemented. Gemma's native
`feed_forward_down_input` site can be replaced in-place, so the oracle can
observe unit 1202 at layer 6, carry that scalar across the unchanged native
blocks, and overwrite only unit 651 at layer 15. The anchor layer remains an
observer; no source weights or parameters are updated.

The scale is fitted without an intercept:

\[
\widehat z_{15,651}
=
\alpha z_{6,1202}.
\]

The default estimate weights each valid position by
\((\partial L/\partial z_{15,651})^2\), where \(L\) is summed causal-LM NLL.
This is a diagonal, single-coordinate Fisher surrogate for choosing the
scalar scale. It is not a full Fisher matrix, a full Jacobian transport, or a
claim that this scale is optimal for every downstream surface. The primary run
uses this Fisher-weighted scale; a matched diagnostic executes the ordinary
unweighted least-squares scale on the same examples and conditions.

For the development run, v9 calibration-A fit positions 0–39 fitted the scale
and disjoint positions 40–79 evaluated it. Each side contains five prompts
from each of eight families. The runner rejects overlap in example identity,
source prompt index, prompt hash, or materialized token content. Guard,
calibration B, validation, and test remained unopened. This is a content-
disjoint split within the same eight synthetic template families, not
generalization to fresh families or natural text. Scale-fit sequences covered
43–256 valid positions, while every evaluation sequence reached the 256-token
cap.

The fitted Fisher-weighted scale was `-0.450315`. Its four
family-disjoint leave-one-fold-out fits ranged from `-0.460140` to
`-0.439631`, retained the negative sign in every fold, and had a maximum
relative scale deviation of `2.37%`. The unweighted fit was `-0.506844`.
The negative sign is meaningful: the later coordinate is reconstructed as an
oppositely signed, scaled version of the earlier coordinate rather than as a
positive copy.

On the unseen fit-only evaluation subset:

| Condition | Consumer-coordinate NRMSE | Consumer-MLP NRMSE | Window-output NRMSE | Final-logit NRMSE | Native-teacher KL/token | Top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| delete layer-15 coordinate | 1.000000 | 0.012232 | 0.006765 | 0.010538 | 0.00022352 | 99.8235% |
| carry correct layer-6 coordinate | 0.450584 | 0.005512 | 0.002715 | 0.002844 | 0.00005234 | 99.7059% |
| carry family-deranged coordinate | 1.136400 | 0.013900 | 0.006860 | 0.010035 | 0.00026290 | 99.6667% |

Relative to deletion, the correct carry recovered `79.70%` of
consumer-coordinate squared error, `79.70%` at the consumer MLP output,
`83.89%` at the layer-6-through-15 window output, and `92.72%` at final
logits. The shuffled control recovered `-29.14%`, `-29.14%`, `-2.84%`, and
`9.33%` respectively. Thus the gain depends on the correct token-aligned
source, rather than merely injecting a scalar with a similar marginal
distribution.

The matched unweighted scale also succeeded: its recoveries were `79.64%` at
the consumer coordinate, `79.64%` at the consumer MLP output, `84.65%` at the
window output, and `92.47%` at final logits. Fisher weighting reduced
native-teacher KL from `0.00005831` to `0.00005234` per token, a `10.24%`
improvement over the unweighted carry, and improved final-logit recovery by
`0.24` percentage points. The unweighted scale improved window-output
recovery by `0.76` points and absolute-NLL recovery by `5.69` points. The
evidence therefore shows that scalar cross-block substitution is robust to
both fits and gives Fisher weighting a modest final-behavior advantage on this
sample; it does not establish uniform or general Fisher superiority.

The correct carry also recovered `76.58%` of deletion's native-teacher KL and
had lower KL than deletion in all eight evaluation families. It beat the
family-shuffled control in seven of eight families. Its absolute ground-truth
NLL displacement recovered `39.09%` of deletion's displacement, while the
shuffled carry recovered `55.20%`; NLL therefore does not supply the
pair-specific evidence. Signed NLL alone is not a fidelity score here: all
three perturbations happened to lower NLL slightly, and deletion retained
marginally higher top-1 agreement than replacement. The paired internal
errors and native-teacher KL carry the causal-substitution evidence instead.
The shuffle is one deterministic, batch-local, family derangement, not a
distribution over random permutations.

All 10,240 valid activation positions had shuffled-control coverage. The
frozen-source audit reported zero change to the anchor observation, every
non-consumer coordinate, invalid positions, parameters, parameter versions,
state hash, and model/execution fingerprints.

This is strong development evidence for a pair-specific cross-block
substitution, not an authorized compression result. The artifact explicitly
sets every guard, calibration-B, compilation, execution, and further-
intervention authority to false, applies no pass/fail threshold, and marks the
proposal unresolved. The oracle still executes the native layer-15 gate/up
computation before overwriting its output, so it realizes zero parameter,
MAC, memory, or latency saving. A compiled executor would have to remove that
generator row, preserve the later decoder column, and then pass a
preregistered fresh-corpus guard.

## Physically merged executor and fresh-family guard

That next rung is now implemented, and it cleanly separates an engineering
success from a scientific failure.

The directed merged-supermode executor keeps layer-6 unit 1202 as an ordinary
native generator and also carries its signed scalar to layer 15. Layer-15 unit
651's gate and up rows are physically absent from the candidate; its down
column remains inside the complete native layer-15 down projection. This is
**merge and reuse**, not deletion:

\[
q_t = z_{6,1202,t},
\qquad
\widehat z_{15,651,t} = -0.450315\,q_t .
\]

Deletion is evaluated only as the counterfactual control
\(\widehat z_{15,651,t}=0\). All intervening layers 7–14 continue to execute
through Gemma's native path. Equivalence is defined on valid query positions;
the carried coordinate is zero on padding rows, which cannot influence valid
queries through the causal attention mask.

The local executor:

- binds both the source weight fingerprint and execution-configuration
  fingerprint;
- owns cloned candidate tensors with no source-storage alias;
- exposes a strict CPU artifact round trip whose execution fingerprint covers
  topology, binding, and tensor state;
- restores the original Gemma MLP modules even if an overlaid forward raises;
- records zero calls to the original anchor and consumer projections during a
  compiled forward;
- reduces the candidate consumer gate/up width from 2,048 to 2,047 while
  retaining the full 640-by-2,048 consumer down projection.

That strict local artifact contains cloned candidate weights. It is stored
only under the ignored `.local-runs/` tree; no Gemma weights or guard prompts
are added to the repository.

The development mechanical smoke was bit-exact at the anchor output, consumer
input, carried coordinate, consumer MLP output, layer-15 output, and final
logits. It therefore established that the physical candidate realizes the
activation oracle; it did not re-establish that the oracle generalizes.

Before opening fresh data, guard v2 froze the corrected executor source,
source model and execution fingerprints, plan, oracle, scale, prompt bytes,
materialized token stream, and numerical thresholds. It reused the still-
unopened v1 corpus: 64 synthetic prompts from eight new topic families,
balanced over micro, compact, medium, and long bands. The audit found no exact,
normalized, family-ID, or 5–8-token-gram overlap with prior prompt artifacts.
The run contained 10,778 valid positions and 10,714 supervised next-token
positions. Calibration B, validation, and test remained unopened.

The compiled path matched the coordinate-replacement oracle exactly—zero
observed difference at every guarded internal and behavioral surface—but the
frozen replacement itself lost decisively to deletion:

| Condition | Consumer-MLP NRMSE | Layer-15 output NRMSE | Final-logit NRMSE | Teacher KL/token | Top-1 agreement | Delta NLL/token |
|---|---:|---:|---:|---:|---:|---:|
| delete unit 651 | 0.002996 | 0.001258 | 0.001904 | 0.00001589 | 99.7760% | -0.00011918 |
| merged supermode / activation oracle | 0.004940 | 0.002316 | 0.003030 | 0.00005765 | 99.7480% | -0.00047891 |
| physically merged executor | 0.004940 | 0.002316 | 0.003030 | 0.00005765 | 99.7480% | -0.00047891 |

Relative to deletion, the merged candidate recovered `-171.80%` of
consumer-MLP squared error, `-239.21%` at the layer-15 output, `-153.25%` at
final logits, and `-262.82%` of native-teacher KL. Negative recovery means the
merge introduced more error than removing the later coordinate entirely.
Every one of the eight fresh families and all four length bands had negative
recovery at all three tensor surfaces and in KL. This is not a threshold-edge
failure or an executor discrepancy.

A post-hoc diagnosis on the consumed guard explains the reversal. The frozen
Fisher-weighted fit used scale `-0.450315`; the development unweighted fit was
`-0.506844`. On fresh data, the best unweighted no-intercept scale would have
been only `-0.048076`, and even that retrospective optimum explains just
`2.49%` of deletion error. Centered anchor/consumer correlation collapsed to
`-0.12935`. The best affine diagnostic had slope `-0.02555`, intercept
`0.01402`, and \(R^2=0.01673\). The fresh consumer coordinate was therefore
mostly a small positive baseline, not a stable scaled copy of the earlier
coordinate. These post-hoc numbers cannot select a new candidate or authorize
another split.

The resource ledger remains mechanically valid for this rejected candidate:
one consumer gate row plus one up row removes 1,280 learned parameters and
1,280 linear MACs per valid token; the fixed signed carry costs one stored
coefficient and one multiply, leaving 1,279 net coefficients and arithmetic
MACs per valid token. The in-memory candidate module tree contains 268,096,896
learned parameters instead of 268,098,176. The evaluation overlay still keeps
the complete base model alive for comparison, so no end-to-end resident-memory,
serialized-model-size, latency, or kernel-speed reduction is claimed.

The conclusion is narrow but strong: the directed merge mechanism and
physical executor work, while this discovered pair is not a stable shared
supermode and is rejected as compression. Future discovery must require
cross-domain activation-relation stability before selecting an edge, or use a
richer/conditional multi-source predictor. The consumed guard cannot be used
to choose that design.

## Cross-block execution: anchor and carry

Weights from different blocks cannot be averaged directly. Normalization,
attention, nonlinear MLPs, and residual additions lie between them. A direct
cross-block 2-to-1 executor instead uses an earlier coordinate as an anchor:

1. Compute an anchor scalar \(q\) at the earlier layer.
2. Apply its ordinary earlier-layer decoder.
3. Carry \(q\) forward as a token-aligned internal graph value.
4. Remove one later gate/up generator row.
5. Inject \(q\) at the later depth through that mode's depth-specific decoder.

Invalid query rows in the carried slot are explicitly zero. The carry is
token-local and therefore does not create a future-to-past edge.

For residual width \(d\), this removes the later gate and up rows, saving
\(2d\) stored coefficients and \(2d\) linear MACs per valid token. The later
decoder still exists as the carry injection, so a cross-block bundle must not
claim the \(3d\) saving of a same-layer width reduction. Low density creates
no additional runtime saving until a measured sparse or conditional kernel
can skip inactive work.

Every edge covers a contiguous layer interval. Overlapping intervals are
merged into one compiled component:

```text
edge 2 -> 3  ┐
edge 3 -> 5  ├── compiled window 2..5
edge 8 -> 9  ┘   compiled window 8..9
```

This lets discovery cover the whole model without requiring the first
executable experiment to replace all 18 blocks monolithically. If selected
intervals eventually connect the whole depth, the same representation becomes
a whole-transformer span.

## Scientific gate

The consumed v9 guard cannot choose density cutoffs, neighbor count, layer
span, pair budget, or window layout. V9 fit may be used only for development.
A representative experiment requires a fresh corpus identity with
family-disjoint fit folds and a fresh A guard.

The fit folds may select exactly one graph and executor composition. Before
the guard opens, the run freezes:

- model revision and source fingerprint;
- prompt, family, token-stream, and position-schedule hashes;
- sketch algorithm, dimensions, and seeds;
- shortlist and exact-edge hashes;
- thresholds and total residual budget;
- selected endpoint-disjoint edges;
- contiguous compiled windows;
- executor tensors and execution fingerprints.

The discovery artifact contains no executable weights and cannot authorize
calibration B. A direct composed executor must first pass every window's
boundary gate, worst-family and length-bucket bounds, whole-model final-hidden
and logit checks, NLL, KL, top-1, causal/mask probes, source-call exclusion,
strict reload, and complete parameter/MAC accounting on the fresh A guard.
Passing that guard means only that the frozen candidate may open one fresh
calibration B.
