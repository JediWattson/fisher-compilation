# Generator causal map

The causal-fingerprint rung measures what happens at the model output when one
generator is absent. That is enough to rank singleton necessity and compare
final effects, but it is not yet a map of how computation propagates through
the compiled stack.

The causal-map rung keeps the frozen full-refit stack unchanged and adds a
multiplex evidence graph:

```text
frozen generator stack
  -> singleton effect similarity
  -> exact pair suppression
  -> directed downstream response
  -> prompt-cohort conditioning
  -> typed causal map
```

“Multiplex” means that different scientific relationships remain different
edge types. They are not collapsed into one convenient but ambiguous graph
score.

The map is analysis-only. It does not merge, share, route, remove, prune,
compile, or execute a replacement graph.

## Nodes

The first complete map has one node for each deployed full-layer generator.
Every node binds:

- layer and causal order;
- deployed generator-plan and fit hashes;
- frozen base or sequential-refit source;
- input, latent, and output widths;
- learned parameters, matrix MACs, and bias additions;
- its exact singleton fingerprint and causal-necessity summaries.

The node is a generator-scale computational unit. Its fitted latent coordinates
remain available for a later mutation fit, but the v1 causal map does not claim
that every latent coordinate is already an independently understood semantic
mode.

## Singleton-effect similarity edges

The existing undirected edge compares two singleton suppressions in a shared
bounded output frame:

- centered anchor-logit effect cosine;
- prompt NLL Spearman;
- top-effect prompt overlap; and
- effect-sign agreement.

This edge answers:

> Do these generators have similar final effects when each is removed alone?

It does not answer whether the generators interact, whether one changes the
other, or whether either can substitute for the other.

## Exact joint-interaction edges

For baseline prompt NLL \(L_0\), singleton suppressions \(L_i,L_j\), and exact
joint suppression \(L_{ij}\), define:

\[
\kappa^{\mathrm{NLL}}_{ij}
=
L_{ij}-L_i-L_j+L_0.
\]

- Positive values are superadditive suppression damage.
- Negative values are subadditive suppression damage.
- Zero is additive in NLL on that prompt.

NLL is nonlinear, so this second difference can be nonzero even when the
underlying logits combine additively. The map therefore also measures the
centered anchor-logit interaction residual:

\[
r_{ij}
=
c_{ij}-c_i-c_j,
\]

where each \(c\) is the exact muted-minus-baseline effect in the same frozen
anchor frame. An additive logit system has \(r_{ij}=0\).

The normalized interaction magnitude is:

\[
\eta_{ij}
=
\frac{\operatorname{RMS}(r_{ij})}
{\sqrt{
\operatorname{mean}(c_i^2)
+
\operatorname{mean}(c_j^2)
}}.
\]

The numerator, denominator, ratio, and defined flag are stored separately.
Joint baseline-to-condition KL and top-1 agreement remain descriptive
magnitude checks.

Joint interaction is symmetric. Its sign can suggest overlapping,
compensatory, or serial behavior, but it does not establish direction or
mediation.

## Directed downstream-response edges

During the singleton suppression of upstream generator \(i\), the executor
also observes every active downstream generator \(j\):

\[
q_{i\rightarrow j}
=
g_j(h_j^{-i})-g_j(h_j^0),
\qquad i<j.
\]

This edge is directed because a later layer cannot change an earlier layer in
the forward pass. It records:

- downstream response RMS;
- downstream baseline-output RMS;
- response-to-baseline RMS ratio and defined flag; and
- cosine between the response and the downstream generator's usual output.

A positive cosine means the downstream response extends its usual output
direction; a negative cosine means it retracts or opposes that direction. This
does not by itself prove compensation.

Every generator earlier than the suppressed source must remain bit-identical.
Those reverse-direction checks are mandatory causal negative controls, not
ordinary low-weight edges.

This is total downstream influence. Intermediate attention, residual, and
normalization operations may mediate it, so the map does not call it a direct
dependency.

## Prompt-cohort edges

The frozen development export already carries declared prompt-family
membership. The JSON map hashes each family identifier, records its exact
opaque membership, and summarizes each generator's necessity inside and
that cohort:

- mean signed and absolute NLL damage;
- mean KL and bounded effect magnitude;
- positive-effect fraction;
- mean top-1 agreement; and
- mean within-prompt absolute-NLL importance rank.

These edges expose conditional computation without inventing a human semantic
label. The current families are adaptive open-development cohorts, not a fresh
guard.

## Complete schedule and bounded memory

For 18 generators, one prompt batch executes:

| condition | forwards |
| --- | ---: |
| generated baseline | 1 |
| singleton suppressions | 18 |
| unordered pair suppressions | 153 |
| total | 172 |

Across the 20-prompt assessment, the exhaustive map performs 3,440 full-model
forwards. That is `9.05x` the 380-forward singleton-fingerprint run.

One stable overlay is installed for the entire batch. Conditions are visited
synchronously in canonical order and immediately reduced. Peak retained map
state does not grow with the 153 pairs:

- one baseline full-vocabulary log-probability tensor;
- bounded singleton anchor effects;
- one current condition output;
- baseline generator outputs used for directed comparisons; and
- prompt-level scalar summaries.

Joint logits and raw generator-output rows are never retained in the
authenticated analysis or JSON artifact.

## Live 18-generator result

The complete run restored the frozen sequentially refit Gemma 3 270M stack,
replayed the previously published singleton fingerprint exactly, and completed
all 3,440 scheduled forwards without an invariance or restoration failure. The
strict artifact contains:

| object | count |
| --- | ---: |
| generator nodes | 18 |
| copied singleton-similarity edges | 153 |
| exact joint-interaction edges | 153 |
| forward directed-response edges | 153 |
| prompts | 20 |
| declared prompt cohorts | 8 |
| generator-cohort affinities | 144 |

The three edge planes are related but not interchangeable. Across all 153
pairs, centered-effect cosine and normalized joint interaction have Spearman
correlation `0.520`, while centered-effect cosine and directed-response ratio
correlate only `0.222`. Joint interaction and directed-response ratio correlate
only `0.086`. A single blended edge score would therefore discard important
structure.

The joint NLL signs are almost balanced: 78 pairs are superadditive and 75 are
subadditive. The strongest superadditive edge is layers 3-10
(`kappa = +8.619503`), and four more of the strongest positive edges converge
on layer 10. The early stack is therefore not just one redundant chain; layer
10 behaves as a complementary convergence region for several early
generators.

Layers 0 and 3 are the largest outgoing response hubs. Their mean downstream
response-to-baseline ratios are `0.825` and `0.722`, compared with `0.545`
over all directed edges. Ratios are always read together with absolute response
RMS because a small downstream baseline can inflate a ratio.

The declared cohorts do not yet reveal a routing split. Their 18-layer
singleton-necessity profiles remain highly similar: pairwise Spearman ranges
from `0.893` to `0.990`. Two cohorts contain only one prompt, and the remaining
cohorts contain only two to four prompts, so this is descriptive evidence
rather than a conditional-execution gate.

### Frozen first motif: layers 3 to 4

The predeclared layers-3-and-4 lead survives all three causal planes:

| plane | measurement | rank among 153 |
| --- | ---: | ---: |
| singleton final-effect similarity | cosine `0.656025` | 3 |
| prompt effect co-variation | Spearman `0.729323` | descriptive |
| exact joint interaction | `kappa = -4.185928` NLL/token | 4th most negative |
| normalized logit interaction | `eta = 0.677071` | 21 |
| directed layer-3 to layer-4 response | ratio `0.802304` | 9 |
| directed response orientation | cosine `-0.622848` | descriptive |

The layer-4 response RMS is `0.104289`, or 80.2% of its ordinary generator
output RMS. When layer 3 is absent, layer 4 retracts strongly against its usual
output direction. Negative response cosine is common in this graph—148 of 153
directed edges are negative—so its sign is not sufficient evidence by itself.
Here it accompanies high final-effect similarity, a large directed response,
and an NLL second difference that is negative on all 20 prompts and in every
declared cohort. Together those measurements make serial lineage or shared
computation a better hypothesis than two interchangeable parallel nodes.

The pair is not remotely deletable. Jointly muting layers 3 and 4 raises NLL
by `7.909090` per token, produces KL `8.110894`, and preserves only `0.35%` of
the baseline top-1 outputs.

Layers 0-1 are even stronger as a raw serial-dependency candidate: their directed
response ratio is `1.140915`, normalized joint interaction is `0.722890`, and
`kappa = -5.355337`. They are also two of the most individually necessary
generators, so layers 3-to-4 remain the already predeclared topology probe.
Layers 0-1 stay as a replication target rather than an adaptive first choice.

No result in this map supports deletion. Every singleton mute worsens NLL on
every prompt, including the least-sensitive layer.

## How the map selects a mutation type

The frozen map should choose the kind of experiment, not authorize the result:

| observed motif | appropriate mutation hypothesis |
| --- | --- |
| Similar singleton effect, superadditive joint damage, limited serial response | parallel compensation; test a shared core or fan-out |
| Strong directed response and subadditive joint damage | serial lineage; test transport, fusion, or a correction edge |
| Similar output direction on different high-effect prompt cohorts | conditional shared slot; test routing without turning required nodes off |
| Opposing output directions with stable joint structure | joint low-rank packing with separate signs/scales |
| Stably negligible singleton and joint effect | removal candidate |

These are hypotheses. A shared core can reduce stored parameters while still
executing twice. Routing can reduce average compute while retaining all
parameters. Physical pruning can reduce both only after an exact reversible
candidate passes its declared fidelity and resource budgets.

## Mutation handoff

The map-to-mutation boundary is:

```text
strict causal map
  -> freeze one relationship and mutation type
  -> fit on separate mutation-fit data
  -> select capacity on separate selection data
  -> install reversible overlay
  -> open fresh family-disjoint guard
  -> accept or roll back
  -> physically lower only after acceptance
```

The current assessment has already influenced the map design and candidate
search. It cannot serve as the fresh guard for the mutation it proposes.

The map produces two deliberately separate handoffs.

### Topology handoff

The frozen topology proposal is a reversible **serial L3-to-L4 supermode**, not
an average of the two generators and not deletion:

1. fit a transport/fused-composition path from the layer-3 generator state into
   a layer-4-specific head;
2. retain an explicit residual correction for computation unique to layer 4;
3. install it behind a scalar gate whose zero setting is the exact current
   two-generator stack;
4. shadow-run the candidate before switching either endpoint;
5. select its capacity on data disjoint from its fit rows; and
6. evaluate it once on a fresh prompt-family-disjoint guard.

This experiment asks whether a typed causal edge can reproduce serial
computation. It is not the safest first compaction target.

### Dense-packing handoff

The first storage-and-compute mutation proposal is layers 12 and 15:

| measurement | layers 12 and 15 |
| --- | ---: |
| singleton NLL damage | `0.636314`, `0.548748` |
| centered-effect cosine | `0.319543` |
| prompt NLL Spearman | `0.756391` |
| joint `kappa` | `+0.016218` |
| normalized interaction `eta` | `0.494833` |
| directed response ratio | `0.411872` |

These are lower-impact, weakly serially coupled generators whose effects
co-vary. The proposed mutation is a shared low-rank basis or codebook with two
layer-specific heads and a small residual correction. Each original generator
remains independently switchable as an exact fallback. The candidate sweeps
local resource ratios `0.90`, `0.75`, and `0.50`; it is accepted only if it
both preserves fidelity and realizes a declared parameter, byte, and MAC
reduction after all heads and correction state are counted.

This pair was discovered adaptively on the map. Freezing it now prevents
further pair shopping, but the current prompts cannot validate it. Layers
14-and-16 provide a lower-impact structural control for the same packing
scaffold.

Only an accepted candidate may proceed to physical packing. That later
lowering must report realized parameters, resident bytes, logical MACs,
backend latency, and fidelity against both the independent-generator baseline
and a matched-budget quantization or SVD control.

The ignored source-safe result is:

```text
.local-runs/google--gemma-3-270m/
  modal-generator-causal-map-dev-v1.json
```

Its scientific payload digest is
`1a25859340cd4772730fc631cdd7d7b859dda73c81d2447bed33c025d1e73afa`.
The complete JSON file digest is
`bbcd06ffff1ae164ae8a18fdd43c325d78d3f2692fc0013dfe36c5af725b895b`.
The artifact is tensor-free and contains no prompt text, token IDs, logits,
activations, generator weights, or model weights. Its 20-prompt assessment is
adaptive open-development evidence, not heldout confirmation.
