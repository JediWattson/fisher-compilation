# Generator causal fingerprints and transactional mutation

The current compiled Gemma stack contains one fitted modal generator at every
MLP layer. Before sharing, merging, routing, removing, or physically pruning
any of those generators, the compiler needs evidence about what each generator
actually does inside the complete compiled model.

This rung collects that evidence without changing the graph:

```text
frozen compiled model
  -> exact singleton generator suppression
  -> prompt-conditioned shared-output effects
  -> causal fingerprints
  -> generator-family hypotheses
```

It is an analysis rung, not a compression rung. All generators, native source
weights, and rollback state remain resident and immutable. Singleton
suppression does not authorize a merge, removal, route, replacement, or
physical pruning.

The mutation protocol described later in this document is the intended next
rung. It is not yet implemented.

## 1. The paired intervention

Let \(G\) denote the frozen, fully generated stack and let \(g_i\) be the
generator installed at layer \(i\). For the same prompt \(x\), compare:

\[
y_*(x)=G(x)
\]

with:

\[
y_{-i}(x)=G_{-i}(x),
\]

where \(G_{-i}\) executes every generator except \(g_i\). At layer \(i\), the
generated residual is replaced by an exact zero. The native MLP is not restored
for that layer.

The intervention therefore asks:

> What changes at the shared model output when this one generator contribution
> is absent from the otherwise unchanged compiled graph?

Every comparison must be paired:

- identical model weights;
- identical generator plans;
- identical prompt and tokenization;
- identical causal positions;
- identical execution order;
- all non-target generators active;
- exactly one suppressed generator;
- no fitting, routing, or parameter mutation during the replay.

The executor must restore the native model and compiled catalog exactly after
every successful or failed replay. The baseline and suppression conditions must
bind the same source-model, refit-overlay, generator-plan, and prompt-split
digests.

### What singleton suppression isolates

Singleton suppression measures the causal necessity of one generator in the
current compiled context. It includes every nonlinear downstream consequence
of removing that contribution. This is stronger evidence than comparing raw
generator weights or local activations alone.

It does not isolate a context-free semantic object. The intervention may move
the suffix off the trajectory on which it was fitted, and another generator
may compensate for, amplify, or depend on the suppressed generator. The result
is a local property of the frozen full-stack graph and the measured prompt
distribution.

Joint suppression, substitution, and an executable mutation are still required
before claiming redundancy.

## 2. The implemented common output frame

Generator outputs at different layers live at different points in the
transformer trajectory. Directly comparing their local residual vectors would
conflate generator identity with layer-local coordinates. Every singleton
intervention ultimately changes the same final vocabulary logits, which gives
all layers a common downstream comparison frame.

For each prompt \(x\) and supervised position \(t\), version 1 constructs the
anchor set:

\[
A_{x,t}
=
\{\text{target token}\}
\cup
\{\text{eight highest baseline non-target logits}\}.
\]

Ties use a stable order. If \(\ell_*\) denotes the full compiled baseline
logits and \(\ell_{-i}\) the logits with generator \(i\) suppressed, the
stored bounded effect is:

\[
c_i(x,t)
=
C_A\left(
\ell_{-i}(x,t,A_{x,t})
-
\ell_*(x,t,A_{x,t})
\right),
\]

where \(C_A\) subtracts the mean over the nine anchor coordinates. Centering
removes the additive softmax gauge. It is therefore equivalent to centering
\(\log p_{-i}-\log p_*\) over the same coordinates.

This is a centered anchor-logit effect, not an output-Fisher tangent. Version
1 does not weight coordinates by the baseline probabilities. The generators
being studied were derived from Fisher-ranked computational modes, but this
causal fingerprint is a separate finite-intervention measurement at the model
output.

For each prompt, the implementation forms a mean anchor-effect Gram
contribution:

\[
G_{ij}(x)
=
\frac{1}{T_x |A|}
\sum_{t,a}
c_i(x,t,a)c_j(x,t,a).
\]

Prompt contributions receive equal weight when they are combined. The
published artifact records the anchor rule, width, centering rule, Gram
weighting, and authenticated tensor digests. It does not publish prompt text,
token IDs, targets, raw logits, token-level effect rows, a per-prompt anchor
catalog, or covered baseline probability mass.

## 3. Prompt-conditioned fingerprints

One global mean can hide conditional computation. The fingerprint therefore
retains one safe summary row per prompt and generator before calculating
whole-split aggregates.

For generator \(i\) and prompt \(x\), the minimum fingerprint contains:

### NLL change

\[
\Delta\operatorname{NLL}_i(x)
=
\operatorname{NLL}_{-i}(x)
-
\operatorname{NLL}_*(x).
\]

- Positive: the suppressed generator helped the compiled baseline on that
  prompt.
- Negative: suppressing it improved NLL on that prompt.
- Near zero: the hard-target likelihood barely changed.

Near-zero NLL does not mean the generator is redundant. It observes only the
target probability and can miss redistribution among non-target tokens.

### Baseline-to-suppressed KL

\[
\operatorname{KL}_i(x)
=
D_{\mathrm{KL}}\left(p_*\;\|\;p_{-i}\right).
\]

This measures how much the complete output distribution moved. It is
nonnegative and direction-free: it measures magnitude, not whether two
generators caused the same change.

### Top-1 agreement

The fraction of supervised positions where the suppressed condition preserves
the baseline argmax. It is easy to interpret, but coarse. A generator can have
a substantial distributional effect while leaving top-1 unchanged.

### Centered anchor-logit effect RMS

\[
R_i(x)
=
\sqrt{
\frac{1}{T_x|A|}
\sum_{t,a}c_i(x,t,a)^2
}.
\]

This is the size of the generator's bounded relative-logit effect in the
shared nine-coordinate frame. It is not a Fisher energy, and it does not
replace the full-vocabulary KL. Version 1 does not serialize token-level target
log-probability changes separately.

## 4. Pairwise generator evidence

The prompt-conditioned fingerprints define a weighted graph over generators.
An edge is evidence about a possible relationship, not an executable rewrite.

### Centered shared anchor-logit cosine

\[
\rho_{ij}
=
\frac{G_{ij}}
{\sqrt{G_{ii}G_{jj}}}.
\]

- Near \(+1\): the generators tend to move the final distribution in the same
  relative-logit direction inside the bounded anchor frame.
- Near \(-1\): they tend to oppose one another.
- Near zero: their measured output directions differ.

Version 1 serializes `0.0` when either norm is zero and marks the pair
`insufficient_causal_variation`; that zero must not be interpreted as
orthogonality.

### Prompt-rank correlation

Spearman correlation is computed over the two vectors of prompt-level NLL
changes. It asks whether the generators become important on the same prompts.
It is about conditional use, not output direction. Two generators can have
similar prompt rankings but different or opposing final-output effects.

### High-effect overlap

For each generator, version 1 selects the five prompts with the largest
absolute NLL change. It records the intersection divided by five and the sign
agreement inside that intersection. Weak overlap with a similar aggregate
direction is more naturally a conditional shared-slot hypothesis than an
unconditional merge.

### Frozen observational policy

The current policy calls a pair aligned only if all four conditions pass:

- centered anchor-effect cosine at least `0.90`;
- prompt NLL Spearman at least `0.80`;
- top-five importance overlap at least `0.60`; and
- sign agreement within that intersection at least `0.80`.

Passing at least two conditions produces `mixed_observational_family_evidence`;
passing fewer produces `distinct_observational_effect_hypothesis`. The pair
must also have at least three prompts, nonzero effect norms, and nonconstant
NLL effects. These labels are discovery aids only.

Version 1 does not compute best scalar gain or deterministic split-half
stability. Those are useful additions before a family can be promoted beyond
an exploratory mutation candidate.

## 5. Family hypotheses are not redundancy claims

A generator family is a set of nodes with evidence that some computation may
be shared. Version 1 proposes that evidence from:

- aligned centered anchor-logit directions;
- similar prompt-conditioned NLL importance;
- overlapping high-effect prompts and matching effect signs; and
- sufficient causal variation at both endpoints.

This evidence supports experiments such as joint factorization or conditional
routing. It does not show that one member can be deleted.

In particular:

- similar effects can arise from serial causal lineage;
- two generators may be individually replaceable but jointly necessary;
- the same output effect may be produced from incompatible layer inputs;
- low singleton effect can be caused by downstream compensation;
- high cosine says nothing about parameter or MAC savings;
- prompt-distribution similarity may not generalize.

The core fingerprint artifact therefore declares:

```text
analysis_only = true
observational_hypotheses_only = true
authorizes_intervention = false
authorizes_mutation = false
authorizes_merge = false
authorizes_routing = false
authorizes_pruning = false
authorizes_compilation = false
authorizes_execution = false
```

## Development result: the 18-generator refit stack

The first live run applied exact singleton suppression to all 18 generators
in the sequentially refit Gemma stack. The baseline for these deltas is the
compiled refit model itself: NLL `2.973636`, compared with native NLL
`2.823987`. Each generator was muted once per prompt while the other 17
remained active.

Every mute increased NLL on every one of the 20 prompts. There is therefore no
near-zero removal or route-off candidate in this sample.

| layer | mean muted-minus-baseline NLL | baseline-to-muted KL | baseline top-1 agreement |
| ---: | ---: | ---: | ---: |
| 3 | +9.0835 | 9.1112 | 3.14% |
| 0 | +8.0574 | 8.4107 | 5.12% |
| 1 | +3.8061 | 3.9289 | 25.08% |
| 4 | +3.0116 | 2.9412 | 32.29% |
| 2 | +2.0955 | 2.2347 | 35.96% |
| 16, least NLL-sensitive | +0.1995 | 0.2870 | 79.76% |

The sensitivity is strongly front-loaded. Mean NLL increases are `4.638` for
layers 0-5, `1.023` for layers 6-11, and `0.420` for layers 12-17. Even layer
16 ranges from `+0.1137` to `+0.4212` across prompts, so “least sensitive”
does not mean dispensable.

The pair graph contains all 153 unordered layer pairs:

| observational label | pair count |
| --- | ---: |
| aligned family hypothesis | 0 |
| mixed family evidence | 50 |
| distinct effect hypothesis | 103 |
| insufficient variation | 0 |

No pair reaches the frozen `0.90` cosine requirement; the maximum is `0.6997`.
The strongest mixed leads are:

| pair | centered effect cosine | NLL Spearman | top-five overlap |
| --- | ---: | ---: | ---: |
| layers 3-4 | 0.6560 | 0.7293 | 0.60 |
| layers 2-4 | 0.6246 | 0.7474 | 0.60 |
| layers 1-2 | 0.6068 | 0.6256 | 0.60 |

Layers 1-4 form a loose early-region pattern, not an interchangeable clique.
For example, layers 1-4 have the largest cosine (`0.6997`) but only `0.20`
top-five prompt overlap. Conversely, layers 13-14 have the largest NLL
Spearman (`0.8827`) but only `0.2304` output-effect cosine. The first pair
moves outputs similarly on different high-effect prompts; the second becomes
important on similar prompts while moving the output differently.

There is real local continuity: adjacent pairs average `0.404` cosine and
`0.519` NLL Spearman, versus `0.206` and `0.356` for nonadjacent pairs.
Cross-layer candidates still exist, but distance is neither a family rule nor
a merge rule.

This result promotes layers 3-4 and layers 2-4 into the next causal-map rung.
It does not yet distinguish shared parallel computation from serial lineage,
and therefore does not yet select a shared-core mutation. It also does not
support deletion, routing a generator completely off, substituting one
generator for another, or physically pruning any state.

The ignored source-safe result is:

```text
.local-runs/google--gemma-3-270m/
  modal-generator-causal-fingerprint-dev-v1.json
```

Its scientific payload digest is
`754d31b333a35208e7bc434a48a2f6ebed99951b0088871dbce5388b6e5c4b17`.
The 20-prompt assessment is adaptive open-development data, and the
nine-coordinate frame is intentionally bounded. These numbers are candidate
discovery, not heldout confirmation or a compression result.

## 6. The multiplex causal map

Singleton similarity is only one relationship. Before selecting a mutation,
the compiler now builds a non-executable map with separate evidence planes:

- the existing singleton final-effect similarity;
- exact double-suppression interaction for every unordered pair;
- directed upstream-to-downstream generator response;
- prompt-cohort conditioning; and
- strict causal-order negative controls.

For 18 generators this is a complete 172-condition schedule per prompt batch:
one baseline, 18 singleton suppressions, and all 153 double suppressions.
Directed responses are captured during the singleton runs and add no model
forwards.

The map decides whether an apparent family is more consistent with parallel
compensation, serial lineage, conditional shared use, or an opposing balance.
Those motifs imply different mutation experiments. See
[`generator-causal-map.md`](generator-causal-map.md) for the exact metrics and
handoff.

The complete 20-prompt run produced all 153 edges in each of the three pair
planes and passed every causal-order negative control. It also changed the
interpretation of the original lead:

- layers 3-4 remain similar at the model output (`0.656025` cosine);
- their exact joint NLL second difference is strongly subadditive
  (`-4.185928`);
- suppressing layer 3 changes layer 4 by 80.2% of layer 4's ordinary output
  RMS; and
- that directed response points strongly against layer 4's usual output
  (`-0.622848` cosine).

That is a serial/fused-composition hypothesis, not evidence that one generator
can be deleted or that their outputs can be averaged. The pair's joint mute
raises NLL by `7.909090` per token.

The graph also contains interactions that singleton similarity misses.
Layers 3-10 have only `0.140828` output-effect cosine but the strongest
superadditive joint term, `+8.619503`. Conversely, layers 13-14 have the
largest singleton NLL Spearman (`0.882707`) but only `0.230381` output-effect
cosine and a positive joint term. Those conflicts are why the map retains
typed planes instead of reducing them to a single family score.

The prompt cohorts do not yet justify conditional routing. Their layer
importance profiles are nearly identical, two cohorts contain only one
prompt, and the other six contain only two to four prompts.

The strict ignored artifact is:

```text
.local-runs/google--gemma-3-270m/
  modal-generator-causal-map-dev-v1.json
```

Its scientific payload digest is
`1a25859340cd4772730fc631cdd7d7b859dda73c81d2447bed33c025d1e73afa`.
This is still adaptive open-development discovery rather than heldout
confirmation.

## 7. Transactional mutation

Mutation is a separate compiler phase that consumes a frozen causal map
and proposes one concrete executable change. It must never modify the baseline
artifact in place.

```text
frozen causal map
  -> immutable mutation proposal
  -> fit candidate
  -> select candidate
  -> install reversible overlay
  -> evaluate fresh guard
  -> atomically accept or roll back
  -> later physical lowering and pruning
```

This protocol is a design for the later mutation implementation. The repository
does not yet provide this mutation executor or mutation artifact. The first
proposal type and relationship must be frozen from the strict causal map, not
from singleton similarity alone. The original layer-local generators remain
the baseline and exact fallback.

### Proposal type: serial supermode with correction

The predeclared layers-3-to-4 topology probe composes an upstream transport
with a layer-4-specific head and an explicit correction branch. It is
installed behind a zero-initialized gate, so the exact original layers remain
the starting point and instant fallback.

This probe asks whether the measured directed relationship is locally
reconstructible. It must not claim compression unless a later physical
lowering actually reduces parameters, bytes, MACs, and measured backend cost
after the intermediate port and correction branch are counted.

### Proposal type: shared basis with layer adapters and scales

Two or more layer-local generators may share a central map while retaining
small endpoint-specific transforms:

\[
\widehat g_i(h_i)
=
D_i\,C(E_i h_i)+b_i.
\]

Here \(C\) is the shared core, while \(E_i\), \(D_i\), scales, and biases adapt
the layer-local input and output frames. Sharing is useful only if:

\[
\operatorname{params}(C)
+
\sum_i \operatorname{params}(E_i,D_i,b_i)
<
\sum_i \operatorname{params}(g_i),
\]

and the same accounting holds for executed MACs under the intended route.

A common core with full-width adapters can be larger than the independent
generators. Adapter, scale, bias, edge, and carried-state costs are never free.
The comparison baseline must be the best independent deployment lowering,
including any exact algebraic fusion available for each linear generator.
Otherwise ordinary per-generator matrix fusion could be misreported as a
cross-generator sharing gain.

The first dense-packing proposal freezes layers 12 and 15 from the completed
map. Their singleton effects co-vary (`0.756391` NLL Spearman), their joint NLL
term is nearly additive (`+0.016218`), their directed response ratio is
comparatively low (`0.411872`), and both are much less individually sensitive
than the early core. The candidate uses a shared low-rank basis with two
endpoint-specific heads and residual corrections; either original generator
can be restored independently.

This pair was selected on the current map, so all ranks and capacities must be
fit and selected on disjoint new data, then evaluated once on a fresh
family-disjoint guard. Layers 14 and 16 are the lower-impact structural
control for the same scaffold.

### Proposal type: merged generator with fan-out

A merge creates one generator node whose state feeds multiple layer-specific
decoders or graph edges. The proposal must define:

- where the shared state is produced;
- which causal inputs it may read;
- every consumer and its layer-local decoder;
- state lifetime and memory;
- fan-out edge parameters and MACs;
- what happens when a consumer requires information unavailable at the
  producer.

The graph must remain causal. A generator cannot read a future layer state to
serve an earlier consumer.

### Proposal type: conditional route

A routing proposal keeps multiple behaviors but executes only the path selected
for the current prompt or token context. It must count:

- router parameters and MACs;
- feature-extraction cost;
- carried routing state;
- default and uncertain routes;
- batching or divergence overhead;
- the cost of any fallback that executes more than one expert.

Average compute savings are insufficient. The report must include route
frequency, worst-route compute, routing errors, prompt-family coverage, and
fidelity by route.

### Proposal type: removal

A removal proposal replaces one generator contribution with zero and stores no
substitute. This is the strongest claim and requires more than a small
singleton effect:

- stable near-zero necessity on discovery data;
- joint-interaction checks against likely compensators;
- a matched exact-removal overlay;
- fresh-guard fidelity within a predeclared budget;
- positive net storage and compute savings after all remaining machinery.

Removal is deletion, not merging. It must be reported separately.

## 8. Split ownership

Mutation creates new learned choices, so its data roles must be explicit and
disjoint:

1. **Fingerprint discovery** measures singleton effects and proposes families.
2. **Mutation fit** estimates shared cores, adapters, scales, decoders, or
   routers.
3. **Mutation selection** chooses among predeclared capacities and proposal
   variants.
4. **Fresh guard** is first opened after the proposal, capacity, thresholds,
   and resource budget are frozen.

Any split used to discover a family, tune a similarity threshold, choose a
proposal type, fit mutation weights, select a rank, or revise a fidelity budget
is not a fresh guard.

The current open-development assessment may be used for exploratory
fingerprints only if it is relabeled as adaptive discovery for all later
mutation claims. A new disjoint guard is then required. Passing fit or
selection data is not evidence of mutation generalization.

## 9. Reversible overlay

Before physical compaction, every mutation runs as an overlay:

- the frozen full-refit generator catalog remains available;
- the candidate has separate authenticated state;
- installation occurs only inside a guarded execution scope;
- source and baseline generator identities are verified before installation;
- every affected layer executes the declared baseline or candidate exactly
  once;
- `finally` restoration runs after success or failure;
- source weights, baseline generator weights, and optimizer state are never
  changed;
- post-run fingerprints verify exact restoration.

The overlay must expose three distinct conditions:

1. frozen full-refit baseline;
2. matched structural control, such as exact removal;
3. proposed mutation.

This makes rollback immediate and makes failure scientifically useful. It also
keeps logical deployable accounting separate from experimental resident memory,
which temporarily includes the source, baseline generators, and mutation.

Jacobians or JVPs may help fit endpoint adapters, initialize transport maps, or
rank candidate mutations cheaply. They are predictors. Exact nonlinear overlay
replay on the fresh guard remains the acceptance authority.

## 10. Frozen resource and fidelity budgets

Every proposal must declare its acceptance budget before fresh-guard
evaluation.

### Resource budget

At minimum:

- baseline and candidate learned parameters;
- physically shared versus endpoint-specific parameters;
- logical stored-parameter savings;
- experimental resident overhead;
- generator, adapter, edge, decoder, and router MACs per token;
- bias additions;
- carried graph-state width and lifetime;
- average and worst-route compute;
- any packing, indexing, or codebook storage;
- whether the native or baseline weights are still resident.

Parameter or MAC reductions do not imply a latency improvement. Kernel latency,
memory traffic, and wall-clock speed require measurements on the intended
backend.

### Fidelity budget

At minimum:

- NLL and delta NLL against the frozen compiled baseline;
- baseline-to-candidate KL per token;
- top-1 agreement;
- downstream-task accuracy where available;
- worst prompt-family and route behavior;
- maximum sequence-level degradation;
- numerical identity controls for unchanged paths.

The budget need not require 100% downstream accuracy retention. It must be
predeclared, compared with an appropriate baseline such as quantization, and
reported without changing the threshold after seeing the guard.

## 11. Atomic acceptance and rollback

A mutation transaction has one of two terminal outcomes.

### Accept

Acceptance requires:

- exact proposal and source lineage;
- passing fit and selection checks;
- positive net resource savings under honest accounting;
- all fresh-guard fidelity thresholds satisfied;
- no source or baseline state mutation;
- a strict-loadable, versioned mutation artifact;
- a complete fallback binding to the previous graph.

Only after these checks may the compiler publish a new logical graph version.
Publication must be atomic: readers see either the complete previous graph or
the complete accepted graph, never a partially rewritten catalog.

### Roll back

Any lineage mismatch, execution error, nonfinite output, accounting drift,
resource-budget miss, or fidelity-budget miss rejects the proposal. The
overlay is removed, the exact previous graph is restored, and no physical state
is deleted. A rejected proposal remains an analysis result, not a partially
deployed mutation.

## 12. Physical lowering and pruning come last

An accepted overlay is still not compact storage. Physical lowering is a later
step that:

1. emits the shared or routed graph weights in their deployable layout;
2. packs dense cores, adapters, decoders, edges, and routing tables;
3. verifies that no duplicate baseline generator weights remain in the
   logical candidate;
4. removes superseded generator weights only after the new artifact is
   authenticated;
5. removes native MLP weights only where complete validated replacements
   exist;
6. reruns fidelity and resource accounting against the physically lowered
   artifact;
7. retains a separately versioned rollback artifact until deployment
   acceptance is complete.

Only this stage can make a physical parameter-storage claim. Fingerprinting,
singleton suppression, a family graph, or even a successful reversible overlay
does not by itself reduce the model file.

## 13. Evidence ladder

The resulting authority ladder is:

| Stage | What it establishes | What it does not establish |
|---|---|---|
| Singleton suppression | One generator's causal effect in the frozen compiled context | Redundancy or interchangeability |
| Causal fingerprint | Prompt-conditioned effect magnitude and shared-output direction | Executable sharing |
| Family hypothesis | A bounded mutation candidate | Fidelity or savings |
| Fitted mutation | A concrete shared, merged, routed, or removed executor | Generalization |
| Fresh-guard overlay | Fidelity and resource evidence under a frozen proposal | Physical compaction |
| Physical lowering | Deployable storage and compute structure | Backend latency without measurement |

Each stage consumes an immutable prior artifact and adds authority only for its
own claim. No stage may infer a later authorization from an earlier diagnostic.
