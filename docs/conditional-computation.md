# Fisher-need conditional computation

## Status

This milestone asks a narrower question than the existing modal executor:

> Can a causal router spend a different modal budget and mode subset on each
> token while preserving behavior better than one static mask with comparable
> average work?

The first implementation is deliberately a **native-output projection
oracle**. It computes a token's route from the input to a frozen transformer
layer, but it still executes that layer to obtain the activation being
projected. The selected route then reconstructs the activation with a hard,
token-specific modal budget.

That separation is important. The experiment can establish that conditional
budget-and-subset allocation is useful and predictable before another
generator is trained. It cannot yet establish that the transformer layer has
been replaced or that its parameters, FLOPs, latency, or energy use have
decreased.

The source-independent follow-up must make the same route decision and
generate the selected modal output directly from the layer or block input,
with zero calls to the native segment. Only that later executor can enter the
mixed compiled runtime.

Run the reference protocol with:

```bash
fisher-graph-conditional-rank
```

The weights-only result and JSON report are written under
`.local-runs/associative-recall/`, which is ignored by Git. The runner refuses
to overwrite an existing result.

## Why conditional budgets are different from static compression

A static projection asks every token to use the same fixed set \(S\) of modal
coordinates:

\[
\widehat a_t^{(S)}
=
\mu+
\left((a_t-\mu)E_S\right)D_S^{\mathsf T}.
\]

This is a strong constraint. One small mask must handle both ordinary tokens
and rare, highly sensitive tokens. Increasing its cardinality enough for the
rare cases then spends the same work everywhere.

Conditional routing chooses a route \(k_t\) for each valid token. Every route
owns both a predeclared budget \(b_k\) and an exact-cardinality mode mask
\(S_k\), where \(|S_k|=b_k\):

\[
\widehat a_t
=
\mu+
\left((a_t-\mu)E_{S_{k_t}}\right)D_{S_{k_t}}^{\mathsf T}.
\]

The route table is fitted on A-router only. Tokens are placed into
total-Fisher-need bins associated with the predeclared budgets. Within each
bin, the table averages need independently by mode and selects exactly the
budgeted number of highest-need modes with deterministic tie breaking.
Consequently, masks are not required to be prefixes or nested: a larger route
may omit a mode used by a smaller route.

In the music analogy, the current experiment decides both how large the
ensemble must be and which sections play for each kind of passage. The later
executor experiment asks whether those sections can generate the passage
without first hearing the native orchestra perform it.

## Fisher need

The router needs an offline teacher signal describing how consequential each
activation is. Let:

- \(a_t\) be the activation at token \(t\);
- \(\mu\) be the calibration mean;
- \(g_t=\partial\mathcal L/\partial a_t\) be the score gradient from an
  independently differentiated sequence;
- \(E\) and \(D\) be the encoder and decoder of a full-width modal codec.

The coordinate carried by mode \(j\) is

\[
c_{tj}=(a_t-\mu)E_j,
\]

and its first-order contribution to the score gradient is

\[
q_{tj}=g_tD_j.
\]

The diagonal first-order Fisher need is

\[
n_{tj}=(c_{tj}q_{tj})^2.
\]

For an orthonormal Fisher basis, \(E=D=V\). For a generalized Fisher codec,
the encoder and decoder are distinct dual bases and must not be silently
replaced by one orthonormal matrix.

The need fingerprint

\[
\mathbf n_t=(n_{t1},\ldots,n_{td})
\]

is used in two ways:

1. its total mass assigns the A-router token to a predeclared budget level;
   and
2. its per-mode pattern selects that level's route-specific exact-budget
   mask.

The current router receives neither \(\mathbf n_t\) nor \(g_t\) at inference.
Those values are teacher-only calibration signals. The inference feature is
the causal layer or block input at token \(t\), optionally transformed by a
frozen input codec whose cost is included in accounting.

This attribution is intentionally diagonal. Squaring each modal contribution
does not represent cross-mode cancellation, higher-order curvature, or the
true downstream loss after an intervention. It is a routing teacher, not a
behavioral acceptance metric. Native suffix replay and end-to-end task
metrics remain mandatory.

For a residual-separated block compiler, the same definition should be
applied to the native block delta

\[
\delta_t=h_{\mathrm{out},t}-h_{\mathrm{in},t}
\]

with a zero mean. That keeps the raw residual stream as an exact bypass and
aligns the teacher with the quantity a future graph must generate.

## Causal router

The reference router is intentionally small. It reads the current input
boundary state and emits one logit for each predeclared route:

\[
\ell_t=W x_t+b,\qquad
\operatorname{route}_t=\arg\max_k \ell_{tk}.
\]

The hard route selects exactly one budget-and-mask entry. There is no soft
mixture over every branch. The input \(x_t\) may already summarize earlier
tokens because it was produced by a causal decoder, but the router may not
read:

- the native layer or block output;
- score gradients or target labels;
- later token states;
- suffix logits; or
- a route chosen with validation or test outcomes.

Future-token perturbation tests must leave every earlier router decision
unchanged. Invalid or padded tokens do not receive routes, do not contribute
to route accounting, and retain the source boundary value.

## Hard grouped execution

Hard routing is useful only if execution follows the route. A reference
implementation groups valid tokens by selected budget:

```text
causal input states
        |
        v
hard budget-and-mask router
        |
        +---- route 0: budget 0, mask S0
        +---- route 1: budget 2, mask S1
        +---- route 2: budget 6, mask S2
        +---- route 3: budget 16, mask S3
        +---- full-budget fallback route
        |
        v
scatter reconstructed tokens back to source order
```

Each group evaluates only the encoder and decoder columns named by that
route's mask. The masks have exact cardinality but are not assumed to be
nested. A dense mask that computes every modal column and zeros some of them
afterward is numerically useful as a control, but it is not conditional
execution.

Even genuine grouped arithmetic does not imply a faster kernel. Routing,
index construction, gathers, small matrix multiplications, and scattering
can cost more than the skipped arithmetic, especially on short sequences.
The reference implementation therefore reports logical active work and a
complete analytic accounting. A latency claim requires a backend-specific
benchmark against the static and source baselines on the same device.

## Split protocol

Conditional routing has two different fitted objects: the modal basis and the
router. They must not share an evaluation split accidentally. The protocol
uses five roles:

### Calibration A-basis

Only A-basis may:

- estimate activation means or delta moments;
- accumulate the full-width Fisher matrix;
- choose or order modal directions; and
- establish basis-stability diagnostics.

### Calibration A-router

With the basis frozen, only A-router may:

- compute per-token Fisher-need fingerprints;
- fit need thresholds or route labels;
- normalize router input features;
- train router parameters; and
- choose a checkpoint under a fixed update schedule.

No basis is refitted after A-router labels are observed.

### Calibration B

Calibration B evaluates the frozen candidate and mandatory controls. It may
lock one predeclared policy or fail closed. It may not update the basis,
thresholds, router, budget schedule, or optimizer.

If B fails a behavior, causality, accounting, identity, or control gate,
validation is not tokenized or model-evaluated.

### Validation

Validation sees exactly the locked conditional policy and its predeclared
controls once. It is not used to choose a threshold, router checkpoint,
budget, basis, or reporting metric.

### Reserved test

The test fixture is parsed, counted, and hashed. It remains
model-unevaluated until the complete protocol—including the
source-independent executor and runtime gates—is frozen.

For paired tasks, both members of a context stay in the same role. For text
models, exact prompts and domain/template families are disjoint across roles.
Broad task forms may repeat only when that limitation is recorded explicitly.

## Mandatory controls

Conditional behavior is meaningful only relative to controls that isolate
where the gain comes from.

### Static-rank controls

Report:

- the leading Fisher prefix at or above the conditional policy's mean active
  budget;
- the smallest leading Fisher prefix passing the same behavior gates; and
- the maximum conditional budget as a quality ceiling.

These controls intentionally retain the basis's canonical global ordering;
they are not route-specific masks optimized with the conditional table. The
first comparison is coordinate-budget matched. It is not automatically MAC
matched because the conditional system also pays for its router and grouping.

### Position-only control

Fixed-format toy sequences make position a serious confound. Positional
embeddings may let an input-state router recover the slot even when logical
position is not passed explicitly.

An A-only position policy must therefore predict routes without token content.
Report within-position route entropy and the learned router's improvement over
that control. On a paired associative-recall task, route differences between
the two query variants are a useful content counterfactual.

If the position-only policy matches the learned router, the result supports
fixed position scheduling, not state-conditioned computation.

### Histogram-matched shuffle

Shuffle learned routes across valid tokens while preserving the exact route
histogram. This spends the same modal-coordinate budget without preserving
the association between state and compute. A gain over this control shows
that routing *which* token receives the larger budget matters.

### Teacher-route ceiling

The Fisher-need labels can route the native-output oracle directly. This is an
unavailable-at-inference ceiling that separates label quality from router
prediction error.

### Identity and native controls

Full-width projection must reproduce the authenticated boundary within its
declared numeric tolerance. A no-op suffix replay must reproduce native logits
and behavior. The source model remains frozen throughout every calibration and
evaluation role.

## Metrics and gates

Behavior reporting should include:

- task accuracy and paired-context accuracy where applicable;
- NLL and delta NLL;
- teacher-to-system KL;
- top-1 agreement;
- per-example and per-length tail metrics; and
- direct boundary error as a diagnostic that cannot select the candidate by
  itself.

Router reporting should include:

- route confusion matrix and macro-F1;
- under-routing and over-routing rates;
- mean absolute budget error;
- mean, p50, p90, and maximum active rank;
- exact route histogram and full-budget fallback rate;
- omitted-need regret relative to the teacher route;
- route distributions by logical position and token role; and
- the learned-versus-position-only and learned-versus-shuffle gaps.

Resource reporting must keep three quantities separate:

1. **active logical work:** the router plus only the selected modal columns;
2. **stored state:** the basis, router, every stored route or specialist, and
   provenance; and
3. **measured runtime:** same-device latency and memory after a real lowering.

Average active rank alone is not a FLOP count. Stored parameters do not shrink
merely because some are inactive on a token. Analytic MACs exclude costs only
when that exclusion is named, and they never stand in for a GPU or CPU timing
measurement.

## Verified result

The canonical layer-0 run strict-loaded successfully from
`.local-runs/associative-recall/layer-0-conditional-rank.pt`. The table reports
the existing 628-example validation role. “Matched static” is the ceiling of
the learned router's A-only mean active rank. “Viable static” is the smallest
leading-Fisher rank that passed the same gates on calibration B.

| Quantity | Native | Learned conditional | Static matched, rank 12 | Viable static, rank 16 | Position-only | Shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Task accuracy | 1.0000 | 1.0000 | 0.7229 | 1.0000 | 1.0000 | 0.5350 |
| Paired accuracy | 1.0000 | 1.0000 | 0.5382 | 1.0000 | 1.0000 | 0.2834 |
| NLL | 0.049155 | 0.075492 | 1.065130 | 0.082039 | 0.076428 | 2.286070 |
| Mean active rank | n/a | 11.535 | 12 | 16 | 10.750 | 11.535 |
| Full-width fallback rate | n/a | 18.63% | 0% | 0% | 12.50% | 18.63% |
| Analytic projection + router MACs | n/a | 4,352,000 | 3,858,432 | 5,144,576 | 3,456,512 | 3,708,928 projection only |

Teacher KL was not recorded in this toy rung and is not inferred from NLL.
The learned conditional path used 27.9% fewer active modes than the smallest
B-viable static rank. After adding its 32-by-4 linear router, its ideal
route-plus-projection count was 15.4% below that rank-16 projection. These are
only modal projection arithmetic figures: the native layer still ran, and no
model-level FLOP, parameter, latency, or energy reduction is claimed.

The learned, position-only, rank-16, and full-rank identity paths passed the
accuracy, paired-accuracy, and maximum \(+0.05\) NLL gates. The rank-12
average-budget control and the histogram-matched shuffle failed. Receiving
the right route therefore mattered. However, the learned router improved NLL
over position-only by only 0.000936, below the preregistered 0.005 state
advantage gate, and paired counterfactuals changed no route. This supports
hard position-conditioned specialist allocation on this fixed-format task;
it does **not** support content-conditioned routing yet.

The checkpoint SHA-256 is
`3216a20395d274f4f820ef79578f084a0268e22e4108a451a45af25327691cc6`.
Input/context hash pairs for A-basis, A-router, B, validation, and reserved
test are:

| Role | Input IDs SHA-256 | Context IDs SHA-256 |
|---|---|---|
| A-basis | `a5a0c28995e1d3aaa99fb7ca6916b771628a12b64a7a9847d1b8e30c8117b931` | `b3ede54adc286421c182bc73a7a73c3556ddc97a427a1430c7a6423314b8ad19` |
| A-router | `739315a890589fa6c12ebcf9efbf91a6c71126e40e3a775cb72a74d1bd6ffe54` | `0077f0b9a8a23c6b83f9494b3cebe3f18889c1439e7bd61f2636e24e0d014810` |
| B | `78fcfb10adfccdbf14cfe621bb90cb9bd32116cb87a5228e5b42e58a818f73c9` | `326e19de06f15c6c50d39eec16cc2f18221108bed1019b9955090ad4a4393dd7` |
| Validation | `d76f7720aa603e48f0a5f95ca33d2cf8385ddddc78a0850416c797b286e0072c` | `cc6a1b3b152ca650eba3463c127df5049f1feab6ada2b8097a066dd1b32da538` |
| Reserved test | `b41093330185da0b02443d261826ac8affdde604d9e7bd261c13e1388749d0d5` | `5918a023e2177254a907924ffd4121cc609fb3f7223bf5dc62be0899350febaf` |

The route budgets were fixed at \((0,6,16,32)\). A-router examples were
partitioned into equal-frequency total-first-order-need bins, and each bin
received its own exact-cardinality Fisher-need mask. The basis saw A-basis
only; it did not see A-router, B, validation, or test. The validation fixture
and checkpoint had been inspected in earlier exploratory work, so this result
is explicitly exploratory rather than confirmatory. The reserved test role
was hashed but never model-evaluated.

The authenticated scientific payload digest is
`77e619116fe081d89d297cdd5af52184fada36c248c372e96d75ffef835abda8`.
The ignored tensor and JSON files have SHA-256 values
`e1b62e2280b5f2c80e0541b652886b36173361244ec955cc606543770d01d03e`
and
`18a9f409c143d75ebd972d1c036bfceb317902ebb5b5d7ee875f641e27bc98fa`.

## Claim boundary

A positive native-output oracle supports the following statement:

> Under the tested checkpoint, basis, split, and route policy, modal
> sensitivity is nonuniform across tokens and sufficiently predictable from a
> causal input state that hard conditional budget-and-subset allocation
> preserves behavior better than the declared static and shuffled controls.

It does not establish:

- a source-layer replacement;
- parameter compression;
- source-block FLOP reduction;
- kernel or wall-clock speedup;
- a source-independent specialist generator;
- stability of the learned specialist masks on another split or checkpoint;
- generalization to variable-length text;
- generalization to another checkpoint or model family; or
- that Fisher need is the unique or optimal routing teacher.

A source-independent graph can claim analytic compute reduction only after it
generates the selected modal delta without running the native segment and
passes the same local, behavioral, causal, and control gates. A speed claim
requires the additional backend benchmark.

## From specialist masks to a specialist generator bank

The current route table already answers both “how many coordinates?” and
“which coordinates?” Its A-only need bins own route-specific specialist
masks. What it does not do is generate those coordinates: the projection
oracle still reads them from the native activation.

A minimal future executor has:

1. a common causal trunk used by every token;
2. a hard router using only the authenticated input boundary;
3. one selected generator corresponding to the route's budget and mode mask;
4. an explicit maximum-capacity or source fallback policy; and
5. an output decoder inside a behaviorally validated span.

This generator bank may use more stored parameters than one static graph while
executing fewer active parameters per token. Storage compression and compute
compression must therefore remain separate claims. Its route masks may reuse
the representation-oracle table, but the generator outputs must be learned
from A-only teacher boundaries and validated without a native-segment call.

The existing soft gated causal executor is not this runtime: it evaluates all
experts on every legal edge and mixes their outputs. It remains useful as a
state-conditioned transport reference, but soft expert probabilities alone do
not skip expert computation.

## Gemma variable-sequence next experiment

The first larger-model rung should remain an oracle until the routing
hypothesis passes on wholly fresh data:

1. strict-bind the model revision, layer range, codec/span predecessors, and
   all prior prompt hashes;
2. create new domain/template-family-disjoint A-basis, A-router, B,
   validation, and reserved-test roles;
3. fit the block-delta basis and Fisher-need teacher on A only;
4. route from the block input with a length-independent feature projection;
5. use hard grouped budgets over valid tokens;
6. compare with static-rank, position-only, length-only, and
   histogram-matched shuffled policies;
7. report results by length, logical position, truncation status, and prompt
   family; and
8. leave test hash-only unless the later source-independent executor also
   passes.

The sequence contract must cover right, left, and sparse padding; offset and
gapped logical positions; multiple prefill lengths; and prefix invariance
under appended future tokens. Position-only and length-only controls are
especially important because a router that merely recognizes prompt length
does not establish content-dependent compute modes.

If the native-output oracle passes, train a hard-routed block-delta graph on a
new A role. Its segmented student path must record zero calls to the native
layers, implement the compiler's `CompiledSegmentExecutor` boundary, and
publish exact route-aware storage and MAC accounting. Only after it passes B
and one locked validation evaluation should it be lowered to an MLX or other
grouped kernel and benchmarked.
