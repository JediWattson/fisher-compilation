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

The latest full-span rung remains a representation oracle. Its causal-prefix
router and complete accounting are implemented below, but its calibration-B
joint gate failed against a static rank-14 comparator. Validation, test, and
graph fitting therefore remained untouched.

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

The variable-format protocol fits the route table on A-mask only. Tokens are
placed into total-Fisher-need bins associated with the predeclared budgets.
Within each bin, the table averages need independently by mode and selects
exactly the budgeted number of highest-need modes with deterministic tie
breaking. The frozen teacher then labels A-router, where only the causal
classifier and metadata controls are fitted. Consequently, masks are not
required to be prefixes or nested: a larger route may omit a mode used by a
smaller route.

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

1. its total mass assigns an A-mask token to a predeclared budget level; and
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

Conditional routing has three distinct calibrated objects: the modal basis,
the need teacher and masks, and the inference router. They must not share an
evaluation split accidentally. The variable-format protocol uses six roles:

### Calibration A-basis

Only A-basis may:

- estimate activation means or delta moments;
- accumulate the full-width Fisher matrix;
- choose or order modal directions; and
- establish basis-stability diagnostics.

### Calibration A-mask

With the basis frozen, only A-mask may:

- compute per-token Fisher-need fingerprints;
- fit total-need thresholds;
- assign route labels; and
- fit the exact-cardinality mask owned by each route.

### Calibration A-router

With the basis, teacher, and masks frozen, only A-router may:

- receive route labels from the frozen teacher;
- normalize router input features;
- train router parameters; and
- fit metadata controls.

No basis, threshold, or mask is refitted after A-router labels are observed.

### Calibration B

Calibration B evaluates the frozen candidate and mandatory controls. It may
lock one predeclared policy or fail closed. It may not update the basis,
thresholds, router, budget schedule, or optimizer.

If B fails a behavior, causality, accounting, identity, or control gate, the
conditional compiler candidate is not applied to validation.

### Validation

Validation sees exactly the locked conditional policy and its predeclared
controls once. It is not used to choose a threshold, router checkpoint,
budget, basis, or reporting metric.

### Reserved test

The compiler test fixture is reconstructed, counted, and hashed. The
conditional candidate is not applied to it until the complete
protocol—including the source-independent executor and runtime gates—is
frozen. This is separate from any earlier native-model evaluation performed
while training the source checkpoint.

For paired tasks, both members of a context stay in the same role. For text
models, exact prompts and domain/template families are disjoint across roles.
Broad task forms may repeat only when that limitation is recorded explicitly.

## Mandatory controls

Conditional behavior is meaningful only relative to controls that isolate
where the gain comes from.

### Static-rank controls

Report both canonical Fisher-prefix masks and global-need-ranked masks:

- the leading Fisher prefix at or above the conditional policy's mean active
  budget;
- the smallest mask of either ordering passing the same behavior gates; and
- the maximum conditional budget as a quality ceiling.

Neither ordering is route-specific. Fisher prefixes retain the basis's
canonical ordering; global-need masks order modes by their A-mask mean need.
The first comparison is coordinate-budget matched. It is not automatically
MAC matched because the conditional system also pays for its router and
grouping.

### Metadata-only controls

Fixed-format toy sequences make position a serious confound. Positional
embeddings may let an input-state router recover the slot even when logical
position is not passed explicitly.

Fit A-router-only hierarchical controls for position, length,
position-plus-length, token-role-plus-position-plus-length, and
token-ID-plus-position-plus-length. Report the learned router's improvement
over the strongest of them. On a paired associative-recall task, route
differences between the two query variants are also a useful content
counterfactual.

If a metadata policy matches the learned router, the result supports a cheap
metadata schedule, not richer state-conditioned computation.

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

- route confusion matrix and macro recall;
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

## Fixed-format verified result

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

## Variable-format calibration-B result

The next experiment removes the fixed-position shortcut. A fresh three-layer
toy model was trained on eight controlled render layouts, lengths 8 through
12, both pair orders, and both possible queries. It reached 100% answer,
paired-context, layout, length, query, and pair-order accuracy on its native
train, validation, and test splits. The conditional compiler then targeted
the middle block delta and allocated disjoint 24-context A-basis, A-mask,
A-router, and calibration-B roles. Its validation candidate was not evaluated
after B failed, and the compiler test role remained hash-only.

Calibration B contained 768 examples and 7,872 valid token rows. Under the
canonical budgets \((0,8,16,24,32)\), the native model had NLL 0.049339 and
100% accuracy. The unavailable-at-inference Fisher teacher passed every
behavior gate at NLL 0.049327 and 100% answer and paired accuracy. This is the
important positive result: the route-specific masks can preserve the task.

The causal linear router did not pass:

| Quantity | Teacher | Learned router | Strongest metadata control | Smallest passing static |
|---|---:|---:|---:|---:|
| NLL | 0.049327 | 0.071922 | 0.049586 | 0.074900 |
| Answer accuracy | 100% | 99.2188% | 100% | 100% |
| Paired accuracy | 100% | 79.1667% | 100% | 100% |
| Mean active rank | route-dependent | 17.0528 | route-dependent | 9 |
| Full-width fallback | route-dependent | 25.889% | route-dependent | 0% |
| Router + projection MAC ratio vs static rank 9 | n/a | 2.17254x | n/a | 1.0x |

The strongest metadata control was
token-role-plus-position-plus-length. It beat the learned router by 0.022337
NLL; equivalently, the learned router's advantage was -0.022337 with a 95%
semantic-context bootstrap interval of [-0.043644, -0.004523]. The learned
policy did
beat a position-and-length-stratified route shuffle, so arbitrary assignment
is not enough, but that does not rescue the failed behavior and compute gates.

Two post-hoc B-only schedules expose the tradeoff more sharply:

- Equal-frequency budgets \((0,2,4,8,32)\) produced genuine content signal
  over the token-ID metadata control: +0.078654 NLL advantage with 95%
  semantic-context interval [0.013632, 0.149695]. But the learned policy
  reached only 98.1771%
  accuracy and 70.8333% paired accuracy, averaged 10.9029 modes, and cost
  1.48922x the passing static rank-9 projection.
- Quantiles \((0.25,0.50,0.75,0.95)\) with budgets
  \((0,2,4,12,32)\) kept the teacher at 100%, reduced the learned mean rank to
  4.2383, eliminated full-width fallback, and reached 0.74870x the static
  analytic MAC count. The learned policy, however, fell to NLL 0.257209,
  93.75% accuracy, and 45.8333% paired accuracy. A conservative posterior
  decision restored 100% behavior at its 70th-percentile rule, but raised the
  mean rank to 11.2602 and cost to 1.52891x static.

These schedules are exploratory diagnostics, not additional selected
candidates. Together they show that the quality gate and the compute gate are
individually reachable, but the current router cannot reach them jointly.
Calibration B therefore failed closed:
`model_level_eligible=false`; the compiler did not fit a model-level
generator or evaluate its held-out candidate.

Reproduce the canonical run with:

```bash
fisher-graph-variable-associative-train \
  --output .local-runs/variable-associative/checkpoint.pt
fisher-graph-variable-conditional \
  --checkpoint .local-runs/variable-associative/checkpoint.pt \
  --output .local-runs/variable-associative/layer-1-variable-conditional.pt
```

The post-hoc teacher/compute frontier cited above used:

```bash
fisher-graph-variable-conditional \
  --checkpoint .local-runs/variable-associative/checkpoint.pt \
  --route-budgets 0,2,4,12,32 \
  --route-quantiles 0.25,0.50,0.75,0.95 \
  --output \
    .local-runs/variable-associative/layer-1-variable-conditional-skewed-95-r12.pt
```

That second command is explicitly B-tuned and must not be treated as a fresh
candidate.

## Full-span query-sparse calibration-B result

The next rung targeted the complete three-block span,
`layer.0.input -> layer.2.output`, rather than compiling one block in
isolation. This changes the causal routing problem. At the first boundary, the
current answer-marker row is only its token plus absolute-position embedding;
it does not yet contain the earlier key/value content. The experiment therefore
gave the router every key row through the answer position while requesting a
route only at the supervised answer row.

This distinction also prevents a misleading optimization claim. Of 7,872
attention-valid rows in calibration B, only 768 answer rows have direct
task-loss demand. The 7,104 prefix rows remain causal inputs. Skipping the
other output rows is **output-demand sparsity**, not evidence that Fisher
routing discovered zero-cost modes.

Five disjoint 24-context train roles were used:

1. `basis_a` fitted the answer-row Fisher basis;
2. `mask_a` fitted frozen total-need thresholds and route masks;
3. `policy_a` selected one of seven preregistered positive-rank schedules;
4. `router_a` used 16 contexts for fitting and eight for router and
   metadata-control selection, then refit the frozen choices on all 24; and
5. `calibration_b` was touched after those choices were frozen.

The selected schedule used budgets \((12,16,24,32)\) and total-need quantiles
\((0.50,0.80,0.95)\). The causal router selected on A was the cheapest
prefix-sum feature map (one zero-decay channel), ridge 0.1, no class
reweighting, and an argmax decision.

| Quantity | Fisher teacher | Causal-prefix router | Pointwise embedding ablation | Smallest passing static |
|---|---:|---:|---:|---:|
| NLL | 0.074960 | 0.089938 | 0.092913 | 0.090521 |
| Delta NLL | +0.025854 | +0.040832 | +0.043807 | +0.041415 |
| p90 absolute delta NLL | 0.072771 | **0.101778** | 0.119374 | 0.096601 |
| Answer / paired / minimum-stratum accuracy | 100% | 100% | 100% | 100% |
| Mean active rank | 17.1563 | 14.3021 | 15.5000 | 14 |
| Router + ideal selective answer-projection MACs | n/a | 1,028,608 | 860,160 | 688,128 |
| Work ratio to static | n/a | **1.49479x** | 1.25x | 1.0x |

The teacher passes behavior, which again shows that the route masks can retain
the answer. It is not a compute win: its mean rank is already larger than the
passing static rank. The causal router preserves top-1 behavior but misses the
preregistered p90 gate by 0.001778. Its route classification accuracy is
39.71%, macro recall is 27.05%, and its average rank is also slightly larger
than static.

There is one real but insufficient signal. The metadata-control identity was
frozen on the `router_a` holdout, where position/length won the deterministic
tie-break. On untouched calibration B, the causal router beats that A-selected
control by 0.015035 NLL with a semantic-context bootstrap interval of
[0.005452, 0.026494]. But its advantage over the pointwise embedding ablation
is only 0.002975 with interval [-0.006694, 0.013639].
Position/length-stratified shuffles likewise produce intervals crossing zero.
The evidence therefore does not isolate a reliable causal-prefix routing
advantage, and it does not jointly satisfy fidelity and compute.

The full accounting makes the distinction concrete:

- native three-block plus vocabulary-head work is 208,152,576 **logical
  valid-row/allowed-edge** MACs. This is not a claim about GPU-issued work;
- a shape-padded, allowed-edge estimate is 245,071,872 MACs. Dense masked
  attention may execute future-edge products too, so literal backend work and
  latency require kernel measurement;
- truncating suffix rows and reading only the answer reduces the fair logical
  native baseline to 182,292,480 MACs;
- the representation oracle still runs the native span, so adding its router
  and ideal selective projection raises logical work to 209,181,184 MACs, or
  1.00494x the logical native reference. The dense diagnostic projection
  actually used by this analysis makes that 210,051,072 MACs;
- router plus ideal selective answer projection alone is only 0.564% of the
  logical prefix-native total, but it omits the residual generator and is not
  a complete executor.

Untrained shared-graph envelopes with hidden width 32 range from 5.19% of
prefix-native complete MACs at one state channel to 36.14% at eight channels,
with 4,517 to 18,860 runtime scalars. Those numbers show architectural
headroom, not achieved compression. No graph was fitted, and the observed
conditional policy is slightly *more* expensive than a rank-14 policy on the
same hypothetical trunk. With the static comparator's router removed and its
head/decoder structurally pruned to rank 14, conditional-to-static complete
MAC ratios are 1.01212x, 1.00338x, and 1.00172x for one, four, and eight state
channels respectively.

This CLI is a gate-only experiment. Even a future pass would mark model-level
fitting eligible rather than fitting a graph inside the calibration command;
that source-independent fit and validation must remain a separate step.

The locked run is reproduced with:

```bash
fisher-graph-variable-full-span \
  --checkpoint .local-runs/variable-associative/checkpoint.pt \
  --output .local-runs/variable-associative/full-span-conditional.pt
```

The ignored `.pt` artifact contains the Fisher basis, teacher, route table,
causal and pointwise routers, controls, hashes, and complete report. It
contains neither source-model weights nor a compiled graph.

## Static full-span generator: the second relational stage matters

The failed conditional gate does not rule out the simpler static branch. Its
fresh calibration-B curve identified rank 14 as the smallest
behavior-preserving native-output span. The follow-up freezes that Fisher
decoder and trains a source-independent graph to generate its answer-row
coordinates directly from the incoming `layer.0.input` prefix. It never uses
the native `layer.2.output` at inference.

The graph is a small causal transformer rather than the earlier one-stage
exponential-state generator. This distinction is structural for associative
recall. One attention stage can bring the queried key to the answer marker or
attach values to their preceding keys, but it cannot reliably perform both
operations. Two stages allow:

1. pair rows to build key/value associations; then
2. the answer row to select the association matching the earlier query key.

The artifact protocol excludes every exact semantic-context hash used by the
predecessor and keeps all 32 renderings of a mapping together. It hash-ranks
the remaining train contexts into:

| Role | Contexts | Purpose |
|---|---:|---|
| `graph_fit_a` | 512 | update graph weights |
| `graph_stop_a` | 128 | choose a checkpoint within each seed |
| `graph_select_a` | 128 | require architecture stability and choose the graph |
| `calibration_c` | 128 | nominal frozen evaluation within this run |
| reserve | 238 | unconsumed by this artifacted run |

A post-run provenance audit found an earlier, unartifacted architecture probe
over train-context rows 120–887: 512 fit, 128 stop, and 128 selection contexts.
That probe informed the two-layer architecture and training recipe. It
overlaps 355 current fit contexts, 84 stop contexts, 83 selection contexts,
and—critically—83 of 128 nominal calibration-C contexts. Calibration C is
therefore exploratory, not clean confirmation. Of the 238 contexts called
reserve by this run, 163 also overlap the earlier probe; only 75 remain clean
relative to the predecessor, prototype, and artifacted experiment. The
artifact now records all of these exact hashes and fails a dedicated
no-prior-development-overlap gate.

Three fixed seeds are run for each declared architecture. An architecture must
pass the strong `graph_select_a` gates in at least two seeds. The selection
order is ideal complete MACs, stored coefficients, NLL, and name; its deployed
seed is the conservative median-NLL strong seed. Within the artifacted run,
calibration C remains uncollected until all of these choices, the checkpoint,
rank, coordinate scale, and decoder are frozen. That sequencing prevents
within-run leakage but does not erase the earlier prototype overlap above.

The locked run is:

```bash
fisher-graph-variable-static-full-span \
  --checkpoint .local-runs/variable-associative/checkpoint.pt \
  --predecessor .local-runs/variable-associative/full-span-conditional.pt \
  --output \
    .local-runs/variable-associative/static-transformer-full-span.pt
```

Selection A cleanly distinguishes depth from width:

| Candidate | Strong seeds | Selected-seed delta NLL | Answer accuracy | Ideal complete MAC ratio | Storage ratio |
|---|---:|---:|---:|---:|---:|
| one layer, width 32 | 0 / 3 | +0.779681 | 48.95% | 0.379171x | 0.412531x |
| two layers, width 16 | 1 / 3 | +0.023574 | 99.58% | 0.198652x | 0.222144x |
| two layers, width 32 | **2 / 3** | **+0.008856** | **100%** | 0.711426x | 0.745865x |

The one-layer control remains near chance even though it has the same hidden
width as the winner. The smallest two-layer graph can solve the task in one
seed but is not robust. The result therefore supports the multi-hop
interpretation rather than a generic “more parameters” explanation.

The frozen two-layer width-32 executor then produced:

| Nominal calibration-C quantity (exploratory) | Result | Strong gate |
|---|---:|---:|
| Native NLL | 0.049715 | reference |
| Replacement NLL | 0.058821 | — |
| Delta NLL | +0.009105 | at most +0.010000 |
| Answer / paired / minimum-stratum accuracy | 100% / 100% / 100% | all 100% |
| Native top-1 agreement | 100% | 100% |
| Native-teacher KL | 0.005243 | at most 0.010000 |
| p90 absolute per-example delta NLL | 0.024399 | at most 0.025000 |
| Context-bootstrap 95% interval for mean degradation | [0.007926, **0.010323**] | upper at most 0.010000 |
| Native source-block calls in replacement | 0 / 0 / 0 | all zero |

Every point-estimate behavior gate passed, as did exact artifact replay,
future-position invariance, zero-fallback ownership, and Fisher-span
membership. Two joint gates fail: the context-bootstrap upper bound
(`0.010323 > 0.010000`) and the post-run provenance requirement that
calibration contain no prior development contexts. The numeric miss is narrow;
the independence failure is categorical. Official validation and test were
not evaluated.

The resource result is real but must be stated at the correct level:

- the executor has 18,670 learned parameters plus 448 fixed decoder
  coefficients, 19,118 total versus 25,632 parameters across the three source
  blocks, a `0.745865x` storage ratio;
- ideal valid-prefix graph work plus the shared vocabulary head is 691,666,944
  matrix MACs across calibration C versus 972,226,560 native, a `0.711426x`
  ratio;
- the current reference implementation executes dense matrix shapes through
  the batch's longest prefix. Its estimate is 937,951,232 complete MACs, or
  `0.964746x` native—not the ideal 0.711426x; and
- normalization, GELU, softmax, masking, additions, gathers, memory traffic,
  kernel launches, and wall-clock latency are not included.

This is the first tested whole-span path in this branch that both owns graph
weights and records zero native-layer calls while passing all point-estimate
fidelity gates. It is stronger than the earlier representation oracle, whose
same rank-14 projection on calibration C had delta NLL +0.041162 and p90
0.102972. The learned graph can optimize useful coordinates *within* the
fixed span instead of merely copying the least-squares native projection.

It is not yet a validation-backed compression result. Calibration C is
consumed development evidence and cannot be used to claim confirmation. A
follow-up must freeze its revision before touching data, and only the 75
hash-audited clean train contexts—not all 238 nominal reserve contexts—remain
available without constructing a fresh dataset. Consuming that small reserve
is an irreversible protocol decision.

## Claim boundary

The positive statement below applies to the fixed-format result:

> Under the tested checkpoint, basis, split, and route policy, modal
> sensitivity is nonuniform across tokens and sufficiently predictable from a
> causal input state that hard conditional budget-and-subset allocation
> preserves behavior better than the declared static and shuffled controls.

The variable-format result establishes a narrower pair of facts: a
Fisher-informed teacher can choose behavior-preserving conditional masks, and
the input state contains some route information beyond declared metadata under
one aggressive schedule. It does not establish a deployable policy with both
fidelity and compute savings.

The full-span routing result adds that query-sparse execution exposes a large
possible whole-model envelope, but the tested Fisher policy does not allocate
less work than a static rank-14 answer projection. The static-generator
follow-up turns part of that envelope into a real source-independent graph:
two causal stages, unlike one, generated a high-fidelity rank-14 answer delta
with lower parameter and ideal-MAC counts. Its frozen calibration point
passed every strong point-estimate gate but missed the context-bootstrap gate
by 0.000323 and overlapped an earlier development probe, so it remains
exploratory pre-validation evidence.

It does not establish:

- a validation-backed source-layer replacement;
- end-to-end model-file compression;
- measured source-block FLOP or energy reduction;
- kernel or wall-clock speedup;
- a conditional source-independent specialist generator;
- stability of the learned specialist masks on another split or checkpoint;
- generalization from controlled variable layouts to natural variable-length
  text;
- generalization to another checkpoint or model family; or
- that Fisher need is the unique or optimal routing teacher.

The static graph now generates the selected modal delta without running the
native segment and has lower structural coefficient and ideal-MAC counts. A
compression claim still requires a fresh confirmatory pass and locked
validation; a speed claim additionally requires a sparse lowered kernel and
same-device benchmark.

## From specialist masks to a specialist generator bank

The current route table already answers both “how many coordinates?” and
“which coordinates?” Its A-only need bins own route-specific specialist
masks. What it does not do is generate those coordinates: the projection
oracle still reads them from the native activation.

The repository now contains
`ConditionalCausalModalBlockExecutor`, a source-free execution interface with:

1. a common causal trunk used by every token;
2. a hard router using either the authenticated input boundary or the shared
   causal hidden state, with the latter enabling prefix-conditioned full-span
   routes without a second causal trunk;
3. one selected generator corresponding to the route's budget and mode mask;
4. exact residual bypass for rank-zero and invalid rows;
5. route-specific output-head and decoder column gathers; and
6. the compiler's `CompiledSegmentExecutor` boundary plus per-call work
   accounting.

It is an implementation scaffold, not a fitted result. Because both the
middle-block and full-span B gates failed, no trunk or specialist head was
trained against the source block, and no model-level fidelity or resource
claim follows from the interface tests. Its source-free status is
executor-local; a mixed runtime must separately prove that dispatch never
selected a native fallback.

This generator bank may use more stored parameters than one static graph while
executing fewer active parameters per token. Storage compression and compute
compression must therefore remain separate claims. Its route masks may reuse
the representation-oracle table, but the generator outputs must be learned
from A-only teacher boundaries and validated without a native-segment call.

The existing soft gated causal executor is not this runtime: it evaluates all
experts on every legal edge and mixes their outputs. It remains useful as a
state-conditioned transport reference, but soft expert probabilities alone do
not skip expert computation.

## Gemma variable-sequence escalation remains blocked

The static graph is materially stronger than the earlier variable-format and
full-span routing gates, but its nominal calibration was exploratory, its
bootstrap also failed, and official validation remained untouched. It does
not yet authorize a new conditional Gemma fit. After a clean toy confirmation
and locked validation, the first larger-model rung should:

1. strict-bind the model revision, layer range, codec/span predecessors, and
   all prior prompt hashes;
2. create new domain/template-family-disjoint A-basis, A-mask, A-router, B,
   validation, and reserved-test roles;
3. fit the block-delta basis, then freeze the A-mask Fisher-need teacher and
   masks before fitting A-router;
4. route from a causal summary available at the compiled boundary or from the
   shared graph hidden state, with a pointwise-boundary ablation;
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
