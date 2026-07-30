# Residual-guided progressive compilation

The progressive compiler treats an approximate modal graph as iteration zero,
not as an all-or-nothing replacement. It repeatedly maps the remaining
source-to-candidate residual, proposes a bounded structural repair, measures
the result, and advances only the accepted candidate head. Once fidelity
passes, the same controller changes from repair to compaction.

The implementation is split between:

- `fisher_graph.compiler.progressive`, the model-independent state machine;
- `fisher_graph.gemma3_l3_l4_progressive_compilation`, the prompt-blind
  binding for the current Gemma L3/L4 rank-64 seed; and
- future model-specific workers which will calculate residual maps, lower
  mutation proposals, and execute source-authoritative measurements.

## Development and assessment boundaries

Repeated development uses three pairwise family-disjoint Calibration-A
roles:

| Role | Reusable? | Purpose |
|---|---:|---|
| `calibration_a_fit` | yes | Fisher/JVP residual mapping and parameter fitting |
| `calibration_a_selection` | yes | choose among preregistered repair or compaction proposals |
| `calibration_a_guard` | one callback per campaign | final veto after the complete challenger lineage is frozen |

The selection split is intentionally named as such. Repeatedly selecting
against it makes it development data, even though it is family-disjoint from
fit. The final A guard is supplied only after no more graph mutations can
occur, and its evaluation binds the already-frozen challenger receipt. A
failed guard terminates the campaign; the runner does not fall through to the
second-best selection candidate. This generic in-process callback is not yet
a cross-run one-shot authority. The Gemma worker must wrap it in a claim-first
ledger keyed by the protocol and A-guard manifest.

Calibration B is not a fourth callback. Its frozen manifest SHA is registered
in the progressive protocol only so every A role can reject it. A passing A
handoff contains no B loader, prompt, example identity, manifest, observation,
or evaluation result.

```text
Calibration-A fit
    ↓
residual map
    ↓
repair proposals ──→ A-selection evaluation ──→ accepted active head
    ↑                                                │
    └──────────────── repeat while fidelity fails ───┘

accepted head passes fidelity
    ↓
compaction proposals ──→ A-selection evaluation ──→ smaller active head
    ↑                                                   │
    └──────────────── repeat while Pareto-positive ─────┘

freeze complete lineage
    ↓
one A-guard veto
    ↓
development-only frozen handoff
    ↓
separate candidate-bound Calibration-B protocol
```

## Acceptance policy

The repair phase uses a normalized worst-axis fidelity burden. The protected
axes are:

- absolute NLL change per token;
- source-to-candidate KL per token;
- aggregate and adverse-tail top-1 agreement;
- adverse-tail absolute NLL change;
- operator NRMSE;
- pooled and worst-family boundary error/cosine;
- valid-target coverage and minimum family signal;
- pooled and worst-family full-width projection error/cosine;
- projection-oracle NLL/KL/top-1 and adverse tails; and
- exact-carrier-oracle NLL/KL/top-1 and adverse tails.

A repair must reduce the worst normalized burden by the frozen minimum and
cannot move any other axis beyond its allowed regression envelope. This
permits a useful trade between fidelity dimensions without allowing one
metric to hide a catastrophic failure in another.

After every fidelity axis passes, a compaction is eligible only when:

1. every fidelity gate remains passed;
2. learned parameters, runtime parameter bytes, and logical MACs/token are
   all non-increasing; and
3. at least one of those resource axes strictly decreases.

The deterministic repair tie-break is fidelity burden, MACs, parameters,
bytes, then candidate ID. The compaction tie-break is MACs, parameters, bytes,
fidelity burden, then candidate ID.

The result retains every proposal, built candidate, and scalar selection
evaluation, including dominated and rejected points. This is the raw
rate-distortion archive; the accepted lineage is a view over it, not a
replacement for it.

## Full resource accounting

Every proposal and built candidate carries the same exact resource footprint.
Each resource axis is divided into:

- compiled graph work;
- retained source-model work; and
- support work such as carrier transforms, routing, lookup, or normalization.

The controller charges all three. Every receipt also binds the candidate
execution, accounting artifact, parameter/compute scopes, runtime and dtype,
and sequence scope. A source island is therefore a legitimate temporary
repair, but it is never free. Hard budgets limit both total cost and the
retained-source fraction, and incomparable scopes fail closed.
The protocol pins the seed's complete resource-footprint receipt, so changing
even one baseline cost or scope creates a different campaign rather than a
quietly revised iteration zero.

Resource receipts also have a `cost_complete` bit plus canonical incomplete
reasons. A candidate with omitted router operations, fallback execution, or
another unknown dimension is rejected before it is built or evaluated. This
does not make caller-supplied accounting authoritative; each model-specific
builder still needs to recompute these numbers and the accounting-artifact
hash from its immutable executable.

## Immutable lineage

The seed artifact, execution, runtime-binding, and resource-footprint hashes
are frozen into the protocol and checked against the seed candidate. Each
residual map binds the active candidate receipt and fit manifest. Each
mutation proposal binds:

- the current active candidate artifact and receipt;
- the exact residual-map receipt;
- the selected residual ranks;
- a mutation recipe SHA; and
- the complete proposed resource footprint.

The built candidate must bind that proposal, preserve a contiguous iteration
number, produce a new artifact hash, and match the proposal's resources.
Selection measurements bind the candidate, selection manifest, protocol, and
resource footprint. Iteration receipts form an active-head chain; a rejected
iteration cannot be followed by another transition, and stale or reordered
heads fail closed.

The current implementation recomputes the complete archived transition chain
before emitting a handoff: protocol-pinned seed, phase choice from the active
head's fidelity, legal loop termination, residual maps, proposal membership,
built candidates, complete manifest/family coverage, deterministic selection,
and the guard-bound frozen challenger. The handoff exposes the accepted
candidate's runtime-binding hash directly for the next candidate-bound
protocol. Before
parallel or distributed campaigns, these immutable in-memory receipts should
be backed by an append-only compare-and-swap session ledger so two workers
cannot publish different children of the same accepted head.

## Gemma iteration zero

`make_gemma3_l3_l4_progressive_protocol` reads only prompt-blind identities
from the existing frozen shadow protocol. It binds:

- source model
  `7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9`;
- rank-64 seed artifact
  `b3e011d8067ff3538888851c476fba03c57f4e9f172f923c20fdd90ac0799f84`;
- factorized refit execution
  `911f9869077be1fec2f8610f2f2cbe4c5c6e01a8d632573bec52f2fcc12d1df9`;
- the complete legacy runtime binding plus basis, plan, tensor-file, live
  execution, and shadow-protocol lineage; and
- forbidden Calibration-B manifest
  `986ee9da505fb056853f4fc7ed4f5eee6e9313f0419f2ca9ebc54e0df8607bdd`.

The fidelity targets reuse all existing frozen behavioral, boundary,
full-width projection-capacity, worst-family, coverage, signal, and
carrier-completeness gates. The caller must supply authenticated
seed resource accounting, source-baseline resource totals, and three new
A-only manifests with pairwise-disjoint families.

The legacy one-shot executor remains frozen to the failed rank-64 candidate.
This is intentional: a progressive winner must not inherit that candidate's
protocol identity or development evidence. The next integration boundary is
an instance-bound shadow protocol/runtime whose payload includes the accepted
A transcript and the new candidate, basis, plan, source-model, and execution
hashes. The host-global B ledger should remain keyed by the B manifest so only
one eventual winner can consume it.

## Model-specific next rung

The generic loop is executable, but it does not yet invent Gemma mutations.
The first real worker should:

1. materialize the new A-fit and A-selection panels without touching A guard;
2. measure the seed's full carrier-aware residual;
3. rank Fisher/JVP-coupled residual directions;
4. lower a small fixed proposal schedule, beginning with carrier widening,
   generator splitting, and residual-edge insertion;
5. recompute executable parameters, bytes, MACs, support work, and retained
   source cost;
6. stream scalar A-selection fidelity for every proposal;
7. repeat until the selection envelope passes, then run compaction proposals;
   and
8. freeze the winner before the claim-first A-guard measurement.

Only after that A guard passes should the candidate-bound one-shot protocol be
built. Calibration B remains unopened during every step above.
