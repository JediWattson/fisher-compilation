# Fisher Compact-Message Completion Probe v2: P0/K Implementation Profile

## Status and authority

This is a scoped implementation and readiness record for
`fisher_compact_message_completion_probe_v2`, not a replacement for the full
scientific specification. The user-supplied base v2 specification and its
non-conflicting E-, F-, G-, H-, I-, and J-series amendments remain
authoritative for a future scientific campaign. The K-series below records
the additional mechanical closure enforced by the model-free kernel.
Normative terms such as **must**, **may**, and **must not** are binding within
this implemented P0/K scope.

There are two deliberately separate scopes:

- **Current corpus campaign:** P0 provenance preflight only. It must terminate
  `development_complete_independent_data_unavailable` without constructing a
  scientific split, loading a model, capturing activations, fitting a
  representation, or evaluating the hypothesis.
- **Future D2 campaign:** a new campaign may proceed beyond P0 only after a
  genuinely new, independently eligible D2 corpus is supplied. The future
  K-series mechanical subset specified here has a model-free authorization,
  row-index, shard-verification, recovery, failure, and terminal kernel for
  synthetic qualification. That kernel does not make a D2 corpus exist and
  does not authorize or execute a scientific campaign.

### Implementation and execution record

The current-P0 runner and the model-free future-D2 protocol kernel are
implemented in this repository. The corrected P0 transaction is qualified
only with synthetic reserve fixtures: it durably claims the one decode,
publishes and byte-binds the sanitized receipt, and only then commits the
aggregate and per-item `retired_exposed` transitions.

One real local P0 attempt predates that correction. Its nine artifacts are at
`.local-runs/google--gemma-3-270m/fisher-compact-message-completion-probe-v2`
and are preserved byte-for-byte for audit. Its final manifest has semantic
self-hash
`ebc4398432bd04a90cc3f147403a3d25856380bfc0dfdd4cdc318167d57b761b`
and exact file SHA-256
`2063453562a1bc46c9deb2cc7703d2fc8a7d8d4f8054e8f21d153b4a0c85ea62`.
The pinned legacy implementation SHA-256 is
`d06a73933cc405e26f790673979501228e77f17102117aab0389b0fa2a773647`;
it committed registry schema v1 before publishing artifact 2. The verifier
therefore reports that attempt as
`legacy_v1_byte_audit_nonconformant`, with `contract_conformant = false`; it
must not be silently upgraded, rewritten, or rerun. Its exact invalidation
reason is
`legacy_v1_registry_commit_preceded_reserve_receipt_publication`. The corrected
implementation uses registry schema v2.

Within this implementation effort there were at least three known reserve
decodes: one by the real P0 attempt and at least two by the old development
test harness before that harness was replaced with synthetic-only fixtures.
Those accesses did not tokenize data, load a model, capture tensors, or
produce a scientific result, but they exhausted and exceeded the one-time
mechanical decode authorization. The corrected runner pins the exhausted
content identity and must fail closed before a fresh registry claim or file
read, regardless of caller-selected output or registry path.

The P0 campaign itself did not open corpus D. A separate broad implementation
audit did cross D's raw payload boundary. No tokenizer, model, activation,
logit, loss, or scientific evaluation was applied to D, but the access is
disclosed here and D remains ineligible under the frozen policy.

The preserved attempt's scientific disposition remains correct:
`development_complete_independent_data_unavailable`. There is no experiment
metric, candidate, development verdict, or guard verdict. A conformant future
D2 scientific campaign still requires genuinely new independent data and a
separate campaign identity.

Nothing in this contract changes the accepted X4/H4 executor or lowering
paths, the five existing fidelity gates, nearest-rank adverse-tail behavior,
or production compiler output. Mutation lowering remains fail-closed.

## Scientific boundary

The prospective learned object is an activation-conditioned compact-message
completion model at the layer-4 attention-output boundary. It may attempt to
complete the downstream-sensitive effect of message information discarded by
the preregistered compact-message sketch. It is not an activation-dependent
teacher `W_O`, an exact teacher Jacobian, a full attention replacement, a
learned MLP replacement, an autoregressive-generation claim, or a realized
compression claim.

No scientific result exists for the current corpus campaign.

## Current P0 contract

### P0 access boundary

The current exposure policy is exactly `strict_metadata_only_v1`.

Corpus D's eligibility decision must read the authoritative access evidence
through these exact nested fields:

```python
semantic_parse_access_D = receipt[
    "authoring_access_receipt"
][
    "semantic_parse_access"
][
    "D_response_validation"
]

tokenizer_access_D = receipt[
    "authoring_access_receipt"
][
    "tokenizer_access"
][
    "D_response_validation"
]
```

The equivalent JSON Pointers are:

```text
/authoring_access_receipt/semantic_parse_access/D_response_validation
/authoring_access_receipt/tokenizer_access/D_response_validation
```

The authoritative source is the frozen prompt-free receipt at
`artifacts/research/gemma3-chart-native-neighbor-fresh-panels-v1.json`. Its
exact file and semantic receipt hashes must verify before field extraction; a
caller-supplied or regenerated receipt is not authoritative. This verification
must not invoke an opaque-preflight helper that opens or byte-reads the role
corpus files.

Both fields must exist and have JSON Boolean type. A missing or non-Boolean
field is a provenance-integrity failure; it must not be interpreted as
`false`. A flattened top-level field is not a substitute.

The current frozen receipt records both values as `true`. Either value alone
disqualifies D from tune, development-confirm, and guard use under
`strict_metadata_only_v1`. D must therefore be classified
`preoutcome_executable_access`, with all four eligibility fields false:

```text
eligible_for_fit = false
eligible_for_tune = false
eligible_for_development_confirm = false
eligible_for_guard = false
```

This classification says only that D fails the strict metadata-only policy.
It does not assert that D outcomes, activations, logits, losses, or scientific
results were observed.

No current-P0 operation may decode, print, tokenize, embed, or execute model
code over D. Broad filesystem traversal and broad content search must exclude
all corpus payload files.

### Narrow reserve-retirement authorization

The file
`examples/gemma3_chart_native_neighbor_reserve_unopened_corpus_v1.json` has
already been exposed and has permanently lost procedural blindness. It may
never be fit, tune, development-confirm, guard, or replacement-guard data.

The original P0 transition was authorized to decode **only this
already-exposed reserve**, once, and only for the E2/E3
canonicalization-and-retirement transaction. That authorization is now
exhausted. It does not extend to D or any other protected corpus and never
authorized tokenization, model loading, activation capture, logits, losses,
or semantic reuse. The following order remains the normative transaction for
synthetic qualification and for auditing the implementation; it is not fresh
authorization to reopen the real reserve.

The corrected transaction exercised by synthetic qualification must follow
this order:

1. Durably freeze `semantic_canonicalization_spec.json`.
2. Durably acquire one exclusive pre-decode claim bound to the campaign,
   reserve path, frozen canonicalization artifact, expected reserve identity,
   and retirement implementation. If a claim already exists without a bound
   sanitized receipt, fail closed without opening the reserve.
3. Open the authorized fixture at most once, calculate and verify its pinned
   exact file SHA-256, and only then decode those same bytes.
4. Decode it as strict UTF-8 JSON without printing, logging, or embedding raw
   content in any artifact.
5. Construct each exact model-input semantic object. Normalize text with NFC;
   map CRLF and bare CR to LF; preserve case, leading and trailing whitespace,
   repeated internal whitespace, punctuation, code fences, role order,
   message order, and ordered structured-content parts. Normalize only role
   labels to `system`, `developer`, `user`, `assistant`, or `tool`.
   Conversation data is represented as ordered `messages` objects containing
   only the normalized `role` and exact normalized `content` supplied to the
   model. A prompt-string corpus is represented as
   `{"messages":[{"role":"user","content":"<normalized prompt>"}]}`.
6. Exclude only explicitly nonsemantic wrapper fields. Never remove a field
   that affects model input.
7. Serialize semantic objects with RFC 8785 or a documented byte-equivalent
   implementation and calculate the new per-item canonical content hashes and
   aggregate content-set hash.
8. Exclusively write and fsync `reserve_retired_exposed.json`, binding the
   reserve path and exact file
   hash, all per-item canonical hashes, aggregate set hash, available semantic
   instance/template-lineage commitments, exposure reason and timestamp,
   canonicalization identity, and retirement implementation identity.
9. In a separate durable transaction, bind the exact receipt byte hash and
   transition the claim `claimed -> receipt_published`.
10. Commit the aggregate and every item
    `unopened -> retired_exposed` transition atomically with the claim's
    `receipt_published -> retired_exposed` transition.

A crash while the claim is still `claimed` is deliberately unrecoverable and
must never cause a second decode. Once `receipt_published` is durably bound,
every later artifact-prefix recovery must use only that sanitized receipt and
the authenticated registry. A receipt may not reconstruct a missing claim,
and a claim may not bind different receipt bytes.

Existing `canonical_prompt_sha256` values are not a substitute for this
transaction: they were produced under a different text-normalization scheme.
No raw reserve content may survive in stdout, stderr, command logs, temporary
artifacts, failure receipts, or final artifacts. Temporary files must be
private, bounded to this transaction, and removed after commit or recovery.

The aggregate is the exact F19 commitment. Reject duplicate canonical content
hashes, sort the unique 32-byte digests lexicographically as raw bytes, and
compute:

```text
SHA256(
  ASCII("fisher-compact-message-completion/content-set/v1")
  || 0x00
  || uint64_be(number_of_items)
  || digest_1 || ... || digest_n
)
```

### Local registry boundary

The P0 registry is `local_development_registry`. It is append-only or provides
equivalent durable, irreversible transitions. It records corpus roles,
canonical content hashes, local analysis-unit commitments, known historical
roles, tune/development-confirm access state, and exposed-reserve retirement.

`local_development_registry_status.json` must state verbatim:

> This registry is host-local and does not establish private-evaluator
> freshness, cross-host exclusivity, or guard authority.

Local SHA-256 or domain-separated local commitments are bookkeeping only. They
cannot authorize or establish a future private guard. When historical raw
content or authoritative unit metadata is unavailable, record
`historical_identity_unverifiable` and conservatively mark the affected
identity guard-ineligible; absence from an incomplete index is not evidence of
freshness.

### Corpus classification and preflight decision

`corpus_role_classification.json` must contain only sanitized provenance:
corpus identifiers, authoritative receipt/artifact hashes, relevant Boolean
access flags, policy version, eligibility decisions, reasons, and no raw
content. The current roles are fixed:

| Corpus | Current eligibility | Reason |
| --- | --- | --- |
| A | fit only | fit source; never tune, development-confirm, or guard |
| B | none | prior scientific use |
| C | none | prior scientific use |
| D | none | strict-policy failure because semantic parse and tokenizer access are true |
| exposed reserve | none | irreversibly `retired_exposed` |

No fallback to B or C is permitted. D must not be reinterpreted as
metadata-only merely because no guard claim occurred.

`preflight_decision.json` must contain these exact decisions:

```text
can_construct_fit = true
can_construct_tune = false
can_construct_development_confirm = false
campaign_can_start = false
reason = "No eligible independent tune/development-confirm corpus"
```

The resulting terminal state is exactly:

```text
development_complete_independent_data_unavailable
```

This is a successful fail-closed protocol outcome, not evidence for or against
the compact-message completion hypothesis.

### Complete P0 artifact contract

Any conformant P0 materialization would create exactly these nine mandatory
artifacts in this order. The corrected implementation demonstrates this only
with synthetic fixtures; the preserved real attempt has this byte order but
is mechanically nonconformant for the separately disclosed transaction-order
reason.

| Order | Artifact | Required role |
| ---: | --- | --- |
| 1 | `semantic_canonicalization_spec.json` | freezes semantic and JSON canonicalization before content hashing |
| 2 | `reserve_retired_exposed.json` | proves the one-time exposed-reserve retirement transaction |
| 3 | `local_development_registry_status.json` | records the non-authoritative local registry and backfill status |
| 4 | `corpus_role_classification.json` | records sanitized A/B/C/D/reserve provenance and eligibility |
| 5 | `preflight_decision.json` | records the exact no-independent-data decision |
| 6 | `commands.md` | records sanitized commands and outcomes without corpus content |
| 7 | `terminal_state.json` | records the terminal and reached-prefix state |
| 8 | `final_report.md` | reports the fail-closed outcome and explicit non-claims |
| 9 | `final_experiment_manifest.json` | final commit point; inventories artifacts 1-8 by exact byte hash and excludes itself |

Every required artifact must be nonempty, schema-valid, and hash-verified.
Self-hashed JSON excludes only its own semantic digest field. Exact artifact
byte hashes are distinct from semantic self-hashes. JSON must be strict UTF-8,
reject NaN and infinity, and use RFC 8785 or a documented byte-equivalent
serialization.

`terminal_state.json`, `final_report.md`, and
`final_experiment_manifest.json` are finalized in that order. The final
manifest is written last. A conformant final manifest would make the P0
campaign and all inventoried artifacts immutable. The legacy manifest makes
the preserved attempt immutable but does not cure its earlier ordering defect.

### Mandatory P0 stop semantics

After `preflight_decision.json` records `campaign_can_start = false`, the
implementation must stop before all of the following:

- independence-mode selection;
- scientific split construction;
- model or tokenizer resolution;
- model, carrier, runtime, execution-budget, or source-inventory sealing;
- fit, tune, or development-confirm authorization;
- activation, message, or tensor capture;
- O1 or any later scientific gate;
- representation, chart, codec, or message-sketch fitting;
- completion-model or comparator fitting;
- candidate/comparator bundle creation;
- development or guard evaluation.

No P1-P12 artifact, experiment metric, model verdict, development verdict, or
guard verdict may exist in this terminal state.

The final report must include this conclusion:

> The protocol completed its fail-closed corpus-provenance preflight. Corpus D
> does not satisfy the frozen strict metadata-only eligibility policy because
> its authoritative provenance records semantic parsing and tokenization
> access. With B and C already scientifically used, no eligible independent
> tune and development-confirm source remains. The campaign therefore
> terminated before source sealing, capture, model execution, or scientific
> evaluation.

It must also state that no raw D content was inspected by the campaign, no
experiment was run, no model candidate was fitted, no development or guard
verdict exists, and the terminal state is
`development_complete_independent_data_unavailable`.

## Future D2 boundary

A future D2 does not resume or mutate the preserved legacy P0 attempt. It
requires a new campaign identifier, a new corpus-role classification and
independence decision, new deterministic splits, new model/runtime/source
seals, new capture authorizations, a new preregistration root, a new
local-development freshness receipt, and a new campaign root. The legacy P0
manifest may be cited only as immutable, explicitly nonconformant provenance.

D2 is eligible under `strict_metadata_only_v1` only if its prior handling was
limited to allowed metadata and its authoritative receipt establishes that no
semantic parse, tokenization, model execution, activation, logits, losses,
outcome-informed inspection, or result-informed design access occurred. A
different exposure policy is a new specification and campaign, not an implicit
exception.

A future development campaign still has no guard authority. Any later guard
requires a separately initialized evaluator-controlled campaign, private raw
reserve, shared transactional registry, immutable evaluator-side result store,
and evaluator-side freshness/claim transaction. No local path, ignored file,
SQLite record, or renamed corpus simulates that boundary.

### Implementation boundary

| Area | Current implementation status |
| --- | --- |
| Corrected P0 transaction | Implemented and conformantly qualified with synthetic fixtures; permanently blocked from reopening the exhausted real reserve |
| Preserved real P0 attempt | Exact nine-artifact byte audit supported; legacy registry v1; explicitly nonconformant |
| K authorization, row-index, and shard plan | Model-free ledger mechanics implemented for synthetic qualification |
| Safetensors evidence | Existing-file parser, row-ID/tensor alignment, plan-order rederivation, and mutation checks implemented; no model-produced shard writer |
| K recovery, failures, prefixes, and terminal tail | Model-free mechanics implemented subject to the verification record below |
| External scientific artifacts | Framing, durability, byte hash, materialization order, and prefix closure enforced; each scientific artifact's domain schema remains the producer's responsibility |
| D2 corpus and splits | Not supplied or implemented |
| Model/tokenizer/runtime/source builders and actual capture | Not implemented or run |
| O1/O2, locator/codec/representation fitting, numerical estimators, J models, ablations, and O7 | Not implemented or run |
| Private guard/evaluator | Not present; no local component has guard authority |

K1-K7 are normative for the ledger and verifier portions of the model-free
future-D2 kernel, while K8's reason registry and failure taxonomy are enforced
mechanically. K8's operator-reference calculation and lower-K scientific
producer behavior remain part of the unimplemented scientific pipeline. All
of this is dormant in the current P0 path, and the omitted detailed scientific
clauses in the base v2 and E-J specifications remain future requirements.

## K-series mechanical closure for future D2

### K1. Commit one canonical row index in both authorization modes

Every fit, tune, and development-confirm capture uses one canonical row-index
payload. It contains the partition role, row-derivation protocol hash, row
count, unique row IDs in ascending lexicographic order of their raw 32-byte
values, `row_set_sha256`, and `ordered_row_sequence_sha256`. Sorting hexadecimal
text, input order, tokenizer traversal order, locale order, filesystem order,
or shard completion order is forbidden.

The row-index payload receives its own domain-separated hash:

```text
row_index_payload_sha256 =
  SHA256(
    ASCII(
      "fisher-compact-message-completion/"
      "evaluation-row-index-payload/v1"
    )
    || 0x00
    || RFC8785(canonical_row_index_payload)
  )
```

The canonical payload schema identifier is exactly
`fisher-compact-message-completion/evaluation-row-index-payload/v1`; the
domain and schema must not use the shorter `row-index-payload/v1` spelling.

The existing row-set and ordered-sequence commitments remain distinct and use
their frozen domains:

```text
fisher-compact-message-completion/evaluation-row-set/v1
fisher-compact-message-completion/evaluation-row-sequence/v1
```

Mode A, `expected_row_set_mode = "precomputed"`, must bind the precomputed
`row_index_payload_sha256`, row-set hash, ordered-sequence hash, row count, and
row-derivation protocol hash in the semantic authorization payload. After
authorization, it verifies and commits those exact payload bytes.

Mode B, `expected_row_set_mode = "derive_after_authorization"`, binds the
row-derivation protocol while the expected payload/set/sequence hashes and row
count are null. After the sticky authorization, it derives and commits the
canonical payload before capture.

Both modes must materialize, from one durable row-index transaction:

```text
row_index_manifest_<partition>.json
<partition>_row_index_committed.json
```

Both artifacts bind identical authorization hash, authorization transaction
ID, partition role, row-derivation protocol hash, row-index payload hash,
row-set hash, ordered-sequence hash, row count, and row-index transaction ID.
An aborted transaction creates neither artifact. Recovery reuses the committed
transaction; it never allocates a replacement identity. Tensor capture cannot
begin before `row_index_committed` in either mode.

This K1 rule supersedes earlier wording that made row-index artifacts
conditional on deferred derivation.

### K2. Co-serialize row identity and rederive captured commitments

Every finalized tensor shard must contain:

```text
capture_row_id_raw32    uint8[n, 32]
```

The first dimension must equal the shard row count and align exactly with every
captured tensor's first dimension. These bytes are evidence, not descriptive
metadata.

Verification must reopen the finalized immutable shard bytes, read
`capture_row_id_raw32`, concatenate shards in the frozen plan order, reject
duplicates or malformed IDs, and independently derive the captured row count,
`captured_row_set_sha256`, and
`captured_ordered_row_sequence_sha256`. It must not trust in-memory IDs or copy
the committed hashes into captured fields. All captured values must exactly
equal their committed counterparts. The right set in a different order is
`invalid_campaign_integrity` with reason `capture_row_order_mismatch`.

### K3. Freeze the exact shard plan

Let `n` be the committed row count and let `L` be the frozen positive shard-row
limit, no greater than 16,384. Shard `k`, for
`0 <= k < ceil(n / L)`, is exactly:

```text
start_row_ordinal = k * L
end_row_ordinal_exclusive = min((k + 1) * L, n)
```

The exact relative path template is:

```text
capture_shards/{partition}/shard-{shard_index:06d}.safetensors
```

`partition` is one of `fit`, `tune`, or `development_confirm`; no absolute
path, `..`, alternate spelling, skipped index, overlapping range, empty shard,
or extra shard is valid.

Before capture, freeze `capture_shard_plan_<partition>.json` with `n`, `L`, the
path template, every shard index and range, row-index payload hash, and
ordered-sequence hash. Its semantic commitment is:

```text
capture_shard_plan_sha256 =
  SHA256(
    ASCII("fisher-compact-message-completion/capture-shard-plan/v1")
    || 0x00
    || RFC8785(canonical_shard_plan_payload)
  )
```

Every shard record and capture transition binds this plan hash. Finalized
shards are immutable and byte-hashed.

### K4. Make `capture_started` durable and interruption-safe

After row-index and shard-plan commitment, but before any tensor or protected
row is written, atomically commit:

```text
<partition>_tensor_capture_started.json
state = "capture_started"
```

The receipt binds the authorization hash and transaction ID, partition role,
row-index transaction and payload hash, committed row-set and ordered-sequence
hashes, shard-plan hash, and `capture_attempt_id`. The attempt ID is allocated
once by the durable start transaction and is reused by every recovery. Capture
cannot start unless this receipt is durable.

The normal state machine is:

```text
unopened
  -> authorized
  -> row_index_committed
  -> capture_started
  -> consumed_success | consumed_failed
```

Row-index failure remains:

```text
unopened -> authorized -> row_index_failed -> consumed_failed
```

A crash before `capture_started` leaves no finalized shard and may recover
idempotently from the same committed authorization and row index. A crash
after `capture_started` remains in `capture_started`; interruption alone does
not transition the attempt to `consumed_failed`. Recovery must reuse the same
`capture_attempt_id`, verify every durable finalized shard against the frozen
plan and exact byte hash, preserve each verified shard unchanged, and
regenerate only missing or explicitly temporary/uncommitted shards. It may not
allocate new identities, rewrite a verified finalized shard, or create a
second capture attempt. After recovery, the original attempt reaches exactly
one success or failure finalization.

### K5. Use strict, exclusive failure receipts

Every operational failure has exactly one primary class:

| `failure_class` | Receipt | Terminal |
| --- | --- | --- |
| `integrity` | `invalid_campaign_receipt.json` | `invalid_campaign_integrity` |
| `measurement_contract` | `invalid_campaign_receipt.json` | `invalid_campaign_measurement_contract` |
| `resource_limit` | `resource_limit_failure.json` | `development_complete_resource_limited` |

Each receipt must use its versioned schema and contain the campaign ID,
failure class, registered reason code, last completed prefix, active prefix,
failure stage, partition when applicable, relevant authorization/row-index/
capture-start/shard hashes, sanitized details, and the exact durably reached
artifact list. It must contain no raw prompt, response, token, or protected
panel content.

Hash, identity, authorization, dimension, row-order, duplicate-row, source, or
artifact contradiction is an integrity failure. A verified-identity runtime
seam, mask, tensor, head, normalization, deterministic replay, or equivalent
execution-contract violation is a measurement-contract failure. Exhaustion of
the frozen resource budget is a resource-limit terminal. The classes are
mutually exclusive and none may emit a scientific veto.

Insufficient support is not one of these classes. It uses
`support_failure_receipt.json`, the reached-prefix rule, and terminal
`development_complete_insufficient_support`.

### K6. Finalize capture success and failure atomically

Each capture attempt has exactly one mutually exclusive finalization
transaction, bound to the same partition and `capture_attempt_id`.

Successful capture atomically commits both:

```text
capture_shard_manifest_<partition>.json
<partition>_capture_transition_consumed_success.json
```

The complete manifest is built only after reopening every finalized shard and
verifying the K1-K3 commitments. Both artifacts share one capture-finalization
transaction ID. The durable transaction commit receipt binds the exact byte
SHA-256 of both artifacts without introducing a circular self-hash, so either
the verified pair commits or neither does.

Failed capture atomically commits:

```text
partial_shard_inventory_<partition>.json   # only if verified complete partial shards exist
capture_failure_<partition>.json
<partition>_capture_transition_consumed_failed.json
```

All present artifacts share one capture-finalization transaction ID. Its
durable commit receipt binds their exact byte hashes, including the optional
partial inventory, without a circular self-hash. A failure transaction must
not create a complete capture-shard manifest. Success and failure transitions
are mutually exclusive, sticky, and cannot coexist. Recovery from an
interrupted finalization must either publish the identical verified
transaction or publish neither branch.

Campaign-level terminal closure is a separate retained rule: when a campaign
terminates, it writes `terminal_state.json`, then `final_report.md`, then
`final_experiment_manifest.json`. The final experiment manifest is last,
inventories exact byte hashes for all and only reached artifacts, excludes
itself, and makes the finalized campaign immutable.

### K7. Apply the conditional reached-prefix matrix

The future campaign uses monotonic P0-P12 prefixes:

| Prefix | Reached only when | Mandatory future artifact families |
| --- | --- | --- |
| P0 | provenance preflight begins | the nine P0 artifacts at terminal closure |
| P1 | eligible independent data exists | independence protocol, split manifest, O1 assignment |
| P2 | campaign identity is sealed | model/carrier/runtime/budget/source identities and row-derivation protocol |
| P3 | fit is authorized/capture attempted | fit authorization pair; K1 row-index pair in both modes; shard plan; conditional start, outcome, shard, or failure artifacts |
| P4 | valid fit capture reaches O1/representation preregistration | O1 protocols/results, full-rank identity smoke, ceiling protocol, representation protocols/manifests, preregistration/freshness/campaign roots |
| P5 | tune is authorized/capture attempted | tune authorization pair; K1 row-index pair; shard plan; conditional start/outcome/shard/failure artifacts; O2 results when capture succeeds |
| P6 | a codec passes O2 | selected codec, frozen representation, M2 and compact-oracle artifacts |
| P7 | initial model fitting begins | reached J-model fit/tune artifacts |
| P8 | all-fit weight refit/replay begins | reached refit and nonselecting replay artifacts |
| P9 | development-confirm freeze is valid | B-star, direct full-rank null, frozen bundles/metadata, development-confirm authorization pair |
| P10 | development-confirm capture is attempted | K1 row-index pair; shard plan; conditional start/outcome/shard/failure artifacts |
| P11 | valid development-confirm capture is evaluated | reached development, ablation, O7, eligibility, and power artifacts |
| P12 | development succeeds | development candidate manifest |

For every capture partition:

- `capture_shard_manifest_<partition>.json` exists only after
  `consumed_success` and complete capture verification.
- Failure after `capture_started` requires `capture_failure_<partition>.json`
  and `consumed_failed`; it also requires
  `partial_shard_inventory_<partition>.json` only if complete immutable partial
  shards exist.
- Row-index failure requires the row-index failure receipt and transitions,
  `tensor_capture_started = false`, and no shard or partial-shard manifest.
- No selected codec exists before O2 success; no later model, bundle,
  development, or candidate artifact exists before its actual stage.
- Every terminal requires all completed prefixes, all durably completed
  artifacts in the active prefix, its terminal-specific receipt, and no
  artifact from an unreached future prefix.

### K8. Delete the impossible chart/head support reason

Delete `zero_fit_monitor_chart_head_support` from the support-failure reason
registry and all schemas, fixtures, reports, and branch logic. With complete
per-row head capture, an active chart either has fit-monitor rows for every
head or the capture is malformed. Therefore:

- no rows for an active chart is `zero_fit_monitor_chart_support` and may enter
  the preregistered lower-K fallback;
- a missing head on an otherwise valid row is a measurement-contract failure;
- an empty strictly-positive codec operator-reference set is
  `no_positive_operator_reference`;
- exhausted fallback is `lower_K_fallback_exhausted`;
- inadequate independent chart coverage is
  `insufficient_independent_chart_support`.

No implementation may emit the deleted reason.

### K9. Required acceptance categories

The complete future scientific implementation plan contains 54 required
acceptance categories. This table is an index, not evidence of conformance and
not a substitute for the detailed vectors in the base v2 and E-J
specifications. A category may be parameterized into multiple test functions,
but no category may be collapsed away.

| Categories | Count | Required coverage |
| --- | ---: | --- |
| Current P0 disposition and access firewall | 8 | exact nested H1 fields; missing/non-Boolean/flattened rejection; A fit-only; B/C excluded; D strict-policy exclusion; no D decode/model/capture; exact terminal |
| P0 canonicalization, reserve retirement, registry, artifacts | 7 | E2 normalization; one-time reserve-only decode; irreversible retirement; registry disclaimer/backfill; exact nine artifacts/order; report non-claims; finalized-P0 immutability |
| Authorization and transaction identity | 5 | semantic authorization hash/nonce; excluded durable transaction ID; atomic authorization pair; aborted transaction; idempotent recovery |
| K1 canonical row-index modes | 6 | Mode A payload binding; Mode B deferred derivation; both emit manifest/commit receipt; identical cross-bindings; duplicate rejection; pre-capture commitment |
| K2 row identity, serialization, and ordering | 6 | raw-byte ordering; `uint8[n,32]`; tensor alignment; reopen-and-rederive; set equality; ordered-sequence mismatch failure |
| K3/K4 shard plan and capture lifecycle | 6 | exact range formula; exact path/domain commitment; contiguous global order; durable `capture_started`; pre/post-start crash recovery; immutable partial-shard handling |
| K5 failure classification and support closure | 5 | strict receipt schemas; exclusive integrity/measurement/resource classes; sanitized content; insufficient-support separation; deleted K8 reason unreachable |
| K6/K7 prefix, hash, and finalization closure | 4 | conditional artifact matrix; reached-prefix-only closure; exact artifact byte hashes; atomic final-manifest-last commit and recovery |
| Representation support and operator references | 3 | fit-monitor/lower-K behavior; positive-only codec reference with zero descriptive reporting; empty-positive-set terminal behavior |
| Numerical and model-mechanics regression | 2 | frozen weighting/median/rho/split/Hamilton/parameter contracts; optional J2 and mandatory J4 comparisons |
| Future campaign and guard separation | 1 | D2 uses a new campaign and cannot mutate P0; local authority cannot claim a guard |
| Existing production behavior | 1 | X4/H4, five fidelity gates, nearest-rank tails, lowering, and fail-closed mutation behavior remain unchanged |
| **Total** | **54** | |

The corrected P0 implementation exercises the applicable P0 categories only
with synthetic fixtures. The model-free K test suite exercises the mechanical
subsets identified in the implementation boundary; it does not cover the
absent corpus/model/scientific pipeline. The full 54-category acceptance set
therefore has not passed.

## Verification and readiness statement

The corrected P0 implementation is synthetically qualified, while the real
P0 endpoint is permanently closed and its preserved result is audit-only. The
model-free K-series kernel is mechanically qualified for its documented
scope. An independent static review found no remaining correctness blocker or
scientific/numerical overclaim, and independently ran the final focused suite:

| Verification set | Result |
| --- | ---: |
| Corrected P0 synthetic fixtures | 36 passed |
| Model-free K protocol | 121 passed |
| Combined new suite, Python 3.14 with warnings as errors | 157 passed |

The root-checkout validation additionally ran 187 selected existing
local-chart, capture, probe, carrier, fidelity, lowering, autonomous-residual,
and X4/H4 factorial regressions. The final combined selected run was therefore
344 passed with Python warnings treated as errors. This is a focused
regression selection, not the unimplemented 54-category scientific acceptance
campaign.

The preserved real P0 verifier also succeeds as a byte audit while reporting
`contract_conformant = false`, verification mode
`legacy_v1_byte_audit_nonconformant`, and invalidation reason
`legacy_v1_registry_commit_preceded_reserve_receipt_publication`. This is a
successful audit of a nonconformant historical attempt, not a corrected real
execution.

None of this authorizes a scientific run with the current A/B/C/D corpus set.
Future D2 execution remains contingent on new eligible data, a new campaign,
implementation of the corpus/model-specific scientific pipeline, and the
complete applicable qualification suite.
