# Signed-g8 graph-wavelet confirmation

This rung tests whether the controlled eight-group signed graph-wavelet
candidate survives two increasingly realistic boundaries:

1. fresh, prompt-free impulse-response origins that were not used to fit or
   select the candidate; and
2. source-authoritative execution on real token-conditioned states from an
   already-consumed Calibration-A fit panel.

The result is deliberately split in two. The graph-wavelet partition is
meaningful in the prompt-free structural experiment, but the current linear,
fixed-reference executor fails badly when inserted into the factorized Gemma
model. The candidate is not accepted compression and is not authorized to
serve outputs.

## Frozen candidate

The candidate is the signed local-SVD arm with eight groups of eight parent
modes and total source rank 45. Each dense local rotation is folded into the
compiled source basis, so partitions and mixers are analysis metadata rather
than runtime transforms.

| Quantity | Signed-g8 rank 45 | Rank-64 plan | Reduction |
|---|---:|---:|---:|
| Stored coefficients | 283,456 | 401,408 | 29.38% |
| Prepared bytes | 2,268,184 | 3,211,800 | 29.38% |

This is a plan-payload reduction only. It does not establish whole-model
parameter reduction, end-to-end compute reduction, or speed.

## Confirmation protocol

The fit boundary remained origins 8, 24, and 40. Before reading fresh
responses, the runner froze:

- the native signed eight-group rank-45 plan;
- 63 deterministic, size-matched random partitions, each with eight groups
  of eight and the same local-SVD fitting rule;
- a rank-45 signed-GFA reference; and
- a rank-45 unrestricted global-SVD descriptive ceiling.

The fresh prompt-free map used origins 12, 28, and 36, sequence length 72,
31 causal lags, and FFT length 64. No model, tokenizer, prompt text, or token
IDs were loaded while scoring this confirmation panel, and no plan was refit
after fresh responses were opened.

## Fresh structural result

Lower relative error and higher cosine are better.

| Rank-45 basis | Fresh pooled relative error | Fresh pooled cosine | Role |
|---|---:|---:|---|
| Global SVD | **0.04709** | **0.99889** | Descriptive capacity ceiling |
| Signed eight-group local SVD | **0.16059** | **0.98702** | Frozen candidate |
| Signed GFA | 0.17218 | 0.98507 | Frozen graph-frequency reference |
| Best of 63 random partitions | 0.16331 | 0.98658 | Size-matched null |
| Median random partition | 0.17093 | — | Size-matched null |

The native graph partition beat all 63 random partitions in pooled squared
error. Its plus-one empirical p-value is `1/64 = 0.015625`, and it recovered
`11.73%` of the median random partition's squared error. It also beat the
signed-GFA reference, so the new wavelet result is not merely a graph-frequency
effect: the graph is useful for defining neighborhoods, while response-derived
local rotations remain the better payload basis.

The candidate passed the preregistered pooled fidelity gates at every fresh
origin:

| Origin | Relative error | Cosine |
|---:|---:|---:|
| 12 | 0.17246 | 0.98502 |
| 28 | 0.15498 | 0.98792 |
| 36 | 0.15284 | 0.98825 |

The one failed preregistered gate was uniformity across the eight native
groups. The native grouping beat the median random control in six groups,
while the protocol required at least seven.

| Native group | Squared-error recovery versus median random control |
|---:|---:|
| 0 | +19.30% |
| 1 | +11.61% |
| 2 | +11.42% |
| 3 | +27.05% |
| 4 | +31.53% |
| 5 | **-30.21%** |
| 6 | +4.71% |
| 7 | **-6.25%** |

Therefore the formal structural outcome is a narrow fail: strong pooled
evidence, but only `6/8` group wins against a frozen `7/8` requirement.

## Source-authoritative token shadow

Because the structural miss was narrow, the runner used only the already
consumed 16-prompt Calibration-A fit panel as a diagnostic smoke test. The
panel contains eight families with two prompts each. Calibration-B,
validation, and test remained sealed. This is not held-out qualification.

The runtime made three factorized-model passes per prompt and preserved the
factorized source path as authoritative. Candidate logits and boundaries were
used only for metrics and were never served. Tokenization was streamed one
prompt at a time and its frozen backend identity was checked before and after
every prompt.

| Measurement | Result | Gate |
|---|---:|---:|
| All-token delta NLL per token | **+2.72583** | at most 0.05 |
| All-token source-to-candidate KL per token | **3.01776** | at most 0.05 |
| All-token top-1 agreement | **40.49%** | at least 95% |
| Affected-token delta NLL per token | **+3.19749** | at most 0.05 |
| Affected-token source-to-candidate KL per token | **3.54322** | at most 0.05 |
| Affected-token top-1 agreement | **30.38%** | at least 95% |
| Target-modal relative error / cosine | **5.5104 / 0.4779** | diagnostic |
| Full-width L4 boundary error / cosine | **1.4342 / 0.1314** | diagnostic |

The all-token view covers 931 supervised tokens. The stricter causally
affected view covers 790 tokens. Every behavioral gate failed, and every one
of the eight prompt families failed as well.

This comparison is incremental to the authenticated **factorized-refit Gemma
source**, not raw Gemma. It asks whether the new L3→L4 edge can reproduce the
existing factorized model's behavior after the selected L3 modal contribution
is clamped out.

## What this teaches us

The structural and token results answer different questions:

- The fresh impulse experiment says the signed graph contains real
  organization. Its local neighborhoods support a rank-45 response basis that
  generalizes across new positions better than 63 random partitions and the
  signed-GFA basis.
- The global-SVD ceiling says rank 45 has ample capacity for the measured
  one-mode, fixed-reference response family. It does not prove that the same
  linear response family spans real token states.
- The token shadow says the present carrier is the blocker. Real tokens excite
  many source modes together, at amplitudes and operating points different
  from isolated one-sigma impulses around a mean reference. Linear
  superposition amplifies the predicted modal signal instead of tracking the
  true conditional L4 change.

In other words, the wavelet graph is a useful **map**, but it is not yet a
complete **mutation rule** for the live model. Improving the partition alone
cannot repair a target-modal error of `5.51`.

## Three-basis A-only localization

The corrected V2 comparison has now run the same source-authoritative token
shadow under three frozen, exactly size-matched rank-45 source bases: signed
local SVD, signed GFA, and unrestricted global SVD. Every arm stores `283,456`
coefficients and prepares `2,268,184` bytes. The runner reused the same 16
Calibration-A fit prompts, one loaded factorized model, and one tokenizer for
all three arms. It made 144 model forwards in total, and the independently
recorded source-execution summaries matched exactly across arms.

| Rank-45 basis | All-token delta NLL | All-token KL | All-token top-1 | Affected delta NLL | Affected KL | Affected top-1 | Target-modal error | Full-boundary error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Signed local SVD g8 | +2.72583 | 3.01776 | 40.49% | +3.19749 | 3.54322 | 30.38% | 5.5104 | 1.4342 |
| Signed GFA | **+2.51838** | **2.80428** | 42.11% | **+2.95037** | **3.29210** | 32.28% | **5.4616** | **1.4336** |
| Global SVD | +2.56873 | 2.88217 | **42.43%** | +3.01021 | 3.38089 | **32.66%** | 5.5441 | 1.4618 |

All ordinary and affected behavioral gates failed for every arm, producing
the pass pattern `000`. The preregistered classification is therefore
`no_rank45_basis_viable_attribution_inconclusive`.

Signed GFA has the best NLL and KL metrics in both views, while global SVD has
slightly higher top-1 agreement. Relative to signed local SVD, signed GFA
recovers only `7.07%` of all-token KL and `7.09%` of affected-token KL, with
top-1 gains of `1.61` and `1.90` percentage points. These are real measured
differences, but they are small relative to the distance from the gates.
Global SVD also does not axiswise dominate both graph arms on the frozen
five-axis affected-behavior burden.

This rules out a simple claim that swapping the rank-45 basis repairs the
executor. It does **not** distinguish a rank-45 capacity limit from a failure
of the shared fixed-reference linear carrier. The structural global-SVD
ceiling measured isolated responses; it was not evidence that the same basis
would span simultaneous, token-conditioned states.

The comparison remains incremental to the authenticated factorized-refit
Gemma source, not raw Gemma. It reused Calibration-A fit data and is neither
held-out evidence nor formal qualification. It establishes no serving,
compression, whole-model parameter, compute, latency, or speed claim.

## Rank-64 capacity and boundary-oracle ladder

The semantically hardened V2 A-only ladder has now run. It reused the
authenticated rank-45 global-SVD result, then made five passes for each of the
same 16 prompts on one loaded factorized model and one tokenizer:

1. native, clamped-reference, and learned rank-64 candidate passes;
2. a truth-leaking pass using the true 64 target modes decoded through the
   authenticated target dual; and
3. a truth-leaking pass injecting the exact native X4 tensor while retaining
   the clamped-Y3 carrier.

This is 80 model forwards in total. The source execution summary exactly
matched the prior rank-45 receipt. Both oracle arms are metrics-only analysis
controls and cannot be served.

| Arm | All-token delta NLL | All-token KL | All-token top-1 | Affected delta NLL | Affected KL | Affected top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Global SVD, rank 45 | +2.56873 | 2.88217 | 42.43% | +3.01021 | 3.38089 | 32.66% |
| Global SVD, rank 64 | +2.76865 | 3.05804 | 41.03% | +3.24393 | 3.58748 | 31.01% |
| True target-64 projection oracle | +2.00137 | 2.35277 | **46.51%** | +2.35431 | 2.76483 | **37.47%** |
| Exact native X4 on clamped carrier | **+1.95224** | **2.31611** | 45.01% | **+2.29173** | **2.72133** | 35.82% |

Every ordinary and affected gate failed for all three live arms, so the frozen
rank-64/projection/exact pass pattern is `000`. Its protocol label is
`exact_x4_continuation_invalid`. V2 explicitly records
`upstream_attribution_valid=false`, `boundary_audit_required=true`, and
`development_localization_complete=false`.

That label has a deliberately narrow meaning. X4 here is
`layer.4.mlp.normalized_input`, not the complete layer-4 residual state. The
oracle restores the exact tensor consumed by the layer-4 MLP, but the pass is
still running on the residual carrier produced after clamping the layer-3 MLP
output. Restoring the normalized MLP input therefore does not restore the
separate residual value to which the MLP output is later added. The failed
oracle invalidates this **normalized-X4-on-clamped-carrier continuation as an
attribution instrument**; it does not show that the downstream transformer
would fail when given its complete native state.

The intermediate changes are descriptive rather than causal localization:

- Removing the source-rank limit did not help. Rank 64 was worse than rank 45
  on NLL, KL, and top-1 in both token views.
- Supplying the true target modes improved rank-64 all-token NLL by `0.76728`,
  KL by `0.70527`, and top-1 by `5.48` percentage points. It still missed every
  gate. Its full-width projection error was `0.97362` with cosine `0.22818`.
- Replacing that projection with exact native X4 improved all-token NLL by only
  another `0.04913` and KL by `0.03666`; top-1 fell by `1.50` points. The
  incomplete carrier dominates enough that neither upstream arm can be judged
  from final logits.

The rank-64 plan stores `401,408` coefficients and prepares `3,211,800` bytes:
`41.61%` more coefficients and `41.60%` more prepared bytes than rank 45. It is
a capacity control, not a compression candidate. Its fit-knot replay is
numerically exact (`1.87e-15` relative error, cosine `1.0`), which makes the
live failure more informative: it is not a failure to reconstruct the frozen
fit tensors.

## Complete-H4 boundary audit

The A-only complete-state audit has now run at `layer.4.output`, abbreviated
H4. This is the complete layer-4 carrier consumed by layer 5, and unlike the
post-attention residual it is a safe intervenable boundary in the current
Gemma adapter. The audit used six authenticated model forwards for each of the
same 16 Calibration-A fit prompts, or 96 forwards total:

1. the existing native, clamped-reference, and rank-64 candidate shadow;
2. an independent native replay capturing exact X4 and H4;
3. a replay with clamped Y3 and exact native X4, capturing its incomplete H4;
   and
4. the same partial replay again, this time requiring the pre-intervention H4
   to match exactly before replacing the entire H4 tensor with native H4.

The rank-64 source receipt, source behavior, partial exact-X4 metrics, and all
16 partial exact-X4 logit hashes matched the corrected V2 ladder exactly.
Replacing the complete H4 carrier then recovered the authoritative full logit
tensor bitwise on every prompt.

| Arm | All-token delta NLL | All-token KL | All-token top-1 | Affected delta NLL | Affected KL | Affected top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Rank-64 replay | +2.76865 | 3.05804 | 41.03% | +3.24393 | 3.58748 | 31.01% |
| Partial exact-X4 replay | +1.95224 | 2.31611 | 45.01% | +2.29173 | 2.72133 | 35.82% |
| **Complete native H4** | **0.00000** | **0.00000** | **100.00%** | **0.00000** | **0.00000** | **100.00%** |

Both exactness and the frozen ordinary-plus-affected fidelity ledger passed,
giving `11 / complete_h4_identity_validated`. The maximum full-logit error was
exactly zero for all 16 prompts.

The first implementation of this audit also exposed an important support
distinction. The graph executor's `target_affected_mask` describes its finite
32-lag prediction support; it is not the complete influence support of the
layer-4 residual carrier after causal attention. The corrected audit therefore
hashes the observed byte-level H4 row-difference mask instead of requiring H4
to be unchanged outside the graph mask.

| Observed incomplete-H4 support | Rows |
|---|---:|
| Changed rows, total | 819 |
| Changed valid rows | 819 |
| Changed padding rows | 0 |
| Changed rows inside graph target support | 802 |
| Changed rows outside graph target support | 17 |
| Prompts with outside-target fan-out | 4 / 16 |

Those 17 rows are descriptive causal fan-out, not an integrity failure. The
full incomplete H4 tensor is still hash-authenticated between the two partial
replays, and the full native H4 tensor—including every sequence row—is what is
injected.

This result validates H4 as an attribution boundary and localizes the failed
normalized-X4 continuation to layer 4 or earlier. It does **not** validate the
current learned generator, authorize serving, or establish compression,
compute, latency, or speed savings. Calibration-B, validation, and test remain
sealed.

## Complete-H4 rank-64 capacity result

The next A-only screen fitted a family/example-balanced rank-64 basis to the
complete correction `native H4 - incomplete H4`. The fit covariance applies a
`1 + cos²` tilt using the gradient of prompt mean NLL with respect to H4. This
is a prompt-conditioned empirical-Fisher proxy, not a full activation-Fisher
estimate and not a learned predictor. At evaluation time the runtime
recomputed the truth-leaking projection from the authenticated basis and
required the submitted correction to match it bitwise.

The offline rank ladder showed that the correction is strongly concentrated,
but not concentrated enough at rank 64:

| Rank | Basis coefficients | Family-balanced energy retained | Row-weighted energy retained | Family-balanced RMSE |
|---:|---:|---:|---:|---:|
| 8 | 5,120 | 71.85% | 71.63% | 11.6977 |
| 16 | 10,240 | 83.84% | 83.61% | 8.8633 |
| 32 | 20,480 | 94.25% | 94.11% | 5.2857 |
| 64 | 40,960 | 99.21% | 99.17% | 1.9557 |

The rank-64 cutoff has a `7.28%` relative spectral gap, so this is not merely
an obviously degenerate cutoff. Its pooled complete-H4 geometry was:

| Support stratum | Rows | NRMSE | Cosine |
|---|---:|---:|---:|
| Full complete-H4 support | 819 | 0.09094 | 0.99586 |
| Graph core | 802 | 0.09092 | 0.99586 |
| Causal tail | 17 | 0.30359 | 0.95280 |

The full and graph-core cosine gates passed, but their `0.05` NRMSE gates did
not. The small causal tail is a distinct bottleneck: all four families that
contain tail rows fail both tail geometry gates. The worst full-family NRMSE
is `0.10134` for the sundial family.

End-to-end source-authoritative shadow behavior improved dramatically over the
partial exact-X4 carrier but still missed every preregistered behavioral gate:

| View | Delta NLL/token | KL/token | Top-1 agreement | p90 absolute delta NLL | p10 top-1 |
|---|---:|---:|---:|---:|---:|
| Ordinary, 931 supervised tokens | +0.05531 | 0.08040 | 85.39% | 0.11812 | 81.16% |
| Complete-H4 support, 803 tokens | +0.06413 | 0.09322 | 83.06% | 0.14591 | 78.69% |

Relative to the earlier partial exact-X4 arm, the rank-64 complete-H4 oracle
recovers `97.17%` of excess NLL, `96.53%` of KL, and `73.44%` of top-1
disagreement. That is strong directional evidence that the corrected boundary
and target are meaningful, but it is not a pass. The formal result is
`11000 / rank64_h4_projection_insufficient`: exact-H4 identity and support
integrity pass; boundary geometry, ordinary behavior, and support behavior
fail. The learned generator was therefore not run, and Calibration-B,
validation, and test remain sealed.

The diagnostic basis would occupy `163,840` float32 bytes and perform `81,920`
projection MACs per support row (`67,092,480` across this A16 panel). Oracle
coordinates and the true H4 residual are excluded because neither is
deployable. Current whole-model parameter reduction is zero, and there is no
serving, compression, compute-saving, latency, or speed claim.

## Complete-H4 two-basis rank ladder

The preregistered same-A capacity screen is complete. It fitted two genuinely
distinct maximum-rank-192 bases from the same authenticated A16 pair manifest:

- an unweighted, family/example-balanced residual second moment; and
- the same residual moment tilted by `1 + cos²` alignment with the prompt-mean
  NLL gradient at H4.

Ranks `64 / 96 / 128 / 192` are nested prefixes of each maximum-rank basis, not
independent refits. All eight arms ran against the same shadow observations.
Every prompt also received a live exact-H4 ceiling, which recovered the
authoritative logits bitwise on all `16/16` prompts. The frozen rank-64 arm was
reproduced exactly, including all scalar metrics and logit hashes.

The Fisher-tilted curve is representative of both bases:

| Rank | Energy retained | Full NRMSE | Core NRMSE | Tail NRMSE | Ordinary delta NLL / KL / top-1 | Complete-support delta NLL / KL / top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 99.17% | 0.09094 | 0.09092 | 0.30359 | +0.05531 / 0.08040 / 85.39% | +0.06413 / 0.09322 / 83.06% |
| 96 | 99.82% | 0.04239 | 0.04237 | 0.23902 | +0.03324 / 0.03301 / 91.94% | +0.03854 / 0.03827 / 90.66% |
| 128 | 99.94% | 0.02384 | 0.02383 | 0.15162 | +0.01957 / 0.01497 / 93.34% | +0.02269 / 0.01736 / 92.28% |
| 192 | 99.99% | 0.00992 | 0.00991 | 0.08779 | +0.00506 / 0.00360 / 96.35% | +0.00587 / 0.00418 / 95.77% |

Rank 192 therefore passes the pooled full/core geometry thresholds and every
pooled ordinary, complete-support, graph-core, and causal-tail behavioral
threshold. It still fails the intentionally stricter distributional gates:

- pooled tail NRMSE is `0.08779`, above `0.05`, across only `17` rows;
- the shell and sundial tail examples have NRMSE `0.10171` and `0.13905`, above
  the per-example/family limit of `0.10`;
- ordinary family top-1 is `94.07%` for shell and `94.62%` for sundial;
- complete-support family top-1 is `93.14% / 93.86%`, and graph-core family
  top-1 is `93.07% / 93.40%`, for shell/sundial respectively; and
- the one supervised obsidian tail token has absolute delta NLL `0.09204`,
  above the per-family limit of `0.05`.

These are not cutoff or numerical failures. Both eigensystems were solved in
CPU float64, every cutoff spectral gap exceeded ten times the maximum
eigenpair residual, and the rank-192 tilted/unweighted projectors overlap by
`0.9999999948`; their maximum principal angle is only `0.000744` radians. The
alignment tilt barely rotates this residual eigenspace and provides no
measurable rank-efficiency advantage.

Every arm consequently has pass pattern `11100000`: frozen exact identity,
live exact ceiling, and support integrity pass; boundary geometry and all four
behavior ledgers fail once every nonempty family is enforced. There is no
stable passing rank through 192, so the learned/LOFO generator remains blocked
and Calibration-B, validation, and test remain sealed.

The diagnostic used `272` model forwards, `16` backwards, two 640-by-640
float64 eigendecompositions, and `1,006,387,200` logical projection MACs across
the eight arms. One rank-192 execution basis contains `122,880` coefficients
(`491,520` bytes at float32) and costs `245,760` projection MACs per support
row. These are capacity-screen costs, not parameter or speed savings; the
truth-derived projection coordinates are not a deployable generator.

## Next rung

The result closes the undifferentiated global-linear H4 ladder through rank
192. Simply fitting its coordinates now would train a generator to imitate an
oracle that already fails the qualification contract. The narrow next capacity
screen is instead a structured residual carrier:

1. retain the rank-192 global projection as a frozen diagnostic baseline;
2. separate the omitted correction into the structurally known graph core and
   17-row causal tail, then test a dedicated tail subspace rather than forcing
   both regimes to share the same rank budget;
3. on the remaining core error, compare a small orthogonal reconstruction-only
   augmentation with a downstream-Jacobian/token-Fisher-sensitive
   augmentation, using nested ranks and the same family-balanced receipts;
4. rerun all pooled, every-family, and every-example geometry and behavior
   gates without weakening thresholds; and
5. only if that structured oracle passes, freeze it and begin family-disjoint
   learned coordinate generation.

This separates two live hypotheses: the rare causal tail needs its own carrier,
while the low-energy core residual may be disproportionately important to
token decisions. It also gives the Fisher signal a meaningful job—ranking the
*omitted error by downstream effect*—instead of applying a very small scalar
tilt to a covariance whose leading eigenspace is already fixed.

## Authenticated local receipts

| Object | SHA-256 |
|---|---|
| Frozen signed-g8 logical candidate | `36b7bbf9e42f1e6e9b3182dc7e853580303fee4a38172ba9cad86724a8a6086b` |
| Frozen signed-g8 tensor file | `9fa9b3e1fd93da96e92f40392030a130b2ca70381bb3d37916c817cf53821515` |
| Frozen 63-control null bundle | `18079c3b49375137198263ba06c04c9f84fb4efe50a2d266b58a0b01d9cc93b8` |
| Fresh prompt-free response tensor file | `d6e666644f7268fb24eaa2c3b8b7862caca786ab3b33e5111379c42a6e60a28c` |
| Structural confirmation report payload | `e779f678049d2ec74ec50c5bd701d8c36ca62fe446e4ec15ccb2da334b9e5c16` |
| Guarded A-fit shadow report payload | `beeb7039a06e158ed76b260919aab404c07b4a7785151e41933374bf58615f72` |
| Guarded A-fit shadow report file | `ab1d28ce9ad1900f75ff91870caef47d782af9d0acfab80933ee7894465455ff` |
| Corrected V2 three-basis report payload | `f83acddb861c4461a765f69b6df6f239d76d287a1fe76b92e93c7728aaa9a513` |
| Corrected V2 three-basis report file | `72425e9bf2a1edbe8e7ea6b96fb624510ac1debd29e1e60c158a1fc17226c5a9` |
| Three-basis source-summary receipt | `e1124a8b4ae14a217b80fe0bf6613e94168e23ff8102ed1bc2768829dee4914a` |
| Rank-64 plan | `599e48786597cbd10ae960637b976fdb0392fc71d1505d4b2544d8a44bc51268` |
| Hardened V2 rank-64/oracle ladder report payload | `31fd41b1413e87d3e1fea3b51bd9eeb03bb02f0b69bfa6c22c4b58bb7e8bac40` |
| Hardened V2 rank-64/oracle ladder report file | `a63b8c51e9364dfe057b7c05b4dc064fe5672d86aa0397159d16abb032f4a9d6` |
| Rank-64/oracle source-summary receipt | `e1124a8b4ae14a217b80fe0bf6613e94168e23ff8102ed1bc2768829dee4914a` |
| Complete-H4 identity audit report payload | `cc12df9b49f88c26991015c8ca7e71f67f1cc447bd56503d72d19dd9fdfd1997` |
| Complete-H4 identity audit report file | `54346c57b0871ab2926af27d07458095e1a799ef16cdd4bf3e86408e0df589d2` |
| Complete-H4 rank-64 projection report payload | `8d767392281502f6fe83a8cb21e68a82fb3a743a26063cc75a8de8b9aa34de70` |
| Complete-H4 rank-64 projection report file | `c20dc948b280c24a4b7f6f9dd43e54e5660b94092ed4446ba9145e05098ab73a` |
| Complete-H4 two-basis rank-ladder report payload | `647c4ad889199dad4f50851218e6df3e4de9254fba247bd98dd15ad5d0cb1cee` |
| Complete-H4 two-basis rank-ladder report file | `eb25c0fef53e6dd0a7a5be9726222278fd95848a5368b0f7225ecfe109cc26b9` |

## Run

```bash
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-freeze
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-null-bundle
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-confirm
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-shadow-dev
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-shadow-bases-dev
fisher-graph-gemma-l3-l4-graph-wavelet-signed-g8-rank64-oracle-dev
fisher-graph-gemma-l3-l4-complete-h4-identity-a-dev
fisher-graph-gemma-l3-l4-complete-h4-projection-a-dev
fisher-graph-gemma-l3-l4-complete-h4-basis-rank-ladder-a-dev
```

All generated artifacts remain under the ignored `.local-runs/` tree. Gemma
weights, prompts, token IDs, activation tensors, logits, and compiled plan
tensors are not written into the committable shadow or confirmation reports.
