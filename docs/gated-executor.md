# Residual-separated gated causal modal executor

This is the first Gemma experiment in the repository that fits an executable
forward graph against a real multi-layer block and then intervenes at the
block output. It tests the hypothesis suggested by the weighted-Jacobian
pilot: keep lag zero separate, and let a small state-conditioned mixture
represent the positive-lag transport that was not stationary across positions
and contexts.

The implementation is intentionally inspectable. It is also currently
**nonviable** as a compression of the pinned Gemma 3 270M layers 4–6 block.
The tested executor has attractive stored-coefficient and analytic-MAC
ratios, but fails the predeclared direct-output and language-behavior gates by
large margins.

## Graph

For input modal coordinate \(x_t\) at logical position \(t\), the generic
executor computes

\[
y_t =
\operatorname{skip}(x_t)
+ x_tW_{\mathrm{same}} + b
+ \sum_{s<t}\sum_{e=1}^{E}
p_{t,s,e}\,(x_sU_e)V_e.
\]

The router probabilities are

\[
p_{t,s}
=
\operatorname{softmax}\!\left(
\phi\!\left(
x_tW_q + x_sW_k
+ \log(1+t-s)\,w_{\mathrm{lag}}
\right)W_r+b_r
\right).
\]

The design makes four distinctions explicit:

- The same-position affine path is independent of cross-token transport.
- Every cross-token expert is low rank: \(U_e\) maps into an expert latent
  width and \(V_e\) maps back to output modes.
- The router sees the current query state, source state, and relative logical
  lag. It has no absolute-position table, so parameter shapes do not depend on
  sequence length.
- Cross-token edges exist only for valid source slots with strictly smaller
  logical positions. Equal-position, future, padding, and over-budget lag
  edges have no executable contribution.

`forward_components()` exposes the same-position output, positive-lag output,
legal-edge mask, and router probabilities. `execution_accounting()` reports
logical parameter and ideal sparse MAC counts. Its MAC count excludes
additions, bias application, router activation, softmax, masking, memory
traffic, and kernel-launch costs; it is not a latency prediction.

## Gemma block replacement

The Gemma runner uses the generalized activation codecs selected by the prior
weighted-Jacobian artifact, but does not assume that the input and output
codec gauges align. It retains the raw residual stream as an exact bypass:

\[
\begin{aligned}
z_{\mathrm{in}}
&= (h_{\mathrm{in}}-\mu_{\mathrm{in}})E_{\mathrm{in},:r},\\
\Delta z
&= \operatorname{GatedExecutor}(z_{\mathrm{in}}),\\
\widehat h_{\mathrm{out}}
&= h_{\mathrm{in}}
+ \Delta zD_{\mathrm{out},:r}^{\mathsf T}.
\end{aligned}
\]

Thus the graph predicts the three-layer block delta, not the complete residual
stream. The live diagnostic still executes the native layers 4–6 so it can
capture the reference output and intervene at the final boundary. The
reported graph MACs therefore describe a hypothetical replacement; the
diagnostic itself is not a faster runtime.

## Split and selection protocol

The committed fixture
`examples/gemma3_gated_executor_prompts.json` contains 16 prompts in each of
four disjoint roles:

- calibration A fits executor weights for exactly 100 optimizer steps;
- calibration B evaluates the predeclared candidate grid and locks one
  configuration;
- validation evaluates that locked configuration once;
- reserved test is schema-checked and hashed, but never tokenized or sent
  through the model.

The fixture is also hash-disjoint from the source weighted-Jacobian prompt
artifact. It is a fresh diagnostic protocol for this executor experiment, not
a representative population benchmark.

The candidate grid is:

- retained ranks 320 and 480 out of width 640;
- one or two positive-lag experts;
- expert rank 16;
- router width 16;
- all legal positive logical lags.

Every candidate must pass all seven gates:

| Gate | Threshold |
|---|---:|
| Retained rank / hidden width | at most 0.75 |
| Stored coefficients / source block parameters | at most 0.75 |
| Analytic MACs / source block analytic MACs | at most 0.75 |
| Block-delta NRMSE | at most 0.20 |
| Block-delta cosine | at least 0.95 |
| Absolute delta NLL/token | at most 0.05 |
| Top-1 agreement | at least 0.95 |

If no candidate passes on calibration B, the protocol still locks a
diagnostic fallback by direct NRMSE, absolute delta NLL, top-1 agreement, and
stable candidate ID. That fallback is explicitly marked nonviable before
validation.

## Reproduce the pinned local run

First create the weighted-Jacobian source artifact described in
[`weighted-jacobian-compilation.md`](weighted-jacobian-compilation.md). Then
run:

```bash
fisher-graph-gemma-gated-executor \
  --weighted-artifact \
    .local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt \
  --model google/gemma-3-270m \
  --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 \
  --local-files-only \
  --prompt-splits examples/gemma3_gated_executor_prompts.json \
  --max-length 128 \
  --retained-ranks 320 480 \
  --expert-counts 1 2 \
  --expert-ranks 16 \
  --router-widths 16 \
  --max-positive-lags none \
  --fit-steps 100 \
  --device cpu \
  --dtype float32
```

The default derived outputs are ignored:

```text
.local-runs/google--gemma-3-270m/layers-4-6-gated-executor.pt
.local-runs/google--gemma-3-270m/layers-4-6-gated-executor.json
```

The pretrained model and tokenizer remain in the external Hugging Face cache.
Neither model files nor generated local artifacts are committed to the
repository. The locked candidate's FP32 runtime state is about 2.17 MB. The
local diagnostic `.pt` is roughly 16 MB because it stores all four fitted
candidate states plus the codec and audit payloads; its file size is not the
deployable locked-runtime storage figure.

## Pinned result

No candidate at rank 320 or 480 passed calibration-B selection. The diagnostic
fallback locked:

```text
rank_320.experts_2.expert_rank_16.router_16.lag_all
```

Its one locked validation evaluation produced:

| Quantity | Result |
|---|---:|
| Block-delta NRMSE | 0.823388 |
| Block-delta cosine | 0.605518 |
| Delta NLL/token | +7.015665 |
| Top-1 agreement | 0.07381 |
| Stored coefficients / source block parameters | 3.2518% |
| Analytic MACs / source block analytic MACs | 3.2290% |

The resource gates pass comfortably. The four fidelity gates fail
comfortably. This is therefore a low-resource graph, but not a compressed
replacement for the block.

The validation accounting includes the retained input mean, input encoder
slice, output decoder slice, and gated graph parameters in stored
coefficients. Its source denominator is the exact 16,720,896 parameters in
Gemma layers 4–6. Analytic MACs include codec encode/decode, same-position
transport, all soft-mixture experts, router projections, and the real
validation sequence lengths. The source denominator includes the block's
linear weight MACs plus QK and AV dot products on the same causal edges.
Neither side includes normalization, activation functions, softmax, RoPE,
masking, additions, or memory traffic.

## Why the target-informed reference matters

The runner also computes the least-squares rank-\(r\) reconstruction of the
*true* block delta independently at every token in the locked output decoder
span. At rank 320 on validation, that direct reference reached:

| Oracle quantity | Result |
|---|---:|
| Block-delta NRMSE | 0.055995 |
| Block-delta cosine | 0.998431 |
| Delta NLL/token after intervention | +6.342280 |
| Top-1 agreement after intervention | 0.088095 |

Because it consumes the true target delta, this is not an inference-time
executor. It is also not a behavioral upper bound: least-squares projection
optimizes Euclidean residual error, not NLL or top-1 agreement, so another
predictor in the same span could trade more L2 error for better behavior.

It is still a highly informative reference. The codec's rank-320 output span
can approximate the block delta closely in global Euclidean energy, yet the
small remaining error is concentrated in directions that the rest of the
model treats as highly consequential. In other words, low block-output MSE is
not aligned with downstream behavioral fidelity.

The trained gated graph is worse than the oracle in direct error, so its
optimization and capacity are not solved either. But improving only that MSE
toward the oracle would still not pass the language-behavior gates. A next
attempt needs a behavior-aware or sensitivity-weighted objective and likely a
different output subspace; it cannot rely on raw residual MSE alone.

The present router also normalizes across experts, not across source tokens.
Every legal earlier token therefore contributes through some expert mixture;
there is no explicit null expert or scalar edge gate that can suppress an
irrelevant source edge. The positive-lag path carried 27.77% of the reported
validation path energy, which establishes that it was active, not that it was
useful. A null edge gate and a parameter-matched same-position-only fit are
needed before attributing any gain or loss to causal transport.

## Controls and claim boundary

The no-op block-output intervention had zero validation delta NLL and exact
top-1 agreement. The full-width codec delta round trip passed in mathematical
FP64 and runtime FP32 checks. The model-state guard verified frozen weights,
the new prompts were hash-disjoint from the source artifact, and generic
executor tests cover future-edge exclusion, padding, logical-position gaps,
variable lengths, and strict artifact loading.

These controls rule out an intervention-hook failure, broken full-width codec,
silent weight update, reused source prompt, or simple causal-mask bug as the
explanation for the negative result.

They do not support any of the following claims:

- that state-conditioned causal edges provide no benefit;
- that all ranks below 640 must fail;
- that more experts, other nonlinearities, or another objective cannot work;
- that modal compression is impossible for Gemma;
- that the analytic MAC ratio predicts a faster GPU or CPU kernel.

The evidence is narrower: for this pinned model revision, fresh diagnostic
fixture, generalized codec, ranks 320/480, one seed, 100-step residual-MSE
fit, and small one/two-expert grid, the replacement is nonviable.

## Projection-ladder follow-up

That representation-isolation experiment is now complete. A source-disjoint
calibration-B ladder evaluated nested generalized-decoder prefixes from rank
480 through full width. No reduced rank passed both behavior gates. Rank 639
was the closest: delta NLL/token was -0.003372, within the absolute 0.05 gate,
but top-1 agreement was 0.9431 versus the required 0.95. The protocol
therefore locked rank 640 identity; its one validation intervention had delta
NLL/token \(+2.73\times10^{-7}\) and exact top-1 agreement.

This is a sharper diagnosis than the rank-320 gated result. Even a
target-informed per-token least-squares projection that preserves
approximately 99.99868% of direct block-delta energy at rank 639 changes too
many token argmaxes. It still does not establish that every rank-639 subspace
fails: the ladder tested prefixes of one locked generalized decoder, not
arbitrary behavior-aware rotations. Validation saw only the rank-640
fallback, and reserved test remained hash-only. See
[`gemma3-270m.md`](gemma3-270m.md#run-the-target-informed-projection-only-behavioral-rank-ladder)
for the full curve and protocol.

The next useful sequence is:

1. compare the omitted codec direction with a fresh-calibration
   minimum-downstream-sensitivity direction at codimension one;
2. if that rotated span passes, build a behavior-aware removal ordering and
   only then extend the removal ladder;
3. train inside a behavior-preserving span with combined block-output and
   downstream-logit or KL loss;
4. add an explicit null/scalar edge gate and a parameter-matched
   same-position-only baseline so positive-lag benefit is identifiable;
5. widen executor capacity only after the representation-level reference
   preserves behavior;
6. keep split locking, identity controls, per-length reporting, and
   reserved-test discipline.

Kernel optimization and larger expert sweeps remain downstream work until a
reduced representation passes the behavioral gates.
