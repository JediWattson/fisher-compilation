# Activation-aware weighted-Jacobian compilation

This rung turns the original “Fisher modes plus Jacobian edges” idea into
separate, testable compiler stages:

1. choose activation coordinates using both score sensitivity and the
   activation distribution;
2. measure signed forward Jacobian edges through a frozen source block;
3. merge those edges with a Fisher-weighted causal SVD;
4. execute the retained signed factors without any future-position slots.

The generic factor/executor is implemented. The Gemma command is still an
analysis and candidate-selection rung: it does not install a replacement
inside Gemma or make a compression, speed, or model-quality claim.

## 1. Node coordinates

Let an activation row be \(h\), with calibration mean \(\mu\), activation
covariance \(C\), and activation-space empirical Fisher matrix \(F\). A linear
codec uses

\[
z = (h-\mu)E,\qquad
\widehat h_k = z_{:k}D_{:k}^{\mathsf T}+\mu.
\]

The full matrices satisfy

\[
ED^{\mathsf T}=I,
\]

so rank \(d\) is an explicit numerical identity control. Rank zero maps valid
positions to the pooled calibration mean.

Three full-width codec families are available:

- **Native Fisher control:** \(E=D=V_F\), in descending Fisher-eigenvalue
  order.
- **Variance-weighted Fisher:** keep the same orthonormal Fisher vectors, but
  order mode \(i\) by
  \(\lambda_i\,\operatorname{Var}[(h-\mu)v_i]\).
- **Generalized Fisher:** regularize \(C\) and \(F\), diagonalize
  \(C_{\rm reg}^{1/2}F_{\rm reg}C_{\rm reg}^{1/2}\), and use the dual pair

  \[
  E=C_{\rm reg}^{-1/2}V,\qquad
  D=C_{\rm reg}^{1/2}V.
  \]

The variance-weighted method is the smallest correction to native Fisher
ordering. The generalized method is more expressive: its prefixes are
activation-aware subspaces, rather than merely a reordering of the original
Fisher eigenvectors. Both are deterministic and fitted only from calibration
statistics.

`StreamingActivationCovariance` accumulates \(C\) in CPU float64 without
retaining activation rows. Rank-deficient generalized fits require explicit
positive covariance and Fisher eigenvalue floors. The artifact records those
floors, condition numbers, and the full-rank identity residual. This exact
covariance path is bounded in sequence count but still stores
\(O(d^2)\) values per activation site; a sketched or structured covariance is
needed at substantially larger residual widths.

## 2. Signed forward edges

For an input mode \(i\), the forward probe perturbs one valid source position
in decoder direction \(D_{\rm in}[:,i]\). It executes the real frozen block
under a JVP and projects the output tangent through
\(E_{\rm out}[:,:q]\). This directly measures

\[
\frac{\partial z_{{\rm out},t}}
     {\partial z_{{\rm in},s}}.
\]

Edges are pooled by exact logical lag \(t-s\):

- lag zero is a same-position edge;
- positive lag is a read from an earlier input position;
- negative lag is measured separately as causal leakage;
- causal edges beyond the requested lag window are measured as omitted past
  energy.

The probe stores both the signed mean and RMS. RMS is diagnostic magnitude,
not an executable edge: replacing a signed Jacobian with RMS would destroy
cancellation and direction.

The moment accounting also separates:

- energy explained by the best constant signed edge at each lag; and
- within-lag variation \(E[J^2]-E[J]^2\).

Both quantities are reported per lag and with lag zero excluded. That
separation matters because a large same-position residual-like edge can make
the aggregate look stationary while the much smaller positive-lag
cross-token edges vary strongly. If positive-lag variation is small, one
stationary edge map is a reasonable first compiler target. If it is large,
prompts or token states are invoking different local linearizations, and a
small causal router plus several signed experts becomes the next experiment.
The current implementation measures this need; it does not yet fit the gated
mixture.

These energy fractions depend on the coordinate gauge. In particular, a
generalized codec can rescale dual encoder and decoder directions. Compare
constant-versus-varying energy only inside one locked codec, not as a scalar
ranking between unrelated codec families. The outer artifact must bind the
probe to the exact codec states and regularization floors used to produce it.

## 3. Fisher-weighted merging

For output position \(t\), collect only its legal input prefix:

\[
J_t =
\begin{bmatrix}
J_{t,0} & J_{t,1} & \cdots & J_{t,t}
\end{bmatrix}.
\]

Version 1 uses one activation covariance block \(C_s\) per source position and
one output Fisher block \(F_t\). It forms

\[
M_t =
F_t^{1/2}
\begin{bmatrix}
J_{t,0}C_0^{1/2} &
\cdots &
J_{t,t}C_t^{1/2}
\end{bmatrix}
\]

and computes an independent SVD for every \(t\):

\[
M_t=U_t\Sigma_tV_t^{\mathsf T}.
\]

Keeping rank \(r_t\) minimizes the weighted local linear error

\[
\left\|
F_t^{1/2}(J_t-\widehat J_t)
\operatorname{blockdiag}
(C_0^{1/2},\ldots,C_t^{1/2})
\right\|_F^2.
\]

The optimal error is exactly the discarded singular-value energy. The
executor stores two signed factors:

\[
A_t=F_t^{-1/2}U_{t,:r_t}\Sigma_{t,:r_t}^{1/2},
\]

\[
B_t=\Sigma_{t,:r_t}^{1/2}V_{t,:r_t}^{\mathsf T}
\operatorname{blockdiag}
(C_0^{-1/2},\ldots,C_t^{-1/2}).
\]

It evaluates

\[
\widehat y_t
=\mu_{y,t}
+A_tB_t
\begin{bmatrix}
x_0-\mu_{x,0}\\
\vdots\\
x_t-\mu_{x,t}
\end{bmatrix}.
\]

Each target is factored independently. This is essential: a single flattened
SVD across all target positions could reintroduce a parameter path from a
future input into an earlier output. Here, output \(t\) has no storage slot
for \(s>t\), so the triangular causal restriction is structural.

Singular covariance or Fisher blocks use support pseudoinverses. A full-rank
factor then reconstructs \(P_FJP_C\), the part visible on both measured PSD
supports. When all blocks are positive definite, full rank reconstructs the
signed Jacobian.

## 4. What can be optimized

For a sequence of length \(T\), input width \(d_i\), and output width \(d_o\),
the unfactored lower-triangular Jacobian has

\[
\frac{T(T+1)}{2}d_id_o
\]

signed coefficients and the same reference multiply-accumulate count.

The factored executor stores and applies

\[
\sum_{t=0}^{T-1}
r_t\big((t+1)d_i+d_o\big)
\]

signed coefficients. Input/output affine means are reported separately.
`CausalWeightedJacobianResult` exposes both counts, their factored-to-dense
ratios, per-edge weighted energy, and the minimal rank at each prefix needed
to retain a requested weighted-energy fraction. It also reports full-reference
and retained-approximation energy by exact lag, making it visible when a
nominal compression curve is almost entirely a lag-zero result.

These are analytic graph counts, not kernel latency. A factor can reduce
coefficients and multiplies while still losing wall-clock time to small
matrix launches, indexing, or memory traffic. Runtime speed requires a
lowering and a same-device benchmark after behavioral validation.

For the bounded lag pilot, the \(T=L+1\) Toeplitz expansion is a synthetic
unshared dense reference. A natural lag-shared implementation stores only
\((L+1)d_id_o\) edge values, so the factor-to-dense ratio must not be reported
as savings over that shared representation, over the source transformer's
parameters, or over its FLOPs. The pilot also replicates pooled \(C\) and
\(F\) at every synthetic position and omits position dependence and
cross-position metric blocks.

## 5. Split protocol for Gemma

The opt-in Gemma rung uses four fixed prompt splits:

- **Calibration A:** fit Fisher, activation covariance, codec families, and
  their regularization provenance.
- **Calibration B:** evaluate the predeclared rank/family schedule, require a
  passing full-width behavioral identity for every family, and lock the first
  passing reduced candidate in canonical rank-then-family order.
- **Validation:** evaluate only the locked candidate, the locked family's
  full-width identity, and the native full-width identity, deduplicating the
  identities for a native lock.
- **Test:** parse and hash only; never tokenize, run, score, or select on it.

The optional forward-JVP pilot runs after selection and is deliberately
bounded by mode count, sequence count, and lag. Its signed mean lag maps can
be expanded into a small causal reference Jacobian and passed through the
weighted factorizer. That demonstrates the complete data path, but it remains
a stationary, calibration-local linearization—not a validated nonlinear
Gemma replacement. Factor energy belongs to this synthetic Toeplitz metric;
it is not the probe's observed-pair RMS energy.

## 6. Remaining acceptance gates

Before calling a Gemma block compiled, all of the following still have to
pass:

1. stable codec selection on representative, variable-length data;
2. sufficiently low within-lag Jacobian variation, or a separately selected
   causal gated mixture;
3. nonlinear/discarded-mode completion where a linear local map is
   insufficient;
4. local block-output equivalence on untouched examples;
5. internal-trajectory and end-to-end NLL/top-1 gates;
6. variable-length, padding, causality, and eventually decode/cache gates;
7. authenticated serialization, source fallback, backend lowering, and a
   same-device latency benchmark;
8. confirmation on genuinely fresh validation data, followed by one
   reserved-test evaluation after every choice and threshold is locked.

Cross-position covariance is also omitted in version 1. That approximation
makes edge energy attributable and keeps the reference small, but it should
not be confused with the full sequence covariance metric.

## Code map

- `linear_codec.py`: streaming activation covariance and all three codecs.
- `modal_ablation.py`: joint or singleton codec-prefix intervention curves.
- `jacobian_probe.py`: exact bounded forward JVPs and lag/regime accounting.
- `weighted_jacobian.py`: causal weighted SVD, rank selection, strict state,
  analytic counts, and signed executor.
- `gemma3_weighted_jacobian_experiment.py`: split-safe opt-in Gemma
  orchestration and ignored artifacts.

Focused synthetic tests cover full-rank codec identity, generalized dual
bases, half-precision promotion, exact toy-block causality, energy accounting,
PSD support behavior, SVD-tail optimality, future-edge rejection, factor
counts, energy-selected ranks, and strict weights-only round trips.
