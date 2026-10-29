"""Calibration and differentiable fitting for the MLX Qwen graph runtime.

Captures, sufficient statistics, and optimization use MLX arrays. Factorizations
use MLX's float32 CPU linear algebra by default; ``numpy_float64`` is an explicit
host precision reference, never an implicit fallback. Importing this module does
not import MLX. These are development fitting tools, not qualification gates.
Set MLX_ENABLE_TF32=0 before importing MLX for full-precision float32 Metal
matrix products. Reports record the environment setting; it is not changed
after the backend has initialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np


def _mx():
    return importlib.import_module("mlx.core")


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or (
        float(value) < 0 if zero else float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _rank(value: int, maximum: int, name: str) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _rows(value: Any, name: str):
    mx = _mx()
    rows = mx.array(value, dtype=mx.float32)
    if rows.ndim != 2 or min(rows.shape) == 0:
        raise ValueError(f"{name} must be a nonempty row matrix")
    if not bool(mx.all(mx.isfinite(rows)).item()):
        raise ValueError(f"{name} must contain finite values")
    return rows


def _weights(weights: Any, count: int):
    mx = _mx()
    w = mx.ones((count,), dtype=mx.float32) if weights is None else mx.array(weights, dtype=mx.float32)
    if w.shape != (count,) or not bool(mx.all(mx.isfinite(w) & (w >= 0)).item()):
        raise ValueError("row weights must be finite, nonnegative, and match rows")
    mass = float(mx.sum(w).item())
    if mass <= 0:
        raise ValueError("row weights need positive total mass")
    return w / mass


def _batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    mx = _mx()
    ids = mx.array(batch["input_ids"], dtype=mx.int32)
    if ids.ndim != 2 or min(ids.shape) == 0:
        raise ValueError("input_ids must be a nonempty batch of sequences")
    mask = mx.array(batch.get("attention_mask", mx.ones_like(ids)), dtype=mx.bool_)
    if mask.shape != ids.shape or not bool(mx.any(mask).item()):
        raise ValueError("attention_mask must match input_ids and contain valid tokens")
    result = {"input_ids": ids, "attention_mask": mask}
    if "position_ids" in batch:
        result["position_ids"] = mx.array(batch["position_ids"], dtype=mx.int32)
    return result


def _select_valid(values: Any, mask: Any):
    mx = _mx()
    if values.shape[:2] != mask.shape:
        raise ValueError("capture and attention mask shapes disagree")
    indices = mx.array(np.flatnonzero(np.asarray(mask).reshape(-1)), dtype=mx.int32)
    return mx.take(values.reshape((-1, values.shape[-1])), indices, axis=0)


def _next_token_targets(runtime, raw, batch):
    valid = np.asarray(batch["attention_mask"])
    ids = np.asarray(batch["input_ids"])
    labels = np.asarray(raw.get("labels", ids))
    if labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be integer and match input_ids")
    lexical = next(runtime.parameters[name] for name in ("lexical.head_weight", "lexical.weight", "lexical.codebook") if name in runtime.parameters)
    for b, t in np.argwhere(valid[:, :-1] & valid[:, 1:]):
        target = int(labels[b, t + 1])
        if target == -100:
            continue
        if not 0 <= target < lexical.shape[0]:
            raise ValueError("supervised label is outside the runtime vocabulary")
        yield int(b), int(t), target


@dataclass(frozen=True)
class MLXMoments:
    mean: Any
    covariance: Any
    second_moment: Any
    row_count: int


def weighted_moments(rows: Any, row_weights: Any = None) -> MLXMoments:
    """Population moments; weights are normalized to sum to one."""
    mx = _mx()
    x = _rows(rows, "rows")
    w = _weights(row_weights, x.shape[0])
    mean = mx.sum(x * w[:, None], axis=0)
    centered = x - mean
    covariance = centered.T @ (centered * w[:, None])
    second = x.T @ (x * w[:, None])
    mx.eval(mean, covariance, second)
    return MLXMoments(mean, covariance, second, x.shape[0])


def capture_runtime_rows(runtime: Any, batches: Sequence[Mapping[str, Any]], sites: Sequence[str]) -> dict[str, Any]:
    """Capture real runtime boundaries, excluding padding before concatenation.

    The caller chooses the calibration batches. No evaluation examples are
    selected or loaded by this function. Captured arrays are stop-gradient data.
    """
    mx = _mx()
    names = tuple(sites)
    if not names or len(set(names)) != len(names) or not batches:
        raise ValueError("capture requires batches and unique site names")
    collected: dict[str, list[Any]] = {name: [] for name in names}
    for raw in batches:
        batch = _batch(raw)
        output = runtime(**batch, capture=names, return_logits=False)
        for name in names:
            if name not in output.captures:
                raise ValueError(f"runtime did not capture {name}")
            values = _select_valid(output.captures[name], batch["attention_mask"])
            collected[name].append(mx.stop_gradient(values))
        mx.eval([v[-1] for v in collected.values()])
    result = {name: mx.concatenate(values, axis=0) for name, values in collected.items()}
    mx.eval(result)
    return result


def capture_empirical_token_fisher(
    runtime: Any,
    batches: Sequence[Mapping[str, Any]],
    sites: Sequence[str],
    *,
    max_tokens: int = 64,
    full_matrix: bool = False,
) -> dict[str, Any]:
    """Exact observed-next-token score VJPs, bounded by ``max_tokens``.

    Each supervised next-token NLL is differentiated separately, avoiding the
    cross-token terms produced by squaring a summed-loss gradient. For each
    site we pool squared gradients over valid boundary positions, then average
    over objectives. Cross-position Fisher blocks are intentionally omitted.
    This is empirical data Fisher, not an expectation over model samples.
    """
    mx = _mx()
    _rank(max_tokens, 2**31 - 1, "max_tokens")
    names = tuple(sites)
    if not names or len(set(names)) != len(names):
        raise ValueError("Fisher capture requires unique site names")
    diagonal: dict[str, Any] = {}
    matrices: dict[str, Any] = {}
    count = 0
    for raw in batches:
        batch = _batch(raw)
        captured = runtime(**batch, capture=names, return_logits=False).captures
        if any(name not in captured for name in names):
            raise ValueError("requested Fisher site was not captured")
        perturbations = {name: mx.zeros_like(captured[name]) for name in names}
        for name in names:
            diagonal.setdefault(name, mx.zeros((captured[name].shape[-1],), dtype=mx.float32))
            if full_matrix:
                width = captured[name].shape[-1]
                matrices.setdefault(name, mx.zeros((width, width), dtype=mx.float32))
        for b, t, target in _next_token_targets(runtime, raw, batch):

            def token_nll(delta):
                logits = runtime(**batch, perturbations=delta).logits[b, t].astype(mx.float32)
                return mx.logsumexp(logits) - logits[target]

            grads = mx.grad(token_nll)(perturbations)
            for name, gradient in grads.items():
                rows = _select_valid(gradient, batch["attention_mask"]).astype(mx.float32)
                diagonal[name] = diagonal[name] + mx.sum(rows * rows, axis=0)
                if full_matrix:
                    matrices[name] = matrices[name] + rows.T @ rows
            count += 1
            mx.eval(diagonal, matrices)
            mx.clear_cache()
            if count >= max_tokens:
                break
        if count >= max_tokens:
            break
    if not count:
        raise ValueError("Fisher capture requires at least one supervised next token")
    result = {
        "diagonal": {name: value / count for name, value in diagonal.items()},
        "matrices": {name: value / count for name, value in matrices.items()},
        "objective_count": count,
        "max_tokens": max_tokens,
        "kind": "observed_next_token_score_fisher_pooled_boundary_positions",
        "cross_position_blocks_included": False,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "backend_default"),
    }
    mx.eval(result["diagonal"], result["matrices"])
    return result


def capture_parameter_fisher(
    runtime: Any,
    batches: Sequence[Mapping[str, Any]],
    parameter_names: Sequence[str],
    *,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Observed-token diagonal parameter Fisher for an explicit small leaf set.

    The returned ``diagonal`` can be passed directly to ``optimize_trajectory``.
    Gradients are computed separately per token and averaged after squaring;
    unselected parameters are fixed and never receive gradient buffers.
    """
    mx = _mx()
    _rank(max_tokens, 2**31 - 1, "max_tokens")
    names = tuple(parameter_names)
    if not names or len(set(names)) != len(names) or any(name not in runtime.parameters for name in names):
        raise ValueError("parameter_names must name unique existing runtime parameters")
    selected = {name: runtime.parameters[name].astype(mx.float32) for name in names}
    deployment_dtypes = {name: runtime.parameters[name].dtype for name in names}
    diagonal = {name: mx.zeros_like(value) for name, value in selected.items()}
    count = 0
    for raw in batches:
        batch = _batch(raw)
        for b, t, target in _next_token_targets(runtime, raw, batch):
            def token_nll(parameters):
                deployed = {name: value.astype(deployment_dtypes[name]) for name, value in parameters.items()}
                logits = runtime(**batch, parameters={**runtime.parameters, **deployed}).logits[b, t].astype(mx.float32)
                return mx.logsumexp(logits) - logits[target]

            gradients = mx.grad(token_nll)(selected)
            diagonal = {name: diagonal[name] + gradient * gradient for name, gradient in gradients.items()}
            mx.eval(diagonal)
            count += 1
            if count >= max_tokens:
                break
        if count >= max_tokens:
            break
    if not count:
        raise ValueError("parameter Fisher requires at least one supervised next token")
    diagonal = {name: value / count for name, value in diagonal.items()}
    mx.eval(diagonal)
    return {
        "diagonal": diagonal, "objective_count": count, "max_tokens": max_tokens,
        "kind": "observed_next_token_diagonal_parameter_fisher",
        "selected_parameters": list(names),
        "deployment_dtypes": {name: str(dtype) for name, dtype in deployment_dtypes.items()},
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "backend_default"),
    }


class _Solver:
    def __init__(self, backend: str):
        if backend not in {"mlx", "numpy_float64"}:
            raise ValueError("solver must be mlx or numpy_float64")
        self.backend = backend

    def solve(self, gram, rhs):
        mx = _mx()
        if self.backend == "numpy_float64":
            result = np.linalg.solve(np.asarray(gram).astype(np.float64), np.asarray(rhs).astype(np.float64))
            return mx.array(result.astype(np.float32))
        return mx.linalg.solve(gram, rhs, stream=mx.cpu)

    def eigh(self, matrix):
        mx = _mx()
        symmetric = (matrix + matrix.T) * 0.5
        if self.backend == "numpy_float64":
            values, vectors = np.linalg.eigh(np.asarray(symmetric).astype(np.float64))
            return mx.array(values.astype(np.float32)), mx.array(vectors.astype(np.float32))
        return mx.linalg.eigh(symmetric, stream=mx.cpu)

    def svd(self, matrix):
        mx = _mx()
        if self.backend == "numpy_float64":
            u, s, vh = np.linalg.svd(np.asarray(matrix).astype(np.float64), full_matrices=False)
            return tuple(mx.array(x.astype(np.float32)) for x in (u, s, vh))
        return mx.linalg.svd(matrix, stream=mx.cpu)


def _leading_basis(gram, rank: int, solver: _Solver):
    mx = _mx()
    _, eigenvectors = solver.eigh(gram)
    basis = eigenvectors[:, -rank:][:, ::-1]
    pivot = mx.argmax(mx.abs(basis), axis=0)
    signs = mx.where(mx.take_along_axis(basis, pivot[None, :], axis=0) < 0, -1.0, 1.0)
    return basis * signs


def anchored_ridge(
    features: Any,
    targets: Any,
    *,
    row_weights: Any = None,
    anchor: Any = None,
    relative_ridge: float = 0.01,
    solver: str = "mlx",
):
    """Trace-relative weighted ridge, returning a [features, outputs] matrix.

    ``numpy_float64`` accumulates and solves this regression on the host in
    float64, then explicitly casts its result to the runtime's float32.
    """
    mx = _mx()
    backend = _Solver(solver)
    ridge = _positive(relative_ridge, "relative_ridge")
    x, y = _rows(features, "features"), _rows(targets, "targets")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature and target row counts differ")
    w = _weights(row_weights, x.shape[0])
    initial = mx.zeros((x.shape[1], y.shape[1])) if anchor is None else _rows(anchor, "anchor")
    if initial.shape != (x.shape[1], y.shape[1]):
        raise ValueError("anchor shape does not match regression")
    if solver == "numpy_float64":
        xn, yn, wn, an = [np.asarray(v).astype(np.float64) for v in (x, y, w, initial)]
        gram = xn.T @ (xn * wn[:, None])
        penalty = ridge * max(float(np.trace(gram)) / x.shape[1], 1e-12)
        fitted = np.linalg.solve(gram + penalty * np.eye(x.shape[1]), xn.T @ (yn * wn[:, None]) + penalty * an)
        result = mx.array(fitted.astype(np.float32))
    else:
        gram = x.T @ (x * w[:, None])
        penalty = ridge * mx.maximum(mx.sum(mx.diag(gram)) / x.shape[1], 1e-12)
        result = backend.solve(gram + penalty * mx.eye(x.shape[1]), x.T @ (y * w[:, None]) + penalty * initial)
    mx.eval(result)
    if not bool(mx.all(mx.isfinite(result)).item()):
        raise FloatingPointError("ridge solve produced nonfinite coefficients")
    return result


@dataclass(frozen=True)
class MLXFitResult:
    runtime: Any
    updates: Mapping[str, Any]
    report: Mapping[str, Any]


def _require_no_incident_transports(runtime: Any, site: str) -> None:
    if any(edge["source"] == site or edge["target"] == site for edge in runtime.config.get("transports", ())):
        raise ValueError("refitting this site would change an existing transport vocabulary; refit transports jointly first")


def fit_reduced_attention(
    runtime: Any,
    layer_index: int,
    captures: Mapping[str, Any],
    *,
    input_rank: int,
    output_rank: int,
    row_weights: Any = None,
    relative_ridge: float = 0.01,
    solver: str = "mlx",
) -> MLXFitResult:
    """Fit usable Q/K/V and O factors from normalized inputs and contexts.

    The input metric uses Q/K norm gains and grouped-query V/O sensitivity;
    it is an activation-weighted operator pullback proxy, not measured Fisher.
    Output approximation minimizes captured context-weighted operator error.
    Qwen has no post-attention RMSNorm, so no Gemma delta normalization is used.
    Fitted coefficients are exported in the source attention's deployment dtype.
    """
    mx = _mx()
    backend = _Solver(solver)
    ridge = _positive(relative_ridge, "relative_ridge")
    prefix = f"attention.{layer_index}."
    _require_no_incident_transports(runtime, prefix[:-1])
    p = runtime.parameters
    deployment_dtype = p[prefix + "q_weight"].dtype
    if prefix + "input_basis" in p:
        raise ValueError("attention fitting requires an exact source attention")
    x = _rows(captures[prefix + "normalized"], "normalized inputs")
    contexts = _rows(captures[prefix + "context"], "attention contexts")
    if len(x) != len(contexts):
        raise ValueError("input/context capture rows must align")
    q, k, v, out = [p[prefix + key].astype(mx.float32) for key in ("q_weight", "k_weight", "v_weight", "output_decoder_weight")]
    _rank(input_rank, x.shape[1], "input_rank")
    _rank(output_rank, min(out.shape), "output_rank")
    weights = _weights(row_weights, x.shape[0])
    config = runtime.layers[layer_index]["attention"]
    head_dim = int(config["head_dimension"])

    def qk_effective(weight, key):
        gain = p.get(prefix + key)
        if gain is None:
            return weight
        spec = config.get("qk_norm", {})
        gain = gain + 1 if spec.get("scale_parameterization") == "unit_offset" else gain
        scale = mx.maximum(mx.abs(gain), 1e-3)
        if key == "q_norm_weight" and config.get("output_gate", False):
            scale = mx.concatenate((scale, mx.ones_like(scale)), axis=0)
        return weight * mx.tile(scale, weight.shape[0] // scale.shape[0])[:, None]

    qe, ke = qk_effective(q, "q_norm_weight"), qk_effective(k, "k_norm_weight")
    repeats = int(config["query_heads"]) // int(config["key_value_heads"])
    # GQA repeats each KV head consecutively, not the entire KV matrix.
    repeated_v = mx.repeat(v.reshape((-1, head_dim, v.shape[1])), repeats, axis=0).reshape((out.shape[1], v.shape[1]))
    ve = out @ repeated_v
    grams = [w.T @ w for w in (qe, ke, ve)]
    metric = sum(scale * g / mx.maximum(mx.sum(mx.diag(g)), 1e-12) for scale, g in zip((1.0, 1.0, 2.0), grams))
    cov = x.T @ (x * weights[:, None])
    cov = cov + ridge * mx.maximum(mx.sum(mx.diag(cov)) / x.shape[1], 1e-12) * mx.eye(x.shape[1])
    basis = _leading_basis(cov @ metric @ cov, input_rank, backend)
    reconstruction = anchored_ridge(x @ basis, x, row_weights=weights, anchor=basis.T, relative_ridge=ridge, solver=solver)

    cc = contexts.T @ (contexts * weights[:, None])
    cc = cc + ridge * mx.maximum(mx.sum(mx.diag(cc)) / cc.shape[0], 1e-12) * mx.eye(cc.shape[0])
    values, vectors = backend.eigh(cc)
    floor = mx.maximum(mx.max(values) * 1e-7, 1e-12)
    roots = mx.sqrt(mx.maximum(values, floor))
    square_root = (vectors * roots) @ vectors.T
    inverse_root = (vectors / roots) @ vectors.T
    u, singular, vh = backend.svd(square_root @ out.T)
    root_s = mx.sqrt(mx.maximum(singular[:output_rank], 0))
    encoder = (inverse_root @ (u[:, :output_rank] * root_s)).T
    decoder = (root_s[:, None] * vh[:output_rank, :]).T
    updates = {
        prefix + "input_basis": basis,
        prefix + "q_weight": q @ reconstruction.T,
        prefix + "k_weight": k @ reconstruction.T,
        prefix + "v_weight": v @ reconstruction.T,
        prefix + "output_encoder_weight": encoder,
        prefix + "output_decoder_weight": decoder,
    }
    updates = {name: value.astype(deployment_dtype) for name, value in updates.items()}
    mx.eval(updates)
    approximation = ((contexts.astype(deployment_dtype) @ updates[prefix + "output_encoder_weight"].T)
                     @ updates[prefix + "output_decoder_weight"].T).astype(mx.float32)
    reference = (contexts.astype(deployment_dtype) @ p[prefix + "output_decoder_weight"].T).astype(mx.float32)
    error = mx.sqrt(mx.sum((approximation - reference) ** 2) / mx.maximum(mx.sum(reference ** 2), 1e-12))
    return MLXFitResult(runtime.with_parameters(updates), updates, {
        "kind": "activation_weighted_attention_operator_pullback",
        "solver": solver, "capture_rows": x.shape[0],
        "input_rank": input_rank, "output_rank": output_rank,
        "context_output_relative_error": float(error.item()),
        "moment_dtype": "float32",
        "factorization_dtype": "float64" if solver == "numpy_float64" else "float32",
        "regression_accumulation_dtype": "float64" if solver == "numpy_float64" else "float32",
        "deployment_dtype": str(deployment_dtype),
        "heldout_confirmation": False,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "backend_default"),
    })


def fit_reduced_deltanet(
    runtime: Any,
    layer_index: int,
    captures: Mapping[str, Any],
    *,
    input_rank: int,
    output_rank: int,
    row_weights: Any = None,
    relative_ridge: float = 0.01,
    solver: str = "mlx",
) -> MLXFitResult:
    """Fit input/output bottlenecks around a GatedDeltaNet block (Play 1).

    Reduces ``in_proj_qkv`` and ``in_proj_z`` through a shared activation-weighted
    operator pullback basis while leaving the tiny decay/write-rate projections
    (``in_proj_a``, ``in_proj_b``) unreduced so exponential state decay does not
    accumulate rank-truncation drift across long recurrences. Output projection
    ``out_proj`` is factored via gated-context-covariance whitened SVD.
    """
    mx = _mx()
    backend = _Solver(solver)
    ridge = _positive(relative_ridge, "relative_ridge")
    if not 0 <= layer_index < len(runtime.layers) or "deltanet" not in runtime.layers[layer_index]:
        raise ValueError("deltanet fitting requires a deltanet layer index")
    prefix = f"deltanet.{layer_index}."
    _require_no_incident_transports(runtime, prefix[:-1])
    p = runtime.parameters
    quantized = prefix + "qkv_weight.scales" in p
    if quantized:
        quant = runtime.config.get("quantization") or {}
        q_gs, q_bits, q_mode = (int(quant.get("group_size", 64)), int(quant.get("bits", 4)),
                                str(quant.get("mode", "affine")))
        deployment_dtype = p[prefix + "qkv_weight.scales"].dtype

        def dense(key):
            return mx.dequantize(p[key], scales=p[key + ".scales"], biases=p.get(key + ".biases"),
                                 group_size=q_gs, bits=q_bits, mode=q_mode).astype(mx.float32)
    else:
        deployment_dtype = p[prefix + "qkv_weight"].dtype

        def dense(key):
            return p[key].astype(mx.float32)

    if prefix + "input_basis" in p or prefix + "output_encoder_weight" in p:
        raise ValueError("deltanet fitting requires an exact source deltanet")
    x = _rows(captures[prefix + "normalized"], "normalized inputs")
    contexts = _rows(captures[prefix + "context"], "deltanet contexts")
    if len(x) != len(contexts):
        raise ValueError("input/context capture rows must align")
    qkv, z, out = [dense(prefix + key) for key in ("qkv_weight", "z_weight", "output_decoder_weight")]
    _rank(input_rank, x.shape[1], "input_rank")
    _rank(output_rank, min(out.shape), "output_rank")
    weights = _weights(row_weights, x.shape[0])
    dc = runtime.layers[layer_index]["deltanet"]
    k_total = int(dc["key_heads"]) * int(dc["key_head_dimension"])
    vh = int(dc["value_heads"])
    q_w, k_w, v_w = qkv[:k_total], qkv[k_total : 2 * k_total], qkv[2 * k_total :]
    norm_w = p.get(prefix + "norm_weight")
    if norm_w is not None:
        gain = norm_w.astype(mx.float32)
        if dc.get("norm", {}).get("scale_parameterization") == "unit_offset":
            gain = gain + 1.0
        gain_v = mx.tile(mx.maximum(mx.abs(gain), 1e-3), vh)[:, None]
        v_eff = v_w * gain_v
        z_eff = z * gain_v
    else:
        v_eff = v_w
        z_eff = z
    grams = [w.T @ w for w in (q_w, k_w, v_eff, z_eff)]
    metric = sum(scale * g / mx.maximum(mx.sum(mx.diag(g)), 1e-12) for scale, g in zip((1.0, 1.0, 2.0, 1.0), grams))
    cov = x.T @ (x * weights[:, None])
    cov = cov + ridge * mx.maximum(mx.sum(mx.diag(cov)) / x.shape[1], 1e-12) * mx.eye(x.shape[1])
    basis = _leading_basis(cov @ metric @ cov, input_rank, backend)
    if input_rank == x.shape[1]:
        reconstruction = basis.T
    else:
        reconstruction = anchored_ridge(
            x @ basis, x, row_weights=weights, anchor=basis.T, relative_ridge=ridge, solver=solver
        )

    cc = contexts.T @ (contexts * weights[:, None])
    cc = cc + ridge * mx.maximum(mx.sum(mx.diag(cc)) / cc.shape[0], 1e-12) * mx.eye(cc.shape[0])
    out_gram = out @ cc @ out.T
    v_out = _leading_basis(out_gram, output_rank, backend)
    encoder_raw = v_out.T @ out
    mode_scale = mx.sqrt(mx.maximum(mx.sqrt(mx.sum(encoder_raw * encoder_raw, axis=1)), 1e-6))
    encoder = encoder_raw / mode_scale[:, None]
    decoder = v_out * mode_scale[None, :]
    if quantized and input_rank == x.shape[1]:
        updates = {}
        dense_updates = {
            prefix + "output_encoder_weight": encoder,
            prefix + "output_decoder_weight": decoder,
        }
    else:
        updates = {prefix + "input_basis": basis.astype(deployment_dtype)}
        dense_updates = {
            prefix + "qkv_weight": qkv @ reconstruction.T,
            prefix + "z_weight": z @ reconstruction.T,
            prefix + "output_encoder_weight": encoder,
            prefix + "output_decoder_weight": decoder,
        }
    for name, value in dense_updates.items():
        if quantized:
            if value.shape[-1] % q_gs:
                raise ValueError(f"reduced {name} input dimension must be a multiple of the quantization group_size")
            w_q, scales, biases = mx.quantize(value.astype(deployment_dtype),
                                              group_size=q_gs, bits=q_bits, mode=q_mode)
            updates[name], updates[name + ".scales"], updates[name + ".biases"] = w_q, scales, biases
        else:
            updates[name] = value.astype(deployment_dtype)
    mx.eval(updates)

    def deployed(key):
        if quantized:
            return mx.dequantize(updates[key], scales=updates[key + ".scales"],
                                 biases=updates[key + ".biases"],
                                 group_size=q_gs, bits=q_bits, mode=q_mode).astype(mx.float32)
        return updates[key].astype(mx.float32)

    approximation = ((contexts @ deployed(prefix + "output_encoder_weight").T)
                     @ deployed(prefix + "output_decoder_weight").T).astype(mx.float32)
    reference = (contexts @ out.T).astype(mx.float32)
    error = mx.sqrt(mx.sum((approximation - reference) ** 2) / mx.maximum(mx.sum(reference ** 2), 1e-12))
    return MLXFitResult(runtime.with_parameters(updates), updates, {
        "kind": "activation_weighted_deltanet_operator_pullback",
        "solver": solver, "capture_rows": x.shape[0],
        "input_rank": input_rank, "output_rank": output_rank,
        "context_output_relative_error": float(error.item()),
        "moment_dtype": "float32",
        "factorization_dtype": "float64" if solver == "numpy_float64" else "float32",
        "regression_accumulation_dtype": "float64" if solver == "numpy_float64" else "float32",
        "deployment_dtype": (f"quantized_{q_bits}bit_gs{q_gs}_{q_mode}" if quantized
                             else str(deployment_dtype)),
        "heldout_confirmation": False,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "backend_default"),
    })


def fit_deltanet_carrier_basis(
    runtime: Any,
    layer_index: int,
    captures: Mapping[str, Any],
    *,
    carrier_rank: int,
    fisher_diagonal: Mapping[str, Any] | None = None,
    row_weights: Any = None,
    relative_ridge: float = 1e-4,
    solver: str = "mlx",
) -> MLXFitResult:
    """Compress the GatedDeltaNet recurrent carrier S along its key/query axis (Play 2).

    Because every operation on the Dk axis of ``S in R^{B x Hv x Dv x Dk}`` is a
    linear contraction with ``k`` or ``q`` and decay ``g`` is scalar per V-head,
    an orthonormal basis ``U_h in R^{Dk x r}`` per K-head group commutes with the
    entire recurrence. Projecting ``q_tilde = q @ U_h`` and ``k_tilde = k @ U_h``
    runs the recurrent state natively in ``R^{B x Hv x Dv x r}``.
    """
    mx = _mx()
    backend = _Solver(solver)
    ridge = _positive(relative_ridge, "relative_ridge", zero=True)
    if not 0 <= layer_index < len(runtime.layers) or "deltanet" not in runtime.layers[layer_index]:
        raise ValueError("carrier basis fitting requires a deltanet layer index")
    prefix = f"deltanet.{layer_index}."
    p = runtime.parameters
    if prefix + "carrier_basis" in p:
        raise ValueError("carrier basis fitting requires an unreduced deltanet carrier")
    dc = runtime.layers[layer_index]["deltanet"]
    kh = int(dc["key_heads"])
    kd = int(dc["key_head_dimension"])
    _rank(carrier_rank, kd, "carrier_rank")
    q_rows = _rows(captures[prefix + "q"], "deltanet q")
    k_rows = _rows(captures[prefix + "k"], "deltanet k")
    if q_rows.shape != k_rows.shape or q_rows.shape[1] != kh * kd:
        raise ValueError("deltanet q/k capture shapes must match [rows, key_heads * key_head_dimension]")
    weights = _weights(row_weights, q_rows.shape[0])
    q = q_rows.reshape((-1, kh, kd))
    k = k_rows.reshape((-1, kh, kd))
    fq = fk = None
    if fisher_diagonal is not None:
        if prefix + "q" in fisher_diagonal:
            fq = mx.maximum(mx.array(fisher_diagonal[prefix + "q"], dtype=mx.float32).reshape((kh, kd)), 1e-12)
        if prefix + "k" in fisher_diagonal:
            fk = mx.maximum(mx.array(fisher_diagonal[prefix + "k"], dtype=mx.float32).reshape((kh, kd)), 1e-12)

    bases = []
    retained_num = 0.0
    retained_den = 0.0
    for h in range(kh):
        qh_rows, kh_rows = q[:, h, :], k[:, h, :]
        cov_q = qh_rows.T @ (qh_rows * weights[:, None])
        cov_k = kh_rows.T @ (kh_rows * weights[:, None])
        cov = cov_q + cov_k
        if fq is not None or fk is not None:
            wq = fq[h] if fq is not None else mx.ones((kd,), dtype=mx.float32)
            wk = fk[h] if fk is not None else mx.ones((kd,), dtype=mx.float32)
            wq = wq / mx.maximum(mx.mean(wq), 1e-12)
            wk = wk / mx.maximum(mx.mean(wk), 1e-12)
            gram = cov_q @ (wq[:, None] * cov_q) + cov_k @ (wk[:, None] * cov_k)
        else:
            gram = cov
        if ridge > 0:
            gram = gram + ridge * mx.maximum(mx.sum(mx.diag(gram)) / kd, 1e-12) * mx.eye(kd)
        u_h = _leading_basis(gram, carrier_rank, backend)
        bases.append(u_h)
        total_e = float(mx.sum(mx.diag(cov)).item())
        kept_e = float(mx.sum(mx.diag(u_h.T @ cov @ u_h)).item())
        retained_num += kept_e
        retained_den += max(total_e, 1e-12)

    deployment_dtype = p.get(prefix + "qkv_weight.scales", p[prefix + "qkv_weight"]).dtype
    carrier_basis = mx.stack(bases, axis=0).astype(deployment_dtype)
    mx.eval(carrier_basis)
    updates = {prefix + "carrier_basis": carrier_basis}
    updated_runtime = runtime.with_parameters(
        updates,
        layer_overrides={layer_index: {"deltanet": {"carrier_rank": carrier_rank}}},
    )
    return MLXFitResult(updated_runtime, updates, {
        "kind": "fisher_weighted_deltanet_carrier_basis" if (fq is not None or fk is not None) else "second_moment_deltanet_carrier_basis",
        "solver": solver,
        "capture_rows": int(q_rows.shape[0]),
        "key_heads": kh,
        "full_carrier_dimension": kd,
        "carrier_rank": carrier_rank,
        "compression_ratio": float(carrier_rank / kd),
        "retained_energy_fraction": float(retained_num / max(retained_den, 1e-12)),
        "fisher_weighted": bool(fq is not None or fk is not None),
        "deployment_dtype": str(deployment_dtype),
        "heldout_confirmation": False,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "backend_default"),
    })


def _swiglu_euler_jacobian_prior(runtime: Any, prefix: str, x: Any, weights: Any):
    """Homogeneous degree-2 Euler Jacobian prior for a source SwiGLU MLP."""
    mx = _mx()
    keys = (prefix + "gate_weight", prefix + "up_weight", prefix + "down_weight")
    if any(key not in runtime.parameters for key in keys):
        return None
    w_gate, w_up, w_down = (runtime.parameters[key].astype(mx.float32) for key in keys)
    g = x @ w_gate.T
    u = x @ w_up.T
    sig = mx.sigmoid(g)
    silu = g * sig
    silu_grad = sig * (1.0 + g * (1.0 - sig))
    mean_grad_u = mx.sum((silu_grad * u) * weights[:, None], axis=0)
    mean_silu = mx.sum(silu * weights[:, None], axis=0)
    prior = 0.5 * (w_down @ (mean_grad_u[:, None] * w_gate + mean_silu[:, None] * w_up))
    mx.eval(prior)
    return prior


def _inverse_target_norm_row_weights(y: Any):
    """Bound outlier-token leverage via clipped inverse squared target norm."""
    mx = _mx()
    s_sq = np.mean(np.asarray(y, dtype=np.float64) ** 2, axis=1)
    median = float(np.median(s_sq))
    clipped = np.clip(median / np.maximum(s_sq, 1e-4), 0.05, 5.0).astype(np.float32)
    return mx.array(clipped)


def fit_chart_mlp(
    runtime: Any,
    layer_index: int,
    captures: Mapping[str, Any],
    *,
    chart_count: int = 4,
    chart_rank: int = 16,
    generator_rank: int | None = None,
    row_weights: Any = None,
    relative_ridge: float = 0.01,
    temperature: float = 1.0,
    kmeans_iterations: int = 8,
    euler_jacobian_anchor: bool = True,
    center_intercept: bool = True,
    inverse_target_norm_weights: bool = True,
    solver: str = "mlx",
) -> MLXFitResult:
    """Fit affine + local chart residuals to real Qwen MLP operator captures.

    Charts emit hidden-space deltas directly; no dense identity decoder is
    allocated. Geometry and regression use only the supplied calibration rows
    and the layer's own source weights when available. Input centers are
    weighted k-means, bases are local activation covariance. When source
    SwiGLU weights are present and ``euler_jacobian_anchor`` is enabled, the
    centered affine base is anchored to the degree-2 Euler Jacobian prior
    ``0.5 * E[J(x)]`` rather than shrinking unexcited directions toward zero.
    Supplying Fisher row weights makes the regression Fisher-row-weighted; this
    function does not mislabel activation PCA as a Fisher eigenspace.
    Solves use float32/reference precision; exports retain the source MLP dtype.
    """
    mx = _mx()
    backend = _Solver(solver)
    _positive(relative_ridge, "relative_ridge")
    temperature = _positive(temperature, "temperature")
    _rank(kmeans_iterations, 10_000, "kmeans_iterations")
    prefix = f"mlp.{layer_index}."
    _require_no_incident_transports(runtime, prefix[:-1])
    source_weight = next(runtime.parameters[prefix + key] for key in ("affine_weight", "gate_weight") if prefix + key in runtime.parameters)
    deployment_dtype = source_weight.dtype
    x, y = _rows(captures[prefix + "normalized"], "MLP inputs"), _rows(captures[prefix + "output"], "MLP targets")
    if x.shape != y.shape:
        raise ValueError("Qwen MLP input/output captures must align in hidden space")
    n, hidden = x.shape
    _rank(chart_count, n, "chart_count")
    _rank(chart_rank, hidden, "chart_rank")
    generator_rank = chart_rank + 1 if generator_rank is None else generator_rank
    _rank(generator_rank, chart_rank + 1, "generator_rank")
    effective_row_weights = (
        _inverse_target_norm_row_weights(y)
        if row_weights is None and inverse_target_norm_weights
        else row_weights
    )
    weights = _weights(effective_row_weights, n)
    # Deterministic farthest-point seeds, with weighted mean as first reference.
    mean = mx.sum(x * weights[:, None], axis=0)
    first = int(mx.argmax(mx.sum((x - mean) ** 2, axis=1) * weights).item())
    centers = [x[first]]
    nearest = mx.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, chart_count):
        chosen = int(mx.argmax(nearest * weights).item())
        centers.append(x[chosen])
        nearest = mx.minimum(nearest, mx.sum((x - centers[-1]) ** 2, axis=1))
    centers = mx.stack(centers)
    for _ in range(kmeans_iterations):
        distance = mx.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        assignment = mx.argmin(distance, axis=1)
        updated = []
        for c in range(chart_count):
            wc = weights * (assignment == c)
            mass = mx.sum(wc)
            updated.append(mx.where(mass > 1e-12, mx.sum(x * wc[:, None], axis=0) / mx.maximum(mass, 1e-12), centers[c]))
        centers = mx.stack(updated)
    displacement = x[:, None, :] - centers[None, :, :]
    distance = mx.sum(displacement ** 2, axis=-1)
    assignment = mx.argmin(distance, axis=1)
    global_moments = weighted_moments(x, weights)
    global_scale = mx.maximum(mx.sum(mx.diag(global_moments.covariance)), 1e-8)
    scales, bases = [], []
    for c in range(chart_count):
        wc = weights * (assignment == c)
        mass = mx.sum(wc)
        local = displacement[:, c, :]
        local_cov = local.T @ (local * wc[:, None]) / mx.maximum(mass, 1e-12)
        # Empty or tiny charts retain a global covariance prior.
        local_cov = local_cov + 0.05 * global_moments.covariance
        bases.append(_leading_basis(local_cov, chart_rank, backend))
        scale = mx.sum(distance[:, c] * wc) / mx.maximum(mass, 1e-12)
        scales.append(mx.sqrt(mx.maximum(scale, global_scale * 0.01)))
    bases, scales = mx.stack(bases), mx.stack(scales)
    membership = mx.softmax(-0.5 * distance / (scales[None, :] ** 2 * temperature), axis=-1)
    coordinates = mx.einsum("nch,chr->ncr", displacement, bases)
    augmented = mx.concatenate((mx.ones((n, chart_count, 1)), coordinates), axis=-1)
    encoders = mx.broadcast_to(mx.eye(chart_rank + 1)[:, :generator_rank], (chart_count, chart_rank + 1, generator_rank))
    features = (membership[:, :, None] * mx.einsum("ncr,crg->ncg", augmented, encoders)).reshape((n, -1))
    euler_prior = _swiglu_euler_jacobian_prior(runtime, prefix, x, weights) if euler_jacobian_anchor else None
    if center_intercept:
        x_mean = mx.sum(x * weights[:, None], axis=0)
        y_mean = mx.sum(y * weights[:, None], axis=0)
        affine_linear = anchored_ridge(
            x - x_mean,
            y - y_mean,
            row_weights=weights,
            anchor=None if euler_prior is None else euler_prior.T,
            relative_ridge=relative_ridge,
            solver=solver,
        )
        affine_weight = affine_linear.T
        affine_bias = y_mean - x_mean @ affine_linear
        residual = y - (x @ affine_linear + affine_bias)
    else:
        affine_features = mx.concatenate((x, mx.ones((n, 1))), axis=1)
        affine_anchor = (
            None
            if euler_prior is None
            else mx.concatenate((euler_prior.T, mx.zeros((1, hidden))), axis=0)
        )
        affine = anchored_ridge(
            affine_features,
            y,
            row_weights=weights,
            anchor=affine_anchor,
            relative_ridge=relative_ridge,
            solver=solver,
        )
        affine_weight, affine_bias = affine[:-1].T, affine[-1]
        residual = y - affine_features @ affine
    decoded = anchored_ridge(features, residual, row_weights=weights, relative_ridge=relative_ridge, solver=solver)
    updates = {
        prefix + "chart_centers": centers, prefix + "chart_bases": bases,
        prefix + "chart_distance_scales": scales,
        prefix + "local_generator_encoders": encoders,
        prefix + "chart_mode_decoders": decoded.reshape((chart_count, generator_rank, hidden)),
        prefix + "affine_weight": affine_weight, prefix + "affine_bias": affine_bias,
    }
    updates = {name: value.astype(deployment_dtype) for name, value in updates.items()}
    mx.eval(updates)
    remove = tuple(prefix + key for key in ("gate_weight", "up_weight", "down_weight", "mode_decoder_weight") if prefix + key in runtime.parameters)
    fitted = runtime.with_parameters(updates, remove=remove, layer_overrides={layer_index: {"mlp": {"temperature": temperature}}})
    # Score the exported coefficient precision, including chart routing roundoff.
    xd = x.astype(deployment_dtype)
    displacement = xd[:, None, :] - updates[prefix + "chart_centers"]
    membership = mx.softmax(-0.5 * mx.sum(displacement ** 2, axis=-1) /
                            (updates[prefix + "chart_distance_scales"] ** 2 * temperature), axis=-1)
    coordinates = mx.einsum("nch,chr->ncr", displacement, updates[prefix + "chart_bases"])
    augmented = mx.concatenate((mx.ones_like(coordinates[..., :1]), coordinates), axis=-1)
    packet = membership[..., None] * mx.einsum("ncr,crg->ncg", augmented, updates[prefix + "local_generator_encoders"])
    prediction = (xd @ updates[prefix + "affine_weight"].T + updates[prefix + "affine_bias"] +
                  mx.einsum("ncg,cgh->nh", packet, updates[prefix + "chart_mode_decoders"])).astype(mx.float32)
    error = mx.sqrt(mx.sum((prediction - y) ** 2) / mx.maximum(mx.sum(y ** 2), 1e-12))
    return MLXFitResult(fitted, updates, {
        "kind": "activation_chart_weighted_ridge", "solver": solver,
        "capture_rows": n, "chart_count": chart_count, "chart_rank": chart_rank,
        "generator_rank": generator_rank, "training_relative_error": float(error.item()),
        "geometry_moment_dtype": "float32",
        "factorization_dtype": "float64" if solver == "numpy_float64" else "float32",
        "regression_accumulation_dtype": "float64" if solver == "numpy_float64" else "float32",
        "deployment_dtype": str(deployment_dtype),
        "fisher_row_weights_supplied": row_weights is not None,
        "euler_jacobian_anchored": euler_prior is not None,
        "centered_intercept": bool(center_intercept),
        "inverse_target_norm_weights": bool(row_weights is None and inverse_target_norm_weights),
        "heldout_confirmation": False,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "backend_default"),
    })


def optimize_trajectory(
    student: Any,
    teacher: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    parameter_names: Sequence[str],
    steps: int = 20,
    learning_rate: float = 1e-3,
    anchor_weight: float = 0.0,
    fisher_diagonal: Mapping[str, Any] | None = None,
    objective: str = "terminal_mse",
    max_grad_norm: float = 1.0,
) -> MLXFitResult:
    """Optimize selected real runtime coefficients through the whole trajectory.

    Uses MLX autodiff and Adam. Teacher outputs are fixed calibration targets;
    unselected student parameters never enter optimizer state. The input runtime
    is unchanged and the best finite *training* objective is returned. No fit
    loss is presented as held-out improvement. Fisher diagonals, when supplied,
    must be measured separately and have exactly the selected parameter shapes.
    Adam masters are float32; loss evaluation and exported updates retain each
    selected parameter's original deployment dtype.
    """
    mx = _mx()
    optim = importlib.import_module("mlx.optimizers")
    _rank(steps, 1_000_000, "steps")
    learning_rate = _positive(learning_rate, "learning_rate")
    anchor_weight = _positive(anchor_weight, "anchor_weight", zero=True)
    max_grad_norm = _positive(max_grad_norm, "max_grad_norm")
    if objective not in {"terminal_mse", "teacher_kl"}:
        raise ValueError("objective must be terminal_mse or teacher_kl")
    names = tuple(parameter_names)
    if not names or len(set(names)) != len(names) or any(name not in student.parameters for name in names):
        raise ValueError("parameter_names must name unique existing student parameters")
    if not batches:
        raise ValueError("optimization requires calibration batches")
    prepared = tuple(_batch(b) for b in batches)
    current = {name: student.parameters[name].astype(mx.float32) for name in names}
    deployment_dtypes = {name: student.parameters[name].dtype for name in names}
    anchor = {name: mx.stop_gradient(value) for name, value in current.items()}
    precision = {}
    if fisher_diagonal is not None:
        if set(fisher_diagonal) != set(names):
            raise ValueError("Fisher diagonals must cover exactly selected parameters")
        for name in names:
            value = mx.array(fisher_diagonal[name], dtype=mx.float32)
            if value.shape != current[name].shape or not bool(mx.all(mx.isfinite(value) & (value >= 0)).item()):
                raise ValueError("Fisher diagonal shapes/values are invalid")
            precision[name] = value
    targets = []
    for batch in prepared:
        output = teacher(**batch, return_logits=objective == "teacher_kl")
        values = output.logits if objective == "teacher_kl" else output.hidden_states
        targets.append(mx.stop_gradient(values.astype(mx.float32)))
    mx.eval(targets)
    total_rows = sum(int(mx.sum(b["attention_mask"]).item()) for b in prepared)
    parameter_count = sum(value.size for value in current.values())

    def loss(selected):
        # Float32 optimizer masters do not change the function being evaluated.
        deployed = {name: value.astype(deployment_dtypes[name]) for name, value in selected.items()}
        merged = {**student.parameters, **deployed}
        total = mx.array(0.0)
        for batch, target in zip(prepared, targets):
            output = student(**batch, parameters=merged, return_logits=objective == "teacher_kl")
            if objective == "terminal_mse":
                per_row = mx.mean((output.hidden_states.astype(mx.float32) - target) ** 2, axis=-1)
            else:
                logits = output.logits.astype(mx.float32)
                teacher_logp = target - mx.logsumexp(target, axis=-1, keepdims=True)
                student_logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                per_row = mx.sum(mx.exp(teacher_logp) * (teacher_logp - student_logp), axis=-1)
            total = total + mx.sum(mx.where(batch["attention_mask"], per_row, 0)) / total_rows
        if anchor_weight:
            penalty = sum(mx.sum((selected[name] - anchor[name]) ** 2 * precision.get(name, 1.0)) for name in names)
            total = total + anchor_weight * penalty / parameter_count
        return total

    initial = float(loss(current).item())
    if not math.isfinite(initial):
        raise FloatingPointError("initial trajectory loss is not finite")
    best, best_loss = dict(current), initial
    history = [initial]
    optimizer = optim.Adam(learning_rate=learning_rate)
    value_and_grad = mx.value_and_grad(loss)
    for _ in range(steps):
        value, gradients = value_and_grad(current)
        norm = mx.sqrt(sum(mx.sum(g * g) for g in gradients.values()))
        mx.eval(value, norm)
        if not math.isfinite(float(value.item())) or not math.isfinite(float(norm.item())):
            raise FloatingPointError("trajectory optimization encountered nonfinite loss/gradient")
        scale = mx.minimum(1.0, max_grad_norm / mx.maximum(norm, 1e-12))
        current = optimizer.apply_gradients({name: g * scale for name, g in gradients.items()}, current)
        mx.eval(current, optimizer.state)
        candidate_loss = float(loss(current).item())
        if not math.isfinite(candidate_loss):
            raise FloatingPointError("trajectory update produced nonfinite loss")
        history.append(candidate_loss)
        if candidate_loss < best_loss:
            best, best_loss = dict(current), candidate_loss
    exported = {name: value.astype(deployment_dtypes[name]) for name, value in best.items()}
    mx.eval(exported)
    return MLXFitResult(student.with_parameters(exported), exported, {
        "kind": "mlx_autodiff_adam_trajectory", "objective": objective,
        "steps": steps, "learning_rate": learning_rate,
        "selected_parameters": list(names), "selected_scalar_count": parameter_count,
        "initial_training_loss": initial, "best_training_loss": best_loss,
        "training_loss_history": history, "anchor_weight": anchor_weight,
        "fisher_precision_supplied": fisher_diagonal is not None,
        "optimizer_master_dtype": "float32",
        "deployment_dtypes": {name: str(dtype) for name, dtype in deployment_dtypes.items()},
        "heldout_confirmation": False,
        "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "backend_default"),
    })


def allocate_waterfill_attention_ranks(
    runtime: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    target_scalars_removed: int,
    layer_indices: Sequence[int] | None = None,
    rank_step: int = 1,
    min_rank: int = 1,
    fisher_max_tokens: int = 32,
) -> dict[str, Any]:
    """Allocate per-layer (input_rank, output_rank) via KKT Fisher water-filling.

    Solves the Lagrangian rate-distortion allocation across all selected
    attention bottlenecks:
        min_{r_{in,l}, r_{out,l}} sum_l Omega_l (D_{in,l}(r_{in,l}) + D_{out,l}(r_{out,l}))
                                  - mu * sum_l Delta_S_l(r_{in,l}, r_{out,l})
    where Omega_l = E[a_l^T F_l a_l] is the scale-invariant Pullback-Fisher
    energy of layer l's attention output, including the unfactored exact
    endpoint (0 distortion, 0 scalars removed) for each side.
    """
    mx = _mx()
    _rank(target_scalars_removed, 2**31 - 1, "target_scalars_removed")
    _rank(rank_step, 2048, "rank_step")
    _rank(min_rank, 2048, "min_rank")
    indices = tuple(range(len(runtime.layers)) if layer_indices is None else layer_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("layer_indices must name unique existing layers")
    prefixes = [
        f"deltanet.{l}." if "deltanet" in runtime.layers[l] else f"attention.{l}."
        for l in indices
    ]
    out_sites = [p + "output" for p in prefixes]
    in_sites = [p + "normalized" for p in prefixes]
    ctx_sites = [p + "context" for p in prefixes]
    captures = capture_runtime_rows(runtime, batches, out_sites + in_sites + ctx_sites)
    fisher = capture_empirical_token_fisher(runtime, batches, out_sites, max_tokens=fisher_max_tokens)

    layer_specs = {}
    for l, prefix in zip(indices, prefixes):
        p = runtime.parameters
        if prefix + "input_basis" in p or prefix + "output_encoder_weight" in p:
            raise ValueError("water-filling requires exact source attention layers")
        y = np.asarray(captures[prefix + "output"].astype(mx.float32), dtype=np.float64)
        f_diag = np.asarray(fisher["diagonal"][prefix + "output"].astype(mx.float32), dtype=np.float64)
        omega = max(float(np.mean(np.sum((y * y) * f_diag[None, :], axis=1))), 1e-12)
        x = captures[prefix + "normalized"].astype(mx.float32)
        c = captures[prefix + "context"].astype(mx.float32)

        if "deltanet" in runtime.layers[l]:
            dc = runtime.layers[l]["deltanet"]
            qkv, z, out = (p[prefix + key].astype(mx.float32) for key in ("qkv_weight", "z_weight", "output_decoder_weight"))
            hidden = int(qkv.shape[1])
            in_proj_width = int(qkv.shape[0] + z.shape[0])
            out_width = int(out.shape[1])
            exact_in_scalars = in_proj_width * hidden
            exact_out_scalars = hidden * out_width
            c_in = hidden + in_proj_width
            c_out = out_width + hidden
            k_total = int(dc["key_heads"]) * int(dc["key_head_dimension"])
            vh = int(dc["value_heads"])
            q_w, k_w, v_w = qkv[:k_total], qkv[k_total : 2 * k_total], qkv[2 * k_total :]
            norm_w = p.get(prefix + "norm_weight")
            if norm_w is not None:
                gain = norm_w.astype(mx.float32)
                if dc.get("norm", {}).get("scale_parameterization") == "unit_offset":
                    gain = gain + 1.0
                gain_v = mx.tile(mx.maximum(mx.abs(gain), 1e-3), vh)[:, None]
                v_w = v_w * gain_v
                z_eff = z * gain_v
            else:
                z_eff = z
            ve = out @ v_w
            ze = out @ z_eff
            grams = [w.T @ w for w in (q_w, k_w, ve, ze)]
            metric = sum(scale * g / mx.maximum(mx.sum(mx.diag(g)), 1e-12) for scale, g in zip((1.0, 1.0, 2.0, 1.0), grams))
        else:
            q, k, v, out = (p[prefix + key].astype(mx.float32) for key in ("q_weight", "k_weight", "v_weight", "output_decoder_weight"))
            hidden = int(q.shape[1])
            q_width = int(q.shape[0])
            kv_width = int(k.shape[0])
            out_width = int(out.shape[1])
            exact_in_scalars = (q_width + 2 * kv_width) * hidden
            exact_out_scalars = hidden * out_width
            c_in = hidden + q_width + 2 * kv_width
            c_out = out_width + hidden
            config = runtime.layers[l]["attention"]
            head_dim = int(config["head_dimension"])

            def qk_effective(weight, key):
                gain = p.get(prefix + key)
                if gain is None:
                    return weight
                spec = config.get("qk_norm", {})
                gain = gain + 1 if spec.get("scale_parameterization") == "unit_offset" else gain
                scale = mx.maximum(mx.abs(gain), 1e-3)
                if key == "q_norm_weight" and config.get("output_gate", False):
                    scale = mx.concatenate((scale, mx.ones_like(scale)), axis=0)
                return weight * mx.tile(scale, weight.shape[0] // scale.shape[0])[:, None]

            qe, ke = qk_effective(q, "q_norm_weight"), qk_effective(k, "k_norm_weight")
            repeats = int(config["query_heads"]) // int(config["key_value_heads"])
            repeated_v = mx.repeat(v.reshape((-1, head_dim, v.shape[1])), repeats, axis=0).reshape((out.shape[1], v.shape[1]))
            ve = out @ repeated_v
            grams = [w.T @ w for w in (qe, ke, ve)]
            metric = sum(scale * g / mx.maximum(mx.sum(mx.diag(g)), 1e-12) for scale, g in zip((1.0, 1.0, 2.0), grams))

        cov_x = (x.T @ x) / x.shape[0]
        cov_x_reg = cov_x + 0.15 * mx.maximum(mx.sum(mx.diag(cov_x)) / hidden, 1e-12) * mx.eye(hidden)
        pb_in = cov_x_reg @ metric @ cov_x_reg
        ev_x = mx.linalg.eigvalsh((pb_in + pb_in.T) * 0.5, stream=mx.cpu)
        cov_c = (c.T @ c) / c.shape[0]
        cov_c_reg = cov_c + 0.15 * mx.maximum(mx.sum(mx.diag(cov_c)) / out_width, 1e-12) * mx.eye(out_width)
        out_gram = out @ cov_c_reg @ out.T
        ev_o = mx.linalg.eigvalsh((out_gram + out_gram.T) * 0.5, stream=mx.cpu)
        mx.eval(ev_x, ev_o)
        ex_np = np.maximum(np.asarray(ev_x, dtype=np.float64)[::-1], 0.0)
        eo_np = np.maximum(np.asarray(ev_o, dtype=np.float64)[::-1], 0.0)
        ex_np /= max(ex_np.sum(), 1e-12)
        eo_np /= max(eo_np.sum(), 1e-12)

        in_tail = np.zeros(hidden + 1, dtype=np.float64)
        out_tail = np.zeros(min(hidden, out_width) + 1, dtype=np.float64)
        in_tail[:hidden] = 1.0 - np.concatenate([[0.0], np.cumsum(ex_np[:-1])])
        out_tail[: min(hidden, out_width)] = 1.0 - np.concatenate([[0.0], np.cumsum(eo_np[: min(hidden, out_width) - 1])])

        max_factored_in = max((exact_in_scalars - 1) // c_in, min_rank)
        max_factored_out = max((exact_out_scalars - 1) // c_out, min_rank)
        in_candidates = [r for r in range(min_rank, min(max_factored_in, hidden) + 1, rank_step)] + [hidden]
        out_candidates = [r for r in range(min_rank, min(max_factored_out, min(hidden, out_width)) + 1, rank_step)] + [min(hidden, out_width)]

        layer_specs[l] = {
            "omega": omega,
            "hidden": hidden,
            "out_full": min(hidden, out_width),
            "exact_in_scalars": exact_in_scalars,
            "exact_out_scalars": exact_out_scalars,
            "c_in": c_in,
            "c_out": c_out,
            "in_tail": in_tail,
            "out_tail": out_tail,
            "in_candidates": in_candidates,
            "out_candidates": out_candidates,
        }

    def evaluate_mu(mu: float):
        allocations = {}
        total_removed = 0
        total_distortion = 0.0
        for l in indices:
            s = layer_specs[l]
            omega = s["omega"]
            best_in = min(
                s["in_candidates"],
                key=lambda r: omega * s["in_tail"][r] - mu * (0 if r == s["hidden"] else s["exact_in_scalars"] - s["c_in"] * r),
            )
            best_out = min(
                s["out_candidates"],
                key=lambda r: omega * s["out_tail"][r] - mu * (0 if r == s["out_full"] else s["exact_out_scalars"] - s["c_out"] * r),
            )
            rem_in = 0 if best_in == s["hidden"] else s["exact_in_scalars"] - s["c_in"] * best_in
            rem_out = 0 if best_out == s["out_full"] else s["exact_out_scalars"] - s["c_out"] * best_out
            dist = omega * (s["in_tail"][best_in] + s["out_tail"][best_out])
            allocations[l] = {
                "input_rank": best_in,
                "output_rank": best_out,
                "input_exact": best_in == s["hidden"],
                "output_exact": best_out == s["out_full"],
                "omega": omega,
                "scalars_removed": int(rem_in + rem_out),
                "fisher_distortion": float(dist),
            }
            total_removed += int(rem_in + rem_out)
            total_distortion += float(dist)
        return total_removed, total_distortion, allocations

    max_rem, _, _ = evaluate_mu(1e9)
    if target_scalars_removed > max_rem:
        raise ValueError(f"target_scalars_removed ({target_scalars_removed}) exceeds maximum achievable ({max_rem})")
    lo, hi = 0.0, 1.0
    while evaluate_mu(hi)[0] < target_scalars_removed:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if evaluate_mu(mid)[0] < target_scalars_removed:
            lo = mid
        else:
            hi = mid
    removed, distortion, allocations = evaluate_mu(hi)
    return {
        "kind": "kkt_pullback_fisher_waterfill_schedule",
        "target_scalars_removed": target_scalars_removed,
        "achieved_scalars_removed": removed,
        "total_fisher_distortion": distortion,
        "lagrange_multiplier_mu": hi,
        "layers": allocations,
    }


def allocate_waterfill_joint_ranks(
    runtime: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    target_scalars_removed: int,
    chart_count: int = 4,
    chart_rank: int = 16,
    generator_rank: int | None = None,
    mlp_expected_relative_error: float = 0.50,
    min_mlp_layer_spacing: int = 2,
    rank_step: int = 1,
    min_rank: int = 384,
    fisher_max_tokens: int = 32,
) -> dict[str, Any]:
    """Joint KKT Pullback-Fisher water-filling across all attention and MLP blocks.

    Ranks MLP layers by Omega_{l,mlp} = E[y_l^T F_{mlp,l} y_l] and selects the
    lowest-distortion non-adjacent MLP blocks whose marginal Fisher distortion
    per scalar removed falls below the water-filling threshold, allocating the
    remaining scalar budget across attention blocks via
    ``allocate_waterfill_attention_ranks``.
    """
    mx = _mx()
    _rank(target_scalars_removed, 2**31 - 1, "target_scalars_removed")
    _rank(chart_count, 1024, "chart_count")
    _rank(chart_rank, 2048, "chart_rank")
    generator_rank = chart_rank + 1 if generator_rank is None else generator_rank
    _rank(generator_rank, chart_rank + 1, "generator_rank")
    err_rel = _positive(mlp_expected_relative_error, "mlp_expected_relative_error")
    layer_count = len(runtime.layers)
    mlp_sites = [f"mlp.{l}.output" for l in range(layer_count)]
    captures = capture_runtime_rows(runtime, batches, mlp_sites)
    fisher = capture_empirical_token_fisher(runtime, batches, mlp_sites, max_tokens=fisher_max_tokens)

    mlp_candidates = []
    for l in range(layer_count):
        prefix = f"mlp.{l}."
        p = runtime.parameters
        if any(prefix + k not in p for k in ("gate_weight", "up_weight", "down_weight")):
            continue
        gw = p[prefix + "gate_weight"]
        inter, hidden = int(gw.shape[0]), int(gw.shape[1])
        exact_scalars = 3 * inter * hidden
        chart_scalars = (
            hidden * hidden
            + hidden
            + chart_count * hidden
            + chart_count * hidden * chart_rank
            + chart_count
            + chart_count * (chart_rank + 1) * generator_rank
            + chart_count * generator_rank * hidden
        )
        rem_mlp = exact_scalars - chart_scalars
        if rem_mlp <= 0:
            continue
        y = np.asarray(captures[prefix + "output"].astype(mx.float32), dtype=np.float64)
        f_diag = np.asarray(fisher["diagonal"][prefix + "output"].astype(mx.float32), dtype=np.float64)
        omega_mlp = max(float(np.mean(np.sum((y * y) * f_diag[None, :], axis=1))), 1e-12)
        dist_mlp = omega_mlp * (err_rel ** 2)
        mlp_candidates.append({
            "layer": l,
            "omega_mlp": omega_mlp,
            "scalars_removed": int(rem_mlp),
            "fisher_distortion": float(dist_mlp),
            "distortion_per_scalar": float(dist_mlp / rem_mlp),
        })

    mlp_candidates.sort(key=lambda item: item["distortion_per_scalar"])

    def _select(remaining_budget: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        used = 0
        for item in mlp_candidates:
            if any(abs(item["layer"] - s["layer"]) < min_mlp_layer_spacing for s in selected):
                continue
            if used + item["scalars_removed"] <= remaining_budget:
                selected.append(item)
                used += item["scalars_removed"]
        return selected

    # KKT marginal-consistency pass: a chart-MLP swap is only optimal if its
    # distortion-per-scalar is below the attention water-filling threshold mu
    # set by the budget it frees. Drop the worst selected MLP while its
    # distortion-per-scalar exceeds that mu, then backfill cheaper ones.
    selected_mlps = _select(target_scalars_removed)
    for _ in range(8):
        removed = sum(s["scalars_removed"] for s in selected_mlps)
        attn_probe = allocate_waterfill_attention_ranks(
            runtime,
            batches,
            target_scalars_removed=max(target_scalars_removed - removed, 0),
            rank_step=rank_step,
            min_rank=min_rank,
            fisher_max_tokens=fisher_max_tokens,
        )
        mu = attn_probe["lagrange_multiplier_mu"]
        worst = max(selected_mlps, key=lambda s: s["distortion_per_scalar"], default=None)
        if worst is None or worst["distortion_per_scalar"] <= mu:
            break
        selected_mlps = [s for s in selected_mlps if s is not worst]
        budget_left = target_scalars_removed - sum(s["scalars_removed"] for s in selected_mlps)
        for item in mlp_candidates:
            if item in selected_mlps or item["distortion_per_scalar"] > mu:
                continue
            if any(abs(item["layer"] - s["layer"]) < min_mlp_layer_spacing for s in selected_mlps):
                continue
            if item["scalars_removed"] <= budget_left:
                selected_mlps.append(item)
                budget_left -= item["scalars_removed"]

    mlp_removed = sum(s["scalars_removed"] for s in selected_mlps)
    remaining = max(target_scalars_removed - mlp_removed, 0)
    if remaining > 0:
        attn_schedule = allocate_waterfill_attention_ranks(
            runtime,
            batches,
            target_scalars_removed=remaining,
            rank_step=rank_step,
            min_rank=min_rank,
            fisher_max_tokens=fisher_max_tokens,
        )
    else:
        attn_schedule = {
            "achieved_scalars_removed": 0,
            "total_fisher_distortion": 0.0,
            "layers": {},
        }

    return {
        "kind": "kkt_pullback_fisher_joint_waterfill_schedule",
        "target_scalars_removed": target_scalars_removed,
        "achieved_scalars_removed": int(mlp_removed + attn_schedule["achieved_scalars_removed"]),
        "mlp_scalars_removed": int(mlp_removed),
        "attention_scalars_removed": int(attn_schedule["achieved_scalars_removed"]),
        "selected_mlp_layers": sorted(item["layer"] for item in selected_mlps),
        "mlp_candidates_ranked": mlp_candidates,
        "attention_schedule": attn_schedule,
    }


def allocate_waterfill_deltanet_carrier_ranks(
    runtime: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    target_state_scalars_removed: int | None = None,
    target_compression_ratio: float | None = None,
    layer_indices: Sequence[int] | None = None,
    rank_step: int = 32,
    min_rank: int = 32,
    fisher_max_tokens: int = 16,
    captures: Mapping[str, Any] | None = None,
    fisher_diagonal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Allocate per-layer GatedDeltaNet carrier ranks r_l via KKT Fisher water-filling (Play 2).

    Solves the Lagrangian rate-distortion allocation across all selected DeltaNet
    recurrent carriers S_l in R^{B x Hv x Dv x r_l}:
        min_{r_l} sum_l Omega_l D_l(r_l) - mu * sum_l Hv * Dv * (Dk - r_l)
    where Omega_l = E[y_l^T F_{deltanet.l.output} y_l] is layer l's Pullback-Fisher
    vitality and D_l(r_l) is the normalized spectral tail of the Fisher-pullback
    carrier Gram ``Cov(q) D_q Cov(q) + Cov(k) D_k Cov(k)`` truncated at rank r_l.
    Default ``rank_step=32, min_rank=32`` keeps every candidate rank aligned to the
    32-element SIMD Metal kernel boundary.
    """
    mx = _mx()
    _rank(rank_step, 2048, "rank_step")
    _rank(min_rank, 2048, "min_rank")
    indices = tuple(
        (i for i, layer in enumerate(runtime.layers) if "deltanet" in layer)
        if layer_indices is None else layer_indices
    )
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("layer_indices must name unique existing deltanet layers")
    for l in indices:
        if not 0 <= l < len(runtime.layers) or "deltanet" not in runtime.layers[l]:
            raise ValueError(f"layer {l} is not a deltanet layer")
        if f"deltanet.{l}.carrier_basis" in runtime.parameters:
            raise ValueError("carrier water-filling requires unreduced source deltanet carriers")

    out_sites = [f"deltanet.{l}.output" for l in indices]
    qk_sites = [s for l in indices for s in (f"deltanet.{l}.q", f"deltanet.{l}.k")]
    needed_sites = out_sites + qk_sites
    if captures is None or any(s not in captures for s in needed_sites):
        captures = capture_runtime_rows(runtime, batches, needed_sites)
    if fisher_diagonal is None or any(s not in fisher_diagonal for s in needed_sites):
        fisher = capture_empirical_token_fisher(runtime, batches, needed_sites, max_tokens=fisher_max_tokens)
        fisher_diagonal = fisher["diagonal"]

    total_exact_scalars = 0
    layer_specs = {}
    for l in indices:
        prefix = f"deltanet.{l}."
        dc = runtime.layers[l]["deltanet"]
        kh = int(dc["key_heads"])
        vh = int(dc["value_heads"])
        kd = int(dc["key_head_dimension"])
        vd = int(dc["value_head_dimension"])
        exact_scalars = vh * vd * kd
        total_exact_scalars += exact_scalars

        y = np.asarray(captures[prefix + "output"].astype(mx.float32), dtype=np.float64)
        f_out = np.asarray(fisher_diagonal[prefix + "output"].astype(mx.float32), dtype=np.float64)
        omega = max(float(np.mean(np.sum((y * y) * f_out[None, :], axis=1))), 1e-12)

        q = captures[prefix + "q"].astype(mx.float32).reshape((-1, kh, kd))
        k = captures[prefix + "k"].astype(mx.float32).reshape((-1, kh, kd))
        fq = mx.maximum(mx.array(fisher_diagonal[prefix + "q"], dtype=mx.float32).reshape((kh, kd)), 1e-12)
        fk = mx.maximum(mx.array(fisher_diagonal[prefix + "k"], dtype=mx.float32).reshape((kh, kd)), 1e-12)
        n_rows = float(q.shape[0])

        head_evs = []
        for h in range(kh):
            qh, kh_rows = q[:, h, :], k[:, h, :]
            cov_q = (qh.T @ qh) / n_rows
            cov_k = (kh_rows.T @ kh_rows) / n_rows
            wq = fq[h] / mx.maximum(mx.mean(fq[h]), 1e-12)
            wk = fk[h] / mx.maximum(mx.mean(fk[h]), 1e-12)
            gram = cov_q @ (wq[:, None] * cov_q) + cov_k @ (wk[:, None] * cov_k)
            ev = mx.linalg.eigvalsh((gram + gram.T) * 0.5, stream=mx.cpu)
            head_evs.append(ev)
        mx.eval(head_evs)
        ev_sum = sum(np.maximum(np.asarray(ev, dtype=np.float64)[::-1], 0.0) for ev in head_evs)
        ev_sum /= max(float(ev_sum.sum()), 1e-12)
        tail = np.zeros(kd + 1, dtype=np.float64)
        tail[:kd] = 1.0 - np.concatenate([[0.0], np.cumsum(ev_sum[:-1])])

        candidates = sorted({*(r for r in range(min_rank, kd + 1, rank_step) if r <= kd), kd})
        layer_specs[l] = {
            "omega": omega,
            "kd": kd,
            "scalars_per_rank": vh * vd,
            "exact_scalars": exact_scalars,
            "tail": tail,
            "candidates": candidates,
        }

    if target_state_scalars_removed is None:
        if target_compression_ratio is None:
            raise ValueError("Specify target_state_scalars_removed or target_compression_ratio")
        ratio = float(target_compression_ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError("target_compression_ratio must lie in (0, 1)")
        target_state_scalars_removed = int(round(total_exact_scalars * (1.0 - ratio)))
    else:
        _rank(target_state_scalars_removed, 2**31 - 1, "target_state_scalars_removed")

    def evaluate_mu(mu: float):
        allocations = {}
        total_removed = 0
        total_distortion = 0.0
        for l in indices:
            s = layer_specs[l]
            omega = s["omega"]
            kd = s["kd"]
            spr = s["scalars_per_rank"]
            best_r = min(
                s["candidates"],
                key=lambda r: omega * s["tail"][r] - mu * (spr * (kd - r)),
            )
            rem = spr * (kd - best_r)
            dist = omega * s["tail"][best_r]
            allocations[l] = {
                "carrier_rank": int(best_r),
                "full_carrier_dimension": int(kd),
                "carrier_exact": best_r == kd,
                "omega": float(omega),
                "tail_fraction": float(s["tail"][best_r]),
                "state_scalars_kept": int(spr * best_r),
                "state_scalars_removed": int(rem),
                "fisher_distortion": float(dist),
            }
            total_removed += int(rem)
            total_distortion += float(dist)
        return total_removed, total_distortion, allocations

    max_rem, _, _ = evaluate_mu(1e9)
    if target_state_scalars_removed > max_rem:
        raise ValueError(
            f"target_state_scalars_removed ({target_state_scalars_removed}) exceeds maximum achievable ({max_rem})"
        )
    lo, hi = 0.0, 1e-6
    while evaluate_mu(hi)[0] < target_state_scalars_removed:
        hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if evaluate_mu(mid)[0] < target_state_scalars_removed:
            lo = mid
        else:
            hi = mid
    removed, distortion, allocations = evaluate_mu(hi)
    kept = total_exact_scalars - removed
    return {
        "kind": "kkt_pullback_fisher_deltanet_carrier_waterfill_schedule",
        "total_exact_state_scalars": int(total_exact_scalars),
        "target_state_scalars_removed": int(target_state_scalars_removed),
        "achieved_state_scalars_removed": int(removed),
        "achieved_state_scalars_kept": int(kept),
        "achieved_compression_ratio": float(kept / max(total_exact_scalars, 1)),
        "total_fisher_distortion": float(distortion),
        "lagrange_multiplier_mu": float(hi),
        "layers": allocations,
    }


__all__ = [
    "MLXMoments", "MLXFitResult", "weighted_moments", "capture_runtime_rows",
    "capture_empirical_token_fisher", "capture_parameter_fisher", "anchored_ridge", "fit_reduced_attention",
    "fit_reduced_deltanet", "fit_deltanet_carrier_basis", "allocate_waterfill_deltanet_carrier_ranks",
    "fit_chart_mlp", "optimize_trajectory", "allocate_waterfill_attention_ranks", "allocate_waterfill_joint_ranks",
]
