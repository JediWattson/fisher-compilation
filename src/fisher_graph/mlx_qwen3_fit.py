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
        return weight * mx.tile(mx.maximum(mx.abs(gain), 1e-3), weight.shape[0] // head_dim)[:, None]

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
    solver: str = "mlx",
) -> MLXFitResult:
    """Fit affine + local chart residuals to real Qwen MLP operator captures.

    Charts emit hidden-space deltas directly; no dense identity decoder is
    allocated. Geometry and regression use only the supplied calibration rows.
    Input centers are weighted k-means, bases are local activation covariance.
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
    weights = _weights(row_weights, n)
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
    affine_features = mx.concatenate((x, mx.ones((n, 1))), axis=1)
    affine = anchored_ridge(affine_features, y, row_weights=weights, relative_ridge=relative_ridge, solver=solver)
    residual = y - affine_features @ affine
    decoded = anchored_ridge(features, residual, row_weights=weights, relative_ridge=relative_ridge, solver=solver)
    updates = {
        prefix + "chart_centers": centers, prefix + "chart_bases": bases,
        prefix + "chart_distance_scales": scales,
        prefix + "local_generator_encoders": encoders,
        prefix + "chart_mode_decoders": decoded.reshape((chart_count, generator_rank, hidden)),
        prefix + "affine_weight": affine[:-1].T, prefix + "affine_bias": affine[-1],
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
        "fisher_row_weights_supplied": row_weights is not None, "heldout_confirmation": False,
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


__all__ = [
    "MLXMoments", "MLXFitResult", "weighted_moments", "capture_runtime_rows",
    "capture_empirical_token_fisher", "capture_parameter_fisher", "anchored_ridge", "fit_reduced_attention",
    "fit_chart_mlp", "optimize_trajectory",
]
