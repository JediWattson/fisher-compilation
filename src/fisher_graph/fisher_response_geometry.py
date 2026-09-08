"""Synthetic normalization-response qualification; no model or corpus loader.

The current teacher is ``T(h, m) = h + RMSNorm(W_O m)``.  Its partial
state-mutation derivative is zero.  All changing message responses in this
module belong to normalization, not live attention.  The source attribution
helper is deliberately neither imported nor changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor


def _tensor(value: Tensor, name: str, ndim: int | None = None) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not bool(torch.isfinite(value).all())
        or any(size == 0 for size in value.shape)
        or (ndim is not None and value.ndim != ndim)
    ):
        raise ValueError(f"{name} must be a finite, nonempty float64 CPU tensor")
    return value


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _normalization_shapes(m: Tensor, W: Tensor, weight: Tensor) -> None:
    if m.ndim < 1 or W.ndim != 2 or weight.shape != (W.shape[0],):
        raise ValueError("normalization requires m[...,p], W[d,p], weight[d]")
    if m.shape[-1] != W.shape[1] or m.dtype != W.dtype or m.dtype != weight.dtype:
        raise ValueError("message, output projection and scale must match")
    if any(
        not v.is_floating_point() or v.device.type != "cpu"
        or not bool(torch.isfinite(v).all()) or v.numel() == 0
        for v in (m, W, weight)
    ):
        raise ValueError("normalization inputs must be finite floating CPU tensors")


def normalization_update(
    m: Tensor, W: Tensor, weight: Tensor, epsilon: float,
    *, semantics: str = "float64",
) -> Tensor:
    """Recompute the normalizer; ``gemma`` qualifies casts, not a model run.

    ``float64`` is mathematical reference arithmetic.  ``gemma`` performs
    output projection in the supplied dtype, RMS/scale arithmetic in fp32,
    then casts the branch update once to the projection dtype.
    """
    _normalization_shapes(m, W, weight)
    eps = _positive(epsilon, "epsilon")
    if semantics not in {"float64", "gemma"}:
        raise ValueError("semantics must be float64 or gemma")
    if semantics == "float64":
        _tensor(m, "message")
    y = m @ W.T
    work = y if semantics == "float64" else y.float()
    gamma = 1 + (weight if semantics == "float64" else weight.float())
    result = (work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)) * gamma
    return result.to(y.dtype)


def normalization_jvp(
    m: Tensor, v: Tensor, W: Tensor, weight: Tensor, epsilon: float,
) -> Tensor:
    """Analytic action Gamma [I/s - yy^T/(d s^3)] W v; no dense Jacobian."""
    for name, value in (("message", m), ("direction", v), ("W", W), ("weight", weight)):
        _tensor(value, name)
    _normalization_shapes(m, W, weight)
    if v.shape != m.shape:
        raise ValueError("direction must match message shape")
    eps = _positive(epsilon, "epsilon")
    y, dy = m @ W.T, v @ W.T
    s = torch.sqrt(y.square().mean(dim=-1, keepdim=True) + eps)
    radial = (y * dy).mean(dim=-1, keepdim=True)
    return (1 + weight) * (dy / s - y * radial / s.pow(3))


def frozen_scale_control(m: Tensor, W: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    """Adversarial control: right values, deliberately wrong RMSNorm gradients.

    This synthetic control mimics the attribution-scale stop-gradient error;
    it does not alter or call the existing attribution helper.
    """
    _tensor(m, "message")
    _normalization_shapes(m, W, weight)
    y = m @ W.T
    s = torch.sqrt(y.detach().square().mean(dim=-1, keepdim=True) + _positive(epsilon, "epsilon"))
    return (1 + weight) * y / s


def jvp_actions(fn: Callable[[Tensor], Tensor], point: Tensor, basis: Tensor) -> Tensor:
    """Columns of derivative actions using the installed autodiff stack."""
    _tensor(point, "point", 1)
    _tensor(basis, "basis", 2)
    if basis.shape[0] != point.numel():
        raise ValueError("input basis width differs from point")
    return torch.stack([
        torch.autograd.functional.jvp(fn, point, direction, create_graph=False, strict=False)[1]
        for direction in basis.T
    ], dim=1)


def central_difference_sweep(
    fn: Callable[[Tensor], Tensor], point: Tensor, direction: Tensor,
    steps: Sequence[float], reference: Tensor,
) -> list[dict[str, float]]:
    """A reported step sweep, not an unqualified derivative ground truth."""
    if point.shape != direction.shape or not steps:
        raise ValueError("finite differences require matching directions and steps")
    result = []
    denominator = max(float(reference.double().norm()), 1e-15)
    for step in steps:
        step = _positive(step, "finite difference step")
        observed = (fn(point + step * direction).double() - fn(point - step * direction).double()) / (2 * step)
        error = float((observed - reference.double()).norm())
        result.append({"step": step, "absolute_error": error, "relative_error": error / denominator})
    return result


def _supported_metric(metric: Tensor, rtol: float) -> tuple[Tensor, Tensor, Tensor]:
    _tensor(metric, "metric", 2)
    matrix_scale = max(float(metric.abs().max()), torch.finfo(torch.float64).tiny)
    if metric.shape[0] != metric.shape[1] or float((metric - metric.T).abs().max()) > _positive(rtol, "metric rtol") * matrix_scale:
        raise ValueError("metric must be square and symmetric")
    values, vectors = torch.linalg.eigh(metric)
    scale = max(float(values.abs().max()), torch.finfo(torch.float64).tiny)
    cutoff = _positive(rtol, "metric rtol") * scale
    if float(values.min()) < -cutoff:
        raise ValueError("metric must be positive semidefinite")
    supported = values > cutoff
    if not bool(supported.any()):
        raise ValueError("metric has no supported positive subspace")
    return values[supported], vectors[:, supported], vectors[:, ~supported]


@dataclass(frozen=True)
class FisherProjection:
    """Fixed supported output coordinates from F=L L^T; never invert F.

    An ambient singular metric is valid.  A singular generator Gram is not:
    select fewer supported generator directions before constructing a chart.
    """
    basis: Tensor
    factor: Tensor
    rtol: float = 1e-12
    gram: Tensor = field(init=False)
    coordinate_map: Tensor = field(init=False)

    def __post_init__(self) -> None:
        basis = _tensor(self.basis, "basis", 2).detach().clone()
        factor = _tensor(self.factor, "Fisher factor", 2).detach().clone()
        if basis.shape[0] != factor.shape[0]:
            raise ValueError("basis and Fisher factor must have the same output width")
        weighted = factor.T @ basis
        gram = weighted.T @ weighted
        values, _, _ = _supported_metric(gram, self.rtol)
        if values.numel() != gram.shape[0]:
            raise ValueError("generator Gram is singular; select its supported subspace first")
        # H^-1 G^T F, with a solve only in supported coefficient space.
        coordinates = torch.linalg.solve(gram, weighted.T @ factor.T)
        for name, value in (("basis", basis), ("factor", factor), ("gram", gram), ("coordinate_map", coordinates)):
            object.__setattr__(self, name, value)

    def encode(self, values: Tensor) -> Tensor:
        return _tensor(values, "output") @ self.coordinate_map.T

    def decode(self, coordinates: Tensor) -> Tensor:
        return _tensor(coordinates, "coordinates") @ self.basis.T

    def project(self, values: Tensor) -> Tensor:
        return self.decode(self.encode(values))

    def fisher_squared(self, values: Tensor) -> Tensor:
        return (_tensor(values, "output") @ self.factor).square().sum(dim=-1)


def metric_operator_analysis(
    operator: Tensor, output_metric: Tensor, input_metric: Tensor | None = None,
    *, rtol: float = 1e-12,
) -> dict[str, Any]:
    """SVD/polar analysis with distinct input/output metric support.

    A visible response to a null input precludes finite gains on the full
    semimetric quotient, even though the explicitly restricted map has an SVD.
    """
    B = _tensor(operator, "operator", 2)
    same_metric = input_metric is None
    if input_metric is None:
        input_metric = output_metric
    out_l, out_e, _ = _supported_metric(output_metric, rtol)
    in_l, in_e, in_null = _supported_metric(input_metric, rtol)
    if B.shape != (output_metric.shape[0], input_metric.shape[0]):
        raise ValueError("operator shape must match output and input metrics")
    encode = out_l.sqrt()[:, None] * out_e.T
    decode = in_e * in_l.rsqrt()[None, :]
    whitened = encode @ B @ decode
    left, singular, right_t = torch.linalg.svd(whitened, full_matrices=False)
    orthogonal = left @ right_t  # a partial isometry for rectangular maps
    positive = (right_t.T * singular) @ right_t
    null_response = float((encode @ B @ in_null).norm())
    cutoff = rtol * max(float(singular.max()), torch.finfo(torch.float64).tiny)
    numerical_rank = int((singular > cutoff).sum())
    rank_deficient = numerical_rank < min(whitened.shape)
    return {
        "whitened": whitened,
        "singular_values": singular,
        "orthogonal_factor": orthogonal,
        "positive_factor": positive,
        "input_supported_rank": int(in_l.numel()),
        "output_supported_rank": int(out_l.numel()),
        "rank_deficient": rank_deficient,
        "polar_factor_unique": not rank_deficient,
        "factor_kind": "common_coordinate_orthogonal_factor" if same_metric else "partial_isometry_between_distinct_metric_spaces",
        "full_semimetric_gain_defined": null_response <= rtol * max(torch.finfo(torch.float64).tiny, float((encode @ B).norm())),
        "null_input_visible_response_norm": null_response,
        "same_input_output_metric": same_metric,
        "reflection": (bool(torch.linalg.det(orthogonal) < 0) if same_metric and not rank_deficient else None),
    }


@dataclass(frozen=True)
class AffineResponseChart:
    """One anchored response chart, with full-input support guards.

    ``predict`` is explicitly unchecked for diagnostic extrapolation;
    ``predict_supported`` rejects outside the train-frozen trust radii.
    """
    state_anchor: Tensor
    message_anchor: Tensor
    beta: Tensor
    B: Tensor
    A: Tensor
    U: Tensor
    Q: Tensor
    G: Tensor
    state_radius: float
    message_radius: float

    def __post_init__(self) -> None:
        for name in ("state_anchor", "message_anchor", "beta", "B", "A", "U", "Q", "G"):
            value = _tensor(getattr(self, name), name).detach()
            # Shared bases are read-only by convention.  Cloning them for each
            # chart would silently change the physical storage accounting.
            object.__setattr__(self, name, value if name in {"U", "Q", "G"} else value.clone())
        d, k = self.G.shape
        if (self.state_anchor.shape != (d,) or self.message_anchor.shape != (self.Q.shape[0],)
            or self.U.shape[0] != d or self.beta.shape != (k,)
            or self.B.shape != (k, self.U.shape[1]) or self.A.shape != (k, self.Q.shape[1])):
            raise ValueError("chart anchors, bases and operators have incompatible shapes")
        for basis in (self.U, self.Q):
            if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=torch.float64), atol=1e-10, rtol=1e-10):
                raise ValueError("state and message bases must be Euclidean orthonormal")
        _positive(self.state_radius, "state support radius")
        _positive(self.message_radius, "message support radius")

    def _inputs(self, h: Tensor, m: Tensor) -> None:
        if _tensor(h, "state", 1).shape != self.state_anchor.shape or _tensor(m, "message", 1).shape != self.message_anchor.shape:
            raise ValueError("chart prediction inputs must match its anchor shapes")

    def in_support(self, h: Tensor, m: Tensor) -> bool:
        self._inputs(h, m)
        return bool((h - self.state_anchor).norm() <= self.state_radius and (m - self.message_anchor).norm() <= self.message_radius)

    def predict(self, h: Tensor, m: Tensor) -> Tensor:
        self._inputs(h, m)
        z = self.U.T @ (h - self.state_anchor)
        a = self.Q.T @ (m - self.message_anchor)
        return self.G @ (self.beta + self.B @ z + self.A @ a)

    def predict_supported(self, h: Tensor, m: Tensor) -> Tensor:
        if not self.in_support(h, m):
            raise ValueError("prediction rejected: state or message is outside chart support")
        return self.predict(h, m)


def _existing_chart_primitives() -> Any:
    """Reuse the model-free carrier/k-means/bootstrap without eager package exports."""
    existing = sys.modules.get("fisher_graph.fisher_local_chart")
    if existing is not None:
        return existing
    name = "_fisher_response_geometry_existing_chart_primitives"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("fisher_local_chart.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _orthogonal_rows(values: Tensor, rank: int) -> Tensor:
    return torch.linalg.svd(values - values.mean(dim=0), full_matrices=False).Vh[:rank].T.contiguous()


def _errors(predicted: Tensor, actual: Tensor, projection: FisherProjection) -> dict[str, float]:
    error = predicted - actual
    denominator_e = max(float(actual.square().sum()), 1e-30)
    denominator_f = max(float(projection.fisher_squared(actual).sum()), 1e-30)
    return {
        "euclidean_rmse": float(error.square().mean().sqrt()),
        "euclidean_relative_l2": math.sqrt(float(error.square().sum()) / denominator_e),
        "fisher_relative_l2": math.sqrt(float(projection.fisher_squared(error).sum()) / denominator_f),
        "mean_fisher_squared_error": float(projection.fisher_squared(error).mean()),
    }


def _tensor_list(value: Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _derivative_qualification(config: Mapping[str, Any], W: Tensor, weight: Tensor, m: Tensor) -> dict[str, Any]:
    eps = config["epsilon"]
    generator = torch.Generator().manual_seed(config["seed"] + 31)
    direction = torch.randn(m.shape, generator=generator, dtype=torch.float64)
    direction /= direction.norm()
    fn = lambda value: normalization_update(value, W, weight, eps)
    analytic = normalization_jvp(m, direction, W, weight, eps)
    _, automatic = torch.autograd.functional.jvp(fn, m, direction)
    y = W @ m
    s = (y.square().mean() + eps).sqrt()
    dense = torch.diag(1 + weight) @ (torch.eye(y.numel(), dtype=torch.float64) / s - torch.outer(y, y) / (y.numel() * s.pow(3))) @ W
    dense_autodiff = torch.autograd.functional.jacobian(fn, m)
    radial_expected = eps * (1 + weight) * y / s.pow(3)
    radial = normalization_jvp(m, m, W, weight, eps)
    _, radial_autodiff = torch.autograd.functional.jvp(fn, m, m)
    _, adversarial = torch.autograd.functional.jvp(lambda value: frozen_scale_control(value, W, weight, eps), m, m)
    sweep = central_difference_sweep(fn, m, direction, config["fd_steps"], analytic)
    tolerances = config["numerical_tolerances"]
    def scaled_error(left: Tensor, right: Tensor, atol: float, rtol: float) -> float:
        return float(((left - right).abs() / (atol + rtol * right.abs())).max())
    scaled_errors = {
        "analytic_matches_autodiff": scaled_error(analytic, automatic, tolerances["analytic_jvp_atol"], tolerances["analytic_jvp_rtol"]),
        "dense_analytic_matches_autodiff": scaled_error(dense, dense_autodiff, tolerances["analytic_jvp_atol"], tolerances["analytic_jvp_rtol"]),
        "radial_identity": scaled_error(radial, radial_expected, tolerances["radial_atol"], tolerances["analytic_jvp_rtol"]),
        "radial_autodiff_identity": scaled_error(radial_autodiff, radial_expected, tolerances["radial_atol"], tolerances["analytic_jvp_rtol"]),
        "frozen_scale_adversary_rejected": scaled_error(adversarial, radial_expected, tolerances["radial_atol"], tolerances["analytic_jvp_rtol"]),
    }
    checks = {
        "analytic_matches_autodiff": bool(torch.allclose(analytic, automatic, atol=tolerances["analytic_jvp_atol"], rtol=tolerances["analytic_jvp_rtol"])),
        "dense_analytic_matches_autodiff": bool(torch.allclose(dense, dense_autodiff, atol=tolerances["analytic_jvp_atol"], rtol=tolerances["analytic_jvp_rtol"])),
        "radial_identity": bool(torch.allclose(radial, radial_expected, atol=tolerances["radial_atol"], rtol=tolerances["analytic_jvp_rtol"])),
        "radial_autodiff_identity": bool(torch.allclose(radial_autodiff, radial_expected, atol=tolerances["radial_atol"], rtol=tolerances["analytic_jvp_rtol"])),
        "frozen_scale_adversary_rejected": not bool(torch.allclose(adversarial, radial_expected, atol=tolerances["radial_atol"], rtol=tolerances["analytic_jvp_rtol"])),
        "finite_difference_sweep": min(row["relative_error"] for row in sweep) <= tolerances["fd_best_relative_error"],
    }
    dtype_rows = []
    for name, dtype, prefix in (("float32", torch.float32, "fp32"), ("float16", torch.float16, "fp16"), ("bfloat16", torch.bfloat16, "bfloat16")):
        mm, ww, gg, vv = m.to(dtype), W.to(dtype), weight.to(dtype), direction.to(dtype)
        native = lambda value: normalization_update(value, ww, gg, eps, semantics="gemma")
        _, native_jvp = torch.autograd.functional.jvp(native, mm, vv)
        # Reference uses the quantized inputs/weights but float64 operations.
        reference = normalization_jvp(mm.double(), vv.double(), ww.double(), gg.double(), eps)
        atol, rtol = tolerances[f"{prefix}_jvp_atol"], tolerances[f"{prefix}_jvp_rtol"]
        passed = bool(torch.allclose(native_jvp.double(), reference, atol=atol, rtol=rtol))
        checks[f"{name}_cast_semantics"] = passed
        dtype_rows.append({
            "input_projection_output_dtype": name, "normalizer_compute_dtype": "float32",
            "max_absolute_jvp_error": float((native_jvp.double() - reference).abs().max()),
            "relative_jvp_error": float((native_jvp.double() - reference).norm() / reference.norm().clamp_min(1e-15)),
            "max_scaled_tolerance_error": scaled_error(native_jvp.double(), reference, atol, rtol),
            "absolute_tolerance": atol, "relative_tolerance": rtol, "passed": passed,
            "finite_difference_sweep": central_difference_sweep(native, mm, vv, config["fd_steps"], native_jvp),
            "finite_difference_interpretation": "reported_only_cast_roundoff_can_dominate_small_steps",
        })
    return {
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "scaled_tolerance_errors": scaled_errors,
        "float64_analytic_autodiff_max_absolute_error": float((analytic - automatic).abs().max()),
        "float64_dense_jacobian_max_absolute_error": float((dense - dense_autodiff).abs().max()),
        "radial_identity_max_absolute_error": float((radial - radial_expected).abs().max()),
        "radial_expected_norm": float(radial_expected.norm()),
        "radial_frozen_scale_control_norm": float(adversarial.norm()),
        "radial_frozen_scale_error_norm": float((adversarial - radial_expected).norm()),
        "float64_finite_difference_sweep": sweep, "dtype_semantics": dtype_rows,
        "scope": "generated_matrices_only_not_target_model_derivative_evidence",
    }


def run_synthetic_qualification(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the frozen, small generated-data comparison; no input data paths.

    The CLI authenticates the exact configuration.  This numerical function is
    independently usable on synthetic dimensions for tests, with no campaign
    authority.  Every learned quantity is computed before evaluation draws.
    """
    start = time.perf_counter()
    helpers = _existing_chart_primitives()
    d, p, k, s, r = (config[key] for key in ("residual_width", "message_width", "output_rank", "state_rank", "message_rank"))
    C, eps, seed = config["chart_count"], config["epsilon"], config["seed"]
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(shape, generator=generator, dtype=torch.float64)
    W, weight = randn(d, p) / math.sqrt(p), randn(d) * 0.15
    message_map, centers = randn(p, d) / math.sqrt(d), randn(C, d) * 1.5

    def draw(families: int, offset: int) -> tuple[Tensor, Tensor, list[str]]:
        states, messages, ids = [], [], []
        for family in range(families):
            center = centers[family % C] + 0.25 * randn(d)
            for _ in range(config["rows_per_family"]):
                h = center + 0.3 * randn(d)
                m = message_map @ h + 0.2 * randn(p)
                states.append(h); messages.append(m); ids.append(f"synthetic-family-{offset + family}")
        return torch.stack(states), torch.stack(messages), ids

    train_h, train_m, train_ids = draw(config["train_families"], 0)
    fn = lambda value: normalization_update(value, W, weight, eps)
    train_y = fn(train_m)
    # Gradients of explicit synthetic linear scores c_i^T delta.  This is a
    # score-gradient second moment, not model Fisher or task-label evidence.
    score_rows = train_y @ (torch.eye(d, dtype=torch.float64) + 0.1 * randn(d, d)) + 0.25 * randn(*train_y.shape)
    carrier = helpers.Rank64FisherCarrier.from_gradient_rows(score_rows, rank=d)
    learned_basis = helpers.LowRankFisherChartBasis.from_samples(carrier=carrier, samples=train_y, coordinate_rank=k, regularization=0.0)
    projection = FisherProjection(learned_basis.ambient_basis, carrier.q * carrier.eigenvalues.sqrt(), rtol=config["metric_rtol"])
    U, Q = _orthogonal_rows(train_h, s), _orthogonal_rows(train_m, r)
    h_scale, m_scale = float(train_h.square().mean().sqrt()), float(train_m.square().mean().sqrt())
    feature = lambda h, m: torch.cat((h / h_scale, m / m_scale), dim=-1)
    train_features = feature(train_h, train_m)
    signature = lambda m: projection.coordinate_map @ jvp_actions(fn, m, Q)
    train_A = torch.stack([signature(m) for m in train_m])
    activation_groups = helpers.deterministic_kmeans_assignments(train_features, chart_count=C, seed=seed, maximum_iterations=config["kmeans_iterations"])
    response_groups = helpers.deterministic_kmeans_assignments(train_A.flatten(start_dim=1), chart_count=C, seed=seed, maximum_iterations=config["kmeans_iterations"])
    shuffled = response_groups.assignments[torch.randperm(len(train_h), generator=generator)]
    memberships = {"global_taylor_affine": torch.zeros(len(train_h), dtype=torch.long), "activation_charts": activation_groups.assignments, "response_charts": response_groups.assignments, "shuffled_charts": shuffled}
    banks, routers, anchors = {}, {}, {}
    for name, assignment in memberships.items():
        charts, router, anchor_ids = [], [], []
        for c in range(int(assignment.max()) + 1):
            indices = torch.where(assignment == c)[0]
            feature_center = train_features[indices].mean(dim=0)
            anchor = int(indices[(train_features[indices] - feature_center).square().sum(dim=1).argmin()])
            h_anchor, m_anchor = train_h[anchor], train_m[anchor]
            charts.append(AffineResponseChart(
                h_anchor, m_anchor, projection.encode(train_y[anchor]), torch.zeros(k, s, dtype=torch.float64),
                train_A[anchor], U, Q, projection.basis,
                max(float((train_h[indices] - h_anchor).norm(dim=1).max()) * config["support_radius_multiplier"], 1e-12),
                max(float((train_m[indices] - m_anchor).norm(dim=1).max()) * config["support_radius_multiplier"], 1e-12),
            ))
            router.append(feature_center); anchor_ids.append(anchor)
        banks[name], routers[name], anchors[name] = charts, torch.stack(router), anchor_ids
    # An additional fitted affine control avoids making the single anchor the
    # only global comparator.  It is labeled separately from analytic A_C.
    train_design = torch.cat((torch.ones(len(train_h), 1, dtype=torch.float64), train_h @ U, train_m @ Q), dim=1)
    global_fit = torch.linalg.lstsq(train_design, projection.encode(train_y), driver="gelsd", rcond=config["metric_rtol"]).solution
    freeze_seconds = time.perf_counter() - start
    # All metric/basis/chart/router/fit decisions above are now frozen.
    eval_h, eval_m, eval_ids = draw(config["evaluation_families"], config["train_families"])
    assert not set(train_ids).intersection(eval_ids)
    actual = fn(eval_m)
    eval_features = feature(eval_h, eval_m)
    eval_A = torch.stack([signature(m) for m in eval_m])
    oracle_assignments = {
        "global_taylor_affine": torch.zeros(len(eval_h), dtype=torch.long),
        "activation_charts": activation_groups.assign(eval_features),
        "response_charts": response_groups.assign(eval_A.flatten(start_dim=1)),
        "shuffled_charts": shuffled[torch.randint(len(shuffled), (len(eval_h),), generator=generator)],
    }
    predictions, stability, routing, finite_rows = {}, {}, {}, {}
    artifact_banks, costs = {}, {}
    basis_scalars = U.numel() + Q.numel() + projection.basis.numel()
    for name, charts in banks.items():
        routed = torch.cdist(eval_features, routers[name]).argmin(dim=1)
        oracle = oracle_assignments[name]
        predict = lambda assignments: torch.stack([charts[int(c)].predict(h, m) for c, h, m in zip(assignments, eval_h, eval_m)])
        pred, oracle_pred = predict(routed), predict(oracle)
        support = torch.tensor([charts[int(c)].in_support(h, m) for c, h, m in zip(routed, eval_h, eval_m)])
        support_distances = torch.tensor([
            [max(float((h - c.state_anchor).norm()) / c.state_radius,
                 float((m - c.message_anchor).norm()) / c.message_radius) for c in charts]
            for h, m in zip(eval_h, eval_m)
        ], dtype=torch.float64)
        predictions[name] = pred
        routed_loss = projection.fisher_squared(pred - actual)
        oracle_loss = projection.fisher_squared(oracle_pred - actual)
        predicted_A = torch.stack([charts[int(c)].A for c in routed])
        within_variance = []
        for c in range(len(charts)):
            rows = train_A[memberships[name] == c]
            within_variance.append(float((rows - rows.mean(dim=0)).square().mean()))
        stability[name] = {
            "train_occupancy": torch.bincount(memberships[name], minlength=len(charts)).tolist(),
            "evaluation_routed_occupancy": torch.bincount(routed, minlength=len(charts)).tolist(),
            "evaluation_oracle_occupancy": torch.bincount(oracle, minlength=len(charts)).tolist(),
            "within_chart_A_mean_squared_variation": within_variance,
            "within_chart_B_variation": 0.0,
            "support_rejection_rate": float((~support).double().mean()),
            "rejected_rows": int((~support).sum()),
            "total_rows": len(eval_h),
            "rejection_rule": "train_frozen_full_h_and_m_euclidean_radii_no_teacher_features",
            "normalized_routed_support_distances": support_distances[torch.arange(len(routed)), routed].tolist(),
            "normalized_nearest_support_distances": support_distances.min(dim=1).values.tolist(),
            "normalized_support_inclusive_limit": 1.0,
        }
        routing[name] = {
            "assignment_agreement": float((routed == oracle).double().mean()),
            "mean_excess_fisher_squared_error_vs_oracle_assignment": float((routed_loss - oracle_loss).mean()),
            "oracle_assignment_error": _errors(oracle_pred, actual, projection),
            "projected_derivative_relative_l2": float((predicted_A - eval_A).norm() / eval_A.norm().clamp_min(1e-15)),
            "oracle_requires_teacher_derivatives": name == "response_charts",
        }
        finite_rows[name] = {
            "unchecked_all_rows": _errors(pred, actual, projection),
            "supported_rows": _errors(pred[support], actual[support], projection) if bool(support.any()) else None,
            "supported_row_count": int(support.sum()),
            "offset_projection_relative_l2": _errors(torch.stack([projection.project(train_y[a]) for a in anchors[name]]), train_y[anchors[name]], projection)["euclidean_relative_l2"],
            "coefficient_offset_fit_error": 0.0,
            "offset_fit_rule": "exact_projected_training_medoid_update_no_fitted_offset_substitution",
        }
        artifact_banks[name] = {
            "charts": [{"state_anchor": _tensor_list(c.state_anchor), "message_anchor": _tensor_list(c.message_anchor), "beta": _tensor_list(c.beta), "B": _tensor_list(c.B), "A": _tensor_list(c.A), "state_radius": c.state_radius, "message_radius": c.message_radius} for c in charts],
            "router_centers": _tensor_list(routers[name]), "training_anchor_row_indices": anchors[name],
        }
        chart_scalars = sum(c.state_anchor.numel() + c.message_anchor.numel() + c.beta.numel() + c.B.numel() + c.A.numel() + 2 for c in charts)
        router_scalars = routers[name].numel() + 2  # common h/m scales
        multiply_adds = d*s + p*r + k*s + k*r + d*k
        costs[name] = {
            "runtime_float64_scalars": basis_scalars + chart_scalars + router_scalars,
            "runtime_tensor_bytes": 8 * (basis_scalars + chart_scalars + router_scalars),
            "shared_basis_scalars": basis_scalars, "chart_scalars_including_offsets_anchors_support": chart_scalars,
            "router_scalars_including_scales": router_scalars,
            "dense_projection_prediction_multiply_accumulates_per_row": multiply_adds,
            "router_distance_dimensions_per_row": len(charts) * (d + p),
            "support_distance_dimensions_per_row": d + p,
            "state_message_centering_subtractions_per_row": d + p,
            "coefficient_additions_per_row": 2*k,
            "source_W_or_norm_weights_required_for_prediction": False,
            "attention_message_required": True,
        }

    eval_design = torch.cat((torch.ones(len(eval_h), 1, dtype=torch.float64), eval_h @ U, eval_m @ Q), dim=1)
    predictions["global_least_squares_affine"] = projection.decode(eval_design @ global_fit)
    finite_rows["global_least_squares_affine"] = {"unchecked_all_rows": _errors(predictions["global_least_squares_affine"], actual, projection), "fit_rule": "training_only_least_squares_coefficients_not_anchor_Jacobians"}
    finite_rows["exact_analytic_normalization"] = {"unchecked_all_rows": _errors(fn(eval_m), actual, projection), "accuracy_role": "exact_teacher_reference_not_an_approximate_competitor"}
    finite_rows["exact_analytic_projected_normalization"] = {"unchecked_all_rows": _errors(projection.project(actual), actual, projection), "accuracy_role": "exact_within_chosen_output_span"}

    # New directions are drawn after every fitting decision.  They were not
    # used to discover either input or output bases.
    directions = randn(*eval_m.shape)
    directions /= directions.norm(dim=1, keepdim=True)
    omitted = directions - directions @ Q @ Q.T
    full_actions = torch.stack([normalization_jvp(m, v, W, weight, eps) for m, v in zip(eval_m, directions)])
    omitted_actions = torch.stack([normalization_jvp(m, v, W, weight, eps) for m, v in zip(eval_m, omitted)])
    state_directions = randn(*eval_h.shape)
    state_directions /= state_directions.norm(dim=1, keepdim=True)
    omitted_state_directions = state_directions - state_directions @ U @ U.T
    response_span = _errors(projection.project(full_actions), full_actions, projection)
    radius_rows = []
    for radius in config["finite_radii"]:
        observed = fn(eval_m + radius * directions) - actual
        per_state_full = radius * full_actions
        per_state_compact = torch.stack([projection.basis @ a @ (Q.T @ (radius * v)) for a, v in zip(eval_A, directions)])
        methods = {}
        for name, charts in banks.items():
            assignments = torch.cdist(eval_features, routers[name]).argmin(dim=1)
            approximated = torch.stack([c.G @ c.A @ (c.Q.T @ (radius * v)) for c, v in zip((charts[int(i)] for i in assignments), directions)])
            methods[name] = _errors(approximated, observed, projection)
        radius_rows.append({"radius": radius, "perturbation_units": "unit_euclidean_message_direction", "exact_per_state_derivative_oracle": _errors(per_state_full, observed, projection), "projected_per_state_derivative_oracle": _errors(per_state_compact, observed, projection), "chart_derivative_predictions_fixed_base_assignment": methods})

    pairwise = {}
    candidate_loss = projection.fisher_squared(predictions["response_charts"] - actual)
    for baseline in ("activation_charts", "global_taylor_affine", "global_least_squares_affine", "shuffled_charts"):
        difference = candidate_loss - projection.fisher_squared(predictions[baseline] - actual)
        interval = helpers.sequence_bootstrap(difference, eval_ids, replicates=config["bootstrap_replicates"], seed=seed)
        pairwise[baseline] = {"mean_response_minus_baseline_fisher_squared_error": float(interval.point), "paired_family_bootstrap_95_percent_interval": [float(interval.lower), float(interval.upper)], "favorable_sign": "negative", "synthetic_only": True}

    derivative = _derivative_qualification(config, W, weight, train_m[0])
    fixed_analysis = metric_operator_analysis(torch.eye(k, dtype=torch.float64), projection.gram)
    subspace = {
        "status": "measured", "metric": "training_synthetic_linear_score_gradient_second_moment",
        "metric_is_target_model_fisher": False, "metric_supported_rank": carrier.rank,
        "generator_rank": k, "state_input_rank": s, "message_input_rank": r,
        "generator_gram_identity_max_absolute_error": float((projection.gram - torch.eye(k, dtype=torch.float64)).abs().max()),
        "independent_direction_count": len(directions),
        "independent_response_outside_G": response_span,
        "omitted_message_direction_energy_fraction": float(omitted.square().sum() / directions.square().sum()),
        "omitted_message_response_energy_fraction": float(omitted_actions.square().sum() / full_actions.square().sum().clamp_min(1e-30)),
        "omitted_state_direction_energy_fraction": float(omitted_state_directions.square().sum() / state_directions.square().sum()),
        "omitted_state_response_energy": 0.0, "state_omission_reason": "K_is_identically_zero_at_current_boundary",
        "fixed_message_projected_J_singular_values": _tensor_list(fixed_analysis["singular_values"]),
        "fixed_message_J_projection_leakage_norm": float((projection.basis - projection.basis @ (projection.coordinate_map @ projection.basis)).norm()),
        "fixed_message_identity_interpretation": "forced_identity_not_evidence_of_attention_dynamics",
    }
    # Timings include feature construction, route, support, coefficients and
    # reconstruction; tiny synthetic CPU timings authorize no speedup claim.
    timing = {}
    for name, charts in banks.items():
        begin = time.perf_counter()
        for h, m in zip(eval_h, eval_m):
            c = charts[int(torch.cdist(feature(h, m).unsqueeze(0), routers[name]).argmin())]
            c.in_support(h, m)
            c.predict(h, m)
        timing[name] = (time.perf_counter() - begin) / len(eval_h)
    begin = time.perf_counter()
    for m in eval_m:
        fn(m)
    timing["exact_analytic_normalization"] = (time.perf_counter() - begin) / len(eval_m)
    begin = time.perf_counter()
    for h, m in zip(eval_h, eval_m):
        design = torch.cat((torch.ones(1, dtype=torch.float64), h @ U, m @ Q))
        projection.decode(design @ global_fit)
    timing["global_least_squares_affine"] = (time.perf_counter() - begin) / len(eval_m)
    begin = time.perf_counter()
    for m in eval_m:
        projection.project(fn(m))
    timing["exact_analytic_projected_normalization"] = (time.perf_counter() - begin) / len(eval_m)
    fitted_scalars = basis_scalars + global_fit.numel()
    costs["global_least_squares_affine"] = {"runtime_float64_scalars": fitted_scalars, "runtime_tensor_bytes": 8*fitted_scalars, "dense_prediction_multiply_accumulates_per_row": d*s + p*r + (1+s+r)*k + d*k, "support_policy": "global_unchecked_control"}
    costs["exact_analytic_normalization"] = {"runtime_float64_scalars": W.numel() + weight.numel() + 1, "runtime_tensor_bytes": 8*(W.numel() + weight.numel() + 1), "projection_multiply_accumulates_per_row": d*p, "rms_square_reduce_scale_width": d, "square_roots_per_row": 1, "attention_message_required": True}
    projected_reference_scalars = W.numel() + weight.numel() + 1 + projection.coordinate_map.numel() + projection.basis.numel()
    costs["exact_analytic_projected_normalization"] = {"runtime_float64_scalars": projected_reference_scalars, "runtime_tensor_bytes": 8*projected_reference_scalars, "projection_multiply_accumulates_per_row": d*p + 2*d*k, "rms_square_reduce_scale_width": d, "square_roots_per_row": 1, "attention_message_required": True}
    return {
        "derivative_correctness": derivative,
        "response_subspace_adequacy": subspace,
        "chart_stability": {"status": "measured", "methods": stability, "matched_multi_chart_count": C, "shuffled_training_occupancy_matches_response": sorted(stability["response_charts"]["train_occupancy"]) == sorted(stability["shuffled_charts"]["train_occupancy"]), "stability_scope": "within_fit_chart_variation_and_evaluation_occupancy_not_refit_stability"},
        "finite_update_fidelity": {"status": "measured", "methods": finite_rows, "finite_perturbation_sweep": radius_rows, "paired_synthetic_family_uncertainty": pairwise, "independent_sampling_unit": "generated_family", "evaluation_family_count": len(set(eval_ids)), "acceptance": "no_required_chart_win_for_synthetic_qualification_no_model_inference", "oracle_warning": "per_state_derivatives_require_teacher_access_and_do_not_predict_full_updates_without_offsets"},
        "router_feasibility": {"status": "measured", "features": "full_h_and_m_each_scaled_by_training_RMS", "feature_availability": "m_exists_only_after_original_attention_normalization_target_is_downstream", "online_teacher_jacobians_required": False, "target_residuals_future_tokens_evaluation_labels_used_by_router": False, "methods": routing, "real_runtime_feasibility": "not_evaluated"},
        "runtime_and_storage_accounting": {
            "status": "measured", "methods": costs, "synthetic_cpu_seconds_per_row_single_pass": timing,
            "training_freeze_seconds": freeze_seconds, "total_synthetic_seconds": time.perf_counter() - start,
            "training_rows": len(train_h), "evaluation_rows": len(eval_h),
            "training_JVP_actions": len(train_h)*r, "evaluation_oracle_JVP_actions": len(eval_h)*r,
            "metric_basis_discovery_and_router_fit_scope": "training_families_only",
            "shared_offline_metric_factor_scalars": projection.factor.numel(), "shared_offline_coordinate_map_scalars": projection.coordinate_map.numel(),
            "shared_offline_response_discovery_center_scalars": response_groups.centers.numel(),
            "native_attention_cost": "required_but_not_executed_or_measured_in_synthetic_qualification",
            "common_residual_skip_additions_per_row_for_h_next": d,
            "speedup_or_compression_claim": False,
            "logical_cost_convention": "multiply_accumulates_counted_once_not_claimed_as_hardware_FLOPs; distances_squares_reductions_comparisons_reported_separately; no_online_JVPs",
            "runtime_tensor_byte_scope": "per_method_shared_basis_numeric_payload_only_not_peak_RSS_Python_objects_or_temporary_tensors; report_and_manifest_file_bytes_are_recorded_separately",
            "artifact": {"G": _tensor_list(projection.basis), "U": _tensor_list(U), "Q": _tensor_list(Q), "F_factor": _tensor_list(projection.factor), "D_G": _tensor_list(projection.coordinate_map), "offline_response_signature_centers": _tensor_list(response_groups.centers), "h_feature_scale": h_scale, "m_feature_scale": m_scale, "banks": artifact_banks, "global_least_squares_coefficients": _tensor_list(global_fit), "synthetic_reference_W": _tensor_list(W), "synthetic_reference_norm_weight": _tensor_list(weight), "epsilon": eps},
        },
    }
