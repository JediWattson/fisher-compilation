"""One training-only compact-intercept compatibility ablation.

The sole candidate uses lambda=1.  Lambda=0 rebuilds the original discovery
and anchored-chart construction as an implementation control.  No function
in this module evaluates the lambda=1 candidate.  Privileged discovery
features and the observable-only runtime router have separate interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import itertools
import json
import math
from typing import Any

import torch
from torch import Tensor


RELATIVE_CONSTANT_TOLERANCE = 1e-12
REPRODUCTION_ATOL = 1e-12
REPRODUCTION_RTOL = 1e-10
ARTIFACT_SCHEMA = "fisher_graph.response_geometry.intercept_compatibility.arm.v1"
_GEOMETRY_KEYS = ("G", "U", "Q", "D_G", "F_factor")
_TRAINING_ARRAY_KEYS = (
    "training_A", "training_beta", "training_b", "training_intercepts",
    "training_outputs", "training_signatures", "raw_response_signatures",
    "chart_intercepts",
)


def _tensor(values: Tensor, name: str, ndim: int | None = None) -> Tensor:
    if (not isinstance(values, Tensor) or values.dtype != torch.float64
            or values.device.type != "cpu" or (ndim is not None and values.ndim != ndim)
            or values.numel() == 0 or not bool(torch.isfinite(values).all())):
        raise ValueError(f"{name} must be a nonempty finite float64 CPU tensor")
    return values


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} has an invalid sign or zero value")
    return value


def fit_block_transform(
    values: Tensor, relative_constant_tolerance: float = RELATIVE_CONSTANT_TOLERANCE,
) -> dict[str, Any]:
    """Fit the frozen scale-relative rule with numerically scaled float64 norms.

    Global units never enter the drop decision: it compares two norms in the
    same temporary arithmetic scale.  An unrepresentable required scale is a
    numerical failure, not evidence that a block is constant.
    """
    values = _tensor(values, "training block", 2)
    if relative_constant_tolerance != RELATIVE_CONSTANT_TOLERANCE:
        raise ValueError("the relative constant tolerance is frozen at 1e-12")
    maximum = float(values.abs().max())
    if maximum == 0:
        return {
            "mean": torch.zeros(values.shape[1], dtype=torch.float64),
            "reference_scale": 0.0, "centered_scale": 0.0,
            "centered_to_reference_ratio": None,
            "relative_constant_tolerance": RELATIVE_CONSTANT_TOLERANCE,
            "dropped": True, "reason": "identically_zero",
            "training_row_count": len(values), "width": values.shape[1],
        }
    scaled = values / maximum
    scaled_mean = scaled.mean(dim=0)
    centered = scaled - scaled_mean
    reference_unit = float(scaled.square().sum(dim=1).mean().sqrt())
    centered_unit = float(centered.square().sum(dim=1).mean().sqrt())
    reference = maximum * reference_unit
    centered_scale = maximum * centered_unit
    if (not all(math.isfinite(v) for v in (reference_unit, centered_unit, reference, centered_scale))
            or reference <= 0 or (centered_unit > 0 and centered_scale == 0)):
        raise ValueError("block scales are not representable in float64")
    ratio = centered_unit / reference_unit
    mean = scaled_mean * maximum
    if not math.isfinite(ratio) or not bool(torch.isfinite(mean).all()):
        raise ValueError("block normalization produced a nonfinite calculation")
    dropped = ratio <= RELATIVE_CONSTANT_TOLERANCE
    return {
        "mean": mean, "reference_scale": reference,
        "centered_scale": centered_scale, "centered_to_reference_ratio": ratio,
        "relative_constant_tolerance": RELATIVE_CONSTANT_TOLERANCE,
        "dropped": dropped, "reason": "relative_constant" if dropped else None,
        "training_row_count": len(values), "width": values.shape[1],
    }


def apply_block_transform(values: Tensor, transform: Mapping[str, Any]) -> Tensor:
    """Apply saved training statistics; evaluation never changes drop decisions."""
    values = _tensor(values, "block values", 2)
    mean = _tensor(transform["mean"], "saved block mean", 1)
    if values.shape[1] != len(mean) or transform["width"] != len(mean):
        raise ValueError("block width differs from its saved training transform")
    if transform["relative_constant_tolerance"] != RELATIVE_CONSTANT_TOLERANCE:
        raise ValueError("saved block tolerance differs from the frozen rule")
    if type(transform["dropped"]) is not bool:
        raise ValueError("saved block drop decision must be boolean")
    reference = _number(transform["reference_scale"], "saved reference scale")
    centered = _number(transform["centered_scale"], "saved centered scale")
    if transform["dropped"]:
        if reference == 0:
            if centered != 0 or transform["reason"] != "identically_zero":
                raise ValueError("inconsistent zero-block metadata")
        elif transform["reason"] != "relative_constant":
            raise ValueError("inconsistent constant-block metadata")
        return torch.zeros_like(values)
    if reference <= 0 or centered <= 0 or transform["reason"] is not None:
        raise ValueError("retained block has invalid saved scales or drop reason")
    # This is arithmetic rescaling of the SAME saved affine transform.  It is
    # not a new statistical normalization or a new constant-block decision.
    arithmetic_scale = max(float(values.abs().max()), float(mean.abs().max()), centered)
    denominator = centered / arithmetic_scale
    if denominator == 0 or not math.isfinite(denominator):
        raise ValueError("block application cannot represent its saved normalization")
    result = (values / arithmetic_scale - mean / arithmetic_scale) / denominator
    if not bool(torch.isfinite(result).all()):
        raise ValueError("block application produced nonfinite normalized values")
    return result


def supported_gram_square_root(
    G: Tensor, F_factor: Tensor, metric_rtol: float = 1e-12,
) -> Tensor:
    """Symmetric H_G root on the original numerically supported subspace."""
    G, F_factor = _tensor(G, "G", 2), _tensor(F_factor, "Fisher factor", 2)
    if G.shape[0] != F_factor.shape[0]:
        raise ValueError("generator and Fisher factor widths differ")
    rtol = _number(metric_rtol, "metric rtol", positive=True)
    weighted = F_factor.T @ G
    gram = weighted.T @ weighted
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("Fisher Gram computation is nonfinite")
    values, vectors = torch.linalg.eigh(gram)
    cutoff = rtol * max(float(values.abs().max()), torch.finfo(torch.float64).tiny)
    if float(values.min()) < -cutoff:
        raise ValueError("Fisher Gram is not positive semidefinite")
    supported = values > cutoff
    if not bool(supported.any()):
        raise ValueError("Fisher Gram has no supported positive subspace")
    result = (vectors[:, supported] * values[supported].sqrt()) @ vectors[:, supported].T
    return _tensor(result, "supported Fisher Gram square root", 2)


def pointwise_compact_intercepts(
    messages: Tensor, outputs: Tensor, A: Tensor, D_G: Tensor, Q: Tensor, G: Tensor,
) -> dict[str, Tensor]:
    """Pointwise compact tangent intercepts, independent of any old assignment."""
    messages, outputs, A = (_tensor(messages, "messages", 2), _tensor(outputs, "outputs", 2), _tensor(A, "projected response operators", 3))
    D_G, Q, G = _tensor(D_G, "D_G", 2), _tensor(Q, "Q", 2), _tensor(G, "G", 2)
    n, p = messages.shape
    if (len(outputs) != n or A.shape != (n, G.shape[1], Q.shape[1])
            or Q.shape[0] != p or D_G.shape != (G.shape[1], outputs.shape[1])
            or G.shape[0] != outputs.shape[1]):
        raise ValueError("compact intercept dimensions do not match")
    beta = outputs @ D_G.T
    a = messages @ Q
    b = beta - torch.bmm(A, a.unsqueeze(-1)).squeeze(-1)
    ambient = b @ G.T
    for name, value in (("beta", beta), ("a", a), ("b", b), ("ambient", ambient)):
        _tensor(value, name, 2)
    return {"beta": beta, "a": a, "b": b, "ambient": ambient}


def runtime_features(arm: Mapping[str, Any], h: Tensor, m: Tensor) -> Tensor:
    h, m = _tensor(h, "runtime states", 2), _tensor(m, "runtime messages", 2)
    if len(h) != len(m):
        raise ValueError("runtime state and message rows differ")
    h_scale = _number(arm["h_feature_scale"], "runtime state feature scale", positive=True)
    m_scale = _number(arm["m_feature_scale"], "runtime message feature scale", positive=True)
    return _tensor(torch.cat((h / h_scale, m / m_scale), dim=-1), "runtime features", 2)


def assign_runtime(arm: Mapping[str, Any], h: Tensor, m: Tensor) -> Tensor:
    """Only h, m, their original saved scales and fitted centroids are accessed."""
    return torch.cdist(runtime_features(arm, h, m), arm["router_centers"]).argmin(dim=1)


def discovery_features(
    arm: Mapping[str, Any], messages: Tensor, outputs: Tensor, A: Tensor,
) -> Tensor:
    """Target-informed diagnostic features; this is never the runtime router."""
    A = _tensor(A, "diagnostic projected response operators", 3)
    response = A.flatten(start_dim=1)
    if arm["compatibility_weight"] == 0:
        return response
    if arm["compatibility_weight"] != 1:
        raise ValueError("only the frozen lambda=0 control and lambda=1 candidate exist")
    geometry = arm["fixed_geometry"]
    pointwise = pointwise_compact_intercepts(messages, outputs, A, geometry["D_G"], geometry["Q"], geometry["G"])
    intercept = pointwise["b"] @ arm["sqrt_H"].T
    return torch.cat((
        apply_block_transform(response, arm["blocks"]["response"]),
        apply_block_transform(intercept, arm["blocks"]["intercept"]),
    ), dim=1)


def _json_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise ValueError(f"unsupported arm artifact value: {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _training_identity(replay: Mapping[str, Any]) -> str:
    tensors, saved = replay["tensors"], replay["saved_artifact"]
    payload = {
        "config": replay["config"], "train_h": replay["train_h"], "train_m": replay["train_m"],
        "fixed_tensors": {name: tensors[name] for name in (*_GEOMETRY_KEYS, "W", "norm_weight")},
        "h_feature_scale": saved["h_feature_scale"], "m_feature_scale": saved["m_feature_scale"],
    }
    return hashlib.sha256(b"intercept-compatibility/training-binding/v1\0" + _canonical(payload)).hexdigest()


def fit_arm(replay: Mapping[str, Any], compatibility_weight: int = 1) -> dict[str, Any]:
    """Fit one declared arm from exactly six training/fixed-data replay fields.

    The lambda=0 and lambda=1 arms share this complete rebuilding pipeline.
    Saved old chart coefficients, labels, medoids and router centers are not
    inputs.  The old banks are used only by ``verify_lambda_zero`` afterwards.
    """
    if type(compatibility_weight) is not int or compatibility_weight not in (0, 1):
        raise ValueError("compatibility weight must be exactly integer 0 or 1")
    config, kernel, tensors = replay["config"], replay["kernel"], replay["tensors"]
    saved = replay["saved_artifact"]
    h, m = _tensor(replay["train_h"], "training states", 2), _tensor(replay["train_m"], "training messages", 2)
    geometry = {name: tensors[name] for name in _GEOMETRY_KEYS}
    G, U, Q, D_G, F_factor = (geometry[name] for name in _GEOMETRY_KEYS)
    phi_fn = lambda messages: kernel.normalization_update(messages, tensors["W"], tensors["norm_weight"], config["epsilon"])
    outputs = phi_fn(m)
    A = torch.stack([D_G @ kernel.jvp_actions(phi_fn, message, Q) for message in m])
    pointwise = pointwise_compact_intercepts(m, outputs, A, D_G, Q, G)
    root = supported_gram_square_root(G, F_factor, config["metric_rtol"])
    raw_response = A.flatten(start_dim=1)
    intercept = pointwise["b"] @ root.T
    arm = {
        "compatibility_weight": compatibility_weight, "config": copy.deepcopy(config),
        "fixed_training_sha256": _training_identity(replay), "fixed_geometry": geometry,
        "h_feature_scale": saved["h_feature_scale"], "m_feature_scale": saved["m_feature_scale"],
        "sqrt_H": root,
        "blocks": {"response": fit_block_transform(raw_response), "intercept": fit_block_transform(intercept)},
        "training_A": A, "training_beta": pointwise["beta"], "training_b": pointwise["b"],
        "training_intercepts": pointwise["ambient"], "training_outputs": outputs,
        "raw_response_signatures": raw_response,
    }
    # Lambda=0 delegates to the raw original response signature.  No centering
    # or block normalization changes that implementation control's discovery.
    signatures = discovery_features(arm, m, outputs, A)
    groups = kernel._existing_chart_primitives().deterministic_kmeans_assignments(
        signatures, chart_count=config["chart_count"], seed=config["seed"],
        maximum_iterations=config["kmeans_iterations"],
    )
    features = runtime_features(arm, h, m)
    charts, centroids, anchors = [], [], []
    for chart_index in range(config["chart_count"]):
        indices = torch.where(groups.assignments == chart_index)[0]
        feature_center = features[indices].mean(dim=0)
        anchor = int(indices[(features[indices] - feature_center).square().sum(dim=1).argmin()])
        charts.append(kernel.AffineResponseChart(
            h[anchor], m[anchor], pointwise["beta"][anchor],
            torch.zeros(G.shape[1], U.shape[1], dtype=torch.float64), A[anchor], U, Q, G,
            max(float((h[indices] - h[anchor]).norm(dim=1).max()) * config["support_radius_multiplier"], 1e-12),
            max(float((m[indices] - m[anchor]).norm(dim=1).max()) * config["support_radius_multiplier"], 1e-12),
        ))
        centroids.append(feature_center)
        anchors.append(anchor)
    intercepts = torch.stack([chart.G @ chart.beta - chart.G @ chart.A @ chart.Q.T @ chart.message_anchor for chart in charts])
    algebra_error = 0.0
    algebra_passed = True
    for chart, compact in zip(charts, intercepts):
        anchored = torch.stack([chart.predict(state, message) for state, message in zip(h, m)])
        reconstructed = compact + (m @ chart.Q) @ chart.A.T @ chart.G.T
        algebra_error = max(algebra_error, float((anchored - reconstructed).abs().max()))
        algebra_passed &= bool(torch.allclose(anchored, reconstructed, atol=REPRODUCTION_ATOL, rtol=REPRODUCTION_RTOL))
    if not algebra_passed:
        raise ValueError("rebuilt chart intercept does not match its anchored predictor")
    arm.update({
        "charts": charts, "training_assignments": groups.assignments.clone(),
        "discovery_centers": groups.centers.clone(), "router_centers": torch.stack(centroids),
        "anchor_indices": anchors, "training_signatures": signatures,
        "chart_intercepts": intercepts,
        "construction_checks": {
            "stored_compact_intercept_matches_implemented_predictor": algebra_passed,
            "stored_compact_intercept_max_absolute_error": algebra_error,
            "all_state_response_operators_zero": all(bool((chart.B == 0).all()) for chart in charts),
            "evaluation_rows_used_in_fitting": False,
            "runtime_features": "original_full_h_and_m_divided_by_original_training_RMS_scales",
            "soft_compatibility_only_no_hard_dispersion_guarantee": True,
        },
    })
    return arm


def serialize_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": ARTIFACT_SCHEMA,
        **{name: arm[name] for name in (
            "compatibility_weight", "config", "fixed_training_sha256", "fixed_geometry",
            "h_feature_scale", "m_feature_scale", "sqrt_H", "blocks", "training_assignments",
            "discovery_centers", "router_centers", "anchor_indices", "construction_checks",
            *_TRAINING_ARRAY_KEYS,
        )},
        "charts": [{name: getattr(chart, name) for name in (
            "state_anchor", "message_anchor", "beta", "B", "A", "state_radius", "message_radius",
        )} for chart in arm["charts"]],
        "storage_classes": {
            "runtime": ["fixed_geometry.G", "fixed_geometry.U", "fixed_geometry.Q", "charts", "router_centers", "h_feature_scale", "m_feature_scale"],
            "privileged_discovery": ["fixed_geometry.D_G", "fixed_geometry.F_factor", "sqrt_H", "blocks", "discovery_centers"],
            "training_only": ["training_assignments", "anchor_indices", *_TRAINING_ARRAY_KEYS],
            "scope": "numeric_payload_classes_not_a_speedup_or_compression_claim; shared_geometry_is_counted_once_per_arm",
        },
    }
    payload = _json_value(payload)
    payload["artifact_sha256"] = hashlib.sha256(b"intercept-compatibility/arm/v1\0" + _canonical(payload)).hexdigest()
    return payload


def load_arm(replay: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate a frozen serialized arm; do not refit or normalize anew."""
    payload = copy.deepcopy(dict(artifact))
    digest = payload.pop("artifact_sha256", None)
    if digest != hashlib.sha256(b"intercept-compatibility/arm/v1\0" + _canonical(payload)).hexdigest():
        raise ValueError("frozen arm artifact digest mismatch")
    if payload["schema"] != ARTIFACT_SCHEMA or type(payload["compatibility_weight"]) is not int or payload["compatibility_weight"] not in (0, 1):
        raise ValueError("unknown frozen arm schema or compatibility weight")
    if _canonical(payload["config"]) != _canonical(replay["config"]) or payload["fixed_training_sha256"] != _training_identity(replay):
        raise ValueError("frozen arm does not belong to the authenticated training fixture")
    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    geometry = {name: tensor(payload["fixed_geometry"][name]) for name in _GEOMETRY_KEYS}
    for name in _GEOMETRY_KEYS:
        if not torch.equal(geometry[name], replay["tensors"][name]):
            raise ValueError("frozen arm changes a fixed metric, coordinate map or basis")
    arm = {name: copy.deepcopy(payload[name]) for name in (
        "compatibility_weight", "config", "fixed_training_sha256", "h_feature_scale",
        "m_feature_scale", "anchor_indices", "construction_checks",
    )}
    arm["fixed_geometry"] = geometry
    for name in ("sqrt_H", "discovery_centers", "router_centers", *_TRAINING_ARRAY_KEYS):
        arm[name] = tensor(payload[name])
        _tensor(arm[name], name)
    arm["training_assignments"] = torch.tensor(payload["training_assignments"], dtype=torch.long)
    arm["blocks"] = copy.deepcopy(payload["blocks"])
    for transform in arm["blocks"].values():
        transform["mean"] = tensor(transform["mean"])
        # Validate saved metadata without deriving a new mean, scale or decision.
        apply_block_transform(transform["mean"].unsqueeze(0), transform)
    G, U, Q = (geometry[name] for name in ("G", "U", "Q"))
    arm["charts"] = [replay["kernel"].AffineResponseChart(
        tensor(row["state_anchor"]), tensor(row["message_anchor"]), tensor(row["beta"]),
        tensor(row["B"]), tensor(row["A"]), U, Q, G, row["state_radius"], row["message_radius"],
    ) for row in payload["charts"]]
    if len(arm["charts"]) != replay["config"]["chart_count"]:
        raise ValueError("frozen arm changes chart capacity")
    return arm


def verify_lambda_zero(
    replay: Mapping[str, Any], arm: Mapping[str, Any], diagnosis_module: Any,
) -> dict[str, Any]:
    """Verify the rebuilt control under ONE globally consistent label mapping.

    Only the lambda=0 control may enter this evaluator.  The sole lambda=1
    candidate is scored separately after its artifact has been frozen.
    """
    if arm["compatibility_weight"] != 0:
        raise ValueError("baseline-equivalence evaluation requires the lambda=0 control")
    count = replay["config"]["chart_count"]
    old_assignments = replay["train_direct_assignments"]["response_charts"]
    rebuilt_assignments = arm["training_assignments"]
    matching = [list(permutation) for permutation in itertools.permutations(range(count))
                if torch.equal(torch.tensor(permutation, dtype=torch.long)[rebuilt_assignments], old_assignments)]
    if len(matching) != 1:
        return {"status": "discrepancy", "reason": "no_unique_global_training_membership_label_permutation", "matching_permutation_count": len(matching)}
    permutation = matching[0]
    inverse = [permutation.index(old) for old in range(count)]
    label_map = torch.tensor(permutation, dtype=torch.long)
    rebuilt_charts = [arm["charts"][i] for i in inverse]
    saved_charts = replay["banks"]["response_charts"]
    checks: dict[str, bool] = {}
    comparisons: list[dict[str, Any]] = []

    def compare(name: str, rebuilt: Any, saved: Any) -> None:
        left = rebuilt if isinstance(rebuilt, Tensor) else torch.as_tensor(rebuilt, dtype=torch.float64)
        right = saved if isinstance(saved, Tensor) else torch.as_tensor(saved, dtype=torch.float64)
        if left.shape != right.shape:
            checks[name] = False
            comparisons.append({"measurement": name, "passed": False, "reason": "shape_mismatch"})
            return
        if left.dtype in (torch.bool, torch.long) and right.dtype in (torch.bool, torch.long):
            passed = bool(torch.equal(left, right))
            error = float((left.to(torch.float64) - right.to(torch.float64)).abs().max()) if left.numel() else 0.0
        else:
            left, right = left.to(torch.float64), right.to(torch.float64)
            passed = bool(torch.allclose(left, right, atol=REPRODUCTION_ATOL, rtol=REPRODUCTION_RTOL))
            error = float((left - right).abs().max()) if left.numel() else 0.0
        checks[name] = passed
        comparisons.append({"measurement": name, "passed": passed, "shape": list(left.shape), "maximum_absolute_difference": error})

    compare("training_memberships", label_map[rebuilt_assignments], old_assignments)
    compare("discovery_centers", arm["discovery_centers"][inverse], replay["tensors"]["response_signature_centers"])
    compare("medoid_row_identities", [arm["anchor_indices"][i] for i in inverse], replay["saved_artifact"]["banks"]["response_charts"]["training_anchor_row_indices"])
    compare("router_centers", arm["router_centers"][inverse], replay["router_centers"]["response_charts"])
    compare("training_response_operators", arm["training_A"], replay["train_signatures"])
    for index, (rebuilt, saved) in enumerate(zip(rebuilt_charts, saved_charts)):
        for name in ("state_anchor", "message_anchor", "beta", "B", "A", "state_radius", "message_radius"):
            compare(f"chart_{index}.{name}", getattr(rebuilt, name), getattr(saved, name))
    h, m, directions = replay["eval_h"], replay["eval_m"], replay["probe_directions"]
    runtime = label_map[assign_runtime(arm, h, m)]
    signatures = discovery_features(arm, m, replay["evaluation_outputs"], replay["evaluation_signatures"])
    direct = label_map[torch.cdist(signatures, arm["discovery_centers"]).square().argmin(dim=1)]
    compare("runtime_assignments", runtime, replay["runtime_assignments"]["response_charts"])
    compare("direct_discovery_assignments", direct, replay["direct_assignments"]["response_charts"])
    factor, kernel = replay["tensors"]["F_factor"], replay["kernel"]
    fn = lambda messages: kernel.normalization_update(messages, replay["tensors"]["W"], replay["tensors"]["norm_weight"], replay["config"]["epsilon"])
    targets = [
        ("residual_update", None, None, replay["evaluation_outputs"].unsqueeze(1)),
        ("derivative_action", None, directions.unsqueeze(1), replay["evaluation_probe_actions"].unsqueeze(1)),
    ]
    targets.extend(("finite_response", radius, (radius * directions).unsqueeze(1), (fn(m + radius * directions) - replay["evaluation_outputs"]).unsqueeze(1)) for radius in replay["config"]["finite_radii"])
    metric_rows = []
    for target_name, radius, probes, truth in targets:
        label = f"{target_name}:{radius}"
        rebuilt_predictions = diagnosis_module.frozen_predictions(rebuilt_charts, h, m, probes, target_name)
        saved_predictions = diagnosis_module.frozen_predictions(saved_charts, h, m, probes, target_name)
        perturbations = probes if target_name == "finite_response" else None
        rebuilt_eligible = diagnosis_module.endpoint_eligibility(rebuilt_charts, h, m, perturbations)
        saved_eligible = diagnosis_module.endpoint_eligibility(saved_charts, h, m, perturbations)
        compare(f"{label}.per_example_all_chart_predictions", rebuilt_predictions, saved_predictions)
        compare(f"{label}.all_chart_eligibility", rebuilt_eligible, saved_eligible)
        rebuilt_oracle = diagnosis_module.supported_oracle(rebuilt_predictions, truth, factor, rebuilt_eligible)
        saved_oracle = diagnosis_module.supported_oracle(saved_predictions, truth, factor, saved_eligible)
        compare(f"{label}.supported_oracle_assignments", rebuilt_oracle, saved_oracle)
        choices = {
            "runtime_router": (runtime, replay["runtime_assignments"]["response_charts"]),
            "direct_discovery": (direct, replay["direct_assignments"]["response_charts"]),
            "best_supported_oracle": (rebuilt_oracle, saved_oracle),
        }
        common_available = rebuilt_eligible.any(dim=1) & saved_eligible.any(dim=1)
        common_selected = common_available.clone()
        for left, right in choices.values():
            common_selected &= rebuilt_eligible[torch.arange(len(h)), left.clamp_min(0)] & (left >= 0)
            common_selected &= saved_eligible[torch.arange(len(h)), right.clamp_min(0)] & (right >= 0)
        for mask_name, mask in (("all_rows", torch.ones(len(h), dtype=torch.bool)), ("common_available", common_available), ("common_selected_support", common_selected)):
            for method, (left, right) in choices.items():
                if bool((((left < 0) | (right < 0)) & mask).any()):
                    metric_rows.append({"target": target_name, "radius": radius, "mask": mask_name, "method": method, "status": "unavailable_abstention"})
                    continue
                new_score = diagnosis_module.score_assignments(rebuilt_predictions, truth, factor, left, mask)
                old_score = diagnosis_module.score_assignments(saved_predictions, truth, factor, right, mask)
                for name in ("fisher_error_numerator", "euclidean_error_numerator", "fisher_target_denominator", "euclidean_target_denominator", "fisher_relative_l2", "euclidean_relative_l2", "euclidean_rmse"):
                    if new_score[name] is None or old_score[name] is None:
                        checks[f"{label}.{mask_name}.{method}.{name}"] = new_score[name] is None and old_score[name] is None
                    else:
                        compare(f"{label}.{mask_name}.{method}.{name}", new_score[name], old_score[name])
                metric_rows.append({"target": target_name, "radius": radius, "mask": mask_name, "method": method, "rebuilt": new_score, "saved_bank": old_score})
    return {
        "status": "reproduced" if all(checks.values()) else "discrepancy",
        "chart_label_permutation_control_to_saved": permutation,
        "permutation_policy": "one_mapping_selected_from_training_membership_and_applied_to_every_object",
        "checks": checks, "numerical_comparisons": comparisons, "aggregate_metric_comparisons": metric_rows,
        "absolute_tolerance": REPRODUCTION_ATOL, "relative_tolerance": REPRODUCTION_RTOL,
        "lambda_one_evaluated": False,
    }
