"""Authenticated replay of one already examined synthetic normalization fixture.

The original report, fitted numeric artifacts, configuration, and source bytes
are immutable inputs.  Only absent generated inputs and probe directions are
replayed.  This module never invokes a fitter, the original qualification
runner, a model loader, or a corpus reader.  Importing it uses stdlib only;
the authenticated numerical kernel imports torch after all integrity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ORIGINAL_DIRECTORY = "artifacts/research/fisher-response-geometry-probe-v1-synthetic"
ORIGINAL_REPORT_SHA256 = "b600bc66c260ca6722fc3b7b66340bda9093c268dda4ebbde058e36dda23d5a6"
ORIGINAL_CONFIG_SHA256 = "b06bbf156555f178b823a7cff2c89642521dda37d774dd59fdce796b16ade3d3"
ORIGINAL_MANIFEST_SHA256 = "ac1697472a775b3b7f0185db3916ece918a31411e7ad4f43d8aa4e55ebbb805c"
ORIGINAL_CONFIG_DOMAIN_SHA256 = "d223c18c27808d011ec00e1f7dd58b8275b9a6694192f4fde55c47eea93e36c0"
ORIGINAL_SOURCE_PATHS = (
    "src/fisher_response_geometry_probe_v1.py",
    "src/fisher_graph/fisher_response_geometry_protocol.py",
    "src/fisher_graph/fisher_response_geometry.py",
    "src/fisher_graph/fisher_local_chart.py",
    "examples/fisher_response_geometry_probe_v1.json",
    "docs/fisher-response-geometry-probe-v1.md",
    "docs/schemas/fisher_response_geometry_probe_v1_report.schema.json",
    "src/fisher_graph/gemma3_attention_generator.py",
)
REPRODUCTION_ABSOLUTE_TOLERANCE = 1e-12
REPRODUCTION_RELATIVE_TOLERANCE = 1e-10


class ReplayIntegrityError(ValueError):
    """A pinned source or artifact differs; no numerical replay is allowed."""


class ReplayReproductionError(ValueError):
    """Preserve discrepancies for a diagnostic report, without silently refitting."""

    def __init__(self, reproduction: dict[str, Any]) -> None:
        super().__init__("the pinned synthetic baseline did not reproduce")
        self.reproduction = reproduction


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_fixed(root: Path, relative: str) -> bytes:
    path = root / relative
    if path.is_symlink() or path.resolve(strict=False) != path or not path.is_file():
        raise ReplayIntegrityError(f"fixed replay input is missing or a symlink: {relative}")
    return path.read_bytes()


def _authenticate(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    """Only the three exact original artifacts and eight explicit sources are read."""

    artifact_pins = {
        "report.json": ORIGINAL_REPORT_SHA256,
        "frozen_config.json": ORIGINAL_CONFIG_SHA256,
        "artifact_manifest.json": ORIGINAL_MANIFEST_SHA256,
    }
    payloads, identities = {}, []
    for name, expected in artifact_pins.items():
        relative = f"{ORIGINAL_DIRECTORY}/{name}"
        payload = _read_fixed(root, relative)
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ReplayIntegrityError(f"original artifact SHA-256 mismatch: {name}")
        payloads[name] = payload
        identities.append({"path": relative, "sha256": actual, "bytes": len(payload)})
    report = json.loads(payloads["report.json"])
    config = json.loads(payloads["frozen_config.json"])
    manifest = json.loads(payloads["artifact_manifest.json"])
    if _canonical(report["config"]) != _canonical(config):
        raise ReplayIntegrityError("saved report and frozen config differ")
    digest = hashlib.sha256(b"fisher-response-geometry/config/v1\0" + _canonical(config)).hexdigest()
    if any(value != ORIGINAL_CONFIG_DOMAIN_SHA256 for value in (
        digest, report["config_sha256"], manifest["config_sha256"],
    )):
        raise ReplayIntegrityError("original domain-separated config digest mismatch")
    expected_inventory = [
        {"path": name, "sha256": artifact_pins[name], "bytes": len(payloads[name])}
        for name in ("frozen_config.json", "report.json")
    ]
    if manifest["artifacts"] != expected_inventory:
        raise ReplayIntegrityError("original artifact inventory mismatch")
    if report["source_manifest"] != manifest["source_manifest"]:
        raise ReplayIntegrityError("original report and manifest source identities differ")
    sources = report["source_manifest"]
    if [entry["path"] for entry in sources] != list(ORIGINAL_SOURCE_PATHS):
        raise ReplayIntegrityError("original source allowlist differs")
    source_bytes = {}
    for entry in sources:
        relative = entry["path"]
        payload = _read_fixed(root, relative)
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ReplayIntegrityError(f"original source SHA-256 mismatch: {relative}")
        source_bytes[relative] = payload
    if _canonical(json.loads(source_bytes["examples/fisher_response_geometry_probe_v1.json"])) != _canonical(config):
        raise ReplayIntegrityError("checked-in original config differs from saved config")
    identity = {
        "artifacts": identities,
        "source_manifest": sources,
        "config_sha256": digest,
        "original_report_sha256": ORIGINAL_REPORT_SHA256,
        "original_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "evidence_class": "additional_diagnostics_on_previously_examined_synthetic_fixture",
        "p0_status": "development_complete_independent_data_unavailable",
    }
    return report, config, identity, source_bytes["src/fisher_graph/fisher_response_geometry.py"]


def _load_kernel(root: Path, authenticated_source: bytes) -> ModuleType:
    relative = "src/fisher_graph/fisher_response_geometry.py"
    name = "_fisher_response_geometry_authenticated_replay_kernel"
    path = root / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ReplayIntegrityError("cannot identify authenticated synthetic kernel")
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[name] = kernel
    try:
        # Compile the already authenticated bytes, bypassing bytecode caches.
        exec(compile(authenticated_source, str(path), "exec"), kernel.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return kernel


@dataclass(frozen=True)
class SavedProjection:
    """Use the saved coordinate map and metric factor without fitting or solving."""

    basis: Any
    factor: Any
    coordinate_map: Any

    @property
    def gram(self) -> Any:
        weighted = self.factor.T @ self.basis
        return weighted.T @ weighted

    def encode(self, values: Any) -> Any:
        return values @ self.coordinate_map.T

    def decode(self, values: Any) -> Any:
        return values @ self.basis.T

    def project(self, values: Any) -> Any:
        return self.decode(self.encode(values))

    def fisher_squared(self, values: Any) -> Any:
        return (values @ self.factor).square().sum(dim=-1)


def _replay_inputs(torch: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Preserve the original private generator's draw order; never fit anything."""

    d, p, chart_count = (config[name] for name in ("residual_width", "message_width", "chart_count"))
    generator = torch.Generator().manual_seed(config["seed"])
    randn = lambda *shape: torch.randn(shape, generator=generator, dtype=torch.float64)
    W, weight = randn(d, p) / math.sqrt(p), randn(d) * 0.15
    message_map, centers = randn(p, d) / math.sqrt(d), randn(chart_count, d) * 1.5

    def draw(families: int, offset: int) -> tuple[Any, Any, list[str]]:
        states, messages, identifiers = [], [], []
        for family in range(families):
            center = centers[family % chart_count] + 0.25 * randn(d)
            for _ in range(config["rows_per_family"]):
                h = center + 0.3 * randn(d)
                m = message_map @ h + 0.2 * randn(p)
                states.append(h)
                messages.append(m)
                identifiers.append(f"synthetic-family-{offset + family}")
        return torch.stack(states), torch.stack(messages), identifiers

    train_h, train_m, train_ids = draw(config["train_families"], 0)
    # The original fitter consumed these random score transform/noise draws.
    # Replay discards them; the saved metric, bases, and maps are loaded instead.
    randn(d, d)
    randn(len(train_h), d)
    training_permutation = torch.randperm(len(train_h), generator=generator)
    eval_h, eval_m, eval_ids = draw(config["evaluation_families"], config["train_families"])
    shuffled_evaluation_indices = torch.randint(len(train_h), (len(eval_h),), generator=generator)
    directions = randn(*eval_m.shape)
    directions /= directions.norm(dim=1, keepdim=True)
    state_directions = randn(*eval_h.shape)
    state_directions /= state_directions.norm(dim=1, keepdim=True)
    return {
        "train_h": train_h, "train_m": train_m, "train_ids": train_ids,
        "eval_h": eval_h, "eval_m": eval_m, "eval_ids": eval_ids,
        "probe_directions": directions, "state_probe_directions": state_directions,
        "training_permutation": training_permutation,
        "shuffled_evaluation_indices": shuffled_evaluation_indices,
        "regenerated_W": W, "regenerated_norm_weight": weight,
    }


def _check_reproduction(replay: dict[str, Any]) -> dict[str, Any]:
    kernel, tensors, config = replay["kernel"], replay["tensors"], replay["config"]
    torch, projection = kernel.torch, replay["projection"]
    train_h, train_m, eval_h, eval_m = (replay[key] for key in ("train_h", "train_m", "eval_h", "eval_m"))
    W, weight, Q = tensors["W"], tensors["norm_weight"], tensors["Q"]
    eps, artifact = config["epsilon"], replay["saved_artifact"]
    report, checks, comparisons = replay["original_report"], {}, []
    fn = lambda m: kernel.normalization_update(m, W, weight, eps)
    actual, train_actual = fn(eval_m), fn(train_m)
    signature = lambda m: projection.coordinate_map @ kernel.jvp_actions(fn, m, Q)
    train_A = torch.stack([signature(m) for m in train_m])
    eval_A = torch.stack([signature(m) for m in eval_m])
    replay["train_signatures"], replay["evaluation_signatures"] = train_A, eval_A
    replay["train_outputs"], replay["evaluation_outputs"] = train_actual, actual

    def compare(name: str, observed: float, saved: float) -> None:
        observed, saved = float(observed), float(saved)
        difference = abs(observed - saved)
        limit = REPRODUCTION_ABSOLUTE_TOLERANCE + REPRODUCTION_RELATIVE_TOLERANCE * abs(saved)
        comparisons.append({
            "measurement": name, "saved": saved, "replayed": observed,
            "absolute_difference": difference, "allowed_difference": limit,
            "within_tolerance": math.isfinite(observed) and difference <= limit,
        })

    def compare_metrics(name: str, observed: dict[str, float], saved: dict[str, float]) -> None:
        if set(observed) != set(saved):
            raise ReplayIntegrityError(f"saved metric fields differ: {name}")
        for key in sorted(saved):
            compare(f"{name}.{key}", observed[key], saved[key])

    checks["regenerated_W_equals_saved_W"] = bool(torch.equal(replay.pop("regenerated_W"), W))
    checks["regenerated_weight_equals_saved_weight"] = bool(torch.equal(replay.pop("regenerated_norm_weight"), weight))
    checks["training_evaluation_families_disjoint"] = not set(replay["train_ids"]).intersection(replay["eval_ids"])
    compare("training_rows", len(train_h), report["sections"]["runtime_and_storage_accounting"]["training_rows"])
    compare("evaluation_rows", len(eval_h), report["sections"]["runtime_and_storage_accounting"]["evaluation_rows"])
    compare("h_feature_scale", float(train_h.square().mean().sqrt()), artifact["h_feature_scale"])
    compare("m_feature_scale", float(train_m.square().mean().sqrt()), artifact["m_feature_scale"])

    feature = lambda h, m: torch.cat((h / artifact["h_feature_scale"], m / artifact["m_feature_scale"]), dim=-1)
    train_features, eval_features = feature(train_h, train_m), feature(eval_h, eval_m)
    replay["train_features"], replay["evaluation_features"] = train_features, eval_features
    centers = tensors["response_signature_centers"]
    runtime_centers = replay["router_centers"]
    train_assignment = {
        "global_taylor_affine": torch.zeros(len(train_h), dtype=torch.long),
        "activation_charts": torch.cdist(train_features, runtime_centers["activation_charts"]).square().argmin(dim=1),
        "response_charts": torch.cdist(train_A.flatten(start_dim=1), centers).square().argmin(dim=1),
    }
    train_assignment["shuffled_charts"] = train_assignment["response_charts"][replay.pop("training_permutation")]
    direct_assignment = {
        "global_taylor_affine": torch.zeros(len(eval_h), dtype=torch.long),
        "activation_charts": torch.cdist(eval_features, runtime_centers["activation_charts"]).square().argmin(dim=1),
        "response_charts": torch.cdist(eval_A.flatten(start_dim=1), centers).square().argmin(dim=1),
        "shuffled_charts": train_assignment["shuffled_charts"][replay.pop("shuffled_evaluation_indices")],
    }
    runtime_assignment = {
        name: torch.cdist(eval_features, values).argmin(dim=1)
        for name, values in runtime_centers.items()
    }
    replay["train_direct_assignments"] = train_assignment
    replay["direct_assignments"], replay["runtime_assignments"] = direct_assignment, runtime_assignment
    baseline_rows, prediction_cache = {}, {}
    for name, charts in replay["banks"].items():
        saved_bank = artifact["banks"][name]
        for index, (chart, row) in enumerate(zip(charts, saved_bank["training_anchor_row_indices"])):
            checks[f"{name}.chart_{index}.state_anchor_row_parity"] = bool(torch.equal(chart.state_anchor, train_h[row]))
            checks[f"{name}.chart_{index}.message_anchor_row_parity"] = bool(torch.equal(chart.message_anchor, train_m[row]))
            compare(f"{name}.chart_{index}.stored_beta_error", float((chart.beta - projection.encode(train_actual[row])).abs().max()), 0.0)
            compare(f"{name}.chart_{index}.stored_A_error", float((chart.A - train_A[row]).abs().max()), 0.0)
        routed, direct = runtime_assignment[name], direct_assignment[name]
        predict = lambda assignment: torch.stack([charts[int(c)].predict(h, m) for c, h, m in zip(assignment, eval_h, eval_m)])
        predicted, direct_predicted = predict(routed), predict(direct)
        prediction_cache[name] = predicted
        support = torch.tensor([charts[int(c)].in_support(h, m) for c, h, m in zip(routed, eval_h, eval_m)])
        observed, direct_observed = kernel._errors(predicted, actual, projection), kernel._errors(direct_predicted, actual, projection)
        saved_finite = report["sections"]["finite_update_fidelity"]["methods"][name]
        saved_router = report["sections"]["router_feasibility"]["methods"][name]
        compare_metrics(f"{name}.runtime_all_rows", observed, saved_finite["unchecked_all_rows"])
        compare_metrics(f"{name}.direct_all_rows", direct_observed, saved_router["oracle_assignment_error"])
        compare(f"{name}.runtime_supported_row_count", int(support.sum()), saved_finite["supported_row_count"])
        if bool(support.any()):
            compare_metrics(f"{name}.runtime_supported_rows", kernel._errors(predicted[support], actual[support], projection), saved_finite["supported_rows"])
        compare(f"{name}.runtime_direct_assignment_agreement", float((routed == direct).double().mean()), saved_router["assignment_agreement"])
        compare(f"{name}.runtime_direct_mean_excess_fisher_error", float((projection.fisher_squared(predicted - actual) - projection.fisher_squared(direct_predicted - actual)).mean()), saved_router["mean_excess_fisher_squared_error_vs_oracle_assignment"])
        predicted_A = torch.stack([charts[int(c)].A for c in routed])
        compare(f"{name}.projected_derivative_relative_l2", float((predicted_A - eval_A).norm() / eval_A.norm().clamp_min(1e-15)), saved_router["projected_derivative_relative_l2"])
        saved_stability = report["sections"]["chart_stability"]["methods"][name]
        checks[f"{name}.training_occupancy_matches_saved"] = torch.bincount(train_assignment[name], minlength=len(charts)).tolist() == saved_stability["train_occupancy"]
        checks[f"{name}.runtime_evaluation_occupancy_matches_saved"] = torch.bincount(routed, minlength=len(charts)).tolist() == saved_stability["evaluation_routed_occupancy"]
        checks[f"{name}.direct_evaluation_occupancy_matches_saved"] = torch.bincount(direct, minlength=len(charts)).tolist() == saved_stability["evaluation_oracle_occupancy"]
        baseline_rows[name] = {"runtime_router": observed, "direct_discovery_assignment": direct_observed, "all_row_count": len(eval_h), "runtime_supported_row_count": int(support.sum())}

    projected = projection.project(actual)
    compare_metrics("exact_analytic_projected_normalization", kernel._errors(projected, actual, projection), report["sections"]["finite_update_fidelity"]["methods"]["exact_analytic_projected_normalization"]["unchecked_all_rows"])
    eval_design = torch.cat((torch.ones(len(eval_h), 1, dtype=torch.float64), eval_h @ tensors["U"], eval_m @ Q), dim=1)
    compare_metrics("global_least_squares_affine.saved_coefficients", kernel._errors(projection.decode(eval_design @ tensors["global_least_squares_coefficients"]), actual, projection), report["sections"]["finite_update_fidelity"]["methods"]["global_least_squares_affine"]["unchecked_all_rows"])
    directions = replay["probe_directions"]
    full_actions = torch.stack([kernel.normalization_jvp(m, q, W, weight, eps) for m, q in zip(eval_m, directions)])
    replay["evaluation_probe_actions"] = full_actions
    compare_metrics("independent_response_outside_G", kernel._errors(projection.project(full_actions), full_actions, projection), report["sections"]["response_subspace_adequacy"]["independent_response_outside_G"])
    for saved_radius in report["sections"]["finite_update_fidelity"]["finite_perturbation_sweep"]:
        radius = saved_radius["radius"]
        actual_response = fn(eval_m + radius * directions) - actual
        for name, charts in replay["banks"].items():
            predicted = torch.stack([charts[int(c)].G @ charts[int(c)].A @ (Q.T @ (radius * q)) for c, q in zip(runtime_assignment[name], directions)])
            compare_metrics(f"radius_{radius}.{name}", kernel._errors(predicted, actual_response, projection), saved_radius["chart_derivative_predictions_fixed_base_assignment"][name])
        compact = torch.stack([projection.basis @ A @ (Q.T @ (radius * q)) for A, q in zip(eval_A, directions)])
        compare_metrics(f"radius_{radius}.exact_per_state_derivative", kernel._errors(radius * full_actions, actual_response, projection), saved_radius["exact_per_state_derivative_oracle"])
        compare_metrics(f"radius_{radius}.projected_per_state_derivative", kernel._errors(compact, actual_response, projection), saved_radius["projected_per_state_derivative_oracle"])
    passed = all(checks.values()) and all(row["within_tolerance"] for row in comparisons)
    return {
        "status": "reproduced" if passed else "discrepancy",
        "exact_checks": checks, "metric_comparisons": comparisons,
        "maximum_absolute_metric_difference": max(row["absolute_difference"] for row in comparisons),
        "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "relative_tolerance": REPRODUCTION_RELATIVE_TOLERANCE,
        "baseline_absolute_residual_update": baseline_rows,
        "metric_definition": "sqrt(sum_all_rows((prediction-truth)^T F_star (prediction-truth))/sum_all_rows(truth^T F_star truth)); all_48_rows_including_unsupported_extrapolations",
        "fitted_artifact_policy": "loaded_unchanged_no_refitting",
        "regenerated_items": "missing_generated_h_m_family_ids_and_original_independent_probe_directions_only",
        "rng_schedule_verification": "reference_W_and_norm_weight_regenerated_for_exact_parity_then_discarded; saved_values_used_for_predictions; original_generator_map_centers_and_score_noise_draw_schedule_replayed_without_fitting",
        "membership_policy": "nearest_saved_discovery_centers_no_clustering_or_router_fit",
        "fisher_squared_normalization_denominator_all_rows": float(projection.fisher_squared(actual).sum()),
        "euclidean_squared_normalization_denominator_all_rows": float(actual.square().sum()),
        "training_row_count": len(train_h), "evaluation_row_count": len(eval_h),
        "training_family_count": len(set(replay["train_ids"])),
        "evaluation_family_count": len(set(replay["eval_ids"])),
        "independent_message_probe_count": len(directions),
    }


def load_replay(root: Path | None = None) -> dict[str, Any]:
    """Authenticate and reproduce the fixed saved fixture, or fail with evidence.

    ``root`` relocates the same pinned repository files for isolated integrity
    tests.  It does not select a dataset or permit an alternative baseline.
    Reproduction discrepancies raise ``ReplayReproductionError`` containing a
    ``reproduction`` dictionary suitable for a separate diagnostic artifact.
    """

    root = (root or Path(__file__).resolve().parents[2]).resolve()
    report, config, identity, source = _authenticate(root)
    kernel = _load_kernel(root, source)
    torch = kernel.torch
    saved = report["sections"]["runtime_and_storage_accounting"]["artifact"]
    tensor = lambda value: torch.tensor(value, dtype=torch.float64)
    mapping = {
        "G": "G", "U": "U", "Q": "Q", "F_factor": "F_factor", "D_G": "D_G",
        "W": "synthetic_reference_W", "norm_weight": "synthetic_reference_norm_weight",
        "response_signature_centers": "offline_response_signature_centers",
        "global_least_squares_coefficients": "global_least_squares_coefficients",
    }
    tensors = {name: tensor(saved[key]) for name, key in mapping.items()}
    projection = SavedProjection(tensors["G"], tensors["F_factor"], tensors["D_G"])
    banks = {}
    for name, bank in saved["banks"].items():
        banks[name] = [kernel.AffineResponseChart(
            state_anchor=tensor(row["state_anchor"]), message_anchor=tensor(row["message_anchor"]),
            beta=tensor(row["beta"]), B=tensor(row["B"]), A=tensor(row["A"]),
            U=tensors["U"], Q=tensors["Q"], G=tensors["G"],
            state_radius=row["state_radius"], message_radius=row["message_radius"],
        ) for row in bank["charts"]]
    replay = {
        "config": config, "original_report": report, "original_identity": identity,
        "saved_artifact": saved, "tensors": tensors, "kernel": kernel,
        "projection": projection, "banks": banks,
        "router_centers": {name: tensor(bank["router_centers"]) for name, bank in saved["banks"].items()},
        **_replay_inputs(torch, config),
    }
    reproduction = _check_reproduction(replay)
    replay["reproduction"] = reproduction
    if reproduction["status"] != "reproduced":
        raise ReplayReproductionError(reproduction)
    return replay
