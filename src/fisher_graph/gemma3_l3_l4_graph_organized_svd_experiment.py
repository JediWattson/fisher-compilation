"""Run a strict graph-organized global-SVD routing experiment on Gemma.

The numerical executor is the rank-45, full-target global SVD fitted only at
origins 8/24/40.  A fit-only signed graph basis assigns its retained source
components to four graph-frequency packs.  Size-matched contiguous and random
packings are evaluated as controls.

The routing inputs are prompt-blind synthetic C2 development directions.  The
upstream Fisher basis remains prompt-derived.  This module does not measure
NLL, task accuracy, whole-model compression, wall-clock latency, or a complete
Gemma block replacement.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import torch
from torch import Tensor

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    fit_conditional_spectral_generator,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    Gemma3SpectralSource,
    _file_sha256,
    _reserve_outputs,
    _response_binding,
    _stage_json,
    _stage_torch,
    _validate_output_path,
    load_gemma3_spectral_source,
)
from . import gemma3_l3_l4_contrast_provider_development as c2_development
from .gemma3_l3_l4_contrast_provider_development_materialization import (
    MaterializedDevelopmentBatch,
    materialize_development_role,
)
from .gemma3_l3_l4_contrast_provider_development_protocol import (
    ContrastProviderDevelopmentProtocol,
    DevelopmentCalibrationBinding,
    FrozenDevelopmentCandidateSet,
    default_contrast_provider_development_protocol,
)
from .graph_organized_svd import (
    GraphOrganizedSVDPlan,
    organize_conditional_svd_with_graph,
)
from .graph_spectral_source_basis import (
    FitOnlyGraphSourceBasis,
    fit_graph_source_bases,
)


__all__ = [
    "DEFAULT_C2_ARTIFACT_SHA256",
    "DEFAULT_C2_LOGICAL_ARTIFACT_SHA256",
    "DEFAULT_C2_REPORT_FILE_SHA256",
    "DEFAULT_C2_REPORT_PAYLOAD_SHA256",
    "DEFAULT_FREQUENCY_BAND_BOUNDARIES",
    "DEFAULT_OUTPUT",
    "DEFAULT_RANDOM_SEEDS",
    "DEFAULT_ROUTE_FRACTIONS",
    "Gemma3GraphOrganizedSVDCandidate",
    "build_graph_organized_plan_set",
    "build_parser",
    "compile_gemma3_graph_organized_svd_candidate",
    "compile_gemma3_l3_l4_graph_organized_svd",
    "evaluate_graph_organized_plan",
    "load_gemma3_graph_organized_svd_candidate",
    "main",
]


DEFAULT_C2_ARTIFACT = c2_development.DEFAULT_OUTPUT
DEFAULT_C2_ARTIFACT_SHA256 = (
    "89ce49a4f5b731d7f9b07e5c143663c66a043244f58cfca9d2505b6263b7c193"
)
DEFAULT_C2_REPORT_FILE_SHA256 = (
    "894ec43907ed0ce0bfa5381936facd47038a6029f3c93a00013f0f94a12705a3"
)
DEFAULT_C2_REPORT_PAYLOAD_SHA256 = (
    "4c99907eff6b72e10f123cae532e1ac44515b55a5c3f070dd8dd4715b8d6992e"
)
DEFAULT_C2_LOGICAL_ARTIFACT_SHA256 = (
    "77c3d569304b26352ff7975a045105d4ee72156600c9489a67fdfa137bfb697f"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-organized-svd-dev-v1.pt"
)
DEFAULT_FREQUENCY_BAND_BOUNDARIES = (0, 8, 16, 32, 64)
DEFAULT_RANDOM_SEEDS = (
    1729,
    3253,
    4513,
    6421,
    8191,
    10007,
    12289,
    16381,
)
DEFAULT_ROUTE_FRACTIONS = (0.90, 0.95, 0.975, 0.99, 1.0)
FIT_EVALUATION_ORIGIN = 24
SELECTION_EVALUATION_ORIGIN = 32
SOURCE_RANK = 45
TARGET_RANK = 64

_SCHEMA = "fisher_graph.gemma3_l3_l4_graph_organized_svd_development"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher-graph:gemma3-graph-organized-svd:v1\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-graph-organized-svd-report:v1\0"
_ROW_MASK_DOMAIN = b"fisher-graph:graph-organized-svd-route-mask:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_C2_REQUIRED_STATE_KEYS = {
    "manifest",
    "artifact_sha256",
    "metric_weight",
    "protocol_state",
    "calibration_state",
    "frozen_candidate_set_state",
    "objective_state",
    "training_spec",
    "controls_state",
    "plan_states",
    "candidate_results",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_ROW_MASK_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(int(width) for width in tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _relative_error_from_squares(
    residual_square: Tensor,
    truth_square: Tensor,
    multiplicities: Tensor,
) -> float:
    numerator = float((residual_square * multiplicities).sum())
    denominator = max(
        float((truth_square * multiplicities).sum()),
        torch.finfo(torch.float64).eps,
    )
    return math.sqrt(max(numerator, 0.0) / denominator)


def _cosine_from_terms(
    dot: Tensor,
    left_square: Tensor,
    right_square: Tensor,
    multiplicities: Tensor,
) -> float:
    numerator = float((dot * multiplicities).sum())
    left = float((left_square * multiplicities).sum())
    right = float((right_square * multiplicities).sum())
    denominator = math.sqrt(max(left * right, 0.0))
    if denominator <= torch.finfo(torch.float64).eps:
        return 1.0 if abs(numerator) <= torch.finfo(torch.float64).eps else 0.0
    return max(-1.0, min(1.0, numerator / denominator))


@dataclass(frozen=True, slots=True)
class _C2Context:
    tensor_file_sha256: str
    report_file_sha256: str
    report_payload_sha256: str
    logical_artifact_sha256: str
    protocol: ContrastProviderDevelopmentProtocol
    calibration: DevelopmentCalibrationBinding
    frozen_candidates: FrozenDevelopmentCandidateSet


def _restore_calibration(
    raw: Mapping[str, object],
) -> DevelopmentCalibrationBinding:
    return DevelopmentCalibrationBinding(
        protocol_sha256=raw["protocol_sha256"],  # type: ignore[arg-type]
        pilot_panel_sha256=raw["pilot_panel_sha256"],  # type: ignore[arg-type]
        calibration_rule_sha256=raw[
            "calibration_rule_sha256"
        ],  # type: ignore[arg-type]
        selected_amplitude=raw["selected_amplitude"],  # type: ignore[arg-type]
        pilot_metric_sha256s=tuple(
            raw["pilot_metric_sha256s"]  # type: ignore[arg-type]
        ),
        artifact_sha256=raw["artifact_sha256"],  # type: ignore[arg-type]
    )


def _restore_frozen_candidates(
    raw: Mapping[str, object],
) -> FrozenDevelopmentCandidateSet:
    return FrozenDevelopmentCandidateSet(
        protocol_sha256=raw["protocol_sha256"],  # type: ignore[arg-type]
        calibration_sha256=raw["calibration_sha256"],  # type: ignore[arg-type]
        calibrated_fit_panel_sha256=raw[
            "calibrated_fit_panel_sha256"
        ],  # type: ignore[arg-type]
        rank_ladder=tuple(raw["rank_ladder"]),  # type: ignore[arg-type]
        candidate_ids=tuple(raw["candidate_ids"]),  # type: ignore[arg-type]
        candidate_artifact_sha256s=tuple(
            raw["candidate_artifact_sha256s"]  # type: ignore[arg-type]
        ),
        artifact_sha256=raw["artifact_sha256"],  # type: ignore[arg-type]
    )


def _load_c2_context(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_file_sha256: str,
    expected_report_payload_sha256: str,
    expected_logical_artifact_sha256: str,
) -> _C2Context:
    source = Path(path)
    expected_file = _sha256(
        expected_file_sha256,
        label="expected C2 tensor file",
    )
    actual_file = _file_sha256(source)
    if actual_file != expected_file:
        raise ValueError("C2 tensor file differs from its pinned SHA-256")
    report_path = source.with_suffix(".json")
    actual_report_file = _file_sha256(report_path)
    if actual_report_file != _sha256(
        expected_report_file_sha256,
        label="expected C2 report file",
    ):
        raise ValueError("C2 report file differs from its pinned SHA-256")
    with report_path.open("r", encoding="utf-8") as handle:
        report = _mapping(json.load(handle), label="C2 report")
    claimed_report = _sha256(
        report.get("report_sha256"),
        label="C2 report payload",
    )
    report_payload = dict(report)
    report_payload.pop("report_sha256")
    if (
        claimed_report
        != _sha256(
            expected_report_payload_sha256,
            label="expected C2 report payload",
        )
        or c2_development._json_sha256(
            report_payload,
            domain=c2_development._REPORT_DOMAIN,
        )
        != claimed_report
    ):
        raise ValueError("C2 report payload authentication failed")
    report_artifact = _mapping(
        report.get("artifact"),
        label="C2 report artifact",
    )
    if report_artifact.get("tensor_file_sha256") != actual_file:
        raise ValueError("C2 report does not bind the tensor file")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    state = _mapping(raw, label="C2 tensor state")
    if set(state) != _C2_REQUIRED_STATE_KEYS:
        raise ValueError("C2 tensor state fields differ")
    logical = _sha256(
        state.get("artifact_sha256"),
        label="C2 logical artifact",
    )
    if (
        logical
        != _sha256(
            expected_logical_artifact_sha256,
            label="expected C2 logical artifact",
        )
        or report.get("artifact_sha256") != logical
    ):
        raise ValueError("C2 logical artifact authentication failed")
    protocol = default_contrast_provider_development_protocol()
    if _canonical_json_bytes(state["protocol_state"]) != _canonical_json_bytes(
        protocol.state_dict()
    ):
        raise ValueError("C2 protocol state differs from the authenticated C2")
    calibration = _restore_calibration(
        _mapping(state["calibration_state"], label="C2 calibration")
    )
    frozen = _restore_frozen_candidates(
        _mapping(
            state["frozen_candidate_set_state"],
            label="C2 frozen candidates",
        )
    )
    if (
        calibration.protocol_sha256 != protocol.protocol_sha256
        or frozen.protocol_sha256 != protocol.protocol_sha256
        or frozen.calibration_sha256 != calibration.artifact_sha256
        or report.get("protocol_sha256") != protocol.protocol_sha256
        or report.get("calibration_sha256") != calibration.artifact_sha256
        or report.get("candidate_set_sha256") != frozen.artifact_sha256
        or report.get("all_candidates_frozen_before_selection_materialization")
        is not True
        or report.get("selection_data_changed_training") is not False
        or report.get("prompt_text_loaded") is not False
        or report.get("token_ids_loaded") is not False
        or report.get("tokenizer_loaded") is not False
    ):
        raise ValueError("C2 split, safety, or provenance binding differs")
    return _C2Context(
        tensor_file_sha256=actual_file,
        report_file_sha256=actual_report_file,
        report_payload_sha256=claimed_report,
        logical_artifact_sha256=logical,
        protocol=protocol,
        calibration=calibration,
        frozen_candidates=frozen,
    )


def build_graph_organized_plan_set(
    base_plan: ConditionalSpectralGeneratorPlan,
    graph_basis: FitOnlyGraphSourceBasis,
    *,
    frequency_band_boundaries: Sequence[int] = (
        DEFAULT_FREQUENCY_BAND_BOUNDARIES
    ),
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
) -> tuple[tuple[str, ...], tuple[GraphOrganizedSVDPlan, ...]]:
    """Build the signed organization and exact size-matched controls."""

    seeds = tuple(random_seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("random seeds must be unique nonnegative integers")
    signed = organize_conditional_svd_with_graph(
        base_plan,
        graph_basis,
        organization_kind="signed_gfa_dyadic",
        frequency_band_boundaries=frequency_band_boundaries,
    )
    contiguous = organize_conditional_svd_with_graph(
        base_plan,
        graph_basis,
        organization_kind="singular_contiguous_control",
        frequency_band_boundaries=frequency_band_boundaries,
        matched_pack_counts=signed.pack_counts,
    )
    keys = ["signed_gfa", "singular_contiguous"]
    plans = [signed, contiguous]
    for seed in seeds:
        keys.append(f"random_size_matched:seed{seed}")
        plans.append(
            organize_conditional_svd_with_graph(
                base_plan,
                graph_basis,
                organization_kind="random_size_matched_control",
                organization_seed=seed,
                frequency_band_boundaries=frequency_band_boundaries,
                matched_pack_counts=signed.pack_counts,
            )
        )
    return tuple(keys), tuple(plans)


def _materialized_rows(
    batches: Sequence[MaterializedDevelopmentBatch],
) -> tuple[Tensor, Tensor, dict[str, int]]:
    values = tuple(batches)
    if not values:
        raise ValueError("development batches cannot be empty")
    raw = torch.cat(
        tuple(batch.values.reshape(-1, batch.values.shape[-1]) for batch in values)
    ).to(dtype=torch.float64)
    nonzero = (raw != 0.0).any(dim=1)
    filtered = raw[nonzero].contiguous()
    if not filtered.numel():
        raise ValueError("every development row was zero")
    unique, multiplicities = torch.unique(
        filtered,
        dim=0,
        sorted=True,
        return_counts=True,
    )
    return (
        unique.contiguous(),
        multiplicities.to(dtype=torch.float64).contiguous(),
        {
            "raw_row_count": int(raw.shape[0]),
            "zero_norm_row_count": int((~nonzero).sum()),
            "nonzero_row_count": int(nonzero.sum()),
            "unique_nonzero_row_count": int(unique.shape[0]),
        },
    )


def _quadratic(value: Tensor, matrix: Tensor) -> Tensor:
    return torch.einsum("nr,rs,ns->n", value, matrix, value)


def evaluate_graph_organized_plan(
    plan: GraphOrganizedSVDPlan,
    standardized_rows: Tensor,
    multiplicities: Tensor,
    row_counts: Mapping[str, int],
    *,
    dense_weighted_kernel: Tensor,
    origin: int,
    role: str,
    plan_key: str,
    route_fractions: Sequence[float] = DEFAULT_ROUTE_FRACTIONS,
) -> tuple[dict[str, object], ...]:
    """Score one frozen organization without materializing large outputs."""

    plan.validate_integrity()
    rows = standardized_rows.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    weights = multiplicities.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    dense = dense_weighted_kernel.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if (
        rows.ndim != 2
        or rows.shape[1] != plan.source_modes
        or weights.shape != (rows.shape[0],)
        or bool((weights <= 0.0).any())
        or dense.shape
        != (plan.source_modes, plan.lag_count, plan.target_modes)
        or not bool(torch.isfinite(rows).all())
        or not bool(torch.isfinite(dense).all())
    ):
        raise ValueError("evaluation rows, weights, or dense kernel are invalid")
    if bool((rows == 0.0).all(dim=1).any()):
        raise ValueError("zero-norm rows must be filtered before route scoring")
    fractions = tuple(float(value) for value in route_fractions)
    if (
        not fractions
        or tuple(sorted(set(fractions))) != fractions
        or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in fractions
        )
    ):
        raise ValueError("route fractions must be increasing values in (0, 1]")
    latent = rows @ plan.source_basis
    core = plan.core_at_origin(origin)
    core_flat = core.permute(1, 0, 2).reshape(plan.source_rank, -1)
    dense_flat = dense.reshape(plan.source_modes, -1)
    core_gram = core_flat @ core_flat.T
    dense_gram = dense_flat @ dense_flat.T
    core_dense = core_flat @ dense_flat.T
    full_square = _quadratic(latent, core_gram).clamp_min(0.0)
    dense_square = _quadratic(rows, dense_gram).clamp_min(0.0)
    full_dense_dot = torch.einsum(
        "nr,rs,ns->n",
        latent,
        core_dense,
        rows,
    )
    full_dense_residual = (
        full_square + dense_square - 2.0 * full_dense_dot
    ).clamp_min(0.0)
    full_dense_error = _relative_error_from_squares(
        full_dense_residual,
        dense_square,
        weights,
    )
    full_dense_cosine = _cosine_from_terms(
        full_dense_dot,
        full_square,
        dense_square,
        weights,
    )
    pack_ranks = (
        plan.pack_offsets[1:] - plan.pack_offsets[:-1]
    ).to(dtype=torch.int64)
    result: list[dict[str, object]] = []
    for fraction in fractions:
        pack_mask, scores = plan.bound_mass_route_mask(
            rows,
            origin=origin,
            retained_bound_fraction=fraction,
        )
        rank_mask = torch.zeros(
            (rows.shape[0], plan.source_rank),
            dtype=torch.bool,
        )
        for pack in range(plan.pack_count):
            start = int(plan.pack_offsets[pack])
            stop = int(plan.pack_offsets[pack + 1])
            rank_mask[:, start:stop] = pack_mask[:, pack : pack + 1]
        routed_latent = torch.where(
            rank_mask,
            latent,
            torch.zeros_like(latent),
        )
        omitted_latent = routed_latent - latent
        routed_square = _quadratic(
            routed_latent,
            core_gram,
        ).clamp_min(0.0)
        omitted_square = _quadratic(
            omitted_latent,
            core_gram,
        ).clamp_min(0.0)
        routed_full_dot = torch.einsum(
            "nr,rs,ns->n",
            routed_latent,
            core_gram,
            latent,
        )
        routed_dense_dot = torch.einsum(
            "nr,rs,ns->n",
            routed_latent,
            core_dense,
            rows,
        )
        routed_dense_residual = (
            routed_square + dense_square - 2.0 * routed_dense_dot
        ).clamp_min(0.0)
        certified = plan.omitted_source_response_bound(
            rows,
            origin=origin,
            pack_mask=pack_mask,
        )
        actual = omitted_square.sqrt()
        bound_violation = float((actual - certified).max())
        active_ranks = (
            pack_mask.to(dtype=torch.int64)
            * pack_ranks.view(1, -1)
        ).sum(dim=1)
        active_packs = pack_mask.sum(dim=1)
        expanded_ranks = torch.repeat_interleave(
            active_ranks,
            weights.to(dtype=torch.int64),
        ).to(dtype=torch.float64)
        total_rows = int(weights.sum())
        active_rank_instances = int(
            (
                active_ranks.to(dtype=torch.float64)
                * weights
            ).sum()
        )
        active_pack_instances = int(
            (
                active_packs.to(dtype=torch.float64)
                * weights
            ).sum()
        )
        source_projection_macs = (
            total_rows * plan.source_modes * plan.source_rank
        )
        cached_core_transport_macs = (
            active_rank_instances * plan.lag_count * plan.target_modes
        )
        cached_core_factorized_macs = (
            source_projection_macs + cached_core_transport_macs
        )
        dense_measured_response_macs = (
            total_rows
            * plan.source_modes
            * plan.lag_count
            * plan.target_modes
        )
        result.append(
            {
                "role": role,
                "origin": origin,
                "plan_key": plan_key,
                "organization_kind": plan.organization_kind,
                "organization_seed": plan.organization_seed,
                "plan_artifact_sha256": plan.artifact_sha256,
                "route_fraction": fraction,
                **dict(row_counts),
                "evaluated_row_count": total_rows,
                "mean_active_rank": active_rank_instances / total_rows,
                "p90_active_rank": float(
                    torch.quantile(expanded_ranks, 0.90)
                ),
                "mean_active_pack_count": (
                    active_pack_instances / total_rows
                ),
                "active_rank_fraction": (
                    active_rank_instances
                    / (total_rows * plan.source_rank)
                ),
                "source_projection_macs": source_projection_macs,
                "cached_core_transport_macs": (
                    cached_core_transport_macs
                ),
                "cached_core_factorized_macs": (
                    cached_core_factorized_macs
                ),
                "dense_measured_response_macs": (
                    dense_measured_response_macs
                ),
                "cached_core_mac_fraction_vs_dense": (
                    cached_core_factorized_macs
                    / dense_measured_response_macs
                ),
                "cached_core_mac_reduction_vs_dense": (
                    1.0
                    - cached_core_factorized_macs
                    / dense_measured_response_macs
                ),
                "routed_relative_error_vs_full_svd": (
                    _relative_error_from_squares(
                        omitted_square,
                        full_square,
                        weights,
                    )
                ),
                "routed_cosine_vs_full_svd": _cosine_from_terms(
                    routed_full_dot,
                    routed_square,
                    full_square,
                    weights,
                ),
                "routed_relative_error_vs_dense_measured_response": (
                    _relative_error_from_squares(
                        routed_dense_residual,
                        dense_square,
                        weights,
                    )
                ),
                "routed_cosine_vs_dense_measured_response": (
                    _cosine_from_terms(
                        routed_dense_dot,
                        routed_square,
                        dense_square,
                        weights,
                    )
                ),
                "full_svd_relative_error_vs_dense_measured_response": (
                    full_dense_error
                ),
                "full_svd_cosine_vs_dense_measured_response": (
                    full_dense_cosine
                ),
                "maximum_actual_omitted_output_norm": float(actual.max()),
                "maximum_certified_omitted_output_bound": float(
                    certified.max()
                ),
                "maximum_bound_violation": bound_violation,
                "certified_omitted_output_bound_holds": (
                    bound_violation <= 2e-10
                ),
                "route_mask_sha256": _tensor_sha256(pack_mask),
                "zero_norm_rows_filtered_before_route_scoring": True,
                "cached_core_assumed": True,
                "router_cost_included_in_cached_core_macs": False,
            }
        )
    return tuple(result)


def _canonical_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rate rows must be a sequence")
    value = json.loads(_canonical_json_bytes(list(rows)).decode("ascii"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise TypeError("rate rows must contain mappings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class Gemma3GraphOrganizedSVDCandidate:
    source_artifact_file_sha256: str
    source_report_file_sha256: str
    source_report_payload_sha256: str
    source_mapping_artifact_sha256: str
    c2_artifact_file_sha256: str
    c2_report_file_sha256: str
    c2_report_payload_sha256: str
    c2_logical_artifact_sha256: str
    c2_protocol_sha256: str
    c2_calibration_sha256: str
    c2_candidate_set_sha256: str
    binding: Mapping[str, object]
    model: Mapping[str, object]
    base_plan: ConditionalSpectralGeneratorPlan
    graph_basis: FitOnlyGraphSourceBasis
    plan_keys: tuple[str, ...]
    plans: tuple[GraphOrganizedSVDPlan, ...]
    rate_rows: tuple[Mapping[str, object], ...]
    conclusions: Mapping[str, object]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for field in (
            "source_artifact_file_sha256",
            "source_report_file_sha256",
            "source_report_payload_sha256",
            "source_mapping_artifact_sha256",
            "c2_artifact_file_sha256",
            "c2_report_file_sha256",
            "c2_report_payload_sha256",
            "c2_logical_artifact_sha256",
            "c2_protocol_sha256",
            "c2_calibration_sha256",
            "c2_candidate_set_sha256",
        ):
            _sha256(getattr(self, field), label=field)
        if not isinstance(self.base_plan, ConditionalSpectralGeneratorPlan):
            raise TypeError("base_plan must be a conditional spectral plan")
        if not isinstance(self.graph_basis, FitOnlyGraphSourceBasis):
            raise TypeError("graph_basis must be a fit-only graph basis")
        self.base_plan.validate_integrity()
        self.graph_basis.validate_integrity()
        keys = tuple(self.plan_keys)
        plans = tuple(self.plans)
        if (
            not keys
            or len(set(keys)) != len(keys)
            or len(keys) != len(plans)
            or any(not isinstance(plan, GraphOrganizedSVDPlan) for plan in plans)
        ):
            raise ValueError("plan keys and plans are invalid")
        for plan in plans:
            plan.validate_integrity()
            if (
                plan.response_binding_sha256
                != self.base_plan.response_binding_sha256
                or plan.source_plan_artifact_sha256
                != self.base_plan.artifact_sha256
                or plan.fit_weighted_kernels_sha256
                != self.base_plan.fit_weighted_kernels_sha256
                or plan.graph_basis_artifact_sha256
                != self.graph_basis.artifact_sha256
                or plan.source_rank != self.base_plan.source_rank
                or plan.target_modes != self.base_plan.target_modes
            ):
                raise ValueError("organized plan provenance differs")
        rows = _canonical_rows(self.rate_rows)
        expected_coordinates = {
            (role, key, fraction)
            for role in ("fit", "selection")
            for key in keys
            for fraction in DEFAULT_ROUTE_FRACTIONS
        }
        actual_coordinates = {
            (
                row.get("role"),
                row.get("plan_key"),
                row.get("route_fraction"),
            )
            for row in rows
        }
        if actual_coordinates != expected_coordinates or len(rows) != len(
            expected_coordinates
        ):
            raise ValueError("rate row coordinates are incomplete")
        if any(
            row.get("zero_norm_rows_filtered_before_route_scoring") is not True
            or row.get("certified_omitted_output_bound_holds") is not True
            for row in rows
        ):
            raise ValueError("rate rows violate filtering or bound requirements")
        object.__setattr__(self, "binding", dict(self.binding))
        object.__setattr__(self, "model", dict(self.model))
        object.__setattr__(self, "plan_keys", keys)
        object.__setattr__(self, "plans", plans)
        object.__setattr__(self, "rate_rows", rows)
        object.__setattr__(self, "conclusions", dict(self.conclusions))
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _sha256(self.artifact_sha256, label="artifact_sha256")
                != computed
            ):
                raise ValueError("graph-organized candidate hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "source_artifact_file_sha256": self.source_artifact_file_sha256,
            "source_report_file_sha256": self.source_report_file_sha256,
            "source_report_payload_sha256": self.source_report_payload_sha256,
            "source_mapping_artifact_sha256": (
                self.source_mapping_artifact_sha256
            ),
            "c2_artifact_file_sha256": self.c2_artifact_file_sha256,
            "c2_report_file_sha256": self.c2_report_file_sha256,
            "c2_report_payload_sha256": self.c2_report_payload_sha256,
            "c2_logical_artifact_sha256": self.c2_logical_artifact_sha256,
            "c2_protocol_sha256": self.c2_protocol_sha256,
            "c2_calibration_sha256": self.c2_calibration_sha256,
            "c2_candidate_set_sha256": self.c2_candidate_set_sha256,
            "binding": dict(self.binding),
            "model": dict(self.model),
            "base_plan_artifact_sha256": self.base_plan.artifact_sha256,
            "graph_basis_artifact_sha256": self.graph_basis.artifact_sha256,
            "plan_keys": self.plan_keys,
            "plan_artifact_sha256s": tuple(
                plan.artifact_sha256 for plan in self.plans
            ),
            "rate_rows": self.rate_rows,
            "conclusions": dict(self.conclusions),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_ARTIFACT_DOMAIN)

    def validate_integrity(self) -> None:
        self.base_plan.validate_integrity()
        self.graph_basis.validate_integrity()
        for plan in self.plans:
            plan.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("graph-organized candidate hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "base_plan": self.base_plan.state_dict(),
            "graph_basis": self.graph_basis.state_dict(),
            "plans": tuple(plan.state_dict() for plan in self.plans),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "Gemma3GraphOrganizedSVDCandidate":
        state = _mapping(raw, label="graph-organized candidate state")
        expected = {
            "schema",
            "format_version",
            "source_artifact_file_sha256",
            "source_report_file_sha256",
            "source_report_payload_sha256",
            "source_mapping_artifact_sha256",
            "c2_artifact_file_sha256",
            "c2_report_file_sha256",
            "c2_report_payload_sha256",
            "c2_logical_artifact_sha256",
            "c2_protocol_sha256",
            "c2_calibration_sha256",
            "c2_candidate_set_sha256",
            "binding",
            "model",
            "base_plan_artifact_sha256",
            "graph_basis_artifact_sha256",
            "plan_keys",
            "plan_artifact_sha256s",
            "rate_rows",
            "conclusions",
            "base_plan",
            "graph_basis",
            "plans",
            "artifact_sha256",
        }
        if set(state) != expected:
            raise ValueError("graph-organized candidate state fields differ")
        if state["schema"] != _SCHEMA or state["format_version"] != _FORMAT_VERSION:
            raise ValueError("graph-organized candidate schema differs")
        base = ConditionalSpectralGeneratorPlan.from_state_dict(
            state["base_plan"]
        )
        graph = FitOnlyGraphSourceBasis.from_state_dict(state["graph_basis"])
        plans = tuple(
            GraphOrganizedSVDPlan.from_state_dict(value)
            for value in state["plans"]  # type: ignore[union-attr]
        )
        result = cls(
            source_artifact_file_sha256=state[
                "source_artifact_file_sha256"
            ],  # type: ignore[arg-type]
            source_report_file_sha256=state[
                "source_report_file_sha256"
            ],  # type: ignore[arg-type]
            source_report_payload_sha256=state[
                "source_report_payload_sha256"
            ],  # type: ignore[arg-type]
            source_mapping_artifact_sha256=state[
                "source_mapping_artifact_sha256"
            ],  # type: ignore[arg-type]
            c2_artifact_file_sha256=state[
                "c2_artifact_file_sha256"
            ],  # type: ignore[arg-type]
            c2_report_file_sha256=state[
                "c2_report_file_sha256"
            ],  # type: ignore[arg-type]
            c2_report_payload_sha256=state[
                "c2_report_payload_sha256"
            ],  # type: ignore[arg-type]
            c2_logical_artifact_sha256=state[
                "c2_logical_artifact_sha256"
            ],  # type: ignore[arg-type]
            c2_protocol_sha256=state[
                "c2_protocol_sha256"
            ],  # type: ignore[arg-type]
            c2_calibration_sha256=state[
                "c2_calibration_sha256"
            ],  # type: ignore[arg-type]
            c2_candidate_set_sha256=state[
                "c2_candidate_set_sha256"
            ],  # type: ignore[arg-type]
            binding=state["binding"],  # type: ignore[arg-type]
            model=state["model"],  # type: ignore[arg-type]
            base_plan=base,
            graph_basis=graph,
            plan_keys=tuple(state["plan_keys"]),  # type: ignore[arg-type]
            plans=plans,
            rate_rows=tuple(state["rate_rows"]),  # type: ignore[arg-type]
            conclusions=state["conclusions"],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
        )
        if (
            state["base_plan_artifact_sha256"] != base.artifact_sha256
            or state["graph_basis_artifact_sha256"] != graph.artifact_sha256
            or tuple(state["plan_artifact_sha256s"])
            != tuple(plan.artifact_sha256 for plan in plans)
        ):
            raise ValueError("serialized plan hashes differ")
        return result


def compile_gemma3_graph_organized_svd_candidate(
    source: Gemma3SpectralSource,
    c2_context: _C2Context,
) -> Gemma3GraphOrganizedSVDCandidate:
    if not isinstance(source, Gemma3SpectralSource):
        raise TypeError("source must be a Gemma3SpectralSource")
    if not isinstance(c2_context, _C2Context):
        raise TypeError("c2_context must be an authenticated C2 context")
    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    sigma = source.source_mode_standard_deviations
    origins = source.mapping.impulse_logical_positions
    response_binding = _response_binding(
        source,
        component="local_central_odd_tangent",
    )
    graph = fit_graph_source_bases(
        local.impulse_responses,
        sigma,
        origins,
        FIT_ORIGINS,
        response_binding_sha256=response_binding,
        fft_length=source.mapping.fft_length,
    )
    base = fit_conditional_spectral_generator(
        local.impulse_responses,
        sigma,
        origins,
        FIT_ORIGINS,
        SOURCE_RANK,
        TARGET_RANK,
        response_binding_sha256=response_binding,
        input_transform="standardized_linear",
        fft_length=source.mapping.fft_length,
    )
    plan_keys, plans = build_graph_organized_plan_set(base, graph)
    # Fit inputs are materialized first.  Selection materialization occurs only
    # after every organization and route threshold is frozen.
    fit_batches = materialize_development_role(
        c2_context.protocol,
        "fit",
        calibration=c2_context.calibration,
    )
    fit_rows, fit_multiplicities, fit_counts = _materialized_rows(fit_batches)
    fit_dense = (
        local.impulse_responses[
            :,
            origins.index(FIT_EVALUATION_ORIGIN),
        ]
        * sigma.view(-1, 1, 1)
    )
    rate_rows: list[dict[str, object]] = []
    for key, plan in zip(plan_keys, plans, strict=True):
        rate_rows.extend(
            evaluate_graph_organized_plan(
                plan,
                fit_rows,
                fit_multiplicities,
                fit_counts,
                dense_weighted_kernel=fit_dense,
                origin=FIT_EVALUATION_ORIGIN,
                role="fit",
                plan_key=key,
            )
        )
    selection_batches = materialize_development_role(
        c2_context.protocol,
        "selection",
        calibration=c2_context.calibration,
        frozen_candidates=c2_context.frozen_candidates,
    )
    selection_rows, selection_multiplicities, selection_counts = (
        _materialized_rows(selection_batches)
    )
    selection_dense = (
        local.impulse_responses[
            :,
            origins.index(SELECTION_EVALUATION_ORIGIN),
        ]
        * sigma.view(-1, 1, 1)
    )
    for key, plan in zip(plan_keys, plans, strict=True):
        rate_rows.extend(
            evaluate_graph_organized_plan(
                plan,
                selection_rows,
                selection_multiplicities,
                selection_counts,
                dense_weighted_kernel=selection_dense,
                origin=SELECTION_EVALUATION_ORIGIN,
                role="selection",
                plan_key=key,
            )
        )
    conclusions = {
        "numerical_compression_basis": "global_svd_rank45_full_target",
        "graph_role": "fit_only_generator_pack_organization",
        "frequency_band_boundaries": DEFAULT_FREQUENCY_BAND_BOUNDARIES,
        "signed_graph_pack_counts": plans[0].pack_counts,
        "fit_origin": FIT_EVALUATION_ORIGIN,
        "selection_origin": SELECTION_EVALUATION_ORIGIN,
        "selection_origin_was_held_out_from_svd_and_graph_fit": True,
        "all_organizations_frozen_before_c2_selection_materialization": True,
        "selection_data_changed_organization_or_thresholds": False,
        "c2_development_selection_was_already_opened_upstream": True,
        "fresh_confirmation_or_assessment_claim": False,
        "zero_norm_rows_filtered_before_route_scoring": True,
        "cached_core_mac_accounting_is_analytic": True,
        "router_cost_is_reported_as_excluded_from_cached_core_macs": True,
        "conditional_compute_is_measured_only_as_analytic_macs": True,
        "wall_clock_latency_measured": False,
        "nll_or_task_accuracy_measured": False,
        "whole_model_or_complete_block_replacement_tested": False,
    }
    return Gemma3GraphOrganizedSVDCandidate(
        source_artifact_file_sha256=source.file_sha256,
        source_report_file_sha256=source.report_file_sha256,
        source_report_payload_sha256=source.report_payload_sha256,
        source_mapping_artifact_sha256=source.mapping.artifact_sha256,
        c2_artifact_file_sha256=c2_context.tensor_file_sha256,
        c2_report_file_sha256=c2_context.report_file_sha256,
        c2_report_payload_sha256=c2_context.report_payload_sha256,
        c2_logical_artifact_sha256=c2_context.logical_artifact_sha256,
        c2_protocol_sha256=c2_context.protocol.protocol_sha256,
        c2_calibration_sha256=c2_context.calibration.artifact_sha256,
        c2_candidate_set_sha256=(
            c2_context.frozen_candidates.artifact_sha256
        ),
        binding=source.binding,
        model=source.model,
        base_plan=base,
        graph_basis=graph,
        plan_keys=plan_keys,
        plans=plans,
        rate_rows=tuple(rate_rows),
        conclusions=conclusions,
    )


def _claim_boundaries() -> dict[str, object]:
    return {
        "prompt_blind_synthetic_c2_directions": True,
        "upstream_fisher_basis_is_prompt_derived": True,
        "fit_only_graph_and_svd_compilation": True,
        "heldout_origin_and_c2_selection_evaluation": True,
        "c2_selection_is_open_development_not_fresh_confirmation": True,
        "fixed_reference_modal_delta_executor": True,
        "complete_gemma_block_replacement": False,
        "natural_prompt_validation": False,
        "nll_kl_or_top1_fidelity": False,
        "downstream_task_accuracy": False,
        "whole_model_parameter_compression": False,
        "runtime_latency_or_speed": False,
        "analytic_cached_core_mac_accounting_only": True,
        "development_only": True,
    }


_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_prompt_blind_c2_directions": False,
    "contains_compiled_plan_tensors": True,
    "contains_scalar_rate_rows": True,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


def _publish_candidate(
    candidate: Gemma3GraphOrganizedSVDCandidate,
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    reservation = _reserve_outputs((output, report_path))
    tensor_stage: Path | None = None
    report_stage: Path | None = None
    try:
        tensor_stage = _stage_torch(candidate.state_dict(), output)
        tensor_digest = _file_sha256(tensor_stage)
        report: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "candidate": candidate.metadata(),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": tensor_digest,
                "tensor_file_bytes": tensor_stage.stat().st_size,
                "report_file": str(report_path),
                "committable": False,
            },
            "scientific_status": _claim_boundaries(),
            "safety": _SAFETY,
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        _canonical_json_bytes(report)
        report_stage = _stage_json(report, report_path)
        reservation.publish((tensor_stage, report_stage))
        return report
    finally:
        reservation.release()
        if tensor_stage is not None:
            tensor_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)


def compile_gemma3_l3_l4_graph_organized_svd(
    *,
    source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    source_artifact_sha256: str = DEFAULT_INTERIOR_ARTIFACT_SHA256,
    source_report_sha256: str = DEFAULT_INTERIOR_REPORT_SHA256,
    c2_artifact_path: Path | str = DEFAULT_C2_ARTIFACT,
    c2_artifact_sha256: str = DEFAULT_C2_ARTIFACT_SHA256,
    c2_report_file_sha256: str = DEFAULT_C2_REPORT_FILE_SHA256,
    c2_report_payload_sha256: str = DEFAULT_C2_REPORT_PAYLOAD_SHA256,
    c2_logical_artifact_sha256: str = (
        DEFAULT_C2_LOGICAL_ARTIFACT_SHA256
    ),
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    if (
        source_artifact_sha256 != DEFAULT_INTERIOR_ARTIFACT_SHA256
        or source_report_sha256 != DEFAULT_INTERIOR_REPORT_SHA256
        or c2_artifact_sha256 != DEFAULT_C2_ARTIFACT_SHA256
        or c2_report_file_sha256 != DEFAULT_C2_REPORT_FILE_SHA256
        or c2_report_payload_sha256 != DEFAULT_C2_REPORT_PAYLOAD_SHA256
        or c2_logical_artifact_sha256
        != DEFAULT_C2_LOGICAL_ARTIFACT_SHA256
    ):
        raise ValueError("compile inputs must equal the pinned source and C2")
    destination = _validate_output_path(output, suffix=".pt")
    if destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite candidate report")
    source = load_gemma3_spectral_source(
        source_artifact_path,
        expected_file_sha256=source_artifact_sha256,
        expected_report_sha256=source_report_sha256,
        expected_origins=INTERIOR_ORIGINS,
    )
    context = _load_c2_context(
        c2_artifact_path,
        expected_file_sha256=c2_artifact_sha256,
        expected_report_file_sha256=c2_report_file_sha256,
        expected_report_payload_sha256=c2_report_payload_sha256,
        expected_logical_artifact_sha256=c2_logical_artifact_sha256,
    )
    candidate = compile_gemma3_graph_organized_svd_candidate(source, context)
    return _publish_candidate(candidate, output=destination)


def load_gemma3_graph_organized_svd_candidate(
    path: Path | str,
    *,
    expected_file_sha256: str,
) -> Gemma3GraphOrganizedSVDCandidate:
    source = Path(path)
    if _file_sha256(source) != _sha256(
        expected_file_sha256,
        label="expected candidate tensor file",
    ):
        raise ValueError("candidate tensor file differs from expected SHA-256")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    return Gemma3GraphOrganizedSVDCandidate.from_state_dict(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=DEFAULT_INTERIOR_ARTIFACT,
    )
    parser.add_argument(
        "--source-artifact-sha256",
        default=DEFAULT_INTERIOR_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--source-report-sha256",
        default=DEFAULT_INTERIOR_REPORT_SHA256,
    )
    parser.add_argument(
        "--c2-artifact",
        type=Path,
        default=DEFAULT_C2_ARTIFACT,
    )
    parser.add_argument(
        "--c2-artifact-sha256",
        default=DEFAULT_C2_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--c2-report-file-sha256",
        default=DEFAULT_C2_REPORT_FILE_SHA256,
    )
    parser.add_argument(
        "--c2-report-payload-sha256",
        default=DEFAULT_C2_REPORT_PAYLOAD_SHA256,
    )
    parser.add_argument(
        "--c2-logical-artifact-sha256",
        default=DEFAULT_C2_LOGICAL_ARTIFACT_SHA256,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compile_gemma3_l3_l4_graph_organized_svd(
        source_artifact_path=args.source_artifact,
        source_artifact_sha256=args.source_artifact_sha256,
        source_report_sha256=args.source_report_sha256,
        c2_artifact_path=args.c2_artifact,
        c2_artifact_sha256=args.c2_artifact_sha256,
        c2_report_file_sha256=args.c2_report_file_sha256,
        c2_report_payload_sha256=args.c2_report_payload_sha256,
        c2_logical_artifact_sha256=args.c2_logical_artifact_sha256,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
