"""Prompt-free phase-aware and graph Fourier analysis of the frozen L3/L4 map.

This rung consumes the already-published Gemma L3/L4 spectral map.  It does
not load Gemma, prompts, token IDs, a tokenizer, or activation rows.  The
source spectra are reauthenticated before three response families are
analyzed:

* one-sided finite secants;
* local symmetric central secants; and
* operating-scale symmetric central secants.

The existing magnitude-plus-phase similarity remains intact.  A new
antisymmetric quadrature companion retains lead-versus-lag orientation, and
complex connection/signed Laplacians provide graph Fourier bases over source
modes.  This is structural analysis, not an executor or compression result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch
from torch import Tensor

from .modal_spectral_mapping import ModalSpectralMapping
from .phase_graph_spectral_analysis import (
    DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
    DEFAULT_RELATIVE_SUPPORT_FLOOR,
    PhaseGraphSpectralAnalysis,
    analyze_phase_graph_spectral_response,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE",
    "DEFAULT_SOURCE_FILE_SHA256",
    "DEFAULT_SOURCE_REPORT_SHA256",
    "LoadedGemma3PhaseGraphSpectralArtifact",
    "analyze_gemma3_l3_l4_phase_graph_spectral",
    "build_parser",
    "describe_gemma3_l3_l4_phase_graph_spectral",
    "load_gemma3_l3_l4_phase_graph_spectral_artifact",
    "main",
]


DEFAULT_SOURCE = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-spectral-map-dev-v2.pt"
)
DEFAULT_SOURCE_FILE_SHA256 = (
    "b88201f33210c8cd5be0d28f144b1963de84f51ba065959391aad3ce496c59d3"
)
DEFAULT_SOURCE_REPORT_SHA256 = (
    "00ac238e9867f001aa8b0926f2f05087b151d39e0cacf6f92d6192c50ac56165"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-phase-graph-spectral-analysis-dev-v1.pt"
)
DEFAULT_NEIGHBOR_COUNT = 6
DEFAULT_MINIMUM_COHERENCE = 0.10
DEFAULT_TOP_PAIR_COUNT = 8

_SCHEMA = "fisher_graph.gemma3_l3_l4_phase_graph_spectral_development"
_SOURCE_SCHEMA = "fisher_graph.gemma3_l3_l4_spectral_mapping_development"
_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-spectral-report:v1\0"
)
_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-phase-graph-spectral:artifact:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-phase-graph-spectral:report:v1\0"
)


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


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return float(value)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return value


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ")


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"could not read {label}") from error
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return dict(_mapping(value, label=label))


@dataclass(frozen=True, slots=True)
class _AuthenticatedSpectralSource:
    path: Path
    report_path: Path
    tensor_file_sha256: str
    report_sha256: str
    mapping: ModalSpectralMapping
    report: Mapping[str, object]


def _load_authenticated_source(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
) -> _AuthenticatedSpectralSource:
    supplied_source = Path(path).expanduser()
    if supplied_source.is_symlink():
        raise ValueError("phase-graph source must not be a symlink")
    source = supplied_source.resolve()
    if source.suffix != ".pt":
        raise ValueError("phase-graph source must use a .pt suffix")
    expected_file = _require_sha256(
        expected_file_sha256,
        label="expected source tensor file",
    )
    expected_report = _require_sha256(
        expected_report_sha256,
        label="expected source report",
    )
    if not source.is_file() or source.is_symlink():
        raise ValueError("phase-graph source must be a regular file")
    actual_file = _file_sha256(source)
    if actual_file != expected_file:
        raise ValueError("phase-graph source tensor file hash differs")
    report_path = source.with_suffix(".json")
    if not report_path.is_file() or report_path.is_symlink():
        raise ValueError("phase-graph source report must be a regular file")
    report = _read_json(report_path, label="phase-graph source report")
    supplied_report = _require_sha256(
        report.get("report_sha256"),
        label="source report SHA-256",
    )
    report_without_hash = dict(report)
    report_without_hash.pop("report_sha256")
    computed_report = _json_sha256(
        report_without_hash,
        domain=_SOURCE_REPORT_DOMAIN,
    )
    artifact_manifest = _mapping(
        report.get("artifact"),
        label="source artifact manifest",
    )
    if (
        supplied_report != computed_report
        or supplied_report != expected_report
        or artifact_manifest.get("tensor_file_sha256") != actual_file
        or report.get("schema") != _SOURCE_SCHEMA
        or report.get("format_version") != 1
    ):
        raise ValueError("phase-graph source report authentication failed")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    state = _mapping(raw, label="phase-graph source artifact")
    required = {
        "schema",
        "format_version",
        "scientific_status",
        "binding",
        "model",
        "protocol",
        "canonical_reference",
        "spectral_mapping",
        "safe_analysis",
        "safety",
    }
    _strict_keys(state, expected=required, label="phase-graph source artifact")
    if (
        state["schema"] != _SOURCE_SCHEMA
        or state["format_version"] != 1
    ):
        raise ValueError("phase-graph source artifact header differs")
    safety = _mapping(state["safety"], label="source safety")
    for field in (
        "contains_source_model_state_dict",
        "contains_tokenizer",
        "contains_prompt_text",
        "contains_token_ids",
        "contains_prompt_activation_rows",
        "contains_score_gradient_rows",
    ):
        if safety.get(field) is not False:
            raise ValueError("phase-graph source safety contract differs")
    if (
        safety.get("contains_spectral_response_tensors") is not True
        or safety.get("artifact_must_remain_outside_git") is not True
    ):
        raise ValueError("phase-graph source tensor contract differs")
    spectral_mapping = ModalSpectralMapping.from_state_dict(
        state["spectral_mapping"]
    )
    report_analysis = _mapping(
        report.get("analysis"),
        label="source report analysis",
    )
    report_mapping = _mapping(
        report_analysis.get("spectral_mapping"),
        label="source report spectral mapping",
    )
    if (
        report_mapping.get("artifact_sha256")
        != spectral_mapping.artifact_sha256
    ):
        raise ValueError("phase-graph source mapping receipt differs")
    return _AuthenticatedSpectralSource(
        path=source,
        report_path=report_path,
        tensor_file_sha256=actual_file,
        report_sha256=computed_report,
        mapping=spectral_mapping,
        report=report,
    )


def _projector_overlap(left: Tensor, right: Tensor, *, rank: int) -> float:
    if rank <= 0:
        return 1.0
    left_basis = left[:, :rank]
    right_basis = right[:, :rank]
    return float(
        (left_basis.mH @ right_basis).abs().square().sum()
    ) / rank


def _relative_frobenius(left: Tensor, right: Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(right))
    numerator = float(torch.linalg.vector_norm(left - right))
    return numerator / max(denominator, torch.finfo(torch.float64).eps)


def _cosine(left: Tensor, right: Tensor) -> float:
    first = left.reshape(-1).to(dtype=torch.float64)
    second = right.reshape(-1).to(dtype=torch.float64)
    denominator = float(
        torch.linalg.vector_norm(first)
        * torch.linalg.vector_norm(second)
    )
    if denominator <= torch.finfo(torch.float64).eps:
        return 1.0 if bool((first == second).all()) else 0.0
    return max(-1.0, min(1.0, float(torch.dot(first, second)) / denominator))


def _cross_response_comparisons(
    analyses: Mapping[str, PhaseGraphSpectralAnalysis],
) -> tuple[dict[str, object], ...]:
    labels = tuple(analyses)
    rows: list[dict[str, object]] = []
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            left = analyses[left_label]
            right = analyses[right_label]
            left.validate_integrity()
            right.validate_integrity()
            if left.source_mode_indices != right.source_mode_indices:
                raise ValueError("phase graphs use different source modes")
            union = left.selected_edge_mask | right.selected_edge_mask
            intersection = left.selected_edge_mask & right.selected_edge_mask
            upper = torch.triu(
                torch.ones_like(union, dtype=torch.bool),
                diagonal=1,
            )
            union_count = int((union & upper).sum())
            intersection_count = int((intersection & upper).sum())
            mode_count = len(left.source_mode_indices)
            rank = min(8, mode_count)
            left_connection_vectors = torch.complex(
                left.connection_eigenvectors_real,
                left.connection_eigenvectors_imag,
            )
            right_connection_vectors = torch.complex(
                right.connection_eigenvectors_real,
                right.connection_eigenvectors_imag,
            )
            rows.append(
                {
                    "left_label": left_label,
                    "right_label": right_label,
                    "edge_jaccard": (
                        intersection_count / union_count
                        if union_count
                        else 1.0
                    ),
                    "selected_edge_union_count": union_count,
                    "selected_edge_intersection_count": intersection_count,
                    "directional_quadrature_cosine": _cosine(
                        left.directional_quadrature,
                        right.directional_quadrature,
                    ),
                    "connection_laplacian_relative_difference": (
                        _relative_frobenius(
                            torch.complex(
                                left.connection_laplacian_real,
                                left.connection_laplacian_imag,
                            ),
                            torch.complex(
                                right.connection_laplacian_real,
                                right.connection_laplacian_imag,
                            ),
                        )
                    ),
                    "signed_laplacian_relative_difference": (
                        _relative_frobenius(
                            left.signed_laplacian,
                            right.signed_laplacian,
                        )
                    ),
                    "connection_low8_projector_overlap": (
                        _projector_overlap(
                            left_connection_vectors,
                            right_connection_vectors,
                            rank=rank,
                        )
                    ),
                    "signed_low8_projector_overlap": _projector_overlap(
                        left.signed_eigenvectors.to(torch.complex128),
                        right.signed_eigenvectors.to(torch.complex128),
                        rank=rank,
                    ),
                    "magnitude_laplacian_relative_difference": (
                        _relative_frobenius(
                            left.magnitude_laplacian,
                            right.magnitude_laplacian,
                        )
                    ),
                    "magnitude_low8_projector_overlap": _projector_overlap(
                        left.magnitude_eigenvectors.to(torch.complex128),
                        right.magnitude_eigenvectors.to(torch.complex128),
                        rank=rank,
                    ),
                }
            )
    return tuple(rows)


def _protocol(
    *,
    neighbor_count: int,
    minimum_coherence: float,
    top_pair_count: int,
) -> dict[str, object]:
    neighbor_count = _positive_int(neighbor_count, label="neighbor count")
    minimum_coherence = _finite_float(
        minimum_coherence,
        label="minimum coherence",
        minimum=0.0,
        maximum=1.0,
    )
    top_pair_count = _positive_int(
        top_pair_count,
        label="top pair count",
    )
    return {
        "response_labels": (
            "finite",
            "local_fraction_sigma",
            "operating_1_sigma",
        ),
        "neighbor_count": neighbor_count,
        "minimum_coherence": minimum_coherence,
        "top_pair_count": top_pair_count,
        "relative_mode_support_floor": DEFAULT_RELATIVE_SUPPORT_FLOOR,
        "eigenvalue_block_tolerance": (
            DEFAULT_EIGENVALUE_BLOCK_TOLERANCE
        ),
        "edge_selection": (
            "visualization_only_deterministic_union_of_per_node_top_k_by_"
            "absolute_raw_complex_coherency"
        ),
        "primary_graph": (
            "dense_raw_complex_spectral_coherency_with_zero_diagonal"
        ),
        "primary_graph_thresholded": False,
        "rfft_energy_weighting": (
            "dc_nyquist_once_interior_twice;real_alignment_is_parseval_"
            "consistent_but_complex_phase_is_fft_length_bound"
        ),
        "legacy_similarity_reused": True,
        "new_phase_statistic": (
            "coherence_weighted_antisymmetric_sine_relative_phase_"
            "quadrature"
        ),
        "graph_fourier_bases": (
            "normalized_dense_complex_connection_laplacian",
            "normalized_dense_signed_raw_complex_alignment_laplacian",
        ),
        "phase_blind_control": (
            "normalized_dense_per_bin_spectral_magnitude_cosine_laplacian"
        ),
        "source_model_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "selection_or_threshold_tuning": False,
        "response_gain_normalized": True,
        "response_norm_diagnostics_retained": True,
    }


def _safety() -> dict[str, bool]:
    return {
        "contains_source_model_state_dict": False,
        "contains_tokenizer": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_prompt_activation_rows": False,
        "contains_score_gradient_rows": False,
        "contains_derived_phase_graph_tensors": True,
        "contains_original_spectral_fingerprint_tensors": False,
        "contains_reused_derived_spectral_similarity": True,
        "artifact_must_remain_outside_git": True,
        "compression_claim": False,
        "coupling_strength_claim": False,
        "directed_transfer_graph_claim": False,
        "executor_claim": False,
        "frequency_resolved_delay_claim": False,
        "speed_claim": False,
        "semantic_cluster_claim": False,
    }


def _logical_payload(
    *,
    source: Mapping[str, object],
    protocol: Mapping[str, object],
    analyses: Mapping[str, PhaseGraphSpectralAnalysis],
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "source": dict(source),
        "protocol": dict(protocol),
        "analysis_artifact_sha256s": {
            label: value.artifact_sha256
            for label, value in analyses.items()
        },
        "cross_response_comparisons": tuple(
            dict(value) for value in comparisons
        ),
        "safety": _safety(),
    }


def _validate_output(path: Path | str) -> Path:
    output = Path(path).expanduser()
    if output.suffix != ".pt":
        raise ValueError("phase-graph output must use a .pt suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite phase-graph output")
    return output


def _stage_save(value: object, destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    path = Path(name)
    try:
        torch.save(value, path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _stage_json(value: Mapping[str, object], destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _publish(
    *,
    output: Path,
    artifact: Mapping[str, object],
    report_without_hash: Mapping[str, object],
) -> dict[str, object]:
    tensor_stage = _stage_save(dict(artifact), output)
    report_stage: Path | None = None
    tensor_published = False
    try:
        tensor_sha256 = _file_sha256(tensor_stage)
        tensor_bytes = tensor_stage.stat().st_size
        report = {
            **dict(report_without_hash),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": tensor_sha256,
                "tensor_file_bytes": tensor_bytes,
                "report_file": str(output.with_suffix(".json")),
                "committable": False,
            },
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        _canonical_json_bytes(report)
        report_stage = _stage_json(report, output.with_suffix(".json"))
        os.link(tensor_stage, output)
        tensor_published = True
        try:
            os.link(report_stage, output.with_suffix(".json"))
        except BaseException:
            output.unlink(missing_ok=True)
            tensor_published = False
            raise
        return report
    finally:
        tensor_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)
        if tensor_published and not output.with_suffix(".json").exists():
            output.unlink(missing_ok=True)


def describe_gemma3_l3_l4_phase_graph_spectral(
    *,
    neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
    minimum_coherence: float = DEFAULT_MINIMUM_COHERENCE,
    top_pair_count: int = DEFAULT_TOP_PAIR_COUNT,
) -> dict[str, object]:
    """Describe this post-hoc rung without opening its source artifact."""

    protocol = _protocol(
        neighbor_count=neighbor_count,
        minimum_coherence=minimum_coherence,
        top_pair_count=top_pair_count,
    )
    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "source_tensor_file_sha256": DEFAULT_SOURCE_FILE_SHA256,
        "source_report_sha256": DEFAULT_SOURCE_REPORT_SHA256,
        "protocol": protocol,
        "source_artifact_loaded": False,
        "model_loaded": False,
        "scientific_scope": (
            "posthoc_prompt_free_phase_and_graph_fourier_structural_"
            "diagnostic"
        ),
        "new_information": (
            "phase_lead_lag_quadrature_weighted_graph_edges_and_graph_"
            "fourier_bases"
        ),
        "compression_claim": False,
        "speed_claim": False,
        "semantic_cluster_claim": False,
    }


def analyze_gemma3_l3_l4_phase_graph_spectral(
    *,
    source_path: Path | str = DEFAULT_SOURCE,
    source_file_sha256: str = DEFAULT_SOURCE_FILE_SHA256,
    source_report_sha256: str = DEFAULT_SOURCE_REPORT_SHA256,
    output: Path | str = DEFAULT_OUTPUT,
    neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
    minimum_coherence: float = DEFAULT_MINIMUM_COHERENCE,
    top_pair_count: int = DEFAULT_TOP_PAIR_COUNT,
) -> dict[str, object]:
    """Authenticate, analyze, and publish the frozen spectral map."""

    destination = _validate_output(output)
    source = _load_authenticated_source(
        source_path,
        expected_file_sha256=source_file_sha256,
        expected_report_sha256=source_report_sha256,
    )
    response_items = (
        ("finite", source.mapping.finite),
        *tuple(
            (response.label, response)
            for response in source.mapping.symmetric_responses
        ),
    )
    expected_labels = (
        "finite",
        "local_fraction_sigma",
        "operating_1_sigma",
    )
    if tuple(label for label, _ in response_items) != expected_labels:
        raise ValueError("phase-graph source response labels differ")
    analyses = {
        label: analyze_phase_graph_spectral_response(
            response,
            neighbor_count=neighbor_count,
            minimum_coherence=minimum_coherence,
            top_pair_count=top_pair_count,
        )
        for label, response in response_items
    }
    comparisons = _cross_response_comparisons(analyses)
    protocol = _protocol(
        neighbor_count=neighbor_count,
        minimum_coherence=minimum_coherence,
        top_pair_count=top_pair_count,
    )
    source_receipt = {
        "tensor_file_sha256": source.tensor_file_sha256,
        "report_sha256": source.report_sha256,
        "spectral_mapping_artifact_sha256": (
            source.mapping.artifact_sha256
        ),
    }
    logical_payload = _logical_payload(
        source=source_receipt,
        protocol=protocol,
        analyses=analyses,
        comparisons=comparisons,
    )
    logical_sha256 = _json_sha256(
        logical_payload,
        domain=_ARTIFACT_DOMAIN,
    )
    artifact = {
        **logical_payload,
        "analyses": {
            label: value.state_dict()
            for label, value in analyses.items()
        },
        "artifact_sha256": logical_sha256,
    }
    report_without_hash = {
        **logical_payload,
        "analyses": {
            label: value.metadata()
            for label, value in analyses.items()
        },
        "artifact_sha256": logical_sha256,
        "interpretation": {
            "legacy_spectral_similarity_already_phase_aware": True,
            "legacy_phase_direction_limit": (
                "cosine_relative_phase_is_even_and_cannot_distinguish_"
                "lead_from_lag"
            ),
            "new_directional_information": (
                "antisymmetric_sine_relative_phase_quadrature"
            ),
            "graph_fourier_definition": (
                "laplacian_eigenvectors_over_source_mode_connectivity"
            ),
            "graph_scope": (
                "globally_pooled_source_response_similarity_graph_not_"
                "directed_l3_to_l4_transfer_graph"
            ),
            "edge_phase_scope": (
                "pooled_across_origin_temporal_frequency_and_target_mode_"
                "not_frequency_resolved_delay"
            ),
            "response_gain_scope": (
                "supported_rows_are_normalized;raw_norm_diagnostics_retained_"
                "but_no_coupling_strength_claim"
            ),
            "rank_scope": (
                "tied_eigenspace_complete_low_laplacian_frequency_prefix_"
                "index_not_effective_rank_or_expert_count"
            ),
            "phase_blind_control": (
                "per_bin_spectral_magnitude_cosine_graph_with_identical_"
                "projected_signal"
            ),
            "not_proven": (
                "semantic mode meaning, causal identification, natural_"
                "prompt fidelity, executor improvement, compression, or speed"
            ),
        },
    }
    return _publish(
        output=destination,
        artifact=artifact,
        report_without_hash=report_without_hash,
    )


@dataclass(frozen=True, slots=True)
class LoadedGemma3PhaseGraphSpectralArtifact:
    state: Mapping[str, object]
    report: Mapping[str, object]
    analyses: Mapping[str, PhaseGraphSpectralAnalysis]
    artifact_sha256: str
    tensor_file_sha256: str
    report_sha256: str


def load_gemma3_l3_l4_phase_graph_spectral_artifact(
    path: Path | str,
    *,
    expected_artifact_sha256: str,
    expected_tensor_file_sha256: str,
    expected_report_sha256: str,
) -> LoadedGemma3PhaseGraphSpectralArtifact:
    """Strictly load and authenticate one phase-graph publication."""

    supplied_source = Path(path).expanduser()
    if supplied_source.is_symlink():
        raise ValueError("phase-graph artifact must not be a symlink")
    source = supplied_source.resolve()
    trusted_artifact = _require_sha256(
        expected_artifact_sha256,
        label="expected phase-graph artifact",
    )
    trusted_tensor = _require_sha256(
        expected_tensor_file_sha256,
        label="expected phase-graph tensor file",
    )
    trusted_report = _require_sha256(
        expected_report_sha256,
        label="expected phase-graph report",
    )
    if (
        source.suffix != ".pt"
        or not source.is_file()
        or source.is_symlink()
    ):
        raise ValueError("phase-graph artifact must be a regular .pt file")
    tensor_sha256 = _file_sha256(source)
    if tensor_sha256 != trusted_tensor:
        raise ValueError("phase-graph tensor file trust anchor differs")
    report = _read_json(
        source.with_suffix(".json"),
        label="phase-graph report",
    )
    supplied_report = _require_sha256(
        report.get("report_sha256"),
        label="phase-graph report SHA-256",
    )
    report_without_hash = dict(report)
    report_without_hash.pop("report_sha256")
    computed_report = _json_sha256(
        report_without_hash,
        domain=_REPORT_DOMAIN,
    )
    manifest = _mapping(report.get("artifact"), label="phase-graph manifest")
    if (
        supplied_report != computed_report
        or supplied_report != trusted_report
        or manifest.get("tensor_file_sha256") != tensor_sha256
        or report.get("artifact_sha256") != trusted_artifact
    ):
        raise ValueError("phase-graph report trust anchor differs")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    state = _mapping(raw, label="phase-graph artifact")
    _strict_keys(
        state,
        expected={
            "schema",
            "format_version",
            "source",
            "protocol",
            "analysis_artifact_sha256s",
            "cross_response_comparisons",
            "safety",
            "analyses",
            "artifact_sha256",
        },
        label="phase-graph artifact",
    )
    if (
        state["schema"] != _SCHEMA
        or state["format_version"] != _FORMAT_VERSION
        or state["artifact_sha256"] != trusted_artifact
    ):
        raise ValueError("phase-graph artifact header differs")
    analysis_states = _mapping(
        state["analyses"],
        label="phase-graph analyses",
    )
    analyses = {
        label: PhaseGraphSpectralAnalysis.from_state_dict(value)
        for label, value in analysis_states.items()
    }
    logical = _logical_payload(
        source=_mapping(state["source"], label="phase-graph source receipt"),
        protocol=_mapping(state["protocol"], label="phase-graph protocol"),
        analyses=analyses,
        comparisons=tuple(state["cross_response_comparisons"]),
    )
    recomputed_artifact = _json_sha256(logical, domain=_ARTIFACT_DOMAIN)
    expected_analysis_metadata = {
        label: analysis.metadata()
        for label, analysis in analyses.items()
    }
    if (
        recomputed_artifact != trusted_artifact
        or state["analysis_artifact_sha256s"]
        != logical["analysis_artifact_sha256s"]
        or _canonical_json_bytes(state["safety"])
        != _canonical_json_bytes(logical["safety"])
        or report.get("analysis_artifact_sha256s")
        != logical["analysis_artifact_sha256s"]
        or _canonical_json_bytes(report.get("analyses"))
        != _canonical_json_bytes(expected_analysis_metadata)
        or report.get("schema") != logical["schema"]
        or report.get("format_version") != logical["format_version"]
        or _canonical_json_bytes(report.get("source"))
        != _canonical_json_bytes(logical["source"])
        or _canonical_json_bytes(report.get("protocol"))
        != _canonical_json_bytes(logical["protocol"])
        or _canonical_json_bytes(report.get("cross_response_comparisons"))
        != _canonical_json_bytes(logical["cross_response_comparisons"])
        or _canonical_json_bytes(report.get("safety"))
        != _canonical_json_bytes(logical["safety"])
    ):
        raise ValueError("phase-graph logical artifact differs")
    return LoadedGemma3PhaseGraphSpectralArtifact(
        state=dict(state),
        report=report,
        analyses=analyses,
        artifact_sha256=recomputed_artifact,
        tensor_file_sha256=tensor_sha256,
        report_sha256=computed_report,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase-aware and graph Fourier analysis of Gemma modes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser("describe")
    analyze = subparsers.add_parser("analyze")
    for command in (describe, analyze):
        command.add_argument(
            "--neighbor-count",
            type=int,
            default=DEFAULT_NEIGHBOR_COUNT,
        )
        command.add_argument(
            "--minimum-coherence",
            type=float,
            default=DEFAULT_MINIMUM_COHERENCE,
        )
        command.add_argument(
            "--top-pair-count",
            type=int,
            default=DEFAULT_TOP_PAIR_COUNT,
        )
    analyze.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    analyze.add_argument(
        "--source-file-sha256",
        default=DEFAULT_SOURCE_FILE_SHA256,
    )
    analyze.add_argument(
        "--source-report-sha256",
        default=DEFAULT_SOURCE_REPORT_SHA256,
    )
    analyze.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    common = {
        "neighbor_count": arguments.neighbor_count,
        "minimum_coherence": arguments.minimum_coherence,
        "top_pair_count": arguments.top_pair_count,
    }
    if arguments.command == "describe":
        result = describe_gemma3_l3_l4_phase_graph_spectral(**common)
    else:
        result = analyze_gemma3_l3_l4_phase_graph_spectral(
            source_path=arguments.source,
            source_file_sha256=arguments.source_file_sha256,
            source_report_sha256=arguments.source_report_sha256,
            output=arguments.output,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
