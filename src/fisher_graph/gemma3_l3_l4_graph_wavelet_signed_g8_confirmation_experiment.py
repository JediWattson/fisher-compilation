"""Fresh no-refit confirmation of the frozen Gemma signed-g8 candidate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import torch

from .conditional_spectral_generator import (
    fit_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    DEFAULT_INTERIOR_ARTIFACT,
    DEFAULT_INTERIOR_ARTIFACT_SHA256,
    DEFAULT_INTERIOR_REPORT_SHA256,
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    _file_sha256,
    _reserve_outputs,
    _stage_json,
    load_gemma3_spectral_source,
)
from .gemma3_l3_l4_graph_wavelet_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_WAVELET_ARTIFACT,
    _tensor_sha256,
    load_gemma3_graph_wavelet_candidate,
)
from .gemma3_l3_l4_graph_wavelet_grouped_comparison_experiment import (
    _energy_ordered_basis,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_candidate import (
    DEFAULT_FROZEN_ARTIFACT_SHA256,
    DEFAULT_FROZEN_REPORT_SHA256,
    DEFAULT_FROZEN_TENSOR_FILE_SHA256,
    DEFAULT_OUTPUT as DEFAULT_CANDIDATE_ARTIFACT,
    SOURCE_BASIS_KIND,
    TARGET_SOURCE_RANK,
    load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle import (
    DEFAULT_NULL_BUNDLE_ARTIFACT_SHA256,
    DEFAULT_NULL_BUNDLE_FILE_SHA256,
    DEFAULT_NULL_BUNDLE_REPORT_SHA256,
    DEFAULT_OUTPUT as DEFAULT_NULL_BUNDLE,
    FRESH_CONFIRMATION_ORIGINS,
    _reconstruct_context,
    load_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle,
    replay_gemma3_l3_l4_graph_wavelet_signed_g8_null_plans,
)
from .gemma3_l3_l4_graph_wavelet_supermode_experiment import (
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_ARTIFACT_SHA256,
    DEFAULT_PARENT_REPORT_SHA256,
    DEFAULT_PARENT_TENSOR_FILE_SHA256,
)
from .graph_wavelet_structural_confirmation import (
    evaluate_graph_wavelet_structural_confirmation,
)


__all__ = [
    "DEFAULT_FRESH_ARTIFACT",
    "DEFAULT_FRESH_ARTIFACT_SHA256",
    "DEFAULT_FRESH_REPORT_SHA256",
    "DEFAULT_OUTPUT",
    "run_gemma3_l3_l4_graph_wavelet_signed_g8_confirmation",
    "main",
]


DEFAULT_FRESH_ARTIFACT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-spectral-map-confirmation-"
    "origins-12-28-36-v1.pt"
)
DEFAULT_FRESH_ARTIFACT_SHA256 = (
    "d6e666644f7268fb24eaa2c3b8b7862caca786ab3b33e5111379c42a6e60a28c"
)
DEFAULT_FRESH_REPORT_SHA256 = (
    "a2b3da07006c2934922f53777601d0bdc2fd5210bd33d92b6ad5223747a3bde0"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "confirmation-origins-12-28-36-v1.json"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_graph_wavelet_signed_g8_confirmation"
)
_FORMAT_VERSION = 1
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-signed-g8-confirmation-report:v1\0"
)
_EXPECTED_SIGNED_GFA_PLAN_SHA256 = (
    "6d85ce80050393ef7cc50ce166f96a14ab01d7856d25e95044262e3b22ba7a94"
)
_EXPECTED_SIGNED_GFA_BASIS_SHA256 = (
    "32f16efb11d80d7be672f5e12214f3de90201652b8cc548c403c0e4906679152"
)
_EXPECTED_GLOBAL_SVD_PLAN_SHA256 = (
    "e0d0a8cb00d391bfcee3197f1633dd88e631a7d30eb48965f56736f59cc83952"
)
_EXPECTED_GLOBAL_SVD_BASIS_SHA256 = (
    "f76d9c897409a6248b88d3b7072e6b6fcb4f7aa263c1053bb996ae6036df97e4"
)
_LINEAGE_FIELDS = (
    "hierarchy_artifact_sha256",
    "source_model_sha256",
    "base_artifact_file_sha256",
    "base_scientific_payload_sha256",
    "refit_artifact_file_sha256",
    "refit_scientific_payload_sha256",
    "layer3_factor_sha256",
    "layer4_factor_sha256",
    "residual_width",
    "upstream_edge_rank",
)
_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_fresh_response_values": False,
    "contains_fit_response_values": False,
    "contains_compiled_plan_tensors": False,
    "contains_scalar_confirmation_metrics": True,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_REPORT_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _same_metadata(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _validate_fresh_lineage(
    fit_source: object,
    fresh_source: object,
) -> None:
    fit_binding = getattr(fit_source, "binding")
    fresh_binding = getattr(fresh_source, "binding")
    if any(
        fit_binding.get(field) != fresh_binding.get(field)
        for field in _LINEAGE_FIELDS
    ):
        raise ValueError("fresh response hierarchy lineage differs")
    if not _same_metadata(
        getattr(fit_source, "model"),
        getattr(fresh_source, "model"),
    ):
        raise ValueError("fresh response model lineage differs")
    fit_scales = getattr(fit_source, "source_mode_standard_deviations")
    fresh_scales = getattr(fresh_source, "source_mode_standard_deviations")
    mapping = getattr(fresh_source, "mapping")
    protocol = getattr(fresh_source, "protocol")
    if (
        not torch.equal(fit_scales, fresh_scales)
        or mapping.impulse_logical_positions != FRESH_CONFIRMATION_ORIGINS
        or mapping.max_lag != 31
        or mapping.fft_length != 64
        or mapping.source_mode_indices != tuple(range(64))
        or protocol.get("sequence_length") != 72
        or protocol.get("new_prompt_text_loaded") is not False
        or protocol.get("new_token_ids_loaded") is not False
        or protocol.get("source_scope") != "factorized_refit"
    ):
        raise ValueError("fresh response ABI differs from preregistration")


def _reference_plans(source: object, parent: object):
    context, _ = _reconstruct_context(source, parent)  # type: ignore[arg-type]
    signed_basis, _ = _energy_ordered_basis(
        context.graph.signed_eigenvectors,
        context.weighted_fit,
    )
    signed_basis = signed_basis[:, :TARGET_SOURCE_RANK].contiguous()
    signed_plan = fit_conditional_spectral_generator_with_source_basis(
        context.fit_kernels,
        context.source_scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        signed_basis,
        context.target_modes,
        source_basis_kind="fixed_orthonormal_control",
        source_basis_fit_weighted_kernels_sha256=(
            context.graph.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=context.response_binding_sha256,
        input_transform="standardized_linear",
        fft_length=context.fft_length,
    )
    global_fit = fit_conditional_spectral_generator(
        context.fit_kernels,
        context.source_scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        TARGET_SOURCE_RANK,
        context.target_modes,
        response_binding_sha256=context.response_binding_sha256,
        input_transform="standardized_linear",
        fft_length=context.fft_length,
    )
    global_plan = fit_conditional_spectral_generator_with_source_basis(
        context.fit_kernels,
        context.source_scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        global_fit.source_basis,
        context.target_modes,
        source_basis_kind="fixed_orthonormal_control",
        source_basis_fit_weighted_kernels_sha256=(
            context.graph.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=context.response_binding_sha256,
        input_transform="standardized_linear",
        fft_length=context.fft_length,
    )
    if (
        signed_plan.artifact_sha256 != _EXPECTED_SIGNED_GFA_PLAN_SHA256
        or _tensor_sha256(signed_basis)
        != _EXPECTED_SIGNED_GFA_BASIS_SHA256
        or global_plan.artifact_sha256 != _EXPECTED_GLOBAL_SVD_PLAN_SHA256
        or _tensor_sha256(global_plan.source_basis)
        != _EXPECTED_GLOBAL_SVD_BASIS_SHA256
    ):
        raise ValueError("frozen signed-GFA or SVD reference differs")
    return signed_plan, global_plan


def _validate_control_replay(
    bundle: object,
    receipts: Sequence[Mapping[str, object]],
) -> None:
    expected = getattr(bundle, "control_plan_receipts")
    if len(receipts) != len(expected) or any(
        not _same_metadata(actual, frozen)
        for actual, frozen in zip(receipts, expected, strict=True)
    ):
        raise ValueError("control plan replay differs from frozen null bundle")


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or ".local-runs" not in destination.parts:
        raise ValueError("confirmation output must be JSON under .local-runs")
    return destination


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    reservation = _reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = _json_sha256(report)
        stage = _stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_graph_wavelet_signed_g8_confirmation(
    *,
    fit_source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = DEFAULT_CANDIDATE_ARTIFACT,
    null_bundle_path: Path | str = DEFAULT_NULL_BUNDLE,
    fresh_artifact_path: Path | str = DEFAULT_FRESH_ARTIFACT,
    fresh_artifact_sha256: str = DEFAULT_FRESH_ARTIFACT_SHA256,
    fresh_report_sha256: str = DEFAULT_FRESH_REPORT_SHA256,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Replay every frozen plan, then read and score the fresh responses."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite confirmation report")
    fit_source = load_gemma3_spectral_source(
        fit_source_artifact_path,
        expected_file_sha256=DEFAULT_INTERIOR_ARTIFACT_SHA256,
        expected_report_sha256=DEFAULT_INTERIOR_REPORT_SHA256,
        expected_origins=INTERIOR_ORIGINS,
    )
    parent = load_gemma3_graph_wavelet_candidate(
        parent_artifact_path,
        expected_artifact_sha256=DEFAULT_PARENT_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_PARENT_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_PARENT_REPORT_SHA256,
    )
    candidate = load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        candidate_artifact_path,
        expected_artifact_sha256=DEFAULT_FROZEN_ARTIFACT_SHA256,
        expected_tensor_file_sha256=DEFAULT_FROZEN_TENSOR_FILE_SHA256,
        expected_report_sha256=DEFAULT_FROZEN_REPORT_SHA256,
    )
    bundle = load_gemma3_l3_l4_graph_wavelet_signed_g8_null_bundle(
        null_bundle_path,
        expected_file_sha256=DEFAULT_NULL_BUNDLE_FILE_SHA256,
        expected_report_sha256=DEFAULT_NULL_BUNDLE_REPORT_SHA256,
        expected_artifact_sha256=DEFAULT_NULL_BUNDLE_ARTIFACT_SHA256,
    )
    panel, control_plans, control_receipts = (
        replay_gemma3_l3_l4_graph_wavelet_signed_g8_null_plans(
            fit_source,
            parent,
            candidate,
        )
    )
    if panel.artifact_sha256 != bundle.panel.artifact_sha256:
        raise ValueError("replayed null panel differs from frozen bundle")
    _validate_control_replay(bundle, control_receipts)
    signed_gfa_plan, global_svd_plan = _reference_plans(fit_source, parent)

    # This is deliberately the first fresh-response load in the runner.
    fresh_source = load_gemma3_spectral_source(
        fresh_artifact_path,
        expected_file_sha256=fresh_artifact_sha256,
        expected_report_sha256=fresh_report_sha256,
        expected_origins=FRESH_CONFIRMATION_ORIGINS,
    )
    _validate_fresh_lineage(fit_source, fresh_source)
    local = fresh_source.mapping.symmetric_by_label["local_fraction_sigma"]
    native_groups = tuple(
        tuple(group) for group in bundle.panel.native_groups
    )
    confirmation = evaluate_graph_wavelet_structural_confirmation(
        panel=bundle.panel,
        native_plan=candidate.plan,
        control_plans=control_plans,
        signed_gfa_reference_plan=signed_gfa_plan,
        global_svd_ceiling_plan=global_svd_plan,
        fresh_central_responses=local.impulse_responses,
        source_scales=fresh_source.source_mode_standard_deviations,
        origins=FRESH_CONFIRMATION_ORIGINS,
        native_groups=native_groups,
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "lineage": {
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "null_bundle_artifact_sha256": bundle.artifact_sha256,
            "null_panel_artifact_sha256": bundle.panel.artifact_sha256,
            "fresh_tensor_file_sha256": fresh_source.file_sha256,
            "fresh_report_file_sha256": fresh_source.report_file_sha256,
            "fresh_report_payload_sha256": (
                fresh_source.report_payload_sha256
            ),
            "fresh_mapping_artifact_sha256": (
                fresh_source.mapping.artifact_sha256
            ),
            "fresh_response_artifact_sha256": local.artifact_sha256,
            "fit_response_binding_sha256": (
                candidate.plan.response_binding_sha256
            ),
            "fresh_response_is_separately_measured": True,
        },
        "protocol": {
            "origins": FRESH_CONFIRMATION_ORIGINS,
            "sequence_length": 72,
            "max_lag": 31,
            "fft_length": 64,
            "fit_origins": FIT_ORIGINS,
            "fit_recomputed_after_fresh_load": False,
            "candidate_refit_on_fresh_values": False,
            "control_refit_on_fresh_values": False,
            "prompt_text_loaded": False,
            "tokenizer_loaded": False,
            "global_svd_role": "descriptive_ceiling_only",
        },
        "confirmation": confirmation.metadata(),
        "resource_accounting": {
            "model_load_count_for_confirmation_scoring": 0,
            "tokenizer_load_count": 0,
            "fresh_model_forward_count_for_confirmation_scoring": 0,
            "fresh_mapping_function_evaluation_count": (
                fresh_source.mapping.function_evaluation_count
            ),
            "candidate_stored_coefficient_count": (
                candidate.plan.stored_coefficient_count
            ),
            "candidate_prepared_storage_bytes": (
                candidate.plan.accounting().prepared_storage_bytes
            ),
            "dense_fit_knot_coefficient_count": (
                candidate.plan.accounting().dense_fit_knot_coefficient_count
            ),
            "coefficient_fraction_of_dense_fit_knots": (
                candidate.plan.accounting().coefficient_fraction_of_dense_fit_knots
            ),
            "whole_model_parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
        },
        "scientific_status": {
            "fresh_prompt_free_structural_confirmation_complete": True,
            "all_primary_structural_gates_passed": confirmation.passed,
            "natural_prompt_fidelity_measured": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "candidate_serving_authorized": False,
        },
        "artifact": {
            "file": str(destination),
            "committable": False,
        },
        "safety": _SAFETY,
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen signed-g8 fresh structural confirmation",
    )
    parser.add_argument("--fit-source-artifact", default=DEFAULT_INTERIOR_ARTIFACT)
    parser.add_argument("--parent-artifact", default=DEFAULT_PARENT_ARTIFACT)
    parser.add_argument("--candidate-artifact", default=DEFAULT_CANDIDATE_ARTIFACT)
    parser.add_argument("--null-bundle", default=DEFAULT_NULL_BUNDLE)
    parser.add_argument("--fresh-artifact", default=DEFAULT_FRESH_ARTIFACT)
    parser.add_argument(
        "--fresh-artifact-sha256",
        default=DEFAULT_FRESH_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--fresh-report-sha256",
        default=DEFAULT_FRESH_REPORT_SHA256,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_graph_wavelet_signed_g8_confirmation(
        fit_source_artifact_path=arguments.fit_source_artifact,
        parent_artifact_path=arguments.parent_artifact,
        candidate_artifact_path=arguments.candidate_artifact,
        null_bundle_path=arguments.null_bundle,
        fresh_artifact_path=arguments.fresh_artifact,
        fresh_artifact_sha256=arguments.fresh_artifact_sha256,
        fresh_report_sha256=arguments.fresh_report_sha256,
        output=arguments.output,
    )
    summary = {
        "report_sha256": report["report_sha256"],
        "artifact": report["artifact"],
        "scientific_status": report["scientific_status"],
        "gate_results": report["confirmation"]["gate_results"],  # type: ignore[index]
        "native": report["confirmation"]["native"]["pooled"],  # type: ignore[index]
        "signed_gfa": report["confirmation"]["signed_gfa_reference"]["pooled"],  # type: ignore[index]
        "global_svd": report["confirmation"]["global_svd_ceiling"]["pooled"],  # type: ignore[index]
        "random_null": report["confirmation"]["random_null_statistics"],  # type: ignore[index]
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
