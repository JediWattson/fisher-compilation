"""Adaptive same-A expansion of the complete-H4 token-Fisher tail ladder.

The completed v1 A16 diagnostic established a failing rank-64 lower bracket
and an exact, passing rank-320 sentinel.  This follow-up authenticates that
commit-marker report before fixing the expanded rank grid
``(64, 96, 128, 160, 192, 256, 320)``.  It then repeats the same held-family
fit/order protocol and finite one-pass executions, including rank 64 and rank
320 as exact overlap controls.

The grid was chosen after inspecting v1.  This is therefore adaptive,
same-Calibration-A, truth-leaking hypothesis evidence only.  It is neither a
fresh confirmation nor a serving, compression, speed, or deployment result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as v1
from .complete_h4_tail_token_fisher import fit_complete_h4_tail_held_family
from .gemma3_l3_l4_complete_h4_one_pass_transfer import _load_committed_basis
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "ADAPTIVE_PARENT_REPORT_FILE_SHA256",
    "ADAPTIVE_PARENT_REPORT_SHA256",
    "DEFAULT_ADAPTIVE_PARENT_REPORT",
    "DEFAULT_MATERIALIZATION_REPORT",
    "DEFAULT_OUTPUT",
    "DEFAULT_TRANSFER_REPORT",
    "EXPANDED_TAIL_RANKS",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v1.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v1.DEFAULT_TRANSFER_REPORT
DEFAULT_ADAPTIVE_PARENT_REPORT = v1.DEFAULT_OUTPUT
DEFAULT_OUTPUT = v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-lofo-adaptive-expanded-ladder-a-fit16-dev-v2.json"
)

ADAPTIVE_PARENT_REPORT_FILE_SHA256 = (
    "3cb0c95f5848975cfc3b9e93c16c3837dc3282f667b04d90c9c899bf241e8f6c"
)
ADAPTIVE_PARENT_REPORT_SHA256 = (
    "d52a89529635c6c6fb6bbaa2ccb48a7a3fdbadfa395ee2ccfdbe79a7c45fea67"
)

EXPANDED_TAIL_RANKS = (64, 96, 128, 160, 192, 256, 320)
_PARENT_TAIL_RANKS = (8, 16, 32, 64, 320)
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_lofo."
    "adaptive_expanded.v2"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-tail-token-fisher-lofo-adaptive-expanded:v2\0"
)
_OVERLAP_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-tail-token-fisher-adaptive-overlap:v1\0"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_adaptive_parent(path: Path | str) -> dict[str, object]:
    """Strict-load and semantically authenticate the completed v1 result."""

    report = v1._load_pinned_report(
        path,
        expected_file_sha256=ADAPTIVE_PARENT_REPORT_FILE_SHA256,
        expected_report_sha256=ADAPTIVE_PARENT_REPORT_SHA256,
        label="adaptive v1 token-Fisher parent",
    )
    protocol = _mapping(report.get("protocol"), label="adaptive parent protocol")
    binding = _mapping(
        report.get("input_binding"), label="adaptive parent input binding"
    )
    science = _mapping(
        report.get("scientific_status"), label="adaptive parent scientific status"
    )
    safety = _mapping(report.get("safety"), label="adaptive parent safety")
    pass_by_rank = _mapping(
        report.get("fidelity_and_geometry_pass_by_rank"),
        label="adaptive parent rank decisions",
    )
    expected_rank_decisions = {
        "8": False,
        "16": False,
        "32": False,
        "64": False,
        "320": True,
    }
    if (
        report.get("schema") != v1._SCHEMA
        or report.get("classification")
        != "tail_endpoint_fisher_finite_ladder_not_supported"
        or report.get("passed") is not False
        or tuple(protocol.get("tail_ranks", ())) != _PARENT_TAIL_RANKS
        or report.get(
            "smallest_tail_rank_at_most_64_clearing_established_gates"
        )
        is not None
        or dict(pass_by_rank) != expected_rank_decisions
        or binding.get("materialization_report_file_sha256")
        != v1.MATERIALIZATION_REPORT_FILE_SHA256
        or binding.get("materialization_report_sha256")
        != v1.MATERIALIZATION_REPORT_SHA256
        or binding.get("transfer_report_file_sha256")
        != v1.TRANSFER_REPORT_FILE_SHA256
        or binding.get("transfer_report_sha256") != v1.TRANSFER_REPORT_SHA256
        or science.get("same_a_truth_leaking_hypothesis_use_only") is not True
        or science.get("fresh_confirmation_panel_opened") is not False
        or science.get("candidate_serving_authorized") is not False
        or science.get("compression_claim") is not False
        or safety.get("contains_prompt_text") is not False
        or safety.get("contains_token_ids") is not False
        or safety.get("contains_only_hashes_counts_and_scalar_metrics") is not True
    ):
        raise ValueError("adaptive v1 token-Fisher parent semantics differ")
    raw_observations = report.get("finite_observation_receipts")
    if not isinstance(raw_observations, list):
        raise ValueError("adaptive parent finite observations differ")
    observations: list[Mapping[str, object]] = []
    for raw in raw_observations:
        observations.append(_mapping(raw, label="adaptive parent observation"))
    observation_set_sha256 = v1._finite_observation_set_sha256(
        observations,
        expected_example_count=v1._EXPECTED_EXAMPLES,
        ranks=_PARENT_TAIL_RANKS,
    )
    if observation_set_sha256 != report.get("finite_observation_set_sha256"):
        raise ValueError("adaptive parent observation-set receipt differs")
    return report


def _transfer_receipts(transfer: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    comparison = _mapping(transfer.get("comparison"), label="transfer comparison")
    if comparison.get("pass_pattern") != "11100000":
        raise ValueError("pinned transfer pass pattern differs")
    raw_receipts = transfer.get("prompt_receipts")
    if not isinstance(raw_receipts, list):
        raise ValueError("pinned transfer prompt receipts differ")
    receipts: dict[str, Mapping[str, object]] = {}
    for raw in raw_receipts:
        receipt = _mapping(raw, label="pinned transfer prompt receipt")
        example_id = v1._identifier(
            receipt.get("example_id"), label="transfer receipt example_id"
        )
        if example_id in receipts:
            raise ValueError("pinned transfer has duplicate prompt receipts")
        receipts[example_id] = receipt
    if len(receipts) != v1._EXPECTED_EXAMPLES:
        raise ValueError("pinned transfer prompt receipt count differs")
    return receipts


def _overlap_receipt(
    parent: Mapping[str, object],
    observations: Sequence[Mapping[str, object]],
) -> str:
    """Require exact rank-64/rank-320 reproduction of the adaptive parent."""

    parent_raw = parent.get("finite_observation_receipts")
    if not isinstance(parent_raw, list):
        raise ValueError("adaptive parent finite observations differ")
    expected: dict[tuple[str, int], str] = {}
    for raw in parent_raw:
        row = _mapping(raw, label="adaptive parent overlap observation")
        rank = row.get("rank")
        if rank not in (64, 320):
            continue
        example_id = v1._identifier(
            row.get("example_id"), label="adaptive parent overlap example_id"
        )
        receipt = row.get("observation_sha256")
        if not isinstance(receipt, str):
            raise ValueError("adaptive parent overlap receipt differs")
        identity = (example_id, int(rank))
        if identity in expected:
            raise ValueError("adaptive parent overlap grid has a duplicate")
        expected[identity] = receipt
    actual: dict[tuple[str, int], str] = {}
    for row in observations:
        rank = row.get("rank")
        if rank not in (64, 320):
            continue
        example_id = v1._identifier(
            row.get("example_id"), label="expanded overlap example_id"
        )
        receipt = row.get("observation_sha256")
        if not isinstance(receipt, str):
            raise ValueError("expanded overlap receipt differs")
        identity = (example_id, int(rank))
        if identity in actual:
            raise ValueError("expanded overlap grid has a duplicate")
        actual[identity] = receipt
    if len(expected) != 2 * v1._EXPECTED_EXAMPLES or actual != expected:
        raise RuntimeError("expanded rank-64/rank-320 overlap differs from v1")
    return v1._domain_sha256(
        tuple((example_id, rank, actual[(example_id, rank)])
              for example_id, rank in sorted(actual)),
        domain=_OVERLAP_DOMAIN,
    )


def _expanded_classification(
    fidelity_and_geometry_pass_by_rank: Mapping[int, bool],
    *,
    integrity_gates_passed: bool,
) -> tuple[int | None, str]:
    """Return the smallest sub-sentinel success and explicit adaptive label."""

    ranks = v1._validated_tail_ranks(EXPANDED_TAIL_RANKS)
    if set(fidelity_and_geometry_pass_by_rank) != set(ranks):
        raise ValueError("expanded classification rank grid differs")
    smallest = next(
        (
            rank
            for rank in ranks
            if rank < v1._D_RANK
            and fidelity_and_geometry_pass_by_rank[rank] is True
        ),
        None,
    )
    if smallest is None:
        return None, "adaptive_same_a_no_tail_rank_below_320_cleared"
    if not integrity_gates_passed:
        return (
            smallest,
            f"adaptive_same_a_smallest_tail_rank_{smallest}_cleared_but_"
            "integrity_gate_failed",
        )
    return (
        smallest,
        f"adaptive_same_a_smallest_tail_rank_{smallest}_cleared_established_gates",
    )


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(report, domain=_REPORT_DOMAIN)
        stage = frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": v1._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic(
    *,
    adaptive_parent_report_path: Path | str = DEFAULT_ADAPTIVE_PARENT_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the pinned adaptive A16 expanded finite tail-rank ladder."""

    ranks = v1._validated_tail_ranks(EXPANDED_TAIL_RANKS)
    destination = v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite expanded tail token-Fisher report")
    parent = _load_adaptive_parent(adaptive_parent_report_path)
    materialization = v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=v1.MATERIALIZATION_REPORT_SHA256,
        label="rank320 materialization",
    )
    transfer = v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=v1.TRANSFER_REPORT_SHA256,
        label="rank320 transfer",
    )
    transfer_receipts = _transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=v1.MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, endpoint_resources = v1._collect_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if len(traces) != v1._EXPECTED_EXAMPLES or len(families) != v1._EXPECTED_FAMILIES:
            raise RuntimeError("A16 expanded endpoint panel shape differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        (
            observations,
            finite_resources,
            behavioral_by_rank,
            geometry_by_rank,
        ) = v1._finite_observations(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            ranks=ranks,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    overlap_receipt_sha256 = _overlap_receipt(parent, observations)
    arms, secondary_gates = v1._summarize_observations(
        observations, ranks=ranks
    )
    finite_observation_set_sha256 = v1._finite_observation_set_sha256(
        observations, ranks=ranks
    )
    causality_passed = all(
        trace.maximum_future_gradient_abs == 0.0
        and trace.future_gradient_nonzero_count == 0
        for trace in traces
    )
    fidelity_and_geometry_pass_by_rank = {
        rank: (
            bool(geometry_by_rank[rank]["gates"]["passed"])
            and all(
                bool(behavioral_by_rank[rank][ledger]["gates"]["passed"])
                for ledger in (
                    "ordinary",
                    "complete_h4_support",
                    "graph_core",
                    "causal_tail",
                )
            )
        )
        for rank in ranks
    }
    k320_arm = next(row for row in arms if row["tail_rank"] == v1._D_RANK)
    integrity_gates = {
        "adaptive_parent_authenticated": True,
        "rank64_and_rank320_overlap_reproduced_exactly": True,
        "all_endpoint_token_vjps_have_zero_future_gradient": causality_passed,
        "k320_full_fitted_span_clears_established_fidelity_and_geometry_gates": (
            fidelity_and_geometry_pass_by_rank[v1._D_RANK]
        ),
        "k320_every_prompt_h4_bitwise_native": bool(
            k320_arm["every_prompt_h4_bitwise_native"]
        ),
        "k320_every_prompt_logits_bitwise_native": bool(
            k320_arm["every_prompt_logits_bitwise_native"]
        ),
    }
    smallest_passing_rank, classification = _expanded_classification(
        fidelity_and_geometry_pass_by_rank,
        integrity_gates_passed=all(integrity_gates.values()),
    )
    primary_gates = {
        **integrity_gates,
        "at_least_one_tail_rank_below_320_clears_all_established_fidelity_and_geometry_gates": (
            smallest_passing_rank is not None
        ),
    }
    resources = v1._build_resource_accounting(
        traces,
        endpoint_resources=endpoint_resources,
        finite_resources=finite_resources,
        ranks=ranks,
    )
    prompt_receipts = tuple(
        {
            **trace.endpoint.metadata(),
            "model_inputs_sha256": trace.model_inputs_sha256,
            "base_x4_sha256": trace.base_x4_sha256,
            "supervised_indices_sha256": trace.supervised_indices_sha256,
            "supervised_targets_sha256": trace.supervised_targets_sha256,
            "endpoint_support_indices_sha256": trace.endpoint_indices_sha256,
            "endpoint_support_targets_sha256": trace.endpoint_targets_sha256,
            "endpoint_support_supervised_token_count": trace.endpoint.supervised_tokens,
            "native_logits_sha256": trace.native_logits_sha256,
            "endpoint_vjp_artifact_sha256": trace.endpoint_vjp_artifact_sha256,
            "endpoint_execution_artifact_sha256": trace.endpoint_execution_artifact_sha256,
            "endpoint_provider_artifact_sha256": trace.endpoint_provider_artifact_sha256,
            "backward_call_count": trace.backward_call_count,
            "compensation_target_semantics": (
                "native_token_nll_minus_d320_endpoint_token_nll"
            ),
            "compensation_target_sign_used_in_fisher_q2_ordering": False,
            "maximum_future_gradient_abs": trace.maximum_future_gradient_abs,
            "future_gradient_nonzero_count": trace.future_gradient_nonzero_count,
            "causality_receipt_sha256": trace.causality_receipt_sha256,
        }
        for trace in traces
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_adaptive_hypothesis_use_only",
            "adaptive_parent_outcome_was_inspected_before_rank_grid_was_fixed": True,
            "adaptive_parent_lower_bracket": 64,
            "adaptive_parent_passing_sentinel": 320,
            "expanded_rank_grid_fixed_before_this_execution": True,
            "tail_ranks": ranks,
            "rank64_overlap_control": True,
            "rank320_full_span_sentinel": True,
            "split": "whole_family_leave_one_out_for_tail_basis_and_fisher_order_only",
            "frozen_d320_was_fit_on_all_a16_families": True,
            "end_to_end_candidate_is_family_disjoint": False,
            "frozen_supported_basis_rank": v1._D_RANK,
            "tail_width": v1._D_RANK,
            "tail_definition": "E=(I-P_D320)(native_H4-base_graph_H4)",
            "endpoint": "actual_cast_once_D320_one_pass_graph_execution",
            "basis_fit": "training_family_equal_unweighted_tail_covariance_full_complement",
            "order": "training_only_family_prompt_token_equal_endpoint_vjp_square",
            "compensation_target_semantics": (
                "native_token_nll_minus_d320_endpoint_token_nll"
            ),
            "compensation_target_sign_used_in_fisher_q2_ordering": False,
            "held_residual_or_gradient_used_for_fit_or_order": False,
            "endpoint_token_fisher_ledger": "complete_h4_support_803",
            "finite_shadow_ledgers": {
                "ordinary": 931,
                "complete_h4_support": 803,
                "graph_core": 790,
                "causal_tail": 13,
            },
            "finite_arm": "P_D320_R_plus_P_first_K_training_fisher_ordered_tail_E",
            "k320_uses_full_fitted_complement_span": True,
            "k320_exact_residual_provider_substitution": False,
        },
        "adaptive_parent_binding": {
            "file": str(adaptive_parent_report_path),
            "file_sha256": ADAPTIVE_PARENT_REPORT_FILE_SHA256,
            "report_sha256": ADAPTIVE_PARENT_REPORT_SHA256,
            "schema": parent.get("schema"),
            "classification": parent.get("classification"),
            "passed": parent.get("passed"),
            "tail_ranks": _PARENT_TAIL_RANKS,
            "rank64_passed": False,
            "rank320_passed": True,
            "overlap_observation_receipt_sha256": overlap_receipt_sha256,
        },
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": v1.MATERIALIZATION_REPORT_FILE_SHA256,
            "materialization_report_sha256": v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": v1.TRANSFER_REPORT_SHA256,
            "transfer_pass_pattern": "11100000",
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": prompt_receipts,
        "finite_ladder": arms,
        "established_behavioral_fidelity_by_rank": {
            str(rank): behavioral_by_rank[rank] for rank in ranks
        },
        "executed_cast_once_geometry_by_rank": {
            str(rank): geometry_by_rank[rank] for rank in ranks
        },
        "fidelity_and_geometry_pass_by_rank": {
            str(rank): fidelity_and_geometry_pass_by_rank[rank] for rank in ranks
        },
        "smallest_tail_rank_below_320_clearing_established_fidelity_and_geometry_gates": (
            smallest_passing_rank
        ),
        "finite_observation_receipts": tuple(observations),
        "finite_observation_set_sha256": finite_observation_set_sha256,
        "primary_gate_results": tuple(sorted(primary_gates.items())),
        "secondary_first_order_gate_results": tuple(sorted(secondary_gates.items())),
        "passed": all(primary_gates.values()),
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_adaptive_hypothesis_use_only": True,
            "adaptive_parent_outcome_was_inspected": True,
            "truth_leaking_native_held_prompt_residuals_instantiate_corrections": True,
            "tail_basis_and_fisher_order_family_disjoint_only": True,
            "frozen_d320_contains_same_a_held_family_information": True,
            "end_to_end_candidate_family_disjoint": False,
            "fresh_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "deployment_claim": False,
            "next_rung_only_if_subsentinel_rank_clears": (
                "freeze_selected_recipe_then_run_fresh_family_disjoint_confirmation"
            ),
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_activation_tensors": False,
            "contains_gradient_tensors": False,
            "contains_token_score_matrices": False,
            "contains_basis_coefficients": False,
            "contains_only_hashes_counts_and_scalar_metrics": True,
            "artifact_must_remain_outside_git": True,
        },
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authenticated adaptive A16 expanded token-Fisher tail-rank ladder."
        )
    )
    parser.add_argument(
        "--adaptive-parent-report",
        type=Path,
        default=DEFAULT_ADAPTIVE_PARENT_REPORT,
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument("--transfer-report", type=Path, default=DEFAULT_TRANSFER_REPORT)
    parser.add_argument("--basis-sidecar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic(
        adaptive_parent_report_path=args.adaptive_parent_report,
        materialization_report_path=args.materialization_report,
        transfer_report_path=args.transfer_report,
        basis_sidecar_path=args.basis_sidecar,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":
    main()
