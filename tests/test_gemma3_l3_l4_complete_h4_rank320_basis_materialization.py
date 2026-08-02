from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_rank320_basis_materialization as materialize,
)
from fisher_graph.gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionFitSequence,
)
from fisher_graph.gemma3_l3_l4_complete_h4_tail_informed_projection import (
    CompleteH4TailProjectionTrace,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _minimal_parent_report() -> dict[str, object]:
    receipts = [
        {
            "example_id": f"example-{index:02d}",
            "family_id": f"family-{index // 2:02d}",
            "pair": {"artifact_sha256": _digest(100 + index)},
            "fit_sequence": {"sequence_sha256": _digest(200 + index)},
        }
        for index in range(16)
    ]
    return {
        "schema": materialize._PARENT_SCHEMA,
        "format_version": 1,
        "role": materialize._PARENT_ROLE,
        "selection": {
            "selected_arm": materialize._SELECTED_ARM,
            "factorial_capacity_gate_passed": True,
            "frozen_basis_one_pass_carrier_transfer_oracle_authorized": True,
        },
        "scientific_status": {
            "tail_informed_factorial_complete": True,
            "all_eight_arms_executed": True,
            "live_exact_h4_ceiling_validated": True,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "tail_informed_fit": {
            "artifact_sha256": materialize.TAIL_INFORMED_FIT_ARTIFACT_SHA256,
            "anchor_rank": 192,
            "tail_rank": 17,
            "max_rank": 320,
            "source_example_count": 16,
            "source_row_count": 819,
            "source_tail_row_count": 17,
            "treatment_basis_rows": {
                "matrix_sha256": materialize.EXPECTED_BASIS_MATRIX_SHA256,
                "shape": [320, 640],
            },
        },
        "fitted_bases": {
            "unweighted": {
                "artifact_sha256": (
                    materialize._EXPECTED_GLOBAL_BASIS_ARTIFACT_SHA256
                ),
                "basis_rows": {
                    "matrix_sha256": (
                        materialize._EXPECTED_GLOBAL_BASIS_MATRIX_SHA256
                    ),
                    "shape": [320, 640],
                },
            }
        },
        "collect_receipts": receipts,
    }


def test_parent_factorial_loader_authenticates_file_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _minimal_parent_report()
    report["report_sha256"] = materialize.frozen._json_sha256(
        report,
        domain=materialize._PARENT_REPORT_DOMAIN,
    )
    path = tmp_path / "parent.json"
    path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        materialize,
        "PARENT_FILE_SHA256",
        materialize.frozen._file_sha256(path),
    )
    monkeypatch.setattr(
        materialize,
        "PARENT_REPORT_SHA256",
        report["report_sha256"],
    )

    loaded = materialize._load_parent_factorial(path)

    assert loaded["report_sha256"] == report["report_sha256"]
    tampered = dict(report)
    tampered["role"] = "tampered"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="file differs"):
        materialize._load_parent_factorial(path)


def test_parent_tail_lineage_is_positional_and_cross_checked() -> None:
    example_ids = tuple(f"example-{index:02d}" for index in range(16))
    family_ids = tuple(f"family-{index // 2:02d}" for index in range(16))
    sequence_sha256s = tuple(_digest(200 + index) for index in range(16))
    pair_sha256s = tuple(_digest(300 + index) for index in range(16))
    report = {
        "fit_manifest": {
            "example_ids": example_ids,
            "family_ids_by_example": family_ids,
            "sequence_sha256s": sequence_sha256s,
        },
        "tail_informed_fit": {
            "source_pair_sha256s": pair_sha256s,
            "source_graph_core_mask_sha256s": tuple(
                _digest(400 + index) for index in range(16)
            ),
            "source_trace_sha256s": tuple(
                _digest(500 + index) for index in range(16)
            ),
            "graph_core_mask_sha256s": tuple(
                _digest(600 + index) for index in range(16)
            ),
        },
        "collect_receipts": [
            {
                "example_id": example_ids[index],
                "family_id": family_ids[index],
                "pair": {"artifact_sha256": pair_sha256s[index]},
                "fit_sequence": {
                    "sequence_sha256": sequence_sha256s[index]
                },
            }
            for index in reversed(range(16))
        ],
    }

    lineage = materialize._parent_lineage_by_example(report)

    assert tuple(lineage) == example_ids
    assert lineage[example_ids[7]]["source_pair_sha256"] == pair_sha256s[7]
    report["collect_receipts"][0]["pair"]["artifact_sha256"] = _digest(999)
    with pytest.raises(ValueError, match="internally inconsistent"):
        materialize._parent_lineage_by_example(report)


class _FakeObservation:
    def __init__(self) -> None:
        self.native_x4 = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        self.native_h4 = torch.tensor(
            [[[4.0, 7.0], [9.0, 13.0], [2.0, 3.0]]],
            dtype=torch.float32,
        )
        self.incomplete_h4 = torch.tensor(
            [[[1.0, 2.0], [3.0, 5.0], [2.0, 3.0]]],
            dtype=torch.float32,
        )
        self.h4_gradient = torch.tensor(
            [[[0.25, -0.5], [1.0, 0.75], [0.0, 0.0]]],
            dtype=torch.float32,
        )
        support = torch.tensor([[True, True, False]])
        target = torch.tensor([[True, False, False]])
        self.complete_h4_support_mask = support
        self.prefix = SimpleNamespace(
            source_modes=torch.tensor([[[1.0], [2.0], [0.0]]]),
            target_affected_mask=target,
            valid_target_mask=torch.tensor([[True, True, False]]),
            artifact_sha256=_digest(701),
        )
        self.native_x4_sha256 = _runtime_tensor_sha256(self.native_x4)
        self.native_h4_sha256 = _runtime_tensor_sha256(self.native_h4)
        self.incomplete_h4_sha256 = _runtime_tensor_sha256(self.incomplete_h4)
        self.h4_gradient_sha256 = _runtime_tensor_sha256(self.h4_gradient)
        self.native_logits_sha256 = _digest(702)
        self.partial_exact_x4_logits_sha256 = _digest(703)
        self.supervised_indices_sha256 = _digest(704)
        self.supervised_targets_sha256 = _digest(705)
        self.objective_receipt_sha256 = _digest(706)
        self.objective_mean_nll = 2.5
        self.objective_ignore_index = -100
        self.supervised_token_count = 2
        self.adapter_execution_sha256 = _digest(707)
        self.model_inputs_sha256 = _digest(708)
        self.execution_grid_sha256 = _digest(709)
        self.bridge_binding_sha256 = _digest(710)
        self.artifact_sha256 = _digest(711)

    def validate_integrity(self) -> None:
        return None


def test_fit_trace_uses_select_then_independent_float64_cast() -> None:
    observation = _FakeObservation()
    example = SimpleNamespace(
        example_id="example-00",
        family_id="family-00",
        prompt="one prompt",
    )
    support_indices = torch.tensor([0, 1])
    expected_residual = (
        observation.native_h4[0].index_select(0, support_indices).to(torch.float64)
        - observation.incomplete_h4[0]
        .index_select(0, support_indices)
        .to(torch.float64)
    )
    expected_sequence = CompleteH4ProjectionFitSequence(
        example_id=example.example_id,
        family_id=example.family_id,
        residual_rows=expected_residual,
        gradient_rows=observation.h4_gradient[0].index_select(
            0,
            support_indices,
        ),
    )
    pair = {
        "native_h4_sha256": observation.native_h4_sha256,
        "incomplete_h4_sha256": observation.incomplete_h4_sha256,
        "h4_gradient_sha256": observation.h4_gradient_sha256,
        "partial_exact_x4_logits_sha256": (
            observation.partial_exact_x4_logits_sha256
        ),
        "supervised_indices_sha256": observation.supervised_indices_sha256,
        "supervised_targets_sha256": observation.supervised_targets_sha256,
        "objective_receipt_sha256": observation.objective_receipt_sha256,
        "objective_mean_nll": observation.objective_mean_nll,
        "objective_ignore_index": -100,
        "objective_reduction": "mean",
        "supervised_token_count": 2,
        "adapter_execution_sha256": observation.adapter_execution_sha256,
        "source_modes_sha256": _runtime_tensor_sha256(
            observation.prefix.source_modes
        ),
        "complete_h4_support_mask_sha256": _runtime_tensor_sha256(
            observation.complete_h4_support_mask
        ),
        "complete_h4_support_rows": 2,
        "graph_target_affected_rows": 1,
        "complete_h4_support_outside_graph_rows": 1,
        "incomplete_h4_difference_rows": 2,
        "incomplete_h4_difference_padding_rows": 0,
        "incomplete_h4_difference_outside_support_rows": 0,
    }
    receipt = {
        "example_id": example.example_id,
        "family_id": example.family_id,
        "prompt_sha256": materialize.frozen._prompt_sha256(example.prompt),
        "tokenized_tokens": 3,
        "supervised_tokens": 2,
        "model_inputs_sha256": observation.model_inputs_sha256,
        "execution_grid_sha256": observation.execution_grid_sha256,
        "pair": pair,
        "fit_sequence": expected_sequence.metadata(),
    }
    parent_lineage = {
        "receipt": receipt,
        "sequence_sha256": expected_sequence.sequence_sha256,
        "source_pair_sha256": _digest(712),
        "source_graph_core_mask_sha256": _runtime_tensor_sha256(
            observation.prefix.target_affected_mask
        ),
        "trace_sha256": _digest(713),
        "trace_graph_core_mask_sha256": _digest(714),
    }

    trace = materialize._fit_trace_from_observation(
        observation,
        example=example,
        model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        supervised_indices=torch.tensor([0, 1]),
        parent_lineage=parent_lineage,
    )

    assert torch.equal(trace.sequence.residual_rows.to_tensor(), expected_residual)
    assert trace.sequence.sequence_sha256 == expected_sequence.sequence_sha256
    assert trace.proof[
        "all_parent_tensor_objective_and_sequence_receipts_matched"
    ] is True
    pair["native_h4_sha256"] = _digest(999)
    with pytest.raises(ValueError, match="native H4 hash differs"):
        materialize._fit_trace_from_observation(
            observation,
            example=example,
            model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
            supervised_indices=torch.tensor([0, 1]),
            parent_lineage=parent_lineage,
        )


def test_tail_trace_admits_frozen_pair_and_mask_only_on_exact_receipt() -> None:
    sequence = CompleteH4ProjectionFitSequence(
        example_id="example-00",
        family_id="family-00",
        residual_rows=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        gradient_rows=torch.tensor([[0.5, 0.25], [0.75, 1.0]]),
    )
    graph_core = torch.tensor([True, False])
    pair_sha256 = _digest(720)
    source_mask_sha256 = _digest(721)
    expected = CompleteH4TailProjectionTrace.from_fit_sequence(
        sequence,
        graph_core,
        source_pair_sha256=pair_sha256,
        source_graph_core_mask_sha256=source_mask_sha256,
    )
    trace = materialize._MaterializationTrace(
        sequence=sequence,
        graph_core_rows=graph_core,
        source_pair_sha256=pair_sha256,
        source_graph_core_mask_sha256=source_mask_sha256,
        expected_trace_sha256=expected.trace_sha256,
        expected_trace_graph_core_mask_sha256=(
            expected.graph_core_mask_sha256
        ),
        proof={},
    )

    admitted = materialize._authenticated_tail_traces((trace,))

    assert admitted[0].trace_sha256 == expected.trace_sha256
    tampered = materialize._MaterializationTrace(
        sequence=sequence,
        graph_core_rows=graph_core,
        source_pair_sha256=_digest(722),
        source_graph_core_mask_sha256=source_mask_sha256,
        expected_trace_sha256=expected.trace_sha256,
        expected_trace_graph_core_mask_sha256=(
            expected.graph_core_mask_sha256
        ),
        proof={},
    )
    with pytest.raises(ValueError, match="tail trace differs"):
        materialize._authenticated_tail_traces((tampered,))


class _FakeSidecar:
    def __init__(self, basis: torch.Tensor, artifact: str) -> None:
        self._basis = basis.clone()
        self.artifact_sha256 = artifact

    def validate_integrity(self) -> None:
        return None

    def basis_tensor(self) -> torch.Tensor:
        return self._basis.clone()


def test_sidecar_publication_recovery_accepts_only_exact_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "basis.pt"
    output.write_bytes(b"authenticated sidecar bytes")
    basis = torch.eye(2, dtype=torch.float64)
    artifact = _digest(800)
    built = _FakeSidecar(basis, artifact)
    existing = _FakeSidecar(basis, artifact)
    monkeypatch.setattr(
        materialize,
        "build_complete_h4_rank320_basis_sidecar",
        lambda value: built,
    )
    monkeypatch.setattr(
        materialize,
        "save_complete_h4_rank320_basis_sidecar",
        lambda *args, **kwargs: pytest.fail("exact orphan must not be overwritten"),
    )

    selected, receipt, recovered = materialize._publish_or_recover_exact_sidecar(
        basis,
        basis_destination=output,
        existing_sidecar=existing,
    )

    assert selected is built
    assert recovered is True
    assert receipt["logical_artifact_sha256"] == artifact
    mismatch = _FakeSidecar(torch.zeros_like(basis), artifact)
    with pytest.raises(ValueError, match="differs from reproduction"):
        materialize._publish_or_recover_exact_sidecar(
            basis,
            basis_destination=output,
            existing_sidecar=mismatch,
        )


def _committed_report_fixture() -> tuple[dict[str, object], dict[str, object]]:
    ids = tuple(f"example-{index:02d}" for index in range(16))
    families = tuple(f"family-{index // 2:02d}" for index in range(16))
    sequences = tuple(_digest(1000 + index) for index in range(16))
    pairs = tuple(_digest(1100 + index) for index in range(16))
    source_masks = tuple(_digest(1200 + index) for index in range(16))
    receipts = []
    proofs = []
    for index, example_id in enumerate(ids):
        pair = {
            "artifact_sha256": pairs[index],
            "native_h4_sha256": _digest(1300 + index),
            "incomplete_h4_sha256": _digest(1400 + index),
            "h4_gradient_sha256": _digest(1500 + index),
            "partial_exact_x4_logits_sha256": _digest(1600 + index),
            "objective_receipt_sha256": _digest(1700 + index),
            "source_modes_sha256": _digest(1800 + index),
            "complete_h4_support_mask_sha256": _digest(1900 + index),
        }
        sequence = {
            "sequence_sha256": sequences[index],
            "residual_rows": {"matrix_sha256": _digest(2000 + index)},
            "gradient_rows": {"matrix_sha256": _digest(2100 + index)},
        }
        receipt = {
            "example_id": example_id,
            "family_id": families[index],
            "prompt_sha256": _digest(2200 + index),
            "pair": pair,
            "fit_sequence": sequence,
        }
        receipts.append(receipt)
        proofs.append(
            {
                "example_id": example_id,
                "family_id": families[index],
                "prompt_sha256": receipt["prompt_sha256"],
                "observation_artifact_sha256": _digest(2300 + index),
                "bridge_binding_sha256": (
                    materialize._GRAPH_BRIDGE_BINDING_SHA256
                ),
                "prefix_artifact_sha256": _digest(2500 + index),
                "native_x4_sha256": _digest(2600 + index),
                "native_h4_sha256": pair["native_h4_sha256"],
                "incomplete_h4_sha256": pair["incomplete_h4_sha256"],
                "h4_gradient_sha256": pair["h4_gradient_sha256"],
                "native_logits_sha256": _digest(2700 + index),
                "partial_exact_x4_logits_sha256": (
                    pair["partial_exact_x4_logits_sha256"]
                ),
                "objective_receipt_sha256": pair["objective_receipt_sha256"],
                "source_modes_sha256": pair["source_modes_sha256"],
                "complete_h4_support_mask_sha256": (
                    pair["complete_h4_support_mask_sha256"]
                ),
                "graph_core_mask_sha256": source_masks[index],
                "residual_rows_matrix_sha256": sequence["residual_rows"][
                    "matrix_sha256"
                ],
                "gradient_rows_matrix_sha256": sequence["gradient_rows"][
                    "matrix_sha256"
                ],
                "fit_sequence_sha256": sequences[index],
                "frozen_parent_pair_sha256": pairs[index],
                "all_parent_tensor_objective_and_sequence_receipts_matched": True,
                "legacy_shadow_reexecuted": False,
            }
        )
    fit_manifest = {
        "example_ids": ids,
        "family_ids_by_example": families,
        "sequence_sha256s": sequences,
    }
    parent = {
        "runtime_binding": {
            "runtime_binding_sha256": _digest(2999),
        },
        "fit_manifest": fit_manifest,
        "tail_informed_fit": {
            "source_pair_sha256s": pairs,
            "source_graph_core_mask_sha256s": source_masks,
            "source_trace_sha256s": tuple(
                _digest(2800 + index) for index in range(16)
            ),
            "graph_core_mask_sha256s": tuple(
                _digest(2900 + index) for index in range(16)
            ),
        },
        "collect_receipts": receipts,
    }
    sidecar_metadata = {
        "schema": "closed-sidecar",
        "projection_rank": 320,
        "residual_width": 640,
        "basis_matrix_sha256": materialize.EXPECTED_BASIS_MATRIX_SHA256,
        "runtime_tensor_sha256": materialize.EXPECTED_RUNTIME_TENSOR_SHA256,
        "projection_basis_artifact_sha256": (
            materialize.EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256
        ),
    }
    sidecar_receipt = {
        "file": ".local-runs/d320.pt",
        "file_sha256": _digest(3000),
        "file_bytes": 1234,
        "logical_artifact_sha256": _digest(3001),
        "committable": False,
    }
    report = {
        "schema": materialize._SCHEMA,
        "format_version": materialize._FORMAT_VERSION,
        "role": materialize._ROLE,
        "lineage": {
            "parent_factorial_file_sha256": materialize.PARENT_FILE_SHA256,
            "parent_factorial_report_sha256": materialize.PARENT_REPORT_SHA256,
            "selected_arm": materialize._SELECTED_ARM,
            "global_fit_basis_artifact_sha256": (
                materialize._EXPECTED_GLOBAL_BASIS_ARTIFACT_SHA256
            ),
            "tail_informed_fit_artifact_sha256": (
                materialize.TAIL_INFORMED_FIT_ARTIFACT_SHA256
            ),
            "fit_to_prefix_lineage_sha256": (
                materialize.FIT_TO_PREFIX_LINEAGE_SHA256
            ),
            "projection_basis_artifact_sha256": (
                materialize.EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256
            ),
        },
        "protocol": {
            "panel": "reused_calibration_a_fit16",
            "collection_per_prompt": (
                "native_boundary_forward_then_clamped_y3_exact_x4_h4_vjp"
            ),
            "model_forwards_per_prompt": 2,
            "backwards_per_prompt": 1,
            "objective": "prompt_mean_next_token_nll",
            "residual_row_arithmetic": (
                "select_then_native_to_float64_minus_incomplete_to_float64"
            ),
            "fit_ordering": "family_id_then_example_id",
            "global_fit": "family_example_macro_unweighted_u320",
            "tail_fit": (
                "u192_then_full_tail_residual_svd_span_then_two_pass_mgs_u320"
            ),
            "legacy_pair_and_graph_mask_lineage_policy": (
                "frozen_parent_receipts_reused_only_after_exact_live_tensor_"
                "objective_sequence_and_mask_proof"
            ),
            "transfer_evaluation": "not_run",
        },
        "carrier_promotion": {
            "d320_source_runtime_binding_sha256": _digest(2999),
            "graph_target_runtime_binding_sha256": (
                materialize._GRAPH_RUNTIME_BINDING_SHA256
            ),
            "graph_target_bridge_binding_sha256": (
                materialize._GRAPH_BRIDGE_BINDING_SHA256
            ),
            "graph_plan_sha256": (
                "10299119071f215c97979edf6b02bec4e7e7cde5a6d2c316a662b802a84aa469"
            ),
            "graph_prepared_type": "PreparedGraphOrganizedSVD",
            "graph_pack_count": 4,
            "fit_consumed_prefix_geometry": {
                "residual_width": 640,
                "source_modes": 64,
                "target_modes": 64,
                "fit_knot_origins": (8, 24, 40),
                "lag_count": 32,
            },
            "d320_source_graph_prediction_rank": 64,
            "graph_target_prediction_rank": 45,
            "prediction_rank_divergence_expected": True,
            "consumed_basis_tensor_sha256s": dict(
                materialize._GRAPH_CARRIER_CONSUMED_BASIS_SHA256S
            ),
            "prompt_free_probe_tensor_sha256s": dict(
                materialize._GRAPH_CARRIER_PROBE_SHA256S
            ),
            "fit_consumed_prefix_fields_bitwise_identical": True,
            "graph_prediction_admitted_to_exact_x4_fit_lineage": False,
            "preflight_model_forward_count": 0,
        },
        "fit_manifest": fit_manifest,
        "reproduction": {
            "observation_receipts": proofs,
            "all_16_parent_fit_sequences_reproduced_exactly": True,
            "all_16_parent_tail_traces_reproduced_exactly": True,
            "unweighted_u320_metadata_reproduced_exactly": True,
            "tail_informed_fit_metadata_reproduced_exactly": True,
            "support_rows": 819,
            "graph_core_rows": 802,
            "causal_tail_rows": 17,
            "legacy_three_pass_shadow_reexecuted": False,
            "frozen_pair_lineage_reused_after_exact_proof": True,
            "frozen_graph_mask_lineage_reused_after_exact_proof": True,
        },
        "materialized_basis": {
            **sidecar_metadata,
            "sidecar_file_sha256": sidecar_receipt["file_sha256"],
            "sidecar_file_bytes": sidecar_receipt["file_bytes"],
        },
        "resource_accounting": {
            "model_load_count": 1,
            "tokenizer_load_count": 1,
            "model_forward_count": 32,
            "backward_count": 16,
            "evaluation_model_forward_count": 0,
            "projection_arm_model_forward_count": 0,
            "transfer_model_forward_count": 0,
            "simultaneously_live_full_vocabulary_logit_tensor_peak": 1,
            "full_vocabulary_logit_roles_at_peak": (
                "native_or_partial_exact_x4_never_both",
            ),
            "retained_fit_sequence_count": 16,
            "retained_fit_sequence_residual_and_gradient_bytes": 819
            * 640
            * 8
            * 2,
            "d320_float64_coefficient_count": 320 * 640,
            "d320_float64_matrix_bytes": 320 * 640 * 8,
            "float64_width_square_eigendecomposition_count": 1,
            "tail_residual_float64_svd_count": 1,
            "inference_projection_macs_executed": 0,
            "basis_fit_linear_algebra_is_offline_only": True,
            "latency_or_speed_claim": False,
            "whole_model_parameter_reduction_claim": False,
        },
        "publication": {
            "report_is_sidecar_commit_marker": True,
            "consumer_must_require_report_and_sidecar": True,
            "sidecar_atomic_no_overwrite_publication": True,
            "report_atomic_no_overwrite_publication": True,
            "recovered_existing_authenticated_orphan_sidecar": False,
            "mismatched_existing_sidecar_accepted": False,
            "existing_report_overwrite_allowed": False,
        },
        "scientific_status": {
            "exact_rank320_basis_materialized": True,
            "same_a_fit_only": True,
            "transfer_evaluation_run": False,
            "frozen_basis_one_pass_carrier_transfer_ready": True,
            "candidate_serving_authorized": False,
            "generator_validated": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "next_rung": "fixed_d320_one_pass_carrier_transfer_oracle",
        },
        "artifact": {
            "file": ".local-runs/materialization.json",
            "basis_sidecar": sidecar_receipt,
            "committable": False,
        },
        "safety": dict(materialize._SAFETY),
    }
    return report, {"report": parent, "sidecar_metadata": sidecar_metadata}


def test_commit_marker_loader_rejects_extra_proof_keys_and_authenticates_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path(".local-runs")
    root.mkdir()
    report, fixtures = _committed_report_fixture()
    parent = fixtures["report"]
    sidecar_metadata = fixtures["sidecar_metadata"]
    sidecar = SimpleNamespace(
        artifact_sha256=_digest(3001),
        metadata=lambda: sidecar_metadata,
    )
    monkeypatch.setattr(
        materialize,
        "_load_parent_factorial",
        lambda path: {"report": parent},
    )
    monkeypatch.setattr(
        materialize,
        "load_complete_h4_rank320_basis_sidecar",
        lambda path: sidecar,
    )
    receipt = report["artifact"]["basis_sidecar"]
    monkeypatch.setattr(
        materialize,
        "_existing_sidecar_receipt",
        lambda *args, **kwargs: dict(receipt),
    )
    path = root / "materialization.json"

    def publish() -> tuple[str, str]:
        report.pop("report_sha256", None)
        report["report_sha256"] = materialize.frozen._json_sha256(
            report,
            domain=materialize._REPORT_DOMAIN,
        )
        path.write_text(json.dumps(report), encoding="utf-8")
        return materialize.frozen._file_sha256(path), report["report_sha256"]

    file_sha256, report_sha256 = publish()
    loaded = materialize.load_complete_h4_rank320_basis_materialization_report(
        path,
        expected_file_sha256=file_sha256,
        expected_report_sha256=report_sha256,
    )
    assert loaded["report_sha256"] == report_sha256
    with pytest.raises(ValueError, match="identity differs"):
        materialize.load_complete_h4_rank320_basis_materialization_report(
            path,
            expected_file_sha256="0" * 64,
            expected_report_sha256=report_sha256,
        )

    report["reproduction"]["observation_receipts"][0]["prompt"] = "forbidden"
    file_sha256, report_sha256 = publish()
    with pytest.raises(ValueError, match="proof keyset differs"):
        materialize.load_complete_h4_rank320_basis_materialization_report(
            path,
            expected_file_sha256=file_sha256,
            expected_report_sha256=report_sha256,
        )

    report["reproduction"]["observation_receipts"][0].pop("prompt")
    report["carrier_promotion"]["consumed_basis_tensor_sha256s"][
        "basis.R3"
    ] = "0" * 64
    file_sha256, report_sha256 = publish()
    with pytest.raises(ValueError, match="carrier promotion differs"):
        materialize.load_complete_h4_rank320_basis_materialization_report(
            path,
            expected_file_sha256=file_sha256,
            expected_report_sha256=report_sha256,
        )

    report["carrier_promotion"]["consumed_basis_tensor_sha256s"] = dict(
        materialize._GRAPH_CARRIER_CONSUMED_BASIS_SHA256S
    )
    report["reproduction"]["observation_receipts"][0][
        "bridge_binding_sha256"
    ] = "0" * 64
    file_sha256, report_sha256 = publish()
    with pytest.raises(ValueError, match="observation bridge lineage differs"):
        materialize.load_complete_h4_rank320_basis_materialization_report(
            path,
            expected_file_sha256=file_sha256,
            expected_report_sha256=report_sha256,
        )


def test_materializer_closes_context_when_parent_lineage_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        parent={"invalid": True},
        carrier_preflight={"receipt": _digest(1)},
        closed=0,
    )

    def close() -> None:
        context.closed += 1

    context.close = close
    monkeypatch.setattr(
        materialize,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        materialize,
        "_parent_lineage_by_example",
        lambda parent: (_ for _ in ()).throw(ValueError("lineage rejected")),
    )

    with pytest.raises(ValueError, match="lineage rejected"):
        materialize.run_gemma3_l3_l4_complete_h4_rank320_basis_materialization(
            basis_output=Path(".local-runs/basis.pt"),
            output=Path(".local-runs/report.json"),
        )
    assert context.closed == 1


def test_live_context_closes_and_rejects_foreign_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = SimpleNamespace(prompt="prompt")
    adapter = SimpleNamespace(
        model_fingerprint=lambda: materialize.frozen._EXPECTED_RAW_MODEL_SHA256
    )
    fit_runtime = SimpleNamespace(validate_integrity=lambda: None)
    runtime = SimpleNamespace(validate_integrity=lambda: None)
    bridge = SimpleNamespace(validate_integrity=lambda: None)
    switcher = SimpleNamespace(closed=0)

    def close() -> None:
        switcher.closed += 1

    switcher.close = close
    tokenizer_stages: list[str] = []
    parent = {
        "lineage": {
            "factorized_live_model_sha256": _digest(900),
            "factorized_adapter_execution_sha256": _digest(901),
        }
    }
    monkeypatch.setattr(
        materialize.frozen,
        "_live_factorized_identity",
        lambda value: (_digest(900), _digest(901)),
    )
    monkeypatch.setattr(
        materialize.frozen,
        "_tokenize_one",
        lambda *args, **kwargs: (
            {"input_ids": torch.tensor([[1, 2]])},
            torch.tensor([0]),
            torch.tensor([2]),
        ),
    )
    carrier_preflight = {"receipt": _digest(902)}
    monkeypatch.setattr(
        materialize,
        "validate_complete_h4_rank320_runtime_graph_abi",
        lambda **kwargs: dict(carrier_preflight),
    )
    context = materialize.CompleteH4Rank320LiveContext(
        examples=(example,),
        adapter=adapter,
        fit_runtime=fit_runtime,
        runtime=runtime,
        bridge=bridge,
        carrier_preflight=carrier_preflight,
        parent=parent,
        panel_receipt={},
        max_length=64,
        device=torch.device("cpu"),
        _tokenizer=object(),
        _tokenizer_integrity_check=tokenizer_stages.append,
        _switcher=switcher,
    )

    context.validate_immutable_inputs()
    assert tokenizer_stages == ["before"]
    assert context.tokenize(example)[0]["input_ids"].shape == (1, 2)
    assert tokenizer_stages[-2:] == ["before", "after"]
    with pytest.raises(ValueError, match="does not belong"):
        context.tokenize(SimpleNamespace(prompt="prompt"))
    context.validate_immutable_inputs()
    assert tokenizer_stages[-1] == "after"
    context.close()
    context.close()
    assert switcher.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        context.tokenize(example)


@pytest.mark.skipif(
    not all(
        Path(path).exists()
        for path in (
            materialize.frozen.DEFAULT_INTERIOR_ARTIFACT,
            materialize.frozen.DEFAULT_PARENT_ARTIFACT,
            materialize.frozen.DEFAULT_CANDIDATE_ARTIFACT,
            materialize.frozen.DEFAULT_BASIS_PACKAGE,
            materialize.DEFAULT_GRAPH_CANDIDATE,
        )
    ),
    reason="locked local carrier artifacts are unavailable",
)
def test_real_artifact_graph_preflight_preserves_only_fit_consumed_prefix() -> None:
    fit_source = materialize.frozen.load_gemma3_spectral_source(
        materialize.frozen.DEFAULT_INTERIOR_ARTIFACT,
        expected_file_sha256=(
            materialize.frozen.DEFAULT_INTERIOR_ARTIFACT_SHA256
        ),
        expected_report_sha256=(
            materialize.frozen.DEFAULT_INTERIOR_REPORT_SHA256
        ),
        expected_origins=materialize.frozen.INTERIOR_ORIGINS,
    )
    graph_parent = materialize.frozen.load_gemma3_graph_wavelet_candidate(
        materialize.frozen.DEFAULT_PARENT_ARTIFACT,
        expected_artifact_sha256=(
            materialize.frozen.DEFAULT_PARENT_ARTIFACT_SHA256
        ),
        expected_tensor_file_sha256=(
            materialize.frozen.DEFAULT_PARENT_TENSOR_FILE_SHA256
        ),
        expected_report_sha256=(
            materialize.frozen.DEFAULT_PARENT_REPORT_SHA256
        ),
    )
    signed_candidate = (
        materialize.frozen
        .load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
            materialize.frozen.DEFAULT_CANDIDATE_ARTIFACT,
            expected_artifact_sha256=(
                materialize.frozen.DEFAULT_FROZEN_ARTIFACT_SHA256
            ),
            expected_tensor_file_sha256=(
                materialize.frozen.DEFAULT_FROZEN_TENSOR_FILE_SHA256
            ),
            expected_report_sha256=(
                materialize.frozen.DEFAULT_FROZEN_REPORT_SHA256
            ),
        )
    )
    plan, _receipt = materialize.frozen.build_rank64_global_svd_plan(
        fit_source,
        graph_parent,
    )
    protocol = (
        materialize.frozen
        .default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    metadata = protocol.metadata()
    graph_binding = metadata["graph_candidate"]
    basis_binding = metadata["prompt_blind_basis"]
    basis = materialize.frozen.load_gemma3_l3_l4_basis_package(
        materialize.frozen.DEFAULT_BASIS_PACKAGE,
        expected_file_sha256=basis_binding["tensor_file_sha256"],
        expected_payload_sha256=basis_binding["logical_payload_sha256"],
    )
    fit_runtime = materialize.Gemma3L3L4ConditionalSpectralShadowRuntime(
        plan,
        basis,
        candidate_artifact_sha256=(
            materialize.frozen._EXPECTED_RANK64_ARM_SHA256
        ),
        candidate_method="global_svd_rank64_capacity_oracle",
        candidate_binding=signed_candidate.binding,
        candidate_model=signed_candidate.model,
        expected_plan_artifact_sha256=(
            materialize.frozen._EXPECTED_RANK64_PLAN_SHA256
        ),
        expected_basis_payload_sha256=basis_binding[
            "logical_payload_sha256"
        ],
        expected_live_model_sha256=graph_binding[
            "factorized_live_execution_sha256"
        ],
        expected_adapter_execution_sha256=graph_binding[
            "factorized_refit_execution_sha256"
        ],
        analysis_device="cpu",
    )
    graph_candidate = materialize.load_gemma3_graph_organized_svd_candidate(
        materialize.DEFAULT_GRAPH_CANDIDATE,
        expected_file_sha256=graph_binding["tensor_file_sha256"],
    )
    graph_runtime = materialize.Gemma3L3L4GraphOrganizedSVDShadowRuntime(
        graph_candidate,
        basis,
        expected_candidate_artifact_sha256=graph_binding[
            "logical_artifact_sha256"
        ],
        expected_basis_payload_sha256=basis_binding[
            "logical_payload_sha256"
        ],
        expected_plan_artifact_sha256=graph_binding[
            "deployment_plan_sha256"
        ],
        expected_live_model_sha256=graph_binding[
            "factorized_live_execution_sha256"
        ],
        expected_adapter_execution_sha256=graph_binding[
            "factorized_refit_execution_sha256"
        ],
        analysis_device="cpu",
    )
    bridge = graph_runtime.export_one_pass_bridge()

    receipt = materialize.validate_complete_h4_rank320_runtime_graph_abi(
        fit_runtime=fit_runtime,
        runtime=graph_runtime,
        bridge=bridge,
    )

    assert receipt["fit_consumed_prefix_fields_bitwise_identical"] is True
    assert receipt["d320_source_graph_prediction_rank"] == 64
    assert receipt["graph_target_prediction_rank"] == 45
    assert receipt["prediction_rank_divergence_expected"] is True
    assert (
        receipt["graph_prediction_admitted_to_exact_x4_fit_lineage"]
        is False
    )
    with pytest.raises(TypeError, match="genuine graph-organized"):
        materialize.validate_complete_h4_rank320_runtime_graph_abi(
            fit_runtime=fit_runtime,
            runtime=fit_runtime,  # type: ignore[arg-type]
            bridge=bridge,
        )


@pytest.mark.parametrize(
    "path",
    (
        Path("outside.json"),
        Path(".local-runs/../escape.json"),
        Path(".local-runs/run.pt"),
    ),
)
def test_report_output_is_lexically_confined(path: Path) -> None:
    with pytest.raises(ValueError, match="under .local-runs"):
        materialize._validate_output(path)
