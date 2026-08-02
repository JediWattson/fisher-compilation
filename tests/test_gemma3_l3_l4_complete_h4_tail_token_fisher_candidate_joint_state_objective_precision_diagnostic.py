from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_objective_precision_diagnostic as diagnostic,
)


def _sha(value: object) -> str:
    return hashlib.sha256(repr(value).encode("ascii")).hexdigest()


def _mock_v9_report() -> dict[str, object]:
    endpoints: list[dict[str, object]] = []
    nodes: list[dict[str, object]] = []
    evidence_ids: list[str] = []
    evidence_hashes: list[str] = []
    for index in range(16):
        example = f"example-{index:02d}"
        family = f"family-{index // 2}"
        token_count = 5 + index % 3
        evidence = _sha(("evidence", example))
        evidence_ids.append(example)
        evidence_hashes.append(evidence)
        base: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "actual_cast_once_endpoint_pair": True,
            "v8_scalar_and_joint_token_KL_hashes_replayed": True,
            "scalar_endpoint": {
                "realized_endpoint_h4_dtype": "torch.float32",
                "endpoint_supervised_token_count": token_count,
            },
            "joint_endpoint": {
                "realized_endpoint_h4_dtype": "torch.float32",
                "endpoint_supervised_token_count": token_count,
            },
        }
        endpoint = {
            **base,
            "artifact_sha256": diagnostic.token_v1._domain_sha256(
                base, domain=diagnostic.v9diag._ENDPOINT_PAIR_DOMAIN
            ),
            "core_evidence_artifact_sha256": evidence,
            "core_scalar_tangent_receipt_sha256": _sha(("scalar", example)),
            "core_held_unit_tangent_receipt_sha256": _sha(("unit", example)),
            "additive_prompt_scalars": {
                "finite_joint_minus_scalar_teacher_KL": index * 1.0e-7
            },
        }
        node_hashes: list[str] = []
        for node_index, (alpha, weight) in enumerate(
            zip(
                diagnostic.GL4_UNIT_INTERVAL_NODES,
                diagnostic.GL4_UNIT_INTERVAL_WEIGHTS,
                strict=True,
            )
        ):
            node: dict[str, object] = {
                "example_id": example,
                "family_id": family,
                "node_index": node_index,
                "path_fraction_hex": alpha.hex(),
                "quadrature_weight_hex": weight.hex(),
                "backward_call_count": (
                    token_count + diagnostic.token_v1._VJP_CHUNK_SIZE - 1
                )
                // diagnostic.token_v1._VJP_CHUNK_SIZE,
            }
            receipt = diagnostic.token_v1._domain_sha256(
                node, domain=diagnostic.v9diag._PATH_NODE_RECEIPT_DOMAIN
            )
            node["receipt_sha256"] = receipt
            nodes.append(node)
            node_hashes.append(receipt)
        endpoint["path_node_receipt_sha256s"] = node_hashes
        endpoints.append(endpoint)
    ordered = tuple(
        str(row["receipt_sha256"])
        for row in sorted(
            nodes,
            key=lambda row: (str(row["example_id"]), int(row["node_index"])),
        )
    )
    return {
        "schema": diagnostic.v9diag._SCHEMA,
        "classification": "scalar_joint_path_closure_unresolved_same_a",
        "passed": False,
        "resources": {
            "total_model_forward_count": 240,
            "total_backward_call_count": 1148,
            "total_candidate_support_row_executions": 8575,
        },
        "endpoint_pair_receipts": endpoints,
        "path_node_receipts": nodes,
        "path_observation_set_sha256": diagnostic.token_v1._domain_sha256(
            ordered, domain=diagnostic.v9diag._PATH_OBSERVATION_SET_DOMAIN
        ),
        "family_equal_scalar_joint_path_attribution": {
            "closure_relative_rmse": 0.089016283613215,
            "closure_cosine": 0.9960743421828786,
            "evidence_example_ids": evidence_ids,
            "evidence_artifact_sha256s": evidence_hashes,
        },
    }


def test_v9_loader_uses_exact_file_and_logical_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    report = _mock_v9_report()

    def load(path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return report

    monkeypatch.setattr(diagnostic.token_v1, "_load_pinned_report", load)
    assert diagnostic._load_v9_report("v9.json") is report
    assert captured["expected_file_sha256"] == diagnostic.V9_REPORT_FILE_SHA256
    assert captured["expected_report_sha256"] == diagnostic.V9_REPORT_SHA256

    invalid = {**report, "classification": "unexpected"}
    monkeypatch.setattr(
        diagnostic.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: invalid,
    )
    with pytest.raises(RuntimeError, match="V9 anchor differs"):
        diagnostic._load_v9_report("v9.json")


def test_v9_index_authenticates_variable_lengths_and_finalized_node_chains() -> None:
    report = _mock_v9_report()
    indexed = diagnostic._index_v9_receipts(report)
    assert len(indexed.endpoints) == 16
    assert len(indexed.nodes) == 64
    assert {
        row["scalar_endpoint"]["endpoint_supervised_token_count"]
        for row in indexed.endpoints.values()
    } == {5, 6, 7}

    tampered_evidence = _mock_v9_report()
    tampered_evidence["endpoint_pair_receipts"][0][  # type: ignore[index]
        "core_evidence_artifact_sha256"
    ] = _sha("tampered")
    with pytest.raises(RuntimeError, match="endpoint pair receipt drifted"):
        diagnostic._index_v9_receipts(tampered_evidence)

    tampered_chain = _mock_v9_report()
    tampered_chain["endpoint_pair_receipts"][0][  # type: ignore[index]
        "path_node_receipt_sha256s"
    ][0] = _sha("wrong-node")
    with pytest.raises(RuntimeError, match="finalized evidence node chain"):
        diagnostic._index_v9_receipts(tampered_chain)

    tampered_node = _mock_v9_report()
    tampered_node["path_node_receipts"][0]["backward_call_count"] += 1  # type: ignore[index,operator]
    with pytest.raises(RuntimeError, match="node receipt drifted"):
        diagnostic._index_v9_receipts(tampered_node)


def test_float64_objective_and_direct_delta_use_the_locked_orientation() -> None:
    teacher = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5], [1.0, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    scalar = teacher + torch.tensor([[[0.2, -0.1, 0.0]]], dtype=torch.float32)
    joint = teacher + torch.tensor([[[-0.1, 0.3, 0.0]]], dtype=torch.float32)
    indices = torch.tensor([0, 1], dtype=torch.int64)

    scalar64 = diagnostic._selected_token_teacher_kl_f64(
        teacher, scalar, indices
    )
    joint64 = diagnostic._selected_token_teacher_kl_f64(teacher, joint, indices)
    direct = diagnostic._selected_token_teacher_kl_direct_delta_f64(
        teacher, scalar, joint, indices
    )
    subtraction, maximum, tolerance = diagnostic._direct_endpoint_crosscheck(
        direct=direct,
        scalar_kl=scalar64,
        joint_kl=joint64,
    )
    assert direct.dtype == torch.float64
    assert torch.equal(subtraction, joint64 - scalar64)
    assert maximum <= tolerance
    assert tolerance == pytest.approx(
        diagnostic.DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE
    )

    changed = direct.clone()
    changed[0] += 1.0e-12
    with pytest.raises(RuntimeError, match="direct and endpoint-subtracted"):
        diagnostic._direct_endpoint_crosscheck(
            direct=changed,
            scalar_kl=scalar64,
            joint_kl=joint64,
        )


def test_captured_float64_objective_requires_bitwise_independent_replay() -> None:
    captured = torch.tensor([0.25, -0.5], dtype=torch.float64)
    diagnostic._require_exact_f64_objective(
        captured=captured,
        independent=captured.clone(),
        label="node",
    )
    changed = captured.clone()
    changed[0] = torch.nextafter(changed[0], torch.tensor(float("inf")))
    with pytest.raises(RuntimeError, match="differs exactly"):
        diagnostic._require_exact_f64_objective(
            captured=captured,
            independent=changed,
            label="node",
        )


def test_direct_crosscheck_tolerance_does_not_scale_with_common_kl_offset() -> None:
    scalar = torch.tensor([1.0e12], dtype=torch.float64)
    joint = scalar + 1.0
    direct = torch.tensor([1.0 + 1.0e-12], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="direct and endpoint-subtracted"):
        diagnostic._direct_endpoint_crosscheck(
            direct=direct,
            scalar_kl=scalar,
            joint_kl=joint,
        )


def test_v9_d32_promotes_operands_before_subtraction_and_binds_scalar() -> None:
    scalar = torch.tensor([2.0**-25], dtype=torch.float32)
    joint = torch.tensor([1.0], dtype=torch.float32)
    delta = diagnostic._legacy_v9_f32_delta(
        scalar_token_kl=scalar,
        joint_token_kl=joint,
    )
    assert torch.equal(delta, joint.double() - scalar.double())
    assert not torch.equal(delta, (joint - scalar).double())
    pinned = {
        "additive_prompt_scalars": {
            "finite_joint_minus_scalar_teacher_KL": float(delta.mean())
        }
    }
    assert diagnostic._require_legacy_v9_f32_delta_replay(
        delta=delta, pinned_endpoint=pinned
    ) == float(delta.mean())
    pinned["additive_prompt_scalars"][  # type: ignore[index]
        "finite_joint_minus_scalar_teacher_KL"
    ] += 1.0e-12
    with pytest.raises(RuntimeError, match="did not replay V9"):
        diagnostic._require_legacy_v9_f32_delta_replay(
            delta=delta, pinned_endpoint=pinned
        )


def test_exact_v10_resource_ledger_uses_actual_variable_length_chunks() -> None:
    endpoint = {
        "base_forward_count": 16,
        "native_forward_count": 16,
        "endpoint_token_vjp_forward_count": 16,
        "endpoint_token_vjp_backward_call_count": 109,
        "complete_h4_support_row_count": 819,
    }
    gradient = {
        "gradient_native_forward_count": 8,
        "gradient_candidate_vjp_forward_count": 56,
        "gradient_candidate_vjp_backward_call_count": 385,
    }
    live = {
        "fresh_native_teacher_forward_count": 16,
        "scalar_endpoint_f64_vjp_forward_count": 16,
        "scalar_endpoint_f64_vjp_backward_call_count": 109,
        "joint_boundary_forward_count": 16,
        "path_f64_vjp_forward_count": 64,
        "path_f64_vjp_backward_call_count": 436,
        "path_quadrature_node_count": 64,
    }
    resources = diagnostic._resource_accounting(
        endpoint_resources=endpoint,
        gradient_resources=gradient,
        live_resources=live,
        row_bank_candidate_support_row_executions=2842,
    )
    assert resources["lineage_model_forward_count"] == 112
    assert resources["lineage_backward_call_count"] == 494
    assert resources["fresh_model_forward_count"] == 112
    assert resources["fresh_backward_call_count"] == 545
    assert resources["total_model_forward_count"] == 224
    assert resources["total_backward_call_count"] == 1039
    assert resources["total_candidate_support_row_executions"] == 7756

    with pytest.raises(RuntimeError, match="resource accounting differs"):
        diagnostic._resource_accounting(
            endpoint_resources=endpoint,
            gradient_resources=gradient,
            live_resources={**live, "path_f64_vjp_backward_call_count": 64},
            row_bank_candidate_support_row_executions=2842,
        )


def _comparison(
    *, closure: bool, finite_rms: float = 1.0, transport_relative: float = 0.0
) -> object:
    return SimpleNamespace(
        closure_passed=closure,
        metrics=SimpleNamespace(
            finite_delta_f64_rms=finite_rms,
            relative_rmse_epsilon=1.0e-12,
            transport_relative_rmse_to_finite_f64=transport_relative,
        ),
    )


def test_scientific_classification_is_preregistered_and_noncausal() -> None:
    assert diagnostic._scientific_classification(_comparison(closure=True)) == (
        "high_precision_closure_established_same_a"
    )
    assert diagnostic._scientific_classification(
        _comparison(closure=False, transport_relative=0.01)
    ) == "small_path_transport_live_dtype_or_finite_rounding_supported_same_a"
    assert diagnostic._scientific_classification(
        _comparison(closure=False, transport_relative=0.051)
    ) == "material_path_transport_higher_order_quadrature_earned_same_a"
    assert diagnostic._scientific_classification(
        _comparison(closure=False, finite_rms=0.0, transport_relative=0.0)
    ) == "objective_precision_source_unresolved_same_a"


def test_integrity_failure_is_fail_closed_before_output(tmp_path: Path) -> None:
    output = tmp_path / diagnostic.DEFAULT_OUTPUT.name
    assert not output.exists()
    with pytest.raises(RuntimeError, match="before publication"):
        diagnostic._require_integrity_gate_results(
            {"V9_authenticated": True, "f64_exact": False}
        )
    assert not output.exists()


def test_no_knob_cli_output_suffix_and_pyproject_entry() -> None:
    assert vars(diagnostic.build_parser().parse_args([])) == {}
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(["--output", "elsewhere.json"])
    assert diagnostic.DEFAULT_OUTPUT.name.endswith(
        "candidate-joint-state-objective-precision-gl4-lofo-a-fit16-dev-v10.json"
    )
    pyproject = Path("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-tail-token-fisher-candidate-"
        "joint-state-objective-precision-gl4-v10-a-dev = "
        '"fisher_graph.gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_'
        'joint_state_objective_precision_diagnostic:main"'
    ) in pyproject
