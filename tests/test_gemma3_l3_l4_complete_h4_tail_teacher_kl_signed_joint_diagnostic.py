from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as diagnostic,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as v1,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded,
)
from fisher_graph.complete_h4_tail_signed_joint_projector import (
    fit_complete_h4_tail_signed_joint_held_family,
)
from fisher_graph.complete_h4_tail_token_fisher import CompleteH4TailEndpointExample
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)


def _prefix() -> Gemma3L3L4OnePassPrefix:
    return Gemma3L3L4OnePassPrefix(
        source_modes=torch.zeros((1, 3, 2), dtype=torch.float64),
        clamped_y3=torch.zeros((1, 3, 640), dtype=torch.float32),
        predicted_target_modal_delta=torch.zeros((1, 3, 2), dtype=torch.float64),
        decoded_base_x4_delta=torch.zeros((1, 3, 640), dtype=torch.float64),
        logical_positions=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        valid_target_mask=torch.tensor([[True, True, True]]),
        source_eligible_mask=torch.tensor([[False, True, False]]),
        target_affected_mask=torch.tensor([[False, True, False]]),
        bridge_binding_sha256="c" * 64,
    )


def _endpoint(example_id: str, family_id: str, *, seed: int) -> CompleteH4TailEndpointExample:
    generator = torch.Generator().manual_seed(seed)
    return CompleteH4TailEndpointExample(
        example_id=example_id,
        family_id=family_id,
        residual_rows=torch.randn((2, 640), generator=generator, dtype=torch.float64),
        token_h4_gradients=torch.randn(
            (3, 2, 640), generator=generator, dtype=torch.float64
        ),
        compensation_target=torch.randn((3,), generator=generator, dtype=torch.float64),
    )


def test_canonical_support_grid_is_batch_major_and_fail_closed() -> None:
    indices = torch.tensor([1, 4, 9], dtype=torch.int64)
    assert torch.equal(
        diagnostic._canonical_support_grid(indices),
        torch.tensor([[0, 1], [0, 4], [0, 9]], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="not canonical"):
        diagnostic._canonical_support_grid(torch.tensor([4, 1], dtype=torch.int64))


def test_teacher_kl_helper_is_zero_at_identity_and_nonnegative() -> None:
    teacher = torch.tensor([[[1.0, -1.0], [0.0, 2.0]]])
    indices = torch.tensor([0, 1], dtype=torch.int64)
    identity = diagnostic._selected_token_teacher_kl(teacher, teacher, indices)
    assert torch.equal(identity, torch.zeros(2, dtype=torch.float64))
    candidate = torch.zeros_like(teacher)
    value = diagnostic._selected_token_teacher_kl(teacher, candidate, indices)
    assert bool((value >= 0.0).all())


def test_signed_provider_binds_distinct_method_semantics_and_single_use() -> None:
    prefix = _prefix()
    base_h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    correction = torch.zeros_like(base_h4, dtype=torch.float64)
    signed = diagnostic._AuthenticatedSignedJointFiniteProvider(
        rank=8,
        arm_kind="teacher_kl_signed_joint_rotated_gain_prefix",
        fold_artifact_sha256="a" * 64,
        model_inputs_sha256="b" * 64,
        bridge_binding_sha256="c" * 64,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base_h4,
        support_mask=support,
        correction=correction,
    )
    pca = diagnostic._AuthenticatedSignedJointFiniteProvider(
        rank=8,
        arm_kind="teacher_kl_fixed_pca_q2_signed_gain_prefix",
        fold_artifact_sha256="a" * 64,
        model_inputs_sha256="b" * 64,
        bridge_binding_sha256="c" * 64,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base_h4,
        support_mask=support,
        correction=correction,
    )
    assert signed.artifact_sha256 != pca.artifact_sha256
    assert "signed_joint_rotated" in str(signed.metadata()["correction_semantics"])
    assert "fixed_residual_PCA" in str(pca.metadata()["correction_semantics"])
    assert torch.equal(signed.correction(prefix, base_h4), correction)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        signed.correction(prefix, base_h4)
    with pytest.raises(ValueError, match="arm kind differs"):
        diagnostic._AuthenticatedSignedJointFiniteProvider(
            rank=320,
            arm_kind="teacher_kl_signed_joint_rotated_gain_prefix",
            fold_artifact_sha256="a" * 64,
            model_inputs_sha256="b" * 64,
            bridge_binding_sha256="c" * 64,
            prefix_artifact_sha256=prefix.artifact_sha256,
            base_h4=base_h4,
            support_mask=support,
            correction=correction,
        )


def test_k320_endpoint_prediction_contracts_only_complement_tail() -> None:
    endpoint = CompleteH4TailEndpointExample(
        example_id="example",
        family_id="family",
        residual_rows=torch.ones((2, 640), dtype=torch.float64),
        token_h4_gradients=torch.ones((1, 2, 640), dtype=torch.float64),
        compensation_target=torch.zeros(1, dtype=torch.float64),
    )
    tail = torch.zeros((2, 640), dtype=torch.float64)
    tail[:, 3] = torch.tensor([2.0, 5.0])
    prediction = diagnostic._endpoint_prediction(
        endpoint, object(), rank=320, tail_rows=tail  # type: ignore[arg-type]
    )
    assert torch.equal(prediction, torch.tensor([7.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="requires tail rows"):
        diagnostic._endpoint_prediction(
            endpoint, object(), rank=320  # type: ignore[arg-type]
        )


def test_early_stop_is_an_at_most_k_prefix_not_zero_gain_padding() -> None:
    supported = torch.eye(640, dtype=torch.float64)[:638]
    examples = tuple(
        CompleteH4TailEndpointExample(
            example_id=f"e{index}",
            family_id=f"f{index}",
            residual_rows=torch.zeros((1, 640), dtype=torch.float64),
            token_h4_gradients=torch.zeros((1, 1, 640), dtype=torch.float64),
            compensation_target=torch.zeros(1, dtype=torch.float64),
        )
        for index in range(3)
    )
    fit = fit_complete_h4_tail_signed_joint_held_family(
        examples, supported_basis=supported, held_family_id="f2", max_directions=64
    )
    assert fit.rank == 0
    tail = torch.randn((1, 640), dtype=torch.float64)
    assert torch.equal(
        diagnostic._signed_tail_prefix(tail, fit, rank=64),
        torch.zeros_like(tail),
    )
    assert torch.equal(
        diagnostic._endpoint_prediction(examples[-1], fit, rank=64),
        torch.zeros(1, dtype=torch.float64),
    )


def test_fixed_pca_signed_control_is_input_reorder_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directions = torch.eye(640, dtype=torch.float64)[:320]
    fake = SimpleNamespace(
        artifact_sha256="d" * 64,
        ordered_basis_rows=lambda: directions,
    )
    monkeypatch.setattr(
        diagnostic,
        "fit_complete_h4_tail_held_family",
        lambda *args, **kwargs: fake,
    )
    examples = tuple(
        _endpoint(f"e{index}", f"f{index // 2}", seed=index)
        for index in range(8)
    )
    first = diagnostic._fit_fixed_pca_signed_control(
        examples, supported_basis=directions, held_family_id="f3"
    )
    second = diagnostic._fit_fixed_pca_signed_control(
        tuple(reversed(examples)),
        supported_basis=directions,
        held_family_id="f3",
    )
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.gains == second.gains


def _observation(example: str, method: str, rank: int) -> dict[str, object]:
    arm_kind = {
        "signed_joint": "teacher_kl_signed_joint_rotated_gain_prefix",
        "fixed_pca_diagonal": "teacher_kl_fixed_pca_q2_signed_gain_prefix",
        "shared_exact_sentinel": "ungained_exact_complement_sentinel",
    }[method]
    row: dict[str, object] = {
        "example_id": example,
        "family_id": f"family-{example}",
        "method": method,
        "rank": rank,
        "requested_rank": rank,
        "rank_semantics": "at_most_direction_budget",
        "effective_direction_count": rank,
        "arm_kind": arm_kind,
    }
    row["observation_sha256"] = v1._domain_sha256(
        row, domain=diagnostic._OBSERVATION_DOMAIN
    )
    return row


def test_observation_set_is_complete_tamper_evident_and_reorder_invariant() -> None:
    rows = [
        _observation(example, method, rank)
        for example in ("e0", "e1")
        for method, ranks in (
            ("signed_joint", (8, 16, 32, 64)),
            ("fixed_pca_diagonal", (8, 16, 32, 64)),
            ("shared_exact_sentinel", (320,)),
        )
        for rank in ranks
    ]
    first = diagnostic._finite_observation_set_sha256(
        rows, expected_example_count=2
    )
    second = diagnostic._finite_observation_set_sha256(
        list(reversed(rows)), expected_example_count=2
    )
    assert first == second
    tampered = deepcopy(rows)
    tampered[0]["family_id"] = "changed"
    with pytest.raises(RuntimeError, match="receipt drifted"):
        diagnostic._finite_observation_set_sha256(
            tampered, expected_example_count=2
        )


def test_control_loader_delegates_to_full_v1_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = {"report_sha256": diagnostic.CONTROL_REPORT_SHA256}
    monkeypatch.setattr(v1, "_load_pinned_report", lambda *args, **kwargs: parent)
    monkeypatch.setattr(expanded, "_load_adaptive_parent", lambda path: parent)
    assert diagnostic._load_control_report("control.json") is parent


def test_expanded_control_loader_checks_parent_and_observation_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "schema": expanded._SCHEMA,
        "classification": "adaptive_same_a_smallest_tail_rank_256_cleared_established_gates",
        "protocol": {"tail_ranks": expanded.EXPANDED_TAIL_RANKS},
        "smallest_tail_rank_below_320_clearing_established_fidelity_and_geometry_gates": 256,
        "input_binding": {
            "materialization_report_file_sha256": v1.MATERIALIZATION_REPORT_FILE_SHA256,
            "transfer_report_file_sha256": v1.TRANSFER_REPORT_FILE_SHA256,
        },
        "adaptive_parent_binding": {
            "file_sha256": diagnostic.CONTROL_REPORT_FILE_SHA256,
            "report_sha256": diagnostic.CONTROL_REPORT_SHA256,
        },
        "scientific_status": {
            "same_a_adaptive_hypothesis_use_only": True,
            "fresh_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
        },
        "finite_observation_receipts": [{}],
        "finite_observation_set_sha256": "e" * 64,
    }
    monkeypatch.setattr(v1, "_load_pinned_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(
        v1, "_finite_observation_set_sha256", lambda *args, **kwargs: "e" * 64
    )
    assert diagnostic._load_expanded_control_report("expanded.json") is report


def test_cli_has_fixed_separate_output_and_no_rank_override() -> None:
    parser = diagnostic.build_parser()
    args = parser.parse_args([])
    assert args.control_report == diagnostic.DEFAULT_CONTROL_REPORT
    assert args.expanded_control_report == diagnostic.DEFAULT_EXPANDED_CONTROL_REPORT
    assert args.output == diagnostic.DEFAULT_OUTPUT
    assert not any(action.dest == "ranks" for action in parser._actions)
