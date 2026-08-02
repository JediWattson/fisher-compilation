from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as v1,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded,
)
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


def test_v1_defaults_remain_fixed_while_shared_helpers_accept_expanded_grid() -> None:
    assert v1._TAIL_RANKS == (8, 16, 32, 64, 320)
    assert expanded.EXPANDED_TAIL_RANKS == (64, 96, 128, 160, 192, 256, 320)
    assert v1._validated_tail_ranks(v1._TAIL_RANKS) is v1._TAIL_RANKS
    assert (
        v1._validated_tail_ranks(expanded.EXPANDED_TAIL_RANKS)
        is expanded.EXPANDED_TAIL_RANKS
    )
    assert expanded.DEFAULT_OUTPUT != v1.DEFAULT_OUTPUT
    assert expanded._SCHEMA != v1._SCHEMA


@pytest.mark.parametrize(
    "ranks",
    (
        [64, 320],
        (64, 64, 320),
        (96, 320),
        (64, 321),
        (64, 256),
        (64, True, 320),
    ),
)
def test_fixed_rank_tuple_validation_is_fail_closed(ranks: object) -> None:
    with pytest.raises(ValueError, match="strictly increasing fixed tuple"):
        v1._validated_tail_ranks(ranks)  # type: ignore[arg-type]


def test_finite_provider_accepts_expanded_rank_but_not_boolean_rank() -> None:
    prefix = _prefix()
    base_h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    correction = torch.zeros_like(base_h4, dtype=torch.float64)
    provider = v1._AuthenticatedFiniteTailProvider(
        rank=96,
        fold_artifact_sha256="a" * 64,
        model_inputs_sha256="b" * 64,
        bridge_binding_sha256="c" * 64,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base_h4,
        support_mask=support,
        correction=correction,
    )
    assert provider.rank == 96
    provider.validate_integrity()

    with pytest.raises(ValueError, match=r"rank must be in \[1, 320\]"):
        v1._AuthenticatedFiniteTailProvider(
            rank=True,  # type: ignore[arg-type]
            fold_artifact_sha256="a" * 64,
            model_inputs_sha256="b" * 64,
            bridge_binding_sha256="c" * 64,
            prefix_artifact_sha256=prefix.artifact_sha256,
            base_h4=base_h4,
            support_mask=support,
            correction=correction,
        )


def _parent_fixture() -> dict[str, object]:
    return {
        "schema": v1._SCHEMA,
        "report_sha256": expanded.ADAPTIVE_PARENT_REPORT_SHA256,
        "classification": "tail_endpoint_fisher_finite_ladder_not_supported",
        "passed": False,
        "protocol": {"tail_ranks": list(v1._TAIL_RANKS)},
        "input_binding": {
            "materialization_report_file_sha256": (
                v1.MATERIALIZATION_REPORT_FILE_SHA256
            ),
            "materialization_report_sha256": v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file_sha256": v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": v1.TRANSFER_REPORT_SHA256,
        },
        "fidelity_and_geometry_pass_by_rank": {
            "8": False,
            "16": False,
            "32": False,
            "64": False,
            "320": True,
        },
        "smallest_tail_rank_at_most_64_clearing_established_gates": None,
        "finite_observation_receipts": [],
        "finite_observation_set_sha256": "d" * 64,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "fresh_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_only_hashes_counts_and_scalar_metrics": True,
        },
    }


def test_adaptive_parent_pins_file_report_and_required_failed_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _parent_fixture()

    def fake_load(
        path: object,
        *,
        expected_file_sha256: str,
        expected_report_sha256: str,
        label: str,
    ) -> dict[str, object]:
        assert str(path) == "parent.json"
        assert expected_file_sha256 == expanded.ADAPTIVE_PARENT_REPORT_FILE_SHA256
        assert expected_report_sha256 == expanded.ADAPTIVE_PARENT_REPORT_SHA256
        assert label == "adaptive v1 token-Fisher parent"
        return parent

    monkeypatch.setattr(v1, "_load_pinned_report", fake_load)
    monkeypatch.setattr(
        v1,
        "_finite_observation_set_sha256",
        lambda *args, **kwargs: "d" * 64,
    )
    assert expanded._load_adaptive_parent("parent.json") is parent

    parent["scientific_status"] = {
        **dict(parent["scientific_status"]),  # type: ignore[arg-type]
        "compression_claim": True,
    }
    with pytest.raises(ValueError, match="parent semantics differ"):
        expanded._load_adaptive_parent("parent.json")


def _overlap_rows() -> list[dict[str, object]]:
    return [
        {
            "example_id": f"example-{example_index:02d}",
            "rank": rank,
            "observation_sha256": f"{example_index * 2 + rank // 320:064x}",
        }
        for example_index in range(v1._EXPECTED_EXAMPLES)
        for rank in (64, 320)
    ]


def test_overlap_receipt_requires_exact_rank64_and_rank320_reproduction() -> None:
    rows = _overlap_rows()
    parent = {"finite_observation_receipts": deepcopy(rows)}
    receipt = expanded._overlap_receipt(parent, deepcopy(rows))
    assert len(receipt) == 64

    rows[0]["observation_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="overlap differs from v1"):
        expanded._overlap_receipt(parent, rows)


def test_expanded_classification_reports_smallest_subsentinel_clearing_rank() -> None:
    decisions = {rank: False for rank in expanded.EXPANDED_TAIL_RANKS}
    decisions[160] = True
    decisions[192] = True
    decisions[320] = True

    smallest, classification = expanded._expanded_classification(
        decisions, integrity_gates_passed=True
    )
    assert smallest == 160
    assert classification == (
        "adaptive_same_a_smallest_tail_rank_160_cleared_established_gates"
    )

    smallest, classification = expanded._expanded_classification(
        decisions, integrity_gates_passed=False
    )
    assert smallest == 160
    assert classification.endswith("integrity_gate_failed")

    decisions = {rank: rank == 320 for rank in expanded.EXPANDED_TAIL_RANKS}
    assert expanded._expanded_classification(
        decisions, integrity_gates_passed=True
    ) == (None, "adaptive_same_a_no_tail_rank_below_320_cleared")


def _observation(*, example_id: str, family_id: str, rank: int) -> dict[str, object]:
    row: dict[str, object] = {
        "example_id": example_id,
        "family_id": family_id,
        "rank": rank,
        "native_mean_nll": 1.0,
        "d320_mean_nll": 2.0,
        "candidate_mean_nll": 1.5,
        "endpoint_baseline_mse": 1.0,
        "endpoint_prediction_mse": 0.25,
        "candidate_h4_bitwise_native": rank == 320,
        "candidate_logits_bitwise_native": rank == 320,
        "full_tail_reconstruction_max_abs_error": 0.0 if rank == 320 else None,
    }
    row["observation_sha256"] = v1._domain_sha256(
        row,
        domain=b"fisher-graph:complete-h4-tail-finite-observation:v1\0",
    )
    return row


def test_generalized_observation_grid_and_summary_use_expanded_ranks() -> None:
    observations = [
        _observation(
            example_id=f"example-{family_index}",
            family_id=f"family-{family_index}",
            rank=rank,
        )
        for rank in expanded.EXPANDED_TAIL_RANKS
        for family_index in range(2)
    ]
    receipt = v1._finite_observation_set_sha256(
        observations,
        expected_example_count=2,
        ranks=expanded.EXPANDED_TAIL_RANKS,
    )
    assert len(receipt) == 64
    arms, gates = v1._summarize_observations(
        observations, ranks=expanded.EXPANDED_TAIL_RANKS
    )
    assert tuple(row["tail_rank"] for row in arms) == expanded.EXPANDED_TAIL_RANKS
    assert gates["k320_every_prompt_h4_bitwise_native"] is True


def test_expanded_cli_has_fixed_grid_and_separate_defaults() -> None:
    parser = expanded.build_parser()
    args = parser.parse_args([])
    assert args.adaptive_parent_report == expanded.DEFAULT_ADAPTIVE_PARENT_REPORT
    assert args.output == expanded.DEFAULT_OUTPUT
    assert not any(action.dest == "ranks" for action in parser._actions)
