from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_l3_l4_graph_wavelet_experiment as parent_experiment
import fisher_graph.gemma3_l3_l4_graph_wavelet_supermode_experiment as experiment
from fisher_graph.gemma3_l3_l4_conditional_spectral_executor_experiment import (
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    SELECTION_ORIGINS,
)
from fisher_graph.graph_spectral_source_basis import fit_graph_source_bases


_BINDING = "7" * 64
_SOURCE_RECEIPT = {
    "tensor_file_sha256": "1" * 64,
    "report_file_sha256": "2" * 64,
    "report_payload_sha256": "3" * 64,
    "mapping_artifact_sha256": "4" * 64,
    "response_artifact_sha256": "5" * 64,
    "source_model_sha256": "6" * 64,
}


def _measurement() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260801)
    responses = torch.randn(
        (64, len(INTERIOR_ORIGINS), 4, 6),
        generator=generator,
        dtype=torch.float64,
    )
    source_mix = torch.randn(
        (64, 12),
        generator=generator,
        dtype=torch.float64,
    )
    shared = torch.randn(
        (12, len(INTERIOR_ORIGINS), 4, 6),
        generator=generator,
        dtype=torch.float64,
    )
    responses += 0.35 * torch.einsum("sc,colt->solt", source_mix, shared)
    scales = torch.linspace(0.5, 1.5, 64, dtype=torch.float64)
    return responses.contiguous(), scales


def _expected_graph_hash(
    responses: torch.Tensor,
    scales: torch.Tensor,
) -> str:
    graph = fit_graph_source_bases(
        responses,
        scales,
        INTERIOR_ORIGINS,
        FIT_ORIGINS,
        response_binding_sha256=_BINDING,
        fft_length=4,
    )
    return graph.artifact_sha256


def _parent_candidate(
    responses: torch.Tensor,
    scales: torch.Tensor,
    graph_hash: str,
) -> parent_experiment.Gemma3GraphWaveletCandidate:
    return parent_experiment._compile_from_response(
        responses,
        scales,
        INTERIOR_ORIGINS,
        response_binding_sha256=_BINDING,
        expected_graph_basis_artifact_sha256=graph_hash,
        source_receipt=_SOURCE_RECEIPT,
        fft_length=4,
        packet_budgets=(64,),
        target_rank=6,
    )


@pytest.fixture(scope="module")
def compiled_pair() -> tuple[
    experiment.Gemma3GraphWaveletSupermodeCandidate,
    experiment.Gemma3GraphWaveletSupermodeCandidate,
]:
    responses, scales = _measurement()
    graph_hash = _expected_graph_hash(responses, scales)
    parent = _parent_candidate(responses, scales, graph_hash)
    first = experiment._compile_from_response(
        responses,
        scales,
        INTERIOR_ORIGINS,
        parent,
        response_binding_sha256=_BINDING,
        expected_graph_basis_artifact_sha256=graph_hash,
        source_receipt=_SOURCE_RECEIPT,
        fft_length=4,
        target_rank=6,
        source_tensor_file_bytes=123,
        parent_tensor_file_bytes=456,
    )
    changed = responses.clone()
    selection_indices = tuple(
        INTERIOR_ORIGINS.index(origin) for origin in SELECTION_ORIGINS
    )
    changed[:, selection_indices] *= -2.0
    changed[:, selection_indices] += 0.75
    second = experiment._compile_from_response(
        changed,
        scales,
        INTERIOR_ORIGINS,
        parent,
        response_binding_sha256=_BINDING,
        expected_graph_basis_artifact_sha256=graph_hash,
        source_receipt=_SOURCE_RECEIPT,
        fft_length=4,
        target_rank=6,
        source_tensor_file_bytes=123,
        parent_tensor_file_bytes=456,
    )
    return first, second


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _rehash_candidate_state(state: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in state.items()
        if key != "artifact_sha256"
    }
    state["artifact_sha256"] = experiment._json_sha256(
        payload,
        domain=experiment._ARTIFACT_DOMAIN,
    )


def _mutable_state(
    candidate: experiment.Gemma3GraphWaveletSupermodeCandidate,
) -> dict[str, object]:
    state = deepcopy(candidate.state_dict())
    assert isinstance(state, dict)
    return state


def _state_row(
    state: dict[str, object],
    method: str,
    rank: int,
) -> dict[str, object]:
    rows = state["rate_rows"]
    assert isinstance(rows, tuple)
    matches = [
        row
        for row in rows
        if row["method"] == method and row["rank"] == rank
    ]
    assert len(matches) == 1
    row = matches[0]
    assert isinstance(row, dict)
    return row


def test_describe_freezes_protocol_without_model_resources() -> None:
    result = experiment.describe_gemma3_l3_l4_graph_wavelet_supermodes()

    assert tuple(result["protocol"]["fit_origins"]) == FIT_ORIGINS
    assert tuple(result["protocol"]["selection_origins"]) == SELECTION_ORIGINS
    assert tuple(result["protocol"]["ranks"]) == tuple(range(45, 53))
    assert tuple(result["protocol"]["method_order"]) == experiment.METHOD_ORDER
    assert result["protocol"]["primary_method"] == "graph_local_merge"
    assert len(result["protocol"]["permuted_graph_local_seeds"]) == 4
    assert set(result["resource_contract"].values()) == {0}
    assert result["safety"]["metadata_only_candidate"] is True


def test_fit_only_paths_bases_plans_and_fit_rows_ignore_selection_values(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    first, changed = compiled_pair

    assert first.parent_receipt == changed.parent_receipt
    assert first.path_receipts == changed.path_receipts
    heldout_changed = []
    for left, right in zip(first.rate_rows, changed.rate_rows, strict=True):
        assert left["method"] == right["method"]
        assert left["rank"] == right["rank"]
        assert left["source_basis_sha256"] == right["source_basis_sha256"]
        assert left["plan_artifact_sha256"] == right["plan_artifact_sha256"]
        assert left["fit_evaluation"] == right["fit_evaluation"]
        assert left["action_prefix_sha256"] == right["action_prefix_sha256"]
        heldout_changed.append(
            left["heldout_evaluation"]["weighted_relative_error"]
            != right["heldout_evaluation"]["weighted_relative_error"]
        )
        assert left["heldout_evaluation"]["fit_origin_overlap"] == []
    assert any(heldout_changed)
    for receipt in first.path_receipts.values():
        assert receipt["heldout_input_used"] is False
        assert len(receipt["action_diagnostics"]) == 19


def test_methods_have_exact_equal_rank_payload_and_honest_basis_kinds(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair

    assert len(candidate.rate_rows) == (
        len(experiment.METHOD_ORDER) * len(experiment.RANKS)
    )
    by_rank: dict[int, list[dict[str, object]]] = {}
    for row in candidate.rate_rows:
        by_rank.setdefault(row["rank"], []).append(row)
    for rank, rows in by_rank.items():
        counts = {
            row["coefficient_payload"]["compiled_plan_float64_scalars"]
            for row in rows
        }
        assert len(counts) == 1
        expected = 64 * rank + 6 * 6 + 3 * 4 * rank * 6
        assert counts == {expected}
        assert all(
            row["coefficient_payload"]["equal_rank_payload_match"]
            for row in rows
        )
    kinds = {
        row["method"]: row["source_basis_kind"]
        for row in candidate.rate_rows
        if row["rank"] == 45
    }
    assert kinds["response_only_merge"] == (
        "fit_only_graph_wavelet_response_only_supermodes"
    )
    assert kinds["graph_local_merge"] == (
        "fit_only_graph_wavelet_local_supermodes"
    )
    for method in experiment.PERMUTED_METHODS:
        assert kinds[method] == (
            "fit_only_graph_wavelet_permuted_topology_supermode_control"
        )
    assert kinds["graph_local_one_hot"] == "fixed_orthonormal_control"
    assert candidate.conclusions["parent_q64_reconstructed_and_verified"] is (
        True
    )
    assert candidate.resource_accounting["model_forward_count"] == 0


def test_primary_rows_report_sse_recovery_and_remain_fail_closed(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair
    primary = tuple(
        row
        for row in candidate.rate_rows
        if row["method"] == "graph_local_merge"
    )

    assert len(primary) == len(experiment.RANKS)
    for row in primary:
        assert len(row["per_origin_sse_recovery_vs_equal_rank_gomp"]) == 2
        assert row["minimum_required_sse_recovery"] == 0.05
        assert row["active_genuine_merge_count"] >= 0
        assert row["passes_compute_gate"] is False
        assert row["passes_controlled_compression_gate"] is False
    assert candidate.conclusions["selected_rank"] is None
    assert candidate.conclusions["controlled_compression_passing_ranks"] == []


def test_rehashed_gate_and_nominee_tampering_fails_semantic_validation(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair

    state = _mutable_state(candidate)
    row = _state_row(state, "gomp_prefix", 45)
    row["passes_fidelity_gate"] = not row["passes_fidelity_gate"]
    _rehash_candidate_state(state)
    with pytest.raises(ValueError, match="passes_fidelity_gate"):
        experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(state)

    state = _mutable_state(candidate)
    primary = _state_row(state, "graph_local_merge", 45)
    primary["passes_merge_recovery_gate"] = not primary[
        "passes_merge_recovery_gate"
    ]
    _rehash_candidate_state(state)
    with pytest.raises(ValueError, match="passes_merge_recovery_gate"):
        experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(state)

    state = _mutable_state(candidate)
    conclusions = state["conclusions"]
    assert isinstance(conclusions, dict)
    actual_nominee = conclusions["development_nominee_rank"]
    fake_nominee = next(
        rank for rank in experiment.RANKS if rank != actual_nominee
    )
    conclusions["development_nominee_rank"] = fake_nominee
    conclusions["development_nominee_plan_artifact_sha256"] = _state_row(
        state,
        "graph_local_merge",
        fake_nominee,
    )["plan_artifact_sha256"]
    _rehash_candidate_state(state)
    with pytest.raises(ValueError, match="conclusions differ"):
        experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(state)


def test_rehashed_path_and_resource_tampering_fails_semantic_validation(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair

    state = _mutable_state(candidate)
    row = _state_row(state, "graph_local_merge", 45)
    row["path_artifact_sha256"] = "f" * 64
    _rehash_candidate_state(state)
    with pytest.raises(ValueError, match="row path binding differs"):
        experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(state)

    state = _mutable_state(candidate)
    row = _state_row(state, "graph_local_merge", 45)
    row["active_action_count"] = int(row["active_action_count"]) + 1
    _rehash_candidate_state(state)
    with pytest.raises(ValueError, match="row path binding differs"):
        experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(state)

    state = _mutable_state(candidate)
    accounting = state["resource_accounting"]
    assert isinstance(accounting, dict)
    accounting["conditional_plan_fit_count"] = (
        int(accounting["conditional_plan_fit_count"]) - 1
    )
    _rehash_candidate_state(state)
    with pytest.raises(ValueError, match="resource accounting differs"):
        experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(state)


def test_one_hot_path_counts_preserve_paired_delete_semantics(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair
    one_hot = next(
        row
        for row in candidate.rate_rows
        if row["method"] == "graph_local_one_hot"
        and row["active_selected_pair_action_count"] > 0
    )

    assert one_hot["active_genuine_merge_count"] == 0
    assert one_hot["active_paired_delete_count"] == (
        one_hot["active_selected_pair_action_count"]
    )

    state = _mutable_state(candidate)
    row = _state_row(state, "graph_local_one_hot", int(one_hot["rank"]))
    row["active_genuine_merge_count"] = row[
        "active_selected_pair_action_count"
    ]
    _rehash_candidate_state(state)
    with pytest.raises(ValueError, match="row path binding differs"):
        experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(state)


def test_rehashed_prepared_storage_formula_tampering_fails(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair
    state = _mutable_state(candidate)
    row = _state_row(state, "graph_local_merge", 45)
    accounting = row["plan_accounting"]
    payload = row["coefficient_payload"]
    protocol = state["protocol"]
    assert isinstance(accounting, dict)
    assert isinstance(payload, dict)
    assert isinstance(protocol, dict)
    fake_bytes = int(accounting["prepared_storage_bytes"]) + 8
    accounting["prepared_storage_bytes"] = fake_bytes
    payload["prepared_runtime_storage_bytes"] = fake_bytes
    full_scalars = (
        64 * 64
        + int(protocol["target_modes"]) ** 2
        + len(FIT_ORIGINS)
        * int(protocol["lag_count"])
        * 64
        * int(protocol["target_modes"])
    )
    full_prepared_bytes = (
        (full_scalars + 64) * 8 + len(FIT_ORIGINS) * 8
    )
    payload["prepared_runtime_storage_fraction_of_full_rank"] = (
        fake_bytes / full_prepared_bytes
    )
    _rehash_candidate_state(state)
    with pytest.raises(ValueError, match="plan accounting"):
        experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(state)


def test_rehashed_source_basis_kind_tampering_fails(
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair
    for method, dishonest_kind in (
        (
            "response_only_merge",
            "fit_only_graph_wavelet_local_supermodes",
        ),
        (
            experiment.PERMUTED_METHODS[0],
            "fit_only_graph_wavelet_local_supermodes",
        ),
    ):
        state = _mutable_state(candidate)
        row = _state_row(state, method, 45)
        row["source_basis_kind"] = dishonest_kind
        _rehash_candidate_state(state)
        with pytest.raises(ValueError, match="source kind differs"):
            experiment.Gemma3GraphWaveletSupermodeCandidate.from_state_dict(
                state
            )


def test_wrong_graph_provenance_fails_closed() -> None:
    responses, scales = _measurement()
    graph_hash = _expected_graph_hash(responses, scales)
    parent = _parent_candidate(responses, scales, graph_hash)

    with pytest.raises(
        ValueError,
        match="supermode graph provenance differs",
    ):
        experiment._compile_from_response(
            responses,
            scales,
            INTERIOR_ORIGINS,
            parent,
            response_binding_sha256=_BINDING,
            expected_graph_basis_artifact_sha256="f" * 64,
            source_receipt=_SOURCE_RECEIPT,
            fft_length=4,
            target_rank=6,
        )


def test_metadata_candidate_round_trips_with_strict_hashes(
    tmp_path: Path,
    compiled_pair: tuple[
        experiment.Gemma3GraphWaveletSupermodeCandidate,
        experiment.Gemma3GraphWaveletSupermodeCandidate,
    ],
) -> None:
    candidate, _ = compiled_pair
    output = tmp_path / "supermode.pt"
    report = experiment._publish_candidate(candidate, output=output)
    raw = torch.load(output, map_location="cpu", weights_only=True)

    assert not _contains_tensor(raw)
    assert raw["safety"]["contains_raw_response_tensors"] is False
    assert raw["safety"]["contains_compiled_plan_tensors"] is False
    restored = experiment.load_gemma3_graph_wavelet_supermode_candidate(
        output,
        expected_artifact_sha256=candidate.artifact_sha256,
        expected_tensor_file_sha256=report["artifact"]["tensor_file_sha256"],
        expected_report_sha256=report["report_sha256"],
    )
    assert restored.artifact_sha256 == candidate.artifact_sha256
    assert restored.metadata() == candidate.metadata()
