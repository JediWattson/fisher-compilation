from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_l3_l4_graph_wavelet_experiment as parent_experiment
import fisher_graph.gemma3_l3_l4_graph_wavelet_grouped_comparison_experiment as experiment
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
_GROUP_COUNTS = (2, 4)
_CONTROL_GROUP_COUNT = 2
_PERMUTATION_SEEDS = (1729, 3253)
_TARGET_RANK = 5


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _rehash(state: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in state.items()
        if key != "artifact_sha256"
    }
    state["artifact_sha256"] = experiment._json_sha256(
        payload,
        domain=experiment._ARTIFACT_DOMAIN,
    )


@pytest.fixture(scope="module")
def compiled_pair() -> tuple[
    experiment.Gemma3GraphWaveletGroupedComparisonCandidate,
    experiment.Gemma3GraphWaveletGroupedComparisonCandidate,
]:
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    responses = torch.randn(
        (8, len(INTERIOR_ORIGINS), 4, 6),
        generator=generator,
        dtype=torch.float64,
    )
    responses += 0.35 * torch.einsum(
        "sc,colt->solt",
        torch.randn((8, 3), generator=generator, dtype=torch.float64),
        torch.randn(
            (3, len(INTERIOR_ORIGINS), 4, 6),
            generator=generator,
            dtype=torch.float64,
        ),
    )
    responses = responses.contiguous()
    scales = torch.linspace(0.5, 1.5, 8, dtype=torch.float64)
    graph = fit_graph_source_bases(
        responses,
        scales,
        INTERIOR_ORIGINS,
        FIT_ORIGINS,
        response_binding_sha256=_BINDING,
        fft_length=4,
    )
    parent = parent_experiment._compile_from_response(
        responses,
        scales,
        INTERIOR_ORIGINS,
        response_binding_sha256=_BINDING,
        expected_graph_basis_artifact_sha256=graph.artifact_sha256,
        source_receipt=_SOURCE_RECEIPT,
        fft_length=4,
        packet_budgets=(8,),
        target_rank=6,
    )

    def compile(value: torch.Tensor):
        return experiment._compile_from_response(
            value,
            scales,
            INTERIOR_ORIGINS,
            parent,
            response_binding_sha256=_BINDING,
            expected_graph_basis_artifact_sha256=graph.artifact_sha256,
            source_receipt=_SOURCE_RECEIPT,
            fft_length=4,
            target_source_rank=_TARGET_RANK,
            group_counts=_GROUP_COUNTS,
            control_group_count=_CONTROL_GROUP_COUNT,
            permutation_seeds=_PERMUTATION_SEEDS,
            source_tensor_file_bytes=123,
            parent_tensor_file_bytes=456,
        )

    first = compile(responses)
    changed = responses.clone()
    selection = torch.tensor(
        [INTERIOR_ORIGINS.index(origin) for origin in SELECTION_ORIGINS],
        dtype=torch.int64,
    )
    changed[:, selection] = changed[:, selection] * -2.0 + 0.75
    return first, compile(changed)


def test_describe_declares_zero_model_resources() -> None:
    result = (
        experiment.describe_gemma3_l3_l4_graph_wavelet_grouped_comparison()
    )

    assert tuple(result["protocol"]["fit_origins"]) == FIT_ORIGINS
    assert tuple(result["protocol"]["selection_origins"]) == SELECTION_ORIGINS
    assert result["protocol"]["target_source_rank"] == 45
    assert tuple(result["protocol"]["group_counts"]) == (4, 8, 16)
    assert set(result["resource_contract"].values()) == {0}
    assert result["safety"]["metadata_only_candidate"] is True
    assert result["claims"]["natural_prompt_or_nll_fidelity_measured"] is False


def test_selection_perturbation_cannot_change_construction_or_fit(
    compiled_pair,
) -> None:
    first, changed = compiled_pair

    assert first.source_receipt == changed.source_receipt
    assert first.parent_receipt == changed.parent_receipt
    assert first.protocol == changed.protocol
    assert first.construction_receipts == changed.construction_receipts
    heldout_changed = []
    for left, right in zip(first.rate_rows, changed.rate_rows, strict=True):
        assert left["method"] == right["method"]
        assert left["source_basis_sha256"] == right["source_basis_sha256"]
        assert left["plan_artifact_sha256"] == right["plan_artifact_sha256"]
        assert left["fit_evaluation"] == right["fit_evaluation"]
        assert left["construction_receipt"] == right["construction_receipt"]
        assert left["heldout_evaluation"]["fit_origin_overlap"] == []
        heldout_changed.append(
            left["heldout_evaluation"]["weighted_relative_error"]
            != right["heldout_evaluation"]["weighted_relative_error"]
        )
    assert any(heldout_changed)


def test_method_panel_and_equal_rank_accounting_are_exact(
    compiled_pair,
) -> None:
    candidate, _ = compiled_pair
    expected_methods = experiment._method_order(
        _GROUP_COUNTS,
        control_group_count=_CONTROL_GROUP_COUNT,
        permutation_seeds=_PERMUTATION_SEEDS,
    )

    assert tuple(candidate.protocol["method_order"]) == expected_methods
    assert tuple(row["method"] for row in candidate.rate_rows) == (
        expected_methods
    )
    assert set(candidate.construction_receipts) == set(expected_methods)
    assert len(candidate.rate_rows) == 17
    expected_coefficients = (
        8 * _TARGET_RANK
        + 6 * 6
        + len(FIT_ORIGINS) * 4 * _TARGET_RANK * 6
    )
    payloads = {
        (
            row["coefficient_payload"]["compiled_plan_float64_scalars"],
            row["coefficient_payload"]["prepared_runtime_storage_bytes"],
        )
        for row in candidate.rate_rows
    }
    assert payloads == {(expected_coefficients, 3576)}
    assert all(row["rank"] == _TARGET_RANK for row in candidate.rate_rows)
    assert candidate.resource_accounting["topology_partition_fit_count"] == 6
    assert candidate.resource_accounting["grouped_basis_fit_count"] == 10
    assert candidate.resource_accounting["grouped_basis_lofo_fit_count"] == 30
    assert candidate.resource_accounting["conditional_plan_fit_count"] == 17
    assert candidate.resource_accounting[
        "conditional_plan_heldout_evaluation_count"
    ] == 17
    assert candidate.resource_accounting["fit_response_float64_scalars"] == 576
    assert candidate.resource_accounting[
        "selection_response_float64_scalars"
    ] == 384
    assert candidate.conclusions[
        "all_methods_have_exact_equal_rank_plan_payload"
    ] is True
    assert candidate.conclusions["compute_gate_measured"] is False

    kinds = {
        row["method"]: row["source_basis_kind"]
        for row in candidate.rate_rows
    }
    assert kinds["graph_local_pair_supermode"] == (
        "fit_only_graph_wavelet_local_supermodes"
    )
    assert kinds["signed_local_svd_g2"] == (
        "fit_only_graph_wavelet_local_block_svd"
    )
    assert kinds["signed_cluster_gfa_g2"] == (
        "fit_only_graph_wavelet_cluster_spectral"
    )
    assert kinds["signed_local_svd_g2_one_hot"] == (
        "fixed_orthonormal_control"
    )


def test_metadata_only_roundtrip_publish_and_tamper(
    compiled_pair,
    tmp_path: Path,
) -> None:
    candidate, _ = compiled_pair
    state = candidate.state_dict()
    assert not _contains_tensor(state)
    restored = (
        experiment.Gemma3GraphWaveletGroupedComparisonCandidate.from_state_dict(
            state
        )
    )
    assert restored.metadata() == candidate.metadata()

    output = tmp_path / "grouped.pt"
    report = experiment._publish_candidate(candidate, output=output)
    loaded = experiment.load_gemma3_graph_wavelet_grouped_comparison_candidate(
        output,
        expected_artifact_sha256=candidate.artifact_sha256,
        expected_tensor_file_sha256=report["artifact"]["tensor_file_sha256"],
        expected_report_sha256=report["report_sha256"],
    )
    assert loaded.metadata() == candidate.metadata()

    tampered = deepcopy(state)
    tampered["rate_rows"][0]["coefficient_payload"][  # type: ignore[index]
        "compiled_plan_float64_scalars"
    ] += 1
    _rehash(tampered)
    with pytest.raises(ValueError, match="accounting differs"):
        experiment.Gemma3GraphWaveletGroupedComparisonCandidate.from_state_dict(
            tampered
        )
