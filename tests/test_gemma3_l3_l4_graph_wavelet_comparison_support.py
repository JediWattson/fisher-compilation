from __future__ import annotations

from types import MappingProxyType

import pytest
import torch

import fisher_graph.gemma3_l3_l4_graph_wavelet_experiment as parent_experiment
from fisher_graph.gemma3_l3_l4_conditional_spectral_executor_experiment import (
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    SELECTION_ORIGINS,
)
from fisher_graph.gemma3_l3_l4_graph_wavelet_comparison_support import (
    MatchedBasisFamily,
    evaluate_frozen_plan_panel,
    freeze_matched_plan_panel,
    reconstruct_authenticated_q64_fit_context,
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


@pytest.fixture(scope="module")
def fixture():
    generator = torch.Generator(device="cpu").manual_seed(20260802)
    responses = torch.randn(
        (8, len(INTERIOR_ORIGINS), 4, 6),
        generator=generator,
        dtype=torch.float64,
    )
    responses += 0.3 * torch.einsum(
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
    context = reconstruct_authenticated_q64_fit_context(
        responses,
        scales,
        INTERIOR_ORIGINS,
        parent,
        response_binding_sha256=_BINDING,
        expected_graph_basis_artifact_sha256=graph.artifact_sha256,
        source_receipt=_SOURCE_RECEIPT,
        fft_length=4,
        expected_parent_rank=8,
    )
    return responses, scales, graph, parent, context


def _families(context):
    ranks = (4, 5)
    reverse = torch.arange(7, -1, -1)
    control = context.q64.index_select(1, reverse)
    return (
        MatchedBasisFamily(
            method="gomp_prefix",
            source_basis_kind="fit_only_graph_wavelet_gomp",
            bases=MappingProxyType(
                {rank: context.q64[:, :rank] for rank in ranks}
            ),
            construction_receipt={"artifact_sha256": "a" * 64},
        ),
        MatchedBasisFamily(
            method="fixed_control",
            source_basis_kind="fixed_orthonormal_control",
            bases=MappingProxyType(
                {rank: control[:, :rank] for rank in ranks}
            ),
            construction_receipt={"artifact_sha256": "b" * 64},
        ),
    )


def test_context_reconstructs_and_authenticates_parent_q64(fixture) -> None:
    _, _, graph, parent, context = fixture

    assert context.fit_kernels.shape == (8, len(FIT_ORIGINS), 4, 6)
    assert len(context.fit_folds) == len(FIT_ORIGINS)
    assert context.graph.artifact_sha256 == graph.artifact_sha256
    assert context.parent_receipt["artifact_sha256"] == parent.artifact_sha256
    assert context.q64.shape == (8, 8)
    torch.testing.assert_close(
        context.q64.T @ context.q64,
        torch.eye(8, dtype=torch.float64),
    )


def test_panel_freezes_before_selection_and_has_exact_accounting(
    fixture,
) -> None:
    responses, _, _, _, context = fixture
    panel = freeze_matched_plan_panel(context, _families(context))

    assert panel.method_order == ("gomp_prefix", "fixed_control")
    assert panel.ranks == (4, 5)
    assert len(panel.plans) == 4
    assert all(row["heldout_evaluation"] is None for row in panel.row_prefixes)
    rows = evaluate_frozen_plan_panel(
        panel,
        responses,
        INTERIOR_ORIGINS,
    )
    for row in rows:
        rank = int(row["rank"])
        expected = 8 * rank + 6 * 6 + 3 * 4 * rank * 6
        assert row["coefficient_payload"][
            "compiled_plan_float64_scalars"
        ] == expected
        assert row["heldout_evaluation"]["fit_origin_overlap"] == ()


def test_selection_changes_cannot_change_context_or_frozen_plans(
    fixture,
) -> None:
    responses, scales, graph, parent, context = fixture
    changed = responses.clone()
    selection = torch.tensor(
        [INTERIOR_ORIGINS.index(origin) for origin in SELECTION_ORIGINS],
        dtype=torch.int64,
    )
    changed[:, selection] = changed[:, selection] * -2.0 + 0.75
    changed_context = reconstruct_authenticated_q64_fit_context(
        changed,
        scales,
        INTERIOR_ORIGINS,
        parent,
        response_binding_sha256=_BINDING,
        expected_graph_basis_artifact_sha256=graph.artifact_sha256,
        source_receipt=_SOURCE_RECEIPT,
        fft_length=4,
        expected_parent_rank=8,
    )
    first = freeze_matched_plan_panel(context, _families(context))
    second = freeze_matched_plan_panel(
        changed_context,
        _families(changed_context),
    )

    assert first.context.parent_receipt == second.context.parent_receipt
    assert tuple(plan.artifact_sha256 for plan in first.plans) == tuple(
        plan.artifact_sha256 for plan in second.plans
    )
    assert first.row_prefixes == second.row_prefixes
    first_rows = evaluate_frozen_plan_panel(
        first,
        responses,
        INTERIOR_ORIGINS,
    )
    changed_rows = evaluate_frozen_plan_panel(
        second,
        changed,
        INTERIOR_ORIGINS,
    )
    assert any(
        left["heldout_evaluation"]["weighted_relative_error"]
        != right["heldout_evaluation"]["weighted_relative_error"]
        for left, right in zip(first_rows, changed_rows, strict=True)
    )


def test_evaluation_rejects_fit_response_drift(fixture) -> None:
    responses, _, _, _, context = fixture
    panel = freeze_matched_plan_panel(context, _families(context))
    changed = responses.clone()
    changed[:, INTERIOR_ORIGINS.index(FIT_ORIGINS[0])] += 0.01

    with pytest.raises(ValueError, match="fit response values differ"):
        evaluate_frozen_plan_panel(
            panel,
            changed,
            INTERIOR_ORIGINS,
        )
