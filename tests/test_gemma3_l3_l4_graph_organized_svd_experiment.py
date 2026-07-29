from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.conditional_spectral_generator import (
    fit_conditional_spectral_generator,
)
import fisher_graph.gemma3_l3_l4_graph_organized_svd_experiment as runner
from fisher_graph.graph_spectral_source_basis import fit_graph_source_bases


_BINDING = "91" * 32
_ORIGINS = (0, 1, 2)
_FIT_ORIGINS = (0, 2)
_BANDS = (0, 2, 4)


def _fixture():
    generator = torch.Generator().manual_seed(703)
    responses = torch.randn(
        4,
        len(_ORIGINS),
        3,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.tensor(
        [0.5, 1.25, 2.0, 0.75],
        dtype=torch.float64,
    )
    graph = fit_graph_source_bases(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        response_binding_sha256=_BINDING,
    )
    base = fit_conditional_spectral_generator(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        4,
        3,
        response_binding_sha256=_BINDING,
    )
    keys, plans = runner.build_graph_organized_plan_set(
        base,
        graph,
        frequency_band_boundaries=_BANDS,
    )
    return responses, scales, graph, base, keys, plans


def _rows():
    generator = torch.Generator().manual_seed(704)
    values = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    multiplicities = torch.tensor(
        [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0],
        dtype=torch.float64,
    )
    counts = {
        "raw_row_count": 14,
        "zero_norm_row_count": 3,
        "nonzero_row_count": 11,
        "unique_nonzero_row_count": 7,
    }
    return values, multiplicities, counts


def test_plan_set_is_deterministic_weight_preserving_and_size_matched() -> None:
    _, _, graph, base, keys, plans = _fixture()
    _, repeated = runner.build_graph_organized_plan_set(
        base,
        graph,
        frequency_band_boundaries=_BANDS,
    )

    assert keys[:2] == ("signed_gfa", "singular_contiguous")
    assert keys[2:] == tuple(
        f"random_size_matched:seed{seed}"
        for seed in runner.DEFAULT_RANDOM_SEEDS
    )
    assert tuple(plan.artifact_sha256 for plan in plans) == tuple(
        plan.artifact_sha256 for plan in repeated
    )
    assert plans[0].pack_counts == (2, 2)
    assert all(plan.pack_counts == plans[0].pack_counts for plan in plans)
    assert all(
        plan.stored_coefficient_count == plans[0].stored_coefficient_count
        for plan in plans
    )
    for origin in _ORIGINS:
        expected = base.weighted_kernel_at_origin(origin)
        for plan in plans:
            torch.testing.assert_close(
                plan.weighted_kernel_at_origin(origin),
                expected,
                atol=3e-12,
                rtol=3e-12,
            )


def test_zero_rows_are_filtered_before_scoring_and_rejected_by_evaluator() -> None:
    row = torch.tensor([1.0, -2.0, 0.5, 0.25], dtype=torch.float64)
    tiny = torch.full_like(row, torch.finfo(torch.float64).tiny)
    batch = SimpleNamespace(
        values=torch.stack((torch.zeros_like(row), row, row, tiny)).unsqueeze(0)
    )
    unique, multiplicities, counts = runner._materialized_rows((batch,))

    assert counts == {
        "raw_row_count": 4,
        "zero_norm_row_count": 1,
        "nonzero_row_count": 3,
        "unique_nonzero_row_count": 2,
    }
    assert any(torch.equal(value, row) for value in unique)
    assert any(torch.equal(value, tiny) for value in unique)
    torch.testing.assert_close(
        torch.sort(multiplicities).values,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
    )

    responses, scales, _, _, _, plans = _fixture()
    dense = responses[:, 1] * scales.view(-1, 1, 1)
    with pytest.raises(
        ValueError,
        match="zero-norm rows must be filtered",
    ):
        runner.evaluate_graph_organized_plan(
            plans[0],
            torch.cat((unique, torch.zeros(1, 4, dtype=torch.float64))),
            torch.ones(3, dtype=torch.float64),
            counts,
            dense_weighted_kernel=dense,
            origin=1,
            role="selection",
            plan_key="signed_gfa",
        )


def test_rate_rows_have_exact_cached_core_accounting_and_safe_bounds() -> None:
    responses, scales, _, _, keys, plans = _fixture()
    rows, multiplicities, counts = _rows()
    dense = responses[:, 1] * scales.view(-1, 1, 1)
    result = runner.evaluate_graph_organized_plan(
        plans[0],
        rows,
        multiplicities,
        counts,
        dense_weighted_kernel=dense,
        origin=1,
        role="selection",
        plan_key=keys[0],
    )

    assert tuple(row["route_fraction"] for row in result) == (
        runner.DEFAULT_ROUTE_FRACTIONS
    )
    assert all(
        row["zero_norm_rows_filtered_before_route_scoring"] is True
        and row["certified_omitted_output_bound_holds"] is True
        and row["router_cost_included_in_cached_core_macs"] is False
        for row in result
    )
    assert [row["mean_active_rank"] for row in result] == sorted(
        row["mean_active_rank"] for row in result
    )
    assert [row["cached_core_factorized_macs"] for row in result] == sorted(
        row["cached_core_factorized_macs"] for row in result
    )
    evaluated = int(sum(multiplicities))
    for row in result:
        active_instances = round(
            float(row["mean_active_rank"]) * evaluated
        )
        expected_source = evaluated * 4 * 4
        expected_core = active_instances * 3 * 3
        expected_dense = evaluated * 4 * 3 * 3
        assert row["source_projection_macs"] == expected_source
        assert row["cached_core_transport_macs"] == expected_core
        assert row["cached_core_factorized_macs"] == (
            expected_source + expected_core
        )
        assert row["dense_measured_response_macs"] == expected_dense
        assert row["cached_core_mac_fraction_vs_dense"] == pytest.approx(
            (expected_source + expected_core) / expected_dense
        )
    all_on = result[-1]
    assert all_on["routed_relative_error_vs_full_svd"] < 2e-8
    assert all_on["routed_cosine_vs_full_svd"] > 1.0 - 2e-12
    assert all_on[
        "routed_relative_error_vs_dense_measured_response"
    ] == pytest.approx(
        all_on["full_svd_relative_error_vs_dense_measured_response"],
        abs=2e-10,
    )


def test_candidate_state_roundtrip_and_rate_row_tamper_rejection() -> None:
    responses, scales, graph, base, keys, plans = _fixture()
    rows, multiplicities, counts = _rows()
    dense = responses[:, 1] * scales.view(-1, 1, 1)
    rate_rows = []
    for role in ("fit", "selection"):
        for key, plan in zip(keys, plans, strict=True):
            rate_rows.extend(
                runner.evaluate_graph_organized_plan(
                    plan,
                    rows,
                    multiplicities,
                    counts,
                    dense_weighted_kernel=dense,
                    origin=1,
                    role=role,
                    plan_key=key,
                )
            )
    candidate = runner.Gemma3GraphOrganizedSVDCandidate(
        source_artifact_file_sha256="01" * 32,
        source_report_file_sha256="02" * 32,
        source_report_payload_sha256="03" * 32,
        source_mapping_artifact_sha256="04" * 32,
        c2_artifact_file_sha256="05" * 32,
        c2_report_file_sha256="06" * 32,
        c2_report_payload_sha256="07" * 32,
        c2_logical_artifact_sha256="08" * 32,
        c2_protocol_sha256="09" * 32,
        c2_calibration_sha256="0a" * 32,
        c2_candidate_set_sha256="0b" * 32,
        binding={"fixture": True},
        model={"fixture": True},
        base_plan=base,
        graph_basis=graph,
        plan_keys=keys,
        plans=plans,
        rate_rows=tuple(rate_rows),
        conclusions={"development_only": True},
    )
    restored = runner.Gemma3GraphOrganizedSVDCandidate.from_state_dict(
        candidate.state_dict()
    )

    assert restored.artifact_sha256 == candidate.artifact_sha256
    assert restored.metadata() == candidate.metadata()

    unknown = copy.deepcopy(candidate.state_dict())
    unknown["future_field"] = True
    with pytest.raises(ValueError, match="state fields differ"):
        runner.Gemma3GraphOrganizedSVDCandidate.from_state_dict(unknown)

    tampered = copy.deepcopy(candidate.state_dict())
    tampered["rate_rows"][0]["mean_active_rank"] = 999.0
    with pytest.raises(ValueError, match="candidate hash mismatch"):
        runner.Gemma3GraphOrganizedSVDCandidate.from_state_dict(tampered)
