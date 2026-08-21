from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_joint_cross_block_chart_executor import (
    build_joint_cross_block_event_schedule,
)
from fisher_graph.gemma3_unified_event_chart_sidecar import (
    UnifiedEventChartPacket,
    UnifiedEventChartRecorder,
    UnifiedEventLocalBaseline,
    apply_fitted_unified_event_local_baseline,
    evaluate_unified_event_edge,
    evaluate_fitted_unified_event_local_baseline,
    fit_fisher_weighted_unified_event_local_baseline,
    fit_forward_unified_event_dense_edge,
    fit_forward_unified_event_edge,
    fit_forward_unified_event_edges,
    local_baseline_from_unified_event_output,
    merge_unified_event_chart_captures,
    packet_from_unified_event_site,
)


_SOURCE = "a" * 64


def _sha(number: int) -> str:
    return f"{number:064x}"


def _event(event_id: str):
    schedule = build_joint_cross_block_event_schedule()
    return next(item for item in schedule if item.event_id == event_id)


def _packet(
    event_id: str,
    coordinates: torch.Tensor,
    *,
    role: str = "output",
    execution: str = _sha(1),
    row_prefix: str = "fit",
) -> UnifiedEventChartPacket:
    event = _event(event_id)
    rows = coordinates.shape[0]
    return UnifiedEventChartPacket(
        event_id=event.event_id,
        event_index=event.event_index,
        event_kind=event.event_kind,
        component_family=event.component_family,
        block_ordinal=event.block_ordinal,
        site_id=f"{event.event_id}.{role}",
        site_role=role,
        row_keys=tuple((row_prefix, index) for index in range(rows)),
        coordinates=coordinates.detach().cpu().clone().contiguous(),
        source_fingerprint=_SOURCE,
        execution_fingerprint=execution,
    )


def _baseline(
    event_id: str,
    *,
    target_flow: torch.Tensor,
    local_flow: torch.Tensor,
    memberships: torch.Tensor,
    execution: str = _sha(1),
    row_prefix: str = "fit",
) -> UnifiedEventLocalBaseline:
    event = _event(event_id)
    rows = target_flow.shape[0]
    return UnifiedEventLocalBaseline(
        event_id=event.event_id,
        event_index=event.event_index,
        event_kind=event.event_kind,
        component_family=event.component_family,
        block_ordinal=event.block_ordinal,
        row_keys=tuple((row_prefix, index) for index in range(rows)),
        target_flow=target_flow.detach().cpu().clone().contiguous(),
        local_predicted_flow=local_flow.detach().cpu().clone().contiguous(),
        target_memberships=memberships.detach().cpu().clone().contiguous(),
        row_fisher_weights=torch.linspace(0.5, 1.5, rows),
        source_fingerprint=_SOURCE,
        execution_fingerprint=execution,
    )


def _synthetic_pair(
    *,
    rows: int,
    execution: str,
    row_prefix: str,
    seed: int,
) -> tuple[UnifiedEventChartPacket, UnifiedEventLocalBaseline]:
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.randn(rows, 2, 3, generator=generator)
    occupancy = torch.sigmoid(torch.randn(rows, generator=generator))
    memberships = torch.stack((occupancy, 1.0 - occupancy), dim=1)
    left = torch.tensor([[1.2], [-0.7], [0.4]])
    right = torch.tensor([[0.8, -1.1, 0.3, 0.6, -0.2]])
    design = coordinates[:, 0, :] * memberships[:, 1, None]
    local = 0.05 * torch.randn(rows, 5, generator=generator)
    target = local + design @ left @ right
    return (
        _packet(
            "mutation.1",
            coordinates,
            execution=execution,
            row_prefix=row_prefix,
        ),
        _baseline(
            "mutation.4",
            target_flow=target,
            local_flow=local,
            memberships=memberships,
            execution=execution,
            row_prefix=row_prefix,
        ),
    )


def _capture(
    *,
    execution: str,
    row_prefix: str,
    valid_mask: torch.Tensor,
    hidden_width: int,
    target_inputs: torch.Tensor | None = None,
    target_outputs: torch.Tensor | None = None,
    offset: float = 0.0,
):
    schedule = build_joint_cross_block_event_schedule()
    rows = int(valid_mask.sum().item())
    recorder = UnifiedEventChartRecorder(
        schedule=schedule,
        source_fingerprint=_SOURCE,
        execution_fingerprint=execution,
        valid_mask=valid_mask,
        row_keys=tuple((row_prefix, index) for index in range(rows)),
    )
    for event in schedule:
        roles = (
            ("output",)
            if event.event_kind == "lexical_ingress"
            else (
                ("input", "output")
                if event.event_kind in {"attention_mutation", "mlp_mutation"}
                else ()
            )
        )
        for role in roles:
            valid_values = torch.full(
                (rows, hidden_width),
                offset + float(event.event_index),
            )
            if event.event_id == "mutation.4" and role == "input":
                assert target_inputs is not None
                valid_values = target_inputs
            if event.event_id == "mutation.4" and role == "output":
                assert target_outputs is not None
                valid_values = target_outputs
            padded = torch.zeros(
                (*valid_mask.shape, hidden_width),
                dtype=valid_values.dtype,
            )
            padded[valid_mask] = valid_values
            recorder.record(event_id=event.event_id, role=role, values=padded)
    return recorder.finish()


def test_recorder_closes_exact_73_site_detached_clone_inventory() -> None:
    schedule = build_joint_cross_block_event_schedule()
    valid_mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    row_keys = (("a", 0), ("a", 1), ("b", 0))
    recorder = UnifiedEventChartRecorder(
        schedule=schedule,
        source_fingerprint=_SOURCE,
        execution_fingerprint=_sha(2),
        valid_mask=valid_mask,
        row_keys=row_keys,
    )
    first_source = torch.full((2, 3, 5), 7.0, requires_grad=True)
    assert (
        recorder.record(
            event_id="lexical.ingress",
            role="output",
            values=first_source,
        )
        is None
    )
    with torch.no_grad():
        first_source.fill_(99.0)
    for event in schedule[1:-2]:
        for role in ("input", "output"):
            value = torch.full(
                (2, 3, 5),
                float(event.event_index + (role == "output")),
                requires_grad=True,
            )
            assert recorder(event_id=event.event_id, role=role, values=value) is None

    capture = recorder.finish()
    assert recorder.closed is True
    assert len(capture.observations) == 73
    assert capture.hidden_width == 5
    assert capture.metadata()["observational_sidecar_only"] is True
    lexical = capture.site("lexical.ingress.output")
    assert lexical.value_semantics == "lexical_hidden_state"
    assert lexical.values.requires_grad is False
    assert lexical.values.device.type == "cpu"
    assert torch.equal(lexical.valid_rows(), torch.full((3, 5), 7.0))
    attention_input = capture.site("mutation.1.input")
    attention_output = capture.site("mutation.1.output")
    assert attention_input.value_semantics == "normalized_mutation_input"
    assert attention_output.value_semantics == "mutation_residual_delta"
    with pytest.raises(RuntimeError, match="closed"):
        recorder.record(event_id="mutation.1", role="input", values=first_source)


def test_capture_packet_and_local_baseline_bind_sites_without_runtime_writeback() -> None:
    schedule = build_joint_cross_block_event_schedule()
    mask = torch.tensor([[True, True]], dtype=torch.bool)
    recorder = UnifiedEventChartRecorder(
        schedule=schedule,
        source_fingerprint=_SOURCE,
        execution_fingerprint=_sha(3),
        valid_mask=mask,
        row_keys=(("x", 0), ("x", 1)),
    )
    for event in schedule:
        roles = (
            ("output",)
            if event.event_kind == "lexical_ingress"
            else (
                ("input", "output")
                if event.event_kind in {"attention_mutation", "mlp_mutation"}
                else ()
            )
        )
        for role in roles:
            recorder.record(
                event_id=event.event_id,
                role=role,
                values=torch.full((1, 2, 4), float(event.event_index + 1)),
            )
    capture = recorder.finish()
    projected = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    source = packet_from_unified_event_site(
        capture,
        event_id="mutation.1",
        role="input",
        coordinates=projected,
    )
    assert source.site_id == "mutation.1.input"
    assert source.coordinate_rank == 2
    exact_target = capture.event_site("mutation.4", "output").valid_rows()
    baseline = local_baseline_from_unified_event_output(
        capture,
        event_id="mutation.4",
        local_predicted_flow=torch.zeros_like(exact_target),
    )
    assert baseline.event_id == "mutation.4"
    assert baseline.target_memberships.shape == (2, 1)
    assert torch.equal(baseline.target_flow, exact_target)


def test_merge_captures_packs_valid_rows_and_preserves_role_identity() -> None:
    first = _capture(
        execution=_sha(30),
        row_prefix="first",
        valid_mask=torch.tensor([[True, False, True]]),
        hidden_width=3,
        target_inputs=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        target_outputs=torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        offset=1.0,
    )
    second = _capture(
        execution=_sha(30),
        row_prefix="second",
        valid_mask=torch.tensor([[False, True]]),
        hidden_width=3,
        target_inputs=torch.tensor([[13.0, 14.0, 15.0]]),
        target_outputs=torch.tensor([[16.0, 17.0, 18.0]]),
        offset=2.0,
    )
    merged = merge_unified_event_chart_captures((first, second))
    assert merged.valid_mask.shape == (1, 3)
    assert bool(merged.valid_mask.all()) is True
    assert merged.row_keys == (("first", 0), ("first", 1), ("second", 0))
    assert torch.equal(
        merged.site("mutation.4.input").valid_rows(),
        torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [13.0, 14.0, 15.0]]
        ),
    )
    mismatched = _capture(
        execution=_sha(31),
        row_prefix="third",
        valid_mask=torch.tensor([[True]]),
        hidden_width=3,
        target_inputs=torch.tensor([[1.0, 1.0, 1.0]]),
        target_outputs=torch.tensor([[2.0, 2.0, 2.0]]),
    )
    with pytest.raises(ValueError, match="schedule/source/execution"):
        merge_unified_event_chart_captures(
            (first, mismatched)
        )


def test_full_width_fisher_local_baseline_fits_bias_and_applies_heldout() -> None:
    generator = torch.Generator().manual_seed(13)
    true_weight = torch.tensor(
        [[0.7, -0.2, 0.4], [-0.3, 0.8, 0.1], [0.5, 0.2, -0.6]]
    )
    true_bias = torch.tensor([1.25, -0.75, 0.5])
    fit_inputs = torch.randn(48, 3, generator=generator)
    fit_outputs = fit_inputs @ true_weight.T + true_bias
    fit_capture = _capture(
        execution=_sha(32),
        row_prefix="local-fit",
        valid_mask=torch.ones((1, 48), dtype=torch.bool),
        hidden_width=3,
        target_inputs=fit_inputs,
        target_outputs=fit_outputs,
    )
    fit = fit_fisher_weighted_unified_event_local_baseline(
        capture=fit_capture,
        event_id="mutation.4",
        row_fisher_weights=torch.linspace(0.4, 1.6, 48),
        relative_ridge=1.0e-9,
    )
    assert fit.metadata()["bias_penalized"] is False
    assert torch.allclose(fit.bias, true_bias, atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(fit.weight, true_weight, atol=2.0e-6, rtol=2.0e-6)

    held_inputs = torch.randn(24, 3, generator=generator)
    held_outputs = held_inputs @ true_weight.T + true_bias
    held_capture = _capture(
        execution=_sha(33),
        row_prefix="local-held",
        valid_mask=torch.ones((2, 12), dtype=torch.bool),
        hidden_width=3,
        target_inputs=held_inputs,
        target_outputs=held_outputs,
    )
    frozen_baseline = apply_fitted_unified_event_local_baseline(
        fit,
        capture=held_capture,
    )
    assert torch.allclose(
        frozen_baseline.local_predicted_flow,
        held_outputs,
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    evaluation = evaluate_fitted_unified_event_local_baseline(
        fit,
        capture=held_capture,
    )
    assert evaluation.explained_fraction > 0.999999999
    assert evaluation.fit_rows_disjoint is True
    assert evaluation.fit_execution_disjoint is True
    assert evaluation.metadata()["heldout_refit_performed"] is False
    with pytest.raises(ValueError, match="reused fit"):
        evaluate_fitted_unified_event_local_baseline(
            fit,
            capture=fit_capture,
        )


def test_rectangular_rank_one_ridge_fits_only_residual_beyond_local_chart() -> None:
    source, target = _synthetic_pair(
        rows=96,
        execution=_sha(4),
        row_prefix="fit",
        seed=19,
    )
    edge = fit_forward_unified_event_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=source,
        target=target,
        edge_rank=1,
        relative_ridge=1.0e-8,
        source_chart_index=0,
        target_chart_index=1,
    )
    prediction = edge.predict(source, target.target_memberships)
    assert edge.left_factor.shape == (3, 1)
    assert edge.right_factor.shape == (1, 5)
    assert edge.spec.cross_block is True
    assert edge.spec.cross_family is True
    assert edge.spec.source_event_id == "mutation.1"
    assert edge.spec.target_event_id == "mutation.4"
    assert edge.fit_explained_fraction > 0.999999
    assert torch.allclose(prediction, target.residual_flow, atol=2.0e-5, rtol=2.0e-5)
    metadata = edge.metadata()
    assert metadata["event_or_layer_features_in_numerical_design"] is False
    assert metadata["installed_in_exact_runtime"] is False


def test_dense_full_map_stores_one_rectangular_coefficient_and_reports_rank() -> None:
    source, target = _synthetic_pair(
        rows=96,
        execution=_sha(40),
        row_prefix="dense",
        seed=21,
    )
    edge = fit_forward_unified_event_dense_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=source,
        target=target,
        relative_ridge=1.0e-8,
        source_chart_index=0,
        target_chart_index=1,
    )
    assert edge.coefficient.shape == (3, 5)
    assert edge.stored_scalar_count == 15
    assert edge.realized_numerical_rank == 1
    assert torch.allclose(
        edge.predict(source, target.target_memberships),
        target.residual_flow,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    metadata = edge.metadata()
    assert metadata["edge_low_rank_factorization_used"] is False
    assert metadata["edge_capacity_rank"] == 3
    assert metadata["realized_numerical_rank"] == 1
    assert metadata["numerical_rank_tolerance"] > 0.0
    assert metadata["coefficient_shape"] == [3, 5]


def test_edge_eligibility_is_event_causal_cross_block_and_cross_family() -> None:
    source, target = _synthetic_pair(
        rows=32,
        execution=_sha(5),
        row_prefix="eligibility",
        seed=23,
    )
    same_block = replace(
        target,
        event_id="mutation.2",
        event_index=_event("mutation.2").event_index,
        event_kind=_event("mutation.2").event_kind,
        component_family=_event("mutation.2").component_family,
        block_ordinal=_event("mutation.2").block_ordinal,
    )
    with pytest.raises(ValueError, match="cross-block"):
        fit_forward_unified_event_edge(
            schedule=build_joint_cross_block_event_schedule(),
            source=source,
            target=same_block,
            edge_rank=1,
            relative_ridge=1.0e-6,
            target_chart_index=1,
        )

    same_family_event = _event("mutation.3")
    same_family = replace(
        target,
        event_id=same_family_event.event_id,
        event_index=same_family_event.event_index,
        event_kind=same_family_event.event_kind,
        component_family=same_family_event.component_family,
        block_ordinal=same_family_event.block_ordinal,
    )
    with pytest.raises(ValueError, match="cross-family"):
        fit_forward_unified_event_edge(
            schedule=build_joint_cross_block_event_schedule(),
            source=source,
            target=same_family,
            edge_rank=1,
            relative_ridge=1.0e-6,
            target_chart_index=1,
        )

    later_source = _packet(
        "mutation.5",
        source.coordinates,
        execution=source.execution_fingerprint,
        row_prefix="eligibility",
    )
    with pytest.raises(ValueError, match="strictly forward"):
        fit_forward_unified_event_edge(
            schedule=build_joint_cross_block_event_schedule(),
            source=later_source,
            target=target,
            edge_rank=1,
            relative_ridge=1.0e-6,
            target_chart_index=1,
            require_cross_block=False,
            require_cross_family=False,
        )


def test_lexical_ingress_is_virtual_block_minus_one_edge_source() -> None:
    source, target = _synthetic_pair(
        rows=48,
        execution=_sha(51),
        row_prefix="lexical",
        seed=41,
    )
    lexical_source = _packet(
        "lexical.ingress",
        source.coordinates,
        execution=source.execution_fingerprint,
        row_prefix="lexical",
    )
    edge = fit_forward_unified_event_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=lexical_source,
        target=target,
        edge_rank=1,
        relative_ridge=1.0e-8,
        source_chart_index=0,
        target_chart_index=1,
    )
    assert edge.spec.cross_block is True
    assert edge.spec.cross_family is True
    assert edge.metadata()["lexical_ingress_virtual_block_ordinal"] == -1


def test_mutation_input_site_is_eligible_as_an_earlier_edge_source() -> None:
    source, target = _synthetic_pair(
        rows=48,
        execution=_sha(52),
        row_prefix="input-source",
        seed=43,
    )
    input_source = replace(
        source,
        site_id="mutation.1.input",
        site_role="input",
    )
    edge = fit_forward_unified_event_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=input_source,
        target=target,
        edge_rank=1,
        relative_ridge=1.0e-8,
        source_chart_index=0,
        target_chart_index=1,
    )
    assert edge.spec.source_site_id == "mutation.1.input"
    assert edge.spec.source_site_role == "input"


def test_heldout_evaluator_scores_real_zero_and_shuffled_without_refit() -> None:
    fit_source, fit_target = _synthetic_pair(
        rows=128,
        execution=_sha(6),
        row_prefix="fit",
        seed=29,
    )
    edge = fit_forward_unified_event_edge(
        schedule=build_joint_cross_block_event_schedule(),
        source=fit_source,
        target=fit_target,
        edge_rank=1,
        relative_ridge=1.0e-8,
        source_chart_index=0,
        target_chart_index=1,
    )
    left_before = edge.left_factor.clone()
    right_before = edge.right_factor.clone()
    held_source, held_target = _synthetic_pair(
        rows=80,
        execution=_sha(7),
        row_prefix="held",
        seed=31,
    )
    result = evaluate_unified_event_edge(
        edge,
        source=held_source,
        target=held_target,
    )
    assert result.fit_rows_disjoint is True
    assert result.fit_execution_disjoint is True
    assert result.real_beats_zero is True
    assert result.real_beats_shuffle is True
    assert result.real_edge_gain_fraction > 0.99999
    assert result.shuffled_source_gain_fraction < 0.2
    assert result.metadata()["heldout_refit_performed"] is False
    assert torch.equal(edge.left_factor, left_before)
    assert torch.equal(edge.right_factor, right_before)
    with pytest.raises(ValueError, match="reused fit"):
        evaluate_unified_event_edge(
            edge,
            source=fit_source,
            target=fit_target,
        )


def test_catalog_fitter_returns_only_positive_eligible_event_edges() -> None:
    source, target = _synthetic_pair(
        rows=64,
        execution=_sha(8),
        row_prefix="catalog",
        seed=37,
    )
    same_block_source = replace(
        source,
        event_id="mutation.3",
        event_index=_event("mutation.3").event_index,
        event_kind=_event("mutation.3").event_kind,
        component_family=_event("mutation.3").component_family,
        block_ordinal=_event("mutation.3").block_ordinal,
        site_id="mutation.3.output",
    )
    edges = fit_forward_unified_event_edges(
        schedule=build_joint_cross_block_event_schedule(),
        sources={
            source.site_id: source,
            same_block_source.site_id: same_block_source,
        },
        targets={target.event_id: target},
        edge_rank=1,
        relative_ridge=1.0e-8,
    )
    assert edges
    assert {item.spec.source_event_id for item in edges} == {"mutation.1"}
    assert all(item.spec.cross_block and item.spec.cross_family for item in edges)
