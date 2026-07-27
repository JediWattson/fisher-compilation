from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Mapping, Sequence

import pytest
import torch
from torch import Tensor

from fisher_graph.modal_connectivity_modes import (
    CausalBoundaryTransfer,
    MessageMoments,
    ModalBoundaryPort,
    ModalConnectivityDecomposition,
    factor_modal_connectivity,
)
from fisher_graph.modal_graph_hierarchy import (
    AffineModalComponent,
    BoundaryInputInjection,
    BoundaryOutputReadout,
    CausalCoarseningGroup,
    DirectModalConnection,
    HierarchicalModalGenerator,
    HierarchicalModeExpansion,
    IdentityModalComponent,
    ImplicitIdentityMap,
    LinearModalGraphLevel,
    ProjectedModalConnection,
    affine_modal_component,
    extract_coarsening_group,
)
from fisher_graph.modal_graph_hierarchy_executor import (
    HierarchySourceResources,
    ModalGraphHierarchyExecutor,
)


FLOAT64 = torch.float64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _matrix(rows: Sequence[Sequence[float]]) -> Tensor:
    return torch.tensor(rows, dtype=FLOAT64)


def _vector(values: Sequence[float]) -> Tensor:
    return torch.tensor(values, dtype=FLOAT64)


def _scalar_component(
    component_id: str,
    causal_order: int,
    *,
    gain: float = 1.0,
    bias: float = 0.0,
    level_index: int = 0,
) -> AffineModalComponent:
    return affine_modal_component(
        component_id=component_id,
        causal_order=causal_order,
        matrix=_matrix([[gain]]),
        bias=_vector([bias]),
        source_artifact_sha256=_sha(f"component-source:{component_id}"),
        level_index=level_index,
    )


def _connection(
    source: str,
    target: str,
    gain: float,
    *,
    evidence_kind: str = "direct_jacobian",
    label: str | None = None,
) -> DirectModalConnection:
    edge_label = label or f"{source}->{target}:{gain}"
    return DirectModalConnection(
        source_component=source,
        source_port=f"{source}.output",
        target_component=target,
        target_port=f"{target}.input",
        matrix=_matrix([[gain]]),
        evidence_kind=evidence_kind,
        evidence_sha256=_sha(f"evidence:{edge_label}"),
    )


def _boundary_port(
    graph_id: str,
    name: str,
    direction: str,
    causal_order: int,
    *,
    width: int = 1,
) -> ModalBoundaryPort:
    return ModalBoundaryPort(
        name=name,
        direction=direction,
        causal_order=causal_order,
        width=width,
        owner_id=graph_id,
    )


def _injection(
    boundary_port: str,
    target: str,
    gain: float = 1.0,
    *,
    label: str | None = None,
) -> BoundaryInputInjection:
    edge_label = label or f"{boundary_port}->{target}:{gain}"
    return BoundaryInputInjection(
        boundary_port=boundary_port,
        target_component=target,
        target_port=f"{target}.input",
        matrix=_matrix([[gain]]),
        cut_edge_sha256=_sha(f"input-cut:{edge_label}"),
    )


def _readout(
    source: str,
    boundary_port: str,
    gain: float = 1.0,
    *,
    label: str | None = None,
) -> BoundaryOutputReadout:
    edge_label = label or f"{source}->{boundary_port}:{gain}"
    return BoundaryOutputReadout(
        source_component=source,
        source_port=f"{source}.output",
        boundary_port=boundary_port,
        matrix=_matrix([[gain]]),
        cut_edge_sha256=_sha(f"output-cut:{edge_label}"),
    )


def _graph(
    graph_id: str,
    *,
    components: Sequence[AffineModalComponent],
    connections: Sequence[DirectModalConnection],
    boundary_inputs: Sequence[ModalBoundaryPort],
    boundary_outputs: Sequence[ModalBoundaryPort],
    input_injections: Sequence[BoundaryInputInjection],
    output_readouts: Sequence[BoundaryOutputReadout],
    output_offsets: Mapping[str, Tensor] | None = None,
    level_index: int = 0,
) -> LinearModalGraphLevel:
    canonical_components = tuple(
        sorted(
            components,
            key=lambda value: (
                value.causal_start,
                value.causal_end,
                value.component_id,
            ),
        )
    )
    canonical_connections = tuple(
        sorted(
            connections,
            key=lambda value: (
                value.source_component,
                value.source_port,
                value.target_component,
                value.target_port,
            ),
        )
    )
    canonical_inputs = tuple(
        sorted(boundary_inputs, key=lambda port: (port.causal_order, port.name))
    )
    canonical_outputs = tuple(
        sorted(boundary_outputs, key=lambda port: (port.causal_order, port.name))
    )
    canonical_injections = tuple(
        sorted(
            input_injections,
            key=lambda value: (
                value.boundary_port,
                value.target_component,
                value.target_port,
                value.artifact_sha256,
            ),
        )
    )
    canonical_readouts = tuple(
        sorted(
            output_readouts,
            key=lambda value: (
                value.source_component,
                value.source_port,
                value.boundary_port,
                value.artifact_sha256,
            ),
        )
    )
    offsets = output_offsets or {}
    return LinearModalGraphLevel(
        graph_id=graph_id,
        level_index=level_index,
        source_artifact_sha256=_sha(f"graph-source:{graph_id}"),
        components=canonical_components,
        connections=canonical_connections,
        boundary_inputs=canonical_inputs,
        boundary_outputs=canonical_outputs,
        input_injections=canonical_injections,
        output_readouts=canonical_readouts,
        output_offsets=tuple(
            offsets.get(port.name, torch.zeros(port.width, dtype=FLOAT64))
            for port in canonical_outputs
        ),
    )


def _scalar_chain() -> LinearModalGraphLevel:
    graph_id = "scalar-chain"
    graph_input = _boundary_port(
        graph_id,
        "scalar-chain.x",
        "input",
        0,
    )
    graph_output = _boundary_port(
        graph_id,
        "scalar-chain.y",
        "output",
        1,
    )
    return _graph(
        graph_id,
        components=(
            _scalar_component("first", 0, gain=2.0, bias=1.0),
            _scalar_component("second", 1, gain=-3.0, bias=4.0),
        ),
        connections=(_connection("first", "second", 1.0),),
        boundary_inputs=(graph_input,),
        boundary_outputs=(graph_output,),
        input_injections=(_injection(graph_input.name, "first"),),
        output_readouts=(_readout("second", graph_output.name),),
    )


def _moments(
    port: ModalBoundaryPort,
    *,
    source_level_sha256: str,
    mean: Tensor | None = None,
    covariance: Tensor | None = None,
    fisher: Tensor | None = None,
) -> MessageMoments:
    width = port.width
    return MessageMoments(
        port=port,
        source_level_sha256=source_level_sha256,
        reduction_id="unit-test-reduction",
        sample_count=32,
        mean=(
            torch.zeros(width, dtype=FLOAT64)
            if mean is None
            else mean
        ),
        covariance=(
            torch.eye(width, dtype=FLOAT64)
            if covariance is None
            else covariance
        ),
        fisher=(
            torch.eye(width, dtype=FLOAT64)
            if fisher is None
            else fisher
        ),
    )


def _decomposition(
    transfer: CausalBoundaryTransfer,
    *,
    retained_ranks: int | Sequence[int] | Mapping[str, int] | None = None,
    input_covariances: Sequence[Tensor] | None = None,
    output_fishers: Sequence[Tensor] | None = None,
) -> ModalConnectivityDecomposition:
    input_covariances = input_covariances or tuple(
        torch.eye(port.width, dtype=FLOAT64)
        for port in transfer.input_ports
    )
    output_fishers = output_fishers or tuple(
        torch.eye(port.width, dtype=FLOAT64)
        for port in transfer.output_ports
    )
    input_moments = tuple(
        _moments(
            port,
            source_level_sha256=transfer.source_level_sha256,
            covariance=covariance,
        )
        for port, covariance in zip(
            transfer.input_ports,
            input_covariances,
            strict=True,
        )
    )
    output_moments = tuple(
        _moments(
            port,
            source_level_sha256=transfer.source_level_sha256,
            mean=transfer.affine_offsets[index],
            fisher=fisher,
        )
        for index, (port, fisher) in enumerate(
            zip(transfer.output_ports, output_fishers, strict=True)
        )
    )
    return factor_modal_connectivity(
        transfer,
        input_moments,
        output_moments,
        retained_ranks=retained_ranks,
        assume_block_diagonal_input_covariance=any(
            len(prefix) > 1 for prefix in transfer.input_prefixes
        ),
    )


def _generator(
    graph: LinearModalGraphLevel,
    decomposition: ModalConnectivityDecomposition,
    *,
    parent_id: str,
) -> HierarchicalModalGenerator:
    group = CausalCoarseningGroup.from_graph(
        graph,
        parent_id=parent_id,
        child_component_ids=tuple(
            component.component_id for component in graph.components
        ),
    )
    return HierarchicalModalGenerator(
        parent_id=parent_id,
        level_index=graph.level_index + 1,
        child_graph=graph,
        decomposition=decomposition,
        coarsening_group=group,
        exact_child_fallback_sha256=graph.artifact_sha256,
    )


def _two_dimensional_transfer(
    *,
    matrix: Tensor,
    offset: Tensor | None = None,
    graph_id: str,
) -> CausalBoundaryTransfer:
    input_port = _boundary_port(
        graph_id,
        f"{graph_id}.x",
        "input",
        0,
        width=2,
    )
    output_port = _boundary_port(
        graph_id,
        f"{graph_id}.y",
        "output",
        0,
        width=2,
    )
    return CausalBoundaryTransfer(
        source_level_sha256=_sha(f"transfer-source:{graph_id}"),
        input_ports=(input_port,),
        output_ports=(output_port,),
        input_prefixes=((input_port.name,),),
        transfer_matrices=(matrix,),
        affine_offsets=(
            torch.zeros(2, dtype=FLOAT64) if offset is None else offset,
        ),
    )


def _two_dimensional_graph(
    *,
    matrix: Tensor,
    graph_id: str,
) -> LinearModalGraphLevel:
    component_id = f"{graph_id}.component"
    component = affine_modal_component(
        component_id=component_id,
        causal_order=0,
        matrix=matrix,
        bias=torch.zeros(2, dtype=FLOAT64),
        source_artifact_sha256=_sha(f"component-source:{component_id}"),
    )
    graph_input = _boundary_port(
        graph_id,
        f"{graph_id}.x",
        "input",
        0,
        width=2,
    )
    graph_output = _boundary_port(
        graph_id,
        f"{graph_id}.y",
        "output",
        0,
        width=2,
    )
    return _graph(
        graph_id,
        components=(component,),
        connections=(),
        boundary_inputs=(graph_input,),
        boundary_outputs=(graph_output,),
        input_injections=(
            BoundaryInputInjection(
                boundary_port=graph_input.name,
                target_component=component_id,
                target_port=f"{component_id}.input",
                matrix=torch.eye(2, dtype=FLOAT64),
                cut_edge_sha256=_sha(f"{graph_id}.input-cut"),
            ),
        ),
        output_readouts=(
            BoundaryOutputReadout(
                source_component=component_id,
                source_port=f"{component_id}.output",
                boundary_port=graph_output.name,
                matrix=torch.eye(2, dtype=FLOAT64),
                cut_edge_sha256=_sha(f"{graph_id}.output-cut"),
            ),
        ),
    )


def test_affine_scalar_chain_reduces_to_h_minus_six_and_c_one() -> None:
    graph = _scalar_chain()
    transfer = graph.boundary_transfer()
    values = {
        "scalar-chain.x": _matrix([[-2.0], [0.0], [3.0]]),
    }

    torch.testing.assert_close(
        transfer.transfer_matrices[0],
        _matrix([[-6.0]]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        transfer.affine_offsets[0],
        _vector([1.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        transfer.execute(values)["scalar-chain.y"],
        graph.execute(values)["scalar-chain.y"],
        rtol=0.0,
        atol=0.0,
    )


def test_full_rank_connectivity_candidate_is_exact_for_scalar_chain() -> None:
    graph = _scalar_chain()
    transfer = graph.boundary_transfer()
    decomposition = _decomposition(transfer)
    values = {
        "scalar-chain.x": _matrix([[-2.0], [0.0], [3.0]]),
    }

    assert decomposition.factors[0].retained_rank == 1
    torch.testing.assert_close(
        decomposition.factors[0].reconstructed_matrix,
        _matrix([[-6.0]]),
        rtol=1e-14,
        atol=1e-14,
    )
    torch.testing.assert_close(
        decomposition.execute_candidate(values)["scalar-chain.y"],
        graph.execute(values)["scalar-chain.y"],
        rtol=1e-14,
        atol=1e-14,
    )


def test_fanout_preserves_distinct_signed_branch_outputs() -> None:
    graph_id = "fanout"
    graph_input = _boundary_port(graph_id, "fanout.x", "input", 0)
    left_output = _boundary_port(graph_id, "fanout.left", "output", 1)
    right_output = _boundary_port(graph_id, "fanout.right", "output", 2)
    graph = _graph(
        graph_id,
        components=(
            _scalar_component("root", 0),
            _scalar_component("left", 1),
            _scalar_component("right", 2),
        ),
        connections=(
            _connection("root", "left", 2.0),
            _connection("root", "right", -3.0),
        ),
        boundary_inputs=(graph_input,),
        boundary_outputs=(left_output, right_output),
        input_injections=(_injection(graph_input.name, "root"),),
        output_readouts=(
            _readout("left", left_output.name),
            _readout("right", right_output.name),
        ),
    )

    result = graph.execute({"fanout.x": _matrix([[4.0]])})
    transfer = graph.boundary_transfer()

    assert set(result) == {"fanout.left", "fanout.right"}
    assert result["fanout.left"].item() == 8.0
    assert result["fanout.right"].item() == -12.0
    assert transfer.block("fanout.left", "fanout.x").item() == 2.0
    assert transfer.block("fanout.right", "fanout.x").item() == -3.0


def test_output_prefix_structurally_omits_later_boundary_input() -> None:
    graph_id = "prefix"
    early_input = _boundary_port(graph_id, "prefix.x0", "input", 0)
    later_input = _boundary_port(graph_id, "prefix.x1", "input", 1)
    early_output = _boundary_port(graph_id, "prefix.y0", "output", 0)
    later_output = _boundary_port(graph_id, "prefix.y1", "output", 1)
    graph = _graph(
        graph_id,
        components=(
            _scalar_component("early", 0),
            _scalar_component("late", 1),
        ),
        connections=(_connection("early", "late", 5.0),),
        boundary_inputs=(early_input, later_input),
        boundary_outputs=(early_output, later_output),
        input_injections=(
            _injection(early_input.name, "early"),
            _injection(later_input.name, "late"),
        ),
        output_readouts=(
            _readout("early", early_output.name),
            _readout("late", later_output.name),
        ),
    )

    transfer = graph.boundary_transfer()

    assert transfer.input_prefixes == (
        ("prefix.x0",),
        ("prefix.x0", "prefix.x1"),
    )
    assert transfer.transfer_matrices[0].shape == (1, 1)
    assert transfer.transfer_matrices[1].shape == (1, 2)
    with pytest.raises(KeyError, match="not in the causal prefix"):
        transfer.block("prefix.y0", "prefix.x1")
    low_later = graph.execute(
        {
            "prefix.x0": _matrix([[2.0]]),
            "prefix.x1": _matrix([[-1.0e9]]),
        }
    )
    high_later = graph.execute(
        {
            "prefix.x0": _matrix([[2.0]]),
            "prefix.x1": _matrix([[1.0e9]]),
        }
    )
    assert torch.equal(low_later["prefix.y0"], high_later["prefix.y0"])


def test_signed_diamond_cancellation_survives_large_direct_edges() -> None:
    graph_id = "diamond"
    graph_input = _boundary_port(graph_id, "diamond.x", "input", 0)
    graph_output = _boundary_port(graph_id, "diamond.y", "output", 3)
    graph = _graph(
        graph_id,
        components=(
            _scalar_component("diamond-root", 0),
            _scalar_component("diamond-left", 1),
            _scalar_component("diamond-right", 2),
            _scalar_component("diamond-sink", 3),
        ),
        connections=(
            _connection("diamond-root", "diamond-left", 1.0e6),
            _connection("diamond-root", "diamond-right", 1.0e6),
            _connection("diamond-left", "diamond-sink", 1.0e6),
            _connection("diamond-right", "diamond-sink", -1.0e6),
            _connection("diamond-root", "diamond-sink", 3.0),
        ),
        boundary_inputs=(graph_input,),
        boundary_outputs=(graph_output,),
        input_injections=(_injection(graph_input.name, "diamond-root"),),
        output_readouts=(_readout("diamond-sink", graph_output.name),),
    )

    transfer = graph.boundary_transfer()
    result = graph.execute({"diamond.x": _matrix([[2.0]])})

    # The two magnitude-1e12 path products cancel.  The small signed direct
    # path remains and must not be hidden by magnitude-only graph reduction.
    assert transfer.transfer_matrices[0].item() == 3.0
    assert result["diamond.y"].item() == 6.0


def test_fisher_weighting_selects_small_raw_gain_and_reports_point_16_tail() -> None:
    transfer = _two_dimensional_transfer(
        graph_id="fisher-rank",
        matrix=torch.diag(_vector([4.0, 1.0])),
    )
    decomposition = _decomposition(
        transfer,
        retained_ranks=1,
        input_covariances=(torch.eye(2, dtype=FLOAT64),),
        output_fishers=(torch.diag(_vector([0.01, 100.0])),),
    )
    factor = decomposition.factors[0]

    torch.testing.assert_close(
        factor.singular_values,
        _vector([10.0, 0.4]),
        rtol=1e-14,
        atol=1e-14,
    )
    torch.testing.assert_close(
        factor.reconstructed_matrix,
        _matrix([[0.0, 0.0], [0.0, 1.0]]),
        rtol=1e-14,
        atol=1e-14,
    )
    assert factor.retained_rank == 1
    assert factor.discarded_weighted_energy == pytest.approx(
        0.16,
        rel=1e-13,
        abs=1e-13,
    )
    assert decomposition.discarded_weighted_energy == pytest.approx(0.16)


def test_full_rank_reconstructs_only_measured_singular_support() -> None:
    source_matrix = torch.diag(_vector([2.0, 3.0]))
    transfer = _two_dimensional_transfer(
        graph_id="singular-support",
        matrix=source_matrix,
    )
    support = torch.diag(_vector([1.0, 0.0]))
    decomposition = _decomposition(
        transfer,
        retained_ranks=2,
        input_covariances=(support,),
        output_fishers=(support,),
    )
    factor = decomposition.factors[0]
    expected_supported_transfer = support @ source_matrix @ support

    assert factor.input_support_ranks == (1,)
    assert factor.output_support_rank == 1
    torch.testing.assert_close(
        factor.reconstructed_matrix,
        expected_supported_transfer,
        rtol=1e-14,
        atol=1e-14,
    )
    source = transfer.execute(
        {"singular-support.x": _matrix([[0.0, 1.0]])}
    )
    candidate = decomposition.execute_candidate(
        {"singular-support.x": _matrix([[0.0, 1.0]])}
    )
    assert source["singular-support.y"].tolist() == [[0.0, 3.0]]
    assert candidate["singular-support.y"].tolist() == [[0.0, 0.0]]


def test_tied_connectivity_spectrum_has_canonical_coordinate_basis() -> None:
    transfer = _two_dimensional_transfer(
        graph_id="tied-spectrum",
        matrix=torch.eye(2, dtype=FLOAT64),
    )

    first = _decomposition(transfer)
    second = _decomposition(transfer)

    torch.testing.assert_close(
        first.factors[0].weighted_left_vectors,
        torch.eye(2, dtype=FLOAT64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.factors[0].weighted_right_vectors,
        torch.eye(2, dtype=FLOAT64),
        rtol=0.0,
        atol=0.0,
    )
    assert first.artifact_sha256 == second.artifact_sha256


def test_decomposition_binds_actual_moment_hashes_and_source_level() -> None:
    graph = _scalar_chain()
    transfer = graph.boundary_transfer()
    decomposition = _decomposition(transfer)
    source_moment = decomposition.input_moments[0]
    replacement_moment = _moments(
        source_moment.port,
        source_level_sha256=source_moment.source_level_sha256,
        covariance=_matrix([[2.0]]),
    )

    with pytest.raises(
        ValueError,
        match="factor input moments do not match",
    ):
        replace(
            decomposition,
            input_moments=(replacement_moment,),
            artifact_sha256="",
        )
    wrong_source = _moments(
        source_moment.port,
        source_level_sha256=_sha("wrong-source-level"),
    )
    with pytest.raises(
        ValueError,
        match="source level does not match",
    ):
        replace(
            decomposition,
            input_moments=(wrong_source,),
            artifact_sha256="",
        )


def test_rank_zero_shadow_is_source_authoritative_and_adaptive_expands() -> None:
    graph = _scalar_chain()
    decomposition = _decomposition(
        graph.boundary_transfer(),
        retained_ranks=0,
    )
    generator = _generator(
        graph,
        decomposition,
        parent_id="rank-zero-parent",
    )
    executor = ModalGraphHierarchyExecutor(generator)
    values = {"scalar-chain.x": _matrix([[1.0], [2.0]])}
    exact = graph.execute(values)

    shadow = executor.execute(
        values,
        mode="shadow",
        expansion_weighted_error_threshold=0.0,
    )
    adaptive = executor.execute(
        values,
        mode="adaptive_validation",
        expansion_weighted_error_threshold=0.0,
    )

    assert decomposition.factors[0].retained_rank == 0
    assert shadow.authoritative_path == "source_shadow"
    assert shadow.candidate_outputs is not None
    assert shadow.maximum_weighted_error == pytest.approx(144.0)
    assert torch.equal(shadow.outputs["scalar-chain.y"], exact["scalar-chain.y"])
    assert shadow.candidate_outputs["scalar-chain.y"].tolist() == [[1.0], [1.0]]
    assert adaptive.expanded_to_source is True
    assert adaptive.authoritative_path == "source_expansion"
    assert torch.equal(
        adaptive.outputs["scalar-chain.y"],
        exact["scalar-chain.y"],
    )


def test_connection_rejects_observational_evidence_as_executable_edge() -> None:
    with pytest.raises(
        ValueError,
        match="direct_jacobian edges only",
    ):
        _connection(
            "observed-source",
            "observed-target",
            1.0,
            evidence_kind="finite_ablation_response",
        )


def test_accounting_separates_active_macs_from_resident_fallback_bytes() -> None:
    graph = _scalar_chain()
    decomposition = _decomposition(
        graph.boundary_transfer(),
        retained_ranks=0,
    )
    generator = _generator(
        graph,
        decomposition,
        parent_id="accounting-parent",
    )
    resources = HierarchySourceResources(
        source_artifact_sha256=generator.exact_child_fallback_sha256,
        learned_parameter_count=101,
        linear_macs_per_row=37,
        storage_bytes=8192,
    )
    executor = ModalGraphHierarchyExecutor(
        generator,
        source_resources=resources,
    )

    candidate = executor.accounting("candidate")
    shadow = executor.accounting("shadow")

    assert candidate.candidate_stored_scalar_count == 2
    assert candidate.candidate_linear_macs_per_row == 0
    assert candidate.candidate_storage_bytes_float64 == 16
    assert candidate.active_linear_macs_per_row == 0
    assert candidate.validation_metric_macs_per_row == 2
    assert candidate.validation_metric_storage_bytes_float64 == 8
    assert shadow.active_linear_macs_per_row == 39
    assert candidate.resident_storage_bytes == 8192 + 16
    assert shadow.resident_storage_bytes == 8192 + 16 + 8
    assert candidate.exact_child_fallback_resident is True
    assert candidate.transitive_leaf_fallback_authorized is False
    assert candidate.deployed_storage_reduction_authorized is False
    assert candidate.latency_reduction_claim is False
    assert candidate.logical_candidate_parameter_delta == 2 - 101
    assert candidate.logical_candidate_mac_delta == 0 - 37


def test_executor_reauthenticates_mutable_tensors_before_every_run() -> None:
    graph = _scalar_chain()
    generator = _generator(
        graph,
        _decomposition(graph.boundary_transfer()),
        parent_id="mutation-guard-parent",
    )
    executor = ModalGraphHierarchyExecutor(generator)
    generator.decomposition.factors[0].restriction.add_(1.0)

    with pytest.raises(
        ValueError,
        match="connectivity factor artifact hash mismatch",
    ):
        executor.execute(
            {"scalar-chain.x": _matrix([[2.0]])},
            mode="candidate",
        )


def test_hierarchical_generator_component_can_be_reused_recursively() -> None:
    leaf_graph = _scalar_chain()
    level_one = _generator(
        leaf_graph,
        _decomposition(leaf_graph.boundary_transfer()),
        parent_id="level-one-parent",
    )
    with pytest.raises(ValueError, match="must remain factorized"):
        level_one.as_component(candidate=True)
    level_one_expansion = level_one.factorized_expansion()
    level_one_graph = level_one_expansion.graph
    assert len(level_one_expansion.mode_ports) == 1
    assert level_one_expansion.mode_ports[0].width == 1
    assert len(level_one_graph.components) == 2
    assert len(level_one_graph.connections) == 1
    assert (
        level_one_expansion.encoder_graph.output_readouts[0].source_port
        == level_one_graph.connections[0].source_port
    )
    torch.testing.assert_close(
        level_one_graph.execute(
            {"scalar-chain.x": _matrix([[2.0]])}
        )["scalar-chain.y"],
        _matrix([[-11.0]]),
        rtol=1e-14,
        atol=1e-14,
    )

    level_two = _generator(
        level_one_expansion.recursive_graph,
        level_one_expansion.factor_recursive_connectivity(),
        parent_id="level-two-parent",
    )
    level_two_expansion = level_two.factorized_expansion()

    assert level_one_graph.level_index == 1
    assert level_two_expansion.graph.level_index == 2
    assert len(level_two_expansion.mode_ports) == 1
    assert (
        level_two.decomposition.factors[0].output_moment_sha256
        == level_one_expansion.recursive_output_moments[
            0
        ].artifact_sha256
    )
    encoded_modes = level_one_expansion.encoder_graph.execute(
        {"scalar-chain.x": _matrix([[2.0]])}
    )
    level_one_modes = level_two_expansion.graph.execute(encoded_modes)
    result = level_one.prolong_modal_outputs(level_one_modes)
    torch.testing.assert_close(
        result["scalar-chain.y"],
        _matrix([[-11.0]]),
        rtol=1e-14,
        atol=1e-14,
    )


def test_reduced_rank_recursion_uses_previous_modal_coordinates() -> None:
    graph = _two_dimensional_graph(
        graph_id="rank-one-recursion",
        matrix=torch.diag(_vector([4.0, 1.0])),
    )
    level_one = _generator(
        graph,
        _decomposition(
            graph.boundary_transfer(),
            retained_ranks=1,
            output_fishers=(
                torch.diag(_vector([0.01, 100.0])),
            ),
        ),
        parent_id="rank-one-level-one",
    )
    expansion_one = level_one.factorized_expansion()
    values = {
        "rank-one-recursion.x": _matrix([[3.0, 2.0]]),
    }

    encoded = expansion_one.encoder_graph.execute(values)
    modal = expansion_one.recursive_graph.execute(encoded)
    mode_name = expansion_one.mode_ports[0].name
    torch.testing.assert_close(
        modal[mode_name],
        _matrix([[2.0 * (10.0**0.5)]]),
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        expansion_one.mode_moments[0].covariance,
        _matrix([[10.0]]),
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        expansion_one.mode_moments[0].fisher,
        _matrix([[10.0]]),
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        expansion_one.mode_moments[0].salience,
        _vector([100.0]),
        rtol=1e-13,
        atol=1e-13,
    )
    level_one_candidate = expansion_one.graph.execute(values)
    torch.testing.assert_close(
        level_one_candidate["rank-one-recursion.y"],
        _matrix([[0.0, 2.0]]),
        rtol=1e-13,
        atol=1e-13,
    )

    recursive_decomposition = (
        expansion_one.factor_recursive_connectivity(retained_ranks=1)
    )
    assert (
        recursive_decomposition.factors[0].output_moment_sha256
        == expansion_one.recursive_output_moments[0].artifact_sha256
    )
    level_two = _generator(
        expansion_one.recursive_graph,
        recursive_decomposition,
        parent_id="rank-one-level-two",
    )
    assert level_two.authorizes_transitive_leaf_fallback is False
    level_two_source = ModalGraphHierarchyExecutor(level_two).execute(
        encoded,
        mode="source",
    )
    assert (
        level_two_source.accounting.transitive_leaf_fallback_authorized
        is False
    )
    immediate_child_reconstruction = level_one.prolong_modal_outputs(
        level_two_source.outputs
    )
    torch.testing.assert_close(
        immediate_child_reconstruction["rank-one-recursion.y"],
        level_one_candidate["rank-one-recursion.y"],
        rtol=1e-13,
        atol=1e-13,
    )
    assert not torch.equal(
        immediate_child_reconstruction["rank-one-recursion.y"],
        graph.execute(values)["rank-one-recursion.y"],
    )
    previous_modes = level_two.factorized_expansion().graph.execute(encoded)
    reconstructed = level_one.prolong_modal_outputs(previous_modes)
    torch.testing.assert_close(
        reconstructed["rank-one-recursion.y"],
        level_one_candidate["rank-one-recursion.y"],
        rtol=1e-13,
        atol=1e-13,
    )


def test_projected_signed_modal_interaction_populates_measured_core() -> None:
    graph_id = "projected-modal-core"
    graph_input = _boundary_port(
        graph_id,
        "projected-modal-core.x",
        "input",
        0,
    )
    late_input = _boundary_port(
        graph_id,
        "projected-modal-core.late-input",
        "input",
        2,
    )
    early_output = _boundary_port(
        graph_id,
        "projected-modal-core.early",
        "output",
        1,
    )
    late_output = _boundary_port(
        graph_id,
        "projected-modal-core.late",
        "output",
        2,
    )
    graph = _graph(
        graph_id,
        components=(
            _scalar_component("projected-root", 0),
            _scalar_component("projected-early", 1),
            _scalar_component("projected-late", 2),
        ),
        connections=(
            _connection("projected-root", "projected-early", 2.0),
            _connection("projected-root", "projected-late", -3.0),
        ),
        boundary_inputs=(graph_input, late_input),
        boundary_outputs=(early_output, late_output),
        input_injections=(
            _injection(graph_input.name, "projected-root"),
            _injection(late_input.name, "projected-late"),
        ),
        output_readouts=(
            _readout("projected-early", early_output.name),
            _readout("projected-late", late_output.name),
        ),
    )
    expansion = _generator(
        graph,
        _decomposition(graph.boundary_transfer()),
        parent_id="projected-modal-parent",
    ).factorized_expansion()
    assert (
        expansion.source_decomposition
        .assumes_block_diagonal_input_covariance
        is True
    )
    assert (
        expansion.source_decomposition
        .assumes_block_diagonal_output_fisher
        is True
    )
    with pytest.raises(ValueError, match="earlier output to a later input"):
        expansion.projected_connection(
            upstream_mode_index=0,
            downstream_mode_index=1,
            downstream_input_port_name=graph_input.name,
            fine_jacobian=_matrix([[2.0]]),
            evidence_sha256=_sha("future-to-past-direct-jacobian"),
        )
    connection = expansion.projected_connection(
        upstream_mode_index=0,
        downstream_mode_index=1,
        downstream_input_port_name=late_input.name,
        fine_jacobian=_matrix([[2.0]]),
        evidence_sha256=_sha("projected-direct-jacobian"),
    )
    downstream_factor = expansion.source_decomposition.factors[1]
    downstream_names = tuple(
        port.name for port in downstream_factor.input_ports
    )
    target_index = downstream_names.index(late_input.name)
    target_start = sum(
        port.width
        for port in downstream_factor.input_ports[:target_index]
    )
    target_width = downstream_factor.input_ports[target_index].width
    expected_projection = (
        downstream_factor.restriction[
            :,
            target_start : target_start + target_width,
        ]
        @ _matrix([[2.0]])
        @ expansion.source_decomposition.factors[0].prolongation
    )
    torch.testing.assert_close(
        connection.matrix,
        expected_projection,
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="does not equal R_j J_ji P_i"):
        replace(
            connection,
            matrix=connection.matrix * 7.0,
            artifact_sha256="",
        )
    stale_connection = replace(
        connection,
        source_expansion_sha256=_sha("stale-modal-expansion"),
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="stale for this modal expansion"):
        expansion.modal_core_graph(connections=(stale_connection,))
    measured_graph = expansion.modal_core_graph(
        connections=(connection,),
    )
    assert isinstance(connection, ProjectedModalConnection)
    measured_input_moments = tuple(
        MessageMoments(
            port=port,
            source_level_sha256=measured_graph.artifact_sha256,
            reduction_id=mode_moments.reduction_id,
            sample_count=mode_moments.sample_count,
            mean=torch.zeros(port.width, dtype=FLOAT64),
            covariance=mode_moments.covariance,
            fisher=mode_moments.fisher,
        )
        for port, mode_moments in zip(
            measured_graph.boundary_inputs,
            expansion.mode_moments,
            strict=True,
        )
    )
    measured_input_covariance = {
        port.name: moments.covariance
        for port, moments in zip(
            measured_graph.boundary_inputs,
            expansion.mode_moments,
            strict=True,
        )
    }
    measured_transfer = measured_graph.boundary_transfer()
    measured_output_moments = tuple(
        MessageMoments(
            port=port,
            source_level_sha256=measured_graph.artifact_sha256,
            reduction_id=expansion.mode_moments[0].reduction_id,
            sample_count=expansion.mode_moments[0].sample_count,
            mean=torch.zeros(port.width, dtype=FLOAT64),
            covariance=(
                matrix
                @ torch.block_diag(
                    *(
                        measured_input_covariance[name]
                        for name in prefix
                    )
                )
                @ matrix.T
            ),
            fisher=torch.eye(port.width, dtype=FLOAT64),
        )
        for port, matrix, prefix in zip(
            measured_graph.boundary_outputs,
            measured_transfer.transfer_matrices,
            measured_transfer.input_prefixes,
            strict=True,
        )
    )
    measured = expansion.measured_modal_core(
        connections=(connection,),
        input_moments=measured_input_moments,
        output_moments=measured_output_moments,
    )
    modal_inputs = {
        expansion.mode_input_ports[0].name: _matrix([[1.0]]),
        expansion.mode_input_ports[1].name: _matrix([[3.0]]),
    }
    result = measured.graph.execute(modal_inputs)
    assert result[expansion.mode_ports[0].name].item() == 1.0
    assert result[expansion.mode_ports[1].name].item() == pytest.approx(
        3.0 + expected_projection.item()
    )
    assert measured.analysis_only is True
    assert measured.authorizes_replacement is False
    with pytest.raises(ValueError, match="measured joint covariance"):
        measured.factor_connectivity()


def test_factor_must_reconstruct_the_linked_transfer_and_moments() -> None:
    transfer = _two_dimensional_transfer(
        graph_id="factor-binding",
        matrix=torch.diag(_vector([4.0, 1.0])),
    )
    alternate = replace(
        transfer,
        transfer_matrices=(torch.diag(_vector([1.0, 4.0])),),
        artifact_sha256="",
    )
    expected = _decomposition(transfer)
    wrong = _decomposition(alternate)
    forged_factor = replace(
        wrong.factors[0],
        source_transfer_sha256=transfer.artifact_sha256,
        artifact_sha256="",
    )

    with pytest.raises(
        ValueError,
        match="does not reconstruct linked transfer and moments",
    ):
        ModalConnectivityDecomposition(
            source_transfer=transfer,
            input_moments=expected.input_moments,
            output_moments=expected.output_moments,
            factors=(forged_factor,),
            relative_eigenvalue_cutoff=(
                expected.relative_eigenvalue_cutoff
            ),
            relative_singular_value_cutoff=(
                expected.relative_singular_value_cutoff
            ),
            assumes_block_diagonal_input_covariance=(
                expected.assumes_block_diagonal_input_covariance
            ),
            assumes_block_diagonal_output_fisher=(
                expected.assumes_block_diagonal_output_fisher
            ),
        )


def test_public_factorized_graph_execution_reauthenticates_tensors() -> None:
    graph = _scalar_chain()
    generator = _generator(
        graph,
        _decomposition(graph.boundary_transfer()),
        parent_id="direct-graph-mutation",
    )
    factorized = generator.as_factorized_graph()
    factorized.components[0].transfer.transfer_matrices[0].add_(1.0)

    with pytest.raises(
        ValueError,
        match="boundary transfer artifact hash mismatch",
    ):
        factorized.execute(
            {"scalar-chain.x": _matrix([[2.0]])}
        )


def test_mode_moments_cannot_be_mutated_and_rehashed() -> None:
    graph = _scalar_chain()
    expansion = _generator(
        graph,
        _decomposition(graph.boundary_transfer()),
        parent_id="mode-moment-mutation",
    ).factorized_expansion()
    expansion.mode_moments[0].fisher.add_(1.0)

    with pytest.raises(
        ValueError,
        match="connectivity mode moments are inconsistent",
    ):
        replace(expansion, artifact_sha256="")
    with pytest.raises(
        ValueError,
        match="do not bind the source generator",
    ):
        replace(
            _generator(
                graph,
                _decomposition(graph.boundary_transfer()),
                parent_id="mode-source-mutation",
            ).factorized_expansion(),
            source_generator_sha256="0" * 64,
            artifact_sha256="",
        )


def test_generator_binds_the_complete_coarsening_group() -> None:
    graph = _scalar_chain()
    generator = _generator(
        graph,
        _decomposition(graph.boundary_transfer()),
        parent_id="group-binding",
    )
    other_graph = _two_dimensional_graph(
        graph_id="unrelated-group-graph",
        matrix=torch.eye(2, dtype=FLOAT64),
    )
    unrelated = CausalCoarseningGroup.from_graph(
        other_graph,
        parent_id=generator.parent_id,
        child_component_ids=tuple(
            component.component_id
            for component in other_graph.components
        ),
    )

    with pytest.raises(ValueError):
        replace(
            generator,
            coarsening_group=unrelated,
            artifact_sha256="",
        )


def test_factorized_wiring_is_implicit_and_zero_cost() -> None:
    identity = ImplicitIdentityMap(width=3)
    assert identity.shape == (3, 3)
    graph = _scalar_chain()
    decomposition = _decomposition(graph.boundary_transfer())
    expansion = _generator(
        graph,
        decomposition,
        parent_id="implicit-wiring",
    ).factorized_expansion()
    fine_maps = tuple(
        value.matrix
        for value in (
            expansion.graph.connections
            + expansion.graph.input_injections
            + expansion.graph.output_readouts
        )
    )
    recursive_maps = tuple(
        value.matrix
        for value in (
            expansion.recursive_graph.input_injections
            + expansion.recursive_graph.output_readouts
        )
    )

    assert all(
        isinstance(value, ImplicitIdentityMap)
        for value in fine_maps + recursive_maps
    )
    assert all(
        isinstance(value, IdentityModalComponent)
        for value in expansion.recursive_graph.components
    )
    assert expansion.recursive_graph.macs_per_row == 0
    assert expansion.recursive_graph.stored_scalar_count == 0
    assert expansion.graph.macs_per_row == decomposition.candidate_macs_per_row
    assert (
        expansion.graph.stored_scalar_count
        <= decomposition.candidate_stored_scalar_count
    )


def test_rank_zero_candidate_cannot_masquerade_as_recursive_mode_port() -> None:
    graph = _scalar_chain()
    generator = _generator(
        graph,
        _decomposition(graph.boundary_transfer(), retained_ranks=0),
        parent_id="rank-zero-expansion",
    )

    with pytest.raises(ValueError, match="rank-zero outputs"):
        generator.factorized_expansion()


def test_boundary_input_fanout_and_output_fanin_are_summed() -> None:
    graph_id = "boundary-fan"
    graph_input = _boundary_port(graph_id, "boundary-fan.x", "input", 0)
    graph_output = _boundary_port(
        graph_id,
        "boundary-fan.y",
        "output",
        1,
    )
    graph = _graph(
        graph_id,
        components=(
            _scalar_component("fan-first", 0, gain=2.0),
            _scalar_component("fan-second", 1, gain=3.0),
        ),
        connections=(),
        boundary_inputs=(graph_input,),
        boundary_outputs=(graph_output,),
        input_injections=(
            _injection(graph_input.name, "fan-first"),
            _injection(graph_input.name, "fan-second"),
        ),
        output_readouts=(
            _readout("fan-first", graph_output.name),
            _readout("fan-second", graph_output.name),
        ),
    )

    result = graph.execute(
        {"boundary-fan.x": _matrix([[3.0]])}
    )
    assert result["boundary-fan.y"].item() == 15.0
    assert (
        graph.boundary_transfer().transfer_matrices[0].item()
        == 5.0
    )


def test_graph_rejects_bias_only_output_without_causal_input_prefix() -> None:
    graph_id = "empty-prefix"
    later_input = _boundary_port(
        graph_id,
        "empty-prefix.x",
        "input",
        1,
    )
    early_output = _boundary_port(
        graph_id,
        "empty-prefix.y",
        "output",
        0,
    )

    with pytest.raises(
        ValueError,
        match="nonempty causal input prefix",
    ):
        _graph(
            graph_id,
            components=(
                _scalar_component("bias-only", 0, bias=2.0),
                _scalar_component("later-input-user", 1),
            ),
            connections=(),
            boundary_inputs=(later_input,),
            boundary_outputs=(early_output,),
            input_injections=(
                _injection(later_input.name, "later-input-user"),
            ),
            output_readouts=(
                _readout("bias-only", early_output.name),
            ),
        )


def test_cut_group_requires_contiguity_and_classifies_every_cut_edge() -> None:
    graph_id = "cut-group"
    graph_input = _boundary_port(graph_id, "cut-group.x", "input", 0)
    graph_output = _boundary_port(graph_id, "cut-group.y", "output", 2)
    incoming = _connection("cut-root", "cut-middle", 2.0)
    outgoing = _connection("cut-middle", "cut-tail", 3.0)
    graph = _graph(
        graph_id,
        components=(
            _scalar_component("cut-root", 0),
            _scalar_component("cut-middle", 1),
            _scalar_component("cut-tail", 2),
        ),
        connections=(incoming, outgoing),
        boundary_inputs=(graph_input,),
        boundary_outputs=(graph_output,),
        input_injections=(_injection(graph_input.name, "cut-root"),),
        output_readouts=(_readout("cut-tail", graph_output.name),),
    )

    with pytest.raises(ValueError, match="contiguous intervals"):
        CausalCoarseningGroup.from_graph(
            graph,
            parent_id="bad-noncontiguous-parent",
            child_component_ids=("cut-root", "cut-tail"),
        )
    with pytest.raises(ValueError, match="causal order exactly"):
        CausalCoarseningGroup.from_graph(
            graph,
            parent_id="bad-order-parent",
            child_component_ids=("cut-middle", "cut-root"),
        )

    group = CausalCoarseningGroup.from_graph(
        graph,
        parent_id="middle-parent",
        child_component_ids=("cut-middle",),
    )
    assert group.internal_connection_sha256s == ()
    assert group.incoming_connection_sha256s == (
        incoming.artifact_sha256,
    )
    assert group.outgoing_connection_sha256s == (
        outgoing.artifact_sha256,
    )
    assert group.boundary_injection_sha256s == ()
    assert group.boundary_readout_sha256s == ()

    extracted = extract_coarsening_group(graph, group)
    assert len(extracted.boundary_inputs) == 1
    assert len(extracted.boundary_outputs) == 1
    assert extracted.input_injections[0].cut_edge_sha256 == (
        incoming.artifact_sha256
    )
    assert extracted.output_readouts[0].cut_edge_sha256 == (
        outgoing.artifact_sha256
    )
    assert extracted.boundary_transfer().transfer_matrices[0].item() == 6.0
    extracted_generator = HierarchicalModalGenerator(
        parent_id=group.parent_id,
        level_index=extracted.level_index + 1,
        child_graph=extracted,
        decomposition=_decomposition(extracted.boundary_transfer()),
        coarsening_group=group,
        exact_child_fallback_sha256=extracted.artifact_sha256,
    )
    assert (
        extracted_generator.coarsening_group.artifact_sha256
        == group.artifact_sha256
    )

    incomplete = replace(
        group,
        incoming_connection_sha256s=(),
        artifact_sha256="",
    )
    with pytest.raises(
        ValueError,
        match="omits or misclassifies a cut edge",
    ):
        incomplete.validate_against(graph)


def _extracted_scalar_chain() -> tuple[
    CausalCoarseningGroup,
    LinearModalGraphLevel,
]:
    graph = _scalar_chain()
    group = CausalCoarseningGroup.from_graph(
        graph,
        parent_id="authenticated-extracted-parent",
        child_component_ids=tuple(
            component.component_id for component in graph.components
        ),
    )
    return group, extract_coarsening_group(graph, group)


def test_extracted_group_rejects_selected_component_body_mutation() -> None:
    group, extracted = _extracted_scalar_chain()
    altered_first = _scalar_component(
        "first",
        0,
        gain=9.0,
        bias=1.0,
    )
    tampered = replace(
        extracted,
        components=tuple(
            altered_first
            if component.component_id == altered_first.component_id
            else component
            for component in extracted.components
        ),
        artifact_sha256="",
    )
    assert (
        tampered.boundary_transfer().artifact_sha256
        != extracted.boundary_transfer().artifact_sha256
    )

    with pytest.raises(ValueError):
        group.validate_extracted_child(tampered)


def test_extracted_group_rejects_input_cut_map_mutation() -> None:
    group, extracted = _extracted_scalar_chain()
    original = extracted.input_injections[0]
    altered = replace(
        original,
        matrix=original.matrix * 7.0,
        artifact_sha256="",
    )
    tampered = replace(
        extracted,
        input_injections=(altered,),
        artifact_sha256="",
    )
    assert (
        tampered.boundary_transfer().artifact_sha256
        != extracted.boundary_transfer().artifact_sha256
    )

    with pytest.raises(ValueError):
        group.validate_extracted_child(tampered)


def test_extracted_group_rejects_output_cut_map_mutation() -> None:
    group, extracted = _extracted_scalar_chain()
    original = extracted.output_readouts[0]
    altered = replace(
        original,
        matrix=original.matrix * 7.0,
        artifact_sha256="",
    )
    tampered = replace(
        extracted,
        output_readouts=(altered,),
        artifact_sha256="",
    )
    assert (
        tampered.boundary_transfer().artifact_sha256
        != extracted.boundary_transfer().artifact_sha256
    )

    with pytest.raises(ValueError):
        group.validate_extracted_child(tampered)


def test_extracted_group_rejects_nonzero_output_offset() -> None:
    group, extracted = _extracted_scalar_chain()
    tampered = replace(
        extracted,
        output_offsets=tuple(
            torch.ones_like(offset)
            for offset in extracted.output_offsets
        ),
        artifact_sha256="",
    )
    assert (
        tampered.boundary_transfer().artifact_sha256
        != extracted.boundary_transfer().artifact_sha256
    )

    with pytest.raises(ValueError):
        group.validate_extracted_child(tampered)


def test_full_group_extraction_preserves_owned_boundary_offset() -> None:
    source = replace(
        _scalar_chain(),
        output_offsets=(_vector([5.0]),),
        artifact_sha256="",
    )
    group = CausalCoarseningGroup.from_graph(
        source,
        parent_id="offset-preserving-parent",
        child_component_ids=tuple(
            component.component_id for component in source.components
        ),
    )
    extracted = extract_coarsening_group(source, group)

    assert extracted.output_offsets[0].item() == 5.0
    assert (
        extracted.boundary_transfer().affine_offsets[0].item()
        == source.boundary_transfer().affine_offsets[0].item()
    )


def test_graph_rejects_duplicate_boundary_terms() -> None:
    source = _scalar_chain()
    injection = source.input_injections[0]
    readout = source.output_readouts[0]

    with pytest.raises(ValueError, match="input injections must be unique"):
        replace(
            source,
            input_injections=(injection, injection),
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="output readouts must be unique"):
        replace(
            source,
            output_readouts=(readout, readout),
            artifact_sha256="",
        )


def _general_measured_modal_core_fixture() -> tuple[
    HierarchicalModeExpansion,
    DirectModalConnection,
    LinearModalGraphLevel,
    tuple[MessageMoments, ...],
    tuple[MessageMoments, ...],
]:
    graph_id = "general-measured-modal-core"
    graph_input = _boundary_port(
        graph_id,
        f"{graph_id}.x",
        "input",
        0,
        width=2,
    )
    early_output = _boundary_port(
        graph_id,
        f"{graph_id}.early",
        "output",
        1,
        width=2,
    )
    late_output = _boundary_port(
        graph_id,
        f"{graph_id}.late",
        "output",
        2,
        width=2,
    )
    early = affine_modal_component(
        component_id=f"{graph_id}.early-component",
        causal_order=1,
        matrix=_matrix([[2.0, 0.3], [-0.2, 1.0]]),
        bias=torch.zeros(2, dtype=FLOAT64),
        source_artifact_sha256=_sha(f"{graph_id}.early-source"),
    )
    late = affine_modal_component(
        component_id=f"{graph_id}.late-component",
        causal_order=2,
        matrix=_matrix([[1.1, -0.4], [0.2, 1.7]]),
        bias=torch.zeros(2, dtype=FLOAT64),
        source_artifact_sha256=_sha(f"{graph_id}.late-source"),
    )
    graph = _graph(
        graph_id,
        components=(early, late),
        connections=(),
        boundary_inputs=(graph_input,),
        boundary_outputs=(early_output, late_output),
        input_injections=(
            BoundaryInputInjection(
                boundary_port=graph_input.name,
                target_component=early.component_id,
                target_port=next(iter(early.input_ports)),
                matrix=torch.eye(2, dtype=FLOAT64),
                cut_edge_sha256=_sha(f"{graph_id}.early-input-cut"),
            ),
            BoundaryInputInjection(
                boundary_port=graph_input.name,
                target_component=late.component_id,
                target_port=next(iter(late.input_ports)),
                matrix=torch.eye(2, dtype=FLOAT64),
                cut_edge_sha256=_sha(f"{graph_id}.late-input-cut"),
            ),
        ),
        output_readouts=(
            BoundaryOutputReadout(
                source_component=early.component_id,
                source_port=next(iter(early.output_ports)),
                boundary_port=early_output.name,
                matrix=torch.eye(2, dtype=FLOAT64),
                cut_edge_sha256=_sha(f"{graph_id}.early-output-cut"),
            ),
            BoundaryOutputReadout(
                source_component=late.component_id,
                source_port=next(iter(late.output_ports)),
                boundary_port=late_output.name,
                matrix=torch.eye(2, dtype=FLOAT64),
                cut_edge_sha256=_sha(f"{graph_id}.late-output-cut"),
            ),
        ),
    )
    expansion = _generator(
        graph,
        _decomposition(graph.boundary_transfer()),
        parent_id=f"{graph_id}.parent",
    ).factorized_expansion()
    upstream, downstream = expansion.recursive_graph.components
    connection = DirectModalConnection(
        source_component=upstream.component_id,
        source_port=next(iter(upstream.output_ports)),
        target_component=downstream.component_id,
        target_port=next(iter(downstream.input_ports)),
        matrix=_matrix([[0.2, -0.1], [0.3, 0.4]]),
        evidence_kind="direct_jacobian",
        evidence_sha256=_sha(f"{graph_id}.direct-modal-jacobian"),
    )
    measured_graph = expansion.modal_core_graph(
        connections=(connection,),
    )
    input_fishers = (
        _matrix([[1.8, 0.15], [0.15, 0.7]]),
        _matrix([[0.9, -0.2], [-0.2, 1.4]]),
    )
    input_moments = tuple(
        MessageMoments(
            port=port,
            source_level_sha256=measured_graph.artifact_sha256,
            reduction_id=mode_moments.reduction_id,
            sample_count=mode_moments.sample_count,
            mean=torch.zeros(port.width, dtype=FLOAT64),
            covariance=mode_moments.covariance,
            fisher=fisher,
        )
        for port, mode_moments, fisher in zip(
            measured_graph.boundary_inputs,
            expansion.mode_moments,
            input_fishers,
            strict=True,
        )
    )
    input_covariance_by_name = {
        port.name: mode_moments.covariance
        for port, mode_moments in zip(
            measured_graph.boundary_inputs,
            expansion.mode_moments,
            strict=True,
        )
    }
    measured_transfer = measured_graph.boundary_transfer()
    covariances = tuple(
        matrix
        @ torch.block_diag(
            *(
                input_covariance_by_name[name]
                for name in prefix
            )
        )
        @ matrix.T
        for matrix, prefix in zip(
            measured_transfer.transfer_matrices,
            measured_transfer.input_prefixes,
            strict=True,
        )
    )
    fishers = (
        _matrix([[1.7, -0.2], [-0.2, 0.9]]),
        _matrix([[0.8, 0.25], [0.25, 1.6]]),
    )
    means = (
        torch.zeros(2, dtype=FLOAT64),
        torch.zeros(2, dtype=FLOAT64),
    )
    output_moments = tuple(
        MessageMoments(
            port=port,
            source_level_sha256=measured_graph.artifact_sha256,
            reduction_id=expansion.mode_moments[0].reduction_id,
            sample_count=expansion.mode_moments[0].sample_count,
            mean=mean,
            covariance=covariance,
            fisher=fisher,
        )
        for port, mean, covariance, fisher in zip(
            measured_graph.boundary_outputs,
            means,
            covariances,
            fishers,
            strict=True,
        )
    )
    return (
        expansion,
        connection,
        measured_graph,
        input_moments,
        output_moments,
    )


def test_measured_modal_core_uses_general_output_message_moments() -> None:
    (
        expansion,
        connection,
        measured_graph,
        input_moments,
        output_moments,
    ) = _general_measured_modal_core_fixture()
    measured = expansion.measured_modal_core(
        connections=(connection,),
        input_moments=input_moments,
        output_moments=output_moments,
    )

    assert measured.graph.artifact_sha256 == measured_graph.artifact_sha256
    assert tuple(
        moments.artifact_sha256 for moments in measured.output_moments
    ) == tuple(
        moments.artifact_sha256 for moments in output_moments
    )
    assert not torch.equal(
        measured.input_moments[0].fisher,
        expansion.mode_moments[0].fisher,
    )
    assert not torch.equal(
        measured.output_moments[0].covariance,
        measured.output_moments[0].fisher,
    )
    assert bool(
        torch.count_nonzero(
            measured.output_moments[1].covariance
            - torch.diag(torch.diag(measured.output_moments[1].covariance))
        ).item()
    )

    with pytest.raises(ValueError, match="measured joint covariance"):
        measured.factor_connectivity(retained_ranks=(2, 2))


@pytest.mark.parametrize(
    ("replacement"),
    (
        {"reduction_id": "different-measurement-reduction"},
        {"sample_count": 33},
    ),
)
def test_measured_modal_core_rejects_mixed_measurement_lineage(
    replacement: dict[str, object],
) -> None:
    (
        expansion,
        connection,
        _,
        input_moments,
        output_moments,
    ) = _general_measured_modal_core_fixture()
    altered = replace(
        output_moments[-1],
        **replacement,
        mean_sha256="",
        covariance_sha256="",
        fisher_sha256="",
        artifact_sha256="",
    )

    with pytest.raises(ValueError):
        measured = expansion.measured_modal_core(
            connections=(connection,),
            input_moments=input_moments,
            output_moments=output_moments[:-1] + (altered,),
        )
        measured.factor_connectivity()
