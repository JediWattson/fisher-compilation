from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import fisher_graph
from fisher_graph.gemma3_l3_l4_hierarchy_experiment import (
    _rank_curve,
    _validate_configuration,
    build_parser,
)
from fisher_graph.modal_connectivity_modes import (
    CausalBoundaryTransfer,
    MessageMoments,
    ModalBoundaryPort,
    ModalConnectivityFactor,
    factor_modal_connectivity,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_factor(name: str) -> ModalConnectivityFactor:
    source_sha256 = _sha256(f"source:{name}")
    input_port = ModalBoundaryPort(
        name=f"{name}.input",
        direction="input",
        causal_order=0,
        width=2,
        owner_id=name,
    )
    output_port = ModalBoundaryPort(
        name=f"{name}.output",
        direction="output",
        causal_order=0,
        width=2,
        owner_id=name,
    )
    transfer = CausalBoundaryTransfer(
        source_level_sha256=source_sha256,
        input_ports=(input_port,),
        output_ports=(output_port,),
        input_prefixes=((input_port.name,),),
        transfer_matrices=(
            torch.diag(torch.tensor((3.0, 1.0), dtype=torch.float64)),
        ),
        affine_offsets=(torch.zeros(2, dtype=torch.float64),),
    )

    def moments(port: ModalBoundaryPort) -> MessageMoments:
        return MessageMoments(
            port=port,
            source_level_sha256=source_sha256,
            reduction_id="synthetic-rank-curve",
            sample_count=8,
            mean=torch.zeros(2, dtype=torch.float64),
            covariance=torch.eye(2, dtype=torch.float64),
            fisher=torch.eye(2, dtype=torch.float64),
        )

    decomposition = factor_modal_connectivity(
        transfer,
        (moments(input_port),),
        (moments(output_port),),
        retained_ranks=2,
    )
    return decomposition.factors[0]


def test_new_l3_l4_analysis_apis_are_public() -> None:
    expected = {
        "CausalEdgeJVPFit",
        "CausalModalPairPlan",
        "EdgeTornModalPairBoundaryContract",
        "GeneratorMessageCapturePlan",
        "JointMessageMomentsResult",
        "MEAN_SOURCE_REFERENCE_TORN_BASE_SEMANTICS",
        "PreparedCausalModalPair",
        "StreamingJointMessageMoments",
        "apply_causal_lag_convolution",
        "bind_causal_modal_pair_plan",
        "estimate_causal_edge_jvp",
        "iter_generator_message_score_gradient_rows",
    }

    assert expected.issubset(fisher_graph.__all__)
    for name in expected:
        assert getattr(fisher_graph, name) is not None


def test_configuration_requires_an_unwritten_pt_artifact(tmp_path: Path) -> None:
    output = tmp_path / "l3-l4.pt"
    assert _validate_configuration(
        output=output,
        ranks=(1, 2),
        edge_rank=1,
        max_lag=2,
        probe_count=4,
        probe_sequences=1,
        ridge=1e-6,
        fit_limit=2,
    ) == (1, 2)

    output.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite hierarchy measurement",
    ):
        _validate_configuration(
            output=output,
            ranks=(1, 2),
            edge_rank=1,
            max_lag=2,
            probe_count=4,
            probe_sequences=1,
            ridge=1e-6,
            fit_limit=2,
        )


def test_rank_curve_is_explicitly_shape_only() -> None:
    layer3 = _synthetic_factor("layer3")
    layer4 = _synthetic_factor("layer4")

    curve = _rank_curve(
        factor3=layer3,
        factor4=layer4,
        ranks=(1, 2),
        max_lag=2,
        flat_pair_parameters=16,
        flat_pair_macs=8,
        full_stack_parameters=100,
        full_stack_macs=80,
        whole_model_parameters=1_000,
    )

    rank1 = curve[0]
    assert rank1["rank"] == 1
    assert rank1["lag_count"] == 3
    assert rank1["lag_edge_parameters"] == 3
    assert rank1["stored_mean_scalars"] == 8
    assert rank1["centering_folded_into_bias"] is False
    assert rank1["candidate_pair_parameters"] == 19
    assert rank1["candidate_pair_macs_per_token"] == 11
    assert rank1["candidate_full_stack_parameters"] == 103
    assert rank1["candidate_full_stack_macs_per_token"] == 83
    assert rank1["candidate_whole_model_parameters"] == 1_003
    assert rank1["minimum_retained_weighted_energy_fraction"] == pytest.approx(
        0.9
    )
    assert rank1["weighted_energy_is_fidelity_metric"] is False
    assert rank1["accounting_status"] == (
        "logical_shape_only_until_reference_base_executor_is_built"
    )


def test_repeated_factorization_has_a_stable_artifact_digest() -> None:
    first = _synthetic_factor("repeatable")
    second = _synthetic_factor("repeatable")

    assert first.artifact_sha256 == second.artifact_sha256
    torch.testing.assert_close(first.restriction, second.restriction)
    torch.testing.assert_close(first.prolongation, second.prolongation)


def test_cli_exposes_the_measurement_controls(tmp_path: Path) -> None:
    output = tmp_path / "measurement.pt"
    arguments = build_parser().parse_args(
        [
            "--output",
            str(output),
            "--ranks",
            "16",
            "32",
            "--edge-rank",
            "32",
            "--max-lag",
            "3",
            "--probe-count",
            "6",
            "--probe-sequences",
            "2",
        ]
    )

    assert arguments.output == output
    assert arguments.ranks == [16, 32]
    assert arguments.edge_rank == 32
    assert arguments.max_lag == 3
    assert arguments.probe_count == 6
    assert arguments.probe_sequences == 2
