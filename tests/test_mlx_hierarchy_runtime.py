from __future__ import annotations

import hashlib
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fisher_graph.mlx_hierarchy_runtime import (
    PreparedMLXHierarchyRuntime,
    mlx_hierarchy_is_installed,
    mlx_hierarchy_is_usable,
)
from fisher_graph.modal_connectivity_modes import (
    CausalBoundaryTransfer,
    MessageMoments,
    ModalBoundaryPort,
    factor_modal_connectivity,
)


FLOAT64 = torch.float64
_MLX_USABLE = mlx_hierarchy_is_usable()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _assert_mlx_close(actual, expected: np.ndarray) -> None:
    actual_array = np.array(actual, copy=True)
    expected_array = np.asarray(expected)
    difference = (
        actual_array.astype(np.float64)
        - expected_array.astype(np.float64)
    )
    denominator = float(
        np.linalg.norm(expected_array.astype(np.float64))
    )
    relative_rms = float(np.linalg.norm(difference)) / max(
        denominator,
        np.finfo(np.float64).tiny,
    )
    peak_scale = max(
        float(np.max(np.abs(expected_array))),
        np.finfo(np.float64).tiny,
    )
    peak_relative = float(np.max(np.abs(difference))) / peak_scale
    assert relative_rms < 0.005
    assert peak_relative < 0.01


def _decomposition(
    *,
    retained_ranks: tuple[int, int] = (1, 1),
):
    graph_id = "mlx-hierarchy-test"
    input_0 = ModalBoundaryPort(
        name=f"{graph_id}.x0",
        direction="input",
        causal_order=0,
        width=2,
        owner_id=graph_id,
    )
    input_1 = ModalBoundaryPort(
        name=f"{graph_id}.x1",
        direction="input",
        causal_order=1,
        width=1,
        owner_id=graph_id,
    )
    output_0 = ModalBoundaryPort(
        name=f"{graph_id}.y0",
        direction="output",
        causal_order=0,
        width=2,
        owner_id=graph_id,
    )
    output_1 = ModalBoundaryPort(
        name=f"{graph_id}.y1",
        direction="output",
        causal_order=1,
        width=2,
        owner_id=graph_id,
    )
    transfer = CausalBoundaryTransfer(
        source_level_sha256=_sha("mlx-hierarchy-source"),
        input_ports=(input_0, input_1),
        output_ports=(output_0, output_1),
        input_prefixes=(
            (input_0.name,),
            (input_0.name, input_1.name),
        ),
        transfer_matrices=(
            torch.tensor(
                [[2.0, -0.5], [0.25, 1.5]],
                dtype=FLOAT64,
            ),
            torch.tensor(
                [[1.0, -2.0, 0.5], [0.25, 1.0, 3.0]],
                dtype=FLOAT64,
            ),
        ),
        affine_offsets=(
            torch.tensor([0.1, -0.2], dtype=FLOAT64),
            torch.tensor([-0.3, 0.4], dtype=FLOAT64),
        ),
    )
    input_moments = tuple(
        MessageMoments(
            port=port,
            source_level_sha256=transfer.source_level_sha256,
            reduction_id="mlx-hierarchy-test-reduction",
            sample_count=32,
            mean=torch.zeros(port.width, dtype=FLOAT64),
            covariance=torch.eye(port.width, dtype=FLOAT64),
            fisher=torch.eye(port.width, dtype=FLOAT64),
        )
        for port in transfer.input_ports
    )
    output_moments = tuple(
        MessageMoments(
            port=port,
            source_level_sha256=transfer.source_level_sha256,
            reduction_id="mlx-hierarchy-test-reduction",
            sample_count=32,
            mean=transfer.affine_offsets[index],
            covariance=torch.eye(port.width, dtype=FLOAT64),
            fisher=torch.eye(port.width, dtype=FLOAT64),
        )
        for index, port in enumerate(transfer.output_ports)
    )
    return factor_modal_connectivity(
        transfer,
        input_moments,
        output_moments,
        retained_ranks=retained_ranks,
        assume_block_diagonal_input_covariance=True,
    )


def test_module_import_does_not_import_optional_mlx_package() -> None:
    script = """
import sys
assert "mlx" not in sys.modules
import fisher_graph.mlx_hierarchy_runtime
assert "mlx" not in sys.modules
assert "mlx.core" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_mlx_discovery_short_circuits_when_parent_is_absent() -> None:
    def find_spec(name: str):
        if name == "mlx":
            return None
        raise ModuleNotFoundError(name)

    with patch(
        "fisher_graph.mlx_hierarchy_runtime.importlib.util.find_spec",
        side_effect=find_spec,
    ):
        assert mlx_hierarchy_is_installed() is False


def test_runtime_rejects_non_decomposition_before_loading_mlx() -> None:
    with pytest.raises(
        TypeError,
        match="ModalConnectivityDecomposition",
    ):
        PreparedMLXHierarchyRuntime.from_decomposition(object())  # type: ignore[arg-type]


@pytest.mark.skipif(
    not _MLX_USABLE,
    reason="MLX device execution is unavailable",
)
def test_compiled_hierarchy_systems_match_torch_and_each_other() -> None:
    import mlx.core as mx

    decomposition = _decomposition()
    runtime = PreparedMLXHierarchyRuntime.from_decomposition(
        decomposition
    )
    torch_inputs = (
        torch.tensor(
            [[1.0, -2.0], [0.5, 0.25], [-1.0, 3.0]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.75], [-2.0], [1.25]],
            dtype=torch.float32,
        ),
    )
    mlx_inputs = tuple(
        mx.array(value.numpy(), dtype=mx.float32)
        for value in torch_inputs
    )

    source = runtime.source_dense(mlx_inputs)
    candidate_dense = runtime.candidate_dense(mlx_inputs)
    candidate_factorized = runtime.candidate_factorized(mlx_inputs)
    mx.eval(*(source + candidate_dense + candidate_factorized))
    mx.synchronize()

    torch_mapping = dict(
        zip(runtime.input_port_names, torch_inputs, strict=True)
    )
    expected_source = decomposition.source_transfer.execute(torch_mapping)
    expected_candidate = decomposition.execute_candidate(torch_mapping)
    for index, name in enumerate(runtime.output_port_names):
        _assert_mlx_close(
            source[index],
            expected_source[name].numpy(),
        )
        _assert_mlx_close(
            candidate_dense[index],
            expected_candidate[name].numpy(),
        )
        _assert_mlx_close(
            candidate_factorized[index],
            np.array(candidate_dense[index]),
        )

    provenance = runtime.runtime_provenance()
    assert (
        provenance["source_decomposition_sha256"]
        == decomposition.artifact_sha256
    )
    assert provenance["state_sha256"] == runtime.state_sha256
    assert provenance["analysis_only"] is True
    assert provenance["benchmark_only"] is True
    assert provenance["authorizes_replacement"] is False
    assert provenance["available_systems"] == (
        "source_dense",
        "candidate_dense",
        "candidate_factorized",
    )
    assert runtime.state_bytes == sum(
        runtime.system_state_bytes.values()
    )
    assert runtime.system_stored_scalar_counts == {
        system: byte_count // 4
        for system, byte_count in runtime.system_state_bytes.items()
    }
    state_names = set(runtime.state_dict())
    assert {
        "candidate_factorized.output_bias.000",
        "candidate_factorized.output_bias.001",
    } <= state_names
    assert not any("input_mean" in name for name in state_names)
    assert not any("output_mean" in name for name in state_names)
    assert provenance["factorized_centering"] == "folded_output_bias"
    assert provenance["factorized_hot_path_stores_input_mean"] is False
    assert provenance["factorized_hot_path_stores_output_mean"] is False
    exposed_bytes = runtime.system_state_bytes
    exposed_bytes["source_dense"] = 0
    assert runtime.system_state_bytes["source_dense"] > 0
    exposed_provenance = runtime.runtime_provenance()
    exposed_provenance["system_state_bytes"]["source_dense"] = 0  # type: ignore[index]
    assert (
        runtime.runtime_provenance()["system_state_bytes"]["source_dense"]  # type: ignore[index]
        > 0
    )
    assert len(runtime.artifact_sha256) == 64


@pytest.mark.skipif(
    not _MLX_USABLE,
    reason="MLX device execution is unavailable",
)
def test_runtime_owns_state_and_validates_canonical_inputs() -> None:
    import mlx.core as mx

    decomposition = _decomposition()
    runtime = PreparedMLXHierarchyRuntime.from_decomposition(
        decomposition
    )
    inputs = (
        mx.zeros((2, 2), dtype=mx.float32),
        mx.zeros((2, 1), dtype=mx.float32),
    )
    before = runtime.source_dense(inputs)
    mx.eval(*before)
    decomposition.source_transfer.transfer_matrices[0].add_(100.0)
    after = runtime.source_dense(inputs)
    mx.eval(*after)
    mx.synchronize()
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(np.array(left), np.array(right))

    with pytest.raises(TypeError, match="must be a tuple"):
        runtime.source_dense(list(inputs))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wrong width"):
        runtime.source_dense(
            (
                mx.zeros((2, 1), dtype=mx.float32),
                inputs[1],
            )
        )
    with pytest.raises(AttributeError, match="immutable"):
        runtime.state_bytes = 0


@pytest.mark.skipif(
    not _MLX_USABLE,
    reason="MLX device execution is unavailable",
)
def test_rank_zero_factorized_path_broadcasts_folded_bias() -> None:
    import mlx.core as mx

    decomposition = _decomposition(retained_ranks=(0, 0))
    runtime = PreparedMLXHierarchyRuntime.from_decomposition(
        decomposition
    )
    inputs = (
        mx.array([[4.0, -3.0], [1.0, 2.0]], dtype=mx.float32),
        mx.array([[0.5], [-2.0]], dtype=mx.float32),
    )
    dense = runtime.candidate_dense(inputs)
    factorized = runtime.candidate_factorized(inputs)
    mx.eval(*(dense + factorized))
    mx.synchronize()

    for dense_output, factorized_output in zip(
        dense,
        factorized,
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.array(factorized_output),
            np.array(dense_output),
        )
    assert runtime.candidate_factorized_macs_per_row == 0
    expected_scalars = sum(runtime.output_widths)
    assert (
        runtime.system_stored_scalar_counts["candidate_factorized"]
        == expected_scalars
    )
