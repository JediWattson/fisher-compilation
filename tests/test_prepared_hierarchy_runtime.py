from __future__ import annotations

import hashlib

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
from fisher_graph.prepared_hierarchy_runtime import (
    PreparedTorchHierarchyRuntime,
)


FLOAT64 = torch.float64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _port(
    *,
    name: str,
    direction: str,
    causal_order: int,
    width: int,
) -> ModalBoundaryPort:
    return ModalBoundaryPort(
        name=name,
        direction=direction,
        causal_order=causal_order,
        width=width,
        owner_id="prepared-runtime-fixture",
    )


def _moments(
    port: ModalBoundaryPort,
    *,
    source_level_sha256: str,
) -> MessageMoments:
    return MessageMoments(
        port=port,
        source_level_sha256=source_level_sha256,
        reduction_id="prepared-runtime-reduction",
        sample_count=64,
        mean=torch.arange(port.width, dtype=FLOAT64) / 4.0,
        covariance=torch.eye(port.width, dtype=FLOAT64),
        fisher=torch.eye(port.width, dtype=FLOAT64),
    )


def _decomposition(
    retained_ranks: tuple[int, int],
) -> ModalConnectivityDecomposition:
    source_sha256 = _sha("prepared-runtime-source")
    x0 = _port(
        name="prepared.x0",
        direction="input",
        causal_order=0,
        width=2,
    )
    x1 = _port(
        name="prepared.x1",
        direction="input",
        causal_order=1,
        width=1,
    )
    y0 = _port(
        name="prepared.y0",
        direction="output",
        causal_order=0,
        width=2,
    )
    y1 = _port(
        name="prepared.y1",
        direction="output",
        causal_order=1,
        width=1,
    )
    transfer = CausalBoundaryTransfer(
        source_level_sha256=source_sha256,
        input_ports=(x0, x1),
        output_ports=(y0, y1),
        input_prefixes=((x0.name,), (x0.name, x1.name)),
        transfer_matrices=(
            torch.tensor(
                [[2.0, -1.0], [0.5, 3.0]],
                dtype=FLOAT64,
            ),
            torch.tensor([[1.0, -2.0, 0.25]], dtype=FLOAT64),
        ),
        affine_offsets=(
            torch.tensor([0.75, -1.25], dtype=FLOAT64),
            torch.tensor([2.5], dtype=FLOAT64),
        ),
    )
    return factor_modal_connectivity(
        transfer,
        (
            _moments(x0, source_level_sha256=source_sha256),
            _moments(x1, source_level_sha256=source_sha256),
        ),
        (
            _moments(y0, source_level_sha256=source_sha256),
            _moments(y1, source_level_sha256=source_sha256),
        ),
        retained_ranks=retained_ranks,
        assume_block_diagonal_input_covariance=True,
    )


def _inputs() -> tuple[Tensor, Tensor]:
    return (
        torch.tensor(
            [[1.0, -2.0], [0.25, 4.0], [-3.0, 1.5]],
            dtype=FLOAT64,
        ),
        torch.tensor([[2.0], [-1.0], [0.5]], dtype=FLOAT64),
    )


def test_full_rank_prepared_paths_equal_the_exact_source() -> None:
    decomposition = _decomposition((2, 1))
    runtime = PreparedTorchHierarchyRuntime.from_decomposition(
        decomposition,
        device="cpu",
        dtype=FLOAT64,
    )
    inputs = _inputs()
    expected = decomposition.source_transfer.execute(
        dict(zip(runtime.input_names, inputs, strict=True))
    )

    assert (
        runtime.source_transfer_sha256
        == decomposition.source_transfer.artifact_sha256
    )
    assert runtime.decomposition_sha256 == decomposition.artifact_sha256
    for outputs in (
        runtime.source_dense(inputs),
        runtime.candidate_dense(inputs),
        runtime.candidate_factorized(inputs),
    ):
        for name, value in zip(runtime.output_names, outputs, strict=True):
            torch.testing.assert_close(
                value,
                expected[name],
                rtol=1e-12,
                atol=1e-12,
            )


def test_reduced_rank_dense_and_factorized_candidates_are_equivalent() -> None:
    decomposition = _decomposition((1, 1))
    runtime = PreparedTorchHierarchyRuntime.from_decomposition(
        decomposition,
        device=torch.device("cpu"),
        dtype=FLOAT64,
    )

    dense = runtime.candidate_dense(_inputs())
    factorized = runtime.candidate_factorized(_inputs())
    for dense_value, factorized_value in zip(
        dense,
        factorized,
        strict=True,
    ):
        torch.testing.assert_close(
            dense_value,
            factorized_value,
            rtol=1e-12,
            atol=1e-12,
        )

    assert not torch.allclose(
        runtime.source_dense(_inputs())[0],
        factorized[0],
    )


def test_hot_paths_do_not_revalidate_or_convert_runtime_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decomposition = _decomposition((1, 1))
    runtime = PreparedTorchHierarchyRuntime.from_decomposition(
        decomposition,
        device="cpu",
        dtype=FLOAT64,
    )

    def unexpected_validation(
        self: ModalConnectivityDecomposition,
    ) -> None:
        raise AssertionError("hot path revalidated the decomposition")

    def unexpected_conversion(
        self: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        raise AssertionError("hot path called Tensor.to")

    monkeypatch.setattr(
        ModalConnectivityDecomposition,
        "validate_integrity",
        unexpected_validation,
    )
    monkeypatch.setattr(torch.Tensor, "to", unexpected_conversion)

    inputs = _inputs()
    runtime.source_dense(inputs)
    runtime.candidate_dense(inputs)
    runtime.candidate_factorized(inputs)


def test_prepared_input_contract_rejects_noncanonical_values() -> None:
    runtime = PreparedTorchHierarchyRuntime.from_decomposition(
        _decomposition((1, 1)),
        device="cpu",
        dtype=FLOAT64,
    )
    x0, x1 = _inputs()

    with pytest.raises(TypeError, match="canonical Tensor tuple"):
        runtime.source_dense([x0, x1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input count"):
        runtime.source_dense((x0,))
    with pytest.raises(TypeError, match="floating Tensor"):
        runtime.source_dense((x0.to(torch.int64), x1))
    with pytest.raises(ValueError, match="wrong width"):
        runtime.source_dense((x0[:, :1], x1))
    with pytest.raises(ValueError, match="wrong dtype"):
        runtime.source_dense((x0.to(torch.float32), x1))
    with pytest.raises(ValueError, match="same leading shape"):
        runtime.source_dense((x0, x1[:2]))
    with pytest.raises(ValueError, match="wrong device"):
        runtime.source_dense(
            (
                torch.empty((3, 2), dtype=FLOAT64, device="meta"),
                torch.empty((3, 1), dtype=FLOAT64, device="meta"),
            )
        )


def test_prepared_accounting_counts_each_resident_control_exactly() -> None:
    runtime = PreparedTorchHierarchyRuntime.from_decomposition(
        _decomposition((1, 1)),
        device="cpu",
        dtype=FLOAT64,
    )
    accounting = runtime.accounting

    # Source: matrices 2x2 and 1x3, plus output biases of widths 2 and 1.
    assert accounting.source_dense_macs_per_row == 7
    assert accounting.source_dense_stored_scalar_count == 10
    # Dense candidate materializes the same matrix shapes and folded biases.
    assert accounting.candidate_dense_macs_per_row == 7
    assert accounting.candidate_dense_stored_scalar_count == 10
    # Factor ranks (1, 1): R/P maps cost 4 + 4, plus three bias scalars.
    assert accounting.candidate_factorized_macs_per_row == 8
    assert accounting.candidate_factorized_stored_scalar_count == 11
    # The dense and factorized candidates share their folded bias tensors.
    assert accounting.prepared_unique_stored_scalar_count == 28
    assert accounting.bytes_per_scalar == 8
    assert accounting.prepared_unique_storage_bytes == 224
    assert accounting.analysis_only is True
    assert accounting.source_dense_is_benchmark_control is True
    assert accounting.authorizes_replacement is False
    assert accounting.authorizes_fallback is False
    assert accounting.deployed_storage_reduction_authorized is False
    assert accounting.latency_reduction_claim is False


@pytest.mark.parametrize(
    "dtype",
    [torch.int64, torch.complex64],
)
def test_preparation_rejects_non_runtime_floating_dtypes(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(ValueError, match="supported floating"):
        PreparedTorchHierarchyRuntime.from_decomposition(
            _decomposition((1, 1)),
            device="cpu",
            dtype=dtype,
        )
