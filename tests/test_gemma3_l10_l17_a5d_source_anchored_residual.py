from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from fisher_graph.computational_modes import (
    ComputationalModeBasis,
    ComputationalModeBinding,
    ComputationalModeConfig,
)
from fisher_graph.gemma3_l10_l17_a5d_source_anchored_residual import (
    _validate_error_statistics,
    build_a5d_source_anchored_residual_targets,
    project_source_anchored_residual_to_joint_decoder_span,
    validate_a5d_source_anchored_residual_receipt,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import (
    LayerFragmentRows,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
)


NODE_ORDER = ("node.zeta", "node.alpha", "node.theta", "node.beta")
RANKS = (48, 38, 48, 48)
WIDTH = 186
FRAGMENTS = {
    name: f"fragment.{index}" for index, name in enumerate(NODE_ORDER)
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bases(*, mean_scale: float = 1.0) -> dict[str, ComputationalModeBasis]:
    result: dict[str, ComputationalModeBasis] = {}
    start = 0
    for index, (name, rank) in enumerate(zip(NODE_ORDER, RANKS, strict=True)):
        stop = start + rank
        binding = ComputationalModeBinding.create(
            mode_set_id=f"{name}/modes",
            source_kind="relocated_layer_fragment",
            output_site="layer.17.mlp.delta",
            source_model_sha256="a" * 64,
            parameter_catalog_sha256="b" * 64,
            fisher_coupling_sha256="c" * 64,
            parameter_cluster_sha256=_sha(name),
            fit_split_sha256="d" * 64,
            eval_split_sha256="e" * 64,
        )
        encoder = torch.eye(WIDTH, dtype=torch.float64)[:, start:stop]
        result[name] = ComputationalModeBasis(
            binding=binding,
            config=ComputationalModeConfig(ranks=(rank,)),
            rank=rank,
            mean_bias=torch.full(
                (WIDTH,),
                mean_scale * float(index + 1),
                dtype=torch.float64,
            ),
            encoder_basis=encoder,
        )
        start = stop
    return result


def _fixture():
    generator = torch.Generator().manual_seed(58_031)
    bases = _bases(mean_scale=3.0)
    observations = 7
    inputs = torch.randn(
        observations, 5, generator=generator, dtype=torch.float64
    )
    row_keys = tuple((f"example-{index // 2}", index % 2) for index in range(6))
    row_keys += (("example-3", 0),)
    coordinates = {
        name: torch.randn(
            observations,
            basis.rank,
            generator=generator,
            dtype=torch.float64,
        )
        for name, basis in bases.items()
    }
    oracle_rows = AlignedFragmentRows(
        rows_by_fragment={
            FRAGMENTS[name]: LayerFragmentRows(
                inputs=inputs,
                contributions=bases[name].decode(coordinates[name]),
                fisher_weights=torch.linspace(
                    1.0 + index,
                    2.0 + index,
                    observations,
                    dtype=torch.float64,
                ),
                sequences=4,
            )
            for index, name in enumerate(NODE_ORDER)
        },
        row_keys=row_keys,
    )
    correction_base = torch.randn(
        observations, WIDTH, generator=generator, dtype=torch.float64
    )
    # Preserve deliberately out-of-span source components in the final four
    # dimensions.  A faulty implementation which projects the source itself
    # would erase these values at alpha zero.
    exact_source_correction = torch.randn(
        observations, WIDTH, generator=generator, dtype=torch.float64
    )
    exact_source_correction[:, -4:] += 20.0
    frozen = correction_base + exact_source_correction
    targets = build_a5d_source_anchored_residual_targets(
        frozen_compiled_block_states=frozen,
        compiled_correction_base_states=correction_base,
        oracle_rows=oracle_rows,
        bases_by_node=bases,
        node_order=NODE_ORDER,
        fragment_id_by_node=FRAGMENTS,
    )
    return targets, bases, coordinates, frozen, correction_base, oracle_rows


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(child) for child in value)
    return False


def test_zero_mean_projection_reconstructs_only_the_joint_decoder_span() -> None:
    targets, bases, _, _, _, _ = _fixture()
    projection = targets.projection

    expected = projection.residual_target.clone()
    expected[:, -4:] = 0.0
    torch.testing.assert_close(
        projection.prediction, expected, rtol=1.0e-12, atol=1.0e-12
    )
    error = projection.prediction - projection.residual_target
    torch.testing.assert_close(
        error @ projection.combined_decoder.T,
        torch.zeros(
            error.shape[0], sum(RANKS), dtype=torch.float64
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    start = 0
    for name, rank in zip(NODE_ORDER, RANKS, strict=True):
        stop = start + rank
        contribution = projection.contributions_by_node[name]
        expected_node = (
            projection.joint_coordinates[:, start:stop]
            @ bases[name].decoder_basis
        )
        torch.testing.assert_close(contribution, expected_node)
        # The source means are large and nonzero.  The residual target must be
        # the zero-mean decode, not ComputationalModeBasis.decode().
        affine_decode = bases[name].decode(
            projection.joint_coordinates[:, start:stop]
        )
        torch.testing.assert_close(
            affine_decode - contribution,
            bases[name].mean.expand_as(contribution),
        )
        assert not torch.equal(affine_decode, contribution)
        start = stop


def test_builder_preserves_exact_source_at_alpha_zero() -> None:
    targets, _, _, frozen, base, _ = _fixture()

    assert torch.equal(
        targets.projection.exact_source_correction,
        frozen - base,
    )
    alpha_zero = targets.candidate_block_states(0.0)
    assert torch.equal(alpha_zero, frozen)
    # In particular, out-of-span source coordinates remain untouched.
    assert torch.equal(alpha_zero[:, -4:], frozen[:, -4:])
    torch.testing.assert_close(
        targets.candidate_block_states(1.0),
        frozen + targets.projection.prediction,
    )
    torch.testing.assert_close(
        targets.candidate_block_states(0.25),
        frozen + 0.25 * targets.projection.prediction,
    )


def test_residual_rows_share_oracle_inputs_weights_and_zero_mean_targets() -> None:
    targets, _, _, _, _, oracle_rows = _fixture()

    assert targets.residual_rows.row_keys == oracle_rows.row_keys
    for name in NODE_ORDER:
        fragment = FRAGMENTS[name]
        residual = targets.residual_rows.rows_by_fragment[fragment]
        oracle = oracle_rows.rows_by_fragment[fragment]
        assert torch.equal(residual.inputs, oracle.inputs)
        assert torch.equal(residual.fisher_weights, oracle.fisher_weights)
        assert torch.equal(
            residual.contributions,
            targets.projection.contributions_by_node[name],
        )


def test_direct_projection_matches_builder_projection() -> None:
    targets, bases, _, _, _, _ = _fixture()
    direct = project_source_anchored_residual_to_joint_decoder_span(
        targets.projection.exact_source_correction,
        targets.projection.oracle_correction,
        bases_by_node=bases,
        node_order=NODE_ORDER,
    )

    assert direct.metadata() == targets.projection.metadata()
    assert torch.equal(direct.joint_coordinates, targets.projection.joint_coordinates)
    assert torch.equal(direct.prediction, targets.projection.prediction)


def test_receipt_is_strict_tensor_free_and_rejects_tampering() -> None:
    targets, _, _, _, _, _ = _fixture()
    receipt = targets.receipt()

    assert not _contains_tensor(receipt)
    assert targets.receipt_sha256 == receipt["receipt_sha256"]
    assert targets.node_order == NODE_ORDER
    assert targets.residual_width == WIDTH
    assert validate_a5d_source_anchored_residual_receipt(receipt) == receipt
    assert receipt["construction"]["source_state_is_projected"] is False
    assert receipt["rows"]["source_affine_means_injected"] is False
    assert receipt["projection"]["combined_decoder_matrix_rank"] == sum(RANKS)

    tampered = copy.deepcopy(receipt)
    tampered["construction"]["source_state_is_projected"] = True
    with pytest.raises(ValueError, match="construction"):
        validate_a5d_source_anchored_residual_receipt(tampered)

    tampered = copy.deepcopy(receipt)
    tampered["projection"]["projected_residual_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt hash"):
        validate_a5d_source_anchored_residual_receipt(tampered)


def test_in_memory_row_tamper_fails_closed() -> None:
    targets, _, _, _, _, _ = _fixture()
    fragment = FRAGMENTS[NODE_ORDER[0]]
    targets.residual_rows.rows_by_fragment[fragment].contributions[0, 0] += 1.0

    with pytest.raises(ValueError, match="fit rows drifted"):
        targets.receipt()
    with pytest.raises(ValueError, match="fit rows drifted"):
        targets.candidate_block_states(0.0)


def test_nested_stored_receipt_tamper_fails_closed() -> None:
    targets, _, _, _, _, _ = _fixture()
    targets._receipt["construction"]["source_state_is_projected"] = True

    with pytest.raises(ValueError, match="construction"):
        targets.receipt()


def test_error_statistics_allow_scale_aware_rms_roundoff() -> None:
    maximum = 1.0e20
    rms = 1.0000000000000002e20
    _validate_error_statistics(
        {
            "max_abs_error": maximum,
            "rms_error": rms,
            "target_rms": rms,
            "nrmse": 1.0,
        },
        label="large finite diagnostic",
    )
@pytest.mark.parametrize("alpha", (-0.1, 1.1, float("nan")))
def test_candidate_alpha_must_be_bounded(alpha: float) -> None:
    targets, _, _, _, _, _ = _fixture()
    with pytest.raises(ValueError, match="alpha"):
        targets.candidate_block_states(alpha)
