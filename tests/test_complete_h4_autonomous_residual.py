from __future__ import annotations

import copy
import stat

import pytest
import torch

from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
    AutonomousCompleteH4TrainingSequence,
    autonomous_complete_h4_residual_provider_state_dict,
    fit_autonomous_complete_h4_output_decoder,
    fit_autonomous_complete_h4_residual,
    load_autonomous_complete_h4_residual_provider,
    save_autonomous_complete_h4_residual_provider,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)


_BRIDGE = "a" * 64
_WIDTH = 640
_SOURCE_RANK = 64


def _decoder(rank: int = 2) -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:rank].contiguous()


def _sequence(
    index: int,
    *,
    length: int = 24,
    first_source: int = 0,
    gradient_scale: float | None = None,
) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(1000 + index)
    positions = torch.arange(length, dtype=torch.int64)
    valid = torch.ones(length, dtype=torch.bool)
    source_mask = torch.arange(length) >= first_source
    support = source_mask.clone()
    source = torch.randn(
        length, _SOURCE_RANK, generator=generator, dtype=torch.float64
    )
    source[~source_mask] = 0
    base = torch.randn(
        length, _WIDTH, generator=generator, dtype=torch.float64
    )
    decoder = _decoder()
    source_kernel = torch.linspace(
        -0.02, 0.02, _SOURCE_RANK * 2, dtype=torch.float64
    ).reshape(
        _SOURCE_RANK, 2
    )
    state_kernel = torch.tensor(
        [[0.20, -0.10], [0.05, 0.15]], dtype=torch.float64
    )
    bias = torch.tensor([0.03, -0.04], dtype=torch.float64)
    modal = source @ source_kernel + (base @ decoder.T) @ state_kernel + bias
    modal[~support] = 0
    native = base + modal @ decoder
    gradients = None
    if gradient_scale is not None:
        gradients = torch.randn(
            length, _WIDTH, generator=generator, dtype=torch.float64
        )
        gradients[0].mul_(gradient_scale)
    return AutonomousCompleteH4TrainingSequence(
        example_id=f"example-{index}",
        family_id=f"family-{index % 2}",
        source_modes=source,
        logical_positions=positions,
        valid_mask=valid,
        source_mask=source_mask,
        support_mask=support,
        base_h4=base,
        native_h4=native,
        reverse_vjp_gradients=gradients,
    )


def _prefix(sequence: AutonomousCompleteH4TrainingSequence) -> Gemma3L3L4OnePassPrefix:
    length = sequence.source_modes.shape[0]
    return Gemma3L3L4OnePassPrefix(
        source_modes=sequence.source_modes.unsqueeze(0),
        clamped_y3=torch.zeros((1, length, _WIDTH), dtype=torch.float64),
        predicted_target_modal_delta=torch.zeros((1, length, 1), dtype=torch.float64),
        decoded_base_x4_delta=torch.zeros((1, length, _WIDTH), dtype=torch.float64),
        logical_positions=sequence.logical_positions.unsqueeze(0),
        valid_target_mask=sequence.valid_mask.unsqueeze(0),
        source_eligible_mask=sequence.source_mask.unsqueeze(0),
        target_affected_mask=sequence.support_mask.unsqueeze(0),
        bridge_binding_sha256=_BRIDGE,
    )


def _fit(*, weighted: bool = False):
    sequences = tuple(
        _sequence(index, gradient_scale=(1.0e12 if weighted else None))
        for index in range(6)
    )
    return fit_autonomous_complete_h4_residual(
        sequences=sequences,
        output_decoder=_decoder(),
        bridge_binding_sha256=_BRIDGE,
        lag_count=1,
        ridge=1.0e-12,
        fit_objective=(
            "reverse_vjp_row_weighted_ridge_v1"
            if weighted
            else "hidden_residual_ridge"
        ),
    )


def test_fit_recovers_linear_residual_and_writes_only_complete_support() -> None:
    training = tuple(_sequence(index) for index in range(6))
    originals = [
        tuple(tensor.clone() for tensor in (item.source_modes, item.base_h4, item.native_h4))
        for item in training
    ]
    provider = fit_autonomous_complete_h4_residual(
        sequences=training,
        output_decoder=_decoder(),
        bridge_binding_sha256=_BRIDGE,
        lag_count=1,
        ridge=1.0e-12,
    )
    held = _sequence(20, first_source=3)
    prefix = _prefix(held)
    base = held.base_h4.unsqueeze(0).clone()
    prefix_snapshot = prefix.artifact_sha256
    base_snapshot = base.clone()
    correction = provider.correction(prefix, base)
    expected = (held.native_h4 - held.base_h4).unsqueeze(0)
    support = prefix.complete_h4_causal_support_mask()
    torch.testing.assert_close(correction[support], expected[support], rtol=0, atol=2e-8)
    assert torch.count_nonzero(correction[~support]) == 0
    assert prefix.artifact_sha256 == prefix_snapshot
    torch.testing.assert_close(base, base_snapshot, rtol=0, atol=0)
    for item, snapshots in zip(training, originals, strict=True):
        for live, snapshot in zip(
            (item.source_modes, item.base_h4, item.native_h4), snapshots, strict=True
        ):
            torch.testing.assert_close(live, snapshot, rtol=0, atol=0)
    assert provider.prepared_float_scalar_count > 0
    assert provider.logical_macs_per_token_upper_bound > 0
    assert provider.metadata()["runtime_forbidden_inputs"] == (
        "native_h4",
        "targets",
        "logits",
        "gradients",
    )


def test_runtime_is_causal_under_future_source_and_state_changes() -> None:
    provider = _fit()
    sequence = _sequence(30)
    prefix = _prefix(sequence)
    base = sequence.base_h4.unsqueeze(0)
    reference = provider.correction(prefix, base)
    cutoff = 10

    changed_source = sequence.source_modes.clone()
    changed_source[cutoff + 1 :] += 1000.0
    changed_base = sequence.base_h4.clone()
    changed_base[cutoff + 1 :] -= 1000.0
    changed = AutonomousCompleteH4TrainingSequence(
        example_id="changed",
        family_id="family-x",
        source_modes=changed_source,
        logical_positions=sequence.logical_positions,
        valid_mask=sequence.valid_mask,
        source_mask=sequence.source_mask,
        support_mask=sequence.support_mask,
        base_h4=changed_base,
        native_h4=changed_base,
    )
    actual = provider.correction(_prefix(changed), changed_base.unsqueeze(0))
    torch.testing.assert_close(
        actual[:, : cutoff + 1], reference[:, : cutoff + 1], rtol=0, atol=0
    )


def test_authentication_geometry_pca_and_vjp_weight_bounds_fail_closed() -> None:
    sequences = tuple(_sequence(index, gradient_scale=1.0e20) for index in range(6))
    provider = fit_autonomous_complete_h4_residual(
        sequences=sequences,
        output_decoder=_decoder(),
        bridge_binding_sha256=_BRIDGE,
        lag_count=1,
        ridge=1.0e-8,
        fit_objective="reverse_vjp_row_weighted_ridge_v1",
        vjp_weight_floor=0.75,
        vjp_weight_ceiling=1.25,
    )
    assert 0.75 <= provider.observed_vjp_multiplier_min <= 1.0
    assert 1.0 <= provider.observed_vjp_multiplier_max <= 1.25
    pca = fit_autonomous_complete_h4_output_decoder(sequences, rank=2)
    torch.testing.assert_close(pca @ pca.T, torch.eye(2, dtype=torch.float64))

    with pytest.raises(ValueError, match="orthonormal"):
        fit_autonomous_complete_h4_residual(
            sequences=sequences,
            output_decoder=torch.ones((2, _WIDTH), dtype=torch.float64),
            bridge_binding_sha256=_BRIDGE,
            lag_count=1,
            ridge=1.0e-8,
        )
    provider.bias[0] += 1.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        provider.validate_integrity()

    malformed = copy.copy(sequences[0])
    malformed.support_mask[0] = False
    with pytest.raises(ValueError, match="causal closure"):
        AutonomousCompleteH4TrainingSequence(
            example_id="bad",
            family_id="bad-family",
            source_modes=malformed.source_modes,
            logical_positions=malformed.logical_positions,
            valid_mask=malformed.valid_mask,
            source_mask=malformed.source_mask,
            support_mask=malformed.support_mask,
            base_h4=malformed.base_h4,
            native_h4=malformed.native_h4,
        )


def test_provider_tensor_roundtrip_is_externally_bound_and_fail_closed(
    tmp_path,
) -> None:
    provider = _fit(weighted=True)
    output = tmp_path / "provider.pt"
    receipt = save_autonomous_complete_h4_residual_provider(provider, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    restored = load_autonomous_complete_h4_residual_provider(
        output,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_file_sha256=receipt["file_sha256"],
        expected_bridge_binding_sha256=_BRIDGE,
    )
    assert restored.metadata() == provider.metadata()
    for name in (
        "output_decoder",
        "lag_source_kernel",
        "state_kernel",
        "bias",
    ):
        torch.testing.assert_close(
            getattr(restored, name), getattr(provider, name), rtol=0, atol=0
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        save_autonomous_complete_h4_residual_provider(provider, output)
    with pytest.raises(ValueError, match="artifact differs"):
        load_autonomous_complete_h4_residual_provider(
            output,
            expected_artifact_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="file differs"):
        load_autonomous_complete_h4_residual_provider(
            output,
            expected_artifact_sha256=provider.artifact_sha256,
            expected_file_sha256="c" * 64,
        )

    tampered_state = autonomous_complete_h4_residual_provider_state_dict(provider)
    tampered_tensors = dict(tampered_state["tensors"])
    tampered_tensors["bias"] = tampered_tensors["bias"].clone()
    tampered_tensors["bias"][0] += 1.0
    tampered_state["tensors"] = tampered_tensors
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered_state, tampered_path)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_autonomous_complete_h4_residual_provider(
            tampered_path,
            expected_artifact_sha256=provider.artifact_sha256,
        )


def test_float32_runtime_correction_preserves_bridge_cast_once_midpoint() -> None:
    decoder = torch.zeros((1, _WIDTH), dtype=torch.float64)
    decoder[0, 0] = 1.0
    float32_ulp_at_one = float(
        torch.nextafter(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(2.0, dtype=torch.float32),
        )
        - torch.tensor(1.0, dtype=torch.float32)
    )
    delta = float32_ulp_at_one / 2.0 + 2.0e-16
    provider = AutonomousCompleteH4ResidualProvider(
        bridge_binding_sha256=_BRIDGE,
        output_decoder=decoder,
        lag_source_kernel=torch.zeros((1, _SOURCE_RANK, 1), dtype=torch.float64),
        state_kernel=torch.zeros((1, 1), dtype=torch.float64),
        bias=torch.tensor([delta], dtype=torch.float64),
        ridge=1.0,
        fit_objective="hidden_residual_ridge",
        fit_row_count=1,
        fit_family_ids=("family",),
        fit_sequence_sha256s=("d" * 64,),
        weighted_residual_rmse=0.0,
        fit_weight_sha256="e" * 64,
    )
    sequence = _sequence(90, length=2)
    realized = torch.zeros((1, 2, _WIDTH), dtype=torch.float32)
    realized[0, 0, 0] = 1.0
    correction = provider.correction(_prefix(sequence), realized)
    assert correction.dtype == torch.float64
    cast_once = (realized.to(torch.float64) + correction).to(torch.float32)
    prematurely_rounded = realized + correction.to(torch.float32)
    assert cast_once[0, 0, 0] == torch.nextafter(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(2.0, dtype=torch.float32),
    )
    assert prematurely_rounded[0, 0, 0] == 1.0
