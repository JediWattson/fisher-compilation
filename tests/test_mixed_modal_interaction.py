from __future__ import annotations

import copy
import math

import pytest
import torch

from fisher_graph.mixed_modal_interaction import (
    MixedModalInteractionArtifact,
    WalshComponents,
    analyze_mixed_modal_interaction,
    build_mixed_modal_interaction_artifact,
    interaction_parity_components,
    pairwise_all_nonadditivity,
    walsh_components,
    walsh_reconstruct,
)


_ORIGIN_BINDING = "ab" * 32
_CANDIDATE_BINDING = "cd" * 32
_BASELINE_BINDING = "ef" * 32


def _artifact() -> MixedModalInteractionArtifact:
    generator = torch.Generator().manual_seed(23)
    shape = (3, 2, 2, 4)
    c00 = torch.randn(*shape, generator=generator, dtype=torch.float64)
    c10 = torch.randn(*shape, generator=generator, dtype=torch.float64)
    c01 = torch.randn(*shape, generator=generator, dtype=torch.float64)
    c11 = torch.randn(*shape, generator=generator, dtype=torch.float64)
    responses = walsh_reconstruct(
        WalshComponents(c00=c00, c10=c10, c01=c01, c11=c11)
    )
    predictions = walsh_reconstruct(
        WalshComponents(
            c00=c00,
            c10=c10,
            c01=c01,
            c11=torch.zeros_like(c11),
        )
    )
    singleton_mode_indices = (1, 2, 3, 7, 4, 12)
    singleton_rows = []
    for pair in range(3):
        singleton_rows.extend(
            (
                torch.stack((c10[pair], -c10[pair]), dim=1),
                torch.stack((c01[pair], -c01[pair]), dim=1),
            )
        )
    singleton_responses = torch.stack(singleton_rows)
    return build_mixed_modal_interaction_artifact(
        responses.to(dtype=torch.float32),
        predictions.to(dtype=torch.float32),
        pair_labels=("near", "middle", "far"),
        pair_indices=((1, 2), (3, 7), (4, 12)),
        pair_families=("local", "local", "separated"),
        pair_amplitudes=torch.tensor(
            [[1.0, 1.5], [0.75, 0.5], [2.0, 1.0]],
            dtype=torch.float32,
        ),
        scales=torch.tensor([0.5, 1.0], dtype=torch.float32),
        origin=28,
        origin_binding_sha256=_ORIGIN_BINDING,
        candidate_binding_sha256=_CANDIDATE_BINDING,
        shared_baseline_sha256=_BASELINE_BINDING,
        singleton_responses=singleton_responses.to(dtype=torch.float32),
        singleton_mode_indices=singleton_mode_indices,
    )


def test_walsh_decomposition_is_exact_parseval_and_exposes_cross_term() -> None:
    artifact = _artifact()
    components = walsh_components(artifact.responses)
    torch.testing.assert_close(
        walsh_reconstruct(components),
        artifact.responses,
        atol=1e-12,
        rtol=1e-12,
    )

    analysis = analyze_mixed_modal_interaction(artifact)
    assert analysis.response_energy.parseval_relative_error < 1e-12
    assert analysis.prediction_energy.parseval_relative_error < 1e-12
    assert abs(
        sum(
            (
                analysis.response_energy.c00_energy_fraction,
                analysis.response_energy.c10_energy_fraction,
                analysis.response_energy.c01_energy_fraction,
                analysis.response_energy.c11_energy_fraction,
            )
        )
        - 1.0
    ) < 1e-12
    assert analysis.c11_cross_energy_fraction > 0.0
    assert analysis.global_c11_candidate_metric.relative_error == pytest.approx(
        1.0
    )
    assert len(analysis.per_scale_candidate_metrics) == 2
    assert tuple(
        metric.family for metric in analysis.per_family_candidate_metrics
    ) == ("local", "separated")
    assert analysis.per_family_candidate_metrics[0].pair_count == 2
    assert len(analysis.per_scale_family_candidate_metrics) == 4
    assert analysis.global_nonadditivity_energy is not None
    assert (
        analysis.global_nonadditivity_energy
        .all_nonadditivity_energy_fraction
        > 0.0
    )
    assert len(analysis.per_scale_nonadditivity_energy) == 2
    assert len(analysis.per_family_nonadditivity_energy) == 2
    assert len(analysis.per_pair_nonadditivity_energy) == 3
    assert len(analysis.per_scale_family_nonadditivity_energy) == 4
    assert len(analysis.per_scale_pair_nonadditivity_energy) == 6
    assert analysis.global_oracle_correction is not None
    assert analysis.global_oracle_correction.truth_leaking_oracle_only
    assert analysis.global_oracle_correction.replacement_authority is False
    oracle = analysis.global_oracle_correction
    assert oracle.signed_residual_error_gain == pytest.approx(
        1.0
        - oracle.oracle_corrected.residual_frobenius
        / oracle.base.residual_frobenius
    )
    assert len(analysis.per_scale_oracle_corrections) == 2
    assert len(analysis.per_family_oracle_corrections) == 2
    assert len(analysis.per_scale_family_oracle_corrections) == 4
    assert analysis.interaction_parity_energy is not None
    assert (
        analysis.interaction_parity_energy.parseval_relative_error
        < 1e-12
    )
    assert len(analysis.per_scale_interaction_parity_energy) == 2
    assert analysis.reconstruction_checks is not None
    assert analysis.reconstruction_checks.metadata()["all_checks_passed"]
    assert tuple(
        value.family
        for value in analysis.per_family_nonadditivity_energy
    ) == ("local", "separated")
    nonadditivity = pairwise_all_nonadditivity(artifact)
    expected_e_add = float(torch.linalg.vector_norm(nonadditivity)) / float(
        torch.linalg.vector_norm(artifact.responses)
    )
    assert analysis.global_nonadditivity_energy.e_add == pytest.approx(
        expected_e_add
    )
    expected = artifact.responses.clone()
    singleton_lookup = {
        mode: index
        for index, mode in enumerate(artifact.singleton_mode_indices)
    }
    signs = ((0, 0), (0, 1), (1, 0), (1, 1))
    assert artifact.singleton_responses is not None
    for pair, (left, right) in enumerate(artifact.pair_indices):
        for sign, (left_sign, right_sign) in enumerate(signs):
            expected[pair, :, sign] -= artifact.singleton_responses[
                singleton_lookup[left], :, left_sign
            ]
            expected[pair, :, sign] -= artifact.singleton_responses[
                singleton_lookup[right], :, right_sign
            ]
    torch.testing.assert_close(nonadditivity, expected)
    torch.testing.assert_close(
        walsh_reconstruct(interaction_parity_components(artifact)),
        nonadditivity,
        atol=1e-12,
        rtol=1e-12,
    )
    assert analysis.causal_claim is False
    assert analysis.semantic_claim is False


def test_artifact_roundtrip_is_canonical_strict_and_tamper_evident() -> None:
    artifact = _artifact()
    assert artifact.responses.device.type == "cpu"
    assert artifact.responses.dtype == torch.float64
    assert artifact.responses.is_contiguous()
    assert artifact.singleton_responses is not None
    assert artifact.singleton_responses.dtype == torch.float64
    assert artifact.singleton_sign_order == ("+", "-")
    assert artifact.shared_baseline_sha256 == _BASELINE_BINDING
    assert artifact.sign_order == ("++", "+-", "-+", "--")
    assert artifact.metadata()["finite_probe_only"] is True

    state = artifact.state_dict()
    restored = MixedModalInteractionArtifact.from_state_dict(state)
    assert restored.artifact_sha256 == artifact.artifact_sha256
    assert restored.metadata() == artifact.metadata()
    torch.testing.assert_close(restored.responses, artifact.responses)
    assert restored.analyze().metadata() == artifact.analyze().metadata()

    tensor_tamper = copy.deepcopy(state)
    tensor_tamper["responses"][0, 0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        MixedModalInteractionArtifact.from_state_dict(tensor_tamper)

    singleton_tamper = copy.deepcopy(state)
    singleton_tamper["singleton_responses"][0, 0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        MixedModalInteractionArtifact.from_state_dict(singleton_tamper)

    metadata_tamper = copy.deepcopy(state)
    metadata_tamper["pair_families"] = (
        "changed",
        "local",
        "separated",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        MixedModalInteractionArtifact.from_state_dict(metadata_tamper)

    derived_tamper = copy.deepcopy(state)
    derived_tamper["lag_count"] = 99
    with pytest.raises(ValueError, match="derived field"):
        MixedModalInteractionArtifact.from_state_dict(derived_tamper)

    unknown_field = copy.deepcopy(state)
    unknown_field["extra"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        MixedModalInteractionArtifact.from_state_dict(unknown_field)

    artifact.responses[0, 0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        artifact.validate_integrity()


def test_geometry_order_and_provenance_are_rejected_fail_closed() -> None:
    artifact = _artifact()
    state = artifact.state_dict()

    bad_order = copy.deepcopy(state)
    bad_order["sign_order"] = ("++", "-+", "+-", "--")
    with pytest.raises(ValueError, match="sign_order"):
        MixedModalInteractionArtifact.from_state_dict(bad_order)

    bad_singleton_order = copy.deepcopy(state)
    bad_singleton_order["singleton_sign_order"] = ("-", "+")
    with pytest.raises(ValueError, match="singleton_sign_order"):
        MixedModalInteractionArtifact.from_state_dict(bad_singleton_order)

    missing_mode = copy.deepcopy(state)
    missing_mode["singleton_mode_indices"] = (1, 2, 3, 7, 4, 99)
    with pytest.raises(ValueError, match="every paired mode"):
        MixedModalInteractionArtifact.from_state_dict(missing_mode)

    reversed_pair = copy.deepcopy(state)
    reversed_pair["pair_indices"] = ((2, 1), (3, 7), (4, 12))
    with pytest.raises(ValueError, match="left < right"):
        MixedModalInteractionArtifact.from_state_dict(reversed_pair)

    with pytest.raises(ValueError, match="shape"):
        build_mixed_modal_interaction_artifact(
            artifact.responses,
            artifact.candidate_predictions[:, :, :, :, :-1],
            pair_labels=artifact.pair_labels,
            pair_indices=artifact.pair_indices,
            pair_families=artifact.pair_families,
            pair_amplitudes=artifact.pair_amplitudes,
            scales=artifact.scales,
            origin=artifact.origin,
            origin_binding_sha256=_ORIGIN_BINDING,
            candidate_binding_sha256=_CANDIDATE_BINDING,
            shared_baseline_sha256=_BASELINE_BINDING,
        )


def test_singleton_controls_are_optional_but_interaction_metrics_require_them(
) -> None:
    source = _artifact()
    artifact = build_mixed_modal_interaction_artifact(
        source.responses,
        source.candidate_predictions,
        pair_labels=source.pair_labels,
        pair_indices=source.pair_indices,
        pair_families=source.pair_families,
        pair_amplitudes=source.pair_amplitudes,
        scales=source.scales,
        origin=source.origin,
        origin_binding_sha256=_ORIGIN_BINDING,
        candidate_binding_sha256=_CANDIDATE_BINDING,
        shared_baseline_sha256=_BASELINE_BINDING,
    )

    assert artifact.has_singleton_controls is False
    analysis = artifact.analyze()
    assert analysis.global_nonadditivity_energy is None
    assert analysis.global_oracle_correction is None
    assert analysis.reconstruction_checks is None
    restored = MixedModalInteractionArtifact.from_state_dict(
        artifact.state_dict()
    )
    assert restored.artifact_sha256 == artifact.artifact_sha256
    with pytest.raises(ValueError, match="singleton controls"):
        pairwise_all_nonadditivity(artifact)

    with pytest.raises(ValueError, match="origin_binding_sha256"):
        build_mixed_modal_interaction_artifact(
            artifact.responses,
            artifact.candidate_predictions,
            pair_labels=artifact.pair_labels,
            pair_indices=artifact.pair_indices,
            pair_families=artifact.pair_families,
            pair_amplitudes=artifact.pair_amplitudes,
            scales=artifact.scales,
            origin=artifact.origin,
            origin_binding_sha256="not-a-digest",
            candidate_binding_sha256=_CANDIDATE_BINDING,
            shared_baseline_sha256=_BASELINE_BINDING,
        )


def test_shared_modes_even_singletons_and_known_interaction_are_exact() -> None:
    generator = torch.Generator().manual_seed(101)
    mode_c0 = torch.randn(
        3,
        2,
        2,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    mode_c1 = torch.randn(
        3,
        2,
        2,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    singleton_responses = torch.stack(
        (mode_c0 + mode_c1, mode_c0 - mode_c1),
        dim=2,
    )
    pair_indices = ((0, 1), (0, 2), (1, 2))
    interaction = WalshComponents(
        c00=torch.randn(
            3, 2, 2, 3, generator=generator, dtype=torch.float64
        ),
        c10=torch.randn(
            3, 2, 2, 3, generator=generator, dtype=torch.float64
        ),
        c01=torch.randn(
            3, 2, 2, 3, generator=generator, dtype=torch.float64
        ),
        c11=torch.randn(
            3, 2, 2, 3, generator=generator, dtype=torch.float64
        ),
    )
    additive_c00 = []
    additive_c10 = []
    additive_c01 = []
    for left, right in pair_indices:
        additive_c00.append(mode_c0[left] + mode_c0[right])
        additive_c10.append(mode_c1[left])
        additive_c01.append(mode_c1[right])
    additive = walsh_reconstruct(
        WalshComponents(
            c00=torch.stack(additive_c00),
            c10=torch.stack(additive_c10),
            c01=torch.stack(additive_c01),
            c11=torch.zeros_like(interaction.c11),
        )
    )
    interaction_responses = walsh_reconstruct(interaction)
    responses = additive + interaction_responses
    artifact = build_mixed_modal_interaction_artifact(
        responses,
        additive,
        pair_labels=("01", "02", "12"),
        pair_indices=pair_indices,
        pair_families=("shared-left", "shared-left", "shared-right"),
        pair_amplitudes=torch.ones(3, 2, dtype=torch.float64),
        scales=torch.tensor([0.5, 1.0], dtype=torch.float64),
        origin=28,
        origin_binding_sha256=_ORIGIN_BINDING,
        candidate_binding_sha256=_CANDIDATE_BINDING,
        shared_baseline_sha256=_BASELINE_BINDING,
        singleton_responses=singleton_responses,
        singleton_mode_indices=(0, 1, 2),
    )

    measured = interaction_parity_components(artifact)
    for field in ("c00", "c10", "c01", "c11"):
        torch.testing.assert_close(
            getattr(measured, field),
            getattr(interaction, field),
            atol=1e-12,
            rtol=1e-12,
        )
    analysis = artifact.analyze()
    assert analysis.global_nonadditivity_energy is not None
    manual_e_add = math.sqrt(
        float(interaction_responses.square().sum())
        / float(responses.square().sum())
    )
    assert analysis.global_nonadditivity_energy.e_add == pytest.approx(
        manual_e_add
    )
    assert analysis.global_oracle_correction is not None
    oracle = analysis.global_oracle_correction
    assert oracle.base.residual_frobenius == pytest.approx(
        float(torch.linalg.vector_norm(interaction_responses))
    )
    assert oracle.oracle_corrected.residual_frobenius < 1e-12
    assert oracle.signed_residual_error_gain == pytest.approx(1.0)
