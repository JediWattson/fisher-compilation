from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json

import pytest

import fisher_graph.gemma3_l3_l4_synthetic_reference_protocol as protocol_module
from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256,
    DEFAULT_PROTOCOL_SHA256,
    SyntheticReferenceProbe,
    SyntheticReferenceProtocol,
    assessment_panel_spec_sha256,
    default_synthetic_reference_protocol,
)


def test_default_protocol_is_exact_prompt_blind_and_hash_authenticated() -> None:
    protocol = default_synthetic_reference_protocol()

    assert protocol.protocol_sha256 == DEFAULT_PROTOCOL_SHA256
    assert DEFAULT_PROTOCOL_SHA256 == (
        "82b6d07830c3410a89f24233fc0d2ddfb0f3c1972739b6fe55144183485b3fb3"
    )
    assert [
        (role.name, role.seed, role.expected_count, role.families)
        for role in protocol.roles
    ] == [
        (
            "fit",
            20_260_728_031,
            80,
            ("rademacher", "ar1", "axis"),
        ),
        (
            "selection",
            20_260_728_071,
            32,
            ("rademacher", "ar1"),
        ),
        (
            "assessment",
            20_260_728_059,
            88,
            (
                "sparse",
                "chirp",
                "axis",
                "radial_collision",
                "null_collision",
            ),
        ),
    ]
    counts = Counter((probe.role, probe.family) for probe in protocol.probes)
    assert counts == {
        ("fit", "rademacher"): 32,
        ("fit", "ar1"): 32,
        ("fit", "axis"): 16,
        ("selection", "rademacher"): 16,
        ("selection", "ar1"): 16,
        ("assessment", "sparse"): 24,
        ("assessment", "chirp"): 24,
        ("assessment", "axis"): 16,
        ("assessment", "radial_collision"): 12,
        ("assessment", "null_collision"): 12,
    }
    assert len(protocol.probes) == 200

    state = protocol.state_dict()
    assert state["schema"].endswith(".v2")
    assert state["format_version"] == 2
    assert (
        state["assessment_panel_spec_sha256"]
        == DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256
    )
    status = state["scientific_status"]
    assert status == {
        "scope": "synthetic_residual_state_probe_specifications_only",
        "prompt_blind_after_frozen_basis_package": True,
        "frozen_basis_package_is_upstream_prompt_conditioned": True,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "score_gradient_rows_loaded": False,
        "prompt_local_kernel_loaded": False,
        "model_called": False,
        "artifact_loaded": False,
        "live_tensor_materialization_performed": False,
        "natural_prompt_fidelity_claim": False,
        "model_replacement_claim": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }
    assert "torch" not in protocol_module.__dict__
    assert "transformers" not in protocol_module.__dict__


def test_v2_selection_is_fresh_while_assessment_identity_is_preserved() -> None:
    protocol = default_synthetic_reference_protocol()
    fit = tuple(probe for probe in protocol.probes if probe.role == "fit")
    selection = tuple(
        probe for probe in protocol.probes if probe.role == "selection"
    )
    assessment = tuple(
        probe for probe in protocol.probes if probe.role == "assessment"
    )
    assessment_hashes = tuple(
        probe.artifact_sha256 for probe in assessment
    )

    # These are the exact v1 endpoints and ordered-panel digest.  Holding the
    # aggregate fixed proves all 88 ordered assessment artifact identities
    # survived the protocol-v2 selection reset.
    assert assessment[0].artifact_sha256 == (
        "4d25679b481b931c2c00163f1ecea1c60d1648cae853832acc4a0320401cb609"
    )
    assert assessment[-1].artifact_sha256 == (
        "3a2a76d50b378c44324cf5d51270a9e1a9a4459b402cf0b95e906acbef26c778"
    )
    assert assessment_panel_spec_sha256(assessment_hashes) == (
        DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256
    )
    assert DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256 == (
        "c690e9f85f5629ab2701fc5db487aea1404864256f5fe24034e35143047af102"
    )
    assert protocol.assessment_panel_spec_sha256 == (
        DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256
    )

    # Fit specifications also remain in their frozen v1 probe domain.
    assert fit[0].artifact_sha256 == (
        "4c5a2d8cabe6f385de6eaf425090f969f45fec0739dc6bbd608962545842540f"
    )
    assert fit[-1].artifact_sha256 == (
        "ee177b37a04911dc09f06de15e5a658747f3640f207fcc8f0a780daf283e3239"
    )

    assert selection[0].artifact_sha256 == (
        "4757b19f63eaba45e11ff9ee3c59b9c9bfabfa7c25ddcdffeae4233225f1a24e"
    )
    assert selection[-1].artifact_sha256 == (
        "38282feb6179facc00318d19f74579f0a64f740febff45ec78b3dd1347adab05"
    )
    assert selection[0].direction_seed == 8_740_534_197_596_504_437
    assert selection[-1].direction_seed == 1_439_586_942_421_924_963
    assert selection[0].artifact_sha256 != (
        "3ad0976af946e4837e9577c34ceeb4cf1c2b85bce0a0cfa2a2e977450a87600c"
    )


def test_v1_selection_and_protocol_envelopes_cannot_authenticate_as_v2() -> None:
    protocol = default_synthetic_reference_protocol()
    selection = next(
        probe for probe in protocol.probes if probe.role == "selection"
    )
    old_probe = selection.state_dict()
    old_probe["direction_seed"] = 3_782_939_586_030_697_990
    old_probe["artifact_sha256"] = (
        "3ad0976af946e4837e9577c34ceeb4cf1c2b85bce0a0cfa2a2e977450a87600c"
    )
    old_probe["probe_id"] = (
        "selection.0000.rademacher.3ad0976af946"
    )
    with pytest.raises(ValueError, match="probe hash mismatch"):
        SyntheticReferenceProbe.from_state_dict(old_probe)

    old_protocol = protocol.state_dict()
    old_protocol["schema"] = (
        "fisher_graph.gemma3_l3_l4_synthetic_reference_protocol"
    )
    old_protocol["format_version"] = 1
    old_protocol["protocol_sha256"] = (
        "43aa289a0fc51713ed3d0ec2cbc91ae45a91cadefee774b1a203dc0d3b6515c1"
    )
    with pytest.raises(ValueError, match="declaration drifted"):
        SyntheticReferenceProtocol.from_state_dict(old_protocol)


def test_assessment_panel_spec_hash_is_ordered_and_strict() -> None:
    protocol = default_synthetic_reference_protocol()
    values = tuple(
        probe.artifact_sha256
        for probe in protocol.probes
        if probe.role == "assessment"
    )

    assert assessment_panel_spec_sha256(tuple(reversed(values))) != (
        DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256
    )
    with pytest.raises(ValueError, match="88 unique"):
        assessment_panel_spec_sha256(values[:-1])
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        assessment_panel_spec_sha256((*values[:-1], "not-a-digest"))


def test_generation_is_deterministic_json_safe_and_strict_round_trip() -> None:
    first = default_synthetic_reference_protocol()
    second = default_synthetic_reference_protocol()

    assert first == second
    assert first.state_dict() == second.state_dict()
    serialized = json.dumps(
        first.state_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert serialized
    restored = SyntheticReferenceProtocol.from_state_dict(first.state_dict())
    assert restored == first
    assert restored.protocol_sha256 == DEFAULT_PROTOCOL_SHA256


def test_role_directions_are_seed_disjoint_and_cover_frozen_geometry() -> None:
    protocol = default_synthetic_reference_protocol()
    stochastic_families = {"rademacher", "ar1", "sparse", "chirp"}
    seeds_by_role = {
        role: {
            probe.direction_seed
            for probe in protocol.probes
            if probe.role == role and probe.family in stochastic_families
        }
        for role in ("fit", "selection", "assessment")
    }
    assert len(seeds_by_role["fit"]) == 64
    assert len(seeds_by_role["selection"]) == 32
    assert len(seeds_by_role["assessment"]) == 48
    assert seeds_by_role["fit"].isdisjoint(seeds_by_role["selection"])
    assert seeds_by_role["fit"].isdisjoint(seeds_by_role["assessment"])
    assert seeds_by_role["selection"].isdisjoint(
        seeds_by_role["assessment"]
    )

    for role in ("fit", "selection", "assessment"):
        rows = [probe for probe in protocol.probes if probe.role == role]
        assert {probe.sequence_length for probe in rows} == {
            32,
            72,
            128,
            256,
        }
        assert {probe.modal_amplitude for probe in rows} == {
            0.05,
            0.25,
            0.5,
            1.0,
        }
        assert {probe.radial_scale for probe in rows} == {0.5, 1.0, 2.0}
        assert {probe.null_coordinate for probe in rows} == {
            -1.0,
            0.0,
            1.0,
        }
        assert all(
            0 <= probe.source_offset < probe.sequence_length for probe in rows
        )
        assert any(probe.source_offset == 0 for probe in rows)
        assert any(
            probe.source_offset >= probe.sequence_length // 2
            for probe in rows
        )


def test_fit_selection_and_assessment_have_separate_authority() -> None:
    protocol = default_synthetic_reference_protocol()
    fit, selection, assessment = protocol.roles

    assert fit.may_fit_coefficients
    assert not fit.may_select_candidate
    assert not fit.may_assess_sealed_candidate
    assert not fit.requires_frozen_candidate

    assert not selection.may_fit_coefficients
    assert selection.may_select_candidate
    assert not selection.may_assess_sealed_candidate
    assert not selection.requires_frozen_candidate

    assert not assessment.may_fit_coefficients
    assert not assessment.may_select_candidate
    assert assessment.may_assess_sealed_candidate
    assert assessment.requires_frozen_candidate

    selection_state = protocol.state_dict()["candidate_selection"]
    assert selection_state["fit_all_rows_before_opening_selection"] is True
    assert selection_state["assessment_may_change_candidate"] is False
    assert selection_state["gate_values_applied_without_rounding"] is True


def test_assessment_uses_sparse_chirp_and_identifiability_controls() -> None:
    protocol = default_synthetic_reference_protocol()
    assessment = [
        probe for probe in protocol.probes if probe.role == "assessment"
    ]

    sparse = [probe for probe in assessment if probe.family == "sparse"]
    assert len(sparse) == 24
    assert all(len(probe.sparse_coordinates) == 4 for probe in sparse)
    assert all(
        len(set(probe.sparse_coordinates)) == 4 for probe in sparse
    )
    chirps = [probe for probe in assessment if probe.family == "chirp"]
    assert len(chirps) == 24
    assert all(probe.chirp_temporal_frequency is not None for probe in chirps)
    assert all(probe.chirp_modal_frequency is not None for probe in chirps)
    assert {probe.chirp_phase_quadrant for probe in chirps} == {0, 1, 2, 3}

    groups = defaultdict(list)
    for probe in assessment:
        if probe.collision_group is not None:
            groups[probe.collision_group].append(probe)

    axis_groups = {
        name: rows
        for name, rows in groups.items()
        if name.startswith("assessment.axis.")
    }
    assert len(axis_groups) == 8
    for rows in axis_groups.values():
        assert len(rows) == 2
        assert {probe.axis_sign for probe in rows} == {-1, 1}
        assert {probe.collision_variant for probe in rows} == {
            "negative",
            "positive",
        }

    radial_groups = {
        name: rows
        for name, rows in groups.items()
        if name.startswith("assessment.radial.")
    }
    assert len(radial_groups) == 4
    for rows in radial_groups.values():
        assert len(rows) == 3
        assert {probe.radial_scale for probe in rows} == {0.5, 1.0, 2.0}
        invariant = {
            (
                probe.sequence_length,
                probe.source_offset,
                probe.modal_amplitude,
                probe.null_coordinate,
                probe.direction_seed,
                probe.axis_mode,
                probe.axis_sign,
            )
            for probe in rows
        }
        assert len(invariant) == 1

    null_groups = {
        name: rows
        for name, rows in groups.items()
        if name.startswith("assessment.null.")
    }
    assert len(null_groups) == 4
    for rows in null_groups.values():
        assert len(rows) == 3
        assert {probe.null_coordinate for probe in rows} == {-1.0, 0.0, 1.0}
        invariant = {
            (
                probe.sequence_length,
                probe.source_offset,
                probe.modal_amplitude,
                probe.radial_scale,
                probe.direction_seed,
                probe.axis_mode,
                probe.axis_sign,
            )
            for probe in rows
        }
        assert len(invariant) == 1


def test_candidate_ladder_and_gates_are_frozen() -> None:
    protocol = default_synthetic_reference_protocol()

    assert [
        (row.kind, row.source_rank, row.target_rank)
        for row in protocol.candidate_ladder
    ] == [
        ("constant", 0, 0),
        ("spectral", 8, 8),
        ("spectral", 16, 16),
        ("spectral", 24, 24),
        ("spectral", 32, 32),
        ("spectral", 48, 48),
        ("dense", 64, 64),
    ]
    assert protocol.gates.state_dict() == {
        "maximum_fisher_weighted_relative_error": 0.225,
        "minimum_reference_cosine": 0.975,
        "minimum_error_reduction_vs_constant": 0.10,
        "minimum_error_reduction_vs_position_only": 0.10,
        "maximum_per_probe_p90_relative_error": 0.35,
        "maximum_worst_panel_relative_error": 0.35,
        "maximum_prepared_vs_analytic_relative_error": 1e-5,
        "maximum_causality_violation": 1e-6,
        "maximum_padding_violation": 1e-6,
        "maximum_repeat_relative_error": 1e-7,
        "minimum_collision_target_relative_difference": 0.01,
        "minimum_in_support_fraction": 0.99,
    }


def test_tampered_probe_or_protocol_hash_fails_closed() -> None:
    state = default_synthetic_reference_protocol().state_dict()
    tampered_probe = copy.deepcopy(state)
    tampered_probe["probes"][0]["modal_amplitude"] = 0.25
    with pytest.raises(ValueError, match="probe hash mismatch"):
        SyntheticReferenceProtocol.from_state_dict(tampered_probe)

    tampered_hash = copy.deepcopy(state)
    tampered_hash["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        SyntheticReferenceProtocol.from_state_dict(tampered_hash)


def test_tampered_role_authority_fails_before_materialization() -> None:
    state = default_synthetic_reference_protocol().state_dict()
    tampered = copy.deepcopy(state)
    tampered["roles"][2]["may_select_candidate"] = True

    with pytest.raises(
        ValueError,
        match="exactly one authority",
    ):
        SyntheticReferenceProtocol.from_state_dict(tampered)
