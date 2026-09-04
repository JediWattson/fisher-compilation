from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from fisher_graph.fisher_compact_message_completion_artifacts import (
    BODY_DEPENDENCY_HASH_BINDINGS,
    artifact_dependencies,
    build_artifact_payload,
)
from fisher_graph.fisher_compact_message_completion_diagnostics import (
    build_full_rank_identity_smoke,
)
from fisher_graph.fisher_compact_message_completion_models import (
    freeze_epsilon_down_from_fit_train,
)
from fisher_graph.fisher_compact_message_completion_qualification import (
    build_o1_qualification,
    fit_selection_geometry,
    row_keys,
    subset_partition_by_units,
)
from fisher_graph.fisher_compact_message_completion_representation import (
    RepresentationQualificationError,
    analytic_ambient_operator,
    build_pre_tune_representation_qualification,
    directed_matching_caliper,
    identify_tensor,
)
from fisher_graph.fisher_compact_message_completion_science import (
    HierarchicalRowKey,
    hierarchical_row_weights,
    weighted_linear_quantile,
)
from fisher_graph.fisher_compact_message_completion_synthetic import (
    build_synthetic_campaign,
)


@pytest.fixture(scope="module")
def fitted_representation():
    campaign = build_synthetic_campaign()
    geometry = fit_selection_geometry(campaign)
    o1 = build_o1_qualification(campaign)
    full_rank = build_full_rank_identity_smoke(campaign, geometry)
    qualification = build_pre_tune_representation_qualification(
        campaign, geometry, o1, full_rank
    )
    return campaign, geometry, o1, full_rank, qualification


def _key(unit: int, sequence: int, position: int) -> HierarchicalRowKey:
    return HierarchicalRowKey(
        unit.to_bytes(32, "big"),
        sequence.to_bytes(32, "big"),
        position,
    )


def test_named_tensor_identity_binds_name_dtype_shape_and_bytes() -> None:
    values = np.arange(4, dtype=np.float32).reshape(2, 2)
    reference = identify_tensor("tensor", values)
    assert reference.dtype == "float32"
    assert reference.shape == (2, 2)
    assert reference.byte_count == 16
    assert reference.sha256 == identify_tensor("tensor", values.copy()).sha256
    assert reference.sha256 != identify_tensor("renamed", values).sha256
    assert reference.sha256 != identify_tensor("tensor", values.astype(np.float64)).sha256
    assert reference.sha256 != identify_tensor("tensor", values.reshape(4)).sha256
    changed = values.copy()
    changed[0, 0] = 10.0
    assert reference.sha256 != identify_tensor("tensor", changed).sha256


def test_directed_caliper_uses_target_normalized_pair_weights() -> None:
    coordinates = torch.tensor([[0.0], [1.0], [3.0], [10.0]], dtype=torch.float64)
    keys = (
        _key(1, 1, 0),
        _key(1, 1, 1),
        _key(2, 2, 0),
        _key(3, 3, 0),
    )
    row_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    result = directed_matching_caliper(coordinates, keys, row_weights)

    distances: list[float] = []
    pair_weights: list[float] = []
    tie_keys: list[bytes] = []
    for target, target_key in enumerate(keys):
        donors = [
            donor
            for donor, donor_key in enumerate(keys)
            if donor_key.analysis_unit_commitment
            != target_key.analysis_unit_commitment
            and donor_key.canonical_sequence_hash
            != target_key.canonical_sequence_hash
        ]
        for donor in donors:
            distances.append(float(abs(coordinates[target, 0] - coordinates[donor, 0])))
            pair_weights.append(float(row_weights[target]) / len(donors))
            tie_keys.append(
                b"fisher-cmc/matching-pair/v1\0"
                + target_key.order_bytes
                + keys[donor].order_bytes
            )
    expected = weighted_linear_quantile(
        distances, pair_weights, 0.25, tie_keys=tie_keys
    )
    assert result.status == "fitted"
    assert result.contributing_target_count == 4
    assert result.directed_pair_count == 10
    assert math.isclose(result.contributing_target_weight, 1.0)
    assert result.normalized_pair_weight_sum == 1.0
    assert result.caliper == expected


def test_pre_tune_qualification_is_complete_deterministic_and_fit_only(
    fitted_representation,
) -> None:
    campaign, geometry, o1, full_rank, qualification = fitted_representation
    repeated = build_pre_tune_representation_qualification(
        campaign, geometry, o1, full_rank
    )
    assert qualification.passed
    assert qualification.identity_sha256 == repeated.identity_sha256
    assert qualification.selected_locator_candidate_id == "K4"
    assert qualification.atlas_selected_k_values == (
        ("A_in", 4),
        ("A_mid", 4),
        ("A_out", 2),
    )
    assert [row.lower_k_fallback_used for row in qualification.locator_atlas_evidence] == [
        False,
        False,
        True,
    ]
    assert all(row["supported"] for row in qualification.locator_chart_support)
    assert len(qualification.matching_calipers.records) == 16
    assert qualification.matching_calipers.passed
    assert qualification.donor_boundaries.quantiles == (0.2, 0.4, 0.6, 0.8)
    assert list(qualification.donor_boundaries.values) == sorted(
        qualification.donor_boundaries.values
    )
    assert qualification.epsilon_down == freeze_epsilon_down_from_fit_train(campaign)

    changed_message = campaign.tune.tensors["message"].copy()
    changed_message += 1000.0
    changed_tensors = dict(campaign.tune.tensors)
    changed_tensors["message"] = changed_message
    changed_campaign = replace(
        campaign,
        tune=replace(campaign.tune, tensors=changed_tensors),
    )
    changed = build_pre_tune_representation_qualification(
        changed_campaign, geometry, o1, full_rank
    )
    assert changed.identity_sha256 == qualification.identity_sha256


def test_donor_boundaries_are_exact_fit_train_weighted_quantiles(
    fitted_representation,
) -> None:
    campaign, geometry, _, _, qualification = fitted_representation
    fit_train = subset_partition_by_units(
        campaign.fit, campaign.fit_split.right_units, partition_role="fit"
    )
    keys = row_keys(fit_train)
    weights = hierarchical_row_weights(keys)
    retained = geometry.message_sketch.transform(fit_train.tensors["message"])
    norms = torch.sqrt(retained.square().sum(dim=(1, 2)))
    expected = tuple(
        weighted_linear_quantile(
            norms,
            weights,
            q,
            tie_keys=tuple(key.order_bytes for key in keys),
        )
        for q in (0.2, 0.4, 0.6, 0.8)
    )
    assert qualification.donor_boundaries.values == expected
    assert qualification.donor_boundaries.fit_train_analysis_unit_count == 38


def test_codec_operators_use_full_message_projection_and_codec_specific_tau(
    fitted_representation,
) -> None:
    campaign, geometry, _, _, qualification = fitted_representation
    source_candidate = geometry.codec_candidates[0]
    q_codec = source_candidate.q_codecs[0]
    basis = geometry.message_sketch.representations[0].basis
    expected = (
        q_codec.decoder
        @ q_codec.encoder
        @ torch.as_tensor(campaign.source.output_weights[0], dtype=torch.float64)
        @ basis
        @ basis.T
    )
    observed = analytic_ambient_operator(
        q_codec, campaign.source.output_weights[0], basis
    )
    assert torch.equal(observed, expected)

    factor = max(
        10.0 * qualification.maximum_o1_relative_tolerance,
        100.0 * float(torch.finfo(torch.float32).eps),
    )
    for codec in qualification.codec_qualifications:
        assert codec.status == "eligible"
        assert len(codec.operator_cells) == 16
        assert len(codec.operator_pair_summaries) == 16
        assert codec.operator_aggregate is not None
        assert codec.operator_reference.status == "valid"
        assert codec.operator_reference.operator_reference_norm is not None
        assert codec.operator_reference.tau_r is not None
        assert math.isclose(
            codec.operator_reference.tau_r,
            factor * max(codec.operator_reference.operator_reference_norm, 1e-12),
            rel_tol=0.0,
            abs_tol=0.0,
        )


def test_per_codec_epsilon_values_and_hashes_are_literal(
    fitted_representation,
) -> None:
    _, geometry, _, _, qualification = fitted_representation
    for source, measured in zip(
        geometry.codec_candidates, qualification.codec_qualifications, strict=True
    ):
        assert measured.epsilon_completion_values == tuple(
            tuple(float(value) for value in row)
            for row in source.epsilon_completion_by_chart_head
        )
        assert measured.epsilon_j0_values == tuple(
            tuple(float(value) for value in row)
            for row in source.epsilon_j0_by_chart_head
        )
        assert measured.epsilon_completion_sha256 == identify_tensor(
            f"codec.{source.codec_id}.epsilon_completion",
            source.epsilon_completion_by_chart_head,
        ).sha256
        assert measured.epsilon_j0_sha256 == identify_tensor(
            f"codec.{source.codec_id}.epsilon_J0",
            source.epsilon_j0_by_chart_head,
        ).sha256


def test_zero_positive_operator_reference_is_invalid_without_manufactured_tau(
    fitted_representation,
) -> None:
    campaign, geometry, o1, full_rank, _ = fitted_representation
    zero_source = replace(
        campaign.source,
        output_weights=np.zeros_like(campaign.source.output_weights),
    )
    result = build_pre_tune_representation_qualification(
        replace(campaign, source=zero_source), geometry, o1, full_rank
    )
    assert not result.passed
    for codec in result.codec_qualifications:
        assert codec.status == "invalid_no_positive_operator_reference"
        assert codec.operator_reference.operator_reference_norm is None
        assert codec.operator_reference.tau_r is None
        assert codec.operator_aggregate is not None
        assert codec.operator_aggregate.zero_ambient_pair_count == 16


def test_all_zero_operator_references_request_preregistered_lower_k_retry(
    fitted_representation,
) -> None:
    campaign, geometry, o1, full_rank, _ = fitted_representation
    zero_source = replace(
        campaign.source,
        output_weights=np.zeros_like(campaign.source.output_weights),
    )
    result = build_pre_tune_representation_qualification(
        replace(campaign, source=zero_source), geometry, o1, full_rank
    )

    assert result.selected_locator_k == 4
    assert result.requires_lower_k_operator_retry

    lower_k_geometry = fit_selection_geometry(
        campaign,
        attempted_locator_k=(2,),
    )
    assert lower_k_geometry.locator.chart_count == 2
    assert lower_k_geometry.mid_locator.chart_count == 2
    assert lower_k_geometry.out_locator.chart_count == 2

    lower_k_full_rank = build_full_rank_identity_smoke(campaign, lower_k_geometry)
    lower_k_result = build_pre_tune_representation_qualification(
        replace(campaign, source=zero_source),
        lower_k_geometry,
        o1,
        lower_k_full_rank,
    )
    assert not lower_k_result.requires_lower_k_operator_retry
    assert lower_k_result.operator_reference_fallback_exhausted


def test_measured_p4_bodies_satisfy_current_strict_schemas(
    fitted_representation,
) -> None:
    _, _, _, _, qualification = fitted_representation
    bodies = qualification.artifact_bodies(
        locator_protocol_sha256="ab" * 32,
        codec_protocol_sha256="cd" * 32,
        carrier_identity_sha256="ef" * 32,
    )
    assert set(bodies) == {
        "locator_protocol.json",
        "locator_candidates.json",
        "codec_protocol.json",
        "codec_candidate_manifests.json",
        "message_sketch_manifest_p32.json",
        "representation_numeric_constants.json",
        "full_rank_identity_smoke.json",
        "selection_representation_manifest.json",
    }
    for name, body in bodies.items():
        dependency_hashes = {
            dependency: "11" * 32 for dependency in artifact_dependencies(name)
        }
        for body_field, dependency_name in BODY_DEPENDENCY_HASH_BINDINGS.get(
            name, {}
        ).items():
            dependency_hashes[dependency_name] = body[body_field]
        build_artifact_payload(
            name,
            campaign_id="synthetic-representation-qualification",
            dependency_hashes=dependency_hashes,
            body=body,
        )


def test_failed_o1_cannot_form_a_pre_tune_representation(
    fitted_representation,
) -> None:
    campaign, geometry, o1, full_rank, _ = fitted_representation
    failed_tolerance = replace(o1.tolerances[0], calibration_passed=False)
    failed_o1 = replace(o1, tolerances=(failed_tolerance,) + o1.tolerances[1:])
    with pytest.raises(RepresentationQualificationError, match="O1 must pass"):
        build_pre_tune_representation_qualification(
            campaign, geometry, failed_o1, full_rank
        )
