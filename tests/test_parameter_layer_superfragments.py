from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import pytest
import torch

from fisher_graph.fisher_prompt_clustering import (
    FisherPromptClusterConfig,
    build_fisher_prompt_clusters,
)
from fisher_graph.parameter_cluster_fragments import (
    ParameterClusterLayerFragmentPlan,
    build_parameter_cluster_layer_fragments,
)
from fisher_graph.parameter_fisher_coupling import (
    NaturalMLPLayerParameterSpec,
    build_grouped_virtual_gate_fisher,
    build_natural_mlp_parameter_group_catalog,
)
from fisher_graph.parameter_layer_superfragments import (
    ParameterLayerSuperfragmentPlan,
    build_parameter_layer_superfragments,
)
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fragment_plan() -> ParameterClusterLayerFragmentPlan:
    layers = tuple(
        NaturalMLPLayerParameterSpec.from_cross_block_layer_spec(
            CrossBlockLayerSpec(
                layer_id=f"model.layers.{ordinal}",
                layer_ordinal=ordinal,
                activation_site=f"model.layers.{ordinal}.mlp.gated",
                width=3,
            ),
            input_width=4,
            output_width=4,
            parameter_prefix=f"model.layers.{ordinal}.mlp",
        )
        for ordinal in range(2)
    )
    catalog = build_natural_mlp_parameter_group_catalog(
        model_fingerprint=_digest("model"),
        layer_specs=layers,
    )
    first = torch.tensor((1.0, 0.2, -0.4, 0.7), dtype=torch.float64)
    second = torch.tensor((-0.2, 0.8, 0.5, -0.3), dtype=torch.float64)
    scores = torch.stack(
        (
            first,
            second,
            -first,
            0.5 * first,
            -0.7 * second,
            1.2 * first,
        ),
        dim=1,
    )
    fisher = build_grouped_virtual_gate_fisher(
        scores,
        catalog=catalog,
        calibration_split_sha256=_digest("fit"),
        objective_sha256=_digest("nll"),
        normalization="sum_over_prompts",
    )
    config = FisherPromptClusterConfig(
        model_fingerprint=catalog.model_fingerprint,
        calibration_split_sha256=fisher.calibration_split_sha256,
        objective_sha256=fisher.objective_sha256,
        source_fisher_coupling_sha256=fisher.artifact_sha256,
        layer_specs=tuple(
            value.cross_block_layer_spec for value in catalog.layer_specs
        ),
        mode_catalog=fisher.fisher_ranked_mode_catalog(),
        cluster_count=2,
        max_iterations=40,
        tolerance=1e-13,
        mode_chunk_size=2,
    )
    clusters = build_fisher_prompt_clusters(scores, config)
    return build_parameter_cluster_layer_fragments(clusters, fisher)


def test_fragments_are_aggregated_exactly_once_per_native_layer() -> None:
    fragments = _fragment_plan()
    plan = build_parameter_layer_superfragments(fragments)

    assert plan.source_fragment_plan_sha256 == fragments.artifact_sha256
    assert plan.source_fragment_count == fragments.fragment_count == 4
    assert plan.layer_count == plan.superfragment_count == 2
    assert plan.source_group_count == plan.assigned_group_count == 6
    assert plan.assigned_native_parameter_count == 6 * (4 + 4 + 4)
    assert tuple(
        value.layer_ordinal for value in plan.superfragments
    ) == (0, 1)
    assert tuple(
        group
        for value in plan.superfragments
        for group in value.group_indices
    ) == tuple(range(6))

    authenticated_members: list[str] = []
    for superfragment in plan.superfragments:
        source_members = fragments.for_layer(superfragment.layer_ordinal)
        authenticated_members.extend(
            superfragment.member_fragment_sha256s
        )
        assert superfragment.member_fragment_sha256s == tuple(
            sorted(value.artifact_sha256 for value in source_members)
        )
        assert superfragment.channel_indices == (0, 1, 2)
        assert superfragment.mode_count == 3
        assert superfragment.native_parameter_count == 3 * (4 + 4 + 4)
        assert superfragment.fisher_mass == pytest.approx(
            sum(value.fisher_mass for value in source_members)
        )
        assert plan.for_layer(superfragment.layer_ordinal) == superfragment

    assert tuple(sorted(authenticated_members)) == (
        plan.source_fragment_sha256s
    )


def test_superfragment_plan_roundtrip_is_deterministic_strict_and_private() -> None:
    fragments = _fragment_plan()
    first = build_parameter_layer_superfragments(fragments)
    second = build_parameter_layer_superfragments(fragments)
    restored = ParameterLayerSuperfragmentPlan.from_state_dict(
        first.state_dict()
    )

    assert first == second == restored
    assert first.state_dict() == second.state_dict() == restored.state_dict()
    assert first.artifact_sha256 == restored.artifact_sha256
    metadata = restored.metadata()
    assert metadata["contains_source_model_weights"] is False
    assert metadata["contains_prompt_text"] is False
    assert metadata["contains_activation_rows"] is False
    assert metadata["contains_gradient_rows"] is False
    assert metadata["contains_parameter_values"] is False
    assert metadata["analysis_only"] is True
    assert metadata["authorizes_execution"] is False
    assert "score_factor" not in repr(restored.state_dict())

    missing = copy.deepcopy(first.state_dict())
    del missing["layer_count"]
    with pytest.raises(ValueError, match="fields"):
        ParameterLayerSuperfragmentPlan.from_state_dict(missing)


def test_roundtrip_authenticates_nested_fragment_and_member_hash_catalogs() -> None:
    plan = build_parameter_layer_superfragments(_fragment_plan())

    nested_fragment_tamper = copy.deepcopy(plan.state_dict())
    nested_fragment_tamper["source_fragment_plan"]["fragments"][0][
        "channel_indices"
    ] = (99,)
    with pytest.raises(ValueError, match="canonical|aligned|hash"):
        ParameterLayerSuperfragmentPlan.from_state_dict(
            nested_fragment_tamper
        )

    member_catalog_tamper = copy.deepcopy(plan.state_dict())
    member_catalog_tamper["superfragments"][0][
        "member_fragment_sha256s"
    ] = (_digest("foreign-fragment"),)
    with pytest.raises(ValueError, match="hash|coverage|exhaust"):
        ParameterLayerSuperfragmentPlan.from_state_dict(
            member_catalog_tamper
        )

    summary_tamper = copy.deepcopy(plan.state_dict())
    summary_tamper["assigned_native_parameter_count"] += 1
    with pytest.raises(ValueError, match="summaries"):
        ParameterLayerSuperfragmentPlan.from_state_dict(summary_tamper)


def test_plan_rejects_duplicate_or_missing_member_authentication() -> None:
    plan = build_parameter_layer_superfragments(_fragment_plan())
    first, second = plan.superfragments
    duplicate_members = replace(
        first,
        member_fragment_sha256s=second.member_fragment_sha256s,
        artifact_sha256="",
    )

    with pytest.raises(ValueError, match="disjointly and exhaustively"):
        ParameterLayerSuperfragmentPlan(
            source_fragment_plan=plan.source_fragment_plan,
            superfragments=(duplicate_members, second),
        )
    with pytest.raises(ValueError, match="disjointly and exhaustively"):
        ParameterLayerSuperfragmentPlan(
            source_fragment_plan=plan.source_fragment_plan,
            superfragments=(first,),
        )


def test_builder_rejects_partial_or_nonexhaustive_channel_coverage() -> None:
    fragment_plan = _fragment_plan()
    retained = fragment_plan.fragments[:-1]
    partial = replace(
        fragment_plan,
        assigned_group_count=sum(
            value.mode_count for value in retained
        ),
        assigned_native_parameter_count=sum(
            value.native_parameter_count for value in retained
        ),
        fragments=retained,
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="exhaustively assign"):
        build_parameter_layer_superfragments(partial)

    overlapping_fragments = tuple(
        replace(
            fragment,
            channel_indices=tuple(range(fragment.mode_count)),
            artifact_sha256="",
        )
        for fragment in fragment_plan.fragments
    )
    overlapping = replace(
        fragment_plan,
        fragments=overlapping_fragments,
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="exhaustively cover"):
        build_parameter_layer_superfragments(overlapping)

    shifted_fragments = []
    for layer_ordinal in (0, 1):
        offset = 1
        for fragment in fragment_plan.for_layer(layer_ordinal):
            shifted_fragments.append(
                replace(
                    fragment,
                    channel_indices=tuple(
                        range(offset, offset + fragment.mode_count)
                    ),
                    artifact_sha256="",
                )
            )
            offset += fragment.mode_count
    shifted = replace(
        fragment_plan,
        fragments=tuple(
            sorted(
                shifted_fragments,
                key=lambda value: (
                    value.cluster_id,
                    value.layer_ordinal,
                    value.layer_id,
                    value.activation_site,
                ),
            )
        ),
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="exhaustively cover"):
        build_parameter_layer_superfragments(shifted)
