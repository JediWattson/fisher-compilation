from __future__ import annotations

import copy
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
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sources() -> tuple[object, object]:
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
    return fisher, build_fisher_prompt_clusters(scores, config)


def test_global_clusters_are_lowered_to_exact_layer_fragments() -> None:
    fisher, clusters = _sources()
    plan = build_parameter_cluster_layer_fragments(clusters, fisher)

    assert plan.fragment_count == 4
    assert plan.assigned_group_count == 6
    assert plan.source_group_count == 6
    assert plan.assigned_native_parameter_count == 6 * (4 + 4 + 4)
    assert {value.cluster_id for value in plan.fragments} == {0, 1}
    assert {value.layer_ordinal for value in plan.fragments} == {0, 1}
    assert sorted(
        group
        for fragment in plan.fragments
        for group in fragment.group_indices
    ) == list(range(6))
    assert all(
        fragment.removed_mode_indices == fragment.channel_indices
        for fragment in plan.fragments
    )
    assert sum(value.fisher_mass for value in plan.fragments) == pytest.approx(
        float(fisher.fisher_mass.sum().item())
    )
    assert plan.top_by_fisher_mass(1)[0].fisher_mass == max(
        value.fisher_mass for value in plan.fragments
    )


def test_fragment_plan_roundtrip_is_strict_and_private() -> None:
    fisher, clusters = _sources()
    plan = build_parameter_cluster_layer_fragments(clusters, fisher)
    restored = ParameterClusterLayerFragmentPlan.from_state_dict(
        plan.state_dict()
    )

    assert restored.artifact_sha256 == plan.artifact_sha256
    assert restored.fragments == plan.fragments
    metadata = restored.metadata()
    assert metadata["contains_source_model_weights"] is False
    assert metadata["contains_prompt_text"] is False
    assert metadata["contains_activation_rows"] is False
    assert "score_factor" not in repr(restored.state_dict())

    tampered = copy.deepcopy(plan.state_dict())
    tampered["fragments"][0]["channel_indices"] = (99,)
    with pytest.raises(ValueError, match="canonical|aligned|hash"):
        ParameterClusterLayerFragmentPlan.from_state_dict(tampered)


def test_builder_rejects_fisher_or_cluster_provenance_mismatch() -> None:
    fisher, clusters = _sources()
    poisoned = copy.deepcopy(clusters.state_dict())
    poisoned["config"]["source_fisher_coupling_sha256"] = _digest("foreign")
    poisoned["config"]["artifact_sha256"] = _digest("fake")
    with pytest.raises(ValueError, match="hash mismatch"):
        type(clusters).from_state_dict(poisoned)

    other_fisher, _ = _sources()
    object.__setattr__(
        other_fisher,
        "artifact_sha256",
        _digest("other-fisher"),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        build_parameter_cluster_layer_fragments(clusters, other_fisher)
