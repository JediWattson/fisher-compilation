from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from fisher_graph.fisher_prompt_clustering import (
    FisherPromptClusterConfig,
    FisherPromptClusterPlan,
    adjusted_rand_index,
    build_fisher_prompt_clusters,
    fisher_prompt_effects_sha256,
    fisher_prompt_cluster_stability,
    prompt_cluster_fisher_signatures,
    prompt_mode_gate_effects,
)
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _catalog(
    width: int,
) -> tuple[CrossBlockLayerSpec, tuple[object, ...]]:
    spec = CrossBlockLayerSpec(
        layer_id="layer.0",
        layer_ordinal=0,
        activation_site="mlp.gated",
        width=width,
    )
    modes = tuple(
        spec.mode_key(index, fisher_rank=index)
        for index in range(width)
    )
    return spec, modes


def _config(
    width: int,
    *,
    cluster_count: int = 2,
    chunk_size: int = 2,
    catalog_permutation: tuple[int, ...] | None = None,
) -> FisherPromptClusterConfig:
    spec, modes = _catalog(width)
    if catalog_permutation is not None:
        modes = tuple(modes[index] for index in catalog_permutation)
    return FisherPromptClusterConfig(
        model_fingerprint=_digest("model"),
        calibration_split_sha256=_digest("split"),
        objective_sha256=_digest("causal-nll"),
        source_fisher_coupling_sha256=_digest("grouped-fisher"),
        layer_specs=(spec,),
        mode_catalog=modes,
        cluster_count=cluster_count,
        max_iterations=50,
        tolerance=1e-13,
        mode_chunk_size=chunk_size,
    )


def _clustered_effects() -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.tensor(
        [1.0, 0.25, -0.1, 0.7, -0.35, 0.2],
        dtype=torch.float64,
    )
    second = torch.tensor(
        [-0.1, 0.9, 0.6, -0.2, 0.15, -0.8],
        dtype=torch.float64,
    )
    second = second - torch.dot(second, first) / torch.dot(
        first,
        first,
    ) * first
    effects = torch.stack(
        (
            1.8 * first,
            -1.1 * first
            + torch.tensor(
                [0.0, 0.002, 0.0, -0.001, 0.0, 0.001],
                dtype=torch.float64,
            ),
            0.7 * first,
            1.5 * second,
            -0.9 * second,
            0.55 * second
            + torch.tensor(
                [0.001, 0.0, -0.001, 0.0, 0.001, 0.0],
                dtype=torch.float64,
            ),
        ),
        dim=1,
    )
    expected = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
    return effects, expected


def test_axial_clusters_recover_positive_and_negative_mode_copies() -> None:
    effects, expected = _clustered_effects()
    plan = build_fisher_prompt_clusters(
        effects,
        _config(effects.shape[1]),
    )

    assert adjusted_rand_index(plan.assignments, expected) == pytest.approx(
        1.0
    )
    assert plan.cluster_counts.tolist() == [3, 3]
    assert set(plan.orientations.tolist()) == {-1, 1}
    assert float(plan.similarities.min().item()) > 0.999
    assert plan.converged
    assert plan.source_effects_sha256 == fisher_prompt_effects_sha256(effects)


def test_build_is_deterministic_and_mode_permutation_safe() -> None:
    effects, _ = _clustered_effects()
    direct_config = _config(effects.shape[1])
    first = build_fisher_prompt_clusters(effects, direct_config)
    repeated = build_fisher_prompt_clusters(effects, direct_config)

    assert first.artifact_sha256 == repeated.artifact_sha256
    assert torch.equal(first.assignments, repeated.assignments)
    assert torch.equal(first.orientations, repeated.orientations)
    assert torch.equal(first.centroids, repeated.centroids)

    permutation = (4, 1, 5, 0, 3, 2)
    permuted = build_fisher_prompt_clusters(
        effects[:, permutation],
        _config(
            effects.shape[1],
            catalog_permutation=permutation,
        ),
    )
    direct_by_mode = {
        mode: int(first.assignments[index].item())
        for index, mode in enumerate(first.mode_catalog)
    }
    permuted_by_mode = {
        mode: int(permuted.assignments[index].item())
        for index, mode in enumerate(permuted.mode_catalog)
    }
    assert direct_by_mode == permuted_by_mode
    assert torch.equal(first.centroids, permuted.centroids)
    assert fisher_prompt_cluster_stability(first, permuted) == pytest.approx(
        1.0
    )


def test_gate_effect_is_invariant_to_per_mode_coordinate_gauge() -> None:
    generator = torch.Generator().manual_seed(19)
    activations = torch.randn(
        4,
        5,
        6,
        generator=generator,
        dtype=torch.float64,
    )
    gradients = torch.randn(
        4,
        5,
        6,
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.tensor(
        [0.25, -0.5, 2.0, -4.0, 8.0, 0.125],
        dtype=torch.float64,
    )
    expected = prompt_mode_gate_effects(activations, gradients)
    changed = prompt_mode_gate_effects(
        activations * scales,
        gradients / scales,
    )
    torch.testing.assert_close(changed, expected, rtol=1e-14, atol=1e-14)

    first = build_fisher_prompt_clusters(expected, _config(6))
    second = build_fisher_prompt_clusters(changed, _config(6))
    assert first.artifact_sha256 == second.artifact_sha256


def test_zero_fisher_energy_modes_are_explicitly_unassigned() -> None:
    effects, _ = _clustered_effects()
    effects = torch.cat(
        (effects, torch.zeros(effects.shape[0], 1, dtype=torch.float64)),
        dim=1,
    )
    plan = build_fisher_prompt_clusters(effects, _config(7))

    assert int(plan.assignments[-1].item()) == -1
    assert int(plan.orientations[-1].item()) == 0
    assert float(plan.similarities[-1].item()) == 0.0
    assert int(plan.cluster_counts.sum().item()) == 6
    assert float(plan.cluster_mass.sum().item()) == pytest.approx(
        float(effects.square().sum().item())
    )


def test_signature_equals_explicit_projected_fisher_trace() -> None:
    effects, _ = _clustered_effects()
    plan = build_fisher_prompt_clusters(effects, _config(6))
    signatures = prompt_cluster_fisher_signatures(effects, plan)

    for prompt in range(effects.shape[0]):
        row = effects[prompt]
        fisher = torch.outer(row, row)
        for cluster in range(plan.cluster_count):
            selector = torch.diag(
                (plan.assignments == cluster).to(dtype=torch.float64)
            )
            explicit = torch.trace(selector @ fisher @ selector)
            torch.testing.assert_close(
                signatures[prompt, cluster],
                explicit,
                rtol=1e-14,
                atol=1e-14,
            )
    torch.testing.assert_close(
        signatures.sum(dim=1),
        effects.square().sum(dim=1),
        rtol=1e-14,
        atol=1e-14,
    )


def test_assignment_and_centroid_results_are_chunk_invariant() -> None:
    effects, _ = _clustered_effects()
    singleton = build_fisher_prompt_clusters(
        effects,
        _config(6, chunk_size=1),
    )
    full = build_fisher_prompt_clusters(
        effects,
        _config(6, chunk_size=64),
    )

    assert torch.equal(singleton.assignments, full.assignments)
    assert torch.equal(singleton.orientations, full.orientations)
    assert torch.equal(singleton.similarities, full.similarities)
    assert torch.equal(singleton.centroids, full.centroids)
    assert torch.equal(singleton.cluster_counts, full.cluster_counts)
    assert torch.equal(singleton.cluster_mass, full.cluster_mass)


def test_state_roundtrip_is_strict_and_detects_hash_poisoning() -> None:
    effects, _ = _clustered_effects()
    plan = build_fisher_prompt_clusters(effects, _config(6))
    restored = FisherPromptClusterPlan.from_state_dict(plan.state_dict())

    assert restored.artifact_sha256 == plan.artifact_sha256
    assert restored.config.artifact_sha256 == plan.config.artifact_sha256
    assert torch.equal(restored.assignments, plan.assignments)
    assert torch.equal(restored.centroids, plan.centroids)

    changed_config = copy.deepcopy(plan.config.state_dict())
    changed_config["cluster_count"] = 3
    with pytest.raises(ValueError, match="hash mismatch"):
        FisherPromptClusterConfig.from_state_dict(changed_config)

    changed_tensor = copy.deepcopy(plan.state_dict())
    changed_tensor["similarities"][0] -= 0.01
    with pytest.raises(ValueError, match="similarities hash mismatch"):
        FisherPromptClusterPlan.from_state_dict(changed_tensor)

    changed_binding = copy.deepcopy(plan.state_dict())
    changed_binding["source_effects_sha256"] = _digest("different-effects")
    with pytest.raises(ValueError, match="plan hash mismatch"):
        FisherPromptClusterPlan.from_state_dict(changed_binding)

    unexpected = copy.deepcopy(plan.state_dict())
    unexpected["surprise"] = True
    with pytest.raises(ValueError, match="state fields"):
        FisherPromptClusterPlan.from_state_dict(unexpected)


def test_adjusted_rand_index_is_label_invariant_and_ignores_zeros() -> None:
    assert adjusted_rand_index(
        [0, 0, 1, 1, -1],
        [7, 7, 3, 3, -1],
    ) == pytest.approx(1.0)
    assert adjusted_rand_index(
        torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        torch.tensor([0, 1, 0, 1], dtype=torch.int64),
    ) == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="jointly assigned"):
        adjusted_rand_index([-1, -1], [0, 1])
    with pytest.raises(ValueError, match="jointly assigned"):
        adjusted_rand_index([0, -1, -1], [0, 1, 1])
