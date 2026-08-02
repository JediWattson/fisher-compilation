from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.conditional_spectral_generator import (
    fit_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from fisher_graph.graph_wavelet_supermode_merge import (
    FitOnlyGraphWaveletSupermodePath,
    GraphWaveletSingletonPrune,
    GraphWaveletSupermodePair,
    fit_graph_wavelet_supermode_merge,
)


def _path_laplacian(node_count: int) -> torch.Tensor:
    adjacency = torch.zeros(
        (node_count, node_count),
        dtype=torch.float64,
    )
    for index in range(node_count - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    return torch.diag(adjacency.sum(dim=1)) - adjacency


def _partition(
    response: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = response.shape[1] // 2
    return (
        response[:, :midpoint].clone(),
        response[:, midpoint:].clone(),
    )


def _mixed_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parent = torch.eye(6, dtype=torch.float64)
    response = torch.tensor(
        [
            [1.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.001, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0],
        ],
        dtype=torch.float64,
    )
    return parent, response, _path_laplacian(6)


def _fit_mixed(
    *,
    method: str = "response_only",
    minimum_rank: int = 4,
    permutation_seed: int = 0,
) -> FitOnlyGraphWaveletSupermodePath:
    parent, response, laplacian = _mixed_fixture()
    return fit_graph_wavelet_supermode_merge(
        parent,
        response,
        laplacian,
        minimum_rank=minimum_rank,
        method=method,  # type: ignore[arg-type]
        fit_fold_responses=_partition(response),
        topology_top_k=3,
        permutation_seed=permutation_seed,
    )


def test_mixed_path_selects_true_merge_and_cheaper_singleton_prune() -> None:
    path = _fit_mixed()

    assert path.available_ranks == (6, 5, 4)
    assert len(path.selected_actions) == 2
    assert isinstance(path.selected_actions[0], GraphWaveletSupermodePair)
    assert path.selected_actions[0].endpoints == (0, 1)
    assert path.selected_actions[0].action_loss == pytest.approx(0.0)
    assert path.selected_actions[0].minimum_squared_loading == pytest.approx(
        0.5
    )
    assert path.selected_actions[0].mixing_participation == pytest.approx(1.0)
    assert path.selected_actions[0].lofo_loading_stability == pytest.approx(
        1.0
    )
    assert isinstance(path.selected_actions[1], GraphWaveletSingletonPrune)
    assert path.selected_actions[1].parent_index == 4
    assert path.selected_actions[1].action_loss == pytest.approx(1.0e-6)
    assert len(path.selected_pairs) == 1
    assert len(path.singleton_prunes) == 1

    for rank in path.available_ranks:
        mixed = path.basis(rank)
        control = path.one_hot_control_basis(rank)
        assert mixed.shape == (6, rank)
        assert control.shape == (6, rank)
        torch.testing.assert_close(
            mixed.T @ mixed,
            torch.eye(rank, dtype=torch.float64),
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        torch.testing.assert_close(
            control.T @ control,
            torch.eye(rank, dtype=torch.float64),
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        torch.testing.assert_close(
            path.paired_delete_control_basis(rank),
            control,
        )

    final = path.basis(4)
    expected_merge = torch.tensor(
        [2.0**-0.5, 2.0**-0.5, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(final[:, 0], expected_merge)
    assert not bool((final[4] != 0.0).any())
    control = path.one_hot_control_basis(4)
    assert bool(torch.equal(control[:, 0], torch.eye(6)[0]))
    assert not bool((control[1] != 0.0).any())
    assert not bool((control[4] != 0.0).any())


def test_paths_are_deterministic_nested_and_parent_sign_invariant() -> None:
    first = _fit_mixed(minimum_rank=3)
    second = _fit_mixed(minimum_rank=3)
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.selected_actions == second.selected_actions

    parent, response, laplacian = _mixed_fixture()
    signed = parent.clone()
    signed[:, 0] *= -1.0
    signed[:, 3] *= -1.0
    sign_replayed = fit_graph_wavelet_supermode_merge(
        signed,
        response,
        laplacian,
        minimum_rank=3,
        method="response_only",
        fit_fold_responses=_partition(response),
        topology_top_k=3,
    )
    assert sign_replayed.artifact_sha256 == first.artifact_sha256

    previous = first.basis(6)
    previous_control = first.one_hot_control_basis(6)
    for rank in (5, 4, 3):
        current = first.basis(rank)
        current_control = first.one_hot_control_basis(rank)
        torch.testing.assert_close(
            previous @ (previous.T @ current),
            current,
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        torch.testing.assert_close(
            previous_control @ (previous_control.T @ current_control),
            current_control,
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        previous = current
        previous_control = current_control


def test_path_can_descend_below_half_rank_without_exhausting_endpoints() -> None:
    path = _fit_mixed(minimum_rank=1)

    assert len(path.selected_actions) == 5
    assert len(path.selected_pairs) <= 1
    assert path.basis(1).shape == (6, 1)
    assert path.one_hot_control_basis(1).shape == (6, 1)
    torch.testing.assert_close(
        path.basis(1).T @ path.basis(1),
        torch.ones((1, 1), dtype=torch.float64),
    )


def test_graph_top_k_changes_pair_and_permuted_null_is_deterministic() -> None:
    parent = torch.eye(5, dtype=torch.float64)
    response = torch.tensor(
        [
            [1.0, 2.0, 1.0, 2.0],
            [1.0, 2.1, 1.0, 2.1],
            [1.0, 2.0, 1.0, 2.0],
            [0.0, 0.0, 2.0, 1.0],
            [2.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    laplacian = _path_laplacian(5)
    common = {
        "minimum_rank": 4,
        "fit_fold_responses": _partition(response),
        "topology_top_k": 1,
    }
    response_only = fit_graph_wavelet_supermode_merge(
        parent,
        response,
        laplacian,
        method="response_only",
        **common,
    )
    graph_local = fit_graph_wavelet_supermode_merge(
        parent,
        response,
        laplacian,
        method="graph_local",
        **common,
    )

    assert isinstance(
        response_only.selected_actions[0],
        GraphWaveletSupermodePair,
    )
    assert response_only.selected_actions[0].endpoints == (0, 2)
    assert isinstance(
        graph_local.selected_actions[0],
        GraphWaveletSupermodePair,
    )
    assert graph_local.selected_actions[0].endpoints == (0, 1)
    assert graph_local.selected_actions[
        0
    ].topology_union_top_k_eligible is True
    assert graph_local.selected_actions[0].topology_interaction > 0.0

    first_null = fit_graph_wavelet_supermode_merge(
        parent,
        response,
        laplacian,
        method="permuted_graph_local",
        permutation_seed=17,
        **common,
    )
    second_null = fit_graph_wavelet_supermode_merge(
        parent,
        response,
        laplacian,
        method="permuted_graph_local",
        permutation_seed=17,
        **common,
    )
    assert first_null.artifact_sha256 == second_null.artifact_sha256
    assert torch.equal(
        first_null.topology_permutation,
        second_null.topology_permutation,
    )
    assert not torch.equal(
        first_null.topology_permutation,
        torch.arange(5, dtype=torch.int64),
    )
    assert not torch.equal(
        first_null.selection_topology_matrix,
        first_null.native_topology_matrix,
    )


def test_graph_local_rejects_zero_weight_top_k_ties() -> None:
    parent = torch.eye(3, dtype=torch.float64)
    response = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    path = fit_graph_wavelet_supermode_merge(
        parent,
        response,
        torch.zeros((3, 3), dtype=torch.float64),
        minimum_rank=2,
        method="graph_local",
        fit_fold_responses=_partition(response),
        topology_top_k=1,
    )

    assert isinstance(path.selected_actions[0], GraphWaveletSingletonPrune)


def test_loading_and_lofo_gates_prevent_one_hot_pairs_from_being_merges() -> None:
    parent = torch.eye(2, dtype=torch.float64)
    laplacian = _path_laplacian(2)

    first_fold = torch.tensor(
        [[3.0, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    second_fold = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0]],
        dtype=torch.float64,
    )
    unstable_response = torch.cat((first_fold, second_fold), dim=1)
    rejected_unstable = fit_graph_wavelet_supermode_merge(
        parent,
        unstable_response,
        laplacian,
        minimum_rank=1,
        method="response_only",
        fit_fold_responses=(first_fold, second_fold),
        minimum_lofo_loading_stability=0.90,
        topology_top_k=1,
    )
    accepted_unstable = fit_graph_wavelet_supermode_merge(
        parent,
        unstable_response,
        laplacian,
        minimum_rank=1,
        method="response_only",
        fit_fold_responses=(first_fold, second_fold),
        minimum_lofo_loading_stability=0.89,
        topology_top_k=1,
    )
    assert isinstance(
        rejected_unstable.selected_actions[0],
        GraphWaveletSingletonPrune,
    )
    assert isinstance(
        accepted_unstable.selected_actions[0],
        GraphWaveletSupermodePair,
    )
    assert accepted_unstable.selected_actions[
        0
    ].lofo_loading_stability == pytest.approx(0.8944271909999159)

    imbalanced_folds = (
        torch.tensor([[5.0, 0.0], [0.5, 0.0]], dtype=torch.float64),
        torch.tensor([[5.0, 0.0], [0.5, 0.0]], dtype=torch.float64),
    )
    imbalanced = torch.cat(imbalanced_folds, dim=1)
    loading_guarded = fit_graph_wavelet_supermode_merge(
        parent,
        imbalanced,
        laplacian,
        minimum_rank=1,
        method="response_only",
        fit_fold_responses=imbalanced_folds,
        minimum_squared_loading=0.10,
        topology_top_k=1,
    )
    loading_open = fit_graph_wavelet_supermode_merge(
        parent,
        imbalanced,
        laplacian,
        minimum_rank=1,
        method="response_only",
        fit_fold_responses=imbalanced_folds,
        minimum_squared_loading=0.0,
        topology_top_k=1,
    )
    assert isinstance(
        loading_guarded.selected_actions[0],
        GraphWaveletSingletonPrune,
    )
    assert isinstance(
        loading_open.selected_actions[0],
        GraphWaveletSupermodePair,
    )
    assert loading_open.selected_actions[0].minimum_squared_loading < 0.10


@pytest.mark.parametrize(
    "uninformative_fold",
    (
        torch.zeros((2, 2), dtype=torch.float64),
        torch.tensor(
            [[1.0e-10, 0.0], [1.0e-10, 0.0]],
            dtype=torch.float64,
        ),
        torch.eye(2, dtype=torch.float64),
    ),
    ids=("zero-energy", "near-zero-energy", "degenerate-direction"),
)
def test_lofo_rejects_unidentifiable_training_directions(
    uninformative_fold: torch.Tensor,
) -> None:
    parent = torch.eye(2, dtype=torch.float64)
    informative_fold = torch.tensor(
        [[2.0, 0.0], [2.0, 0.0]],
        dtype=torch.float64,
    )
    response = torch.cat((informative_fold, uninformative_fold), dim=1)
    path = fit_graph_wavelet_supermode_merge(
        parent,
        response,
        _path_laplacian(2),
        minimum_rank=1,
        method="response_only",
        fit_fold_responses=(informative_fold, uninformative_fold),
        topology_top_k=1,
    )

    assert isinstance(path.selected_actions[0], GraphWaveletSingletonPrune)


def test_state_roundtrip_report_and_tamper_detection() -> None:
    path = _fit_mixed(minimum_rank=3)
    restored = FitOnlyGraphWaveletSupermodePath.from_state_dict(
        path.state_dict()
    )
    assert restored.artifact_sha256 == path.artifact_sha256
    assert restored.report()["heldout_input_used"] is False
    assert restored.report()["fit_scope"] == (
        "caller_supplied_weighted_fit_response_and_fit_folds_only"
    )
    assert restored.report()["merge_action_count"] >= 1
    assert len(restored.report()["action_diagnostics"]) == 3

    bad_hash = copy.deepcopy(path.state_dict())
    bad_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact hash"):
        FitOnlyGraphWaveletSupermodePath.from_state_dict(bad_hash)

    mutable = FitOnlyGraphWaveletSupermodePath.from_state_dict(
        path.state_dict()
    )
    mutable.parent_basis[0, 0] += 0.25
    with pytest.raises(ValueError, match="artifact hash"):
        mutable.validate_integrity()

    bad_actions = copy.deepcopy(path.state_dict())
    actions = list(bad_actions["selected_actions"])
    actions[0] = dict(actions[0])
    actions[0]["action_loss"] = float(actions[0]["action_loss"]) + 0.1
    bad_actions["selected_actions"] = tuple(actions)
    with pytest.raises(ValueError):
        FitOnlyGraphWaveletSupermodePath.from_state_dict(bad_actions)


def test_state_replays_topology_method_semantics() -> None:
    graph_local = _fit_mixed(method="graph_local")
    bad_native = copy.deepcopy(graph_local.state_dict())
    selection = bad_native["selection_topology_matrix"].clone()
    selection[0, 0] += 0.25
    bad_native["selection_topology_matrix"] = selection
    bad_native["artifact_sha256"] = ""
    with pytest.raises(
        ValueError,
        match="selection topology must equal native topology",
    ):
        FitOnlyGraphWaveletSupermodePath.from_state_dict(bad_native)

    permuted = _fit_mixed(
        method="permuted_graph_local",
        permutation_seed=17,
    )
    bad_permutation = copy.deepcopy(permuted.state_dict())
    permutation = bad_permutation["topology_permutation"].clone()
    permutation[0], permutation[1] = (
        permutation[1].clone(),
        permutation[0].clone(),
    )
    bad_permutation["topology_permutation"] = permutation
    bad_permutation["artifact_sha256"] = ""
    with pytest.raises(
        ValueError,
        match="does not replay permutation_seed",
    ):
        FitOnlyGraphWaveletSupermodePath.from_state_dict(bad_permutation)


def test_fit_only_inputs_and_geometry_fail_closed() -> None:
    parent, response, laplacian = _mixed_fixture()
    folds = _partition(response)
    common = {
        "minimum_rank": 4,
        "method": "response_only",
        "fit_fold_responses": folds,
        "topology_top_k": 3,
    }

    bad_parent = parent.clone()
    bad_parent[:, 1] = bad_parent[:, 0]
    with pytest.raises(ValueError, match="orthonormal"):
        fit_graph_wavelet_supermode_merge(
            bad_parent,
            response,
            laplacian,
            **common,
        )

    drifted_folds = (folds[0], folds[1] * 1.01)
    with pytest.raises(ValueError, match="partition"):
        fit_graph_wavelet_supermode_merge(
            parent,
            response,
            laplacian,
            **{**common, "fit_fold_responses": drifted_folds},
        )

    asymmetric = laplacian.clone()
    asymmetric[0, 1] += 0.25
    with pytest.raises(ValueError, match="symmetric"):
        fit_graph_wavelet_supermode_merge(
            parent,
            response,
            asymmetric,
            **common,
        )

    with pytest.raises(ValueError, match="topology_top_k"):
        fit_graph_wavelet_supermode_merge(
            parent,
            response,
            laplacian,
            **{**common, "topology_top_k": 6},
        )

    with pytest.raises(ValueError, match="nonzero permutation seed"):
        fit_graph_wavelet_supermode_merge(
            parent,
            response,
            laplacian,
            **{**common, "permutation_seed": 1},
        )


def test_equal_rank_conditional_plans_have_identical_payload() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    responses = torch.randn(
        (4, 3, 3, 3),
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.tensor([0.8, 1.0, 1.2, 1.4], dtype=torch.float64)
    weighted = responses * scales.view(-1, 1, 1, 1)
    folds = tuple(weighted[:, index : index + 1] for index in range(3))
    binding = "a" * 64
    parent_plan = fit_conditional_spectral_generator(
        responses,
        scales,
        (8, 24, 40),
        (8, 24, 40),
        4,
        3,
        response_binding_sha256=binding,
    )
    path = fit_graph_wavelet_supermode_merge(
        torch.eye(4, dtype=torch.float64),
        weighted,
        _path_laplacian(4),
        minimum_rank=3,
        method="response_only",
        fit_fold_responses=folds,
        topology_top_k=3,
    )
    mixed = fit_conditional_spectral_generator_with_source_basis(
        responses,
        scales,
        (8, 24, 40),
        (8, 24, 40),
        path.basis(3),
        3,
        source_basis_kind="fit_only_graph_wavelet_gomp",
        source_basis_fit_weighted_kernels_sha256=(
            parent_plan.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=binding,
    )
    one_hot = fit_conditional_spectral_generator_with_source_basis(
        responses,
        scales,
        (8, 24, 40),
        (8, 24, 40),
        path.one_hot_control_basis(3),
        3,
        source_basis_kind="fit_only_graph_wavelet_gomp",
        source_basis_fit_weighted_kernels_sha256=(
            parent_plan.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=binding,
    )
    assert mixed.source_rank == one_hot.source_rank == 3
    assert mixed.stored_coefficient_count == one_hot.stored_coefficient_count
    assert mixed.stored_coefficient_count == (
        4 * 3 + 3 * 3 + 3 * 3 * 3 * 3
    )
