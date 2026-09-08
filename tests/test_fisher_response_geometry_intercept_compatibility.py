from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
import json
import math

import pytest
import torch

import fisher_graph.fisher_response_geometry as geometry
import fisher_graph.fisher_response_geometry_failure_diagnosis as diagnosis
import fisher_graph.fisher_response_geometry_intercept_compatibility as compatibility
from fisher_graph.fisher_response_geometry_replay import load_replay


DTYPE = torch.float64


def _tensor(values: object) -> torch.Tensor:
    return torch.tensor(values, dtype=DTYPE)


def _unit_training_replay() -> dict[str, object]:
    """A literal unit fixture; never evaluates the proposed scientific arm."""
    messages = _tensor([
        [1.0, 0.0], [1.0, 0.1], [0.9, -0.1],
        [-1.0, 0.0], [-1.0, 0.1], [-0.9, -0.1],
        [0.0, 1.0], [0.1, 1.0], [-0.1, 0.9],
        [0.0, -1.0], [0.1, -1.0], [-0.1, -0.9],
    ])
    states = torch.cat((messages, messages[:, :1] + 0.2 * messages[:, 1:]), dim=1)
    G = _tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    U, Q = G.clone(), _tensor([[1.0], [0.0]])
    factor = torch.diag(_tensor([2.0, 1.0, 3.0]))
    projection = geometry.FisherProjection(basis=G, factor=factor)
    W, weight = _tensor([[1.0, 0.2], [-0.1, 0.7], [0.3, -0.2]]), _tensor([0.1, -0.1, 0.2])
    teacher = lambda m: geometry.normalization_update(m, W, weight, 0.01)
    outputs = teacher(messages)
    A = torch.stack([
        projection.coordinate_map @ geometry.jvp_actions(teacher, m, Q)
        for m in messages
    ])
    h_scale = float(states.square().mean().sqrt())
    m_scale = float(messages.square().mean().sqrt())
    return {
        "kernel": geometry,
        "tensors": {"W": W, "norm_weight": weight, "G": G, "U": U, "Q": Q,
                    "F_factor": factor, "D_G": projection.coordinate_map},
        "projection": projection,
        "config": {"epsilon": 0.01, "seed": 7, "chart_count": 3,
                   "residual_width": 3, "message_width": 2,
                   "output_rank": 2, "state_rank": 2, "message_rank": 1,
                   "kmeans_iterations": 30, "support_radius_multiplier": 1.1,
                   "metric_rtol": 1e-12, "finite_radii": [0.001, 0.01, 0.1]},
        "saved_artifact": {"h_feature_scale": h_scale, "m_feature_scale": m_scale},
        "train_h": states, "train_m": messages,
        "train_features": torch.cat((states / h_scale, messages / m_scale), dim=1),
        "train_outputs": outputs, "train_signatures": A,
        "train_ids": [f"unit-family-{i // 3}" for i in range(len(messages))],
    }


class _TrainingOnly(Mapping[str, object]):
    def __init__(self, values: Mapping[str, object]):
        self.values = values

    def __getitem__(self, key: str) -> object:
        if key.startswith(("eval", "probe", "state_probe")):
            raise AssertionError(f"fitting accessed forbidden evaluation field: {key}")
        if key in {"banks", "router_centers", "direct_assignments", "runtime_assignments"}:
            raise AssertionError(f"fitting reused an old fitted bank or evaluation assignment: {key}")
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def test_pointwise_compact_intercept_preserves_both_input_and_output_truncation() -> None:
    replay = _unit_training_replay()
    tensors = replay["tensors"]
    messages = replay["train_m"][:2]
    outputs = replay["train_outputs"][:2]
    A = replay["train_signatures"][:2]
    D, Q, G = tensors["D_G"], tensors["Q"], tensors["G"]
    result = compatibility.pointwise_compact_intercepts(messages, outputs, A, D, Q, G)
    expected_beta = outputs @ D.T
    expected_a = messages @ Q
    expected_b = expected_beta - torch.einsum("nkr,nr->nk", A, expected_a)
    torch.testing.assert_close(result["beta"], expected_beta)
    torch.testing.assert_close(result["a"], expected_a)
    torch.testing.assert_close(result["b"], expected_b)
    torch.testing.assert_close(result["ambient"], expected_b @ G.T)

    teacher = lambda m: geometry.normalization_update(m, tensors["W"], tensors["norm_weight"], 0.01)
    full_R = torch.stack([geometry.jvp_actions(teacher, m, torch.eye(2, dtype=DTYPE)) for m in messages])
    exact_intercept = outputs - torch.einsum("ndp,np->nd", full_R, messages)
    projected_exact = exact_intercept @ D.T @ G.T
    assert not torch.allclose(result["ambient"], exact_intercept)
    assert not torch.allclose(result["ambient"], projected_exact)
    for i in range(len(messages)):
        chart = geometry.AffineResponseChart(
            state_anchor=replay["train_h"][i], message_anchor=messages[i],
            beta=result["beta"][i], B=torch.zeros((2, 2), dtype=DTYPE), A=A[i],
            U=tensors["U"], Q=Q, G=G, state_radius=10.0, message_radius=10.0,
        )
        test_message = messages[i] + _tensor([0.3, -0.2])
        actual = chart.predict(chart.state_anchor, test_message)
        expected = result["ambient"][i] + G @ A[i] @ Q.T @ test_message
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(diagnosis.compact_intercept(chart), result["ambient"][i])


@pytest.mark.parametrize("multiplier", [1e-300, 1e300, -1e-300, -1e300])
def test_block_normalization_is_scale_safe_and_globally_scale_invariant(multiplier: float) -> None:
    values = _tensor([[4.0, 2.0], [-1.0, 6.0], [2.0, -3.0], [3.0, 1.0]])
    original = compatibility.fit_block_transform(values)
    transformed = compatibility.fit_block_transform(multiplier * values)
    assert original["dropped"] is False
    assert transformed["dropped"] is False
    assert math.isfinite(transformed["reference_scale"]) and transformed["reference_scale"] > 0
    assert math.isfinite(transformed["centered_scale"]) and transformed["centered_scale"] > 0
    before = compatibility.apply_block_transform(values, original)
    after = compatibility.apply_block_transform(multiplier * values, transformed)
    torch.testing.assert_close(after, math.copysign(1.0, multiplier) * before, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.pdist(after), torch.pdist(before), rtol=1e-12, atol=1e-12)
    assert transformed["reference_scale"] / abs(multiplier) == pytest.approx(original["reference_scale"], rel=1e-12)
    assert transformed["centered_scale"] / abs(multiplier) == pytest.approx(original["centered_scale"], rel=1e-12)


@pytest.mark.parametrize("kind", ["zero", "constant", "near_constant", "varying"])
def test_block_classification_distinguishes_degeneracy_without_inventing_features(kind: str) -> None:
    rows = {
        "zero": torch.zeros((4, 2), dtype=DTYPE),
        "constant": _tensor([[2.0, -3.0]]).repeat(4, 1),
        "near_constant": _tensor([[2.0, -3.0], [2.0 + 1e-13, -3.0], [2.0 - 1e-13, -3.0], [2.0, -3.0]]),
        "varying": _tensor([[2.0, -3.0], [2.0 + 1e-5, -3.0], [2.0 - 1e-5, -3.0], [2.0, -3.0]]),
    }[kind]
    transform = compatibility.fit_block_transform(rows)
    assert transform["dropped"] is (kind != "varying")
    normalized = compatibility.apply_block_transform(rows, transform)
    if transform["dropped"]:
        assert torch.count_nonzero(normalized).item() == 0
        assert transform["reason"]
    else:
        assert float(normalized.square().sum(dim=1).mean()) == pytest.approx(1.0, rel=1e-12)
    for multiplier in (1e-200, 1e200, -1e200):
        scaled = compatibility.fit_block_transform(multiplier * rows)
        assert scaled["dropped"] == transform["dropped"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_blocks_fail_instead_of_becoming_constant(bad: float) -> None:
    with pytest.raises(ValueError):
        compatibility.fit_block_transform(_tensor([[1.0, bad], [2.0, 3.0]]))
    dropped = compatibility.fit_block_transform(torch.zeros((2, 2), dtype=DTYPE))
    with pytest.raises(ValueError):
        compatibility.apply_block_transform(_tensor([[bad, 0.0]]), dropped)


def test_evaluation_transform_uses_saved_training_statistics_without_reclassification() -> None:
    training = _tensor([[0.0, 1.0], [2.0, 3.0]])
    transform = compatibility.fit_block_transform(training)
    saved_mean = transform["mean"].clone()
    saved_scale = transform["centered_scale"]
    evaluation = _tensor([[100.0, 200.0], [-50.0, -100.0]])
    observed = compatibility.apply_block_transform(evaluation, transform)
    torch.testing.assert_close(observed, (evaluation - saved_mean) / saved_scale)
    torch.testing.assert_close(transform["mean"], saved_mean, rtol=0.0, atol=0.0)
    assert transform["centered_scale"] == saved_scale
    assert torch.linalg.vector_norm(observed.mean(dim=0)).item() > 1.0
    dropped = compatibility.fit_block_transform(torch.zeros((2, 2), dtype=DTYPE))
    torch.testing.assert_close(compatibility.apply_block_transform(evaluation, dropped), torch.zeros_like(evaluation))
    assert dropped["dropped"] is True


def test_scaled_arithmetic_avoids_mean_and_centering_overflow_when_result_is_representable() -> None:
    values = torch.full((10, 1), 1e308, dtype=DTYPE)
    values[-1, 0] = -1e308
    transform = compatibility.fit_block_transform(values)
    assert transform["dropped"] is False
    assert transform["reference_scale"] == pytest.approx(1e308)
    assert transform["centered_scale"] == pytest.approx(6e307)
    normalized = compatibility.apply_block_transform(values, transform)
    expected = torch.full((10, 1), 1.0 / 3.0, dtype=DTYPE)
    expected[-1, 0] = -3.0
    torch.testing.assert_close(normalized, expected, rtol=1e-12, atol=1e-12)


def test_unrepresentable_required_norm_fails_even_for_a_constant_block() -> None:
    with pytest.raises(ValueError):
        compatibility.fit_block_transform(_tensor([[1.6e308, 1.6e308], [1.6e308, 1.6e308]]))


def test_metric_root_makes_intercept_distances_equal_reconstructed_fisher_distances() -> None:
    G = _tensor([[1.0, 0.2], [0.3, 1.1], [0.2, -0.4]])
    factor = torch.diag(_tensor([2.0, 1.0, 3.0]))
    root = compatibility.supported_gram_square_root(G, factor)
    gram = (factor.T @ G).T @ (factor.T @ G)
    torch.testing.assert_close(root, root.T, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(root @ root, gram, rtol=1e-12, atol=1e-12)
    b = _tensor([[1.0, -2.0], [-0.5, 0.7], [0.8, 1.2]])
    encoded = b @ root.T
    ambient_weighted = (b @ G.T) @ factor
    torch.testing.assert_close(torch.pdist(encoded), torch.pdist(ambient_weighted), rtol=1e-12, atol=1e-12)


def test_consistent_coordinate_rotations_preserve_compact_intercepts_and_discovery_distances() -> None:
    replay = _unit_training_replay()
    tensors = replay["tensors"]
    G, Q, D = tensors["G"], tensors["Q"], tensors["D_G"]
    O, sign = _tensor([[0.6, -0.8], [0.8, 0.6]]), _tensor([[-1.0]])
    A = replay["train_signatures"]
    baseline = compatibility.pointwise_compact_intercepts(replay["train_m"], replay["train_outputs"], A, D, Q, G)
    rotated_A = torch.einsum("ak,nkr,rs->nas", O.T, A, sign)
    rotated = compatibility.pointwise_compact_intercepts(
        replay["train_m"], replay["train_outputs"], rotated_A, O.T @ D, Q @ sign, G @ O,
    )
    torch.testing.assert_close(rotated["ambient"], baseline["ambient"], rtol=1e-12, atol=1e-12)
    first_root = compatibility.supported_gram_square_root(G, tensors["F_factor"])
    second_root = compatibility.supported_gram_square_root(G @ O, tensors["F_factor"])
    first_z, second_z = baseline["b"] @ first_root.T, rotated["b"] @ second_root.T
    first_x, second_x = A.flatten(start_dim=1), rotated_A.flatten(start_dim=1)
    first_signature, second_signature = [], []
    for a, b in ((first_x, second_x), (first_z, second_z)):
        first_signature.append(compatibility.apply_block_transform(a, compatibility.fit_block_transform(a)))
        second_signature.append(compatibility.apply_block_transform(b, compatibility.fit_block_transform(b)))
    torch.testing.assert_close(
        torch.pdist(torch.cat(first_signature, dim=1)),
        torch.pdist(torch.cat(second_signature, dim=1)), rtol=1e-11, atol=1e-11,
    )


def test_same_jacobian_opposite_compact_offsets_remain_distinguishable() -> None:
    messages = _tensor([[0.7, -1.2], [-0.7, 1.2]])
    identity = torch.eye(2, dtype=DTYPE)
    fn = lambda m: geometry.normalization_update(m, identity, _tensor([0.0, 0.0]), 0.01)
    outputs = fn(messages)
    A = torch.stack([geometry.jvp_actions(fn, m, identity) for m in messages])
    torch.testing.assert_close(A[0], A[1], rtol=0.0, atol=0.0)
    compact = compatibility.pointwise_compact_intercepts(messages, outputs, A, identity, identity, identity)
    torch.testing.assert_close(compact["b"][0], -compact["b"][1], rtol=1e-12, atol=1e-12)
    x = A.flatten(start_dim=1)
    response_transform = compatibility.fit_block_transform(x)
    intercept_transform = compatibility.fit_block_transform(compact["b"])
    assert response_transform["dropped"] is True
    assert intercept_transform["dropped"] is False
    signatures = torch.cat((
        compatibility.apply_block_transform(x, response_transform),
        compatibility.apply_block_transform(compact["b"], intercept_transform),
    ), dim=1)
    assert torch.linalg.vector_norm(signatures[0] - signatures[1]).item() == pytest.approx(2.0)


@pytest.mark.parametrize("weight", [0, 1])
def test_fitting_reads_training_fields_only_and_rebuilds_chart_predictors(weight: int) -> None:
    replay = _unit_training_replay()
    guarded = _TrainingOnly(replay)
    arm = compatibility.fit_arm(guarded, weight)
    assert len(arm["charts"]) == 3
    assert len(arm["training_assignments"]) == len(replay["train_m"])
    assert len(arm["anchor_indices"]) == 3
    for chart, anchor in zip(arm["charts"], arm["anchor_indices"]):
        torch.testing.assert_close(chart.state_anchor, replay["train_h"][anchor], rtol=0.0, atol=0.0)
        torch.testing.assert_close(chart.message_anchor, replay["train_m"][anchor], rtol=0.0, atol=0.0)
        torch.testing.assert_close(chart.beta, replay["train_outputs"][anchor] @ replay["tensors"]["D_G"].T)
        torch.testing.assert_close(chart.A, replay["train_signatures"][anchor])
        assert torch.count_nonzero(chart.B).item() == 0
        torch.testing.assert_close(diagnosis.compact_intercept(chart), chart.G @ chart.beta - chart.G @ chart.A @ chart.Q.T @ chart.message_anchor)
    if weight == 1:
        with pytest.raises(ValueError):
            compatibility.verify_lambda_zero(guarded, arm, diagnosis)


def test_runtime_router_has_no_teacher_access_and_serialization_reuses_saved_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = _unit_training_replay()
    arm = compatibility.fit_arm(_TrainingOnly(replay), 1)
    payload = compatibility.serialize_arm(arm)
    # A fitted artifact must be JSON-safe and survive the actual persisted representation.
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    restored = compatibility.load_arm(replay, json.loads(encoded))
    h, m = replay["train_h"], replay["train_m"]
    baseline_assignments = compatibility.assign_runtime(arm, h, m)
    baseline_features = compatibility.discovery_features(arm, m, replay["train_outputs"], replay["train_signatures"])
    restored_features = compatibility.discovery_features(restored, m, replay["train_outputs"], replay["train_signatures"])
    torch.testing.assert_close(restored_features, baseline_features, rtol=0.0, atol=0.0)
    assert torch.equal(compatibility.assign_runtime(restored, h, m), baseline_assignments)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("runtime router accessed a teacher or discovery fit")

    for name in ("normalization_update", "normalization_jvp", "jvp_actions"):
        monkeypatch.setattr(geometry, name, forbidden)
    monkeypatch.setattr(compatibility, "discovery_features", forbidden)
    monkeypatch.setattr(compatibility, "fit_block_transform", forbidden)
    assert torch.equal(compatibility.assign_runtime(restored, h, m), baseline_assignments)
    expected_runtime_features = torch.cat((
        h / replay["saved_artifact"]["h_feature_scale"],
        m / replay["saved_artifact"]["m_feature_scale"],
    ), dim=1)
    torch.testing.assert_close(compatibility.runtime_features(restored, h, m), expected_runtime_features)


def test_lambda_zero_rebuild_matches_the_authenticated_baseline() -> None:
    # This is the original-arm equivalence control; no lambda=1 evaluation occurs.
    replay = load_replay()
    arm = compatibility.fit_arm(replay, 0)
    assert arm["charts"] is not replay["banks"]["response_charts"]
    assert all(new is not old for new, old in zip(arm["charts"], replay["banks"]["response_charts"]))
    receipt = compatibility.verify_lambda_zero(replay, arm, diagnosis)
    assert receipt["status"] == "reproduced"


def test_lambda_zero_verifier_accepts_one_consistent_permutation_of_every_chart_object() -> None:
    replay = load_replay()
    arm = compatibility.fit_arm(replay, 0)
    order = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(3)
    permuted = dict(arm)
    for key in ("discovery_centers", "router_centers", "chart_intercepts"):
        permuted[key] = arm[key][order]
    permuted["charts"] = [arm["charts"][i] for i in order.tolist()]
    permuted["anchor_indices"] = [arm["anchor_indices"][i] for i in order.tolist()]
    permuted["training_assignments"] = inverse[arm["training_assignments"]]
    receipt = compatibility.verify_lambda_zero(replay, permuted, diagnosis)
    assert receipt["status"] == "reproduced"
    assert receipt["chart_label_permutation_control_to_saved"] == order.tolist()
    assert all(receipt["checks"].values())


def test_lambda_zero_verifier_rejects_inconsistent_permutation_and_changed_coefficients() -> None:
    replay = load_replay()
    arm = compatibility.fit_arm(replay, 0)
    inconsistent = {**arm, "discovery_centers": arm["discovery_centers"][[2, 0, 1]]}
    receipt = compatibility.verify_lambda_zero(replay, inconsistent, diagnosis)
    assert receipt["status"] == "discrepancy"
    assert receipt["checks"]["discovery_centers"] is False

    changed = dict(arm)
    changed["charts"] = [replace(arm["charts"][0], beta=arm["charts"][0].beta + 0.01), *arm["charts"][1:]]
    receipt = compatibility.verify_lambda_zero(replay, changed, diagnosis)
    assert receipt["status"] == "discrepancy"
    assert any(name.endswith(".beta") and not passed for name, passed in receipt["checks"].items())
    assert any("per_example_all_chart_predictions" in name and not passed for name, passed in receipt["checks"].items())
