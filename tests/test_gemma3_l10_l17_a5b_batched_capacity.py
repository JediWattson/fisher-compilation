from __future__ import annotations

import pytest
import torch

import fisher_graph.gemma3_l10_l17_a5_frozen_affine_capacity_oracle as a5a
import fisher_graph.gemma3_l10_l17_a5b_batched_capacity as a5b
from fisher_graph.downstream_affine_coordinate_solver import (
    DownstreamAffineSolverConfig,
)


class _NonlinearTokenLocalHead:
    def __init__(self) -> None:
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def project_logits(self, hidden_states, sequence, *, trace=None):
        del trace
        self.calls += 1
        self.shapes.append(tuple(hidden_states.shape))
        assert hidden_states.shape[:2] == (
            sequence.batch_size,
            sequence.query_length,
        )
        score = hidden_states[..., 0] + hidden_states[..., -1]
        return torch.stack(
            (score, -score, 0.25 * score.square()), dim=-1
        )


class _CrossTokenHead:
    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        shared = hidden_states[..., -1].mean(dim=1, keepdim=True)
        score = hidden_states[..., 0] + shared
        return torch.stack((score, -score, score.square()), dim=-1)


class _SharedQueryHead:
    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        score = hidden_states[..., 0].mean(dim=1, keepdim=True)
        return torch.stack((score, -score, score.square()), dim=-1)


class _BatchShapeSensitiveTokenLocalHead:
    """Token-local semantics with a deterministic GEMM-vs-GEMV-sized offset."""

    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        score = hidden_states[..., 0] + hidden_states[..., -1]
        if hidden_states.shape[1] > 1:
            score = score + 5.0e-5
        return torch.stack((score, -score, 0.25 * score.square()), dim=-1)


class _DirectedPositionHead:
    """Only output row two reads source row zero."""

    def project_logits(self, hidden_states, sequence, *, trace=None):
        del sequence, trace
        score = hidden_states[..., 0].clone()
        score[:, 2] = score[:, 2] + 0.75 * hidden_states[:, 0, -1]
        return torch.stack((score, -score, 0.5 * score.square()), dim=-1)


def _image() -> a5a.FrozenAffineImage:
    width = 183
    decoder = torch.zeros((182, width), dtype=torch.float64)
    decoder[:, :182] = torch.eye(182, dtype=torch.float64)
    return a5a.FrozenAffineImage(
        node_order=("a", "b", "c", "d"),
        rank_by_node=(46, 46, 45, 45),
        mean_sum=torch.zeros(width, dtype=torch.float64),
        decoder=decoder,
        basis_sha256_by_node=tuple(f"{index + 1:064x}" for index in range(4)),
        mean_sha256_by_node=tuple(
            f"{index + 10:064x}" for index in range(4)
        ),
        decoder_sha256_by_node=tuple(
            f"{index + 20:064x}" for index in range(4)
        ),
    )


def _inputs(row_count: int = 4) -> dict[str, torch.Tensor]:
    if row_count > 4:
        raise ValueError("fixture has four rows")
    image = _image()
    native = torch.zeros((4, 183), dtype=torch.float32)
    native[:, 0] = torch.tensor((0.40, -0.60, 0.80, -0.30))
    native[:, -1] = torch.tensor((1.00, -0.75, 0.50, -1.20))
    native = native[:row_count].contiguous()
    target = native.double().contiguous()
    initial = image.euclidean_initial_coordinates(target)
    a4_baseline = (image.mean_sum + initial @ image.decoder).float()
    zeros = torch.zeros_like(native)
    return {
        "native_state": native,
        "compiled_post_attention_residual": zeros.clone(),
        "compiled_compact_retained_delta": zeros.clone(),
        "target_correction": target,
        "a4_float64_projection_correction": a4_baseline,
    }


def _config() -> DownstreamAffineSolverConfig:
    return DownstreamAffineSolverConfig(
        steps=48,
        learning_rate=0.35,
        ridge=0.017,
        trust_radius=0.75,
    )


def test_batched_wrapper_matches_a5a_serial_row_solver() -> None:
    inputs = _inputs()
    serial_adapter = _NonlinearTokenLocalHead()
    batched_adapter = _NonlinearTokenLocalHead()
    serial = a5a.solve_frozen_affine_capacity_rows(
        adapter=serial_adapter,  # type: ignore[arg-type]
        image=_image(),
        **inputs,
        row_chunk_size=1,
        solver_config=_config(),
    )
    batched = a5b.solve_batched_frozen_affine_capacity_rows(
        adapter=batched_adapter,  # type: ignore[arg-type]
        image=_image(),
        **inputs,
        row_chunk_size=4,
        solver_config=_config(),
    )

    torch.testing.assert_close(
        batched.selected_coefficients,
        serial.selected_coefficients,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert torch.equal(batched.initial_coefficients, serial.initial_coefficients)
    assert torch.equal(batched.initial_correction, serial.initial_correction)
    assert torch.equal(batched.initial_state, serial.initial_state)
    assert torch.equal(batched.selected_correction, serial.selected_correction)
    assert torch.equal(batched.selected_state, serial.selected_state)
    assert batched.selected_coefficients.dtype == torch.float64
    assert batched.selected_correction.dtype == torch.float32
    assert batched.selected_state.dtype == torch.float32
    assert batched.receipt["throughput_change_only"] is True
    assert batched.receipt[
        "selected_not_worse_than_initial_for_every_token"
    ] is True
    assert batched.receipt["initial_kl"]["mean"] == pytest.approx(
        serial.receipt["initial_kl_per_row"], abs=1.0e-15
    )
    assert batched.receipt["selected_kl"]["mean"] == pytest.approx(
        serial.receipt["selected_kl_per_row"], abs=1.0e-15
    )
    assert batched_adapter.calls < serial_adapter.calls

    def assert_tensor_free(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for child in value.values():
                assert_tensor_free(child)
        elif isinstance(value, list):
            for child in value:
                assert_tensor_free(child)

    assert_tensor_free(batched.receipt)
    assert batched.receipt["contains_tensor_payloads"] is False


def test_batch_size_and_row_permutation_do_not_change_token_solutions() -> None:
    inputs = _inputs()
    image = _image()

    def solve(
        values: dict[str, torch.Tensor], chunk_size: int
    ) -> a5a.FrozenAffineCapacitySolution:
        return a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
            image=image,
            **values,
            row_chunk_size=chunk_size,
            solver_config=_config(),
        )

    batch_one = solve(inputs, 1)
    batch_two = solve(inputs, 2)
    batch_four = solve(inputs, 4)
    for candidate in (batch_two, batch_four):
        torch.testing.assert_close(
            candidate.selected_coefficients,
            batch_one.selected_coefficients,
            rtol=0.0,
            atol=1.0e-15,
        )
        assert torch.equal(
            candidate.selected_correction, batch_one.selected_correction
        )
        assert torch.equal(candidate.selected_state, batch_one.selected_state)

    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    permuted_inputs = {
        name: value[permutation].contiguous()
        for name, value in inputs.items()
    }
    permuted = solve(permuted_inputs, 3)
    torch.testing.assert_close(
        permuted.selected_coefficients[inverse],
        batch_one.selected_coefficients,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert torch.equal(
        permuted.selected_correction[inverse], batch_one.selected_correction
    )
    assert torch.equal(permuted.selected_state[inverse], batch_one.selected_state)


def test_a4_float64_projection_then_one_cast_is_an_exact_lock() -> None:
    width = 183
    decoder = torch.zeros((182, width), dtype=torch.float64)
    decoder[:, :182] = torch.eye(182, dtype=torch.float64)
    decoder[:, -1] = torch.linspace(-3.0, 5.0, 182, dtype=torch.float64)
    image = a5a.FrozenAffineImage(
        node_order=("a", "b", "c", "d"),
        rank_by_node=(46, 46, 45, 45),
        mean_sum=torch.linspace(-100.0, 100.0, width, dtype=torch.float64),
        decoder=decoder,
        basis_sha256_by_node=tuple(f"{index + 1:064x}" for index in range(4)),
        mean_sha256_by_node=tuple(
            f"{index + 10:064x}" for index in range(4)
        ),
        decoder_sha256_by_node=tuple(
            f"{index + 20:064x}" for index in range(4)
        ),
    )
    generator = torch.Generator().manual_seed(7)
    target = 1_000.0 * torch.randn(
        (2, width), generator=generator, dtype=torch.float64
    )
    coordinates = image.euclidean_initial_coordinates(target)
    a4_baseline = (image.mean_sum + coordinates @ image.decoder).float()
    cast_before_matmul = (
        image.mean_sum.float()
        + coordinates.float() @ image.decoder.float()
    )
    assert not torch.equal(a4_baseline, cast_before_matmul)
    zeros = torch.zeros_like(a4_baseline)

    adapter = _NonlinearTokenLocalHead()
    solution = a5b.solve_batched_frozen_affine_capacity_rows(
        adapter=adapter,  # type: ignore[arg-type]
        image=image,
        native_state=a4_baseline,
        compiled_post_attention_residual=zeros,
        compiled_compact_retained_delta=zeros,
        target_correction=target,
        a4_float64_projection_correction=a4_baseline,
        row_chunk_size=2,
        solver_config=DownstreamAffineSolverConfig(
            steps=1, learning_rate=0.01, ridge=0.0, trust_radius=None
        ),
    )
    assert torch.equal(solution.initial_correction, a4_baseline)
    assert torch.equal(solution.initial_state, a4_baseline)
    assert solution.receipt[
        "initial_correction_bit_identical_to_a4_float64_one_cast"
    ] is True

    wrong = a4_baseline.clone()
    wrong[0, 0] = torch.nextafter(
        wrong[0, 0], torch.tensor(float("inf"), dtype=wrong.dtype)
    )
    with pytest.raises(RuntimeError, match="differs from A4"):
        a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
            image=image,
            native_state=a4_baseline,
            compiled_post_attention_residual=zeros,
            compiled_compact_retained_delta=zeros,
            target_correction=target,
            a4_float64_projection_correction=wrong,
            row_chunk_size=2,
            solver_config=DownstreamAffineSolverConfig(steps=1),
        )


def test_wrapper_rejects_non_token_local_or_shared_query_heads() -> None:
    inputs = _inputs(2)
    common = {
        "image": _image(),
        **inputs,
        "row_chunk_size": 2,
        "solver_config": DownstreamAffineSolverConfig(steps=1),
    }
    with pytest.raises(RuntimeError, match="not token-local"):
        a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=_CrossTokenHead(),  # type: ignore[arg-type]
            **common,
        )
    with pytest.raises(RuntimeError, match=r"\[1, rows, vocab\]"):
        a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=_SharedQueryHead(),  # type: ignore[arg-type]
            **common,
        )


def test_full_envelope_separates_shape_drift_from_cross_row_dependence() -> None:
    inputs = _inputs()
    native = inputs["native_state"]
    a4_state = (
        inputs["compiled_post_attention_residual"]
        + inputs["compiled_compact_retained_delta"]
        + inputs["a4_float64_projection_correction"]
    ).contiguous()

    shape_only = a5b.diagnose_token_locality_envelope(
        adapter=_BatchShapeSensitiveTokenLocalHead(),  # type: ignore[arg-type]
        native_state=native,
        a4_compiled_state=a4_state,
        row_chunk_size=2,
    )
    assert shape_only["chunk_count"] == 2
    assert shape_only["canonical_singleton_guard_passed"] is False
    assert shape_only["same_shape_counterfactual_passed"] is True
    counterfactual = shape_only["aggregate"][
        "same_shape_peer_counterfactual"
    ]
    assert counterfactual["native_teacher"]["checked_row_count"] == 4
    assert counterfactual["a4_compiled"]["checked_row_count"] == 4

    cross_row = a5b.diagnose_token_locality_envelope(
        adapter=_CrossTokenHead(),  # type: ignore[arg-type]
        native_state=native,
        a4_compiled_state=a4_state,
        row_chunk_size=2,
    )
    assert cross_row["same_shape_counterfactual_passed"] is False
    assert cross_row["aggregate"]["same_shape_peer_counterfactual"][
        "native_teacher"
    ]["failing_chunk_count"] > 0

    def assert_tensor_free(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for child in value.values():
                assert_tensor_free(child)
        elif isinstance(value, list):
            for child in value:
                assert_tensor_free(child)

    assert_tensor_free(shape_only)
    assert shape_only["changes_solver_authorization"] is False


def test_full_envelope_covers_every_chunk_without_early_exit() -> None:
    inputs = _inputs()
    diagnostic = a5b.diagnose_token_locality_envelope(
        adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
        native_state=inputs["native_state"],
        a4_compiled_state=inputs["a4_float64_projection_correction"],
        row_chunk_size=3,
    )

    assert diagnostic["row_count"] == 4
    assert diagnostic["chunk_count"] == 2
    assert [chunk["row_count"] for chunk in diagnostic["chunks"]] == [3, 1]
    assert diagnostic["canonical_singleton_guard_passed"] is True
    assert diagnostic["same_shape_counterfactual_passed"] is True
    counterfactual = diagnostic["aggregate"][
        "same_shape_peer_counterfactual"
    ]
    assert counterfactual["native_teacher"][
        "uncheckable_singleton_row_count"
    ] == 1


def test_wrapper_rejects_noncanonical_target_and_invalid_batch_size() -> None:
    inputs = _inputs(2)
    bad_target = dict(inputs)
    bad_target["target_correction"] = inputs["target_correction"].float()
    with pytest.raises(ValueError, match="canonical CPU float64"):
        a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
            image=_image(),
            **bad_target,
            row_chunk_size=2,
            solver_config=DownstreamAffineSolverConfig(steps=1),
        )
    with pytest.raises(ValueError, match="positive integer"):
        a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
            image=_image(),
            **inputs,
            row_chunk_size=0,
            solver_config=DownstreamAffineSolverConfig(steps=1),
        )


def test_historical_default_keeps_singleton_locality_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_directed_audit(**_: object) -> object:
        raise AssertionError("default policy entered directed audit")

    monkeypatch.setattr(
        a5b, "audit_same_shape_off_row_locality", forbidden_directed_audit
    )
    solution = a5b.solve_batched_frozen_affine_capacity_rows(
        adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
        image=_image(),
        **_inputs(2),
        row_chunk_size=2,
        solver_config=DownstreamAffineSolverConfig(steps=1),
    )

    assert "token_locality_policy" not in solution.receipt["batching"]
    locality = solution.receipt["chunk_receipts"][0]["token_locality"]
    assert locality["method"] == (
        "batched_projection_vs_same_rows_projected_as_singletons"
    )


@pytest.mark.parametrize(
    "adapter",
    (_CrossTokenHead(), _DirectedPositionHead()),
    ids=("mean_coupling", "position_specific_coupling"),
)
def test_directed_policy_rejects_mean_and_position_specific_coupling(
    adapter: object,
) -> None:
    with pytest.raises(RuntimeError, match="same-shape directed off-row"):
        a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=adapter,  # type: ignore[arg-type]
            image=_image(),
            **_inputs(4),
            row_chunk_size=4,
            token_locality_policy=(
                a5b.TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW
            ),
            solver_config=DownstreamAffineSolverConfig(steps=1),
        )


def test_directed_policy_fails_closed_on_singleton_chunk() -> None:
    with pytest.raises(RuntimeError, match="fails closed.*singleton"):
        a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
            image=_image(),
            **_inputs(1),
            row_chunk_size=1,
            token_locality_policy=(
                a5b.TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW
            ),
            solver_config=DownstreamAffineSolverConfig(steps=1),
        )


def test_directed_receipt_is_tensor_free_and_uses_canonical_a4_grouping() -> None:
    image = _image()
    row_count = 2
    zero = torch.zeros((row_count, image.residual_width), dtype=torch.float32)
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(float("inf"), dtype=torch.float32),
    )
    target = torch.zeros_like(zero, dtype=torch.float64)
    target[:, 0] = float(smallest.item())
    initial64 = image.euclidean_initial_coordinates(target)
    a4_correction = (image.mean_sum + initial64 @ image.decoder).float()
    post_attention = torch.ones_like(zero)
    retained_delta = -torch.ones_like(zero)
    canonical_state = (
        post_attention + (retained_delta + a4_correction)
    ).contiguous()
    regrouped_state = (
        (post_attention + retained_delta) + a4_correction
    ).contiguous()
    assert not torch.equal(canonical_state, regrouped_state)

    adapter = _NonlinearTokenLocalHead()
    solution = a5b.solve_batched_frozen_affine_capacity_rows(
        adapter=adapter,  # type: ignore[arg-type]
        image=image,
        native_state=canonical_state,
        compiled_post_attention_residual=post_attention,
        compiled_compact_retained_delta=retained_delta,
        target_correction=target,
        a4_float64_projection_correction=a4_correction,
        row_chunk_size=row_count,
        token_locality_policy=(
            a5b.TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW
        ),
        solver_config=DownstreamAffineSolverConfig(
            steps=1, learning_rate=0.01, ridge=0.0, trust_radius=None
        ),
    )

    locality = solution.receipt["chunk_receipts"][0]["token_locality"]
    assert set(locality) == {
        "policy",
        "method",
        "probe_states",
        "row_count",
        "nontrivial_multirow_probe",
        "absolute_tolerance",
        "relative_tolerance",
        "teacher",
        "a4_baseline",
        "directed_receipt_sha256_by_probe",
        "native_teacher_baseline_logits_reused_by_solver",
        "changes_solver_authorization",
        "passed",
    }
    assert locality["policy"] == (
        a5b.TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW
    )
    assert locality["changes_solver_authorization"] is True
    assert locality["native_teacher_baseline_logits_reused_by_solver"] is True
    assert locality["teacher"]["scientific_role"] == "solver_authorization"
    assert locality["a4_baseline"]["scientific_role"] == "solver_authorization"
    assert locality["a4_baseline"]["input_rows_sha256"] == (
        a5a._tensor_sha256(canonical_state)
    )
    assert locality["a4_baseline"]["input_rows_sha256"] != (
        a5a._tensor_sha256(regrouped_state)
    )
    assert locality["directed_receipt_sha256_by_probe"] == {
        "teacher": locality["teacher"]["receipt_sha256"],
        "a4_baseline": locality["a4_baseline"]["receipt_sha256"],
    }
    directed_projection_count = 2 * (row_count + 1)
    assert locality["teacher"]["projection_call_count"] == row_count + 1
    assert locality["a4_baseline"]["projection_call_count"] == row_count + 1
    assert adapter.shapes[:directed_projection_count] == [
        (1, row_count, image.residual_width)
    ] * directed_projection_count
    assert (1, 1, image.residual_width) not in adapter.shapes

    def assert_tensor_free(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for child in value.values():
                assert_tensor_free(child)
        elif isinstance(value, list):
            for child in value:
                assert_tensor_free(child)

    assert_tensor_free(locality)


def test_directed_policy_rejects_drifted_reused_native_baseline_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_audit = a5b.audit_same_shape_off_row_locality

    def drifted_audit(**kwargs: object) -> object:
        result = real_audit(**kwargs)  # type: ignore[arg-type]
        if kwargs["probe_name"] != "native_teacher":
            return result
        drifted = result.baseline_logits.clone()
        drifted[0, 0] = torch.nextafter(
            drifted[0, 0],
            torch.tensor(float("inf"), dtype=drifted.dtype),
        )
        return type(result)(baseline_logits=drifted, receipt=result.receipt)

    monkeypatch.setattr(a5b, "audit_same_shape_off_row_locality", drifted_audit)
    with pytest.raises(RuntimeError, match="baseline tensor hash mismatch"):
        a5b.solve_batched_frozen_affine_capacity_rows(
            adapter=_NonlinearTokenLocalHead(),  # type: ignore[arg-type]
            image=_image(),
            **_inputs(2),
            row_chunk_size=2,
            token_locality_policy=(
                a5b.TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW
            ),
            solver_config=DownstreamAffineSolverConfig(steps=1),
        )
