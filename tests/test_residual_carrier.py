from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph.residual_carrier import (
    DenseResidualCarrierFactory,
    ResidualCarrier,
    ResidualCarrierFactory,
    ResidualCarrierSession,
    ResidualGraphExecutionContext,
)


def _sequence(
    mask: Tensor,
    *,
    positions: Tensor | None = None,
    phase: str = "prefill",
    cache_positions: Tensor | None = None,
    cache_state: object | None = None,
) -> SequenceContext:
    if positions is None:
        positions = torch.arange(
            mask.shape[1],
            dtype=torch.long,
            device=mask.device,
        ).unsqueeze(0).expand(mask.shape[0], -1)
    return SequenceContext(
        query_valid_mask=mask,
        key_valid_mask=mask.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=cache_positions,
        phase=phase,  # type: ignore[arg-type]
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=cache_positions is not None,
        ),
        cache_state=cache_state,
    )


def _context(mask: Tensor | None = None) -> ResidualGraphExecutionContext:
    if mask is None:
        mask = torch.tensor(
            [
                [False, True, True, True],
                [True, True, True, False],
            ]
        )
    return ResidualGraphExecutionContext.bind(_sequence(mask))


def _session(
    initial: Tensor,
    context: ResidualGraphExecutionContext,
    *,
    expected: tuple[str, ...] = ("layer.0.attention", "layer.0.mlp"),
    factory: object | None = None,
) -> ResidualCarrierSession:
    return ResidualCarrierSession.begin(
        initial,
        context,
        expected,
        "model.residual",
        factory=factory,
    )


def test_context_binds_an_independent_contiguous_prefill_snapshot() -> None:
    sequence = _sequence(
        torch.tensor(
            [
                [False, False, True, True],
                [True, True, True, False],
            ]
        ),
        positions=torch.tensor(
            [
                [99, 99, 4, 7],
                [10, 11, 15, 99],
            ]
        ),
    )
    context = ResidualGraphExecutionContext.bind(sequence)

    assert context.batch_size == 2
    assert context.query_length == 4
    assert context.valid_token_count == 5
    assert context.metadata()["packed_segments_supported"] is False
    assert context.query_valid_mask.data_ptr() != (
        sequence.query_valid_mask.data_ptr()
    )
    assert context.logical_positions.data_ptr() != (
        sequence.logical_positions.data_ptr()
    )
    context.assert_unchanged()


def test_context_rejects_cache_decode_mask_drift_and_unsupported_rows() -> None:
    with pytest.raises(ValueError, match="nonempty batch and sequence"):
        ResidualGraphExecutionContext.bind(
            _sequence(torch.ones(0, 2, dtype=torch.bool))
        )
    with pytest.raises(ValueError, match="prefill only"):
        ResidualGraphExecutionContext.bind(
            _sequence(torch.ones(1, 2, dtype=torch.bool), phase="decode")
        )
    with pytest.raises(ValueError, match="cache state"):
        ResidualGraphExecutionContext.bind(
            _sequence(
                torch.ones(1, 2, dtype=torch.bool),
                cache_positions=torch.arange(2),
            )
        )

    unequal_mask = _sequence(torch.ones(1, 3, dtype=torch.bool))
    unequal_mask.key_valid_mask[:, -1] = False
    with pytest.raises(ValueError, match="masks must be equal"):
        ResidualGraphExecutionContext.bind(unequal_mask)

    unequal_positions = _sequence(torch.ones(1, 3, dtype=torch.bool))
    unequal_positions.key_logical_positions[:, -1] = 9
    with pytest.raises(ValueError, match="positions must be equal"):
        ResidualGraphExecutionContext.bind(unequal_positions)

    with pytest.raises(ValueError, match="at least one valid token"):
        ResidualGraphExecutionContext.bind(
            _sequence(torch.zeros(1, 3, dtype=torch.bool))
        )
    with pytest.raises(ValueError, match="gapped padding"):
        ResidualGraphExecutionContext.bind(
            _sequence(torch.tensor([[True, False, True]]))
        )
    with pytest.raises(ValueError, match="segment-reset"):
        ResidualGraphExecutionContext.bind(
            _sequence(
                torch.ones(1, 4, dtype=torch.bool),
                positions=torch.tensor([[0, 1, 0, 1]]),
            )
        )


def test_dense_session_masks_padding_versions_and_finishes_exact_plan() -> None:
    context = _context()
    torch.manual_seed(803)
    initial = torch.randn(2, 4, 3)
    caller_before = initial.clone()
    session = _session(initial, context)

    assert session.active and session.live and session.version == 0
    normalized = session.normalized_view(nn.Identity())
    normalized.add_(1000.0)
    torch.testing.assert_close(session.materialize(), initial, rtol=0, atol=0)

    first_delta = torch.ones_like(initial)
    first_delta[~context.query_valid_mask] = 100_000.0
    first_receipt = session.apply_mutation("layer.0.attention", first_delta)
    first = session.materialize()
    expected_first = initial.clone()
    expected_first[context.query_valid_mask] += 1.0
    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(
        first[~context.query_valid_mask],
        initial[~context.query_valid_mask],
        rtol=0,
        atol=0,
    )
    assert session.version == 1
    assert first_receipt is session.receipts[0]
    assert first_receipt.version_before == 0
    assert first_receipt.version_after == 1
    assert first_receipt.valid_token_count == 6
    assert first_receipt.invalid_token_count == 2

    retained_first = first.clone()
    session.apply_mutation(
        "layer.0.mlp",
        2.0 * torch.ones_like(initial),
        combiner=lambda state, delta: state - delta,
    )
    second = session.materialize()
    expected_second = expected_first.clone()
    expected_second[context.query_valid_mask] -= 2.0
    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)
    torch.testing.assert_close(first, retained_first, rtol=0, atol=0)
    torch.testing.assert_close(initial, caller_before, rtol=0, atol=0)

    carrier = session.finish()
    assert isinstance(carrier, ResidualCarrier)
    assert not session.active and not session.live and session.version == 2
    assert carrier.version == 2
    torch.testing.assert_close(
        carrier.materialize(),
        expected_second,
        rtol=0,
        atol=0,
    )
    assert session.metadata["status"] == "finished"
    assert session.metadata["committed_mutation_count"] == 2


def test_mutation_order_and_backend_validation_are_atomic() -> None:
    context = _context(torch.ones(1, 3, dtype=torch.bool))
    initial = torch.randn(1, 3, 2)
    session = _session(initial, context)
    before = session.materialize()

    with pytest.raises(RuntimeError, match="order mismatch"):
        session.apply_mutation("layer.0.mlp", torch.ones_like(initial))
    assert session.version == 0 and session.receipts == ()
    torch.testing.assert_close(session.materialize(), before, rtol=0, atol=0)

    with pytest.raises(ValueError, match="match carrier shape"):
        session.apply_mutation(
            "layer.0.attention",
            torch.ones(1, 3, 3),
        )
    assert session.version == 0 and session.receipts == ()

    with pytest.raises(ValueError, match="dtype and device"):
        session.apply_mutation(
            "layer.0.attention",
            torch.ones_like(initial, dtype=torch.float64),
        )
    assert session.version == 0 and session.receipts == ()

    with pytest.raises(RuntimeError, match="before every mutation"):
        session.finish()
    assert session.active and session.live

    session.apply_mutation("layer.0.attention", torch.ones_like(initial))
    session.apply_mutation("layer.0.mlp", torch.ones_like(initial))
    with pytest.raises(RuntimeError, match="no remaining mutations"):
        session.apply_mutation("extra", torch.ones_like(initial))
    assert session.version == 2 and len(session.receipts) == 2


def test_combiner_and_normalizer_cannot_mutate_a_prior_carrier_state() -> None:
    context = _context(torch.ones(1, 3, dtype=torch.bool))
    initial = torch.randn(1, 3, 2)
    session = _session(initial, context)

    def destructive_normalizer(state: Tensor) -> Tensor:
        return state.add_(7.0)

    normalized = session.normalized_view(destructive_normalizer)
    torch.testing.assert_close(normalized, initial + 7.0, rtol=0, atol=0)
    torch.testing.assert_close(session.materialize(), initial, rtol=0, atol=0)

    def destructive_combiner(state: Tensor, delta: Tensor) -> Tensor:
        state.mul_(2.0)
        delta.add_(3.0)
        return state + delta

    session.apply_mutation(
        "layer.0.attention",
        torch.ones_like(initial),
        destructive_combiner,
    )
    post = session.materialize()
    torch.testing.assert_close(post, 2.0 * initial + 4.0, rtol=0, atol=0)
    torch.testing.assert_close(initial, session._caller_initial_snapshot)


def test_session_rejects_caller_bound_sequence_and_context_mutation() -> None:
    mask = torch.ones(1, 3, dtype=torch.bool)
    sequence = _sequence(mask)
    context = ResidualGraphExecutionContext.bind(sequence)
    initial = torch.randn(1, 3, 2)
    session = _session(initial, context)
    initial.add_(1.0)
    with pytest.raises(RuntimeError, match="caller initial_state changed"):
        session.materialize()
    session.abort()

    context = ResidualGraphExecutionContext.bind(sequence)
    session = _session(torch.randn(1, 3, 2), context)
    context.logical_positions[:, -1] = 99
    with pytest.raises(RuntimeError, match="context was mutated"):
        session.normalized_view(nn.Identity())
    session.abort()

    sequence = _sequence(mask)
    context = ResidualGraphExecutionContext.bind(sequence)
    session = _session(torch.randn(1, 3, 2), context)
    sequence.logical_positions[:, -1] = 99
    with pytest.raises(RuntimeError, match="bound SequenceContext changed"):
        session.materialize()
    session.abort()


def test_abort_discards_live_state_and_is_idempotent() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    session = _session(torch.randn(1, 2, 3), context)
    session.apply_mutation(
        "layer.0.attention",
        torch.ones(1, 2, 3),
    )
    session.abort()
    session.abort()

    assert not session.active and not session.live
    assert session.version == 1
    assert session.metadata["status"] == "aborted"
    with pytest.raises(RuntimeError, match="not active"):
        session.materialize()
    with pytest.raises(RuntimeError, match="not active"):
        session.finish()


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Size, str]] = []

    def create(
        self,
        initial_state: Tensor,
        context: ResidualGraphExecutionContext,
        *,
        wire_id: str,
    ) -> ResidualCarrier:
        self.calls.append((initial_state.shape, wire_id))
        return DenseResidualCarrierFactory().create(
            initial_state,
            context,
            wire_id=wire_id,
        )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_factory_seam_and_metadata_are_representation_neutral() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    initial = torch.randn(1, 2, 3)
    factory = _RecordingFactory()
    session = _session(initial, context, factory=factory)

    assert factory.calls == [(initial.shape, "model.residual")]
    session.apply_mutation("layer.0.attention", torch.ones_like(initial))
    session.apply_mutation("layer.0.mlp", torch.ones_like(initial))
    carrier = session.finish()
    metadata = dict(carrier.metadata())

    assert metadata["backend_kind"] == "dense_exact_v1"
    assert metadata["learned_parameter_count"] == 0
    assert metadata["residual_state_scalar_count"] == initial.numel()
    assert metadata["padding_baseline_scalar_count"] == initial.numel()
    assert metadata["dynamic_state_scalar_count"] == 2 * initial.numel()
    assert not _contains_tensor(metadata)
    assert not _contains_tensor(session.metadata)


class _MaterializationSpyCarrier:
    def __init__(self, delegate: ResidualCarrier, calls: list[int]) -> None:
        self.delegate = delegate
        self.calls = calls

    @property
    def backend_kind(self) -> str:
        return self.delegate.backend_kind

    @property
    def wire_id(self) -> str:
        return self.delegate.wire_id

    @property
    def shape(self) -> torch.Size:
        return self.delegate.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.delegate.dtype

    @property
    def device(self) -> torch.device:
        return self.delegate.device

    @property
    def version(self) -> int:
        return self.delegate.version

    def normalized_view(self, normalizer: object) -> Tensor:
        return self.delegate.normalized_view(normalizer)  # type: ignore[arg-type]

    def apply_mutation(
        self,
        delta: Tensor,
        combiner: object = None,
    ) -> ResidualCarrier:
        return _MaterializationSpyCarrier(
            self.delegate.apply_mutation(  # type: ignore[arg-type]
                delta,
                combiner,
            ),
            self.calls,
        )

    def materialize(self) -> Tensor:
        self.calls.append(self.version)
        return self.delegate.materialize()

    def metadata(self) -> Mapping[str, object]:
        return self.delegate.metadata()


class _MaterializationSpyFactory:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def create(
        self,
        initial_state: Tensor,
        context: ResidualGraphExecutionContext,
        *,
        wire_id: str,
    ) -> ResidualCarrier:
        return _MaterializationSpyCarrier(
            DenseResidualCarrierFactory().create(
                initial_state,
                context,
                wire_id=wire_id,
            ),
            self.calls,
        )


def test_mutations_commit_without_implicit_dense_materialization() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    initial = torch.randn(1, 2, 3)
    factory = _MaterializationSpyFactory()
    assert isinstance(factory, ResidualCarrierFactory)
    session = _session(initial, context, factory=factory)

    receipt = session.apply_mutation(
        "layer.0.attention",
        torch.ones_like(initial),
    )
    assert receipt.version_after == 1
    assert factory.calls == []
    torch.testing.assert_close(
        session.materialize(),
        initial + 1,
        rtol=0,
        atol=0,
    )
    assert factory.calls == [1]

    session.apply_mutation("layer.0.mlp", torch.ones_like(initial))
    assert factory.calls == [1]
    session.finish()


class _InvalidMaterializationCarrier(_MaterializationSpyCarrier):
    def apply_mutation(
        self,
        delta: Tensor,
        combiner: object = None,
    ) -> ResidualCarrier:
        return _InvalidMaterializationCarrier(
            self.delegate.apply_mutation(  # type: ignore[arg-type]
                delta,
                combiner,
            ),
            self.calls,
        )

    def materialize(self) -> Tensor:
        return torch.zeros((), dtype=self.dtype, device=self.device)


class _InvalidMaterializationFactory(_MaterializationSpyFactory):
    def create(
        self,
        initial_state: Tensor,
        context: ResidualGraphExecutionContext,
        *,
        wire_id: str,
    ) -> ResidualCarrier:
        return _InvalidMaterializationCarrier(
            DenseResidualCarrierFactory().create(
                initial_state,
                context,
                wire_id=wire_id,
            ),
            self.calls,
        )


def test_explicit_materialization_is_validated_and_fails_closed() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    initial = torch.randn(1, 2, 3)
    session = _session(
        initial,
        context,
        factory=_InvalidMaterializationFactory(),
    )
    session.apply_mutation("layer.0.attention", torch.ones_like(initial))

    with pytest.raises(ValueError, match="preserve carrier shape"):
        session.materialize()
    assert not session.active and not session.live
    assert session.metadata["status"] == "aborted"


class _InPlaceCarrier(_MaterializationSpyCarrier):
    def apply_mutation(
        self,
        delta: Tensor,
        combiner: object = None,
    ) -> ResidualCarrier:
        self.delegate = self.delegate.apply_mutation(  # type: ignore[arg-type]
            delta,
            combiner,
        )
        return self


class _InPlaceFactory(_MaterializationSpyFactory):
    def create(
        self,
        initial_state: Tensor,
        context: ResidualGraphExecutionContext,
        *,
        wire_id: str,
    ) -> ResidualCarrier:
        return _InPlaceCarrier(
            DenseResidualCarrierFactory().create(
                initial_state,
                context,
                wire_id=wire_id,
            ),
            self.calls,
        )


def test_in_place_backend_mutation_is_rejected_and_session_fails_closed() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    initial = torch.randn(1, 2, 3)
    session = _session(initial, context, factory=_InPlaceFactory())

    with pytest.raises(RuntimeError, match="distinct persistent value"):
        session.apply_mutation(
            "layer.0.attention",
            torch.ones_like(initial),
        )
    assert session.version == 0 and session.receipts == ()
    assert not session.active and not session.live


def test_factory_and_callback_input_mutation_are_rejected_fail_closed() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    initial = torch.zeros(1, 2, 3)

    class MutatingFactory:
        def create(
            self,
            initial_state: Tensor,
            bound_context: ResidualGraphExecutionContext,
            *,
            wire_id: str,
        ) -> ResidualCarrier:
            initial_state.add_(1)
            return DenseResidualCarrierFactory().create(
                initial_state,
                bound_context,
                wire_id=wire_id,
            )

    with pytest.raises(RuntimeError, match="factory changed caller"):
        _session(initial, context, factory=MutatingFactory())

    initial = torch.zeros(1, 2, 3)
    session = _session(initial, context)

    def mutating_combiner(state: Tensor, delta: Tensor) -> Tensor:
        initial.add_(5)
        return state + delta

    with pytest.raises(RuntimeError, match="caller initial_state changed"):
        session.apply_mutation(
            "layer.0.attention",
            torch.ones_like(initial),
            mutating_combiner,
        )
    assert session.version == 0 and session.receipts == ()
    assert not session.active and not session.live


def test_closed_sessions_release_tensor_and_context_references() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    initial = torch.randn(1, 2, 3)
    session = _session(initial, context)
    session.apply_mutation("layer.0.attention", torch.ones_like(initial))
    session.apply_mutation("layer.0.mlp", torch.ones_like(initial))
    session.finish()

    assert session._caller_initial_reference is None
    assert session._caller_initial_snapshot is None
    assert session._context is None
    assert session.metadata["context"]["valid_token_count"] == 2

    aborted = _session(initial, context)
    aborted.abort()
    assert aborted._caller_initial_reference is None
    assert aborted._caller_initial_snapshot is None
    assert aborted._context is None
    assert aborted.metadata["status"] == "aborted"


def test_nonfinite_initial_values_are_not_false_mutation_positives() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    initial = torch.tensor([[[float("nan")], [float("inf")]]])
    session = _session(initial, context)
    torch.testing.assert_close(
        session.materialize(),
        initial,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    session.abort()


def test_invalid_session_plan_and_finished_lifecycle_are_rejected() -> None:
    context = _context(torch.ones(1, 2, dtype=torch.bool))
    initial = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="requires expected mutations"):
        _session(initial, context, expected=())
    with pytest.raises(ValueError, match="must be unique"):
        _session(initial, context, expected=("same", "same"))
    with pytest.raises(TypeError, match="sequence of strings"):
        ResidualCarrierSession.begin(
            initial,
            context,
            "not-a-sequence-of-ids",  # type: ignore[arg-type]
            "model.residual",
        )

    session = _session(initial, context)
    session.apply_mutation("layer.0.attention", torch.ones_like(initial))
    session.apply_mutation("layer.0.mlp", torch.ones_like(initial))
    session.finish()
    with pytest.raises(RuntimeError, match="finished.*cannot abort"):
        session.abort()
