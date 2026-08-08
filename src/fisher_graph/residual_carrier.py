"""Exact residual-state wire primitives for compiled transformer graphs.

The dense implementation is deliberately conservative.  Each model forward
owns one transactional session, each mutation creates a new carrier state,
and invalid token rows remain equal to the incoming residual state.  The
public protocol is representation-neutral so a modal carrier can later sit
behind the same stack executor without changing its control flow.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from .adapters import SequenceContext


StateNormalizer = Callable[[Tensor], Tensor]
StateCombiner = Callable[[Tensor, Tensor], Tensor]


def _dtype_name(value: torch.dtype) -> str:
    return str(value).removeprefix("torch.")


def _require_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _same_tensor(value: object, expected: Tensor) -> bool:
    if (
        not isinstance(value, Tensor)
        or value.shape != expected.shape
        or value.dtype != expected.dtype
        or value.device != expected.device
    ):
        return False
    if value.is_floating_point() or value.is_complex():
        return torch.allclose(
            value,
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    return torch.equal(value, expected)


@dataclass(frozen=True, slots=True, eq=False)
class ResidualGraphExecutionContext:
    """Validated, integrity-checked sequence state for one residual wire."""

    query_valid_mask: Tensor
    key_valid_mask: Tensor
    logical_positions: Tensor
    key_logical_positions: Tensor
    attention_mask_supplied: bool
    position_ids_supplied: bool
    _snapshots: tuple[Tensor, Tensor, Tensor, Tensor] = field(
        repr=False,
        compare=False,
    )
    _source_sequence: SequenceContext = field(repr=False, compare=False)
    _source_snapshots: tuple[Tensor, Tensor, Tensor, Tensor] = field(
        repr=False,
        compare=False,
    )

    @classmethod
    def bind(
        cls,
        sequence: SequenceContext,
    ) -> ResidualGraphExecutionContext:
        """Validate and snapshot a cache-free, non-packed prefill context."""

        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill":
            raise ValueError("residual graph execution supports prefill only")
        if sequence.batch_size <= 0 or sequence.query_length <= 0:
            raise ValueError(
                "residual graph execution requires a nonempty batch and sequence"
            )
        if (
            sequence.cache_state is not None
            or sequence.cache_positions is not None
            or sequence.input_origin.cache_positions_supplied
        ):
            raise ValueError(
                "residual graph execution does not support cache state"
            )
        if not torch.equal(
            sequence.query_valid_mask,
            sequence.key_valid_mask,
        ):
            raise ValueError("residual graph query/key masks must be equal")
        if not torch.equal(
            sequence.logical_positions,
            sequence.key_logical_positions,
        ):
            raise ValueError(
                "residual graph query/key logical positions must be equal"
            )

        mask = sequence.query_valid_mask
        positions = sequence.logical_positions
        for row_index in range(sequence.batch_size):
            valid_indices = torch.nonzero(
                mask[row_index],
                as_tuple=False,
            ).flatten()
            if valid_indices.numel() == 0:
                raise ValueError(
                    "residual graph rows must contain at least one valid token"
                )
            first = int(valid_indices[0].item())
            last = int(valid_indices[-1].item())
            if int(valid_indices.numel()) != last - first + 1:
                raise ValueError(
                    "residual graph does not support gapped padding"
                )
            valid_positions = positions[row_index].index_select(
                0,
                valid_indices,
            )
            if (
                valid_positions.numel() > 1
                and bool((valid_positions[1:] <= valid_positions[:-1]).any())
            ):
                raise ValueError(
                    "residual graph does not support segment-reset positions"
                )

        source_values = (
            sequence.query_valid_mask,
            sequence.key_valid_mask,
            sequence.logical_positions,
            sequence.key_logical_positions,
        )
        values = tuple(value.detach().clone() for value in source_values)
        snapshots = tuple(value.clone() for value in values)
        source_snapshots = tuple(
            value.detach().clone() for value in source_values
        )
        return cls(
            query_valid_mask=values[0],
            key_valid_mask=values[1],
            logical_positions=values[2],
            key_logical_positions=values[3],
            attention_mask_supplied=(
                sequence.input_origin.attention_mask_supplied
            ),
            position_ids_supplied=(
                sequence.input_origin.position_ids_supplied
            ),
            _snapshots=snapshots,  # type: ignore[arg-type]
            _source_sequence=sequence,
            _source_snapshots=source_snapshots,  # type: ignore[arg-type]
        )

    @property
    def batch_size(self) -> int:
        return int(self.query_valid_mask.shape[0])

    @property
    def query_length(self) -> int:
        return int(self.query_valid_mask.shape[1])

    @property
    def valid_token_count(self) -> int:
        return int(self.query_valid_mask.sum().item())

    @property
    def device(self) -> torch.device:
        return self.query_valid_mask.device

    def assert_unchanged(self) -> None:
        """Reject in-place changes to this snapshot or its bound source."""

        current = (
            self.query_valid_mask,
            self.key_valid_mask,
            self.logical_positions,
            self.key_logical_positions,
        )
        if any(
            not _same_tensor(value, snapshot)
            for value, snapshot in zip(
                current,
                self._snapshots,
                strict=True,
            )
        ):
            raise RuntimeError("residual graph execution context was mutated")

        source = self._source_sequence
        source_current = (
            source.query_valid_mask,
            source.key_valid_mask,
            source.logical_positions,
            source.key_logical_positions,
        )
        if (
            source.phase != "prefill"
            or source.cache_state is not None
            or source.cache_positions is not None
            or source.input_origin.cache_positions_supplied
            or source.input_origin.attention_mask_supplied
            is not self.attention_mask_supplied
            or source.input_origin.position_ids_supplied
            is not self.position_ids_supplied
            or any(
                not _same_tensor(value, snapshot)
                for value, snapshot in zip(
                    source_current,
                    self._source_snapshots,
                    strict=True,
                )
            )
        ):
            raise RuntimeError(
                "bound SequenceContext changed during residual execution"
            )

    def metadata(self) -> dict[str, object]:
        self.assert_unchanged()
        return {
            "schema": "fisher_graph.residual_graph_execution_context.v1",
            "phase": "prefill",
            "cache_supported": False,
            "batch_size": self.batch_size,
            "query_length": self.query_length,
            "valid_token_count": self.valid_token_count,
            "padding": "contiguous_left_right_or_none",
            "packed_segments_supported": False,
            "attention_mask_supplied": self.attention_mask_supplied,
            "position_ids_supplied": self.position_ids_supplied,
            "device": str(self.device),
        }


@dataclass(frozen=True, slots=True)
class ResidualMutationReceipt:
    """One ordered mutation committed to a residual carrier session."""

    wire_id: str
    mutation_id: str
    version_before: int
    version_after: int
    valid_token_count: int
    invalid_token_count: int
    combiner_kind: str

    def __post_init__(self) -> None:
        _require_identifier(self.wire_id, label="wire_id")
        _require_identifier(self.mutation_id, label="mutation_id")
        _require_identifier(self.combiner_kind, label="combiner_kind")
        if (
            type(self.version_before) is not int
            or self.version_before < 0
            or self.version_after != self.version_before + 1
        ):
            raise ValueError("residual mutation receipt versions are invalid")
        if (
            type(self.valid_token_count) is not int
            or type(self.invalid_token_count) is not int
            or self.valid_token_count <= 0
            or self.invalid_token_count < 0
        ):
            raise ValueError("residual mutation receipt token counts are invalid")

    def metadata(self) -> dict[str, object]:
        return {
            "wire_id": self.wire_id,
            "mutation_id": self.mutation_id,
            "version_before": self.version_before,
            "version_after": self.version_after,
            "valid_token_count": self.valid_token_count,
            "invalid_token_count": self.invalid_token_count,
            "combiner_kind": self.combiner_kind,
        }


@runtime_checkable
class ResidualCarrier(Protocol):
    """Representation-neutral residual carrier used by graph executors.

    Implementations are persistent values: ``normalized_view`` and
    ``materialize`` must not expose backend-owned mutable storage, and
    ``apply_mutation`` must return a distinct next value without modifying the
    receiver or caller-owned tensors.
    """

    @property
    def backend_kind(self) -> str: ...

    @property
    def wire_id(self) -> str: ...

    @property
    def shape(self) -> torch.Size: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def version(self) -> int: ...

    def normalized_view(self, normalizer: StateNormalizer) -> Tensor: ...

    def apply_mutation(
        self,
        delta: Tensor,
        combiner: StateCombiner | None = None,
    ) -> ResidualCarrier: ...

    def materialize(self) -> Tensor: ...

    def metadata(self) -> Mapping[str, object]: ...


@runtime_checkable
class ResidualCarrierFactory(Protocol):
    """Factory seam for one independently owned carrier per model forward."""

    def create(
        self,
        initial_state: Tensor,
        context: ResidualGraphExecutionContext,
        *,
        wire_id: str,
    ) -> ResidualCarrier: ...


class _DenseResidualCarrier:
    """Persistent-value dense backend; updates never modify prior values."""

    __slots__ = (
        "_context",
        "_initial_state",
        "_state",
        "_version",
        "_wire_id",
    )

    backend_kind = "dense_exact_v1"

    def __init__(
        self,
        state: Tensor,
        initial_state: Tensor,
        context: ResidualGraphExecutionContext,
        *,
        wire_id: str,
        version: int,
    ) -> None:
        self._validate_tensor(state, context, label="carrier state")
        self._validate_tensor(
            initial_state,
            context,
            label="carrier initial state",
        )
        if state.shape != initial_state.shape:
            raise ValueError("carrier state and initial state shapes differ")
        if type(version) is not int or version < 0:
            raise ValueError("carrier version must be nonnegative")
        self._wire_id = _require_identifier(wire_id, label="wire_id")
        self._context = context
        self._state = state.clone()
        self._initial_state = initial_state.clone()
        self._version = version

    @staticmethod
    def _validate_tensor(
        value: object,
        context: ResidualGraphExecutionContext,
        *,
        label: str,
    ) -> None:
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise ValueError(f"{label} must have shape [batch, sequence, width]")
        if value.shape[:2] != (
            context.batch_size,
            context.query_length,
        ):
            raise ValueError(f"{label} does not match the execution context")
        if not value.is_floating_point() or value.shape[-1] <= 0:
            raise ValueError(f"{label} must have floating point residual width")
        if value.device != context.device:
            raise ValueError(f"{label} is on the wrong device")

    @property
    def wire_id(self) -> str:
        return self._wire_id

    @property
    def shape(self) -> torch.Size:
        return self._state.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._state.dtype

    @property
    def device(self) -> torch.device:
        return self._state.device

    @property
    def version(self) -> int:
        return self._version

    def normalized_view(self, normalizer: StateNormalizer) -> Tensor:
        self._context.assert_unchanged()
        if not callable(normalizer):
            raise TypeError("normalizer must be callable")
        try:
            result = normalizer(self._state.clone())
        finally:
            self._context.assert_unchanged()
        self._validate_result(result, label="normalized carrier view")
        return result

    def _validate_result(self, value: object, *, label: str) -> None:
        if not isinstance(value, Tensor) or value.shape != self.shape:
            raise ValueError(f"{label} must preserve carrier shape")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError(f"{label} must preserve carrier dtype and device")

    def apply_mutation(
        self,
        delta: Tensor,
        combiner: StateCombiner | None = None,
    ) -> ResidualCarrier:
        self._context.assert_unchanged()
        if not isinstance(delta, Tensor) or delta.shape != self.shape:
            raise ValueError("residual mutation must match carrier shape")
        if delta.dtype != self.dtype or delta.device != self.device:
            raise ValueError(
                "residual mutation must match carrier dtype and device"
            )
        if combiner is not None and not callable(combiner):
            raise TypeError("residual mutation combiner must be callable")

        try:
            valid = self._context.query_valid_mask.unsqueeze(-1)
            masked_delta = delta.masked_fill(~valid, 0)
            if combiner is None:
                combined = self._state + masked_delta
            else:
                combined = combiner(
                    self._state.clone(),
                    masked_delta.clone(),
                )
            self._validate_result(combined, label="combined residual state")
            next_state = torch.where(valid, combined, self._initial_state)
            result = _DenseResidualCarrier(
                next_state,
                self._initial_state,
                self._context,
                wire_id=self.wire_id,
                version=self.version + 1,
            )
        finally:
            self._context.assert_unchanged()
        return result

    def materialize(self) -> Tensor:
        self._context.assert_unchanged()
        result = self._state.clone()
        self._context.assert_unchanged()
        return result

    def metadata(self) -> Mapping[str, object]:
        self._context.assert_unchanged()
        return {
            "backend_kind": self.backend_kind,
            "wire_id": self.wire_id,
            "shape": list(self.shape),
            "dtype": _dtype_name(self.dtype),
            "device": str(self.device),
            "version": self.version,
            "residual_state_scalar_count": self._state.numel(),
            "padding_baseline_scalar_count": self._initial_state.numel(),
            "dynamic_state_scalar_count": (
                self._state.numel() + self._initial_state.numel()
            ),
            "learned_parameter_count": 0,
            "fixed_runtime_coefficient_count": 0,
            "dense_materialization_available": True,
        }


class DenseResidualCarrierFactory:
    """Create independent dense carriers for individual model forwards."""

    __slots__ = ()
    backend_kind = _DenseResidualCarrier.backend_kind

    def create(
        self,
        initial_state: Tensor,
        context: ResidualGraphExecutionContext,
        *,
        wire_id: str,
    ) -> ResidualCarrier:
        context.assert_unchanged()
        try:
            result = _DenseResidualCarrier(
                initial_state,
                initial_state,
                context,
                wire_id=wire_id,
                version=0,
            )
        finally:
            context.assert_unchanged()
        return result


class ResidualCarrierSession:
    """Ordered transactional facade around one representation backend."""

    __slots__ = (
        "_backend_kind",
        "_caller_initial_reference",
        "_caller_initial_snapshot",
        "_carrier",
        "_context",
        "_context_metadata",
        "_device",
        "_dtype",
        "_expected_mutation_ids",
        "_receipts",
        "_shape",
        "_status",
        "_version",
        "_wire_id",
    )

    def __init__(self) -> None:
        raise TypeError("use ResidualCarrierSession.begin(...)")

    @classmethod
    def begin(
        cls,
        initial_state: Tensor,
        context: ResidualGraphExecutionContext,
        expected_mutation_ids: Sequence[str],
        wire_id: str,
        factory: ResidualCarrierFactory | None = None,
    ) -> ResidualCarrierSession:
        if not isinstance(context, ResidualGraphExecutionContext):
            raise TypeError(
                "context must be a ResidualGraphExecutionContext"
            )
        context.assert_unchanged()
        if not isinstance(initial_state, Tensor) or initial_state.ndim != 3:
            raise ValueError(
                "initial_state must have shape [batch, sequence, width]"
            )
        if initial_state.shape[:2] != (
            context.batch_size,
            context.query_length,
        ):
            raise ValueError("initial_state does not match context shape")
        if not initial_state.is_floating_point() or initial_state.shape[-1] <= 0:
            raise ValueError("initial_state must have floating residual width")
        if initial_state.device != context.device:
            raise ValueError("initial_state and context must share a device")

        if isinstance(expected_mutation_ids, (str, bytes)) or not isinstance(
            expected_mutation_ids,
            Sequence,
        ):
            raise TypeError("expected mutation ids must be a sequence of strings")
        mutation_ids = tuple(expected_mutation_ids)
        if not mutation_ids:
            raise ValueError("residual session requires expected mutations")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in mutation_ids
        ):
            raise ValueError("expected mutation ids must be nonempty strings")
        if len(set(mutation_ids)) != len(mutation_ids):
            raise ValueError("expected mutation ids must be unique")
        resolved_wire_id = _require_identifier(wire_id, label="wire_id")
        resolved_factory = (
            DenseResidualCarrierFactory() if factory is None else factory
        )
        if not isinstance(resolved_factory, ResidualCarrierFactory):
            raise TypeError("residual carrier factory must expose create()")
        caller_initial_snapshot = initial_state.detach().clone()
        context_metadata = context.metadata()

        def assert_begin_inputs_unchanged() -> None:
            context.assert_unchanged()
            if not _same_tensor(initial_state, caller_initial_snapshot):
                raise RuntimeError(
                    "residual carrier factory changed caller initial_state"
                )

        try:
            carrier = resolved_factory.create(
                initial_state,
                context,
                wire_id=resolved_wire_id,
            )
            if not isinstance(carrier, ResidualCarrier):
                raise TypeError(
                    "residual carrier factory returned an invalid backend"
                )
            backend_kind = _require_identifier(
                carrier.backend_kind,
                label="carrier backend_kind",
            )
            if (
                carrier.wire_id != resolved_wire_id
                or type(carrier.version) is not int
                or carrier.version != 0
                or carrier.shape != initial_state.shape
                or carrier.dtype != initial_state.dtype
                or carrier.device != initial_state.device
            ):
                raise ValueError("initial residual carrier metadata is invalid")
        finally:
            assert_begin_inputs_unchanged()

        self = object.__new__(cls)
        self._backend_kind = backend_kind
        self._caller_initial_reference = initial_state
        self._caller_initial_snapshot = caller_initial_snapshot
        self._carrier = carrier
        self._context = context
        self._context_metadata = context_metadata
        self._device = initial_state.device
        self._dtype = initial_state.dtype
        self._expected_mutation_ids = mutation_ids
        self._receipts: list[ResidualMutationReceipt] = []
        self._shape = initial_state.shape
        self._status = "active"
        self._version = 0
        self._wire_id = resolved_wire_id
        return self

    @property
    def active(self) -> bool:
        return self._status == "active"

    @property
    def live(self) -> bool:
        return self.active and self._carrier is not None

    @property
    def version(self) -> int:
        return self._version

    @property
    def receipts(self) -> tuple[ResidualMutationReceipt, ...]:
        return tuple(self._receipts)

    def _assert_execution_inputs_unchanged(self) -> None:
        context = self._context
        reference = self._caller_initial_reference
        snapshot = self._caller_initial_snapshot
        if (
            not isinstance(context, ResidualGraphExecutionContext)
            or not isinstance(reference, Tensor)
            or not isinstance(snapshot, Tensor)
        ):
            raise RuntimeError("residual carrier session resources were released")
        context.assert_unchanged()
        if not _same_tensor(reference, snapshot):
            raise RuntimeError(
                "caller initial_state changed during residual execution"
            )

    def _assert_active(self) -> ResidualCarrier:
        if not self.active or self._carrier is None:
            raise RuntimeError("residual carrier session is not active")
        self._assert_execution_inputs_unchanged()
        return self._carrier

    def _release(self, *, status: str) -> None:
        self._carrier = None
        self._caller_initial_reference = None
        self._caller_initial_snapshot = None
        self._context = None
        self._status = status

    def _invoke_backend(
        self,
        callback: Callable[[ResidualCarrier], object],
    ) -> object:
        carrier = self._assert_active()
        try:
            result = callback(carrier)
        except BaseException as error:
            try:
                self._assert_execution_inputs_unchanged()
            except BaseException as guard_error:
                self._release(status="aborted")
                raise guard_error from error
            self._release(status="aborted")
            raise
        try:
            self._assert_execution_inputs_unchanged()
        except BaseException:
            self._release(status="aborted")
            raise
        return result

    def _validate_tensor_result(self, value: object, *, label: str) -> Tensor:
        if not isinstance(value, Tensor) or value.shape != self._shape:
            raise ValueError(f"{label} must preserve carrier shape")
        if value.dtype != self._dtype or value.device != self._device:
            raise ValueError(f"{label} must preserve carrier dtype and device")
        return value

    def _validate_next_carrier(
        self,
        value: object,
        *,
        previous: ResidualCarrier,
    ) -> ResidualCarrier:
        if not isinstance(value, ResidualCarrier):
            raise TypeError("carrier mutation returned an invalid backend")
        if value is previous:
            raise RuntimeError(
                "carrier mutation must return a distinct persistent value"
            )
        if (
            value.backend_kind != self._backend_kind
            or value.wire_id != self._wire_id
            or value.shape != self._shape
            or value.dtype != self._dtype
            or value.device != self._device
            or type(value.version) is not int
            or value.version != self.version + 1
        ):
            raise RuntimeError("carrier mutation changed backend invariants")
        return value

    def normalized_view(self, normalizer: StateNormalizer) -> Tensor:
        if not callable(normalizer):
            raise TypeError("normalizer must be callable")

        def read(carrier: ResidualCarrier) -> Tensor:
            return self._validate_tensor_result(
                carrier.normalized_view(normalizer),
                label="normalized carrier view",
            )

        result = self._invoke_backend(read)
        assert isinstance(result, Tensor)
        return result

    @staticmethod
    def _combiner_kind(combiner: StateCombiner | None) -> str:
        if combiner is None:
            return "add"
        module = getattr(combiner, "__module__", None)
        qualname = getattr(combiner, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str):
            return f"{module}.{qualname}"
        kind = type(combiner)
        return f"{kind.__module__}.{kind.__qualname__}"

    def apply_mutation(
        self,
        mutation_id: str,
        delta: Tensor,
        combiner: StateCombiner | None = None,
    ) -> ResidualMutationReceipt:
        self._assert_active()
        supplied = _require_identifier(mutation_id, label="mutation_id")
        if self.version >= len(self._expected_mutation_ids):
            raise RuntimeError(
                "residual carrier session has no remaining mutations"
            )
        expected = self._expected_mutation_ids[self.version]
        if supplied != expected:
            raise RuntimeError(
                "residual mutation order mismatch: "
                f"expected {expected!r}, received {supplied!r}"
            )
        if not isinstance(delta, Tensor) or delta.shape != self._shape:
            raise ValueError("residual mutation must match carrier shape")
        if delta.dtype != self._dtype or delta.device != self._device:
            raise ValueError(
                "residual mutation must match carrier dtype and device"
            )
        if combiner is not None and not callable(combiner):
            raise TypeError("residual mutation combiner must be callable")

        def mutate(carrier: ResidualCarrier) -> ResidualCarrier:
            return self._validate_next_carrier(
                carrier.apply_mutation(delta, combiner),
                previous=carrier,
            )

        next_carrier = cast(ResidualCarrier, self._invoke_backend(mutate))
        context = self._context
        assert isinstance(context, ResidualGraphExecutionContext)
        receipt = ResidualMutationReceipt(
            wire_id=self._wire_id,
            mutation_id=supplied,
            version_before=self.version,
            version_after=self.version + 1,
            valid_token_count=context.valid_token_count,
            invalid_token_count=(
                context.batch_size * context.query_length
                - context.valid_token_count
            ),
            combiner_kind=self._combiner_kind(combiner),
        )
        self._carrier = next_carrier
        self._version += 1
        self._receipts.append(receipt)
        return receipt

    def materialize(self) -> Tensor:
        def read(carrier: ResidualCarrier) -> Tensor:
            return self._validate_tensor_result(
                carrier.materialize(),
                label="materialized carrier state",
            ).clone()

        result = self._invoke_backend(read)
        assert isinstance(result, Tensor)
        return result

    def finish(self) -> ResidualCarrier:
        carrier = self._assert_active()
        if self.version != len(self._expected_mutation_ids):
            raise RuntimeError(
                "residual carrier session cannot finish before every mutation"
            )
        self._release(status="finished")
        return carrier

    def abort(self) -> None:
        if self._status == "finished":
            raise RuntimeError("finished residual carrier session cannot abort")
        if self._status == "aborted":
            return
        self._release(status="aborted")

    @property
    def metadata(self) -> dict[str, object]:
        if self.active:
            self._assert_execution_inputs_unchanged()
        return {
            "schema": "fisher_graph.residual_carrier_session.v1",
            "wire_id": self._wire_id,
            "backend_kind": self._backend_kind,
            "status": self._status,
            "active": self.active,
            "live": self.live,
            "version": self.version,
            "expected_mutation_count": len(self._expected_mutation_ids),
            "expected_mutation_ids": list(self._expected_mutation_ids),
            "committed_mutation_count": len(self._receipts),
            "receipts": [receipt.metadata() for receipt in self._receipts],
            "shape": list(self._shape),
            "dtype": _dtype_name(self._dtype),
            "device": str(self._device),
            "caller_initial_state_guarded": True,
            "context": dict(self._context_metadata),
        }


__all__ = [
    "DenseResidualCarrierFactory",
    "ResidualCarrier",
    "ResidualCarrierFactory",
    "ResidualCarrierSession",
    "ResidualGraphExecutionContext",
    "ResidualMutationReceipt",
]
