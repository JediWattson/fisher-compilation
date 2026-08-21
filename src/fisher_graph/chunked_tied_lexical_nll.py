"""Bounded exact NLL/VJP for Gemma's tied lexical readout.

The production entrypoint is deliberately narrower than the numeric kernel:
it freezes the 32 prediction-row by 4096 vocabulary-column schedule, requires
an authenticated no-padding q->t batch axis, and accepts only a bound immutable
tied-readout tensor.  Torch remains a local import so this source file itself
is safe to inspect during a model-free preflight.

No prompt text, token IDs, logits, terminal values, or gradients are serialized.
Receipts contain counts, public row-axis metadata, scalar errors/losses, and
domain-separated hashes only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


KERNEL_ID = "gemma3_full_vocab_softcap_shifted_nll_vjp.chunked.v1"
PARITY_KERNEL_ID = "gemma3_full_vocab_softcap_shifted_nll_vjp.dense_smoke.v1"
FIXTURE_KERNEL_ID = "fixture_only.tied_softcap_nll_vjp.chunked.v1"

PRODUCTION_PREDICTION_POSITION_CHUNK_SIZE = 32
PRODUCTION_VOCABULARY_CHUNK_SIZE = 4096
PRODUCTION_HASH_ROW_CHUNK_SIZE = 4096

DENSE_PARITY_SELECTOR_ID = "batch_flattened_first_two_prediction_rows.v1"
DENSE_PARITY_SELECTED_PREDICTION_ROW_COUNT = 2
DENSE_PARITY_ABSOLUTE_TOLERANCE = 2.0e-5
DENSE_PARITY_RELATIVE_TOLERANCE = 2.0e-5

EXECUTABLE_PANEL_ROLES = (
    "A_chart_transport_fit",
    "B_neighbor_discovery",
    "C_neighbor_confirmation",
    "D_response_validation",
)

_RECEIPT_DOMAIN = b"fisher_graph:chunked-tied-lexical-nll:v2\0"
_CONFIG_DOMAIN = b"fisher_graph:chunked-tied-lexical-nll-config:v2\0"
_BACKEND_DOMAIN = b"fisher_graph:chunked-tied-lexical-nll-backend:v2\0"
_KERNEL_RECEIPT_DOMAIN = (
    b"fisher_graph:chunked-tied-lexical-nll-kernel-receipt:v2\0"
)
_TENSOR_DOMAIN = b"fisher_graph:chunked-tied-lexical-nll-tensor:v2\0"
_PARITY_DOMAIN = b"fisher_graph:chunked-tied-lexical-nll-parity:v2\0"
_BATCH_AXIS_DOMAIN = b"fisher_graph:chunked-tied-lexical-batch-axis:v1\0"
_TIED_BINDING_DOMAIN = b"fisher_graph:chunked-tied-readout-binding:v1\0"
_SELECTED_ROW_INDEX_DOMAIN = (
    b"fisher_graph:chunked-tied-lexical-selected-row-index:v1\0"
)

# This is intentionally identical to the frozen fresh-panel author's domain.
# It lets this model-independent helper recompute, rather than trust, every
# per-example `(role,family,example,q,t=q+1)` stream commitment.
_FRESH_PANEL_ROW_STREAM_DOMAIN = (
    b"fisher-graph:chart-neighbor-fresh:row-stream:v1\0"
)

_BINDING_FACTORY_TOKEN = object()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonempty_id(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > 512
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} must be a short nonempty identifier")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} fields drifted; missing={missing}, extra={extra}")


def _normalize_softcap(value: object) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("logit_softcap must be finite and positive or None")
    return float(value)


def _fresh_row_stream_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        _FRESH_PANEL_ROW_STREAM_DOMAIN + _canonical_json_bytes(list(rows))
    ).hexdigest()


def objective_row_stream_sha256(
    *,
    role: str,
    family_id: str,
    example_id: str,
    token_count: int,
) -> str:
    """Recompute one frozen example's exact `(role,family,example,q,t)` SHA."""

    role = _nonempty_id(role, label="role")
    family_id = _nonempty_id(family_id, label="family_id")
    example_id = _nonempty_id(example_id, label="example_id")
    token_count = _positive_int(token_count, label="token_count")
    if token_count < 2:
        raise ValueError("objective row stream requires at least two tokens")
    return _fresh_row_stream_sha256(
        [
            {
                "role": role,
                "family_id": family_id,
                "example_id": example_id,
                "prediction_query_position": query_position,
                "target_label_position": query_position + 1,
            }
            for query_position in range(token_count - 1)
        ]
    )


@dataclass(frozen=True, slots=True)
class BatchObjectiveExample:
    family_id: str
    example_id: str
    target_token_count: int
    frozen_objective_row_stream_sha256: str

    def __post_init__(self) -> None:
        _nonempty_id(self.family_id, label="family_id")
        _nonempty_id(self.example_id, label="example_id")
        _positive_int(self.target_token_count, label="target_token_count")
        if self.target_token_count < 2:
            raise ValueError("target_token_count must be at least two")
        _require_sha256(
            self.frozen_objective_row_stream_sha256,
            label="frozen_objective_row_stream_sha256",
        )

    def payload(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "frozen_objective_row_stream_sha256": (
                self.frozen_objective_row_stream_sha256
            ),
            "target_token_count": self.target_token_count,
        }


@dataclass(frozen=True, slots=True)
class BatchObjectiveAxis:
    """Exact local q/t catalog plus its frozen role-level parents."""

    role: str
    examples: tuple[BatchObjectiveExample, ...]
    role_score_objective_receipt_sha256: str
    role_ordered_objective_row_catalog_sha256: str

    def __post_init__(self) -> None:
        if self.role not in EXECUTABLE_PANEL_ROLES:
            raise ValueError("batch objective role is not executable")
        if (
            not isinstance(self.examples, tuple)
            or not self.examples
            or any(
                not isinstance(example, BatchObjectiveExample)
                for example in self.examples
            )
        ):
            raise TypeError("batch objective examples must be a nonempty tuple")
        _require_sha256(
            self.role_score_objective_receipt_sha256,
            label="role_score_objective_receipt_sha256",
        )
        _require_sha256(
            self.role_ordered_objective_row_catalog_sha256,
            label="role_ordered_objective_row_catalog_sha256",
        )
        token_counts = {example.target_token_count for example in self.examples}
        if len(token_counts) != 1:
            raise ValueError(
                "one rectangular batch requires equal exact per-example lengths"
            )
        identities = [example.example_id for example in self.examples]
        if len(set(identities)) != len(identities):
            raise ValueError("batch objective example IDs must be unique")
        for example in self.examples:
            observed = objective_row_stream_sha256(
                role=self.role,
                family_id=example.family_id,
                example_id=example.example_id,
                token_count=example.target_token_count,
            )
            if observed != example.frozen_objective_row_stream_sha256:
                raise ValueError(
                    f"frozen q/t row stream drifted for {example.example_id}"
                )

    @property
    def batch_size(self) -> int:
        return len(self.examples)

    @property
    def token_count(self) -> int:
        return self.examples[0].target_token_count

    @property
    def prediction_row_count(self) -> int:
        return self.batch_size * (self.token_count - 1)

    def ordered_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "role": self.role,
                "family_id": example.family_id,
                "example_id": example.example_id,
                "prediction_query_position": query_position,
                "target_label_position": query_position + 1,
            }
            for example in self.examples
            for query_position in range(example.target_token_count - 1)
        )

    @property
    def batch_ordered_objective_row_catalog_sha256(self) -> str:
        return _fresh_row_stream_sha256(self.ordered_rows())

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "all_tokens_valid_required": True,
            "batch_ordered_objective_row_catalog_sha256": (
                self.batch_ordered_objective_row_catalog_sha256
            ),
            "batch_size": self.batch_size,
            "examples": [example.payload() for example in self.examples],
            "flatten_order": "ordered_example_then_ascending_q",
            "padding_rows_forbidden": True,
            "position_relation": "target_label_position_equals_q_plus_1",
            "prediction_query_position_range": "0_through_L_minus_2",
            "prediction_row_count": self.prediction_row_count,
            "role": self.role,
            "role_ordered_objective_row_catalog_sha256": (
                self.role_ordered_objective_row_catalog_sha256
            ),
            "role_score_objective_receipt_sha256": (
                self.role_score_objective_receipt_sha256
            ),
            "token_count": self.token_count,
        }

    @property
    def sha256(self) -> str:
        return _digest(_BATCH_AXIS_DOMAIN, self.payload_without_sha256())

    def canonical_payload(self) -> dict[str, object]:
        return {**self.payload_without_sha256(), "axis_sha256": self.sha256}


def parse_batch_objective_axis_payload(
    value: Mapping[str, object],
) -> BatchObjectiveAxis:
    expected_fields = {
        "all_tokens_valid_required",
        "axis_sha256",
        "batch_ordered_objective_row_catalog_sha256",
        "batch_size",
        "examples",
        "flatten_order",
        "padding_rows_forbidden",
        "position_relation",
        "prediction_query_position_range",
        "prediction_row_count",
        "role",
        "role_ordered_objective_row_catalog_sha256",
        "role_score_objective_receipt_sha256",
        "token_count",
    }
    _require_exact_fields(value, expected_fields, label="batch objective axis")
    examples_value = value["examples"]
    if not isinstance(examples_value, list) or not examples_value:
        raise ValueError("batch objective examples payload is invalid")
    examples: list[BatchObjectiveExample] = []
    example_fields = {
        "example_id",
        "family_id",
        "frozen_objective_row_stream_sha256",
        "target_token_count",
    }
    for index, item in enumerate(examples_value):
        if not isinstance(item, Mapping):
            raise TypeError(f"batch objective example {index} is not a mapping")
        _require_exact_fields(
            item,
            example_fields,
            label=f"batch objective example {index}",
        )
        examples.append(
            BatchObjectiveExample(
                family_id=item["family_id"],  # type: ignore[arg-type]
                example_id=item["example_id"],  # type: ignore[arg-type]
                target_token_count=item["target_token_count"],  # type: ignore[arg-type]
                frozen_objective_row_stream_sha256=item[
                    "frozen_objective_row_stream_sha256"
                ],  # type: ignore[arg-type]
            )
        )
    axis = BatchObjectiveAxis(
        role=value["role"],  # type: ignore[arg-type]
        examples=tuple(examples),
        role_score_objective_receipt_sha256=value[
            "role_score_objective_receipt_sha256"
        ],  # type: ignore[arg-type]
        role_ordered_objective_row_catalog_sha256=value[
            "role_ordered_objective_row_catalog_sha256"
        ],  # type: ignore[arg-type]
    )
    expected_payload = axis.canonical_payload()
    for field_name, expected in expected_payload.items():
        if value[field_name] != expected:
            raise ValueError(f"batch objective axis {field_name} drifted")
    return axis


@dataclass(frozen=True, slots=True)
class ChunkedTiedLexicalNLLConfig:
    """Frozen production resource schedule; custom chunks are forbidden."""

    prediction_position_chunk_size: int = (
        PRODUCTION_PREDICTION_POSITION_CHUNK_SIZE
    )
    vocabulary_chunk_size: int = PRODUCTION_VOCABULARY_CHUNK_SIZE
    kernel_id: str = KERNEL_ID

    def __post_init__(self) -> None:
        if type(self.prediction_position_chunk_size) is not int or (
            self.prediction_position_chunk_size
            != PRODUCTION_PREDICTION_POSITION_CHUNK_SIZE
        ):
            raise ValueError("production prediction chunk size must remain 32")
        if type(self.vocabulary_chunk_size) is not int or (
            self.vocabulary_chunk_size != PRODUCTION_VOCABULARY_CHUNK_SIZE
        ):
            raise ValueError("production vocabulary chunk size must remain 4096")
        if self.kernel_id != KERNEL_ID:
            raise ValueError("chunked tied lexical NLL kernel_id drifted")

    @property
    def maximum_single_logit_matrix_scalar_count(self) -> int:
        return (
            self.prediction_position_chunk_size * self.vocabulary_chunk_size
        )

    def payload(self) -> dict[str, object]:
        return {
            "full_B_L_V_logits_materialized": False,
            "kernel_id": self.kernel_id,
            "objective": "sum_next_token_NLL_over_q_0_through_L_minus_2",
            "prediction_position_chunk_size": (
                self.prediction_position_chunk_size
            ),
            "production_chunks_frozen": True,
            "softcap_derivative_included": True,
            "terminal_gradient_shape": "B_by_L_by_H",
            "two_pass_logsumexp_then_probability_VJP": True,
            "vocabulary_chunk_size": self.vocabulary_chunk_size,
        }

    def canonical_payload(self) -> dict[str, object]:
        return self.payload()

    @property
    def sha256(self) -> str:
        return _digest(_CONFIG_DOMAIN, self.payload())


@dataclass(frozen=True, slots=True)
class FixtureChunkedTiedLexicalNLLConfig:
    """Explicitly non-production chunk controls for tiny numeric tests only."""

    prediction_position_chunk_size: int = 3
    vocabulary_chunk_size: int = 7
    kernel_id: str = FIXTURE_KERNEL_ID

    def __post_init__(self) -> None:
        _positive_int(
            self.prediction_position_chunk_size,
            label="fixture prediction_position_chunk_size",
        )
        _positive_int(
            self.vocabulary_chunk_size,
            label="fixture vocabulary_chunk_size",
        )
        if self.kernel_id != FIXTURE_KERNEL_ID:
            raise ValueError("fixture chunked lexical kernel_id drifted")


def parse_chunked_tied_lexical_nll_config_payload(
    value: Mapping[str, object],
) -> ChunkedTiedLexicalNLLConfig:
    config = ChunkedTiedLexicalNLLConfig()
    expected = config.payload()
    _require_exact_fields(value, set(expected), label="chunked NLL config")
    if dict(value) != expected:
        raise ValueError("production chunked NLL config drifted")
    return config


def _tensor_chunks(value: Any, *, row_chunk_size: int) -> Any:
    if value.ndim == 0:
        yield value.detach().contiguous()
        return
    for start in range(0, int(value.shape[0]), row_chunk_size):
        yield value[start : start + row_chunk_size].detach().contiguous()


def tensor_stream_sha256(
    value: Any,
    *,
    row_chunk_size: int = PRODUCTION_HASH_ROW_CHUNK_SIZE,
) -> str:
    """Hash and finite-check a CPU tensor without a full-tensor copy/mask."""

    import torch

    row_chunk_size = _positive_int(row_chunk_size, label="row_chunk_size")
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type != "cpu"
    ):
        raise TypeError("authenticated tensor must be a dense CPU torch.Tensor")
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {"dtype": str(value.dtype), "shape": list(value.shape)}
        )
    )
    for chunk in _tensor_chunks(value, row_chunk_size=row_chunk_size):
        if chunk.is_floating_point() and not bool(torch.isfinite(chunk).all()):
            raise ValueError("authenticated floating tensor must be finite")
        digest.update(chunk.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TiedReadoutWeightReceipt:
    binding_kind: str
    tensor_sha256: str
    readout_state_sha256: str
    tied_lexical_lineage_sha256: str
    vocabulary_size: int
    hidden_width: int
    dtype: str
    device: str
    contiguous: bool
    dedicated_storage: bool
    requires_grad: bool
    element_count: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.binding_kind != "exact_tied_readout_weight.v1":
            raise ValueError("tied readout binding kind drifted")
        for label in (
            "tensor_sha256",
            "readout_state_sha256",
            "tied_lexical_lineage_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        for label in ("vocabulary_size", "hidden_width", "element_count"):
            _positive_int(getattr(self, label), label=label)
        if self.element_count != self.vocabulary_size * self.hidden_width:
            raise ValueError("tied readout element count drifted")
        if (
            not isinstance(self.dtype, str)
            or (
                not self.dtype.startswith("torch.float")
                and self.dtype != "torch.bfloat16"
            )
            or self.device != "cpu"
            or self.contiguous is not True
            or self.dedicated_storage is not True
            or self.requires_grad is not False
        ):
            raise ValueError("tied readout runtime contract drifted")
        if self.receipt_sha256 != _digest(
            _TIED_BINDING_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("tied readout binding receipt drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "binding_kind": self.binding_kind,
            "contiguous": self.contiguous,
            "dedicated_storage": self.dedicated_storage,
            "device": self.device,
            "dtype": self.dtype,
            "element_count": self.element_count,
            "hidden_width": self.hidden_width,
            "readout_state_sha256": self.readout_state_sha256,
            "requires_grad": self.requires_grad,
            "tensor_sha256": self.tensor_sha256,
            "tied_lexical_lineage_sha256": self.tied_lexical_lineage_sha256,
            "vocabulary_size": self.vocabulary_size,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {**self.payload_without_sha256(), "receipt_sha256": self.receipt_sha256}


class TiedReadoutWeightBinding:
    """Live immutable tensor handle plus a serializable lineage receipt."""

    __slots__ = (
        "_data_ptr",
        "_receipt",
        "_storage_data_ptr",
        "_storage_nbytes",
        "_storage_offset",
        "_stride",
        "_tensor",
        "_version",
    )

    def __init__(
        self,
        *,
        tensor: Any,
        receipt: TiedReadoutWeightReceipt,
        factory_token: object,
    ) -> None:
        if factory_token is not _BINDING_FACTORY_TOKEN:
            raise TypeError("use bind_tied_readout_weight()")
        self._tensor = tensor
        self._receipt = receipt
        self._data_ptr = int(tensor.data_ptr())
        self._storage_data_ptr = int(tensor.untyped_storage().data_ptr())
        self._storage_nbytes = int(tensor.untyped_storage().nbytes())
        self._storage_offset = int(tensor.storage_offset())
        self._stride = tuple(int(value) for value in tensor.stride())
        self._version = int(tensor._version)

    @property
    def tensor(self) -> Any:
        return self._tensor

    @property
    def receipt(self) -> TiedReadoutWeightReceipt:
        return self._receipt

    def validate_unchanged(self) -> None:
        tensor = self._tensor
        expected_bytes = self._receipt.element_count * tensor.element_size()
        if (
            int(tensor.data_ptr()) != self._data_ptr
            or int(tensor.untyped_storage().data_ptr()) != self._storage_data_ptr
            or int(tensor.untyped_storage().nbytes()) != self._storage_nbytes
            or int(tensor.storage_offset()) != self._storage_offset
            or tuple(int(value) for value in tensor.stride()) != self._stride
            or int(tensor._version) != self._version
            or self._storage_nbytes != expected_bytes
            or tensor.requires_grad
        ):
            raise RuntimeError("tied readout weight changed after binding")
        observed_sha256 = tensor_stream_sha256(tensor)
        if observed_sha256 != self._receipt.tensor_sha256:
            raise RuntimeError("tied readout weight changed after binding")


def bind_tied_readout_weight(
    tensor: Any,
    *,
    readout_state_sha256: str,
    tied_lexical_lineage_sha256: str,
) -> TiedReadoutWeightBinding:
    """Bind one dedicated, immutable CPU `[V,H]` tied-readout tensor."""

    import torch

    _require_sha256(readout_state_sha256, label="readout_state_sha256")
    _require_sha256(
        tied_lexical_lineage_sha256,
        label="tied_lexical_lineage_sha256",
    )
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.ndim != 2
        or not tensor.is_floating_point()
        or tensor.device.type != "cpu"
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or int(tensor.shape[0]) <= 0
        or int(tensor.shape[1]) <= 0
        or int(tensor.storage_offset()) != 0
    ):
        raise ValueError(
            "tied readout weight must be immutable, contiguous CPU [V,H]"
        )
    element_count = int(tensor.numel())
    expected_bytes = element_count * tensor.element_size()
    if int(tensor.untyped_storage().nbytes()) != expected_bytes:
        raise ValueError("tied readout weight must own one dedicated storage")
    tensor_sha256 = tensor_stream_sha256(tensor)
    values = {
        "binding_kind": "exact_tied_readout_weight.v1",
        "tensor_sha256": tensor_sha256,
        "readout_state_sha256": readout_state_sha256,
        "tied_lexical_lineage_sha256": tied_lexical_lineage_sha256,
        "vocabulary_size": int(tensor.shape[0]),
        "hidden_width": int(tensor.shape[1]),
        "dtype": str(tensor.dtype),
        "device": "cpu",
        "contiguous": True,
        "dedicated_storage": True,
        "requires_grad": False,
        "element_count": element_count,
    }
    payload = {
        "binding_kind": values["binding_kind"],
        "contiguous": values["contiguous"],
        "dedicated_storage": values["dedicated_storage"],
        "device": values["device"],
        "dtype": values["dtype"],
        "element_count": values["element_count"],
        "hidden_width": values["hidden_width"],
        "readout_state_sha256": values["readout_state_sha256"],
        "requires_grad": values["requires_grad"],
        "tensor_sha256": values["tensor_sha256"],
        "tied_lexical_lineage_sha256": values[
            "tied_lexical_lineage_sha256"
        ],
        "vocabulary_size": values["vocabulary_size"],
    }
    receipt = TiedReadoutWeightReceipt(
        **values,
        receipt_sha256=_digest(_TIED_BINDING_DOMAIN, payload),
    )
    return TiedReadoutWeightBinding(
        tensor=tensor,
        receipt=receipt,
        factory_token=_BINDING_FACTORY_TOKEN,
    )


def parse_tied_readout_weight_receipt_payload(
    value: Mapping[str, object],
) -> TiedReadoutWeightReceipt:
    expected = {
        "binding_kind",
        "contiguous",
        "dedicated_storage",
        "device",
        "dtype",
        "element_count",
        "hidden_width",
        "readout_state_sha256",
        "receipt_sha256",
        "requires_grad",
        "tensor_sha256",
        "tied_lexical_lineage_sha256",
        "vocabulary_size",
    }
    _require_exact_fields(value, expected, label="tied readout binding receipt")
    return TiedReadoutWeightReceipt(
        binding_kind=value["binding_kind"],  # type: ignore[arg-type]
        tensor_sha256=value["tensor_sha256"],  # type: ignore[arg-type]
        readout_state_sha256=value["readout_state_sha256"],  # type: ignore[arg-type]
        tied_lexical_lineage_sha256=value[
            "tied_lexical_lineage_sha256"
        ],  # type: ignore[arg-type]
        vocabulary_size=value["vocabulary_size"],  # type: ignore[arg-type]
        hidden_width=value["hidden_width"],  # type: ignore[arg-type]
        dtype=value["dtype"],  # type: ignore[arg-type]
        device=value["device"],  # type: ignore[arg-type]
        contiguous=value["contiguous"],  # type: ignore[arg-type]
        dedicated_storage=value["dedicated_storage"],  # type: ignore[arg-type]
        requires_grad=value["requires_grad"],  # type: ignore[arg-type]
        element_count=value["element_count"],  # type: ignore[arg-type]
        receipt_sha256=value["receipt_sha256"],  # type: ignore[arg-type]
    )


def backend_receipt_sha256(
    config: ChunkedTiedLexicalNLLConfig = ChunkedTiedLexicalNLLConfig(),
) -> str:
    import torch

    if not isinstance(config, ChunkedTiedLexicalNLLConfig):
        raise TypeError("backend receipt requires the frozen production config")
    return _digest(
        _BACKEND_DOMAIN,
        {
            "config_sha256": config.sha256,
            "device": "cpu",
            "reduction_dtype": "torch.float64",
            "torch_build_config_sha256": hashlib.sha256(
                torch.__config__.show().encode("utf-8")
            ).hexdigest(),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_version": str(torch.__version__),
        },
    )


def _validate_finite_cpu_float_matrix(value: Any, *, label: str) -> None:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or value.device.type != "cpu"
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) <= 0
    ):
        raise ValueError(f"{label} must be a nonempty CPU floating matrix")
    tensor_stream_sha256(value)


def _chunked_softcap_nll_vjp_rows(
    query_values: Any,
    labels: Any,
    tied_readout_weight: Any,
    *,
    logit_softcap: float | None,
    prediction_position_chunk_size: int,
    vocabulary_chunk_size: int,
    weight_prevalidated: bool,
) -> tuple[float, Any]:
    import torch

    _validate_finite_cpu_float_matrix(query_values, label="query_values")
    if not weight_prevalidated:
        _validate_finite_cpu_float_matrix(
            tied_readout_weight,
            label="tied readout weight",
        )
    elif (
        not isinstance(tied_readout_weight, torch.Tensor)
        or tied_readout_weight.ndim != 2
        or tied_readout_weight.device.type != "cpu"
        or not tied_readout_weight.is_floating_point()
    ):
        raise ValueError("prevalidated tied readout weight is invalid")
    if tied_readout_weight.shape[1] != query_values.shape[1]:
        raise ValueError("tied readout hidden width disagrees with query values")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.shape[0] != query_values.shape[0]
        or labels.dtype not in (torch.int32, torch.int64)
        or labels.device.type != "cpu"
        or bool((labels < 0).any())
        or bool((labels >= tied_readout_weight.shape[0]).any())
    ):
        raise ValueError("labels must be valid CPU vocabulary indices")
    softcap = _normalize_softcap(logit_softcap)
    rows = int(query_values.shape[0])
    width = int(query_values.shape[1])
    vocabulary = int(tied_readout_weight.shape[0])
    output = torch.empty((rows, width), dtype=torch.float64)
    losses: list[float] = []
    with torch.no_grad():
        for row_start in range(0, rows, prediction_position_chunk_size):
            row_end = min(rows, row_start + prediction_position_chunk_size)
            queries = query_values[row_start:row_end].to(torch.float64)
            row_labels = labels[row_start:row_end].to(torch.int64)
            running_max = torch.full(
                (row_end - row_start,),
                -torch.inf,
                dtype=torch.float64,
            )
            running_exp_sum = torch.zeros_like(running_max)
            for vocab_start in range(0, vocabulary, vocabulary_chunk_size):
                vocab_end = min(vocabulary, vocab_start + vocabulary_chunk_size)
                weight = tied_readout_weight[vocab_start:vocab_end].to(
                    torch.float64
                )
                raw_logits = torch.mm(queries, weight.T)
                capped_logits = (
                    raw_logits
                    if softcap is None
                    else softcap * torch.tanh(raw_logits / softcap)
                )
                local_max = capped_logits.max(dim=1).values
                updated_max = torch.maximum(running_max, local_max)
                running_exp_sum = (
                    running_exp_sum * torch.exp(running_max - updated_max)
                    + torch.exp(capped_logits - updated_max[:, None]).sum(dim=1)
                )
                running_max = updated_max
            log_normalizer = running_max + torch.log(running_exp_sum)
            gold_weight = tied_readout_weight[row_labels].to(torch.float64)
            gold_raw = (queries * gold_weight).sum(dim=1)
            gold_capped = (
                gold_raw
                if softcap is None
                else softcap * torch.tanh(gold_raw / softcap)
            )
            losses.extend((log_normalizer - gold_capped).tolist())

            gradient = torch.zeros_like(queries)
            for vocab_start in range(0, vocabulary, vocabulary_chunk_size):
                vocab_end = min(vocabulary, vocab_start + vocabulary_chunk_size)
                weight = tied_readout_weight[vocab_start:vocab_end].to(
                    torch.float64
                )
                raw_logits = torch.mm(queries, weight.T)
                if softcap is None:
                    capped_logits = raw_logits
                    derivative = torch.ones_like(raw_logits)
                else:
                    tangent = torch.tanh(raw_logits / softcap)
                    capped_logits = softcap * tangent
                    derivative = 1.0 - tangent.square()
                probability = torch.exp(capped_logits - log_normalizer[:, None])
                gradient.addmm_(probability * derivative, weight)
            gold_derivative = (
                torch.ones_like(gold_raw)
                if softcap is None
                else 1.0 - torch.tanh(gold_raw / softcap).square()
            )
            gradient -= gold_derivative[:, None] * gold_weight
            output[row_start:row_end] = gradient
    return math.fsum(losses), output


def fixture_chunked_softcap_nll_vjp_rows(
    query_values: Any,
    labels: Any,
    tied_readout_weight: Any,
    *,
    logit_softcap: float | None,
    config: FixtureChunkedTiedLexicalNLLConfig = (
        FixtureChunkedTiedLexicalNLLConfig()
    ),
) -> tuple[float, Any]:
    """Test-only numeric kernel with deliberately configurable tiny chunks."""

    if not isinstance(config, FixtureChunkedTiedLexicalNLLConfig):
        raise TypeError("fixture numeric kernel requires its fixture-only config")
    return _chunked_softcap_nll_vjp_rows(
        query_values,
        labels,
        tied_readout_weight,
        logit_softcap=logit_softcap,
        prediction_position_chunk_size=(
            config.prediction_position_chunk_size
        ),
        vocabulary_chunk_size=config.vocabulary_chunk_size,
        weight_prevalidated=False,
    )


@dataclass(frozen=True, slots=True)
class ChunkedTiedLexicalNLLReceipt:
    kernel_id: str
    config_sha256: str
    batch_objective_axis_sha256: str
    batch_ordered_objective_row_catalog_sha256: str
    role_ordered_objective_row_catalog_sha256: str
    score_objective_receipt_sha256: str
    valid_token_mask_sha256: str
    terminal_values_sha256: str
    tied_readout_binding_sha256: str
    tied_readout_weight_sha256: str
    input_token_ids_sha256: str
    terminal_gradient_sha256: str
    backend_receipt_sha256: str
    batch_size: int
    token_count: int
    prediction_row_count: int
    hidden_width: int
    vocabulary_size: int
    logit_softcap: float | None
    summed_next_token_nll: float
    zero_terminal_gradient_row_count: int
    maximum_single_logit_matrix_scalar_count: int
    full_prediction_by_vocabulary_matrix_materialized: bool
    all_tokens_valid: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.kernel_id != KERNEL_ID:
            raise ValueError("chunked NLL receipt kernel drifted")
        for label in (
            "config_sha256",
            "batch_objective_axis_sha256",
            "batch_ordered_objective_row_catalog_sha256",
            "role_ordered_objective_row_catalog_sha256",
            "score_objective_receipt_sha256",
            "valid_token_mask_sha256",
            "terminal_values_sha256",
            "tied_readout_binding_sha256",
            "tied_readout_weight_sha256",
            "input_token_ids_sha256",
            "terminal_gradient_sha256",
            "backend_receipt_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        for label in (
            "batch_size",
            "token_count",
            "prediction_row_count",
            "hidden_width",
            "vocabulary_size",
            "zero_terminal_gradient_row_count",
            "maximum_single_logit_matrix_scalar_count",
        ):
            _positive_int(getattr(self, label), label=label)
        softcap = _normalize_softcap(self.logit_softcap)
        if softcap != self.logit_softcap:
            raise ValueError("receipt logit_softcap was not normalized")
        if type(self.summed_next_token_nll) is not float:
            raise ValueError("summed_next_token_nll must be a float")
        expected_maximum = min(
            self.prediction_row_count,
            PRODUCTION_PREDICTION_POSITION_CHUNK_SIZE,
        ) * min(self.vocabulary_size, PRODUCTION_VOCABULARY_CHUNK_SIZE)
        if (
            self.config_sha256 != ChunkedTiedLexicalNLLConfig().sha256
            or self.token_count < 2
            or self.prediction_row_count
            != self.batch_size * (self.token_count - 1)
            or self.zero_terminal_gradient_row_count != self.batch_size
            or self.maximum_single_logit_matrix_scalar_count != expected_maximum
            or self.full_prediction_by_vocabulary_matrix_materialized
            is not (
                self.prediction_row_count
                <= PRODUCTION_PREDICTION_POSITION_CHUNK_SIZE
                and self.vocabulary_size <= PRODUCTION_VOCABULARY_CHUNK_SIZE
            )
            or self.all_tokens_valid is not True
            or not math.isfinite(self.summed_next_token_nll)
            or self.summed_next_token_nll < 0.0
        ):
            raise ValueError("chunked NLL receipt dimensions or contract drifted")
        if self.receipt_sha256 != _digest(
            _RECEIPT_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("chunked NLL VJP receipt digest drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "all_tokens_valid": self.all_tokens_valid,
            "backend_receipt_sha256": self.backend_receipt_sha256,
            "batch_objective_axis_sha256": self.batch_objective_axis_sha256,
            "batch_ordered_objective_row_catalog_sha256": (
                self.batch_ordered_objective_row_catalog_sha256
            ),
            "batch_size": self.batch_size,
            "config_sha256": self.config_sha256,
            "full_B_L_V_logits_materialized": False,
            "full_prediction_by_vocabulary_matrix_materialized": (
                self.full_prediction_by_vocabulary_matrix_materialized
            ),
            "hidden_width": self.hidden_width,
            "input_token_ids_sha256": self.input_token_ids_sha256,
            "kernel_id": self.kernel_id,
            "logit_softcap": self.logit_softcap,
            "maximum_single_logit_matrix_scalar_count": (
                self.maximum_single_logit_matrix_scalar_count
            ),
            "prediction_row_count": self.prediction_row_count,
            "role_ordered_objective_row_catalog_sha256": (
                self.role_ordered_objective_row_catalog_sha256
            ),
            "score_objective_receipt_sha256": (
                self.score_objective_receipt_sha256
            ),
            "summed_next_token_nll": self.summed_next_token_nll,
            "terminal_gradient_sha256": self.terminal_gradient_sha256,
            "terminal_values_sha256": self.terminal_values_sha256,
            "tied_readout_binding_sha256": (
                self.tied_readout_binding_sha256
            ),
            "tied_readout_weight_sha256": self.tied_readout_weight_sha256,
            "token_count": self.token_count,
            "valid_token_mask_sha256": self.valid_token_mask_sha256,
            "vocabulary_size": self.vocabulary_size,
            "zero_terminal_gradient_row_count": (
                self.zero_terminal_gradient_row_count
            ),
        }

    def canonical_payload(self) -> dict[str, object]:
        return {**self.payload_without_sha256(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ChunkedTiedLexicalNLLResult:
    terminal_gradient: Any
    receipt: ChunkedTiedLexicalNLLReceipt


def _receipt_from_values(
    receipt_type: type[Any],
    *,
    domain: bytes,
    values: dict[str, object],
) -> Any:
    provisional = object.__new__(receipt_type)
    for field_name, field_value in values.items():
        object.__setattr__(provisional, field_name, field_value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return receipt_type(
        **values,
        receipt_sha256=_digest(domain, provisional.payload_without_sha256()),
    )


def _validate_runtime_batch(
    *,
    terminal_normalized_output: Any,
    input_token_ids: Any,
    valid_token_mask: Any,
    batch_axis: BatchObjectiveAxis,
    tied_readout_weight: TiedReadoutWeightBinding,
) -> tuple[int, int, int]:
    import torch

    if not isinstance(batch_axis, BatchObjectiveAxis):
        raise TypeError("batch_axis must be an authenticated BatchObjectiveAxis")
    if not isinstance(tied_readout_weight, TiedReadoutWeightBinding):
        raise TypeError("tied_readout_weight must be a bound tied-readout object")
    tied_readout_weight.validate_unchanged()
    if (
        not isinstance(terminal_normalized_output, torch.Tensor)
        or terminal_normalized_output.ndim != 3
        or terminal_normalized_output.device.type != "cpu"
        or not terminal_normalized_output.is_floating_point()
    ):
        raise ValueError("terminal normalized output must be finite CPU [B,L,H]")
    tensor_stream_sha256(terminal_normalized_output)
    if (
        not isinstance(input_token_ids, torch.Tensor)
        or input_token_ids.ndim != 2
        or input_token_ids.shape != terminal_normalized_output.shape[:2]
        or input_token_ids.dtype not in (torch.int32, torch.int64)
        or input_token_ids.device.type != "cpu"
    ):
        raise ValueError("input token IDs must align with terminal [B,L]")
    if (
        not isinstance(valid_token_mask, torch.Tensor)
        or valid_token_mask.dtype != torch.bool
        or valid_token_mask.device.type != "cpu"
        or valid_token_mask.shape != input_token_ids.shape
    ):
        raise ValueError("valid_token_mask must be boolean CPU [B,L]")
    if not bool(valid_token_mask.all()):
        raise ValueError("padding rows are forbidden; valid_token_mask must be all true")
    batch, token_count, width = map(int, terminal_normalized_output.shape)
    if (
        batch != batch_axis.batch_size
        or token_count != batch_axis.token_count
        or token_count < 2
    ):
        raise ValueError("runtime batch shape disagrees with exact objective axis")
    binding_receipt = tied_readout_weight.receipt
    if width != binding_receipt.hidden_width:
        raise ValueError("terminal width disagrees with tied readout binding")
    if bool((input_token_ids < 0).any()) or bool(
        (input_token_ids >= binding_receipt.vocabulary_size).any()
    ):
        raise ValueError("input token ID is outside the tied vocabulary")
    return batch, token_count, width


def compute_chunked_tied_lexical_nll_vjp(
    *,
    terminal_normalized_output: Any,
    input_token_ids: Any,
    valid_token_mask: Any,
    batch_axis: BatchObjectiveAxis,
    tied_readout_weight: TiedReadoutWeightBinding,
    logit_softcap: float | None,
    config: ChunkedTiedLexicalNLLConfig = ChunkedTiedLexicalNLLConfig(),
) -> ChunkedTiedLexicalNLLResult:
    """Return the exact full-vocabulary NLL cotangent with no `[B,L,V]`."""

    import torch

    if not isinstance(config, ChunkedTiedLexicalNLLConfig):
        raise TypeError("production NLL requires the frozen production config")
    softcap = _normalize_softcap(logit_softcap)
    batch, token_count, width = _validate_runtime_batch(
        terminal_normalized_output=terminal_normalized_output,
        input_token_ids=input_token_ids,
        valid_token_mask=valid_token_mask,
        batch_axis=batch_axis,
        tied_readout_weight=tied_readout_weight,
    )
    flat_queries = terminal_normalized_output[:, :-1, :].reshape(-1, width)
    flat_labels = input_token_ids[:, 1:].reshape(-1)
    loss, flat_gradient64 = _chunked_softcap_nll_vjp_rows(
        flat_queries,
        flat_labels,
        tied_readout_weight.tensor,
        logit_softcap=softcap,
        prediction_position_chunk_size=(
            config.prediction_position_chunk_size
        ),
        vocabulary_chunk_size=config.vocabulary_chunk_size,
        weight_prevalidated=True,
    )
    gradient = torch.zeros_like(terminal_normalized_output)
    gradient[:, :-1, :] = flat_gradient64.reshape(
        batch,
        token_count - 1,
        width,
    ).to(dtype=terminal_normalized_output.dtype)
    if not bool((gradient[:, -1, :] == 0).all()):
        raise RuntimeError("terminal no-label gradient row must be exactly zero")
    prediction_rows = batch_axis.prediction_row_count
    vocabulary = tied_readout_weight.receipt.vocabulary_size
    values: dict[str, object] = {
        "kernel_id": config.kernel_id,
        "config_sha256": config.sha256,
        "batch_objective_axis_sha256": batch_axis.sha256,
        "batch_ordered_objective_row_catalog_sha256": (
            batch_axis.batch_ordered_objective_row_catalog_sha256
        ),
        "role_ordered_objective_row_catalog_sha256": (
            batch_axis.role_ordered_objective_row_catalog_sha256
        ),
        "score_objective_receipt_sha256": (
            batch_axis.role_score_objective_receipt_sha256
        ),
        "valid_token_mask_sha256": tensor_stream_sha256(valid_token_mask),
        "terminal_values_sha256": tensor_stream_sha256(
            terminal_normalized_output
        ),
        "tied_readout_binding_sha256": (
            tied_readout_weight.receipt.receipt_sha256
        ),
        "tied_readout_weight_sha256": (
            tied_readout_weight.receipt.tensor_sha256
        ),
        "input_token_ids_sha256": tensor_stream_sha256(input_token_ids),
        "terminal_gradient_sha256": tensor_stream_sha256(gradient),
        "backend_receipt_sha256": backend_receipt_sha256(config),
        "batch_size": batch,
        "token_count": token_count,
        "prediction_row_count": prediction_rows,
        "hidden_width": width,
        "vocabulary_size": vocabulary,
        "logit_softcap": softcap,
        "summed_next_token_nll": loss,
        "zero_terminal_gradient_row_count": batch,
        "maximum_single_logit_matrix_scalar_count": min(
            prediction_rows,
            config.prediction_position_chunk_size,
        )
        * min(vocabulary, config.vocabulary_chunk_size),
        "full_prediction_by_vocabulary_matrix_materialized": (
            prediction_rows <= config.prediction_position_chunk_size
            and vocabulary <= config.vocabulary_chunk_size
        ),
        "all_tokens_valid": True,
    }
    receipt = _receipt_from_values(
        ChunkedTiedLexicalNLLReceipt,
        domain=_RECEIPT_DOMAIN,
        values=values,
    )
    return ChunkedTiedLexicalNLLResult(
        terminal_gradient=gradient,
        receipt=receipt,
    )


def parse_chunked_tied_lexical_nll_receipt_payload(
    value: Mapping[str, object],
    *,
    expected_axis: BatchObjectiveAxis | None = None,
    expected_weight: TiedReadoutWeightBinding | TiedReadoutWeightReceipt | None = None,
) -> ChunkedTiedLexicalNLLReceipt:
    expected_fields = {
        "all_tokens_valid",
        "backend_receipt_sha256",
        "batch_objective_axis_sha256",
        "batch_ordered_objective_row_catalog_sha256",
        "batch_size",
        "config_sha256",
        "full_B_L_V_logits_materialized",
        "full_prediction_by_vocabulary_matrix_materialized",
        "hidden_width",
        "input_token_ids_sha256",
        "kernel_id",
        "logit_softcap",
        "maximum_single_logit_matrix_scalar_count",
        "prediction_row_count",
        "receipt_sha256",
        "role_ordered_objective_row_catalog_sha256",
        "score_objective_receipt_sha256",
        "summed_next_token_nll",
        "terminal_gradient_sha256",
        "terminal_values_sha256",
        "tied_readout_binding_sha256",
        "tied_readout_weight_sha256",
        "token_count",
        "valid_token_mask_sha256",
        "vocabulary_size",
        "zero_terminal_gradient_row_count",
    }
    _require_exact_fields(value, expected_fields, label="chunked NLL receipt")
    if value["full_B_L_V_logits_materialized"] is not False:
        raise ValueError("chunked NLL receipt materialization claim drifted")
    receipt = ChunkedTiedLexicalNLLReceipt(
        kernel_id=value["kernel_id"],  # type: ignore[arg-type]
        config_sha256=value["config_sha256"],  # type: ignore[arg-type]
        batch_objective_axis_sha256=value[
            "batch_objective_axis_sha256"
        ],  # type: ignore[arg-type]
        batch_ordered_objective_row_catalog_sha256=value[
            "batch_ordered_objective_row_catalog_sha256"
        ],  # type: ignore[arg-type]
        role_ordered_objective_row_catalog_sha256=value[
            "role_ordered_objective_row_catalog_sha256"
        ],  # type: ignore[arg-type]
        score_objective_receipt_sha256=value[
            "score_objective_receipt_sha256"
        ],  # type: ignore[arg-type]
        valid_token_mask_sha256=value[
            "valid_token_mask_sha256"
        ],  # type: ignore[arg-type]
        terminal_values_sha256=value["terminal_values_sha256"],  # type: ignore[arg-type]
        tied_readout_binding_sha256=value[
            "tied_readout_binding_sha256"
        ],  # type: ignore[arg-type]
        tied_readout_weight_sha256=value[
            "tied_readout_weight_sha256"
        ],  # type: ignore[arg-type]
        input_token_ids_sha256=value["input_token_ids_sha256"],  # type: ignore[arg-type]
        terminal_gradient_sha256=value[
            "terminal_gradient_sha256"
        ],  # type: ignore[arg-type]
        backend_receipt_sha256=value["backend_receipt_sha256"],  # type: ignore[arg-type]
        batch_size=value["batch_size"],  # type: ignore[arg-type]
        token_count=value["token_count"],  # type: ignore[arg-type]
        prediction_row_count=value["prediction_row_count"],  # type: ignore[arg-type]
        hidden_width=value["hidden_width"],  # type: ignore[arg-type]
        vocabulary_size=value["vocabulary_size"],  # type: ignore[arg-type]
        logit_softcap=value["logit_softcap"],  # type: ignore[arg-type]
        summed_next_token_nll=value["summed_next_token_nll"],  # type: ignore[arg-type]
        zero_terminal_gradient_row_count=value[
            "zero_terminal_gradient_row_count"
        ],  # type: ignore[arg-type]
        maximum_single_logit_matrix_scalar_count=value[
            "maximum_single_logit_matrix_scalar_count"
        ],  # type: ignore[arg-type]
        full_prediction_by_vocabulary_matrix_materialized=value[
            "full_prediction_by_vocabulary_matrix_materialized"
        ],  # type: ignore[arg-type]
        all_tokens_valid=value["all_tokens_valid"],  # type: ignore[arg-type]
        receipt_sha256=value["receipt_sha256"],  # type: ignore[arg-type]
    )
    if expected_axis is not None and (
        receipt.batch_objective_axis_sha256 != expected_axis.sha256
        or receipt.batch_ordered_objective_row_catalog_sha256
        != expected_axis.batch_ordered_objective_row_catalog_sha256
        or receipt.role_ordered_objective_row_catalog_sha256
        != expected_axis.role_ordered_objective_row_catalog_sha256
        or receipt.score_objective_receipt_sha256
        != expected_axis.role_score_objective_receipt_sha256
    ):
        raise ValueError("chunked NLL receipt disagrees with expected batch axis")
    if expected_weight is not None:
        weight_receipt = (
            expected_weight.receipt
            if isinstance(expected_weight, TiedReadoutWeightBinding)
            else expected_weight
        )
        if not isinstance(weight_receipt, TiedReadoutWeightReceipt):
            raise TypeError("expected_weight has the wrong binding type")
        if (
            receipt.tied_readout_binding_sha256
            != weight_receipt.receipt_sha256
            or receipt.tied_readout_weight_sha256
            != weight_receipt.tensor_sha256
        ):
            raise ValueError("chunked NLL receipt disagrees with expected weight")
    return receipt


@dataclass(frozen=True, slots=True)
class DenseParitySmokeReceipt:
    kernel_id: str
    chunked_config_sha256: str
    source_chunked_nll_receipt_sha256: str
    batch_objective_axis_sha256: str
    batch_ordered_objective_row_catalog_sha256: str
    score_objective_receipt_sha256: str
    tied_readout_binding_sha256: str
    tied_readout_weight_sha256: str
    terminal_values_sha256: str
    input_token_ids_sha256: str
    valid_token_mask_sha256: str
    selected_row_index_sha256: str
    selected_row_catalog_sha256: str
    selected_query_values_sha256: str
    selected_labels_sha256: str
    backend_receipt_sha256: str
    selector_id: str
    selected_prediction_row_count: int
    dense_reference_dtype: str
    logit_softcap: float | None
    loss_absolute_error: float
    gradient_max_absolute_error: float
    gradient_normalized_rmse: float
    absolute_tolerance: float
    relative_tolerance: float
    passed: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.kernel_id != PARITY_KERNEL_ID:
            raise ValueError("dense parity kernel drifted")
        for label in (
            "chunked_config_sha256",
            "source_chunked_nll_receipt_sha256",
            "batch_objective_axis_sha256",
            "batch_ordered_objective_row_catalog_sha256",
            "score_objective_receipt_sha256",
            "tied_readout_binding_sha256",
            "tied_readout_weight_sha256",
            "terminal_values_sha256",
            "input_token_ids_sha256",
            "valid_token_mask_sha256",
            "selected_row_index_sha256",
            "selected_row_catalog_sha256",
            "selected_query_values_sha256",
            "selected_labels_sha256",
            "backend_receipt_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if (
            self.chunked_config_sha256
            != ChunkedTiedLexicalNLLConfig().sha256
            or self.selector_id != DENSE_PARITY_SELECTOR_ID
            or self.selected_prediction_row_count
            != DENSE_PARITY_SELECTED_PREDICTION_ROW_COUNT
            or self.absolute_tolerance != DENSE_PARITY_ABSOLUTE_TOLERANCE
            or self.relative_tolerance != DENSE_PARITY_RELATIVE_TOLERANCE
            or self.dense_reference_dtype != "torch.float64"
            or self.passed is not True
        ):
            raise ValueError("dense parity frozen protocol drifted")
        softcap = _normalize_softcap(self.logit_softcap)
        if softcap != self.logit_softcap:
            raise ValueError("dense parity softcap was not normalized")
        for label in (
            "loss_absolute_error",
            "gradient_max_absolute_error",
            "gradient_normalized_rmse",
            "absolute_tolerance",
            "relative_tolerance",
        ):
            value = getattr(self, label)
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be a finite nonnegative float")
        if self.receipt_sha256 != _digest(
            _PARITY_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("dense parity smoke digest drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "absolute_tolerance": self.absolute_tolerance,
            "backend_receipt_sha256": self.backend_receipt_sha256,
            "batch_objective_axis_sha256": self.batch_objective_axis_sha256,
            "batch_ordered_objective_row_catalog_sha256": (
                self.batch_ordered_objective_row_catalog_sha256
            ),
            "chunked_config_sha256": self.chunked_config_sha256,
            "dense_reference_dtype": self.dense_reference_dtype,
            "gradient_max_absolute_error": self.gradient_max_absolute_error,
            "gradient_normalized_rmse": self.gradient_normalized_rmse,
            "input_token_ids_sha256": self.input_token_ids_sha256,
            "kernel_id": self.kernel_id,
            "logit_softcap": self.logit_softcap,
            "loss_absolute_error": self.loss_absolute_error,
            "passed": self.passed,
            "relative_tolerance": self.relative_tolerance,
            "score_objective_receipt_sha256": (
                self.score_objective_receipt_sha256
            ),
            "selected_labels_sha256": self.selected_labels_sha256,
            "selected_prediction_row_count": (
                self.selected_prediction_row_count
            ),
            "selected_query_values_sha256": self.selected_query_values_sha256,
            "selected_row_catalog_sha256": self.selected_row_catalog_sha256,
            "selected_row_index_sha256": self.selected_row_index_sha256,
            "selector_id": self.selector_id,
            "source_chunked_nll_receipt_sha256": (
                self.source_chunked_nll_receipt_sha256
            ),
            "terminal_values_sha256": self.terminal_values_sha256,
            "tied_readout_binding_sha256": (
                self.tied_readout_binding_sha256
            ),
            "tied_readout_weight_sha256": self.tied_readout_weight_sha256,
            "valid_token_mask_sha256": self.valid_token_mask_sha256,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {**self.payload_without_sha256(), "receipt_sha256": self.receipt_sha256}


def run_dense_full_logit_parity_smoke(
    *,
    source_result: ChunkedTiedLexicalNLLResult,
    terminal_normalized_output: Any,
    input_token_ids: Any,
    valid_token_mask: Any,
    batch_axis: BatchObjectiveAxis,
    tied_readout_weight: TiedReadoutWeightBinding,
    logit_softcap: float | None,
    config: ChunkedTiedLexicalNLLConfig = ChunkedTiedLexicalNLLConfig(),
) -> DenseParitySmokeReceipt:
    """Run the fixed float64 first-two-row dense smoke on one real batch."""

    import torch

    if not isinstance(source_result, ChunkedTiedLexicalNLLResult):
        raise TypeError("dense parity requires its source chunked NLL result")
    if not isinstance(config, ChunkedTiedLexicalNLLConfig):
        raise TypeError("dense parity requires the frozen production config")
    softcap = _normalize_softcap(logit_softcap)
    _, token_count, width = _validate_runtime_batch(
        terminal_normalized_output=terminal_normalized_output,
        input_token_ids=input_token_ids,
        valid_token_mask=valid_token_mask,
        batch_axis=batch_axis,
        tied_readout_weight=tied_readout_weight,
    )
    source_receipt = parse_chunked_tied_lexical_nll_receipt_payload(
        source_result.receipt.canonical_payload(),
        expected_axis=batch_axis,
        expected_weight=tied_readout_weight,
    )
    current_terminal_sha = tensor_stream_sha256(terminal_normalized_output)
    current_input_sha = tensor_stream_sha256(input_token_ids)
    current_mask_sha = tensor_stream_sha256(valid_token_mask)
    current_gradient_sha = tensor_stream_sha256(source_result.terminal_gradient)
    if (
        source_receipt.terminal_values_sha256 != current_terminal_sha
        or source_receipt.input_token_ids_sha256 != current_input_sha
        or source_receipt.valid_token_mask_sha256 != current_mask_sha
        or source_receipt.terminal_gradient_sha256 != current_gradient_sha
        or source_receipt.logit_softcap != softcap
        or source_receipt.backend_receipt_sha256
        != backend_receipt_sha256(config)
    ):
        raise ValueError("dense parity source batch disagrees with NLL receipt")
    flat_queries = terminal_normalized_output[:, :-1, :].reshape(-1, width)
    flat_labels = input_token_ids[:, 1:].reshape(-1)
    selected_count = DENSE_PARITY_SELECTED_PREDICTION_ROW_COUNT
    if int(flat_queries.shape[0]) < selected_count:
        raise ValueError("dense parity source has fewer than two prediction rows")
    rows = flat_queries[:selected_count]
    labels = flat_labels[:selected_count]
    chunk_loss, chunk_gradient64 = _chunked_softcap_nll_vjp_rows(
        rows,
        labels,
        tied_readout_weight.tensor,
        logit_softcap=softcap,
        prediction_position_chunk_size=(
            config.prediction_position_chunk_size
        ),
        vocabulary_chunk_size=config.vocabulary_chunk_size,
        weight_prevalidated=True,
    )
    dense_rows = rows.detach().to(torch.float64).clone().requires_grad_(True)
    dense_weight = tied_readout_weight.tensor.to(torch.float64)
    raw = torch.mm(dense_rows, dense_weight.T)
    logits = raw if softcap is None else softcap * torch.tanh(raw / softcap)
    dense_loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction="sum",
    )
    dense_gradient = torch.autograd.grad(dense_loss, dense_rows)[0]
    chunk_gradient = chunk_gradient64
    difference = chunk_gradient - dense_gradient
    loss_error = abs(chunk_loss - float(dense_loss.detach()))
    max_abs = float(difference.abs().max())
    nrmse = float(
        difference.to(torch.float64).square().mean().sqrt()
        / dense_gradient.to(torch.float64)
        .square()
        .mean()
        .sqrt()
        .clamp_min(1.0e-30)
    )
    passed = bool(
        torch.allclose(
            chunk_gradient,
            dense_gradient,
            rtol=DENSE_PARITY_RELATIVE_TOLERANCE,
            atol=DENSE_PARITY_ABSOLUTE_TOLERANCE,
        )
        and loss_error
        <= DENSE_PARITY_ABSOLUTE_TOLERANCE
        + DENSE_PARITY_RELATIVE_TOLERANCE * abs(float(dense_loss.detach()))
    )
    if not passed:
        raise RuntimeError("chunked lexical NLL failed dense full-logit parity")
    selected_indexes = tuple(range(selected_count))
    selected_rows = batch_axis.ordered_rows()[:selected_count]
    values: dict[str, object] = {
        "kernel_id": PARITY_KERNEL_ID,
        "chunked_config_sha256": config.sha256,
        "source_chunked_nll_receipt_sha256": source_receipt.receipt_sha256,
        "batch_objective_axis_sha256": batch_axis.sha256,
        "batch_ordered_objective_row_catalog_sha256": (
            batch_axis.batch_ordered_objective_row_catalog_sha256
        ),
        "score_objective_receipt_sha256": (
            batch_axis.role_score_objective_receipt_sha256
        ),
        "tied_readout_binding_sha256": (
            tied_readout_weight.receipt.receipt_sha256
        ),
        "tied_readout_weight_sha256": (
            tied_readout_weight.receipt.tensor_sha256
        ),
        "terminal_values_sha256": current_terminal_sha,
        "input_token_ids_sha256": current_input_sha,
        "valid_token_mask_sha256": current_mask_sha,
        "selected_row_index_sha256": _digest(
            _SELECTED_ROW_INDEX_DOMAIN,
            list(selected_indexes),
        ),
        "selected_row_catalog_sha256": _fresh_row_stream_sha256(selected_rows),
        "selected_query_values_sha256": tensor_stream_sha256(rows),
        "selected_labels_sha256": tensor_stream_sha256(labels),
        "backend_receipt_sha256": source_receipt.backend_receipt_sha256,
        "selector_id": DENSE_PARITY_SELECTOR_ID,
        "selected_prediction_row_count": selected_count,
        "dense_reference_dtype": str(dense_rows.dtype),
        "logit_softcap": softcap,
        "loss_absolute_error": loss_error,
        "gradient_max_absolute_error": max_abs,
        "gradient_normalized_rmse": nrmse,
        "absolute_tolerance": DENSE_PARITY_ABSOLUTE_TOLERANCE,
        "relative_tolerance": DENSE_PARITY_RELATIVE_TOLERANCE,
        "passed": True,
    }
    return _receipt_from_values(
        DenseParitySmokeReceipt,
        domain=_PARITY_DOMAIN,
        values=values,
    )


def parse_dense_parity_smoke_receipt_payload(
    value: Mapping[str, object],
    *,
    expected_source: ChunkedTiedLexicalNLLReceipt | None = None,
    expected_axis: BatchObjectiveAxis | None = None,
    expected_weight: TiedReadoutWeightBinding | TiedReadoutWeightReceipt | None = None,
) -> DenseParitySmokeReceipt:
    expected_fields = {
        "absolute_tolerance",
        "backend_receipt_sha256",
        "batch_objective_axis_sha256",
        "batch_ordered_objective_row_catalog_sha256",
        "chunked_config_sha256",
        "dense_reference_dtype",
        "gradient_max_absolute_error",
        "gradient_normalized_rmse",
        "input_token_ids_sha256",
        "kernel_id",
        "logit_softcap",
        "loss_absolute_error",
        "passed",
        "receipt_sha256",
        "relative_tolerance",
        "score_objective_receipt_sha256",
        "selected_labels_sha256",
        "selected_prediction_row_count",
        "selected_query_values_sha256",
        "selected_row_catalog_sha256",
        "selected_row_index_sha256",
        "selector_id",
        "source_chunked_nll_receipt_sha256",
        "terminal_values_sha256",
        "tied_readout_binding_sha256",
        "tied_readout_weight_sha256",
        "valid_token_mask_sha256",
    }
    _require_exact_fields(value, expected_fields, label="dense parity smoke receipt")
    receipt = DenseParitySmokeReceipt(
        kernel_id=value["kernel_id"],  # type: ignore[arg-type]
        chunked_config_sha256=value["chunked_config_sha256"],  # type: ignore[arg-type]
        source_chunked_nll_receipt_sha256=value[
            "source_chunked_nll_receipt_sha256"
        ],  # type: ignore[arg-type]
        batch_objective_axis_sha256=value[
            "batch_objective_axis_sha256"
        ],  # type: ignore[arg-type]
        batch_ordered_objective_row_catalog_sha256=value[
            "batch_ordered_objective_row_catalog_sha256"
        ],  # type: ignore[arg-type]
        score_objective_receipt_sha256=value[
            "score_objective_receipt_sha256"
        ],  # type: ignore[arg-type]
        tied_readout_binding_sha256=value[
            "tied_readout_binding_sha256"
        ],  # type: ignore[arg-type]
        tied_readout_weight_sha256=value[
            "tied_readout_weight_sha256"
        ],  # type: ignore[arg-type]
        terminal_values_sha256=value["terminal_values_sha256"],  # type: ignore[arg-type]
        input_token_ids_sha256=value["input_token_ids_sha256"],  # type: ignore[arg-type]
        valid_token_mask_sha256=value[
            "valid_token_mask_sha256"
        ],  # type: ignore[arg-type]
        selected_row_index_sha256=value[
            "selected_row_index_sha256"
        ],  # type: ignore[arg-type]
        selected_row_catalog_sha256=value[
            "selected_row_catalog_sha256"
        ],  # type: ignore[arg-type]
        selected_query_values_sha256=value[
            "selected_query_values_sha256"
        ],  # type: ignore[arg-type]
        selected_labels_sha256=value["selected_labels_sha256"],  # type: ignore[arg-type]
        backend_receipt_sha256=value["backend_receipt_sha256"],  # type: ignore[arg-type]
        selector_id=value["selector_id"],  # type: ignore[arg-type]
        selected_prediction_row_count=value[
            "selected_prediction_row_count"
        ],  # type: ignore[arg-type]
        dense_reference_dtype=value["dense_reference_dtype"],  # type: ignore[arg-type]
        logit_softcap=value["logit_softcap"],  # type: ignore[arg-type]
        loss_absolute_error=value["loss_absolute_error"],  # type: ignore[arg-type]
        gradient_max_absolute_error=value[
            "gradient_max_absolute_error"
        ],  # type: ignore[arg-type]
        gradient_normalized_rmse=value[
            "gradient_normalized_rmse"
        ],  # type: ignore[arg-type]
        absolute_tolerance=value["absolute_tolerance"],  # type: ignore[arg-type]
        relative_tolerance=value["relative_tolerance"],  # type: ignore[arg-type]
        passed=value["passed"],  # type: ignore[arg-type]
        receipt_sha256=value["receipt_sha256"],  # type: ignore[arg-type]
    )
    if expected_source is not None and (
        receipt.source_chunked_nll_receipt_sha256
        != expected_source.receipt_sha256
        or receipt.terminal_values_sha256
        != expected_source.terminal_values_sha256
        or receipt.input_token_ids_sha256
        != expected_source.input_token_ids_sha256
        or receipt.valid_token_mask_sha256
        != expected_source.valid_token_mask_sha256
        or receipt.backend_receipt_sha256
        != expected_source.backend_receipt_sha256
        or receipt.logit_softcap != expected_source.logit_softcap
    ):
        raise ValueError("dense parity receipt disagrees with source NLL")
    if expected_axis is not None and (
        receipt.batch_objective_axis_sha256 != expected_axis.sha256
        or receipt.batch_ordered_objective_row_catalog_sha256
        != expected_axis.batch_ordered_objective_row_catalog_sha256
        or receipt.score_objective_receipt_sha256
        != expected_axis.role_score_objective_receipt_sha256
    ):
        raise ValueError("dense parity receipt disagrees with batch axis")
    if expected_weight is not None:
        weight_receipt = (
            expected_weight.receipt
            if isinstance(expected_weight, TiedReadoutWeightBinding)
            else expected_weight
        )
        if not isinstance(weight_receipt, TiedReadoutWeightReceipt):
            raise TypeError("expected_weight has the wrong binding type")
        if (
            receipt.tied_readout_binding_sha256
            != weight_receipt.receipt_sha256
            or receipt.tied_readout_weight_sha256
            != weight_receipt.tensor_sha256
        ):
            raise ValueError("dense parity receipt disagrees with tied weight")
    return receipt


def chunked_tied_lexical_nll_kernel_receipt(
    config: ChunkedTiedLexicalNLLConfig = ChunkedTiedLexicalNLLConfig(),
) -> dict[str, object]:
    if not isinstance(config, ChunkedTiedLexicalNLLConfig):
        raise TypeError("kernel receipt requires the frozen production config")
    payload: dict[str, object] = {
        "backend_receipt_sha256": backend_receipt_sha256(config),
        "config": config.payload(),
        "config_sha256": config.sha256,
        "full_B_L_V_logits_materialized": False,
        "kernel_id": KERNEL_ID,
        "maximum_single_logit_matrix_float64_bytes": (
            config.maximum_single_logit_matrix_scalar_count * 8
        ),
        "maximum_single_logit_matrix_scalar_count": (
            config.maximum_single_logit_matrix_scalar_count
        ),
        "padding_rows_forbidden": True,
        "prediction_rows_derived_as_exact_q_with_t_equal_q_plus_1": True,
        "production_chunks_frozen": True,
        "raw_logits_or_gradients_persisted": False,
        "terminal_no_label_row_is_exact_zero": True,
    }
    return {**payload, "receipt_sha256": _digest(_KERNEL_RECEIPT_DOMAIN, payload)}


def parse_chunked_tied_lexical_nll_kernel_receipt_payload(
    value: Mapping[str, object],
) -> dict[str, object]:
    expected = chunked_tied_lexical_nll_kernel_receipt()
    _require_exact_fields(value, set(expected), label="chunked NLL kernel receipt")
    if dict(value) != expected:
        raise ValueError("chunked NLL kernel receipt drifted")
    return dict(expected)


__all__ = [
    "BatchObjectiveAxis",
    "BatchObjectiveExample",
    "ChunkedTiedLexicalNLLConfig",
    "ChunkedTiedLexicalNLLReceipt",
    "ChunkedTiedLexicalNLLResult",
    "DENSE_PARITY_ABSOLUTE_TOLERANCE",
    "DENSE_PARITY_RELATIVE_TOLERANCE",
    "DENSE_PARITY_SELECTED_PREDICTION_ROW_COUNT",
    "DENSE_PARITY_SELECTOR_ID",
    "DenseParitySmokeReceipt",
    "EXECUTABLE_PANEL_ROLES",
    "FIXTURE_KERNEL_ID",
    "FixtureChunkedTiedLexicalNLLConfig",
    "KERNEL_ID",
    "PARITY_KERNEL_ID",
    "PRODUCTION_HASH_ROW_CHUNK_SIZE",
    "PRODUCTION_PREDICTION_POSITION_CHUNK_SIZE",
    "PRODUCTION_VOCABULARY_CHUNK_SIZE",
    "TiedReadoutWeightBinding",
    "TiedReadoutWeightReceipt",
    "backend_receipt_sha256",
    "bind_tied_readout_weight",
    "chunked_tied_lexical_nll_kernel_receipt",
    "compute_chunked_tied_lexical_nll_vjp",
    "fixture_chunked_softcap_nll_vjp_rows",
    "objective_row_stream_sha256",
    "parse_batch_objective_axis_payload",
    "parse_chunked_tied_lexical_nll_config_payload",
    "parse_chunked_tied_lexical_nll_kernel_receipt_payload",
    "parse_chunked_tied_lexical_nll_receipt_payload",
    "parse_dense_parity_smoke_receipt_payload",
    "parse_tied_readout_weight_receipt_payload",
    "run_dense_full_logit_parity_smoke",
    "tensor_stream_sha256",
]
