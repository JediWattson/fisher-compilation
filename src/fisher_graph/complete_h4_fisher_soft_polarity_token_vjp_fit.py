"""Pure token-VJP fitting primitives for a two-coordinate soft-polarity field.

This module is deliberately policy neutral.  It contracts already-computed
token loss pullbacks with two already-computed local H4 tangents, records the
result as tamper-evident training evidence, and exposes a damped mean-KL
natural/OPG solve plus a squared-KL residual-GN control.  It does not choose
prompts, feature ladders, damping ladders, trust ladders, or serving policy.

Raw token losses, token pullbacks, and contracted token gradients are kept in
memory only.  Metadata contains tensor hashes and scalar summaries, never raw
tensor values.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor


__all__ = [
    "SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER",
    "SOFT_POLARITY_TOKEN_VJP_SOLVER_KINDS",
    "SOFT_POLARITY_TOKEN_VJP_TAU_FLOOR",
    "SoftPolarityTokenVJPAggregate",
    "SoftPolarityTokenVJPFit",
    "SoftPolarityTokenVJPFitArguments",
    "SoftPolarityTokenVJPNaturalDirection",
    "SoftPolarityTokenVJPPromptRecord",
    "aggregate_soft_polarity_token_vjp_records",
    "build_soft_polarity_token_vjp_natural_direction",
    "contract_soft_polarity_token_h4_vjps",
    "fit_soft_polarity_token_vjp_step",
]


SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER = (
    "field_bias",
    "field_slope",
)
SOFT_POLARITY_TOKEN_VJP_SOLVER_KINDS = (
    "mean_kl_natural_opg",
    "squared_kl_residual_gn",
)
SOFT_POLARITY_TOKEN_VJP_TAU_FLOOR = 2.0**-24

_PROMPT_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-prompt:v1\0"
_AGGREGATE_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-aggregate:v1\0"
_FIT_ARGUMENTS_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-fit-args:v1\0"
_FIT_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-fit:v1\0"
_NATURAL_DIRECTION_DOMAIN = (
    b"fisher-graph:soft-polarity-token-vjp-natural-direction:v1\0"
)
_TENSOR_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_float(value: object, *, label: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _exact_f64_tensor(value: object, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.float64
        or value.layout != torch.strided
        or value.ndim != ndim
        or 0 in value.shape
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{label} must be a finite nonempty strided float64 tensor"
        )
    return value.detach().to(device="cpu").clone().contiguous()


def _tensor_sha256(value: Tensor) -> str:
    tensor = _exact_f64_tensor(value, label="hashed tensor", ndim=value.ndim)
    payload = tensor.numpy().astype("<f8", copy=False).tobytes(order="C")
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + _canonical_json_bytes(
            {
                "dtype": "float64-little-endian",
                "shape": tuple(tensor.shape),
            }
        )
        + payload
    ).hexdigest()


def _hash_tuple(values: object, *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a nonempty sequence of SHA-256 values")
    try:
        selected = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            f"{label} must be a nonempty sequence of SHA-256 values"
        ) from error
    if not selected:
        raise ValueError(f"{label} must be nonempty")
    checked = tuple(
        _require_sha256(value, label=f"{label} entry") for value in selected
    )
    if checked != tuple(sorted(set(checked))):
        raise ValueError(f"{label} must be sorted and unique")
    return checked


def _string_tuple(values: object, *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    try:
        selected = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{label} must be a sequence") from error
    checked = tuple(
        _identifier(value, label=f"{label} entry") for value in selected
    )
    if not checked or checked != tuple(sorted(set(checked))):
        raise ValueError(f"{label} must be nonempty, sorted, and unique")
    return checked


def _count_tuple(
    values: object,
    *,
    label: str,
) -> tuple[tuple[str, int], ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    try:
        selected = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{label} must be a sequence") from error
    checked: list[tuple[str, int]] = []
    for item in selected:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{label} entries must be (family_id, count) pairs")
        family_id = _identifier(item[0], label=f"{label} family_id")
        count = item[1]
        if type(count) is not int or count <= 0:
            raise ValueError(f"{label} counts must be positive integers")
        checked.append((family_id, count))
    result = tuple(checked)
    if not result or result != tuple(sorted(result)):
        raise ValueError(f"{label} must be nonempty and sorted")
    if len({family_id for family_id, _ in result}) != len(result):
        raise ValueError(f"{label} family IDs must be unique")
    return result


def _validate_contraction_inputs(
    *,
    token_h4_gradients: Tensor,
    local_h4_tangents: Tensor,
    canonical_support_mask: Tensor,
    supervised_indices: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if (
        not isinstance(token_h4_gradients, Tensor)
        or token_h4_gradients.dtype != torch.float64
        or token_h4_gradients.layout != torch.strided
        or token_h4_gradients.ndim != 4
        or 0 in token_h4_gradients.shape
        or not bool(torch.isfinite(token_h4_gradients).all())
    ):
        raise ValueError(
            "token_h4_gradients must be finite nonempty strided float64 [T,B,S,W]"
        )
    if (
        not isinstance(local_h4_tangents, Tensor)
        or local_h4_tangents.dtype != torch.float64
        or local_h4_tangents.layout != torch.strided
        or local_h4_tangents.ndim != 4
        or 0 in local_h4_tangents.shape
        or not bool(torch.isfinite(local_h4_tangents).all())
    ):
        raise ValueError(
            "local_h4_tangents must be finite nonempty strided float64 [2,B,S,W]"
        )
    token_count, batch_count, sequence_count, width = token_h4_gradients.shape
    if (
        local_h4_tangents.shape
        != (
            len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER),
            batch_count,
            sequence_count,
            width,
        )
        or local_h4_tangents.device != token_h4_gradients.device
    ):
        raise ValueError("token VJP and local tangent geometry differs")
    if (
        not isinstance(canonical_support_mask, Tensor)
        or canonical_support_mask.dtype != torch.bool
        or canonical_support_mask.layout != torch.strided
        or canonical_support_mask.shape != (batch_count, sequence_count)
        or not bool(canonical_support_mask.any())
    ):
        raise ValueError(
            "canonical_support_mask must be nonempty bool [B,S] support"
        )
    if (
        not isinstance(supervised_indices, Tensor)
        or supervised_indices.dtype != torch.int64
        or supervised_indices.layout != torch.strided
        or supervised_indices.shape != (token_count, 2)
    ):
        raise ValueError("supervised_indices must be canonical int64 [T,2]")

    indices = supervised_indices.detach().to(device="cpu").clone().contiguous()
    batches = indices[:, 0]
    positions = indices[:, 1]
    if bool((batches < 0).any()) or bool((batches >= batch_count).any()):
        raise ValueError("supervised_indices contains an out-of-range batch")
    if bool((positions < 0).any()) or bool((positions >= sequence_count).any()):
        raise ValueError("supervised_indices contains an out-of-range position")
    flat = batches * sequence_count + positions
    if token_count > 1 and not bool((flat[1:] > flat[:-1]).all()):
        raise ValueError(
            "supervised_indices must be strictly increasing in batch-major order"
        )

    support = (
        canonical_support_mask.detach()
        .to(device=token_h4_gradients.device)
        .clone()
        .contiguous()
    )
    device_batches = batches.to(device=support.device)
    device_positions = positions.to(device=support.device)
    if not bool(support[device_batches, device_positions].all()):
        raise ValueError("every supervised index must belong to canonical support")
    return token_h4_gradients, local_h4_tangents, support, indices


def contract_soft_polarity_token_h4_vjps(
    *,
    token_h4_gradients: Tensor,
    local_h4_tangents: Tensor,
    canonical_support_mask: Tensor,
    supervised_indices: Tensor,
) -> Tensor:
    """Contract exact token H4 pullbacks with two causal local tangents.

    The returned matrix is

    ``Q[t,p] = sum(b,s,w, dL_t/dH4[b,s,w] * dH4[b,s,w]/dp)``.

    Only canonical support in the supervised token's own batch at positions
    no later than that token is admissible.  Tangent values outside canonical
    support and nonzero products in another batch or a future position are
    rejected rather than silently discarded.
    """

    gradients, tangents, support, indices = _validate_contraction_inputs(
        token_h4_gradients=token_h4_gradients,
        local_h4_tangents=local_h4_tangents,
        canonical_support_mask=canonical_support_mask,
        supervised_indices=supervised_indices,
    )
    if bool((tangents * (~support).unsqueeze(0).unsqueeze(-1)).ne(0.0).any()):
        raise ValueError("local H4 tangents leak outside canonical support")

    token_count, batch_count, sequence_count, _ = gradients.shape
    result = torch.empty(
        (token_count, len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER)),
        dtype=torch.float64,
        device=gradients.device,
    )
    sequence_axis = torch.arange(sequence_count, device=gradients.device)
    batch_axis = torch.arange(batch_count, device=gradients.device)
    for token_index in range(token_count):
        batch = int(indices[token_index, 0])
        position = int(indices[token_index, 1])
        allowed = (
            support
            & (batch_axis[:, None] == batch)
            & (sequence_axis[None, :] <= position)
        )
        products = gradients[token_index].unsqueeze(0) * tangents
        if bool(products[:, ~allowed, :].ne(0.0).any()):
            raise ValueError(
                "token/tangent contraction contains future or cross-batch leakage"
            )
        result[token_index] = products[:, allowed, :].sum(dim=(1, 2))
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("token VJP contraction produced nonfinite values")
    return result.detach().to(device="cpu").clone().contiguous()


@dataclass(frozen=True, slots=True)
class SoftPolarityTokenVJPPromptRecord:
    """Immutable, tamper-evident per-prompt token fitting evidence."""

    feature_id: str
    family_id: str
    example_id: str
    reference_b: float
    reference_a: float
    derivative_convention: str
    derivative_artifact_sha256s: tuple[str, ...]
    token_teacher_kl: Tensor = field(repr=False)
    token_parameter_gradients: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        feature_id = _identifier(self.feature_id, label="feature_id")
        family_id = _identifier(self.family_id, label="family_id")
        example_id = _identifier(self.example_id, label="example_id")
        reference_b = _strict_float(self.reference_b, label="reference_b")
        reference_a = _strict_float(self.reference_a, label="reference_a")
        convention = _identifier(
            self.derivative_convention, label="derivative_convention"
        )
        artifacts = _hash_tuple(
            self.derivative_artifact_sha256s,
            label="derivative_artifact_sha256s",
        )
        teacher_kl = _exact_f64_tensor(
            self.token_teacher_kl, label="token_teacher_kl", ndim=1
        )
        gradients = _exact_f64_tensor(
            self.token_parameter_gradients,
            label="token_parameter_gradients",
            ndim=2,
        )
        if (
            gradients.shape
            != (
                teacher_kl.numel(),
                len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER),
            )
            or bool((teacher_kl < 0.0).any())
        ):
            raise ValueError("prompt token fitting evidence geometry differs")
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "reference_b", reference_b)
        object.__setattr__(self, "reference_a", reference_a)
        object.__setattr__(self, "derivative_convention", convention)
        object.__setattr__(self, "derivative_artifact_sha256s", artifacts)
        object.__setattr__(self, "token_teacher_kl", teacher_kl)
        object.__setattr__(self, "token_parameter_gradients", gradients)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_PROMPT_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_token_count(self) -> int:
        return int(self.token_teacher_kl.numel())

    def token_teacher_kl_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.token_teacher_kl.clone().contiguous()

    def token_parameter_gradients_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.token_parameter_gradients.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "feature_id": self.feature_id,
            "family_id": self.family_id,
            "example_id": self.example_id,
            "reference_b": self.reference_b,
            "reference_b_hex": self.reference_b.hex(),
            "reference_a": self.reference_a,
            "reference_a_hex": self.reference_a.hex(),
            "parameter_order": SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER,
            "derivative_convention": self.derivative_convention,
            "derivative_artifact_sha256s": self.derivative_artifact_sha256s,
            "supervised_token_count": self.supervised_token_count,
            "token_teacher_kl_shape": tuple(self.token_teacher_kl.shape),
            "token_teacher_kl_sha256": _tensor_sha256(self.token_teacher_kl),
            "token_parameter_gradients_shape": tuple(
                self.token_parameter_gradients.shape
            ),
            "token_parameter_gradients_sha256": _tensor_sha256(
                self.token_parameter_gradients
            ),
            "mean_token_teacher_kl": float(self.token_teacher_kl.mean()),
            "raw_token_teacher_kl_serialized": False,
            "raw_token_parameter_gradients_serialized": False,
            "raw_tensors_serialized": False,
            "training_only": True,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            self.token_teacher_kl.dtype != torch.float64
            or self.token_teacher_kl.device.type != "cpu"
            or self.token_parameter_gradients.dtype != torch.float64
            or self.token_parameter_gradients.device.type != "cpu"
            or self.token_parameter_gradients.shape
            != (
                self.supervised_token_count,
                len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER),
            )
            or bool((self.token_teacher_kl < 0.0).any())
            or not bool(torch.isfinite(self.token_teacher_kl).all())
            or not bool(torch.isfinite(self.token_parameter_gradients).all())
        ):
            raise RuntimeError("soft-polarity token VJP prompt tensors drifted")
        expected = _sha256(
            _PROMPT_DOMAIN, self.metadata(include_artifact=False)
        )
        if expected != _require_sha256(
            self.artifact_sha256, label="prompt record artifact"
        ):
            raise RuntimeError("soft-polarity token VJP prompt record drifted")


@dataclass(frozen=True, slots=True)
class SoftPolarityTokenVJPAggregate:
    """Family-balanced sufficient statistics for one held-family fit."""

    feature_id: str
    held_family_id: str
    reference_b: float
    reference_a: float
    derivative_convention: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    prompt_record_sha256s: tuple[str, ...]
    family_prompt_counts: tuple[tuple[str, int], ...]
    family_token_counts: tuple[tuple[str, int], ...]
    prompt_count: int
    token_count: int
    mean_token_teacher_kl: float
    mean_parameter_gradient: Tensor = field(repr=False)
    residual_gradient_c: Tensor = field(repr=False)
    gradient_gram: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        feature_id = _identifier(self.feature_id, label="aggregate feature_id")
        held_family = _identifier(
            self.held_family_id, label="aggregate held_family_id"
        )
        reference_b = _strict_float(self.reference_b, label="aggregate reference_b")
        reference_a = _strict_float(self.reference_a, label="aggregate reference_a")
        convention = _identifier(
            self.derivative_convention,
            label="aggregate derivative_convention",
        )
        family_ids = _string_tuple(
            self.training_family_ids, label="training_family_ids"
        )
        example_ids = _string_tuple(
            self.training_example_ids, label="training_example_ids"
        )
        record_hashes = _hash_tuple(
            self.prompt_record_sha256s, label="prompt_record_sha256s"
        )
        prompt_counts = _count_tuple(
            self.family_prompt_counts, label="family_prompt_counts"
        )
        token_counts = _count_tuple(
            self.family_token_counts, label="family_token_counts"
        )
        if held_family in family_ids:
            raise ValueError("held family may not appear in training families")
        if (
            tuple(name for name, _ in prompt_counts) != family_ids
            or tuple(name for name, _ in token_counts) != family_ids
        ):
            raise ValueError("aggregate family count keys differ")
        if (
            type(self.prompt_count) is not int
            or self.prompt_count <= 0
            or self.prompt_count != sum(value for _, value in prompt_counts)
            or self.prompt_count != len(example_ids)
            or self.prompt_count != len(record_hashes)
        ):
            raise ValueError("aggregate prompt count differs")
        if (
            type(self.token_count) is not int
            or self.token_count <= 0
            or self.token_count != sum(value for _, value in token_counts)
        ):
            raise ValueError("aggregate token count differs")
        mean_kl = _strict_float(
            self.mean_token_teacher_kl,
            label="aggregate mean_token_teacher_kl",
            nonnegative=True,
        )
        mean_gradient = _exact_f64_tensor(
            self.mean_parameter_gradient,
            label="aggregate mean_parameter_gradient",
            ndim=1,
        )
        c = _exact_f64_tensor(
            self.residual_gradient_c,
            label="aggregate residual_gradient_c",
            ndim=1,
        )
        gram = _exact_f64_tensor(
            self.gradient_gram,
            label="aggregate gradient_gram",
            ndim=2,
        )
        parameter_count = len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER)
        if (
            mean_gradient.shape != (parameter_count,)
            or c.shape != (parameter_count,)
            or gram.shape != (parameter_count, parameter_count)
            or not torch.equal(gram, gram.T)
        ):
            raise ValueError("aggregate sufficient-statistic geometry differs")
        tolerance = (
            torch.finfo(torch.float64).eps
            * max(1.0, float(gram.abs().max()))
            * 32.0
        )
        if float(torch.linalg.eigvalsh(gram).min()) < -tolerance:
            raise ValueError("aggregate gradient Gram must be positive semidefinite")

        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "held_family_id", held_family)
        object.__setattr__(self, "reference_b", reference_b)
        object.__setattr__(self, "reference_a", reference_a)
        object.__setattr__(self, "derivative_convention", convention)
        object.__setattr__(self, "training_family_ids", family_ids)
        object.__setattr__(self, "training_example_ids", example_ids)
        object.__setattr__(self, "prompt_record_sha256s", record_hashes)
        object.__setattr__(self, "family_prompt_counts", prompt_counts)
        object.__setattr__(self, "family_token_counts", token_counts)
        object.__setattr__(self, "mean_token_teacher_kl", mean_kl)
        object.__setattr__(self, "mean_parameter_gradient", mean_gradient)
        object.__setattr__(self, "residual_gradient_c", c)
        object.__setattr__(self, "gradient_gram", gram)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_AGGREGATE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def mean_parameter_gradient_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.mean_parameter_gradient.clone().contiguous()

    def residual_gradient_c_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.residual_gradient_c.clone().contiguous()

    def gradient_gram_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.gradient_gram.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "feature_id": self.feature_id,
            "held_family_id": self.held_family_id,
            "reference_b": self.reference_b,
            "reference_b_hex": self.reference_b.hex(),
            "reference_a": self.reference_a,
            "reference_a_hex": self.reference_a.hex(),
            "parameter_order": SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER,
            "derivative_convention": self.derivative_convention,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "prompt_record_sha256s": self.prompt_record_sha256s,
            "family_prompt_counts": self.family_prompt_counts,
            "family_token_counts": self.family_token_counts,
            "family_weighting": "families_equal",
            "prompt_weighting_within_family": "prompts_equal",
            "token_weighting_within_prompt": "tokens_equal",
            "prompt_count": self.prompt_count,
            "token_count": self.token_count,
            "mean_token_teacher_kl": self.mean_token_teacher_kl,
            "mean_token_teacher_kl_hex": self.mean_token_teacher_kl.hex(),
            "mean_parameter_gradient_shape": tuple(
                self.mean_parameter_gradient.shape
            ),
            "mean_parameter_gradient_sha256": _tensor_sha256(
                self.mean_parameter_gradient
            ),
            "residual_gradient_c_shape": tuple(self.residual_gradient_c.shape),
            "residual_gradient_c_sha256": _tensor_sha256(
                self.residual_gradient_c
            ),
            "gradient_gram_shape": tuple(self.gradient_gram.shape),
            "gradient_gram_sha256": _tensor_sha256(self.gradient_gram),
            "residual_gradient_definition": "mean_token_Q_times_token_KL",
            "mean_parameter_gradient_definition": "mean_token_Q",
            "gradient_gram_definition": "mean_token_outer_product_Q_Q",
            "raw_prompt_tensors_serialized": False,
            "raw_sufficient_statistics_serialized": False,
            "raw_tensors_serialized": False,
            "training_only": True,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        parameter_count = len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER)
        if (
            self.mean_parameter_gradient.dtype != torch.float64
            or self.mean_parameter_gradient.device.type != "cpu"
            or self.mean_parameter_gradient.shape != (parameter_count,)
            or self.residual_gradient_c.dtype != torch.float64
            or self.residual_gradient_c.device.type != "cpu"
            or self.residual_gradient_c.shape != (parameter_count,)
            or self.gradient_gram.dtype != torch.float64
            or self.gradient_gram.device.type != "cpu"
            or self.gradient_gram.shape != (parameter_count, parameter_count)
            or not torch.equal(self.gradient_gram, self.gradient_gram.T)
            or not bool(torch.isfinite(self.mean_parameter_gradient).all())
            or not bool(torch.isfinite(self.residual_gradient_c).all())
            or not bool(torch.isfinite(self.gradient_gram).all())
        ):
            raise RuntimeError("soft-polarity token VJP aggregate tensors drifted")
        expected = _sha256(
            _AGGREGATE_DOMAIN, self.metadata(include_artifact=False)
        )
        if expected != _require_sha256(
            self.artifact_sha256, label="aggregate artifact"
        ):
            raise RuntimeError("soft-polarity token VJP aggregate drifted")


def aggregate_soft_polarity_token_vjp_records(
    records: Iterable[SoftPolarityTokenVJPPromptRecord],
    *,
    held_family_id: str,
) -> SoftPolarityTokenVJPAggregate:
    """Build equal-family/equal-prompt/equal-token sufficient statistics."""

    held = _identifier(held_family_id, label="held_family_id")
    selected = tuple(records)
    if not selected:
        raise ValueError("at least one prompt record is required")
    if any(not isinstance(record, SoftPolarityTokenVJPPromptRecord) for record in selected):
        raise TypeError("records must contain token VJP prompt records")
    for record in selected:
        record.validate_integrity()
    ordered = tuple(sorted(selected, key=lambda item: (item.family_id, item.example_id)))
    if len({record.example_id for record in ordered}) != len(ordered):
        raise ValueError("training example IDs must be globally unique")
    if any(record.family_id == held for record in ordered):
        raise ValueError("held-family records may not enter fit aggregation")

    first = ordered[0]
    for record in ordered[1:]:
        if (
            record.feature_id != first.feature_id
            or record.reference_b.hex() != first.reference_b.hex()
            or record.reference_a.hex() != first.reference_a.hex()
            or record.derivative_convention != first.derivative_convention
        ):
            raise ValueError("prompt record fit identity differs")

    by_family: dict[str, list[SoftPolarityTokenVJPPromptRecord]] = defaultdict(list)
    for record in ordered:
        by_family[record.family_id].append(record)

    family_mean_gradients: list[Tensor] = []
    family_residual_gradients: list[Tensor] = []
    family_grams: list[Tensor] = []
    family_mean_kls: list[float] = []
    prompt_counts: list[tuple[str, int]] = []
    token_counts: list[tuple[str, int]] = []
    for family_id in sorted(by_family):
        family_records = by_family[family_id]
        prompt_mean_gradients: list[Tensor] = []
        prompt_residual_gradients: list[Tensor] = []
        prompt_grams: list[Tensor] = []
        prompt_mean_kls: list[float] = []
        for record in family_records:
            q = record.token_parameter_gradients
            kl = record.token_teacher_kl
            prompt_mean_gradients.append(q.mean(dim=0))
            prompt_residual_gradients.append((q * kl[:, None]).mean(dim=0))
            prompt_grams.append((q.T @ q) / q.shape[0])
            prompt_mean_kls.append(float(kl.mean()))
        family_mean_gradients.append(torch.stack(prompt_mean_gradients).mean(dim=0))
        family_residual_gradients.append(
            torch.stack(prompt_residual_gradients).mean(dim=0)
        )
        family_grams.append(torch.stack(prompt_grams).mean(dim=0))
        family_mean_kls.append(sum(prompt_mean_kls) / len(prompt_mean_kls))
        prompt_counts.append((family_id, len(family_records)))
        token_counts.append(
            (family_id, sum(record.supervised_token_count for record in family_records))
        )

    return SoftPolarityTokenVJPAggregate(
        feature_id=first.feature_id,
        held_family_id=held,
        reference_b=first.reference_b,
        reference_a=first.reference_a,
        derivative_convention=first.derivative_convention,
        training_family_ids=tuple(sorted(by_family)),
        training_example_ids=tuple(sorted(record.example_id for record in ordered)),
        prompt_record_sha256s=tuple(sorted(record.artifact_sha256 for record in ordered)),
        family_prompt_counts=tuple(prompt_counts),
        family_token_counts=tuple(token_counts),
        prompt_count=len(ordered),
        token_count=sum(record.supervised_token_count for record in ordered),
        mean_token_teacher_kl=sum(family_mean_kls) / len(family_mean_kls),
        mean_parameter_gradient=torch.stack(family_mean_gradients).mean(dim=0),
        residual_gradient_c=torch.stack(family_residual_gradients).mean(dim=0),
        gradient_gram=torch.stack(family_grams).mean(dim=0),
    )


@dataclass(frozen=True, slots=True)
class SoftPolarityTokenVJPNaturalDirection:
    """Frozen trace-ridge, unit-L-infinity natural-direction authority."""

    aggregate_artifact_sha256: str
    feature_id: str
    held_family_id: str
    reference_b: float
    reference_a: float
    ridge_multiplier: float
    gradient_gram_trace: float
    tau: float
    damping: float
    mean_parameter_gradient: Tensor = field(repr=False)
    gradient_gram: Tensor = field(repr=False)
    raw_direction: Tensor = field(repr=False)
    direction: Tensor = field(repr=False)
    direction_linf: float
    predicted_derivative: float
    no_op: bool
    no_op_reason: str | None
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        aggregate_hash = _require_sha256(
            self.aggregate_artifact_sha256,
            label="natural direction aggregate artifact",
        )
        feature_id = _identifier(
            self.feature_id, label="natural direction feature_id"
        )
        held_family_id = _identifier(
            self.held_family_id, label="natural direction held_family_id"
        )
        reference_b = _strict_float(
            self.reference_b, label="natural direction reference_b"
        )
        reference_a = _strict_float(
            self.reference_a, label="natural direction reference_a"
        )
        ridge = _strict_float(
            self.ridge_multiplier,
            label="natural direction ridge_multiplier",
            nonnegative=True,
        )
        gram_trace = _strict_float(
            self.gradient_gram_trace,
            label="natural direction gradient_gram_trace",
            nonnegative=True,
        )
        tau = _strict_float(
            self.tau, label="natural direction tau", nonnegative=True
        )
        damping = _strict_float(
            self.damping,
            label="natural direction damping",
            nonnegative=True,
        )
        mean_gradient = _exact_f64_tensor(
            self.mean_parameter_gradient,
            label="natural direction mean_parameter_gradient",
            ndim=1,
        )
        gram = _exact_f64_tensor(
            self.gradient_gram,
            label="natural direction gradient_gram",
            ndim=2,
        )
        raw = _exact_f64_tensor(
            self.raw_direction,
            label="natural direction raw_direction",
            ndim=1,
        )
        direction = _exact_f64_tensor(
            self.direction,
            label="natural direction direction",
            ndim=1,
        )
        parameter_count = len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER)
        if (
            mean_gradient.shape != (parameter_count,)
            or gram.shape != (parameter_count, parameter_count)
            or raw.shape != (parameter_count,)
            or direction.shape != (parameter_count,)
            or not torch.equal(gram, gram.T)
        ):
            raise ValueError("natural direction tensor geometry differs")
        observed_trace = float(torch.trace(gram))
        expected_tau = max(
            observed_trace / float(parameter_count),
            SOFT_POLARITY_TOKEN_VJP_TAU_FLOOR,
        )
        if (
            gram_trace.hex() != observed_trace.hex()
            or tau.hex() != expected_tau.hex()
            or damping.hex() != (ridge * tau).hex()
        ):
            raise ValueError("natural direction trace-scaled damping differs")
        direction_linf = _strict_float(
            self.direction_linf,
            label="natural direction L-infinity norm",
            nonnegative=True,
        )
        observed_linf = float(direction.abs().max())
        predicted = _strict_float(
            self.predicted_derivative,
            label="natural direction predicted_derivative",
        )
        if direction_linf.hex() != observed_linf.hex():
            raise ValueError("natural direction L-infinity norm differs")
        if type(self.no_op) is not bool:
            raise ValueError("natural direction no_op must be bool")
        if self.no_op:
            reason = _identifier(
                self.no_op_reason, label="natural direction no_op_reason"
            )
            if (
                not torch.equal(direction, torch.zeros_like(direction))
                or direction_linf != 0.0
                or predicted != 0.0
            ):
                raise ValueError("natural direction no-op must be exact zero")
        else:
            if self.no_op_reason is not None:
                raise ValueError(
                    "active natural direction may not have a no-op reason"
                )
            reason = None
            raw_linf = float(raw.abs().max())
            if raw_linf <= 0.0:
                raise ValueError("active natural direction must have a raw solve")
            expected_direction = (raw / raw_linf).contiguous()
            maximum_index = int(torch.argmax(raw.abs()))
            expected_direction[maximum_index] = math.copysign(
                1.0, float(raw[maximum_index])
            )
            expected_predicted = float(mean_gradient @ direction)
            if (
                direction_linf.hex() != 1.0.hex()
                or not torch.equal(direction, expected_direction)
                or predicted.hex() != expected_predicted.hex()
                or predicted >= 0.0
            ):
                raise ValueError(
                    "active natural direction must be normalized and descending"
                )

        object.__setattr__(self, "aggregate_artifact_sha256", aggregate_hash)
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "held_family_id", held_family_id)
        object.__setattr__(self, "reference_b", reference_b)
        object.__setattr__(self, "reference_a", reference_a)
        object.__setattr__(self, "ridge_multiplier", ridge)
        object.__setattr__(self, "gradient_gram_trace", gram_trace)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "mean_parameter_gradient", mean_gradient)
        object.__setattr__(self, "gradient_gram", gram)
        object.__setattr__(self, "raw_direction", raw)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "direction_linf", direction_linf)
        object.__setattr__(self, "predicted_derivative", predicted)
        object.__setattr__(self, "no_op_reason", reason)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(
                _NATURAL_DIRECTION_DOMAIN,
                self.metadata(include_artifact=False),
            ),
        )
        self.validate_integrity()

    @property
    def direction_b(self) -> float:
        return float(self.direction[0])

    @property
    def direction_a(self) -> float:
        return float(self.direction[1])

    def raw_direction_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.raw_direction.clone().contiguous()

    def direction_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.direction.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "method": "mean_kl_natural_opg_trace_scaled_ridge_linf_direction",
            "parameter_order": SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER,
            "feature_id": self.feature_id,
            "held_family_id": self.held_family_id,
            "reference_b": self.reference_b,
            "reference_a": self.reference_a,
            "aggregate_artifact_sha256": self.aggregate_artifact_sha256,
            "ridge_multiplier": self.ridge_multiplier,
            "gradient_gram_trace": self.gradient_gram_trace,
            "tau": self.tau,
            "damping": self.damping,
            "direction_b": self.direction_b,
            "direction_a": self.direction_a,
            "direction_linf": self.direction_linf,
            "predicted_derivative": self.predicted_derivative,
            "no_op": self.no_op,
            "no_op_reason": self.no_op_reason,
            "mean_parameter_gradient_sha256": _tensor_sha256(
                self.mean_parameter_gradient
            ),
            "gradient_gram_sha256": _tensor_sha256(self.gradient_gram),
            "raw_direction_sha256": _tensor_sha256(self.raw_direction),
            "direction_sha256": _tensor_sha256(self.direction),
            "raw_fit_tensors_serialized": False,
            "training_only": True,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        parameter_count = len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER)
        if (
            self.mean_parameter_gradient.dtype != torch.float64
            or self.mean_parameter_gradient.device.type != "cpu"
            or self.mean_parameter_gradient.shape != (parameter_count,)
            or self.gradient_gram.dtype != torch.float64
            or self.gradient_gram.device.type != "cpu"
            or self.gradient_gram.shape != (parameter_count, parameter_count)
            or self.raw_direction.dtype != torch.float64
            or self.raw_direction.device.type != "cpu"
            or self.raw_direction.shape != (parameter_count,)
            or self.direction.dtype != torch.float64
            or self.direction.device.type != "cpu"
            or self.direction.shape != (parameter_count,)
            or not all(
                bool(torch.isfinite(value).all())
                for value in (
                    self.mean_parameter_gradient,
                    self.gradient_gram,
                    self.raw_direction,
                    self.direction,
                )
            )
        ):
            raise RuntimeError("soft-polarity natural direction tensors drifted")
        expected = _sha256(
            _NATURAL_DIRECTION_DOMAIN,
            self.metadata(include_artifact=False),
        )
        if expected != _require_sha256(
            self.artifact_sha256, label="natural direction artifact"
        ):
            raise RuntimeError("soft-polarity natural direction receipt drifted")


def _natural_direction_result(
    aggregate: SoftPolarityTokenVJPAggregate,
    *,
    ridge_multiplier: float,
    gradient_gram_trace: float,
    tau: float,
    damping: float,
    raw_direction: Tensor,
    direction: Tensor,
    predicted_derivative: float,
    no_op_reason: str | None,
) -> SoftPolarityTokenVJPNaturalDirection:
    return SoftPolarityTokenVJPNaturalDirection(
        aggregate_artifact_sha256=aggregate.artifact_sha256,
        feature_id=aggregate.feature_id,
        held_family_id=aggregate.held_family_id,
        reference_b=aggregate.reference_b,
        reference_a=aggregate.reference_a,
        ridge_multiplier=ridge_multiplier,
        gradient_gram_trace=gradient_gram_trace,
        tau=tau,
        damping=damping,
        mean_parameter_gradient=aggregate.mean_parameter_gradient,
        gradient_gram=aggregate.gradient_gram,
        raw_direction=raw_direction,
        direction=direction,
        direction_linf=float(direction.abs().max()),
        predicted_derivative=predicted_derivative,
        no_op=no_op_reason is not None,
        no_op_reason=no_op_reason,
    )


def build_soft_polarity_token_vjp_natural_direction(
    aggregate: SoftPolarityTokenVJPAggregate,
    *,
    ridge_multiplier: float,
) -> SoftPolarityTokenVJPNaturalDirection:
    """Build the V20q trace-scaled, unit-L-infinity natural direction.

    ``tau = max(trace(F) / 2, 2^-24)`` and
    ``raw = -(F + ridge_multiplier * tau * I)^-1 g`` are frozen here.  The
    caller remains responsible for choosing a ridge multiplier and any finite
    alpha children.
    """

    if not isinstance(aggregate, SoftPolarityTokenVJPAggregate):
        raise TypeError("aggregate must be a token VJP aggregate")
    aggregate.validate_integrity()
    ridge = _strict_float(
        ridge_multiplier,
        label="natural direction ridge_multiplier",
        nonnegative=True,
    )
    parameter_count = len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER)
    gram = aggregate.gradient_gram
    gradient = aggregate.mean_parameter_gradient
    gram_trace = float(torch.trace(gram))
    tau = max(
        gram_trace / float(parameter_count),
        SOFT_POLARITY_TOKEN_VJP_TAU_FLOOR,
    )
    damping = ridge * tau
    if not math.isfinite(damping):
        raise ValueError("natural direction ridge times tau must remain finite")
    zero = torch.zeros(parameter_count, dtype=torch.float64)
    if bool((gradient == 0.0).all()):
        return _natural_direction_result(
            aggregate,
            ridge_multiplier=ridge,
            gradient_gram_trace=gram_trace,
            tau=tau,
            damping=damping,
            raw_direction=zero,
            direction=zero,
            predicted_derivative=0.0,
            no_op_reason="zero_mean_kl_gradient",
        )
    damped = gram + damping * torch.eye(parameter_count, dtype=torch.float64)
    try:
        raw = torch.linalg.solve(damped, -gradient).contiguous()
    except RuntimeError:
        return _natural_direction_result(
            aggregate,
            ridge_multiplier=ridge,
            gradient_gram_trace=gram_trace,
            tau=tau,
            damping=damping,
            raw_direction=zero,
            direction=zero,
            predicted_derivative=0.0,
            no_op_reason="singular_damped_system",
        )
    if not bool(torch.isfinite(raw).all()):
        return _natural_direction_result(
            aggregate,
            ridge_multiplier=ridge,
            gradient_gram_trace=gram_trace,
            tau=tau,
            damping=damping,
            raw_direction=zero,
            direction=zero,
            predicted_derivative=0.0,
            no_op_reason="nonfinite_damped_direction",
        )
    raw_linf = float(raw.abs().max())
    if raw_linf <= 0.0:
        return _natural_direction_result(
            aggregate,
            ridge_multiplier=ridge,
            gradient_gram_trace=gram_trace,
            tau=tau,
            damping=damping,
            raw_direction=raw,
            direction=zero,
            predicted_derivative=0.0,
            no_op_reason="zero_damped_direction",
        )
    direction = (raw / raw_linf).contiguous()
    maximum_index = int(torch.argmax(raw.abs()))
    direction[maximum_index] = math.copysign(1.0, float(raw[maximum_index]))
    predicted = float(gradient @ direction)
    if not math.isfinite(predicted) or predicted >= 0.0:
        return _natural_direction_result(
            aggregate,
            ridge_multiplier=ridge,
            gradient_gram_trace=gram_trace,
            tau=tau,
            damping=damping,
            raw_direction=raw,
            direction=zero,
            predicted_derivative=0.0,
            no_op_reason="non_descent_direction",
        )
    return _natural_direction_result(
        aggregate,
        ridge_multiplier=ridge,
        gradient_gram_trace=gram_trace,
        tau=tau,
        damping=damping,
        raw_direction=raw,
        direction=direction,
        predicted_derivative=predicted,
        no_op_reason=None,
    )


@dataclass(frozen=True, slots=True)
class SoftPolarityTokenVJPFitArguments:
    """Caller-selected numerical controls for one fit, with no ladder policy."""

    damping: float
    trust_l2_bound: float
    solver_kind: str = "mean_kl_natural_opg"
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        damping = _strict_float(self.damping, label="damping", nonnegative=True)
        trust = _strict_float(
            self.trust_l2_bound, label="trust_l2_bound", nonnegative=True
        )
        solver_kind = _identifier(self.solver_kind, label="solver_kind")
        if solver_kind not in SOFT_POLARITY_TOKEN_VJP_SOLVER_KINDS:
            raise ValueError(
                "solver_kind must select a declared token VJP fit solver"
            )
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "trust_l2_bound", trust)
        object.__setattr__(self, "solver_kind", solver_kind)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FIT_ARGUMENTS_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "damping": self.damping,
            "damping_hex": self.damping.hex(),
            "trust_l2_bound": self.trust_l2_bound,
            "trust_l2_bound_hex": self.trust_l2_bound.hex(),
            "solver_kind": self.solver_kind,
            "protocol_ladder_selected": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        expected = _sha256(
            _FIT_ARGUMENTS_DOMAIN, self.metadata(include_artifact=False)
        )
        if expected != _require_sha256(
            self.artifact_sha256, label="fit arguments artifact"
        ):
            raise RuntimeError("soft-polarity token VJP fit arguments drifted")


@dataclass(frozen=True, slots=True)
class SoftPolarityTokenVJPFit:
    """Tamper-evident receipt for a conservative two-coordinate fit step."""

    aggregate_artifact_sha256: str
    arguments_artifact_sha256: str
    feature_id: str
    held_family_id: str
    reference_b: float
    reference_a: float
    damping: float
    trust_l2_bound: float
    solver_kind: str
    mean_parameter_gradient: Tensor = field(repr=False)
    residual_gradient_c: Tensor = field(repr=False)
    selected_rhs: Tensor = field(repr=False)
    gradient_gram: Tensor = field(repr=False)
    damped_system: Tensor = field(repr=False)
    raw_step: Tensor = field(repr=False)
    applied_step: Tensor = field(repr=False)
    raw_step_l2: float
    applied_step_l2: float
    trust_scale: float
    proposed_b: float
    proposed_a: float
    predicted_derivative: float
    no_op: bool
    no_op_reason: str | None
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        aggregate_hash = _require_sha256(
            self.aggregate_artifact_sha256, label="aggregate artifact"
        )
        arguments_hash = _require_sha256(
            self.arguments_artifact_sha256, label="fit arguments artifact"
        )
        feature_id = _identifier(self.feature_id, label="fit feature_id")
        held_family = _identifier(self.held_family_id, label="fit held_family_id")
        reference_b = _strict_float(self.reference_b, label="fit reference_b")
        reference_a = _strict_float(self.reference_a, label="fit reference_a")
        damping = _strict_float(self.damping, label="fit damping", nonnegative=True)
        trust = _strict_float(
            self.trust_l2_bound, label="fit trust_l2_bound", nonnegative=True
        )
        solver_kind = _identifier(self.solver_kind, label="fit solver_kind")
        if solver_kind not in SOFT_POLARITY_TOKEN_VJP_SOLVER_KINDS:
            raise ValueError("fit solver_kind is not declared")
        mean_gradient = _exact_f64_tensor(
            self.mean_parameter_gradient,
            label="fit mean_parameter_gradient",
            ndim=1,
        )
        c = _exact_f64_tensor(
            self.residual_gradient_c, label="fit residual_gradient_c", ndim=1
        )
        selected_rhs = _exact_f64_tensor(
            self.selected_rhs, label="fit selected_rhs", ndim=1
        )
        gram = _exact_f64_tensor(
            self.gradient_gram, label="fit gradient_gram", ndim=2
        )
        damped = _exact_f64_tensor(
            self.damped_system, label="fit damped_system", ndim=2
        )
        raw = _exact_f64_tensor(self.raw_step, label="fit raw_step", ndim=1)
        applied = _exact_f64_tensor(
            self.applied_step, label="fit applied_step", ndim=1
        )
        parameter_count = len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER)
        if (
            mean_gradient.shape != (parameter_count,)
            or c.shape != (parameter_count,)
            or selected_rhs.shape != (parameter_count,)
            or gram.shape != (parameter_count, parameter_count)
            or damped.shape != (parameter_count, parameter_count)
            or raw.shape != (parameter_count,)
            or applied.shape != (parameter_count,)
            or not torch.equal(gram, gram.T)
            or not torch.equal(damped, damped.T)
        ):
            raise ValueError("fit tensor geometry differs")
        expected_rhs = (
            mean_gradient
            if solver_kind == "mean_kl_natural_opg"
            else c
        )
        if not torch.equal(selected_rhs, expected_rhs):
            raise ValueError("fit selected RHS differs from solver kind")
        raw_l2 = _strict_float(self.raw_step_l2, label="raw_step_l2", nonnegative=True)
        applied_l2 = _strict_float(
            self.applied_step_l2, label="applied_step_l2", nonnegative=True
        )
        trust_scale = _strict_float(
            self.trust_scale, label="trust_scale", nonnegative=True
        )
        proposed_b = _strict_float(self.proposed_b, label="proposed_b")
        proposed_a = _strict_float(self.proposed_a, label="proposed_a")
        predicted = _strict_float(
            self.predicted_derivative, label="predicted_derivative"
        )
        if type(self.no_op) is not bool:
            raise ValueError("no_op must be bool")
        if self.no_op:
            reason = _identifier(self.no_op_reason, label="no_op_reason")
            if (
                not torch.equal(applied, torch.zeros_like(applied))
                or proposed_b.hex() != reference_b.hex()
                or proposed_a.hex() != reference_a.hex()
                or applied_l2 != 0.0
                or trust_scale != 0.0
                or predicted != 0.0
            ):
                raise ValueError("no-op fit must preserve the exact reference")
        else:
            if self.no_op_reason is not None:
                raise ValueError("non-no-op fit may not have a no-op reason")
            reason = None
            if predicted >= 0.0 or applied_l2 <= 0.0 or applied_l2 > trust:
                raise ValueError("applied fit step must be bounded and descending")
        object.__setattr__(self, "aggregate_artifact_sha256", aggregate_hash)
        object.__setattr__(self, "arguments_artifact_sha256", arguments_hash)
        object.__setattr__(self, "feature_id", feature_id)
        object.__setattr__(self, "held_family_id", held_family)
        object.__setattr__(self, "reference_b", reference_b)
        object.__setattr__(self, "reference_a", reference_a)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "trust_l2_bound", trust)
        object.__setattr__(self, "solver_kind", solver_kind)
        object.__setattr__(self, "mean_parameter_gradient", mean_gradient)
        object.__setattr__(self, "residual_gradient_c", c)
        object.__setattr__(self, "selected_rhs", selected_rhs)
        object.__setattr__(self, "gradient_gram", gram)
        object.__setattr__(self, "damped_system", damped)
        object.__setattr__(self, "raw_step", raw)
        object.__setattr__(self, "applied_step", applied)
        object.__setattr__(self, "raw_step_l2", raw_l2)
        object.__setattr__(self, "applied_step_l2", applied_l2)
        object.__setattr__(self, "trust_scale", trust_scale)
        object.__setattr__(self, "proposed_b", proposed_b)
        object.__setattr__(self, "proposed_a", proposed_a)
        object.__setattr__(self, "predicted_derivative", predicted)
        object.__setattr__(self, "no_op_reason", reason)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FIT_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def raw_step_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.raw_step.clone().contiguous()

    def applied_step_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.applied_step.clone().contiguous()

    def selected_rhs_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.selected_rhs.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "aggregate_artifact_sha256": self.aggregate_artifact_sha256,
            "arguments_artifact_sha256": self.arguments_artifact_sha256,
            "feature_id": self.feature_id,
            "held_family_id": self.held_family_id,
            "parameter_order": SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER,
            "reference_b": self.reference_b,
            "reference_b_hex": self.reference_b.hex(),
            "reference_a": self.reference_a,
            "reference_a_hex": self.reference_a.hex(),
            "damping": self.damping,
            "damping_hex": self.damping.hex(),
            "trust_l2_bound": self.trust_l2_bound,
            "trust_l2_bound_hex": self.trust_l2_bound.hex(),
            "solver_kind": self.solver_kind,
            "mean_parameter_gradient_sha256": _tensor_sha256(
                self.mean_parameter_gradient
            ),
            "residual_gradient_c_sha256": _tensor_sha256(
                self.residual_gradient_c
            ),
            "selected_rhs_sha256": _tensor_sha256(self.selected_rhs),
            "selected_rhs_definition": (
                "mean_token_Q"
                if self.solver_kind == "mean_kl_natural_opg"
                else "mean_token_Q_times_token_KL"
            ),
            "gradient_gram_sha256": _tensor_sha256(self.gradient_gram),
            "damped_system_sha256": _tensor_sha256(self.damped_system),
            "raw_step_sha256": _tensor_sha256(self.raw_step),
            "applied_step_sha256": _tensor_sha256(self.applied_step),
            "raw_step_l2": self.raw_step_l2,
            "applied_step_l2": self.applied_step_l2,
            "trust_scale": self.trust_scale,
            "proposed_b": self.proposed_b,
            "proposed_b_hex": self.proposed_b.hex(),
            "proposed_a": self.proposed_a,
            "proposed_a_hex": self.proposed_a.hex(),
            "predicted_derivative": self.predicted_derivative,
            "no_op": self.no_op,
            "no_op_reason": self.no_op_reason,
            "method": (
                "one_step_damped_mean_KL_natural_gradient_with_OPG"
                if self.solver_kind == "mean_kl_natural_opg"
                else "one_step_damped_squared_KL_residual_Gauss_Newton_with_OPG"
            ),
            "not_claimed_as": (
                "exact_Fisher_or_exact_GGN"
                if self.solver_kind == "mean_kl_natural_opg"
                else "mean_KL_natural_gradient_or_exact_GGN"
            ),
            "raw_tensors_serialized": False,
            "training_only": True,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if not all(
            tensor.dtype == torch.float64
            and tensor.device.type == "cpu"
            and bool(torch.isfinite(tensor).all())
            for tensor in (
                self.mean_parameter_gradient,
                self.residual_gradient_c,
                self.selected_rhs,
                self.gradient_gram,
                self.damped_system,
                self.raw_step,
                self.applied_step,
            )
        ):
            raise RuntimeError("soft-polarity token VJP fit tensors drifted")
        expected = _sha256(_FIT_DOMAIN, self.metadata(include_artifact=False))
        if expected != _require_sha256(self.artifact_sha256, label="fit artifact"):
            raise RuntimeError("soft-polarity token VJP fit receipt drifted")


def _no_op_fit(
    aggregate: SoftPolarityTokenVJPAggregate,
    arguments: SoftPolarityTokenVJPFitArguments,
    *,
    damped_system: Tensor,
    reason: str,
    raw_step: Tensor | None = None,
) -> SoftPolarityTokenVJPFit:
    zero = torch.zeros(
        len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER), dtype=torch.float64
    )
    raw = zero if raw_step is None else raw_step
    selected_rhs = (
        aggregate.mean_parameter_gradient
        if arguments.solver_kind == "mean_kl_natural_opg"
        else aggregate.residual_gradient_c
    )
    return SoftPolarityTokenVJPFit(
        aggregate_artifact_sha256=aggregate.artifact_sha256,
        arguments_artifact_sha256=arguments.artifact_sha256,
        feature_id=aggregate.feature_id,
        held_family_id=aggregate.held_family_id,
        reference_b=aggregate.reference_b,
        reference_a=aggregate.reference_a,
        damping=arguments.damping,
        trust_l2_bound=arguments.trust_l2_bound,
        solver_kind=arguments.solver_kind,
        mean_parameter_gradient=aggregate.mean_parameter_gradient,
        residual_gradient_c=aggregate.residual_gradient_c,
        selected_rhs=selected_rhs,
        gradient_gram=aggregate.gradient_gram,
        damped_system=damped_system,
        raw_step=raw,
        applied_step=zero,
        raw_step_l2=float(torch.linalg.vector_norm(raw)),
        applied_step_l2=0.0,
        trust_scale=0.0,
        proposed_b=aggregate.reference_b,
        proposed_a=aggregate.reference_a,
        predicted_derivative=0.0,
        no_op=True,
        no_op_reason=reason,
    )


def fit_soft_polarity_token_vjp_step(
    aggregate: SoftPolarityTokenVJPAggregate,
    *,
    arguments: SoftPolarityTokenVJPFitArguments,
) -> SoftPolarityTokenVJPFit:
    """Solve one caller-bounded two-dimensional natural or residual step.

    Positive damping is allowed to regularize a rank-deficient OPG matrix.
    Evidence with no gradient component in the observed OPG range, an
    unsolved damped system, a zero trust bound, and any non-descending result
    return the exact reference values as an explicit no-op.
    """

    if not isinstance(aggregate, SoftPolarityTokenVJPAggregate):
        raise TypeError("aggregate must be a token VJP aggregate")
    if not isinstance(arguments, SoftPolarityTokenVJPFitArguments):
        raise TypeError("arguments must be frozen token VJP fit arguments")
    aggregate.validate_integrity()
    arguments.validate_integrity()
    parameter_count = len(SOFT_POLARITY_TOKEN_VJP_PARAMETER_ORDER)
    identity = torch.eye(parameter_count, dtype=torch.float64)
    gram = aggregate.gradient_gram
    rhs = (
        aggregate.mean_parameter_gradient
        if arguments.solver_kind == "mean_kl_natural_opg"
        else aggregate.residual_gradient_c
    )
    damped = (gram + arguments.damping * identity).contiguous()

    if bool((rhs == 0.0).all()):
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason=(
                "zero_mean_kl_gradient"
                if arguments.solver_kind == "mean_kl_natural_opg"
                else "zero_residual_gradient"
            ),
        )
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    threshold = (
        torch.finfo(torch.float64).eps
        * max(1.0, float(eigenvalues.abs().max()))
        * 32.0
    )
    supported = eigenvalues > threshold
    supported_rhs = eigenvectors[:, supported].T @ rhs
    supported_rhs_tolerance = (
        torch.finfo(torch.float64).eps
        * max(1.0, float(rhs.abs().max()))
        * 32.0
    )
    if not bool(supported.any()) or float(supported_rhs.abs().max()) <= (
        supported_rhs_tolerance
    ):
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason="zero_supported_gradient_evidence",
        )
    if arguments.trust_l2_bound == 0.0:
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason="zero_trust_bound",
        )

    scale = max(float(damped.abs().max()), float(rhs.abs().max()))
    if not math.isfinite(scale) or scale <= 0.0:
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason="singular_damped_system",
        )
    try:
        raw = torch.linalg.solve(damped / scale, -rhs / scale).contiguous()
    except RuntimeError:
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason="singular_damped_system",
        )
    if not bool(torch.isfinite(raw).all()):
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason="nonfinite_damped_step",
        )
    raw_l2 = float(torch.linalg.vector_norm(raw))
    if raw_l2 == 0.0:
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason="zero_damped_step",
            raw_step=raw,
        )
    trust_scale = min(1.0, arguments.trust_l2_bound / raw_l2)
    applied = (raw * trust_scale).contiguous()
    predicted = float(rhs @ applied)
    if not math.isfinite(predicted) or predicted >= 0.0:
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason="non_descent_step",
            raw_step=raw,
        )
    applied_l2 = float(torch.linalg.vector_norm(applied))
    proposed_b = aggregate.reference_b + float(applied[0])
    proposed_a = aggregate.reference_a + float(applied[1])
    if not math.isfinite(proposed_b) or not math.isfinite(proposed_a):
        return _no_op_fit(
            aggregate,
            arguments,
            damped_system=damped,
            reason="nonfinite_proposal",
            raw_step=raw,
        )
    return SoftPolarityTokenVJPFit(
        aggregate_artifact_sha256=aggregate.artifact_sha256,
        arguments_artifact_sha256=arguments.artifact_sha256,
        feature_id=aggregate.feature_id,
        held_family_id=aggregate.held_family_id,
        reference_b=aggregate.reference_b,
        reference_a=aggregate.reference_a,
        damping=arguments.damping,
        trust_l2_bound=arguments.trust_l2_bound,
        solver_kind=arguments.solver_kind,
        mean_parameter_gradient=aggregate.mean_parameter_gradient,
        residual_gradient_c=aggregate.residual_gradient_c,
        selected_rhs=rhs,
        gradient_gram=gram,
        damped_system=damped,
        raw_step=raw,
        applied_step=applied,
        raw_step_l2=raw_l2,
        applied_step_l2=applied_l2,
        trust_scale=trust_scale,
        proposed_b=proposed_b,
        proposed_a=proposed_a,
        predicted_derivative=predicted,
        no_op=False,
        no_op_reason=None,
    )
