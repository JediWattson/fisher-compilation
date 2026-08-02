"""Pure V8 finite support for the frozen V5/V6/V7 K64 arms.

V8 does not fit or select a candidate.  It authenticates three frozen gain
fields, records their finite teacher-KL observations on the exact
sixteen-prompt outer-held grid, and reports the preregistered V5 support gates
without authorizing serving:

``v5_static_plus``
    The V5 ``+1/64`` K64 carrier.
``v6_exact_scalar``
    The exact one-scalar control fitted beside V6/V7.
``v7_joint``
    The five-parameter joint intercept-plus-state field, evaluated from the
    frozen codec on pre-gate base-H4 support rows.

The unit arm is only a pinned, non-executed reference.  This module has no
model dependency and performs no finite forward itself; diagnostics can use
the differentiable row-correction helper with their existing V5 executor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from fisher_graph.complete_h4_tail_candidate_gain_microstep import (
    symmetric_microstep_gains,
)
from fisher_graph.complete_h4_tail_candidate_gain_refit_v4 import (
    CANDIDATE_GAIN_RANK,
    CandidateConditionedK64MeanKLRefit,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_gain_field import (
    CandidateConditionedK64JointStateGainFieldFit,
)
from fisher_graph.complete_h4_tail_candidate_state_gain_field import (
    STATE_FEATURE_RANK,
    STATE_GAIN_BASE_STEP,
    CandidateConditionedK64StateFeatureCodec,
    CandidateConditionedK64StaticAmplitudeControlFit,
    encode_candidate_conditioned_k64_state_features,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
)


__all__ = [
    "THREE_ARM_FINITE_NAMES",
    "V8_EXPECTED_HELD_CELL_COUNT",
    "V8_EXPECTED_CANDIDATE_FORWARD_COUNT",
    "CandidateConditionedK64ThreeArmGainSupport",
    "CandidateConditionedK64ThreeArmFiniteExample",
    "CandidateConditionedK64ThreeArmFiniteComparison",
    "candidate_conditioned_k64_gain_scaled_tail_rows",
    "candidate_conditioned_k64_gain_correction_rows",
    "build_candidate_conditioned_k64_three_arm_gain_support",
    "compare_candidate_conditioned_k64_three_arm_finite_examples",
]


THREE_ARM_FINITE_NAMES = (
    "v5_static_plus",
    "v6_exact_scalar",
    "v7_joint",
)
V8_EXPECTED_HELD_CELL_COUNT = 16
V8_EXPECTED_CANDIDATE_FORWARD_COUNT = 3 * V8_EXPECTED_HELD_CELL_COUNT

_EXPECTED_OUTER_FAMILIES = 8
_EXPECTED_HELD_CELLS_PER_FOLD = 2
_USEFUL_RELATIVE_IMPROVEMENT = 0.02
_MINIMUM_FAMILY_WIN_COUNT = 6
_WORST_FAMILY_RATIO = 1.05
_ABSOLUTE_KL_TOLERANCE = 1.0e-8
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SUPPORT_DOMAIN = b"fisher-graph:candidate-k64-three-arm-support:v8\0"
_EXAMPLE_DOMAIN = b"fisher-graph:candidate-k64-three-arm-example:v8\0"
_COMPARISON_DOMAIN = b"fisher-graph:candidate-k64-three-arm-comparison:v8\0"


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


def _float64(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or any(int(width) <= 0 for width in value.shape)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty floating tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and nonnegative") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _finite_ratio(candidate: float, source: float) -> float:
    if source == 0.0:
        return 1.0 if candidate == 0.0 else torch.finfo(torch.float64).max
    return candidate / source


def _validate_row_algebra_inputs(
    *, tail_rows: Tensor, ordered_directions: Tensor, gains: Tensor
) -> tuple[int, int]:
    if (
        not isinstance(tail_rows, Tensor)
        or tail_rows.ndim != 2
        or not tail_rows.is_floating_point()
        or not isinstance(ordered_directions, Tensor)
        or ordered_directions.ndim != 2
        or not ordered_directions.is_floating_point()
        or not isinstance(gains, Tensor)
        or gains.ndim not in (1, 2)
        or not gains.is_floating_point()
    ):
        raise ValueError("K64 row correction requires floating tensors")
    rows, width = (int(value) for value in tail_rows.shape)
    if (
        rows <= 0
        or width <= 0
        or ordered_directions.shape != (CANDIDATE_GAIN_RANK, width)
        or (
            gains.shape != (CANDIDATE_GAIN_RANK,)
            and gains.shape != (rows, CANDIDATE_GAIN_RANK)
        )
        or tail_rows.device != ordered_directions.device
        or gains.device != tail_rows.device
        or tail_rows.dtype != ordered_directions.dtype
        or gains.dtype != tail_rows.dtype
        or not bool(torch.isfinite(tail_rows).all())
        or not bool(torch.isfinite(ordered_directions).all())
        or not bool(torch.isfinite(gains).all())
    ):
        raise ValueError("K64 row correction geometry differs")
    return rows, width


def candidate_conditioned_k64_gain_scaled_tail_rows(
    *,
    tail_rows: Tensor,
    ordered_directions: Tensor,
    gains: Tensor,
) -> Tensor:
    """Apply static ``K64`` or row-specific ``R x K64`` gains to tail rows.

    The operation intentionally preserves device, dtype, and autograd.  A
    one-dimensional gain vector is the exact algebra used by the V5 finite
    executor; a two-dimensional field is its row-conditioned generalization.
    """

    _validate_row_algebra_inputs(
        tail_rows=tail_rows,
        ordered_directions=ordered_directions,
        gains=gains,
    )
    coordinates = tail_rows @ ordered_directions.T
    return ((coordinates * gains) @ ordered_directions).contiguous()


def candidate_conditioned_k64_gain_correction_rows(
    *,
    supported_rows: Tensor,
    tail_rows: Tensor,
    ordered_directions: Tensor,
    gains: Tensor,
) -> Tensor:
    """Return the V5 supported component plus a static or row-wise K64 tail."""

    rows, width = _validate_row_algebra_inputs(
        tail_rows=tail_rows,
        ordered_directions=ordered_directions,
        gains=gains,
    )
    if (
        not isinstance(supported_rows, Tensor)
        or supported_rows.shape != (rows, width)
        or not supported_rows.is_floating_point()
        or supported_rows.device != tail_rows.device
        or supported_rows.dtype != tail_rows.dtype
        or not bool(torch.isfinite(supported_rows).all())
    ):
        raise ValueError("supported row geometry differs")
    return (
        supported_rows
        + candidate_conditioned_k64_gain_scaled_tail_rows(
            tail_rows=tail_rows,
            ordered_directions=ordered_directions,
            gains=gains,
        )
    ).contiguous()


def _fit_bindings_match(
    refit: CandidateConditionedK64MeanKLRefit,
    codec: CandidateConditionedK64StateFeatureCodec,
    scalar: CandidateConditionedK64StaticAmplitudeControlFit,
    joint: CandidateConditionedK64JointStateGainFieldFit,
) -> bool:
    delta = refit.mean_proposed_gains_tensor() - 1.0
    return (
        refit.held_family_id
        == codec.held_family_id
        == scalar.held_family_id
        == joint.held_family_id
        and scalar.refit_artifact_sha256 == refit.artifact_sha256
        and joint.refit_artifact_sha256 == refit.artifact_sha256
        and joint.codec_artifact_sha256 == codec.artifact_sha256
        and refit.training_family_ids == codec.training_family_ids
        and refit.training_example_ids == codec.training_example_ids
        and scalar.training_family_ids == joint.training_family_ids
        and scalar.training_example_ids == joint.training_example_ids
        and scalar.training_example_artifact_sha256s
        == joint.training_example_artifact_sha256s
        and scalar.training_family_ids == codec.training_family_ids
        and scalar.training_example_ids == codec.training_example_ids
        and torch.equal(scalar.mean_gain_delta, delta)
        and torch.equal(joint.mean_gain_delta, delta)
        and torch.equal(joint.feature_scale, codec.feature_scale)
        and scalar.mean_gradient == float(joint.mean_gradient[0])
        and scalar.fisher_energy == float(joint.joint_fisher_gram[0, 0])
    )


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64ThreeArmGainSupport:
    """Frozen per-prompt gains for the three uniformly executed V8 arms."""

    phase: str
    held_family_id: str
    example_id: str
    family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    parent_fold_artifact_sha256: str
    refit_artifact_sha256: str
    codec_artifact_sha256: str
    scalar_fit_artifact_sha256: str
    joint_fit_artifact_sha256: str
    ordered_directions_codec_sha256: str
    ordered_directions_refit_sha256: str
    base_h4_support_rows_sha256: str
    standardized_state_features: Tensor = field(repr=False)
    static_plus_gains: Tensor = field(repr=False)
    exact_scalar_gains: Tensor = field(repr=False)
    joint_row_gains: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        phase = str(self.phase)
        held = _identifier(self.held_family_id, label="held_family_id")
        example = _identifier(self.example_id, label="finite example_id")
        family = _identifier(self.family_id, label="finite family_id")
        families = tuple(
            _identifier(value, label="training family_id")
            for value in self.training_family_ids
        )
        training_ids = tuple(
            _identifier(value, label="training example_id")
            for value in self.training_example_ids
        )
        features = _float64(
            self.standardized_state_features,
            label="runtime standardized state features",
            ndim=2,
        )
        static = _float64(
            self.static_plus_gains, label="V5 static-plus gains", ndim=1
        )
        scalar = _float64(
            self.exact_scalar_gains, label="exact V6 scalar gains", ndim=1
        )
        joint = _float64(
            self.joint_row_gains, label="V7 joint row gains", ndim=2
        )
        if (
            phase != "held"
            or len(families) != 7
            or families != tuple(sorted(set(families)))
            or held in families
            or len(training_ids) != 7
            or training_ids != tuple(sorted(set(training_ids)))
            or example in training_ids
            or family != held
            or features.shape[1] != STATE_FEATURE_RANK
            or static.shape != (CANDIDATE_GAIN_RANK,)
            or scalar.shape != static.shape
            or joint.shape != (features.shape[0], CANDIDATE_GAIN_RANK)
            or bool((static < 0.0).any())
            or bool((static > 1.5).any())
            or bool((scalar < 0.0).any())
            or bool((scalar > 1.5).any())
            or bool((joint < 0.0).any())
            or bool((joint > 1.5).any())
        ):
            raise ValueError("three-arm gain support geometry differs")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "example_id", example)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", training_ids)
        for name in (
            "parent_fold_artifact_sha256",
            "refit_artifact_sha256",
            "codec_artifact_sha256",
            "scalar_fit_artifact_sha256",
            "joint_fit_artifact_sha256",
            "ordered_directions_codec_sha256",
            "ordered_directions_refit_sha256",
            "base_h4_support_rows_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), label=name),
            )
        object.__setattr__(self, "standardized_state_features", features)
        object.__setattr__(self, "static_plus_gains", static)
        object.__setattr__(self, "exact_scalar_gains", scalar)
        object.__setattr__(self, "joint_row_gains", joint)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_SUPPORT_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def support_row_count(self) -> int:
        return int(self.joint_row_gains.shape[0])

    @property
    def fold_binding(self) -> tuple[str, ...]:
        return (
            self.parent_fold_artifact_sha256,
            self.refit_artifact_sha256,
            self.codec_artifact_sha256,
            self.scalar_fit_artifact_sha256,
            self.joint_fit_artifact_sha256,
            self.ordered_directions_codec_sha256,
            self.ordered_directions_refit_sha256,
        )

    def gains_tensor(self, arm: str) -> Tensor:
        self.validate_integrity()
        if arm == "v5_static_plus":
            return self.static_plus_gains.clone().contiguous()
        if arm == "v6_exact_scalar":
            return self.exact_scalar_gains.clone().contiguous()
        if arm == "v7_joint":
            return self.joint_row_gains.clone().contiguous()
        raise ValueError("unknown V8 finite arm")

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "phase": self.phase,
            "held_family_id": self.held_family_id,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "parent_fold_artifact_sha256": self.parent_fold_artifact_sha256,
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "codec_artifact_sha256": self.codec_artifact_sha256,
            "scalar_fit_artifact_sha256": self.scalar_fit_artifact_sha256,
            "joint_fit_artifact_sha256": self.joint_fit_artifact_sha256,
            "ordered_directions_codec_sha256": (
                self.ordered_directions_codec_sha256
            ),
            "ordered_directions_refit_sha256": (
                self.ordered_directions_refit_sha256
            ),
            "base_h4_support_rows_sha256": self.base_h4_support_rows_sha256,
            "standardized_state_features_sha256": _runtime_tensor_sha256(
                self.standardized_state_features
            ),
            "static_plus_gains_sha256": _runtime_tensor_sha256(
                self.static_plus_gains
            ),
            "exact_scalar_gains_sha256": _runtime_tensor_sha256(
                self.exact_scalar_gains
            ),
            "joint_row_gains_sha256": _runtime_tensor_sha256(
                self.joint_row_gains
            ),
            "support_row_count": self.support_row_count,
            "rank": CANDIDATE_GAIN_RANK,
            "arms": THREE_ARM_FINITE_NAMES,
            "feature_source": "pre_gate_bridge_base_H4_support_rows",
            "codec_center_and_scale_frozen": True,
            "fits_recomputed_or_refit_from_held_examples": False,
            "all_three_arms_executed_uniformly": True,
            "unit_is_a_pinned_nonexecuted_reference": True,
            "held_family_used_for_fit_or_tune": False,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            self.standardized_state_features.dtype != torch.float64
            or self.standardized_state_features.device.type != "cpu"
            or self.standardized_state_features.requires_grad
            or not self.standardized_state_features.is_contiguous()
            or self.static_plus_gains.dtype != torch.float64
            or self.static_plus_gains.device.type != "cpu"
            or self.static_plus_gains.requires_grad
            or not self.static_plus_gains.is_contiguous()
            or self.exact_scalar_gains.dtype != torch.float64
            or self.exact_scalar_gains.device.type != "cpu"
            or self.exact_scalar_gains.requires_grad
            or not self.exact_scalar_gains.is_contiguous()
            or self.joint_row_gains.dtype != torch.float64
            or self.joint_row_gains.device.type != "cpu"
            or self.joint_row_gains.requires_grad
            or not self.joint_row_gains.is_contiguous()
            or _sha256(_SUPPORT_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="three-arm support")
        ):
            raise RuntimeError("three-arm gain support payload drifted")


def build_candidate_conditioned_k64_three_arm_gain_support(
    refit: CandidateConditionedK64MeanKLRefit,
    codec: CandidateConditionedK64StateFeatureCodec,
    scalar_fit: CandidateConditionedK64StaticAmplitudeControlFit,
    joint_fit: CandidateConditionedK64JointStateGainFieldFit,
    *,
    phase: str,
    example_id: str,
    family_id: str,
    base_h4_support_rows: Tensor,
    ordered_directions: Tensor,
) -> CandidateConditionedK64ThreeArmGainSupport:
    """Authenticate frozen fits and materialize one prompt's three gain fields."""

    if not isinstance(refit, CandidateConditionedK64MeanKLRefit):
        raise TypeError("V8 support requires a V4 mean-KL refit")
    if not isinstance(codec, CandidateConditionedK64StateFeatureCodec):
        raise TypeError("V8 support requires a frozen V6 feature codec")
    if not isinstance(scalar_fit, CandidateConditionedK64StaticAmplitudeControlFit):
        raise TypeError("V8 support requires the exact V6 scalar fit")
    if not isinstance(joint_fit, CandidateConditionedK64JointStateGainFieldFit):
        raise TypeError("V8 support requires the frozen V7 joint fit")
    refit.validate_integrity()
    codec.validate_integrity()
    scalar_fit.validate_integrity()
    joint_fit.validate_integrity()
    if not _fit_bindings_match(refit, codec, scalar_fit, joint_fit):
        raise ValueError("V5/V6/V7 frozen fit bindings differ")
    rows = _float64(
        base_h4_support_rows,
        label="pre-gate base-H4 support rows",
        ndim=2,
    )
    directions = _float64(
        ordered_directions,
        label="ordered K64 directions",
        ndim=2,
    )
    features = encode_candidate_conditioned_k64_state_features(
        codec,
        base_h4_support_rows=rows,
        ordered_directions=directions,
    )
    if (
        _runtime_tensor_sha256(directions) != refit.ordered_directions_sha256
        or joint_fit.ordered_directions_codec_sha256
        != codec.ordered_directions_sha256
        or joint_fit.ordered_directions_refit_sha256
        != refit.ordered_directions_sha256
    ):
        raise ValueError("V8 runtime direction authentication differs")
    static = symmetric_microstep_gains(refit, sign=1)
    scalar = scalar_fit.gains_tensor()
    joint = joint_fit.row_gains_tensor(features)
    if not torch.equal(static, joint_fit.static_plus_gains_tensor()):
        raise RuntimeError("V7 zero-logit carrier does not replay V5 static plus")
    zero_logit_joint = (
        1.0
        + STATE_GAIN_BASE_STEP
        * torch.ones(rows.shape[0], 1, dtype=torch.float64)
        * joint_fit.mean_gain_delta[None, :]
    ).contiguous()
    if not torch.equal(zero_logit_joint, static.unsqueeze(0).expand_as(joint)):
        raise RuntimeError("row-wise zero-logit carrier differs from V5 plus")
    scalar_logit_joint = (
        1.0
        + STATE_GAIN_BASE_STEP
        * (
            1.0
            + torch.tanh(
                torch.tensor(scalar_fit.applied_coefficient, dtype=torch.float64)
            )
        )
        * joint_fit.mean_gain_delta
    ).contiguous()
    if not torch.equal(scalar, scalar_logit_joint):
        raise RuntimeError("constant joint logit does not replay exact V6 scalar")
    return CandidateConditionedK64ThreeArmGainSupport(
        phase=phase,
        held_family_id=refit.held_family_id,
        example_id=example_id,
        family_id=family_id,
        training_family_ids=codec.training_family_ids,
        training_example_ids=codec.training_example_ids,
        parent_fold_artifact_sha256=refit.parent_fold_artifact_sha256,
        refit_artifact_sha256=refit.artifact_sha256,
        codec_artifact_sha256=codec.artifact_sha256,
        scalar_fit_artifact_sha256=scalar_fit.artifact_sha256,
        joint_fit_artifact_sha256=joint_fit.artifact_sha256,
        ordered_directions_codec_sha256=codec.ordered_directions_sha256,
        ordered_directions_refit_sha256=refit.ordered_directions_sha256,
        base_h4_support_rows_sha256=_runtime_tensor_sha256(rows),
        standardized_state_features=features,
        static_plus_gains=static,
        exact_scalar_gains=scalar,
        joint_row_gains=joint,
    )


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64ThreeArmFiniteExample:
    """Same-token finite KL ledger for one authenticated prompt/fold cell."""

    gain_support: CandidateConditionedK64ThreeArmGainSupport = field(repr=False)
    model_inputs_sha256: str
    bridge_binding_sha256: str
    prefix_artifact_sha256: str
    support_mask_sha256: str
    teacher_logits_sha256: str
    endpoint_supervised_grid_sha256: str
    pinned_unit_receipt_sha256: str
    pinned_unit_token_teacher_kl_sha256: str
    pinned_unit_mean_teacher_kl: float
    arm_provider_artifact_sha256s: tuple[str, str, str]
    arm_execution_artifact_sha256s: tuple[str, str, str]
    arm_correction_rows_sha256s: tuple[str, str, str]
    arm_full_correction_sha256s: tuple[str, str, str]
    pinned_v5_static_plus_token_teacher_kl_sha256: str | None
    static_plus_token_teacher_kl: Tensor = field(repr=False)
    exact_scalar_token_teacher_kl: Tensor = field(repr=False)
    joint_token_teacher_kl: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.gain_support, CandidateConditionedK64ThreeArmGainSupport
        ):
            raise TypeError("finite example requires typed three-arm support")
        self.gain_support.validate_integrity()
        providers = tuple(
            _require_sha256(value, label="arm provider artifact")
            for value in self.arm_provider_artifact_sha256s
        )
        executions = tuple(
            _require_sha256(value, label="arm execution artifact")
            for value in self.arm_execution_artifact_sha256s
        )
        correction_rows = tuple(
            _require_sha256(value, label="arm correction rows")
            for value in self.arm_correction_rows_sha256s
        )
        full_corrections = tuple(
            _require_sha256(value, label="arm full correction")
            for value in self.arm_full_correction_sha256s
        )
        static = _float64(
            self.static_plus_token_teacher_kl,
            label="V5 static-plus token teacher KL",
            ndim=1,
        )
        scalar = _float64(
            self.exact_scalar_token_teacher_kl,
            label="exact V6 scalar token teacher KL",
            ndim=1,
        )
        joint = _float64(
            self.joint_token_teacher_kl,
            label="V7 joint token teacher KL",
            ndim=1,
        )
        pinned_plus = self.pinned_v5_static_plus_token_teacher_kl_sha256
        if pinned_plus is not None:
            pinned_plus = _require_sha256(pinned_plus, label="pinned V5 plus KL")
        if (
            len(providers) != len(THREE_ARM_FINITE_NAMES)
            or len(set(providers)) != len(providers)
            or len(executions) != len(THREE_ARM_FINITE_NAMES)
            or len(correction_rows) != len(THREE_ARM_FINITE_NAMES)
            or len(full_corrections) != len(THREE_ARM_FINITE_NAMES)
            or static.shape != scalar.shape
            or joint.shape != static.shape
            or bool((static < 0.0).any())
            or bool((scalar < 0.0).any())
            or bool((joint < 0.0).any())
            or (
                pinned_plus is not None
                and pinned_plus != _runtime_tensor_sha256(static)
            )
        ):
            raise ValueError("three-arm finite example evidence differs")
        for name in (
            "model_inputs_sha256",
            "bridge_binding_sha256",
            "prefix_artifact_sha256",
            "support_mask_sha256",
            "teacher_logits_sha256",
            "endpoint_supervised_grid_sha256",
            "pinned_unit_receipt_sha256",
            "pinned_unit_token_teacher_kl_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "pinned_unit_mean_teacher_kl",
            _finite_nonnegative(
                self.pinned_unit_mean_teacher_kl,
                label="pinned unit mean teacher KL",
            ),
        )
        object.__setattr__(self, "arm_provider_artifact_sha256s", providers)
        object.__setattr__(self, "arm_execution_artifact_sha256s", executions)
        object.__setattr__(self, "arm_correction_rows_sha256s", correction_rows)
        object.__setattr__(self, "arm_full_correction_sha256s", full_corrections)
        object.__setattr__(
            self,
            "pinned_v5_static_plus_token_teacher_kl_sha256",
            pinned_plus,
        )
        object.__setattr__(self, "static_plus_token_teacher_kl", static)
        object.__setattr__(self, "exact_scalar_token_teacher_kl", scalar)
        object.__setattr__(self, "joint_token_teacher_kl", joint)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def phase(self) -> str:
        return self.gain_support.phase

    @property
    def held_family_id(self) -> str:
        return self.gain_support.held_family_id

    @property
    def example_id(self) -> str:
        return self.gain_support.example_id

    @property
    def family_id(self) -> str:
        return self.gain_support.family_id

    def token_teacher_kl_tensor(self, arm: str) -> Tensor:
        self.validate_integrity()
        if arm == "v5_static_plus":
            return self.static_plus_token_teacher_kl.clone().contiguous()
        if arm == "v6_exact_scalar":
            return self.exact_scalar_token_teacher_kl.clone().contiguous()
        if arm == "v7_joint":
            return self.joint_token_teacher_kl.clone().contiguous()
        raise ValueError("unknown V8 finite arm")

    def mean_teacher_kl(self, arm: str) -> float:
        return float(self.token_teacher_kl_tensor(arm).mean())

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        tensors = (
            self.static_plus_token_teacher_kl,
            self.exact_scalar_token_teacher_kl,
            self.joint_token_teacher_kl,
        )
        arm_hashes = {
            arm: _runtime_tensor_sha256(value)
            for arm, value in zip(THREE_ARM_FINITE_NAMES, tensors)
        }
        arm_means = {
            arm: float(value.mean())
            for arm, value in zip(THREE_ARM_FINITE_NAMES, tensors)
        }
        result: dict[str, object] = {
            "phase": self.phase,
            "held_family_id": self.held_family_id,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "gain_support_artifact_sha256": self.gain_support.artifact_sha256,
            "refit_artifact_sha256": self.gain_support.refit_artifact_sha256,
            "codec_artifact_sha256": self.gain_support.codec_artifact_sha256,
            "scalar_fit_artifact_sha256": (
                self.gain_support.scalar_fit_artifact_sha256
            ),
            "joint_fit_artifact_sha256": (
                self.gain_support.joint_fit_artifact_sha256
            ),
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_support_rows_sha256": (
                self.gain_support.base_h4_support_rows_sha256
            ),
            "support_mask_sha256": self.support_mask_sha256,
            "teacher_logits_sha256": self.teacher_logits_sha256,
            "endpoint_supervised_grid_sha256": (
                self.endpoint_supervised_grid_sha256
            ),
            "supervised_token_count": int(
                self.static_plus_token_teacher_kl.numel()
            ),
            "pinned_unit_receipt_sha256": self.pinned_unit_receipt_sha256,
            "pinned_unit_token_teacher_kl_sha256": (
                self.pinned_unit_token_teacher_kl_sha256
            ),
            "pinned_unit_mean_teacher_kl": self.pinned_unit_mean_teacher_kl,
            "pinned_v5_static_plus_token_teacher_kl_sha256": (
                self.pinned_v5_static_plus_token_teacher_kl_sha256
            ),
            "static_plus_replayed_pinned_v5_exactly": (
                True
                if self.pinned_v5_static_plus_token_teacher_kl_sha256
                is not None
                else None
            ),
            "arm_provider_artifact_sha256s": tuple(
                zip(THREE_ARM_FINITE_NAMES, self.arm_provider_artifact_sha256s)
            ),
            "arm_execution_artifact_sha256s": tuple(
                zip(THREE_ARM_FINITE_NAMES, self.arm_execution_artifact_sha256s)
            ),
            "arm_correction_rows_sha256s": tuple(
                zip(THREE_ARM_FINITE_NAMES, self.arm_correction_rows_sha256s)
            ),
            "arm_full_correction_sha256s": tuple(
                zip(THREE_ARM_FINITE_NAMES, self.arm_full_correction_sha256s)
            ),
            "arm_token_teacher_kl_sha256s": tuple(arm_hashes.items()),
            "arm_mean_teacher_kl": tuple(arm_means.items()),
            "three_candidate_forwards_executed": True,
            "unit_reference_executed_in_v8": False,
            "same_teacher_logits_and_supervised_grid_for_all_arms": True,
            "held_family_used_for_fit_or_tune": False,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        self.gain_support.validate_integrity()
        if (
            self.static_plus_token_teacher_kl.dtype != torch.float64
            or self.static_plus_token_teacher_kl.device.type != "cpu"
            or self.static_plus_token_teacher_kl.requires_grad
            or not self.static_plus_token_teacher_kl.is_contiguous()
            or self.exact_scalar_token_teacher_kl.dtype != torch.float64
            or self.exact_scalar_token_teacher_kl.device.type != "cpu"
            or self.exact_scalar_token_teacher_kl.requires_grad
            or not self.exact_scalar_token_teacher_kl.is_contiguous()
            or self.joint_token_teacher_kl.dtype != torch.float64
            or self.joint_token_teacher_kl.device.type != "cpu"
            or self.joint_token_teacher_kl.requires_grad
            or not self.joint_token_teacher_kl.is_contiguous()
            or _sha256(_EXAMPLE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="three-arm example")
        ):
            raise RuntimeError("three-arm finite example payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64ThreeArmFiniteComparison:
    """Family-then-prompt-equal summary of the held-only V8 comparison."""

    phase: str
    held_family_ids: tuple[str, ...]
    cell_keys: tuple[tuple[str, str, str], ...]
    example_artifact_sha256s: tuple[str, ...]
    fold_bindings: tuple[tuple[str, tuple[str, ...]], ...]
    prompt_count_by_outer_family: tuple[int, ...]
    outer_family_mean_teacher_kl: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        phase = str(self.phase)
        families = tuple(
            _identifier(value, label="outer held family")
            for value in self.held_family_ids
        )
        keys = tuple(
            (
                _identifier(held, label="cell held family"),
                _identifier(family, label="cell prompt family"),
                _identifier(example, label="cell example"),
            )
            for held, family, example in self.cell_keys
        )
        evidence = tuple(
            _require_sha256(value, label="finite example artifact")
            for value in self.example_artifact_sha256s
        )
        bindings = tuple(
            (
                _identifier(held, label="binding held family"),
                tuple(
                    _require_sha256(value, label="fold binding artifact")
                    for value in binding
                ),
            )
            for held, binding in self.fold_bindings
        )
        counts = tuple(int(value) for value in self.prompt_count_by_outer_family)
        means = _float64(
            self.outer_family_mean_teacher_kl,
            label="outer-family finite means",
            ndim=2,
        )
        if (
            phase != "held"
            or len(families) != _EXPECTED_OUTER_FAMILIES
            or families != tuple(sorted(set(families)))
            or len(keys) != V8_EXPECTED_HELD_CELL_COUNT
            or keys != tuple(sorted(set(keys)))
            or len(evidence) != V8_EXPECTED_HELD_CELL_COUNT
            or len(bindings) != _EXPECTED_OUTER_FAMILIES
            or tuple(held for held, _ in bindings) != families
            or any(len(binding) != 7 for _, binding in bindings)
            or counts
            != (_EXPECTED_HELD_CELLS_PER_FOLD,) * _EXPECTED_OUTER_FAMILIES
            or means.shape != (_EXPECTED_OUTER_FAMILIES, 4)
            or bool((means < 0.0).any())
        ):
            raise ValueError("three-arm finite comparison geometry differs")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "held_family_ids", families)
        object.__setattr__(self, "cell_keys", keys)
        object.__setattr__(self, "example_artifact_sha256s", evidence)
        object.__setattr__(self, "fold_bindings", bindings)
        object.__setattr__(self, "prompt_count_by_outer_family", counts)
        object.__setattr__(self, "outer_family_mean_teacher_kl", means)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_COMPARISON_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def cell_count(self) -> int:
        return len(self.cell_keys)

    def family_equal_mean_teacher_kl(self, arm: str) -> float:
        columns = ("pinned_unit_reference",) + THREE_ARM_FINITE_NAMES
        if arm not in columns:
            raise ValueError("unknown finite comparison arm")
        return math.fsum(
            float(value)
            for value in self.outer_family_mean_teacher_kl[:, columns.index(arm)]
        ) / _EXPECTED_OUTER_FAMILIES

    def _pairwise(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for candidate, source in (
            ("v6_exact_scalar", "v5_static_plus"),
            ("v7_joint", "v5_static_plus"),
            ("v7_joint", "v6_exact_scalar"),
        ):
            candidate_mean = self.family_equal_mean_teacher_kl(candidate)
            source_mean = self.family_equal_mean_teacher_kl(source)
            candidate_column = THREE_ARM_FINITE_NAMES.index(candidate) + 1
            source_column = THREE_ARM_FINITE_NAMES.index(source) + 1
            per_family_delta = (
                self.outer_family_mean_teacher_kl[:, candidate_column]
                - self.outer_family_mean_teacher_kl[:, source_column]
            )
            per_family_ratio = tuple(
                _finite_ratio(float(candidate_value), float(source_value))
                for candidate_value, source_value in zip(
                    self.outer_family_mean_teacher_kl[:, candidate_column],
                    self.outer_family_mean_teacher_kl[:, source_column],
                )
            )
            result[f"{candidate}_vs_{source}"] = {
                "family_equal_delta_candidate_minus_source": (
                    candidate_mean - source_mean
                ),
                "family_macro_relative_improvement": (
                    (source_mean - candidate_mean)
                    / max(source_mean, torch.finfo(torch.float64).tiny)
                ),
                "family_equal_ratio_candidate_over_source": _finite_ratio(
                    candidate_mean, source_mean
                ),
                "outer_family_strict_improvement_count": int(
                    (per_family_delta < 0.0).sum()
                ),
                "outer_family_tie_count": int((per_family_delta == 0.0).sum()),
                "worst_outer_family_ratio_candidate_over_source": max(
                    per_family_ratio
                ),
                "outer_family_ratio_candidate_over_source": per_family_ratio,
                "outer_family_delta_candidate_minus_source": tuple(
                    float(value) for value in per_family_delta
                ),
            }
        return result

    def _joint_control_gates(self) -> dict[str, dict[str, object]]:
        pairwise = self._pairwise()
        result: dict[str, dict[str, object]] = {}
        for source in ("v5_static_plus", "v6_exact_scalar"):
            row = pairwise[f"v7_joint_vs_{source}"]
            candidate_column = THREE_ARM_FINITE_NAMES.index("v7_joint") + 1
            source_column = THREE_ARM_FINITE_NAMES.index(source) + 1
            candidate = self.outer_family_mean_teacher_kl[:, candidate_column]
            control = self.outer_family_mean_teacher_kl[:, source_column]
            gates = {
                "family_macro_relative_improvement_at_least_2pct": (
                    float(row["family_macro_relative_improvement"])
                    >= _USEFUL_RELATIVE_IMPROVEMENT
                ),
                "held_family_improvement_count_at_least_6_of_8": (
                    int(row["outer_family_strict_improvement_count"])
                    >= _MINIMUM_FAMILY_WIN_COUNT
                ),
                "every_family_within_1_05_times_control_plus_1e_minus_8": all(
                    float(candidate_value)
                    <= _WORST_FAMILY_RATIO * float(control_value)
                    + _ABSOLUTE_KL_TOLERANCE
                    for candidate_value, control_value in zip(candidate, control)
                ),
            }
            result[source] = {
                "gates": tuple(sorted(gates.items())),
                "passed": all(gates.values()),
                "required_family_macro_relative_improvement": (
                    _USEFUL_RELATIVE_IMPROVEMENT
                ),
                "required_held_family_improvement_count": (
                    _MINIMUM_FAMILY_WIN_COUNT
                ),
                "worst_family_ratio_cap": _WORST_FAMILY_RATIO,
                "absolute_KL_tolerance": _ABSOLUTE_KL_TOLERANCE,
            }
        return result

    @property
    def joint_vs_static_plus_passed(self) -> bool:
        return bool(self._joint_control_gates()["v5_static_plus"]["passed"])

    @property
    def joint_vs_exact_scalar_passed(self) -> bool:
        return bool(self._joint_control_gates()["v6_exact_scalar"]["passed"])

    @property
    def joint_cleared_both_controls(self) -> bool:
        return self.joint_vs_static_plus_passed and self.joint_vs_exact_scalar_passed

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        columns = ("pinned_unit_reference",) + THREE_ARM_FINITE_NAMES
        result: dict[str, object] = {
            "phase": self.phase,
            "held_family_ids": self.held_family_ids,
            "cell_keys": self.cell_keys,
            "example_artifact_sha256s": self.example_artifact_sha256s,
            "fold_bindings": self.fold_bindings,
            "cell_count": self.cell_count,
            "candidate_execution_count": self.cell_count * len(THREE_ARM_FINITE_NAMES),
            "prompt_count_by_outer_family": self.prompt_count_by_outer_family,
            "outer_family_mean_teacher_kl_sha256": _runtime_tensor_sha256(
                self.outer_family_mean_teacher_kl
            ),
            "family_equal_mean_teacher_kl": tuple(
                (name, self.family_equal_mean_teacher_kl(name))
                for name in columns
            ),
            "pairwise": self._pairwise(),
            "joint_vs_each_control_gates": self._joint_control_gates(),
            "joint_cleared_both_controls": self.joint_cleared_both_controls,
            "aggregation": (
                "two_prompts_equal_within_family_then_equal_over_eight_families"
            ),
            "arms_selected_or_ranked": False,
            "finite_thresholds_source": "unchanged_v5_held_teacher_KL_gates",
            "top1_gates_applied_in_core": False,
            "unit_reference_executed_in_v8": False,
            "held_only_no_tune_selection_or_refit": True,
            "v5_static_plus_role": (
                "counterfactual_plus_carrier_not_v5_selected_arm"
            ),
            "inherited_parent_forward_count": 112,
            "held_native_forward_count": 16,
            "held_candidate_forward_count": 48,
            "total_forward_count": 176,
            "inherited_total_backward_call_count": 494,
            "v8_additional_backward_call_count": 0,
            "total_backward_call_count": 494,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            self.outer_family_mean_teacher_kl.dtype != torch.float64
            or self.outer_family_mean_teacher_kl.device.type != "cpu"
            or self.outer_family_mean_teacher_kl.requires_grad
            or not self.outer_family_mean_teacher_kl.is_contiguous()
            or _sha256(_COMPARISON_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="finite comparison")
        ):
            raise RuntimeError("three-arm finite comparison payload drifted")


def compare_candidate_conditioned_k64_three_arm_finite_examples(
    examples: Sequence[CandidateConditionedK64ThreeArmFiniteExample],
    *,
    phase: str = "held",
) -> CandidateConditionedK64ThreeArmFiniteComparison:
    """Validate all sixteen held cells and compare joint to both controls."""

    supplied = tuple(examples)
    if any(
        not isinstance(value, CandidateConditionedK64ThreeArmFiniteExample)
        for value in supplied
    ):
        raise TypeError("finite comparison evidence must be typed V8 examples")
    values = tuple(
        sorted(
            supplied,
            key=lambda value: (
                value.held_family_id,
                value.family_id,
                value.example_id,
            ),
        )
    )
    for value in values:
        value.validate_integrity()
    if not values or any(value.phase != phase for value in values):
        raise ValueError("finite comparison phase differs")
    held_families = tuple(sorted({value.held_family_id for value in values}))
    prompt_families = {value.family_id for value in values}
    if (
        len(held_families) != _EXPECTED_OUTER_FAMILIES
        or prompt_families != set(held_families)
    ):
        raise ValueError("finite comparison outer family universe differs")
    by_held: dict[str, list[CandidateConditionedK64ThreeArmFiniteExample]] = {
        held: [] for held in held_families
    }
    for value in values:
        by_held[value.held_family_id].append(value)
    if phase != "held" or len(values) != V8_EXPECTED_HELD_CELL_COUNT:
        raise ValueError("finite comparison cell count differs")
    held_ids: set[str] = set()
    for held, cells in by_held.items():
        if (
            len(cells) != _EXPECTED_HELD_CELLS_PER_FOLD
            or any(value.family_id != held for value in cells)
            or any(value.example_id in held_ids for value in cells)
        ):
            raise ValueError("outer-held prompt coverage differs")
        held_ids.update(value.example_id for value in cells)
    fold_bindings: list[tuple[str, tuple[str, ...]]] = []
    outer_means: list[Tensor] = []
    for held in held_families:
        cells = by_held[held]
        bindings = {value.gain_support.fold_binding for value in cells}
        if len(bindings) != 1:
            raise ValueError("frozen fold binding varies across finite prompts")
        fold_bindings.append((held, next(iter(bindings))))
        cell_rows = tuple(
            (
                value.pinned_unit_mean_teacher_kl,
                *(value.mean_teacher_kl(arm) for arm in THREE_ARM_FINITE_NAMES),
            )
            for value in cells
        )
        outer_means.append(
            torch.tensor(
                tuple(
                    math.fsum(row[column] for row in cell_rows)
                    / _EXPECTED_HELD_CELLS_PER_FOLD
                    for column in range(4)
                ),
                dtype=torch.float64,
            )
        )
    return CandidateConditionedK64ThreeArmFiniteComparison(
        phase=phase,
        held_family_ids=held_families,
        cell_keys=tuple(
            (value.held_family_id, value.family_id, value.example_id)
            for value in values
        ),
        example_artifact_sha256s=tuple(value.artifact_sha256 for value in values),
        fold_bindings=tuple(fold_bindings),
        prompt_count_by_outer_family=tuple(
            len(by_held[held]) for held in held_families
        ),
        outer_family_mean_teacher_kl=torch.stack(outer_means),
    )
