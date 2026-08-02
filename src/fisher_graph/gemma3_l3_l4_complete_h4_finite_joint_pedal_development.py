"""V19 finite-objective joint direction/pedal outer-LOFO development rung.

This rung keeps the authenticated V18 K256 parent, Fisher/PCA router, rank-16
conditional direction, and pointwise 0.25 trust ball.  It changes the fit
authority: direction factors and a sigmoid pedal are jointly updated against
the *executed* float64 full-vocabulary ``KL(source || candidate)`` through the
complete Gemma suffix.  The fixed optimizer is four full-batch Adam steps and
checkpoints zero through four are all finite-evaluated before the earliest
minimum is frozen.

The A16 panel remains an outer leave-one-family-out development screen.  A
fold capability exposes source-teacher rows for only the fourteen training
prompts; held rows can be transiently materialized by the one-time collection
pass, but are capability-excluded and never consumed by that fold's fit.  Held
evaluation starts only after all checkpoint and provider hashes are frozen.
No guard or Calibration B is opened, and this module makes no serving,
compression, speed, or end-to-end FLOP claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from .complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
    AutonomousCompleteH4TrainingSequence,
)
from .complete_h4_fisher_conditional_pedal import (
    AutonomousCompleteH4FisherXYPedalProvider,
    fisher_xy_pedal_fit_support_mask,
    fisher_xy_pointwise_bounded_direction,
)
from .complete_h4_fisher_conditional_residual import _training_parent_modal
from .complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict,
    canonical_balanced_rank_svd_retraction,
    fisher_finite_joint_pedal_control,
    fisher_finite_joint_modal_delta,
    load_autonomous_complete_h4_fisher_finite_joint_pedal_provider,
    refit_autonomous_complete_h4_fisher_finite_joint_pedal,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_fisher_pedal_development as _v18
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassExecution,
    Gemma3L3L4OnePassPrefix,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_PROVIDER_OUTPUT",
    "PARENT_ID",
    "FISHER_START_ID",
    "FISHER_UNIT_ID",
    "FISHER_INTERCEPT_ID",
    "FISHER_CONDITIONAL_ID",
    "PCA_CONDITIONAL_ID",
    "exact_float64_teacher_kl",
    "canonical_balanced_rank_factors",
    "choose_earliest_checkpoint",
    "build_finite_joint_pedal_development_report",
    "run_gemma3_l3_l4_complete_h4_finite_joint_pedal_development",
    "build_parser",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_V18_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-fisher-pedal-r16-k256-"
    "outer-lofo-a-fit16-dev-v18.json"
)
_V18_LOGICAL_SHA256 = (
    "93a757c2efb5388000536a259eb5e654ece9691a3ba016d9404fec152a65c641"
)
_V18_FILE_SHA256 = (
    "95394ee0a643e13d959eff5a5d1180643ef59cc4c5d1d56652e5480229f27c81"
)
_V18_CLASSIFICATION = "fisher_pedal_pointwise_trust_insufficient"

DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-finite-joint-pedal-r16-k256-"
    "outer-lofo-a-fit16-dev-v19.json"
)
DEFAULT_PROVIDER_OUTPUT = DEFAULT_OUTPUT.with_suffix(".provider.pt")
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_finite_joint_pedal_"
    "outer_lofo_development.v19"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-finite-joint-pedal-dev:v19\0"
_COLLECTION_DOMAIN = b"fisher-graph:finite-joint-pedal-collection:v19\0"
_TEACHER_VAULT_DOMAIN = b"fisher-graph:finite-joint-pedal-teacher-vault:v19\0"
_CAPABILITY_DOMAIN = b"fisher-graph:finite-joint-pedal-capability:v19\0"
_PROTOCOL_DOMAIN = b"fisher-graph:finite-joint-pedal-protocol:v19\0"
_TRAJECTORY_DOMAIN = b"fisher-graph:finite-joint-pedal-trajectory:v19\0"
_CHECKPOINT_DOMAIN = b"fisher-graph:finite-joint-pedal-checkpoint:v19\0"
_OWNERSHIP_DOMAIN = b"fisher-graph:finite-joint-pedal-ownership:v19\0"
_RUNTIME_DOMAIN = b"fisher-graph:finite-joint-pedal-held-runtime:v19\0"

_EXPECTED_PROMPTS = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_TRAINING_PROMPTS_PER_FOLD = 14
_EXPECTED_HELD_PROMPTS_PER_FOLD = 2
_EXPECTED_OUTER_FOLDS = 8
_EXPECTED_OUTER_PROVIDER_FITS = 40
_EXPECTED_FULL_MODEL_FORWARDS = 1_280
_EXPECTED_BACKWARD_TRAVERSALS = 912
_EXPECTED_CAUSAL_CHECKS = 96
_EXPECTED_VOCABULARY = 262_144
_EXPECTED_OPTIMIZER_FORWARDS_PER_ARM_FOLD = 70
_EXPECTED_OPTIMIZER_BACKWARDS_PER_ARM_FOLD = 56

_CONDITIONAL_RANK = 16
_OPTIMIZER_STEPS = 4
_DIRECTION_LEARNING_RATE = 1.0e-3
_PEDAL_LEARNING_RATE = 2.5e-2
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPSILON = 1.0e-8
_TRUST_FRACTION = 0.25

PARENT_ID = "r256_l8_reverse_vjp_parent"
FISHER_START_ID = "r256_l8_reverse_vjp_fisher_v18_start_r16"
FISHER_UNIT_ID = "r256_l8_reverse_vjp_fisher_finite_joint_unit_r16"
FISHER_INTERCEPT_ID = "r256_l8_reverse_vjp_fisher_finite_joint_intercept_r16"
FISHER_CONDITIONAL_ID = "r256_l8_reverse_vjp_fisher_finite_joint_conditional_r16"
PCA_CONDITIONAL_ID = "r256_l8_reverse_vjp_activation_pca_finite_joint_conditional_r16"
_ARM_IDS = (
    PARENT_ID,
    FISHER_START_ID,
    FISHER_UNIT_ID,
    FISHER_INTERCEPT_ID,
    FISHER_CONDITIONAL_ID,
    PCA_CONDITIONAL_ID,
)
_JOINT_IDS = (
    FISHER_UNIT_ID,
    FISHER_INTERCEPT_ID,
    FISHER_CONDITIONAL_ID,
    PCA_CONDITIONAL_ID,
)
_REQUIRED_LEDGERS = ("ordinary", "complete_h4_support", "graph_core")

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "finite_full_suffix_teacher_kl_joint_direction_sigmoid_pedal",
    "outer_split": "eight_family_leave_one_family_out",
    "parent": "K256_L8_reverse_VJP_weighted_V18_parent",
    "coordinate_initializers": ("reverse_vjp_fisher", "activation_pca"),
    "conditional_rank": _CONDITIONAL_RANK,
    "initial_direction_matrix": "two_times_V18_direction_product",
    "initial_factorization": "canonical_balanced_rank16_SVD",
    "initial_pedal_slopes": (0.0, 0.0, 0.0),
    "initial_pedal_intercept": 0.0,
    "pedal": "sigmoid",
    "trust_fraction": _TRUST_FRACTION,
    "optimizer": "full_batch_float64_Adam",
    "steps": _OPTIMIZER_STEPS,
    "direction_factor_learning_rate": _DIRECTION_LEARNING_RATE,
    "pedal_learning_rate": _PEDAL_LEARNING_RATE,
    "adam_betas": (_ADAM_BETA1, _ADAM_BETA2),
    "adam_epsilon": _ADAM_EPSILON,
    "weight_decay": 0.0,
    "gradient_clipping": False,
    "accumulation_order": "family_id_then_example_id",
    "objective": (
        "family_equal_example_equal_token_mean_exact_float64_full_vocab_"
        "KL_source_to_candidate_through_full_suffix"
    ),
    "full_vocabulary_width": _EXPECTED_VOCABULARY,
    "checkpoint_selection": "minimum_training_objective_earliest_exact_tie",
    "checkpoint_indices": (0, 1, 2, 3, 4),
    "h4_cast_derivative": "straight_through_estimator_at_cast_once_boundary",
}
_FIT_PROTOCOL_SHA256 = _v14._sha256(_FIXED_PROTOCOL, domain=_PROTOCOL_DOMAIN)

_OUTER_THRESHOLDS = {
    "parent_absolute_delta_nll_relative_improvement_min": 0.05,
    "parent_kl_relative_improvement_min": 0.05,
    "parent_aggregate_top1_gain_min": 0.02,
    "parent_family_absolute_delta_nll_win_count_min": 6,
    "parent_worst_family_relative_regression_max": 0.02,
    "start_absolute_delta_nll_relative_improvement_min": 0.01,
    "start_family_absolute_delta_nll_win_count_min": 5,
    "intercept_absolute_delta_nll_relative_improvement_min": 0.01,
    "intercept_family_absolute_delta_nll_win_count_min": 5,
    "unit_family_absolute_delta_nll_win_count_min": 5,
    "objective_macro_relative_improvement_min": 0.01,
    "objective_absolute_improvement_floor": 1.0e-12,
    "objective_roundoff_multiplier": 128.0,
    "pedal_variation_min": math.sqrt(torch.finfo(torch.float64).eps),
}


def _is_under_local_runs(path: Path) -> bool:
    return ".local-runs" in path.resolve(strict=False).parts


def _same_destination(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _sha256_identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one canonical lowercase SHA-256")
    return value


def _validate_output(path: Path | str) -> Path:
    selected = Path(path)
    if selected.suffix != ".json" or not _is_under_local_runs(selected):
        raise ValueError("V19 output must be JSON under .local-runs")
    return selected


def _validate_provider_output(path: Path | str) -> Path:
    selected = Path(path)
    if selected.suffix != ".pt" or not _is_under_local_runs(selected):
        raise ValueError("V19 provider output must be PT under .local-runs")
    return selected


def _validate_prerequisites() -> dict[str, object]:
    receipt = _v18._validate_prerequisite_report(
        _V18_OUTPUT,
        logical_sha256=_V18_LOGICAL_SHA256,
        file_sha256=_V18_FILE_SHA256,
        classification=_V18_CLASSIFICATION,
        format_version=18,
    )
    payload = json.loads(_V18_OUTPUT.read_text(encoding="utf-8"))
    if payload.get("candidate") is not None:
        raise RuntimeError("V19 requires the pinned V18 null candidate")
    geometry = payload.get("coordinate_geometry_qualification")
    arms = payload.get("arms")
    if not isinstance(geometry, Mapping) or geometry.get("passed") is not True:
        raise RuntimeError("V19 requires the pinned passed V18 coordinate geometry")
    if not isinstance(arms, Mapping):
        raise RuntimeError("V18 prerequisite arm rows differ")
    lineage: dict[str, object] = {}
    for arm_id in (PARENT_ID, _v18.FISHER_PEDAL_ID, _v18.PCA_PEDAL_ID):
        arm = arms.get(arm_id)
        hashes = (
            arm.get("fold_provider_artifact_sha256s")
            if isinstance(arm, Mapping)
            else None
        )
        if not isinstance(hashes, Mapping) or len(hashes) != _EXPECTED_OUTER_FOLDS:
            raise RuntimeError("V18 prerequisite fold lineage differs")
        lineage[arm_id] = dict(hashes)
    return {
        "v18": receipt,
        "v18_candidate_was_null": True,
        "v18_coordinate_geometry_qualification": dict(geometry),
        "v18_fold_provider_artifact_sha256s": lineage,
    }


def exact_float64_teacher_kl(
    teacher_logits: Tensor,
    candidate_logits: Tensor,
) -> Tensor:
    """Return token-mean full-vocabulary ``KL(teacher || candidate)``.

    Both operands are promoted before any probability arithmetic.  The
    teacher is detached so source rows can never become an optimization path.
    Neither input is mutated.
    """

    if (
        not isinstance(teacher_logits, Tensor)
        or not isinstance(candidate_logits, Tensor)
        or teacher_logits.ndim != 2
        or teacher_logits.shape != candidate_logits.shape
        or int(teacher_logits.shape[0]) <= 0
        or int(teacher_logits.shape[1]) <= 1
        or not teacher_logits.is_floating_point()
        or not candidate_logits.is_floating_point()
        or not bool(torch.isfinite(teacher_logits).all())
        or not bool(torch.isfinite(candidate_logits).all())
    ):
        raise ValueError("teacher-KL operands must be matching finite [N,V] tensors")
    teacher = teacher_logits.detach().to(
        device=candidate_logits.device,
        dtype=torch.float64,
    )
    candidate = candidate_logits.to(dtype=torch.float64)
    teacher_logp = F.log_softmax(teacher, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    token_kl = (teacher_logp.exp() * (teacher_logp - candidate_logp)).sum(dim=-1)
    result = token_kl.mean()
    if result.dtype != torch.float64 or result.ndim != 0 or not bool(torch.isfinite(result)):
        raise RuntimeError("float64 teacher KL became invalid")
    return result


def canonical_balanced_rank_factors(
    matrix: Tensor,
    *,
    rank: int,
) -> tuple[Tensor, Tensor]:
    """Return deterministic SVD-balanced ``L,R`` with ``L@R ~= matrix``."""

    if (
        not isinstance(matrix, Tensor)
        or matrix.ndim != 2
        or not matrix.is_floating_point()
        or not bool(torch.isfinite(matrix).all())
        or type(rank) is not int
        or rank <= 0
        or rank > min(int(matrix.shape[0]), int(matrix.shape[1]))
    ):
        raise ValueError("balanced factorization requires a finite matrix and valid rank")
    values = matrix.detach().to(device="cpu", dtype=torch.float64).contiguous()
    left, right = canonical_balanced_rank_svd_retraction(values, rank=rank)
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise RuntimeError("balanced factors became nonfinite")
    return left, right


def choose_earliest_checkpoint(scores: Sequence[float]) -> int:
    """Choose the smallest checkpoint at the exact minimum finite score."""

    selected = tuple(float(value) for value in scores)
    if not selected or any(not math.isfinite(value) for value in selected):
        raise ValueError("checkpoint scores must be a nonempty finite sequence")
    minimum = min(selected)
    return selected.index(minimum)


@dataclass(frozen=True, slots=True)
class _JointState:
    direction_left: Tensor
    direction_right: Tensor
    pedal_weight: Tensor
    pedal_bias: Tensor

    def __post_init__(self) -> None:
        selected: dict[str, Tensor] = {}
        for name in (
            "direction_left",
            "direction_right",
            "pedal_weight",
            "pedal_bias",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.device.type == "meta"
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{name} must be a finite floating tensor")
            selected[name] = value.detach().to(
                device="cpu", dtype=torch.float64
            ).contiguous().clone()
        if (
            selected["direction_left"].ndim != 2
            or selected["direction_right"].ndim != 2
            or selected["direction_left"].shape[1] != _CONDITIONAL_RANK
            or selected["direction_right"].shape[0] != _CONDITIONAL_RANK
            or selected["direction_left"].shape[0]
            != 3 * selected["direction_right"].shape[1]
            or selected["pedal_weight"].shape != (3,)
            or selected["pedal_bias"].shape != (1,)
        ):
            raise ValueError("joint parameter state geometry differs")
        for name, value in selected.items():
            object.__setattr__(self, name, value)

    def receipt(self) -> dict[str, object]:
        return {
            "direction_left_sha256": _v14._tensor_sha256(self.direction_left),
            "direction_right_sha256": _v14._tensor_sha256(self.direction_right),
            "pedal_weight_sha256": _v14._tensor_sha256(self.pedal_weight),
            "pedal_bias_sha256": _v14._tensor_sha256(self.pedal_bias),
        }


@dataclass(frozen=True, slots=True)
class _AdamMoments:
    first: _JointState
    second: _JointState
    step: int


def _zero_state(like: _JointState) -> _JointState:
    return _JointState(
        direction_left=torch.zeros_like(like.direction_left),
        direction_right=torch.zeros_like(like.direction_right),
        pedal_weight=torch.zeros_like(like.pedal_weight),
        pedal_bias=torch.zeros_like(like.pedal_bias),
    )


def _adam_step(
    state: _JointState,
    gradients: _JointState,
    moments: _AdamMoments,
) -> tuple[_JointState, _AdamMoments]:
    """Apply one preregistered no-decay, no-clipping float64 Adam step."""

    if moments.step < 0:
        raise ValueError("Adam step must be nonnegative")
    next_step = moments.step + 1
    first_values: dict[str, Tensor] = {}
    second_values: dict[str, Tensor] = {}
    parameter_values: dict[str, Tensor] = {}
    for name in (
        "direction_left",
        "direction_right",
        "pedal_weight",
        "pedal_bias",
    ):
        parameter = getattr(state, name)
        gradient = getattr(gradients, name)
        first = _ADAM_BETA1 * getattr(moments.first, name) + (
            1.0 - _ADAM_BETA1
        ) * gradient
        second = _ADAM_BETA2 * getattr(moments.second, name) + (
            1.0 - _ADAM_BETA2
        ) * gradient.square()
        first_hat = first / (1.0 - _ADAM_BETA1**next_step)
        second_hat = second / (1.0 - _ADAM_BETA2**next_step)
        learning_rate = (
            _DIRECTION_LEARNING_RATE
            if name in ("direction_left", "direction_right")
            else _PEDAL_LEARNING_RATE
        )
        updated = parameter - learning_rate * first_hat / (
            torch.sqrt(second_hat) + _ADAM_EPSILON
        )
        if not bool(torch.isfinite(updated).all()):
            raise RuntimeError("Adam update became nonfinite")
        first_values[name] = first
        second_values[name] = second
        parameter_values[name] = updated
    updated_state = _JointState(**parameter_values)
    updated_moments = _AdamMoments(
        first=_JointState(**first_values),
        second=_JointState(**second_values),
        step=next_step,
    )
    return updated_state, updated_moments


class _TeacherRowCapability:
    """A fold-scoped view that cannot name or retrieve held teacher rows."""

    def __init__(
        self,
        rows: Mapping[str, Tensor],
        families: Mapping[str, str],
        *,
        held_family_id: str | None,
    ) -> None:
        if not rows or set(rows) != set(families):
            raise ValueError("teacher capability rows/families differ")
        held = None if held_family_id is None else _v14._identifier(
            held_family_id, label="held family"
        )
        if held is not None and held in set(families.values()):
            raise ValueError("held family entered teacher capability")
        self._rows = dict(rows)
        self._families = dict(families)
        self._hashes = {
            key: _v14._tensor_sha256(value) for key, value in self._rows.items()
        }
        self._held_family_id = held
        self._accesses: list[str] = []
        self._artifact_sha256 = _v14._sha256(
            {
                "authorized_example_ids": tuple(sorted(self._rows)),
                "authorized_family_ids": tuple(sorted(set(self._families.values()))),
                "held_family_id": held,
                "teacher_row_sha256s": dict(sorted(self._hashes.items())),
            },
            domain=_CAPABILITY_DOMAIN,
        )

    @property
    def access_count(self) -> int:
        return len(self._accesses)

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    def get(self, example_id: str, *, family_id: str) -> Tensor:
        example = _v14._identifier(example_id, label="teacher example")
        family = _v14._identifier(family_id, label="teacher family")
        if example not in self._rows or self._families.get(example) != family:
            raise PermissionError("teacher row is outside this fold capability")
        value = self._rows[example]
        if _v14._tensor_sha256(value) != self._hashes[example]:
            raise RuntimeError("teacher capability tensor mutated")
        self._accesses.append(example)
        return value

    def receipt(self) -> dict[str, object]:
        if any(_v14._tensor_sha256(self._rows[key]) != value for key, value in self._hashes.items()):
            raise RuntimeError("teacher capability payload drifted")
        counts = {
            example: self._accesses.count(example) for example in sorted(self._rows)
        }
        return {
            "artifact_sha256": self.artifact_sha256,
            "held_family_id": self._held_family_id,
            "authorized_example_count": len(self._rows),
            "authorized_family_count": len(set(self._families.values())),
            "access_count": self.access_count,
            "per_example_access_counts": counts,
            "held_family_capability_excluded": self._held_family_id is not None,
            "teacher_rows_consumed_only_through_capability": True,
        }


class _TeacherRowVault:
    """Immutable CPU snapshot with authenticated fold capability issuance."""

    def __init__(
        self,
        rows: Mapping[str, Tensor],
        families: Mapping[str, str],
    ) -> None:
        if not rows or set(rows) != set(families):
            raise ValueError("teacher vault rows/families differ")
        copied: dict[str, Tensor] = {}
        for key, value in rows.items():
            example = _v14._identifier(key, label="teacher vault example")
            _v14._identifier(families[key], label="teacher vault family")
            if (
                not isinstance(value, Tensor)
                or value.ndim != 2
                or not value.is_floating_point()
                or int(value.shape[0]) <= 0
                or int(value.shape[1]) <= 1
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("teacher vault needs finite [N,V] rows")
            copied[example] = value.detach().to(device="cpu").contiguous().clone()
        self._rows = copied
        self._families = dict(families)
        self._hashes = {
            key: _v14._tensor_sha256(value) for key, value in copied.items()
        }
        self._artifact_sha256 = _v14._sha256(
            {
                "example_family_ids": dict(sorted(self._families.items())),
                "teacher_row_sha256s": dict(sorted(self._hashes.items())),
            },
            domain=_TEACHER_VAULT_DOMAIN,
        )

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    def validate_integrity(self) -> None:
        if any(_v14._tensor_sha256(self._rows[key]) != value for key, value in self._hashes.items()):
            raise RuntimeError("teacher vault tensor mutated")
        if _v14._sha256(
            {
                "example_family_ids": dict(sorted(self._families.items())),
                "teacher_row_sha256s": dict(sorted(self._hashes.items())),
            },
            domain=_TEACHER_VAULT_DOMAIN,
        ) != self.artifact_sha256:
            raise RuntimeError("teacher vault receipt drifted")

    def capability(
        self,
        authorized_example_ids: Sequence[str],
        *,
        held_family_id: str | None,
    ) -> _TeacherRowCapability:
        self.validate_integrity()
        selected = tuple(sorted(authorized_example_ids))
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("teacher capability authorization differs")
        if any(key not in self._rows for key in selected):
            raise PermissionError("unknown example requested from teacher vault")
        return _TeacherRowCapability(
            {key: self._rows[key] for key in selected},
            {key: self._families[key] for key in selected},
            held_family_id=held_family_id,
        )

    def receipt(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "artifact_sha256": self.artifact_sha256,
            "example_count": len(self._rows),
            "family_count": len(set(self._families.values())),
            "teacher_row_sha256s": dict(sorted(self._hashes.items())),
            "source_rows_cached_in_native_dtype_on_cpu": True,
            "float64_teacher_log_probabilities_or_probabilities_cached": False,
        }


def _collect_fit_records_and_teacher_vault(
    context: object,
) -> tuple[tuple[_v14._FitRecord, ...], _TeacherRowVault]:
    """Collect V14 traces while retaining only supervised source-logit rows."""

    bridge = getattr(context, "bridge")
    adapter = getattr(context, "adapter")
    tokenize = getattr(context, "tokenize")
    bridge_binding = _v14._identifier(
        getattr(bridge, "bridge_binding_sha256", None),
        label="bridge binding",
    )
    records: list[_v14._FitRecord] = []
    teacher_rows: dict[str, Tensor] = {}
    teacher_families: dict[str, str] = {}
    for example in sorted(
        tuple(getattr(context, "examples")),
        key=lambda value: _v14._identifier(
            getattr(value, "example_id", None), label="example_id"
        ),
    ):
        example_id = _v14._identifier(
            getattr(example, "example_id", None), label="example_id"
        )
        family_id = _v14._identifier(
            getattr(example, "family_id", None), label="family_id"
        )
        model_inputs, supervised_indices, supervised_targets = _v14._retokenize(
            tokenize, example
        )
        model_inputs_hash = gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        base, gradient = bridge.execute_h4_vjp(
            adapter,
            model_inputs,
            objective=_v14._mean_supervised_nll(
                supervised_indices, supervised_targets
            ),
        )
        source_logits, native_h4, native_positions, native_valid = _v14._native_boundary(
            adapter, model_inputs
        )
        selected_teacher = _v14._select_sequence_rows(
            source_logits, supervised_indices
        ).detach().to(device="cpu").contiguous()
        prefix = getattr(base, "prefix", None)
        base_h4 = getattr(base, "candidate_h4", None)
        if (
            not isinstance(prefix, Gemma3L3L4OnePassPrefix)
            or not isinstance(base_h4, Tensor)
            or not isinstance(gradient, Tensor)
            or base_h4.shape != native_h4.shape
            or gradient.shape != base_h4.shape
            or base_h4.shape[0] != 1
            or getattr(base, "model_inputs_sha256", None) != model_inputs_hash
            or getattr(base, "bridge_binding_sha256", None) != bridge_binding
            or selected_teacher.shape[0] != supervised_indices.numel()
            or selected_teacher.shape[1] != _EXPECTED_VOCABULARY
        ):
            raise RuntimeError("V19 collection boundary binding differs")
        prefix.validate_integrity()
        support = prefix.complete_h4_causal_support_mask()[0].detach().to(
            device="cpu"
        )
        source = prefix.source_eligible_mask[0].detach().to(device="cpu")
        valid = prefix.valid_target_mask[0].detach().to(device="cpu")
        core = prefix.target_affected_mask[0].detach().to(device="cpu")
        if (
            not _v14._bitwise_equal(native_positions, prefix.logical_positions)
            or not _v14._bitwise_equal(native_valid, prefix.valid_target_mask)
            or bool((core & ~support).any())
            or bool((support & ~valid).any())
            or not _v14._bitwise_equal(
                native_h4[0][~support.to(native_h4.device)],
                base_h4[0][~support.to(base_h4.device)],
            )
        ):
            raise RuntimeError("V19 collection causal support differs")
        sequence = AutonomousCompleteH4TrainingSequence(
            example_id=example_id,
            family_id=family_id,
            source_modes=prefix.source_modes[0],
            logical_positions=prefix.logical_positions[0],
            valid_mask=valid,
            source_mask=source,
            support_mask=support,
            base_h4=base_h4[0],
            native_h4=native_h4[0],
            reverse_vjp_gradients=gradient[0],
        )
        positions = supervised_indices.detach().to(device="cpu")
        support_supervised = support.index_select(0, positions)
        core_supervised = core.index_select(0, positions)
        ledger_indices = {
            "ordinary": torch.arange(positions.numel(), dtype=torch.int64),
            "complete_h4_support": torch.nonzero(
                support_supervised, as_tuple=False
            ).flatten(),
            "graph_core": torch.nonzero(
                core_supervised, as_tuple=False
            ).flatten(),
            "causal_tail": torch.nonzero(
                support_supervised & ~core_supervised, as_tuple=False
            ).flatten(),
        }
        receipt = {
            "example_id": example_id,
            "family_id": family_id,
            "prompt_sha256": _v14._prompt_sha256(getattr(example, "prompt")),
            "model_inputs_sha256": model_inputs_hash,
            "supervised_indices_sha256": _v14._tensor_sha256(supervised_indices),
            "supervised_targets_sha256": _v14._tensor_sha256(supervised_targets),
            "training_sequence_sha256": sequence.artifact_sha256,
            "base_execution_sha256": getattr(base, "artifact_sha256"),
            "base_h4_sha256": _v14._tensor_sha256(base_h4),
            "native_h4_sha256": _v14._tensor_sha256(native_h4),
            "reverse_vjp_sha256": _v14._tensor_sha256(gradient),
            "support_mask_sha256": _v14._tensor_sha256(support),
            "teacher_supervised_rows_sha256": _v14._tensor_sha256(
                selected_teacher
            ),
        }
        records.append(
            _v14._FitRecord(
                example=example,
                sequence=sequence,
                prompt_sha256=str(receipt["prompt_sha256"]),
                model_inputs_sha256=model_inputs_hash,
                supervised_indices_sha256=str(
                    receipt["supervised_indices_sha256"]
                ),
                supervised_targets_sha256=str(
                    receipt["supervised_targets_sha256"]
                ),
                supervised_token_count=int(supervised_indices.numel()),
                ledger_indices=ledger_indices,
                receipt_sha256=_v14._sha256(
                    receipt, domain=_COLLECTION_DOMAIN
                ),
            )
        )
        teacher_rows[example_id] = selected_teacher
        teacher_families[example_id] = family_id
        del (
            model_inputs,
            source_logits,
            selected_teacher,
            base,
            gradient,
            native_h4,
        )
    families = {record.sequence.family_id for record in records}
    if (
        len(records) != _EXPECTED_PROMPTS
        or len(families) != _EXPECTED_FAMILIES
        or len(teacher_rows) != _EXPECTED_PROMPTS
    ):
        raise RuntimeError("V19 authenticated A16 collection geometry differs")
    vault = _TeacherRowVault(teacher_rows, teacher_families)
    vault.validate_integrity()
    return tuple(records), vault


def _initial_joint_state(
    start_provider: AutonomousCompleteH4FisherXYPedalProvider,
) -> _JointState:
    start_provider.validate_integrity()
    dense = 2.0 * (
        start_provider.direction_left @ start_provider.direction_right
    )
    left, right = canonical_balanced_rank_factors(
        dense,
        rank=_CONDITIONAL_RANK,
    )
    reconstructed = left @ right
    tolerance = 2.0e-11 * max(1.0, float(torch.linalg.vector_norm(dense)))
    if float(torch.linalg.vector_norm(reconstructed - dense)) > tolerance:
        raise RuntimeError("V19 canonical balanced initialization lost V18 direction")
    return _JointState(
        direction_left=left,
        direction_right=right,
        pedal_weight=torch.zeros(3, dtype=torch.float64),
        pedal_bias=torch.zeros(1, dtype=torch.float64),
    )


def _family_equal_mean(
    prompt_values: Mapping[str, float],
    records: Sequence[_v14._FitRecord],
) -> tuple[float, dict[str, float]]:
    by_family: dict[str, list[float]] = {}
    for record in sorted(
        records,
        key=lambda value: (value.sequence.family_id, value.sequence.example_id),
    ):
        example_id = record.sequence.example_id
        if example_id not in prompt_values:
            raise ValueError("family-equal objective omitted a prompt")
        value = float(prompt_values[example_id])
        if not math.isfinite(value):
            raise ValueError("family-equal objective contains a nonfinite prompt")
        by_family.setdefault(record.sequence.family_id, []).append(value)
    if set(prompt_values) != {
        record.sequence.example_id for record in records
    }:
        raise ValueError("family-equal objective contains an unknown prompt")
    family_means = {
        family: math.fsum(values) / len(values)
        for family, values in sorted(by_family.items())
    }
    return (
        math.fsum(family_means.values()) / len(family_means),
        family_means,
    )


def _teacher_kl_objective(
    teacher_rows: Tensor,
    supervised_indices: Tensor,
) -> tuple[Callable[[object], Tensor], list[float]]:
    teacher_sha = _v14._tensor_sha256(teacher_rows)
    positions = supervised_indices.detach().to(device="cpu").contiguous().clone()
    captured: list[float] = []

    def objective(run: object) -> Tensor:
        logits = getattr(run, "logits", None)
        if not isinstance(logits, Tensor):
            raise TypeError("V19 teacher-KL objective requires adapter logits")
        selected = _v14._select_sequence_rows(logits, positions)
        loss = exact_float64_teacher_kl(teacher_rows, selected)
        captured.append(float(loss.detach()))
        if _v14._tensor_sha256(teacher_rows) != teacher_sha:
            raise RuntimeError("V19 teacher rows mutated inside objective")
        return loss

    return objective, captured


def _provisional_provider(
    start_provider: AutonomousCompleteH4FisherXYPedalProvider,
    state: _JointState,
    *,
    held_family_id: str | None,
    coordinate_objective: str,
    checkpoint: int,
) -> AutonomousCompleteH4FisherFiniteJointPedalProvider:
    evidence = _v14._sha256(
        {
            "status": "finite_checkpoint_execution",
            "held_family_id": held_family_id,
            "coordinate_objective": coordinate_objective,
            "checkpoint": checkpoint,
            "state": state.receipt(),
            "start_provider_artifact_sha256": start_provider.artifact_sha256,
        },
        domain=_CHECKPOINT_DOMAIN,
    )
    return refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start_provider,
        direction_left=state.direction_left,
        direction_right=state.direction_right,
        pedal_weight=state.pedal_weight,
        pedal_bias=state.pedal_bias,
        fit_protocol_sha256=_FIT_PROTOCOL_SHA256,
        fit_evidence_sha256=evidence,
        pedal_mode="conditional",
    )


def _local_ste_parameter_gradients(
    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    state: _JointState,
    sequence: AutonomousCompleteH4TrainingSequence,
    h4_gradient: Tensor,
) -> _JointState:
    """Contract suffix dKL/dH4 through the modal head under cast-once STE."""

    if (
        not isinstance(h4_gradient, Tensor)
        or h4_gradient.shape != (1, *sequence.base_h4.shape)
        or not h4_gradient.is_floating_point()
        or not bool(torch.isfinite(h4_gradient).all())
    ):
        raise ValueError("V19 suffix H4 gradient geometry differs")
    parent = _training_parent_modal(provider.parent_provider, sequence)
    coordinates = provider.bounded_coordinates(parent)
    left = state.direction_left.detach().clone().requires_grad_(True)
    right = state.direction_right.detach().clone().requires_grad_(True)
    weight = state.pedal_weight.detach().clone().requires_grad_(True)
    bias = state.pedal_bias.detach().clone().requires_grad_(True)
    delta = fisher_finite_joint_modal_delta(
        parent,
        coordinates,
        left,
        right,
        weight,
        bias,
        pedal_mode="conditional",
        trust_fraction=_TRUST_FRACTION,
    )
    decoded = delta @ provider.parent_provider.output_decoder
    gradient = h4_gradient[0].detach().to(device="cpu", dtype=torch.float64)
    surrogate = (decoded * gradient).sum()
    values = torch.autograd.grad(
        surrogate,
        (left, right, weight, bias),
        retain_graph=False,
        create_graph=False,
    )
    return _JointState(
        direction_left=values[0],
        direction_right=values[1],
        pedal_weight=values[2],
        pedal_bias=values[3],
    )


def _mean_gradient(values: Sequence[_JointState]) -> _JointState:
    if not values:
        raise ValueError("full-batch gradient cannot be empty")
    count = float(len(values))
    totals = _zero_state(values[0])
    for value in values:
        totals = _JointState(
            direction_left=totals.direction_left + value.direction_left,
            direction_right=totals.direction_right + value.direction_right,
            pedal_weight=totals.pedal_weight + value.pedal_weight,
            pedal_bias=totals.pedal_bias + value.pedal_bias,
        )
    return _JointState(
        direction_left=totals.direction_left / count,
        direction_right=totals.direction_right / count,
        pedal_weight=totals.pedal_weight / count,
        pedal_bias=totals.pedal_bias / count,
    )


@dataclass(frozen=True, slots=True)
class _OptimizationResult:
    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider
    checkpoint_scores: tuple[float, ...]
    selected_checkpoint: int
    receipt: Mapping[str, object]


def _optimize_joint_provider(
    context: object,
    records: Sequence[_v14._FitRecord],
    capability: _TeacherRowCapability,
    start_provider: AutonomousCompleteH4FisherXYPedalProvider,
    *,
    held_family_id: str | None,
    coordinate_objective: str,
) -> _OptimizationResult:
    ordered = tuple(
        sorted(
            records,
            key=lambda value: (
                value.sequence.family_id,
                value.sequence.example_id,
            ),
        )
    )
    if not ordered:
        raise ValueError("V19 optimizer requires training records")
    family_counts: dict[str, int] = {}
    for record in ordered:
        family_counts[record.sequence.family_id] = (
            family_counts.get(record.sequence.family_id, 0) + 1
        )
    if set(family_counts.values()) != {2}:
        raise RuntimeError(
            "V19 uniform prompt-gradient averaging requires two prompts per family"
        )
    if tuple(start_provider.fit_sequence_sha256s) != tuple(
        sorted(record.sequence.artifact_sha256 for record in ordered)
    ):
        raise RuntimeError("V19 optimizer/start ownership differs")
    if tuple(start_provider.fit_family_ids) != tuple(
        sorted({record.sequence.family_id for record in ordered})
    ):
        raise RuntimeError("V19 optimizer/start families differ")
    state = _initial_joint_state(start_provider)
    zero = _zero_state(state)
    moments = _AdamMoments(first=zero, second=zero, step=0)
    states: list[_JointState] = [state]
    scores: list[float] = []
    family_scores: list[dict[str, float]] = []
    checkpoint_provider_artifacts: list[str] = []
    access_before = capability.access_count

    for checkpoint in range(_OPTIMIZER_STEPS):
        provider = _provisional_provider(
            start_provider,
            state,
            held_family_id=held_family_id,
            coordinate_objective=coordinate_objective,
            checkpoint=checkpoint,
        )
        prompt_scores: dict[str, float] = {}
        prompt_gradients: list[_JointState] = []
        for record in ordered:
            model_inputs, supervised_indices, supervised_targets = _v14._retokenize(
                context.tokenize, record.example
            )
            if (
                gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
                != record.model_inputs_sha256
                or _v14._tensor_sha256(supervised_indices)
                != record.supervised_indices_sha256
                or _v14._tensor_sha256(supervised_targets)
                != record.supervised_targets_sha256
            ):
                raise RuntimeError("V19 optimizer retokenization drifted")
            teacher = capability.get(
                record.sequence.example_id,
                family_id=record.sequence.family_id,
            )
            objective, captured = _teacher_kl_objective(
                teacher, supervised_indices
            )
            execution, h4_gradient = context.bridge.execute_h4_vjp(
                context.adapter,
                model_inputs,
                objective=objective,
                h4_head=provider,
            )
            if (
                len(captured) != 1
                or execution.h4_head_sha256 != provider.artifact_sha256
            ):
                raise RuntimeError("V19 optimizer execution authority differs")
            prompt_scores[record.sequence.example_id] = captured[0]
            prompt_gradients.append(
                _local_ste_parameter_gradients(
                    provider,
                    state,
                    record.sequence,
                    h4_gradient,
                )
            )
            del model_inputs, execution, h4_gradient, teacher
        score, per_family = _family_equal_mean(prompt_scores, ordered)
        scores.append(score)
        family_scores.append(per_family)
        checkpoint_provider_artifacts.append(provider.artifact_sha256)
        state, moments = _adam_step(
            state,
            _mean_gradient(prompt_gradients),
            moments,
        )
        states.append(state)

    # Checkpoint four is scored by actual finite execution without a fifth VJP.
    provider = _provisional_provider(
        start_provider,
        state,
        held_family_id=held_family_id,
        coordinate_objective=coordinate_objective,
        checkpoint=_OPTIMIZER_STEPS,
    )
    prompt_scores = {}
    for record in ordered:
        model_inputs, supervised_indices, supervised_targets = _v14._retokenize(
            context.tokenize, record.example
        )
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            != record.model_inputs_sha256
            or _v14._tensor_sha256(supervised_indices)
            != record.supervised_indices_sha256
            or _v14._tensor_sha256(supervised_targets)
            != record.supervised_targets_sha256
        ):
            raise RuntimeError("V19 final checkpoint retokenization drifted")
        teacher = capability.get(
            record.sequence.example_id,
            family_id=record.sequence.family_id,
        )
        execution = context.bridge.execute(
            context.adapter,
            model_inputs,
            h4_head=provider,
        )
        selected = _v14._select_sequence_rows(
            execution.logits, supervised_indices
        )
        prompt_scores[record.sequence.example_id] = float(
            exact_float64_teacher_kl(teacher, selected)
        )
        del model_inputs, execution, teacher
    score, per_family = _family_equal_mean(prompt_scores, ordered)
    scores.append(score)
    family_scores.append(per_family)
    checkpoint_provider_artifacts.append(provider.artifact_sha256)
    selected_checkpoint = choose_earliest_checkpoint(scores)
    selected_state = states[selected_checkpoint]
    evidence_payload = {
        "held_family_id": held_family_id,
        "coordinate_objective": coordinate_objective,
        "fit_protocol_sha256": _FIT_PROTOCOL_SHA256,
        "start_provider_artifact_sha256": start_provider.artifact_sha256,
        "training_family_ids": tuple(start_provider.fit_family_ids),
        "training_sequence_sha256s": tuple(start_provider.fit_sequence_sha256s),
        "training_record_receipt_sha256s": tuple(
            record.receipt_sha256 for record in ordered
        ),
        "checkpoint_scores": tuple(scores),
        "checkpoint_family_scores": tuple(family_scores),
        "checkpoint_provider_artifact_sha256s": tuple(
            checkpoint_provider_artifacts
        ),
        "checkpoint_state_receipts": tuple(value.receipt() for value in states),
        "selected_checkpoint": selected_checkpoint,
        "selected_state": selected_state.receipt(),
        "teacher_capability_artifact_sha256": capability.artifact_sha256,
        "teacher_capability_access_count": capability.access_count - access_before,
        "forward_count": len(ordered) * (_OPTIMIZER_STEPS + 1),
        "backward_count": len(ordered) * _OPTIMIZER_STEPS,
        "cast_once_derivative": "straight_through_estimator",
    }
    evidence = _v14._sha256(evidence_payload, domain=_TRAJECTORY_DOMAIN)
    selected_provider = refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start_provider,
        direction_left=selected_state.direction_left,
        direction_right=selected_state.direction_right,
        pedal_weight=selected_state.pedal_weight,
        pedal_bias=selected_state.pedal_bias,
        fit_protocol_sha256=_FIT_PROTOCOL_SHA256,
        fit_evidence_sha256=evidence,
        pedal_mode="conditional",
    )
    selected_provider.validate_integrity()
    initial_state = states[0]
    absolute_improvement = scores[0] - scores[selected_checkpoint]
    numerical_floor = max(
        float(_OUTER_THRESHOLDS["objective_absolute_improvement_floor"]),
        float(_OUTER_THRESHOLDS["objective_roundoff_multiplier"])
        * torch.finfo(torch.float64).eps
        * abs(scores[0]),
    )
    relative_improvement = (
        absolute_improvement / scores[0] if scores[0] > 0.0 else 0.0
    )
    fit_runtime = _held_runtime_diagnostics(
        selected_provider,
        tuple(record.sequence for record in ordered),
    )
    factor_left_changed = not torch.equal(
        selected_state.direction_left, initial_state.direction_left
    )
    factor_right_changed = not torch.equal(
        selected_state.direction_right, initial_state.direction_right
    )
    pedal_weight_changed = not torch.equal(
        selected_state.pedal_weight, initial_state.pedal_weight
    )
    pedal_bias_changed = not torch.equal(
        selected_state.pedal_bias, initial_state.pedal_bias
    )
    initial_product = initial_state.direction_left @ initial_state.direction_right
    selected_product = selected_state.direction_left @ selected_state.direction_right
    direction_product_changed = not torch.equal(selected_product, initial_product)
    beta_vector_changed = not torch.equal(
        torch.cat((selected_state.pedal_weight, selected_state.pedal_bias)),
        torch.cat((initial_state.pedal_weight, initial_state.pedal_bias)),
    )
    numerical_rank = int(torch.linalg.matrix_rank(selected_product))
    fit_qualification = {
        "selected_checkpoint_after_zero": selected_checkpoint > 0,
        "objective_absolute_improvement": absolute_improvement,
        "objective_numerical_improvement_floor": numerical_floor,
        "objective_improved_beyond_numerical_floor": (
            absolute_improvement > numerical_floor
        ),
        "objective_macro_relative_improvement": relative_improvement,
        "objective_macro_relative_improvement_at_least_threshold": (
            relative_improvement
            >= float(_OUTER_THRESHOLDS["objective_macro_relative_improvement_min"])
        ),
        "direction_left_changed": factor_left_changed,
        "direction_right_changed": factor_right_changed,
        "pedal_weight_changed": pedal_weight_changed,
        "pedal_bias_changed": pedal_bias_changed,
        "direction_product_changed": direction_product_changed,
        "beta_vector_changed": beta_vector_changed,
        "selected_state_finite": True,
        "selected_direction_nominal_rank": int(selected_state.direction_left.shape[1]),
        "selected_direction_numerical_rank": numerical_rank,
        "selected_direction_rank_is_16": numerical_rank == _CONDITIONAL_RANK,
        "fit_effective_pedal_nonconstant_on_direction_energy_supported_rows": (
            fit_runtime["pedal_nonconstant"] is True
        ),
    }
    fit_qualification["passed"] = all(
        value is True
        for key, value in fit_qualification.items()
        if key
        in {
            "selected_checkpoint_after_zero",
            "objective_improved_beyond_numerical_floor",
            "direction_product_changed",
            "beta_vector_changed",
            "selected_state_finite",
            "selected_direction_rank_is_16",
            "fit_effective_pedal_nonconstant_on_direction_energy_supported_rows",
        }
    )
    receipt = {
        **evidence_payload,
        "fit_evidence_sha256": evidence,
        "selected_provider_artifact_sha256": selected_provider.artifact_sha256,
        "strictly_improved_checkpoint_zero": scores[selected_checkpoint] < scores[0],
        "fit_runtime_diagnostic": fit_runtime,
        "fit_qualification": fit_qualification,
        "capability_receipt": capability.receipt(),
    }
    return _OptimizationResult(
        provider=selected_provider,
        checkpoint_scores=tuple(scores),
        selected_checkpoint=selected_checkpoint,
        receipt=receipt,
    )


def _fit_parent(
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
    *,
    bridge_binding_sha256: str,
) -> AutonomousCompleteH4ResidualProvider:
    return _v18._fit_parent(
        sequences,
        bridge_binding_sha256=bridge_binding_sha256,
    )


def _fit_v18_start(
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
    *,
    parent: AutonomousCompleteH4ResidualProvider,
    coordinate_objective: str,
) -> AutonomousCompleteH4FisherXYPedalProvider:
    return _v18._fit_child(
        sequences,
        parent=parent,
        coordinate_objective=coordinate_objective,
        pedal_mode="conditional",
    )


def _serving_resources(
    provider: AutonomousCompleteH4ResidualProvider
    | AutonomousCompleteH4FisherXYPedalProvider
    | AutonomousCompleteH4FisherFiniteJointPedalProvider,
) -> dict[str, object]:
    if isinstance(provider, AutonomousCompleteH4ResidualProvider):
        return _v18._parent_resources(provider)
    metadata = provider.metadata()
    result = {
        "scope": "provider_only_including_K256_parent",
        "prepared_float_scalar_count": metadata["prepared_float_scalar_count"],
        "runtime_parameter_bytes_float64": metadata[
            "runtime_parameter_bytes_float64"
        ],
        "logical_macs_per_token_upper_bound": metadata[
            "logical_macs_per_token_upper_bound"
        ],
        "incremental_child_prepared_float_scalar_count": metadata[
            "incremental_prepared_float_scalar_count"
        ],
        "incremental_child_runtime_parameter_bytes_float64": metadata[
            "incremental_runtime_parameter_bytes_float64"
        ],
        "incremental_child_logical_macs_per_token_upper_bound": metadata[
            "incremental_logical_macs_per_token_upper_bound"
        ],
        "retained_gemma_parameters_excluded": True,
        "base_bridge_and_full_suffix_macs_excluded": True,
        "sigmoid_norm_clamp_and_scalar_ops_excluded_from_matrix_macs": True,
        "end_to_end_model_parameter_or_flop_claim": False,
    }
    if (
        result["prepared_float_scalar_count"] != 377_608
        or result["runtime_parameter_bytes_float64"] != 377_608 * 8
        or result["logical_macs_per_token_upper_bound"] != 541_187
        or result["incremental_child_prepared_float_scalar_count"] != 16_904
        or result["incremental_child_runtime_parameter_bytes_float64"]
        != 16_904 * 8
        or result["incremental_child_logical_macs_per_token_upper_bound"]
        != 16_899
    ):
        raise RuntimeError("V19 child resource geometry differs from V18")
    return result


def _validate_joint_provider(
    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    start_provider: AutonomousCompleteH4FisherXYPedalProvider,
    pedal_mode: str,
    expected_family_count: int,
) -> None:
    if not isinstance(provider, AutonomousCompleteH4FisherFiniteJointPedalProvider):
        raise TypeError("V19 provider type differs")
    provider.validate_integrity()
    start_provider.validate_integrity()
    if (
        provider.start_provider_artifact_sha256
        != start_provider.artifact_sha256
        or provider.parent_provider.artifact_sha256
        != start_provider.parent_provider.artifact_sha256
        or provider.fit_protocol_sha256 != _FIT_PROTOCOL_SHA256
        or provider.coordinate_objective != start_provider.coordinate_objective
        or provider.pedal_mode != pedal_mode
        or provider.trust_fraction != _TRUST_FRACTION
        or provider.fit_family_ids != start_provider.fit_family_ids
        or provider.fit_sequence_sha256s != start_provider.fit_sequence_sha256s
        or len(provider.fit_family_ids) != expected_family_count
    ):
        raise RuntimeError("V19 provider protocol/ownership differs")
    _serving_resources(provider)


def _fold_ownership_receipt(
    provider: AutonomousCompleteH4ResidualProvider
    | AutonomousCompleteH4FisherXYPedalProvider
    | AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    held_family_id: str,
    held_sequences: Sequence[AutonomousCompleteH4TrainingSequence],
) -> dict[str, object]:
    held = _v14._identifier(held_family_id, label="held family")
    selected = tuple(sorted(held_sequences, key=lambda value: value.artifact_sha256))
    if (
        len(selected) != _EXPECTED_HELD_PROMPTS_PER_FOLD
        or {value.family_id for value in selected} != {held}
        or held in set(provider.fit_family_ids)
        or any(
            value.artifact_sha256 in set(provider.fit_sequence_sha256s)
            for value in selected
        )
    ):
        raise RuntimeError("V19 fold provider leaked held ownership")
    payload: dict[str, object] = {
        "held_family_id": held,
        "held_sequence_sha256s": tuple(
            value.artifact_sha256 for value in selected
        ),
        "provider_artifact_sha256": provider.artifact_sha256,
        "fit_family_ids": tuple(provider.fit_family_ids),
        "fit_sequence_sha256s": tuple(provider.fit_sequence_sha256s),
        "held_family_absent_from_fit_family_ids": True,
        "held_sequences_disjoint_from_fit_sequences": True,
    }
    if not isinstance(provider, AutonomousCompleteH4ResidualProvider):
        payload.update(
            {
                "parent_provider_artifact_sha256": (
                    provider.parent_provider.artifact_sha256
                ),
                "coordinate_objective": provider.coordinate_objective,
                "pedal_mode": provider.pedal_mode,
            }
        )
    payload["receipt_sha256"] = _v14._sha256(
        payload, domain=_OWNERSHIP_DOMAIN
    )
    return payload


def _held_runtime_diagnostics(
    provider: AutonomousCompleteH4FisherXYPedalProvider
    | AutonomousCompleteH4FisherFiniteJointPedalProvider,
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
) -> dict[str, object]:
    pedal_rows: list[Tensor] = []
    coordinate_rows: list[Tensor] = []
    bounded_ratios: list[Tensor] = []
    emitted_ratios: list[Tensor] = []
    per_sequence: list[dict[str, object]] = []
    for sequence in sorted(sequences, key=lambda value: value.artifact_sha256):
        parent = _training_parent_modal(provider.parent_provider, sequence)
        coordinates = provider.bounded_coordinates(parent)
        direction = provider.unbounded_direction(parent, coordinates)
        bounded = fisher_xy_pointwise_bounded_direction(
            parent,
            direction,
            trust_fraction=_TRUST_FRACTION,
        )
        pedal = provider.pedal_values(coordinates)
        emitted = pedal.unsqueeze(1) * bounded
        support = sequence.support_mask
        direction_energy_support = fisher_xy_pedal_fit_support_mask(
            parent,
            bounded,
        )
        effective_support = support & direction_energy_support
        if not bool(effective_support.any()):
            raise RuntimeError(
                "V19 runtime diagnostic has no direction-energy-supported row"
            )
        parent_norm = torch.linalg.vector_norm(parent[support], dim=1)
        bounded_norm = torch.linalg.vector_norm(bounded[support], dim=1)
        emitted_norm = torch.linalg.vector_norm(emitted[support], dim=1)
        safe = torch.where(
            parent_norm > 0.0,
            parent_norm,
            torch.ones_like(parent_norm),
        )
        bounded_ratio = torch.where(
            parent_norm > 0.0,
            bounded_norm / safe,
            torch.zeros_like(parent_norm),
        )
        emitted_ratio = torch.where(
            parent_norm > 0.0,
            emitted_norm / safe,
            torch.zeros_like(parent_norm),
        )
        if (
            bool((bounded[~support] != 0.0).any())
            or bool((emitted[~support] != 0.0).any())
            or float(bounded_ratio.max()) > _TRUST_FRACTION + 1.0e-14
            or float(emitted_ratio.max()) > _TRUST_FRACTION + 1.0e-14
        ):
            raise RuntimeError("V19 held runtime escaped support/trust")
        pedal_rows.append(pedal[effective_support].detach().to(device="cpu"))
        coordinate_rows.append(
            coordinates[effective_support].detach().to(device="cpu")
        )
        bounded_ratios.append(bounded_ratio.detach().to(device="cpu"))
        emitted_ratios.append(emitted_ratio.detach().to(device="cpu"))
        per_sequence.append(
            {
                "sequence_artifact_sha256": sequence.artifact_sha256,
                "support_row_count": int(support.sum()),
                "direction_energy_supported_row_count": int(
                    effective_support.sum()
                ),
                "parent_modal_sha256": _v14._tensor_sha256(parent),
                "bounded_coordinates_sha256": _v14._tensor_sha256(coordinates),
                "bounded_direction_sha256": _v14._tensor_sha256(bounded),
                "pedal_sha256": _v14._tensor_sha256(pedal),
                "emitted_delta_sha256": _v14._tensor_sha256(emitted),
                "max_bounded_direction_to_parent_norm_ratio": float(
                    bounded_ratio.max()
                ),
                "max_emitted_delta_to_parent_norm_ratio": float(
                    emitted_ratio.max()
                ),
            }
        )
    pedals = torch.cat(pedal_rows)
    coordinates = torch.cat(coordinate_rows)
    bounded_ratio = torch.cat(bounded_ratios)
    emitted_ratio = torch.cat(emitted_ratios)
    pedal_mean = float(pedals.mean())
    pedal_std = float(torch.sqrt((pedals - pedal_mean).square().mean()))
    payload = {
        "provider_artifact_sha256": provider.artifact_sha256,
        "held_sequence_count": len(per_sequence),
        "held_support_row_count": int(pedals.numel()),
        "pedal_statistics_scope": "direction_energy_supported_rows_only",
        "pedal_min": float(pedals.min()),
        "pedal_mean": pedal_mean,
        "pedal_std": pedal_std,
        "pedal_max": float(pedals.max()),
        "pedal_nonconstant": bool(
            pedal_std
            > float(_OUTER_THRESHOLDS["pedal_variation_min"])
            and float(pedals.max() - pedals.min())
            > float(_OUTER_THRESHOLDS["pedal_variation_min"])
        ),
        "coordinate_absolute_max": float(coordinates.abs().max()),
        "max_bounded_direction_to_parent_norm_ratio": float(
            bounded_ratio.max()
        ),
        "max_emitted_delta_to_parent_norm_ratio": float(emitted_ratio.max()),
        "pointwise_trust_passed": bool(
            float(bounded_ratio.max()) <= _TRUST_FRACTION + 1.0e-14
            and float(emitted_ratio.max()) <= _TRUST_FRACTION + 1.0e-14
        ),
        "per_sequence": tuple(per_sequence),
        "native_h4_logits_targets_or_gradients_required_at_runtime": False,
    }
    return {
        **payload,
        "receipt_sha256": _v14._sha256(payload, domain=_RUNTIME_DOMAIN),
    }


def _fidelity_macro(row: Mapping[str, object], key: str) -> float:
    fidelity = row.get("fidelity")
    if not isinstance(fidelity, Mapping):
        raise TypeError("V19 arm omitted fidelity")
    ordinary = fidelity.get("ordinary")
    if not isinstance(ordinary, Mapping):
        raise TypeError("V19 arm omitted ordinary fidelity")
    family_summary = ordinary.get("family_summary")
    if not isinstance(family_summary, Mapping):
        raise TypeError("V19 fidelity omitted family summary")
    macro = family_summary.get("macro")
    if not isinstance(macro, Mapping):
        raise TypeError("V19 fidelity omitted family macro")
    value = macro.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"V19 fidelity macro {key} differs")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V19 fidelity macro {key} is nonfinite")
    return result


def _fidelity_aggregate(row: Mapping[str, object], key: str) -> float:
    fidelity = row.get("fidelity")
    ordinary = fidelity.get("ordinary") if isinstance(fidelity, Mapping) else None
    aggregate = ordinary.get("aggregate") if isinstance(ordinary, Mapping) else None
    value = aggregate.get(key) if isinstance(aggregate, Mapping) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"V19 fidelity aggregate {key} differs")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V19 fidelity aggregate {key} is nonfinite")
    return result


def _relative_improvement(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        raise ValueError("relative-improvement reference must be positive")
    return (reference - candidate) / reference


def _family_absolute_delta_nll_wins(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> int:
    def rows(arm: Mapping[str, object]) -> dict[str, float]:
        fidelity = arm["fidelity"]
        ordinary = fidelity["ordinary"]  # type: ignore[index]
        family_summary = ordinary["family_summary"]  # type: ignore[index]
        values = family_summary["families"]  # type: ignore[index]
        if not isinstance(values, Sequence):
            raise TypeError("V19 family rows differ")
        return {
            str(value["family_id"]): float(value["absolute_delta_nll_per_token"])
            for value in values
            if isinstance(value, Mapping)
        }

    reference_rows = rows(reference)
    candidate_rows = rows(candidate)
    if set(reference_rows) != set(candidate_rows) or len(reference_rows) != 8:
        raise ValueError("V19 family comparison coverage differs")
    return sum(
        candidate_rows[family] < reference_rows[family]
        for family in reference_rows
    )


def _family_absolute_delta_nll_rows(
    arm: Mapping[str, object],
) -> dict[str, float]:
    fidelity = arm.get("fidelity")
    ordinary = fidelity.get("ordinary") if isinstance(fidelity, Mapping) else None
    summary = ordinary.get("family_summary") if isinstance(ordinary, Mapping) else None
    values = summary.get("families") if isinstance(summary, Mapping) else None
    if not isinstance(values, Sequence):
        raise TypeError("V19 family fidelity rows differ")
    rows: dict[str, float] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("V19 family fidelity row must be a mapping")
        family = _v14._identifier(value.get("family_id"), label="fidelity family")
        metric = value.get("absolute_delta_nll_per_token")
        if not isinstance(metric, (int, float)) or isinstance(metric, bool):
            raise TypeError("V19 family absolute delta NLL differs")
        rows[family] = float(metric)
    if len(rows) != _EXPECTED_FAMILIES or any(
        not math.isfinite(value) or value < 0.0 for value in rows.values()
    ):
        raise ValueError("V19 family absolute delta NLL coverage differs")
    return rows


def _worst_family_relative_regression(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> float:
    baseline = _family_absolute_delta_nll_rows(reference)
    selected = _family_absolute_delta_nll_rows(candidate)
    if set(baseline) != set(selected):
        raise ValueError("V19 family regression coverage differs")
    regressions: list[float] = []
    for family, reference_value in baseline.items():
        candidate_value = selected[family]
        if reference_value == 0.0:
            regressions.append(0.0 if candidate_value == 0.0 else math.inf)
        else:
            regressions.append((candidate_value - reference_value) / reference_value)
    return max(regressions)


def _ledger_no_regression(
    parent: Mapping[str, object],
    candidate: Mapping[str, object],
    ledger: str,
) -> bool:
    def aggregate(arm: Mapping[str, object]) -> Mapping[str, object]:
        fidelity = arm.get("fidelity")
        row = fidelity.get(ledger) if isinstance(fidelity, Mapping) else None
        result = row.get("aggregate") if isinstance(row, Mapping) else None
        if not isinstance(result, Mapping):
            raise TypeError(f"V19 {ledger} aggregate differs")
        return result

    baseline = aggregate(parent)
    selected = aggregate(candidate)
    return (
        abs(float(selected["delta_nll_per_token"]))
        <= abs(float(baseline["delta_nll_per_token"]))
        and float(selected["source_to_candidate_kl_per_token"])
        <= float(baseline["source_to_candidate_kl_per_token"])
        and float(selected["top1_agreement_to_source"])
        >= float(baseline["top1_agreement_to_source"])
    )


def _required_absolute_passed(row: Mapping[str, object]) -> bool:
    fidelity = row.get("fidelity")
    if not isinstance(fidelity, Mapping):
        return False
    return all(
        isinstance(fidelity.get(ledger), Mapping)
        and fidelity[ledger].get("gates", {}).get("passed") is True  # type: ignore[union-attr]
        for ledger in _REQUIRED_LEDGERS
    )


def _evaluate_outer_gates(
    arms: Mapping[str, Mapping[str, object]],
    *,
    coordinate_geometry: Mapping[str, object],
) -> dict[str, object]:
    if set(arms) != set(_ARM_IDS):
        raise ValueError("V19 outer gate arms differ")
    if coordinate_geometry.get("passed") is not True:
        raise ValueError("V19 requires pinned passed V18 coordinate geometry")
    parent = arms[PARENT_ID]
    start = arms[FISHER_START_ID]
    unit = arms[FISHER_UNIT_ID]
    intercept = arms[FISHER_INTERCEPT_ID]
    fisher = arms[FISHER_CONDITIONAL_ID]
    pca = arms[PCA_CONDITIONAL_ID]
    keys = (
        "source_to_candidate_kl_per_token",
        "absolute_delta_nll_per_token",
    )
    values = {
        arm: {key: _fidelity_macro(arms[arm], key) for key in keys}
        for arm in (
            PARENT_ID,
            FISHER_START_ID,
            FISHER_UNIT_ID,
            FISHER_INTERCEPT_ID,
            FISHER_CONDITIONAL_ID,
            PCA_CONDITIONAL_ID,
        )
    }
    training_receipts = fisher.get("optimization_receipts")
    pca_receipts = pca.get("optimization_receipts")
    if not isinstance(training_receipts, Mapping) or not isinstance(
        pca_receipts, Mapping
    ):
        raise TypeError("V19 conditional arms omitted optimization receipts")
    def optimization_passed(receipts: Mapping[str, object]) -> bool:
        if len(receipts) != _EXPECTED_OUTER_FOLDS:
            return False
        for held, receipt in receipts.items():
            if not isinstance(held, str) or not isinstance(receipt, Mapping):
                return False
            qualification = receipt.get("fit_qualification")
            capability = receipt.get("capability_receipt")
            if (
                not isinstance(qualification, Mapping)
                or qualification.get("passed") is not True
                or not isinstance(capability, Mapping)
                or capability.get("held_family_id") != held
                or capability.get("held_family_capability_excluded") is not True
            ):
                return False
        return True

    training_pass = optimization_passed(training_receipts)
    pca_training_pass = optimization_passed(pca_receipts)

    def training_macro(receipts: Mapping[str, object]) -> dict[str, float]:
        starts: list[float] = []
        selected_values: list[float] = []
        for receipt in receipts.values():
            if not isinstance(receipt, Mapping):
                raise TypeError("V19 optimization receipt must be a mapping")
            scores = receipt.get("checkpoint_scores")
            checkpoint = receipt.get("selected_checkpoint")
            if (
                not isinstance(scores, Sequence)
                or len(scores) != _OPTIMIZER_STEPS + 1
                or type(checkpoint) is not int
                or checkpoint < 0
                or checkpoint >= len(scores)
            ):
                raise ValueError("V19 checkpoint score receipt differs")
            starts.append(float(scores[0]))
            selected_values.append(float(scores[checkpoint]))
        start_mean = math.fsum(starts) / len(starts)
        selected_mean = math.fsum(selected_values) / len(selected_values)
        return {
            "checkpoint_zero_mean": start_mean,
            "selected_checkpoint_mean": selected_mean,
            "relative_improvement": _relative_improvement(
                start_mean, selected_mean
            ),
        }

    fisher_training_macro = training_macro(training_receipts)
    pca_training_macro = training_macro(pca_receipts)
    start_improvements = {
        key: _relative_improvement(
            values[FISHER_START_ID][key], values[FISHER_CONDITIONAL_ID][key]
        )
        for key in keys
    }
    intercept_improvements = {
        key: _relative_improvement(
            values[FISHER_INTERCEPT_ID][key],
            values[FISHER_CONDITIONAL_ID][key],
        )
        for key in keys
    }
    top1 = {
        arm: _fidelity_aggregate(arms[arm], "top1_agreement_to_source")
        for arm in (
            PARENT_ID,
            FISHER_START_ID,
            FISHER_UNIT_ID,
            FISHER_INTERCEPT_ID,
            FISHER_CONDITIONAL_ID,
            PCA_CONDITIONAL_ID,
        )
    }
    family_wins = {
        "parent": _family_absolute_delta_nll_wins(parent, fisher),
        "start": _family_absolute_delta_nll_wins(start, fisher),
        "intercept": _family_absolute_delta_nll_wins(intercept, fisher),
        "unit": _family_absolute_delta_nll_wins(unit, fisher),
    }
    runtime = {
        arm: arms[arm].get("held_runtime_diagnostics")
        for arm in _JOINT_IDS
    }
    trust_pass = all(
        isinstance(receipts, Mapping)
        and len(receipts) == _EXPECTED_OUTER_FOLDS
        and all(
            isinstance(value, Mapping)
            and value.get("pointwise_trust_passed") is True
            for value in receipts.values()
        )
        for receipts in runtime.values()
    )
    def variation_passed(arm: str) -> bool:
        receipts = runtime[arm]
        return bool(
            isinstance(receipts, Mapping)
            and len(receipts) == _EXPECTED_OUTER_FOLDS
            and all(
                isinstance(value, Mapping)
                and value.get("pedal_nonconstant") is True
                for value in receipts.values()
            )
        )

    fisher_variation_pass = variation_passed(FISHER_CONDITIONAL_ID)
    pca_variation_pass = variation_passed(PCA_CONDITIONAL_ID)
    checks = {
        "inherited_v18_coordinate_geometry_passed": True,
        "fisher_training_every_fold_strictly_improved_checkpoint_zero": training_pass,
        "fisher_eight_fold_macro_training_objective_improvement_at_least_1pct": (
            fisher_training_macro["relative_improvement"]
            >= float(_OUTER_THRESHOLDS["objective_macro_relative_improvement_min"])
        ),
        "fisher_absolute_delta_NLL_relative_improvement_vs_parent_at_least_5pct": (
            _relative_improvement(
                values[PARENT_ID]["absolute_delta_nll_per_token"],
                values[FISHER_CONDITIONAL_ID]["absolute_delta_nll_per_token"],
            )
            >= float(
                _OUTER_THRESHOLDS[
                    "parent_absolute_delta_nll_relative_improvement_min"
                ]
            )
        ),
        "fisher_KL_relative_improvement_vs_parent_at_least_5pct": (
            _relative_improvement(
                values[PARENT_ID]["source_to_candidate_kl_per_token"],
                values[FISHER_CONDITIONAL_ID]["source_to_candidate_kl_per_token"],
            )
            >= float(_OUTER_THRESHOLDS["parent_kl_relative_improvement_min"])
        ),
        "fisher_aggregate_top1_gain_vs_parent_at_least_0p02": (
            top1[FISHER_CONDITIONAL_ID] - top1[PARENT_ID]
            >= float(_OUTER_THRESHOLDS["parent_aggregate_top1_gain_min"])
        ),
        "fisher_family_absolute_delta_NLL_wins_vs_parent_at_least_6": (
            family_wins["parent"]
            >= int(_OUTER_THRESHOLDS["parent_family_absolute_delta_nll_win_count_min"])
        ),
        "fisher_worst_family_relative_regression_vs_parent_at_most_2pct": (
            _worst_family_relative_regression(parent, fisher)
            <= float(_OUTER_THRESHOLDS["parent_worst_family_relative_regression_max"])
        ),
        "fisher_support_and_graph_core_no_aggregate_regression_vs_parent": all(
            _ledger_no_regression(parent, fisher, ledger)
            for ledger in ("complete_h4_support", "graph_core")
        ),
        "fisher_absolute_delta_NLL_relative_improvement_vs_v18_start_at_least_1pct": (
            start_improvements["absolute_delta_nll_per_token"]
            >= float(_OUTER_THRESHOLDS["start_absolute_delta_nll_relative_improvement_min"])
        ),
        "fisher_KL_not_higher_than_v18_start": (
            values[FISHER_CONDITIONAL_ID]["source_to_candidate_kl_per_token"]
            <= values[FISHER_START_ID]["source_to_candidate_kl_per_token"]
        ),
        "fisher_top1_not_below_v18_start": top1[FISHER_CONDITIONAL_ID]
        >= top1[FISHER_START_ID],
        "fisher_family_absolute_delta_NLL_wins_vs_start_at_least_5": (
            family_wins["start"]
            >= int(_OUTER_THRESHOLDS["start_family_absolute_delta_nll_win_count_min"])
        ),
        "fisher_absolute_delta_NLL_relative_improvement_vs_intercept_at_least_1pct": (
            intercept_improvements["absolute_delta_nll_per_token"]
            >= float(
                _OUTER_THRESHOLDS[
                    "intercept_absolute_delta_nll_relative_improvement_min"
                ]
            )
        ),
        "fisher_KL_not_higher_than_intercept": (
            values[FISHER_CONDITIONAL_ID]["source_to_candidate_kl_per_token"]
            <= values[FISHER_INTERCEPT_ID]["source_to_candidate_kl_per_token"]
        ),
        "fisher_top1_not_below_intercept": top1[FISHER_CONDITIONAL_ID]
        >= top1[FISHER_INTERCEPT_ID],
        "fisher_family_absolute_delta_NLL_wins_vs_intercept_at_least_5": (
            family_wins["intercept"]
            >= int(_OUTER_THRESHOLDS["intercept_family_absolute_delta_nll_win_count_min"])
        ),
        "fisher_absolute_delta_NLL_strictly_below_unit": (
            values[FISHER_CONDITIONAL_ID]["absolute_delta_nll_per_token"]
            < values[FISHER_UNIT_ID]["absolute_delta_nll_per_token"]
        ),
        "fisher_KL_not_higher_than_unit": (
            values[FISHER_CONDITIONAL_ID]["source_to_candidate_kl_per_token"]
            <= values[FISHER_UNIT_ID]["source_to_candidate_kl_per_token"]
        ),
        "fisher_top1_not_below_unit": top1[FISHER_CONDITIONAL_ID]
        >= top1[FISHER_UNIT_ID],
        "fisher_family_absolute_delta_NLL_wins_vs_unit_at_least_5": (
            family_wins["unit"]
            >= int(_OUTER_THRESHOLDS["unit_family_absolute_delta_nll_win_count_min"])
        ),
        "fisher_absolute_delta_NLL_strictly_below_PCA": (
            values[FISHER_CONDITIONAL_ID]["absolute_delta_nll_per_token"]
            < values[PCA_CONDITIONAL_ID]["absolute_delta_nll_per_token"]
        ),
        "fisher_KL_not_higher_than_PCA": (
            values[FISHER_CONDITIONAL_ID]["source_to_candidate_kl_per_token"]
            <= values[PCA_CONDITIONAL_ID]["source_to_candidate_kl_per_token"]
        ),
        "fisher_top1_not_below_PCA": top1[FISHER_CONDITIONAL_ID]
        >= top1[PCA_CONDITIONAL_ID],
        "fisher_required_absolute_ledgers_pass": _required_absolute_passed(fisher),
        "every_joint_fold_passes_pointwise_trust": trust_pass,
        "every_fisher_conditional_fold_has_nonconstant_pedal": fisher_variation_pass,
    }
    return {
        "thresholds": dict(_OUTER_THRESHOLDS),
        "training_checkpoint_passed": training_pass,
        "pca_training_checkpoint_diagnostic_passed": pca_training_pass,
        "pca_held_pedal_variation_diagnostic_passed": pca_variation_pass,
        "fisher_training_objective_macro": fisher_training_macro,
        "pca_training_objective_macro": pca_training_macro,
        "macro_values": values,
        "aggregate_top1": top1,
        "relative_improvement_vs_v18_start": start_improvements,
        "relative_improvement_vs_intercept": intercept_improvements,
        "family_absolute_delta_nll_win_counts": family_wins,
        "inherited_coordinate_geometry": dict(coordinate_geometry),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _work_accounting(*, full_refit_fitted: bool) -> dict[str, object]:
    if type(full_refit_fitted) is not bool:
        raise TypeError("full-refit flag must be boolean")
    breakdown = {
        "collection_native_source_forwards": _EXPECTED_PROMPTS,
        "collection_base_vjp_forwards": _EXPECTED_PROMPTS,
        "collection_base_vjp_backwards": _EXPECTED_PROMPTS,
        "outer_checkpoint_forwards": (
            _EXPECTED_OUTER_FOLDS
            * 2
            * _EXPECTED_OPTIMIZER_FORWARDS_PER_ARM_FOLD
        ),
        "outer_checkpoint_backwards": (
            _EXPECTED_OUTER_FOLDS
            * 2
            * _EXPECTED_OPTIMIZER_BACKWARDS_PER_ARM_FOLD
        ),
        "evaluation_native_source_forwards": _EXPECTED_PROMPTS,
        "evaluation_base_forwards": _EXPECTED_PROMPTS,
        "evaluation_arm_forwards": _EXPECTED_PROMPTS * len(_ARM_IDS),
        "conditional_full_refit_forwards": (
            _EXPECTED_PROMPTS * (_OPTIMIZER_STEPS + 1)
            if full_refit_fitted
            else 0
        ),
        "conditional_full_refit_backwards": (
            _EXPECTED_PROMPTS * _OPTIMIZER_STEPS if full_refit_fitted else 0
        ),
    }
    forwards = sum(
        int(value)
        for key, value in breakdown.items()
        if "forward" in key
    )
    backwards = sum(
        int(value)
        for key, value in breakdown.items()
        if "backward" in key
    )
    expected_forwards = _EXPECTED_FULL_MODEL_FORWARDS + (
        80 if full_refit_fitted else 0
    )
    expected_backwards = _EXPECTED_BACKWARD_TRAVERSALS + (
        64 if full_refit_fitted else 0
    )
    local_head_contractions = (
        _EXPECTED_OUTER_FOLDS
        * 2
        * _EXPECTED_OPTIMIZER_BACKWARDS_PER_ARM_FOLD
        + (64 if full_refit_fitted else 0)
    )
    if forwards != expected_forwards or backwards != expected_backwards:
        raise RuntimeError("V19 work accounting differs")
    return {
        "outer_provider_fit_count": _EXPECTED_OUTER_PROVIDER_FITS,
        "expected_outer_provider_fit_count": _EXPECTED_OUTER_PROVIDER_FITS,
        "outer_provider_fit_count_semantics": (
            "per_fold_parent_plus_fisher_start_plus_pca_start_plus_"
            "fisher_finite_optimization_plus_pca_finite_optimization;_"
            "checkpoint_materializations_and_unit_intercept_controls_are_not_fits"
        ),
        "conditional_full_panel_provider_fit_count": 3 if full_refit_fitted else 0,
        "full_model_forward_count": forwards,
        "expected_full_model_forward_count": expected_forwards,
        "full_suffix_backward_traversal_count": backwards,
        "expected_full_suffix_backward_traversal_count": expected_backwards,
        "local_head_autograd_contraction_count": local_head_contractions,
        "expected_local_head_autograd_contraction_count": local_head_contractions,
        "total_autograd_grad_call_count": backwards + local_head_contractions,
        "no_full_refit_total_autograd_grad_call_count": 1_808,
        "no_full_refit_nominal_forward_count": _EXPECTED_FULL_MODEL_FORWARDS,
        "no_full_refit_nominal_backward_count": _EXPECTED_BACKWARD_TRAVERSALS,
        "breakdown": breakdown,
    }


def _validate_full_refit_qualification(
    qualification: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "provider_artifact_sha256",
        "parent_provider_artifact_sha256",
        "start_provider_artifact_sha256",
        "optimization_receipt",
        "runtime_diagnostic",
        "serving_resources",
        "passed",
    }
    if set(qualification) != required:
        raise ValueError("V19 full-refit qualification fields differ")
    result = dict(qualification)
    for key in (
        "provider_artifact_sha256",
        "parent_provider_artifact_sha256",
        "start_provider_artifact_sha256",
    ):
        _sha256_identifier(result.get(key), label=f"V19 full refit {key}")
    optimization = result.get("optimization_receipt")
    runtime = result.get("runtime_diagnostic")
    resources = result.get("serving_resources")
    if (
        not isinstance(optimization, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(resources, Mapping)
        or type(result.get("passed")) is not bool
    ):
        raise TypeError("V19 full-refit qualification payload differs")
    fit = optimization.get("fit_qualification")
    capability = optimization.get("capability_receipt")
    recomputed = bool(
        isinstance(fit, Mapping)
        and fit.get("passed") is True
        and fit.get(
            "objective_macro_relative_improvement_at_least_threshold"
        )
        is True
        and optimization.get("selected_provider_artifact_sha256")
        == result["provider_artifact_sha256"]
        and optimization.get("start_provider_artifact_sha256")
        == result["start_provider_artifact_sha256"]
        and isinstance(optimization.get("training_family_ids"), Sequence)
        and len(optimization["training_family_ids"]) == _EXPECTED_FAMILIES  # type: ignore[arg-type]
        and isinstance(optimization.get("training_sequence_sha256s"), Sequence)
        and len(optimization["training_sequence_sha256s"]) == _EXPECTED_PROMPTS  # type: ignore[arg-type]
        and isinstance(capability, Mapping)
        and capability.get("held_family_id") is None
        and capability.get("authorized_example_count") == _EXPECTED_PROMPTS
        and capability.get("authorized_family_count") == _EXPECTED_FAMILIES
        and capability.get("access_count")
        == _EXPECTED_PROMPTS * (_OPTIMIZER_STEPS + 1)
        and runtime.get("pointwise_trust_passed") is True
        and runtime.get("pedal_nonconstant") is True
        and runtime.get("provider_artifact_sha256")
        == result["provider_artifact_sha256"]
        and runtime.get("held_sequence_count") == _EXPECTED_PROMPTS
        and resources.get("prepared_float_scalar_count") == 377_608
        and resources.get("logical_macs_per_token_upper_bound") == 541_187
    )
    if result["passed"] is not recomputed:
        raise ValueError("V19 full-refit qualification decision drifted")
    return result


def build_finite_joint_pedal_development_report(
    *,
    artifact_path: Path | str,
    panel: Mapping[str, object],
    bridge_binding_sha256: str,
    folds: Sequence[Mapping[str, object]],
    prerequisites: Mapping[str, object],
    fit_collection: Mapping[str, object],
    base_fidelity: Mapping[str, object],
    arms: Mapping[str, Mapping[str, object]],
    full_refit_qualification: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
    integrity: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic scalar/hash-only V19 development report."""

    destination = _validate_output(artifact_path)
    if set(arms) != set(_ARM_IDS):
        raise ValueError("V19 report arm set differs")
    selected_folds = tuple(folds)
    held = tuple(
        _v14._identifier(
            fold.get("held_family_id") if isinstance(fold, Mapping) else None,
            label="V19 held family",
        )
        for fold in selected_folds
    )
    if len(selected_folds) != _EXPECTED_OUTER_FOLDS or len(set(held)) != len(held):
        raise ValueError("V19 report outer-fold geometry differs")
    if prerequisites.get("v18_candidate_was_null") is not True:
        raise ValueError("V19 report prerequisite null-candidate receipt differs")
    geometry = prerequisites.get("v18_coordinate_geometry_qualification")
    if not isinstance(geometry, Mapping):
        raise TypeError("V19 report prerequisite geometry differs")
    outer = _evaluate_outer_gates(arms, coordinate_geometry=geometry)
    validated_full = (
        None
        if full_refit_qualification is None
        else _validate_full_refit_qualification(full_refit_qualification)
    )
    full_passed = validated_full is not None and validated_full["passed"] is True
    ready = outer["passed"] is True and full_passed
    if (full_refit_qualification is not None) != (outer["passed"] is True):
        raise ValueError("V19 conditional full refit does not match outer gates")
    if (candidate is not None) != ready:
        raise ValueError("V19 candidate does not match complete qualification")
    if candidate is not None:
        if (
            candidate.get("arm_id") != FISHER_CONDITIONAL_ID
            or candidate.get("provider_artifact_sha256")
            != validated_full.get("provider_artifact_sha256")  # type: ignore[union-attr]
        ):
            raise ValueError("V19 candidate/full refit binding differs")
    if outer["passed"] is not True:
        checks = outer.get("checks")
        failed = tuple(
            key
            for key, value in sorted(checks.items())  # type: ignore[union-attr]
            if value is not True
        )
        classification = (
            "finite_joint_pedal_outer_fidelity_insufficient"
            if any("absolute" in key or "KL" in key or "top1" in key for key in failed)
            else "finite_joint_pedal_outer_mechanism_insufficient"
        )
    elif not full_passed:
        classification = "finite_joint_pedal_full_refit_qualification_insufficient"
    else:
        classification = "finite_joint_pedal_oof_candidate_ready_for_fresh_protocol"
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": 19,
        "scientific_status": (
            "opened_calibration_a_fixed_finite_teacher_kl_joint_direction_"
            "sigmoid_pedal_outer_lofo_development"
        ),
        "artifact": {
            "path": destination.as_posix(),
            "write_once": True,
            "scalar_and_hash_only": True,
            "provider_tensor_sidecar_conditional": True,
        },
        "panel": dict(panel),
        "bridge_binding_sha256": _v14._identifier(
            bridge_binding_sha256, label="V19 bridge binding"
        ),
        "prerequisites": dict(prerequisites),
        "fixed_protocol": {
            **_FIXED_PROTOCOL,
            "fit_protocol_sha256": _FIT_PROTOCOL_SHA256,
        },
        "outer_lofo": {
            "fold_count": len(selected_folds),
            "folds": selected_folds,
            "held_family_ids": held,
            "held_family_logits_may_have_been_transiently_materialized_during_collection": True,
            "held_teacher_rows_capability_excluded_and_not_consumed_by_each_fold_fit": True,
            "held_evaluation_consumed_only_after_checkpoint_and_provider_hashes_froze": True,
        },
        "fit_collection": dict(fit_collection),
        "base_fidelity": dict(base_fidelity),
        "arms": {key: dict(value) for key, value in sorted(arms.items())},
        "outer_qualification": outer,
        "full_refit_qualification": (
            None
            if validated_full is None
            else validated_full
        ),
        "candidate": None if candidate is None else dict(candidate),
        "integrity": dict(integrity),
        "classification": classification,
        "passed": ready,
        "candidate_readiness": ready,
        "fresh_guard_authorized": False,
        "calibration_b_authorized": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "end_to_end_parameter_or_flop_claim": False,
        "success_authorizes": (
            "fresh_family_disjoint_shadow_and_NLL_protocol_only"
            if ready
            else "no_candidate_and_no_fresh_protocol"
        ),
    }
    _v14._scalar_report(report)
    return report


def _publish(
    report: dict[str, object],
    *,
    output: Path,
    provider: AutonomousCompleteH4FisherFiniteJointPedalProvider | None,
    provider_output: Path,
) -> dict[str, object]:
    if (
        _same_destination(output, _V18_OUTPUT)
        or _same_destination(provider_output, _V18_OUTPUT.with_suffix(".provider.pt"))
    ):
        raise ValueError("V19 publication must preserve write-once V18")
    candidate = report.get("candidate")
    if (provider is None) != (candidate is None):
        raise ValueError("V19 published provider must match candidate")
    destinations = (output,) if provider is None else (output, provider_output)
    reservation = _v14._reserve_outputs(destinations)
    report_stage: Path | None = None
    provider_stage: Path | None = None
    try:
        if provider is not None:
            if not isinstance(candidate, Mapping):
                raise TypeError("V19 candidate must be a mapping")
            provider.validate_integrity()
            if candidate.get("provider_artifact_sha256") != provider.artifact_sha256:
                raise ValueError("V19 candidate/provider artifact differs")
            provider_stage = _v14._stage_torch(
                autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict(
                    provider
                ),
                provider_output,
            )
            provider_file_sha256 = _v14._file_sha256(provider_stage)
            restored = load_autonomous_complete_h4_fisher_finite_joint_pedal_provider(
                provider_stage,
                expected_artifact_sha256=provider.artifact_sha256,
                expected_file_sha256=provider_file_sha256,
                expected_bridge_binding_sha256=provider.bridge_binding_sha256,
                expected_start_provider_artifact_sha256=(
                    provider.start_provider_artifact_sha256
                ),
            )
            if restored.metadata() != provider.metadata():
                raise RuntimeError("staged V19 provider roundtrip drifted")
            report["candidate"] = {
                **dict(candidate),
                "provider_tensor_artifact": {
                    "path": provider_output.as_posix(),
                    "file_sha256": provider_file_sha256,
                    "file_bytes": provider_stage.stat().st_size,
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "bridge_binding_sha256": provider.bridge_binding_sha256,
                    "start_provider_artifact_sha256": (
                        provider.start_provider_artifact_sha256
                    ),
                    "write_once": True,
                    "file_mode": "0600",
                    "contains_runtime_provider_tensors_only": True,
                    "contains_logits_targets_gradients_optimizer_state_or_examples": False,
                },
            }
        _v14._scalar_report(report)
        report["report_sha256"] = _v14._sha256(
            report, domain=_REPORT_DOMAIN
        )
        report_stage = _v14._stage_json(report, output)
        staged = (
            (report_stage,)
            if provider_stage is None
            else (report_stage, provider_stage)
        )
        reservation.publish(staged)
        if provider is not None:
            receipt = report["candidate"]["provider_tensor_artifact"]  # type: ignore[index]
            load_autonomous_complete_h4_fisher_finite_joint_pedal_provider(
                provider_output,
                expected_artifact_sha256=provider.artifact_sha256,
                expected_file_sha256=str(receipt["file_sha256"]),
                expected_bridge_binding_sha256=provider.bridge_binding_sha256,
                expected_start_provider_artifact_sha256=(
                    provider.start_provider_artifact_sha256
                ),
            )
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _v14._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        for stage in (report_stage, provider_stage):
            if stage is not None:
                stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_finite_joint_pedal_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    provider_output: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the fixed A16 V19 finite-objective outer-LOFO development screen."""

    destination = _validate_output(output)
    provider_destination = _validate_provider_output(
        destination.with_suffix(".provider.pt")
        if provider_output is None
        else provider_output
    )
    if (
        _same_destination(destination, _V18_OUTPUT)
        or _same_destination(
            provider_destination, _V18_OUTPUT.with_suffix(".provider.pt")
        )
    ):
        raise ValueError("V19 runner must preserve the write-once V18 rung")
    if _same_destination(destination, provider_destination):
        raise ValueError("V19 report and provider outputs must differ")
    if destination.exists():
        raise FileExistsError("refusing to overwrite V19 report")
    if provider_destination.exists():
        raise FileExistsError("refusing to overwrite V19 provider")

    prerequisites = _validate_prerequisites()
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        records, teacher_vault = _collect_fit_records_and_teacher_vault(context)
        bridge_binding = _v14._identifier(
            context.bridge.bridge_binding_sha256,
            label="V19 bridge binding",
        )
        families = tuple(sorted({record.sequence.family_id for record in records}))
        folds = _v14.build_outer_lofo_splits(families)
        parents: dict[str, AutonomousCompleteH4ResidualProvider] = {}
        starts: dict[str, dict[str, AutonomousCompleteH4FisherXYPedalProvider]] = {
            "reverse_vjp_fisher": {},
            "activation_pca": {},
        }
        optimized: dict[
            str, dict[str, AutonomousCompleteH4FisherFiniteJointPedalProvider]
        ] = {
            "reverse_vjp_fisher": {},
            "activation_pca": {},
        }
        controls: dict[
            str, dict[str, AutonomousCompleteH4FisherFiniteJointPedalProvider]
        ] = {"unit": {}, "intercept": {}}
        optimization_receipts: dict[str, dict[str, Mapping[str, object]]] = {
            "reverse_vjp_fisher": {},
            "activation_pca": {},
        }
        held_sequences_by_family: dict[
            str, tuple[AutonomousCompleteH4TrainingSequence, ...]
        ] = {}

        for fold in folds:
            held = _v14._identifier(fold["held_family_id"], label="held family")
            training_records = tuple(
                record for record in records if record.sequence.family_id != held
            )
            held_records = tuple(
                record for record in records if record.sequence.family_id == held
            )
            training_sequences = tuple(
                record.sequence for record in training_records
            )
            held_sequences = tuple(record.sequence for record in held_records)
            if (
                len(training_records) != _EXPECTED_TRAINING_PROMPTS_PER_FOLD
                or len(held_records) != _EXPECTED_HELD_PROMPTS_PER_FOLD
                or held in {record.sequence.family_id for record in training_records}
            ):
                raise RuntimeError("V19 outer-LOFO ownership differs")
            parent = _fit_parent(
                training_sequences,
                bridge_binding_sha256=bridge_binding,
            )
            _v18._validate_parent(
                parent, expected_fit_family_count=_EXPECTED_FAMILIES - 1
            )
            pinned_lineage = prerequisites[
                "v18_fold_provider_artifact_sha256s"
            ]
            if (
                not isinstance(pinned_lineage, Mapping)
                or not isinstance(pinned_lineage.get(PARENT_ID), Mapping)
                or pinned_lineage[PARENT_ID].get(held)  # type: ignore[union-attr]
                != parent.artifact_sha256
            ):
                raise RuntimeError("V19 parent does not reconstruct pinned V18 fold")
            parents[held] = parent
            held_sequences_by_family[held] = held_sequences
            authorized = tuple(
                record.sequence.example_id for record in training_records
            )
            for objective in ("reverse_vjp_fisher", "activation_pca"):
                start = _fit_v18_start(
                    training_sequences,
                    parent=parent,
                    coordinate_objective=objective,
                )
                _v18._validate_child(
                    start,
                    coordinate_objective=objective,
                    pedal_mode="conditional",
                    expected_parent_artifact_sha256=parent.artifact_sha256,
                    expected_fit_family_count=_EXPECTED_FAMILIES - 1,
                )
                pinned_arm_id = (
                    _v18.FISHER_PEDAL_ID
                    if objective == "reverse_vjp_fisher"
                    else _v18.PCA_PEDAL_ID
                )
                pinned_hashes = pinned_lineage.get(pinned_arm_id)  # type: ignore[union-attr]
                if (
                    not isinstance(pinned_hashes, Mapping)
                    or pinned_hashes.get(held) != start.artifact_sha256
                ):
                    raise RuntimeError(
                        "V19 initializer does not reconstruct pinned V18 fold"
                    )
                capability = teacher_vault.capability(
                    authorized,
                    held_family_id=held,
                )
                result = _optimize_joint_provider(
                    context,
                    training_records,
                    capability,
                    start,
                    held_family_id=held,
                    coordinate_objective=objective,
                )
                _validate_joint_provider(
                    result.provider,
                    start_provider=start,
                    pedal_mode="conditional",
                    expected_family_count=_EXPECTED_FAMILIES - 1,
                )
                starts[objective][held] = start
                optimized[objective][held] = result.provider
                optimization_receipts[objective][held] = result.receipt
            fisher = optimized["reverse_vjp_fisher"][held]
            for mode in ("unit", "intercept"):
                control = fisher_finite_joint_pedal_control(
                    fisher,
                    pedal_mode=mode,
                )
                _validate_joint_provider(
                    control,
                    start_provider=starts["reverse_vjp_fisher"][held],
                    pedal_mode=mode,
                    expected_family_count=_EXPECTED_FAMILIES - 1,
                )
                controls[mode][held] = control
            resources = {
                json.dumps(_serving_resources(provider), sort_keys=True)
                for provider in (
                    starts["reverse_vjp_fisher"][held],
                    starts["activation_pca"][held],
                    optimized["reverse_vjp_fisher"][held],
                    optimized["activation_pca"][held],
                    controls["unit"][held],
                    controls["intercept"][held],
                )
            }
            if len(resources) != 1:
                raise RuntimeError("V19 child arms are not parameter/MAC matched")

        # Only after every fold's provider/checkpoint hash is frozen do held
        # source rows enter outer evaluation.
        manifests = _v14._ledger_manifests(records)
        ledger_coverage = _v14._ledger_coverage(manifests)
        accumulators = {
            arm: {
                ledger: SourceAuthoritativeShadowFidelityAccumulator(
                    manifest,
                    gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
                )
                for ledger, manifest in manifests.items()
            }
            for arm in ("base", *_ARM_IDS)
        }
        causal_checks = 0
        for record in records:
            model_inputs, supervised_indices, supervised_targets = _v14._retokenize(
                context.tokenize, record.example
            )
            if (
                gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
                != record.model_inputs_sha256
                or _v14._tensor_sha256(supervised_indices)
                != record.supervised_indices_sha256
                or _v14._tensor_sha256(supervised_targets)
                != record.supervised_targets_sha256
            ):
                raise RuntimeError("V19 held evaluation retokenization drifted")
            source_logits, _native_h4, native_positions, native_valid = (
                _v14._native_boundary(context.adapter, model_inputs)
            )
            base = context.bridge.execute(context.adapter, model_inputs)
            if (
                not isinstance(base, Gemma3L3L4OnePassExecution)
                or not _v14._bitwise_equal(
                    native_positions, base.prefix.logical_positions
                )
                or not _v14._bitwise_equal(
                    native_valid, base.prefix.valid_target_mask
                )
            ):
                raise RuntimeError("V19 held base execution differs")
            _v14._add_shadow_rows(
                accumulators["base"],
                record=record,
                source_logits=source_logits,
                candidate_logits=base.logits,
                supervised_indices=supervised_indices,
                supervised_targets=supervised_targets,
            )
            held = record.sequence.family_id
            fold_providers: tuple[
                tuple[
                    str,
                    AutonomousCompleteH4ResidualProvider
                    | AutonomousCompleteH4FisherXYPedalProvider
                    | AutonomousCompleteH4FisherFiniteJointPedalProvider,
                ],
                ...,
            ] = (
                (PARENT_ID, parents[held]),
                (FISHER_START_ID, starts["reverse_vjp_fisher"][held]),
                (FISHER_UNIT_ID, controls["unit"][held]),
                (FISHER_INTERCEPT_ID, controls["intercept"][held]),
                (
                    FISHER_CONDITIONAL_ID,
                    optimized["reverse_vjp_fisher"][held],
                ),
                (PCA_CONDITIONAL_ID, optimized["activation_pca"][held]),
            )
            support = base.prefix.complete_h4_causal_support_mask()
            for arm_id, provider in fold_providers:
                if held in set(provider.fit_family_ids):
                    raise RuntimeError("held family leaked into V19 provider")
                execution = context.bridge.execute(
                    context.adapter,
                    model_inputs,
                    h4_head=provider,
                )
                if (
                    execution.h4_head_sha256 != provider.artifact_sha256
                    or execution.prefix.artifact_sha256
                    != base.prefix.artifact_sha256
                    or not _v14._bitwise_equal(
                        execution.candidate_h4[
                            ~support.to(execution.candidate_h4.device)
                        ],
                        base.candidate_h4[
                            ~support.to(base.candidate_h4.device)
                        ],
                    )
                ):
                    raise RuntimeError("V19 provider escaped causal support")
                _v14._add_shadow_rows(
                    accumulators[arm_id],
                    record=record,
                    source_logits=source_logits,
                    candidate_logits=execution.logits,
                    supervised_indices=supervised_indices,
                    supervised_targets=supervised_targets,
                )
                causal_checks += 1
                del execution
            del model_inputs, source_logits, base

        fidelity = {
            arm: {
                ledger: accumulator.finalize()
                for ledger, accumulator in ledgers.items()
            }
            for arm, ledgers in accumulators.items()
        }
        pinned_v18_payload = json.loads(_V18_OUTPUT.read_text(encoding="utf-8"))
        pinned_v18_arms = pinned_v18_payload.get("arms")
        pinned_parent_row = (
            pinned_v18_arms.get(PARENT_ID)
            if isinstance(pinned_v18_arms, Mapping)
            else None
        )
        pinned_start_row = (
            pinned_v18_arms.get(_v18.FISHER_PEDAL_ID)
            if isinstance(pinned_v18_arms, Mapping)
            else None
        )
        if (
            fidelity["base"] != pinned_v18_payload.get("base_fidelity")
            or not isinstance(pinned_parent_row, Mapping)
            or fidelity[PARENT_ID] != pinned_parent_row.get("fidelity")
            or not isinstance(pinned_start_row, Mapping)
            or fidelity[FISHER_START_ID] != pinned_start_row.get("fidelity")
        ):
            raise RuntimeError(
                "V19 base/parent/Fisher-start held fidelity does not exactly replay V18"
            )
        v18_fidelity_replay_receipt = _v14._sha256(
            {
                "base_fidelity": fidelity["base"],
                "parent_fidelity": fidelity[PARENT_ID],
                "fisher_start_fidelity": fidelity[FISHER_START_ID],
            },
            domain=_OWNERSHIP_DOMAIN,
        )
        provider_maps: dict[
            str,
            Mapping[
                str,
                AutonomousCompleteH4ResidualProvider
                | AutonomousCompleteH4FisherXYPedalProvider
                | AutonomousCompleteH4FisherFiniteJointPedalProvider,
            ],
        ] = {
            PARENT_ID: parents,
            FISHER_START_ID: starts["reverse_vjp_fisher"],
            FISHER_UNIT_ID: controls["unit"],
            FISHER_INTERCEPT_ID: controls["intercept"],
            FISHER_CONDITIONAL_ID: optimized["reverse_vjp_fisher"],
            PCA_CONDITIONAL_ID: optimized["activation_pca"],
        }
        arm_rows: dict[str, Mapping[str, object]] = {}
        for arm_id, providers in provider_maps.items():
            sample = providers[sorted(providers)[0]]
            row: dict[str, object] = {
                "arm_id": arm_id,
                "outer_fold_count": len(providers),
                "every_fold_fit_family_count": _EXPECTED_FAMILIES - 1,
                "fold_provider_artifact_sha256s": {
                    held: provider.artifact_sha256
                    for held, provider in sorted(providers.items())
                },
                "fold_ownership_receipts": {
                    held: _fold_ownership_receipt(
                        provider,
                        held_family_id=held,
                        held_sequences=held_sequences_by_family[held],
                    )
                    for held, provider in sorted(providers.items())
                },
                "serving_resources": _serving_resources(sample),
                "fidelity": fidelity[arm_id],
            }
            if not isinstance(sample, AutonomousCompleteH4ResidualProvider):
                row.update(
                    {
                        "coordinate_objective": sample.coordinate_objective,
                        "pedal_mode": sample.pedal_mode,
                        "conditional_rank": _CONDITIONAL_RANK,
                        "held_runtime_diagnostics": {
                            held: _held_runtime_diagnostics(
                                provider,
                                held_sequences_by_family[held],
                            )
                            for held, provider in sorted(providers.items())
                        },
                    }
                )
            if arm_id == FISHER_CONDITIONAL_ID:
                row["optimization_receipts"] = optimization_receipts[
                    "reverse_vjp_fisher"
                ]
            elif arm_id == PCA_CONDITIONAL_ID:
                row["optimization_receipts"] = optimization_receipts[
                    "activation_pca"
                ]
            elif arm_id in (FISHER_UNIT_ID, FISHER_INTERCEPT_ID):
                row["derived_from_fold_conditional_provider_artifact_sha256s"] = {
                    held: optimized["reverse_vjp_fisher"][held].artifact_sha256
                    for held in sorted(providers)
                }
            arm_rows[arm_id] = row

        geometry = prerequisites.get("v18_coordinate_geometry_qualification")
        if not isinstance(geometry, Mapping):
            raise TypeError("V19 prerequisite geometry differs")
        outer = _evaluate_outer_gates(
            arm_rows,
            coordinate_geometry=geometry,
        )
        fitted_full_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider | None = None
        publish_provider: AutonomousCompleteH4FisherFiniteJointPedalProvider | None = None
        full_qualification: dict[str, object] | None = None
        if outer["passed"] is True:
            all_sequences = tuple(record.sequence for record in records)
            full_parent = _fit_parent(
                all_sequences,
                bridge_binding_sha256=bridge_binding,
            )
            _v18._validate_parent(
                full_parent, expected_fit_family_count=_EXPECTED_FAMILIES
            )
            full_start = _fit_v18_start(
                all_sequences,
                parent=full_parent,
                coordinate_objective="reverse_vjp_fisher",
            )
            full_capability = teacher_vault.capability(
                tuple(record.sequence.example_id for record in records),
                held_family_id=None,
            )
            full_result = _optimize_joint_provider(
                context,
                records,
                full_capability,
                full_start,
                held_family_id=None,
                coordinate_objective="reverse_vjp_fisher",
            )
            fitted_full_provider = full_result.provider
            _validate_joint_provider(
                fitted_full_provider,
                start_provider=full_start,
                pedal_mode="conditional",
                expected_family_count=_EXPECTED_FAMILIES,
            )
            runtime = _held_runtime_diagnostics(
                fitted_full_provider, all_sequences
            )
            full_passed = bool(
                full_result.receipt["fit_qualification"]["passed"] is True  # type: ignore[index]
                and full_result.receipt["fit_qualification"][  # type: ignore[index]
                    "objective_macro_relative_improvement_at_least_threshold"
                ]
                is True
                and runtime["pointwise_trust_passed"] is True
                and runtime["pedal_nonconstant"] is True
            )
            full_qualification = {
                "provider_artifact_sha256": fitted_full_provider.artifact_sha256,
                "parent_provider_artifact_sha256": (
                    fitted_full_provider.parent_provider.artifact_sha256
                ),
                "start_provider_artifact_sha256": (
                    fitted_full_provider.start_provider_artifact_sha256
                ),
                "optimization_receipt": full_result.receipt,
                "runtime_diagnostic": runtime,
                "serving_resources": _serving_resources(fitted_full_provider),
                "passed": full_passed,
            }
            if full_passed:
                publish_provider = fitted_full_provider

        candidate = (
            None
            if publish_provider is None
            else {
                "arm_id": FISHER_CONDITIONAL_ID,
                "provider_artifact_sha256": publish_provider.artifact_sha256,
                "parent_provider_artifact_sha256": (
                    publish_provider.parent_provider.artifact_sha256
                ),
                "start_provider_artifact_sha256": (
                    publish_provider.start_provider_artifact_sha256
                ),
                "provider": publish_provider.metadata(),
                "serving_resources": _serving_resources(publish_provider),
                "fit_family_ids": publish_provider.fit_family_ids,
                "fit_sequence_sha256s": publish_provider.fit_sequence_sha256s,
                "fresh_family_disjoint_protocol_required_before_serving": True,
            }
        )
        context.validate_immutable_inputs()
        teacher_vault.validate_integrity()
        work = _work_accounting(
            full_refit_fitted=fitted_full_provider is not None
        )
        integrity = {
            "outer_fold_count": len(folds),
            "ledger_coverage": ledger_coverage,
            **work,
            "causal_off_support_execution_checks": causal_checks,
            "expected_causal_off_support_execution_checks": _EXPECTED_CAUSAL_CHECKS,
            "teacher_vault": teacher_vault.receipt(),
            "outer_teacher_capability_count": 2 * len(folds),
            "every_outer_capability_authorized_fourteen_training_prompts": all(
                receipt["capability_receipt"]["authorized_example_count"]
                == _EXPECTED_TRAINING_PROMPTS_PER_FOLD
                for objective in optimization_receipts.values()
                for receipt in objective.values()
            ),
            "every_outer_capability_accessed_each_training_prompt_five_times": all(
                set(
                    receipt["capability_receipt"]["per_example_access_counts"].values()
                )
                == {_OPTIMIZER_STEPS + 1}
                for objective in optimization_receipts.values()
                for receipt in objective.values()
            ),
            "held_rows_capability_excluded_and_not_consumed_by_each_fold_fit": True,
            "held_evaluation_started_after_all_outer_provider_hashes_froze": True,
            "every_parent_fisher_start_and_pca_start_fold_exactly_reconstructed_pinned_v18_artifact": True,
            "coordinate_geometry_inherited_only_after_exact_v18_lineage_reconstruction": True,
            "base_parent_and_fisher_start_held_fidelity_exactly_replayed_pinned_v18": True,
            "pinned_v18_fidelity_replay_receipt_sha256": v18_fidelity_replay_receipt,
            "source_teacher_rows_entered_serving_provider": False,
            "optimizer_state_entered_serving_provider": False,
            "cast_once_boundary_gradient_was_STE": True,
            "guard_opened": False,
            "calibration_b_opened": False,
        }
        if (
            len(parents) != _EXPECTED_OUTER_FOLDS
            or causal_checks != _EXPECTED_CAUSAL_CHECKS
            or work["outer_provider_fit_count"] != _EXPECTED_OUTER_PROVIDER_FITS
            or not integrity[
                "every_outer_capability_authorized_fourteen_training_prompts"
            ]
            or not integrity[
                "every_outer_capability_accessed_each_training_prompt_five_times"
            ]
        ):
            raise RuntimeError("V19 exact execution geometry differs")
        report = build_finite_joint_pedal_development_report(
            artifact_path=destination,
            panel=context.panel_receipt,
            bridge_binding_sha256=bridge_binding,
            folds=folds,
            prerequisites=prerequisites,
            fit_collection={
                "prompt_count": len(records),
                "family_count": len(families),
                "supervised_token_count": sum(
                    record.supervised_token_count for record in records
                ),
                "trace_receipt_sha256s": tuple(
                    record.receipt_sha256 for record in records
                ),
                "teacher_vault": teacher_vault.receipt(),
                "full_native_logits_transiently_materialized_then_discarded": True,
                "only_supervised_source_rows_cached_in_native_dtype": True,
                "held_rows_cached_but_capability_excluded_and_not_consumed_by_fold_fit": True,
                "raw_fit_trace_or_teacher_tensor_serialization": False,
            },
            base_fidelity=fidelity["base"],
            arms=arm_rows,
            full_refit_qualification=full_qualification,
            candidate=candidate,
            integrity=integrity,
        )
        return _publish(
            report,
            output=destination,
            provider=publish_provider,
            provider_output=provider_destination,
        )
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider-output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_finite_joint_pedal_development(
        output=arguments.output,
        provider_output=arguments.provider_output,
        cache_dir=arguments.cache_dir,
    )
    print(
        json.dumps(
            {
                "path": report["artifact"]["path"],  # type: ignore[index]
                "report_sha256": report["report_sha256"],
                "classification": report["classification"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
