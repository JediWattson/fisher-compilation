"""A16 held-family teacher-KL signed-joint complete-H4 tail screen.

This is a separate hypothesis-use experiment layered on the authenticated
rank-320 carrier.  Exact token-wise gradients of
``KL(native_logits || D320_logits)`` are collected on the canonical complete-
H4 support grid.  Seven training families greedily discover signed joint
directions in ``null(D320)`` and fit nonnegative gains; the eighth family is
used only for finite evaluation.  Gained prefixes at K=8/16/32/64 are
compared with the already-completed PCA/diagonal-v1 control without rerunning
that control.  K=320 is an explicitly ungained exact-complement sentinel.

Native residuals and native teacher logits make this truth-leaking,
same-Calibration-A hypothesis evidence.  It does not fit a serving provider
and makes no compression, speed, or deployment claim.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from .complete_h4_tail_signed_joint_projector import (
    CompleteH4TailSignedJointHeldFamilyFit,
    complete_h4_tail_signed_joint_prediction,
    complete_h4_tail_signed_joint_scores,
    fit_complete_h4_tail_signed_joint_held_family,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailEndpointExample,
    fit_complete_h4_tail_held_family,
    project_complete_h4_tail_rows,
)
from .gemma3_l3_l4_complete_h4_projection import CompleteH4ProjectionFitSequence
from .gemma3_l3_l4_complete_h4_one_pass_transfer import (
    AuthenticatedCompleteH4TransferProvider,
    _ValidatedRank320BasisContract,
    _load_committed_basis,
    _native_boundary,
    _retokenize,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
    _require_sha256,
    _runtime_tensor_sha256,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "CONTROL_REPORT_FILE_SHA256",
    "CONTROL_REPORT_SHA256",
    "DEFAULT_CONTROL_REPORT",
    "DEFAULT_EXPANDED_CONTROL_REPORT",
    "DEFAULT_MATERIALIZATION_REPORT",
    "DEFAULT_OUTPUT",
    "DEFAULT_TRANSFER_REPORT",
    "SIGNED_JOINT_RANKS",
    "run_gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v1.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v1.DEFAULT_TRANSFER_REPORT
DEFAULT_CONTROL_REPORT = v1.DEFAULT_OUTPUT
DEFAULT_EXPANDED_CONTROL_REPORT = expanded.DEFAULT_OUTPUT
DEFAULT_OUTPUT = v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "teacher-kl-signed-joint-lofo-finite-ladder-a-fit16-dev-v1.json"
)

CONTROL_REPORT_FILE_SHA256 = (
    "3cb0c95f5848975cfc3b9e93c16c3837dc3282f667b04d90c9c899bf241e8f6c"
)
CONTROL_REPORT_SHA256 = (
    "d52a89529635c6c6fb6bbaa2ccb48a7a3fdbadfa395ee2ccfdbe79a7c45fea67"
)
EXPANDED_CONTROL_REPORT_FILE_SHA256 = (
    "e7736b60084c8e5bbb83f44cc77613e09242848230137def3afa862162284721"
)
EXPANDED_CONTROL_REPORT_SHA256 = (
    "26010938b5b81dbce9e05607acd46e5b9e0beea1d981edbd91d5e841365799fa"
)

SIGNED_JOINT_RANKS = (8, 16, 32, 64, 320)
_SIGNED_MAX_RANK = 64
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_teacher_kl_signed_joint_lofo.v1"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-tail-teacher-kl-signed-joint-lofo:v1\0"
)
_OBSERVATION_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-signed-joint-observation:v1\0"
)
_OBSERVATION_SET_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-signed-joint-observation-set:v1\0"
)
_CAUSALITY_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-causality:v1\0"
)
_CONTROL_COMPARISON_DOMAIN = (
    b"fisher-graph:complete-h4-tail-signed-joint-control-comparison:v1\0"
)
_PROVIDER_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-signed-joint-provider:v1\0"
)
_PCA_CONTROL_DOMAIN = (
    b"fisher-graph:complete-h4-tail-teacher-kl-fixed-pca-signed-control:v1\0"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_control_report(path: Path | str) -> dict[str, object]:
    """Authenticate the completed v1 PCA/diagonal control and its grid."""

    report = v1._load_pinned_report(
        path,
        expected_file_sha256=CONTROL_REPORT_FILE_SHA256,
        expected_report_sha256=CONTROL_REPORT_SHA256,
        label="teacher-KL signed-joint PCA/diagonal control",
    )
    # The expanded-rung loader already fail-closes on the v1 protocol,
    # materialization/transfer pins, scientific status, safety, rank decisions,
    # and every finite-observation receipt.  Keep one authority for that
    # contract rather than implementing a weaker parallel validator here.
    authenticated = expanded._load_adaptive_parent(path)
    if authenticated != report:
        raise RuntimeError("signed-joint control loaders disagree")
    return report


def _load_expanded_control_report(path: Path | str) -> dict[str, object]:
    report = v1._load_pinned_report(
        path,
        expected_file_sha256=EXPANDED_CONTROL_REPORT_FILE_SHA256,
        expected_report_sha256=EXPANDED_CONTROL_REPORT_SHA256,
        label="teacher-KL signed-joint expanded PCA/diagonal reference",
    )
    binding = _mapping(report.get("input_binding"), label="expanded reference binding")
    parent_binding = _mapping(
        report.get("adaptive_parent_binding"),
        label="expanded reference parent binding",
    )
    science = _mapping(
        report.get("scientific_status"), label="expanded reference science"
    )
    ranks = tuple(_mapping(report.get("protocol"), label="expanded reference protocol").get("tail_ranks", ()))
    if (
        report.get("schema") != expanded._SCHEMA
        or report.get("classification")
        != "adaptive_same_a_smallest_tail_rank_256_cleared_established_gates"
        or ranks != expanded.EXPANDED_TAIL_RANKS
        or report.get(
            "smallest_tail_rank_below_320_clearing_established_fidelity_and_geometry_gates"
        )
        != 256
        or parent_binding.get("file_sha256")
        != CONTROL_REPORT_FILE_SHA256
        or parent_binding.get("report_sha256") != CONTROL_REPORT_SHA256
        or binding.get("materialization_report_file_sha256")
        != v1.MATERIALIZATION_REPORT_FILE_SHA256
        or binding.get("transfer_report_file_sha256")
        != v1.TRANSFER_REPORT_FILE_SHA256
        or science.get("same_a_adaptive_hypothesis_use_only") is not True
        or science.get("fresh_confirmation_panel_opened") is not False
        or science.get("candidate_serving_authorized") is not False
        or science.get("compression_claim") is not False
    ):
        raise ValueError("expanded PCA/diagonal reference semantics differ")
    raw_observations = report.get("finite_observation_receipts")
    if not isinstance(raw_observations, list):
        raise ValueError("expanded reference observations differ")
    receipt = v1._finite_observation_set_sha256(
        [_mapping(row, label="expanded reference observation") for row in raw_observations],
        ranks=expanded.EXPANDED_TAIL_RANKS,
    )
    if receipt != report.get("finite_observation_set_sha256"):
        raise RuntimeError("expanded reference observation set drifted")
    return report


def _transfer_receipts(
    transfer: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    return expanded._transfer_receipts(transfer)


class _AuthenticatedSignedJointFiniteProvider(Gemma3L3L4CorrectionProvider):
    """Single-use integrity-bound gained prefix or exact K320 sentinel."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "rank",
        "arm_kind",
        "fold_artifact_sha256",
        "model_inputs_sha256",
        "bridge_binding_sha256",
        "prefix_artifact_sha256",
        "base_h4_sha256",
        "support_mask_sha256",
        "correction_sha256",
        "_support",
        "_correction",
        "_used",
    )

    def __init__(
        self,
        *,
        rank: int,
        arm_kind: str,
        fold_artifact_sha256: str,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        base_h4: Tensor,
        support_mask: Tensor,
        correction: Tensor,
    ) -> None:
        allowed_bounded_arms = frozenset(
            {
                "teacher_kl_signed_joint_rotated_gain_prefix",
                "teacher_kl_fixed_pca_q2_signed_gain_prefix",
            }
        )
        if type(rank) is not int or rank not in SIGNED_JOINT_RANKS:
            raise ValueError("signed-joint provider rank is outside the fixed ladder")
        if (
            rank == v1._D_RANK
            and arm_kind != "ungained_exact_complement_sentinel"
        ) or (rank < v1._D_RANK and arm_kind not in allowed_bounded_arms):
            raise ValueError("signed-joint provider arm kind differs from rank")
        if (
            not isinstance(base_h4, Tensor)
            or base_h4.ndim != 3
            or base_h4.shape[-1] != v1._WIDTH
            or not base_h4.is_floating_point()
            or not isinstance(support_mask, Tensor)
            or support_mask.shape != base_h4.shape[:2]
            or support_mask.dtype != torch.bool
            or not isinstance(correction, Tensor)
            or correction.shape != base_h4.shape
            or not correction.is_floating_point()
        ):
            raise ValueError("signed-joint provider tensor geometry differs")
        support = support_mask.detach().to(device="cpu").clone().contiguous()
        delta = correction.detach().to(
            device="cpu", dtype=torch.float64
        ).clone().contiguous()
        if not bool(torch.isfinite(delta).all()) or bool((delta[~support] != 0).any()):
            raise ValueError("signed-joint correction escapes support")
        self.site = v1._H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.rank = rank
        self.arm_kind = arm_kind
        self.fold_artifact_sha256 = _require_sha256(
            fold_artifact_sha256, label="signed-joint fold artifact"
        )
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="signed-joint model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="signed-joint bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="signed-joint prefix"
        )
        self.base_h4_sha256 = _runtime_tensor_sha256(base_h4)
        self._support = support
        self._correction = delta
        self.support_mask_sha256 = _runtime_tensor_sha256(support)
        self.correction_sha256 = _runtime_tensor_sha256(delta)
        self._used = False
        self.artifact_sha256 = self._computed_sha256()
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.complete_h4_teacher_kl_signed_joint_provider.v1",
            "rank": self.rank,
            "arm_kind": self.arm_kind,
            "site": self.site,
            "write_scope": self.write_scope,
            "fold_artifact_sha256": self.fold_artifact_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_sha256": self.base_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "correction_sha256": self.correction_sha256,
            "correction_semantics": (
                {
                    "teacher_kl_signed_joint_rotated_gain_prefix": (
                        "P_D320_R_plus_gain_scaled_training_only_teacher_KL_"
                        "signed_joint_rotated_prefix"
                    ),
                    "teacher_kl_fixed_pca_q2_signed_gain_prefix": (
                        "P_D320_R_plus_gain_scaled_training_only_teacher_KL_"
                        "fixed_residual_PCA_q2_ordered_prefix"
                    ),
                    "ungained_exact_complement_sentinel": (
                        "P_D320_R_plus_ungained_full_orthogonal_complement_tail"
                    ),
                }[self.arm_kind]
            ),
            "exact_residual_provider_substitution_used": False,
            "held_native_tail_instantiated": True,
            "single_use": True,
            "truth_leaking_hypothesis_use_only": True,
            "serving_authorized": False,
        }

    def _computed_sha256(self) -> str:
        return v1._domain_sha256(self._payload(), domain=_PROVIDER_DOMAIN)

    @property
    def used(self) -> bool:
        return self._used

    def validate_integrity(self) -> None:
        if (
            self.site != v1._H4_SITE
            or self.write_scope != "complete_h4_causal_support"
            or _runtime_tensor_sha256(self._support) != self.support_mask_sha256
            or _runtime_tensor_sha256(self._correction) != self.correction_sha256
            or bool((self._correction[~self._support] != 0).any())
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise RuntimeError("signed-joint finite provider payload drifted")

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("signed-joint finite provider cannot be reused")
        prefix.validate_integrity()
        if (
            prefix.artifact_sha256 != self.prefix_artifact_sha256
            or prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or _runtime_tensor_sha256(realized_state) != self.base_h4_sha256
            or _runtime_tensor_sha256(
                prefix.complete_h4_causal_support_mask()
                .detach()
                .to(device="cpu")
                .contiguous()
            )
            != self.support_mask_sha256
        ):
            raise RuntimeError("signed-joint provider reached another execution")
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(slots=True)
class _LiveTeacherKLTrace:
    example: object = field(repr=False)
    example_id: str
    family_id: str
    model_inputs_sha256: str
    supervised_indices_sha256: str
    supervised_targets_sha256: str
    endpoint_indices_sha256: str
    endpoint_targets_sha256: str
    prefix: Gemma3L3L4OnePassPrefix = field(repr=False)
    base_x4_sha256: str
    base_h4: Tensor = field(repr=False)
    native_h4: Tensor = field(repr=False)
    support_indices: Tensor = field(repr=False)
    selected_by_ledger: Mapping[str, Tensor] = field(repr=False)
    endpoint: CompleteH4TailEndpointExample = field(repr=False)
    native_token_nll: Tensor = field(repr=False)
    d320_token_nll: Tensor = field(repr=False)
    native_ordinary_token_nll: Tensor = field(repr=False)
    d320_ordinary_token_nll: Tensor = field(repr=False)
    native_logits_sha256: str
    teacher_logits_sha256: str
    endpoint_vjp_artifact_sha256: str
    endpoint_execution_artifact_sha256: str
    endpoint_provider_artifact_sha256: str
    backward_call_count: int
    maximum_future_gradient_abs: float
    future_gradient_nonzero_count: int
    causality_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _FixedPCASignedControlFit:
    held_family_id: str
    pca_diagonal_fit_artifact_sha256: str
    training_endpoint_artifact_sha256s: tuple[str, ...]
    directions: Tensor = field(repr=False)
    gains: tuple[float, ...]
    initial_rmse: float
    final_rmse: float
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        held = v1._identifier(self.held_family_id, label="PCA control held family")
        source_fit = _require_sha256(
            self.pca_diagonal_fit_artifact_sha256,
            label="PCA control source fit",
        )
        evidence = tuple(
            _require_sha256(value, label="PCA control endpoint")
            for value in self.training_endpoint_artifact_sha256s
        )
        directions = self.directions.detach().to(
            device="cpu", dtype=torch.float64
        ).clone().contiguous()
        gains = tuple(float(value) for value in self.gains)
        if (
            directions.shape != (_SIGNED_MAX_RANK, v1._WIDTH)
            or len(gains) != _SIGNED_MAX_RANK
            or not evidence
            or not bool(torch.isfinite(directions).all())
            or not torch.allclose(
                directions @ directions.T,
                torch.eye(_SIGNED_MAX_RANK, dtype=torch.float64),
                rtol=0.0,
                atol=1.0e-9,
            )
            or any(not math.isfinite(value) or not 0.0 <= value <= 2.0 for value in gains)
            or not math.isfinite(self.initial_rmse)
            or not math.isfinite(self.final_rmse)
            or self.initial_rmse < 0.0
            or self.final_rmse < 0.0
            or self.final_rmse > self.initial_rmse + 1.0e-12
        ):
            raise ValueError("fixed-PCA signed control fit is invalid")
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "pca_diagonal_fit_artifact_sha256", source_fit)
        object.__setattr__(self, "training_endpoint_artifact_sha256s", evidence)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(
            self,
            "artifact_sha256",
            v1._domain_sha256(self.metadata(include_artifact=False), domain=_PCA_CONTROL_DOMAIN),
        )
        self.validate_integrity()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "pca_diagonal_fit_artifact_sha256": self.pca_diagonal_fit_artifact_sha256,
            "training_endpoint_artifact_sha256s": self.training_endpoint_artifact_sha256s,
            "directions_shape": tuple(self.directions.shape),
            "directions_sha256": _runtime_tensor_sha256(self.directions),
            "gains": self.gains,
            "gains_sha256": _runtime_tensor_sha256(
                torch.tensor(self.gains, dtype=torch.float64)
            ),
            "initial_rmse": self.initial_rmse,
            "final_rmse": self.final_rmse,
            "basis_and_order": "training_only_residual_PCA_then_teacher_KL_q2_diagonal_order",
            "gain_fit": "same_signed_sequential_closed_interval_0_2",
            "weighting": "equal_family_then_equal_prompt_then_equal_token",
            "held_family_used_for_basis_order_or_gain": False,
            "raw_evidence_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            _runtime_tensor_sha256(self.directions)
            != self.metadata(include_artifact=False)["directions_sha256"]
            or v1._domain_sha256(
                self.metadata(include_artifact=False), domain=_PCA_CONTROL_DOMAIN
            )
            != self.artifact_sha256
        ):
            raise RuntimeError("fixed-PCA signed control fit drifted")


def _nested_scalar_mean(
    examples: Sequence[CompleteH4TailEndpointExample],
    values: Mapping[str, Tensor],
) -> Tensor:
    by_family: dict[str, list[Tensor]] = defaultdict(list)
    for example in examples:
        value = values[example.example_id]
        if value.shape != (example.supervised_tokens,):
            raise ValueError("nested scalar statistic length differs")
        by_family[example.family_id].append(value.mean())
    return torch.stack(
        [torch.stack(by_family[family]).mean() for family in sorted(by_family)]
    ).mean()


def _fit_fixed_pca_signed_control(
    examples: Sequence[CompleteH4TailEndpointExample],
    *,
    supported_basis: Tensor,
    held_family_id: str,
) -> _FixedPCASignedControlFit:
    pca_fit = fit_complete_h4_tail_held_family(
        examples,
        supported_basis=supported_basis,
        held_family_id=held_family_id,
    )
    ordered_examples = tuple(sorted(examples, key=lambda value: value.example_id))
    if (
        len({example.example_id for example in ordered_examples})
        != len(ordered_examples)
    ):
        raise ValueError("fixed-PCA control endpoint ids must be unique")
    for example in ordered_examples:
        example.validate_integrity()
    training = tuple(
        example
        for example in ordered_examples
        if example.family_id != held_family_id
    )
    directions = pca_fit.ordered_basis_rows()[:_SIGNED_MAX_RANK].contiguous()
    scores = {
        example.example_id: complete_h4_tail_signed_joint_scores(
            example, directions
        )
        for example in training
    }
    residuals = {
        example.example_id: example.compensation_target.clone()
        for example in training
    }
    initial_rmse = math.sqrt(
        float(
            _nested_scalar_mean(
                training, {key: value.square() for key, value in residuals.items()}
            )
        )
    )
    gains: list[float] = []
    total_tokens = sum(example.supervised_tokens for example in training)
    eps = torch.finfo(torch.float64).eps
    for index in range(_SIGNED_MAX_RANK):
        q = {key: value[:, index] for key, value in scores.items()}
        numerator = float(
            _nested_scalar_mean(
                training, {key: residuals[key] * q[key] for key in residuals}
            )
        )
        second = float(
            _nested_scalar_mean(training, {key: value.square() for key, value in q.items()})
        )
        gain_epsilon = 256.0 * eps * max(v1._D_RANK, len(training), total_tokens) * second
        gain = 0.0 if second <= 0.0 else min(max(numerator / (second + gain_epsilon), 0.0), 2.0)
        gains.append(gain)
        for key in residuals:
            residuals[key] = (residuals[key] - gain * q[key]).contiguous()
    final_rmse = math.sqrt(
        float(
            _nested_scalar_mean(
                training, {key: value.square() for key, value in residuals.items()}
            )
        )
    )
    return _FixedPCASignedControlFit(
        held_family_id=held_family_id,
        pca_diagonal_fit_artifact_sha256=pca_fit.artifact_sha256,
        training_endpoint_artifact_sha256s=tuple(
            example.artifact_sha256 for example in training
        ),
        directions=directions,
        gains=tuple(gains),
        initial_rmse=initial_rmse,
        final_rmse=final_rmse,
    )


def _fixed_pca_signed_prediction(
    endpoint: CompleteH4TailEndpointExample,
    fit: _FixedPCASignedControlFit,
    *,
    rank: int,
) -> Tensor:
    fit.validate_integrity()
    if type(rank) is not int or rank not in SIGNED_JOINT_RANKS[:-1]:
        raise ValueError("fixed-PCA control rank is outside the bounded ladder")
    scores = complete_h4_tail_signed_joint_scores(endpoint, fit.directions[:rank])
    gains = torch.tensor(fit.gains[:rank], dtype=torch.float64)
    return (scores @ gains).contiguous()


def _fixed_pca_signed_tail_prefix(
    tail_rows: Tensor,
    fit: _FixedPCASignedControlFit,
    *,
    rank: int,
) -> Tensor:
    fit.validate_integrity()
    if type(rank) is not int or rank not in SIGNED_JOINT_RANKS[:-1]:
        raise ValueError("fixed-PCA control rank is outside the bounded ladder")
    directions = fit.directions[:rank]
    gains = torch.tensor(fit.gains[:rank], dtype=torch.float64)
    return (((tail_rows @ directions.T) * gains) @ directions).contiguous()


def _canonical_support_grid(endpoint_indices: Tensor) -> Tensor:
    if (
        not isinstance(endpoint_indices, Tensor)
        or endpoint_indices.ndim != 1
        or endpoint_indices.dtype != torch.int64
        or endpoint_indices.numel() <= 0
    ):
        raise ValueError("endpoint support indices must be nonempty int64 [N]")
    values = endpoint_indices.detach().to(device="cpu").contiguous()
    if bool((values < 0).any()) or (
        values.numel() > 1 and not bool((values[1:] > values[:-1]).all())
    ):
        raise ValueError("endpoint support indices are not canonical")
    return torch.stack((torch.zeros_like(values), values), dim=1).contiguous()


def _selected_token_teacher_kl(
    teacher_logits: Tensor,
    candidate_logits: Tensor,
    indices: Tensor,
) -> Tensor:
    if teacher_logits.shape != candidate_logits.shape or teacher_logits.ndim != 3:
        raise ValueError("teacher/candidate logits grids differ")
    teacher = frozen._select_sequence_rows(teacher_logits, indices)
    candidate = frozen._select_sequence_rows(candidate_logits, indices)
    if teacher.dtype in (torch.float16, torch.bfloat16):
        teacher = teacher.float()
    if candidate.dtype in (torch.float16, torch.bfloat16):
        candidate = candidate.float()
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    candidate_logp = torch.log_softmax(candidate, dim=-1)
    return (
        teacher_logp.exp() * (teacher_logp - candidate_logp)
    ).sum(dim=-1).detach().to(device="cpu", dtype=torch.float64).contiguous()


def _signed_tail_prefix(
    tail_rows: Tensor,
    fit: CompleteH4TailSignedJointHeldFamilyFit,
    *,
    rank: int,
) -> Tensor:
    """Return the gained signed prefix, or the ungained exact K320 sentinel."""

    fit.validate_integrity()
    if type(rank) is not int or rank not in SIGNED_JOINT_RANKS:
        raise ValueError("signed-joint rank is outside the fixed ladder")
    tail = tail_rows.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if tail.ndim != 2 or tail.shape[1] != fit.ambient_width:
        raise ValueError("signed-joint tail rows differ from the fit")
    if rank == v1._D_RANK:
        return tail.clone().contiguous()
    effective_rank = min(rank, fit.rank)
    if effective_rank == 0:
        return torch.zeros_like(tail)
    directions = fit.directions_tensor()[:effective_rank]
    gains = torch.tensor(fit.gains[:effective_rank], dtype=torch.float64)
    return (((tail @ directions.T) * gains) @ directions).contiguous()


def _endpoint_prediction(
    endpoint: CompleteH4TailEndpointExample,
    fit: CompleteH4TailSignedJointHeldFamilyFit,
    *,
    rank: int,
    tail_rows: Tensor | None = None,
) -> Tensor:
    if rank == v1._D_RANK:
        # The sentinel changes H4 by the whole native tail without the learned
        # gains.  Its tangent receipt therefore contracts that displacement
        # directly with the exact endpoint gradient.
        if tail_rows is None:
            raise ValueError("K320 endpoint prediction requires tail rows")
        tail = tail_rows.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        if tail.shape != endpoint.residual_rows.shape:
            raise ValueError("K320 endpoint prediction tail geometry differs")
        return torch.einsum(
            "rw,trw->t", tail, endpoint.token_h4_gradients
        ).contiguous()
    return complete_h4_tail_signed_joint_prediction(
        endpoint, fit, rank=min(rank, fit.rank)
    )


def _collect_teacher_kl_endpoint_traces(
    *,
    context: object,
    basis: Tensor,
    basis_binding: Mapping[str, str],
    transfer_receipts: Mapping[str, Mapping[str, object]],
) -> tuple[list[_LiveTeacherKLTrace], dict[str, int]]:
    """Collect one authenticated D320 teacher-KL VJP bank per A16 prompt."""

    adapter = getattr(context, "adapter")
    bridge = getattr(context, "bridge")
    tokenize = getattr(context, "tokenize")
    examples = tuple(getattr(context, "examples"))
    bridge.validate_integrity()
    bridge_sha256 = _require_sha256(
        bridge.bridge_binding_sha256, label="teacher-KL signed-joint bridge"
    )
    basis_contract = _ValidatedRank320BasisContract.build(basis, basis_binding)
    traces: list[_LiveTeacherKLTrace] = []
    resources = {
        "base_forward_count": 0,
        "native_teacher_forward_count": 0,
        "endpoint_teacher_kl_vjp_forward_count": 0,
        "endpoint_teacher_kl_vjp_backward_call_count": 0,
        "ordinary_supervised_token_count": 0,
        "endpoint_support_supervised_token_count": 0,
        "complete_h4_support_row_count": 0,
        "graph_core_row_count": 0,
        "causal_tail_row_count": 0,
        "graph_core_supervised_token_count": 0,
        "causal_tail_supervised_token_count": 0,
    }
    for example in sorted(examples, key=lambda value: value.example_id):
        example_id = v1._identifier(example.example_id, label="example_id")
        family_id = v1._identifier(example.family_id, label="family_id")
        prior = transfer_receipts.get(example_id)
        if prior is None:
            raise ValueError("pinned transfer omitted an A16 prompt receipt")
        model_inputs, indices, targets = _retokenize(tokenize, example)
        model_inputs_sha256 = gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        indices_sha256 = _runtime_tensor_sha256(indices)
        targets_sha256 = _runtime_tensor_sha256(targets)
        if (
            prior.get("family_id") != family_id
            or prior.get("supervised_indices_sha256") != indices_sha256
            or prior.get("supervised_targets_sha256") != targets_sha256
        ):
            raise RuntimeError("teacher-KL supervision differs from pinned transfer")

        base = bridge.execute(adapter, model_inputs)
        resources["base_forward_count"] += 1
        if getattr(base, "model_forward_count", None) != 1:
            raise RuntimeError("teacher-KL base execution forward count differs")
        prefix = base.prefix
        prefix.validate_integrity()
        base_h4 = base.candidate_h4
        if (
            prior.get("model_inputs_sha256") != model_inputs_sha256
            or prior.get("bridge_binding_sha256") != bridge_sha256
            or prior.get("prefix_artifact_sha256") != prefix.artifact_sha256
            or prior.get("base_candidate_h4_sha256")
            != _runtime_tensor_sha256(base_h4)
            or prior.get("base_candidate_x4_sha256")
            != _runtime_tensor_sha256(base.candidate_x4)
        ):
            raise RuntimeError("teacher-KL base identity differs from pinned transfer")

        teacher_logits, native_h4, native_positions, native_valid = _native_boundary(
            adapter, model_inputs
        )
        resources["native_teacher_forward_count"] += 1
        if (
            native_h4.shape != base_h4.shape
            or not v1._bitwise_equal(native_positions, prefix.logical_positions)
            or not v1._bitwise_equal(native_valid, prefix.valid_target_mask)
        ):
            raise RuntimeError("teacher-KL native/base boundary differs")
        teacher_logits_sha256 = _runtime_tensor_sha256(teacher_logits)
        if (
            teacher_logits_sha256 != prior.get("source_logits_sha256")
            or _runtime_tensor_sha256(native_h4) != prior.get("native_h4_sha256")
        ):
            raise RuntimeError("teacher-KL native boundary differs from pinned transfer")

        support = prefix.complete_h4_causal_support_mask().detach().to(device="cpu")
        core = prefix.target_affected_mask.detach().to(device="cpu")
        support_indices = torch.nonzero(support[0], as_tuple=False).flatten().to(
            dtype=torch.int64
        )
        positions_cpu = indices.detach().to(device="cpu")
        support_supervised = support[0].index_select(0, positions_cpu)
        core_supervised = core[0].index_select(0, positions_cpu)
        endpoint_selection = torch.nonzero(
            support_supervised, as_tuple=False
        ).flatten().to(dtype=torch.int64)
        endpoint_indices = indices.index_select(
            0, endpoint_selection.to(indices.device)
        )
        endpoint_targets = targets.index_select(
            0, endpoint_selection.to(targets.device)
        )
        endpoint_grid = _canonical_support_grid(
            endpoint_indices.detach().to(device="cpu", dtype=torch.int64)
        )
        selected_by_ledger = {
            "ordinary": torch.arange(indices.numel(), dtype=torch.int64),
            "complete_h4_support": endpoint_selection,
            "graph_core": torch.nonzero(
                core_supervised, as_tuple=False
            ).flatten().to(dtype=torch.int64),
            "causal_tail": torch.nonzero(
                support_supervised & ~core_supervised, as_tuple=False
            ).flatten().to(dtype=torch.int64),
        }
        if (
            prior.get("complete_h4_support_mask_sha256")
            != _runtime_tensor_sha256(support)
            or prior.get("complete_h4_support_rows") != int(support.sum())
            or prior.get("graph_core_rows") != int(core.sum())
            or prior.get("causal_tail_rows") != int((support & ~core).sum())
        ):
            raise RuntimeError("teacher-KL endpoint support differs from pinned transfer")

        provider = AuthenticatedCompleteH4TransferProvider(
            role="rank320_projection",
            model_inputs_sha256=model_inputs_sha256,
            bridge_binding_sha256=bridge_sha256,
            prefix_artifact_sha256=prefix.artifact_sha256,
            base_h4=base_h4,
            native_h4=native_h4,
            basis_contract=basis_contract,
            support_mask=support,
        )
        token_vjp = bridge.execute_h4_token_teacher_kl_vjps(
            adapter,
            model_inputs,
            teacher_logits=teacher_logits,
            supervised_indices=endpoint_grid,
            vjp_chunk_size=v1._VJP_CHUNK_SIZE,
            h4_head=provider,
        )
        token_vjp.validate_integrity()
        provider.validate_integrity()
        if not provider.used:
            raise RuntimeError("authenticated D320 teacher-KL provider was not consumed")
        resources["endpoint_teacher_kl_vjp_forward_count"] += 1
        resources["endpoint_teacher_kl_vjp_backward_call_count"] += (
            token_vjp.backward_call_count
        )
        resources["ordinary_supervised_token_count"] += int(indices.numel())
        resources["endpoint_support_supervised_token_count"] += int(
            endpoint_indices.numel()
        )
        resources["complete_h4_support_row_count"] += int(support.sum())
        resources["graph_core_row_count"] += int(core.sum())
        resources["causal_tail_row_count"] += int((support & ~core).sum())
        resources["graph_core_supervised_token_count"] += int(
            selected_by_ledger["graph_core"].numel()
        )
        resources["causal_tail_supervised_token_count"] += int(
            selected_by_ledger["causal_tail"].numel()
        )

        if not torch.equal(
            token_vjp.supervised_indices.detach().to(device="cpu"), endpoint_grid
        ):
            raise RuntimeError("teacher-KL endpoint token order differs")
        projected_provider = prior.get("projected_provider")
        if not isinstance(projected_provider, Mapping):
            raise ValueError("pinned transfer projected provider receipt differs")
        endpoint_execution = token_vjp.execution
        if (
            endpoint_execution.model_inputs_sha256 != model_inputs_sha256
            or endpoint_execution.bridge_binding_sha256 != bridge_sha256
            or endpoint_execution.prefix.artifact_sha256 != prefix.artifact_sha256
            or endpoint_execution.h4_head_sha256 != provider.artifact_sha256
            or _runtime_tensor_sha256(endpoint_execution.candidate_x4)
            != prior.get("base_candidate_x4_sha256")
            or provider.artifact_sha256 != projected_provider.get("artifact_sha256")
            or _runtime_tensor_sha256(endpoint_execution.candidate_h4)
            != prior.get("projected_h4_sha256")
            or _runtime_tensor_sha256(endpoint_execution.logits)
            != prior.get("projected_logits_sha256")
            or token_vjp.teacher_logits_sha256 != teacher_logits_sha256
            or tuple(token_vjp.teacher_logits_shape)
            != tuple(int(size) for size in teacher_logits.shape)
        ):
            raise RuntimeError(
                "teacher-KL endpoint VJP does not reproduce pinned D320 transfer"
            )

        native_token_nll = v1._selected_token_nll(
            teacher_logits, endpoint_indices, endpoint_targets
        )
        native_ordinary_token_nll = v1._selected_token_nll(
            teacher_logits, indices, targets
        )
        d320_token_nll = v1._selected_token_nll(
            endpoint_execution.logits, endpoint_indices, endpoint_targets
        )
        d320_ordinary_token_nll = v1._selected_token_nll(
            endpoint_execution.logits, indices, targets
        )
        independent_kl = _selected_token_teacher_kl(
            teacher_logits, endpoint_execution.logits, endpoint_indices
        )
        token_kl = token_vjp.token_kl_divergences.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        if not torch.allclose(token_kl, independent_kl, rtol=0.0, atol=1.0e-6):
            raise RuntimeError("teacher-KL endpoint objective authority differs")

        base_cpu = base_h4.detach().to(device="cpu", dtype=torch.float64)
        native_cpu = native_h4.detach().to(device="cpu", dtype=torch.float64)
        residual_rows = (
            native_cpu[0].index_select(0, support_indices)
            - base_cpu[0].index_select(0, support_indices)
        ).contiguous()
        gradient_rows = (
            token_vjp.h4_gradients.detach()
            .to(device="cpu", dtype=torch.float64)[:, 0]
            .index_select(1, support_indices)
            .contiguous()
        )
        projected_rows = ((residual_rows @ basis.T) @ basis).contiguous()
        expected_candidate = base_h4.detach().to(device="cpu").clone()
        expected_candidate[0].index_copy_(
            0,
            support_indices,
            (
                base_h4.detach()[0]
                .index_select(0, support_indices.to(base_h4.device))
                .to(device="cpu", dtype=torch.float64)
                + projected_rows
            ).to(dtype=base_h4.dtype),
        )
        if not v1._bitwise_equal(
            endpoint_execution.candidate_h4.detach().to(device="cpu"),
            expected_candidate,
        ):
            raise RuntimeError("teacher-KL endpoint candidate H4 is not base plus D320")

        logical_cpu = prefix.logical_positions.detach().to(device="cpu")
        supervised_logical = logical_cpu[0].index_select(
            0, endpoint_indices.detach().to(device="cpu")
        )
        support_logical = logical_cpu[0].index_select(0, support_indices)
        maximum_future = 0.0
        future_nonzero = 0
        for token_index in range(int(endpoint_indices.numel())):
            later = support_logical > supervised_logical[token_index]
            if bool(later.any()):
                future = gradient_rows[token_index, later]
                maximum_future = max(maximum_future, float(future.abs().max()))
                future_nonzero += int((future != 0).sum())
        causality_payload = {
            "example_id": example_id,
            "supervised_token_count": int(endpoint_indices.numel()),
            "support_row_count": int(support_indices.numel()),
            "maximum_future_gradient_abs_hex": maximum_future.hex(),
            "future_gradient_nonzero_count": future_nonzero,
            "teacher_kl_vjp_artifact_sha256": token_vjp.artifact_sha256,
        }
        causality_receipt = v1._domain_sha256(
            causality_payload, domain=_CAUSALITY_DOMAIN
        )
        if maximum_future != 0.0 or future_nonzero != 0:
            raise RuntimeError("teacher-KL VJP leaks into logically later H4 rows")

        endpoint = CompleteH4TailEndpointExample(
            example_id=example_id,
            family_id=family_id,
            residual_rows=residual_rows,
            token_h4_gradients=gradient_rows,
            compensation_target=(-token_kl).contiguous(),
        )
        traces.append(
            _LiveTeacherKLTrace(
                example=example,
                example_id=example_id,
                family_id=family_id,
                model_inputs_sha256=model_inputs_sha256,
                supervised_indices_sha256=indices_sha256,
                supervised_targets_sha256=targets_sha256,
                endpoint_indices_sha256=_runtime_tensor_sha256(endpoint_indices),
                endpoint_targets_sha256=_runtime_tensor_sha256(endpoint_targets),
                prefix=prefix,
                base_x4_sha256=_runtime_tensor_sha256(base.candidate_x4),
                base_h4=base_h4.detach().clone().contiguous(),
                native_h4=native_h4.detach().to(device="cpu").contiguous(),
                support_indices=support_indices,
                selected_by_ledger={
                    key: value.clone().contiguous()
                    for key, value in selected_by_ledger.items()
                },
                endpoint=endpoint,
                native_token_nll=native_token_nll,
                d320_token_nll=d320_token_nll,
                native_ordinary_token_nll=native_ordinary_token_nll,
                d320_ordinary_token_nll=d320_ordinary_token_nll,
                native_logits_sha256=teacher_logits_sha256,
                teacher_logits_sha256=teacher_logits_sha256,
                endpoint_vjp_artifact_sha256=token_vjp.artifact_sha256,
                endpoint_execution_artifact_sha256=(
                    token_vjp.execution.artifact_sha256
                ),
                endpoint_provider_artifact_sha256=provider.artifact_sha256,
                backward_call_count=token_vjp.backward_call_count,
                maximum_future_gradient_abs=maximum_future,
                future_gradient_nonzero_count=future_nonzero,
                causality_receipt_sha256=causality_receipt,
            )
        )
        del base, teacher_logits, native_h4, token_vjp, model_inputs

    expected = {
        "ordinary_supervised_token_count": v1._EXPECTED_ORDINARY_TOKENS,
        "endpoint_support_supervised_token_count": v1._EXPECTED_SUPPORT_TOKENS,
        "complete_h4_support_row_count": v1._EXPECTED_SUPPORT_ROWS,
        "graph_core_row_count": v1._EXPECTED_GRAPH_CORE_ROWS,
        "causal_tail_row_count": v1._EXPECTED_CAUSAL_TAIL_ROWS,
        "graph_core_supervised_token_count": v1._EXPECTED_GRAPH_CORE_TOKENS,
        "causal_tail_supervised_token_count": v1._EXPECTED_CAUSAL_TAIL_TOKENS,
    }
    if any(resources[key] != value for key, value in expected.items()):
        raise RuntimeError("teacher-KL endpoint support/token accounting differs")
    return traces, resources


def _finite_dual_arm_observations(
    *,
    context: object,
    traces: Sequence[_LiveTeacherKLTrace],
    basis: Tensor,
    signed_fits: Mapping[str, CompleteH4TailSignedJointHeldFamilyFit],
    pca_control_fits: Mapping[str, _FixedPCASignedControlFit],
) -> tuple[
    list[dict[str, object]],
    dict[str, int],
    dict[str, dict[int, dict[str, object]]],
    dict[str, dict[int, dict[str, object]]],
]:
    """Finite-evaluate both matched teacher-KL methods and one shared K320."""

    adapter = getattr(context, "adapter")
    bridge = getattr(context, "bridge")
    tokenize = getattr(context, "tokenize")
    bounded_ranks = SIGNED_JOINT_RANKS[:-1]
    methods = ("signed_joint", "fixed_pca_diagonal")
    keys = tuple((method, rank) for method in methods for rank in bounded_ranks) + (
        ("shared_exact_sentinel", v1._D_RANK),
    )
    ledgers = (
        "ordinary",
        "complete_h4_support",
        "graph_core",
        "causal_tail",
    )
    manifests = {
        ledger: {
            trace.example_id: trace.family_id
            for trace in traces
            if trace.selected_by_ledger[ledger].numel() > 0
        }
        for ledger in ledgers
    }
    fidelity = {
        key: {
            ledger: SourceAuthoritativeShadowFidelityAccumulator(
                manifests[ledger], gates=ESTABLISHED_SHADOW_FIDELITY_GATES
            )
            for ledger in ledgers
        }
        for key in keys
    }
    geometry_traces: list[object] = []
    for trace in traces:
        geometry_traces.append(
            SimpleNamespace(
                example=trace.example,
                fit_sequence=CompleteH4ProjectionFitSequence(
                    example_id=trace.example_id,
                    family_id=trace.family_id,
                    residual_rows=trace.endpoint.residual_rows,
                ),
                support_indices=trace.support_indices,
                graph_core_rows=(
                    trace.prefix.target_affected_mask.detach()
                    .to(device="cpu")[0]
                    .index_select(0, trace.support_indices)
                ),
            )
        )
    executed_rows: dict[tuple[str, int], dict[str, Tensor]] = {
        key: {} for key in keys
    }
    observations: list[dict[str, object]] = []
    forward_by_key = {key: 0 for key in keys}
    finite_native_forward_count = 0

    for trace in traces:
        signed_fit = signed_fits[trace.family_id]
        pca_fit = pca_control_fits[trace.family_id]
        signed_fit.validate_integrity()
        pca_fit.validate_integrity()
        residual = trace.endpoint.residual_rows
        supported_rows = ((residual @ basis.T) @ basis).contiguous()
        tail_rows = project_complete_h4_tail_rows(residual, basis)
        model_inputs, indices, targets = _retokenize(tokenize, trace.example)
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
            != trace.model_inputs_sha256
            or _runtime_tensor_sha256(indices) != trace.supervised_indices_sha256
            or _runtime_tensor_sha256(targets) != trace.supervised_targets_sha256
        ):
            raise RuntimeError("signed-joint finite retokenization drifted")
        source_logits, native_h4, native_positions, native_valid = _native_boundary(
            adapter, model_inputs
        )
        finite_native_forward_count += 1
        if (
            _runtime_tensor_sha256(source_logits) != trace.native_logits_sha256
            or not v1._bitwise_equal(
                native_h4.detach().to(device="cpu"), trace.native_h4
            )
            or not v1._bitwise_equal(native_positions, trace.prefix.logical_positions)
            or not v1._bitwise_equal(native_valid, trace.prefix.valid_target_mask)
        ):
            raise RuntimeError("signed-joint finite native boundary drifted")
        source_selected = frozen._select_sequence_rows(source_logits, indices)
        support = trace.prefix.complete_h4_causal_support_mask().detach().to(
            device="cpu"
        )

        arms: list[tuple[str, int, Tensor, Tensor, str, str]] = []
        for rank in bounded_ranks:
            arms.append(
                (
                    "signed_joint",
                    rank,
                    _signed_tail_prefix(tail_rows, signed_fit, rank=rank),
                    _endpoint_prediction(
                        trace.endpoint, signed_fit, rank=rank
                    ),
                    signed_fit.artifact_sha256,
                    "teacher_kl_signed_joint_rotated_gain_prefix",
                )
            )
            arms.append(
                (
                    "fixed_pca_diagonal",
                    rank,
                    _fixed_pca_signed_tail_prefix(tail_rows, pca_fit, rank=rank),
                    _fixed_pca_signed_prediction(
                        trace.endpoint, pca_fit, rank=rank
                    ),
                    pca_fit.artifact_sha256,
                    "teacher_kl_fixed_pca_q2_signed_gain_prefix",
                )
            )
        arms.append(
            (
                "shared_exact_sentinel",
                v1._D_RANK,
                tail_rows.clone().contiguous(),
                _endpoint_prediction(
                    trace.endpoint,
                    signed_fit,
                    rank=v1._D_RANK,
                    tail_rows=tail_rows,
                ),
                signed_fit.artifact_sha256,
                "ungained_exact_complement_sentinel",
            )
        )

        for method, rank, tail_prefix, prediction, fit_sha256, arm_kind in arms:
            key = (method, rank)
            correction_rows = (supported_rows + tail_prefix).contiguous()
            correction = torch.zeros(
                trace.base_h4.shape, dtype=torch.float64, device="cpu"
            )
            correction[0].index_copy_(
                0, trace.support_indices, correction_rows
            )
            provider = _AuthenticatedSignedJointFiniteProvider(
                rank=rank,
                arm_kind=arm_kind,
                fold_artifact_sha256=fit_sha256,
                model_inputs_sha256=trace.model_inputs_sha256,
                bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
                prefix_artifact_sha256=trace.prefix.artifact_sha256,
                base_h4=trace.base_h4,
                support_mask=support,
                correction=correction,
            )
            execution = bridge.execute(adapter, model_inputs, h4_head=provider)
            forward_by_key[key] += 1
            provider.validate_integrity()
            if (
                getattr(execution, "model_forward_count", None) != 1
                or not provider.used
                or execution.model_inputs_sha256 != trace.model_inputs_sha256
                or execution.bridge_binding_sha256
                != trace.prefix.bridge_binding_sha256
                or execution.prefix.artifact_sha256 != trace.prefix.artifact_sha256
                or execution.h4_head_sha256 != provider.artifact_sha256
                or _runtime_tensor_sha256(execution.candidate_x4)
                != trace.base_x4_sha256
            ):
                raise RuntimeError("signed-joint finite execution binding differs")
            expected_h4 = trace.base_h4.clone()
            support_on_live = trace.support_indices.to(expected_h4.device)
            expected_h4[0].index_copy_(
                0,
                support_on_live,
                (
                    trace.base_h4[0]
                    .index_select(0, support_on_live)
                    .to(dtype=torch.float64)
                    + correction_rows.to(trace.base_h4.device)
                ).to(dtype=trace.base_h4.dtype),
            )
            if not v1._bitwise_equal(execution.candidate_h4.detach(), expected_h4):
                raise RuntimeError("signed-joint finite H4 differs from bound correction")

            candidate_nll = v1._selected_token_nll(
                execution.logits, indices, targets
            )
            endpoint_selection = trace.selected_by_ledger["complete_h4_support"]
            endpoint_indices = indices.index_select(
                0, endpoint_selection.to(indices.device)
            )
            endpoint_targets = targets.index_select(
                0, endpoint_selection.to(targets.device)
            )
            candidate_endpoint_nll = v1._selected_token_nll(
                execution.logits, endpoint_indices, endpoint_targets
            )
            candidate_endpoint_kl = _selected_token_teacher_kl(
                source_logits, execution.logits, endpoint_indices
            )
            candidate_selected = frozen._select_sequence_rows(
                execution.logits, indices
            )
            for ledger, selected in trace.selected_by_ledger.items():
                if selected.numel() == 0:
                    continue
                fidelity[key][ledger].add(
                    ShadowFidelityExample(
                        example_id=trace.example_id,
                        family_id=trace.family_id,
                        source_logits=source_selected.index_select(
                            0, selected.to(source_selected.device)
                        ),
                        candidate_logits=candidate_selected.index_select(
                            0, selected.to(candidate_selected.device)
                        ),
                        targets=targets.index_select(
                            0, selected.to(targets.device)
                        ),
                    )
                )
            actual_rows = (
                execution.candidate_h4.detach().to(
                    device="cpu", dtype=torch.float64
                )[0].index_select(0, trace.support_indices)
                - trace.base_h4.to(device="cpu", dtype=torch.float64)[0].index_select(
                    0, trace.support_indices
                )
            ).contiguous()
            executed_rows[key][trace.example_id] = actual_rows
            target = trace.endpoint.compensation_target
            reconstruction_error = (
                float((tail_prefix - tail_rows).abs().max())
                if rank == v1._D_RANK
                else None
            )
            observation: dict[str, object] = {
                "example_id": trace.example_id,
                "family_id": trace.family_id,
                "method": method,
                "rank": rank,
                "requested_rank": rank,
                "rank_semantics": "at_most_direction_budget",
                "arm_kind": arm_kind,
                "effective_direction_count": (
                    min(rank, signed_fit.rank)
                    if method == "signed_joint"
                    else (rank if method == "fixed_pca_diagonal" else v1._D_RANK)
                ),
                "fold_artifact_sha256": fit_sha256,
                "provider_artifact_sha256": provider.artifact_sha256,
                "execution_artifact_sha256": execution.artifact_sha256,
                "native_mean_nll": float(trace.native_token_nll.mean()),
                "d320_mean_nll": float(trace.d320_token_nll.mean()),
                "candidate_mean_nll": float(candidate_endpoint_nll.mean()),
                "ordinary_candidate_mean_nll": float(candidate_nll.mean()),
                "d320_mean_teacher_kl": float(
                    (-trace.endpoint.compensation_target).mean()
                ),
                "candidate_mean_teacher_kl": float(candidate_endpoint_kl.mean()),
                "endpoint_baseline_mse": float(target.square().mean()),
                "endpoint_prediction_mse": float(
                    (prediction - target).square().mean()
                ),
                "candidate_h4_bitwise_native": v1._bitwise_equal(
                    execution.candidate_h4.detach().to(device="cpu"), trace.native_h4
                ),
                "candidate_logits_bitwise_native": (
                    _runtime_tensor_sha256(execution.logits)
                    == trace.native_logits_sha256
                ),
                "full_tail_reconstruction_max_abs_error": reconstruction_error,
                "exact_residual_provider_substitution_used": False,
                "executed_correction_rows_sha256": _runtime_tensor_sha256(actual_rows),
            }
            observation["observation_sha256"] = v1._domain_sha256(
                observation, domain=_OBSERVATION_DOMAIN
            )
            observations.append(observation)
            del execution, provider, correction, candidate_selected

        del model_inputs, source_logits, source_selected, native_h4

    behavioral_flat = {
        key: {ledger: fidelity[key][ledger].finalize() for ledger in ledgers}
        for key in keys
    }
    geometry_flat = {
        key: v1.ladder._geometry_with_examples(
            geometry_traces,
            executed_rows[key],
            candidate_semantics=(
                f"actual_cast_once_d320_plus_{key[0]}_teacher_kl_tail_k{key[1]}"
            ),
        )
        for key in keys
    }
    behavioral: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    geometry: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for key in keys:
        behavioral[key[0]][key[1]] = behavioral_flat[key]
        geometry[key[0]][key[1]] = geometry_flat[key]
    finite_resources = {
        "finite_native_forward_count": finite_native_forward_count,
        "finite_candidate_forward_count": sum(forward_by_key.values()),
        "finite_signed_joint_forward_count": sum(
            count for (method, _rank), count in forward_by_key.items()
            if method == "signed_joint"
        ),
        "finite_fixed_pca_diagonal_forward_count": sum(
            count for (method, _rank), count in forward_by_key.items()
            if method == "fixed_pca_diagonal"
        ),
        "finite_shared_exact_sentinel_forward_count": forward_by_key[
            ("shared_exact_sentinel", v1._D_RANK)
        ],
    }
    return observations, finite_resources, dict(behavioral), dict(geometry)


def _finite_observation_set_sha256(
    observations: Sequence[Mapping[str, object]],
    *,
    expected_example_count: int = v1._EXPECTED_EXAMPLES,
) -> str:
    expected_methods = {
        "signed_joint": set(SIGNED_JOINT_RANKS[:-1]),
        "fixed_pca_diagonal": set(SIGNED_JOINT_RANKS[:-1]),
        "shared_exact_sentinel": {v1._D_RANK},
    }
    expected_count = expected_example_count * sum(
        len(ranks) for ranks in expected_methods.values()
    )
    if len(observations) != expected_count:
        raise ValueError("signed-joint finite observation count differs")
    identities: set[tuple[str, str, int]] = set()
    receipts: dict[tuple[str, str, int], str] = {}
    examples: set[str] = set()
    family_by_example: dict[str, str] = {}
    expected_arm_kind = {
        "signed_joint": "teacher_kl_signed_joint_rotated_gain_prefix",
        "fixed_pca_diagonal": "teacher_kl_fixed_pca_q2_signed_gain_prefix",
        "shared_exact_sentinel": "ungained_exact_complement_sentinel",
    }
    for raw in observations:
        row = dict(raw)
        receipt = row.pop("observation_sha256", None)
        example_id = v1._identifier(
            row.get("example_id"), label="signed-joint observation example"
        )
        method = row.get("method")
        rank = row.get("rank")
        family_id = v1._identifier(
            row.get("family_id"), label="signed-joint observation family"
        )
        if (
            not isinstance(method, str)
            or method not in expected_methods
            or type(rank) is not int
            or rank not in expected_methods[method]
            or row.get("requested_rank") != rank
            or row.get("rank_semantics") != "at_most_direction_budget"
            or type(row.get("effective_direction_count")) is not int
            or not 0 <= int(row["effective_direction_count"]) <= rank
            or (
                method != "signed_joint"
                and int(row["effective_direction_count"]) != rank
            )
            or row.get("arm_kind") != expected_arm_kind.get(str(method))
        ):
            raise ValueError("signed-joint observation method/rank differs")
        prior_family = family_by_example.setdefault(example_id, family_id)
        if prior_family != family_id:
            raise ValueError("signed-joint observation family differs across methods")
        identity = (example_id, method, rank)
        if identity in identities:
            raise ValueError("signed-joint observation grid has a duplicate")
        expected = v1._domain_sha256(row, domain=_OBSERVATION_DOMAIN)
        if receipt != expected:
            raise RuntimeError("signed-joint finite observation receipt drifted")
        identities.add(identity)
        receipts[identity] = expected
        examples.add(example_id)
    if len(examples) != expected_example_count:
        raise ValueError("signed-joint observation example grid is incomplete")
    for example_id in examples:
        for method, ranks in expected_methods.items():
            if any((example_id, method, rank) not in identities for rank in ranks):
                raise ValueError("signed-joint observation method grid is incomplete")
    return v1._domain_sha256(
        tuple((*identity, receipts[identity]) for identity in sorted(receipts)),
        domain=_OBSERVATION_SET_DOMAIN,
    )


def _summarize_dual_arm_observations(
    observations: Sequence[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, bool]]]:
    sentinel = tuple(
        row for row in observations if row["method"] == "shared_exact_sentinel"
    )
    arms_by_method: dict[str, list[dict[str, object]]] = {}
    gates_by_method: dict[str, dict[str, bool]] = {}
    for method in ("signed_joint", "fixed_pca_diagonal"):
        selected = [row for row in observations if row["method"] == method]
        selected.extend(sentinel)
        arms, gates = v1._summarize_observations(
            selected, ranks=SIGNED_JOINT_RANKS
        )
        arms_by_method[method] = arms
        gates_by_method[method] = gates
    return arms_by_method, gates_by_method


def _paired_method_comparison(
    *,
    arms_by_method: Mapping[str, Sequence[Mapping[str, object]]],
    behavioral: Mapping[str, Mapping[int, Mapping[str, object]]],
    geometry: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> tuple[dict[str, object], ...]:
    signed_arms = {
        int(row["tail_rank"]): row for row in arms_by_method["signed_joint"]
    }
    control_arms = {
        int(row["tail_rank"]): row
        for row in arms_by_method["fixed_pca_diagonal"]
    }
    rows: list[dict[str, object]] = []
    for rank in SIGNED_JOINT_RANKS[:-1]:
        signed = signed_arms[rank]
        control = control_arms[rank]
        behavior_deltas: dict[str, object] = {}
        for ledger in (
            "ordinary",
            "complete_h4_support",
            "graph_core",
            "causal_tail",
        ):
            signed_report = _mapping(
                behavioral["signed_joint"][rank][ledger],
                label="signed behavior report",
            )
            control_report = _mapping(
                behavioral["fixed_pca_diagonal"][rank][ledger],
                label="control behavior report",
            )
            signed_aggregate = _mapping(
                signed_report.get("aggregate"), label="signed behavior aggregate"
            )
            control_aggregate = _mapping(
                control_report.get("aggregate"), label="control behavior aggregate"
            )
            signed_per_prompt = _mapping(
                signed_report.get("per_prompt"), label="signed behavior per-prompt"
            )
            control_per_prompt = _mapping(
                control_report.get("per_prompt"), label="control behavior per-prompt"
            )
            signed_p90 = _mapping(
                signed_per_prompt.get("absolute_delta_nll_per_token"),
                label="signed behavior p90",
            )
            control_p90 = _mapping(
                control_per_prompt.get("absolute_delta_nll_per_token"),
                label="control behavior p90",
            )
            behavior_deltas[ledger] = {
                "signed_minus_control_raw_delta_nll_per_token": (
                    float(signed_aggregate["delta_nll_per_token"])
                    - float(control_aggregate["delta_nll_per_token"])
                ),
                "signed_minus_control_absolute_delta_nll_per_token": (
                    abs(float(signed_aggregate["delta_nll_per_token"]))
                    - abs(float(control_aggregate["delta_nll_per_token"]))
                ),
                "signed_minus_control_prompt_p90_absolute_delta_nll_per_token": (
                    float(signed_p90["p90"]) - float(control_p90["p90"])
                ),
                "signed_minus_control_source_to_candidate_kl_per_token": (
                    float(signed_aggregate["source_to_candidate_kl_per_token"])
                    - float(control_aggregate["source_to_candidate_kl_per_token"])
                ),
                "signed_minus_control_top1_agreement": (
                    float(signed_aggregate["top1_agreement_to_source"])
                    - float(control_aggregate["top1_agreement_to_source"])
                ),
            }
        geometry_deltas: dict[str, object] = {}
        for stratum in ("full", "graph_core", "causal_tail"):
            signed_pooled = _mapping(
                _mapping(
                    geometry["signed_joint"][rank].get("pooled"),
                    label="signed pooled geometry",
                ).get(stratum),
                label="signed geometry stratum",
            )
            control_pooled = _mapping(
                _mapping(
                    geometry["fixed_pca_diagonal"][rank].get("pooled"),
                    label="control pooled geometry",
                ).get(stratum),
                label="control geometry stratum",
            )
            geometry_deltas[stratum] = {
                "signed_minus_control_cosine": (
                    float(signed_pooled["cosine"])
                    - float(control_pooled["cosine"])
                ),
                "signed_minus_control_normalized_rmse": (
                    float(signed_pooled["normalized_rmse"])
                    - float(control_pooled["normalized_rmse"])
                ),
            }
        row: dict[str, object] = {
            "rank": rank,
            "signed_minus_control_endpoint_rmse_after": (
                float(signed["family_macro_endpoint_rmse_after"])
                - float(control["family_macro_endpoint_rmse_after"])
            ),
            "signed_minus_control_absolute_nll_gap_after": (
                float(signed["family_macro_absolute_nll_gap_after"])
                - float(control["family_macro_absolute_nll_gap_after"])
            ),
            "behavioral_deltas": behavior_deltas,
            "geometry_deltas": geometry_deltas,
            "negative_absolute_error_or_kl_delta_favors_signed_joint": True,
            "raw_signed_delta_nll_difference_has_no_uniform_favor_direction": True,
            "positive_top1_or_cosine_delta_favors_signed_joint": True,
        }
        row["comparison_sha256"] = v1._domain_sha256(
            row, domain=_CONTROL_COMPARISON_DOMAIN
        )
        rows.append(row)
    return tuple(rows)


def _build_resource_accounting(
    traces: Sequence[_LiveTeacherKLTrace],
    *,
    endpoint_resources: Mapping[str, int],
    finite_resources: Mapping[str, int],
    signed_fits: Mapping[str, CompleteH4TailSignedJointHeldFamilyFit],
) -> dict[str, object]:
    expected_backward = sum(
        (trace.endpoint.supervised_tokens + v1._VJP_CHUNK_SIZE - 1)
        // v1._VJP_CHUNK_SIZE
        for trace in traces
    )
    if (
        endpoint_resources["endpoint_teacher_kl_vjp_backward_call_count"]
        != expected_backward
    ):
        raise RuntimeError("teacher-KL VJP backward accounting differs")
    support_rows = sum(trace.endpoint.residual_rows.shape[0] for trace in traces)
    by_family: dict[str, tuple[_LiveTeacherKLTrace, ...]] = {
        family: tuple(trace for trace in traces if trace.family_id != family)
        for family in sorted(signed_fits)
    }
    c = v1._D_RANK
    w = v1._WIDTH
    signed_coordinate_transform_macs = sum(
        trace.endpoint.residual_rows.shape[0] * w * c
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * w
        * c
        for training in by_family.values()
        for trace in training
    )
    signed_operator_and_score_macs = 0
    signed_projector_dense_macs = 0
    signed_eigh_calls = 0
    for family, fit in signed_fits.items():
        # A stopped fit evaluates one final non-selected operator/eigensystem;
        # a requested-rank fit evaluates exactly one per retained direction.
        attempts = fit.rank + (
            0 if fit.stop_reason == "requested_rank_reached" else 1
        )
        signed_eigh_calls += attempts
        for step in range(attempts):
            for trace in by_family[family]:
                rows = trace.endpoint.residual_rows.shape[0]
                tokens = trace.endpoint.supervised_tokens
                signed_operator_and_score_macs += (
                    tokens * rows * c  # rho-weighted gradient reduction
                    + c * rows * c  # A.T @ weighted G
                    + rows * c  # amplitude @ direction
                    + tokens * rows * c  # gradient @ direction
                    + tokens * rows  # amplitude-gradient reduction
                )
            if step:
                signed_projector_dense_macs += (
                    c * step * c  # prior.T @ prior
                    + 2 * c * c * c  # P @ M @ P
                )
    pca_full_complement_projection_and_covariance_macs = sum(
        3 * trace.endpoint.residual_rows.shape[0] * c * w
        + trace.endpoint.residual_rows.shape[0] * c * c
        for training in by_family.values()
        for trace in training
    )
    pca_q2_score_macs = sum(
        trace.endpoint.residual_rows.shape[0] * c * w
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * c
        * w
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * c
        for training in by_family.values()
        for trace in training
    )
    pca_matched_gain_rescore_macs = sum(
        trace.endpoint.residual_rows.shape[0] * _SIGNED_MAX_RANK * w
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * _SIGNED_MAX_RANK
        * w
        + trace.endpoint.supervised_tokens
        * trace.endpoint.residual_rows.shape[0]
        * _SIGNED_MAX_RANK
        for training in by_family.values()
        for trace in training
    )
    bounded_prefix_macs_by_method_rank = {
        "signed_joint": {
            str(rank): sum(
                2
                * trace.endpoint.residual_rows.shape[0]
                * min(rank, signed_fits[trace.family_id].rank)
                * v1._WIDTH
                + trace.endpoint.residual_rows.shape[0]
                * min(rank, signed_fits[trace.family_id].rank)
                for trace in traces
            )
            for rank in SIGNED_JOINT_RANKS[:-1]
        },
        "fixed_pca_diagonal": {
            str(rank): 2 * support_rows * rank * v1._WIDTH + support_rows * rank
            for rank in SIGNED_JOINT_RANKS[:-1]
        },
    }
    return {
        **endpoint_resources,
        **finite_resources,
        "vjp_chunk_size": v1._VJP_CHUNK_SIZE,
        "expected_backward_call_count_from_support_tokens": expected_backward,
        "total_model_forward_count": (
            endpoint_resources["base_forward_count"]
            + endpoint_resources["native_teacher_forward_count"]
            + endpoint_resources["endpoint_teacher_kl_vjp_forward_count"]
            + finite_resources["finite_native_forward_count"]
            + finite_resources["finite_candidate_forward_count"]
        ),
        "complete_h4_support_row_count": support_rows,
        "peak_simultaneously_retained_full_sequence_vocabulary_tensor_count": 4,
        "peak_full_vocabulary_residency_reason": (
            "base_logits_plus_native_teacher_logits_plus_detached_teacher_snapshot_plus_teacher_KL_VJP_candidate_logits"
        ),
        "additional_selected_token_by_vocabulary_matrices": (
            "teacher_and_candidate_log_probability_matrices_plus_KL_temporaries"
        ),
        "bounded_prefix_projection_logical_macs_by_method_rank": (
            bounded_prefix_macs_by_method_rank
        ),
        "shared_exact_sentinel_tail_copy_scalar_count": (
            support_rows * v1._WIDTH
        ),
        "endpoint_d320_provider_plus_independent_reconstruction_logical_macs": (
            4 * support_rows * c * w
        ),
        "finite_supported_and_complement_projection_logical_macs": (
            4 * support_rows * c * w
        ),
        "finite_candidate_correction_and_independent_check_scalar_adds": (
            27 * support_rows * w
        ),
        "endpoint_native_minus_base_residual_scalar_subtractions": support_rows * w,
        "finite_tail_and_nine_arm_actual_row_scalar_subtractions": (
            10 * support_rows * w
        ),
        "signed_joint_fold_count": v1._EXPECTED_FAMILIES,
        "signed_joint_steps_per_fold": _SIGNED_MAX_RANK,
        "signed_joint_actual_retained_directions_by_fold": {
            family: fit.rank for family, fit in sorted(signed_fits.items())
        },
        "signed_joint_streamed_coordinate_transform_logical_macs": (
            signed_coordinate_transform_macs
        ),
        "signed_joint_streamed_operator_and_direction_score_logical_macs": (
            signed_operator_and_score_macs
        ),
        "signed_joint_dense_deflation_projector_logical_macs": (
            signed_projector_dense_macs
        ),
        "signed_joint_symmetric_eigh_320_by_320_call_count": signed_eigh_calls,
        "fixed_pca_diagonal_fold_count": v1._EXPECTED_FAMILIES,
        "fixed_pca_signed_gain_steps_per_fold": _SIGNED_MAX_RANK,
        "fixed_pca_full_complement_projection_and_covariance_logical_macs": (
            pca_full_complement_projection_and_covariance_macs
        ),
        "fixed_pca_q2_score_logical_macs": pca_q2_score_macs,
        "fixed_pca_matched_gain_rescore_logical_macs": (
            pca_matched_gain_rescore_macs
        ),
        "fixed_pca_symmetric_eigh_320_by_320_call_count": v1._EXPECTED_FAMILIES,
        "fixed_pca_ambient_basis_lift_logical_macs": (
            v1._EXPECTED_FAMILIES * c * c * w
        ),
        "canonical_complement_two_pass_mgs_construction_count": (
            2 * v1._EXPECTED_FAMILIES
        ),
        "full_tail_reconstruction_and_integrity_check_logical_macs": (
            14 * support_rows * c * w
        ),
        "signed_joint_operator_implementation": (
            "streamed_prompt_A_transpose_times_mean_token_rho_G_no_T_by_C_by_C_tensor"
        ),
        "symmetric_eigensolver_internal_flops": (
            "executed_analysis_work_not_estimated_as_logical_macs; exact call_counts_and_shapes_reported"
        ),
        "serving_learned_parameter_count": "not_applicable_no_serving_artifact",
        "serving_logical_macs_per_token": "not_applicable_no_serving_artifact",
    }


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(
            report, domain=_REPORT_DOMAIN
        )
        stage = frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": v1._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic(
    *,
    control_report_path: Path | str = DEFAULT_CONTROL_REPORT,
    expanded_control_report_path: Path | str = DEFAULT_EXPANDED_CONTROL_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the pinned same-A teacher-KL signed-joint dual-arm screen."""

    destination = v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite teacher-KL signed-joint report")
    historical_control = _load_control_report(control_report_path)
    expanded_control = _load_expanded_control_report(expanded_control_report_path)
    materialization = v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=v1.MATERIALIZATION_REPORT_SHA256,
        label="rank320 materialization",
    )
    transfer = v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=v1.TRANSFER_REPORT_SHA256,
        label="rank320 transfer",
    )
    transfer_receipts = _transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=v1.MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, endpoint_resources = _collect_teacher_kl_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if len(traces) != v1._EXPECTED_EXAMPLES or len(families) != v1._EXPECTED_FAMILIES:
            raise RuntimeError("A16 teacher-KL signed-joint panel shape differs")
        endpoints = tuple(trace.endpoint for trace in traces)
        signed_fits = {
            family: fit_complete_h4_tail_signed_joint_held_family(
                endpoints,
                supported_basis=basis,
                held_family_id=family,
                max_directions=_SIGNED_MAX_RANK,
            )
            for family in families
        }
        pca_control_fits = {
            family: _fit_fixed_pca_signed_control(
                endpoints,
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        observations, finite_resources, behavioral, geometry = (
            _finite_dual_arm_observations(
                context=context,
                traces=traces,
                basis=basis,
                signed_fits=signed_fits,
                pca_control_fits=pca_control_fits,
            )
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    observation_set_sha256 = _finite_observation_set_sha256(observations)
    arms_by_method, secondary_gates_by_method = _summarize_dual_arm_observations(
        observations
    )
    paired_comparison = _paired_method_comparison(
        arms_by_method=arms_by_method,
        behavioral=behavioral,
        geometry=geometry,
    )
    causality_passed = all(
        trace.maximum_future_gradient_abs == 0.0
        and trace.future_gradient_nonzero_count == 0
        for trace in traces
    )
    direction_availability_by_rank = {
        rank: all(fit.rank >= rank for fit in signed_fits.values())
        for rank in SIGNED_JOINT_RANKS[:-1]
    }
    fidelity_and_geometry_pass: dict[str, dict[int, bool]] = {
        "signed_joint": {},
        "fixed_pca_diagonal": {},
        "shared_exact_sentinel": {},
    }
    for method in ("signed_joint", "fixed_pca_diagonal"):
        for rank in SIGNED_JOINT_RANKS[:-1]:
            fidelity_and_geometry_pass[method][rank] = (
                bool(geometry[method][rank]["gates"]["passed"])
                and all(
                    bool(behavioral[method][rank][ledger]["gates"]["passed"])
                    for ledger in (
                        "ordinary",
                        "complete_h4_support",
                        "graph_core",
                        "causal_tail",
                    )
                )
            )
    sentinel_passed = (
        bool(
            geometry["shared_exact_sentinel"][v1._D_RANK]["gates"]["passed"]
        )
        and all(
            bool(
                behavioral["shared_exact_sentinel"][v1._D_RANK][ledger]["gates"][
                    "passed"
                ]
            )
            for ledger in (
                "ordinary",
                "complete_h4_support",
                "graph_core",
                "causal_tail",
            )
        )
    )
    fidelity_and_geometry_pass["shared_exact_sentinel"][v1._D_RANK] = sentinel_passed
    signed_passing_ranks = tuple(
        rank
        for rank in SIGNED_JOINT_RANKS[:-1]
        if direction_availability_by_rank[rank]
        and fidelity_and_geometry_pass["signed_joint"][rank]
    )
    pca_passing_ranks = tuple(
        rank
        for rank in SIGNED_JOINT_RANKS[:-1]
        if fidelity_and_geometry_pass["fixed_pca_diagonal"][rank]
    )
    smallest_signed_rank = (
        None if not signed_passing_ranks else min(signed_passing_ranks)
    )
    smallest_pca_rank = None if not pca_passing_ranks else min(pca_passing_ranks)
    if smallest_signed_rank is None:
        matched_algorithm_classification = "signed_joint_no_bounded_established_gate_pass"
    elif smallest_pca_rank is None or smallest_signed_rank < smallest_pca_rank:
        matched_algorithm_classification = (
            "signed_joint_lower_rank_established_gate_advantage_over_matched_PCA_q2"
        )
    elif smallest_signed_rank == smallest_pca_rank:
        matched_algorithm_classification = (
            "signed_joint_and_matched_PCA_q2_first_pass_at_same_rank"
        )
    else:
        matched_algorithm_classification = (
            "matched_PCA_q2_lower_rank_established_gate_advantage"
        )
    sentinel_arm = next(
        row
        for row in arms_by_method["signed_joint"]
        if row["tail_rank"] == v1._D_RANK
    )
    primary_gates = {
        "all_teacher_kl_vjps_have_zero_future_gradient": causality_passed,
        "shared_k320_full_tail_reconstruction_at_most_1e_minus_9": (
            float(sentinel_arm["maximum_full_tail_reconstruction_abs_error"])
            <= 1.0e-9
        ),
        "shared_k320_every_prompt_h4_bitwise_native": bool(
            sentinel_arm["every_prompt_h4_bitwise_native"]
        ),
        "shared_k320_every_prompt_logits_bitwise_native": bool(
            sentinel_arm["every_prompt_logits_bitwise_native"]
        ),
        "shared_k320_clears_established_fidelity_and_geometry_gates": sentinel_passed,
        "at_least_one_available_signed_joint_k_le_64_clears_all_established_gates": (
            bool(signed_passing_ranks)
        ),
    }
    resources = _build_resource_accounting(
        traces,
        endpoint_resources=endpoint_resources,
        finite_resources=finite_resources,
        signed_fits=signed_fits,
    )
    prompt_receipts = tuple(
        {
            **trace.endpoint.metadata(),
            "model_inputs_sha256": trace.model_inputs_sha256,
            "base_x4_sha256": trace.base_x4_sha256,
            "supervised_indices_sha256": trace.supervised_indices_sha256,
            "supervised_targets_sha256": trace.supervised_targets_sha256,
            "endpoint_support_indices_sha256": trace.endpoint_indices_sha256,
            "endpoint_support_targets_sha256": trace.endpoint_targets_sha256,
            "endpoint_support_supervised_token_count": trace.endpoint.supervised_tokens,
            "native_logits_sha256": trace.native_logits_sha256,
            "teacher_logits_sha256": trace.teacher_logits_sha256,
            "teacher_kl_vjp_artifact_sha256": trace.endpoint_vjp_artifact_sha256,
            "endpoint_execution_artifact_sha256": (
                trace.endpoint_execution_artifact_sha256
            ),
            "endpoint_provider_artifact_sha256": (
                trace.endpoint_provider_artifact_sha256
            ),
            "backward_call_count": trace.backward_call_count,
            "compensation_target_semantics": "negative_KL_native_teacher_to_D320_candidate",
            "maximum_future_gradient_abs": trace.maximum_future_gradient_abs,
            "future_gradient_nonzero_count": trace.future_gradient_nonzero_count,
            "causality_receipt_sha256": trace.causality_receipt_sha256,
        }
        for trace in traces
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_hypothesis_use_only",
            "split": "whole_family_leave_one_out_for_every_direction_order_and_gain",
            "frozen_d320_was_fit_on_all_a16_families": True,
            "end_to_end_candidate_is_family_disjoint": False,
            "frozen_supported_basis_rank": v1._D_RANK,
            "tail_width": v1._D_RANK,
            "requested_tail_ranks": SIGNED_JOINT_RANKS,
            "bounded_rank_semantics": (
                "at_most_direction_budget_with_per_observation_effective_direction_count"
            ),
            "teacher_objective": "token_KL(native_teacher||D320_candidate)",
            "signed_compensation_target": "negative_endpoint_teacher_KL",
            "endpoint_token_grid": "canonical_batch_major_complete_h4_support_803",
            "signed_joint_order": (
                "training_only_equal_family_prompt_token_signed_joint_greedy_rotation"
            ),
            "signed_joint_gain_constraint": "closed_interval_0_2",
            "same_objective_control": (
                "training_only_residual_PCA_then_teacher_KL_q2_order_with_matched_signed_gains"
            ),
            "finite_signed_arm": "D320_plus_gain_scaled_signed_joint_tail_prefix",
            "finite_control_arm": "D320_plus_gain_scaled_fixed_PCA_q2_tail_prefix",
            "shared_k320_arm": "D320_plus_ungained_exact_orthogonal_complement_tail",
            "shared_k320_exact_residual_provider_substitution": False,
            "finite_shadow_ledgers": {
                "ordinary": v1._EXPECTED_ORDINARY_TOKENS,
                "complete_h4_support": v1._EXPECTED_SUPPORT_TOKENS,
                "graph_core": v1._EXPECTED_GRAPH_CORE_TOKENS,
                "causal_tail": v1._EXPECTED_CAUSAL_TAIL_TOKENS,
            },
        },
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": v1.MATERIALIZATION_REPORT_FILE_SHA256,
            "materialization_report_sha256": v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": v1.TRANSFER_REPORT_SHA256,
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
            "historical_v1_control_file": str(control_report_path),
            "historical_v1_control_file_sha256": CONTROL_REPORT_FILE_SHA256,
            "historical_v1_control_report_sha256": CONTROL_REPORT_SHA256,
            "historical_expanded_control_file": str(expanded_control_report_path),
            "historical_expanded_control_file_sha256": (
                EXPANDED_CONTROL_REPORT_FILE_SHA256
            ),
            "historical_expanded_control_report_sha256": (
                EXPANDED_CONTROL_REPORT_SHA256
            ),
        },
        "signed_joint_folds": tuple(
            signed_fits[family].metadata() for family in families
        ),
        "fixed_pca_same_teacher_kl_control_folds": tuple(
            pca_control_fits[family].metadata() for family in families
        ),
        "signed_joint_direction_availability_by_requested_rank": {
            str(rank): direction_availability_by_rank[rank]
            for rank in SIGNED_JOINT_RANKS[:-1]
        },
        "prompt_receipts": prompt_receipts,
        "finite_ladder_by_method": arms_by_method,
        "established_behavioral_fidelity_by_method_rank": {
            method: {str(rank): value for rank, value in ranks.items()}
            for method, ranks in behavioral.items()
        },
        "executed_cast_once_geometry_by_method_rank": {
            method: {str(rank): value for rank, value in ranks.items()}
            for method, ranks in geometry.items()
        },
        "fidelity_and_geometry_pass_by_method_rank": {
            method: {str(rank): value for rank, value in ranks.items()}
            for method, ranks in fidelity_and_geometry_pass.items()
        },
        "smallest_signed_joint_rank_clearing_established_gates": (
            smallest_signed_rank
        ),
        "smallest_fixed_pca_control_rank_clearing_established_gates": (
            smallest_pca_rank
        ),
        "matched_teacher_kl_algorithm_comparison_classification": (
            matched_algorithm_classification
        ),
        "same_teacher_kl_paired_method_comparison": paired_comparison,
        "historical_mixed_objective_references": {
            "attribution_warning": (
                "historical references change both objective and basis; they cannot attribute a win to rotation"
            ),
            "v1": {
                "schema": historical_control.get("schema"),
                "classification": historical_control.get("classification"),
                "report_sha256": CONTROL_REPORT_SHA256,
                "finite_ladder": historical_control.get("finite_ladder"),
            },
            "adaptive_expanded": {
                "schema": expanded_control.get("schema"),
                "classification": expanded_control.get("classification"),
                "report_sha256": EXPANDED_CONTROL_REPORT_SHA256,
                "finite_ladder": expanded_control.get("finite_ladder"),
            },
        },
        "finite_observation_receipts": tuple(observations),
        "finite_observation_set_sha256": observation_set_sha256,
        "primary_gate_results": tuple(sorted(primary_gates.items())),
        "secondary_first_order_gate_results_by_method": secondary_gates_by_method,
        "passed": all(primary_gates.values()),
        "classification": (
            "same_a_teacher_kl_signed_joint_bounded_fidelity_supported"
            if all(primary_gates.values())
            else "same_a_teacher_kl_signed_joint_bounded_fidelity_not_supported"
        ),
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "tail_direction_order_and_gain_whole_family_disjoint": True,
            "frozen_d320_contains_same_a_held_family_information": True,
            "held_native_tail_used_to_instantiate_finite_correction": True,
            "end_to_end_candidate_family_disjoint": False,
            "fresh_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "matched_objective_joint_algorithm_comparison_available": True,
            "rotation_only_attribution_authorized": False,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_activation_tensors": False,
            "contains_gradient_tensors": False,
            "contains_direction_or_basis_tensors": False,
            "contains_token_score_matrices": False,
            "contains_only_hashes_counts_and_scalar_metrics": True,
            "artifact_must_remain_outside_git": True,
        },
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run A16 teacher-KL signed-joint and matched PCA finite ladders."
    )
    parser.add_argument("--control-report", type=Path, default=DEFAULT_CONTROL_REPORT)
    parser.add_argument(
        "--expanded-control-report",
        type=Path,
        default=DEFAULT_EXPANDED_CONTROL_REPORT,
    )
    parser.add_argument(
        "--materialization-report", type=Path, default=DEFAULT_MATERIALIZATION_REPORT
    )
    parser.add_argument("--transfer-report", type=Path, default=DEFAULT_TRANSFER_REPORT)
    parser.add_argument("--basis-sidecar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic(
        control_report_path=args.control_report,
        expanded_control_report_path=args.expanded_control_report,
        materialization_report_path=args.materialization_report,
        transfer_report_path=args.transfer_report,
        basis_sidecar_path=args.basis_sidecar,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(f"wrote {result['artifact']['file']}")  # type: ignore[index]


if __name__ == "__main__":
    main()
