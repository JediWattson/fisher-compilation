"""Calibration-A worker for progressive Gemma L3/L4 compilation.

This module is the model-specific data plane behind
``compiler.progressive``.  It deliberately keeps the current rank-64
five-pass runtime in its honest role: an evaluator and residual probe, never
a serving executor.  The worker:

* binds three pairwise-disjoint Calibration-A panels;
* evaluates source-authoritative candidate, projection, and carrier metrics;
* maps the remaining residual at the complete ``layer.4.output`` boundary;
* ranks NLL-VJP/activation-gradient-Gram-aware residual directions; and
* exposes strict callbacks for the generic progressive controller.

Mutation lowering is a separate capability.  A campaign may inspect the seed
and produce a residual map immediately, but proposal/build callbacks fail
closed until a candidate-bound lowerer is supplied.  This prevents the locked
legacy shadow runtime, its native fallbacks, or its multi-pass measurement
cost from being presented as a deployable compressed model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Literal, Protocol

import torch
from torch import Tensor

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .compiler.progressive import (
    CandidateEvaluation,
    DevelopmentEvaluationCoverage,
    FitDevelopmentView,
    FrozenCalibrationAChallenger,
    GuardDevelopmentView,
    MutationProposal,
    ProgressiveBehavioralFidelity,
    ProgressiveCandidate,
    ProgressiveCompilationProtocol,
    ProgressiveFidelity,
    ProgressivePhase,
    ResidualMap,
    ResidualTarget,
    SelectionDevelopmentView,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    derive_gemma3_l3_l4_supervised_boundary,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .shadow_fidelity import (
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


DevelopmentRole = Literal[
    "calibration_a_fit",
    "calibration_a_selection",
    "calibration_a_guard",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_H4_SITE = "layer.4.output"
_X4_SITE = "layer.4.mlp.normalized_input"
_HASH_DOMAIN = b"fisher-graph:gemma3-l3-l4-progressive-worker:v1\0"
LEGACY_RANK64_INCOMPLETE_COST_REASONS = (
    "multi_pass_shadow_measurement",
    "native_boundary_fallback",
    "no_one_pass_serving_executable",
)


class GemmaProgressiveWorkerError(RuntimeError):
    """Base class for fail-closed worker errors."""


class GemmaMutationLoweringUnavailableError(GemmaProgressiveWorkerError):
    """Raised when a campaign requests an unregistered executable mutation."""


class GemmaGuardAuthorityRequiredError(GemmaProgressiveWorkerError):
    """Raised when the final A guard lacks durable claim-first authority."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(kind: str, value: object) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be a portable nonempty identifier")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise TypeError("hash inputs must be materialized strided tensors")
    canonical = value.detach().to(device="cpu").contiguous()
    payload = {
        "dtype": str(canonical.dtype),
        "shape": tuple(int(width) for width in canonical.shape),
    }
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(b"tensor\0")
    digest.update(_canonical_json_bytes(payload))
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _calibration_example_sha256(batch: CalibrationBatch) -> str:
    return _sha256(
        "calibration-example",
        {
            "model_inputs_sha256": (
                gemma3_l3_l4_shadow_model_inputs_sha256(batch.model_inputs)
            ),
            "targets_sha256": _tensor_sha256(batch.targets),
            "valid_positions_sha256": _tensor_sha256(
                batch.valid_positions
            ),
            "shared_input_names": tuple(sorted(batch.shared_input_names)),
        },
    )


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _finite_scalar(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def gemma_progressive_panel_membership_receipt_sha256(
    *,
    role: DevelopmentRole,
    manifest_sha256: str,
    family_by_example: Mapping[str, str],
) -> str:
    """Commit exact panel membership without materializing token tensors."""

    if role not in (
        "calibration_a_fit",
        "calibration_a_selection",
        "calibration_a_guard",
    ):
        raise ValueError("unsupported development role")
    manifest = _require_sha256(
        manifest_sha256,
        label="manifest_sha256",
    )
    if not isinstance(family_by_example, Mapping) or not family_by_example:
        raise ValueError("family_by_example must be a nonempty mapping")
    members: list[tuple[str, str]] = []
    for example_id, family_id in family_by_example.items():
        members.append(
            (
                _require_identifier(example_id, label="example_id"),
                _require_identifier(family_id, label="family_id"),
            )
        )
    if len({example_id for example_id, _ in members}) != len(members):
        raise ValueError("panel example identities must be unique")
    return _sha256(
        "panel-membership",
        {
            "role": role,
            "manifest_sha256": manifest,
            "members": tuple(sorted(members)),
        },
    )


@dataclass(frozen=True, slots=True)
class GemmaProgressiveExample:
    """One family-bound single-example calibration batch."""

    example_id: str
    family_id: str
    batch: CalibrationBatch
    _model_inputs_sha256: str = field(init=False, repr=False)
    _calibration_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_identifier(self.example_id, label="example_id")
        _require_identifier(self.family_id, label="family_id")
        if not isinstance(self.batch, CalibrationBatch):
            raise TypeError("batch must be a CalibrationBatch")
        if self.batch.batch_size != 1:
            raise ValueError(
                "the strict Gemma shadow worker requires batch size one"
            )
        snapshot = CalibrationBatch(
            model_inputs={
                name: value.detach().clone()
                for name, value in self.batch.model_inputs.items()
            },
            targets=self.batch.targets.detach().clone(),
            valid_positions=self.batch.valid_positions.detach().clone(),
            shared_input_names=self.batch.shared_input_names,
            example_ids=self.batch.example_ids,
        )
        object.__setattr__(self, "batch", snapshot)
        if (
            self.batch.example_ids is not None
            and self.batch.example_ids != (self.example_id,)
        ):
            raise ValueError(
                "calibration batch example identity differs from the panel"
            )
        input_ids = self.batch.model_inputs.get("input_ids")
        if (
            not isinstance(input_ids, Tensor)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise ValueError(
                "Gemma progressive examples require integer [1, sequence] "
                "input_ids"
            )
        if (
            self.batch.targets.dtype not in (torch.int32, torch.int64)
            or self.batch.targets.shape != input_ids.shape
            or self.batch.valid_positions.shape != input_ids.shape
        ):
            raise ValueError(
                "Gemma progressive targets and valid positions must match "
                "input_ids, with integer targets"
            )
        object.__setattr__(
            self,
            "_model_inputs_sha256",
            gemma3_l3_l4_shadow_model_inputs_sha256(
                self.batch.model_inputs
            ),
        )
        object.__setattr__(
            self,
            "_calibration_sha256",
            _calibration_example_sha256(self.batch),
        )

    @property
    def model_inputs_sha256(self) -> str:
        return self._model_inputs_sha256

    @property
    def calibration_sha256(self) -> str:
        return self._calibration_sha256

    def validate_integrity(self) -> None:
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(
                self.batch.model_inputs
            )
            != self._model_inputs_sha256
            or _calibration_example_sha256(self.batch)
            != self._calibration_sha256
        ):
            raise ValueError(
                "materialized calibration example changed after binding"
            )


@dataclass(frozen=True, slots=True)
class GemmaProgressivePanel:
    """A materialized, raw-payload-private Calibration-A panel."""

    role: DevelopmentRole
    manifest_sha256: str
    examples: tuple[GemmaProgressiveExample, ...]
    forbidden_manifest_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in (
            "calibration_a_fit",
            "calibration_a_selection",
            "calibration_a_guard",
        ):
            raise ValueError("unsupported development role")
        manifest = _require_sha256(
            self.manifest_sha256,
            label="manifest_sha256",
        )
        if (
            type(self.examples) is not tuple
            or not self.examples
            or any(
                not isinstance(example, GemmaProgressiveExample)
                for example in self.examples
            )
        ):
            raise ValueError(
                "examples must be a nonempty tuple of Gemma examples"
            )
        forbidden = tuple(
            sorted(
                {
                    _require_sha256(value, label="forbidden manifest")
                    for value in self.forbidden_manifest_sha256s
                }
            )
        )
        if forbidden != self.forbidden_manifest_sha256s:
            raise ValueError(
                "forbidden manifests must be sorted and contain no duplicates"
            )
        if manifest in forbidden:
            raise ValueError(
                "a Calibration-A panel cannot use a forbidden assessment "
                "manifest"
            )
        example_ids = tuple(example.example_id for example in self.examples)
        input_ids = tuple(
            example.model_inputs_sha256 for example in self.examples
        )
        calibration_ids = tuple(
            example.calibration_sha256 for example in self.examples
        )
        if (
            len(set(example_ids)) != len(example_ids)
            or len(set(input_ids)) != len(input_ids)
            or len(set(calibration_ids)) != len(calibration_ids)
        ):
            raise ValueError(
                "panel examples, model inputs, and calibration rows must not "
                "repeat"
            )

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(sorted({example.family_id for example in self.examples}))

    @property
    def membership_receipt_sha256(self) -> str:
        return gemma_progressive_panel_membership_receipt_sha256(
            role=self.role,
            manifest_sha256=self.manifest_sha256,
            family_by_example={
                example.example_id: example.family_id
                for example in self.examples
            },
        )

    @property
    def model_inputs_receipt_sha256(self) -> str:
        return _sha256(
            "panel-calibration-inputs",
            {
                "role": self.role,
                "manifest_sha256": self.manifest_sha256,
                "members": tuple(
                    sorted(
                        (
                            example.example_id,
                            example.model_inputs_sha256,
                            example.calibration_sha256,
                        )
                        for example in self.examples
                    )
                ),
            },
        )

    @property
    def binding_sha256(self) -> str:
        """Bind the declared manifest to exact members and token tensors."""

        return _sha256(
            "panel-binding",
            {
                "role": self.role,
                "manifest_sha256": self.manifest_sha256,
                "membership_receipt_sha256": (
                    self.membership_receipt_sha256
                ),
                "model_inputs_receipt_sha256": (
                    self.model_inputs_receipt_sha256
                ),
            },
        )

    def validate_view(
        self,
        view: (
            FitDevelopmentView
            | SelectionDevelopmentView
            | GuardDevelopmentView
        ),
    ) -> None:
        for example in self.examples:
            example.validate_integrity()
        if (
            view.role != self.role
            or view.manifest_sha256 != self.manifest_sha256
            or view.example_count != len(self.examples)
            or view.family_ids != self.family_ids
        ):
            raise ValueError(
                "materialized panel differs from the compiler development "
                "view"
            )


def make_gemma_progressive_panel(
    *,
    role: DevelopmentRole,
    manifest_sha256: str,
    batches: Sequence[CalibrationBatch],
    family_by_example: Mapping[str, str],
    forbidden_manifest_sha256s: tuple[str, ...] = (),
) -> GemmaProgressivePanel:
    """Split replayable batches into strict single-example panel members."""

    if isinstance(batches, (str, bytes)) or not isinstance(
        batches,
        Sequence,
    ):
        raise TypeError("batches must be a calibration-batch sequence")
    if not isinstance(family_by_example, Mapping):
        raise TypeError("family_by_example must be a mapping")
    examples: list[GemmaProgressiveExample] = []
    seen: set[str] = set()
    for batch in batches:
        if not isinstance(batch, CalibrationBatch):
            raise TypeError("every panel batch must be a CalibrationBatch")
        if batch.example_ids is None:
            raise ValueError(
                "panel construction requires explicit calibration example IDs"
            )
        for index, example_id in enumerate(batch.example_ids):
            if example_id in seen:
                raise ValueError(f"duplicate panel example: {example_id!r}")
            seen.add(example_id)
            try:
                family_id = family_by_example[example_id]
            except KeyError as error:
                raise ValueError(
                    f"missing family for panel example {example_id!r}"
                ) from error
            examples.append(
                GemmaProgressiveExample(
                    example_id=example_id,
                    family_id=family_id,
                    batch=batch.sample(index),
                )
            )
    if set(family_by_example) != seen:
        raise ValueError(
            "family manifest must name exactly the materialized panel examples"
        )
    return GemmaProgressivePanel(
        role=role,
        manifest_sha256=manifest_sha256,
        examples=tuple(examples),
        forbidden_manifest_sha256s=forbidden_manifest_sha256s,
    )


@dataclass(frozen=True, slots=True)
class GemmaTwoHeadFitSequence:
    """Private A-fit trace for finite-displacement X4/H4 head fitting."""

    example_id: str
    family_id: str
    model_inputs_sha256: str
    runtime_binding_sha256: str
    source_modes: Tensor
    logical_positions: Tensor
    valid_target_mask: Tensor
    source_eligible_mask: Tensor
    target_affected_mask: Tensor
    native_x4: Tensor
    candidate_x4: Tensor
    native_h4: Tensor
    candidate_h4: Tensor
    x4_loss_gradient: Tensor
    h4_loss_gradient: Tensor
    candidate_h4_loss_gradient: Tensor | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.example_id, label="example_id")
        _require_identifier(self.family_id, label="family_id")
        _require_sha256(
            self.model_inputs_sha256,
            label="model_inputs_sha256",
        )
        _require_sha256(
            self.runtime_binding_sha256,
            label="runtime_binding_sha256",
        )
        if (
            not isinstance(self.source_modes, Tensor)
            or self.source_modes.ndim != 2
            or not self.source_modes.is_floating_point()
            or self.source_modes.shape[0] <= 0
            or self.source_modes.shape[1] <= 0
        ):
            raise ValueError("fit source modes must have shape [S, r]")
        sequence_length = int(self.source_modes.shape[0])
        if (
            not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.shape != (sequence_length,)
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("fit logical positions must have shape [S]")
        for name in (
            "valid_target_mask",
            "source_eligible_mask",
            "target_affected_mask",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (sequence_length,)
                or value.dtype != torch.bool
            ):
                raise ValueError(f"{name} must be a boolean [S] tensor")
        if (
            bool(
                (
                    self.source_eligible_mask
                    & ~self.valid_target_mask
                ).any()
            )
            or bool(
                (
                    self.target_affected_mask
                    & ~self.valid_target_mask
                ).any()
            )
            or not bool(self.target_affected_mask.any())
        ):
            raise ValueError("fit masks have inconsistent support")
        positions = self.logical_positions[self.valid_target_mask]
        if (
            positions.numel() == 0
            or bool((positions < 0).any())
            or (
                positions.numel() > 1
                and not bool(torch.all(positions[1:] > positions[:-1]))
            )
        ):
            raise ValueError(
                "fit logical positions must be nonnegative and increasing"
            )
        streams = (
            self.native_x4,
            self.candidate_x4,
            self.native_h4,
            self.candidate_h4,
            self.x4_loss_gradient,
            self.h4_loss_gradient,
        )
        if (
            any(
                not isinstance(value, Tensor)
                or value.ndim != 2
                or not value.is_floating_point()
                for value in streams
            )
            or any(value.shape != streams[0].shape for value in streams[1:])
            or streams[0].shape[0] != sequence_length
            or streams[0].shape[1] <= 0
        ):
            raise ValueError("fit boundary tensors must share [S, width]")
        candidate_gradient = self.candidate_h4_loss_gradient
        if candidate_gradient is not None and (
            not isinstance(candidate_gradient, Tensor)
            or candidate_gradient.shape != streams[0].shape
            or not candidate_gradient.is_floating_point()
        ):
            raise ValueError(
                "candidate H4 loss gradient must match [S, width]"
            )
        active = self.target_affected_mask
        finite = (self.source_modes[self.valid_target_mask],)
        finite += tuple(value[active] for value in streams)
        if candidate_gradient is not None:
            finite += (candidate_gradient[active],)
        if any(not bool(torch.isfinite(value).all()) for value in finite):
            raise ValueError("fit trace must be finite on authenticated rows")
        inactive_source = ~self.source_eligible_mask
        if bool(inactive_source.any()) and not bool(
            (self.source_modes[inactive_source] == 0).all()
        ):
            raise ValueError("source modes must be zero off source support")

    @property
    def width(self) -> int:
        return int(self.native_x4.shape[1])

    @property
    def source_rank(self) -> int:
        return int(self.source_modes.shape[1])

    @property
    def affected_rows(self) -> int:
        return int(self.target_affected_mask.sum())

    @property
    def x4_residual_rows(self) -> Tensor:
        return (
            self.native_x4[self.target_affected_mask]
            - self.candidate_x4[self.target_affected_mask]
        )

    @property
    def h4_residual_rows(self) -> Tensor:
        return (
            self.native_h4[self.target_affected_mask]
            - self.candidate_h4[self.target_affected_mask]
        )

    @property
    def artifact_sha256(self) -> str:
        return _sha256(
            "two-head-fit-sequence",
            {
                "example_id": self.example_id,
                "family_id": self.family_id,
                "model_inputs_sha256": self.model_inputs_sha256,
                "runtime_binding_sha256": self.runtime_binding_sha256,
                "tensor_sha256s": {
                    name: _tensor_sha256(getattr(self, name))
                    for name in (
                        "source_modes",
                        "logical_positions",
                        "valid_target_mask",
                        "source_eligible_mask",
                        "target_affected_mask",
                        "native_x4",
                        "candidate_x4",
                        "native_h4",
                        "candidate_h4",
                        "x4_loss_gradient",
                        "h4_loss_gradient",
                    )
                }
                | {
                    "candidate_h4_loss_gradient": (
                        None
                        if self.candidate_h4_loss_gradient is None
                        else _tensor_sha256(
                            self.candidate_h4_loss_gradient
                        )
                    )
                },
            },
        )

    def detached_copy(self) -> GemmaTwoHeadFitSequence:
        return GemmaTwoHeadFitSequence(
            example_id=self.example_id,
            family_id=self.family_id,
            model_inputs_sha256=self.model_inputs_sha256,
            runtime_binding_sha256=self.runtime_binding_sha256,
            source_modes=self.source_modes.detach().clone(),
            logical_positions=self.logical_positions.detach().clone(),
            valid_target_mask=self.valid_target_mask.detach().clone(),
            source_eligible_mask=self.source_eligible_mask.detach().clone(),
            target_affected_mask=self.target_affected_mask.detach().clone(),
            native_x4=self.native_x4.detach().clone(),
            candidate_x4=self.candidate_x4.detach().clone(),
            native_h4=self.native_h4.detach().clone(),
            candidate_h4=self.candidate_h4.detach().clone(),
            x4_loss_gradient=self.x4_loss_gradient.detach().clone(),
            h4_loss_gradient=self.h4_loss_gradient.detach().clone(),
            candidate_h4_loss_gradient=(
                None
                if self.candidate_h4_loss_gradient is None
                else self.candidate_h4_loss_gradient.detach().clone()
            ),
        )


@dataclass(frozen=True, slots=True)
class GemmaL3L4DevelopmentObservation:
    """One scalar-bound observation with private transient tensor payloads."""

    example_id: str
    family_id: str
    model_inputs_sha256: str
    runtime_binding_sha256: str
    source_logits: Tensor
    candidate_logits: Tensor
    projection_oracle_logits: Tensor
    carrier_oracle_logits: Tensor
    targets: Tensor
    source_target_modes: Tensor
    candidate_target_modes: Tensor
    source_full_width_delta: Tensor
    projection_full_width_delta: Tensor
    valid_target_rows: int
    affected_target_rows: int
    carrier_residual_rows: Tensor | None = None
    carrier_loss_gradient_rows: Tensor | None = None
    complete_boundary_oracle_max_abs_logit_error: float | None = None
    two_head_fit_sequence: GemmaTwoHeadFitSequence | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.example_id, label="example_id")
        _require_identifier(self.family_id, label="family_id")
        _require_sha256(
            self.model_inputs_sha256,
            label="model_inputs_sha256",
        )
        _require_sha256(
            self.runtime_binding_sha256,
            label="runtime_binding_sha256",
        )
        logits = (
            self.source_logits,
            self.candidate_logits,
            self.projection_oracle_logits,
            self.carrier_oracle_logits,
        )
        if (
            any(
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.ndim != 2
                for value in logits
            )
            or any(value.shape != logits[0].shape for value in logits[1:])
            or not isinstance(self.targets, Tensor)
            or self.targets.ndim != 1
            or self.targets.shape[0] != logits[0].shape[0]
            or self.targets.dtype not in (torch.int32, torch.int64)
            or self.targets.numel() == 0
        ):
            raise ValueError(
                "behavioral tensors must be aligned supervised logits and "
                "targets"
            )
        modal_shape = self.source_target_modes.shape
        full_shape = self.source_full_width_delta.shape
        if (
            not self.source_target_modes.is_floating_point()
            or self.source_target_modes.ndim != 2
            or self.candidate_target_modes.shape != modal_shape
            or not self.candidate_target_modes.is_floating_point()
            or not self.source_full_width_delta.is_floating_point()
            or self.source_full_width_delta.ndim != 2
            or self.projection_full_width_delta.shape != full_shape
            or not self.projection_full_width_delta.is_floating_point()
            or modal_shape[0] != full_shape[0]
            or modal_shape[0] != self.affected_target_rows
            or type(self.valid_target_rows) is not int
            or self.valid_target_rows <= 0
            or type(self.affected_target_rows) is not int
            or not 0 < self.affected_target_rows <= self.valid_target_rows
        ):
            raise ValueError("boundary observation geometry differs")
        finite = (*logits, self.source_target_modes, self.candidate_target_modes)
        finite += (
            self.source_full_width_delta,
            self.projection_full_width_delta,
        )
        if any(not bool(torch.isfinite(value).all()) for value in finite):
            raise ValueError("development observations must be finite")
        residual = self.carrier_residual_rows
        gradient = self.carrier_loss_gradient_rows
        oracle_error = self.complete_boundary_oracle_max_abs_logit_error
        if (residual is None) != (gradient is None):
            raise ValueError(
                "carrier residual and NLL-gradient rows are atomic"
            )
        if residual is None:
            if oracle_error is not None:
                raise ValueError(
                    "complete-boundary oracle requires residual collection"
                )
        else:
            if (
                residual.ndim != 2
                or gradient is None
                or gradient.shape != residual.shape
                or residual.shape[0] != self.affected_target_rows
                or residual.shape[1] != full_shape[1]
                or not residual.is_floating_point()
                or not gradient.is_floating_point()
                or not bool(torch.isfinite(residual).all())
                or not bool(torch.isfinite(gradient).all())
                or oracle_error is None
                or not math.isfinite(oracle_error)
                or oracle_error < 0.0
            ):
                raise ValueError(
                    "carrier residual/NLL-gradient observation geometry differs"
                )
        fit_sequence = self.two_head_fit_sequence
        if fit_sequence is not None:
            expected_h4_gradient = (
                fit_sequence.h4_loss_gradient
                if fit_sequence.candidate_h4_loss_gradient is None
                else fit_sequence.candidate_h4_loss_gradient
            )
            if (
                residual is None
                or not isinstance(fit_sequence, GemmaTwoHeadFitSequence)
                or fit_sequence.example_id != self.example_id
                or fit_sequence.family_id != self.family_id
                or fit_sequence.model_inputs_sha256
                != self.model_inputs_sha256
                or fit_sequence.runtime_binding_sha256
                != self.runtime_binding_sha256
                or fit_sequence.affected_rows != self.affected_target_rows
                or fit_sequence.width != full_shape[1]
                or not torch.equal(
                    fit_sequence.h4_residual_rows,
                    residual.to(
                        device=fit_sequence.h4_residual_rows.device,
                        dtype=fit_sequence.h4_residual_rows.dtype,
                    ),
                )
                or not torch.equal(
                    expected_h4_gradient[
                        fit_sequence.target_affected_mask
                    ],
                    gradient.to(
                        device=expected_h4_gradient.device,
                        dtype=expected_h4_gradient.dtype,
                    ),
                )
            ):
                raise ValueError(
                    "two-head fit trace differs from the carrier observation"
                )

    @property
    def artifact_sha256(self) -> str:
        tensors = {
            "source_logits": self.source_logits,
            "candidate_logits": self.candidate_logits,
            "projection_oracle_logits": self.projection_oracle_logits,
            "carrier_oracle_logits": self.carrier_oracle_logits,
            "targets": self.targets,
            "source_target_modes": self.source_target_modes,
            "candidate_target_modes": self.candidate_target_modes,
            "source_full_width_delta": self.source_full_width_delta,
            "projection_full_width_delta": (
                self.projection_full_width_delta
            ),
            "carrier_residual_rows": self.carrier_residual_rows,
            "carrier_loss_gradient_rows": (
                self.carrier_loss_gradient_rows
            ),
        }
        return _sha256(
            "development-observation",
            {
                "example_id": self.example_id,
                "family_id": self.family_id,
                "model_inputs_sha256": self.model_inputs_sha256,
                "runtime_binding_sha256": self.runtime_binding_sha256,
                "valid_target_rows": self.valid_target_rows,
                "affected_target_rows": self.affected_target_rows,
                "complete_boundary_oracle_max_abs_logit_error": (
                    self.complete_boundary_oracle_max_abs_logit_error
                ),
                "tensor_sha256s": {
                    name: None if value is None else _tensor_sha256(value)
                    for name, value in tensors.items()
                },
                "two_head_fit_sequence_sha256": (
                    None
                    if self.two_head_fit_sequence is None
                    else self.two_head_fit_sequence.artifact_sha256
                ),
            },
        )


class GemmaProgressiveExecutable(Protocol):
    """Candidate-bound evaluator used by the worker registry."""

    @property
    def candidate_artifact_sha256(self) -> str: ...

    @property
    def candidate_execution_sha256(self) -> str: ...

    @property
    def runtime_binding_sha256(self) -> str: ...

    def observe(
        self,
        example: GemmaProgressiveExample,
        *,
        collect_carrier_fisher: bool,
    ) -> GemmaL3L4DevelopmentObservation: ...


class GemmaMutationLowerer(Protocol):
    """Optional candidate-bound mutation implementation."""

    def propose(
        self,
        *,
        parent: ProgressiveCandidate,
        residual_map: ResidualMap,
        analysis: GemmaCarrierResidualAnalysis,
        phase: ProgressivePhase,
    ) -> Sequence[MutationProposal]: ...

    def build(
        self,
        *,
        parent: ProgressiveCandidate,
        proposal: MutationProposal,
        analysis: GemmaCarrierResidualAnalysis,
    ) -> tuple[ProgressiveCandidate, GemmaProgressiveExecutable]: ...


class GemmaGuardClaimAuthority(Protocol):
    """Durable claim-first authority for the one-use A guard."""

    def claim(
        self,
        *,
        protocol_sha256: str,
        guard_manifest_sha256: str,
        challenger_receipt_sha256: str,
    ) -> str: ...


class GemmaGuardPanelProvider(Protocol):
    """Hash-only guard identity plus claim-gated materialization."""

    manifest_sha256: str
    example_count: int
    family_ids: tuple[str, ...]
    membership_receipt_sha256: str
    preclaim_binding_sha256: str

    def open_after_claim(
        self,
        claim_sha256: str,
    ) -> GemmaProgressivePanel: ...


class LegacyRank64GemmaProgressiveExecutable:
    """A-only evaluator/probe around the frozen legacy rank-64 runtime."""

    def __init__(
        self,
        *,
        adapter: Gemma3CausalLMAdapter,
        runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
        candidate_execution_sha256: str,
    ) -> None:
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(
            runtime,
            Gemma3L3L4GraphOrganizedSVDShadowRuntime,
        ):
            raise TypeError("runtime must be the strict rank-64 shadow runtime")
        self._adapter = adapter
        self._runtime = runtime
        self._candidate_execution_sha256 = _require_sha256(
            candidate_execution_sha256,
            label="candidate execution",
        )
        if (
            self._candidate_execution_sha256
            != runtime.adapter_execution_sha256
        ):
            raise ValueError(
                "candidate execution must equal the authenticated adapter "
                "execution"
            )
        self._objective = CausalLanguageModelNLL()
        self._one_pass_bridge = runtime.export_one_pass_bridge()
        self._authenticate()

    @property
    def candidate_artifact_sha256(self) -> str:
        return self._runtime.candidate_artifact_sha256

    @property
    def candidate_execution_sha256(self) -> str:
        return self._candidate_execution_sha256

    @property
    def runtime_binding_sha256(self) -> str:
        return self._runtime.runtime_binding_sha256

    def validate_integrity(self) -> None:
        """Re-authenticate the bound model, adapter, and measurement runtime."""

        self._authenticate()

    def _authenticate(self) -> None:
        self._runtime.validate_integrity()
        self._one_pass_bridge.validate_integrity()
        if (
            self._one_pass_bridge.parent_runtime_binding_sha256
            != self._runtime.runtime_binding_sha256
        ):
            raise ValueError("one-pass bridge lineage differs from the runtime")
        if self._adapter.module.training or any(
            module.training for module in self._adapter.module.modules()
        ):
            raise ValueError("source Gemma must remain completely in eval mode")
        if (
            self._adapter.model_fingerprint()
            != self._runtime.live_model_sha256
            or self._adapter.execution_fingerprint()
            != self._runtime.adapter_execution_sha256
        ):
            raise ValueError(
                "live Gemma model or execution semantics differ from the "
                "authenticated runtime"
            )
        sites = {site.id: site for site in self._adapter.activation_sites}
        if _H4_SITE not in sites or not sites[_H4_SITE].intervenable:
            raise ValueError(
                "Gemma adapter complete layer-4 output ABI drifted"
            )

    def _forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        capture_sites: tuple[str, ...],
        interventions: Mapping[str, object] | None = None,
        retain_gradients: bool = False,
    ):
        self._authenticate()
        try:
            return self._adapter.forward(
                model_inputs,
                capture_sites=capture_sites,
                interventions=interventions,  # type: ignore[arg-type]
                retain_gradients=retain_gradients,
            )
        finally:
            self._authenticate()

    @staticmethod
    def _gather_logits(logits: Tensor, indices: Tensor) -> Tensor:
        return (
            logits[0]
            .index_select(0, indices.to(device=logits.device))
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        )

    def _oracle_injections(
        self,
        shadow,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Derive dynamic-width X4 oracles without a B-frozen assembler."""

        self._runtime.validate_result_binding(shadow)
        source_full = shadow.authoritative_x4 - shadow.reference_x4
        valid = shadow.valid_target_mask
        if not bool(torch.isfinite(source_full[valid]).all()):
            raise ValueError("valid source X4 deltas must be finite")
        encoded_input = torch.where(
            valid.unsqueeze(-1),
            source_full,
            torch.zeros_like(source_full),
        )
        source_modes = self._runtime.encode_target_delta(encoded_input)
        projection_full = self._runtime.decode_target_modal_delta(
            source_modes
        )
        projection_x4 = shadow.authoritative_x4.detach().clone()
        affected = shadow.target_affected_mask
        projection_x4[affected] = (
            shadow.reference_x4
            + projection_full.to(
                device=shadow.reference_x4.device,
                dtype=shadow.reference_x4.dtype,
            )
        )[affected]
        carrier_x4 = shadow.authoritative_x4.detach().clone().contiguous()
        return (
            projection_x4.contiguous(),
            carrier_x4,
            source_full,
            source_modes,
            projection_full,
        )

    def _two_boundary_nll_gradient_rows(
        self,
        *,
        example: GemmaProgressiveExample,
        shadow,
    ) -> tuple[Tensor, Tensor, float, GemmaTwoHeadFitSequence]:
        model_inputs = example.batch.model_inputs
        h4_reached = False
        native_x4_leaf: Tensor | None = None

        def detach_x4(original: Tensor) -> Tensor:
            nonlocal native_x4_leaf
            if native_x4_leaf is not None:
                raise RuntimeError("native X4 leaf intervention repeated")
            if not _bitwise_equal(original, shadow.authoritative_x4):
                raise RuntimeError(
                    "native NLL-gradient pass reached a non-authenticated X4"
                )
            native_x4_leaf = original.detach().requires_grad_(True)
            return native_x4_leaf

        with torch.enable_grad():
            native = self._forward(
                model_inputs,
                capture_sites=(_X4_SITE, _H4_SITE),
                interventions={_X4_SITE: detach_x4},
                retain_gradients=True,
            )
            native_x4 = native.activations[_X4_SITE]
            native_h4 = native.activations[_H4_SITE]
            if native_x4_leaf is None or native_x4 is not native_x4_leaf:
                raise RuntimeError(
                    "native NLL-gradient pass did not bind the detached X4 leaf"
                )
            loss = self._objective(native, example.batch)
            native_x4_gradient, native_h4_gradient = torch.autograd.grad(
                loss,
                (native_x4, native_h4),
                retain_graph=False,
                create_graph=False,
            )
        if not _bitwise_equal(native.logits, shadow.authoritative_logits):
            raise RuntimeError(
                "NLL-gradient pass differs from authenticated native logits"
            )

        with torch.no_grad():
            one_pass = self._one_pass_bridge.execute(
                self._adapter,
                model_inputs,
            )
        prefix = one_pass.prefix
        bridge_checks = {
            "model_inputs": (
                one_pass.model_inputs_sha256
                == shadow.model_inputs_sha256
            ),
            "logical_positions": torch.equal(
                prefix.logical_positions,
                shadow.logical_positions,
            ),
            "valid_target_mask": torch.equal(
                prefix.valid_target_mask,
                shadow.valid_target_mask,
            ),
            "source_eligible_mask": torch.equal(
                prefix.source_eligible_mask,
                shadow.source_eligible_mask,
            ),
            "target_affected_mask": torch.equal(
                prefix.target_affected_mask,
                shadow.target_affected_mask,
            ),
            "source_modes": torch.equal(
                prefix.source_modes,
                shadow.source_modes,
            ),
            "clamped_y3": _bitwise_equal(
                prefix.clamped_y3,
                shadow.clamped_y3,
            ),
            "reference_x4": _bitwise_equal(
                one_pass.reference_x4,
                shadow.reference_x4,
            ),
            "candidate_x4_on_support": _bitwise_equal(
                one_pass.candidate_x4[shadow.target_affected_mask],
                shadow.candidate_x4[shadow.target_affected_mask],
            ),
        }
        bridge_mismatches = tuple(
            name
            for name, matches in bridge_checks.items()
            if not matches
        )
        if bridge_mismatches:
            raise RuntimeError(
                "one-pass base carrier differs from the authenticated "
                "graph: "
                + ", ".join(bridge_mismatches)
            )

        reached_y3 = False
        reached_x4 = False

        def intervene_y3(original: Tensor) -> Tensor:
            nonlocal reached_y3
            if not _bitwise_equal(original, shadow.native_y3):
                raise RuntimeError(
                    "carrier probe reached a non-authenticated native Y3"
                )
            reached_y3 = True
            return shadow.clamped_y3

        def intervene_x4(original: Tensor) -> Tensor:
            nonlocal reached_x4
            if not _bitwise_equal(original, shadow.reference_x4):
                raise RuntimeError(
                    "carrier probe reached a non-authenticated reference X4"
                )
            reached_x4 = True
            return shadow.candidate_x4

        with torch.no_grad():
            candidate = self._forward(
                model_inputs,
                capture_sites=(_H4_SITE,),
                interventions={
                    "layer.3.mlp.operator_output": intervene_y3,
                    "layer.4.mlp.normalized_input": intervene_x4,
                },
            )
        if (
            not reached_y3
            or not reached_x4
            or not _bitwise_equal(
                candidate.logits,
                shadow.candidate_logits,
            )
        ):
            raise RuntimeError(
                "carrier probe did not reproduce the authenticated seed path"
            )
        candidate_h4 = candidate.activations[_H4_SITE]

        reached_y3 = False
        reached_x4 = False

        def intervene_h4(original: Tensor) -> Tensor:
            nonlocal h4_reached
            if not _bitwise_equal(original, candidate_h4):
                raise RuntimeError(
                    "complete-boundary oracle reached an unexpected H4"
                )
            h4_reached = True
            return native_h4.detach()

        with torch.no_grad():
            oracle = self._forward(
                model_inputs,
                capture_sites=(),
                interventions={
                    "layer.3.mlp.operator_output": intervene_y3,
                    "layer.4.mlp.normalized_input": intervene_x4,
                    _H4_SITE: intervene_h4,
                },
            )
        if not reached_y3 or not reached_x4 or not h4_reached:
            raise RuntimeError(
                "complete-boundary oracle did not reach every boundary"
            )
        oracle_error = float(
            (
                oracle.logits.detach().to(dtype=torch.float64)
                - shadow.authoritative_logits.detach().to(
                    dtype=torch.float64
                )
            )
            .abs()
            .max()
        )
        affected = shadow.target_affected_mask
        if native_x4.shape[0] != 1:
            raise ValueError(
                "progressive fit examples must contain one sequence"
            )
        fit_sequence = GemmaTwoHeadFitSequence(
            example_id=example.example_id,
            family_id=example.family_id,
            model_inputs_sha256=shadow.model_inputs_sha256,
            runtime_binding_sha256=self.runtime_binding_sha256,
            source_modes=(
                prefix.source_modes[0]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            logical_positions=(
                prefix.logical_positions[0]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .contiguous()
            ),
            valid_target_mask=(
                prefix.valid_target_mask[0]
                .detach()
                .to(device="cpu")
                .contiguous()
            ),
            source_eligible_mask=(
                prefix.source_eligible_mask[0]
                .detach()
                .to(device="cpu")
                .contiguous()
            ),
            target_affected_mask=(
                prefix.target_affected_mask[0]
                .detach()
                .to(device="cpu")
                .contiguous()
            ),
            native_x4=(
                native_x4.detach()[0]
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            candidate_x4=(
                one_pass.candidate_x4.detach()[0]
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            native_h4=(
                native_h4.detach()[0]
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            candidate_h4=(
                one_pass.candidate_h4.detach()[0]
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            x4_loss_gradient=(
                native_x4_gradient.detach()[0]
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            h4_loss_gradient=(
                native_h4_gradient.detach()[0]
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
        )
        # Preserve the exact float32 boundary values first, then form their
        # finite displacement in the analysis dtype.  Subtracting in the live
        # dtype before promotion loses low bits and would disagree with the
        # authenticated private fit trace.
        residual = fit_sequence.h4_residual_rows
        gradient = fit_sequence.h4_loss_gradient[
            fit_sequence.target_affected_mask
        ]
        return (
            residual.contiguous(),
            gradient.contiguous(),
            oracle_error,
            fit_sequence,
        )

    def observe(
        self,
        example: GemmaProgressiveExample,
        *,
        collect_carrier_fisher: bool,
    ) -> GemmaL3L4DevelopmentObservation:
        if not isinstance(example, GemmaProgressiveExample):
            raise TypeError("example must be a GemmaProgressiveExample")
        example.validate_integrity()
        self._authenticate()
        model_inputs = example.batch.model_inputs
        shadow = self._runtime.execute_model_shadow(
            self._adapter,
            model_inputs,
            arm="all_on",
        )
        (
            projection_x4,
            carrier_x4,
            source_full_width_delta,
            source_target_modes,
            projection_full_width_delta,
        ) = self._oracle_injections(shadow)
        projection = self._runtime.execute_oracle_suffix(
            self._adapter,
            model_inputs,
            shadow,
            projection_x4,
            role="projection_64",
        )
        carrier = self._runtime.execute_oracle_suffix(
            self._adapter,
            model_inputs,
            shadow,
            carrier_x4,
            role="exact_x4_carrier",
        )
        input_ids = model_inputs["input_ids"]
        boundary_indices, derived_targets = (
            derive_gemma3_l3_l4_supervised_boundary(
                input_ids,
                shadow.valid_target_mask,
            )
        )
        observed_valid = (
            example.batch.valid_positions.detach().to(device="cpu")
        )
        expected_valid = shadow.valid_target_mask.detach().to(device="cpu")
        if not torch.equal(observed_valid, expected_valid):
            raise ValueError(
                "calibration valid positions differ from the authenticated "
                "Gemma sequence"
            )
        batch_targets = example.batch.targets.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        expected_targets = torch.full_like(batch_targets, -100)
        expected_targets[0, boundary_indices] = derived_targets
        if not torch.equal(batch_targets, expected_targets):
            raise ValueError(
                "calibration targets must equal the exact authenticated "
                "causal next-token target mask"
            )
        affected_supervised = (
            shadow.target_affected_mask[0]
            .detach()
            .to(device="cpu")
            .index_select(0, boundary_indices)
        )
        if not bool(affected_supervised.any()):
            raise ValueError(
                "example has no causally affected supervised tokens"
            )
        # Fidelity is measured across the complete supervised sequence.  The
        # affected mask remains the eligibility gate and the boundary-analysis
        # support, but excluding unaffected logits would hide collateral error
        # introduced by the earlier Y3 clamp.
        indices = boundary_indices
        targets = derived_targets.contiguous()
        affected = shadow.target_affected_mask
        residual_rows: Tensor | None = None
        gradient_rows: Tensor | None = None
        complete_error: float | None = None
        fit_sequence: GemmaTwoHeadFitSequence | None = None
        if collect_carrier_fisher:
            (
                residual_rows,
                gradient_rows,
                complete_error,
                fit_sequence,
            ) = (
                self._two_boundary_nll_gradient_rows(
                    example=example,
                    shadow=shadow,
                )
            )
        modal_affected = affected.to(device=source_target_modes.device)
        projection_affected = affected.to(
            device=projection_full_width_delta.device
        )
        observation = GemmaL3L4DevelopmentObservation(
            example_id=example.example_id,
            family_id=example.family_id,
            model_inputs_sha256=shadow.model_inputs_sha256,
            runtime_binding_sha256=shadow.runtime_binding_sha256,
            source_logits=self._gather_logits(
                shadow.authoritative_logits,
                indices,
            ),
            candidate_logits=self._gather_logits(
                shadow.candidate_logits,
                indices,
            ),
            projection_oracle_logits=self._gather_logits(
                projection.logits,
                indices,
            ),
            carrier_oracle_logits=self._gather_logits(
                carrier.logits,
                indices,
            ),
            targets=targets,
            source_target_modes=(
                source_target_modes[modal_affected]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            candidate_target_modes=(
                shadow.predicted_target_modal_delta[affected]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            source_full_width_delta=(
                source_full_width_delta[affected]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            projection_full_width_delta=(
                projection_full_width_delta[projection_affected]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            ),
            valid_target_rows=int(shadow.valid_target_mask.sum()),
            affected_target_rows=int(affected.sum()),
            carrier_residual_rows=residual_rows,
            carrier_loss_gradient_rows=gradient_rows,
            complete_boundary_oracle_max_abs_logit_error=complete_error,
            two_head_fit_sequence=fit_sequence,
        )
        if observation.model_inputs_sha256 != example.model_inputs_sha256:
            raise RuntimeError("runtime model-input authentication drifted")
        self._authenticate()
        return observation


@dataclass(frozen=True, slots=True)
class GemmaCarrierResidualAnalysis:
    """Private H4/X4 fit artifact behind one scalar ``ResidualMap`` receipt."""

    protocol_sha256: str
    fit_manifest_sha256: str
    candidate_artifact_sha256: str
    candidate_receipt_sha256: str
    runtime_binding_sha256: str
    location: str
    directions: Tensor
    residual_eigenvalues: Tensor
    loss_couplings: Tensor
    total_residual_energy: float
    family_row_counts: tuple[tuple[str, int], ...]
    observation_sha256s: tuple[str, ...]
    complete_boundary_oracle_max_abs_logit_error: float
    x4_directions: Tensor | None = None
    x4_residual_eigenvalues: Tensor | None = None
    x4_loss_couplings: Tensor | None = None
    x4_total_residual_energy: float | None = None
    fit_sequences: tuple[GemmaTwoHeadFitSequence, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "fit_manifest_sha256",
            "candidate_artifact_sha256",
            "candidate_receipt_sha256",
            "runtime_binding_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.location != _H4_SITE:
            raise ValueError("carrier analysis must use layer.4.output")
        if (
            not isinstance(self.directions, Tensor)
            or not self.directions.is_floating_point()
            or self.directions.ndim != 2
            or self.directions.shape[0] <= 0
            or self.directions.shape[1] <= 0
            or self.residual_eigenvalues.shape
            != (self.directions.shape[0],)
            or self.loss_couplings.shape != (self.directions.shape[0],)
            or not bool(torch.isfinite(self.directions).all())
            or not bool(torch.isfinite(self.residual_eigenvalues).all())
            or not bool(torch.isfinite(self.loss_couplings).all())
            or bool((self.residual_eigenvalues < 0).any())
            or bool((self.loss_couplings < 0).any())
        ):
            raise ValueError("residual analysis tensors are invalid")
        total_energy = _finite_scalar(
            self.total_residual_energy,
            label="total_residual_energy",
        )
        if total_energy < 0.0 or (
            float(self.residual_eigenvalues.sum())
            > total_energy
            + max(1.0, total_energy) * 1.0e-10
        ):
            raise ValueError(
                "total residual energy cannot omit selected eigenvalue energy"
            )
        identity = self.directions @ self.directions.T
        error = float(
            (
                identity
                - torch.eye(
                    self.directions.shape[0],
                    dtype=self.directions.dtype,
                    device=self.directions.device,
                )
            )
            .abs()
            .max()
        )
        if error > 1.0e-8:
            raise ValueError("residual directions must be orthonormal")
        if (
            type(self.family_row_counts) is not tuple
            or not self.family_row_counts
            or self.family_row_counts
            != tuple(sorted(self.family_row_counts))
            or len({name for name, _ in self.family_row_counts})
            != len(self.family_row_counts)
            or any(
                not isinstance(count, int)
                or count <= 0
                or not isinstance(name, str)
                or not name
                for name, count in self.family_row_counts
            )
        ):
            raise ValueError("family row counts must be canonical and positive")
        if (
            type(self.observation_sha256s) is not tuple
            or not self.observation_sha256s
            or self.observation_sha256s
            != tuple(sorted(set(self.observation_sha256s)))
        ):
            raise ValueError(
                "observation identities must be sorted and unique"
            )
        for value in self.observation_sha256s:
            _require_sha256(value, label="observation identity")
        if (
            not math.isfinite(
                self.complete_boundary_oracle_max_abs_logit_error
            )
            or self.complete_boundary_oracle_max_abs_logit_error < 0.0
        ):
            raise ValueError("complete-boundary oracle error must be finite")
        x4_values = (
            self.x4_directions,
            self.x4_residual_eigenvalues,
            self.x4_loss_couplings,
            self.x4_total_residual_energy,
        )
        if any(value is None for value in x4_values):
            if not all(value is None for value in x4_values):
                raise ValueError("X4 residual analysis fields are atomic")
            if self.fit_sequences:
                raise ValueError("fit sequences require X4 residual analysis")
        else:
            x4_directions = self.x4_directions
            x4_eigenvalues = self.x4_residual_eigenvalues
            x4_couplings = self.x4_loss_couplings
            assert x4_directions is not None
            assert x4_eigenvalues is not None
            assert x4_couplings is not None
            assert self.x4_total_residual_energy is not None
            if (
                not isinstance(x4_directions, Tensor)
                or x4_directions.ndim != 2
                or not x4_directions.is_floating_point()
                or x4_directions.shape[0] <= 0
                or x4_directions.shape[1] != self.directions.shape[1]
                or x4_eigenvalues.shape != (x4_directions.shape[0],)
                or x4_couplings.shape != (x4_directions.shape[0],)
                or not bool(torch.isfinite(x4_directions).all())
                or not bool(torch.isfinite(x4_eigenvalues).all())
                or not bool(torch.isfinite(x4_couplings).all())
                or bool((x4_eigenvalues < 0).any())
                or bool((x4_couplings < 0).any())
            ):
                raise ValueError("X4 residual analysis tensors are invalid")
            x4_identity = x4_directions @ x4_directions.T
            x4_error = float(
                (
                    x4_identity
                    - torch.eye(
                        x4_directions.shape[0],
                        dtype=x4_directions.dtype,
                        device=x4_directions.device,
                    )
                )
                .abs()
                .max()
            )
            if x4_error > 1.0e-8:
                raise ValueError("X4 residual directions must be orthonormal")
            x4_total = _finite_scalar(
                self.x4_total_residual_energy,
                label="x4_total_residual_energy",
            )
            if x4_total < 0.0 or (
                float(x4_eigenvalues.sum())
                > x4_total + max(1.0, x4_total) * 1.0e-10
            ):
                raise ValueError("X4 residual energy accounting is invalid")
            if (
                type(self.fit_sequences) is not tuple
                or not self.fit_sequences
                or any(
                    not isinstance(value, GemmaTwoHeadFitSequence)
                    for value in self.fit_sequences
                )
                or tuple(
                    (value.family_id, value.example_id)
                    for value in self.fit_sequences
                )
                != tuple(
                    sorted(
                        (value.family_id, value.example_id)
                        for value in self.fit_sequences
                    )
                )
                or len(
                    {
                        value.artifact_sha256
                        for value in self.fit_sequences
                    }
                )
                != len(self.fit_sequences)
                or any(
                    value.runtime_binding_sha256
                    != self.runtime_binding_sha256
                    or value.width != self.directions.shape[1]
                    for value in self.fit_sequences
                )
            ):
                raise ValueError(
                    "two-head fit sequences must be canonical and bound"
                )

    @property
    def artifact_sha256(self) -> str:
        return _sha256(
            "carrier-residual-analysis",
            {
                "protocol_sha256": self.protocol_sha256,
                "fit_manifest_sha256": self.fit_manifest_sha256,
                "candidate_artifact_sha256": (
                    self.candidate_artifact_sha256
                ),
                "candidate_receipt_sha256": (
                    self.candidate_receipt_sha256
                ),
                "runtime_binding_sha256": self.runtime_binding_sha256,
                "location": self.location,
                "directions_sha256": _tensor_sha256(self.directions),
                "residual_eigenvalues_sha256": _tensor_sha256(
                    self.residual_eigenvalues
                ),
                "loss_couplings_sha256": _tensor_sha256(
                    self.loss_couplings
                ),
                "total_residual_energy": self.total_residual_energy,
                "selected_residual_energy": float(
                    self.residual_eigenvalues.sum()
                ),
                "family_row_counts": self.family_row_counts,
                "observation_sha256s": self.observation_sha256s,
                "complete_boundary_oracle_max_abs_logit_error": (
                    self.complete_boundary_oracle_max_abs_logit_error
                ),
                "x4_location": (
                    None
                    if self.x4_directions is None
                    else _X4_SITE
                ),
                "x4_directions_sha256": (
                    None
                    if self.x4_directions is None
                    else _tensor_sha256(self.x4_directions)
                ),
                "x4_residual_eigenvalues_sha256": (
                    None
                    if self.x4_residual_eigenvalues is None
                    else _tensor_sha256(self.x4_residual_eigenvalues)
                ),
                "x4_loss_couplings_sha256": (
                    None
                    if self.x4_loss_couplings is None
                    else _tensor_sha256(self.x4_loss_couplings)
                ),
                "x4_total_residual_energy": (
                    self.x4_total_residual_energy
                ),
                "fit_sequence_sha256s": tuple(
                    value.artifact_sha256 for value in self.fit_sequences
                ),
                "raw_prompt_or_activation_payload_in_receipt": False,
                "jvp_validation_performed": False,
                "ranking": (
                    "family_example_macro_residual_covariance_with_nll_"
                    "vjp_alignment_and_activation_gradient_gram_coupling"
                ),
            },
        )

    def residual_map(
        self,
        *,
        iteration: int,
    ) -> ResidualMap:
        energy = self.residual_eigenvalues
        total_energy = self.total_residual_energy
        h4_targets = tuple(
            ResidualTarget(
                rank=rank,
                location=self.location,
                direction_sha256=_tensor_sha256(
                    self.directions[rank].contiguous()
                ),
                residual_energy_fraction=(
                    0.0
                    if total_energy == 0.0
                    else float(energy[rank]) / total_energy
                ),
                loss_coupling=float(self.loss_couplings[rank]),
                jvp_gain=0.0,
            )
            for rank in range(self.directions.shape[0])
        )
        x4_targets: tuple[ResidualTarget, ...] = ()
        if self.x4_directions is not None:
            assert self.x4_residual_eigenvalues is not None
            assert self.x4_loss_couplings is not None
            assert self.x4_total_residual_energy is not None
            start = len(h4_targets)
            x4_targets = tuple(
                ResidualTarget(
                    rank=start + index,
                    location=_X4_SITE,
                    direction_sha256=_tensor_sha256(
                        self.x4_directions[index].contiguous()
                    ),
                    residual_energy_fraction=(
                        0.0
                        if self.x4_total_residual_energy == 0.0
                        else float(
                            self.x4_residual_eigenvalues[index]
                        )
                        / self.x4_total_residual_energy
                    ),
                    loss_coupling=float(self.x4_loss_couplings[index]),
                    jvp_gain=0.0,
                )
                for index in range(self.x4_directions.shape[0])
            )
        return ResidualMap(
            protocol_sha256=self.protocol_sha256,
            fit_manifest_sha256=self.fit_manifest_sha256,
            candidate_artifact_sha256=self.candidate_artifact_sha256,
            candidate_receipt_sha256=self.candidate_receipt_sha256,
            iteration=iteration,
            mapper_id=(
                "gemma3-l3-l4-h4-nll-vjp-residual-svd"
                if self.x4_directions is None
                else "gemma3-l3-l4-two-boundary-nll-vjp-residual-svd"
            ),
            mapper_version=1 if self.x4_directions is None else 2,
            analysis_artifact_sha256=self.artifact_sha256,
            targets=h4_targets + x4_targets,
        )

    def detached_copy(self) -> GemmaCarrierResidualAnalysis:
        """Return a tensor-isolated copy for an untrusted lowerer/caller."""

        return GemmaCarrierResidualAnalysis(
            protocol_sha256=self.protocol_sha256,
            fit_manifest_sha256=self.fit_manifest_sha256,
            candidate_artifact_sha256=self.candidate_artifact_sha256,
            candidate_receipt_sha256=self.candidate_receipt_sha256,
            runtime_binding_sha256=self.runtime_binding_sha256,
            location=self.location,
            directions=self.directions.detach().clone(),
            residual_eigenvalues=(
                self.residual_eigenvalues.detach().clone()
            ),
            loss_couplings=self.loss_couplings.detach().clone(),
            total_residual_energy=self.total_residual_energy,
            family_row_counts=self.family_row_counts,
            observation_sha256s=self.observation_sha256s,
            complete_boundary_oracle_max_abs_logit_error=(
                self.complete_boundary_oracle_max_abs_logit_error
            ),
            x4_directions=(
                None
                if self.x4_directions is None
                else self.x4_directions.detach().clone()
            ),
            x4_residual_eigenvalues=(
                None
                if self.x4_residual_eigenvalues is None
                else self.x4_residual_eigenvalues.detach().clone()
            ),
            x4_loss_couplings=(
                None
                if self.x4_loss_couplings is None
                else self.x4_loss_couplings.detach().clone()
            ),
            x4_total_residual_energy=self.x4_total_residual_energy,
            fit_sequences=tuple(
                value.detached_copy() for value in self.fit_sequences
            ),
        )


class _BoundaryAccumulator:
    def __init__(self, panel: GemmaProgressivePanel) -> None:
        family_manifest = {
            example.example_id: example.family_id
            for example in panel.examples
        }
        self._candidate = SourceAuthoritativeShadowFidelityAccumulator(
            family_manifest
        )
        self._projection = SourceAuthoritativeShadowFidelityAccumulator(
            family_manifest
        )
        self._carrier = SourceAuthoritativeShadowFidelityAccumulator(
            family_manifest
        )
        self._modal = {
            family: {
                "residual": 0.0,
                "source": 0.0,
                "candidate": 0.0,
                "dot": 0.0,
            }
            for family in panel.family_ids
        }
        self._projection_boundary = {
            family: {
                "residual": 0.0,
                "source": 0.0,
                "candidate": 0.0,
                "dot": 0.0,
            }
            for family in panel.family_ids
        }
        self.valid_rows = 0
        self.affected_rows = 0
        self.supervised_tokens = 0
        self.observation_sha256s: list[str] = []

    @staticmethod
    def _add_geometry(
        totals: dict[str, float],
        source: Tensor,
        candidate: Tensor,
    ) -> None:
        residual = candidate - source
        totals["residual"] += float(residual.square().sum())
        totals["source"] += float(source.square().sum())
        totals["candidate"] += float(candidate.square().sum())
        totals["dot"] += float((source * candidate).sum())

    def add(self, observation: GemmaL3L4DevelopmentObservation) -> None:
        common = {
            "example_id": observation.example_id,
            "family_id": observation.family_id,
            "source_logits": observation.source_logits,
            "targets": observation.targets,
        }
        self._candidate.add(
            ShadowFidelityExample(
                **common,
                candidate_logits=observation.candidate_logits,
            )
        )
        self._projection.add(
            ShadowFidelityExample(
                **common,
                candidate_logits=observation.projection_oracle_logits,
            )
        )
        self._carrier.add(
            ShadowFidelityExample(
                **common,
                candidate_logits=observation.carrier_oracle_logits,
            )
        )
        self._add_geometry(
            self._modal[observation.family_id],
            observation.source_target_modes,
            observation.candidate_target_modes,
        )
        self._add_geometry(
            self._projection_boundary[observation.family_id],
            observation.source_full_width_delta,
            observation.projection_full_width_delta,
        )
        self.valid_rows += observation.valid_target_rows
        self.affected_rows += observation.affected_target_rows
        self.supervised_tokens += int(observation.targets.numel())
        self.observation_sha256s.append(observation.artifact_sha256)

    @staticmethod
    def _behavior(summary: Mapping[str, object]) -> ProgressiveBehavioralFidelity:
        aggregate = summary["aggregate"]
        per_prompt = summary["per_prompt"]
        if not isinstance(aggregate, Mapping) or not isinstance(
            per_prompt,
            Mapping,
        ):
            raise TypeError("shadow summary structure differs")
        absolute_tail = per_prompt["absolute_delta_nll_per_token"]
        top1_tail = per_prompt["top1_agreement_to_source"]
        if not isinstance(absolute_tail, Mapping) or not isinstance(
            top1_tail,
            Mapping,
        ):
            raise TypeError("shadow tail summary structure differs")
        return ProgressiveBehavioralFidelity(
            absolute_delta_nll_per_token=abs(
                float(aggregate["delta_nll_per_token"])
            ),
            source_to_candidate_kl_per_token=float(
                aggregate["source_to_candidate_kl_per_token"]
            ),
            top1_agreement_to_source=float(
                aggregate["top1_agreement_to_source"]
            ),
            per_prompt_p90_absolute_delta_nll_per_token=float(
                absolute_tail["p90"]
            ),
            per_prompt_p10_top1_agreement_to_source=float(
                top1_tail["p10"]
            ),
        )

    @staticmethod
    def _geometry(
        values: Mapping[str, float],
    ) -> tuple[float, float, float]:
        source = values["source"]
        candidate = values["candidate"]
        residual = values["residual"]
        relative = (
            math.sqrt(residual / source)
            if source > 0.0
            else 0.0
            if residual == 0.0
            else math.inf
        )
        denominator = math.sqrt(source * candidate)
        cosine = (
            values["dot"] / denominator
            if denominator > 0.0
            else 1.0
            if source == 0.0 and candidate == 0.0
            else 0.0
        )
        return relative, max(0.0, min(1.0, cosine)), math.sqrt(source)

    def finalize(self) -> ProgressiveFidelity:
        candidate = self._behavior(self._candidate.finalize())
        projection_behavior = self._behavior(self._projection.finalize())
        carrier_behavior = self._behavior(self._carrier.finalize())
        modal_total = {
            name: sum(family[name] for family in self._modal.values())
            for name in ("residual", "source", "candidate", "dot")
        }
        projection_total = {
            name: sum(
                family[name]
                for family in self._projection_boundary.values()
            )
            for name in ("residual", "source", "candidate", "dot")
        }
        modal_relative, modal_cosine, _ = self._geometry(modal_total)
        projection_relative, projection_cosine, _ = self._geometry(
            projection_total
        )
        modal_families = tuple(
            self._geometry(values) for values in self._modal.values()
        )
        projection_families = tuple(
            self._geometry(values)
            for values in self._projection_boundary.values()
        )
        if (
            self.valid_rows <= 0
            or not 0 < self.affected_rows <= self.valid_rows
            or self.supervised_tokens <= 0
        ):
            raise ValueError("development evaluation coverage is empty")
        return ProgressiveFidelity(
            candidate_behavior=candidate,
            projection_oracle_behavior=projection_behavior,
            carrier_oracle_behavior=carrier_behavior,
            operator_nrmse=projection_relative,
            boundary_relative_error=modal_relative,
            boundary_cosine=modal_cosine,
            valid_target_coverage=self.affected_rows / self.valid_rows,
            worst_family_boundary_relative_error=max(
                item[0] for item in modal_families
            ),
            worst_family_boundary_cosine=min(
                item[1] for item in modal_families
            ),
            minimum_family_source_modal_signal_l2_norm=min(
                item[2] for item in modal_families
            ),
            projection_full_width_relative_error=projection_relative,
            projection_full_width_cosine=projection_cosine,
            worst_family_projection_relative_error=max(
                item[0] for item in projection_families
            ),
            worst_family_projection_cosine=min(
                item[1] for item in projection_families
            ),
            minimum_family_source_full_width_signal_l2_norm=min(
                item[2] for item in projection_families
            ),
        )


class Gemma3L3L4ProgressiveWorker:
    """Candidate registry plus callbacks for the generic progressive loop."""

    def __init__(
        self,
        *,
        protocol: ProgressiveCompilationProtocol,
        panels: Mapping[DevelopmentRole, GemmaProgressivePanel],
        seed_candidate: ProgressiveCandidate,
        seed_executable: GemmaProgressiveExecutable,
        max_residual_directions: int = 16,
        mutation_lowerer: GemmaMutationLowerer | None = None,
        guard_claim_authority: GemmaGuardClaimAuthority | None = None,
        guard_panel_provider: GemmaGuardPanelProvider | None = None,
        selection_only: bool = False,
        complete_boundary_oracle_atol: float = 1.0e-8,
    ) -> None:
        if not isinstance(protocol, ProgressiveCompilationProtocol):
            raise TypeError("protocol must be ProgressiveCompilationProtocol")
        protocol.validate_integrity()
        if not isinstance(seed_candidate, ProgressiveCandidate):
            raise TypeError("seed_candidate must be ProgressiveCandidate")
        if type(max_residual_directions) is not int or not (
            1 <= max_residual_directions <= 256
        ):
            raise ValueError(
                "max_residual_directions must be an integer in [1, 256]"
            )
        if (
            not math.isfinite(complete_boundary_oracle_atol)
            or complete_boundary_oracle_atol < 0.0
        ):
            raise ValueError(
                "complete_boundary_oracle_atol must be finite and nonnegative"
            )
        if type(selection_only) is not bool:
            raise TypeError("selection_only must be a bool")
        if selection_only:
            if (
                guard_claim_authority is not None
                or guard_panel_provider is not None
            ):
                raise ValueError(
                    "selection-only worker cannot receive guard capabilities"
                )
            eager_guard = False
            expected_roles = {
                "calibration_a_fit",
                "calibration_a_selection",
            }
        else:
            eager_guard = guard_panel_provider is None
            expected_roles = {
                "calibration_a_fit",
                "calibration_a_selection",
                *(("calibration_a_guard",) if eager_guard else ()),
            }
        if not isinstance(panels, Mapping) or set(panels) != expected_roles:
            raise ValueError(
                "panels must contain fit/selection and exactly one eager "
                "guard panel or claim-gated guard provider"
            )
        materialized = dict(panels)
        for role, panel in materialized.items():
            if (
                not isinstance(panel, GemmaProgressivePanel)
                or panel.role != role
            ):
                raise ValueError("panel role binding differs")
            if (
                set(panel.forbidden_manifest_sha256s)
                != set(protocol.forbidden_assessment_manifest_sha256s)
            ):
                raise ValueError(
                    "panel forbidden-manifest set differs from the protocol"
                )
        materialized["calibration_a_fit"].validate_view(protocol.fit_view())
        materialized["calibration_a_selection"].validate_view(
            protocol.selection_view()
        )
        if eager_guard:
            materialized["calibration_a_guard"].validate_view(
                protocol.guard_view()
            )
        for role, panel in materialized.items():
            if (
                panel.binding_sha256
                != protocol.development_role_binding_sha256(role)
            ):
                raise ValueError(
                    f"{role} panel contents differ from the frozen protocol"
                )
        if guard_panel_provider is not None:
            for name in (
                "manifest_sha256",
                "membership_receipt_sha256",
                "preclaim_binding_sha256",
            ):
                _require_sha256(
                    getattr(guard_panel_provider, name, None),
                    label=f"guard provider {name}",
                )
            if (
                guard_panel_provider.manifest_sha256
                != protocol.corpus.guard_manifest_sha256
                or guard_panel_provider.example_count
                != protocol.corpus.guard_example_count
                or guard_panel_provider.family_ids
                != protocol.corpus.guard_family_ids
                or guard_panel_provider.preclaim_binding_sha256
                != protocol.development_role_binding_sha256(
                    "calibration_a_guard"
                )
                or not callable(
                    getattr(
                        guard_panel_provider,
                        "open_after_claim",
                        None,
                    )
                )
            ):
                raise ValueError(
                    "guard provider differs from the frozen protocol"
                )
        all_examples: set[str] = set()
        all_inputs: set[str] = set()
        all_calibration: set[str] = set()
        for panel in materialized.values():
            for example in panel.examples:
                if (
                    example.example_id in all_examples
                    or example.model_inputs_sha256 in all_inputs
                    or example.calibration_sha256 in all_calibration
                ):
                    raise ValueError(
                        "Calibration-A roles must not share examples or "
                        "model inputs"
                    )
                all_examples.add(example.example_id)
                all_inputs.add(example.model_inputs_sha256)
                all_calibration.add(example.calibration_sha256)
        self._protocol = protocol
        self._panels = materialized
        self._max_residual_directions = max_residual_directions
        self._mutation_lowerer = mutation_lowerer
        self._guard_claim_authority = guard_claim_authority
        self._guard_panel_provider = guard_panel_provider
        self._selection_only = selection_only
        self._complete_boundary_oracle_atol = (
            complete_boundary_oracle_atol
        )
        self._executables: dict[str, GemmaProgressiveExecutable] = {}
        self._candidate_receipts: dict[str, str] = {}
        self._analysis_by_map: dict[str, GemmaCarrierResidualAnalysis] = {}
        self._analysis_identity_by_map: dict[str, str] = {}
        self._guard_claim_sha256: str | None = None
        if (
            seed_candidate.mutation_kind != "seed"
            or seed_candidate.artifact_sha256
            != protocol.seed_candidate_artifact_sha256
            or seed_candidate.execution_sha256
            != protocol.seed_candidate_execution_sha256
            or seed_candidate.runtime_binding_sha256
            != protocol.seed_runtime_binding_sha256
            or seed_candidate.resources.receipt_sha256
            != protocol.seed_resource_receipt_sha256
        ):
            raise ValueError(
                "seed candidate differs from the frozen worker protocol"
            )
        if isinstance(
            seed_executable,
            LegacyRank64GemmaProgressiveExecutable,
        ):
            reasons = set(seed_candidate.resources.incomplete_cost_reasons)
            required = set(LEGACY_RANK64_INCOMPLETE_COST_REASONS)
            if (
                seed_candidate.resources.cost_complete
                or not required.issubset(reasons)
            ):
                raise ValueError(
                    "legacy rank-64 measurement runtime requires incomplete "
                    "resource accounting with all canonical reasons"
                )
        self._register_executable(seed_candidate, seed_executable)

    @property
    def protocol(self) -> ProgressiveCompilationProtocol:
        return self._protocol

    def _register_executable(
        self,
        candidate: ProgressiveCandidate,
        executable: GemmaProgressiveExecutable,
    ) -> None:
        if (
            executable.candidate_artifact_sha256
            != candidate.artifact_sha256
            or executable.candidate_execution_sha256
            != candidate.execution_sha256
            or executable.runtime_binding_sha256
            != candidate.runtime_binding_sha256
        ):
            raise ValueError(
                "candidate receipt and executable identity differ"
            )
        if candidate.artifact_sha256 in self._executables:
            raise ValueError("candidate executable is already registered")
        self._executables[candidate.artifact_sha256] = executable
        self._candidate_receipts[candidate.artifact_sha256] = (
            candidate.receipt_sha256
        )

    def _executable(
        self,
        candidate: ProgressiveCandidate,
    ) -> GemmaProgressiveExecutable:
        try:
            executable = self._executables[candidate.artifact_sha256]
        except KeyError as error:
            raise ValueError(
                "candidate has no registered immutable executable"
            ) from error
        if (
            executable.candidate_artifact_sha256
            != candidate.artifact_sha256
            or executable.candidate_execution_sha256
            != candidate.execution_sha256
            or executable.runtime_binding_sha256
            != candidate.runtime_binding_sha256
            or self._candidate_receipts[candidate.artifact_sha256]
            != candidate.receipt_sha256
        ):
            raise RuntimeError(
                "registered candidate receipt or executable identity drifted"
            )
        return executable

    def _evaluate(
        self,
        *,
        candidate: ProgressiveCandidate,
        panel: GemmaProgressivePanel,
        development_role: Literal[
            "calibration_a_selection",
            "calibration_a_guard",
        ],
        challenger_receipt_sha256: str | None,
        guard_claim_sha256: str | None,
    ) -> CandidateEvaluation:
        executable = self._executable(candidate)
        accumulator = _BoundaryAccumulator(panel)
        for example in panel.examples:
            observation = executable.observe(
                example,
                collect_carrier_fisher=False,
            )
            if (
                observation.example_id != example.example_id
                or observation.family_id != example.family_id
                or observation.model_inputs_sha256
                != example.model_inputs_sha256
                or observation.runtime_binding_sha256
                != candidate.runtime_binding_sha256
            ):
                raise ValueError(
                    "executable observation differs from the panel/candidate"
                )
            accumulator.add(observation)
        fidelity = accumulator.finalize()
        coverage = DevelopmentEvaluationCoverage(
            manifest_sha256=panel.manifest_sha256,
            expected_example_count=len(panel.examples),
            observed_example_count=len(accumulator.observation_sha256s),
            expected_family_ids=panel.family_ids,
            observed_family_ids=panel.family_ids,
            supervised_token_count=accumulator.supervised_tokens,
            membership_receipt_sha256=(
                panel.membership_receipt_sha256
            ),
            model_inputs_receipt_sha256=(
                panel.model_inputs_receipt_sha256
            ),
            complete=True,
        )
        evaluation_artifact = _sha256(
            "candidate-evaluation",
            {
                "protocol_sha256": self._protocol.artifact_sha256,
                "development_role": development_role,
                "manifest_sha256": panel.manifest_sha256,
                "candidate_receipt_sha256": candidate.receipt_sha256,
                "observation_sha256s": tuple(
                    sorted(accumulator.observation_sha256s)
                ),
                "coverage": coverage.to_dict(),
                "fidelity": fidelity.to_dict(),
                "guard_claim_sha256": guard_claim_sha256,
                "raw_prompt_or_activation_payload_in_receipt": False,
            },
        )
        return CandidateEvaluation(
            protocol_sha256=self._protocol.artifact_sha256,
            development_role=development_role,
            manifest_sha256=panel.manifest_sha256,
            candidate_artifact_sha256=candidate.artifact_sha256,
            candidate_receipt_sha256=candidate.receipt_sha256,
            evaluation_artifact_sha256=evaluation_artifact,
            coverage=coverage,
            fidelity=fidelity,
            resources=candidate.resources,
            challenger_receipt_sha256=challenger_receipt_sha256,
            guard_claim_sha256=guard_claim_sha256,
        )

    def evaluate_selection(
        self,
        candidate: ProgressiveCandidate,
        selection: SelectionDevelopmentView,
    ) -> CandidateEvaluation:
        panel = self._panels["calibration_a_selection"]
        panel.validate_view(selection)
        if selection.protocol_sha256 != self._protocol.artifact_sha256:
            raise ValueError("selection view belongs to another protocol")
        return self._evaluate(
            candidate=candidate,
            panel=panel,
            development_role="calibration_a_selection",
            challenger_receipt_sha256=None,
            guard_claim_sha256=None,
        )

    def evaluate_guard(
        self,
        challenger: FrozenCalibrationAChallenger,
        guard: GuardDevelopmentView,
    ) -> CandidateEvaluation:
        if self._selection_only:
            raise GemmaGuardAuthorityRequiredError(
                "selection-only worker has no guard capability"
            )
        if not isinstance(challenger, FrozenCalibrationAChallenger):
            raise TypeError(
                "challenger must be a FrozenCalibrationAChallenger"
            )
        if self._guard_claim_authority is None:
            raise GemmaGuardAuthorityRequiredError(
                "Gemma A-guard evaluation requires a durable claim-first "
                "authority"
            )
        if self._guard_claim_sha256 is not None:
            raise GemmaGuardAuthorityRequiredError(
                "this worker campaign has already consumed its A guard"
            )
        if guard.protocol_sha256 != self._protocol.artifact_sha256:
            raise ValueError("guard view belongs to another protocol")
        candidate = challenger.candidate
        if (
            challenger.protocol_sha256 != self._protocol.artifact_sha256
            or challenger.selection_evaluation.manifest_sha256
            != self._protocol.corpus.selection_manifest_sha256
        ):
            raise ValueError(
                "frozen challenger belongs to another worker campaign"
            )
        self._executable(candidate)
        claim = self._guard_claim_authority.claim(
            protocol_sha256=self._protocol.artifact_sha256,
            guard_manifest_sha256=(
                self._protocol.corpus.guard_manifest_sha256
            ),
            challenger_receipt_sha256=challenger.receipt_sha256,
        )
        self._guard_claim_sha256 = _require_sha256(
            claim,
            label="guard claim",
        )
        if self._guard_panel_provider is None:
            panel = self._panels["calibration_a_guard"]
        else:
            panel = self._guard_panel_provider.open_after_claim(
                self._guard_claim_sha256
            )
            if not isinstance(panel, GemmaProgressivePanel):
                raise TypeError(
                    "guard provider must materialize a Gemma panel"
                )
            if (
                panel.membership_receipt_sha256
                != self._guard_panel_provider.membership_receipt_sha256
            ):
                raise ValueError(
                    "materialized guard membership differs from preclaim"
                )
            opened_examples = {
                example.example_id
                for role in (
                    "calibration_a_fit",
                    "calibration_a_selection",
                )
                for example in self._panels[role].examples
            }
            opened_inputs = {
                example.model_inputs_sha256
                for role in (
                    "calibration_a_fit",
                    "calibration_a_selection",
                )
                for example in self._panels[role].examples
            }
            opened_calibration = {
                example.calibration_sha256
                for role in (
                    "calibration_a_fit",
                    "calibration_a_selection",
                )
                for example in self._panels[role].examples
            }
            if any(
                example.example_id in opened_examples
                or example.model_inputs_sha256 in opened_inputs
                or example.calibration_sha256 in opened_calibration
                for example in panel.examples
            ):
                raise ValueError(
                    "claim-opened guard overlaps fit or selection data"
                )
        panel.validate_view(guard)
        return self._evaluate(
            candidate=candidate,
            panel=panel,
            development_role="calibration_a_guard",
            challenger_receipt_sha256=challenger.receipt_sha256,
            guard_claim_sha256=self._guard_claim_sha256,
        )

    def map_residual(
        self,
        candidate: ProgressiveCandidate,
        fit: FitDevelopmentView,
    ) -> ResidualMap:
        panel = self._panels["calibration_a_fit"]
        panel.validate_view(fit)
        if fit.protocol_sha256 != self._protocol.artifact_sha256:
            raise ValueError("fit view belongs to another protocol")
        executable = self._executable(candidate)
        family_covariances: dict[str, Tensor] = {}
        family_fishers: dict[str, Tensor] = {}
        x4_family_covariances: dict[str, Tensor] = {}
        x4_family_fishers: dict[str, Tensor] = {}
        family_rows: dict[str, int] = {}
        family_examples: dict[str, int] = {}
        observation_ids: list[str] = []
        fit_sequences: list[GemmaTwoHeadFitSequence] = []
        saw_two_head_trace: bool | None = None
        runtime_binding: str | None = None
        width: int | None = None
        maximum_oracle_error = 0.0
        for example in panel.examples:
            observation = executable.observe(
                example,
                collect_carrier_fisher=True,
            )
            residual = observation.carrier_residual_rows
            gradient = observation.carrier_loss_gradient_rows
            oracle_error = (
                observation.complete_boundary_oracle_max_abs_logit_error
            )
            if (
                residual is None
                or gradient is None
                or oracle_error is None
                or observation.model_inputs_sha256
                != example.model_inputs_sha256
                or observation.example_id != example.example_id
                or observation.family_id != example.family_id
                or observation.runtime_binding_sha256
                != candidate.runtime_binding_sha256
            ):
                raise ValueError(
                    "fit observation lacks its carrier/NLL-gradient binding"
                )
            residual = residual.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            gradient = gradient.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            if runtime_binding is None:
                runtime_binding = observation.runtime_binding_sha256
                width = residual.shape[1]
            elif (
                observation.runtime_binding_sha256 != runtime_binding
                or residual.shape[1] != width
            ):
                raise ValueError("fit executable geometry changed mid-panel")
            row_residual_square = residual.square().sum(dim=1)
            row_gradient_square = gradient.square().sum(dim=1)
            alignment = (residual * gradient).sum(dim=1).square()
            alignment = alignment / (
                row_residual_square * row_gradient_square + 1.0e-30
            )
            fisher_weight = 1.0 + alignment.clamp(min=0.0, max=1.0)
            weighted = residual * fisher_weight.sqrt().unsqueeze(1)
            family = example.family_id
            covariance = weighted.T @ weighted
            fisher = gradient.T @ gradient
            if family not in family_covariances:
                family_covariances[family] = (
                    covariance / residual.shape[0]
                )
                family_fishers[family] = fisher / residual.shape[0]
                family_rows[family] = residual.shape[0]
                family_examples[family] = 1
            else:
                family_covariances[family] += (
                    covariance / residual.shape[0]
                )
                family_fishers[family] += fisher / residual.shape[0]
                family_rows[family] += residual.shape[0]
                family_examples[family] += 1
            fit_sequence = observation.two_head_fit_sequence
            has_two_head_trace = fit_sequence is not None
            if saw_two_head_trace is None:
                saw_two_head_trace = has_two_head_trace
            elif saw_two_head_trace != has_two_head_trace:
                raise ValueError(
                    "fit panel mixed legacy and two-head observations"
                )
            if fit_sequence is not None:
                x4_residual = fit_sequence.x4_residual_rows.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
                x4_gradient = fit_sequence.x4_loss_gradient[
                    fit_sequence.target_affected_mask
                ].detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )
                x4_residual_square = x4_residual.square().sum(dim=1)
                x4_gradient_square = x4_gradient.square().sum(dim=1)
                x4_alignment = (
                    (x4_residual * x4_gradient).sum(dim=1).square()
                    / (
                        x4_residual_square * x4_gradient_square
                        + 1.0e-30
                    )
                )
                x4_weight = 1.0 + x4_alignment.clamp(
                    min=0.0,
                    max=1.0,
                )
                x4_weighted = (
                    x4_residual * x4_weight.sqrt().unsqueeze(1)
                )
                x4_covariance = x4_weighted.T @ x4_weighted
                x4_fisher = x4_gradient.T @ x4_gradient
                if family not in x4_family_covariances:
                    x4_family_covariances[family] = (
                        x4_covariance / x4_residual.shape[0]
                    )
                    x4_family_fishers[family] = (
                        x4_fisher / x4_residual.shape[0]
                    )
                else:
                    x4_family_covariances[family] += (
                        x4_covariance / x4_residual.shape[0]
                    )
                    x4_family_fishers[family] += (
                        x4_fisher / x4_residual.shape[0]
                    )
                fit_sequences.append(fit_sequence.detached_copy())
            maximum_oracle_error = max(maximum_oracle_error, oracle_error)
            observation_ids.append(observation.artifact_sha256)
        if runtime_binding is None or width is None:
            raise ValueError("fit panel produced no residual observations")
        if maximum_oracle_error > self._complete_boundary_oracle_atol:
            raise RuntimeError(
                "layer.4.output failed the complete-boundary oracle: "
                f"{maximum_oracle_error:.6g} > "
                f"{self._complete_boundary_oracle_atol:.6g}"
            )
        macro_covariance = sum(
            family_covariances[family] / family_examples[family]
            for family in sorted(family_covariances)
        ) / len(family_covariances)
        macro_fisher = sum(
            family_fishers[family] / family_examples[family]
            for family in sorted(family_fishers)
        ) / len(family_fishers)
        macro_covariance = (
            (macro_covariance + macro_covariance.T) * 0.5
        ).contiguous()
        eigenvalues, eigenvectors = torch.linalg.eigh(macro_covariance)
        order = torch.argsort(eigenvalues, descending=True)
        count = min(
            self._max_residual_directions,
            width,
            int((eigenvalues > 0.0).sum()),
        )
        if count <= 0:
            count = 1
        selected_values = (
            eigenvalues.index_select(0, order[:count])
            .clamp_min(0.0)
            .contiguous()
        )
        total_residual_energy = float(
            eigenvalues.clamp_min(0.0).sum()
        )
        directions = (
            eigenvectors.index_select(1, order[:count])
            .T.contiguous()
        )
        pivot_indices = directions.abs().argmax(dim=1)
        pivot_values = directions.gather(
            1,
            pivot_indices.unsqueeze(1),
        ).squeeze(1)
        directions = (
            directions
            * torch.where(
                pivot_values < 0.0,
                -torch.ones_like(pivot_values),
                torch.ones_like(pivot_values),
            ).unsqueeze(1)
        ).contiguous()
        loss_couplings = torch.einsum(
            "kw,wx,kx->k",
            directions,
            macro_fisher,
            directions,
        ).clamp_min(0.0)
        x4_directions: Tensor | None = None
        x4_selected_values: Tensor | None = None
        x4_loss_couplings: Tensor | None = None
        x4_total_residual_energy: float | None = None
        if saw_two_head_trace:
            if (
                set(x4_family_covariances) != set(family_rows)
                or set(x4_family_fishers) != set(family_rows)
                or len(fit_sequences) != len(panel.examples)
            ):
                raise ValueError("two-head fit trace coverage is incomplete")
            x4_macro_covariance = sum(
                x4_family_covariances[family]
                / family_examples[family]
                for family in sorted(x4_family_covariances)
            ) / len(x4_family_covariances)
            x4_macro_fisher = sum(
                x4_family_fishers[family]
                / family_examples[family]
                for family in sorted(x4_family_fishers)
            ) / len(x4_family_fishers)
            x4_macro_covariance = (
                (x4_macro_covariance + x4_macro_covariance.T) * 0.5
            ).contiguous()
            x4_eigenvalues, x4_eigenvectors = torch.linalg.eigh(
                x4_macro_covariance
            )
            x4_order = torch.argsort(x4_eigenvalues, descending=True)
            x4_count = min(
                self._max_residual_directions,
                width,
                int((x4_eigenvalues > 0.0).sum()),
            )
            if x4_count <= 0:
                x4_count = 1
            x4_selected_values = (
                x4_eigenvalues.index_select(0, x4_order[:x4_count])
                .clamp_min(0.0)
                .contiguous()
            )
            x4_total_residual_energy = float(
                x4_eigenvalues.clamp_min(0.0).sum()
            )
            x4_directions = (
                x4_eigenvectors.index_select(1, x4_order[:x4_count])
                .T.contiguous()
            )
            x4_pivots = x4_directions.abs().argmax(dim=1)
            x4_pivot_values = x4_directions.gather(
                1,
                x4_pivots.unsqueeze(1),
            ).squeeze(1)
            x4_directions = (
                x4_directions
                * torch.where(
                    x4_pivot_values < 0.0,
                    -torch.ones_like(x4_pivot_values),
                    torch.ones_like(x4_pivot_values),
                ).unsqueeze(1)
            ).contiguous()
            x4_loss_couplings = torch.einsum(
                "kw,wx,kx->k",
                x4_directions,
                x4_macro_fisher,
                x4_directions,
            ).clamp_min(0.0)
        analysis = GemmaCarrierResidualAnalysis(
            protocol_sha256=self._protocol.artifact_sha256,
            fit_manifest_sha256=panel.manifest_sha256,
            candidate_artifact_sha256=candidate.artifact_sha256,
            candidate_receipt_sha256=candidate.receipt_sha256,
            runtime_binding_sha256=runtime_binding,
            location=_H4_SITE,
            directions=directions.to(
                device="cpu",
                dtype=torch.float64,
            ),
            residual_eigenvalues=selected_values.to(
                device="cpu",
                dtype=torch.float64,
            ),
            loss_couplings=loss_couplings.to(
                device="cpu",
                dtype=torch.float64,
            ),
            total_residual_energy=total_residual_energy,
            family_row_counts=tuple(sorted(family_rows.items())),
            observation_sha256s=tuple(sorted(observation_ids)),
            complete_boundary_oracle_max_abs_logit_error=(
                maximum_oracle_error
            ),
            x4_directions=(
                None
                if x4_directions is None
                else x4_directions.to(
                    device="cpu",
                    dtype=torch.float64,
                )
            ),
            x4_residual_eigenvalues=(
                None
                if x4_selected_values is None
                else x4_selected_values.to(
                    device="cpu",
                    dtype=torch.float64,
                )
            ),
            x4_loss_couplings=(
                None
                if x4_loss_couplings is None
                else x4_loss_couplings.to(
                    device="cpu",
                    dtype=torch.float64,
                )
            ),
            x4_total_residual_energy=x4_total_residual_energy,
            fit_sequences=tuple(
                sorted(
                    fit_sequences,
                    key=lambda value: (
                        value.family_id,
                        value.example_id,
                    ),
                )
            ),
        )
        residual_map = analysis.residual_map(iteration=candidate.iteration)
        self._analysis_by_map[residual_map.receipt_sha256] = analysis
        self._analysis_identity_by_map[residual_map.receipt_sha256] = (
            analysis.artifact_sha256
        )
        return residual_map

    def residual_analysis(
        self,
        residual_map: ResidualMap,
    ) -> GemmaCarrierResidualAnalysis:
        try:
            analysis = self._analysis_by_map[residual_map.receipt_sha256]
        except KeyError as error:
            raise KeyError(
                "residual map was not produced by this worker"
            ) from error
        expected = self._analysis_identity_by_map[
            residual_map.receipt_sha256
        ]
        if (
            expected != residual_map.analysis_artifact_sha256
            or analysis.artifact_sha256 != expected
        ):
            raise RuntimeError("residual-map analysis registry drifted")
        return analysis.detached_copy()

    def propose_mutations(
        self,
        candidate: ProgressiveCandidate,
        residual_map: ResidualMap,
        phase: ProgressivePhase,
    ) -> Sequence[MutationProposal]:
        if self._mutation_lowerer is None:
            raise GemmaMutationLoweringUnavailableError(
                "residual analysis is ready, but no candidate-bound Gemma "
                "mutation lowerer is registered"
            )
        analysis = self.residual_analysis(residual_map)
        proposals = self._mutation_lowerer.propose(
            parent=candidate,
            residual_map=residual_map,
            analysis=analysis,
            phase=phase,
        )
        if analysis.artifact_sha256 != residual_map.analysis_artifact_sha256:
            raise RuntimeError("mutation lowerer changed its analysis input")
        return proposals

    def build_candidate(
        self,
        parent: ProgressiveCandidate,
        proposal: MutationProposal,
    ) -> ProgressiveCandidate:
        if self._mutation_lowerer is None:
            raise GemmaMutationLoweringUnavailableError(
                "candidate construction requires a candidate-bound Gemma "
                "mutation lowerer"
            )
        try:
            analysis = self._analysis_by_map[
                proposal.residual_map_sha256
            ]
        except KeyError as error:
            raise ValueError(
                "proposal does not reference this worker's residual map"
            ) from error
        expected_analysis = self._analysis_identity_by_map[
            proposal.residual_map_sha256
        ]
        if analysis.artifact_sha256 != expected_analysis:
            raise RuntimeError("residual analysis registry drifted")
        lowerer_analysis = analysis.detached_copy()
        child, executable = self._mutation_lowerer.build(
            parent=parent,
            proposal=proposal,
            analysis=lowerer_analysis,
        )
        if lowerer_analysis.artifact_sha256 != expected_analysis:
            raise RuntimeError("mutation lowerer changed its analysis input")
        self._register_executable(child, executable)
        return child


__all__ = [
    "DevelopmentRole",
    "Gemma3L3L4ProgressiveWorker",
    "GemmaCarrierResidualAnalysis",
    "GemmaGuardAuthorityRequiredError",
    "GemmaGuardClaimAuthority",
    "GemmaGuardPanelProvider",
    "GemmaL3L4DevelopmentObservation",
    "GemmaMutationLowerer",
    "GemmaMutationLoweringUnavailableError",
    "GemmaProgressiveExample",
    "GemmaProgressiveExecutable",
    "GemmaProgressivePanel",
    "GemmaProgressiveWorkerError",
    "GemmaTwoHeadFitSequence",
    "LEGACY_RANK64_INCOMPLETE_COST_REASONS",
    "LegacyRank64GemmaProgressiveExecutable",
    "gemma_progressive_panel_membership_receipt_sha256",
    "make_gemma_progressive_panel",
]
