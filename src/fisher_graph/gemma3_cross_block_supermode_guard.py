"""Fresh-data guard for the directed Gemma cross-block supermode.

The guard evaluates four frozen-model paths on the same replayable batches:
native execution, native coordinate deletion, native coordinate replacement,
and the physically row-pruned full-model executor.  It measures whether the
compiled path realizes the replacement oracle and whether that replacement
recovers deletion damage.  Passing this diagnostic never grants execution,
compilation, calibration-B, validation, or test authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .compiler.calibration import CalibrationBatch
from .gemma3_cross_block_replacement_oracle import (
    _capture_sites,
    _run_with_coordinate_intervention,
    _streaming_logit_sums,
    validate_gemma3_cross_block_replacement_oracle_artifact,
)
from .gemma3_cross_block_row_pruned_executor import (
    Gemma3CrossBlockModelExecutor,
)
from .structured_mlp_cross_block_plan import (
    StructuredMLPCrossBlockPlan,
    UnresolvedCrossBlockCarryProposal,
)


GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_SCHEMA = (
    "fisher_graph.gemma3_cross_block_supermode_guard"
)
GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_FORMAT_VERSION = 1
GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_THRESHOLDS_SCHEMA = (
    "fisher_graph.gemma3_cross_block_supermode_guard_thresholds"
)
GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_THRESHOLDS_FORMAT_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_DOMAIN = (
    b"fisher_graph.gemma3_cross_block_supermode_guard.v1\0"
)
_THRESHOLD_DOMAIN = (
    b"fisher_graph.gemma3_cross_block_supermode_guard.thresholds.v1\0"
)
_STREAM_DOMAIN = (
    b"fisher_graph.gemma3_cross_block_supermode_guard.stream.v1\0"
)
_SURFACES = (
    "consumer_mlp_output",
    "window_output",
    "final_logits",
)
_CANDIDATES = ("coordinate_ablation", "coordinate_replacement", "compiled")
_SAFETY = {
    "contains_source_model_weights": False,
    "contains_executable_weights": False,
    "contains_optimizer_state": False,
    "contains_corpus_rows": False,
    "contains_activation_rows": False,
    "contains_prompt_text": False,
    "guard_evaluation_only": True,
    "candidate_remains_experimental": True,
    "authorizes_further_intervention": False,
    "authorizes_compilation": False,
    "authorizes_executor_construction": False,
    "authorizes_execution": False,
    "authorizes_calibration_a_guard": False,
    "authorizes_guard": False,
    "authorizes_b": False,
    "authorizes_calibration_b": False,
    "authorizes_validation": False,
    "authorizes_test": False,
    "authorizes_deployment": False,
}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes = _ARTIFACT_DOMAIN) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_json_bytes(value))
    return digest.hexdigest()


def _assert_json_safe(value: object, *, path: str = "value") -> None:
    if isinstance(value, Tensor):
        raise RuntimeError(f"{path} contains a Tensor")
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"{path} contains an invalid mapping key")
            _assert_json_safe(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_json_safe(nested, path=f"{path}[{index}]")
        return
    raise RuntimeError(f"{path} contains unsupported type {type(value)!r}")


def _tensor_digest_update(digest: object, tensor: Tensor) -> None:
    if not isinstance(digest, type(hashlib.sha256())):
        raise TypeError("digest must be a SHA-256 object")
    value = tensor.detach().to(device="cpu").contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())


def _validated_examples_and_labels(
    batches: Sequence[CalibrationBatch],
    family_by_example: Mapping[str, str],
    length_by_example: Mapping[str, str],
) -> tuple[tuple[CalibrationBatch, ...], tuple[str, ...]]:
    if isinstance(batches, (str, bytes)) or not isinstance(
        batches,
        Sequence,
    ):
        raise TypeError("guard batches must be a sequence")
    frozen = tuple(batches)
    if not frozen or any(
        not isinstance(batch, CalibrationBatch) for batch in frozen
    ):
        raise ValueError("guard requires nonempty CalibrationBatch values")
    if not isinstance(family_by_example, Mapping) or not isinstance(
        length_by_example,
        Mapping,
    ):
        raise TypeError("family and length labels must be mappings")
    example_ids: list[str] = []
    for batch in frozen:
        if batch.example_ids is None:
            raise ValueError("guard batches require explicit example_ids")
        if any(
            tensor.requires_grad
            for tensor in (
                *batch.model_inputs.values(),
                batch.targets,
                batch.valid_positions,
            )
        ):
            raise ValueError("guard batch tensors must be frozen")
        example_ids.extend(batch.example_ids)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("guard example_ids must be globally unique")
    expected = set(example_ids)
    if set(family_by_example) != expected or set(length_by_example) != expected:
        raise ValueError(
            "family and length labels must exactly cover guard examples"
        )
    for label, values in (
        ("family", family_by_example),
        ("length", length_by_example),
    ):
        if any(
            not isinstance(value, str) or not value
            for value in values.values()
        ):
            raise ValueError(f"{label} labels must be nonempty strings")
    return frozen, tuple(example_ids)


def gemma3_cross_block_guard_stream_sha256(
    batches: Sequence[CalibrationBatch],
    *,
    family_by_example: Mapping[str, str],
    length_by_example: Mapping[str, str],
) -> str:
    """Hash materialized tensors, batch grouping, identities, and strata."""

    frozen, _ = _validated_examples_and_labels(
        batches,
        family_by_example,
        length_by_example,
    )
    digest = hashlib.sha256()
    digest.update(_STREAM_DOMAIN)
    for batch_index, batch in enumerate(frozen):
        digest.update(f"batch:{batch_index};".encode("ascii"))
        assert batch.example_ids is not None
        for example_id in batch.example_ids:
            encoded = example_id.encode("utf-8")
            digest.update(f"id:{len(encoded)}:".encode("ascii"))
            digest.update(encoded)
            for kind, value in (
                ("family", family_by_example[example_id]),
                ("length", length_by_example[example_id]),
            ):
                label = value.encode("utf-8")
                digest.update(
                    f"{kind}:{len(label)}:".encode("ascii")
                )
                digest.update(label)
        for name in sorted(batch.model_inputs):
            digest.update(f"input:{name};".encode("utf-8"))
            _tensor_digest_update(digest, batch.model_inputs[name])
        digest.update(b"targets;")
        _tensor_digest_update(digest, batch.targets)
        digest.update(b"valid_positions;")
        _tensor_digest_update(digest, batch.valid_positions)
        for name in sorted(batch.shared_input_names):
            digest.update(f"shared:{name};".encode("utf-8"))
    return digest.hexdigest()


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class Gemma3CrossBlockSupermodeGuardThresholds:
    """Immutable, content-hashed thresholds fixed before guard access."""

    source_model_fingerprint: str
    source_execution_fingerprint: str
    source_plan_artifact_sha256: str
    source_replacement_oracle_artifact_sha256: str
    guard_stream_sha256: str
    compiled_oracle_surface_maximum_absolute_error: float
    compiled_oracle_behavior_maximum_absolute_difference: float
    minimum_surface_recovery_fraction: float
    maximum_absolute_delta_nll_per_token: float
    maximum_teacher_kl_per_token: float
    minimum_top1_agreement: float
    minimum_nll_recovery_fraction: float
    minimum_kl_recovery_fraction: float
    minimum_top1_recovery_fraction: float
    apply_to_each_family: bool = True
    apply_to_each_length: bool = True
    expected_removed_parameters: int = 1280
    expected_removed_linear_macs_per_valid_token: int = 1280
    fixed_before_guard: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("source_model_fingerprint", self.source_model_fingerprint),
            (
                "source_execution_fingerprint",
                self.source_execution_fingerprint,
            ),
            (
                "source_plan_artifact_sha256",
                self.source_plan_artifact_sha256,
            ),
            (
                "source_replacement_oracle_artifact_sha256",
                self.source_replacement_oracle_artifact_sha256,
            ),
            ("guard_stream_sha256", self.guard_stream_sha256),
        ):
            _require_sha256(value, label=label)
        nonnegative = (
            "compiled_oracle_surface_maximum_absolute_error",
            "compiled_oracle_behavior_maximum_absolute_difference",
            "maximum_absolute_delta_nll_per_token",
            "maximum_teacher_kl_per_token",
        )
        for name in nonnegative:
            if _finite(getattr(self, name), label=name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        recoveries = (
            "minimum_surface_recovery_fraction",
            "minimum_nll_recovery_fraction",
            "minimum_kl_recovery_fraction",
            "minimum_top1_recovery_fraction",
        )
        for name in recoveries:
            if _finite(getattr(self, name), label=name) > 1:
                raise ValueError(f"{name} cannot exceed one")
        top1 = _finite(
            self.minimum_top1_agreement,
            label="minimum_top1_agreement",
        )
        if not 0 <= top1 <= 1:
            raise ValueError("minimum_top1_agreement must lie in [0, 1]")
        if (
            type(self.apply_to_each_family) is not bool
            or type(self.apply_to_each_length) is not bool
            or self.expected_removed_parameters != 1280
            or self.expected_removed_linear_macs_per_valid_token != 1280
            or self.fixed_before_guard is not True
        ):
            raise ValueError(
                "guard threshold protocol or exact savings drifted"
            )

    @property
    def artifact_sha256(self) -> str:
        return _json_sha256(
            self._payload(),
            domain=_THRESHOLD_DOMAIN,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_THRESHOLDS_SCHEMA,
            "format_version": (
                GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_THRESHOLDS_FORMAT_VERSION
            ),
            **asdict(self),
        }

    def metadata(self) -> dict[str, object]:
        payload = self._payload()
        return {**payload, "artifact_sha256": self.artifact_sha256}


class _PairedSurfaceTotals:
    def __init__(self) -> None:
        self.value_count = 0
        self.square_error = 0.0
        self.reference_square = 0.0
        self.maximum_absolute_error = 0.0

    def update(
        self,
        reference: Tensor,
        candidate: Tensor,
        valid_positions: Tensor,
    ) -> None:
        if (
            reference.shape != candidate.shape
            or reference.ndim != 3
            or reference.shape[:2] != valid_positions.shape
        ):
            raise ValueError("paired guard surface shapes are invalid")
        if valid_positions.dtype is not torch.bool or not valid_positions.any():
            raise ValueError("guard surface mask must select positions")
        left = reference.detach().double()[valid_positions]
        right = candidate.detach().double()[valid_positions]
        difference = right - left
        self.value_count += difference.numel()
        self.square_error += float(difference.square().sum().item())
        self.reference_square += float(left.square().sum().item())
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(difference.abs().max().item()),
        )

    def finish(self) -> dict[str, object]:
        if self.value_count <= 0:
            raise ValueError("guard surface metric observed no values")
        floor = torch.finfo(torch.float64).eps
        return {
            "value_count": self.value_count,
            "rmse": math.sqrt(self.square_error / self.value_count),
            "nrmse": math.sqrt(
                self.square_error / max(self.reference_square, floor)
            ),
            "maximum_absolute_error": self.maximum_absolute_error,
            "square_error": self.square_error,
            "reference_square": self.reference_square,
        }


class _CandidateTotals:
    def __init__(self) -> None:
        self.supervised_tokens = 0
        self.native_nll = 0.0
        self.candidate_nll = 0.0
        self.teacher_kl = 0.0
        self.top1_matches = 0
        self.surfaces = {
            name: _PairedSurfaceTotals() for name in _SURFACES
        }

    def update(
        self,
        *,
        native_logits: Tensor,
        candidate_logits: Tensor,
        targets: Tensor,
        valid_positions: Tensor,
        native_surfaces: Mapping[str, Tensor],
        candidate_surfaces: Mapping[str, Tensor],
        ignore_index: int,
    ) -> None:
        targets = targets.to(device=native_logits.device)
        valid = valid_positions.to(
            device=native_logits.device,
            dtype=torch.bool,
        )
        supervised = targets != ignore_index
        if (
            targets.shape != native_logits.shape[:2]
            or candidate_logits.shape != native_logits.shape
            or (supervised & ~valid).any()
            or not supervised.any()
        ):
            raise ValueError("guard target/logit alignment is invalid")
        native_nll, candidate_nll, kl, matches = _streaming_logit_sums(
            native_logits[supervised],
            candidate_logits[supervised],
            targets[supervised],
        )
        count = int(supervised.sum().item())
        self.supervised_tokens += count
        self.native_nll += native_nll
        self.candidate_nll += candidate_nll
        self.teacher_kl += kl
        self.top1_matches += matches
        for name in _SURFACES:
            self.surfaces[name].update(
                native_surfaces[name],
                candidate_surfaces[name],
                valid.to(device=native_surfaces[name].device),
            )

    def finish(self) -> dict[str, object]:
        count = self.supervised_tokens
        if count <= 0:
            raise ValueError("guard behavior metric observed no targets")
        return {
            "supervised_token_count": count,
            "native_mean_nll": self.native_nll / count,
            "candidate_mean_nll": self.candidate_nll / count,
            "delta_nll_per_token": (
                self.candidate_nll - self.native_nll
            )
            / count,
            "teacher_kl_per_token": self.teacher_kl / count,
            "top1_agreement": self.top1_matches / count,
            "surfaces": {
                name: metric.finish()
                for name, metric in self.surfaces.items()
            },
        }


def _recovery(numerator: float, deletion: float) -> float | None:
    floor = torch.finfo(torch.float64).eps
    if deletion <= floor:
        return None
    return 1.0 - numerator / deletion


class _StratumTotals:
    def __init__(self) -> None:
        self.candidates = {
            name: _CandidateTotals() for name in _CANDIDATES
        }
        self.compiled_exactness = {
            name: _PairedSurfaceTotals() for name in _SURFACES
        }

    def update(
        self,
        *,
        native_logits: Tensor,
        ablation_logits: Tensor,
        replacement_logits: Tensor,
        compiled_logits: Tensor,
        targets: Tensor,
        valid_positions: Tensor,
        native_surfaces: Mapping[str, Tensor],
        ablation_surfaces: Mapping[str, Tensor],
        replacement_surfaces: Mapping[str, Tensor],
        compiled_surfaces: Mapping[str, Tensor],
        ignore_index: int,
    ) -> None:
        candidates = {
            "coordinate_ablation": (ablation_logits, ablation_surfaces),
            "coordinate_replacement": (
                replacement_logits,
                replacement_surfaces,
            ),
            "compiled": (compiled_logits, compiled_surfaces),
        }
        for name, (logits, surfaces) in candidates.items():
            self.candidates[name].update(
                native_logits=native_logits,
                candidate_logits=logits,
                targets=targets,
                valid_positions=valid_positions,
                native_surfaces=native_surfaces,
                candidate_surfaces=surfaces,
                ignore_index=ignore_index,
            )
        for name in _SURFACES:
            self.compiled_exactness[name].update(
                replacement_surfaces[name],
                compiled_surfaces[name],
                valid_positions.to(
                    device=replacement_surfaces[name].device,
                    dtype=torch.bool,
                ),
            )

    def finish(self) -> dict[str, object]:
        conditions = {
            name: totals.finish()
            for name, totals in self.candidates.items()
        }
        ablation = conditions["coordinate_ablation"]
        compiled = conditions["compiled"]
        assert isinstance(ablation["surfaces"], Mapping)
        assert isinstance(compiled["surfaces"], Mapping)
        surface_recovery = {}
        for name in _SURFACES:
            deletion_surface = ablation["surfaces"][name]
            compiled_surface = compiled["surfaces"][name]
            assert isinstance(deletion_surface, Mapping)
            assert isinstance(compiled_surface, Mapping)
            surface_recovery[name] = _recovery(
                float(compiled_surface["square_error"]),
                float(deletion_surface["square_error"]),
            )
        behavior_recovery = {
            "absolute_nll_displacement": _recovery(
                abs(float(compiled["delta_nll_per_token"])),
                abs(float(ablation["delta_nll_per_token"])),
            ),
            "teacher_kl": _recovery(
                float(compiled["teacher_kl_per_token"]),
                float(ablation["teacher_kl_per_token"]),
            ),
            "top1_disagreement": _recovery(
                1.0 - float(compiled["top1_agreement"]),
                1.0 - float(ablation["top1_agreement"]),
            ),
        }
        replacement = conditions["coordinate_replacement"]
        compiled_exactness = {
            "surfaces": {
                name: totals.finish()
                for name, totals in self.compiled_exactness.items()
            },
            "behavior_absolute_differences": {
                name: abs(float(compiled[name]) - float(replacement[name]))
                for name in (
                    "delta_nll_per_token",
                    "teacher_kl_per_token",
                    "top1_agreement",
                )
            },
        }
        return {
            "conditions": conditions,
            "compiled_vs_coordinate_replacement": compiled_exactness,
            "compiled_recovery_fraction_vs_coordinate_ablation": {
                "surfaces": surface_recovery,
                "behavior": behavior_recovery,
            },
        }


def _slice_surface(values: Mapping[str, Tensor], index: int) -> dict[str, Tensor]:
    return {
        name: values[name][index : index + 1] for name in _SURFACES
    }


def _tensor_layer_output(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], Tensor):
            return output[0]
    raise TypeError("Gemma consumer layer output does not expose a Tensor")


def _extract_logits(output: object) -> Tensor:
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    if isinstance(output, Mapping):
        value = output.get("logits")
        if isinstance(value, Tensor):
            return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, Tensor) and value.ndim == 3:
                return value
    raise TypeError("compiled Gemma output does not expose logits")


def _proposal_for_executor(
    plan: StructuredMLPCrossBlockPlan,
    model_executor: Gemma3CrossBlockModelExecutor,
) -> UnresolvedCrossBlockCarryProposal:
    proposal_id = model_executor.executor.binding.proposal_id
    matches = tuple(
        proposal
        for proposal in plan.proposals
        if proposal.proposal_id == proposal_id
    )
    if len(matches) != 1:
        raise ValueError("compiled proposal is absent from the plan")
    proposal = matches[0]
    binding = model_executor.executor.binding
    if (
        proposal.anchor.layer_id != binding.anchor_layer_id
        or proposal.anchor_source_index != binding.anchor_source_index
        or proposal.consumer.layer_id != binding.consumer_layer_id
        or proposal.consumer_source_index != binding.consumer_source_index
    ):
        raise ValueError("compiled native coordinates differ from the plan")
    return proposal


def _projection_modules(
    adapter: Gemma3CausalLMAdapter,
    model_executor: Gemma3CrossBlockModelExecutor,
    proposal: UnresolvedCrossBlockCarryProposal,
) -> dict[str, dict[str, nn.Module]]:
    anchor_mlp = getattr(
        adapter.source_module(proposal.anchor.layer_id),
        "mlp",
        None,
    )
    consumer_mlp = getattr(
        adapter.source_module(proposal.consumer.layer_id),
        "mlp",
        None,
    )
    if not isinstance(anchor_mlp, nn.Module) or not isinstance(
        consumer_mlp,
        nn.Module,
    ):
        raise TypeError("Gemma source layers do not expose MLP modules")
    source = {
        f"{side}_{projection}": getattr(module, f"{projection}_proj")
        for side, module in (
            ("anchor", anchor_mlp),
            ("consumer", consumer_mlp),
        )
        for projection in ("gate", "up", "down")
    }
    executor = model_executor.executor
    candidate = {
        name: getattr(executor, f"{name}_proj")
        for name in (
            "anchor_gate",
            "anchor_up",
            "anchor_down",
            "consumer_gate",
            "consumer_up",
            "consumer_down",
        )
    }
    if any(
        not isinstance(module, nn.Module)
        for module in (*source.values(), *candidate.values())
    ):
        raise TypeError("projection call audit requires torch modules")
    return {"source": source, "candidate": candidate}


def _empty_call_counts(
    modules: Mapping[str, Mapping[str, nn.Module]],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        condition: {
            origin: {name: 0 for name in values}
            for origin, values in modules.items()
        }
        for condition in ("native", *_CANDIDATES)
    }


def _gate_one(
    metrics: Mapping[str, object],
    thresholds: Gemma3CrossBlockSupermodeGuardThresholds,
) -> dict[str, bool]:
    exact = metrics["compiled_vs_coordinate_replacement"]
    recovery = metrics[
        "compiled_recovery_fraction_vs_coordinate_ablation"
    ]
    conditions = metrics["conditions"]
    if not all(
        isinstance(value, Mapping)
        for value in (exact, recovery, conditions)
    ):
        raise RuntimeError("guard metric schema drifted")
    exact_surfaces = exact["surfaces"]
    behavior_difference = exact["behavior_absolute_differences"]
    surface_recovery = recovery["surfaces"]
    behavior_recovery = recovery["behavior"]
    compiled = conditions["compiled"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            exact_surfaces,
            behavior_difference,
            surface_recovery,
            behavior_recovery,
            compiled,
        )
    ):
        raise RuntimeError("guard metric nesting drifted")
    gates: dict[str, bool] = {}
    for name in _SURFACES:
        surface = exact_surfaces[name]
        assert isinstance(surface, Mapping)
        gates[f"compiled_exact_{name}"] = (
            float(surface["maximum_absolute_error"])
            <= thresholds.compiled_oracle_surface_maximum_absolute_error
        )
        value = surface_recovery[name]
        gates[f"compiled_recovers_{name}"] = (
            value is not None
            and float(value)
            >= thresholds.minimum_surface_recovery_fraction
        )
    for name, value in behavior_difference.items():
        gates[f"compiled_exact_{name}"] = (
            float(value)
            <= thresholds.compiled_oracle_behavior_maximum_absolute_difference
        )
    gates["compiled_absolute_delta_nll"] = (
        abs(float(compiled["delta_nll_per_token"]))
        <= thresholds.maximum_absolute_delta_nll_per_token
    )
    gates["compiled_teacher_kl"] = (
        float(compiled["teacher_kl_per_token"])
        <= thresholds.maximum_teacher_kl_per_token
    )
    gates["compiled_top1"] = (
        float(compiled["top1_agreement"])
        >= thresholds.minimum_top1_agreement
    )
    for name, minimum in (
        (
            "absolute_nll_displacement",
            thresholds.minimum_nll_recovery_fraction,
        ),
        ("teacher_kl", thresholds.minimum_kl_recovery_fraction),
        (
            "top1_disagreement",
            thresholds.minimum_top1_recovery_fraction,
        ),
    ):
        value = behavior_recovery[name]
        gates[f"compiled_recovers_{name}"] = (
            minimum < 0
            or (value is not None and float(value) >= minimum)
        )
    return gates


def _all_authorities_false(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key.startswith("authorizes_") and nested is not False:
                return False
            if not _all_authorities_false(nested):
                return False
    elif isinstance(value, (tuple, list)):
        return all(_all_authorities_false(item) for item in value)
    return True


def validate_gemma3_cross_block_supermode_guard_artifact(
    artifact: Mapping[str, object],
) -> None:
    """Validate the strict JSON envelope, content hash, and no-authority ABI."""

    expected = {
        "schema",
        "format_version",
        "binding",
        "protocol",
        "metrics",
        "gates",
        "physical_execution_audit",
        "resource_accounting",
        "source_audit",
        "safety",
        "artifact_sha256",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected:
        raise ValueError("supermode guard artifact fields are invalid")
    if (
        artifact["schema"] != GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_SCHEMA
        or artifact["format_version"]
        != GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_FORMAT_VERSION
    ):
        raise ValueError("supermode guard artifact schema is invalid")
    _assert_json_safe(artifact, path="artifact")
    payload = dict(artifact)
    supplied = _require_sha256(
        payload.pop("artifact_sha256"),
        label="artifact_sha256",
    )
    if supplied != _json_sha256(payload):
        raise ValueError("supermode guard artifact hash mismatch")
    if artifact["safety"] != _SAFETY or not _all_authorities_false(artifact):
        raise ValueError("supermode guard artifact authority is invalid")
    binding = artifact["binding"]
    if not isinstance(binding, Mapping) or set(binding) != {
        "source_model_fingerprint",
        "source_execution_fingerprint",
        "source_plan_artifact_sha256",
        "source_replacement_oracle_artifact_sha256",
        "compiled_executor_fingerprint",
        "threshold_artifact_sha256",
        "guard_stream_sha256",
    }:
        raise ValueError("supermode guard binding is invalid")
    for label, value in binding.items():
        _require_sha256(value, label=label)
    protocol = artifact["protocol"]
    if (
        not isinstance(protocol, Mapping)
        or set(protocol)
        != {
            "thresholds",
            "thresholds_fixed_before_guard",
            "candidate_frozen_before_guard",
            "native_ablation_replacement_compiled_paired",
            "coordinate_replacement_scale_frozen_from_oracle",
            "family_and_length_labels_frozen_in_stream_hash",
            "batch_count",
            "example_count",
            "family_counts",
            "length_counts",
            "ignore_index",
        }
        or protocol.get("thresholds_fixed_before_guard") is not True
        or protocol.get("candidate_frozen_before_guard") is not True
        or protocol.get("native_ablation_replacement_compiled_paired")
        is not True
        or protocol.get(
            "coordinate_replacement_scale_frozen_from_oracle"
        )
        is not True
        or protocol.get(
            "family_and_length_labels_frozen_in_stream_hash"
        )
        is not True
        or type(protocol.get("batch_count")) is not int
        or int(protocol["batch_count"]) <= 0
        or type(protocol.get("example_count")) is not int
        or int(protocol["example_count"]) <= 0
        or type(protocol.get("ignore_index")) is not int
    ):
        raise ValueError("supermode guard protocol is invalid")
    threshold_state = protocol["thresholds"]
    threshold_fields = set(
        Gemma3CrossBlockSupermodeGuardThresholds.__dataclass_fields__
    )
    if (
        not isinstance(threshold_state, Mapping)
        or set(threshold_state)
        != {
            "schema",
            "format_version",
            "artifact_sha256",
            *threshold_fields,
        }
        or threshold_state.get("schema")
        != GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_THRESHOLDS_SCHEMA
        or threshold_state.get("format_version")
        != GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_THRESHOLDS_FORMAT_VERSION
    ):
        raise ValueError("supermode guard threshold metadata is invalid")
    threshold_values = {
        name: threshold_state[name] for name in threshold_fields
    }
    try:
        thresholds = Gemma3CrossBlockSupermodeGuardThresholds(
            **threshold_values  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "supermode guard threshold metadata is invalid"
        ) from error
    if (
        thresholds.metadata() != dict(threshold_state)
        or thresholds.artifact_sha256
        != binding["threshold_artifact_sha256"]
        or thresholds.source_model_fingerprint
        != binding["source_model_fingerprint"]
        or thresholds.source_execution_fingerprint
        != binding["source_execution_fingerprint"]
        or thresholds.source_plan_artifact_sha256
        != binding["source_plan_artifact_sha256"]
        or thresholds.source_replacement_oracle_artifact_sha256
        != binding["source_replacement_oracle_artifact_sha256"]
        or thresholds.guard_stream_sha256
        != binding["guard_stream_sha256"]
    ):
        raise ValueError("supermode guard threshold binding is invalid")
    family_counts = protocol["family_counts"]
    length_counts = protocol["length_counts"]
    if (
        not isinstance(family_counts, Mapping)
        or not family_counts
        or not isinstance(length_counts, Mapping)
        or not length_counts
        or any(
            not isinstance(key, str)
            or not key
            or type(value) is not int
            or value <= 0
            for counts in (family_counts, length_counts)
            for key, value in counts.items()
        )
        or sum(family_counts.values()) != protocol["example_count"]
        or sum(length_counts.values()) != protocol["example_count"]
    ):
        raise ValueError("supermode guard stratum counts are invalid")
    metrics = artifact["metrics"]
    if (
        not isinstance(metrics, Mapping)
        or set(metrics)
        != {"overall", "family", "length", "family_length"}
        or not isinstance(metrics["overall"], Mapping)
        or set(metrics["overall"]) != {"all"}
        or not isinstance(metrics["family"], Mapping)
        or set(metrics["family"]) != set(family_counts)
        or not isinstance(metrics["length"], Mapping)
        or set(metrics["length"]) != set(length_counts)
        or not isinstance(metrics["family_length"], Mapping)
        or not metrics["family_length"]
    ):
        raise ValueError("supermode guard metrics are invalid")
    gates = artifact["gates"]
    if (
        not isinstance(gates, Mapping)
        or set(gates)
        != {
            "evaluated_strata",
            "passed",
            "passing_does_not_authorize_any_next_split_or_execution",
        }
        or type(gates["passed"]) is not bool
        or gates[
            "passing_does_not_authorize_any_next_split_or_execution"
        ]
        is not True
        or not isinstance(gates["evaluated_strata"], Mapping)
    ):
        raise ValueError("supermode guard gates are invalid")
    selected_kinds = ["overall"]
    if thresholds.apply_to_each_family:
        selected_kinds.append("family")
    if thresholds.apply_to_each_length:
        selected_kinds.append("length")
    expected_gates = {
        kind: {
            key: _gate_one(value, thresholds)
            for key, value in metrics[kind].items()
        }
        for kind in selected_kinds
    }
    expected_pass = all(
        value
        for strata in expected_gates.values()
        for gate_map in strata.values()
        for value in gate_map.values()
    )
    if (
        gates["evaluated_strata"] != expected_gates
        or gates["passed"] is not expected_pass
    ):
        raise ValueError("supermode guard gates do not match metrics")
    resource = artifact["resource_accounting"]
    if (
        not isinstance(resource, Mapping)
        or set(resource)
        != {
            "source_whole_model_learned_parameters",
            "candidate_whole_model_learned_parameters",
            "removed_learned_parameters",
            "removed_linear_macs_per_valid_token",
            "carry_scale_macs_per_valid_token",
            "net_arithmetic_macs_saved_per_valid_token",
            "guard_valid_position_count",
            "removed_linear_macs_over_guard",
            "carry_scale_macs_over_guard",
            "net_arithmetic_macs_saved_over_guard",
            "kernel_latency_speedup_claimed",
            "compression_materialized_in_evaluated_in_memory_overlay",
            "serialized_model_size_reduction_measured",
            "deployable_compressed_model_artifact_claimed",
        }
        or resource.get("removed_learned_parameters") != 1280
        or resource.get("removed_linear_macs_per_valid_token") != 1280
        or resource.get("carry_scale_macs_per_valid_token") != 1
        or resource.get("net_arithmetic_macs_saved_per_valid_token") != 1279
        or type(resource.get("guard_valid_position_count")) is not int
        or int(resource["guard_valid_position_count"]) <= 0
        or resource.get("removed_linear_macs_over_guard")
        != int(resource["guard_valid_position_count"]) * 1280
        or resource.get("carry_scale_macs_over_guard")
        != resource["guard_valid_position_count"]
        or resource.get("net_arithmetic_macs_saved_over_guard")
        != int(resource["guard_valid_position_count"]) * 1279
        or type(resource.get("source_whole_model_learned_parameters"))
        is not int
        or type(resource.get("candidate_whole_model_learned_parameters"))
        is not int
        or int(resource["source_whole_model_learned_parameters"])
        - int(resource["candidate_whole_model_learned_parameters"])
        != 1280
        or resource.get("kernel_latency_speedup_claimed") is not False
        or resource.get(
            "compression_materialized_in_evaluated_in_memory_overlay"
        )
        is not True
        or resource.get("serialized_model_size_reduction_measured")
        is not False
        or resource.get("deployable_compressed_model_artifact_claimed")
        is not False
    ):
        raise ValueError("supermode guard resource accounting is invalid")
    source_audit = artifact["source_audit"]
    if (
        not isinstance(source_audit, Mapping)
        or set(source_audit)
        != {
            "guard_stream_unchanged",
            "source_model_state_unchanged",
            "source_execution_fingerprint_unchanged",
            "source_parameter_versions_unchanged",
            "source_parameter_gradients_absent",
            "compiled_executor_state_unchanged",
            "compiled_executor_parameter_versions_unchanged",
            "compiled_executor_parameter_gradients_absent",
            "source_modules_restored_after_overlay",
        }
        or any(value is not True for value in source_audit.values())
    ):
        raise ValueError("supermode guard frozen-source audit is invalid")
    physical = artifact["physical_execution_audit"]
    if (
        not isinstance(physical, Mapping)
        or set(physical)
        != {
            "projection_call_counts",
            "expected_calls_per_projection_per_condition",
            "all_projection_call_counts_exact",
            "compiled_source_consumer_gate_calls_zero",
            "compiled_source_consumer_up_calls_zero",
            "candidate_consumer_gate_rows",
            "source_consumer_gate_rows",
            "candidate_consumer_up_rows",
            "source_consumer_up_rows",
            "candidate_consumer_rows_physically_reduced_by_one",
            "full_consumer_down_projection_preserved",
        }
        or physical.get("compiled_source_consumer_gate_calls_zero")
        is not True
        or physical.get("compiled_source_consumer_up_calls_zero") is not True
        or physical.get("candidate_consumer_rows_physically_reduced_by_one")
        is not True
        or physical.get("all_projection_call_counts_exact") is not True
        or physical.get("full_consumer_down_projection_preserved") is not True
        or type(physical.get("candidate_consumer_gate_rows")) is not int
        or type(physical.get("source_consumer_gate_rows")) is not int
        or type(physical.get("candidate_consumer_up_rows")) is not int
        or type(physical.get("source_consumer_up_rows")) is not int
        or int(physical["candidate_consumer_gate_rows"]) + 1
        != physical["source_consumer_gate_rows"]
        or int(physical["candidate_consumer_up_rows"]) + 1
        != physical["source_consumer_up_rows"]
    ):
        raise ValueError("supermode guard projection audit is invalid")
    call_counts = physical["projection_call_counts"]
    expected_calls = physical["expected_calls_per_projection_per_condition"]
    projection_names = {
        "anchor_gate",
        "anchor_up",
        "anchor_down",
        "consumer_gate",
        "consumer_up",
        "consumer_down",
    }
    if (
        not isinstance(call_counts, Mapping)
        or set(call_counts) != {"native", *_CANDIDATES}
        or expected_calls != protocol["batch_count"]
    ):
        raise ValueError("supermode guard projection calls are invalid")
    for condition, origins in call_counts.items():
        if not isinstance(origins, Mapping) or set(origins) != {
            "source",
            "candidate",
        }:
            raise ValueError("supermode guard projection calls are invalid")
        for origin, counts in origins.items():
            expected_count = (
                expected_calls
                if (
                    condition != "compiled" and origin == "source"
                )
                or (condition == "compiled" and origin == "candidate")
                else 0
            )
            if (
                not isinstance(counts, Mapping)
                or set(counts) != projection_names
                or any(
                    type(value) is not int or value != expected_count
                    for value in counts.values()
                )
            ):
                raise ValueError(
                    "supermode guard projection calls are invalid"
                )


def run_gemma3_cross_block_supermode_guard(
    model_executor: Gemma3CrossBlockModelExecutor,
    guard_batches: Sequence[CalibrationBatch],
    *,
    family_by_example: Mapping[str, str],
    length_by_example: Mapping[str, str],
    plan: StructuredMLPCrossBlockPlan,
    replacement_oracle_artifact: Mapping[str, object],
    thresholds: Gemma3CrossBlockSupermodeGuardThresholds,
    ignore_index: int = -100,
) -> dict[str, object]:
    """Evaluate the compiled supermode against native paired controls."""

    if not isinstance(model_executor, Gemma3CrossBlockModelExecutor):
        raise TypeError("model_executor must be a Gemma3CrossBlockModelExecutor")
    if not isinstance(plan, StructuredMLPCrossBlockPlan):
        raise TypeError("plan must be a StructuredMLPCrossBlockPlan")
    if not isinstance(
        thresholds,
        Gemma3CrossBlockSupermodeGuardThresholds,
    ):
        raise TypeError("thresholds must be preregistered guard thresholds")
    if type(ignore_index) is not int:
        raise TypeError("ignore_index must be an integer")
    validate_gemma3_cross_block_replacement_oracle_artifact(
        replacement_oracle_artifact
    )
    batches, example_ids = _validated_examples_and_labels(
        guard_batches,
        family_by_example,
        length_by_example,
    )
    stream_sha256 = gemma3_cross_block_guard_stream_sha256(
        batches,
        family_by_example=family_by_example,
        length_by_example=length_by_example,
    )
    adapter = model_executor.adapter
    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("guard requires the concrete Gemma 3 adapter")
    executor = model_executor.executor
    proposal = _proposal_for_executor(plan, model_executor)
    oracle_binding = replacement_oracle_artifact["binding"]
    oracle_proposal = replacement_oracle_artifact["proposal"]
    oracle_fit = replacement_oracle_artifact["fit"]
    if not all(
        isinstance(value, Mapping)
        for value in (oracle_binding, oracle_proposal, oracle_fit)
    ):
        raise ValueError("replacement oracle metadata is invalid")
    source_fingerprint = adapter.model_fingerprint()
    source_execution_fingerprint = adapter.execution_fingerprint()
    oracle_sha256 = _require_sha256(
        replacement_oracle_artifact["artifact_sha256"],
        label="replacement_oracle_artifact_sha256",
    )
    if (
        plan.source_model_fingerprint != source_fingerprint
        or getattr(
            executor.binding,
            "source_execution_fingerprint",
            None,
        )
        != source_execution_fingerprint
        or plan.artifact_sha256
        != executor.binding.source_plan_artifact_sha256
        or oracle_binding.get("source_model_fingerprint")
        != source_fingerprint
        or oracle_binding.get("source_plan_artifact_sha256")
        != plan.artifact_sha256
        or oracle_sha256
        != executor.binding.source_replacement_oracle_artifact_sha256
        or oracle_proposal.get("proposal_id") != proposal.proposal_id
        or float(oracle_fit.get("selected_scale"))
        != executor.binding.carry_scale
    ):
        raise ValueError("guard source, plan, oracle, and executor disagree")
    if (
        thresholds.source_model_fingerprint != source_fingerprint
        or thresholds.source_execution_fingerprint
        != source_execution_fingerprint
        or thresholds.source_plan_artifact_sha256 != plan.artifact_sha256
        or thresholds.source_replacement_oracle_artifact_sha256
        != oracle_sha256
        or thresholds.guard_stream_sha256 != stream_sha256
    ):
        raise ValueError("preregistered guard threshold binding disagrees")
    if executor.residual_width != 640:
        raise ValueError(
            "exact 1280-row-savings guard requires Gemma residual width 640"
        )
    removed_parameters = (
        executor.source_pair_parameter_count - executor.learned_parameter_count
    )
    if removed_parameters != 1280:
        raise RuntimeError("compiled parameter savings are not exactly 1280")
    for batch in batches:
        prepared = adapter.prepare_sequence(batch.model_inputs)
        if not torch.equal(
            batch.valid_positions.to(
                device=prepared.query_valid_mask.device,
                dtype=torch.bool,
            ),
            prepared.query_valid_mask,
        ):
            raise ValueError(
                "compiled/oracle exactness requires batch valid positions "
                "to equal the model attention mask"
            )

    source_state_before = module_state_fingerprint(adapter.module)
    source_execution_before = source_execution_fingerprint
    executor_state_before = module_state_fingerprint(executor)
    source_versions_before = tuple(
        parameter._version for parameter in adapter.module.parameters()
    )
    executor_versions_before = tuple(
        parameter._version for parameter in executor.parameters()
    )
    modules = _projection_modules(adapter, model_executor, proposal)
    call_counts = _empty_call_counts(modules)
    active_condition: list[str | None] = [None]
    handles = []
    for origin, projections in modules.items():
        for name, module in projections.items():

            def count_call(
                _module: nn.Module,
                _args: tuple[object, ...],
                _output: object,
                *,
                origin: str = origin,
                name: str = name,
            ) -> None:
                condition = active_condition[0]
                if condition is None:
                    raise RuntimeError(
                        "projection executed outside an audited condition"
                    )
                call_counts[condition][origin][name] += 1

            handles.append(module.register_forward_hook(count_call))

    capture_sites = _capture_sites(adapter, proposal)
    scale = executor.binding.carry_scale
    totals: dict[str, dict[str, _StratumTotals]] = {
        "overall": {"all": _StratumTotals()},
        "family": {},
        "length": {},
        "family_length": {},
    }
    total_valid_positions = 0
    candidate_parameter_counts: set[int] = set()
    try:
        with torch.no_grad():
            for batch in batches:
                condition_runs = {}
                for audit_name, oracle_name in (
                    ("native", "native"),
                    ("coordinate_ablation", "consumer_ablation"),
                    ("coordinate_replacement", "carried_replacement"),
                ):
                    active_condition[0] = audit_name
                    condition_runs[audit_name] = (
                        _run_with_coordinate_intervention(
                            adapter,
                            batch,
                            proposal,
                            condition=oracle_name,
                            carry_scale=scale,
                            capture_sites=capture_sites,
                        )
                    )
                    active_condition[0] = None
                consumer_layer = adapter.source_module(
                    proposal.consumer.layer_id
                )
                compiled_window: list[Tensor] = []

                def capture_window(
                    _module: nn.Module,
                    _args: tuple[object, ...],
                    output: object,
                ) -> None:
                    compiled_window.append(_tensor_layer_output(output))

                window_handle = consumer_layer.register_forward_hook(
                    capture_window
                )
                try:
                    active_condition[0] = "compiled"
                    compiled_execution = model_executor(batch.model_inputs)
                    active_condition[0] = None
                finally:
                    window_handle.remove()
                if len(compiled_window) != 1:
                    raise RuntimeError(
                        "compiled consumer window output was not observed once"
                    )
                candidate_parameter_counts.add(
                    compiled_execution.candidate_whole_model_learned_parameters
                )
                native_run = condition_runs["native"][0]
                ablation_run = condition_runs["coordinate_ablation"][0]
                replacement_run = condition_runs[
                    "coordinate_replacement"
                ][0]
                native_surfaces = {
                    "consumer_mlp_output": native_run.activations[
                        capture_sites["consumer_mlp_output"]
                    ],
                    "window_output": native_run.activations[
                        capture_sites["window_output"]
                    ],
                    "final_logits": native_run.logits,
                }
                ablation_surfaces = {
                    "consumer_mlp_output": ablation_run.activations[
                        capture_sites["consumer_mlp_output"]
                    ],
                    "window_output": ablation_run.activations[
                        capture_sites["window_output"]
                    ],
                    "final_logits": ablation_run.logits,
                }
                replacement_surfaces = {
                    "consumer_mlp_output": replacement_run.activations[
                        capture_sites["consumer_mlp_output"]
                    ],
                    "window_output": replacement_run.activations[
                        capture_sites["window_output"]
                    ],
                    "final_logits": replacement_run.logits,
                }
                compiled_logits = _extract_logits(
                    compiled_execution.model_output
                )
                compiled_surfaces = {
                    "consumer_mlp_output": (
                        compiled_execution.consumer_mlp_output
                    ),
                    "window_output": compiled_window[0],
                    "final_logits": compiled_logits,
                }
                assert batch.example_ids is not None
                for index, example_id in enumerate(batch.example_ids):
                    family = family_by_example[example_id]
                    length = length_by_example[example_id]
                    family_length = json.dumps(
                        [family, length],
                        separators=(",", ":"),
                    )
                    keys = (
                        ("overall", "all"),
                        ("family", family),
                        ("length", length),
                        ("family_length", family_length),
                    )
                    for kind, key in keys:
                        stratum = totals[kind].setdefault(
                            key,
                            _StratumTotals(),
                        )
                        stratum.update(
                            native_logits=native_run.logits[
                                index : index + 1
                            ],
                            ablation_logits=ablation_run.logits[
                                index : index + 1
                            ],
                            replacement_logits=replacement_run.logits[
                                index : index + 1
                            ],
                            compiled_logits=compiled_logits[
                                index : index + 1
                            ],
                            targets=batch.targets[index : index + 1],
                            valid_positions=batch.valid_positions[
                                index : index + 1
                            ],
                            native_surfaces=_slice_surface(
                                native_surfaces,
                                index,
                            ),
                            ablation_surfaces=_slice_surface(
                                ablation_surfaces,
                                index,
                            ),
                            replacement_surfaces=_slice_surface(
                                replacement_surfaces,
                                index,
                            ),
                            compiled_surfaces=_slice_surface(
                                compiled_surfaces,
                                index,
                            ),
                            ignore_index=ignore_index,
                        )
                total_valid_positions += int(
                    batch.valid_positions.sum().item()
                )
    finally:
        active_condition[0] = None
        for handle in reversed(handles):
            handle.remove()

    metrics = {
        kind: {
            key: value.finish()
            for key, value in sorted(strata.items())
        }
        for kind, strata in totals.items()
    }
    expected_calls = len(batches)
    exact_call_counts = True
    for condition in ("native", *_CANDIDATES):
        for origin, projections in call_counts[condition].items():
            expected = (
                expected_calls
                if (
                    condition != "compiled"
                    and origin == "source"
                )
                or (condition == "compiled" and origin == "candidate")
                else 0
            )
            exact_call_counts = exact_call_counts and all(
                count == expected for count in projections.values()
            )
    source_consumer_gate = modules["source"]["consumer_gate"]
    candidate_consumer_gate = modules["candidate"]["consumer_gate"]
    source_consumer_up = modules["source"]["consumer_up"]
    candidate_consumer_up = modules["candidate"]["consumer_up"]
    if not all(
        isinstance(value, nn.Linear)
        for value in (
            source_consumer_gate,
            candidate_consumer_gate,
            source_consumer_up,
            candidate_consumer_up,
        )
    ):
        raise TypeError("physical row audit requires linear projections")
    rows_reduced = (
        candidate_consumer_gate.out_features
        == source_consumer_gate.out_features - 1
        and candidate_consumer_up.out_features
        == source_consumer_up.out_features - 1
    )
    if not exact_call_counts or not rows_reduced:
        raise RuntimeError("physical projection execution audit failed")

    finished_gates: dict[str, dict[str, dict[str, bool]]] = {}
    selected_kinds = ["overall"]
    if thresholds.apply_to_each_family:
        selected_kinds.append("family")
    if thresholds.apply_to_each_length:
        selected_kinds.append("length")
    for kind in selected_kinds:
        finished_gates[kind] = {
            key: _gate_one(value, thresholds)
            for key, value in metrics[kind].items()
        }
    gate_passed = all(
        passed
        for strata in finished_gates.values()
        for gates in strata.values()
        for passed in gates.values()
    )
    gates: dict[str, object] = {
        "evaluated_strata": finished_gates,
        "passed": gate_passed,
        "passing_does_not_authorize_any_next_split_or_execution": True,
    }

    stream_after = gemma3_cross_block_guard_stream_sha256(
        batches,
        family_by_example=family_by_example,
        length_by_example=length_by_example,
    )
    source_state_after = module_state_fingerprint(adapter.module)
    source_execution_after = adapter.execution_fingerprint()
    executor_state_after = module_state_fingerprint(executor)
    source_versions_after = tuple(
        parameter._version for parameter in adapter.module.parameters()
    )
    executor_versions_after = tuple(
        parameter._version for parameter in executor.parameters()
    )
    source_audit = {
        "guard_stream_unchanged": stream_after == stream_sha256,
        "source_model_state_unchanged": (
            source_state_after == source_state_before
        ),
        "source_execution_fingerprint_unchanged": (
            source_execution_after == source_execution_before
        ),
        "source_parameter_versions_unchanged": (
            source_versions_after == source_versions_before
        ),
        "source_parameter_gradients_absent": all(
            parameter.grad is None for parameter in adapter.module.parameters()
        ),
        "compiled_executor_state_unchanged": (
            executor_state_after == executor_state_before
        ),
        "compiled_executor_parameter_versions_unchanged": (
            executor_versions_after == executor_versions_before
        ),
        "compiled_executor_parameter_gradients_absent": all(
            parameter.grad is None for parameter in executor.parameters()
        ),
        "source_modules_restored_after_overlay": (
            adapter.model_fingerprint() == source_fingerprint
        ),
    }
    if any(value is not True for value in source_audit.values()):
        raise RuntimeError("guard mutated frozen source, candidate, or batches")
    if len(candidate_parameter_counts) != 1:
        raise RuntimeError("candidate whole-model parameter count drifted")

    physical_audit = {
        "projection_call_counts": call_counts,
        "expected_calls_per_projection_per_condition": expected_calls,
        "all_projection_call_counts_exact": exact_call_counts,
        "compiled_source_consumer_gate_calls_zero": (
            call_counts["compiled"]["source"]["consumer_gate"] == 0
        ),
        "compiled_source_consumer_up_calls_zero": (
            call_counts["compiled"]["source"]["consumer_up"] == 0
        ),
        "candidate_consumer_gate_rows": (
            candidate_consumer_gate.out_features
        ),
        "source_consumer_gate_rows": source_consumer_gate.out_features,
        "candidate_consumer_up_rows": candidate_consumer_up.out_features,
        "source_consumer_up_rows": source_consumer_up.out_features,
        "candidate_consumer_rows_physically_reduced_by_one": rows_reduced,
        "full_consumer_down_projection_preserved": True,
    }
    candidate_parameters = next(iter(candidate_parameter_counts))
    source_parameters = candidate_parameters + removed_parameters
    resource_accounting = {
        "source_whole_model_learned_parameters": source_parameters,
        "candidate_whole_model_learned_parameters": candidate_parameters,
        "removed_learned_parameters": removed_parameters,
        "removed_linear_macs_per_valid_token": 1280,
        "carry_scale_macs_per_valid_token": 1,
        "net_arithmetic_macs_saved_per_valid_token": 1279,
        "guard_valid_position_count": total_valid_positions,
        "removed_linear_macs_over_guard": (
            total_valid_positions * 1280
        ),
        "carry_scale_macs_over_guard": total_valid_positions,
        "net_arithmetic_macs_saved_over_guard": (
            total_valid_positions * 1279
        ),
        "kernel_latency_speedup_claimed": False,
        "compression_materialized_in_evaluated_in_memory_overlay": True,
        "serialized_model_size_reduction_measured": False,
        "deployable_compressed_model_artifact_claimed": False,
    }
    family_counts = {
        family: sum(
            family_by_example[example_id] == family
            for example_id in example_ids
        )
        for family in sorted(set(family_by_example.values()))
    }
    length_counts = {
        length: sum(
            length_by_example[example_id] == length
            for example_id in example_ids
        )
        for length in sorted(set(length_by_example.values()))
    }
    payload = {
        "schema": GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_SCHEMA,
        "format_version": GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_FORMAT_VERSION,
        "binding": {
            "source_model_fingerprint": source_fingerprint,
            "source_execution_fingerprint": source_execution_before,
            "source_plan_artifact_sha256": plan.artifact_sha256,
            "source_replacement_oracle_artifact_sha256": oracle_sha256,
            "compiled_executor_fingerprint": executor_state_before,
            "threshold_artifact_sha256": thresholds.artifact_sha256,
            "guard_stream_sha256": stream_sha256,
        },
        "protocol": {
            "thresholds": thresholds.metadata(),
            "thresholds_fixed_before_guard": True,
            "candidate_frozen_before_guard": True,
            "native_ablation_replacement_compiled_paired": True,
            "coordinate_replacement_scale_frozen_from_oracle": True,
            "family_and_length_labels_frozen_in_stream_hash": True,
            "batch_count": len(batches),
            "example_count": len(example_ids),
            "family_counts": family_counts,
            "length_counts": length_counts,
            "ignore_index": ignore_index,
        },
        "metrics": metrics,
        "gates": gates,
        "physical_execution_audit": physical_audit,
        "resource_accounting": resource_accounting,
        "source_audit": source_audit,
        "safety": dict(_SAFETY),
    }
    _assert_json_safe(payload, path="guard")
    artifact = {**payload, "artifact_sha256": _json_sha256(payload)}
    validate_gemma3_cross_block_supermode_guard_artifact(artifact)
    return artifact


__all__ = [
    "GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_FORMAT_VERSION",
    "GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_SCHEMA",
    "GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_THRESHOLDS_FORMAT_VERSION",
    "GEMMA3_CROSS_BLOCK_SUPERMODE_GUARD_THRESHOLDS_SCHEMA",
    "Gemma3CrossBlockSupermodeGuardThresholds",
    "gemma3_cross_block_guard_stream_sha256",
    "run_gemma3_cross_block_supermode_guard",
    "validate_gemma3_cross_block_supermode_guard_artifact",
]
