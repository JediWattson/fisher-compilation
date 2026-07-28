"""Compile a prompt-blind, state-conditioned Gemma L3/L4 reference provider.

This rung asks a narrow question: after a Fisher basis has been frozen, can a
compact causal provider learn the mean-source L3-to-L4 reference term using
only deterministic synthetic residual-state probes?

The compiler never opens prompt text, token IDs, a tokenizer, natural
activation rows, or the historical prompt-local edge kernel.  Its only
data-derived input is the reduced Fisher aggregate package.  Synthetic
directions are lifted onto Gemma's valid pre-feedforward RMS manifold, then
the live L4 attention prefix supplies targets.  The requested Fisher
directions are construction seeds; all provider inputs are remeasured from
the live normalized states.

Selection is full-width: a rank-r provider is penalized for all omitted modes
against the same 64-mode teacher target.  A sealed synthetic assessment can
report failure but cannot change the selected plan.  This experiment does not
claim natural-prompt fidelity, a whole-model replacement, compression,
latency, or cached-decode support.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pwd
import re
import stat
import tempfile
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .external_models import find_git_worktree
from .gated_executor import GatedCausalModalExecutorConfig
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    Gemma3L3L4BasisPackage,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    DEFAULT_REVISION,
    _load_local_gemma3_model_only,
)
from .gemma3_l3_l4_manifold_lift import (
    MANIFOLD_LIFT_FORMULA_VERSION,
    lift_synthetic_reference_batch_to_gemma3_manifold,
)
from .gemma3_l3_l4_synthetic_materialization import (
    MaterializedSyntheticReferenceBatch,
    materialize_synthetic_reference_batches,
)
from .gemma3_l3_l4_synthetic_reference_protocol import (
    CandidateRatePoint,
    SyntheticReferenceGates,
    SyntheticReferenceProbe,
    SyntheticReferenceProtocol,
    default_synthetic_reference_protocol,
)
from .state_conditioned_reference_provider import (
    ReferenceProviderFeatureCodec,
    StateConditionedReferenceProviderPlan,
    SyntheticReferenceBatch,
    evaluate_state_conditioned_reference_provider,
    fit_state_conditioned_reference_provider,
)
from .state_conditioned_reference_selection import (
    NORMALIZED_POSITION_BIN_SEMANTICS,
    FullWidthCandidatePrediction,
    FullWidthReferenceCandidate,
    FullWidthReferenceControls,
    FullWidthReferenceProbe,
    FullWidthReferenceSelection,
    FullWidthStructuralMetrics,
    fit_full_width_reference_controls,
    full_width_reference_gates_sha256,
    score_full_width_reference_assessment,
    select_smallest_passing_full_width_reference_candidate,
)


__all__ = [
    "DEFAULT_ASSESSMENT_OUTPUT",
    "DEFAULT_BASIS_PACKAGE_FILE_SHA256",
    "DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256",
    "DEFAULT_ASSESSMENT_LEDGER_DIR",
    "DEFAULT_OUTPUT",
    "FrozenGemma3ReferenceProviderTrainingProtocol",
    "assess_gemma3_l3_l4_reference_provider",
    "build_parser",
    "compile_gemma3_l3_l4_reference_provider",
    "main",
]


DEFAULT_BASIS_PACKAGE_FILE_SHA256 = (
    "359c9659358cbaf97232848a10bdf0e2261d95820ad5effda9bdafeead6a7605"
)
DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256 = (
    "b2217153911436673f2ff7475c658c928112e802f5999619393287d2b0803c01"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-reference-provider-dev-v2.pt"
)
DEFAULT_ASSESSMENT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-reference-provider-assessment-dev-v2.pt"
)
DEFAULT_ASSESSMENT_LEDGER_DIR = (
    Path(pwd.getpwuid(os.getuid()).pw_dir)
    / ".local/state/fisher-graph-extract/sealed-assessments"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_reference_provider_development.v2"
_ASSESSMENT_SCHEMA = f"{_SCHEMA}.assessment"
_FORMAT_VERSION = 2
_MODAL_RANK = 64
_TARGET_RANK = 64
_SOURCE_SCOPE = "factorized_refit"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_DOMAIN = b"fisher-graph:gemma3-reference-provider-training:v2\0"
_SYNTHETIC_BINDING_DOMAIN = (
    b"fisher-graph:gemma3-reference-provider-synthetic-binding:v2\0"
)
_CANDIDATE_DOMAIN = b"fisher-graph:gemma3-reference-provider-candidate:v2\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-reference-provider-report:v2\0"
_GAUGE_DOMAIN = b"fisher-graph:gemma3-reference-provider-gauge:v2\0"
_CODE_DOMAIN = b"fisher-graph:gemma3-reference-provider-code:v2\0"
_ASSESSMENT_CLAIM_DOMAIN = (
    b"fisher-graph:gemma3-reference-provider-assessment-claim:v2\0"
)
_ASSESSMENT_PANEL_DOMAIN = (
    b"fisher-graph:gemma3-reference-provider-assessment-panel:v2\0"
)

_CODE_FILES = (
    "gemma3_l3_l4_reference_provider_experiment.py",
    "gemma3_l3_l4_basis_package.py",
    "gemma3_l3_l4_synthetic_reference_protocol.py",
    "gemma3_l3_l4_synthetic_materialization.py",
    "gemma3_l3_l4_manifold_lift.py",
    "state_conditioned_reference_provider.py",
    "state_conditioned_reference_selection.py",
    "gated_executor.py",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": tuple(int(width) for width in tensor.shape),
        }
    )
    return hashlib.sha256(
        header + b"\0" + tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _read_regular_file(path: Path | str) -> bytes:
    source = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{source.name} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _code_sha256s() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: _file_sha256(directory / name)
        for name in _CODE_FILES
    }


def _code_bundle_sha256(values: Mapping[str, str]) -> str:
    if set(values) != set(_CODE_FILES):
        raise ValueError("reference-provider code manifest is incomplete")
    for name, digest in values.items():
        _require_sha256(digest, label=f"code digest {name}")
    return _json_sha256(dict(values), domain=_CODE_DOMAIN)


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("reference-provider output must use a .pt suffix")
    report = destination.with_suffix(".json")
    if destination.exists() or report.exists():
        raise FileExistsError("refusing to overwrite reference-provider output")
    worktree = find_git_worktree(Path(__file__))
    resolved = destination.expanduser().resolve()
    if worktree is not None:
        root = worktree.resolve()
        if resolved == root or root in resolved.parents:
            relative = resolved.relative_to(root)
            if not relative.parts or relative.parts[0] not in (
                ".local-runs",
                "local-runs",
            ):
                raise ValueError(
                    "reference-provider tensor outputs in the worktree must "
                    "stay under an ignored local-runs directory"
                )
    return destination


def _stage_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _publish_artifact(
    state: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite reference-provider output")
    tensor_stage = _stage_path(output)
    report_stage = _stage_path(report_path)
    published: list[Path] = []
    try:
        torch.save(dict(state), tensor_stage)
        report = {
            **dict(report_payload),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": _file_sha256(tensor_stage),
                "tensor_file_bytes": tensor_stage.stat().st_size,
                "report_file": str(report_path),
                "committable": False,
            },
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        with report_stage.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tensor_stage, output)
        published.append(output)
        os.link(report_stage, report_path)
        published.append(report_path)
        return report
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        tensor_stage.unlink(missing_ok=True)
        report_stage.unlink(missing_ok=True)


def _claim_synthetic_assessment(
    *,
    protocol: SyntheticReferenceProtocol,
    basis_payload_sha256: str,
    source_model_sha256: str,
    ledger_dir: Path | str,
) -> dict[str, object]:
    """Irreversibly claim the full frozen assessment before materialization."""

    assessment = tuple(
        probe for probe in protocol.probes if probe.role == "assessment"
    )
    if len(assessment) != 88 or tuple(
        probe.ordinal for probe in assessment
    ) != tuple(range(88)):
        raise ValueError("assessment claim requires the full frozen 88 probes")
    assessment_panel_spec_sha256 = _require_sha256(
        protocol.assessment_panel_spec_sha256,
        label="assessment panel specification",
    )
    identity = {
        "schema": (
            "fisher_graph.gemma3_l3_l4_reference_provider_"
            "assessment_claim.v2"
        ),
        "format_version": _FORMAT_VERSION,
        "assessment_panel_spec_sha256": assessment_panel_spec_sha256,
        "basis_payload_sha256": _require_sha256(
            basis_payload_sha256,
            label="assessment basis payload",
        ),
        "source_model_sha256": _require_sha256(
            source_model_sha256,
            label="assessment source model",
        ),
        "role": "full_assessment_one_shot",
        "probe_count": len(assessment),
        "ordered_probe_sha256s": tuple(
            probe.artifact_sha256 for probe in assessment
        ),
        "candidate_independent": True,
        "output_independent": True,
        "subset_independent": True,
        "claim_survives_later_failure": True,
    }
    claim_sha256 = _json_sha256(
        identity,
        domain=_ASSESSMENT_CLAIM_DOMAIN,
    )
    directory = Path(ledger_dir)
    directory.mkdir(parents=True, exist_ok=True)
    claim_path = directory / (
        f"full-{assessment_panel_spec_sha256}-"
        f"{basis_payload_sha256}-{source_model_sha256}.claim.json"
    )
    payload = {
        **identity,
        "synthetic_protocol_sha256": _require_sha256(
            protocol.protocol_sha256,
            label="synthetic protocol provenance",
        ),
        "claim_sha256": claim_sha256,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            "the full frozen synthetic assessment is already claimed for "
            "this basis, source model, and assessment panel"
        ) from error
    try:
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return {
        "claim_file": str(claim_path),
        "claim_file_sha256": _file_sha256(claim_path),
        "claim_identity_sha256": claim_sha256,
        "assessment_panel_spec_sha256": assessment_panel_spec_sha256,
        "probe_count": len(assessment),
        "claim_survives_later_failure": True,
    }


@dataclass(frozen=True, slots=True)
class FrozenGemma3ReferenceProviderTrainingProtocol:
    """Preregistered training choices fixed before selection is opened."""

    steps: int = 300
    learning_rate: float = 1e-3
    base_seed: int = 20_260_728_101
    expert_count: int = 2
    expert_rank_cap: int = 16
    router_width: int = 16
    router_activation: str = "tanh"
    source_normalized_routing: bool = True
    target_scale_floor: float = 1e-8
    null_scale_floor: float = 1e-8
    log_rms_scale_floor: float = 1e-8
    support_relative_margin: float = 1e-6
    support_absolute_margin: float = 1e-9
    feature_gauge: str = (
        "live_realized_fisher_sigma_coordinates_centered_fit_identity_metric"
    )
    target_gauge: str = "fit_centered_per_mode_population_standard_deviation"
    metric_gauge: str = (
        "raw_balanced_l4_modal_coordinates_times_sqrt_frozen_singular_values"
    )
    position_control: str = NORMALIZED_POSITION_BIN_SEMANTICS
    position_bin_count: int = 16
    support_rule: str = (
        "fit_max_l2_radius_of_encoded_nonconstant_features_plus_frozen_margin"
    )
    manifold_lift: str = (
        "neutral_mean_direction_unit_rms_active_fisher_seed_unit_rms_"
        "nonnull_radial_absolute_gain_null_v1"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "expert_count",
            "expert_rank_cap",
            "router_width",
            "position_bin_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if type(self.base_seed) is not int or self.base_seed < 0:
            raise ValueError("base_seed must be nonnegative")
        for name in (
            "learning_rate",
            "target_scale_floor",
            "null_scale_floor",
            "log_rms_scale_floor",
            "support_relative_margin",
            "support_absolute_margin",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.router_activation != "tanh":
            raise ValueError("the frozen router activation must be tanh")
        if self.source_normalized_routing is not True:
            raise ValueError(
                "the frozen executor must use source-normalized routing"
            )
        for name in (
            "feature_gauge",
            "target_gauge",
            "metric_gauge",
            "position_control",
            "support_rule",
            "manifold_lift",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ):
                raise ValueError(f"{name} must be nonempty")
        expected = _json_sha256(
            self._payload(),
            domain=_TRAINING_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="training protocol",
                )
                != expected
            ):
                raise ValueError("training protocol hash mismatch")
        object.__setattr__(self, "artifact_sha256", expected)

    def _payload(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "base_seed": self.base_seed,
            "expert_count": self.expert_count,
            "expert_rank_cap": self.expert_rank_cap,
            "router_width": self.router_width,
            "router_activation": self.router_activation,
            "source_normalized_routing": self.source_normalized_routing,
            "target_scale_floor": self.target_scale_floor,
            "null_scale_floor": self.null_scale_floor,
            "log_rms_scale_floor": self.log_rms_scale_floor,
            "support_relative_margin": self.support_relative_margin,
            "support_absolute_margin": self.support_absolute_margin,
            "feature_gauge": self.feature_gauge,
            "target_gauge": self.target_gauge,
            "metric_gauge": self.metric_gauge,
            "position_control": self.position_control,
            "position_bin_count": self.position_bin_count,
            "support_rule": self.support_rule,
            "manifold_lift": self.manifold_lift,
            "selection_data_can_change_training": False,
            "early_stopping": False,
            "fit_schedule": "deterministic_full_batch_fixed_steps",
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: Mapping[str, object],
    ) -> FrozenGemma3ReferenceProviderTrainingProtocol:
        constructor_fields = {
            "steps",
            "learning_rate",
            "base_seed",
            "expert_count",
            "expert_rank_cap",
            "router_width",
            "router_activation",
            "source_normalized_routing",
            "target_scale_floor",
            "null_scale_floor",
            "log_rms_scale_floor",
            "support_relative_margin",
            "support_absolute_margin",
            "feature_gauge",
            "target_gauge",
            "metric_gauge",
            "position_control",
            "position_bin_count",
            "support_rule",
            "manifold_lift",
            "artifact_sha256",
        }
        expected = constructor_fields | {
            "selection_data_can_change_training",
            "early_stopping",
            "fit_schedule",
        }
        if set(raw) != expected:
            raise ValueError("training protocol state fields drifted")
        if (
            raw["selection_data_can_change_training"] is not False
            or raw["early_stopping"] is not False
            or raw["fit_schedule"]
            != "deterministic_full_batch_fixed_steps"
        ):
            raise ValueError("training protocol firewall fields drifted")
        return cls(
            **{
                name: raw[name]
                for name in constructor_fields
            }  # type: ignore[arg-type]
        )

    def executor_config(
        self,
        *,
        source_rank: int,
        target_rank: int,
        null_modes: int,
    ) -> GatedCausalModalExecutorConfig:
        if (
            type(source_rank) is not int
            or type(target_rank) is not int
            or type(null_modes) is not int
            or source_rank <= 0
            or target_rank <= 0
            or null_modes <= 0
        ):
            raise ValueError("executor ranks must be positive")
        return GatedCausalModalExecutorConfig(
            input_modes=source_rank + null_modes + 2,
            output_modes=target_rank,
            expert_count=self.expert_count,
            expert_rank=min(
                self.expert_rank_cap,
                source_rank,
                target_rank,
            ),
            router_width=self.router_width,
            same_position_skip=False,
            max_positive_lag=None,
            router_activation="tanh",
            source_normalized_routing=self.source_normalized_routing,
        )


@dataclass(frozen=True, slots=True)
class _MeasuredSyntheticProbe:
    probe: SyntheticReferenceProbe
    requested_materialization_sha256: str
    modal_coordinates: Tensor
    null_coordinates: Tensor
    row_rms: Tensor
    target_modes: Tensor
    logical_positions: Tensor
    valid_mask: Tensor
    lift_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.probe, SyntheticReferenceProbe):
            raise TypeError("probe must be a SyntheticReferenceProbe")
        _require_sha256(
            self.requested_materialization_sha256,
            label="requested materialization",
        )
        for name, ndim in (
            ("modal_coordinates", 3),
            ("null_coordinates", 3),
            ("row_rms", 2),
            ("target_modes", 3),
            ("logical_positions", 2),
            ("valid_mask", 2),
        ):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != ndim:
                raise TypeError(f"{name} has invalid tensor geometry")
        if self.modal_coordinates.shape != (
            1,
            self.probe.sequence_length,
            _MODAL_RANK,
        ):
            raise ValueError("measured source modes have invalid geometry")
        if self.target_modes.shape != (
            1,
            self.probe.sequence_length,
            _TARGET_RANK,
        ):
            raise ValueError("measured target modes have invalid geometry")
        if self.null_coordinates.shape[:2] != self.row_rms.shape:
            raise ValueError("null and RMS rows differ")
        if self.logical_positions.shape != self.row_rms.shape:
            raise ValueError("logical positions differ from measured rows")
        if self.valid_mask.shape != self.row_rms.shape:
            raise ValueError("valid mask differs from measured rows")
        if not isinstance(self.lift_metadata, Mapping):
            raise TypeError("lift metadata must be a mapping")


def _role_probes(
    protocol: SyntheticReferenceProtocol,
    role: str,
) -> tuple[SyntheticReferenceProbe, ...]:
    result = tuple(probe for probe in protocol.probes if probe.role == role)
    if not result:
        raise ValueError(f"synthetic role {role!r} is empty")
    return result


def _synthetic_binding_sha256(
    *,
    basis: Gemma3L3L4BasisPackage,
    protocol: SyntheticReferenceProtocol,
    training: FrozenGemma3ReferenceProviderTrainingProtocol,
    norm_sha256: str,
    rate_point: CandidateRatePoint,
) -> str:
    return _json_sha256(
        {
            "basis_payload_sha256": basis.basis_payload_sha256,
            "source_model_sha256": basis.source_model_sha256,
            "synthetic_protocol_sha256": protocol.protocol_sha256,
            "training_protocol_sha256": training.artifact_sha256,
            "pre_feedforward_norm_sha256": _require_sha256(
                norm_sha256,
                label="pre-feedforward norm",
            ),
            "rate_point": rate_point.state_dict(),
            "source_site": "layer.3.attention.output",
            "target_site": "layer.4.mlp.normalized_input",
            "target_reference": "l3_mlp_operator_output_clamped_to_frozen_mean",
        },
        domain=_SYNTHETIC_BINDING_DOMAIN,
    )


def _population_center_scale(
    rows: Sequence[Tensor],
    *,
    floor: float,
) -> tuple[Tensor, Tensor]:
    if not rows:
        raise ValueError("statistics require nonempty rows")
    matrix = torch.cat(
        [
            value.detach().to(device="cpu", dtype=torch.float64).reshape(
                -1, value.shape[-1]
            )
            for value in rows
        ],
        dim=0,
    )
    center = matrix.mean(dim=0)
    scale = (matrix - center).square().mean(dim=0).sqrt()
    if bool((scale <= floor).any()):
        raise ValueError("fit statistics contain a degenerate scale")
    return center.contiguous(), scale.contiguous()


def _scalar_center_scale(
    rows: Sequence[Tensor],
    *,
    floor: float,
) -> tuple[Tensor, Tensor]:
    return _population_center_scale(rows, floor=floor)


def _provider_batch(
    measured: Sequence[_MeasuredSyntheticProbe],
    *,
    split: str,
    source_rank: int,
    target_rank: int,
    synthetic_binding_sha256: str,
) -> tuple[SyntheticReferenceBatch, ...]:
    by_length: dict[int, list[_MeasuredSyntheticProbe]] = {}
    for value in measured:
        by_length.setdefault(value.probe.sequence_length, []).append(value)
    result: list[SyntheticReferenceBatch] = []
    for length in sorted(by_length):
        rows = sorted(
            by_length[length],
            key=lambda value: value.probe.ordinal,
        )
        result.append(
            SyntheticReferenceBatch(
                split=split,  # type: ignore[arg-type]
                modal_coordinates=torch.cat(
                    [
                        value.modal_coordinates[..., :source_rank]
                        for value in rows
                    ],
                    dim=0,
                ),
                null_coordinates=torch.cat(
                    [value.null_coordinates for value in rows],
                    dim=0,
                ),
                row_rms=torch.cat(
                    [value.row_rms for value in rows],
                    dim=0,
                ),
                target_modes=torch.cat(
                    [
                        value.target_modes[..., :target_rank]
                        for value in rows
                    ],
                    dim=0,
                ),
                logical_positions=torch.cat(
                    [value.logical_positions for value in rows],
                    dim=0,
                ),
                valid_mask=torch.cat(
                    [value.valid_mask for value in rows],
                    dim=0,
                ),
                synthetic_binding_sha256=synthetic_binding_sha256,
            )
        )
    return tuple(result)


def _padded_structural_batch(
    measured: Sequence[_MeasuredSyntheticProbe],
    *,
    split: str,
    source_rank: int,
    target_rank: int,
    synthetic_binding_sha256: str,
) -> SyntheticReferenceBatch:
    """Prefix-pad every probe into one batch so padding checks are nonvacuous."""

    if not measured:
        raise ValueError("padded structural validation requires measured probes")
    ordered = tuple(sorted(measured, key=lambda value: value.probe.ordinal))
    maximum_length = max(value.probe.sequence_length for value in ordered)
    batch_size = len(ordered)
    null_modes = int(ordered[0].null_coordinates.shape[-1])
    modal = torch.zeros(
        batch_size,
        maximum_length,
        source_rank,
        dtype=torch.float64,
    )
    null = torch.zeros(
        batch_size,
        maximum_length,
        null_modes,
        dtype=torch.float64,
    )
    row_rms = torch.zeros(
        batch_size,
        maximum_length,
        dtype=torch.float64,
    )
    target = torch.zeros(
        batch_size,
        maximum_length,
        target_rank,
        dtype=torch.float64,
    )
    positions = torch.full(
        (batch_size, maximum_length),
        -1,
        dtype=torch.int64,
    )
    valid = torch.zeros(
        batch_size,
        maximum_length,
        dtype=torch.bool,
    )
    for row, value in enumerate(ordered):
        length = value.probe.sequence_length
        start = maximum_length - length
        modal[row, start:] = value.modal_coordinates[
            0, :, :source_rank
        ]
        null[row, start:] = value.null_coordinates[0]
        row_rms[row, start:] = value.row_rms[0]
        target[row, start:] = value.target_modes[0, :, :target_rank]
        positions[row, start:] = value.logical_positions[0]
        valid[row, start:] = value.valid_mask[0]
    invalid_row_count = int((~valid).sum().item())
    if invalid_row_count <= 0:
        raise ValueError(
            "padded structural validation must contain invalid prefix rows"
        )
    return SyntheticReferenceBatch(
        split=split,  # type: ignore[arg-type]
        modal_coordinates=modal,
        null_coordinates=null,
        row_rms=row_rms,
        target_modes=target,
        logical_positions=positions,
        valid_mask=valid,
        synthetic_binding_sha256=synthetic_binding_sha256,
    )


def _feature_codec(
    fit: Sequence[_MeasuredSyntheticProbe],
    *,
    source_rank: int,
    source_binding_sha256: str,
    training: FrozenGemma3ReferenceProviderTrainingProtocol,
) -> ReferenceProviderFeatureCodec:
    modal_rows = [
        value.modal_coordinates[..., :source_rank] for value in fit
    ]
    modal_center, _modal_scale = _population_center_scale(
        modal_rows,
        floor=0.0,
    )
    null_center, null_scale = _scalar_center_scale(
        [value.null_coordinates for value in fit],
        floor=training.null_scale_floor,
    )
    log_rms = [torch.log(value.row_rms).unsqueeze(-1) for value in fit]
    log_center, log_scale = _scalar_center_scale(
        log_rms,
        floor=training.log_rms_scale_floor,
    )
    return ReferenceProviderFeatureCodec(
        modal_center=modal_center,
        modal_whitener=torch.eye(source_rank, dtype=torch.float64),
        null_center=null_center,
        null_scale=null_scale,
        log_rms_center=float(log_center[0]),
        log_rms_scale=float(log_scale[0]),
        source_binding_sha256=source_binding_sha256,
    )


def _load_live_dependencies(
    *,
    basis_package_path: Path | str,
    basis_package_file_sha256: str,
    basis_package_payload_sha256: str,
    model_id: str,
    revision: str,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
) -> tuple[
    Gemma3L3L4BasisPackage,
    Gemma3CausalLMAdapter,
    nn.Module,
    nn.Module,
    float,
]:
    basis = load_gemma3_l3_l4_basis_package(
        basis_package_path,
        expected_file_sha256=basis_package_file_sha256,
        expected_payload_sha256=basis_package_payload_sha256,
    )
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != basis.source_model_sha256:
        raise ValueError("live Gemma fingerprint differs from basis package")
    if adapter.module.training or any(
        parameter.requires_grad for parameter in adapter.module.parameters()
    ):
        raise ValueError("reference-provider teacher requires frozen eval Gemma")
    layer3_spec = adapter.layer("layer.3")
    transformer = layer3_spec.transformer
    if (
        transformer is None
        or transformer.feed_forward_input_norm.kind != "rms_norm"
        or transformer.feed_forward_input_norm.scale_parameterization
        != "unit_offset"
    ):
        raise ValueError("live L3 RMSNorm semantics drifted")
    layer3 = adapter.source_module("layer.3")
    pre_ff3 = getattr(layer3, "pre_feedforward_layernorm", None)
    post_ff3 = getattr(layer3, "post_feedforward_layernorm", None)
    if not isinstance(pre_ff3, nn.Module) or not isinstance(
        post_ff3, nn.Module
    ):
        raise TypeError("live Gemma L3 normalization modules are missing")
    layer4 = adapter.source_module("layer.4")
    for name in (
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
    ):
        if not isinstance(getattr(layer4, name, None), nn.Module):
            raise TypeError("live Gemma L4 attention prefix is incomplete")
    return (
        basis,
        adapter,
        pre_ff3,
        post_ff3,
        transformer.feed_forward_input_norm.epsilon,
    )


def _measure_synthetic_role(
    *,
    role: str,
    protocol: SyntheticReferenceProtocol,
    basis: Gemma3L3L4BasisPackage,
    adapter: Gemma3CausalLMAdapter,
    pre_ff3: nn.Module,
    post_ff3: nn.Module,
    epsilon: float,
) -> tuple[tuple[_MeasuredSyntheticProbe, ...], dict[str, object]]:
    probes = _role_probes(protocol, role)
    specifications = {
        probe.artifact_sha256: probe for probe in probes
    }
    materialized = materialize_synthetic_reference_batches(
        protocol,
        probes,
    )
    first_parameter = next(adapter.module.parameters(), None)
    if first_parameter is None or not first_parameter.is_floating_point():
        raise TypeError("live Gemma model has no floating parameters")
    device = first_parameter.device
    dtype = first_parameter.dtype
    y3_mean = basis.y3_mean.to(device=device, dtype=dtype)
    x4_mean = basis.x4_mean.to(device=device, dtype=dtype)
    r4 = basis.R4[:_TARGET_RANK].to(device=device, dtype=dtype)
    segment4 = adapter.segment("layer.4")
    measured: list[_MeasuredSyntheticProbe] = []
    lift_rows: list[dict[str, object]] = []
    model_before = adapter.model_fingerprint()
    norm_before = module_state_fingerprint(pre_ff3)
    for request_batch in materialized:
        batch_probes = tuple(
            specifications[digest]
            for digest in request_batch.probe_artifact_sha256s
        )
        lift = lift_synthetic_reference_batch_to_gemma3_manifold(
            basis,
            pre_ff3,
            epsilon=epsilon,
            batch=request_batch,
            probes=batch_probes,
        )
        lift.validate_integrity()
        batch_size = lift.batch_size
        length = lift.sequence_length
        positions = torch.arange(
            length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)
        valid = torch.ones(
            batch_size,
            length,
            dtype=torch.bool,
            device=device,
        )
        placeholder = torch.zeros(
            batch_size,
            length,
            basis.residual_width,
            dtype=dtype,
            device=device,
        )
        sequence = adapter.prepare_sequence(
            {
                "inputs_embeds": placeholder,
                "attention_mask": valid,
                "position_ids": positions,
            }
        )
        del placeholder
        hidden = lift.hidden_states.to(device=device, dtype=dtype)
        y3 = y3_mean.view(1, 1, -1).expand(
            batch_size,
            length,
            -1,
        )
        with torch.no_grad():
            hidden3_reference = hidden + post_ff3(y3)
            x4 = adapter.run_attention_prefix(
                segment4,
                hidden3_reference,
                sequence,
            ).normalized_mlp_input
            target = (x4 - x4_mean.view(1, 1, -1)) @ r4.T
        target64 = target.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        positions64 = positions.detach().to(device="cpu")
        valid64 = valid.detach().to(device="cpu")
        lift_metadata = lift.metadata()
        lift_rows.append(lift_metadata)
        for index, probe in enumerate(batch_probes):
            measured.append(
                _MeasuredSyntheticProbe(
                    probe=probe,
                    requested_materialization_sha256=(
                        request_batch.probe_tensor_sha256s[index]
                    ),
                    modal_coordinates=(
                        lift.absolute_realized_standardized_modes[
                            index : index + 1
                        ]
                    ),
                    null_coordinates=(
                        lift.normalized_null_features[index : index + 1]
                    ),
                    row_rms=lift.row_rms[index : index + 1],
                    target_modes=target64[index : index + 1],
                    logical_positions=positions64[index : index + 1],
                    valid_mask=valid64[index : index + 1],
                    lift_metadata={
                        "lift_artifact_sha256": lift.artifact_sha256,
                        "lift_diagnostics_sha256": (
                            lift.diagnostics_sha256
                        ),
                        "formula_version": lift.formula_version,
                        "probe_id": probe.probe_id,
                    },
                )
            )
    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_before
    ):
        raise RuntimeError("live model changed during synthetic measurement")
    measured.sort(key=lambda value: value.probe.ordinal)
    if tuple(value.probe.ordinal for value in measured) != tuple(
        range(len(probes))
    ):
        raise RuntimeError("measured role order drifted")
    return tuple(measured), {
        "role": role,
        "probe_count": len(measured),
        "family_counts": {
            family: sum(
                value.probe.family == family for value in measured
            )
            for family in sorted({value.probe.family for value in measured})
        },
        "materialized_batch_sha256s": tuple(
            value.artifact_sha256 for value in materialized
        ),
        "manifold_lift_sha256s": tuple(
            value["artifact_sha256"] for value in lift_rows
        ),
        "manifold_lift_diagnostics": tuple(
            value["diagnostics"] for value in lift_rows
        ),
        "l4_attention_prefix_execution_count": len(materialized),
        "l3_mlp_body_execution_count": 0,
        "l4_mlp_body_execution_count": 0,
        "tokenizer_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "formula_version": MANIFOLD_LIFT_FORMULA_VERSION,
    }


@dataclass(frozen=True, slots=True)
class _FittedReferenceCandidate:
    candidate_id: str
    rate_point: CandidateRatePoint
    synthetic_binding_sha256: str
    plan: StateConditionedReferenceProviderPlan
    support_radius: float


def _candidate_id(rate_point: CandidateRatePoint) -> str:
    return (
        f"{rate_point.kind}-"
        f"r{rate_point.source_rank:02d}-"
        f"t{rate_point.target_rank:02d}"
    )


def _fisher_metric_weight(
    basis: Gemma3L3L4BasisPackage,
) -> Tensor:
    """Return the diagonal square-root gauge for the frozen L4 Fisher metric."""

    spectrum = basis.S4[:_TARGET_RANK].detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if (
        spectrum.shape != (_TARGET_RANK,)
        or not bool(torch.isfinite(spectrum).all())
        or bool((spectrum <= 0.0).any())
    ):
        raise ValueError(
            "the retained L4 Fisher spectrum must be finite and positive"
        )
    return torch.sqrt(spectrum).contiguous()


def _standardized_gauge_sha256(
    *,
    basis: Gemma3L3L4BasisPackage,
    training: FrozenGemma3ReferenceProviderTrainingProtocol,
    metric_weight: Tensor,
) -> str:
    return _json_sha256(
        {
            "basis_payload_sha256": basis.basis_payload_sha256,
            "source_model_sha256": basis.source_model_sha256,
            "training_protocol_sha256": training.artifact_sha256,
            "metric_weight_sha256": _tensor_sha256(metric_weight),
            "metric_width": _TARGET_RANK,
            "metric_semantics": training.metric_gauge,
        },
        domain=_GAUGE_DOMAIN,
    )


def _full_width_probes(
    measured: Sequence[_MeasuredSyntheticProbe],
    *,
    metric_weight: Tensor,
    standardized_gauge_sha256: str,
    split: str | None = None,
    carry_collision_identity: bool = True,
) -> tuple[FullWidthReferenceProbe, ...]:
    result: list[FullWidthReferenceProbe] = []
    for value in measured:
        probe_split = value.probe.role if split is None else split
        collision_group = (
            value.probe.collision_group
            if carry_collision_identity and probe_split == "assessment"
            else None
        )
        collision_variant = (
            value.probe.collision_variant
            if carry_collision_identity and probe_split == "assessment"
            else None
        )
        result.append(
            FullWidthReferenceProbe(
                probe_id=value.probe.probe_id,
                split=probe_split,  # type: ignore[arg-type]
                family=value.probe.family,
                standardized_target=(
                    value.target_modes
                    * metric_weight.view(1, 1, -1)
                ),
                logical_positions=value.logical_positions,
                valid_mask=value.valid_mask,
                standardized_gauge_sha256=standardized_gauge_sha256,
                collision_group=collision_group,
                collision_variant=collision_variant,
            )
        )
    return tuple(result)


def _assessment_panel_binding(
    *,
    protocol: SyntheticReferenceProtocol,
    measured: Sequence[_MeasuredSyntheticProbe],
    assessment_scoring_probes: Sequence[FullWidthReferenceProbe],
    standardized_gauge_sha256: str,
) -> tuple[dict[str, object], str]:
    """Bind the complete measured assessment panel before score publication."""

    if not isinstance(protocol, SyntheticReferenceProtocol):
        raise TypeError("protocol must be SyntheticReferenceProtocol")
    measured_values = tuple(measured)
    scoring_values = tuple(assessment_scoring_probes)
    if any(
        not isinstance(value, _MeasuredSyntheticProbe)
        for value in measured_values
    ):
        raise TypeError("measured assessment rows have invalid types")
    if any(
        not isinstance(value, FullWidthReferenceProbe)
        for value in scoring_values
    ):
        raise TypeError("assessment scoring probes have invalid types")
    gauge_sha256 = _require_sha256(
        standardized_gauge_sha256,
        label="assessment standardized gauge",
    )
    expected_specs = _role_probes(protocol, "assessment")
    if (
        len(expected_specs) != 88
        or len(measured_values) != len(expected_specs)
        or len(scoring_values) != len(expected_specs)
        or tuple(value.probe.probe_id for value in measured_values)
        != tuple(value.probe_id for value in expected_specs)
        or tuple(value.probe_id for value in scoring_values)
        != tuple(value.probe_id for value in expected_specs)
        or any(value.split != "assessment" for value in scoring_values)
        or tuple(value.family for value in scoring_values)
        != tuple(value.family for value in expected_specs)
        or tuple(
            (value.collision_group, value.collision_variant)
            for value in scoring_values
        )
        != tuple(
            (value.collision_group, value.collision_variant)
            for value in expected_specs
        )
        or any(
            value.standardized_gauge_sha256 != gauge_sha256
            for value in scoring_values
        )
    ):
        raise RuntimeError(
            "measured assessment panel differs from the frozen 88 probes"
        )
    panel = {
        "schema": (
            "fisher_graph.gemma3_l3_l4_reference_provider_"
            "measured_assessment_panel.v2"
        ),
        "format_version": _FORMAT_VERSION,
        "split": "assessment",
        "probe_count": len(scoring_values),
        "ordered_probe_ids": tuple(
            value.probe_id for value in scoring_values
        ),
        "ordered_families": tuple(value.family for value in scoring_values),
        "ordered_protocol_probe_sha256s": tuple(
            value.artifact_sha256 for value in expected_specs
        ),
        "ordered_full_width_target_probe_sha256s": tuple(
            value.artifact_sha256 for value in scoring_values
        ),
        "collision_probe_count": sum(
            value.collision_group is not None for value in scoring_values
        ),
        "synthetic_protocol_sha256": protocol.protocol_sha256,
        "assessment_panel_spec_sha256": (
            protocol.assessment_panel_spec_sha256
        ),
        "standardized_gauge_sha256": gauge_sha256,
    }
    return panel, _json_sha256(panel, domain=_ASSESSMENT_PANEL_DOMAIN)


def _support_radius(
    *,
    codec: ReferenceProviderFeatureCodec,
    batches: Sequence[SyntheticReferenceBatch],
    training: FrozenGemma3ReferenceProviderTrainingProtocol,
) -> float:
    runtime = codec.prepare(dtype=torch.float64, device="cpu")
    maximum = 0.0
    with torch.no_grad():
        for batch in batches:
            encoded = runtime(
                batch.modal_coordinates,
                batch.null_coordinates,
                batch.row_rms,
                batch.valid_mask,
            )
            radii = torch.linalg.vector_norm(encoded[..., 1:], dim=-1)
            maximum = max(maximum, float(radii[batch.valid_mask].max().item()))
    return (
        maximum * (1.0 + training.support_relative_margin)
        + training.support_absolute_margin
    )


def _in_support_fraction(
    *,
    codec: ReferenceProviderFeatureCodec,
    batches: Sequence[SyntheticReferenceBatch],
    support_radius: float,
) -> float:
    runtime = codec.prepare(dtype=torch.float64, device="cpu")
    supported = 0
    total = 0
    with torch.no_grad():
        for batch in batches:
            encoded = runtime(
                batch.modal_coordinates,
                batch.null_coordinates,
                batch.row_rms,
                batch.valid_mask,
            )
            radii = torch.linalg.vector_norm(encoded[..., 1:], dim=-1)
            selected = radii[batch.valid_mask]
            supported += int((selected <= support_radius).sum().item())
            total += int(selected.numel())
    if total <= 0:
        raise ValueError("support measurement contains no valid rows")
    return supported / total


def _runtime_predictions(
    fitted: _FittedReferenceCandidate,
    measured: Sequence[_MeasuredSyntheticProbe],
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, ...]:
    runtime = fitted.plan.prepare(dtype=dtype, device="cpu")
    result: list[Tensor] = []
    with torch.no_grad():
        for value in measured:
            result.append(
                runtime(
                    value.modal_coordinates[
                        ..., : fitted.rate_point.source_rank
                    ].to(dtype=dtype),
                    value.null_coordinates.to(dtype=dtype),
                    value.row_rms.to(dtype=dtype),
                    valid_mask=value.valid_mask,
                    logical_positions=value.logical_positions,
                )
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
    return tuple(result)


def _relative_error(left: Sequence[Tensor], right: Sequence[Tensor]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("relative-error tensor sequences are not aligned")
    numerator = sum(
        float((a - b).square().sum().item())
        for a, b in zip(left, right, strict=True)
    )
    denominator = sum(
        float(value.square().sum().item()) for value in right
    )
    return math.sqrt(numerator / max(denominator, 1e-24))


def _padding_violation(
    plan: StateConditionedReferenceProviderPlan,
    batches: Sequence[SyntheticReferenceBatch],
) -> tuple[float, dict[str, object]]:
    """Measure valid-row sensitivity to invalid prefix data numerically."""

    runtime = plan.prepare(dtype=torch.float64, device="cpu")
    numerator = 0.0
    denominator = 0.0
    maximum_valid_difference = 0.0
    maximum_invalid_output = 0.0
    invalid_row_count = 0
    with torch.no_grad():
        for batch in batches:
            mask = batch.valid_mask
            invalid_row_count += int((~mask).sum().item())
            baseline = runtime(
                batch.modal_coordinates,
                batch.null_coordinates,
                batch.row_rms,
                valid_mask=mask,
                logical_positions=batch.logical_positions,
            )
            changed = runtime(
                torch.where(
                    mask.unsqueeze(-1),
                    batch.modal_coordinates,
                    torch.full_like(batch.modal_coordinates, 9_973.0),
                ),
                torch.where(
                    mask.unsqueeze(-1),
                    batch.null_coordinates,
                    torch.full_like(batch.null_coordinates, -4_113.0),
                ),
                torch.where(
                    mask,
                    batch.row_rms,
                    torch.full_like(batch.row_rms, 31.0),
                ),
                valid_mask=mask,
                logical_positions=torch.where(
                    mask,
                    batch.logical_positions,
                    torch.full_like(batch.logical_positions, -777),
                ),
            )
            valid_difference = baseline[mask] - changed[mask]
            invalid_outputs = torch.cat(
                (baseline[~mask].reshape(-1), changed[~mask].reshape(-1))
            )
            numerator += float(valid_difference.square().sum().item())
            numerator += float(invalid_outputs.square().sum().item())
            denominator += float(baseline[mask].square().sum().item())
            if valid_difference.numel():
                maximum_valid_difference = max(
                    maximum_valid_difference,
                    float(valid_difference.abs().max().item()),
                )
            if invalid_outputs.numel():
                maximum_invalid_output = max(
                    maximum_invalid_output,
                    float(invalid_outputs.abs().max().item()),
                )
    if invalid_row_count <= 0:
        raise ValueError(
            "padding validation is vacuous because no invalid rows exist"
        )
    return math.sqrt(numerator / max(denominator, 1e-24)), {
        "semantics": (
            "relative_output_change_on_valid_rows_plus_invalid_output_energy_"
            "after_mutating_invalid_prefix_features_and_positions"
        ),
        "invalid_row_count": invalid_row_count,
        "maximum_absolute_valid_row_difference": maximum_valid_difference,
        "maximum_absolute_invalid_output": maximum_invalid_output,
        "nonvacuous": True,
    }


def _selection_candidate(
    fitted: _FittedReferenceCandidate,
    *,
    measured: Sequence[_MeasuredSyntheticProbe],
    full_probes: Sequence[FullWidthReferenceProbe],
    metric_weight: Tensor,
    standardized_gauge_sha256: str,
    synthetic_batches: Sequence[SyntheticReferenceBatch],
    structural_batches: Sequence[SyntheticReferenceBatch] | None = None,
) -> tuple[FullWidthReferenceCandidate, dict[str, object]]:
    raw64 = _runtime_predictions(fitted, measured, dtype=torch.float64)
    raw32 = _runtime_predictions(fitted, measured, dtype=torch.float32)
    retained_predictions = tuple(
        FullWidthCandidatePrediction(
            probe_id=probe.probe_id,
            retained_standardized_prediction=(
                prediction
                * metric_weight[
                    : fitted.rate_point.target_rank
                ].view(1, 1, -1)
            ),
            standardized_gauge_sha256=standardized_gauge_sha256,
        )
        for probe, prediction in zip(
            full_probes,
            raw64,
            strict=True,
        )
    )
    evaluation = evaluate_state_conditioned_reference_provider(
        fitted.plan,
        (
            synthetic_batches
            if structural_batches is None
            else structural_batches
        ),
        required_split=synthetic_batches[0].split,
    )
    prepared_error = _relative_error(raw32, raw64)
    support = _in_support_fraction(
        codec=fitted.plan.feature_codec,
        batches=synthetic_batches,
        support_radius=fitted.support_radius,
    )
    padding_violation, padding_metadata = _padding_violation(
        fitted.plan,
        (
            synthetic_batches
            if structural_batches is None
            else structural_batches
        ),
    )
    structural = FullWidthStructuralMetrics(
        prepared_vs_analytic_relative_error=prepared_error,
        causality_violation=0.0 if evaluation.causal_prefix_exact else 1.0,
        padding_violation=padding_violation,
        repeat_relative_error=0.0 if evaluation.repeat_exact else 1.0,
        in_support_fraction=support,
    )
    candidate = FullWidthReferenceCandidate(
        candidate_id=fitted.candidate_id,
        source_rank=fitted.rate_point.source_rank,
        target_rank=fitted.rate_point.target_rank,
        stored_scalar_count=(
            fitted.plan.accounting().total_stored_scalar_count
        ),
        predictions=retained_predictions,
        structural_metrics=structural,
        candidate_binding_sha256=fitted.plan.artifact_sha256,
    )
    return candidate, {
        "provider_evaluation_sha256": evaluation.evaluation_sha256,
        "provider_evaluation": asdict(evaluation),
        "prepared_float32_vs_canonical_float64_relative_error": (
            prepared_error
        ),
        "support_rule": (
            "fit_max_l2_radius_of_encoded_nonconstant_features_plus_"
            "frozen_margin"
        ),
        "support_radius": fitted.support_radius,
        "in_support_fraction": support,
        "padding_validation": padding_metadata,
    }


def _deferred_collision_gates(
    gates: SyntheticReferenceGates,
) -> SyntheticReferenceGates:
    return replace(
        gates,
        minimum_collision_target_relative_difference=0.0,
    )


def _controls_from_state(
    state: Mapping[str, object],
) -> FullWidthReferenceControls:
    expected = {
        "artifact_kind",
        "format_version",
        "fit_target_center_sha256",
        "normalized_position_bin_centers_sha256",
        "fit_target_center",
        "normalized_position_bin_centers",
        "normalized_position_bin_counts",
        "fit_probe_ids",
        "fit_probe_sha256s",
        "standardized_gauge_sha256",
        "position_semantics",
        "artifact_sha256",
    }
    if not expected.issubset(state):
        raise ValueError("stored full-width controls are incomplete")
    if set(state) != expected:
        raise ValueError("stored full-width controls fields drifted")
    if (
        state["artifact_kind"]
        != "fisher_graph.full_width_reference_controls"
    ):
        raise ValueError("stored controls envelope drifted")
    if state["format_version"] != 1:
        raise ValueError("stored controls envelope drifted")
    restored = FullWidthReferenceControls(
        fit_target_center=state["fit_target_center"],  # type: ignore[arg-type]
        normalized_position_bin_centers=state[
            "normalized_position_bin_centers"
        ],  # type: ignore[arg-type]
        normalized_position_bin_counts=tuple(
            int(value)
            for value in state[  # type: ignore[union-attr]
                "normalized_position_bin_counts"
            ]
        ),
        fit_probe_ids=tuple(
            str(value)
            for value in state["fit_probe_ids"]  # type: ignore[union-attr]
        ),
        fit_probe_sha256s=tuple(
            str(value)
            for value in state["fit_probe_sha256s"]  # type: ignore[union-attr]
        ),
        standardized_gauge_sha256=str(
            state["standardized_gauge_sha256"]
        ),
        position_semantics=str(state["position_semantics"]),
        artifact_sha256=str(state["artifact_sha256"]),
    )
    canonical_state = restored.state_dict()
    if any(
        state[name] != canonical_state[name]
        for name in (
            "fit_target_center_sha256",
            "normalized_position_bin_centers_sha256",
        )
    ):
        raise ValueError("stored full-width control tensor binding drifted")
    return restored


def _validate_restored_selection_protocol_binding(
    *,
    selection: FullWidthReferenceSelection,
    controls: FullWidthReferenceControls,
    protocol: SyntheticReferenceProtocol,
    manifest: Mapping[str, object],
) -> str:
    """Authenticate the prompt-blind selection panel against its protocol."""

    if not isinstance(selection, FullWidthReferenceSelection):
        raise TypeError("selection must be FullWidthReferenceSelection")
    if not isinstance(controls, FullWidthReferenceControls):
        raise TypeError("controls must be FullWidthReferenceControls")
    if not isinstance(protocol, SyntheticReferenceProtocol):
        raise TypeError("protocol must be SyntheticReferenceProtocol")
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")

    deferred_gates = _deferred_collision_gates(protocol.gates)
    expected_gates_sha256 = full_width_reference_gates_sha256(
        deferred_gates
    )
    fit_specs = _role_probes(protocol, "fit")
    selection_specs = _role_probes(protocol, "selection")
    expected_fit_ids = tuple(sorted(probe.probe_id for probe in fit_specs))
    expected_selection_families = {
        probe.probe_id: probe.family for probe in selection_specs
    }
    expected_candidate_ids = {
        _candidate_id(rate) for rate in protocol.candidate_ladder[1:]
    }
    scores_by_id = {
        score.candidate_id: score for score in selection.candidate_scores
    }
    if (
        len(fit_specs) != 80
        or len(selection_specs) != 32
        or manifest.get("selection_sha256") != selection.artifact_sha256
        or selection.controls_artifact_sha256 != controls.artifact_sha256
        or selection.gates_sha256 != expected_gates_sha256
        or selection.collision_probe_sha256s
        or len(selection.selection_probe_sha256s) != len(selection_specs)
        or controls.fit_probe_ids != expected_fit_ids
        or len(controls.fit_probe_sha256s) != len(fit_specs)
        or set(scores_by_id) != expected_candidate_ids
        or manifest.get("collision_gate_deferred_to_sealed_assessment")
        is not True
        or manifest.get("selection_collision_threshold") != 0.0
        or manifest.get("assessment_collision_threshold")
        != protocol.gates.minimum_collision_target_relative_difference
    ):
        raise ValueError("candidate selection binding mismatch")
    for score in selection.candidate_scores:
        probe_families = {
            metric.probe_id: metric.family
            for metric in score.probe_metrics
        }
        if (
            score.gates_sha256 != expected_gates_sha256
            or score.collision_metrics
            or score.minimum_collision_target_relative_difference != 0.0
            or not score.gate_flags.collision_target_relative_difference
            or probe_families != expected_selection_families
            or len(score.probe_metrics) != len(selection_specs)
        ):
            raise ValueError(
                "candidate score or selection panel binding drifted"
            )
    return expected_gates_sha256


def compile_gemma3_l3_l4_reference_provider(
    *,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    basis_package_file_sha256: str = DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    basis_package_payload_sha256: str = (
        DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
    ),
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    training: FrozenGemma3ReferenceProviderTrainingProtocol | None = None,
) -> dict[str, object]:
    """Fit every frozen rank before opening selection; never open assessment."""

    protocol = default_synthetic_reference_protocol()
    frozen_training = (
        FrozenGemma3ReferenceProviderTrainingProtocol()
        if training is None
        else training
    )
    if not isinstance(
        frozen_training,
        FrozenGemma3ReferenceProviderTrainingProtocol,
    ):
        raise TypeError("training must be a frozen training protocol")
    destination = _validate_output_path(output)
    code_sha256s = _code_sha256s()
    code_bundle_sha256 = _code_bundle_sha256(code_sha256s)
    (
        basis,
        adapter,
        pre_ff3,
        post_ff3,
        epsilon,
    ) = _load_live_dependencies(
        basis_package_path=basis_package_path,
        basis_package_file_sha256=basis_package_file_sha256,
        basis_package_payload_sha256=basis_package_payload_sha256,
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    model_before = adapter.model_fingerprint()
    norm_sha256 = module_state_fingerprint(pre_ff3)
    fit, fit_measurement = _measure_synthetic_role(
        role="fit",
        protocol=protocol,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
    )
    target_center, target_scale = _population_center_scale(
        [value.target_modes for value in fit],
        floor=frozen_training.target_scale_floor,
    )
    metric_weight = _fisher_metric_weight(basis)
    standardized_gauge_sha256 = _standardized_gauge_sha256(
        basis=basis,
        training=frozen_training,
        metric_weight=metric_weight,
    )
    fit_full_probes = _full_width_probes(
        fit,
        metric_weight=metric_weight,
        standardized_gauge_sha256=standardized_gauge_sha256,
    )
    controls = fit_full_width_reference_controls(
        fit_probes=fit_full_probes,
        position_bin_count=frozen_training.position_bin_count,
    )

    # FIREWALL: every nonconstant plan is fit and frozen before selection is
    # materialized or its live teacher targets exist.
    fitted: list[_FittedReferenceCandidate] = []
    for index, rate_point in enumerate(protocol.candidate_ladder[1:]):
        binding = _synthetic_binding_sha256(
            basis=basis,
            protocol=protocol,
            training=frozen_training,
            norm_sha256=norm_sha256,
            rate_point=rate_point,
        )
        codec = _feature_codec(
            fit,
            source_rank=rate_point.source_rank,
            source_binding_sha256=binding,
            training=frozen_training,
        )
        fit_batches = _provider_batch(
            fit,
            split="fit",
            source_rank=rate_point.source_rank,
            target_rank=rate_point.target_rank,
            synthetic_binding_sha256=binding,
        )
        support_radius = _support_radius(
            codec=codec,
            batches=fit_batches,
            training=frozen_training,
        )
        plan = fit_state_conditioned_reference_provider(
            feature_codec=codec,
            target_center=target_center[: rate_point.target_rank],
            target_scale=target_scale[: rate_point.target_rank],
            fit_batches=fit_batches,
            executor_config=frozen_training.executor_config(
                source_rank=rate_point.source_rank,
                target_rank=rate_point.target_rank,
                null_modes=codec.null_modes,
            ),
            steps=frozen_training.steps,
            learning_rate=frozen_training.learning_rate,
            seed=frozen_training.base_seed + index,
        )
        fitted.append(
            _FittedReferenceCandidate(
                candidate_id=_candidate_id(rate_point),
                rate_point=rate_point,
                synthetic_binding_sha256=binding,
                plan=plan,
                support_radius=support_radius,
            )
        )

    selection_measured, selection_measurement = _measure_synthetic_role(
        role="selection",
        protocol=protocol,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
    )
    selection_full_probes = _full_width_probes(
        selection_measured,
        metric_weight=metric_weight,
        standardized_gauge_sha256=standardized_gauge_sha256,
    )
    selection_candidates: list[FullWidthReferenceCandidate] = []
    candidate_runtime_metadata: dict[str, object] = {}
    selection_batches_by_candidate: dict[
        str, tuple[SyntheticReferenceBatch, ...]
    ] = {}
    for value in fitted:
        batches = _provider_batch(
            selection_measured,
            split="selection",
            source_rank=value.rate_point.source_rank,
            target_rank=value.rate_point.target_rank,
            synthetic_binding_sha256=value.synthetic_binding_sha256,
        )
        selection_batches_by_candidate[value.candidate_id] = batches
        padded_batch = _padded_structural_batch(
            selection_measured,
            split="selection",
            source_rank=value.rate_point.source_rank,
            target_rank=value.rate_point.target_rank,
            synthetic_binding_sha256=value.synthetic_binding_sha256,
        )
        candidate, runtime_metadata = _selection_candidate(
            value,
            measured=selection_measured,
            full_probes=selection_full_probes,
            metric_weight=metric_weight,
            standardized_gauge_sha256=standardized_gauge_sha256,
            synthetic_batches=batches,
            structural_batches=(padded_batch,),
        )
        selection_candidates.append(candidate)
        candidate_runtime_metadata[value.candidate_id] = runtime_metadata

    selection_gates = _deferred_collision_gates(protocol.gates)
    selection = select_smallest_passing_full_width_reference_candidate(
        controls=controls,
        selection_probes=selection_full_probes,
        collision_probes=(),
        candidates=tuple(selection_candidates),
        gates=selection_gates,
    )
    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_sha256
        or _code_sha256s() != code_sha256s
    ):
        raise RuntimeError(
            "model, normalization, or code changed during provider compile"
        )

    plans_by_id = {value.candidate_id: value.plan for value in fitted}
    rates_by_id = {
        value.candidate_id: value.rate_point.state_dict() for value in fitted
    }
    support_by_id = {
        value.candidate_id: value.support_radius for value in fitted
    }
    selected_id = selection.selected_candidate_id
    selected_plan_sha256 = (
        None
        if selected_id is None
        else plans_by_id[selected_id].artifact_sha256
    )
    manifest = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "basis_package_file_sha256": basis_package_file_sha256,
        "basis_package_payload_sha256": basis.basis_payload_sha256,
        "source_model_sha256": basis.source_model_sha256,
        "synthetic_protocol_sha256": protocol.protocol_sha256,
        "assessment_panel_spec_sha256": (
            protocol.assessment_panel_spec_sha256
        ),
        "training_protocol_sha256": frozen_training.artifact_sha256,
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": code_bundle_sha256,
        "pre_feedforward_norm_sha256": norm_sha256,
        "standardized_gauge_sha256": standardized_gauge_sha256,
        "metric_weight_sha256": _tensor_sha256(metric_weight),
        "metric_gauge": frozen_training.metric_gauge,
        "target_center_sha256": _tensor_sha256(target_center),
        "target_scale_sha256": _tensor_sha256(target_scale),
        "controls_sha256": controls.artifact_sha256,
        "candidate_plan_sha256s": {
            name: plan.artifact_sha256
            for name, plan in sorted(plans_by_id.items())
        },
        "candidate_rate_points": {
            name: rates_by_id[name] for name in sorted(rates_by_id)
        },
        "candidate_support_radii": {
            name: support_by_id[name] for name in sorted(support_by_id)
        },
        "selection_sha256": selection.artifact_sha256,
        "selected_candidate_id": selected_id,
        "selected_plan_sha256": selected_plan_sha256,
        "assessment_opened": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "collision_gate_deferred_to_sealed_assessment": True,
        "selection_collision_threshold": 0.0,
        "assessment_collision_threshold": (
            protocol.gates.minimum_collision_target_relative_difference
        ),
        "source_scope": _SOURCE_SCOPE,
        "scientific_claim": (
            "prompt_blind_synthetic_fit_and_selection_only"
        ),
    }
    logical_artifact_sha256 = _json_sha256(
        manifest,
        domain=_CANDIDATE_DOMAIN,
    )
    state = {
        "manifest": manifest,
        "artifact_sha256": logical_artifact_sha256,
        "target_center": target_center,
        "target_scale": target_scale,
        "metric_weight": metric_weight,
        "controls_state": controls.state_dict(),
        "plan_states": {
            name: plan.state_dict()
            for name, plan in sorted(plans_by_id.items())
        },
        "selection_state": selection.state_dict(),
        "training_protocol_state": frozen_training.state_dict(),
        "synthetic_protocol_state": protocol.state_dict(),
    }
    report_payload = {
        **manifest,
        "artifact_sha256": logical_artifact_sha256,
        "fit_measurement": fit_measurement,
        "selection_measurement": selection_measurement,
        "candidate_runtime_metadata": candidate_runtime_metadata,
        "selection": selection.state_dict(),
        "rate_curve": [
            {
                "candidate_id": score.candidate_id,
                "source_rank": score.source_rank,
                "target_rank": score.target_rank,
                "stored_scalar_count": score.stored_scalar_count,
                "fisher_weighted_relative_error": (
                    score.fisher_weighted_relative_error
                ),
                "reference_cosine": score.reference_cosine,
                "error_reduction_vs_constant": (
                    score.error_reduction_vs_constant
                ),
                "error_reduction_vs_position_only": (
                    score.error_reduction_vs_position_only
                ),
                "maximum_per_probe_p90_relative_error": (
                    score.maximum_per_probe_p90_relative_error
                ),
                "worst_family_relative_error": (
                    score.worst_family_relative_error
                ),
                "gate_flags": score.gate_flags.state_dict(),
                "passed_selection_gates": score.passed,
            }
            for score in selection.candidate_scores
        ],
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_tokenizer": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_natural_activation_rows": False,
            "contains_provider_parameters": True,
            "committable": False,
        },
    }
    return _publish_artifact(state, report_payload, output=destination)


@dataclass(frozen=True, slots=True)
class _LoadedReferenceCompilation:
    manifest: Mapping[str, object]
    artifact_sha256: str
    file_sha256: str
    report_sha256: str
    target_center: Tensor
    target_scale: Tensor
    metric_weight: Tensor
    controls: FullWidthReferenceControls
    selection: FullWidthReferenceSelection
    selected_id: str
    selected_rate_point: CandidateRatePoint
    selected_plan: StateConditionedReferenceProviderPlan
    selected_support_radius: float
    protocol: SyntheticReferenceProtocol
    training: FrozenGemma3ReferenceProviderTrainingProtocol


def _load_reference_compilation(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
) -> _LoadedReferenceCompilation:
    candidate_path = Path(path)
    payload = _read_regular_file(candidate_path)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    if file_sha256 != _require_sha256(
        expected_file_sha256,
        label="candidate file",
    ):
        raise ValueError("candidate file SHA-256 mismatch")
    report_payload = _read_regular_file(candidate_path.with_suffix(".json"))
    report = json.loads(report_payload.decode("utf-8"))
    if not isinstance(report, Mapping):
        raise TypeError("candidate report must be a mapping")
    supplied_report_sha256 = report.get("report_sha256")
    report_without_hash = dict(report)
    report_without_hash.pop("report_sha256", None)
    computed_report_sha256 = _json_sha256(
        report_without_hash,
        domain=_REPORT_DOMAIN,
    )
    if (
        supplied_report_sha256 != computed_report_sha256
        or supplied_report_sha256
        != _require_sha256(
            expected_report_sha256,
            label="candidate report",
        )
    ):
        raise ValueError("candidate report SHA-256 mismatch")
    report_artifact = report.get("artifact")
    if (
        not isinstance(report_artifact, Mapping)
        or report_artifact.get("tensor_file_sha256") != file_sha256
    ):
        raise ValueError("candidate report does not bind the tensor file")

    raw = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(raw, Mapping) or set(raw) != {
        "manifest",
        "artifact_sha256",
        "target_center",
        "target_scale",
        "metric_weight",
        "controls_state",
        "plan_states",
        "selection_state",
        "training_protocol_state",
        "synthetic_protocol_state",
    }:
        raise ValueError("candidate tensor fields do not match frozen format")
    manifest = raw["manifest"]
    if not isinstance(manifest, Mapping):
        raise TypeError("candidate manifest must be a mapping")
    artifact_sha256 = _json_sha256(
        manifest,
        domain=_CANDIDATE_DOMAIN,
    )
    if (
        raw["artifact_sha256"] != artifact_sha256
        or report.get("artifact_sha256") != artifact_sha256
        or manifest.get("schema") != _SCHEMA
        or manifest.get("format_version") != _FORMAT_VERSION
    ):
        raise ValueError("candidate logical artifact binding mismatch")
    code_sha256s = manifest.get("code_sha256s")
    if (
        not isinstance(code_sha256s, Mapping)
        or dict(code_sha256s) != _code_sha256s()
        or manifest.get("code_bundle_sha256")
        != _code_bundle_sha256(dict(code_sha256s))
    ):
        raise ValueError("candidate code binding differs from live code")
    protocol_state = raw["synthetic_protocol_state"]
    if not isinstance(protocol_state, Mapping):
        raise TypeError("stored synthetic protocol must be a mapping")
    protocol = SyntheticReferenceProtocol.from_state_dict(protocol_state)
    if (
        protocol.protocol_sha256
        != default_synthetic_reference_protocol().protocol_sha256
        or manifest.get("synthetic_protocol_sha256")
        != protocol.protocol_sha256
        or manifest.get("assessment_panel_spec_sha256")
        != protocol.assessment_panel_spec_sha256
    ):
        raise ValueError("candidate synthetic protocol drifted")
    training_state = raw["training_protocol_state"]
    if not isinstance(training_state, Mapping):
        raise TypeError("stored training protocol must be a mapping")
    training = FrozenGemma3ReferenceProviderTrainingProtocol.from_state_dict(
        training_state
    )
    if (
        manifest.get("training_protocol_sha256")
        != training.artifact_sha256
    ):
        raise ValueError("candidate training protocol drifted")
    target_center = raw["target_center"]
    target_scale = raw["target_scale"]
    metric_weight = raw["metric_weight"]
    if (
        not isinstance(target_center, Tensor)
        or not isinstance(target_scale, Tensor)
        or not isinstance(metric_weight, Tensor)
    ):
        raise TypeError("candidate target gauge tensors are missing")
    target_center = target_center.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    target_scale = target_scale.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    metric_weight = metric_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if (
        target_center.shape != (_TARGET_RANK,)
        or target_scale.shape != (_TARGET_RANK,)
        or metric_weight.shape != (_TARGET_RANK,)
        or not bool(torch.isfinite(target_center).all())
        or not bool(torch.isfinite(target_scale).all())
        or not bool(torch.isfinite(metric_weight).all())
        or bool((target_scale <= 0).any())
        or bool((metric_weight <= 0).any())
        or manifest.get("target_center_sha256")
        != _tensor_sha256(target_center)
        or manifest.get("target_scale_sha256")
        != _tensor_sha256(target_scale)
        or manifest.get("metric_weight_sha256")
        != _tensor_sha256(metric_weight)
        or manifest.get("metric_gauge") != training.metric_gauge
    ):
        raise ValueError("candidate target gauge is invalid")
    controls_state = raw["controls_state"]
    if not isinstance(controls_state, Mapping):
        raise TypeError("candidate controls state must be a mapping")
    controls = _controls_from_state(controls_state)
    if (
        manifest.get("controls_sha256") != controls.artifact_sha256
        or manifest.get("standardized_gauge_sha256")
        != controls.standardized_gauge_sha256
    ):
        raise ValueError("candidate controls binding mismatch")
    selection_state = raw["selection_state"]
    if not isinstance(selection_state, Mapping):
        raise TypeError("candidate selection state must be a mapping")
    selection = FullWidthReferenceSelection.from_state_dict(selection_state)
    _validate_restored_selection_protocol_binding(
        selection=selection,
        controls=controls,
        protocol=protocol,
        manifest=manifest,
    )
    selected_id = manifest.get("selected_candidate_id")
    selected_plan_sha256 = manifest.get("selected_plan_sha256")
    if (
        not isinstance(selected_id, str)
        or not selected_id
        or not isinstance(selected_plan_sha256, str)
        or selection.selected_candidate_id != selected_id
    ):
        raise ValueError(
            "compiled rate ladder selected no candidate; assessment is closed"
        )
    plan_states = raw["plan_states"]
    plan_sha256s = manifest.get("candidate_plan_sha256s")
    rate_points = manifest.get("candidate_rate_points")
    support_radii = manifest.get("candidate_support_radii")
    expected_rates = {
        _candidate_id(rate): rate
        for rate in protocol.candidate_ladder[1:]
    }
    expected_ids = set(expected_rates)
    scores_by_id = {
        score.candidate_id: score for score in selection.candidate_scores
    }
    if (
        not isinstance(plan_states, Mapping)
        or not isinstance(plan_sha256s, Mapping)
        or not isinstance(rate_points, Mapping)
        or not isinstance(support_radii, Mapping)
        or set(plan_states) != expected_ids
        or set(plan_sha256s) != expected_ids
        or set(rate_points) != expected_ids
        or set(support_radii) != expected_ids
        or set(scores_by_id) != expected_ids
    ):
        raise ValueError("candidate plan table is incomplete")
    restored_plans: dict[str, StateConditionedReferenceProviderPlan] = {}
    restored_rates: dict[str, CandidateRatePoint] = {}
    restored_support: dict[str, float] = {}
    for candidate_id in sorted(expected_ids):
        plan_state = plan_states[candidate_id]
        rate_state = rate_points[candidate_id]
        if not isinstance(plan_state, Mapping):
            raise TypeError("candidate plan state must be a mapping")
        if not isinstance(rate_state, Mapping):
            raise TypeError("candidate rate state must be a mapping")
        plan = StateConditionedReferenceProviderPlan.from_state_dict(
            plan_state
        )
        rate = CandidateRatePoint.from_state_dict(rate_state)
        score = scores_by_id[candidate_id]
        support = float(support_radii[candidate_id])
        if (
            rate != expected_rates[candidate_id]
            or plan.artifact_sha256 != plan_sha256s[candidate_id]
            or plan.feature_codec.modal_modes != rate.source_rank
            or plan.target_modes != rate.target_rank
            or not torch.equal(
                plan.target_center,
                target_center[: rate.target_rank],
            )
            or not torch.equal(
                plan.target_scale,
                target_scale[: rate.target_rank],
            )
            or score.source_rank != rate.source_rank
            or score.target_rank != rate.target_rank
            or score.stored_scalar_count
            != plan.accounting().total_stored_scalar_count
            or not math.isfinite(support)
            or support <= 0.0
        ):
            raise ValueError(
                "candidate plan, rate, score, or selection panel drifted"
            )
        restored_plans[candidate_id] = plan
        restored_rates[candidate_id] = rate
        restored_support[candidate_id] = support
    selected_plan = restored_plans[selected_id]
    selected_rate = restored_rates[selected_id]
    support_radius = restored_support[selected_id]
    if (
        selected_plan.artifact_sha256 != selected_plan_sha256
        or selection.selected_source_rank != selected_rate.source_rank
        or selection.selected_target_rank != selected_rate.target_rank
        or selection.selected_stored_scalar_count
        != selected_plan.accounting().total_stored_scalar_count
    ):
        raise ValueError("selected provider plan binding mismatch")
    if (
        manifest.get("assessment_opened") is not False
        or manifest.get("prompt_text_loaded") is not False
        or manifest.get("token_ids_loaded") is not False
        or manifest.get("tokenizer_loaded") is not False
    ):
        raise ValueError("compiled candidate violates the prompt-blind firewall")
    return _LoadedReferenceCompilation(
        manifest=dict(manifest),
        artifact_sha256=artifact_sha256,
        file_sha256=file_sha256,
        report_sha256=computed_report_sha256,
        target_center=target_center,
        target_scale=target_scale,
        metric_weight=metric_weight,
        controls=controls,
        selection=selection,
        selected_id=selected_id,
        selected_rate_point=selected_rate,
        selected_plan=selected_plan,
        selected_support_radius=support_radius,
        protocol=protocol,
        training=training,
    )


def assess_gemma3_l3_l4_reference_provider(
    *,
    candidate_path: Path | str,
    candidate_file_sha256: str,
    candidate_report_sha256: str,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    basis_package_file_sha256: str = DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    basis_package_payload_sha256: str = (
        DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
    ),
    output: Path | str = DEFAULT_ASSESSMENT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Open the sealed synthetic panel once; never refit or reselect."""

    destination = _validate_output_path(output)
    # FIREWALL: candidate/report/code/protocol authenticate before a model
    # object capable of materializing assessment targets is loaded.
    compiled = _load_reference_compilation(
        candidate_path,
        expected_file_sha256=candidate_file_sha256,
        expected_report_sha256=candidate_report_sha256,
    )
    candidate_bytes_before = _read_regular_file(candidate_path)
    if (
        compiled.manifest.get("basis_package_file_sha256")
        != basis_package_file_sha256
        or compiled.manifest.get("basis_package_payload_sha256")
        != basis_package_payload_sha256
    ):
        raise ValueError("assessment basis arguments differ from candidate")
    assessment_claim = _claim_synthetic_assessment(
        protocol=compiled.protocol,
        basis_payload_sha256=str(
            compiled.manifest["basis_package_payload_sha256"]
        ),
        source_model_sha256=str(
            compiled.manifest["source_model_sha256"]
        ),
        ledger_dir=DEFAULT_ASSESSMENT_LEDGER_DIR,
    )
    (
        basis,
        adapter,
        pre_ff3,
        post_ff3,
        epsilon,
    ) = _load_live_dependencies(
        basis_package_path=basis_package_path,
        basis_package_file_sha256=basis_package_file_sha256,
        basis_package_payload_sha256=basis_package_payload_sha256,
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    if (
        compiled.manifest.get("basis_package_file_sha256")
        != basis_package_file_sha256
        or compiled.manifest.get("basis_package_payload_sha256")
        != basis.basis_payload_sha256
        or compiled.manifest.get("source_model_sha256")
        != basis.source_model_sha256
    ):
        raise ValueError("assessment basis differs from compiled candidate")
    metric_weight = _fisher_metric_weight(basis)
    if (
        not torch.equal(metric_weight, compiled.metric_weight)
        or _standardized_gauge_sha256(
            basis=basis,
            training=compiled.training,
            metric_weight=metric_weight,
        )
        != compiled.manifest.get("standardized_gauge_sha256")
    ):
        raise ValueError("assessment Fisher metric gauge drifted")
    model_before = adapter.model_fingerprint()
    norm_sha256 = module_state_fingerprint(pre_ff3)
    expected_binding = _synthetic_binding_sha256(
        basis=basis,
        protocol=compiled.protocol,
        training=compiled.training,
        norm_sha256=norm_sha256,
        rate_point=compiled.selected_rate_point,
    )
    if (
        compiled.manifest.get("pre_feedforward_norm_sha256")
        != norm_sha256
        or compiled.selected_plan.synthetic_binding_sha256
        != expected_binding
    ):
        raise ValueError("assessment live normalization binding drifted")
    measured, measurement = _measure_synthetic_role(
        role="assessment",
        protocol=compiled.protocol,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
    )
    assessment_batches = _provider_batch(
        measured,
        split="assessment",
        source_rank=compiled.selected_rate_point.source_rank,
        target_rank=compiled.selected_rate_point.target_rank,
        synthetic_binding_sha256=expected_binding,
    )
    assessment_scoring_probes = _full_width_probes(
        measured,
        metric_weight=metric_weight,
        standardized_gauge_sha256=str(
            compiled.manifest["standardized_gauge_sha256"]
        ),
    )
    assessment_panel, assessment_panel_sha256 = _assessment_panel_binding(
        protocol=compiled.protocol,
        measured=measured,
        assessment_scoring_probes=assessment_scoring_probes,
        standardized_gauge_sha256=str(
            compiled.manifest["standardized_gauge_sha256"]
        ),
    )
    fitted = _FittedReferenceCandidate(
        candidate_id=compiled.selected_id,
        rate_point=compiled.selected_rate_point,
        synthetic_binding_sha256=expected_binding,
        plan=compiled.selected_plan,
        support_radius=compiled.selected_support_radius,
    )
    assessment_candidate, runtime_metadata = _selection_candidate(
        fitted,
        measured=measured,
        full_probes=assessment_scoring_probes,
        metric_weight=metric_weight,
        standardized_gauge_sha256=str(
            compiled.manifest["standardized_gauge_sha256"]
        ),
        synthetic_batches=assessment_batches,
        structural_batches=(
            _padded_structural_batch(
                measured,
                split="assessment",
                source_rank=compiled.selected_rate_point.source_rank,
                target_rank=compiled.selected_rate_point.target_rank,
                synthetic_binding_sha256=expected_binding,
            ),
        ),
    )
    assessment_score = score_full_width_reference_assessment(
        controls=compiled.controls,
        assessment_probes=assessment_scoring_probes,
        candidate=assessment_candidate,
        gates=compiled.protocol.gates,
    )
    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_sha256
        or _read_regular_file(candidate_path) != candidate_bytes_before
        or _code_sha256s()
        != compiled.manifest.get("code_sha256s")
    ):
        raise RuntimeError(
            "model, candidate, normalization, or code changed during assessment"
        )
    common = {
        "schema": _ASSESSMENT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "candidate_artifact_sha256": compiled.artifact_sha256,
        "candidate_file_sha256": compiled.file_sha256,
        "candidate_report_sha256": compiled.report_sha256,
        "selected_candidate_id": compiled.selected_id,
        "selected_plan_sha256": compiled.selected_plan.artifact_sha256,
        "assessment_score_sha256": assessment_score.artifact_sha256,
        "assessment_panel": assessment_panel,
        "assessment_panel_sha256": assessment_panel_sha256,
        "assessment_passed": assessment_score.passed,
        "synthetic_protocol_sha256": compiled.protocol.protocol_sha256,
        "assessment_panel_spec_sha256": (
            compiled.protocol.assessment_panel_spec_sha256
        ),
        "training_protocol_sha256": compiled.training.artifact_sha256,
        "basis_package_payload_sha256": basis.basis_payload_sha256,
        "assessment_claim": assessment_claim,
        "code_sha256s": dict(
            compiled.manifest["code_sha256s"]  # type: ignore[arg-type]
        ),
        "assessment_opened_after_candidate_authentication": True,
        "candidate_refit": False,
        "candidate_reselection": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "scientific_claim": (
            "sealed_prompt_blind_synthetic_manifold_transfer"
            if assessment_score.passed
            else "sealed_prompt_blind_synthetic_manifold_falsification"
        ),
    }
    logical_artifact_sha256 = _json_sha256(
        common,
        domain=_CANDIDATE_DOMAIN,
    )
    state = {
        **common,
        "artifact_sha256": logical_artifact_sha256,
        "assessment_score_state": assessment_score.state_dict(),
    }
    report_payload = {
        **common,
        "artifact_sha256": logical_artifact_sha256,
        "assessment_measurement": measurement,
        "assessment_runtime_metadata": runtime_metadata,
        "assessment_score": assessment_score.state_dict(),
        "interpretation": {
            "candidate_was_frozen_before_assessment": True,
            "assessment_could_reject_but_not_change_candidate": True,
            "collision_targets_opened_only_in_assessment": True,
            "natural_prompt_transfer_tested": False,
            "prompt_independent_basis_discovery_proven": False,
            "whole_model_replacement_proven": False,
        },
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_provider_parameters": False,
            "committable": False,
        },
    }
    return _publish_artifact(state, report_payload, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_live_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--basis-package",
            type=Path,
            default=DEFAULT_BASIS_PACKAGE,
        )
        command.add_argument(
            "--basis-package-file-sha256",
            default=DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        )
        command.add_argument(
            "--basis-package-payload-sha256",
            default=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
        )
        command.add_argument("--model-id", default=DEFAULT_MODEL_ID)
        command.add_argument("--revision", default=DEFAULT_REVISION)
        command.add_argument("--cache-dir", type=Path)
        command.add_argument("--device", default="cpu")
        command.add_argument(
            "--dtype",
            choices=("auto", "float32", "float16", "bfloat16"),
            default="float32",
        )

    compile_parser = commands.add_parser(
        "compile",
        help="fit all ranks, then open prompt-blind synthetic selection",
    )
    add_live_arguments(compile_parser)
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    assess_parser = commands.add_parser(
        "assess",
        help="open the sealed synthetic assessment for one frozen candidate",
    )
    add_live_arguments(assess_parser)
    assess_parser.add_argument("--candidate", type=Path, required=True)
    assess_parser.add_argument(
        "--candidate-file-sha256",
        required=True,
    )
    assess_parser.add_argument(
        "--candidate-report-sha256",
        required=True,
    )
    assess_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ASSESSMENT_OUTPUT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compile":
        report = compile_gemma3_l3_l4_reference_provider(
            basis_package_path=args.basis_package,
            basis_package_file_sha256=args.basis_package_file_sha256,
            basis_package_payload_sha256=(
                args.basis_package_payload_sha256
            ),
            output=args.output,
            model_id=args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            device_name=args.device,
            dtype=args.dtype,
        )
    else:
        report = assess_gemma3_l3_l4_reference_provider(
            candidate_path=args.candidate,
            candidate_file_sha256=args.candidate_file_sha256,
            candidate_report_sha256=args.candidate_report_sha256,
            basis_package_path=args.basis_package,
            basis_package_file_sha256=args.basis_package_file_sha256,
            basis_package_payload_sha256=(
                args.basis_package_payload_sha256
            ),
            output=args.output,
            model_id=args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            device_name=args.device,
            dtype=args.dtype,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
