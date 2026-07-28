"""Run the fresh one-shot V3 assessment of the frozen Gemma L3/L4 provider.

V3 is assessment-only.  It authenticates the exact V2 ``spectral-r08-t08``
plan, its fit-only controls, the Fisher basis, the complete fresh panel, and
the V3 scoring code before any live V3 teacher target is materialized.  It
contains no fit, calibration, rank-selection, refit, fallback, or retry path.

The output is evidence rather than a checkpoint: only hashes, bindings,
scalar metrics, and tensor-free score states are published.  Raw teacher
targets, candidate predictions, model parameters, prompts, and token IDs are
never serialized.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import tempfile

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .external_models import find_git_worktree
from .gemma3_experiment import DEFAULT_MODEL_ID
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    Gemma3L3L4BasisPackage,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_frozen_provider_assessment_v3_lift import (
    lift_frozen_provider_assessment_v3_batch,
)
from .gemma3_l3_l4_frozen_provider_assessment_v3_materialization import (
    materialize_v3_panel,
)
from .gemma3_l3_l4_frozen_provider_assessment_v3_protocol import (
    DEFAULT_V3_PANEL_SPEC_SHA256,
    DEFAULT_V3_PROTOCOL_SHA256,
    V3AssessmentProtocol,
    V3ProbeSpec,
    default_v3_assessment_protocol,
)
from .gemma3_l3_l4_reference_provider_experiment import (
    DEFAULT_ASSESSMENT_LEDGER_DIR,
    DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    _FittedReferenceCandidate,
    _LoadedReferenceCompilation,
    _deferred_collision_gates,
    _fisher_metric_weight,
    _load_live_dependencies,
    _load_reference_compilation,
    _padded_structural_batch,
    _provider_batch,
    _read_regular_file,
    _selection_candidate,
    _standardized_gauge_sha256,
    _synthetic_binding_sha256,
)
from .gemma3_l3_l4_spectral_mapping_experiment import DEFAULT_REVISION
from .state_conditioned_contrast_assessment import (
    ContrastAssessmentGates,
    ContrastAssessmentResult,
    ContrastDefinition,
    ContrastObservation,
    assess_state_conditioned_contrasts,
)
from .state_conditioned_reference_selection import (
    FullWidthReferenceCandidate,
    FullWidthReferenceProbe,
    FullWidthReferenceControls,
    FullWidthCandidateScore,
    reconstruct_full_width_prediction,
    score_full_width_reference_assessment,
)


__all__ = [
    "DEFAULT_CANDIDATE",
    "DEFAULT_OUTPUT",
    "authenticate_frozen_v2_candidate",
    "assess_frozen_provider_v3",
    "build_parser",
    "describe_frozen_provider_v3",
    "main",
]


DEFAULT_CANDIDATE = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-reference-provider-dev-v2.pt"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-reference-provider-assessment-dev-v3.pt"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_frozen_provider_assessment_development.v3"
)
_FORMAT_VERSION = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TARGET_RANK = 64
_TEACHER_REPLAY_COUNT = 3
_CLAIM_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-frozen-provider-v3-claim:v1\0"
)
_PANEL_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-frozen-provider-v3-measured-panel:v1\0"
)
_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-frozen-provider-v3-artifact:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-frozen-provider-v3-report:v1\0"
)
_CODE_DOMAIN = b"fisher-graph:gemma3-l3-l4-frozen-provider-v3-code:v1\0"
_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-frozen-provider-v3-tensor:v1\0"
)
_V3_CODE_FILES = (
    "gemma3_l3_l4_frozen_provider_assessment_v3_protocol.py",
    "gemma3_l3_l4_frozen_provider_assessment_v3_materialization.py",
    "gemma3_l3_l4_frozen_provider_assessment_v3_lift.py",
    "state_conditioned_contrast_assessment.py",
    "gemma3_l3_l4_frozen_provider_assessment_v3.py",
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


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": tuple(int(size) for size in tensor.shape),
        }
    )
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + header
        + b"\0"
        + tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _file_sha256(path: Path | str) -> str:
    return hashlib.sha256(_read_regular_file(path)).hexdigest()


def _code_sha256s() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    values = {
        name: _file_sha256(directory / name)
        for name in _V3_CODE_FILES
    }
    if set(values) != set(_V3_CODE_FILES):
        raise RuntimeError("V3 assessment code manifest is incomplete")
    return values


def _code_bundle_sha256(values: Mapping[str, str]) -> str:
    if set(values) != set(_V3_CODE_FILES):
        raise ValueError("V3 assessment code manifest is incomplete")
    for name, digest in values.items():
        _require_sha256(digest, label=f"V3 code digest {name}")
    return _json_sha256(dict(values), domain=_CODE_DOMAIN)


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("V3 assessment output must use a .pt suffix")
    report = destination.with_suffix(".json")
    if destination.exists() or report.exists():
        raise FileExistsError("refusing to overwrite V3 assessment output")
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
                    "V3 assessment outputs in the worktree must remain "
                    "under an ignored local-runs directory"
                )
    return destination


@dataclass(frozen=True, slots=True)
class _OutputReservation:
    tensor_lock: Path
    report_lock: Path

    def release(self) -> None:
        self.report_lock.unlink(missing_ok=True)
        self.tensor_lock.unlink(missing_ok=True)


def _exclusive_marker(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reserve_output_pair(destination: Path) -> _OutputReservation:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensor_lock = destination.parent / f".{destination.name}.v3-reserved"
    report = destination.with_suffix(".json")
    report_lock = report.parent / f".{report.name}.v3-reserved"
    payload = b"fisher-graph V3 assessment output reservation\n"
    _exclusive_marker(tensor_lock, payload)
    try:
        _exclusive_marker(report_lock, payload)
    except BaseException:
        tensor_lock.unlink(missing_ok=True)
        raise
    return _OutputReservation(
        tensor_lock=tensor_lock,
        report_lock=report_lock,
    )


def _stage_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _publish_v3_artifact(
    state: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite V3 assessment output")
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


def authenticate_frozen_v2_candidate(
    candidate_path: Path | str = DEFAULT_CANDIDATE,
    *,
    protocol: V3AssessmentProtocol | None = None,
) -> _LoadedReferenceCompilation:
    """Authenticate the one admissible V2 plan before a V3 panel claim."""

    frozen_protocol = protocol or default_v3_assessment_protocol()
    binding = frozen_protocol.frozen_candidate
    compiled = _load_reference_compilation(
        candidate_path,
        expected_file_sha256=binding.file_sha256,
        expected_report_sha256=binding.report_sha256,
    )
    actual = {
        "artifact_sha256": compiled.artifact_sha256,
        "file_sha256": compiled.file_sha256,
        "report_sha256": compiled.report_sha256,
        "candidate_id": compiled.selected_id,
        "source_rank": compiled.selected_rate_point.source_rank,
        "target_rank": compiled.selected_rate_point.target_rank,
        "stored_scalar_count": (
            compiled.selected_plan.accounting().total_stored_scalar_count
        ),
        "selected_plan_sha256": compiled.selected_plan.artifact_sha256,
        "selection_sha256": compiled.selection.artifact_sha256,
        "controls_sha256": compiled.controls.artifact_sha256,
        "standardized_gauge_sha256": compiled.manifest.get(
            "standardized_gauge_sha256"
        ),
        "metric_weight_sha256": compiled.manifest.get(
            "metric_weight_sha256"
        ),
        "training_protocol_sha256": compiled.training.artifact_sha256,
        "basis_payload_sha256": compiled.manifest.get(
            "basis_package_payload_sha256"
        ),
        "source_model_sha256": compiled.manifest.get(
            "source_model_sha256"
        ),
    }
    expected = binding.state_dict()
    if actual != expected:
        differences = tuple(
            name for name in expected if actual.get(name) != expected[name]
        )
        raise ValueError(
            "candidate differs from the frozen V2 trust anchor: "
            + ", ".join(differences)
        )
    return compiled


def _authenticate_basis(
    path: Path | str,
    *,
    protocol: V3AssessmentProtocol,
) -> Gemma3L3L4BasisPackage:
    basis = load_gemma3_l3_l4_basis_package(
        path,
        expected_file_sha256=DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        expected_payload_sha256=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    binding = protocol.frozen_candidate
    if (
        basis.basis_payload_sha256 != binding.basis_payload_sha256
        or basis.source_model_sha256 != binding.source_model_sha256
    ):
        raise ValueError("basis differs from the frozen V2 candidate binding")
    return basis


def _claim_v3_panel_once(
    *,
    protocol: V3AssessmentProtocol,
    gates: ContrastAssessmentGates,
    code_sha256s: Mapping[str, str],
    ledger_dir: Path | str,
) -> dict[str, object]:
    """Irreversibly spend this target panel, independent of candidate/output."""

    binding = protocol.frozen_candidate
    uniqueness = {
        "schema": "fisher_graph.frozen_provider_v3_claim_identity.v1",
        "format_version": _FORMAT_VERSION,
        "panel_spec_sha256": protocol.panel_spec_sha256,
        "basis_payload_sha256": binding.basis_payload_sha256,
        "source_model_sha256": binding.source_model_sha256,
        "candidate_independent": True,
        "gates_independent": True,
        "output_independent": True,
        "subset_independent": True,
    }
    uniqueness_sha256 = _json_sha256(
        uniqueness,
        domain=_CLAIM_DOMAIN,
    )
    receipt = {
        **uniqueness,
        "uniqueness_sha256": uniqueness_sha256,
        "protocol_sha256": protocol.protocol_sha256,
        "probe_count": len(protocol.probes),
        "ordered_probe_sha256s": tuple(
            probe.artifact_sha256 for probe in protocol.probes
        ),
        "frozen_candidate": binding.state_dict(),
        "contrast_gates": gates.state_dict(),
        "contrast_gates_sha256": gates.artifact_sha256,
        "code_sha256s": dict(code_sha256s),
        "code_bundle_sha256": _code_bundle_sha256(code_sha256s),
        "claim_precedes_live_v3_teacher_target": True,
        "claim_survives_later_failure": True,
    }
    receipt["receipt_sha256"] = _json_sha256(
        receipt,
        domain=_CLAIM_DOMAIN,
    )
    directory = Path(ledger_dir)
    directory.mkdir(parents=True, exist_ok=True)
    claim_path = directory / f"full-v3-{uniqueness_sha256}.claim.json"
    encoded = (
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        _exclusive_marker(claim_path, encoded)
    except FileExistsError as error:
        raise FileExistsError(
            "the fresh V3 panel is already claimed for this basis and "
            "source model"
        ) from error
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return {
        "claim_file": str(claim_path),
        "claim_file_sha256": _file_sha256(claim_path),
        "uniqueness_sha256": uniqueness_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "panel_spec_sha256": protocol.panel_spec_sha256,
        "probe_count": len(protocol.probes),
        "claim_survives_later_failure": True,
    }


@dataclass(frozen=True, slots=True)
class _MeasuredV3Probe:
    probe: V3ProbeSpec
    requested_materialization_sha256: str
    modal_coordinates: Tensor
    null_coordinates: Tensor
    row_rms: Tensor
    target_modes: Tensor
    target_replays: tuple[Tensor, ...]
    logical_positions: Tensor
    valid_mask: Tensor
    lift_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.probe, V3ProbeSpec):
            raise TypeError("measured V3 probe must retain its frozen spec")
        _require_sha256(
            self.requested_materialization_sha256,
            label="V3 requested materialization",
        )
        length = self.probe.sequence_length
        expected = {
            "modal_coordinates": (1, length, _TARGET_RANK),
            "row_rms": (1, length),
            "target_modes": (1, length, _TARGET_RANK),
            "logical_positions": (1, length),
            "valid_mask": (1, length),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.shape != shape:
                raise ValueError(f"measured V3 {name} has invalid geometry")
        if (
            not isinstance(self.null_coordinates, Tensor)
            or self.null_coordinates.ndim != 3
            or self.null_coordinates.shape[:2] != (1, length)
            or self.null_coordinates.shape[-1] <= 0
        ):
            raise ValueError("measured V3 null coordinates have invalid geometry")
        if (
            type(self.target_replays) is not tuple
            or len(self.target_replays) != _TEACHER_REPLAY_COUNT
            or any(
                not isinstance(value, Tensor)
                or value.shape != (1, length, _TARGET_RANK)
                for value in self.target_replays
            )
            or not torch.equal(self.target_modes, self.target_replays[0])
        ):
            raise ValueError("measured V3 teacher replays are invalid")
        floating = (
            self.modal_coordinates,
            self.null_coordinates,
            self.row_rms,
            *self.target_replays,
        )
        if any(
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
            for value in floating
        ):
            raise ValueError("measured V3 tensors must be finite and floating")
        if self.logical_positions.dtype != torch.int64:
            raise TypeError("measured V3 logical positions must be int64")
        if self.valid_mask.dtype != torch.bool or not bool(
            self.valid_mask.all()
        ):
            raise ValueError("measured V3 panel must use fully valid sequences")
        if not isinstance(self.lift_metadata, Mapping):
            raise TypeError("measured V3 lift metadata must be a mapping")


def _distribution(values: Sequence[float]) -> dict[str, float]:
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    if tensor.numel() <= 0:
        raise ValueError("distribution requires at least one value")
    return {
        "minimum": float(tensor.min()),
        "median": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "maximum": float(tensor.max()),
    }


def _relative_difference(left: Tensor, right: Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(left - right))
    denominator = max(
        float(torch.linalg.vector_norm(right)),
        torch.finfo(torch.float64).tiny,
    )
    return numerator / denominator


def _measure_v3_panel(
    *,
    protocol: V3AssessmentProtocol,
    basis: Gemma3L3L4BasisPackage,
    adapter: Gemma3CausalLMAdapter,
    pre_ff3: nn.Module,
    post_ff3: nn.Module,
    epsilon: float,
) -> tuple[tuple[_MeasuredV3Probe, ...], dict[str, object]]:
    """Materialize the entire already-claimed panel and replay its teacher."""

    materialized = materialize_v3_panel(protocol)
    specifications = {probe.probe_id: probe for probe in protocol.probes}
    first_parameter = next(adapter.module.parameters(), None)
    if first_parameter is None or not first_parameter.is_floating_point():
        raise TypeError("live Gemma model has no floating parameters")
    device = first_parameter.device
    dtype = first_parameter.dtype
    y3_mean = basis.y3_mean.to(device=device, dtype=dtype)
    x4_mean = basis.x4_mean.to(device=device, dtype=dtype)
    r4 = basis.R4[:_TARGET_RANK].to(device=device, dtype=dtype)
    segment4 = adapter.segment("layer.4")
    measured: list[_MeasuredV3Probe] = []
    lift_rows: list[dict[str, object]] = []
    target_replay_errors: list[float] = []
    model_before = adapter.model_fingerprint()
    norm_before = module_state_fingerprint(pre_ff3)

    for requested in materialized:
        batch_probes = tuple(
            specifications[probe_id] for probe_id in requested.probe_ids
        )
        lift = lift_frozen_provider_assessment_v3_batch(
            basis,
            pre_ff3,
            epsilon=epsilon,
            batch=requested,
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
        y3 = y3_mean.view(1, 1, -1).expand(batch_size, length, -1)
        target_replays: list[Tensor] = []
        with torch.no_grad():
            hidden3_reference = hidden + post_ff3(y3)
            for _ in range(_TEACHER_REPLAY_COUNT):
                x4 = adapter.run_attention_prefix(
                    segment4,
                    hidden3_reference,
                    sequence,
                ).normalized_mlp_input
                target_replays.append(
                    ((x4 - x4_mean.view(1, 1, -1)) @ r4.T)
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .contiguous()
                )
        positions64 = positions.detach().to(device="cpu")
        valid64 = valid.detach().to(device="cpu")
        lift_metadata = lift.metadata()
        lift_rows.append(lift_metadata)
        for index, probe in enumerate(batch_probes):
            replays = tuple(
                value[index : index + 1] for value in target_replays
            )
            target_replay_errors.extend(
                _relative_difference(value, replays[0])
                for value in replays[1:]
            )
            measured.append(
                _MeasuredV3Probe(
                    probe=probe,
                    requested_materialization_sha256=(
                        requested.probe_tensor_sha256s[index]
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
                    target_modes=replays[0],
                    target_replays=replays,
                    logical_positions=positions64[index : index + 1],
                    valid_mask=valid64[index : index + 1],
                    lift_metadata={
                        "lift_artifact_sha256": lift.artifact_sha256,
                        "lift_diagnostics_sha256": (
                            lift.diagnostics_sha256
                        ),
                        "probe_id": probe.probe_id,
                    },
                )
            )

    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_before
    ):
        raise RuntimeError("live model changed during the V3 assessment")
    measured.sort(key=lambda value: value.probe.ordinal)
    if tuple(value.probe.ordinal for value in measured) != tuple(range(48)):
        raise RuntimeError("measured V3 probe order drifted")
    ordered_target_hashes = tuple(
        tuple(_tensor_sha256(replay) for replay in value.target_replays)
        for value in measured
    )
    return tuple(measured), {
        "schema": "fisher_graph.frozen_provider_v3_measurement.v1",
        "probe_count": len(measured),
        "teacher_replay_count": _TEACHER_REPLAY_COUNT,
        "family_counts": {
            family: sum(value.probe.family == family for value in measured)
            for family in sorted(
                {value.probe.family for value in measured}
            )
        },
        "materialized_batch_sha256s": tuple(
            value.artifact_sha256 for value in materialized
        ),
        "manifold_lift_sha256s": tuple(
            str(value["artifact_sha256"]) for value in lift_rows
        ),
        "ordered_target_replay_sha256s": ordered_target_hashes,
        "target_replay_relative_error": _distribution(
            target_replay_errors
        ),
        "l4_attention_prefix_execution_count": (
            len(materialized) * _TEACHER_REPLAY_COUNT
        ),
        "l3_mlp_body_execution_count": 0,
        "l4_mlp_body_execution_count": 0,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
    }


def _full_width_probes(
    measured: Sequence[_MeasuredV3Probe],
    *,
    metric_weight: Tensor,
    standardized_gauge_sha256: str,
) -> tuple[FullWidthReferenceProbe, ...]:
    return tuple(
        FullWidthReferenceProbe(
            probe_id=value.probe.probe_id,
            split="assessment",
            family=value.probe.family,
            standardized_target=(
                value.target_modes
                * metric_weight.view(1, 1, -1)
            ),
            logical_positions=value.logical_positions,
            valid_mask=value.valid_mask,
            standardized_gauge_sha256=standardized_gauge_sha256,
        )
        for value in measured
    )


def _ordinary_fidelity_panel(
    *,
    candidate: FullWidthReferenceCandidate,
    full_probes: Sequence[FullWidthReferenceProbe],
) -> tuple[
    tuple[FullWidthReferenceProbe, ...],
    FullWidthReferenceCandidate,
]:
    fidelity_families = {"multitone", "block_sparse"}
    fidelity_probes = tuple(
        probe for probe in full_probes if probe.family in fidelity_families
    )
    fidelity_probe_ids = {probe.probe_id for probe in fidelity_probes}
    if len(fidelity_probes) != 16 or len(fidelity_probe_ids) != 16:
        raise RuntimeError("V3 ordinary fidelity panel must contain 16 probes")
    fidelity_candidate = FullWidthReferenceCandidate(
        candidate_id=candidate.candidate_id,
        source_rank=candidate.source_rank,
        target_rank=candidate.target_rank,
        stored_scalar_count=candidate.stored_scalar_count,
        predictions=tuple(
            prediction
            for prediction in candidate.predictions
            if prediction.probe_id in fidelity_probe_ids
        ),
        structural_metrics=candidate.structural_metrics,
        candidate_binding_sha256=candidate.candidate_binding_sha256,
    )
    if len(fidelity_candidate.predictions) != 16:
        raise RuntimeError(
            "V3 ordinary fidelity predictions are incomplete"
        )
    return fidelity_probes, fidelity_candidate


def _score_frozen_provider(
    *,
    compiled: _LoadedReferenceCompilation,
    measured: Sequence[_MeasuredV3Probe],
    full_probes: Sequence[FullWidthReferenceProbe],
    metric_weight: Tensor,
    expected_binding: str,
) -> tuple[
    FullWidthReferenceCandidate,
    FullWidthCandidateScore,
    dict[str, object],
]:
    fitted = _FittedReferenceCandidate(
        candidate_id=compiled.selected_id,
        rate_point=compiled.selected_rate_point,
        synthetic_binding_sha256=expected_binding,
        plan=compiled.selected_plan,
        support_radius=compiled.selected_support_radius,
    )
    synthetic_batches = _provider_batch(
        measured,  # type: ignore[arg-type]
        split="assessment",
        source_rank=compiled.selected_rate_point.source_rank,
        target_rank=compiled.selected_rate_point.target_rank,
        synthetic_binding_sha256=expected_binding,
    )
    candidate, runtime_metadata = _selection_candidate(
        fitted,
        measured=measured,  # type: ignore[arg-type]
        full_probes=full_probes,
        metric_weight=metric_weight,
        standardized_gauge_sha256=(
            compiled.controls.standardized_gauge_sha256
        ),
        synthetic_batches=synthetic_batches,
        structural_batches=(
            _padded_structural_batch(
                measured,  # type: ignore[arg-type]
                split="assessment",
                source_rank=compiled.selected_rate_point.source_rank,
                target_rank=compiled.selected_rate_point.target_rank,
                synthetic_binding_sha256=expected_binding,
            ),
        ),
    )
    fidelity_probes, fidelity_candidate = _ordinary_fidelity_panel(
        candidate=candidate,
        full_probes=full_probes,
    )
    score = score_full_width_reference_assessment(
        controls=compiled.controls,
        assessment_probes=fidelity_probes,
        candidate=fidelity_candidate,
        gates=_deferred_collision_gates(compiled.protocol.gates),
    )
    runtime_metadata = {
        **runtime_metadata,
        "all_endpoint_prediction_count": len(candidate.predictions),
        "ordinary_fidelity_probe_count": len(fidelity_probes),
        "ordinary_fidelity_families": ("block_sparse", "multitone"),
        "contrast_control_endpoints_excluded_from_ordinary_fidelity": True,
        "ordinary_fidelity_candidate_sha256": (
            fidelity_candidate.artifact_sha256
        ),
    }
    return candidate, score, runtime_metadata


def _contrast_observations(
    *,
    protocol: V3AssessmentProtocol,
    measured: Sequence[_MeasuredV3Probe],
    full_probes: Sequence[FullWidthReferenceProbe],
    candidate: FullWidthReferenceCandidate,
    controls: FullWidthReferenceControls,
    metric_weight: Tensor,
) -> tuple[
    tuple[ContrastObservation, ...],
    dict[str, dict[str, str]],
]:
    measured_by_id = {value.probe.probe_id: value for value in measured}
    probes_by_id = {value.probe_id: value for value in full_probes}
    targets_by_id = {
        probe_id: value.standardized_target
        for probe_id, value in probes_by_id.items()
    }
    predictions_by_id = {
        prediction.probe_id: reconstruct_full_width_prediction(
            controls=controls,
            probe=probes_by_id[prediction.probe_id],
            prediction=prediction,
        )
        for prediction in candidate.predictions
    }
    if (
        set(measured_by_id)
        != set(targets_by_id)
        or set(targets_by_id) != set(predictions_by_id)
    ):
        raise RuntimeError("V3 target and candidate endpoint identities differ")

    observations: list[ContrastObservation] = []
    identities: dict[str, dict[str, str]] = {}
    for group in protocol.contrast_groups:
        for pair_index, (left_id, right_id) in enumerate(
            group.canonical_variant_pairs
        ):
            contrast_id = f"{group.group_id}.pair.{pair_index:02d}"
            role = (
                "expected_sensitivity"
                if group.intent == "sensitivity"
                else "intended_null"
            )
            definition = ContrastDefinition(
                contrast_id=contrast_id,
                family=group.family,
                role=role,
                coefficients=(-1.0, 1.0),
            )
            canonical_endpoints = (
                targets_by_id[left_id],
                targets_by_id[right_id],
            )
            canonical_delta = (
                canonical_endpoints[1] - canonical_endpoints[0]
            )
            replay_candidates: list[
                tuple[float, tuple[Tensor, Tensor]]
            ] = []
            for replay_index in range(1, _TEACHER_REPLAY_COUNT):
                replay = (
                    measured_by_id[left_id].target_replays[replay_index]
                    * metric_weight.view(1, 1, -1),
                    measured_by_id[right_id].target_replays[replay_index]
                    * metric_weight.view(1, 1, -1),
                )
                replay_delta = replay[1] - replay[0]
                replay_candidates.append(
                    (
                        float(
                            torch.linalg.vector_norm(
                                replay_delta - canonical_delta
                            )
                        ),
                        replay,
                    )
                )
            # The replay selected here is the preregistered maximum observed
            # contrast discrepancy, not whichever replay helps the candidate.
            repeated_teacher = max(
                replay_candidates,
                key=lambda value: value[0],
            )[1]
            candidate_endpoints = (
                predictions_by_id[left_id],
                predictions_by_id[right_id],
            )
            observations.append(
                ContrastObservation(
                    definition=definition,
                    teacher_endpoints=canonical_endpoints,
                    repeated_teacher_endpoints=repeated_teacher,
                    candidate_endpoints=candidate_endpoints,
                    repeated_candidate_endpoints=tuple(
                        value.clone() for value in candidate_endpoints
                    ),
                )
            )
            identities[contrast_id] = {
                "group_id": group.group_id,
                "family": group.family,
                "intent": group.intent,
                "rank_stratum": group.rank_stratum,
                "left_probe_id": left_id,
                "right_probe_id": right_id,
            }
    if len(observations) != 24 or len(identities) != 24:
        raise RuntimeError("V3 contrast expansion did not produce 24 pairs")
    return tuple(observations), identities


def _contrast_coverage(
    *,
    contrast_result: ContrastAssessmentResult,
    identities: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    scores = {
        score.contrast_id: score
        for score in contrast_result.contrast_scores
    }
    if set(scores) != set(identities):
        raise RuntimeError("V3 contrast score identities drifted")
    family_rows: dict[str, dict[str, object]] = {}
    for family, intent in (
        ("radial_block_sensitivity", "sensitivity"),
        ("signed_block_sensitivity", "sensitivity"),
        ("null_single_invariance", "invariance"),
    ):
        family_ids = tuple(
            contrast_id
            for contrast_id, identity in identities.items()
            if identity["family"] == family
        )
        if not family_ids:
            raise RuntimeError("V3 contrast family is empty")
        desired_status = (
            "eligible_sensitivity"
            if intent == "sensitivity"
            else "valid_intended_null"
        )
        passing = tuple(
            contrast_id
            for contrast_id in family_ids
            if scores[contrast_id].teacher_status == desired_status
        )
        strata = tuple(
            sorted(
                {
                    identities[contrast_id]["rank_stratum"]
                    for contrast_id in passing
                }
            )
        )
        family_rows[family] = {
            "intent": intent,
            "planned_contrast_count": len(family_ids),
            "teacher_qualified_contrast_count": len(passing),
            "qualified_rank_strata": strata,
            "retained_and_discarded_covered": (
                set(strata) == {"retained", "discarded"}
            ),
        }
    passed = all(
        bool(value["retained_and_discarded_covered"])
        for value in family_rows.values()
    )
    return {
        "family_coverage": family_rows,
        "all_families_cover_retained_and_discarded_strata": passed,
    }


def _measured_panel_binding(
    *,
    protocol: V3AssessmentProtocol,
    measurement: Mapping[str, object],
    full_probes: Sequence[FullWidthReferenceProbe],
    candidate: FullWidthReferenceCandidate,
    contrast_result: ContrastAssessmentResult,
    code_sha256s: Mapping[str, str],
) -> tuple[dict[str, object], str]:
    panel = {
        "schema": "fisher_graph.frozen_provider_v3_measured_panel.v1",
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "panel_spec_sha256": protocol.panel_spec_sha256,
        "probe_count": len(protocol.probes),
        "ordered_probe_ids": tuple(
            probe.probe_id for probe in protocol.probes
        ),
        "ordered_protocol_probe_sha256s": tuple(
            probe.artifact_sha256 for probe in protocol.probes
        ),
        "ordered_full_width_target_probe_sha256s": tuple(
            probe.artifact_sha256 for probe in full_probes
        ),
        "ordered_candidate_prediction_sha256s": tuple(
            prediction.artifact_sha256
            for prediction in candidate.predictions
        ),
        "materialized_batch_sha256s": tuple(
            measurement["materialized_batch_sha256s"]  # type: ignore[arg-type]
        ),
        "manifold_lift_sha256s": tuple(
            measurement["manifold_lift_sha256s"]  # type: ignore[arg-type]
        ),
        "ordered_target_replay_sha256s": tuple(
            tuple(value)
            for value in measurement[  # type: ignore[index]
                "ordered_target_replay_sha256s"
            ]
        ),
        "contrast_assessment_sha256": contrast_result.artifact_sha256,
        "code_sha256s": dict(code_sha256s),
        "code_bundle_sha256": _code_bundle_sha256(code_sha256s),
        "candidate_refit": False,
        "candidate_reselection": False,
        "assessment_target_used_for_threshold_tuning": False,
    }
    return panel, _json_sha256(panel, domain=_PANEL_DOMAIN)


def _assessment_outcome(
    *,
    fidelity_score: FullWidthCandidateScore,
    contrast_result: ContrastAssessmentResult,
    coverage: Mapping[str, object],
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if contrast_result.overall_status == "invalid":
        return "invalid_artifact_or_runtime", contrast_result.reason_codes
    if contrast_result.overall_status == "teacher_null_failure":
        return "teacher_invariance_falsified", contrast_result.reason_codes
    coverage_passed = bool(
        coverage["all_families_cover_retained_and_discarded_strata"]
    )
    if (
        contrast_result.overall_status == "panel_inconclusive"
        or not coverage_passed
    ):
        reasons.extend(contrast_result.reason_codes)
        if not coverage_passed:
            reasons.append(
                "contrast_families_do_not_cover_both_rank_strata"
            )
        if any("numerically" in value for value in reasons):
            return "panel_inconclusive_noise", tuple(sorted(set(reasons)))
        return (
            "panel_inconclusive_sensitivity",
            tuple(sorted(set(reasons))),
        )
    support = fidelity_score.structural_metrics.in_support_fraction
    if support < 0.99:
        return (
            "panel_out_of_support",
            ("candidate_feature_support_fraction_below_frozen_gate",),
        )
    if not fidelity_score.passed:
        failed = tuple(
            name
            for name, passed in fidelity_score.gate_flags.state_dict().items()
            if name != "all_passed" and passed is False
        )
        return "provider_failed_fidelity", failed
    if contrast_result.overall_status == "candidate_fail":
        return (
            "provider_failed_sensitive_contrast",
            contrast_result.reason_codes,
        )
    if contrast_result.overall_status != "pass":
        raise RuntimeError("unrecognized V3 contrast outcome")
    return "provider_passed", ()


def _assert_tensor_free(value: object, *, label: str) -> None:
    if isinstance(value, Tensor):
        raise ValueError(f"{label} must not contain tensors")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_tensor_free(item, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_tensor_free(item, label=f"{label}[{index}]")


def describe_frozen_provider_v3(
    *,
    candidate_path: Path | str = DEFAULT_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
) -> dict[str, object]:
    """Authenticate and describe V3 without opening or materializing targets."""

    protocol = default_v3_assessment_protocol()
    gates = ContrastAssessmentGates()
    compiled = authenticate_frozen_v2_candidate(
        candidate_path,
        protocol=protocol,
    )
    basis = _authenticate_basis(
        basis_package_path,
        protocol=protocol,
    )
    code_sha256s = _code_sha256s()
    return {
        "schema": f"{_SCHEMA}.description",
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "panel_spec_sha256": protocol.panel_spec_sha256,
        "probe_count": len(protocol.probes),
        "contrast_group_count": len(protocol.contrast_groups),
        "contrast_pair_count": sum(
            len(group.canonical_variant_pairs)
            for group in protocol.contrast_groups
        ),
        "frozen_candidate": protocol.frozen_candidate.state_dict(),
        "authenticated_candidate_artifact_sha256": (
            compiled.artifact_sha256
        ),
        "authenticated_basis_payload_sha256": (
            basis.basis_payload_sha256
        ),
        "contrast_gates": gates.state_dict(),
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": _code_bundle_sha256(code_sha256s),
        "live_model_loaded": False,
        "panel_materialized": False,
        "teacher_target_opened": False,
        "claim_created": False,
    }


def assess_frozen_provider_v3(
    *,
    candidate_path: Path | str = DEFAULT_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Spend the fresh V3 panel once and assess the exact frozen V2 plan."""

    protocol = default_v3_assessment_protocol()
    if (
        protocol.protocol_sha256 != DEFAULT_V3_PROTOCOL_SHA256
        or protocol.panel_spec_sha256 != DEFAULT_V3_PANEL_SPEC_SHA256
    ):
        raise ValueError("V3 protocol trust anchors drifted")
    gates = ContrastAssessmentGates()

    # Candidate and basis authentication happen before output reservation and
    # before the one-shot claim.  Neither operation can expose V3 targets.
    compiled = authenticate_frozen_v2_candidate(
        candidate_path,
        protocol=protocol,
    )
    basis_before = _authenticate_basis(
        basis_package_path,
        protocol=protocol,
    )
    candidate_bytes_before = _read_regular_file(candidate_path)
    basis_bytes_before = _read_regular_file(basis_package_path)
    destination = _validate_output_path(output)
    reservation = _reserve_output_pair(destination)
    code_sha256s = _code_sha256s()
    claim: dict[str, object] | None = None
    try:
        # This is the irreversible boundary.  No materializer, live Gemma
        # module, or target-producing path has executed above this line.
        claim = _claim_v3_panel_once(
            protocol=protocol,
            gates=gates,
            code_sha256s=code_sha256s,
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
            basis_package_file_sha256=(
                DEFAULT_BASIS_PACKAGE_FILE_SHA256
            ),
            basis_package_payload_sha256=(
                DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            model_id=DEFAULT_MODEL_ID,
            revision=DEFAULT_REVISION,
            cache_dir=cache_dir,
            device_name=device_name,
            dtype=dtype,
        )
        if (
            basis.basis_payload_sha256
            != basis_before.basis_payload_sha256
            or basis.source_model_sha256 != basis_before.source_model_sha256
        ):
            raise ValueError("live basis differs from pre-claim basis")
        metric_weight = _fisher_metric_weight(basis)
        standardized_gauge_sha256 = _standardized_gauge_sha256(
            basis=basis,
            training=compiled.training,
            metric_weight=metric_weight,
        )
        if (
            not torch.equal(metric_weight, compiled.metric_weight)
            or standardized_gauge_sha256
            != protocol.frozen_candidate.standardized_gauge_sha256
        ):
            raise ValueError("V3 Fisher metric gauge drifted")
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
            raise ValueError("V3 live normalization binding drifted")

        measured, measurement = _measure_v3_panel(
            protocol=protocol,
            basis=basis,
            adapter=adapter,
            pre_ff3=pre_ff3,
            post_ff3=post_ff3,
            epsilon=epsilon,
        )
        full_probes = _full_width_probes(
            measured,
            metric_weight=metric_weight,
            standardized_gauge_sha256=standardized_gauge_sha256,
        )
        candidate, fidelity_score, runtime_metadata = (
            _score_frozen_provider(
                compiled=compiled,
                measured=measured,
                full_probes=full_probes,
                metric_weight=metric_weight,
                expected_binding=expected_binding,
            )
        )
        contrast_observations, contrast_identities = (
            _contrast_observations(
                protocol=protocol,
                measured=measured,
                full_probes=full_probes,
                candidate=candidate,
                controls=compiled.controls,
                metric_weight=metric_weight,
            )
        )
        contrast_result = assess_state_conditioned_contrasts(
            contrast_observations,
            gates=gates,
        )
        coverage = _contrast_coverage(
            contrast_result=contrast_result,
            identities=contrast_identities,
        )
        outcome, outcome_reasons = _assessment_outcome(
            fidelity_score=fidelity_score,
            contrast_result=contrast_result,
            coverage=coverage,
        )
        panel, panel_sha256 = _measured_panel_binding(
            protocol=protocol,
            measurement=measurement,
            full_probes=full_probes,
            candidate=candidate,
            contrast_result=contrast_result,
            code_sha256s=code_sha256s,
        )
        if (
            _read_regular_file(candidate_path) != candidate_bytes_before
            or _read_regular_file(basis_package_path) != basis_bytes_before
            or _code_sha256s() != code_sha256s
            or adapter.model_fingerprint()
            != protocol.frozen_candidate.source_model_sha256
            or module_state_fingerprint(pre_ff3) != norm_sha256
        ):
            raise RuntimeError(
                "candidate, basis, code, model, or norm changed during V3"
            )

        common = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "outcome": outcome,
            "outcome_reasons": outcome_reasons,
            "provider_passed": outcome == "provider_passed",
            "protocol_sha256": protocol.protocol_sha256,
            "panel_spec_sha256": protocol.panel_spec_sha256,
            "measured_panel_sha256": panel_sha256,
            "frozen_candidate": protocol.frozen_candidate.state_dict(),
            "candidate_artifact_sha256": compiled.artifact_sha256,
            "candidate_file_sha256": compiled.file_sha256,
            "candidate_report_sha256": compiled.report_sha256,
            "selected_plan_sha256": compiled.selected_plan.artifact_sha256,
            "basis_package_file_sha256": (
                DEFAULT_BASIS_PACKAGE_FILE_SHA256
            ),
            "basis_package_payload_sha256": basis.basis_payload_sha256,
            "source_model_sha256": basis.source_model_sha256,
            "standardized_gauge_sha256": standardized_gauge_sha256,
            "contrast_gates_sha256": gates.artifact_sha256,
            "fidelity_gates_sha256": fidelity_score.gates_sha256,
            "fidelity_score_sha256": fidelity_score.artifact_sha256,
            "contrast_assessment_sha256": contrast_result.artifact_sha256,
            "claim": claim,
            "code_sha256s": code_sha256s,
            "code_bundle_sha256": _code_bundle_sha256(code_sha256s),
            "candidate_refit": False,
            "candidate_reselection": False,
            "candidate_parameters_changed": False,
            "assessment_opened_after_claim": True,
            "prompt_text_loaded": False,
            "token_ids_loaded": False,
            "tokenizer_loaded": False,
            "natural_activation_rows_loaded": False,
            "natural_prompt_transfer_tested": False,
            "whole_model_replacement_tested": False,
        }
        logical_artifact_sha256 = _json_sha256(
            common,
            domain=_ARTIFACT_DOMAIN,
        )
        state = {
            **common,
            "artifact_sha256": logical_artifact_sha256,
            "protocol_state": protocol.state_dict(),
            "measured_panel": panel,
            "fidelity_score_state": fidelity_score.state_dict(),
            "contrast_assessment_state": contrast_result.state_dict(),
            "contrast_coverage": coverage,
            "contrast_identities": contrast_identities,
        }
        report_payload = {
            **state,
            "assessment_measurement": measurement,
            "assessment_runtime_metadata": runtime_metadata,
            "interpretation": {
                "candidate_was_frozen_before_v3_design": True,
                "candidate_was_frozen_before_v3_targets": True,
                "weak_teacher_contrasts_scored_against_candidate": False,
                "intended_nulls_used_direction_metrics": False,
                "v3_was_designed_after_v2_results": True,
                "all_v3_target_identities_are_fresh": True,
                "prompt_blind_after_prompt_conditioned_basis": True,
                "natural_prompt_fidelity_claim": False,
                "whole_model_replacement_claim": False,
                "compression_claim": False,
                "latency_claim": False,
            },
            "safety": {
                "contains_source_model_state_dict": False,
                "contains_provider_parameters": False,
                "contains_raw_teacher_targets": False,
                "contains_raw_candidate_predictions": False,
                "contains_prompt_text": False,
                "contains_token_ids": False,
                "committable": False,
            },
        }
        _assert_tensor_free(state, label="V3 tensor artifact")
        _assert_tensor_free(report_payload, label="V3 JSON report")
        return _publish_v3_artifact(
            state,
            report_payload,
            output=destination,
        )
    finally:
        reservation.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_artifact_paths(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--candidate",
            type=Path,
            default=DEFAULT_CANDIDATE,
        )
        command.add_argument(
            "--basis-package",
            type=Path,
            default=DEFAULT_BASIS_PACKAGE,
        )

    describe = commands.add_parser(
        "describe",
        help="authenticate and describe V3 without opening its targets",
    )
    add_artifact_paths(describe)

    assess = commands.add_parser(
        "assess",
        help="irreversibly claim and run the complete fresh V3 panel",
    )
    add_artifact_paths(assess)
    assess.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    assess.add_argument("--cache-dir", type=Path)
    assess.add_argument("--device", default="cpu")
    assess.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        report = describe_frozen_provider_v3(
            candidate_path=args.candidate,
            basis_package_path=args.basis_package,
        )
    else:
        report = assess_frozen_provider_v3(
            candidate_path=args.candidate,
            basis_package_path=args.basis_package,
            output=args.output,
            cache_dir=args.cache_dir,
            device_name=args.device,
            dtype=args.dtype,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
