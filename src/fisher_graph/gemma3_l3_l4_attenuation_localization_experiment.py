"""Localize weak Gemma L3/L4 collision contrasts without changing v2.

This is a retrospective diagnostic over the already-opened reference-provider
v2 assessment.  It authenticates the frozen candidate, failed assessment,
one-shot claim, basis, source model, and every regenerated collision target
before computing any localization metric.

The diagnostic reexecutes the 40 collision endpoints and analyzes all 32
within-group pairs.  It records finite checkpoint secants, midpoint JVPs,
contrast-aligned VJPs, residual/attention cancellation, and retained Fisher
subspace capture.  The 16 canonical minimum-separation pairs are marked as
the exact witnesses of the original collision gate.

No target-conditioned value from this artifact may be used to refit, reselect,
rank, or modify the frozen provider.  Any compiler change informed by this
open-panel postmortem requires a genuinely fresh sealed v3 assessment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

import torch
from torch import Tensor, nn

from .activations import ActivationTrace
from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .checkpoint_contrast import CheckpointContrastThresholds
from .external_models import find_git_worktree
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    resolve_torch_device,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    Gemma3L3L4BasisPackage,
)
from .gemma3_l3_l4_manifold_lift import (
    Gemma3L3L4ManifoldLift,
    lift_synthetic_reference_batch_to_gemma3_manifold,
)
from .gemma3_l3_l4_reference_provider_experiment import (
    DEFAULT_ASSESSMENT_OUTPUT,
    DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    DEFAULT_OUTPUT as DEFAULT_CANDIDATE,
    _CANDIDATE_DOMAIN as _V2_CANDIDATE_DOMAIN,
    _CODE_FILES as _V2_CODE_FILES,
    _MeasuredSyntheticProbe,
    _REPORT_DOMAIN as _V2_REPORT_DOMAIN,
    _code_sha256s as _v2_code_sha256s,
    _full_width_probes,
    _json_sha256 as _v2_json_sha256,
    _load_live_dependencies,
    _load_reference_compilation,
    _read_regular_file as _read_v2_regular_file,
    _role_probes,
)
from .gemma3_l3_l4_spectral_mapping_experiment import DEFAULT_REVISION
from .gemma3_l3_l4_synthetic_materialization import (
    MaterializedSyntheticReferenceBatch,
    materialize_synthetic_reference_batches,
)
from .gemma3_l3_l4_synthetic_reference_protocol import (
    SyntheticReferenceProbe,
)
from .state_conditioned_reference_selection import (
    FullWidthCandidateScore,
    FullWidthReferenceProbe,
)


__all__ = [
    "DEFAULT_ASSESSMENT",
    "DEFAULT_ASSESSMENT_FILE_SHA256",
    "DEFAULT_ASSESSMENT_REPORT_SHA256",
    "DEFAULT_CANDIDATE",
    "DEFAULT_CANDIDATE_FILE_SHA256",
    "DEFAULT_CANDIDATE_REPORT_SHA256",
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_attenuation_localization_experiment",
]


DEFAULT_CANDIDATE_FILE_SHA256 = (
    "37bd6fbda9b3660777f0388561e4e8d7d1a28e3958bcb98c69ca302cd1f77ae1"
)
DEFAULT_CANDIDATE_REPORT_SHA256 = (
    "1e14518f915821aa7448b6f4799e322e2451074b3030ba4107c6a2a0924be4d9"
)
DEFAULT_ASSESSMENT = DEFAULT_ASSESSMENT_OUTPUT
DEFAULT_ASSESSMENT_FILE_SHA256 = (
    "a4175def42020f1b13a370e7ee9308dcc2be3b3439960987418573ba4379b2dd"
)
DEFAULT_ASSESSMENT_REPORT_SHA256 = (
    "613856ec39a7d0cac21cc6e41a155a4609c73ea05e4daa01ccf1affe26153b6e"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-attenuation-localization-dev-v6.pt"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_attenuation_localization_development.v1"
)
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher-graph:gemma3-l3-l4-attenuation:v1\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-attenuation-report:v1\0"
_PANEL_SHA256 = (
    "417540555cea3ce00071b495386287c6656921c964ef47ac26715c282f07ab0a"
)
_PANEL_SPEC_SHA256 = (
    "c690e9f85f5629ab2701fc5db487aea1404864256f5fe24034e35143047af102"
)
_ASSESSMENT_ARTIFACT_SHA256 = (
    "21500080aed580e91b605a6fdd01984dcc41676c0dea96a7813ee0ec4a8cc57d"
)
_CLAIM_FILE_SHA256 = (
    "4361b88bfd3802c0688c4063c598ddc8708ad2754247e47aebaf9bc23c82bf27"
)
_SOURCE_MODEL_SHA256 = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_COLLISION_THRESHOLD = 0.01
_TARGET_RANK = 64
_EXPECTED_COLLISION_ENDPOINTS = 40
_EXPECTED_COLLISION_GROUPS = 16
_EXPECTED_COLLISION_PAIRS = 32
_CONTRAST_THRESHOLDS = CheckpointContrastThresholds()
_NUMERIC_FLOOR_EPSILON_MULTIPLIER = (
    _CONTRAST_THRESHOLDS.numeric_floor_epsilon_multiplier
)
_RESOLVED_NOISE_MULTIPLIER = (
    _CONTRAST_THRESHOLDS.resolved_noise_multiplier
)
_CURVATURE_RELATIVE_ERROR = (
    _CONTRAST_THRESHOLDS.maximum_linearization_relative_error
)
_ATTENUATION_RATIO = _CONTRAST_THRESHOLDS.localization_contraction_ratio
_CANCELLATION_COSINE = -0.75
_CANCELLATION_MERGE_RATIO = 0.5
_RETAINED_FISHER_FRACTION = 0.25

_L4_SITES = (
    "layer.4.input",
    "layer.4.attention.normalized_input",
    "layer.4.attention.operator_output",
    "layer.4.attention.delta",
    "layer.4.post_attention",
    "layer.4.mlp.normalized_input",
)
_CHECKPOINT_NAMES = (
    "layer.3.post_attention_hidden",
    "layer.3.mlp.normalized_input",
    *_L4_SITES,
    "l4.target_modes",
    "l4.fisher_weighted_target",
)
_CAUSAL_PATH = (
    "layer.3.post_attention_hidden",
    *_L4_SITES,
    "l4.target_modes",
    "l4.fisher_weighted_target",
)
_SIDE_BRANCHES = ("layer.3.mlp.normalized_input",)
_HIDDEN_INDEX = 0
_ACTUAL_X3_INDEX = 1
_L4_INPUT_INDEX = 2
_L4_NORMALIZED_INDEX = 3
_ATTENTION_OUTPUT_INDEX = 4
_ATTENTION_DELTA_INDEX = 5
_POST_ATTENTION_INDEX = 6
_X4_INDEX = 7
_RAW_TARGET_INDEX = 8
_WEIGHTED_TARGET_INDEX = 9
_LOCAL_CODE_FILES = (
    "checkpoint_contrast.py",
    "gemma3_l3_l4_attenuation_localization_experiment.py",
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


def _local_code_sha256s() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: _file_sha256(root / name) for name in _LOCAL_CODE_FILES}


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


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("attenuation-localization output must use .pt")
    report = destination.with_suffix(".json")
    if destination.exists() or report.exists():
        raise FileExistsError(
            "refusing to overwrite attenuation-localization output"
        )
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
                    "attenuation artifacts in the worktree must remain under "
                    "an ignored local-runs directory"
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
        raise FileExistsError(
            "refusing to overwrite attenuation-localization output"
        )
    state_stage = _stage_path(output)
    report_stage = _stage_path(report_path)
    published: list[Path] = []
    try:
        torch.save(dict(state), state_stage)
        report = {
            **dict(report_payload),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": _file_sha256(state_stage),
                "tensor_file_bytes": state_stage.stat().st_size,
                "report_file": str(report_path),
                "contains_raw_activations": False,
                "contains_model_or_provider_parameters": False,
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
        os.link(state_stage, output)
        published.append(output)
        os.link(report_stage, report_path)
        published.append(report_path)
        return report
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        state_stage.unlink(missing_ok=True)
        report_stage.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _OpenedAssessment:
    state: Mapping[str, object]
    report: Mapping[str, object]
    score: FullWidthCandidateScore
    panel: Mapping[str, object]
    file_sha256: str
    report_sha256: str
    artifact_sha256: str
    claim_file_sha256: str
    claim_path: Path
    claim_bytes: bytes


def _load_opened_reference_assessment(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
    compiled: object,
) -> _OpenedAssessment:
    assessment_path = Path(path)
    payload = _read_regular_file(assessment_path)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise ValueError("assessment tensor file SHA-256 mismatch")
    report_bytes = _read_regular_file(assessment_path.with_suffix(".json"))
    report = json.loads(report_bytes.decode("utf-8"))
    if not isinstance(report, Mapping):
        raise TypeError("assessment report must be a mapping")
    supplied_report_sha256 = report.get("report_sha256")
    report_without_hash = dict(report)
    report_without_hash.pop("report_sha256", None)
    computed_report_sha256 = _v2_json_sha256(
        report_without_hash,
        domain=_V2_REPORT_DOMAIN,
    )
    if (
        supplied_report_sha256 != computed_report_sha256
        or supplied_report_sha256 != expected_report_sha256
    ):
        raise ValueError("assessment logical report SHA-256 mismatch")
    artifact_record = report.get("artifact")
    if (
        not isinstance(artifact_record, Mapping)
        or artifact_record.get("tensor_file_sha256") != file_sha256
    ):
        raise ValueError("assessment report does not bind its tensor file")

    raw = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    expected_fields = {
        "schema",
        "format_version",
        "candidate_artifact_sha256",
        "candidate_file_sha256",
        "candidate_report_sha256",
        "selected_candidate_id",
        "selected_plan_sha256",
        "assessment_score_sha256",
        "assessment_panel",
        "assessment_panel_sha256",
        "assessment_passed",
        "synthetic_protocol_sha256",
        "assessment_panel_spec_sha256",
        "training_protocol_sha256",
        "basis_package_payload_sha256",
        "assessment_claim",
        "code_sha256s",
        "assessment_opened_after_candidate_authentication",
        "candidate_refit",
        "candidate_reselection",
        "prompt_text_loaded",
        "token_ids_loaded",
        "tokenizer_loaded",
        "natural_activation_rows_loaded",
        "scientific_claim",
        "artifact_sha256",
        "assessment_score_state",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise ValueError("assessment tensor fields do not match frozen v2")
    common = dict(raw)
    supplied_artifact_sha256 = common.pop("artifact_sha256")
    score_state = common.pop("assessment_score_state")
    computed_artifact_sha256 = _v2_json_sha256(
        common,
        domain=_V2_CANDIDATE_DOMAIN,
    )
    if (
        supplied_artifact_sha256 != computed_artifact_sha256
        or supplied_artifact_sha256 != _ASSESSMENT_ARTIFACT_SHA256
        or report.get("artifact_sha256") != computed_artifact_sha256
    ):
        raise ValueError("assessment logical artifact binding mismatch")

    required_compiled = {
        "artifact_sha256": getattr(compiled, "artifact_sha256"),
        "file_sha256": getattr(compiled, "file_sha256"),
        "report_sha256": getattr(compiled, "report_sha256"),
        "selected_id": getattr(compiled, "selected_id"),
        "selected_plan_sha256": getattr(
            getattr(compiled, "selected_plan"),
            "artifact_sha256",
        ),
        "protocol_sha256": getattr(
            getattr(compiled, "protocol"),
            "protocol_sha256",
        ),
        "training_sha256": getattr(
            getattr(compiled, "training"),
            "artifact_sha256",
        ),
    }
    if (
        raw["candidate_artifact_sha256"]
        != required_compiled["artifact_sha256"]
        or raw["candidate_file_sha256"] != required_compiled["file_sha256"]
        or raw["candidate_report_sha256"]
        != required_compiled["report_sha256"]
        or raw["selected_candidate_id"] != required_compiled["selected_id"]
        or raw["selected_plan_sha256"]
        != required_compiled["selected_plan_sha256"]
        or raw["synthetic_protocol_sha256"]
        != required_compiled["protocol_sha256"]
        or raw["training_protocol_sha256"]
        != required_compiled["training_sha256"]
    ):
        raise ValueError("assessment differs from the frozen candidate")
    if (
        raw["assessment_panel_sha256"] != _PANEL_SHA256
        or raw["assessment_panel_spec_sha256"] != _PANEL_SPEC_SHA256
        or raw["basis_package_payload_sha256"]
        != DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        or raw["assessment_opened_after_candidate_authentication"] is not True
        or raw["candidate_refit"] is not False
        or raw["candidate_reselection"] is not False
        or raw["assessment_passed"] is not False
    ):
        raise ValueError("assessment frozen decision or panel binding drifted")
    if any(
        raw[field] is not False
        for field in (
            "prompt_text_loaded",
            "token_ids_loaded",
            "tokenizer_loaded",
            "natural_activation_rows_loaded",
        )
    ):
        raise ValueError("assessment prompt-blind firewall was violated")
    if (
        not isinstance(score_state, Mapping)
        or not isinstance(raw["assessment_panel"], Mapping)
    ):
        raise TypeError("assessment score or panel is not a mapping")
    score = FullWidthCandidateScore.from_state_dict(score_state)
    if (
        score.artifact_sha256 != raw["assessment_score_sha256"]
        or score.passed
        or score.gate_flags.collision_target_relative_difference
    ):
        raise ValueError("assessment score no longer witnesses collision failure")
    live_v2_code = _v2_code_sha256s()
    if (
        not isinstance(raw["code_sha256s"], Mapping)
        or dict(raw["code_sha256s"]) != live_v2_code
        or dict(raw["code_sha256s"])
        != dict(getattr(compiled, "manifest")["code_sha256s"])
        or set(live_v2_code) != set(_V2_CODE_FILES)
    ):
        raise ValueError("assessment v2 code bundle differs from live code")

    claim = raw["assessment_claim"]
    if not isinstance(claim, Mapping):
        raise TypeError("assessment claim must be a mapping")
    claim_path = claim.get("claim_file")
    claim_file_sha256 = claim.get("claim_file_sha256")
    claim_bytes = (
        _read_regular_file(claim_path)
        if isinstance(claim_path, str)
        else b""
    )
    if (
        not isinstance(claim_path, str)
        or claim_file_sha256 != _CLAIM_FILE_SHA256
        or hashlib.sha256(claim_bytes).hexdigest() != claim_file_sha256
        or claim.get("assessment_panel_spec_sha256") != _PANEL_SPEC_SHA256
        or claim.get("probe_count") != 88
    ):
        raise ValueError("assessment one-shot claim binding is invalid")
    panel = dict(raw["assessment_panel"])
    if (
        panel.get("probe_count") != 88
        or panel.get("collision_probe_count") != _EXPECTED_COLLISION_ENDPOINTS
        or panel.get("assessment_panel_spec_sha256") != _PANEL_SPEC_SHA256
        or panel.get("synthetic_protocol_sha256")
        != required_compiled["protocol_sha256"]
    ):
        raise ValueError("assessment measured panel binding drifted")
    return _OpenedAssessment(
        state=dict(raw),
        report=dict(report),
        score=score,
        panel=panel,
        file_sha256=file_sha256,
        report_sha256=computed_report_sha256,
        artifact_sha256=computed_artifact_sha256,
        claim_file_sha256=str(claim_file_sha256),
        claim_path=Path(claim_path),
        claim_bytes=claim_bytes,
    )


@dataclass(frozen=True, slots=True)
class _Endpoint:
    probe: SyntheticReferenceProbe
    requested_modes: Tensor
    checkpoints: tuple[Tensor, ...]
    repeated_checkpoints: tuple[Tensor, ...]
    row_rms: Tensor
    normalized_null_features: Tensor
    logical_positions: Tensor
    valid_mask: Tensor
    target_probe_sha256: str
    lift_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class _Pair:
    pair_id: str
    family: str
    collision_group: str
    left_variant: str
    right_variant: str
    left_probe_id: str
    right_probe_id: str
    sequence_length: int
    source_offset: int
    target_relative_difference: float
    gate_witness: bool


def _prepare_sequence(
    adapter: Gemma3CausalLMAdapter,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    residual_width: int,
) -> object:
    positions = torch.arange(
        sequence_length,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)
    valid = torch.ones(
        batch_size,
        sequence_length,
        device=device,
        dtype=torch.bool,
    )
    placeholder = torch.zeros(
        batch_size,
        sequence_length,
        residual_width,
        device=device,
        dtype=dtype,
    )
    return adapter.prepare_sequence(
        {
            "inputs_embeds": placeholder,
            "attention_mask": valid,
            "position_ids": positions,
        }
    )


def _checkpoint_function(
    *,
    adapter: Gemma3CausalLMAdapter,
    pre_ff3: nn.Module,
    sequence: object,
    layer3_mean_delta: Tensor,
    x4_mean: Tensor,
    r4: Tensor,
    metric_weight: Tensor,
):
    segment4 = adapter.segment("layer.4")

    def run(hidden: Tensor) -> tuple[Tensor, ...]:
        actual_x3 = pre_ff3(hidden)
        hidden3_reference = hidden + layer3_mean_delta.view(1, 1, -1)
        trace = ActivationTrace(
            retain_grad=False,
            store=True,
            capture_sites=_L4_SITES,
        )
        prefix = adapter.run_attention_prefix(
            segment4,
            hidden3_reference,
            sequence,  # type: ignore[arg-type]
            trace=trace,
        )
        trace.assert_all_captures_seen()
        raw_target = (
            prefix.normalized_mlp_input - x4_mean.view(1, 1, -1)
        ) @ r4.T
        weighted_target = raw_target * metric_weight.view(1, 1, -1)
        return (
            hidden,
            actual_x3,
            *(trace[name] for name in _L4_SITES),
            raw_target,
            weighted_target,
        )

    return run


def _collision_probes(
    protocol: object,
) -> tuple[SyntheticReferenceProbe, ...]:
    probes = tuple(
        probe
        for probe in _role_probes(protocol, "assessment")  # type: ignore[arg-type]
        if probe.collision_group is not None
    )
    if (
        len(probes) != _EXPECTED_COLLISION_ENDPOINTS
        or Counter(probe.family for probe in probes)
        != Counter({"axis": 16, "radial_collision": 12, "null_collision": 12})
        or len({probe.collision_group for probe in probes})
        != _EXPECTED_COLLISION_GROUPS
    ):
        raise ValueError("frozen collision endpoint geometry drifted")
    return probes


def _target_hashes_by_probe(
    panel: Mapping[str, object],
) -> dict[str, str]:
    probe_ids = panel.get("ordered_probe_ids")
    hashes = panel.get("ordered_full_width_target_probe_sha256s")
    if (
        not isinstance(probe_ids, (tuple, list))
        or not isinstance(hashes, (tuple, list))
        or len(probe_ids) != 88
        or len(hashes) != 88
        or len(set(probe_ids)) != 88
    ):
        raise ValueError("assessment target hash table is invalid")
    return {
        str(probe_id): str(digest)
        for probe_id, digest in zip(probe_ids, hashes, strict=True)
    }


def _measure_collision_endpoints(
    *,
    compiled: object,
    opened: _OpenedAssessment,
    basis: Gemma3L3L4BasisPackage,
    adapter: Gemma3CausalLMAdapter,
    pre_ff3: nn.Module,
    post_ff3: nn.Module,
    epsilon: float,
) -> tuple[
    dict[str, _Endpoint],
    dict[int, object],
    dict[str, object],
]:
    probes = _collision_probes(getattr(compiled, "protocol"))
    probes_by_hash = {probe.artifact_sha256: probe for probe in probes}
    expected_hashes = _target_hashes_by_probe(opened.panel)
    standardized_gauge_sha256 = str(
        getattr(compiled, "manifest")["standardized_gauge_sha256"]
    )
    metric_weight64 = getattr(compiled, "metric_weight").detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    first_parameter = next(adapter.module.parameters(), None)
    if first_parameter is None or not first_parameter.is_floating_point():
        raise TypeError("live Gemma model has no floating parameter")
    device = first_parameter.device
    dtype = first_parameter.dtype
    y3_mean = basis.y3_mean.to(device=device, dtype=dtype)
    x4_mean = basis.x4_mean.to(device=device, dtype=dtype)
    r4 = basis.R4[:_TARGET_RANK].to(device=device, dtype=dtype)
    metric_weight = metric_weight64.to(device=device, dtype=dtype)
    with torch.no_grad():
        layer3_mean_delta = post_ff3(y3_mean)
    if (
        layer3_mean_delta.shape != y3_mean.shape
        or not bool(torch.isfinite(layer3_mean_delta).all())
    ):
        raise ValueError("frozen layer-3 mean delta is invalid")

    endpoints: dict[str, _Endpoint] = {}
    sequences_by_length: dict[int, object] = {}
    batch_artifact_sha256s: list[str] = []
    lift_artifact_sha256s: list[str] = []
    target_hashes: list[str] = []
    for materialized in materialize_synthetic_reference_batches(
        getattr(compiled, "protocol"),
        probes,
    ):
        batch_probes = tuple(
            probes_by_hash[digest]
            for digest in materialized.probe_artifact_sha256s
        )
        lift = lift_synthetic_reference_batch_to_gemma3_manifold(
            basis,
            pre_ff3,
            epsilon=epsilon,
            batch=materialized,
            probes=batch_probes,
        )
        lift.validate_integrity()
        sequence = _prepare_sequence(
            adapter,
            batch_size=lift.batch_size,
            sequence_length=lift.sequence_length,
            device=device,
            dtype=dtype,
            residual_width=basis.residual_width,
        )
        sequences_by_length[lift.sequence_length] = sequence
        run = _checkpoint_function(
            adapter=adapter,
            pre_ff3=pre_ff3,
            sequence=sequence,
            layer3_mean_delta=layer3_mean_delta,
            x4_mean=x4_mean,
            r4=r4,
            metric_weight=metric_weight,
        )
        hidden = lift.hidden_states.to(device=device, dtype=dtype)
        with torch.no_grad():
            outputs = run(hidden)
            repeated = run(hidden)
        if len(outputs) != len(_CHECKPOINT_NAMES) or len(repeated) != len(
            _CHECKPOINT_NAMES
        ):
            raise RuntimeError("checkpoint path returned an invalid tuple")
        outputs64 = tuple(
            value.detach().to(device="cpu", dtype=torch.float64).contiguous()
            for value in outputs
        )
        repeated64 = tuple(
            value.detach().to(device="cpu", dtype=torch.float64).contiguous()
            for value in repeated
        )
        if not torch.equal(outputs64[_ACTUAL_X3_INDEX], lift.actual_x3):
            maximum = float(
                (
                    outputs64[_ACTUAL_X3_INDEX] - lift.actual_x3
                ).abs().max()
            )
            raise RuntimeError(
                "checkpoint replay differs from manifold x3 "
                f"(maximum {maximum})"
            )
        positions = torch.arange(
            lift.sequence_length,
            dtype=torch.int64,
        ).unsqueeze(0).expand(lift.batch_size, -1)
        valid = torch.ones(
            lift.batch_size,
            lift.sequence_length,
            dtype=torch.bool,
        )
        measured: list[_MeasuredSyntheticProbe] = []
        for index, probe in enumerate(batch_probes):
            measured.append(
                _MeasuredSyntheticProbe(
                    probe=probe,
                    requested_materialization_sha256=(
                        materialized.probe_tensor_sha256s[index]
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
                    target_modes=outputs64[_RAW_TARGET_INDEX][
                        index : index + 1
                    ],
                    logical_positions=positions[index : index + 1],
                    valid_mask=valid[index : index + 1],
                    lift_metadata={
                        "lift_artifact_sha256": lift.artifact_sha256,
                        "lift_diagnostics_sha256": lift.diagnostics_sha256,
                        "formula_version": lift.formula_version,
                        "probe_id": probe.probe_id,
                    },
                )
            )
        scoring = _full_width_probes(
            measured,
            metric_weight=metric_weight64,
            standardized_gauge_sha256=standardized_gauge_sha256,
        )
        if len(scoring) != len(batch_probes):
            raise RuntimeError("collision target scoring rows are incomplete")
        for index, (probe, full_probe) in enumerate(
            zip(batch_probes, scoring, strict=True)
        ):
            expected_hash = expected_hashes.get(probe.probe_id)
            if full_probe.artifact_sha256 != expected_hash:
                raise ValueError(
                    "regenerated collision target differs from the opened "
                    f"assessment: {probe.probe_id}"
                )
            endpoint_outputs = list(
                value[index : index + 1] for value in outputs64
            )
            # Preserve the exact float64 Fisher gauge used by the v2 gate,
            # rather than the differentiable float32 multiplication used by
            # the JVP/VJP path.
            endpoint_outputs[_WEIGHTED_TARGET_INDEX] = (
                full_probe.standardized_target
            )
            endpoint_repeated = list(
                value[index : index + 1] for value in repeated64
            )
            endpoint_repeated[_WEIGHTED_TARGET_INDEX] = (
                endpoint_repeated[_RAW_TARGET_INDEX]
                * metric_weight64.view(1, 1, -1)
            )
            endpoints[probe.probe_id] = _Endpoint(
                probe=probe,
                requested_modes=lift.requested_standardized_modes[
                    index : index + 1
                ],
                checkpoints=tuple(endpoint_outputs),
                repeated_checkpoints=tuple(endpoint_repeated),
                row_rms=lift.row_rms[index : index + 1],
                normalized_null_features=lift.normalized_null_features[
                    index : index + 1
                ],
                logical_positions=positions[index : index + 1],
                valid_mask=valid[index : index + 1],
                target_probe_sha256=full_probe.artifact_sha256,
                lift_artifact_sha256=lift.artifact_sha256,
            )
            target_hashes.append(full_probe.artifact_sha256)
        batch_artifact_sha256s.append(materialized.artifact_sha256)
        lift_artifact_sha256s.append(lift.artifact_sha256)
    if (
        len(endpoints) != _EXPECTED_COLLISION_ENDPOINTS
        or len(set(target_hashes)) != _EXPECTED_COLLISION_ENDPOINTS
    ):
        raise RuntimeError("authenticated collision endpoints are incomplete")
    return endpoints, sequences_by_length, {
        "collision_endpoint_count": len(endpoints),
        "materialized_batch_artifact_sha256s": tuple(
            batch_artifact_sha256s
        ),
        "manifold_lift_artifact_sha256s": tuple(lift_artifact_sha256s),
        "reproduced_target_probe_sha256s": tuple(target_hashes),
        "all_40_target_hashes_match_opened_assessment": True,
        "l3_mlp_body_execution_count": 0,
        "l4_mlp_body_execution_count": 0,
        "tokenizer_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "natural_activation_rows_loaded": False,
    }


def _target_relative_difference(
    left: _Endpoint,
    right: _Endpoint,
) -> float:
    if (
        left.checkpoints[_WEIGHTED_TARGET_INDEX].shape
        != right.checkpoints[_WEIGHTED_TARGET_INDEX].shape
        or not torch.equal(left.valid_mask, right.valid_mask)
        or not torch.equal(left.logical_positions, right.logical_positions)
    ):
        raise ValueError("collision endpoints are not aligned")
    left_rows = left.checkpoints[_WEIGHTED_TARGET_INDEX][left.valid_mask]
    right_rows = right.checkpoints[_WEIGHTED_TARGET_INDEX][right.valid_mask]
    left_norm = float(torch.linalg.vector_norm(left_rows))
    right_norm = float(torch.linalg.vector_norm(right_rows))
    difference = float(torch.linalg.vector_norm(right_rows - left_rows))
    return difference / max(0.5 * (left_norm + right_norm), 1e-12)


def _build_pairs(
    endpoints: Mapping[str, _Endpoint],
    *,
    score: FullWidthCandidateScore,
) -> tuple[_Pair, ...]:
    grouped: dict[str, list[_Endpoint]] = defaultdict(list)
    for endpoint in endpoints.values():
        group = endpoint.probe.collision_group
        if group is None:
            raise ValueError("collision endpoint is missing its group")
        grouped[group].append(endpoint)
    if len(grouped) != _EXPECTED_COLLISION_GROUPS:
        raise ValueError("collision group count drifted")
    score_by_group = {
        value.collision_group: value.minimum_pairwise_target_relative_difference
        for value in score.collision_metrics
    }
    pairs: list[_Pair] = []
    for group in sorted(grouped):
        variants = sorted(
            grouped[group],
            key=lambda value: str(value.probe.collision_variant),
        )
        pair_rows: list[tuple[_Endpoint, _Endpoint, float]] = []
        for left_index, left in enumerate(variants):
            for right in variants[left_index + 1 :]:
                pair_rows.append(
                    (
                        left,
                        right,
                        _target_relative_difference(left, right),
                    )
                )
        if not pair_rows:
            raise ValueError("collision group contains no pair")
        witness_left, witness_right, witness_value = min(
            pair_rows,
            key=lambda row: (
                row[2],
                str(row[0].probe.collision_variant),
                str(row[1].probe.collision_variant),
            ),
        )
        stored_value = score_by_group.get(group)
        if (
            stored_value is None
            or not math.isclose(
                witness_value,
                stored_value,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                f"reproduced collision gate value drifted for {group}"
            )
        for left, right, value in pair_rows:
            left_variant = str(left.probe.collision_variant)
            right_variant = str(right.probe.collision_variant)
            pairs.append(
                _Pair(
                    pair_id=f"{group}:{left_variant}--{right_variant}",
                    family=left.probe.family,
                    collision_group=group,
                    left_variant=left_variant,
                    right_variant=right_variant,
                    left_probe_id=left.probe.probe_id,
                    right_probe_id=right.probe.probe_id,
                    sequence_length=left.probe.sequence_length,
                    source_offset=left.probe.source_offset,
                    target_relative_difference=value,
                    gate_witness=(
                        left is witness_left and right is witness_right
                    ),
                )
            )
    if (
        len(pairs) != _EXPECTED_COLLISION_PAIRS
        or sum(pair.gate_witness for pair in pairs)
        != _EXPECTED_COLLISION_GROUPS
    ):
        raise RuntimeError("collision pair or witness count drifted")
    return tuple(pairs)


def _l2(value: Tensor) -> float:
    return float(
        torch.linalg.vector_norm(
            value.detach().to(device="cpu", dtype=torch.float64)
        )
    )


def _cosine(left: Tensor, right: Tensor, *, floor: float = 1e-24) -> float | None:
    left64 = left.detach().to(device="cpu", dtype=torch.float64).flatten()
    right64 = right.detach().to(device="cpu", dtype=torch.float64).flatten()
    left_norm = float(torch.linalg.vector_norm(left64))
    right_norm = float(torch.linalg.vector_norm(right64))
    if left_norm <= floor or right_norm <= floor:
        return None
    return float(torch.dot(left64, right64) / (left_norm * right_norm))


def _region_energy_fractions(
    value: Tensor,
    *,
    source_offset: int,
) -> dict[str, float]:
    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim < 3 or tensor.shape[0] != 1:
        raise ValueError("region energy requires one [1, sequence, ...] tensor")
    row_energy = tensor.square().flatten(start_dim=2).sum(dim=-1)[0]
    total = float(row_energy.sum())
    if total == 0.0:
        return {
            "pre_source_fraction": 0.0,
            "source_row_fraction": 0.0,
            "post_source_fraction": 0.0,
        }
    return {
        "pre_source_fraction": float(row_energy[:source_offset].sum()) / total,
        "source_row_fraction": float(row_energy[source_offset]) / total,
        "post_source_fraction": float(
            row_energy[source_offset + 1 :].sum()
        )
        / total,
    }


def _finite_checkpoint_metric(
    *,
    name: str,
    left: Tensor,
    right: Tensor,
    left_repeat: Tensor,
    right_repeat: Tensor,
    midpoint: Tensor,
    jvp: Tensor,
    vjp: Tensor | None,
    source_offset: int,
    input_jvp_l2: float,
    target_delta_l2: float,
    execution_epsilon: float,
    execution_tiny: float,
) -> dict[str, object]:
    delta = right - left
    left_l2 = _l2(left)
    right_l2 = _l2(right)
    secant_l2 = _l2(delta)
    scale = max(
        0.5 * (left_l2 + right_l2),
        execution_tiny,
    )
    repeat_noise = max(
        _l2(left_repeat - left),
        _l2(right_repeat - right),
    )
    floor = max(
        repeat_noise,
        execution_epsilon * _NUMERIC_FLOOR_EPSILON_MULTIPLIER * scale,
    )
    jvp_l2 = _l2(jvp)
    vjp_l2 = None if vjp is None else _l2(vjp)
    aligned = (
        None
        if vjp is None
        else float(
            torch.dot(
                vjp.detach().to(torch.float64).flatten(),
                delta.detach().to(torch.float64).flatten(),
            )
        )
    )
    return {
        "checkpoint_name": name,
        "left_sha256": _tensor_sha256(left),
        "right_sha256": _tensor_sha256(right),
        "midpoint_sha256": _tensor_sha256(midpoint),
        "jvp_sha256": _tensor_sha256(jvp),
        "left_l2": left_l2,
        "right_l2": right_l2,
        "secant_l2": secant_l2,
        "repeat_noise_l2": repeat_noise,
        "numeric_floor_l2": floor,
        "symmetric_relative_separation": secant_l2 / scale,
        "resolved": (
            secant_l2 > _RESOLVED_NOISE_MULTIPLIER * floor
        ),
        "jvp_l2": jvp_l2,
        "midpoint_jvp_relative_response": jvp_l2 / scale,
        "jvp_secant_cosine": _cosine(jvp, delta, floor=floor),
        "midpoint_linearization_relative_error": (
            _l2(jvp - delta) / max(secant_l2, floor)
        ),
        "cumulative_jvp_gain_from_l3_hidden": (
            jvp_l2 / max(input_jvp_l2, torch.finfo(delta.dtype).tiny)
        ),
        "vjp_l2": vjp_l2,
        "vjp_sha256": None if vjp is None else _tensor_sha256(vjp),
        "vjp_secant_inner_product": aligned,
        "contrast_aligned_fraction": (
            None
            if aligned is None
            else aligned / max(target_delta_l2, floor)
        ),
        "secant_energy_regions": _region_energy_fractions(
            delta,
            source_offset=source_offset,
        ),
        "jvp_energy_regions": _region_energy_fractions(
            jvp,
            source_offset=source_offset,
        ),
        "vjp_energy_regions": (
            None
            if vjp is None
            else _region_energy_fractions(
                vjp,
                source_offset=source_offset,
            )
        ),
    }


def _lift_request_metric(left: _Endpoint, right: _Endpoint) -> dict[str, object]:
    request_delta = right.requested_modes - left.requested_modes
    hidden_delta = (
        right.checkpoints[_HIDDEN_INDEX] - left.checkpoints[_HIDDEN_INDEX]
    )
    actual_x3_delta = (
        right.checkpoints[_ACTUAL_X3_INDEX]
        - left.checkpoints[_ACTUAL_X3_INDEX]
    )
    null_delta = (
        right.normalized_null_features - left.normalized_null_features
    )
    rms_delta = right.row_rms - left.row_rms
    return {
        "left_requested_modes_sha256": _tensor_sha256(left.requested_modes),
        "right_requested_modes_sha256": _tensor_sha256(right.requested_modes),
        "requested_mode_secant_l2": _l2(request_delta),
        "realized_hidden_secant_l2": _l2(hidden_delta),
        "realized_actual_x3_secant_l2": _l2(actual_x3_delta),
        "normalized_null_feature_secant_l2": _l2(null_delta),
        "row_rms_secant_l2": _l2(rms_delta),
        "request_to_hidden_is_not_a_differentiated_path": True,
        "requested_modes_are_construction_seeds_only": True,
    }


def _residual_cancellation_metrics(
    left: _Endpoint,
    right: _Endpoint,
) -> dict[str, float | None]:
    residual = (
        right.checkpoints[_L4_INPUT_INDEX]
        - left.checkpoints[_L4_INPUT_INDEX]
    )
    attention = (
        right.checkpoints[_ATTENTION_DELTA_INDEX]
        - left.checkpoints[_ATTENTION_DELTA_INDEX]
    )
    merged = (
        right.checkpoints[_POST_ATTENTION_INDEX]
        - left.checkpoints[_POST_ATTENTION_INDEX]
    )
    denominator = _l2(residual) + _l2(attention)
    additivity_error = _l2(merged - residual - attention) / max(
        _l2(merged),
        1e-24,
    )
    return {
        "residual_attention_secant_cosine": _cosine(residual, attention),
        "merge_cancellation_ratio": (
            _l2(merged) / denominator if denominator > 0.0 else 0.0
        ),
        "residual_additivity_relative_error": additivity_error,
    }


def _retained_fisher_capture(
    *,
    left: _Endpoint,
    right: _Endpoint,
    basis: Gemma3L3L4BasisPackage,
) -> dict[str, float]:
    delta_x4 = (
        right.checkpoints[_X4_INDEX] - left.checkpoints[_X4_INDEX]
    ).to(dtype=torch.float64)
    all_modes = delta_x4 @ basis.R4.T
    weights = torch.sqrt(basis.S4).view(1, 1, -1)
    weighted = all_modes * weights
    full_energy = float(weighted.square().sum())
    retained_energy = float(weighted[..., :_TARGET_RANK].square().sum())
    return {
        "full_width_fisher_weighted_secant_l2": math.sqrt(full_energy),
        "retained_64_fisher_weighted_secant_l2": math.sqrt(retained_energy),
        "retained_64_fisher_energy_fraction": (
            retained_energy / full_energy if full_energy > 0.0 else 0.0
        ),
    }


def _target_causal_leakage_fraction(
    checkpoints: Sequence[Mapping[str, object]],
) -> float:
    by_name = {str(row["checkpoint_name"]): row for row in checkpoints}
    target = by_name.get("l4.fisher_weighted_target")
    if target is None:
        raise ValueError("weighted target checkpoint is missing")
    regions = target.get("jvp_energy_regions")
    if not isinstance(regions, Mapping):
        raise TypeError("weighted target JVP regions must be a mapping")
    value = regions.get("pre_source_fraction")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("weighted target causal leakage must be in [0, 1]")
    return float(value)


def _mechanism_summary(
    *,
    pair: _Pair,
    checkpoints: Sequence[Mapping[str, object]],
    cancellation: Mapping[str, float | None],
    fisher_capture: Mapping[str, float],
    adjoint_relative_error: float,
    causal_leakage_fraction: float,
) -> dict[str, object]:
    by_name = {str(row["checkpoint_name"]): row for row in checkpoints}
    target = by_name["l4.fisher_weighted_target"]
    target_resolved = bool(target["resolved"])
    if not target_resolved:
        teacher_status = "numerically_unresolved"
    elif pair.target_relative_difference < _COLLISION_THRESHOLD:
        teacher_status = "teacher_construct_failure"
    else:
        teacher_status = "teacher_contrast_eligible"

    chain = (
        "layer.3.post_attention_hidden",
        "layer.4.input",
        "layer.4.attention.normalized_input",
        "layer.4.attention.operator_output",
        "layer.4.attention.delta",
        "layer.4.post_attention",
        "layer.4.mlp.normalized_input",
        "l4.fisher_weighted_target",
    )
    diagnostic_valid = (
        target_resolved
        and float(target["midpoint_linearization_relative_error"])
        <= _CURVATURE_RELATIVE_ERROR
        and adjoint_relative_error
        <= _CONTRAST_THRESHOLDS.maximum_adjoint_relative_error
        and causal_leakage_fraction
        <= _CONTRAST_THRESHOLDS.maximum_causal_leakage_fraction
    )
    contractions: list[tuple[float, str]] = []
    for previous, current in zip(chain, chain[1:]):
        previous_row = by_name[previous]
        current_row = by_name[current]
        previous_response = float(
            by_name[previous]["midpoint_jvp_relative_response"]
        )
        current_response = float(
            by_name[current]["midpoint_jvp_relative_response"]
        )
        if (
            diagnostic_valid
            and bool(previous_row["resolved"])
            and bool(current_row["resolved"])
            and float(previous_row["midpoint_linearization_relative_error"])
            <= _CURVATURE_RELATIVE_ERROR
            and float(current_row["midpoint_linearization_relative_error"])
            <= _CURVATURE_RELATIVE_ERROR
            and float(previous_row["jvp_l2"])
            > float(previous_row["numeric_floor_l2"])
        ):
            contractions.append(
                (
                    current_response / previous_response,
                    f"{previous} -> {current}",
                )
            )
    contractions.sort(key=lambda value: (value[0], value[1]))
    minimum_ratio, minimum_transition = (
        contractions[0] if contractions else (1.0, None)
    )
    observations: list[str] = []
    sharp_contraction = (
        minimum_transition is not None
        and minimum_ratio <= _ATTENUATION_RATIO
        and (
            len(contractions) == 1
            or minimum_ratio
            <= (
                contractions[1][0]
                * _CONTRAST_THRESHOLDS.localization_dominance_ratio
            )
        )
    )
    if sharp_contraction:
        observations.append(
            (
                "reference_baseline_relative_dilution"
                if minimum_transition
                == "layer.3.post_attention_hidden -> layer.4.input"
                else "sharp_midpoint_jvp_relative_response_contraction"
            )
        )
    if (
        target_resolved
        and float(target["midpoint_linearization_relative_error"])
        > _CURVATURE_RELATIVE_ERROR
    ):
        observations.append("local_curvature")
    cancellation_cosine = cancellation["residual_attention_secant_cosine"]
    if (
        diagnostic_valid
        and bool(by_name["layer.4.input"]["resolved"])
        and bool(by_name["layer.4.attention.delta"]["resolved"])
        and bool(by_name["layer.4.post_attention"]["resolved"])
        and cancellation_cosine is not None
        and cancellation_cosine <= _CANCELLATION_COSINE
        and float(cancellation["merge_cancellation_ratio"])
        <= _CANCELLATION_MERGE_RATIO
    ):
        observations.append("residual_attention_cancellation")
    if (
        target_resolved
        and fisher_capture["retained_64_fisher_energy_fraction"]
        <= _RETAINED_FISHER_FRACTION
    ):
        observations.append("retained_fisher_subspace_miss")
    if (
        diagnostic_valid
        and bool(by_name["layer.4.input"]["resolved"])
        and bool(
            by_name["layer.4.attention.normalized_input"]["resolved"]
        )
        and float(
            by_name["layer.4.input"]["symmetric_relative_separation"]
        )
        > 0.0
        and float(
            by_name["layer.4.attention.normalized_input"][
                "symmetric_relative_separation"
            ]
        )
        <= _ATTENUATION_RATIO
        * float(
            by_name["layer.4.input"]["symmetric_relative_separation"]
        )
    ):
        observations.append("l4_input_norm_attenuation")
    if (
        diagnostic_valid
        and bool(by_name["layer.4.post_attention"]["resolved"])
        and bool(by_name["layer.4.mlp.normalized_input"]["resolved"])
        and float(
            by_name["layer.4.post_attention"][
                "symmetric_relative_separation"
            ]
        )
        > 0.0
        and float(
            by_name["layer.4.mlp.normalized_input"][
                "symmetric_relative_separation"
            ]
        )
        <= _ATTENUATION_RATIO
        * float(
            by_name["layer.4.post_attention"][
                "symmetric_relative_separation"
            ]
        )
    ):
        observations.append("pre_ff_norm_attenuation")
    return {
        "teacher_status": teacher_status,
        "mechanism_observations": tuple(observations),
        "dominant_midpoint_jvp_relative_response_contraction_transition": (
            minimum_transition if sharp_contraction else None
        ),
        "dominant_midpoint_jvp_relative_response_contraction_ratio": (
            minimum_ratio if sharp_contraction else None
        ),
        "minimum_valid_midpoint_jvp_relative_response_transition": (
            minimum_transition
        ),
        "minimum_valid_midpoint_jvp_relative_response_ratio": (
            minimum_ratio if minimum_transition is not None else None
        ),
        "diagnostic_numerically_valid": diagnostic_valid,
        "causal_leakage_fraction": causal_leakage_fraction,
        "candidate_tracking_failure_assigned": False,
        "observed_contraction_is_not_alone_a_sufficient_cause": True,
        "classification_is_retrospective_and_exploratory": True,
    }


def _analyze_pairs(
    *,
    pairs: Sequence[_Pair],
    endpoints: Mapping[str, _Endpoint],
    basis: Gemma3L3L4BasisPackage,
    adapter: Gemma3CausalLMAdapter,
    pre_ff3: nn.Module,
    post_ff3: nn.Module,
    compiled: object,
) -> tuple[dict[str, object], ...]:
    first_parameter = next(adapter.module.parameters(), None)
    if first_parameter is None:
        raise TypeError("live model has no parameter")
    device = first_parameter.device
    dtype = first_parameter.dtype
    y3_mean = basis.y3_mean.to(device=device, dtype=dtype)
    x4_mean = basis.x4_mean.to(device=device, dtype=dtype)
    r4 = basis.R4[:_TARGET_RANK].to(device=device, dtype=dtype)
    metric_weight = getattr(compiled, "metric_weight").to(
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        layer3_mean_delta = post_ff3(y3_mean)
    grouped: dict[int, list[_Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.sequence_length].append(pair)
    results: list[dict[str, object]] = []
    for length in sorted(grouped):
        length_pairs = sorted(grouped[length], key=lambda value: value.pair_id)
        left_hidden = torch.cat(
            [
                endpoints[pair.left_probe_id].checkpoints[_HIDDEN_INDEX]
                for pair in length_pairs
            ],
            dim=0,
        ).to(device=device, dtype=dtype)
        right_hidden = torch.cat(
            [
                endpoints[pair.right_probe_id].checkpoints[_HIDDEN_INDEX]
                for pair in length_pairs
            ],
            dim=0,
        ).to(device=device, dtype=dtype)
        midpoint = (left_hidden + right_hidden) * 0.5
        tangent = right_hidden - left_hidden
        sequence = _prepare_sequence(
            adapter,
            batch_size=len(length_pairs),
            sequence_length=length,
            device=device,
            dtype=dtype,
            residual_width=basis.residual_width,
        )
        run = _checkpoint_function(
            adapter=adapter,
            pre_ff3=pre_ff3,
            sequence=sequence,
            layer3_mean_delta=layer3_mean_delta,
            x4_mean=x4_mean,
            r4=r4,
            metric_weight=metric_weight,
        )
        with torch.enable_grad():
            midpoint_outputs, jvps = torch.autograd.functional.jvp(
                run,
                midpoint,
                tangent,
                create_graph=False,
                strict=True,
            )
        if (
            not isinstance(midpoint_outputs, tuple)
            or not isinstance(jvps, tuple)
            or len(midpoint_outputs) != len(_CHECKPOINT_NAMES)
            or len(jvps) != len(_CHECKPOINT_NAMES)
        ):
            raise RuntimeError("batched midpoint JVP returned invalid checkpoints")

        exact_target_deltas = torch.cat(
            [
                endpoints[pair.right_probe_id].checkpoints[
                    _WEIGHTED_TARGET_INDEX
                ]
                - endpoints[pair.left_probe_id].checkpoints[
                    _WEIGHTED_TARGET_INDEX
                ]
                for pair in length_pairs
            ],
            dim=0,
        )
        denominators: list[float] = []
        for index, pair in enumerate(length_pairs):
            left_target = endpoints[pair.left_probe_id].checkpoints[
                _WEIGHTED_TARGET_INDEX
            ]
            right_target = endpoints[pair.right_probe_id].checkpoints[
                _WEIGHTED_TARGET_INDEX
            ]
            scale = max(
                0.5 * (_l2(left_target) + _l2(right_target)),
                torch.finfo(dtype).tiny,
            )
            floor = (
                torch.finfo(dtype).eps
                * _NUMERIC_FLOOR_EPSILON_MULTIPLIER
                * scale
            )
            denominators.append(
                max(_l2(exact_target_deltas[index : index + 1]), floor)
            )
        denominator = torch.tensor(
            denominators,
            device=device,
            dtype=dtype,
        ).view(-1, 1, 1)
        cotangent = exact_target_deltas.to(
            device=device,
            dtype=dtype,
        ) / denominator

        midpoint_leaf = midpoint.detach().requires_grad_(True)
        with torch.enable_grad():
            vjp_outputs = run(midpoint_leaf)
            scalar = (vjp_outputs[_WEIGHTED_TARGET_INDEX] * cotangent).sum()
            input_gradient = torch.autograd.grad(
                scalar,
                midpoint_leaf,
                retain_graph=True,
                allow_unused=False,
            )[0]
            checkpoint_gradients = torch.autograd.grad(
                scalar,
                vjp_outputs,
                retain_graph=False,
                allow_unused=True,
            )
        midpoint64 = tuple(
            value.detach().to(device="cpu", dtype=torch.float64)
            for value in midpoint_outputs
        )
        jvps64 = tuple(
            value.detach().to(device="cpu", dtype=torch.float64)
            for value in jvps
        )
        vjps64 = tuple(
            None
            if value is None
            else value.detach().to(device="cpu", dtype=torch.float64)
            for value in checkpoint_gradients
        )
        input_gradient64 = input_gradient.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        tangent64 = tangent.detach().to(device="cpu", dtype=torch.float64)
        cotangent64 = cotangent.detach().to(device="cpu", dtype=torch.float64)

        for index, pair in enumerate(length_pairs):
            left = endpoints[pair.left_probe_id]
            right = endpoints[pair.right_probe_id]
            target_delta_l2 = _l2(
                right.checkpoints[_WEIGHTED_TARGET_INDEX]
                - left.checkpoints[_WEIGHTED_TARGET_INDEX]
            )
            input_jvp_l2 = _l2(tangent64[index : index + 1])
            checkpoint_rows: list[dict[str, object]] = []
            for checkpoint_index, name in enumerate(_CHECKPOINT_NAMES):
                checkpoint_rows.append(
                    _finite_checkpoint_metric(
                        name=name,
                        left=left.checkpoints[checkpoint_index],
                        right=right.checkpoints[checkpoint_index],
                        left_repeat=left.repeated_checkpoints[checkpoint_index],
                        right_repeat=right.repeated_checkpoints[
                            checkpoint_index
                        ],
                        midpoint=midpoint64[checkpoint_index][
                            index : index + 1
                        ],
                        jvp=jvps64[checkpoint_index][index : index + 1],
                        vjp=(
                            None
                            if vjps64[checkpoint_index] is None
                            else vjps64[checkpoint_index][index : index + 1]
                        ),
                        source_offset=pair.source_offset,
                        input_jvp_l2=input_jvp_l2,
                        target_delta_l2=target_delta_l2,
                        execution_epsilon=float(torch.finfo(dtype).eps),
                        execution_tiny=float(torch.finfo(dtype).tiny),
                    )
                )
            input_adjoint = float(
                torch.dot(
                    input_gradient64[index].flatten(),
                    tangent64[index].flatten(),
                )
            )
            output_adjoint = float(
                torch.dot(
                    cotangent64[index].flatten(),
                    jvps64[_WEIGHTED_TARGET_INDEX][index].flatten(),
                )
            )
            output_floor = float(
                checkpoint_rows[_WEIGHTED_TARGET_INDEX][
                    "numeric_floor_l2"
                ]
            )
            adjoint_error = abs(input_adjoint - output_adjoint) / max(
                abs(input_adjoint),
                abs(output_adjoint),
                output_floor,
            )
            causal_leakage = _target_causal_leakage_fraction(checkpoint_rows)
            cancellation = _residual_cancellation_metrics(left, right)
            fisher_capture = _retained_fisher_capture(
                left=left,
                right=right,
                basis=basis,
            )
            mechanism = _mechanism_summary(
                pair=pair,
                checkpoints=checkpoint_rows,
                cancellation=cancellation,
                fisher_capture=fisher_capture,
                adjoint_relative_error=adjoint_error,
                causal_leakage_fraction=causal_leakage,
            )
            results.append(
                {
                    "pair_id": pair.pair_id,
                    "family": pair.family,
                    "collision_group": pair.collision_group,
                    "left_variant": pair.left_variant,
                    "right_variant": pair.right_variant,
                    "left_probe_id": pair.left_probe_id,
                    "right_probe_id": pair.right_probe_id,
                    "left_target_probe_sha256": left.target_probe_sha256,
                    "right_target_probe_sha256": right.target_probe_sha256,
                    "sequence_length": pair.sequence_length,
                    "source_offset": pair.source_offset,
                    "target_relative_difference": (
                        pair.target_relative_difference
                    ),
                    "collision_threshold": _COLLISION_THRESHOLD,
                    "gate_witness": pair.gate_witness,
                    "lift_request": _lift_request_metric(left, right),
                    "checkpoints": tuple(checkpoint_rows),
                    "jvp_vjp_adjoint_left": input_adjoint,
                    "jvp_vjp_adjoint_right": output_adjoint,
                    "jvp_vjp_adjoint_relative_error": adjoint_error,
                    "midpoint_input_sha256": _tensor_sha256(
                        midpoint64[_HIDDEN_INDEX][index : index + 1]
                    ),
                    "input_tangent_sha256": _tensor_sha256(
                        tangent64[index : index + 1]
                    ),
                    "output_cotangent_sha256": _tensor_sha256(
                        cotangent64[index : index + 1]
                    ),
                    "input_gradient_sha256": _tensor_sha256(
                        input_gradient64[index : index + 1]
                    ),
                    "residual_attention_cancellation": cancellation,
                    "retained_fisher_capture": fisher_capture,
                    **mechanism,
                }
            )
    results.sort(key=lambda value: str(value["pair_id"]))
    if len(results) != _EXPECTED_COLLISION_PAIRS:
        raise RuntimeError("batched localization result count drifted")
    return tuple(results)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("distribution requires at least one value")
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return {
        "minimum": float(tensor.min()),
        "median": float(torch.quantile(tensor, 0.5)),
        "p90": float(torch.quantile(tensor, 0.9)),
        "maximum": float(tensor.max()),
    }


def _aggregate_results(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(pairs) != _EXPECTED_COLLISION_PAIRS:
        raise ValueError("aggregate requires the complete collision pair set")
    witnesses = tuple(pair for pair in pairs if bool(pair["gate_witness"]))
    if len(witnesses) != _EXPECTED_COLLISION_GROUPS:
        raise ValueError("aggregate requires one witness per collision group")

    family_rows: dict[str, dict[str, object]] = {}
    families = sorted({str(pair["family"]) for pair in pairs})
    for family in families:
        rows = tuple(pair for pair in pairs if pair["family"] == family)
        family_witnesses = tuple(
            pair for pair in rows if bool(pair["gate_witness"])
        )
        statuses = Counter(
            str(pair["teacher_status"]) for pair in family_witnesses
        )
        observations: Counter[str] = Counter()
        transitions: Counter[str] = Counter()
        for pair in family_witnesses:
            observations.update(
                str(value) for value in pair["mechanism_observations"]  # type: ignore[union-attr]
            )
            transition = pair[
                "dominant_midpoint_jvp_relative_response_contraction_transition"
            ]
            if transition is not None:
                transitions[str(transition)] += 1
        family_rows[family] = {
            "pair_count": len(rows),
            "gate_witness_count": len(family_witnesses),
            "gate_witnesses_at_or_above_collision_threshold": sum(
                float(pair["target_relative_difference"])
                >= _COLLISION_THRESHOLD
                for pair in family_witnesses
            ),
            "gate_witness_conclusion": (
                "teacher_contrast_eligible"
                if all(
                    float(pair["target_relative_difference"])
                    >= _COLLISION_THRESHOLD
                    for pair in family_witnesses
                )
                else (
                    "teacher_contrast_ineligible"
                    if all(
                        float(pair["target_relative_difference"])
                        < _COLLISION_THRESHOLD
                        for pair in family_witnesses
                    )
                    else "mixed_teacher_contrast_eligibility"
                )
            ),
            "target_relative_difference": _distribution(
                [
                    float(pair["target_relative_difference"])
                    for pair in rows
                ]
            ),
            "gate_witness_target_relative_difference": _distribution(
                [
                    float(pair["target_relative_difference"])
                    for pair in family_witnesses
                ]
            ),
            "teacher_status_counts": dict(sorted(statuses.items())),
            "mechanism_observation_counts": dict(
                sorted(observations.items())
            ),
            "dominant_jvp_relative_response_contraction_transition_counts": dict(
                sorted(transitions.items())
            ),
            "retained_64_fisher_energy_fraction": _distribution(
                [
                    float(
                        pair["retained_fisher_capture"][  # type: ignore[index]
                            "retained_64_fisher_energy_fraction"
                        ]
                    )
                    for pair in rows
                ]
            ),
            "final_midpoint_linearization_relative_error": _distribution(
                [
                    float(
                        pair["checkpoints"][-1][  # type: ignore[index]
                            "midpoint_linearization_relative_error"
                        ]
                    )
                    for pair in rows
                ]
            ),
            "gate_witness_lift_request": {
                key: _distribution(
                    [
                        float(pair["lift_request"][key])  # type: ignore[index]
                        for pair in family_witnesses
                    ]
                )
                for key in (
                    "requested_mode_secant_l2",
                    "realized_hidden_secant_l2",
                    "realized_actual_x3_secant_l2",
                    "normalized_null_feature_secant_l2",
                    "row_rms_secant_l2",
                )
            },
            "gate_witness_checkpoints": {
                name: {
                    "resolved_count": sum(
                        bool(pair["checkpoints"][index]["resolved"])  # type: ignore[index]
                        for pair in family_witnesses
                    ),
                    "symmetric_relative_separation": _distribution(
                        [
                            float(
                                pair["checkpoints"][index][  # type: ignore[index]
                                    "symmetric_relative_separation"
                                ]
                            )
                            for pair in family_witnesses
                        ]
                    ),
                    "cumulative_jvp_gain_from_l3_hidden": _distribution(
                        [
                            float(
                                pair["checkpoints"][index][  # type: ignore[index]
                                    "cumulative_jvp_gain_from_l3_hidden"
                                ]
                            )
                            for pair in family_witnesses
                        ]
                    ),
                    "midpoint_jvp_relative_response": _distribution(
                        [
                            float(
                                pair["checkpoints"][index][  # type: ignore[index]
                                    "midpoint_jvp_relative_response"
                                ]
                            )
                            for pair in family_witnesses
                        ]
                    ),
                }
                for index, name in enumerate(_CHECKPOINT_NAMES)
            },
        }

    all_observations: Counter[str] = Counter()
    for pair in witnesses:
        all_observations.update(
            str(value) for value in pair["mechanism_observations"]  # type: ignore[union-attr]
        )
    failed_witnesses = tuple(
        pair
        for pair in witnesses
        if float(pair["target_relative_difference"]) < _COLLISION_THRESHOLD
    )
    eligible_witnesses = tuple(
        pair
        for pair in witnesses
        if float(pair["target_relative_difference"]) >= _COLLISION_THRESHOLD
    )
    failed_observations = {
        str(value)
        for pair in failed_witnesses
        for value in pair["mechanism_observations"]  # type: ignore[union-attr]
    }
    eligible_observations = {
        str(value)
        for pair in eligible_witnesses
        for value in pair["mechanism_observations"]  # type: ignore[union-attr]
    }
    max_adjoint = max(
        float(pair["jvp_vjp_adjoint_relative_error"]) for pair in pairs
    )
    max_pre_source_jvp = max(
        float(
            checkpoint["jvp_energy_regions"]["pre_source_fraction"]  # type: ignore[index]
        )
        for pair in pairs
        for checkpoint in pair["checkpoints"]  # type: ignore[union-attr]
    )
    return {
        "collision_endpoint_count": _EXPECTED_COLLISION_ENDPOINTS,
        "collision_group_count": _EXPECTED_COLLISION_GROUPS,
        "unordered_pair_count": _EXPECTED_COLLISION_PAIRS,
        "gate_witness_count": len(witnesses),
        "collision_threshold": _COLLISION_THRESHOLD,
        "gate_witnesses_at_or_above_threshold": sum(
            float(pair["target_relative_difference"]) >= _COLLISION_THRESHOLD
            for pair in witnesses
        ),
        "gate_witnesses_below_threshold": sum(
            float(pair["target_relative_difference"]) < _COLLISION_THRESHOLD
            for pair in witnesses
        ),
        "gate_witness_target_relative_difference": _distribution(
            [
                float(pair["target_relative_difference"])
                for pair in witnesses
            ]
        ),
        "gate_witness_mechanism_observation_counts": dict(
            sorted(all_observations.items())
        ),
        "observations_shared_with_passing_controls": tuple(
            sorted(failed_observations & eligible_observations)
        ),
        "observations_exclusive_to_failed_witnesses": tuple(
            sorted(failed_observations - eligible_observations)
        ),
        "failed_witnesses_with_retained_fisher_subspace_miss": sum(
            "retained_fisher_subspace_miss"
            in pair["mechanism_observations"]  # type: ignore[operator]
            for pair in failed_witnesses
        ),
        "failed_witnesses_with_residual_attention_cancellation": sum(
            "residual_attention_cancellation"
            in pair["mechanism_observations"]  # type: ignore[operator]
            for pair in failed_witnesses
        ),
        "diagnostic_conclusion": (
            "collision_failure_precedes_candidate_tracking_and_is_a_teacher_"
            "contrast_eligibility_failure"
        ),
        "candidate_predictions_entered_collision_metric": False,
        "maximum_jvp_vjp_adjoint_relative_error": max_adjoint,
        "maximum_pre_source_jvp_energy_fraction": max_pre_source_jvp,
        "families": family_rows,
    }


def run_gemma3_l3_l4_attenuation_localization_experiment(
    *,
    candidate_path: Path | str = DEFAULT_CANDIDATE,
    candidate_file_sha256: str = DEFAULT_CANDIDATE_FILE_SHA256,
    candidate_report_sha256: str = DEFAULT_CANDIDATE_REPORT_SHA256,
    assessment_path: Path | str = DEFAULT_ASSESSMENT,
    assessment_file_sha256: str = DEFAULT_ASSESSMENT_FILE_SHA256,
    assessment_report_sha256: str = DEFAULT_ASSESSMENT_REPORT_SHA256,
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
) -> dict[str, object]:
    """Run the authenticated retrospective collision attenuation trace."""

    destination = _validate_output_path(output)
    candidate_bytes_before = _read_v2_regular_file(candidate_path)
    candidate_report_bytes_before = _read_v2_regular_file(
        Path(candidate_path).with_suffix(".json")
    )
    assessment_bytes_before = _read_regular_file(assessment_path)
    assessment_report_bytes_before = _read_regular_file(
        Path(assessment_path).with_suffix(".json")
    )
    basis_package_bytes_before = _read_regular_file(basis_package_path)
    # The strict v2 candidate and opened-assessment bindings are authenticated
    # before loading a live model capable of reexecuting target endpoints.
    compiled = _load_reference_compilation(
        candidate_path,
        expected_file_sha256=candidate_file_sha256,
        expected_report_sha256=candidate_report_sha256,
    )
    opened = _load_opened_reference_assessment(
        assessment_path,
        expected_file_sha256=assessment_file_sha256,
        expected_report_sha256=assessment_report_sha256,
        compiled=compiled,
    )
    if (
        compiled.manifest.get("basis_package_file_sha256")
        != basis_package_file_sha256
        or compiled.manifest.get("basis_package_payload_sha256")
        != basis_package_payload_sha256
        or compiled.manifest.get("source_model_sha256")
        != _SOURCE_MODEL_SHA256
    ):
        raise ValueError("live arguments differ from the frozen v2 bindings")

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
        basis.source_model_sha256 != _SOURCE_MODEL_SHA256
        or basis.basis_payload_sha256
        != DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        or not torch.equal(
            compiled.metric_weight,
            torch.sqrt(basis.S4[:_TARGET_RANK]),
        )
    ):
        raise ValueError("live basis or Fisher gauge differs from v2")
    model_before = adapter.model_fingerprint()
    pre_ff_before = module_state_fingerprint(pre_ff3)
    post_ff_before = module_state_fingerprint(post_ff3)
    v2_code_before = _v2_code_sha256s()
    local_code_before = _local_code_sha256s()
    if model_before != _SOURCE_MODEL_SHA256:
        raise ValueError("live model fingerprint differs from frozen source")
    if any(parameter.grad is not None for parameter in adapter.module.parameters()):
        raise ValueError("live model contains stale parameter gradients")

    endpoints, _sequences, endpoint_measurement = (
        _measure_collision_endpoints(
            compiled=compiled,
            opened=opened,
            basis=basis,
            adapter=adapter,
            pre_ff3=pre_ff3,
            post_ff3=post_ff3,
            epsilon=epsilon,
        )
    )
    pairs = _build_pairs(endpoints, score=opened.score)
    pair_results = _analyze_pairs(
        pairs=pairs,
        endpoints=endpoints,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        compiled=compiled,
    )
    aggregate = _aggregate_results(pair_results)

    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != pre_ff_before
        or module_state_fingerprint(post_ff3) != post_ff_before
        or _v2_code_sha256s() != v2_code_before
        or _local_code_sha256s() != local_code_before
        or _read_v2_regular_file(candidate_path) != candidate_bytes_before
        or _read_v2_regular_file(
            Path(candidate_path).with_suffix(".json")
        )
        != candidate_report_bytes_before
        or _read_regular_file(assessment_path) != assessment_bytes_before
        or _read_regular_file(
            Path(assessment_path).with_suffix(".json")
        )
        != assessment_report_bytes_before
        or _read_regular_file(basis_package_path)
        != basis_package_bytes_before
        or _read_regular_file(opened.claim_path) != opened.claim_bytes
        or any(
            parameter.grad is not None
            for parameter in adapter.module.parameters()
        )
    ):
        raise RuntimeError(
            "model, norms, artifacts, code, or parameter gradients changed "
            "during attenuation localization"
        )

    execution_device = resolve_torch_device(device_name)
    common = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "candidate_artifact_sha256": compiled.artifact_sha256,
        "candidate_file_sha256": compiled.file_sha256,
        "candidate_report_sha256": compiled.report_sha256,
        "selected_candidate_id": compiled.selected_id,
        "selected_plan_sha256": compiled.selected_plan.artifact_sha256,
        "assessment_artifact_sha256": opened.artifact_sha256,
        "assessment_file_sha256": opened.file_sha256,
        "assessment_report_sha256": opened.report_sha256,
        "assessment_panel_sha256": _PANEL_SHA256,
        "assessment_panel_spec_sha256": _PANEL_SPEC_SHA256,
        "assessment_claim_file_sha256": opened.claim_file_sha256,
        "basis_package_file_sha256": basis_package_file_sha256,
        "basis_package_payload_sha256": basis.basis_payload_sha256,
        "source_model_sha256": basis.source_model_sha256,
        "synthetic_protocol_sha256": compiled.protocol.protocol_sha256,
        "training_protocol_sha256": compiled.training.artifact_sha256,
        "v2_code_sha256s": dict(v2_code_before),
        "localization_code_sha256s": dict(local_code_before),
        "checkpoint_catalog": _CHECKPOINT_NAMES,
        "causal_path": _CAUSAL_PATH,
        "side_branches": _SIDE_BRANCHES,
        "contrast_thresholds": _CONTRAST_THRESHOLDS.state_dict(),
        "execution": {
            "torch_version": torch.__version__,
            "device": str(execution_device),
            "dtype": dtype,
            "attention_implementation": getattr(
                adapter.module.config,
                "_attn_implementation",
                None,
            ),
            "model_eval": not adapter.module.training,
            "model_parameters_frozen": not any(
                parameter.requires_grad
                for parameter in adapter.module.parameters()
            ),
        },
        "safety": {
            "assessment_panel_previously_opened": True,
            "assessment_targets_reexecuted_for_internal_localization": True,
            "new_sealed_panel_opened": False,
            "assessment_score_recomputed": False,
            "candidate_refit": False,
            "candidate_reselection": False,
            "candidate_predictions_used_for_collision_localization": False,
            "target_derived_vjp_may_become_compiler_input": False,
            "open_panel_changes_require_fresh_v3_assessment": True,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_natural_activation_rows": False,
            "contains_raw_checkpoint_tensors": False,
            "contains_model_or_provider_parameters": False,
        },
    }
    analysis = {
        "endpoint_measurement": endpoint_measurement,
        "aggregate": aggregate,
        "pairs": pair_results,
    }
    logical_artifact_sha256 = _json_sha256(
        {**common, "analysis": analysis},
        domain=_ARTIFACT_DOMAIN,
    )
    state = {
        **common,
        "artifact_sha256": logical_artifact_sha256,
        "analysis": analysis,
    }
    report_payload = {
        **state,
        "interpretation": {
            "formal_v2_decision_changed": False,
            "diagnostic_scope": (
                "retrospective_teacher_path_attenuation_on_consumed_v2_panel"
            ),
            "candidate_tracking_failure_can_be_assigned": False,
            "natural_prompt_transfer_tested": False,
            "whole_model_replacement_tested": False,
            "nll_measured": False,
            "latency_measured": False,
        },
    }
    return _publish_artifact(state, report_payload, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--candidate-file-sha256",
        default=DEFAULT_CANDIDATE_FILE_SHA256,
    )
    parser.add_argument(
        "--candidate-report-sha256",
        default=DEFAULT_CANDIDATE_REPORT_SHA256,
    )
    parser.add_argument("--assessment", type=Path, default=DEFAULT_ASSESSMENT)
    parser.add_argument(
        "--assessment-file-sha256",
        default=DEFAULT_ASSESSMENT_FILE_SHA256,
    )
    parser.add_argument(
        "--assessment-report-sha256",
        default=DEFAULT_ASSESSMENT_REPORT_SHA256,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument(
        "--basis-package-file-sha256",
        default=DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    )
    parser.add_argument(
        "--basis-package-payload-sha256",
        default=DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_attenuation_localization_experiment(
        candidate_path=arguments.candidate,
        candidate_file_sha256=arguments.candidate_file_sha256,
        candidate_report_sha256=arguments.candidate_report_sha256,
        assessment_path=arguments.assessment,
        assessment_file_sha256=arguments.assessment_file_sha256,
        assessment_report_sha256=arguments.assessment_report_sha256,
        basis_package_path=arguments.basis_package,
        basis_package_file_sha256=arguments.basis_package_file_sha256,
        basis_package_payload_sha256=(
            arguments.basis_package_payload_sha256
        ),
        output=arguments.output,
        model_id=arguments.model_id,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    aggregate = report["analysis"]["aggregate"]  # type: ignore[index]
    print(
        "Localized "
        f"{aggregate['unordered_pair_count']} collision contrasts; "  # type: ignore[index]
        f"{aggregate['gate_witnesses_at_or_above_threshold']} of "  # type: ignore[index]
        f"{aggregate['gate_witness_count']} gate witnesses met the "  # type: ignore[index]
        "original teacher-separation threshold."
    )
    print(f"Wrote {arguments.output}")
    print(f"Wrote {arguments.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
