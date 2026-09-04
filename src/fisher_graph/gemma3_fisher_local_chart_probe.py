"""Executable stage-aware Fisher local-chart probe for Gemma 3 layer 4.

The experiment is intentionally phase separated.  ``fit-select`` may open
only the reusable Calibration-A fit and selection roles and freezes a
weights-only candidate.  ``guard`` authenticates and dry-runs that candidate
against selection data before it asks the repository-global ledger for the
one irreversible A-guard claim.

This module is an analysis runner, not a compiler integration.  Its optional
full-sequence check replaces only ``layer.4.output`` in a source-authoritative
shadow forward; it never registers an overlay with the default executor.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shlex
import sys
import time
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .fisher_local_chart import (
    AffineCoordinateGenerator,
    AffineDestinationRouter,
    ChartBaselineSet,
    ConstantMeanGenerator,
    DestinationFrequencyBaseline,
    EuclideanChartBasis,
    FeatureField,
    FeatureSchema,
    FrozenAtlas,
    LocalChart,
    LowRankFisherChartBasis,
    Rank64FisherCarrier,
    TypedTransitionTable,
    build_typed_transition_table,
    fit_destination_frequency_baselines,
    fit_equal_capacity_chart_baselines,
    flow_capture_metrics,
    prediction_metrics,
    routing_metrics,
    sequence_bootstrap,
    support_drift_metrics,
    validate_generator_feature_schema,
    validate_router_feature_schema,
)
from .fisher_local_chart_carrier import (
    DEFAULT_CARRIER_SIDECAR,
    DEFAULT_FISHER_SOURCE,
    extract_fisher_local_chart_carrier,
    load_fisher_local_chart_carrier,
)
from .fisher_local_chart_protocol import (
    FisherLocalChartFrozenCandidate,
    claim_and_open_fisher_local_chart_guard,
    freeze_fisher_local_chart_candidate,
    load_fisher_local_chart_frozen_candidate,
    sequence_split_manifest_from_corpus,
    write_fisher_local_chart_guard_result,
)
from .gemma3_experiment import DEFAULT_MODEL_ID, resolve_gemma3_huggingface_paths
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
)
from .gemma3_l3_l4_progressive_a_campaign import (
    materialize_gemma3_l3_l4_progressive_panel,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveACorpus,
    Gemma3L3L4ProgressiveARolePrompts,
    load_gemma3_l3_l4_progressive_a_corpus,
)
from .gemma3_l3_l4_progressive_worker import GemmaProgressivePanel
from .gemma3_l3_l4_spectral_mapping_experiment import (
    DEFAULT_REVISION,
    _load_local_gemma3_model_only,
)
from .gemma3_fisher_local_chart_capture import (
    ATTENTION_SUMMARY_FEATURE_NAMES,
    LAYER4_CANONICAL_SITES,
    Layer4SequenceCapture,
    MLPProbePartitionManifest,
    capture_layer4_sequence,
    make_mlp_probe_partition,
)
from .shadow_fidelity import SourceAuthoritativeShadowFidelityAccumulator


SCHEMA = "fisher_graph.gemma3_fisher_local_chart_probe.v1"
REPORT_SCHEMA = "fisher_graph.gemma3_fisher_local_chart_probe.report.v1"
EXPERIMENT_ID = "gemma3-layer4-fisher-local-chart-v1"
LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_CORPUS_ARTIFACT = LOCAL_ROOT / "progressive-a-staged-v2.corpus.json"
DEFAULT_FIT_INPUT = LOCAL_ROOT / "progressive-a-pilot-v1.fit.json"
DEFAULT_SELECTION_INPUT = LOCAL_ROOT / "progressive-a-pilot-v1.selection.json"
DEFAULT_GUARD_INPUT = LOCAL_ROOT / "progressive-a-staged-v2.guard.json"
DEFAULT_OUTPUT_DIR = LOCAL_ROOT / "fisher-local-chart-probe-v1"
DEFAULT_CANDIDATE = DEFAULT_OUTPUT_DIR / "frozen_candidate.pt"
DEFAULT_CHART_COUNT = 4
DEFAULT_COORDINATE_RANK = 8
DEFAULT_RIDGE_GRID = (1.0e-6, 1.0e-4, 1.0e-2)
# Support is calibrated at one coverage fixed before any A-selection outcomes
# are observed.  A grid would be degenerate here because the generator does not
# change with the radius and out-of-support rate falls monotonically as coverage
# increases.
DEFAULT_SUPPORT_COVERAGES = (0.95,)
DEFAULT_CHART_SEED = 42_004
DEFAULT_PARTITION_SEED = 4_240_064
DEFAULT_BOOTSTRAP_REPLICATES = 500
PARTITION_SCHEMES = ("contiguous", "interleaved")
BASELINE_IDS = ("B0", "B1", "B2", "B3", "B4")
STAGES = ("h", "h_A", "h_plus")
_PROTOCOL_SOURCE_NAMES = (
    "fisher_local_chart.py",
    "fisher_local_chart_carrier.py",
    "fisher_local_chart_protocol.py",
    "gemma3_fisher_local_chart_capture.py",
    "gemma3_fisher_local_chart_probe.py",
    "shadow_fidelity.py",
    "gemma3_experiment.py",
    "gemma3_l3_l4_graph_organized_svd_shadow_protocol.py",
    "gemma3_l3_l4_graph_organized_svd_shadow_qualification.py",
    "gemma3_l3_l4_progressive_a_campaign.py",
    "gemma3_l3_l4_progressive_a_corpus.py",
    "gemma3_l3_l4_progressive_guard_ledger.py",
    "gemma3_l3_l4_progressive_worker.py",
    "gemma3_l3_l4_spectral_mapping_experiment.py",
    "adapters/base.py",
    "adapters/gemma3.py",
)

FINAL_ARTIFACT_NAMES = (
    "exact_layer4_boundary_design.md",
    "experiment_manifest.json",
    "sequence_split_manifest.json",
    "subregion_definitions.json",
    "mlp_partitions.json",
    "chart_rank_tables.json",
    "tangent_capture_tables.json",
    "affine_generator_tables.json",
    "attention_context_comparison.json",
    "teacher_composed_comparison.json",
    "chart_support_tables.json",
    "fisher_similarity_matrices.json",
    "typed_transition_matrices.json",
    "chart_transition.dot",
    "full_sequence_fidelity.json",
    "resource_runtime.json",
    "final_report.md",
    "commands.md",
    "guard_result.json",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: object, *, domain: str) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_source_sha256s() -> dict[str, str]:
    source_root = Path(__file__).parent
    return {
        name: _file_sha256(source_root / name) for name in _PROTOCOL_SOURCE_NAMES
    }


# Capture source bytes immediately after imports and before any scientific work.
# Every later protocol calculation rechecks disk against this immutable snapshot,
# so a candidate cannot bind post-fit source bytes that differ from the code the
# fresh interpreter imported.
_IMPORTED_PROTOCOL_SOURCE_SHA256S = _protocol_source_sha256s()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    receipt = hashlib.sha256()
    receipt.update(str(tensor.dtype).encode("ascii"))
    receipt.update(b"\0")
    receipt.update(_canonical_json_bytes(list(tensor.shape)))
    receipt.update(b"\0")
    receipt.update(tensor.view(torch.uint8).numpy().tobytes())
    return receipt.hexdigest()


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("exclusive artifact write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: object) -> None:
    _exclusive_write(path, _canonical_json_bytes(_jsonable(value)) + b"\n")


def _write_text(path: Path, value: str) -> None:
    _exclusive_write(path, value.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    chart_count: int = DEFAULT_CHART_COUNT
    coordinate_rank: int = DEFAULT_COORDINATE_RANK
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID
    support_coverages: tuple[float, ...] = DEFAULT_SUPPORT_COVERAGES
    chart_seed: int = DEFAULT_CHART_SEED
    partition_seed: int = DEFAULT_PARTITION_SEED
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES

    def __post_init__(self) -> None:
        if (
            isinstance(self.chart_count, bool)
            or not isinstance(self.chart_count, int)
            or self.chart_count <= 0
        ):
            raise ValueError("chart_count must be a positive integer")
        if (
            isinstance(self.coordinate_rank, bool)
            or not isinstance(self.coordinate_rank, int)
            or not 0 < self.coordinate_rank <= 64
        ):
            raise ValueError("coordinate_rank must lie in [1, 64]")
        ridges = tuple(float(value) for value in self.ridge_grid)
        if (
            not ridges
            or any(not math.isfinite(value) or value <= 0.0 for value in ridges)
            or len(set(ridges)) != len(ridges)
        ):
            raise ValueError("ridge_grid must contain unique finite positive values")
        coverages = tuple(float(value) for value in self.support_coverages)
        if (
            len(coverages) != 1
            or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in coverages)
        ):
            raise ValueError(
                "support_coverages must contain exactly one preregistered finite value in (0, 1]"
            )
        if (
            isinstance(self.chart_seed, bool)
            or not isinstance(self.chart_seed, int)
            or self.chart_seed < 0
        ):
            raise ValueError("chart_seed must be a nonnegative integer")
        if (
            isinstance(self.partition_seed, bool)
            or not isinstance(self.partition_seed, int)
            or self.partition_seed < 0
        ):
            raise ValueError("partition_seed must be a nonnegative integer")
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates <= 0
        ):
            raise ValueError("bootstrap_replicates must be a positive integer")
        object.__setattr__(self, "ridge_grid", ridges)
        object.__setattr__(self, "support_coverages", coverages)

    def to_dict(self) -> dict[str, object]:
        return {
            "chart_count": self.chart_count,
            "coordinate_rank": self.coordinate_rank,
            "ridge_grid": list(self.ridge_grid),
            "support_coverages": list(self.support_coverages),
            "chart_seed": self.chart_seed,
            "partition_seed": self.partition_seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "accumulation_dtype": "float64",
            "attention_context_features": list(ATTENTION_SUMMARY_FEATURE_NAMES),
            "mlp_partition_schemes": list(PARTITION_SCHEMES),
        }


@dataclass(slots=True)
class RoleRows:
    role: str
    sequence_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    logical_positions: Tensor
    h: Tensor
    h_A: Tensor
    h_plus: Tensor
    v_A: Tensor
    v_M: Tensor
    attention_components: Tensor
    attention_context: Tensor
    mlp_components: Mapping[str, Tensor]
    capture_diagnostics: tuple[Mapping[str, object], ...]
    teacher_forward_count: int
    capture_seconds: float

    @property
    def row_count(self) -> int:
        return int(self.h.shape[0])

    @property
    def v_total(self) -> Tensor:
        return self.h_plus - self.h

    @property
    def state_cloud(self) -> Tensor:
        return torch.cat((self.h, self.h_A, self.h_plus), dim=0)

    def assert_valid(self) -> None:
        rows = self.row_count
        if rows <= 0 or len(self.sequence_ids) != rows or len(self.family_ids) != rows:
            raise ValueError("role capture rows and sequence provenance differ")
        expected = (rows, 640)
        for name in ("h", "h_A", "h_plus", "v_A", "v_M"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"captured {name} has the wrong shape")
        if tuple(self.attention_components.shape) != (rows, 4, 640):
            raise ValueError("attention component rows have the wrong shape")
        if tuple(self.attention_context.shape) != (rows, 4, 5):
            raise ValueError("attention context rows have the wrong shape")
        if (
            tuple(self.logical_positions.shape) != (rows,)
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or bool((self.logical_positions < 0).any())
        ):
            raise ValueError("logical positions must be nonnegative row-aligned integers")
        if set(self.mlp_components) != set(PARTITION_SCHEMES) or any(
            tuple(value.shape) != (rows, 4, 640)
            for value in self.mlp_components.values()
        ):
            raise ValueError("MLP probe block rows have the wrong shape")
        if not torch.allclose(self.h_A, self.h + self.v_A, atol=1e-4, rtol=1e-5):
            raise ValueError("captured attention stage identity failed")
        if not torch.allclose(self.h_plus, self.h_A + self.v_M, atol=1e-4, rtol=1e-5):
            raise ValueError("captured MLP stage identity failed")


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    corpus: Gemma3L3L4ProgressiveACorpus
    tokenizer: object
    tokenizer_contract: Mapping[str, object]
    adapter: Gemma3CausalLMAdapter
    carrier_artifact: object
    split_manifest: object


def _runtime_context(
    *,
    corpus_artifact_path: Path,
    fit_input_path: Path,
    selection_input_path: Path,
    guard_input_path: Path,
    carrier_path: Path,
    cache_dir: Path | None,
) -> RuntimeContext:
    legacy_protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    legacy_protocol.validate_integrity()
    tokenizer, tokenizer_contract = _load_and_validate_frozen_local_tokenizer(
        protocol=legacy_protocol
    )
    role_paths = {
        "calibration_a_fit": fit_input_path,
        "calibration_a_selection": selection_input_path,
        "calibration_a_guard": guard_input_path,
    }
    corpus = load_gemma3_l3_l4_progressive_a_corpus(
        corpus_artifact_path,
        role_input_paths=role_paths,  # type: ignore[arg-type]
        tokenizer_contract=tokenizer_contract,
    )
    split_manifest = sequence_split_manifest_from_corpus(corpus.artifact)
    carrier_artifact = load_fisher_local_chart_carrier(carrier_path)
    paths = resolve_gemma3_huggingface_paths(cache_dir)
    model = _load_local_gemma3_model_only(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        cache_dir=paths["hub_cache"],
        device=torch.device("cpu"),
        dtype="float32",
    )
    adapter = Gemma3CausalLMAdapter(model)
    carrier = cast(Any, carrier_artifact).carrier
    if not isinstance(carrier, Rank64FisherCarrier):
        raise TypeError("carrier sidecar did not expose Rank64FisherCarrier")
    if carrier.hidden_width != 640 or carrier.rank != 64:
        raise ValueError("production carrier must be rank 64 at hidden width 640")
    return RuntimeContext(
        corpus=corpus,
        tokenizer=tokenizer,
        tokenizer_contract=tokenizer_contract,
        adapter=adapter,
        carrier_artifact=carrier_artifact,
        split_manifest=split_manifest,
    )


def _materialize_panel(
    runtime: RuntimeContext,
    role_input: Gemma3L3L4ProgressiveARolePrompts,
) -> GemmaProgressivePanel:
    role = role_input.role
    return materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=runtime.tokenizer,
        role_input=role_input,
        view=runtime.corpus.preclaim_view(role),
        max_length=int(runtime.tokenizer_contract["max_length"]),
        device=torch.device(str(runtime.tokenizer_contract["device"])),
        forbidden_manifest_sha256s=(
            runtime.corpus.artifact.forbidden_assessment_manifest_sha256s
        ),
    )


def _diagnostics(capture: Layer4SequenceCapture, *, scheme: str) -> dict[str, object]:
    return {
        "scheme": scheme,
        "sequence_ids": list(capture.sequence_ids),
        "absolute_tolerance": capture.absolute_tolerance,
        "relative_tolerance": capture.relative_tolerance,
        "maximum_future_attention_mass": capture.attention_summary.maximum_future_mass,
        "maximum_invalid_key_attention_mass": capture.attention_summary.maximum_invalid_key_mass,
        "identities": {
            row.name: {
                "maximum_absolute_error": row.maximum_absolute_error,
                "relative_l2_error": row.relative_l2_error,
                "reference_maximum_absolute_value": row.reference_maximum_absolute_value,
            }
            for row in capture.diagnostics.all
        },
    }


def _capture_panel(
    runtime: RuntimeContext,
    panel: GemmaProgressivePanel,
    *,
    config: ProbeConfig,
) -> RoleRows:
    started = time.perf_counter()
    sequence_ids: list[str] = []
    family_ids: list[str] = []
    positions: list[Tensor] = []
    values: dict[str, list[Tensor]] = {
        name: [] for name in ("h", "h_A", "h_plus", "v_A", "v_M")
    }
    attention_components: list[Tensor] = []
    attention_context: list[Tensor] = []
    mlp_components: dict[str, list[Tensor]] = {
        scheme: [] for scheme in PARTITION_SCHEMES
    }
    receipts: list[Mapping[str, object]] = []
    teacher_forwards = 0
    for example in panel.examples:
        example.validate_integrity()
        base_capture: Layer4SequenceCapture | None = None
        for scheme in PARTITION_SCHEMES:
            capture = capture_layer4_sequence(
                runtime.adapter,
                example.batch.model_inputs,
                sequence_ids=(example.example_id,),
                mlp_partition_scheme=cast(Any, scheme),
                mlp_partition_seed=config.partition_seed,
                absolute_tolerance=1.0e-4,
                relative_tolerance=1.0e-5,
            )
            teacher_forwards += 1
            receipts.append(_diagnostics(capture, scheme=scheme))
            mask = capture.query_valid_mask[0]
            mlp_components[scheme].append(capture.mlp_block_v[0, mask].float())
            if base_capture is None:
                base_capture = capture
                row_count = int(mask.sum().item())
                sequence_ids.extend((example.example_id,) * row_count)
                family_ids.extend((example.family_id,) * row_count)
                positions.append(capture.logical_positions[0, mask])
                for name in values:
                    values[name].append(getattr(capture, name)[0, mask].float())
                attention_components.append(capture.attention_head_v[0, mask].float())
                # Capture layout is [batch, head, query, feature].
                attention_context.append(
                    capture.attention_summary.features[0, :, mask, :]
                    .permute(1, 0, 2)
                    .float()
                )
            else:
                if not torch.equal(mask, base_capture.query_valid_mask[0]):
                    raise RuntimeError("partition captures changed token membership")
                for name in ("h", "h_A", "h_plus", "v_A", "v_M"):
                    if not torch.equal(getattr(capture, name), getattr(base_capture, name)):
                        raise RuntimeError("partition capture changed native stage tensors")
                if not torch.equal(
                    capture.attention_head_v, base_capture.attention_head_v
                ):
                    raise RuntimeError("partition capture changed attention components")
    role_rows = RoleRows(
        role=panel.role,
        sequence_ids=tuple(sequence_ids),
        family_ids=tuple(family_ids),
        logical_positions=torch.cat(positions).long(),
        h=torch.cat(values["h"]),
        h_A=torch.cat(values["h_A"]),
        h_plus=torch.cat(values["h_plus"]),
        v_A=torch.cat(values["v_A"]),
        v_M=torch.cat(values["v_M"]),
        attention_components=torch.cat(attention_components),
        attention_context=torch.cat(attention_context),
        mlp_components={
            scheme: torch.cat(chunks) for scheme, chunks in mlp_components.items()
        },
        capture_diagnostics=tuple(receipts),
        teacher_forward_count=teacher_forwards,
        capture_seconds=time.perf_counter() - started,
    )
    role_rows.assert_valid()
    role_alias = {
        "calibration_a_fit": "A_fit",
        "calibration_a_selection": "A_selection",
        "calibration_a_guard": "A_guard",
    }[panel.role]
    cast(Any, runtime.split_manifest).assert_row_provenance(
        role_alias, role_rows.sequence_ids
    )
    expected_families = runtime.corpus.preclaim_view(panel.role).family_by_example
    if any(
        expected_families.get(sequence_id) != family_id
        for sequence_id, family_id in zip(
            role_rows.sequence_ids, role_rows.family_ids, strict=True
        )
    ):
        raise ValueError("expanded token rows escaped their sequence-family binding")
    return role_rows


def _fixed_geometry_sha256(
    *,
    routing_centers: Tensor,
    assignments: Tensor,
    chart_origins: Tensor,
    seed: int,
) -> str:
    return _sha256_json(
        {
            "routing_centers": _tensor_sha256(routing_centers),
            "assignments": _tensor_sha256(assignments),
            "chart_origins": _tensor_sha256(chart_origins),
            "seed": seed,
        },
        domain="fisher-local-chart-fixed-geometry-v2",
    )


def _shared_assignment_hash(baselines: Mapping[str, ChartBaselineSet]) -> str:
    if set(baselines) != set(PARTITION_SCHEMES):
        raise ValueError("fixed geometry requires both partition schemes")
    first = baselines[PARTITION_SCHEMES[0]].neighborhoods
    first_origins = torch.stack(
        [
            chart.basis.center
            for chart in baselines[PARTITION_SCHEMES[0]].atlases[BASELINE_IDS[0]].charts
        ]
    )
    for scheme in PARTITION_SCHEMES:
        item = baselines[scheme]
        if not torch.equal(item.neighborhoods.centers, first.centers):
            raise ValueError("partition controls changed fixed routing centers")
        if not torch.equal(item.neighborhoods.assignments, first.assignments):
            raise ValueError("partition controls changed fixed neighborhood assignments")
        if item.neighborhoods.seed != first.seed:
            raise ValueError("partition controls changed fixed neighborhood seed")
        for arm in BASELINE_IDS:
            origins = torch.stack(
                [chart.basis.center for chart in item.atlases[arm].charts]
            )
            if not torch.equal(origins, first_origins):
                raise ValueError("controls changed fixed local chart origins")
    return _fixed_geometry_sha256(
        routing_centers=first.centers,
        assignments=first.assignments,
        chart_origins=first_origins,
        seed=first.seed,
    )


def _fit_baselines(
    fit: RoleRows,
    carrier: Rank64FisherCarrier,
    *,
    config: ProbeConfig,
) -> dict[str, ChartBaselineSet]:
    state_rows = fit.state_cloud
    neighborhood_seed_states = (fit.h + fit.h_A + fit.h_plus) / 3.0
    zero_flow = torch.zeros_like(fit.v_A)
    # Rows follow state_cloud order: h, h_A, h_plus.  Attach each flow only
    # to its actual source stage.  The complete flow is an auxiliary h-sourced
    # total; h_plus is a terminal state and contributes no outgoing flow.
    total_rows = torch.cat(
        (
            torch.stack((fit.v_A, fit.v_total, zero_flow), dim=1),
            torch.stack((fit.v_M, zero_flow, zero_flow), dim=1),
            torch.zeros(
                (fit.row_count, 3, fit.h.shape[1]), dtype=fit.h.dtype
            ),
        ),
        dim=0,
    )
    result: dict[str, ChartBaselineSet] = {}
    for scheme in PARTITION_SCHEMES:
        zero_components = torch.zeros_like(fit.attention_components)
        components = torch.cat(
            (
                torch.cat((fit.attention_components, zero_components), dim=1),
                torch.cat((zero_components, fit.mlp_components[scheme]), dim=1),
                torch.zeros(
                    (fit.row_count, 8, fit.h.shape[1]), dtype=fit.h.dtype
                ),
            ),
            dim=0,
        )
        result[scheme] = fit_equal_capacity_chart_baselines(
            states=state_rows,
            total_flows=total_rows,
            component_flows=components,
            carrier=carrier,
            chart_count=config.chart_count,
            coordinate_rank=config.coordinate_rank,
            fisher_regularization=float(carrier.eigenvalues.max()) * 1.0e-8,
            seed=config.chart_seed,
            neighborhood_seed_states=neighborhood_seed_states,
            feature_schema=FeatureSchema(
                (
                    FeatureField("stage_state", "state"),
                    FeatureField("total_stage_flows", "total_flow"),
                    FeatureField("exact_components", "component_flow"),
                )
            ),
        )
    _shared_assignment_hash(result)
    centers = result[PARTITION_SCHEMES[0]].neighborhoods.centers
    for stage, states in (("h", fit.h), ("h_A", fit.h_A)):
        counts = torch.bincount(
            _nearest_fixed_assignments(centers, states),
            minlength=config.chart_count,
        )
        if bool((counts == 0).any()):
            empty = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"fixed state-cloud neighborhoods leave {stage} source charts empty: {empty}; "
                "choose a smaller preregistered chart_count before freezing"
            )
    return result


def _nearest_fixed_assignments(centers: Tensor, states: Tensor) -> Tensor:
    return torch.cdist(states.double(), centers.double()).square().argmin(dim=1)


def _role_assignments(rows: RoleRows, baseline_set: ChartBaselineSet) -> dict[str, Tensor]:
    centers = baseline_set.neighborhoods.centers
    return {
        "h": _nearest_fixed_assignments(centers, rows.h),
        "h_A": _nearest_fixed_assignments(centers, rows.h_A),
        "h_plus": _nearest_fixed_assignments(centers, rows.h_plus),
    }


def _calibrate_support(
    baselines: Mapping[str, ChartBaselineSet],
    selection: RoleRows,
    *,
    coverage: float,
) -> dict[str, ChartBaselineSet]:
    calibrated: dict[str, ChartBaselineSet] = {}
    for scheme, baseline_set in baselines.items():
        assignments = _role_assignments(selection, baseline_set)
        state_assignments = torch.cat(
            (assignments["h"], assignments["h_A"], assignments["h_plus"])
        )
        atlases = {
            arm: atlas.calibrate_from_selection(
                selection.state_cloud,
                state_assignments,
                coverage=coverage,
            )
            for arm, atlas in baseline_set.atlases.items()
        }
        calibrated[scheme] = ChartBaselineSet(
            neighborhoods=baseline_set.neighborhoods,
            atlases=atlases,
        )
    return calibrated


def _model_state(
    affine: AffineCoordinateGenerator,
    constant: ConstantMeanGenerator,
) -> dict[str, object]:
    return {
        "weight": affine.weight,
        "bias": affine.bias,
        "mean": constant.mean,
        "relative_ridge": affine.relative_ridge,
    }


def _fit_one_generator(
    *,
    chart: LocalChart,
    source_states: Tensor,
    target_flows: Tensor,
    extra_features: Tensor | None,
    ridge: float,
    stage: Literal["attention", "mlp"],
) -> dict[str, object]:
    coordinates = chart.basis.encode(source_states)
    features = (
        coordinates
        if extra_features is None
        else torch.cat((coordinates, extra_features.double()), dim=1)
    )
    fields = [
        FeatureField(f"source_coordinate_{index}", "source_coordinate")
        for index in range(coordinates.shape[1])
    ]
    if extra_features is not None:
        fields.extend(
            FeatureField(name, "attention_summary")
            for name in ATTENTION_SUMMARY_FEATURE_NAMES
        )
    schema = FeatureSchema(tuple(fields))
    validate_generator_feature_schema(schema, stage=stage)
    target_coordinates = chart.basis.encode_displacements(target_flows)
    constant = ConstantMeanGenerator.fit(target_coordinates)
    affine = AffineCoordinateGenerator.fit(
        features,
        target_coordinates,
        relative_ridge=ridge,
        feature_schema=schema,
        stage=stage,
    )
    return _model_state(affine, constant)


def _fit_arm_generators(
    atlas: FrozenAtlas,
    rows: RoleRows,
    assignments: Mapping[str, Tensor],
    *,
    scheme: str,
    ridge: float,
) -> dict[str, object]:
    """Fit one identical affine family for every component and direct total."""

    targets: dict[str, dict[str, object]] = {}

    def fit_target(
        name: str,
        *,
        source: Tensor,
        source_assignments: Tensor,
        target: Tensor,
        stage: Literal["attention", "mlp"],
        extras: Tensor | None,
    ) -> None:
        chart_states: list[dict[str, object]] = []
        for chart_index, chart in enumerate(atlas.charts):
            mask = source_assignments == chart_index
            if not bool(mask.any()):
                raise ValueError(f"chart {chart_index} has no fit rows for {name}")
            chart_states.append(
                _fit_one_generator(
                    chart=chart,
                    source_states=source[mask],
                    target_flows=target[mask],
                    extra_features=(None if extras is None else extras[mask]),
                    ridge=ridge,
                    stage=stage,
                )
            )
        targets[name] = {
            "stage": stage,
            "uses_causal_context": extras is not None,
            "charts": chart_states,
        }

    for head in range(4):
        fit_target(
            f"attention_head_{head}_source",
            source=rows.h,
            source_assignments=assignments["h"],
            target=rows.attention_components[:, head],
            stage="attention",
            extras=None,
        )
        fit_target(
            f"attention_head_{head}_context",
            source=rows.h,
            source_assignments=assignments["h"],
            target=rows.attention_components[:, head],
            stage="attention",
            extras=rows.attention_context[:, head],
        )
    mean_context = rows.attention_context.mean(dim=1)
    fit_target(
        "attention_total_source",
        source=rows.h,
        source_assignments=assignments["h"],
        target=rows.v_A,
        stage="attention",
        extras=None,
    )
    fit_target(
        "attention_total_context",
        source=rows.h,
        source_assignments=assignments["h"],
        target=rows.v_A,
        stage="attention",
        extras=mean_context,
    )
    for block in range(4):
        fit_target(
            f"mlp_block_{block}",
            source=rows.h_A,
            source_assignments=assignments["h_A"],
            target=rows.mlp_components[scheme][:, block],
            stage="mlp",
            extras=None,
        )
    fit_target(
        "mlp_total",
        source=rows.h_A,
        source_assignments=assignments["h_A"],
        target=rows.v_M,
        stage="mlp",
        extras=None,
    )
    fit_target(
        "complete_total",
        source=rows.h,
        source_assignments=assignments["h"],
        target=rows.v_total,
        stage="attention",
        extras=mean_context,
    )
    return {
        "scheme": scheme,
        "ridge": ridge,
        "output_representation": "source_chart_displacement_coordinates",
        "targets": targets,
    }


def _predict_generator(
    atlas: FrozenAtlas,
    model: Mapping[str, object],
    source_states: Tensor,
    assignments: Tensor,
    *,
    extra_features: Tensor | None,
    constant: bool = False,
) -> Tensor:
    chart_models = cast(Sequence[Mapping[str, object]], model["charts"])
    if len(chart_models) != atlas.chart_count:
        raise ValueError("generator chart count differs from atlas")
    result = torch.empty(
        source_states.shape[0], source_states.shape[1], dtype=torch.float64
    )
    for chart_index, chart in enumerate(atlas.charts):
        mask = assignments == chart_index
        if not bool(mask.any()):
            continue
        state = chart_models[chart_index]
        coordinates = chart.basis.encode(source_states[mask])
        features = (
            coordinates
            if extra_features is None
            else torch.cat((coordinates, extra_features[mask].double()), dim=1)
        )
        mean = cast(Tensor, state["mean"]).double()
        if constant:
            predicted_coordinates = mean.unsqueeze(0).expand(features.shape[0], -1)
        else:
            weight = cast(Tensor, state["weight"]).double()
            bias = cast(Tensor, state["bias"]).double()
            predicted_coordinates = features @ weight.T + bias
        result[mask] = chart.basis.decode_displacements(predicted_coordinates)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("generator prediction produced non-finite values")
    return result


def _target_model(models: Mapping[str, object], name: str) -> Mapping[str, object]:
    targets = cast(Mapping[str, object], models["targets"])
    value = targets.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(f"candidate generator is missing target {name}")
    return cast(Mapping[str, object], value)


def _predict_components(
    atlas: FrozenAtlas,
    models: Mapping[str, object],
    rows: RoleRows,
    assignments: Mapping[str, Tensor],
    *,
    scheme: str,
    attention_condition: Literal["source", "context"] = "context",
    constant: bool = False,
) -> dict[str, Tensor]:
    attention_parts = []
    for head in range(4):
        extras = (
            None
            if attention_condition == "source"
            else rows.attention_context[:, head]
        )
        attention_parts.append(
            _predict_generator(
                atlas,
                _target_model(
                    models, f"attention_head_{head}_{attention_condition}"
                ),
                rows.h,
                assignments["h"],
                extra_features=extras,
                constant=constant,
            )
        )
    mlp_parts = [
        _predict_generator(
            atlas,
            _target_model(models, f"mlp_block_{block}"),
            rows.h_A,
            assignments["h_A"],
            extra_features=None,
            constant=constant,
        )
        for block in range(4)
    ]
    attention_stack = torch.stack(attention_parts, dim=1)
    mlp_stack = torch.stack(mlp_parts, dim=1)
    return {
        "attention_components": attention_stack,
        "mlp_components": mlp_stack,
        "attention_sum": attention_stack.sum(dim=1),
        "mlp_sum": mlp_stack.sum(dim=1),
    }


def _predict_direct_totals(
    atlas: FrozenAtlas,
    models: Mapping[str, object],
    rows: RoleRows,
    assignments: Mapping[str, Tensor],
    *,
    constant: bool = False,
) -> dict[str, Tensor]:
    mean_context = rows.attention_context.mean(dim=1)
    return {
        "attention_source": _predict_generator(
            atlas,
            _target_model(models, "attention_total_source"),
            rows.h,
            assignments["h"],
            extra_features=None,
            constant=constant,
        ),
        "attention_context": _predict_generator(
            atlas,
            _target_model(models, "attention_total_context"),
            rows.h,
            assignments["h"],
            extra_features=mean_context,
            constant=constant,
        ),
        "mlp": _predict_generator(
            atlas,
            _target_model(models, "mlp_total"),
            rows.h_A,
            assignments["h_A"],
            extra_features=None,
            constant=constant,
        ),
        "complete": _predict_generator(
            atlas,
            _target_model(models, "complete_total"),
            rows.h,
            assignments["h"],
            extra_features=mean_context,
            constant=constant,
        ),
    }


def _composed_prediction(
    atlas: FrozenAtlas,
    models: Mapping[str, object],
    rows: RoleRows,
    baseline_set: ChartBaselineSet,
) -> dict[str, Tensor]:
    assignments = _role_assignments(rows, baseline_set)
    components = _predict_components(
        atlas, models, rows, assignments, scheme=str(models["scheme"])
    )
    predicted_h_A = rows.h.double() + components["attention_sum"]
    predicted_mlp_assignments = _nearest_fixed_assignments(
        baseline_set.neighborhoods.centers, predicted_h_A
    )
    predicted_mlp_parts = []
    for block in range(4):
        predicted_mlp_parts.append(
            _predict_generator(
                atlas,
                _target_model(models, f"mlp_block_{block}"),
                predicted_h_A,
                predicted_mlp_assignments,
                extra_features=None,
            )
        )
    predicted_mlp = torch.stack(predicted_mlp_parts, dim=1).sum(dim=1)
    return {
        "predicted_h_A": predicted_h_A,
        "teacher_h_plus": rows.h_A.double() + components["mlp_sum"],
        "composed_h_plus": predicted_h_A + predicted_mlp,
        "predicted_mlp_assignments": predicted_mlp_assignments,
        "teacher_mlp_assignments": assignments["h_A"],
        "attention_components": components["attention_components"],
        "teacher_mlp_components": components["mlp_components"],
    }


def _chart_state(chart: LocalChart) -> dict[str, object]:
    basis = chart.basis
    common: dict[str, object] = {
        "center": basis.center,
        "coordinate_mean": chart.coordinate_mean,
        "coordinate_precision": chart.coordinate_precision,
        "normal_scale_squared": chart.normal_scale_squared,
        "tangential_radius_squared": chart.tangential_radius_squared,
        "tangential_weight": chart.tangential_weight,
    }
    if isinstance(basis, EuclideanChartBasis):
        common.update({"basis_kind": "euclidean", "directions": basis.directions})
    else:
        common.update(
            {
                "basis_kind": "shared_output_fisher_supported",
                "carrier_directions": basis.carrier_directions,
                "fisher_regularization": basis.regularization,
            }
        )
    return common


def _atlas_state(atlas: FrozenAtlas, centers: Tensor) -> dict[str, object]:
    return {
        "baseline": atlas.baseline,
        "fixed_neighborhood_centers": centers,
        "initial_assignments": atlas.initial_assignments,
        "charts": [_chart_state(chart) for chart in atlas.charts],
    }


def _atlas_from_state(
    state: Mapping[str, object], carrier: Rank64FisherCarrier
) -> FrozenAtlas:
    charts = []
    for raw in cast(Sequence[Mapping[str, object]], state["charts"]):
        center = cast(Tensor, raw["center"])
        if raw["basis_kind"] == "euclidean":
            basis = EuclideanChartBasis(center, cast(Tensor, raw["directions"]))
        elif raw["basis_kind"] == "shared_output_fisher_supported":
            basis = LowRankFisherChartBasis(
                center,
                carrier,
                cast(Tensor, raw["carrier_directions"]),
                float(raw["fisher_regularization"]),
            )
        else:
            raise ValueError("unknown serialized chart basis kind")
        charts.append(
            LocalChart(
                basis=basis,
                coordinate_mean=cast(Tensor, raw["coordinate_mean"]),
                coordinate_precision=cast(Tensor, raw["coordinate_precision"]),
                normal_scale_squared=float(raw["normal_scale_squared"]),
                tangential_radius_squared=float(raw["tangential_radius_squared"]),
                tangential_weight=float(raw["tangential_weight"]),
            )
        )
    return FrozenAtlas(
        baseline=str(state["baseline"]),
        charts=tuple(charts),
        initial_assignments=cast(Tensor, state["initial_assignments"]),
    )


def _candidate_atlas(
    bundle: Mapping[str, object],
    carrier: Rank64FisherCarrier,
    *,
    arm: str,
    scheme: str,
) -> tuple[FrozenAtlas, Tensor]:
    baseline = cast(Mapping[str, object], cast(Mapping[str, object], bundle["baselines"])[arm])
    schemes = cast(Mapping[str, object], baseline["schemes"])
    state = cast(Mapping[str, object], schemes[scheme])
    return _atlas_from_state(state, carrier), cast(Tensor, state["fixed_neighborhood_centers"])


def _candidate_shared_geometry(
    bundle: Mapping[str, object],
) -> tuple[Tensor, Tensor, Tensor]:
    """Authenticate fixed gate centers, assignments, and chart origins."""

    baselines = cast(Mapping[str, object], bundle["baselines"])
    selected = cast(Mapping[str, object], bundle["selected_hyperparameters"])
    seed = int(selected["chart_seed"])
    reference_centers: Tensor | None = None
    reference_assignments: Tensor | None = None
    reference_origins: Tensor | None = None
    receipts: set[str] = set()
    for arm in BASELINE_IDS:
        baseline = cast(Mapping[str, object], baselines[arm])
        receipts.add(str(baseline["assignment_sha256"]))
        schemes = cast(Mapping[str, object], baseline["schemes"])
        for scheme in PARTITION_SCHEMES:
            state = cast(Mapping[str, object], schemes[scheme])
            centers = state.get("fixed_neighborhood_centers")
            assignments = state.get("initial_assignments")
            charts = state.get("charts")
            if not isinstance(centers, Tensor) or centers.ndim != 2:
                raise ValueError("candidate fixed routing centers are invalid")
            if not isinstance(assignments, Tensor) or assignments.ndim != 1:
                raise ValueError("candidate fixed assignments are invalid")
            if not isinstance(charts, (tuple, list)) or len(charts) != centers.shape[0]:
                raise ValueError("candidate chart origins do not match routing centers")
            raw_origins = []
            for chart in charts:
                if not isinstance(chart, Mapping) or not isinstance(
                    chart.get("center"), Tensor
                ):
                    raise ValueError("candidate local chart origin is invalid")
                raw_origins.append(cast(Tensor, chart["center"]))
            origins = torch.stack(raw_origins)
            if origins.shape != centers.shape:
                raise ValueError("candidate local chart origins have the wrong shape")
            if reference_centers is None:
                reference_centers = centers
                reference_assignments = assignments
                reference_origins = origins
            elif (
                not torch.equal(centers, reference_centers)
                or not torch.equal(assignments, reference_assignments)
                or not torch.equal(origins, reference_origins)
            ):
                raise ValueError(
                    "candidate controls do not share fixed routing centers, "
                    "assignments, and local chart origins"
                )
    assert reference_centers is not None
    assert reference_assignments is not None
    assert reference_origins is not None
    expected = _fixed_geometry_sha256(
        routing_centers=reference_centers,
        assignments=reference_assignments,
        chart_origins=reference_origins,
        seed=seed,
    )
    if receipts != {expected}:
        raise ValueError("candidate fixed-geometry receipt mismatch")
    return reference_centers, reference_assignments, reference_origins


def _project_by_chart(
    atlas: FrozenAtlas, values: Tensor, assignments: Tensor
) -> Tensor:
    result = torch.empty_like(values, dtype=torch.float64)
    for chart_index, chart in enumerate(atlas.charts):
        mask = assignments == chart_index
        if bool(mask.any()):
            result[mask] = chart.basis.project_displacements(values[mask])
    return result


def _flow_metrics(
    atlas: FrozenAtlas,
    values: Tensor,
    assignments: Tensor,
    carrier: Rank64FisherCarrier,
) -> dict[str, object]:
    projected = _project_by_chart(atlas, values, assignments)
    residual = values.double() - projected
    euclidean_target = float(values.double().square().sum())
    euclidean_residual = float(residual.square().sum())
    fisher_target = float(carrier.metric_norm_squared(values).sum())
    fisher_residual = float(carrier.metric_norm_squared(residual).sum())
    return {
        "rows": int(values.shape[0]),
        "euclidean_target_energy": euclidean_target,
        "euclidean_residual_energy": euclidean_residual,
        "euclidean_captured_fraction": 1.0
        - euclidean_residual / max(euclidean_target, torch.finfo(torch.float64).tiny),
        "fisher_target_energy": fisher_target,
        "fisher_residual_energy": fisher_residual,
        "fisher_captured_fraction": 1.0
        - fisher_residual / max(fisher_target, torch.finfo(torch.float64).tiny),
    }


def _rank_curve(
    atlas: FrozenAtlas,
    values: Tensor,
    assignments: Tensor,
    carrier: Rank64FisherCarrier,
) -> list[dict[str, object]]:
    curve = []
    ranks = tuple(rank for rank in (1, 2, 4, 8) if rank <= atlas.coordinate_rank)
    for rank in ranks:
        projected = torch.empty_like(values, dtype=torch.float64)
        for chart_index, chart in enumerate(atlas.charts):
            mask = assignments == chart_index
            if not bool(mask.any()):
                continue
            basis = chart.basis
            displacement = values[mask].double()
            if isinstance(basis, EuclideanChartBasis):
                directions = basis.directions[:, :rank]
                projected[mask] = (displacement @ directions) @ directions.T
            else:
                directions = basis.carrier_directions[:, :rank]
                roots = carrier.regularized_eigenvalues(basis.regularization).sqrt()
                coordinates = ((displacement @ carrier.q) * roots) @ directions
                ambient = carrier.q @ (roots.reciprocal().unsqueeze(1) * directions)
                projected[mask] = coordinates @ ambient.T
        residual = values.double() - projected
        e_target = float(values.double().square().sum())
        e_residual = float(residual.square().sum())
        f_target = float(carrier.metric_norm_squared(values).sum())
        f_residual = float(carrier.metric_norm_squared(residual).sum())
        curve.append(
            {
                "rank": rank,
                "euclidean_captured_fraction": 1.0 - e_residual / max(e_target, 1e-300),
                "fisher_captured_fraction": 1.0 - f_residual / max(f_target, 1e-300),
            }
        )
    return curve


def _state_reconstruction_metrics(
    atlas: FrozenAtlas,
    states: Tensor,
    assignments: Tensor,
    carrier: Rank64FisherCarrier,
) -> dict[str, float]:
    reconstructed = torch.empty_like(states, dtype=torch.float64)
    for chart_index, chart in enumerate(atlas.charts):
        mask = assignments == chart_index
        if bool(mask.any()):
            reconstructed[mask] = chart.basis.project(states[mask])
    return prediction_metrics(
        reconstructed,
        states,
        evaluation_carrier=carrier,
    )


def _stage_output_metrics(
    prediction: Tensor,
    target: Tensor,
    carrier: Rank64FisherCarrier,
) -> dict[str, float]:
    metrics = prediction_metrics(prediction, target, evaluation_carrier=carrier)
    left = prediction.double()
    right = target.double()
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
    relative_norm = (
        torch.linalg.vector_norm(left - right, dim=1)
        / torch.linalg.vector_norm(right, dim=1).clamp_min(1e-300)
    )
    metrics.update(
        {
            "mean_cosine_similarity": float(cosine.mean()),
            "mean_relative_norm_error": float(relative_norm.mean()),
        }
    )
    return metrics


def _assigned_support(
    atlas: FrozenAtlas, states: Tensor, assignments: Tensor
) -> tuple[Tensor, dict[str, float]]:
    supported = torch.empty(states.shape[0], dtype=torch.bool)
    combined = torch.empty(states.shape[0], dtype=torch.float64)
    tangent = torch.empty(states.shape[0], dtype=torch.float64)
    normal = torch.empty(states.shape[0], dtype=torch.float64)
    for chart_index, chart in enumerate(atlas.charts):
        mask = assignments == chart_index
        if not bool(mask.any()):
            continue
        evaluation = chart.evaluate_support(states[mask])
        supported[mask] = evaluation.supported
        combined[mask] = evaluation.combined_error
        tangent[mask] = evaluation.tangential_error
        normal[mask] = evaluation.normal_error
    all_tangent = []
    for chart in atlas.charts:
        evaluation = chart.evaluate_support(states)
        radius = math.sqrt(chart.tangential_radius_squared)
        all_tangent.append(evaluation.tangential_error.sqrt() - radius)
    nearest_excess = torch.stack(all_tangent, dim=1).min(dim=1).values.clamp_min(0)
    return supported, {
        "out_of_support_rate": float((~supported).double().mean()),
        "mean_normal_space_reconstruction_residual": float(normal.mean()),
        "mean_tangential_mahalanobis": float(tangent.mean()),
        "mean_combined_support_error": float(combined.mean()),
        "mean_distance_to_nearest_supported_chart": float(nearest_excess.mean()),
        "inside_rows": int(supported.sum()),
        "outside_rows": int((~supported).sum()),
    }


def _fit_routers(
    atlas: FrozenAtlas,
    rows: RoleRows,
    assignments: Mapping[str, Tensor],
    *,
    ridge: float,
) -> dict[str, object]:
    chart_count = atlas.chart_count

    def fit_stage(stage: Literal["attention", "mlp"]) -> dict[str, object]:
        if stage == "attention":
            source_states = rows.h
            source_labels = assignments["h"]
            destinations = assignments["h_A"]
            context = rows.attention_context.mean(dim=1).double()
        else:
            source_states = rows.h_A
            source_labels = assignments["h_A"]
            destinations = assignments["h_plus"]
            context = None
        global_frequency, source_frequencies = fit_destination_frequency_baselines(
            source_labels,
            destinations,
            stage=stage,
            chart_count=chart_count,
        )
        models = []
        for source_chart, chart in enumerate(atlas.charts):
            mask = source_labels == source_chart
            if not bool(mask.any()):
                raise ValueError(f"router source chart {source_chart} has no fit rows")
            coordinates = chart.basis.encode(source_states[mask])
            features = (
                coordinates
                if context is None
                else torch.cat((coordinates, context[mask]), dim=1)
            )
            fields = [
                FeatureField(f"source_coordinate_{index}", "source_coordinate")
                for index in range(coordinates.shape[1])
            ]
            if context is not None:
                fields.extend(
                    FeatureField(name, "attention_summary")
                    for name in ATTENTION_SUMMARY_FEATURE_NAMES
                )
            schema = FeatureSchema(tuple(fields))
            validate_router_feature_schema(schema, stage=stage)
            router = AffineDestinationRouter.fit(
                features,
                destinations[mask],
                stage=stage,
                source_chart=source_chart,
                destination_count=chart_count,
                relative_ridge=ridge,
                maximum_iterations=100,
                feature_schema=schema,
            )
            models.append(
                {
                    "weight": router.weight,
                    "bias": router.bias,
                    "relative_ridge": router.relative_ridge,
                    "feature_width": router.feature_width,
                }
            )
        return {
            "stage": stage,
            "models": models,
            "global_frequency": global_frequency.probabilities,
            "source_frequencies": [
                source_frequencies[index].probabilities
                for index in range(chart_count)
            ],
        }

    return {stage: fit_stage(cast(Any, stage)) for stage in ("attention", "mlp")}


def _router_probabilities(
    atlas: FrozenAtlas,
    state: Mapping[str, object],
    rows: RoleRows,
    assignments: Mapping[str, Tensor],
    *,
    stage: Literal["attention", "mlp"],
    baseline: Literal["affine", "source_frequency", "global_frequency"],
) -> Tensor:
    stage_state = cast(Mapping[str, object], state[stage])
    source_states = rows.h if stage == "attention" else rows.h_A
    source_labels = assignments["h"] if stage == "attention" else assignments["h_A"]
    context = rows.attention_context.mean(dim=1).double() if stage == "attention" else None
    global_frequency = cast(Tensor, stage_state["global_frequency"]).double()
    source_frequencies = cast(Sequence[Tensor], stage_state["source_frequencies"])
    model_states = cast(Sequence[Mapping[str, object]], stage_state["models"])
    result = torch.empty(rows.row_count, atlas.chart_count, dtype=torch.float64)
    for source_chart, chart in enumerate(atlas.charts):
        mask = source_labels == source_chart
        if not bool(mask.any()):
            continue
        if baseline == "global_frequency":
            result[mask] = global_frequency
        elif baseline == "source_frequency":
            result[mask] = source_frequencies[source_chart].double()
        else:
            coordinates = chart.basis.encode(source_states[mask])
            features = (
                coordinates
                if context is None
                else torch.cat((coordinates, context[mask]), dim=1)
            )
            model = model_states[source_chart]
            logits = features @ cast(Tensor, model["weight"]).double().T
            logits = logits + cast(Tensor, model["bias"]).double()
            result[mask] = logits.softmax(dim=1)
    return result


def _evaluate_router(
    atlas: FrozenAtlas,
    router_state: Mapping[str, object],
    rows: RoleRows,
    assignments: Mapping[str, Tensor],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for stage in ("attention", "mlp"):
        typed_stage = cast(Literal["attention", "mlp"], stage)
        destinations = assignments["h_A"] if stage == "attention" else assignments["h_plus"]
        conditions = {}
        for condition in ("affine", "source_frequency", "global_frequency"):
            probabilities = _router_probabilities(
                atlas,
                router_state,
                rows,
                assignments,
                stage=typed_stage,
                baseline=cast(Any, condition),
            )
            conditions[condition] = routing_metrics(probabilities, destinations)
        affine = cast(Mapping[str, float], conditions["affine"])
        source_frequency = cast(Mapping[str, float], conditions["source_frequency"])
        conditions["delta_vs_source_frequency"] = {
            "cross_entropy_reduction": source_frequency["cross_entropy"] - affine["cross_entropy"],
            "top1_improvement": affine["top1"] - source_frequency["top1"],
        }
        source_labels = assignments["h"] if stage == "attention" else assignments["h_A"]
        matrix = torch.zeros(atlas.chart_count, atlas.chart_count, dtype=torch.float64)
        for source, destination in zip(source_labels.tolist(), destinations.tolist(), strict=True):
            matrix[source, destination] += 1
        source_counts = matrix.sum(dim=1)
        populated_sources = source_counts > 0
        probabilities = matrix / source_counts.unsqueeze(1).clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(dim=1)
        conditions["transition_structure"] = {
            "supported_edge_count": int((matrix > 0).sum()),
            "possible_edge_count": int(matrix.numel()),
            "sparsity": float((matrix == 0).double().mean()),
            "populated_source_chart_count": int(populated_sources.sum()),
            "empty_source_chart_count": int((~populated_sources).sum()),
            "mean_source_transition_entropy": float(entropy[populated_sources].mean()),
            "counts": matrix,
        }
        result[stage] = conditions
    return result


def _fisher_similarity_matrix(
    rows: RoleRows,
    carrier: Rank64FisherCarrier,
    *,
    scheme: str,
) -> dict[str, object]:
    components = torch.cat(
        (rows.attention_components, rows.mlp_components[scheme]), dim=1
    ).double()
    names = [f"attention_head_{index}" for index in range(4)] + [
        f"mlp_probe_block_{index}" for index in range(4)
    ]
    means = components.mean(dim=0)
    supported = means @ carrier.q
    weighted = supported * carrier.eigenvalues.sqrt()
    norms = torch.linalg.vector_norm(weighted, dim=1).clamp_min(1e-300)
    similarities = (weighted @ weighted.T) / (norms[:, None] * norms[None, :])
    return {"names": names, "matrix": similarities}


def _bootstrap_ratio_delta(
    candidate_residual: Tensor,
    control_residual: Tensor,
    target_energy: Tensor,
    sequence_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
    estimand: str,
) -> dict[str, object]:
    if (
        candidate_residual.shape != control_residual.shape
        or candidate_residual.shape != target_energy.shape
        or candidate_residual.shape[0] != len(sequence_ids)
    ):
        raise ValueError("bootstrap residual and target energies must be row-aligned")
    values = torch.stack(
        (
            candidate_residual.double(),
            control_residual.double(),
            target_energy.double(),
        ),
        dim=-1,
    )

    def relative_error_delta(sample: Tensor) -> Tensor:
        candidate = sample[..., 0].sum(dim=0)
        control = sample[..., 1].sum(dim=0)
        target = sample[..., 2].sum(dim=0).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        # For tangent capture there is one value per component; its reported
        # estimand is the unweighted mean of those component-wise ratios.  For
        # affine prediction these are scalars and mean() is an identity.
        return ((control - candidate) / target).mean()

    estimate = sequence_bootstrap(
        values,
        sequence_ids,
        replicates=replicates,
        confidence=0.95,
        seed=seed,
        statistic=relative_error_delta,
    )
    return {
        "meaning": "positive favors candidate",
        "estimand": estimand,
        "point": float(estimate.point),
        "lower_95": float(estimate.lower),
        "upper_95": float(estimate.upper),
        "replicates": replicates,
        "bootstrap_unit": "sequence",
        "seed": seed,
    }


def _ci_evidence_status(interval: Mapping[str, object]) -> str:
    lower = float(interval["lower_95"])
    upper = float(interval["upper_95"])
    if lower > 0.0:
        return "positive"
    if upper < 0.0:
        return "negative"
    return "inconclusive"


def _evaluate_arm(
    *,
    arm: str,
    scheme: str,
    atlas: FrozenAtlas,
    baseline_set: ChartBaselineSet,
    models: Mapping[str, object],
    routers: Mapping[str, object],
    rows: RoleRows,
    carrier: Rank64FisherCarrier,
) -> dict[str, object]:
    assignments = _role_assignments(rows, baseline_set)
    tangent: dict[str, object] = {}
    for head in range(4):
        tangent[f"attention_head_{head}"] = _flow_metrics(
            atlas, rows.attention_components[:, head], assignments["h"], carrier
        )
    for block in range(4):
        tangent[f"mlp_probe_block_{block}"] = _flow_metrics(
            atlas,
            rows.mlp_components[scheme][:, block],
            assignments["h_A"],
            carrier,
        )
    tangent["total_attention"] = _flow_metrics(
        atlas, rows.v_A, assignments["h"], carrier
    )
    tangent["total_mlp"] = _flow_metrics(
        atlas, rows.v_M, assignments["h_A"], carrier
    )
    tangent["complete_layer"] = _flow_metrics(
        atlas, rows.v_total, assignments["h"], carrier
    )

    state_reconstruction = {
        stage: _state_reconstruction_metrics(
            atlas, getattr(rows, stage), assignments[stage], carrier
        )
        for stage in STAGES
    }
    rank_curves = {
        stage: _rank_curve(
            atlas,
            getattr(rows, stage).double() - torch.stack(
                [atlas.charts[int(index)].basis.center for index in assignments[stage]]
            ),
            assignments[stage],
            carrier,
        )
        for stage in STAGES
    }

    component_predictions: dict[str, Mapping[str, Tensor]] = {}
    for condition in ("source", "context"):
        component_predictions[condition] = _predict_components(
            atlas,
            models,
            rows,
            assignments,
            scheme=scheme,
            attention_condition=cast(Any, condition),
        )
    constant_predictions = _predict_components(
        atlas,
        models,
        rows,
        assignments,
        scheme=scheme,
        attention_condition="context",
        constant=True,
    )
    direct = _predict_direct_totals(atlas, models, rows, assignments)
    direct_constant = _predict_direct_totals(
        atlas, models, rows, assignments, constant=True
    )
    affine: dict[str, object] = {
        "attention_components": {},
        "mlp_components": {},
        "component_sums": {},
        "direct_totals": {},
    }
    for head in range(4):
        target = rows.attention_components[:, head]
        source_prediction = component_predictions["source"]["attention_components"][:, head]
        context_prediction = component_predictions["context"]["attention_components"][:, head]
        constant_prediction = constant_predictions["attention_components"][:, head]
        cast(dict[str, object], affine["attention_components"])[f"head_{head}"] = {
            "constant": prediction_metrics(constant_prediction, target, evaluation_carrier=carrier),
            "source_only": prediction_metrics(source_prediction, target, evaluation_carrier=carrier),
            "causal_context": prediction_metrics(context_prediction, target, evaluation_carrier=carrier),
        }
    for block in range(4):
        target = rows.mlp_components[scheme][:, block]
        prediction = component_predictions["context"]["mlp_components"][:, block]
        constant_prediction = constant_predictions["mlp_components"][:, block]
        cast(dict[str, object], affine["mlp_components"])[f"probe_block_{block}"] = {
            "constant": prediction_metrics(constant_prediction, target, evaluation_carrier=carrier),
            "affine": prediction_metrics(prediction, target, evaluation_carrier=carrier),
        }
    for name, target, prediction, constant_prediction in (
        (
            "attention_source_only",
            rows.v_A,
            component_predictions["source"]["attention_sum"],
            constant_predictions["attention_sum"],
        ),
        (
            "attention_causal_context",
            rows.v_A,
            component_predictions["context"]["attention_sum"],
            constant_predictions["attention_sum"],
        ),
        (
            "mlp_teacher_forced",
            rows.v_M,
            component_predictions["context"]["mlp_sum"],
            constant_predictions["mlp_sum"],
        ),
    ):
        affine_prediction = prediction_metrics(prediction, target, evaluation_carrier=carrier)
        constant_metric = prediction_metrics(
            constant_prediction, target, evaluation_carrier=carrier
        )
        cast(dict[str, object], affine["component_sums"])[name] = {
            "constant": constant_metric,
            "affine": affine_prediction,
            "fisher_improvement_over_constant": (
                constant_metric["fisher_relative_squared_error"]
                - affine_prediction["fisher_relative_squared_error"]
            ),
            "euclidean_improvement_over_constant": (
                constant_metric["euclidean_relative_squared_error"]
                - affine_prediction["euclidean_relative_squared_error"]
            ),
        }
    for name, target, prediction, constant_prediction in (
        ("attention_source_only", rows.v_A, direct["attention_source"], direct_constant["attention_source"]),
        ("attention_causal_context", rows.v_A, direct["attention_context"], direct_constant["attention_context"]),
        ("mlp_teacher_forced", rows.v_M, direct["mlp"], direct_constant["mlp"]),
        ("complete_layer", rows.v_total, direct["complete"], direct_constant["complete"]),
    ):
        affine_prediction = prediction_metrics(prediction, target, evaluation_carrier=carrier)
        constant_metric = prediction_metrics(constant_prediction, target, evaluation_carrier=carrier)
        cast(dict[str, object], affine["direct_totals"])[name] = {
            "constant": constant_metric,
            "affine": affine_prediction,
            "fisher_improvement_over_constant": (
                constant_metric["fisher_relative_squared_error"]
                - affine_prediction["fisher_relative_squared_error"]
            ),
        }

    composed = _composed_prediction(atlas, models, rows, baseline_set)
    stage_outputs = {
        "attention": _stage_output_metrics(
            composed["predicted_h_A"], rows.h_A, carrier
        ),
        "mlp_teacher_forced": _stage_output_metrics(
            composed["teacher_h_plus"], rows.h_plus, carrier
        ),
        "composed": _stage_output_metrics(
            composed["composed_h_plus"], rows.h_plus, carrier
        ),
    }
    support: dict[str, object] = {}
    support_vectors: dict[str, Tensor] = {}
    for stage in STAGES:
        vector, summary = _assigned_support(
            atlas, getattr(rows, stage), assignments[stage]
        )
        support_vectors[stage] = vector
        support[stage] = summary
    predicted_support, predicted_summary = _assigned_support(
        atlas,
        composed["predicted_h_A"],
        composed["predicted_mlp_assignments"].long(),
    )
    support["predicted_h_A"] = predicted_summary
    support["composed_support_drift"] = support_drift_metrics(
        support_vectors["h_A"], predicted_support
    )
    # Separate predictive performance by the source chart's genuine support.
    for label, prediction, target, source_support in (
        ("attention", composed["predicted_h_A"], rows.h_A, support_vectors["h"]),
        ("mlp_teacher", composed["teacher_h_plus"], rows.h_plus, support_vectors["h_A"]),
        ("composed", composed["composed_h_plus"], rows.h_plus, predicted_support),
    ):
        support[f"{label}_inside_performance"] = (
            _stage_output_metrics(prediction[source_support], target[source_support], carrier)
            if bool(source_support.any())
            else {"rows": 0}
        )
        support[f"{label}_outside_performance"] = (
            _stage_output_metrics(prediction[~source_support], target[~source_support], carrier)
            if bool((~source_support).any())
            else {"rows": 0}
        )
    router = _evaluate_router(atlas, routers, rows, assignments)
    attention_transition_supported = support_vectors["h"] & support_vectors["h_A"]
    mlp_transition_supported = support_vectors["h_A"] & support_vectors["h_plus"]
    for stage, supported_rows, source_labels, destination_labels in (
        (
            "attention",
            attention_transition_supported,
            assignments["h"],
            assignments["h_A"],
        ),
        (
            "mlp",
            mlp_transition_supported,
            assignments["h_A"],
            assignments["h_plus"],
        ),
    ):
        stage_router = cast(dict[str, object], router[stage])
        stage_router["out_of_support_transition_rate"] = float(
            (~supported_rows).double().mean()
        )
        structure = cast(dict[str, object], stage_router["transition_structure"])
        raw_counts = cast(Tensor, structure["counts"])
        supported_counts = torch.zeros_like(raw_counts)
        for source, destination in zip(
            source_labels[supported_rows].tolist(),
            destination_labels[supported_rows].tolist(),
            strict=True,
        ):
            supported_counts[source, destination] += 1
        structure["observed_edge_count"] = structure["supported_edge_count"]
        structure["supported_edge_count"] = int((supported_counts > 0).sum())
        structure["supported_transition_count"] = int(supported_rows.sum())
        structure["supported_counts"] = supported_counts
    return {
        "arm": arm,
        "scheme": scheme,
        "rows": rows.row_count,
        "state_reconstruction": state_reconstruction,
        "rank_curves": rank_curves,
        "tangent_capture": tangent,
        "affine_generators": affine,
        "stage_outputs": stage_outputs,
        "support": support,
        "router": router,
    }


def _fit_generator_grid(
    baselines: Mapping[str, ChartBaselineSet],
    fit: RoleRows,
    selection: RoleRows,
    carrier: Rank64FisherCarrier,
    *,
    config: ProbeConfig,
) -> tuple[float, dict[str, dict[str, Mapping[str, object]]], dict[str, object]]:
    by_ridge: dict[float, dict[str, dict[str, Mapping[str, object]]]] = {}
    transcript: dict[str, object] = {}
    for ridge in config.ridge_grid:
        scheme_models: dict[str, dict[str, Mapping[str, object]]] = {}
        scores = []
        per_arm: dict[str, float] = {}
        for scheme, baseline_set in baselines.items():
            fit_assignments = _role_assignments(fit, baseline_set)
            selection_assignments = _role_assignments(selection, baseline_set)
            arm_models: dict[str, Mapping[str, object]] = {}
            for arm, atlas in baseline_set.atlases.items():
                models = _fit_arm_generators(
                    atlas,
                    fit,
                    fit_assignments,
                    scheme=scheme,
                    ridge=ridge,
                )
                arm_models[arm] = models
                predicted = _predict_components(
                    atlas,
                    models,
                    selection,
                    selection_assignments,
                    scheme=scheme,
                )
                attention_error = carrier.metric_norm_squared(
                    predicted["attention_sum"] - selection.v_A
                ).mean()
                mlp_error = carrier.metric_norm_squared(
                    predicted["mlp_sum"] - selection.v_M
                ).mean()
                normalizer = (
                    carrier.metric_norm_squared(selection.v_A).mean()
                    + carrier.metric_norm_squared(selection.v_M).mean()
                ).clamp_min(1e-300)
                score = float((attention_error + mlp_error) / normalizer)
                per_arm[f"{scheme}.{arm}"] = score
                scores.append(score)
            scheme_models[scheme] = arm_models
        aggregate = math.fsum(scores) / len(scores)
        transcript[f"{ridge:.12g}"] = {
            "aggregate_normalized_shared_fisher_error": aggregate,
            "per_partition_arm": per_arm,
        }
        by_ridge[ridge] = scheme_models
    selected = min(config.ridge_grid, key=lambda value: cast(Mapping[str, float], transcript[f"{value:.12g}"])["aggregate_normalized_shared_fisher_error"])
    return selected, by_ridge[selected], {
        "criterion": "minimum mean A-selection normalized shared-Fisher attention-plus-MLP component-sum error across every arm and partition",
        "scores": transcript,
        "selected_ridge": selected,
    }


def _select_support_coverage(
    baselines: Mapping[str, ChartBaselineSet],
    models: Mapping[str, Mapping[str, Mapping[str, object]]],
    selection: RoleRows,
    carrier: Rank64FisherCarrier,
    *,
    config: ProbeConfig,
) -> tuple[float, dict[str, ChartBaselineSet], dict[str, object]]:
    if len(config.support_coverages) != 1:  # Defensive for untyped callers.
        raise ValueError("support coverage must be preregistered as one value")
    coverage = config.support_coverages[0]
    calibrated = _calibrate_support(baselines, selection, coverage=coverage)
    diagnostics: dict[str, object] = {}
    for scheme, baseline_set in calibrated.items():
        atlas = baseline_set.atlases["B4"]
        composed = _composed_prediction(
            atlas, models[scheme]["B4"], selection, baseline_set
        )
        error = prediction_metrics(
            composed["composed_h_plus"],
            selection.h_plus,
            evaluation_carrier=carrier,
        )["fisher_relative_squared_error"]
        assignments = _role_assignments(selection, baseline_set)
        supported, _summary = _assigned_support(
            atlas, selection.h_A, assignments["h_A"]
        )
        diagnostics[scheme] = {
            "composed_fisher_relative_squared_error": float(error),
            "intermediate_out_of_support_rate": float((~supported).double().mean()),
        }
    return coverage, calibrated, {
        "criterion": "single support coverage preregistered before A-selection; A-selection calibrates radii and supplies diagnostics but does not select coverage",
        "preregistered": True,
        "diagnostics": diagnostics,
        "selected_coverage": coverage,
    }


def _protocol_sha256(
    config: ProbeConfig,
    *,
    split_manifest_sha256: str,
    carrier_sha256: str,
) -> str:
    if _protocol_source_sha256s() != _IMPORTED_PROTOCOL_SOURCE_SHA256S:
        raise RuntimeError(
            "protocol source files changed after this interpreter imported them"
        )
    payload = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config": config.to_dict(),
        "split_manifest_sha256": split_manifest_sha256,
        "carrier_sha256": carrier_sha256,
        "model_id": DEFAULT_MODEL_ID,
        "resolved_commit": DEFAULT_REVISION,
        "source_sha256s": dict(_IMPORTED_PROTOCOL_SOURCE_SHA256S),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": importlib_metadata.version("transformers"),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "scientific_contract": {
            "state_cloud": ["h", "h_A", "h_plus"],
            "attention_source": "h",
            "mlp_source": "h_A",
            "metric": "shared_layer4_output_empirical_fisher_rank64",
            "exact_earlier_local_fisher_claim": False,
            "calibration_b_prohibited": True,
            "guard_open_before_freeze": False,
            "default_executor_modified": False,
            "recurrent_generator_sharing": False,
        },
    }
    return _sha256_json(payload, domain="fisher-local-chart-protocol-v1")


def _capacity_tensor(
    state: Mapping[str, object], key: str, *, ndim: int, label: str
) -> Tensor:
    value = state.get(key)
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{label} {key} must be a rank-{ndim} tensor")
    return value


def _capacity_signature(
    generator_state: Mapping[str, object],
    router_state: Mapping[str, object],
) -> dict[str, object]:
    """Derive downstream capacity from the tensors that will be serialized."""

    targets = generator_state.get("targets")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("generator capacity requires nonempty targets")
    generator_layout: dict[str, object] = {}
    generator_weight_elements = 0
    generator_bias_elements = 0
    constant_mean_elements = 0
    for target_name, raw_target in sorted(targets.items()):
        if not isinstance(target_name, str) or not isinstance(raw_target, Mapping):
            raise ValueError("generator target capacity entry is invalid")
        chart_models = raw_target.get("charts")
        if not isinstance(chart_models, (tuple, list)) or not chart_models:
            raise ValueError(f"generator target {target_name} has no chart models")
        feature_widths: list[int] = []
        output_widths: list[int] = []
        for raw_model in chart_models:
            if not isinstance(raw_model, Mapping):
                raise ValueError(f"generator target {target_name} model is invalid")
            weight = _capacity_tensor(
                raw_model, "weight", ndim=2, label=f"generator target {target_name}"
            )
            bias = _capacity_tensor(
                raw_model, "bias", ndim=1, label=f"generator target {target_name}"
            )
            mean = _capacity_tensor(
                raw_model, "mean", ndim=1, label=f"generator target {target_name}"
            )
            if bias.shape != (weight.shape[0],) or mean.shape != (weight.shape[0],):
                raise ValueError(
                    f"generator target {target_name} output, bias, and constant mean differ"
                )
            feature_widths.append(int(weight.shape[1]))
            output_widths.append(int(weight.shape[0]))
            generator_weight_elements += weight.numel()
            generator_bias_elements += bias.numel()
            constant_mean_elements += mean.numel()
        generator_layout[target_name] = {
            "chart_model_count": len(chart_models),
            "feature_widths": feature_widths,
            "output_widths": output_widths,
        }

    router_layout: dict[str, object] = {}
    router_weight_elements = 0
    router_bias_elements = 0
    global_frequency_elements = 0
    source_frequency_elements = 0
    if set(router_state) != {"attention", "mlp"}:
        raise ValueError("router capacity requires exactly attention and mlp stages")
    for stage in ("attention", "mlp"):
        raw_stage = router_state[stage]
        if not isinstance(raw_stage, Mapping):
            raise ValueError(f"router stage {stage} is invalid")
        models = raw_stage.get("models")
        source_frequencies = raw_stage.get("source_frequencies")
        if not isinstance(models, (tuple, list)) or not models:
            raise ValueError(f"router stage {stage} has no source-chart models")
        if not isinstance(source_frequencies, (tuple, list)):
            raise ValueError(f"router stage {stage} source frequencies are invalid")
        global_frequency = _capacity_tensor(
            raw_stage, "global_frequency", ndim=1, label=f"router stage {stage}"
        )
        chart_count = len(models)
        if global_frequency.shape != (chart_count,) or len(source_frequencies) != chart_count:
            raise ValueError(f"router stage {stage} chart capacities differ")
        feature_widths = []
        destination_widths = []
        for raw_model in models:
            if not isinstance(raw_model, Mapping):
                raise ValueError(f"router stage {stage} model is invalid")
            weight = _capacity_tensor(
                raw_model, "weight", ndim=2, label=f"router stage {stage}"
            )
            bias = _capacity_tensor(
                raw_model, "bias", ndim=1, label=f"router stage {stage}"
            )
            if weight.shape[0] != chart_count or bias.shape != (chart_count,):
                raise ValueError(f"router stage {stage} destination capacity differs")
            feature_widths.append(int(weight.shape[1]))
            destination_widths.append(int(weight.shape[0]))
            router_weight_elements += weight.numel()
            router_bias_elements += bias.numel()
        for probability in source_frequencies:
            if not isinstance(probability, Tensor) or probability.shape != (chart_count,):
                raise ValueError(f"router stage {stage} source frequency capacity differs")
            source_frequency_elements += probability.numel()
        global_frequency_elements += global_frequency.numel()
        router_layout[stage] = {
            "source_chart_model_count": chart_count,
            "feature_widths": feature_widths,
            "destination_widths": destination_widths,
        }

    learned = (
        generator_weight_elements
        + generator_bias_elements
        + constant_mean_elements
        + router_weight_elements
        + router_bias_elements
        + global_frequency_elements
        + source_frequency_elements
    )
    return {
        "generator_target_layout": generator_layout,
        "generator_affine_weight_elements": generator_weight_elements,
        "generator_affine_bias_elements": generator_bias_elements,
        "generator_constant_mean_elements": constant_mean_elements,
        "router_stage_layout": router_layout,
        "router_affine_weight_elements": router_weight_elements,
        "router_affine_bias_elements": router_bias_elements,
        "router_global_frequency_elements": global_frequency_elements,
        "router_source_frequency_elements": source_frequency_elements,
        "total_serialized_model_elements": learned,
    }


def _equal_capacity_signature(
    generators: Mapping[str, Mapping[str, Mapping[str, object]]],
    routers: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    signatures = {
        f"{scheme}.{arm}": _capacity_signature(
            generators[scheme][arm], routers[scheme][arm]
        )
        for scheme in PARTITION_SCHEMES
        for arm in BASELINE_IDS
    }
    encoded = {_canonical_json_bytes(value) for value in signatures.values()}
    if len(encoded) != 1:
        raise ValueError("B0-B4 and partition controls do not have equal actual capacity")
    return next(iter(signatures.values()))


def _bundle_from_fit(
    *,
    baselines: Mapping[str, ChartBaselineSet],
    generators: Mapping[str, Mapping[str, Mapping[str, object]]],
    routers: Mapping[str, Mapping[str, Mapping[str, object]]],
    evaluations: Mapping[str, object],
    ridge_transcript: Mapping[str, object],
    support_transcript: Mapping[str, object],
    selected_ridge: float,
    selected_coverage: float,
    selected_scheme: str,
    attention_affine_control_arm: str,
    mlp_affine_control_arm: str,
    assignment_sha256: str,
    protocol_sha256: str,
    split_manifest_sha256: str,
    carrier_sha256: str,
    corpus_artifact_sha256: str,
    guard_manifest_sha256: str,
    model_config_sha256: str,
    source_model_sha256: str,
    source_execution_sha256: str,
    config: ProbeConfig,
) -> dict[str, object]:
    capacity_signature = _equal_capacity_signature(generators, routers)
    baseline_payload: dict[str, object] = {}
    for arm in BASELINE_IDS:
        baseline_payload[arm] = {
            "assignment_sha256": assignment_sha256,
            "capacity_signature": capacity_signature,
            "schemes": {
                scheme: _atlas_state(
                    baselines[scheme].atlases[arm],
                    baselines[scheme].neighborhoods.centers,
                )
                for scheme in PARTITION_SCHEMES
            },
        }
    support = {
        scheme: {
            arm: {
                "coverage": selected_coverage,
                "normal_scale_squared": [
                    chart.normal_scale_squared
                    for chart in baselines[scheme].atlases[arm].charts
                ],
                "tangential_radius_squared": [
                    chart.tangential_radius_squared
                    for chart in baselines[scheme].atlases[arm].charts
                ],
            }
            for arm in BASELINE_IDS
        }
        for scheme in PARTITION_SCHEMES
    }
    return {
        "baselines": baseline_payload,
        "selected_hyperparameters": {
            "selected_baseline": "B4",
            "selected_partition_scheme": selected_scheme,
            "attention_affine_control_arm": attention_affine_control_arm,
            "mlp_affine_control_arm": mlp_affine_control_arm,
            "ridge": selected_ridge,
            "support_coverage": selected_coverage,
            "chart_count": config.chart_count,
            "coordinate_rank": config.coordinate_rank,
            "partition_seed": config.partition_seed,
            "chart_seed": config.chart_seed,
            "neighborhood_seed": "A_fit per-token centroid of h, h_A, and h_plus",
        },
        "generators": {
            scheme: dict(generators[scheme]) for scheme in PARTITION_SCHEMES
        },
        "routers": {
            scheme: dict(routers[scheme]) for scheme in PARTITION_SCHEMES
        },
        "support": support,
        "selection_transcript": {
            "ridge": dict(ridge_transcript),
            "support": dict(support_transcript),
            "evaluations": dict(evaluations),
        },
        "bindings": {
            "protocol_sha256": protocol_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "carrier_sha256": carrier_sha256,
            "corpus_artifact_sha256": corpus_artifact_sha256,
            "guard_manifest_sha256": guard_manifest_sha256,
            "model_config_sha256": model_config_sha256,
            "source_model_sha256": source_model_sha256,
            "source_execution_sha256": source_execution_sha256,
        },
        "report_schema": REPORT_SCHEMA,
    }


def _selected_candidate_parts(
    candidate: FisherLocalChartFrozenCandidate,
    carrier: Rank64FisherCarrier,
) -> tuple[str, str, FrozenAtlas, Tensor, Mapping[str, object], Mapping[str, object]]:
    bundle = candidate.bundle
    _candidate_shared_geometry(bundle)
    selected = cast(Mapping[str, object], bundle["selected_hyperparameters"])
    arm = str(selected["selected_baseline"])
    scheme = str(selected["selected_partition_scheme"])
    atlas, centers = _candidate_atlas(bundle, carrier, arm=arm, scheme=scheme)
    generators = cast(
        Mapping[str, object],
        cast(Mapping[str, object], cast(Mapping[str, object], bundle["generators"])[scheme])[arm],
    )
    routers = cast(
        Mapping[str, object],
        cast(Mapping[str, object], cast(Mapping[str, object], bundle["routers"])[scheme])[arm],
    )
    return arm, scheme, atlas, centers, generators, routers


def _baseline_set_for_candidate(
    candidate: FisherLocalChartFrozenCandidate,
    carrier: Rank64FisherCarrier,
    *,
    scheme: str,
) -> ChartBaselineSet:
    shared_centers, shared_assignments, _shared_origins = _candidate_shared_geometry(
        candidate.bundle
    )
    atlases: dict[str, FrozenAtlas] = {}
    centers: Tensor | None = None
    for arm in BASELINE_IDS:
        atlas, arm_centers = _candidate_atlas(
            candidate.bundle, carrier, arm=arm, scheme=scheme
        )
        atlases[arm] = atlas
        if centers is None:
            centers = arm_centers
        elif not torch.equal(centers, arm_centers):
            raise ValueError("candidate baseline centers differ")
    assert centers is not None
    if not torch.equal(centers, shared_centers):
        raise ValueError("candidate scheme differs from shared fixed routing centers")
    from .fisher_local_chart import NeighborhoodAssignments

    neighborhoods = NeighborhoodAssignments(
        centers=shared_centers,
        assignments=shared_assignments,
        seed=int(cast(Mapping[str, object], candidate.bundle["selected_hyperparameters"])["chart_seed"]),
        iterations=1,
        inertia=0.0,
    )
    return ChartBaselineSet(neighborhoods=neighborhoods, atlases=atlases)


def _evaluate_candidate_role(
    candidate: FisherLocalChartFrozenCandidate,
    carrier: Rank64FisherCarrier,
    rows: RoleRows,
) -> dict[str, object]:
    evaluations: dict[str, object] = {}
    generator_root = cast(Mapping[str, object], candidate.bundle["generators"])
    router_root = cast(Mapping[str, object], candidate.bundle["routers"])
    for scheme in PARTITION_SCHEMES:
        baseline_set = _baseline_set_for_candidate(
            candidate, carrier, scheme=scheme
        )
        scheme_generators = cast(Mapping[str, object], generator_root[scheme])
        scheme_routers = cast(Mapping[str, object], router_root[scheme])
        evaluations[scheme] = {
            arm: _evaluate_arm(
                arm=arm,
                scheme=scheme,
                atlas=baseline_set.atlases[arm],
                baseline_set=baseline_set,
                models=cast(Mapping[str, object], scheme_generators[arm]),
                routers=cast(Mapping[str, object], scheme_routers[arm]),
                rows=rows,
                carrier=carrier,
            )
            for arm in BASELINE_IDS
        }
    return evaluations


def _subset_role_rows(rows: RoleRows, sequence_id: str) -> RoleRows:
    mask = torch.tensor(
        [identity == sequence_id for identity in rows.sequence_ids],
        dtype=torch.bool,
    )
    if not bool(mask.any()):
        raise KeyError(f"capture rows omit sequence {sequence_id}")
    return RoleRows(
        role=rows.role,
        sequence_ids=tuple(
            identity for identity, keep in zip(rows.sequence_ids, mask.tolist(), strict=True) if keep
        ),
        family_ids=tuple(
            identity for identity, keep in zip(rows.family_ids, mask.tolist(), strict=True) if keep
        ),
        logical_positions=rows.logical_positions[mask],
        h=rows.h[mask],
        h_A=rows.h_A[mask],
        h_plus=rows.h_plus[mask],
        v_A=rows.v_A[mask],
        v_M=rows.v_M[mask],
        attention_components=rows.attention_components[mask],
        attention_context=rows.attention_context[mask],
        mlp_components={scheme: value[mask] for scheme, value in rows.mlp_components.items()},
        capture_diagnostics=(),
        teacher_forward_count=0,
        capture_seconds=0.0,
    )


def _candidate_model_free_dry_run(
    candidate: FisherLocalChartFrozenCandidate,
    carrier: Rank64FisherCarrier,
    selection: RoleRows,
) -> dict[str, object]:
    """Re-evaluate and authenticate the complete frozen A-selection grid.

    A selected-arm-only pass would leave the other equal-capacity controls
    unauthenticated.  The one-shot claim therefore depends on reproducing all
    ten scheme/arm cells exactly from the reloaded tensor bundle.
    """

    result = _evaluate_candidate_role(candidate, carrier, selection)
    if set(result) != set(PARTITION_SCHEMES) or any(
        not isinstance(result[scheme], Mapping)
        or set(cast(Mapping[str, object], result[scheme])) != set(BASELINE_IDS)
        for scheme in PARTITION_SCHEMES
    ):
        raise ValueError("preclaim A-selection dry run omitted a scheme/baseline cell")
    transcript = candidate.bundle.get("selection_transcript")
    if not isinstance(transcript, Mapping):
        raise ValueError("frozen candidate omits its selection transcript")
    frozen_evaluations = transcript.get("evaluations")
    if not isinstance(frozen_evaluations, Mapping):
        raise ValueError("frozen candidate omits its evaluation transcript")
    frozen_roles = frozen_evaluations.get("roles")
    if not isinstance(frozen_roles, Mapping):
        raise ValueError("frozen candidate omits its role evaluations")
    frozen_selection = frozen_roles.get("A_selection")
    if not isinstance(frozen_selection, Mapping):
        raise ValueError("frozen candidate omits A-selection evaluations")
    # Canonical JSON provides an exact comparison across torch.save's
    # list/tuple normalization while also rejecting NaN and infinity.
    rendered = _canonical_json_bytes(_jsonable(result))
    expected = _canonical_json_bytes(_jsonable(frozen_selection))
    if rendered != expected:
        raise ValueError(
            "recomputed A-selection evaluations differ from the frozen transcript"
        )
    return result


def _full_sequence_fidelity(
    runtime: RuntimeContext,
    panel: GemmaProgressivePanel,
    guard_rows: RoleRows,
    candidate: FisherLocalChartFrozenCandidate,
) -> tuple[dict[str, object], int, float]:
    started = time.perf_counter()
    carrier = cast(Any, runtime.carrier_artifact).carrier
    _arm, scheme, atlas, _centers, generators, _routers = _selected_candidate_parts(
        candidate, carrier
    )
    baseline_set = _baseline_set_for_candidate(candidate, carrier, scheme=scheme)
    accumulator = SourceAuthoritativeShadowFidelityAccumulator(
        {example.example_id: example.family_id for example in panel.examples}
    )
    forward_count = 0
    for example in panel.examples:
        captured = _subset_role_rows(guard_rows, example.example_id)
        composed = _composed_prediction(
            atlas, generators, captured, baseline_set
        )["composed_h_plus"]
        with torch.inference_mode():
            source = runtime.adapter.forward(example.batch.model_inputs)
            forward_count += 1
            valid = source.sequence.query_valid_mask
            if int(valid.sum()) != composed.shape[0]:
                raise RuntimeError("fidelity capture and source valid rows differ")

            def replace_h_plus(native: Tensor) -> Tensor:
                prediction = torch.zeros_like(native)
                prediction[valid] = composed.to(
                    device=native.device, dtype=native.dtype
                )
                # Native values survive only in padding rows, never as an
                # unsupported-chart oracle on a real token.
                return torch.where(valid.unsqueeze(-1), prediction, native)

            shadow = runtime.adapter.forward(
                example.batch.model_inputs,
                interventions={LAYER4_CANONICAL_SITES["h_plus"]: replace_h_plus},
            )
            forward_count += 1
        supervised = example.batch.targets != -100
        accumulator.add(
            {
                "example_id": example.example_id,
                "family_id": example.family_id,
                "source_logits": source.logits[supervised],
                "candidate_logits": shadow.logits[supervised],
                "targets": example.batch.targets[supervised],
            }
        )
    result = accumulator.finalize()
    result["qualification"] = {
        "mode": "opt_in_source_authoritative_full_prompt_shadow",
        "intervention_site": "layer.4.output",
        "default_compiler_or_executor_modified": False,
        "source_outputs_remain_authoritative": True,
        "autoregressive_generation_tested": False,
        "native_fallback_on_valid_unsupported_rows": False,
        "legacy_gate_thresholds_are_diagnostic_not_selected_by_this_probe": True,
    }
    return result, forward_count, time.perf_counter() - started


def _tensor_storage_report(value: object) -> dict[str, int]:
    totals: dict[str, int] = {}

    def visit(item: object) -> None:
        if isinstance(item, Tensor):
            key = str(item.dtype)
            totals[key] = totals.get(key, 0) + item.numel() * item.element_size()
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return totals


def _tensor_parameter_count(value: object) -> int:
    if isinstance(value, Tensor):
        return int(value.numel())
    if isinstance(value, Mapping):
        return sum(_tensor_parameter_count(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_parameter_count(item) for item in value)
    return 0


def _peak_host_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _run_inspect(args: argparse.Namespace) -> dict[str, object]:
    runtime = _runtime_context(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        selection_input_path=args.selection_input,
        guard_input_path=args.guard_input,
        carrier_path=args.carrier,
        cache_dir=args.cache_dir,
    )
    layer_spec = runtime.adapter.layer("layer.4")
    native = runtime.adapter.source_module("layer.4")
    attention = layer_spec.attention
    transformer = layer_spec.transformer
    if attention is None or transformer is None:
        raise RuntimeError("adapter omitted structured layer-4 semantics")
    config = getattr(runtime.adapter.module, "config", None)
    result = {
        "schema": SCHEMA,
        "phase": "inspect",
        "model": {
            "model_id": DEFAULT_MODEL_ID,
            "resolved_commit": DEFAULT_REVISION,
            "hidden_width": layer_spec.residual_width,
            "query_heads": attention.query_heads,
            "key_value_heads": attention.key_value_heads,
            "head_dimension": attention.head_dimension,
            "intermediate_width": transformer.feed_forward.intermediate_width,
            "layer_type": getattr(config, "layer_types", [None] * 5)[4],
            "sliding_window": getattr(config, "sliding_window", None),
            "runtime_dtype": str(next(native.parameters()).dtype),
            "device": str(next(native.parameters()).device),
        },
        "exact_boundary": {
            "h": LAYER4_CANONICAL_SITES["h"],
            "q_A": LAYER4_CANONICAL_SITES["q_A"],
            "v_A": LAYER4_CANONICAL_SITES["v_A"],
            "h_A": LAYER4_CANONICAL_SITES["h_A"],
            "historical_X4": LAYER4_CANONICAL_SITES["X4"],
            "q_M": LAYER4_CANONICAL_SITES["q_M"],
            "v_M": LAYER4_CANONICAL_SITES["v_M"],
            "h_plus_historical_H4": LAYER4_CANONICAL_SITES["h_plus"],
            "state_cloud": ["h", "h_A", "h_plus"],
            "attention_generator_source": "h",
            "mlp_generator_source": "h_A",
        },
        "normalization": {
            "post_attention": type(native.post_attention_layernorm).__name__,
            "post_attention_epsilon": float(native.post_attention_layernorm.eps),
            "post_mlp": type(native.post_feedforward_layernorm).__name__,
            "post_mlp_epsilon": float(native.post_feedforward_layernorm.eps),
            "compute_dtype": "float32",
            "parameterization": "unit_offset_one_plus_weight",
            "component_rule": "one realized scale from the complete q, shared by every component",
        },
        "carrier": cast(Any, runtime.carrier_artifact).metadata(),
        "split": cast(Any, runtime.split_manifest).to_dict(),
        "guard_opened": runtime.corpus.guard_opened,
        "guard_consumed": runtime.corpus.guard_consumed,
    }
    _write_json(args.output_dir / "boundary_inspection.json", result)
    return result


def _run_extract_carrier(args: argparse.Namespace) -> dict[str, object]:
    if args.carrier.exists():
        artifact = load_fisher_local_chart_carrier(args.carrier)
        action = "authenticated_existing"
    else:
        artifact = extract_fisher_local_chart_carrier(
            source_path=args.fisher_source,
            output_path=args.carrier,
            rank=64,
        )
        action = "extracted_exclusively"
    return {"phase": "extract-carrier", "action": action, **artifact.metadata()}


def _single_example_panel(panel: GemmaProgressivePanel) -> GemmaProgressivePanel:
    return GemmaProgressivePanel(
        role=panel.role,
        manifest_sha256=panel.manifest_sha256,
        examples=(panel.examples[0],),
        forbidden_manifest_sha256s=panel.forbidden_manifest_sha256s,
    )


def _run_smoke(args: argparse.Namespace, *, config: ProbeConfig) -> dict[str, object]:
    runtime = _runtime_context(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        selection_input_path=args.selection_input,
        guard_input_path=args.guard_input,
        carrier_path=args.carrier,
        cache_dir=args.cache_dir,
    )
    fit_input = runtime.corpus.open_development_role("calibration_a_fit")
    panel = _single_example_panel(_materialize_panel(runtime, fit_input))
    rows = _capture_panel(runtime, panel, config=config)
    identities = [
        identity
        for receipt in rows.capture_diagnostics
        for identity in cast(Mapping[str, object], receipt["identities"]).values()
    ]
    result = {
        "schema": SCHEMA,
        "phase": "smoke",
        "sequence_id": panel.examples[0].example_id,
        "token_rows": rows.row_count,
        "teacher_forward_count": rows.teacher_forward_count,
        "capture_seconds": rows.capture_seconds,
        "absolute_tolerance": 1.0e-4,
        "relative_tolerance": 1.0e-5,
        "maximum_observed_absolute_error": max(
            float(cast(Mapping[str, float], item)["maximum_absolute_error"])
            for item in identities
        ),
        "maximum_observed_relative_l2_error": max(
            float(cast(Mapping[str, float], item)["relative_l2_error"])
            for item in identities
        ),
        "captures": rows.capture_diagnostics,
        "guard_opened": runtime.corpus.guard_opened,
        "guard_consumed": runtime.corpus.guard_consumed,
    }
    _write_json(args.output_dir / "smoke_capture.json", result)
    return result


def _affine_component_sum_error(
    arm_evaluation: Mapping[str, object], key: str
) -> float:
    affine = cast(Mapping[str, object], arm_evaluation["affine_generators"])
    sums = cast(Mapping[str, object], affine["component_sums"])
    condition = cast(Mapping[str, object], sums[key])
    metric = cast(Mapping[str, object], condition["affine"])
    return float(metric["fisher_relative_squared_error"])


def _run_fit_select(args: argparse.Namespace, *, config: ProbeConfig) -> dict[str, object]:
    if args.candidate.exists() or args.candidate.with_suffix(".pt.manifest.json").exists():
        raise FileExistsError("refusing to replace an existing frozen candidate")
    report_path = args.output_dir / "fit_selection_report.json"
    if report_path.exists():
        raise FileExistsError("refusing to replace an existing fit/selection report")
    phase_started = time.perf_counter()
    runtime = _runtime_context(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        selection_input_path=args.selection_input,
        guard_input_path=args.guard_input,
        carrier_path=args.carrier,
        cache_dir=args.cache_dir,
    )
    carrier_artifact = cast(Any, runtime.carrier_artifact)
    carrier: Rank64FisherCarrier = carrier_artifact.carrier
    split = cast(Any, runtime.split_manifest)
    initial_protocol_sha256 = _protocol_sha256(
        config,
        split_manifest_sha256=split.manifest_sha256,
        carrier_sha256=carrier_artifact.carrier_sha256,
    )
    initial_model_sha256 = runtime.adapter.model_fingerprint()
    initial_execution_sha256 = runtime.adapter.execution_fingerprint()
    if runtime.corpus.guard_opened or runtime.corpus.guard_consumed:
        raise RuntimeError("guard state changed before fit/selection")
    fit_input = runtime.corpus.open_development_role("calibration_a_fit")
    selection_input = runtime.corpus.open_development_role("calibration_a_selection")
    fit_panel = _materialize_panel(runtime, fit_input)
    selection_panel = _materialize_panel(runtime, selection_input)
    fit_rows = _capture_panel(runtime, fit_panel, config=config)
    selection_rows = _capture_panel(runtime, selection_panel, config=config)
    fit_started = time.perf_counter()
    uncalibrated = _fit_baselines(fit_rows, carrier, config=config)
    selected_ridge, generators, ridge_transcript = _fit_generator_grid(
        uncalibrated, fit_rows, selection_rows, carrier, config=config
    )
    selected_coverage, baselines, support_transcript = _select_support_coverage(
        uncalibrated,
        generators,
        selection_rows,
        carrier,
        config=config,
    )
    routers: dict[str, dict[str, Mapping[str, object]]] = {}
    for scheme, baseline_set in baselines.items():
        fit_assignments = _role_assignments(fit_rows, baseline_set)
        routers[scheme] = {
            arm: _fit_routers(
                atlas,
                fit_rows,
                fit_assignments,
                ridge=selected_ridge,
            )
            for arm, atlas in baseline_set.atlases.items()
        }
    fit_seconds = time.perf_counter() - fit_started
    evaluation_started = time.perf_counter()
    role_evaluations: dict[str, object] = {}
    for role_name, rows in (("A_fit", fit_rows), ("A_selection", selection_rows)):
        role_evaluations[role_name] = {
            scheme: {
                arm: _evaluate_arm(
                    arm=arm,
                    scheme=scheme,
                    atlas=baseline_set.atlases[arm],
                    baseline_set=baseline_set,
                    models=generators[scheme][arm],
                    routers=routers[scheme][arm],
                    rows=rows,
                    carrier=carrier,
                )
                for arm in BASELINE_IDS
            }
            for scheme, baseline_set in baselines.items()
        }
    similarities = {
        role_name: {
            scheme: _fisher_similarity_matrix(rows, carrier, scheme=scheme)
            for scheme in PARTITION_SCHEMES
        }
        for role_name, rows in (("A_fit", fit_rows), ("A_selection", selection_rows))
    }
    evaluation_seconds = time.perf_counter() - evaluation_started
    selection_results = cast(Mapping[str, object], role_evaluations["A_selection"])
    selected_scheme = min(
        PARTITION_SCHEMES,
        key=lambda scheme: float(
            cast(
                Mapping[str, object],
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], selection_results[scheme])["B4"],
                )["stage_outputs"],
            )["composed"]["fisher_relative_squared_error"]  # type: ignore[index]
        ),
    )
    selected_scheme_results = cast(
        Mapping[str, object], selection_results[selected_scheme]
    )
    attention_affine_control_arm = min(
        BASELINE_IDS[:-1],
        key=lambda arm: _affine_component_sum_error(
            cast(Mapping[str, object], selected_scheme_results[arm]),
            "attention_causal_context",
        ),
    )
    mlp_affine_control_arm = min(
        BASELINE_IDS[:-1],
        key=lambda arm: _affine_component_sum_error(
            cast(Mapping[str, object], selected_scheme_results[arm]),
            "mlp_teacher_forced",
        ),
    )
    assignment_sha256 = _shared_assignment_hash(baselines)
    protocol_sha256 = _protocol_sha256(
        config,
        split_manifest_sha256=split.manifest_sha256,
        carrier_sha256=carrier_artifact.carrier_sha256,
    )
    if protocol_sha256 != initial_protocol_sha256:
        raise RuntimeError("protocol identity changed during fit-select")
    source_model_sha256 = runtime.adapter.model_fingerprint()
    source_execution_sha256 = runtime.adapter.execution_fingerprint()
    if (
        source_model_sha256 != initial_model_sha256
        or source_execution_sha256 != initial_execution_sha256
    ):
        raise RuntimeError("Gemma model or execution semantics changed during fit-select")
    evaluation_bundle = {
        "roles": role_evaluations,
        "fisher_similarity_matrices": similarities,
    }
    bundle = _bundle_from_fit(
        baselines=baselines,
        generators=generators,
        routers=routers,
        evaluations=evaluation_bundle,
        ridge_transcript=ridge_transcript,
        support_transcript=support_transcript,
        selected_ridge=selected_ridge,
        selected_coverage=selected_coverage,
        selected_scheme=selected_scheme,
        attention_affine_control_arm=attention_affine_control_arm,
        mlp_affine_control_arm=mlp_affine_control_arm,
        assignment_sha256=assignment_sha256,
        protocol_sha256=protocol_sha256,
        split_manifest_sha256=split.manifest_sha256,
        carrier_sha256=carrier_artifact.carrier_sha256,
        corpus_artifact_sha256=runtime.corpus.artifact.artifact_sha256,
        guard_manifest_sha256=runtime.corpus.preclaim_view("calibration_a_guard").manifest_sha256,
        model_config_sha256=carrier_artifact.config_sha256,
        source_model_sha256=source_model_sha256,
        source_execution_sha256=source_execution_sha256,
        config=config,
    )
    if (
        _protocol_sha256(
            config,
            split_manifest_sha256=split.manifest_sha256,
            carrier_sha256=carrier_artifact.carrier_sha256,
        )
        != protocol_sha256
    ):
        raise RuntimeError("protocol identity changed before candidate freeze")
    candidate = freeze_fisher_local_chart_candidate(
        bundle=bundle,
        protocol_sha256=protocol_sha256,
        split_manifest_sha256=split.manifest_sha256,
        carrier_sha256=carrier_artifact.carrier_sha256,
        path=args.candidate,
    )
    # Exercise the exact reloaded tensors, not the pre-save Python objects.
    candidate = load_fisher_local_chart_frozen_candidate(candidate.path)
    _candidate_model_free_dry_run(candidate, carrier, selection_rows)
    result = {
        "schema": SCHEMA,
        "phase": "fit-select",
        "candidate": {
            "path": str(candidate.path),
            "manifest_path": str(candidate.manifest_path),
            "candidate_sha256": candidate.candidate_sha256,
            "candidate_file_sha256": candidate.candidate_file_sha256,
            "protocol_sha256": candidate.protocol_sha256,
            "split_manifest_sha256": candidate.split_manifest_sha256,
            "carrier_sha256": candidate.carrier_sha256,
        },
        "bindings": dict(cast(Mapping[str, object], candidate.bundle["bindings"])),
        "selected": bundle["selected_hyperparameters"],
        "split_manifest": split.to_dict(),
        "carrier": carrier_artifact.metadata(),
        "capture": {
            "A_fit": {
                "sequence_count": len(fit_panel.examples),
                "token_rows": fit_rows.row_count,
                "teacher_forward_count": fit_rows.teacher_forward_count,
                "seconds": fit_rows.capture_seconds,
                "diagnostics": fit_rows.capture_diagnostics,
            },
            "A_selection": {
                "sequence_count": len(selection_panel.examples),
                "token_rows": selection_rows.row_count,
                "teacher_forward_count": selection_rows.teacher_forward_count,
                "seconds": selection_rows.capture_seconds,
                "diagnostics": selection_rows.capture_diagnostics,
            },
        },
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.perf_counter() - phase_started,
        "peak_host_bytes": _peak_host_bytes(),
        "guard_opened": runtime.corpus.guard_opened,
        "guard_consumed": runtime.corpus.guard_consumed,
        "evaluations": evaluation_bundle,
    }
    _write_json(report_path, result)
    return result


def _component_fisher_energy_rows(
    atlas: FrozenAtlas,
    rows: RoleRows,
    assignments: Mapping[str, Tensor],
    carrier: Rank64FisherCarrier,
    *,
    scheme: str,
) -> tuple[Tensor, Tensor]:
    targets: list[Tensor] = []
    residuals: list[Tensor] = []
    for head in range(4):
        target = rows.attention_components[:, head]
        residual = target.double() - _project_by_chart(
            atlas, target, assignments["h"]
        )
        targets.append(carrier.metric_norm_squared(target))
        residuals.append(carrier.metric_norm_squared(residual))
    for block in range(4):
        target = rows.mlp_components[scheme][:, block]
        residual = target.double() - _project_by_chart(
            atlas, target, assignments["h_A"]
        )
        targets.append(carrier.metric_norm_squared(target))
        residuals.append(carrier.metric_norm_squared(residual))
    return torch.stack(targets, dim=1), torch.stack(residuals, dim=1)


def _generator_fisher_energy_rows(
    atlas: FrozenAtlas,
    models: Mapping[str, object],
    rows: RoleRows,
    assignments: Mapping[str, Tensor],
    carrier: Rank64FisherCarrier,
    *,
    scheme: str,
) -> dict[str, Tensor]:
    prediction = _predict_components(
        atlas, models, rows, assignments, scheme=scheme
    )
    attention_residual = prediction["attention_sum"] - rows.v_A.double()
    mlp_residual = prediction["mlp_sum"] - rows.v_M.double()
    return {
        "attention_target": carrier.metric_norm_squared(rows.v_A),
        "attention_residual": carrier.metric_norm_squared(attention_residual),
        "mlp_target": carrier.metric_norm_squared(rows.v_M),
        "mlp_residual": carrier.metric_norm_squared(mlp_residual),
    }


def _verdicts_and_bootstrap(
    candidate: FisherLocalChartFrozenCandidate,
    carrier: Rank64FisherCarrier,
    guard_rows: RoleRows,
    guard_evaluations: Mapping[str, object],
    fidelity: Mapping[str, object],
    *,
    config: ProbeConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    selected = cast(Mapping[str, object], candidate.bundle["selected_hyperparameters"])
    scheme = str(selected["selected_partition_scheme"])
    selected_scheme = cast(Mapping[str, object], guard_evaluations[scheme])
    b4 = cast(Mapping[str, object], selected_scheme["B4"])
    b3 = cast(Mapping[str, object], selected_scheme["B3"])

    def component_average(arm: Mapping[str, object], metric: str) -> float:
        tangent = cast(Mapping[str, object], arm["tangent_capture"])
        names = [f"attention_head_{index}" for index in range(4)] + [
            f"mlp_probe_block_{index}" for index in range(4)
        ]
        return math.fsum(
            float(cast(Mapping[str, object], tangent[name])[metric]) for name in names
        ) / len(names)

    fisher_tangent_b4 = component_average(b4, "fisher_captured_fraction")
    fisher_tangent_b3 = component_average(b3, "fisher_captured_fraction")
    tangent_delta = fisher_tangent_b4 - fisher_tangent_b3

    attention_b4 = _affine_component_sum_error(b4, "attention_causal_context")
    mlp_b4 = _affine_component_sum_error(b4, "mlp_teacher_forced")
    attention_control_arm = str(selected["attention_affine_control_arm"])
    mlp_control_arm = str(selected["mlp_affine_control_arm"])
    if attention_control_arm not in BASELINE_IDS[:-1] or mlp_control_arm not in BASELINE_IDS[:-1]:
        raise ValueError("frozen affine comparator must be one of B0-B3")
    attention_control = _affine_component_sum_error(
        cast(Mapping[str, object], selected_scheme[attention_control_arm]),
        "attention_causal_context",
    )
    mlp_control = _affine_component_sum_error(
        cast(Mapping[str, object], selected_scheme[mlp_control_arm]),
        "mlp_teacher_forced",
    )
    source_error = _affine_component_sum_error(b4, "attention_source_only")
    context_error = attention_b4
    attention_constant = float(
        cast(
            Mapping[str, object],
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], b4["affine_generators"])["component_sums"],
            )["attention_causal_context"],
        )["constant"]["fisher_relative_squared_error"]  # type: ignore[index]
    )
    stage_outputs = cast(Mapping[str, object], b4["stage_outputs"])
    attention_stage_error = float(
        cast(Mapping[str, object], stage_outputs["attention"])["fisher_relative_squared_error"]
    )
    teacher_error = float(
        cast(Mapping[str, object], stage_outputs["mlp_teacher_forced"])["fisher_relative_squared_error"]
    )
    composed_error = float(
        cast(Mapping[str, object], stage_outputs["composed"])["fisher_relative_squared_error"]
    )
    support_drift = float(
        cast(
            Mapping[str, object],
            cast(Mapping[str, object], b4["support"])["composed_support_drift"],
        )["support_drift_rate"]
    )
    router = cast(Mapping[str, object], b4["router"])
    routing = {}
    routing_positive = []
    for stage in ("attention", "mlp"):
        stage_result = cast(Mapping[str, object], router[stage])
        delta = cast(Mapping[str, object], stage_result["delta_vs_source_frequency"])
        ce_gain = float(delta["cross_entropy_reduction"])
        top1_gain = float(delta["top1_improvement"])
        routing[stage] = {
            "cross_entropy_reduction_vs_source_frequency": ce_gain,
            "top1_improvement_vs_source_frequency": top1_gain,
            "supports_predictability": ce_gain > 0.0,
        }
        routing_positive.append(ce_gain > 0.0)
    fidelity_gate = bool(cast(Mapping[str, object], fidelity["gates"])["passed"])

    b4_set = _baseline_set_for_candidate(candidate, carrier, scheme=scheme)
    assignments = _role_assignments(guard_rows, b4_set)
    b4_atlas = b4_set.atlases["B4"]
    b3_atlas = b4_set.atlases["B3"]
    tangent_target, b4_tangent_residual = _component_fisher_energy_rows(
        b4_atlas, guard_rows, assignments, carrier, scheme=scheme
    )
    b3_tangent_target, b3_tangent_residual = _component_fisher_energy_rows(
        b3_atlas, guard_rows, assignments, carrier, scheme=scheme
    )
    if not torch.equal(tangent_target, b3_tangent_target):
        raise RuntimeError("tangent bootstrap target energy changed across arms")
    tangent_bootstrap = _bootstrap_ratio_delta(
        b4_tangent_residual,
        b3_tangent_residual,
        tangent_target,
        guard_rows.sequence_ids,
        replicates=config.bootstrap_replicates,
        seed=config.chart_seed,
        estimand="unweighted mean across eight components of B4 minus B3 shared-Fisher captured fraction",
    )

    generator_root = cast(Mapping[str, object], candidate.bundle["generators"])
    scheme_generators = cast(Mapping[str, object], generator_root[scheme])
    generator_energy: dict[str, dict[str, Tensor]] = {}
    for arm in dict.fromkeys(("B4", attention_control_arm, mlp_control_arm)):
        generator_energy[arm] = _generator_fisher_energy_rows(
            b4_set.atlases[arm],
            cast(Mapping[str, object], scheme_generators[arm]),
            guard_rows,
            assignments,
            carrier,
            scheme=scheme,
        )
    b4_energy = generator_energy["B4"]
    attention_control_energy = generator_energy[attention_control_arm]
    mlp_control_energy = generator_energy[mlp_control_arm]
    if not torch.equal(
        b4_energy["attention_target"], attention_control_energy["attention_target"]
    ) or not torch.equal(b4_energy["mlp_target"], mlp_control_energy["mlp_target"]):
        raise RuntimeError("affine bootstrap target energy changed across arms")
    attention_bootstrap = _bootstrap_ratio_delta(
        b4_energy["attention_residual"],
        attention_control_energy["attention_residual"],
        b4_energy["attention_target"],
        guard_rows.sequence_ids,
        replicates=config.bootstrap_replicates,
        seed=config.chart_seed,
        estimand=(
            "best Euclidean control minus B4 attention causal-context component-sum "
            "shared-Fisher relative squared error"
        ),
    )
    mlp_bootstrap = _bootstrap_ratio_delta(
        b4_energy["mlp_residual"],
        mlp_control_energy["mlp_residual"],
        b4_energy["mlp_target"],
        guard_rows.sequence_ids,
        replicates=config.bootstrap_replicates,
        seed=config.chart_seed,
        estimand=(
            "best Euclidean control minus B4 teacher-forced MLP component-sum "
            "shared-Fisher relative squared error"
        ),
    )
    expected_points = {
        "tangent": tangent_delta,
        "attention affine": attention_control - attention_b4,
        "MLP affine": mlp_control - mlp_b4,
    }
    intervals = {
        "tangent": tangent_bootstrap,
        "attention affine": attention_bootstrap,
        "MLP affine": mlp_bootstrap,
    }
    for label, expected in expected_points.items():
        point = float(intervals[label]["point"])
        if not math.isclose(point, expected, rel_tol=1.0e-10, abs_tol=1.0e-12):
            raise RuntimeError(f"{label} bootstrap point does not match reported estimand")

    tangent_status = _ci_evidence_status(tangent_bootstrap)
    attention_status = _ci_evidence_status(attention_bootstrap)
    mlp_status = _ci_evidence_status(mlp_bootstrap)
    bootstrap = {
        "tangent_B4_vs_B3_mean_component_capture_delta": tangent_bootstrap,
        "affine_attention_B4_vs_best_euclidean_control_error_delta": attention_bootstrap,
        "affine_mlp_teacher_forced_B4_vs_best_euclidean_control_error_delta": mlp_bootstrap,
    }
    verdicts = {
        "fisher_tangent_capture": {
            "status": tangent_status,
            "status_basis": "two-sided 95% whole-sequence bootstrap interval",
            "B4_mean_component_capture": fisher_tangent_b4,
            "B3_mean_component_capture": fisher_tangent_b3,
            "delta_tangent": tangent_delta,
            "bootstrap_95": tangent_bootstrap,
            "interpretation": (
                "The 95% sequence-bootstrap interval supports greater held-out component-flow capture for the shared-output-Fisher chart."
                if tangent_status == "positive"
                else "The 95% sequence-bootstrap interval supports greater held-out component-flow capture for the B3 component-flow PCA control."
                if tangent_status == "negative"
                else "The 95% sequence-bootstrap interval overlaps zero, so tangent-capture evidence is inconclusive."
            ),
        },
        "affine_generators": {
            "attention": {
                "status": attention_status,
                "status_basis": "two-sided 95% whole-sequence bootstrap interval",
                "B4_fisher_relative_squared_error": attention_b4,
                "best_euclidean_control_error": attention_control,
                "best_euclidean_control_arm": attention_control_arm,
                "delta_affine": attention_control - attention_b4,
                "bootstrap_95": attention_bootstrap,
            },
            "mlp_teacher_forced": {
                "status": mlp_status,
                "status_basis": "two-sided 95% whole-sequence bootstrap interval",
                "B4_fisher_relative_squared_error": mlp_b4,
                "best_euclidean_control_error": mlp_control,
                "best_euclidean_control_arm": mlp_control_arm,
                "delta_affine": mlp_control - mlp_b4,
                "bootstrap_95": mlp_bootstrap,
                "interpretation": (
                    "The 95% sequence-bootstrap interval supports the selected Fisher-coordinate affine MLP representation."
                    if mlp_status == "positive"
                    else "The 95% sequence-bootstrap interval supports the best Euclidean MLP control."
                    if mlp_status == "negative"
                    else "The 95% sequence-bootstrap interval overlaps zero, so the MLP affine comparison is inconclusive."
                ),
            },
        },
        "attention_context_sufficiency": {
            "source_only_works_vs_constant": source_error < attention_constant,
            "causal_context_improves": context_error < source_error,
            "source_only_error": source_error,
            "causal_context_error": context_error,
            "constant_error": attention_constant,
            "incremental_context_value": source_error - context_error,
            "status": (
                "source_state_sufficient"
                if source_error < attention_constant and context_error >= source_error
                else "context_supplies_missing_routing_information"
                if context_error < source_error and context_error < attention_constant
                else "selected_affine_attention_representation_not_supported"
            ),
        },
        "graph_routing": {
            "status": "positive" if all(routing_positive) else "mixed_or_negative",
            "stages": routing,
            "interpretation": "Attention and MLP transitions remain separate; no merged routing score is used.",
        },
        "composition": {
            "status": (
                "preserved"
                if composed_error <= teacher_error + attention_stage_error
                else "degraded_beyond_separate_stage_errors"
            ),
            "attention_stage_error": attention_stage_error,
            "teacher_forced_mlp_output_error": teacher_error,
            "composed_output_error": composed_error,
            "support_drift_rate": support_drift,
            "interpretation": (
                "Composition remains within the sum of separately observed normalized stage errors."
                if composed_error <= teacher_error + attention_stage_error
                else "Teacher-forced accuracy does not survive composition; attention error and support drift must be separated from MLP nonlinearity."
            ),
        },
        "full_sequence_fidelity": {
            "status": "legacy_diagnostic_gates_pass" if fidelity_gate else "legacy_diagnostic_gates_fail",
            "legacy_diagnostic_gates_passed": fidelity_gate,
            "delta_nll_per_token": cast(Mapping[str, object], fidelity["aggregate"])["delta_nll_per_token"],
            "source_to_candidate_kl_per_token": cast(Mapping[str, object], fidelity["aggregate"])["source_to_candidate_kl_per_token"],
            "top1_agreement_to_source": cast(Mapping[str, object], fidelity["aggregate"])["top1_agreement_to_source"],
            "qualification": "The established fidelity thresholds are diagnostic; this probe did not tune them on A-selection.",
        },
    }

    return verdicts, bootstrap


def _partition_stability(
    guard_evaluations: Mapping[str, object],
) -> dict[str, object]:
    selected_rows = {
        scheme: cast(
            Mapping[str, object],
            cast(Mapping[str, object], guard_evaluations[scheme])["B4"],
        )
        for scheme in PARTITION_SCHEMES
    }

    def scalar(scheme: str, path: Sequence[str]) -> float:
        value: object = selected_rows[scheme]
        for key in path:
            value = cast(Mapping[str, object], value)[key]
        return float(value)

    comparisons = {
        "mlp_tangent_capture": {
            scheme: (
                math.fsum(
                    scalar(
                        scheme,
                        (
                            "tangent_capture",
                            f"mlp_probe_block_{block}",
                            "fisher_captured_fraction",
                        ),
                    )
                    for block in range(4)
                )
                / 4
            )
            for scheme in PARTITION_SCHEMES
        }
    }
    comparisons["mlp_affine_error"] = {
        scheme: scalar(
            scheme,
            (
                "affine_generators",
                "component_sums",
                "mlp_teacher_forced",
                "affine",
                "fisher_relative_squared_error",
            ),
        )
        for scheme in PARTITION_SCHEMES
    }
    comparisons["graph_attention_cross_entropy"] = {
        scheme: scalar(scheme, ("router", "attention", "affine", "cross_entropy"))
        for scheme in PARTITION_SCHEMES
    }
    comparisons["graph_mlp_cross_entropy"] = {
        scheme: scalar(scheme, ("router", "mlp", "affine", "cross_entropy"))
        for scheme in PARTITION_SCHEMES
    }
    relative_differences = {}
    for name, values in comparisons.items():
        typed = cast(Mapping[str, float], values)
        left, right = (typed[scheme] for scheme in PARTITION_SCHEMES)
        relative_differences[name] = abs(left - right) / max(abs(left), abs(right), 1e-12)
    sensitive = any(value > 0.10 for value in relative_differences.values())
    return {
        "material_relative_difference_threshold": 0.10,
        "metrics": comparisons,
        "relative_differences": relative_differences,
        "partition_sensitive": sensitive,
        "verdict": (
            "MLP conclusions are partition-sensitive"
            if sensitive
            else "MLP conclusions are stable across the two prescribed probe partitions at the 10% materiality threshold"
        ),
        "qualification": "The four channel groups are deterministic probe blocks, not discovered functional generators.",
    }


def _evaluations_section(
    roles: Mapping[str, object], key: str
) -> dict[str, object]:
    return {
        role: {
            scheme: {
                arm: cast(Mapping[str, object], arm_value)[key]
                for arm, arm_value in cast(Mapping[str, object], scheme_value).items()
            }
            for scheme, scheme_value in cast(Mapping[str, object], role_value).items()
        }
        for role, role_value in roles.items()
    }


def _transition_artifact(
    roles: Mapping[str, object],
    candidate: FisherLocalChartFrozenCandidate,
    carrier: Rank64FisherCarrier,
    guard_rows: RoleRows,
) -> tuple[dict[str, object], str]:
    matrices: dict[str, object] = {}
    for role, role_value in roles.items():
        matrices[role] = {}
        for scheme, scheme_value in cast(Mapping[str, object], role_value).items():
            cast(dict[str, object], matrices[role])[scheme] = {}
            for arm, arm_value in cast(Mapping[str, object], scheme_value).items():
                router = cast(Mapping[str, object], cast(Mapping[str, object], arm_value)["router"])
                cast(dict[str, object], cast(dict[str, object], matrices[role])[scheme])[arm] = {
                    stage: cast(
                        Mapping[str, object],
                        cast(Mapping[str, object], router[stage])["transition_structure"],
                    )
                    for stage in ("attention", "mlp")
                }
    selected = cast(Mapping[str, object], candidate.bundle["selected_hyperparameters"])
    scheme = str(selected["selected_partition_scheme"])
    baseline_set = _baseline_set_for_candidate(candidate, carrier, scheme=scheme)
    atlas = baseline_set.atlases["B4"]
    assignments = _role_assignments(guard_rows, baseline_set)
    h_supported, _ = _assigned_support(atlas, guard_rows.h, assignments["h"])
    h_a_supported, _ = _assigned_support(atlas, guard_rows.h_A, assignments["h_A"])
    h_plus_supported, _ = _assigned_support(
        atlas, guard_rows.h_plus, assignments["h_plus"]
    )
    chart_count = atlas.chart_count
    one_hot = lambda labels: torch.nn.functional.one_hot(  # noqa: E731
        labels, chart_count
    ).double()
    attention_table = build_typed_transition_table(
        one_hot(assignments["h"]),
        one_hot(assignments["h_A"]),
        one_hot(assignments["h_plus"]),
        supported=h_supported & h_a_supported,
    )
    mlp_table = build_typed_transition_table(
        one_hot(assignments["h"]),
        one_hot(assignments["h_A"]),
        one_hot(assignments["h_plus"]),
        supported=h_a_supported & h_plus_supported,
    )
    table = TypedTransitionTable(
        attention=attention_table.attention,
        mlp=mlp_table.mlp,
    )

    def distribution(role: str, stage: str) -> Tensor:
        value = cast(
            Mapping[str, object],
            cast(
                Mapping[str, object],
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], roles[role])[scheme],
                )["B4"],
            )["router"],
        )[stage]
        counts = cast(
            Tensor,
            cast(Mapping[str, object], value)["transition_structure"]["counts"],  # type: ignore[index]
        ).double()
        return counts / counts.sum().clamp_min(1)

    drift = {}
    for stage in ("attention", "mlp"):
        fit_distribution = distribution("A_fit", stage)
        drift[stage] = {
            role: float((distribution(role, stage) - fit_distribution).abs().sum() / 2)
            for role in ("A_selection", "A_guard")
        }
    return {
        "schema": REPORT_SCHEMA,
        "typed_edges": "input --attention--> intermediate --MLP--> output",
        "matrices": matrices,
        "selected_supported_transition_matrices": {
            "attention": table.attention,
            "mlp": table.mlp,
        },
        "selected_route_total_variation_drift_from_A_fit": drift,
    }, table.to_dot()


def _next_experiment(verdicts: Mapping[str, object]) -> str:
    tangent = cast(Mapping[str, object], verdicts["fisher_tangent_capture"])
    affine = cast(Mapping[str, object], verdicts["affine_generators"])
    context = cast(Mapping[str, object], verdicts["attention_context_sufficiency"])
    composition = cast(Mapping[str, object], verdicts["composition"])
    if tangent["status"] != "positive":
        return (
            "Estimate stage-local Fisher pullbacks at h and h_A, then repeat the same frozen-neighborhood rank sweep; this directly tests whether the shared output carrier is the limiting boundary assumption."
        )
    if cast(Mapping[str, object], affine["mlp_teacher_forced"])["status"] != "positive":
        return (
            "Increase sequence-level A-fit diversity and test a rank/chart-count grid under the same teacher-forced MLP protocol before adding any nonlinear or shared generator family."
        )
    if context["status"] == "context_supplies_missing_routing_information":
        return (
            "Run the separately labeled compact transported-value-sketch context ablation, holding geometry and affine capacity fixed, to locate the missing attention information."
        )
    if composition["status"] != "preserved":
        return (
            "Fit a support-aware attention correction on fresh A-development sequences and test whether reducing h_A support drift restores composition without changing the teacher-forced MLP model."
        )
    return (
        "Repeat the frozen protocol on a larger family-disjoint sequence corpus and a preregistered rank/chart-count grid. Recurrent sharing remains unjustified until held-out vector fields agree over overlapping chart support."
    )


def _markdown_report(
    *,
    candidate: FisherLocalChartFrozenCandidate,
    carrier_metadata: Mapping[str, object],
    guard_evaluations: Mapping[str, object],
    fidelity: Mapping[str, object],
    verdicts: Mapping[str, object],
    partition_stability: Mapping[str, object],
    invariant_summary: Mapping[str, object],
    sequence_counts: Mapping[str, int],
    next_experiment: str,
) -> str:
    selected = cast(Mapping[str, object], candidate.bundle["selected_hyperparameters"])
    scheme = str(selected["selected_partition_scheme"])
    b4 = cast(
        Mapping[str, object],
        cast(Mapping[str, object], guard_evaluations[scheme])["B4"],
    )
    b3 = cast(
        Mapping[str, object],
        cast(Mapping[str, object], guard_evaluations[scheme])["B3"],
    )
    tangent_verdict = cast(Mapping[str, object], verdicts["fisher_tangent_capture"])
    affine_verdict = cast(Mapping[str, object], verdicts["affine_generators"])
    context_verdict = cast(Mapping[str, object], verdicts["attention_context_sufficiency"])
    graph_verdict = cast(Mapping[str, object], verdicts["graph_routing"])
    composition_verdict = cast(Mapping[str, object], verdicts["composition"])
    fidelity_verdict = cast(Mapping[str, object], verdicts["full_sequence_fidelity"])
    aggregate_fidelity = cast(Mapping[str, object], fidelity["aggregate"])
    fit_sequence_count = int(sequence_counts["A_fit"])
    selection_sequence_count = int(sequence_counts["A_selection"])
    guard_sequence_count = int(sequence_counts["A_guard"])
    total_sequence_count = (
        fit_sequence_count + selection_sequence_count + guard_sequence_count
    )
    return f"""# Fisher local-chart probe — Gemma 3 layer 4

## Scope and frozen selection

This experiment is a stage-aware local atlas under the **shared
`layer.4.output` empirical Fisher metric**.  The carrier is rank 64 and retains
{float(carrier_metadata['retained_trace_fraction']):.6f} of the source empirical
Fisher trace.  It is not the exact local Fisher at `h` or `h_A`, and the
experiment makes no compression, deployment, or recurrent-sharing claim.

- Frozen candidate: `{candidate.candidate_sha256}`
- Selected arm: `{selected['selected_baseline']}`
- Selected MLP probe partition: `{scheme}`
- Charts / coordinate rank: `{selected['chart_count']}` / `{selected['coordinate_rank']}`
- Affine ridge: `{float(selected['ridge']):.6g}`
- Tangential support coverage: `{float(selected['support_coverage']):.3f}`

## Exact implemented transition

```text
h --attention--> h_A --MLP--> h_plus
v_A = h_A - h
v_M = h_plus - h_A
v_total = v_A + v_M
```

For each attention head, `q_A,j` is formed with its exact `o_proj` slice and
then multiplied by one shared realized `D_A(q_A)`.  For each MLP probe block,
`q_M,r` is formed after `down_proj` and multiplied by one shared realized
`D_M(q_M)`.  Neither path independently RMS-normalizes components.  The atlas
state cloud is exactly `{{h, h_A, h_plus}}`; historical `X4` is the normalized
MLP input and historical `H4` is `h_plus`.

The supported Fisher basis is implemented as

```text
C_Q = Q.T C Q
Lambda_reg = Lambda + ridge I_64
S_Q = sqrt(Lambda_reg) C_Q sqrt(Lambda_reg)
B = Q invsqrt(Lambda_reg) U_k
z(s) = U_k.T sqrt(Lambda_reg) Q.T (s - mu)
```

No dense 640-by-640 Fisher matrix is materialized.

## One-shot A-guard headline

| Quantity | Result |
|---|---:|
| Mean component Fisher capture B4 | {float(tangent_verdict['B4_mean_component_capture']):.6f} |
| Mean component Fisher capture B3 | {float(tangent_verdict['B3_mean_component_capture']):.6f} |
| Delta tangent (B4 - B3) | {float(tangent_verdict['delta_tangent']):.6f} |
| Attention B4 affine Fisher relative error | {float(cast(Mapping[str, object], affine_verdict['attention'])['B4_fisher_relative_squared_error']):.6f} |
| Teacher MLP B4 affine Fisher relative error | {float(cast(Mapping[str, object], affine_verdict['mlp_teacher_forced'])['B4_fisher_relative_squared_error']):.6f} |
| Composed output Fisher relative error | {float(composition_verdict['composed_output_error']):.6f} |
| Support drift rate | {float(composition_verdict['support_drift_rate']):.6f} |
| Delta NLL / token | {float(aggregate_fidelity['delta_nll_per_token']):.6f} |
| Source-to-candidate KL / token | {float(aggregate_fidelity['source_to_candidate_kl_per_token']):.6f} |
| Top-1 agreement | {float(aggregate_fidelity['top1_agreement_to_source']):.6f} |

## Independent verdicts

1. **Fisher tangent capture — `{tangent_verdict['status']}`.** {tangent_verdict['interpretation']}
2. **Affine generators — attention `{cast(Mapping[str, object], affine_verdict['attention'])['status']}`, teacher-forced MLP `{cast(Mapping[str, object], affine_verdict['mlp_teacher_forced'])['status']}`.** These are coordinate-predictability verdicts, separate from tangent containment.
3. **Attention context sufficiency — `{context_verdict['status']}`.** Source-only Fisher relative error is {float(context_verdict['source_only_error']):.6f}; causal-context error is {float(context_verdict['causal_context_error']):.6f}.
4. **Graph routing — `{graph_verdict['status']}`.** Attention and MLP destination softmaxes and frequency controls are reported separately.
5. **Composition — `{composition_verdict['status']}`.** {composition_verdict['interpretation']}
6. **Full-sequence fidelity — `{fidelity_verdict['status']}`.** The source path remained authoritative; the established gates are diagnostic rather than selection-tuned for this probe.

## Additivity and partition controls

- Configured invariant tolerances: absolute `{float(invariant_summary['absolute_tolerance']):.3g}`, relative-L2 `{float(invariant_summary['relative_tolerance']):.3g}`.
- Largest observed absolute closure: `{float(invariant_summary['maximum_absolute_error']):.8g}`.
- Largest observed relative-L2 closure: `{float(invariant_summary['maximum_relative_l2_error']):.8g}`.
- Partition conclusion: **{partition_stability['verdict']}**.

## Limitations

- This corpus supplies {total_sequence_count} sequences ({fit_sequence_count} fit, {selection_sequence_count} selection, {guard_sequence_count} one-shot guard); all bootstrap units are whole sequences.
- The common metric is a rank-64 output-boundary carrier, not full-rank Fisher geometry or a stage-local pullback.
- Attention context uses five routing summaries only; no transported-value sketch is included.
- Full-sequence fidelity measures full-prompt next-token logits, not cached autoregressive generation.
- MLP groups are two deterministic four-way probe partitions, not discovered functional regions.
- Recurrent generator sharing was not implemented and is not supported by mean-vector similarity alone.

## Most important next experiment

{next_experiment}
"""


def _capture_invariant_summary(
    fit_report: Mapping[str, object], guard_rows: RoleRows
) -> dict[str, object]:
    identities: list[Mapping[str, object]] = []
    capture = cast(Mapping[str, object], fit_report.get("capture", {}))
    for role in ("A_fit", "A_selection"):
        role_capture = capture.get(role)
        if not isinstance(role_capture, Mapping):
            continue
        for receipt in cast(Sequence[Mapping[str, object]], role_capture["diagnostics"]):
            identities.extend(
                cast(Mapping[str, object], receipt["identities"]).values()  # type: ignore[arg-type]
            )
    for receipt in guard_rows.capture_diagnostics:
        identities.extend(
            cast(Mapping[str, object], receipt["identities"]).values()  # type: ignore[arg-type]
        )
    if not identities:
        raise ValueError("no capture identities available for final report")
    return {
        "absolute_tolerance": 1.0e-4,
        "relative_tolerance": 1.0e-5,
        "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in identities),
        "maximum_relative_l2_error": max(float(row["relative_l2_error"]) for row in identities),
        "identity_observations": len(identities),
        "accumulation_dtype": "float64",
    }


def _candidate_identity(
    candidate: FisherLocalChartFrozenCandidate,
) -> tuple[str, str, str, str, str]:
    return (
        candidate.protocol_sha256,
        candidate.split_manifest_sha256,
        candidate.carrier_sha256,
        candidate.candidate_sha256,
        candidate.candidate_file_sha256,
    )


def _load_fit_report_with_sha256(
    path: Path,
    candidate: FisherLocalChartFrozenCandidate,
) -> tuple[dict[str, object], str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("fit/selection report cannot be opened safely") from error
    try:
        details = os.fstat(descriptor)
        if details.st_size <= 0 or details.st_size > 256 * 1024 * 1024:
            raise ValueError("fit/selection report size is invalid")
        chunks: list[bytes] = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("fit/selection report ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("fit/selection report changed while read")
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
    try:
        raw = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fit/selection report is not valid JSON") from error
    if not isinstance(raw, dict) or (
        _canonical_json_bytes(_jsonable(raw)) + b"\n" != encoded
    ):
        raise ValueError("fit/selection report is not canonical JSON")
    if raw.get("schema") != SCHEMA or raw.get("phase") != "fit-select":
        raise ValueError("fit/selection report schema/phase mismatch")

    candidate_receipt = raw.get("candidate")
    if not isinstance(candidate_receipt, Mapping):
        raise ValueError("fit/selection report omits its candidate receipt")
    expected_candidate_receipt = {
        "candidate_sha256": candidate.candidate_sha256,
        "candidate_file_sha256": candidate.candidate_file_sha256,
        "protocol_sha256": candidate.protocol_sha256,
        "split_manifest_sha256": candidate.split_manifest_sha256,
        "carrier_sha256": candidate.carrier_sha256,
    }
    if any(
        candidate_receipt.get(key) != value
        for key, value in expected_candidate_receipt.items()
    ):
        raise ValueError("fit/selection report does not bind the frozen candidate")
    try:
        report_candidate_path = Path(str(candidate_receipt["path"])).resolve()
        report_manifest_path = Path(str(candidate_receipt["manifest_path"])).resolve()
    except (KeyError, OSError) as error:
        raise ValueError("fit/selection report candidate paths are invalid") from error
    if (
        report_candidate_path != candidate.path.resolve()
        or report_manifest_path != candidate.manifest_path.resolve()
    ):
        raise ValueError("fit/selection report binds different candidate paths")

    frozen_bindings = candidate.bundle.get("bindings")
    report_bindings = raw.get("bindings")
    if not isinstance(frozen_bindings, Mapping) or not isinstance(
        report_bindings, Mapping
    ):
        raise ValueError("fit/selection report omits frozen bindings")
    if _canonical_json_bytes(_jsonable(report_bindings)) != _canonical_json_bytes(
        _jsonable(frozen_bindings)
    ):
        raise ValueError("fit/selection report binding receipt mismatch")

    transcript = candidate.bundle.get("selection_transcript")
    if not isinstance(transcript, Mapping):
        raise ValueError("frozen candidate omits selection transcript")
    if _canonical_json_bytes(_jsonable(raw.get("evaluations"))) != _canonical_json_bytes(
        _jsonable(transcript.get("evaluations"))
    ):
        raise ValueError("fit/selection report evaluations differ from the candidate")
    selected = candidate.bundle.get("selected_hyperparameters")
    if _canonical_json_bytes(_jsonable(raw.get("selected"))) != _canonical_json_bytes(
        _jsonable(selected)
    ):
        raise ValueError("fit/selection report selection differs from the candidate")
    if raw.get("guard_opened") is not False or raw.get("guard_consumed") is not False:
        raise ValueError("fit/selection report was not produced before A-guard")
    return raw, hashlib.sha256(encoded).hexdigest()


def _load_fit_report(
    path: Path, candidate: FisherLocalChartFrozenCandidate
) -> dict[str, object]:
    return _load_fit_report_with_sha256(path, candidate)[0]


@dataclass(frozen=True, slots=True)
class _PreclaimSelectionRehearsalClaim:
    """Non-authoritative marker used only to render reports in memory."""

    candidate_sha256: str

    def metadata(self) -> dict[str, object]:
        return {
            "state": "preclaim_selection_rehearsal_without_guard_authority",
            "candidate_sha256": self.candidate_sha256,
            "guard_claimed": False,
            "guard_opened": False,
        }


def _selection_preclaim_rehearsal(
    *,
    args: argparse.Namespace,
    config: ProbeConfig,
    runtime: RuntimeContext,
    candidate: FisherLocalChartFrozenCandidate,
    fit_report: Mapping[str, object],
    selection_panel: GemmaProgressivePanel,
    selection_rows: RoleRows,
    selection_evaluations: Mapping[str, object],
) -> dict[str, object]:
    """Exercise every remaining fallible analysis/report path before claim.

    The rehearsal is intentionally in-memory and labels selection as a
    rehearsal surrogate only.  It neither writes final artifacts nor grants
    access to A-guard.  The real one-shot result is still rendered again from
    the actual guard rows after the irreversible claim.
    """

    fidelity, forward_count, fidelity_seconds = _full_sequence_fidelity(
        runtime, selection_panel, selection_rows, candidate
    )
    carrier = cast(Any, runtime.carrier_artifact).carrier
    verdicts, bootstrap = _verdicts_and_bootstrap(
        candidate,
        carrier,
        selection_rows,
        selection_evaluations,
        fidelity,
        config=config,
    )
    partition_stability = _partition_stability(selection_evaluations)
    transcript = candidate.bundle.get("selection_transcript")
    if not isinstance(transcript, Mapping):
        raise ValueError("candidate omits selection transcript during rehearsal")
    frozen_evaluations = transcript.get("evaluations")
    if not isinstance(frozen_evaluations, Mapping):
        raise ValueError("candidate omits frozen evaluations during rehearsal")
    frozen_roles = frozen_evaluations.get("roles")
    if not isinstance(frozen_roles, Mapping):
        raise ValueError("candidate omits frozen roles during rehearsal")
    rehearsal_roles = dict(frozen_roles)
    rehearsal_roles["A_guard"] = dict(selection_evaluations)
    summaries, encoded = _prepare_final_artifacts(
        args=args,
        config=config,
        runtime=runtime,
        candidate=candidate,
        claim=_PreclaimSelectionRehearsalClaim(candidate.candidate_sha256),
        fit_report=fit_report,
        roles=rehearsal_roles,
        guard_rows=selection_rows,
        preclaim_selection_rows=selection_rows,
        fidelity=fidelity,
        fidelity_forward_count=forward_count,
        fidelity_seconds=fidelity_seconds,
        guard_evaluation_seconds=0.0,
        verdicts=verdicts,
        bootstrap=bootstrap,
        partition_stability=partition_stability,
    )
    expected_names = set(FINAL_ARTIFACT_NAMES) - {"guard_result.json"}
    if set(encoded) != expected_names or any(not value for value in encoded.values()):
        raise RuntimeError("preclaim report rehearsal omitted a final artifact")
    receipt = {
        "mode": "A_selection_only_no_guard_authority",
        "full_sequence_forward_count": forward_count,
        "full_sequence_fidelity_sha256": _sha256_json(
            _jsonable(fidelity), domain="fisher-local-chart-preclaim-fidelity-v1"
        ),
        "verdict_sha256": _sha256_json(
            _jsonable(verdicts), domain="fisher-local-chart-preclaim-verdict-v1"
        ),
        "transition_sha256": _sha256_json(
            _jsonable(summaries["transition_artifact"]),
            domain="fisher-local-chart-preclaim-transition-v1",
        ),
        "rendered_artifact_sha256s": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in sorted(encoded.items())
        },
    }
    # One last finite/canonical serialization check is part of the rehearsal.
    _canonical_json_bytes(_jsonable(receipt))
    return receipt


def _commands_markdown(args: argparse.Namespace) -> str:
    repo = Path(__file__).resolve().parents[2]
    python = repo / ".venv/bin/python"
    module = "fisher_graph.gemma3_fisher_local_chart_probe"
    common = [
        "--corpus-artifact",
        str(args.corpus_artifact),
        "--fit-input",
        str(args.fit_input),
        "--selection-input",
        str(args.selection_input),
        "--guard-input",
        str(args.guard_input),
        "--fisher-source",
        str(args.fisher_source),
        "--carrier",
        str(args.carrier),
        "--output-dir",
        str(args.output_dir),
        "--candidate",
        str(args.candidate),
        "--chart-count",
        str(args.chart_count),
        "--coordinate-rank",
        str(args.coordinate_rank),
        "--ridge-grid",
        *(format(float(value), ".17g") for value in args.ridge_grid),
        "--support-coverages",
        *(format(float(value), ".17g") for value in args.support_coverages),
        "--chart-seed",
        str(args.chart_seed),
        "--partition-seed",
        str(args.partition_seed),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
    ]
    if args.cache_dir is not None:
        common.extend(("--cache-dir", str(args.cache_dir)))

    def command(phase: str, *, acknowledge_guard: bool = False) -> str:
        tokens = [str(python), "-m", module, phase, *common]
        if acknowledge_guard:
            tokens.append("--acknowledge-one-shot-guard")
        return shlex.join(tokens)

    extract = shlex.join(
        [
            str(python),
            "-m",
            module,
            "extract-carrier",
            "--fisher-source",
            str(args.fisher_source),
            "--carrier",
            str(args.carrier),
        ]
    )
    return f"""# Exact reproduction commands

Run from `{repo}` with the pinned local model/cache already present.

```bash
cd {shlex.quote(str(repo))}
{extract}
{command("inspect")}
{command("smoke")}
{command("fit-select")}
{command("guard", acknowledge_guard=True)}
```

The final command performs the manifest-global one-shot A-guard claim.  Do
not rerun it after a post-claim failure; the durable claim intentionally burns
that guard attempt.
"""


def _boundary_design_markdown() -> str:
    return """# Exact Gemma 3 layer-4 boundary design

The pinned `google/gemma-3-270m` decoder layer is modeled as two typed
residual stages:

```text
h   = layer.4.input
q_A = layer.4.attention.operator_output
v_A = layer.4.attention.delta = D_A(q_A) q_A
h_A = layer.4.post_attention = h + v_A

x_M = layer.4.mlp.normalized_input  (historical X4)
q_M = layer.4.mlp.operator_output
v_M = layer.4.mlp.delta = D_M(q_M) q_M
h_+ = layer.4.output = h_A + v_M  (historical H4)
```

The atlas state cloud is `{h, h_A, h_+}`. Attention generators use `h` and
permitted causal summaries; MLP generators use `h_A`. Historical `X4` is the
pre-feed-forward normalized input and is deliberately not substituted for
the MLP residual source.

Fixed state-only membership uses gate centroids `g_i` learned from each
A-fit token's `(h + h_A + h_+) / 3`.  Those gates assign every row in the
three-stage cloud; the affine chart origin `mu_i` is then the mean of the
assigned full-cloud rows.  Gate centroids, assignments, and chart origins are
identical and receipt-verified across B0--B4 and both MLP partitions.

Gemma RMSNorm computes its scale in float32 with epsilon `1e-6` and the
unit-offset learned parameterization `(1 + weight)`. Each attention head
pre-normalization contribution is formed through its matching `o_proj` input
slice, and every head receives the single scale realized from complete
`q_A`. Each deterministic MLP probe block is formed through its matching
`down_proj` input-column slice, and every block receives the single scale
realized from complete `q_M`. No component is independently normalized.

Consequently the checked identities are:

```text
q_A = sum_j q_A,j                 v_A = sum_j D_A(q_A) q_A,j
q_M = sum_r q_M,r                 v_M = sum_r D_M(q_M) q_M,r
h_A = h + v_A                     h_+ = h_A + v_M
h_+ - h = sum_j v_A,j + sum_r v_M,r
```

All closure diagnostics accumulate in float64. The experiment applies the
same rank-64 empirical Fisher/VJP carrier from `layer.4.output` as a shared
downstream-sensitivity metric; it does not claim exact stage-local Fisher
geometry at `h` or `h_A`.
"""


def _prepare_final_artifacts(
    *,
    args: argparse.Namespace,
    config: ProbeConfig,
    runtime: RuntimeContext,
    candidate: FisherLocalChartFrozenCandidate,
    claim: object,
    fit_report: Mapping[str, object],
    roles: Mapping[str, object],
    guard_rows: RoleRows,
    preclaim_selection_rows: RoleRows,
    fidelity: Mapping[str, object],
    fidelity_forward_count: int,
    fidelity_seconds: float,
    guard_evaluation_seconds: float,
    verdicts: Mapping[str, object],
    bootstrap: Mapping[str, object],
    partition_stability: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, bytes]]:
    carrier_artifact = cast(Any, runtime.carrier_artifact)
    split = cast(Any, runtime.split_manifest)
    invariant_summary = _capture_invariant_summary(fit_report, guard_rows)
    transition_artifact, dot = _transition_artifact(
        roles, candidate, carrier_artifact.carrier, guard_rows
    )
    selection = cast(Mapping[str, object], candidate.bundle["selected_hyperparameters"])
    generator_root = cast(Mapping[str, object], candidate.bundle["generators"])
    router_root = cast(Mapping[str, object], candidate.bundle["routers"])
    baselines_root = cast(Mapping[str, object], candidate.bundle["baselines"])
    all_tensor_bytes = _tensor_storage_report(candidate.bundle)
    fit_capture = cast(Mapping[str, object], fit_report["capture"])
    sequence_counts = {
        "A_fit": int(cast(Mapping[str, object], fit_capture["A_fit"])["sequence_count"]),
        "A_selection": int(
            cast(Mapping[str, object], fit_capture["A_selection"])["sequence_count"]
        ),
        "A_guard": len(set(guard_rows.sequence_ids)),
    }
    resources = {
        "schema": REPORT_SCHEMA,
        "qualification": "analysis storage and parameter accounting; not realized model compression",
        "chart_tensor_elements": _tensor_parameter_count(baselines_root),
        "generator_parameter_elements": _tensor_parameter_count(generator_root),
        "router_parameter_elements": _tensor_parameter_count(router_root),
        "serialized_candidate_bytes": candidate.path.stat().st_size,
        "serialized_tensor_bytes_by_dtype": all_tensor_bytes,
        "peak_host_bytes": _peak_host_bytes(),
        "peak_device_bytes": 0,
        "device": "cpu",
        "sequence_counts": sequence_counts,
        "teacher_capture_forward_count": (
            int(cast(Mapping[str, object], cast(Mapping[str, object], fit_report["capture"])["A_fit"])["teacher_forward_count"])
            + int(cast(Mapping[str, object], cast(Mapping[str, object], fit_report["capture"])["A_selection"])["teacher_forward_count"])
            + preclaim_selection_rows.teacher_forward_count
            + guard_rows.teacher_forward_count
        ),
        "full_sequence_shadow_forward_count": fidelity_forward_count,
        "capture_seconds": {
            "A_fit": cast(Mapping[str, object], cast(Mapping[str, object], fit_report["capture"])["A_fit"])["seconds"],
            "A_selection_fit_phase": cast(Mapping[str, object], cast(Mapping[str, object], fit_report["capture"])["A_selection"])["seconds"],
            "A_selection_preclaim_dry_run": preclaim_selection_rows.capture_seconds,
            "A_guard": guard_rows.capture_seconds,
        },
        "fit_seconds": fit_report["fit_seconds"],
        "evaluation_seconds": {
            "fit_selection": fit_report["evaluation_seconds"],
            "guard": guard_evaluation_seconds,
            "full_sequence_fidelity": fidelity_seconds,
        },
    }
    partition_manifests = {
        scheme: {
            "intermediate_width": manifest.intermediate_width,
            "block_count": manifest.block_count,
            "scheme": manifest.scheme,
            "seed": manifest.seed,
            "block_sizes": manifest.block_sizes,
            "channel_indices": manifest.channel_indices,
        }
        for scheme in PARTITION_SCHEMES
        for manifest in (
            make_mlp_probe_partition(
                2048,
                scheme=cast(Any, scheme),
                block_count=4,
                seed=config.partition_seed,
            ),
        )
    }
    subregions = {
        "schema": REPORT_SCHEMA,
        "attention": {
            "count": 4,
            "source_state": "h",
            "definition": "head context slice after realized attention, passed through the matching o_proj column slice before one shared D_A(q_A)",
            "head_dimension": 256,
            "o_proj_input_width": 1024,
            "formula": "q_A,j=context_j@W_O,j.T; v_A,j=D_A(q_A)q_A,j",
        },
        "mlp": {
            "count": 4,
            "source_state": "h_A",
            "definition": "deterministic exhaustive intermediate-channel probe block after down projection before one shared D_M(q_M)",
            "formula": "q_M,r=down_input[I_r]@W_down[:,I_r].T; v_M,r=D_M(q_M)q_M,r",
            "discovered_functional_regions": False,
        },
        "complete_additivity": "h_plus-h=sum_j(v_A,j)+sum_r(v_M,r)",
        "invariants": invariant_summary,
    }
    transcript = cast(Mapping[str, object], candidate.bundle["selection_transcript"])
    similarity = dict(
        cast(Mapping[str, object], cast(Mapping[str, object], transcript["evaluations"])["fisher_similarity_matrices"])
    )
    similarity["A_guard"] = {
        scheme: _fisher_similarity_matrix(
            guard_rows, carrier_artifact.carrier, scheme=scheme
        )
        for scheme in PARTITION_SCHEMES
    }
    next_experiment = _next_experiment(verdicts)
    experiment_manifest = {
        "schema": SCHEMA,
        "format_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "model": {
            "model_id": DEFAULT_MODEL_ID,
            "resolved_commit": DEFAULT_REVISION,
            "source_model_sha256": cast(Mapping[str, object], candidate.bundle["bindings"])["source_model_sha256"],
            "source_execution_sha256": cast(Mapping[str, object], candidate.bundle["bindings"])["source_execution_sha256"],
            "dtype": "float32",
            "device": "cpu",
        },
        "candidate": {
            "candidate_sha256": candidate.candidate_sha256,
            "candidate_file_sha256": candidate.candidate_file_sha256,
            "protocol_sha256": candidate.protocol_sha256,
            "path": str(candidate.path),
        },
        "protocol_source_sha256s": dict(_IMPORTED_PROTOCOL_SOURCE_SHA256S),
        "claim": cast(Any, claim).metadata(),
        "carrier": carrier_artifact.metadata(),
        "config": config.to_dict(),
        "selected_hyperparameters": dict(selection),
        "scientific_scope": {
            "three_ordered_questions_reported_independently": True,
            "shared_output_fisher_not_exact_earlier_local_fisher": True,
            "full_dense_fisher_materialized": False,
            "calibration_b_used": False,
            "default_executor_modified": False,
            "recurrent_generator_sharing": False,
            "fixed_gate_assignment_and_chart_origin_receipt_verified": True,
            "contains_prompt_text_or_token_ids": False,
            "guard_evaluation_count": 1,
        },
        "artifact_inventory": list(FINAL_ARTIFACT_NAMES),
    }
    artifacts: dict[str, object | str] = {
        "exact_layer4_boundary_design.md": _boundary_design_markdown(),
        "experiment_manifest.json": experiment_manifest,
        "sequence_split_manifest.json": split.to_dict(),
        "subregion_definitions.json": subregions,
        "mlp_partitions.json": {
            "schema": REPORT_SCHEMA,
            "manifests": partition_manifests,
            "stability": dict(partition_stability),
        },
        "chart_rank_tables.json": {
            "schema": REPORT_SCHEMA,
            "selected": dict(selection),
            "state_reconstruction": _evaluations_section(roles, "state_reconstruction"),
            "rank_curves": _evaluations_section(roles, "rank_curves"),
        },
        "tangent_capture_tables.json": {
            "schema": REPORT_SCHEMA,
            "tables": _evaluations_section(roles, "tangent_capture"),
            "bootstrap": dict(bootstrap),
            "verdict": verdicts["fisher_tangent_capture"],
        },
        "affine_generator_tables.json": {
            "schema": REPORT_SCHEMA,
            "tables": _evaluations_section(roles, "affine_generators"),
            "bootstrap": dict(bootstrap),
            "verdict": verdicts["affine_generators"],
        },
        "attention_context_comparison.json": {
            "schema": REPORT_SCHEMA,
            "tables": {
                role: {
                    scheme: {
                        arm: cast(Mapping[str, object], cast(Mapping[str, object], cast(Mapping[str, object], arm_value)["affine_generators"])["component_sums"])["attention_source_only"]
                        | {"causal_context": cast(Mapping[str, object], cast(Mapping[str, object], cast(Mapping[str, object], arm_value)["affine_generators"])["component_sums"])["attention_causal_context"]}
                        for arm, arm_value in cast(Mapping[str, object], scheme_value).items()
                    }
                    for scheme, scheme_value in cast(Mapping[str, object], role_value).items()
                }
                for role, role_value in roles.items()
            },
            "verdict": verdicts["attention_context_sufficiency"],
            "transported_value_sketch_included": False,
        },
        "teacher_composed_comparison.json": {
            "schema": REPORT_SCHEMA,
            "tables": _evaluations_section(roles, "stage_outputs"),
            "support": _evaluations_section(roles, "support"),
            "verdict": verdicts["composition"],
        },
        "chart_support_tables.json": {
            "schema": REPORT_SCHEMA,
            "normal_space_and_tangential_support": True,
            "tables": _evaluations_section(roles, "support"),
        },
        "fisher_similarity_matrices.json": {
            "schema": REPORT_SCHEMA,
            "metric": "shared layer.4.output empirical Fisher rank64",
            "matrices": similarity,
        },
        "typed_transition_matrices.json": transition_artifact,
        "chart_transition.dot": dot,
        "full_sequence_fidelity.json": dict(fidelity),
        "resource_runtime.json": resources,
        "final_report.md": _markdown_report(
            candidate=candidate,
            carrier_metadata=carrier_artifact.metadata(),
            guard_evaluations=cast(Mapping[str, object], roles["A_guard"]),
            fidelity=fidelity,
            verdicts=verdicts,
            partition_stability=partition_stability,
            invariant_summary=invariant_summary,
            sequence_counts=sequence_counts,
            next_experiment=next_experiment,
        ),
        "commands.md": _commands_markdown(args),
    }
    encoded: dict[str, bytes] = {}
    for name, value in artifacts.items():
        encoded[name] = (
            cast(str, value).encode("utf-8")
            if isinstance(value, str)
            else _canonical_json_bytes(_jsonable(value)) + b"\n"
        )
    return {
        "invariant_summary": invariant_summary,
        "resources": resources,
        "next_experiment": next_experiment,
        "transition_artifact": transition_artifact,
    }, encoded


def _run_guard(args: argparse.Namespace, *, config: ProbeConfig) -> dict[str, object]:
    if not bool(getattr(args, "acknowledge_one_shot_guard", False)):
        raise ValueError(
            "guard requires --acknowledge-one-shot-guard before any irreversible claim"
        )
    fit_report_path = args.output_dir / "fit_selection_report.json"
    if not fit_report_path.is_file():
        raise FileNotFoundError("fit-select must complete before the one-shot guard")
    for name in FINAL_ARTIFACT_NAMES:
        if (args.output_dir / name).exists():
            raise FileExistsError(
                f"refusing to claim A-guard while final artifact already exists: {name}"
            )
    runtime = _runtime_context(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        selection_input_path=args.selection_input,
        guard_input_path=args.guard_input,
        carrier_path=args.carrier,
        cache_dir=args.cache_dir,
    )
    carrier_artifact = cast(Any, runtime.carrier_artifact)
    carrier: Rank64FisherCarrier = carrier_artifact.carrier
    split = cast(Any, runtime.split_manifest)
    expected_protocol = _protocol_sha256(
        config,
        split_manifest_sha256=split.manifest_sha256,
        carrier_sha256=carrier_artifact.carrier_sha256,
    )

    def assert_protocol_snapshot() -> None:
        if (
            _protocol_sha256(
                config,
                split_manifest_sha256=split.manifest_sha256,
                carrier_sha256=carrier_artifact.carrier_sha256,
            )
            != expected_protocol
        ):
            raise RuntimeError("protocol identity changed before A-guard claim")

    preflight_candidate = load_fisher_local_chart_frozen_candidate(args.candidate)
    if (
        preflight_candidate.protocol_sha256 != expected_protocol
        or preflight_candidate.split_manifest_sha256 != split.manifest_sha256
        or preflight_candidate.carrier_sha256 != carrier_artifact.carrier_sha256
    ):
        raise ValueError("live code/config/corpus/carrier differs from frozen candidate")
    bindings = cast(Mapping[str, object], preflight_candidate.bundle["bindings"])
    live_model_sha256 = runtime.adapter.model_fingerprint()
    live_execution_sha256 = runtime.adapter.execution_fingerprint()
    if live_model_sha256 != bindings.get("source_model_sha256"):
        raise ValueError("live Gemma weights differ from the frozen candidate")
    if live_execution_sha256 != bindings.get("source_execution_sha256"):
        raise ValueError(
            "live Gemma execution semantics differ from the frozen candidate"
        )
    preflight_identity = _candidate_identity(preflight_candidate)
    fit_report, fit_report_sha256 = _load_fit_report_with_sha256(
        fit_report_path, preflight_candidate
    )

    # This complete selection re-evaluation is intentionally before the
    # claim.  It exercises model capture, frozen tensor reload, generators,
    # support and routers without consulting guard text.
    selection_input = runtime.corpus.open_development_role("calibration_a_selection")
    selection_panel = _materialize_panel(runtime, selection_input)
    selection_rows = _capture_panel(runtime, selection_panel, config=config)

    rehearsal_state: dict[str, object] = {}

    def dry_run(candidate: FisherLocalChartFrozenCandidate) -> None:
        # ``claim_and_open`` reloads the path before invoking this callback.
        # Bind that actual callback object (including serialized file bytes),
        # rather than trusting only the earlier convenience preflight object.
        assert_protocol_snapshot()
        if _candidate_identity(candidate) != preflight_identity:
            raise ValueError("candidate changed between preflight and claim callback")
        callback_report, callback_report_sha256 = _load_fit_report_with_sha256(
            fit_report_path, candidate
        )
        if callback_report_sha256 != fit_report_sha256 or (
            _canonical_json_bytes(_jsonable(callback_report))
            != _canonical_json_bytes(_jsonable(fit_report))
        ):
            raise ValueError("fit/selection report changed before claim callback")
        if (
            runtime.adapter.model_fingerprint() != live_model_sha256
            or runtime.adapter.execution_fingerprint() != live_execution_sha256
        ):
            raise ValueError("Gemma execution state changed before claim callback")
        selection_evaluations = _candidate_model_free_dry_run(
            candidate, carrier, selection_rows
        )
        rehearsal_state.update(
            _selection_preclaim_rehearsal(
                args=args,
                config=config,
                runtime=runtime,
                candidate=candidate,
                fit_report=callback_report,
                selection_panel=selection_panel,
                selection_rows=selection_rows,
                selection_evaluations=selection_evaluations,
            )
        )
        if (
            runtime.adapter.model_fingerprint() != live_model_sha256
            or runtime.adapter.execution_fingerprint() != live_execution_sha256
        ):
            raise ValueError("Gemma execution state changed during preclaim rehearsal")
        if runtime.corpus.guard_opened or runtime.corpus.guard_consumed:
            raise RuntimeError("preclaim rehearsal unexpectedly opened A-guard")
        assert_protocol_snapshot()

    candidate, claim, guard_input = claim_and_open_fisher_local_chart_guard(
        corpus=runtime.corpus,
        candidate_path=args.candidate,
        expected_protocol_sha256=expected_protocol,
        expected_split_manifest_sha256=split.manifest_sha256,
        expected_carrier_sha256=carrier_artifact.carrier_sha256,
        dry_run=dry_run,
    )
    if _candidate_identity(candidate) != preflight_identity:
        raise RuntimeError("claimed candidate differs from the rehearsed candidate")
    if not rehearsal_state:
        raise RuntimeError("claim proceeded without a completed preclaim rehearsal")
    # Any exception beyond this line burns the attempt by design.  Never
    # silently retry this command with the same guard.
    guard_panel = _materialize_panel(runtime, guard_input)
    guard_rows = _capture_panel(runtime, guard_panel, config=config)
    evaluation_started = time.perf_counter()
    guard_evaluations = _evaluate_candidate_role(candidate, carrier, guard_rows)
    guard_evaluation_seconds = time.perf_counter() - evaluation_started
    fidelity, fidelity_forward_count, fidelity_seconds = _full_sequence_fidelity(
        runtime, guard_panel, guard_rows, candidate
    )
    transcript = cast(Mapping[str, object], candidate.bundle["selection_transcript"])
    frozen_evaluations = cast(Mapping[str, object], transcript["evaluations"])
    roles = dict(cast(Mapping[str, object], frozen_evaluations["roles"]))
    roles["A_guard"] = guard_evaluations
    verdicts, bootstrap = _verdicts_and_bootstrap(
        candidate,
        carrier,
        guard_rows,
        guard_evaluations,
        fidelity,
        config=config,
    )
    partition_stability = _partition_stability(guard_evaluations)
    summaries, encoded_artifacts = _prepare_final_artifacts(
        args=args,
        config=config,
        runtime=runtime,
        candidate=candidate,
        claim=claim,
        fit_report=fit_report,
        roles=roles,
        guard_rows=guard_rows,
        preclaim_selection_rows=selection_rows,
        fidelity=fidelity,
        fidelity_forward_count=fidelity_forward_count,
        fidelity_seconds=fidelity_seconds,
        guard_evaluation_seconds=guard_evaluation_seconds,
        verdicts=verdicts,
        bootstrap=bootstrap,
        partition_stability=partition_stability,
    )
    # Every remaining file has already been rendered into bytes before the
    # first result becomes visible, minimizing fallible post-claim work.
    guard_result = write_fisher_local_chart_guard_result(
        path=args.output_dir / "guard_result.json",
        candidate=candidate,
        claim=claim,
        metrics=cast(
            Mapping[str, object],
            _jsonable(
                {
                    "evaluation": guard_evaluations,
                    "full_sequence_fidelity": fidelity,
                    "verdicts": verdicts,
                    "sequence_bootstrap": bootstrap,
                    "partition_stability": partition_stability,
                    "preclaim_rehearsal": rehearsal_state,
                }
            ),
        ),
    )
    for name, encoded in encoded_artifacts.items():
        if name == "guard_result.json":
            continue
        _exclusive_write(args.output_dir / name, encoded)
    return {
        "schema": SCHEMA,
        "phase": "guard",
        "evaluation_count": 1,
        "candidate_sha256": candidate.candidate_sha256,
        "claim_sha256": claim.claim_sha256,
        "guard_result_sha256": guard_result["result_sha256"],
        "verdicts": verdicts,
        "partition_stability": partition_stability,
        "invariants": summaries["invariant_summary"],
        "resources": summaries["resources"],
        "preclaim_rehearsal": rehearsal_state,
        "next_experiment": summaries["next_experiment"],
        "artifact_directory": str(args.output_dir),
    }


def _config_from_args(args: argparse.Namespace) -> ProbeConfig:
    return ProbeConfig(
        chart_count=args.chart_count,
        coordinate_rank=args.coordinate_rank,
        ridge_grid=tuple(args.ridge_grid),
        support_coverages=tuple(args.support_coverages),
        chart_seed=args.chart_seed,
        partition_seed=args.partition_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus-artifact", type=Path, default=DEFAULT_CORPUS_ARTIFACT)
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_FIT_INPUT)
    parser.add_argument("--selection-input", type=Path, default=DEFAULT_SELECTION_INPUT)
    parser.add_argument("--guard-input", type=Path, default=DEFAULT_GUARD_INPUT)
    parser.add_argument("--fisher-source", type=Path, default=DEFAULT_FISHER_SOURCE)
    parser.add_argument("--carrier", type=Path, default=DEFAULT_CARRIER_SIDECAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--chart-count", type=int, default=DEFAULT_CHART_COUNT)
    parser.add_argument("--coordinate-rank", type=int, default=DEFAULT_COORDINATE_RANK)
    parser.add_argument(
        "--ridge-grid", type=float, nargs="+", default=list(DEFAULT_RIDGE_GRID)
    )
    parser.add_argument(
        "--support-coverages",
        type=float,
        nargs="+",
        default=list(DEFAULT_SUPPORT_COVERAGES),
    )
    parser.add_argument("--chart-seed", type=int, default=DEFAULT_CHART_SEED)
    parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the stage-aware Gemma layer-4 Fisher local-chart probe."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "extract-carrier", "smoke", "fit-select", "guard", "run"):
        child = subparsers.add_parser(name)
        _add_common_arguments(child)
        if name in {"guard", "run"}:
            child.add_argument(
                "--acknowledge-one-shot-guard",
                action="store_true",
                help="explicitly acknowledge the irreversible manifest-global A-guard claim",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"guard", "run"} and not args.acknowledge_one_shot_guard:
        parser.error(
            f"{args.command} requires --acknowledge-one-shot-guard because it makes an irreversible A-guard claim"
        )
    if args.output_dir != DEFAULT_OUTPUT_DIR and args.candidate == DEFAULT_CANDIDATE:
        args.candidate = args.output_dir / "frozen_candidate.pt"
    config = _config_from_args(args)
    if args.command == "extract-carrier":
        result = _run_extract_carrier(args)
    elif args.command == "inspect":
        result = _run_inspect(args)
    elif args.command == "smoke":
        result = _run_smoke(args, config=config)
    elif args.command == "fit-select":
        result = _run_fit_select(args, config=config)
    elif args.command == "guard":
        result = _run_guard(args, config=config)
    elif args.command == "run":
        phases = {}
        phases["extract-carrier"] = _run_extract_carrier(args)
        phases["inspect"] = _run_inspect(args)
        phases["smoke"] = _run_smoke(args, config=config)
        phases["fit-select"] = _run_fit_select(args, config=config)
        phases["guard"] = _run_guard(args, config=config)
        result = {"schema": SCHEMA, "phase": "run", "phases": phases}
    else:  # pragma: no cover - argparse enforces this branch.
        raise AssertionError(args.command)
    print(json.dumps(_jsonable(result), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
