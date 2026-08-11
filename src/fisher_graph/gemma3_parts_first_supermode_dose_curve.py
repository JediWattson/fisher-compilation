"""Frozen exploratory post-C alpha curve for the parts-first pilot.

This runner does not discover, refit, or reselect a supermode.  It authenticates
the published model capture and pilot report, deterministically replays their
A/B selection, and then applies the already-selected four-stage native gate to
the same twelve C rows at ``alpha = .99, .97, .90, .50``.  ``alpha = 1`` is a
literal native no-op control, so the serialized schedule is
``[1.00, .99, .97, .90, .50]``.

The result is an exploratory post-C diagnostic, not fresh confirmation.  In
particular, the original chart-only intervention mask is preserved even though
the selected chart contains both path-active and path-inactive rows.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Final, Protocol

from .gemma3_parts_first_native_intervention import (
    NativePartGateExecutionReceipt,
)
from .gemma3_parts_first_supermode_experiment import (
    CaptureRole,
    FiniteInterventionEvidence,
    FrozenChartAtlas,
    FrozenSupermodeArtifact,
    PartsFirstCaptureBundle,
    PartsFirstPilotConfig,
    _canonical_catalog,
    _confirm_on_B,
    _discover_on_A,
    _path_active,
    validate_parts_first_supermode_report,
)
from .gemma3_parts_first_supermode_model_runner import (
    DEFAULT_CAPTURE_OUTPUT,
    DEFAULT_REPORT_OUTPUT,
    PINNED_PANEL_PATH,
    PartsFirstLiveModelPilot,
    PartsFirstModelRunnerConfig,
    _receipt_set_sha256,
    native_gates_for_frozen_artifact,
)


DOSE_CURVE_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.post_c_alpha_dose_curve.v1"
)
DOSE_CURVE_FORMAT_VERSION: Final = 1
FROZEN_FINITE_ALPHAS: Final = (0.99, 0.97, 0.9, 0.5)
FROZEN_SCHEDULE: Final = (1.0, *FROZEN_FINITE_ALPHAS)
LOCAL_LINEAR_ALPHAS: Final = (0.99, 0.97)
DEFAULT_DOSE_CURVE_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "parts-first-supermode-post-c-alpha-dose-curve-v1.json"
)

# These identify the one published pilot this post-C diagnostic is allowed to
# reuse.  A different capture or selection needs a new, explicitly versioned
# protocol rather than a command-line substitution.
PUBLISHED_CAPTURE_SHA256: Final = (
    "0598ddf5ab0454d023f19e20803a671b253ea35984aabe45bf15256ee9565b72"
)
PUBLISHED_REPORT_SHA256: Final = (
    "4592cf43a21e785126d5abdc0a013c0d3fd7c553e76f21b89535d73a14ab3dbb"
)
PUBLISHED_CANDIDATE_SHA256: Final = (
    "69949bf62566f4f564a3ef9f8919e07ae1733a827658ec9976033996b41ba2a3"
)
PUBLISHED_CHART_ATLAS_SHA256: Final = (
    "2d056e28ba783e1b482307c6bfa848ae6bf6f12dae984a951fbc2daaae26b298"
)
PUBLISHED_CONFIG_SHA256: Final = (
    "48451a98059ea5ab1568362e69b7c906a1265cefa148b70e88b060cf9c4d8b61"
)
PUBLISHED_CANDIDATE_CATALOG_SHA256: Final = (
    "11e2159b532245da1db3b1703cdac2ef854e190467f9a2db86c66964b844ca7d"
)
PUBLISHED_C_ROLE_SHA256: Final = (
    "14f8e213ff9fd7ab3c75cfab94daabc34e6c07fddfa0e66907655f971855d4e3"
)
PUBLISHED_C_ROW_IDS_SHA256: Final = (
    "fb1d810594d485a8a552a0502d6866bba158c14190d6fcbc2cb31e4efd1ed271"
)
PUBLISHED_C_EVIDENCE_SHA256: Final = (
    "6c265fb1c7958f01181f8e87f0b934abd145fc68c74d8a1b94e37a00e481e448"
)
PUBLISHED_C_GATE_RECEIPT_SHA256: Final = (
    "49f24a89d3f86b2ef47a6f005f501667b462a672010507139925b67a3203c3e1"
)
PUBLISHED_SOURCE_MODEL_SHA256: Final = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
PUBLISHED_SOURCE_EXECUTION_SHA256: Final = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
PUBLISHED_EXACT_CONTROL_EVIDENCE_SHA256: Final = (
    "f07c4d66832de3a0de0ea0191537b988cb2fc930fd7ee3d28a74ee424e851fe3"
)
PUBLISHED_CHART_ID: Final = "chart.0"
PUBLISHED_C_ROWS: Final = (
    ("addition-09", "integer_addition"),
    ("addition-10", "integer_addition"),
    ("subtraction-09", "integer_subtraction"),
    ("subtraction-10", "integer_subtraction"),
    ("comparison-09", "integer_comparison"),
    ("comparison-10", "integer_comparison"),
    ("antonym-09", "lexical_antonym"),
    ("antonym-10", "lexical_antonym"),
    ("category-09", "category_completion"),
    ("category-10", "category_completion"),
    ("capital-09", "world_capitals"),
    ("capital-10", "world_capitals"),
)
PUBLISHED_PATH_ACTIVE_ROW_IDS: Final = (
    "addition-09",
    "addition-10",
    "subtraction-09",
)
PUBLISHED_STAGE_DIRECTIONAL_L2: Final = (
    0.07058111701278755,
    0.739295431947124,
    3.114909454948369,
    26.258520870651854,
)
PUBLISHED_TOTAL_DIRECTIONAL_L2: Final = 28.688246356446093

_REPORT_DOMAIN = b"fisher_graph:parts-first-post-c-dose-curve-report:v1\0"
_NUMERICAL_EPSILON_FLOOR: Final = 5.0e-5
_MIXED_RELATIVE_TOLERANCE: Final = 0.05
_LOCAL_LINEAR_RELATIVE_L2_LIMIT: Final = 0.05


__all__ = [
    "DEFAULT_DOSE_CURVE_OUTPUT",
    "DOSE_CURVE_SCHEMA",
    "FROZEN_FINITE_ALPHAS",
    "FROZEN_SCHEDULE",
    "FrozenPublishedPilot",
    "PartsFirstDoseCurveConfig",
    "PartsFirstPostCDoseCurveReport",
    "build_post_c_dose_curve_preflight",
    "load_and_replay_published_pilot",
    "main",
    "run_parts_first_post_c_dose_curve",
    "validate_post_c_dose_curve_report",
    "write_post_c_dose_curve_report",
]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object, *, domain: bytes = b"") -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_validated_source_report(path: Path) -> dict[str, object]:
    value = _load_strict_json_object(
        path,
        label="published parts-first pilot report",
    )
    return validate_parts_first_supermode_report(value)


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_strict_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _load_published_capture(path: Path) -> PartsFirstCaptureBundle:
    return PartsFirstCaptureBundle.from_dict(
        _load_strict_json_object(path, label="published parts-first capture")
    )


@dataclass(frozen=True, slots=True)
class PartsFirstDoseCurveConfig:
    capture_path: Path = DEFAULT_CAPTURE_OUTPUT
    source_report_path: Path = DEFAULT_REPORT_OUTPUT
    output_path: Path = DEFAULT_DOSE_CURVE_OUTPUT
    panel_path: Path = PINNED_PANEL_PATH
    cache_dir: Path | None = None
    maximum_prompt_tokens: int = 128

    def __post_init__(self) -> None:
        for name in (
            "capture_path",
            "source_report_path",
            "output_path",
            "panel_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if any(
            getattr(self, name).suffix != ".json"
            for name in ("capture_path", "source_report_path", "output_path")
        ):
            raise ValueError("dose-curve inputs and output must use .json")
        if len({self.capture_path, self.source_report_path, self.output_path}) != 3:
            raise ValueError("dose-curve input and output paths must be distinct")
        if (
            type(self.maximum_prompt_tokens) is not int
            or self.maximum_prompt_tokens < 2
        ):
            raise ValueError("maximum_prompt_tokens must be at least two")


@dataclass(frozen=True, slots=True)
class FrozenPublishedPilot:
    capture: PartsFirstCaptureBundle
    source_report: Mapping[str, object]
    pilot_config: PartsFirstPilotConfig
    chart_atlas: FrozenChartAtlas
    artifact: FrozenSupermodeArtifact
    C_role: CaptureRole


def _configuration_sha256(config: PartsFirstPilotConfig) -> str:
    return _sha256(config.to_dict())


def load_and_replay_published_pilot(
    config: PartsFirstDoseCurveConfig = PartsFirstDoseCurveConfig(),
) -> FrozenPublishedPilot:
    """Authenticate the published artifacts and deterministically replay A/B."""

    bundle = _load_published_capture(config.capture_path)
    report = _load_validated_source_report(config.source_report_path)
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("published report omitted its configuration")
    pilot_config = PartsFirstPilotConfig.from_dict(configuration)
    roles = bundle.role_by_id
    discovery = _discover_on_A(
        role=roles["A_discovery"],
        axis=bundle.response_axis,
        catalog=_canonical_catalog(),
        config=pilot_config,
    )
    confirmation = _confirm_on_B(
        role=roles["B_confirmation"],
        axis=bundle.response_axis,
        discovery=discovery,
        config=pilot_config,
    )
    if discovery.to_dict() != report.get("A_discovery"):
        raise ValueError("deterministic A replay differs from the published report")
    if confirmation.to_dict() != report.get("B_confirmation"):
        raise ValueError("deterministic B replay differs from the published report")

    selected = confirmation.selected_candidate_sha256
    if selected is None:
        raise ValueError("published B replay did not select a candidate")
    artifact = next(
        (item for item in discovery.candidates if item.candidate_sha256 == selected),
        None,
    )
    if artifact is None:
        raise ValueError("published B selection is absent from replayed A")
    C_role = roles["C_intervention"]
    c_receipt = report.get("C_intervention")
    capture_section = report.get("capture")
    preflight = report.get("preflight")
    if not all(
        isinstance(item, Mapping)
        for item in (c_receipt, capture_section, preflight)
    ):
        raise ValueError("published report omitted a required binding section")

    expected_hashes = {
        "capture": (bundle.capture_sha256, PUBLISHED_CAPTURE_SHA256),
        "report": (report.get("report_sha256"), PUBLISHED_REPORT_SHA256),
        "candidate": (selected, PUBLISHED_CANDIDATE_SHA256),
        "chart atlas": (
            discovery.chart_atlas.atlas_sha256,
            PUBLISHED_CHART_ATLAS_SHA256,
        ),
        "configuration": (
            _configuration_sha256(pilot_config),
            PUBLISHED_CONFIG_SHA256,
        ),
        "candidate catalog": (
            discovery.candidate_catalog_sha256,
            PUBLISHED_CANDIDATE_CATALOG_SHA256,
        ),
        "C role": (C_role.role_sha256, PUBLISHED_C_ROLE_SHA256),
        "C row IDs": (C_role.row_ids_sha256, PUBLISHED_C_ROW_IDS_SHA256),
        "source model": (
            bundle.source_model_fingerprint_sha256,
            PUBLISHED_SOURCE_MODEL_SHA256,
        ),
        "source execution": (
            bundle.source_execution_fingerprint_sha256,
            PUBLISHED_SOURCE_EXECUTION_SHA256,
        ),
    }
    drift = [
        label for label, (observed, expected) in expected_hashes.items()
        if observed != expected
    ]
    if drift:
        raise ValueError(f"published dose-curve binding drifted: {', '.join(drift)}")
    if (
        capture_section.get("capture_sha256") != bundle.capture_sha256
        or capture_section.get("source_model_fingerprint_sha256")
        != bundle.source_model_fingerprint_sha256
        or capture_section.get("source_execution_fingerprint_sha256")
        != bundle.source_execution_fingerprint_sha256
        or preflight.get("configuration_sha256") != PUBLISHED_CONFIG_SHA256
        or c_receipt.get("candidate_sha256") != selected
        or c_receipt.get("C_role_sha256") != C_role.role_sha256
        or c_receipt.get("evidence_sha256") != PUBLISHED_C_EVIDENCE_SHA256
        or c_receipt.get("native_gate_execution_receipt_sha256")
        != PUBLISHED_C_GATE_RECEIPT_SHA256
        or c_receipt.get("selected_chart_row_count") != 12
        or artifact.candidate.chart_id != PUBLISHED_CHART_ID
        or bundle.source_kind != "model_capture"
        or not bundle.exact_lift_receipt.passed
    ):
        raise ValueError("published source report and capture do not close")
    observed_rows = tuple((row.row_id, row.family_id) for row in C_role.rows)
    if observed_rows != PUBLISHED_C_ROWS:
        raise ValueError("published C row order or family lineage drifted")
    coordinate_index = {
        (item.global_atom_ordinal, item.execution_ordinal): index
        for index, item in enumerate(bundle.response_axis)
    }
    path_active_ids = tuple(
        row.row_id
        for row in C_role.rows
        if _path_active(
            row,
            artifact.candidate,
            discovery.chart_atlas,
            coordinate_index,
        )
    )
    if path_active_ids != PUBLISHED_PATH_ACTIVE_ROW_IDS:
        raise ValueError("published C path-active stratum drifted")
    return FrozenPublishedPilot(
        capture=bundle,
        source_report=report,
        pilot_config=pilot_config,
        chart_atlas=discovery.chart_atlas,
        artifact=artifact,
        C_role=C_role,
    )


def build_post_c_dose_curve_preflight(
    config: PartsFirstDoseCurveConfig = PartsFirstDoseCurveConfig(),
) -> dict[str, object]:
    frozen = load_and_replay_published_pilot(config)
    return {
        "schema": "fisher_graph.gemma3_parts_first.post_c_dose_curve_preflight.v1",
        "ready_to_load": True,
        "model_loaded": False,
        "local_files_only": True,
        "capture_sha256": frozen.capture.capture_sha256,
        "source_report_sha256": frozen.source_report["report_sha256"],
        "selected_candidate_sha256": frozen.artifact.candidate_sha256,
        "chart_atlas_sha256": frozen.chart_atlas.atlas_sha256,
        "configuration_sha256": _configuration_sha256(frozen.pilot_config),
        "C_role_sha256": frozen.C_role.role_sha256,
        "C_row_ids_sha256": frozen.C_role.row_ids_sha256,
        "C_row_count": len(frozen.C_role.rows),
        "path_active_row_count": len(PUBLISHED_PATH_ACTIVE_ROW_IDS),
        "path_inactive_chart_only_row_count": (
            len(frozen.C_role.rows) - len(PUBLISHED_PATH_ACTIVE_ROW_IDS)
        ),
        "frozen_schedule": list(FROZEN_SCHEDULE),
        "A_B_selection_replayed": True,
        "fit_refit_or_reselection_count": 0,
        "chart_only_mask_preserved": True,
        "exploratory_post_C": True,
        "fresh_confirmation_claimed": False,
    }


class _DoseCurveRuntime(Protocol):
    adapter: object
    exact_lift_precheck: object
    panel_roles: tuple[object, ...]
    batches_by_example_id: dict[str, object]
    _capture_bundle: PartsFirstCaptureBundle | None

    def _score_batch(
        self,
        batch: object,
        *,
        gates: tuple[object, ...] | None,
    ) -> tuple[
        float,
        tuple[float, ...],
        NativePartGateExecutionReceipt | None,
    ]: ...

    def _first_order_delta(
        self,
        artifact: FrozenSupermodeArtifact,
        row: object,
        *,
        alpha: float,
    ) -> float: ...


def _adapter_fingerprints(runtime: _DoseCurveRuntime) -> tuple[str, str]:
    model_fingerprint = getattr(runtime.adapter, "model_fingerprint", None)
    execution_fingerprint = getattr(runtime.adapter, "execution_fingerprint", None)
    if not callable(model_fingerprint) or not callable(execution_fingerprint):
        raise TypeError("dose-curve runtime adapter omitted fingerprint methods")
    return str(model_fingerprint()), str(execution_fingerprint())


def _bind_published_capture_to_runtime(
    runtime: _DoseCurveRuntime,
    frozen: FrozenPublishedPilot,
) -> None:
    model_sha, execution_sha = _adapter_fingerprints(runtime)
    exact_receipt_sha = getattr(
        runtime.exact_lift_precheck,
        "receipt_sha256",
        None,
    )
    if (
        model_sha != frozen.capture.source_model_fingerprint_sha256
        or execution_sha != frozen.capture.source_execution_fingerprint_sha256
        or exact_receipt_sha
        != frozen.capture.exact_lift_receipt.exact_control_evidence_sha256
    ):
        raise RuntimeError("loaded runtime differs from the published capture")
    role_examples = {
        getattr(role, "role", ""): tuple(
            getattr(example, "example_id", "")
            for example in getattr(role, "examples", ())
        )
        for role in runtime.panel_roles
    }
    expected_C_ids = tuple(row.row_id for row in frozen.C_role.rows)
    if (
        role_examples.get("C_intervention") != expected_C_ids
        or any(row_id not in runtime.batches_by_example_id for row_id in expected_C_ids)
    ):
        raise RuntimeError("loaded runtime does not reproduce the published C panel")
    if runtime._capture_bundle is not None:
        raise RuntimeError("dose-curve runtime already has a capture attached")
    runtime._capture_bundle = frozen.capture


def _native_receipt_to_dict(
    receipt: NativePartGateExecutionReceipt,
) -> dict[str, object]:
    if not isinstance(receipt, NativePartGateExecutionReceipt):
        raise TypeError("native gate result omitted its typed receipt")
    return {
        "gate_states": list(receipt.gate_states),
        "hook_call_counts": list(receipt.hook_call_counts),
        "hook_execution_order": list(receipt.hook_execution_order),
        "mask_true_counts": list(receipt.mask_true_counts),
        "alpha_one_control": receipt.alpha_one_control,
        "hooks_removed": receipt.hooks_removed,
        "weights_mutated_by_hooks": receipt.weights_mutated_by_hooks,
        "later_native_computation_reran": (
            receipt.later_native_computation_reran
        ),
        "receipt_sha256": receipt.receipt_sha256,
    }


def _expected_published_gate_states(
    alpha: float,
) -> tuple[dict[str, object], ...]:
    static = (
        (
            0,
            "tied_lexical_hidden_column",
            "lexical_ingress",
            "lexical_ingress",
            None,
            [283, 439],
        ),
        (21, "attention_qo_head", "layer.10.attention", "attention", 10, [0, 1]),
        (36, "mlp_channel", "layer.17.mlp", "mlp", 17, [435, 1853]),
        (
            38,
            "tied_lexical_hidden_column",
            "lexical_readout",
            "lexical_readout",
            None,
            [393, 585],
        ),
    )
    return tuple(
        {
            "gate_id": (
                f"gate.{PUBLISHED_CANDIDATE_SHA256[:16]}.event.{execution}"
            ),
            "family": family,
            "boundary": {
                "boundary_id": boundary_id,
                "boundary_kind": boundary_kind,
                "execution_ordinal": execution,
                "layer_ordinal": layer,
            },
            "part_ordinals": parts,
            "alpha": alpha,
            "chart_id": PUBLISHED_CHART_ID,
        }
        for execution, family, boundary_id, boundary_kind, layer, parts in static
    )


def _native_receipt_from_dict(
    value: Mapping[str, object],
) -> NativePartGateExecutionReceipt:
    _exact_keys(
        value,
        {
            "gate_states",
            "hook_call_counts",
            "hook_execution_order",
            "mask_true_counts",
            "alpha_one_control",
            "hooks_removed",
            "weights_mutated_by_hooks",
            "later_native_computation_reran",
            "receipt_sha256",
        },
        label="native gate receipt",
    )
    gate_states = value["gate_states"]
    if not isinstance(gate_states, list) or not all(
        isinstance(item, Mapping) for item in gate_states
    ):
        raise ValueError("native gate receipt states are invalid")
    sequence_fields = (
        "hook_call_counts",
        "hook_execution_order",
        "mask_true_counts",
    )
    if any(not isinstance(value[name], list) for name in sequence_fields):
        raise ValueError("native gate receipt arrays are invalid")
    return NativePartGateExecutionReceipt(
        gate_states=tuple(dict(item) for item in gate_states),
        hook_call_counts=tuple(value["hook_call_counts"]),  # type: ignore[arg-type]
        hook_execution_order=tuple(  # type: ignore[arg-type]
            value["hook_execution_order"]
        ),
        mask_true_counts=tuple(value["mask_true_counts"]),  # type: ignore[arg-type]
        alpha_one_control=value["alpha_one_control"],  # type: ignore[arg-type]
        hooks_removed=value["hooks_removed"],  # type: ignore[arg-type]
        weights_mutated_by_hooks=value[  # type: ignore[arg-type]
            "weights_mutated_by_hooks"
        ],
        later_native_computation_reran=value[  # type: ignore[arg-type]
            "later_native_computation_reran"
        ],
        receipt_sha256=str(value["receipt_sha256"]),
    )


@dataclass(frozen=True, slots=True)
class _SweepExecution:
    source_model_before: str
    source_model_after: str
    source_execution_before: str
    source_execution_after: str
    fresh_baseline_scores: tuple[float, ...]
    fresh_baseline_chart_ids: tuple[str, ...]
    alpha_one_scores: tuple[float, ...]
    alpha_one_chart_ids: tuple[str, ...]
    baseline_repeat_max_abs_error: float
    alpha_one_max_abs_error: float
    numerical_epsilon: float
    alpha_one_receipts: tuple[NativePartGateExecutionReceipt, ...]
    finite_evidences: tuple[FiniteInterventionEvidence, ...]
    finite_receipts: tuple[tuple[NativePartGateExecutionReceipt, ...], ...]


def _run_frozen_native_sweep(
    runtime: _DoseCurveRuntime,
    frozen: FrozenPublishedPilot,
) -> _SweepExecution:
    source_model_before, source_execution_before = _adapter_fingerprints(runtime)
    role = frozen.C_role
    atlas = frozen.chart_atlas
    artifact = frozen.artifact
    baseline = tuple(row.target_score for row in role.rows)
    baseline_charts = tuple(
        atlas.chart_id(row.chart_coordinates) for row in role.rows
    )
    selected = tuple(
        chart_id == artifact.candidate.chart_id for chart_id in baseline_charts
    )
    if selected != (True,) * len(role.rows):
        raise RuntimeError("published chart-only C mask no longer selects all 12 rows")

    alpha_one_gates = native_gates_for_frozen_artifact(artifact, alpha=1.0)
    fresh_scores: list[float] = []
    fresh_charts: list[str] = []
    noop_scores: list[float] = []
    noop_charts: list[str] = []
    noop_receipts: list[NativePartGateExecutionReceipt] = []
    for row, expected_chart in zip(role.rows, baseline_charts, strict=True):
        batch = runtime.batches_by_example_id[row.row_id]
        fresh_score, fresh_chart, fresh_receipt = runtime._score_batch(
            batch,
            gates=None,
        )
        if fresh_receipt is not None:
            raise RuntimeError("ungated baseline unexpectedly emitted a gate receipt")
        routed_fresh_chart = atlas.chart_id(fresh_chart)
        if routed_fresh_chart != expected_chart:
            raise RuntimeError("fresh baseline changed the frozen C chart route")
        noop_score, noop_chart, noop_receipt = runtime._score_batch(
            batch,
            gates=alpha_one_gates,
        )
        if noop_receipt is None:
            raise RuntimeError("alpha-one control omitted its native gate receipt")
        fresh_scores.append(fresh_score)
        fresh_charts.append(routed_fresh_chart)
        noop_scores.append(noop_score)
        noop_charts.append(atlas.chart_id(noop_chart))
        noop_receipts.append(noop_receipt)

    baseline_repeat_error = max(
        abs(observed - expected)
        for observed, expected in zip(fresh_scores, baseline, strict=True)
    )
    alpha_one_error = max(
        abs(noop - fresh)
        for noop, fresh in zip(noop_scores, fresh_scores, strict=True)
    )
    numerical_epsilon = max(
        _NUMERICAL_EPSILON_FLOOR,
        10.0 * max(baseline_repeat_error, alpha_one_error),
    )

    evidences: list[FiniteInterventionEvidence] = []
    receipts_by_alpha: list[tuple[NativePartGateExecutionReceipt, ...]] = []
    for alpha in FROZEN_FINITE_ALPHAS:
        finite_gates = native_gates_for_frozen_artifact(artifact, alpha=alpha)
        intervened: list[float] = []
        online_charts: list[str] = []
        predicted: list[float] = []
        finite_receipts: list[NativePartGateExecutionReceipt] = []
        receipt_set: list[
            tuple[str, str, NativePartGateExecutionReceipt]
        ] = []
        for row, noop_receipt in zip(role.rows, noop_receipts, strict=True):
            batch = runtime.batches_by_example_id[row.row_id]
            score, chart, receipt = runtime._score_batch(
                batch,
                gates=finite_gates,
            )
            if receipt is None:
                raise RuntimeError("finite gate omitted its native receipt")
            intervened.append(score)
            online_charts.append(atlas.chart_id(chart))
            predicted.append(
                runtime._first_order_delta(artifact, row, alpha=alpha)
            )
            finite_receipts.append(receipt)
            receipt_set.extend(
                (
                    (row.row_id, "alpha_one", noop_receipt),
                    (row.row_id, "finite", receipt),
                )
            )
        model_after_alpha, execution_after_alpha = _adapter_fingerprints(runtime)
        evidence = FiniteInterventionEvidence(
            candidate_sha256=artifact.candidate_sha256,
            c_row_ids_sha256=role.row_ids_sha256,
            gate_alpha=alpha,
            baseline_scores=baseline,
            intervened_scores=tuple(intervened),
            first_order_predicted_score_deltas=tuple(predicted),
            baseline_chart_ids=baseline_charts,
            online_chart_ids_after=tuple(online_charts),
            native_noop_max_abs_error=max(
                baseline_repeat_error,
                alpha_one_error,
            ),
            finite_intervention_candidate_count=1,
            finite_intervention_forward_count=len(role.rows),
            executor_kind="live_model_bridge",
            native_gate_execution_receipt_sha256=(
                _receipt_set_sha256(receipt_set)
            ),
            source_model_fingerprint_before=source_model_before,
            source_model_fingerprint_after=model_after_alpha,
            source_execution_fingerprint_before=source_execution_before,
            source_execution_fingerprint_after=execution_after_alpha,
        )
        evidences.append(evidence)
        receipts_by_alpha.append(tuple(finite_receipts))

    source_model_after, source_execution_after = _adapter_fingerprints(runtime)
    if (
        source_model_after != source_model_before
        or source_execution_after != source_execution_before
    ):
        raise RuntimeError("dose curve changed source model or execution state")
    return _SweepExecution(
        source_model_before=source_model_before,
        source_model_after=source_model_after,
        source_execution_before=source_execution_before,
        source_execution_after=source_execution_after,
        fresh_baseline_scores=tuple(fresh_scores),
        fresh_baseline_chart_ids=tuple(fresh_charts),
        alpha_one_scores=tuple(noop_scores),
        alpha_one_chart_ids=tuple(noop_charts),
        baseline_repeat_max_abs_error=baseline_repeat_error,
        alpha_one_max_abs_error=alpha_one_error,
        numerical_epsilon=numerical_epsilon,
        alpha_one_receipts=tuple(noop_receipts),
        finite_evidences=tuple(evidences),
        finite_receipts=tuple(receipts_by_alpha),
    )


def _path_activity_by_row(
    frozen: FrozenPublishedPilot,
) -> dict[str, bool]:
    coordinate_index = {
        (item.global_atom_ordinal, item.execution_ordinal): index
        for index, item in enumerate(frozen.capture.response_axis)
    }
    return {
        row.row_id: _path_active(
            row,
            frozen.artifact.candidate,
            frozen.chart_atlas,
            coordinate_index,
        )
        for row in frozen.C_role.rows
    }


def _stage_directional_l2_inventory(
    frozen: FrozenPublishedPilot,
) -> list[dict[str, object]]:
    coordinate_index = {
        (item.global_atom_ordinal, item.execution_ordinal): index
        for index, item in enumerate(frozen.capture.response_axis)
    }
    vectors: list[tuple[float, ...]] = []
    for stage in frozen.artifact.candidate.stages:
        values = tuple(
            sum(
                row.atom_responses[
                    coordinate_index[(ordinal, stage.execution_ordinal)]
                ]
                for ordinal in stage.atom_ordinals
            )
            for row in frozen.C_role.rows
        )
        vectors.append(values)
    norms = [
        math.sqrt(sum(value * value for value in values))
        for values in vectors
    ]
    total_vector = tuple(sum(values) for values in zip(*vectors, strict=True))
    total_directional_l2 = math.sqrt(
        sum(value * value for value in total_vector)
    )
    if total_directional_l2 <= 0.0:
        raise RuntimeError("selected path has no C directional response")
    dominant = max(range(len(norms)), key=norms.__getitem__)
    return [
        {
            "stage_index": index,
            "family_id": stage.family.value,
            "boundary_id": stage.boundary.boundary_id,
            "execution_ordinal": stage.execution_ordinal,
            "atom_ordinals": list(stage.atom_ordinals),
            "directional_l2": norms[index],
            "directional_l2_share": norms[index] / total_directional_l2,
            "dominant": index == dominant,
        }
        for index, stage in enumerate(frozen.artifact.candidate.stages)
    ]


def _raw_finite_rows(
    frozen: FrozenPublishedPilot,
    evidence: FiniteInterventionEvidence,
    *,
    numerical_epsilon: float,
) -> list[dict[str, object]]:
    activity = _path_activity_by_row(frozen)
    alpha = evidence.gate_alpha
    rows: list[dict[str, object]] = []
    for index, row in enumerate(frozen.C_role.rows):
        baseline = evidence.baseline_scores[index]
        intervened = evidence.intervened_scores[index]
        actual = intervened - baseline
        predicted = evidence.first_order_predicted_score_deltas[index]
        signal = max(abs(actual), abs(predicted))
        informative = signal >= 10.0 * numerical_epsilon
        tolerance = (
            numerical_epsilon
            + _MIXED_RELATIVE_TOLERANCE * abs(predicted)
        )
        mixed_pass = abs(actual - predicted) <= tolerance
        sign_agreement = (
            None
            if not informative
            else bool(
                (actual > 0.0 and predicted > 0.0)
                or (actual < 0.0 and predicted < 0.0)
            )
        )
        rows.append(
            {
                "row_id": row.row_id,
                "family_id": row.family_id,
                "chart_selected": (
                    evidence.baseline_chart_ids[index]
                    == frozen.artifact.candidate.chart_id
                ),
                "path_active": activity[row.row_id],
                "baseline_score": baseline,
                "intervened_score": intervened,
                "actual_delta": actual,
                "first_order_predicted_delta": predicted,
                "actual_slope_from_alpha_one": actual / (alpha - 1.0),
                "first_order_predicted_slope": predicted / (alpha - 1.0),
                "baseline_chart_id": evidence.baseline_chart_ids[index],
                "online_chart_id": evidence.online_chart_ids_after[index],
                "router_switched": (
                    evidence.baseline_chart_ids[index]
                    != evidence.online_chart_ids_after[index]
                ),
                "signal_scale": signal,
                "informative": informative,
                "mixed_tolerance": tolerance,
                "mixed_tolerance_passed": mixed_pass,
                "sign_agreement": sign_agreement,
            }
        )
    return rows


def _aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    numerical_epsilon: float,
    router_stability_informative: bool,
    subgroup_note: str,
    attenuation_from_alpha_one: float = 1.0,
    include_family_breakdown: bool = True,
) -> dict[str, object]:
    if not rows:
        raise ValueError("dose-curve stratum must contain at least one row")
    actual = tuple(_finite(row["actual_delta"], label="actual delta") for row in rows)
    predicted = tuple(
        _finite(
            row["first_order_predicted_delta"],
            label="predicted delta",
        )
        for row in rows
    )
    errors = tuple(a - p for a, p in zip(actual, predicted, strict=True))
    actual_l2 = math.sqrt(sum(value * value for value in actual))
    predicted_l2 = math.sqrt(sum(value * value for value in predicted))
    error_l2 = math.sqrt(sum(value * value for value in errors))
    attenuation = _finite(
        attenuation_from_alpha_one,
        label="attenuation_from_alpha_one",
    )
    if attenuation <= 0.0:
        raise ValueError("attenuation_from_alpha_one must be positive")
    l2_floor = numerical_epsilon * math.sqrt(len(rows))
    relative_l2 = error_l2 / max(predicted_l2, l2_floor)
    cosine = (
        None
        if actual_l2 <= l2_floor or predicted_l2 <= l2_floor
        else sum(a * p for a, p in zip(actual, predicted, strict=True))
        / (actual_l2 * predicted_l2)
    )
    informative = tuple(row for row in rows if row["informative"] is True)
    sign_values = tuple(
        row["sign_agreement"]
        for row in informative
        if isinstance(row["sign_agreement"], bool)
    )
    switch_count = sum(row["router_switched"] is True for row in rows)
    legacy_errors = tuple(
        abs(a - p) / max(abs(a), abs(p), 1.0e-12)
        for a, p in zip(actual, predicted, strict=True)
    )
    result: dict[str, object] = {
        "row_count": len(rows),
        "family_count": len({str(row["family_id"]) for row in rows}),
        "family_ids": sorted({str(row["family_id"]) for row in rows}),
        "informative_row_count": len(informative),
        "mean_absolute_actual_delta": sum(abs(value) for value in actual)
        / len(actual),
        "legacy_maximum_relative_error": max(legacy_errors),
        "relative_l2_error": relative_l2,
        "quadratic_scaled_remainder_l2": error_l2 / (attenuation * attenuation),
        "quadratic_scaled_remainder_definition": (
            "l2(actual-predicted)/(1-alpha)^2"
        ),
        "cosine_similarity": cosine,
        "sign_agreement_fraction_on_informative_rows": (
            None
            if not sign_values
            else sum(value is True for value in sign_values) / len(sign_values)
        ),
        "all_informative_rows_mixed_tolerance_passed": bool(
            informative
            and all(row["mixed_tolerance_passed"] is True for row in informative)
        ),
        "all_informative_rows_sign_agree": bool(
            sign_values and all(value is True for value in sign_values)
        ),
        "uniform_all_rows_mixed_tolerance_passed": all(
            row["mixed_tolerance_passed"] is True for row in rows
        ),
        "router_switch_count": switch_count,
        "router_stability_fraction": 1.0 - switch_count / len(rows),
        "router_stability_informative": router_stability_informative,
        "subgroup_note": subgroup_note,
    }
    if include_family_breakdown:
        family_groups = {
            family_id: tuple(
                row for row in rows if str(row["family_id"]) == family_id
            )
            for family_id in sorted({str(row["family_id"]) for row in rows})
        }
        per_family = {
            family_id: _aggregate_rows(
                family_rows,
                numerical_epsilon=numerical_epsilon,
                router_stability_informative=router_stability_informative,
                subgroup_note=f"deterministic family block: {family_id}",
                attenuation_from_alpha_one=attenuation,
                include_family_breakdown=False,
            )
            for family_id, family_rows in family_groups.items()
        }
        cosine_values = tuple(
            float(item["cosine_similarity"])
            for item in per_family.values()
            if item["cosine_similarity"] is not None
        )
        sign_values = tuple(
            float(item["sign_agreement_fraction_on_informative_rows"])
            for item in per_family.values()
            if item["sign_agreement_fraction_on_informative_rows"] is not None
        )
        family_count = len(per_family)
        result["per_family"] = per_family
        result["family_macro_average"] = {
            "represented_family_count": family_count,
            "mean_relative_l2_error": sum(
                float(item["relative_l2_error"])
                for item in per_family.values()
            )
            / family_count,
            "mean_quadratic_scaled_remainder_l2": sum(
                float(item["quadratic_scaled_remainder_l2"])
                for item in per_family.values()
            )
            / family_count,
            "mean_absolute_actual_delta": sum(
                float(item["mean_absolute_actual_delta"])
                for item in per_family.values()
            )
            / family_count,
            "cosine_defined_family_count": len(cosine_values),
            "mean_cosine_similarity_over_defined_families": (
                None
                if not cosine_values
                else sum(cosine_values) / len(cosine_values)
            ),
            "sign_defined_family_count": len(sign_values),
            "mean_sign_agreement_over_defined_families": (
                None if not sign_values else sum(sign_values) / len(sign_values)
            ),
            "fraction_families_all_informative_rows_mixed_passed": sum(
                item["all_informative_rows_mixed_tolerance_passed"] is True
                for item in per_family.values()
            )
            / family_count,
            "fraction_families_all_informative_rows_sign_agree": sum(
                item["all_informative_rows_sign_agree"] is True
                for item in per_family.values()
            )
            / family_count,
        }
    return result


def _alpha_result(
    frozen: FrozenPublishedPilot,
    sweep: _SweepExecution,
    evidence: FiniteInterventionEvidence,
    finite_receipts: tuple[NativePartGateExecutionReceipt, ...],
) -> dict[str, object]:
    rows = _raw_finite_rows(
        frozen,
        evidence,
        numerical_epsilon=sweep.numerical_epsilon,
    )
    active = [row for row in rows if row["path_active"] is True]
    inactive = [row for row in rows if row["path_active"] is False]
    router_informative = frozen.chart_atlas.selected_chart_count > 1
    return {
        "alpha": evidence.gate_alpha,
        "attenuation_from_alpha_one": 1.0 - evidence.gate_alpha,
        "finite_intervention_evidence": evidence.to_dict(),
        "native_gate_receipts": [
            {
                "row_id": row.row_id,
                "condition": "finite",
                **_native_receipt_to_dict(receipt),
            }
            for row, receipt in zip(
                frozen.C_role.rows,
                finite_receipts,
                strict=True,
            )
        ],
        "rows": rows,
        "aggregates": {
            "chart_selected_all_12_primary": _aggregate_rows(
                rows,
                numerical_epsilon=sweep.numerical_epsilon,
                router_stability_informative=router_informative,
                attenuation_from_alpha_one=1.0 - evidence.gate_alpha,
                subgroup_note=(
                    "primary original chart-only gate mask; contains both "
                    "path-active and path-inactive rows"
                ),
            ),
            "path_active_3_exploratory": _aggregate_rows(
                active,
                numerical_epsilon=sweep.numerical_epsilon,
                router_stability_informative=router_informative,
                attenuation_from_alpha_one=1.0 - evidence.gate_alpha,
                subgroup_note=(
                    "exploratory reused-C subgroup; n=3 across two semantic "
                    "families; not fresh confirmation"
                ),
            ),
            "path_inactive_chart_only_9_exploratory": _aggregate_rows(
                inactive,
                numerical_epsilon=sweep.numerical_epsilon,
                router_stability_informative=router_informative,
                attenuation_from_alpha_one=1.0 - evidence.gate_alpha,
                subgroup_note=(
                    "exploratory reused-C complement; n=9 selected only by "
                    "the original chart mask"
                ),
            ),
        },
    }


def _stratum_local_linear_pass(
    alpha_results: Sequence[Mapping[str, object]],
    *,
    stratum: str,
    noop_control_passed: bool,
) -> bool:
    by_alpha = {float(item["alpha"]): item for item in alpha_results}
    required: list[Mapping[str, object]] = []
    for alpha in LOCAL_LINEAR_ALPHAS:
        result = by_alpha.get(alpha)
        if not isinstance(result, Mapping):
            return False
        aggregates = result.get("aggregates")
        if not isinstance(aggregates, Mapping):
            return False
        aggregate = aggregates.get(stratum)
        if not isinstance(aggregate, Mapping):
            return False
        required.append(aggregate)
    return bool(
        noop_control_passed
        and all(
            _finite(item["relative_l2_error"], label="relative L2")
            <= _LOCAL_LINEAR_RELATIVE_L2_LIMIT
            and item["informative_row_count"] > 0
            and item["all_informative_rows_mixed_tolerance_passed"] is True
            and item["all_informative_rows_sign_agree"] is True
            and (
                item["router_stability_informative"] is False
                or item["router_switch_count"] == 0
            )
            for item in required
        )
    )


def _largest_contiguous_attenuation(
    alpha_results: Sequence[Mapping[str, object]],
    *,
    stratum: str,
    noop_control_passed: bool,
) -> float:
    if not noop_control_passed:
        return 0.0
    result = 0.0
    for item in alpha_results:
        aggregates = item["aggregates"]
        assert isinstance(aggregates, Mapping)
        aggregate = aggregates[stratum]
        assert isinstance(aggregate, Mapping)
        passed = bool(
            _finite(aggregate["relative_l2_error"], label="relative L2")
            <= _LOCAL_LINEAR_RELATIVE_L2_LIMIT
            and aggregate["informative_row_count"] > 0
            and aggregate["all_informative_rows_mixed_tolerance_passed"] is True
            and aggregate["all_informative_rows_sign_agree"] is True
            and (
                aggregate["router_stability_informative"] is False
                or aggregate["router_switch_count"] == 0
            )
        )
        if not passed:
            break
        result = _finite(
            item["attenuation_from_alpha_one"],
            label="attenuation",
        )
    return result


def _build_alpha_one_control(
    frozen: FrozenPublishedPilot,
    sweep: _SweepExecution,
) -> dict[str, object]:
    activity = _path_activity_by_row(frozen)
    rows = []
    receipt_set = []
    for index, (row, receipt) in enumerate(
        zip(frozen.C_role.rows, sweep.alpha_one_receipts, strict=True)
    ):
        receipt_set.append((row.row_id, "alpha_one", receipt))
        rows.append(
            {
                "row_id": row.row_id,
                "family_id": row.family_id,
                "chart_selected": True,
                "path_active": activity[row.row_id],
                "captured_baseline_score": row.target_score,
                "fresh_baseline_score": sweep.fresh_baseline_scores[index],
                "alpha_one_score": sweep.alpha_one_scores[index],
                "baseline_repeat_delta": (
                    sweep.fresh_baseline_scores[index] - row.target_score
                ),
                "alpha_one_noop_delta": (
                    sweep.alpha_one_scores[index]
                    - sweep.fresh_baseline_scores[index]
                ),
                "fresh_baseline_chart_id": sweep.fresh_baseline_chart_ids[index],
                "alpha_one_chart_id": sweep.alpha_one_chart_ids[index],
                "router_switched": (
                    sweep.fresh_baseline_chart_ids[index]
                    != sweep.alpha_one_chart_ids[index]
                ),
                "native_gate_receipt": _native_receipt_to_dict(receipt),
            }
        )
    maximum_error = max(
        sweep.baseline_repeat_max_abs_error,
        sweep.alpha_one_max_abs_error,
    )
    return {
        "alpha": 1.0,
        "rows": rows,
        "baseline_repeat_max_abs_error": sweep.baseline_repeat_max_abs_error,
        "alpha_one_max_abs_error": sweep.alpha_one_max_abs_error,
        "numerical_epsilon": sweep.numerical_epsilon,
        "epsilon_definition": (
            "max(5e-5,10*max(baseline_repeat_max_abs_error,"
            "alpha_one_max_abs_error))"
        ),
        "native_gate_receipt_set_sha256": _receipt_set_sha256(receipt_set),
        "router_switch_count": sum(
            row["router_switched"] is True for row in rows
        ),
        "passed": bool(
            maximum_error <= _NUMERICAL_EPSILON_FLOOR
            and all(row["router_switched"] is False for row in rows)
        ),
    }


def _require_exact_half_replay(
    frozen: FrozenPublishedPilot,
    alpha_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    half = next(
        (item for item in alpha_results if float(item["alpha"]) == 0.5),
        None,
    )
    source_C = frozen.source_report["C_intervention"]
    if not isinstance(half, Mapping) or not isinstance(source_C, Mapping):
        raise RuntimeError("alpha=.50 replay or original C receipt is unavailable")
    evidence_value = half["finite_intervention_evidence"]
    aggregates = half["aggregates"]
    if not isinstance(evidence_value, Mapping) or not isinstance(aggregates, Mapping):
        raise RuntimeError("alpha=.50 replay structure is invalid")
    evidence = FiniteInterventionEvidence.from_dict(evidence_value)
    primary = aggregates["chart_selected_all_12_primary"]
    if not isinstance(primary, Mapping):
        raise RuntimeError("alpha=.50 primary aggregate is invalid")
    comparisons = {
        "evidence_sha256": (
            evidence.evidence_sha256,
            source_C.get("evidence_sha256"),
        ),
        "native_gate_execution_receipt_sha256": (
            evidence.native_gate_execution_receipt_sha256,
            source_C.get("native_gate_execution_receipt_sha256"),
        ),
        "legacy_maximum_relative_error": (
            primary.get("legacy_maximum_relative_error"),
            source_C.get("maximum_first_order_relative_error"),
        ),
        "mean_absolute_actual_delta": (
            primary.get("mean_absolute_actual_delta"),
            source_C.get("mean_absolute_score_delta_on_selected_chart"),
        ),
        "router_stability_fraction": (
            primary.get("router_stability_fraction"),
            source_C.get("router_stability_fraction"),
        ),
        "native_noop_max_abs_error": (
            evidence.native_noop_max_abs_error,
            source_C.get("native_noop_max_abs_error"),
        ),
    }
    mismatches = [
        name for name, (observed, expected) in comparisons.items()
        if observed != expected
    ]
    if mismatches:
        raise RuntimeError(
            "alpha=.50 did not exactly replay original C: " + ", ".join(mismatches)
        )
    return {
        "exact_replay_passed": True,
        "original_C_evidence_sha256": evidence.evidence_sha256,
        "original_C_native_gate_receipt_sha256": (
            evidence.native_gate_execution_receipt_sha256
        ),
        "original_C_failure_reasons": list(source_C.get("failure_reasons", [])),
        "original_C_passed": source_C.get("passed"),
    }


def _source_bindings(frozen: FrozenPublishedPilot) -> dict[str, object]:
    return {
        "capture_sha256": frozen.capture.capture_sha256,
        "source_report_sha256": frozen.source_report["report_sha256"],
        "selected_candidate_sha256": frozen.artifact.candidate_sha256,
        "chart_atlas_sha256": frozen.chart_atlas.atlas_sha256,
        "pilot_configuration_sha256": _configuration_sha256(
            frozen.pilot_config
        ),
        "candidate_catalog_sha256": PUBLISHED_CANDIDATE_CATALOG_SHA256,
        "C_role_sha256": frozen.C_role.role_sha256,
        "C_row_ids_sha256": frozen.C_role.row_ids_sha256,
        "exact_control_evidence_sha256": (
            frozen.capture.exact_lift_receipt.exact_control_evidence_sha256
        ),
        "source_model_fingerprint_sha256": (
            frozen.capture.source_model_fingerprint_sha256
        ),
        "source_execution_fingerprint_sha256": (
            frozen.capture.source_execution_fingerprint_sha256
        ),
    }


def _build_report_payload(
    frozen: FrozenPublishedPilot,
    sweep: _SweepExecution,
) -> dict[str, object]:
    if len(sweep.finite_evidences) != len(FROZEN_FINITE_ALPHAS):
        raise RuntimeError("finite sweep did not produce the frozen alpha schedule")
    alpha_one = _build_alpha_one_control(frozen, sweep)
    alpha_results = [
        _alpha_result(frozen, sweep, evidence, receipts)
        for evidence, receipts in zip(
            sweep.finite_evidences,
            sweep.finite_receipts,
            strict=True,
        )
    ]
    half_replay = _require_exact_half_replay(frozen, alpha_results)
    noop_passed = alpha_one["passed"] is True
    primary_key = "chart_selected_all_12_primary"
    active_key = "path_active_3_exploratory"
    primary_local = _stratum_local_linear_pass(
        alpha_results,
        stratum=primary_key,
        noop_control_passed=noop_passed,
    )
    active_local = _stratum_local_linear_pass(
        alpha_results,
        stratum=active_key,
        noop_control_passed=noop_passed,
    )
    active_attenuation = _largest_contiguous_attenuation(
        alpha_results,
        stratum=active_key,
        noop_control_passed=noop_passed,
    )
    maximum_scheduled_attenuation = 1.0 - FROZEN_FINITE_ALPHAS[-1]
    active_wide_range = bool(
        active_local and active_attenuation == maximum_scheduled_attenuation
    )
    finite_nonlinearity = bool(active_local and not active_wide_range)
    stage_inventory = _stage_directional_l2_inventory(frozen)
    dominant_stage = next(item for item in stage_inventory if item["dominant"])
    outcome = (
        "local_only_with_finite_nonlinearity_on_path_active_reused_C"
        if finite_nonlinearity and not primary_local
        else (
            "wide_range_first_order_agreement_on_path_active_reused_C_only"
            if active_wide_range and not primary_local
            else (
                "local_first_order_binding_observed_chart_wide_reused_C_only"
                if primary_local
                else "no_local_first_order_binding_observed_on_reused_C"
            )
        )
    )
    return {
        "schema": DOSE_CURVE_SCHEMA,
        "format_version": DOSE_CURVE_FORMAT_VERSION,
        "diagnostic_outcome": outcome,
        "source_bindings": _source_bindings(frozen),
        "frozen_protocol": {
            "schedule": list(FROZEN_SCHEDULE),
            "finite_alphas": list(FROZEN_FINITE_ALPHAS),
            "local_linear_alphas": list(LOCAL_LINEAR_ALPHAS),
            "C_rows": [
                {
                    "row_id": row_id,
                    "family_id": family_id,
                    "path_active": row_id in PUBLISHED_PATH_ACTIVE_ROW_IDS,
                }
                for row_id, family_id in PUBLISHED_C_ROWS
            ],
            "selected_chart_id": PUBLISHED_CHART_ID,
            "selected_chart_count": frozen.chart_atlas.selected_chart_count,
            "intervention_mask": "frozen_baseline_chart_only",
            "chart_only_mask_gates_all_12": True,
            "path_active_row_count": 3,
            "path_inactive_chart_only_row_count": 9,
            "fit_refit_or_reselection_count": 0,
            "same_C_rows_reused": True,
        },
        "alpha_one_control": alpha_one,
        "finite_alpha_results": alpha_results,
        "alpha_half_original_C_replay": half_replay,
        "local_linear_diagnostic": {
            "relative_l2_limit": _LOCAL_LINEAR_RELATIVE_L2_LIMIT,
            "mixed_row_tolerance": "epsilon+0.05*abs(predicted)",
            "uninformative_rule": "max(abs(actual),abs(predicted))<10*epsilon",
            "chart_wide_local_linear_pass": primary_local,
            "path_active_local_linear_observed": active_local,
            "primary_interpretive_stratum": "path_active_3_exploratory",
            "all_12_role": "exact_original_chart_only_intervention_replay",
            "path_active_largest_contiguous_attenuation_from_alpha_one": (
                active_attenuation
            ),
            "path_active_smallest_contiguous_alpha": 1.0 - active_attenuation,
            "path_active_wide_range_first_order_agreement": active_wide_range,
            "finite_nonlinearity_observed_beyond_local_radius": (
                finite_nonlinearity
            ),
            "router_stability_informative": (
                frozen.chart_atlas.selected_chart_count > 1
            ),
            "router_note": (
                "atlas K=1, so router stability is a replay value only and is "
                "excluded as affirmative local-linear evidence"
            ),
        },
        "stage_directional_l2_inventory": {
            "scope": "all_12_reused_C_rows",
            "share_denominator": "l2_norm_of_summed_four_stage_direction",
            "summed_four_stage_directional_l2": PUBLISHED_TOTAL_DIRECTIONAL_L2,
            "stages": stage_inventory,
            "dominant_stage_boundary_id": dominant_stage["boundary_id"],
            "dominant_stage_directional_l2_share": dominant_stage[
                "directional_l2_share"
            ],
            "four_stage_coherence_established": False,
            "caveat": (
                "one stage dominates the first-order direction, so local "
                "agreement cannot by itself establish four-stage coherence"
            ),
            "inventory_sha256": _sha256(stage_inventory),
        },
        "execution_firewall": {
            "model_load_count": 1,
            "A_B_selection_replay_count": 1,
            "fit_refit_or_reselection_count": 0,
            "finite_candidate_count": 1,
            "finite_alpha_count": 4,
            "C_row_count_per_alpha": 12,
            "chart_only_mask_preserved": True,
            "source_model_unchanged": (
                sweep.source_model_before == sweep.source_model_after
            ),
            "source_execution_unchanged": (
                sweep.source_execution_before == sweep.source_execution_after
            ),
            "local_files_only": True,
        },
        "validation_boundary": {
            "structural_consistency_validated": True,
            "published_alpha_half_replay_authenticated": True,
            "novel_intervened_scores_externally_attested": False,
            "novel_scores_cryptographically_bound_to_native_receipts": False,
            "independent_novel_alpha_rerun_completed": False,
            "trust_requirement": (
                "the final report hash is a publication commitment after this "
                "execution; novel observations require an independent rerun "
                "for external confirmation"
            ),
        },
        "scientific_status": {
            "exploratory_post_C": True,
            "same_C_rows_reused": True,
            "fresh_confirmation": False,
            "supermode_confirmed": False,
            "chart_wide_uniform_pass": primary_local,
            "finite_nonlinearity_observed": finite_nonlinearity,
            "wide_range_first_order_agreement": active_wide_range,
            "path_active_subgroup_is_n3_two_families": True,
            "four_stage_coherence_established": False,
            "novel_alpha_observations_externally_confirmed": False,
            "compression_attempted": False,
        },
        "claims": {
            "fresh_confirmation": False,
            "supermode_confirmed": False,
            "four_stage_coherence": False,
            "compression_success": False,
            "parameter_savings": False,
            "MAC_savings": False,
            "latency_improvement": False,
        },
    }


@dataclass(frozen=True, slots=True)
class PartsFirstPostCDoseCurveReport:
    payload: Mapping[str, object]
    report_sha256: str = ""

    def __post_init__(self) -> None:
        materialized = json.loads(json.dumps(self.payload, allow_nan=False))
        if not isinstance(materialized, dict):
            raise TypeError("dose-curve report payload must be a mapping")
        object.__setattr__(self, "payload", materialized)
        computed = _sha256(materialized, domain=_REPORT_DOMAIN)
        if self.report_sha256 == "":
            object.__setattr__(self, "report_sha256", computed)
        elif self.report_sha256 != computed:
            raise ValueError("dose-curve report hash mismatch")
        validate_post_c_dose_curve_report(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {**dict(self.payload), "report_sha256": self.report_sha256}


def _finite_receipt_record(
    value: Mapping[str, object],
) -> tuple[str, NativePartGateExecutionReceipt]:
    _exact_keys(
        value,
        {
            "row_id",
            "condition",
            "gate_states",
            "hook_call_counts",
            "hook_execution_order",
            "mask_true_counts",
            "alpha_one_control",
            "hooks_removed",
            "weights_mutated_by_hooks",
            "later_native_computation_reran",
            "receipt_sha256",
        },
        label="finite native receipt record",
    )
    if value["condition"] != "finite":
        raise ValueError("finite native receipt condition drifted")
    receipt_value = {
        key: item
        for key, item in value.items()
        if key not in {"row_id", "condition"}
    }
    return str(value["row_id"]), _native_receipt_from_dict(receipt_value)


def _expected_raw_row(
    *,
    row_id: str,
    family_id: str,
    path_active: bool,
    evidence: FiniteInterventionEvidence,
    index: int,
    epsilon: float,
) -> dict[str, object]:
    baseline = evidence.baseline_scores[index]
    intervened = evidence.intervened_scores[index]
    actual = intervened - baseline
    predicted = evidence.first_order_predicted_score_deltas[index]
    signal = max(abs(actual), abs(predicted))
    informative = signal >= 10.0 * epsilon
    tolerance = epsilon + _MIXED_RELATIVE_TOLERANCE * abs(predicted)
    return {
        "row_id": row_id,
        "family_id": family_id,
        "chart_selected": (
            evidence.baseline_chart_ids[index] == PUBLISHED_CHART_ID
        ),
        "path_active": path_active,
        "baseline_score": baseline,
        "intervened_score": intervened,
        "actual_delta": actual,
        "first_order_predicted_delta": predicted,
        "actual_slope_from_alpha_one": actual / (evidence.gate_alpha - 1.0),
        "first_order_predicted_slope": predicted / (evidence.gate_alpha - 1.0),
        "baseline_chart_id": evidence.baseline_chart_ids[index],
        "online_chart_id": evidence.online_chart_ids_after[index],
        "router_switched": (
            evidence.baseline_chart_ids[index]
            != evidence.online_chart_ids_after[index]
        ),
        "signal_scale": signal,
        "informative": informative,
        "mixed_tolerance": tolerance,
        "mixed_tolerance_passed": abs(actual - predicted) <= tolerance,
        "sign_agreement": (
            None
            if not informative
            else bool(
                (actual > 0.0 and predicted > 0.0)
                or (actual < 0.0 and predicted < 0.0)
            )
        ),
    }


def _validate_alpha_one_control(
    value: object,
) -> tuple[
    dict[str, object],
    tuple[NativePartGateExecutionReceipt, ...],
]:
    if not isinstance(value, Mapping):
        raise ValueError("alpha-one control must be an object")
    _exact_keys(
        value,
        {
            "alpha",
            "rows",
            "baseline_repeat_max_abs_error",
            "alpha_one_max_abs_error",
            "numerical_epsilon",
            "epsilon_definition",
            "native_gate_receipt_set_sha256",
            "router_switch_count",
            "passed",
        },
        label="alpha-one control",
    )
    rows = value["rows"]
    if (
        type(value["alpha"]) is not float
        or value["alpha"] != 1.0
        or not isinstance(rows, list)
        or len(rows) != 12
    ):
        raise ValueError("alpha-one control inventory drifted")
    receipts: list[NativePartGateExecutionReceipt] = []
    repeat_errors: list[float] = []
    noop_errors: list[float] = []
    switches = 0
    for index, ((row_id, family_id), row_value) in enumerate(
        zip(PUBLISHED_C_ROWS, rows, strict=True)
    ):
        if not isinstance(row_value, Mapping):
            raise ValueError("alpha-one row must be an object")
        _exact_keys(
            row_value,
            {
                "row_id",
                "family_id",
                "chart_selected",
                "path_active",
                "captured_baseline_score",
                "fresh_baseline_score",
                "alpha_one_score",
                "baseline_repeat_delta",
                "alpha_one_noop_delta",
                "fresh_baseline_chart_id",
                "alpha_one_chart_id",
                "router_switched",
                "native_gate_receipt",
            },
            label="alpha-one row",
        )
        expected_active = row_id in PUBLISHED_PATH_ACTIVE_ROW_IDS
        if (
            row_value["row_id"] != row_id
            or row_value["family_id"] != family_id
            or row_value["chart_selected"] is not True
            or row_value["path_active"] is not expected_active
        ):
            raise ValueError("alpha-one row identity or stratum drifted")
        captured = _finite(
            row_value["captured_baseline_score"],
            label="captured baseline",
        )
        fresh = _finite(
            row_value["fresh_baseline_score"],
            label="fresh baseline",
        )
        noop = _finite(row_value["alpha_one_score"], label="alpha-one score")
        repeat_delta = fresh - captured
        noop_delta = noop - fresh
        if (
            row_value["baseline_repeat_delta"] != repeat_delta
            or row_value["alpha_one_noop_delta"] != noop_delta
            or row_value["router_switched"]
            is not (
                row_value["fresh_baseline_chart_id"]
                != row_value["alpha_one_chart_id"]
            )
        ):
            raise ValueError("alpha-one row arithmetic does not close")
        receipt_value = row_value["native_gate_receipt"]
        if not isinstance(receipt_value, Mapping):
            raise ValueError("alpha-one native receipt is invalid")
        receipt = _native_receipt_from_dict(receipt_value)
        if not receipt.alpha_one_control:
            raise ValueError("alpha-one row contains a finite receipt")
        if (
            receipt.gate_states != _expected_published_gate_states(1.0)
            or receipt.mask_true_counts != (1, 1, 1, 1)
        ):
            raise ValueError("alpha-one native gate state drifted")
        receipts.append(receipt)
        repeat_errors.append(abs(repeat_delta))
        noop_errors.append(abs(noop_delta))
        switches += row_value["router_switched"] is True
    repeat_max = max(repeat_errors)
    noop_max = max(noop_errors)
    epsilon = max(_NUMERICAL_EPSILON_FLOOR, 10.0 * max(repeat_max, noop_max))
    receipt_sha = _receipt_set_sha256(
        tuple(
            (row_id, "alpha_one", receipt)
            for (row_id, _family), receipt in zip(
                PUBLISHED_C_ROWS,
                receipts,
                strict=True,
            )
        )
    )
    passed = bool(
        max(repeat_max, noop_max) <= _NUMERICAL_EPSILON_FLOOR
        and switches == 0
    )
    if (
        value["baseline_repeat_max_abs_error"] != repeat_max
        or value["alpha_one_max_abs_error"] != noop_max
        or value["numerical_epsilon"] != epsilon
        or value["epsilon_definition"]
        != (
            "max(5e-5,10*max(baseline_repeat_max_abs_error,"
            "alpha_one_max_abs_error))"
        )
        or value["native_gate_receipt_set_sha256"] != receipt_sha
        or value["router_switch_count"] != switches
        or value["passed"] is not passed
    ):
        raise ValueError("alpha-one aggregate does not close")
    return dict(value), tuple(receipts)


def _validate_finite_alpha_result(
    value: object,
    *,
    expected_alpha: float,
    epsilon: float,
    expected_noop_error: float,
    alpha_one_receipts: tuple[NativePartGateExecutionReceipt, ...],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("finite alpha result must be an object")
    _exact_keys(
        value,
        {
            "alpha",
            "attenuation_from_alpha_one",
            "finite_intervention_evidence",
            "native_gate_receipts",
            "rows",
            "aggregates",
        },
        label="finite alpha result",
    )
    if (
        type(value["alpha"]) is not float
        or type(value["attenuation_from_alpha_one"]) is not float
        or value["alpha"] != expected_alpha
        or value["attenuation_from_alpha_one"] != 1.0 - expected_alpha
    ):
        raise ValueError("finite alpha schedule drifted")
    evidence_value = value["finite_intervention_evidence"]
    if not isinstance(evidence_value, Mapping):
        raise ValueError("finite evidence must be an object")
    evidence = FiniteInterventionEvidence.from_dict(evidence_value)
    if (
        evidence.gate_alpha != expected_alpha
        or evidence.candidate_sha256 != PUBLISHED_CANDIDATE_SHA256
        or evidence.c_row_ids_sha256 != PUBLISHED_C_ROW_IDS_SHA256
        or evidence.source_model_fingerprint_before
        != PUBLISHED_SOURCE_MODEL_SHA256
        or evidence.source_model_fingerprint_after
        != PUBLISHED_SOURCE_MODEL_SHA256
        or evidence.source_execution_fingerprint_before
        != PUBLISHED_SOURCE_EXECUTION_SHA256
        or evidence.source_execution_fingerprint_after
        != PUBLISHED_SOURCE_EXECUTION_SHA256
        or evidence.native_noop_max_abs_error != expected_noop_error
        or len(evidence.baseline_scores) != 12
        or evidence.baseline_chart_ids != (PUBLISHED_CHART_ID,) * 12
    ):
        raise ValueError("finite evidence source binding drifted")

    receipts_value = value["native_gate_receipts"]
    if not isinstance(receipts_value, list) or len(receipts_value) != 12:
        raise ValueError("finite receipt inventory drifted")
    finite_receipts: list[NativePartGateExecutionReceipt] = []
    receipt_set: list[tuple[str, str, NativePartGateExecutionReceipt]] = []
    for (expected_row_id, _family), raw, noop_receipt in zip(
        PUBLISHED_C_ROWS,
        receipts_value,
        alpha_one_receipts,
        strict=True,
    ):
        if not isinstance(raw, Mapping):
            raise ValueError("finite receipt record must be an object")
        row_id, receipt = _finite_receipt_record(raw)
        if row_id != expected_row_id or receipt.alpha_one_control:
            raise ValueError("finite receipt row or condition drifted")
        if (
            receipt.gate_states != _expected_published_gate_states(expected_alpha)
            or receipt.mask_true_counts != (1, 1, 1, 1)
        ):
            raise ValueError("finite native gate alpha drifted")
        finite_receipts.append(receipt)
        receipt_set.extend(
            (
                (row_id, "alpha_one", noop_receipt),
                (row_id, "finite", receipt),
            )
        )
    if (
        _receipt_set_sha256(receipt_set)
        != evidence.native_gate_execution_receipt_sha256
    ):
        raise ValueError("finite native receipt set does not close")

    rows_value = value["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != 12:
        raise ValueError("finite raw row inventory drifted")
    expected_rows = [
        _expected_raw_row(
            row_id=row_id,
            family_id=family_id,
            path_active=row_id in PUBLISHED_PATH_ACTIVE_ROW_IDS,
            evidence=evidence,
            index=index,
            epsilon=epsilon,
        )
        for index, (row_id, family_id) in enumerate(PUBLISHED_C_ROWS)
    ]
    if rows_value != expected_rows:
        raise ValueError("finite per-row arithmetic or lineage does not close")

    aggregates = value["aggregates"]
    if not isinstance(aggregates, Mapping):
        raise ValueError("finite aggregates must be an object")
    expected_aggregates = {
        "chart_selected_all_12_primary": _aggregate_rows(
            expected_rows,
            numerical_epsilon=epsilon,
            router_stability_informative=False,
            attenuation_from_alpha_one=1.0 - expected_alpha,
            subgroup_note=(
                "primary original chart-only gate mask; contains both "
                "path-active and path-inactive rows"
            ),
        ),
        "path_active_3_exploratory": _aggregate_rows(
            [row for row in expected_rows if row["path_active"] is True],
            numerical_epsilon=epsilon,
            router_stability_informative=False,
            attenuation_from_alpha_one=1.0 - expected_alpha,
            subgroup_note=(
                "exploratory reused-C subgroup; n=3 across two semantic "
                "families; not fresh confirmation"
            ),
        ),
        "path_inactive_chart_only_9_exploratory": _aggregate_rows(
            [row for row in expected_rows if row["path_active"] is False],
            numerical_epsilon=epsilon,
            router_stability_informative=False,
            attenuation_from_alpha_one=1.0 - expected_alpha,
            subgroup_note=(
                "exploratory reused-C complement; n=9 selected only by "
                "the original chart mask"
            ),
        ),
    }
    if aggregates != expected_aggregates:
        raise ValueError("finite alpha aggregates do not close")
    return dict(value)


def _require_half_anchored_evidence_lineage(
    finite_results: Sequence[Mapping[str, object]],
) -> FiniteInterventionEvidence:
    if len(finite_results) != len(FROZEN_FINITE_ALPHAS):
        raise ValueError("finite evidence lineage is incomplete")
    half_value = finite_results[-1]["finite_intervention_evidence"]
    if not isinstance(half_value, Mapping):
        raise ValueError("alpha=.50 evidence must be an object")
    half = FiniteInterventionEvidence.from_dict(half_value)
    half_directional = tuple(
        -2.0 * value for value in half.first_order_predicted_score_deltas
    )
    for alpha, result in zip(
        FROZEN_FINITE_ALPHAS,
        finite_results,
        strict=True,
    ):
        evidence_value = result["finite_intervention_evidence"]
        if not isinstance(evidence_value, Mapping):
            raise ValueError("finite evidence must be an object")
        evidence = FiniteInterventionEvidence.from_dict(evidence_value)
        expected_prediction = tuple(
            (alpha - 1.0) * directional for directional in half_directional
        )
        if (
            evidence.executor_kind != "live_model_bridge"
            or evidence.baseline_scores != half.baseline_scores
            or evidence.baseline_chart_ids != half.baseline_chart_ids
            or evidence.online_chart_ids_after != half.online_chart_ids_after
            or evidence.native_noop_max_abs_error
            != half.native_noop_max_abs_error
            or evidence.first_order_predicted_score_deltas
            != expected_prediction
        ):
            raise ValueError(
                "novel-alpha evidence is not anchored to published alpha=.50"
            )
    return half


def _require_alpha_one_half_anchor(
    alpha_one: Mapping[str, object],
    half: FiniteInterventionEvidence,
) -> None:
    rows = alpha_one.get("rows")
    if not isinstance(rows, list) or len(rows) != len(half.baseline_scores):
        raise ValueError("alpha-one rows do not align with alpha=.50")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("alpha-one row must be an object")
        baseline = half.baseline_scores[index]
        chart = half.baseline_chart_ids[index]
        if (
            row.get("captured_baseline_score") != baseline
            or row.get("fresh_baseline_score") != baseline
            or row.get("alpha_one_score") != baseline
            or row.get("fresh_baseline_chart_id") != chart
            or row.get("alpha_one_chart_id") != chart
            or row.get("chart_selected")
            is not (chart == PUBLISHED_CHART_ID)
        ):
            raise ValueError("alpha-one values are not anchored to alpha=.50")


def validate_post_c_dose_curve_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("dose-curve report must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(materialized, dict):
        raise TypeError("dose-curve report must be an object")
    _exact_keys(
        materialized,
        {
            "schema",
            "format_version",
            "diagnostic_outcome",
            "source_bindings",
            "frozen_protocol",
            "alpha_one_control",
            "finite_alpha_results",
            "alpha_half_original_C_replay",
            "local_linear_diagnostic",
            "stage_directional_l2_inventory",
            "execution_firewall",
            "validation_boundary",
            "scientific_status",
            "claims",
            "report_sha256",
        },
        label="dose-curve report",
    )
    report_sha = materialized.pop("report_sha256")
    if (
        materialized["schema"] != DOSE_CURVE_SCHEMA
        or type(materialized["format_version"]) is not int
        or materialized["format_version"] != DOSE_CURVE_FORMAT_VERSION
        or report_sha != _sha256(materialized, domain=_REPORT_DOMAIN)
    ):
        raise ValueError("dose-curve report header or hash is invalid")

    source = materialized["source_bindings"]
    if not isinstance(source, Mapping):
        raise ValueError("dose-curve source bindings must be an object")
    _exact_keys(
        source,
        {
            "capture_sha256",
            "source_report_sha256",
            "selected_candidate_sha256",
            "chart_atlas_sha256",
            "pilot_configuration_sha256",
            "candidate_catalog_sha256",
            "C_role_sha256",
            "C_row_ids_sha256",
            "exact_control_evidence_sha256",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
        },
        label="dose-curve source bindings",
    )
    expected_source = {
        "capture_sha256": PUBLISHED_CAPTURE_SHA256,
        "source_report_sha256": PUBLISHED_REPORT_SHA256,
        "selected_candidate_sha256": PUBLISHED_CANDIDATE_SHA256,
        "chart_atlas_sha256": PUBLISHED_CHART_ATLAS_SHA256,
        "pilot_configuration_sha256": PUBLISHED_CONFIG_SHA256,
        "candidate_catalog_sha256": PUBLISHED_CANDIDATE_CATALOG_SHA256,
        "C_role_sha256": PUBLISHED_C_ROLE_SHA256,
        "C_row_ids_sha256": PUBLISHED_C_ROW_IDS_SHA256,
        "exact_control_evidence_sha256": (
            PUBLISHED_EXACT_CONTROL_EVIDENCE_SHA256
        ),
        "source_model_fingerprint_sha256": PUBLISHED_SOURCE_MODEL_SHA256,
        "source_execution_fingerprint_sha256": (
            PUBLISHED_SOURCE_EXECUTION_SHA256
        ),
    }
    if source != expected_source:
        raise ValueError("dose-curve report is not bound to the published pilot")

    protocol = materialized["frozen_protocol"]
    if not isinstance(protocol, Mapping):
        raise ValueError("dose-curve frozen protocol must be an object")
    _exact_keys(
        protocol,
        {
            "schedule",
            "finite_alphas",
            "local_linear_alphas",
            "C_rows",
            "selected_chart_id",
            "selected_chart_count",
            "intervention_mask",
            "chart_only_mask_gates_all_12",
            "path_active_row_count",
            "path_inactive_chart_only_row_count",
            "fit_refit_or_reselection_count",
            "same_C_rows_reused",
        },
        label="dose-curve frozen protocol",
    )
    rows_protocol = protocol.get("C_rows")
    expected_rows_protocol = [
        {
            "row_id": row_id,
            "family_id": family_id,
            "path_active": row_id in PUBLISHED_PATH_ACTIVE_ROW_IDS,
        }
        for row_id, family_id in PUBLISHED_C_ROWS
    ]
    if (
        protocol.get("schedule") != list(FROZEN_SCHEDULE)
        or protocol.get("finite_alphas") != list(FROZEN_FINITE_ALPHAS)
        or protocol.get("local_linear_alphas") != list(LOCAL_LINEAR_ALPHAS)
        or rows_protocol != expected_rows_protocol
        or protocol.get("selected_chart_id") != PUBLISHED_CHART_ID
        or any(
            type(protocol.get(name)) is not int
            for name in (
                "selected_chart_count",
                "path_active_row_count",
                "path_inactive_chart_only_row_count",
                "fit_refit_or_reselection_count",
            )
        )
        or protocol.get("selected_chart_count") != 1
        or protocol.get("intervention_mask") != "frozen_baseline_chart_only"
        or protocol.get("chart_only_mask_gates_all_12") is not True
        or protocol.get("path_active_row_count") != 3
        or protocol.get("path_inactive_chart_only_row_count") != 9
        or protocol.get("fit_refit_or_reselection_count") != 0
        or protocol.get("same_C_rows_reused") is not True
    ):
        raise ValueError("dose-curve frozen protocol drifted")

    alpha_one, alpha_one_receipts = _validate_alpha_one_control(
        materialized["alpha_one_control"]
    )
    finite_values = materialized["finite_alpha_results"]
    if not isinstance(finite_values, list) or len(finite_values) != 4:
        raise ValueError("dose-curve finite schedule is incomplete")
    finite_results = [
        _validate_finite_alpha_result(
            item,
            expected_alpha=alpha,
            epsilon=float(alpha_one["numerical_epsilon"]),
            expected_noop_error=max(
                float(alpha_one["baseline_repeat_max_abs_error"]),
                float(alpha_one["alpha_one_max_abs_error"]),
            ),
            alpha_one_receipts=alpha_one_receipts,
        )
        for alpha, item in zip(FROZEN_FINITE_ALPHAS, finite_values, strict=True)
    ]

    half = finite_results[-1]
    half_evidence = _require_half_anchored_evidence_lineage(finite_results)
    _require_alpha_one_half_anchor(alpha_one, half_evidence)
    half_primary = half["aggregates"]
    assert isinstance(half_primary, Mapping)
    half_primary = half_primary["chart_selected_all_12_primary"]
    assert isinstance(half_primary, Mapping)
    replay = materialized["alpha_half_original_C_replay"]
    expected_replay = {
        "exact_replay_passed": True,
        "original_C_evidence_sha256": PUBLISHED_C_EVIDENCE_SHA256,
        "original_C_native_gate_receipt_sha256": (
            PUBLISHED_C_GATE_RECEIPT_SHA256
        ),
        "original_C_failure_reasons": [
            "first_order_finite_agreement_failed"
        ],
        "original_C_passed": False,
    }
    if (
        replay != expected_replay
        or half_evidence.evidence_sha256 != PUBLISHED_C_EVIDENCE_SHA256
        or half_evidence.native_gate_execution_receipt_sha256
        != PUBLISHED_C_GATE_RECEIPT_SHA256
        or half_primary["legacy_maximum_relative_error"]
        != 1.836985357687236
        or half_primary["mean_absolute_actual_delta"]
        != 1.3384903222322464
    ):
        raise ValueError("alpha=.50 original C replay does not close")

    noop_passed = alpha_one["passed"] is True
    primary_local = _stratum_local_linear_pass(
        finite_results,
        stratum="chart_selected_all_12_primary",
        noop_control_passed=noop_passed,
    )
    active_local = _stratum_local_linear_pass(
        finite_results,
        stratum="path_active_3_exploratory",
        noop_control_passed=noop_passed,
    )
    active_attenuation = _largest_contiguous_attenuation(
        finite_results,
        stratum="path_active_3_exploratory",
        noop_control_passed=noop_passed,
    )
    maximum_scheduled_attenuation = 1.0 - FROZEN_FINITE_ALPHAS[-1]
    active_wide_range = bool(
        active_local and active_attenuation == maximum_scheduled_attenuation
    )
    finite_nonlinearity = bool(active_local and not active_wide_range)
    diagnostic = materialized["local_linear_diagnostic"]
    if not isinstance(diagnostic, Mapping):
        raise ValueError("dose-curve local-linear diagnostic must be an object")
    _exact_keys(
        diagnostic,
        {
            "relative_l2_limit",
            "mixed_row_tolerance",
            "uninformative_rule",
            "chart_wide_local_linear_pass",
            "path_active_local_linear_observed",
            "primary_interpretive_stratum",
            "all_12_role",
            "path_active_largest_contiguous_attenuation_from_alpha_one",
            "path_active_smallest_contiguous_alpha",
            "path_active_wide_range_first_order_agreement",
            "finite_nonlinearity_observed_beyond_local_radius",
            "router_stability_informative",
            "router_note",
        },
        label="dose-curve local-linear diagnostic",
    )
    if (
        diagnostic.get("relative_l2_limit")
        != _LOCAL_LINEAR_RELATIVE_L2_LIMIT
        or diagnostic.get("mixed_row_tolerance")
        != "epsilon+0.05*abs(predicted)"
        or diagnostic.get("uninformative_rule")
        != "max(abs(actual),abs(predicted))<10*epsilon"
        or diagnostic.get("chart_wide_local_linear_pass") is not primary_local
        or diagnostic.get("path_active_local_linear_observed") is not active_local
        or diagnostic.get(
            "path_active_largest_contiguous_attenuation_from_alpha_one"
        )
        != active_attenuation
        or diagnostic.get("path_active_smallest_contiguous_alpha")
        != 1.0 - active_attenuation
        or diagnostic.get("path_active_wide_range_first_order_agreement")
        is not active_wide_range
        or diagnostic.get("finite_nonlinearity_observed_beyond_local_radius")
        is not finite_nonlinearity
        or diagnostic.get("router_stability_informative") is not False
        or diagnostic.get("primary_interpretive_stratum")
        != "path_active_3_exploratory"
        or diagnostic.get("all_12_role")
        != "exact_original_chart_only_intervention_replay"
        or diagnostic.get("router_note")
        != (
            "atlas K=1, so router stability is a replay value only and is "
            "excluded as affirmative local-linear evidence"
        )
    ):
        raise ValueError("dose-curve local-linear verdict does not close")

    stage_inventory = materialized["stage_directional_l2_inventory"]
    if not isinstance(stage_inventory, Mapping):
        raise ValueError("stage directional inventory must be an object")
    _exact_keys(
        stage_inventory,
        {
            "scope",
            "share_denominator",
            "summed_four_stage_directional_l2",
            "stages",
            "dominant_stage_boundary_id",
            "dominant_stage_directional_l2_share",
            "four_stage_coherence_established",
            "caveat",
            "inventory_sha256",
        },
        label="stage directional inventory",
    )
    stages = stage_inventory.get("stages")
    expected_stage_identities = (
        (0, "tied_lexical_hidden_column", "lexical_ingress", 0, [283, 439]),
        (1, "attention_qo_head", "layer.10.attention", 21, [680, 681]),
        (2, "mlp_channel", "layer.17.mlp", 36, [35963, 37381]),
        (3, "tied_lexical_hidden_column", "lexical_readout", 38, [393, 585]),
    )
    if (
        not isinstance(stages, list)
        or len(stages) != 4
        or stage_inventory.get("inventory_sha256") != _sha256(stages)
        or sum(
            item.get("dominant") is True
            for item in stages
            if isinstance(item, Mapping)
        )
        != 1
        or stage_inventory.get("dominant_stage_boundary_id")
        != "lexical_readout"
        or stage_inventory.get("share_denominator")
        != "l2_norm_of_summed_four_stage_direction"
        or stage_inventory.get("scope") != "all_12_reused_C_rows"
        or stage_inventory.get("summed_four_stage_directional_l2")
        != PUBLISHED_TOTAL_DIRECTIONAL_L2
        or stage_inventory.get("four_stage_coherence_established") is not False
        or stage_inventory.get("caveat")
        != (
            "one stage dominates the first-order direction, so local "
            "agreement cannot by itself establish four-stage coherence"
        )
    ):
        raise ValueError("stage directional inventory does not close")
    for index, (item, identity, expected_l2) in enumerate(
        zip(
            stages,
            expected_stage_identities,
            PUBLISHED_STAGE_DIRECTIONAL_L2,
            strict=True,
        )
    ):
        if not isinstance(item, Mapping):
            raise ValueError("stage directional item must be an object")
        _exact_keys(
            item,
            {
                "stage_index",
                "family_id",
                "boundary_id",
                "execution_ordinal",
                "atom_ordinals",
                "directional_l2",
                "directional_l2_share",
                "dominant",
            },
            label="stage directional item",
        )
        observed_identity = (
            item["stage_index"],
            item["family_id"],
            item["boundary_id"],
            item["execution_ordinal"],
            item["atom_ordinals"],
        )
        expected_share = expected_l2 / PUBLISHED_TOTAL_DIRECTIONAL_L2
        if (
            type(item["stage_index"]) is not int
            or type(item["execution_ordinal"]) is not int
            or
            observed_identity != identity
            or item["directional_l2"] != expected_l2
            or item["directional_l2_share"] != expected_share
            or item["dominant"] is not (index == 3)
        ):
            raise ValueError("published stage directional inventory drifted")
    if stage_inventory.get("dominant_stage_directional_l2_share") != (
        PUBLISHED_STAGE_DIRECTIONAL_L2[-1] / PUBLISHED_TOTAL_DIRECTIONAL_L2
    ):
        raise ValueError("dominant stage directional share drifted")

    expected_outcome = (
        "local_only_with_finite_nonlinearity_on_path_active_reused_C"
        if finite_nonlinearity and not primary_local
        else (
            "wide_range_first_order_agreement_on_path_active_reused_C_only"
            if active_wide_range and not primary_local
            else (
                "local_first_order_binding_observed_chart_wide_reused_C_only"
                if primary_local
                else "no_local_first_order_binding_observed_on_reused_C"
            )
        )
    )
    firewall = materialized["execution_firewall"]
    validation_boundary = materialized["validation_boundary"]
    status = materialized["scientific_status"]
    claims = materialized["claims"]
    if not all(
        isinstance(item, Mapping)
        for item in (firewall, validation_boundary, status, claims)
    ):
        raise ValueError("dose-curve firewall or claim sections are invalid")
    _exact_keys(
        firewall,
        {
            "model_load_count",
            "A_B_selection_replay_count",
            "fit_refit_or_reselection_count",
            "finite_candidate_count",
            "finite_alpha_count",
            "C_row_count_per_alpha",
            "chart_only_mask_preserved",
            "source_model_unchanged",
            "source_execution_unchanged",
            "local_files_only",
        },
        label="dose-curve execution firewall",
    )
    expected_validation_boundary = {
        "structural_consistency_validated": True,
        "published_alpha_half_replay_authenticated": True,
        "novel_intervened_scores_externally_attested": False,
        "novel_scores_cryptographically_bound_to_native_receipts": False,
        "independent_novel_alpha_rerun_completed": False,
        "trust_requirement": (
            "the final report hash is a publication commitment after this "
            "execution; novel observations require an independent rerun "
            "for external confirmation"
        ),
    }
    if validation_boundary != expected_validation_boundary:
        raise ValueError("dose-curve validation boundary drifted")
    _exact_keys(
        status,
        {
            "exploratory_post_C",
            "same_C_rows_reused",
            "fresh_confirmation",
            "supermode_confirmed",
            "chart_wide_uniform_pass",
            "finite_nonlinearity_observed",
            "wide_range_first_order_agreement",
            "path_active_subgroup_is_n3_two_families",
            "four_stage_coherence_established",
            "novel_alpha_observations_externally_confirmed",
            "compression_attempted",
        },
        label="dose-curve scientific status",
    )
    _exact_keys(
        claims,
        {
            "fresh_confirmation",
            "supermode_confirmed",
            "four_stage_coherence",
            "compression_success",
            "parameter_savings",
            "MAC_savings",
            "latency_improvement",
        },
        label="dose-curve claims",
    )
    if (
        materialized["diagnostic_outcome"] != expected_outcome
        or any(
            type(firewall.get(name)) is not int
            for name in (
                "model_load_count",
                "A_B_selection_replay_count",
                "fit_refit_or_reselection_count",
                "finite_candidate_count",
                "finite_alpha_count",
                "C_row_count_per_alpha",
            )
        )
        or firewall.get("model_load_count") != 1
        or firewall.get("A_B_selection_replay_count") != 1
        or firewall.get("fit_refit_or_reselection_count") != 0
        or firewall.get("finite_candidate_count") != 1
        or firewall.get("finite_alpha_count") != 4
        or firewall.get("C_row_count_per_alpha") != 12
        or firewall.get("chart_only_mask_preserved") is not True
        or firewall.get("local_files_only") is not True
        or firewall.get("source_model_unchanged") is not True
        or firewall.get("source_execution_unchanged") is not True
        or status.get("exploratory_post_C") is not True
        or status.get("same_C_rows_reused") is not True
        or status.get("fresh_confirmation") is not False
        or status.get("supermode_confirmed") is not False
        or status.get("chart_wide_uniform_pass") is not primary_local
        or status.get("finite_nonlinearity_observed") is not finite_nonlinearity
        or status.get("wide_range_first_order_agreement") is not active_wide_range
        or status.get("path_active_subgroup_is_n3_two_families") is not True
        or status.get("four_stage_coherence_established") is not False
        or status.get("novel_alpha_observations_externally_confirmed") is not False
        or status.get("compression_attempted") is not False
        or any(item is not False for item in claims.values())
    ):
        raise ValueError("dose-curve claim boundary or execution firewall drifted")
    materialized["report_sha256"] = report_sha
    return materialized


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_post_c_dose_curve_report(path: Path) -> dict[str, object]:
    value = _load_strict_json_object(
        path,
        label="post-C dose-curve report",
    )
    return validate_post_c_dose_curve_report(value)


def write_post_c_dose_curve_report(
    report: PartsFirstPostCDoseCurveReport | Mapping[str, object],
    destination: Path | str,
) -> dict[str, object]:
    value = report.to_dict() if isinstance(
        report,
        PartsFirstPostCDoseCurveReport,
    ) else report
    validated = validate_post_c_dose_curve_report(value)
    target = Path(destination)
    if target.suffix != ".json":
        raise ValueError("post-C dose-curve output must use .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    temporary_exists = True
    published_stat: os.stat_result | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(validated, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        loaded = _load_post_c_dose_curve_report(temporary)
        if loaded != validated:
            raise RuntimeError("dose-curve report changed across strict temp reload")
        published_stat = temporary.stat()
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite a post-C dose-curve report"
            ) from error
        temporary.unlink()
        temporary_exists = False
        try:
            _fsync_directory(target.parent)
            if not os.path.samestat(published_stat, target.stat()):
                raise RuntimeError(
                    "dose-curve target inode changed during publication"
                )
        except BaseException:
            # Once linked, the pathname may be replaced by another actor.
            # Never unlink it during error cleanup based on a racy stat check.
            raise
        return loaded
    finally:
        if temporary_exists:
            temporary.unlink(missing_ok=True)


RuntimeLoader = Callable[[PartsFirstModelRunnerConfig], _DoseCurveRuntime]


def _default_runtime_loader(
    config: PartsFirstModelRunnerConfig,
) -> _DoseCurveRuntime:
    return PartsFirstLiveModelPilot.load(config)


def run_parts_first_post_c_dose_curve(
    *,
    config: PartsFirstDoseCurveConfig = PartsFirstDoseCurveConfig(),
    runtime_loader: RuntimeLoader | None = None,
    write_output: bool = True,
) -> PartsFirstPostCDoseCurveReport:
    """Run the frozen four-alpha curve with one local-cache-only model load."""

    frozen = load_and_replay_published_pilot(config)
    model_config = PartsFirstModelRunnerConfig(
        panel_path=config.panel_path,
        cache_dir=config.cache_dir,
        local_files_only=True,
        maximum_prompt_tokens=config.maximum_prompt_tokens,
        capture_output=config.capture_path,
        report_output=config.source_report_path,
    )
    loader = _default_runtime_loader if runtime_loader is None else runtime_loader
    runtime = loader(model_config)
    _bind_published_capture_to_runtime(runtime, frozen)
    sweep = _run_frozen_native_sweep(runtime, frozen)
    report = PartsFirstPostCDoseCurveReport(
        _build_report_payload(frozen, sweep)
    )
    if write_output:
        write_post_c_dose_curve_report(report, config.output_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE_OUTPUT)
    parser.add_argument(
        "--source-report",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DOSE_CURVE_OUTPUT)
    parser.add_argument("--panel", type=Path, default=PINNED_PANEL_PATH)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PartsFirstDoseCurveConfig(
        capture_path=args.capture,
        source_report_path=args.source_report,
        output_path=args.output,
        panel_path=args.panel,
        cache_dir=args.cache_dir,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
    )
    if args.preflight_only:
        output = build_post_c_dose_curve_preflight(config)
    else:
        report = run_parts_first_post_c_dose_curve(config=config)
        output = {
            "diagnostic_outcome": report.payload["diagnostic_outcome"],
            "output": config.output_path.as_posix(),
            "report_sha256": report.report_sha256,
        }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
