"""Run the preregistered exact-model Gemma edge-sidecar experiment.

This worker deliberately keeps the model and its complete 39-event execution
exact.  The 73 unified chart sites are detached observations; local predictors
and dense cross-block edges are fitted after execution and are never installed
in the residual carrier or lexical head.

The module is model-facing, but importing it and constructing its protocol or
report helpers performs no model load and no experiment run.  The held-out
causal rung is exposed as a callback seam so a separately audited JVP/shadow
worker can consume the frozen C split without gaining a model mutation path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters import Gemma3CausalLMAdapter, SequenceContext
from .gemma3_ablation_experiment import _FrozenModelTensorGuard
from .gemma3_cross_layer_chart_transport_model_experiment import (
    _tokenize_keyed_batches,
)
from .gemma3_exact_edge_sidecar_protocol import (
    DEFAULT_OUTPUT as DEFAULT_PROTOCOL_PATH,
    PINNED_CHART_SITE_COUNT,
    PINNED_EVENT_COUNT,
    PINNED_HIDDEN_WIDTH,
    PINNED_LAYER_COUNT,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_PANEL_PATH,
    PINNED_PANEL_SHA256,
    PINNED_SOURCE_PARAMETER_COUNT,
    PINNED_SPLIT_ID_SHA256S,
    build_exact_edge_sidecar_protocol,
    load_exact_edge_sidecar_protocol,
)
from .gemma3_exact_edge_sidecar_causality import (
    ExactEdgeCausalEvaluation,
    evaluate_exact_dense_edge_causality,
)
from .gemma3_experiment import (
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_joint_cross_block_chart_executor import (
    JOINT_EXACT_CONTROL_ARM_ID,
    Gemma3JointCrossBlockChartExecutor,
    JointGraphWholeModelExecution,
    build_joint_cross_block_event_schedule,
)
from .gemma3_joint_cross_block_compiler import (
    Gemma3JointGraphCompilation,
    compile_gemma3_joint_exact_control,
)
from .gemma3_one_block_residual_form_experiment import (
    _LogitMetricAccumulator,
    _native_nll_sum,
)
from .gemma3_unified_event_chart_sidecar import (
    FittedUnifiedEventDenseEdge,
    FittedUnifiedEventLocalBaseline,
    UnifiedEventChartCapture,
    UnifiedEventChartPacket,
    UnifiedEventChartRecorder,
    UnifiedEventLocalBaseline,
    apply_fitted_unified_event_local_baseline,
    evaluate_fitted_unified_event_local_baseline,
    fit_fisher_weighted_unified_event_local_baseline,
    fit_forward_unified_event_dense_edge,
    merge_unified_event_chart_captures,
    packet_from_unified_event_site,
)


EXACT_EDGE_SIDECAR_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_exact_edge_sidecar.report.v1"
)
EXACT_EDGE_SIDECAR_EXPERIMENT_FORMAT_VERSION = 1
EXPECTED_PROTOCOL_SHA256 = (
    "c6adfb5d8b8be32e0917e704c97860101d8cd05b5f0c1f343cdebdcdeb78cd96"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/exact-edge-sidecar/report-v1.json"
)
DEFAULT_BATCH_SIZE = 4
DETERMINISTIC_SEED = 1729
RELATIVE_RIDGE = 1.0e-4
BOOTSTRAP_RESAMPLES = 10_000
JVP_DIRECTION_COUNT_PER_EDGE = 8
INTERVENTION_EPSILON_LADDER = (0.001, 0.003, 0.01)
DIRECTION_PROTOCOL_LITERAL = (
    "deterministic_Rademacher_minus_one_plus_one_on_valid_source_rows_"
    "with_RMS_exactly_one_not_selected_on_C"
)
DIRECTION_IMPLEMENTATION_RECEIPT = (
    "valid_row_rademacher_RMS_one_then_scaled_by_source_RMS"
)
FINITE_DISPLACEMENT_SCALE_LITERAL = (
    "epsilon_times_baseline_source_site_RMS_times_Rademacher_direction"
)
ROLE_ORDER = ("A_fit", "B_predictive", "C_intervention")
ARM_ORDER = (
    "local_target_baseline",
    "real_forward_cross_block_edges",
    "zero_edge_control",
    "row_shuffled_control",
    "family_shuffled_control",
    "source_provenance_permutation_control",
)
EDGE_ARM_ORDER = ARM_ORDER[1:]
NEGATIVE_CONTROL_ARM_IDS = ARM_ORDER[2:]
TARGET_EVENT_IDS = tuple(f"mutation.{index}" for index in range(1, 37))
_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-exact-edge-sidecar-report:v1\0"
)
_ARTIFACT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-exact-edge-sidecar-artifact:v1\0"
)
_EXECUTION_HASH_DOMAIN = (
    b"fisher_graph:gemma3-exact-edge-sidecar-execution:v1\0"
)
_DESIGN_HASH_DOMAIN = (
    b"fisher_graph:gemma3-exact-edge-sidecar-design:v1\0"
)


def _progress(message: str) -> None:
    print(f"[exact-edge-sidecar] {message}", flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes = b"") -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _report_sha256(value: Mapping[str, object]) -> str:
    return _json_sha256(value, domain=_REPORT_HASH_DOMAIN)


def _tensor_sha256(value: Tensor, *, metadata: object) -> str:
    if not isinstance(value, Tensor) or value.device.type != "cpu":
        raise ValueError("hashed sidecar tensors must reside on CPU")
    tensor = value.detach().contiguous()
    return hashlib.sha256(
        _ARTIFACT_HASH_DOMAIN
        + _canonical_json_bytes(metadata)
        + tensor.numpy().tobytes()
    ).hexdigest()


def _ordered_ids_sha256(example_ids: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(example_ids))).hexdigest()


def _row_identity_sha256(row_keys: Sequence[tuple[str, int]]) -> str:
    return _json_sha256(
        tuple(row_keys),
        domain=b"fisher_graph:exact-edge-sidecar-row-identity:v1\0",
    )


def _peak_resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSD-compatible CI images report KiB.
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class PinnedEdgeSidecarExample:
    example_id: str
    family_id: str
    prompt: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.example_id, self.family_id, self.prompt)
        ):
            raise ValueError("panel example identity and prompt must be nonempty")


@dataclass(frozen=True, slots=True)
class PinnedEdgeSidecarSplit:
    panel_path: Path
    panel_sha256: str
    A_fit: tuple[PinnedEdgeSidecarExample, ...]
    B_predictive: tuple[PinnedEdgeSidecarExample, ...]
    C_intervention: tuple[PinnedEdgeSidecarExample, ...]

    def __post_init__(self) -> None:
        expected_counts = {"A_fit": 36, "B_predictive": 12, "C_intervention": 12}
        seen: set[str] = set()
        families: set[str] = set()
        for role in ROLE_ORDER:
            examples = getattr(self, role)
            if len(examples) != expected_counts[role]:
                raise ValueError(f"{role} example count does not match the protocol")
            ids = tuple(item.example_id for item in examples)
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("edge-sidecar split examples must be disjoint")
            seen.update(ids)
            families.update(item.family_id for item in examples)
            if _ordered_ids_sha256(ids) != PINNED_SPLIT_ID_SHA256S[role]:
                raise ValueError(f"{role} ordered identity SHA-256 drifted")
        if len(seen) != 60 or len(families) != 6:
            raise ValueError("edge-sidecar panel inventory did not close")
        if self.panel_sha256 != PINNED_PANEL_SHA256:
            raise ValueError("edge-sidecar panel SHA-256 is not pinned")

    def examples_for(self, role: str) -> tuple[PinnedEdgeSidecarExample, ...]:
        if role not in ROLE_ORDER:
            raise ValueError("capture role is not preregistered")
        return getattr(self, role)

    @property
    def family_by_example(self) -> dict[str, str]:
        return {
            item.example_id: item.family_id
            for role in ROLE_ORDER
            for item in getattr(self, role)
        }

    def metadata(self) -> dict[str, object]:
        return {
            "panel_path": str(self.panel_path),
            "panel_sha256": self.panel_sha256,
            "family_count": 6,
            "total_example_count": 60,
            "roles": {
                role: {
                    "example_count": len(getattr(self, role)),
                    "example_ids_sha256": _ordered_ids_sha256(
                        tuple(item.example_id for item in getattr(self, role))
                    ),
                    "family_assignment_sha256": _json_sha256(
                        tuple(
                            (item.example_id, item.family_id)
                            for item in getattr(self, role)
                        ),
                        domain=(
                            b"fisher_graph:exact-edge-sidecar-family-role:v1\0"
                        ),
                    ),
                }
                for role in ROLE_ORDER
            },
            "example_and_prompt_disjoint_across_roles": True,
            "families_overlap_across_roles": True,
            "contains_prompt_text": False,
            "contains_token_ids": False,
        }


def load_pinned_edge_sidecar_split(
    panel_path: Path | str = PINNED_PANEL_PATH,
) -> PinnedEdgeSidecarSplit:
    """Open and authenticate the frozen 60-example A/B/C panel."""

    path = Path(panel_path)
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("pinned edge-sidecar panel is unavailable") from error
    if hashlib.sha256(raw_bytes).hexdigest() != PINNED_PANEL_SHA256:
        raise ValueError("pinned edge-sidecar panel bytes drifted")
    examples = payload.get("examples") if isinstance(payload, Mapping) else None
    if not isinstance(examples, list) or len(examples) != 60:
        raise ValueError("pinned edge-sidecar panel must contain 60 examples")
    by_family: dict[str, list[PinnedEdgeSidecarExample]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for raw in examples:
        if not isinstance(raw, Mapping):
            raise ValueError("pinned edge-sidecar example must be an object")
        example = PinnedEdgeSidecarExample(
            example_id=raw.get("example_id"),  # type: ignore[arg-type]
            family_id=raw.get("family_id"),  # type: ignore[arg-type]
            prompt=raw.get("prompt"),  # type: ignore[arg-type]
        )
        if example.example_id in seen_ids or example.prompt in seen_prompts:
            raise ValueError("pinned panel ids and prompts must be unique")
        seen_ids.add(example.example_id)
        seen_prompts.add(example.prompt)
        by_family[example.family_id].append(example)
    if len(by_family) != 6 or any(len(rows) != 10 for rows in by_family.values()):
        raise ValueError("pinned panel must contain six ten-example families")
    role_rows: dict[str, list[PinnedEdgeSidecarExample]] = {
        role: [] for role in ROLE_ORDER
    }
    for family_id in sorted(by_family):
        rows = by_family[family_id]
        role_rows["A_fit"].extend(rows[:6])
        role_rows["B_predictive"].extend(rows[6:8])
        role_rows["C_intervention"].extend(rows[8:10])
    return PinnedEdgeSidecarSplit(
        panel_path=path,
        panel_sha256=PINNED_PANEL_SHA256,
        A_fit=tuple(role_rows["A_fit"]),
        B_predictive=tuple(role_rows["B_predictive"]),
        C_intervention=tuple(role_rows["C_intervention"]),
    )


@dataclass(frozen=True, slots=True)
class EdgeSidecarPreparedBatch:
    input_ids: Tensor
    sequence: SequenceContext
    valid_mask: Tensor
    row_keys: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_ids, Tensor)
            or self.input_ids.ndim != 2
            or self.input_ids.dtype not in (torch.int32, torch.int64)
            or not isinstance(self.sequence, SequenceContext)
            or tuple(self.input_ids.shape)
            != (self.sequence.batch_size, self.sequence.query_length)
        ):
            raise ValueError("prepared edge-sidecar input batch is invalid")
        mask = self.valid_mask
        if (
            not isinstance(mask, Tensor)
            or mask.shape != self.input_ids.shape
            or mask.dtype != torch.bool
            or not torch.equal(mask.cpu(), self.sequence.query_valid_mask.cpu())
        ):
            raise ValueError("prepared edge-sidecar valid mask drifted")
        if len(self.row_keys) != int(mask.sum().item()):
            raise ValueError("prepared batch row identity does not match its mask")
        if len(set(self.row_keys)) != len(self.row_keys):
            raise ValueError("prepared batch row keys must be unique")


@dataclass(frozen=True, slots=True)
class EdgeSidecarRoleCapture:
    role: str
    capture: UnifiedEventChartCapture
    score_gradients: tuple[tuple[str, Tensor], ...]
    example_ids_sha256: str
    family_by_example: tuple[tuple[str, str], ...]
    exact_gate: Mapping[str, object]
    exact_forward_count: int
    native_forward_count: int
    backward_count: int

    def __post_init__(self) -> None:
        if self.role not in ROLE_ORDER:
            raise ValueError("edge-sidecar capture role is invalid")
        if not isinstance(self.capture, UnifiedEventChartCapture):
            raise TypeError("edge-sidecar capture is invalid")
        gradients = dict(self.score_gradients)
        if tuple(gradients) != tuple(
            item.site_id for item in self.capture.observations
        ):
            raise ValueError("gradient inventory does not match all 73 sites")
        rows = len(self.capture.row_keys)
        for site_id, value in self.score_gradients:
            if (
                not isinstance(value, Tensor)
                or value.shape != (rows, self.capture.hidden_width)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"score gradient {site_id} is invalid")
        if not _is_sha256(self.example_ids_sha256):
            raise ValueError("capture example identity SHA-256 is invalid")
        if len(dict(self.family_by_example)) * 1 != len(self.family_by_example):
            raise ValueError("capture family mapping contains duplicate examples")
        if (
            self.exact_forward_count <= 0
            or self.backward_count <= 0
            or self.native_forward_count < 0
        ):
            raise ValueError("capture execution counts are invalid")
        if self.exact_gate.get("passed") is not True:
            raise ValueError("exact observer gate did not pass")

    @property
    def gradients(self) -> dict[str, Tensor]:
        return dict(self.score_gradients)

    @property
    def families(self) -> dict[str, str]:
        return dict(self.family_by_example)

    def fisher_row_weights(self, site_id: str) -> Tensor:
        gradient = self.gradients[site_id]
        weights = gradient.double().square().mean(dim=1)
        positive = weights[weights > 0]
        reference = (
            float(positive.mean()) if positive.numel() else 1.0
        )
        floor = max(reference * 1.0e-12, torch.finfo(torch.float64).tiny)
        return weights.clamp_min(floor).to(dtype=torch.float32).contiguous()

    def metadata(self) -> dict[str, object]:
        return {
            "role": self.role,
            "example_ids_sha256": self.example_ids_sha256,
            "row_identity_sha256": self.capture.row_identity_sha256,
            "family_assignment_sha256": _json_sha256(
                tuple(
                    (example_id, family)
                    for example_id, family in self.family_by_example
                ),
                domain=b"fisher_graph:exact-edge-sidecar-family-role:v1\0",
            ),
            "row_count": len(self.capture.row_keys),
            "site_count": len(self.capture.observations),
            "score_gradient_site_count": len(self.score_gradients),
            "hidden_width": self.capture.hidden_width,
            "fisher_row_weight_formula": (
                "mean_hidden_of_exact_next_token_NLL_score_gradient_squared_"
                "with_relative_1e-12_positive_floor"
            ),
            "exact_gate": copy.deepcopy(dict(self.exact_gate)),
            "exact_forward_count": self.exact_forward_count,
            "native_forward_count": self.native_forward_count,
            "backward_count": self.backward_count,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_activations": False,
            "contains_gradients": False,
        }


NativeLogitsProvider = Callable[[EdgeSidecarPreparedBatch], Tensor]


def _next_token_score_objective(
    logits: Tensor,
    batch: EdgeSidecarPreparedBatch,
) -> tuple[Tensor, int]:
    valid = batch.valid_mask[:, :-1] & batch.valid_mask[:, 1:]
    labels = batch.input_ids[:, 1:][valid]
    if labels.numel() == 0:
        raise ValueError("edge-sidecar batch contains no next-token targets")
    objective = F.cross_entropy(
        logits[:, :-1, :][valid].float(),
        labels,
        reduction="sum",
    )
    return objective, int(labels.numel())


def _same_exact_execution(
    baseline: JointGraphWholeModelExecution,
    observed: JointGraphWholeModelExecution,
) -> bool:
    return (
        torch.equal(baseline.output, observed.output)
        and torch.equal(baseline.normalized_output, observed.normalized_output)
        and torch.equal(baseline.logits, observed.logits)
        and baseline.event_ids == observed.event_ids
        and baseline.mutation_receipts == observed.mutation_receipts
        and baseline.operator_receipts == observed.operator_receipts
        and baseline.mutation_chart_input_sha256s
        == observed.mutation_chart_input_sha256s
        and baseline.mutation_delta_sha256s == observed.mutation_delta_sha256s
        and baseline.carrier_version == observed.carrier_version == 36
        and baseline.transport_edge_count == observed.transport_edge_count == 0
    )


def _role_execution_fingerprint(
    *,
    role: str,
    source_fingerprint: str,
    row_keys: Sequence[tuple[str, int]],
) -> str:
    return hashlib.sha256(
        _EXECUTION_HASH_DOMAIN
        + _canonical_json_bytes(
            {
                "role": role,
                "source_fingerprint": source_fingerprint,
                "row_identity_sha256": _row_identity_sha256(row_keys),
            }
        )
    ).hexdigest()


def capture_exact_edge_sidecar_role(
    *,
    executor: Gemma3JointCrossBlockChartExecutor,
    batches: Sequence[EdgeSidecarPreparedBatch],
    role: str,
    source_fingerprint: str,
    example_ids_sha256: str,
    family_by_example: Mapping[str, str],
    native_logits_provider: NativeLogitsProvider | None = None,
) -> EdgeSidecarRoleCapture:
    """Capture all 73 sites and exact score gradients for one frozen role.

    Each batch is executed twice by the exact standalone executor: once with
    the sidecar disabled and once with the detached observer plus one-use
    gradient tape.  A native-logit provider is optional for unit fixtures but
    is required by the canonical model-facing runner.
    """

    if not isinstance(executor, Gemma3JointCrossBlockChartExecutor):
        raise TypeError("executor must be a joint exact graph executor")
    if executor.arm_id != JOINT_EXACT_CONTROL_ARM_ID or len(executor.transports):
        raise ValueError("edge sidecar capture requires the transport-free exact arm")
    if role not in ROLE_ORDER:
        raise ValueError("edge-sidecar capture role is invalid")
    materialized = tuple(batches)
    if not materialized or any(
        not isinstance(item, EdgeSidecarPreparedBatch) for item in materialized
    ):
        raise ValueError("edge-sidecar role requires prepared batches")
    if not _is_sha256(source_fingerprint) or not _is_sha256(example_ids_sha256):
        raise ValueError("edge-sidecar role fingerprints are invalid")
    all_rows = tuple(key for batch in materialized for key in batch.row_keys)
    if len(set(all_rows)) != len(all_rows):
        raise ValueError("edge-sidecar role row identities must be unique")
    ordered_example_ids: list[str] = []
    positions_by_example: dict[str, list[int]] = defaultdict(list)
    for example_id, position in all_rows:
        if example_id not in positions_by_example:
            ordered_example_ids.append(example_id)
        positions_by_example[example_id].append(position)
    if any(
        positions != list(range(len(positions)))
        for positions in positions_by_example.values()
    ):
        raise ValueError("logical token positions must be contiguous per example")
    observed_id_sha = _ordered_ids_sha256(ordered_example_ids)
    if (
        observed_id_sha != example_ids_sha256
        or observed_id_sha != PINNED_SPLIT_ID_SHA256S[role]
    ):
        raise ValueError("prepared batches do not bind the pinned role identities")
    role_examples = {example_id for example_id, _ in all_rows}
    if set(family_by_example) != role_examples or any(
        not isinstance(value, str) or not value
        for value in family_by_example.values()
    ):
        raise ValueError("capture family identities do not close over role rows")
    execution_fingerprint = _role_execution_fingerprint(
        role=role,
        source_fingerprint=source_fingerprint,
        row_keys=all_rows,
    )
    capture_parts: list[UnifiedEventChartCapture] = []
    gradient_parts: dict[str, list[Tensor]] = defaultdict(list)
    native_accumulator = _LogitMetricAccumulator()
    native_nll_sum = 0.0
    native_tokens = 0
    native_valid_query_literal_parity = True
    sidecar_literal_parity = True
    mutation_input_parity = True
    mutation_delta_parity = True
    event_order_equal = True
    source_guard = _FrozenModelTensorGuard(executor)
    schedule = build_joint_cross_block_event_schedule()
    for batch in materialized:
        with torch.inference_mode():
            baseline = executor.forward_components(batch.input_ids, batch.sequence)
            native_logits = (
                None
                if native_logits_provider is None
                else native_logits_provider(batch)
            )
        recorder = UnifiedEventChartRecorder(
            schedule=schedule,
            source_fingerprint=source_fingerprint,
            execution_fingerprint=execution_fingerprint,
            valid_mask=batch.valid_mask,
            row_keys=batch.row_keys,
        )
        observed = executor.forward_components_with_observer(
            batch.input_ids,
            batch.sequence,
            observer=recorder,
            retain_gradient_tape=True,
        )
        part = recorder.finish()
        sidecar_literal_parity = sidecar_literal_parity and _same_exact_execution(
            baseline,
            observed.execution,
        )
        mutation_input_parity = mutation_input_parity and (
            baseline.mutation_chart_input_sha256s
            == observed.execution.mutation_chart_input_sha256s
        )
        mutation_delta_parity = mutation_delta_parity and (
            baseline.mutation_delta_sha256s
            == observed.execution.mutation_delta_sha256s
        )
        event_order_equal = event_order_equal and (
            observed.execution.event_ids == executor.ordered_event_ids
            and len(observed.execution.event_ids) == PINNED_EVENT_COUNT
        )
        objective, _ = _next_token_score_objective(
            observed.execution.logits,
            batch,
        )
        if observed.gradient_tape is None:
            raise RuntimeError("exact observer omitted its gradient tape")
        gradients = observed.gradient_tape.score_gradients(objective)
        for site_id, value in gradients.items():
            gradient_parts[site_id].append(
                value[batch.valid_mask.cpu()].contiguous()
            )
        capture_parts.append(part)
        if native_logits is not None:
            if not isinstance(native_logits, Tensor):
                raise TypeError("native logits provider must return a Tensor")
            native_valid_query_literal_parity = (
                native_valid_query_literal_parity
                and torch.equal(
                    native_logits[batch.valid_mask],
                    baseline.logits[batch.valid_mask],
                )
            )
            native_accumulator.update(
                native_logits,
                baseline.logits,
                {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.valid_mask,
                },
            )
            nll_sum, token_count = _native_nll_sum(
                native_logits,
                {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.valid_mask,
                },
            )
            native_nll_sum += nll_sum
            native_tokens += token_count
    source_guard.assert_unchanged()
    capture = merge_unified_event_chart_captures(capture_parts)
    ordered_site_ids = tuple(item.site_id for item in capture.observations)
    merged_gradients = tuple(
        (
            site_id,
            torch.cat(gradient_parts[site_id], dim=0)
            .to(device="cpu", dtype=torch.float32)
            .contiguous(),
        )
        for site_id in ordered_site_ids
    )
    if native_logits_provider is None:
        native_metrics: dict[str, object] = {
            "run": False,
            "literal_valid_query_logit_parity": None,
        }
    else:
        if native_tokens <= 0:
            raise RuntimeError("native exact gate produced no supervised token")
        candidate = native_accumulator.result(
            native_nll_per_token=native_nll_sum / native_tokens
        )
        native_metrics = {
            "run": True,
            "literal_valid_query_logit_parity": (
                native_valid_query_literal_parity
            ),
            "supervised_token_count": native_tokens,
            **candidate,
        }
    exact_gate = {
        "passed": bool(
            sidecar_literal_parity
            and event_order_equal
            and (
                native_logits_provider is None
                or (
                    native_valid_query_literal_parity
                    and native_metrics.get("delta_nll_per_token") == 0.0
                    and native_metrics.get("native_to_candidate_kl_per_query")
                    == 0.0
                    and native_metrics.get("top1_agreement_to_native") == 1.0
                    and native_metrics.get("maximum_absolute_logit_error") == 0.0
                )
            )
        ),
        "event_order_equal": event_order_equal,
        "every_mutation_chart_input_byte_identical": mutation_input_parity,
        "every_mutation_delta_byte_identical": mutation_delta_parity,
        "final_carrier_byte_identical": sidecar_literal_parity,
        "normalized_output_byte_identical": sidecar_literal_parity,
        "logits_byte_identical": sidecar_literal_parity,
        "source_state_fingerprint_unchanged": True,
        "native_exact_control": native_metrics,
    }
    if exact_gate["passed"] is not True:
        failed_fields = tuple(
            key
            for key, value in exact_gate.items()
            if key != "native_exact_control" and value is not True
        )
        raise RuntimeError(
            "exact 39-event sidecar gate failed literal parity: "
            f"failed_fields={failed_fields!r}; "
            f"native_exact_control={native_metrics!r}"
        )
    return EdgeSidecarRoleCapture(
        role=role,
        capture=capture,
        score_gradients=merged_gradients,
        example_ids_sha256=example_ids_sha256,
        family_by_example=tuple(
            (example_id, family_by_example[example_id])
            for example_id in ordered_example_ids
        ),
        exact_gate=exact_gate,
        exact_forward_count=2 * len(materialized),
        native_forward_count=(
            len(materialized) if native_logits_provider is not None else 0
        ),
        backward_count=len(materialized),
    )


@dataclass(frozen=True, slots=True)
class FrozenDenseEdgeArtifact:
    edge: FittedUnifiedEventDenseEdge
    claimed_source_site_id: str
    packet_source_site_id: str
    coefficient_sha256: str
    fit_design_sha256: str
    numerical_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.edge, FittedUnifiedEventDenseEdge):
            raise TypeError("edge artifact must contain a dense edge")
        if self.claimed_source_site_id != self.edge.spec.source_site_id:
            raise ValueError("edge artifact source label drifted")
        if not isinstance(self.packet_source_site_id, str) or not self.packet_source_site_id:
            raise ValueError("edge artifact packet source is invalid")
        if not _is_sha256(self.coefficient_sha256) or not _is_sha256(
            self.fit_design_sha256
        ):
            raise ValueError("edge artifact hashes are invalid")
        coefficient = self.edge.coefficient
        if coefficient.shape != (PINNED_HIDDEN_WIDTH, PINNED_HIDDEN_WIDTH):
            raise ValueError("edge artifact is not a dense full-width 640x640 map")
        if self.edge.edge_rank != PINNED_HIDDEN_WIDTH:
            raise ValueError("edge artifact capacity is not full width")
        if (
            type(self.numerical_rank) is not int
            or self.numerical_rank <= 0
            or self.numerical_rank > PINNED_HIDDEN_WIDTH
            or self.numerical_rank != self.edge.realized_numerical_rank
        ):
            raise ValueError("edge numerical rank is invalid")
        expected = _tensor_sha256(
            coefficient,
            metadata={
                "spec": self.edge.spec.metadata(),
                "shape": list(coefficient.shape),
                "dtype": str(coefficient.dtype),
                "fit_design_sha256": self.fit_design_sha256,
            },
        )
        if expected != self.coefficient_sha256:
            raise ValueError("edge coefficient SHA-256 does not close")

    def assert_integrity(self) -> None:
        expected = _tensor_sha256(
            self.edge.coefficient,
            metadata={
                "spec": self.edge.spec.metadata(),
                "shape": list(self.edge.coefficient.shape),
                "dtype": str(self.edge.coefficient.dtype),
                "fit_design_sha256": self.fit_design_sha256,
            },
        )
        if (
            expected != self.coefficient_sha256
            or self.edge.realized_numerical_rank != self.numerical_rank
        ):
            raise RuntimeError("frozen dense edge coefficient changed")

    def metadata(self) -> dict[str, object]:
        self.assert_integrity()
        result = {
            **self.edge.metadata(),
            "claimed_source_site_id": self.claimed_source_site_id,
            "packet_source_site_id": self.packet_source_site_id,
            "coefficient_sha256": self.coefficient_sha256,
            "fit_design_sha256": self.fit_design_sha256,
            "numerical_rank": self.numerical_rank,
            "coefficient_serialized_in_report": False,
            "dense_640_by_640": True,
            "factorized": False,
        }
        if self.edge.spec.source_event_kind == "lexical_ingress":
            result["source_component_family"] = "lexical"
            result["runtime_source_component_family"] = (
                self.edge.spec.source_component_family
            )
        return result


@dataclass(frozen=True, slots=True)
class FrozenEdgeSidecarArm:
    arm_id: str
    edges: tuple[FrozenDenseEdgeArtifact, ...]
    hard_zero: bool
    fit_permutation_sha256: str | None

    def __post_init__(self) -> None:
        if self.arm_id not in EDGE_ARM_ORDER:
            raise ValueError("edge arm id is not preregistered")
        if type(self.hard_zero) is not bool:
            raise TypeError("edge arm hard-zero flag must be boolean")
        if self.hard_zero != (self.arm_id == "zero_edge_control"):
            raise ValueError("only the zero-edge arm may be hard zero")
        if len({item.edge.spec.target_event_id for item in self.edges}) != len(
            self.edges
        ):
            raise ValueError("edge arm exceeds one incoming edge per target")
        if any(
            not item.edge.spec.cross_block
            or not item.edge.spec.cross_family
            or item.edge.spec.source_event_index >= item.edge.spec.target_event_index
            for item in self.edges
        ):
            raise ValueError("edge arm contains an ineligible edge")
        if self.fit_permutation_sha256 is not None and not _is_sha256(
            self.fit_permutation_sha256
        ):
            raise ValueError("edge arm permutation hash is invalid")

    @property
    def topology(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                item.claimed_source_site_id,
                item.edge.spec.target_event_id,
            )
            for item in self.edges
        )

    @property
    def stored_scalar_count(self) -> int:
        return 0 if self.hard_zero else sum(
            item.edge.stored_scalar_count for item in self.edges
        )

    def metadata(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "edge_count": len(self.edges),
            "hard_zero": self.hard_zero,
            "fit_permutation_sha256": self.fit_permutation_sha256,
            "maximum_incoming_edges_per_target": 1,
            "stored_scalar_count": self.stored_scalar_count,
            "edges": [item.metadata() for item in self.edges],
            "model_weights_mutated": False,
            "model_carrier_mutated_by_sidecar": False,
            "model_logits_mutated_by_sidecar": False,
            "compression": False,
        }


@dataclass(frozen=True, slots=True)
class FittedEdgeSidecarCatalog:
    local_baselines: tuple[FittedUnifiedEventLocalBaseline, ...]
    arms: tuple[FrozenEdgeSidecarArm, ...]
    artifact_sha256: str
    fit_candidate_count: int
    fit_dense_solve_count: int
    fit_design_mac_estimate: int
    fit_dense_solve_flop_lower_bound: int
    rank_decomposition_flop_lower_bound: int
    provenance_ineligible_target_count: int

    def __post_init__(self) -> None:
        if (
            len(self.local_baselines) != 36
            or tuple(item.event_id for item in self.local_baselines)
            != TARGET_EVENT_IDS
        ):
            raise ValueError("local baseline catalog must cover all 36 mutations")
        if tuple(item.arm_id for item in self.arms) != EDGE_ARM_ORDER:
            raise ValueError("edge arm catalog order drifted")
        real = self.arm("real_forward_cross_block_edges")
        if not real.edges:
            raise ValueError("real edge catalog selected no A-fit edge")
        real_topology = real.topology
        for arm in self.arms[1:]:
            if arm.topology != real_topology:
                raise ValueError("negative controls do not share real topology/budget")
        if not _is_sha256(self.artifact_sha256):
            raise ValueError("edge catalog SHA-256 is invalid")
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.fit_candidate_count,
                self.fit_dense_solve_count,
                self.fit_design_mac_estimate,
                self.fit_dense_solve_flop_lower_bound,
                self.rank_decomposition_flop_lower_bound,
            )
        ):
            raise ValueError("edge catalog analysis accounting is invalid")
        if (
            type(self.provenance_ineligible_target_count) is not int
            or self.provenance_ineligible_target_count < 0
        ):
            raise ValueError("provenance ineligible target count is invalid")
        if self.artifact_sha256 != _catalog_sha256(self, include_digest=False):
            raise ValueError("edge catalog SHA-256 does not close")

    def arm(self, arm_id: str) -> FrozenEdgeSidecarArm:
        for arm in self.arms:
            if arm.arm_id == arm_id:
                return arm
        raise KeyError(arm_id)

    @property
    def local_baseline_scalar_count(self) -> int:
        return sum(item.stored_scalar_count for item in self.local_baselines)

    @property
    def sidecar_edge_scalar_count(self) -> int:
        return sum(item.stored_scalar_count for item in self.arms)

    def metadata(self, *, include_digest: bool = True) -> dict[str, object]:
        self.assert_integrity()
        result = _catalog_payload(
            local_baselines=self.local_baselines,
            arms=self.arms,
            fit_candidate_count=self.fit_candidate_count,
            fit_dense_solve_count=self.fit_dense_solve_count,
            fit_design_mac_estimate=self.fit_design_mac_estimate,
            fit_dense_solve_flop_lower_bound=(
                self.fit_dense_solve_flop_lower_bound
            ),
            rank_decomposition_flop_lower_bound=(
                self.rank_decomposition_flop_lower_bound
            ),
            provenance_ineligible_target_count=(
                self.provenance_ineligible_target_count
            ),
        )
        if include_digest:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def assert_integrity(self) -> None:
        for arm in self.arms:
            for artifact in arm.edges:
                artifact.assert_integrity()
        if self.artifact_sha256 != _catalog_sha256(self, include_digest=False):
            raise RuntimeError("frozen edge-sidecar catalog tensors changed")


def _catalog_payload(
    *,
    local_baselines: Sequence[FittedUnifiedEventLocalBaseline],
    arms: Sequence[FrozenEdgeSidecarArm],
    fit_candidate_count: int,
    fit_dense_solve_count: int,
    fit_design_mac_estimate: int,
    fit_dense_solve_flop_lower_bound: int,
    rank_decomposition_flop_lower_bound: int,
    provenance_ineligible_target_count: int,
) -> dict[str, object]:
    local = tuple(local_baselines)
    frozen_arms = tuple(arms)
    real = next(
        item
        for item in frozen_arms
        if item.arm_id == "real_forward_cross_block_edges"
    )
    return {
            "schema": "fisher_graph.gemma3_exact_edge_sidecar.edge_arm_catalog.v1",
            "local_baseline_count": len(local),
            "local_baseline_scalar_count": sum(
                item.stored_scalar_count for item in local
            ),
            "local_baselines": [
                {
                    **item.metadata(),
                    "weight_sha256": _tensor_sha256(
                        item.weight,
                        metadata={
                            "event_id": item.event_id,
                            "kind": "local_baseline_weight",
                        },
                    ),
                    "bias_sha256": _tensor_sha256(
                        item.bias,
                        metadata={
                            "event_id": item.event_id,
                            "kind": "local_baseline_bias",
                        },
                    ),
                    "coefficients_serialized_in_report": False,
                }
                for item in local
            ],
            "arm_order": list(EDGE_ARM_ORDER),
            "arms": [item.metadata() for item in frozen_arms],
            "real_edge_count": len(real.edges),
            "sidecar_edge_scalar_count": sum(
                item.stored_scalar_count for item in frozen_arms
            ),
            "fit_candidate_count": fit_candidate_count,
            "fit_dense_solve_count": fit_dense_solve_count,
            "fit_design_mac_estimate": fit_design_mac_estimate,
            "fit_dense_solve_flop_lower_bound": (
                fit_dense_solve_flop_lower_bound
            ),
            "rank_decomposition_flop_lower_bound": (
                rank_decomposition_flop_lower_bound
            ),
            "provenance_ineligible_target_count": (
                provenance_ineligible_target_count
            ),
            "provenance_ineligible_target_reason": (
                "fewer_than_two_distinct_eligible_source_packets"
            ),
            "cost_estimates_are_backend_independent_lower_bounds": True,
            "full_width_coordinates": True,
            "dense_640_by_640_edges": True,
            "low_rank_factorization_used": False,
            "maximum_real_incoming_edges_per_target": 1,
            "B_or_C_used_for_fit_or_selection": False,
            "installed_in_exact_runtime": False,
            "model_mutated": False,
            "compression": False,
        }


def _catalog_sha256(
    catalog: FittedEdgeSidecarCatalog,
    *,
    include_digest: bool,
) -> str:
    payload = _catalog_payload(
        local_baselines=catalog.local_baselines,
        arms=catalog.arms,
        fit_candidate_count=catalog.fit_candidate_count,
        fit_dense_solve_count=catalog.fit_dense_solve_count,
        fit_design_mac_estimate=catalog.fit_design_mac_estimate,
        fit_dense_solve_flop_lower_bound=(
            catalog.fit_dense_solve_flop_lower_bound
        ),
        rank_decomposition_flop_lower_bound=(
            catalog.rank_decomposition_flop_lower_bound
        ),
        provenance_ineligible_target_count=(
            catalog.provenance_ineligible_target_count
        ),
    )
    if include_digest:
        payload["artifact_sha256"] = catalog.artifact_sha256
    return _json_sha256(payload, domain=_ARTIFACT_HASH_DOMAIN)


def _catalog_sha256_from_parts(
    *,
    local_baselines: Sequence[FittedUnifiedEventLocalBaseline],
    arms: Sequence[FrozenEdgeSidecarArm],
    fit_candidate_count: int,
    fit_dense_solve_count: int,
    fit_design_mac_estimate: int,
    fit_dense_solve_flop_lower_bound: int,
    rank_decomposition_flop_lower_bound: int,
    provenance_ineligible_target_count: int,
) -> str:
    return _json_sha256(
        _catalog_payload(
            local_baselines=local_baselines,
            arms=arms,
            fit_candidate_count=fit_candidate_count,
            fit_dense_solve_count=fit_dense_solve_count,
            fit_design_mac_estimate=fit_design_mac_estimate,
            fit_dense_solve_flop_lower_bound=fit_dense_solve_flop_lower_bound,
            rank_decomposition_flop_lower_bound=(
                rank_decomposition_flop_lower_bound
            ),
            provenance_ineligible_target_count=(
                provenance_ineligible_target_count
            ),
        ),
        domain=_ARTIFACT_HASH_DOMAIN,
    )


def _packet_design_sha256(packet: UnifiedEventChartPacket) -> str:
    coordinates = packet.coordinates[:, 0, :]
    return hashlib.sha256(
        _DESIGN_HASH_DOMAIN
        + _canonical_json_bytes(
            {
                "shape": list(coordinates.shape),
                "dtype": str(coordinates.dtype),
                "chart_index": 0,
            }
        )
        + coordinates.contiguous().numpy().tobytes()
    ).hexdigest()


def _edge_artifact(
    edge: FittedUnifiedEventDenseEdge,
    *,
    claimed_source_site_id: str,
    packet_source_site_id: str,
    fit_design_sha256: str,
) -> FrozenDenseEdgeArtifact:
    numerical_rank = edge.realized_numerical_rank
    metadata = {
        "spec": edge.spec.metadata(),
        "shape": list(edge.coefficient.shape),
        "dtype": str(edge.coefficient.dtype),
        "fit_design_sha256": fit_design_sha256,
    }
    return FrozenDenseEdgeArtifact(
        edge=edge,
        claimed_source_site_id=claimed_source_site_id,
        packet_source_site_id=packet_source_site_id,
        coefficient_sha256=_tensor_sha256(edge.coefficient, metadata=metadata),
        fit_design_sha256=fit_design_sha256,
        numerical_rank=numerical_rank,
    )


def _packet_with_coordinates(
    claimed: UnifiedEventChartPacket,
    coordinates: Tensor,
) -> UnifiedEventChartPacket:
    return UnifiedEventChartPacket(
        event_id=claimed.event_id,
        event_index=claimed.event_index,
        event_kind=claimed.event_kind,
        component_family=claimed.component_family,
        block_ordinal=claimed.block_ordinal,
        site_id=claimed.site_id,
        site_role=claimed.site_role,
        row_keys=claimed.row_keys,
        coordinates=coordinates.detach().to(device="cpu").clone().contiguous(),
        source_fingerprint=claimed.source_fingerprint,
        execution_fingerprint=claimed.execution_fingerprint,
    )


def _all_source_packets(
    capture: UnifiedEventChartCapture,
) -> dict[str, UnifiedEventChartPacket]:
    return {
        site.site_id: packet_from_unified_event_site(
            capture,
            event_id=site.event_id,
            role=site.role,
        )
        for site in capture.observations
    }


def _eligible_sources(
    packets: Mapping[str, UnifiedEventChartPacket],
    target: UnifiedEventLocalBaseline,
) -> tuple[UnifiedEventChartPacket, ...]:
    eligible: list[UnifiedEventChartPacket] = []
    target_block = target.block_ordinal
    if target_block is None:
        raise ValueError("edge target must belong to a Gemma block")
    for packet in packets.values():
        source_block = -1 if packet.event_kind == "lexical_ingress" else packet.block_ordinal
        if (
            source_block is not None
            and source_block < target_block
            and packet.event_index < target.event_index
            and packet.component_family != target.component_family
        ):
            eligible.append(packet)
    keyed = tuple((_packet_design_sha256(item), item) for item in eligible)
    hashes = tuple(key for key, _ in keyed)
    if len(set(hashes)) != len(hashes):
        raise RuntimeError(
            "A-fit discovery has a provenance-blind indistinguishable design tie"
        )
    return tuple(item for _, item in sorted(keyed, key=lambda pair: pair[0]))


def _row_shuffle_permutation(row_count: int) -> Tensor:
    if row_count < 2:
        raise ValueError("row shuffle requires at least two A-fit rows")
    generator = torch.Generator().manual_seed(DETERMINISTIC_SEED)
    permutation = torch.randperm(row_count, generator=generator)
    if bool(torch.equal(permutation, torch.arange(row_count))):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


def _family_shuffle_permutation(
    row_keys: Sequence[tuple[str, int]],
    family_by_example: Mapping[str, str],
) -> Tensor:
    families = tuple(sorted(set(family_by_example.values())))
    if len(families) != 6:
        raise ValueError("family control requires all six prompt families")
    groups: dict[str, list[int]] = defaultdict(list)
    for row_index, (example_id, _position) in enumerate(row_keys):
        groups[family_by_example[example_id]].append(row_index)
    ordered = [
        row
        for family in families
        for row in sorted(
            groups[family],
            key=lambda index: (row_keys[index][1], index),
        )
    ]
    largest = max(len(rows) for rows in groups.values())
    if largest * 2 > len(ordered):
        raise ValueError("family control has no full family derangement")
    donors = ordered[largest:] + ordered[:largest]
    permutation = [-1] * len(row_keys)
    for source, donor in zip(ordered, donors, strict=True):
        permutation[source] = donor
    result = torch.tensor(permutation, dtype=torch.long)
    if sorted(permutation) != list(range(len(row_keys))) or any(
        family_by_example[row_keys[index][0]]
        == family_by_example[row_keys[int(donor)][0]]
        for index, donor in enumerate(result)
    ):
        raise RuntimeError("family derangement did not preserve its firewall")
    return result


def _permutation_sha256(permutation: Tensor, *, kind: str) -> str:
    return hashlib.sha256(
        _ARTIFACT_HASH_DOMAIN
        + _canonical_json_bytes({"kind": kind, "row_count": permutation.numel()})
        + permutation.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _fit_dense_edge(
    *,
    schedule: Sequence[object],
    source: UnifiedEventChartPacket,
    target: UnifiedEventLocalBaseline,
) -> FittedUnifiedEventDenseEdge:
    edge = fit_forward_unified_event_dense_edge(
        schedule=schedule,  # type: ignore[arg-type]
        source=source,
        target=target,
        relative_ridge=RELATIVE_RIDGE,
        require_cross_block=True,
        require_cross_family=True,
    )
    if edge.coefficient.shape != (PINNED_HIDDEN_WIDTH, PINNED_HIDDEN_WIDTH):
        raise RuntimeError("dense edge fitter returned a compressed coefficient")
    return edge


def _fit_dense_artifact(
    *,
    schedule: Sequence[object],
    source: UnifiedEventChartPacket,
    target: UnifiedEventLocalBaseline,
    claimed_source_site_id: str,
    packet_source_site_id: str,
) -> FrozenDenseEdgeArtifact:
    return _edge_artifact(
        _fit_dense_edge(schedule=schedule, source=source, target=target),
        claimed_source_site_id=claimed_source_site_id,
        packet_source_site_id=packet_source_site_id,
        fit_design_sha256=_packet_design_sha256(source),
    )


def fit_exact_edge_sidecar_catalog(
    fit_capture: EdgeSidecarRoleCapture,
) -> FittedEdgeSidecarCatalog:
    """Fit A-only local baselines, discover real edges, and freeze controls."""

    if not isinstance(fit_capture, EdgeSidecarRoleCapture):
        raise TypeError("fit_capture must be an EdgeSidecarRoleCapture")
    if fit_capture.role != "A_fit":
        raise ValueError("edge catalog may only be fitted on A_fit")
    capture = fit_capture.capture
    if (
        capture.hidden_width != PINNED_HIDDEN_WIDTH
        or len(capture.observations) != PINNED_CHART_SITE_COUNT
    ):
        raise ValueError("A-fit capture is not the full-width 73-site contract")
    schedule = capture.schedule
    local_fits: list[FittedUnifiedEventLocalBaseline] = []
    local_targets: dict[str, UnifiedEventLocalBaseline] = {}
    row_count = len(capture.row_keys)
    width = capture.hidden_width
    local_design_macs = 0
    local_solve_flops = 0
    for event_id in TARGET_EVENT_IDS:
        weights = fit_capture.fisher_row_weights(f"{event_id}.output")
        fit = fit_fisher_weighted_unified_event_local_baseline(
            capture=capture,
            event_id=event_id,
            row_fisher_weights=weights,
            relative_ridge=RELATIVE_RIDGE,
        )
        if fit.weight.shape != (width, width):
            raise RuntimeError("local baseline is not exact full width")
        local_fits.append(fit)
        local_targets[event_id] = apply_fitted_unified_event_local_baseline(
            fit,
            capture=capture,
            row_fisher_weights=weights,
        )
        local_design_macs += row_count * (width + 1) * (2 * width + 1)
        design_width = width + 1
        local_solve_flops += math.ceil(
            (2.0 / 3.0) * design_width**3
            + 2.0 * design_width * design_width * width
        )

    packets = _all_source_packets(capture)
    selected: list[FrozenDenseEdgeArtifact] = []
    eligible_by_target: dict[str, tuple[UnifiedEventChartPacket, ...]] = {}
    candidate_count = 0
    dense_solve_count = 0
    edge_design_macs = 0
    edge_solve_flops = 0
    provenance_ineligible_target_count = 0
    for event_id in TARGET_EVENT_IDS:
        target = local_targets[event_id]
        eligible = _eligible_sources(packets, target)
        eligible_by_target[event_id] = eligible
        if len(eligible) < 2:
            provenance_ineligible_target_count += 1
            continue
        best: tuple[
            tuple[float, str, int],
            FittedUnifiedEventDenseEdge,
            UnifiedEventChartPacket,
            str,
        ] | None = None
        for source in eligible:
            candidate_count += 1
            dense_solve_count += 1
            edge_design_macs += 2 * row_count * width * width
            edge_solve_flops += math.ceil((8.0 / 3.0) * width**3)
            try:
                edge = _fit_dense_edge(
                    schedule=schedule,
                    source=source,
                    target=target,
                )
            except RuntimeError as error:
                if str(error) == (
                    "dense edge does not improve the target-local baseline"
                ):
                    continue
                raise
            design_sha = _packet_design_sha256(source)
            key = (-edge.fit_explained_fraction, design_sha, 0)
            if best is None or key < best[0]:
                best = (key, edge, source, design_sha)
        if best is not None:
            _, edge, source, design_sha = best
            selected.append(
                _edge_artifact(
                    edge,
                    claimed_source_site_id=source.site_id,
                    packet_source_site_id=source.site_id,
                    fit_design_sha256=design_sha,
                )
            )
    if not selected:
        raise RuntimeError("A-fit discovery produced no positive dense edge")
    selected.sort(
        key=lambda item: (
            item.edge.spec.target_event_index,
            item.edge.spec.target_chart_index,
        )
    )

    row_permutation = _row_shuffle_permutation(row_count)
    family_permutation = _family_shuffle_permutation(
        capture.row_keys,
        fit_capture.families,
    )
    row_sha = _permutation_sha256(row_permutation, kind="A_fit_row_shuffle")
    family_sha = _permutation_sha256(
        family_permutation,
        kind="A_fit_family_derangement",
    )
    row_edges: list[FrozenDenseEdgeArtifact] = []
    family_edges: list[FrozenDenseEdgeArtifact] = []
    provenance_edges: list[FrozenDenseEdgeArtifact] = []
    provenance_assignments: list[dict[str, str]] = []
    for real in selected:
        claimed = packets[real.claimed_source_site_id]
        target = local_targets[real.edge.spec.target_event_id]
        for destination, permutation in (
            (row_edges, row_permutation),
            (family_edges, family_permutation),
        ):
            shuffled = _packet_with_coordinates(
                claimed,
                claimed.coordinates[permutation],
            )
            dense_solve_count += 1
            edge_design_macs += 2 * row_count * width * width
            edge_solve_flops += math.ceil((8.0 / 3.0) * width**3)
            destination.append(
                _fit_dense_artifact(
                    schedule=schedule,
                    source=shuffled,
                    target=target,
                    claimed_source_site_id=claimed.site_id,
                    packet_source_site_id=claimed.site_id,
                )
            )
        eligible = eligible_by_target[target.event_id]
        design_order = tuple(
            sorted(
                eligible,
                key=_packet_design_sha256,
            )
        )
        if len(design_order) < 2:
            raise RuntimeError("provenance control requires two eligible sources")
        claimed_index = next(
            index
            for index, item in enumerate(design_order)
            if item.site_id == claimed.site_id
        )
        donor = design_order[(claimed_index + 1) % len(design_order)]
        relabeled = _packet_with_coordinates(claimed, donor.coordinates)
        dense_solve_count += 1
        edge_design_macs += 2 * row_count * width * width
        edge_solve_flops += math.ceil((8.0 / 3.0) * width**3)
        provenance_edges.append(
            _fit_dense_artifact(
                schedule=schedule,
                source=relabeled,
                target=target,
                claimed_source_site_id=claimed.site_id,
                packet_source_site_id=donor.site_id,
            )
        )
        provenance_assignments.append(
            {
                "claimed_source_site_id": claimed.site_id,
                "packet_source_site_id": donor.site_id,
                "target_event_id": target.event_id,
            }
        )
    provenance_sha = _json_sha256(
        provenance_assignments,
        domain=_ARTIFACT_HASH_DOMAIN,
    )
    real_arm = FrozenEdgeSidecarArm(
        arm_id="real_forward_cross_block_edges",
        edges=tuple(selected),
        hard_zero=False,
        fit_permutation_sha256=None,
    )
    arms = (
        real_arm,
        FrozenEdgeSidecarArm(
            arm_id="zero_edge_control",
            edges=tuple(selected),
            hard_zero=True,
            fit_permutation_sha256=None,
        ),
        FrozenEdgeSidecarArm(
            arm_id="row_shuffled_control",
            edges=tuple(row_edges),
            hard_zero=False,
            fit_permutation_sha256=row_sha,
        ),
        FrozenEdgeSidecarArm(
            arm_id="family_shuffled_control",
            edges=tuple(family_edges),
            hard_zero=False,
            fit_permutation_sha256=family_sha,
        ),
        FrozenEdgeSidecarArm(
            arm_id="source_provenance_permutation_control",
            edges=tuple(provenance_edges),
            hard_zero=False,
            fit_permutation_sha256=provenance_sha,
        ),
    )
    local_tuple = tuple(local_fits)
    fit_design_macs = local_design_macs + edge_design_macs
    fit_solve_flops = local_solve_flops + edge_solve_flops
    rank_decomposition_count = len(selected) + len(row_edges) + len(
        family_edges
    ) + len(provenance_edges)
    rank_decomposition_flops = rank_decomposition_count * 4 * width**3
    artifact_sha = _catalog_sha256_from_parts(
        local_baselines=local_tuple,
        arms=arms,
        fit_candidate_count=candidate_count,
        fit_dense_solve_count=dense_solve_count,
        fit_design_mac_estimate=fit_design_macs,
        fit_dense_solve_flop_lower_bound=fit_solve_flops,
        rank_decomposition_flop_lower_bound=rank_decomposition_flops,
        provenance_ineligible_target_count=provenance_ineligible_target_count,
    )
    return FittedEdgeSidecarCatalog(
        local_baselines=local_tuple,
        arms=arms,
        artifact_sha256=artifact_sha,
        fit_candidate_count=candidate_count,
        fit_dense_solve_count=dense_solve_count,
        fit_design_mac_estimate=fit_design_macs,
        fit_dense_solve_flop_lower_bound=fit_solve_flops,
        rank_decomposition_flop_lower_bound=rank_decomposition_flops,
        provenance_ineligible_target_count=provenance_ineligible_target_count,
    )


@dataclass(frozen=True, slots=True)
class EdgeSidecarPredictiveReceipt:
    parent_edge_catalog_sha256: str
    B_row_identity_sha256: str
    arm_metrics: tuple[tuple[str, Mapping[str, object]], ...]
    per_target_metrics: tuple[Mapping[str, object], ...]
    real_gain_margin_over_controls: tuple[tuple[str, float], ...]
    paired_family_bootstrap_lower_bound: float
    bootstrap_resamples: int
    prediction_mac_estimate: int
    passed: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.parent_edge_catalog_sha256) or not _is_sha256(
            self.B_row_identity_sha256
        ):
            raise ValueError("predictive receipt parent/row hash is invalid")
        if tuple(key for key, _ in self.arm_metrics) != ARM_ORDER:
            raise ValueError("predictive receipt arm order drifted")
        if tuple(key for key, _ in self.real_gain_margin_over_controls) != (
            NEGATIVE_CONTROL_ARM_IDS
        ):
            raise ValueError("predictive control margin inventory drifted")
        if (
            type(self.bootstrap_resamples) is not int
            or self.bootstrap_resamples != BOOTSTRAP_RESAMPLES
            or type(self.prediction_mac_estimate) is not int
            or self.prediction_mac_estimate <= 0
            or not math.isfinite(self.paired_family_bootstrap_lower_bound)
            or type(self.passed) is not bool
            or not _is_sha256(self.receipt_sha256)
        ):
            raise ValueError("predictive receipt numerics are invalid")
        if self.receipt_sha256 != _predictive_receipt_sha256(
            self,
            include_digest=False,
        ):
            raise ValueError("predictive receipt SHA-256 does not close")

    @property
    def metrics_by_arm(self) -> dict[str, Mapping[str, object]]:
        return dict(self.arm_metrics)

    @property
    def margins(self) -> dict[str, float]:
        return dict(self.real_gain_margin_over_controls)

    def metadata(self, *, include_digest: bool = True) -> dict[str, object]:
        result = {
            "schema": (
                "fisher_graph.gemma3_exact_edge_sidecar.predictive_receipt.v1"
            ),
            "parent_edge_catalog_sha256": self.parent_edge_catalog_sha256,
            "B_row_identity_sha256": self.B_row_identity_sha256,
            "arm_order": list(ARM_ORDER),
            "arm_metrics": {
                key: copy.deepcopy(dict(value))
                for key, value in self.arm_metrics
            },
            "per_target_metrics": [
                copy.deepcopy(dict(value)) for value in self.per_target_metrics
            ],
            "real_gain_margin_over_controls": dict(
                self.real_gain_margin_over_controls
            ),
            "paired_family_bootstrap_lower_bound": (
                self.paired_family_bootstrap_lower_bound
            ),
            "bootstrap_seed": DETERMINISTIC_SEED,
            "bootstrap_resamples": self.bootstrap_resamples,
            "prediction_mac_estimate": self.prediction_mac_estimate,
            "minimum_real_gain_vs_local": 0.02,
            "minimum_real_gain_margin_over_each_negative_control": 0.01,
            "edge_catalog_pruned_or_refitted_on_B": False,
            "heldout_refit_performed": False,
            "passed": self.passed,
        }
        if include_digest:
            result["receipt_sha256"] = self.receipt_sha256
        return result


def _predictive_receipt_sha256(
    receipt: EdgeSidecarPredictiveReceipt,
    *,
    include_digest: bool,
) -> str:
    return _json_sha256(
        receipt.metadata(include_digest=include_digest),
        domain=_ARTIFACT_HASH_DOMAIN,
    )


def _predictive_receipt_sha256_from_parts(
    *,
    parent_edge_catalog_sha256: str,
    B_row_identity_sha256: str,
    arm_metrics: Sequence[tuple[str, Mapping[str, object]]],
    per_target_metrics: Sequence[Mapping[str, object]],
    margins: Sequence[tuple[str, float]],
    bootstrap_lower_bound: float,
    prediction_mac_estimate: int,
    passed: bool,
) -> str:
    payload = {
        "schema": "fisher_graph.gemma3_exact_edge_sidecar.predictive_receipt.v1",
        "parent_edge_catalog_sha256": parent_edge_catalog_sha256,
        "B_row_identity_sha256": B_row_identity_sha256,
        "arm_order": list(ARM_ORDER),
        "arm_metrics": {key: dict(value) for key, value in arm_metrics},
        "per_target_metrics": [dict(value) for value in per_target_metrics],
        "real_gain_margin_over_controls": dict(margins),
        "paired_family_bootstrap_lower_bound": bootstrap_lower_bound,
        "bootstrap_seed": DETERMINISTIC_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "prediction_mac_estimate": prediction_mac_estimate,
        "minimum_real_gain_vs_local": 0.02,
        "minimum_real_gain_margin_over_each_negative_control": 0.01,
        "edge_catalog_pruned_or_refitted_on_B": False,
        "heldout_refit_performed": False,
        "passed": passed,
    }
    return _json_sha256(payload, domain=_ARTIFACT_HASH_DOMAIN)


def _heldout_packet_for_artifact(
    artifact: FrozenDenseEdgeArtifact,
    packets: Mapping[str, UnifiedEventChartPacket],
) -> UnifiedEventChartPacket:
    claimed = packets[artifact.claimed_source_site_id]
    donor = packets[artifact.packet_source_site_id]
    if donor.site_id == claimed.site_id:
        return claimed
    return _packet_with_coordinates(claimed, donor.coordinates)


def _paired_family_bootstrap_lower_bound(
    *,
    local_by_family: Mapping[str, float],
    real_by_family: Mapping[str, float],
) -> float:
    families = tuple(sorted(local_by_family))
    if len(families) != 6 or set(real_by_family) != set(families):
        raise ValueError("predictive bootstrap requires all six heldout families")
    generator = torch.Generator().manual_seed(DETERMINISTIC_SEED)
    samples = torch.randint(
        0,
        len(families),
        (BOOTSTRAP_RESAMPLES, len(families)),
        generator=generator,
    )
    local = torch.tensor(
        [float(local_by_family[item]) for item in families],
        dtype=torch.float64,
    )
    real = torch.tensor(
        [float(real_by_family[item]) for item in families],
        dtype=torch.float64,
    )
    sampled_local = local[samples].sum(dim=1).clamp_min(1.0e-30)
    sampled_real = real[samples].sum(dim=1)
    gains = (sampled_local - sampled_real) / sampled_local
    return float(torch.quantile(gains, 0.025).item())


def evaluate_exact_edge_sidecar_predictive(
    *,
    catalog: FittedEdgeSidecarCatalog,
    heldout_capture: EdgeSidecarRoleCapture,
) -> EdgeSidecarPredictiveReceipt:
    """Evaluate frozen A-fit artifacts on B without selection or refitting."""

    if not isinstance(catalog, FittedEdgeSidecarCatalog):
        raise TypeError("catalog must be a FittedEdgeSidecarCatalog")
    catalog.assert_integrity()
    if not isinstance(heldout_capture, EdgeSidecarRoleCapture):
        raise TypeError("heldout_capture must be an EdgeSidecarRoleCapture")
    if heldout_capture.role != "B_predictive":
        raise ValueError("predictive evaluation may only use B_predictive")
    capture = heldout_capture.capture
    if capture.hidden_width != PINNED_HIDDEN_WIDTH:
        raise ValueError("B capture is not full width")
    packets = _all_source_packets(capture)
    local_fits = {item.event_id: item for item in catalog.local_baselines}
    targets: dict[str, UnifiedEventLocalBaseline] = {}
    all_local_sse = 0.0
    all_zero_sse = 0.0
    for event_id in TARGET_EVENT_IDS:
        weights = heldout_capture.fisher_row_weights(f"{event_id}.output")
        evaluation = evaluate_fitted_unified_event_local_baseline(
            local_fits[event_id],
            capture=capture,
            row_fisher_weights=weights,
            require_disjoint_from_fit=True,
        )
        all_local_sse += evaluation.local_baseline_weighted_sse
        all_zero_sse += evaluation.zero_flow_weighted_sse
        targets[event_id] = apply_fitted_unified_event_local_baseline(
            local_fits[event_id],
            capture=capture,
            row_fisher_weights=weights,
        )
    real_arm = catalog.arm("real_forward_cross_block_edges")
    selected_targets = tuple(
        artifact.edge.spec.target_event_id for artifact in real_arm.edges
    )
    local_selected_sse = 0.0
    local_by_target: dict[str, float] = {}
    row_local_by_target: dict[str, Tensor] = {}
    for event_id in selected_targets:
        target = targets[event_id]
        weights = target.row_fisher_weights.double()
        weights = weights / weights.mean().clamp_min(1.0e-12)
        row_sse = weights * target.residual_flow.double().square().sum(dim=1)
        value = float(row_sse.sum())
        row_local_by_target[event_id] = row_sse
        local_by_target[event_id] = value
        local_selected_sse += value
    if local_selected_sse <= 0.0:
        raise RuntimeError("B local residual leaves no predictive task")

    arm_metrics: list[tuple[str, Mapping[str, object]]] = [
        (
            "local_target_baseline",
            {
                "target_count": 36,
                "selected_edge_target_count": len(selected_targets),
                "all_target_zero_flow_weighted_sse": all_zero_sse,
                "all_target_local_weighted_sse": all_local_sse,
                "selected_target_local_weighted_sse": local_selected_sse,
                "relative_SSE_gain_vs_local": 0.0,
                "residual_nrmse_vs_local": 1.0,
            },
        )
    ]
    per_target_by_arm: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    real_by_family = {family: 0.0 for family in set(heldout_capture.families.values())}
    local_by_family = {family: 0.0 for family in real_by_family}
    prediction_macs = (
        len(TARGET_EVENT_IDS)
        * len(capture.row_keys)
        * PINNED_HIDDEN_WIDTH
        * PINNED_HIDDEN_WIDTH
    )
    for arm in catalog.arms:
        total_sse = 0.0
        for artifact in arm.edges:
            event_id = artifact.edge.spec.target_event_id
            target = targets[event_id]
            residual = target.residual_flow.double()
            weights = target.row_fisher_weights.double()
            weights = weights / weights.mean().clamp_min(1.0e-12)
            if arm.hard_zero:
                prediction = torch.zeros_like(residual)
            else:
                packet = _heldout_packet_for_artifact(artifact, packets)
                prediction = artifact.edge.predict(
                    packet,
                    target.target_memberships,
                ).double()
                prediction_macs += (
                    len(capture.row_keys)
                    * artifact.edge.source_rank
                    * artifact.edge.target_flow_width
                )
            row_sse = weights * (residual - prediction).square().sum(dim=1)
            edge_sse = float(row_sse.sum())
            local_sse = local_by_target[event_id]
            gain = (local_sse - edge_sse) / max(local_sse, 1.0e-30)
            total_sse += edge_sse
            per_target_by_arm[event_id][arm.arm_id] = {
                "weighted_sse": edge_sse,
                "relative_SSE_gain_vs_local": gain,
                "residual_nrmse_vs_local": math.sqrt(
                    edge_sse / max(local_sse, 1.0e-30)
                ),
            }
            if arm.arm_id == "real_forward_cross_block_edges":
                for row_index, (example_id, _) in enumerate(capture.row_keys):
                    family = heldout_capture.families[example_id]
                    real_by_family[family] += float(row_sse[row_index])
                    local_by_family[family] += float(
                        row_local_by_target[event_id][row_index]
                    )
        gain = (local_selected_sse - total_sse) / local_selected_sse
        arm_metrics.append(
            (
                arm.arm_id,
                {
                    "edge_count": len(arm.edges),
                    "weighted_sse": total_sse,
                    "relative_SSE_gain_vs_local": gain,
                    "residual_nrmse_vs_local": math.sqrt(
                        total_sse / local_selected_sse
                    ),
                    "hard_zero": arm.hard_zero,
                    "fit_rows_disjoint": True,
                    "fit_execution_disjoint": True,
                    "heldout_refit_performed": False,
                },
            )
        )
    metrics_lookup = dict(arm_metrics)
    real_gain = float(
        metrics_lookup["real_forward_cross_block_edges"][
            "relative_SSE_gain_vs_local"
        ]
    )
    margins = tuple(
        (
            arm_id,
            real_gain
            - float(metrics_lookup[arm_id]["relative_SSE_gain_vs_local"]),
        )
        for arm_id in NEGATIVE_CONTROL_ARM_IDS
    )
    per_target = tuple(
        {
            "target_event_id": event_id,
            "local_weighted_sse": local_by_target[event_id],
            "arms": per_target_by_arm[event_id],
            "real_gain_positive": (
                per_target_by_arm[event_id][
                    "real_forward_cross_block_edges"
                ]["relative_SSE_gain_vs_local"]
                > 0.0
            ),
        }
        for event_id in selected_targets
    )
    bootstrap_lower = _paired_family_bootstrap_lower_bound(
        local_by_family=local_by_family,
        real_by_family=real_by_family,
    )
    passed = bool(
        real_gain >= 0.02
        and all(value >= 0.01 for _, value in margins)
        and bootstrap_lower > 0.0
        and all(item["real_gain_positive"] is True for item in per_target)
    )
    digest = _predictive_receipt_sha256_from_parts(
        parent_edge_catalog_sha256=catalog.artifact_sha256,
        B_row_identity_sha256=capture.row_identity_sha256,
        arm_metrics=arm_metrics,
        per_target_metrics=per_target,
        margins=margins,
        bootstrap_lower_bound=bootstrap_lower,
        prediction_mac_estimate=prediction_macs,
        passed=passed,
    )
    catalog.assert_integrity()
    return EdgeSidecarPredictiveReceipt(
        parent_edge_catalog_sha256=catalog.artifact_sha256,
        B_row_identity_sha256=capture.row_identity_sha256,
        arm_metrics=tuple(arm_metrics),
        per_target_metrics=per_target,
        real_gain_margin_over_controls=margins,
        paired_family_bootstrap_lower_bound=bootstrap_lower,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        prediction_mac_estimate=prediction_macs,
        passed=passed,
        receipt_sha256=digest,
    )


@dataclass(frozen=True, slots=True)
class EdgeSidecarCausalRequest:
    protocol_sha256: str
    source_fingerprint: str
    executor: Gemma3JointCrossBlockChartExecutor
    catalog: FittedEdgeSidecarCatalog
    predictive_receipt: EdgeSidecarPredictiveReceipt
    C_capture: EdgeSidecarRoleCapture
    batches: tuple[EdgeSidecarPreparedBatch, ...]

    def __post_init__(self) -> None:
        if self.protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("causal request protocol binding drifted")
        if not _is_sha256(self.source_fingerprint):
            raise ValueError("causal request source fingerprint is invalid")
        if (
            not isinstance(self.executor, Gemma3JointCrossBlockChartExecutor)
            or self.executor.arm_id != JOINT_EXACT_CONTROL_ARM_ID
            or len(self.executor.transports)
        ):
            raise ValueError("causal request requires the transport-free exact arm")
        self.catalog.assert_integrity()
        if (
            self.predictive_receipt.parent_edge_catalog_sha256
            != self.catalog.artifact_sha256
            or self.C_capture.role != "C_intervention"
            or self.C_capture.capture.source_fingerprint != self.source_fingerprint
            or not self.batches
        ):
            raise ValueError("causal request artifact lineage is invalid")
        rows = tuple(key for batch in self.batches for key in batch.row_keys)
        if rows != self.C_capture.capture.row_keys:
            raise ValueError("causal request batches differ from the frozen C capture")

    def metadata(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "source_fingerprint": self.source_fingerprint,
            "edge_catalog_sha256": self.catalog.artifact_sha256,
            "predictive_receipt_sha256": self.predictive_receipt.receipt_sha256,
            "C_row_identity_sha256": self.C_capture.capture.row_identity_sha256,
            "C_batch_count": len(self.batches),
            "real_edge_count": len(
                self.catalog.arm("real_forward_cross_block_edges").edges
            ),
            "direction_count_per_edge_per_batch": (
                JVP_DIRECTION_COUNT_PER_EDGE
            ),
            "direction_normalization": (
                DIRECTION_IMPLEMENTATION_RECEIPT
            ),
            "protocol_direction_spec": DIRECTION_PROTOCOL_LITERAL,
            "finite_displacement_scale": FINITE_DISPLACEMENT_SCALE_LITERAL,
            "epsilon_times_source_RMS": list(INTERVENTION_EPSILON_LADDER),
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "model_mutation_authorized": False,
            "compression": False,
        }


@dataclass(frozen=True, slots=True)
class EdgeSidecarCausalEdgeMetric:
    source_site_id: str
    target_event_id: str
    C_batch_count: int
    minimum_real_jvp_cosine: float
    maximum_real_jvp_relative_norm_error: float
    minimum_real_finite_cosine: float
    maximum_real_finite_relative_norm_error: float
    maximum_adjacent_epsilon_derivative_disagreement: float
    control_minimum_jvp_cosines: tuple[tuple[str, float], ...]
    control_minimum_finite_cosines: tuple[tuple[str, float], ...]
    exact_forward_count: int
    exact_jvp_count: int
    backward_or_jvp_count: int
    finite_intervention_forward_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_site_id, str)
            or not self.source_site_id
            or not isinstance(self.target_event_id, str)
            or not self.target_event_id.startswith("mutation.")
            or type(self.C_batch_count) is not int
            or self.C_batch_count <= 0
        ):
            raise ValueError("causal edge endpoint/batch identity is invalid")
        cosine_values = (
            self.minimum_real_jvp_cosine,
            self.minimum_real_finite_cosine,
            *(value for _, value in self.control_minimum_jvp_cosines),
            *(value for _, value in self.control_minimum_finite_cosines),
        )
        error_values = (
            self.maximum_real_jvp_relative_norm_error,
            self.maximum_real_finite_relative_norm_error,
            self.maximum_adjacent_epsilon_derivative_disagreement,
        )
        if any(
            not math.isfinite(value) or value < -1.0 or value > 1.0
            for value in cosine_values
        ) or any(
            not math.isfinite(value) or value < 0.0 for value in error_values
        ):
            raise ValueError("causal edge metrics are invalid")
        if tuple(key for key, _ in self.control_minimum_jvp_cosines) != (
            NEGATIVE_CONTROL_ARM_IDS
        ) or tuple(key for key, _ in self.control_minimum_finite_cosines) != (
            NEGATIVE_CONTROL_ARM_IDS
        ):
            raise ValueError("causal edge negative-control inventory drifted")
        if (
            dict(self.control_minimum_jvp_cosines)["zero_edge_control"] != 0.0
            or dict(self.control_minimum_finite_cosines)["zero_edge_control"]
            != 0.0
        ):
            raise ValueError("zero-edge causal cosine must be exactly zero")
        expected_jvp = (
            4 * JVP_DIRECTION_COUNT_PER_EDGE * self.C_batch_count
        )
        expected_finite = (
            4
            * 2
            * JVP_DIRECTION_COUNT_PER_EDGE
            * len(INTERVENTION_EPSILON_LADDER)
            * self.C_batch_count
        )
        expected_forward = (
            4
            * (
                2
                + JVP_DIRECTION_COUNT_PER_EDGE
                + 2
                * JVP_DIRECTION_COUNT_PER_EDGE
                * len(INTERVENTION_EPSILON_LADDER)
            )
            * self.C_batch_count
        )
        if (
            self.exact_jvp_count != expected_jvp
            or self.backward_or_jvp_count != expected_jvp
            or self.finite_intervention_forward_count != expected_finite
            or self.exact_forward_count != expected_forward
        ):
            raise ValueError("causal edge execution-count ledger does not close")

    @property
    def jvp_margins(self) -> dict[str, float]:
        return {
            arm_id: self.minimum_real_jvp_cosine - value
            for arm_id, value in self.control_minimum_jvp_cosines
        }

    @property
    def finite_margins(self) -> dict[str, float]:
        return {
            arm_id: self.minimum_real_finite_cosine - value
            for arm_id, value in self.control_minimum_finite_cosines
        }

    @property
    def jvp_passed(self) -> bool:
        return (
            self.minimum_real_jvp_cosine >= 0.90
            and self.maximum_real_jvp_relative_norm_error <= 0.25
            and all(value >= 0.05 for value in self.jvp_margins.values())
        )

    @property
    def finite_passed(self) -> bool:
        return (
            self.minimum_real_finite_cosine >= 0.80
            and self.maximum_real_finite_relative_norm_error <= 0.35
            and self.maximum_adjacent_epsilon_derivative_disagreement <= 0.10
            and all(value >= 0.05 for value in self.finite_margins.values())
        )

    @property
    def passed(self) -> bool:
        return self.jvp_passed and self.finite_passed

    def metadata(self) -> dict[str, object]:
        return {
            "source_site_id": self.source_site_id,
            "target_event_id": self.target_event_id,
            "C_batch_count": self.C_batch_count,
            "minimum_real_jvp_cosine": self.minimum_real_jvp_cosine,
            "maximum_real_jvp_relative_norm_error": (
                self.maximum_real_jvp_relative_norm_error
            ),
            "minimum_real_finite_cosine": self.minimum_real_finite_cosine,
            "maximum_real_finite_relative_norm_error": (
                self.maximum_real_finite_relative_norm_error
            ),
            "maximum_adjacent_epsilon_derivative_disagreement": (
                self.maximum_adjacent_epsilon_derivative_disagreement
            ),
            "control_minimum_jvp_cosines": dict(
                self.control_minimum_jvp_cosines
            ),
            "control_minimum_finite_cosines": dict(
                self.control_minimum_finite_cosines
            ),
            "jvp_cosine_margins_over_controls": self.jvp_margins,
            "finite_cosine_margins_over_controls": self.finite_margins,
            "minimum_required_control_margin": 0.05,
            "exact_forward_count": self.exact_forward_count,
            "exact_jvp_count": self.exact_jvp_count,
            "backward_or_jvp_count": self.backward_or_jvp_count,
            "finite_intervention_forward_count": (
                self.finite_intervention_forward_count
            ),
            "jvp_passed": self.jvp_passed,
            "finite_intervention_passed": self.finite_passed,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class EdgeSidecarCausalReceipt:
    protocol_sha256: str
    parent_edge_catalog_sha256: str
    parent_predictive_receipt_sha256: str
    C_row_identity_sha256: str
    source_fingerprint_before: str
    source_fingerprint_after: str
    edge_metrics: tuple[EdgeSidecarCausalEdgeMetric, ...]
    peak_resident_bytes: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        hashes = (
            self.protocol_sha256,
            self.parent_edge_catalog_sha256,
            self.parent_predictive_receipt_sha256,
            self.C_row_identity_sha256,
            self.source_fingerprint_before,
            self.source_fingerprint_after,
            self.receipt_sha256,
        )
        if any(not _is_sha256(value) for value in hashes):
            raise ValueError("causal receipt artifact hash is invalid")
        if self.protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("causal receipt protocol binding drifted")
        if self.source_fingerprint_before != self.source_fingerprint_after:
            raise ValueError("causal worker changed the exact source fingerprint")
        if not self.edge_metrics or len(
            {(item.source_site_id, item.target_event_id) for item in self.edge_metrics}
        ) != len(self.edge_metrics):
            raise ValueError("causal receipt edge inventory is invalid")
        if type(self.peak_resident_bytes) is not int or self.peak_resident_bytes <= 0:
            raise ValueError("causal receipt peak memory is invalid")
        if self.receipt_sha256 != _causal_receipt_sha256(
            self,
            include_digest=False,
        ):
            raise ValueError("causal receipt SHA-256 does not close")

    @property
    def exact_forward_count(self) -> int:
        return sum(item.exact_forward_count for item in self.edge_metrics)

    @property
    def backward_or_jvp_count(self) -> int:
        return sum(item.backward_or_jvp_count for item in self.edge_metrics)

    @property
    def finite_intervention_forward_count(self) -> int:
        return sum(
            item.finite_intervention_forward_count for item in self.edge_metrics
        )

    @property
    def passed(self) -> bool:
        return bool(self.edge_metrics) and all(item.passed for item in self.edge_metrics)

    def metadata(self, *, include_digest: bool = True) -> dict[str, object]:
        result = {
            "schema": "fisher_graph.gemma3_exact_edge_sidecar.causal_receipt.v1",
            "protocol_sha256": self.protocol_sha256,
            "parent_edge_catalog_sha256": self.parent_edge_catalog_sha256,
            "parent_predictive_receipt_sha256": (
                self.parent_predictive_receipt_sha256
            ),
            "C_row_identity_sha256": self.C_row_identity_sha256,
            "source_fingerprint_before": self.source_fingerprint_before,
            "source_fingerprint_after": self.source_fingerprint_after,
            "source_state_fingerprint_restored_after_each_probe": True,
            "direction_count_per_edge_per_batch": JVP_DIRECTION_COUNT_PER_EDGE,
            "direction_normalization": (
                DIRECTION_IMPLEMENTATION_RECEIPT
            ),
            "protocol_direction_spec": DIRECTION_PROTOCOL_LITERAL,
            "finite_displacement_scale": FINITE_DISPLACEMENT_SCALE_LITERAL,
            "epsilon_times_source_RMS": list(INTERVENTION_EPSILON_LADDER),
            "edge_metrics": [item.metadata() for item in self.edge_metrics],
            "edge_count": len(self.edge_metrics),
            "exact_forward_count": self.exact_forward_count,
            "backward_or_JVP_count": self.backward_or_jvp_count,
            "finite_intervention_forward_count": (
                self.finite_intervention_forward_count
            ),
            "peak_resident_bytes": self.peak_resident_bytes,
            "JVP_gate_passed": all(item.jvp_passed for item in self.edge_metrics),
            "finite_intervention_gate_passed": all(
                item.finite_passed for item in self.edge_metrics
            ),
            "negative_control_margin_evaluated_per_edge": True,
            "baseline_model_output_replaced": False,
            "model_weights_mutated": False,
            "shadow_execution_only": True,
            "passed": self.passed,
        }
        if include_digest:
            result["receipt_sha256"] = self.receipt_sha256
        return result


def _causal_receipt_sha256(
    receipt: EdgeSidecarCausalReceipt,
    *,
    include_digest: bool,
) -> str:
    payload = receipt.metadata(include_digest=include_digest)
    return _json_sha256(payload, domain=_ARTIFACT_HASH_DOMAIN)


def build_edge_sidecar_causal_receipt(
    *,
    request: EdgeSidecarCausalRequest,
    edge_metrics: Sequence[EdgeSidecarCausalEdgeMetric],
    source_fingerprint_after: str,
    peak_resident_bytes: int,
) -> EdgeSidecarCausalReceipt:
    if not isinstance(request, EdgeSidecarCausalRequest):
        raise TypeError("request must be an EdgeSidecarCausalRequest")
    request.catalog.assert_integrity()
    metrics = tuple(edge_metrics)
    real = request.catalog.arm("real_forward_cross_block_edges")
    expected = tuple(
        (item.claimed_source_site_id, item.edge.spec.target_event_id)
        for item in real.edges
    )
    if tuple((item.source_site_id, item.target_event_id) for item in metrics) != expected:
        raise ValueError("causal metric inventory differs from the real edge catalog")
    common = {
        "schema": "fisher_graph.gemma3_exact_edge_sidecar.causal_receipt.v1",
        "protocol_sha256": request.protocol_sha256,
        "parent_edge_catalog_sha256": request.catalog.artifact_sha256,
        "parent_predictive_receipt_sha256": (
            request.predictive_receipt.receipt_sha256
        ),
        "C_row_identity_sha256": request.C_capture.capture.row_identity_sha256,
        "source_fingerprint_before": request.source_fingerprint,
        "source_fingerprint_after": source_fingerprint_after,
        "source_state_fingerprint_restored_after_each_probe": True,
        "direction_count_per_edge_per_batch": JVP_DIRECTION_COUNT_PER_EDGE,
        "direction_normalization": (
            DIRECTION_IMPLEMENTATION_RECEIPT
        ),
        "protocol_direction_spec": DIRECTION_PROTOCOL_LITERAL,
        "finite_displacement_scale": FINITE_DISPLACEMENT_SCALE_LITERAL,
        "epsilon_times_source_RMS": list(INTERVENTION_EPSILON_LADDER),
        "edge_metrics": [item.metadata() for item in metrics],
        "edge_count": len(metrics),
        "exact_forward_count": sum(item.exact_forward_count for item in metrics),
        "backward_or_JVP_count": sum(
            item.backward_or_jvp_count for item in metrics
        ),
        "finite_intervention_forward_count": sum(
            item.finite_intervention_forward_count for item in metrics
        ),
        "peak_resident_bytes": peak_resident_bytes,
        "JVP_gate_passed": all(item.jvp_passed for item in metrics),
        "finite_intervention_gate_passed": all(
            item.finite_passed for item in metrics
        ),
        "negative_control_margin_evaluated_per_edge": True,
        "baseline_model_output_replaced": False,
        "model_weights_mutated": False,
        "shadow_execution_only": True,
        "passed": bool(metrics) and all(item.passed for item in metrics),
    }
    digest = _json_sha256(common, domain=_ARTIFACT_HASH_DOMAIN)
    request.catalog.assert_integrity()
    return EdgeSidecarCausalReceipt(
        protocol_sha256=request.protocol_sha256,
        parent_edge_catalog_sha256=request.catalog.artifact_sha256,
        parent_predictive_receipt_sha256=(
            request.predictive_receipt.receipt_sha256
        ),
        C_row_identity_sha256=request.C_capture.capture.row_identity_sha256,
        source_fingerprint_before=request.source_fingerprint,
        source_fingerprint_after=source_fingerprint_after,
        edge_metrics=metrics,
        peak_resident_bytes=peak_resident_bytes,
        receipt_sha256=digest,
    )


CausalEvaluator = Callable[[EdgeSidecarCausalRequest], EdgeSidecarCausalReceipt]


def evaluate_exact_edge_sidecar_causal(
    request: EdgeSidecarCausalRequest,
) -> EdgeSidecarCausalReceipt:
    """Run the audited built-in C evaluator across every frozen real edge.

    The zero arm has an analytic zero prediction and requires no shadow call.
    Real, row-shuffled, family-shuffled, and provenance-permuted coefficients
    each receive the same deterministic directions and C batches.
    """

    if not isinstance(request, EdgeSidecarCausalRequest):
        raise TypeError("request must be an EdgeSidecarCausalRequest")
    request.catalog.assert_integrity()
    executor_guard = _FrozenModelTensorGuard(request.executor)
    local = {item.event_id: item for item in request.catalog.local_baselines}
    arms = {
        arm_id: request.catalog.arm(arm_id)
        for arm_id in (
            "real_forward_cross_block_edges",
            "row_shuffled_control",
            "family_shuffled_control",
            "source_provenance_permutation_control",
        )
    }
    metrics: list[EdgeSidecarCausalEdgeMetric] = []
    real_edges = arms["real_forward_cross_block_edges"].edges
    for edge_index, real_artifact in enumerate(real_edges):
        evaluations: dict[str, list[ExactEdgeCausalEvaluation]] = {
            arm_id: [] for arm_id in arms
        }
        target_id = real_artifact.edge.spec.target_event_id
        for arm_id, arm in arms.items():
            artifact = arm.edges[edge_index]
            if (
                artifact.claimed_source_site_id
                != real_artifact.claimed_source_site_id
                or artifact.edge.spec.target_event_id != target_id
            ):
                raise RuntimeError("causal control topology drifted")
            for batch in request.batches:
                evaluations[arm_id].append(
                    evaluate_exact_dense_edge_causality(
                        executor=request.executor,
                        input_ids=batch.input_ids,
                        sequence=batch.sequence,
                        edge=artifact.edge,
                        local_baseline=local[target_id],
                        direction_count=JVP_DIRECTION_COUNT_PER_EDGE,
                        epsilon_ladder=INTERVENTION_EPSILON_LADDER,
                    )
                )
        real_values = evaluations["real_forward_cross_block_edges"]
        control_jvp = (
            ("zero_edge_control", 0.0),
            *(
                (
                    arm_id,
                    min(item.minimum_jvp_cosine for item in evaluations[arm_id]),
                )
                for arm_id in (
                    "row_shuffled_control",
                    "family_shuffled_control",
                    "source_provenance_permutation_control",
                )
            ),
        )
        control_finite = (
            ("zero_edge_control", 0.0),
            *(
                (
                    arm_id,
                    min(
                        item.minimum_finite_cosine
                        for item in evaluations[arm_id]
                    ),
                )
                for arm_id in (
                    "row_shuffled_control",
                    "family_shuffled_control",
                    "source_provenance_permutation_control",
                )
            ),
        )
        all_evaluations = tuple(
            item
            for arm_id in arms
            for item in evaluations[arm_id]
        )
        metrics.append(
            EdgeSidecarCausalEdgeMetric(
                source_site_id=real_artifact.claimed_source_site_id,
                target_event_id=target_id,
                C_batch_count=len(request.batches),
                minimum_real_jvp_cosine=min(
                    item.minimum_jvp_cosine for item in real_values
                ),
                maximum_real_jvp_relative_norm_error=max(
                    item.maximum_jvp_relative_error for item in real_values
                ),
                minimum_real_finite_cosine=min(
                    item.minimum_finite_cosine for item in real_values
                ),
                maximum_real_finite_relative_norm_error=max(
                    item.maximum_finite_relative_error for item in real_values
                ),
                maximum_adjacent_epsilon_derivative_disagreement=max(
                    item.maximum_adjacent_finite_derivative_disagreement
                    for item in real_values
                ),
                control_minimum_jvp_cosines=control_jvp,
                control_minimum_finite_cosines=control_finite,
                exact_forward_count=sum(
                    item.exact_forward_count for item in all_evaluations
                ),
                exact_jvp_count=sum(
                    item.exact_jvp_count for item in all_evaluations
                ),
                backward_or_jvp_count=sum(
                    item.backward_or_jvp_count for item in all_evaluations
                ),
                finite_intervention_forward_count=sum(
                    item.finite_intervention_forward_count
                    for item in all_evaluations
                ),
            )
        )
    executor_guard.assert_unchanged()
    request.catalog.assert_integrity()
    return build_edge_sidecar_causal_receipt(
        request=request,
        edge_metrics=metrics,
        source_fingerprint_after=request.source_fingerprint,
        peak_resident_bytes=_peak_resident_bytes(),
    )


def _protocol_receipt(path: Path | str) -> tuple[dict[str, object], dict[str, object]]:
    observed = load_exact_edge_sidecar_protocol(path)
    canonical = build_exact_edge_sidecar_protocol()
    if observed != canonical:
        raise ValueError("exact edge-sidecar protocol artifact is not canonical")
    if observed.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("runner protocol SHA-256 binding is stale")
    receipt = {
        "schema": observed["schema"],
        "format_version": observed["format_version"],
        "protocol_sha256": observed["protocol_sha256"],
        "artifact_path": str(path),
        "authenticated_against_current_builder": True,
    }
    return observed, receipt


def _prepare_role_batches(
    *,
    tokenizer: object,
    adapter: Gemma3CausalLMAdapter,
    examples: Sequence[PinnedEdgeSidecarExample],
    batch_size: int,
    device: torch.device,
) -> tuple[EdgeSidecarPreparedBatch, ...]:
    prompts = tuple(item.prompt for item in examples)
    example_ids = tuple(item.example_id for item in examples)
    keyed = _tokenize_keyed_batches(
        tokenizer,
        prompts,
        example_ids,
        batch_size=batch_size,
        device=device,
    )
    result: list[EdgeSidecarPreparedBatch] = []
    for item in keyed:
        model_inputs = item.model_inputs
        input_ids = model_inputs.get("input_ids")
        valid_mask = model_inputs.get("attention_mask")
        if not isinstance(input_ids, Tensor) or not isinstance(valid_mask, Tensor):
            raise RuntimeError("tokenizer omitted input_ids or attention_mask")
        result.append(
            EdgeSidecarPreparedBatch(
                input_ids=input_ids,
                sequence=adapter.prepare_sequence(model_inputs),
                valid_mask=valid_mask.bool(),
                row_keys=tuple(item.row_keys),
            )
        )
    return tuple(result)


def _chart_catalog_receipt(
    *,
    protocol: Mapping[str, object],
    capture: EdgeSidecarRoleCapture,
) -> dict[str, object]:
    contract = protocol.get("unified_exact_chart_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("protocol chart contract is unavailable")
    sites = contract.get("sites")
    if not isinstance(sites, list):
        raise ValueError("protocol chart sites are unavailable")
    observed_ids = tuple(item.site_id for item in capture.capture.observations)
    protocol_ids = tuple(
        item.get("site_id") for item in sites if isinstance(item, Mapping)
    )
    if observed_ids != protocol_ids or len(observed_ids) != PINNED_CHART_SITE_COUNT:
        raise RuntimeError("observed chart inventory differs from the protocol")
    payload: dict[str, object] = {
        "schema": "fisher_graph.gemma3_exact_edge_sidecar.chart_catalog.v1",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "site_count": PINNED_CHART_SITE_COUNT,
        "source_eligible_site_count": PINNED_CHART_SITE_COUNT,
        "target_eligible_site_count": 36,
        "hidden_width": PINNED_HIDDEN_WIDTH,
        "coordinate_rank": PINNED_HIDDEN_WIDTH,
        "chart_count_per_site": 1,
        "identity_basis_stored_scalar_count": 0,
        "sites": copy.deepcopy(sites),
        "observed_site_ids_match_protocol": True,
        "full_width_exact_coordinates": True,
        "contains_activations": False,
        "contains_gradients": False,
        "installed_in_model": False,
        "compression": False,
    }
    payload["artifact_sha256"] = _json_sha256(
        payload,
        domain=_ARTIFACT_HASH_DOMAIN,
    )
    return payload


def _aggregate_exact_gate(
    captures: Mapping[str, EdgeSidecarRoleCapture],
) -> dict[str, object]:
    if tuple(captures) != ROLE_ORDER:
        raise ValueError("exact gate capture role order drifted")
    gates = {role: captures[role].exact_gate for role in ROLE_ORDER}
    required_boolean_fields = (
        "passed",
        "event_order_equal",
        "every_mutation_chart_input_byte_identical",
        "every_mutation_delta_byte_identical",
        "final_carrier_byte_identical",
        "normalized_output_byte_identical",
        "logits_byte_identical",
        "source_state_fingerprint_unchanged",
    )
    passed = all(
        gate.get(field) is True
        for gate in gates.values()
        for field in required_boolean_fields
    ) and all(
        isinstance(gate.get("native_exact_control"), Mapping)
        and gate["native_exact_control"].get("run") is True  # type: ignore[index]
        and gate["native_exact_control"].get(  # type: ignore[index]
            "literal_valid_query_logit_parity"
        )
        is True
        for gate in gates.values()
    )
    return {
        "passed": passed,
        "event_count": PINNED_EVENT_COUNT,
        "mutation_count": 36,
        "attention_mutation_count": 18,
        "mlp_mutation_count": 18,
        "residual_carrier_initial_version": 0,
        "residual_carrier_final_version": 36,
        "role_gates": {
            role: copy.deepcopy(dict(gates[role])) for role in ROLE_ORDER
        },
        "every_mutation_delta_hash_compared_disabled_vs_enabled": True,
        "tied_lexical_ingress_and_egress_exact": True,
        "all_model_weights_exact": True,
        "sidecar_wrote_model_state": False,
        "sidecar_wrote_model_logits": False,
    }


def _qualification(
    *,
    exact_gate_passed: bool,
    predictive_passed: bool,
    causal_receipt: EdgeSidecarCausalReceipt | None,
) -> dict[str, object]:
    jvp = causal_receipt is not None and all(
        item.jvp_passed for item in causal_receipt.edge_metrics
    )
    finite = causal_receipt is not None and all(
        item.finite_passed for item in causal_receipt.edge_metrics
    )
    return _qualification_from_flags(
        exact_gate_passed=exact_gate_passed,
        predictive_passed=predictive_passed,
        jvp_passed=jvp,
        finite_passed=finite,
    )


def _qualification_from_flags(
    *,
    exact_gate_passed: bool,
    predictive_passed: bool,
    jvp_passed: bool,
    finite_passed: bool,
) -> dict[str, object]:
    edge_axis = bool(
        exact_gate_passed
        and predictive_passed
        and jvp_passed
        and finite_passed
    )
    reasons: list[str] = []
    if not exact_gate_passed:
        reasons.append("exact_39_event_gate_failed")
    if not predictive_passed:
        reasons.append("heldout_B_predictive_gate_failed")
    if not jvp_passed:
        reasons.append("heldout_C_JVP_gate_failed_or_not_run")
    if not finite_passed:
        reasons.append("heldout_C_finite_intervention_gate_failed_or_not_run")
    return {
        "exact_39_event_gate_passed": exact_gate_passed,
        "heldout_B_predictive_gate_passed": predictive_passed,
        "heldout_C_JVP_gate_passed": jvp_passed,
        "heldout_C_finite_intervention_gate_passed": finite_passed,
        "edge_axis_passed": edge_axis,
        "passed": edge_axis,
        "qualified_as_edge_causality_result": edge_axis,
        "combined_edge_and_compression_arm_unlocked": False,
        "compression_claim_authorized": False,
        "compression_success_claimed": False,
        "model_mutation_authorized": False,
        "failure_reasons": reasons,
    }


def _resource_ledger(
    *,
    captures: Mapping[str, EdgeSidecarRoleCapture],
    catalog: FittedEdgeSidecarCatalog,
    predictive: EdgeSidecarPredictiveReceipt,
    causal: EdgeSidecarCausalReceipt | None,
    peak_resident_bytes: int,
) -> dict[str, object]:
    capture_exact = sum(item.exact_forward_count for item in captures.values())
    capture_native = sum(item.native_forward_count for item in captures.values())
    capture_backward = sum(item.backward_count for item in captures.values())
    causal_exact = 0 if causal is None else causal.exact_forward_count
    causal_backward = 0 if causal is None else causal.backward_or_jvp_count
    finite = 0 if causal is None else causal.finite_intervention_forward_count
    return {
        "source_model_parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
        "model_parameter_reduction_fraction": 0.0,
        "model_storage_reduction_fraction": 0.0,
        "model_MAC_reduction_fraction": 0.0,
        "model_parameter_count_after_experiment": PINNED_SOURCE_PARAMETER_COUNT,
        "implicit_identity_chart_scalar_count": 0,
        "sidecar_local_baseline_scalar_count": (
            catalog.local_baseline_scalar_count
        ),
        "sidecar_edge_scalar_count": catalog.sidecar_edge_scalar_count,
        "sidecar_cost_subtracted_from_model_cost": False,
        "fit_design_MAC_estimate": catalog.fit_design_mac_estimate,
        "fit_dense_solve_FLOP_lower_bound": (
            catalog.fit_dense_solve_flop_lower_bound
        ),
        "rank_decomposition_FLOP_lower_bound": (
            catalog.rank_decomposition_flop_lower_bound
        ),
        "heldout_prediction_MAC_estimate": predictive.prediction_mac_estimate,
        "capture_exact_forward_count": capture_exact,
        "capture_native_forward_count": capture_native,
        "causal_exact_forward_count": causal_exact,
        "exact_forward_count": capture_exact + causal_exact,
        "capture_backward_count": capture_backward,
        "causal_backward_or_JVP_count": causal_backward,
        "backward_or_JVP_count": capture_backward + causal_backward,
        "finite_intervention_forward_count": finite,
        "peak_resident_bytes": max(
            peak_resident_bytes,
            0 if causal is None else causal.peak_resident_bytes,
        ),
        "fit_cost_is_analysis_only": True,
        "sidecar_inference_cost_claimed": False,
        "latency_improvement_claimed": False,
        "compression": False,
    }


def _assemble_report(
    *,
    protocol_receipt: Mapping[str, object],
    protocol: Mapping[str, object],
    split: PinnedEdgeSidecarSplit,
    model_metadata: Mapping[str, object],
    compilation: Gemma3JointGraphCompilation,
    captures: Mapping[str, EdgeSidecarRoleCapture],
    catalog: FittedEdgeSidecarCatalog,
    predictive: EdgeSidecarPredictiveReceipt,
    causal: EdgeSidecarCausalReceipt | None,
    source_fingerprint_before: str,
    source_fingerprint_after: str,
    batch_size: int,
    peak_resident_bytes: int,
) -> dict[str, object]:
    exact_gate = _aggregate_exact_gate(captures)
    chart_catalog = _chart_catalog_receipt(
        protocol=protocol,
        capture=captures["A_fit"],
    )
    qualification = _qualification(
        exact_gate_passed=exact_gate["passed"] is True,
        predictive_passed=predictive.passed,
        causal_receipt=causal,
    )
    resources = _resource_ledger(
        captures=captures,
        catalog=catalog,
        predictive=predictive,
        causal=causal,
        peak_resident_bytes=peak_resident_bytes,
    )
    report: dict[str, object] = {
        "schema": EXACT_EDGE_SIDECAR_EXPERIMENT_SCHEMA,
        "format_version": EXACT_EDGE_SIDECAR_EXPERIMENT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_sidecar_coefficients": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "protocol": copy.deepcopy(dict(protocol_receipt)),
        "scope_firewall": {
            "axis": "edge_structure_only",
            "compression": False,
            "compressed_candidate_present": False,
            "candidate_model_produced": False,
            "parameter_removal": False,
            "parameter_merging": False,
            "rank_reduction": False,
            "weight_approximation": False,
            "all_model_weights_exact": True,
            "all_model_ranks_exact": True,
            "model_output_mutated_by_sidecar": False,
            "model_carrier_mutated_by_sidecar": False,
            "edge_predictions_installed_in_model": False,
            "compression_success_claimed": False,
            "latency_improvement_claimed": False,
        },
        "model": copy.deepcopy(dict(model_metadata)),
        "data": split.metadata(),
        "configuration": {
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "batch_size": batch_size,
            "deterministic_seed": DETERMINISTIC_SEED,
            "relative_ridge": RELATIVE_RIDGE,
            "coordinate_rank": PINNED_HIDDEN_WIDTH,
            "edge_coefficient_shape": [
                PINNED_HIDDEN_WIDTH,
                PINNED_HIDDEN_WIDTH,
            ],
            "edge_low_rank_factorization_used": False,
            "maximum_incoming_edges_per_target": 1,
            "JVP_direction_count_per_edge_per_batch": (
                JVP_DIRECTION_COUNT_PER_EDGE
            ),
            "direction_normalization": (
                DIRECTION_IMPLEMENTATION_RECEIPT
            ),
            "protocol_direction_spec": DIRECTION_PROTOCOL_LITERAL,
            "finite_displacement_scale": FINITE_DISPLACEMENT_SCALE_LITERAL,
            "intervention_epsilon_ladder": list(
                INTERVENTION_EPSILON_LADDER
            ),
        },
        "compilation": {
            "exact_control": compilation.metadata(),
            "exact_39_event_gate": exact_gate,
            "source_model_fingerprint_before": source_fingerprint_before,
            "source_model_fingerprint_after": source_fingerprint_after,
            "source_model_unchanged": (
                source_fingerprint_before == source_fingerprint_after
            ),
        },
        "captures": {
            role: captures[role].metadata() for role in ROLE_ORDER
        },
        "chart_catalog": chart_catalog,
        "edge_catalog": catalog.metadata(),
        "heldout_predictive_receipt": predictive.metadata(),
        "heldout_causal_receipt": (
            None if causal is None else causal.metadata()
        ),
        "analysis_cost": resources,
        "qualification": qualification,
    }
    report["report_sha256"] = _report_sha256(report)
    return report


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"exact edge-sidecar report {label} is invalid")
    return value


def _verify_nested_digest(
    value: Mapping[str, object],
    *,
    digest_key: str,
) -> None:
    supplied = value.get(digest_key)
    if not _is_sha256(supplied):
        raise ValueError(f"nested {digest_key} is invalid")
    payload = {key: item for key, item in value.items() if key != digest_key}
    if supplied != _json_sha256(payload, domain=_ARTIFACT_HASH_DOMAIN):
        raise ValueError(f"nested {digest_key} does not authenticate its payload")


def validate_exact_edge_sidecar_experiment_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on scientific, lineage, resource, and hash drift."""

    if not isinstance(report, Mapping):
        raise TypeError("exact edge-sidecar report must be a mapping")
    expected_top = {
        "schema",
        "format_version",
        "contains_model_weights",
        "contains_executor_weights",
        "contains_sidecar_coefficients",
        "contains_prompt_text",
        "contains_token_ids",
        "protocol",
        "scope_firewall",
        "model",
        "data",
        "configuration",
        "compilation",
        "captures",
        "chart_catalog",
        "edge_catalog",
        "heldout_predictive_receipt",
        "heldout_causal_receipt",
        "analysis_cost",
        "qualification",
        "report_sha256",
    }
    if set(report) != expected_top:
        raise ValueError("exact edge-sidecar report top-level schema drifted")
    if (
        report.get("schema") != EXACT_EDGE_SIDECAR_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != EXACT_EDGE_SIDECAR_EXPERIMENT_FORMAT_VERSION
        or any(
            report.get(key) is not False
            for key in (
                "contains_model_weights",
                "contains_executor_weights",
                "contains_sidecar_coefficients",
                "contains_prompt_text",
                "contains_token_ids",
            )
        )
    ):
        raise ValueError("exact edge-sidecar report identity/content flags drifted")
    protocol = _mapping(report.get("protocol"), label="protocol")
    if (
        protocol.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or protocol.get("authenticated_against_current_builder") is not True
    ):
        raise ValueError("exact edge-sidecar report protocol lineage drifted")
    scope = _mapping(report.get("scope_firewall"), label="scope firewall")
    false_scope = (
        "compression",
        "compressed_candidate_present",
        "candidate_model_produced",
        "parameter_removal",
        "parameter_merging",
        "rank_reduction",
        "weight_approximation",
        "model_output_mutated_by_sidecar",
        "model_carrier_mutated_by_sidecar",
        "edge_predictions_installed_in_model",
        "compression_success_claimed",
        "latency_improvement_claimed",
    )
    if (
        scope.get("axis") != "edge_structure_only"
        or any(scope.get(key) is not False for key in false_scope)
        or scope.get("all_model_weights_exact") is not True
        or scope.get("all_model_ranks_exact") is not True
    ):
        raise ValueError("exact edge-sidecar scope firewall drifted")
    model = _mapping(report.get("model"), label="model")
    if (
        model.get("model_id") != PINNED_MODEL_ID
        or model.get("requested_revision") != PINNED_MODEL_REVISION
    ):
        raise ValueError("exact edge-sidecar model provenance drifted")
    data = _mapping(report.get("data"), label="data")
    roles = _mapping(data.get("roles"), label="data roles")
    if (
        data.get("panel_sha256") != PINNED_PANEL_SHA256
        or data.get("family_count") != 6
        or data.get("total_example_count") != 60
        or data.get("contains_prompt_text") is not False
        or data.get("contains_token_ids") is not False
        or tuple(roles) != ROLE_ORDER
    ):
        raise ValueError("exact edge-sidecar data firewall drifted")
    for role, count in zip(ROLE_ORDER, (36, 12, 12), strict=True):
        receipt = _mapping(roles.get(role), label=f"data role {role}")
        if (
            receipt.get("example_count") != count
            or receipt.get("example_ids_sha256")
            != PINNED_SPLIT_ID_SHA256S[role]
            or not _is_sha256(receipt.get("family_assignment_sha256"))
        ):
            raise ValueError("exact edge-sidecar split identity drifted")
    configuration = _mapping(report.get("configuration"), label="configuration")
    if (
        configuration.get("device") != "cpu"
        or configuration.get("dtype") != "float32"
        or configuration.get("deterministic_seed") != DETERMINISTIC_SEED
        or configuration.get("relative_ridge") != RELATIVE_RIDGE
        or configuration.get("coordinate_rank") != PINNED_HIDDEN_WIDTH
        or configuration.get("edge_coefficient_shape") != [640, 640]
        or configuration.get("edge_low_rank_factorization_used") is not False
        or configuration.get("maximum_incoming_edges_per_target") != 1
        or configuration.get("direction_normalization")
        != DIRECTION_IMPLEMENTATION_RECEIPT
        or configuration.get("protocol_direction_spec")
        != DIRECTION_PROTOCOL_LITERAL
        or configuration.get("finite_displacement_scale")
        != FINITE_DISPLACEMENT_SCALE_LITERAL
        or configuration.get("intervention_epsilon_ladder")
        != list(INTERVENTION_EPSILON_LADDER)
    ):
        raise ValueError("exact edge-sidecar configuration drifted")
    compilation = _mapping(report.get("compilation"), label="compilation")
    exact_compilation = _mapping(
        compilation.get("exact_control"),
        label="exact compilation",
    )
    manifest = _mapping(
        exact_compilation.get("executor_manifest"),
        label="executor manifest",
    )
    if (
        exact_compilation.get("compilation_kind") != "exact_39_event_control"
        or exact_compilation.get("exact_parameter_inventory_parity") is not True
        or exact_compilation.get("source_model_parameter_count")
        != PINNED_SOURCE_PARAMETER_COUNT
        or manifest.get("exact_control_arm") is not True
        or manifest.get("event_count") != 39
        or manifest.get("attention_mutation_count") != 18
        or manifest.get("mlp_mutation_count") != 18
        or manifest.get("direct_compiled_attention_mutation_count") != 18
        or manifest.get("transport_edge_count") != 0
        or manifest.get("tied_lexical_ingress_and_egress_present") is not True
    ):
        raise ValueError("exact 39-event compilation receipt drifted")
    exact_gate = _mapping(
        compilation.get("exact_39_event_gate"),
        label="exact gate",
    )
    if (
        exact_gate.get("passed") is not True
        or exact_gate.get("event_count") != 39
        or exact_gate.get("attention_mutation_count") != 18
        or exact_gate.get("mlp_mutation_count") != 18
        or exact_gate.get("residual_carrier_final_version") != 36
        or exact_gate.get(
            "every_mutation_delta_hash_compared_disabled_vs_enabled"
        )
        is not True
        or compilation.get("source_model_unchanged") is not True
        or compilation.get("source_model_fingerprint_before")
        != compilation.get("source_model_fingerprint_after")
    ):
        raise ValueError("exact observer/native gate drifted")
    capture_receipts = _mapping(report.get("captures"), label="captures")
    if tuple(capture_receipts) != ROLE_ORDER:
        raise ValueError("capture receipt role order drifted")
    row_hashes: list[str] = []
    for role in ROLE_ORDER:
        capture = _mapping(capture_receipts[role], label=f"capture {role}")
        row_hash = capture.get("row_identity_sha256")
        role_data = _mapping(roles[role], label=f"data role {role}")
        if (
            capture.get("site_count") != 73
            or capture.get("score_gradient_site_count") != 73
            or capture.get("hidden_width") != 640
            or capture.get("example_ids_sha256")
            != PINNED_SPLIT_ID_SHA256S[role]
            or not _is_sha256(row_hash)
            or capture.get("family_assignment_sha256")
            != role_data.get("family_assignment_sha256")
            or capture.get("contains_prompt_text") is not False
            or capture.get("contains_token_ids") is not False
            or _mapping(capture.get("exact_gate"), label="capture gate").get(
                "passed"
            )
            is not True
        ):
            raise ValueError("capture receipt drifted")
        exact_forward = capture.get("exact_forward_count")
        native_forward = capture.get("native_forward_count")
        backward = capture.get("backward_count")
        if (
            not isinstance(native_forward, int)
            or isinstance(native_forward, bool)
            or native_forward <= 0
            or backward != native_forward
            or exact_forward != 2 * native_forward
        ):
            raise ValueError("capture per-role execution count ratio drifted")
        row_hashes.append(row_hash)  # type: ignore[arg-type]
        role_gate = _mapping(capture.get("exact_gate"), label="capture gate")
        native_gate = _mapping(
            role_gate.get("native_exact_control"),
            label="capture native gate",
        )
        if (
            any(
                role_gate.get(field) is not True
                for field in (
                    "passed",
                    "event_order_equal",
                    "every_mutation_chart_input_byte_identical",
                    "every_mutation_delta_byte_identical",
                    "final_carrier_byte_identical",
                    "normalized_output_byte_identical",
                    "logits_byte_identical",
                    "source_state_fingerprint_unchanged",
                )
            )
            or native_gate.get("run") is not True
            or native_gate.get("literal_valid_query_logit_parity") is not True
            or native_gate.get("delta_nll_per_token") != 0.0
            or native_gate.get("native_to_candidate_kl_per_query") != 0.0
            or native_gate.get("top1_agreement_to_native") != 1.0
            or native_gate.get("maximum_absolute_logit_error") != 0.0
        ):
            raise ValueError("capture exact/native literal gate drifted")
    if len(set(row_hashes)) != len(ROLE_ORDER):
        raise ValueError("A/B/C capture row identities must be disjoint")
    aggregate_role_gates = _mapping(
        exact_gate.get("role_gates"),
        label="aggregate exact role gates",
    )
    if tuple(aggregate_role_gates) != ROLE_ORDER or any(
        dict(_mapping(aggregate_role_gates[role], label="aggregate role gate"))
        != dict(
            _mapping(
                _mapping(capture_receipts[role], label="capture").get(
                    "exact_gate"
                ),
                label="capture exact gate",
            )
        )
        for role in ROLE_ORDER
    ):
        raise ValueError("aggregate exact gate differs from role receipts")
    chart = _mapping(report.get("chart_catalog"), label="chart catalog")
    _verify_nested_digest(chart, digest_key="artifact_sha256")
    sites = chart.get("sites")
    if (
        chart.get("site_count") != 73
        or chart.get("source_eligible_site_count") != 73
        or chart.get("target_eligible_site_count") != 36
        or chart.get("coordinate_rank") != 640
        or chart.get("identity_basis_stored_scalar_count") != 0
        or chart.get("compression") is not False
        or not isinstance(sites, list)
        or len(sites) != 73
        or [item.get("site_id") for item in sites if isinstance(item, Mapping)]
        != [
            "lexical.ingress.output",
            *(
                site_id
                for index in range(1, 37)
                for site_id in (
                    f"mutation.{index}.input",
                    f"mutation.{index}.output",
                )
            ),
        ]
    ):
        raise ValueError("chart catalog drifted")
    edge_catalog = _mapping(report.get("edge_catalog"), label="edge catalog")
    _verify_nested_digest(edge_catalog, digest_key="artifact_sha256")
    edge_arms = edge_catalog.get("arms")
    if (
        edge_catalog.get("local_baseline_count") != 36
        or edge_catalog.get("full_width_coordinates") is not True
        or edge_catalog.get("dense_640_by_640_edges") is not True
        or edge_catalog.get("low_rank_factorization_used") is not False
        or edge_catalog.get("maximum_real_incoming_edges_per_target") != 1
        or edge_catalog.get("B_or_C_used_for_fit_or_selection") is not False
        or edge_catalog.get("installed_in_exact_runtime") is not False
        or edge_catalog.get("model_mutated") is not False
        or edge_catalog.get("compression") is not False
        or not isinstance(edge_arms, list)
        or [item.get("arm_id") for item in edge_arms if isinstance(item, Mapping)]
        != list(EDGE_ARM_ORDER)
    ):
        raise ValueError("edge catalog scope/capacity drifted")
    real_topology: list[tuple[object, object]] | None = None
    real_edge_count = edge_catalog.get("real_edge_count")
    if not isinstance(real_edge_count, int) or real_edge_count <= 0:
        raise ValueError("edge catalog contains no real edge")
    for raw_arm in edge_arms:
        arm = _mapping(raw_arm, label="edge arm")
        edges = arm.get("edges")
        if (
            arm.get("edge_count") != real_edge_count
            or arm.get("maximum_incoming_edges_per_target") != 1
            or arm.get("compression") is not False
            or not isinstance(edges, list)
        ):
            raise ValueError("edge arm budget drifted")
        topology: list[tuple[object, object]] = []
        targets_seen: set[object] = set()
        for raw_edge in edges:
            edge = _mapping(raw_edge, label="edge")
            target_id = edge.get("target_event_id")
            if (
                edge.get("strictly_forward") is not True
                or edge.get("cross_block") is not True
                or edge.get("cross_family") is not True
                or edge.get("coefficient_shape") != [640, 640]
                or edge.get("edge_low_rank_factorization_used") is not False
                or edge.get("dense_640_by_640") is not True
                or edge.get("factorized") is not False
                or not _is_sha256(edge.get("coefficient_sha256"))
                or not _is_sha256(edge.get("fit_design_sha256"))
                or not isinstance(edge.get("realized_numerical_rank"), int)
                or isinstance(edge.get("realized_numerical_rank"), bool)
                or int(edge.get("realized_numerical_rank", 0)) < 1
                or int(edge.get("realized_numerical_rank", 0)) > 640
                or edge.get("coefficient_serialized_in_report") is not False
                or target_id in targets_seen
            ):
                raise ValueError("dense edge metadata drifted")
            targets_seen.add(target_id)
            topology.append((edge.get("claimed_source_site_id"), target_id))
        if real_topology is None:
            real_topology = topology
        elif topology != real_topology:
            raise ValueError("negative control topology differs from real")
    predictive = _mapping(
        report.get("heldout_predictive_receipt"),
        label="predictive receipt",
    )
    _verify_nested_digest(predictive, digest_key="receipt_sha256")
    predictive_arms = _mapping(
        predictive.get("arm_metrics"),
        label="predictive arm metrics",
    )
    margins = _mapping(
        predictive.get("real_gain_margin_over_controls"),
        label="predictive margins",
    )
    real_gain = _mapping(
        predictive_arms.get("real_forward_cross_block_edges"),
        label="real predictive metrics",
    ).get("relative_SSE_gain_vs_local")
    if (
        predictive.get("arm_order") != list(ARM_ORDER)
        or set(predictive_arms) != set(ARM_ORDER)
    ):
        raise ValueError("predictive arm order drifted")
    if (
        _mapping(
            predictive_arms["local_target_baseline"],
            label="local predictive metrics",
        ).get("relative_SSE_gain_vs_local")
        != 0.0
        or _mapping(
            predictive_arms["zero_edge_control"],
            label="zero predictive metrics",
        ).get("relative_SSE_gain_vs_local")
        != 0.0
    ):
        raise ValueError("predictive local/zero control drifted")
    recomputed_margins: dict[str, float] = {}
    if isinstance(real_gain, (int, float)) and not isinstance(real_gain, bool):
        for arm_id in NEGATIVE_CONTROL_ARM_IDS:
            control_gain = _mapping(
                predictive_arms.get(arm_id),
                label=f"predictive {arm_id}",
            ).get("relative_SSE_gain_vs_local")
            if not isinstance(control_gain, (int, float)) or isinstance(
                control_gain,
                bool,
            ):
                raise ValueError("predictive control gain is invalid")
            recomputed_margins[arm_id] = float(real_gain) - float(control_gain)
    if set(margins) != set(NEGATIVE_CONTROL_ARM_IDS) or any(
        not isinstance(margins.get(arm_id), (int, float))
        or isinstance(margins.get(arm_id), bool)
        or float(margins[arm_id]) != recomputed_margins.get(arm_id)
        for arm_id in NEGATIVE_CONTROL_ARM_IDS
    ):
        raise ValueError("predictive margins were not recomputed from controls")
    recomputed_predictive = bool(
        isinstance(real_gain, (int, float))
        and not isinstance(real_gain, bool)
        and float(real_gain) >= 0.02
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) >= 0.01
            for value in margins.values()
        )
        and isinstance(predictive.get("paired_family_bootstrap_lower_bound"), (int, float))
        and float(predictive["paired_family_bootstrap_lower_bound"]) > 0.0
        and all(
            isinstance(item, Mapping) and item.get("real_gain_positive") is True
            for item in predictive.get("per_target_metrics", [])
        )
    )
    if (
        predictive.get("parent_edge_catalog_sha256")
        != edge_catalog.get("artifact_sha256")
        or predictive.get("B_row_identity_sha256") != row_hashes[1]
        or predictive.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES
        or predictive.get("heldout_refit_performed") is not False
        or predictive.get("edge_catalog_pruned_or_refitted_on_B") is not False
        or predictive.get("passed") is not recomputed_predictive
    ):
        raise ValueError("predictive receipt gate/lineage drifted")
    causal_value = report.get("heldout_causal_receipt")
    causal_passed = False
    causal_jvp_passed = False
    causal_finite_passed = False
    if causal_value is not None:
        causal = _mapping(causal_value, label="causal receipt")
        _verify_nested_digest(causal, digest_key="receipt_sha256")
        edge_metrics = causal.get("edge_metrics")
        if (
            causal.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
            or causal.get("parent_edge_catalog_sha256")
            != edge_catalog.get("artifact_sha256")
            or causal.get("parent_predictive_receipt_sha256")
            != predictive.get("receipt_sha256")
            or causal.get("C_row_identity_sha256") != row_hashes[2]
            or causal.get("source_fingerprint_before")
            != causal.get("source_fingerprint_after")
            or causal.get("direction_normalization")
            != DIRECTION_IMPLEMENTATION_RECEIPT
            or causal.get("protocol_direction_spec")
            != DIRECTION_PROTOCOL_LITERAL
            or causal.get("finite_displacement_scale")
            != FINITE_DISPLACEMENT_SCALE_LITERAL
            or causal.get("epsilon_times_source_RMS")
            != list(INTERVENTION_EPSILON_LADDER)
            or causal.get("edge_count") != real_edge_count
            or not isinstance(edge_metrics, list)
            or len(edge_metrics) != real_edge_count
            or causal.get("negative_control_margin_evaluated_per_edge") is not True
            or causal.get("baseline_model_output_replaced") is not False
            or causal.get("model_weights_mutated") is not False
        ):
            raise ValueError("causal receipt lineage/scope drifted")
        total_forward = total_backward = total_finite = 0
        metric_passes: list[bool] = []
        causal_topology: list[tuple[object, object]] = []
        for raw_metric in edge_metrics:
            metric = _mapping(raw_metric, label="causal edge metric")
            causal_topology.append(
                (metric.get("source_site_id"), metric.get("target_event_id"))
            )
            jvp_margins = _mapping(
                metric.get("jvp_cosine_margins_over_controls"),
                label="causal JVP margins",
            )
            finite_margins = _mapping(
                metric.get("finite_cosine_margins_over_controls"),
                label="causal finite margins",
            )
            control_jvp = _mapping(
                metric.get("control_minimum_jvp_cosines"),
                label="causal JVP controls",
            )
            control_finite = _mapping(
                metric.get("control_minimum_finite_cosines"),
                label="causal finite controls",
            )
            if (
                set(jvp_margins) != set(NEGATIVE_CONTROL_ARM_IDS)
                or set(finite_margins) != set(NEGATIVE_CONTROL_ARM_IDS)
                or set(control_jvp) != set(NEGATIVE_CONTROL_ARM_IDS)
                or set(control_finite) != set(NEGATIVE_CONTROL_ARM_IDS)
                or control_jvp.get("zero_edge_control") != 0.0
                or control_finite.get("zero_edge_control") != 0.0
            ):
                raise ValueError("causal per-control inventory drifted")
            real_jvp = metric.get("minimum_real_jvp_cosine")
            real_finite = metric.get("minimum_real_finite_cosine")
            if not isinstance(real_jvp, (int, float)) or isinstance(
                real_jvp,
                bool,
            ) or not isinstance(real_finite, (int, float)) or isinstance(
                real_finite,
                bool,
            ):
                raise ValueError("causal real cosines are invalid")
            for arm_id in NEGATIVE_CONTROL_ARM_IDS:
                if (
                    float(jvp_margins[arm_id])
                    != float(real_jvp) - float(control_jvp[arm_id])
                    or float(finite_margins[arm_id])
                    != float(real_finite) - float(control_finite[arm_id])
                ):
                    raise ValueError("causal control margin was fabricated")
            C_batch_count = metric.get("C_batch_count")
            if not isinstance(C_batch_count, int) or C_batch_count <= 0:
                raise ValueError("causal edge C batch count is invalid")
            expected_jvp_count = 4 * JVP_DIRECTION_COUNT_PER_EDGE * C_batch_count
            expected_finite_count = (
                4
                * 2
                * JVP_DIRECTION_COUNT_PER_EDGE
                * len(INTERVENTION_EPSILON_LADDER)
                * C_batch_count
            )
            expected_forward_count = (
                4
                * (
                    2
                    + JVP_DIRECTION_COUNT_PER_EDGE
                    + 2
                    * JVP_DIRECTION_COUNT_PER_EDGE
                    * len(INTERVENTION_EPSILON_LADDER)
                )
                * C_batch_count
            )
            if (
                metric.get("exact_forward_count") != expected_forward_count
                or metric.get("exact_jvp_count") != expected_jvp_count
                or metric.get("backward_or_jvp_count") != expected_jvp_count
                or metric.get("finite_intervention_forward_count")
                != expected_finite_count
            ):
                raise ValueError("causal per-edge call count was fabricated")
            jvp_ok = bool(
                float(metric.get("minimum_real_jvp_cosine", -2.0)) >= 0.90
                and float(metric.get("maximum_real_jvp_relative_norm_error", math.inf)) <= 0.25
                and all(float(value) >= 0.05 for value in jvp_margins.values())
            )
            finite_ok = bool(
                float(metric.get("minimum_real_finite_cosine", -2.0)) >= 0.80
                and float(metric.get("maximum_real_finite_relative_norm_error", math.inf)) <= 0.35
                and float(
                    metric.get(
                        "maximum_adjacent_epsilon_derivative_disagreement",
                        math.inf,
                    )
                )
                <= 0.10
                and all(float(value) >= 0.05 for value in finite_margins.values())
            )
            if (
                metric.get("jvp_passed") is not jvp_ok
                or metric.get("finite_intervention_passed") is not finite_ok
                or metric.get("passed") is not (jvp_ok and finite_ok)
            ):
                raise ValueError("causal per-edge threshold recomputation drifted")
            total_forward += int(metric.get("exact_forward_count", -1))
            total_backward += int(metric.get("backward_or_jvp_count", -1))
            total_finite += int(metric.get("finite_intervention_forward_count", -1))
            metric_passes.append(jvp_ok and finite_ok)
        if causal_topology != real_topology:
            raise ValueError("causal receipt edge topology differs from real")
        if (
            causal.get("exact_forward_count") != total_forward
            or causal.get("backward_or_JVP_count") != total_backward
            or causal.get("finite_intervention_forward_count") != total_finite
            or causal.get("JVP_gate_passed") is not all(
                bool(_mapping(item, label="causal edge").get("jvp_passed"))
                for item in edge_metrics
            )
            or causal.get("finite_intervention_gate_passed") is not all(
                bool(
                    _mapping(item, label="causal edge").get(
                        "finite_intervention_passed"
                    )
                )
                for item in edge_metrics
            )
            or causal.get("passed") is not all(metric_passes)
        ):
            raise ValueError("causal execution/gate ledger drifted")
        causal_jvp_passed = all(
            bool(_mapping(item, label="causal edge").get("jvp_passed"))
            for item in edge_metrics
        )
        causal_finite_passed = all(
            bool(
                _mapping(item, label="causal edge").get(
                    "finite_intervention_passed"
                )
            )
            for item in edge_metrics
        )
        causal_passed = causal_jvp_passed and causal_finite_passed
    costs = _mapping(report.get("analysis_cost"), label="analysis cost")
    capture_exact = sum(
        int(_mapping(capture_receipts[role], label="capture").get("exact_forward_count", -1))
        for role in ROLE_ORDER
    )
    capture_native = sum(
        int(_mapping(capture_receipts[role], label="capture").get("native_forward_count", -1))
        for role in ROLE_ORDER
    )
    capture_backward = sum(
        int(_mapping(capture_receipts[role], label="capture").get("backward_count", -1))
        for role in ROLE_ORDER
    )
    causal_forward = 0 if causal_value is None else int(
        _mapping(causal_value, label="causal").get("exact_forward_count", -1)
    )
    causal_backward = 0 if causal_value is None else int(
        _mapping(causal_value, label="causal").get("backward_or_JVP_count", -1)
    )
    causal_finite = 0 if causal_value is None else int(
        _mapping(causal_value, label="causal").get(
            "finite_intervention_forward_count",
            -1,
        )
    )
    if (
        costs.get("source_model_parameter_count") != PINNED_SOURCE_PARAMETER_COUNT
        or costs.get("model_parameter_count_after_experiment")
        != PINNED_SOURCE_PARAMETER_COUNT
        or any(
            costs.get(key) != 0.0
            for key in (
                "model_parameter_reduction_fraction",
                "model_storage_reduction_fraction",
                "model_MAC_reduction_fraction",
            )
        )
        or costs.get("implicit_identity_chart_scalar_count") != 0
        or costs.get("sidecar_local_baseline_scalar_count")
        != edge_catalog.get("local_baseline_scalar_count")
        or costs.get("sidecar_edge_scalar_count")
        != edge_catalog.get("sidecar_edge_scalar_count")
        or costs.get("sidecar_cost_subtracted_from_model_cost") is not False
        or costs.get("fit_design_MAC_estimate")
        != edge_catalog.get("fit_design_mac_estimate")
        or costs.get("fit_dense_solve_FLOP_lower_bound")
        != edge_catalog.get("fit_dense_solve_flop_lower_bound")
        or costs.get("rank_decomposition_FLOP_lower_bound")
        != edge_catalog.get("rank_decomposition_flop_lower_bound")
        or costs.get("heldout_prediction_MAC_estimate")
        != predictive.get("prediction_mac_estimate")
        or costs.get("capture_exact_forward_count") != capture_exact
        or costs.get("capture_native_forward_count") != capture_native
        or costs.get("causal_exact_forward_count") != causal_forward
        or costs.get("exact_forward_count") != capture_exact + causal_forward
        or costs.get("capture_backward_count") != capture_backward
        or costs.get("causal_backward_or_JVP_count") != causal_backward
        or costs.get("backward_or_JVP_count")
        != capture_backward + causal_backward
        or costs.get("finite_intervention_forward_count") != causal_finite
        or not isinstance(costs.get("peak_resident_bytes"), int)
        or int(costs["peak_resident_bytes"]) <= 0
        or costs.get("latency_improvement_claimed") is not False
        or costs.get("fit_cost_is_analysis_only") is not True
        or costs.get("sidecar_inference_cost_claimed") is not False
        or costs.get("compression") is not False
    ):
        raise ValueError("exact edge-sidecar analysis cost ledger drifted")
    qualification = _mapping(report.get("qualification"), label="qualification")
    expected_qualification = _qualification_from_flags(
        exact_gate_passed=True,
        predictive_passed=recomputed_predictive,
        jvp_passed=causal_jvp_passed,
        finite_passed=causal_finite_passed,
    )
    if dict(qualification) != expected_qualification:
        raise ValueError("exact edge-sidecar qualification was not recomputed")
    without_hash = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    if report.get("report_sha256") != _report_sha256(without_hash):
        raise ValueError("exact edge-sidecar report SHA-256 is invalid")
    return copy.deepcopy(dict(report))


def load_exact_edge_sidecar_experiment_report(
    source: Path | str,
) -> dict[str, object]:
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("exact edge-sidecar report is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("exact edge-sidecar report must be a JSON object")
    return validate_exact_edge_sidecar_experiment_report(value)


def write_exact_edge_sidecar_experiment_report(
    report: Mapping[str, object],
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    authenticated = validate_exact_edge_sidecar_experiment_report(report)
    path = Path(destination)
    if path.suffix != ".json":
        raise ValueError("exact edge-sidecar report output must use .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(authenticated, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        raise FileExistsError(
            "refusing to overwrite an exact edge-sidecar report"
        ) from None
    return authenticated


def run_gemma3_exact_edge_sidecar_experiment(
    *,
    revision: str = PINNED_MODEL_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    protocol_path: Path | str = DEFAULT_PROTOCOL_PATH,
    panel_path: Path | str = PINNED_PANEL_PATH,
    model_id: str = PINNED_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = DEFAULT_BATCH_SIZE,
    causal_evaluator: CausalEvaluator | None = evaluate_exact_edge_sidecar_causal,
) -> dict[str, object]:
    """Run the exact A/B/C edge axis; never create a compressed candidate."""

    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("exact edge-sidecar report output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite an exact edge-sidecar report")
    if model_id != PINNED_MODEL_ID or revision != PINNED_MODEL_REVISION:
        raise ValueError("exact edge-sidecar runner requires the pinned Gemma revision")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("exact edge-sidecar runner requires CPU float32")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if causal_evaluator is not None and not callable(causal_evaluator):
        raise TypeError("causal_evaluator must be callable")
    protocol, protocol_receipt = _protocol_receipt(protocol_path)
    split = load_pinned_edge_sidecar_split(panel_path)
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("load pinned local Gemma exact reference")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if (
        len(adapter.layers) != PINNED_LAYER_COUNT
        or sum(parameter.numel() for parameter in model.parameters())
        != PINNED_SOURCE_PARAMETER_COUNT
    ):
        raise RuntimeError("loaded source model does not match the pinned inventory")
    source_before = adapter.model_fingerprint()
    source_execution_before = adapter.execution_fingerprint()
    source_guard = _FrozenModelTensorGuard(model)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    captures: dict[str, EdgeSidecarRoleCapture] = {}
    causal: EdgeSidecarCausalReceipt | None = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DETERMINISTIC_SEED)
        torch.use_deterministic_algorithms(True)
        try:
            _progress("compile exact tied lexical + all attention + all MLP body")
            compilation = compile_gemma3_joint_exact_control(
                adapter,
                dtype=torch.float32,
                device=device,
            )
            exact_executor = compilation.executor
            exact_guard = _FrozenModelTensorGuard(exact_executor)
            prepared: dict[str, tuple[EdgeSidecarPreparedBatch, ...]] = {}
            for role in ROLE_ORDER:
                examples = split.examples_for(role)
                prepared[role] = _prepare_role_batches(
                    tokenizer=tokenizer,
                    adapter=adapter,
                    examples=examples,
                    batch_size=batch_size,
                    device=device,
                )
                role_families = {
                    item.example_id: item.family_id for item in examples
                }

                def native_provider(
                    batch: EdgeSidecarPreparedBatch,
                ) -> Tensor:
                    native = adapter.forward(
                        {
                            "input_ids": batch.input_ids,
                            "attention_mask": batch.valid_mask,
                        }
                    )
                    return native.logits

                _progress(f"capture exact 73-site {role} role + score tape")
                captures[role] = capture_exact_edge_sidecar_role(
                    executor=exact_executor,
                    batches=prepared[role],
                    role=role,
                    source_fingerprint=source_before,
                    example_ids_sha256=PINNED_SPLIT_ID_SHA256S[role],
                    family_by_example=role_families,
                    native_logits_provider=native_provider,
                )
            _progress("fit A-only full-width local baselines and dense edges")
            catalog = fit_exact_edge_sidecar_catalog(captures["A_fit"])
            _progress("evaluate frozen real/zero/shuffled controls on B")
            predictive = evaluate_exact_edge_sidecar_predictive(
                catalog=catalog,
                heldout_capture=captures["B_predictive"],
            )
            if causal_evaluator is not None and predictive.passed:
                request = EdgeSidecarCausalRequest(
                    protocol_sha256=EXPECTED_PROTOCOL_SHA256,
                    source_fingerprint=source_before,
                    executor=exact_executor,
                    catalog=catalog,
                    predictive_receipt=predictive,
                    C_capture=captures["C_intervention"],
                    batches=prepared["C_intervention"],
                )
                _progress("invoke heldout C JVP/shadow causal evaluator")
                causal = causal_evaluator(request)
                if not isinstance(causal, EdgeSidecarCausalReceipt):
                    raise TypeError(
                        "causal evaluator must return EdgeSidecarCausalReceipt"
                    )
                if (
                    causal.protocol_sha256 != EXPECTED_PROTOCOL_SHA256
                    or causal.parent_edge_catalog_sha256
                    != catalog.artifact_sha256
                    or causal.parent_predictive_receipt_sha256
                    != predictive.receipt_sha256
                    or causal.C_row_identity_sha256
                    != captures["C_intervention"].capture.row_identity_sha256
                    or causal.source_fingerprint_before != source_before
                ):
                    raise ValueError("causal evaluator returned unrelated evidence")
            exact_guard.assert_unchanged()
            catalog.assert_integrity()
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)
    source_guard.assert_unchanged()
    source_after = adapter.model_fingerprint()
    source_execution_after = adapter.execution_fingerprint()
    if (
        source_after != source_before
        or source_execution_after != source_execution_before
    ):
        raise RuntimeError("source Gemma changed during exact edge-sidecar work")
    report = _assemble_report(
        protocol_receipt=protocol_receipt,
        protocol=protocol,
        split=split,
        model_metadata=_model_provenance(
            model,
            model_id=model_id,
            requested_revision=revision,
        ),
        compilation=compilation,
        captures=captures,
        catalog=catalog,
        predictive=predictive,
        causal=causal,
        source_fingerprint_before=source_before,
        source_fingerprint_after=source_after,
        batch_size=batch_size,
        peak_resident_bytes=_peak_resident_bytes(),
    )
    authenticated = write_exact_edge_sidecar_experiment_report(
        report,
        destination,
    )
    _progress(f"wrote {destination}")
    gc.collect()
    return authenticated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-compression exact Gemma edge-sidecar rung. The built-in "
            "C JVP/shadow evaluator runs only after the held-out B gate passes."
        )
    )
    parser.add_argument("--revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--panel", type=Path, default=PINNED_PANEL_PATH)
    parser.add_argument("--model-id", default=PINNED_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="authenticate an existing report instead of running the model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.check:
        report = load_exact_edge_sidecar_experiment_report(arguments.output)
        action = "checked"
    else:
        report = run_gemma3_exact_edge_sidecar_experiment(
            revision=arguments.revision,
            output=arguments.output,
            protocol_path=arguments.protocol,
            panel_path=arguments.panel,
            model_id=arguments.model_id,
            cache_dir=arguments.cache_dir,
            device_name=arguments.device,
            dtype=arguments.dtype,
            batch_size=arguments.batch_size,
        )
        action = "written"
    print(
        json.dumps(
            {
                "action": action,
                "output": str(arguments.output),
                "report_sha256": report["report_sha256"],
                "edge_axis_passed": _mapping(
                    report["qualification"],
                    label="qualification",
                )["edge_axis_passed"],
                "compression": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ARM_ORDER",
    "DEFAULT_OUTPUT",
    "DEFAULT_PROTOCOL_PATH",
    "EDGE_ARM_ORDER",
    "EXACT_EDGE_SIDECAR_EXPERIMENT_FORMAT_VERSION",
    "EXACT_EDGE_SIDECAR_EXPERIMENT_SCHEMA",
    "EXPECTED_PROTOCOL_SHA256",
    "EdgeSidecarCausalEdgeMetric",
    "EdgeSidecarCausalReceipt",
    "EdgeSidecarCausalRequest",
    "EdgeSidecarPreparedBatch",
    "EdgeSidecarPredictiveReceipt",
    "EdgeSidecarRoleCapture",
    "FittedEdgeSidecarCatalog",
    "FrozenDenseEdgeArtifact",
    "FrozenEdgeSidecarArm",
    "PinnedEdgeSidecarExample",
    "PinnedEdgeSidecarSplit",
    "build_edge_sidecar_causal_receipt",
    "capture_exact_edge_sidecar_role",
    "evaluate_exact_edge_sidecar_predictive",
    "evaluate_exact_edge_sidecar_causal",
    "fit_exact_edge_sidecar_catalog",
    "load_exact_edge_sidecar_experiment_report",
    "load_pinned_edge_sidecar_split",
    "main",
    "run_gemma3_exact_edge_sidecar_experiment",
    "validate_exact_edge_sidecar_experiment_report",
    "write_exact_edge_sidecar_experiment_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
