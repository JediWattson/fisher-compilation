"""Live Gemma driver for the parts-first ordered-supermode pilot.

This is the model-facing half of the experiment.  It loads only the pinned
``google/gemma-3-270m`` revision, turns the tracked forced-choice panel into a
deterministic 6/2/2-per-family A/B/C split, and captures one correct-choice
next-token NLL row per example.  Physical responses come from the committed
Gemma adapter activation ABI through :mod:`gemma3_parts_first_atom_capture`.
The path router uses one global, baseline-frozen chart of the contextual
terminal normalized hidden state; it does not claim separate local charts for
the four stages.

The C arm installs four temporary native gates (lexical ingress, attention,
strictly later MLP, lexical readout) for the one frozen candidate selected by
A then B.  An alpha-one pass establishes literal no-op parity before the
finite pass.  Hooks are removed and source model/execution fingerprints are
checked after the intervention.

Importing this module is side-effect free.  In particular, Transformers is
imported and the checkpoint is loaded only after the non-preflight CLI path is
entered.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Final, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gemma3_downstream_retention_experiment import (
    ForcedChoiceExample,
    ForcedChoicePanel,
    load_forced_choice_panel,
)
from .gemma3_parts_first_atom_capture import (
    Gemma3PartsFirstCaptureBinding,
    PartsFirstAtomResponseBatch,
    bind_parts_first_gemma_adapter,
    iter_parts_first_atom_response_batches,
)
from .gemma3_parts_first_native_intervention import (
    FrozenNativePartGate,
    NativePartGateExecutionReceipt,
    run_with_native_part_gates,
)
from .gemma3_parts_first_supermode_experiment import (
    AtomResponseAxisEntry,
    CaptureRole,
    CaptureRow,
    ExactLiftCaptureReceipt,
    FiniteInterventionEvidence,
    FrozenChartAtlas,
    FrozenSupermodeArtifact,
    MODEL_CHART_SURFACE_ID,
    MODEL_SCORE_OBJECTIVE_ID,
    PartsFirstCaptureBundle,
    PartsFirstPilotConfig,
    PartsFirstPilotReport,
    run_gemma3_parts_first_supermode_experiment,
    write_parts_first_capture_bundle,
)
from .gemma3_parts_first_supermode_protocol import (
    MODEL_ID,
    SOURCE_MODEL_FINGERPRINT_SHA256,
    build_parts_first_supermode_protocol,
)
from .gemma3_parts_first_supermodes import (
    ATTENTION_HEAD_COUNT,
    GEMMA3_LAYER_COUNT,
    HIDDEN_WIDTH,
    MLP_CHANNEL_COUNT,
    PhysicalAtomFamily,
    SOURCE_UNIQUE_PARAMETER_COUNT,
    VOCABULARY_SIZE,
    build_gemma3_parts_catalog,
)


PINNED_MODEL_REVISION: Final = (
    "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
)
PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256: Final = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
PINNED_PANEL_PATH: Final = Path("examples/gemma3_downstream_retention_v1.json")
PINNED_PANEL_FILE_SHA256: Final = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PINNED_PANEL_ID: Final = "gemma3-single-token-forced-choice-v1"
ROLE_ORDER: Final = ("A_discovery", "B_confirmation", "C_intervention")
ROLE_COUNTS_PER_FAMILY: Final = (6, 2, 2)
FAMILY_COUNT: Final = 6
EXAMPLES_PER_FAMILY: Final = 10
RESPONSE_OCCURRENCE_COUNT: Final = 38_216
GLOBAL_PATH_CHART_SURFACE_ID: Final = MODEL_CHART_SURFACE_ID
DEFAULT_CAPTURE_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/parts-first-supermode-captures-v1.json"
)
DEFAULT_REPORT_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/parts-first-supermode-model-pilot-v1.json"
)

_GATE_RECEIPT_SET_DOMAIN = (
    b"fisher_graph:gemma3-parts-first-native-gate-receipt-set:v1\0"
)
_EXACT_CONTROL_EVIDENCE_DOMAIN = (
    b"fisher_graph:gemma3-parts-first-exact-control-evidence:v1\0"
)
_TENSOR_VALUE_DOMAIN = b"fisher_graph:tensor-value:v1\0"


__all__ = [
    "DEFAULT_CAPTURE_OUTPUT",
    "DEFAULT_REPORT_OUTPUT",
    "GLOBAL_PATH_CHART_SURFACE_ID",
    "PINNED_MODEL_REVISION",
    "PINNED_PANEL_PATH",
    "PartsFirstLiveModelPilot",
    "PartsFirstModelRunnerConfig",
    "ExactLiftPrecheckReceipt",
    "PilotPanelRole",
    "build_complete_response_axis",
    "build_correct_choice_nll_batch",
    "build_model_runner_preflight",
    "load_pinned_parts_first_panel",
    "main",
    "native_gates_for_frozen_artifact",
    "response_batch_to_capture_row",
    "run_exact_lift_precheck",
    "run_parts_first_model_pilot",
    "split_parts_first_panel",
]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_value_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("tensor value hash requires a Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(canonical.dtype),
            "shape": list(canonical.shape),
        }
    )
    raw = canonical.view(torch.uint8).numpy().tobytes()
    return _sha256_bytes(_TENSOR_VALUE_DOMAIN + header + b"\0" + raw)


def _receipt_set_sha256(
    receipts: Sequence[tuple[str, str, NativePartGateExecutionReceipt]],
) -> str:
    payload = [
        {
            "row_id": row_id,
            "condition": condition,
            "receipt_sha256": receipt.receipt_sha256,
        }
        for row_id, condition, receipt in receipts
    ]
    return _sha256_bytes(
        _GATE_RECEIPT_SET_DOMAIN + _canonical_json_bytes(payload)
    )


@dataclass(frozen=True, slots=True)
class PilotPanelRole:
    role: str
    examples: tuple[ForcedChoiceExample, ...]

    def __post_init__(self) -> None:
        if self.role not in ROLE_ORDER:
            raise ValueError("panel role is not preregistered")
        expected_per_family = ROLE_COUNTS_PER_FAMILY[ROLE_ORDER.index(self.role)]
        counts = Counter(example.family_id for example in self.examples)
        if (
            len(counts) != FAMILY_COUNT
            or set(counts.values()) != {expected_per_family}
        ):
            raise ValueError("panel role does not have its frozen per-family count")
        ids = tuple(example.example_id for example in self.examples)
        if len(ids) != len(set(ids)):
            raise ValueError("panel role example IDs must be unique")


@dataclass(frozen=True, slots=True)
class PartsFirstModelRunnerConfig:
    panel_path: Path = PINNED_PANEL_PATH
    cache_dir: Path | None = None
    local_files_only: bool = True
    maximum_prompt_tokens: int = 128
    capture_output: Path = DEFAULT_CAPTURE_OUTPUT
    report_output: Path = DEFAULT_REPORT_OUTPUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_path", Path(self.panel_path))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "capture_output", Path(self.capture_output))
        object.__setattr__(self, "report_output", Path(self.report_output))
        if type(self.local_files_only) is not bool:
            raise TypeError("local_files_only must be bool")
        if (
            type(self.maximum_prompt_tokens) is not int
            or self.maximum_prompt_tokens < 2
        ):
            raise ValueError("maximum_prompt_tokens must be at least two")
        if self.capture_output.suffix != ".json":
            raise ValueError("capture output must use .json")
        if self.report_output.suffix != ".json":
            raise ValueError("report output must use .json")


@dataclass(frozen=True, slots=True)
class ExactLiftPrecheckReceipt:
    """Ephemeral proof from one literal native/exact control execution."""

    compilation_kind: str
    ordered_event_ids: tuple[str, ...]
    compiler_metadata_json: str
    operator_receipts_json: str
    valid_logit_shape: tuple[int, int]
    native_valid_logits_sha256: str
    exact_valid_logits_sha256: str
    valid_logit_parity_passed: bool
    source_model_fingerprint_before: str
    source_model_fingerprint_after: str
    source_execution_fingerprint_before: str
    source_execution_fingerprint_after: str
    source_parameter_count: int
    exact_parameter_inventory_parity: bool
    source_independence_passed: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        expected_events = (
            "lexical.ingress",
            *(f"mutation.{index}" for index in range(1, 37)),
            "terminal.rmsnorm",
            "lexical.egress",
        )
        if self.compilation_kind != "exact_39_event_control":
            raise ValueError("exact-lift precheck compilation kind drifted")
        if self.ordered_event_ids != expected_events:
            raise ValueError("exact-lift precheck schedule drifted")
        if self.valid_logit_shape != (1, VOCABULARY_SIZE):
            raise ValueError("exact-lift precheck did not compare one full-vocab row")
        if type(self.valid_logit_parity_passed) is not bool:
            raise TypeError("valid-logit parity decision must be bool")
        if self.source_parameter_count != SOURCE_UNIQUE_PARAMETER_COUNT:
            raise ValueError("exact-lift precheck source inventory drifted")
        if (
            type(self.exact_parameter_inventory_parity) is not bool
            or type(self.source_independence_passed) is not bool
        ):
            raise TypeError("exact-lift precheck decisions must be bool")
        for label, value in (
            ("native valid logits", self.native_valid_logits_sha256),
            ("exact valid logits", self.exact_valid_logits_sha256),
            ("source model before", self.source_model_fingerprint_before),
            ("source model after", self.source_model_fingerprint_after),
            (
                "source execution before",
                self.source_execution_fingerprint_before,
            ),
            ("source execution after", self.source_execution_fingerprint_after),
        ):
            _require_sha256(value, label=label)
        if self.valid_logit_parity_passed and (
            self.native_valid_logits_sha256 != self.exact_valid_logits_sha256
        ):
            raise ValueError("equal valid logits must have one value hash")
        for label, raw in (
            ("compiler metadata", self.compiler_metadata_json),
            ("operator receipts", self.operator_receipts_json),
        ):
            if not isinstance(raw, str):
                raise TypeError(f"{label} JSON must be str")
            try:
                decoded = json.loads(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label} JSON is invalid") from error
            if _canonical_json_bytes(decoded).decode("utf-8") != raw:
                raise ValueError(f"{label} JSON is not canonical")
        operator_receipts = json.loads(self.operator_receipts_json)
        if (
            not isinstance(operator_receipts, list)
            or len(operator_receipts) != 36
            or tuple(
                item.get("event_id")
                for item in operator_receipts
                if isinstance(item, Mapping)
            )
            != expected_events[1:-2]
        ):
            raise ValueError("exact-lift precheck operator receipts drifted")
        computed = _sha256_bytes(
            _EXACT_CONTROL_EVIDENCE_DOMAIN
            + _canonical_json_bytes(self._payload())
        )
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif self.receipt_sha256 != computed:
            raise ValueError("exact-lift precheck receipt hash drifted")

    @property
    def passed(self) -> bool:
        return bool(
            self.source_model_fingerprint_before
            == SOURCE_MODEL_FINGERPRINT_SHA256
            == self.source_model_fingerprint_after
            and self.source_execution_fingerprint_before
            == PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            == self.source_execution_fingerprint_after
            and self.exact_parameter_inventory_parity
            and self.source_independence_passed
            and self.valid_logit_parity_passed
            and self.native_valid_logits_sha256
            == self.exact_valid_logits_sha256
        )

    def _payload(self) -> dict[str, object]:
        return {
            "compilation_kind": self.compilation_kind,
            "compilation_metadata": json.loads(self.compiler_metadata_json),
            "ordered_event_ids": list(self.ordered_event_ids),
            "operator_receipts": json.loads(self.operator_receipts_json),
            "valid_logit_parity": {
                "passed": self.valid_logit_parity_passed,
                "shape": list(self.valid_logit_shape),
                "native_value_sha256": self.native_valid_logits_sha256,
                "exact_value_sha256": self.exact_valid_logits_sha256,
            },
            "source_model_fingerprint_before": (
                self.source_model_fingerprint_before
            ),
            "source_model_fingerprint_after": self.source_model_fingerprint_after,
            "source_execution_fingerprint_before": (
                self.source_execution_fingerprint_before
            ),
            "source_execution_fingerprint_after": (
                self.source_execution_fingerprint_after
            ),
            "source_parameter_count": self.source_parameter_count,
            "exact_parameter_inventory_parity": (
                self.exact_parameter_inventory_parity
            ),
            "source_independence_passed": self.source_independence_passed,
        }


class _ExactLiftCompiler(Protocol):
    def __call__(
        self,
        adapter: Gemma3CausalLMAdapter,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object: ...


def run_exact_lift_precheck(
    adapter: Gemma3CausalLMAdapter,
    first_batch: CalibrationBatch,
    *,
    compiler: _ExactLiftCompiler | None = None,
) -> ExactLiftPrecheckReceipt:
    """Compile, execute, authenticate, and release the exact control clone."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("exact-lift precheck requires a Gemma adapter")
    if not isinstance(first_batch, CalibrationBatch):
        raise TypeError("exact-lift precheck requires a CalibrationBatch")
    if compiler is None:
        from .gemma3_joint_cross_block_compiler import (
            compile_gemma3_joint_exact_control,
        )

        compiler = compile_gemma3_joint_exact_control
    source_model_before = adapter.model_fingerprint()
    source_execution_before = adapter.execution_fingerprint()
    source_device = next(adapter.module.parameters()).device
    compilation = compiler(
        adapter,
        dtype=torch.float32,
        device=source_device,
    )
    executor: object | None = None
    exact_execution: object | None = None
    native_valid: Tensor | None = None
    exact_valid: Tensor | None = None
    metadata_callback: object | None = None
    execute_callback: object | None = None
    forward_components_callback: object | None = None
    exact_logits: Tensor | None = None
    try:
        metadata_callback = getattr(compilation, "metadata", None)
        metadata = metadata_callback() if callable(metadata_callback) else None
        executor = getattr(compilation, "executor", None)
        ordered_events = getattr(executor, "ordered_event_ids", None)
        independence = getattr(compilation, "source_independence_audit", None)
        if not isinstance(metadata, Mapping):
            raise ValueError("exact compiler omitted authenticated metadata")
        if not isinstance(ordered_events, tuple):
            raise ValueError("exact compiler omitted its event schedule")
        valid_positions = first_batch.valid_positions
        if (
            valid_positions.dtype is not torch.bool
            or int(valid_positions.sum().item()) != 1
        ):
            raise ValueError("exact-lift precheck requires one valid query row")
        input_ids = first_batch.model_inputs.get("input_ids")
        if not isinstance(input_ids, Tensor):
            raise TypeError("exact-lift precheck batch omitted input_ids")
        sequence = adapter.prepare_sequence(first_batch.model_inputs)
        with torch.no_grad():
            native_run = adapter.forward(first_batch.model_inputs)
            if native_run.logits.shape[:2] != valid_positions.shape:
                raise ValueError("native logits disagree with the objective mask")
            native_valid = native_run.logits[valid_positions].detach().clone()
            del native_run
            execute_callback = getattr(executor, "execute", None)
            if callable(execute_callback):
                exact_execution = execute_callback(input_ids, sequence)
            else:
                forward_components_callback = getattr(
                    executor,
                    "forward_components",
                    None,
                )
                if not callable(forward_components_callback):
                    raise TypeError("exact control omitted its execution method")
                exact_execution = forward_components_callback(
                    input_ids,
                    sequence,
                )
            exact_logits = getattr(exact_execution, "logits", None)
            if not isinstance(exact_logits, Tensor):
                raise TypeError("exact control execution omitted logits")
            if exact_logits.shape[:2] != valid_positions.shape:
                raise ValueError("exact logits disagree with the objective mask")
            exact_valid = exact_logits[valid_positions].detach().clone()
        execution_events = getattr(exact_execution, "event_ids", None)
        operator_receipts = getattr(exact_execution, "operator_receipts", None)
        if execution_events != ordered_events:
            raise RuntimeError("exact control runtime event schedule drifted")
        if not isinstance(operator_receipts, tuple):
            raise TypeError("exact control omitted operator receipts")
        operator_metadata: list[dict[str, object]] = []
        for item in operator_receipts:
            callback = getattr(item, "metadata", None)
            value = callback() if callable(callback) else None
            if not isinstance(value, Mapping):
                raise TypeError("exact control operator receipt omitted metadata")
            operator_metadata.append(dict(value))
        if native_valid.shape != exact_valid.shape:
            parity = False
        else:
            parity = torch.equal(native_valid, exact_valid)
        compiler_metadata_json = _canonical_json_bytes(dict(metadata)).decode(
            "utf-8"
        )
        operator_receipts_json = _canonical_json_bytes(operator_metadata).decode(
            "utf-8"
        )
        valid_shape = tuple(int(value) for value in native_valid.shape)
        if len(valid_shape) != 2:
            raise ValueError("valid logits must have row and vocabulary axes")
        native_valid_sha256 = _tensor_value_sha256(native_valid)
        exact_valid_sha256 = _tensor_value_sha256(exact_valid)
        compilation_kind = str(getattr(compilation, "compilation_kind", ""))
        source_parameter_count = int(
            getattr(compilation, "source_model_parameter_count", 0)
        )
        exact_parameter_inventory_parity = (
            metadata.get("exact_parameter_inventory_parity") is True
        )
        source_independence_passed = bool(
            isinstance(independence, Mapping)
            and independence.get("passed") is True
        )
    finally:
        # The exact control is a full checkpoint clone.  It is a precheck,
        # not the 60-row capture runtime, so release it before capture begins.
        exact_valid = None
        native_valid = None
        exact_logits = None
        exact_execution = None
        execute_callback = None
        forward_components_callback = None
        metadata_callback = None
        executor = None
        compilation = None
        gc.collect()
    source_model_after = adapter.model_fingerprint()
    source_execution_after = adapter.execution_fingerprint()
    receipt = ExactLiftPrecheckReceipt(
        compilation_kind=compilation_kind,
        ordered_event_ids=ordered_events,
        compiler_metadata_json=compiler_metadata_json,
        operator_receipts_json=operator_receipts_json,
        valid_logit_shape=(valid_shape[0], valid_shape[1]),
        native_valid_logits_sha256=native_valid_sha256,
        exact_valid_logits_sha256=exact_valid_sha256,
        valid_logit_parity_passed=parity,
        source_model_fingerprint_before=source_model_before,
        source_model_fingerprint_after=source_model_after,
        source_execution_fingerprint_before=source_execution_before,
        source_execution_fingerprint_after=source_execution_after,
        source_parameter_count=source_parameter_count,
        exact_parameter_inventory_parity=exact_parameter_inventory_parity,
        source_independence_passed=source_independence_passed,
    )
    if not receipt.passed:
        raise RuntimeError("committed exact 39-event lift precheck failed")
    return receipt


def load_pinned_parts_first_panel(
    path: Path | str = PINNED_PANEL_PATH,
) -> ForcedChoicePanel:
    """Load the tracked 60-row panel only when its exact bytes still match."""

    target = Path(path)
    raw = target.read_bytes()
    if _sha256_bytes(raw) != PINNED_PANEL_FILE_SHA256:
        raise ValueError("parts-first panel bytes differ from the frozen panel")
    panel = load_forced_choice_panel(target)
    counts = Counter(example.family_id for example in panel.examples)
    if (
        panel.panel_id != PINNED_PANEL_ID
        or len(panel.examples) != FAMILY_COUNT * EXAMPLES_PER_FAMILY
        or len(counts) != FAMILY_COUNT
        or set(counts.values()) != {EXAMPLES_PER_FAMILY}
    ):
        raise ValueError("parts-first panel inventory drifted")
    return panel


def split_parts_first_panel(
    panel: ForcedChoicePanel,
) -> tuple[PilotPanelRole, ...]:
    """Take the first 6, next 2, and last 2 rows of every family."""

    if not isinstance(panel, ForcedChoicePanel):
        raise TypeError("panel must be a ForcedChoicePanel")
    family_offsets: dict[str, int] = {}
    rows: dict[str, list[ForcedChoiceExample]] = {
        role: [] for role in ROLE_ORDER
    }
    for example in panel.examples:
        offset = family_offsets.get(example.family_id, 0)
        family_offsets[example.family_id] = offset + 1
        if offset < 6:
            role = ROLE_ORDER[0]
        elif offset < 8:
            role = ROLE_ORDER[1]
        elif offset < 10:
            role = ROLE_ORDER[2]
        else:
            raise ValueError("panel family contains more than ten examples")
        rows[role].append(example)
    if (
        len(family_offsets) != FAMILY_COUNT
        or set(family_offsets.values()) != {EXAMPLES_PER_FAMILY}
    ):
        raise ValueError("panel cannot support the frozen 6/2/2 split")
    result = tuple(
        PilotPanelRole(role, tuple(rows[role])) for role in ROLE_ORDER
    )
    ids = [
        example.example_id
        for role in result
        for example in role.examples
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("A, B, and C panel identities overlap")
    return result


def build_complete_response_axis() -> tuple[AtomResponseAxisEntry, ...]:
    """Build all 38,216 physical occurrences in exact execution order."""

    axis: list[AtomResponseAxisEntry] = [
        AtomResponseAxisEntry(column, 0) for column in range(HIDDEN_WIDTH)
    ]
    attention_start = HIDDEN_WIDTH
    mlp_start = HIDDEN_WIDTH + GEMMA3_LAYER_COUNT * ATTENTION_HEAD_COUNT
    for layer in range(GEMMA3_LAYER_COUNT):
        axis.extend(
            AtomResponseAxisEntry(
                attention_start + layer * ATTENTION_HEAD_COUNT + head,
                1 + 2 * layer,
            )
            for head in range(ATTENTION_HEAD_COUNT)
        )
        axis.extend(
            AtomResponseAxisEntry(
                mlp_start + layer * MLP_CHANNEL_COUNT + channel,
                2 + 2 * layer,
            )
            for channel in range(MLP_CHANNEL_COUNT)
        )
    axis.extend(
        AtomResponseAxisEntry(column, 38) for column in range(HIDDEN_WIDTH)
    )
    result = tuple(axis)
    if len(result) != RESPONSE_OCCURRENCE_COUNT:
        raise AssertionError("complete response occurrence count drifted")
    if len({item.global_atom_ordinal for item in result}) != 37_576:
        raise AssertionError("physical atom coverage drifted")
    return result


def _tokenizer_tensor(encoded: object, name: str) -> Tensor:
    value = encoded.get(name) if isinstance(encoded, Mapping) else getattr(
        encoded,
        name,
        None,
    )
    if not isinstance(value, Tensor):
        raise TypeError(f"tokenizer output omitted Tensor {name!r}")
    return value


def build_correct_choice_nll_batch(
    tokenizer: object,
    example: ForcedChoiceExample,
    *,
    device: torch.device,
    maximum_prompt_tokens: int = 128,
) -> CalibrationBatch:
    """Create one prompt with exactly one supervised next-token target."""

    if not isinstance(example, ForcedChoiceExample):
        raise TypeError("example must be a ForcedChoiceExample")
    if not callable(tokenizer) or not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("tokenizer lacks callable and encode interfaces")
    if type(maximum_prompt_tokens) is not int or maximum_prompt_tokens < 2:
        raise ValueError("maximum_prompt_tokens must be at least two")
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("tokenizer must define a padding or EOS token")
        setattr(tokenizer, "pad_token", eos_token)
    if hasattr(tokenizer, "padding_side"):
        setattr(tokenizer, "padding_side", "right")
    encoded = tokenizer(
        [example.prompt],
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=maximum_prompt_tokens,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = _tokenizer_tensor(encoded, "input_ids")
    attention_mask = _tokenizer_tensor(encoded, "attention_mask").to(
        dtype=torch.bool
    )
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape != attention_mask.shape
        or not bool(attention_mask.any())
    ):
        raise ValueError("tokenized prompt has an invalid batch shape")
    correct = example.choices[example.correct_choice]
    choice_ids = tokenizer.encode(" " + correct, add_special_tokens=False)
    if not isinstance(choice_ids, list) or len(choice_ids) != 1:
        raise ValueError(
            "the correct forced choice must be exactly one leading-space token"
        )
    positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    final_position = positions.masked_fill(~attention_mask, -1).amax(dim=1)
    if int(final_position.item()) < 0:
        raise ValueError("forced-choice prompt is empty")
    targets = torch.full_like(input_ids, -100)
    valid_positions = torch.zeros_like(attention_mask)
    row = int(final_position.item())
    targets[0, row] = int(choice_ids[0])
    valid_positions[0, row] = True
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids.to(device=device),
            "attention_mask": attention_mask.to(device=device),
        },
        targets=targets.to(device=device),
        valid_positions=valid_positions.to(device=device),
        example_ids=(example.example_id,),
    )


def response_batch_to_capture_row(
    example: ForcedChoiceExample,
    response: PartsFirstAtomResponseBatch,
    *,
    axis: Sequence[AtomResponseAxisEntry],
) -> tuple[CaptureRow, bool]:
    """Flatten one authenticated 38-block batch without crossing boundaries."""

    if not isinstance(response, PartsFirstAtomResponseBatch):
        raise TypeError("response must be a PartsFirstAtomResponseBatch")
    if response.example_id != example.example_id or len(response.row_keys) != 1:
        raise ValueError("response batch is not one row for the requested example")
    coordinates: list[tuple[int, int]] = []
    responses: list[float] = []
    observer_passed = True
    for block in response.blocks:
        if block.part_responses.shape[0] != 1:
            raise ValueError("response block is not singleton-row")
        coordinates.extend(
            (ordinal, block.event_index)
            for ordinal in block.physical_atom_ordinals
        )
        responses.extend(float(value) for value in block.part_responses[0].tolist())
        observer_passed = bool(
            observer_passed
            and block.receipt.operator_closure_passed
            and block.receipt.response_closure_passed
        )
    expected = tuple(
        (item.global_atom_ordinal, item.execution_ordinal) for item in axis
    )
    if tuple(coordinates) != expected:
        raise ValueError("captured block coordinates differ from the complete axis")
    # One global path chart is frozen on the contextual terminal normalized
    # hidden state.  Unlike token embeddings, this state has already bound the
    # sequence/KV/mask/position/window effects and responds to earlier gates.
    chart = response.blocks[-1].context_values[0]
    row = CaptureRow(
        row_id=example.example_id,
        family_id=example.family_id,
        chart_coordinates=tuple(float(value) for value in chart.tolist()),
        target_score=float(response.loss),
        atom_responses=tuple(responses),
    )
    return row, observer_passed


class _ModelLoader(Protocol):
    def __call__(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: Path,
        device: torch.device,
        dtype: str,
        local_files_only: bool,
    ) -> tuple[object, nn.Module]: ...


class _ExactLiftPrechecker(Protocol):
    def __call__(
        self,
        adapter: Gemma3CausalLMAdapter,
        first_batch: CalibrationBatch,
    ) -> ExactLiftPrecheckReceipt: ...


def _default_model_loader() -> _ModelLoader:
    # Lazy imports are part of the preflight contract.
    from .gemma3_experiment import load_gemma3

    return load_gemma3


def _resolve_cache_dir(value: Path | None) -> Path:
    # Path validation is also lazy so importing this module performs no
    # environment or repository discovery.
    from .gemma3_experiment import resolve_gemma3_huggingface_paths

    return resolve_gemma3_huggingface_paths(value)["hub_cache"]


def _extract_logits(output: object) -> Tensor:
    value = getattr(output, "logits", None)
    if isinstance(value, Tensor):
        return value
    if isinstance(output, Mapping):
        value = output.get("logits")
        if isinstance(value, Tensor):
            return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("model output omitted logits")


def native_gates_for_frozen_artifact(
    artifact: FrozenSupermodeArtifact,
    *,
    alpha: float,
) -> tuple[FrozenNativePartGate, ...]:
    """Translate the four global-ordinal stages to native local part gates."""

    if not isinstance(artifact, FrozenSupermodeArtifact):
        raise TypeError("artifact must be a FrozenSupermodeArtifact")
    if not isinstance(alpha, float) or not math.isfinite(alpha):
        raise ValueError("alpha must be a finite exact float")
    catalog = build_gemma3_parts_catalog()
    result: list[FrozenNativePartGate] = []
    for stage in artifact.candidate.stages:
        local: list[int] = []
        for ordinal in stage.atom_ordinals:
            atom = catalog.atom_id_from_global_ordinal(ordinal)
            if atom.family is not stage.family:
                raise ValueError("frozen stage family and atom catalog disagree")
            local.append(atom.part_ordinal)
        result.append(
            FrozenNativePartGate(
                gate_id=(
                    f"gate.{artifact.candidate_sha256[:16]}."
                    f"event.{stage.execution_ordinal}"
                ),
                family=stage.family,
                boundary=stage.boundary,
                part_ordinals=tuple(sorted(set(local))),
                alpha=alpha,
                chart_id=artifact.candidate.chart_id,
            )
        )
    gates = tuple(result)
    expected_families = (
        PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
        PhysicalAtomFamily.ATTENTION_QO_HEAD,
        PhysicalAtomFamily.MLP_CHANNEL,
        PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
    )
    if (
        len(gates) != 4
        or tuple(gate.family for gate in gates) != expected_families
        or tuple(gate.boundary.execution_ordinal for gate in gates)
        != tuple(sorted(gate.boundary.execution_ordinal for gate in gates))
    ):
        raise ValueError("selected artifact is not a four-stage ordered path")
    return gates


@dataclass(slots=True)
class PartsFirstLiveModelPilot:
    """Loaded batch-one runtime shared by capture and the one C candidate."""

    tokenizer: object
    model: nn.Module
    adapter: Gemma3CausalLMAdapter
    binding: Gemma3PartsFirstCaptureBinding
    exact_lift_precheck: ExactLiftPrecheckReceipt
    panel_roles: tuple[PilotPanelRole, ...]
    batches_by_example_id: dict[str, CalibrationBatch]
    allow_test_nonproduction_gate_shape: bool = False
    _capture_bundle: PartsFirstCaptureBundle | None = None

    @classmethod
    def load(
        cls,
        config: PartsFirstModelRunnerConfig = PartsFirstModelRunnerConfig(),
        *,
        model_loader: _ModelLoader | None = None,
        exact_lift_prechecker: _ExactLiftPrechecker | None = None,
    ) -> PartsFirstLiveModelPilot:
        panel = load_pinned_parts_first_panel(config.panel_path)
        roles = split_parts_first_panel(panel)
        loader = _default_model_loader() if model_loader is None else model_loader
        device = torch.device("cpu")
        tokenizer, model = loader(
            model_id=MODEL_ID,
            revision=PINNED_MODEL_REVISION,
            cache_dir=_resolve_cache_dir(config.cache_dir),
            device=device,
            dtype="float32",
            local_files_only=config.local_files_only,
        )
        if not isinstance(model, nn.Module):
            raise TypeError("model loader did not return a torch module")
        model.eval().requires_grad_(False)
        adapter = Gemma3CausalLMAdapter(model)
        binding = bind_parts_first_gemma_adapter(adapter)
        if (
            binding.source_model_fingerprint != SOURCE_MODEL_FINGERPRINT_SHA256
            or binding.execution_fingerprint
            != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ):
            raise ValueError("loaded source differs from the pinned pilot")
        batches = {
            example.example_id: build_correct_choice_nll_batch(
                tokenizer,
                example,
                device=device,
                maximum_prompt_tokens=config.maximum_prompt_tokens,
            )
            for role in roles
            for example in role.examples
        }
        prechecker = (
            run_exact_lift_precheck
            if exact_lift_prechecker is None
            else exact_lift_prechecker
        )
        first_a_example = roles[0].examples[0]
        exact_lift = prechecker(
            adapter,
            batches[first_a_example.example_id],
        )
        if (
            not isinstance(exact_lift, ExactLiftPrecheckReceipt)
            or not exact_lift.passed
        ):
            raise RuntimeError("exact-lift precheck did not pass")
        return cls(
            tokenizer=tokenizer,
            model=model,
            adapter=adapter,
            binding=binding,
            exact_lift_precheck=exact_lift,
            panel_roles=roles,
            batches_by_example_id=batches,
        )

    @property
    def capture_bundle(self) -> PartsFirstCaptureBundle:
        if self._capture_bundle is None:
            raise RuntimeError("capture has not been run")
        return self._capture_bundle

    def capture(self) -> PartsFirstCaptureBundle:
        """Capture 60 independent objective rows and close the exact axis."""

        if self._capture_bundle is not None:
            return self._capture_bundle
        axis = build_complete_response_axis()
        model_before = self.adapter.model_fingerprint()
        execution_before = self.adapter.execution_fingerprint()
        if (
            model_before != self.binding.source_model_fingerprint
            or execution_before != self.binding.execution_fingerprint
        ):
            raise RuntimeError("source changed after the capture binding was frozen")
        captured_roles: list[CaptureRole] = []
        all_observers_passed = True
        for panel_role in self.panel_roles:
            captured_rows: list[CaptureRow] = []
            role_observer_passed = True
            for example in panel_role.examples:
                batch = self.batches_by_example_id[example.example_id]
                stream = iter_parts_first_atom_response_batches(
                    self.binding,
                    (batch,),
                    score_objective=CausalLanguageModelNLL(),
                    accumulation_dtype=torch.float64,
                )
                materialized = tuple(stream)
                if len(materialized) != 1:
                    raise RuntimeError(
                        "one example did not yield exactly one response batch"
                    )
                row, passed = response_batch_to_capture_row(
                    example,
                    materialized[0],
                    axis=axis,
                )
                captured_rows.append(row)
                role_observer_passed = bool(role_observer_passed and passed)
            captured_roles.append(
                CaptureRole(
                    role=panel_role.role,
                    rows=tuple(captured_rows),
                    exact_observer_passed=role_observer_passed,
                )
            )
            all_observers_passed = bool(
                all_observers_passed and role_observer_passed
            )
        model_after = self.adapter.model_fingerprint()
        execution_after = self.adapter.execution_fingerprint()
        unchanged = bool(
            model_after == model_before and execution_after == execution_before
        )
        protocol = build_parts_first_supermode_protocol()
        catalog = build_gemma3_parts_catalog()
        bundle = PartsFirstCaptureBundle(
            source_kind="model_capture",
            model_id=MODEL_ID,
            source_model_fingerprint_sha256=model_before,
            source_execution_fingerprint_sha256=execution_before,
            score_objective_id=MODEL_SCORE_OBJECTIVE_ID,
            chart_surface_id=MODEL_CHART_SURFACE_ID,
            protocol_sha256=str(protocol["protocol_sha256"]),
            atom_universe_sha256=catalog.catalog_sha256,
            response_axis=axis,
            roles=tuple(captured_roles),
            exact_lift_receipt=ExactLiftCaptureReceipt(
                model_facing=True,
                exact_39_event_order_passed=self.exact_lift_precheck.passed,
                all_37576_physical_atoms_closed=(
                    len({item.global_atom_ordinal for item in axis}) == 37_576
                ),
                all_shared_substrate_slices_closed=bool(
                    all_observers_passed and self.exact_lift_precheck.passed
                ),
                tied_lexical_storage_identity_passed=True,
                source_state_unchanged=unchanged,
                exact_control_evidence_sha256=(
                    self.exact_lift_precheck.receipt_sha256
                ),
            ),
        )
        self._capture_bundle = bundle
        return bundle

    def _score_batch(
        self,
        batch: CalibrationBatch,
        *,
        gates: tuple[FrozenNativePartGate, ...] | None,
    ) -> tuple[float, tuple[float, ...], NativePartGateExecutionReceipt | None]:
        backbone = getattr(self.model, "model", None)
        terminal_norm = getattr(backbone, "norm", None)
        if not isinstance(terminal_norm, nn.Module):
            raise TypeError("model omitted its terminal RMSNorm module")
        target_positions = batch.valid_positions.nonzero(as_tuple=False)
        if tuple(target_positions.shape) != (1, 2):
            raise ValueError("native C scoring requires one objective row")
        batch_ordinal = int(target_positions[0, 0])
        query_ordinal = int(target_positions[0, 1])
        captured_terminal_chart: list[Tensor] = []

        def forward() -> tuple[float, tuple[float, ...]]:
            def terminal_chart_hook(
                _module: nn.Module,
                _arguments: tuple[object, ...],
                output: object,
            ) -> None:
                if not isinstance(output, Tensor) or output.ndim != 3:
                    raise TypeError("terminal RMSNorm hook emitted an invalid tensor")
                captured_terminal_chart.append(
                    output[batch_ordinal, query_ordinal].detach().to(
                        device="cpu",
                        dtype=torch.float64,
                    )
                )

            handle = terminal_norm.register_forward_hook(terminal_chart_hook)
            try:
                with torch.no_grad():
                    output = self.model(**dict(batch.model_inputs))
                    logits = _extract_logits(output)
                    target = batch.targets[batch_ordinal, query_ordinal].to(
                        device=logits.device
                    )
                    score = F.cross_entropy(
                        logits[batch_ordinal, query_ordinal]
                        .to(dtype=torch.float32)
                        .unsqueeze(0),
                        target.unsqueeze(0),
                        reduction="sum",
                    )
            finally:
                handle.remove()
            if len(captured_terminal_chart) != 1:
                raise RuntimeError("terminal RMSNorm did not execute exactly once")
            return float(score.item()), tuple(
                float(value) for value in captured_terminal_chart[0].tolist()
            )

        if gates is None:
            score, chart = forward()
            return score, chart, None
        row_mask = batch.valid_positions.to(device=next(self.model.parameters()).device)
        result, receipt = run_with_native_part_gates(
            self.model,
            gates,
            {gate.gate_id: row_mask for gate in gates},
            forward,
            allow_test_nonproduction_shape=self.allow_test_nonproduction_gate_shape,
        )
        return result[0], result[1], receipt

    def _first_order_delta(
        self,
        artifact: FrozenSupermodeArtifact,
        row: CaptureRow,
        *,
        alpha: float,
    ) -> float:
        axis = self.capture_bundle.response_axis
        index = {
            (item.global_atom_ordinal, item.execution_ordinal): ordinal
            for ordinal, item in enumerate(axis)
        }
        directional = 0.0
        for stage in artifact.candidate.stages:
            directional += sum(
                row.atom_responses[index[(atom, stage.execution_ordinal)]]
                for atom in stage.atom_ordinals
            )
        # This is a scalar objective directional derivative for simultaneous
        # gates.  It is not used as a cross-depth generator representation.
        return (alpha - 1.0) * directional

    def run_finite_intervention(
        self,
        artifact: FrozenSupermodeArtifact,
        role: CaptureRole,
        atlas: FrozenChartAtlas,
        config: PartsFirstPilotConfig,
    ) -> FiniteInterventionEvidence:
        """Run alpha-one parity and one finite candidate over fresh C rows."""

        if role.role != "C_intervention":
            raise ValueError("native intervention runner accepts C only")
        bundle = self.capture_bundle
        source_model_before = self.adapter.model_fingerprint()
        source_execution_before = self.adapter.execution_fingerprint()
        if (
            source_model_before != bundle.source_model_fingerprint_sha256
            or source_execution_before
            != bundle.source_execution_fingerprint_sha256
        ):
            raise RuntimeError("source drifted before the finite intervention")
        baseline = tuple(row.target_score for row in role.rows)
        baseline_charts = tuple(
            atlas.chart_id(row.chart_coordinates) for row in role.rows
        )
        alpha_one_gates = native_gates_for_frozen_artifact(
            artifact,
            alpha=1.0,
        )
        finite_gates = native_gates_for_frozen_artifact(
            artifact,
            alpha=float(config.gate_alpha),
        )
        intervened: list[float] = []
        online_charts: list[str] = []
        predicted: list[float] = []
        noop_errors: list[float] = []
        receipts: list[tuple[str, str, NativePartGateExecutionReceipt]] = []
        for row, baseline_score, baseline_chart in zip(
            role.rows,
            baseline,
            baseline_charts,
            strict=True,
        ):
            batch = self.batches_by_example_id[row.row_id]
            selected = baseline_chart == artifact.candidate.chart_id
            fresh_baseline, fresh_baseline_chart, _ = self._score_batch(
                batch,
                gates=None,
            )
            if atlas.chart_id(fresh_baseline_chart) != baseline_chart:
                raise RuntimeError("fresh C baseline changed frozen chart routing")
            noop_errors.append(abs(fresh_baseline - baseline_score))
            if selected:
                noop, _noop_chart, noop_receipt = self._score_batch(
                    batch,
                    gates=alpha_one_gates,
                )
                finite, finite_chart, finite_receipt = self._score_batch(
                    batch,
                    gates=finite_gates,
                )
                assert noop_receipt is not None and finite_receipt is not None
                receipts.extend(
                    (
                        (row.row_id, "alpha_one", noop_receipt),
                        (row.row_id, "finite", finite_receipt),
                    )
                )
                online = atlas.chart_id(finite_chart)
                prediction = self._first_order_delta(
                    artifact,
                    row,
                    alpha=config.gate_alpha,
                )
                noop_errors.append(abs(noop - fresh_baseline))
            else:
                # The frozen chart mask is false: execute the finite condition
                # without installing gates, so later arrays retain one fresh
                # native row per C example and the row remains unchanged.
                finite, finite_chart, _ = self._score_batch(batch, gates=None)
                online = atlas.chart_id(finite_chart)
                prediction = 0.0
            intervened.append(finite)
            online_charts.append(online)
            predicted.append(prediction)
        if not receipts:
            raise RuntimeError("selected candidate had no native C chart support")
        source_model_after = self.adapter.model_fingerprint()
        source_execution_after = self.adapter.execution_fingerprint()
        if (
            source_model_after != source_model_before
            or source_execution_after != source_execution_before
        ):
            raise RuntimeError("native gate intervention changed source state")
        return FiniteInterventionEvidence(
            candidate_sha256=artifact.candidate_sha256,
            c_row_ids_sha256=role.row_ids_sha256,
            gate_alpha=config.gate_alpha,
            baseline_scores=baseline,
            intervened_scores=tuple(intervened),
            first_order_predicted_score_deltas=tuple(predicted),
            baseline_chart_ids=baseline_charts,
            online_chart_ids_after=tuple(online_charts),
            native_noop_max_abs_error=max(noop_errors),
            finite_intervention_candidate_count=1,
            finite_intervention_forward_count=len(role.rows),
            executor_kind="live_model_bridge",
            native_gate_execution_receipt_sha256=_receipt_set_sha256(receipts),
            source_model_fingerprint_before=source_model_before,
            source_model_fingerprint_after=source_model_after,
            source_execution_fingerprint_before=source_execution_before,
            source_execution_fingerprint_after=source_execution_after,
        )


def build_model_runner_preflight(
    config: PartsFirstModelRunnerConfig = PartsFirstModelRunnerConfig(),
) -> dict[str, object]:
    """Validate the frozen panel/axis without importing or loading Gemma."""

    panel = load_pinned_parts_first_panel(config.panel_path)
    roles = split_parts_first_panel(panel)
    axis = build_complete_response_axis()
    return {
        "schema": "fisher_graph.gemma3_parts_first.model_runner_preflight.v1",
        "ready_to_load": True,
        "model_loaded": False,
        "transformers_imported_by_runner": False,
        "model_id": MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_dtype": "float32",
        "device": "cpu",
        "model_batch_size": 1,
        "local_files_only": config.local_files_only,
        "global_path_chart_surface_id": GLOBAL_PATH_CHART_SURFACE_ID,
        "global_path_chart_scope": "single_global_baseline_contextual_chart",
        "per_stage_local_charts_claimed": False,
        "panel_path": config.panel_path.as_posix(),
        "panel_file_sha256": panel.file_sha256,
        "family_count": FAMILY_COUNT,
        "role_example_counts": {
            role.role: len(role.examples) for role in roles
        },
        "role_family_counts": {
            role.role: dict(sorted(Counter(
                example.family_id for example in role.examples
            ).items()))
            for role in roles
        },
        "score_objective_id": MODEL_SCORE_OBJECTIVE_ID,
        "objective_vocabulary": "full_262144_row_language_model_head",
        "response_occurrence_count": len(axis),
        "physical_atom_count": len(
            {item.global_atom_ordinal for item in axis}
        ),
        "event_count": 39,
        "atom_emitting_event_count": 38,
        "C_candidate_count_if_selected": 1,
        "compression_claimed": False,
        "pilot_only": True,
    }


def run_parts_first_model_pilot(
    *,
    runner_config: PartsFirstModelRunnerConfig = PartsFirstModelRunnerConfig(),
    pilot_config: PartsFirstPilotConfig = PartsFirstPilotConfig(),
    model_loader: _ModelLoader | None = None,
    exact_lift_prechecker: _ExactLiftPrechecker | None = None,
    write_capture: bool = True,
) -> tuple[PartsFirstCaptureBundle, PartsFirstPilotReport]:
    """Load, capture A/B/C, and run the frozen A->B->one-C protocol."""

    runtime = PartsFirstLiveModelPilot.load(
        runner_config,
        model_loader=model_loader,
        exact_lift_prechecker=exact_lift_prechecker,
    )
    bundle = runtime.capture()
    if write_capture:
        write_parts_first_capture_bundle(bundle, runner_config.capture_output)
    report = run_gemma3_parts_first_supermode_experiment(
        captures=bundle,
        intervention_runner=runtime.run_finite_intervention,
        config=pilot_config,
        output=runner_config.report_output,
    )
    if not isinstance(report, PartsFirstPilotReport):
        raise RuntimeError("model pilot unexpectedly returned a preflight")
    return bundle, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--panel", type=Path, default=PINNED_PANEL_PATH)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow network checkpoint download instead of requiring local cache",
    )
    parser.add_argument("--maximum-prompt-tokens", type=int, default=128)
    parser.add_argument("--capture-output", type=Path, default=DEFAULT_CAPTURE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runner_config = PartsFirstModelRunnerConfig(
        panel_path=args.panel,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
        capture_output=args.capture_output,
        report_output=args.output,
    )
    if args.preflight_only:
        print(
            json.dumps(
                build_model_runner_preflight(runner_config),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    bundle, report = run_parts_first_model_pilot(runner_config=runner_config)
    print(
        json.dumps(
            {
                "capture_sha256": bundle.capture_sha256,
                "capture_output": runner_config.capture_output.as_posix(),
                "report_sha256": report.report_sha256,
                "report_output": runner_config.report_output.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
