"""Train and gate a full-width source-free replacement for one Gemma layer.

This rung intentionally performs no modal truncation.  It isolates the
generator question by running:

```
native prefix -> full-width student -> native suffix -> LM head
```

Calibration A owns activation-Fisher estimation and every optimizer update.
Calibration B is a single pass/fail lock.  Validation is not tokenized unless
the frozen causal student passes B; test is always parse-and-hash-only.  A
storage-matched control trains the same mini-transformer with every attention
branch output zeroed, making the value of cross-position computation
inspectable without changing the parameter shapes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import (
    Gemma3CausalLMAdapter,
    ModelAdapter,
    SequenceContext,
    SequenceInputOrigin,
)
from .compiler.calibration import CalibrationBatch
from .full_width_single_layer_executor import (
    FullWidthSingleLayerExecutor,
    FullWidthSingleLayerExecutorConfig,
)
from .gemma3_ablation_experiment import (
    _FrozenModelTensorGuard,
    _is_sha256,
    _update_payload_digest,
    _validate_model_metadata,
)
from .gemma3_codimension_rotation_experiment import (
    _file_sha256,
    _semantic_numeric_equal,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_gated_executor_experiment import (
    _BoundaryBatch,
    _aggregate_direct_examples,
    _materialize_split,
    _source_block_macs,
    _source_block_static,
)
from .gemma3_rotated_span_executor_experiment import (
    _TrainingBatch,
    _aggregate_behavior_with_kl,
    _behavior_examples_with_kl,
    _behavior_gates,
    _collect_training_batches,
    _direct_rows,
    _graph_structural_probes,
    _run_native_stack,
    _run_replacement_with_call_audit,
    _run_suffix_from_boundary,
    _tensor_sha256,
)
from .gemma3_stability_experiment import (
    Gemma3PromptSplits,
    _library_versions,
    _ordered_prompt_hash_digest,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .modal_ablation import _example_ids


DEFAULT_LAYER_INDEX = 4
DEFAULT_HIDDEN_WIDTH = 128
DEFAULT_EXECUTOR_LAYERS = 2
DEFAULT_HEAD_COUNT = 4
DEFAULT_FEED_FORWARD_WIDTH = 512
DEFAULT_FISHER_FLOOR = 1e-4
DEFAULT_RIDGE_SCALE_FLOOR = 1e-4
DEFAULT_LOCAL_WARMUP_STEPS = 400
DEFAULT_TRAIN_STEPS = 2_800
DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE = 8
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_LOCAL_MSE_WEIGHT = 0.10
DEFAULT_LOCAL_FISHER_WEIGHT = 0.10
DEFAULT_GROUND_TRUTH_WEIGHT = 0.25
DEFAULT_TEACHER_KL_WEIGHT = 4.0

DEFAULT_NLL_ATOL = 0.05
DEFAULT_TOP1_MIN = 0.95
DEFAULT_TEACHER_KL_MAX = 0.05
DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX = 0.10
DEFAULT_PER_PROMPT_P10_TOP1_MIN = 0.90
DEFAULT_MAX_STORED_COEFFICIENT_RATIO = 0.75
DEFAULT_MAX_ANALYTIC_MAC_RATIO = 0.75
DEFAULT_BLOCK_DELTA_NRMSE_MAX = 0.02
DEFAULT_BLOCK_DELTA_COSINE_MIN = 0.999
DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS = 256
DEFAULT_MINIMUM_HELDOUT_PROMPTS = 64
DEFAULT_MINIMUM_FISHER_ROWS = 10_000
DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS = 50_000
DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS = 5_000
DEFAULT_MINIMUM_LENGTH_BUCKETS = 4

PROMPT_STATUS = (
    "full_width_single_layer_fresh_a_b_validation_test_hash_only"
)
FAMILY_STATUS = "full_width_single_layer_family_disjoint_roles"
_FAMILY_SCHEMA = "fisher_graph.gemma3_prompt_family_manifest"
_ARTIFACT_SCHEMA = "fisher_graph.gemma3_full_width_single_layer_executor"
_ARTIFACT_FORMAT_VERSION = 2
_PAYLOAD_DOMAIN = (
    b"fisher_graph.gemma3_full_width_single_layer_executor_payload.v2\0"
)
_REPORT_DOMAIN = (
    b"fisher_graph.gemma3_full_width_single_layer_executor_report.v2\0"
)
_SPLIT_NAMES = (
    "calibration_a",
    "calibration_b",
    "validation",
    "test",
)


@dataclass(frozen=True, slots=True)
class PromptFamilyManifest:
    """Family ownership for every prompt in the four compiler roles."""

    calibration_a: tuple[str, ...]
    calibration_b: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    scientific_status: str

    def __post_init__(self) -> None:
        if self.scientific_status != FAMILY_STATUS:
            raise ValueError("prompt family manifest status is not canonical")
        role_sets: dict[str, set[str]] = {}
        for name in _SPLIT_NAMES:
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in values
                )
            ):
                raise ValueError(
                    f"{name} family ids must be nonempty strings"
                )
            role_sets[name] = set(values)
        for index, left in enumerate(_SPLIT_NAMES):
            for right in _SPLIT_NAMES[index + 1 :]:
                if role_sets[left] & role_sets[right]:
                    raise ValueError(
                        "prompt family ids must be disjoint across roles"
                    )

    def validate_counts(self, prompts: Gemma3PromptSplits) -> None:
        for name in _SPLIT_NAMES:
            if len(getattr(self, name)) != len(getattr(prompts, name)):
                raise ValueError(
                    f"{name} prompt and family counts do not match"
                )

    def metadata(self) -> dict[str, object]:
        return {
            "scientific_status": self.scientific_status,
            "counts": {
                name: len(getattr(self, name))
                for name in _SPLIT_NAMES
            },
            "unique_family_counts": {
                name: len(set(getattr(self, name)))
                for name in _SPLIT_NAMES
            },
            "ordered_family_sha256": {
                name: hashlib.sha256(
                    json.dumps(
                        getattr(self, name),
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                for name in _SPLIT_NAMES
            },
            "cross_role_overlap_count": 0,
        }


def load_prompt_family_manifest(
    path: Path | str,
    *,
    prompts: Gemma3PromptSplits,
) -> PromptFamilyManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema",
        "format_version",
        "scientific_status",
        *_SPLIT_NAMES,
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("prompt family manifest fields are invalid")
    if (
        raw["schema"] != _FAMILY_SCHEMA
        or raw["format_version"] != 1
    ):
        raise ValueError("unsupported prompt family manifest")
    values: dict[str, tuple[str, ...]] = {}
    for name in _SPLIT_NAMES:
        role = raw[name]
        if not isinstance(role, list):
            raise TypeError(f"{name} family ids must be a JSON list")
        values[name] = tuple(role)
    manifest = PromptFamilyManifest(
        calibration_a=values["calibration_a"],
        calibration_b=values["calibration_b"],
        validation=values["validation"],
        test=values["test"],
        scientific_status=str(raw["scientific_status"]),
    )
    manifest.validate_counts(prompts)
    return manifest


def default_gemma3_full_width_single_layer_output(
    model_id: str = DEFAULT_MODEL_ID,
    layer_index: int = DEFAULT_LAYER_INDEX,
) -> Path:
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("._-")
    return (
        Path(".local-runs")
        / (slug or "gemma3-model")
        / f"layer-{layer_index}-full-width-single-layer-executor.pt"
    )


def _finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return result


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_REPORT_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()


def _single_prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(
        json.dumps(
            [prompt],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _tracked_prompt_exclusion_audit(
    prompts: Gemma3PromptSplits,
    *,
    prompt_path: Path,
) -> dict[str, object]:
    """Reject exact prompt reuse from tracked Gemma experiment fixtures."""

    current = prompt_path.resolve()
    fixture_files = []
    excluded: set[str] = set()
    examples = Path(__file__).resolve().parents[2] / "examples"
    if not examples.is_dir():
        raise FileNotFoundError(
            "tracked Gemma prompt fixtures are unavailable; prompt "
            "exclusion cannot be authenticated"
        )
    if examples.is_dir():
        for path in sorted(examples.glob("gemma3*prompts*.json")):
            if path.resolve() == current:
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, Mapping):
                continue
            count_before = len(excluded)
            for name in _SPLIT_NAMES:
                values = raw.get(name)
                if not isinstance(values, list):
                    continue
                excluded.update(
                    _single_prompt_sha256(value.strip())
                    for value in values
                    if isinstance(value, str) and value.strip()
                )
            if len(excluded) > count_before:
                fixture_files.append(
                    {
                        "path": str(path),
                        "file_sha256": _file_sha256(path),
                    }
                )
        for path in sorted(examples.glob("gemma3*prompts*.txt")):
            if path.resolve() == current:
                continue
            count_before = len(excluded)
            try:
                values = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            excluded.update(
                _single_prompt_sha256(value.strip())
                for value in values
                if value.strip()
            )
            if len(excluded) > count_before:
                fixture_files.append(
                    {
                        "path": str(path),
                        "file_sha256": _file_sha256(path),
                    }
                )
    fresh_metadata = prompts.metadata()
    per_prompt = fresh_metadata["per_prompt_sha256"]
    assert isinstance(per_prompt, Mapping)
    fresh = {
        digest
        for values in per_prompt.values()
        for digest in values  # type: ignore[union-attr]
    }
    overlap = fresh & excluded
    if overlap:
        raise ValueError(
            "single-layer prompts overlap tracked prior Gemma fixtures"
        )
    return {
        "fresh_prompt_count": len(fresh),
        "tracked_excluded_prompt_count": len(excluded),
        "tracked_fixture_files": fixture_files,
        "overlap_count": 0,
        "verified_before_model_load_or_tokenization": True,
        "scope": (
            "exact_text_hashes_in_tracked_example_json_and_text_fixtures; "
            "family manifest owns semantic split disjointness"
        ),
    }


def _require_prompt_protocol(
    prompts: Gemma3PromptSplits,
    *,
    minimum_calibration_a_prompts: int,
    minimum_heldout_prompts: int,
) -> None:
    if prompts.scientific_status != PROMPT_STATUS:
        raise ValueError("single-layer prompt fixture status is not canonical")
    if len(prompts.calibration_a) < minimum_calibration_a_prompts:
        raise ValueError("calibration A prompt count is below the minimum")
    for name in ("calibration_b", "validation", "test"):
        if len(getattr(prompts, name)) < minimum_heldout_prompts:
            raise ValueError(f"{name} prompt count is below the minimum")


def _tokenized_content_hashes(
    stream: Mapping[str, object],
    *,
    split_name: str,
) -> set[str]:
    examples = stream.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{split_name} tokenized stream has no examples")
    hashes = set()
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError(
                f"{split_name} tokenized example is invalid"
            )
        value = example.get("content_sha256")
        if not _is_sha256(value) or value in hashes:
            raise ValueError(
                f"{split_name} tokenized content hashes are invalid"
            )
        hashes.add(value)
    return hashes


def _tokenized_stream_contract(
    stream: Mapping[str, object],
    *,
    split_name: str,
    minimum_supervised_tokens: int,
    minimum_length_buckets: int,
) -> dict[str, object]:
    examples = stream.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{split_name} tokenized stream has no examples")
    supervised = []
    valid = []
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError(
                f"{split_name} tokenized example is invalid"
            )
        supervised_value = example.get("supervised_positions")
        valid_value = example.get("valid_tokens")
        if (
            type(supervised_value) is not int
            or supervised_value <= 0
            or type(valid_value) is not int
            or valid_value <= 0
        ):
            raise ValueError(
                f"{split_name} token counts are invalid"
            )
        supervised.append(supervised_value)
        valid.append(valid_value)
    total_supervised = sum(supervised)
    if total_supervised < minimum_supervised_tokens:
        raise ValueError(
            f"{split_name} supervised-token count is below the minimum"
        )
    bucket_edges = (32, 64, 128)
    populated = {
        (
            "1-32"
            if length <= bucket_edges[0]
            else (
                "33-64"
                if length <= bucket_edges[1]
                else (
                    "65-128"
                    if length <= bucket_edges[2]
                    else "129+"
                )
            )
        )
        for length in valid
    }
    if len(populated) < minimum_length_buckets:
        raise ValueError(
            f"{split_name} does not populate enough length buckets"
        )
    return {
        "sequences": len(examples),
        "supervised_tokens": total_supervised,
        "valid_tokens": sum(valid),
        "minimum_supervised_tokens": minimum_supervised_tokens,
        "populated_length_buckets": tuple(sorted(populated)),
        "minimum_length_buckets": minimum_length_buckets,
        "pre_truncation_length_available": False,
        "truncation_rate_claim": False,
        "passed": True,
    }


def _assert_tokenized_content_disjointness(
    streams: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    hashes = {
        name: _tokenized_content_hashes(stream, split_name=name)
        for name, stream in streams.items()
    }
    overlaps = {}
    names = tuple(hashes)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = hashes[left] & hashes[right]
            overlaps[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise ValueError(
                    "tokenized content overlaps across compiler roles: "
                    f"{left} and {right}"
                )
    return {
        "split_content_counts": {
            name: len(values) for name, values in hashes.items()
        },
        "pairwise_overlap_counts": overlaps,
        "passed": True,
    }


def _require_complete_middle_layer_demand(
    adapter: ModelAdapter,
    training: Sequence[_TrainingBatch],
) -> None:
    for item in training:
        sequence = adapter.prepare_sequence(item.batch.model_inputs)
        if not torch.equal(
            sequence.query_valid_mask,
            sequence.key_valid_mask,
        ):
            raise ValueError(
                "middle-layer replacement requires every valid key row to "
                "be an updated query row"
            )


def _module_storage_pointers(module: nn.Module) -> set[int]:
    pointers = set()
    for value in (*module.parameters(), *module.buffers()):
        if value.numel() > 0:
            pointers.add(value.untyped_storage().data_ptr())
    return pointers


def _assert_source_independence(
    source: nn.Module,
    executors: Mapping[str, nn.Module],
) -> dict[str, object]:
    """Reject object, module, or storage aliasing with the source model."""

    if not isinstance(source, nn.Module) or not executors:
        raise ValueError("source and executor candidates are required")
    source_parameter_ids = {
        id(parameter) for parameter in source.parameters()
    }
    source_module_ids = {id(module) for module in source.modules()}
    source_storage_pointers = _module_storage_pointers(source)
    candidate_audits: dict[str, dict[str, object]] = {}
    for name, executor in executors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("executor candidate names must be nonempty")
        if not isinstance(executor, nn.Module):
            raise TypeError("executor candidates must be modules")
        parameter_aliases = source_parameter_ids & {
            id(parameter) for parameter in executor.parameters()
        }
        module_aliases = source_module_ids & {
            id(module) for module in executor.modules()
        }
        storage_aliases = source_storage_pointers & (
            _module_storage_pointers(executor)
        )
        if parameter_aliases or module_aliases or storage_aliases:
            raise RuntimeError(
                f"{name} aliases source-model objects or tensor storage"
            )
        candidate_audits[name] = {
            "parameter_object_alias_count": 0,
            "module_object_alias_count": 0,
            "tensor_storage_alias_count": 0,
            "passed": True,
        }
    return {
        "scope": (
            "parameter_objects_module_objects_and_parameter_or_buffer_"
            "storage"
        ),
        "source_parameter_count": len(source_parameter_ids),
        "source_module_count": len(source_module_ids),
        "source_tensor_storage_count": len(source_storage_pointers),
        "candidates": candidate_audits,
        "passed": True,
    }


def _attention_visibility_contract(
    adapter: ModelAdapter,
    *,
    layer_id: str,
    maximum_length: int,
) -> dict[str, object]:
    """Bind the global-causal student to an equivalent source visibility."""

    layer = adapter.layer(layer_id)
    attention = layer.attention
    if attention is None:
        raise ValueError("selected layer does not expose attention metadata")
    if attention.kind == "global_causal":
        equivalence = "source_global_causal"
    elif attention.kind == "sliding_causal":
        if (
            attention.window_size is None
            or maximum_length > attention.window_size
        ):
            raise ValueError(
                "global-causal student is not visibility-equivalent to the "
                "selected sliding layer at the requested maximum length"
            )
        equivalence = (
            "global_student_equals_source_sliding_visibility_only_because_"
            "maximum_length_does_not_exceed_window"
        )
    else:
        raise ValueError(
            f"unsupported source attention topology: {attention.kind!r}"
        )
    return {
        "source_attention_kind": attention.kind,
        "source_window_size": attention.window_size,
        "student_attention_kind": "global_causal",
        "maximum_tokenized_length": maximum_length,
        "visibility_equivalence": equivalence,
        "prefill_only": True,
        "decode_or_cache_claim": False,
        "rope_equivalence_claim": False,
        "nonzero_position_offset_claim": False,
        "passed": True,
    }


def _source_accounting_manifest(
    adapter: ModelAdapter,
    *,
    layer_ids: tuple[str, ...],
) -> dict[str, object]:
    """Describe enough source geometry to recompute parameter/MAC ledgers."""

    parameter_entries = []
    linear_weight_entries = []
    seen_parameters = set()
    linear_by_layer: dict[str, int] = {}
    attention_by_layer: dict[str, dict[str, object]] = {}
    for layer_id in layer_ids:
        module = adapter.source_module(layer_id)
        for name, parameter in module.named_parameters():
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            parameter_entries.append(
                {
                    "layer_id": layer_id,
                    "name": name,
                    "shape": tuple(parameter.shape),
                    "numel": parameter.numel(),
                    "element_size": parameter.element_size(),
                }
            )
        seen_linear = set()
        linear_coefficients = 0
        for module_name, child in module.named_modules():
            if isinstance(child, nn.Linear) and id(child) not in seen_linear:
                seen_linear.add(id(child))
                linear_coefficients += child.weight.numel()
                linear_weight_entries.append(
                    {
                        "layer_id": layer_id,
                        "module_name": module_name,
                        "shape": tuple(child.weight.shape),
                        "numel": child.weight.numel(),
                    }
                )
        linear_by_layer[layer_id] = linear_coefficients
        attention = adapter.layer(layer_id).attention
        if attention is None:
            raise ValueError(
                "source accounting requires attention metadata"
            )
        attention_by_layer[layer_id] = {
            "kind": attention.kind,
            "query_heads": attention.query_heads,
            "head_dimension": attention.head_dimension,
            "window_size": attention.window_size,
        }
    body: dict[str, object] = {
        "scope": "exact_selected_source_module_geometry",
        "layer_ids": layer_ids,
        "parameter_entries": parameter_entries,
        "parameter_count": sum(
            int(entry["numel"]) for entry in parameter_entries
        ),
        "parameter_bytes": sum(
            int(entry["numel"]) * int(entry["element_size"])
            for entry in parameter_entries
        ),
        "linear_weight_entries": linear_weight_entries,
        "linear_weight_coefficients_by_layer": linear_by_layer,
        "attention_by_layer": attention_by_layer,
        "sequence_accounting_contract": {
            "phase": "prefill",
            "padding_side": "right",
            "logical_position_domain": "zero_contiguous",
            "query_and_key_valid_masks_equal": True,
            "source_mac_recomputed_from_recorded_valid_lengths": True,
        },
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **body,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _full_width_structural_probes(
    adapter: ModelAdapter,
    executor: FullWidthSingleLayerExecutor,
    training: Sequence[_TrainingBatch],
) -> dict[str, object]:
    """Probe causality plus synthetic right-padding invariance."""

    inherited = _graph_structural_probes(
        adapter,
        executor,  # type: ignore[arg-type]
        training,
    )
    first = training[0]
    valid = first.batch.valid_positions[0]
    selected = valid.nonzero(as_tuple=False).flatten()
    if not selected.numel():
        raise ValueError("structural probe requires a valid token")
    compact_hidden = first.block_input[0, selected].unsqueeze(0)
    compact_positions = (
        adapter.prepare_sequence(first.batch.model_inputs)
        .logical_positions[0, selected]
        .unsqueeze(0)
    )
    compact_mask = torch.ones(
        1,
        compact_hidden.shape[1],
        dtype=torch.bool,
        device=compact_hidden.device,
    )
    origin = SequenceInputOrigin(
        attention_mask_supplied=True,
        position_ids_supplied=True,
        cache_positions_supplied=False,
    )
    compact_sequence = SequenceContext(
        query_valid_mask=compact_mask,
        key_valid_mask=compact_mask,
        logical_positions=compact_positions,
        key_logical_positions=compact_positions,
        cache_positions=None,
        phase="prefill",
        input_origin=origin,
    )
    padding_slots = 2
    last_position = int(compact_positions[0, -1].item())
    padded_positions = torch.cat(
        (
            compact_positions,
            torch.arange(
                last_position + 1,
                last_position + padding_slots + 1,
                dtype=compact_positions.dtype,
                device=compact_positions.device,
            ).unsqueeze(0),
        ),
        dim=1,
    )
    padded_mask = torch.cat(
        (
            compact_mask,
            torch.zeros(
                1,
                padding_slots,
                dtype=torch.bool,
                device=compact_mask.device,
            ),
        ),
        dim=1,
    )
    padded_sequence = SequenceContext(
        query_valid_mask=padded_mask,
        key_valid_mask=padded_mask,
        logical_positions=padded_positions,
        key_logical_positions=padded_positions,
        cache_positions=None,
        phase="prefill",
        input_origin=origin,
    )
    padded_hidden = torch.cat(
        (
            compact_hidden,
            torch.randn(
                1,
                padding_slots,
                compact_hidden.shape[-1],
                dtype=compact_hidden.dtype,
                device=compact_hidden.device,
            ),
        ),
        dim=1,
    )
    perturbed_hidden = padded_hidden.clone()
    perturbed_hidden[:, -padding_slots:] += 1_000_000.0
    with torch.no_grad():
        compact_output = executor(compact_hidden, compact_sequence)
        padded_output = executor(padded_hidden, padded_sequence)
        perturbed_output = executor(
            perturbed_hidden,
            padded_sequence,
        )
    appended_error = float(
        (
            padded_output[:, : compact_hidden.shape[1]]
            - compact_output
        )
        .abs()
        .max()
        .item()
    )
    perturbation_error = float(
        (
            perturbed_output[:, : compact_hidden.shape[1]]
            - padded_output[:, : compact_hidden.shape[1]]
        )
        .abs()
        .max()
        .item()
    )
    tolerance = 1e-5
    synthetic_passed = (
        appended_error <= tolerance
        and perturbation_error <= tolerance
    )
    return {
        **inherited,
        "synthetic_invalid_padding_slots": padding_slots,
        "synthetic_appended_padding_max_valid_error": appended_error,
        "synthetic_invalid_value_perturbation_max_valid_error": (
            perturbation_error
        ),
        "synthetic_padding_tolerance": tolerance,
        "synthetic_padding_passed": synthetic_passed,
        "passed": inherited["passed"] is True and synthetic_passed,
    }


def _activation_fisher(
    adapter: ModelAdapter,
    training: Sequence[_TrainingBatch],
    *,
    plan: object,
    fisher_floor: float,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Estimate a width-pooled empirical activation score-sensitivity matrix."""

    if not training:
        raise ValueError("training batches cannot be empty")
    width = int(training[0].block_output.shape[-1])
    matrix = torch.zeros(width, width, dtype=torch.float64)
    rows = 0
    for item in training:
        sequence = adapter.prepare_sequence(item.batch.model_inputs)
        boundary = item.block_output.detach().clone().requires_grad_(True)
        _, _, logits = _run_suffix_from_boundary(
            adapter,
            item.batch,
            plan=plan,  # type: ignore[arg-type]
            sequence=sequence,
            boundary_output=boundary,
            selected_positions=item.selected_positions,
            full_logits=False,
        )
        if logits is None:
            raise RuntimeError("Fisher suffix logits were not produced")
        targets = item.ground_truth_targets.to(device=logits.device)
        loss = F.cross_entropy(
            logits.float(),
            targets,
            reduction="sum",
        )
        gradient = torch.autograd.grad(
            loss,
            boundary,
            retain_graph=False,
            create_graph=False,
        )[0]
        selected = gradient[item.batch.valid_positions].to(
            device="cpu",
            dtype=torch.float64,
        )
        matrix.add_(selected.T @ selected)
        rows += int(selected.shape[0])
    if rows <= 0:
        raise ValueError("activation Fisher has no valid gradient rows")
    matrix.div_(rows)
    matrix = ((matrix + matrix.T) * 0.5).contiguous()
    diagonal = torch.diagonal(matrix).clone()
    mean = float(diagonal.mean().item())
    if mean <= torch.finfo(torch.float64).tiny:
        raise RuntimeError("activation Fisher diagonal has zero mean")
    normalized = (diagonal / mean).clamp_min(fisher_floor)
    normalized /= normalized.mean()
    eigenvalues = torch.linalg.eigvalsh(matrix)
    trace = float(eigenvalues.sum().item())
    positive = eigenvalues.clamp_min(0)
    descending = positive.flip(0)
    cumulative = descending.cumsum(0)

    def capture_rank(fraction: float) -> int:
        if trace <= torch.finfo(torch.float64).tiny:
            return width
        threshold = fraction * trace
        return int(
            torch.searchsorted(cumulative, threshold).item()
        ) + 1

    return matrix, normalized, {
        "estimator": (
            "width_pooled_empirical_ground_truth_ce_score_sensitivity_"
            "from_selected_supervised_targets"
        ),
        "expected_model_fisher_claim": False,
        "cross_position_blocks_included": False,
        "training_uses_full_matrix": True,
        "rows": rows,
        "width": width,
        "trace": trace,
        "minimum_eigenvalue": float(eigenvalues.min().item()),
        "maximum_eigenvalue": float(eigenvalues.max().item()),
        "rank_for_90_percent_trace": capture_rank(0.90),
        "rank_for_99_percent_trace": capture_rank(0.99),
        "rank_for_99_9_percent_trace": capture_rank(0.999),
        "diagonal_floor_before_renormalization": fisher_floor,
        "normalized_diagonal_min": float(normalized.min().item()),
        "normalized_diagonal_max": float(normalized.max().item()),
        "normalized_diagonal_mean": float(normalized.mean().item()),
        "matrix_sha256": _tensor_sha256(
            matrix,
            domain=b"fisher_graph.full_width_layer.fisher_matrix.v1\0",
        ),
        "normalized_diagonal_sha256": _tensor_sha256(
            normalized,
            domain=b"fisher_graph.full_width_layer.fisher_diagonal.v1\0",
        ),
    }


def _delta_scale(
    training: Sequence[_TrainingBatch],
    *,
    floor: float,
) -> tuple[Tensor, dict[str, object]]:
    rows = []
    for item in training:
        valid = item.batch.valid_positions
        rows.append(
            (
                item.block_output[valid].to(torch.float64)
                - item.block_input[valid].to(torch.float64)
            ).cpu()
        )
    values = torch.cat(rows)
    scale = values.square().mean(dim=0).sqrt().clamp_min(floor)
    return scale, {
        "estimator": "calibration_a_per_coordinate_block_delta_rms",
        "rows": int(values.shape[0]),
        "floor": floor,
        "minimum": float(scale.min().item()),
        "maximum": float(scale.max().item()),
        "mean": float(scale.mean().item()),
        "sha256": _tensor_sha256(
            scale,
            domain=b"fisher_graph.full_width_layer.delta_scale.v1\0",
        ),
    }


def _normalized_fisher_metric(
    matrix: Tensor,
    delta_scale: Tensor,
    *,
    eigenvalue_floor: float,
) -> tuple[Tensor, dict[str, object]]:
    """Return a PSD, mean-eigenvalue-one metric in scaled-delta coordinates."""

    if (
        not isinstance(matrix, Tensor)
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or delta_scale.shape != (matrix.shape[0],)
    ):
        raise ValueError("Fisher metric inputs have incompatible shapes")
    scaled = (
        delta_scale.to(torch.float64).unsqueeze(1)
        * matrix.to(torch.float64)
        * delta_scale.to(torch.float64).unsqueeze(0)
    )
    scaled = (scaled + scaled.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(scaled)
    maximum = max(
        float(eigenvalues.max().item()),
        torch.finfo(torch.float64).tiny,
    )
    floor = eigenvalue_floor * maximum
    floored = eigenvalues.clamp_min(floor)
    metric = (
        (eigenvectors * floored.unsqueeze(0)) @ eigenvectors.T
    )
    metric = (metric + metric.T) * 0.5
    mean_eigenvalue = float(
        torch.diagonal(metric).sum().item() / matrix.shape[0]
    )
    metric /= mean_eigenvalue
    return metric.contiguous(), {
        "coordinate_system": "per_coordinate_delta_rms_standardized",
        "full_quadratic_form_used_in_training": True,
        "eigenvalue_floor_relative_to_maximum": eigenvalue_floor,
        "pre_floor_minimum_eigenvalue": float(eigenvalues.min().item()),
        "pre_floor_maximum_eigenvalue": float(eigenvalues.max().item()),
        "post_normalization_trace": float(
            torch.diagonal(metric).sum().item()
        ),
        "sha256": _tensor_sha256(
            metric,
            domain=b"fisher_graph.full_width_layer.training_metric.v1\0",
        ),
    }


def _make_executor(
    *,
    width: int,
    hidden_width: int,
    executor_layers: int,
    head_count: int,
    feed_forward_width: int,
    causal_edges_enabled: bool,
    seed: int,
    device: torch.device,
) -> FullWidthSingleLayerExecutor:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        executor = FullWidthSingleLayerExecutor(
            FullWidthSingleLayerExecutorConfig(
                residual_width=width,
                hidden_width=hidden_width,
                layer_count=executor_layers,
                head_count=head_count,
                feed_forward_width=feed_forward_width,
                causal_edges_enabled=causal_edges_enabled,
            ),
            dtype=torch.float32,
            device="cpu",
        )
    return executor.to(device=device)


def _local_losses(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    fisher_positions: Tensor,
    *,
    delta_scale: Tensor,
    fisher_metric: Tensor,
) -> tuple[Tensor, Tensor]:
    if (
        fisher_positions.shape != valid.shape
        or fisher_positions.dtype is not torch.bool
        or bool((fisher_positions & ~valid).any())
        or not bool(fisher_positions.any())
    ):
        raise ValueError(
            "Fisher-loss positions must be a nonempty subset of valid rows"
        )
    error = (
        prediction.float() - target.float()
    ) / delta_scale.to(device=prediction.device, dtype=torch.float32)
    raw = error[valid].square().mean()
    selected = error[fisher_positions]
    metric = fisher_metric.to(
        device=prediction.device,
        dtype=torch.float32,
    )
    fisher = torch.einsum(
        "ni,ij,nj->n",
        selected,
        metric,
        selected,
    ).clamp_min(0).mean() / selected.shape[1]
    return raw, fisher


def _fit_executor(
    adapter: ModelAdapter,
    executor: FullWidthSingleLayerExecutor,
    training: Sequence[_TrainingBatch],
    *,
    plan: object,
    delta_scale: Tensor,
    fisher_metric: Tensor,
    local_warmup_steps: int,
    train_steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    local_mse_weight: float,
    local_fisher_weight: float,
    ground_truth_weight: float,
    teacher_kl_weight: float,
) -> dict[str, object]:
    source_parameter_ids = {
        id(parameter) for parameter in adapter.module.parameters()
    }
    parameters = tuple(executor.parameters())
    if not parameters or any(
        id(parameter) in source_parameter_ids for parameter in parameters
    ):
        raise RuntimeError("executor aliases source-model parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    runtime_delta_scale = delta_scale.to(
        device=executor.device,
        dtype=torch.float32,
    )
    runtime_fisher_metric = fisher_metric.to(
        device=executor.device,
        dtype=torch.float32,
    )
    executor.train()
    total_steps = local_warmup_steps + train_steps
    snapshots = []
    minimum_total = math.inf
    first: dict[str, object] | None = None
    last: dict[str, object] | None = None
    for step in range(total_steps):
        item = training[step % len(training)]
        sequence = adapter.prepare_sequence(item.batch.model_inputs)
        source = item.block_input.to(device=executor.device)
        target = item.block_output.to(device=executor.device)
        optimizer.zero_grad(set_to_none=True)
        predicted = executor(source, sequence)
        local_mse, local_fisher = _local_losses(
            predicted,
            target,
            item.batch.valid_positions,
            item.selected_positions,
            delta_scale=runtime_delta_scale,
            fisher_metric=runtime_fisher_metric,
        )
        ground_truth_ce = predicted.new_zeros(())
        teacher_kl = predicted.new_zeros(())
        if step >= local_warmup_steps:
            _, _, logits = _run_suffix_from_boundary(
                adapter,
                item.batch,
                plan=plan,  # type: ignore[arg-type]
                sequence=sequence,
                boundary_output=predicted,
                selected_positions=item.selected_positions,
                full_logits=False,
            )
            if logits is None:
                raise RuntimeError("student suffix logits were not produced")
            student_log = F.log_softmax(logits.float(), dim=-1)
            teacher = item.teacher_logits.to(
                device=logits.device,
                dtype=torch.float32,
            )
            teacher_log = F.log_softmax(teacher, dim=-1)
            ground_truth_ce = F.cross_entropy(
                logits.float(),
                item.ground_truth_targets.to(device=logits.device),
                reduction="mean",
            )
            teacher_kl = F.kl_div(
                student_log,
                teacher_log,
                reduction="batchmean",
                log_target=True,
            )
            total = (
                local_mse_weight * local_mse
                + local_fisher_weight * local_fisher
                + ground_truth_weight * ground_truth_ce
                + teacher_kl_weight * teacher_kl
            )
            phase = "downstream_distillation"
        else:
            total = (
                local_mse_weight * local_mse
                + local_fisher_weight * local_fisher
            )
            phase = "all_row_mse_and_fisher_quadratic_local_warmup"
        if not torch.isfinite(total):
            raise RuntimeError("single-layer executor loss is nonfinite")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if any(
            parameter.grad is not None
            for parameter in adapter.module.parameters()
        ):
            raise RuntimeError("source-model parameter received a gradient")
        optimizer.step()
        row = {
            "step": step + 1,
            "phase": phase,
            "batch_index": step % len(training),
            "local_scale_normalized_mse": float(
                local_mse.detach().item()
            ),
            "local_full_fisher_quadratic": float(
                local_fisher.detach().item()
            ),
            "ground_truth_cross_entropy": float(
                ground_truth_ce.detach().item()
            ),
            "teacher_kl": float(teacher_kl.detach().item()),
            "total_loss": float(total.detach().item()),
            "gradient_norm_before_clip": float(
                gradient_norm.detach().item()
            ),
        }
        if first is None:
            first = row
        last = row
        minimum_total = min(minimum_total, float(row["total_loss"]))
        if (
            step == 0
            or step + 1 == total_steps
            or (step + 1) % max(1, total_steps // 8) == 0
        ):
            snapshots.append(row)
    executor.eval()
    if first is None or last is None:
        raise RuntimeError("executor training performed no updates")
    return {
        "local_warmup_steps": local_warmup_steps,
        "downstream_train_steps": train_steps,
        "total_steps": total_steps,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "local_scale_normalized_mse_weight": local_mse_weight,
        "local_fisher_weight": local_fisher_weight,
        "ground_truth_cross_entropy_weight": ground_truth_weight,
        "teacher_kl_weight": teacher_kl_weight,
        "fixed_update_schedule": True,
        "local_mse_position_scope": "all_valid_rows",
        "local_fisher_quadratic_position_scope": (
            "deterministically_selected_supervised_rows"
        ),
        "checkpoint_selection": "final_fixed_step",
        "early_stopping": False,
        "first_update": first,
        "last_update": last,
        "minimum_observed_total_loss": minimum_total,
        "snapshots": snapshots,
        "source_parameter_gradients_observed": False,
    }


def _split_evaluation(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: object,
    candidates: Mapping[str, FullWidthSingleLayerExecutor],
    include_controls: bool,
) -> dict[str, object]:
    behavior_rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in candidates
    }
    direct_rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in candidates
    }
    if include_controls:
        for name in ("native_boundary_replay", "identity_layer_skip"):
            behavior_rows[name] = []
            direct_rows[name] = []
    sequence_offset = 0
    call_audits: dict[str, dict[str, int]] = {
        name: {
            "executor_calls": 0,
            "source_layer_calls": 0,
        }
        for name in candidates
    }
    prefix_errors: dict[str, list[float]] = {
        name: [] for name in candidates
    }
    replay_errors = []
    boundaries = []
    logical_accounting: dict[str, dict[str, int]] = {
        name: {
            "logical_total_macs": 0,
            "reference_dense_prefix_total_macs": 0,
            "valid_key_tokens": 0,
            "demanded_query_tokens": 0,
            "logical_causal_key_pairs": 0,
        }
        for name in candidates
    }

    with torch.no_grad():
        for batch in batches:
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            native = _run_native_stack(
                adapter,
                batch,
                plan=plan,  # type: ignore[arg-type]
                full_logits=True,
            )
            if native.logits is None:
                raise RuntimeError("native evaluation logits are missing")
            if not torch.equal(
                native.sequence.query_valid_mask,
                native.sequence.key_valid_mask,
            ):
                raise ValueError(
                    "middle-layer evaluation requires full valid-row demand"
                )
            boundary = _BoundaryBatch(
                input_hidden=native.block_input.detach(),
                output_hidden=native.block_output.detach(),
                valid_positions=batch.valid_positions,
                logical_positions=native.sequence.logical_positions,
                example_ids=ids,
            )
            boundaries.append(boundary)
            for name, executor in candidates.items():
                replacement, audit = _run_replacement_with_call_audit(
                    adapter,
                    batch,
                    plan=plan,  # type: ignore[arg-type]
                    executor=executor,  # type: ignore[arg-type]
                    full_logits=True,
                )
                if replacement.logits is None:
                    raise RuntimeError(
                        "replacement evaluation logits are missing"
                    )
                call_audits[name]["executor_calls"] += int(
                    audit["executor_calls"]
                )
                call_audits[name]["source_layer_calls"] += int(
                    audit["source_block_calls_total"]
                )
                prefix_errors[name].append(
                    float(
                        (
                            replacement.block_input.to(torch.float64)
                            - native.block_input.to(torch.float64)
                        )
                        .abs()
                        .max()
                        .item()
                    )
                )
                behavior_rows[name].extend(
                    _behavior_examples_with_kl(
                        batch=batch,
                        example_ids=ids,
                        baseline_logits=native.logits,
                        predicted_logits=replacement.logits,
                    )
                )
                direct_rows[name].extend(
                    _direct_rows(boundary, replacement.block_output)
                )
                accounting = executor.logical_accounting(native.sequence)
                logical_accounting[name]["logical_total_macs"] += (
                    accounting.logical_total_macs
                )
                logical_accounting[name][
                    "reference_dense_prefix_total_macs"
                ] += accounting.reference_dense_prefix_total_macs
                for field in (
                    "valid_key_tokens",
                    "demanded_query_tokens",
                    "logical_causal_key_pairs",
                ):
                    logical_accounting[name][field] += int(
                        getattr(accounting, field)
                    )

            if include_controls:
                _, replay_logits, _ = _run_suffix_from_boundary(
                    adapter,
                    batch,
                    plan=plan,  # type: ignore[arg-type]
                    sequence=native.sequence,
                    boundary_output=native.block_output,
                    full_logits=True,
                )
                _, skip_logits, _ = _run_suffix_from_boundary(
                    adapter,
                    batch,
                    plan=plan,  # type: ignore[arg-type]
                    sequence=native.sequence,
                    boundary_output=native.block_input,
                    full_logits=True,
                )
                if replay_logits is None or skip_logits is None:
                    raise RuntimeError("control logits are missing")
                replay_errors.append(
                    float(
                        (
                            replay_logits.float()
                            - native.logits.float()
                        )
                        .abs()
                        .max()
                        .item()
                    )
                )
                behavior_rows["native_boundary_replay"].extend(
                    _behavior_examples_with_kl(
                        batch=batch,
                        example_ids=ids,
                        baseline_logits=native.logits,
                        predicted_logits=replay_logits,
                    )
                )
                direct_rows["native_boundary_replay"].extend(
                    _direct_rows(boundary, native.block_output)
                )
                behavior_rows["identity_layer_skip"].extend(
                    _behavior_examples_with_kl(
                        batch=batch,
                        example_ids=ids,
                        baseline_logits=native.logits,
                        predicted_logits=skip_logits,
                    )
                )
                direct_rows["identity_layer_skip"].extend(
                    _direct_rows(boundary, native.block_input)
                )

    width = next(iter(candidates.values())).width
    execution_audits = {}
    for name, values in call_audits.items():
        execution_audits[name] = {
            "batches": len(batches),
            "executor_calls": values["executor_calls"],
            "source_block_calls_total": values["source_layer_calls"],
            "source_layer_calls": {
                layer_id: 0
                for layer_id in plan.layer_ids  # type: ignore[attr-defined]
            },
            "maximum_prefix_boundary_replay_error": max(
                prefix_errors[name],
                default=0.0,
            ),
            "native_layers_skipped": plan.layer_ids,  # type: ignore[attr-defined]
            "passed": (
                values["source_layer_calls"] == 0
                and values["executor_calls"] == len(batches)
                and max(prefix_errors[name], default=0.0) == 0.0
            ),
        }
    return {
        "behavior": {
            name: _aggregate_behavior_with_kl(rows)
            for name, rows in behavior_rows.items()
        },
        "direct": {
            name: _aggregate_direct_examples(rows, width=width)
            for name, rows in direct_rows.items()
        },
        "execution_audits": execution_audits,
        "native_boundary_replay": {
            "evaluated": include_controls,
            "maximum_absolute_logit_error": (
                max(replay_errors) if replay_errors else None
            ),
            "tolerance": 1e-5,
            "passed": (
                bool(replay_errors)
                and max(replay_errors) <= 1e-5
                if include_controls
                else None
            ),
        },
        "logical_accounting": logical_accounting,
        "boundaries": tuple(boundaries),
    }


def _candidate_accounting(
    executor: FullWidthSingleLayerExecutor,
    logical: Mapping[str, int],
    *,
    source_static: Mapping[str, object],
    source_macs: Mapping[str, object],
) -> dict[str, object]:
    stored = executor.total_runtime_coefficient_count
    source_parameters = int(source_static["parameter_count"])
    source_total_macs = int(source_macs["total_macs"])
    logical_macs = int(logical["logical_total_macs"])
    reference_macs = int(
        logical["reference_dense_prefix_total_macs"]
    )
    return {
        "learned_parameter_count": executor.learned_parameter_count,
        "fixed_identity_decoder_coefficient_count": (
            executor.fixed_runtime_coefficient_count
        ),
        "runtime_stored_coefficient_count": stored,
        "source_layer_parameter_count": source_parameters,
        "stored_coefficient_ratio_to_source": stored / source_parameters,
        "logical_analytic_mac_count": logical_macs,
        "reference_dense_prefix_mac_count": reference_macs,
        "source_layer_analytic_mac_count": source_total_macs,
        "analytic_mac_ratio_to_source": logical_macs / source_total_macs,
        "reference_dense_mac_ratio_to_source": (
            reference_macs / source_total_macs
        ),
        "valid_key_tokens": int(logical["valid_key_tokens"]),
        "demanded_query_tokens": int(logical["demanded_query_tokens"]),
        "logical_causal_key_pairs": int(
            logical["logical_causal_key_pairs"]
        ),
        "causal_edge_control": executor.causal_edge_control,
        "identity_decoder_is_structural_not_learned": True,
        "normalization_and_softmax_operations_excluded": True,
        "latency_or_kernel_speed_claim": False,
    }


def _resource_gates(
    accounting: Mapping[str, object],
    *,
    max_stored_coefficient_ratio: float,
    max_analytic_mac_ratio: float,
) -> dict[str, bool]:
    return {
        "stored_coefficient_ratio": (
            float(accounting["stored_coefficient_ratio_to_source"])
            <= max_stored_coefficient_ratio
        ),
        "analytic_mac_ratio": (
            float(accounting["analytic_mac_ratio_to_source"])
            <= max_analytic_mac_ratio
        ),
    }


def _direct_gates(
    direct: Mapping[str, object],
    *,
    block_delta_nrmse_max: float,
    block_delta_cosine_min: float,
) -> dict[str, bool]:
    return {
        "block_delta_nrmse": (
            float(direct["block_delta_nrmse"])
            <= block_delta_nrmse_max
        ),
        "block_delta_cosine": (
            float(direct["block_delta_cosine"])
            >= block_delta_cosine_min
        ),
    }


def _build_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
    scientific_digest: str,
) -> dict[str, object]:
    training = payload["training"]
    assert isinstance(training, Mapping)
    fisher = training["activation_fisher"]
    assert isinstance(fisher, Mapping)
    fisher_summary = {
        key: copy.deepcopy(value)
        for key, value in fisher.items()
        if key not in {
            "matrix",
            "normalized_diagonal",
            "training_metric",
        }
    }
    delta_scale = training["delta_scale"]
    assert isinstance(delta_scale, Mapping)
    delta_scale_summary = {
        key: copy.deepcopy(value)
        for key, value in delta_scale.items()
        if key != "values"
    }
    executors = payload["executors"]
    assert isinstance(executors, Mapping)
    return {
        "schema": payload["schema"],
        "format_version": payload["format_version"],
        "scientific_status": copy.deepcopy(payload["scientific_status"]),
        "model": copy.deepcopy(payload["model"]),
        "protocol": copy.deepcopy(payload["protocol"]),
        "training": {
            "activation_fisher": fisher_summary,
            "delta_scale": delta_scale_summary,
            "full_causal": copy.deepcopy(training["full_causal"]),
            "same_position_control": copy.deepcopy(
                training["same_position_control"]
            ),
            "structural_probes": copy.deepcopy(
                training["structural_probes"]
            ),
        },
        "selection": copy.deepcopy(payload["selection"]),
        "validation": copy.deepcopy(payload["validation"]),
        "executors": {
            name: {
                "execution_fingerprint": state[
                    "execution_fingerprint"
                ],
                "causal_edges_enabled": state[
                    "causal_edges_enabled"
                ],
            }
            for name, state in executors.items()
        },
        "artifact": {
            "tensor_file": tensor_file,
            "tensor_file_ignored_by_git_policy": True,
            "contains_model_weights": False,
            "contains_executor_weights": True,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def run_gemma3_full_width_single_layer_experiment(
    *,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    layer_index: int = DEFAULT_LAYER_INDEX,
    max_length: int = 256,
    tokenization_batch_size: int = 1,
    hidden_width: int = DEFAULT_HIDDEN_WIDTH,
    executor_layers: int = DEFAULT_EXECUTOR_LAYERS,
    head_count: int = DEFAULT_HEAD_COUNT,
    feed_forward_width: int = DEFAULT_FEED_FORWARD_WIDTH,
    fisher_floor: float = DEFAULT_FISHER_FLOOR,
    delta_scale_floor: float = DEFAULT_RIDGE_SCALE_FLOOR,
    local_warmup_steps: int = DEFAULT_LOCAL_WARMUP_STEPS,
    train_steps: int = DEFAULT_TRAIN_STEPS,
    train_positions_per_sequence: int = (
        DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE
    ),
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
    local_mse_weight: float = DEFAULT_LOCAL_MSE_WEIGHT,
    local_fisher_weight: float = DEFAULT_LOCAL_FISHER_WEIGHT,
    ground_truth_weight: float = DEFAULT_GROUND_TRUTH_WEIGHT,
    teacher_kl_weight: float = DEFAULT_TEACHER_KL_WEIGHT,
    selection_nll_atol: float = DEFAULT_NLL_ATOL,
    selection_top1_min: float = DEFAULT_TOP1_MIN,
    selection_teacher_kl_max: float = DEFAULT_TEACHER_KL_MAX,
    selection_p90_abs_nll_max: float = (
        DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX
    ),
    selection_p10_top1_min: float = (
        DEFAULT_PER_PROMPT_P10_TOP1_MIN
    ),
    block_delta_nrmse_max: float = DEFAULT_BLOCK_DELTA_NRMSE_MAX,
    block_delta_cosine_min: float = DEFAULT_BLOCK_DELTA_COSINE_MIN,
    max_stored_coefficient_ratio: float = (
        DEFAULT_MAX_STORED_COEFFICIENT_RATIO
    ),
    max_analytic_mac_ratio: float = DEFAULT_MAX_ANALYTIC_MAC_RATIO,
    minimum_calibration_a_prompts: int = (
        DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
    ),
    minimum_heldout_prompts: int = DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    minimum_fisher_rows: int = DEFAULT_MINIMUM_FISHER_ROWS,
    minimum_train_supervised_tokens: int = (
        DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS
    ),
    minimum_heldout_supervised_tokens: int = (
        DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
    ),
    minimum_length_buckets: int = DEFAULT_MINIMUM_LENGTH_BUCKETS,
    seed: int = 91_104,
    device_name: str = "cpu",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Fit on A, lock on B, and conditionally validate one layer."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    for label, value, minimum in (
        ("layer_index", layer_index, 0),
        ("max_length", max_length, 2),
        ("tokenization_batch_size", tokenization_batch_size, 1),
        ("hidden_width", hidden_width, 1),
        ("executor_layers", executor_layers, 1),
        ("head_count", head_count, 1),
        ("feed_forward_width", feed_forward_width, 1),
        ("local_warmup_steps", local_warmup_steps, 0),
        ("train_steps", train_steps, 1),
        (
            "train_positions_per_sequence",
            train_positions_per_sequence,
            1,
        ),
        (
            "minimum_calibration_a_prompts",
            minimum_calibration_a_prompts,
            1,
        ),
        ("minimum_heldout_prompts", minimum_heldout_prompts, 1),
        ("minimum_fisher_rows", minimum_fisher_rows, 1),
        (
            "minimum_train_supervised_tokens",
            minimum_train_supervised_tokens,
            1,
        ),
        (
            "minimum_heldout_supervised_tokens",
            minimum_heldout_supervised_tokens,
            1,
        ),
        ("minimum_length_buckets", minimum_length_buckets, 1),
        ("seed", seed, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if hidden_width % head_count:
        raise ValueError("hidden_width must be divisible by head_count")
    if minimum_length_buckets > 4:
        raise ValueError("minimum_length_buckets cannot exceed four")
    fisher_floor = _finite(
        fisher_floor,
        label="fisher_floor",
        minimum=torch.finfo(torch.float64).tiny,
    )
    delta_scale_floor = _finite(
        delta_scale_floor,
        label="delta_scale_floor",
        minimum=torch.finfo(torch.float64).tiny,
    )
    learning_rate = _finite(
        learning_rate,
        label="learning_rate",
        minimum=torch.finfo(torch.float64).tiny,
    )
    weight_decay = _finite(
        weight_decay,
        label="weight_decay",
        minimum=0.0,
    )
    gradient_clip_norm = _finite(
        gradient_clip_norm,
        label="gradient_clip_norm",
        minimum=torch.finfo(torch.float64).tiny,
    )
    local_mse_weight = _finite(
        local_mse_weight,
        label="local_mse_weight",
        minimum=0.0,
    )
    local_fisher_weight = _finite(
        local_fisher_weight,
        label="local_fisher_weight",
        minimum=0.0,
    )
    ground_truth_weight = _finite(
        ground_truth_weight,
        label="ground_truth_weight",
        minimum=0.0,
    )
    teacher_kl_weight = _finite(
        teacher_kl_weight,
        label="teacher_kl_weight",
        minimum=0.0,
    )
    if (
        local_mse_weight == 0
        and local_fisher_weight == 0
        and ground_truth_weight == 0
        and teacher_kl_weight == 0
    ):
        raise ValueError("at least one downstream loss weight must be positive")
    thresholds = {
        "nll_atol": _finite(
            selection_nll_atol,
            label="selection_nll_atol",
            minimum=0.0,
        ),
        "top1_min": _finite(
            selection_top1_min,
            label="selection_top1_min",
            minimum=0.0,
            maximum=1.0,
        ),
        "teacher_kl_max": _finite(
            selection_teacher_kl_max,
            label="selection_teacher_kl_max",
            minimum=0.0,
        ),
        "p90_abs_nll_max": _finite(
            selection_p90_abs_nll_max,
            label="selection_p90_abs_nll_max",
            minimum=0.0,
        ),
        "p10_top1_min": _finite(
            selection_p10_top1_min,
            label="selection_p10_top1_min",
            minimum=0.0,
            maximum=1.0,
        ),
        "block_delta_nrmse_max": _finite(
            block_delta_nrmse_max,
            label="block_delta_nrmse_max",
            minimum=0.0,
        ),
        "block_delta_cosine_min": _finite(
            block_delta_cosine_min,
            label="block_delta_cosine_min",
            minimum=-1.0,
            maximum=1.0,
        ),
        "max_stored_coefficient_ratio": _finite(
            max_stored_coefficient_ratio,
            label="max_stored_coefficient_ratio",
            minimum=0.0,
        ),
        "max_analytic_mac_ratio": _finite(
            max_analytic_mac_ratio,
            label="max_analytic_mac_ratio",
            minimum=0.0,
        ),
    }

    prompt_path = Path(prompt_splits_path)
    family_path = Path(family_manifest_path)
    prompts = load_gemma3_prompt_splits(prompt_path)
    _require_prompt_protocol(
        prompts,
        minimum_calibration_a_prompts=minimum_calibration_a_prompts,
        minimum_heldout_prompts=minimum_heldout_prompts,
    )
    families = load_prompt_family_manifest(
        family_path,
        prompts=prompts,
    )
    prompt_exclusions = _tracked_prompt_exclusion_audit(
        prompts,
        prompt_path=prompt_path,
    )
    prompt_metadata = prompts.metadata()
    family_metadata = families.metadata()

    resolved_output = (
        default_gemma3_full_width_single_layer_output(
            model_id,
            layer_index,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    if resolved_output.exists() or resolved_output.with_suffix(
        ".json"
    ).exists():
        raise FileExistsError(
            "refusing to overwrite a single-layer scientific artifact"
        )

    device = resolve_torch_device(device_name)
    if device.type == "mps":
        raise ValueError(
            "the single-layer Fisher audit requires CPU or CUDA float64"
        )
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    model.eval()
    model.requires_grad_(False)
    guard = _FrozenModelTensorGuard(model)
    adapter = Gemma3CausalLMAdapter(model)
    plan = adapter.plan_layer_block(layer_index, layer_index)
    width = plan.widths[0]
    if plan.widths != (width, width):
        raise ValueError("selected Gemma layer changes residual width")
    visibility_contract = _attention_visibility_contract(
        adapter,
        layer_id=plan.layer_ids[0],
        maximum_length=max_length,
    )
    source_accounting_manifest = _source_accounting_manifest(
        adapter,
        layer_ids=plan.layer_ids,
    )
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )

    train_batches, train_stream = _materialize_split(
        tokenizer,
        prompts.calibration_a,
        split_name="calibration_a",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    train_contract = _tokenized_stream_contract(
        train_stream,
        split_name="calibration_a",
        minimum_supervised_tokens=minimum_train_supervised_tokens,
        minimum_length_buckets=minimum_length_buckets,
    )
    training = _collect_training_batches(
        adapter,
        train_batches,
        plan=plan,
        positions_per_sequence=train_positions_per_sequence,
    )
    _require_complete_middle_layer_demand(adapter, training)
    delta_scale, delta_scale_report = _delta_scale(
        training,
        floor=delta_scale_floor,
    )
    fisher_matrix, fisher_diagonal, fisher_report = _activation_fisher(
        adapter,
        training,
        plan=plan,
        fisher_floor=fisher_floor,
    )
    if int(fisher_report["rows"]) < minimum_fisher_rows:
        raise ValueError("activation Fisher row count is below the minimum")
    fisher_metric, fisher_metric_report = _normalized_fisher_metric(
        fisher_matrix,
        delta_scale,
        eigenvalue_floor=fisher_floor,
    )
    candidates = {
        "full_causal": _make_executor(
            width=width,
            hidden_width=hidden_width,
            executor_layers=executor_layers,
            head_count=head_count,
            feed_forward_width=feed_forward_width,
            causal_edges_enabled=True,
            seed=seed,
            device=device,
        ),
        "same_position_control": _make_executor(
            width=width,
            hidden_width=hidden_width,
            executor_layers=executor_layers,
            head_count=head_count,
            feed_forward_width=feed_forward_width,
            causal_edges_enabled=False,
            seed=seed,
            device=device,
        ),
    }
    source_independence = _assert_source_independence(model, candidates)

    training_reports = {}
    structural_probes = {}
    for name, executor in candidates.items():
        training_reports[name] = _fit_executor(
            adapter,
            executor,
            training,
            plan=plan,
            delta_scale=delta_scale,
            fisher_metric=fisher_metric,
            local_warmup_steps=local_warmup_steps,
            train_steps=train_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            local_mse_weight=local_mse_weight,
            local_fisher_weight=local_fisher_weight,
            ground_truth_weight=ground_truth_weight,
            teacher_kl_weight=teacher_kl_weight,
        )
        structural_probes[name] = _full_width_structural_probes(
            adapter,
            executor,
            training,
        )
        if structural_probes[name]["passed"] is not True:
            raise RuntimeError(
                f"{name} failed causal or padding structural probes"
            )
    guard.assert_unchanged()

    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        prompts.calibration_b,
        split_name="calibration_b",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_contract = _tokenized_stream_contract(
        selection_stream,
        split_name="calibration_b",
        minimum_supervised_tokens=minimum_heldout_supervised_tokens,
        minimum_length_buckets=minimum_length_buckets,
    )
    tokenized_content_audit = _assert_tokenized_content_disjointness(
        {
            "calibration_a": train_stream,
            "calibration_b": selection_stream,
        }
    )
    selection_result = _split_evaluation(
        adapter,
        selection_batches,
        plan=plan,
        candidates=candidates,
        include_controls=True,
    )
    selection_boundaries = selection_result.pop("boundaries")
    source_static = _source_block_static(adapter, plan)
    if (
        source_static["parameter_count"]
        != source_accounting_manifest["parameter_count"]
        or source_static["parameter_bytes"]
        != source_accounting_manifest["parameter_bytes"]
        or source_static["linear_weight_coefficients_by_layer"]
        != source_accounting_manifest[
            "linear_weight_coefficients_by_layer"
        ]
    ):
        raise RuntimeError(
            "source accounting manifest does not match live source modules"
        )
    source_macs = _source_block_macs(
        adapter,
        plan,
        selection_boundaries,  # type: ignore[arg-type]
        static=source_static,
    )
    selection_accounting = {
        name: _candidate_accounting(
            executor,
            selection_result["logical_accounting"][name],  # type: ignore[index]
            source_static=source_static,
            source_macs=source_macs,
        )
        for name, executor in candidates.items()
    }
    selection_gates = {
        name: _behavior_gates(
            selection_result["behavior"][name],  # type: ignore[index]
            nll_atol=thresholds["nll_atol"],
            top1_min=thresholds["top1_min"],
            teacher_kl_max=thresholds["teacher_kl_max"],
            p90_abs_nll_max=thresholds["p90_abs_nll_max"],
            p10_top1_min=thresholds["p10_top1_min"],
        )
        for name in candidates
    }
    selection_direct_gates = {
        name: _direct_gates(
            selection_result["direct"][name],  # type: ignore[index]
            block_delta_nrmse_max=thresholds[
                "block_delta_nrmse_max"
            ],
            block_delta_cosine_min=thresholds[
                "block_delta_cosine_min"
            ],
        )
        for name in candidates
    }
    resource_gates = _resource_gates(
        selection_accounting["full_causal"],
        max_stored_coefficient_ratio=thresholds[
            "max_stored_coefficient_ratio"
        ],
        max_analytic_mac_ratio=thresholds["max_analytic_mac_ratio"],
    )
    replay_passed = (
        selection_result["native_boundary_replay"]["passed"]  # type: ignore[index]
        is True
    )
    selection_passed = (
        all(selection_gates["full_causal"].values())
        and all(selection_direct_gates["full_causal"].values())
        and all(resource_gates.values())
        and selection_result["execution_audits"]["full_causal"][  # type: ignore[index]
            "passed"
        ]
        is True
        and replay_passed
    )
    guard.assert_unchanged()

    validation_evaluated = selection_passed
    validation_stream: dict[str, object] | None = None
    validation_payload: dict[str, object]
    validation_passed = False
    validation_accounting: dict[str, dict[str, object]] | None = None
    if validation_evaluated:
        validation_batches, validation_stream = _materialize_split(
            tokenizer,
            prompts.validation,
            split_name="validation",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        validation_contract = _tokenized_stream_contract(
            validation_stream,
            split_name="validation",
            minimum_supervised_tokens=(
                minimum_heldout_supervised_tokens
            ),
            minimum_length_buckets=minimum_length_buckets,
        )
        tokenized_content_audit = (
            _assert_tokenized_content_disjointness(
                {
                    "calibration_a": train_stream,
                    "calibration_b": selection_stream,
                    "validation": validation_stream,
                }
            )
        )
        validation_result = _split_evaluation(
            adapter,
            validation_batches,
            plan=plan,
            candidates=candidates,
            include_controls=True,
        )
        validation_boundaries = validation_result.pop("boundaries")
        validation_source_macs = _source_block_macs(
            adapter,
            plan,
            validation_boundaries,  # type: ignore[arg-type]
            static=source_static,
        )
        validation_accounting = {
            name: _candidate_accounting(
                executor,
                validation_result["logical_accounting"][name],  # type: ignore[index]
                source_static=source_static,
                source_macs=validation_source_macs,
            )
            for name, executor in candidates.items()
        }
        validation_gates = {
            name: _behavior_gates(
                validation_result["behavior"][name],  # type: ignore[index]
                nll_atol=thresholds["nll_atol"],
                top1_min=thresholds["top1_min"],
                teacher_kl_max=thresholds["teacher_kl_max"],
                p90_abs_nll_max=thresholds["p90_abs_nll_max"],
                p10_top1_min=thresholds["p10_top1_min"],
            )
            for name in candidates
        }
        validation_direct_gates = {
            name: _direct_gates(
                validation_result["direct"][name],  # type: ignore[index]
                block_delta_nrmse_max=thresholds[
                    "block_delta_nrmse_max"
                ],
                block_delta_cosine_min=thresholds[
                    "block_delta_cosine_min"
                ],
            )
            for name in candidates
        }
        validation_resource = _resource_gates(
            validation_accounting["full_causal"],
            max_stored_coefficient_ratio=thresholds[
                "max_stored_coefficient_ratio"
            ],
            max_analytic_mac_ratio=thresholds[
                "max_analytic_mac_ratio"
            ],
        )
        validation_passed = (
            all(validation_gates["full_causal"].values())
            and all(validation_direct_gates["full_causal"].values())
            and all(validation_resource.values())
            and validation_result["execution_audits"][  # type: ignore[index]
                "full_causal"
            ]["passed"]
            is True
            and validation_result["native_boundary_replay"][  # type: ignore[index]
                "passed"
            ]
            is True
        )
        validation_payload = {
            "evaluated": True,
            "reason": "calibration_b_passed_locked_executors_evaluated",
            "behavior": validation_result["behavior"],
            "direct": validation_result["direct"],
            "direct_gates": validation_direct_gates,
            "execution_audits": validation_result[
                "execution_audits"
            ],
            "native_boundary_replay": validation_result[
                "native_boundary_replay"
            ],
            "behavior_gates": validation_gates,
            "resource_gates": validation_resource,
            "accounting": validation_accounting,
            "passed": validation_passed,
            "tokenized_stream": validation_stream,
            "tokenized_stream_contract": validation_contract,
        }
    else:
        validation_payload = {
            "evaluated": False,
            "reason": "calibration_b_failed_validation_not_tokenized",
            "behavior": None,
            "direct": None,
            "direct_gates": None,
            "execution_audits": None,
            "native_boundary_replay": None,
            "behavior_gates": None,
            "resource_gates": None,
            "accounting": None,
            "passed": False,
            "tokenized_stream": None,
            "tokenized_stream_contract": None,
        }
    guard.assert_unchanged()

    control_passed = (
        all(selection_gates["same_position_control"].values())
        and all(
            selection_direct_gates["same_position_control"].values()
        )
        and selection_result["execution_audits"][
            "same_position_control"
        ]["passed"]  # type: ignore[index]
        is True
    )
    selection_payload = {
        "behavior": selection_result["behavior"],
        "direct": selection_result["direct"],
        "direct_gates": selection_direct_gates,
        "execution_audits": selection_result["execution_audits"],
        "native_boundary_replay": selection_result[
            "native_boundary_replay"
        ],
        "behavior_gates": selection_gates,
        "resource_gates": resource_gates,
        "accounting": selection_accounting,
        "full_causal_passed": selection_passed,
        "same_position_control_passed": control_passed,
        "causal_edge_threshold_separation_observed": (
            selection_passed and not control_passed
        ),
        "locked_candidate": (
            "full_causal" if selection_passed else None
        ),
        "tokenized_stream": selection_stream,
        "tokenized_stream_contract": selection_contract,
    }
    recorded_protocol_passed = selection_passed and validation_passed
    tokenized_splits = {
        "calibration_a": train_stream,
        "calibration_b": selection_stream,
    }
    if validation_stream is not None:
        tokenized_splits["validation"] = validation_stream
    tokenized_stream_contracts: dict[str, object] = {
        "calibration_a": train_contract,
        "calibration_b": selection_contract,
    }
    if validation_evaluated:
        tokenized_stream_contracts["validation"] = validation_contract
    data_minima = {
        "minimum_calibration_a_prompts": minimum_calibration_a_prompts,
        "minimum_heldout_prompts_per_role": minimum_heldout_prompts,
        "minimum_fisher_rows": minimum_fisher_rows,
        "minimum_train_supervised_tokens": (
            minimum_train_supervised_tokens
        ),
        "minimum_heldout_supervised_tokens_per_role": (
            minimum_heldout_supervised_tokens
        ),
        "minimum_populated_length_buckets_per_tokenized_role": (
            minimum_length_buckets
        ),
    }
    strong_data_minima_enforced = (
        minimum_calibration_a_prompts
        >= DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
        and minimum_heldout_prompts >= DEFAULT_MINIMUM_HELDOUT_PROMPTS
        and minimum_fisher_rows >= DEFAULT_MINIMUM_FISHER_ROWS
        and minimum_train_supervised_tokens
        >= DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS
        and minimum_heldout_supervised_tokens
        >= DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
        and minimum_length_buckets >= DEFAULT_MINIMUM_LENGTH_BUCKETS
    )
    resolved_commit = model_metadata.get("resolved_commit")
    immutable_model_revision_recorded = (
        isinstance(resolved_commit, str)
        and re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved_commit)
        is not None
    )
    payload: dict[str, object] = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_executor_weights": True,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "scientific_status": {
            "scope": (
                "full_width_source_independent_single_gemma_layer_"
                "replacement"
            ),
            "rank_reduction_attempted": False,
            "retained_rank": width,
            "residual_width": width,
            "activation_fisher_computed": True,
            "calibration_a_executors_fitted": True,
            "calibration_b_evaluated": True,
            "calibration_b_passed": selection_passed,
            "validation_evaluated": validation_evaluated,
            "validation_passed": validation_passed,
            "test_evaluated": False,
            "source_layer_calls_in_student_path": 0,
            "source_layer_removed_from_student_path": True,
            "source_independent_executor": True,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "candidate_passed_recorded_single_seed_protocol": (
                recorded_protocol_passed
            ),
            "strong_data_minima_enforced": strong_data_minima_enforced,
            "immutable_model_revision_recorded": (
                immutable_model_revision_recorded
            ),
            "fidelity_viable_replacement": False,
            "model_level_promotion_authorized": False,
            "parameter_reduction_supported": False,
            "analytic_mac_reduction_supported": False,
            "parameter_reduction_observed_under_recorded_protocol": False,
            "analytic_mac_reduction_observed_under_recorded_protocol": False,
            "source_resource_denominators_reverified_on_load": False,
            "causal_edge_value_identified": False,
            "causal_edge_threshold_separation_observed": selection_payload[
                "causal_edge_threshold_separation_observed"
            ],
            "latency_or_kernel_speed_claim": False,
        },
        "model": model_metadata,
        "protocol": {
            "layer_index": layer_index,
            "layer_ids": plan.layer_ids,
            "canonical_boundaries": plan.activation_sites,
            "residual_width": width,
            "retained_rank": width,
            "rank_reduction_attempted": False,
            "executor_architecture": {
                "hidden_width": hidden_width,
                "layer_count": executor_layers,
                "head_count": head_count,
                "feed_forward_width": feed_forward_width,
                "full_causal_and_storage_matched_attention_ablation": True,
            },
            "maximum_tokenized_length": max_length,
            "tokenization_batch_size": tokenization_batch_size,
            "optimization_seed": seed,
            "single_seed_protocol": True,
            "numeric_policy": {
                "requested_device": device_name,
                "resolved_device": str(device),
                "requested_model_dtype": dtype,
                "resolved_model_dtype": model_metadata["dtype"],
                "executor_parameter_dtype": "torch.float32",
                "fisher_accumulation_dtype": "torch.float64",
                "local_fisher_quadratic_dtype": "torch.float32",
            },
            "source_attention_visibility": visibility_contract,
            "source_accounting_manifest": source_accounting_manifest,
            "training_split": "calibration_a_only",
            "selection_policy": (
                "one_predeclared_full_causal_candidate_passes_all_b_gates"
            ),
            "validation_policy": (
                "tokenize_once_only_after_calibration_b_passes"
            ),
            "test_policy": "parse_validate_hash_only",
            "student_execution": (
                "native_prefix_single_full_width_executor_native_suffix"
            ),
            "native_layer_output_available_to_student": False,
            "thresholds": thresholds,
            "data_minima": data_minima,
            "strong_data_minima_enforced": strong_data_minima_enforced,
            "prompt_splits": prompt_metadata,
            "prompt_families": family_metadata,
            "prompt_exclusions": prompt_exclusions,
            "prompt_fixture_file_sha256": _file_sha256(prompt_path),
            "family_manifest_file_sha256": _file_sha256(family_path),
            "tokenized_splits": tokenized_splits,
            "tokenized_stream_contracts": tokenized_stream_contracts,
            "tokenized_content_disjointness": tokenized_content_audit,
            "library_versions": _library_versions(),
            "tokenizer": _tokenizer_provenance(tokenizer),
            "model_state_guard": guard.metadata(),
            "source_independence": source_independence,
        },
        "executors": {
            name: executor.artifact_state_dict()
            for name, executor in candidates.items()
        },
        "training": {
            "activation_fisher": {
                **fisher_report,
                "matrix": fisher_matrix,
                "normalized_diagonal": fisher_diagonal,
                "training_metric": fisher_metric,
                "training_metric_report": fisher_metric_report,
            },
            "delta_scale": {
                **delta_scale_report,
                "values": delta_scale,
            },
            "full_causal": training_reports["full_causal"],
            "same_position_control": training_reports[
                "same_position_control"
            ],
            "structural_probes": structural_probes,
            "tokenized_stream": train_stream,
        },
        "selection": selection_payload,
        "validation": validation_payload,
    }
    digest = _scientific_payload_sha256(payload)
    report = _build_report(
        payload,
        tensor_file=str(resolved_output),
        scientific_digest=digest,
    )
    report_digest = _report_sha256(report)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "scientific_payload_sha256": digest,
            "report_sha256": report_digest,
        },
        resolved_output,
    )
    resolved_output.with_suffix(".json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _strict_mapping(
    value: object,
    *,
    label: str,
    fields: set[str] | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if fields is not None and set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _strict_float64_tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...],
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or tuple(value.shape) != shape
        or not torch.isfinite(value).all()
    ):
        raise ValueError(
            f"{label} must be a finite CPU float64 tensor with shape "
            f"{shape}"
        )
    return value


def _validate_tensor_locations(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Tensor):
        executor_state = (
            len(path) >= 5
            and path[0] == "executors"
            and path[1] in {
                "full_causal",
                "same_position_control",
            }
            and path[2:4] == ("executor", "model_state_dict")
        )
        scientific_tensor = path in {
            ("training", "activation_fisher", "matrix"),
            (
                "training",
                "activation_fisher",
                "normalized_diagonal",
            ),
            ("training", "activation_fisher", "training_metric"),
            ("training", "delta_scale", "values"),
        }
        if not executor_state and not scientific_tensor:
            raise ValueError(
                "single-layer artifact contains a Tensor outside declared "
                f"executor or Fisher fields: {'.'.join(path)}"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact mapping keys must be strings")
            _validate_tensor_locations(item, path=(*path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_tensor_locations(
                item,
                path=(*path, str(index)),
            )


def _validate_behavior_aggregate(
    value: object,
    *,
    stream: Mapping[str, object],
    label: str,
) -> Mapping[str, object]:
    behavior = _strict_mapping(value, label=f"{label} behavior")
    examples = behavior.get("examples")
    stream_examples = stream.get("examples")
    if (
        not isinstance(examples, list)
        or not examples
        or not isinstance(stream_examples, list)
        or len(examples) != len(stream_examples)
    ):
        raise ValueError(f"{label} behavior examples are invalid")
    for row, stream_row in zip(examples, stream_examples, strict=True):
        if not isinstance(row, Mapping) or not isinstance(
            stream_row,
            Mapping,
        ):
            raise ValueError(f"{label} behavior row is invalid")
        supervised = row.get("supervised_tokens")
        teacher_kl = row.get("teacher_kl_summed")
        teacher_kl_per_token = row.get("teacher_kl_per_token")
        if (
            row.get("example_id") != stream_row.get("example_id")
            or supervised != stream_row.get("supervised_positions")
            or type(supervised) is not int
            or supervised <= 0
            or not isinstance(teacher_kl, float)
            or not math.isfinite(teacher_kl)
            or teacher_kl < -1e-6
            or not isinstance(teacher_kl_per_token, float)
            or not math.isfinite(teacher_kl_per_token)
            or not math.isclose(
                teacher_kl_per_token,
                teacher_kl / supervised,
                rel_tol=1e-11,
                abs_tol=1e-13,
            )
        ):
            raise ValueError(f"{label} behavior row is invalid")
    recomputed = _aggregate_behavior_with_kl(examples)
    if not _semantic_numeric_equal(behavior, recomputed):
        raise ValueError(
            f"{label} behavior aggregate does not recompute"
        )
    return behavior


def _validate_direct_aggregate(
    value: object,
    *,
    stream: Mapping[str, object],
    width: int,
    label: str,
) -> Mapping[str, object]:
    direct = _strict_mapping(value, label=f"{label} direct")
    examples = direct.get("examples")
    stream_examples = stream.get("examples")
    if (
        not isinstance(examples, list)
        or not examples
        or not isinstance(stream_examples, list)
        or len(examples) != len(stream_examples)
    ):
        raise ValueError(f"{label} direct examples are invalid")
    for row, stream_row in zip(examples, stream_examples, strict=True):
        if (
            not isinstance(row, Mapping)
            or not isinstance(stream_row, Mapping)
            or row.get("example_id") != stream_row.get("example_id")
            or row.get("valid_tokens") != stream_row.get("valid_tokens")
        ):
            raise ValueError(f"{label} direct row is invalid")
    recomputed = _aggregate_direct_examples(examples, width=width)
    if not _semantic_numeric_equal(direct, recomputed):
        raise ValueError(f"{label} direct aggregate does not recompute")
    return direct


def _validate_execution_audit(
    value: object,
    *,
    layer_ids: tuple[str, ...],
    label: str,
) -> None:
    audit = _strict_mapping(value, label=f"{label} execution audit")
    layer_calls = audit.get("source_layer_calls")
    valid = (
        type(audit.get("batches")) is int
        and audit["batches"] > 0
        and audit.get("executor_calls") == audit["batches"]
        and audit.get("source_block_calls_total") == 0
        and isinstance(layer_calls, Mapping)
        and tuple(layer_calls) == layer_ids
        and all(layer_calls[layer_id] == 0 for layer_id in layer_ids)
        and tuple(audit.get("native_layers_skipped", ())) == layer_ids
        and audit.get("maximum_prefix_boundary_replay_error") == 0.0
    )
    if audit.get("passed") is not valid or not valid:
        raise ValueError(f"{label} execution audit does not recompute")


def _validate_native_replay(value: object, *, label: str) -> None:
    replay = _strict_mapping(
        value,
        label=f"{label} native boundary replay",
        fields={
            "evaluated",
            "maximum_absolute_logit_error",
            "tolerance",
            "passed",
        },
    )
    error = _finite(
        replay["maximum_absolute_logit_error"],
        label=f"{label} replay error",
        minimum=0.0,
    )
    tolerance = _finite(
        replay["tolerance"],
        label=f"{label} replay tolerance",
        minimum=0.0,
    )
    expected = error <= tolerance
    if (
        replay["evaluated"] is not True
        or replay["passed"] is not expected
        or not expected
    ):
        raise ValueError(f"{label} native boundary replay is invalid")


def _manifest_shape_numel(value: object, *, label: str) -> int:
    if (
        not isinstance(value, tuple)
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise ValueError(f"{label} shape is invalid")
    return math.prod(value)


def _validate_source_accounting_manifest(
    value: object,
    *,
    layer_ids: tuple[str, ...],
    visibility: Mapping[str, object],
) -> Mapping[str, object]:
    manifest = _strict_mapping(
        value,
        label="single-layer source accounting manifest",
        fields={
            "scope",
            "layer_ids",
            "parameter_entries",
            "parameter_count",
            "parameter_bytes",
            "linear_weight_entries",
            "linear_weight_coefficients_by_layer",
            "attention_by_layer",
            "sequence_accounting_contract",
            "manifest_sha256",
        },
    )
    if (
        manifest["scope"] != "exact_selected_source_module_geometry"
        or manifest["layer_ids"] != layer_ids
        or not _is_sha256(manifest["manifest_sha256"])
    ):
        raise ValueError("single-layer source accounting manifest is invalid")

    parameter_entries = manifest["parameter_entries"]
    if not isinstance(parameter_entries, list) or not parameter_entries:
        raise ValueError("source parameter manifest entries are invalid")
    parameter_shapes: dict[tuple[str, str], tuple[tuple[int, ...], int]] = {}
    parameter_count = 0
    parameter_bytes = 0
    for index, raw_entry in enumerate(parameter_entries):
        entry = _strict_mapping(
            raw_entry,
            label=f"source parameter manifest entry {index}",
            fields={
                "layer_id",
                "name",
                "shape",
                "numel",
                "element_size",
            },
        )
        layer_id = entry["layer_id"]
        name = entry["name"]
        if (
            layer_id not in layer_ids
            or not isinstance(name, str)
            or not name
            or type(entry["element_size"]) is not int
            or entry["element_size"] <= 0
        ):
            raise ValueError("source parameter manifest entry is invalid")
        numel = _manifest_shape_numel(
            entry["shape"],
            label=f"source parameter {layer_id}.{name}",
        )
        if entry["numel"] != numel:
            raise ValueError("source parameter manifest numel is invalid")
        key = (layer_id, name)
        if key in parameter_shapes:
            raise ValueError("source parameter manifest names are duplicated")
        parameter_shapes[key] = (entry["shape"], numel)
        parameter_count += numel
        parameter_bytes += numel * entry["element_size"]
    if (
        manifest["parameter_count"] != parameter_count
        or manifest["parameter_bytes"] != parameter_bytes
    ):
        raise ValueError("source parameter totals do not recompute")

    linear_entries = manifest["linear_weight_entries"]
    if not isinstance(linear_entries, list) or not linear_entries:
        raise ValueError("source linear-weight manifest entries are invalid")
    linear_counts = {layer_id: 0 for layer_id in layer_ids}
    seen_linear_names: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(linear_entries):
        entry = _strict_mapping(
            raw_entry,
            label=f"source linear-weight manifest entry {index}",
            fields={"layer_id", "module_name", "shape", "numel"},
        )
        layer_id = entry["layer_id"]
        module_name = entry["module_name"]
        if (
            layer_id not in layer_ids
            or not isinstance(module_name, str)
            or (layer_id, module_name) in seen_linear_names
        ):
            raise ValueError("source linear-weight manifest entry is invalid")
        shape = entry["shape"]
        numel = _manifest_shape_numel(
            shape,
            label=f"source linear weight {layer_id}.{module_name}",
        )
        if len(shape) != 2 or entry["numel"] != numel:
            raise ValueError("source linear-weight geometry is invalid")
        parameter_name = (
            f"{module_name}.weight" if module_name else "weight"
        )
        if parameter_shapes.get((layer_id, parameter_name)) != (shape, numel):
            raise ValueError(
                "source linear weight is not bound to a parameter entry"
            )
        seen_linear_names.add((layer_id, module_name))
        linear_counts[layer_id] += numel
    saved_linear_counts = _strict_mapping(
        manifest["linear_weight_coefficients_by_layer"],
        label="source linear-weight totals",
        fields=set(layer_ids),
    )
    if (
        saved_linear_counts != linear_counts
        or any(count <= 0 for count in linear_counts.values())
    ):
        raise ValueError("source linear-weight totals do not recompute")

    attention_by_layer = _strict_mapping(
        manifest["attention_by_layer"],
        label="source attention geometry",
        fields=set(layer_ids),
    )
    for layer_id in layer_ids:
        attention = _strict_mapping(
            attention_by_layer[layer_id],
            label=f"source attention geometry {layer_id}",
            fields={
                "kind",
                "query_heads",
                "head_dimension",
                "window_size",
            },
        )
        kind = attention["kind"]
        window = attention["window_size"]
        if (
            kind not in {"global_causal", "sliding_causal"}
            or type(attention["query_heads"]) is not int
            or attention["query_heads"] <= 0
            or type(attention["head_dimension"]) is not int
            or attention["head_dimension"] <= 0
            or (
                kind == "global_causal"
                and window is not None
            )
            or (
                kind == "sliding_causal"
                and (type(window) is not int or window <= 0)
            )
        ):
            raise ValueError("source attention geometry is invalid")
    selected_attention = attention_by_layer[layer_ids[0]]
    if (
        selected_attention["kind"]
        != visibility["source_attention_kind"]
        or selected_attention["window_size"]
        != visibility["source_window_size"]
    ):
        raise ValueError(
            "source attention manifest and visibility disagree"
        )

    sequence_contract = _strict_mapping(
        manifest["sequence_accounting_contract"],
        label="source accounting sequence contract",
        fields={
            "phase",
            "padding_side",
            "logical_position_domain",
            "query_and_key_valid_masks_equal",
            "source_mac_recomputed_from_recorded_valid_lengths",
        },
    )
    if sequence_contract != {
        "phase": "prefill",
        "padding_side": "right",
        "logical_position_domain": "zero_contiguous",
        "query_and_key_valid_masks_equal": True,
        "source_mac_recomputed_from_recorded_valid_lengths": True,
    }:
        raise ValueError("source accounting sequence contract is invalid")

    manifest_body = {
        key: manifest[key]
        for key in manifest
        if key != "manifest_sha256"
    }
    encoded = json.dumps(
        manifest_body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != manifest["manifest_sha256"]:
        raise ValueError("source accounting manifest digest is invalid")
    return manifest


def _recomputed_accounting_ledger(
    executor: FullWidthSingleLayerExecutor,
    *,
    stream: Mapping[str, object],
    tokenization_batch_size: int,
    source_manifest: Mapping[str, object],
) -> dict[str, int]:
    if type(tokenization_batch_size) is not int or tokenization_batch_size <= 0:
        raise ValueError("tokenization batch size is invalid")
    examples = stream.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("tokenized stream cannot recompute accounting")
    lengths = []
    for example in examples:
        if (
            not isinstance(example, Mapping)
            or type(example.get("valid_tokens")) is not int
            or example["valid_tokens"] <= 0
        ):
            raise ValueError("tokenized valid lengths are invalid")
        lengths.append(int(example["valid_tokens"]))

    logical = {
        "logical_total_macs": 0,
        "reference_dense_prefix_total_macs": 0,
        "valid_key_tokens": 0,
        "demanded_query_tokens": 0,
        "logical_causal_key_pairs": 0,
    }
    for start in range(0, len(lengths), tokenization_batch_size):
        chunk = lengths[start : start + tokenization_batch_size]
        width = max(chunk)
        positions = torch.arange(
            width,
            dtype=torch.long,
            device=executor.device,
        ).unsqueeze(0).expand(len(chunk), -1)
        valid = positions < torch.tensor(
            chunk,
            dtype=torch.long,
            device=executor.device,
        ).unsqueeze(1)
        sequence = SequenceContext(
            query_valid_mask=valid,
            key_valid_mask=valid,
            logical_positions=positions,
            key_logical_positions=positions,
            cache_positions=None,
            phase="prefill",
            input_origin=SequenceInputOrigin(
                attention_mask_supplied=True,
                position_ids_supplied=False,
                cache_positions_supplied=False,
            ),
        )
        accounting = executor.logical_accounting(sequence)
        logical["logical_total_macs"] += accounting.logical_total_macs
        logical["reference_dense_prefix_total_macs"] += (
            accounting.reference_dense_prefix_total_macs
        )
        for field in (
            "valid_key_tokens",
            "demanded_query_tokens",
            "logical_causal_key_pairs",
        ):
            logical[field] += int(getattr(accounting, field))

    valid_positions = sum(lengths)
    linear_by_layer = source_manifest[
        "linear_weight_coefficients_by_layer"
    ]
    attention_by_layer = source_manifest["attention_by_layer"]
    assert isinstance(linear_by_layer, Mapping)
    assert isinstance(attention_by_layer, Mapping)
    source_macs = 0
    for layer_id in source_manifest["layer_ids"]:
        attention = attention_by_layer[layer_id]
        assert isinstance(attention, Mapping)
        window = attention["window_size"]
        if window is None:
            edges = sum(length * (length + 1) // 2 for length in lengths)
        else:
            edges = sum(
                sum(min(position + 1, int(window)) for position in range(length))
                for length in lengths
            )
        source_macs += (
            valid_positions * int(linear_by_layer[layer_id])
            + edges
            * 2
            * int(attention["query_heads"])
            * int(attention["head_dimension"])
        )
    return {
        **logical,
        "source_layer_parameter_count": int(
            source_manifest["parameter_count"]
        ),
        "source_layer_analytic_mac_count": source_macs,
    }


def _validate_candidate_accounting(
    value: object,
    *,
    executor: FullWidthSingleLayerExecutor,
    stream: Mapping[str, object],
    tokenization_batch_size: int,
    source_manifest: Mapping[str, object],
    label: str,
) -> Mapping[str, object]:
    accounting = _strict_mapping(value, label=f"{label} accounting")
    expected_counts = _recomputed_accounting_ledger(
        executor,
        stream=stream,
        tokenization_batch_size=tokenization_batch_size,
        source_manifest=source_manifest,
    )
    learned = accounting.get("learned_parameter_count")
    fixed = accounting.get("fixed_identity_decoder_coefficient_count")
    stored = accounting.get("runtime_stored_coefficient_count")
    source_parameters = accounting.get("source_layer_parameter_count")
    logical_macs = accounting.get("logical_analytic_mac_count")
    reference_macs = accounting.get(
        "reference_dense_prefix_mac_count"
    )
    source_macs = accounting.get("source_layer_analytic_mac_count")
    valid = (
        learned == executor.learned_parameter_count
        and fixed == executor.fixed_runtime_coefficient_count
        and stored == executor.total_runtime_coefficient_count
        and source_parameters
        == expected_counts["source_layer_parameter_count"]
        and logical_macs == expected_counts["logical_total_macs"]
        and type(reference_macs) is int
        and reference_macs
        == expected_counts["reference_dense_prefix_total_macs"]
        and reference_macs >= logical_macs
        and source_macs
        == expected_counts["source_layer_analytic_mac_count"]
        and accounting.get("valid_key_tokens")
        == expected_counts["valid_key_tokens"]
        and accounting.get("demanded_query_tokens")
        == expected_counts["demanded_query_tokens"]
        and accounting.get("logical_causal_key_pairs")
        == expected_counts["logical_causal_key_pairs"]
        and accounting.get("causal_edge_control")
        == executor.causal_edge_control
        and accounting.get(
            "identity_decoder_is_structural_not_learned"
        )
        is True
        and accounting.get(
            "normalization_and_softmax_operations_excluded"
        )
        is True
        and accounting.get("latency_or_kernel_speed_claim") is False
    )
    if not valid:
        raise ValueError(f"{label} accounting is invalid")
    expected_ratios = {
        "stored_coefficient_ratio_to_source": (
            stored / source_parameters
        ),
        "analytic_mac_ratio_to_source": logical_macs / source_macs,
        "reference_dense_mac_ratio_to_source": (
            reference_macs / source_macs
        ),
    }
    for field, expected in expected_ratios.items():
        if not math.isclose(
            float(accounting.get(field, math.nan)),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{label} accounting ratio is invalid")
    return accounting


def _validate_structural_probes(value: object) -> None:
    probes = _strict_mapping(
        value,
        label="single-layer structural probes",
    )
    if set(probes) != {"full_causal", "same_position_control"}:
        raise ValueError("single-layer structural probe candidates are invalid")
    for name, raw_probe in probes.items():
        probe = _strict_mapping(
            raw_probe,
            label=f"{name} structural probe",
        )
        causal = _finite(
            probe.get("future_slot_perturbation_max_earlier_error"),
            label=f"{name} future perturbation error",
            minimum=0.0,
        ) <= _finite(
            probe.get("future_slot_tolerance"),
            label=f"{name} future perturbation tolerance",
            minimum=0.0,
        )
        batching = _finite(
            probe.get("padded_batch_vs_single_max_valid_error"),
            label=f"{name} padded batching error",
            minimum=0.0,
        ) <= _finite(
            probe.get("batching_tolerance"),
            label=f"{name} batching tolerance",
            minimum=0.0,
        )
        appended = _finite(
            probe.get("synthetic_appended_padding_max_valid_error"),
            label=f"{name} appended-padding error",
            minimum=0.0,
        )
        perturbed = _finite(
            probe.get(
                "synthetic_invalid_value_perturbation_max_valid_error"
            ),
            label=f"{name} invalid-padding perturbation error",
            minimum=0.0,
        )
        synthetic_tolerance = _finite(
            probe.get("synthetic_padding_tolerance"),
            label=f"{name} synthetic padding tolerance",
            minimum=0.0,
        )
        synthetic = (
            type(probe.get("synthetic_invalid_padding_slots")) is int
            and probe["synthetic_invalid_padding_slots"] >= 2
            and appended <= synthetic_tolerance
            and perturbed <= synthetic_tolerance
        )
        if (
            probe.get("future_slot_causality_passed") is not causal
            or probe.get("batching_equivalence_passed") is not batching
            or probe.get("synthetic_padding_passed") is not synthetic
            or probe.get("passed") is not (
                causal and batching and synthetic
            )
            or probe.get("passed") is not True
        ):
            raise ValueError(
                f"{name} structural probes do not recompute"
            )


def load_gemma3_full_width_single_layer_artifact(
    path: Path | str,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, object]:
    """Strictly restore an artifact and recompute its scientific decisions."""

    source = Path(path)
    raw = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
    )
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "contains_executor_weights",
        "contains_prompt_text",
        "contains_tokenizer_state",
        "scientific_status",
        "model",
        "protocol",
        "executors",
        "training",
        "selection",
        "validation",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("single-layer artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
        or raw["contains_model_weights"] is not False
        or raw["contains_executor_weights"] is not True
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
        or not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("single-layer artifact header is invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    digest = _scientific_payload_sha256(payload)
    if digest != raw["scientific_payload_sha256"]:
        raise ValueError("single-layer scientific payload digest mismatch")
    _validate_tensor_locations(payload)

    model = _validate_model_metadata(raw["model"])
    protocol = _strict_mapping(
        raw["protocol"],
        label="single-layer protocol",
        fields={
            "layer_index",
            "layer_ids",
            "canonical_boundaries",
            "residual_width",
            "retained_rank",
            "rank_reduction_attempted",
            "executor_architecture",
            "maximum_tokenized_length",
            "tokenization_batch_size",
            "optimization_seed",
            "single_seed_protocol",
            "numeric_policy",
            "source_attention_visibility",
            "source_accounting_manifest",
            "training_split",
            "selection_policy",
            "validation_policy",
            "test_policy",
            "student_execution",
            "native_layer_output_available_to_student",
            "thresholds",
            "data_minima",
            "strong_data_minima_enforced",
            "prompt_splits",
            "prompt_families",
            "prompt_exclusions",
            "prompt_fixture_file_sha256",
            "family_manifest_file_sha256",
            "tokenized_splits",
            "tokenized_stream_contracts",
            "tokenized_content_disjointness",
            "library_versions",
            "tokenizer",
            "model_state_guard",
            "source_independence",
        },
    )
    status = _strict_mapping(
        raw["scientific_status"],
        label="single-layer scientific status",
        fields={
            "scope",
            "rank_reduction_attempted",
            "retained_rank",
            "residual_width",
            "activation_fisher_computed",
            "calibration_a_executors_fitted",
            "calibration_b_evaluated",
            "calibration_b_passed",
            "validation_evaluated",
            "validation_passed",
            "test_evaluated",
            "source_layer_calls_in_student_path",
            "source_layer_removed_from_student_path",
            "source_independent_executor",
            "model_weights_changed",
            "model_weights_in_artifact",
            "candidate_passed_recorded_single_seed_protocol",
            "strong_data_minima_enforced",
            "immutable_model_revision_recorded",
            "fidelity_viable_replacement",
            "model_level_promotion_authorized",
            "parameter_reduction_supported",
            "analytic_mac_reduction_supported",
            "parameter_reduction_observed_under_recorded_protocol",
            "analytic_mac_reduction_observed_under_recorded_protocol",
            "source_resource_denominators_reverified_on_load",
            "causal_edge_value_identified",
            "causal_edge_threshold_separation_observed",
            "latency_or_kernel_speed_claim",
        },
    )
    training = _strict_mapping(
        raw["training"],
        label="single-layer training",
        fields={
            "activation_fisher",
            "delta_scale",
            "full_causal",
            "same_position_control",
            "structural_probes",
            "tokenized_stream",
        },
    )
    selection = _strict_mapping(
        raw["selection"],
        label="single-layer selection",
        fields={
            "behavior",
            "direct",
            "direct_gates",
            "execution_audits",
            "native_boundary_replay",
            "behavior_gates",
            "resource_gates",
            "accounting",
            "full_causal_passed",
            "same_position_control_passed",
            "causal_edge_threshold_separation_observed",
            "locked_candidate",
            "tokenized_stream",
            "tokenized_stream_contract",
        },
    )
    validation = _strict_mapping(
        raw["validation"],
        label="single-layer validation",
        fields={
            "evaluated",
            "reason",
            "behavior",
            "direct",
            "direct_gates",
            "execution_audits",
            "native_boundary_replay",
            "behavior_gates",
            "resource_gates",
            "accounting",
            "passed",
            "tokenized_stream",
            "tokenized_stream_contract",
        },
    )

    raw_executors = raw["executors"]
    if (
        not isinstance(raw_executors, Mapping)
        or set(raw_executors)
        != {"full_causal", "same_position_control"}
    ):
        raise ValueError("single-layer executor candidates are invalid")
    executors = {
        name: FullWidthSingleLayerExecutor.from_artifact_state_dict(
            state,  # type: ignore[arg-type]
            map_location=map_location,
        )
        for name, state in raw_executors.items()
    }
    if (
        not executors["full_causal"].config.causal_edges_enabled
        or executors[
            "same_position_control"
        ].config.causal_edges_enabled
        or executors["full_causal"].width
        != executors["same_position_control"].width
    ):
        raise ValueError("single-layer candidate semantics are invalid")

    width = executors["full_causal"].width
    architecture = _strict_mapping(
        protocol["executor_architecture"],
        label="single-layer executor architecture",
        fields={
            "hidden_width",
            "layer_count",
            "head_count",
            "feed_forward_width",
            "full_causal_and_storage_matched_attention_ablation",
        },
    )
    full_config = executors["full_causal"].config
    control_config = executors["same_position_control"].config
    layer_ids = protocol["layer_ids"]
    boundaries = protocol["canonical_boundaries"]
    if (
        type(protocol["layer_index"]) is not int
        or protocol["layer_index"] < 0
        or not isinstance(layer_ids, tuple)
        or len(layer_ids) != 1
        or not all(isinstance(item, str) and item for item in layer_ids)
        or not isinstance(boundaries, tuple)
        or len(boundaries) != 2
        or protocol.get("residual_width") != width
        or protocol.get("retained_rank") != width
        or protocol.get("rank_reduction_attempted") is not False
        or model.get("hidden_size") not in {None, width}
        or architecture["hidden_width"] != full_config.hidden_width
        or architecture["layer_count"] != full_config.layer_count
        or architecture["head_count"] != full_config.head_count
        or architecture["feed_forward_width"]
        != full_config.feed_forward_width
        or architecture[
            "full_causal_and_storage_matched_attention_ablation"
        ]
        is not True
        or asdict(full_config)
        != {
            **asdict(control_config),
            "causal_edges_enabled": True,
        }
        or status.get("retained_rank") != width
        or status.get("residual_width") != width
        or status.get("rank_reduction_attempted") is not False
    ):
        raise ValueError(
            "single-layer geometry or architecture binding is invalid"
        )

    numeric = _strict_mapping(
        protocol["numeric_policy"],
        label="single-layer numeric policy",
        fields={
            "requested_device",
            "resolved_device",
            "requested_model_dtype",
            "resolved_model_dtype",
            "executor_parameter_dtype",
            "fisher_accumulation_dtype",
            "local_fisher_quadratic_dtype",
        },
    )
    if (
        type(protocol["optimization_seed"]) is not int
        or protocol["optimization_seed"] < 0
        or type(protocol["tokenization_batch_size"]) is not int
        or protocol["tokenization_batch_size"] <= 0
        or protocol["single_seed_protocol"] is not True
        or numeric["resolved_model_dtype"] != model["dtype"]
        or numeric["executor_parameter_dtype"] != "torch.float32"
        or numeric["fisher_accumulation_dtype"] != "torch.float64"
        or numeric["local_fisher_quadratic_dtype"] != "torch.float32"
        or protocol["training_split"] != "calibration_a_only"
        or protocol["test_policy"] != "parse_validate_hash_only"
        or protocol["native_layer_output_available_to_student"] is not False
    ):
        raise ValueError("single-layer numeric or split policy is invalid")

    maximum_length = protocol["maximum_tokenized_length"]
    visibility = _strict_mapping(
        protocol["source_attention_visibility"],
        label="single-layer attention visibility",
        fields={
            "source_attention_kind",
            "source_window_size",
            "student_attention_kind",
            "maximum_tokenized_length",
            "visibility_equivalence",
            "prefill_only",
            "decode_or_cache_claim",
            "rope_equivalence_claim",
            "nonzero_position_offset_claim",
            "passed",
        },
    )
    source_kind = visibility["source_attention_kind"]
    if (
        type(maximum_length) is not int
        or maximum_length < 2
        or visibility["maximum_tokenized_length"] != maximum_length
        or visibility["student_attention_kind"] != "global_causal"
        or source_kind not in {"global_causal", "sliding_causal"}
        or visibility["prefill_only"] is not True
        or visibility["decode_or_cache_claim"] is not False
        or visibility["rope_equivalence_claim"] is not False
        or visibility["nonzero_position_offset_claim"] is not False
        or visibility["passed"] is not True
        or (
            source_kind == "sliding_causal"
            and (
                type(visibility["source_window_size"]) is not int
                or visibility["source_window_size"] < maximum_length
            )
        )
        or (
            source_kind == "global_causal"
            and visibility["source_window_size"] is not None
        )
    ):
        raise ValueError("single-layer attention visibility is invalid")
    source_manifest = _validate_source_accounting_manifest(
        protocol["source_accounting_manifest"],
        layer_ids=layer_ids,
        visibility=visibility,
    )

    thresholds = _strict_mapping(
        protocol["thresholds"],
        label="single-layer thresholds",
        fields={
            "nll_atol",
            "top1_min",
            "teacher_kl_max",
            "p90_abs_nll_max",
            "p10_top1_min",
            "block_delta_nrmse_max",
            "block_delta_cosine_min",
            "max_stored_coefficient_ratio",
            "max_analytic_mac_ratio",
        },
    )
    for name, lower, upper in (
        ("nll_atol", 0.0, None),
        ("top1_min", 0.0, 1.0),
        ("teacher_kl_max", 0.0, None),
        ("p90_abs_nll_max", 0.0, None),
        ("p10_top1_min", 0.0, 1.0),
        ("block_delta_nrmse_max", 0.0, None),
        ("block_delta_cosine_min", -1.0, 1.0),
        ("max_stored_coefficient_ratio", 0.0, None),
        ("max_analytic_mac_ratio", 0.0, None),
    ):
        _finite(
            thresholds[name],
            label=f"single-layer threshold {name}",
            minimum=lower,
            maximum=upper,
        )

    minima = _strict_mapping(
        protocol["data_minima"],
        label="single-layer data minima",
        fields={
            "minimum_calibration_a_prompts",
            "minimum_heldout_prompts_per_role",
            "minimum_fisher_rows",
            "minimum_train_supervised_tokens",
            "minimum_heldout_supervised_tokens_per_role",
            "minimum_populated_length_buckets_per_tokenized_role",
        },
    )
    if any(type(value) is not int or value <= 0 for value in minima.values()):
        raise ValueError("single-layer data minima are invalid")
    expected_strong_minima = (
        minima["minimum_calibration_a_prompts"]
        >= DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
        and minima["minimum_heldout_prompts_per_role"]
        >= DEFAULT_MINIMUM_HELDOUT_PROMPTS
        and minima["minimum_fisher_rows"] >= DEFAULT_MINIMUM_FISHER_ROWS
        and minima["minimum_train_supervised_tokens"]
        >= DEFAULT_MINIMUM_TRAIN_SUPERVISED_TOKENS
        and minima["minimum_heldout_supervised_tokens_per_role"]
        >= DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
        and minima[
            "minimum_populated_length_buckets_per_tokenized_role"
        ]
        >= DEFAULT_MINIMUM_LENGTH_BUCKETS
    )
    if (
        protocol["strong_data_minima_enforced"]
        is not expected_strong_minima
    ):
        raise ValueError("single-layer strong-minima status is invalid")

    prompt_splits = _strict_mapping(
        protocol["prompt_splits"],
        label="single-layer prompt split provenance",
        fields={
            "scientific_status",
            "counts",
            "normalized_sha256",
            "per_prompt_sha256",
        },
    )
    counts = _strict_mapping(
        prompt_splits["counts"],
        label="single-layer prompt counts",
        fields=set(_SPLIT_NAMES),
    )
    normalized_hashes = _strict_mapping(
        prompt_splits["normalized_sha256"],
        label="single-layer normalized prompt hashes",
        fields=set(_SPLIT_NAMES),
    )
    prompt_hashes = _strict_mapping(
        prompt_splits["per_prompt_sha256"],
        label="single-layer per-prompt hashes",
        fields=set(_SPLIT_NAMES),
    )
    all_prompt_hashes = []
    for split_name in _SPLIT_NAMES:
        hashes = prompt_hashes[split_name]
        minimum_count = (
            minima["minimum_calibration_a_prompts"]
            if split_name == "calibration_a"
            else minima["minimum_heldout_prompts_per_role"]
        )
        if (
            type(counts[split_name]) is not int
            or counts[split_name] < minimum_count
            or not isinstance(hashes, list)
            or len(hashes) != counts[split_name]
            or any(not _is_sha256(item) for item in hashes)
            or normalized_hashes[split_name]
            != _ordered_prompt_hash_digest(hashes)
        ):
            raise ValueError("single-layer prompt provenance is invalid")
        all_prompt_hashes.extend(hashes)
    if (
        prompt_splits["scientific_status"] != PROMPT_STATUS
        or len(set(all_prompt_hashes)) != len(all_prompt_hashes)
        or not _is_sha256(protocol["prompt_fixture_file_sha256"])
        or not _is_sha256(protocol["family_manifest_file_sha256"])
    ):
        raise ValueError("single-layer prompt disjointness is invalid")

    families = _strict_mapping(
        protocol["prompt_families"],
        label="single-layer prompt families",
        fields={
            "scientific_status",
            "counts",
            "unique_family_counts",
            "ordered_family_sha256",
            "cross_role_overlap_count",
        },
    )
    if (
        families["scientific_status"] != FAMILY_STATUS
        or families["counts"] != counts
        or families["cross_role_overlap_count"] != 0
    ):
        raise ValueError("single-layer prompt family provenance is invalid")
    for field in ("unique_family_counts", "ordered_family_sha256"):
        mapping = _strict_mapping(
            families[field],
            label=f"single-layer family {field}",
            fields=set(_SPLIT_NAMES),
        )
        for split_name in _SPLIT_NAMES:
            if (
                field == "unique_family_counts"
                and (
                    type(mapping[split_name]) is not int
                    or not 1
                    <= mapping[split_name]
                    <= counts[split_name]
                )
            ) or (
                field == "ordered_family_sha256"
                and not _is_sha256(mapping[split_name])
            ):
                raise ValueError(
                    "single-layer prompt family metadata is invalid"
                )

    exclusions = _strict_mapping(
        protocol["prompt_exclusions"],
        label="single-layer prompt exclusions",
    )
    if (
        exclusions.get("overlap_count") != 0
        or exclusions.get(
            "verified_before_model_load_or_tokenization"
        )
        is not True
    ):
        raise ValueError("single-layer prompt exclusions are invalid")

    validation_evaluated = validation["evaluated"]
    if type(validation_evaluated) is not bool:
        raise ValueError("single-layer validation flag is invalid")
    expected_stream_names = ("calibration_a", "calibration_b")
    if validation_evaluated:
        expected_stream_names = (*expected_stream_names, "validation")
    raw_streams = _strict_mapping(
        protocol["tokenized_splits"],
        label="single-layer tokenized splits",
        fields=set(expected_stream_names),
    )
    streams = {
        split_name: _validated_tokenized_stream(
            raw_streams[split_name],
            split_name=split_name,
        )[0]
        for split_name in expected_stream_names
    }
    if (
        training["tokenized_stream"] != streams["calibration_a"]
        or selection["tokenized_stream"] != streams["calibration_b"]
        or (
            validation_evaluated
            and validation["tokenized_stream"] != streams["validation"]
        )
        or (
            not validation_evaluated
            and validation["tokenized_stream"] is not None
        )
    ):
        raise ValueError("single-layer duplicated streams are inconsistent")
    for split_name, stream in streams.items():
        if (
            stream["source_prompt_sha256"]
            != prompt_hashes[split_name]
        ):
            raise ValueError(
                "single-layer tokenized stream prompt binding is invalid"
            )
    streamed_prompt_hashes = {
        item
        for stream in streams.values()
        for item in stream["source_prompt_sha256"]
    }
    if streamed_prompt_hashes & set(prompt_hashes["test"]):
        raise ValueError("reserved test prompts were tokenized")

    raw_contracts = _strict_mapping(
        protocol["tokenized_stream_contracts"],
        label="single-layer tokenized stream contracts",
        fields=set(expected_stream_names),
    )
    recomputed_contracts = {}
    for split_name, stream in streams.items():
        minimum_tokens = (
            minima["minimum_train_supervised_tokens"]
            if split_name == "calibration_a"
            else minima[
                "minimum_heldout_supervised_tokens_per_role"
            ]
        )
        recomputed_contracts[split_name] = _tokenized_stream_contract(
            stream,
            split_name=split_name,
            minimum_supervised_tokens=minimum_tokens,
            minimum_length_buckets=minima[
                "minimum_populated_length_buckets_per_tokenized_role"
            ],
        )
    if (
        not _semantic_numeric_equal(raw_contracts, recomputed_contracts)
        or selection["tokenized_stream_contract"]
        != recomputed_contracts["calibration_b"]
        or (
            validation_evaluated
            and validation["tokenized_stream_contract"]
            != recomputed_contracts["validation"]
        )
        or (
            not validation_evaluated
            and validation["tokenized_stream_contract"] is not None
        )
    ):
        raise ValueError(
            "single-layer tokenized stream contracts do not recompute"
        )
    expected_content_audit = _assert_tokenized_content_disjointness(
        streams
    )
    if (
        protocol["tokenized_content_disjointness"]
        != expected_content_audit
    ):
        raise ValueError(
            "single-layer tokenized-content audit does not recompute"
        )

    fisher = _strict_mapping(
        training["activation_fisher"],
        label="single-layer activation Fisher",
        fields={
            "estimator",
            "expected_model_fisher_claim",
            "cross_position_blocks_included",
            "training_uses_full_matrix",
            "rows",
            "width",
            "trace",
            "minimum_eigenvalue",
            "maximum_eigenvalue",
            "rank_for_90_percent_trace",
            "rank_for_99_percent_trace",
            "rank_for_99_9_percent_trace",
            "diagonal_floor_before_renormalization",
            "normalized_diagonal_min",
            "normalized_diagonal_max",
            "normalized_diagonal_mean",
            "matrix_sha256",
            "normalized_diagonal_sha256",
            "matrix",
            "normalized_diagonal",
            "training_metric",
            "training_metric_report",
        },
    )
    matrix = _strict_float64_tensor(
        fisher["matrix"],
        label="single-layer Fisher matrix",
        shape=(width, width),
    )
    normalized_diagonal = _strict_float64_tensor(
        fisher["normalized_diagonal"],
        label="single-layer normalized Fisher diagonal",
        shape=(width,),
    )
    metric = _strict_float64_tensor(
        fisher["training_metric"],
        label="single-layer Fisher training metric",
        shape=(width, width),
    )
    delta = _strict_mapping(
        training["delta_scale"],
        label="single-layer delta scale",
        fields={
            "estimator",
            "rows",
            "floor",
            "minimum",
            "maximum",
            "mean",
            "sha256",
            "values",
        },
    )
    delta_values = _strict_float64_tensor(
        delta["values"],
        label="single-layer delta scale values",
        shape=(width,),
    )
    if (
        not torch.equal(matrix, matrix.T)
        or not torch.equal(metric, metric.T)
        or not (delta_values > 0).all()
        or fisher["width"] != width
        or type(fisher["rows"]) is not int
        or fisher["rows"] < minima["minimum_fisher_rows"]
        or delta["rows"] != fisher["rows"]
        or fisher["estimator"]
        != (
            "width_pooled_empirical_ground_truth_ce_score_sensitivity_"
            "from_selected_supervised_targets"
        )
        or fisher["expected_model_fisher_claim"] is not False
        or fisher["cross_position_blocks_included"] is not False
        or fisher["training_uses_full_matrix"] is not True
        or fisher["matrix_sha256"]
        != _tensor_sha256(
            matrix,
            domain=(
                b"fisher_graph.full_width_layer.fisher_matrix.v1\0"
            ),
        )
        or fisher["normalized_diagonal_sha256"]
        != _tensor_sha256(
            normalized_diagonal,
            domain=(
                b"fisher_graph.full_width_layer.fisher_diagonal.v1\0"
            ),
        )
        or delta["sha256"]
        != _tensor_sha256(
            delta_values,
            domain=b"fisher_graph.full_width_layer.delta_scale.v1\0",
        )
    ):
        raise ValueError("single-layer Fisher tensor binding is invalid")
    eigenvalues = torch.linalg.eigvalsh(matrix)
    maximum_eigenvalue = float(eigenvalues.max().item())
    if float(eigenvalues.min().item()) < -max(
        1e-12,
        abs(maximum_eigenvalue) * 1e-10,
    ):
        raise ValueError("single-layer Fisher matrix is not PSD")
    diagonal_floor = _finite(
        fisher["diagonal_floor_before_renormalization"],
        label="single-layer Fisher diagonal floor",
        minimum=torch.finfo(torch.float64).tiny,
    )
    raw_diagonal = torch.diagonal(matrix)
    expected_diagonal = (
        raw_diagonal / raw_diagonal.mean()
    ).clamp_min(diagonal_floor)
    expected_diagonal /= expected_diagonal.mean()
    if not torch.allclose(
        normalized_diagonal,
        expected_diagonal,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "single-layer normalized Fisher diagonal is invalid"
        )
    expected_metric, expected_metric_report = _normalized_fisher_metric(
        matrix,
        delta_values,
        eigenvalue_floor=diagonal_floor,
    )
    trace = float(eigenvalues.sum().item())
    descending = eigenvalues.clamp_min(0).flip(0)
    cumulative = descending.cumsum(0)

    def expected_capture_rank(fraction: float) -> int:
        if trace <= torch.finfo(torch.float64).tiny:
            return width
        return (
            int(
                torch.searchsorted(
                    cumulative,
                    fraction * trace,
                ).item()
            )
            + 1
        )

    if (
        not torch.allclose(
            metric,
            expected_metric,
            rtol=1e-11,
            atol=1e-13,
        )
        or not _semantic_numeric_equal(
            fisher["training_metric_report"],
            expected_metric_report,
        )
        or not math.isclose(
            float(fisher["trace"]),
            trace,
            rel_tol=1e-11,
            abs_tol=1e-14,
        )
        or fisher["rank_for_90_percent_trace"]
        != expected_capture_rank(0.90)
        or fisher["rank_for_99_percent_trace"]
        != expected_capture_rank(0.99)
        or fisher["rank_for_99_9_percent_trace"]
        != expected_capture_rank(0.999)
        or not math.isclose(
            float(fisher["normalized_diagonal_min"]),
            float(normalized_diagonal.min().item()),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(fisher["normalized_diagonal_max"]),
            float(normalized_diagonal.max().item()),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(fisher["normalized_diagonal_mean"]),
            float(normalized_diagonal.mean().item()),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(fisher["minimum_eigenvalue"]),
            float(eigenvalues.min().item()),
            rel_tol=1e-11,
            abs_tol=1e-14,
        )
        or not math.isclose(
            float(fisher["maximum_eigenvalue"]),
            maximum_eigenvalue,
            rel_tol=1e-11,
            abs_tol=1e-14,
        )
        or not math.isclose(
            float(delta["minimum"]),
            float(delta_values.min().item()),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(delta["maximum"]),
            float(delta_values.max().item()),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(delta["mean"]),
            float(delta_values.mean().item()),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("single-layer Fisher statistics are invalid")

    for name in ("full_causal", "same_position_control"):
        training_report = _strict_mapping(
            training[name],
            label=f"{name} training report",
        )
        if (
            type(training_report.get("local_warmup_steps")) is not int
            or type(training_report.get("downstream_train_steps")) is not int
            or training_report.get("total_steps")
            != training_report["local_warmup_steps"]
            + training_report["downstream_train_steps"]
            or training_report.get("fixed_update_schedule") is not True
            or training_report.get("local_mse_position_scope")
            != "all_valid_rows"
            or training_report.get(
                "local_fisher_quadratic_position_scope"
            )
            != "deterministically_selected_supervised_rows"
            or training_report.get("checkpoint_selection")
            != "final_fixed_step"
            or training_report.get("early_stopping") is not False
            or training_report.get(
                "source_parameter_gradients_observed"
            )
            is not False
        ):
            raise ValueError(f"{name} training report is invalid")
    _validate_structural_probes(training["structural_probes"])

    independence = _strict_mapping(
        protocol["source_independence"],
        label="single-layer source independence",
    )
    independence_candidates = independence.get("candidates")
    if (
        independence.get("passed") is not True
        or not isinstance(independence_candidates, Mapping)
        or set(independence_candidates)
        != {"full_causal", "same_position_control"}
        or any(
            not isinstance(candidate, Mapping)
            or candidate.get("parameter_object_alias_count") != 0
            or candidate.get("module_object_alias_count") != 0
            or candidate.get("tensor_storage_alias_count") != 0
            or candidate.get("passed") is not True
            for candidate in independence_candidates.values()
        )
    ):
        raise ValueError("single-layer source-independence audit is invalid")

    candidate_names = {"full_causal", "same_position_control"}
    aggregate_names = candidate_names | {
        "native_boundary_replay",
        "identity_layer_skip",
    }

    def validate_evaluated_section(
        section: Mapping[str, object],
        *,
        stream: Mapping[str, object],
        label: str,
    ) -> tuple[
        bool,
        bool,
        dict[str, Mapping[str, object]],
    ]:
        behavior = _strict_mapping(
            section["behavior"],
            label=f"{label} behavior candidates",
            fields=aggregate_names,
        )
        direct = _strict_mapping(
            section["direct"],
            label=f"{label} direct candidates",
            fields=aggregate_names,
        )
        behavior_values = {
            name: _validate_behavior_aggregate(
                behavior[name],
                stream=stream,
                label=f"{label} {name}",
            )
            for name in aggregate_names
        }
        direct_values = {
            name: _validate_direct_aggregate(
                direct[name],
                stream=stream,
                width=width,
                label=f"{label} {name}",
            )
            for name in aggregate_names
        }
        behavior_gates = _strict_mapping(
            section["behavior_gates"],
            label=f"{label} behavior gates",
            fields=candidate_names,
        )
        direct_gates = _strict_mapping(
            section["direct_gates"],
            label=f"{label} direct gates",
            fields=candidate_names,
        )
        for name in candidate_names:
            expected_behavior_gates = _behavior_gates(
                behavior_values[name],
                nll_atol=float(thresholds["nll_atol"]),
                top1_min=float(thresholds["top1_min"]),
                teacher_kl_max=float(thresholds["teacher_kl_max"]),
                p90_abs_nll_max=float(
                    thresholds["p90_abs_nll_max"]
                ),
                p10_top1_min=float(thresholds["p10_top1_min"]),
            )
            expected_direct_gates = _direct_gates(
                direct_values[name],
                block_delta_nrmse_max=float(
                    thresholds["block_delta_nrmse_max"]
                ),
                block_delta_cosine_min=float(
                    thresholds["block_delta_cosine_min"]
                ),
            )
            if (
                behavior_gates[name] != expected_behavior_gates
                or direct_gates[name] != expected_direct_gates
            ):
                raise ValueError(f"{label} gates do not recompute")
        audits = _strict_mapping(
            section["execution_audits"],
            label=f"{label} execution audits",
            fields=candidate_names,
        )
        for name in candidate_names:
            _validate_execution_audit(
                audits[name],
                layer_ids=layer_ids,
                label=f"{label} {name}",
            )
        _validate_native_replay(
            section["native_boundary_replay"],
            label=label,
        )
        raw_accounting = _strict_mapping(
            section["accounting"],
            label=f"{label} accounting candidates",
            fields=candidate_names,
        )
        accounting = {
            name: _validate_candidate_accounting(
                raw_accounting[name],
                executor=executors[name],
                stream=stream,
                tokenization_batch_size=int(
                    protocol["tokenization_batch_size"]
                ),
                source_manifest=source_manifest,
                label=f"{label} {name}",
            )
            for name in candidate_names
        }
        resource_gates = _strict_mapping(
            section["resource_gates"],
            label=f"{label} resource gates",
            fields={
                "stored_coefficient_ratio",
                "analytic_mac_ratio",
            },
        )
        expected_resource = _resource_gates(
            accounting["full_causal"],
            max_stored_coefficient_ratio=float(
                thresholds["max_stored_coefficient_ratio"]
            ),
            max_analytic_mac_ratio=float(
                thresholds["max_analytic_mac_ratio"]
            ),
        )
        if resource_gates != expected_resource:
            raise ValueError(f"{label} resource gates do not recompute")
        full_passed = (
            all(behavior_gates["full_causal"].values())
            and all(direct_gates["full_causal"].values())
            and all(resource_gates.values())
        )
        control_passed = (
            all(behavior_gates["same_position_control"].values())
            and all(direct_gates["same_position_control"].values())
        )
        return full_passed, control_passed, accounting

    selection_passed, control_passed, selection_accounting = (
        validate_evaluated_section(
            selection,
            stream=streams["calibration_b"],
            label="selection",
        )
    )
    expected_separation = selection_passed and not control_passed
    if (
        selection["full_causal_passed"] is not selection_passed
        or selection["same_position_control_passed"] is not control_passed
        or selection["causal_edge_threshold_separation_observed"]
        is not expected_separation
        or selection["locked_candidate"]
        != ("full_causal" if selection_passed else None)
    ):
        raise ValueError("single-layer selection decision is invalid")

    validation_accounting: dict[str, Mapping[str, object]] | None = None
    if selection_passed:
        if validation_evaluated is not True:
            raise ValueError(
                "passing selection must have evaluated validation"
            )
        (
            validation_passed,
            _validation_control_passed,
            validation_accounting,
        ) = validate_evaluated_section(
            validation,
            stream=streams["validation"],
            label="validation",
        )
        if (
            validation["reason"]
            != "calibration_b_passed_locked_executors_evaluated"
            or validation["passed"] is not validation_passed
        ):
            raise ValueError("single-layer validation decision is invalid")
    else:
        validation_passed = False
        if (
            validation_evaluated is not False
            or validation["reason"]
            != "calibration_b_failed_validation_not_tokenized"
            or validation["passed"] is not False
            or any(
                validation[field] is not None
                for field in (
                    "behavior",
                    "direct",
                    "direct_gates",
                    "execution_audits",
                    "native_boundary_replay",
                    "behavior_gates",
                    "resource_gates",
                    "accounting",
                    "tokenized_stream",
                    "tokenized_stream_contract",
                )
            )
        ):
            raise ValueError("unevaluated validation contains results")

    recorded_protocol_passed = selection_passed and validation_passed
    resolved_commit = model.get("resolved_commit")
    expected_immutable_revision = (
        isinstance(resolved_commit, str)
        and re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved_commit)
        is not None
    )
    if (
        status["scope"]
        != "full_width_source_independent_single_gemma_layer_replacement"
        or status["activation_fisher_computed"] is not True
        or status["calibration_a_executors_fitted"] is not True
        or status["calibration_b_evaluated"] is not True
        or status["calibration_b_passed"] is not selection_passed
        or status["validation_evaluated"] is not validation_evaluated
        or status["validation_passed"] is not validation_passed
        or status["test_evaluated"] is not False
        or status["source_layer_calls_in_student_path"] != 0
        or status["source_layer_removed_from_student_path"] is not True
        or status["source_independent_executor"] is not True
        or status["model_weights_changed"] is not False
        or status["model_weights_in_artifact"] is not False
        or status[
            "candidate_passed_recorded_single_seed_protocol"
        ]
        is not recorded_protocol_passed
        or status["strong_data_minima_enforced"]
        is not expected_strong_minima
        or status["immutable_model_revision_recorded"]
        is not expected_immutable_revision
        or status["fidelity_viable_replacement"] is not False
        or status["model_level_promotion_authorized"] is not False
        or status["parameter_reduction_supported"] is not False
        or status["analytic_mac_reduction_supported"] is not False
        or status[
            "parameter_reduction_observed_under_recorded_protocol"
        ]
        is not False
        or status[
            "analytic_mac_reduction_observed_under_recorded_protocol"
        ]
        is not False
        or status["source_resource_denominators_reverified_on_load"]
        is not False
        or status["causal_edge_value_identified"] is not False
        or status["causal_edge_threshold_separation_observed"]
        is not expected_separation
        or status["latency_or_kernel_speed_claim"] is not False
    ):
        raise ValueError("single-layer scientific status is invalid")

    report_path = source.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("single-layer JSON report is invalid")
    report_artifact = _strict_mapping(
        report.get("artifact"),
        label="single-layer report artifact",
    )
    tensor_file = report_artifact.get("tensor_file")
    if not isinstance(tensor_file, str) or not tensor_file:
        raise ValueError("single-layer report tensor file is invalid")
    expected_report = _build_report(
        payload,
        tensor_file=tensor_file,
        scientific_digest=digest,
    )
    if (
        _report_sha256(report) != raw["report_sha256"]
        or report
        != json.loads(
            json.dumps(
                expected_report,
                sort_keys=True,
                allow_nan=False,
            )
        )
    ):
        raise ValueError("single-layer JSON report is invalid")
    return {
        "model": copy.deepcopy(model),
        "protocol": copy.deepcopy(protocol),
        "executors": executors,
        "training": copy.deepcopy(raw["training"]),
        "selection": copy.deepcopy(selection),
        "validation": copy.deepcopy(validation),
        "scientific_status": copy.deepcopy(status),
        "metadata": {
            "scientific_payload_sha256": digest,
            "report_sha256": raw["report_sha256"],
            "tensor_file_sha256": _file_sha256(source),
        },
        "report": copy.deepcopy(report),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and gate a full-width source-free replacement for one "
            "Gemma 3 text-decoder layer."
        )
    )
    parser.add_argument("--prompt-splits", type=Path, required=True)
    parser.add_argument("--family-manifest", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--tokenization-batch-size", type=int, default=1)
    parser.add_argument(
        "--hidden-width",
        type=int,
        default=DEFAULT_HIDDEN_WIDTH,
    )
    parser.add_argument(
        "--executor-layers",
        type=int,
        default=DEFAULT_EXECUTOR_LAYERS,
    )
    parser.add_argument(
        "--head-count",
        type=int,
        default=DEFAULT_HEAD_COUNT,
    )
    parser.add_argument(
        "--feed-forward-width",
        type=int,
        default=DEFAULT_FEED_FORWARD_WIDTH,
    )
    parser.add_argument(
        "--local-warmup-steps",
        type=int,
        default=DEFAULT_LOCAL_WARMUP_STEPS,
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_TRAIN_STEPS,
    )
    parser.add_argument(
        "--train-positions-per-sequence",
        type=int,
        default=DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
    )
    parser.add_argument(
        "--local-mse-weight",
        type=float,
        default=DEFAULT_LOCAL_MSE_WEIGHT,
    )
    parser.add_argument(
        "--local-fisher-weight",
        type=float,
        default=DEFAULT_LOCAL_FISHER_WEIGHT,
    )
    parser.add_argument(
        "--ground-truth-weight",
        type=float,
        default=DEFAULT_GROUND_TRUTH_WEIGHT,
    )
    parser.add_argument(
        "--teacher-kl-weight",
        type=float,
        default=DEFAULT_TEACHER_KL_WEIGHT,
    )
    parser.add_argument(
        "--selection-nll-atol",
        type=float,
        default=DEFAULT_NLL_ATOL,
    )
    parser.add_argument(
        "--selection-top1-min",
        type=float,
        default=DEFAULT_TOP1_MIN,
    )
    parser.add_argument(
        "--selection-teacher-kl-max",
        type=float,
        default=DEFAULT_TEACHER_KL_MAX,
    )
    parser.add_argument(
        "--selection-p90-abs-nll-max",
        type=float,
        default=DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
    )
    parser.add_argument(
        "--selection-p10-top1-min",
        type=float,
        default=DEFAULT_PER_PROMPT_P10_TOP1_MIN,
    )
    parser.add_argument(
        "--block-delta-nrmse-max",
        type=float,
        default=DEFAULT_BLOCK_DELTA_NRMSE_MAX,
    )
    parser.add_argument(
        "--block-delta-cosine-min",
        type=float,
        default=DEFAULT_BLOCK_DELTA_COSINE_MIN,
    )
    parser.add_argument(
        "--max-stored-coefficient-ratio",
        type=float,
        default=DEFAULT_MAX_STORED_COEFFICIENT_RATIO,
    )
    parser.add_argument(
        "--max-analytic-mac-ratio",
        type=float,
        default=DEFAULT_MAX_ANALYTIC_MAC_RATIO,
    )
    parser.add_argument("--seed", type=int, default=91_104)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma3_full_width_single_layer_experiment(
        prompt_splits_path=args.prompt_splits,
        family_manifest_path=args.family_manifest,
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        layer_index=args.layer_index,
        max_length=args.max_length,
        tokenization_batch_size=args.tokenization_batch_size,
        hidden_width=args.hidden_width,
        executor_layers=args.executor_layers,
        head_count=args.head_count,
        feed_forward_width=args.feed_forward_width,
        local_warmup_steps=args.local_warmup_steps,
        train_steps=args.train_steps,
        train_positions_per_sequence=args.train_positions_per_sequence,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        local_mse_weight=args.local_mse_weight,
        local_fisher_weight=args.local_fisher_weight,
        ground_truth_weight=args.ground_truth_weight,
        teacher_kl_weight=args.teacher_kl_weight,
        selection_nll_atol=args.selection_nll_atol,
        selection_top1_min=args.selection_top1_min,
        selection_teacher_kl_max=args.selection_teacher_kl_max,
        selection_p90_abs_nll_max=args.selection_p90_abs_nll_max,
        selection_p10_top1_min=args.selection_p10_top1_min,
        block_delta_nrmse_max=args.block_delta_nrmse_max,
        block_delta_cosine_min=args.block_delta_cosine_min,
        max_stored_coefficient_ratio=args.max_stored_coefficient_ratio,
        max_analytic_mac_ratio=args.max_analytic_mac_ratio,
        seed=args.seed,
        device_name=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FAMILY_STATUS",
    "PROMPT_STATUS",
    "PromptFamilyManifest",
    "build_parser",
    "default_gemma3_full_width_single_layer_output",
    "load_gemma3_full_width_single_layer_artifact",
    "load_prompt_family_manifest",
    "main",
    "run_gemma3_full_width_single_layer_experiment",
]


if __name__ == "__main__":
    raise SystemExit(main())
