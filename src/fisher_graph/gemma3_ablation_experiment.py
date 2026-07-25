"""Validation-only full-width Fisher-modal ablations for Gemma 3.

This experiment fits complete pooled activation-Fisher bases on calibration
prompts, then projects selected native layer outputs jointly during a paired
validation replay.  It evaluates representational sufficiency in the original
frozen dense model; it does not fit an executor or claim compilation,
compression, speed, or memory savings.

Reserved test prompts are parsed and hashed but are never tokenized or passed
through the model.  The tensor artifact contains derived Fisher tensors and
metric ledgers only, never source-model weights, prompt text, or tokenizer
state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CausalLanguageModelNLL
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    make_causal_lm_calibration_batches,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_stability_experiment import (
    DEFAULT_PROMPT_SPLITS,
    _CalibrationStreamProvenance,
    _library_versions,
    _ordered_prompt_hash_digest,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .modal_ablation import (
    ModalAblationCondition,
    ModalAblationResult,
    PooledModalProjection,
    build_modal_ablation_conditions,
    evaluate_causal_lm_modal_ablation,
)
from .streaming_analysis import (
    StreamingFisherCollection,
    collect_streaming_fisher_modes,
)


DEFAULT_START_LAYER = 4
DEFAULT_END_LAYER = 6
DEFAULT_RANKS = (640, 512, 384, 256, 192, 128, 96, 64, 32, 0)
DEFAULT_FULL_RANK_NLL_ATOL = 1e-5
_ARTIFACT_SCHEMA = "fisher_graph.gemma3_modal_ablation"
_ARTIFACT_FORMAT_VERSION = 1
_FULL_TRACE_RTOL = 1e-8
_FULL_TRACE_ATOL = 1e-12


def default_gemma3_ablation_output(
    model_id: str = DEFAULT_MODEL_ID,
    start_layer: int = DEFAULT_START_LAYER,
    end_layer: int = DEFAULT_END_LAYER,
) -> Path:
    """Return an ignored model/block-specific ablation output path."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("layer range must be nonnegative and ascending")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("._-")
    if not slug:
        slug = "gemma3-model"
    return (
        Path(".local-runs")
        / slug
        / f"layers-{start_layer}-{end_layer}-modal-ablation.pt"
    )


def _validated_rank_schedule(
    ranks: Iterable[int],
    *,
    width: int,
) -> tuple[int, ...]:
    if isinstance(ranks, (str, bytes)):
        raise TypeError("ranks must be an iterable of nonnegative integers")
    try:
        requested = tuple(ranks)
    except TypeError as error:
        raise TypeError(
            "ranks must be an iterable of nonnegative integers"
        ) from error
    if not requested:
        raise ValueError("ranks cannot be empty")
    if any(
        type(rank) is not int or not 0 <= rank <= width
        for rank in requested
    ):
        raise ValueError(
            f"ranks must contain integers between 0 and width {width}"
        )
    canonical = tuple(dict.fromkeys(requested))
    if width not in canonical:
        raise ValueError(
            "rank schedule must include residual width as an identity control"
        )
    return canonical


class _FrozenModelTensorGuard:
    """Detect tensor replacement or in-place model-state mutation."""

    def __init__(self, model: nn.Module) -> None:
        if model.training:
            raise ValueError("modal ablation requires model.eval()")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(
                "modal ablation requires all source-model weights frozen"
            )
        self._model = model
        self._parameters = self._snapshot(
            model.named_parameters(remove_duplicate=False)
        )
        self._buffers = self._snapshot(
            model.named_buffers(remove_duplicate=False)
        )

    @staticmethod
    def _snapshot(
        tensors: Iterable[tuple[str, Tensor]],
    ) -> dict[str, tuple[Tensor, int, int]]:
        return {
            name: (
                tensor,
                int(tensor._version),
                int(tensor.untyped_storage().data_ptr()),
            )
            for name, tensor in tensors
        }

    @staticmethod
    def _assert_snapshot(
        *,
        label: str,
        expected: Mapping[str, tuple[Tensor, int, int]],
        actual: Iterable[tuple[str, Tensor]],
    ) -> None:
        current = dict(actual)
        if set(current) != set(expected):
            raise RuntimeError(f"source-model {label} names changed")
        for name, (original, version, data_ptr) in expected.items():
            tensor = current[name]
            if tensor is not original:
                raise RuntimeError(
                    f"source-model {label} object {name!r} was replaced"
                )
            if int(tensor._version) != version:
                raise RuntimeError(
                    f"source-model {label} {name!r} was mutated in place"
                )
            if int(tensor.untyped_storage().data_ptr()) != data_ptr:
                raise RuntimeError(
                    f"source-model {label} {name!r} storage was replaced"
                )

    def assert_unchanged(self) -> None:
        if self._model.training:
            raise RuntimeError("source model did not remain in eval mode")
        if any(
            parameter.requires_grad
            for parameter in self._model.parameters()
        ):
            raise RuntimeError("source-model parameters were unfrozen")
        self._assert_snapshot(
            label="parameter",
            expected=self._parameters,
            actual=self._model.named_parameters(remove_duplicate=False),
        )
        self._assert_snapshot(
            label="buffer",
            expected=self._buffers,
            actual=self._model.named_buffers(remove_duplicate=False),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "verified": True,
            "training": False,
            "parameters_frozen": True,
            "parameter_tensors": len(self._parameters),
            "buffer_tensors": len(self._buffers),
            "checks": (
                "tensor_object_identity",
                "tensor_version_counter",
                "tensor_storage_identity",
            ),
        }


def _update_payload_digest(digest: object, value: object) -> None:
    if not isinstance(digest, type(hashlib.sha256())):
        raise TypeError("digest must be a hashlib SHA-256 object")
    if value is None:
        digest.update(b"N;")
    elif isinstance(value, bool):
        digest.update(b"B1;" if value else b"B0;")
    elif type(value) is int:
        digest.update(f"I{value};".encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific payload floats must be finite")
        digest.update(f"F{value.hex()};".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"S{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        digest.update(b";")
    elif isinstance(value, Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError("scientific payload tensors must be finite")
        digest.update(b"T")
        _update_payload_digest(digest, str(tensor.dtype))
        _update_payload_digest(digest, tuple(tensor.shape))
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(f"{len(raw)}:".encode("ascii"))
        digest.update(raw)
        digest.update(b";")
    elif isinstance(value, Mapping):
        keys = sorted(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("scientific payload mapping keys must be strings")
        digest.update(f"M{len(keys)}[".encode("ascii"))
        for key in keys:
            _update_payload_digest(digest, key)
            _update_payload_digest(digest, value[key])
        digest.update(b"];")
    elif isinstance(value, tuple):
        digest.update(f"U{len(value)}[".encode("ascii"))
        for item in value:
            _update_payload_digest(digest, item)
        digest.update(b"];")
    elif isinstance(value, list):
        digest.update(f"L{len(value)}[".encode("ascii"))
        for item in value:
            _update_payload_digest(digest, item)
        digest.update(b"];")
    else:
        raise TypeError(
            "scientific payload contains unsupported "
            f"{type(value).__qualname__}"
        )


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.gemma3_modal_ablation_payload.v1\0")
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    serialized = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.gemma3_modal_ablation_report.v1\0")
    digest.update(serialized)
    return digest.hexdigest()


def _full_rank_condition(
    conditions: Sequence[ModalAblationCondition],
    *,
    sites: Sequence[str],
    width: int,
) -> ModalAblationCondition:
    expected = {site: width for site in sites}
    matches = [
        condition
        for condition in conditions
        if dict(condition.retained_modes) == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "ablation conditions must contain exactly one joint "
            "full-rank identity control"
        )
    return matches[0]


def _full_width_basis_audit(
    collection: StreamingFisherCollection,
    *,
    boundaries: Sequence[str],
    width: int,
) -> dict[str, dict[str, object]]:
    if tuple(collection.bases) != tuple(boundaries):
        raise RuntimeError(
            "calibration collection does not match canonical boundaries"
        )
    audit: dict[str, dict[str, object]] = {}
    for site in boundaries:
        basis = collection.bases[site]
        result = basis.fisher
        if (
            result.width != width
            or result.modes != width
            or result.requested_rank != width
            or result.vectors.shape != (width, width)
        ):
            raise RuntimeError(
                f"{site!r} calibration basis is not exactly full width"
            )
        # Constructor validation includes complete-basis orthonormality and
        # descending Fisher eigenvalue ordering.  The dummy mask is not used
        # until projection call time.
        PooledModalProjection(
            basis=basis,
            retained_modes=width,
            valid_positions=torch.ones((1, 1), dtype=torch.bool),
        )
        trace_error = abs(result.retained_trace - result.fisher_trace)
        trace_scale = max(abs(result.fisher_trace), _FULL_TRACE_ATOL)
        relative_error = trace_error / trace_scale
        trace_passed = math.isclose(
            result.retained_trace,
            result.fisher_trace,
            rel_tol=_FULL_TRACE_RTOL,
            abs_tol=_FULL_TRACE_ATOL,
        )
        audit[site] = {
            "width": width,
            "modes": result.modes,
            "requested_rank": result.requested_rank,
            "sketch_rows": result.sketch_rows,
            "fisher_trace": result.fisher_trace,
            "retained_trace": result.retained_trace,
            "absolute_trace_error": trace_error,
            "relative_trace_error": relative_error,
            "trace_rtol": _FULL_TRACE_RTOL,
            "trace_atol": _FULL_TRACE_ATOL,
            "trace_passed": trace_passed,
            "orthonormal_complete_basis": True,
        }
        if not trace_passed:
            raise RuntimeError(
                f"{site!r} full-width basis lost Fisher trace: "
                f"retained={result.retained_trace:.9g}, "
                f"exact={result.fisher_trace:.9g}"
            )
    return audit


def _build_report(
    *,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    calibration: StreamingFisherCollection,
    full_width_basis_audit: Mapping[str, Mapping[str, object]],
    validation: ModalAblationResult | Mapping[str, object],
    full_rank_control: Mapping[str, object],
    output: Path,
    scientific_payload_sha256: str,
) -> dict[str, object]:
    validation_metadata = (
        validation.metadata()
        if isinstance(validation, ModalAblationResult)
        else copy.deepcopy(dict(validation))
    )
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": {
            "scope": "validation_only_frozen_dense_model_ablation",
            "test_split_evaluated": False,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "prompt_text_in_artifact": False,
            "tokenizer_state_in_artifact": False,
            "pooled_calibration_center": True,
            "joint_output_interventions": True,
            "full_rank_identity_passed": True,
            "compilation_claim": False,
            "compression_claim": False,
        },
        "model": copy.deepcopy(dict(model)),
        "protocol": copy.deepcopy(dict(protocol)),
        "analysis": {
            "calibration": {
                "collection": calibration.metadata(),
                "full_width_basis_audit": copy.deepcopy(
                    dict(full_width_basis_audit)
                ),
            },
            "validation": validation_metadata,
            "full_rank_identity": copy.deepcopy(
                dict(full_rank_control)
            ),
        },
        "artifact": {
            "tensor_output": output.name,
            "contains_model_state_dict": False,
            "contains_tokenizer": False,
            "contains_prompt_text": False,
            "scientific_payload_sha256": scientific_payload_sha256,
        },
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_close(
    actual: object,
    expected: float,
    *,
    label: str,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> float:
    if (
        type(actual) is not float
        or not math.isfinite(actual)
        or not math.isclose(
            actual,
            expected,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    ):
        raise ValueError(f"{label} is invalid")
    return actual


_AGGREGATE_FIELDS = {
    "sequences",
    "supervised_tokens",
    "summed_nll",
    "mean_sequence_summed_nll",
    "nll_per_token",
    "top1_matches",
    "top1_agreement_to_baseline",
    "delta_summed_nll",
    "delta_mean_sequence_summed_nll",
    "delta_nll_per_token",
}
_EXAMPLE_FIELDS = {
    "example_id",
    "supervised_tokens",
    "baseline_summed_nll",
    "baseline_nll_per_token",
    "ablated_summed_nll",
    "ablated_nll_per_token",
    "delta_summed_nll",
    "delta_nll_per_token",
    "top1_matches",
    "top1_agreement_to_baseline",
}


def _validate_aggregate(
    value: object,
    *,
    baseline_summed_nll: float | None,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _AGGREGATE_FIELDS:
        raise ValueError(f"{label} aggregate fields are invalid")
    sequences = value["sequences"]
    supervised_tokens = value["supervised_tokens"]
    top1_matches = value["top1_matches"]
    if (
        type(sequences) is not int
        or sequences <= 0
        or type(supervised_tokens) is not int
        or supervised_tokens < sequences
        or type(top1_matches) is not int
        or not 0 <= top1_matches <= supervised_tokens
    ):
        raise ValueError(f"{label} aggregate counts are invalid")
    summed_nll = value["summed_nll"]
    if (
        type(summed_nll) is not float
        or not math.isfinite(summed_nll)
        or summed_nll < 0
    ):
        raise ValueError(f"{label} aggregate NLL is invalid")
    _assert_close(
        value["mean_sequence_summed_nll"],
        summed_nll / sequences,
        label=f"{label} mean sequence NLL",
    )
    _assert_close(
        value["nll_per_token"],
        summed_nll / supervised_tokens,
        label=f"{label} NLL per token",
    )
    _assert_close(
        value["top1_agreement_to_baseline"],
        top1_matches / supervised_tokens,
        label=f"{label} top-1 agreement",
    )
    reference = (
        summed_nll
        if baseline_summed_nll is None
        else baseline_summed_nll
    )
    delta = summed_nll - reference
    _assert_close(
        value["delta_summed_nll"],
        delta,
        label=f"{label} summed NLL delta",
    )
    _assert_close(
        value["delta_mean_sequence_summed_nll"],
        delta / sequences,
        label=f"{label} mean sequence NLL delta",
    )
    _assert_close(
        value["delta_nll_per_token"],
        delta / supervised_tokens,
        label=f"{label} NLL-per-token delta",
    )
    if baseline_summed_nll is None and (
        top1_matches != supervised_tokens
        or value["top1_agreement_to_baseline"] != 1.0
    ):
        raise ValueError("baseline aggregate identity fields are invalid")
    return copy.deepcopy(dict(value))


def _validate_example(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _EXAMPLE_FIELDS:
        raise ValueError(f"{label} example fields are invalid")
    example_id = value["example_id"]
    tokens = value["supervised_tokens"]
    top1_matches = value["top1_matches"]
    if (
        not isinstance(example_id, str)
        or not example_id
        or type(tokens) is not int
        or tokens <= 0
        or type(top1_matches) is not int
        or not 0 <= top1_matches <= tokens
    ):
        raise ValueError(f"{label} example counts are invalid")
    baseline_nll = value["baseline_summed_nll"]
    ablated_nll = value["ablated_summed_nll"]
    if (
        type(baseline_nll) is not float
        or not math.isfinite(baseline_nll)
        or baseline_nll < 0
        or type(ablated_nll) is not float
        or not math.isfinite(ablated_nll)
        or ablated_nll < 0
    ):
        raise ValueError(f"{label} example NLL is invalid")
    _assert_close(
        value["baseline_nll_per_token"],
        baseline_nll / tokens,
        label=f"{label} baseline NLL per token",
    )
    _assert_close(
        value["ablated_nll_per_token"],
        ablated_nll / tokens,
        label=f"{label} ablated NLL per token",
    )
    _assert_close(
        value["delta_summed_nll"],
        ablated_nll - baseline_nll,
        label=f"{label} summed NLL delta",
    )
    _assert_close(
        value["delta_nll_per_token"],
        (ablated_nll - baseline_nll) / tokens,
        label=f"{label} NLL-per-token delta",
    )
    agreement = _assert_close(
        value["top1_agreement_to_baseline"],
        top1_matches / tokens,
        label=f"{label} top-1 agreement",
    )
    if not 0.0 <= agreement <= 1.0:
        raise ValueError(f"{label} top-1 agreement is out of range")
    return copy.deepcopy(dict(value))


def _validate_ablation_metadata(
    value: object,
    *,
    expected_conditions: Sequence[Mapping[str, object]],
    expected_example_ids: Sequence[str],
    expected_supervised_tokens_by_example: Sequence[int],
    expected_supervised_tokens: int,
) -> dict[str, object]:
    if (
        len(expected_example_ids)
        != len(expected_supervised_tokens_by_example)
        or sum(expected_supervised_tokens_by_example)
        != expected_supervised_tokens
    ):
        raise ValueError(
            "validation provenance example accounting is inconsistent"
        )
    if not isinstance(value, Mapping) or set(value) != {
        "baseline",
        "conditions",
    }:
        raise ValueError("modal-ablation validation fields are invalid")
    baseline = _validate_aggregate(
        value["baseline"],
        baseline_summed_nll=None,
        label="baseline",
    )
    if (
        baseline["sequences"] != len(expected_example_ids)
        or baseline["supervised_tokens"] != expected_supervised_tokens
    ):
        raise ValueError(
            "modal-ablation baseline does not match validation provenance"
        )
    raw_conditions = value["conditions"]
    if (
        not isinstance(raw_conditions, list)
        or len(raw_conditions) != len(expected_conditions)
    ):
        raise ValueError("modal-ablation condition ledger is invalid")
    baseline_rows: list[tuple[str, int, float, float]] | None = None
    validated_conditions: list[dict[str, object]] = []
    names: set[str] = set()
    for condition_index, (raw, expected_condition) in enumerate(
        zip(raw_conditions, expected_conditions, strict=True)
    ):
        label = f"condition {condition_index}"
        if not isinstance(raw, Mapping) or set(raw) != {
            "condition",
            "aggregate",
            "examples",
        }:
            raise ValueError(f"{label} fields are invalid")
        condition = raw["condition"]
        if (
            not isinstance(condition, Mapping)
            or set(condition) != {"name", "retained_modes"}
            or dict(condition) != dict(expected_condition)
        ):
            raise ValueError(f"{label} identity is invalid")
        name = condition["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("modal-ablation condition names are invalid")
        names.add(name)
        aggregate = _validate_aggregate(
            raw["aggregate"],
            baseline_summed_nll=float(baseline["summed_nll"]),
            label=label,
        )
        examples = raw["examples"]
        if (
            not isinstance(examples, list)
            or len(examples) != baseline["sequences"]
        ):
            raise ValueError(f"{label} example ledger is invalid")
        validated_examples = [
            _validate_example(
                example,
                label=f"{label} example {index}",
            )
            for index, example in enumerate(examples)
        ]
        if [
            example["example_id"] for example in validated_examples
        ] != list(expected_example_ids):
            raise ValueError(
                f"{label} example IDs do not match validation provenance"
            )
        if [
            example["supervised_tokens"]
            for example in validated_examples
        ] != list(expected_supervised_tokens_by_example):
            raise ValueError(
                f"{label} per-example supervised-token counts do not "
                "match validation provenance"
            )
        rows = [
            (
                str(example["example_id"]),
                int(example["supervised_tokens"]),
                float(example["baseline_summed_nll"]),
                float(example["baseline_nll_per_token"]),
            )
            for example in validated_examples
        ]
        if baseline_rows is None:
            baseline_rows = rows
        elif rows != baseline_rows:
            raise ValueError(
                "modal-ablation paired baseline ledger is inconsistent"
            )
        if sum(row[1] for row in rows) != baseline["supervised_tokens"]:
            raise ValueError(
                "modal-ablation example token accounting is invalid"
            )
        _assert_close(
            sum(row[2] for row in rows),
            float(baseline["summed_nll"]),
            label=f"{label} baseline example sum",
        )
        _assert_close(
            sum(
                float(example["ablated_summed_nll"])
                for example in validated_examples
            ),
            float(aggregate["summed_nll"]),
            label=f"{label} ablated example sum",
        )
        if (
            sum(
                int(example["top1_matches"])
                for example in validated_examples
            )
            != aggregate["top1_matches"]
        ):
            raise ValueError(f"{label} top-1 accounting is invalid")
        validated_conditions.append(
            {
                "condition": copy.deepcopy(dict(condition)),
                "aggregate": aggregate,
                "examples": validated_examples,
            }
        )
    return {
        "baseline": baseline,
        "conditions": validated_conditions,
    }


def _validate_model_metadata(value: object) -> dict[str, object]:
    fields = {
        "model_id",
        "requested_revision",
        "resolved_commit",
        "model_class",
        "config_sha256",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "maximum_context",
        "parameter_count",
        "device",
        "dtype",
        "weights_in_artifact",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("ablation model metadata fields are invalid")
    if (
        not isinstance(value["model_id"], str)
        or not value["model_id"]
        or not isinstance(value["model_class"], str)
        or not value["model_class"]
        or not _is_sha256(value["config_sha256"])
        or type(value["parameter_count"]) is not int
        or value["parameter_count"] < 0
        or value["weights_in_artifact"] is not False
    ):
        raise ValueError("ablation model metadata is invalid")
    for name in (
        "requested_revision",
        "resolved_commit",
        "model_type",
        "device",
        "dtype",
    ):
        if value[name] is not None and not isinstance(value[name], str):
            raise ValueError("ablation model string metadata is invalid")
    for name in ("hidden_size", "num_hidden_layers", "maximum_context"):
        if value[name] is not None and (
            type(value[name]) is not int or value[name] <= 0
        ):
            raise ValueError("ablation model dimension metadata is invalid")
    return copy.deepcopy(dict(value))


def _validate_prompt_provenance(
    value: object,
    *,
    calibration_stream: Mapping[str, object],
    validation_stream: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "scientific_status",
        "counts",
        "normalized_sha256",
        "per_prompt_sha256",
    }:
        raise ValueError("ablation prompt provenance fields are invalid")
    if (
        not isinstance(value["scientific_status"], str)
        or not value["scientific_status"]
    ):
        raise ValueError("ablation prompt scientific status is invalid")
    split_names = {
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    }
    counts = value["counts"]
    normalized = value["normalized_sha256"]
    per_prompt = value["per_prompt_sha256"]
    if (
        not isinstance(counts, Mapping)
        or set(counts) != split_names
        or not isinstance(normalized, Mapping)
        or set(normalized) != split_names
        or not isinstance(per_prompt, Mapping)
        or set(per_prompt) != split_names
    ):
        raise ValueError("ablation prompt provenance mappings are invalid")
    all_hashes: list[str] = []
    for split in split_names:
        hashes = per_prompt[split]
        if (
            type(counts[split]) is not int
            or counts[split] <= 0
            or not isinstance(hashes, list)
            or len(hashes) != counts[split]
            or any(not _is_sha256(item) for item in hashes)
            or normalized[split] != _ordered_prompt_hash_digest(hashes)
        ):
            raise ValueError("ablation prompt counts or digests are invalid")
        all_hashes.extend(hashes)
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("ablation prompt hashes must be disjoint")
    expected_calibration = (
        per_prompt["calibration_a"] + per_prompt["calibration_b"]
    )
    if (
        calibration_stream["source_prompt_sha256"]
        != expected_calibration
        or calibration_stream["sequences"] != len(expected_calibration)
        or validation_stream["source_prompt_sha256"]
        != per_prompt["validation"]
        or validation_stream["sequences"] != counts["validation"]
    ):
        raise ValueError(
            "tokenized streams do not match frozen prompt provenance"
        )
    streamed = set(
        calibration_stream["source_prompt_sha256"]
        + validation_stream["source_prompt_sha256"]
    )
    if streamed & set(per_prompt["test"]):
        raise ValueError("reserved test prompts appear in a tokenized stream")
    return copy.deepcopy(dict(value))


def _validate_protocol(
    value: object,
    *,
    calibration_stream: Mapping[str, object],
    validation_stream: Mapping[str, object],
) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
    fields = {
        "start_layer",
        "end_layer_inclusive",
        "layer_ids",
        "canonical_boundaries",
        "boundary_widths",
        "leaf_boundary",
        "gated_output_sites",
        "residual_width",
        "ranks",
        "conditions",
        "include_singletons",
        "extraction_rank",
        "sketch_rows",
        "maximum_tokenized_length",
        "tokenization_batch_size",
        "gradient_batching",
        "validation_batching",
        "score",
        "score_compute_dtype",
        "centering",
        "scope",
        "normalizer",
        "gate",
        "intervention_order",
        "full_rank_nll_atol",
        "test_policy",
        "claim",
        "cache_policy",
        "model_state_guard",
        "library_versions",
        "tokenizer",
        "tokenized_splits",
        "prompt_splits",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("ablation protocol fields are invalid")
    start = value["start_layer"]
    end = value["end_layer_inclusive"]
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end < start
    ):
        raise ValueError("ablation protocol layer range is invalid")
    layers = value["layer_ids"]
    boundaries = value["canonical_boundaries"]
    widths = value["boundary_widths"]
    outputs = value["gated_output_sites"]
    width = value["residual_width"]
    if (
        type(layers) is not tuple
        or len(layers) != end - start + 1
        or len(set(layers)) != len(layers)
        or any(not isinstance(name, str) or not name for name in layers)
        or type(boundaries) is not tuple
        or len(boundaries) != len(layers) + 1
        or len(set(boundaries)) != len(boundaries)
        or any(not isinstance(name, str) or not name for name in boundaries)
        or type(widths) is not tuple
        or type(width) is not int
        or width <= 0
        or widths != (width,) * len(boundaries)
        or value["leaf_boundary"] != boundaries[0]
        or outputs != boundaries[1:]
    ):
        raise ValueError("ablation protocol block boundaries are invalid")
    ranks = _validated_rank_schedule(value["ranks"], width=width)
    if ranks != value["ranks"]:
        raise ValueError("ablation protocol rank schedule is not canonical")
    include_singletons = value["include_singletons"]
    if not isinstance(include_singletons, bool):
        raise ValueError("ablation singleton policy is invalid")
    expected_conditions = tuple(
        condition.metadata()
        for condition in build_modal_ablation_conditions(
            sites=outputs,
            ranks=ranks,
            include_joint=True,
            include_singletons=include_singletons,
        )
    )
    conditions = value["conditions"]
    if (
        type(conditions) is not tuple
        or tuple(dict(item) for item in conditions)
        != tuple(dict(item) for item in expected_conditions)
    ):
        raise ValueError("ablation protocol conditions are invalid")
    if (
        value["extraction_rank"] != width
        or type(value["sketch_rows"]) is not int
        or value["sketch_rows"] < width + 1
        or type(value["maximum_tokenized_length"]) is not int
        or value["maximum_tokenized_length"] < 2
        or type(value["tokenization_batch_size"]) is not int
        or value["tokenization_batch_size"] <= 0
    ):
        raise ValueError("ablation protocol extraction settings are invalid")
    fixed = {
        "gradient_batching": "one_sequence_at_a_time",
        "validation_batching": "one_forward_per_batch_per_condition",
        "score": "summed_hard_target_next_token_nll",
        "score_compute_dtype": (
            "float32_for_float16_or_bfloat16_logits"
        ),
        "centering": "pooled_calibration_mean",
        "scope": "width_pooled",
        "normalizer": "valid_activation_positions",
        "gate": "keep_top_k_remove_low_modes",
        "intervention_order": "joint_native_forward_order",
        "test_policy": "parse_validate_hash_only",
        "claim": (
            "frozen_dense_model_representational_sufficiency_only"
        ),
        "cache_policy": "external_to_git_worktree",
    }
    if any(value[name] != expected for name, expected in fixed.items()):
        raise ValueError("ablation protocol scientific semantics are invalid")
    tolerance = value["full_rank_nll_atol"]
    if (
        type(tolerance) is not float
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("ablation full-rank tolerance is invalid")
    guard = value["model_state_guard"]
    if (
        not isinstance(guard, Mapping)
        or set(guard)
        != {
            "verified",
            "training",
            "parameters_frozen",
            "parameter_tensors",
            "buffer_tensors",
            "checks",
        }
        or guard["verified"] is not True
        or guard["training"] is not False
        or guard["parameters_frozen"] is not True
        or type(guard["parameter_tensors"]) is not int
        or guard["parameter_tensors"] < 0
        or type(guard["buffer_tensors"]) is not int
        or guard["buffer_tensors"] < 0
        or guard["checks"]
        != (
            "tensor_object_identity",
            "tensor_version_counter",
            "tensor_storage_identity",
        )
    ):
        raise ValueError("ablation model-state guard metadata is invalid")
    libraries = value["library_versions"]
    if (
        not isinstance(libraries, Mapping)
        or set(libraries)
        != {
            "python",
            "torch",
            "transformers",
            "tokenizers",
            "sentencepiece",
        }
        or any(
            item is not None and not isinstance(item, str)
            for item in libraries.values()
        )
    ):
        raise ValueError("ablation library provenance is invalid")
    tokenizer = value["tokenizer"]
    if (
        not isinstance(tokenizer, Mapping)
        or set(tokenizer)
        != {
            "tokenizer_class",
            "name_or_path",
            "configuration_sha256",
        }
        or not isinstance(tokenizer["tokenizer_class"], str)
        or not tokenizer["tokenizer_class"]
        or (
            tokenizer["name_or_path"] is not None
            and not isinstance(tokenizer["name_or_path"], str)
        )
        or not _is_sha256(tokenizer["configuration_sha256"])
    ):
        raise ValueError("ablation tokenizer provenance is invalid")
    streams = value["tokenized_splits"]
    if (
        not isinstance(streams, Mapping)
        or set(streams) != {"calibration_full", "validation"}
        or streams["calibration_full"] != calibration_stream
        or streams["validation"] != validation_stream
    ):
        raise ValueError("ablation tokenized split provenance is invalid")
    _validate_prompt_provenance(
        value["prompt_splits"],
        calibration_stream=calibration_stream,
        validation_stream=validation_stream,
    )
    return copy.deepcopy(dict(value)), expected_conditions


def _validate_full_rank_identity(
    value: object,
    *,
    expected_condition: Mapping[str, object],
    validation: Mapping[str, object],
    tolerance: float,
) -> dict[str, object]:
    fields = {
        "condition",
        "baseline_nll_per_token",
        "projected_nll_per_token",
        "delta_nll_per_token",
        "absolute_delta_nll_per_token",
        "maximum_absolute_example_delta_nll_per_token",
        "maximum_absolute_example_delta_summed_nll",
        "top1_agreement_to_baseline",
        "top1_identity",
        "tolerance",
        "passed",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("full-rank identity fields are invalid")
    if dict(value["condition"]) != dict(expected_condition):
        raise ValueError("full-rank identity condition is invalid")
    matching = [
        condition
        for condition in validation["conditions"]
        if condition["condition"]["name"] == expected_condition["name"]
    ]
    if len(matching) != 1:
        raise ValueError("full-rank validation condition is missing")
    result = matching[0]
    aggregate = result["aggregate"]
    baseline = validation["baseline"]
    baseline_nll = float(baseline["nll_per_token"])
    projected_nll = float(aggregate["nll_per_token"])
    delta = float(aggregate["delta_nll_per_token"])
    maximum_example_delta = max(
        abs(float(example["delta_nll_per_token"]))
        for example in result["examples"]
    )
    maximum_example_sum = max(
        abs(float(example["delta_summed_nll"]))
        for example in result["examples"]
    )
    agreement = float(aggregate["top1_agreement_to_baseline"])
    expected_values = {
        "baseline_nll_per_token": baseline_nll,
        "projected_nll_per_token": projected_nll,
        "delta_nll_per_token": delta,
        "absolute_delta_nll_per_token": abs(delta),
        "maximum_absolute_example_delta_nll_per_token": (
            maximum_example_delta
        ),
        "maximum_absolute_example_delta_summed_nll": maximum_example_sum,
        "top1_agreement_to_baseline": agreement,
        "tolerance": tolerance,
    }
    for name, expected in expected_values.items():
        _assert_close(
            value[name],
            expected,
            label=f"full-rank identity {name}",
        )
    expected_passed = (
        abs(delta) <= tolerance
        and maximum_example_delta <= tolerance
        and agreement == 1.0
    )
    if (
        value["top1_identity"] is not (agreement == 1.0)
        or value["passed"] is not expected_passed
        or value["passed"] is not True
    ):
        raise ValueError("full-rank identity gate is invalid or failed")
    return copy.deepcopy(dict(value))


def load_gemma3_ablation_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly load and cross-check an analysis-only ablation artifact."""

    artifact_path = Path(path)
    raw = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "contains_prompt_text",
        "contains_tokenizer_state",
        "model",
        "protocol",
        "calibration",
        "validation",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("Gemma ablation artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
    ):
        raise ValueError("unsupported Gemma ablation artifact")
    if (
        raw["contains_model_weights"] is not False
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
    ):
        raise ValueError("ablation artifact contains forbidden source state")
    if (
        not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("ablation artifact digest fields are invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    scientific_digest = _scientific_payload_sha256(payload)
    if scientific_digest != raw["scientific_payload_sha256"]:
        raise ValueError("ablation scientific payload digest mismatch")

    calibration_entry = raw["calibration"]
    validation_entry = raw["validation"]
    if (
        not isinstance(calibration_entry, Mapping)
        or set(calibration_entry)
        != {
            "collection",
            "tokenized_stream",
            "full_width_basis_audit",
        }
        or not isinstance(validation_entry, Mapping)
        or set(validation_entry)
        != {
            "modal_ablation",
            "tokenized_stream",
            "full_rank_identity",
        }
    ):
        raise ValueError("ablation scientific payload structure is invalid")
    calibration_stream, calibration_valid_tokens = (
        _validated_tokenized_stream(
            calibration_entry["tokenized_stream"],
            split_name="calibration_full",
        )
    )
    validation_stream, _ = _validated_tokenized_stream(
        validation_entry["tokenized_stream"],
        split_name="validation",
    )
    protocol, expected_conditions = _validate_protocol(
        raw["protocol"],
        calibration_stream=calibration_stream,
        validation_stream=validation_stream,
    )
    model = _validate_model_metadata(raw["model"])
    collection_state = calibration_entry["collection"]
    if not isinstance(collection_state, Mapping):
        raise ValueError("ablation calibration collection is invalid")
    calibration = StreamingFisherCollection.from_state_dict(
        collection_state
    )
    boundaries = protocol["canonical_boundaries"]
    width = protocol["residual_width"]
    assert isinstance(boundaries, tuple)
    assert isinstance(width, int)
    hidden_size = model["hidden_size"]
    if hidden_size is not None and hidden_size != width:
        raise ValueError(
            "model hidden size does not match ablation residual width"
        )
    model_layers = model["num_hidden_layers"]
    end_layer = protocol["end_layer_inclusive"]
    assert isinstance(end_layer, int)
    if model_layers is not None and end_layer >= model_layers:
        raise ValueError(
            "ablation layer range exceeds model layer metadata"
        )
    sketch_rows = protocol["sketch_rows"]
    normalizer = protocol["normalizer"]
    scope = protocol["scope"]
    assert isinstance(sketch_rows, int)
    assert isinstance(normalizer, str)
    assert isinstance(scope, str)
    for site, basis in calibration.bases.items():
        result = basis.fisher
        if (
            result.observations != calibration_valid_tokens
            or result.rows_seen != calibration_valid_tokens
        ):
            raise ValueError(
                f"{site!r} Fisher row accounting does not match "
                "calibration tokenized provenance"
            )
        if (
            result.sketch_rows != sketch_rows
            or result.score_reduction != "sum"
            or result.normalizer != normalizer
            or result.scope != scope
        ):
            raise ValueError(
                f"{site!r} Fisher estimator semantics do not match "
                "the ablation protocol"
            )
    recomputed_audit = _full_width_basis_audit(
        calibration,
        boundaries=boundaries,
        width=width,
    )
    if calibration_entry["full_width_basis_audit"] != recomputed_audit:
        raise ValueError("full-width Fisher basis audit mismatch")
    if calibration.sequences != calibration_stream["sequences"]:
        raise ValueError(
            "calibration collection does not match tokenized provenance"
        )
    expected_example_ids = tuple(
        str(example["example_id"])
        for example in validation_stream["examples"]
    )
    expected_supervised_tokens_by_example = tuple(
        int(example["supervised_positions"])
        for example in validation_stream["examples"]
    )
    expected_supervised_tokens = sum(
        expected_supervised_tokens_by_example
    )
    validation = _validate_ablation_metadata(
        validation_entry["modal_ablation"],
        expected_conditions=expected_conditions,
        expected_example_ids=expected_example_ids,
        expected_supervised_tokens_by_example=(
            expected_supervised_tokens_by_example
        ),
        expected_supervised_tokens=expected_supervised_tokens,
    )
    gated_sites = protocol["gated_output_sites"]
    assert isinstance(gated_sites, tuple)
    expected_full_condition = {
        "name": f"joint.rank_{width}",
        "retained_modes": {
            site: width for site in gated_sites
        },
    }
    full_rank_control = _validate_full_rank_identity(
        validation_entry["full_rank_identity"],
        expected_condition=expected_full_condition,
        validation=validation,
        tolerance=float(protocol["full_rank_nll_atol"]),
    )

    report_path = artifact_path.with_suffix(".json")
    raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw_report, Mapping):
        raise ValueError("Gemma ablation JSON report must be an object")
    if _report_sha256(raw_report) != raw["report_sha256"]:
        raise ValueError("ablation JSON report digest mismatch")
    expected_report = _build_report(
        model=model,
        protocol=protocol,
        calibration=calibration,
        full_width_basis_audit=recomputed_audit,
        validation=validation,
        full_rank_control=full_rank_control,
        output=artifact_path,
        scientific_payload_sha256=scientific_digest,
    )
    canonical_expected = json.loads(
        json.dumps(
            expected_report,
            sort_keys=True,
            allow_nan=False,
        )
    )
    if raw_report != canonical_expected:
        raise ValueError(
            "ablation JSON report does not match scientific payload"
        )
    return {
        "calibration": calibration,
        "validation": validation,
        "metadata": {
            "schema": raw["schema"],
            "format_version": raw["format_version"],
            "model": model,
            "protocol": protocol,
            "full_width_basis_audit": recomputed_audit,
            "full_rank_identity": full_rank_control,
            "scientific_payload_sha256": scientific_digest,
            "report_sha256": raw["report_sha256"],
        },
        "report": copy.deepcopy(dict(raw_report)),
    }


def run_gemma3_ablation(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    start_layer: int = DEFAULT_START_LAYER,
    end_layer: int = DEFAULT_END_LAYER,
    max_length: int = 128,
    tokenization_batch_size: int = 4,
    ranks: Iterable[int] = DEFAULT_RANKS,
    sketch_rows: int | None = None,
    include_singletons: bool = False,
    full_rank_nll_atol: float = DEFAULT_FULL_RANK_NLL_ATOL,
    device_name: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Run a full-width calibration fit and validation modal-ablation curve."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if (
        type(start_layer) is not int
        or type(end_layer) is not int
        or start_layer < 0
        or end_layer < start_layer
    ):
        raise ValueError("layer range must be nonnegative and ascending")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    if not isinstance(include_singletons, bool):
        raise TypeError("include_singletons must be boolean")
    if (
        isinstance(full_rank_nll_atol, bool)
        or not isinstance(full_rank_nll_atol, (int, float))
        or not math.isfinite(float(full_rank_nll_atol))
        or full_rank_nll_atol < 0
    ):
        raise ValueError(
            "full_rank_nll_atol must be finite and nonnegative"
        )
    resolved_output = (
        default_gemma3_ablation_output(
            model_id,
            start_layer,
            end_layer,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")

    prompt_splits = load_gemma3_prompt_splits(prompt_splits_path)
    device = resolve_torch_device(device_name)
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
    plan = adapter.plan_layer_block(start_layer, end_layer)
    if len(set(plan.widths)) != 1:
        raise ValueError(
            "modal ablation requires one shared residual width across block"
        )
    width = plan.widths[0]
    requested_ranks = _validated_rank_schedule(ranks, width=width)
    resolved_sketch_rows = (
        width + 1 if sketch_rows is None else sketch_rows
    )
    if (
        type(resolved_sketch_rows) is not int
        or resolved_sketch_rows < width + 1
    ):
        raise ValueError(
            "sketch_rows must be an integer at least residual width + 1"
        )

    calibration_prompts = (
        prompt_splits.calibration_a + prompt_splits.calibration_b
    )
    calibration_provenance = _CalibrationStreamProvenance(
        "calibration_full",
        calibration_prompts,
    )
    calibration_batches = make_causal_lm_calibration_batches(
        tokenizer,
        calibration_prompts,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    calibration = collect_streaming_fisher_modes(
        adapter,
        calibration_provenance.wrap(calibration_batches),
        activation_names=plan.activation_sites,
        score_objective=CausalLanguageModelNLL(),
        rank=width,
        sketch_rows=resolved_sketch_rows,
        leaf_activation_name=plan.leaf_activation_name,
    )
    calibration_stream = calibration_provenance.metadata()
    full_width_basis_audit = _full_width_basis_audit(
        calibration,
        boundaries=plan.activation_sites,
        width=width,
    )
    guard.assert_unchanged()

    gated_output_sites = plan.activation_sites[1:]
    conditions = build_modal_ablation_conditions(
        sites=gated_output_sites,
        ranks=requested_ranks,
        include_joint=True,
        include_singletons=include_singletons,
    )
    full_condition = _full_rank_condition(
        conditions,
        sites=gated_output_sites,
        width=width,
    )
    validation_provenance = _CalibrationStreamProvenance(
        "validation",
        prompt_splits.validation,
    )
    validation_batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompt_splits.validation,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    validation = evaluate_causal_lm_modal_ablation(
        adapter,
        validation_provenance.wrap(validation_batches),
        bases={
            site: calibration.bases[site]
            for site in gated_output_sites
        },
        conditions=conditions,
        objective=CausalLanguageModelNLL(),
    )
    validation_stream = validation_provenance.metadata()
    guard.assert_unchanged()

    full_result = validation.condition(full_condition.name)
    full_rank_delta = full_result.aggregate.delta_nll_per_token
    maximum_example_delta = max(
        abs(example.delta_nll_per_token)
        for example in full_result.examples
    )
    maximum_example_summed_delta = max(
        abs(example.delta_summed_nll)
        for example in full_result.examples
    )
    top1_identity = (
        full_result.aggregate.top1_agreement_to_baseline == 1.0
    )
    identity_passed = (
        abs(full_rank_delta) <= float(full_rank_nll_atol)
        and maximum_example_delta <= float(full_rank_nll_atol)
        and top1_identity
    )
    full_rank_control = {
        "condition": full_condition.metadata(),
        "baseline_nll_per_token": validation.baseline.nll_per_token,
        "projected_nll_per_token": full_result.aggregate.nll_per_token,
        "delta_nll_per_token": full_rank_delta,
        "absolute_delta_nll_per_token": abs(full_rank_delta),
        "maximum_absolute_example_delta_nll_per_token": (
            maximum_example_delta
        ),
        "maximum_absolute_example_delta_summed_nll": (
            maximum_example_summed_delta
        ),
        "top1_agreement_to_baseline": (
            full_result.aggregate.top1_agreement_to_baseline
        ),
        "top1_identity": top1_identity,
        "tolerance": float(full_rank_nll_atol),
        "passed": identity_passed,
    }
    if not identity_passed:
        raise RuntimeError(
            "joint full-rank projector failed baseline NLL identity gate: "
            f"aggregate_abs_delta={abs(full_rank_delta):.9g}, "
            f"max_example_abs_delta={maximum_example_delta:.9g}, "
            f"top1_agreement="
            f"{full_result.aggregate.top1_agreement_to_baseline:.9g}, "
            f"tolerance={float(full_rank_nll_atol):.9g}"
        )

    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    tokenized_splits = {
        "calibration_full": calibration_stream,
        "validation": validation_stream,
    }
    protocol = {
        "start_layer": start_layer,
        "end_layer_inclusive": end_layer,
        "layer_ids": plan.layer_ids,
        "canonical_boundaries": plan.activation_sites,
        "boundary_widths": plan.widths,
        "leaf_boundary": plan.leaf_activation_name,
        "gated_output_sites": gated_output_sites,
        "residual_width": width,
        "ranks": requested_ranks,
        "conditions": tuple(
            condition.metadata() for condition in conditions
        ),
        "include_singletons": include_singletons,
        "extraction_rank": width,
        "sketch_rows": resolved_sketch_rows,
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "gradient_batching": "one_sequence_at_a_time",
        "validation_batching": "one_forward_per_batch_per_condition",
        "score": "summed_hard_target_next_token_nll",
        "score_compute_dtype": (
            "float32_for_float16_or_bfloat16_logits"
        ),
        "centering": "pooled_calibration_mean",
        "scope": "width_pooled",
        "normalizer": "valid_activation_positions",
        "gate": "keep_top_k_remove_low_modes",
        "intervention_order": "joint_native_forward_order",
        "full_rank_nll_atol": float(full_rank_nll_atol),
        "test_policy": "parse_validate_hash_only",
        "claim": "frozen_dense_model_representational_sufficiency_only",
        "cache_policy": "external_to_git_worktree",
        "model_state_guard": guard.metadata(),
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "tokenized_splits": tokenized_splits,
        "prompt_splits": prompt_splits.metadata(),
    }
    artifact_payload = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "model": model_metadata,
        "protocol": protocol,
        "calibration": {
            "collection": calibration.state_dict(),
            "tokenized_stream": calibration_stream,
            "full_width_basis_audit": full_width_basis_audit,
        },
        "validation": {
            "modal_ablation": validation.metadata(),
            "tokenized_stream": validation_stream,
            "full_rank_identity": full_rank_control,
        },
    }
    scientific_digest = _scientific_payload_sha256(artifact_payload)
    report = _build_report(
        model=model_metadata,
        protocol=protocol,
        calibration=calibration,
        full_width_basis_audit=full_width_basis_audit,
        validation=validation,
        full_rank_control=full_rank_control,
        output=resolved_output,
        scientific_payload_sha256=scientific_digest,
    )
    report_digest = _report_sha256(report)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **artifact_payload,
            "scientific_payload_sha256": scientific_digest,
            "report_sha256": report_digest,
        },
        resolved_output,
    )
    resolved_output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit complete Gemma residual Fisher bases on calibration A+B "
            "and evaluate joint output-site modal ablations on validation."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help="validate external Hugging Face paths without loading a model",
    )
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument("--start-layer", type=int, default=DEFAULT_START_LAYER)
    parser.add_argument("--end-layer", type=int, default=DEFAULT_END_LAYER)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument(
        "--retained-ranks",
        "--ranks",
        dest="ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANKS),
    )
    parser.add_argument("--sketch-rows", type=int)
    parser.add_argument(
        "--include-single-sites",
        "--include-singletons",
        dest="include_singletons",
        action="store_true",
    )
    parser.add_argument(
        "--full-rank-nll-atol",
        type=float,
        default=DEFAULT_FULL_RANK_NLL_ATOL,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.check_paths_only:
        paths = resolve_gemma3_huggingface_paths(arguments.cache_dir)
        print("Validated external Hugging Face write paths; no model loaded:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return
    output = (
        arguments.output
        if arguments.output is not None
        else default_gemma3_ablation_output(
            arguments.model,
            arguments.start_layer,
            arguments.end_layer,
        )
    )
    report = run_gemma3_ablation(
        model_id=arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        start_layer=arguments.start_layer,
        end_layer=arguments.end_layer,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        ranks=arguments.ranks,
        sketch_rows=arguments.sketch_rows,
        include_singletons=arguments.include_singletons,
        full_rank_nll_atol=arguments.full_rank_nll_atol,
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=output,
    )
    identity = report["analysis"]["full_rank_identity"]
    assert isinstance(identity, Mapping)
    print(f"Wrote modal-ablation analysis to {output}")
    print(
        "Full-rank identity delta NLL/token: "
        f"{identity['delta_nll_per_token']}"
    )
    print(f"Report: {output.with_suffix('.json')}")
    print("Reserved test prompts were not tokenized or model-evaluated.")
    print("No pretrained weights, prompts, or tokenizer state were written.")
    print("This result makes no compilation or compression claim.")


if __name__ == "__main__":
    main()
