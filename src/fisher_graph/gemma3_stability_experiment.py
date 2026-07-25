"""Split stability and exact held-out replay for Gemma 3 Fisher modes.

This is the second opt-in external-model rung.  It extracts independent
streaming Fisher bases on two frozen calibration splits, compares their
principal-angle geometry, and measures exact held-out Rayleigh energy for
both frozen bases on a third split.  It never serializes model weights or
retains calibration gradient rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _json_compatible,
    _model_provenance,
    default_gemma3_output,
    load_gemma3,
    make_causal_lm_calibration_batches,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .streaming_analysis import (
    StreamingFisherCollection,
    collect_streaming_fisher_modes,
    iter_activation_score_gradient_rows,
)
from .streaming_validation import (
    FisherSubspaceStability,
    StreamingRayleighEnergyEstimator,
    StreamingRayleighEnergyResult,
    compare_fisher_subspaces,
)


DEFAULT_PROMPT_SPLITS = Path("examples/gemma3_stability_prompts.json")
DEFAULT_RANKS = (8, 16, 32, 64)
_PROMPT_SCHEMA = "fisher_graph.gemma3_prompt_splits"
_PROMPT_FORMAT_VERSION = 1
_ARTIFACT_SCHEMA = "fisher_graph.gemma3_fisher_stability"
_ARTIFACT_FORMAT_VERSION = 2


class _Digest(Protocol):
    def update(self, value: bytes) -> object: ...

    def hexdigest(self) -> str: ...


def _update_tensor_digest(
    digest: _Digest,
    *,
    name: str,
    tensor: Tensor,
) -> None:
    canonical = tensor.detach().to(device="cpu").contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            list(canonical.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    digest.update(b"\0")


def _update_calibration_batch_digest(
    digest: _Digest,
    batch: CalibrationBatch,
) -> None:
    digest.update(b"fisher_graph.calibration_batch.v1\0")
    digest.update(
        json.dumps(
            {
                "example_ids": batch.example_ids,
                "shared_input_names": sorted(batch.shared_input_names),
                "model_input_names": sorted(batch.model_inputs),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    for name in sorted(batch.model_inputs):
        _update_tensor_digest(
            digest,
            name=f"model_inputs.{name}",
            tensor=batch.model_inputs[name],
        )
    _update_tensor_digest(digest, name="targets", tensor=batch.targets)
    _update_tensor_digest(
        digest,
        name="valid_positions",
        tensor=batch.valid_positions,
    )


def _tokenized_example_content_sha256(
    sample: CalibrationBatch,
) -> str:
    if sample.batch_size != 1:
        raise ValueError("tokenized content hashing requires one example")
    valid = sample.valid_positions[0].to(device="cpu")
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.tokenized_example_content.v1\0")
    for name in sorted(sample.model_inputs):
        value = sample.model_inputs[name]
        if name in sample.shared_input_names or value.ndim == 0:
            selected = value
        elif value.ndim >= 2 and value.shape[1] == valid.numel():
            selected = value[0, valid.to(device=value.device)]
        else:
            selected = value[0]
        _update_tensor_digest(
            digest,
            name=f"model_inputs.{name}",
            tensor=selected,
        )
    _update_tensor_digest(
        digest,
        name="targets",
        tensor=sample.targets[0, valid.to(device=sample.targets.device)],
    )
    return digest.hexdigest()


class _CalibrationStreamProvenance:
    """Hash exact emitted calibration tensors without retaining them."""

    def __init__(
        self,
        split_name: str,
        prompts: Sequence[str],
    ) -> None:
        if not prompts:
            raise ValueError("stream provenance prompts cannot be empty")
        self.split_name = split_name
        self._source_prompt_sha256 = [
            _prompt_digest((prompt,))
            for prompt in prompts
        ]
        self._digest = hashlib.sha256()
        self._digest.update(b"fisher_graph.calibration_stream.v1\0")
        self._digest.update(split_name.encode("utf-8"))
        self._digest.update(b"\0")
        self._batches = 0
        self._examples: list[dict[str, object]] = []

    def wrap(
        self,
        batches: Iterable[CalibrationBatch],
    ) -> Iterator[CalibrationBatch]:
        for batch in batches:
            if not isinstance(batch, CalibrationBatch):
                raise TypeError(
                    "tokenized calibration stream must contain "
                    "CalibrationBatch values"
                )
            _update_calibration_batch_digest(self._digest, batch)
            self._batches += 1
            for index in range(batch.batch_size):
                sample = batch.sample(index)
                example_digest = hashlib.sha256()
                _update_calibration_batch_digest(example_digest, sample)
                self._examples.append(
                    {
                        "example_id": (
                            None
                            if sample.example_ids is None
                            else sample.example_ids[0]
                        ),
                        "serialized_sha256": example_digest.hexdigest(),
                        "content_sha256": (
                            _tokenized_example_content_sha256(sample)
                        ),
                        "valid_tokens": int(
                            sample.valid_positions.sum().item()
                        ),
                        "supervised_positions": int(
                            (sample.targets != -100).sum().item()
                        ),
                    }
                )
            yield batch

    def metadata(self) -> dict[str, object]:
        if not self._examples:
            raise ValueError(
                f"tokenized split {self.split_name!r} was not consumed"
            )
        if len(self._examples) != len(self._source_prompt_sha256):
            raise ValueError(
                "tokenized sequence count does not match source prompts"
            )
        return {
            "schema": "fisher_graph.tokenized_calibration_stream",
            "format_version": 2,
            "split": self.split_name,
            "batches": self._batches,
            "sequences": len(self._examples),
            "serialized_sha256": self._digest.hexdigest(),
            "source_prompt_sha256": list(self._source_prompt_sha256),
            "examples": list(self._examples),
        }


def _tokenizer_provenance(tokenizer: object) -> dict[str, object]:
    config = {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "init_kwargs": getattr(tokenizer, "init_kwargs", None),
    }
    serialized = json.dumps(
        _json_compatible(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "configuration_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _library_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": _installed_version("transformers"),
        "tokenizers": _installed_version("tokenizers"),
        "sentencepiece": _installed_version("sentencepiece"),
    }


def default_gemma3_stability_output(
    model_id: str = DEFAULT_MODEL_ID,
    layer_index: int = 0,
) -> Path:
    """Return an ignored model/layer-specific stability artifact path."""

    fisher_output = default_gemma3_output(model_id, layer_index)
    return fisher_output.with_name(
        f"layer-{layer_index}-fisher-stability.pt"
    )


def _prompt_digest(prompts: Sequence[str]) -> str:
    payload = json.dumps(
        list(prompts),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_prompt_hash_digest(prompt_hashes: Sequence[str]) -> str:
    payload = json.dumps(
        list(prompt_hashes),
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _normalized_prompt_split(
    name: str,
    value: object,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty JSON list")
    if any(not isinstance(prompt, str) for prompt in value):
        raise TypeError(f"{name} prompts must be strings")
    prompts = tuple(prompt.strip() for prompt in value)
    if any(not prompt for prompt in prompts):
        raise ValueError(f"{name} prompts must contain nonempty text")
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"{name} cannot contain duplicate prompts")
    return prompts


@dataclass(frozen=True, slots=True)
class Gemma3PromptSplits:
    """Three disjoint, frozen prompt splits for stability and replay."""

    calibration_a: tuple[str, ...]
    calibration_b: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    scientific_status: str

    def __post_init__(self) -> None:
        for name, prompts in (
            ("calibration_a", self.calibration_a),
            ("calibration_b", self.calibration_b),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if type(prompts) is not tuple or not prompts:
                raise ValueError(f"{name} must be a nonempty tuple")
            if any(
                not isinstance(prompt, str) or not prompt
                for prompt in prompts
            ):
                raise ValueError(f"{name} prompts must be nonempty strings")
            if len(set(prompts)) != len(prompts):
                raise ValueError(f"{name} cannot contain duplicate prompts")
        if not isinstance(self.scientific_status, str) or not (
            self.scientific_status
        ):
            raise ValueError("scientific_status must be a nonempty string")
        a = set(self.calibration_a)
        b = set(self.calibration_b)
        validation = set(self.validation)
        test = set(self.test)
        if (
            a & b
            or a & validation
            or a & test
            or b & validation
            or b & test
            or validation & test
        ):
            raise ValueError("prompt splits must be pairwise disjoint")

    def metadata(self) -> dict[str, object]:
        splits = {
            "calibration_a": self.calibration_a,
            "calibration_b": self.calibration_b,
            "validation": self.validation,
            "test": self.test,
        }
        per_prompt_sha256 = {
            name: [
                _prompt_digest((prompt,))
                for prompt in prompts
            ]
            for name, prompts in splits.items()
        }
        return {
            "scientific_status": self.scientific_status,
            "counts": {
                name: len(prompts)
                for name, prompts in splits.items()
            },
            "normalized_sha256": {
                name: _ordered_prompt_hash_digest(
                    per_prompt_sha256[name]
                )
                for name in splits
            },
            "per_prompt_sha256": per_prompt_sha256,
        }


def load_gemma3_prompt_splits(
    path: Path | str,
) -> Gemma3PromptSplits:
    """Load strict, disjoint prompt splits from a JSON file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("Gemma 3 prompt split file must contain an object")
    required = {
        "schema",
        "format_version",
        "scientific_status",
        "calibration_a",
        "calibration_b",
        "validation",
        "test",
    }
    if set(raw) != required:
        raise ValueError(
            "Gemma 3 prompt split fields do not match format version 1"
        )
    if raw["schema"] != _PROMPT_SCHEMA:
        raise ValueError("unsupported Gemma 3 prompt split schema")
    if raw["format_version"] != _PROMPT_FORMAT_VERSION:
        raise ValueError("unsupported Gemma 3 prompt split format")
    return Gemma3PromptSplits(
        calibration_a=_normalized_prompt_split(
            "calibration_a",
            raw["calibration_a"],
        ),
        calibration_b=_normalized_prompt_split(
            "calibration_b",
            raw["calibration_b"],
        ),
        validation=_normalized_prompt_split(
            "validation",
            raw["validation"],
        ),
        test=_normalized_prompt_split(
            "test",
            raw["test"],
        ),
        scientific_status=str(raw["scientific_status"]),
    )


def _validated_ranks(ranks: Iterable[int]) -> tuple[int, ...]:
    if isinstance(ranks, (str, bytes)):
        raise TypeError("ranks must be an iterable of positive integers")
    try:
        requested = tuple(ranks)
    except TypeError as error:
        raise TypeError(
            "ranks must be an iterable of positive integers"
        ) from error
    if not requested:
        raise ValueError("ranks cannot be empty")
    if any(type(rank) is not int or rank <= 0 for rank in requested):
        raise ValueError("ranks must contain positive integers")
    return tuple(sorted(set(requested)))


def _collect_split(
    *,
    split_name: str,
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
    prompts: Sequence[str],
    activation_names: tuple[str, ...],
    leaf_activation_name: str,
    max_length: int,
    tokenization_batch_size: int,
    device: torch.device,
    rank: int,
    sketch_rows: int,
) -> tuple[StreamingFisherCollection, dict[str, object]]:
    provenance = _CalibrationStreamProvenance(split_name, prompts)
    batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompts,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    collection = collect_streaming_fisher_modes(
        adapter,
        provenance.wrap(batches),
        activation_names=activation_names,
        score_objective=CausalLanguageModelNLL(),
        rank=rank,
        sketch_rows=sketch_rows,
        leaf_activation_name=leaf_activation_name,
    )
    return collection, provenance.metadata()


def _replay_validation(
    *,
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
    prompts: Sequence[str],
    calibration: Mapping[str, StreamingFisherCollection],
    activation_names: tuple[str, ...],
    leaf_activation_name: str,
    max_length: int,
    tokenization_batch_size: int,
    device: torch.device,
    rank: int,
) -> tuple[
    dict[str, dict[str, StreamingRayleighEnergyResult]],
    float,
    int,
    dict[str, object],
]:
    estimators = {
        split_name: {
            activation_name: (
                StreamingRayleighEnergyEstimator.from_fisher_result(
                    collection.bases[activation_name].fisher,
                    rank=rank,
                )
            )
            for activation_name in activation_names
        }
        for split_name, collection in calibration.items()
    }
    provenance = _CalibrationStreamProvenance("validation", prompts)
    batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompts,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    loss_total = 0.0
    sequences = 0
    rows = iter_activation_score_gradient_rows(
        adapter,
        provenance.wrap(batches),
        activation_names=activation_names,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=leaf_activation_name,
    )
    try:
        for sample in rows:
            for split_estimators in estimators.values():
                for activation_name, estimator in split_estimators.items():
                    estimator.update(sample.score_gradients[activation_name])
            loss_total += sample.loss
            sequences += 1
    finally:
        rows.close()
    if sequences == 0:
        raise ValueError("validation prompt stream cannot be empty")
    return (
        {
            split_name: {
                activation_name: estimator.finalize()
                for activation_name, estimator in split_estimators.items()
            }
            for split_name, split_estimators in estimators.items()
        },
        loss_total / sequences,
        sequences,
        provenance.metadata(),
    )


def _prefix_sketch_fraction(
    collection: StreamingFisherCollection,
    activation_name: str,
    rank: int,
) -> float:
    result = collection.bases[activation_name].fisher
    if result.fisher_trace == 0:
        return 0.0
    return min(
        result.eigenvalues[:rank].sum().item() / result.fisher_trace,
        1.0,
    )


def _relative_eigengap(
    collection: StreamingFisherCollection,
    activation_name: str,
    rank: int,
) -> float | None:
    eigenvalues = collection.bases[activation_name].eigenvalues
    if rank >= eigenvalues.numel():
        return None
    boundary = eigenvalues[rank - 1].item()
    following = eigenvalues[rank].item()
    if boundary == 0:
        return 0.0
    return max((boundary - following) / boundary, 0.0)


def _rank_curve(
    *,
    activation_name: str,
    ranks: Sequence[int],
    calibration: Mapping[str, StreamingFisherCollection],
    validation: Mapping[
        str,
        Mapping[str, StreamingRayleighEnergyResult],
    ],
    stability: FisherSubspaceStability,
) -> list[dict[str, object]]:
    points_by_rank = {point.rank: point for point in stability.points}
    curve = []
    for rank in ranks:
        point = points_by_rank[rank]
        validation_a = validation["calibration_a"][
            activation_name
        ].retained_fraction(rank)
        validation_b = validation["calibration_b"][
            activation_name
        ].retained_fraction(rank)
        curve.append(
            {
                "rank": rank,
                "calibration_a_sketch_fraction": (
                    _prefix_sketch_fraction(
                        calibration["calibration_a"],
                        activation_name,
                        rank,
                    )
                ),
                "calibration_b_sketch_fraction": (
                    _prefix_sketch_fraction(
                        calibration["calibration_b"],
                        activation_name,
                        rank,
                    )
                ),
                "calibration_full_sketch_fraction": (
                    _prefix_sketch_fraction(
                        calibration["calibration_full"],
                        activation_name,
                        rank,
                    )
                ),
                "calibration_a_relative_eigengap": _relative_eigengap(
                    calibration["calibration_a"],
                    activation_name,
                    rank,
                ),
                "calibration_b_relative_eigengap": _relative_eigengap(
                    calibration["calibration_b"],
                    activation_name,
                    rank,
                ),
                "calibration_full_relative_eigengap": _relative_eigengap(
                    calibration["calibration_full"],
                    activation_name,
                    rank,
                ),
                "split_mean_squared_overlap": (
                    point.mean_squared_overlap
                ),
                "split_minimum_principal_cosine": (
                    point.minimum_principal_cosine
                ),
                "split_largest_principal_angle_degrees": (
                    point.largest_principal_angle_degrees
                ),
                "validation_a_exact_rayleigh_fraction": validation_a,
                "validation_b_exact_rayleigh_fraction": validation_b,
                "validation_full_exact_rayleigh_fraction": (
                    validation["calibration_full"][
                        activation_name
                    ].retained_fraction(rank)
                ),
                "validation_exact_rayleigh_fraction_min": min(
                    validation_a,
                    validation_b,
                ),
                "validation_exact_rayleigh_fraction_gap": abs(
                    validation_a - validation_b
                ),
            }
        )
    return curve


def _save_stability_artifact(
    *,
    output: Path,
    model: Mapping[str, object],
    protocol: Mapping[str, object],
    calibration: Mapping[str, StreamingFisherCollection],
    validation: Mapping[
        str,
        Mapping[str, StreamingRayleighEnergyResult],
    ],
    tokenized_splits: Mapping[str, Mapping[str, object]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": _ARTIFACT_SCHEMA,
            "format_version": _ARTIFACT_FORMAT_VERSION,
            "contains_model_weights": False,
            "model": dict(model),
            "protocol": dict(protocol),
            "calibration": {
                name: {
                    "collection": collection.state_dict(),
                    "tokenized_stream": copy.deepcopy(
                        tokenized_splits[name]
                    ),
                }
                for name, collection in calibration.items()
            },
            "validation": {
                "tokenized_stream": copy.deepcopy(
                    tokenized_splits["validation"]
                ),
                "bases": {
                    split_name: {
                        activation_name: result.state_dict()
                        for activation_name, result in results.items()
                    }
                    for split_name, results in validation.items()
                },
            },
        },
        output,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_tokenized_stream(
    value: object,
    *,
    split_name: str,
) -> tuple[dict[str, object], int]:
    if not isinstance(value, Mapping):
        raise TypeError("tokenized stream provenance must be a mapping")
    expected = {
        "schema",
        "format_version",
        "split",
        "batches",
        "sequences",
        "serialized_sha256",
        "source_prompt_sha256",
        "examples",
    }
    if set(value) != expected:
        raise ValueError("tokenized stream provenance fields are invalid")
    if (
        value["schema"] != "fisher_graph.tokenized_calibration_stream"
        or value["format_version"] != 2
        or value["split"] != split_name
    ):
        raise ValueError("tokenized stream provenance identity is invalid")
    batches = value["batches"]
    sequences = value["sequences"]
    if (
        type(batches) is not int
        or type(sequences) is not int
        or not 1 <= batches <= sequences
    ):
        raise ValueError("tokenized stream counts are invalid")
    if not _is_sha256(value["serialized_sha256"]):
        raise ValueError("tokenized stream digest is invalid")
    source_prompt_hashes = value["source_prompt_sha256"]
    if (
        not isinstance(source_prompt_hashes, list)
        or len(source_prompt_hashes) != sequences
        or any(not _is_sha256(item) for item in source_prompt_hashes)
        or len(set(source_prompt_hashes)) != sequences
    ):
        raise ValueError("tokenized source prompt digests are invalid")
    examples = value["examples"]
    if not isinstance(examples, list) or len(examples) != sequences:
        raise ValueError("tokenized stream examples are invalid")
    example_ids = set()
    total_valid_tokens = 0
    for index, raw_example in enumerate(examples):
        if not isinstance(raw_example, Mapping) or set(raw_example) != {
            "example_id",
            "serialized_sha256",
            "content_sha256",
            "valid_tokens",
            "supervised_positions",
        }:
            raise ValueError("tokenized example provenance fields are invalid")
        example_id = raw_example["example_id"]
        if (
            not isinstance(example_id, str)
            or example_id != f"prompt.{index:06d}"
            or example_id in example_ids
        ):
            raise ValueError("tokenized example IDs are invalid")
        if not _is_sha256(raw_example["serialized_sha256"]):
            raise ValueError("tokenized example digest is invalid")
        if not _is_sha256(raw_example["content_sha256"]):
            raise ValueError("tokenized example content digest is invalid")
        valid_tokens = raw_example["valid_tokens"]
        supervised = raw_example["supervised_positions"]
        if (
            type(valid_tokens) is not int
            or valid_tokens < 2
            or type(supervised) is not int
            or supervised != valid_tokens - 1
        ):
            raise ValueError("tokenized example token counts are invalid")
        example_ids.add(example_id)
        total_valid_tokens += valid_tokens
    return copy.deepcopy(dict(value)), total_valid_tokens


def _validate_prompt_split_metadata(
    value: object,
    *,
    streams: Mapping[str, Mapping[str, object]],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "scientific_status",
        "counts",
        "normalized_sha256",
        "per_prompt_sha256",
    }:
        raise ValueError("prompt split provenance fields are invalid")
    if not isinstance(value["scientific_status"], str) or not value[
        "scientific_status"
    ]:
        raise ValueError("prompt split scientific status is invalid")
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
        raise ValueError("prompt split provenance mappings are invalid")
    for split_name in split_names:
        count = counts[split_name]
        hashes = per_prompt[split_name]
        if (
            type(count) is not int
            or count <= 0
            or not isinstance(hashes, list)
            or len(hashes) != count
            or not _is_sha256(normalized[split_name])
            or any(not _is_sha256(item) for item in hashes)
        ):
            raise ValueError("prompt split counts or digests are invalid")
        if normalized[split_name] != _ordered_prompt_hash_digest(hashes):
            raise ValueError(
                "prompt split aggregate digest does not match prompt hashes"
            )
    all_prompt_hashes = [
        digest
        for split_name in split_names
        for digest in per_prompt[split_name]
    ]
    if len(set(all_prompt_hashes)) != len(all_prompt_hashes):
        raise ValueError("prompt split hashes must be pairwise disjoint")
    for split_name in ("calibration_a", "calibration_b", "validation"):
        if counts[split_name] != streams[split_name]["sequences"]:
            raise ValueError(
                "prompt split count does not match tokenized provenance"
            )
    if streams["calibration_full"]["sequences"] != (
        counts["calibration_a"] + counts["calibration_b"]
    ):
        raise ValueError(
            "combined calibration count does not match prompt splits"
        )
    for split_name in ("calibration_a", "calibration_b", "validation"):
        if (
            streams[split_name]["source_prompt_sha256"]
            != per_prompt[split_name]
        ):
            raise ValueError(
                "tokenized source prompts do not match prompt provenance"
            )
    expected_full_prompt_hashes = (
        per_prompt["calibration_a"] + per_prompt["calibration_b"]
    )
    if (
        streams["calibration_full"]["source_prompt_sha256"]
        != expected_full_prompt_hashes
    ):
        raise ValueError(
            "combined calibration source prompts are not calibration A+B"
        )
    streamed_prompt_hashes = {
        digest
        for split_name in (
            "calibration_a",
            "calibration_b",
            "calibration_full",
            "validation",
        )
        for digest in streams[split_name]["source_prompt_sha256"]
    }
    if streamed_prompt_hashes & set(per_prompt["test"]):
        raise ValueError("reserved test prompts appear in a tokenized stream")
    expected_full_content = [
        example["content_sha256"]
        for split_name in ("calibration_a", "calibration_b")
        for example in streams[split_name]["examples"]
    ]
    actual_full_content = [
        example["content_sha256"]
        for example in streams["calibration_full"]["examples"]
    ]
    if actual_full_content != expected_full_content:
        raise ValueError(
            "combined calibration token content is not calibration A+B"
        )


def _validate_stability_protocol(
    protocol: Mapping[str, object],
    *,
    calibration: Mapping[str, StreamingFisherCollection],
    validation: Mapping[
        str,
        Mapping[str, StreamingRayleighEnergyResult],
    ],
    streams: Mapping[str, Mapping[str, object]],
    valid_token_totals: Mapping[str, int],
) -> None:
    expected = {
        "layer_index",
        "activation_sites",
        "ranks",
        "maximum_rank",
        "extraction_rank",
        "sketch_rows",
        "maximum_tokenized_length",
        "gradient_batching",
        "score",
        "score_compute_dtype",
        "scope",
        "normalizer",
        "leaf_boundary",
        "cache_policy",
        "library_versions",
        "tokenizer",
        "tokenized_splits",
        "prompt_splits",
    }
    if set(protocol) != expected:
        raise ValueError("stability protocol fields are invalid")
    layer_index = protocol["layer_index"]
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("stability protocol layer index is invalid")
    expected_sites = (
        f"layer.{layer_index}.input",
        f"layer.{layer_index}.output",
    )
    if protocol["activation_sites"] != expected_sites:
        raise ValueError("stability protocol activation sites are invalid")
    if any(
        tuple(collection.bases) != expected_sites
        for collection in calibration.values()
    ):
        raise ValueError(
            "calibration activation sites do not match the protocol"
        )
    raw_ranks = protocol["ranks"]
    if type(raw_ranks) is not tuple:
        raise ValueError("stability protocol ranks are invalid")
    ranks = _validated_ranks(raw_ranks)
    if ranks != raw_ranks:
        raise ValueError("stability protocol ranks are not canonical")
    maximum_rank = protocol["maximum_rank"]
    extraction_rank = protocol["extraction_rank"]
    sketch_rows = protocol["sketch_rows"]
    if type(maximum_rank) is not int or maximum_rank != max(ranks):
        raise ValueError("stability protocol maximum rank is invalid")
    if (
        type(extraction_rank) is not int
        or not maximum_rank <= extraction_rank <= maximum_rank + 1
    ):
        raise ValueError("stability protocol extraction rank is invalid")
    if type(sketch_rows) is not int or sketch_rows <= extraction_rank:
        raise ValueError("stability protocol sketch rows are invalid")
    if protocol["leaf_boundary"] != expected_sites[0]:
        raise ValueError("stability protocol leaf boundary is invalid")
    fixed_fields = {
        "gradient_batching": "one_sequence_at_a_time",
        "score": "summed_hard_target_next_token_nll",
        "score_compute_dtype": (
            "float32_for_float16_or_bfloat16_logits"
        ),
        "scope": "width_pooled",
        "normalizer": "valid_activation_positions",
        "cache_policy": "external_to_git_worktree",
    }
    if any(protocol[name] != value for name, value in fixed_fields.items()):
        raise ValueError("stability protocol scientific semantics are invalid")
    maximum_length = protocol["maximum_tokenized_length"]
    if (
        type(maximum_length) is not int
        or maximum_length < 2
        or maximum_length
        < max(
            example["valid_tokens"]
            for stream in streams.values()
            for example in stream["examples"]
        )
    ):
        raise ValueError("stability protocol token limit is invalid")
    raw_streams = protocol["tokenized_splits"]
    if not isinstance(raw_streams, Mapping) or set(raw_streams) != set(
        streams
    ):
        raise ValueError("stability protocol tokenized splits are invalid")
    if any(raw_streams[name] != streams[name] for name in streams):
        raise ValueError(
            "stability protocol tokenized provenance does not match payloads"
        )
    libraries = protocol["library_versions"]
    if not isinstance(libraries, Mapping) or set(libraries) != {
        "python",
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
    }:
        raise ValueError("stability protocol library versions are invalid")
    if any(
        value is not None and not isinstance(value, str)
        for value in libraries.values()
    ):
        raise ValueError("stability protocol library version is invalid")
    tokenizer = protocol["tokenizer"]
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
        "tokenizer_class",
        "name_or_path",
        "configuration_sha256",
    }:
        raise ValueError("stability protocol tokenizer provenance is invalid")
    if (
        not isinstance(tokenizer["tokenizer_class"], str)
        or not tokenizer["tokenizer_class"]
        or (
            tokenizer["name_or_path"] is not None
            and not isinstance(tokenizer["name_or_path"], str)
        )
        or not _is_sha256(tokenizer["configuration_sha256"])
    ):
        raise ValueError("stability protocol tokenizer metadata is invalid")
    _validate_prompt_split_metadata(
        protocol["prompt_splits"],
        streams=streams,
    )
    for split_name, collection in calibration.items():
        stream = streams[split_name]
        if collection.sequences != stream["sequences"]:
            raise ValueError(
                "calibration sequence count does not match tokenized stream"
            )
        for basis in collection.bases.values():
            fisher = basis.fisher
            if (
                basis.observations != valid_token_totals[split_name]
                or fisher.rows_seen != valid_token_totals[split_name]
                or fisher.requested_rank != extraction_rank
                or fisher.modes != extraction_rank
                or fisher.sketch_rows != sketch_rows
                or fisher.scope != protocol["scope"]
                or fisher.score_reduction != "sum"
                or fisher.normalizer != protocol["normalizer"]
            ):
                raise ValueError(
                    "calibration Fisher payload does not match its protocol"
                )
    if (
        streams["calibration_full"]["sequences"]
        != streams["calibration_a"]["sequences"]
        + streams["calibration_b"]["sequences"]
        or valid_token_totals["calibration_full"]
        != valid_token_totals["calibration_a"]
        + valid_token_totals["calibration_b"]
    ):
        raise ValueError("combined calibration stream is inconsistent")
    for results in validation.values():
        for result in results.values():
            if (
                result.observations != valid_token_totals["validation"]
                or result.rows_seen != valid_token_totals["validation"]
            ):
                raise ValueError(
                    "validation replay count does not match tokenized stream"
                )


def load_gemma3_stability_artifact(
    path: Path | str,
) -> tuple[
    dict[str, StreamingFisherCollection],
    dict[str, dict[str, StreamingRayleighEnergyResult]],
    dict[str, object],
]:
    """Load a strict analysis-only stability artifact."""

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise TypeError("Gemma 3 stability artifact must contain a mapping")
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "model",
        "protocol",
        "calibration",
        "validation",
    }
    if set(raw) != required:
        raise ValueError(
            "Gemma 3 stability artifact fields do not match format version 2"
        )
    if raw["schema"] != _ARTIFACT_SCHEMA:
        raise ValueError("unsupported Gemma 3 stability artifact schema")
    if raw["format_version"] != _ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported Gemma 3 stability artifact format")
    if raw["contains_model_weights"] is not False:
        raise ValueError("stability artifact unexpectedly claims model weights")
    raw_model = raw["model"]
    raw_protocol = raw["protocol"]
    raw_calibration = raw["calibration"]
    raw_validation = raw["validation"]
    if not isinstance(raw_model, Mapping) or not isinstance(
        raw_protocol,
        Mapping,
    ):
        raise TypeError("artifact model and protocol must be mappings")
    if raw_model.get("weights_in_artifact") is not False:
        raise ValueError("artifact model metadata does not exclude weights")
    if not isinstance(raw_calibration, Mapping) or set(raw_calibration) != {
        "calibration_a",
        "calibration_b",
        "calibration_full",
    }:
        raise ValueError("artifact calibration splits are invalid")
    if not isinstance(raw_validation, Mapping) or set(raw_validation) != {
        "tokenized_stream",
        "bases",
    }:
        raise ValueError("artifact validation payload is invalid")
    calibration = {}
    streams = {}
    valid_token_totals = {}
    for name, entry in raw_calibration.items():
        if (
            not isinstance(name, str)
            or not isinstance(entry, Mapping)
            or set(entry) != {"collection", "tokenized_stream"}
        ):
            raise TypeError("calibration states must be named mappings")
        state = entry["collection"]
        if not isinstance(state, Mapping):
            raise TypeError("calibration collection state must be a mapping")
        calibration[name] = StreamingFisherCollection.from_state_dict(state)
        stream, valid_tokens = _validated_tokenized_stream(
            entry["tokenized_stream"],
            split_name=name,
        )
        streams[name] = stream
        valid_token_totals[name] = valid_tokens
    validation_stream, validation_valid_tokens = (
        _validated_tokenized_stream(
            raw_validation["tokenized_stream"],
            split_name="validation",
        )
    )
    streams["validation"] = validation_stream
    valid_token_totals["validation"] = validation_valid_tokens
    raw_validation_bases = raw_validation["bases"]
    if (
        not isinstance(raw_validation_bases, Mapping)
        or set(raw_validation_bases)
        != {
            "calibration_a",
            "calibration_b",
            "calibration_full",
        }
    ):
        raise ValueError("artifact validation basis splits are invalid")
    validation = {}
    for split_name, raw_results in raw_validation_bases.items():
        if not isinstance(split_name, str) or not isinstance(
            raw_results,
            Mapping,
        ):
            raise TypeError("validation states must be named mappings")
        results = {}
        for activation_name, state in raw_results.items():
            if not isinstance(activation_name, str) or not isinstance(
                state,
                Mapping,
            ):
                raise TypeError(
                    "validation result states must be named mappings"
                )
            results[activation_name] = (
                StreamingRayleighEnergyResult.from_state_dict(state)
            )
        validation[split_name] = results
    expected_sites = set(calibration["calibration_a"].bases)
    if any(
        set(collection.bases) != expected_sites
        for collection in calibration.values()
    ):
        raise ValueError("calibration artifacts disagree on activation sites")
    if any(set(results) != expected_sites for results in validation.values()):
        raise ValueError("validation artifacts disagree on activation sites")
    _validate_stability_protocol(
        raw_protocol,
        calibration=calibration,
        validation=validation,
        streams=streams,
        valid_token_totals=valid_token_totals,
    )
    maximum_rank = raw_protocol.get("maximum_rank")
    if type(maximum_rank) is not int or maximum_rank <= 0:
        raise ValueError("artifact protocol maximum_rank is invalid")
    accumulation_dtypes = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    for split_name, results in validation.items():
        for activation_name, result in results.items():
            if result.activation_name != activation_name:
                raise ValueError(
                    "validation result activation name does not match its key"
                )
            source = calibration[split_name].bases[
                activation_name
            ].fisher
            if result.width != source.width:
                raise ValueError(
                    "validation result width does not match its frozen basis"
                )
            if result.modes != maximum_rank:
                raise ValueError(
                    "validation result mode count does not match maximum_rank"
                )
            for field in ("scope", "score_reduction", "normalizer"):
                if getattr(result, field) != getattr(source, field):
                    raise ValueError(
                        "validation result provenance does not match its "
                        f"frozen basis: {field}"
                    )
            try:
                accumulation_dtype = accumulation_dtypes[
                    result.accumulation_dtype
                ]
            except KeyError as error:
                raise ValueError(
                    "validation result accumulation dtype is unsupported"
                ) from error
            expected_estimator = (
                StreamingRayleighEnergyEstimator.from_fisher_result(
                    source,
                    rank=maximum_rank,
                    accumulation_dtype=accumulation_dtype,
                )
            )
            if result.basis_sha256 != expected_estimator.basis_sha256:
                raise ValueError(
                    "validation result is not bound to its named frozen basis"
                )
    for activation_name in expected_sites:
        reference = validation["calibration_a"][activation_name]
        reference_accounting = (
            reference.observations,
            reference.nonzero_observations,
            reference.rows_seen,
            reference.squared_gradient_norm_sum,
            reference.fisher_trace,
            reference.accumulation_dtype,
        )
        for split_name in ("calibration_b", "calibration_full"):
            candidate = validation[split_name][activation_name]
            candidate_accounting = (
                candidate.observations,
                candidate.nonzero_observations,
                candidate.rows_seen,
                candidate.squared_gradient_norm_sum,
                candidate.fisher_trace,
                candidate.accumulation_dtype,
            )
            if candidate_accounting != reference_accounting:
                raise ValueError(
                    "validation results do not share held-out row accounting"
                )
    return (
        calibration,
        validation,
        {
            "schema": raw["schema"],
            "format_version": raw["format_version"],
            "contains_model_weights": False,
            "model": dict(raw_model),
            "protocol": dict(raw_protocol),
        },
    )


def run_gemma3_stability(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    layer_index: int = 0,
    max_length: int = 128,
    tokenization_batch_size: int = 4,
    ranks: Iterable[int] = DEFAULT_RANKS,
    sketch_rows: int | None = None,
    device_name: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Run split-specific extraction, stability, and exact held-out replay."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    requested_ranks = _validated_ranks(ranks)
    maximum_rank = max(requested_ranks)
    resolved_sketch_rows = (
        2 * maximum_rank if sketch_rows is None else sketch_rows
    )
    if (
        type(resolved_sketch_rows) is not int
        or resolved_sketch_rows <= maximum_rank
    ):
        raise ValueError(
            "sketch_rows must be an integer greater than the maximum rank"
        )
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    resolved_output = (
        default_gemma3_stability_output(model_id, layer_index)
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
    adapter = Gemma3CausalLMAdapter(model)
    if layer_index >= len(adapter.layers):
        raise ValueError(
            f"layer_index must be between 0 and {len(adapter.layers) - 1}"
        )
    layer = adapter.layers[layer_index]
    if maximum_rank > layer.residual_width:
        raise ValueError(
            f"maximum rank cannot exceed residual width "
            f"{layer.residual_width}"
        )
    # Retain one extra mode when the width and sketch permit it so the
    # eigengap at the largest requested prefix has a following eigenvalue.
    extraction_rank = min(
        maximum_rank + 1,
        layer.residual_width,
        resolved_sketch_rows - 1,
    )
    activation_names = (layer.input_site, layer.output_site)
    calibration = {}
    tokenized_splits = {}
    calibration_prompts = {
        "calibration_a": prompt_splits.calibration_a,
        "calibration_b": prompt_splits.calibration_b,
        "calibration_full": (
            prompt_splits.calibration_a
            + prompt_splits.calibration_b
        ),
    }
    for split_name, prompts in calibration_prompts.items():
        collection, tokenized = _collect_split(
            split_name=split_name,
            adapter=adapter,
            tokenizer=tokenizer,
            prompts=prompts,
            activation_names=activation_names,
            leaf_activation_name=layer.input_site,
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
            rank=extraction_rank,
            sketch_rows=resolved_sketch_rows,
        )
        calibration[split_name] = collection
        tokenized_splits[split_name] = tokenized
    stability = {
        activation_name: compare_fisher_subspaces(
            calibration["calibration_a"]
            .bases[activation_name]
            .fisher,
            calibration["calibration_b"]
            .bases[activation_name]
            .fisher,
            ranks=requested_ranks,
        )
        for activation_name in activation_names
    }
    (
        validation,
        validation_mean_loss,
        validation_sequences,
        validation_tokenized,
    ) = (
        _replay_validation(
            adapter=adapter,
            tokenizer=tokenizer,
            prompts=prompt_splits.validation,
            calibration=calibration,
            activation_names=activation_names,
            leaf_activation_name=layer.input_site,
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
            rank=maximum_rank,
        )
    )
    tokenized_splits["validation"] = validation_tokenized
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    protocol = {
        "layer_index": layer_index,
        "activation_sites": activation_names,
        "ranks": requested_ranks,
        "maximum_rank": maximum_rank,
        "extraction_rank": extraction_rank,
        "sketch_rows": resolved_sketch_rows,
        "maximum_tokenized_length": max_length,
        "gradient_batching": "one_sequence_at_a_time",
        "score": "summed_hard_target_next_token_nll",
        "score_compute_dtype": (
            "float32_for_float16_or_bfloat16_logits"
        ),
        "scope": "width_pooled",
        "normalizer": "valid_activation_positions",
        "leaf_boundary": layer.input_site,
        "cache_policy": "external_to_git_worktree",
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "tokenized_splits": tokenized_splits,
        "prompt_splits": prompt_splits.metadata(),
    }
    analysis = {
        "calibration": {
            name: collection.metadata()
            for name, collection in calibration.items()
        },
        "stability": {
            name: report.metadata()
            for name, report in stability.items()
        },
        "validation": {
            "mean_loss": validation_mean_loss,
            "sequences": validation_sequences,
            "bases": {
                split_name: {
                    activation_name: result.metadata()
                    for activation_name, result in results.items()
                }
                for split_name, results in validation.items()
            },
        },
        "rank_curve": {
            activation_name: _rank_curve(
                activation_name=activation_name,
                ranks=requested_ranks,
                calibration=calibration,
                validation=validation,
                stability=stability[activation_name],
            )
            for activation_name in activation_names
        },
    }
    report = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": {
            "scope": "split_stability_and_exact_validation_replay",
            "prompt_protocol": prompt_splits.scientific_status,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "compilation_claim": False,
            "quality_validation_claim": False,
            "acceptance_thresholds_defined": False,
            "test_split_evaluated": False,
        },
        "model": model_metadata,
        "protocol": protocol,
        "analysis": analysis,
        "artifact": {
            "tensor_output": resolved_output.name,
            "contains_model_state_dict": False,
            "contains_tokenizer": False,
        },
    }
    _save_stability_artifact(
        output=resolved_output,
        model=model_metadata,
        protocol=protocol,
        calibration=calibration,
        validation=validation,
        tokenized_splits=tokenized_splits,
    )
    resolved_output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Gemma 3 Fisher subspace stability across two frozen "
            "calibration splits and exact Rayleigh energy on validation."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help=(
            "validate and print every Hugging Face write path, then exit "
            "without importing Transformers or loading a model"
        ),
    )
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=DEFAULT_RANKS,
    )
    parser.add_argument("--sketch-rows", type=int)
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
        else default_gemma3_stability_output(
            arguments.model,
            arguments.layer_index,
        )
    )
    report = run_gemma3_stability(
        model_id=arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        layer_index=arguments.layer_index,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        ranks=arguments.ranks,
        sketch_rows=arguments.sketch_rows,
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=output,
    )
    analysis = report["analysis"]
    assert isinstance(analysis, Mapping)
    rank_curve = analysis["rank_curve"]
    assert isinstance(rank_curve, Mapping)
    maximum_rank = max(_validated_ranks(arguments.ranks))
    print(
        f"Wrote split stability and exact validation replay to {output}"
    )
    for activation_name, raw_curve in rank_curve.items():
        assert isinstance(raw_curve, list)
        point = next(
            item
            for item in raw_curve
            if isinstance(item, Mapping)
            and item.get("rank") == maximum_rank
        )
        print(
            f"{activation_name} rank {maximum_rank}: "
            f"overlap={point['split_mean_squared_overlap']:.4f}, "
            "validation_full="
            f"{point['validation_full_exact_rayleigh_fraction']:.4f}"
        )
    print(f"Report: {output.with_suffix('.json')}")
    print("No pretrained model weights were written to either output.")


if __name__ == "__main__":
    main()
