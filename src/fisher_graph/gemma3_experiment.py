"""Opt-in one-layer Gemma 3 streaming activation-Fisher experiment.

This module never vendors, copies, or serializes pretrained model weights.
Hugging Face acquisition is lazy and its writable paths are required to live
outside the active Git worktree.  The output contains only small analysis
tensors and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import (
    CalibrationBatch,
    CausalLanguageModelNLL,
)
from .external_models import (
    external_huggingface_cache_dir,
    find_git_worktree,
    huggingface_local_paths,
)
from .streaming_analysis import (
    StreamingFisherCollection,
    collect_streaming_fisher_modes,
)


DEFAULT_MODEL_ID = "google/gemma-3-270m"


def default_gemma3_output(
    model_id: str = DEFAULT_MODEL_ID,
    layer_index: int = 0,
) -> Path:
    """Return an ignored output path that identifies the model and layer."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("._-")
    if not slug:
        slug = "gemma3-model"
    return (
        Path(".local-runs")
        / slug
        / f"layer-{layer_index}-streaming-fisher.pt"
    )


DEFAULT_OUTPUT = default_gemma3_output()
DEFAULT_SMOKE_PROMPTS = (
    "The quiet library opened before sunrise.",
    "A small red boat crossed the lake.",
    "Write one sentence about a patient scientist.",
    "Music can make a complicated pattern easier to hear.",
    "The compiler replaced one layer and checked the answer.",
    "An apple, a pear, and an orange sat on the table.",
    "When the rain stopped, the street reflected the sky.",
    "A careful experiment changes one thing at a time.",
)
_ARTIFACT_SCHEMA = "fisher_graph.gemma3_streaming_fisher"
_ARTIFACT_FORMAT_VERSION = 1


def _transformers_classes() -> tuple[type[Any], type[Any], str]:
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Gemma support is optional; install it with "
            '`pip install -e ".[gemma]"`'
        ) from error
    return AutoTokenizer, AutoModelForCausalLM, transformers.__version__


def resolve_torch_device(name: str) -> torch.device:
    """Resolve a requested analysis device without importing Transformers."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested but is not available")
    return device


def _model_dtype(name: str) -> str | torch.dtype:
    values: dict[str, str | torch.dtype] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported model dtype: {name!r}") from error


def _model_dtype_load_kwargs(
    name: str,
    transformers_version: str,
) -> dict[str, str | torch.dtype]:
    """Use the dtype spelling supported by the installed Transformers major."""

    match = re.match(r"^(\d+)", transformers_version)
    if match is None:
        raise RuntimeError(
            "could not determine the installed Transformers major version "
            f"from {transformers_version!r}"
        )
    keyword = "dtype" if int(match.group(1)) >= 5 else "torch_dtype"
    return {keyword: _model_dtype(name)}


def load_gemma3(
    *,
    model_id: str,
    revision: str | None,
    cache_dir: Path,
    device: torch.device,
    dtype: str,
    local_files_only: bool,
) -> tuple[object, nn.Module]:
    """Load a tokenizer and text causal LM from an already-validated cache."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if revision is not None and (
        not isinstance(revision, str) or not revision
    ):
        raise ValueError("revision must be a nonempty string when provided")
    tokenizer_class, model_class, transformers_version = _transformers_classes()
    common: dict[str, object] = {
        "cache_dir": str(cache_dir),
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    try:
        tokenizer = tokenizer_class.from_pretrained(
            model_id,
            use_fast=True,
            **common,
        )
        model = model_class.from_pretrained(
            model_id,
            use_safetensors=True,
            attn_implementation="eager",
            **_model_dtype_load_kwargs(dtype, transformers_version),
            **common,
        )
    except OSError as error:
        raise RuntimeError(
            f"could not load {model_id!r}; accept its Gemma license on "
            "Hugging Face, authenticate with `hf auth login`, and verify "
            "the requested revision/cache"
        ) from error
    if not isinstance(model, nn.Module):
        raise TypeError("AutoModelForCausalLM did not return a torch module")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    disable_checkpointing = getattr(
        model,
        "gradient_checkpointing_disable",
        None,
    )
    if callable(disable_checkpointing):
        disable_checkpointing()
    return tokenizer, model


def read_prompts(
    *,
    prompt_file: Path | None,
    inline_prompts: Sequence[str],
) -> tuple[str, ...]:
    """Read nonempty prompts, falling back to a small plumbing smoke set."""

    if any(not isinstance(prompt, str) for prompt in inline_prompts):
        raise TypeError("inline prompts must be strings")
    explicit_source = prompt_file is not None or bool(inline_prompts)
    prompts = []
    if prompt_file is not None:
        prompts.extend(prompt_file.read_text(encoding="utf-8").splitlines())
    prompts.extend(inline_prompts)
    normalized = tuple(prompt.strip() for prompt in prompts if prompt.strip())
    if not normalized:
        if explicit_source:
            raise ValueError(
                "explicit calibration prompts contain no nonempty text"
            )
        return DEFAULT_SMOKE_PROMPTS
    return normalized


def _tokenizer_output_tensor(
    encoded: object,
    name: str,
) -> Tensor:
    if isinstance(encoded, Mapping):
        value = encoded.get(name)
    else:
        value = getattr(encoded, name, None)
    if not isinstance(value, Tensor):
        raise TypeError(f"tokenizer output must contain Tensor {name!r}")
    return value


def make_causal_lm_calibration_batches(
    tokenizer: object,
    prompts: Sequence[str],
    *,
    max_length: int,
    tokenization_batch_size: int,
    device: torch.device,
    ignore_index: int = -100,
) -> Iterator[CalibrationBatch]:
    """Tokenize text and build explicit next-token targets.

    The logits at position ``t`` are supervised with the token at ``t + 1``.
    Padding transitions and the final token have ``ignore_index`` targets.
    """

    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if type(tokenization_batch_size) is not int or tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
    if not prompts:
        raise ValueError("at least one prompt is required")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must be nonempty strings")
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("tokenizer must define a pad or EOS token")
        setattr(tokenizer, "pad_token", eos_token)
    if hasattr(tokenizer, "padding_side"):
        setattr(tokenizer, "padding_side", "right")

    def generate() -> Iterator[CalibrationBatch]:
        for start in range(0, len(prompts), tokenization_batch_size):
            prompt_chunk = tuple(
                prompts[start : start + tokenization_batch_size]
            )
            encoded = tokenizer(
                list(prompt_chunk),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            input_ids = _tokenizer_output_tensor(encoded, "input_ids")
            attention_mask = _tokenizer_output_tensor(
                encoded,
                "attention_mask",
            )
            if (
                input_ids.ndim != 2
                or attention_mask.shape != input_ids.shape
                or input_ids.shape[0] != len(prompt_chunk)
            ):
                raise ValueError(
                    "tokenizer tensors must have aligned "
                    "[batch, sequence] shapes"
                )
            valid = attention_mask.to(dtype=torch.bool)
            targets = torch.full_like(input_ids, ignore_index)
            supervised = valid[:, :-1] & valid[:, 1:]
            targets[:, :-1] = torch.where(
                supervised,
                input_ids[:, 1:],
                torch.full_like(input_ids[:, 1:], ignore_index),
            )
            for offset in range(len(prompt_chunk)):
                if not (targets[offset] != ignore_index).any():
                    raise ValueError(
                        f"prompt {start + offset} has fewer than two tokens "
                        "after tokenization"
                    )
            yield CalibrationBatch(
                model_inputs={
                    "input_ids": input_ids.to(device),
                    "attention_mask": valid.to(device),
                },
                targets=targets.to(device),
                valid_positions=valid.to(device),
                example_ids=tuple(
                    f"prompt.{index:06d}"
                    for index in range(start, start + len(prompt_chunk))
                ),
            )

    return generate()


def _json_compatible(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _model_provenance(
    model: nn.Module,
    *,
    model_id: str,
    requested_revision: str | None,
) -> dict[str, object]:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("loaded model does not expose config")
    to_dict = getattr(config, "to_dict", None)
    config_payload = (
        to_dict()
        if callable(to_dict)
        else {
            name: getattr(config, name)
            for name in (
                "model_type",
                "architectures",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "sliding_window",
                "layer_types",
            )
            if hasattr(config, name)
        }
    )
    serialized = json.dumps(
        _json_compatible(config_payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    first_parameter = next(model.parameters(), None)
    return {
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_commit": getattr(config, "_commit_hash", None),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "config_sha256": hashlib.sha256(serialized).hexdigest(),
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "maximum_context": getattr(config, "max_position_embeddings", None),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": (
            None if first_parameter is None else str(first_parameter.device)
        ),
        "dtype": (
            None if first_parameter is None else str(first_parameter.dtype)
        ),
        "weights_in_artifact": False,
    }


def _analysis_report(
    *,
    collection: StreamingFisherCollection,
    model: nn.Module,
    model_id: str,
    revision: str | None,
    layer_index: int,
    max_length: int,
    prompts: Sequence[str],
    output: Path,
) -> dict[str, object]:
    prompt_bytes = json.dumps(
        list(prompts),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": {
            "scope": "one_layer_opt_in_calibration_rung",
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "compilation_claim": False,
            "quality_validation_claim": False,
        },
        "model": _model_provenance(
            model,
            model_id=model_id,
            requested_revision=revision,
        ),
        "protocol": {
            "layer_index": layer_index,
            "activation_sites": tuple(collection.bases),
            "prompt_count": len(prompts),
            "normalized_prompts_sha256": hashlib.sha256(
                prompt_bytes
            ).hexdigest(),
            "maximum_tokenized_length": max_length,
            "gradient_batching": "one_sequence_at_a_time",
            "tokenization_residency": "one_minibatch_on_analysis_device",
            "score": "summed_hard_target_next_token_nll",
            "score_compute_dtype": (
                "float32_for_float16_or_bfloat16_logits"
            ),
            "scope": "width_pooled",
            "normalizer": "valid_activation_positions",
            "length_weighting": "longer_sequences_contribute_more_rows",
            "leaf_boundary": next(iter(collection.bases)),
            "cache_policy": "external_to_git_worktree",
        },
        "analysis": collection.metadata(),
        "artifact": {
            "tensor_output": output.name,
            "contains_model_state_dict": False,
            "contains_tokenizer": False,
        },
    }


def load_gemma3_fisher_artifact(
    path: Path | str,
) -> tuple[StreamingFisherCollection, dict[str, object]]:
    """Load and validate an analysis-only artifact written by this module."""

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise TypeError("Gemma 3 Fisher artifact must contain a mapping")
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "model",
        "protocol",
        "collection",
    }
    if set(raw) != required:
        raise ValueError(
            "Gemma 3 Fisher artifact fields do not match format version 1"
        )
    if raw["schema"] != _ARTIFACT_SCHEMA:
        raise ValueError("unsupported Gemma 3 Fisher artifact schema")
    if raw["format_version"] != _ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported Gemma 3 Fisher artifact format")
    if raw["contains_model_weights"] is not False:
        raise ValueError("analysis artifact unexpectedly claims model weights")
    model = raw["model"]
    protocol = raw["protocol"]
    collection_state = raw["collection"]
    if not isinstance(model, Mapping) or not isinstance(protocol, Mapping):
        raise TypeError("artifact model and protocol metadata must be mappings")
    if model.get("weights_in_artifact") is not False:
        raise ValueError("artifact model metadata does not exclude weights")
    if not isinstance(collection_state, Mapping):
        raise TypeError("artifact collection state must be a mapping")
    collection = StreamingFisherCollection.from_state_dict(collection_state)
    metadata = {
        "schema": raw["schema"],
        "format_version": raw["format_version"],
        "contains_model_weights": False,
        "model": dict(model),
        "protocol": dict(protocol),
    }
    return collection, metadata


def resolve_gemma3_huggingface_paths(
    cache_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Validate and report every Hugging Face path used by this rung."""

    package_worktree = find_git_worktree(Path(__file__))
    resolved_cache = external_huggingface_cache_dir(
        cache_dir,
        additional_repository_roots=(
            () if package_worktree is None else (package_worktree,)
        ),
    )
    return huggingface_local_paths(resolved_cache)


def run_gemma3_fisher(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_file: Path | None = None,
    inline_prompts: Sequence[str] = (),
    layer_index: int = 0,
    max_length: int = 128,
    tokenization_batch_size: int = 4,
    rank: int = 32,
    sketch_rows: int | None = None,
    device_name: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Run the opt-in calibration experiment and write analysis-only files."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    output = (
        default_gemma3_output(model_id, layer_index)
        if output is None
        else Path(output)
    )
    if output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    if type(rank) is not int or rank <= 0:
        raise ValueError("rank must be positive")
    if sketch_rows is not None and (
        type(sketch_rows) is not int or sketch_rows <= rank
    ):
        raise ValueError("sketch_rows must be an integer greater than rank")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    device = resolve_torch_device(device_name)
    resolved_cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    prompts = read_prompts(
        prompt_file=prompt_file,
        inline_prompts=inline_prompts,
    )
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=resolved_cache,
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
    if rank > layer.residual_width:
        raise ValueError(
            f"rank cannot exceed residual width {layer.residual_width}"
        )
    batches = make_causal_lm_calibration_batches(
        tokenizer,
        prompts,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    collection = collect_streaming_fisher_modes(
        adapter,
        batches,
        activation_names=(layer.input_site, layer.output_site),
        score_objective=CausalLanguageModelNLL(),
        rank=rank,
        sketch_rows=sketch_rows,
        leaf_activation_name=layer.input_site,
    )
    report = _analysis_report(
        collection=collection,
        model=model,
        model_id=model_id,
        revision=revision,
        layer_index=layer_index,
        max_length=max_length,
        prompts=prompts,
        output=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": _ARTIFACT_SCHEMA,
            "format_version": _ARTIFACT_FORMAT_VERSION,
            "contains_model_weights": False,
            "model": report["model"],
            "protocol": report["protocol"],
            "collection": collection.state_dict(),
        },
        output,
    )
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one Gemma 3 layer's input/output activation Fisher "
            "modes without putting model weights in this repository."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "External Hugging Face cache. The command rejects any cache or "
            "credential path inside the active Git worktree."
        ),
    )
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help=(
            "validate and print every Hugging Face write path, then exit "
            "without importing Transformers or loading a model"
        ),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="UTF-8 file containing one calibration prompt per line.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Inline calibration prompt; may be supplied more than once.",
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--sketch-rows",
        type=int,
        help="Frequent Directions rows; defaults to twice --rank.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "analysis-only .pt path; defaults to an ignored path derived "
            "from --model and --layer-index"
        ),
    )
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
        else default_gemma3_output(
            arguments.model,
            arguments.layer_index,
        )
    )
    report = run_gemma3_fisher(
        model_id=arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_file=arguments.prompts,
        inline_prompts=tuple(arguments.prompt),
        layer_index=arguments.layer_index,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        rank=arguments.rank,
        sketch_rows=arguments.sketch_rows,
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=output,
    )
    analysis = report["analysis"]
    assert isinstance(analysis, Mapping)
    print(
        f"Wrote analysis-only artifact for {analysis['sequences']} sequences "
        f"to {output}"
    )
    print(f"Report: {output.with_suffix('.json')}")
    print("No pretrained model weights were written to either output.")


if __name__ == "__main__":
    main()
