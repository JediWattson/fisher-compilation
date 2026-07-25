"""Fresh split-safe behavioral rank ladder for a frozen Gemma block.

This experiment deliberately does *not* fit an executor.  At each retained
rank it reads the native block delta and computes its per-token least-squares
projection into the selected output-decoder span.  The result is therefore a
target-informed Euclidean reference, not an inference-time implementation and
not a behavioral upper bound.

Only calibration B sees the complete rank ladder.  The smallest rank passing
the aggregate NLL and top-1 gates is locked and evaluated once on validation.
Calibration A and test remain parse/hash-only.  Rank 640 is a mandatory
full-width identity control; a failed identity stops the run before validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter, LayerBlockBoundaryPlan, ModelAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gemma3_ablation_experiment import (
    _FrozenModelTensorGuard,
    _is_sha256,
    _update_payload_digest,
    _validate_model_metadata,
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
    _RuntimeCodec,
    _aggregate_direct_examples,
    _behavior_aggregate,
    _behavior_examples,
    _collect_boundaries,
    _evaluate_direct,
    _evaluate_full_width_delta_roundtrip,
    _finite,
    _identity_passed,
    _materialize_split,
    _runtime_codec,
    load_gemma3_gated_executor_artifact,
)
from .gemma3_stability_experiment import (
    _library_versions,
    _ordered_prompt_hash_digest,
    _tokenizer_provenance,
    _validated_tokenized_stream,
    load_gemma3_prompt_splits,
)
from .gemma3_weighted_jacobian_experiment import (
    _codec_state_sha256,
    load_gemma3_weighted_jacobian_artifact,
)
from .linear_codec import LinearActivationCodec
from .modal_ablation import _causal_lm_batch_scores, _example_ids


DEFAULT_PROMPT_SPLITS = Path(
    "examples/gemma3_projection_ladder_prompts.json"
)
DEFAULT_RANKS = (
    480,
    512,
    544,
    576,
    592,
    608,
    616,
    624,
    632,
    636,
    638,
    639,
    640,
)
DEFAULT_NLL_ATOL = 0.05
DEFAULT_TOP1_MIN = 0.95
DEFAULT_IDENTITY_NLL_ATOL = 1e-5
DEFAULT_MAX_MEANINGFUL_RETAINED_FRACTION = 0.75

_ARTIFACT_SCHEMA = "fisher_graph.gemma3_projection_ladder"
_ARTIFACT_FORMAT_VERSION = 1
_PAYLOAD_DOMAIN = b"fisher_graph.gemma3_projection_ladder_payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.gemma3_projection_ladder_report.v1\0"


def default_gemma3_projection_ladder_output(
    model_id: str = DEFAULT_MODEL_ID,
    start_layer: int = 4,
    end_layer: int = 6,
) -> Path:
    """Return the ignored model/block-specific projection artifact path."""

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
    return (
        Path(".local-runs")
        / (slug or "gemma3-model")
        / f"layers-{start_layer}-{end_layer}-projection-ladder.pt"
    )


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


def _validated_ranks(
    values: Iterable[int],
    *,
    width: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("ranks must be an iterable of integers")
    ranks = tuple(values)
    if (
        not ranks
        or any(type(rank) is not int or not 0 < rank <= width for rank in ranks)
    ):
        raise ValueError("ranks must be positive and at most residual width")
    canonical = tuple(sorted(set(ranks)))
    if width not in canonical:
        raise ValueError("rank schedule must include full residual width")
    if width == 640 and canonical != DEFAULT_RANKS:
        raise ValueError(
            "width-640 projection ladder must use the preregistered ranks"
        )
    return canonical


@dataclass(frozen=True, slots=True)
class ProjectionCandidate:
    retained_rank: int
    residual_width: int

    def __post_init__(self) -> None:
        if (
            type(self.retained_rank) is not int
            or type(self.residual_width) is not int
            or self.residual_width <= 0
            or not 0 < self.retained_rank <= self.residual_width
        ):
            raise ValueError("projection candidate geometry is invalid")

    @property
    def candidate_id(self) -> str:
        return f"rank_{self.retained_rank}.target_ls_projection"

    @property
    def retained_fraction(self) -> float:
        return self.retained_rank / self.residual_width

    @property
    def removed_dimensions(self) -> int:
        return self.residual_width - self.retained_rank

    def metadata(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "retained_rank": self.retained_rank,
            "residual_width": self.residual_width,
            "retained_fraction": self.retained_fraction,
            "removed_dimensions": self.removed_dimensions,
            "projection": (
                "target_informed_per_token_least_squares_in_output_"
                "decoder_span"
            ),
        }


def _prompt_hash_set(metadata: Mapping[str, object]) -> set[str]:
    per_prompt = metadata.get("per_prompt_sha256")
    if not isinstance(per_prompt, Mapping):
        raise ValueError("prompt provenance lacks per-prompt hashes")
    values = {
        digest
        for split_hashes in per_prompt.values()
        if isinstance(split_hashes, (list, tuple))
        for digest in split_hashes
    }
    if not values or any(not _is_sha256(value) for value in values):
        raise ValueError("prompt provenance contains invalid hashes")
    return values


def _assert_prompt_disjointness(
    *,
    fresh: Mapping[str, object],
    weighted_protocol: Mapping[str, object],
    gated_protocol: Mapping[str, object],
) -> dict[str, object]:
    fresh_hashes = _prompt_hash_set(fresh)
    weighted_prompts = weighted_protocol.get("prompt_splits")
    gated_prompts = gated_protocol.get("prompt_splits")
    if not isinstance(weighted_prompts, Mapping) or not isinstance(
        gated_prompts,
        Mapping,
    ):
        raise ValueError("predecessor prompt provenance is missing")
    weighted_hashes = _prompt_hash_set(weighted_prompts)
    gated_hashes = _prompt_hash_set(gated_prompts)
    if fresh_hashes & weighted_hashes:
        raise ValueError("projection prompts overlap weighted source prompts")
    if fresh_hashes & gated_hashes:
        raise ValueError("projection prompts overlap gated executor prompts")
    return {
        "fresh_prompt_sha256": tuple(sorted(fresh_hashes)),
        "weighted_prompt_sha256": tuple(sorted(weighted_hashes)),
        "gated_prompt_sha256": tuple(sorted(gated_hashes)),
        "fresh_count": len(fresh_hashes),
        "weighted_count": len(weighted_hashes),
        "gated_count": len(gated_hashes),
        "weighted_overlap_count": 0,
        "gated_overlap_count": 0,
        "verified_before_model_load_or_tokenization": True,
    }


def _evaluate_projection_behavior(
    adapter: ModelAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    plan: LayerBlockBoundaryPlan,
    rank_codecs: Mapping[str, _RuntimeCodec],
    full_width_codec: LinearActivationCodec | None = None,
) -> dict[str, dict[str, object]]:
    """Evaluate target-informed interventions with one native baseline."""

    objective = CausalLanguageModelNLL()
    condition_names = list(rank_codecs)
    if full_width_codec is not None:
        condition_names.append("full_width_codec_delta_roundtrip")
    examples: dict[str, list[dict[str, object]]] = {
        name: [] for name in condition_names
    }
    sequence_offset = 0
    module = adapter.module
    if module.training or any(
        parameter.requires_grad for parameter in module.parameters()
    ):
        raise ValueError("projection behavior requires a frozen eval model")
    codec_device = (
        next(iter(rank_codecs.values())).output_decoder.device
        if rank_codecs
        else torch.device("cpu")
    )
    full_encoder = (
        None
        if full_width_codec is None
        else full_width_codec.encoder.to(
            device=codec_device,
            dtype=torch.float64,
        )
    )
    full_decoder = (
        None
        if full_width_codec is None
        else full_width_codec.decoder.to(
            device=codec_device,
            dtype=torch.float64,
        )
    )
    with torch.inference_mode():
        for batch in batches:
            ids = _example_ids(batch, sequence_offset=sequence_offset)
            sequence_offset += batch.batch_size
            native = adapter.forward(batch.model_inputs)
            baseline = _causal_lm_batch_scores(
                native.logits,
                batch,
                objective=objective,
            )
            for condition_name in condition_names:
                captured: dict[str, Tensor] = {}

                def capture_input(value: Tensor) -> Tensor:
                    captured["input"] = value
                    return value

                def replace_output(
                    value: Tensor,
                    *,
                    name: str = condition_name,
                ) -> Tensor:
                    source = captured.get("input")
                    if source is None:
                        raise RuntimeError(
                            "block output ran before input capture"
                        )
                    source64 = source.to(torch.float64)
                    delta = value.to(torch.float64) - source64
                    if name == "full_width_codec_delta_roundtrip":
                        assert full_encoder is not None
                        assert full_decoder is not None
                        prediction = (
                            source64
                            + (delta @ full_encoder) @ full_decoder.T
                        )
                    else:
                        codec = rank_codecs[name]
                        prediction = source64 + (
                            delta @ codec.output_least_squares_encoder
                        ) @ codec.output_decoder.to(torch.float64).T
                    return torch.where(
                        batch.valid_positions.unsqueeze(-1),
                        prediction.to(dtype=value.dtype),
                        value,
                    )

                projected = adapter.forward(
                    batch.model_inputs,
                    interventions={
                        plan.activation_sites[0]: capture_input,
                        plan.activation_sites[-1]: replace_output,
                    },
                )
                scores = _causal_lm_batch_scores(
                    projected.logits,
                    batch,
                    objective=objective,
                )
                examples[condition_name].extend(
                    _behavior_examples(
                        batch=batch,
                        example_ids=ids,
                        baseline=baseline,
                        predicted=scores,
                    )
                )
    return {
        name: _behavior_aggregate(rows)
        for name, rows in examples.items()
    }


def _behavior_gate(
    behavior: Mapping[str, object],
    *,
    nll_atol: float,
    top1_min: float,
) -> dict[str, bool]:
    return {
        "absolute_delta_nll": (
            abs(float(behavior["delta_nll_per_token"])) <= nll_atol
        ),
        "top1_agreement": (
            float(behavior["top1_agreement_to_baseline"]) >= top1_min
        ),
    }


def _candidate_ledger(
    candidates: Sequence[ProjectionCandidate],
    *,
    direct: Mapping[str, Mapping[str, object]],
    behavior: Mapping[str, Mapping[str, object]],
    nll_atol: float,
    top1_min: float,
) -> list[dict[str, object]]:
    rows = []
    for candidate in candidates:
        gates = _behavior_gate(
            behavior[candidate.candidate_id],
            nll_atol=nll_atol,
            top1_min=top1_min,
        )
        rows.append(
            {
                "candidate": candidate.metadata(),
                "direct_diagnostic": copy.deepcopy(
                    dict(direct[candidate.candidate_id])
                ),
                "behavior": copy.deepcopy(
                    dict(behavior[candidate.candidate_id])
                ),
                "behavior_gates": gates,
                "behavior_fidelity_passed": all(gates.values()),
                "direct_metrics_influence_lock": False,
            }
        )
    return rows


def _lock_candidate(
    candidates: Sequence[ProjectionCandidate],
    ledger: Sequence[Mapping[str, object]],
) -> tuple[ProjectionCandidate, dict[str, object]]:
    if len(candidates) != len(ledger) or not candidates:
        raise ValueError("candidate schedule and ledger must align")
    width = candidates[0].residual_width
    passing_reduced = [
        candidate
        for candidate, row in zip(candidates, ledger, strict=True)
        if candidate.retained_rank < width
        and row["behavior_fidelity_passed"] is True
    ]
    if passing_reduced:
        locked = passing_reduced[0]
        selection_failed = False
        reason = "smallest_reduced_rank_passing_behavioral_gates"
    else:
        locked = next(
            candidate
            for candidate in candidates
            if candidate.retained_rank == width
        )
        selection_failed = True
        reason = "no_reduced_rank_passed_full_width_identity_fallback"
    return locked, {
        "ordering": "retained_rank_ascending",
        "direct_metrics_influence_lock": False,
        "fidelity_lock_rank": locked.retained_rank,
        "locked_candidate_id": locked.candidate_id,
        "selection_failed": selection_failed,
        "reduced_candidate_found": not selection_failed,
        "reason": reason,
        "calibration_b_only": True,
        "ledger": [copy.deepcopy(dict(row)) for row in ledger],
    }


def _assert_nested_direct_error(
    candidates: Sequence[ProjectionCandidate],
    direct: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    values = [
        float(direct[candidate.candidate_id]["mse"])
        for candidate in candidates
    ]
    for previous, current in zip(values, values[1:]):
        tolerance = max(1e-12, abs(previous) * 1e-8)
        if current > previous + tolerance:
            raise RuntimeError(
                "nested projection MSE increased with retained rank"
            )
    return {
        "passed": True,
        "metric": "raw_block_output_mse",
        "required_relation": "nonincreasing_with_retained_rank",
        "values": tuple(values),
        "numeric_tolerance": "max_1e-12_or_1e-8_relative",
    }


def _full_width_controls_passed(
    *,
    rank_direct: Mapping[str, object],
    rank_behavior: Mapping[str, object],
    runtime_direct: Mapping[str, object],
    mathematical_direct: Mapping[str, object],
    codec_behavior: Mapping[str, object],
    identity_nll_atol: float,
) -> bool:
    direct_controls = (
        rank_direct,
        runtime_direct,
        mathematical_direct,
    )
    return (
        all(
            float(row["block_delta_nrmse"]) <= 1e-5
            and float(row["block_delta_cosine"]) >= 1.0 - 1e-7
            for row in direct_controls
        )
        and _identity_passed(
            rank_behavior,
            nll_atol=identity_nll_atol,
        )
        and _identity_passed(
            codec_behavior,
            nll_atol=identity_nll_atol,
        )
    )


def _build_report(
    payload: Mapping[str, object],
    *,
    output: Path,
    scientific_digest: str,
) -> dict[str, object]:
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": copy.deepcopy(
            dict(payload["scientific_status"])  # type: ignore[arg-type]
        ),
        "model": copy.deepcopy(
            dict(payload["model"])  # type: ignore[arg-type]
        ),
        "predecessors": copy.deepcopy(
            dict(payload["predecessors"])  # type: ignore[arg-type]
        ),
        "protocol": copy.deepcopy(
            dict(payload["protocol"])  # type: ignore[arg-type]
        ),
        "analysis": {
            "selection": copy.deepcopy(
                dict(payload["selection"])  # type: ignore[arg-type]
            ),
            "validation": copy.deepcopy(
                dict(payload["validation"])  # type: ignore[arg-type]
            ),
        },
        "artifact": {
            "tensor_output": output.name,
            "contains_model_state_dict": False,
            "contains_codec_state": True,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def run_gemma3_projection_ladder(
    *,
    weighted_artifact_path: Path | str,
    gated_artifact_path: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
    prompt_splits_path: Path | str = DEFAULT_PROMPT_SPLITS,
    max_length: int = 128,
    tokenization_batch_size: int = 4,
    ranks: Iterable[int] = DEFAULT_RANKS,
    selection_nll_atol: float = DEFAULT_NLL_ATOL,
    selection_top1_min: float = DEFAULT_TOP1_MIN,
    identity_nll_atol: float = DEFAULT_IDENTITY_NLL_ATOL,
    max_meaningful_retained_fraction: float = (
        DEFAULT_MAX_MEANINGFUL_RETAINED_FRACTION
    ),
    device_name: str = "cpu",
    dtype: str = "auto",
    local_files_only: bool = False,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Locate and validate the first behaviorally faithful output span."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    nll_atol = _finite(
        selection_nll_atol,
        label="selection_nll_atol",
        minimum=0.0,
    )
    top1_min = _finite(
        selection_top1_min,
        label="selection_top1_min",
        minimum=0.0,
        maximum=1.0,
    )
    identity_tolerance = _finite(
        identity_nll_atol,
        label="identity_nll_atol",
        minimum=0.0,
    )
    meaningful_fraction = _finite(
        max_meaningful_retained_fraction,
        label="max_meaningful_retained_fraction",
        minimum=0.0,
        maximum=1.0,
    )

    # Both predecessors are strict-loaded before model access.  The gated
    # artifact binds this follow-up to the exact negative result that motivated
    # the ladder; the weighted artifact remains the authority for codec state.
    weighted = load_gemma3_weighted_jacobian_artifact(
        weighted_artifact_path
    )
    gated = load_gemma3_gated_executor_artifact(gated_artifact_path)
    weighted_metadata = weighted["metadata"]
    gated_metadata = gated["metadata"]
    weighted_selection = weighted["selection"]
    weighted_codecs = weighted["codecs"]
    assert isinstance(weighted_metadata, Mapping)
    assert isinstance(gated_metadata, Mapping)
    assert isinstance(weighted_selection, Mapping)
    assert isinstance(weighted_codecs, Mapping)
    weighted_model = weighted_metadata["model"]
    weighted_protocol = weighted_metadata["protocol"]
    gated_model = gated["model"]
    gated_protocol = gated_metadata["protocol"]
    gated_source = gated_metadata["source_analysis"]
    assert isinstance(weighted_model, Mapping)
    assert isinstance(weighted_protocol, Mapping)
    assert isinstance(gated_model, Mapping)
    assert isinstance(gated_protocol, Mapping)
    assert isinstance(gated_source, Mapping)
    if (
        gated_source.get("scientific_payload_sha256")
        != weighted_metadata["scientific_payload_sha256"]
    ):
        raise ValueError("gated artifact does not bind the weighted source")
    for field in ("schema", "format_version", "report_sha256"):
        if gated_source.get(field) != weighted_metadata[field]:
            raise ValueError(
                f"gated and weighted predecessor {field} bindings disagree"
            )
    if (
        gated_source.get("locked_candidate")
        != weighted_selection.get("locked_candidate")
        or gated_source.get("codec_variant_id")
        != weighted_selection["locked_candidate"]["variant_id"]  # type: ignore[index]
    ):
        raise ValueError("gated predecessor locked codec binding disagrees")
    if gated["validation"]["overall_viable"] is not False:  # type: ignore[index]
        raise ValueError(
            "projection ladder requires the nonviable gated predecessor"
        )
    if any(
        weighted_model.get(field) != gated_model.get(field)
        for field in ("model_id", "config_sha256", "resolved_commit")
    ):
        raise ValueError("predecessor model bindings disagree")
    for field in (
        "start_layer",
        "end_layer_inclusive",
        "canonical_boundaries",
        "residual_width",
    ):
        if weighted_protocol.get(field) != gated_protocol.get(field):
            raise ValueError(
                f"predecessor block {field} bindings disagree"
            )
    if weighted_model.get("model_id") != model_id:
        raise ValueError("requested model_id does not match predecessors")
    if revision is not None and revision not in {
        weighted_model.get("requested_revision"),
        weighted_model.get("resolved_commit"),
    }:
        raise ValueError("explicit revision does not match predecessors")
    locked_source = weighted_selection.get("locked_candidate")
    if not isinstance(locked_source, Mapping):
        raise ValueError("weighted source lacks a locked codec")
    variant_id = locked_source.get("variant_id")
    if not isinstance(variant_id, str) or variant_id not in weighted_codecs:
        raise ValueError("weighted source codec variant is invalid")

    prompts = load_gemma3_prompt_splits(prompt_splits_path)
    prompt_metadata = prompts.metadata()
    disjointness = _assert_prompt_disjointness(
        fresh=prompt_metadata,
        weighted_protocol=weighted_protocol,
        gated_protocol=gated_protocol,
    )
    width = weighted_protocol.get("residual_width")
    start_layer = weighted_protocol.get("start_layer")
    end_layer = weighted_protocol.get("end_layer_inclusive")
    boundaries = weighted_protocol.get("canonical_boundaries")
    if (
        type(width) is not int
        or width <= 0
        or type(start_layer) is not int
        or type(end_layer) is not int
        or not isinstance(boundaries, tuple)
        or len(boundaries) < 2
    ):
        raise ValueError("weighted source block geometry is invalid")
    by_site = weighted_codecs[variant_id]
    if not isinstance(by_site, Mapping):
        raise ValueError("weighted codec site mapping is invalid")
    output_codec = by_site[boundaries[-1]]
    if not isinstance(output_codec, LinearActivationCodec):
        raise ValueError("weighted output codec is invalid")
    if (
        _codec_state_sha256(output_codec)
        != gated_metadata["source_analysis"]["output_codec_sha256"]  # type: ignore[index]
        or _codec_state_sha256(output_codec)
        != _codec_state_sha256(gated["output_codec"])  # type: ignore[arg-type]
    ):
        raise ValueError("gated and weighted output codecs disagree")
    # _runtime_codec also takes the input codec, but only its output-side
    # least-squares fields are used in this projection-only experiment.
    input_codec = by_site[boundaries[0]]
    if not isinstance(input_codec, LinearActivationCodec):
        raise ValueError("weighted input codec is invalid")

    resolved_ranks = _validated_ranks(ranks, width=width)
    candidates = tuple(
        ProjectionCandidate(rank, width) for rank in resolved_ranks
    )
    resolved_output = (
        default_gemma3_projection_ladder_output(
            model_id,
            start_layer,
            end_layer,
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")

    device = resolve_torch_device(device_name)
    if device.type == "mps":
        raise ValueError(
            "projection ladder requires CPU or CUDA because its "
            "target-informed least-squares controls use float64"
        )
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    requested_revision = (
        revision
        if revision is not None
        else (
            weighted_model.get("resolved_commit")
            or weighted_model.get("requested_revision")
        )
    )
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=requested_revision,
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
    if (
        plan.activation_sites != boundaries
        or plan.widths != (width,) * len(boundaries)
    ):
        raise ValueError("live adapter block does not match predecessors")
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=requested_revision,
    )
    for field in ("model_id", "config_sha256", "hidden_size"):
        if weighted_model.get(field) != model_metadata.get(field):
            raise ValueError(f"live model {field} does not match source")
    if (
        weighted_model.get("resolved_commit") is not None
        and model_metadata.get("resolved_commit") is not None
        and weighted_model["resolved_commit"]
        != model_metadata["resolved_commit"]
    ):
        raise ValueError("live model commit does not match source")

    runtime_codecs = {
        rank: _runtime_codec(
            input_codec,
            output_codec,
            rank=rank,
            device=device,
        )
        for rank in resolved_ranks
    }

    # Calibration A remains entirely unmaterialized.  Calibration B alone sees
    # the rank curve.
    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        prompts.calibration_b,
        split_name="calibration_b",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_boundaries = _collect_boundaries(
        adapter,
        selection_batches,
        plan=plan,
    )
    condition_codecs = {
        candidate.candidate_id: runtime_codecs[candidate.retained_rank]
        for candidate in candidates
    }
    selection_direct = {
        candidate.candidate_id: _evaluate_direct(
            None,
            runtime_codecs[candidate.retained_rank],
            selection_boundaries,
            oracle=True,
        )
        for candidate in candidates
    }
    selection_behavior_all = _evaluate_projection_behavior(
        adapter,
        selection_batches,
        plan=plan,
        rank_codecs=condition_codecs,
        full_width_codec=output_codec,
    )
    selection_behavior = {
        candidate.candidate_id: selection_behavior_all[
            candidate.candidate_id
        ]
        for candidate in candidates
    }
    nested_control = _assert_nested_direct_error(
        candidates,
        selection_direct,
    )
    full_candidate = candidates[-1]
    if full_candidate.retained_rank != width:
        raise RuntimeError("canonical schedule does not end at full width")
    full_runtime_direct = _evaluate_full_width_delta_roundtrip(
        output_codec,
        selection_boundaries,
        device=device,
        dtype=torch.float32,
    )
    full_mathematical_direct = _evaluate_full_width_delta_roundtrip(
        output_codec,
        selection_boundaries,
        device=device,
        dtype=torch.float64,
    )
    full_width_passed = _full_width_controls_passed(
        rank_direct=selection_direct[full_candidate.candidate_id],
        rank_behavior=selection_behavior[full_candidate.candidate_id],
        runtime_direct=full_runtime_direct,
        mathematical_direct=full_mathematical_direct,
        codec_behavior=selection_behavior_all[
            "full_width_codec_delta_roundtrip"
        ],
        identity_nll_atol=identity_tolerance,
    )
    if not full_width_passed:
        raise RuntimeError(
            "calibration-B full-width projection identity failed"
        )
    ledger = _candidate_ledger(
        candidates,
        direct=selection_direct,
        behavior=selection_behavior,
        nll_atol=nll_atol,
        top1_min=top1_min,
    )
    locked, lock = _lock_candidate(candidates, ledger)
    locked_b_row = next(
        row
        for row in ledger
        if row["candidate"]["candidate_id"] == locked.candidate_id  # type: ignore[index]
    )
    b_behavior_passed = (
        locked_b_row["behavior_fidelity_passed"] is True
    )
    guard.assert_unchanged()

    # Validation sees exactly one target-informed locked-rank intervention per
    # batch.  The full ladder and full-codec behavioral control are not replayed.
    validation_batches, validation_stream = _materialize_split(
        tokenizer,
        prompts.validation,
        split_name="validation",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    validation_boundaries = _collect_boundaries(
        adapter,
        validation_batches,
        plan=plan,
    )
    validation_direct = _evaluate_direct(
        None,
        runtime_codecs[locked.retained_rank],
        validation_boundaries,
        oracle=True,
    )
    validation_behavior = _evaluate_projection_behavior(
        adapter,
        validation_batches,
        plan=plan,
        rank_codecs={
            locked.candidate_id: runtime_codecs[locked.retained_rank]
        },
    )[locked.candidate_id]
    validation_gates = _behavior_gate(
        validation_behavior,
        nll_atol=nll_atol,
        top1_min=top1_min,
    )
    validation_passed = all(validation_gates.values())
    validation_full_runtime_direct = (
        _evaluate_full_width_delta_roundtrip(
            output_codec,
            validation_boundaries,
            device=device,
            dtype=torch.float32,
        )
    )
    validation_full_mathematical_direct = (
        _evaluate_full_width_delta_roundtrip(
            output_codec,
            validation_boundaries,
            device=device,
            dtype=torch.float64,
        )
    )
    reduced_rank = locked.retained_rank < width
    fidelity_viable = (
        b_behavior_passed and validation_passed and reduced_rank
    )
    meaningful_rank_compression = (
        fidelity_viable
        and locked.retained_fraction <= meaningful_fraction
    )
    guard.assert_unchanged()

    predecessors = {
        "weighted": {
            "schema": weighted_metadata["schema"],
            "format_version": weighted_metadata["format_version"],
            "scientific_payload_sha256": weighted_metadata[
                "scientific_payload_sha256"
            ],
            "report_sha256": weighted_metadata["report_sha256"],
            "locked_candidate": copy.deepcopy(dict(locked_source)),
            "codec_variant_id": variant_id,
            "output_codec_sha256": _codec_state_sha256(output_codec),
            "model_binding": {
                field: weighted_model.get(field)
                for field in (
                    "model_id",
                    "config_sha256",
                    "resolved_commit",
                    "hidden_size",
                    "num_hidden_layers",
                )
            },
            "block_geometry": {
                "start_layer": start_layer,
                "end_layer_inclusive": end_layer,
                "layer_ids": plan.layer_ids,
                "canonical_boundaries": plan.activation_sites,
                "residual_width": width,
            },
        },
        "gated": {
            "scientific_payload_sha256": gated_metadata[
                "scientific_payload_sha256"
            ],
            "report_sha256": gated_metadata["report_sha256"],
            "overall_viable": gated["validation"]["overall_viable"],  # type: ignore[index]
            "output_codec_sha256": _codec_state_sha256(
                gated["output_codec"]  # type: ignore[arg-type]
            ),
            "weighted_source_scientific_payload_sha256": gated_source[
                "scientific_payload_sha256"
            ],
            "weighted_source_report_sha256": gated_source[
                "report_sha256"
            ],
        },
        "prompt_disjointness": disjointness,
    }
    protocol = {
        "start_layer": start_layer,
        "end_layer_inclusive": end_layer,
        "layer_ids": plan.layer_ids,
        "canonical_boundaries": plan.activation_sites,
        "residual_width": width,
        "candidate_schedule": tuple(
            candidate.metadata() for candidate in candidates
        ),
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "projection": (
            "target_informed_per_token_least_squares_in_output_decoder_"
            "span"
        ),
        "inference_executor": False,
        "behavioral_upper_bound_claim": False,
        "direct_metrics_influence_lock": False,
        "selection_policy": (
            "calibration_b_smallest_reduced_rank_passing_aggregate_"
            "behavior_else_full_width_identity"
        ),
        "validation_policy": (
            "one_locked_rank_intervention_per_batch_no_rank_curve"
        ),
        "calibration_a_policy": "parse_validate_hash_only",
        "test_policy": "parse_validate_hash_only",
        "selection_nll_atol": nll_atol,
        "selection_top1_min": top1_min,
        "identity_nll_atol": identity_tolerance,
        "max_meaningful_retained_fraction": meaningful_fraction,
        "compression_claim": False,
        "parameter_mac_speed_claim": False,
        "model_state_guard": guard.metadata(),
        "library_versions": _library_versions(),
        "tokenizer": _tokenizer_provenance(tokenizer),
        "tokenized_splits": {
            "calibration_b": selection_stream,
            "validation": validation_stream,
        },
        "prompt_splits": prompt_metadata,
    }
    payload = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "scientific_status": {
            "scope": "target_informed_projection_behavioral_rank_ladder",
            "calibration_a_evaluated": False,
            "calibration_b_rank_curve_evaluated": True,
            "validation_locked_before_evaluation": True,
            "locked_validation_interventions_per_batch": 1,
            "test_evaluated": False,
            "model_weights_changed": False,
            "model_weights_in_artifact": False,
            "prompt_text_in_artifact": False,
            "inference_executor": False,
            "behavioral_upper_bound_claim": False,
            "compression_claim": False,
            "parameter_mac_speed_claim": False,
            "selection_failed": lock["selection_failed"],
            "reduced_fidelity_candidate_found": (
                lock["reduced_candidate_found"]
            ),
            "locked_validation_fidelity_passed": validation_passed,
            "fidelity_viable_reduced_rank": fidelity_viable,
            "meaningful_rank_compression": meaningful_rank_compression,
        },
        "model": model_metadata,
        "predecessors": predecessors,
        "protocol": protocol,
        "output_codec": output_codec.state_dict(),
        "selection": {
            "candidate_direct_diagnostics": selection_direct,
            "candidate_behavior": selection_behavior,
            "nested_projection_error_control": nested_control,
            "full_width_identity": {
                "rank_direct": selection_direct[
                    full_candidate.candidate_id
                ],
                "rank_behavior": selection_behavior[
                    full_candidate.candidate_id
                ],
                "runtime_float32_codec_direct": full_runtime_direct,
                "mathematical_float64_codec_direct": (
                    full_mathematical_direct
                ),
                "codec_behavior": selection_behavior_all[
                    "full_width_codec_delta_roundtrip"
                ],
                "passed": True,
            },
            "lock": lock,
            "tokenized_stream": selection_stream,
        },
        "validation": {
            "locked_candidate": locked.metadata(),
            "direct_diagnostic": validation_direct,
            "behavior": validation_behavior,
            "behavior_gates": validation_gates,
            "behavior_fidelity_passed": validation_passed,
            "full_width_codec_direct_control": {
                "runtime_float32": validation_full_runtime_direct,
                "mathematical_float64": (
                    validation_full_mathematical_direct
                ),
            },
            "fidelity_viable_reduced_rank": fidelity_viable,
            "meaningful_rank_compression": meaningful_rank_compression,
            "tokenized_stream": validation_stream,
        },
    }
    digest = _scientific_payload_sha256(payload)
    report = _build_report(
        payload,
        output=resolved_output,
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
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _recompute_behavior_aggregate(
    value: object,
    *,
    stream: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or not isinstance(
        value.get("examples"),
        list,
    ):
        raise ValueError("behavior aggregate examples are invalid")
    examples = value["examples"]
    expected_examples = stream["examples"]
    if not isinstance(expected_examples, list):
        raise ValueError("tokenized stream examples are invalid")
    if tuple(
        (row.get("example_id"), row.get("supervised_tokens"))
        for row in examples
        if isinstance(row, Mapping)
    ) != tuple(
        (row.get("example_id"), row.get("supervised_positions"))
        for row in expected_examples
        if isinstance(row, Mapping)
    ):
        raise ValueError("behavior examples do not bind tokenized stream")
    recomputed = _behavior_aggregate(examples)  # type: ignore[arg-type]
    if recomputed != value:
        raise ValueError("behavior aggregate does not recompute from examples")
    return recomputed


def _recompute_direct_aggregate(
    value: object,
    *,
    stream: Mapping[str, object],
    width: int,
    kind: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or not isinstance(
        value.get("examples"),
        list,
    ):
        raise ValueError("direct aggregate examples are invalid")
    examples = value["examples"]
    expected_examples = stream["examples"]
    if not isinstance(expected_examples, list):
        raise ValueError("tokenized stream examples are invalid")
    if tuple(
        (row.get("example_id"), row.get("valid_tokens"))
        for row in examples
        if isinstance(row, Mapping)
    ) != tuple(
        (row.get("example_id"), row.get("valid_tokens"))
        for row in expected_examples
        if isinstance(row, Mapping)
    ):
        raise ValueError("direct examples do not bind tokenized stream")
    recomputed = _aggregate_direct_examples(
        examples,  # type: ignore[arg-type]
        width=width,
    )
    if kind == "target_ls":
        recomputed["path_energy"] = {
            "same_position": 0.0,
            "positive_lag": 0.0,
            "positive_lag_fraction_of_path_energy": 0.0,
        }
        recomputed["router"] = {
            "positive_lag_edges": 0,
            "mean_entropy": 0.0,
            "mean_max_probability": 0.0,
            "collapsed_fraction_not_materialized": True,
        }
    elif kind in {"codec_float32", "codec_float64"}:
        recomputed["control"] = (
            "locked_family_full_width_delta_roundtrip"
        )
        recomputed["compute_dtype"] = (
            "float32" if kind == "codec_float32" else "float64"
        )
    else:
        raise ValueError("unknown direct aggregate kind")
    if recomputed != value:
        raise ValueError("direct aggregate does not recompute from examples")
    return recomputed


def load_gemma3_projection_ladder_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Strictly load and semantically recompute a projection-ladder artifact."""

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
        "scientific_status",
        "model",
        "predecessors",
        "protocol",
        "output_codec",
        "selection",
        "validation",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("projection-ladder artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
        or raw["contains_model_weights"] is not False
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
    ):
        raise ValueError("unsupported or unsafe projection-ladder artifact")
    if (
        not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("projection-ladder digest fields are invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    digest = _scientific_payload_sha256(payload)
    if digest != raw["scientific_payload_sha256"]:
        raise ValueError("projection-ladder scientific digest mismatch")
    model = _validate_model_metadata(raw["model"])
    predecessors = raw["predecessors"]
    protocol = raw["protocol"]
    codec_state = raw["output_codec"]
    selection = raw["selection"]
    validation = raw["validation"]
    status = raw["scientific_status"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            predecessors,
            protocol,
            codec_state,
            selection,
            validation,
            status,
        )
    ):
        raise ValueError("projection-ladder payload mappings are invalid")

    predecessor_fields = {
        "weighted",
        "gated",
        "prompt_disjointness",
    }
    weighted_binding = predecessors.get("weighted")
    gated_binding = predecessors.get("gated")
    disjointness = predecessors.get("prompt_disjointness")
    if (
        set(predecessors) != predecessor_fields
        or not isinstance(weighted_binding, Mapping)
        or not isinstance(gated_binding, Mapping)
        or not isinstance(disjointness, Mapping)
    ):
        raise ValueError("projection predecessor bindings are invalid")
    weighted_fields = {
        "schema",
        "format_version",
        "scientific_payload_sha256",
        "report_sha256",
        "locked_candidate",
        "codec_variant_id",
        "output_codec_sha256",
        "model_binding",
        "block_geometry",
    }
    gated_fields = {
        "scientific_payload_sha256",
        "report_sha256",
        "overall_viable",
        "output_codec_sha256",
        "weighted_source_scientific_payload_sha256",
        "weighted_source_report_sha256",
    }
    if (
        set(weighted_binding) != weighted_fields
        or set(gated_binding) != gated_fields
        or weighted_binding["schema"]
        != "fisher_graph.gemma3_weighted_jacobian"
        or gated_binding["overall_viable"] is not False
        or any(
            not _is_sha256(value)
            for value in (
                weighted_binding["scientific_payload_sha256"],
                weighted_binding["report_sha256"],
                weighted_binding["output_codec_sha256"],
                gated_binding["scientific_payload_sha256"],
                gated_binding["report_sha256"],
                gated_binding["output_codec_sha256"],
                gated_binding[
                    "weighted_source_scientific_payload_sha256"
                ],
                gated_binding["weighted_source_report_sha256"],
            )
        )
        or gated_binding[
            "weighted_source_scientific_payload_sha256"
        ]
        != weighted_binding["scientific_payload_sha256"]
        or gated_binding["weighted_source_report_sha256"]
        != weighted_binding["report_sha256"]
    ):
        raise ValueError("projection predecessor metadata is invalid")
    output_codec = LinearActivationCodec.from_state_dict(codec_state)
    codec_sha = _codec_state_sha256(output_codec)
    predecessor_locked = weighted_binding["locked_candidate"]
    if (
        not isinstance(predecessor_locked, Mapping)
        or predecessor_locked.get("variant_id")
        != weighted_binding["codec_variant_id"]
        or codec_sha != weighted_binding["output_codec_sha256"]
        or codec_sha != gated_binding["output_codec_sha256"]
    ):
        raise ValueError("projection output codec binding is invalid")

    protocol_fields = {
        "start_layer",
        "end_layer_inclusive",
        "layer_ids",
        "canonical_boundaries",
        "residual_width",
        "candidate_schedule",
        "maximum_tokenized_length",
        "tokenization_batch_size",
        "projection",
        "inference_executor",
        "behavioral_upper_bound_claim",
        "direct_metrics_influence_lock",
        "selection_policy",
        "validation_policy",
        "calibration_a_policy",
        "test_policy",
        "selection_nll_atol",
        "selection_top1_min",
        "identity_nll_atol",
        "max_meaningful_retained_fraction",
        "compression_claim",
        "parameter_mac_speed_claim",
        "model_state_guard",
        "library_versions",
        "tokenizer",
        "tokenized_splits",
        "prompt_splits",
    }
    if set(protocol) != protocol_fields:
        raise ValueError("projection-ladder protocol fields are invalid")
    width = protocol["residual_width"]
    start = protocol["start_layer"]
    end = protocol["end_layer_inclusive"]
    boundaries = protocol["canonical_boundaries"]
    schedule = protocol["candidate_schedule"]
    if (
        type(width) is not int
        or width <= 0
        or type(start) is not int
        or type(end) is not int
        or start < 0
        or end < start
        or not isinstance(boundaries, tuple)
        or len(boundaries) != end - start + 2
        or not isinstance(schedule, tuple)
        or not schedule
        or output_codec.width != width
        or output_codec.activation_name != boundaries[-1]
    ):
        raise ValueError("projection-ladder geometry is invalid")
    model_binding = weighted_binding["model_binding"]
    block_geometry = weighted_binding["block_geometry"]
    if (
        not isinstance(model_binding, Mapping)
        or set(model_binding)
        != {
            "model_id",
            "config_sha256",
            "resolved_commit",
            "hidden_size",
            "num_hidden_layers",
        }
        or any(
            model_binding[field] != model.get(field)
            for field in model_binding
        )
        or not isinstance(block_geometry, Mapping)
        or set(block_geometry)
        != {
            "start_layer",
            "end_layer_inclusive",
            "layer_ids",
            "canonical_boundaries",
            "residual_width",
        }
        or block_geometry
        != {
            "start_layer": start,
            "end_layer_inclusive": end,
            "layer_ids": protocol["layer_ids"],
            "canonical_boundaries": boundaries,
            "residual_width": width,
        }
    ):
        raise ValueError("projection predecessor model/block binding is invalid")
    ranks = tuple(
        row.get("retained_rank") if isinstance(row, Mapping) else None
        for row in schedule
    )
    canonical_ranks = _validated_ranks(ranks, width=width)  # type: ignore[arg-type]
    candidates = tuple(
        ProjectionCandidate(rank, width) for rank in canonical_ranks
    )
    if schedule != tuple(candidate.metadata() for candidate in candidates):
        raise ValueError("projection candidate schedule is noncanonical")
    if (
        protocol["projection"]
        != "target_informed_per_token_least_squares_in_output_decoder_span"
        or protocol["inference_executor"] is not False
        or protocol["behavioral_upper_bound_claim"] is not False
        or protocol["direct_metrics_influence_lock"] is not False
        or protocol["selection_policy"]
        != "calibration_b_smallest_reduced_rank_passing_aggregate_"
        "behavior_else_full_width_identity"
        or protocol["validation_policy"]
        != "one_locked_rank_intervention_per_batch_no_rank_curve"
        or protocol["calibration_a_policy"] != "parse_validate_hash_only"
        or protocol["test_policy"] != "parse_validate_hash_only"
        or protocol["compression_claim"] is not False
        or protocol["parameter_mac_speed_claim"] is not False
        or type(protocol["maximum_tokenized_length"]) is not int
        or protocol["maximum_tokenized_length"] < 2
        or type(protocol["tokenization_batch_size"]) is not int
        or protocol["tokenization_batch_size"] <= 0
    ):
        raise ValueError("projection-ladder scientific semantics are invalid")
    nll_atol = _finite(
        protocol["selection_nll_atol"],
        label="selection_nll_atol",
        minimum=0.0,
    )
    top1_min = _finite(
        protocol["selection_top1_min"],
        label="selection_top1_min",
        minimum=0.0,
        maximum=1.0,
    )
    identity_tolerance = _finite(
        protocol["identity_nll_atol"],
        label="identity_nll_atol",
        minimum=0.0,
    )
    meaningful_fraction = _finite(
        protocol["max_meaningful_retained_fraction"],
        label="max_meaningful_retained_fraction",
        minimum=0.0,
        maximum=1.0,
    )

    streams = protocol["tokenized_splits"]
    prompt_metadata = protocol["prompt_splits"]
    if (
        not isinstance(streams, Mapping)
        or tuple(streams) != ("calibration_b", "validation")
        or not isinstance(prompt_metadata, Mapping)
    ):
        raise ValueError("projection tokenized split provenance is invalid")
    selection_stream, _ = _validated_tokenized_stream(
        streams["calibration_b"],
        split_name="calibration_b",
    )
    validation_stream, _ = _validated_tokenized_stream(
        streams["validation"],
        split_name="validation",
    )
    counts = prompt_metadata.get("counts")
    per_prompt = prompt_metadata.get("per_prompt_sha256")
    normalized = prompt_metadata.get("normalized_sha256")
    split_names = ("calibration_a", "calibration_b", "validation", "test")
    if (
        not isinstance(counts, Mapping)
        or tuple(counts) != split_names
        or not isinstance(per_prompt, Mapping)
        or tuple(per_prompt) != split_names
        or not isinstance(normalized, Mapping)
        or tuple(normalized) != split_names
    ):
        raise ValueError("projection prompt provenance is invalid")
    all_hashes = []
    for split_name in split_names:
        hashes = per_prompt[split_name]
        if (
            type(counts[split_name]) is not int
            or counts[split_name] <= 0
            or not isinstance(hashes, list)
            or len(hashes) != counts[split_name]
            or any(not _is_sha256(value) for value in hashes)
            or normalized[split_name]
            != _ordered_prompt_hash_digest(hashes)
        ):
            raise ValueError("projection prompt hashes are invalid")
        all_hashes.extend(hashes)
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("projection prompt hashes must be disjoint")
    if (
        selection_stream["source_prompt_sha256"]
        != per_prompt["calibration_b"]
        or validation_stream["source_prompt_sha256"]
        != per_prompt["validation"]
        or selection_stream["sequences"] != counts["calibration_b"]
        or validation_stream["sequences"] != counts["validation"]
    ):
        raise ValueError("projection streams do not bind prompt hashes")
    streamed = set(selection_stream["source_prompt_sha256"]) | set(
        validation_stream["source_prompt_sha256"]
    )
    if streamed & (
        set(per_prompt["calibration_a"]) | set(per_prompt["test"])
    ):
        raise ValueError("hash-only prompt split appears in model stream")
    disjointness_fields = {
        "fresh_prompt_sha256",
        "weighted_prompt_sha256",
        "gated_prompt_sha256",
        "fresh_count",
        "weighted_count",
        "gated_count",
        "weighted_overlap_count",
        "gated_overlap_count",
        "verified_before_model_load_or_tokenization",
    }
    if set(disjointness) != disjointness_fields:
        raise ValueError("predecessor prompt disjointness fields are invalid")
    fresh_hashes = set(disjointness["fresh_prompt_sha256"])
    weighted_hashes = set(disjointness["weighted_prompt_sha256"])
    gated_hashes = set(disjointness["gated_prompt_sha256"])
    if (
        fresh_hashes != set(all_hashes)
        or len(fresh_hashes) != disjointness["fresh_count"]
        or len(weighted_hashes) != disjointness["weighted_count"]
        or len(gated_hashes) != disjointness["gated_count"]
        or any(
            not _is_sha256(value)
            for value in fresh_hashes | weighted_hashes | gated_hashes
        )
        or fresh_hashes & weighted_hashes
        or fresh_hashes & gated_hashes
        or disjointness["weighted_overlap_count"] != 0
        or disjointness["gated_overlap_count"] != 0
        or disjointness[
            "verified_before_model_load_or_tokenization"
        ]
        is not True
    ):
        raise ValueError("predecessor prompt disjointness is invalid")

    selection_fields = {
        "candidate_direct_diagnostics",
        "candidate_behavior",
        "nested_projection_error_control",
        "full_width_identity",
        "lock",
        "tokenized_stream",
    }
    validation_fields = {
        "locked_candidate",
        "direct_diagnostic",
        "behavior",
        "behavior_gates",
        "behavior_fidelity_passed",
        "full_width_codec_direct_control",
        "fidelity_viable_reduced_rank",
        "meaningful_rank_compression",
        "tokenized_stream",
    }
    if set(selection) != selection_fields or set(validation) != validation_fields:
        raise ValueError("projection analysis fields are invalid")
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    direct = selection["candidate_direct_diagnostics"]
    behavior = selection["candidate_behavior"]
    lock = selection["lock"]
    full_width = selection["full_width_identity"]
    if (
        not isinstance(direct, Mapping)
        or tuple(direct) != candidate_ids
        or not isinstance(behavior, Mapping)
        or tuple(behavior) != candidate_ids
        or not isinstance(lock, Mapping)
        or not isinstance(full_width, Mapping)
    ):
        raise ValueError("projection selection mappings are invalid")
    for candidate_id in candidate_ids:
        _recompute_direct_aggregate(
            direct[candidate_id],
            stream=selection_stream,
            width=width,
            kind="target_ls",
        )
        _recompute_behavior_aggregate(
            behavior[candidate_id],
            stream=selection_stream,
        )
    expected_nested = _assert_nested_direct_error(candidates, direct)
    if selection["nested_projection_error_control"] != expected_nested:
        raise ValueError("nested projection error control does not recompute")
    expected_ledger = _candidate_ledger(
        candidates,
        direct=direct,
        behavior=behavior,
        nll_atol=nll_atol,
        top1_min=top1_min,
    )
    expected_locked, expected_lock = _lock_candidate(
        candidates,
        expected_ledger,
    )
    if lock != expected_lock:
        raise ValueError("projection selection lock does not recompute")
    full_fields = {
        "rank_direct",
        "rank_behavior",
        "runtime_float32_codec_direct",
        "mathematical_float64_codec_direct",
        "codec_behavior",
        "passed",
    }
    if set(full_width) != full_fields:
        raise ValueError("full-width identity fields are invalid")
    full_candidate = candidates[-1]
    _recompute_direct_aggregate(
        full_width["runtime_float32_codec_direct"],
        stream=selection_stream,
        width=width,
        kind="codec_float32",
    )
    _recompute_direct_aggregate(
        full_width["mathematical_float64_codec_direct"],
        stream=selection_stream,
        width=width,
        kind="codec_float64",
    )
    _recompute_behavior_aggregate(
        full_width["codec_behavior"],
        stream=selection_stream,
    )
    if (
        full_width["rank_direct"] != direct[full_candidate.candidate_id]
        or full_width["rank_behavior"]
        != behavior[full_candidate.candidate_id]
        or full_width["passed"] is not True
        or not _full_width_controls_passed(
            rank_direct=full_width["rank_direct"],  # type: ignore[arg-type]
            rank_behavior=full_width["rank_behavior"],  # type: ignore[arg-type]
            runtime_direct=full_width[
                "runtime_float32_codec_direct"
            ],  # type: ignore[arg-type]
            mathematical_direct=full_width[
                "mathematical_float64_codec_direct"
            ],  # type: ignore[arg-type]
            codec_behavior=full_width["codec_behavior"],  # type: ignore[arg-type]
            identity_nll_atol=identity_tolerance,
        )
    ):
        raise ValueError("full-width identity does not recompute")
    if (
        selection["tokenized_stream"] != selection_stream
        or validation["tokenized_stream"] != validation_stream
        or validation["locked_candidate"] != expected_locked.metadata()
    ):
        raise ValueError("projection split or locked candidate alias is invalid")
    validation_behavior = validation["behavior"]
    if not isinstance(validation_behavior, Mapping):
        raise ValueError("validation behavior is invalid")
    _recompute_direct_aggregate(
        validation["direct_diagnostic"],
        stream=validation_stream,
        width=width,
        kind="target_ls",
    )
    _recompute_behavior_aggregate(
        validation_behavior,
        stream=validation_stream,
    )
    expected_validation_gates = _behavior_gate(
        validation_behavior,
        nll_atol=nll_atol,
        top1_min=top1_min,
    )
    validation_passed = all(expected_validation_gates.values())
    if (
        validation["behavior_gates"] != expected_validation_gates
        or validation["behavior_fidelity_passed"] is not validation_passed
    ):
        raise ValueError("validation behavior gates do not recompute")
    selected_b_row = next(
        row
        for row in expected_ledger
        if row["candidate"]["candidate_id"]  # type: ignore[index]
        == expected_locked.candidate_id
    )
    selected_b_passed = (
        selected_b_row["behavior_fidelity_passed"] is True
    )
    reduced = expected_locked.retained_rank < width
    fidelity_viable = selected_b_passed and validation_passed and reduced
    meaningful = (
        fidelity_viable
        and expected_locked.retained_fraction <= meaningful_fraction
    )
    if (
        validation["fidelity_viable_reduced_rank"] is not fidelity_viable
        or validation["meaningful_rank_compression"] is not meaningful
    ):
        raise ValueError("validation rank conclusions do not recompute")
    direct_control = validation["full_width_codec_direct_control"]
    if (
        not isinstance(direct_control, Mapping)
        or set(direct_control) != {
            "runtime_float32",
            "mathematical_float64",
        }
        or any(
            float(direct_control[name]["block_delta_nrmse"]) > 1e-5  # type: ignore[index]
            or float(direct_control[name]["block_delta_cosine"])  # type: ignore[index]
            < 1.0 - 1e-7
            for name in ("runtime_float32", "mathematical_float64")
        )
    ):
        raise ValueError("validation full-width direct control is invalid")
    _recompute_direct_aggregate(
        direct_control["runtime_float32"],
        stream=validation_stream,
        width=width,
        kind="codec_float32",
    )
    _recompute_direct_aggregate(
        direct_control["mathematical_float64"],
        stream=validation_stream,
        width=width,
        kind="codec_float64",
    )

    status_fields = {
        "scope",
        "calibration_a_evaluated",
        "calibration_b_rank_curve_evaluated",
        "validation_locked_before_evaluation",
        "locked_validation_interventions_per_batch",
        "test_evaluated",
        "model_weights_changed",
        "model_weights_in_artifact",
        "prompt_text_in_artifact",
        "inference_executor",
        "behavioral_upper_bound_claim",
        "compression_claim",
        "parameter_mac_speed_claim",
        "selection_failed",
        "reduced_fidelity_candidate_found",
        "locked_validation_fidelity_passed",
        "fidelity_viable_reduced_rank",
        "meaningful_rank_compression",
    }
    if (
        set(status) != status_fields
        or status["scope"]
        != "target_informed_projection_behavioral_rank_ladder"
        or status["calibration_a_evaluated"] is not False
        or status["calibration_b_rank_curve_evaluated"] is not True
        or status["validation_locked_before_evaluation"] is not True
        or status["locked_validation_interventions_per_batch"] != 1
        or status["test_evaluated"] is not False
        or status["model_weights_changed"] is not False
        or status["model_weights_in_artifact"] is not False
        or status["prompt_text_in_artifact"] is not False
        or status["inference_executor"] is not False
        or status["behavioral_upper_bound_claim"] is not False
        or status["compression_claim"] is not False
        or status["parameter_mac_speed_claim"] is not False
        or status["selection_failed"] is not expected_lock[
            "selection_failed"
        ]
        or status["reduced_fidelity_candidate_found"]
        is not expected_lock["reduced_candidate_found"]
        or status["locked_validation_fidelity_passed"]
        is not validation_passed
        or status["fidelity_viable_reduced_rank"] is not fidelity_viable
        or status["meaningful_rank_compression"] is not meaningful
    ):
        raise ValueError("projection scientific status is invalid")

    expected_report = _build_report(
        payload,
        output=artifact_path,
        scientific_digest=digest,
    )
    report = json.loads(
        artifact_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(report, Mapping)
        or _report_sha256(report) != raw["report_sha256"]
        or report
        != json.loads(
            json.dumps(
                expected_report,
                sort_keys=True,
                allow_nan=False,
            )
        )
    ):
        raise ValueError("projection JSON report does not match payload")
    return {
        "model": model,
        "output_codec": output_codec,
        "locked_candidate": expected_locked.metadata(),
        "selection": copy.deepcopy(dict(selection)),
        "validation": copy.deepcopy(dict(validation)),
        "metadata": {
            "scientific_payload_sha256": digest,
            "report_sha256": raw["report_sha256"],
            "protocol": copy.deepcopy(dict(protocol)),
            "predecessors": copy.deepcopy(dict(predecessors)),
        },
        "report": copy.deepcopy(dict(report)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Locate the first behaviorally faithful target-informed output "
            "projection rank for a frozen Gemma block."
        )
    )
    parser.add_argument("--weighted-artifact", type=Path, required=True)
    parser.add_argument("--gated-artifact", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--prompt-splits",
        type=Path,
        default=DEFAULT_PROMPT_SPLITS,
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument(
        "--retained-ranks",
        dest="ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANKS),
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
        "--identity-nll-atol",
        type=float,
        default=DEFAULT_IDENTITY_NLL_ATOL,
    )
    parser.add_argument(
        "--max-meaningful-retained-fraction",
        type=float,
        default=DEFAULT_MAX_MEANINGFUL_RETAINED_FRACTION,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_projection_ladder(
        weighted_artifact_path=arguments.weighted_artifact,
        gated_artifact_path=arguments.gated_artifact,
        model_id=arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        prompt_splits_path=arguments.prompt_splits,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        ranks=arguments.ranks,
        selection_nll_atol=arguments.selection_nll_atol,
        selection_top1_min=arguments.selection_top1_min,
        identity_nll_atol=arguments.identity_nll_atol,
        max_meaningful_retained_fraction=(
            arguments.max_meaningful_retained_fraction
        ),
        device_name=arguments.device,
        dtype=arguments.dtype,
        local_files_only=arguments.local_files_only,
        output=arguments.output,
    )
    validation = report["analysis"]["validation"]
    assert isinstance(validation, Mapping)
    print(
        json.dumps(
            {
                "output": report["artifact"]["tensor_output"],  # type: ignore[index]
                "locked_candidate": validation["locked_candidate"],
                "delta_nll_per_token": validation["behavior"][
                    "delta_nll_per_token"
                ],
                "top1_agreement": validation["behavior"][
                    "top1_agreement_to_baseline"
                ],
                "fidelity_viable_reduced_rank": validation[
                    "fidelity_viable_reduced_rank"
                ],
                "meaningful_rank_compression": validation[
                    "meaningful_rank_compression"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
