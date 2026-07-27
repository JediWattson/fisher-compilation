"""Exploratory Gemma dense-supermode fit/guard experiment.

This runner deliberately reuses the already-consumed structured-strong-v9
calibration-A guard.  It can measure whether the new groupwise dense
``512 -> 384`` construction is worth a fresh experiment, but its result can
never authorize calibration B, validation, test, or a scientific compression
claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .adapters import Gemma3CausalLMAdapter
from .gemma3_ablation_experiment import (
    _FrozenModelTensorGuard,
    _is_sha256,
    _update_payload_digest,
)
from .gemma3_codimension_rotation_experiment import _file_sha256
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_width_single_layer_experiment import (
    DEFAULT_MINIMUM_FISHER_ROWS,
    DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE,
    _direct_gates,
    _require_complete_middle_layer_demand,
    _tokenized_stream_contract,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_rotated_span_executor_experiment import _behavior_gates
from .gemma3_stability_experiment import (
    _library_versions,
    _tokenizer_provenance,
)
from .gemma3_structured_mlp_pseudo_unit_a_experiment import (
    DEFAULT_DOWN_RIDGE,
    DEFAULT_LAYER_INDEX,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MINIMUM_PARTITION_LENGTH_BUCKETS,
    DEFAULT_MINIMUM_PARTITION_SUPERVISED_TOKENS,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    _assert_no_tensor,
    _authenticate_format5_parent,
    _build_deletion_baseline,
    _load_structured_v9_corpus_preflight,
    _partition_binding,
    _publish_artifact_pair,
    _publish_json_exclusive,
    _safe_score_collection_report,
    _standard_thresholds,
    _without_boundaries,
)
from .gemma3_structured_mlp_compression_a_experiment import (
    _standard_minima,
)
from .gemma3_structured_single_layer_experiment import (
    DEFAULT_NATIVE_PARITY_TOLERANCE,
    _branch_gates,
    collect_structured_training_batches,
    evaluate_structured_candidates,
    load_gemma3_structured_single_layer_artifact,
)
from .structured_mlp_compression_pipeline import (
    collect_gemma_mlp_fisher_taylor_batches,
)
from .structured_mlp_dense_supermode_pipeline import (
    _NATIVE_PIVOT_REPORT_DOMAIN,
    DenseSupermodeFitWeights,
    StructuredMLPDenseSupermodeCandidate,
    build_structured_mlp_dense_supermode_candidate,
    build_structured_mlp_dense_supermode_native_pivot_control,
)
from .structured_mlp_dense_supermodes import (
    DenseSupermodeObjectiveWeights,
    build_fisher_jacobian_dense_supermode_plan,
)
from .structured_operator_bootstrap import (
    structured_operator_coefficient_sha256,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


GEMMA_DENSE_SUPERMODE_DEV_SCHEMA = (
    "fisher_graph.gemma3_structured_mlp_dense_supermode_development"
)
GEMMA_DENSE_SUPERMODE_DEV_FORMAT_VERSION = 1
GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH = 2_048
GEMMA_DENSE_SUPERMODE_POOL_WIDTH = 512
GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH = 384
GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH = 1_920
DEFAULT_GENERATOR_STEPS = 256
DEFAULT_GENERATOR_LEARNING_RATE = 2e-3
DEFAULT_GENERATOR_MINIBATCH_ROWS = 256
DEFAULT_GENERATOR_GRADIENT_CLIP_NORM = 1.0

_DENSE = "dense_supermode_512_to_384"
_NATIVE_PIVOT = "native_pivot_pruning_2048_to_1920_down_refit"
_DIAGONAL_DELETION = "diagonal_fisher_deletion_2048_to_1920_down_refit"
_CANDIDATE_NAMES = (_DENSE, _NATIVE_PIVOT, _DIAGONAL_DELETION)
_PAYLOAD_DOMAIN = (
    b"fisher_graph.gemma3_dense_supermode_dev.payload.v1\0"
)
_REPORT_DOMAIN = b"fisher_graph.gemma3_dense_supermode_dev.report.v1\0"


def _frozen_generator_protocol() -> dict[str, object]:
    return {
        "steps": DEFAULT_GENERATOR_STEPS,
        "learning_rate": DEFAULT_GENERATOR_LEARNING_RATE,
        "minibatch_rows": DEFAULT_GENERATOR_MINIBATCH_ROWS,
        "gradient_clip_norm": DEFAULT_GENERATOR_GRADIENT_CLIP_NORM,
        "fixed_before_reused_guard_access": True,
    }


def _progress(message: str) -> None:
    print(
        f"[gemma-dense-supermode-dev] {message}",
        file=sys.stderr,
        flush=True,
    )


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def _report_sha256(value: Mapping[str, object]) -> str:
    return _json_sha256(value, domain=_REPORT_DOMAIN)


def _required_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class _MaterializedSplit:
    batches: tuple[object, ...]
    stream: Mapping[str, object]
    contract: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _BuiltCandidates:
    executors: Mapping[str, StructuredTransformerLayerExecutor]
    dense: StructuredMLPDenseSupermodeCandidate
    native_pivot_report: Mapping[str, object]
    deletion_report: Mapping[str, object]
    score_report: Mapping[str, object]
    plan_report: Mapping[str, object]
    structured_training_batches: int


def _candidate_fingerprints(
    value: _BuiltCandidates,
) -> dict[str, str]:
    if set(value.executors) != set(_CANDIDATE_NAMES):
        raise ValueError("dense-supermode development candidates are invalid")
    return {
        name: executor.execution_fingerprint()
        for name, executor in value.executors.items()
    }


def _execute_fit_then_guard(
    *,
    materialize_fit: Callable[[], object],
    build_from_fit: Callable[[object], _BuiltCandidates],
    materialize_guard: Callable[[], object],
    evaluate_guard: Callable[[_BuiltCandidates, object], object],
) -> tuple[object, _BuiltCandidates, object, object, Mapping[str, str]]:
    """Freeze the extensible local candidate map before guard access."""

    fit = materialize_fit()
    candidates = build_from_fit(fit)
    frozen = _candidate_fingerprints(candidates)
    if any(not _is_sha256(value) for value in frozen.values()):
        raise ValueError("candidate fingerprints are invalid")
    guard = materialize_guard()
    evaluation = evaluate_guard(candidates, guard)
    if _candidate_fingerprints(candidates) != frozen:
        raise RuntimeError("guard evaluation mutated a frozen candidate")
    return fit, candidates, guard, evaluation, frozen


def _diagnostic_gate_report(
    evaluation: Mapping[str, object],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    """Apply ordinary gates without granting any held-out authorization."""

    behavior = evaluation.get("behavior")
    direct = evaluation.get("direct")
    branches = evaluation.get("branches")
    audits = evaluation.get("execution_audits")
    parity = evaluation.get("ordinary_vs_segmented_native")
    replay = evaluation.get("native_boundary_replay")
    if not all(
        isinstance(value, Mapping)
        for value in (behavior, direct, branches, audits, parity, replay)
    ):
        raise ValueError("development guard evaluation is incomplete")
    parity_passed = parity.get("passed") is True  # type: ignore[union-attr]
    replay_passed = replay.get("passed") is True  # type: ignore[union-attr]
    global_controls = parity_passed and replay_passed
    rows: dict[str, object] = {}
    for name in _CANDIDATE_NAMES:
        behavior_row = behavior.get(name)  # type: ignore[union-attr]
        direct_row = direct.get(name)  # type: ignore[union-attr]
        branch_row = branches.get(name)  # type: ignore[union-attr]
        audit = audits.get(name)  # type: ignore[union-attr]
        if not all(
            isinstance(value, Mapping)
            for value in (behavior_row, direct_row, branch_row, audit)
        ):
            raise ValueError(f"development candidate {name!r} is incomplete")
        behavior_result = _behavior_gates(
            behavior_row,  # type: ignore[arg-type]
            nll_atol=thresholds["nll_atol"],
            top1_min=thresholds["top1_min"],
            teacher_kl_max=thresholds["teacher_kl_max"],
            p90_abs_nll_max=thresholds["p90_abs_nll_max"],
            p10_top1_min=thresholds["p10_top1_min"],
        )
        direct_result = _direct_gates(
            direct_row,  # type: ignore[arg-type]
            block_delta_nrmse_max=thresholds["block_delta_nrmse_max"],
            block_delta_cosine_min=thresholds["block_delta_cosine_min"],
        )
        branch_result = _branch_gates(
            branch_row,  # type: ignore[arg-type]
            nrmse_max=thresholds["branch_delta_nrmse_max"],
            cosine_min=thresholds["branch_delta_cosine_min"],
        )
        standard_passed = (
            all(behavior_result.values())
            and all(direct_result.values())
            and all(branch_result.values())
            and audit.get("passed") is True  # type: ignore[union-attr]
            and global_controls
        )
        rows[name] = {
            "behavior": behavior_result,
            "direct": direct_result,
            "branches": branch_result,
            "execution": audit.get("passed") is True,  # type: ignore[union-attr]
            "standard_passed": standard_passed,
            "authorizes_calibration_b": False,
            "nonconfirmatory_reused_guard": True,
        }
    dense = rows[_DENSE]
    assert isinstance(dense, Mapping)
    dense_direct = direct.get(_DENSE)  # type: ignore[union-attr]
    if not isinstance(dense_direct, Mapping):
        raise ValueError("dense development direct metrics are missing")
    dense_block_nrmse = float(dense_direct["block_delta_nrmse"])
    margin_passed = (
        dense_block_nrmse
        <= thresholds["primary_guard_block_delta_nrmse_max"]
    )
    comparisons: dict[str, object] = {}
    for control_name in _CANDIDATE_NAMES:
        if control_name == _DENSE:
            continue
        control_direct = direct.get(control_name)  # type: ignore[union-attr]
        if not isinstance(control_direct, Mapping):
            raise ValueError(
                f"control direct metrics {control_name!r} are missing"
            )
        control_block_nrmse = float(
            control_direct["block_delta_nrmse"]
        )
        comparisons[control_name] = {
            "metric": "block_delta_nrmse",
            "dense": dense_block_nrmse,
            "control": control_block_nrmse,
            "dense_minus_control": (
                dense_block_nrmse - control_block_nrmse
            ),
            "dense_to_control_ratio": (
                dense_block_nrmse / control_block_nrmse
                if control_block_nrmse > 0.0
                else None
            ),
            "dense_strictly_better": (
                dense_block_nrmse < control_block_nrmse
            ),
        }
    beats_every_control = all(
        comparison["dense_strictly_better"] is True
        for comparison in comparisons.values()  # type: ignore[union-attr]
    )
    dense_diagnostic_passed = (
        dense["standard_passed"] is True
        and margin_passed
        and beats_every_control
    )
    dense_row = dict(dense)
    dense_row["primary_guard_block_delta_nrmse_margin"] = margin_passed
    dense_row["strictly_beats_every_control_on_block_delta_nrmse"] = (
        beats_every_control
    )
    rows[_DENSE] = dense_row
    return {
        "candidates": rows,
        "ordinary_vs_segmented_native": parity_passed,
        "native_boundary_replay": replay_passed,
        "block_delta_nrmse_comparisons": comparisons,
        "dense_diagnostic_passed": dense_diagnostic_passed,
        "any_candidate_authorizes_calibration_b": False,
        "policy": (
            "dense_standard_gates_and_block_nrmse_at_most_0_015_and_"
            "strictly_better_than_every_matched_control_on_consumed_v9_"
            "a_guard; development_only_and_never_authorizes_heldout"
        ),
    }


def _dense_standard_gates_passed(
    gates: Mapping[str, object],
) -> bool:
    candidates = _required_mapping(
        gates.get("candidates"),
        label="diagnostic gate candidates",
    )
    dense = _required_mapping(
        candidates.get(_DENSE),
        label="dense diagnostic gates",
    )
    return dense.get("standard_passed") is True


def _resource_report(
    parent: StructuredTransformerLayerExecutor,
    candidates: Mapping[str, StructuredTransformerLayerExecutor],
    logical_accounting: Mapping[str, object],
) -> dict[str, object]:
    source_width = parent.config.transformer.feed_forward.intermediate_width
    residual_width = parent.width
    source_parameters = parent.learned_parameter_count
    source_mlp_macs = 3 * residual_width * source_width
    rows: dict[str, object] = {}
    for name, candidate in candidates.items():
        logical = logical_accounting.get(name)
        if not isinstance(logical, Mapping):
            raise ValueError("candidate logical accounting is invalid")
        retained_width = (
            candidate.config.transformer.feed_forward.intermediate_width
        )
        candidate_mlp_macs = 3 * residual_width * retained_width
        valid_tokens = int(logical["valid_tokens"])
        candidate_total = int(logical["logical_total_macs"])
        source_total = candidate_total + valid_tokens * (
            source_mlp_macs - candidate_mlp_macs
        )
        rows[name] = {
            "source_intermediate_width": source_width,
            "retained_intermediate_width": retained_width,
            "source_layer_parameters": source_parameters,
            "candidate_layer_parameters": candidate.learned_parameter_count,
            "removed_layer_parameters": (
                source_parameters - candidate.learned_parameter_count
            ),
            "retained_layer_parameter_ratio": (
                candidate.learned_parameter_count / source_parameters
            ),
            "source_mlp_linear_macs_per_valid_token": source_mlp_macs,
            "candidate_mlp_linear_macs_per_valid_token": candidate_mlp_macs,
            "removed_mlp_linear_macs_per_valid_token": (
                source_mlp_macs - candidate_mlp_macs
            ),
            "guard_source_layer_analytic_macs": source_total,
            "guard_candidate_layer_analytic_macs": candidate_total,
            "guard_analytic_mac_ratio": candidate_total / source_total,
            "latency_or_kernel_speed_claim": False,
        }
    return {
        "scope": "single_gemma_layer_4",
        "source_width": source_width,
        "runtime_width": GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH,
        "candidates": rows,
    }


def _validate_tensor_locations(value: object) -> None:
    def walk(current: object, path: tuple[str, ...]) -> None:
        if isinstance(current, torch.Tensor):
            if len(path) < 3 or path[:2] != (
                "executor",
                "model_state_dict",
            ):
                raise ValueError(
                    "development artifact contains a tensor outside the "
                    "source-free dense executor state"
                )
            return
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "development artifact mapping keys must be strings"
                    )
                walk(item, (*path, str(key)))
        elif isinstance(current, (tuple, list)):
            for index, item in enumerate(current):
                walk(item, (*path, str(index)))

    walk(value, ())


def _build_json_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
    scientific_payload_sha256: str | None,
    main_artifact_written: bool,
) -> dict[str, object]:
    report = {
        **{
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key != "executor"
        },
        "artifact": {
            "tensor_file": tensor_file if main_artifact_written else None,
            "main_artifact_written": main_artifact_written,
            "contains_compressed_executor_weights": main_artifact_written,
            "contains_source_model_weights": False,
            "contains_prompt_text": False,
            "contains_tokenizer_state": False,
            "contains_teacher_targets": False,
            "contains_fisher_taylor_scores": False,
            "scientific_payload_sha256": scientific_payload_sha256,
        },
    }
    _assert_no_tensor(report, label="dense-supermode JSON report")
    return report


def run_gemma3_structured_mlp_dense_supermode_dev_experiment(
    *,
    parent_artifact_path: Path | str,
    prompt_splits_path: Path | str,
    family_manifest_path: Path | str,
    corpus_audit_path: Path | str,
    revision: str,
    output: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    layer_index: int = DEFAULT_LAYER_INDEX,
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Fit on v9 A-fit and diagnose on the already-consumed v9 A-guard."""

    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", revision) is None
    ):
        raise ValueError("revision must be an exact lowercase commit hash")
    if (
        layer_index != DEFAULT_LAYER_INDEX
        or max_length != DEFAULT_MAX_LENGTH
        or tokenization_batch_size != DEFAULT_TOKENIZATION_BATCH_SIZE
    ):
        raise ValueError(
            "dense-supermode development requires layer 4, max length 256, "
            "and tokenization batch size 4"
        )
    if device_name != "cpu" or dtype != "float32":
        raise ValueError(
            "dense-supermode development requires CPU float32 execution"
        )
    resolved_output = Path(output)
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    report_path = resolved_output.with_suffix(".json")
    if resolved_output.exists() or report_path.exists():
        raise FileExistsError(
            "refusing to overwrite a dense-supermode development result"
        )

    _progress("preflight: authenticate the complete v9 corpus")
    corpus = _load_structured_v9_corpus_preflight(
        prompt_splits_path=prompt_splits_path,
        family_manifest_path=family_manifest_path,
        corpus_audit_path=corpus_audit_path,
    )
    device = resolve_torch_device(device_name)
    _progress("parent: strict-load and authenticate the format-5 executor")
    loaded_parent = load_gemma3_structured_single_layer_artifact(
        parent_artifact_path,
        map_location=device,
    )
    parent, parent_training, _ = _authenticate_format5_parent(
        loaded_parent,
        model_id=model_id,
        revision=revision,
        layer_index=layer_index,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
    )
    parent_metadata = loaded_parent.get("metadata")
    parent_model = loaded_parent.get("model")
    bootstrap = parent_training.get("bootstrap")
    if (
        not isinstance(parent_metadata, Mapping)
        or not isinstance(parent_model, Mapping)
        or not isinstance(bootstrap, Mapping)
    ):
        raise ValueError("format-5 parent binding is incomplete")
    parent_fingerprint = parent.execution_fingerprint()
    if (
        parent.config.transformer.feed_forward.intermediate_width
        != GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
    ):
        raise ValueError("dense-supermode parent width must be 2048")
    parent_binding = {
        "artifact_tensor_file_sha256": parent_metadata[
            "tensor_file_sha256"
        ],
        "artifact_scientific_payload_sha256": parent_metadata[
            "scientific_payload_sha256"
        ],
        "artifact_report_sha256": parent_metadata["report_sha256"],
        "artifact_format_version": 5,
        "model_resolved_commit": parent_model["resolved_commit"],
        "layer_index": layer_index,
        "layer_id": bootstrap["layer_id"],
        "primary_execution_fingerprint": parent_fingerprint,
        "primary_coefficient_sha256": (
            structured_operator_coefficient_sha256(parent)
        ),
    }

    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load the pinned local Gemma checkpoint")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    source_guard = _FrozenModelTensorGuard(model)
    model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    if (
        model_metadata.get("resolved_commit") != revision
        or model_metadata.get("config_sha256")
        != parent_model.get("config_sha256")
    ):
        raise ValueError("local pinned model does not match the parent")
    adapter = Gemma3CausalLMAdapter(model)
    layer_plan = adapter.plan_layer_block(layer_index, layer_index)
    layer_id = layer_plan.layer_ids[0]
    if layer_id != parent_binding["layer_id"]:
        raise ValueError("local layer does not match the parent layer")

    def materialize(
        prompts: Sequence[str],
        *,
        split_name: str,
    ) -> _MaterializedSplit:
        batches, stream = _materialize_split(
            tokenizer,
            prompts,
            split_name=split_name,
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        contract = _tokenized_stream_contract(
            stream,
            split_name=split_name,
            minimum_supervised_tokens=(
                DEFAULT_MINIMUM_PARTITION_SUPERVISED_TOKENS
            ),
            minimum_length_buckets=(
                DEFAULT_MINIMUM_PARTITION_LENGTH_BUCKETS
            ),
        )
        return _MaterializedSplit(
            batches=tuple(batches),
            stream=stream,
            contract=contract,
        )

    def materialize_fit() -> _MaterializedSplit:
        _progress("fit: tokenize calibration-A fit")
        return materialize(
            corpus.fit.prompts,
            split_name="calibration_a_fit",
        )

    def build_from_fit(value: object) -> _BuiltCandidates:
        if not isinstance(value, _MaterializedSplit):
            raise TypeError("fit materialization is invalid")
        split_sha256 = value.stream.get("serialized_sha256")
        if not _is_sha256(split_sha256):
            raise ValueError("A-fit token stream digest is invalid")
        _progress("fit: capture structured teacher targets")
        training = collect_structured_training_batches(
            adapter,
            value.batches,  # type: ignore[arg-type]
            layer_id=layer_id,
            positions_per_sequence=DEFAULT_TRAIN_POSITIONS_PER_SEQUENCE,
        )
        _require_complete_middle_layer_demand(
            adapter,
            training,  # type: ignore[arg-type]
        )
        _progress("fit: collect MLP Fisher/Taylor activation scores")
        score_batches, score_report = (
            collect_gemma_mlp_fisher_taylor_batches(
                adapter,
                value.batches,  # type: ignore[arg-type]
                layer_id=layer_id,
                calibration_split_sha256=split_sha256,
            )
        )
        accounting = score_report.get("accounting")
        if (
            not isinstance(accounting, Mapping)
            or int(accounting.get("valid_rows", 0))
            < DEFAULT_MINIMUM_FISHER_ROWS
        ):
            raise ValueError("A-fit Fisher rows are below the minimum")
        sites = parent.config.transformer.operator_sites
        if sites is None:
            raise ValueError("parent has no structured MLP activation site")
        _progress("fit: build the Fisher/output-aware 512-to-384 plan")
        dense_plan = build_fisher_jacobian_dense_supermode_plan(
            score_batches,
            source_down_weight=parent.feed_forward.down_proj.weight,
            calibration_split_sha256=split_sha256,
            activation_site=sites.feed_forward_down_input,
            parent_executor_fingerprint=parent_fingerprint,
            pool_width=GEMMA_DENSE_SUPERMODE_POOL_WIDTH,
            retained_pool_width=GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH,
            expected_source_width=GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH,
            objective_weights=DenseSupermodeObjectiveWeights(),
        )
        targets = tuple(item.targets for item in training)
        _progress("fit: synthesize the dense 1920-wide executor")
        dense = build_structured_mlp_dense_supermode_candidate(
            parent,
            dense_plan,
            targets,
            score_batches,
            calibration_split_sha256=split_sha256,
            fit_weights=DenseSupermodeFitWeights(),
            generator_steps=DEFAULT_GENERATOR_STEPS,
            generator_learning_rate=DEFAULT_GENERATOR_LEARNING_RATE,
            generator_minibatch_rows=DEFAULT_GENERATOR_MINIBATCH_ROWS,
            generator_gradient_clip_norm=(
                DEFAULT_GENERATOR_GRADIENT_CLIP_NORM
            ),
        )
        dense.validate_integrity()
        _progress("fit: build the equal-width native-pivot pruning control")
        native_pivot, native_pivot_report = (
            build_structured_mlp_dense_supermode_native_pivot_control(
                parent,
                dense_plan,
                targets,
                score_batches,
                calibration_split_sha256=split_sha256,
                down_ridge=DEFAULT_DOWN_RIDGE,
            )
        )
        _progress("fit: build the equal-width diagonal-Fisher deletion control")
        deletion, deletion_report = _build_deletion_baseline(
            parent,
            targets,
            score_batches,
            calibration_split_sha256=split_sha256,
            down_ridge=DEFAULT_DOWN_RIDGE,
        )
        executors = {
            _DENSE: dense.executor,
            _NATIVE_PIVOT: native_pivot,
            _DIAGONAL_DELETION: deletion,
        }
        if parent.execution_fingerprint() != parent_fingerprint:
            raise RuntimeError("A-fit construction mutated the parent")
        source_guard.assert_unchanged()
        _assert_no_tensor(dense.report, label="dense pipeline report")
        _assert_no_tensor(
            native_pivot_report,
            label="native-pivot control report",
        )
        _assert_no_tensor(
            deletion_report,
            label="deletion pipeline report",
        )
        return _BuiltCandidates(
            executors=executors,
            dense=dense,
            native_pivot_report=copy.deepcopy(
                dict(native_pivot_report)
            ),
            deletion_report=copy.deepcopy(dict(deletion_report)),
            score_report=_safe_score_collection_report(score_report),
            plan_report=copy.deepcopy(dense_plan.metadata()),
            structured_training_batches=len(training),
        )

    def materialize_guard() -> _MaterializedSplit:
        _progress("guard: tokenize reused nonconfirmatory calibration-A guard")
        return materialize(
            corpus.guard.prompts,
            split_name="calibration_a_guard",
        )

    def evaluate_guard(
        candidates: _BuiltCandidates,
        guard: object,
    ) -> object:
        if not isinstance(guard, _MaterializedSplit):
            raise TypeError("guard materialization is invalid")
        _progress("guard: run native-stack replacement evaluation")
        return evaluate_structured_candidates(
            adapter,
            guard.batches,  # type: ignore[arg-type]
            plan=layer_plan,
            layer_id=layer_id,
            candidates=candidates.executors,
            native_parity_tolerance=DEFAULT_NATIVE_PARITY_TOLERANCE,
        )

    fit, built, guard, raw_evaluation, frozen = _execute_fit_then_guard(
        materialize_fit=materialize_fit,
        build_from_fit=build_from_fit,
        materialize_guard=materialize_guard,
        evaluate_guard=evaluate_guard,
    )
    if (
        not isinstance(fit, _MaterializedSplit)
        or not isinstance(guard, _MaterializedSplit)
        or not isinstance(raw_evaluation, Mapping)
    ):
        raise RuntimeError("fit/guard orchestration returned invalid values")
    source_guard.assert_unchanged()
    if parent.execution_fingerprint() != parent_fingerprint:
        raise RuntimeError("guard evaluation mutated the parent")
    built.dense.validate_integrity()

    evaluation = _without_boundaries(raw_evaluation)
    thresholds = _standard_thresholds()
    gates = _diagnostic_gate_report(evaluation, thresholds=thresholds)
    evaluation["gates"] = gates
    logical = evaluation.get("logical_accounting")
    if not isinstance(logical, Mapping):
        raise ValueError("guard logical accounting is invalid")
    resources = _resource_report(parent, built.executors, logical)
    fit_sha256 = fit.stream.get("serialized_sha256")
    guard_sha256 = guard.stream.get("serialized_sha256")
    if (
        not _is_sha256(fit_sha256)
        or not _is_sha256(guard_sha256)
        or fit_sha256 == guard_sha256
    ):
        raise ValueError("fit and reused-guard stream bindings are invalid")
    dense_passed = gates["dense_diagnostic_passed"] is True

    protocol = {
        "corpus": copy.deepcopy(dict(corpus.source_corpus)),
        "calibration_a_fit_partition": _partition_binding(corpus.fit),
        "calibration_a_guard_partition": _partition_binding(corpus.guard),
        "fit_guard_family_disjoint": True,
        "fit_guard_prompt_disjoint": True,
        "fit_may_construct_or_train_candidate": True,
        "guard_may_update_candidate": False,
        "guard_is_reused_and_nonconfirmatory": True,
        "guard_was_consumed_before_this_method_was_frozen": True,
        "guard_result_may_authorize_calibration_b": False,
        "source_width": GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH,
        "pool_width": GEMMA_DENSE_SUPERMODE_POOL_WIDTH,
        "retained_pool_width": GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH,
        "runtime_width": GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH,
        "pool_selection": "lowest_diagonal_fisher_stable_source_index",
        "candidate_names": _CANDIDATE_NAMES,
        "primary_candidate": _DENSE,
        "control_candidates": (_NATIVE_PIVOT, _DIAGONAL_DELETION),
        "generator": _frozen_generator_protocol(),
        "maximum_tokenized_length": max_length,
        "tokenization_batch_size": tokenization_batch_size,
        "layer_index": layer_index,
        "layer_id": layer_id,
        "data_minima": _standard_minima(),
        "thresholds": thresholds,
        "calibration_b_tokenized": False,
        "calibration_b_evaluated": False,
        "validation_tokenized": False,
        "validation_evaluated": False,
        "test_tokenized": False,
        "test_evaluated": False,
        "heldout_ledger_created": False,
        "tokenizer": _tokenizer_provenance(tokenizer),
        "library_versions": _library_versions(),
    }
    fit_report = {
        "partition": _partition_binding(corpus.fit),
        "tokenized_stream": copy.deepcopy(dict(fit.stream)),
        "tokenized_stream_contract": copy.deepcopy(dict(fit.contract)),
        "structured_training_batches": built.structured_training_batches,
        "score_collection": copy.deepcopy(dict(built.score_report)),
        "plan_sha256": built.plan_report["plan_sha256"],
        "candidate_fingerprints_frozen_before_guard": copy.deepcopy(
            dict(frozen)
        ),
    }
    guard_report = {
        "partition": _partition_binding(corpus.guard),
        "tokenized_stream": copy.deepcopy(dict(guard.stream)),
        "tokenized_stream_contract": copy.deepcopy(dict(guard.contract)),
        "fresh_for_this_method": False,
        "nonconfirmatory": True,
        "candidate_fingerprints_before": copy.deepcopy(dict(frozen)),
        "candidate_fingerprints_after": _candidate_fingerprints(built),
        "candidate_mutation_observed": False,
        "evaluation": evaluation,
        "dense_diagnostic_passed": dense_passed,
        "authorizes_calibration_b": False,
    }
    scientific_status = {
        "outcome": (
            "reused_guard_diagnostic_passed"
            if dense_passed
            else "reused_guard_diagnostic_failed"
        ),
        "calibration_a_fit_completed": True,
        "reused_calibration_a_guard_evaluated": True,
        "reused_guard_nonconfirmatory": True,
        "dense_candidate_passed_ordinary_diagnostic_gates": (
            _dense_standard_gates_passed(gates)
        ),
        "source_free_candidate_artifact_written": dense_passed,
        "ready_for_fresh_fit_guard_confirmation": dense_passed,
        "ready_for_calibration_b": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "scientific_compression_success": False,
        "parameter_reduction_measured": True,
        "analytic_mac_reduction_measured": True,
        "latency_or_kernel_speed_claim": False,
    }
    metadata_payload: dict[str, object] = {
        "schema": GEMMA_DENSE_SUPERMODE_DEV_SCHEMA,
        "format_version": GEMMA_DENSE_SUPERMODE_DEV_FORMAT_VERSION,
        "scientific_status": scientific_status,
        "model": model_metadata,
        "protocol": protocol,
        "parent": parent_binding,
        "calibration_a_fit": fit_report,
        "reused_calibration_a_guard": guard_report,
        "pipeline": copy.deepcopy(dict(built.dense.report)),
        "native_pivot_baseline": copy.deepcopy(
            dict(built.native_pivot_report)
        ),
        "deletion_baseline": copy.deepcopy(dict(built.deletion_report)),
        "resource_report": resources,
    }
    _assert_no_tensor(metadata_payload, label="dense development metadata")
    if not dense_passed:
        _progress("publish: diagnostic failed; write JSON only")
        report = _build_json_report(
            metadata_payload,
            tensor_file=resolved_output.name,
            scientific_payload_sha256=None,
            main_artifact_written=False,
        )
        report["report_sha256"] = _report_sha256(report)
        _publish_json_exclusive(
            report_path,
            report,
            must_remain_absent=(resolved_output,),
        )
        return report

    _progress("publish: diagnostic passed; write source-free executor and JSON")
    payload = {
        **metadata_payload,
        "contains_source_model_weights": False,
        "contains_compressed_executor_weights": True,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "contains_teacher_targets": False,
        "contains_fisher_taylor_scores": False,
        "executor": copy.deepcopy(dict(built.dense.artifact_state)),
    }
    _validate_tensor_locations(payload)
    scientific_digest = _payload_sha256(payload)
    report = _build_json_report(
        payload,
        tensor_file=resolved_output.name,
        scientific_payload_sha256=scientific_digest,
        main_artifact_written=True,
    )
    report_digest = _report_sha256(report)
    artifact = {
        **payload,
        "scientific_payload_sha256": scientific_digest,
        "report_sha256": report_digest,
    }
    published = _publish_artifact_pair(
        resolved_output,
        artifact,
        {**report, "report_sha256": report_digest},
        load_published=lambda: load_gemma3_dense_supermode_dev_artifact(
            resolved_output,
            map_location=device,
        ),
    )
    if not isinstance(published, dict):
        raise RuntimeError("published dense-supermode artifact did not reload")
    return published


def _candidate_identity_map(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    result = _required_mapping(value, label=label)
    if set(result) != set(_CANDIDATE_NAMES) or any(
        not _is_sha256(fingerprint) for fingerprint in result.values()
    ):
        raise ValueError(f"{label} is invalid")
    return result


def _validate_success_artifact_bindings(
    raw: Mapping[str, object],
    candidate: StructuredMLPDenseSupermodeCandidate,
) -> None:
    """Recompute the semantic bindings that make a success artifact usable."""

    executor = candidate.executor
    actual_fingerprint = executor.execution_fingerprint()
    protocol = _required_mapping(raw.get("protocol"), label="protocol")
    parent = _required_mapping(raw.get("parent"), label="parent")
    model = _required_mapping(raw.get("model"), label="model")
    fit = _required_mapping(
        raw.get("calibration_a_fit"),
        label="calibration-A fit",
    )
    guard = _required_mapping(
        raw.get("reused_calibration_a_guard"),
        label="reused calibration-A guard",
    )
    pipeline = _required_mapping(raw.get("pipeline"), label="pipeline")
    native = _required_mapping(
        raw.get("native_pivot_baseline"),
        label="native-pivot baseline",
    )
    deletion = _required_mapping(
        raw.get("deletion_baseline"),
        label="deletion baseline",
    )
    resources = _required_mapping(
        raw.get("resource_report"),
        label="resource report",
    )
    status = _required_mapping(
        raw.get("scientific_status"),
        label="scientific status",
    )

    fit_stream = _required_mapping(
        fit.get("tokenized_stream"),
        label="fit tokenized stream",
    )
    guard_stream = _required_mapping(
        guard.get("tokenized_stream"),
        label="guard tokenized stream",
    )
    fit_sha256 = fit_stream.get("serialized_sha256")
    guard_sha256 = guard_stream.get("serialized_sha256")
    fit_partition = _required_mapping(
        protocol.get("calibration_a_fit_partition"),
        label="fit partition",
    )
    guard_partition = _required_mapping(
        protocol.get("calibration_a_guard_partition"),
        label="guard partition",
    )
    thresholds = _required_mapping(
        protocol.get("thresholds"),
        label="thresholds",
    )
    evaluation = _required_mapping(
        guard.get("evaluation"),
        label="guard evaluation",
    )
    recorded_gates = _required_mapping(
        evaluation.get("gates"),
        label="guard gates",
    )
    evaluation_without_gates = dict(evaluation)
    evaluation_without_gates.pop("gates")
    expected_gates = _diagnostic_gate_report(
        evaluation_without_gates,
        thresholds=thresholds,  # type: ignore[arg-type]
    )
    if recorded_gates != expected_gates:
        raise ValueError("dense-supermode guard gates do not recompute")

    frozen = _candidate_identity_map(
        fit.get("candidate_fingerprints_frozen_before_guard"),
        label="fit candidate fingerprints",
    )
    before = _candidate_identity_map(
        guard.get("candidate_fingerprints_before"),
        label="guard-before candidate fingerprints",
    )
    after = _candidate_identity_map(
        guard.get("candidate_fingerprints_after"),
        label="guard-after candidate fingerprints",
    )
    pipeline_provenance = _required_mapping(
        pipeline.get("provenance"),
        label="pipeline provenance",
    )
    pipeline_rung = _required_mapping(
        pipeline.get("rung"),
        label="pipeline rung",
    )
    pipeline_plan = _required_mapping(
        pipeline.get("plan"),
        label="pipeline plan",
    )
    pipeline_policy = _required_mapping(
        pipeline.get("data_policy"),
        label="pipeline data policy",
    )
    native_provenance = _required_mapping(
        native.get("provenance"),
        label="native-pivot provenance",
    )
    native_artifact = _required_mapping(
        native.get("artifact"),
        label="native-pivot artifact",
    )
    native_refit = _required_mapping(
        native.get("terminal_projection_refit"),
        label="native-pivot terminal refit",
    )
    native_refit_targets = _required_mapping(
        native.get("refit_targets"),
        label="native-pivot refit targets",
    )
    deletion_selection = _required_mapping(
        deletion.get("selection"),
        label="deletion selection",
    )
    deletion_refit = _required_mapping(
        deletion.get("terminal_projection_refit"),
        label="deletion terminal refit",
    )
    deletion_refit_targets = _required_mapping(
        deletion.get("refit_targets"),
        label="deletion refit targets",
    )
    plan_sha256 = pipeline_plan.get("plan_sha256")
    parent_fingerprint = parent.get("primary_execution_fingerprint")

    native_payload = dict(native)
    native_report_sha256 = native_payload.pop("report_sha256", None)
    if (
        not _is_sha256(native_report_sha256)
        or _json_sha256(
            native_payload,
            domain=_NATIVE_PIVOT_REPORT_DOMAIN,
        )
        != native_report_sha256
    ):
        raise ValueError("native-pivot report digest is invalid")

    expected_dense_rung = {
        "source_intermediate_width": GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH,
        "pool_width": GEMMA_DENSE_SUPERMODE_POOL_WIDTH,
        "retained_pool_width": GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH,
        "exact_singleton_width": (
            GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
            - GEMMA_DENSE_SUPERMODE_POOL_WIDTH
        ),
        "runtime_intermediate_width": GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH,
        "removed_intermediate_width": (
            GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
            - GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
        ),
        "kind": "groupwise_dense_k_to_r_supermode_synthesis",
    }
    if (
        executor.config.transformer.feed_forward.intermediate_width
        != GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
        or frozen != before
        or before != after
        or before[_DENSE] != actual_fingerprint
        or pipeline_provenance.get("candidate_execution_fingerprint")
        != actual_fingerprint
        or pipeline_provenance.get("parent_executor_fingerprint")
        != parent_fingerprint
        or pipeline_provenance.get("plan_sha256") != plan_sha256
        or dict(pipeline_rung) != expected_dense_rung
        or pipeline_policy.get("calibration_split_sha256") != fit_sha256
        or pipeline_plan.get("calibration_split_sha256") != fit_sha256
        or pipeline_plan.get("source_width")
        != GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
        or pipeline_plan.get("pool_width")
        != GEMMA_DENSE_SUPERMODE_POOL_WIDTH
        or pipeline_plan.get("retained_pool_width")
        != GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH
        or pipeline_plan.get("runtime_width")
        != GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
        or fit.get("plan_sha256") != plan_sha256
        or native_provenance.get("control_execution_fingerprint")
        != before[_NATIVE_PIVOT]
        or native_provenance.get("parent_executor_fingerprint")
        != parent_fingerprint
        or native_provenance.get("plan_sha256") != plan_sha256
        or native_artifact.get("execution_fingerprint")
        != before[_NATIVE_PIVOT]
        or native_refit.get("executor_fingerprint_after")
        != before[_NATIVE_PIVOT]
        or native_refit_targets.get(
            "candidate_execution_fingerprint_before_refit"
        )
        != native_refit.get("executor_fingerprint_before")
        or deletion.get("execution_fingerprint")
        != before[_DIAGONAL_DELETION]
        or deletion_refit.get("executor_fingerprint_after")
        != before[_DIAGONAL_DELETION]
        or deletion_refit_targets.get(
            "candidate_execution_fingerprint_before_refit"
        )
        != deletion_refit.get("executor_fingerprint_before")
        or deletion_selection.get("parent_executor_fingerprint")
        != parent_fingerprint
        or deletion_selection.get("calibration_split_sha256")
        != fit_sha256
        or deletion_selection.get("source_width")
        != GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
        or deletion_selection.get("retained_width")
        != GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
    ):
        raise ValueError(
            "dense-supermode fit/guard candidate bindings are invalid"
        )

    if (
        protocol.get("candidate_names") != _CANDIDATE_NAMES
        or protocol.get("primary_candidate") != _DENSE
        or protocol.get("control_candidates")
        != (_NATIVE_PIVOT, _DIAGONAL_DELETION)
        or protocol.get("generator") != _frozen_generator_protocol()
        or protocol.get("source_width")
        != GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
        or protocol.get("pool_width") != GEMMA_DENSE_SUPERMODE_POOL_WIDTH
        or protocol.get("retained_pool_width")
        != GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH
        or protocol.get("runtime_width")
        != GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
        or protocol.get("pool_selection")
        != "lowest_diagonal_fisher_stable_source_index"
        or protocol.get("layer_index") != DEFAULT_LAYER_INDEX
        or protocol.get("maximum_tokenized_length") != DEFAULT_MAX_LENGTH
        or protocol.get("tokenization_batch_size")
        != DEFAULT_TOKENIZATION_BATCH_SIZE
        or protocol.get("data_minima") != _standard_minima()
        or thresholds != _standard_thresholds()
        or protocol.get("fit_guard_family_disjoint") is not True
        or protocol.get("fit_guard_prompt_disjoint") is not True
        or protocol.get("guard_may_update_candidate") is not False
        or protocol.get("guard_is_reused_and_nonconfirmatory") is not True
        or protocol.get("guard_result_may_authorize_calibration_b")
        is not False
        or protocol.get("calibration_b_tokenized") is not False
        or protocol.get("calibration_b_evaluated") is not False
        or protocol.get("validation_tokenized") is not False
        or protocol.get("validation_evaluated") is not False
        or protocol.get("test_tokenized") is not False
        or protocol.get("test_evaluated") is not False
        or fit.get("partition") != fit_partition
        or guard.get("partition") != guard_partition
        or fit_stream.get("split") != "calibration_a_fit"
        or guard_stream.get("split") != "calibration_a_guard"
        or not _is_sha256(fit_sha256)
        or not _is_sha256(guard_sha256)
        or fit_sha256 == guard_sha256
        or parent.get("artifact_format_version") != 5
        or parent.get("model_resolved_commit")
        != model.get("resolved_commit")
        or parent.get("layer_index") != DEFAULT_LAYER_INDEX
        or parent.get("layer_id") != protocol.get("layer_id")
    ):
        raise ValueError("dense-supermode frozen protocol binding is invalid")

    if (
        guard.get("fresh_for_this_method") is not False
        or guard.get("nonconfirmatory") is not True
        or guard.get("candidate_mutation_observed") is not False
        or guard.get("dense_diagnostic_passed") is not True
        or guard.get("authorizes_calibration_b") is not False
        or expected_gates.get("dense_diagnostic_passed") is not True
        or expected_gates.get("any_candidate_authorizes_calibration_b")
        is not False
        or status.get("outcome") != "reused_guard_diagnostic_passed"
        or status.get("calibration_a_fit_completed") is not True
        or status.get("reused_calibration_a_guard_evaluated") is not True
        or status.get("reused_guard_nonconfirmatory") is not True
        or status.get(
            "dense_candidate_passed_ordinary_diagnostic_gates"
        )
        is not True
        or status.get("source_free_candidate_artifact_written") is not True
        or status.get("ready_for_fresh_fit_guard_confirmation") is not True
        or status.get("ready_for_calibration_b") is not False
        or status.get("calibration_b_opened") is not False
        or status.get("validation_opened") is not False
        or status.get("test_opened") is not False
        or status.get("scientific_compression_success") is not False
    ):
        raise ValueError(
            "dense-supermode success status or guard policy is invalid"
        )

    logical_accounting = _required_mapping(
        evaluation.get("logical_accounting"),
        label="guard logical accounting",
    )
    resource_candidates = _required_mapping(
        resources.get("candidates"),
        label="candidate resource rows",
    )
    residual_width = executor.width
    candidate_parameters = executor.learned_parameter_count
    removed_parameters = 3 * residual_width * (
        GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
        - GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
    )
    source_parameters = candidate_parameters + removed_parameters
    source_mlp_macs = (
        3 * residual_width * GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
    )
    candidate_mlp_macs = (
        3 * residual_width * GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
    )
    if (
        resources.get("scope") != "single_gemma_layer_4"
        or resources.get("source_width")
        != GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
        or resources.get("runtime_width")
        != GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
        or set(resource_candidates) != set(_CANDIDATE_NAMES)
        or set(logical_accounting) != set(_CANDIDATE_NAMES)
    ):
        raise ValueError("dense-supermode resource scope is invalid")
    for name in _CANDIDATE_NAMES:
        logical = _required_mapping(
            logical_accounting.get(name),
            label=f"{name} logical accounting",
        )
        row = _required_mapping(
            resource_candidates.get(name),
            label=f"{name} resource row",
        )
        valid_tokens = logical.get("valid_tokens")
        candidate_total = logical.get("logical_total_macs")
        if (
            type(valid_tokens) is not int
            or valid_tokens <= 0
            or type(candidate_total) is not int
            or candidate_total <= 0
        ):
            raise ValueError(f"{name} logical accounting is invalid")
        source_total = candidate_total + valid_tokens * (
            source_mlp_macs - candidate_mlp_macs
        )
        expected_row = {
            "source_intermediate_width": (
                GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
            ),
            "retained_intermediate_width": (
                GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
            ),
            "source_layer_parameters": source_parameters,
            "candidate_layer_parameters": candidate_parameters,
            "removed_layer_parameters": removed_parameters,
            "retained_layer_parameter_ratio": (
                candidate_parameters / source_parameters
            ),
            "source_mlp_linear_macs_per_valid_token": source_mlp_macs,
            "candidate_mlp_linear_macs_per_valid_token": candidate_mlp_macs,
            "removed_mlp_linear_macs_per_valid_token": (
                source_mlp_macs - candidate_mlp_macs
            ),
            "guard_source_layer_analytic_macs": source_total,
            "guard_candidate_layer_analytic_macs": candidate_total,
            "guard_analytic_mac_ratio": candidate_total / source_total,
            "latency_or_kernel_speed_claim": False,
        }
        if row != expected_row:
            raise ValueError(
                f"{name} resource accounting does not recompute"
            )


def _load_and_validate_json_sibling(
    source: Path,
    payload: Mapping[str, object],
    *,
    scientific_payload_sha256: str,
    report_sha256: str,
) -> Mapping[str, object]:
    report_path = source.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report_without_digest = json.loads(
        json.dumps(
            _build_json_report(
                payload,
                tensor_file=source.name,
                scientific_payload_sha256=scientific_payload_sha256,
                main_artifact_written=True,
            ),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    expected_report = {
        **expected_report_without_digest,
        "report_sha256": report_sha256,
    }
    if not isinstance(report, Mapping) or report != expected_report:
        raise ValueError("dense-supermode JSON sibling binding is invalid")
    if _report_sha256(expected_report_without_digest) != report_sha256:
        raise ValueError("dense-supermode JSON report digest mismatch")
    return report


def load_gemma3_dense_supermode_dev_artifact(
    path: Path | str,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, object]:
    """Strictly load a successful source-free development candidate."""

    source = Path(path)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "format_version",
        "scientific_status",
        "model",
        "protocol",
        "parent",
        "calibration_a_fit",
        "reused_calibration_a_guard",
        "pipeline",
        "native_pivot_baseline",
        "deletion_baseline",
        "resource_report",
        "contains_source_model_weights",
        "contains_compressed_executor_weights",
        "contains_prompt_text",
        "contains_tokenizer_state",
        "contains_teacher_targets",
        "contains_fisher_taylor_scores",
        "executor",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("dense-supermode development artifact fields invalid")
    if (
        raw["schema"] != GEMMA_DENSE_SUPERMODE_DEV_SCHEMA
        or raw["format_version"]
        != GEMMA_DENSE_SUPERMODE_DEV_FORMAT_VERSION
        or raw["contains_source_model_weights"] is not False
        or raw["contains_compressed_executor_weights"] is not True
        or raw["contains_prompt_text"] is not False
        or raw["contains_tokenizer_state"] is not False
        or raw["contains_teacher_targets"] is not False
        or raw["contains_fisher_taylor_scores"] is not False
        or not _is_sha256(raw["scientific_payload_sha256"])
        or not _is_sha256(raw["report_sha256"])
    ):
        raise ValueError("dense-supermode development artifact header invalid")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    if _payload_sha256(payload) != raw["scientific_payload_sha256"]:
        raise ValueError("dense-supermode scientific payload digest mismatch")
    _validate_tensor_locations(payload)
    executor_state = raw["executor"]
    pipeline = raw["pipeline"]
    if not isinstance(executor_state, Mapping) or not isinstance(
        pipeline,
        Mapping,
    ):
        raise ValueError("dense-supermode executor or pipeline is invalid")
    executor = StructuredTransformerLayerExecutor.from_artifact_state_dict(
        executor_state,
        map_location=map_location,
    )
    candidate = StructuredMLPDenseSupermodeCandidate(
        executor=executor,
        artifact_state=executor_state,
        report=pipeline,
    )
    _validate_success_artifact_bindings(raw, candidate)
    report = _load_and_validate_json_sibling(
        source,
        payload,
        scientific_payload_sha256=raw["scientific_payload_sha256"],
        report_sha256=raw["report_sha256"],
    )
    report_path = source.with_suffix(".json")
    return {
        **{
            key: copy.deepcopy(value)
            for key, value in raw.items()
            if key != "executor"
        },
        "executor": candidate.executor,
        "report": copy.deepcopy(dict(report)),
        "metadata": {
            "tensor_file_sha256": _file_sha256(source),
            "report_file_sha256": _file_sha256(report_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Gemma layer-4 dense 512-to-384 supermodes on v9 A-fit "
            "and evaluate on the reused, nonconfirmatory v9 A-guard."
        )
    )
    parser.add_argument("--parent-artifact", required=True)
    parser.add_argument("--prompt-splits", required=True)
    parser.add_argument("--family-manifest", required=True)
    parser.add_argument("--corpus-audit", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir")
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_gemma3_structured_mlp_dense_supermode_dev_experiment(
        parent_artifact_path=arguments.parent_artifact,
        prompt_splits_path=arguments.prompt_splits,
        family_manifest_path=arguments.family_manifest,
        corpus_audit_path=arguments.corpus_audit,
        revision=arguments.revision,
        output=arguments.output,
        model_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
        layer_index=arguments.layer_index,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    print(
        json.dumps(
            {
                "scientific_status": result["scientific_status"],
                "resource_report": result["resource_report"],
                "metadata": result.get("metadata"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "GEMMA_DENSE_SUPERMODE_DEV_FORMAT_VERSION",
    "GEMMA_DENSE_SUPERMODE_DEV_SCHEMA",
    "GEMMA_DENSE_SUPERMODE_POOL_WIDTH",
    "GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH",
    "GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH",
    "GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH",
    "build_parser",
    "load_gemma3_dense_supermode_dev_artifact",
    "main",
    "run_gemma3_structured_mlp_dense_supermode_dev_experiment",
]


if __name__ == "__main__":
    raise SystemExit(main())
