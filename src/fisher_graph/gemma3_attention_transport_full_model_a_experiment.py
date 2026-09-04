"""Run the A-only, all-18 live attention V/O replacement experiment.

This is a development screen, not a fresh confirmation.  It uses the stable
36/9 split inside the previously exposed A panel.  Only endpoint-macroblock
teacher captures reach fitting.  After the shared dictionary and router are
frozen and independently reconstructed, the middle block is read once for
open-loop Fisher diagnostics and the full native/shared/control trajectories
are run on the nine fixed outer batches.  B and C are never opened.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


DEFAULT_REPORT_OUTPUT = Path(
    "artifacts/gemma3-attention-transport-full-model-cross-layer-a-experiment.json"
)
DEFAULT_TEMP_DIRECTORY = Path(
    "artifacts/.gemma3-attention-transport-full-model-cross-layer-a-temp"
)
REPORT_SCHEMA = (
    "fisher_graph.gemma3_attention_transport."
    "full_model_cross_layer_a_experiment.v1"
)
_REPORT_DOMAIN = b"fisher_graph:full-model-cross-layer-a-experiment:v1\0"
_PINNED_A_CORPUS_FILE_SHA256 = (
    "112023d40793f1ff1c9952cc8b44e7d02b1f24a3473a72c941c4a494c2d6d0bf"
)
_PINNED_A_ROLE_PANEL_BINDING_SHA256 = (
    "bc11c01c66d25daffc96fa69778c2ab34e8d482bcec1de4b9d6bf1b6235b1432"
)
_PINNED_A_BATCH_PLAN_SHA256 = (
    "7983848a72046f598c0e2a439b7fb3e97802f7de95b48af7eddf178a5dd215b1"
)
_PINNED_A_OUTER_SPLIT_SHA256 = (
    "70fe80a40639a9312253b66064afb942683ffc41327d32f29b1a317b483d90c9"
)
_PINNED_A_INNER_SPLIT_SHA256 = (
    "10896b3a92a30d624b7c3647a9d3b3950aa2711f08153aaaeb90118dac66d9a8"
)
_SOURCE_PATHS = (
    "src/fisher_graph/gemma3_attention_transport_full_model_cross_layer.py",
    "src/fisher_graph/gemma3_attention_transport_full_model_partition.py",
    "src/fisher_graph/gemma3_attention_transport_full_model_fit.py",
    "src/fisher_graph/gemma3_attention_transport_full_model_runtime.py",
    "src/fisher_graph/gemma3_attention_transport_full_model_a_experiment.py",
    "src/fisher_graph/gemma3_joint_cross_block_chart_executor.py",
    "src/fisher_graph/gemma3_joint_cross_block_compiler.py",
    "src/fisher_graph/gemma3_attention_transport_compiler.py",
    "src/fisher_graph/gemma3_attention_transport_runtime.py",
    "pyproject.toml",
)
_EXPECTED_BATCH_COUNT = 45
_EXPECTED_INVOCATIONS_PER_BATCH = 18
_ROWS_PER_INVOCATION = 16
_LOGIT_ROW_CHUNK = 8
_PARITY_MAX_NORMALIZED_TERMINAL_RMSE = 1.0e-4
_PARITY_MAX_LOGIT_RMSE = 2.0e-3
_PARITY_MIN_TOP1_AGREEMENT = 0.999


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_REPORT_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256s(repository_root: Path) -> dict[str, str]:
    return {
        relative: _file_sha256(repository_root / relative)
        for relative in _SOURCE_PATHS
    }


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass(slots=True)
class _BehaviorAccumulator:
    token_count: int = 0
    teacher_nll_sum: float = 0.0
    candidate_nll_sum: float = 0.0
    kl_sum: float = 0.0
    top1_match_count: int = 0
    logit_squared_error_sum: float = 0.0
    logit_scalar_count: int = 0

    def update(
        self,
        *,
        teacher_states: Tensor,
        candidate_states: Tensor,
        input_ids: Tensor,
        tied_weight: Tensor,
        logit_softcap: float | None,
        row_chunk: int = _LOGIT_ROW_CHUNK,
    ) -> None:
        if (
            teacher_states.shape != candidate_states.shape
            or teacher_states.ndim != 3
            or teacher_states.shape[:2] != input_ids.shape
            or input_ids.shape[1] < 2
        ):
            raise ValueError("behavioral comparison tensors do not align")
        teacher_rows = teacher_states[:, :-1].reshape(-1, teacher_states.shape[-1])
        candidate_rows = candidate_states[:, :-1].reshape(
            -1, candidate_states.shape[-1]
        )
        targets = input_ids[:, 1:].reshape(-1)
        for start in range(0, int(targets.shape[0]), row_chunk):
            stop = min(start + row_chunk, int(targets.shape[0]))
            teacher_logits = F.linear(teacher_rows[start:stop], tied_weight)
            candidate_logits = F.linear(candidate_rows[start:stop], tied_weight)
            if logit_softcap is not None:
                cap = float(logit_softcap)
                teacher_logits = torch.tanh(teacher_logits / cap) * cap
                candidate_logits = torch.tanh(candidate_logits / cap) * cap
            teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
            candidate_logp = torch.log_softmax(candidate_logits.float(), dim=-1)
            chunk_targets = targets[start:stop]
            self.teacher_nll_sum += float(
                -teacher_logp.gather(1, chunk_targets[:, None]).sum()
            )
            self.candidate_nll_sum += float(
                -candidate_logp.gather(1, chunk_targets[:, None]).sum()
            )
            probabilities = teacher_logp.exp()
            self.kl_sum += float(
                (probabilities * (teacher_logp - candidate_logp)).sum()
            )
            self.top1_match_count += int(
                (
                    teacher_logits.argmax(dim=-1)
                    == candidate_logits.argmax(dim=-1)
                ).sum()
            )
            difference = candidate_logits.float() - teacher_logits.float()
            self.logit_squared_error_sum += float(difference.square().sum())
            self.logit_scalar_count += int(difference.numel())
            self.token_count += stop - start

    def payload(self) -> dict[str, object]:
        if self.token_count <= 0 or self.logit_scalar_count <= 0:
            raise RuntimeError("behavioral accumulator has no supervised rows")
        teacher = self.teacher_nll_sum / self.token_count
        candidate = self.candidate_nll_sum / self.token_count
        raw_kl = self.kl_sum / self.token_count
        return {
            "supervised_token_count": self.token_count,
            "native_nll_per_token": teacher,
            "candidate_nll_per_token": candidate,
            "delta_nll_per_token": candidate - teacher,
            "teacher_to_candidate_kl_per_token": max(0.0, raw_kl),
            "raw_teacher_to_candidate_kl_per_token": raw_kl,
            "top1_agreement": self.top1_match_count / self.token_count,
            "logit_rmse": math.sqrt(
                self.logit_squared_error_sum / self.logit_scalar_count
            ),
            "logit_row_chunk": _LOGIT_ROW_CHUNK,
        }


def _terminal_states(
    executor: object,
    input_ids: Tensor,
    sequence: object,
) -> tuple[Tensor, dict[str, object]]:
    observed = executor.forward_backbone_with_observer(
        input_ids,
        sequence,
        observer=lambda **_values: None,
    )
    execution = observed.execution
    states = execution.normalized_output.detach().clone()
    attention_receipts = tuple(
        receipt
        for receipt in execution.operator_receipts
        if receipt.event_kind == "attention_mutation"
    )
    payload = {
        "carrier_version": int(execution.carrier_version),
        "transport_edge_count": int(execution.transport_edge_count),
        "attention_receipt_count": len(attention_receipts),
        "attention_generator_kinds": sorted(
            {
                str(receipt.generator_kind)
                for receipt in attention_receipts
            }
        ),
        "source_projection_call_count": sum(
            int(receipt.source_projection_call_count or 0)
            for receipt in attention_receipts
        ),
        "source_fallback_call_count": sum(
            int(receipt.source_fallback_call_count or 0)
            for receipt in attention_receipts
        ),
    }
    del observed
    gc.collect()
    return states, payload


def _independent_rank8_ceiling(
    captures: Sequence[object],
    *,
    euclidean_floor: float,
) -> dict[str, object]:
    """Post-hoc same-row ceiling with one independent rank-8 map per layer."""

    from fisher_graph.gemma3_attention_transport_compiler import (
        samplewise_joint_fisher_loss,
    )

    groups: dict[str, list[object]] = {}
    for capture in captures:
        groups.setdefault(str(capture.invocation_id), []).append(capture)
    predictions = []
    targets = []
    gradients = []
    per_invocation = []
    hidden_width: int | None = None
    for invocation_id in sorted(groups, key=lambda value: int(value.split(".")[1])):
        group = groups[invocation_id]
        contexts = torch.cat(tuple(value.head_contexts for value in group)).sum(dim=1)
        target = torch.cat(tuple(value.valid_operator_outputs for value in group))
        gradient = torch.cat(tuple(value.valid_score_gradients for value in group))
        hidden_width = int(contexts.shape[1])
        gram = contexts @ contexts.T
        ridge = 1.0e-6 * max(1.0, float(gram.diagonal().mean()))
        dense = contexts.T @ torch.linalg.solve(
            gram + ridge * torch.eye(gram.shape[0], dtype=torch.float64),
            target,
        )
        left, singular, right_h = torch.linalg.svd(dense, full_matrices=False)
        rank = min(8, int(singular.shape[0]))
        reduced = (
            left[:, :rank] * singular[:rank].unsqueeze(0)
        ) @ right_h[:rank]
        prediction = contexts @ reduced
        fitted = float(
            samplewise_joint_fisher_loss(
                prediction,
                target,
                gradient,
                euclidean_floor=euclidean_floor,
            )
        )
        zero = float(
            samplewise_joint_fisher_loss(
                torch.zeros_like(target),
                target,
                gradient,
                euclidean_floor=euclidean_floor,
            )
        )
        predictions.append(prediction)
        targets.append(target)
        gradients.append(gradient)
        per_invocation.append((invocation_id, fitted / zero))
    assert hidden_width is not None
    joined_prediction = torch.cat(predictions)
    joined_target = torch.cat(targets)
    joined_gradient = torch.cat(gradients)
    fitted = float(
        samplewise_joint_fisher_loss(
            joined_prediction,
            joined_target,
            joined_gradient,
            euclidean_floor=euclidean_floor,
        )
    )
    zero = float(
        samplewise_joint_fisher_loss(
            torch.zeros_like(joined_target),
            joined_target,
            joined_gradient,
            euclidean_floor=euclidean_floor,
        )
    )
    projected_all_layer_parameters = 18 * 2 * hidden_width * 8
    return {
        "scope": "posthoc_same_rows_open_loop_capacity_ceiling_not_candidate",
        "trajectory_executed": False,
        "rank_per_layer": 8,
        "context_reduction": "sum_query_head_contexts",
        "held_layer_parameter_count": len(groups) * 2 * hidden_width * 8,
        "projected_all_18_parameter_count": projected_all_layer_parameters,
        "parameter_cap": 186_049,
        "within_parameter_cap": projected_all_layer_parameters <= 186_049,
        "relative_fisher_loss": fitted / zero,
        "per_invocation_relative_fisher_loss": per_invocation,
        "same_rows_used_for_fit_and_evaluation": True,
        "confirmation_claimed": False,
    }


def _atomic_publish(path: Path, report: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = _canonical_json_bytes(report) + b"\n"
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def run_full_model_cross_layer_A_experiment(
    *,
    binding: object,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    cache_dir: Path | str | None = None,
    progress: Any | None = None,
) -> dict[str, object]:
    """Fit, freeze, and evaluate the fixed A-only full-model experiment."""

    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    output = Path(report_path)
    if output.suffix != ".json":
        raise ValueError("full-model report must use .json")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        FIT_ROLE,
        _capture_batch_plan,
        role_panel_binding_sha256,
        static_preflight,
        validate_runtime_qualification,
    )

    preflight = static_preflight(binding)
    preflight.validate_receipt()
    repository_root = Path(preflight.binding.repository_root)
    source_hashes = _source_sha256s(repository_root)

    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )
    from fisher_graph.gemma3_chart_native_neighbor_runtime import (
        _prepare_sequence,
    )
    from fisher_graph.gemma3_attention_transport_full_model_cross_layer import (
        check_full_model_cross_layer_attention_transport_state_dict,
    )
    from fisher_graph.gemma3_attention_transport_full_model_fit import (
        FullModelCrossLayerFitConfig,
        evaluate_full_model_cross_layer_fisher,
        fit_full_model_cross_layer_attention_transport,
    )
    from fisher_graph.gemma3_attention_transport_full_model_partition import (
        A_OUTER_DEV_BATCH_ORDINALS,
        A_OUTER_FIT_BATCH_ORDINALS,
        ENDPOINT_TRAIN_INVOCATION_IDS,
        MIDDLE_HOLDOUT_INVOCATION_IDS,
        NATIVE_VALUE_OUTPUT_PARAMETERS,
        FullModelFirewallReceipt,
        FullModelGateMeasurements,
        build_attention_transport_full_model_protocol,
        build_full_model_stable_A_split,
        evaluate_full_model_gate_decision,
    )
    from fisher_graph.gemma3_attention_transport_full_model_runtime import (
        FrozenCrossLayerTransportRuntime,
        attention_wrapper_tripwire_payload,
        build_native_head_composite_parity_executor,
        build_native_qk_shared_vo_executor,
    )
    from fisher_graph.gemma3_attention_transport_global_gated_residual_spectrum_partition import (
        partition_stable_A_capture_records,
    )
    from fisher_graph.gemma3_attention_transport_runtime import (
        build_pinned_gemma3_attention_transport_capture_runtime,
        sample_attention_transport_captured_invocation,
    )
    role_manifest = preflight.fresh_manifest["roles"][FIT_ROLE]
    if role_manifest.get("file_sha256") != _PINNED_A_CORPUS_FILE_SHA256:
        raise RuntimeError("pinned A corpus identity drifted")
    panel_binding = role_panel_binding_sha256(preflight.fresh_manifest, FIT_ROLE)
    if panel_binding != _PINNED_A_ROLE_PANEL_BINDING_SHA256:
        raise RuntimeError("pinned A panel binding drifted")
    stable_split = build_full_model_stable_A_split(
        role_panel_binding_sha256=panel_binding,
        ordered_members=role_manifest["ordered_members"],
    )
    protocol = build_attention_transport_full_model_protocol(stable_split)
    if (
        stable_split.catalog.batch_plan_artifact_sha256
        != _PINNED_A_BATCH_PLAN_SHA256
        or stable_split.outer_split_artifact_sha256
        != _PINNED_A_OUTER_SPLIT_SHA256
        or stable_split.inner_split_artifact_sha256
        != _PINNED_A_INNER_SPLIT_SHA256
    ):
        raise RuntimeError("pinned stable A split drifted")

    panel_source = SequentialPanelSource(
        repository_root=repository_root,
        manifest=preflight.fresh_manifest,
        receipt=preflight.fresh_receipt,
        preflight=preflight.opaque_panel_receipt,
    )
    sink_records: list[object] = []
    sink_active = False

    def sink(captured: object) -> None:
        if not sink_active:
            raise RuntimeError("full-model capture sink called outside A capture")
        sampled = sample_attention_transport_captured_invocation(
            captured,
            maximum_rows=_ROWS_PER_INVOCATION,
        )
        sink_records.append(replace(sampled, role="A"))

    runtime = None
    try:
        runtime = build_pinned_gemma3_attention_transport_capture_runtime(
            preflight,
            sink=sink,
            cache_dir=cache_dir,
        )
        qualification = runtime.runtime_qualification_receipt()
        validate_runtime_qualification(preflight=preflight, receipt=qualification)
        corpus = panel_source.open_role(FIT_ROLE)
        plan = tuple(_capture_batch_plan(corpus))
        if len(plan) != _EXPECTED_BATCH_COUNT:
            raise RuntimeError("A batch plan count drifted")
        dense_parity_receipt = runtime.run_dense_lexical_parity_smoke(
            preflight=preflight,
            raw_example=corpus["examples"][0],
            role_manifest=role_manifest,
        )
        sink_active = True
        for index, batch in enumerate(plan):
            before = len(sink_records)
            captured = runtime.capture_batch(
                preflight=preflight,
                role=FIT_ROLE,
                raw_examples=batch,
                role_manifest=role_manifest,
            )
            emitted = sink_records[before:]
            if (
                len(emitted) != _EXPECTED_INVOCATIONS_PER_BATCH
                or any(
                    value.base_capture_receipt_sha256
                    != captured.execution_receipt_sha256
                    for value in emitted
                )
            ):
                raise RuntimeError("A transport capture ledger drifted")
            if progress is not None:
                progress(
                    {
                        "phase": "A_capture",
                        "completed": index + 1,
                        "total": len(plan),
                    }
                )
        sink_active = False
        if len(sink_records) != _EXPECTED_BATCH_COUNT * _EXPECTED_INVOCATIONS_PER_BATCH:
            raise RuntimeError("A transport capture inventory drifted")
        partition = partition_stable_A_capture_records(
            sink_records,
            catalog=stable_split.catalog,
        )
        fit_captures = tuple(
            record.capture
            for record in partition.outer_fit_records
            if record.capture.invocation_id in ENDPOINT_TRAIN_INVOCATION_IDS
        )
        held_captures = tuple(
            record.capture
            for record in partition.outer_dev_records
            if record.capture.invocation_id in MIDDLE_HOLDOUT_INVOCATION_IDS
        )
        if len(fit_captures) != 36 * 12 or len(held_captures) != 9 * 6:
            raise RuntimeError("full-model structural split row inventory drifted")
        del partition
        sink_records.clear()
        gc.collect()

        fit_config = FullModelCrossLayerFitConfig()
        fit_result = fit_full_model_cross_layer_attention_transport(
            fit_captures,
            config=fit_config,
            progress=progress,
        )
        candidate_state = fit_result.artifact.state_dict()
        candidate = check_full_model_cross_layer_attention_transport_state_dict(
            candidate_state
        )
        if candidate.artifact_sha256 != fit_result.artifact.artifact_sha256:
            raise RuntimeError("independent candidate reconstruction drifted")
        candidate_sha = candidate.artifact_sha256
        accounting = candidate.parameter_accounting(
            native_v_o_parameter_count=NATIVE_VALUE_OUTPUT_PARAMETERS
        )
        if (
            accounting.compiled_parameters > 186_049
            or accounting.free_layer_id_parameters != 0
            or accounting.parameter_compression_eligible is not True
        ):
            raise RuntimeError("candidate parameter budget did not close")

        exact_executor = runtime._executor
        tied_weight = runtime._tied_weight.tensor
        catalogs = runtime._transport_head_operator_catalogs
        parity_ordinal = A_OUTER_FIT_BATCH_ORDINALS[0]
        _, parity_ids = runtime._members_and_tokens(
            role=FIT_ROLE,
            raw_examples=plan[parity_ordinal],
            role_manifest=role_manifest,
        )
        parity_sequence = _prepare_sequence(runtime._adapter, parity_ids)
        native_parity_states, native_parity_receipt = _terminal_states(
            exact_executor,
            parity_ids,
            parity_sequence,
        )
        parity_executor = build_native_head_composite_parity_executor(
            exact_executor,
            head_operators_by_invocation={
                invocation_id: catalog.head_operators
                for invocation_id, catalog in catalogs.items()
            },
        )
        composite_states, composite_receipt = _terminal_states(
            parity_executor,
            parity_ids,
            parity_sequence,
        )
        parity_accumulator = _BehaviorAccumulator()
        parity_accumulator.update(
            teacher_states=native_parity_states,
            candidate_states=composite_states,
            input_ids=parity_ids,
            tied_weight=tied_weight,
            logit_softcap=exact_executor.final_logit_softcap,
        )
        parity_metrics = parity_accumulator.payload()
        terminal_error = composite_states.float() - native_parity_states.float()
        normalized_terminal_rmse = math.sqrt(
            float(terminal_error.square().mean())
            / max(1.0e-30, float(native_parity_states.float().square().mean()))
        )
        parity_tripwire = attention_wrapper_tripwire_payload(parity_executor)
        parity_passed = (
            normalized_terminal_rmse <= _PARITY_MAX_NORMALIZED_TERMINAL_RMSE
            and float(parity_metrics["logit_rmse"]) <= _PARITY_MAX_LOGIT_RMSE
            and float(parity_metrics["top1_agreement"]) >= _PARITY_MIN_TOP1_AGREEMENT
            and parity_tripwire["native_v_o_calls_zero"] is True
        )
        seam_parity = {
            "passed": parity_passed,
            "normalized_terminal_rmse": normalized_terminal_rmse,
            "maximum_absolute_terminal_error": float(terminal_error.abs().max()),
            "behavior": parity_metrics,
            "thresholds": {
                "normalized_terminal_rmse_max": _PARITY_MAX_NORMALIZED_TERMINAL_RMSE,
                "logit_rmse_max": _PARITY_MAX_LOGIT_RMSE,
                "top1_agreement_min": _PARITY_MIN_TOP1_AGREEMENT,
            },
            "native_receipt": native_parity_receipt,
            "composite_receipt": composite_receipt,
            "tripwire": parity_tripwire,
        }
        del parity_executor, composite_states, native_parity_states
        gc.collect()
        if not parity_passed:
            raise RuntimeError("exact head-composite seam parity failed")
        if progress is not None:
            progress({"phase": "seam_parity", "completed": 1, "total": 1, **seam_parity})

        held_evaluation = evaluate_full_model_cross_layer_fisher(
            candidate,
            held_captures,
            euclidean_floor=fit_config.euclidean_floor,
        )
        held_per_invocation = dict(
            held_evaluation.per_invocation_deployable_relative_fisher_loss
        )
        independent_ceiling = _independent_rank8_ceiling(
            held_captures,
            euclidean_floor=fit_config.euclidean_floor,
        )

        evaluation_batches = []
        for ordinal in A_OUTER_DEV_BATCH_ORDINALS:
            _, input_ids = runtime._members_and_tokens(
                role=FIT_ROLE,
                raw_examples=plan[ordinal],
                role_manifest=role_manifest,
            )
            evaluation_batches.append(
                (ordinal, input_ids, _prepare_sequence(runtime._adapter, input_ids))
            )
        native_states = []
        native_receipts = []
        for index, (ordinal, input_ids, sequence) in enumerate(evaluation_batches):
            states, receipt = _terminal_states(exact_executor, input_ids, sequence)
            native_states.append(states)
            native_receipts.append(receipt)
            if progress is not None:
                progress(
                    {
                        "phase": "full_model_native",
                        "completed": index + 1,
                        "total": len(evaluation_batches),
                        "batch_ordinal": ordinal,
                    }
                )

        shared_runtime = FrozenCrossLayerTransportRuntime(
            candidate,
            dtype=torch.float32,
            device="cpu",
        )
        arm_payloads: dict[str, dict[str, object]] = {}
        mode_to_arm = {
            "normal": "shared_zero_shot_top2",
            "zero_held": "middle_block_zeroed_control",
            "permuted_held": "middle_block_structural_feature_permutation_control",
        }
        for mode, arm_id in mode_to_arm.items():
            executor = build_native_qk_shared_vo_executor(
                exact_executor,
                shared_runtime=shared_runtime,
                control_mode=mode,
            )
            accumulator = _BehaviorAccumulator()
            receipts = []
            for index, ((ordinal, input_ids, sequence), teacher_states) in enumerate(
                zip(evaluation_batches, native_states, strict=True)
            ):
                candidate_states, receipt = _terminal_states(
                    executor,
                    input_ids,
                    sequence,
                )
                accumulator.update(
                    teacher_states=teacher_states,
                    candidate_states=candidate_states,
                    input_ids=input_ids,
                    tied_weight=tied_weight,
                    logit_softcap=exact_executor.final_logit_softcap,
                )
                receipts.append(receipt)
                del candidate_states
                if progress is not None:
                    progress(
                        {
                            "phase": f"full_model_{mode}",
                            "completed": index + 1,
                            "total": len(evaluation_batches),
                            "batch_ordinal": ordinal,
                        }
                    )
            tripwire = attention_wrapper_tripwire_payload(executor)
            if (
                tripwire["native_v_o_calls_zero"] is not True
                or tripwire["native_v_o_state_names_absent"] is not True
                or any(receipt["carrier_version"] != 36 for receipt in receipts)
                or any(receipt["transport_edge_count"] != 0 for receipt in receipts)
                or any(
                    receipt["attention_generator_kinds"]
                    != ["native_qk_shared_vo"]
                    for receipt in receipts
                )
            ):
                raise RuntimeError(f"full-model {mode} runtime tripwire failed")
            arm_payloads[arm_id] = {
                "behavior": accumulator.payload(),
                "runtime_tripwire": tripwire,
                "execution_receipts": receipts,
            }
            del executor
            gc.collect()

        normal_behavior = arm_payloads["shared_zero_shot_top2"]["behavior"]
        zero_behavior = arm_payloads["middle_block_zeroed_control"]["behavior"]
        permuted_behavior = arm_payloads[
            "middle_block_structural_feature_permutation_control"
        ]["behavior"]
        assert isinstance(normal_behavior, Mapping)
        assert isinstance(zero_behavior, Mapping)
        assert isinstance(permuted_behavior, Mapping)
        normal_delta = float(normal_behavior["delta_nll_per_token"])
        zero_delta = float(zero_behavior["delta_nll_per_token"])
        permuted_delta = float(permuted_behavior["delta_nll_per_token"])
        measurements = FullModelGateMeasurements(
            held_block_relative_fisher_loss=(
                held_evaluation.deployable_relative_fisher_loss
            ),
            mutation_23_relative_fisher_loss=held_per_invocation["mutation.23"],
            held_block_oracle_relative_fisher_loss=(
                held_evaluation.oracle_relative_fisher_loss
            ),
            delta_nll_per_token=normal_delta,
            kl_per_token=float(
                normal_behavior["teacher_to_candidate_kl_per_token"]
            ),
            top1_agreement=float(normal_behavior["top1_agreement"]),
            real_gain_over_zeroed_control=zero_delta - normal_delta,
            permuted_control_gain=zero_delta - permuted_delta,
        )
        firewall = FullModelFirewallReceipt(
            protocol_sha256=protocol.protocol_sha256,
            candidate_artifact_sha256=candidate_sha,
            candidate_frozen_before_outer_A_dev=True,
            heldout_teacher_in_candidate_lineage=False,
            heldout_teacher_reads_before_freeze=0,
            outer_A_dev_reads_before_freeze=0,
            outer_A_dev_opened_after_freeze=True,
            B_guard_opened_before_A_pass=False,
            B_guard_reads_before_A_pass=0,
            C_assessment_opened_before_B_pass=False,
            C_assessment_reads_before_B_pass=0,
            refit_after_sealed_read=False,
            reselection_after_sealed_read=False,
            oracle_used_in_full_model_trajectory=False,
            native_internal_states_used_in_candidate_trajectory=False,
            free_layer_parameter_count=0,
            replaced_value_output_site_count=18,
            native_value_output_parameters_in_executor=0,
            teacher_payload_values_in_executor=0,
            control_permutation_frozen_before_outer_A_dev=True,
            candidate_published_before_decision=False,
        )
        decision = evaluate_full_model_gate_decision(
            protocol=protocol,
            firewall=firewall,
            measurements=measurements,
        )

        counters = runtime.counters.payload()
        runtime.close()
        runtime = None
        if panel_source.opened_roles != (FIT_ROLE,):
            raise RuntimeError("full-model experiment opened a role beyond A")
        if _source_sha256s(repository_root) != source_hashes:
            raise RuntimeError("full-model implementation changed during execution")

        report_without_hash: dict[str, object] = {
            "schema": REPORT_SCHEMA,
            "format_version": 1,
            "scientific_status": {
                "development_only": True,
                "A_previously_exposed": True,
                "B_opened_by_this_run": False,
                "C_ever_opened": False,
                "confirmation_claimed": False,
                "seam_parity_passed": True,
                "A_screen_passed": decision.passed,
            },
            "model": {
                "model_id": str(qualification.model_id),
                "model_revision": str(qualification.model_revision),
                "hidden_width": int(qualification.hidden_width),
                "runtime_qualification_receipt_sha256": (
                    qualification.qualification_receipt_sha256
                ),
                "dense_lexical_parity_receipt_sha256": dense_parity_receipt,
            },
            "protocol": protocol.state_dict(),
            "stable_A_split": stable_split.state_dict(),
            "fit": {
                "fit_capture_count": fit_result.fit_capture_count,
                "fit_row_count": fit_result.fit_row_count,
                "endpoint_invocation_ids": fit_result.endpoint_invocation_ids,
                "held_invocation_ids_absent_from_fit": all(
                    value not in fit_result.endpoint_invocation_ids
                    for value in MIDDLE_HOLDOUT_INVOCATION_IDS
                ),
                "base_bank_artifact_sha256": (
                    fit_result.base_bank_artifact_sha256
                ),
                "structural_round_relative_fisher_losses": (
                    fit_result.structural_round_relative_fisher_losses
                ),
                "router_training_losses": fit_result.router_training_losses,
                "endpoint_evaluation": {
                    "row_count": fit_result.endpoint_evaluation.row_count,
                    "deployable_relative_fisher_loss": (
                        fit_result.endpoint_evaluation.deployable_relative_fisher_loss
                    ),
                    "oracle_relative_fisher_loss": (
                        fit_result.endpoint_evaluation.oracle_relative_fisher_loss
                    ),
                },
                "config": {
                    name: getattr(fit_config, name)
                    for name in fit_config.__dataclass_fields__
                },
            },
            "candidate": {
                "artifact_sha256": candidate_sha,
                "independent_state_reconstruction_passed": True,
                "source_artifact_sha256": shared_runtime.source_artifact_sha256,
                "executor_runtime_artifact_sha256": shared_runtime.artifact_sha256,
                "parameter_accounting": {
                    name: getattr(accounting, name)
                    for name in accounting.__dataclass_fields__
                },
                "all_18_live_replacements": True,
                "native_qk_retained": True,
                "native_v_o_retained": False,
                "native_mlps_lexical_terminal_retained": True,
                "free_layer_id_parameters": 0,
            },
            "seam_parity": seam_parity,
            "held_middle_open_loop_fisher": {
                "row_count": held_evaluation.row_count,
                "deployable_relative_fisher_loss": (
                    held_evaluation.deployable_relative_fisher_loss
                ),
                "oracle_relative_fisher_loss": (
                    held_evaluation.oracle_relative_fisher_loss
                ),
                "router_oracle_relative_fisher_loss_gap": (
                    held_evaluation.deployable_relative_fisher_loss
                    - held_evaluation.oracle_relative_fisher_loss
                ),
                "per_invocation_deployable_relative_fisher_loss": (
                    held_evaluation.per_invocation_deployable_relative_fisher_loss
                ),
                "per_invocation_oracle_relative_fisher_loss": (
                    held_evaluation.per_invocation_oracle_relative_fisher_loss
                ),
            },
            "full_model_outer_A_dev": {
                "batch_ordinals": A_OUTER_DEV_BATCH_ORDINALS,
                "native_execution_receipts": native_receipts,
                "arms": arm_payloads,
            },
            "budget_capped_independent_per_layer_ceiling": independent_ceiling,
            "gate_decision": decision.state_dict(),
            "runtime": {
                "capture_counters": counters,
                "opened_roles": panel_source.opened_roles,
                "B_opened": False,
                "C_opened": False,
            },
            "implementation_source_sha256s": source_hashes,
        }
        report = {
            **report_without_hash,
            "report_sha256": _report_sha256(report_without_hash),
        }
        _atomic_publish(output, report)
        return report
    finally:
        sink_active = False
        if runtime is not None:
            runtime.close()


def check_full_model_cross_layer_A_experiment(
    report_path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    """Independently authenticate the published JSON and current sources."""

    path = Path(report_path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("full-model report must be a JSON object")
    observed_hash = value.pop("report_sha256", None)
    expected_hash = _report_sha256(value)
    if observed_hash != expected_hash:
        raise ValueError("full-model report SHA-256 drifted")
    if value.get("schema") != REPORT_SCHEMA or value.get("format_version") != 1:
        raise ValueError("full-model report schema drifted")
    scientific = value.get("scientific_status")
    if (
        not isinstance(scientific, Mapping)
        or scientific.get("B_opened_by_this_run") is not False
        or scientific.get("C_ever_opened") is not False
        or scientific.get("confirmation_claimed") is not False
        or scientific.get("seam_parity_passed") is not True
    ):
        raise ValueError("full-model scientific status drifted")
    decision = value.get("gate_decision")
    if not isinstance(decision, Mapping) or not isinstance(
        decision.get("failed_gates"), list
    ):
        raise ValueError("full-model gate decision is invalid")
    if repository_root is not None:
        sources = value.get("implementation_source_sha256s")
        if sources != _source_sha256s(Path(repository_root)):
            raise ValueError("full-model implementation source binding drifted")
    return {
        "schema": REPORT_SCHEMA,
        "report_sha256": observed_hash,
        "candidate_artifact_sha256": value["candidate"]["artifact_sha256"],
        "seam_parity_passed": True,
        "A_screen_passed": bool(decision["passed"]),
        "failed_gates": decision["failed_gates"],
        "B_opened": False,
        "C_opened": False,
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or check the A-only full-model cross-layer V/O experiment"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.repository_root.resolve()
    report_path = _resolve(root, arguments.report)
    if arguments.check:
        print(
            json.dumps(
                check_full_model_cross_layer_A_experiment(
                    report_path,
                    repository_root=root,
                ),
                sort_keys=True,
            )
        )
        return 0

    temporary = _resolve(
        root,
        arguments.temp_directory
        if arguments.temp_directory is not None
        else DEFAULT_TEMP_DIRECTORY,
    )
    report_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary.mkdir(mode=0o700, parents=True, exist_ok=True)
    from fisher_graph.gemma3_chart_native_neighbor_experiment import (
        build_default_execution_binding,
    )

    binding = build_default_execution_binding(
        repository_root=root,
        output_directory=report_path.parent,
        temp_directory=temporary,
    )

    def emit(value: Mapping[str, object]) -> None:
        print(json.dumps(dict(value), sort_keys=True), flush=True)

    report = run_full_model_cross_layer_A_experiment(
        binding=binding,
        report_path=report_path,
        cache_dir=arguments.cache_dir,
        progress=emit,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REPORT_OUTPUT",
    "DEFAULT_TEMP_DIRECTORY",
    "REPORT_SCHEMA",
    "check_full_model_cross_layer_A_experiment",
    "run_full_model_cross_layer_A_experiment",
]
