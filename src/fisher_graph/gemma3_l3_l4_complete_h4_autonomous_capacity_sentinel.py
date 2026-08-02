"""A-only K640 outer-LOFO capacity sentinel for autonomous complete-H4.

V15 is deliberately a single fixed capacity control, not another rank search.
It reuses V14's authenticated fit traces, causal provider ABI, full-vocabulary
shadow ledgers, and write-once publication machinery while keeping V14's
four-recipe protocol immutable.  The residual decoder spans all 640 H4
directions; therefore failure localizes the remaining miss beyond PCA rank
truncation and success authorizes distillation on reusable Calibration A.

Native H4 and reverse-VJP gradients remain fit-only.  At evaluation time the
provider reads only the one-pass prefix and realized pre-correction H4.  The
fresh guard and Calibration B are never opened by this runner.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

from .complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
    autonomous_complete_h4_residual_provider_state_dict,
    load_autonomous_complete_h4_residual_provider,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassExecution,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_PROVIDER_OUTPUT",
    "K640_CAPACITY_RECIPE",
    "build_autonomous_capacity_sentinel_report",
    "run_gemma3_l3_l4_complete_h4_autonomous_capacity_sentinel",
    "build_parser",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-autonomous-residual-k640-capacity-"
    "outer-lofo-a-fit16-dev-v15.json"
)
DEFAULT_PROVIDER_OUTPUT = DEFAULT_OUTPUT.with_suffix(".provider.pt")
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_autonomous_residual_"
    "k640_capacity_outer_lofo_development.v15"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-autonomous-k640-capacity-dev:v15\0"
_V14_REPORT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-autonomous-residual-"
    "outer-lofo-a-fit16-dev-v14.json"
)
_V14_LOGICAL_SHA256 = (
    "01803d62e106de05acafcd000308ae2f861f2be9c6bb879fd2d7f4c9e611f906"
)
_V14_FILE_SHA256 = (
    "fc78f37790898dd3acbded89f6da8a2fa9ee466217d1eba3f123e570859384ad"
)
_V14_CLASSIFICATION = "autonomous_complete_h4_oof_recipes_insufficient"
_EXPECTED_PROMPTS = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_TRAINING_PROMPTS_PER_FOLD = 14
_EXPECTED_FULL_SUPPORT_ROWS = 819
_EXPECTED_MINIMUM_TRAINING_SUPPORT_ROWS = 703
_EXPECTED_OUTER_PROVIDER_FITS = 8
_EXPECTED_FULL_MODEL_FORWARDS = 80
_EXPECTED_BACKWARD_VJP_TRAVERSALS = 16
_EXPECTED_CAUSAL_CHECKS = 16
_EXPECTED_PROVIDER_SCALARS = 1_147_520
_EXPECTED_PROVIDER_MACS = 1_556_480

K640_CAPACITY_RECIPE = _v14.AutonomousResidualRecipe(
    recipe_id="r640_l8_reverse_vjp_capacity",
    rank=640,
    lag_count=8,
    ridge=1.0e-4,
    fit_objective="reverse_vjp_row_weighted_ridge_v1",
)


def _validate_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".json" or ".local-runs" not in output.parts:
        raise ValueError("V15 capacity output must be JSON under .local-runs")
    return output


def _validate_provider_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".pt" or ".local-runs" not in output.parts:
        raise ValueError("V15 capacity provider must be PT under .local-runs")
    return output


def _expected_v14_prerequisite_receipt() -> dict[str, object]:
    return {
        "path": _V14_REPORT.as_posix(),
        "format_version": 14,
        "report_sha256": _V14_LOGICAL_SHA256,
        "file_sha256": _V14_FILE_SHA256,
        "classification": _V14_CLASSIFICATION,
        "passed": False,
        "candidate": None,
        "guard_opened": False,
        "calibration_b_opened": False,
    }


def _validate_v14_prerequisite_report() -> dict[str, object]:
    path = _V14_REPORT
    if not path.is_file() or _v14._file_sha256(path) != _V14_FILE_SHA256:
        raise RuntimeError("V15 prerequisite V14 file drifted")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("V15 prerequisite V14 report is unreadable") from error
    integrity = payload.get("integrity") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") != 14
        or payload.get("report_sha256") != _V14_LOGICAL_SHA256
        or payload.get("classification") != _V14_CLASSIFICATION
        or payload.get("passed") is not False
        or payload.get("candidate") is not None
        or not isinstance(integrity, Mapping)
        or integrity.get("guard_opened") is not False
        or integrity.get("calibration_b_opened") is not False
    ):
        raise RuntimeError("V15 prerequisite V14 semantics drifted")
    return _expected_v14_prerequisite_receipt()


def _validated_prerequisites(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"v14"}:
        raise ValueError("V15 prerequisites must contain exactly V14")
    receipt = value.get("v14")
    expected = _expected_v14_prerequisite_receipt()
    if not isinstance(receipt, Mapping) or dict(receipt) != expected:
        raise ValueError("V15 prerequisite V14 receipt differs")
    return {"v14": expected}


def _sentinel_passed(recipe_row: Mapping[str, object]) -> bool:
    expected = K640_CAPACITY_RECIPE.metadata()
    if (
        any(recipe_row.get(name) != value for name, value in expected.items())
        or recipe_row.get("recipe_sha256")
        != K640_CAPACITY_RECIPE.artifact_sha256
        or recipe_row.get("outer_fold_count") != _EXPECTED_OUTER_PROVIDER_FITS
        or recipe_row.get("every_fold_fit_family_count")
        != _EXPECTED_FAMILIES - 1
    ):
        raise ValueError("V15 recipe row differs from the frozen K640 sentinel")
    resources = recipe_row.get("serving_resources")
    if not isinstance(resources, Mapping):
        raise TypeError("V15 recipe row omitted serving resources")
    if (
        resources.get("prepared_float_scalar_count") != _EXPECTED_PROVIDER_SCALARS
        or resources.get("logical_macs_per_token_upper_bound")
        != _EXPECTED_PROVIDER_MACS
    ):
        raise ValueError("V15 recipe resource geometry differs")
    fidelity = recipe_row.get("fidelity")
    if not isinstance(fidelity, Mapping):
        raise TypeError("V15 recipe row omitted fidelity")
    return all(
        _v14._required_ledger_passed(ledger, fidelity.get(ledger))
        for ledger in _v14._REQUIRED_LEDGERS
    )


def _validate_sentinel_provider(
    provider: AutonomousCompleteH4ResidualProvider,
    *,
    expected_fit_family_count: int,
) -> None:
    if not isinstance(provider, AutonomousCompleteH4ResidualProvider):
        raise TypeError("V15 requires an autonomous complete-H4 provider")
    provider.validate_integrity()
    resources = _v14._provider_resources(provider)
    if (
        provider.rank != K640_CAPACITY_RECIPE.rank
        or provider.state_rank != K640_CAPACITY_RECIPE.rank
        or provider.lag_count != K640_CAPACITY_RECIPE.lag_count
        or provider.state_encoder is not None
        or provider.fit_objective != K640_CAPACITY_RECIPE.fit_objective
        or not math.isclose(
            provider.ridge, K640_CAPACITY_RECIPE.ridge, rel_tol=0.0, abs_tol=0.0
        )
        or provider.vjp_weight_floor != 0.5
        or provider.vjp_weight_ceiling != 2.0
        or len(provider.fit_family_ids) != expected_fit_family_count
        or resources["prepared_float_scalar_count"] != _EXPECTED_PROVIDER_SCALARS
        or resources["logical_macs_per_token_upper_bound"]
        != _EXPECTED_PROVIDER_MACS
    ):
        raise RuntimeError("V15 K640 provider geometry differs")


def _work_accounting(
    *,
    prompt_count: int,
    outer_fold_count: int,
    full_provider_fitted: bool,
) -> dict[str, object]:
    if (
        prompt_count != _EXPECTED_PROMPTS
        or outer_fold_count != _EXPECTED_FAMILIES
        or type(full_provider_fitted) is not bool
    ):
        raise RuntimeError("V15 capacity work geometry differs")
    breakdown = {
        "fit_native_source_forwards": prompt_count,
        "fit_base_vjp_forwards": prompt_count,
        "fit_base_vjp_backward_traversals": prompt_count,
        "evaluation_native_source_forwards": prompt_count,
        "evaluation_base_forwards": prompt_count,
        "evaluation_recipe_forwards": prompt_count,
    }
    total_forwards = sum(
        int(breakdown[name])
        for name in (
            "fit_native_source_forwards",
            "fit_base_vjp_forwards",
            "evaluation_native_source_forwards",
            "evaluation_base_forwards",
            "evaluation_recipe_forwards",
        )
    )
    total_backwards = int(breakdown["fit_base_vjp_backward_traversals"])
    conditional_fit_count = int(full_provider_fitted)
    if (
        total_forwards != _EXPECTED_FULL_MODEL_FORWARDS
        or total_backwards != _EXPECTED_BACKWARD_VJP_TRAVERSALS
        or outer_fold_count != _EXPECTED_OUTER_PROVIDER_FITS
    ):
        raise RuntimeError("V15 capacity exact work count differs")
    return {
        "outer_fold_provider_fit_count": outer_fold_count,
        "expected_outer_fold_provider_fit_count": _EXPECTED_OUTER_PROVIDER_FITS,
        "conditional_full_panel_provider_fit_count": conditional_fit_count,
        "fit_provider_count": outer_fold_count + conditional_fit_count,
        "expected_fit_provider_count": (
            _EXPECTED_OUTER_PROVIDER_FITS + conditional_fit_count
        ),
        "full_model_forward_count": total_forwards,
        "expected_full_model_forward_count": _EXPECTED_FULL_MODEL_FORWARDS,
        "backward_vjp_traversal_count": total_backwards,
        "expected_backward_vjp_traversal_count": (
            _EXPECTED_BACKWARD_VJP_TRAVERSALS
        ),
        "full_model_work_breakdown": {
            **breakdown,
            "total_forwards": total_forwards,
            "total_backward_vjp_traversals": total_backwards,
        },
    }


def build_autonomous_capacity_sentinel_report(
    *,
    artifact_path: Path | str,
    panel: Mapping[str, object],
    bridge_binding_sha256: str,
    folds: Sequence[Mapping[str, object]],
    prerequisites: Mapping[str, object],
    fit_collection: Mapping[str, object],
    base_fidelity: Mapping[str, object],
    recipe_row: Mapping[str, object],
    candidate: Mapping[str, object] | None,
    integrity: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic scalar-only V15 capacity report."""

    validated_prerequisites = _validated_prerequisites(prerequisites)
    passed = _sentinel_passed(recipe_row)
    candidate_id = None if candidate is None else candidate.get("recipe_id")
    if (not passed) != (candidate is None) or (
        candidate is not None
        and candidate_id != K640_CAPACITY_RECIPE.recipe_id
    ):
        raise ValueError("V15 candidate does not match the K640 OOF result")
    if len(folds) != _EXPECTED_FAMILIES:
        raise ValueError("V15 requires exactly eight outer folds")
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": 15,
        "scientific_status": (
            "opened_calibration_a_single_k640_outer_lofo_capacity_sentinel"
        ),
        "artifact": {"path": Path(artifact_path).as_posix()},
        "panel": dict(panel),
        "prerequisites": validated_prerequisites,
        "bridge_binding_sha256": _v14._identifier(
            bridge_binding_sha256, label="bridge binding"
        ),
        "capacity_sentinel": {
            **K640_CAPACITY_RECIPE.metadata(),
            "recipe_sha256": K640_CAPACITY_RECIPE.artifact_sha256,
            "single_fixed_recipe": True,
            "full_h4_output_span": True,
            "rank_selection_performed": False,
            "comparison_to_v14_is_descriptive_only": True,
        },
        "outer_lofo": {
            "folds": [dict(value) for value in folds],
            "decoder_fit_inside_each_training_fold": True,
            "provider_fit_inside_each_training_fold": True,
            "held_family_excluded_from_decoder_and_provider": True,
        },
        "execution_scope": {
            "semantics": (
                "full_vocabulary_full_suffix_behavioral_shadow_from_one_"
                "complete_h4_correction_boundary"
            ),
            "replacement_boundary": "layer.4.output",
            "replacement_boundary_count": 1,
            "full_vocabulary_logits_evaluated": True,
            "untouched_downstream_gemma_layers_final_norm_and_lm_head_executed": True,
            "whole_model_compiled": False,
            "layer_4_computation_deleted": False,
            "source_model_parameters_retained": True,
        },
        "fit_collection": dict(fit_collection),
        "base_fidelity": dict(base_fidelity),
        "recipe": dict(recipe_row),
        "qualification": {
            "rule": (
                "single_k640_recipe_must_pass_every_required_ledger_with_"
                "eight_family_coverage"
            ),
            "required_ledgers": _v14._REQUIRED_LEDGERS,
            "passed": passed,
        },
        "candidate": None if candidate is None else dict(candidate),
        "integrity": dict(integrity),
        "passed": passed,
        "classification": (
            "autonomous_complete_h4_k640_oof_capacity_ceiling_reached"
            if passed
            else "autonomous_complete_h4_k640_oof_capacity_ceiling_insufficient"
        ),
        "success_authorizes": (
            "distill_k640_capacity_ceiling_on_reusable_calibration_a"
            if passed
            else "enlarge_or_nonlinearize_conditional_residual_map_on_reusable_calibration_a"
        ),
        "fresh_guard_authorized": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }
    _v14._scalar_report(report)
    return report


def _publish(
    report: dict[str, object],
    *,
    output: Path,
    provider: AutonomousCompleteH4ResidualProvider | None,
    provider_output: Path,
) -> dict[str, object]:
    candidate = report.get("candidate")
    if (provider is None) != (candidate is None):
        raise ValueError("V15 published provider must match the capacity candidate")
    destinations = (output,) if provider is None else (output, provider_output)
    reservation = _v14._reserve_outputs(destinations)
    report_stage: Path | None = None
    provider_stage: Path | None = None
    try:
        if provider is not None:
            if not isinstance(candidate, Mapping):
                raise TypeError("V15 selected candidate must be a mapping")
            _validate_sentinel_provider(
                provider, expected_fit_family_count=_EXPECTED_FAMILIES
            )
            if candidate.get("provider_artifact_sha256") != provider.artifact_sha256:
                raise ValueError("V15 candidate and provider differ")
            provider_stage = _v14._stage_torch(
                autonomous_complete_h4_residual_provider_state_dict(provider),
                provider_output,
            )
            provider_file_sha256 = _v14._file_sha256(provider_stage)
            restored = load_autonomous_complete_h4_residual_provider(
                provider_stage,
                expected_artifact_sha256=provider.artifact_sha256,
                expected_file_sha256=provider_file_sha256,
                expected_bridge_binding_sha256=provider.bridge_binding_sha256,
            )
            if restored.metadata() != provider.metadata():
                raise RuntimeError("staged V15 provider roundtrip drifted")
            report["candidate"] = {
                **dict(candidate),
                "provider_tensor_artifact": {
                    "path": provider_output.as_posix(),
                    "file_sha256": provider_file_sha256,
                    "file_bytes": provider_stage.stat().st_size,
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "bridge_binding_sha256": provider.bridge_binding_sha256,
                    "write_once": True,
                    "file_mode": "0600",
                    "contains_runtime_provider_tensors_only": True,
                    "contains_native_h4_logits_targets_or_gradients": False,
                },
            }
        _v14._scalar_report(report)
        report["report_sha256"] = _v14._sha256(
            report, domain=_REPORT_DOMAIN
        )
        report_stage = _v14._stage_json(report, output)
        staged = (
            (report_stage,)
            if provider_stage is None
            else (report_stage, provider_stage)
        )
        reservation.publish(staged)
        if provider is not None:
            receipt = report["candidate"]["provider_tensor_artifact"]  # type: ignore[index]
            load_autonomous_complete_h4_residual_provider(
                provider_output,
                expected_artifact_sha256=provider.artifact_sha256,
                expected_file_sha256=str(receipt["file_sha256"]),  # type: ignore[index]
                expected_bridge_binding_sha256=provider.bridge_binding_sha256,
            )
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": _v14._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        for stage in (report_stage, provider_stage):
            if stage is not None:
                stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_autonomous_capacity_sentinel(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    provider_output: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Fit and stream the single fixed K640 A16 outer-LOFO sentinel."""

    destination = _validate_output(output)
    provider_destination = _validate_provider_output(
        destination.with_suffix(".provider.pt")
        if provider_output is None
        else provider_output
    )
    if provider_destination == destination:
        raise ValueError("V15 report and provider outputs must differ")
    if destination.exists():
        raise FileExistsError("refusing to overwrite autonomous V15 report")
    if provider_destination.exists():
        raise FileExistsError("refusing to overwrite autonomous V15 provider")

    prerequisites = {"v14": _validate_v14_prerequisite_report()}
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        records = _v14._collect_fit_records(context)
        bridge_binding = _v14._identifier(
            context.bridge.bridge_binding_sha256, label="bridge binding"
        )
        families = tuple(sorted({record.sequence.family_id for record in records}))
        folds = _v14.build_outer_lofo_splits(families)
        full_support_rows = sum(
            int(record.sequence.support_mask.sum()) for record in records
        )
        training_support_rows: dict[str, int] = {}
        providers: dict[str, AutonomousCompleteH4ResidualProvider] = {}
        for fold in folds:
            held = str(fold["held_family_id"])
            training = tuple(
                record.sequence
                for record in records
                if record.sequence.family_id != held
            )
            row_count = sum(int(value.support_mask.sum()) for value in training)
            training_support_rows[held] = row_count
            if (
                len(training) != _EXPECTED_TRAINING_PROMPTS_PER_FOLD
                or held in {value.family_id for value in training}
                or row_count < K640_CAPACITY_RECIPE.rank
            ):
                raise RuntimeError("V15 outer-LOFO training ownership differs")
            provider = _v14._fit_provider(
                training,
                K640_CAPACITY_RECIPE,
                bridge_binding_sha256=bridge_binding,
            )
            _validate_sentinel_provider(
                provider, expected_fit_family_count=_EXPECTED_FAMILIES - 1
            )
            providers[held] = provider
        minimum_training_rows = min(training_support_rows.values())
        if (
            full_support_rows != _EXPECTED_FULL_SUPPORT_ROWS
            or minimum_training_rows != _EXPECTED_MINIMUM_TRAINING_SUPPORT_ROWS
        ):
            raise RuntimeError("V15 authenticated support geometry differs")

        manifests = _v14._ledger_manifests(records)
        ledger_coverage = _v14._ledger_coverage(manifests)
        accumulators = {
            arm: {
                ledger: SourceAuthoritativeShadowFidelityAccumulator(
                    manifest,
                    gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
                )
                for ledger, manifest in manifests.items()
            }
            for arm in ("base", K640_CAPACITY_RECIPE.recipe_id)
        }
        causal_checks = 0
        for record in records:
            model_inputs, supervised_indices, supervised_targets = _v14._retokenize(
                context.tokenize, record.example
            )
            if (
                gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
                != record.model_inputs_sha256
                or _v14._tensor_sha256(supervised_indices)
                != record.supervised_indices_sha256
                or _v14._tensor_sha256(supervised_targets)
                != record.supervised_targets_sha256
            ):
                raise RuntimeError("V15 shadow retokenization drifted")
            source_logits, native_h4, native_positions, native_valid = (
                _v14._native_boundary(context.adapter, model_inputs)
            )
            base = context.bridge.execute(context.adapter, model_inputs)
            if (
                not isinstance(base, Gemma3L3L4OnePassExecution)
                or not _v14._bitwise_equal(
                    native_positions, base.prefix.logical_positions
                )
                or not _v14._bitwise_equal(
                    native_valid, base.prefix.valid_target_mask
                )
            ):
                raise RuntimeError("V15 shadow base sequence differs")
            _v14._add_shadow_rows(
                accumulators["base"],
                record=record,
                source_logits=source_logits,
                candidate_logits=base.logits,
                supervised_indices=supervised_indices,
                supervised_targets=supervised_targets,
            )
            provider = providers[record.sequence.family_id]
            if record.sequence.family_id in set(provider.fit_family_ids):
                raise RuntimeError("held family leaked into V15 provider")
            support = base.prefix.complete_h4_causal_support_mask()
            candidate = context.bridge.execute(
                context.adapter, model_inputs, h4_head=provider
            )
            if (
                candidate.h4_head_sha256 != provider.artifact_sha256
                or candidate.prefix.artifact_sha256 != base.prefix.artifact_sha256
                or not _v14._bitwise_equal(
                    candidate.candidate_h4[
                        ~support.to(candidate.candidate_h4.device)
                    ],
                    base.candidate_h4[~support.to(base.candidate_h4.device)],
                )
            ):
                raise RuntimeError("V15 provider escaped causal support")
            _v14._add_shadow_rows(
                accumulators[K640_CAPACITY_RECIPE.recipe_id],
                record=record,
                source_logits=source_logits,
                candidate_logits=candidate.logits,
                supervised_indices=supervised_indices,
                supervised_targets=supervised_targets,
            )
            causal_checks += 1
            del model_inputs, source_logits, native_h4, base, candidate

        fidelity = {
            arm: {ledger: value.finalize() for ledger, value in ledgers.items()}
            for arm, ledgers in accumulators.items()
        }
        resource_rows = [_v14._provider_resources(value) for value in providers.values()]
        if len({tuple(sorted(value.items())) for value in resource_rows}) != 1:
            raise RuntimeError("V15 outer-fold provider resources differ")
        resources = resource_rows[0]
        recipe_row = {
            **K640_CAPACITY_RECIPE.metadata(),
            "recipe_sha256": K640_CAPACITY_RECIPE.artifact_sha256,
            "outer_fold_count": len(providers),
            "every_fold_fit_family_count": _EXPECTED_FAMILIES - 1,
            "fold_provider_artifact_sha256s": {
                held: provider.artifact_sha256
                for held, provider in sorted(providers.items())
            },
            "serving_resources": resources,
            "fidelity": fidelity[K640_CAPACITY_RECIPE.recipe_id],
        }
        passed = _sentinel_passed(recipe_row)
        full_provider: AutonomousCompleteH4ResidualProvider | None = None
        if passed:
            full_provider = _v14._fit_provider(
                tuple(record.sequence for record in records),
                K640_CAPACITY_RECIPE,
                bridge_binding_sha256=bridge_binding,
            )
            _validate_sentinel_provider(
                full_provider, expected_fit_family_count=_EXPECTED_FAMILIES
            )
        candidate = (
            None
            if full_provider is None
            else {
                "recipe_id": K640_CAPACITY_RECIPE.recipe_id,
                "provider_artifact_sha256": full_provider.artifact_sha256,
                "provider": full_provider.metadata(),
                "serving_resources": _v14._provider_resources(full_provider),
                "fit_family_count": _EXPECTED_FAMILIES,
                "serving_inputs": (
                    "model_inputs",
                    "one_pass_prefix_source_modes_and_masks",
                    "pre_correction_realized_h4",
                ),
                "native_h4_logits_targets_or_gradients_required_at_runtime": False,
                "capacity_ceiling_only": True,
            }
        )
        context.validate_immutable_inputs()
        work = _work_accounting(
            prompt_count=len(records),
            outer_fold_count=len(folds),
            full_provider_fitted=full_provider is not None,
        )
        integrity = {
            "outer_fold_count": len(folds),
            "ledger_coverage": ledger_coverage,
            **work,
            "full_support_row_count": full_support_rows,
            "expected_full_support_row_count": _EXPECTED_FULL_SUPPORT_ROWS,
            "minimum_outer_training_support_row_count": minimum_training_rows,
            "expected_minimum_outer_training_support_row_count": (
                _EXPECTED_MINIMUM_TRAINING_SUPPORT_ROWS
            ),
            "training_support_row_counts_by_held_family": training_support_rows,
            "causal_off_support_execution_checks": causal_checks,
            "expected_causal_off_support_execution_checks": _EXPECTED_CAUSAL_CHECKS,
            "source_native_data_entered_serving_provider": False,
            "full_provider_fit_was_conditional_on_oof_pass": True,
            "guard_opened": False,
            "calibration_b_opened": False,
        }
        if (
            len(providers) != _EXPECTED_OUTER_PROVIDER_FITS
            or causal_checks != _EXPECTED_CAUSAL_CHECKS
            or work["full_model_forward_count"] != _EXPECTED_FULL_MODEL_FORWARDS
            or work["backward_vjp_traversal_count"]
            != _EXPECTED_BACKWARD_VJP_TRAVERSALS
        ):
            raise RuntimeError("V15 capacity exact execution count differs")
        report = build_autonomous_capacity_sentinel_report(
            artifact_path=destination,
            panel=context.panel_receipt,
            bridge_binding_sha256=bridge_binding,
            folds=folds,
            prerequisites=prerequisites,
            fit_collection={
                "prompt_count": len(records),
                "family_count": len(families),
                "supervised_token_count": sum(
                    value.supervised_token_count for value in records
                ),
                "trace_receipt_sha256s": [value.receipt_sha256 for value in records],
                "native_h4_and_reverse_vjp_fit_only": True,
                "raw_fit_trace_tensor_serialization": False,
                "conditional_runtime_provider_tensor_sidecar": True,
            },
            base_fidelity=fidelity["base"],
            recipe_row=recipe_row,
            candidate=candidate,
            integrity=integrity,
        )
        return _publish(
            report,
            output=destination,
            provider=full_provider,
            provider_output=provider_destination,
        )
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider-output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_autonomous_capacity_sentinel(
        output=args.output,
        provider_output=args.provider_output,
        cache_dir=args.cache_dir,
    )
    print(
        json.dumps(
            {
                "path": report["artifact"]["path"],  # type: ignore[index]
                "report_sha256": report["report_sha256"],
                "classification": report["classification"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
