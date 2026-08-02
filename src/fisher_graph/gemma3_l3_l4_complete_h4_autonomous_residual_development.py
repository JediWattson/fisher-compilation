"""A-only outer-LOFO development runner for autonomous complete-H4 heads.

The runner deliberately separates fitting from serving.  Native H4 and the
reverse VJP of supervised NLL are admitted only while constructing private
``AutonomousCompleteH4TrainingSequence`` objects.  Every measured candidate
is subsequently executed from model inputs through the one-pass bridge with a
frozen provider; native H4, source logits, targets, and gradients are not
arguments to the provider.

The opened corpus is the authenticated A16/eight-family panel.  Each recipe is
measured out of family: the decoder and provider for a held family are derived
from the other seven families only.  A full-panel provider is fitted only when
one cross-fitted recipe passes every frozen source-authoritative shadow gate.
This rung never opens a guard or Calibration B.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .complete_h4_autonomous_residual import (
    AutonomousCompleteH4ResidualProvider,
    AutonomousCompleteH4TrainingSequence,
    autonomous_complete_h4_residual_provider_state_dict,
    fit_autonomous_complete_h4_output_decoder,
    fit_autonomous_complete_h4_residual,
    load_autonomous_complete_h4_residual_provider,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import (
    _mean_supervised_nll,
    _native_boundary,
    _retokenize,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_conditional_spectral_shadow_evaluation import (
    _prompt_sha256,
    _scalar_report,
    _select_sequence_rows,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_candidate import (
    _file_sha256,
    _reserve_outputs,
    _stage_json,
    _stage_torch,
)
from .gemma3_l3_l4_graph_wavelet_signed_g8_shadow_basis_comparison import (
    _canonical_json_bytes,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassExecution,
    Gemma3L3L4OnePassPrefix,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "AutonomousResidualRecipe",
    "DEFAULT_OUTPUT",
    "DEFAULT_PROVIDER_OUTPUT",
    "DEFAULT_RECIPES",
    "build_autonomous_residual_development_report",
    "build_outer_lofo_splits",
    "choose_passing_recipe",
    "run_gemma3_l3_l4_complete_h4_autonomous_residual_development",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-autonomous-residual-"
    "outer-lofo-a-fit16-dev-v14.json"
)
DEFAULT_PROVIDER_OUTPUT = DEFAULT_OUTPUT.with_suffix(".provider.pt")
_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_autonomous_residual_"
    "outer_lofo_development.v14"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-autonomous-residual-dev:v14\0"
_RECIPE_DOMAIN = b"fisher-graph:complete-h4-autonomous-residual-recipe:v14\0"
_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-autonomous-residual-trace:v14\0"
_EXPECTED_PROMPTS = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_OUTER_PROVIDER_FITS = 32
_EXPECTED_FIT_SOURCE_FORWARDS = 16
_EXPECTED_FIT_VJP_FORWARDS = 16
_EXPECTED_FIT_VJP_BACKWARDS = 16
_EXPECTED_EVAL_SOURCE_FORWARDS = 16
_EXPECTED_EVAL_BASE_FORWARDS = 16
_EXPECTED_EVAL_RECIPE_FORWARDS = 64
_EXPECTED_FULL_MODEL_FORWARDS = 128
_REQUIRED_LEDGERS = ("ordinary", "complete_h4_support", "graph_core")
_ALL_LEDGERS = (*_REQUIRED_LEDGERS, "causal_tail")
_FIT_OBJECTIVES = frozenset(
    {"hidden_residual_ridge", "reverse_vjp_row_weighted_ridge_v1"}
)


def _sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty canonical string")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("runtime hash input must be a Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"fisher-graph:complete-h4-autonomous-runtime-tensor:v14\0")
    digest.update(
        _canonical_json_bytes(
            {
                "shape": tuple(int(width) for width in canonical.shape),
                "dtype": str(canonical.dtype),
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


@dataclass(frozen=True, slots=True)
class AutonomousResidualRecipe:
    """One prospectively frozen, small autonomous-head fit recipe."""

    recipe_id: str
    rank: int
    lag_count: int
    ridge: float
    fit_objective: str

    def __post_init__(self) -> None:
        _identifier(self.recipe_id, label="recipe_id")
        if type(self.rank) is not int or not 0 < self.rank <= 640:
            raise ValueError("recipe rank must lie in [1, 640]")
        if type(self.lag_count) is not int or self.lag_count <= 0:
            raise ValueError("recipe lag_count must be positive")
        if (
            isinstance(self.ridge, bool)
            or not isinstance(self.ridge, (int, float))
            or not math.isfinite(float(self.ridge))
            or float(self.ridge) <= 0.0
        ):
            raise ValueError("recipe ridge must be finite and positive")
        if self.fit_objective not in _FIT_OBJECTIVES:
            raise ValueError("recipe fit objective differs")

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.metadata(), domain=_RECIPE_DOMAIN)

    def metadata(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "rank": self.rank,
            "lag_count": self.lag_count,
            "ridge": float(self.ridge),
            "fit_objective": self.fit_objective,
            "state_encoder": "same_training_only_output_decoder",
            "decoder_fit": "canonical_uncentered_residual_pca_training_only",
        }


# Keep the bank small, but include the capacity range that previously separated
# an insufficient K64 oracle from the passing K256/K320 sentinels.  The paired
# K256 rows isolate the value of reverse-VJP weighting.
DEFAULT_RECIPES = (
    AutonomousResidualRecipe(
        "r64_l8_hidden", 64, 8, 1.0e-4, "hidden_residual_ridge"
    ),
    AutonomousResidualRecipe(
        "r256_l8_hidden", 256, 8, 1.0e-4, "hidden_residual_ridge"
    ),
    AutonomousResidualRecipe(
        "r256_l8_reverse_vjp",
        256,
        8,
        1.0e-4,
        "reverse_vjp_row_weighted_ridge_v1",
    ),
    AutonomousResidualRecipe(
        "r320_l8_reverse_vjp",
        320,
        8,
        1.0e-4,
        "reverse_vjp_row_weighted_ridge_v1",
    ),
)


def build_outer_lofo_splits(
    family_ids: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Return deterministic train-seven/hold-one family folds."""

    if isinstance(family_ids, (str, bytes)) or not isinstance(
        family_ids, Sequence
    ):
        raise TypeError("family_ids must be a sequence")
    ordered = tuple(sorted({_identifier(value, label="family_id") for value in family_ids}))
    if len(ordered) != _EXPECTED_FAMILIES:
        raise ValueError("outer LOFO requires exactly eight families")
    return tuple(
        {
            "held_family_id": held,
            "training_family_ids": tuple(
                family for family in ordered if family != held
            ),
        }
        for held in ordered
    )


def _shadow_passed(value: Mapping[str, object]) -> bool:
    gates = value.get("gates")
    return isinstance(gates, Mapping) and gates.get("passed") is True


def _required_ledger_passed(name: str, value: object) -> bool:
    if not isinstance(value, Mapping) or not _shadow_passed(value):
        return False
    manifest = value.get("manifest")
    if not isinstance(manifest, Mapping):
        return False
    expected = manifest.get("expected_examples")
    observed = manifest.get("observed_examples")
    family_count = manifest.get("family_count")
    return (
        manifest.get("complete") is True
        and type(expected) is int
        and type(observed) is int
        and expected == observed
        and expected >= _EXPECTED_FAMILIES
        and family_count == _EXPECTED_FAMILIES
        and (name != "ordinary" or expected == _EXPECTED_PROMPTS)
    )


def choose_passing_recipe(
    recipe_rows: Sequence[Mapping[str, object]],
) -> str | None:
    """Choose the cheapest recipe passing all required OOF ledgers."""

    passing: list[tuple[int, int, str]] = []
    for row in recipe_rows:
        recipe_id = _identifier(row.get("recipe_id"), label="recipe_id")
        fidelity = row.get("fidelity")
        resources = row.get("serving_resources")
        if not isinstance(fidelity, Mapping) or not isinstance(resources, Mapping):
            raise TypeError("recipe row omitted fidelity or resources")
        if not all(
            _required_ledger_passed(ledger, fidelity.get(ledger))
            for ledger in _REQUIRED_LEDGERS
        ):
            continue
        macs = resources.get("logical_macs_per_token_upper_bound")
        params = resources.get("prepared_float_scalar_count")
        if type(macs) is not int or macs <= 0 or type(params) is not int or params <= 0:
            raise ValueError("recipe resources must be positive exact integers")
        passing.append((macs, params, recipe_id))
    return None if not passing else min(passing)[2]


@dataclass(slots=True)
class _FitRecord:
    example: object
    sequence: AutonomousCompleteH4TrainingSequence
    prompt_sha256: str
    model_inputs_sha256: str
    supervised_indices_sha256: str
    supervised_targets_sha256: str
    supervised_token_count: int
    ledger_indices: Mapping[str, Tensor]
    receipt_sha256: str


def _collect_fit_records(context: object) -> tuple[_FitRecord, ...]:
    bridge = getattr(context, "bridge")
    adapter = getattr(context, "adapter")
    tokenize = getattr(context, "tokenize")
    bridge_binding = _identifier(
        getattr(bridge, "bridge_binding_sha256", None),
        label="bridge binding",
    )
    records: list[_FitRecord] = []
    for example in sorted(
        tuple(getattr(context, "examples")),
        key=lambda value: _identifier(getattr(value, "example_id", None), label="example_id"),
    ):
        example_id = _identifier(getattr(example, "example_id", None), label="example_id")
        family_id = _identifier(getattr(example, "family_id", None), label="family_id")
        model_inputs, supervised_indices, supervised_targets = _retokenize(tokenize, example)
        model_inputs_hash = gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        base, gradient = bridge.execute_h4_vjp(
            adapter,
            model_inputs,
            objective=_mean_supervised_nll(supervised_indices, supervised_targets),
        )
        source_logits, native_h4, native_positions, native_valid = _native_boundary(
            adapter, model_inputs
        )
        del source_logits
        prefix = getattr(base, "prefix", None)
        base_h4 = getattr(base, "candidate_h4", None)
        if (
            not isinstance(prefix, Gemma3L3L4OnePassPrefix)
            or not isinstance(base_h4, Tensor)
            or not isinstance(gradient, Tensor)
            or base_h4.shape != native_h4.shape
            or gradient.shape != base_h4.shape
            or base_h4.shape[0] != 1
            or getattr(base, "model_inputs_sha256", None) != model_inputs_hash
            or getattr(base, "bridge_binding_sha256", None) != bridge_binding
        ):
            raise RuntimeError("autonomous fit boundary binding differs")
        prefix.validate_integrity()
        support = prefix.complete_h4_causal_support_mask()[0].detach().to(device="cpu")
        source = prefix.source_eligible_mask[0].detach().to(device="cpu")
        valid = prefix.valid_target_mask[0].detach().to(device="cpu")
        core = prefix.target_affected_mask[0].detach().to(device="cpu")
        if (
            not _bitwise_equal(native_positions, prefix.logical_positions)
            or not _bitwise_equal(native_valid, prefix.valid_target_mask)
            or bool((core & ~support).any())
            or bool((support & ~valid).any())
            or not _bitwise_equal(
                native_h4[0][~support.to(native_h4.device)],
                base_h4[0][~support.to(base_h4.device)],
            )
        ):
            raise RuntimeError("autonomous fit causal support differs")
        sequence = AutonomousCompleteH4TrainingSequence(
            example_id=example_id,
            family_id=family_id,
            source_modes=prefix.source_modes[0],
            logical_positions=prefix.logical_positions[0],
            valid_mask=valid,
            source_mask=source,
            support_mask=support,
            base_h4=base_h4[0],
            native_h4=native_h4[0],
            reverse_vjp_gradients=gradient[0],
        )
        positions = supervised_indices.detach().to(device="cpu")
        support_supervised = support.index_select(0, positions)
        core_supervised = core.index_select(0, positions)
        ledger_indices = {
            "ordinary": torch.arange(positions.numel(), dtype=torch.int64),
            "complete_h4_support": torch.nonzero(support_supervised, as_tuple=False).flatten(),
            "graph_core": torch.nonzero(core_supervised, as_tuple=False).flatten(),
            "causal_tail": torch.nonzero(support_supervised & ~core_supervised, as_tuple=False).flatten(),
        }
        receipt = {
            "example_id": example_id,
            "family_id": family_id,
            "prompt_sha256": _prompt_sha256(getattr(example, "prompt")),
            "model_inputs_sha256": model_inputs_hash,
            "supervised_indices_sha256": _tensor_sha256(supervised_indices),
            "supervised_targets_sha256": _tensor_sha256(supervised_targets),
            "training_sequence_sha256": sequence.artifact_sha256,
            "base_execution_sha256": getattr(base, "artifact_sha256"),
            "base_h4_sha256": _tensor_sha256(base_h4),
            "native_h4_sha256": _tensor_sha256(native_h4),
            "reverse_vjp_sha256": _tensor_sha256(gradient),
            "support_mask_sha256": _tensor_sha256(support),
        }
        records.append(
            _FitRecord(
                example=example,
                sequence=sequence,
                prompt_sha256=str(receipt["prompt_sha256"]),
                model_inputs_sha256=model_inputs_hash,
                supervised_indices_sha256=str(receipt["supervised_indices_sha256"]),
                supervised_targets_sha256=str(receipt["supervised_targets_sha256"]),
                supervised_token_count=int(supervised_indices.numel()),
                ledger_indices=ledger_indices,
                receipt_sha256=_sha256(receipt, domain=_RECEIPT_DOMAIN),
            )
        )
        del model_inputs, base, gradient, native_h4
    families = {record.sequence.family_id for record in records}
    if len(records) != _EXPECTED_PROMPTS or len(families) != _EXPECTED_FAMILIES:
        raise RuntimeError("authenticated A16 fit panel geometry differs")
    return tuple(records)


def _fit_provider(
    sequences: Sequence[AutonomousCompleteH4TrainingSequence],
    recipe: AutonomousResidualRecipe,
    *,
    bridge_binding_sha256: str,
) -> AutonomousCompleteH4ResidualProvider:
    decoder = fit_autonomous_complete_h4_output_decoder(sequences, rank=recipe.rank)
    provider = fit_autonomous_complete_h4_residual(
        sequences=sequences,
        output_decoder=decoder,
        state_encoder=None,
        bridge_binding_sha256=bridge_binding_sha256,
        lag_count=recipe.lag_count,
        ridge=recipe.ridge,
        fit_objective=recipe.fit_objective,
    )
    provider.validate_integrity()
    if tuple(provider.fit_family_ids) != tuple(sorted({value.family_id for value in sequences})):
        raise RuntimeError("autonomous provider fit-family binding differs")
    return provider


def _ledger_manifests(records: Sequence[_FitRecord]) -> dict[str, dict[str, str]]:
    result = {name: {} for name in _ALL_LEDGERS}
    for record in records:
        for name, indices in record.ledger_indices.items():
            if int(indices.numel()) > 0:
                result[name][record.sequence.example_id] = record.sequence.family_id
    if any(not result[name] for name in _ALL_LEDGERS):
        raise RuntimeError("autonomous shadow ledger is empty")
    expected_families = {record.sequence.family_id for record in records}
    if (
        len(records) != _EXPECTED_PROMPTS
        or len(expected_families) != _EXPECTED_FAMILIES
        or len(result["ordinary"]) != _EXPECTED_PROMPTS
        or any(
            set(result[name].values()) != expected_families
            for name in _REQUIRED_LEDGERS
        )
    ):
        raise RuntimeError(
            "every required autonomous shadow ledger must cover all eight families"
        )
    return result


def _ledger_coverage(
    manifests: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, int]]:
    return {
        name: {
            "example_count": len(manifest),
            "family_count": len(set(manifest.values())),
        }
        for name, manifest in manifests.items()
    }


def _add_shadow_rows(
    accumulators: Mapping[str, SourceAuthoritativeShadowFidelityAccumulator],
    *,
    record: _FitRecord,
    source_logits: Tensor,
    candidate_logits: Tensor,
    supervised_indices: Tensor,
    supervised_targets: Tensor,
) -> None:
    source_selected = _select_sequence_rows(source_logits, supervised_indices)
    candidate_selected = _select_sequence_rows(candidate_logits, supervised_indices)
    for ledger, selected in record.ledger_indices.items():
        if int(selected.numel()) == 0:
            continue
        selected_device = selected.to(source_selected.device)
        accumulators[ledger].add(
            ShadowFidelityExample(
                example_id=record.sequence.example_id,
                family_id=record.sequence.family_id,
                source_logits=source_selected.index_select(0, selected_device),
                candidate_logits=candidate_selected.index_select(0, selected_device),
                targets=supervised_targets.index_select(
                    0, selected.to(supervised_targets.device)
                ),
            )
        )


def _provider_resources(provider: AutonomousCompleteH4ResidualProvider) -> dict[str, object]:
    params = int(provider.prepared_float_scalar_count)
    macs = int(provider.logical_macs_per_token_upper_bound)
    if params <= 0 or macs <= 0:
        raise RuntimeError("autonomous provider resource accounting differs")
    return {
        "scope": "incremental_autonomous_complete_h4_provider_only",
        "prepared_float_scalar_count": params,
        "runtime_parameter_bytes_float64": params * 8,
        "logical_macs_per_token_upper_bound": macs,
        "retained_gemma_parameters_excluded": True,
        "base_bridge_and_full_suffix_macs_excluded": True,
        "end_to_end_model_parameter_or_flop_claim": False,
    }


def _work_accounting(
    *,
    prompt_count: int,
    recipe_count: int,
    outer_fold_count: int,
    full_provider_fitted: bool,
) -> dict[str, object]:
    if (
        prompt_count != _EXPECTED_PROMPTS
        or recipe_count != len(DEFAULT_RECIPES)
        or outer_fold_count != _EXPECTED_FAMILIES
        or type(full_provider_fitted) is not bool
    ):
        raise RuntimeError("autonomous development work geometry differs")
    breakdown = {
        "fit_native_source_forwards": prompt_count,
        "fit_base_vjp_forwards": prompt_count,
        "fit_base_vjp_backward_traversals": prompt_count,
        "evaluation_native_source_forwards": prompt_count,
        "evaluation_base_forwards": prompt_count,
        "evaluation_recipe_forwards": prompt_count * recipe_count,
    }
    expected_breakdown = {
        "fit_native_source_forwards": _EXPECTED_FIT_SOURCE_FORWARDS,
        "fit_base_vjp_forwards": _EXPECTED_FIT_VJP_FORWARDS,
        "fit_base_vjp_backward_traversals": _EXPECTED_FIT_VJP_BACKWARDS,
        "evaluation_native_source_forwards": _EXPECTED_EVAL_SOURCE_FORWARDS,
        "evaluation_base_forwards": _EXPECTED_EVAL_BASE_FORWARDS,
        "evaluation_recipe_forwards": _EXPECTED_EVAL_RECIPE_FORWARDS,
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
    outer_provider_fits = recipe_count * outer_fold_count
    conditional_provider_fits = int(full_provider_fitted)
    if (
        breakdown != expected_breakdown
        or total_forwards != _EXPECTED_FULL_MODEL_FORWARDS
        or total_backwards != _EXPECTED_FIT_VJP_BACKWARDS
        or outer_provider_fits != _EXPECTED_OUTER_PROVIDER_FITS
    ):
        raise RuntimeError("autonomous development exact work count differs")
    return {
        "outer_fold_provider_fit_count": outer_provider_fits,
        "expected_outer_fold_provider_fit_count": _EXPECTED_OUTER_PROVIDER_FITS,
        "conditional_full_panel_provider_fit_count": conditional_provider_fits,
        "fit_provider_count": outer_provider_fits + conditional_provider_fits,
        "expected_fit_provider_count": (
            _EXPECTED_OUTER_PROVIDER_FITS + conditional_provider_fits
        ),
        "full_model_forward_count": total_forwards,
        "expected_full_model_forward_count": _EXPECTED_FULL_MODEL_FORWARDS,
        "backward_vjp_traversal_count": total_backwards,
        "expected_backward_vjp_traversal_count": _EXPECTED_FIT_VJP_BACKWARDS,
        "full_model_work_breakdown": {
            **breakdown,
            "total_forwards": total_forwards,
            "total_backward_vjp_traversals": total_backwards,
        },
    }


def _validate_recipes(recipes: Sequence[AutonomousResidualRecipe]) -> tuple[AutonomousResidualRecipe, ...]:
    if isinstance(recipes, (str, bytes)) or not isinstance(recipes, Sequence):
        raise TypeError("recipes must be a sequence")
    values = tuple(recipes)
    if len(values) != 4 or any(not isinstance(value, AutonomousResidualRecipe) for value in values):
        raise ValueError("V14 requires exactly four autonomous recipes")
    if len({value.recipe_id for value in values}) != len(values):
        raise ValueError("autonomous recipe IDs must be unique")
    return values


def _validate_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".json" or ".local-runs" not in output.parts:
        raise ValueError("autonomous development output must be JSON under .local-runs")
    return output


def _validate_provider_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".pt" or ".local-runs" not in output.parts:
        raise ValueError("autonomous provider output must be PT under .local-runs")
    return output


def build_autonomous_residual_development_report(
    *,
    artifact_path: Path | str,
    panel: Mapping[str, object],
    bridge_binding_sha256: str,
    recipes: Sequence[AutonomousResidualRecipe],
    folds: Sequence[Mapping[str, object]],
    fit_collection: Mapping[str, object],
    base_fidelity: Mapping[str, object],
    recipe_rows: Sequence[Mapping[str, object]],
    candidate: Mapping[str, object] | None,
    integrity: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic scalar-only V14 report before publication."""

    selected_recipes = _validate_recipes(recipes)
    selected_id = choose_passing_recipe(recipe_rows)
    candidate_id = None if candidate is None else candidate.get("recipe_id")
    if (selected_id is None) != (candidate is None) or candidate_id != selected_id:
        raise ValueError("full candidate does not match OOF recipe selection")
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": 14,
        "scientific_status": "opened_calibration_a_outer_lofo_development",
        "artifact": {"path": Path(artifact_path).as_posix()},
        "panel": dict(panel),
        "bridge_binding_sha256": _identifier(
            bridge_binding_sha256, label="bridge binding"
        ),
        "recipe_bank": [
            {**value.metadata(), "recipe_sha256": value.artifact_sha256}
            for value in selected_recipes
        ],
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
        "recipes": [dict(value) for value in recipe_rows],
        "selection": {
            "rule": (
                "all_required_ledgers_with_eight_family_coverage_pass_then_"
                "min_incremental_provider_macs_params_recipe_id"
            ),
            "required_ledgers": _REQUIRED_LEDGERS,
            "selected_recipe_id": selected_id,
            "passed": selected_id is not None,
        },
        "candidate": None if candidate is None else dict(candidate),
        "integrity": dict(integrity),
        "passed": selected_id is not None,
        "classification": (
            "autonomous_complete_h4_oof_candidate_ready_for_fresh_guard"
            if selected_id is not None
            else "autonomous_complete_h4_oof_recipes_insufficient"
        ),
        "success_authorizes": (
            "freeze_one_candidate_then_open_fresh_family_disjoint_guard"
            if selected_id is not None
            else None
        ),
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }
    _scalar_report(report)
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
        raise ValueError("published provider must match the selected candidate")
    destinations = (output,) if provider is None else (output, provider_output)
    reservation = _reserve_outputs(destinations)
    report_stage: Path | None = None
    provider_stage: Path | None = None
    try:
        staged: tuple[Path, ...]
        if provider is not None:
            if not isinstance(candidate, Mapping):
                raise TypeError("selected candidate must be a mapping")
            provider.validate_integrity()
            if (
                candidate.get("provider_artifact_sha256")
                != provider.artifact_sha256
            ):
                raise ValueError("selected candidate and provider differ")
            provider_stage = _stage_torch(
                autonomous_complete_h4_residual_provider_state_dict(provider),
                provider_output,
            )
            provider_file_sha256 = _file_sha256(provider_stage)
            restored = load_autonomous_complete_h4_residual_provider(
                provider_stage,
                expected_artifact_sha256=provider.artifact_sha256,
                expected_file_sha256=provider_file_sha256,
                expected_bridge_binding_sha256=provider.bridge_binding_sha256,
            )
            if restored.metadata() != provider.metadata():
                raise RuntimeError("staged autonomous provider roundtrip drifted")
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
        _scalar_report(report)
        report["report_sha256"] = _sha256(report, domain=_REPORT_DOMAIN)
        report_stage = _stage_json(report, output)
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
                "file_sha256": _file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        for stage in (report_stage, provider_stage):
            if stage is not None:
                stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_autonomous_residual_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    provider_output: Path | str | None = None,
    cache_dir: Path | str | None = None,
    recipes: Sequence[AutonomousResidualRecipe] = DEFAULT_RECIPES,
) -> dict[str, object]:
    """Fit and stream the fixed A16 outer-LOFO autonomous-head screen."""

    destination = _validate_output(output)
    provider_destination = _validate_provider_output(
        destination.with_suffix(".provider.pt")
        if provider_output is None
        else provider_output
    )
    if provider_destination == destination:
        raise ValueError("report and provider outputs must differ")
    if destination.exists():
        raise FileExistsError("refusing to overwrite autonomous V14 report")
    if provider_destination.exists():
        raise FileExistsError("refusing to overwrite autonomous V14 provider")
    selected_recipes = _validate_recipes(recipes)
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        records = _collect_fit_records(context)
        bridge_binding = _identifier(
            context.bridge.bridge_binding_sha256, label="bridge binding"
        )
        families = tuple(sorted({record.sequence.family_id for record in records}))
        folds = build_outer_lofo_splits(families)
        providers: dict[str, dict[str, AutonomousCompleteH4ResidualProvider]] = {}
        for recipe in selected_recipes:
            providers[recipe.recipe_id] = {}
            for fold in folds:
                held = str(fold["held_family_id"])
                training = tuple(
                    record.sequence for record in records if record.sequence.family_id != held
                )
                if len(training) != 14 or held in {value.family_id for value in training}:
                    raise RuntimeError("outer LOFO training ownership differs")
                providers[recipe.recipe_id][held] = _fit_provider(
                    training, recipe, bridge_binding_sha256=bridge_binding
                )

        manifests = _ledger_manifests(records)
        ledger_coverage = _ledger_coverage(manifests)
        accumulators = {
            arm: {
                ledger: SourceAuthoritativeShadowFidelityAccumulator(
                    manifest,
                    gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
                )
                for ledger, manifest in manifests.items()
            }
            for arm in ("base", *(value.recipe_id for value in selected_recipes))
        }
        causal_checks = 0
        for record in records:
            model_inputs, supervised_indices, supervised_targets = _retokenize(
                context.tokenize, record.example
            )
            if (
                gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
                != record.model_inputs_sha256
                or _tensor_sha256(supervised_indices) != record.supervised_indices_sha256
                or _tensor_sha256(supervised_targets) != record.supervised_targets_sha256
            ):
                raise RuntimeError("autonomous shadow retokenization drifted")
            source_logits, _native_h4, native_positions, native_valid = _native_boundary(
                context.adapter, model_inputs
            )
            base = context.bridge.execute(context.adapter, model_inputs)
            if (
                not isinstance(base, Gemma3L3L4OnePassExecution)
                or not _bitwise_equal(native_positions, base.prefix.logical_positions)
                or not _bitwise_equal(native_valid, base.prefix.valid_target_mask)
            ):
                raise RuntimeError("autonomous shadow base sequence differs")
            _add_shadow_rows(
                accumulators["base"],
                record=record,
                source_logits=source_logits,
                candidate_logits=base.logits,
                supervised_indices=supervised_indices,
                supervised_targets=supervised_targets,
            )
            support = base.prefix.complete_h4_causal_support_mask()
            for recipe in selected_recipes:
                provider = providers[recipe.recipe_id][record.sequence.family_id]
                if record.sequence.family_id in set(provider.fit_family_ids):
                    raise RuntimeError("held family leaked into autonomous provider")
                candidate = context.bridge.execute(
                    context.adapter, model_inputs, h4_head=provider
                )
                if (
                    candidate.h4_head_sha256 != provider.artifact_sha256
                    or candidate.prefix.artifact_sha256 != base.prefix.artifact_sha256
                    or not _bitwise_equal(
                        candidate.candidate_h4[~support.to(candidate.candidate_h4.device)],
                        base.candidate_h4[~support.to(base.candidate_h4.device)],
                    )
                ):
                    raise RuntimeError("autonomous provider escaped causal support")
                _add_shadow_rows(
                    accumulators[recipe.recipe_id],
                    record=record,
                    source_logits=source_logits,
                    candidate_logits=candidate.logits,
                    supervised_indices=supervised_indices,
                    supervised_targets=supervised_targets,
                )
                causal_checks += 1
                del candidate
            del model_inputs, source_logits, _native_h4, base

        fidelity = {
            arm: {ledger: value.finalize() for ledger, value in ledgers.items()}
            for arm, ledgers in accumulators.items()
        }
        recipe_rows: list[dict[str, object]] = []
        for recipe in selected_recipes:
            fold_providers = providers[recipe.recipe_id]
            resource_rows = [_provider_resources(value) for value in fold_providers.values()]
            if len({tuple(sorted(value.items())) for value in resource_rows}) != 1:
                raise RuntimeError("outer-fold provider resources differ")
            resources = resource_rows[0]
            row = {
                **recipe.metadata(),
                "recipe_sha256": recipe.artifact_sha256,
                "outer_fold_count": len(fold_providers),
                "every_fold_fit_family_count": 7,
                "fold_provider_artifact_sha256s": {
                    held: provider.artifact_sha256
                    for held, provider in sorted(fold_providers.items())
                },
                "serving_resources": resources,
                "fidelity": fidelity[recipe.recipe_id],
            }
            recipe_rows.append(row)
        selected_id = choose_passing_recipe(recipe_rows)
        full_provider: AutonomousCompleteH4ResidualProvider | None = None
        if selected_id is not None:
            selected_recipe = next(value for value in selected_recipes if value.recipe_id == selected_id)
            full_provider = _fit_provider(
                tuple(record.sequence for record in records),
                selected_recipe,
                bridge_binding_sha256=bridge_binding,
            )
        candidate = (
            None
            if full_provider is None
            else {
                "recipe_id": selected_id,
                "provider_artifact_sha256": full_provider.artifact_sha256,
                "provider": full_provider.metadata(),
                "serving_resources": _provider_resources(full_provider),
                "fit_family_count": 8,
                "serving_inputs": (
                    "model_inputs",
                    "one_pass_prefix_source_modes_and_masks",
                    "pre_correction_realized_h4",
                ),
                "native_h4_logits_targets_or_gradients_required_at_runtime": False,
            }
        )
        context.validate_immutable_inputs()
        work = _work_accounting(
            prompt_count=len(records),
            recipe_count=len(selected_recipes),
            outer_fold_count=len(folds),
            full_provider_fitted=full_provider is not None,
        )
        integrity = {
            "outer_fold_count": len(folds),
            "ledger_coverage": ledger_coverage,
            **work,
            "causal_off_support_execution_checks": causal_checks,
            "expected_causal_off_support_execution_checks": 64,
            "source_native_data_entered_serving_provider": False,
            "full_provider_fit_was_conditional_on_oof_pass": True,
            "guard_opened": False,
            "calibration_b_opened": False,
        }
        if (
            sum(len(value) for value in providers.values())
            != _EXPECTED_OUTER_PROVIDER_FITS
            or integrity["causal_off_support_execution_checks"] != 64
            or integrity["full_model_forward_count"]
            != _EXPECTED_FULL_MODEL_FORWARDS
            or integrity["backward_vjp_traversal_count"]
            != _EXPECTED_FIT_VJP_BACKWARDS
        ):
            raise RuntimeError("autonomous development exact work count differs")
        report = build_autonomous_residual_development_report(
            artifact_path=destination,
            panel=context.panel_receipt,
            bridge_binding_sha256=bridge_binding,
            recipes=selected_recipes,
            folds=folds,
            fit_collection={
                "prompt_count": len(records),
                "family_count": len(families),
                "supervised_token_count": sum(value.supervised_token_count for value in records),
                "trace_receipt_sha256s": [value.receipt_sha256 for value in records],
                "native_h4_and_reverse_vjp_fit_only": True,
                "raw_fit_trace_tensor_serialization": False,
                "conditional_runtime_provider_tensor_sidecar": True,
            },
            base_fidelity=fidelity["base"],
            recipe_rows=recipe_rows,
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
    report = run_gemma3_l3_l4_complete_h4_autonomous_residual_development(
        output=args.output,
        provider_output=args.provider_output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps({
        "path": report["artifact"]["path"],  # type: ignore[index]
        "report_sha256": report["report_sha256"],
        "classification": report["classification"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
