"""Leakage-aware evaluation for an edgeless full-MLP-stack replacement.

This evaluator accepts only an already frozen stack executor and an explicitly
declared assessment membership.  It compares native inference, generated
full-stack inference, and matched deletion while validating the physical and
logical execution evidence returned by every batch.

Metrics are accumulated as per-token terms and reduced once with
``math.fsum``.  Consequently they do not average batch means and are invariant
to how the same ordered assessment examples are batched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Protocol

import torch
from torch import Tensor

from .compiler.calibration import CalibrationBatch


__all__ = ["evaluate_full_mlp_stack_conditions"]


_REPLACEMENT_SCOPE = "full_native_mlp_stack_replacement"
_RETAINED_NATIVE_COMPONENTS = (
    "embeddings",
    "attention",
    "normalization",
    "language_model_head",
)
_STATIC_FIELDS = (
    "replacement_scope",
    "replaced_layer_count",
    "removed_mode_count",
    "source_whole_model_learned_parameters",
    "logical_native_mlp_stack_learned_parameters",
    "logical_retained_native_non_mlp_learned_parameters",
    "logical_generator_stack_learned_parameters",
    "logical_candidate_learned_parameters",
    "logical_net_stored_parameter_savings",
    "experimental_resident_source_learned_parameters",
    "experimental_resident_compiled_learned_parameters",
    "experimental_resident_total_learned_parameters",
    "experimental_resident_overhead_vs_logical_candidate",
    "native_components_retained",
    "logical_candidate_excludes_native_mlp_stack",
    "experimental_resident_source_state_retained",
)
_LOGICAL_FIELDS = (
    "logical_linear_macs_native_mlp_stack",
    "logical_generator_macs",
    "logical_executed_generator_macs",
    "logical_generator_bias_additions",
    "logical_executed_generator_bias_additions",
    "net_logical_macs_saved",
)


class _AdapterLike(Protocol):
    module: object


class _ExecutorLike(Protocol):
    replaced_layer_count: int
    removed_mode_count: int
    compiled_mlps: object

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str,
    ) -> object: ...


def _model_logits(output: object) -> Tensor:
    logits = (
        output.get("logits")
        if isinstance(output, Mapping)
        else getattr(output, "logits", None)
    )
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError(
            "model output must expose finite [batch, sequence, vocab] logits"
        )
    return logits


def _selected_logits_and_targets(
    logits: Tensor,
    batch: CalibrationBatch,
) -> tuple[Tensor, Tensor]:
    targets = batch.targets.to(device=logits.device)
    if targets.shape != logits.shape[:2]:
        raise ValueError("evaluation targets and logits positions differ")
    supervised = targets != -100
    valid = batch.valid_positions.to(device=logits.device)
    if valid.shape != supervised.shape or bool((supervised & ~valid).any()):
        raise ValueError(
            "supervised targets must be a subset of valid positions"
        )
    if not bool(supervised.any()):
        raise ValueError("evaluation batch has no supervised tokens")
    return (
        logits[supervised].detach().to(device="cpu", dtype=torch.float64),
        targets[supervised].detach().to(device="cpu", dtype=torch.long),
    )


def _nll_terms(logits: Tensor, targets: Tensor) -> list[float]:
    log_normalizer = torch.logsumexp(logits, dim=-1)
    row = torch.arange(targets.shape[0])
    return (
        -(logits[row, targets] - log_normalizer)
    ).tolist()


def _candidate_terms(
    native_logits: Tensor,
    candidate_logits: Tensor,
    targets: Tensor,
    *,
    vocabulary_chunk_size: int,
) -> tuple[list[float], list[float], int]:
    if candidate_logits.shape != native_logits.shape:
        raise ValueError("native and candidate supervised logits differ")
    native_lse = torch.logsumexp(native_logits, dim=-1)
    candidate_lse = torch.logsumexp(candidate_logits, dim=-1)
    row = torch.arange(targets.shape[0])
    nll = -(
        candidate_logits[row, targets] - candidate_lse
    )
    kl_rows = torch.zeros(
        native_logits.shape[0],
        dtype=torch.float64,
    )
    for start in range(0, native_logits.shape[1], vocabulary_chunk_size):
        stop = min(start + vocabulary_chunk_size, native_logits.shape[1])
        native_log_probability = (
            native_logits[:, start:stop] - native_lse[:, None]
        )
        candidate_log_probability = (
            candidate_logits[:, start:stop] - candidate_lse[:, None]
        )
        kl_rows += (
            native_log_probability.exp()
            * (native_log_probability - candidate_log_probability)
        ).sum(dim=1)
    top1_matches = int(
        (
            candidate_logits.argmax(dim=-1)
            == native_logits.argmax(dim=-1)
        ).sum().item()
    )
    return nll.tolist(), kl_rows.tolist(), top1_matches


def _canonical_mode_counts(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("expected_mode_counts_by_layer must be a sequence")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
    ):
        raise ValueError(
            "expected_mode_counts_by_layer must contain positive integers"
        )
    return result


def _validate_executor_declaration(
    executor: _ExecutorLike,
    mode_counts: tuple[int, ...],
) -> None:
    if getattr(executor, "replaced_layer_count", None) != len(mode_counts):
        raise ValueError("executor does not cover every declared layer")
    if getattr(executor, "removed_mode_count", None) != sum(mode_counts):
        raise ValueError("executor does not cover every declared mode")
    compiled = getattr(executor, "compiled_mlps", None)
    keys = getattr(compiled, "keys", None)
    if not callable(keys) or tuple(keys()) != tuple(
        str(index) for index in range(len(mode_counts))
    ):
        raise ValueError("compiled MLP catalog does not cover declared layers")
    for ordinal, expected_modes in enumerate(mode_counts):
        candidate = compiled[str(ordinal)]
        if (
            getattr(candidate, "removed_mode_count", None) != expected_modes
            or getattr(candidate, "removed_mode_indices", None)
            != tuple(range(expected_modes))
            or getattr(candidate, "is_full_native_replacement", None) is not True
        ):
            raise ValueError(
                "compiled MLP does not remove every declared native mode"
            )


def _static_fields(
    execution: object,
    *,
    expected_layers: int,
    expected_modes: int,
    label: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in _STATIC_FIELDS:
        if not hasattr(execution, field):
            raise TypeError(f"{label} execution is missing {field}")
        result[field] = getattr(execution, field)
    if result["replacement_scope"] != _REPLACEMENT_SCOPE:
        raise ValueError(f"{label} is not a full native MLP stack replacement")
    if (
        result["replaced_layer_count"] != expected_layers
        or result["removed_mode_count"] != expected_modes
    ):
        raise ValueError(f"{label} full-stack layer/mode coverage drifted")
    if result["native_components_retained"] != _RETAINED_NATIVE_COMPONENTS:
        raise ValueError(f"{label} retained-native declaration drifted")
    if (
        result["logical_candidate_excludes_native_mlp_stack"] is not True
        or result["experimental_resident_source_state_retained"] is not True
    ):
        raise ValueError(f"{label} logical/resident declaration drifted")

    integer_fields = tuple(
        field
        for field in _STATIC_FIELDS
        if field
        not in {
            "replacement_scope",
            "native_components_retained",
            "logical_candidate_excludes_native_mlp_stack",
            "experimental_resident_source_state_retained",
        }
    )
    if any(type(result[field]) is not int for field in integer_fields):
        raise TypeError(f"{label} static accounting must use exact integers")
    source = int(result["source_whole_model_learned_parameters"])
    native_mlp = int(
        result["logical_native_mlp_stack_learned_parameters"]
    )
    retained = int(
        result["logical_retained_native_non_mlp_learned_parameters"]
    )
    generated = int(
        result["logical_generator_stack_learned_parameters"]
    )
    candidate = int(result["logical_candidate_learned_parameters"])
    resident_source = int(
        result["experimental_resident_source_learned_parameters"]
    )
    resident_compiled = int(
        result["experimental_resident_compiled_learned_parameters"]
    )
    resident_total = int(
        result["experimental_resident_total_learned_parameters"]
    )
    if (
        source <= 0
        or native_mlp <= 0
        or generated <= 0
        or retained != source - native_mlp
        or candidate != retained + generated
        or result["logical_net_stored_parameter_savings"]
        != source - candidate
        or resident_source != source
        or resident_compiled != generated
        or resident_total != resident_source + resident_compiled
        or result["experimental_resident_overhead_vs_logical_candidate"]
        != resident_total - candidate
        or result["experimental_resident_overhead_vs_logical_candidate"]
        != native_mlp
    ):
        raise ValueError(f"{label} static parameter accounting is inconsistent")
    return result


def _logical_fields(
    execution: object,
    *,
    condition: str,
    valid_tokens: int,
    native_mlp_parameters: int,
    label: str,
) -> dict[str, int]:
    if getattr(execution, "condition", None) != condition:
        raise ValueError(f"{label} execution condition drifted")
    if getattr(execution, "valid_tokens", None) != valid_tokens:
        raise ValueError(f"{label} valid-token accounting drifted")
    result: dict[str, int] = {}
    for field in _LOGICAL_FIELDS:
        value = getattr(execution, field, None)
        if type(value) is not int:
            raise TypeError(f"{label} {field} must be an exact integer")
        result[field] = value
    native_macs = result["logical_linear_macs_native_mlp_stack"]
    generator_macs = result["logical_generator_macs"]
    executed_macs = result["logical_executed_generator_macs"]
    generator_additions = result["logical_generator_bias_additions"]
    executed_additions = result[
        "logical_executed_generator_bias_additions"
    ]
    expected_executed_macs = (
        generator_macs if condition == "generated" else 0
    )
    expected_executed_additions = (
        generator_additions if condition == "generated" else 0
    )
    if (
        native_macs != valid_tokens * native_mlp_parameters
        or generator_macs <= 0
        or executed_macs != expected_executed_macs
        or executed_additions != expected_executed_additions
        or result["net_logical_macs_saved"]
        != native_macs - executed_macs
    ):
        raise ValueError(f"{label} logical compute accounting is inconsistent")
    return result


def _accumulate(
    totals: dict[str, int],
    values: Mapping[str, int],
) -> None:
    for field, value in values.items():
        totals[field] += value


def evaluate_full_mlp_stack_conditions(
    adapter: _AdapterLike,
    executor: _ExecutorLike,
    batches: Sequence[CalibrationBatch],
    *,
    expected_example_ids: Sequence[str],
    expected_mode_counts_by_layer: Sequence[int],
    vocabulary_chunk_size: int = 16384,
    assessment_role: str = "open_development_assessment",
) -> dict[str, object]:
    """Evaluate one frozen edgeless full-stack candidate and deletion control."""

    materialized = tuple(batches)
    if (
        not materialized
        or any(not isinstance(batch, CalibrationBatch) for batch in materialized)
    ):
        raise ValueError("batches must contain CalibrationBatch values")
    if (
        isinstance(expected_example_ids, (str, bytes))
        or not isinstance(expected_example_ids, Sequence)
    ):
        raise TypeError("expected_example_ids must be a sequence")
    expected_ids = tuple(expected_example_ids)
    observed_ids = tuple(
        example_id
        for batch in materialized
        for example_id in (
            batch.example_ids if batch.example_ids is not None else ()
        )
    )
    if (
        not expected_ids
        or len(expected_ids) != len(set(expected_ids))
        or any(not isinstance(value, str) or not value for value in expected_ids)
        or any(batch.example_ids is None for batch in materialized)
        or observed_ids != expected_ids
    ):
        raise ValueError(
            "assessment batches do not match the declared example membership"
        )
    if type(vocabulary_chunk_size) is not int or vocabulary_chunk_size <= 0:
        raise ValueError("vocabulary_chunk_size must be positive")
    if assessment_role != "open_development_assessment":
        raise ValueError(
            "this evaluator cannot relabel open development as a closed split"
        )
    mode_counts = _canonical_mode_counts(expected_mode_counts_by_layer)
    _validate_executor_declaration(executor, mode_counts)
    native_model = getattr(adapter, "module", None)
    if not callable(native_model):
        raise TypeError("adapter must expose a callable native module")

    native_nll_terms: list[float] = []
    candidate_nll_terms = {
        "generated_full_stack": [],
        "matched_deletion": [],
    }
    candidate_kl_terms = {
        "generated_full_stack": [],
        "matched_deletion": [],
    }
    top1_matches = {
        "generated_full_stack": 0,
        "matched_deletion": 0,
    }
    static_by_condition: dict[str, dict[str, object]] = {}
    logical_totals = {
        name: {field: 0 for field in _LOGICAL_FIELDS}
        for name in ("generated_full_stack", "matched_deletion")
    }
    supervised_tokens = 0
    logical_valid_tokens = 0

    for batch in materialized:
        call_inputs: dict[str, object] = dict(batch.model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        with torch.no_grad():
            native_output = native_model(**call_inputs)
            generated = executor.run(
                batch.model_inputs,
                condition="generated",
            )
            deletion = executor.run(
                batch.model_inputs,
                condition="matched_deletion",
            )
        native_logits, targets = _selected_logits_and_targets(
            _model_logits(native_output),
            batch,
        )
        native_nll_terms.extend(_nll_terms(native_logits, targets))
        supervised_tokens += targets.numel()
        valid_tokens = int(batch.valid_positions.sum().item())
        logical_valid_tokens += valid_tokens

        for name, execution, condition in (
            ("generated_full_stack", generated, "generated"),
            ("matched_deletion", deletion, "matched_deletion"),
        ):
            candidate_logits, candidate_targets = (
                _selected_logits_and_targets(
                    _model_logits(getattr(execution, "model_output", None)),
                    batch,
                )
            )
            if not torch.equal(targets, candidate_targets):
                raise ValueError(f"{name} evaluation targets drifted")
            nll, kl, matches = _candidate_terms(
                native_logits,
                candidate_logits,
                targets,
                vocabulary_chunk_size=vocabulary_chunk_size,
            )
            candidate_nll_terms[name].extend(nll)
            candidate_kl_terms[name].extend(kl)
            top1_matches[name] += matches

            static = _static_fields(
                execution,
                expected_layers=len(mode_counts),
                expected_modes=sum(mode_counts),
                label=name,
            )
            prior_static = static_by_condition.setdefault(name, static)
            if prior_static != static:
                raise ValueError(f"{name} static accounting changed by batch")
            logical = _logical_fields(
                execution,
                condition=condition,
                valid_tokens=valid_tokens,
                native_mlp_parameters=int(
                    static[
                        "logical_native_mlp_stack_learned_parameters"
                    ]
                ),
                label=name,
            )
            _accumulate(logical_totals[name], logical)

    if supervised_tokens <= 0:
        raise ValueError("assessment has no supervised tokens")
    if (
        static_by_condition["generated_full_stack"]
        != static_by_condition["matched_deletion"]
    ):
        raise ValueError(
            "generated and matched-deletion physical scopes differ"
        )
    generated_totals = logical_totals["generated_full_stack"]
    deletion_totals = logical_totals["matched_deletion"]
    for field in (
        "logical_linear_macs_native_mlp_stack",
        "logical_generator_macs",
        "logical_generator_bias_additions",
    ):
        if generated_totals[field] != deletion_totals[field]:
            raise ValueError(
                "generated and matched-deletion logical scopes differ"
            )
    if (
        generated_totals["logical_executed_generator_macs"]
        != generated_totals["logical_generator_macs"]
        or generated_totals[
            "logical_executed_generator_bias_additions"
        ]
        != generated_totals["logical_generator_bias_additions"]
        or deletion_totals["logical_executed_generator_macs"] != 0
        or deletion_totals[
            "logical_executed_generator_bias_additions"
        ]
        != 0
    ):
        raise ValueError("generated/deletion executed-work controls drifted")

    native_nll = math.fsum(native_nll_terms) / supervised_tokens
    conditions: dict[str, object] = {
        "native": {
            "nll_per_token": native_nll,
            "delta_nll_per_token": 0.0,
            "native_to_candidate_kl_per_token": 0.0,
            "top1_agreement_to_native": 1.0,
        }
    }
    for name in ("generated_full_stack", "matched_deletion"):
        nll = math.fsum(candidate_nll_terms[name]) / supervised_tokens
        conditions[name] = {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": max(
                math.fsum(candidate_kl_terms[name]) / supervised_tokens,
                0.0,
            ),
            "top1_agreement_to_native": (
                top1_matches[name] / supervised_tokens
            ),
        }
    static = static_by_condition["generated_full_stack"]
    return {
        "execution_path": "edgeless_full_mlp_stack_rung",
        "assessment_role": assessment_role,
        "heldout_confirmation": False,
        "assessment_membership_exact": True,
        "assessment_used_for_fitting": False,
        "supervised_tokens": supervised_tokens,
        "logical_valid_tokens": logical_valid_tokens,
        "declared_scope": {
            "replacement_scope": _REPLACEMENT_SCOPE,
            "layer_count": len(mode_counts),
            "removed_mode_count": sum(mode_counts),
            "mode_counts_by_layer": mode_counts,
            "all_declared_layers_and_modes_replaced": True,
        },
        "conditions": conditions,
        "control_validation": {
            "physical_scope_identical": True,
            "generated_compute_executed": True,
            "matched_deletion_compute_zero": True,
        },
        "resource_accounting": {
            "generated_full_stack": {
                **static,
                **generated_totals,
                "storage_scope": "standalone_logical_candidate",
            },
            "matched_deletion": {
                **static,
                **deletion_totals,
                "storage_scope": (
                    "runtime_branch_generator_parameters_still_resident"
                ),
            },
        },
        "latency_or_kernel_speed_claim": False,
    }
