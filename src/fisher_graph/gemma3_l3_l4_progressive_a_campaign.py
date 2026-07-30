"""Run one strict three-way Calibration-A progressive Gemma campaign.

The campaign is development-only.  It authenticates the frozen rank-64
measurement runtime, restores the complete factorized Gemma carrier, and runs
the generic progressive controller with candidate-bound X4/H4 residual heads.
Fit and selection text are opened normally.  Guard text is not read or
tokenized until a manifest-global durable claim has been created for the
frozen challenger.

Calibration B is never accepted as an input by this module.  The resulting
report contains hashes, scalar metrics, resource accounting, and (when
available) a serialized two-head candidate.  It contains no prompt text,
token IDs, activation rows, gradients, or fit sequences.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Literal

import torch
from torch import Tensor, nn

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .compiler.progressive import (
    CandidateEvaluation,
    DevelopmentCorpus,
    FrozenProgressiveCandidateHandoff,
    ProgressiveCompilationProtocol,
    ProgressiveCompilationResult,
    ProgressiveResourceBudget,
    ProgressiveResourceFootprint,
    freeze_progressive_candidate,
    run_progressive_compilation,
    run_progressive_development_search,
)
from .gemma3_experiment import (
    make_causal_lm_calibration_batches,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    load_gemma3_l3_l4_basis_package,
)
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CANDIDATE,
    load_gemma3_graph_organized_svd_candidate,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    _load_and_validate_frozen_local_tokenizer,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveACorpus,
    Gemma3L3L4ProgressiveARolePreclaimView,
    Gemma3L3L4ProgressiveARolePrompts,
    load_gemma3_l3_l4_progressive_a_corpus,
)
from .gemma3_l3_l4_progressive_compilation import (
    current_gemma3_l3_l4_progressive_seed,
    gemma3_l3_l4_legacy_progressive_binding_metadata,
    make_gemma3_l3_l4_progressive_protocol,
)
from .gemma3_l3_l4_progressive_guard_ledger import (
    Gemma3L3L4ProgressiveGuardClaimAuthority,
)
from .gemma3_l3_l4_progressive_worker import (
    Gemma3L3L4ProgressiveWorker,
    GemmaGuardPanelProvider,
    GemmaProgressivePanel,
    LEGACY_RANK64_INCOMPLETE_COST_REASONS,
    LegacyRank64GemmaProgressiveExecutable,
    gemma_progressive_panel_membership_receipt_sha256,
    make_gemma_progressive_panel,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    DEFAULT_REVISION,
    _load_local_gemma3_model_only,
)
from .gemma3_l3_l4_two_head_lowerer import (
    GEMMA_TWO_HEAD_COMPUTE_SCOPE,
    GEMMA_TWO_HEAD_PARAMETER_SCOPE,
    GEMMA_TWO_HEAD_RUNTIME_DTYPE,
    GEMMA_TWO_HEAD_RUNTIME_ID,
    GemmaL3L4TwoHeadMutationLowerer,
    ResidualHeadConditioning,
    ResidualHeadFitObjective,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "DEFAULT_PROGRESSIVE_A_CORPUS_ARTIFACT",
    "DEFAULT_PROGRESSIVE_A_FIT_INPUT",
    "DEFAULT_PROGRESSIVE_A_GUARD_INPUT",
    "DEFAULT_PROGRESSIVE_A_REPORT",
    "DEFAULT_PROGRESSIVE_A_SELECTION_INPUT",
    "Gemma3L3L4ClaimGatedGuardPanelProvider",
    "build_gemma3_l3_l4_progressive_resource_envelope",
    "gemma3_l3_l4_guard_preclaim_binding_sha256",
    "main",
    "materialize_gemma3_l3_l4_progressive_panel",
    "run_gemma3_l3_l4_progressive_a_campaign",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_progressive_a_campaign"
_FORMAT_VERSION = 4
_CAMPAIGN_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-a-campaign:v4\0"
)
_GUARD_PRECLAIM_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-guard-preclaim-binding:v1\0"
)
_SEQUENCE_SCOPE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-sequence-scope:v1\0"
)
_RESOURCE_ACCOUNTING_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-seed-resources:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-progressive-a-report:v4\0"
)
_FACTORIZED_SCOPE = "factorized_refit"

_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_PROGRESSIVE_A_CORPUS_ARTIFACT = (
    _LOCAL_ROOT / "progressive-a-pilot-v1.corpus.json"
)
DEFAULT_PROGRESSIVE_A_FIT_INPUT = (
    _LOCAL_ROOT / "progressive-a-pilot-v1.fit.json"
)
DEFAULT_PROGRESSIVE_A_SELECTION_INPUT = (
    _LOCAL_ROOT / "progressive-a-pilot-v1.selection.json"
)
DEFAULT_PROGRESSIVE_A_GUARD_INPUT = (
    _LOCAL_ROOT / "progressive-a-pilot-v1.guard.json"
)
DEFAULT_PROGRESSIVE_A_REPORT = (
    _LOCAL_ROOT / "progressive-a-pilot-v1.campaign.json"
)


def _canonical_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {
            key: _canonical_json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value)!r}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _model_resources(model: nn.Module) -> tuple[int, int, int]:
    parameters = tuple(model.parameters())
    learned = sum(int(value.numel()) for value in parameters)
    runtime_bytes = sum(
        int(value.numel() * value.element_size())
        for value in parameters
    )
    linear_macs = sum(
        int(module.weight.numel())
        for module in model.modules()
        if isinstance(module, nn.Linear)
    )
    if min(learned, runtime_bytes, linear_macs) <= 0:
        raise ValueError("model resource scope is empty")
    return learned, runtime_bytes, linear_macs


def _role_batches(
    tokenizer: object,
    role_input: Gemma3L3L4ProgressiveARolePrompts,
    *,
    max_length: int,
    device: torch.device,
) -> tuple[CalibrationBatch, ...]:
    """Tokenize one role one example at a time and install prompt-hash IDs."""

    generated = tuple(
        make_causal_lm_calibration_batches(
            tokenizer,
            role_input.prompts,
            max_length=max_length,
            tokenization_batch_size=1,
            device=device,
        )
    )
    if (
        len(generated) != len(role_input.prompts)
        or any(batch.batch_size != 1 for batch in generated)
    ):
        raise RuntimeError("strict role tokenization lost example boundaries")
    return tuple(
        CalibrationBatch(
            model_inputs={
                name: value.detach().clone()
                for name, value in batch.model_inputs.items()
            },
            targets=batch.targets.detach().clone(),
            valid_positions=batch.valid_positions.detach().clone(),
            shared_input_names=batch.shared_input_names,
            example_ids=(example_id,),
        )
        for batch, example_id in zip(
            generated,
            role_input.ordered_prompt_sha256s,
            strict=True,
        )
    )


def materialize_gemma3_l3_l4_progressive_panel(
    *,
    tokenizer: object,
    role_input: Gemma3L3L4ProgressiveARolePrompts,
    view: Gemma3L3L4ProgressiveARolePreclaimView,
    max_length: int,
    device: torch.device,
    forbidden_manifest_sha256s: tuple[str, ...],
) -> GemmaProgressivePanel:
    """Authenticate and tokenize one opened A role into a strict panel."""

    if role_input.role != view.role:
        raise ValueError("opened role and preclaim view differ")
    if (
        role_input.ordered_prompt_sha256s
        != view.ordered_prompt_sha256s
        or role_input.family_ids != view.ordered_family_ids
        or role_input.source_file_sha256
        != view.role_input_file_sha256
    ):
        raise ValueError("opened role differs from its frozen membership")
    batches = _role_batches(
        tokenizer,
        role_input,
        max_length=max_length,
        device=device,
    )
    panel = make_gemma_progressive_panel(
        role=view.role,
        manifest_sha256=view.manifest_sha256,
        batches=batches,
        family_by_example=role_input.family_by_example,
        forbidden_manifest_sha256s=forbidden_manifest_sha256s,
    )
    expected_membership = (
        gemma_progressive_panel_membership_receipt_sha256(
            role=view.role,
            manifest_sha256=view.manifest_sha256,
            family_by_example=role_input.family_by_example,
        )
    )
    if panel.membership_receipt_sha256 != expected_membership:
        raise RuntimeError("materialized panel membership receipt drifted")
    return panel


def gemma3_l3_l4_guard_preclaim_binding_sha256(
    *,
    corpus_artifact_sha256: str,
    tokenizer_contract_sha256: str,
    view: Gemma3L3L4ProgressiveARolePreclaimView,
) -> str:
    """Commit guard source membership and tokenizer semantics before opening."""

    if view.role != "calibration_a_guard":
        raise ValueError("guard preclaim binding requires the guard role")
    membership = gemma_progressive_panel_membership_receipt_sha256(
        role=view.role,
        manifest_sha256=view.manifest_sha256,
        family_by_example=view.family_by_example,
    )
    return _domain_sha256(
        _GUARD_PRECLAIM_DOMAIN,
        {
            "format_version": 1,
            "corpus_artifact_sha256": corpus_artifact_sha256,
            "tokenizer_contract_sha256": tokenizer_contract_sha256,
            "role_view": view.to_dict(),
            "membership_receipt_sha256": membership,
            "guard_prompt_or_token_payload_opened": False,
        },
    )


class Gemma3L3L4ClaimGatedGuardPanelProvider(GemmaGuardPanelProvider):
    """Materialize exact guard tokens only after the durable claim exists."""

    def __init__(
        self,
        *,
        corpus: Gemma3L3L4ProgressiveACorpus,
        authority: Gemma3L3L4ProgressiveGuardClaimAuthority,
        tokenizer: object,
        max_length: int,
        device: torch.device,
        forbidden_manifest_sha256s: tuple[str, ...],
    ) -> None:
        if not isinstance(corpus, Gemma3L3L4ProgressiveACorpus):
            raise TypeError("corpus must be a progressive A corpus")
        if not isinstance(
            authority,
            Gemma3L3L4ProgressiveGuardClaimAuthority,
        ):
            raise TypeError("authority must be the durable guard authority")
        self._corpus = corpus
        self._authority = authority
        self._tokenizer = tokenizer
        self._max_length = _positive_int(
            max_length,
            label="max_length",
        )
        self._device = torch.device(device)
        self._forbidden = forbidden_manifest_sha256s
        view = corpus.preclaim_view("calibration_a_guard")
        self.manifest_sha256 = view.manifest_sha256
        self.example_count = view.example_count
        self.family_ids = view.family_ids
        self.membership_receipt_sha256 = (
            gemma_progressive_panel_membership_receipt_sha256(
                role=view.role,
                manifest_sha256=view.manifest_sha256,
                family_by_example=view.family_by_example,
            )
        )
        self.preclaim_binding_sha256 = (
            gemma3_l3_l4_guard_preclaim_binding_sha256(
                corpus_artifact_sha256=corpus.artifact.artifact_sha256,
                tokenizer_contract_sha256=(
                    corpus.artifact.tokenizer_contract_sha256
                ),
                view=view,
            )
        )
        self._opened = False

    def open_after_claim(self, claim_sha256: str) -> GemmaProgressivePanel:
        if self._opened:
            raise RuntimeError("guard provider is one use")
        receipt = self._authority.receipt
        if receipt is None or receipt.claim_sha256 != claim_sha256:
            raise RuntimeError(
                "durable guard receipt must exist before materialization"
            )
        # Consume the provider before any source read or tokenization.  A
        # later failure must not create a retry path.
        self._opened = True
        role_input = self._corpus.open_guard_after_claim(receipt)
        panel = materialize_gemma3_l3_l4_progressive_panel(
            tokenizer=self._tokenizer,
            role_input=role_input,
            view=self._corpus.preclaim_view("calibration_a_guard"),
            max_length=self._max_length,
            device=self._device,
            forbidden_manifest_sha256s=self._forbidden,
        )
        if panel.membership_receipt_sha256 != self.membership_receipt_sha256:
            raise RuntimeError(
                "claim-opened guard membership differs from preclaim"
            )
        return panel


def _development_corpus(
    corpus: Gemma3L3L4ProgressiveACorpus,
) -> DevelopmentCorpus:
    fit = corpus.preclaim_view("calibration_a_fit")
    selection = corpus.preclaim_view("calibration_a_selection")
    guard = corpus.preclaim_view("calibration_a_guard")
    return DevelopmentCorpus(
        corpus_id=corpus.artifact.corpus_id,
        fit_manifest_sha256=fit.manifest_sha256,
        selection_manifest_sha256=selection.manifest_sha256,
        guard_manifest_sha256=guard.manifest_sha256,
        fit_example_count=fit.example_count,
        selection_example_count=selection.example_count,
        guard_example_count=guard.example_count,
        fit_family_ids=fit.family_ids,
        selection_family_ids=selection.family_ids,
        guard_family_ids=guard.family_ids,
    )


def _sequence_scope_sha256(
    *,
    corpus: Gemma3L3L4ProgressiveACorpus,
    max_length: int,
) -> str:
    return _domain_sha256(
        _SEQUENCE_SCOPE_DOMAIN,
        {
            "format_version": 1,
            "corpus_artifact_sha256": corpus.artifact.artifact_sha256,
            "tokenizer_contract_sha256": (
                corpus.artifact.tokenizer_contract_sha256
            ),
            "maximum_sequence_length": max_length,
            "logical_macs_normalization": "per_valid_token_upper_bound",
        },
    )


def build_gemma3_l3_l4_progressive_resource_envelope(
    *,
    candidate_execution_sha256: str,
    sequence_scope_sha256: str,
    raw_model_resources: tuple[int, int, int],
    factorized_model_resources: tuple[int, int, int],
    bridge_float_count: int,
    bridge_integer_count: int,
    bridge_runtime_bytes: int,
    bridge_logical_macs_per_token: int,
    residual_width: int,
    source_modes: int,
    head_rank: int,
    max_residual_directions: int,
    lag_count: int,
    h4_conditioning: ResidualHeadConditioning = "l3_source_modes",
) -> tuple[ProgressiveResourceFootprint, ProgressiveResourceBudget]:
    """Build the incomplete seed receipt and preregistered joint-head budget."""

    raw_parameters, raw_bytes, raw_macs = raw_model_resources
    factor_parameters, factor_bytes, factor_macs = (
        factorized_model_resources
    )
    for label, value in (
        ("raw parameters", raw_parameters),
        ("raw bytes", raw_bytes),
        ("raw MACs", raw_macs),
        ("factorized parameters", factor_parameters),
        ("factorized bytes", factor_bytes),
        ("factorized MACs", factor_macs),
        ("bridge floats", bridge_float_count),
        ("bridge runtime bytes", bridge_runtime_bytes),
        ("bridge MACs", bridge_logical_macs_per_token),
        ("residual width", residual_width),
        ("source modes", source_modes),
        ("head rank", head_rank),
        ("maximum residual directions", max_residual_directions),
        ("lag count", lag_count),
    ):
        _positive_int(value, label=label)
    if type(bridge_integer_count) is not int or bridge_integer_count < 0:
        raise ValueError("bridge integer count must be nonnegative")
    if h4_conditioning not in (
        "l3_source_modes",
        "l3_source_modes_plus_realized_h4_decoder_modes_v1",
    ):
        raise ValueError("unsupported H4 conditioning")
    float_bytes = bridge_float_count * 8
    integer_bytes = bridge_runtime_bytes - float_bytes
    if integer_bytes < 0:
        raise ValueError("bridge runtime bytes undercount float storage")
    accounting = {
        "format_version": 1,
        "candidate_execution_sha256": candidate_execution_sha256,
        "sequence_scope_sha256": sequence_scope_sha256,
        "raw_source": {
            "learned_parameters": raw_parameters,
            "runtime_parameter_bytes": raw_bytes,
            "linear_weight_macs_per_token": raw_macs,
        },
        "factorized_carrier": {
            "learned_parameters": factor_parameters,
            "runtime_parameter_bytes": factor_bytes,
            "linear_weight_macs_per_token": factor_macs,
        },
        "compiled_bridge": {
            "prepared_float_scalar_count": bridge_float_count,
            "prepared_integer_value_count": bridge_integer_count,
            "runtime_parameter_bytes": bridge_runtime_bytes,
            "logical_macs_per_token_upper_bound": (
                bridge_logical_macs_per_token
            ),
        },
        "incomplete_cost_reasons": (
            LEGACY_RANK64_INCOMPLETE_COST_REASONS
        ),
    }
    seed = ProgressiveResourceFootprint(
        candidate_execution_sha256=candidate_execution_sha256,
        accounting_artifact_sha256=_domain_sha256(
            _RESOURCE_ACCOUNTING_DOMAIN,
            accounting,
        ),
        parameter_scope=GEMMA_TWO_HEAD_PARAMETER_SCOPE,
        compute_scope=GEMMA_TWO_HEAD_COMPUTE_SCOPE,
        runtime_id=GEMMA_TWO_HEAD_RUNTIME_ID,
        runtime_dtype=GEMMA_TWO_HEAD_RUNTIME_DTYPE,
        sequence_scope_sha256=sequence_scope_sha256,
        compiled_learned_parameters=bridge_float_count,
        retained_source_learned_parameters=factor_parameters,
        support_learned_parameters=0,
        compiled_runtime_parameter_bytes=float_bytes,
        retained_source_runtime_parameter_bytes=factor_bytes,
        support_runtime_parameter_bytes=integer_bytes,
        compiled_logical_macs_per_token=bridge_logical_macs_per_token,
        retained_source_logical_macs_per_token=factor_macs,
        support_logical_macs_per_token=0,
        cost_complete=False,
        incomplete_cost_reasons=LEGACY_RANK64_INCOMPLETE_COST_REASONS,
    )
    rank = min(head_rank, max_residual_directions, residual_width)
    floats_per_head = rank * (
        residual_width + lag_count * source_modes
    )
    conditioned_state_floats = (
        rank * rank
        if h4_conditioning
        == "l3_source_modes_plus_realized_h4_decoder_modes_v1"
        else 0
    )
    conditioned_state_macs = (
        residual_width * rank + rank * rank
        if conditioned_state_floats
        else 0
    )
    maximum_parameters = (
        factor_parameters
        + bridge_float_count
        + 2 * floats_per_head
        + conditioned_state_floats
    )
    maximum_bytes = (
        factor_bytes
        + bridge_runtime_bytes
        + 16 * floats_per_head
        + 8 * conditioned_state_floats
    )
    maximum_macs = (
        factor_macs
        + bridge_logical_macs_per_token
        + 2 * floats_per_head
        + conditioned_state_macs
    )

    def upper_ratio(numerator: int, denominator: int) -> float:
        return math.nextafter(numerator / denominator, math.inf)

    budget = ProgressiveResourceBudget(
        parameter_scope=GEMMA_TWO_HEAD_PARAMETER_SCOPE,
        compute_scope=GEMMA_TWO_HEAD_COMPUTE_SCOPE,
        runtime_id=GEMMA_TWO_HEAD_RUNTIME_ID,
        runtime_dtype=GEMMA_TWO_HEAD_RUNTIME_DTYPE,
        sequence_scope_sha256=sequence_scope_sha256,
        source_learned_parameters=raw_parameters,
        source_runtime_parameter_bytes=raw_bytes,
        source_logical_macs_per_token=raw_macs,
        max_total_parameter_fraction=upper_ratio(
            maximum_parameters,
            raw_parameters,
        ),
        max_total_parameter_byte_fraction=upper_ratio(
            maximum_bytes,
            raw_bytes,
        ),
        max_total_mac_fraction=upper_ratio(
            maximum_macs,
            raw_macs,
        ),
        max_retained_source_parameter_fraction=upper_ratio(
            factor_parameters,
            raw_parameters,
        ),
        max_retained_source_parameter_byte_fraction=upper_ratio(
            factor_bytes,
            raw_bytes,
        ),
        max_retained_source_mac_fraction=upper_ratio(
            factor_macs,
            raw_macs,
        ),
    )
    return seed, budget


def _h4_conditioning_contract(
    *,
    h4_conditioning: ResidualHeadConditioning,
    residual_width: int,
    head_rank: int,
    max_residual_directions: int,
) -> dict[str, object]:
    effective_rank = min(
        head_rank,
        max_residual_directions,
        residual_width,
    )
    state_conditioned = (
        h4_conditioning
        == "l3_source_modes_plus_realized_h4_decoder_modes_v1"
    )
    return {
        "input_signal": (
            "archived_l3_source_modes_only"
            if not state_conditioned
            else (
                "archived_l3_source_modes_plus_current_realized_"
                "post_x4_h4_projected_on_the_output_decoder"
            )
        ),
        "feature_is_pointwise": True,
        "future_positions_read": False,
        "output_decoder_reused_as_input_encoder": state_conditioned,
        "additional_stored_projector_parameters": 0,
        "effective_state_rank": effective_rank if state_conditioned else 0,
        "additional_state_kernel_parameters": (
            effective_rank * effective_rank
            if state_conditioned
            else 0
        ),
        "serving_model_forward_count": 1,
    }


def _campaign_spec(
    *,
    corpus: Gemma3L3L4ProgressiveACorpus,
    protocol: ProgressiveCompilationProtocol,
    max_length: int,
    device_name: str,
    dtype: str,
    residual_width: int,
    head_rank: int,
    lag_count: int,
    ridge: float,
    h4_fit_objective: ResidualHeadFitObjective,
    h4_conditioning: ResidualHeadConditioning,
    max_residual_directions: int,
    max_iterations: int,
    defer_guard: bool,
    adaptive_parent_lineage: Mapping[str, object] | None,
) -> dict[str, object]:
    protocol_metadata = protocol.metadata()
    fidelity = _mapping(
        protocol_metadata["fidelity_targets"],
        label="progressive fidelity policy",
    )
    loop = _mapping(
        protocol_metadata["loop"],
        label="progressive loop policy",
    )
    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "corpus_artifact_sha256": corpus.artifact.artifact_sha256,
        "tokenizer_contract_sha256": (
            corpus.artifact.tokenizer_contract_sha256
        ),
        "maximum_sequence_length": max_length,
        "device": device_name,
        "dtype": dtype,
        "head_rank": head_rank,
        "lag_count": lag_count,
        "ridge": ridge,
        "h4_fit_objective": h4_fit_objective,
        "h4_fit_objective_contract": {
            "input_signal": (
                "x4_candidate_conditioned_summed_nll_vjp_at_h4"
                if h4_fit_objective
                == "candidate_nll_vjp_metric_ridge_v1"
                else (
                    "source_native_summed_nll_vjp_at_h4"
                    if h4_fit_objective
                    == "source_nll_vjp_metric_ridge_v1"
                    else "none_hidden_residual_only"
                )
            ),
            "finite_displacement": (
                "native_h4_minus_x4_conditioned_candidate_h4"
            ),
            "metric": (
                "identity_plus_unit_nll_vjp_outer_product"
                if h4_fit_objective
                in (
                    "source_nll_vjp_metric_ridge_v1",
                    "candidate_nll_vjp_metric_ridge_v1",
                )
                else "identity"
            ),
            "gradient_normalization": (
                "not_applicable"
                if h4_fit_objective == "hidden_residual_ridge"
                else "per_row_unit_norm_with_sqrt_eps_global_floor"
            ),
            "family_example_balancing_preserved": True,
            "serving_shape_changes": False,
        },
        "h4_conditioning": h4_conditioning,
        "h4_conditioning_contract": _h4_conditioning_contract(
            h4_conditioning=h4_conditioning,
            residual_width=residual_width,
            head_rank=head_rank,
            max_residual_directions=max_residual_directions,
        ),
        "maximum_residual_directions": max_residual_directions,
        "maximum_iterations": max_iterations,
        "candidate_schedule": "forced_x4_then_remeasure_then_h4",
        "adaptive_policy_id": "staged-execution-fidelity-v2",
        "protocol_sha256": protocol_metadata["artifact_sha256"],
        "acceptance_policy": fidelity["acceptance"],
        "staging_transitions": loop["staging_transitions"],
        "adaptive_parent_lineage": adaptive_parent_lineage,
        "compact_after_fidelity": False,
        "guard_policy": (
            "deferred_after_frozen_selection"
            if defer_guard
            else "claim_once_on_eligible_frozen_challenger"
        ),
        "calibration_b_authorized": False,
    }


def _adaptive_parent_lineage(
    path: Path | str | None,
    *,
    expected_fixed_controls: Mapping[str, object] | None = None,
    expected_parent_conditioning: ResidualHeadConditioning | None = None,
) -> dict[str, object] | None:
    if path is None:
        return None
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = _mapping(payload, label="adaptive parent report")
    safety = _mapping(
        report.get("safety"),
        label="adaptive parent safety",
    )
    if safety.get("calibration_b_opened") is not False:
        raise ValueError("adaptive parent must attest Calibration B closed")
    parent_spec = _mapping(
        report.get("campaign_spec"),
        label="adaptive parent campaign spec",
    )
    parent_conditioning = parent_spec.get(
        "h4_conditioning",
        "l3_source_modes",
    )
    if expected_fixed_controls is not None:
        observed_controls = {
            name: parent_spec.get(name)
            for name in expected_fixed_controls
        }
        if observed_controls != dict(expected_fixed_controls):
            raise ValueError(
                "adaptive parent fixed controls differ from this "
                "conditioning experiment"
            )
    if (
        expected_parent_conditioning is not None
        and parent_conditioning != expected_parent_conditioning
    ):
        raise ValueError(
            "adaptive parent conditioning is not the required baseline"
        )
    return {
        "report_file_sha256": _file_sha256(source),
        "report_sha256": report.get("report_sha256"),
        "transcript_sha256": report.get("transcript_sha256"),
        "protocol_sha256": report.get("protocol_sha256"),
        "campaign_spec_sha256": report.get("campaign_spec_sha256"),
        "fixed_control_snapshot": (
            None
            if expected_fixed_controls is None
            else dict(expected_fixed_controls)
        ),
        "parent_h4_conditioning": parent_conditioning,
        "guard_opened": safety.get("guard_opened"),
        "guard_consumed": safety.get("guard_consumed"),
        "role": "adaptive_development_parent",
    }


def _evaluation_qualification(
    *,
    protocol: ProgressiveCompilationProtocol,
    evaluation: CandidateEvaluation,
) -> dict[str, object]:
    targets = protocol.fidelity_targets
    execution = targets.execution_fidelity_ratios(evaluation.fidelity)
    structural = targets.structural_diagnostic_ratios(
        evaluation.fidelity
    )
    return {
        "candidate_receipt_sha256": evaluation.candidate_receipt_sha256,
        "evaluation_receipt_sha256": evaluation.receipt_sha256,
        "role": evaluation.development_role,
        "qualification_claim": "candidate_execution_fidelity_only",
        "execution_fidelity_passed": (
            targets.passes_execution_fidelity(evaluation.fidelity)
        ),
        "execution_max_normalized_burden": max(execution.values()),
        "execution_failed_axes": tuple(
            name for name, value in execution.items() if value > 1.0
        ),
        "structural_max_normalized_burden": (
            0.0 if not structural else max(structural.values())
        ),
        "structural_failed_axes": tuple(
            name for name, value in structural.items() if value > 1.0
        ),
        "structural_failures_are_diagnostic": True,
    }


def _source_code_sha256s() -> dict[str, str]:
    root = Path(__file__).parent
    names = (
        "gemma3_l3_l4_progressive_a_campaign.py",
        "gemma3_l3_l4_progressive_a_corpus.py",
        "gemma3_l3_l4_progressive_guard_ledger.py",
        "gemma3_l3_l4_progressive_worker.py",
        "gemma3_l3_l4_two_head_lowerer.py",
        "gemma3_l3_l4_graph_organized_svd_shadow_runtime.py",
        "gemma3_l3_l4_progressive_compilation.py",
        "causal_edge_jvp.py",
        "conditional_quadratic_edge.py",
        "radial_finite_displacement_correction.py",
        "compiler/calibration.py",
        "compiler/progressive.py",
    )
    return {
        name: _file_sha256(root / name)
        for name in names
    }


def _publish_report(
    *,
    output: Path,
    report: Mapping[str, object],
    candidate_state: Mapping[str, object] | None,
) -> dict[str, object]:
    """Publish tensor-free JSON and optional candidate state without overwrite."""

    if output.suffix != ".json":
        raise ValueError("campaign report output must end in .json")
    candidate_output = output.with_suffix(".candidate.pt")
    if output.exists() or (
        candidate_state is not None and candidate_output.exists()
    ):
        raise FileExistsError("refusing to overwrite campaign output")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_stage: Path | None = None
    candidate_stage: Path | None = None
    published: list[Path] = []
    try:
        artifact_record: dict[str, object] = {
            "candidate_tensor_file": None,
            "candidate_tensor_file_sha256": None,
            "candidate_tensor_file_bytes": 0,
            "contains_model_weights": False,
            "committable": False,
        }
        if candidate_state is not None:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{candidate_output.name}.",
                suffix=".tmp",
                dir=output.parent,
            )
            os.close(descriptor)
            candidate_stage = Path(name)
            torch.save(dict(candidate_state), candidate_stage)
            artifact_record = {
                "candidate_tensor_file": str(candidate_output),
                "candidate_tensor_file_sha256": _file_sha256(
                    candidate_stage
                ),
                "candidate_tensor_file_bytes": (
                    candidate_stage.stat().st_size
                ),
                "contains_model_weights": False,
                "committable": False,
            }
        payload = {
            **dict(report),
            "artifact": artifact_record,
        }
        payload["report_sha256"] = _domain_sha256(
            _REPORT_DOMAIN,
            payload,
        )
        descriptor, name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        os.close(descriptor)
        report_stage = Path(name)
        encoded = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        with report_stage.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if candidate_stage is not None:
            os.link(candidate_stage, candidate_output)
            published.append(candidate_output)
        os.link(report_stage, output)
        published.append(output)
        directory = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return payload
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)
        if candidate_stage is not None:
            candidate_stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_progressive_a_campaign(
    *,
    corpus_artifact_path: Path | str = (
        DEFAULT_PROGRESSIVE_A_CORPUS_ARTIFACT
    ),
    fit_input_path: Path | str = DEFAULT_PROGRESSIVE_A_FIT_INPUT,
    selection_input_path: Path | str = (
        DEFAULT_PROGRESSIVE_A_SELECTION_INPUT
    ),
    guard_input_path: Path | str = DEFAULT_PROGRESSIVE_A_GUARD_INPUT,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    adaptive_parent_report_path: Path | str | None = None,
    output: Path | str = DEFAULT_PROGRESSIVE_A_REPORT,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    head_rank: int = 8,
    lag_count: int = 32,
    ridge: float = 1.0e-6,
    h4_fit_objective: ResidualHeadFitObjective = (
        "hidden_residual_ridge"
    ),
    h4_conditioning: ResidualHeadConditioning = "l3_source_modes",
    max_residual_directions: int = 8,
    max_iterations: int = 3,
    defer_guard: bool = False,
) -> dict[str, object]:
    """Run a complete family-disjoint A campaign without touching B."""

    for label, value in (
        ("head_rank", head_rank),
        ("lag_count", lag_count),
        ("max_residual_directions", max_residual_directions),
        ("max_iterations", max_iterations),
    ):
        _positive_int(value, label=label)
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    if h4_fit_objective not in (
        "hidden_residual_ridge",
        "source_nll_vjp_metric_ridge_v1",
        "candidate_nll_vjp_metric_ridge_v1",
    ):
        raise ValueError("unsupported H4 fit objective")
    if h4_conditioning not in (
        "l3_source_modes",
        "l3_source_modes_plus_realized_h4_decoder_modes_v1",
    ):
        raise ValueError("unsupported H4 conditioning")
    if type(defer_guard) is not bool:
        raise TypeError("defer_guard must be a bool")
    destination = Path(output)
    if destination.exists() or destination.with_suffix(
        ".candidate.pt"
    ).exists():
        raise FileExistsError("refusing to overwrite campaign output")

    legacy_protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    legacy_protocol.validate_integrity()
    legacy_metadata = legacy_protocol.metadata()
    tokenizer_contract = dict(
        _mapping(
            legacy_metadata["tokenizer"],
            label="frozen tokenizer",
        )
    )
    max_length = int(tokenizer_contract["max_length"])
    role_paths = {
        "calibration_a_fit": Path(fit_input_path),
        "calibration_a_selection": Path(selection_input_path),
        "calibration_a_guard": Path(guard_input_path),
    }
    corpus = load_gemma3_l3_l4_progressive_a_corpus(
        corpus_artifact_path,
        role_input_paths=role_paths,  # type: ignore[arg-type]
        tokenizer_contract=tokenizer_contract,
    )
    forbidden = corpus.artifact.forbidden_assessment_manifest_sha256s
    if forbidden != (
        str(
            _mapping(
                _mapping(
                    legacy_metadata["corpus"],
                    label="legacy corpus",
                )["calibration_b_manifest"],
                label="legacy B manifest",
            )["artifact_sha256"]
        ),
    ):
        raise ValueError("corpus forbidden identity differs from frozen B")

    conditioning_comparison = (
        h4_conditioning
        == "l3_source_modes_plus_realized_h4_decoder_modes_v1"
    )
    adaptive_parent = _adaptive_parent_lineage(
        adaptive_parent_report_path,
        expected_fixed_controls=(
            {
                "corpus_artifact_sha256": (
                    corpus.artifact.artifact_sha256
                ),
                "maximum_sequence_length": max_length,
                "device": device_name,
                "dtype": dtype,
                "head_rank": head_rank,
                "lag_count": lag_count,
                "ridge": ridge,
                "h4_fit_objective": h4_fit_objective,
                "maximum_residual_directions": max_residual_directions,
                "maximum_iterations": max_iterations,
                "candidate_schedule": (
                    "forced_x4_then_remeasure_then_h4"
                ),
                "guard_policy": (
                    "deferred_after_frozen_selection"
                    if defer_guard
                    else (
                        "claim_once_on_eligible_frozen_challenger"
                    )
                ),
            }
            if conditioning_comparison
            else None
        ),
        expected_parent_conditioning=(
            "l3_source_modes"
            if conditioning_comparison
            else None
        ),
    )
    code_sha256s_before = _source_code_sha256s()

    tokenizer, live_tokenizer_contract = (
        _load_and_validate_frozen_local_tokenizer(
            protocol=legacy_protocol,
        )
    )
    if _canonical_json_bytes(live_tokenizer_contract) != (
        _canonical_json_bytes(tokenizer_contract)
    ):
        raise ValueError("live tokenizer differs from the campaign contract")

    fit_input = corpus.open_development_role("calibration_a_fit")
    selection_input = corpus.open_development_role(
        "calibration_a_selection"
    )
    token_device = torch.device(str(tokenizer_contract["device"]))
    fit_panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=tokenizer,
        role_input=fit_input,
        view=corpus.preclaim_view("calibration_a_fit"),
        max_length=max_length,
        device=token_device,
        forbidden_manifest_sha256s=forbidden,
    )
    selection_panel = materialize_gemma3_l3_l4_progressive_panel(
        tokenizer=tokenizer,
        role_input=selection_input,
        view=corpus.preclaim_view("calibration_a_selection"),
        max_length=max_length,
        device=token_device,
        forbidden_manifest_sha256s=forbidden,
    )
    if corpus.guard_opened or corpus.guard_consumed:
        raise RuntimeError("guard was opened before candidate freeze")

    model_metadata = _mapping(
        legacy_metadata["model"],
        label="legacy model",
    )
    graph_binding = _mapping(
        legacy_metadata["graph_candidate"],
        label="legacy graph candidate",
    )
    basis_binding = _mapping(
        legacy_metadata["prompt_blind_basis"],
        label="legacy basis",
    )
    runtime_binding = _mapping(
        legacy_metadata["runtime_binding_contract"],
        label="legacy runtime binding",
    )
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype=dtype,
    )
    adapter = Gemma3CausalLMAdapter(model)
    raw_model_sha256 = adapter.model_fingerprint()
    if raw_model_sha256 != str(model_metadata["source_model_sha256"]):
        raise ValueError("live raw Gemma differs from the frozen source")
    raw_resources = _model_resources(adapter.module)

    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_FACTORIZED_SCOPE: catalog.replacements},
    )
    result: ProgressiveCompilationResult
    handoff: FrozenProgressiveCandidateHandoff | None = None
    candidate_state: Mapping[str, object] | None = None
    head_fit_audit: list[dict[str, object]] = []
    authority = (
        None
        if defer_guard
        else Gemma3L3L4ProgressiveGuardClaimAuthority()
    )
    try:
        switcher.switch(_FACTORIZED_SCOPE)
        factorized_model_sha256 = adapter.model_fingerprint()
        factorized_execution_sha256 = adapter.execution_fingerprint()
        if (
            factorized_model_sha256
            != str(graph_binding["factorized_live_execution_sha256"])
            or factorized_execution_sha256
            != str(graph_binding["factorized_refit_execution_sha256"])
        ):
            raise ValueError(
                "live factorized Gemma differs from the frozen execution"
            )
        factorized_resources = _model_resources(adapter.module)
        candidate_path = Path(graph_candidate_path)
        basis_path = Path(basis_package_path)
        graph_candidate = load_gemma3_graph_organized_svd_candidate(
            candidate_path,
            expected_file_sha256=str(
                graph_binding["tensor_file_sha256"]
            ),
        )
        basis = load_gemma3_l3_l4_basis_package(
            basis_path,
            expected_file_sha256=str(
                basis_binding["tensor_file_sha256"]
            ),
            expected_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
        )
        runtime = Gemma3L3L4GraphOrganizedSVDShadowRuntime(
            graph_candidate,
            basis,
            expected_candidate_artifact_sha256=str(
                graph_binding["logical_artifact_sha256"]
            ),
            expected_basis_payload_sha256=str(
                basis_binding["logical_payload_sha256"]
            ),
            expected_plan_artifact_sha256=str(
                graph_binding["deployment_plan_sha256"]
            ),
            expected_live_model_sha256=str(
                graph_binding["factorized_live_execution_sha256"]
            ),
            expected_adapter_execution_sha256=str(
                graph_binding["factorized_refit_execution_sha256"]
            ),
            analysis_device="cpu",
        )
        # The progressive worker has a larger authenticated observation ABI
        # than the frozen Calibration-B runtime contract.  Its runtime receipt
        # is therefore expected to differ while the candidate, basis, plan,
        # live model, and adapter execution identities above remain pinned.
        legacy_runtime_binding_sha256 = str(
            runtime_binding["artifact_sha256"]
        )
        seed_executable = LegacyRank64GemmaProgressiveExecutable(
            adapter=adapter,
            runtime=runtime,
            candidate_execution_sha256=factorized_execution_sha256,
        )
        bridge = runtime.export_one_pass_bridge()
        sequence_scope = _sequence_scope_sha256(
            corpus=corpus,
            max_length=max_length,
        )
        seed_resources, resource_budget = (
            build_gemma3_l3_l4_progressive_resource_envelope(
                candidate_execution_sha256=factorized_execution_sha256,
                sequence_scope_sha256=sequence_scope,
                raw_model_resources=raw_resources,
                factorized_model_resources=factorized_resources,
                bridge_float_count=bridge.prepared_float_scalar_count,
                bridge_integer_count=bridge.prepared_integer_value_count,
                bridge_runtime_bytes=(
                    bridge.prepared_runtime_parameter_bytes
                ),
                bridge_logical_macs_per_token=(
                    bridge.logical_macs_per_token_upper_bound
                ),
                residual_width=bridge.residual_width,
                source_modes=bridge.source_modes,
                head_rank=head_rank,
                max_residual_directions=max_residual_directions,
                lag_count=lag_count,
                h4_conditioning=h4_conditioning,
            )
        )
        seed = current_gemma3_l3_l4_progressive_seed(
            resources=seed_resources,
            runtime_binding_sha256=runtime.runtime_binding_sha256,
        )
        guard_view = corpus.preclaim_view("calibration_a_guard")
        guard_preclaim_binding = (
            gemma3_l3_l4_guard_preclaim_binding_sha256(
                corpus_artifact_sha256=(
                    corpus.artifact.artifact_sha256
                ),
                tokenizer_contract_sha256=(
                    corpus.artifact.tokenizer_contract_sha256
                ),
                view=guard_view,
            )
        )
        protocol = make_gemma3_l3_l4_progressive_protocol(
            corpus=_development_corpus(corpus),
            seed_runtime_binding_sha256=runtime.runtime_binding_sha256,
            fit_panel_binding_sha256=fit_panel.binding_sha256,
            selection_panel_binding_sha256=(
                selection_panel.binding_sha256
            ),
            guard_preclaim_binding_sha256=guard_preclaim_binding,
            resource_budget=resource_budget,
            seed_resources=seed_resources,
            max_iterations=max_iterations,
            residual_head_rank=head_rank,
            compact_after_fidelity=False,
        )
        campaign_spec = _campaign_spec(
            corpus=corpus,
            protocol=protocol,
            max_length=max_length,
            device_name=device_name,
            dtype=dtype,
            residual_width=bridge.residual_width,
            head_rank=head_rank,
            lag_count=lag_count,
            ridge=ridge,
            h4_fit_objective=h4_fit_objective,
            h4_conditioning=h4_conditioning,
            max_residual_directions=max_residual_directions,
            max_iterations=max_iterations,
            defer_guard=defer_guard,
            adaptive_parent_lineage=adaptive_parent,
        )
        campaign_spec_sha256 = _domain_sha256(
            _CAMPAIGN_DOMAIN,
            campaign_spec,
        )
        guard_provider = (
            None
            if defer_guard
            else Gemma3L3L4ClaimGatedGuardPanelProvider(
                corpus=corpus,
                authority=authority,
                tokenizer=tokenizer,
                max_length=max_length,
                device=token_device,
                forbidden_manifest_sha256s=forbidden,
            )
        )
        lowerer = GemmaL3L4TwoHeadMutationLowerer(
            adapter=adapter,
            shadow_runtime=runtime,
            source_probe=seed_executable,
            head_rank=head_rank,
            lag_count=lag_count,
            ridge=ridge,
            h4_fit_objective=h4_fit_objective,
            h4_conditioning=h4_conditioning,
            proposal_schedule="x4_then_h4",
        )
        worker = Gemma3L3L4ProgressiveWorker(
            protocol=protocol,
            panels={
                "calibration_a_fit": fit_panel,
                "calibration_a_selection": selection_panel,
            },
            seed_candidate=seed,
            seed_executable=seed_executable,
            max_residual_directions=max_residual_directions,
            mutation_lowerer=lowerer,
            guard_claim_authority=(
                None if defer_guard else authority
            ),
            guard_panel_provider=guard_provider,
            selection_only=defer_guard,
        )
        seed_selection = worker.evaluate_selection(
            seed,
            protocol.selection_view(),
        )
        if defer_guard:
            result = run_progressive_development_search(
                protocol=protocol,
                seed_candidate=seed,
                seed_selection_evaluation=seed_selection,
                map_residual=worker.map_residual,
                propose_mutations=worker.propose_mutations,
                build_candidate=worker.build_candidate,
                evaluate_selection=worker.evaluate_selection,
            )
        else:
            result = run_progressive_compilation(
                protocol=protocol,
                seed_candidate=seed,
                seed_selection_evaluation=seed_selection,
                map_residual=worker.map_residual,
                propose_mutations=worker.propose_mutations,
                build_candidate=worker.build_candidate,
                evaluate_selection=worker.evaluate_selection,
                evaluate_guard=worker.evaluate_guard,
            )
        result.validate_against(protocol)
        if result.status == "ready_for_candidate_binding":
            handoff = freeze_progressive_candidate(
                protocol=protocol,
                result=result,
            )
        artifact_for = getattr(lowerer, "artifact_for", None)
        if callable(artifact_for):
            for archived_candidate in result.candidate_archive:
                if archived_candidate.mutation_kind == "seed":
                    continue
                archived_artifact = artifact_for(archived_candidate)
                head_fit_audit.append(
                    {
                        "candidate_id": archived_candidate.candidate_id,
                        "candidate_artifact_sha256": (
                            archived_candidate.artifact_sha256
                        ),
                        "candidate_receipt_sha256": (
                            archived_candidate.receipt_sha256
                        ),
                        "selected_as_final": (
                            archived_candidate.receipt_sha256
                            == result.final_candidate.receipt_sha256
                        ),
                        "heads": tuple(
                            {
                                "site": head.site,
                                "fit_objective": head.fit_objective,
                                "conditioning": head.conditioning,
                                "rank": head.rank,
                                "source_rank": head.source_rank,
                                "state_rank": int(
                                    head.state_kernel.shape[0]
                                ),
                                "state_kernel_scalar_count": int(
                                    head.state_kernel.numel()
                                ),
                                "lag_count": head.lag_count,
                                "fit_row_count": head.fit_row_count,
                                "weighted_residual_rmse": (
                                    head.weighted_residual_rmse
                                ),
                                "normalized_nll_direction_rmse": (
                                    head.normalized_nll_direction_rmse
                                ),
                                "linearized_nll_residual_rmse": (
                                    head.linearized_nll_residual_rmse
                                ),
                                "prepared_float_scalar_count": (
                                    head.prepared_float_scalar_count
                                ),
                                "logical_macs_per_token_upper_bound": (
                                    head.logical_macs_per_token_upper_bound
                                ),
                                "artifact_sha256": head.artifact_sha256,
                            }
                            for head in archived_artifact.heads
                        ),
                    }
                )
                if (
                    archived_candidate.receipt_sha256
                    == result.final_candidate.receipt_sha256
                ):
                    state_dict = getattr(
                        archived_artifact,
                        "state_dict",
                        None,
                    )
                    if callable(state_dict):
                        candidate_state = state_dict()
        if corpus.guard_opened != (result.guard_evaluation is not None):
            raise RuntimeError("corpus guard state differs from result")
        receipt = None if authority is None else authority.receipt
        if (
            result.guard_evaluation is not None
            and (
                receipt is None
                or result.guard_evaluation.guard_claim_sha256
                != receipt.claim_sha256
            )
        ):
            raise RuntimeError("guard evaluation lacks its durable receipt")

        code_sha256s_after = _source_code_sha256s()
        if code_sha256s_after != code_sha256s_before:
            raise RuntimeError("campaign source code changed during execution")
        if (
            adapter.model_fingerprint() != factorized_model_sha256
            or adapter.execution_fingerprint()
            != factorized_execution_sha256
            or _file_sha256(candidate_path)
            != str(graph_binding["tensor_file_sha256"])
            or _file_sha256(basis_path)
            != str(basis_binding["tensor_file_sha256"])
        ):
            raise RuntimeError(
                "model execution or frozen artifacts changed during campaign"
            )

        guard_receipt = None if receipt is None else receipt.metadata()
        report = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "campaign_spec": campaign_spec,
            "campaign_spec_sha256": campaign_spec_sha256,
            "protocol": protocol.metadata(),
            "protocol_sha256": protocol.artifact_sha256,
            "corpus": corpus.artifact.to_dict(),
            "tokenizer_contract_sha256": (
                corpus.artifact.tokenizer_contract_sha256
            ),
            "panel_receipts": {
                "fit_binding_sha256": fit_panel.binding_sha256,
                "selection_binding_sha256": (
                    selection_panel.binding_sha256
                ),
                "guard_preclaim_binding_sha256": (
                    guard_preclaim_binding
                ),
            },
            "resource_accounting": {
                "raw_source": {
                    "learned_parameters": raw_resources[0],
                    "runtime_parameter_bytes": raw_resources[1],
                    "linear_weight_macs_per_token": raw_resources[2],
                },
                "factorized_carrier": {
                    "learned_parameters": factorized_resources[0],
                    "runtime_parameter_bytes": factorized_resources[1],
                    "linear_weight_macs_per_token": (
                        factorized_resources[2]
                    ),
                },
                "seed": seed_resources.to_dict(),
                "budget": resource_budget.to_dict(),
            },
            "result": result.to_dict(),
            "qualification": {
                "claim": "candidate_execution_fidelity_only",
                "full_structural_equivalence_claim": False,
                "selection_archive": tuple(
                    _evaluation_qualification(
                        protocol=protocol,
                        evaluation=evaluation,
                    )
                    for evaluation
                    in result.selection_evaluation_archive
                ),
                "guard": (
                    None
                    if result.guard_evaluation is None
                    else _evaluation_qualification(
                        protocol=protocol,
                        evaluation=result.guard_evaluation,
                    )
                ),
            },
            "head_fit_audit": tuple(head_fit_audit),
            "transcript_sha256": result.transcript_sha256,
            "guard_claim_receipt": guard_receipt,
            "handoff": None if handoff is None else handoff.to_dict(),
            "handoff_receipt_sha256": (
                None if handoff is None else handoff.receipt_sha256
            ),
            "bindings": {
                "legacy": (
                    gemma3_l3_l4_legacy_progressive_binding_metadata()
                ),
                "raw_model_sha256": raw_model_sha256,
                "factorized_model_sha256": factorized_model_sha256,
                "factorized_execution_sha256": (
                    factorized_execution_sha256
                ),
                "progressive_runtime_binding_sha256": (
                    runtime.runtime_binding_sha256
                ),
                "legacy_b_runtime_binding_sha256": (
                    legacy_runtime_binding_sha256
                ),
                "graph_candidate_file_sha256": _file_sha256(
                    candidate_path
                ),
                "basis_file_sha256": _file_sha256(basis_path),
                "base_artifact_file_sha256": _file_sha256(
                    base_artifact_path
                ),
                "refit_artifact_file_sha256": _file_sha256(
                    refit_artifact_path
                ),
                "source_code_sha256s": code_sha256s_before,
            },
            "safety": {
                "calibration_b_opened": False,
                "calibration_b_loader_present": False,
                "guard_opened": corpus.guard_opened,
                "guard_consumed": corpus.guard_consumed,
                "guard_deferred": defer_guard,
                "guard_capability_constructed": authority is not None,
                "prompt_text_in_report": False,
                "token_ids_in_report": False,
                "activation_rows_in_report": False,
                "gradient_rows_in_report": False,
                "fit_sequences_in_report": False,
                "model_weights_in_candidate_artifact": False,
                "development_only": True,
                "compression_claim": False,
                "latency_claim": False,
                "full_structural_equivalence_claim": False,
            },
        }
        return _publish_report(
            output=destination,
            report=report,
            candidate_state=candidate_state,
        )
    finally:
        switcher.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_CORPUS_ARTIFACT,
    )
    parser.add_argument(
        "--fit-input",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_FIT_INPUT,
    )
    parser.add_argument(
        "--selection-input",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_SELECTION_INPUT,
    )
    parser.add_argument(
        "--guard-input",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_GUARD_INPUT,
    )
    parser.add_argument(
        "--graph-candidate",
        type=Path,
        default=DEFAULT_GRAPH_CANDIDATE,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument(
        "--adaptive-parent-report",
        type=Path,
        help="Prior development report to bind as adaptive lineage",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROGRESSIVE_A_REPORT,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--head-rank", type=int, default=8)
    parser.add_argument("--lag-count", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument(
        "--h4-fit-objective",
        choices=(
            "hidden_residual_ridge",
            "source_nll_vjp_metric_ridge_v1",
            "candidate_nll_vjp_metric_ridge_v1",
        ),
        default="hidden_residual_ridge",
    )
    parser.add_argument(
        "--h4-conditioning",
        choices=(
            "l3_source_modes",
            "l3_source_modes_plus_realized_h4_decoder_modes_v1",
        ),
        default="l3_source_modes",
    )
    parser.add_argument(
        "--max-residual-directions",
        type=int,
        default=8,
    )
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--defer-guard",
        action="store_true",
        help=(
            "Run fit/selection only and freeze an eligible challenger "
            "without constructing a guard capability"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_progressive_a_campaign(
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        selection_input_path=args.selection_input,
        guard_input_path=args.guard_input,
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        adaptive_parent_report_path=args.adaptive_parent_report,
        output=args.output,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        head_rank=args.head_rank,
        lag_count=args.lag_count,
        ridge=args.ridge,
        h4_fit_objective=args.h4_fit_objective,
        h4_conditioning=args.h4_conditioning,
        max_residual_directions=args.max_residual_directions,
        max_iterations=args.max_iterations,
        defer_guard=args.defer_guard,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
