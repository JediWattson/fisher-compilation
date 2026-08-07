"""Leakage-safe nested ridge selection for A5b coordinate generators.

A5b supplies one authenticated row bank from the seven families available to
an outer fold.  This module performs the next, purely in-memory step:

* each of those seven families is held out once;
* rank-16 frozen-basis generators are refit on the other six families for
  every ridge candidate;
* the held family is scored through the adapter's *final* normalization and
  language-model head against the native block state;
* the exact frozen compiled block state is scored as the fallback baseline;
* after ridge selection, a winning candidate is refit on all seven families.

The post-CV fit uses every outer-training example.  Because the historical
generator fitter requires a disjoint evaluation key set, its descriptive
evaluation is a re-keyed replay of ``bridge.audit_rows``.  It is explicitly
recorded as non-independent and is never used for ridge selection.

All row, state, logit, and fitted-factor tensors remain ephemeral.  The public
receipt contains only scalar diagnostics, counts, booleans, and hashes.  The
returned object separately retains the selected executable fit (or represents
the frozen fallback) so a runner can execute the decision without refitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from .adapters import SequenceContext, SequenceInputOrigin
from .gemma3_l10_l17_trajectory_correction_fitting import (
    FrozenBasisGeneratorFit,
    fit_frozen_basis_coordinate_generators,
)
from .gemma3_modal_generator_dev_experiment import LayerFragmentRows
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering
from .modal_generators import apply_modal_generator


__all__ = [
    "A5C_FIXED_GENERATOR_RANK",
    "A5C_RIDGE_GRID",
    "GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_SCHEMA",
    "A5cAuthenticatedRowBank",
    "A5cFamilyRidgeCvSelection",
    "select_a5c_family_disjoint_ridge",
    "validate_a5c_family_ridge_cv_receipt",
]


GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5c_family_ridge_cv"
)
GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_FORMAT_VERSION = 1
A5C_FIXED_GENERATOR_RANK = 16
A5C_RIDGE_GRID = (0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0)

_INNER_FOLD_COUNT = 7
_INNER_TRAINING_FAMILY_COUNT = 6
_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-family-ridge-cv:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-tensor:v1\0"
_MEMBERSHIP_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-membership:v1\0"
_SPLIT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5c-split:v1\0"
_DESCRIPTIVE_KEY_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-descriptive-key:v1\0"
)
_ROW_SIGNATURE_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-input-row-signature:v1\0"
)
_INPUT_CAST_LINEAGE_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5c-runtime-input-cast:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_GENERATOR_RUNTIME_DTYPE = torch.float32
_INPUT_CAST_ATOL = float(torch.finfo(torch.float32).eps)
_INPUT_CAST_RTOL = float(torch.finfo(torch.float32).eps)
_SELECTION_OBJECTIVE = (
    "seven_family_equal_mean_fisher_weighted_exact_float64_"
    "full_vocabulary_native_to_candidate_final_head_kl"
)
_STATE_DIAGNOSTIC = (
    "seven_family_equal_mean_fisher_weighted_block_state_nrmse_to_native"
)
_FALLBACK_RULE = (
    "use_frozen_compiled_block_state_unless_best_ridge_strictly_reduces_"
    "the_selection_objective"
)
_SAFETY = {
    "contains_tensors": False,
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "outer_held_family_rows_accepted": False,
    "outer_held_family_accessed": False,
    "heldout_confirmation": False,
    "serving_authorized": False,
}


@runtime_checkable
class A5cAuthenticatedRowBank(Protocol):
    """Minimal authenticated row-bank contract consumed by nested A5c CV.

    Both the original A5b bridge and a breadth-aware A5c splitter satisfy this
    contract.  Keeping split ownership here structural prevents the CV helper
    from introducing a second split authority.
    """

    all_rows: AlignedFragmentRows
    audit_rows: AlignedFragmentRows
    node_order: tuple[str, ...]
    family_alias_by_example: Mapping[str, str]
    training_family_aliases: tuple[str, ...]
    held_family_alias: str
    receipt_sha256: str

    def receipt(self) -> dict[str, object]: ...


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _contains_tensor(value: object) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(_contains_tensor(child) for child in value)
    return False


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is outside its finite range")
    return result


def _matrix(value: Tensor, *, rows: int, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout is not torch.strided
        or value.ndim != 2
        or value.shape[0] != rows
        or value.shape[1] <= 0
        or not value.is_floating_point()
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite CPU floating row matrix")
    return value.detach().contiguous()


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {"shape": tuple(canonical.shape), "dtype": str(canonical.dtype)}
        )
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _runtime_input_cast_audit(inputs: Tensor) -> tuple[Tensor, dict[str, object]]:
    source = _matrix(inputs, rows=inputs.shape[0], label="compiled inputs")
    runtime = source.to(dtype=_CANDIDATE_GENERATOR_RUNTIME_DTYPE).contiguous()
    restored = runtime.to(dtype=source.dtype)
    error = (restored - source).abs()
    passed = torch.allclose(
        restored,
        source,
        rtol=_INPUT_CAST_RTOL,
        atol=_INPUT_CAST_ATOL,
    )
    if not passed:
        raise ValueError("A5c float32 runtime-input cast exceeds its tolerance")
    payload = {
        "policy": (
            "cast_compiled_generator_input_to_candidate_state_dtype_before_"
            "factor_casts_and_both_generator_matmuls"
        ),
        "source_dtype": str(source.dtype),
        "runtime_dtype": str(runtime.dtype),
        "absolute_tolerance": _INPUT_CAST_ATOL,
        "relative_tolerance": _INPUT_CAST_RTOL,
        "source_input_sha256": _tensor_sha256(source),
        "runtime_input_sha256": _tensor_sha256(runtime),
        "max_abs_cast_error": float(error.max().item()),
        "rms_cast_error": float(error.square().mean().sqrt().item()),
        "passed": True,
    }
    return runtime, {
        **payload,
        "lineage_sha256": _domain_sha256(_INPUT_CAST_LINEAGE_DOMAIN, payload),
    }


def _ridge_grid(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in result
        )
        or tuple(sorted(float(value) for value in result))
        != tuple(float(value) for value in result)
    ):
        raise ValueError("ridge_grid must be unique, sorted, finite, and nonnegative")
    return tuple(float(value) for value in result)


def _bridge_compiled_inputs_sha256(
    receipt: Mapping[str, object],
) -> str:
    """Read the one A5b-domain input identity from either row-bank receipt."""

    source = receipt.get("source")
    if isinstance(source, Mapping) and "bridge_compiled_inputs_sha256" in source:
        return _require_sha256(
            source["bridge_compiled_inputs_sha256"],
            label="A5c breadth bridge-domain compiled inputs",
        )
    authentication = receipt.get("authentication")
    if isinstance(authentication, Mapping):
        return _require_sha256(
            authentication.get("compiled_inputs_sha256"),
            label="A5b bridge-domain compiled inputs",
        )
    raise ValueError(
        "A5c row-bank receipt lacks a bridge-domain compiled-input identity"
    )


def _head_only_context(rows: int, device: torch.device) -> SequenceContext:
    valid = torch.ones((1, rows), dtype=torch.bool, device=device)
    positions = torch.arange(rows, dtype=torch.long, device=device)[None, :]
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        logical_positions=positions,
        key_logical_positions=positions.clone(),
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=False,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
    )


def _project_final_head(adapter: object, states: Tensor) -> Tensor:
    project = getattr(adapter, "project_logits", None)
    if not callable(project):
        raise TypeError("adapter must expose project_logits")
    context = _head_only_context(states.shape[0], states.device)
    with torch.no_grad():
        logits = project(states[None, :, :], context)
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] != states.shape[0]
        or logits.shape[2] <= 1
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
    ):
        raise RuntimeError("adapter final head returned invalid logits")
    return logits[0].detach().to(device="cpu").contiguous()


def _exact_kl_per_row(teacher_logits: Tensor, candidate_logits: Tensor) -> Tensor:
    if teacher_logits.shape != candidate_logits.shape or teacher_logits.ndim != 2:
        raise ValueError("teacher and candidate logits must be aligned matrices")
    teacher = teacher_logits.to(device="cpu", dtype=torch.float64)
    candidate = candidate_logits.to(device="cpu", dtype=torch.float64)
    teacher_logp = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
    candidate_logp = candidate - torch.logsumexp(
        candidate, dim=-1, keepdim=True
    )
    return (
        teacher_logp.exp() * (teacher_logp - candidate_logp)
    ).sum(dim=-1).clamp_min(0.0)


def _chunked_final_head_kl(
    adapter: object,
    native_states: Tensor,
    candidate_states: Tensor,
    weights: Tensor,
    *,
    chunk_rows: int,
) -> float:
    if (
        native_states.shape != candidate_states.shape
        or weights.shape != (native_states.shape[0],)
        or type(chunk_rows) is not int
        or chunk_rows <= 0
    ):
        raise ValueError("chunked final-head KL inputs are invalid")
    total = 0.0
    for start in range(0, native_states.shape[0], chunk_rows):
        stop = min(start + chunk_rows, native_states.shape[0])
        teacher_logits = _project_final_head(adapter, native_states[start:stop])
        candidate_logits = _project_final_head(
            adapter, candidate_states[start:stop]
        )
        row_kl = _exact_kl_per_row(teacher_logits, candidate_logits)
        total += float((weights[start:stop] * row_kl).sum().item())
        del teacher_logits, candidate_logits, row_kl
    return max(total, 0.0)


def _weighted_state_nrmse(
    candidate: Tensor,
    native: Tensor,
    weights: Tensor,
) -> float:
    if candidate.shape != native.shape or weights.shape != (native.shape[0],):
        raise ValueError("state diagnostic tensors are not aligned")
    values = candidate.to(dtype=torch.float64) - native.to(dtype=torch.float64)
    teacher = native.to(dtype=torch.float64)
    normalized = weights.to(dtype=torch.float64) / weights.sum()
    mse = (normalized * values.square().mean(dim=-1)).sum()
    reference = (normalized * teacher.square().mean(dim=-1)).sum()
    return math.sqrt(float(mse.item())) / max(
        math.sqrt(float(reference.item())), 1.0e-30
    )


def _membership_sha256(
    ownership: Mapping[str, str],
    aliases: Sequence[str],
) -> str:
    selected = set(aliases)
    return _domain_sha256(
        _MEMBERSHIP_DOMAIN,
        tuple(
            sorted(
                (example_id, alias)
                for example_id, alias in ownership.items()
                if alias in selected
            )
        ),
    )


def _rows_membership_sha256(
    rows: AlignedFragmentRows,
    ownership: Mapping[str, str],
) -> str:
    examples = {example_id for example_id, _ in rows.row_keys}
    return _domain_sha256(
        _MEMBERSHIP_DOMAIN,
        tuple(sorted((example_id, ownership[example_id]) for example_id in examples)),
    )


def _input_row_signatures(rows: AlignedFragmentRows) -> tuple[str, ...]:
    fragments = tuple(rows.rows_by_fragment.values())
    reference = fragments[0].inputs
    if any(not torch.equal(value.inputs, reference) for value in fragments[1:]):
        raise ValueError("A5b node inputs are not one shared compiled row bank")
    signatures: list[str] = []
    for row in reference:
        canonical = row.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(_ROW_SIGNATURE_DOMAIN)
        digest.update(
            _canonical_json_bytes(
                {"shape": tuple(canonical.shape), "dtype": str(canonical.dtype)}
            )
        )
        digest.update(b"\0")
        digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
        signatures.append(digest.hexdigest())
    return tuple(signatures)


def _subset_equal_family_rows(
    rows: AlignedFragmentRows,
    ownership: Mapping[str, str],
    aliases: Sequence[str],
    *,
    excluded_input_signatures: set[str] | None = None,
) -> AlignedFragmentRows:
    families = tuple(aliases)
    if not families or len(families) != len(set(families)):
        raise ValueError("row subset families must be unique and nonempty")
    selected = set(families)
    signatures = _input_row_signatures(rows)
    excluded = set() if excluded_input_signatures is None else set(
        excluded_input_signatures
    )
    indices = tuple(
        index
        for index, (example_id, _) in enumerate(rows.row_keys)
        if ownership[example_id] in selected and signatures[index] not in excluded
    )
    if not indices:
        raise ValueError("row subset is empty")
    index = torch.tensor(indices, dtype=torch.long)
    selected_keys = tuple(rows.row_keys[offset] for offset in indices)
    present_examples = {example_id for example_id, _ in selected_keys}
    if {ownership[example_id] for example_id in present_examples} != selected:
        raise ValueError("row subset does not cover every requested family")
    normalized: dict[str, LayerFragmentRows] = {}
    for fragment_id, fragment_rows in rows.rows_by_fragment.items():
        raw = fragment_rows.fisher_weights.index_select(0, index).contiguous()
        weights = torch.zeros_like(raw)
        for alias in families:
            mask = torch.tensor(
                [ownership[example_id] == alias for example_id, _ in selected_keys],
                dtype=torch.bool,
            )
            mass = raw[mask].sum()
            if float(mass.item()) <= 0.0:
                raise ValueError("one family has zero Fisher mass")
            weights[mask] = raw[mask] / mass / len(families)
        if not math.isclose(
            float(weights.sum().item()), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise RuntimeError("equal-family Fisher normalization drifted")
        normalized[fragment_id] = LayerFragmentRows(
            inputs=fragment_rows.inputs.index_select(0, index),
            contributions=fragment_rows.contributions.index_select(0, index),
            fisher_weights=weights,
            sequences=len(present_examples),
        )
    return AlignedFragmentRows(
        rows_by_fragment=normalized,
        row_keys=selected_keys,
    )


def _descriptive_rekey_rows(
    rows: AlignedFragmentRows,
    *,
    binding_sha256: str,
) -> AlignedFragmentRows:
    rekeyed: list[tuple[str, int]] = []
    for index, key in enumerate(rows.row_keys):
        digest = _domain_sha256(
            _DESCRIPTIVE_KEY_DOMAIN,
            {"binding_sha256": binding_sha256, "row_ordinal": index, "key": key},
        )
        rekeyed.append((f"a5c-descriptive-replay-{digest}", key[1]))
    return AlignedFragmentRows(
        rows_by_fragment=rows.rows_by_fragment,
        row_keys=tuple(rekeyed),
    )


def _joint_fisher_weights(
    rows: AlignedFragmentRows,
    indices: Tensor,
) -> Tensor:
    weights = torch.stack(
        tuple(
            fragment.fisher_weights.index_select(0, indices)
            for fragment in rows.rows_by_fragment.values()
        ),
        dim=0,
    ).sum(dim=0)
    if float(weights.sum().item()) <= 0.0:
        raise ValueError("held-family joint Fisher mass is zero")
    return (weights / weights.sum()).contiguous()


def _predict_fit_correction(
    fit: FrozenBasisGeneratorFit,
    inputs: Tensor,
    node_order: Sequence[str],
) -> Tensor:
    if set(fit.lowerings_by_node) != set(node_order):
        raise RuntimeError("fitted lowering catalog differs from A5b nodes")
    outputs = tuple(
        apply_modal_generator(
            inputs,
            fit.lowerings_by_node[name].fused_residual_plan,
        )
        for name in node_order
    )
    if not outputs or len({tuple(value.shape) for value in outputs}) != 1:
        raise RuntimeError("fitted node corrections have inconsistent shapes")
    result = torch.stack(outputs, dim=0).sum(dim=0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("fitted joint correction became non-finite")
    return result.contiguous()


def _split_sha256(
    *,
    bridge_sha256: str,
    ridge: float,
    held_alias: str,
    role: str,
    rows: AlignedFragmentRows,
) -> str:
    return _domain_sha256(
        _SPLIT_DOMAIN,
        {
            "bridge_receipt_sha256": bridge_sha256,
            "ridge_hex": ridge.hex(),
            "held_inner_family_alias": held_alias,
            "role": role,
            "row_key_sha256": rows.row_key_sha256,
            "observations": rows.observations,
            "sequences": rows.sequences,
        },
    )


@dataclass(frozen=True, slots=True)
class _HeldFamilyHeadReference:
    alias: str
    indices: Tensor = field(repr=False)
    weights: Tensor = field(repr=False)
    native_states: Tensor = field(repr=False)
    frozen_states: Tensor = field(repr=False)
    frozen_kl: float
    frozen_state_nrmse: float


@dataclass(frozen=True, slots=True)
class A5cFamilyRidgeCvSelection:
    """Selected all-seven correction plus a strict tensor-free receipt."""

    selected_ridge: float | None
    use_frozen_fallback: bool
    correction_fit: FrozenBasisGeneratorFit | None = field(repr=False)
    node_order: tuple[str, ...]
    residual_width: int
    _receipt: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        validated = validate_a5c_family_ridge_cv_receipt(self._receipt)
        selection = validated["selection"]
        assert isinstance(selection, Mapping)
        if (
            self.use_frozen_fallback
            is not bool(selection["use_frozen_fallback"])
            or self.selected_ridge != selection["selected_ridge"]
            or (self.use_frozen_fallback != (self.correction_fit is None))
            or type(self.residual_width) is not int
            or self.residual_width <= 0
        ):
            raise ValueError("A5c executable selection contradicts its receipt")
        if self.correction_fit is not None:
            self.correction_fit.graph_plan.validate_integrity()
            final = validated["final_refit"]
            assert isinstance(final, Mapping)
            if self.correction_fit.graph_plan.artifact_sha256 != final[
                "graph_sha256"
            ]:
                raise ValueError("A5c final fit differs from its receipt")
        retained = json.loads(json.dumps(validated, allow_nan=False))
        object.__setattr__(self, "_receipt", MappingProxyType(retained))

    def receipt(self) -> dict[str, object]:
        return json.loads(json.dumps(dict(self._receipt), allow_nan=False))

    def predict_correction(self, inputs: Tensor) -> Tensor:
        """Execute the post-CV correction on the real float32 runtime path.

        A frozen fallback means the caller must execute the authenticated
        source graph.  Returning zeros here would incorrectly turn that choice
        into deletion, so fallback prediction fails closed.
        """

        if (
            not isinstance(inputs, Tensor)
            or not inputs.is_floating_point()
            or inputs.ndim < 1
            or inputs.numel() == 0
            or not bool(torch.isfinite(inputs).all())
        ):
            raise ValueError("A5c prediction inputs must be finite and floating")
        if self.correction_fit is None:
            raise RuntimeError(
                "A5c selected the frozen fallback; execute the source graph"
            )
        runtime_inputs = inputs.to(
            dtype=_CANDIDATE_GENERATOR_RUNTIME_DTYPE
        ).contiguous()
        return _predict_fit_correction(
            self.correction_fit, runtime_inputs, self.node_order
        )


def _strict_fields(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _validate_metric(value: object, *, label: str) -> Mapping[str, object]:
    metric = _strict_fields(
        value,
        {
            "fisher_weighted_final_head_kl",
            "fisher_weighted_state_nrmse",
        },
        label=label,
    )
    for name in metric:
        _finite(metric[name], label=f"{label} {name}", minimum=0.0)
    return metric


def validate_a5c_family_ridge_cv_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly authenticate a tensor-free A5c nested-CV receipt."""

    if not isinstance(value, Mapping):
        raise TypeError("A5c ridge-CV receipt must be a mapping")
    raw = dict(value)
    _strict_fields(
        raw,
        {
            "schema",
            "format_version",
            "scientific_role",
            "source",
            "configuration",
            "ownership",
            "candidates",
            "selection",
            "final_refit",
            "safety",
            "receipt_sha256",
        },
        label="A5c receipt",
    )
    if (
        raw["schema"] != GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_SCHEMA
        or raw["format_version"]
        != GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_FORMAT_VERSION
        or raw["scientific_role"]
        != "outer_training_only_nested_family_disjoint_ridge_selection"
        or raw["safety"] != _SAFETY
        or _contains_tensor(raw)
    ):
        raise ValueError("A5c receipt header or safety contract drifted")

    source = _strict_fields(
        raw["source"],
        {
            "bridge_receipt_sha256",
            "source_graph_sha256",
            "source_model_sha256",
            "source_lowering_sha256_by_node",
            "native_block_states_sha256",
            "frozen_compiled_block_states_sha256",
            "compiled_correction_base_states_sha256",
            "all_rows_key_sha256",
            "bridge_compiled_inputs_sha256",
            "all_rows_input_sha256",
            "candidate_runtime_input_cast",
            "final_head_token_locality_lineage_sha256",
        },
        label="A5c source",
    )
    for name in (
        "bridge_receipt_sha256",
        "source_graph_sha256",
        "source_model_sha256",
        "native_block_states_sha256",
        "frozen_compiled_block_states_sha256",
        "compiled_correction_base_states_sha256",
        "all_rows_key_sha256",
        "bridge_compiled_inputs_sha256",
        "all_rows_input_sha256",
        "final_head_token_locality_lineage_sha256",
    ):
        _require_sha256(source[name], label=f"A5c source {name}")
    input_cast = _strict_fields(
        source["candidate_runtime_input_cast"],
        {
            "policy",
            "source_dtype",
            "runtime_dtype",
            "absolute_tolerance",
            "relative_tolerance",
            "source_input_sha256",
            "runtime_input_sha256",
            "max_abs_cast_error",
            "rms_cast_error",
            "passed",
            "lineage_sha256",
        },
        label="A5c runtime input cast",
    )
    if (
        input_cast["policy"]
        != (
            "cast_compiled_generator_input_to_candidate_state_dtype_before_"
            "factor_casts_and_both_generator_matmuls"
        )
        or input_cast["source_dtype"] != "torch.float64"
        or input_cast["runtime_dtype"] != "torch.float32"
        or input_cast["absolute_tolerance"] != _INPUT_CAST_ATOL
        or input_cast["relative_tolerance"] != _INPUT_CAST_RTOL
        or input_cast["source_input_sha256"] != source["all_rows_input_sha256"]
        or input_cast["passed"] is not True
        or _finite(
            input_cast["max_abs_cast_error"],
            label="A5c runtime input-cast max error",
            minimum=0.0,
        )
        < 0.0
        or _finite(
            input_cast["rms_cast_error"],
            label="A5c runtime input-cast RMS error",
            minimum=0.0,
        )
        < 0.0
    ):
        raise ValueError("A5c runtime input-cast audit drifted")
    for name in ("source_input_sha256", "runtime_input_sha256", "lineage_sha256"):
        _require_sha256(input_cast[name], label=f"A5c runtime input cast {name}")
    cast_payload = dict(input_cast)
    supplied_cast_lineage = cast_payload.pop("lineage_sha256")
    if supplied_cast_lineage != _domain_sha256(
        _INPUT_CAST_LINEAGE_DOMAIN, cast_payload
    ):
        raise ValueError("A5c runtime input-cast lineage hash mismatch")
    lowering_hashes = source["source_lowering_sha256_by_node"]
    if not isinstance(lowering_hashes, Mapping) or len(lowering_hashes) != 4:
        raise ValueError("A5c source lowering catalog must contain four nodes")
    for name, digest in lowering_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("A5c node names must be nonempty")
        _require_sha256(digest, label=f"A5c source lowering {name}")

    config = _strict_fields(
        raw["configuration"],
        {
            "generator_rank",
            "ridge_grid",
            "ridge_grid_hex",
            "inner_fold_count",
            "inner_training_family_count",
            "inner_evaluation_family_count",
            "selection_objective",
            "state_diagnostic",
            "fallback_rule",
            "output_boundary",
            "duplicate_cross_split_input_policy",
            "candidate_generator_execution_dtype",
            "candidate_generator_execution_policy",
            "final_head_chunk_rows",
            "final_head_chunking",
        },
        label="A5c configuration",
    )
    grid = _ridge_grid(config["ridge_grid"])  # type: ignore[arg-type]
    if (
        config["generator_rank"] != A5C_FIXED_GENERATOR_RANK
        or config["ridge_grid_hex"] != [ridge.hex() for ridge in grid]
        or config["inner_fold_count"] != _INNER_FOLD_COUNT
        or config["inner_training_family_count"]
        != _INNER_TRAINING_FAMILY_COUNT
        or config["inner_evaluation_family_count"] != 1
        or config["selection_objective"] != _SELECTION_OBJECTIVE
        or config["state_diagnostic"] != _STATE_DIAGNOSTIC
        or config["fallback_rule"] != _FALLBACK_RULE
        or not isinstance(config["output_boundary"], str)
        or not config["output_boundary"]
        or config["duplicate_cross_split_input_policy"]
        != (
            "exclude_from_inner_fit_every_row_whose_compiled_input_signature_"
            "occurs_in_that_folds_evaluation_family_then_require_zero_overlap"
        )
        or config["candidate_generator_execution_dtype"] != "torch.float32"
        or config["candidate_generator_execution_policy"]
        != (
            "cast_inputs_then_cast_generator_factors_to_input_dtype_before_"
            "both_matmuls_matching_modal_graph_executor"
        )
        or type(config["final_head_chunk_rows"]) is not int
        or config["final_head_chunk_rows"] <= 0
        or config["final_head_chunking"]
        != (
            "fixed_row_chunks_over_token_local_final_norm_and_lm_head_"
            "with_float64_full_vocabulary_kl_accumulation"
        )
    ):
        raise ValueError("A5c configuration drifted")

    ownership = _strict_fields(
        raw["ownership"],
        {
            "outer_training_family_aliases",
            "outer_held_family_alias",
            "outer_training_family_count",
            "outer_training_example_count",
            "outer_training_observation_count",
            "outer_training_membership_sha256",
            "outer_held_family_present_in_ownership",
            "outer_held_family_states_or_rows_accessed",
        },
        label="A5c ownership",
    )
    aliases = tuple(ownership["outer_training_family_aliases"])  # type: ignore[arg-type]
    if (
        len(aliases) != _INNER_FOLD_COUNT
        or len(set(aliases)) != _INNER_FOLD_COUNT
        or any(not isinstance(alias, str) or not alias for alias in aliases)
        or not isinstance(ownership["outer_held_family_alias"], str)
        or ownership["outer_held_family_alias"] in aliases
        or ownership["outer_training_family_count"] != _INNER_FOLD_COUNT
        or type(ownership["outer_training_example_count"]) is not int
        or ownership["outer_training_example_count"] < _INNER_FOLD_COUNT
        or type(ownership["outer_training_observation_count"]) is not int
        or ownership["outer_training_observation_count"]
        < ownership["outer_training_example_count"]
        or ownership["outer_held_family_present_in_ownership"] is not False
        or ownership["outer_held_family_states_or_rows_accessed"] is not False
    ):
        raise ValueError("A5c ownership contract drifted")
    _require_sha256(
        ownership["outer_training_membership_sha256"],
        label="A5c training membership",
    )

    candidates = raw["candidates"]
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or len(candidates) != len(grid)
    ):
        raise ValueError("A5c candidates do not cover the ridge grid")
    parsed_candidates: list[tuple[float, float, float]] = []
    common_frozen_kl: float | None = None
    common_frozen_state: float | None = None
    frozen_metric_by_alias: dict[str, tuple[float, float]] = {}
    for ridge, raw_candidate in zip(grid, candidates, strict=True):
        candidate = _strict_fields(
            raw_candidate,
            {
                "ridge",
                "ridge_hex",
                "inner_fold_count",
                "family_equal_candidate_final_head_kl",
                "family_equal_frozen_final_head_kl",
                "family_equal_candidate_state_nrmse",
                "family_equal_frozen_state_nrmse",
                "folds",
            },
            label="A5c candidate",
        )
        if (
            candidate["ridge"] != ridge
            or candidate["ridge_hex"] != ridge.hex()
            or candidate["inner_fold_count"] != _INNER_FOLD_COUNT
        ):
            raise ValueError("A5c candidate ridge identity drifted")
        folds = candidate["folds"]
        if (
            not isinstance(folds, Sequence)
            or isinstance(folds, (str, bytes))
            or len(folds) != _INNER_FOLD_COUNT
        ):
            raise ValueError("A5c candidate fold count drifted")
        candidate_kls: list[float] = []
        frozen_kls: list[float] = []
        candidate_states: list[float] = []
        frozen_states: list[float] = []
        for alias, raw_fold in zip(aliases, folds, strict=True):
            fold = _strict_fields(
                raw_fold,
                {
                    "held_inner_family_alias",
                    "training_family_count",
                    "fit_observations",
                    "fit_examples",
                    "evaluation_observations",
                    "evaluation_examples",
                    "fit_membership_sha256",
                    "evaluation_membership_sha256",
                    "fit_row_key_sha256",
                    "evaluation_row_key_sha256",
                    "fit_split_sha256",
                    "evaluation_split_sha256",
                    "fit_rows_removed_for_signature_overlap",
                    "input_signature_overlap_count",
                    "input_signatures_disjoint",
                    "graph_sha256",
                    "lowering_sha256_by_node",
                    "candidate",
                    "frozen_baseline",
                },
                label="A5c inner fold",
            )
            if (
                fold["held_inner_family_alias"] != alias
                or fold["training_family_count"]
                != _INNER_TRAINING_FAMILY_COUNT
                or any(
                    type(fold[name]) is not int or fold[name] <= 0
                    for name in (
                        "fit_observations",
                        "fit_examples",
                        "evaluation_observations",
                        "evaluation_examples",
                    )
                )
                or fold["input_signature_overlap_count"] != 0
                or fold["input_signatures_disjoint"] is not True
                or type(fold["fit_rows_removed_for_signature_overlap"])
                is not int
                or fold["fit_rows_removed_for_signature_overlap"] < 0
            ):
                raise ValueError("A5c inner fold ownership or overlap drifted")
            for name in (
                "fit_membership_sha256",
                "evaluation_membership_sha256",
                "fit_row_key_sha256",
                "evaluation_row_key_sha256",
                "fit_split_sha256",
                "evaluation_split_sha256",
                "graph_sha256",
            ):
                _require_sha256(fold[name], label=f"A5c fold {name}")
            expected_fit_split = _domain_sha256(
                _SPLIT_DOMAIN,
                {
                    "bridge_receipt_sha256": source[
                        "bridge_receipt_sha256"
                    ],
                    "ridge_hex": ridge.hex(),
                    "held_inner_family_alias": alias,
                    "role": "inner_fit_six_families",
                    "row_key_sha256": fold["fit_row_key_sha256"],
                    "observations": fold["fit_observations"],
                    "sequences": fold["fit_examples"],
                },
            )
            expected_eval_split = _domain_sha256(
                _SPLIT_DOMAIN,
                {
                    "bridge_receipt_sha256": source[
                        "bridge_receipt_sha256"
                    ],
                    "ridge_hex": ridge.hex(),
                    "held_inner_family_alias": alias,
                    "role": "inner_evaluation_one_family",
                    "row_key_sha256": fold["evaluation_row_key_sha256"],
                    "observations": fold["evaluation_observations"],
                    "sequences": fold["evaluation_examples"],
                },
            )
            if (
                fold["fit_split_sha256"] != expected_fit_split
                or fold["evaluation_split_sha256"] != expected_eval_split
            ):
                raise ValueError("A5c inner split lineage is contradictory")
            hashes = fold["lowering_sha256_by_node"]
            if not isinstance(hashes, Mapping) or set(hashes) != set(
                lowering_hashes
            ):
                raise ValueError("A5c fold lowering catalog drifted")
            for name, digest in hashes.items():
                _require_sha256(digest, label=f"A5c fold lowering {name}")
            candidate_metric = _validate_metric(
                fold["candidate"], label="A5c candidate fold metric"
            )
            frozen_metric = _validate_metric(
                fold["frozen_baseline"], label="A5c frozen fold metric"
            )
            candidate_kls.append(float(candidate_metric["fisher_weighted_final_head_kl"]))
            frozen_kls.append(float(frozen_metric["fisher_weighted_final_head_kl"]))
            candidate_states.append(float(candidate_metric["fisher_weighted_state_nrmse"]))
            frozen_states.append(float(frozen_metric["fisher_weighted_state_nrmse"]))
            frozen_pair = (frozen_kls[-1], frozen_states[-1])
            if alias not in frozen_metric_by_alias:
                frozen_metric_by_alias[alias] = frozen_pair
            elif frozen_metric_by_alias[alias] != frozen_pair:
                raise ValueError(
                    "A5c frozen fold metric changed across ridge candidates"
                )
        aggregates = (
            ("family_equal_candidate_final_head_kl", candidate_kls),
            ("family_equal_frozen_final_head_kl", frozen_kls),
            ("family_equal_candidate_state_nrmse", candidate_states),
            ("family_equal_frozen_state_nrmse", frozen_states),
        )
        for name, values in aggregates:
            observed = _finite(candidate[name], label=f"A5c {name}", minimum=0.0)
            expected = math.fsum(values) / len(values)
            if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ValueError(f"A5c {name} contradicts its folds")
        candidate_kl = float(candidate["family_equal_candidate_final_head_kl"])
        frozen_kl = float(candidate["family_equal_frozen_final_head_kl"])
        if common_frozen_kl is None:
            common_frozen_kl = frozen_kl
        elif not math.isclose(frozen_kl, common_frozen_kl, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("A5c frozen baseline changed across ridge candidates")
        frozen_state = float(candidate["family_equal_frozen_state_nrmse"])
        if common_frozen_state is None:
            common_frozen_state = frozen_state
        elif frozen_state != common_frozen_state:
            raise ValueError(
                "A5c frozen state baseline changed across ridge candidates"
            )
        parsed_candidates.append((candidate_kl, ridge, frozen_kl))

    winner_kl, winner_ridge, baseline_kl = min(
        parsed_candidates, key=lambda item: (item[0], item[1])
    )
    fallback = not winner_kl < baseline_kl
    selection = _strict_fields(
        raw["selection"],
        {
            "objective",
            "winner_ridge_before_fallback",
            "winner_ridge_hex_before_fallback",
            "winner_family_equal_final_head_kl",
            "frozen_family_equal_final_head_kl",
            "absolute_kl_improvement",
            "best_ridge_strictly_improves_frozen",
            "use_frozen_fallback",
            "selected_ridge",
            "selected_ridge_hex",
        },
        label="A5c selection",
    )
    expected_selected = None if fallback else winner_ridge
    if (
        selection["objective"] != _SELECTION_OBJECTIVE
        or selection["winner_ridge_before_fallback"] != winner_ridge
        or selection["winner_ridge_hex_before_fallback"] != winner_ridge.hex()
        or not math.isclose(
            _finite(selection["winner_family_equal_final_head_kl"], label="A5c winner KL"),
            winner_kl,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite(selection["frozen_family_equal_final_head_kl"], label="A5c frozen KL"),
            baseline_kl,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            _finite(selection["absolute_kl_improvement"], label="A5c KL improvement"),
            baseline_kl - winner_kl,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        or selection["best_ridge_strictly_improves_frozen"] is not (not fallback)
        or selection["use_frozen_fallback"] is not fallback
        or selection["selected_ridge"] != expected_selected
        or selection["selected_ridge_hex"]
        != (None if fallback else winner_ridge.hex())
    ):
        raise ValueError("A5c selection contradicts candidate scores")

    final = _strict_fields(
        raw["final_refit"],
        {
            "performed",
            "selected_ridge",
            "fit_uses_all_outer_training_examples",
            "fit_observations",
            "fit_examples",
            "fit_fisher_normalization",
            "descriptive_eval_source",
            "descriptive_eval_is_subset_of_final_fit",
            "descriptive_eval_is_independent",
            "descriptive_eval_used_for_selection",
            "fit_split_sha256",
            "descriptive_eval_split_sha256",
            "graph_sha256",
            "lowering_sha256_by_node",
        },
        label="A5c final refit",
    )
    if fallback:
        if final != {
            "performed": False,
            "selected_ridge": None,
            "fit_uses_all_outer_training_examples": False,
            "fit_observations": 0,
            "fit_examples": 0,
            "fit_fisher_normalization": "not_applicable_frozen_fallback",
            "descriptive_eval_source": "not_applicable_frozen_fallback",
            "descriptive_eval_is_subset_of_final_fit": False,
            "descriptive_eval_is_independent": False,
            "descriptive_eval_used_for_selection": False,
            "fit_split_sha256": None,
            "descriptive_eval_split_sha256": None,
            "graph_sha256": None,
            "lowering_sha256_by_node": {},
        }:
            raise ValueError("A5c fallback final-refit receipt drifted")
    else:
        if (
            final["performed"] is not True
            or final["selected_ridge"] != winner_ridge
            or final["fit_uses_all_outer_training_examples"] is not True
            or final["fit_observations"]
            != ownership["outer_training_observation_count"]
            or final["fit_examples"] != ownership["outer_training_example_count"]
            or final["fit_fisher_normalization"]
            != "equal_total_mass_per_outer_training_family_and_node"
            or final["descriptive_eval_source"]
            != "bridge_audit_rows_rekeyed_after_becoming_subset_of_all_rows_fit"
            or final["descriptive_eval_is_subset_of_final_fit"] is not True
            or final["descriptive_eval_is_independent"] is not False
            or final["descriptive_eval_used_for_selection"] is not False
        ):
            raise ValueError("A5c post-CV refit contract drifted")
        for name in (
            "fit_split_sha256",
            "descriptive_eval_split_sha256",
            "graph_sha256",
        ):
            _require_sha256(final[name], label=f"A5c final refit {name}")
        hashes = final["lowering_sha256_by_node"]
        if not isinstance(hashes, Mapping) or set(hashes) != set(lowering_hashes):
            raise ValueError("A5c final-refit lowering catalog drifted")
        for name, digest in hashes.items():
            _require_sha256(digest, label=f"A5c final lowering {name}")

    supplied = _require_sha256(raw["receipt_sha256"], label="A5c receipt")
    payload = dict(raw)
    payload.pop("receipt_sha256")
    if supplied != _domain_sha256(_REPORT_DOMAIN, payload):
        raise ValueError("A5c receipt hash mismatch")
    return raw


def _authenticate_modal_generator_lowering(
    value: object,
    *,
    label: str,
) -> ModalGeneratorLowering:
    """Authenticate a lowering through its strict public serialization API."""

    if not isinstance(value, ModalGeneratorLowering):
        raise TypeError(f"{label} must be a ModalGeneratorLowering")
    authenticated = ModalGeneratorLowering.from_state_dict(value.state_dict())
    if authenticated.artifact_sha256 != value.artifact_sha256:
        raise ValueError(f"{label} artifact identity drifted during authentication")
    return authenticated


def _authenticate_lowering_catalog(
    values: Mapping[str, object],
    node_order: Sequence[str],
    *,
    label: str,
) -> dict[str, ModalGeneratorLowering]:
    if not isinstance(values, Mapping) or set(values) != set(node_order):
        raise RuntimeError(f"{label} catalog differs from the frozen node order")
    return {
        name: _authenticate_modal_generator_lowering(
            values[name], label=f"{label} {name}"
        )
        for name in node_order
    }


def _fit_hashes(
    fit: FrozenBasisGeneratorFit,
    node_order: Sequence[str],
) -> tuple[str, dict[str, str]]:
    graph = fit.graph_plan
    graph.validate_integrity()
    if tuple(graph.traversal_order) != tuple(node_order):
        raise RuntimeError("A5c fit changed the frozen node order")
    lowerings = _authenticate_lowering_catalog(
        fit.lowerings_by_node,
        node_order,
        label="A5c fitted lowering",
    )
    for name in node_order:
        lowering = lowerings[name]
        if lowering.coordinate_generator_plan.rank != A5C_FIXED_GENERATOR_RANK:
            raise RuntimeError("A5c fit did not preserve fixed generator rank 16")
    return graph.artifact_sha256, {
        name: lowerings[name].artifact_sha256 for name in node_order
    }


def select_a5c_family_disjoint_ridge(
    *,
    bridge: A5cAuthenticatedRowBank,
    source_graph: ModalGeneratorGraphPlan,
    source_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    adapter: object,
    native_block_states: Tensor,
    frozen_compiled_block_states: Tensor,
    compiled_correction_base_states: Tensor,
    output_boundary: str,
    ridge_grid: Sequence[float] = A5C_RIDGE_GRID,
    final_head_chunk_rows: int = 8,
    final_head_token_locality_lineage_sha256: str,
) -> A5cFamilyRidgeCvSelection:
    """Select ridge by seven inner family folds and refit all seven families.

    All three state banks must be in the exact ``bridge.all_rows`` order.
    ``frozen_compiled_block_states`` is the uncorrected source-graph output and
    is used only as the fallback comparator.  Candidate states are constructed
    from ``compiled_correction_base_states`` (compiled post-attention residual
    plus compact-retained post-feed-forward delta) plus the new correction.
    This distinction prevents the source correction from being counted twice.

    Common compiled-input rows (for example a universal BOS prefix) are
    removed from each inner fold's *fit* side whenever their exact signature
    occurs in that fold's evaluation family.  The actual fit/evaluation inputs
    are then required to have zero signature overlap.  If exclusion empties a
    family, the helper fails before fitting rather than overstating evidence.
    """

    if not isinstance(bridge, A5cAuthenticatedRowBank):
        raise TypeError("bridge must implement the authenticated A5c row-bank protocol")
    bridge_receipt = bridge.receipt()
    if (
        not isinstance(bridge_receipt, Mapping)
        or bridge_receipt.get("receipt_sha256") != bridge.receipt_sha256
    ):
        raise ValueError("A5c row-bank receipt identity drifted")
    bridge_input_sha256 = _bridge_compiled_inputs_sha256(bridge_receipt)
    training_aliases = tuple(bridge.training_family_aliases)
    if len(training_aliases) != _INNER_FOLD_COUNT:
        raise ValueError("A5c requires exactly seven outer-training families")
    ownership = dict(bridge.family_alias_by_example)
    if (
        set(ownership.values()) != set(training_aliases)
        or bridge.held_family_alias in ownership.values()
    ):
        raise ValueError("A5c ownership contains outer held-family data")
    if not isinstance(output_boundary, str) or not output_boundary:
        raise ValueError("output_boundary must be nonempty")
    grid = _ridge_grid(ridge_grid)
    if type(final_head_chunk_rows) is not int or final_head_chunk_rows <= 0:
        raise ValueError("final_head_chunk_rows must be positive")
    token_locality_lineage = _require_sha256(
        final_head_token_locality_lineage_sha256,
        label="A5c final-head token-locality lineage",
    )

    source_graph.validate_integrity()
    node_order = tuple(bridge.node_order)
    if (
        tuple(source_graph.traversal_order) != node_order
        or bool(source_graph.interactions)
        or set(source_lowerings_by_node) != set(node_order)
    ):
        raise ValueError("A5c source graph differs from the four-node A5b graph")
    source_lowerings_by_node = _authenticate_lowering_catalog(
        source_lowerings_by_node,
        node_order,
        label="A5c source lowering",
    )
    adapter_fingerprint = getattr(adapter, "model_fingerprint", None)
    if not callable(adapter_fingerprint):
        raise TypeError("adapter must expose model_fingerprint")
    source_model_sha256 = _require_sha256(
        adapter_fingerprint(), label="A5c adapter model fingerprint"
    )
    if source_graph.model_fingerprint != source_model_sha256:
        raise ValueError("A5c source graph does not bind the adapter")

    all_rows = bridge.all_rows
    rows = all_rows.observations
    native = _matrix(native_block_states, rows=rows, label="native block states")
    frozen = _matrix(
        frozen_compiled_block_states,
        rows=rows,
        label="frozen compiled block states",
    )
    correction_base = _matrix(
        compiled_correction_base_states,
        rows=rows,
        label="compiled correction base states",
    )
    if (
        native.shape != frozen.shape
        or native.shape != correction_base.shape
        or native.dtype != frozen.dtype
        or native.dtype != correction_base.dtype
    ):
        raise ValueError("native, frozen, and correction-base states must align")
    widths = {node.output_width for node in source_graph.nodes}
    input_widths = {node.input_width for node in source_graph.nodes}
    if widths != {native.shape[1]} or len(input_widths) != 1:
        raise ValueError("A5c graph/state widths disagree")
    shared_inputs = next(iter(all_rows.rows_by_fragment.values())).inputs
    if shared_inputs.shape[1] != next(iter(input_widths)):
        raise ValueError("A5c compiled-input width differs from the source graph")
    runtime_shared_inputs, input_cast_audit = _runtime_input_cast_audit(
        shared_inputs
    )
    if runtime_shared_inputs.dtype != native.dtype:
        raise ValueError(
            "A5c candidate state dtype differs from generator runtime dtype"
        )
    signatures = _input_row_signatures(all_rows)
    key_to_index = {key: index for index, key in enumerate(all_rows.row_keys)}

    fold_rows: dict[
        str,
        tuple[
            AlignedFragmentRows,
            AlignedFragmentRows,
            int,
            _HeldFamilyHeadReference,
        ],
    ] = {}
    for held_alias in training_aliases:
        fit_aliases = tuple(
            alias for alias in training_aliases if alias != held_alias
        )
        eval_source_indices = tuple(
            index
            for index, (example_id, _) in enumerate(all_rows.row_keys)
            if ownership[example_id] == held_alias
        )
        eval_signatures = {signatures[index] for index in eval_source_indices}
        original_fit_count = sum(
            ownership[example_id] in set(fit_aliases)
            for example_id, _ in all_rows.row_keys
        )
        fit_rows = _subset_equal_family_rows(
            all_rows,
            ownership,
            fit_aliases,
            excluded_input_signatures=eval_signatures,
        )
        eval_rows = _subset_equal_family_rows(
            all_rows,
            ownership,
            (held_alias,),
        )
        fit_signatures = set(_input_row_signatures(fit_rows))
        actual_eval_signatures = set(_input_row_signatures(eval_rows))
        overlap = fit_signatures & actual_eval_signatures
        if overlap:
            raise RuntimeError("A5c signature exclusion failed")
        removed = original_fit_count - fit_rows.observations
        if removed < 0:
            raise RuntimeError("A5c signature-exclusion accounting drifted")
        eval_indices = torch.tensor(
            [key_to_index[key] for key in eval_rows.row_keys], dtype=torch.long
        )
        weights = _joint_fisher_weights(
            eval_rows,
            torch.arange(eval_rows.observations, dtype=torch.long),
        )
        native_fold = native.index_select(0, eval_indices)
        frozen_fold = frozen.index_select(0, eval_indices)
        frozen_kl = _chunked_final_head_kl(
            adapter,
            native_fold,
            frozen_fold,
            weights,
            chunk_rows=final_head_chunk_rows,
        )
        reference = _HeldFamilyHeadReference(
            alias=held_alias,
            indices=eval_indices,
            weights=weights,
            native_states=native_fold,
            frozen_states=frozen_fold,
            frozen_kl=frozen_kl,
            frozen_state_nrmse=_weighted_state_nrmse(
                frozen_fold, native_fold, weights
            ),
        )
        fold_rows[held_alias] = (fit_rows, eval_rows, removed, reference)

    candidate_receipts: list[dict[str, object]] = []
    for ridge in grid:
        fold_receipts: list[dict[str, object]] = []
        for held_alias in training_aliases:
            fit_rows, eval_rows, removed, reference = fold_rows[held_alias]
            fit_split = _split_sha256(
                bridge_sha256=bridge.receipt_sha256,
                ridge=ridge,
                held_alias=held_alias,
                role="inner_fit_six_families",
                rows=fit_rows,
            )
            eval_split = _split_sha256(
                bridge_sha256=bridge.receipt_sha256,
                ridge=ridge,
                held_alias=held_alias,
                role="inner_evaluation_one_family",
                rows=eval_rows,
            )
            fitted = fit_frozen_basis_coordinate_generators(
                fit_rows,
                eval_rows,
                source_graph=source_graph,
                source_lowerings_by_node=source_lowerings_by_node,
                fit_split_sha256=fit_split,
                eval_split_sha256=eval_split,
                generator_rank=A5C_FIXED_GENERATOR_RANK,
                ridge=ridge,
                output_boundary=output_boundary,
            )
            graph_sha256, lowering_sha256_by_node = _fit_hashes(
                fitted, node_order
            )
            evaluation_inputs = next(
                iter(eval_rows.rows_by_fragment.values())
            ).inputs.to(dtype=reference.frozen_states.dtype).contiguous()
            expected_runtime_inputs = runtime_shared_inputs.index_select(
                0, reference.indices
            ).contiguous()
            if not torch.equal(evaluation_inputs, expected_runtime_inputs):
                raise RuntimeError("A5c fold runtime-input cast/order drifted")
            correction = _predict_fit_correction(
                fitted, evaluation_inputs, node_order
            )
            candidate_base = correction_base.index_select(0, reference.indices)
            candidate_states = candidate_base + correction
            candidate_kl = _chunked_final_head_kl(
                adapter,
                reference.native_states,
                candidate_states,
                reference.weights,
                chunk_rows=final_head_chunk_rows,
            )
            fold_receipts.append(
                {
                    "held_inner_family_alias": held_alias,
                    "training_family_count": _INNER_TRAINING_FAMILY_COUNT,
                    "fit_observations": fit_rows.observations,
                    "fit_examples": fit_rows.sequences,
                    "evaluation_observations": eval_rows.observations,
                    "evaluation_examples": eval_rows.sequences,
                    "fit_membership_sha256": _rows_membership_sha256(
                        fit_rows, ownership
                    ),
                    "evaluation_membership_sha256": _rows_membership_sha256(
                        eval_rows, ownership
                    ),
                    "fit_row_key_sha256": fit_rows.row_key_sha256,
                    "evaluation_row_key_sha256": eval_rows.row_key_sha256,
                    "fit_split_sha256": fit_split,
                    "evaluation_split_sha256": eval_split,
                    "fit_rows_removed_for_signature_overlap": removed,
                    "input_signature_overlap_count": 0,
                    "input_signatures_disjoint": True,
                    "graph_sha256": graph_sha256,
                    "lowering_sha256_by_node": lowering_sha256_by_node,
                    "candidate": {
                        "fisher_weighted_final_head_kl": candidate_kl,
                        "fisher_weighted_state_nrmse": _weighted_state_nrmse(
                            candidate_states,
                            reference.native_states,
                            reference.weights,
                        ),
                    },
                    "frozen_baseline": {
                        "fisher_weighted_final_head_kl": reference.frozen_kl,
                        "fisher_weighted_state_nrmse": (
                            reference.frozen_state_nrmse
                        ),
                    },
                }
            )
        def family_mean(metric: str, field: str) -> float:
            return math.fsum(
                float(fold[metric][field])  # type: ignore[index]
                for fold in fold_receipts
            ) / len(fold_receipts)

        candidate_receipts.append(
            {
                "ridge": ridge,
                "ridge_hex": ridge.hex(),
                "inner_fold_count": len(fold_receipts),
                "family_equal_candidate_final_head_kl": family_mean(
                    "candidate", "fisher_weighted_final_head_kl"
                ),
                "family_equal_frozen_final_head_kl": family_mean(
                    "frozen_baseline", "fisher_weighted_final_head_kl"
                ),
                "family_equal_candidate_state_nrmse": family_mean(
                    "candidate", "fisher_weighted_state_nrmse"
                ),
                "family_equal_frozen_state_nrmse": family_mean(
                    "frozen_baseline", "fisher_weighted_state_nrmse"
                ),
                "folds": fold_receipts,
            }
        )

    winner = min(
        candidate_receipts,
        key=lambda item: (
            float(item["family_equal_candidate_final_head_kl"]),
            float(item["ridge"]),
        ),
    )
    winner_ridge = float(winner["ridge"])
    winner_kl = float(winner["family_equal_candidate_final_head_kl"])
    frozen_kl = float(winner["family_equal_frozen_final_head_kl"])
    use_frozen = not winner_kl < frozen_kl

    final_fit: FrozenBasisGeneratorFit | None = None
    if use_frozen:
        final_refit: dict[str, object] = {
            "performed": False,
            "selected_ridge": None,
            "fit_uses_all_outer_training_examples": False,
            "fit_observations": 0,
            "fit_examples": 0,
            "fit_fisher_normalization": "not_applicable_frozen_fallback",
            "descriptive_eval_source": "not_applicable_frozen_fallback",
            "descriptive_eval_is_subset_of_final_fit": False,
            "descriptive_eval_is_independent": False,
            "descriptive_eval_used_for_selection": False,
            "fit_split_sha256": None,
            "descriptive_eval_split_sha256": None,
            "graph_sha256": None,
            "lowering_sha256_by_node": {},
        }
        selected_ridge: float | None = None
    else:
        full_fit_rows = _subset_equal_family_rows(
            all_rows, ownership, training_aliases
        )
        descriptive_binding = _domain_sha256(
            _SPLIT_DOMAIN,
            {
                "bridge_receipt_sha256": bridge.receipt_sha256,
                "selected_ridge_hex": winner_ridge.hex(),
                "role": "post_cv_descriptive_audit_replay",
                "source_row_key_sha256": bridge.audit_rows.row_key_sha256,
            },
        )
        descriptive_rows = _descriptive_rekey_rows(
            bridge.audit_rows, binding_sha256=descriptive_binding
        )
        fit_split = _split_sha256(
            bridge_sha256=bridge.receipt_sha256,
            ridge=winner_ridge,
            held_alias="post_cv_all_seven",
            role="post_cv_fit_all_seven_families",
            rows=full_fit_rows,
        )
        descriptive_split = _split_sha256(
            bridge_sha256=bridge.receipt_sha256,
            ridge=winner_ridge,
            held_alias="post_cv_none_descriptive_replay",
            role="post_cv_non_independent_descriptive_evaluation",
            rows=descriptive_rows,
        )
        final_fit = fit_frozen_basis_coordinate_generators(
            full_fit_rows,
            descriptive_rows,
            source_graph=source_graph,
            source_lowerings_by_node=source_lowerings_by_node,
            fit_split_sha256=fit_split,
            eval_split_sha256=descriptive_split,
            generator_rank=A5C_FIXED_GENERATOR_RANK,
            ridge=winner_ridge,
            output_boundary=output_boundary,
        )
        final_graph_sha, final_lowering_hashes = _fit_hashes(
            final_fit, node_order
        )
        final_refit = {
            "performed": True,
            "selected_ridge": winner_ridge,
            "fit_uses_all_outer_training_examples": True,
            "fit_observations": full_fit_rows.observations,
            "fit_examples": full_fit_rows.sequences,
            "fit_fisher_normalization": (
                "equal_total_mass_per_outer_training_family_and_node"
            ),
            "descriptive_eval_source": (
                "bridge_audit_rows_rekeyed_after_becoming_subset_of_all_rows_fit"
            ),
            "descriptive_eval_is_subset_of_final_fit": True,
            "descriptive_eval_is_independent": False,
            "descriptive_eval_used_for_selection": False,
            "fit_split_sha256": fit_split,
            "descriptive_eval_split_sha256": descriptive_split,
            "graph_sha256": final_graph_sha,
            "lowering_sha256_by_node": final_lowering_hashes,
        }
        selected_ridge = winner_ridge

    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_SCHEMA,
        "format_version": GEMMA3_L10_L17_A5C_FAMILY_RIDGE_CV_FORMAT_VERSION,
        "scientific_role": (
            "outer_training_only_nested_family_disjoint_ridge_selection"
        ),
        "source": {
            "bridge_receipt_sha256": bridge.receipt_sha256,
            "source_graph_sha256": source_graph.artifact_sha256,
            "source_model_sha256": source_model_sha256,
            "source_lowering_sha256_by_node": {
                name: source_lowerings_by_node[name].artifact_sha256
                for name in node_order
            },
            "native_block_states_sha256": _tensor_sha256(native),
            "frozen_compiled_block_states_sha256": _tensor_sha256(frozen),
            "compiled_correction_base_states_sha256": _tensor_sha256(
                correction_base
            ),
            "all_rows_key_sha256": all_rows.row_key_sha256,
            "bridge_compiled_inputs_sha256": bridge_input_sha256,
            "all_rows_input_sha256": _tensor_sha256(shared_inputs),
            "candidate_runtime_input_cast": input_cast_audit,
            "final_head_token_locality_lineage_sha256": (
                token_locality_lineage
            ),
        },
        "configuration": {
            "generator_rank": A5C_FIXED_GENERATOR_RANK,
            "ridge_grid": list(grid),
            "ridge_grid_hex": [ridge.hex() for ridge in grid],
            "inner_fold_count": _INNER_FOLD_COUNT,
            "inner_training_family_count": _INNER_TRAINING_FAMILY_COUNT,
            "inner_evaluation_family_count": 1,
            "selection_objective": _SELECTION_OBJECTIVE,
            "state_diagnostic": _STATE_DIAGNOSTIC,
            "fallback_rule": _FALLBACK_RULE,
            "output_boundary": output_boundary,
            "duplicate_cross_split_input_policy": (
                "exclude_from_inner_fit_every_row_whose_compiled_input_signature_"
                "occurs_in_that_folds_evaluation_family_then_require_zero_overlap"
            ),
            "candidate_generator_execution_dtype": "torch.float32",
            "candidate_generator_execution_policy": (
                "cast_inputs_then_cast_generator_factors_to_input_dtype_before_"
                "both_matmuls_matching_modal_graph_executor"
            ),
            "final_head_chunk_rows": final_head_chunk_rows,
            "final_head_chunking": (
                "fixed_row_chunks_over_token_local_final_norm_and_lm_head_"
                "with_float64_full_vocabulary_kl_accumulation"
            ),
        },
        "ownership": {
            "outer_training_family_aliases": list(training_aliases),
            "outer_held_family_alias": bridge.held_family_alias,
            "outer_training_family_count": len(training_aliases),
            "outer_training_example_count": len(ownership),
            "outer_training_observation_count": rows,
            "outer_training_membership_sha256": _membership_sha256(
                ownership, training_aliases
            ),
            "outer_held_family_present_in_ownership": False,
            "outer_held_family_states_or_rows_accessed": False,
        },
        "candidates": candidate_receipts,
        "selection": {
            "objective": _SELECTION_OBJECTIVE,
            "winner_ridge_before_fallback": winner_ridge,
            "winner_ridge_hex_before_fallback": winner_ridge.hex(),
            "winner_family_equal_final_head_kl": winner_kl,
            "frozen_family_equal_final_head_kl": frozen_kl,
            "absolute_kl_improvement": frozen_kl - winner_kl,
            "best_ridge_strictly_improves_frozen": not use_frozen,
            "use_frozen_fallback": use_frozen,
            "selected_ridge": selected_ridge,
            "selected_ridge_hex": (
                None if selected_ridge is None else selected_ridge.hex()
            ),
        },
        "final_refit": final_refit,
        "safety": dict(_SAFETY),
    }
    payload["receipt_sha256"] = _domain_sha256(_REPORT_DOMAIN, payload)
    validated = validate_a5c_family_ridge_cv_receipt(payload)
    return A5cFamilyRidgeCvSelection(
        selected_ridge=selected_ridge,
        use_frozen_fallback=use_frozen,
        correction_fit=final_fit,
        node_order=node_order,
        residual_width=native.shape[1],
        _receipt=validated,
    )
