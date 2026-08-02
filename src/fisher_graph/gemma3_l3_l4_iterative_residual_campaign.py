"""Fit-only live campaign for one iterative Gemma X4/H4 residual repair.

The campaign deliberately owns model execution but not the residual recipe.
Four narrow callbacks supply the fixed recipe's scalar fit record, fold fit,
pure report builder, and optional full-data refit.  This keeps the live
boundary testable while the recipe remains independently replayable.

Execution is two-phase and family-blocked:

* Phase A runs one direct source pass and one accepted-X4 + lag-B NLL-VJP
  pass for every reusable fit example.
* Eight leave-one-family-out providers are fit from scalar records only.
* Phase B runs one fresh direct source pass and one corresponding OOF
  composite-provider pass for every example.

For the frozen sixteen-example/eight-family expanded A-fit panel this is
exactly sixty-four model forwards.  Candidate outputs are reduced immediately
to scalar observations.  Selection, guard, Calibration-B, and assessment data
are not accepted by this module.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol

import torch
from torch import Tensor

from .compiler.calibration import CausalLanguageModelNLL
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    derive_gemma3_l3_l4_supervised_boundary,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
    measure_gemma_h4_damping_finite_nll_observation,
)
from .gemma3_l3_l4_progressive_worker import GemmaProgressivePanel
from .gemma3_l3_l4_two_head_lowerer import _tensor_sha256
from .shadow_fidelity import ShadowFidelityExample


__all__ = [
    "DEFAULT_GEMMA_ITERATIVE_RESIDUAL_CAMPAIGN_RECIPE",
    "GemmaIterativeResidualCampaignRecipe",
    "GemmaIterativeResidualCampaignResult",
    "GemmaIterativeResidualLiveCollection",
    "collect_gemma_iterative_residual_campaign_live",
    "publish_gemma_iterative_residual_campaign_report",
    "run_gemma_iterative_residual_campaign",
]


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_EXPECTED_EXAMPLE_COUNT = 16
_EXPECTED_FAMILY_COUNT = 8
_EXPECTED_EXAMPLES_PER_FAMILY = 2
_COLLECTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-residual-campaign-collection:v1\0"
)
_RESOURCE_DOMAIN = (
    b"fisher-graph:gemma-iterative-residual-resources:v1\0"
)
_RETENTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-residual-retained-provider:v1\0"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _assert_scalar_hash_only(value: object, *, path: str = "payload") -> None:
    if isinstance(value, Tensor):
        raise TypeError(f"{path} contains a tensor")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite scalar")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            _assert_scalar_hash_only(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_scalar_hash_only(nested, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported payload {type(value)!r}")


def _scalar_payload(value: object, *, label: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"{label} must expose a scalar to_dict payload")
        result = dict(to_dict())
    _assert_scalar_hash_only(result, path=label)
    return result


def _field_name(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None
    ):
        raise ValueError(f"{label} must be a snake-case field name")
    return value


def _provider_int(
    provider: object,
    *,
    attribute: str,
    fallback_attribute: str | None,
    label: str,
) -> int:
    value = getattr(provider, attribute, None)
    if value is None and fallback_attribute is not None:
        value = getattr(provider, fallback_attribute, None)
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class GemmaIterativeResidualCampaignRecipe:
    """Strict scalar protocol for one candidate inside the live LOFO shell.

    Model execution and source authority are invariant across recipes.  This
    descriptor names the candidate's fixed-dimensional linearization, binds
    provider-owned resource claims, and owns the recipe-specific OOF row.
    """

    recipe_id: str
    fit_record_jacobian_field: str
    fold_coefficient_field: str
    coefficient_count: int
    learned_parameter_attribute: str
    learned_parameter_fallback_attribute: str | None
    expected_learned_parameter_count: int
    logical_macs_attribute: str
    logical_macs_fallback_attribute: str | None
    expected_logical_macs_per_token_upper_bound: int | None
    logical_macs_must_equal_residual_width: bool
    extra_resource_expectations: tuple[
        tuple[str, str, int], ...
    ] = ()
    audit_recipe_fields: tuple[tuple[str, object], ...] = ()
    provider_audit_fields: tuple[tuple[str, str], ...] = ()
    parent_tensor_audit_fields: tuple[tuple[str, str], ...] = ()
    fold_projection_field: str = "linearization_extrapolation"
    fold_projection_count_audit_field: str = (
        "fold_linearization_extrapolation_count"
    )
    projection_interpretation_audit_field: str = (
        "coefficient_clipping_interpretation"
    )
    projection_interpretation: str = (
        "linearization_extrapolation_not_free_improvement"
    )
    resource_envelope_error: str = (
        "iterative provider exceeds its resource envelope"
    )
    linearization_error: str = (
        "OOF linearization requires finite recipe coordinates"
    )

    def __post_init__(self) -> None:
        _field_name(self.recipe_id, label="recipe_id")
        _field_name(
            self.fit_record_jacobian_field,
            label="fit-record Jacobian field",
        )
        _field_name(
            self.fold_coefficient_field,
            label="fold coefficient field",
        )
        if (
            type(self.coefficient_count) is not int
            or self.coefficient_count <= 0
            or type(self.expected_learned_parameter_count) is not int
            or self.expected_learned_parameter_count <= 0
        ):
            raise ValueError(
                "recipe coefficient and learned-parameter counts must be "
                "positive"
            )
        for value, label in (
            (self.learned_parameter_attribute, "learned parameter attribute"),
            (self.logical_macs_attribute, "logical MAC attribute"),
        ):
            _field_name(value, label=label)
        for value, label in (
            (
                self.learned_parameter_fallback_attribute,
                "learned parameter fallback attribute",
            ),
            (
                self.logical_macs_fallback_attribute,
                "logical MAC fallback attribute",
            ),
        ):
            if value is not None:
                _field_name(value, label=label)
        expected_macs = self.expected_logical_macs_per_token_upper_bound
        if (
            expected_macs is not None
            and (type(expected_macs) is not int or expected_macs < 0)
        ):
            raise ValueError(
                "expected logical MAC count must be nonnegative"
            )
        if (expected_macs is None) != self.logical_macs_must_equal_residual_width:
            raise ValueError(
                "recipe must bind logical MACs either to a fixed value or "
                "to residual width"
            )
        if (
            type(self.extra_resource_expectations) is not tuple
            or type(self.audit_recipe_fields) is not tuple
            or type(self.provider_audit_fields) is not tuple
            or type(self.parent_tensor_audit_fields) is not tuple
        ):
            raise TypeError("recipe receipt descriptors must be tuples")

        resource_keys: list[str] = []
        for value in self.extra_resource_expectations:
            if type(value) is not tuple or len(value) != 3:
                raise ValueError(
                    "extra resource expectations must be field, attribute, "
                    "expected-value triples"
                )
            field, attribute, expected = value
            resource_keys.append(
                _field_name(field, label="extra resource field")
            )
            _field_name(attribute, label="extra resource attribute")
            if type(expected) is not int or expected < 0:
                raise ValueError(
                    "extra resource expectations must be nonnegative integers"
                )
        if len(resource_keys) != len(set(resource_keys)):
            raise ValueError("extra resource fields must be unique")

        audit_keys: list[str] = []
        for value in self.audit_recipe_fields:
            if type(value) is not tuple or len(value) != 2:
                raise ValueError(
                    "audit recipe fields must be field-value pairs"
                )
            field, payload = value
            audit_keys.append(_field_name(field, label="audit recipe field"))
            _assert_scalar_hash_only(payload, path=f"recipe audit.{field}")
        for descriptors, label in (
            (self.provider_audit_fields, "provider audit"),
            (self.parent_tensor_audit_fields, "parent tensor audit"),
        ):
            for value in descriptors:
                if type(value) is not tuple or len(value) != 2:
                    raise ValueError(
                        f"{label} fields must be field-attribute pairs"
                    )
                field, attribute = value
                audit_keys.append(_field_name(field, label=f"{label} field"))
                _field_name(attribute, label=f"{label} attribute")
        _field_name(
            self.fold_projection_field,
            label="fold projection field",
        )
        for value, label in (
            (
                self.fold_projection_count_audit_field,
                "fold projection count audit field",
            ),
            (
                self.projection_interpretation_audit_field,
                "projection interpretation audit field",
            ),
        ):
            audit_keys.append(_field_name(value, label=label))
        if len(audit_keys) != len(set(audit_keys)):
            raise ValueError("recipe audit fields must be unique")
        for value, label in (
            (
                self.projection_interpretation,
                "projection interpretation",
            ),
            (self.resource_envelope_error, "resource envelope error"),
            (self.linearization_error, "linearization error"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be nonempty")

    def provider_coefficients(
        self,
        provider: Gemma3L3L4CorrectionProvider,
    ) -> tuple[float, ...]:
        try:
            raw = getattr(provider, self.fold_coefficient_field)
        except AttributeError as error:
            raise ValueError(
                "OOF provider omitted its recipe coefficients"
            ) from error
        try:
            result = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "OOF provider recipe coefficients are invalid"
            ) from error
        if len(result) != self.coefficient_count or any(
            not math.isfinite(value) for value in result
        ):
            raise ValueError(
                "OOF provider must expose four finite coefficients"
                if self is DEFAULT_GEMMA_ITERATIVE_RESIDUAL_CAMPAIGN_RECIPE
                else "OOF provider must expose finite recipe coefficients"
            )
        return result

    def provider_resource_receipt(
        self,
        provider: Gemma3L3L4CorrectionProvider,
    ) -> dict[str, int]:
        learned = _provider_int(
            provider,
            attribute=self.learned_parameter_attribute,
            fallback_attribute=self.learned_parameter_fallback_attribute,
            label="provider learned parameter count",
        )
        macs = _provider_int(
            provider,
            attribute=self.logical_macs_attribute,
            fallback_attribute=self.logical_macs_fallback_attribute,
            label="provider logical MAC count",
        )
        extras: dict[str, int] = {}
        for field, attribute, expected in self.extra_resource_expectations:
            observed = _provider_int(
                provider,
                attribute=attribute,
                fallback_attribute=None,
                label=f"provider {field}",
            )
            if observed != expected:
                raise RuntimeError(self.resource_envelope_error)
            extras[field] = observed
        return {
            "learned_parameter_count": learned,
            "logical_macs_per_token_upper_bound": macs,
            **extras,
        }

    def provider_audit_receipt(
        self,
        provider: Gemma3L3L4CorrectionProvider,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for field, attribute in self.provider_audit_fields:
            if not hasattr(provider, attribute):
                raise ValueError(
                    f"OOF provider omitted audit attribute {attribute}"
                )
            value = getattr(provider, attribute)
            _assert_scalar_hash_only(value, path=f"provider audit.{field}")
            result[field] = value
        return result

    def parent_tensor_audit_receipt(
        self,
        parent_h4: object,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for field, attribute in self.parent_tensor_audit_fields:
            value = getattr(parent_h4, attribute, None)
            if not isinstance(value, Tensor):
                raise ValueError(
                    f"parent H4 omitted tensor audit attribute {attribute}"
                )
            result[field] = _tensor_sha256(value)
        return result

    def validate_resource_envelope(
        self,
        *,
        resources: Mapping[str, int],
        residual_width: int,
    ) -> None:
        if (
            resources.get("learned_parameter_count")
            != self.expected_learned_parameter_count
        ):
            raise RuntimeError(self.resource_envelope_error)
        macs = resources.get("logical_macs_per_token_upper_bound")
        if type(macs) is not int or macs < 0 or macs > 1_024:
            raise RuntimeError(self.resource_envelope_error)
        if self.logical_macs_must_equal_residual_width:
            if macs != residual_width:
                raise RuntimeError(
                    "position-scale MAC receipt differs from residual width"
                    if self
                    is DEFAULT_GEMMA_ITERATIVE_RESIDUAL_CAMPAIGN_RECIPE
                    else self.resource_envelope_error
                )
        elif macs != self.expected_logical_macs_per_token_upper_bound:
            raise RuntimeError(self.resource_envelope_error)

    def build_oof_row(
        self,
        *,
        parent_observation: GemmaH4DampingFiniteNLLObservation,
        candidate_observation: GemmaH4DampingFiniteNLLObservation,
        fit_record: Mapping[str, object],
        fold_receipt: Mapping[str, object],
        provider_artifact_sha256: str,
        candidate_execution_sha256: str,
    ) -> dict[str, object]:
        try:
            jacobian = tuple(
                float(value)
                for value in fit_record[self.fit_record_jacobian_field]
            )
            coefficients = tuple(
                float(value)
                for value in fold_receipt[self.fold_coefficient_field]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(self.linearization_error) from error
        if (
            len(jacobian) != self.coefficient_count
            or len(coefficients) != self.coefficient_count
            or any(
                not math.isfinite(value)
                for value in (*jacobian, *coefficients)
            )
        ):
            raise ValueError(self.linearization_error)
        parent_signed = (
            parent_observation.candidate_summed_nll
            - parent_observation.source_summed_nll
        ) / parent_observation.supervised_tokens
        exact_candidate_signed = (
            candidate_observation.candidate_summed_nll
            - candidate_observation.source_summed_nll
        ) / candidate_observation.supervised_tokens
        predicted_candidate_signed = parent_signed + math.fsum(
            left * right
            for left, right in zip(jacobian, coefficients, strict=True)
        )
        return {
            "example_id": parent_observation.example_id,
            "family_id": parent_observation.family_id,
            "held_family_id": parent_observation.family_id,
            "parent_signed_delta_nll_per_token": parent_signed,
            "predicted_candidate_signed_delta_nll_per_token": (
                predicted_candidate_signed
            ),
            "exact_candidate_signed_delta_nll_per_token": (
                exact_candidate_signed
            ),
            self.fit_record_jacobian_field: jacobian,
            self.fold_coefficient_field: coefficients,
            "train_example_ids": fold_receipt["train_example_ids"],
            "train_family_ids": fold_receipt["train_family_ids"],
            "fit_record_sha256": fit_record["fit_record_sha256"],
            "fold_receipt_sha256": fold_receipt["fold_receipt_sha256"],
            "provider_artifact_sha256": provider_artifact_sha256,
            "candidate_execution_sha256": candidate_execution_sha256,
            "candidate_observation_sha256": (
                candidate_observation.observation_sha256
            ),
        }


DEFAULT_GEMMA_ITERATIVE_RESIDUAL_CAMPAIGN_RECIPE = (
    GemmaIterativeResidualCampaignRecipe(
        recipe_id="causal_position_scale",
        fit_record_jacobian_field="jacobian_by_bin",
        fold_coefficient_field="coefficients_by_bin",
        coefficient_count=4,
        learned_parameter_attribute=(
            "marginal_prepared_float_scalar_count"
        ),
        learned_parameter_fallback_attribute="prepared_float_scalar_count",
        expected_learned_parameter_count=4,
        logical_macs_attribute=(
            "marginal_logical_macs_per_token_upper_bound"
        ),
        logical_macs_fallback_attribute=(
            "logical_macs_per_token_upper_bound"
        ),
        expected_logical_macs_per_token_upper_bound=None,
        logical_macs_must_equal_residual_width=True,
        audit_recipe_fields=(
            ("position_bin_count", 4),
            (
                "position_bin_semantics",
                "causal_logical_position_[0_3]_[4_7]_[8_15]_[16_plus]",
            ),
        ),
        resource_envelope_error=(
            "fixed four-bin provider exceeds its resource envelope"
        ),
        linearization_error=(
            "OOF linearization requires four finite bins"
        ),
    )
)


class _ArtifactLike(Protocol):
    artifact_sha256: str
    execution_sha256: str
    runtime_binding_sha256: str
    bridge_binding_sha256: str
    live_model_sha256: str
    adapter_execution_sha256: str

    def validate_integrity(self) -> None: ...

    def head(self, site: str) -> object | None: ...


class _AdapterLike(Protocol):
    def model_fingerprint(self) -> str: ...

    def execution_fingerprint(self) -> str: ...

    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        capture_sites: tuple[str, ...],
    ) -> object: ...


class _BridgeLike(Protocol):
    bridge_binding_sha256: str

    def validate_integrity(self) -> None: ...

    def execute(
        self,
        adapter: _AdapterLike,
        model_inputs: Mapping[str, Tensor],
        *,
        x4_head: object | None,
        h4_head: object | None,
    ) -> object: ...

    def execute_h4_vjp(
        self,
        adapter: _AdapterLike,
        model_inputs: Mapping[str, Tensor],
        *,
        objective: Callable[[object], Tensor],
        x4_head: object | None,
        h4_head: object | None,
    ) -> tuple[object, Tensor]: ...


FitRecordBuilder = Callable[..., object]
FoldFitter = Callable[..., Gemma3L3L4CorrectionProvider]
ReportBuilder = Callable[..., Mapping[str, object]]
FullFitter = Callable[..., Gemma3L3L4CorrectionProvider]


@dataclass(frozen=True, slots=True)
class GemmaIterativeResidualLiveCollection:
    """Scalar outputs and provenance from the exact two-phase live run."""

    fit_records: tuple[Mapping[str, object], ...]
    parent_observations: tuple[
        GemmaH4DampingFiniteNLLObservation, ...
    ]
    candidate_observations: tuple[
        GemmaH4DampingFiniteNLLObservation, ...
    ]
    oof_rows: tuple[Mapping[str, object], ...]
    fold_receipts: tuple[Mapping[str, object], ...]
    resources: Mapping[str, object]
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            type(self.fit_records) is not tuple
            or len(self.fit_records) != _EXPECTED_EXAMPLE_COUNT
            or type(self.parent_observations) is not tuple
            or len(self.parent_observations) != _EXPECTED_EXAMPLE_COUNT
            or type(self.candidate_observations) is not tuple
            or len(self.candidate_observations)
            != _EXPECTED_EXAMPLE_COUNT
            or type(self.oof_rows) is not tuple
            or len(self.oof_rows) != _EXPECTED_EXAMPLE_COUNT
            or type(self.fold_receipts) is not tuple
            or len(self.fold_receipts) != _EXPECTED_FAMILY_COUNT
        ):
            raise ValueError("iterative campaign collection geometry differs")
        _assert_scalar_hash_only(
            {
                "fit_records": self.fit_records,
                "parent_observations": tuple(
                    row.to_dict() for row in self.parent_observations
                ),
                "candidate_observations": tuple(
                    row.to_dict() for row in self.candidate_observations
                ),
                "oof_rows": self.oof_rows,
                "fold_receipts": self.fold_receipts,
                "resources": self.resources,
                "audit": self.audit,
            },
            path="collection",
        )


@dataclass(frozen=True, slots=True)
class GemmaIterativeResidualCampaignResult:
    """One replayable report plus an optional retained full-fit provider."""

    collection: GemmaIterativeResidualLiveCollection
    report: Mapping[str, object]
    retained_provider: Gemma3L3L4CorrectionProvider | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.collection,
            GemmaIterativeResidualLiveCollection,
        ):
            raise TypeError("collection must be a live campaign collection")
        _assert_scalar_hash_only(self.report, path="campaign report")
        if self.retained_provider is not None:
            self.retained_provider.validate_integrity()


def _gather_logits(logits: Tensor, indices: Tensor) -> Tensor:
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or not logits.is_floating_point()
    ):
        raise ValueError("live logits must have shape [1, sequence, vocab]")
    return (
        logits[0]
        .index_select(0, indices.to(logits.device))
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )


def _panel_manifest(panel: GemmaProgressivePanel) -> dict[str, str]:
    if not isinstance(panel, GemmaProgressivePanel):
        raise TypeError("panel must be a strict GemmaProgressivePanel")
    manifest = {
        example.example_id: example.family_id
        for example in panel.examples
    }
    counts = Counter(manifest.values())
    if (
        panel.role != "calibration_a_fit"
        or len(manifest) != _EXPECTED_EXAMPLE_COUNT
        or len(counts) != _EXPECTED_FAMILY_COUNT
        or set(counts.values()) != {_EXPECTED_EXAMPLES_PER_FAMILY}
    ):
        raise ValueError(
            "iterative campaign requires reusable 16-by-8 A-fit panel"
        )
    return manifest


def _validate_parent(
    *,
    panel: GemmaProgressivePanel,
    adapter: _AdapterLike,
    bridge: _BridgeLike,
    parent: _ArtifactLike,
) -> tuple[object, object]:
    bridge.validate_integrity()
    parent.validate_integrity()
    model_sha256 = adapter.model_fingerprint()
    execution_sha256 = adapter.execution_fingerprint()
    if (
        parent.bridge_binding_sha256 != bridge.bridge_binding_sha256
        or parent.live_model_sha256 != model_sha256
        or parent.adapter_execution_sha256 != execution_sha256
    ):
        raise ValueError(
            "iterative parent differs from the live factorized runtime"
        )
    x4 = parent.head(_X4_SITE)
    h4 = parent.head(_H4_SITE)
    if x4 is None or h4 is None:
        raise ValueError("iterative parent must contain accepted X4 and lag B")
    for label, head in (("accepted X4", x4), ("lag B", h4)):
        validate = getattr(head, "validate_integrity", None)
        if not callable(validate):
            raise TypeError(f"{label} lacks integrity validation")
        validate()
        _require_sha256(
            getattr(head, "artifact_sha256", None),
            label=f"{label} artifact",
        )
    if (
        getattr(x4, "site", _X4_SITE) != _X4_SITE
        or getattr(h4, "site", _H4_SITE) != _H4_SITE
        or getattr(h4, "conditioning", None) != "l3_source_modes"
        or getattr(h4, "fit_manifest_sha256", panel.manifest_sha256)
        != panel.manifest_sha256
    ):
        raise ValueError("iterative parent head semantics differ")
    return x4, h4


def _validate_execution(
    execution: object,
    *,
    example_model_inputs_sha256: str,
    bridge_binding_sha256: str,
    x4_head: object,
    h4_head: object,
    label: str,
) -> None:
    validate = getattr(execution, "validate_integrity", None)
    if not callable(validate):
        raise TypeError(f"{label} execution lacks validation")
    validate()
    if (
        getattr(execution, "model_forward_count", None) != 1
        or getattr(execution, "model_inputs_sha256", None)
        != example_model_inputs_sha256
        or getattr(execution, "bridge_binding_sha256", None)
        != bridge_binding_sha256
        or getattr(execution, "x4_head_sha256", None)
        != getattr(x4_head, "artifact_sha256", None)
        or getattr(execution, "h4_head_sha256", None)
        != getattr(h4_head, "artifact_sha256", None)
    ):
        raise ValueError(f"{label} execution identity differs")


def _source_authority(
    *,
    adapter: _AdapterLike,
    example: object,
) -> tuple[object, Tensor, Tensor, Tensor, Tensor]:
    batch = getattr(example, "batch", None)
    model_inputs = getattr(batch, "model_inputs", None)
    if not isinstance(model_inputs, Mapping):
        raise TypeError("campaign example omitted model inputs")
    with torch.no_grad():
        source = adapter.forward(model_inputs, capture_sites=())
    sequence = getattr(source, "sequence", None)
    valid = getattr(sequence, "query_valid_mask", None)
    logical_positions = getattr(sequence, "logical_positions", None)
    input_ids = model_inputs.get("input_ids")
    if (
        not isinstance(input_ids, Tensor)
        or not isinstance(valid, Tensor)
        or valid.dtype != torch.bool
        or valid.shape != input_ids.shape
        or not isinstance(logical_positions, Tensor)
        or logical_positions.shape != valid.shape
    ):
        raise ValueError("direct source authority grid differs")
    indices, targets = derive_gemma3_l3_l4_supervised_boundary(
        input_ids,
        valid,
    )
    expected_targets = torch.full_like(batch.targets, -100)
    expected_targets[0, indices.to(expected_targets.device)] = targets.to(
        expected_targets.device
    )
    if (
        not torch.equal(batch.valid_positions.to(valid.device), valid)
        or not torch.equal(batch.targets, expected_targets)
    ):
        raise ValueError("campaign targets differ from causal boundary")
    logits = _gather_logits(getattr(source, "logits", None), indices)
    return source, logits, indices, targets, logical_positions


def _observation(
    *,
    example: object,
    source_logits: Tensor,
    candidate_logits: Tensor,
    targets: Tensor,
) -> GemmaH4DampingFiniteNLLObservation:
    return measure_gemma_h4_damping_finite_nll_observation(
        ShadowFidelityExample(
            example_id=str(getattr(example, "example_id")),
            family_id=str(getattr(example, "family_id")),
            source_logits=source_logits,
            candidate_logits=candidate_logits,
            targets=targets,
        )
    )


def _same_source(
    left: GemmaH4DampingFiniteNLLObservation,
    right: GemmaH4DampingFiniteNLLObservation,
) -> bool:
    return (
        left.example_id == right.example_id
        and left.family_id == right.family_id
        and left.supervised_tokens == right.supervised_tokens
        and left.source_summed_nll == right.source_summed_nll
        and left.source_logits_sha256 == right.source_logits_sha256
        and left.targets_sha256 == right.targets_sha256
    )


def _provider_receipt(
    *,
    provider: Gemma3L3L4CorrectionProvider,
    held_family: str,
    training_records: Sequence[Mapping[str, object]],
    recipe: GemmaIterativeResidualCampaignRecipe,
) -> dict[str, object]:
    provider.validate_integrity()
    if provider.site != _H4_SITE:
        raise ValueError("OOF provider is bound to the wrong activation site")
    artifact_sha256 = _require_sha256(
        provider.artifact_sha256,
        label="OOF provider",
    )
    training_example_ids = tuple(
        sorted(str(row["example_id"]) for row in training_records)
    )
    training_family_ids = tuple(
        sorted({str(row["family_id"]) for row in training_records})
    )
    if (
        held_family in training_family_ids
        or len(training_example_ids) != 14
        or len(training_family_ids) != 7
    ):
        raise RuntimeError("OOF provider fit leaked its held family")
    coefficients = recipe.provider_coefficients(provider)
    training_record_sha256s = tuple(
        sorted(
            str(
                row.get(
                    "fit_record_sha256",
                    row.get("record_sha256", ""),
                )
            )
            for row in training_records
        )
    )
    fold_fit = getattr(provider, "fold_fit", None)
    fold_to_dict = getattr(fold_fit, "to_dict", None)
    if not callable(fold_to_dict):
        raise TypeError("OOF provider must expose its strict fold fit")
    payload = dict(
        _scalar_payload(
            fold_to_dict(),
            label="OOF fold fit",
        )
    )
    del artifact_sha256
    if (
        payload.get("held_family_id") != held_family
        or tuple(payload.get("train_example_ids", ()))
        != training_example_ids
        or tuple(payload.get("train_family_ids", ()))
        != training_family_ids
        or tuple(payload.get("train_fit_record_sha256s", ()))
        != training_record_sha256s
        or tuple(payload.get(recipe.fold_coefficient_field, ()))
        != coefficients
    ):
        raise RuntimeError("OOF fold-fit receipt differs from training")
    _require_sha256(
        payload.get("fold_receipt_sha256"),
        label="OOF fold receipt",
    )
    return payload


def _retained_provider_receipt(
    *,
    provider: Gemma3L3L4CorrectionProvider,
    parent_artifact_sha256: str,
    parent_h4_sha256: str,
    bridge_binding_sha256: str,
    recipe: GemmaIterativeResidualCampaignRecipe,
) -> dict[str, object]:
    """Bind a successful all-development refit to the final report."""

    provider.validate_integrity()
    fold_fit = getattr(provider, "fold_fit", None)
    to_dict = getattr(fold_fit, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("retained provider must expose its strict full fit")
    fit = dict(_scalar_payload(to_dict(), label="retained full fit"))
    if (
        fit.get("held_family_id") != "__full_fit__"
        or len(tuple(fit.get("train_example_ids", ()))) != 16
        or len(tuple(fit.get("train_family_ids", ()))) != 8
    ):
        raise ValueError("retained provider does not bind the full fit panel")
    coefficients = recipe.provider_coefficients(provider)
    if (
        tuple(fit.get(recipe.fold_coefficient_field, ()))
        != coefficients
    ):
        raise RuntimeError(
            "retained provider full-fit receipt differs from provider"
        )
    provider_resources = recipe.provider_resource_receipt(provider)
    payload: dict[str, object] = {
        "provider_artifact_sha256": _require_sha256(
            provider.artifact_sha256,
            label="retained provider",
        ),
        "parent_artifact_sha256": _require_sha256(
            parent_artifact_sha256,
            label="retained parent",
        ),
        "parent_h4_head_sha256": _require_sha256(
            parent_h4_sha256,
            label="retained parent H4",
        ),
        "bridge_binding_sha256": _require_sha256(
            bridge_binding_sha256,
            label="retained bridge",
        ),
        **provider_resources,
        "full_fit": fit,
    }
    payload["retention_receipt_sha256"] = _sha256(
        _RETENTION_DOMAIN,
        payload,
    )
    return payload


def _report_retained(report: Mapping[str, object]) -> bool:
    decision = report.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("campaign report omitted its retention decision")
    retained = decision.get("retained")
    if type(retained) is not bool:
        raise ValueError("campaign report retention decision is not boolean")
    return retained


def collect_gemma_iterative_residual_campaign_live(
    *,
    panel: GemmaProgressivePanel,
    adapter: _AdapterLike,
    bridge: _BridgeLike,
    parent_artifact: _ArtifactLike,
    make_fit_record: FitRecordBuilder,
    fit_fold: FoldFitter,
    build_report: ReportBuilder,
    fit_full: FullFitter,
    lineage: Mapping[str, object] | None = None,
    recipe: GemmaIterativeResidualCampaignRecipe | None = None,
) -> GemmaIterativeResidualCampaignResult:
    """Run the exact 64-forward fit-only LOFO campaign."""

    manifest = _panel_manifest(panel)
    campaign_recipe = (
        DEFAULT_GEMMA_ITERATIVE_RESIDUAL_CAMPAIGN_RECIPE
        if recipe is None
        else recipe
    )
    if not isinstance(
        campaign_recipe,
        GemmaIterativeResidualCampaignRecipe,
    ):
        raise TypeError(
            "recipe must be a strict iterative residual campaign recipe"
        )
    for label, callback in (
        ("make_fit_record", make_fit_record),
        ("fit_fold", fit_fold),
        ("build_report", build_report),
        ("fit_full", fit_full),
    ):
        if not callable(callback):
            raise TypeError(f"{label} must be callable")
    x4_head, parent_h4 = _validate_parent(
        panel=panel,
        adapter=adapter,
        bridge=bridge,
        parent=parent_artifact,
    )
    source_model_sha256 = adapter.model_fingerprint()
    source_execution_sha256 = adapter.execution_fingerprint()
    parent_artifact_sha256 = parent_artifact.artifact_sha256
    parent_h4_sha256 = getattr(parent_h4, "artifact_sha256")
    objective = CausalLanguageModelNLL()
    records: list[Mapping[str, object]] = []
    parent_observations: list[
        GemmaH4DampingFiniteNLLObservation
    ] = []
    parent_execution_sha256s: list[str] = []
    parent_execution_sha256_by_example: dict[str, str] = {}
    model_inputs_sha256_by_example: dict[str, str] = {}

    # Phase A: source authority plus the parent NLL VJP.
    for example in panel.examples:
        example.validate_integrity()
        source, source_logits, indices, targets, _positions = (
            _source_authority(adapter=adapter, example=example)
        )
        parent_execution, gradient = bridge.execute_h4_vjp(
            adapter,
            example.batch.model_inputs,
            objective=lambda run, batch=example.batch: objective(run, batch),
            x4_head=x4_head,
            h4_head=parent_h4,
        )
        _validate_execution(
            parent_execution,
            example_model_inputs_sha256=example.model_inputs_sha256,
            bridge_binding_sha256=bridge.bridge_binding_sha256,
            x4_head=x4_head,
            h4_head=parent_h4,
            label="parent VJP",
        )
        candidate_h4 = getattr(parent_execution, "candidate_h4", None)
        prefix = getattr(parent_execution, "prefix", None)
        if (
            not isinstance(gradient, Tensor)
            or not isinstance(candidate_h4, Tensor)
            or gradient.shape != candidate_h4.shape
            or not gradient.is_floating_point()
            or not bool(torch.isfinite(gradient).all())
            or prefix is None
        ):
            raise ValueError("parent VJP geometry differs")
        parent_correction = parent_h4.correction(prefix, candidate_h4)
        parent_h4.validate_integrity()
        if (
            not isinstance(parent_correction, Tensor)
            or parent_correction.shape != candidate_h4.shape
            or not parent_correction.is_floating_point()
        ):
            raise ValueError("lag-B correction geometry differs")
        parent_logits = _gather_logits(
            getattr(parent_execution, "logits", None),
            indices,
        )
        parent_observation = _observation(
            example=example,
            source_logits=source_logits,
            candidate_logits=parent_logits,
            targets=targets,
        )
        record_value = make_fit_record(
            example=example,
            parent_execution=parent_execution,
            gradient=gradient,
            lag_b_correction=parent_correction,
            parent_observation=parent_observation,
        )
        record = dict(
            _scalar_payload(record_value, label="campaign fit record")
        )
        if (
            record.get("example_id") != example.example_id
            or record.get("family_id") != example.family_id
            or record.get("model_inputs_sha256")
            != example.model_inputs_sha256
        ):
            raise ValueError("fit record identity differs from the panel")
        _require_sha256(
            record.get("fit_record_sha256"),
            label="fit record",
        )
        records.append(record)
        parent_observations.append(parent_observation)
        parent_execution_sha256 = _require_sha256(
            getattr(parent_execution, "artifact_sha256", None),
            label="parent VJP execution",
        )
        parent_execution_sha256s.append(parent_execution_sha256)
        parent_execution_sha256_by_example[
            example.example_id
        ] = parent_execution_sha256
        model_inputs_sha256_by_example[
            example.example_id
        ] = example.model_inputs_sha256
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(
                example.batch.model_inputs
            )
            != example.model_inputs_sha256
        ):
            raise RuntimeError("campaign model inputs changed in phase A")
        del (
            source,
            source_logits,
            indices,
            targets,
            parent_execution,
            gradient,
            candidate_h4,
            prefix,
            parent_correction,
            parent_logits,
            parent_observation,
            record_value,
        )

    canonical_records = tuple(
        sorted(records, key=lambda row: str(row["example_id"]))
    )
    if len(
        {str(row["fit_record_sha256"]) for row in canonical_records}
    ) != _EXPECTED_EXAMPLE_COUNT:
        raise ValueError("fit record identities must be unique")
    record_payload_before_fits = _canonical_json_bytes(canonical_records)
    canonical_parent = tuple(
        sorted(parent_observations, key=lambda row: row.example_id)
    )
    parent_by_example = {
        row.example_id: row for row in canonical_parent
    }

    # Fit exactly one realization of the fixed recipe per held family.
    providers: dict[str, Gemma3L3L4CorrectionProvider] = {}
    fold_receipts: list[Mapping[str, object]] = []
    fold_provider_resource_receipts: list[Mapping[str, int]] = []
    fold_provider_audit_receipts: list[Mapping[str, object]] = []
    fold_provider_sha256s: list[str] = []
    fold_provider_sha256_by_family: dict[str, str] = {}
    for held_family in sorted(set(manifest.values())):
        training = tuple(
            row
            for row in canonical_records
            if row["family_id"] != held_family
        )
        training_payload_before_fit = _canonical_json_bytes(training)
        provider = fit_fold(
            records=training,
            held_family=held_family,
            parent_h4=parent_h4,
        )
        if _canonical_json_bytes(training) != training_payload_before_fit:
            raise RuntimeError(
                "fold fitting mutated the alpha0 parent or fit records"
            )
        if not isinstance(provider, Gemma3L3L4CorrectionProvider):
            raise TypeError(
                "fold fitter must return an authenticated H4 provider"
            )
        receipt = _provider_receipt(
            provider=provider,
            held_family=held_family,
            training_records=training,
            recipe=campaign_recipe,
        )
        providers[held_family] = provider
        fold_receipts.append(receipt)
        fold_provider_resource_receipts.append(
            campaign_recipe.provider_resource_receipt(provider)
        )
        fold_provider_audit_receipts.append(
            campaign_recipe.provider_audit_receipt(provider)
        )
        provider_sha256 = _require_sha256(
            provider.artifact_sha256,
            label="OOF provider",
        )
        fold_provider_sha256s.append(provider_sha256)
        fold_provider_sha256_by_family[held_family] = provider_sha256
        parent_artifact.validate_integrity()
        if (
            parent_artifact.artifact_sha256 != parent_artifact_sha256
            or getattr(parent_h4, "artifact_sha256") != parent_h4_sha256
            or _canonical_json_bytes(canonical_records)
            != record_payload_before_fits
        ):
            raise RuntimeError(
                "fold fitting mutated the alpha0 parent or fit records"
            )

    # Phase B: fresh source authority plus the corresponding OOF provider.
    candidate_observations: list[
        GemmaH4DampingFiniteNLLObservation
    ] = []
    candidate_execution_sha256s: list[str] = []
    candidate_execution_sha256_by_example: dict[str, str] = {}
    for example in panel.examples:
        source, source_logits, indices, targets, _positions = (
            _source_authority(adapter=adapter, example=example)
        )
        provider = providers[example.family_id]
        with torch.no_grad():
            candidate_execution = bridge.execute(
                adapter,
                example.batch.model_inputs,
                x4_head=x4_head,
                h4_head=provider,
            )
        _validate_execution(
            candidate_execution,
            example_model_inputs_sha256=example.model_inputs_sha256,
            bridge_binding_sha256=bridge.bridge_binding_sha256,
            x4_head=x4_head,
            h4_head=provider,
            label="OOF candidate",
        )
        candidate_logits = _gather_logits(
            getattr(candidate_execution, "logits", None),
            indices,
        )
        candidate_observation = _observation(
            example=example,
            source_logits=source_logits,
            candidate_logits=candidate_logits,
            targets=targets,
        )
        if not _same_source(
            parent_by_example[example.example_id],
            candidate_observation,
        ):
            raise RuntimeError(
                "phase-B source authority differs from phase A"
            )
        candidate_observations.append(candidate_observation)
        candidate_execution_sha256 = _require_sha256(
            getattr(candidate_execution, "artifact_sha256", None),
            label="OOF candidate execution",
        )
        candidate_execution_sha256s.append(candidate_execution_sha256)
        candidate_execution_sha256_by_example[
            example.example_id
        ] = candidate_execution_sha256
        if (
            gemma3_l3_l4_shadow_model_inputs_sha256(
                example.batch.model_inputs
            )
            != example.model_inputs_sha256
        ):
            raise RuntimeError("campaign model inputs changed in phase B")
        del (
            source,
            source_logits,
            indices,
            targets,
            provider,
            candidate_execution,
            candidate_logits,
            candidate_observation,
        )

    canonical_candidate = tuple(
        sorted(candidate_observations, key=lambda row: row.example_id)
    )
    canonical_folds = tuple(
        sorted(fold_receipts, key=lambda row: str(row["held_family_id"]))
    )
    canonical_resource_receipts = {
        _canonical_json_bytes(value)
        for value in fold_provider_resource_receipts
    }
    canonical_provider_audit_receipts = {
        _canonical_json_bytes(value)
        for value in fold_provider_audit_receipts
    }
    if (
        len(canonical_resource_receipts) != 1
        or len(canonical_provider_audit_receipts) != 1
    ):
        raise RuntimeError("OOF provider resource geometry differs by fold")
    fold_projection_values = tuple(
        row.get(campaign_recipe.fold_projection_field)
        for row in canonical_folds
    )
    if any(type(value) is not bool for value in fold_projection_values):
        raise ValueError(
            "OOF fold receipt omitted its recipe projection decision"
        )
    fold_projection_count = sum(
        bool(value) for value in fold_projection_values
    )
    provider_resources = dict(fold_provider_resource_receipts[0])
    provider_audit = dict(fold_provider_audit_receipts[0])
    learned_parameter_count = int(
        provider_resources["learned_parameter_count"]
    )
    logical_macs_per_token = int(
        provider_resources["logical_macs_per_token_upper_bound"]
    )
    residual_width = int(
        getattr(bridge, "residual_width", logical_macs_per_token)
    )
    if residual_width <= 0:
        raise ValueError("bridge residual width must be positive")
    campaign_recipe.validate_resource_envelope(
        resources=provider_resources,
        residual_width=residual_width,
    )
    resource_payload: dict[str, object] = {
        "learned_parameter_count": learned_parameter_count,
        "logical_macs_per_token_upper_bound": logical_macs_per_token,
        **{
            key: value
            for key, value in provider_resources.items()
            if key
            not in {
                "learned_parameter_count",
                "logical_macs_per_token_upper_bound",
            }
        },
        "serving_model_forward_count": 1,
        "parent_head_reused_not_duplicated": True,
        "parent_artifact_sha256": parent_artifact_sha256,
        "parent_h4_head_sha256": parent_h4_sha256,
        "candidate_provider_artifact_sha256_by_family": dict(
            sorted(fold_provider_sha256_by_family.items())
        ),
        "residual_width": residual_width,
    }
    resources = {
        **resource_payload,
        "resource_receipt_sha256": _sha256(
            _RESOURCE_DOMAIN,
            resource_payload,
        ),
    }
    audit: dict[str, object] = {
        "execution_mode": (
            "fit_only_two_phase_family_blocked_iterative_residual"
        ),
        "example_count": _EXPECTED_EXAMPLE_COUNT,
        "family_count": _EXPECTED_FAMILY_COUNT,
        "outer_fold_count": _EXPECTED_FAMILY_COUNT,
        **dict(campaign_recipe.audit_recipe_fields),
        **provider_audit,
        **campaign_recipe.parent_tensor_audit_receipt(parent_h4),
        "phase_a_source_forward_count": 16,
        "phase_a_parent_vjp_forward_count": 16,
        "phase_b_source_forward_count": 16,
        "phase_b_candidate_forward_count": 16,
        "total_model_forward_count": 64,
        "model_forward_count_per_example": 4,
        "one_semantic_candidate_per_iteration": True,
        "family_blocked_leave_one_family_out": True,
        "source_rerun_between_phases": True,
        "source_identity_equal_across_phases": True,
        "parent_observation_count": 16,
        "candidate_observation_count": 16,
        "fit_record_count": 16,
        "fit_records_scalar_hash_only": True,
        "candidate_executions_released_between_examples": True,
        "raw_prompts_retained": False,
        "raw_token_ids_retained": False,
        "raw_logits_retained": False,
        "raw_activations_retained": False,
        "gradient_tensors_retained": False,
        "model_weights_retained": False,
        "source_model_sha256": source_model_sha256,
        "source_execution_sha256": source_execution_sha256,
        "parent_artifact_sha256": parent_artifact_sha256,
        "parent_h4_artifact_sha256": parent_h4_sha256,
        "accepted_x4_head_sha256": _require_sha256(
            getattr(x4_head, "artifact_sha256", None),
            label="accepted X4 head",
        ),
        "fit_manifest_sha256": panel.manifest_sha256,
        "residual_width": residual_width,
        "parent_prepared_float_scalar_count": int(
            getattr(parent_artifact, "prepared_float_scalar_count")
        ),
        "parent_logical_macs_per_token_upper_bound": int(
            getattr(
                parent_artifact,
                "logical_macs_per_token_upper_bound",
            )
        ),
        "bridge_binding_sha256": bridge.bridge_binding_sha256,
        "parent_execution_sha256s": tuple(
            sorted(parent_execution_sha256s)
        ),
        "parent_execution_sha256_by_example": dict(
            sorted(parent_execution_sha256_by_example.items())
        ),
        "model_inputs_sha256_by_example": dict(
            sorted(model_inputs_sha256_by_example.items())
        ),
        "candidate_execution_sha256s": tuple(
            sorted(candidate_execution_sha256s)
        ),
        "candidate_execution_sha256_by_example": dict(
            sorted(candidate_execution_sha256_by_example.items())
        ),
        "fold_provider_artifact_sha256s": tuple(
            sorted(fold_provider_sha256s)
        ),
        "fold_provider_artifact_sha256_by_family": dict(
            sorted(fold_provider_sha256_by_family.items())
        ),
        campaign_recipe.fold_projection_count_audit_field: (
            fold_projection_count
        ),
        campaign_recipe.projection_interpretation_audit_field: (
            campaign_recipe.projection_interpretation
        ),
        "selection_input_opened": False,
        "guard_input_opened": False,
        "calibration_b_opened": False,
        "assessment_input_opened": False,
        "development_only": True,
    }
    records_by_example = {
        str(row["example_id"]): row for row in canonical_records
    }
    folds_by_family = {
        str(row["held_family_id"]): row for row in canonical_folds
    }
    candidate_by_example = {
        row.example_id: row for row in canonical_candidate
    }
    oof_rows: list[Mapping[str, object]] = []
    for parent_observation in canonical_parent:
        example_id = parent_observation.example_id
        family_id = parent_observation.family_id
        record = records_by_example[example_id]
        fold = folds_by_family[family_id]
        candidate_observation = candidate_by_example[example_id]
        oof_rows.append(
            campaign_recipe.build_oof_row(
                parent_observation=parent_observation,
                candidate_observation=candidate_observation,
                fit_record=record,
                fold_receipt=fold,
                provider_artifact_sha256=(
                    fold_provider_sha256_by_family[family_id]
                ),
                candidate_execution_sha256=(
                    candidate_execution_sha256_by_example[example_id]
                ),
            )
        )
    canonical_oof = tuple(
        sorted(oof_rows, key=lambda row: str(row["example_id"]))
    )
    collection = GemmaIterativeResidualLiveCollection(
        fit_records=canonical_records,
        parent_observations=canonical_parent,
        candidate_observations=canonical_candidate,
        oof_rows=canonical_oof,
        fold_receipts=canonical_folds,
        resources=resources,
        audit=audit,
    )
    report_arguments = {
        "parent_observations": canonical_parent,
        "candidate_observations": canonical_candidate,
        "oof_rows": canonical_oof,
        "fit_records": canonical_records,
        "fold_receipts": canonical_folds,
        "manifest": manifest,
        "resources": resources,
        "lineage": {} if lineage is None else dict(lineage),
        "audit": audit,
    }
    report = dict(
        build_report(
            **report_arguments,
            retained_full_fit_receipt=None,
            provisional=True,
        )
    )
    _assert_scalar_hash_only(report, path="campaign report")

    retained_provider: Gemma3L3L4CorrectionProvider | None = None
    retained_receipt: Mapping[str, object] | None = None
    if _report_retained(report):
        retained_provider = fit_full(
            records=canonical_records,
            parent_h4=parent_h4,
        )
        if not isinstance(
            retained_provider,
            Gemma3L3L4CorrectionProvider,
        ):
            raise TypeError(
                "full fitter must return an authenticated H4 provider"
            )
        retained_provider.validate_integrity()
        if retained_provider.site != _H4_SITE:
            raise ValueError("full-fit provider is bound to the wrong site")
        retained_receipt = _retained_provider_receipt(
            provider=retained_provider,
            parent_artifact_sha256=parent_artifact_sha256,
            parent_h4_sha256=parent_h4_sha256,
            bridge_binding_sha256=bridge.bridge_binding_sha256,
            recipe=campaign_recipe,
        )
        retained_resources = {
            key: retained_receipt[key]
            for key in provider_resources
        }
        if retained_resources != provider_resources:
            raise RuntimeError(
                "retained provider resource geometry differs from OOF folds"
            )
        if (
            campaign_recipe.provider_audit_receipt(retained_provider)
            != provider_audit
        ):
            raise RuntimeError(
                "retained provider audit geometry differs from OOF folds"
            )
        report = dict(
            build_report(
                **report_arguments,
                retained_full_fit_receipt=retained_receipt,
                provisional=False,
            )
        )
        _assert_scalar_hash_only(report, path="final campaign report")
    parent_artifact.validate_integrity()
    if (
        parent_artifact.artifact_sha256 != parent_artifact_sha256
        or getattr(parent_h4, "artifact_sha256") != parent_h4_sha256
        or _canonical_json_bytes(canonical_records)
        != record_payload_before_fits
        or adapter.model_fingerprint() != source_model_sha256
        or adapter.execution_fingerprint() != source_execution_sha256
    ):
        raise RuntimeError("iterative campaign mutated its frozen parent")
    return GemmaIterativeResidualCampaignResult(
        collection=collection,
        report=report,
        retained_provider=retained_provider,
    )


def run_gemma_iterative_residual_campaign(
    *,
    panel: GemmaProgressivePanel,
    adapter: _AdapterLike,
    bridge: _BridgeLike,
    parent_artifact: _ArtifactLike,
    make_fit_record: FitRecordBuilder,
    fit_fold: FoldFitter,
    build_report: ReportBuilder,
    fit_full: FullFitter,
    output: Path | str,
    lineage: Mapping[str, object] | None = None,
    recipe: GemmaIterativeResidualCampaignRecipe | None = None,
) -> GemmaIterativeResidualCampaignResult:
    """Run and publish one already-materialized fit-only live campaign."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite iterative residual campaign report"
        )
    result = collect_gemma_iterative_residual_campaign_live(
        panel=panel,
        adapter=adapter,
        bridge=bridge,
        parent_artifact=parent_artifact,
        make_fit_record=make_fit_record,
        fit_fold=fit_fold,
        build_report=build_report,
        fit_full=fit_full,
        lineage=lineage,
        recipe=recipe,
    )
    publish_gemma_iterative_residual_campaign_report(
        destination,
        result.report,
    )
    return result


def publish_gemma_iterative_residual_campaign_report(
    path: Path | str,
    report: Mapping[str, object],
) -> None:
    """Atomically publish scalar JSON without replacing an existing file."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite iterative residual campaign report"
        )
    _assert_scalar_hash_only(report, path="campaign report")
    if _report_retained(report) and (
        report.get("retained_full_fit") is None
        and report.get("retained_full_fit_receipt") is None
    ):
        raise ValueError(
            "retained campaign report requires its full-fit provider receipt"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    stage = Path(stage_name)
    try:
        with stage.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, destination)
        directory = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        stage.unlink(missing_ok=True)
