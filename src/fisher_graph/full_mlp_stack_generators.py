"""Aggregate-before-modes fitting for one complete native MLP layer.

This module fits the complete layer contribution before discovering residual
computational modes.  It therefore avoids fitting and lowering one modal
generator per Fisher fragment:

``full layer rows -> residual modes -> coordinate generator -> fused residual``.

The selected coordinate generator is retained for analysis provenance.  The
dense fused plan is independently executable and intentionally carries no
``parameter_cluster_fragment_sha256`` binding, because it replaces the whole
native MLP rather than one singular fragment.  Its source generator, basis,
Fisher coupling, and layer-superfragment lineage remain authenticated by the
binding and enclosing result artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch

from .computational_modes import (
    ComputationalModeBasis,
    ComputationalModeBinding,
    ComputationalModeRateCurve,
    fit_computational_mode_rate_curve,
)
from .gemma3_full_mlp_stack_rows import FullMLPStackLayerRows
from .modal_generators import (
    ModalGeneratorBinding,
    ModalGeneratorConfig,
    ModalGeneratorFactors,
    ModalGeneratorPlan,
    ModalGeneratorRateCurve,
    fit_modal_generator_rate_curve,
)
from .parameter_layer_superfragments import ParameterLayerSuperfragment


__all__ = [
    "FullMLPStackGeneratorFit",
    "fit_full_mlp_stack_generators",
]


_FORMAT_VERSION = 1
_KIND = "fisher_graph.full_mlp_stack_generator_fit"
_DOMAIN = b"fisher_graph.full_mlp_stack_generators.fit.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_prompt_text": False,
    "contains_raw_fit_rows": False,
    "contains_raw_eval_rows": False,
    "contains_fisher_row_weights": False,
    "contains_computational_mode_basis": True,
    "contains_generator_weights": True,
    "evaluation_used_for_basis_fit": False,
    "evaluation_used_for_generator_fit": False,
    "evaluation_used_for_rank_selection": False,
    "dense_plan_executable_without_fragment_lowering": True,
}


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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_fields(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


def _positive_int_tuple(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            f"{label} must be positive, unique, and strictly increasing"
        )
    return result


def _validate_rows(
    fit_rows: FullMLPStackLayerRows,
    eval_rows: FullMLPStackLayerRows,
    superfragment: ParameterLayerSuperfragment,
) -> None:
    if not isinstance(fit_rows, FullMLPStackLayerRows):
        raise TypeError("fit_rows must be FullMLPStackLayerRows")
    if not isinstance(eval_rows, FullMLPStackLayerRows):
        raise TypeError("eval_rows must be FullMLPStackLayerRows")
    if not isinstance(superfragment, ParameterLayerSuperfragment):
        raise TypeError(
            "superfragment must be a ParameterLayerSuperfragment"
        )
    superfragment.validate_integrity()
    shared_fields = (
        "layer_ordinal",
        "layer_id",
        "input_site",
        "activation_site",
        "output_site",
        "intermediate_width",
        "fragment_ids",
        "fragment_sha256s",
    )
    if any(
        getattr(fit_rows, name) != getattr(eval_rows, name)
        for name in shared_fields
    ):
        raise ValueError("fit and evaluation rows describe different layers")
    if (
        fit_rows.layer_ordinal != superfragment.layer_ordinal
        or fit_rows.layer_id != superfragment.layer_id
        or fit_rows.input_site != superfragment.input_site
        or fit_rows.activation_site != superfragment.activation_site
        or fit_rows.output_site != superfragment.output_site
        or fit_rows.intermediate_width != superfragment.mode_count
        or fit_rows.inputs.shape[1] != superfragment.input_width
        or eval_rows.inputs.shape[1] != superfragment.input_width
        or fit_rows.contributions.shape[1] != superfragment.output_width
        or eval_rows.contributions.shape[1] != superfragment.output_width
    ):
        raise ValueError(
            "full-layer rows do not match the layer superfragment schema"
        )
    if tuple(sorted(fit_rows.fragment_sha256s)) != (
        superfragment.member_fragment_sha256s
    ):
        raise ValueError(
            "full-layer rows do not authenticate every superfragment member"
        )
    if fit_rows.row_key_sha256 == eval_rows.row_key_sha256 or not set(
        fit_rows.row_keys
    ).isdisjoint(eval_rows.row_keys):
        raise ValueError("fit and evaluation row identities must be disjoint")


def _dense_fused_residual_plan(
    coordinate_plan: ModalGeneratorPlan,
    basis: ComputationalModeBasis,
    *,
    superfragment: ParameterLayerSuperfragment,
    superfragment_plan_sha256: str,
) -> ModalGeneratorPlan:
    factors = coordinate_plan.factors
    decoder = basis.decoder_basis
    fused_output = (factors.output_factor @ decoder).contiguous()
    fused_bias = basis.mean_bias.clone()
    if factors.bias is not None:
        fused_bias = fused_bias + factors.bias @ decoder
    fused_factors = ModalGeneratorFactors(
        rank=coordinate_plan.rank,
        input_factor=factors.input_factor,
        output_factor=fused_output,
        bias=fused_bias.contiguous(),
    )
    binding = ModalGeneratorBinding.create(
        generator_id=(
            f"{coordinate_plan.binding.generator_id}.dense-full-layer"
        ),
        input_kind=coordinate_plan.binding.input_kind,
        input_site=coordinate_plan.binding.input_site,
        output_site=coordinate_plan.binding.output_site,
        source_model_sha256=coordinate_plan.binding.source_model_sha256,
        input_catalog_sha256=(
            coordinate_plan.binding.input_catalog_sha256
        ),
        output_catalog_sha256=superfragment.parameter_catalog_sha256,
        cluster_plan_sha256=superfragment_plan_sha256,
        fit_split_sha256=coordinate_plan.binding.fit_split_sha256,
        eval_split_sha256=coordinate_plan.binding.eval_split_sha256,
        target_kind="cluster_residual_contribution",
        fisher_coupling_sha256=(
            coordinate_plan.binding.fisher_coupling_sha256
        ),
        computational_mode_basis_sha256=basis.artifact_sha256,
        # Whole-layer execution must not claim one singular fragment.
        parameter_cluster_fragment_sha256=None,
        source_generator_plan_sha256=coordinate_plan.artifact_sha256,
    )
    config = ModalGeneratorConfig(
        ranks=(coordinate_plan.rank,),
        fit_intercept=True,
        ridge=coordinate_plan.config.ridge,
        bias_mac_policy=coordinate_plan.config.bias_mac_policy,
        selection_rule="fixed_rank",
        selected_rank=coordinate_plan.rank,
    )
    matrix_count = (
        fused_factors.input_width * fused_factors.rank
        + fused_factors.rank * fused_factors.output_width
    )
    macs_per_token = matrix_count
    if config.bias_mac_policy == "count_bias_additions":
        macs_per_token += fused_factors.output_width
    return ModalGeneratorPlan(
        binding=binding,
        config=config,
        factors=fused_factors,
        parameter_count=matrix_count + fused_factors.output_width,
        macs_per_token=macs_per_token,
    )


@dataclass(frozen=True, slots=True)
class FullMLPStackGeneratorFit:
    """Authenticated full-layer mode, generator, and dense-fusion result."""

    superfragment: ParameterLayerSuperfragment
    superfragment_plan_sha256: str
    fit_row_key_sha256: str
    eval_row_key_sha256: str
    computational_modes: ComputationalModeRateCurve
    coordinate_generators: ModalGeneratorRateCurve
    selected_basis: ComputationalModeBasis
    selected_coordinate_plan: ModalGeneratorPlan
    dense_fused_residual_plan: ModalGeneratorPlan
    artifact_sha256: str = ""
    artifact_kind: str = _KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_raw_fit_rows: bool = False
    contains_raw_eval_rows: bool = False
    contains_fisher_row_weights: bool = False
    contains_computational_mode_basis: bool = True
    contains_generator_weights: bool = True
    evaluation_used_for_basis_fit: bool = False
    evaluation_used_for_generator_fit: bool = False
    evaluation_used_for_rank_selection: bool = False
    dense_plan_executable_without_fragment_lowering: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.superfragment, ParameterLayerSuperfragment):
            raise TypeError(
                "superfragment must be ParameterLayerSuperfragment"
            )
        self.superfragment.validate_integrity()
        for name in (
            "superfragment_plan_sha256",
            "fit_row_key_sha256",
            "eval_row_key_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.fit_row_key_sha256 == self.eval_row_key_sha256:
            raise ValueError("fit and evaluation row-key hashes must differ")
        if not isinstance(
            self.computational_modes,
            ComputationalModeRateCurve,
        ):
            raise TypeError(
                "computational_modes must be ComputationalModeRateCurve"
            )
        if not isinstance(
            self.coordinate_generators,
            ModalGeneratorRateCurve,
        ):
            raise TypeError(
                "coordinate_generators must be ModalGeneratorRateCurve"
            )
        if not isinstance(self.selected_basis, ComputationalModeBasis):
            raise TypeError(
                "selected_basis must be ComputationalModeBasis"
            )
        if not isinstance(
            self.selected_coordinate_plan,
            ModalGeneratorPlan,
        ):
            raise TypeError(
                "selected_coordinate_plan must be ModalGeneratorPlan"
            )
        if not isinstance(
            self.dense_fused_residual_plan,
            ModalGeneratorPlan,
        ):
            raise TypeError(
                "dense_fused_residual_plan must be ModalGeneratorPlan"
            )

        # Strict nested round trips authenticate every ladder tensor, not just
        # each curve's outer digest.
        authenticated_modes = ComputationalModeRateCurve.from_state_dict(
            self.computational_modes.state_dict()
        )
        authenticated_generators = ModalGeneratorRateCurve.from_state_dict(
            self.coordinate_generators.state_dict()
        )
        self.selected_basis.validate_integrity()
        self.selected_coordinate_plan.validate_integrity()
        self.dense_fused_residual_plan.validate_integrity()
        curve_basis = authenticated_modes.selected_basis
        curve_coordinate_plan = authenticated_generators.selected_plan
        if (
            curve_basis is None
            or curve_basis.artifact_sha256
            != self.selected_basis.artifact_sha256
        ):
            raise ValueError(
                "selected_basis is not the fixed selection from its curve"
            )
        if (
            curve_coordinate_plan is None
            or curve_coordinate_plan.artifact_sha256
            != self.selected_coordinate_plan.artifact_sha256
        ):
            raise ValueError(
                "selected_coordinate_plan is not the fixed generator "
                "selection"
            )
        self._validate_lineage()
        expected_dense = _dense_fused_residual_plan(
            self.selected_coordinate_plan,
            self.selected_basis,
            superfragment=self.superfragment,
            superfragment_plan_sha256=self.superfragment_plan_sha256,
        )
        if (
            self.dense_fused_residual_plan.artifact_sha256
            != expected_dense.artifact_sha256
        ):
            raise ValueError(
                "dense residual plan is not the authenticated coordinate/"
                "basis fusion"
            )
        for name, expected in _SAFETY_METADATA.items():
            if getattr(self, name) is not expected:
                raise ValueError(
                    "full-layer generator safety metadata is invalid"
                )
        if (
            self.artifact_kind != _KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("full-layer generator fit header is invalid")
        computed = _json_sha256(self._payload(), domain=_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("full-layer generator fit hash mismatch")

    def _validate_lineage(self) -> None:
        superfragment = self.superfragment
        mode_binding = self.computational_modes.binding
        coordinate = self.selected_coordinate_plan
        coordinate_binding = coordinate.binding
        dense = self.dense_fused_residual_plan
        dense_binding = dense.binding
        checks = (
            (
                mode_binding.source_model_sha256,
                superfragment.source_model_sha256,
                "mode/superfragment source model",
            ),
            (
                mode_binding.parameter_catalog_sha256,
                superfragment.parameter_catalog_sha256,
                "mode/superfragment parameter catalog",
            ),
            (
                mode_binding.fisher_coupling_sha256,
                superfragment.source_fisher_coupling_sha256,
                "mode/superfragment Fisher coupling",
            ),
            (
                mode_binding.parameter_cluster_sha256,
                superfragment.artifact_sha256,
                "mode/superfragment artifact",
            ),
            (
                mode_binding.output_site,
                superfragment.output_site,
                "mode/superfragment output site",
            ),
            (
                coordinate_binding.source_model_sha256,
                superfragment.source_model_sha256,
                "coordinate/superfragment source model",
            ),
            (
                coordinate_binding.input_catalog_sha256,
                superfragment.input_catalog_sha256,
                "coordinate/superfragment input catalog",
            ),
            (
                coordinate_binding.output_catalog_sha256,
                self.selected_basis.artifact_sha256,
                "coordinate output catalog/basis",
            ),
            (
                coordinate_binding.cluster_plan_sha256,
                self.superfragment_plan_sha256,
                "coordinate/superfragment plan",
            ),
            (
                coordinate_binding.fisher_coupling_sha256,
                superfragment.source_fisher_coupling_sha256,
                "coordinate/superfragment Fisher coupling",
            ),
            (
                coordinate_binding.computational_mode_basis_sha256,
                self.selected_basis.artifact_sha256,
                "coordinate/basis",
            ),
            (
                coordinate_binding.parameter_cluster_fragment_sha256,
                superfragment.artifact_sha256,
                "coordinate/superfragment",
            ),
            (
                dense_binding.source_model_sha256,
                superfragment.source_model_sha256,
                "dense/superfragment source model",
            ),
            (
                dense_binding.input_catalog_sha256,
                superfragment.input_catalog_sha256,
                "dense/superfragment input catalog",
            ),
            (
                dense_binding.output_catalog_sha256,
                superfragment.parameter_catalog_sha256,
                "dense output/parameter catalog",
            ),
            (
                dense_binding.cluster_plan_sha256,
                self.superfragment_plan_sha256,
                "dense/superfragment plan",
            ),
            (
                dense_binding.fisher_coupling_sha256,
                superfragment.source_fisher_coupling_sha256,
                "dense/superfragment Fisher coupling",
            ),
            (
                dense_binding.computational_mode_basis_sha256,
                self.selected_basis.artifact_sha256,
                "dense/basis",
            ),
            (
                dense_binding.source_generator_plan_sha256,
                coordinate.artifact_sha256,
                "dense/source coordinate generator",
            ),
            (
                mode_binding.fit_split_sha256,
                coordinate_binding.fit_split_sha256,
                "mode/coordinate fit split",
            ),
            (
                mode_binding.eval_split_sha256,
                coordinate_binding.eval_split_sha256,
                "mode/coordinate evaluation split",
            ),
            (
                coordinate_binding.fit_split_sha256,
                dense_binding.fit_split_sha256,
                "coordinate/dense fit split",
            ),
            (
                coordinate_binding.eval_split_sha256,
                dense_binding.eval_split_sha256,
                "coordinate/dense evaluation split",
            ),
            (
                coordinate_binding.input_site,
                superfragment.input_site,
                "coordinate/superfragment input site",
            ),
            (
                coordinate_binding.output_site,
                superfragment.output_site,
                "coordinate/superfragment output site",
            ),
            (
                dense_binding.input_site,
                superfragment.input_site,
                "dense/superfragment input site",
            ),
            (
                dense_binding.output_site,
                superfragment.output_site,
                "dense/superfragment output site",
            ),
        )
        for first, second, label in checks:
            if first != second:
                raise ValueError(f"{label} mismatch")
        if mode_binding.source_kind != "layer_fragment":
            raise ValueError(
                "whole-layer computational modes must use layer source kind"
            )
        if coordinate_binding.target_kind != (
            "computational_mode_coordinates"
        ):
            raise ValueError("coordinate generator target kind is invalid")
        if dense_binding.target_kind != "cluster_residual_contribution":
            raise ValueError("dense fused target kind is invalid")
        if dense_binding.parameter_cluster_fragment_sha256 is not None:
            raise ValueError(
                "whole-layer dense execution cannot bind one singular "
                "fragment"
            )
        if (
            coordinate.output_width != self.selected_basis.rank
            or dense.input_width != superfragment.input_width
            or dense.output_width != superfragment.output_width
            or self.computational_modes.fit_rows
            != self.coordinate_generators.zero_fit_metrics.observations
            or self.computational_modes.eval_rows
            != self.coordinate_generators.zero_eval_metrics.observations
        ):
            raise ValueError(
                "full-layer generator widths or observation counts drifted"
            )

    @property
    def executable_plan(self) -> ModalGeneratorPlan:
        return self.dense_fused_residual_plan

    @property
    def selected_mode_rank(self) -> int:
        return self.selected_basis.rank

    @property
    def selected_generator_rank(self) -> int:
        return self.selected_coordinate_plan.rank

    @property
    def structural_metadata(self) -> dict[str, object]:
        return {
            "aggregation_order": "full_native_layer_before_modes",
            "replacement_scope": "complete_native_mlp",
            "layer_ordinal": self.superfragment.layer_ordinal,
            "layer_id": self.superfragment.layer_id,
            "input_site": self.superfragment.input_site,
            "output_site": self.superfragment.output_site,
            "input_width": self.superfragment.input_width,
            "intermediate_width": self.superfragment.mode_count,
            "residual_width": self.superfragment.output_width,
            "source_fragment_count": (
                self.superfragment.member_fragment_count
            ),
            "source_fragment_sha256s": (
                self.superfragment.member_fragment_sha256s
            ),
            "selected_mode_rank": self.selected_mode_rank,
            "selected_generator_rank": self.selected_generator_rank,
            "singular_fragment_lowering_required": False,
            "dense_plan_parameter_cluster_fragment_sha256": None,
        }

    @property
    def resource_metadata(self) -> dict[str, object]:
        native_parameters = self.superfragment.native_parameter_count
        dense_parameters = self.dense_fused_residual_plan.parameter_count
        native_macs = native_parameters
        dense_macs = self.dense_fused_residual_plan.macs_per_token
        coordinate_parameters = (
            self.selected_coordinate_plan.parameter_count
        )
        return {
            "native_mlp_parameter_count": native_parameters,
            "native_mlp_linear_macs_per_token": native_macs,
            "selected_basis_stored_scalar_count": (
                self.selected_basis.stored_scalar_count
            ),
            "coordinate_generator_parameter_count": coordinate_parameters,
            "coordinate_generator_macs_per_token": (
                self.selected_coordinate_plan.macs_per_token
            ),
            "selected_unfused_stored_scalar_count": (
                self.selected_basis.stored_scalar_count
                + coordinate_parameters
            ),
            "dense_fused_parameter_count": dense_parameters,
            "dense_fused_macs_per_token": dense_macs,
            "net_stored_parameter_savings": (
                native_parameters - dense_parameters
            ),
            "net_linear_macs_saved_per_token": native_macs - dense_macs,
            "dense_parameter_reduction_fraction": (
                1.0 - dense_parameters / native_parameters
            ),
            "dense_linear_mac_reduction_fraction": (
                1.0 - dense_macs / native_macs
            ),
            "dense_execution_stores_basis_separately": False,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "superfragment_sha256": self.superfragment.artifact_sha256,
            "superfragment_plan_sha256": (
                self.superfragment_plan_sha256
            ),
            "source_fragment_plan_sha256": (
                self.superfragment.source_fragment_plan_sha256
            ),
            "source_cluster_plan_sha256": (
                self.superfragment.source_cluster_plan_sha256
            ),
            "source_fisher_coupling_sha256": (
                self.superfragment.source_fisher_coupling_sha256
            ),
            "parameter_catalog_sha256": (
                self.superfragment.parameter_catalog_sha256
            ),
            "source_model_sha256": self.superfragment.source_model_sha256,
            "fit_row_key_sha256": self.fit_row_key_sha256,
            "eval_row_key_sha256": self.eval_row_key_sha256,
            "computational_modes_sha256": (
                self.computational_modes.artifact_sha256
            ),
            "coordinate_generators_sha256": (
                self.coordinate_generators.artifact_sha256
            ),
            "selected_basis_sha256": self.selected_basis.artifact_sha256,
            "selected_coordinate_plan_sha256": (
                self.selected_coordinate_plan.artifact_sha256
            ),
            "dense_fused_residual_plan_sha256": (
                self.dense_fused_residual_plan.artifact_sha256
            ),
            "structural_metadata": self.structural_metadata,
            "resource_metadata": self.resource_metadata,
            **_SAFETY_METADATA,
        }

    def validate_integrity(self) -> None:
        self.superfragment.validate_integrity()
        self.selected_basis.validate_integrity()
        self.selected_coordinate_plan.validate_integrity()
        self.dense_fused_residual_plan.validate_integrity()
        self._validate_lineage()
        if _json_sha256(self._payload(), domain=_DOMAIN) != self.artifact_sha256:
            raise ValueError("full-layer generator fit hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "superfragment": self.superfragment.state_dict(),
            "computational_modes": self.computational_modes.state_dict(),
            "coordinate_generators": self.coordinate_generators.state_dict(),
            "selected_basis": self.selected_basis.state_dict(),
            "selected_coordinate_plan": (
                self.selected_coordinate_plan.state_dict()
            ),
            "dense_fused_residual_plan": (
                self.dense_fused_residual_plan.state_dict()
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FullMLPStackGeneratorFit:
        expected = {
            "artifact_kind",
            "format_version",
            "superfragment_sha256",
            "superfragment_plan_sha256",
            "source_fragment_plan_sha256",
            "source_cluster_plan_sha256",
            "source_fisher_coupling_sha256",
            "parameter_catalog_sha256",
            "source_model_sha256",
            "fit_row_key_sha256",
            "eval_row_key_sha256",
            "computational_modes_sha256",
            "coordinate_generators_sha256",
            "selected_basis_sha256",
            "selected_coordinate_plan_sha256",
            "dense_fused_residual_plan_sha256",
            "structural_metadata",
            "resource_metadata",
            *set(_SAFETY_METADATA),
            "artifact_sha256",
            "superfragment",
            "computational_modes",
            "coordinate_generators",
            "selected_basis",
            "selected_coordinate_plan",
            "dense_fused_residual_plan",
        }
        _strict_fields(
            state,
            expected=expected,
            label="full-layer generator fit",
        )
        for name in (
            "superfragment",
            "computational_modes",
            "coordinate_generators",
            "selected_basis",
            "selected_coordinate_plan",
            "dense_fused_residual_plan",
        ):
            if not isinstance(state[name], Mapping):
                raise TypeError(f"{name} state must be a mapping")
        superfragment = ParameterLayerSuperfragment.from_state_dict(
            state["superfragment"]
        )
        computational_modes = ComputationalModeRateCurve.from_state_dict(
            state["computational_modes"]
        )
        coordinate_generators = ModalGeneratorRateCurve.from_state_dict(
            state["coordinate_generators"]
        )
        selected_basis = ComputationalModeBasis.from_state_dict(
            state["selected_basis"]
        )
        selected_coordinate_plan = ModalGeneratorPlan.from_state_dict(
            state["selected_coordinate_plan"]
        )
        dense_fused_residual_plan = ModalGeneratorPlan.from_state_dict(
            state["dense_fused_residual_plan"]
        )
        nested_hashes = (
            ("superfragment_sha256", superfragment.artifact_sha256),
            (
                "computational_modes_sha256",
                computational_modes.artifact_sha256,
            ),
            (
                "coordinate_generators_sha256",
                coordinate_generators.artifact_sha256,
            ),
            ("selected_basis_sha256", selected_basis.artifact_sha256),
            (
                "selected_coordinate_plan_sha256",
                selected_coordinate_plan.artifact_sha256,
            ),
            (
                "dense_fused_residual_plan_sha256",
                dense_fused_residual_plan.artifact_sha256,
            ),
        )
        for name, actual in nested_hashes:
            if state[name] != actual:
                raise ValueError(f"{name} does not match nested artifact")
        restored = cls(
            superfragment=superfragment,
            superfragment_plan_sha256=state[
                "superfragment_plan_sha256"
            ],  # type: ignore[arg-type]
            fit_row_key_sha256=state[
                "fit_row_key_sha256"
            ],  # type: ignore[arg-type]
            eval_row_key_sha256=state[
                "eval_row_key_sha256"
            ],  # type: ignore[arg-type]
            computational_modes=computational_modes,
            coordinate_generators=coordinate_generators,
            selected_basis=selected_basis,
            selected_coordinate_plan=selected_coordinate_plan,
            dense_fused_residual_plan=dense_fused_residual_plan,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                name: state[name] for name in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )
        for name, expected_value in restored._payload().items():
            if state[name] != expected_value:
                raise ValueError(
                    "full-layer generator fit summaries are invalid"
                )
        return restored


def fit_full_mlp_stack_generators(
    fit_rows: FullMLPStackLayerRows,
    eval_rows: FullMLPStackLayerRows,
    *,
    superfragment: ParameterLayerSuperfragment,
    source_model_sha256: str,
    parameter_catalog_sha256: str,
    fisher_coupling_sha256: str,
    superfragment_plan_sha256: str,
    fit_split_sha256: str,
    eval_split_sha256: str,
    mode_ranks: Sequence[int],
    selected_mode_rank: int,
    generator_ranks: Sequence[int],
    selected_generator_rank: int,
    ridge: float = 0.0,
) -> FullMLPStackGeneratorFit:
    """Fit fixed full-layer mode/generator ladders and a dense fused plan."""

    _validate_rows(fit_rows, eval_rows, superfragment)
    for name, value in (
        ("source_model_sha256", source_model_sha256),
        ("parameter_catalog_sha256", parameter_catalog_sha256),
        ("fisher_coupling_sha256", fisher_coupling_sha256),
        ("superfragment_plan_sha256", superfragment_plan_sha256),
        ("fit_split_sha256", fit_split_sha256),
        ("eval_split_sha256", eval_split_sha256),
    ):
        _require_sha256(value, label=name)
    if (
        source_model_sha256 != superfragment.source_model_sha256
        or parameter_catalog_sha256
        != superfragment.parameter_catalog_sha256
        or fisher_coupling_sha256
        != superfragment.source_fisher_coupling_sha256
    ):
        raise ValueError(
            "explicit provenance hashes do not match the superfragment"
        )
    if fit_split_sha256 == eval_split_sha256:
        raise ValueError("fit and evaluation split hashes must differ")
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)):
        raise TypeError("ridge must be a real number")
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")

    mode_rank_values = _positive_int_tuple(
        mode_ranks,
        label="mode_ranks",
    )
    generator_rank_values = _positive_int_tuple(
        generator_ranks,
        label="generator_ranks",
    )
    if selected_mode_rank not in mode_rank_values:
        raise ValueError("selected_mode_rank must be in mode_ranks")
    if selected_generator_rank not in generator_rank_values:
        raise ValueError(
            "selected_generator_rank must be in generator_ranks"
        )
    maximum_mode_rank = min(
        fit_rows.observations,
        superfragment.output_width,
    )
    if mode_rank_values[-1] > maximum_mode_rank:
        raise ValueError(
            "mode ranks cannot exceed the fit-row/residual thin rank"
        )
    if generator_rank_values[-1] > selected_mode_rank:
        raise ValueError(
            "generator ranks cannot exceed the selected mode rank"
        )
    if generator_rank_values[-1] > superfragment.input_width:
        raise ValueError(
            "generator ranks cannot exceed the native input width"
        )

    mode_binding = ComputationalModeBinding.create(
        mode_set_id=(
            f"layer.{superfragment.layer_ordinal}.full-mlp-superfragment"
        ),
        source_kind="layer_fragment",
        output_site=superfragment.output_site,
        source_model_sha256=source_model_sha256,
        parameter_catalog_sha256=parameter_catalog_sha256,
        fisher_coupling_sha256=fisher_coupling_sha256,
        parameter_cluster_sha256=superfragment.artifact_sha256,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
    )
    computational_modes = fit_computational_mode_rate_curve(
        fit_rows.contributions,
        fit_rows.fisher_weights,
        eval_rows.contributions,
        eval_rows.fisher_weights,
        mode_rank_values,
        binding=mode_binding,
        selection_rule="fixed_rank",
        selected_rank=selected_mode_rank,
    )
    selected_basis = computational_modes.selected_basis
    if selected_basis is None:
        raise RuntimeError("fixed computational-mode rank was not selected")

    coordinate_binding = ModalGeneratorBinding.create(
        generator_id=(
            f"full-mlp.layer-{superfragment.layer_ordinal}."
            "modal-generator"
        ),
        input_kind="native_layer_input",
        input_site=superfragment.input_site,
        output_site=superfragment.output_site,
        source_model_sha256=source_model_sha256,
        input_catalog_sha256=superfragment.input_catalog_sha256,
        output_catalog_sha256=selected_basis.artifact_sha256,
        cluster_plan_sha256=superfragment_plan_sha256,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256=fisher_coupling_sha256,
        computational_mode_basis_sha256=selected_basis.artifact_sha256,
        # This coordinate analysis is bound to the aggregate layer record.
        parameter_cluster_fragment_sha256=superfragment.artifact_sha256,
    )
    coordinate_generators = fit_modal_generator_rate_curve(
        fit_rows.inputs,
        selected_basis.encode(fit_rows.contributions),
        fit_rows.fisher_weights,
        eval_rows.inputs,
        selected_basis.encode(eval_rows.contributions),
        generator_rank_values,
        binding=coordinate_binding,
        fisher_weights_eval=eval_rows.fisher_weights,
        fit_intercept=True,
        ridge=ridge,
        bias_mac_policy="matrix_multiplies_only",
        selection_rule="fixed_rank",
        selected_rank=selected_generator_rank,
    )
    selected_coordinate_plan = coordinate_generators.selected_plan
    if selected_coordinate_plan is None:
        raise RuntimeError("fixed modal-generator rank was not selected")
    dense_fused_residual_plan = _dense_fused_residual_plan(
        selected_coordinate_plan,
        selected_basis,
        superfragment=superfragment,
        superfragment_plan_sha256=superfragment_plan_sha256,
    )
    return FullMLPStackGeneratorFit(
        superfragment=superfragment,
        superfragment_plan_sha256=superfragment_plan_sha256,
        fit_row_key_sha256=fit_rows.row_key_sha256,
        eval_row_key_sha256=eval_rows.row_key_sha256,
        computational_modes=computational_modes,
        coordinate_generators=coordinate_generators,
        selected_basis=selected_basis,
        selected_coordinate_plan=selected_coordinate_plan,
        dense_fused_residual_plan=dense_fused_residual_plan,
    )
