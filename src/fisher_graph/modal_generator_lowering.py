"""Authenticated lowering from generated modes to graph and dense execution.

The coordinate-target modal generator predicts the frozen coordinates exposed
by :class:`~fisher_graph.computational_modes.ComputationalModeBasis`.  This
module validates the complete provenance chain and produces two equivalent
runtime forms:

* graph weights whose node state is exactly the generated computational-mode
  coordinate vector, with interactions operating in that coordinate system;
* a dense residual generator plan with the mode decoder algebraically fused
  into the generator for a conventional Gemma MLP replacement.

The lowering artifact retains authenticated copies of its source plan, mode
basis, fragment plan, graph weights, and fused plan.  A basis cannot therefore
be swapped without invalidating either tensor algebra or provenance hashes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re

import torch
from torch import Tensor

from .computational_modes import ComputationalModeBasis
from .modal_generator_graph import (
    LinearModalGeneratorNodeWeights,
    ModalGeneratorNode,
)
from .modal_generators import (
    ModalGeneratorBinding,
    ModalGeneratorConfig,
    ModalGeneratorFactors,
    ModalGeneratorPlan,
)
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)


__all__ = [
    "ModalGeneratorLowering",
    "lower_coordinate_modal_generator",
    "validate_coordinate_generator_compatibility",
]


_KIND = "fisher_graph.modal_generator_coordinate_lowering"
_FORMAT_VERSION = 1
_DOMAIN = b"fisher_graph.modal_generator_coordinate_lowering.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


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


def _strict_fields(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


def _authenticated_plan(plan: ModalGeneratorPlan) -> ModalGeneratorPlan:
    if not isinstance(plan, ModalGeneratorPlan):
        raise TypeError("generator_plan must be a ModalGeneratorPlan")
    return ModalGeneratorPlan.from_state_dict(plan.state_dict())


def _authenticated_basis(
    basis: ComputationalModeBasis,
) -> ComputationalModeBasis:
    if not isinstance(basis, ComputationalModeBasis):
        raise TypeError(
            "computational_mode_basis must be a ComputationalModeBasis"
        )
    return ComputationalModeBasis.from_state_dict(basis.state_dict())


def _authenticated_fragment_plan(
    plan: ParameterClusterLayerFragmentPlan,
) -> ParameterClusterLayerFragmentPlan:
    if not isinstance(plan, ParameterClusterLayerFragmentPlan):
        raise TypeError(
            "fragment_plan must be a ParameterClusterLayerFragmentPlan"
        )
    return ParameterClusterLayerFragmentPlan.from_state_dict(
        plan.state_dict()
    )


def _bound_fragment(
    basis: ComputationalModeBasis,
    fragment_plan: ParameterClusterLayerFragmentPlan,
) -> ParameterClusterLayerFragment:
    matching = tuple(
        fragment
        for fragment in fragment_plan.fragments
        if fragment.artifact_sha256
        == basis.binding.parameter_cluster_sha256
    )
    if len(matching) != 1:
        raise ValueError(
            "computational-mode basis must bind exactly one fragment in "
            "the supplied fragment plan"
        )
    return matching[0]


def validate_coordinate_generator_compatibility(
    generator_plan: ModalGeneratorPlan,
    computational_mode_basis: ComputationalModeBasis,
    fragment_plan: ParameterClusterLayerFragmentPlan,
) -> ParameterClusterLayerFragment:
    """Validate the complete coordinate-generator provenance and shape chain."""

    if not isinstance(generator_plan, ModalGeneratorPlan):
        raise TypeError("generator_plan must be a ModalGeneratorPlan")
    if not isinstance(computational_mode_basis, ComputationalModeBasis):
        raise TypeError(
            "computational_mode_basis must be a ComputationalModeBasis"
        )
    if not isinstance(fragment_plan, ParameterClusterLayerFragmentPlan):
        raise TypeError(
            "fragment_plan must be a ParameterClusterLayerFragmentPlan"
        )
    # Strict roundtrips authenticate all nested tensors before inspecting
    # scientific fields.
    plan = _authenticated_plan(generator_plan)
    basis = _authenticated_basis(computational_mode_basis)
    fragments = _authenticated_fragment_plan(fragment_plan)
    binding = plan.binding
    basis_binding = basis.binding
    fragment = _bound_fragment(basis, fragments)

    coordinate_target_kinds = {
        "computational_mode_coordinates",
        "relocated_computational_mode_coordinates",
    }
    if binding.target_kind not in coordinate_target_kinds:
        raise ValueError(
            "generator plan must target computational-mode coordinates"
        )
    relocated = (
        binding.target_kind == "relocated_computational_mode_coordinates"
    )
    if relocated != (basis_binding.source_kind == "relocated_layer_fragment"):
        raise ValueError(
            "relocated generator and computational-mode bindings disagree"
        )
    if relocated and binding.source_generator_plan_sha256 is None:
        raise ValueError(
            "relocated coordinate generators must bind their source plan"
        )
    if basis_binding.mode_set_id != fragment.fragment_id:
        raise ValueError(
            "computational-mode mode_set_id does not match fragment_id"
        )
    checks = (
        (
            binding.source_model_sha256,
            basis_binding.source_model_sha256,
            "generator and basis source model",
        ),
        (
            basis_binding.source_model_sha256,
            fragments.source_model_sha256,
            "basis and fragment-plan source model",
        ),
        (
            basis_binding.parameter_catalog_sha256,
            fragments.parameter_catalog_sha256,
            "basis and fragment-plan parameter catalog",
        ),
        (
            binding.fisher_coupling_sha256,
            basis_binding.fisher_coupling_sha256,
            "generator and basis Fisher coupling",
        ),
        (
            basis_binding.fisher_coupling_sha256,
            fragments.source_fisher_coupling_sha256,
            "basis and fragment-plan Fisher coupling",
        ),
        (
            binding.parameter_cluster_fragment_sha256,
            fragment.artifact_sha256,
            "generator and basis parameter-cluster fragment",
        ),
        (
            basis_binding.parameter_cluster_sha256,
            fragment.artifact_sha256,
            "basis and selected parameter-cluster fragment",
        ),
        (
            binding.cluster_plan_sha256,
            fragments.artifact_sha256,
            "generator and shared fragment plan",
        ),
        (
            binding.fit_split_sha256,
            basis_binding.fit_split_sha256,
            "generator and basis fit split",
        ),
        (
            binding.eval_split_sha256,
            basis_binding.eval_split_sha256,
            "generator and basis evaluation split",
        ),
        (
            binding.computational_mode_basis_sha256,
            basis.artifact_sha256,
            "generator and computational-mode basis",
        ),
        (
            binding.output_catalog_sha256,
            basis.artifact_sha256,
            "coordinate output catalog and computational-mode basis",
        ),
        (
            binding.output_site,
            basis_binding.output_site,
            "generator and basis output site",
        ),
        (
            binding.input_site,
            fragment.input_site,
            "generator and fragment input site",
        ),
        (
            binding.input_catalog_sha256,
            fragment.input_catalog_sha256,
            "generator and fragment input catalog",
        ),
            (
                plan.input_width,
                fragment.input_width,
                "generator and fragment input width",
            ),
            (
                basis.residual_width,
                fragment.output_width,
                "basis and fragment output width",
            ),
        )
    for first, second, label in checks:
        if first != second:
            raise ValueError(f"{label} mismatch")
    if relocated:
        if binding.output_site == fragment.output_site:
            raise ValueError(
                "relocated generator output site must differ from its source "
                "fragment"
            )
    elif binding.output_site != fragment.output_site:
        raise ValueError("generator and fragment output site mismatch")
    if plan.output_width != basis.rank:
        raise ValueError(
            "coordinate generator output width must equal mode-basis rank"
        )
    return fragment


def _graph_weights(
    plan: ModalGeneratorPlan,
    basis: ComputationalModeBasis,
) -> LinearModalGeneratorNodeWeights:
    factors = plan.factors
    # Preserve the generator's operation order while exposing only its final
    # K-dimensional output as graph state.  Interactions cannot accidentally
    # act on the private reduced-regression rank.
    mean_bias = basis.mean_bias.clone()
    return LinearModalGeneratorNodeWeights(
        generator_artifact_sha256=plan.artifact_sha256,
        source_model_sha256=plan.binding.source_model_sha256,
        parameter_cluster_plan_sha256=plan.binding.cluster_plan_sha256,
        state_kind="computational_mode_coordinates",
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=(
            plan.binding.parameter_cluster_fragment_sha256
        ),
        input_factor=factors.input_factor,
        state_factor=factors.output_factor,
        latent_bias=(
            None if factors.bias is None else factors.bias.clone()
        ),
        output_factor=basis.decoder_basis,
        output_bias=(
            mean_bias if bool(torch.count_nonzero(mean_bias)) else None
        ),
    )


def _fused_residual_plan(
    coordinate_plan: ModalGeneratorPlan,
    basis: ComputationalModeBasis,
) -> ModalGeneratorPlan:
    source = coordinate_plan
    factors = source.factors
    decoder = basis.decoder_basis
    fused_output = (factors.output_factor @ decoder).contiguous()
    fused_bias = basis.mean_bias.clone()
    if factors.bias is not None:
        fused_bias = fused_bias + factors.bias @ decoder
    has_bias = bool(torch.count_nonzero(fused_bias))
    bias = fused_bias.contiguous() if has_bias else None
    binding = ModalGeneratorBinding.create(
        generator_id=f"{source.binding.generator_id}.dense-fused",
        input_kind=source.binding.input_kind,
        input_site=source.binding.input_site,
        output_site=source.binding.output_site,
        source_model_sha256=source.binding.source_model_sha256,
        input_catalog_sha256=source.binding.input_catalog_sha256,
        # The target is now the native residual space.  The natural parameter
        # catalog is the stable catalog for that physical output boundary.
        output_catalog_sha256=basis.binding.parameter_catalog_sha256,
        cluster_plan_sha256=source.binding.cluster_plan_sha256,
        fit_split_sha256=source.binding.fit_split_sha256,
        eval_split_sha256=source.binding.eval_split_sha256,
        target_kind="cluster_residual_contribution",
        fisher_coupling_sha256=source.binding.fisher_coupling_sha256,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=(
            source.binding.parameter_cluster_fragment_sha256
        ),
        source_generator_plan_sha256=source.artifact_sha256,
    )
    config = ModalGeneratorConfig(
        ranks=(source.rank,),
        fit_intercept=has_bias,
        ridge=source.config.ridge,
        bias_mac_policy=source.config.bias_mac_policy,
        selection_rule="fixed_rank",
        selected_rank=source.rank,
    )
    fused_factors = ModalGeneratorFactors(
        rank=source.rank,
        input_factor=factors.input_factor,
        output_factor=fused_output,
        bias=bias,
    )
    matrix_count = (
        fused_factors.input_width * fused_factors.rank
        + fused_factors.rank * fused_factors.output_width
    )
    parameter_count = matrix_count + (
        fused_factors.output_width if has_bias else 0
    )
    macs_per_token = matrix_count
    if (
        has_bias
        and config.bias_mac_policy == "count_bias_additions"
    ):
        macs_per_token += fused_factors.output_width
    return ModalGeneratorPlan(
        binding=binding,
        config=config,
        factors=fused_factors,
        parameter_count=parameter_count,
        macs_per_token=macs_per_token,
    )


@dataclass(frozen=True, slots=True)
class ModalGeneratorLowering:
    """Authenticated coordinate, graph, and fused-residual lowering chain."""

    coordinate_generator_plan: ModalGeneratorPlan
    computational_mode_basis: ComputationalModeBasis
    fragment_plan: ParameterClusterLayerFragmentPlan
    graph_weights: LinearModalGeneratorNodeWeights
    fused_residual_plan: ModalGeneratorPlan
    selected_fragment_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(
            self.coordinate_generator_plan,
            ModalGeneratorPlan,
        ):
            raise TypeError(
                "coordinate_generator_plan must be ModalGeneratorPlan"
            )
        if not isinstance(
            self.computational_mode_basis,
            ComputationalModeBasis,
        ):
            raise TypeError(
                "computational_mode_basis must be ComputationalModeBasis"
            )
        if not isinstance(
            self.fragment_plan,
            ParameterClusterLayerFragmentPlan,
        ):
            raise TypeError(
                "fragment_plan must be ParameterClusterLayerFragmentPlan"
            )
        if not isinstance(
            self.graph_weights,
            LinearModalGeneratorNodeWeights,
        ):
            raise TypeError(
                "graph_weights must be LinearModalGeneratorNodeWeights"
            )
        if not isinstance(self.fused_residual_plan, ModalGeneratorPlan):
            raise TypeError(
                "fused_residual_plan must be ModalGeneratorPlan"
            )
        fragment = validate_coordinate_generator_compatibility(
            self.coordinate_generator_plan,
            self.computational_mode_basis,
            self.fragment_plan,
        )
        if (
            _require_sha256(
                self.selected_fragment_sha256,
                label="selected_fragment_sha256",
            )
            != fragment.artifact_sha256
        ):
            raise ValueError(
                "selected_fragment_sha256 does not match the bound fragment"
            )
        expected_graph = _graph_weights(
            self.coordinate_generator_plan,
            self.computational_mode_basis,
        )
        if (
            self.graph_weights.artifact_sha256
            != expected_graph.artifact_sha256
        ):
            raise ValueError(
                "graph weights are not the authenticated coordinate lowering"
            )
        expected_fused = _fused_residual_plan(
            self.coordinate_generator_plan,
            self.computational_mode_basis,
        )
        if (
            self.fused_residual_plan.artifact_sha256
            != expected_fused.artifact_sha256
        ):
            raise ValueError(
                "fused residual plan is not the authenticated basis lowering"
            )
        if (
            self.artifact_kind != _KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal-generator lowering header is invalid")
        computed = _json_sha256(self._payload(), domain=_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("modal-generator lowering hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def generator_id(self) -> str:
        return self.coordinate_generator_plan.binding.generator_id

    @property
    def mode_set_id(self) -> str:
        return self.computational_mode_basis.binding.mode_set_id

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "generator_id": self.generator_id,
            "mode_set_id": self.mode_set_id,
            "coordinate_generator_plan_sha256": (
                self.coordinate_generator_plan.artifact_sha256
            ),
            "computational_mode_basis_sha256": (
                self.computational_mode_basis.artifact_sha256
            ),
            "fragment_plan_sha256": self.fragment_plan.artifact_sha256,
            "selected_fragment_sha256": self.selected_fragment_sha256,
            "graph_weights_sha256": self.graph_weights.artifact_sha256,
            "fused_residual_plan_sha256": (
                self.fused_residual_plan.artifact_sha256
            ),
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "coordinate_generator_plan": (
                self.coordinate_generator_plan.state_dict()
            ),
            "computational_mode_basis": (
                self.computational_mode_basis.state_dict()
            ),
            "fragment_plan": self.fragment_plan.state_dict(),
            "graph_weights": self.graph_weights.state_dict(),
            "fused_residual_plan": self.fused_residual_plan.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalGeneratorLowering:
        fields = {
            "artifact_kind",
            "format_version",
            "generator_id",
            "mode_set_id",
            "coordinate_generator_plan_sha256",
            "computational_mode_basis_sha256",
            "fragment_plan_sha256",
            "selected_fragment_sha256",
            "graph_weights_sha256",
            "fused_residual_plan_sha256",
            "artifact_sha256",
            "coordinate_generator_plan",
            "computational_mode_basis",
            "fragment_plan",
            "graph_weights",
            "fused_residual_plan",
        }
        _strict_fields(state, expected=fields, label="modal-generator lowering")
        plan = ModalGeneratorPlan.from_state_dict(
            state["coordinate_generator_plan"]  # type: ignore[arg-type]
        )
        basis = ComputationalModeBasis.from_state_dict(
            state["computational_mode_basis"]  # type: ignore[arg-type]
        )
        fragments = ParameterClusterLayerFragmentPlan.from_state_dict(
            state["fragment_plan"]  # type: ignore[arg-type]
        )
        graph_weights = LinearModalGeneratorNodeWeights.from_state_dict(
            state["graph_weights"]  # type: ignore[arg-type]
        )
        fused = ModalGeneratorPlan.from_state_dict(
            state["fused_residual_plan"]  # type: ignore[arg-type]
        )
        nested = (
            (
                "coordinate_generator_plan_sha256",
                plan.artifact_sha256,
            ),
            (
                "computational_mode_basis_sha256",
                basis.artifact_sha256,
            ),
            ("fragment_plan_sha256", fragments.artifact_sha256),
            ("graph_weights_sha256", graph_weights.artifact_sha256),
            ("fused_residual_plan_sha256", fused.artifact_sha256),
        )
        for name, actual in nested:
            if state[name] != actual:
                raise ValueError(f"{name} does not match nested artifact")
        result = cls(
            coordinate_generator_plan=plan,
            computational_mode_basis=basis,
            fragment_plan=fragments,
            graph_weights=graph_weights,
            fused_residual_plan=fused,
            selected_fragment_sha256=state[
                "selected_fragment_sha256"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        if (
            state["generator_id"] != result.generator_id
            or state["mode_set_id"] != result.mode_set_id
        ):
            raise ValueError(
                "serialized lowering identities do not match nested artifacts"
            )
        return result

    def to_graph_node(
        self,
        *,
        name: str,
        causal_order: int,
        input_boundary: str | None = None,
        output_boundary: str | None = None,
    ) -> ModalGeneratorNode:
        """Create a graph node while keeping node naming caller-controlled."""

        return ModalGeneratorNode(
            name=name,
            causal_order=causal_order,
            input_boundary=(
                self.coordinate_generator_plan.binding.input_site
                if input_boundary is None
                else input_boundary
            ),
            output_boundary=(
                self.coordinate_generator_plan.binding.output_site
                if output_boundary is None
                else output_boundary
            ),
            weights=self.graph_weights,
        )


def lower_coordinate_modal_generator(
    generator_plan: ModalGeneratorPlan,
    computational_mode_basis: ComputationalModeBasis,
    fragment_plan: ParameterClusterLayerFragmentPlan,
) -> ModalGeneratorLowering:
    """Lower one coordinate generator to graph and dense residual forms."""

    plan = _authenticated_plan(generator_plan)
    basis = _authenticated_basis(computational_mode_basis)
    fragments = _authenticated_fragment_plan(fragment_plan)
    fragment = validate_coordinate_generator_compatibility(
        plan,
        basis,
        fragments,
    )
    return ModalGeneratorLowering(
        coordinate_generator_plan=plan,
        computational_mode_basis=basis,
        fragment_plan=fragments,
        graph_weights=_graph_weights(plan, basis),
        fused_residual_plan=_fused_residual_plan(plan, basis),
        selected_fragment_sha256=fragment.artifact_sha256,
    )
