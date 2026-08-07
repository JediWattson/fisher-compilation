"""Fixed-capacity A3 correction fitting for a compiled Gemma trajectory.

The correction keeps the already-frozen Layer-17 modal decoder bases and
projects one joint raw-MLP compensation target into their combined affine
span.  Only the input-to-coordinate generator coefficients are refitted.
Consequently node ranks, graph parameter count, and graph MAC count remain
identical to the source edgeless graph.

All row tensors are ephemeral.  Metadata emitted by this module contains only
hashes, shapes, scalar diagnostics, and artifact identities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import Tensor

from .computational_modes import (
    ComputationalModeBasis,
    ComputationalModeBinding,
)
from .gemma3_modal_generator_dev_experiment import LayerFragmentRows
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import (
    ModalGeneratorLowering,
    lower_coordinate_modal_generator,
)
from .modal_generators import (
    ModalGeneratorBinding,
    ModalGeneratorRateCurve,
    fit_modal_generator_rate_curve,
)


__all__ = [
    "FrozenBasisGeneratorFit",
    "JointFrozenBasisProjection",
    "build_a3_raw_mlp_target",
    "build_projected_correction_rows",
    "fit_frozen_basis_coordinate_generators",
    "project_joint_target_to_frozen_bases",
    "replace_layer_nodes_in_composed_graph",
]


_TENSOR_DOMAIN = b"fisher-graph:gemma3-l10-l17-a3-row-tensor:v1\0"
_COMPUTATIONAL_MODE_TENSOR_DOMAIN = (
    b"fisher_graph.computational_modes.tensor.v1\0"
)
_PROJECTION_METHOD = (
    "float64_affine_sum_svd_pseudoinverse_minimum_norm"
)


def _matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(f"{label} must be a nonempty floating matrix")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(f"{tuple(canonical.shape)}\0{canonical.dtype}\0".encode())
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _computational_mode_tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError("computational-mode tensor hash input must be finite")
    digest = hashlib.sha256()
    digest.update(_COMPUTATIONAL_MODE_TENSOR_DOMAIN)
    digest.update(f"{tuple(canonical.shape)}\0float64\0".encode("utf-8"))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _ordered_names(
    values: Mapping[str, object],
    order: Sequence[str],
) -> tuple[str, ...]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("catalog must be a nonempty mapping")
    names = tuple(order)
    if (
        not names
        or names != tuple(dict.fromkeys(names))
        or set(names) != set(values)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("catalog order must exactly cover unique names")
    return names


def build_a3_raw_mlp_target(
    native_full_mlp_output: Tensor,
    compiled_compact_retained_mlp_output: Tensor,
) -> Tensor:
    """Return native-full minus the exact compact retained MLP replay.

    The caller must supply the authenticated compact Layer-17 operator output
    evaluated on the Layer-10-compiled normalized input.  Algebraic
    ``compiled_full - selected_contributions`` reconstruction is deliberately
    not accepted here; it is a rounding audit, not the v2 correction target.
    """

    native = _matrix(native_full_mlp_output, label="native full MLP output")
    retained = _matrix(
        compiled_compact_retained_mlp_output,
        label="compiled compact retained MLP output",
    )
    if native.shape != retained.shape:
        raise ValueError("native and compact retained MLP outputs disagree")
    target = native - retained
    if not bool(torch.isfinite(target).all()):
        raise ValueError("A3 raw-MLP target became non-finite")
    return target.contiguous()


@dataclass(frozen=True, slots=True)
class JointFrozenBasisProjection:
    """Ephemeral joint projection plus source-safe scalar/hash metadata."""

    node_order: tuple[str, ...]
    coordinates_by_node: Mapping[str, Tensor]
    contributions_by_node: Mapping[str, Tensor]
    target: Tensor
    prediction: Tensor
    basis_sha256_by_node: Mapping[str, str]
    mean_bias_sha256_by_node: Mapping[str, str]
    decoder_basis_sha256_by_node: Mapping[str, str]
    affine_offset_sha256: str
    combined_basis_rank: int

    def __post_init__(self) -> None:
        names = _ordered_names(self.coordinates_by_node, self.node_order)
        if (
            set(self.contributions_by_node) != set(names)
            or set(self.basis_sha256_by_node) != set(names)
            or set(self.mean_bias_sha256_by_node) != set(names)
            or set(self.decoder_basis_sha256_by_node) != set(names)
        ):
            raise ValueError("projection node catalogs disagree")
        target = _matrix(self.target, label="joint target")
        prediction = _matrix(self.prediction, label="joint prediction")
        if target.shape != prediction.shape:
            raise ValueError("joint projection target/prediction disagree")
        for name in names:
            coordinates = _matrix(
                self.coordinates_by_node[name],
                label=f"{name} coordinates",
            )
            contribution = _matrix(
                self.contributions_by_node[name],
                label=f"{name} contribution",
            )
            if (
                coordinates.shape[0] != target.shape[0]
                or contribution.shape != target.shape
            ):
                raise ValueError("joint projection row shape drifted")
            for digest in (
                self.basis_sha256_by_node[name],
                self.mean_bias_sha256_by_node[name],
                self.decoder_basis_sha256_by_node[name],
            ):
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError("projection basis hash is invalid")
        if (
            type(self.combined_basis_rank) is not int
            or self.combined_basis_rank <= 0
        ):
            raise ValueError("combined_basis_rank must be positive")
        if (
            not isinstance(self.affine_offset_sha256, str)
            or len(self.affine_offset_sha256) != 64
        ):
            raise ValueError("affine_offset_sha256 is invalid")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(
            self,
            "coordinates_by_node",
            {name: _matrix(self.coordinates_by_node[name], label=name) for name in names},
        )
        object.__setattr__(
            self,
            "contributions_by_node",
            {
                name: _matrix(self.contributions_by_node[name], label=name)
                for name in names
            },
        )
        object.__setattr__(
            self,
            "basis_sha256_by_node",
            dict(self.basis_sha256_by_node),
        )
        object.__setattr__(
            self,
            "mean_bias_sha256_by_node",
            dict(self.mean_bias_sha256_by_node),
        )
        object.__setattr__(
            self,
            "decoder_basis_sha256_by_node",
            dict(self.decoder_basis_sha256_by_node),
        )

    def metadata(self) -> dict[str, object]:
        error = self.prediction - self.target
        target_rms = math.sqrt(float(self.target.square().mean().item()))
        rmse = math.sqrt(float(error.square().mean().item()))
        return {
            "projection_method": _PROJECTION_METHOD,
            "node_order": self.node_order,
            "basis_sha256_by_node": dict(self.basis_sha256_by_node),
            "mean_bias_sha256_by_node": dict(
                self.mean_bias_sha256_by_node
            ),
            "decoder_basis_sha256_by_node": dict(
                self.decoder_basis_sha256_by_node
            ),
            "affine_offset_sha256": self.affine_offset_sha256,
            "combined_basis_rank": self.combined_basis_rank,
            "observation_count": self.target.shape[0],
            "residual_width": self.target.shape[1],
            "target_sha256": _tensor_sha256(self.target),
            "prediction_sha256": _tensor_sha256(self.prediction),
            "coordinate_sha256_by_node": {
                name: _tensor_sha256(self.coordinates_by_node[name])
                for name in self.node_order
            },
            "contribution_sha256_by_node": {
                name: _tensor_sha256(self.contributions_by_node[name])
                for name in self.node_order
            },
            "rmse": rmse,
            "target_rms": target_rms,
            "nrmse": rmse / max(target_rms, 1e-30),
            "max_abs_error": float(error.abs().max().item()),
            "offline_projection_only": True,
            "runtime_parameter_count": 0,
            "runtime_macs_per_token": 0,
        }


def project_joint_target_to_frozen_bases(
    target: Tensor,
    bases_by_node: Mapping[str, ComputationalModeBasis],
    *,
    node_order: Sequence[str],
) -> JointFrozenBasisProjection:
    """Project one total correction into the sum of frozen affine codecs."""

    values = _matrix(target, label="joint correction target")
    names = _ordered_names(bases_by_node, node_order)
    bases: list[ComputationalModeBasis] = []
    for name in names:
        basis = bases_by_node[name]
        if not isinstance(basis, ComputationalModeBasis):
            raise TypeError("bases_by_node must contain computational modes")
        basis.validate_integrity()
        if basis.residual_width != values.shape[1]:
            raise ValueError("frozen basis residual width drifted")
        bases.append(basis)

    mean_sum = torch.stack(tuple(basis.mean for basis in bases)).sum(dim=0)
    combined_encoder = torch.cat(
        tuple(basis.encoder_basis for basis in bases),
        dim=1,
    )
    centered = values - mean_sum
    # ``pinv`` is deterministic for these frozen CPU float64 matrices and
    # returns the minimum-norm coordinates when node spans overlap.
    coordinates = centered @ torch.linalg.pinv(combined_encoder).T
    coordinates_by_node: dict[str, Tensor] = {}
    contributions_by_node: dict[str, Tensor] = {}
    start = 0
    for name, basis in zip(names, bases, strict=True):
        stop = start + basis.rank
        current = coordinates[:, start:stop].contiguous()
        coordinates_by_node[name] = current
        contributions_by_node[name] = basis.decode(current).contiguous()
        start = stop
    prediction = torch.stack(
        tuple(contributions_by_node[name] for name in names),
        dim=0,
    ).sum(dim=0)
    return JointFrozenBasisProjection(
        node_order=names,
        coordinates_by_node=coordinates_by_node,
        contributions_by_node=contributions_by_node,
        target=values,
        prediction=prediction,
        basis_sha256_by_node={
            name: basis.artifact_sha256
            for name, basis in zip(names, bases, strict=True)
        },
        mean_bias_sha256_by_node={
            name: basis.mean_bias_sha256
            for name, basis in zip(names, bases, strict=True)
        },
        decoder_basis_sha256_by_node={
            name: basis.decoder_basis_sha256
            for name, basis in zip(names, bases, strict=True)
        },
        affine_offset_sha256=_computational_mode_tensor_sha256(mean_sum),
        combined_basis_rank=int(torch.linalg.matrix_rank(combined_encoder).item()),
    )


def build_projected_correction_rows(
    *,
    inputs: Tensor,
    target: Tensor,
    fisher_weights_by_node: Mapping[str, Tensor],
    fragment_id_by_node: Mapping[str, str],
    bases_by_node: Mapping[str, ComputationalModeBasis],
    node_order: Sequence[str],
    row_keys: tuple[tuple[str, int], ...],
    sequences: int,
) -> tuple[AlignedFragmentRows, JointFrozenBasisProjection]:
    """Build per-fragment fit rows from one jointly projected A3 target."""

    names = _ordered_names(bases_by_node, node_order)
    if set(fisher_weights_by_node) != set(names) or set(
        fragment_id_by_node
    ) != set(names):
        raise ValueError("correction row node catalogs disagree")
    features = _matrix(inputs, label="compiled Layer17 inputs")
    projection = project_joint_target_to_frozen_bases(
        target,
        bases_by_node,
        node_order=names,
    )
    if features.shape[0] != projection.target.shape[0]:
        raise ValueError("correction inputs and targets disagree")
    fragment_ids = tuple(fragment_id_by_node[name] for name in names)
    if len(fragment_ids) != len(set(fragment_ids)) or any(
        not isinstance(value, str) or not value for value in fragment_ids
    ):
        raise ValueError("fragment_id_by_node must be one-to-one")
    rows = {
        fragment_id_by_node[name]: LayerFragmentRows(
            inputs=features,
            contributions=projection.contributions_by_node[name],
            fisher_weights=fisher_weights_by_node[name],
            sequences=sequences,
        )
        for name in names
    }
    return AlignedFragmentRows(rows_by_fragment=rows, row_keys=row_keys), projection


@dataclass(frozen=True, slots=True)
class FrozenBasisGeneratorFit:
    """New coordinate generators over byte-identical frozen decoder bases."""

    graph_plan: ModalGeneratorGraphPlan
    lowerings_by_node: Mapping[str, ModalGeneratorLowering]
    rate_curves_by_node: Mapping[str, ModalGeneratorRateCurve]
    source_mean_bias_sha256_by_node: Mapping[str, str]
    source_decoder_basis_sha256_by_node: Mapping[str, str]

    def __post_init__(self) -> None:
        self.graph_plan.validate_integrity()
        names = self.graph_plan.traversal_order
        if (
            self.graph_plan.interactions
            or set(self.lowerings_by_node) != set(names)
            or set(self.rate_curves_by_node) != set(names)
            or set(self.source_mean_bias_sha256_by_node) != set(names)
            or set(self.source_decoder_basis_sha256_by_node) != set(names)
        ):
            raise ValueError("frozen-basis generator fit catalogs disagree")
        for name in names:
            lowering = self.lowerings_by_node[name]
            if (
                lowering.computational_mode_basis.mean_bias_sha256
                != self.source_mean_bias_sha256_by_node[name]
                or lowering.computational_mode_basis.decoder_basis_sha256
                != self.source_decoder_basis_sha256_by_node[name]
            ):
                raise ValueError("correction fit changed a frozen affine basis")

    def metadata(self) -> dict[str, object]:
        return {
            "graph_sha256": self.graph_plan.artifact_sha256,
            "node_order": self.graph_plan.traversal_order,
            "parameter_count": self.graph_plan.parameter_count,
            "macs_per_token": self.graph_plan.macs_per_token,
            "interaction_count": 0,
            "source_mean_bias_sha256_by_node": dict(
                self.source_mean_bias_sha256_by_node
            ),
            "source_decoder_basis_sha256_by_node": dict(
                self.source_decoder_basis_sha256_by_node
            ),
            "lowering_sha256_by_node": {
                name: self.lowerings_by_node[name].artifact_sha256
                for name in self.graph_plan.traversal_order
            },
            "generator_plan_sha256_by_node": {
                name: self.lowerings_by_node[
                    name
                ].coordinate_generator_plan.artifact_sha256
                for name in self.graph_plan.traversal_order
            },
        }


def fit_frozen_basis_coordinate_generators(
    fit_rows: AlignedFragmentRows,
    eval_rows: AlignedFragmentRows,
    *,
    source_graph: ModalGeneratorGraphPlan,
    source_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    fit_split_sha256: str,
    eval_split_sha256: str,
    generator_rank: int = 16,
    ridge: float = 0.0,
    output_boundary: str | None = None,
) -> FrozenBasisGeneratorFit:
    """Refit coordinate generators while preserving all decoder tensors.

    ``output_boundary`` is an explicit application-boundary relocation.  The
    default keeps the source graph's operator-output bindings byte-for-byte
    compatible.  A distinct boundary produces new, authenticated generator
    and computational-mode bindings while retaining the exact source mean,
    encoder, and decoder tensors.
    """

    source_graph.validate_integrity()
    if source_graph.interactions:
        raise ValueError("source Layer17 correction graph must be edgeless")
    names = source_graph.traversal_order
    if set(source_lowerings_by_node) != set(names):
        raise ValueError("source graph/lowering catalogs disagree")
    if type(generator_rank) is not int or generator_rank <= 0:
        raise ValueError("generator_rank must be positive")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    if set(fit_rows.row_keys) & set(eval_rows.row_keys):
        raise ValueError("correction fit and evaluation row keys overlap")

    new_lowerings: dict[str, ModalGeneratorLowering] = {}
    curves: dict[str, ModalGeneratorRateCurve] = {}
    nodes = []
    source_mean_hashes: dict[str, str] = {}
    source_decoder_hashes: dict[str, str] = {}
    expected_fragments: list[str] = []
    for source_node in source_graph.nodes:
        name = source_node.name
        source = source_lowerings_by_node[name]
        source_basis = source.computational_mode_basis
        source_basis.validate_integrity()
        fragment = next(
            (
                value
                for value in source.fragment_plan.fragments
                if value.artifact_sha256 == source.selected_fragment_sha256
            ),
            None,
        )
        if fragment is None:
            raise ValueError("source lowering fragment binding is unavailable")
        fragment_id = fragment.fragment_id
        expected_fragments.append(fragment_id)
        try:
            fit_fragment = fit_rows.rows_by_fragment[fragment_id]
            eval_fragment = eval_rows.rows_by_fragment[fragment_id]
        except KeyError as error:
            raise ValueError("correction rows do not cover source fragments") from error
        if (
            fit_fragment.inputs.shape[1] != source_node.input_width
            or eval_fragment.inputs.shape[1] != source_node.input_width
            or fit_fragment.contributions.shape[1] != source_node.output_width
            or eval_fragment.contributions.shape[1] != source_node.output_width
            or generator_rank > min(source_node.input_width, source_basis.rank)
        ):
            raise ValueError("correction row width or generator rank drifted")
        source_binding = source.coordinate_generator_plan.binding
        source_mode_binding = source_basis.binding
        runtime_output_site = (
            source_binding.output_site
            if output_boundary is None
            else output_boundary
        )
        relocated_output = runtime_output_site != source_binding.output_site
        rebound_mode_binding = ComputationalModeBinding.create(
            mode_set_id=source_mode_binding.mode_set_id,
            source_kind=(
                "relocated_layer_fragment"
                if relocated_output
                else source_mode_binding.source_kind
            ),
            output_site=runtime_output_site,
            source_model_sha256=source_mode_binding.source_model_sha256,
            parameter_catalog_sha256=(
                source_mode_binding.parameter_catalog_sha256
            ),
            fisher_coupling_sha256=(
                source_mode_binding.fisher_coupling_sha256
            ),
            parameter_cluster_sha256=(
                source_mode_binding.parameter_cluster_sha256
            ),
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=eval_split_sha256,
        )
        basis = ComputationalModeBasis(
            binding=rebound_mode_binding,
            config=source_basis.config,
            rank=source_basis.rank,
            mean_bias=source_basis.mean_bias,
            encoder_basis=source_basis.encoder_basis,
        )
        if (
            basis.mean_bias_sha256 != source_basis.mean_bias_sha256
            or basis.encoder_basis_sha256
            != source_basis.encoder_basis_sha256
            or basis.decoder_basis_sha256
            != source_basis.decoder_basis_sha256
        ):
            raise RuntimeError("trajectory correction changed frozen basis tensors")
        binding = ModalGeneratorBinding.create(
            generator_id=source_binding.generator_id,
            input_kind=source_binding.input_kind,
            input_site=source_binding.input_site,
            output_site=runtime_output_site,
            source_model_sha256=source_binding.source_model_sha256,
            input_catalog_sha256=source_binding.input_catalog_sha256,
            output_catalog_sha256=basis.artifact_sha256,
            cluster_plan_sha256=source.fragment_plan.artifact_sha256,
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=eval_split_sha256,
            target_kind=(
                "relocated_computational_mode_coordinates"
                if relocated_output
                else "computational_mode_coordinates"
            ),
            fisher_coupling_sha256=source_binding.fisher_coupling_sha256,
            computational_mode_basis_sha256=basis.artifact_sha256,
            parameter_cluster_fragment_sha256=fragment.artifact_sha256,
            source_generator_plan_sha256=(
                source.coordinate_generator_plan.artifact_sha256
            ),
        )
        curve = fit_modal_generator_rate_curve(
            fit_fragment.inputs,
            basis.encode(fit_fragment.contributions),
            fit_fragment.fisher_weights,
            eval_fragment.inputs,
            basis.encode(eval_fragment.contributions),
            (generator_rank,),
            binding=binding,
            fisher_weights_eval=eval_fragment.fisher_weights,
            fit_intercept=True,
            ridge=ridge,
            bias_mac_policy="matrix_multiplies_only",
            selection_rule="fixed_rank",
            selected_rank=generator_rank,
        )
        plan = curve.selected_plan
        if plan is None:
            raise RuntimeError("fixed correction generator rank was not selected")
        lowering = lower_coordinate_modal_generator(
            plan,
            basis,
            source.fragment_plan,
        )
        if (
            lowering.computational_mode_basis.decoder_basis_sha256
            != source_basis.decoder_basis_sha256
        ):
            raise RuntimeError("correction lowering changed the frozen decoder")
        new_lowerings[name] = lowering
        curves[name] = curve
        source_mean_hashes[name] = source_basis.mean_bias_sha256
        source_decoder_hashes[name] = source_basis.decoder_basis_sha256
        nodes.append(
            lowering.to_graph_node(
                name=name,
                causal_order=source_node.causal_order,
                input_boundary=source_node.input_boundary,
                output_boundary=runtime_output_site,
            )
        )

    if (
        set(fit_rows.rows_by_fragment) != set(expected_fragments)
        or set(eval_rows.rows_by_fragment) != set(expected_fragments)
    ):
        raise ValueError("correction rows contain a non-source fragment")

    graph = ModalGeneratorGraphPlan(
        model_fingerprint=source_graph.model_fingerprint,
        parameter_cluster_plan_sha256=(
            source_graph.parameter_cluster_plan_sha256
        ),
        nodes=tuple(nodes),
        interactions=(),
    )
    if (
        graph.parameter_count != source_graph.parameter_count
        or graph.macs_per_token != source_graph.macs_per_token
        or tuple(node.latent_width for node in graph.nodes)
        != tuple(node.latent_width for node in source_graph.nodes)
    ):
        raise RuntimeError("trajectory correction changed fixed graph resources")
    return FrozenBasisGeneratorFit(
        graph_plan=graph,
        lowerings_by_node=new_lowerings,
        rate_curves_by_node=curves,
        source_mean_bias_sha256_by_node=source_mean_hashes,
        source_decoder_basis_sha256_by_node=source_decoder_hashes,
    )


def replace_layer_nodes_in_composed_graph(
    composed_graph: ModalGeneratorGraphPlan,
    replacement_graph: ModalGeneratorGraphPlan,
    *,
    layer_ordinal: int,
) -> ModalGeneratorGraphPlan:
    """Replace one edgeless layer's nodes while retaining all other edges."""

    composed_graph.validate_integrity()
    replacement_graph.validate_integrity()
    if type(layer_ordinal) is not int or layer_ordinal < 0:
        raise ValueError("layer_ordinal must be nonnegative")
    if (
        replacement_graph.interactions
        or replacement_graph.model_fingerprint
        != composed_graph.model_fingerprint
        or replacement_graph.parameter_cluster_plan_sha256
        != composed_graph.parameter_cluster_plan_sha256
    ):
        raise ValueError("replacement graph identity or edge policy drifted")
    boundary = f"layer.{layer_ordinal}."
    replaced = tuple(
        node
        for node in composed_graph.nodes
        if node.input_boundary.startswith(boundary)
        and node.output_boundary.startswith(boundary)
    )
    replacements = tuple(replacement_graph.nodes)
    replaced_names = {node.name for node in replaced}
    if (
        not replaced
        or {node.name for node in replacements} != replaced_names
        or any(
            not node.input_boundary.startswith(boundary)
            or not node.output_boundary.startswith(boundary)
            for node in replacements
        )
        or any(
            edge.source_node in replaced_names
            or edge.target_node in replaced_names
            for edge in composed_graph.interactions
        )
    ):
        raise ValueError(
            "composed layer replacement requires an edgeless target layer"
        )
    replacement_by_name = {node.name: node for node in replacements}
    nodes = tuple(
        replacement_by_name.get(node.name, node) for node in composed_graph.nodes
    )
    result = ModalGeneratorGraphPlan(
        model_fingerprint=composed_graph.model_fingerprint,
        parameter_cluster_plan_sha256=(
            composed_graph.parameter_cluster_plan_sha256
        ),
        nodes=nodes,
        interactions=composed_graph.interactions,
    )
    expected_parameters = (
        composed_graph.parameter_count
        - sum(node.weights.parameter_count for node in replaced)
        + replacement_graph.parameter_count
    )
    expected_macs = (
        composed_graph.macs_per_token
        - sum(node.weights.macs_per_token for node in replaced)
        + replacement_graph.macs_per_token
    )
    if (
        result.parameter_count != expected_parameters
        or result.macs_per_token != expected_macs
    ):
        raise RuntimeError("composed layer replacement accounting drifted")
    return result
