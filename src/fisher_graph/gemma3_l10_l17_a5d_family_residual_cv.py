"""Family-disjoint ridge/alpha selection for source-anchored A5d residuals.

The A5d target builder expresses the desired change *relative to* the exact
frozen compiled state.  This module fits rank-16 generators for those
zero-mean residual rows and jointly selects a ridge and a continuous residual
scale without ever replacing the source state:

``candidate = frozen_source + alpha * predicted_residual``.

Each ridge is fit once per inner family fold.  The resulting prediction is
then scored for every alpha.  The alpha-zero branch deliberately performs no
residual arithmetic and is therefore the exact frozen-source control.  A
positive-alpha candidate must strictly improve that control or the selection
falls back to alpha zero without a final refit.

Only scalar diagnostics, booleans, counts, and hashes enter the receipt.  Row,
state, logit, factor, and basis tensors remain ephemeral.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType

import torch
from torch import Tensor

from .computational_modes import ComputationalModeBasis, ComputationalModeBinding
from .gemma3_l10_l17_a5c_family_ridge_cv import (
    A5cAuthenticatedRowBank,
    _authenticate_lowering_catalog,
    _bridge_compiled_inputs_sha256,
    _chunked_final_head_kl,
    _descriptive_rekey_rows,
    _input_row_signatures,
    _joint_fisher_weights,
    _membership_sha256,
    _rows_membership_sha256,
    _runtime_input_cast_audit,
    _subset_equal_family_rows,
    _weighted_state_nrmse,
)
from .gemma3_l10_l17_a5d_source_anchored_residual import (
    A5dSourceAnchoredResidualTargets,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    FrozenBasisGeneratorFit,
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
    apply_modal_generator,
    fit_modal_generator_rate_curve,
)


__all__ = [
    "A5D_ALPHA_GRID",
    "A5D_FIXED_GENERATOR_RANK",
    "A5D_RIDGE_GRID",
    "GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_SCHEMA",
    "A5dFamilyResidualCvSelection",
    "fit_zero_mean_residual_generators",
    "select_a5d_family_disjoint_residual",
    "validate_a5d_family_residual_cv_receipt",
]


GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5d_family_residual_cv"
)
GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_FORMAT_VERSION = 1
A5D_FIXED_GENERATOR_RANK = 16
A5D_RIDGE_GRID = (0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0)
A5D_ALPHA_GRID = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)

_INNER_FOLD_COUNT = 7
_INNER_TRAINING_FAMILY_COUNT = 6
_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5d-family-residual-cv:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5d-family-residual-tensor:v1\0"
_SPLIT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5d-family-residual-split:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_DTYPE = torch.float32
_SELECTION_OBJECTIVE = (
    "seven_family_equal_mean_fisher_weighted_exact_float64_full_vocabulary_"
    "native_to_source_anchored_candidate_final_head_kl"
)
_STATE_DIAGNOSTIC = (
    "seven_family_equal_mean_fisher_weighted_block_state_nrmse_to_native"
)
_TIE_BREAK = (
    "minimum_final_head_kl_then_smaller_alpha_then_stronger_larger_ridge"
)
_FALLBACK_RULE = (
    "use_bit_exact_frozen_source_alpha_zero_unless_best_positive_alpha_"
    "strictly_reduces_the_selection_objective"
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


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout is not torch.strided:
        raise TypeError("tensor hash input must be a strided Tensor")
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


def _numeric_grid(
    values: Sequence[float],
    *,
    label: str,
    bounded: bool,
) -> tuple[float, ...]:
    result = tuple(values)
    converted = tuple(float(value) for value in result)
    if (
        not result
        or len(result) != len(set(converted))
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (bounded and float(value) > 1.0)
            for value in result
        )
        or converted != tuple(sorted(converted))
    ):
        qualifier = " in [0, 1]" if bounded else " and nonnegative"
        raise ValueError(f"{label} must be unique, sorted, finite{qualifier}")
    return converted


def _split_sha256(
    *,
    target_sha256: str,
    ridge: float,
    held_alias: str,
    role: str,
    rows: AlignedFragmentRows,
) -> str:
    return _domain_sha256(
        _SPLIT_DOMAIN,
        {
            "residual_target_receipt_sha256": target_sha256,
            "ridge_hex": ridge.hex(),
            "held_inner_family_alias": held_alias,
            "role": role,
            "row_key_sha256": rows.row_key_sha256,
            "observations": rows.observations,
            "sequences": rows.sequences,
        },
    )


def _subset_exact_keys(
    rows: AlignedFragmentRows,
    keys: Sequence[tuple[str, int]],
) -> AlignedFragmentRows:
    requested = tuple(keys)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("descriptive residual keys must be unique and nonempty")
    index_by_key = {key: index for index, key in enumerate(rows.row_keys)}
    if any(key not in index_by_key for key in requested):
        raise ValueError("descriptive residual keys are outside the target bank")
    indices = torch.tensor(
        [index_by_key[key] for key in requested], dtype=torch.long
    )
    examples = {example_id for example_id, _ in requested}
    return AlignedFragmentRows(
        rows_by_fragment={
            fragment_id: LayerFragmentRows(
                inputs=fragment.inputs.index_select(0, indices),
                contributions=fragment.contributions.index_select(0, indices),
                fisher_weights=fragment.fisher_weights.index_select(0, indices),
                sequences=len(examples),
            )
            for fragment_id, fragment in rows.rows_by_fragment.items()
        },
        row_keys=requested,
    )


def _candidate_states(
    frozen_states: Tensor,
    predicted_residual: Tensor | None,
    alpha: float,
) -> Tensor:
    """Return the exact source at alpha zero without residual arithmetic."""

    if alpha == 0.0:
        return frozen_states.clone()
    if predicted_residual is None or predicted_residual.shape != frozen_states.shape:
        raise ValueError("positive-alpha residual prediction is not aligned")
    return (frozen_states + alpha * predicted_residual).contiguous()


def _predict_fit_residual(
    fit: FrozenBasisGeneratorFit,
    inputs: Tensor,
    node_order: Sequence[str],
) -> Tensor:
    if set(fit.lowerings_by_node) != set(node_order):
        raise RuntimeError("residual fit lowering catalog differs from source")
    outputs = tuple(
        apply_modal_generator(inputs, fit.lowerings_by_node[name].fused_residual_plan)
        for name in node_order
    )
    if not outputs or len({tuple(value.shape) for value in outputs}) != 1:
        raise RuntimeError("residual fit node outputs have inconsistent shapes")
    result = torch.stack(outputs, dim=0).sum(dim=0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("residual fit prediction became non-finite")
    return result.contiguous()


def _zero_mean_expected_parameter_count(source_graph: object) -> int:
    """Remove only source affine output means from graph storage accounting."""

    removed = 0
    for node in source_graph.nodes:
        weights = getattr(node, "weights", None)
        output_bias = getattr(weights, "output_bias", None)
        if output_bias is not None:
            removed += int(output_bias.numel())
    return int(source_graph.parameter_count) - removed


def fit_zero_mean_residual_generators(
    fit_rows: AlignedFragmentRows,
    eval_rows: AlignedFragmentRows,
    *,
    source_graph: ModalGeneratorGraphPlan,
    source_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    fit_split_sha256: str,
    eval_split_sha256: str,
    generator_rank: int = A5D_FIXED_GENERATOR_RANK,
    ridge: float = 0.0,
    output_boundary: str | None = None,
) -> FrozenBasisGeneratorFit:
    """Fit residual generators with zero means and source-identical decoders."""

    source_graph.validate_integrity()
    if source_graph.interactions:
        raise ValueError("A5d source residual graph must be edgeless")
    names = tuple(source_graph.traversal_order)
    if set(source_lowerings_by_node) != set(names):
        raise ValueError("A5d source graph/lowering catalogs disagree")
    if type(generator_rank) is not int or generator_rank <= 0:
        raise ValueError("generator_rank must be positive")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    if set(fit_rows.row_keys) & set(eval_rows.row_keys):
        raise ValueError("residual fit and evaluation row keys overlap")
    _require_sha256(fit_split_sha256, label="A5d fit split")
    _require_sha256(eval_split_sha256, label="A5d evaluation split")

    new_lowerings: dict[str, ModalGeneratorLowering] = {}
    curves = {}
    nodes = []
    zero_mean_hashes: dict[str, str] = {}
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
            raise ValueError("A5d source fragment binding is unavailable")
        fragment_id = fragment.fragment_id
        expected_fragments.append(fragment_id)
        try:
            fit_fragment = fit_rows.rows_by_fragment[fragment_id]
            eval_fragment = eval_rows.rows_by_fragment[fragment_id]
        except KeyError as error:
            raise ValueError(
                "A5d residual rows do not cover source fragments"
            ) from error
        if (
            fit_fragment.inputs.shape[1] != source_node.input_width
            or eval_fragment.inputs.shape[1] != source_node.input_width
            or fit_fragment.contributions.shape[1] != source_node.output_width
            or eval_fragment.contributions.shape[1] != source_node.output_width
            or generator_rank > min(source_node.input_width, source_basis.rank)
        ):
            raise ValueError("A5d residual row width or rank drifted")

        source_binding = source.coordinate_generator_plan.binding
        source_mode_binding = source_basis.binding
        runtime_output_site = (
            source_binding.output_site if output_boundary is None else output_boundary
        )
        # Relocation is defined against the native fragment boundary, not
        # against the immediately preceding lowering.  The source lowering
        # may itself already be relocated to the compiled block boundary.
        relocated = runtime_output_site != fragment.output_site
        mode_binding = ComputationalModeBinding.create(
            mode_set_id=source_mode_binding.mode_set_id,
            source_kind=(
                "relocated_layer_fragment" if relocated else "layer_fragment"
            ),
            output_site=runtime_output_site,
            source_model_sha256=source_mode_binding.source_model_sha256,
            parameter_catalog_sha256=source_mode_binding.parameter_catalog_sha256,
            fisher_coupling_sha256=source_mode_binding.fisher_coupling_sha256,
            parameter_cluster_sha256=(
                source_mode_binding.parameter_cluster_sha256
            ),
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=eval_split_sha256,
        )
        basis = ComputationalModeBasis(
            binding=mode_binding,
            config=source_basis.config,
            rank=source_basis.rank,
            mean_bias=torch.zeros_like(source_basis.mean_bias),
            encoder_basis=source_basis.encoder_basis,
        )
        if (
            bool(torch.count_nonzero(basis.mean_bias))
            or basis.decoder_basis_sha256 != source_basis.decoder_basis_sha256
            or not torch.equal(basis.decoder_basis, source_basis.decoder_basis)
        ):
            raise RuntimeError("A5d residual basis changed the source decoder")

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
                if relocated
                else "computational_mode_coordinates"
            ),
            fisher_coupling_sha256=source_binding.fisher_coupling_sha256,
            computational_mode_basis_sha256=basis.artifact_sha256,
            parameter_cluster_fragment_sha256=fragment.artifact_sha256,
            source_generator_plan_sha256=(
                source.coordinate_generator_plan.artifact_sha256
                if relocated
                else None
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
            bias_mac_policy=(
                source.coordinate_generator_plan.config.bias_mac_policy
            ),
            selection_rule="fixed_rank",
            selected_rank=generator_rank,
        )
        plan = curve.selected_plan
        if plan is None:
            raise RuntimeError("A5d fixed residual rank was not selected")
        lowering = lower_coordinate_modal_generator(
            plan, basis, source.fragment_plan
        )
        if (
            bool(torch.count_nonzero(lowering.computational_mode_basis.mean_bias))
            or lowering.computational_mode_basis.decoder_basis_sha256
            != source_basis.decoder_basis_sha256
            or not torch.equal(
                lowering.computational_mode_basis.decoder_basis,
                source_basis.decoder_basis,
            )
        ):
            raise RuntimeError("A5d lowering changed zero-mean decoder semantics")
        new_lowerings[name] = lowering
        curves[name] = curve
        zero_mean_hashes[name] = basis.mean_bias_sha256
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
        raise ValueError("A5d residual rows contain a non-source fragment")
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=source_graph.model_fingerprint,
        parameter_cluster_plan_sha256=source_graph.parameter_cluster_plan_sha256,
        nodes=tuple(nodes),
        interactions=(),
    )
    expected_parameters = _zero_mean_expected_parameter_count(source_graph)
    if (
        graph.parameter_count != expected_parameters
        or graph.macs_per_token != source_graph.macs_per_token
        or tuple(node.latent_width for node in graph.nodes)
        != tuple(node.latent_width for node in source_graph.nodes)
    ):
        raise RuntimeError(
            "A5d residual fit changed fixed graph resources: "
            f"parameters {graph.parameter_count}/{expected_parameters}, "
            f"MACs {graph.macs_per_token}/{source_graph.macs_per_token}, "
            "latent widths "
            f"{tuple(node.latent_width for node in graph.nodes)}/"
            f"{tuple(node.latent_width for node in source_graph.nodes)}"
        )
    return FrozenBasisGeneratorFit(
        graph_plan=graph,
        lowerings_by_node=new_lowerings,
        rate_curves_by_node=curves,
        source_mean_bias_sha256_by_node=zero_mean_hashes,
        source_decoder_basis_sha256_by_node=source_decoder_hashes,
    )


def _fit_evidence(
    fit: FrozenBasisGeneratorFit,
    *,
    source_graph: ModalGeneratorGraphPlan,
    source_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    node_order: Sequence[str],
) -> dict[str, object]:
    graph = fit.graph_plan
    graph.validate_integrity()
    names = tuple(node_order)
    if (
        tuple(graph.traversal_order) != names
        or graph.interactions
        or graph.parameter_count
        != _zero_mean_expected_parameter_count(source_graph)
        or graph.macs_per_token != source_graph.macs_per_token
    ):
        raise RuntimeError("A5d fitted residual graph changed topology/resources")
    lowerings = _authenticate_lowering_catalog(
        fit.lowerings_by_node, names, label="A5d fitted residual lowering"
    )
    zero_hashes: dict[str, str] = {}
    decoder_hashes: dict[str, str] = {}
    lowering_hashes: dict[str, str] = {}
    for name in names:
        lowering = lowerings[name]
        basis = lowering.computational_mode_basis
        source_basis = source_lowerings_by_node[name].computational_mode_basis
        if (
            lowering.coordinate_generator_plan.rank != A5D_FIXED_GENERATOR_RANK
            or bool(torch.count_nonzero(basis.mean_bias))
            or basis.decoder_basis_sha256 != source_basis.decoder_basis_sha256
            or not torch.equal(basis.decoder_basis, source_basis.decoder_basis)
        ):
            raise RuntimeError("A5d fitted residual basis invariant failed")
        zero_hashes[name] = basis.mean_bias_sha256
        decoder_hashes[name] = basis.decoder_basis_sha256
        lowering_hashes[name] = lowering.artifact_sha256
    return {
        "graph_sha256": graph.artifact_sha256,
        "lowering_sha256_by_node": lowering_hashes,
        "zero_mean_sha256_by_node": zero_hashes,
        "decoder_sha256_by_node": decoder_hashes,
        "all_residual_basis_means_exactly_zero": True,
        "all_residual_decoders_byte_identical_to_source": True,
        "parameter_count": graph.parameter_count,
        "source_parameter_count": source_graph.parameter_count,
        "removed_source_affine_mean_parameters": (
            source_graph.parameter_count - graph.parameter_count
        ),
        "macs_per_token": graph.macs_per_token,
        "parameters_match_zero_mean_source_expectation": True,
        "macs_equal_source": True,
    }


@dataclass(frozen=True, slots=True)
class _HeldFamilyReference:
    alias: str
    indices: Tensor = field(repr=False)
    weights: Tensor = field(repr=False)
    native_states: Tensor = field(repr=False)
    frozen_states: Tensor = field(repr=False)
    frozen_kl: float
    frozen_state_nrmse: float


@dataclass(frozen=True, slots=True)
class A5dFamilyResidualCvSelection:
    """Executable source-anchored residual selection and strict receipt."""

    selected_alpha: float
    selected_ridge: float | None
    use_frozen_fallback: bool
    residual_fit: FrozenBasisGeneratorFit | None = field(repr=False)
    node_order: tuple[str, ...]
    residual_width: int
    _receipt: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        validated = validate_a5d_family_residual_cv_receipt(self._receipt)
        selection = validated["selection"]
        assert isinstance(selection, Mapping)
        if (
            self.selected_alpha != selection["selected_alpha"]
            or self.selected_ridge != selection["selected_ridge"]
            or self.use_frozen_fallback
            is not bool(selection["use_frozen_fallback"])
            or self.use_frozen_fallback != (self.residual_fit is None)
            or self.use_frozen_fallback != (self.selected_alpha == 0.0)
            or type(self.residual_width) is not int
            or self.residual_width <= 0
        ):
            raise ValueError("A5d executable selection contradicts its receipt")
        if self.residual_fit is not None:
            final = validated["final_refit"]
            assert isinstance(final, Mapping)
            final_fit = final["fit"]
            assert isinstance(final_fit, Mapping)
            if (
                self.residual_fit.graph_plan.artifact_sha256
                != final_fit["graph_sha256"]
            ):
                raise ValueError("A5d final residual fit differs from its receipt")
        object.__setattr__(
            self,
            "_receipt",
            MappingProxyType(json.loads(json.dumps(validated, allow_nan=False))),
        )

    def receipt(self) -> dict[str, object]:
        return json.loads(json.dumps(dict(self._receipt), allow_nan=False))

    def predict_residual(self, inputs: Tensor) -> Tensor:
        if self.residual_fit is None:
            raise RuntimeError("A5d selected alpha zero; no residual fit was retained")
        if (
            not isinstance(inputs, Tensor)
            or not inputs.is_floating_point()
            or inputs.ndim < 1
            or inputs.numel() == 0
            or not bool(torch.isfinite(inputs).all())
        ):
            raise ValueError("A5d prediction inputs must be finite and floating")
        runtime = inputs.to(dtype=_RUNTIME_DTYPE).contiguous()
        return _predict_fit_residual(self.residual_fit, runtime, self.node_order)

    def candidate_block_states(
        self,
        frozen_states: Tensor,
        inputs: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(frozen_states, Tensor) or frozen_states.ndim != 2:
            raise ValueError("A5d frozen states must be a row matrix")
        rows = frozen_states.shape[0]
        frozen = _matrix(frozen_states, rows=rows, label="A5d frozen states")
        if frozen.shape[1] != self.residual_width:
            raise ValueError("A5d frozen-state width drifted")
        if self.selected_alpha == 0.0:
            return _candidate_states(frozen, None, 0.0)
        if inputs is None:
            raise ValueError("positive-alpha execution requires generator inputs")
        residual = self.predict_residual(inputs)
        if residual.dtype != frozen.dtype:
            frozen = frozen.to(dtype=residual.dtype)
        return _candidate_states(frozen, residual, self.selected_alpha)


def select_a5d_family_disjoint_residual(
    *,
    bridge: A5cAuthenticatedRowBank,
    targets: A5dSourceAnchoredResidualTargets,
    source_graph: ModalGeneratorGraphPlan,
    source_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    adapter: object,
    native_block_states: Tensor,
    frozen_compiled_block_states: Tensor,
    output_boundary: str,
    ridge_grid: Sequence[float] = A5D_RIDGE_GRID,
    alpha_grid: Sequence[float] = A5D_ALPHA_GRID,
    final_head_chunk_rows: int = 8,
    final_head_token_locality_lineage_sha256: str,
) -> A5dFamilyResidualCvSelection:
    """Jointly select residual ridge/alpha without touching the outer family."""

    if not isinstance(bridge, A5cAuthenticatedRowBank):
        raise TypeError("bridge must implement the authenticated A5c row-bank protocol")
    if not isinstance(targets, A5dSourceAnchoredResidualTargets):
        raise TypeError("targets must be A5dSourceAnchoredResidualTargets")
    bridge_receipt = bridge.receipt()
    target_receipt = targets.receipt()
    if (
        not isinstance(bridge_receipt, Mapping)
        or bridge_receipt.get("receipt_sha256") != bridge.receipt_sha256
        or target_receipt.get("receipt_sha256") != targets.receipt_sha256
    ):
        raise ValueError("A5d source receipt identity drifted")
    bridge_input_sha256 = _bridge_compiled_inputs_sha256(bridge_receipt)
    training_aliases = tuple(bridge.training_family_aliases)
    ownership = dict(bridge.family_alias_by_example)
    if (
        len(training_aliases) != _INNER_FOLD_COUNT
        or set(ownership.values()) != set(training_aliases)
        or bridge.held_family_alias in ownership.values()
    ):
        raise ValueError("A5d ownership contains outer-held or incomplete data")
    if not isinstance(output_boundary, str) or not output_boundary:
        raise ValueError("output_boundary must be nonempty")
    ridges = _numeric_grid(ridge_grid, label="ridge_grid", bounded=False)
    alphas = _numeric_grid(alpha_grid, label="alpha_grid", bounded=True)
    if not alphas or alphas[0] != 0.0:
        raise ValueError("alpha_grid must begin with the exact zero control")
    if type(final_head_chunk_rows) is not int or final_head_chunk_rows <= 0:
        raise ValueError("final_head_chunk_rows must be positive")
    token_locality = _require_sha256(
        final_head_token_locality_lineage_sha256,
        label="A5d final-head token-locality lineage",
    )

    source_graph.validate_integrity()
    node_order = tuple(bridge.node_order)
    if (
        tuple(source_graph.traversal_order) != node_order
        or source_graph.interactions
        or set(source_lowerings_by_node) != set(node_order)
        or tuple(targets.node_order) != node_order
    ):
        raise ValueError("A5d source graph, target, and row-bank nodes disagree")
    source_lowerings = _authenticate_lowering_catalog(
        source_lowerings_by_node, node_order, label="A5d source lowering"
    )
    source_decoder_hashes = {
        name: source_lowerings[name].computational_mode_basis.decoder_basis_sha256
        for name in node_order
    }
    source_mean_hashes = {
        name: source_lowerings[name].computational_mode_basis.mean_bias_sha256
        for name in node_order
    }
    target_source = target_receipt["source"]
    if (
        not isinstance(target_source, Mapping)
        or dict(target_source["decoder_sha256_by_node"])
        != source_decoder_hashes
        or dict(target_source["mean_sha256_by_node"]) != source_mean_hashes
    ):
        raise ValueError("A5d target bases differ from executable source bases")
    adapter_fingerprint = getattr(adapter, "model_fingerprint", None)
    if not callable(adapter_fingerprint):
        raise TypeError("adapter must expose model_fingerprint")
    model_sha256 = _require_sha256(
        adapter_fingerprint(), label="A5d adapter model fingerprint"
    )
    if source_graph.model_fingerprint != model_sha256:
        raise ValueError("A5d source graph does not bind the adapter")

    all_rows = targets.residual_rows
    if (
        all_rows.row_keys != bridge.all_rows.row_keys
        or tuple(all_rows.rows_by_fragment)
        != tuple(bridge.all_rows.rows_by_fragment)
    ):
        raise ValueError("A5d residual targets are not in authenticated row order")
    shared_inputs = next(iter(all_rows.rows_by_fragment.values())).inputs
    bridge_inputs = next(iter(bridge.all_rows.rows_by_fragment.values())).inputs
    if not torch.equal(shared_inputs, bridge_inputs):
        raise ValueError("A5d residual and bridge compiled inputs differ")
    rows = all_rows.observations
    native = _matrix(native_block_states, rows=rows, label="native block states")
    frozen = _matrix(
        frozen_compiled_block_states,
        rows=rows,
        label="frozen compiled block states",
    )
    if (
        native.shape != frozen.shape
        or native.dtype != frozen.dtype
        or not torch.equal(
            frozen.to(dtype=torch.float64),
            targets.frozen_compiled_block_states,
        )
        or targets.residual_width != native.shape[1]
    ):
        raise ValueError("A5d native/frozen/target state banks disagree")
    widths = {node.output_width for node in source_graph.nodes}
    input_widths = {node.input_width for node in source_graph.nodes}
    if widths != {native.shape[1]} or input_widths != {shared_inputs.shape[1]}:
        raise ValueError("A5d graph and state widths disagree")
    runtime_inputs, input_cast_audit = _runtime_input_cast_audit(shared_inputs)
    if runtime_inputs.dtype != native.dtype:
        raise ValueError("A5d state dtype differs from generator runtime dtype")

    signatures = _input_row_signatures(all_rows)
    key_to_index = {key: index for index, key in enumerate(all_rows.row_keys)}
    fold_rows: dict[
        str,
        tuple[
            AlignedFragmentRows,
            AlignedFragmentRows,
            int,
            _HeldFamilyReference,
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
            all_rows, ownership, (held_alias,)
        )
        overlap = set(_input_row_signatures(fit_rows)) & set(
            _input_row_signatures(eval_rows)
        )
        if overlap:
            raise RuntimeError("A5d signature exclusion failed")
        removed = original_fit_count - fit_rows.observations
        if removed < 0:
            raise RuntimeError("A5d signature-exclusion accounting drifted")
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
        fold_rows[held_alias] = (
            fit_rows,
            eval_rows,
            removed,
            _HeldFamilyReference(
                alias=held_alias,
                indices=eval_indices,
                weights=weights,
                native_states=native_fold,
                frozen_states=frozen_fold,
                frozen_kl=frozen_kl,
                frozen_state_nrmse=_weighted_state_nrmse(
                    frozen_fold, native_fold, weights
                ),
            ),
        )

    candidate_receipts: list[dict[str, object]] = []
    for ridge in ridges:
        fold_receipts: list[dict[str, object]] = []
        for held_alias in training_aliases:
            fit_rows, eval_rows, removed, reference = fold_rows[held_alias]
            fit_split = _split_sha256(
                target_sha256=targets.receipt_sha256,
                ridge=ridge,
                held_alias=held_alias,
                role="inner_fit_six_families",
                rows=fit_rows,
            )
            eval_split = _split_sha256(
                target_sha256=targets.receipt_sha256,
                ridge=ridge,
                held_alias=held_alias,
                role="inner_evaluation_one_family",
                rows=eval_rows,
            )
            fitted = fit_zero_mean_residual_generators(
                fit_rows,
                eval_rows,
                source_graph=source_graph,
                source_lowerings_by_node=source_lowerings,
                fit_split_sha256=fit_split,
                eval_split_sha256=eval_split,
                generator_rank=A5D_FIXED_GENERATOR_RANK,
                ridge=ridge,
                output_boundary=output_boundary,
            )
            fit_evidence = _fit_evidence(
                fitted,
                source_graph=source_graph,
                source_lowerings_by_node=source_lowerings,
                node_order=node_order,
            )
            evaluation_inputs = next(
                iter(eval_rows.rows_by_fragment.values())
            ).inputs.to(dtype=_RUNTIME_DTYPE).contiguous()
            if not torch.equal(
                evaluation_inputs,
                runtime_inputs.index_select(0, reference.indices),
            ):
                raise RuntimeError("A5d fold runtime-input cast/order drifted")
            predicted = _predict_fit_residual(
                fitted, evaluation_inputs, node_order
            )
            if predicted.shape != reference.frozen_states.shape:
                raise RuntimeError("A5d predicted residual shape drifted")
            alpha_metrics: list[dict[str, object]] = []
            for alpha in alphas:
                if alpha == 0.0:
                    candidate_kl = reference.frozen_kl
                    candidate_nrmse = reference.frozen_state_nrmse
                else:
                    candidate_states = _candidate_states(
                        reference.frozen_states, predicted, alpha
                    )
                    candidate_kl = _chunked_final_head_kl(
                        adapter,
                        reference.native_states,
                        candidate_states,
                        reference.weights,
                        chunk_rows=final_head_chunk_rows,
                    )
                    candidate_nrmse = _weighted_state_nrmse(
                        candidate_states,
                        reference.native_states,
                        reference.weights,
                    )
                alpha_metrics.append(
                    {
                        "alpha": alpha,
                        "alpha_hex": alpha.hex(),
                        "is_bit_exact_frozen_control": alpha == 0.0,
                        "candidate": {
                            "fisher_weighted_final_head_kl": candidate_kl,
                            "fisher_weighted_state_nrmse": candidate_nrmse,
                        },
                    }
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
                    "fit": fit_evidence,
                    "frozen_baseline": {
                        "fisher_weighted_final_head_kl": reference.frozen_kl,
                        "fisher_weighted_state_nrmse": (
                            reference.frozen_state_nrmse
                        ),
                    },
                    "alpha_metrics": alpha_metrics,
                }
            )

        alpha_summaries: list[dict[str, object]] = []
        for alpha_index, alpha in enumerate(alphas):
            kls = [
                float(fold["alpha_metrics"][alpha_index]["candidate"][
                    "fisher_weighted_final_head_kl"
                ])
                for fold in fold_receipts
            ]
            state_values = [
                float(fold["alpha_metrics"][alpha_index]["candidate"][
                    "fisher_weighted_state_nrmse"
                ])
                for fold in fold_receipts
            ]
            alpha_summaries.append(
                {
                    "alpha": alpha,
                    "alpha_hex": alpha.hex(),
                    "family_equal_final_head_kl": math.fsum(kls) / len(kls),
                    "family_equal_state_nrmse": (
                        math.fsum(state_values) / len(state_values)
                    ),
                }
            )
        candidate_receipts.append(
            {
                "ridge": ridge,
                "ridge_hex": ridge.hex(),
                "inner_fold_count": len(fold_receipts),
                "fit_count": len(fold_receipts),
                "alpha_summaries": alpha_summaries,
                "folds": fold_receipts,
            }
        )

    joint_scores = [
        (
            float(summary["family_equal_final_head_kl"]),
            float(summary["alpha"]),
            -float(candidate["ridge"]),
            float(candidate["ridge"]),
            float(summary["family_equal_state_nrmse"]),
        )
        for candidate in candidate_receipts
        for summary in candidate["alpha_summaries"]
    ]
    winner_kl, winner_alpha, _negative_ridge, winner_ridge, winner_state = min(
        joint_scores, key=lambda item: (item[0], item[1], item[2])
    )
    baseline_values = [
        float(candidate["alpha_summaries"][0]["family_equal_final_head_kl"])
        for candidate in candidate_receipts
    ]
    if len(set(baseline_values)) != 1:
        raise RuntimeError("A5d alpha-zero baseline changed across ridges")
    baseline_kl = baseline_values[0]
    use_frozen = winner_alpha == 0.0 or not winner_kl < baseline_kl

    final_fit: FrozenBasisGeneratorFit | None = None
    if use_frozen:
        selected_alpha = 0.0
        selected_ridge: float | None = None
        final_refit: dict[str, object] = {
            "performed": False,
            "selected_ridge": None,
            "selected_alpha": 0.0,
            "fit_uses_all_outer_training_examples": False,
            "fit_observations": 0,
            "fit_examples": 0,
            "fit_fisher_normalization": "not_applicable_alpha_zero_fallback",
            "descriptive_eval_source": "not_applicable_alpha_zero_fallback",
            "descriptive_eval_is_subset_of_final_fit": False,
            "descriptive_eval_is_independent": False,
            "descriptive_eval_used_for_selection": False,
            "fit_split_sha256": None,
            "descriptive_eval_split_sha256": None,
            "fit": None,
        }
    else:
        selected_alpha = winner_alpha
        selected_ridge = winner_ridge
        full_fit_rows = _subset_equal_family_rows(
            all_rows, ownership, training_aliases
        )
        descriptive_source = _subset_exact_keys(
            all_rows, bridge.audit_rows.row_keys
        )
        descriptive_rows = _descriptive_rekey_rows(
            descriptive_source,
            binding_sha256=_domain_sha256(
                _SPLIT_DOMAIN,
                {
                    "target_receipt_sha256": targets.receipt_sha256,
                    "selected_ridge_hex": winner_ridge.hex(),
                    "selected_alpha_hex": winner_alpha.hex(),
                    "role": "post_cv_descriptive_audit_replay",
                    "source_row_key_sha256": descriptive_source.row_key_sha256,
                },
            ),
        )
        fit_split = _split_sha256(
            target_sha256=targets.receipt_sha256,
            ridge=winner_ridge,
            held_alias="post_cv_all_seven",
            role="post_cv_fit_all_seven_families",
            rows=full_fit_rows,
        )
        descriptive_split = _split_sha256(
            target_sha256=targets.receipt_sha256,
            ridge=winner_ridge,
            held_alias="post_cv_none_descriptive_replay",
            role="post_cv_non_independent_descriptive_evaluation",
            rows=descriptive_rows,
        )
        final_fit = fit_zero_mean_residual_generators(
            full_fit_rows,
            descriptive_rows,
            source_graph=source_graph,
            source_lowerings_by_node=source_lowerings,
            fit_split_sha256=fit_split,
            eval_split_sha256=descriptive_split,
            generator_rank=A5D_FIXED_GENERATOR_RANK,
            ridge=winner_ridge,
            output_boundary=output_boundary,
        )
        final_evidence = _fit_evidence(
            final_fit,
            source_graph=source_graph,
            source_lowerings_by_node=source_lowerings,
            node_order=node_order,
        )
        final_refit = {
            "performed": True,
            "selected_ridge": winner_ridge,
            "selected_alpha": winner_alpha,
            "fit_uses_all_outer_training_examples": True,
            "fit_observations": full_fit_rows.observations,
            "fit_examples": full_fit_rows.sequences,
            "fit_fisher_normalization": (
                "equal_total_mass_per_outer_training_family_and_node"
            ),
            "descriptive_eval_source": (
                "residual_target_rows_matching_bridge_audit_keys_rekeyed_after_"
                "becoming_subset_of_all_rows_fit"
            ),
            "descriptive_eval_is_subset_of_final_fit": True,
            "descriptive_eval_is_independent": False,
            "descriptive_eval_used_for_selection": False,
            "fit_split_sha256": fit_split,
            "descriptive_eval_split_sha256": descriptive_split,
            "fit": final_evidence,
        }

    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_SCHEMA,
        "format_version": (
            GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_FORMAT_VERSION
        ),
        "scientific_role": (
            "outer_training_only_nested_family_disjoint_source_anchored_"
            "residual_ridge_alpha_selection"
        ),
        "source": {
            "bridge_receipt_sha256": bridge.receipt_sha256,
            "residual_target_receipt_sha256": targets.receipt_sha256,
            "source_graph_sha256": source_graph.artifact_sha256,
            "source_graph_parameter_count": source_graph.parameter_count,
            "source_graph_macs_per_token": source_graph.macs_per_token,
            "source_model_sha256": model_sha256,
            "source_lowering_sha256_by_node": {
                name: source_lowerings[name].artifact_sha256 for name in node_order
            },
            "source_mean_sha256_by_node": source_mean_hashes,
            "source_decoder_sha256_by_node": source_decoder_hashes,
            "native_block_states_sha256": _tensor_sha256(native),
            "frozen_compiled_block_states_sha256": _tensor_sha256(frozen),
            "all_rows_key_sha256": all_rows.row_key_sha256,
            "bridge_compiled_inputs_sha256": bridge_input_sha256,
            # This field deliberately stays in the A5c runtime-cast hash
            # domain because the imported cast audit owns that identity.
            "all_rows_input_sha256": input_cast_audit[
                "source_input_sha256"
            ],
            "candidate_runtime_input_cast": input_cast_audit,
            "final_head_token_locality_lineage_sha256": token_locality,
        },
        "configuration": {
            "generator_rank": A5D_FIXED_GENERATOR_RANK,
            "ridge_grid": list(ridges),
            "ridge_grid_hex": [ridge.hex() for ridge in ridges],
            "alpha_grid": list(alphas),
            "alpha_grid_hex": [alpha.hex() for alpha in alphas],
            "inner_fold_count": _INNER_FOLD_COUNT,
            "inner_training_family_count": _INNER_TRAINING_FAMILY_COUNT,
            "inner_evaluation_family_count": 1,
            "selection_objective": _SELECTION_OBJECTIVE,
            "state_diagnostic": _STATE_DIAGNOSTIC,
            "tie_break": _TIE_BREAK,
            "fallback_rule": _FALLBACK_RULE,
            "candidate_formula": (
                "bit_exact_frozen_state_if_alpha_zero_else_frozen_state_plus_"
                "alpha_times_zero_mean_predicted_residual"
            ),
            "fit_reuse_policy": (
                "fit_once_per_ridge_and_family_fold_then_score_all_alphas"
            ),
            "output_boundary": output_boundary,
            "duplicate_cross_split_input_policy": (
                "exclude_from_inner_fit_every_row_whose_compiled_input_signature_"
                "occurs_in_that_folds_evaluation_family_then_require_zero_overlap"
            ),
            "candidate_generator_execution_dtype": "torch.float32",
            "final_head_chunk_rows": final_head_chunk_rows,
            "final_head_chunking": (
                "a5c_exact_fixed_row_chunks_over_token_local_final_norm_and_lm_"
                "head_with_float64_full_vocabulary_kl_accumulation"
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
            "tie_break": _TIE_BREAK,
            "winner_ridge_before_fallback": winner_ridge,
            "winner_ridge_hex_before_fallback": winner_ridge.hex(),
            "winner_alpha_before_fallback": winner_alpha,
            "winner_alpha_hex_before_fallback": winner_alpha.hex(),
            "winner_family_equal_final_head_kl": winner_kl,
            "winner_family_equal_state_nrmse": winner_state,
            "alpha_zero_family_equal_final_head_kl": baseline_kl,
            "absolute_kl_improvement": baseline_kl - winner_kl,
            "best_positive_alpha_strictly_improves_alpha_zero": not use_frozen,
            "use_frozen_fallback": use_frozen,
            "selected_alpha": selected_alpha,
            "selected_alpha_hex": selected_alpha.hex(),
            "selected_ridge": selected_ridge,
            "selected_ridge_hex": (
                None if selected_ridge is None else selected_ridge.hex()
            ),
        },
        "final_refit": final_refit,
        "safety": dict(_SAFETY),
    }
    payload["receipt_sha256"] = _domain_sha256(_REPORT_DOMAIN, payload)
    validated = validate_a5d_family_residual_cv_receipt(payload)
    return A5dFamilyResidualCvSelection(
        selected_alpha=selected_alpha,
        selected_ridge=selected_ridge,
        use_frozen_fallback=use_frozen,
        residual_fit=final_fit,
        node_order=node_order,
        residual_width=native.shape[1],
        _receipt=validated,
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


def _finite(value: object, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} is outside its finite range")
    return result


def _validate_metric(value: object, *, label: str) -> tuple[float, float]:
    metric = _strict_fields(
        value,
        {
            "fisher_weighted_final_head_kl",
            "fisher_weighted_state_nrmse",
        },
        label=label,
    )
    return (
        _finite(
            metric["fisher_weighted_final_head_kl"],
            label=f"{label} KL",
        ),
        _finite(
            metric["fisher_weighted_state_nrmse"],
            label=f"{label} state NRMSE",
        ),
    )


def _validate_hash_catalog(
    value: object,
    *,
    names: Sequence[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{label} catalog drifted")
    for name in names:
        _require_sha256(value[name], label=f"{label} {name}")
    return value


def _validate_fit_evidence(
    value: object,
    *,
    names: Sequence[str],
    source_decoder_hashes: Mapping[str, object],
    source_parameters: int,
    source_macs: int,
    label: str,
) -> Mapping[str, object]:
    evidence = _strict_fields(
        value,
        {
            "graph_sha256",
            "lowering_sha256_by_node",
            "zero_mean_sha256_by_node",
            "decoder_sha256_by_node",
            "all_residual_basis_means_exactly_zero",
            "all_residual_decoders_byte_identical_to_source",
            "parameter_count",
            "source_parameter_count",
            "removed_source_affine_mean_parameters",
            "macs_per_token",
            "parameters_match_zero_mean_source_expectation",
            "macs_equal_source",
        },
        label=label,
    )
    _require_sha256(evidence["graph_sha256"], label=f"{label} graph")
    _validate_hash_catalog(
        evidence["lowering_sha256_by_node"], names=names, label=f"{label} lowering"
    )
    _validate_hash_catalog(
        evidence["zero_mean_sha256_by_node"],
        names=names,
        label=f"{label} zero mean",
    )
    decoder_hashes = _validate_hash_catalog(
        evidence["decoder_sha256_by_node"],
        names=names,
        label=f"{label} decoder",
    )
    if (
        dict(decoder_hashes) != dict(source_decoder_hashes)
        or evidence["all_residual_basis_means_exactly_zero"] is not True
        or evidence["all_residual_decoders_byte_identical_to_source"] is not True
        or evidence["parameters_match_zero_mean_source_expectation"] is not True
        or evidence["macs_equal_source"] is not True
        or evidence["source_parameter_count"] != source_parameters
        or type(evidence["removed_source_affine_mean_parameters"]) is not int
        or evidence["removed_source_affine_mean_parameters"] < 0
        or evidence["parameter_count"]
        != source_parameters - evidence["removed_source_affine_mean_parameters"]
        or evidence["macs_per_token"] != source_macs
    ):
        raise ValueError(f"{label} basis/resource invariant drifted")
    return evidence


def validate_a5d_family_residual_cv_receipt(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Strictly authenticate one tensor-free A5d nested-CV receipt."""

    if not isinstance(value, Mapping):
        raise TypeError("A5d residual-CV receipt must be a mapping")
    if _contains_tensor(value):
        raise TypeError("A5d residual-CV receipt must be tensor-free")
    raw = json.loads(json.dumps(dict(value), allow_nan=False))
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
        label="A5d receipt",
    )
    if (
        raw["schema"] != GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_SCHEMA
        or raw["format_version"]
        != GEMMA3_L10_L17_A5D_FAMILY_RESIDUAL_CV_FORMAT_VERSION
        or raw["scientific_role"]
        != (
            "outer_training_only_nested_family_disjoint_source_anchored_"
            "residual_ridge_alpha_selection"
        )
        or raw["safety"] != _SAFETY
    ):
        raise ValueError("A5d receipt header or safety contract drifted")

    source = _strict_fields(
        raw["source"],
        {
            "bridge_receipt_sha256",
            "residual_target_receipt_sha256",
            "source_graph_sha256",
            "source_graph_parameter_count",
            "source_graph_macs_per_token",
            "source_model_sha256",
            "source_lowering_sha256_by_node",
            "source_mean_sha256_by_node",
            "source_decoder_sha256_by_node",
            "native_block_states_sha256",
            "frozen_compiled_block_states_sha256",
            "all_rows_key_sha256",
            "bridge_compiled_inputs_sha256",
            "all_rows_input_sha256",
            "candidate_runtime_input_cast",
            "final_head_token_locality_lineage_sha256",
        },
        label="A5d source",
    )
    for name in (
        "bridge_receipt_sha256",
        "residual_target_receipt_sha256",
        "source_graph_sha256",
        "source_model_sha256",
        "native_block_states_sha256",
        "frozen_compiled_block_states_sha256",
        "all_rows_key_sha256",
        "bridge_compiled_inputs_sha256",
        "all_rows_input_sha256",
        "final_head_token_locality_lineage_sha256",
    ):
        _require_sha256(source[name], label=f"A5d source {name}")
    if (
        type(source["source_graph_parameter_count"]) is not int
        or source["source_graph_parameter_count"] <= 0
        or type(source["source_graph_macs_per_token"]) is not int
        or source["source_graph_macs_per_token"] <= 0
    ):
        raise ValueError("A5d source resource counts are invalid")
    lowering_hashes = source["source_lowering_sha256_by_node"]
    if not isinstance(lowering_hashes, Mapping) or len(lowering_hashes) != 4:
        raise ValueError("A5d source lowering catalog must contain four nodes")
    names = tuple(lowering_hashes)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("A5d source node names are invalid")
    _validate_hash_catalog(lowering_hashes, names=names, label="A5d source lowering")
    _validate_hash_catalog(
        source["source_mean_sha256_by_node"],
        names=names,
        label="A5d source mean",
    )
    source_decoder_hashes = _validate_hash_catalog(
        source["source_decoder_sha256_by_node"],
        names=names,
        label="A5d source decoder",
    )
    input_cast = source["candidate_runtime_input_cast"]
    if not isinstance(input_cast, Mapping):
        raise ValueError("A5d runtime input-cast evidence is missing")
    # A5c owns the exact cast-audit schema; pin the safety-critical fields and
    # bind the complete mapping through the enclosing receipt hash.
    if (
        input_cast.get("source_input_sha256") != source["all_rows_input_sha256"]
        or input_cast.get("runtime_dtype") != "torch.float32"
        or input_cast.get("passed") is not True
    ):
        raise ValueError("A5d runtime input-cast evidence drifted")
    for name in ("source_input_sha256", "runtime_input_sha256", "lineage_sha256"):
        _require_sha256(input_cast.get(name), label=f"A5d input cast {name}")

    config = _strict_fields(
        raw["configuration"],
        {
            "generator_rank",
            "ridge_grid",
            "ridge_grid_hex",
            "alpha_grid",
            "alpha_grid_hex",
            "inner_fold_count",
            "inner_training_family_count",
            "inner_evaluation_family_count",
            "selection_objective",
            "state_diagnostic",
            "tie_break",
            "fallback_rule",
            "candidate_formula",
            "fit_reuse_policy",
            "output_boundary",
            "duplicate_cross_split_input_policy",
            "candidate_generator_execution_dtype",
            "final_head_chunk_rows",
            "final_head_chunking",
        },
        label="A5d configuration",
    )
    ridges = _numeric_grid(
        config["ridge_grid"], label="receipt ridge_grid", bounded=False
    )
    alphas = _numeric_grid(
        config["alpha_grid"], label="receipt alpha_grid", bounded=True
    )
    if (
        alphas[0] != 0.0
        or config["generator_rank"] != A5D_FIXED_GENERATOR_RANK
        or config["ridge_grid_hex"] != [value.hex() for value in ridges]
        or config["alpha_grid_hex"] != [value.hex() for value in alphas]
        or config["inner_fold_count"] != _INNER_FOLD_COUNT
        or config["inner_training_family_count"]
        != _INNER_TRAINING_FAMILY_COUNT
        or config["inner_evaluation_family_count"] != 1
        or config["selection_objective"] != _SELECTION_OBJECTIVE
        or config["state_diagnostic"] != _STATE_DIAGNOSTIC
        or config["tie_break"] != _TIE_BREAK
        or config["fallback_rule"] != _FALLBACK_RULE
        or config["candidate_formula"]
        != (
            "bit_exact_frozen_state_if_alpha_zero_else_frozen_state_plus_"
            "alpha_times_zero_mean_predicted_residual"
        )
        or config["fit_reuse_policy"]
        != "fit_once_per_ridge_and_family_fold_then_score_all_alphas"
        or not isinstance(config["output_boundary"], str)
        or not config["output_boundary"]
        or config["duplicate_cross_split_input_policy"]
        != (
            "exclude_from_inner_fit_every_row_whose_compiled_input_signature_"
            "occurs_in_that_folds_evaluation_family_then_require_zero_overlap"
        )
        or config["candidate_generator_execution_dtype"] != "torch.float32"
        or type(config["final_head_chunk_rows"]) is not int
        or config["final_head_chunk_rows"] <= 0
        or config["final_head_chunking"]
        != (
            "a5c_exact_fixed_row_chunks_over_token_local_final_norm_and_lm_"
            "head_with_float64_full_vocabulary_kl_accumulation"
        )
    ):
        raise ValueError("A5d configuration drifted")

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
        label="A5d ownership",
    )
    aliases = tuple(ownership["outer_training_family_aliases"])
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
        raise ValueError("A5d outer-held ownership barrier drifted")
    _require_sha256(
        ownership["outer_training_membership_sha256"],
        label="A5d training membership",
    )

    candidates = raw["candidates"]
    if (
        not isinstance(candidates, list)
        or len(candidates) != len(ridges)
    ):
        raise ValueError("A5d candidates do not cover the ridge grid")
    joint_scores: list[tuple[float, float, float, float, float]] = []
    baseline_by_alias: dict[str, tuple[float, float]] = {}
    baseline_aggregates: list[float] = []
    for ridge, raw_candidate in zip(ridges, candidates, strict=True):
        candidate = _strict_fields(
            raw_candidate,
            {
                "ridge",
                "ridge_hex",
                "inner_fold_count",
                "fit_count",
                "alpha_summaries",
                "folds",
            },
            label="A5d ridge candidate",
        )
        if (
            candidate["ridge"] != ridge
            or candidate["ridge_hex"] != ridge.hex()
            or candidate["inner_fold_count"] != _INNER_FOLD_COUNT
            or candidate["fit_count"] != _INNER_FOLD_COUNT
        ):
            raise ValueError("A5d ridge candidate identity drifted")
        folds = candidate["folds"]
        if not isinstance(folds, list) or len(folds) != _INNER_FOLD_COUNT:
            raise ValueError("A5d fold count drifted")
        fold_values: list[list[tuple[float, float]]] = [
            [] for _ in alphas
        ]
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
                    "fit",
                    "frozen_baseline",
                    "alpha_metrics",
                },
                label="A5d inner fold",
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
                or type(fold["fit_rows_removed_for_signature_overlap"])
                is not int
                or fold["fit_rows_removed_for_signature_overlap"] < 0
                or fold["input_signature_overlap_count"] != 0
                or fold["input_signatures_disjoint"] is not True
            ):
                raise ValueError("A5d fold ownership/signature barrier drifted")
            for name in (
                "fit_membership_sha256",
                "evaluation_membership_sha256",
                "fit_row_key_sha256",
                "evaluation_row_key_sha256",
                "fit_split_sha256",
                "evaluation_split_sha256",
            ):
                _require_sha256(fold[name], label=f"A5d fold {name}")
            expected_fit = _domain_sha256(
                _SPLIT_DOMAIN,
                {
                    "residual_target_receipt_sha256": source[
                        "residual_target_receipt_sha256"
                    ],
                    "ridge_hex": ridge.hex(),
                    "held_inner_family_alias": alias,
                    "role": "inner_fit_six_families",
                    "row_key_sha256": fold["fit_row_key_sha256"],
                    "observations": fold["fit_observations"],
                    "sequences": fold["fit_examples"],
                },
            )
            expected_eval = _domain_sha256(
                _SPLIT_DOMAIN,
                {
                    "residual_target_receipt_sha256": source[
                        "residual_target_receipt_sha256"
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
                fold["fit_split_sha256"] != expected_fit
                or fold["evaluation_split_sha256"] != expected_eval
            ):
                raise ValueError("A5d fold split lineage is contradictory")
            _validate_fit_evidence(
                fold["fit"],
                names=names,
                source_decoder_hashes=source_decoder_hashes,
                source_parameters=source["source_graph_parameter_count"],
                source_macs=source["source_graph_macs_per_token"],
                label="A5d fold fit",
            )
            frozen_pair = _validate_metric(
                fold["frozen_baseline"], label="A5d frozen baseline"
            )
            if alias in baseline_by_alias and baseline_by_alias[alias] != frozen_pair:
                raise ValueError("A5d frozen baseline changed across ridges")
            baseline_by_alias.setdefault(alias, frozen_pair)
            alpha_metrics = fold["alpha_metrics"]
            if not isinstance(alpha_metrics, list) or len(alpha_metrics) != len(alphas):
                raise ValueError("A5d fold alpha coverage drifted")
            for index, (alpha, raw_alpha) in enumerate(
                zip(alphas, alpha_metrics, strict=True)
            ):
                alpha_metric = _strict_fields(
                    raw_alpha,
                    {
                        "alpha",
                        "alpha_hex",
                        "is_bit_exact_frozen_control",
                        "candidate",
                    },
                    label="A5d fold alpha metric",
                )
                pair = _validate_metric(
                    alpha_metric["candidate"], label="A5d alpha candidate"
                )
                if (
                    alpha_metric["alpha"] != alpha
                    or alpha_metric["alpha_hex"] != alpha.hex()
                    or alpha_metric["is_bit_exact_frozen_control"]
                    is not (alpha == 0.0)
                    or (alpha == 0.0 and pair != frozen_pair)
                ):
                    raise ValueError("A5d alpha-zero or alpha identity drifted")
                fold_values[index].append(pair)

        summaries = candidate["alpha_summaries"]
        if not isinstance(summaries, list) or len(summaries) != len(alphas):
            raise ValueError("A5d alpha summaries drifted")
        for alpha, raw_summary, values in zip(
            alphas, summaries, fold_values, strict=True
        ):
            summary = _strict_fields(
                raw_summary,
                {
                    "alpha",
                    "alpha_hex",
                    "family_equal_final_head_kl",
                    "family_equal_state_nrmse",
                },
                label="A5d alpha summary",
            )
            expected_kl = math.fsum(pair[0] for pair in values) / len(values)
            expected_state = math.fsum(pair[1] for pair in values) / len(values)
            observed_kl = _finite(
                summary["family_equal_final_head_kl"], label="A5d summary KL"
            )
            observed_state = _finite(
                summary["family_equal_state_nrmse"],
                label="A5d summary state NRMSE",
            )
            if (
                summary["alpha"] != alpha
                or summary["alpha_hex"] != alpha.hex()
                or not math.isclose(
                    observed_kl, expected_kl, rel_tol=1.0e-12, abs_tol=1.0e-15
                )
                or not math.isclose(
                    observed_state,
                    expected_state,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            ):
                raise ValueError("A5d family-equal alpha summary contradicts folds")
            joint_scores.append(
                (observed_kl, alpha, -ridge, ridge, observed_state)
            )
            if alpha == 0.0:
                baseline_aggregates.append(observed_kl)

    if len(set(baseline_aggregates)) != 1:
        raise ValueError("A5d alpha-zero aggregate changed across ridges")
    winner_kl, winner_alpha, _neg_ridge, winner_ridge, winner_state = min(
        joint_scores, key=lambda item: (item[0], item[1], item[2])
    )
    baseline_kl = baseline_aggregates[0]
    fallback = winner_alpha == 0.0 or not winner_kl < baseline_kl
    selection = _strict_fields(
        raw["selection"],
        {
            "objective",
            "tie_break",
            "winner_ridge_before_fallback",
            "winner_ridge_hex_before_fallback",
            "winner_alpha_before_fallback",
            "winner_alpha_hex_before_fallback",
            "winner_family_equal_final_head_kl",
            "winner_family_equal_state_nrmse",
            "alpha_zero_family_equal_final_head_kl",
            "absolute_kl_improvement",
            "best_positive_alpha_strictly_improves_alpha_zero",
            "use_frozen_fallback",
            "selected_alpha",
            "selected_alpha_hex",
            "selected_ridge",
            "selected_ridge_hex",
        },
        label="A5d selection",
    )
    selected_alpha = 0.0 if fallback else winner_alpha
    selected_ridge = None if fallback else winner_ridge
    if (
        selection["objective"] != _SELECTION_OBJECTIVE
        or selection["tie_break"] != _TIE_BREAK
        or selection["winner_ridge_before_fallback"] != winner_ridge
        or selection["winner_ridge_hex_before_fallback"] != winner_ridge.hex()
        or selection["winner_alpha_before_fallback"] != winner_alpha
        or selection["winner_alpha_hex_before_fallback"] != winner_alpha.hex()
        or selection["winner_family_equal_final_head_kl"] != winner_kl
        or selection["winner_family_equal_state_nrmse"] != winner_state
        or selection["alpha_zero_family_equal_final_head_kl"] != baseline_kl
        or not math.isclose(
            _finite(
                selection["absolute_kl_improvement"],
                label="A5d KL improvement",
                minimum=0.0,
            ),
            baseline_kl - winner_kl,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        or selection["best_positive_alpha_strictly_improves_alpha_zero"]
        is not (not fallback)
        or selection["use_frozen_fallback"] is not fallback
        or selection["selected_alpha"] != selected_alpha
        or selection["selected_alpha_hex"] != selected_alpha.hex()
        or selection["selected_ridge"] != selected_ridge
        or selection["selected_ridge_hex"]
        != (None if selected_ridge is None else selected_ridge.hex())
    ):
        raise ValueError("A5d selection contradicts candidate scores")

    final = _strict_fields(
        raw["final_refit"],
        {
            "performed",
            "selected_ridge",
            "selected_alpha",
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
            "fit",
        },
        label="A5d final refit",
    )
    if fallback:
        expected_final = {
            "performed": False,
            "selected_ridge": None,
            "selected_alpha": 0.0,
            "fit_uses_all_outer_training_examples": False,
            "fit_observations": 0,
            "fit_examples": 0,
            "fit_fisher_normalization": "not_applicable_alpha_zero_fallback",
            "descriptive_eval_source": "not_applicable_alpha_zero_fallback",
            "descriptive_eval_is_subset_of_final_fit": False,
            "descriptive_eval_is_independent": False,
            "descriptive_eval_used_for_selection": False,
            "fit_split_sha256": None,
            "descriptive_eval_split_sha256": None,
            "fit": None,
        }
        if final != expected_final:
            raise ValueError("A5d fallback final-refit receipt drifted")
    else:
        if (
            final["performed"] is not True
            or final["selected_ridge"] != selected_ridge
            or final["selected_alpha"] != selected_alpha
            or final["fit_uses_all_outer_training_examples"] is not True
            or final["fit_observations"]
            != ownership["outer_training_observation_count"]
            or final["fit_examples"] != ownership["outer_training_example_count"]
            or final["fit_fisher_normalization"]
            != "equal_total_mass_per_outer_training_family_and_node"
            or final["descriptive_eval_source"]
            != (
                "residual_target_rows_matching_bridge_audit_keys_rekeyed_after_"
                "becoming_subset_of_all_rows_fit"
            )
            or final["descriptive_eval_is_subset_of_final_fit"] is not True
            or final["descriptive_eval_is_independent"] is not False
            or final["descriptive_eval_used_for_selection"] is not False
        ):
            raise ValueError("A5d final refit contract drifted")
        _require_sha256(final["fit_split_sha256"], label="A5d final fit split")
        _require_sha256(
            final["descriptive_eval_split_sha256"],
            label="A5d final descriptive split",
        )
        _validate_fit_evidence(
            final["fit"],
            names=names,
            source_decoder_hashes=source_decoder_hashes,
            source_parameters=source["source_graph_parameter_count"],
            source_macs=source["source_graph_macs_per_token"],
            label="A5d final fit",
        )

    supplied = _require_sha256(raw["receipt_sha256"], label="A5d receipt")
    payload = dict(raw)
    payload.pop("receipt_sha256")
    if supplied != _domain_sha256(_REPORT_DOMAIN, payload):
        raise ValueError("A5d receipt hash mismatch")
    return raw
