"""Source-anchored, zero-mean residual targets for the Layer-17 A5d rung.

This module deliberately stops before generator fitting, nested selection, or
Gemma execution.  It performs the one boundary-sensitive translation needed
by those later stages:

* retain the captured source-graph block state exactly;
* form the desired additive correction relative to that exact source state;
* project only that correction into the linear span of the already-frozen
  decoder tensors; and
* expose zero-mean per-node rows suitable for a future residual generator.

The source affine means are authenticated but are never decoded into the
residual rows.  In particular, callers must not pass these rows to an affine
decoder which restores the source means.
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

from .computational_modes import ComputationalModeBasis
from .gemma3_l10_l17_a5_frozen_affine_capacity_oracle import (
    build_frozen_affine_image,
)
from .gemma3_modal_generator_dev_experiment import LayerFragmentRows
from .gemma3_modal_generator_terminal_fanin import AlignedFragmentRows


__all__ = [
    "GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_SCHEMA",
    "A5dSourceAnchoredResidualTargets",
    "A5dZeroMeanJointResidualProjection",
    "build_a5d_source_anchored_residual_targets",
    "project_source_anchored_residual_to_joint_decoder_span",
    "validate_a5d_source_anchored_residual_receipt",
]


GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5d_source_anchored_residual"
)
GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_FORMAT_VERSION = 1

_NODE_COUNT = 4
_JOINT_COORDINATE_WIDTH = 182
_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5d-source-anchored-residual:v1\0"
)
_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a5d-source-anchored-tensor:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUM_RTOL = 1.0e-11
_SUM_ATOL = 1.0e-11
_ORTHOGONALITY_SCALE = 1.0e-9

_SCIENTIFIC_ROLE = (
    "outer_training_only_exact_source_anchored_zero_mean_residual_targets"
)
_SOURCE_FORMULA = (
    "frozen_compiled_block_states-minus-compiled_correction_base_states"
)
_TARGET_FORMULA = (
    "summed_oracle_node_contributions-minus-exact_source_correction"
)
_PROJECTION_FORMULA = (
    "residual_target@pinv(concatenated_decoder.T).T@concatenated_decoder"
)
_NODE_DECODE_FORMULA = "delta_coordinates@source_decoder_basis"
_PROJECTION_METHOD = (
    "float64_linear_decoder_svd_pseudoinverse_minimum_l2_norm"
)
_SAFETY = {
    "contains_tensors": False,
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "outer_held_family_rows_accepted": False,
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


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout is not torch.strided:
        raise TypeError("tensor hash input must be a strided Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "shape": tuple(int(width) for width in canonical.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


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


def _strict_fields(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")
    return value


def _finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (
        minimum is not None and result < minimum
    ):
        raise ValueError(f"{label} is outside its allowed range")
    return result


def _canonical_matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.layout is not torch.strided
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(f"{label} must be a nonempty floating matrix")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result.clone()


def _error_statistics(value: Tensor, *, target: Tensor) -> dict[str, float]:
    if value.shape != target.shape or value.numel() == 0:
        raise ValueError("error and target tensors must be aligned")
    error = value.detach().to(device="cpu", dtype=torch.float64)
    reference = target.detach().to(device="cpu", dtype=torch.float64)
    rms = float(torch.sqrt(error.square().mean()).item())
    target_rms = float(torch.sqrt(reference.square().mean()).item())
    return {
        "max_abs_error": float(error.abs().max().item()),
        "rms_error": rms,
        "target_rms": target_rms,
        "nrmse": rms / max(target_rms, 1.0e-30),
    }


def _ordered_bases(
    bases_by_node: Mapping[str, ComputationalModeBasis],
    node_order: Sequence[str],
) -> tuple[tuple[str, ...], tuple[ComputationalModeBasis, ...]]:
    names = tuple(node_order)
    if (
        len(names) != _NODE_COUNT
        or len(set(names)) != _NODE_COUNT
        or set(names) != set(bases_by_node)
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("A5d basis order must exactly cover four named nodes")
    bases: list[ComputationalModeBasis] = []
    for name in names:
        basis = bases_by_node[name]
        if not isinstance(basis, ComputationalModeBasis):
            raise TypeError("A5d basis catalog contains a non-basis value")
        basis.validate_integrity()
        bases.append(basis)
    return names, tuple(bases)


@dataclass(frozen=True, slots=True)
class A5dZeroMeanJointResidualProjection:
    """Ephemeral minimum-norm projection of an additive block residual."""

    node_order: tuple[str, ...]
    rank_by_node: tuple[int, ...]
    exact_source_correction: Tensor = field(repr=False)
    oracle_correction: Tensor = field(repr=False)
    residual_target: Tensor = field(repr=False)
    joint_coordinates: Tensor = field(repr=False)
    contributions_by_node: Mapping[str, Tensor] = field(repr=False)
    prediction: Tensor = field(repr=False)
    combined_decoder: Tensor = field(repr=False)
    basis_sha256_by_node: Mapping[str, str]
    mean_sha256_by_node: Mapping[str, str]
    decoder_sha256_by_node: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            type(self.node_order) is not tuple
            or len(self.node_order) != _NODE_COUNT
            or len(set(self.node_order)) != _NODE_COUNT
            or type(self.rank_by_node) is not tuple
            or len(self.rank_by_node) != _NODE_COUNT
            or any(type(rank) is not int or rank <= 0 for rank in self.rank_by_node)
            or sum(self.rank_by_node) != _JOINT_COORDINATE_WIDTH
        ):
            raise ValueError("A5d projection node/rank catalog is invalid")
        source = _canonical_matrix(
            self.exact_source_correction, label="exact source correction"
        )
        oracle = _canonical_matrix(
            self.oracle_correction, label="oracle correction"
        )
        residual = _canonical_matrix(self.residual_target, label="residual target")
        coordinates = _canonical_matrix(
            self.joint_coordinates, label="joint residual coordinates"
        )
        prediction = _canonical_matrix(
            self.prediction, label="projected residual"
        )
        decoder = _canonical_matrix(
            self.combined_decoder, label="combined decoder"
        )
        if (
            source.shape != oracle.shape
            or residual.shape != source.shape
            or prediction.shape != source.shape
            or coordinates.shape
            != (source.shape[0], _JOINT_COORDINATE_WIDTH)
            or decoder.shape
            != (_JOINT_COORDINATE_WIDTH, source.shape[1])
        ):
            raise ValueError("A5d projection tensor shapes are inconsistent")
        if not torch.equal(residual, oracle - source):
            raise ValueError("A5d residual target is not oracle minus source")
        expected_prediction = coordinates @ decoder
        if not torch.allclose(
            prediction,
            expected_prediction,
            rtol=_SUM_RTOL,
            atol=_SUM_ATOL,
        ):
            raise ValueError("A5d projected residual differs from its coordinates")

        contributions = dict(self.contributions_by_node)
        if set(contributions) != set(self.node_order):
            raise ValueError("A5d residual contribution catalog is incomplete")
        canonical_contributions: dict[str, Tensor] = {}
        start = 0
        for name, rank in zip(self.node_order, self.rank_by_node, strict=True):
            stop = start + rank
            value = _canonical_matrix(
                contributions[name], label=f"A5d residual contribution {name}"
            )
            if value.shape != source.shape:
                raise ValueError("A5d residual contribution width drifted")
            expected = coordinates[:, start:stop] @ decoder[start:stop]
            if not torch.allclose(
                value, expected, rtol=_SUM_RTOL, atol=_SUM_ATOL
            ):
                raise ValueError(
                    "A5d residual node decode injected a non-decoder term"
                )
            canonical_contributions[name] = value
            start = stop
        summed = torch.stack(
            tuple(canonical_contributions[name] for name in self.node_order)
        ).sum(dim=0)
        if not torch.allclose(
            summed, prediction, rtol=_SUM_RTOL, atol=_SUM_ATOL
        ):
            raise ValueError("A5d residual node contributions do not sum")

        hash_catalogs: list[tuple[str, Mapping[str, str]]] = [
            ("basis", self.basis_sha256_by_node),
            ("mean", self.mean_sha256_by_node),
            ("decoder", self.decoder_sha256_by_node),
        ]
        for label, catalog in hash_catalogs:
            copied = dict(catalog)
            if set(copied) != set(self.node_order):
                raise ValueError(f"A5d {label} hash catalog is incomplete")
            for digest in copied.values():
                _require_sha256(digest, label=f"A5d {label}")
            object.__setattr__(
                self,
                f"{label}_sha256_by_node",
                MappingProxyType(copied),
            )
        for name, value in (
            ("exact_source_correction", source),
            ("oracle_correction", oracle),
            ("residual_target", residual),
            ("joint_coordinates", coordinates),
            ("prediction", prediction),
            ("combined_decoder", decoder),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "contributions_by_node",
            MappingProxyType(canonical_contributions),
        )
        self.validate_integrity()

    @property
    def observations(self) -> int:
        return int(self.residual_target.shape[0])

    @property
    def residual_width(self) -> int:
        return int(self.residual_target.shape[1])

    def validate_integrity(self) -> None:
        if not torch.equal(
            self.residual_target,
            self.oracle_correction - self.exact_source_correction,
        ):
            raise ValueError("A5d residual target tensor drifted")
        if not torch.allclose(
            self.prediction,
            self.joint_coordinates @ self.combined_decoder,
            rtol=_SUM_RTOL,
            atol=_SUM_ATOL,
        ):
            raise ValueError("A5d residual projection tensor drifted")
        start = 0
        decoded: list[Tensor] = []
        for name, rank in zip(self.node_order, self.rank_by_node, strict=True):
            stop = start + rank
            expected = (
                self.joint_coordinates[:, start:stop]
                @ self.combined_decoder[start:stop]
            )
            value = self.contributions_by_node[name]
            if not torch.allclose(
                value, expected, rtol=_SUM_RTOL, atol=_SUM_ATOL
            ):
                raise ValueError("A5d zero-mean node contribution drifted")
            decoded.append(value)
            start = stop
        summed = torch.stack(tuple(decoded)).sum(dim=0)
        if not torch.allclose(
            summed, self.prediction, rtol=_SUM_RTOL, atol=_SUM_ATOL
        ):
            raise ValueError("A5d residual contribution sum drifted")
        error = self.prediction - self.residual_target
        normal_residual = error @ self.combined_decoder.T
        tolerance = _ORTHOGONALITY_SCALE * max(
            1.0, float(self.residual_target.abs().max().item())
        )
        if float(normal_residual.abs().max().item()) > tolerance:
            raise ValueError("A5d residual projection is not span-orthogonal")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        summed = torch.stack(
            tuple(
                self.contributions_by_node[name] for name in self.node_order
            )
        ).sum(dim=0)
        projection_error = self.prediction - self.residual_target
        sum_error = summed - self.prediction
        normal_residual = projection_error @ self.combined_decoder.T
        tolerance = _ORTHOGONALITY_SCALE * max(
            1.0, float(self.residual_target.abs().max().item())
        )
        return {
            "method": _PROJECTION_METHOD,
            "projection_formula": _PROJECTION_FORMULA,
            "joint_coordinate_width": _JOINT_COORDINATE_WIDTH,
            "combined_decoder_matrix_rank": int(
                torch.linalg.matrix_rank(self.combined_decoder).item()
            ),
            "combined_decoder_sha256": _tensor_sha256(self.combined_decoder),
            "residual_target_sha256": _tensor_sha256(self.residual_target),
            "joint_coordinates_sha256": _tensor_sha256(self.joint_coordinates),
            "projected_residual_sha256": _tensor_sha256(self.prediction),
            "residual_contribution_sha256_by_node": {
                name: _tensor_sha256(self.contributions_by_node[name])
                for name in self.node_order
            },
            "summed_node_contribution_sha256": _tensor_sha256(summed),
            "projection_error": _error_statistics(
                projection_error, target=self.residual_target
            ),
            "per_node_sum_error": _error_statistics(
                sum_error, target=self.prediction
            ),
            "normal_equation_orthogonality": {
                "max_abs_error": float(normal_residual.abs().max().item()),
                "rms_error": float(
                    torch.sqrt(normal_residual.square().mean()).item()
                ),
                "absolute_tolerance": tolerance,
                "passed": True,
            },
        }


def project_source_anchored_residual_to_joint_decoder_span(
    exact_source_correction: Tensor,
    oracle_correction: Tensor,
    *,
    bases_by_node: Mapping[str, ComputationalModeBasis],
    node_order: Sequence[str],
) -> A5dZeroMeanJointResidualProjection:
    """Project ``oracle - exact_source`` without using affine mode means."""

    names, bases = _ordered_bases(bases_by_node, node_order)
    image = build_frozen_affine_image(bases_by_node, node_order=names)
    source = _canonical_matrix(
        exact_source_correction, label="exact source correction"
    )
    oracle = _canonical_matrix(oracle_correction, label="oracle correction")
    if (
        source.shape != oracle.shape
        or source.shape[1] != image.residual_width
    ):
        raise ValueError("A5d source/oracle correction shapes disagree")
    residual = (oracle - source).contiguous()
    # This is a linear residual projection.  ``image.mean_sum`` is
    # intentionally unused: restoring it here would inject one source affine
    # mean per node into an additive correction.
    coordinates = (
        residual @ torch.linalg.pinv(image.decoder.T).T
    ).contiguous()
    prediction = (coordinates @ image.decoder).contiguous()
    contributions: dict[str, Tensor] = {}
    start = 0
    for name, basis in zip(names, bases, strict=True):
        stop = start + basis.rank
        contributions[name] = (
            coordinates[:, start:stop] @ basis.decoder_basis
        ).contiguous()
        start = stop
    return A5dZeroMeanJointResidualProjection(
        node_order=names,
        rank_by_node=tuple(basis.rank for basis in bases),
        exact_source_correction=source,
        oracle_correction=oracle,
        residual_target=residual,
        joint_coordinates=coordinates,
        contributions_by_node=contributions,
        prediction=prediction,
        combined_decoder=image.decoder,
        basis_sha256_by_node={
            name: basis.artifact_sha256
            for name, basis in zip(names, bases, strict=True)
        },
        mean_sha256_by_node={
            name: basis.mean_bias_sha256
            for name, basis in zip(names, bases, strict=True)
        },
        decoder_sha256_by_node={
            name: basis.decoder_basis_sha256
            for name, basis in zip(names, bases, strict=True)
        },
    )


@dataclass(frozen=True, slots=True)
class A5dSourceAnchoredResidualTargets:
    """Zero-mean fit rows plus their exact-source provenance."""

    residual_rows: AlignedFragmentRows
    projection: A5dZeroMeanJointResidualProjection = field(repr=False)
    frozen_compiled_block_states: Tensor = field(repr=False)
    compiled_correction_base_states: Tensor = field(repr=False)
    fragment_id_by_node: Mapping[str, str]
    _receipt: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.residual_rows, AlignedFragmentRows):
            raise TypeError("residual_rows must be AlignedFragmentRows")
        if not isinstance(
            self.projection, A5dZeroMeanJointResidualProjection
        ):
            raise TypeError("projection must be an A5d residual projection")
        frozen = _canonical_matrix(
            self.frozen_compiled_block_states,
            label="frozen compiled block states",
        )
        base = _canonical_matrix(
            self.compiled_correction_base_states,
            label="compiled correction base states",
        )
        if (
            frozen.shape != base.shape
            or frozen.shape
            != (
                self.projection.observations,
                self.projection.residual_width,
            )
            or not torch.equal(
                self.projection.exact_source_correction, frozen - base
            )
        ):
            raise ValueError("A5d source-state decomposition drifted")
        fragments = dict(self.fragment_id_by_node)
        if (
            set(fragments) != set(self.projection.node_order)
            or len(set(fragments.values())) != _NODE_COUNT
            or tuple(self.residual_rows.rows_by_fragment)
            != tuple(fragments[name] for name in self.projection.node_order)
        ):
            raise ValueError("A5d residual fragment catalog is invalid")
        validated = validate_a5d_source_anchored_residual_receipt(self._receipt)
        # Break every reference to the caller-owned receipt tree.  Nested
        # dictionaries remain ordinary JSON containers so strict validators
        # can consume them, but every public read revalidates the self-hash
        # below before returning a fresh copy.
        stored_receipt = json.loads(json.dumps(validated, allow_nan=False))
        object.__setattr__(self, "frozen_compiled_block_states", frozen)
        object.__setattr__(self, "compiled_correction_base_states", base)
        object.__setattr__(
            self, "fragment_id_by_node", MappingProxyType(fragments)
        )
        object.__setattr__(self, "_receipt", MappingProxyType(stored_receipt))
        self._validate_in_memory()

    def _validate_in_memory(self) -> None:
        self.projection.validate_integrity()
        receipt = self._receipt
        validated = validate_a5d_source_anchored_residual_receipt(
            json.loads(json.dumps(dict(receipt), allow_nan=False))
        )
        if validated != dict(receipt):
            raise ValueError("A5d stored receipt canonical form drifted")
        source = receipt["source"]
        rows = receipt["rows"]
        projection = receipt["projection"]
        assert isinstance(source, Mapping)
        assert isinstance(rows, Mapping)
        assert isinstance(projection, Mapping)
        if (
            _tensor_sha256(self.frozen_compiled_block_states)
            != source["frozen_compiled_block_states_sha256"]
            or _tensor_sha256(self.compiled_correction_base_states)
            != source["compiled_correction_base_states_sha256"]
            or _tensor_sha256(self.projection.exact_source_correction)
            != source["exact_source_correction_sha256"]
            or self.projection.metadata() != projection
            or self.residual_rows.row_key_sha256 != rows["row_key_sha256"]
        ):
            raise ValueError("A5d source-anchored target tensors drifted")
        fisher_hashes = rows["fisher_weights_sha256_by_node"]
        contribution_hashes = rows[
            "residual_contribution_sha256_by_node"
        ]
        assert isinstance(fisher_hashes, Mapping)
        assert isinstance(contribution_hashes, Mapping)
        for name in self.projection.node_order:
            fragment = self.fragment_id_by_node[name]
            fragment_rows = self.residual_rows.rows_by_fragment[fragment]
            if (
                _tensor_sha256(fragment_rows.inputs)
                != rows["shared_inputs_sha256"]
                or _tensor_sha256(fragment_rows.fisher_weights)
                != fisher_hashes[name]
                or _tensor_sha256(fragment_rows.contributions)
                != contribution_hashes[name]
            ):
                raise ValueError("A5d residual fit rows drifted")

    def receipt(self) -> dict[str, object]:
        self._validate_in_memory()
        return json.loads(json.dumps(dict(self._receipt), allow_nan=False))

    @property
    def receipt_sha256(self) -> str:
        self._validate_in_memory()
        value = self._receipt["receipt_sha256"]
        return _require_sha256(value, label="A5d receipt")

    @property
    def node_order(self) -> tuple[str, ...]:
        return self.projection.node_order

    @property
    def residual_width(self) -> int:
        return self.projection.residual_width

    def candidate_block_states(self, alpha: float) -> Tensor:
        """Apply a projected residual while preserving the alpha-zero source."""

        self._validate_in_memory()
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a real scalar")
        value = float(alpha)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("alpha must be finite and lie in [0, 1]")
        if value == 0.0:
            # Deliberately avoid arithmetic so the source state is bit-exact.
            return self.frozen_compiled_block_states.clone()
        return (
            self.frozen_compiled_block_states + value * self.projection.prediction
        ).contiguous()


def _receipt_payload(
    *,
    frozen: Tensor,
    base: Tensor,
    oracle_rows: AlignedFragmentRows,
    residual_rows: AlignedFragmentRows,
    projection: A5dZeroMeanJointResidualProjection,
    fragment_id_by_node: Mapping[str, str],
) -> dict[str, object]:
    names = projection.node_order
    oracle_contribution_hashes: dict[str, str] = {}
    fisher_hashes: dict[str, str] = {}
    residual_contribution_hashes: dict[str, str] = {}
    input_hashes: set[str] = set()
    for name in names:
        fragment = fragment_id_by_node[name]
        oracle_fragment = oracle_rows.rows_by_fragment[fragment]
        residual_fragment = residual_rows.rows_by_fragment[fragment]
        oracle_contribution_hashes[name] = _tensor_sha256(
            oracle_fragment.contributions
        )
        fisher_hashes[name] = _tensor_sha256(residual_fragment.fisher_weights)
        residual_contribution_hashes[name] = _tensor_sha256(
            residual_fragment.contributions
        )
        input_hashes.add(_tensor_sha256(residual_fragment.inputs))
    if len(input_hashes) != 1:
        raise RuntimeError("A5d residual rows do not share one input tensor")
    return {
        "schema": GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_SCHEMA,
        "format_version": (
            GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_FORMAT_VERSION
        ),
        "scientific_role": _SCIENTIFIC_ROLE,
        "source": {
            "frozen_compiled_block_states_sha256": _tensor_sha256(frozen),
            "compiled_correction_base_states_sha256": _tensor_sha256(base),
            "exact_source_correction_sha256": _tensor_sha256(
                projection.exact_source_correction
            ),
            "oracle_correction_sha256": _tensor_sha256(
                projection.oracle_correction
            ),
            "oracle_contribution_sha256_by_node": oracle_contribution_hashes,
            "basis_sha256_by_node": dict(projection.basis_sha256_by_node),
            "mean_sha256_by_node": dict(projection.mean_sha256_by_node),
            "decoder_sha256_by_node": dict(projection.decoder_sha256_by_node),
        },
        "construction": {
            "source_correction_formula": _SOURCE_FORMULA,
            "residual_target_formula": _TARGET_FORMULA,
            "residual_node_decode_formula": _NODE_DECODE_FORMULA,
            "projection_uses_affine_means": False,
            "source_state_is_projected": False,
            "source_mapping_preserved_at_alpha_zero": True,
            "node_order": list(names),
            "rank_by_node": list(projection.rank_by_node),
            "joint_coordinate_width": _JOINT_COORDINATE_WIDTH,
            "residual_width": projection.residual_width,
            "observations": projection.observations,
            "sequences": residual_rows.sequences,
        },
        "projection": projection.metadata(),
        "rows": {
            "fragment_id_by_node": dict(fragment_id_by_node),
            "row_key_sha256": residual_rows.row_key_sha256,
            "shared_inputs_sha256": next(iter(input_hashes)),
            "fisher_weights_sha256_by_node": fisher_hashes,
            "residual_contribution_sha256_by_node": (
                residual_contribution_hashes
            ),
            "source_affine_means_injected": False,
            "fit_target_kind": "zero_mean_additive_block_residual",
        },
        "safety": dict(_SAFETY),
    }


def build_a5d_source_anchored_residual_targets(
    *,
    frozen_compiled_block_states: Tensor,
    compiled_correction_base_states: Tensor,
    oracle_rows: AlignedFragmentRows,
    bases_by_node: Mapping[str, ComputationalModeBasis],
    node_order: Sequence[str],
    fragment_id_by_node: Mapping[str, str],
) -> A5dSourceAnchoredResidualTargets:
    """Construct authenticated zero-mean rows relative to the exact source."""

    names, bases = _ordered_bases(bases_by_node, node_order)
    if not isinstance(oracle_rows, AlignedFragmentRows):
        raise TypeError("oracle_rows must be AlignedFragmentRows")
    fragments = dict(fragment_id_by_node)
    if (
        set(fragments) != set(names)
        or len(set(fragments.values())) != _NODE_COUNT
        or tuple(oracle_rows.rows_by_fragment)
        != tuple(fragments[name] for name in names)
    ):
        raise ValueError("A5d oracle fragment catalog is invalid")
    frozen = _canonical_matrix(
        frozen_compiled_block_states, label="frozen compiled block states"
    )
    base = _canonical_matrix(
        compiled_correction_base_states,
        label="compiled correction base states",
    )
    if frozen.shape != base.shape or frozen.shape[0] != oracle_rows.observations:
        raise ValueError("A5d source states and oracle rows are not aligned")
    oracle_contributions: list[Tensor] = []
    shared_inputs: Tensor | None = None
    for name, basis in zip(names, bases, strict=True):
        rows = oracle_rows.rows_by_fragment[fragments[name]]
        if rows.contributions.shape != frozen.shape:
            raise ValueError("A5d oracle contribution width drifted")
        if basis.residual_width != frozen.shape[1]:
            raise ValueError("A5d basis and source-state widths disagree")
        if shared_inputs is None:
            shared_inputs = rows.inputs
        elif not torch.equal(shared_inputs, rows.inputs):
            raise ValueError("A5d oracle nodes do not share compiled inputs")
        oracle_contributions.append(rows.contributions)
    oracle = torch.stack(tuple(oracle_contributions)).sum(dim=0).contiguous()
    exact_source = (frozen - base).contiguous()
    projection = project_source_anchored_residual_to_joint_decoder_span(
        exact_source,
        oracle,
        bases_by_node=bases_by_node,
        node_order=names,
    )
    residual_rows = AlignedFragmentRows(
        rows_by_fragment={
            fragments[name]: LayerFragmentRows(
                inputs=(
                    oracle_rows.rows_by_fragment[fragments[name]].inputs.clone()
                ),
                contributions=projection.contributions_by_node[name].clone(),
                fisher_weights=(
                    oracle_rows.rows_by_fragment[
                        fragments[name]
                    ].fisher_weights.clone()
                ),
                sequences=oracle_rows.sequences,
            )
            for name in names
        },
        row_keys=oracle_rows.row_keys,
    )
    payload = _receipt_payload(
        frozen=frozen,
        base=base,
        oracle_rows=oracle_rows,
        residual_rows=residual_rows,
        projection=projection,
        fragment_id_by_node=fragments,
    )
    payload["receipt_sha256"] = _domain_sha256(_RECEIPT_DOMAIN, payload)
    return A5dSourceAnchoredResidualTargets(
        residual_rows=residual_rows,
        projection=projection,
        frozen_compiled_block_states=frozen,
        compiled_correction_base_states=base,
        fragment_id_by_node=fragments,
        _receipt=payload,
    )


def _validate_error_statistics(value: object, *, label: str) -> None:
    raw = _strict_fields(
        value,
        {"max_abs_error", "rms_error", "target_rms", "nrmse"},
        label=label,
    )
    maximum = _finite(raw["max_abs_error"], label=f"{label} maximum", minimum=0.0)
    rms = _finite(raw["rms_error"], label=f"{label} RMS", minimum=0.0)
    target_rms = _finite(raw["target_rms"], label=f"{label} target RMS", minimum=0.0)
    nrmse = _finite(raw["nrmse"], label=f"{label} NRMSE", minimum=0.0)
    maximum_roundoff = max(
        1.0e-15,
        8.0 * math.ulp(max(maximum, rms)),
    )
    if rms > maximum + maximum_roundoff or not math.isclose(
        nrmse,
        rms / max(target_rms, 1.0e-30),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError(f"{label} scalar diagnostics contradict each other")


def _validate_hash_catalog(
    value: object,
    *,
    names: tuple[str, ...],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{label} catalog is incomplete")
    for name in names:
        _require_sha256(value[name], label=f"{label} {name}")
    return value


def validate_a5d_source_anchored_residual_receipt(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Strictly validate and return one tensor-free A5d target receipt."""

    if not isinstance(receipt, Mapping) or _contains_tensor(receipt):
        raise TypeError("A5d residual receipt must be a tensor-free mapping")
    raw = json.loads(json.dumps(dict(receipt), allow_nan=False))
    top = _strict_fields(
        raw,
        {
            "schema",
            "format_version",
            "scientific_role",
            "source",
            "construction",
            "projection",
            "rows",
            "safety",
            "receipt_sha256",
        },
        label="A5d residual receipt",
    )
    if (
        top["schema"]
        != GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_SCHEMA
        or top["format_version"]
        != GEMMA3_L10_L17_A5D_SOURCE_ANCHORED_RESIDUAL_FORMAT_VERSION
        or top["scientific_role"] != _SCIENTIFIC_ROLE
    ):
        raise ValueError("A5d residual receipt header is invalid")
    construction = _strict_fields(
        top["construction"],
        {
            "source_correction_formula",
            "residual_target_formula",
            "residual_node_decode_formula",
            "projection_uses_affine_means",
            "source_state_is_projected",
            "source_mapping_preserved_at_alpha_zero",
            "node_order",
            "rank_by_node",
            "joint_coordinate_width",
            "residual_width",
            "observations",
            "sequences",
        },
        label="A5d residual construction",
    )
    names = tuple(construction["node_order"])
    ranks = tuple(construction["rank_by_node"])
    if (
        len(names) != _NODE_COUNT
        or len(set(names)) != _NODE_COUNT
        or any(not isinstance(name, str) or not name for name in names)
        or len(ranks) != _NODE_COUNT
        or any(type(rank) is not int or rank <= 0 for rank in ranks)
        or sum(ranks) != _JOINT_COORDINATE_WIDTH
        or construction["joint_coordinate_width"] != _JOINT_COORDINATE_WIDTH
        or type(construction["residual_width"]) is not int
        or construction["residual_width"] <= 0
        or type(construction["observations"]) is not int
        or construction["observations"] <= 0
        or type(construction["sequences"]) is not int
        or construction["sequences"] <= 0
        or construction["source_correction_formula"] != _SOURCE_FORMULA
        or construction["residual_target_formula"] != _TARGET_FORMULA
        or construction["residual_node_decode_formula"] != _NODE_DECODE_FORMULA
        or construction["projection_uses_affine_means"] is not False
        or construction["source_state_is_projected"] is not False
        or construction["source_mapping_preserved_at_alpha_zero"] is not True
    ):
        raise ValueError("A5d residual construction is invalid")

    source = _strict_fields(
        top["source"],
        {
            "frozen_compiled_block_states_sha256",
            "compiled_correction_base_states_sha256",
            "exact_source_correction_sha256",
            "oracle_correction_sha256",
            "oracle_contribution_sha256_by_node",
            "basis_sha256_by_node",
            "mean_sha256_by_node",
            "decoder_sha256_by_node",
        },
        label="A5d residual source",
    )
    for field_name in (
        "frozen_compiled_block_states_sha256",
        "compiled_correction_base_states_sha256",
        "exact_source_correction_sha256",
        "oracle_correction_sha256",
    ):
        _require_sha256(source[field_name], label=f"A5d {field_name}")
    for field_name in (
        "oracle_contribution_sha256_by_node",
        "basis_sha256_by_node",
        "mean_sha256_by_node",
        "decoder_sha256_by_node",
    ):
        _validate_hash_catalog(source[field_name], names=names, label=field_name)

    projection = _strict_fields(
        top["projection"],
        {
            "method",
            "projection_formula",
            "joint_coordinate_width",
            "combined_decoder_matrix_rank",
            "combined_decoder_sha256",
            "residual_target_sha256",
            "joint_coordinates_sha256",
            "projected_residual_sha256",
            "residual_contribution_sha256_by_node",
            "summed_node_contribution_sha256",
            "projection_error",
            "per_node_sum_error",
            "normal_equation_orthogonality",
        },
        label="A5d residual projection",
    )
    if (
        projection["method"] != _PROJECTION_METHOD
        or projection["projection_formula"] != _PROJECTION_FORMULA
        or projection["joint_coordinate_width"] != _JOINT_COORDINATE_WIDTH
        or projection["combined_decoder_matrix_rank"]
        != _JOINT_COORDINATE_WIDTH
    ):
        raise ValueError("A5d residual projection declaration is invalid")
    for field_name in (
        "combined_decoder_sha256",
        "residual_target_sha256",
        "joint_coordinates_sha256",
        "projected_residual_sha256",
        "summed_node_contribution_sha256",
    ):
        _require_sha256(projection[field_name], label=f"A5d {field_name}")
    projected_contributions = _validate_hash_catalog(
        projection["residual_contribution_sha256_by_node"],
        names=names,
        label="A5d projected residual contribution",
    )
    _validate_error_statistics(
        projection["projection_error"], label="A5d projection error"
    )
    _validate_error_statistics(
        projection["per_node_sum_error"], label="A5d node sum error"
    )
    orthogonality = _strict_fields(
        projection["normal_equation_orthogonality"],
        {"max_abs_error", "rms_error", "absolute_tolerance", "passed"},
        label="A5d projection orthogonality",
    )
    orth_max = _finite(
        orthogonality["max_abs_error"],
        label="A5d orthogonality maximum",
        minimum=0.0,
    )
    orth_rms = _finite(
        orthogonality["rms_error"],
        label="A5d orthogonality RMS",
        minimum=0.0,
    )
    orth_tolerance = _finite(
        orthogonality["absolute_tolerance"],
        label="A5d orthogonality tolerance",
        minimum=0.0,
    )
    if (
        orthogonality["passed"] is not True
        or orth_rms
        > orth_max + max(1.0e-15, 8.0 * math.ulp(max(orth_max, orth_rms)))
        or orth_max > orth_tolerance
    ):
        raise ValueError("A5d projection orthogonality audit failed")

    rows = _strict_fields(
        top["rows"],
        {
            "fragment_id_by_node",
            "row_key_sha256",
            "shared_inputs_sha256",
            "fisher_weights_sha256_by_node",
            "residual_contribution_sha256_by_node",
            "source_affine_means_injected",
            "fit_target_kind",
        },
        label="A5d residual rows",
    )
    fragments = rows["fragment_id_by_node"]
    if (
        not isinstance(fragments, Mapping)
        or set(fragments) != set(names)
        or len(set(fragments.values())) != _NODE_COUNT
        or any(not isinstance(value, str) or not value for value in fragments.values())
        or rows["source_affine_means_injected"] is not False
        or rows["fit_target_kind"] != "zero_mean_additive_block_residual"
    ):
        raise ValueError("A5d residual row declaration is invalid")
    _require_sha256(rows["row_key_sha256"], label="A5d row key")
    _require_sha256(rows["shared_inputs_sha256"], label="A5d shared inputs")
    _validate_hash_catalog(
        rows["fisher_weights_sha256_by_node"],
        names=names,
        label="A5d Fisher weights",
    )
    row_contributions = _validate_hash_catalog(
        rows["residual_contribution_sha256_by_node"],
        names=names,
        label="A5d row residual contribution",
    )
    if dict(row_contributions) != dict(projected_contributions):
        raise ValueError("A5d projected and row contribution hashes disagree")
    if top["safety"] != _SAFETY:
        raise ValueError("A5d residual safety declaration is invalid")

    supplied = _require_sha256(top["receipt_sha256"], label="A5d receipt")
    payload = dict(raw)
    del payload["receipt_sha256"]
    if supplied != _domain_sha256(_RECEIPT_DOMAIN, payload):
        raise ValueError("A5d residual receipt hash mismatch")
    return raw
