"""Authenticated pairwise pseudo-unit coordinates for structured MLPs.

Low-influence native units are placed in a pairing pool.  Pair quality is
then judged jointly by activation representability, output representability,
and the Fisher residual produced by the pair's actual encoder and decoder.
Each chosen pair becomes one executable scalar coordinate; unpooled units
remain exact singleton coordinates.

This module builds only the source-bound coordinate plan.  It deliberately
does not construct or optimize an executor.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .structured_layer_distillation import StructuredLayerProvenance
from .structured_mlp_compression import StructuredMLPFisherTaylorBatch


STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_ALGORITHM = (
    "low_fisher_executed_residual_minimax_pair_output_rank1_v2"
)
STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_SCHEMA = (
    "fisher_graph.structured_mlp_pseudo_unit_bundling"
)
STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_FORMAT_VERSION = 3

_SCORE_DOMAIN = b"fisher_graph.structured_mlp.bundle.scores.v1\0"
_ACTIVATION_GRAM_DOMAIN = (
    b"fisher_graph.structured_mlp.bundle.activation_gram.v1\0"
)
_FISHER_GRAM_DOMAIN = (
    b"fisher_graph.structured_mlp.bundle.fisher_gram.v1\0"
)
_CROSS_SQUARE_MOMENT_DOMAIN = (
    b"fisher_graph.structured_mlp.bundle.cross_square_moment.v1\0"
)
_ACTIVATION_SQUARE_GRADIENT_CROSS_MOMENT_DOMAIN = (
    b"fisher_graph.structured_mlp.bundle."
    b"activation_square_gradient_cross_moment.v1\0"
)
_ACTIVATION_CROSS_GRADIENT_SQUARE_MOMENT_DOMAIN = (
    b"fisher_graph.structured_mlp.bundle."
    b"activation_cross_gradient_square_moment.v1\0"
)
_DOWN_GRAM_DOMAIN = b"fisher_graph.structured_mlp.bundle.down_gram.v1\0"
_BATCH_SET_DOMAIN = b"fisher_graph.structured_mlp.bundle.batches.v1\0"
_ENCODER_DOMAIN = b"fisher_graph.structured_mlp.bundle.encoder.v1\0"
_DECODER_DOMAIN = b"fisher_graph.structured_mlp.bundle.decoder.v1\0"
_DOWN_WEIGHT_DOMAIN = (
    b"fisher_graph.structured_mlp.bundle.source_down_weight.v1\0"
)
_PLAN_DOMAIN = b"fisher_graph.structured_mlp.bundle.plan.v3\0"


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _json_sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _tensor_mapping_sha256(
    values: Mapping[str, Tensor],
    *,
    domain: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for name in sorted(values):
        tensor = values[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_fp64(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.ndim != ndim
    ):
        raise ValueError(f"{label} must be a floating rank-{ndim} tensor")
    result = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result.clone()


def _provenance_dict(
    provenance: StructuredLayerProvenance,
) -> dict[str, str]:
    return {
        "layer_id": provenance.layer_id,
        "output_site": provenance.output_site,
        "source_segment_fingerprint": (
            provenance.source_segment_fingerprint
        ),
    }


def _ordered_batches(
    batches: Sequence[StructuredMLPFisherTaylorBatch],
) -> tuple[StructuredMLPFisherTaylorBatch, ...]:
    values = tuple(batches)
    if not values:
        raise ValueError("Fisher/Taylor batches cannot be empty")
    if any(
        not isinstance(batch, StructuredMLPFisherTaylorBatch)
        for batch in values
    ):
        raise TypeError(
            "batches must contain StructuredMLPFisherTaylorBatch values"
        )
    result = tuple(sorted(values, key=lambda batch: batch.batch_id))
    batch_ids = tuple(batch.batch_id for batch in result)
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("Fisher/Taylor batch ids must be unique")
    return result


def _batch_records(
    batches: Sequence[StructuredMLPFisherTaylorBatch],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "batch_id": batch.batch_id,
            "valid_rows": batch.valid_rows,
            "input_sha256": batch.input_sha256(),
        }
        for batch in _ordered_batches(batches)
    )


def structured_mlp_fisher_batch_set_sha256(
    batches: Sequence[StructuredMLPFisherTaylorBatch],
) -> str:
    """Hash valid activation/gradient rows in stable batch-id order."""

    return _json_sha256(
        _batch_records(batches),
        domain=_BATCH_SET_DOMAIN,
    )


def structured_mlp_source_down_weight_sha256(
    source_down_weight: Tensor,
) -> str:
    """Hash a down weight after canonical CPU-FP64 conversion."""

    value = _canonical_fp64(
        source_down_weight,
        label="source_down_weight",
        ndim=2,
    )
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("source_down_weight dimensions must be positive")
    return _tensor_mapping_sha256(
        {"source_down_weight": value},
        domain=_DOWN_WEIGHT_DOMAIN,
    )


def _symmetric_eigenvalues(block: Tensor) -> tuple[float, float]:
    a = float(block[0, 0].item())
    b = float(block[0, 1].item())
    c = float(block[1, 1].item())
    center = 0.5 * (a + c)
    radius = 0.5 * math.hypot(a - c, 2.0 * b)
    lower = center - radius
    upper = center + radius
    tolerance = (
        torch.finfo(torch.float64).eps
        * max(abs(a), abs(b), abs(c), 1.0)
        * 128.0
    )
    if lower < -tolerance:
        raise ValueError("two-by-two moment block is not positive semidefinite")
    return max(lower, 0.0), max(upper, 0.0)


def _canonical_top_mode(block: Tensor) -> Tensor:
    """Return the canonical-sign leading eigenvector of a PSD 2x2 block."""

    a = float(block[0, 0].item())
    b = float(block[0, 1].item())
    c = float(block[1, 1].item())
    lower, upper = _symmetric_eigenvalues(block)
    tolerance = (
        torch.finfo(torch.float64).eps
        * max(abs(lower), abs(upper), 1.0)
        * 128.0
    )
    if upper - lower <= tolerance:
        result = torch.tensor([1.0, 0.0], dtype=torch.float64)
    elif b == 0.0:
        result = (
            torch.tensor([1.0, 0.0], dtype=torch.float64)
            if a >= c
            else torch.tensor([0.0, 1.0], dtype=torch.float64)
        )
    else:
        theta = 0.5 * math.atan2(2.0 * b, a - c)
        result = torch.tensor(
            [math.cos(theta), math.sin(theta)],
            dtype=torch.float64,
        )
        result /= torch.linalg.vector_norm(result)
    pivot = int(result.abs().argmax().item())
    if float(result[pivot].item()) < 0.0:
        result.neg_()
    return result.contiguous()


def _psd_sqrt_and_pseudoinverse_sqrt(
    block: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute deterministic matrix functions; zero null directions."""

    symmetric = ((block + block.T) * 0.5).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    scale = max(float(eigenvalues.abs().max().item()), 1.0)
    tolerance = torch.finfo(torch.float64).eps * scale * 128.0
    if float(eigenvalues.min().item()) < -tolerance:
        raise ValueError("activation Gram block is not positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0.0)
    square_roots = eigenvalues.sqrt()
    inverse_square_roots = torch.where(
        eigenvalues > tolerance,
        square_roots.reciprocal(),
        torch.zeros_like(square_roots),
    )
    square_root = (
        eigenvectors @ torch.diag(square_roots) @ eigenvectors.T
    )
    pseudoinverse_square_root = (
        eigenvectors
        @ torch.diag(inverse_square_roots)
        @ eigenvectors.T
    )
    return (
        ((square_root + square_root.T) * 0.5).contiguous(),
        (
            (pseudoinverse_square_root + pseudoinverse_square_root.T)
            * 0.5
        ).contiguous(),
    )


def _output_eigenvalues(
    activation_block: Tensor,
    down_block: Tensor,
) -> tuple[float, float]:
    trace = float(
        torch.sum(activation_block * down_block).item()
    )
    activation_lower, activation_upper = _symmetric_eigenvalues(
        activation_block
    )
    down_lower, down_upper = _symmetric_eigenvalues(down_block)
    determinant = (
        activation_lower
        * activation_upper
        * down_lower
        * down_upper
    )
    scale = max(abs(trace), abs(determinant) ** 0.5, 1.0)
    tolerance = torch.finfo(torch.float64).eps * scale * 256.0
    if trace < -tolerance or determinant < -tolerance:
        raise ValueError("output moment block is not positive semidefinite")
    trace = max(trace, 0.0)
    determinant = max(determinant, 0.0)
    discriminant = trace * trace - 4.0 * determinant
    if discriminant < -tolerance * scale:
        raise ValueError("output moment eigenvalue discriminant is invalid")
    root = math.sqrt(max(discriminant, 0.0))
    return (
        max(0.5 * (trace - root), 0.0),
        max(0.5 * (trace + root), 0.0),
    )


def _output_rank_one_coordinates(
    activation_block: Tensor,
    down_block: Tensor,
    fisher_block: Tensor,
) -> tuple[tuple[float, float], tuple[float, float]]:
    square_root, pseudoinverse_square_root = (
        _psd_sqrt_and_pseudoinverse_sqrt(activation_block)
    )
    output_operator = square_root @ down_block @ square_root
    output_operator = (
        (output_operator + output_operator.T) * 0.5
    ).contiguous()
    _, output_upper = _symmetric_eigenvalues(output_operator)
    scale = max(output_upper, 1.0)
    if output_upper <= torch.finfo(torch.float64).eps * scale * 128.0:
        mode = _canonical_top_mode(fisher_block)
        loading = mode
    else:
        mode = _canonical_top_mode(output_operator)
        loading = pseudoinverse_square_root @ mode
        norm = torch.linalg.vector_norm(loading)
        tolerance = torch.finfo(torch.float64).eps * 128.0
        if float(norm.item()) <= tolerance:
            loading = _canonical_top_mode(fisher_block)
        else:
            loading /= norm
    pivot = int(loading.abs().argmax().item())
    if float(loading[pivot].item()) < 0.0:
        loading.neg_()

    denominator = float(
        (loading @ activation_block @ loading).item()
    )
    tolerance = (
        torch.finfo(torch.float64).eps
        * max(
            float(activation_block.abs().max().item()),
            1.0,
        )
        * 128.0
    )
    if denominator <= tolerance:
        reconstruction = loading.clone()
    else:
        reconstruction = activation_block @ loading / denominator
    return (
        (float(loading[0].item()), float(loading[1].item())),
        (
            float(reconstruction[0].item()),
            float(reconstruction[1].item()),
        ),
    )


@dataclass(frozen=True, slots=True)
class StructuredMLPPseudoUnitPair:
    """One selected edge and its executed rank-one Fisher accounting."""

    source_indices: tuple[int, int]
    executed_fisher_residual_fraction: float
    executed_fisher_residual_energy: float
    native_fisher_scalar_energy: float
    fisher_normalization_energy: float
    fisher_contribution_vector_rank_one_tail_fraction: float
    fisher_contribution_vector_rank_one_tail_energy: float
    output_loss_fraction: float
    generator_loss_proxy: float
    output_retained_energy: float
    output_dropped_energy: float
    loadings: tuple[float, float]
    reconstruction_loadings: tuple[float, float]

    def __post_init__(self) -> None:
        contribution_tail_fraction = (
            self.fisher_contribution_vector_rank_one_tail_fraction
        )
        if (
            type(self.source_indices) is not tuple
            or len(self.source_indices) != 2
            or any(type(index) is not int for index in self.source_indices)
            or not 0 <= self.source_indices[0] < self.source_indices[1]
        ):
            raise ValueError(
                "pair source_indices must be two ascending nonnegative "
                "integers"
            )
        values = (
            self.executed_fisher_residual_fraction,
            self.executed_fisher_residual_energy,
            self.native_fisher_scalar_energy,
            self.fisher_normalization_energy,
            contribution_tail_fraction,
            self.fisher_contribution_vector_rank_one_tail_energy,
            self.output_loss_fraction,
            self.generator_loss_proxy,
            self.output_retained_energy,
            self.output_dropped_energy,
        )
        if (
            any(type(value) is not float for value in values)
            or any(not math.isfinite(value) for value in values)
            or self.executed_fisher_residual_fraction < 0.0
            or self.executed_fisher_residual_energy < 0.0
            or self.native_fisher_scalar_energy < 0.0
            or self.fisher_normalization_energy <= 0.0
            or (
                self.fisher_normalization_energy + 1e-15
                < self.native_fisher_scalar_energy
            )
            or not (
                0.0
                <= contribution_tail_fraction
                <= 0.5 + 1e-12
            )
            or self.fisher_contribution_vector_rank_one_tail_energy
            < 0.0
            or not 0.0 <= self.output_loss_fraction <= 0.5 + 1e-12
            or not 0.0 <= self.generator_loss_proxy <= 1.0 + 1e-12
            or self.output_retained_energy < 0.0
            or self.output_dropped_energy < 0.0
            or self.output_retained_energy < self.output_dropped_energy
        ):
            raise ValueError("pair loss or energy metadata is invalid")
        self._validate_loading(
            self.loadings,
            label="pair loadings",
            require_unit=True,
            require_canonical=True,
        )
        self._validate_loading(
            self.reconstruction_loadings,
            label="pair reconstruction_loadings",
            require_unit=False,
            require_canonical=False,
        )

    @staticmethod
    def _validate_loading(
        values: tuple[float, float],
        *,
        label: str,
        require_unit: bool,
        require_canonical: bool,
    ) -> None:
        if (
            type(values) is not tuple
            or len(values) != 2
            or any(type(value) is not float for value in values)
            or any(not math.isfinite(value) for value in values)
        ):
            raise ValueError(f"{label} must contain two finite floats")
        tensor = torch.tensor(values, dtype=torch.float64)
        if require_unit and not torch.allclose(
            torch.dot(tensor, tensor),
            torch.ones((), dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{label} must have unit Euclidean norm")
        if require_canonical:
            pivot = int(tensor.abs().argmax().item())
            if float(tensor[pivot].item()) < 0.0:
                raise ValueError(f"{label} must use a canonical sign")

    @property
    def priority_max_loss(self) -> float:
        return max(
            self.executed_fisher_residual_fraction,
            2.0 * self.output_loss_fraction,
            self.generator_loss_proxy,
        )

    @property
    def priority_sum_loss(self) -> float:
        return (
            self.executed_fisher_residual_fraction
            + 2.0 * self.output_loss_fraction
            + self.generator_loss_proxy
        )

    @property
    def fisher_loss_fraction(self) -> float:
        """Compatibility alias for the executed residual fraction."""

        return self.executed_fisher_residual_fraction

    def metadata(self) -> dict[str, object]:
        return {
            "source_indices": self.source_indices,
            "f": self.executed_fisher_residual_fraction,
            "o": self.output_loss_fraction,
            "g": self.generator_loss_proxy,
            "priority_max_loss": self.priority_max_loss,
            "priority_sum_loss": self.priority_sum_loss,
            "executed_fisher_residual_fraction": (
                self.executed_fisher_residual_fraction
            ),
            "executed_fisher_residual_energy": (
                self.executed_fisher_residual_energy
            ),
            "native_fisher_scalar_energy": (
                self.native_fisher_scalar_energy
            ),
            "fisher_normalization_energy": (
                self.fisher_normalization_energy
            ),
            "fisher_contribution_vector_rank_one_tail_fraction": (
                self.fisher_contribution_vector_rank_one_tail_fraction
            ),
            "fisher_contribution_vector_rank_one_tail_energy": (
                self.fisher_contribution_vector_rank_one_tail_energy
            ),
            "output_retained_energy": self.output_retained_energy,
            "output_dropped_energy": self.output_dropped_energy,
            "loadings": self.loadings,
            "reconstruction_loadings": (
                self.reconstruction_loadings
            ),
        }


def _fisher_normalization_floor(fisher_gram: Tensor) -> float:
    """Return one scale-aware floor shared by every possible pool edge."""

    diagonal = torch.diagonal(fisher_gram)
    native_energy = (
        diagonal[:, None]
        + diagonal[None, :]
        + 2.0 * fisher_gram
    )
    upper = native_energy[
        torch.triu_indices(
            fisher_gram.shape[0],
            fisher_gram.shape[1],
            offset=1,
        ).unbind()
    ]
    scale = max(
        float(fisher_gram.abs().max().item()),
        float(torch.finfo(torch.float64).tiny),
    )
    tolerance = torch.finfo(torch.float64).eps * scale * 512.0
    if float(upper.min().item()) < -tolerance:
        raise ValueError("native pair Fisher scalar energy is negative")
    mean_native = float(upper.clamp_min(0.0).mean().item())
    return max(
        mean_native * 1e-6,
        scale * torch.finfo(torch.float64).eps * 1024.0,
        float(torch.finfo(torch.float64).tiny),
    )


def _executed_fisher_residual(
    *,
    fisher_gram: Tensor,
    cross_square_moment: Tensor,
    activation_square_gradient_cross_moment: Tensor,
    activation_cross_gradient_square_moment: Tensor,
    first_position: int,
    second_position: int,
    loadings: tuple[float, float],
    reconstruction: tuple[float, float],
    normalization_floor: float,
) -> tuple[float, float, float, float]:
    """Evaluate ``E[(g.T @ (z - r * (a.T @ z)))**2]`` for one edge."""

    i = first_position
    j = second_position
    ai, aj = loadings
    ri, rj = reconstruction
    # The executed scalar residual is a fixed linear combination of
    # [z_i g_i, z_j g_j, z_i g_j, z_j g_i].  K supplies the native/native
    # entries; A, U, and V below supply the cross-gradient entries.
    coefficients = (
        1.0 - ai * ri,
        1.0 - aj * rj,
        -ai * rj,
        -aj * ri,
    )
    c0, c1, c2, c3 = coefficients
    kii = float(fisher_gram[i, i].item())
    kjj = float(fisher_gram[j, j].item())
    kij = float(fisher_gram[i, j].item())
    aij = float(cross_square_moment[i, j].item())
    aji = float(cross_square_moment[j, i].item())
    uij = float(
        activation_square_gradient_cross_moment[i, j].item()
    )
    uji = float(
        activation_square_gradient_cross_moment[j, i].item()
    )
    vij = float(
        activation_cross_gradient_square_moment[i, j].item()
    )
    vji = float(
        activation_cross_gradient_square_moment[j, i].item()
    )
    residual_terms = (
        c0 * c0 * kii,
        c1 * c1 * kjj,
        c2 * c2 * aij,
        c3 * c3 * aji,
        2.0 * c0 * c1 * kij,
        2.0 * c0 * c2 * uij,
        2.0 * c0 * c3 * vij,
        2.0 * c1 * c2 * vji,
        2.0 * c1 * c3 * uji,
        2.0 * c2 * c3 * kij,
    )
    residual_energy = math.fsum(residual_terms)
    residual_scale = max(
        math.fsum(abs(term) for term in residual_terms),
        float(torch.finfo(torch.float64).tiny),
    )
    tolerance = (
        torch.finfo(torch.float64).eps * residual_scale * 1024.0
    )
    if residual_energy < -tolerance:
        raise ValueError("executed pair Fisher residual energy is negative")
    residual_energy = max(residual_energy, 0.0)

    native_terms = (kii, kjj, 2.0 * kij)
    native_energy = math.fsum(native_terms)
    native_scale = max(
        math.fsum(abs(term) for term in native_terms),
        float(torch.finfo(torch.float64).tiny),
    )
    tolerance = torch.finfo(torch.float64).eps * native_scale * 512.0
    if native_energy < -tolerance:
        raise ValueError("native pair Fisher scalar energy is negative")
    native_energy = max(native_energy, 0.0)
    normalization_energy = max(native_energy, normalization_floor)
    return (
        float(residual_energy / normalization_energy),
        float(residual_energy),
        float(native_energy),
        float(normalization_energy),
    )


def _pair_record(
    *,
    pool_indices: tuple[int, ...],
    activation_gram: Tensor,
    fisher_gram: Tensor,
    cross_square_moment: Tensor,
    activation_square_gradient_cross_moment: Tensor,
    activation_cross_gradient_square_moment: Tensor,
    down_gram: Tensor,
    first_position: int,
    second_position: int,
    fisher_normalization_floor: float,
) -> StructuredMLPPseudoUnitPair:
    positions = torch.tensor(
        (first_position, second_position),
        dtype=torch.long,
    )
    activation_block = activation_gram[positions][:, positions]
    fisher_block = fisher_gram[positions][:, positions]
    down_block = down_gram[positions][:, positions]
    (
        fisher_contribution_tail_energy,
        fisher_contribution_head_energy,
    ) = _symmetric_eigenvalues(
        fisher_block
    )
    fisher_trace = (
        fisher_contribution_tail_energy
        + fisher_contribution_head_energy
    )
    fisher_contribution_tail_fraction = (
        fisher_contribution_tail_energy / fisher_trace
        if fisher_trace > 0.0
        else 0.0
    )
    output_dropped, output_retained = _output_eigenvalues(
        activation_block,
        down_block,
    )
    output_trace = output_dropped + output_retained
    output_fraction = (
        output_dropped / output_trace if output_trace > 0.0 else 0.0
    )
    activation_product = float(
        (
            activation_block[0, 0] * activation_block[1, 1]
        ).item()
    )
    if activation_product <= 0.0:
        generator_loss = 0.0
    else:
        correlation_square = float(
            activation_block[0, 1].square().item()
            / activation_product
        )
        generator_loss = min(max(1.0 - correlation_square, 0.0), 1.0)
    loadings, reconstruction = _output_rank_one_coordinates(
        activation_block,
        down_block,
        fisher_block,
    )
    (
        executed_fisher_fraction,
        executed_fisher_energy,
        native_fisher_energy,
        fisher_normalization_energy,
    ) = _executed_fisher_residual(
        fisher_gram=fisher_gram,
        cross_square_moment=cross_square_moment,
        activation_square_gradient_cross_moment=(
            activation_square_gradient_cross_moment
        ),
        activation_cross_gradient_square_moment=(
            activation_cross_gradient_square_moment
        ),
        first_position=first_position,
        second_position=second_position,
        loadings=loadings,
        reconstruction=reconstruction,
        normalization_floor=fisher_normalization_floor,
    )
    return StructuredMLPPseudoUnitPair(
        source_indices=(
            pool_indices[first_position],
            pool_indices[second_position],
        ),
        executed_fisher_residual_fraction=executed_fisher_fraction,
        executed_fisher_residual_energy=executed_fisher_energy,
        native_fisher_scalar_energy=native_fisher_energy,
        fisher_normalization_energy=fisher_normalization_energy,
        fisher_contribution_vector_rank_one_tail_fraction=float(
            fisher_contribution_tail_fraction
        ),
        fisher_contribution_vector_rank_one_tail_energy=float(
            fisher_contribution_tail_energy
        ),
        output_loss_fraction=float(output_fraction),
        generator_loss_proxy=float(generator_loss),
        output_retained_energy=float(output_retained),
        output_dropped_energy=float(output_dropped),
        loadings=loadings,
        reconstruction_loadings=reconstruction,
    )


def _match_pool_units(
    pool_indices: tuple[int, ...],
    activation_gram: Tensor,
    fisher_gram: Tensor,
    cross_square_moment: Tensor,
    activation_square_gradient_cross_moment: Tensor,
    activation_cross_gradient_square_moment: Tensor,
    down_gram: Tensor,
) -> tuple[StructuredMLPPseudoUnitPair, ...]:
    fisher_normalization_floor = _fisher_normalization_floor(
        fisher_gram
    )
    edges: list[
        tuple[
            float,
            float,
            int,
            int,
            int,
            int,
            StructuredMLPPseudoUnitPair,
        ]
    ] = []
    for first_position, first_source in enumerate(pool_indices):
        for second_position in range(
            first_position + 1,
            len(pool_indices),
        ):
            pair = _pair_record(
                pool_indices=pool_indices,
                activation_gram=activation_gram,
                fisher_gram=fisher_gram,
                cross_square_moment=cross_square_moment,
                activation_square_gradient_cross_moment=(
                    activation_square_gradient_cross_moment
                ),
                activation_cross_gradient_square_moment=(
                    activation_cross_gradient_square_moment
                ),
                down_gram=down_gram,
                first_position=first_position,
                second_position=second_position,
                fisher_normalization_floor=fisher_normalization_floor,
            )
            edges.append(
                (
                    pair.priority_max_loss,
                    pair.priority_sum_loss,
                    first_source,
                    pool_indices[second_position],
                    first_position,
                    second_position,
                    pair,
                )
            )
    edges.sort(key=lambda edge: edge[:-1])
    matched: set[int] = set()
    pairs: list[StructuredMLPPseudoUnitPair] = []
    for edge in edges:
        first_position = edge[4]
        second_position = edge[5]
        if (
            first_position in matched
            or second_position in matched
        ):
            continue
        matched.add(first_position)
        matched.add(second_position)
        pairs.append(edge[6])
    if len(matched) != len(pool_indices):
        raise RuntimeError("could not construct a perfect pool matching")
    return tuple(sorted(pairs, key=lambda pair: pair.source_indices))


def _dense_encoder_decoder(
    *,
    source_width: int,
    singleton_indices: tuple[int, ...],
    pairs: tuple[StructuredMLPPseudoUnitPair, ...],
) -> tuple[Tensor, Tensor]:
    retained_width = len(singleton_indices) + len(pairs)
    encoder = torch.zeros(
        source_width,
        retained_width,
        dtype=torch.float64,
    )
    decoder = torch.zeros_like(encoder)
    for coordinate, source_index in enumerate(singleton_indices):
        encoder[source_index, coordinate] = 1.0
        decoder[source_index, coordinate] = 1.0
    offset = len(singleton_indices)
    for pair_index, pair in enumerate(pairs):
        coordinate = offset + pair_index
        first, second = pair.source_indices
        encoder[first, coordinate] = pair.loadings[0]
        encoder[second, coordinate] = pair.loadings[1]
        decoder[first, coordinate] = pair.reconstruction_loadings[0]
        decoder[second, coordinate] = pair.reconstruction_loadings[1]
    return encoder, decoder


def _plan_payload(
    *,
    provenance: StructuredLayerProvenance,
    calibration_split_sha256: str,
    activation_site: str,
    parent_executor_fingerprint: str,
    source_width: int,
    retained_width: int,
    output_width: int,
    valid_rows: int,
    batch_ids: tuple[str, ...],
    input_batches_sha256: str,
    source_down_weight_sha256: str,
    diagonal_scores_sha256: str,
    pool_indices: tuple[int, ...],
    pool_activation_gram_sha256: str,
    pool_fisher_gram_sha256: str,
    pool_cross_square_moment_sha256: str,
    pool_activation_square_gradient_cross_moment_sha256: str,
    pool_activation_cross_gradient_square_moment_sha256: str,
    pool_down_gram_sha256: str,
    singleton_indices: tuple[int, ...],
    pairs: tuple[StructuredMLPPseudoUnitPair, ...],
    dense_loadings_sha256: str,
    dense_reconstruction_loadings_sha256: str,
) -> dict[str, object]:
    pair_count = source_width - retained_width
    return {
        "schema": STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_SCHEMA,
        "format_version": (
            STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_FORMAT_VERSION
        ),
        "algorithm": STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_ALGORITHM,
        "provenance": _provenance_dict(provenance),
        "calibration_split_sha256": calibration_split_sha256,
        "activation_site": activation_site,
        "parent_executor_fingerprint": parent_executor_fingerprint,
        "source_width": source_width,
        "retained_width": retained_width,
        "output_width": output_width,
        "pair_count": pair_count,
        "pool_size": 2 * pair_count,
        "valid_rows": valid_rows,
        "batch_ids": batch_ids,
        "input_batches_sha256": input_batches_sha256,
        "source_down_weight_sha256": source_down_weight_sha256,
        "diagonal_scores_sha256": diagonal_scores_sha256,
        "pool_indices": pool_indices,
        "pool_activation_gram_sha256": pool_activation_gram_sha256,
        "pool_fisher_gram_sha256": pool_fisher_gram_sha256,
        "pool_cross_square_moment_sha256": (
            pool_cross_square_moment_sha256
        ),
        "pool_activation_square_gradient_cross_moment_sha256": (
            pool_activation_square_gradient_cross_moment_sha256
        ),
        "pool_activation_cross_gradient_square_moment_sha256": (
            pool_activation_cross_gradient_square_moment_sha256
        ),
        "pool_down_gram_sha256": pool_down_gram_sha256,
        "singleton_indices": singleton_indices,
        "pairs": tuple(pair.metadata() for pair in pairs),
        "dense_loadings_sha256": dense_loadings_sha256,
        "dense_reconstruction_loadings_sha256": (
            dense_reconstruction_loadings_sha256
        ),
        "score_semantics": "mean_valid_row_squared_z_times_score_gradient",
        "activation_moment_semantics": "mean_valid_row_z_transpose_z",
        "fisher_moment_semantics": (
            "mean_valid_row_z_times_gradient_outer_product"
        ),
        "cross_square_moment_semantics": (
            "entry_i_j_mean_valid_row_z_i_squared_times_g_j_squared"
        ),
        "activation_square_gradient_cross_moment_semantics": (
            "entry_i_j_mean_valid_row_z_i_squared_times_g_i_times_g_j"
        ),
        "activation_cross_gradient_square_moment_semantics": (
            "entry_i_j_mean_valid_row_z_i_times_z_j_times_g_i_squared"
        ),
        "down_moment_semantics": "source_down_columns_transpose_product",
        "pool_selection": "lowest_diagonal_fisher_stable_source_index",
        "edge_losses": {
            "f": (
                "executed_first_order_fisher_scalar_residual_energy_"
                "over_stable_native_pair_energy"
            ),
            "o": "lambda_min_C_half_B_C_half_over_trace",
            "g": "one_minus_squared_activation_correlation",
        },
        "edge_references": {
            "fisher_contribution_vector_rank_one_tail": {
                "energy": (
                    "lambda_min_K_exact_rank_one_tail_of_h_equals_z_times_g"
                ),
                "fraction": "lambda_min_K_over_trace_K",
                "relation_to_executed_fisher_residual": (
                    "reference_only_not_a_bound"
                ),
            },
        },
        "matching": (
            "ascending_greedy_max_f_2o_g_then_sum_then_source_pair"
        ),
        "coordinate_order": (
            "ascending_singletons_then_lexicographic_pairs"
        ),
        "pair_encoder": "normalized_C_pseudoinverse_half_top_output_mode",
        "pair_decoder": "C_a_over_a_transpose_C_a",
    }


@dataclass(frozen=True, slots=True)
class StructuredMLPPseudoUnitBundlingPlan:
    """Authenticated moment-based pairing and executable coordinates."""

    provenance: StructuredLayerProvenance
    calibration_split_sha256: str
    activation_site: str
    parent_executor_fingerprint: str
    source_width: int
    retained_width: int
    output_width: int
    valid_rows: int
    batch_ids: tuple[str, ...]
    input_batches_sha256: str
    source_down_weight_sha256: str
    diagonal_scores: Tensor
    diagonal_scores_sha256: str
    pool_indices: tuple[int, ...]
    pool_activation_gram: Tensor
    pool_activation_gram_sha256: str
    pool_fisher_gram: Tensor
    pool_fisher_gram_sha256: str
    pool_cross_square_moment: Tensor
    pool_cross_square_moment_sha256: str
    pool_activation_square_gradient_cross_moment: Tensor
    pool_activation_square_gradient_cross_moment_sha256: str
    pool_activation_cross_gradient_square_moment: Tensor
    pool_activation_cross_gradient_square_moment_sha256: str
    pool_down_gram: Tensor
    pool_down_gram_sha256: str
    singleton_indices: tuple[int, ...]
    pairs: tuple[StructuredMLPPseudoUnitPair, ...]
    dense_loadings: Tensor
    dense_loadings_sha256: str
    dense_reconstruction_loadings: Tensor
    dense_reconstruction_loadings_sha256: str
    plan_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, StructuredLayerProvenance):
            raise TypeError("provenance must be StructuredLayerProvenance")
        for label, value in (
            ("calibration_split_sha256", self.calibration_split_sha256),
            (
                "parent_executor_fingerprint",
                self.parent_executor_fingerprint,
            ),
            ("input_batches_sha256", self.input_batches_sha256),
            (
                "source_down_weight_sha256",
                self.source_down_weight_sha256,
            ),
            ("diagonal_scores_sha256", self.diagonal_scores_sha256),
            (
                "pool_activation_gram_sha256",
                self.pool_activation_gram_sha256,
            ),
            (
                "pool_fisher_gram_sha256",
                self.pool_fisher_gram_sha256,
            ),
            (
                "pool_cross_square_moment_sha256",
                self.pool_cross_square_moment_sha256,
            ),
            (
                "pool_activation_square_gradient_cross_moment_sha256",
                (
                    self.pool_activation_square_gradient_cross_moment_sha256
                ),
            ),
            (
                "pool_activation_cross_gradient_square_moment_sha256",
                (
                    self.pool_activation_cross_gradient_square_moment_sha256
                ),
            ),
            ("pool_down_gram_sha256", self.pool_down_gram_sha256),
            ("dense_loadings_sha256", self.dense_loadings_sha256),
            (
                "dense_reconstruction_loadings_sha256",
                self.dense_reconstruction_loadings_sha256,
            ),
            ("plan_sha256", self.plan_sha256),
        ):
            _require_sha256(value, label=label)
        if not isinstance(self.activation_site, str) or not self.activation_site:
            raise ValueError("activation_site must be nonempty")
        if (
            type(self.source_width) is not int
            or type(self.retained_width) is not int
            or not 0 < self.retained_width < self.source_width
            or 2 * (self.source_width - self.retained_width)
            > self.source_width
            or type(self.output_width) is not int
            or self.output_width <= 0
            or type(self.valid_rows) is not int
            or self.valid_rows <= 0
            or type(self.batch_ids) is not tuple
            or not self.batch_ids
            or tuple(sorted(self.batch_ids)) != self.batch_ids
            or len(set(self.batch_ids)) != len(self.batch_ids)
            or any(
                not isinstance(batch_id, str) or not batch_id
                for batch_id in self.batch_ids
            )
        ):
            raise ValueError("pseudo-unit plan scalar metadata is invalid")

        scores = self._tensor(
            self.diagonal_scores,
            label="diagonal_scores",
            shape=(self.source_width,),
        )
        object.__setattr__(self, "diagonal_scores", scores)
        if (
            bool((scores < 0.0).any())
            or float(scores.sum().item()) <= 0.0
            or self._tensor_sha(
                "diagonal_scores",
                scores,
                _SCORE_DOMAIN,
            )
            != self.diagonal_scores_sha256
        ):
            raise ValueError("diagonal scores or digest are invalid")
        ranked_low = tuple(
            int(index)
            for index in torch.argsort(
                scores,
                descending=False,
                stable=True,
            ).tolist()
        )
        expected_pool = tuple(sorted(ranked_low[: self.pool_size]))
        if self.pool_indices != expected_pool:
            raise ValueError("low-influence pool indices are invalid")

        moment_shape = (self.pool_size, self.pool_size)
        activation_gram = self._moment(
            self.pool_activation_gram,
            label="pool_activation_gram",
            shape=moment_shape,
            digest=self.pool_activation_gram_sha256,
            domain=_ACTIVATION_GRAM_DOMAIN,
        )
        fisher_gram = self._moment(
            self.pool_fisher_gram,
            label="pool_fisher_gram",
            shape=moment_shape,
            digest=self.pool_fisher_gram_sha256,
            domain=_FISHER_GRAM_DOMAIN,
        )
        down_gram = self._moment(
            self.pool_down_gram,
            label="pool_down_gram",
            shape=moment_shape,
            digest=self.pool_down_gram_sha256,
            domain=_DOWN_GRAM_DOMAIN,
        )
        cross_square_moment = self._cross_moment(
            self.pool_cross_square_moment,
            label="pool_cross_square_moment",
            shape=moment_shape,
            digest=self.pool_cross_square_moment_sha256,
            domain=_CROSS_SQUARE_MOMENT_DOMAIN,
        )
        activation_square_gradient_cross_moment = self._cross_moment(
            self.pool_activation_square_gradient_cross_moment,
            label="pool_activation_square_gradient_cross_moment",
            shape=moment_shape,
            digest=(
                self.pool_activation_square_gradient_cross_moment_sha256
            ),
            domain=_ACTIVATION_SQUARE_GRADIENT_CROSS_MOMENT_DOMAIN,
        )
        activation_cross_gradient_square_moment = self._cross_moment(
            self.pool_activation_cross_gradient_square_moment,
            label="pool_activation_cross_gradient_square_moment",
            shape=moment_shape,
            digest=(
                self.pool_activation_cross_gradient_square_moment_sha256
            ),
            domain=_ACTIVATION_CROSS_GRADIENT_SQUARE_MOMENT_DOMAIN,
        )
        object.__setattr__(
            self,
            "pool_activation_gram",
            activation_gram,
        )
        object.__setattr__(self, "pool_fisher_gram", fisher_gram)
        object.__setattr__(
            self,
            "pool_cross_square_moment",
            cross_square_moment,
        )
        object.__setattr__(
            self,
            "pool_activation_square_gradient_cross_moment",
            activation_square_gradient_cross_moment,
        )
        object.__setattr__(
            self,
            "pool_activation_cross_gradient_square_moment",
            activation_cross_gradient_square_moment,
        )
        object.__setattr__(self, "pool_down_gram", down_gram)
        pool_tensor = torch.tensor(self.pool_indices, dtype=torch.long)
        if not torch.allclose(
            torch.diagonal(fisher_gram),
            scores[pool_tensor],
            rtol=1e-11,
            atol=1e-14,
        ):
            raise ValueError(
                "pool Fisher diagonal does not match diagonal scores"
            )
        fisher_diagonal = torch.diagonal(fisher_gram)
        if (
            bool((cross_square_moment < 0.0).any())
            or not torch.allclose(
                torch.diagonal(cross_square_moment),
                fisher_diagonal,
                rtol=1e-11,
                atol=1e-14,
            )
            or not torch.allclose(
                torch.diagonal(
                    activation_square_gradient_cross_moment
                ),
                fisher_diagonal,
                rtol=1e-11,
                atol=1e-14,
            )
            or not torch.allclose(
                torch.diagonal(
                    activation_cross_gradient_square_moment
                ),
                fisher_diagonal,
                rtol=1e-11,
                atol=1e-14,
            )
        ):
            raise ValueError("pool cross-moment diagonals are invalid")
        activation_square_limits = (
            fisher_diagonal[:, None] * cross_square_moment
        )
        activation_cross_limits = (
            fisher_diagonal[:, None] * cross_square_moment.T
        )
        activation_square_values = (
            activation_square_gradient_cross_moment.square()
        )
        activation_cross_values = (
            activation_cross_gradient_square_moment.square()
        )
        activation_square_tolerance = (
            torch.finfo(torch.float64).eps
            * torch.maximum(
                torch.maximum(
                    activation_square_limits,
                    activation_square_values,
                ),
                torch.ones_like(activation_square_limits),
            )
            * 512.0
        )
        activation_cross_tolerance = (
            torch.finfo(torch.float64).eps
            * torch.maximum(
                torch.maximum(
                    activation_cross_limits,
                    activation_cross_values,
                ),
                torch.ones_like(activation_cross_limits),
            )
            * 512.0
        )
        if bool(
            (
                activation_square_values
                > activation_square_limits + activation_square_tolerance
            ).any()
        ) or bool(
            (
                activation_cross_values
                > activation_cross_limits + activation_cross_tolerance
            ).any()
        ):
            raise ValueError("pool cross moments violate covariance bounds")

        pool_set = set(self.pool_indices)
        expected_singletons = tuple(
            index
            for index in range(self.source_width)
            if index not in pool_set
        )
        if self.singleton_indices != expected_singletons:
            raise ValueError("pseudo-unit singleton indices are invalid")
        expected_pairs = _match_pool_units(
            self.pool_indices,
            activation_gram,
            fisher_gram,
            cross_square_moment,
            activation_square_gradient_cross_moment,
            activation_cross_gradient_square_moment,
            down_gram,
        )
        if (
            type(self.pairs) is not tuple
            or any(
                not isinstance(pair, StructuredMLPPseudoUnitPair)
                for pair in self.pairs
            )
            or self.pairs != expected_pairs
        ):
            raise ValueError(
                "pseudo-unit matching or modal reductions are invalid"
            )

        expected_encoder, expected_decoder = _dense_encoder_decoder(
            source_width=self.source_width,
            singleton_indices=self.singleton_indices,
            pairs=self.pairs,
        )
        encoder = self._tensor(
            self.dense_loadings,
            label="dense_loadings",
            shape=(self.source_width, self.retained_width),
        )
        decoder = self._tensor(
            self.dense_reconstruction_loadings,
            label="dense_reconstruction_loadings",
            shape=(self.source_width, self.retained_width),
        )
        object.__setattr__(self, "dense_loadings", encoder)
        object.__setattr__(
            self,
            "dense_reconstruction_loadings",
            decoder,
        )
        if (
            not torch.equal(encoder, expected_encoder)
            or not torch.equal(decoder, expected_decoder)
            or self._tensor_sha(
                "dense_loadings",
                encoder,
                _ENCODER_DOMAIN,
            )
            != self.dense_loadings_sha256
            or self._tensor_sha(
                "dense_reconstruction_loadings",
                decoder,
                _DECODER_DOMAIN,
            )
            != self.dense_reconstruction_loadings_sha256
        ):
            raise ValueError("dense encoder, decoder, or digest is invalid")
        identity = torch.eye(
            self.retained_width,
            dtype=torch.float64,
        )
        if not torch.allclose(
            encoder.T @ encoder,
            identity,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "dense encoder loadings must have orthonormal columns"
            )
        payload = self._payload()
        if _json_sha256(payload, domain=_PLAN_DOMAIN) != self.plan_sha256:
            raise ValueError("pseudo-unit plan digest is invalid")

    @staticmethod
    def _tensor(
        value: Tensor,
        *,
        label: str,
        shape: tuple[int, ...],
    ) -> Tensor:
        result = _canonical_fp64(value, label=label, ndim=len(shape))
        if result.shape != shape:
            raise ValueError(f"{label} must have shape {shape}")
        return result

    @staticmethod
    def _tensor_sha(
        name: str,
        value: Tensor,
        domain: bytes,
    ) -> str:
        return _tensor_mapping_sha256({name: value}, domain=domain)

    @classmethod
    def _moment(
        cls,
        value: Tensor,
        *,
        label: str,
        shape: tuple[int, int],
        digest: str,
        domain: bytes,
    ) -> Tensor:
        result = cls._tensor(value, label=label, shape=shape)
        if not torch.equal(result, result.T):
            raise ValueError(f"{label} must be exactly symmetric")
        diagonal = torch.diagonal(result)
        if bool((diagonal < 0.0).any()):
            raise ValueError(f"{label} must have nonnegative diagonal")
        limits = torch.sqrt(diagonal[:, None] * diagonal[None, :])
        tolerance = (
            torch.finfo(torch.float64).eps
            * torch.maximum(limits, torch.ones_like(limits))
            * 256.0
        )
        if bool((result.abs() > limits + tolerance).any()):
            raise ValueError(f"{label} violates Gram covariance bounds")
        if cls._tensor_sha(label, result, domain) != digest:
            raise ValueError(f"{label} digest is invalid")
        return result

    @classmethod
    def _cross_moment(
        cls,
        value: Tensor,
        *,
        label: str,
        shape: tuple[int, int],
        digest: str,
        domain: bytes,
    ) -> Tensor:
        result = cls._tensor(value, label=label, shape=shape)
        if cls._tensor_sha(label, result, domain) != digest:
            raise ValueError(f"{label} digest is invalid")
        return result

    def _payload(self) -> dict[str, object]:
        return _plan_payload(
            provenance=self.provenance,
            calibration_split_sha256=self.calibration_split_sha256,
            activation_site=self.activation_site,
            parent_executor_fingerprint=(
                self.parent_executor_fingerprint
            ),
            source_width=self.source_width,
            retained_width=self.retained_width,
            output_width=self.output_width,
            valid_rows=self.valid_rows,
            batch_ids=self.batch_ids,
            input_batches_sha256=self.input_batches_sha256,
            source_down_weight_sha256=(
                self.source_down_weight_sha256
            ),
            diagonal_scores_sha256=self.diagonal_scores_sha256,
            pool_indices=self.pool_indices,
            pool_activation_gram_sha256=(
                self.pool_activation_gram_sha256
            ),
            pool_fisher_gram_sha256=self.pool_fisher_gram_sha256,
            pool_cross_square_moment_sha256=(
                self.pool_cross_square_moment_sha256
            ),
            pool_activation_square_gradient_cross_moment_sha256=(
                self.pool_activation_square_gradient_cross_moment_sha256
            ),
            pool_activation_cross_gradient_square_moment_sha256=(
                self.pool_activation_cross_gradient_square_moment_sha256
            ),
            pool_down_gram_sha256=self.pool_down_gram_sha256,
            singleton_indices=self.singleton_indices,
            pairs=self.pairs,
            dense_loadings_sha256=self.dense_loadings_sha256,
            dense_reconstruction_loadings_sha256=(
                self.dense_reconstruction_loadings_sha256
            ),
        )

    @property
    def pair_count(self) -> int:
        return self.source_width - self.retained_width

    @property
    def pool_size(self) -> int:
        return 2 * self.pair_count

    @property
    def singleton_count(self) -> int:
        return len(self.singleton_indices)

    @property
    def coordinate_sources(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            (source_index,)
            for source_index in self.singleton_indices
        ) + tuple(pair.source_indices for pair in self.pairs)

    @property
    def total_executed_fisher_residual_energy(self) -> float:
        return float(
            math.fsum(
                pair.executed_fisher_residual_energy
                for pair in self.pairs
            )
        )

    @property
    def total_native_pair_fisher_scalar_energy(self) -> float:
        return float(
            math.fsum(
                pair.native_fisher_scalar_energy
                for pair in self.pairs
            )
        )

    @property
    def total_fisher_normalization_energy(self) -> float:
        return float(
            math.fsum(
                pair.fisher_normalization_energy
                for pair in self.pairs
            )
        )

    @property
    def executed_fisher_residual_fraction(self) -> float:
        return (
            self.total_executed_fisher_residual_energy
            / self.total_fisher_normalization_energy
        )

    @property
    def total_fisher_contribution_vector_rank_one_tail_energy(
        self,
    ) -> float:
        return float(
            math.fsum(
                pair.fisher_contribution_vector_rank_one_tail_energy
                for pair in self.pairs
            )
        )

    @property
    def pooled_fisher_trace(self) -> float:
        return float(torch.trace(self.pool_fisher_gram).item())

    @property
    def dense_pair_loadings(self) -> Tensor:
        return self.dense_loadings[:, self.singleton_count :].clone()

    @property
    def dense_pair_reconstruction_loadings(self) -> Tensor:
        return self.dense_reconstruction_loadings[
            :, self.singleton_count :
        ].clone()

    def ideal_pair_features(self, projection_input: Tensor) -> Tensor:
        """Gather and mix only the two source columns for every pair."""

        if (
            not isinstance(projection_input, Tensor)
            or not projection_input.is_floating_point()
            or projection_input.ndim < 1
            or projection_input.shape[-1] != self.source_width
        ):
            raise ValueError(
                "projection_input must be floating with source_width on "
                "its final axis"
            )
        first_indices = torch.tensor(
            tuple(pair.source_indices[0] for pair in self.pairs),
            device=projection_input.device,
            dtype=torch.long,
        )
        second_indices = torch.tensor(
            tuple(pair.source_indices[1] for pair in self.pairs),
            device=projection_input.device,
            dtype=torch.long,
        )
        pair_loadings = torch.tensor(
            tuple(pair.loadings for pair in self.pairs),
            device=projection_input.device,
            dtype=projection_input.dtype,
        )
        first = projection_input.index_select(-1, first_indices)
        second = projection_input.index_select(-1, second_indices)
        return (
            first * pair_loadings[:, 0]
            + second * pair_loadings[:, 1]
        )

    def ideal_features(self, projection_input: Tensor) -> Tensor:
        """Return singleton gathers plus pair mixtures without a dense GEMM."""

        if (
            not isinstance(projection_input, Tensor)
            or not projection_input.is_floating_point()
            or projection_input.ndim < 1
            or projection_input.shape[-1] != self.source_width
        ):
            raise ValueError(
                "projection_input must be floating with source_width on "
                "its final axis"
            )
        singleton_indices = torch.tensor(
            self.singleton_indices,
            device=projection_input.device,
            dtype=torch.long,
        )
        singletons = projection_input.index_select(
            -1,
            singleton_indices,
        )
        return torch.cat(
            (singletons, self.ideal_pair_features(projection_input)),
            dim=-1,
        )

    def reconstruct_features(self, ideal_features: Tensor) -> Tensor:
        if (
            not isinstance(ideal_features, Tensor)
            or not ideal_features.is_floating_point()
            or ideal_features.ndim < 1
            or ideal_features.shape[-1] != self.retained_width
        ):
            raise ValueError(
                "ideal_features must be floating with retained_width on "
                "its final axis"
            )
        return ideal_features @ self.dense_reconstruction_loadings.T.to(
            device=ideal_features.device,
            dtype=ideal_features.dtype,
        )

    def validate_batches(
        self,
        batches: Sequence[StructuredMLPFisherTaylorBatch],
    ) -> None:
        ordered = _ordered_batches(batches)
        if (
            tuple(batch.batch_id for batch in ordered) != self.batch_ids
            or sum(batch.valid_rows for batch in ordered) != self.valid_rows
            or any(
                batch.provenance != self.provenance
                or batch.source_width != self.source_width
                for batch in ordered
            )
            or structured_mlp_fisher_batch_set_sha256(ordered)
            != self.input_batches_sha256
        ):
            raise ValueError(
                "Fisher/Taylor batches do not match the bundling plan"
            )

    def validate_source_down_weight(
        self,
        source_down_weight: Tensor,
    ) -> None:
        if (
            not isinstance(source_down_weight, Tensor)
            or source_down_weight.ndim != 2
            or source_down_weight.shape
            != (self.output_width, self.source_width)
            or structured_mlp_source_down_weight_sha256(
                source_down_weight
            )
            != self.source_down_weight_sha256
        ):
            raise ValueError(
                "source_down_weight does not match the bundling plan"
            )

    def direct_down_weight(
        self,
        source_down_weight: Tensor,
    ) -> Tensor:
        """Return ``D @ R`` after authenticating the exact source D."""

        self.validate_source_down_weight(source_down_weight)
        return source_down_weight @ self.dense_reconstruction_loadings.to(
            device=source_down_weight.device,
            dtype=source_down_weight.dtype,
        )

    def digests(self) -> dict[str, str]:
        return {
            "input_batches_sha256": self.input_batches_sha256,
            "source_down_weight_sha256": (
                self.source_down_weight_sha256
            ),
            "diagonal_scores_sha256": self.diagonal_scores_sha256,
            "pool_activation_gram_sha256": (
                self.pool_activation_gram_sha256
            ),
            "pool_fisher_gram_sha256": (
                self.pool_fisher_gram_sha256
            ),
            "pool_cross_square_moment_sha256": (
                self.pool_cross_square_moment_sha256
            ),
            "pool_activation_square_gradient_cross_moment_sha256": (
                self.pool_activation_square_gradient_cross_moment_sha256
            ),
            "pool_activation_cross_gradient_square_moment_sha256": (
                self.pool_activation_cross_gradient_square_moment_sha256
            ),
            "pool_down_gram_sha256": self.pool_down_gram_sha256,
            "dense_loadings_sha256": self.dense_loadings_sha256,
            "dense_reconstruction_loadings_sha256": (
                self.dense_reconstruction_loadings_sha256
            ),
            "plan_sha256": self.plan_sha256,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "coordinate_sources": self.coordinate_sources,
            "total_executed_fisher_residual_energy": (
                self.total_executed_fisher_residual_energy
            ),
            "total_native_pair_fisher_scalar_energy": (
                self.total_native_pair_fisher_scalar_energy
            ),
            "total_fisher_normalization_energy": (
                self.total_fisher_normalization_energy
            ),
            "executed_fisher_residual_fraction": (
                self.executed_fisher_residual_fraction
            ),
            "total_fisher_contribution_vector_rank_one_tail_energy": (
                self.total_fisher_contribution_vector_rank_one_tail_energy
            ),
            "pooled_fisher_trace": self.pooled_fisher_trace,
            "plan_sha256": self.plan_sha256,
        }


def build_fisher_pseudo_unit_bundling_plan(
    batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    source_down_weight: Tensor,
    calibration_split_sha256: str,
    activation_site: str,
    parent_executor_fingerprint: str,
    retained_width: int,
    expected_source_width: int | None = None,
) -> StructuredMLPPseudoUnitBundlingPlan:
    """Build a padding-safe plan scored by executed Fisher residuals."""

    ordered = _ordered_batches(batches)
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    _require_sha256(
        parent_executor_fingerprint,
        label="parent_executor_fingerprint",
    )
    if not isinstance(activation_site, str) or not activation_site:
        raise ValueError("activation_site must be nonempty")
    if type(retained_width) is not int:
        raise TypeError("retained_width must be an integer")
    if expected_source_width is not None and (
        type(expected_source_width) is not int
        or expected_source_width <= 0
    ):
        raise ValueError(
            "expected_source_width must be a positive integer"
        )
    down_weight = _canonical_fp64(
        source_down_weight,
        label="source_down_weight",
        ndim=2,
    )

    provenance = ordered[0].provenance
    source_width = ordered[0].source_width
    pair_count = source_width - retained_width
    if (
        not 0 < retained_width < source_width
        or 2 * pair_count > source_width
        or (
            expected_source_width is not None
            and source_width != expected_source_width
        )
        or down_weight.shape[0] <= 0
        or down_weight.shape[1] != source_width
        or any(
            batch.provenance != provenance
            or batch.source_width != source_width
            for batch in ordered
        )
    ):
        raise ValueError(
            "batches, source_down_weight, widths, or provenance are "
            "inconsistent; pair bundling requires retained_width >= "
            "ceil(source_width / 2)"
        )

    score_sum = torch.zeros(source_width, dtype=torch.float64)
    valid_rows = 0
    for batch in ordered:
        valid = batch.valid_mask
        activations = batch.projection_input[valid].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        gradients = batch.score_gradient[valid].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        influence = activations * gradients
        if not bool(torch.isfinite(influence).all()):
            raise RuntimeError(
                "Fisher influence accumulation produced nonfinite values"
            )
        score_sum.add_(influence.square().sum(dim=0))
        valid_rows += influence.shape[0]
    if (
        valid_rows <= 0
        or not bool(torch.isfinite(score_sum).all())
        or float(score_sum.sum().item()) <= 0.0
    ):
        raise RuntimeError("Fisher diagonal accumulation is invalid")
    scores = score_sum / valid_rows
    ranked_low = tuple(
        int(index)
        for index in torch.argsort(
            scores,
            descending=False,
            stable=True,
        ).tolist()
    )
    pool_indices = tuple(sorted(ranked_low[: 2 * pair_count]))
    pool_tensor = torch.tensor(pool_indices, dtype=torch.long)

    activation_sum = torch.zeros(
        len(pool_indices),
        len(pool_indices),
        dtype=torch.float64,
    )
    fisher_sum = torch.zeros_like(activation_sum)
    cross_square_sum = torch.zeros_like(activation_sum)
    activation_square_gradient_cross_sum = torch.zeros_like(
        activation_sum
    )
    activation_cross_gradient_square_sum = torch.zeros_like(
        activation_sum
    )
    for batch in ordered:
        valid = batch.valid_mask
        device_pool = pool_tensor.to(batch.projection_input.device)
        activations = batch.projection_input[valid].index_select(
            -1,
            device_pool,
        ).detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        gradients = batch.score_gradient[valid].index_select(
            -1,
            device_pool,
        ).detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        influence = activations * gradients
        activation_square = activations.square()
        gradient_square = gradients.square()
        activation_sum.add_(activations.T @ activations)
        fisher_sum.add_(influence.T @ influence)
        cross_square_sum.add_(activation_square.T @ gradient_square)
        activation_square_gradient_cross_sum.add_(
            (activation_square * gradients).T @ gradients
        )
        activation_cross_gradient_square_sum.add_(
            (activations * gradient_square).T @ activations
        )
    activation_gram = activation_sum / valid_rows
    fisher_gram = fisher_sum / valid_rows
    cross_square_moment = cross_square_sum / valid_rows
    activation_square_gradient_cross_moment = (
        activation_square_gradient_cross_sum / valid_rows
    )
    activation_cross_gradient_square_moment = (
        activation_cross_gradient_square_sum / valid_rows
    )
    activation_gram = (
        (activation_gram + activation_gram.T) * 0.5
    ).contiguous()
    fisher_gram = ((fisher_gram + fisher_gram.T) * 0.5).contiguous()
    cross_square_moment = cross_square_moment.contiguous()
    activation_square_gradient_cross_moment = (
        activation_square_gradient_cross_moment.contiguous()
    )
    activation_cross_gradient_square_moment = (
        activation_cross_gradient_square_moment.contiguous()
    )
    down_pool = down_weight[:, pool_tensor]
    down_gram = (down_pool.T @ down_pool).contiguous()
    down_gram = ((down_gram + down_gram.T) * 0.5).contiguous()

    pool_set = set(pool_indices)
    singleton_indices = tuple(
        index for index in range(source_width) if index not in pool_set
    )
    pairs = _match_pool_units(
        pool_indices,
        activation_gram,
        fisher_gram,
        cross_square_moment,
        activation_square_gradient_cross_moment,
        activation_cross_gradient_square_moment,
        down_gram,
    )
    encoder, decoder = _dense_encoder_decoder(
        source_width=source_width,
        singleton_indices=singleton_indices,
        pairs=pairs,
    )

    input_sha256 = structured_mlp_fisher_batch_set_sha256(ordered)
    down_weight_sha256 = structured_mlp_source_down_weight_sha256(
        down_weight
    )
    score_sha256 = _tensor_mapping_sha256(
        {"diagonal_scores": scores},
        domain=_SCORE_DOMAIN,
    )
    activation_sha256 = _tensor_mapping_sha256(
        {"pool_activation_gram": activation_gram},
        domain=_ACTIVATION_GRAM_DOMAIN,
    )
    fisher_sha256 = _tensor_mapping_sha256(
        {"pool_fisher_gram": fisher_gram},
        domain=_FISHER_GRAM_DOMAIN,
    )
    cross_square_sha256 = _tensor_mapping_sha256(
        {"pool_cross_square_moment": cross_square_moment},
        domain=_CROSS_SQUARE_MOMENT_DOMAIN,
    )
    activation_square_gradient_cross_sha256 = _tensor_mapping_sha256(
        {
            "pool_activation_square_gradient_cross_moment": (
                activation_square_gradient_cross_moment
            )
        },
        domain=_ACTIVATION_SQUARE_GRADIENT_CROSS_MOMENT_DOMAIN,
    )
    activation_cross_gradient_square_sha256 = _tensor_mapping_sha256(
        {
            "pool_activation_cross_gradient_square_moment": (
                activation_cross_gradient_square_moment
            )
        },
        domain=_ACTIVATION_CROSS_GRADIENT_SQUARE_MOMENT_DOMAIN,
    )
    down_gram_sha256 = _tensor_mapping_sha256(
        {"pool_down_gram": down_gram},
        domain=_DOWN_GRAM_DOMAIN,
    )
    encoder_sha256 = _tensor_mapping_sha256(
        {"dense_loadings": encoder},
        domain=_ENCODER_DOMAIN,
    )
    decoder_sha256 = _tensor_mapping_sha256(
        {"dense_reconstruction_loadings": decoder},
        domain=_DECODER_DOMAIN,
    )
    batch_ids = tuple(batch.batch_id for batch in ordered)
    payload = _plan_payload(
        provenance=provenance,
        calibration_split_sha256=calibration_split_sha256,
        activation_site=activation_site,
        parent_executor_fingerprint=parent_executor_fingerprint,
        source_width=source_width,
        retained_width=retained_width,
        output_width=int(down_weight.shape[0]),
        valid_rows=valid_rows,
        batch_ids=batch_ids,
        input_batches_sha256=input_sha256,
        source_down_weight_sha256=down_weight_sha256,
        diagonal_scores_sha256=score_sha256,
        pool_indices=pool_indices,
        pool_activation_gram_sha256=activation_sha256,
        pool_fisher_gram_sha256=fisher_sha256,
        pool_cross_square_moment_sha256=cross_square_sha256,
        pool_activation_square_gradient_cross_moment_sha256=(
            activation_square_gradient_cross_sha256
        ),
        pool_activation_cross_gradient_square_moment_sha256=(
            activation_cross_gradient_square_sha256
        ),
        pool_down_gram_sha256=down_gram_sha256,
        singleton_indices=singleton_indices,
        pairs=pairs,
        dense_loadings_sha256=encoder_sha256,
        dense_reconstruction_loadings_sha256=decoder_sha256,
    )
    return StructuredMLPPseudoUnitBundlingPlan(
        provenance=provenance,
        calibration_split_sha256=calibration_split_sha256,
        activation_site=activation_site,
        parent_executor_fingerprint=parent_executor_fingerprint,
        source_width=source_width,
        retained_width=retained_width,
        output_width=int(down_weight.shape[0]),
        valid_rows=valid_rows,
        batch_ids=batch_ids,
        input_batches_sha256=input_sha256,
        source_down_weight_sha256=down_weight_sha256,
        diagonal_scores=scores,
        diagonal_scores_sha256=score_sha256,
        pool_indices=pool_indices,
        pool_activation_gram=activation_gram,
        pool_activation_gram_sha256=activation_sha256,
        pool_fisher_gram=fisher_gram,
        pool_fisher_gram_sha256=fisher_sha256,
        pool_cross_square_moment=cross_square_moment,
        pool_cross_square_moment_sha256=cross_square_sha256,
        pool_activation_square_gradient_cross_moment=(
            activation_square_gradient_cross_moment
        ),
        pool_activation_square_gradient_cross_moment_sha256=(
            activation_square_gradient_cross_sha256
        ),
        pool_activation_cross_gradient_square_moment=(
            activation_cross_gradient_square_moment
        ),
        pool_activation_cross_gradient_square_moment_sha256=(
            activation_cross_gradient_square_sha256
        ),
        pool_down_gram=down_gram,
        pool_down_gram_sha256=down_gram_sha256,
        singleton_indices=singleton_indices,
        pairs=pairs,
        dense_loadings=encoder,
        dense_loadings_sha256=encoder_sha256,
        dense_reconstruction_loadings=decoder,
        dense_reconstruction_loadings_sha256=decoder_sha256,
        plan_sha256=_json_sha256(payload, domain=_PLAN_DOMAIN),
    )


__all__ = [
    "STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_ALGORITHM",
    "STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_FORMAT_VERSION",
    "STRUCTURED_MLP_PSEUDO_UNIT_BUNDLING_SCHEMA",
    "StructuredMLPPseudoUnitBundlingPlan",
    "StructuredMLPPseudoUnitPair",
    "build_fisher_pseudo_unit_bundling_plan",
    "structured_mlp_fisher_batch_set_sha256",
    "structured_mlp_source_down_weight_sha256",
]
