"""Authenticated groupwise dense coordinates for structured gated MLPs.

The plan produced here compacts a pool of ``K`` native MLP units into ``R``
latent supermodes while leaving every unit outside the pool exact.  It uses an
uncentered activation moment, a downstream-output metric, and a score-gradient
Fisher metric to derive a generalized encoder/decoder pair.

The encoder and decoder are compiler-only.  A valid deployment must synthesize
the ``R`` coordinates directly from the residual stream and pre-fold the
decoder into the new down projection; computing the native ``K`` activations
at runtime would defeat the compaction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .linear_codec import build_generalized_fisher_codec
from .structured_layer_distillation import StructuredLayerProvenance
from .structured_mlp_compression import StructuredMLPFisherTaylorBatch


STRUCTURED_MLP_DENSE_SUPERMODE_ALGORITHM = (
    "groupwise_output_fisher_generalized_codec_native_procrustes_v1"
)
STRUCTURED_MLP_DENSE_SUPERMODE_SCHEMA = (
    "fisher_graph.structured_mlp_dense_supermode_plan"
)
STRUCTURED_MLP_DENSE_SUPERMODE_FORMAT_VERSION = 1

_BATCH_SET_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.batch_set.v1\0"
)
_DOWN_WEIGHT_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.down_weight.v1\0"
)
_SCORE_DOMAIN = b"fisher_graph.structured_mlp.dense_supermode.score.v1\0"
_ENCODER_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.encoder.v1\0"
)
_DECODER_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.decoder.v1\0"
)
_PRE_ORIENTATION_SPECTRUM_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode."
    b"pre_orientation_spectrum.v1\0"
)
_ORIENTATION_DOMAIN = (
    b"fisher_graph.structured_mlp.dense_supermode.orientation.v1\0"
)
_PLAN_DOMAIN = b"fisher_graph.structured_mlp.dense_supermode.plan.v1\0"


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
    ordered = tuple(sorted(values, key=lambda batch: batch.batch_id))
    batch_ids = tuple(batch.batch_id for batch in ordered)
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("Fisher/Taylor batch ids must be unique")
    return ordered


def structured_mlp_dense_batch_set_sha256(
    batches: Sequence[StructuredMLPFisherTaylorBatch],
) -> str:
    """Hash only authenticated valid rows; padding values are irrelevant."""

    records = tuple(
        {
            "batch_id": batch.batch_id,
            "valid_rows": batch.valid_rows,
            "input_sha256": batch.input_sha256(),
        }
        for batch in _ordered_batches(batches)
    )
    return _json_sha256(records, domain=_BATCH_SET_DOMAIN)


def structured_mlp_dense_down_weight_sha256(weight: Tensor) -> str:
    if (
        not isinstance(weight, Tensor)
        or weight.ndim != 2
        or not weight.is_floating_point()
    ):
        raise ValueError(
            "source_down_weight must be one floating rank-2 tensor"
        )
    canonical = weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError("source_down_weight must be finite")
    return _tensor_mapping_sha256(
        {"source_down_weight": canonical},
        domain=_DOWN_WEIGHT_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class DenseSupermodeObjectiveWeights:
    """Relative weights for identity, output, and Fisher reconstruction."""

    activation: float = 0.05
    output: float = 1.0
    fisher: float = 1.0

    def __post_init__(self) -> None:
        for name in ("activation", "output", "fisher"):
            value = getattr(self, name)
            if (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"{name} objective weight must be finite and nonnegative"
                )
            object.__setattr__(self, name, float(value))
        if self.total <= 0.0:
            raise ValueError(
                "at least one dense-supermode objective weight must be "
                "positive"
            )

    @property
    def total(self) -> float:
        return self.activation + self.output + self.fisher

    def to_dict(self) -> dict[str, float]:
        return {
            "activation": self.activation,
            "output": self.output,
            "fisher": self.fisher,
        }


def _plan_payload(
    *,
    provenance: StructuredLayerProvenance,
    calibration_split_sha256: str,
    activation_site: str,
    parent_executor_fingerprint: str,
    source_width: int,
    pool_indices: tuple[int, ...],
    singleton_indices: tuple[int, ...],
    retained_pool_width: int,
    output_width: int,
    valid_rows: int,
    batch_ids: tuple[str, ...],
    input_batches_sha256: str,
    source_down_weight_sha256: str,
    diagonal_scores_sha256: str,
    pool_selection: str,
    objective_weights: DenseSupermodeObjectiveWeights,
    activation_energy: float,
    output_energy: float,
    fisher_energy: float,
    activation_floor: float,
    metric_floor: float,
    encoder_sha256: str,
    decoder_sha256: str,
    retained_pre_orientation_eigenspectrum_sha256: str,
    orientation_sha256: str,
    pivot_source_indices: tuple[int, ...],
    retained_spectrum_sum: float,
    total_spectrum_sum: float,
    relative_boundary_gap: float,
) -> dict[str, object]:
    pool_width = len(pool_indices)
    runtime_width = len(singleton_indices) + retained_pool_width
    return {
        "schema": STRUCTURED_MLP_DENSE_SUPERMODE_SCHEMA,
        "format_version": STRUCTURED_MLP_DENSE_SUPERMODE_FORMAT_VERSION,
        "algorithm": STRUCTURED_MLP_DENSE_SUPERMODE_ALGORITHM,
        "provenance": _provenance_dict(provenance),
        "calibration_split_sha256": calibration_split_sha256,
        "activation_site": activation_site,
        "parent_executor_fingerprint": parent_executor_fingerprint,
        "source_width": source_width,
        "pool_width": pool_width,
        "retained_pool_width": retained_pool_width,
        "removed_width": pool_width - retained_pool_width,
        "runtime_width": runtime_width,
        "output_width": output_width,
        "valid_rows": valid_rows,
        "batch_ids": batch_ids,
        "input_batches_sha256": input_batches_sha256,
        "source_down_weight_sha256": source_down_weight_sha256,
        "diagonal_scores_sha256": diagonal_scores_sha256,
        "pool_indices": pool_indices,
        "singleton_indices": singleton_indices,
        "pool_selection": pool_selection,
        "objective_weights": objective_weights.to_dict(),
        "objective": {
            "weights": objective_weights.to_dict(),
            "activation_metric": "identity_over_trace_C",
            "output_metric": "D_pool_transpose_D_pool_over_trace_Mo_C",
            "fisher_metric": "mean_score_gradient_outer_over_trace_Mf_C",
            "activation_moment": "uncentered_mean_z_transpose_z",
            "activation_energy": activation_energy,
            "output_energy": output_energy,
            "fisher_energy": fisher_energy,
            "activation_floor": activation_floor,
            "metric_floor": metric_floor,
            "separable_metric_is_initializer_only": True,
            "sampled_score_contraction_required_during_generator_fit": True,
        },
        "codec": {
            "kind": "generalized_fisher_dual_encoder_decoder",
            "encoder_shape": (pool_width, retained_pool_width),
            "decoder_shape": (pool_width, retained_pool_width),
            "encoder_sha256": encoder_sha256,
            "decoder_sha256": decoder_sha256,
            "retained_pre_orientation_eigenspectrum_sha256": (
                retained_pre_orientation_eigenspectrum_sha256
            ),
            "retained_spectrum_sum": retained_spectrum_sum,
            "total_spectrum_sum": total_spectrum_sum,
            "retained_spectrum_fraction": (
                retained_spectrum_sum / total_spectrum_sum
            ),
            "spectrum_semantics": (
                "generalized_operator_eigenvalues_before_native_"
                "procrustes_orientation"
            ),
            "relative_boundary_gap": relative_boundary_gap,
        },
        "orientation": {
            "kind": (
                "rank_revealing_decoder_pivots_then_orthogonal_procrustes_"
                "toward_normalized_native_activations"
            ),
            "pivot_source_indices": pivot_source_indices,
            "orientation_sha256": orientation_sha256,
            "projection_preserved": True,
        },
        "runtime_contract": {
            "coordinate_order": (
                "ascending_exact_singletons_then_dense_supermodes"
            ),
            "native_pool_features_may_not_be_computed": True,
            "encoder_and_decoder_are_analysis_fit_only": True,
            "decoder_must_be_folded_into_down_projection": True,
            "retained_coordinates_must_be_synthesized_directly": True,
        },
    }


@dataclass(frozen=True, slots=True)
class StructuredMLPDenseSupermodePlan:
    """Authenticated ``K -> R`` groupwise dense-supermode coordinate plan."""

    provenance: StructuredLayerProvenance
    calibration_split_sha256: str
    activation_site: str
    parent_executor_fingerprint: str
    source_width: int
    pool_indices: tuple[int, ...]
    singleton_indices: tuple[int, ...]
    retained_pool_width: int
    output_width: int
    valid_rows: int
    batch_ids: tuple[str, ...]
    input_batches_sha256: str
    source_down_weight_sha256: str
    diagonal_scores: Tensor
    diagonal_scores_sha256: str
    pool_selection: str
    objective_weights: DenseSupermodeObjectiveWeights
    activation_energy: float
    output_energy: float
    fisher_energy: float
    activation_floor: float
    metric_floor: float
    encoder: Tensor
    encoder_sha256: str
    decoder: Tensor
    decoder_sha256: str
    retained_pre_orientation_eigenspectrum: Tensor
    retained_pre_orientation_eigenspectrum_sha256: str
    orientation_sha256: str
    pivot_source_indices: tuple[int, ...]
    retained_spectrum_sum: float
    total_spectrum_sum: float
    relative_boundary_gap: float
    plan_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, StructuredLayerProvenance):
            raise TypeError(
                "provenance must be StructuredLayerProvenance"
            )
        for label, value in (
            (
                "calibration_split_sha256",
                self.calibration_split_sha256,
            ),
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
            ("encoder_sha256", self.encoder_sha256),
            ("decoder_sha256", self.decoder_sha256),
            (
                "retained_pre_orientation_eigenspectrum_sha256",
                self.retained_pre_orientation_eigenspectrum_sha256,
            ),
            ("orientation_sha256", self.orientation_sha256),
            ("plan_sha256", self.plan_sha256),
        ):
            _require_sha256(value, label=label)
        if (
            not isinstance(self.activation_site, str)
            or not self.activation_site
            or type(self.source_width) is not int
            or self.source_width <= 1
            or type(self.retained_pool_width) is not int
            or not 0 < self.retained_pool_width < len(self.pool_indices)
            or type(self.output_width) is not int
            or self.output_width <= 0
            or type(self.valid_rows) is not int
            or self.valid_rows <= 0
            or tuple(sorted(self.pool_indices)) != self.pool_indices
            or len(set(self.pool_indices)) != len(self.pool_indices)
            or tuple(sorted(self.singleton_indices))
            != self.singleton_indices
            or len(set(self.singleton_indices))
            != len(self.singleton_indices)
            or set(self.pool_indices) & set(self.singleton_indices)
            or tuple(
                sorted((*self.pool_indices, *self.singleton_indices))
            )
            != tuple(range(self.source_width))
            or type(self.batch_ids) is not tuple
            or not self.batch_ids
            or tuple(sorted(self.batch_ids)) != self.batch_ids
            or len(set(self.batch_ids)) != len(self.batch_ids)
            or not isinstance(
                self.objective_weights,
                DenseSupermodeObjectiveWeights,
            )
            or not isinstance(self.pool_selection, str)
            or not self.pool_selection
        ):
            raise ValueError("dense-supermode plan metadata is invalid")
        if (
            type(self.pivot_source_indices) is not tuple
            or len(self.pivot_source_indices)
            != self.retained_pool_width
            or len(set(self.pivot_source_indices))
            != len(self.pivot_source_indices)
            or not set(self.pivot_source_indices).issubset(
                self.pool_indices
            )
        ):
            raise ValueError(
                "dense-supermode pivot source indices are invalid"
            )

        pool_width = self.pool_width
        scores = self.diagonal_scores.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        encoder = self.encoder.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        decoder = self.decoder.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        retained_spectrum = (
            self.retained_pre_orientation_eigenspectrum.detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
            .contiguous()
        )
        if (
            scores.shape != (self.source_width,)
            or encoder.shape
            != (pool_width, self.retained_pool_width)
            or decoder.shape
            != (pool_width, self.retained_pool_width)
            or retained_spectrum.shape != (self.retained_pool_width,)
            or any(
                not bool(torch.isfinite(value).all())
                for value in (
                    scores,
                    encoder,
                    decoder,
                    retained_spectrum,
                )
            )
            or bool((scores < 0.0).any())
            or bool((retained_spectrum < 0.0).any())
            or (
                self.retained_pool_width > 1
                and bool(
                    (
                        retained_spectrum[:-1]
                        < retained_spectrum[1:] - 1e-10
                    ).any()
                )
            )
        ):
            raise ValueError(
                "dense-supermode score or codec tensors are invalid"
            )
        identity = torch.eye(
            self.retained_pool_width,
            dtype=torch.float64,
        )
        if not torch.allclose(
            encoder.mT @ decoder,
            identity,
            rtol=1e-8,
            atol=1e-8,
        ):
            raise ValueError(
                "dense-supermode encoder and decoder must be dual"
            )
        for name, value, domain, digest in (
            (
                "diagonal_scores",
                scores,
                _SCORE_DOMAIN,
                self.diagonal_scores_sha256,
            ),
            (
                "encoder",
                encoder,
                _ENCODER_DOMAIN,
                self.encoder_sha256,
            ),
            (
                "decoder",
                decoder,
                _DECODER_DOMAIN,
                self.decoder_sha256,
            ),
            (
                "retained_pre_orientation_eigenspectrum",
                retained_spectrum,
                _PRE_ORIENTATION_SPECTRUM_DOMAIN,
                self.retained_pre_orientation_eigenspectrum_sha256,
            ),
        ):
            if (
                _tensor_mapping_sha256(
                    {name: value},
                    domain=domain,
                )
                != digest
            ):
                raise ValueError(
                    f"dense-supermode {name} digest is invalid"
                )
        object.__setattr__(self, "diagonal_scores", scores.clone())
        object.__setattr__(self, "encoder", encoder.clone())
        object.__setattr__(self, "decoder", decoder.clone())
        object.__setattr__(
            self,
            "retained_pre_orientation_eigenspectrum",
            retained_spectrum.clone(),
        )

        objective_energy_pairs = (
            (self.objective_weights.activation, self.activation_energy),
            (self.objective_weights.output, self.output_energy),
            (self.objective_weights.fisher, self.fisher_energy),
        )
        if (
            any(
                type(energy) is not float
                or not math.isfinite(energy)
                or energy < 0.0
                or ((weight == 0.0) != (energy == 0.0))
                for weight, energy in objective_energy_pairs
            )
            or any(
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
                for value in (
                    self.activation_floor,
                    self.metric_floor,
                    self.retained_spectrum_sum,
                    self.total_spectrum_sum,
                )
            )
            or type(self.relative_boundary_gap) is not float
            or not math.isfinite(self.relative_boundary_gap)
            or self.relative_boundary_gap < 0.0
            or self.retained_spectrum_sum
            > self.total_spectrum_sum + 1e-10
            or not math.isclose(
                self.retained_spectrum_sum,
                float(retained_spectrum.sum().item()),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "dense-supermode energy or regularization metadata is invalid"
            )
        self.validate_integrity()

    @property
    def pool_width(self) -> int:
        return len(self.pool_indices)

    @property
    def singleton_count(self) -> int:
        return len(self.singleton_indices)

    @property
    def removed_width(self) -> int:
        return self.pool_width - self.retained_pool_width

    @property
    def runtime_width(self) -> int:
        return self.singleton_count + self.retained_pool_width

    @property
    def retained_spectrum_fraction(self) -> float:
        self.validate_integrity()
        return self.retained_spectrum_sum / self.total_spectrum_sum

    def _payload(self) -> dict[str, object]:
        return _plan_payload(
            provenance=self.provenance,
            calibration_split_sha256=self.calibration_split_sha256,
            activation_site=self.activation_site,
            parent_executor_fingerprint=(
                self.parent_executor_fingerprint
            ),
            source_width=self.source_width,
            pool_indices=self.pool_indices,
            singleton_indices=self.singleton_indices,
            retained_pool_width=self.retained_pool_width,
            output_width=self.output_width,
            valid_rows=self.valid_rows,
            batch_ids=self.batch_ids,
            input_batches_sha256=self.input_batches_sha256,
            source_down_weight_sha256=self.source_down_weight_sha256,
            diagonal_scores_sha256=self.diagonal_scores_sha256,
            pool_selection=self.pool_selection,
            objective_weights=self.objective_weights,
            activation_energy=self.activation_energy,
            output_energy=self.output_energy,
            fisher_energy=self.fisher_energy,
            activation_floor=self.activation_floor,
            metric_floor=self.metric_floor,
            encoder_sha256=self.encoder_sha256,
            decoder_sha256=self.decoder_sha256,
            retained_pre_orientation_eigenspectrum_sha256=(
                self.retained_pre_orientation_eigenspectrum_sha256
            ),
            orientation_sha256=self.orientation_sha256,
            pivot_source_indices=self.pivot_source_indices,
            retained_spectrum_sum=self.retained_spectrum_sum,
            total_spectrum_sum=self.total_spectrum_sum,
            relative_boundary_gap=self.relative_boundary_gap,
        )

    def validate_integrity(self) -> None:
        """Revalidate mutable tensor state and its authenticated payload."""

        scores = self.diagonal_scores.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        encoder = self.encoder.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        decoder = self.decoder.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        retained_spectrum = (
            self.retained_pre_orientation_eigenspectrum.detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
            .contiguous()
        )
        if (
            scores.shape != (self.source_width,)
            or encoder.shape
            != (self.pool_width, self.retained_pool_width)
            or decoder.shape
            != (self.pool_width, self.retained_pool_width)
            or retained_spectrum.shape != (self.retained_pool_width,)
            or any(
                not bool(torch.isfinite(value).all())
                for value in (
                    scores,
                    encoder,
                    decoder,
                    retained_spectrum,
                )
            )
        ):
            raise ValueError(
                "dense-supermode tensor integrity is invalid"
            )
        identity = torch.eye(
            self.retained_pool_width,
            dtype=torch.float64,
        )
        if not torch.allclose(
            encoder.mT @ decoder,
            identity,
            rtol=1e-8,
            atol=1e-8,
        ):
            raise ValueError(
                "dense-supermode encoder and decoder must be dual"
            )
        for name, value, domain, expected_sha256 in (
            (
                "diagonal_scores",
                scores,
                _SCORE_DOMAIN,
                self.diagonal_scores_sha256,
            ),
            (
                "encoder",
                encoder,
                _ENCODER_DOMAIN,
                self.encoder_sha256,
            ),
            (
                "decoder",
                decoder,
                _DECODER_DOMAIN,
                self.decoder_sha256,
            ),
            (
                "retained_pre_orientation_eigenspectrum",
                retained_spectrum,
                _PRE_ORIENTATION_SPECTRUM_DOMAIN,
                self.retained_pre_orientation_eigenspectrum_sha256,
            ),
        ):
            if (
                _tensor_mapping_sha256(
                    {name: value},
                    domain=domain,
                )
                != expected_sha256
            ):
                raise ValueError(
                    f"dense-supermode {name} digest is invalid"
                )
        if _json_sha256(
            self._payload(),
            domain=_PLAN_DOMAIN,
        ) != self.plan_sha256:
            raise ValueError("dense-supermode plan digest is invalid")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        result = self._payload()
        result["plan_sha256"] = self.plan_sha256
        return result

    def validate_batches(
        self,
        batches: Sequence[StructuredMLPFisherTaylorBatch],
    ) -> None:
        self.validate_integrity()
        ordered = _ordered_batches(batches)
        if (
            tuple(batch.batch_id for batch in ordered) != self.batch_ids
            or any(
                batch.provenance != self.provenance
                or batch.source_width != self.source_width
                for batch in ordered
            )
            or sum(batch.valid_rows for batch in ordered) != self.valid_rows
            or structured_mlp_dense_batch_set_sha256(ordered)
            != self.input_batches_sha256
        ):
            raise ValueError(
                "Fisher/Taylor batches do not match dense-supermode plan"
            )

    def validate_source_down_weight(self, weight: Tensor) -> None:
        self.validate_integrity()
        if (
            tuple(weight.shape)
            != (self.output_width, self.source_width)
            or structured_mlp_dense_down_weight_sha256(weight)
            != self.source_down_weight_sha256
        ):
            raise ValueError(
                "source down projection does not match dense-supermode plan"
            )

    def pool_features(self, features: Tensor) -> Tensor:
        self.validate_integrity()
        if (
            not isinstance(features, Tensor)
            or not features.is_floating_point()
            or features.shape[-1] != self.source_width
        ):
            raise ValueError(
                "features must end in the plan source width"
            )
        indices = torch.tensor(
            self.pool_indices,
            dtype=torch.long,
            device=features.device,
        )
        return features.index_select(-1, indices)

    def ideal_coordinates(self, features: Tensor) -> Tensor:
        self.validate_integrity()
        pool = self.pool_features(features)
        encoder = self.encoder.to(
            device=pool.device,
            dtype=pool.dtype,
        )
        return pool @ encoder

    def reconstruct_pool_features(self, coordinates: Tensor) -> Tensor:
        self.validate_integrity()
        if (
            not isinstance(coordinates, Tensor)
            or not coordinates.is_floating_point()
            or coordinates.shape[-1] != self.retained_pool_width
        ):
            raise ValueError(
                "coordinates must end in the retained pool width"
            )
        decoder = self.decoder.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        return coordinates @ decoder.mT

    def direct_supermode_down_weight(
        self,
        source_down_weight: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        self.validate_source_down_weight(source_down_weight)
        pool = torch.tensor(
            self.pool_indices,
            dtype=torch.long,
            device=source_down_weight.device,
        )
        decoder = self.decoder.to(
            device=source_down_weight.device,
            dtype=source_down_weight.dtype,
        )
        return source_down_weight.index_select(1, pool) @ decoder


def _positive_energy(
    metric: Tensor,
    covariance: Tensor,
    *,
    label: str,
) -> float:
    energy = float(torch.trace(metric @ covariance).item())
    if not math.isfinite(energy) or energy <= torch.finfo(
        torch.float64
    ).tiny:
        raise ValueError(f"{label} has no positive calibration energy")
    return energy


def _normalized_objective_term(
    metric: Tensor,
    covariance: Tensor,
    *,
    weight: float,
    label: str,
) -> tuple[float, Tensor]:
    if weight == 0.0:
        return 0.0, torch.zeros_like(metric)
    energy = _positive_energy(metric, covariance, label=label)
    return energy, metric * (weight / energy)


def _stable_low_score_pool(
    scores: Tensor,
    pool_width: int,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            sorted(
                range(scores.numel()),
                key=lambda index: (
                    float(scores[index].item()),
                    index,
                ),
            )[:pool_width]
        )
    )


def _native_oriented_codec(
    activations: Tensor,
    encoder: Tensor,
    decoder: Tensor,
    *,
    retained_width: int,
) -> tuple[Tensor, Tensor, tuple[int, ...], Tensor]:
    residual_rows = decoder.clone()
    selected: list[int] = []
    for _ in range(retained_width):
        leverage = residual_rows.square().sum(dim=1)
        for position in selected:
            leverage[position] = -1.0
        pivot_position = min(
            range(decoder.shape[0]),
            key=lambda index: (
                -float(leverage[index].item()),
                index,
            ),
        )
        pivot_norm = torch.linalg.vector_norm(
            residual_rows[pivot_position]
        )
        if float(pivot_norm.item()) <= torch.finfo(torch.float64).tiny:
            raise ValueError(
                "retained decoder subspace lacks independent native pivots"
            )
        direction = residual_rows[pivot_position] / pivot_norm
        residual_rows = (
            residual_rows
            - (residual_rows @ direction).unsqueeze(1)
            * direction.unsqueeze(0)
        )
        selected.append(pivot_position)
    pivot_positions = tuple(selected)
    pivot = activations[
        :,
        torch.tensor(pivot_positions, dtype=torch.long),
    ]
    ideal = activations @ encoder

    def normalize(values: Tensor) -> Tensor:
        scales = torch.sqrt(values.square().mean(dim=0)).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        return values / scales

    cross = normalize(ideal).mT @ normalize(pivot)
    left, _, right_h = torch.linalg.svd(cross, full_matrices=False)
    orientation = left @ right_h
    oriented_encoder = (encoder @ orientation).contiguous()
    oriented_decoder = (decoder @ orientation).contiguous()
    oriented_ideal = activations @ oriented_encoder
    for coordinate in range(retained_width):
        correlation = torch.dot(
            oriented_ideal[:, coordinate],
            pivot[:, coordinate],
        )
        if float(correlation.item()) < 0.0:
            oriented_encoder[:, coordinate].neg_()
            oriented_decoder[:, coordinate].neg_()
            orientation[:, coordinate].neg_()
    return (
        oriented_encoder,
        oriented_decoder,
        pivot_positions,
        orientation.contiguous(),
    )


def build_fisher_jacobian_dense_supermode_plan(
    batches: Sequence[StructuredMLPFisherTaylorBatch],
    *,
    source_down_weight: Tensor,
    calibration_split_sha256: str,
    activation_site: str,
    parent_executor_fingerprint: str,
    retained_pool_width: int,
    pool_width: int | None = None,
    pool_indices: Sequence[int] | None = None,
    expected_source_width: int | None = None,
    objective_weights: DenseSupermodeObjectiveWeights | None = None,
    activation_floor_fraction: float = 1e-8,
    metric_floor_fraction: float = 1e-8,
) -> StructuredMLPDenseSupermodePlan:
    """Build a groupwise Fisher/output-aware ``K -> R`` coordinate plan."""

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
    source_width = ordered[0].source_width
    if (
        any(
            batch.provenance != ordered[0].provenance
            or batch.source_width != source_width
            for batch in ordered
        )
        or (
            expected_source_width is not None
            and (
                type(expected_source_width) is not int
                or expected_source_width != source_width
            )
        )
    ):
        raise ValueError(
            "dense-supermode batches or expected width are incompatible"
        )
    if (
        type(retained_pool_width) is not int
        or retained_pool_width <= 0
    ):
        raise ValueError(
            "retained_pool_width must be a positive integer"
        )
    for label, value in (
        ("activation_floor_fraction", activation_floor_fraction),
        ("metric_floor_fraction", metric_floor_fraction),
    ):
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{label} must be finite and positive")

    down = source_down_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if (
        down.ndim != 2
        or down.shape[1] != source_width
        or not bool(torch.isfinite(down).all())
    ):
        raise ValueError(
            "source_down_weight must have shape [output, source_width]"
        )
    valid_rows = sum(batch.valid_rows for batch in ordered)
    diagonal_score_sum = torch.zeros(
        source_width,
        dtype=torch.float64,
    )
    for batch in ordered:
        activation_rows = (
            batch.projection_input[batch.valid_mask]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        gradient_rows = (
            batch.score_gradient[batch.valid_mask]
            .detach()
            .to(device="cpu", dtype=torch.float64)
        )
        if (
            activation_rows.ndim != 2
            or activation_rows.shape[1] != source_width
            or gradient_rows.shape != activation_rows.shape
            or not bool(torch.isfinite(activation_rows).all())
            or not bool(torch.isfinite(gradient_rows).all())
        ):
            raise ValueError(
                "valid dense-supermode activation and gradient rows are "
                "invalid"
            )
        diagonal_score_sum.add_(
            (activation_rows * gradient_rows).square().sum(dim=0)
        )
    diagonal_scores = diagonal_score_sum / valid_rows
    if pool_indices is not None and pool_width is not None:
        raise ValueError(
            "provide either pool_indices or pool_width, not both"
        )
    if pool_indices is not None:
        resolved_pool = tuple(sorted(pool_indices))
        if (
            not resolved_pool
            or any(type(index) is not int for index in resolved_pool)
            or len(set(resolved_pool)) != len(resolved_pool)
            or resolved_pool[0] < 0
            or resolved_pool[-1] >= source_width
        ):
            raise ValueError(
                "pool_indices must be unique in-range integers"
            )
        pool_selection = "explicit_ascending_source_indices"
    else:
        resolved_width = source_width if pool_width is None else pool_width
        if (
            type(resolved_width) is not int
            or not 1 < resolved_width <= source_width
        ):
            raise ValueError(
                "pool_width must be an integer between 2 and source width"
            )
        resolved_pool = _stable_low_score_pool(
            diagonal_scores,
            resolved_width,
        )
        pool_selection = (
            "all_source_units"
            if resolved_width == source_width
            else "lowest_diagonal_fisher_stable_source_index"
        )
    if retained_pool_width >= len(resolved_pool):
        raise ValueError(
            "retained_pool_width must be smaller than the pool width"
        )
    pool_set = set(resolved_pool)
    singleton_indices = tuple(
        index
        for index in range(source_width)
        if index not in pool_set
    )
    pool_tensor = torch.tensor(resolved_pool, dtype=torch.long)
    z = torch.cat(
        tuple(
            batch.projection_input[batch.valid_mask]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .index_select(1, pool_tensor)
            for batch in ordered
        ),
        dim=0,
    )
    score = torch.cat(
        tuple(
            batch.score_gradient[batch.valid_mask]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .index_select(1, pool_tensor)
            for batch in ordered
        ),
        dim=0,
    )
    d_pool = down.index_select(1, pool_tensor)
    covariance = (z.mT @ z / valid_rows).contiguous()
    score_fisher = (score.mT @ score / valid_rows).contiguous()
    output_metric = (d_pool.mT @ d_pool).contiguous()
    identity_metric = torch.eye(len(resolved_pool), dtype=torch.float64)
    weights = (
        DenseSupermodeObjectiveWeights()
        if objective_weights is None
        else objective_weights
    )
    if not isinstance(weights, DenseSupermodeObjectiveWeights):
        raise TypeError(
            "objective_weights must be DenseSupermodeObjectiveWeights"
        )
    activation_energy, activation_term = _normalized_objective_term(
        identity_metric,
        covariance,
        weight=weights.activation,
        label="activation objective",
    )
    output_energy, output_term = _normalized_objective_term(
        output_metric,
        covariance,
        weight=weights.output,
        label="output objective",
    )
    fisher_energy, fisher_term = _normalized_objective_term(
        score_fisher,
        covariance,
        weight=weights.fisher,
        label="Fisher objective",
    )
    metric = activation_term + output_term + fisher_term
    metric = ((metric + metric.mT) * 0.5).contiguous()
    covariance_scale = max(
        float(torch.linalg.eigvalsh(covariance).max().item()),
        torch.finfo(torch.float64).tiny,
    )
    metric_scale = max(
        float(torch.linalg.eigvalsh(metric).max().item()),
        torch.finfo(torch.float64).tiny,
    )
    activation_floor = float(activation_floor_fraction) * covariance_scale
    metric_floor = float(metric_floor_fraction) * metric_scale
    codec = build_generalized_fisher_codec(
        covariance=covariance,
        fisher_matrix=metric,
        alpha=activation_floor,
        beta=metric_floor,
        activation_name=activation_site,
        mean=torch.zeros(len(resolved_pool), dtype=torch.float64),
    )
    raw_encoder = codec.encoder[:, :retained_pool_width]
    raw_decoder = codec.decoder[:, :retained_pool_width]
    (
        encoder,
        decoder,
        pivot_positions,
        orientation,
    ) = _native_oriented_codec(
        z,
        raw_encoder,
        raw_decoder,
        retained_width=retained_pool_width,
    )
    pivot_source_indices = tuple(
        resolved_pool[position] for position in pivot_positions
    )
    retained_pre_orientation_eigenspectrum = codec.importance_scores[
        :retained_pool_width
    ].clone()
    retained_spectrum_sum = float(
        retained_pre_orientation_eigenspectrum.sum().item()
    )
    total_spectrum_sum = float(codec.importance_scores.sum().item())
    boundary_top = float(
        codec.importance_scores[retained_pool_width - 1].item()
    )
    boundary_bottom = float(
        codec.importance_scores[retained_pool_width].item()
    )
    relative_gap = max(
        (boundary_top - boundary_bottom)
        / max(boundary_top, torch.finfo(torch.float64).tiny),
        0.0,
    )
    score_sha256 = _tensor_mapping_sha256(
        {"diagonal_scores": diagonal_scores},
        domain=_SCORE_DOMAIN,
    )
    encoder_sha256 = _tensor_mapping_sha256(
        {"encoder": encoder},
        domain=_ENCODER_DOMAIN,
    )
    decoder_sha256 = _tensor_mapping_sha256(
        {"decoder": decoder},
        domain=_DECODER_DOMAIN,
    )
    retained_spectrum_sha256 = _tensor_mapping_sha256(
        {
            "retained_pre_orientation_eigenspectrum": (
                retained_pre_orientation_eigenspectrum
            )
        },
        domain=_PRE_ORIENTATION_SPECTRUM_DOMAIN,
    )
    orientation_sha256 = _tensor_mapping_sha256(
        {"orientation": orientation},
        domain=_ORIENTATION_DOMAIN,
    )
    batch_ids = tuple(batch.batch_id for batch in ordered)
    input_sha256 = structured_mlp_dense_batch_set_sha256(ordered)
    down_sha256 = structured_mlp_dense_down_weight_sha256(down)
    payload = _plan_payload(
        provenance=ordered[0].provenance,
        calibration_split_sha256=calibration_split_sha256,
        activation_site=activation_site,
        parent_executor_fingerprint=parent_executor_fingerprint,
        source_width=source_width,
        pool_indices=resolved_pool,
        singleton_indices=singleton_indices,
        retained_pool_width=retained_pool_width,
        output_width=int(down.shape[0]),
        valid_rows=valid_rows,
        batch_ids=batch_ids,
        input_batches_sha256=input_sha256,
        source_down_weight_sha256=down_sha256,
        diagonal_scores_sha256=score_sha256,
        pool_selection=pool_selection,
        objective_weights=weights,
        activation_energy=float(activation_energy),
        output_energy=float(output_energy),
        fisher_energy=float(fisher_energy),
        activation_floor=float(activation_floor),
        metric_floor=float(metric_floor),
        encoder_sha256=encoder_sha256,
        decoder_sha256=decoder_sha256,
        retained_pre_orientation_eigenspectrum_sha256=(
            retained_spectrum_sha256
        ),
        orientation_sha256=orientation_sha256,
        pivot_source_indices=pivot_source_indices,
        retained_spectrum_sum=retained_spectrum_sum,
        total_spectrum_sum=total_spectrum_sum,
        relative_boundary_gap=float(relative_gap),
    )
    return StructuredMLPDenseSupermodePlan(
        provenance=ordered[0].provenance,
        calibration_split_sha256=calibration_split_sha256,
        activation_site=activation_site,
        parent_executor_fingerprint=parent_executor_fingerprint,
        source_width=source_width,
        pool_indices=resolved_pool,
        singleton_indices=singleton_indices,
        retained_pool_width=retained_pool_width,
        output_width=int(down.shape[0]),
        valid_rows=valid_rows,
        batch_ids=batch_ids,
        input_batches_sha256=input_sha256,
        source_down_weight_sha256=down_sha256,
        diagonal_scores=diagonal_scores,
        diagonal_scores_sha256=score_sha256,
        pool_selection=pool_selection,
        objective_weights=weights,
        activation_energy=float(activation_energy),
        output_energy=float(output_energy),
        fisher_energy=float(fisher_energy),
        activation_floor=float(activation_floor),
        metric_floor=float(metric_floor),
        encoder=encoder,
        encoder_sha256=encoder_sha256,
        decoder=decoder,
        decoder_sha256=decoder_sha256,
        retained_pre_orientation_eigenspectrum=(
            retained_pre_orientation_eigenspectrum
        ),
        retained_pre_orientation_eigenspectrum_sha256=(
            retained_spectrum_sha256
        ),
        orientation_sha256=orientation_sha256,
        pivot_source_indices=pivot_source_indices,
        retained_spectrum_sum=retained_spectrum_sum,
        total_spectrum_sum=total_spectrum_sum,
        relative_boundary_gap=float(relative_gap),
        plan_sha256=_json_sha256(payload, domain=_PLAN_DOMAIN),
    )


__all__ = [
    "STRUCTURED_MLP_DENSE_SUPERMODE_ALGORITHM",
    "STRUCTURED_MLP_DENSE_SUPERMODE_FORMAT_VERSION",
    "STRUCTURED_MLP_DENSE_SUPERMODE_SCHEMA",
    "DenseSupermodeObjectiveWeights",
    "StructuredMLPDenseSupermodePlan",
    "build_fisher_jacobian_dense_supermode_plan",
    "structured_mlp_dense_batch_set_sha256",
    "structured_mlp_dense_down_weight_sha256",
]
