"""Bounded-memory discovery of cross-block pseudo-unit hypotheses.

This module deliberately stops before execution.  A first streaming pass
builds deterministic CountSketch summaries of native MLP coordinates ``z`` and
their signed score contributions ``z * dscore/dz``.  A second pass replays only
the sparse cross-block shortlist and accumulates exact row- and sequence-scope
moments.  The resulting hypotheses are useful inputs to a later common-boundary
executor fit; they do not authorize a replacement, execution, or a calibration
B evaluation.

The signed contribution is comparable across layers when every activation tap
is differentiated against the same scalar score.  It is invariant to the
coordinate gauge change ``z -> c*z, gradient -> gradient/c``.  Its correlation
is only a discovery signal: serial causal lineage and context-dependent
downstream directions still require an executable window-level guard.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import struct

import torch
from torch import Tensor

from .streaming_analysis import ActivationScoreGradientRows


_SKETCH_KIND = "fisher_graph.cross_block_mode_discovery_sketch"
_RESULT_KIND = "fisher_graph.cross_block_mode_discovery_result"
_FORMAT_VERSION = 1
_SKETCH_DOMAIN = b"fisher_graph.cross_block_count_sketch.v1\0"
_SEQUENCE_DOMAIN = b"fisher_graph.cross_block_sequence.v1\0"
_STREAM_DOMAIN = b"fisher_graph.cross_block_stream_multiset.v1\0"
_ARTIFACT_DOMAIN = b"fisher_graph.cross_block_artifact.v1\0"
_UINT256_MODULUS = 1 << 256


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value  # type: ignore[return-value]


def _json_sha256(value: object, *, domain: bytes = _ARTIFACT_DOMAIN) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(serialized)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor, *, label: str) -> str:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} must be a finite CPU float64 Tensor")
    canonical = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.cross_block_tensor.v1\0")
    digest.update(
        f"{tuple(canonical.shape)}\0float64\0".encode()
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _float_close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=2e-11, abs_tol=2e-12)


def _tensor_close(first: Tensor, second: Tensor) -> bool:
    scale = max(
        float(first.abs().max().item()) if first.numel() else 0.0,
        float(second.abs().max().item()) if second.numel() else 0.0,
        1.0,
    )
    return torch.allclose(
        first,
        second,
        rtol=2e-11,
        atol=2e-12 * scale,
    )


def _canonical_pair(
    first: ModeKey,
    second: ModeKey,
) -> tuple[ModeKey, ModeKey]:
    if first == second:
        raise ValueError("a cross-block pair must connect distinct modes")
    return (first, second) if first < second else (second, first)


@dataclass(frozen=True, order=True, slots=True)
class ModeKey:
    """Stable identity of one native scalar coordinate."""

    layer_ordinal: int
    layer_id: str
    activation_site: str
    mode_index: int
    fisher_rank: int

    def __post_init__(self) -> None:
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be nonnegative")
        _require_nonempty(self.layer_id, label="layer_id")
        _require_nonempty(self.activation_site, label="activation_site")
        if type(self.mode_index) is not int or self.mode_index < 0:
            raise ValueError("mode_index must be nonnegative")
        if type(self.fisher_rank) is not int or self.fisher_rank < 0:
            raise ValueError("fisher_rank must be nonnegative")

    def metadata(self) -> dict[str, object]:
        return {
            "layer_ordinal": self.layer_ordinal,
            "layer_id": self.layer_id,
            "activation_site": self.activation_site,
            "mode_index": self.mode_index,
            "fisher_rank": self.fisher_rank,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> ModeKey:
        expected = {
            "layer_ordinal",
            "layer_id",
            "activation_site",
            "mode_index",
            "fisher_rank",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("mode key fields are invalid")
        return cls(
            layer_ordinal=int(state["layer_ordinal"]),
            layer_id=str(state["layer_id"]),
            activation_site=str(state["activation_site"]),
            mode_index=int(state["mode_index"]),
            fisher_rank=int(state["fisher_rank"]),
        )


@dataclass(frozen=True, slots=True)
class CrossBlockLayerSpec:
    """Native layer/site whose coordinate ranks are derived by pass one."""

    layer_id: str
    layer_ordinal: int
    activation_site: str
    width: int

    def __post_init__(self) -> None:
        _require_nonempty(self.layer_id, label="layer_id")
        _require_nonempty(self.activation_site, label="activation_site")
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be nonnegative")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("width must be positive")

    def mode_key(self, mode_index: int, *, fisher_rank: int) -> ModeKey:
        if type(mode_index) is not int or not 0 <= mode_index < self.width:
            raise ValueError("mode_index is outside the layer width")
        return ModeKey(
            layer_ordinal=self.layer_ordinal,
            layer_id=self.layer_id,
            activation_site=self.activation_site,
            mode_index=mode_index,
            fisher_rank=fisher_rank,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "layer_ordinal": self.layer_ordinal,
            "activation_site": self.activation_site,
            "width": self.width,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockLayerSpec:
        expected = {
            "layer_id",
            "layer_ordinal",
            "activation_site",
            "width",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block layer fields are invalid")
        return cls(
            layer_id=str(state["layer_id"]),
            layer_ordinal=int(state["layer_ordinal"]),
            activation_site=str(state["activation_site"]),
            width=int(state["width"]),
        )


def _validated_layer_specs(
    values: Sequence[CrossBlockLayerSpec],
) -> tuple[CrossBlockLayerSpec, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("layer_specs must be a sequence")
    specs = tuple(values)
    if len(specs) < 2 or any(
        not isinstance(spec, CrossBlockLayerSpec) for spec in specs
    ):
        raise ValueError("at least two cross-block layer specs are required")
    expected = tuple(sorted(specs, key=lambda spec: spec.layer_ordinal))
    if specs != expected:
        raise ValueError("layer_specs must be ordered by layer_ordinal")
    if len({spec.layer_ordinal for spec in specs}) != len(specs):
        raise ValueError("layer ordinals must be unique")
    if len({spec.layer_id for spec in specs}) != len(specs):
        raise ValueError("layer ids must be unique")
    if len({spec.activation_site for spec in specs}) != len(specs):
        raise ValueError("activation sites must be unique")
    return specs


@dataclass(frozen=True, slots=True)
class CrossBlockDiscoveryProvenance:
    """Bindings shared by both streaming discovery passes."""

    model_fingerprint: str
    calibration_split_sha256: str
    objective_sha256: str
    score_reduction: str
    normalizer: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.model_fingerprint,
            label="model_fingerprint",
        )
        _require_sha256(
            self.calibration_split_sha256,
            label="calibration_split_sha256",
        )
        _require_sha256(self.objective_sha256, label="objective_sha256")
        _require_nonempty(self.score_reduction, label="score_reduction")
        _require_nonempty(self.normalizer, label="normalizer")

    def metadata(self) -> dict[str, object]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "calibration_split_sha256": self.calibration_split_sha256,
            "objective_sha256": self.objective_sha256,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockDiscoveryProvenance:
        expected = {
            "model_fingerprint",
            "calibration_split_sha256",
            "objective_sha256",
            "score_reduction",
            "normalizer",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block provenance fields are invalid")
        return cls(
            model_fingerprint=str(state["model_fingerprint"]),
            calibration_split_sha256=str(
                state["calibration_split_sha256"]
            ),
            objective_sha256=str(state["objective_sha256"]),
            score_reduction=str(state["score_reduction"]),
            normalizer=str(state["normalizer"]),
        )


@dataclass(frozen=True, slots=True)
class CrossBlockSketchConfig:
    """Fixed-memory first-pass and sparse-neighbor configuration."""

    sketch_size: int = 256
    sketch_seed: int = 0
    per_layer_pool_size: int = 256
    neighbors_per_mode: int = 8
    proxy_min_signed_correlation: float = -1.0

    def __post_init__(self) -> None:
        for label, value in (
            ("sketch_size", self.sketch_size),
            ("per_layer_pool_size", self.per_layer_pool_size),
            ("neighbors_per_mode", self.neighbors_per_mode),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if type(self.sketch_seed) is not int or self.sketch_seed < 0:
            raise ValueError("sketch_seed must be nonnegative")
        if (
            not isinstance(self.proxy_min_signed_correlation, float)
            or not math.isfinite(self.proxy_min_signed_correlation)
            or not -1.0 <= self.proxy_min_signed_correlation <= 1.0
        ):
            raise ValueError(
                "proxy_min_signed_correlation must lie in [-1, 1]"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "sketch_size": self.sketch_size,
            "sketch_seed": self.sketch_seed,
            "per_layer_pool_size": self.per_layer_pool_size,
            "neighbors_per_mode": self.neighbors_per_mode,
            "proxy_min_signed_correlation": (
                self.proxy_min_signed_correlation
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockSketchConfig:
        expected = {
            "sketch_size",
            "sketch_seed",
            "per_layer_pool_size",
            "neighbors_per_mode",
            "proxy_min_signed_correlation",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block sketch config fields are invalid")
        return cls(
            sketch_size=int(state["sketch_size"]),
            sketch_seed=int(state["sketch_seed"]),
            per_layer_pool_size=int(state["per_layer_pool_size"]),
            neighbors_per_mode=int(state["neighbors_per_mode"]),
            proxy_min_signed_correlation=float(
                state["proxy_min_signed_correlation"]
            ),
        )


def _effective_support_fraction(
    square_sum: float,
    fourth_sum: float,
    observations: int,
) -> float:
    if square_sum <= 0.0 or fourth_sum <= 0.0:
        return 0.0
    value = square_sum * square_sum / (observations * fourth_sum)
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class CrossBlockModeSketch:
    """One mode's fixed-size sketches and exact scalar moments."""

    key: ModeKey
    observations: int
    activation_square_sum: float
    activation_fourth_sum: float
    influence_square_sum: float
    influence_fourth_sum: float
    activation_sketch: Tensor
    influence_sketch: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.key, ModeKey):
            raise TypeError("key must be a ModeKey")
        if type(self.observations) is not int or self.observations <= 0:
            raise ValueError("observations must be positive")
        for label, value in (
            ("activation_square_sum", self.activation_square_sum),
            ("activation_fourth_sum", self.activation_fourth_sum),
            ("influence_square_sum", self.influence_square_sum),
            ("influence_fourth_sum", self.influence_fourth_sum),
        ):
            if (
                not isinstance(value, float)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{label} must be finite and nonnegative")
        if (
            self.activation_sketch.ndim != 1
            or self.influence_sketch.shape != self.activation_sketch.shape
            or self.activation_sketch.numel() == 0
        ):
            raise ValueError("mode sketches must be equal nonempty vectors")
        for label, value in (
            ("activation_sketch", self.activation_sketch),
            ("influence_sketch", self.influence_sketch),
        ):
            _tensor_sha256(value, label=label)
            object.__setattr__(self, label, value.detach().clone())

    @property
    def activation_density(self) -> float:
        return _effective_support_fraction(
            self.activation_square_sum,
            self.activation_fourth_sum,
            self.observations,
        )

    @property
    def influence_density(self) -> float:
        return _effective_support_fraction(
            self.influence_square_sum,
            self.influence_fourth_sum,
            self.observations,
        )

    @property
    def fisher_energy(self) -> float:
        return self.influence_square_sum / self.observations

    def metadata(self) -> dict[str, object]:
        return {
            "key": self.key.metadata(),
            "observations": self.observations,
            "activation_square_sum": self.activation_square_sum,
            "activation_fourth_sum": self.activation_fourth_sum,
            "influence_square_sum": self.influence_square_sum,
            "influence_fourth_sum": self.influence_fourth_sum,
            "activation_density": self.activation_density,
            "influence_density": self.influence_density,
            "fisher_energy": self.fisher_energy,
            "activation_sketch_sha256": _tensor_sha256(
                self.activation_sketch,
                label="activation_sketch",
            ),
            "influence_sketch_sha256": _tensor_sha256(
                self.influence_sketch,
                label="influence_sketch",
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "key": self.key.metadata(),
            "observations": self.observations,
            "activation_square_sum": self.activation_square_sum,
            "activation_fourth_sum": self.activation_fourth_sum,
            "influence_square_sum": self.influence_square_sum,
            "influence_fourth_sum": self.influence_fourth_sum,
            "activation_sketch": self.activation_sketch.clone(),
            "influence_sketch": self.influence_sketch.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockModeSketch:
        expected = {
            "key",
            "observations",
            "activation_square_sum",
            "activation_fourth_sum",
            "influence_square_sum",
            "influence_fourth_sum",
            "activation_sketch",
            "influence_sketch",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block mode sketch fields are invalid")
        if not isinstance(state["key"], Mapping):
            raise TypeError("mode sketch key must be a mapping")
        if not isinstance(state["activation_sketch"], Tensor) or not isinstance(
            state["influence_sketch"], Tensor
        ):
            raise TypeError("mode sketches must be Tensors")
        return cls(
            key=ModeKey.from_state_dict(state["key"]),
            observations=int(state["observations"]),
            activation_square_sum=float(state["activation_square_sum"]),
            activation_fourth_sum=float(state["activation_fourth_sum"]),
            influence_square_sum=float(state["influence_square_sum"]),
            influence_fourth_sum=float(state["influence_fourth_sum"]),
            activation_sketch=state["activation_sketch"],
            influence_sketch=state["influence_sketch"],
        )


@dataclass(frozen=True, slots=True)
class CrossBlockProxyEdge:
    """Sparse first-pass neighbor edge; never an execution authorization."""

    first: ModeKey
    second: ModeKey
    proxy_signed_influence_correlation: float

    def __post_init__(self) -> None:
        if not isinstance(self.first, ModeKey) or not isinstance(
            self.second, ModeKey
        ):
            raise TypeError("proxy endpoints must be ModeKey values")
        expected = _canonical_pair(self.first, self.second)
        if (self.first, self.second) != expected:
            raise ValueError("proxy endpoints must use canonical order")
        if self.first.layer_ordinal == self.second.layer_ordinal:
            raise ValueError("a cross-block edge must connect distinct layers")
        if (
            not isinstance(self.proxy_signed_influence_correlation, float)
            or not math.isfinite(
                self.proxy_signed_influence_correlation
            )
            or not -1.0
            <= self.proxy_signed_influence_correlation
            <= 1.0
        ):
            raise ValueError("proxy correlation must lie in [-1, 1]")

    @property
    def endpoints(self) -> tuple[ModeKey, ModeKey]:
        return self.first, self.second

    def metadata(self) -> dict[str, object]:
        return {
            "first": self.first.metadata(),
            "second": self.second.metadata(),
            "proxy_signed_influence_correlation": (
                self.proxy_signed_influence_correlation
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockProxyEdge:
        expected = {
            "first",
            "second",
            "proxy_signed_influence_correlation",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block proxy edge fields are invalid")
        if not isinstance(state["first"], Mapping) or not isinstance(
            state["second"], Mapping
        ):
            raise TypeError("proxy endpoint states must be mappings")
        return cls(
            first=ModeKey.from_state_dict(state["first"]),
            second=ModeKey.from_state_dict(state["second"]),
            proxy_signed_influence_correlation=float(
                state["proxy_signed_influence_correlation"]
            ),
        )


class _StreamMultisetDigest:
    """Order-independent fixed-memory digest of independent sequences."""

    def __init__(self) -> None:
        self.count = 0
        self.row_count = 0
        self.xor = 0
        self.sum = 0

    def update(self, digest: bytes, *, rows: int) -> None:
        if len(digest) != 32:
            raise ValueError("sequence digest must contain 32 bytes")
        value = int.from_bytes(digest, "big")
        self.count += 1
        self.row_count += rows
        self.xor ^= value
        self.sum = (self.sum + value) % _UINT256_MODULUS

    def hexdigest(self) -> str:
        digest = hashlib.sha256()
        digest.update(_STREAM_DOMAIN)
        digest.update(struct.pack("<QQ", self.count, self.row_count))
        digest.update(self.xor.to_bytes(32, "big"))
        digest.update(self.sum.to_bytes(32, "big"))
        return digest.hexdigest()


def _sequence_digest(
    rows: ActivationScoreGradientRows,
    specs: tuple[CrossBlockLayerSpec, ...],
) -> bytes:
    if rows.example_id is None:
        raise ValueError(
            "cross-block discovery requires stable example_id values"
        )
    digest = hashlib.sha256()
    digest.update(_SEQUENCE_DOMAIN)
    encoded_id = rows.example_id.encode()
    digest.update(struct.pack("<Q", len(encoded_id)))
    digest.update(encoded_id)
    positions = rows.logical_positions.to(
        device="cpu",
        dtype=torch.int64,
    ).contiguous()
    digest.update(struct.pack("<Q", positions.numel()))
    digest.update(positions.numpy().tobytes(order="C"))
    for spec in specs:
        try:
            activations = rows.activations[spec.activation_site]
            gradients = rows.score_gradients[spec.activation_site]
        except KeyError as error:
            raise KeyError(
                f"row stream is missing {spec.activation_site!r}"
            ) from error
        if activations.shape != (rows.observations, spec.width):
            raise ValueError(
                f"{spec.activation_site!r} rows do not match its layer spec"
            )
        for value in (activations, gradients):
            canonical = value.detach().to(
                device="cpu",
                dtype=torch.float64,
            ).contiguous()
            digest.update(canonical.numpy().tobytes(order="C"))
    return digest.digest()


def _sketch_location(
    *,
    seed: int,
    example_id: str,
    logical_position: int,
    sketch_size: int,
) -> tuple[int, float]:
    digest = hashlib.sha256()
    digest.update(_SKETCH_DOMAIN)
    digest.update(struct.pack("<Qq", seed, logical_position))
    encoded = example_id.encode()
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    raw = digest.digest()
    bucket = int.from_bytes(raw[:8], "little") % sketch_size
    sign = 1.0 if raw[8] & 1 else -1.0
    return bucket, sign


def _proxy_correlation(
    first: CrossBlockModeSketch,
    second: CrossBlockModeSketch,
) -> float:
    denominator = math.sqrt(
        first.influence_square_sum * second.influence_square_sum
    )
    if denominator <= 0.0:
        return 0.0
    estimate = float(
        torch.dot(first.influence_sketch, second.influence_sketch).item()
        / denominator
    )
    return min(max(estimate, -1.0), 1.0)


def _mode_pool_sort_key(
    mode: CrossBlockModeSketch,
) -> tuple[float, float, float, float, ModeKey]:
    return (
        max(mode.influence_density, mode.activation_density),
        mode.influence_density,
        mode.activation_density,
        mode.fisher_energy,
        mode.key,
    )


def _derived_fisher_ranks(energies: Tensor) -> tuple[int, ...]:
    """Rank descending Fisher energy with stable source-index ties."""

    if (
        energies.ndim != 1
        or energies.numel() == 0
        or energies.device.type != "cpu"
        or energies.dtype != torch.float64
        or not torch.isfinite(energies).all()
        or (energies < 0).any()
    ):
        raise ValueError("Fisher energies must be a finite CPU float64 vector")
    ordered = torch.argsort(
        energies,
        descending=True,
        stable=True,
    ).tolist()
    ranks = [0] * energies.numel()
    for rank, source_index in enumerate(ordered):
        ranks[source_index] = rank
    return tuple(ranks)


def _shortlist_edges(
    modes: tuple[CrossBlockModeSketch, ...],
    specs: tuple[CrossBlockLayerSpec, ...],
    config: CrossBlockSketchConfig,
) -> tuple[tuple[ModeKey, ...], tuple[CrossBlockProxyEdge, ...]]:
    by_layer: dict[int, list[CrossBlockModeSketch]] = {
        spec.layer_ordinal: [] for spec in specs
    }
    for mode in modes:
        by_layer[mode.key.layer_ordinal].append(mode)
    pool_modes: list[CrossBlockModeSketch] = []
    for spec in specs:
        ranked = sorted(
            by_layer[spec.layer_ordinal],
            key=_mode_pool_sort_key,
        )
        pool_modes.extend(ranked[: config.per_layer_pool_size])
    pool_modes.sort(key=lambda mode: mode.key)
    pool_keys = tuple(mode.key for mode in pool_modes)

    # Each layer-pair materializes at most ``pool_size ** 2`` correlations.
    # Stable matrix argsorts provide deterministic local top-k candidates;
    # only the small union of per-layer top-k lists reaches Python.
    candidates_by_source: dict[
        ModeKey,
        list[tuple[float, ModeKey]],
    ] = {mode.key: [] for mode in pool_modes}
    pooled_by_layer = {
        spec.layer_ordinal: tuple(
            mode
            for mode in pool_modes
            if mode.key.layer_ordinal == spec.layer_ordinal
        )
        for spec in specs
    }
    for first_spec_index, first_spec in enumerate(specs):
        first_modes = pooled_by_layer[first_spec.layer_ordinal]
        if not first_modes:
            continue
        first_sketch = torch.stack(
            tuple(mode.influence_sketch for mode in first_modes)
        )
        first_norm = torch.tensor(
            tuple(
                math.sqrt(mode.influence_square_sum)
                for mode in first_modes
            ),
            dtype=torch.float64,
        )
        for second_spec in specs[first_spec_index + 1 :]:
            second_modes = pooled_by_layer[second_spec.layer_ordinal]
            if not second_modes:
                continue
            second_sketch = torch.stack(
                tuple(mode.influence_sketch for mode in second_modes)
            )
            second_norm = torch.tensor(
                tuple(
                    math.sqrt(mode.influence_square_sum)
                    for mode in second_modes
                ),
                dtype=torch.float64,
            )
            denominator = first_norm[:, None] * second_norm[None, :]
            correlation = first_sketch @ second_sketch.T
            correlation = torch.where(
                denominator > 0.0,
                correlation / denominator.clamp_min(
                    torch.finfo(torch.float64).tiny
                ),
                torch.zeros_like(correlation),
            ).clamp(-1.0, 1.0)
            local_k = min(
                config.neighbors_per_mode,
                max(len(first_modes), len(second_modes)),
            )
            first_order = torch.argsort(
                correlation,
                dim=1,
                descending=True,
                stable=True,
            )[:, : min(local_k, len(second_modes))]
            for source_index, source in enumerate(first_modes):
                for target_index in first_order[source_index].tolist():
                    value = float(
                        correlation[source_index, target_index].item()
                    )
                    if value >= config.proxy_min_signed_correlation:
                        candidates_by_source[source.key].append(
                            (-value, second_modes[target_index].key)
                        )
            second_order = torch.argsort(
                correlation.T,
                dim=1,
                descending=True,
                stable=True,
            )[:, : min(local_k, len(first_modes))]
            for source_index, source in enumerate(second_modes):
                for target_index in second_order[source_index].tolist():
                    value = float(
                        correlation[target_index, source_index].item()
                    )
                    if value >= config.proxy_min_signed_correlation:
                        candidates_by_source[source.key].append(
                            (-value, first_modes[target_index].key)
                        )

    selected: dict[
        tuple[ModeKey, ModeKey],
        CrossBlockProxyEdge,
    ] = {}
    for source in pool_modes:
        candidates = sorted(candidates_by_source[source.key])
        for negative_correlation, target_key in candidates[
            : config.neighbors_per_mode
        ]:
            first, second = _canonical_pair(source.key, target_key)
            correlation_value = -negative_correlation
            edge = CrossBlockProxyEdge(
                first=first,
                second=second,
                proxy_signed_influence_correlation=float(
                    correlation_value
                ),
            )
            previous = selected.get((first, second))
            if (
                previous is None
                or edge.proxy_signed_influence_correlation
                > previous.proxy_signed_influence_correlation
            ):
                selected[(first, second)] = edge
    edges = tuple(
        selected[key]
        for key in sorted(selected)
    )
    return pool_keys, edges


@dataclass(frozen=True, slots=True)
class CrossBlockDiscoverySketch:
    """Authenticated fixed-memory first-pass artifact."""

    provenance: CrossBlockDiscoveryProvenance
    layer_specs: tuple[CrossBlockLayerSpec, ...]
    config: CrossBlockSketchConfig
    sequences: int
    observations: int
    row_stream_sha256: str
    modes: tuple[CrossBlockModeSketch, ...]
    pool_mode_keys: tuple[ModeKey, ...]
    proxy_edges: tuple[CrossBlockProxyEdge, ...]
    artifact_sha256: str
    artifact_kind: str = _SKETCH_KIND
    format_version: int = _FORMAT_VERSION
    contains_corpus_rows: bool = False
    discovery_only: bool = True
    authorizes_execution: bool = False
    authorizes_b: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, CrossBlockDiscoveryProvenance):
            raise TypeError("provenance is invalid")
        specs = _validated_layer_specs(self.layer_specs)
        if specs != self.layer_specs:
            raise ValueError("layer_specs are not canonical")
        if not isinstance(self.config, CrossBlockSketchConfig):
            raise TypeError("config is invalid")
        if (
            type(self.sequences) is not int
            or self.sequences <= 0
            or type(self.observations) is not int
            or self.observations < self.sequences
        ):
            raise ValueError("stream counts are invalid")
        _require_sha256(
            self.row_stream_sha256,
            label="row_stream_sha256",
        )
        expected_keys_list: list[ModeKey] = []
        offset = 0
        for spec in specs:
            layer_modes = self.modes[offset : offset + spec.width]
            if len(layer_modes) != spec.width:
                raise ValueError("mode sketches do not cover every layer")
            ranks = _derived_fisher_ranks(
                torch.tensor(
                    tuple(
                        mode.influence_square_sum
                        for mode in layer_modes
                    ),
                    dtype=torch.float64,
                )
            )
            expected_keys_list.extend(
                spec.mode_key(index, fisher_rank=ranks[index])
                for index in range(spec.width)
            )
            offset += spec.width
        expected_keys = tuple(expected_keys_list)
        if (
            type(self.modes) is not tuple
            or tuple(mode.key for mode in self.modes) != expected_keys
            or any(
                mode.observations != self.observations
                or mode.activation_sketch.numel() != self.config.sketch_size
                for mode in self.modes
            )
        ):
            raise ValueError("mode sketches do not match the layer specs")
        known = set(expected_keys)
        if (
            type(self.pool_mode_keys) is not tuple
            or tuple(sorted(self.pool_mode_keys)) != self.pool_mode_keys
            or len(set(self.pool_mode_keys)) != len(self.pool_mode_keys)
            or not set(self.pool_mode_keys).issubset(known)
        ):
            raise ValueError("pool mode keys are invalid")
        if (
            type(self.proxy_edges) is not tuple
            or tuple(
                sorted(
                    self.proxy_edges,
                    key=lambda edge: edge.endpoints,
                )
            )
            != self.proxy_edges
            or any(
                edge.first not in self.pool_mode_keys
                or edge.second not in self.pool_mode_keys
                for edge in self.proxy_edges
            )
        ):
            raise ValueError("proxy edges are invalid")
        expected_pool, expected_edges = _shortlist_edges(
            self.modes,
            specs,
            self.config,
        )
        if (
            expected_pool != self.pool_mode_keys
            or len(expected_edges) != len(self.proxy_edges)
            or any(
                left.endpoints != right.endpoints
                or not _float_close(
                    left.proxy_signed_influence_correlation,
                    right.proxy_signed_influence_correlation,
                )
                for left, right in zip(
                    expected_edges,
                    self.proxy_edges,
                    strict=True,
                )
            )
        ):
            raise ValueError("proxy shortlist does not match mode sketches")
        if (
            self.artifact_kind != _SKETCH_KIND
            or self.format_version != _FORMAT_VERSION
            or self.contains_corpus_rows
            or not self.discovery_only
            or self.authorizes_execution
            or self.authorizes_b
        ):
            raise ValueError("sketch safety metadata is invalid")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("cross-block sketch artifact hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_corpus_rows": self.contains_corpus_rows,
            "discovery_only": self.discovery_only,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_b": self.authorizes_b,
            "provenance": self.provenance.metadata(),
            "layer_specs": tuple(spec.metadata() for spec in self.layer_specs),
            "config": self.config.metadata(),
            "sequences": self.sequences,
            "observations": self.observations,
            "row_stream_sha256": self.row_stream_sha256,
            "modes": tuple(mode.metadata() for mode in self.modes),
            "pool_mode_keys": tuple(
                key.metadata() for key in self.pool_mode_keys
            ),
            "proxy_edges": tuple(
                edge.metadata() for edge in self.proxy_edges
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload())

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_corpus_rows": self.contains_corpus_rows,
            "discovery_only": self.discovery_only,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_b": self.authorizes_b,
            "provenance": self.provenance.metadata(),
            "layer_specs": tuple(
                spec.metadata() for spec in self.layer_specs
            ),
            "config": self.config.metadata(),
            "sequences": self.sequences,
            "observations": self.observations,
            "row_stream_sha256": self.row_stream_sha256,
            "modes": tuple(mode.state_dict() for mode in self.modes),
            "pool_mode_keys": tuple(
                key.metadata() for key in self.pool_mode_keys
            ),
            "proxy_edges": tuple(
                edge.metadata() for edge in self.proxy_edges
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockDiscoverySketch:
        expected = {
            "artifact_kind",
            "format_version",
            "contains_corpus_rows",
            "discovery_only",
            "authorizes_execution",
            "authorizes_b",
            "provenance",
            "layer_specs",
            "config",
            "sequences",
            "observations",
            "row_stream_sha256",
            "modes",
            "pool_mode_keys",
            "proxy_edges",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block sketch state fields are invalid")
        for label in (
            "provenance",
            "config",
        ):
            if not isinstance(state[label], Mapping):
                raise TypeError(f"{label} state must be a mapping")
        for label in (
            "layer_specs",
            "modes",
            "pool_mode_keys",
            "proxy_edges",
        ):
            if not isinstance(state[label], tuple):
                raise TypeError(f"{label} state must be a tuple")
        return cls(
            provenance=CrossBlockDiscoveryProvenance.from_state_dict(
                state["provenance"]
            ),
            layer_specs=tuple(
                CrossBlockLayerSpec.from_state_dict(value)
                for value in state["layer_specs"]  # type: ignore[union-attr]
            ),
            config=CrossBlockSketchConfig.from_state_dict(
                state["config"]
            ),
            sequences=int(state["sequences"]),
            observations=int(state["observations"]),
            row_stream_sha256=str(state["row_stream_sha256"]),
            modes=tuple(
                CrossBlockModeSketch.from_state_dict(value)
                for value in state["modes"]  # type: ignore[union-attr]
            ),
            pool_mode_keys=tuple(
                ModeKey.from_state_dict(value)
                for value in state["pool_mode_keys"]  # type: ignore[union-attr]
            ),
            proxy_edges=tuple(
                CrossBlockProxyEdge.from_state_dict(value)
                for value in state["proxy_edges"]  # type: ignore[union-attr]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
            contains_corpus_rows=bool(state["contains_corpus_rows"]),
            discovery_only=bool(state["discovery_only"]),
            authorizes_execution=bool(state["authorizes_execution"]),
            authorizes_b=bool(state["authorizes_b"]),
        )


def build_cross_block_discovery_sketch(
    rows: Iterable[ActivationScoreGradientRows],
    *,
    layer_specs: Sequence[CrossBlockLayerSpec],
    provenance: CrossBlockDiscoveryProvenance,
    config: CrossBlockSketchConfig = CrossBlockSketchConfig(),
) -> CrossBlockDiscoverySketch:
    """Stream fixed-size whole-model mode sketches without retaining rows."""

    specs = _validated_layer_specs(layer_specs)
    if not isinstance(provenance, CrossBlockDiscoveryProvenance):
        raise TypeError("provenance is invalid")
    if not isinstance(config, CrossBlockSketchConfig):
        raise TypeError("config is invalid")

    activation_sketches = {
        spec.layer_ordinal: torch.zeros(
            spec.width,
            config.sketch_size,
            dtype=torch.float64,
        )
        for spec in specs
    }
    influence_sketches = {
        spec.layer_ordinal: torch.zeros_like(
            activation_sketches[spec.layer_ordinal]
        )
        for spec in specs
    }
    activation_square = {
        spec.layer_ordinal: torch.zeros(spec.width, dtype=torch.float64)
        for spec in specs
    }
    activation_fourth = {
        spec.layer_ordinal: torch.zeros(spec.width, dtype=torch.float64)
        for spec in specs
    }
    influence_square = {
        spec.layer_ordinal: torch.zeros(spec.width, dtype=torch.float64)
        for spec in specs
    }
    influence_fourth = {
        spec.layer_ordinal: torch.zeros(spec.width, dtype=torch.float64)
        for spec in specs
    }
    stream_digest = _StreamMultisetDigest()

    for sequence in rows:
        if not isinstance(sequence, ActivationScoreGradientRows):
            raise TypeError(
                "rows must contain ActivationScoreGradientRows values"
            )
        sequence_hash = _sequence_digest(sequence, specs)
        stream_digest.update(sequence_hash, rows=sequence.observations)
        assert sequence.example_id is not None
        buckets_and_signs = tuple(
            _sketch_location(
                seed=config.sketch_seed,
                example_id=sequence.example_id,
                logical_position=int(position),
                sketch_size=config.sketch_size,
            )
            for position in sequence.logical_positions.tolist()
        )
        for spec in specs:
            activations = sequence.activations[
                spec.activation_site
            ].to(torch.float64)
            gradients = sequence.score_gradients[
                spec.activation_site
            ].to(torch.float64)
            influence = activations * gradients
            activation_square[spec.layer_ordinal].add_(
                activations.square().sum(dim=0)
            )
            activation_fourth[spec.layer_ordinal].add_(
                activations.pow(4).sum(dim=0)
            )
            influence_square[spec.layer_ordinal].add_(
                influence.square().sum(dim=0)
            )
            influence_fourth[spec.layer_ordinal].add_(
                influence.pow(4).sum(dim=0)
            )
            for row_index, (bucket, sign) in enumerate(buckets_and_signs):
                activation_sketches[spec.layer_ordinal][
                    :, bucket
                ].add_(activations[row_index], alpha=sign)
                influence_sketches[spec.layer_ordinal][
                    :, bucket
                ].add_(influence[row_index], alpha=sign)

    if stream_digest.count == 0:
        raise ValueError("cannot build a cross-block sketch from no sequences")

    modes: list[CrossBlockModeSketch] = []
    for spec in specs:
        ranks = _derived_fisher_ranks(
            influence_square[spec.layer_ordinal]
        )
        for index in range(spec.width):
            key = spec.mode_key(index, fisher_rank=ranks[index])
            modes.append(
                CrossBlockModeSketch(
                    key=key,
                    observations=stream_digest.row_count,
                    activation_square_sum=float(
                        activation_square[spec.layer_ordinal][index].item()
                    ),
                    activation_fourth_sum=float(
                        activation_fourth[spec.layer_ordinal][index].item()
                    ),
                    influence_square_sum=float(
                        influence_square[spec.layer_ordinal][index].item()
                    ),
                    influence_fourth_sum=float(
                        influence_fourth[spec.layer_ordinal][index].item()
                    ),
                    activation_sketch=activation_sketches[
                        spec.layer_ordinal
                    ][index],
                    influence_sketch=influence_sketches[
                        spec.layer_ordinal
                    ][index],
                )
            )
    mode_tuple = tuple(modes)
    pool, edges = _shortlist_edges(mode_tuple, specs, config)
    temporary = {
        "artifact_kind": _SKETCH_KIND,
        "format_version": _FORMAT_VERSION,
        "contains_corpus_rows": False,
        "discovery_only": True,
        "authorizes_execution": False,
        "authorizes_b": False,
        "provenance": provenance.metadata(),
        "layer_specs": tuple(spec.metadata() for spec in specs),
        "config": config.metadata(),
        "sequences": stream_digest.count,
        "observations": stream_digest.row_count,
        "row_stream_sha256": stream_digest.hexdigest(),
        "modes": tuple(mode.metadata() for mode in mode_tuple),
        "pool_mode_keys": tuple(key.metadata() for key in pool),
        "proxy_edges": tuple(edge.metadata() for edge in edges),
    }
    return CrossBlockDiscoverySketch(
        provenance=provenance,
        layer_specs=specs,
        config=config,
        sequences=stream_digest.count,
        observations=stream_digest.row_count,
        row_stream_sha256=stream_digest.hexdigest(),
        modes=mode_tuple,
        pool_mode_keys=pool,
        proxy_edges=edges,
        artifact_sha256=_json_sha256(temporary),
    )


def rescreen_cross_block_discovery_sketch(
    sketch: CrossBlockDiscoverySketch,
    *,
    config: CrossBlockSketchConfig,
) -> CrossBlockDiscoverySketch:
    """Rebuild only the proxy shortlist from authenticated mode sketches.

    The expensive activation/score-gradient stream is unchanged.  This is
    useful when a development run broadens the eligible per-layer pool or the
    proxy neighborhood while preserving the original CountSketch basis and
    exact replay digest.  Changing the sketch width or seed would change that
    basis and therefore requires a new first pass.
    """

    if not isinstance(sketch, CrossBlockDiscoverySketch):
        raise TypeError("sketch must be a CrossBlockDiscoverySketch")
    if not isinstance(config, CrossBlockSketchConfig):
        raise TypeError("config must be a CrossBlockSketchConfig")
    if (
        config.sketch_size != sketch.config.sketch_size
        or config.sketch_seed != sketch.config.sketch_seed
    ):
        raise ValueError(
            "rescreening must preserve the authenticated sketch size and seed"
        )
    pool, edges = _shortlist_edges(
        sketch.modes,
        sketch.layer_specs,
        config,
    )
    temporary = {
        "artifact_kind": _SKETCH_KIND,
        "format_version": _FORMAT_VERSION,
        "contains_corpus_rows": False,
        "discovery_only": True,
        "authorizes_execution": False,
        "authorizes_b": False,
        "provenance": sketch.provenance.metadata(),
        "layer_specs": tuple(
            spec.metadata() for spec in sketch.layer_specs
        ),
        "config": config.metadata(),
        "sequences": sketch.sequences,
        "observations": sketch.observations,
        "row_stream_sha256": sketch.row_stream_sha256,
        "modes": tuple(mode.metadata() for mode in sketch.modes),
        "pool_mode_keys": tuple(key.metadata() for key in pool),
        "proxy_edges": tuple(edge.metadata() for edge in edges),
    }
    return CrossBlockDiscoverySketch(
        provenance=sketch.provenance,
        layer_specs=sketch.layer_specs,
        config=config,
        sequences=sketch.sequences,
        observations=sketch.observations,
        row_stream_sha256=sketch.row_stream_sha256,
        modes=sketch.modes,
        pool_mode_keys=pool,
        proxy_edges=edges,
        artifact_sha256=_json_sha256(temporary),
    )


@dataclass(frozen=True, slots=True)
class CrossBlockExactCriteria:
    """Fit-split criteria for retaining discovery hypotheses.

    Passing these criteria still does not authorize an executor.  Thresholds
    must be frozen before a held-out execution guard is consumed.
    """

    min_row_signed_correlation: float = 0.9
    min_sequence_signed_correlation: float = 0.9
    min_energy_balance: float = 0.1
    min_absolute_activation_correlation: float = 0.9
    max_activation_rank_one_tail_fraction: float = 0.05
    min_coactivity: float = 0.5
    max_endpoint_activation_density: float = 0.5
    max_endpoint_influence_density: float = 0.5
    fold_count: int = 0
    min_fold_signed_correlation: float = 0.8
    relative_energy_floor: float = 1e-12

    def __post_init__(self) -> None:
        for label, value in (
            (
                "min_row_signed_correlation",
                self.min_row_signed_correlation,
            ),
            (
                "min_sequence_signed_correlation",
                self.min_sequence_signed_correlation,
            ),
            ("min_energy_balance", self.min_energy_balance),
            (
                "min_absolute_activation_correlation",
                self.min_absolute_activation_correlation,
            ),
            (
                "max_activation_rank_one_tail_fraction",
                self.max_activation_rank_one_tail_fraction,
            ),
            ("min_coactivity", self.min_coactivity),
            (
                "max_endpoint_activation_density",
                self.max_endpoint_activation_density,
            ),
            (
                "max_endpoint_influence_density",
                self.max_endpoint_influence_density,
            ),
            (
                "min_fold_signed_correlation",
                self.min_fold_signed_correlation,
            ),
        ):
            if (
                not isinstance(value, float)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{label} must lie in [0, 1]")
        if type(self.fold_count) is not int or (
            self.fold_count not in (0, 1) and self.fold_count < 2
        ):
            raise ValueError("fold_count must be 0, 1, or at least 2")
        if (
            not isinstance(self.relative_energy_floor, float)
            or not math.isfinite(self.relative_energy_floor)
            or not 0.0 < self.relative_energy_floor < 1.0
        ):
            raise ValueError("relative_energy_floor must lie in (0, 1)")

    @property
    def uses_fold_stability(self) -> bool:
        return self.fold_count >= 2

    def metadata(self) -> dict[str, object]:
        return {
            "min_row_signed_correlation": (
                self.min_row_signed_correlation
            ),
            "min_sequence_signed_correlation": (
                self.min_sequence_signed_correlation
            ),
            "min_energy_balance": self.min_energy_balance,
            "min_absolute_activation_correlation": (
                self.min_absolute_activation_correlation
            ),
            "max_activation_rank_one_tail_fraction": (
                self.max_activation_rank_one_tail_fraction
            ),
            "min_coactivity": self.min_coactivity,
            "max_endpoint_activation_density": (
                self.max_endpoint_activation_density
            ),
            "max_endpoint_influence_density": (
                self.max_endpoint_influence_density
            ),
            "fold_count": self.fold_count,
            "min_fold_signed_correlation": (
                self.min_fold_signed_correlation
            ),
            "relative_energy_floor": self.relative_energy_floor,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockExactCriteria:
        expected = {
            "min_row_signed_correlation",
            "min_sequence_signed_correlation",
            "min_energy_balance",
            "min_absolute_activation_correlation",
            "max_activation_rank_one_tail_fraction",
            "min_coactivity",
            "max_endpoint_activation_density",
            "max_endpoint_influence_density",
            "fold_count",
            "min_fold_signed_correlation",
            "relative_energy_floor",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block exact criteria fields are invalid")
        return cls(
            min_row_signed_correlation=float(
                state["min_row_signed_correlation"]
            ),
            min_sequence_signed_correlation=float(
                state["min_sequence_signed_correlation"]
            ),
            min_energy_balance=float(state["min_energy_balance"]),
            min_absolute_activation_correlation=float(
                state["min_absolute_activation_correlation"]
            ),
            max_activation_rank_one_tail_fraction=float(
                state["max_activation_rank_one_tail_fraction"]
            ),
            min_coactivity=float(state["min_coactivity"]),
            max_endpoint_activation_density=float(
                state["max_endpoint_activation_density"]
            ),
            max_endpoint_influence_density=float(
                state["max_endpoint_influence_density"]
            ),
            fold_count=int(state["fold_count"]),
            min_fold_signed_correlation=float(
                state["min_fold_signed_correlation"]
            ),
            relative_energy_floor=float(state["relative_energy_floor"]),
        )


def _symmetric_2x2_eigenvalues(matrix: Tensor) -> tuple[float, float]:
    if matrix.shape != (2, 2):
        raise ValueError("pair moment matrix must have shape [2, 2]")
    symmetric = (matrix + matrix.T) * 0.5
    a = float(symmetric[0, 0].item())
    b = float(symmetric[0, 1].item())
    c = float(symmetric[1, 1].item())
    center = 0.5 * (a + c)
    radius = 0.5 * math.hypot(a - c, 2.0 * b)
    lower = center - radius
    upper = center + radius
    tolerance = (
        torch.finfo(torch.float64).eps
        * max(abs(a), abs(b), abs(c), 1.0)
        * 512.0
    )
    if lower < -tolerance:
        raise ValueError("pair moment matrix is not positive semidefinite")
    return max(lower, 0.0), max(upper, 0.0)


def _correlation(matrix: Tensor, *, floor: float) -> float | None:
    first = float(matrix[0, 0].item())
    second = float(matrix[1, 1].item())
    if first <= floor or second <= floor:
        return None
    value = float(matrix[0, 1].item()) / math.sqrt(first * second)
    tolerance = torch.finfo(torch.float64).eps * 1024.0
    if not -1.0 - tolerance <= value <= 1.0 + tolerance:
        raise ValueError("pair moments violate the correlation bound")
    return min(max(value, -1.0), 1.0)


def _energy_balance(matrix: Tensor, *, floor: float) -> float:
    first = float(matrix[0, 0].item())
    second = float(matrix[1, 1].item())
    maximum = max(first, second)
    if maximum <= floor:
        return 0.0
    return min(first, second) / maximum


def _tail_fraction(matrix: Tensor, *, floor: float) -> float:
    lower, upper = _symmetric_2x2_eigenvalues(matrix)
    total = lower + upper
    return 0.0 if total <= floor else lower / total


def _fold_index(example_id: str, fold_count: int) -> int:
    digest = hashlib.sha256()
    digest.update(b"fisher_graph.cross_block_fold.v1\0")
    digest.update(example_id.encode())
    return int.from_bytes(digest.digest()[:8], "little") % fold_count


@dataclass(slots=True)
class _ExactPairAccumulator:
    row_fisher_gram: Tensor
    sequence_fisher_gram: Tensor
    activation_gram: Tensor
    coactivity_cross_sum: float
    row_observations: int
    sequence_observations: int
    fold_sequence_fisher_grams: tuple[Tensor, ...]
    fold_sequence_counts: list[int]

    @classmethod
    def create(cls, fold_count: int) -> _ExactPairAccumulator:
        return cls(
            row_fisher_gram=torch.zeros(2, 2, dtype=torch.float64),
            sequence_fisher_gram=torch.zeros(2, 2, dtype=torch.float64),
            activation_gram=torch.zeros(2, 2, dtype=torch.float64),
            coactivity_cross_sum=0.0,
            row_observations=0,
            sequence_observations=0,
            fold_sequence_fisher_grams=tuple(
                torch.zeros(2, 2, dtype=torch.float64)
                for _ in range(fold_count)
            ),
            fold_sequence_counts=[0] * fold_count,
        )

    def update(
        self,
        first_activation: Tensor,
        first_influence: Tensor,
        second_activation: Tensor,
        second_influence: Tensor,
        *,
        fold: int | None,
    ) -> None:
        activations = torch.stack(
            (first_activation, second_activation),
            dim=1,
        ).to(torch.float64)
        influences = torch.stack(
            (first_influence, second_influence),
            dim=1,
        ).to(torch.float64)
        self.activation_gram.add_(activations.T @ activations)
        self.row_fisher_gram.add_(influences.T @ influences)
        self.coactivity_cross_sum += float(
            (first_influence * second_influence).abs().sum().item()
        )
        sequence_influence = influences.sum(dim=0)
        sequence_outer = torch.outer(
            sequence_influence,
            sequence_influence,
        )
        self.sequence_fisher_gram.add_(sequence_outer)
        self.row_observations += influences.shape[0]
        self.sequence_observations += 1
        if fold is not None:
            self.fold_sequence_fisher_grams[fold].add_(sequence_outer)
            self.fold_sequence_counts[fold] += 1


_CLASSIFICATIONS = {
    "static_merge_hypothesis",
    "zero_energy",
    "negative_correlation",
    "noncoactive",
    "proxy_only_dissimilar",
    "endpoint_density_too_high",
    "energy_imbalanced",
    "activation_not_rank_one",
    "fold_unstable",
}


@dataclass(frozen=True, slots=True)
class CrossBlockPairEvidence:
    """Exact shortlist moments and a fail-closed discovery classification."""

    first: ModeKey
    second: ModeKey
    proxy_signed_influence_correlation: float
    first_activation_density: float
    second_activation_density: float
    first_influence_density: float
    second_influence_density: float
    row_fisher_gram: Tensor
    sequence_fisher_gram: Tensor
    activation_gram: Tensor
    row_observations: int
    sequence_observations: int
    coactivity_cross_sum: float
    fold_sequence_fisher_grams: tuple[Tensor, ...]
    fold_sequence_counts: tuple[int, ...]
    row_signed_influence_correlation: float | None
    sequence_signed_influence_correlation: float | None
    energy_balance: float
    absolute_activation_correlation: float | None
    activation_rank_one_tail_fraction: float
    coactivity: float
    fold_signed_influence_correlations: tuple[float | None, ...]
    classification: str
    priority_max_loss: float
    priority_sum_loss: float
    discovery_only: bool = True
    authorizes_static_merge: bool = False
    authorizes_execution: bool = False
    authorizes_b: bool = False

    def __post_init__(self) -> None:
        if (self.first, self.second) != _canonical_pair(
            self.first,
            self.second,
        ):
            raise ValueError("evidence endpoints must use canonical order")
        if self.first.layer_ordinal == self.second.layer_ordinal:
            raise ValueError("evidence must connect distinct layers")
        if self.classification not in _CLASSIFICATIONS:
            raise ValueError("unknown cross-block pair classification")
        for label, value in (
            ("row_fisher_gram", self.row_fisher_gram),
            ("sequence_fisher_gram", self.sequence_fisher_gram),
            ("activation_gram", self.activation_gram),
        ):
            if value.shape != (2, 2):
                raise ValueError(f"{label} must have shape [2, 2]")
            _tensor_sha256(value, label=label)
            if not _tensor_close(value, value.T):
                raise ValueError(f"{label} must be symmetric")
            _symmetric_2x2_eigenvalues(value)
            object.__setattr__(self, label, value.detach().clone())
        if (
            type(self.row_observations) is not int
            or self.row_observations <= 0
            or type(self.sequence_observations) is not int
            or not 0 < self.sequence_observations <= self.row_observations
        ):
            raise ValueError("evidence observation counts are invalid")
        if (
            not isinstance(self.coactivity_cross_sum, float)
            or not math.isfinite(self.coactivity_cross_sum)
            or self.coactivity_cross_sum < 0.0
        ):
            raise ValueError("coactivity_cross_sum is invalid")
        if (
            type(self.fold_sequence_fisher_grams) is not tuple
            or type(self.fold_sequence_counts) is not tuple
            or len(self.fold_sequence_fisher_grams)
            != len(self.fold_sequence_counts)
            or len(self.fold_sequence_fisher_grams)
            != len(self.fold_signed_influence_correlations)
            or any(
                type(count) is not int or count < 0
                for count in self.fold_sequence_counts
            )
            or sum(self.fold_sequence_counts)
            not in (0, self.sequence_observations)
        ):
            raise ValueError("fold evidence is invalid")
        for matrix in self.fold_sequence_fisher_grams:
            if matrix.shape != (2, 2):
                raise ValueError("fold Fisher Grams must have shape [2, 2]")
            _tensor_sha256(matrix, label="fold_sequence_fisher_gram")
            _symmetric_2x2_eigenvalues(matrix)
        for label, value in (
            (
                "proxy_signed_influence_correlation",
                self.proxy_signed_influence_correlation,
            ),
            ("first_activation_density", self.first_activation_density),
            ("second_activation_density", self.second_activation_density),
            ("first_influence_density", self.first_influence_density),
            ("second_influence_density", self.second_influence_density),
            ("energy_balance", self.energy_balance),
            (
                "activation_rank_one_tail_fraction",
                self.activation_rank_one_tail_fraction,
            ),
            ("coactivity", self.coactivity),
            ("priority_max_loss", self.priority_max_loss),
            ("priority_sum_loss", self.priority_sum_loss),
        ):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        for value in (
            self.row_signed_influence_correlation,
            self.sequence_signed_influence_correlation,
            self.absolute_activation_correlation,
            *self.fold_signed_influence_correlations,
        ):
            if value is not None and (
                not isinstance(value, float)
                or not math.isfinite(value)
                or not -1.0 <= value <= 1.0
            ):
                raise ValueError("correlations must lie in [-1, 1]")
        if (
            not -1.0 <= self.proxy_signed_influence_correlation <= 1.0
            or not 0.0 <= self.first_activation_density <= 1.0
            or not 0.0 <= self.second_activation_density <= 1.0
            or not 0.0 <= self.first_influence_density <= 1.0
            or not 0.0 <= self.second_influence_density <= 1.0
            or not 0.0 <= self.energy_balance <= 1.0
            or not 0.0
            <= self.activation_rank_one_tail_fraction
            <= 0.5 + 1e-12
            or not 0.0 <= self.coactivity <= 1.0 + 1e-12
            or self.priority_max_loss < 0.0
            or self.priority_sum_loss < 0.0
        ):
            raise ValueError("pair evidence metrics are out of range")
        if (
            not self.discovery_only
            or self.authorizes_static_merge
            or self.authorizes_execution
            or self.authorizes_b
        ):
            raise ValueError("pair evidence cannot authorize execution")

    @property
    def endpoints(self) -> tuple[ModeKey, ModeKey]:
        return self.first, self.second

    @property
    def is_static_merge_hypothesis(self) -> bool:
        return self.classification == "static_merge_hypothesis"

    @property
    def max_endpoint_activation_density(self) -> float:
        return max(
            self.first_activation_density,
            self.second_activation_density,
        )

    @property
    def max_endpoint_influence_density(self) -> float:
        return max(
            self.first_influence_density,
            self.second_influence_density,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "first": self.first.metadata(),
            "second": self.second.metadata(),
            "proxy_signed_influence_correlation": (
                self.proxy_signed_influence_correlation
            ),
            "density_scope": "valid_row_effective_support",
            "first_activation_density": self.first_activation_density,
            "second_activation_density": self.second_activation_density,
            "first_influence_density": self.first_influence_density,
            "second_influence_density": self.second_influence_density,
            "max_endpoint_activation_density": (
                self.max_endpoint_activation_density
            ),
            "max_endpoint_influence_density": (
                self.max_endpoint_influence_density
            ),
            "row_fisher_gram_sha256": _tensor_sha256(
                self.row_fisher_gram,
                label="row_fisher_gram",
            ),
            "sequence_fisher_gram_sha256": _tensor_sha256(
                self.sequence_fisher_gram,
                label="sequence_fisher_gram",
            ),
            "activation_gram_sha256": _tensor_sha256(
                self.activation_gram,
                label="activation_gram",
            ),
            "row_observations": self.row_observations,
            "sequence_observations": self.sequence_observations,
            "coactivity_cross_sum": self.coactivity_cross_sum,
            "fold_sequence_fisher_gram_sha256": tuple(
                _tensor_sha256(
                    value,
                    label="fold_sequence_fisher_gram",
                )
                for value in self.fold_sequence_fisher_grams
            ),
            "fold_sequence_counts": self.fold_sequence_counts,
            "row_signed_influence_correlation": (
                self.row_signed_influence_correlation
            ),
            "sequence_signed_influence_correlation": (
                self.sequence_signed_influence_correlation
            ),
            "energy_balance": self.energy_balance,
            "absolute_activation_correlation": (
                self.absolute_activation_correlation
            ),
            "activation_rank_one_tail_fraction": (
                self.activation_rank_one_tail_fraction
            ),
            "coactivity": self.coactivity,
            "fold_signed_influence_correlations": (
                self.fold_signed_influence_correlations
            ),
            "classification": self.classification,
            "priority_max_loss": self.priority_max_loss,
            "priority_sum_loss": self.priority_sum_loss,
            "discovery_only": self.discovery_only,
            "authorizes_static_merge": self.authorizes_static_merge,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_b": self.authorizes_b,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "first": self.first.metadata(),
            "second": self.second.metadata(),
            "proxy_signed_influence_correlation": (
                self.proxy_signed_influence_correlation
            ),
            "first_activation_density": self.first_activation_density,
            "second_activation_density": self.second_activation_density,
            "first_influence_density": self.first_influence_density,
            "second_influence_density": self.second_influence_density,
            "row_fisher_gram": self.row_fisher_gram.clone(),
            "sequence_fisher_gram": self.sequence_fisher_gram.clone(),
            "activation_gram": self.activation_gram.clone(),
            "row_observations": self.row_observations,
            "sequence_observations": self.sequence_observations,
            "coactivity_cross_sum": self.coactivity_cross_sum,
            "fold_sequence_fisher_grams": tuple(
                value.clone()
                for value in self.fold_sequence_fisher_grams
            ),
            "fold_sequence_counts": self.fold_sequence_counts,
            "row_signed_influence_correlation": (
                self.row_signed_influence_correlation
            ),
            "sequence_signed_influence_correlation": (
                self.sequence_signed_influence_correlation
            ),
            "energy_balance": self.energy_balance,
            "absolute_activation_correlation": (
                self.absolute_activation_correlation
            ),
            "activation_rank_one_tail_fraction": (
                self.activation_rank_one_tail_fraction
            ),
            "coactivity": self.coactivity,
            "fold_signed_influence_correlations": (
                self.fold_signed_influence_correlations
            ),
            "classification": self.classification,
            "priority_max_loss": self.priority_max_loss,
            "priority_sum_loss": self.priority_sum_loss,
            "discovery_only": self.discovery_only,
            "authorizes_static_merge": self.authorizes_static_merge,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_b": self.authorizes_b,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockPairEvidence:
        expected = {
            "first",
            "second",
            "proxy_signed_influence_correlation",
            "first_activation_density",
            "second_activation_density",
            "first_influence_density",
            "second_influence_density",
            "row_fisher_gram",
            "sequence_fisher_gram",
            "activation_gram",
            "row_observations",
            "sequence_observations",
            "coactivity_cross_sum",
            "fold_sequence_fisher_grams",
            "fold_sequence_counts",
            "row_signed_influence_correlation",
            "sequence_signed_influence_correlation",
            "energy_balance",
            "absolute_activation_correlation",
            "activation_rank_one_tail_fraction",
            "coactivity",
            "fold_signed_influence_correlations",
            "classification",
            "priority_max_loss",
            "priority_sum_loss",
            "discovery_only",
            "authorizes_static_merge",
            "authorizes_execution",
            "authorizes_b",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block pair evidence fields are invalid")
        if not isinstance(state["first"], Mapping) or not isinstance(
            state["second"], Mapping
        ):
            raise TypeError("evidence endpoint states must be mappings")
        for label in (
            "row_fisher_gram",
            "sequence_fisher_gram",
            "activation_gram",
        ):
            if not isinstance(state[label], Tensor):
                raise TypeError(f"{label} must be a Tensor")
        fold_grams = state["fold_sequence_fisher_grams"]
        fold_counts = state["fold_sequence_counts"]
        fold_correlations = state["fold_signed_influence_correlations"]
        if (
            not isinstance(fold_grams, tuple)
            or not isinstance(fold_counts, tuple)
            or not isinstance(fold_correlations, tuple)
        ):
            raise TypeError("fold evidence state must use tuples")
        return cls(
            first=ModeKey.from_state_dict(state["first"]),
            second=ModeKey.from_state_dict(state["second"]),
            proxy_signed_influence_correlation=float(
                state["proxy_signed_influence_correlation"]
            ),
            first_activation_density=float(
                state["first_activation_density"]
            ),
            second_activation_density=float(
                state["second_activation_density"]
            ),
            first_influence_density=float(
                state["first_influence_density"]
            ),
            second_influence_density=float(
                state["second_influence_density"]
            ),
            row_fisher_gram=state["row_fisher_gram"],
            sequence_fisher_gram=state["sequence_fisher_gram"],
            activation_gram=state["activation_gram"],
            row_observations=int(state["row_observations"]),
            sequence_observations=int(state["sequence_observations"]),
            coactivity_cross_sum=float(state["coactivity_cross_sum"]),
            fold_sequence_fisher_grams=tuple(fold_grams),
            fold_sequence_counts=tuple(
                int(value) for value in fold_counts
            ),
            row_signed_influence_correlation=(
                None
                if state["row_signed_influence_correlation"] is None
                else float(state["row_signed_influence_correlation"])
            ),
            sequence_signed_influence_correlation=(
                None
                if state["sequence_signed_influence_correlation"] is None
                else float(state["sequence_signed_influence_correlation"])
            ),
            energy_balance=float(state["energy_balance"]),
            absolute_activation_correlation=(
                None
                if state["absolute_activation_correlation"] is None
                else float(state["absolute_activation_correlation"])
            ),
            activation_rank_one_tail_fraction=float(
                state["activation_rank_one_tail_fraction"]
            ),
            coactivity=float(state["coactivity"]),
            fold_signed_influence_correlations=tuple(
                None if value is None else float(value)
                for value in fold_correlations
            ),
            classification=str(state["classification"]),
            priority_max_loss=float(state["priority_max_loss"]),
            priority_sum_loss=float(state["priority_sum_loss"]),
            discovery_only=bool(state["discovery_only"]),
            authorizes_static_merge=bool(
                state["authorizes_static_merge"]
            ),
            authorizes_execution=bool(state["authorizes_execution"]),
            authorizes_b=bool(state["authorizes_b"]),
        )


def _pair_evidence(
    edge: CrossBlockProxyEdge,
    accumulator: _ExactPairAccumulator,
    criteria: CrossBlockExactCriteria,
    first_mode: CrossBlockModeSketch,
    second_mode: CrossBlockModeSketch,
) -> CrossBlockPairEvidence:
    row_gram = accumulator.row_fisher_gram
    sequence_gram = accumulator.sequence_fisher_gram
    activation_gram = accumulator.activation_gram
    scale = max(
        float(row_gram.diagonal().max().item()),
        float(sequence_gram.diagonal().max().item()),
        float(activation_gram.diagonal().max().item()),
        float(torch.finfo(torch.float64).tiny),
    )
    floor = max(
        scale * criteria.relative_energy_floor,
        float(torch.finfo(torch.float64).tiny),
    )
    row_correlation = _correlation(row_gram, floor=floor)
    sequence_correlation = _correlation(sequence_gram, floor=floor)
    activation_correlation = _correlation(
        activation_gram,
        floor=floor,
    )
    absolute_activation_correlation = (
        None
        if activation_correlation is None
        else abs(activation_correlation)
    )
    balance = min(
        _energy_balance(row_gram, floor=floor),
        _energy_balance(sequence_gram, floor=floor),
    )
    activation_tail = _tail_fraction(
        activation_gram,
        floor=floor,
    )
    coactivity_denominator = math.sqrt(
        max(float(row_gram[0, 0].item()), 0.0)
        * max(float(row_gram[1, 1].item()), 0.0)
    )
    coactivity = (
        0.0
        if coactivity_denominator <= floor
        else min(
            max(
                accumulator.coactivity_cross_sum
                / coactivity_denominator,
                0.0,
            ),
            1.0,
        )
    )
    fold_correlations = tuple(
        _correlation(matrix, floor=floor)
        if count > 0
        else None
        for matrix, count in zip(
            accumulator.fold_sequence_fisher_grams,
            accumulator.fold_sequence_counts,
            strict=True,
        )
    )

    if row_correlation is None or sequence_correlation is None:
        classification = "zero_energy"
    elif row_correlation < 0.0 or sequence_correlation < 0.0:
        classification = "negative_correlation"
    elif (
        sequence_correlation
        >= criteria.min_sequence_signed_correlation
        and coactivity < criteria.min_coactivity
    ):
        # Same sequence-level behavior can arise at disjoint token rows.  It
        # is a conditional/shared-slot lead, not a static generator merge.
        classification = "noncoactive"
    elif (
        row_correlation < criteria.min_row_signed_correlation
        or sequence_correlation
        < criteria.min_sequence_signed_correlation
    ):
        classification = "proxy_only_dissimilar"
    elif (
        max(
            first_mode.activation_density,
            second_mode.activation_density,
        )
        > criteria.max_endpoint_activation_density
        or max(
            first_mode.influence_density,
            second_mode.influence_density,
        )
        > criteria.max_endpoint_influence_density
    ):
        classification = "endpoint_density_too_high"
    elif balance < criteria.min_energy_balance:
        classification = "energy_imbalanced"
    elif (
        absolute_activation_correlation is None
        or absolute_activation_correlation
        < criteria.min_absolute_activation_correlation
        or activation_tail
        > criteria.max_activation_rank_one_tail_fraction
    ):
        classification = "activation_not_rank_one"
    elif coactivity < criteria.min_coactivity:
        classification = "noncoactive"
    elif criteria.uses_fold_stability and any(
        value is None
        or value < criteria.min_fold_signed_correlation
        for value in fold_correlations
    ):
        classification = "fold_unstable"
    else:
        classification = "static_merge_hypothesis"

    losses = (
        1.0 if row_correlation is None else 1.0 - row_correlation,
        (
            1.0
            if sequence_correlation is None
            else 1.0 - sequence_correlation
        ),
        1.0 - balance,
        (
            1.0
            if absolute_activation_correlation is None
            else 1.0 - absolute_activation_correlation
        ),
        min(2.0 * activation_tail, 1.0),
        1.0 - coactivity,
        max(
            first_mode.activation_density,
            second_mode.activation_density,
        ),
        max(
            first_mode.influence_density,
            second_mode.influence_density,
        ),
    )
    if criteria.uses_fold_stability:
        fold_loss = max(
            1.0 if value is None else 1.0 - value
            for value in fold_correlations
        )
        losses = (*losses, fold_loss)
    return CrossBlockPairEvidence(
        first=edge.first,
        second=edge.second,
        proxy_signed_influence_correlation=(
            edge.proxy_signed_influence_correlation
        ),
        first_activation_density=float(first_mode.activation_density),
        second_activation_density=float(second_mode.activation_density),
        first_influence_density=float(first_mode.influence_density),
        second_influence_density=float(second_mode.influence_density),
        row_fisher_gram=row_gram,
        sequence_fisher_gram=sequence_gram,
        activation_gram=activation_gram,
        row_observations=accumulator.row_observations,
        sequence_observations=accumulator.sequence_observations,
        coactivity_cross_sum=float(accumulator.coactivity_cross_sum),
        fold_sequence_fisher_grams=tuple(
            value.clone()
            for value in accumulator.fold_sequence_fisher_grams
        ),
        fold_sequence_counts=tuple(accumulator.fold_sequence_counts),
        row_signed_influence_correlation=row_correlation,
        sequence_signed_influence_correlation=sequence_correlation,
        energy_balance=float(balance),
        absolute_activation_correlation=absolute_activation_correlation,
        activation_rank_one_tail_fraction=float(activation_tail),
        coactivity=float(coactivity),
        fold_signed_influence_correlations=fold_correlations,
        classification=classification,
        priority_max_loss=float(max(losses)),
        priority_sum_loss=float(math.fsum(losses)),
    )


def _select_endpoint_disjoint(
    evidence: tuple[CrossBlockPairEvidence, ...],
) -> tuple[CrossBlockPairEvidence, ...]:
    qualifying = sorted(
        (
            value
            for value in evidence
            if value.is_static_merge_hypothesis
        ),
        key=lambda value: (
            value.priority_max_loss,
            value.priority_sum_loss,
            value.first,
            value.second,
        ),
    )
    used: set[ModeKey] = set()
    selected: list[CrossBlockPairEvidence] = []
    for value in qualifying:
        if value.first in used or value.second in used:
            continue
        used.add(value.first)
        used.add(value.second)
        selected.append(value)
    return tuple(
        sorted(selected, key=lambda value: value.endpoints)
    )


def _fold_assignment_sha256(
    fold_assignment: Mapping[str, int] | None,
    *,
    fold_count: int,
) -> str:
    if fold_assignment is None:
        return _json_sha256(
            {
                "kind": "stable_example_id_hash",
                "fold_count": fold_count,
            },
            domain=b"fisher_graph.cross_block_fold_assignment.v1\0",
        )
    if not isinstance(fold_assignment, Mapping):
        raise TypeError("fold_assignment must be a mapping or None")
    items = []
    for example_id, fold in fold_assignment.items():
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(
                "fold_assignment keys must be nonempty example ids"
            )
        if (
            type(fold) is not int
            or fold_count < 2
            or not 0 <= fold < fold_count
        ):
            raise ValueError(
                "fold_assignment values must lie inside fold_count"
            )
        items.append((example_id, fold))
    return _json_sha256(
        {
            "kind": "caller_mapping",
            "fold_count": fold_count,
            "items": tuple(sorted(items)),
        },
        domain=b"fisher_graph.cross_block_fold_assignment.v1\0",
    )


@dataclass(frozen=True, slots=True)
class CrossBlockDiscoveryResult:
    """Exact, selected discovery hypotheses with no execution authority."""

    sketch_artifact_sha256: str
    provenance: CrossBlockDiscoveryProvenance
    layer_specs: tuple[CrossBlockLayerSpec, ...]
    criteria: CrossBlockExactCriteria
    fold_assignment_sha256: str
    sequences: int
    observations: int
    row_stream_sha256: str
    evidence: tuple[CrossBlockPairEvidence, ...]
    selected_hypotheses: tuple[CrossBlockPairEvidence, ...]
    artifact_sha256: str
    artifact_kind: str = _RESULT_KIND
    format_version: int = _FORMAT_VERSION
    contains_corpus_rows: bool = False
    discovery_only: bool = True
    authorizes_static_merge: bool = False
    authorizes_execution: bool = False
    authorizes_b: bool = False

    def __post_init__(self) -> None:
        _require_sha256(
            self.sketch_artifact_sha256,
            label="sketch_artifact_sha256",
        )
        _require_sha256(
            self.fold_assignment_sha256,
            label="fold_assignment_sha256",
        )
        _require_sha256(
            self.row_stream_sha256,
            label="row_stream_sha256",
        )
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if not isinstance(self.provenance, CrossBlockDiscoveryProvenance):
            raise TypeError("result provenance is invalid")
        if _validated_layer_specs(self.layer_specs) != self.layer_specs:
            raise ValueError("result layer specs are not canonical")
        if not isinstance(self.criteria, CrossBlockExactCriteria):
            raise TypeError("result criteria are invalid")
        if (
            type(self.sequences) is not int
            or self.sequences <= 0
            or type(self.observations) is not int
            or self.observations < self.sequences
        ):
            raise ValueError("result stream counts are invalid")
        if (
            type(self.evidence) is not tuple
            or tuple(
                sorted(self.evidence, key=lambda value: value.endpoints)
            )
            != self.evidence
            or len({value.endpoints for value in self.evidence})
            != len(self.evidence)
        ):
            raise ValueError("result evidence is not canonical")
        expected_selected = _select_endpoint_disjoint(self.evidence)
        if (
            type(self.selected_hypotheses) is not tuple
            or len(expected_selected) != len(self.selected_hypotheses)
            or any(
                left.metadata() != right.metadata()
                for left, right in zip(
                    expected_selected,
                    self.selected_hypotheses,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "selected hypotheses do not match deterministic selection"
            )
        endpoints = tuple(
            key
            for value in self.selected_hypotheses
            for key in value.endpoints
        )
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("selected hypotheses reuse a mode endpoint")
        if (
            self.artifact_kind != _RESULT_KIND
            or self.format_version != _FORMAT_VERSION
            or self.contains_corpus_rows
            or not self.discovery_only
            or self.authorizes_static_merge
            or self.authorizes_execution
            or self.authorizes_b
        ):
            raise ValueError("result safety metadata is invalid")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("cross-block discovery result hash mismatch")

    @property
    def selected_pairs(
        self,
    ) -> tuple[tuple[ModeKey, ModeKey], ...]:
        return tuple(
            value.endpoints for value in self.selected_hypotheses
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_corpus_rows": self.contains_corpus_rows,
            "discovery_only": self.discovery_only,
            "authorizes_static_merge": self.authorizes_static_merge,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_b": self.authorizes_b,
            "sketch_artifact_sha256": self.sketch_artifact_sha256,
            "provenance": self.provenance.metadata(),
            "layer_specs": tuple(
                spec.metadata() for spec in self.layer_specs
            ),
            "criteria": self.criteria.metadata(),
            "fold_assignment_sha256": self.fold_assignment_sha256,
            "sequences": self.sequences,
            "observations": self.observations,
            "row_stream_sha256": self.row_stream_sha256,
            "evidence": tuple(
                value.metadata() for value in self.evidence
            ),
            "selected_hypotheses": tuple(
                value.metadata() for value in self.selected_hypotheses
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload())

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "selected_pair_count": len(self.selected_hypotheses),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_corpus_rows": self.contains_corpus_rows,
            "discovery_only": self.discovery_only,
            "authorizes_static_merge": self.authorizes_static_merge,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_b": self.authorizes_b,
            "sketch_artifact_sha256": self.sketch_artifact_sha256,
            "provenance": self.provenance.metadata(),
            "layer_specs": tuple(
                spec.metadata() for spec in self.layer_specs
            ),
            "criteria": self.criteria.metadata(),
            "fold_assignment_sha256": self.fold_assignment_sha256,
            "sequences": self.sequences,
            "observations": self.observations,
            "row_stream_sha256": self.row_stream_sha256,
            "evidence": tuple(
                value.state_dict() for value in self.evidence
            ),
            "selected_hypotheses": tuple(
                value.state_dict() for value in self.selected_hypotheses
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockDiscoveryResult:
        expected = {
            "artifact_kind",
            "format_version",
            "contains_corpus_rows",
            "discovery_only",
            "authorizes_static_merge",
            "authorizes_execution",
            "authorizes_b",
            "sketch_artifact_sha256",
            "provenance",
            "layer_specs",
            "criteria",
            "fold_assignment_sha256",
            "sequences",
            "observations",
            "row_stream_sha256",
            "evidence",
            "selected_hypotheses",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("cross-block result state fields are invalid")
        for label in ("provenance", "criteria"):
            if not isinstance(state[label], Mapping):
                raise TypeError(f"{label} state must be a mapping")
        for label in (
            "layer_specs",
            "evidence",
            "selected_hypotheses",
        ):
            if not isinstance(state[label], tuple):
                raise TypeError(f"{label} state must be a tuple")
        return cls(
            sketch_artifact_sha256=str(
                state["sketch_artifact_sha256"]
            ),
            provenance=CrossBlockDiscoveryProvenance.from_state_dict(
                state["provenance"]
            ),
            layer_specs=tuple(
                CrossBlockLayerSpec.from_state_dict(value)
                for value in state["layer_specs"]  # type: ignore[union-attr]
            ),
            criteria=CrossBlockExactCriteria.from_state_dict(
                state["criteria"]
            ),
            fold_assignment_sha256=str(
                state["fold_assignment_sha256"]
            ),
            sequences=int(state["sequences"]),
            observations=int(state["observations"]),
            row_stream_sha256=str(state["row_stream_sha256"]),
            evidence=tuple(
                CrossBlockPairEvidence.from_state_dict(value)
                for value in state["evidence"]  # type: ignore[union-attr]
            ),
            selected_hypotheses=tuple(
                CrossBlockPairEvidence.from_state_dict(value)
                for value in state[
                    "selected_hypotheses"
                ]  # type: ignore[union-attr]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
            contains_corpus_rows=bool(state["contains_corpus_rows"]),
            discovery_only=bool(state["discovery_only"]),
            authorizes_static_merge=bool(
                state["authorizes_static_merge"]
            ),
            authorizes_execution=bool(state["authorizes_execution"]),
            authorizes_b=bool(state["authorizes_b"]),
        )


def replay_cross_block_discovery_shortlist(
    rows: Iterable[ActivationScoreGradientRows],
    *,
    sketch: CrossBlockDiscoverySketch,
    criteria: CrossBlockExactCriteria = CrossBlockExactCriteria(),
    fold_assignment: Mapping[str, int] | None = None,
) -> CrossBlockDiscoveryResult:
    """Replay only shortlisted edges into exact row/sequence moments."""

    if not isinstance(sketch, CrossBlockDiscoverySketch):
        raise TypeError("sketch must be a CrossBlockDiscoverySketch")
    if not isinstance(criteria, CrossBlockExactCriteria):
        raise TypeError("criteria must be CrossBlockExactCriteria")
    if fold_assignment is not None and not criteria.uses_fold_stability:
        raise ValueError(
            "fold_assignment requires criteria.fold_count of at least 2"
        )
    fold_assignment_sha256 = _fold_assignment_sha256(
        fold_assignment,
        fold_count=criteria.fold_count,
    )
    accumulators = {
        edge.endpoints: _ExactPairAccumulator.create(
            criteria.fold_count if criteria.uses_fold_stability else 0
        )
        for edge in sketch.proxy_edges
    }
    mode_by_key = {mode.key: mode for mode in sketch.modes}
    needed = {
        key
        for edge in sketch.proxy_edges
        for key in edge.endpoints
    }
    stream_digest = _StreamMultisetDigest()
    seen_fold_examples: set[str] = set()
    specs = sketch.layer_specs

    for sequence in rows:
        if not isinstance(sequence, ActivationScoreGradientRows):
            raise TypeError(
                "rows must contain ActivationScoreGradientRows values"
            )
        sequence_hash = _sequence_digest(sequence, specs)
        stream_digest.update(sequence_hash, rows=sequence.observations)
        assert sequence.example_id is not None
        if criteria.uses_fold_stability:
            if fold_assignment is None:
                fold = _fold_index(
                    sequence.example_id,
                    criteria.fold_count,
                )
            else:
                try:
                    fold = fold_assignment[sequence.example_id]
                except KeyError as error:
                    raise ValueError(
                        "fold_assignment does not cover replay example "
                        f"{sequence.example_id!r}"
                    ) from error
                if (
                    type(fold) is not int
                    or not 0 <= fold < criteria.fold_count
                ):
                    raise ValueError(
                        "fold_assignment contains an invalid fold"
                    )
                seen_fold_examples.add(sequence.example_id)
        else:
            fold = None

        activations_by_key: dict[ModeKey, Tensor] = {}
        influences_by_key: dict[ModeKey, Tensor] = {}
        for key in needed:
            activations = sequence.activations[key.activation_site][
                :, key.mode_index
            ].to(torch.float64)
            gradients = sequence.score_gradients[key.activation_site][
                :, key.mode_index
            ].to(torch.float64)
            activations_by_key[key] = activations
            influences_by_key[key] = activations * gradients
        for edge in sketch.proxy_edges:
            accumulator = accumulators[edge.endpoints]
            accumulator.update(
                activations_by_key[edge.first],
                influences_by_key[edge.first],
                activations_by_key[edge.second],
                influences_by_key[edge.second],
                fold=fold,
            )

    if stream_digest.count == 0:
        raise ValueError("cannot replay an empty cross-block stream")
    if (
        stream_digest.count != sketch.sequences
        or stream_digest.row_count != sketch.observations
        or stream_digest.hexdigest() != sketch.row_stream_sha256
    ):
        raise ValueError(
            "exact replay rows do not match the discovery sketch"
        )
    if fold_assignment is not None and seen_fold_examples != set(
        fold_assignment
    ):
        raise ValueError(
            "fold_assignment must exactly cover replay example ids"
        )

    evidence = tuple(
        sorted(
            (
                _pair_evidence(
                    edge,
                    accumulators[edge.endpoints],
                    criteria,
                    mode_by_key[edge.first],
                    mode_by_key[edge.second],
                )
                for edge in sketch.proxy_edges
            ),
            key=lambda value: value.endpoints,
        )
    )
    selected = _select_endpoint_disjoint(evidence)
    temporary = {
        "artifact_kind": _RESULT_KIND,
        "format_version": _FORMAT_VERSION,
        "contains_corpus_rows": False,
        "discovery_only": True,
        "authorizes_static_merge": False,
        "authorizes_execution": False,
        "authorizes_b": False,
        "sketch_artifact_sha256": sketch.artifact_sha256,
        "provenance": sketch.provenance.metadata(),
        "layer_specs": tuple(spec.metadata() for spec in specs),
        "criteria": criteria.metadata(),
        "fold_assignment_sha256": fold_assignment_sha256,
        "sequences": stream_digest.count,
        "observations": stream_digest.row_count,
        "row_stream_sha256": stream_digest.hexdigest(),
        "evidence": tuple(value.metadata() for value in evidence),
        "selected_hypotheses": tuple(
            value.metadata() for value in selected
        ),
    }
    return CrossBlockDiscoveryResult(
        sketch_artifact_sha256=sketch.artifact_sha256,
        provenance=sketch.provenance,
        layer_specs=specs,
        criteria=criteria,
        fold_assignment_sha256=fold_assignment_sha256,
        sequences=stream_digest.count,
        observations=stream_digest.row_count,
        row_stream_sha256=stream_digest.hexdigest(),
        evidence=evidence,
        selected_hypotheses=selected,
        artifact_sha256=_json_sha256(temporary),
    )


__all__ = [
    "CrossBlockDiscoveryProvenance",
    "CrossBlockDiscoveryResult",
    "CrossBlockDiscoverySketch",
    "CrossBlockExactCriteria",
    "CrossBlockLayerSpec",
    "CrossBlockModeSketch",
    "CrossBlockPairEvidence",
    "CrossBlockProxyEdge",
    "CrossBlockSketchConfig",
    "ModeKey",
    "build_cross_block_discovery_sketch",
    "rescreen_cross_block_discovery_sketch",
    "replay_cross_block_discovery_shortlist",
]
