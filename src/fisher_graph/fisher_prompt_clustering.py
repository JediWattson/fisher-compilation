"""Prompt-conditioned Fisher clustering for scalar model modes.

For one prompt ``x`` and scalar mode ``j``, the signed gate effect is

``r_j(x) = sum_t z_tj * d NLL(x) / d z_tj``.

The outer product ``r(x) r(x).T`` is the prompt-conditioned empirical Fisher
contribution in this mode basis.  This module clusters *mode columns* across
prompts, treating a column and its negative as the same axis.  The resulting
cluster signature is exact for a coordinate-selector projector:

``s_C(x) = tr(P_C r(x) r(x).T P_C) = sum_{j in C} r_j(x)^2``.

Clustering uses deterministic axial spherical k-means.  Assignment maximizes
absolute cosine similarity, records the selected sign separately, and updates
each centroid with Fisher-energy weights.  All reductions that affect cluster
identity use stable mode-catalog order, so mode chunking and input-column
permutations cannot silently change tie-breaking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
    ModeKey,
)


_CONFIG_KIND = "fisher_graph.fisher_prompt_cluster_config"
_PLAN_KIND = "fisher_graph.fisher_prompt_cluster_plan"
_FORMAT_VERSION = 1
_CONFIG_HASH_DOMAIN = b"fisher_graph.fisher_prompt_cluster_config.v1\0"
_PLAN_HASH_DOMAIN = b"fisher_graph.fisher_prompt_cluster_plan.v1\0"
_TENSOR_HASH_DOMAIN = b"fisher_graph.fisher_prompt_cluster_tensor.v1\0"
_EFFECTS_HASH_DOMAIN = b"fisher_graph.fisher_prompt_effects.v1\0"
_SCORE_DEFINITION = "prompt_sum_z_times_d_nll_dz"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _json_sha256(value: object, *, domain: bytes) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(payload)
    return digest.hexdigest()


def _tensor_sha256(
    value: Tensor,
    *,
    label: str,
    dtype: torch.dtype,
    domain: bytes = _TENSOR_HASH_DOMAIN,
) -> str:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"{label} must be a contiguous CPU {dtype} Tensor"
        )
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    canonical = value.detach()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        (
            f"{tuple(canonical.shape)}\0{str(canonical.dtype)}\0"
        ).encode("utf-8")
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _as_effect_matrix(value: Tensor, *, label: str = "effects") -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(
            f"{label} must be a nonempty floating [prompts, modes] matrix"
        )
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError(f"{label} must contain only finite values")
    return canonical.clone()


def fisher_prompt_effects_sha256(effects: Tensor) -> str:
    """Hash the canonical prompt-by-mode effect matrix used for clustering."""

    matrix = _as_effect_matrix(effects)
    return _tensor_sha256(
        matrix,
        label="effects",
        dtype=torch.float64,
        domain=_EFFECTS_HASH_DOMAIN,
    )


def _validate_layer_specs(
    values: tuple[CrossBlockLayerSpec, ...],
) -> tuple[CrossBlockLayerSpec, ...]:
    if (
        type(values) is not tuple
        or not values
        or any(not isinstance(value, CrossBlockLayerSpec) for value in values)
    ):
        raise ValueError(
            "layer_specs must be a nonempty tuple of CrossBlockLayerSpec"
        )
    if values != tuple(
        sorted(values, key=lambda value: value.layer_ordinal)
    ):
        raise ValueError("layer_specs must be ordered by layer_ordinal")
    if len({value.layer_ordinal for value in values}) != len(values):
        raise ValueError("layer ordinals must be unique")
    if len({value.layer_id for value in values}) != len(values):
        raise ValueError("layer ids must be unique")
    return values


def _validate_mode_catalog(
    values: tuple[ModeKey, ...],
    *,
    layer_specs: tuple[CrossBlockLayerSpec, ...],
) -> tuple[ModeKey, ...]:
    if (
        type(values) is not tuple
        or not values
        or any(not isinstance(value, ModeKey) for value in values)
    ):
        raise ValueError(
            "mode_catalog must be a nonempty tuple of ModeKey values"
        )
    coordinates = tuple(
        (value.layer_ordinal, value.mode_index) for value in values
    )
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("mode catalog coordinates must be unique")
    if len(values) != len(set(values)):
        raise ValueError("mode catalog entries must be unique")
    by_ordinal = {value.layer_ordinal: value for value in layer_specs}
    for mode in values:
        try:
            spec = by_ordinal[mode.layer_ordinal]
        except KeyError as error:
            raise ValueError(
                "mode catalog entry is outside the layer specs"
            ) from error
        if (
            mode.layer_id != spec.layer_id
            or mode.activation_site != spec.activation_site
            or mode.mode_index >= spec.width
        ):
            raise ValueError(
                "mode catalog entry does not match its layer spec"
            )
    return values


@dataclass(frozen=True, slots=True)
class FisherPromptClusterConfig:
    """Authenticated provenance and policy for prompt-Fisher clustering."""

    model_fingerprint: str
    calibration_split_sha256: str
    objective_sha256: str
    source_fisher_coupling_sha256: str
    layer_specs: tuple[CrossBlockLayerSpec, ...]
    mode_catalog: tuple[ModeKey, ...]
    cluster_count: int
    max_iterations: int = 100
    tolerance: float = 1e-10
    mode_chunk_size: int = 4096
    score_definition: str = _SCORE_DEFINITION
    artifact_sha256: str = ""
    artifact_kind: str = _CONFIG_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.model_fingerprint,
            label="model_fingerprint",
        )
        _require_sha256(
            self.calibration_split_sha256,
            label="calibration_split_sha256",
        )
        _require_sha256(
            self.objective_sha256,
            label="objective_sha256",
        )
        _require_sha256(
            self.source_fisher_coupling_sha256,
            label="source_fisher_coupling_sha256",
        )
        _validate_layer_specs(self.layer_specs)
        _validate_mode_catalog(
            self.mode_catalog,
            layer_specs=self.layer_specs,
        )
        _require_positive_int(self.cluster_count, label="cluster_count")
        _require_positive_int(
            self.max_iterations,
            label="max_iterations",
        )
        _require_positive_int(
            self.mode_chunk_size,
            label="mode_chunk_size",
        )
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not math.isfinite(float(self.tolerance))
            or float(self.tolerance) < 0.0
        ):
            raise ValueError("tolerance must be finite and nonnegative")
        object.__setattr__(self, "tolerance", float(self.tolerance))
        if (
            self.score_definition != _SCORE_DEFINITION
            or self.artifact_kind != _CONFIG_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("prompt-cluster config semantics are invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        else:
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            if self.artifact_sha256 != computed:
                raise ValueError("prompt-cluster config hash mismatch")

    @property
    def mode_count(self) -> int:
        return len(self.mode_catalog)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "model_fingerprint": self.model_fingerprint,
            "calibration_split_sha256": self.calibration_split_sha256,
            "objective_sha256": self.objective_sha256,
            "source_fisher_coupling_sha256": (
                self.source_fisher_coupling_sha256
            ),
            "layer_specs": tuple(
                value.metadata() for value in self.layer_specs
            ),
            "mode_catalog": tuple(
                value.metadata() for value in self.mode_catalog
            ),
            "cluster_count": self.cluster_count,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "mode_chunk_size": self.mode_chunk_size,
            "score_definition": self.score_definition,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._payload(),
            domain=_CONFIG_HASH_DOMAIN,
        )

    def validate_integrity(self) -> None:
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("prompt-cluster config hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "mode_count": self.mode_count,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FisherPromptClusterConfig:
        expected = {
            "artifact_kind",
            "format_version",
            "model_fingerprint",
            "calibration_split_sha256",
            "objective_sha256",
            "source_fisher_coupling_sha256",
            "layer_specs",
            "mode_catalog",
            "cluster_count",
            "max_iterations",
            "tolerance",
            "mode_chunk_size",
            "score_definition",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("prompt-cluster config state fields are invalid")
        if type(state["layer_specs"]) is not tuple or type(
            state["mode_catalog"]
        ) is not tuple:
            raise TypeError("prompt-cluster config catalogs must be tuples")
        return cls(
            model_fingerprint=str(state["model_fingerprint"]),
            calibration_split_sha256=str(
                state["calibration_split_sha256"]
            ),
            objective_sha256=str(state["objective_sha256"]),
            source_fisher_coupling_sha256=str(
                state["source_fisher_coupling_sha256"]
            ),
            layer_specs=tuple(
                CrossBlockLayerSpec.from_state_dict(value)
                for value in state["layer_specs"]
            ),
            mode_catalog=tuple(
                ModeKey.from_state_dict(value)
                for value in state["mode_catalog"]
            ),
            cluster_count=int(state["cluster_count"]),
            max_iterations=int(state["max_iterations"]),
            tolerance=float(state["tolerance"]),
            mode_chunk_size=int(state["mode_chunk_size"]),
            score_definition=str(state["score_definition"]),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
        )


@dataclass(frozen=True, slots=True)
class FisherPromptClusterPlan:
    """Strict-loadable prompt-conditioned Fisher mode partition."""

    config: FisherPromptClusterConfig
    prompt_count: int
    source_effects_sha256: str
    assignments: Tensor
    orientations: Tensor
    similarities: Tensor
    centroids: Tensor
    cluster_counts: Tensor
    cluster_mass: Tensor
    iterations: int
    converged: bool
    assignments_sha256: str
    orientations_sha256: str
    similarities_sha256: str
    centroids_sha256: str
    cluster_counts_sha256: str
    cluster_mass_sha256: str
    artifact_sha256: str
    artifact_kind: str = _PLAN_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.config, FisherPromptClusterConfig):
            raise TypeError("config must be FisherPromptClusterConfig")
        self.config.validate_integrity()
        _require_positive_int(self.prompt_count, label="prompt_count")
        _require_sha256(
            self.source_effects_sha256,
            label="source_effects_sha256",
        )
        _require_positive_int(self.iterations, label="iterations")
        if self.iterations > self.config.max_iterations:
            raise ValueError("iterations exceeds configured maximum")
        if type(self.converged) is not bool:
            raise TypeError("converged must be boolean")

        tensor_specs = (
            ("assignments", torch.int64, (self.mode_count,)),
            ("orientations", torch.int8, (self.mode_count,)),
            ("similarities", torch.float64, (self.mode_count,)),
            (
                "centroids",
                torch.float64,
                (self.cluster_count, self.prompt_count),
            ),
            ("cluster_counts", torch.int64, (self.cluster_count,)),
            ("cluster_mass", torch.float64, (self.cluster_count,)),
        )
        for name, dtype, shape in tensor_specs:
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != dtype
                or tuple(value.shape) != shape
                or (
                    value.is_floating_point()
                    and not bool(torch.isfinite(value).all())
                )
            ):
                raise ValueError(
                    f"{name} must be a finite CPU {dtype} Tensor "
                    f"with shape {shape}"
                )
            object.__setattr__(
                self,
                name,
                value.detach().contiguous().clone(),
            )

        if bool(
            ((self.assignments < -1) | (
                self.assignments >= self.cluster_count
            )).any()
        ):
            raise ValueError("assignments contain an invalid cluster index")
        if bool(
            ~(
                (self.orientations == -1)
                | (self.orientations == 0)
                | (self.orientations == 1)
            ).all()
        ):
            raise ValueError("orientations must contain only -1, 0, or 1")
        unassigned = self.assignments == -1
        if (
            not torch.equal(self.orientations == 0, unassigned)
            or bool((self.similarities < 0).any())
            or bool((self.similarities > 1.0 + 1e-12).any())
            or bool((self.similarities[unassigned] != 0).any())
        ):
            raise ValueError(
                "assignment orientation/similarity semantics are invalid"
            )
        expected_counts = torch.bincount(
            self.assignments[~unassigned],
            minlength=self.cluster_count,
        )
        if (
            not torch.equal(self.cluster_counts, expected_counts)
            or bool((self.cluster_mass < 0).any())
            or bool(
                (self.cluster_mass[self.cluster_counts == 0] != 0).any()
            )
        ):
            raise ValueError("cluster count/mass summaries are invalid")
        norms = torch.linalg.vector_norm(self.centroids, dim=1)
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("every cluster centroid must have unit norm")
        if (
            self.artifact_kind != _PLAN_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("prompt-cluster plan header is invalid")
        for name in (
            "assignments_sha256",
            "orientations_sha256",
            "similarities_sha256",
            "centroids_sha256",
            "cluster_counts_sha256",
            "cluster_mass_sha256",
            "artifact_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        self.validate_integrity()

    @property
    def cluster_count(self) -> int:
        return self.config.cluster_count

    @property
    def mode_count(self) -> int:
        return self.config.mode_count

    @property
    def mode_catalog(self) -> tuple[ModeKey, ...]:
        return self.config.mode_catalog

    @property
    def assigned_mode_count(self) -> int:
        return int((self.assignments >= 0).sum().item())

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "config_artifact_sha256": self.config.artifact_sha256,
            "prompt_count": self.prompt_count,
            "source_effects_sha256": self.source_effects_sha256,
            "iterations": self.iterations,
            "converged": self.converged,
            "assignments_sha256": self.assignments_sha256,
            "orientations_sha256": self.orientations_sha256,
            "similarities_sha256": self.similarities_sha256,
            "centroids_sha256": self.centroids_sha256,
            "cluster_counts_sha256": self.cluster_counts_sha256,
            "cluster_mass_sha256": self.cluster_mass_sha256,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_PLAN_HASH_DOMAIN)

    def validate_integrity(self) -> None:
        self.config.validate_integrity()
        for name, dtype in (
            ("assignments", torch.int64),
            ("orientations", torch.int8),
            ("similarities", torch.float64),
            ("centroids", torch.float64),
            ("cluster_counts", torch.int64),
            ("cluster_mass", torch.float64),
        ):
            expected = getattr(self, f"{name}_sha256")
            actual = _tensor_sha256(
                getattr(self, name),
                label=name,
                dtype=dtype,
            )
            if actual != expected:
                raise ValueError(f"prompt-cluster {name} hash mismatch")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("prompt-cluster plan hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "config": self.config.metadata(),
            "cluster_counts": tuple(
                int(value) for value in self.cluster_counts.tolist()
            ),
            "cluster_mass": tuple(
                float(value) for value in self.cluster_mass.tolist()
            ),
            "assigned_mode_count": self.assigned_mode_count,
            "mode_count": self.mode_count,
            "cluster_count": self.cluster_count,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "config": self.config.state_dict(),
            "prompt_count": self.prompt_count,
            "source_effects_sha256": self.source_effects_sha256,
            "assignments": self.assignments.detach().clone(),
            "orientations": self.orientations.detach().clone(),
            "similarities": self.similarities.detach().clone(),
            "centroids": self.centroids.detach().clone(),
            "cluster_counts": self.cluster_counts.detach().clone(),
            "cluster_mass": self.cluster_mass.detach().clone(),
            "iterations": self.iterations,
            "converged": self.converged,
            "assignments_sha256": self.assignments_sha256,
            "orientations_sha256": self.orientations_sha256,
            "similarities_sha256": self.similarities_sha256,
            "centroids_sha256": self.centroids_sha256,
            "cluster_counts_sha256": self.cluster_counts_sha256,
            "cluster_mass_sha256": self.cluster_mass_sha256,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FisherPromptClusterPlan:
        expected = {
            "artifact_kind",
            "format_version",
            "config",
            "prompt_count",
            "source_effects_sha256",
            "assignments",
            "orientations",
            "similarities",
            "centroids",
            "cluster_counts",
            "cluster_mass",
            "iterations",
            "converged",
            "assignments_sha256",
            "orientations_sha256",
            "similarities_sha256",
            "centroids_sha256",
            "cluster_counts_sha256",
            "cluster_mass_sha256",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("prompt-cluster plan state fields are invalid")
        if not isinstance(state["config"], Mapping):
            raise TypeError("prompt-cluster plan config must be a mapping")
        for name in (
            "assignments",
            "orientations",
            "similarities",
            "centroids",
            "cluster_counts",
            "cluster_mass",
        ):
            if not isinstance(state[name], Tensor):
                raise TypeError(f"prompt-cluster {name} must be a Tensor")
        return cls(
            config=FisherPromptClusterConfig.from_state_dict(
                state["config"]
            ),
            prompt_count=int(state["prompt_count"]),
            source_effects_sha256=str(state["source_effects_sha256"]),
            assignments=state["assignments"],
            orientations=state["orientations"],
            similarities=state["similarities"],
            centroids=state["centroids"],
            cluster_counts=state["cluster_counts"],
            cluster_mass=state["cluster_mass"],
            iterations=int(state["iterations"]),
            converged=state["converged"],
            assignments_sha256=str(state["assignments_sha256"]),
            orientations_sha256=str(state["orientations_sha256"]),
            similarities_sha256=str(state["similarities_sha256"]),
            centroids_sha256=str(state["centroids_sha256"]),
            cluster_counts_sha256=str(
                state["cluster_counts_sha256"]
            ),
            cluster_mass_sha256=str(state["cluster_mass_sha256"]),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
        )


def prompt_mode_gate_effects(
    activations: Tensor,
    nll_gradients: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Return ``sum_t z_tj * dNLL/dz_tj`` for each prompt and mode.

    Both inputs have shape ``[prompts, tokens, modes]``.  A boolean
    ``valid_mask`` with shape ``[prompts, tokens]`` can exclude padding or
    unsupervised token positions.  The result is a CPU float64 matrix.
    """

    if (
        not isinstance(activations, Tensor)
        or not isinstance(nll_gradients, Tensor)
        or activations.shape != nll_gradients.shape
        or activations.ndim != 3
        or min(activations.shape) <= 0
        or not activations.is_floating_point()
        or not nll_gradients.is_floating_point()
    ):
        raise ValueError(
            "activations and nll_gradients must share nonempty floating "
            "shape [prompts, tokens, modes]"
        )
    values = activations.detach().to(device="cpu", dtype=torch.float64)
    gradients = nll_gradients.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    if (
        not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(gradients).all())
    ):
        raise ValueError("activations and nll_gradients must be finite")
    products = values * gradients
    if valid_mask is not None:
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.dtype != torch.bool
            or tuple(valid_mask.shape) != tuple(activations.shape[:2])
        ):
            raise ValueError(
                "valid_mask must be boolean with shape [prompts, tokens]"
            )
        products = products * valid_mask.detach().to(
            device="cpu",
        ).unsqueeze(-1)
    return products.sum(dim=1).contiguous()


def _catalog_order(config: FisherPromptClusterConfig) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(config.mode_count),
            key=lambda index: config.mode_catalog[index],
        )
    )


def _initialize_centroids(
    unit_columns: Tensor,
    energies: Tensor,
    *,
    config: FisherPromptClusterConfig,
) -> Tensor:
    eligible = energies > 0
    eligible_count = int(eligible.sum().item())
    if eligible_count < config.cluster_count:
        raise ValueError(
            "cluster_count cannot exceed the number of nonzero-energy modes"
        )
    catalog_order = _catalog_order(config)
    eligible_indices = [index for index in catalog_order if eligible[index]]
    first = min(
        eligible_indices,
        key=lambda index: (
            -float(energies[index].item()),
            config.mode_catalog[index],
        ),
    )
    selected = [first]
    selected_mask = torch.zeros(
        config.mode_count,
        dtype=torch.bool,
    )
    selected_mask[first] = True
    maximum_similarities = (
        unit_columns.T @ unit_columns[:, first]
    ).abs()
    catalog_ranks = torch.empty(
        config.mode_count,
        dtype=torch.int64,
    )
    for rank, index in enumerate(catalog_order):
        catalog_ranks[index] = rank
    while len(selected) < config.cluster_count:
        candidates = eligible & ~selected_mask
        candidate_indices = candidates.nonzero(as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            raise RuntimeError("deterministic centroid initialization failed")
        candidate_similarities = maximum_similarities.index_select(
            0,
            candidate_indices,
        )
        minimum_similarity = candidate_similarities.min()
        candidate_indices = candidate_indices[
            candidate_similarities == minimum_similarity
        ]
        candidate_energies = energies.index_select(0, candidate_indices)
        maximum_energy = candidate_energies.max()
        candidate_indices = candidate_indices[
            candidate_energies == maximum_energy
        ]
        candidate_catalog_ranks = catalog_ranks.index_select(
            0,
            candidate_indices,
        )
        best_index = int(
            candidate_indices[
                candidate_catalog_ranks.argmin()
            ].item()
        )
        selected.append(best_index)
        selected_mask[best_index] = True
        similarities = (
            unit_columns.T @ unit_columns[:, best_index]
        ).abs()
        maximum_similarities = torch.maximum(
            maximum_similarities,
            similarities,
        )
    return torch.stack(
        [unit_columns[:, index] for index in selected],
        dim=0,
    ).contiguous()


def _assign_axial(
    unit_columns: Tensor,
    energies: Tensor,
    centroids: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    mode_count = unit_columns.shape[1]
    assignments = torch.full((mode_count,), -1, dtype=torch.int64)
    orientations = torch.zeros((mode_count,), dtype=torch.int8)
    similarities = torch.zeros((mode_count,), dtype=torch.float64)
    for start in range(0, mode_count, chunk_size):
        stop = min(start + chunk_size, mode_count)
        eligible = energies[start:stop] > 0
        if not bool(eligible.any()):
            continue
        signed = centroids @ unit_columns[:, start:stop]
        absolute = signed.abs()
        maximum, clusters = absolute.max(dim=0)
        selected_signed = signed.gather(
            0,
            clusters.unsqueeze(0),
        ).squeeze(0)
        chunk_assignments = torch.where(
            eligible,
            clusters,
            torch.full_like(clusters, -1),
        )
        chunk_orientations = torch.where(
            eligible,
            torch.where(
                selected_signed >= 0,
                torch.ones_like(clusters, dtype=torch.int8),
                -torch.ones_like(clusters, dtype=torch.int8),
            ),
            torch.zeros_like(clusters, dtype=torch.int8),
        )
        assignments[start:stop] = chunk_assignments
        orientations[start:stop] = chunk_orientations
        similarities[start:stop] = torch.where(
            eligible,
            maximum.clamp(min=0.0, max=1.0).round(decimals=14),
            torch.zeros_like(maximum),
        )
    return assignments, orientations, similarities


def _update_centroids(
    unit_columns: Tensor,
    energies: Tensor,
    assignments: Tensor,
    orientations: Tensor,
    previous: Tensor,
    *,
    catalog_order: tuple[int, ...],
) -> Tensor:
    updated = torch.zeros_like(previous)
    for index in catalog_order:
        cluster = int(assignments[index].item())
        if cluster < 0:
            continue
        weight = float(energies[index].item()) * int(
            orientations[index].item()
        )
        updated[cluster].add_(unit_columns[:, index], alpha=weight)
    for cluster in range(updated.shape[0]):
        norm = float(torch.linalg.vector_norm(updated[cluster]).item())
        if norm == 0.0:
            updated[cluster].copy_(previous[cluster])
            continue
        updated[cluster].div_(norm)
        if float(torch.dot(updated[cluster], previous[cluster]).item()) < 0.0:
            updated[cluster].neg_()
    if not bool(torch.isfinite(updated).all()):
        raise ValueError("Fisher-weighted centroid update became nonfinite")
    return updated.contiguous()


def _plan_payload(
    *,
    config: FisherPromptClusterConfig,
    prompt_count: int,
    source_effects_sha256: str,
    iterations: int,
    converged: bool,
    tensor_hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "artifact_kind": _PLAN_KIND,
        "format_version": _FORMAT_VERSION,
        "config_artifact_sha256": config.artifact_sha256,
        "prompt_count": prompt_count,
        "source_effects_sha256": source_effects_sha256,
        "iterations": iterations,
        "converged": converged,
        **{
            f"{name}_sha256": tensor_hashes[name]
            for name in (
                "assignments",
                "orientations",
                "similarities",
                "centroids",
                "cluster_counts",
                "cluster_mass",
            )
        },
    }


def build_fisher_prompt_clusters(
    effects: Tensor,
    config: FisherPromptClusterConfig,
) -> FisherPromptClusterPlan:
    """Cluster prompt-conditioned signed mode effects into Fisher axes."""

    if not isinstance(config, FisherPromptClusterConfig):
        raise TypeError("config must be FisherPromptClusterConfig")
    config.validate_integrity()
    matrix = _as_effect_matrix(effects)
    if matrix.shape[1] != config.mode_count:
        raise ValueError("effects mode width does not match the mode catalog")

    energies = matrix.square().sum(dim=0)
    if not bool(torch.isfinite(energies).all()):
        raise ValueError("prompt Fisher energies became nonfinite")
    unit_columns = torch.zeros_like(matrix)
    positive = energies > 0
    unit_columns[:, positive] = (
        matrix[:, positive] / energies[positive].sqrt().unsqueeze(0)
    )
    centroids = _initialize_centroids(
        unit_columns,
        energies,
        config=config,
    )
    catalog_order = _catalog_order(config)
    converged = False
    iterations = 0
    for iterations in range(1, config.max_iterations + 1):
        assignments, orientations, _ = _assign_axial(
            unit_columns,
            energies,
            centroids,
            chunk_size=config.mode_chunk_size,
        )
        updated = _update_centroids(
            unit_columns,
            energies,
            assignments,
            orientations,
            centroids,
            catalog_order=catalog_order,
        )
        axial_changes = 1.0 - (
            updated * centroids
        ).sum(dim=1).abs().clamp(max=1.0)
        centroids = updated
        if float(axial_changes.max().item()) <= config.tolerance:
            converged = True
            break

    assignments, orientations, similarities = _assign_axial(
        unit_columns,
        energies,
        centroids,
        chunk_size=config.mode_chunk_size,
    )
    assigned = assignments >= 0
    cluster_counts = torch.bincount(
        assignments[assigned],
        minlength=config.cluster_count,
    ).to(dtype=torch.int64)
    cluster_mass = torch.zeros(
        config.cluster_count,
        dtype=torch.float64,
    )
    for index in catalog_order:
        cluster = int(assignments[index].item())
        if cluster >= 0:
            cluster_mass[cluster] += energies[index]

    assignments = assignments.contiguous()
    orientations = orientations.contiguous()
    similarities = similarities.contiguous()
    centroids = centroids.contiguous()
    cluster_counts = cluster_counts.contiguous()
    cluster_mass = cluster_mass.contiguous()
    tensors = {
        "assignments": assignments,
        "orientations": orientations,
        "similarities": similarities,
        "centroids": centroids,
        "cluster_counts": cluster_counts,
        "cluster_mass": cluster_mass,
    }
    dtypes = {
        "assignments": torch.int64,
        "orientations": torch.int8,
        "similarities": torch.float64,
        "centroids": torch.float64,
        "cluster_counts": torch.int64,
        "cluster_mass": torch.float64,
    }
    tensor_hashes = {
        name: _tensor_sha256(
            value,
            label=name,
            dtype=dtypes[name],
        )
        for name, value in tensors.items()
    }
    source_effects_sha256 = fisher_prompt_effects_sha256(matrix)
    payload = _plan_payload(
        config=config,
        prompt_count=int(matrix.shape[0]),
        source_effects_sha256=source_effects_sha256,
        iterations=iterations,
        converged=converged,
        tensor_hashes=tensor_hashes,
    )
    return FisherPromptClusterPlan(
        config=config,
        prompt_count=int(matrix.shape[0]),
        source_effects_sha256=source_effects_sha256,
        assignments=assignments,
        orientations=orientations,
        similarities=similarities,
        centroids=centroids,
        cluster_counts=cluster_counts,
        cluster_mass=cluster_mass,
        iterations=iterations,
        converged=converged,
        assignments_sha256=tensor_hashes["assignments"],
        orientations_sha256=tensor_hashes["orientations"],
        similarities_sha256=tensor_hashes["similarities"],
        centroids_sha256=tensor_hashes["centroids"],
        cluster_counts_sha256=tensor_hashes["cluster_counts"],
        cluster_mass_sha256=tensor_hashes["cluster_mass"],
        artifact_sha256=_json_sha256(
            payload,
            domain=_PLAN_HASH_DOMAIN,
        ),
    )


def prompt_cluster_fisher_signatures(
    effects: Tensor,
    plan: FisherPromptClusterPlan,
) -> Tensor:
    """Project prompt empirical Fisher contributions onto mode clusters.

    ``effects`` may contain new prompts, but its mode columns must use the
    authenticated catalog order in ``plan``.  The returned CPU float64 matrix
    has shape ``[prompts, clusters]``.
    """

    if not isinstance(plan, FisherPromptClusterPlan):
        raise TypeError("plan must be FisherPromptClusterPlan")
    plan.validate_integrity()
    matrix = _as_effect_matrix(effects)
    if matrix.shape[1] != plan.mode_count:
        raise ValueError("effects mode width does not match the cluster plan")
    signatures = torch.zeros(
        (matrix.shape[0], plan.cluster_count),
        dtype=torch.float64,
    )
    squared = matrix.square()
    for cluster in range(plan.cluster_count):
        selected = plan.assignments == cluster
        if bool(selected.any()):
            signatures[:, cluster] = squared[:, selected].sum(dim=1)
    return signatures.contiguous()


def _integer_labels(
    values: Tensor | Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    if isinstance(values, Tensor):
        if (
            values.device.type != "cpu"
            or values.dtype != torch.int64
            or values.ndim != 1
        ):
            raise ValueError(f"{label} must be a CPU int64 vector")
        return tuple(int(value) for value in values.tolist())
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be an integer label sequence")
    if any(type(value) is not int for value in values):
        raise TypeError(f"{label} must contain only integers")
    return tuple(values)


def adjusted_rand_index(
    first: Tensor | Sequence[int],
    second: Tensor | Sequence[int],
    *,
    ignore_unassigned: bool = True,
) -> float:
    """Compute adjusted Rand index without scipy or sklearn.

    Negative labels are treated as unassigned and omitted when
    ``ignore_unassigned`` is true.
    """

    first_labels = _integer_labels(first, label="first")
    second_labels = _integer_labels(second, label="second")
    if len(first_labels) != len(second_labels):
        raise ValueError("assignment vectors must have equal length")
    pairs = [
        (left, right)
        for left, right in zip(first_labels, second_labels, strict=True)
        if not ignore_unassigned or (left >= 0 and right >= 0)
    ]
    if len(pairs) < 2:
        raise ValueError(
            "adjusted Rand index requires at least two jointly assigned items"
        )
    if not ignore_unassigned and any(
        left < 0 or right < 0 for left, right in pairs
    ):
        raise ValueError("negative assignment labels require ignoring")
    contingency: dict[tuple[int, int], int] = {}
    first_counts: dict[int, int] = {}
    second_counts: dict[int, int] = {}
    for left, right in pairs:
        contingency[(left, right)] = (
            contingency.get((left, right), 0) + 1
        )
        first_counts[left] = first_counts.get(left, 0) + 1
        second_counts[right] = second_counts.get(right, 0) + 1

    def choose_two(value: int) -> int:
        return value * (value - 1) // 2

    sample_pairs = choose_two(len(pairs))
    index = sum(choose_two(value) for value in contingency.values())
    first_index = sum(choose_two(value) for value in first_counts.values())
    second_index = sum(choose_two(value) for value in second_counts.values())
    expected = first_index * second_index / sample_pairs
    maximum = 0.5 * (first_index + second_index)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0
    return float((index - expected) / denominator)


def prompt_cluster_adjusted_rand_index(
    first: Tensor | Sequence[int],
    second: Tensor | Sequence[int],
    *,
    ignore_unassigned: bool = True,
) -> float:
    """Named wrapper for prompt-Fisher assignment stability."""

    return adjusted_rand_index(
        first,
        second,
        ignore_unassigned=ignore_unassigned,
    )


def fisher_prompt_cluster_stability(
    first: FisherPromptClusterPlan,
    second: FisherPromptClusterPlan,
) -> float:
    """Return ARI after aligning two plans by authenticated mode identity."""

    if not isinstance(first, FisherPromptClusterPlan) or not isinstance(
        second,
        FisherPromptClusterPlan,
    ):
        raise TypeError("both values must be FisherPromptClusterPlan")
    first.validate_integrity()
    second.validate_integrity()
    if (
        first.config.model_fingerprint
        != second.config.model_fingerprint
        or first.config.objective_sha256
        != second.config.objective_sha256
    ):
        raise ValueError(
            "stability plans must bind the same model and objective"
        )
    second_index = {
        mode: index for index, mode in enumerate(second.mode_catalog)
    }
    if set(first.mode_catalog) != set(second.mode_catalog):
        raise ValueError("stability plans must contain the same mode catalog")
    aligned_second = torch.tensor(
        [
            int(second.assignments[second_index[mode]].item())
            for mode in first.mode_catalog
        ],
        dtype=torch.int64,
    )
    return adjusted_rand_index(first.assignments, aligned_second)


__all__ = [
    "FisherPromptClusterConfig",
    "FisherPromptClusterPlan",
    "adjusted_rand_index",
    "build_fisher_prompt_clusters",
    "fisher_prompt_cluster_stability",
    "fisher_prompt_effects_sha256",
    "prompt_cluster_adjusted_rand_index",
    "prompt_cluster_fisher_signatures",
    "prompt_mode_gate_effects",
]
