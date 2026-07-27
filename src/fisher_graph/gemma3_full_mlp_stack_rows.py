"""Memory-bounded native rows for one exhaustive Gemma MLP layer.

The whole-stack compiler has 64 Fisher fragments at each Gemma MLP.  Retaining
one ``LayerFragmentRows`` object per fragment duplicates the normalized layer
input and materializes many residual targets.  This module instead reduces one
layer's complete fragment partition into a single ephemeral row table:

* ``inputs`` is the shared native normalized MLP input;
* ``contributions`` is the exact complete native MLP residual contribution;
* ``fisher_weights`` sums the virtual-gate Fisher weight over every native
  channel; and
* ``row_keys`` binds every row to an example and logical token position.

Memory therefore scales with one layer's ``X`` and ``Y`` tables, not with the
number of Fisher fragments in that layer.  Nothing in this module is a saved
artifact and no source weights are retained by the returned record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json

import torch
from torch import Tensor

from .gemma3_modal_generator_dev_experiment import LayerFragmentRows
from .parameter_cluster_fragments import ParameterClusterLayerFragment
from .streaming_analysis import ActivationScoreGradientRows


__all__ = [
    "FullMLPStackLayerRows",
    "collect_full_mlp_stack_layer_rows",
    "collect_full_mlp_stack_rows",
]


_ROW_KEY_DOMAIN = b"fisher_graph.gemma3_full_mlp_stack.layer_rows.v1\0"


def _row_key_sha256(row_keys: tuple[tuple[str, int], ...]) -> str:
    encoded = json.dumps(
        row_keys,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_ROW_KEY_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()


def _canonical_layer_fragments(
    fragments: Sequence[ParameterClusterLayerFragment],
    *,
    intermediate_width: int,
) -> tuple[ParameterClusterLayerFragment, ...]:
    if isinstance(fragments, (str, bytes)) or not isinstance(
        fragments,
        Sequence,
    ):
        raise TypeError("fragments must be a sequence")
    values = tuple(fragments)
    if not values or any(
        not isinstance(value, ParameterClusterLayerFragment)
        for value in values
    ):
        raise ValueError("layer fragments must be nonempty fragment values")
    if type(intermediate_width) is not int or intermediate_width <= 0:
        raise ValueError("intermediate_width must be positive")
    for value in values:
        value.validate_integrity()

    canonical = tuple(
        sorted(
            values,
            key=lambda value: (
                value.cluster_id,
                value.fragment_id,
                value.artifact_sha256,
            ),
        )
    )
    first = canonical[0]
    shared_fields = (
        "layer_ordinal",
        "layer_id",
        "activation_site",
        "input_site",
        "output_site",
        "input_catalog_sha256",
        "input_width",
        "output_width",
        "source_cluster_plan_sha256",
        "source_fisher_coupling_sha256",
        "parameter_catalog_sha256",
        "source_model_sha256",
    )
    if any(
        any(
            getattr(fragment, field) != getattr(first, field)
            for field in shared_fields
        )
        for fragment in canonical[1:]
    ):
        raise ValueError("fragments do not describe one common MLP layer")
    if (
        len({value.fragment_id for value in canonical}) != len(canonical)
        or len({value.artifact_sha256 for value in canonical}) != len(canonical)
    ):
        raise ValueError("layer fragments must be unique")

    channels = tuple(
        channel
        for fragment in canonical
        for channel in fragment.channel_indices
    )
    if len(channels) != len(set(channels)):
        raise ValueError("layer fragment channels overlap")
    if tuple(sorted(channels)) != tuple(range(intermediate_width)):
        raise ValueError(
            "layer fragment channels do not exhaust the native MLP width"
        )
    groups = tuple(
        group
        for fragment in canonical
        for group in fragment.group_indices
    )
    if len(groups) != len(set(groups)):
        raise ValueError("layer fragment parameter groups overlap")
    expected_parameters = intermediate_width * (
        2 * first.input_width + first.output_width
    )
    if sum(
        fragment.native_parameter_count for fragment in canonical
    ) != expected_parameters:
        raise ValueError("exhaustive layer native parameter count drifted")
    return canonical


@dataclass(frozen=True, slots=True)
class FullMLPStackLayerRows:
    """One immutable, ephemeral row table for a complete native MLP layer."""

    layer_ordinal: int
    layer_id: str
    input_site: str
    activation_site: str
    output_site: str
    intermediate_width: int
    fragment_ids: tuple[str, ...]
    fragment_sha256s: tuple[str, ...]
    inputs: Tensor
    contributions: Tensor
    fisher_weights: Tensor
    row_keys: tuple[tuple[str, int], ...]
    sequences: int
    row_key_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be nonnegative")
        for name in ("layer_id", "input_site", "activation_site", "output_site"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")
        if self.input_site == self.activation_site:
            raise ValueError("input and activation sites must differ")
        if type(self.intermediate_width) is not int or (
            self.intermediate_width <= 0
        ):
            raise ValueError("intermediate_width must be positive")
        if (
            type(self.fragment_ids) is not tuple
            or not self.fragment_ids
            or len(self.fragment_ids) != len(set(self.fragment_ids))
            or any(
                not isinstance(value, str) or not value
                for value in self.fragment_ids
            )
            or type(self.fragment_sha256s) is not tuple
            or len(self.fragment_sha256s) != len(self.fragment_ids)
            or len(self.fragment_sha256s) != len(set(self.fragment_sha256s))
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in self.fragment_sha256s
            )
        ):
            raise ValueError("fragment identity catalog is invalid")

        normalized = LayerFragmentRows(
            inputs=self.inputs,
            contributions=self.contributions,
            fisher_weights=self.fisher_weights,
            sequences=self.sequences,
        )
        object.__setattr__(self, "inputs", normalized.inputs)
        object.__setattr__(self, "contributions", normalized.contributions)
        object.__setattr__(self, "fisher_weights", normalized.fisher_weights)
        object.__setattr__(self, "sequences", normalized.sequences)
        if (
            self.inputs.shape[0] != len(self.row_keys)
            or not self.row_keys
            or len(self.row_keys) != len(set(self.row_keys))
            or any(
                type(key) is not tuple
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or type(key[1]) is not int
                or key[1] < 0
                for key in self.row_keys
            )
        ):
            raise ValueError("row_keys must uniquely cover the row axis")
        computed = _row_key_sha256(self.row_keys)
        if self.row_key_sha256 == "":
            object.__setattr__(self, "row_key_sha256", computed)
        elif self.row_key_sha256 != computed:
            raise ValueError("row-key hash mismatch")

    @property
    def observations(self) -> int:
        return self.inputs.shape[0]


def _canonical_down_projection(
    down_projection_weight: Tensor,
) -> Tensor:
    if (
        not isinstance(down_projection_weight, Tensor)
        or down_projection_weight.ndim != 2
        or not down_projection_weight.is_floating_point()
        or down_projection_weight.shape[0] <= 0
        or down_projection_weight.shape[1] <= 0
    ):
        raise ValueError("down_projection_weight must be a floating matrix")
    down = down_projection_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(down).all()):
        raise ValueError("down_projection_weight must be finite")
    return down


def collect_full_mlp_stack_rows(
    rows: Iterable[ActivationScoreGradientRows],
    *,
    fragments_by_layer: Mapping[
        int,
        Sequence[ParameterClusterLayerFragment],
    ],
    down_projection_weights: Mapping[int, Tensor],
) -> tuple[FullMLPStackLayerRows, ...]:
    """Collect every declared complete MLP layer in one shared row replay.

    The input stream must expose exactly the union of the normalized-input and
    down-input sites for all declared layers.  Results are ordered by layer
    ordinal and reuse one exact row-key tuple.  Only one ``X``, ``Y``, and
    Fisher-weight table is retained per layer; no per-fragment row table is
    created.
    """

    if not isinstance(fragments_by_layer, Mapping) or not fragments_by_layer:
        raise ValueError("fragments_by_layer must be a nonempty mapping")
    if not isinstance(down_projection_weights, Mapping) or set(
        down_projection_weights
    ) != set(fragments_by_layer):
        raise ValueError(
            "down_projection_weights must exactly cover declared layers"
        )
    declared_ordinals = tuple(fragments_by_layer)
    if any(
        type(ordinal) is not int or ordinal < 0
        for ordinal in declared_ordinals
    ):
        raise ValueError("layer ordinals must be nonnegative integers")
    ordinals = tuple(sorted(declared_ordinals))

    prepared: dict[
        int,
        tuple[
            tuple[ParameterClusterLayerFragment, ...],
            Tensor,
        ],
    ] = {}
    all_fragment_ids: list[str] = []
    all_fragment_sha256s: list[str] = []
    all_group_indices: list[int] = []
    expected_sites: set[str] = set()
    common_provenance: tuple[str, str, str, str] | None = None
    for ordinal in ordinals:
        down = _canonical_down_projection(down_projection_weights[ordinal])
        selected = _canonical_layer_fragments(
            fragments_by_layer[ordinal],
            intermediate_width=down.shape[1],
        )
        first = selected[0]
        if first.layer_ordinal != ordinal:
            raise ValueError(
                "fragments_by_layer key differs from fragment layer ordinal"
            )
        if down.shape[0] != first.output_width:
            raise ValueError(
                "down projection output width differs from fragments"
            )
        provenance = (
            first.source_cluster_plan_sha256,
            first.source_fisher_coupling_sha256,
            first.parameter_catalog_sha256,
            first.source_model_sha256,
        )
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise ValueError("full-stack layer provenance differs")
        layer_sites = {first.input_site, first.activation_site}
        if expected_sites & layer_sites:
            raise ValueError("full-stack layers reuse an activation site")
        expected_sites.update(layer_sites)
        all_fragment_ids.extend(
            fragment.fragment_id for fragment in selected
        )
        all_fragment_sha256s.extend(
            fragment.artifact_sha256 for fragment in selected
        )
        all_group_indices.extend(
            group
            for fragment in selected
            for group in fragment.group_indices
        )
        prepared[ordinal] = (selected, down)
    if (
        len(all_fragment_ids) != len(set(all_fragment_ids))
        or len(all_fragment_sha256s) != len(set(all_fragment_sha256s))
    ):
        raise ValueError("full-stack fragments are not globally unique")
    if len(all_group_indices) != len(set(all_group_indices)):
        raise ValueError("full-stack parameter groups overlap across layers")

    inputs: dict[int, list[Tensor]] = {
        ordinal: [] for ordinal in ordinals
    }
    contributions: dict[int, list[Tensor]] = {
        ordinal: [] for ordinal in ordinals
    }
    fisher_weights: dict[int, list[Tensor]] = {
        ordinal: [] for ordinal in ordinals
    }
    row_keys: list[tuple[str, int]] = []
    seen_row_keys: set[tuple[str, int]] = set()
    sequences = 0
    iterator = iter(rows)
    try:
        for row in iterator:
            if set(row.activations) != expected_sites:
                raise ValueError(
                    "full-stack rows must expose exactly the input and "
                    "down-input site union"
                )
            if row.example_id is None:
                raise ValueError("full-stack rows require stable example ids")
            keys = tuple(
                (row.example_id, int(position))
                for position in row.logical_positions.tolist()
            )
            if any(key in seen_row_keys for key in keys):
                raise ValueError("full-stack row keys are duplicated")

            for ordinal in ordinals:
                selected, down = prepared[ordinal]
                first = selected[0]
                x = row.activations[first.input_site].to(
                    dtype=torch.float64
                )
                z = row.activations[first.activation_site].to(
                    dtype=torch.float64
                )
                gradient = row.score_gradients[first.activation_site].to(
                    dtype=torch.float64
                )
                if (
                    x.shape[0] != z.shape[0]
                    or x.shape[1] != first.input_width
                    or z.shape != gradient.shape
                    or z.shape[1] != down.shape[1]
                ):
                    raise ValueError(
                        "full-stack activation row shapes disagree"
                    )
                inputs[ordinal].append(x)
                contributions[ordinal].append(z @ down.T)
                fisher_weights[ordinal].append(
                    (z * gradient).square().sum(dim=1)
                )
            row_keys.extend(keys)
            seen_row_keys.update(keys)
            sequences += 1
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    if sequences <= 0:
        raise ValueError("full-stack row stream cannot be empty")

    shared_row_keys = tuple(row_keys)
    return tuple(
        FullMLPStackLayerRows(
            layer_ordinal=ordinal,
            layer_id=prepared[ordinal][0][0].layer_id,
            input_site=prepared[ordinal][0][0].input_site,
            activation_site=prepared[ordinal][0][0].activation_site,
            output_site=prepared[ordinal][0][0].output_site,
            intermediate_width=prepared[ordinal][1].shape[1],
            fragment_ids=tuple(
                value.fragment_id for value in prepared[ordinal][0]
            ),
            fragment_sha256s=tuple(
                value.artifact_sha256 for value in prepared[ordinal][0]
            ),
            inputs=torch.cat(inputs[ordinal], dim=0),
            contributions=torch.cat(contributions[ordinal], dim=0),
            fisher_weights=torch.cat(fisher_weights[ordinal], dim=0),
            row_keys=shared_row_keys,
            sequences=sequences,
        )
        for ordinal in ordinals
    )


def collect_full_mlp_stack_layer_rows(
    rows: Iterable[ActivationScoreGradientRows],
    *,
    fragments: Sequence[ParameterClusterLayerFragment],
    down_projection_weight: Tensor,
) -> FullMLPStackLayerRows:
    """Collect one complete MLP layer without per-fragment row duplication."""

    if isinstance(fragments, (str, bytes)) or not isinstance(
        fragments,
        Sequence,
    ):
        raise TypeError("fragments must be a sequence")
    values = tuple(fragments)
    if not values or not isinstance(
        values[0],
        ParameterClusterLayerFragment,
    ):
        raise ValueError("layer fragments must be nonempty fragment values")
    result = collect_full_mlp_stack_rows(
        rows,
        fragments_by_layer={values[0].layer_ordinal: values},
        down_projection_weights={
            values[0].layer_ordinal: down_projection_weight,
        },
    )
    return result[0]
