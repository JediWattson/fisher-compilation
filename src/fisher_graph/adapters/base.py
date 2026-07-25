"""Model-independent contracts for analysis and segment compilation.

Adapters isolate the compiler from a model library's forward signature, layer
container, attention-mask representation, and cache implementation.  The
portable specifications in this module intentionally contain no module
objects.  Runtime-only values, including prepared masks and caches, live in
``SequenceContext``.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from ..activations import ActivationIntervention, ActivationTrace

ActivationRole = Literal[
    "segment_input",
    "segment_output",
    "internal",
    "model_output",
]
LengthPolicy = Literal["fixed", "bounded_dynamic", "dynamic"]
PaddingSide = Literal["none", "left", "right", "either", "sparse"]
MaskRepresentation = Literal[
    "boolean_valid",
    "binary_valid",
    "additive",
    "adapter_owned",
]
ExecutionPhase = Literal["prefill", "decode"]


def _require_nonempty(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class ActivationSite:
    """One stable, adapter-owned activation identifier.

    Modal-analysis sites should use the canonical axis layout
    ``("batch", "sequence", "feature")``.  Other layouts remain useful for
    inspection but are not implicitly flattened by the adapter.

    ``alias_of`` describes baseline tensor identity at an adjacent boundary.
    The alias remains a distinct intervention point: intervening on the later
    site is observably different from intervening on the earlier site.
    """

    id: str
    role: ActivationRole
    axes: tuple[str, ...]
    width: int | None
    owner_layer: str | None = None
    alias_of: str | None = None
    intervenable: bool = True
    fisher_default: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.id, field="activation site id")
        if self.role not in (
            "segment_input",
            "segment_output",
            "internal",
            "model_output",
        ):
            raise ValueError(f"unsupported activation role: {self.role!r}")
        if not self.axes or any(
            not isinstance(axis, str) or not axis for axis in self.axes
        ):
            raise ValueError("activation axes must be nonempty strings")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("activation axes cannot contain duplicates")
        if self.width is not None and (
            type(self.width) is not int or self.width <= 0
        ):
            raise ValueError("activation width must be positive when provided")
        if self.owner_layer is not None:
            _require_nonempty(
                self.owner_layer,
                field="activation owner_layer",
            )
        if self.alias_of is not None:
            _require_nonempty(self.alias_of, field="activation alias_of")
            if self.alias_of == self.id:
                raise ValueError("an activation site cannot alias itself")
        if type(self.intervenable) is not bool:
            raise ValueError("intervenable must be boolean")
        if type(self.fisher_default) is not bool:
            raise ValueError("fisher_default must be boolean")
        if self.fisher_default and not self.modal_eligible:
            raise ValueError(
                "default Fisher sites must use "
                "[batch, sequence, feature] axes"
            )

    @property
    def modal_eligible(self) -> bool:
        """Whether the site already has the residual-stream modal layout."""

        return (
            self.axes == ("batch", "sequence", "feature")
            and self.width is not None
        )


@dataclass(frozen=True, slots=True)
class MaskPolicy:
    """Static attention-mask semantics exposed by one model adapter."""

    causal: bool
    padding_side: PaddingSide
    representation: MaskRepresentation = "boolean_valid"
    requires_first_token_valid: bool = False

    def __post_init__(self) -> None:
        if type(self.causal) is not bool:
            raise ValueError("causal must be boolean")
        if self.padding_side not in (
            "none",
            "left",
            "right",
            "either",
            "sparse",
        ):
            raise ValueError(f"unsupported padding side: {self.padding_side!r}")
        if self.representation not in (
            "boolean_valid",
            "binary_valid",
            "additive",
            "adapter_owned",
        ):
            raise ValueError(
                f"unsupported mask representation: {self.representation!r}"
            )
        if type(self.requires_first_token_valid) is not bool:
            raise ValueError("requires_first_token_valid must be boolean")
        if self.padding_side == "left" and self.requires_first_token_valid:
            raise ValueError(
                "left padding conflicts with requires_first_token_valid"
            )


@dataclass(frozen=True, slots=True)
class RopeSpec:
    """Portable rotary-position configuration for a layer, when applicable."""

    kind: str
    theta: float | None = None
    rotary_dimension: int | None = None
    scaling_kind: str | None = None
    scaling_factor: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.kind, field="RoPE kind")
        if self.theta is not None and (
            not isinstance(self.theta, (float, int))
            or isinstance(self.theta, bool)
            or not torch.isfinite(torch.tensor(float(self.theta)))
            or self.theta <= 0
        ):
            raise ValueError("RoPE theta must be finite and positive")
        if self.rotary_dimension is not None and (
            type(self.rotary_dimension) is not int
            or self.rotary_dimension <= 0
        ):
            raise ValueError(
                "rotary_dimension must be positive when provided"
            )
        if self.scaling_kind is not None:
            _require_nonempty(
                self.scaling_kind,
                field="RoPE scaling_kind",
            )
        if self.scaling_factor is not None and (
            not isinstance(self.scaling_factor, (float, int))
            or isinstance(self.scaling_factor, bool)
            or not torch.isfinite(torch.tensor(float(self.scaling_factor)))
            or self.scaling_factor <= 0
        ):
            raise ValueError("RoPE scaling_factor must be finite and positive")


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """Layer-local attention topology.

    Layer-local metadata is necessary for decoders that alternate global and
    sliding-window attention or use different cache policies by layer.
    """

    kind: str
    query_heads: int
    key_value_heads: int
    head_dimension: int
    query_scale: float | None
    qk_norm: bool
    window_size: int | None = None
    rope: RopeSpec | None = None
    cache_kind: str = "none"

    def __post_init__(self) -> None:
        _require_nonempty(self.kind, field="attention kind")
        for name, value in (
            ("query_heads", self.query_heads),
            ("key_value_heads", self.key_value_heads),
            ("head_dimension", self.head_dimension),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.query_heads % self.key_value_heads != 0:
            raise ValueError(
                "query_heads must be divisible by key_value_heads"
            )
        if self.query_scale is not None and (
            not isinstance(self.query_scale, (float, int))
            or isinstance(self.query_scale, bool)
            or not torch.isfinite(torch.tensor(float(self.query_scale)))
            or self.query_scale <= 0
        ):
            raise ValueError(
                "query_scale must be finite and positive when provided"
            )
        if type(self.qk_norm) is not bool:
            raise ValueError("qk_norm must be boolean")
        if self.window_size is not None and (
            type(self.window_size) is not int or self.window_size <= 0
        ):
            raise ValueError("window_size must be positive when provided")
        if self.rope is not None and not isinstance(self.rope, RopeSpec):
            raise TypeError("rope must be a RopeSpec when provided")
        _require_nonempty(self.cache_kind, field="attention cache_kind")


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    """Static sequence capabilities of a source model or compiled backend."""

    length_policy: LengthPolicy
    minimum_length: int
    maximum_length: int | None
    mask: MaskPolicy
    position_kind: str
    supports_prefill: bool = True
    supports_decode: bool = False
    cache_kind: str = "none"

    def __post_init__(self) -> None:
        if self.length_policy not in (
            "fixed",
            "bounded_dynamic",
            "dynamic",
        ):
            raise ValueError(
                f"unsupported length policy: {self.length_policy!r}"
            )
        if type(self.minimum_length) is not int or self.minimum_length <= 0:
            raise ValueError("minimum_length must be positive")
        if self.maximum_length is not None and (
            type(self.maximum_length) is not int
            or self.maximum_length < self.minimum_length
        ):
            raise ValueError(
                "maximum_length must be at least minimum_length"
            )
        if self.length_policy in ("fixed", "bounded_dynamic"):
            if self.maximum_length is None:
                raise ValueError(
                    f"{self.length_policy} requires maximum_length"
                )
        if (
            self.length_policy == "fixed"
            and self.maximum_length != self.minimum_length
        ):
            raise ValueError(
                "fixed length policy requires equal minimum and maximum"
            )
        if not isinstance(self.mask, MaskPolicy):
            raise TypeError("mask must be a MaskPolicy")
        _require_nonempty(self.position_kind, field="position_kind")
        if type(self.supports_prefill) is not bool:
            raise ValueError("supports_prefill must be boolean")
        if type(self.supports_decode) is not bool:
            raise ValueError("supports_decode must be boolean")
        if not self.supports_prefill and not self.supports_decode:
            raise ValueError(
                "a sequence specification must support prefill or decode"
            )
        _require_nonempty(self.cache_kind, field="sequence cache_kind")
        if self.supports_decode and self.cache_kind == "none":
            raise ValueError("decode support requires a cache kind")

    def validate_length(self, length: int) -> None:
        if type(length) is not int or length < self.minimum_length:
            raise ValueError(
                f"sequence length must be at least {self.minimum_length}"
            )
        if self.maximum_length is not None and length > self.maximum_length:
            raise ValueError(
                f"sequence length {length} exceeds maximum "
                f"{self.maximum_length}"
            )


@dataclass(slots=True)
class SequenceContext:
    """Runtime sequence state shared by source and compiled segment calls.

    ``logical_positions`` and ``key_logical_positions`` describe semantic
    query/key locations. ``cache_positions`` select physical cache slots.
    They are intentionally separate because sliding, static, and hybrid
    caches need not use logical token positions as their storage indices.

    ``adapter_payload`` may hold model-library-specific prepared values such as
    a four-dimensional additive mask or rotary sine/cosine tensors.  Compiler
    passes must treat it as opaque.
    """

    query_valid_mask: Tensor
    key_valid_mask: Tensor
    logical_positions: Tensor
    key_logical_positions: Tensor
    cache_positions: Tensor | None
    phase: ExecutionPhase
    input_origin: SequenceInputOrigin
    cache_state: object | None = None
    adapter_payload: object | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("query_valid_mask", self.query_valid_mask),
            ("key_valid_mask", self.key_valid_mask),
            ("logical_positions", self.logical_positions),
            ("key_logical_positions", self.key_logical_positions),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if value.ndim != 2:
                raise ValueError(f"{name} must have shape [batch, sequence]")
        if self.query_valid_mask.dtype is not torch.bool:
            raise ValueError("query_valid_mask must be boolean")
        if self.key_valid_mask.dtype is not torch.bool:
            raise ValueError("key_valid_mask must be boolean")
        if self.logical_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("logical_positions must use an integer dtype")
        batch_size, query_length = self.query_valid_mask.shape
        if self.key_valid_mask.shape[0] != batch_size:
            raise ValueError("query and key masks must share a batch size")
        if self.logical_positions.shape != (batch_size, query_length):
            raise ValueError(
                "logical_positions must match the query mask shape"
            )
        if self.key_logical_positions.shape != self.key_valid_mask.shape:
            raise ValueError(
                "key_logical_positions must match the key mask shape"
            )
        if self.key_logical_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "key_logical_positions must use an integer dtype"
            )
        devices = {
            self.query_valid_mask.device,
            self.key_valid_mask.device,
            self.logical_positions.device,
            self.key_logical_positions.device,
        }
        if len(devices) != 1:
            raise ValueError(
                "sequence masks and logical positions must share a device"
            )
        if self.cache_positions is not None:
            if not isinstance(self.cache_positions, Tensor):
                raise TypeError("cache_positions must be a Tensor")
            if self.cache_positions.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "cache_positions must use an integer dtype"
                )
            if self.cache_positions.device != self.logical_positions.device:
                raise ValueError(
                    "cache_positions must share the sequence-context device"
                )
            if self.cache_positions.ndim == 1:
                if self.cache_positions.shape[0] != query_length:
                    raise ValueError(
                        "one-dimensional cache_positions must have query length"
                    )
            elif self.cache_positions.shape != (batch_size, query_length):
                raise ValueError(
                    "cache_positions must have shape [query] or [batch, query]"
                )
        if self.phase not in ("prefill", "decode"):
            raise ValueError(f"unsupported execution phase: {self.phase!r}")
        if not isinstance(self.input_origin, SequenceInputOrigin):
            raise TypeError("input_origin must be a SequenceInputOrigin")

    @property
    def batch_size(self) -> int:
        return self.query_valid_mask.shape[0]

    @property
    def query_length(self) -> int:
        return self.query_valid_mask.shape[1]

    @property
    def key_length(self) -> int:
        return self.key_valid_mask.shape[1]

    @property
    def device(self) -> torch.device:
        return self.logical_positions.device


@dataclass(frozen=True, slots=True)
class SequenceInputOrigin:
    """Caller-input provenance retained after adapter normalization.

    Normalized masks and positions intentionally erase model-library details,
    but runtime ABI guards still need to distinguish an omitted input from an
    explicitly supplied one.  These flags describe standardized input roles,
    not model-specific keyword names.
    """

    attention_mask_supplied: bool
    position_ids_supplied: bool
    cache_positions_supplied: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("attention_mask_supplied", self.attention_mask_supplied),
            ("position_ids_supplied", self.position_ids_supplied),
            ("cache_positions_supplied", self.cache_positions_supplied),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be boolean")


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """One native layer and its potentially heterogeneous execution policy."""

    id: str
    ordinal: int
    input_site: str
    output_site: str
    residual_width: int
    kind: str
    attention: AttentionSpec | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.id, field="layer id")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("layer ordinal must be nonnegative")
        _require_nonempty(self.input_site, field="layer input_site")
        _require_nonempty(self.output_site, field="layer output_site")
        if self.input_site == self.output_site:
            raise ValueError("layer input and output sites must differ")
        if (
            type(self.residual_width) is not int
            or self.residual_width <= 0
        ):
            raise ValueError("layer residual_width must be positive")
        _require_nonempty(self.kind, field="layer kind")
        if self.attention is not None and not isinstance(
            self.attention,
            AttentionSpec,
        ):
            raise TypeError("attention must be an AttentionSpec when provided")
        if self.source_path is not None:
            _require_nonempty(self.source_path, field="layer source_path")


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """A contiguous compiler unit made from one or more native layers."""

    id: str
    ordinal: int
    layer_ids: tuple[str, ...]
    input_site: str
    output_site: str
    input_width: int
    output_width: int

    def __post_init__(self) -> None:
        _require_nonempty(self.id, field="segment id")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("segment ordinal must be nonnegative")
        if not self.layer_ids:
            raise ValueError("a segment must contain at least one layer")
        if any(
            not isinstance(layer_id, str) or not layer_id
            for layer_id in self.layer_ids
        ):
            raise ValueError("segment layer ids must be nonempty strings")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("segment layer ids cannot contain duplicates")
        _require_nonempty(self.input_site, field="segment input_site")
        _require_nonempty(self.output_site, field="segment output_site")
        if self.input_site == self.output_site:
            raise ValueError("segment input and output sites must differ")
        for name, value in (
            ("input_width", self.input_width),
            ("output_width", self.output_width),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class LayerBlockBoundaryPlan:
    """Canonical residual boundaries for a contiguous native-layer block.

    The first boundary is the selected block's explicit input intervention
    site.  Every later boundary is a native layer output.  In particular, the
    plan omits a later layer's input alias when it names the same baseline
    tensor as the preceding output.  This prevents Fisher collection from
    counting one residual boundary twice while preserving a distinct leaf at
    the entrance to an arbitrary block.
    """

    layer_ids: tuple[str, ...]
    layer_ordinals: tuple[int, ...]
    activation_sites: tuple[str, ...]
    widths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.layer_ids:
            raise ValueError("a layer block must contain at least one layer")
        if len(self.layer_ids) != len(self.layer_ordinals):
            raise ValueError("layer ids and ordinals must have equal length")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("layer block ids cannot contain duplicates")
        if any(
            not isinstance(layer_id, str) or not layer_id
            for layer_id in self.layer_ids
        ):
            raise ValueError("layer block ids must be nonempty strings")
        if any(
            type(ordinal) is not int or ordinal < 0
            for ordinal in self.layer_ordinals
        ):
            raise ValueError("layer block ordinals must be nonnegative")
        expected_ordinals = tuple(
            range(
                self.layer_ordinals[0],
                self.layer_ordinals[0] + len(self.layer_ordinals),
            )
        )
        if self.layer_ordinals != expected_ordinals:
            raise ValueError("layer block ordinals must be contiguous")
        if len(self.activation_sites) != len(self.layer_ids) + 1:
            raise ValueError(
                "a layer block must have one more boundary than layers"
            )
        if any(
            not isinstance(site, str) or not site
            for site in self.activation_sites
        ):
            raise ValueError("block activation sites must be nonempty strings")
        if len(set(self.activation_sites)) != len(self.activation_sites):
            raise ValueError("block activation sites must be unique")
        if len(self.widths) != len(self.activation_sites):
            raise ValueError("block widths must match activation boundaries")
        if any(type(width) is not int or width <= 0 for width in self.widths):
            raise ValueError("block boundary widths must be positive")

    @property
    def start_ordinal(self) -> int:
        return self.layer_ordinals[0]

    @property
    def end_ordinal(self) -> int:
        return self.layer_ordinals[-1]

    @property
    def leaf_activation_name(self) -> str:
        """Boundary to detach when only the block suffix needs autograd."""

        return self.activation_sites[0]

    @property
    def transitions(self) -> tuple[tuple[str, str], ...]:
        """Adjacent input/output boundary pairs in execution order."""

        return tuple(zip(self.activation_sites, self.activation_sites[1:]))


@dataclass(slots=True)
class AdapterRun:
    """Normalized result of a complete model invocation."""

    logits: Tensor
    activations: Mapping[str, Tensor]
    sequence: SequenceContext
    raw_output: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logits, Tensor):
            raise TypeError("adapter logits must be a Tensor")
        if not isinstance(self.activations, Mapping):
            raise TypeError("adapter activations must be a mapping")
        if any(
            not isinstance(name, str) or not isinstance(value, Tensor)
            for name, value in self.activations.items()
        ):
            raise TypeError("adapter activations must map strings to Tensors")
        if not isinstance(self.sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")


@dataclass(slots=True)
class SegmentRun:
    """Normalized result of a source or compiled segment invocation."""

    hidden_states: Tensor
    sequence: SequenceContext
    raw_output: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_states, Tensor):
            raise TypeError("segment hidden_states must be a Tensor")
        if not isinstance(self.sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")


def module_state_fingerprint(module: nn.Module) -> str:
    """Hash tensor names, shapes, dtypes, and bytes in a module state dict."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not isinstance(value, Tensor):
            raise TypeError(f"state value {name!r} is not a Tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ModelAdapter(ABC):
    """Boundary between compiler analysis and a concrete model family."""

    @property
    @abstractmethod
    def module(self) -> nn.Module:
        """Underlying source model without registering it on the adapter."""

    @property
    @abstractmethod
    def sequence_spec(self) -> SequenceSpec:
        """Static source-model sequence capabilities."""

    @property
    @abstractmethod
    def activation_sites(self) -> tuple[ActivationSite, ...]:
        """Stable activation catalog in execution order."""

    @property
    @abstractmethod
    def layers(self) -> tuple[LayerSpec, ...]:
        """Native layer catalog in execution order."""

    @property
    @abstractmethod
    def segments(self) -> tuple[SegmentSpec, ...]:
        """Default independently replaceable compiler segments."""

    @property
    def default_fisher_sites(self) -> tuple[str, ...]:
        """Recommended ordered activation set for a standard calibration run."""

        return tuple(
            site.id for site in self.activation_sites if site.fisher_default
        )

    def plan_layer_block(
        self,
        start_ordinal: int,
        end_ordinal: int,
    ) -> LayerBlockBoundaryPlan:
        """Plan nonduplicated residual boundaries for an inclusive layer range.

        Adjacent layers may expose their shared residual tensor either under
        one canonical name or as a later input site whose ``alias_of`` points
        to the preceding output.  Both adapter conventions collapse to the
        same ordered boundary plan.
        """

        for label, value in (
            ("start_ordinal", start_ordinal),
            ("end_ordinal", end_ordinal),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be nonnegative")
        if end_ordinal < start_ordinal:
            raise ValueError("end_ordinal cannot precede start_ordinal")
        layer_catalog = self.layers
        layers_by_ordinal = {layer.ordinal: layer for layer in layer_catalog}
        if len(layers_by_ordinal) != len(layer_catalog):
            raise ValueError("adapter layer ordinals must be unique")
        ordinals = tuple(range(start_ordinal, end_ordinal + 1))
        missing = set(ordinals) - layers_by_ordinal.keys()
        if missing:
            raise ValueError(
                "layer block ordinals are outside the adapter catalog: "
                f"{sorted(missing)}"
            )
        selected = tuple(layers_by_ordinal[index] for index in ordinals)
        sites_by_id = {site.id: site for site in self.activation_sites}
        boundaries = (selected[0].input_site,) + tuple(
            layer.output_site for layer in selected
        )
        widths = tuple(layer.residual_width for layer in selected)
        boundary_widths = (selected[0].residual_width, *widths)

        for site_name, expected_width in zip(
            boundaries,
            boundary_widths,
            strict=True,
        ):
            try:
                site = sites_by_id[site_name]
            except KeyError as error:
                raise ValueError(
                    f"layer block boundary {site_name!r} is not cataloged"
                ) from error
            if not site.modal_eligible:
                raise ValueError(
                    f"layer block boundary {site_name!r} is not modal eligible"
                )
            if site.width != expected_width:
                raise ValueError(
                    f"layer block boundary {site_name!r} width does not "
                    "match its layer"
                )
        for previous, current in zip(selected, selected[1:]):
            if current.input_site == previous.output_site:
                if current.residual_width != previous.residual_width:
                    raise ValueError(
                        "adjacent canonical layer boundaries disagree on width"
                    )
                continue
            try:
                current_input = sites_by_id[current.input_site]
            except KeyError as error:
                raise ValueError(
                    f"layer input {current.input_site!r} is not cataloged"
                ) from error
            if current_input.alias_of != previous.output_site:
                raise ValueError(
                    f"{current.input_site!r} must alias preceding boundary "
                    f"{previous.output_site!r}"
                )
            if current_input.width not in (
                previous.residual_width,
                current.residual_width,
            ) or previous.residual_width != current.residual_width:
                raise ValueError(
                    "adjacent layer boundary aliases disagree on width"
                )
        return LayerBlockBoundaryPlan(
            layer_ids=tuple(layer.id for layer in selected),
            layer_ordinals=ordinals,
            activation_sites=boundaries,
            widths=boundary_widths,
        )

    def activation_site(self, site_id: str) -> ActivationSite:
        for site in self.activation_sites:
            if site.id == site_id:
                return site
        raise KeyError(f"unknown activation site: {site_id!r}")

    def layer(self, layer_id: str) -> LayerSpec:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(f"unknown layer: {layer_id!r}")

    def segment(self, segment_id: str) -> SegmentSpec:
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        raise KeyError(f"unknown segment: {segment_id!r}")

    @abstractmethod
    def prepare_sequence(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        phase: ExecutionPhase = "prefill",
        cache_state: object | None = None,
    ) -> SequenceContext:
        """Normalize logical positions, masks, and cache state for one run."""

    @abstractmethod
    def forward(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        phase: ExecutionPhase = "prefill",
        cache_state: object | None = None,
        capture_sites: Collection[str] = (),
        interventions: Mapping[str, ActivationIntervention] | None = None,
        retain_gradients: bool = False,
    ) -> AdapterRun:
        """Run the complete model with adapter-owned instrumentation."""

    @abstractmethod
    def embed(
        self,
        model_inputs: Mapping[str, Tensor],
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        """Produce the residual stream at the first segment boundary."""

    @abstractmethod
    def project_logits(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> Tensor:
        """Run the post-layer normalization/head and return model logits."""

    @abstractmethod
    def run_segment(
        self,
        segment: SegmentSpec,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        """Run one native segment against an already prepared context."""

    @abstractmethod
    def source_module(self, layer_id: str) -> nn.Module:
        """Return the native module for provenance and diagnostics."""

    def semantic_fingerprint(self) -> str:
        """Hash adapter-owned execution semantics without tensor state."""

        payload = {
            "adapter_class": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "sequence": asdict(self.sequence_spec),
            "activation_sites": [
                asdict(site) for site in self.activation_sites
            ],
            "layers": [asdict(layer) for layer in self.layers],
            "segments": [asdict(segment) for segment in self.segments],
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def model_fingerprint(self) -> str:
        """Bind source tensor state to adapter-visible execution semantics."""

        digest = hashlib.sha256()
        digest.update(self.semantic_fingerprint().encode("ascii"))
        digest.update(module_state_fingerprint(self.module).encode("ascii"))
        return digest.hexdigest()

    @abstractmethod
    def execution_fingerprint(self) -> str:
        """Hash live, non-tensor values that can affect model execution.

        The mixed runtime separately authenticates ``state_dict`` tensor
        values.  This fingerprint must cover mutable execution options that
        tensor state does not represent, such as attention scales, norm
        epsilon values, dropout configuration, and model-family flags.
        """

    def segment_fingerprint(self, segment: SegmentSpec) -> str:
        """Bind segment tensor state to its layer and sequence semantics."""

        digest = hashlib.sha256()
        semantic_payload = {
            "sequence": asdict(self.sequence_spec),
            "segment": asdict(segment),
        }
        digest.update(
            json.dumps(
                semantic_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for layer_id in segment.layer_ids:
            digest.update(layer_id.encode("utf-8"))
            digest.update(
                json.dumps(
                    asdict(self.layer(layer_id)),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            digest.update(
                module_state_fingerprint(
                    self.source_module(layer_id)
                ).encode("ascii")
            )
        return digest.hexdigest()

    @abstractmethod
    def replaced_segments(
        self,
        replacements: Mapping[str, nn.Module],
    ) -> AbstractContextManager[None]:
        """Atomically install replacements and restore originals on exit."""


__all__ = [
    "ActivationRole",
    "ActivationSite",
    "AdapterRun",
    "AttentionSpec",
    "ExecutionPhase",
    "LayerSpec",
    "LayerBlockBoundaryPlan",
    "LengthPolicy",
    "MaskPolicy",
    "MaskRepresentation",
    "ModelAdapter",
    "PaddingSide",
    "RopeSpec",
    "SegmentRun",
    "SegmentSpec",
    "SequenceContext",
    "SequenceInputOrigin",
    "SequenceSpec",
    "module_state_fingerprint",
]
