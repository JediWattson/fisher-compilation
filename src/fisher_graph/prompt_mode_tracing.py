"""Source-safe prompt-conditioned traces over native activation coordinates.

For one independently differentiated prompt, an activation coordinate ``z_j``
and its score gradient ``g_j = d score / d z_j`` define two different Fisher
summaries:

``r_j(x) = sum_t z_tj * g_tj``
    The coherent, signed derivative with respect to a virtual multiplicative
    gate on coordinate ``j``.

``q_j(x) = sum_t (z_tj * g_tj) ** 2``
    The token-additive Fisher need.  Unlike ``r_j(x) ** 2``, this does not
    include cross-position cancellation or reinforcement.

This module consumes token-position rows and retains only prompt-by-mode
numeric summaries.  It deliberately excludes prompt text, raw example ids,
token ids, decoded tokens, context windows, and token-row activations or
gradients from its authenticated artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re

import torch
from torch import Tensor

from .streaming_analysis import ActivationScoreGradientRows
from .structured_mlp_cross_block_bundling import CrossBlockLayerSpec


_ARTIFACT_KIND = "fisher_graph.prompt_conditioned_mode_trace"
_FORMAT_VERSION = 1
_ARTIFACT_DOMAIN = b"fisher_graph.prompt_mode_trace.artifact.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.prompt_mode_trace.tensor.v1\0"
_EXAMPLE_ID_DOMAIN = b"fisher_graph.prompt_mode_trace.example_id.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_prompt_text": False,
    "contains_raw_example_ids": False,
    "contains_token_ids": False,
    "contains_decoded_tokens": False,
    "contains_context_windows": False,
    "contains_token_row_activations": False,
    "contains_token_row_gradients": False,
    "analysis_only": True,
    "authorizes_intervention": False,
    "authorizes_compilation": False,
    "authorizes_execution": False,
}

_TENSOR_FIELDS = (
    "prompt_effects",
    "additive_fisher_need",
    "activation_means",
    "activation_rms",
    "activation_positive_fractions",
    "valid_row_counts",
    "losses",
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    digest = hashlib.sha256()
    digest.update(_ARTIFACT_DOMAIN)
    digest.update(_json_bytes(value))
    return digest.hexdigest()


def _tensor_sha256(value: Tensor, *, label: str) -> str:
    if not isinstance(value, Tensor) or value.device.type != "cpu":
        raise ValueError(f"{label} must be a CPU Tensor")
    if value.dtype not in (torch.float64, torch.int64):
        raise ValueError(f"{label} must use float64 or int64")
    if value.dtype == torch.float64 and not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite")
    canonical = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        f"{tuple(canonical.shape)}\0{canonical.dtype}\0".encode("utf-8")
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def prompt_example_id_sha256(example_id: str) -> str:
    """Return the source-safe identifier retained in prompt trace artifacts."""

    value = _require_nonempty(example_id, label="example_id")
    digest = hashlib.sha256()
    digest.update(_EXAMPLE_ID_DOMAIN)
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _canonical_layer_specs(
    values: Sequence[CrossBlockLayerSpec],
) -> tuple[CrossBlockLayerSpec, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("layer_specs must be a sequence")
    specs = tuple(values)
    if not specs or any(
        not isinstance(spec, CrossBlockLayerSpec) for spec in specs
    ):
        raise ValueError(
            "layer_specs must contain at least one CrossBlockLayerSpec"
        )
    expected = tuple(
        sorted(
            specs,
            key=lambda spec: (
                spec.layer_ordinal,
                spec.layer_id,
                spec.activation_site,
            ),
        )
    )
    if specs != expected:
        raise ValueError(
            "layer_specs must be in canonical layer/site order"
        )
    if len({spec.activation_site for spec in specs}) != len(specs):
        raise ValueError("activation sites must be unique")

    ordinal_to_id: dict[int, str] = {}
    id_to_ordinal: dict[str, int] = {}
    for spec in specs:
        prior_id = ordinal_to_id.setdefault(
            spec.layer_ordinal,
            spec.layer_id,
        )
        prior_ordinal = id_to_ordinal.setdefault(
            spec.layer_id,
            spec.layer_ordinal,
        )
        if prior_id != spec.layer_id or prior_ordinal != spec.layer_ordinal:
            raise ValueError(
                "layer ids and ordinals must define a one-to-one catalog"
            )
    return specs


@dataclass(frozen=True, slots=True)
class PromptModeTraceProvenance:
    """Source bindings for one prompt-conditioned trace collection."""

    source_model_fingerprint: str
    calibration_split_sha256: str
    objective_sha256: str
    score_reduction: str = "sum"
    normalizer: str = "independent_sequence"

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_model_fingerprint,
            label="source_model_fingerprint",
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
            "source_model_fingerprint": self.source_model_fingerprint,
            "calibration_split_sha256": self.calibration_split_sha256,
            "objective_sha256": self.objective_sha256,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> PromptModeTraceProvenance:
        fields = {
            "source_model_fingerprint",
            "calibration_split_sha256",
            "objective_sha256",
            "score_reduction",
            "normalizer",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("prompt trace provenance fields are invalid")
        return cls(
            source_model_fingerprint=state[  # type: ignore[arg-type]
                "source_model_fingerprint"
            ],
            calibration_split_sha256=state[  # type: ignore[arg-type]
                "calibration_split_sha256"
            ],
            objective_sha256=state["objective_sha256"],  # type: ignore[arg-type]
            score_reduction=state["score_reduction"],  # type: ignore[arg-type]
            normalizer=state["normalizer"],  # type: ignore[arg-type]
        )


def _trace_payload(
    *,
    provenance: PromptModeTraceProvenance,
    layer_specs: tuple[CrossBlockLayerSpec, ...],
    example_id_sha256s: tuple[str, ...],
    prompt_effects: Tensor,
    additive_fisher_need: Tensor,
    activation_means: Tensor,
    activation_rms: Tensor,
    activation_positive_fractions: Tensor,
    valid_row_counts: Tensor,
    losses: Tensor,
) -> dict[str, object]:
    tensors = {
        "prompt_effects": prompt_effects,
        "additive_fisher_need": additive_fisher_need,
        "activation_means": activation_means,
        "activation_rms": activation_rms,
        "activation_positive_fractions": activation_positive_fractions,
        "valid_row_counts": valid_row_counts,
        "losses": losses,
    }
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "format_version": _FORMAT_VERSION,
        **_SAFETY_METADATA,
        "provenance": provenance.metadata(),
        "layer_specs": tuple(spec.metadata() for spec in layer_specs),
        "example_id_sha256s": example_id_sha256s,
        "example_count": prompt_effects.shape[0],
        "mode_count": prompt_effects.shape[1],
        "tensor_sha256s": {
            name: _tensor_sha256(tensor, label=name)
            for name, tensor in tensors.items()
        },
    }


@dataclass(frozen=True, slots=True)
class PromptModeTrace:
    """Authenticated prompt-by-native-mode summaries with no token rows."""

    provenance: PromptModeTraceProvenance
    layer_specs: tuple[CrossBlockLayerSpec, ...]
    example_id_sha256s: tuple[str, ...]
    prompt_effects: Tensor
    additive_fisher_need: Tensor
    activation_means: Tensor
    activation_rms: Tensor
    activation_positive_fractions: Tensor
    valid_row_counts: Tensor
    losses: Tensor
    artifact_sha256: str
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_raw_example_ids: bool = False
    contains_token_ids: bool = False
    contains_decoded_tokens: bool = False
    contains_context_windows: bool = False
    contains_token_row_activations: bool = False
    contains_token_row_gradients: bool = False
    analysis_only: bool = True
    authorizes_intervention: bool = False
    authorizes_compilation: bool = False
    authorizes_execution: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, PromptModeTraceProvenance):
            raise TypeError(
                "provenance must be PromptModeTraceProvenance"
            )
        specs = _canonical_layer_specs(self.layer_specs)
        if type(self.layer_specs) is not tuple or specs != self.layer_specs:
            raise ValueError("layer_specs must be a canonical tuple")
        if type(self.example_id_sha256s) is not tuple:
            raise TypeError("example_id_sha256s must be a tuple")
        for index, value in enumerate(self.example_id_sha256s):
            _require_sha256(value, label=f"example_id_sha256s[{index}]")
        if len(set(self.example_id_sha256s)) != len(
            self.example_id_sha256s
        ):
            raise ValueError("example id hashes must be unique")

        example_count = len(self.example_id_sha256s)
        mode_count = sum(spec.width for spec in specs)
        if example_count <= 0:
            raise ValueError("a prompt trace cannot be empty")
        matrix_shape = (example_count, mode_count)
        matrix_fields = (
            "prompt_effects",
            "additive_fisher_need",
            "activation_means",
            "activation_rms",
            "activation_positive_fractions",
        )
        for name in matrix_fields:
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or value.shape != matrix_shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{name} must be a finite CPU float64 Tensor with "
                    f"shape {matrix_shape}"
                )
            object.__setattr__(
                self,
                name,
                value.detach().clone().contiguous(),
            )

        if (
            not isinstance(self.valid_row_counts, Tensor)
            or self.valid_row_counts.device.type != "cpu"
            or self.valid_row_counts.dtype != torch.int64
            or self.valid_row_counts.shape != (example_count,)
            or (self.valid_row_counts <= 0).any()
        ):
            raise ValueError(
                "valid_row_counts must be a positive CPU int64 vector "
                "with one entry per prompt"
            )
        object.__setattr__(
            self,
            "valid_row_counts",
            self.valid_row_counts.detach().clone().contiguous(),
        )
        if (
            not isinstance(self.losses, Tensor)
            or self.losses.device.type != "cpu"
            or self.losses.dtype != torch.float64
            or self.losses.shape != (example_count,)
            or not torch.isfinite(self.losses).all()
        ):
            raise ValueError(
                "losses must be a finite CPU float64 vector with one "
                "entry per prompt"
            )
        object.__setattr__(
            self,
            "losses",
            self.losses.detach().clone().contiguous(),
        )

        if (self.additive_fisher_need < 0).any():
            raise ValueError("additive_fisher_need must be nonnegative")
        if (self.activation_rms < 0).any():
            raise ValueError("activation_rms must be nonnegative")
        if (
            (self.activation_positive_fractions < 0).any()
            or (self.activation_positive_fractions > 1).any()
        ):
            raise ValueError(
                "activation_positive_fractions must be between zero and one"
            )
        rms_slack = (
            2e-12
            + 2e-11
            * torch.maximum(
                self.activation_rms,
                self.activation_means.abs(),
            )
        )
        if (
            self.activation_means.abs()
            > self.activation_rms + rms_slack
        ).any():
            raise ValueError(
                "activation RMS cannot be smaller than absolute mean"
            )

        if (
            type(self.artifact_kind) is not str
            or self.artifact_kind != _ARTIFACT_KIND
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
            or any(
                getattr(self, name) is not expected
                for name, expected in _SAFETY_METADATA.items()
            )
        ):
            raise ValueError("prompt trace safety metadata is invalid")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("prompt trace artifact hash mismatch")

    @property
    def example_count(self) -> int:
        return len(self.example_id_sha256s)

    @property
    def mode_count(self) -> int:
        return self.prompt_effects.shape[1]

    @property
    def mean_loss(self) -> float:
        return float(self.losses.mean().item())

    def site_slice(self, activation_site: str) -> slice:
        """Return the native-mode column slice for one activation site."""

        _require_nonempty(activation_site, label="activation_site")
        offset = 0
        for spec in self.layer_specs:
            next_offset = offset + spec.width
            if spec.activation_site == activation_site:
                return slice(offset, next_offset)
            offset = next_offset
        raise KeyError(f"unknown activation site {activation_site!r}")

    def _payload(self) -> dict[str, object]:
        return _trace_payload(
            provenance=self.provenance,
            layer_specs=self.layer_specs,
            example_id_sha256s=self.example_id_sha256s,
            prompt_effects=self.prompt_effects,
            additive_fisher_need=self.additive_fisher_need,
            activation_means=self.activation_means,
            activation_rms=self.activation_rms,
            activation_positive_fractions=(
                self.activation_positive_fractions
            ),
            valid_row_counts=self.valid_row_counts,
            losses=self.losses,
        )

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload())

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "mean_loss": self.mean_loss,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            **{
                name: getattr(self, name)
                for name in _SAFETY_METADATA
            },
            "provenance": self.provenance.state_dict(),
            "layer_specs": tuple(
                spec.metadata() for spec in self.layer_specs
            ),
            "example_id_sha256s": self.example_id_sha256s,
            **{
                name: getattr(self, name).detach().clone()
                for name in _TENSOR_FIELDS
            },
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> PromptModeTrace:
        fields = {
            "artifact_kind",
            "format_version",
            *_SAFETY_METADATA,
            "provenance",
            "layer_specs",
            "example_id_sha256s",
            *_TENSOR_FIELDS,
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("prompt trace artifact fields are invalid")
        raw_provenance = state["provenance"]
        raw_specs = state["layer_specs"]
        raw_ids = state["example_id_sha256s"]
        if not isinstance(raw_provenance, Mapping):
            raise TypeError("prompt trace provenance must be a mapping")
        if not isinstance(raw_specs, tuple):
            raise TypeError("prompt trace layer_specs must be a tuple")
        if not isinstance(raw_ids, tuple):
            raise TypeError(
                "prompt trace example_id_sha256s must be a tuple"
            )
        if any(not isinstance(value, Mapping) for value in raw_specs):
            raise TypeError("prompt trace layer spec states must be mappings")
        spec_fields = {
            "layer_id",
            "layer_ordinal",
            "activation_site",
            "width",
        }
        for value in raw_specs:
            if set(value) != spec_fields:
                raise ValueError("prompt trace layer spec fields are invalid")
            if (
                type(value["layer_id"]) is not str
                or type(value["activation_site"]) is not str
                or type(value["layer_ordinal"]) is not int
                or type(value["width"]) is not int
            ):
                raise TypeError("prompt trace layer spec types are invalid")
        tensors: dict[str, Tensor] = {}
        for name in _TENSOR_FIELDS:
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            tensors[name] = value
        return cls(
            provenance=PromptModeTraceProvenance.from_state_dict(
                raw_provenance
            ),
            layer_specs=tuple(
                CrossBlockLayerSpec.from_state_dict(value)
                for value in raw_specs
            ),
            example_id_sha256s=raw_ids,  # type: ignore[arg-type]
            prompt_effects=tensors["prompt_effects"],
            additive_fisher_need=tensors["additive_fisher_need"],
            activation_means=tensors["activation_means"],
            activation_rms=tensors["activation_rms"],
            activation_positive_fractions=tensors[
                "activation_positive_fractions"
            ],
            valid_row_counts=tensors["valid_row_counts"],
            losses=tensors["losses"],
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                name: state[name]
                for name in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )


def _create_prompt_mode_trace(
    *,
    provenance: PromptModeTraceProvenance,
    layer_specs: tuple[CrossBlockLayerSpec, ...],
    example_id_sha256s: tuple[str, ...],
    prompt_effects: Tensor,
    additive_fisher_need: Tensor,
    activation_means: Tensor,
    activation_rms: Tensor,
    activation_positive_fractions: Tensor,
    valid_row_counts: Tensor,
    losses: Tensor,
) -> PromptModeTrace:
    payload = _trace_payload(
        provenance=provenance,
        layer_specs=layer_specs,
        example_id_sha256s=example_id_sha256s,
        prompt_effects=prompt_effects,
        additive_fisher_need=additive_fisher_need,
        activation_means=activation_means,
        activation_rms=activation_rms,
        activation_positive_fractions=activation_positive_fractions,
        valid_row_counts=valid_row_counts,
        losses=losses,
    )
    return PromptModeTrace(
        provenance=provenance,
        layer_specs=layer_specs,
        example_id_sha256s=example_id_sha256s,
        prompt_effects=prompt_effects,
        additive_fisher_need=additive_fisher_need,
        activation_means=activation_means,
        activation_rms=activation_rms,
        activation_positive_fractions=activation_positive_fractions,
        valid_row_counts=valid_row_counts,
        losses=losses,
        artifact_sha256=_json_sha256(payload),
    )


def collect_prompt_mode_trace(
    rows: Iterable[ActivationScoreGradientRows],
    *,
    layer_specs: Sequence[CrossBlockLayerSpec],
    provenance: PromptModeTraceProvenance,
) -> PromptModeTrace:
    """Consume independent prompt rows into an authenticated numeric trace.

    Site columns are concatenated in the exact supplied canonical catalog
    order.  Each row must have a unique, nonempty ``example_id``; only its
    domain-separated SHA-256 digest is retained.  The iterator is closed when
    collection succeeds or fails.
    """

    if not isinstance(provenance, PromptModeTraceProvenance):
        raise TypeError("provenance must be PromptModeTraceProvenance")
    specs = _canonical_layer_specs(layer_specs)
    expected_sites = {spec.activation_site for spec in specs}

    example_hashes: list[str] = []
    seen_example_ids: set[str] = set()
    effect_rows: list[Tensor] = []
    additive_rows: list[Tensor] = []
    mean_rows: list[Tensor] = []
    rms_rows: list[Tensor] = []
    positive_rows: list[Tensor] = []
    row_counts: list[int] = []
    losses: list[float] = []

    iterator = iter(rows)
    try:
        for sequence_index, sequence in enumerate(iterator):
            if not isinstance(sequence, ActivationScoreGradientRows):
                raise TypeError(
                    "rows must contain ActivationScoreGradientRows"
                )
            if sequence.example_id is None:
                raise ValueError(
                    f"row {sequence_index} is missing an example_id"
                )
            if sequence.example_id in seen_example_ids:
                raise ValueError(
                    f"duplicate example_id {sequence.example_id!r}"
                )
            seen_example_ids.add(sequence.example_id)
            example_hashes.append(
                prompt_example_id_sha256(sequence.example_id)
            )

            if set(sequence.activations) != expected_sites:
                missing = sorted(
                    expected_sites - set(sequence.activations)
                )
                unexpected = sorted(
                    set(sequence.activations) - expected_sites
                )
                raise ValueError(
                    "prompt trace sites do not match the catalog; "
                    f"missing={missing}, unexpected={unexpected}"
                )

            sequence_effects: list[Tensor] = []
            sequence_additive: list[Tensor] = []
            sequence_means: list[Tensor] = []
            sequence_rms: list[Tensor] = []
            sequence_positive: list[Tensor] = []
            for spec in specs:
                activation = sequence.activations[
                    spec.activation_site
                ].to(dtype=torch.float64)
                gradient = sequence.score_gradients[
                    spec.activation_site
                ].to(dtype=torch.float64)
                if activation.shape[1] != spec.width:
                    raise ValueError(
                        f"{spec.activation_site!r} width "
                        f"{activation.shape[1]} does not match catalog "
                        f"width {spec.width}"
                    )
                influence = activation * gradient
                sequence_effects.append(influence.sum(dim=0))
                sequence_additive.append(influence.square().sum(dim=0))
                sequence_means.append(activation.mean(dim=0))
                sequence_rms.append(
                    activation.square().mean(dim=0).sqrt()
                )
                sequence_positive.append(
                    (activation > 0).to(dtype=torch.float64).mean(dim=0)
                )

            summaries = (
                sequence_effects,
                sequence_additive,
                sequence_means,
                sequence_rms,
                sequence_positive,
            )
            if any(
                not torch.isfinite(torch.cat(values)).all()
                for values in summaries
            ):
                raise ValueError(
                    f"row {sequence_index} produced a non-finite summary"
                )
            effect_rows.append(torch.cat(sequence_effects))
            additive_rows.append(torch.cat(sequence_additive))
            mean_rows.append(torch.cat(sequence_means))
            rms_rows.append(torch.cat(sequence_rms))
            positive_rows.append(torch.cat(sequence_positive))
            row_counts.append(sequence.observations)
            losses.append(sequence.loss)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    if not effect_rows:
        raise ValueError("cannot trace an empty prompt row stream")
    return _create_prompt_mode_trace(
        provenance=provenance,
        layer_specs=specs,
        example_id_sha256s=tuple(example_hashes),
        prompt_effects=torch.stack(effect_rows).to(dtype=torch.float64),
        additive_fisher_need=torch.stack(additive_rows).to(
            dtype=torch.float64
        ),
        activation_means=torch.stack(mean_rows).to(dtype=torch.float64),
        activation_rms=torch.stack(rms_rows).to(dtype=torch.float64),
        activation_positive_fractions=torch.stack(positive_rows).to(
            dtype=torch.float64
        ),
        valid_row_counts=torch.tensor(row_counts, dtype=torch.int64),
        losses=torch.tensor(losses, dtype=torch.float64),
    )


__all__ = [
    "PromptModeTrace",
    "PromptModeTraceProvenance",
    "collect_prompt_mode_trace",
    "prompt_example_id_sha256",
]
