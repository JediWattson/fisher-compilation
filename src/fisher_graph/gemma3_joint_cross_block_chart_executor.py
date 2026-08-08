"""Joint whole-model chart runtime for the pinned Gemma 3 event graph.

This module closes the execution abstraction that the earlier MLP-only chart
overlay deliberately left open.  One executor owns the tied lexical ingress
and egress, all 18 sequence-aware attention mutations, all 18 token-local MLP
mutations, the ordered residual carrier, terminal normalization, and sparse
strictly-forward transports that may cross block and component boundaries.

The runtime consumes already-fitted modules.  It does not load Gemma, prompts,
or a teacher, and it does not prescribe how Fisher rows or charts are fitted.
In particular, the existence of this executor is not compression evidence:
parameter, fidelity, and compute claims belong to the model-facing worker that
materializes and evaluates a concrete candidate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import re
import torch
from torch import Tensor, nn

from .adapters import SequenceContext
from .residual_carrier import (
    ResidualCarrierFactory,
    ResidualCarrierSession,
    ResidualGraphExecutionContext,
    ResidualMutationReceipt,
)


PINNED_GEMMA3_BLOCK_COUNT = 18
PINNED_GEMMA3_GLOBAL_ATTENTION_BLOCKS = (5, 11, 17)
PINNED_GEMMA3_MUTATION_COUNT = 36
PINNED_GEMMA3_EVENT_COUNT = 39

JOINT_CROSS_BLOCK_CHART_ARM_ID = "joint_cross_block_chart_candidate"
JOINT_EXACT_CONTROL_ARM_ID = "joint_exact_39_event_control"

_EVENT_KINDS = frozenset(
    {
        "lexical_ingress",
        "attention_mutation",
        "mlp_mutation",
        "terminal_rmsnorm",
        "lexical_egress",
    }
)
_ATTENTION_KINDS = frozenset({"sliding_512", "global"})
_TOKEN_DEPENDENCIES = frozenset(
    {"vocabulary_boundary", "causal_prefix_mixer", "token_local"}
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")


def _require_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _module_physical_spans(module: nn.Module) -> dict[int, int]:
    """Return persistent physical tensor spans, including tied storage once."""

    spans: dict[int, tuple[int, int]] = {}
    for value in module.state_dict().values():
        if not isinstance(value, Tensor) or not value.numel():
            continue
        storage = value.untyped_storage()
        pointer = storage.data_ptr()
        if pointer:
            receipt = storage.nbytes(), value.element_size()
            previous = spans.get(pointer)
            if previous is not None and previous[1] != receipt[1]:
                raise ValueError("one storage span has inconsistent scalar views")
            if previous is None or receipt[0] > previous[0]:
                spans[pointer] = receipt
    return {
        pointer: byte_count // scalar_bytes
        for pointer, (byte_count, scalar_bytes) in spans.items()
    }


def _module_scalar_count(module: nn.Module) -> int:
    return sum(_module_physical_spans(module).values())


def _tensor_value_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"fisher_graph:joint-event-tensor:v1\0")
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class JointGraphEventSpec:
    """One event in the exact Gemma ingress/body/norm/egress schedule."""

    event_index: int
    event_id: str
    event_kind: str
    block_ordinal: int | None
    attention_kind: str | None
    token_dependency: str

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("event_index must be nonnegative")
        _require_id(self.event_id, label="event_id")
        if self.event_kind not in _EVENT_KINDS:
            raise ValueError("event_kind is invalid")
        if self.token_dependency not in _TOKEN_DEPENDENCIES:
            raise ValueError("token_dependency is invalid")

        is_mutation = self.event_kind in {
            "attention_mutation",
            "mlp_mutation",
        }
        if is_mutation:
            if (
                type(self.block_ordinal) is not int
                or self.block_ordinal < 0
            ):
                raise ValueError("mutation events require a block ordinal")
        elif self.block_ordinal is not None:
            raise ValueError("boundary events cannot have a block ordinal")

        if self.event_kind == "attention_mutation":
            if (
                self.attention_kind not in _ATTENTION_KINDS
                or self.token_dependency != "causal_prefix_mixer"
            ):
                raise ValueError("attention event contract is invalid")
        elif self.attention_kind is not None:
            raise ValueError("only attention events may name attention_kind")

        expected_dependency = {
            "lexical_ingress": "vocabulary_boundary",
            "attention_mutation": "causal_prefix_mixer",
            "mlp_mutation": "token_local",
            "terminal_rmsnorm": "token_local",
            "lexical_egress": "vocabulary_boundary",
        }[self.event_kind]
        if self.token_dependency != expected_dependency:
            raise ValueError("event token dependency disagrees with its kind")

    @property
    def component_family(self) -> str:
        if self.event_kind == "attention_mutation":
            return "attention"
        if self.event_kind == "mlp_mutation":
            return "mlp"
        if self.event_kind in {"lexical_ingress", "lexical_egress"}:
            return "lexical"
        return "terminal"

    def metadata(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "block_ordinal": self.block_ordinal,
            "attention_kind": self.attention_kind,
            "token_dependency": self.token_dependency,
            "component_family": self.component_family,
        }


def build_joint_cross_block_event_schedule(
    layer_count: int = PINNED_GEMMA3_BLOCK_COUNT,
) -> tuple[JointGraphEventSpec, ...]:
    """Build the deterministic 39-event schedule used by the Gemma arm."""

    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError("layer_count must be positive")
    events = [
        JointGraphEventSpec(
            event_index=0,
            event_id="lexical.ingress",
            event_kind="lexical_ingress",
            block_ordinal=None,
            attention_kind=None,
            token_dependency="vocabulary_boundary",
        )
    ]
    for block in range(layer_count):
        attention_index = len(events)
        attention_kind = (
            "global"
            if block in PINNED_GEMMA3_GLOBAL_ATTENTION_BLOCKS
            else "sliding_512"
        )
        events.append(
            JointGraphEventSpec(
                event_index=attention_index,
                event_id=f"mutation.{attention_index}",
                event_kind="attention_mutation",
                block_ordinal=block,
                attention_kind=attention_kind,
                token_dependency="causal_prefix_mixer",
            )
        )
        mlp_index = len(events)
        events.append(
            JointGraphEventSpec(
                event_index=mlp_index,
                event_id=f"mutation.{mlp_index}",
                event_kind="mlp_mutation",
                block_ordinal=block,
                attention_kind=None,
                token_dependency="token_local",
            )
        )
    terminal_index = len(events)
    events.extend(
        (
            JointGraphEventSpec(
                event_index=terminal_index,
                event_id="terminal.rmsnorm",
                event_kind="terminal_rmsnorm",
                block_ordinal=None,
                attention_kind=None,
                token_dependency="token_local",
            ),
            JointGraphEventSpec(
                event_index=terminal_index + 1,
                event_id="lexical.egress",
                event_kind="lexical_egress",
                block_ordinal=None,
                attention_kind=None,
                token_dependency="vocabulary_boundary",
            ),
        )
    )
    return tuple(events)


@dataclass(frozen=True, slots=True)
class JointGraphMutationRuntimeReceipt:
    """Normalized source-tripwire and rank receipt for one mutation call."""

    event_id: str
    event_kind: str
    endpoint_kind: str
    incoming_generator_rank: int
    emitted_generator_rank: int
    operator_receipt_present: bool
    generator_kind: str | None
    attention_kind: str | None
    logical_causal_key_pairs: int | None
    source_projection_call_count: int | None
    source_fallback_call_count: int | None

    def metadata(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "endpoint_kind": self.endpoint_kind,
            "incoming_generator_rank": self.incoming_generator_rank,
            "emitted_generator_rank": self.emitted_generator_rank,
            "operator_receipt_present": self.operator_receipt_present,
            "generator_kind": self.generator_kind,
            "attention_kind": self.attention_kind,
            "logical_causal_key_pairs": self.logical_causal_key_pairs,
            "source_projection_call_count": (
                self.source_projection_call_count
            ),
            "source_fallback_call_count": self.source_fallback_call_count,
        }


@dataclass(frozen=True, slots=True)
class JointGraphMutationExecution:
    """One fitted mutation's residual delta and transport packet."""

    chart_input: Tensor
    delta: Tensor
    emitted_generator_latents: Tensor
    runtime_receipt: JointGraphMutationRuntimeReceipt

    def __post_init__(self) -> None:
        if (
            not isinstance(self.chart_input, Tensor)
            or self.chart_input.ndim != 3
        ):
            raise ValueError(
                "mutation chart input must be [batch, sequence, width]"
            )
        if not isinstance(self.delta, Tensor) or self.delta.ndim != 3:
            raise ValueError("mutation delta must be [batch, sequence, width]")
        if self.chart_input.shape != self.delta.shape:
            raise ValueError("mutation chart input and delta shapes disagree")
        if (
            not isinstance(self.emitted_generator_latents, Tensor)
            or self.emitted_generator_latents.ndim != 4
            or self.emitted_generator_latents.shape[:2]
            != self.delta.shape[:2]
        ):
            raise ValueError(
                "emitted latents must be [batch, sequence, charts, rank]"
            )
        if (
            self.emitted_generator_latents.device != self.delta.device
            or self.emitted_generator_latents.dtype != self.delta.dtype
            or self.chart_input.device != self.delta.device
            or self.chart_input.dtype != self.delta.dtype
        ):
            raise ValueError("mutation output tensors must share dtype/device")
        if not isinstance(
            self.runtime_receipt,
            JointGraphMutationRuntimeReceipt,
        ):
            raise TypeError("mutation runtime receipt is invalid")


@dataclass(frozen=True, slots=True)
class FactorizedHiddenChartExecution:
    """Inspectable output from a compact hidden-state chart generator."""

    output: Tensor
    memberships: Tensor
    coordinates: Tensor
    emitted_generator_latents: Tensor


class FactorizedHiddenChartGenerator(nn.Module):
    """Compact chart correction around an identity or affine hidden base.

    The identity base is an operation and stores no dense identity matrix.
    Local corrections use ``C`` chart geometries, an ``(R + 1) -> G`` encoder,
    and a ``G -> H`` decoder per chart.  Incoming cross-event messages enter
    the same ``G`` packet before decoding.
    """

    def __init__(
        self,
        *,
        base_kind: str,
        chart_centers: Tensor,
        chart_bases: Tensor,
        chart_distance_scales: Tensor,
        local_generator_encoders: Tensor,
        chart_hidden_decoders: Tensor,
        chart_temperature: float = 1.0,
        affine_weight: Tensor | None = None,
        affine_bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if base_kind not in {"identity", "affine"}:
            raise ValueError("base_kind must be identity or affine")
        if chart_centers.ndim != 2 or not all(chart_centers.shape):
            raise ValueError("chart_centers must have shape [charts, hidden]")
        charts, hidden = chart_centers.shape
        if (
            chart_bases.ndim != 3
            or chart_bases.shape[:2] != (charts, hidden)
            or not chart_bases.shape[2]
        ):
            raise ValueError(
                "chart_bases must have shape [charts, hidden, rank]"
            )
        chart_rank = chart_bases.shape[2]
        if chart_rank > hidden:
            raise ValueError("chart rank cannot exceed hidden width")
        if chart_distance_scales.shape != (charts,):
            raise ValueError("chart distance scales must have shape [charts]")
        if (
            local_generator_encoders.ndim != 3
            or local_generator_encoders.shape[:2]
            != (charts, chart_rank + 1)
            or not local_generator_encoders.shape[2]
        ):
            raise ValueError("local generator encoder shape is invalid")
        generator_rank = local_generator_encoders.shape[2]
        if chart_hidden_decoders.shape != (
            charts,
            generator_rank,
            hidden,
        ):
            raise ValueError("chart hidden decoder shape is invalid")
        if (
            not isinstance(chart_temperature, (int, float))
            or isinstance(chart_temperature, bool)
            or not math.isfinite(float(chart_temperature))
            or float(chart_temperature) <= 0
        ):
            raise ValueError("chart_temperature must be finite and positive")
        tensors = (
            chart_centers,
            chart_bases,
            chart_distance_scales,
            local_generator_encoders,
            chart_hidden_decoders,
        )
        if any(not value.is_floating_point() for value in tensors):
            raise ValueError("hidden chart tensors must be floating point")
        if any(
            value.dtype != chart_centers.dtype
            or value.device != chart_centers.device
            or not bool(torch.isfinite(value).all())
            for value in tensors
        ):
            raise ValueError(
                "hidden chart tensors must share finite dtype and device"
            )
        if bool((chart_distance_scales <= 0).any()):
            raise ValueError("chart distance scales must be positive")

        if base_kind == "identity":
            if affine_weight is not None or affine_bias is not None:
                raise ValueError("identity base cannot store affine tensors")
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        else:
            if (
                not isinstance(affine_weight, Tensor)
                or not isinstance(affine_bias, Tensor)
                or affine_weight.shape != (hidden, hidden)
                or affine_bias.shape != (hidden,)
                or affine_weight.dtype != chart_centers.dtype
                or affine_bias.dtype != chart_centers.dtype
                or affine_weight.device != chart_centers.device
                or affine_bias.device != chart_centers.device
                or not bool(torch.isfinite(affine_weight).all())
                or not bool(torch.isfinite(affine_bias).all())
            ):
                raise ValueError("affine base tensors are invalid")
            self.affine_weight = nn.Parameter(
                affine_weight.detach().clone().contiguous(),
                requires_grad=False,
            )
            self.affine_bias = nn.Parameter(
                affine_bias.detach().clone().contiguous(),
                requires_grad=False,
            )

        self.base_kind = base_kind
        self.hidden_width = hidden
        self.chart_count = charts
        self.chart_rank = chart_rank
        self.generator_rank = generator_rank
        self.incoming_generator_rank = generator_rank
        self.emitted_generator_rank = generator_rank
        self.chart_temperature = float(chart_temperature)
        self.register_buffer(
            "chart_centers",
            chart_centers.detach().clone().contiguous(),
        )
        self.register_buffer(
            "chart_bases",
            chart_bases.detach().clone().contiguous(),
        )
        self.register_buffer(
            "chart_distance_scales",
            chart_distance_scales.detach().clone().contiguous(),
        )
        self.local_generator_encoders = nn.Parameter(
            local_generator_encoders.detach().clone().contiguous(),
            requires_grad=False,
        )
        self.chart_hidden_decoders = nn.Parameter(
            chart_hidden_decoders.detach().clone().contiguous(),
            requires_grad=False,
        )
        self.eval().requires_grad_(False)

    @classmethod
    def identity_zero_local_control(
        cls,
        *,
        hidden_width: int,
        chart_count: int,
        chart_rank: int,
        generator_rank: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> FactorizedHiddenChartGenerator:
        """Build an exact identity with zero local and incoming correction."""

        for label, value in (
            ("hidden_width", hidden_width),
            ("chart_count", chart_count),
            ("chart_rank", chart_rank),
            ("generator_rank", generator_rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if chart_rank > hidden_width:
            raise ValueError("chart_rank cannot exceed hidden_width")
        resolved_device = torch.device(device)
        centers = torch.zeros(
            chart_count,
            hidden_width,
            dtype=dtype,
            device=resolved_device,
        )
        basis = torch.eye(
            hidden_width,
            chart_rank,
            dtype=dtype,
            device=resolved_device,
        ).expand(chart_count, -1, -1).clone()
        return cls(
            base_kind="identity",
            chart_centers=centers,
            chart_bases=basis,
            chart_distance_scales=torch.ones(
                chart_count,
                dtype=dtype,
                device=resolved_device,
            ),
            local_generator_encoders=torch.zeros(
                chart_count,
                chart_rank + 1,
                generator_rank,
                dtype=dtype,
                device=resolved_device,
            ),
            chart_hidden_decoders=torch.zeros(
                chart_count,
                generator_rank,
                hidden_width,
                dtype=dtype,
                device=resolved_device,
            ),
        )

    @property
    def stored_scalar_count(self) -> int:
        return _module_scalar_count(self)

    @property
    def dense_base_scalar_count(self) -> int:
        return 0 if self.base_kind == "identity" else self.hidden_width**2 + self.hidden_width

    def execute_with_incoming(
        self,
        features: Tensor,
        incoming_generator_latents: Tensor,
    ) -> FactorizedHiddenChartExecution:
        if (
            not isinstance(features, Tensor)
            or features.ndim != 3
            or features.shape[-1] != self.hidden_width
            or features.dtype != self.chart_centers.dtype
            or features.device != self.chart_centers.device
        ):
            raise ValueError("hidden chart features are invalid")
        expected_incoming = (
            *features.shape[:2],
            self.chart_count,
            self.incoming_generator_rank,
        )
        if (
            not isinstance(incoming_generator_latents, Tensor)
            or incoming_generator_latents.shape != expected_incoming
            or incoming_generator_latents.dtype != features.dtype
            or incoming_generator_latents.device != features.device
        ):
            raise ValueError("hidden chart incoming packet is invalid")
        displacement = features.unsqueeze(-2) - self.chart_centers
        distance = displacement.square().sum(dim=-1)
        membership_logits = -0.5 * distance / (
            self.chart_distance_scales.square() * self.chart_temperature
        )
        memberships = membership_logits.softmax(dim=-1)
        coordinates = torch.einsum(
            "...ch,chr->...cr",
            displacement,
            self.chart_bases,
        )
        augmented = torch.cat(
            (torch.ones_like(coordinates[..., :1]), coordinates),
            dim=-1,
        )
        local = torch.einsum(
            "...cr,crg->...cg",
            augmented,
            self.local_generator_encoders,
        )
        emitted = memberships.unsqueeze(-1) * (
            local + incoming_generator_latents
        )
        correction = torch.einsum(
            "...cg,cgh->...h",
            emitted,
            self.chart_hidden_decoders,
        )
        if self.base_kind == "identity":
            base = features
        else:
            assert self.affine_weight is not None
            assert self.affine_bias is not None
            base = torch.nn.functional.linear(
                features,
                self.affine_weight,
                self.affine_bias,
            )
        return FactorizedHiddenChartExecution(
            output=base + correction,
            memberships=memberships,
            coordinates=coordinates,
            emitted_generator_latents=emitted,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "factorized_hidden_chart_generator",
            "base_kind": self.base_kind,
            "hidden_width": self.hidden_width,
            "chart_count": self.chart_count,
            "chart_rank": self.chart_rank,
            "incoming_generator_rank": self.incoming_generator_rank,
            "emitted_generator_rank": self.emitted_generator_rank,
            "dense_identity_matrix_stored": False,
            "dense_base_scalar_count": self.dense_base_scalar_count,
            "stored_scalar_count": self.stored_scalar_count,
        }


class ExactTokenLocalChartGenerator(nn.Module):
    """Exact cloned token-local operator for the 39-event control.

    The operator is executor-owned and has no live teacher reference.  It
    intentionally retains cloned checkpoint coefficients and refuses transport
    perturbations, so it is an exactness control rather than a compression arm.
    """

    def __init__(
        self,
        *,
        operator: nn.Module,
        hidden_width: int,
        chart_count: int,
        generator_rank: int,
    ) -> None:
        super().__init__()
        if not isinstance(operator, nn.Module):
            raise TypeError("operator must be an nn.Module")
        for label, value in (
            ("hidden_width", hidden_width),
            ("chart_count", chart_count),
            ("generator_rank", generator_rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        self.operator = operator
        self.hidden_width = hidden_width
        self.chart_count = chart_count
        self.generator_rank = generator_rank
        self.incoming_generator_rank = generator_rank
        self.emitted_generator_rank = generator_rank
        self.eval().requires_grad_(False)

    @property
    def stored_scalar_count(self) -> int:
        return _module_scalar_count(self)

    def execute_with_incoming(
        self,
        features: Tensor,
        incoming_generator_latents: Tensor,
    ) -> FactorizedHiddenChartExecution:
        if (
            not isinstance(features, Tensor)
            or features.ndim != 3
            or features.shape[-1] != self.hidden_width
        ):
            raise ValueError("exact token-local features are invalid")
        expected = (
            *features.shape[:2],
            self.chart_count,
            self.incoming_generator_rank,
        )
        if incoming_generator_latents.shape != expected:
            raise ValueError("exact token-local incoming packet is invalid")
        if bool(torch.count_nonzero(incoming_generator_latents)):
            raise ValueError(
                "exact token-local control rejects nonzero transport"
            )
        output = self.operator(features)
        if not isinstance(output, Tensor) or output.shape != features.shape:
            raise ValueError("exact token-local operator must preserve shape")
        emitted = torch.zeros(
            expected,
            dtype=features.dtype,
            device=features.device,
        )
        return FactorizedHiddenChartExecution(
            output=output,
            memberships=torch.zeros(
                *features.shape[:2],
                self.chart_count,
                dtype=features.dtype,
                device=features.device,
            ),
            coordinates=torch.zeros(
                *features.shape[:2],
                self.chart_count,
                1,
                dtype=features.dtype,
                device=features.device,
            ),
            emitted_generator_latents=emitted,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "exact_token_local_chart_generator",
            "hidden_width": self.hidden_width,
            "chart_count": self.chart_count,
            "incoming_generator_rank": self.incoming_generator_rank,
            "emitted_generator_rank": self.emitted_generator_rank,
            "stored_scalar_count": self.stored_scalar_count,
            "live_teacher_reference_present": False,
            "source_checkpoint_coefficients_retained": True,
            "nonzero_transport_accepted": False,
            "compression_attempted": False,
        }


class JointGraphGeneratorMutation(nn.Module):
    """One sequence-aware attention or token-local MLP chart mutation.

    The preferred attention path supplies ``compiled_operator`` implementing
    ``execute_with_incoming(state, sequence, incoming)`` and returning
    ``delta`` plus ``emitted_generator_latents``.  This lets a complete
    attention generator own normalization, Q/K/V coordinates, causal routing,
    RoPE, grouped-query attention, output projection, and post-normalization
    exactly once.

    The pipeline path remains for MLP charts and controlled ablations.  Its
    ``chart_generator`` implements ``execute_with_incoming(value, incoming)``
    and returns ``output`` plus ``emitted_generator_latents``.
    """

    def __init__(
        self,
        *,
        event_spec: JointGraphEventSpec,
        input_norm: nn.Module | None = None,
        chart_generator: nn.Module | None = None,
        output_norm: nn.Module | None = None,
        sequence_operator: nn.Module | None = None,
        compiled_operator: nn.Module | None = None,
        source_checkpoint_coefficients_retained: bool = False,
    ) -> None:
        super().__init__()
        if event_spec.event_kind not in {
            "attention_mutation",
            "mlp_mutation",
        }:
            raise ValueError("generator mutation requires a body event")
        direct = compiled_operator is not None
        if direct:
            if not isinstance(compiled_operator, nn.Module) or not callable(
                getattr(compiled_operator, "execute_with_incoming", None)
            ):
                raise TypeError(
                    "compiled_operator must expose execute_with_incoming()"
                )
            if any(
                value is not None
                for value in (
                    input_norm,
                    chart_generator,
                    output_norm,
                    sequence_operator,
                )
            ):
                raise ValueError(
                    "direct compiled mutation cannot also own a chart pipeline"
                )
            if event_spec.event_kind == "attention_mutation":
                manifest_method = getattr(
                    compiled_operator,
                    "architecture_manifest",
                    None,
                )
                manifest = (
                    manifest_method() if callable(manifest_method) else None
                )
                expected_attention_kind = (
                    "global_causal"
                    if event_spec.attention_kind == "global"
                    else "sliding_causal"
                )
                attention_contract = (
                    manifest.get("attention_contract")
                    if isinstance(manifest, Mapping)
                    else None
                )
                if (
                    not isinstance(manifest, Mapping)
                    or manifest.get("kind")
                    != "gemma3_attention_generator"
                    or manifest.get("invocation_id") != event_spec.event_id
                    or manifest.get("source_independent_runtime") is not True
                    or manifest.get("source_fallback") is not False
                    or not isinstance(attention_contract, Mapping)
                    or attention_contract.get("kind")
                    != expected_attention_kind
                ):
                    raise ValueError(
                        "compiled attention does not authenticate its event contract"
                    )
        else:
            if not all(
                isinstance(module, nn.Module)
                for module in (input_norm, chart_generator, output_norm)
            ):
                raise TypeError("mutation components must be nn.Modules")
            if not callable(
                getattr(chart_generator, "execute_with_incoming", None)
            ):
                raise TypeError(
                    "chart_generator must expose execute_with_incoming()"
                )
            if event_spec.event_kind == "attention_mutation":
                if not isinstance(sequence_operator, nn.Module):
                    raise TypeError(
                        "attention mutation requires a sequence_operator"
                    )
            elif sequence_operator is not None:
                raise ValueError("MLP mutations cannot own a sequence operator")
        if type(source_checkpoint_coefficients_retained) is not bool:
            raise TypeError(
                "source_checkpoint_coefficients_retained must be bool"
            )

        endpoint = compiled_operator if direct else chart_generator
        assert isinstance(endpoint, nn.Module)
        for label in ("hidden_width", "chart_count"):
            value = getattr(endpoint, label, None)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"mutation endpoint omitted positive integer {label}"
                )
        fallback_rank = getattr(endpoint, "generator_rank", None)
        for label in (
            "incoming_generator_rank",
            "emitted_generator_rank",
        ):
            value = getattr(endpoint, label, fallback_rank)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"mutation endpoint omitted positive integer {label}"
                )

        self.event_spec = event_spec
        self.input_norm = input_norm
        self.sequence_operator = sequence_operator
        self.chart_generator = chart_generator
        self.output_norm = output_norm
        self.compiled_operator = compiled_operator
        self.source_checkpoint_coefficients_retained = (
            source_checkpoint_coefficients_retained
        )
        self.eval().requires_grad_(False)

    @property
    def hidden_width(self) -> int:
        return int(self._endpoint.hidden_width)

    @property
    def chart_count(self) -> int:
        return int(self._endpoint.chart_count)

    @property
    def incoming_chart_count(self) -> int:
        return int(
            getattr(self._endpoint, "incoming_chart_count", self.chart_count)
        )

    @property
    def emitted_chart_count(self) -> int:
        return int(
            getattr(self._endpoint, "emitted_chart_count", self.chart_count)
        )

    @property
    def incoming_generator_rank(self) -> int:
        value = getattr(self._endpoint, "incoming_generator_rank", None)
        if value is None:
            value = getattr(self._endpoint, "generator_rank")
        return int(value)

    @property
    def emitted_generator_rank(self) -> int:
        value = getattr(self._endpoint, "emitted_generator_rank", None)
        if value is None:
            value = getattr(self._endpoint, "generator_rank")
        return int(value)

    @property
    def generator_rank(self) -> int:
        if self.incoming_generator_rank != self.emitted_generator_rank:
            raise RuntimeError(
                "asymmetric mutation endpoint has no single generator rank"
            )
        return self.incoming_generator_rank

    @property
    def _endpoint(self) -> nn.Module:
        endpoint = (
            self.compiled_operator
            if self.compiled_operator is not None
            else self.chart_generator
        )
        if not isinstance(endpoint, nn.Module):
            raise RuntimeError("validated mutation endpoint disappeared")
        return endpoint

    @property
    def stored_scalar_count(self) -> int:
        return _module_scalar_count(self)

    def execute_with_incoming(
        self,
        state: Tensor,
        sequence: SequenceContext,
        incoming_generator_latents: Tensor,
        *,
        chart_input_perturbation: Tensor | None = None,
    ) -> JointGraphMutationExecution:
        if (
            not isinstance(state, Tensor)
            or state.ndim != 3
            or state.shape[-1] != self.hidden_width
        ):
            raise ValueError("mutation state width or rank drifted")
        if state.shape[:2] != (
            sequence.batch_size,
            sequence.query_length,
        ):
            raise ValueError("mutation state does not match sequence context")
        expected_incoming_shape = (
            *state.shape[:2],
            self.incoming_chart_count,
            self.incoming_generator_rank,
        )
        if incoming_generator_latents.shape != expected_incoming_shape:
            raise ValueError("incoming generator latent shape drifted")
        if (
            incoming_generator_latents.dtype != state.dtype
            or incoming_generator_latents.device != state.device
        ):
            raise ValueError("incoming generator latent dtype/device drifted")
        if chart_input_perturbation is not None and (
            not isinstance(chart_input_perturbation, Tensor)
            or chart_input_perturbation.shape != state.shape
            or chart_input_perturbation.dtype != state.dtype
            or chart_input_perturbation.device != state.device
            or not bool(torch.isfinite(chart_input_perturbation).all())
        ):
            raise ValueError("chart input perturbation must match mutation state")

        if self.compiled_operator is not None:
            if chart_input_perturbation is None:
                compiled_execution = self.compiled_operator.execute_with_incoming(
                    state,
                    sequence,
                    incoming_generator_latents,
                )
            else:
                compiled_execution = self.compiled_operator.execute_with_incoming(
                    state,
                    sequence,
                    incoming_generator_latents,
                    normalized_input_perturbation=chart_input_perturbation,
                )
            chart_input = getattr(
                compiled_execution,
                "normalized_input",
                state,
            )
            delta = getattr(compiled_execution, "delta", None)
            emitted = getattr(
                compiled_execution,
                "emitted_generator_latents",
                None,
            )
            operator_receipt = getattr(
                compiled_execution,
                "receipt",
                None,
            )
        else:
            assert self.input_norm is not None
            assert self.chart_generator is not None
            assert self.output_norm is not None
            normalized = self.input_norm(state)
            if normalized.shape != state.shape:
                raise ValueError("input normalization must preserve state shape")
            if chart_input_perturbation is not None:
                normalized = normalized + chart_input_perturbation.masked_fill(
                    ~sequence.query_valid_mask.unsqueeze(-1),
                    0,
                )
            chart_input = normalized
            if self.event_spec.event_kind == "attention_mutation":
                assert self.sequence_operator is not None
                features = self.sequence_operator(normalized, sequence)
                if (
                    not isinstance(features, Tensor)
                    or features.shape != state.shape
                ):
                    raise ValueError(
                        "attention sequence operator must preserve state shape"
                    )
            else:
                features = normalized
            chart_execution = self.chart_generator.execute_with_incoming(
                features,
                incoming_generator_latents,
            )
            raw_delta = getattr(chart_execution, "output", None)
            emitted = getattr(
                chart_execution,
                "emitted_generator_latents",
                None,
            )
            operator_receipt = getattr(chart_execution, "receipt", None)
            if (
                not isinstance(raw_delta, Tensor)
                or raw_delta.shape != state.shape
            ):
                raise ValueError(
                    "chart generator output must preserve state shape"
                )
            delta = self.output_norm(raw_delta)
        if (
            not isinstance(chart_input, Tensor)
            or chart_input.shape != state.shape
            or chart_input.dtype != state.dtype
            or chart_input.device != state.device
        ):
            raise ValueError("mutation chart input must preserve state")
        if not isinstance(delta, Tensor) or delta.shape != state.shape:
            raise ValueError("mutation delta must preserve state shape")
        expected_emitted_shape = (
            *state.shape[:2],
            self.emitted_chart_count,
            self.emitted_generator_rank,
        )
        if (
            not isinstance(emitted, Tensor)
            or emitted.shape != expected_emitted_shape
        ):
            raise ValueError("mutation emitted generator packet drifted")
        valid = sequence.query_valid_mask.unsqueeze(-1)
        source_projection_calls = getattr(
            operator_receipt,
            "source_projection_call_count",
            None,
        )
        source_fallback_calls = getattr(
            operator_receipt,
            "source_fallback_call_count",
            None,
        )
        if self.compiled_operator is not None:
            expected_attention_kind = (
                "global_causal"
                if self.event_spec.attention_kind == "global"
                else "sliding_causal"
            )
            if (
                operator_receipt is None
                or getattr(operator_receipt, "invocation_id", None)
                != self.event_spec.event_id
                or getattr(operator_receipt, "attention_kind", None)
                != expected_attention_kind
                or getattr(operator_receipt, "input_coordinate_rank", None)
                != self.incoming_generator_rank
                or getattr(operator_receipt, "output_coordinate_rank", None)
                != self.emitted_generator_rank
                or source_projection_calls != 0
                or source_fallback_calls != 0
            ):
                raise RuntimeError(
                    "compiled attention runtime receipt failed its tripwire"
                )
        runtime_receipt = JointGraphMutationRuntimeReceipt(
            event_id=self.event_spec.event_id,
            event_kind=self.event_spec.event_kind,
            endpoint_kind=type(self._endpoint).__qualname__,
            incoming_generator_rank=self.incoming_generator_rank,
            emitted_generator_rank=self.emitted_generator_rank,
            operator_receipt_present=operator_receipt is not None,
            generator_kind=getattr(
                operator_receipt,
                "generator_kind",
                getattr(self._endpoint, "generator_kind", None),
            ),
            attention_kind=getattr(
                operator_receipt,
                "attention_kind",
                None,
            ),
            logical_causal_key_pairs=getattr(
                operator_receipt,
                "logical_causal_key_pairs",
                None,
            ),
            source_projection_call_count=source_projection_calls,
            source_fallback_call_count=source_fallback_calls,
        )
        return JointGraphMutationExecution(
            chart_input=chart_input,
            delta=delta.masked_fill(~valid, 0),
            emitted_generator_latents=emitted,
            runtime_receipt=runtime_receipt,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "event": self.event_spec.metadata(),
            "hidden_width": self.hidden_width,
            "incoming_chart_count": self.incoming_chart_count,
            "emitted_chart_count": self.emitted_chart_count,
            "incoming_generator_rank": self.incoming_generator_rank,
            "emitted_generator_rank": self.emitted_generator_rank,
            "direct_compiled_operator_present": (
                self.compiled_operator is not None
            ),
            "sequence_operator_present": (
                self.compiled_operator is not None
                or self.sequence_operator is not None
            ),
            "endpoint_kind": type(self._endpoint).__qualname__,
            "source_checkpoint_coefficients_retained": (
                self.source_checkpoint_coefficients_retained
            ),
            "stored_scalar_count": self.stored_scalar_count,
        }


@dataclass(frozen=True, slots=True)
class JointGraphTransportSpec:
    """One transport between fitted generator invocations."""

    transport_id: str
    source_event_id: str
    target_event_id: str
    source_chart_index: int
    target_chart_index: int
    token_relation: str = "same_token"

    def __post_init__(self) -> None:
        _require_id(self.transport_id, label="transport_id")
        _require_id(self.source_event_id, label="source_event_id")
        _require_id(self.target_event_id, label="target_event_id")
        if self.source_event_id == self.target_event_id:
            raise ValueError("transport self edges are forbidden")
        if (
            type(self.source_chart_index) is not int
            or self.source_chart_index < 0
            or type(self.target_chart_index) is not int
            or self.target_chart_index < 0
        ):
            raise ValueError("transport chart indices must be nonnegative")
        if self.token_relation != "same_token":
            raise ValueError(
                "joint runtime v1 supports same-token transports only"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "transport_id": self.transport_id,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "source_chart_index": self.source_chart_index,
            "target_chart_index": self.target_chart_index,
            "token_relation": self.token_relation,
        }


class JointGraphCrossBlockTransportEdge(nn.Module):
    """A dense latent map on one sparse, strictly-forward chart edge."""

    def __init__(
        self,
        *,
        spec: JointGraphTransportSpec,
        transport_weight: Tensor,
        source_generator_rank: int | None = None,
        target_generator_rank: int | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(transport_weight, Tensor)
            or transport_weight.ndim != 2
            or not transport_weight.is_floating_point()
            or not bool(torch.isfinite(transport_weight).all())
            or not transport_weight.shape[0]
            or not transport_weight.shape[1]
        ):
            raise ValueError(
                "transport_weight must be a finite floating matrix"
            )
        inferred_source, inferred_target = transport_weight.shape
        if source_generator_rank is not None and (
            type(source_generator_rank) is not int
            or source_generator_rank != inferred_source
        ):
            raise ValueError("source_generator_rank disagrees with weight")
        if target_generator_rank is not None and (
            type(target_generator_rank) is not int
            or target_generator_rank != inferred_target
        ):
            raise ValueError("target_generator_rank disagrees with weight")
        self.spec = spec
        self.source_generator_rank = inferred_source
        self.target_generator_rank = inferred_target
        self.transport_weight = nn.Parameter(
            transport_weight.detach().clone().contiguous(),
            requires_grad=False,
        )
        self.eval().requires_grad_(False)

    @property
    def stored_scalar_count(self) -> int:
        return self.transport_weight.numel()

    def forward(self, source_generator_latent: Tensor) -> Tensor:
        if source_generator_latent.shape[-1] != self.source_generator_rank:
            raise ValueError("source generator rank drifted")
        return source_generator_latent @ self.transport_weight

    def metadata(self) -> dict[str, object]:
        return {
            **self.spec.metadata(),
            "source_generator_rank": self.source_generator_rank,
            "target_generator_rank": self.target_generator_rank,
            "stored_scalar_count": self.stored_scalar_count,
            "strictly_forward": True,
        }


class _JointGraphMessageSession:
    """Per-forward transport state with deterministic packet release."""

    def __init__(
        self,
        *,
        mutations: Mapping[str, JointGraphGeneratorMutation],
        edges: Sequence[JointGraphCrossBlockTransportEdge],
        valid_positions: Tensor,
        event_indexes: Mapping[str, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._mutations = dict(mutations)
        self._valid_positions = valid_positions
        self._event_indexes = dict(event_indexes)
        self._dtype = dtype
        self._device = device
        self._outgoing: dict[
            str, list[JointGraphCrossBlockTransportEdge]
        ] = defaultdict(list)
        self._incoming: dict[
            str, list[JointGraphCrossBlockTransportEdge]
        ] = defaultdict(list)
        for edge in edges:
            self._outgoing[edge.spec.source_event_id].append(edge)
            self._incoming[edge.spec.target_event_id].append(edge)
        self._last_consumer = {
            source: max(
                self._event_indexes[edge.spec.target_event_id]
                for edge in outgoing
            )
            for source, outgoing in self._outgoing.items()
        }
        self._messages: dict[str, Tensor] = {}
        self._published: set[str] = set()
        self._consumed: set[str] = set()
        self.peak_live_packets = 0
        self.peak_live_packet_bytes = 0
        self.closed = False

    def incoming_for(
        self,
        target: JointGraphGeneratorMutation,
    ) -> Tensor:
        event_id = target.event_spec.event_id
        if self.closed or event_id in self._consumed:
            raise RuntimeError("mutation consumed transport state twice")
        self._consumed.add(event_id)
        incoming = torch.zeros(
            (
                *self._valid_positions.shape,
                target.incoming_chart_count,
                target.incoming_generator_rank,
            ),
            dtype=self._dtype,
            device=self._device,
        )
        for edge in self._incoming.get(event_id, ()):
            try:
                source = self._messages[edge.spec.source_event_id]
            except KeyError as error:
                raise RuntimeError(
                    "transport target executed before its source"
                ) from error
            incoming[..., edge.spec.target_chart_index, :] += edge(
                source[..., edge.spec.source_chart_index, :]
            )
        target_index = self._event_indexes[event_id]
        for source, last_consumer in self._last_consumer.items():
            if last_consumer == target_index:
                self._messages.pop(source, None)
        return incoming

    def publish(
        self,
        mutation: JointGraphGeneratorMutation,
        emitted: Tensor,
    ) -> None:
        event_id = mutation.event_spec.event_id
        if self.closed or event_id in self._published:
            raise RuntimeError("mutation published transport state twice")
        self._published.add(event_id)
        if event_id not in self._outgoing:
            return
        masked = emitted.masked_fill(
            ~self._valid_positions[..., None, None],
            0,
        )
        self._messages[event_id] = masked
        self.peak_live_packets = max(
            self.peak_live_packets,
            len(self._messages),
        )
        self.peak_live_packet_bytes = max(
            self.peak_live_packet_bytes,
            sum(
                value.numel() * value.element_size()
                for value in self._messages.values()
            ),
        )

    def finish(self, expected_event_ids: Sequence[str]) -> None:
        if self.closed:
            raise RuntimeError("transport message session already closed")
        self.closed = True
        if self._published != set(expected_event_ids):
            raise RuntimeError("not every mutation published exactly once")
        if self._consumed != set(expected_event_ids):
            raise RuntimeError("not every mutation consumed exactly once")
        if self._messages:
            raise RuntimeError("transport packets remained live after execution")

    def abort(self) -> None:
        self._messages.clear()
        self.closed = True


JointGraphEventObserver = Callable[..., object]


def _expected_observation_site_ids() -> tuple[str, ...]:
    return (
        "lexical.ingress.output",
        *(
            site_id
            for mutation_index in range(1, PINNED_GEMMA3_MUTATION_COUNT + 1)
            for site_id in (
                f"mutation.{mutation_index}.input",
                f"mutation.{mutation_index}.output",
            )
        ),
    )


class JointGraphEventGradientTape:
    """One-use exact suffix-gradient tape for the 73 chart sites.

    The public observer only receives detached clones.  This private-value tape
    retains the corresponding live tensors so the model-facing edge worker can
    obtain exact causal-language-model score gradients after constructing its
    scalar loss.  Calling :meth:`score_gradients` never mutates model state.
    """

    __slots__ = ("_consumed", "_site_ids", "_tensors")

    def __init__(
        self,
        observations: Sequence[tuple[str, Tensor]],
    ) -> None:
        materialized = tuple(observations)
        expected = _expected_observation_site_ids()
        if tuple(site_id for site_id, _ in materialized) != expected:
            raise ValueError("event gradient tape site order drifted")
        tensors = tuple(value for _, value in materialized)
        if any(
            not isinstance(value, Tensor)
            or value.ndim != 3
            or not value.requires_grad
            for value in tensors
        ):
            raise ValueError("event gradient tape requires live rank-three tensors")
        if len({id(value) for value in tensors}) != len(tensors):
            raise ValueError("event gradient tape tensors must be distinct")
        self._site_ids = expected
        self._tensors = tensors
        self._consumed = False

    @property
    def site_ids(self) -> tuple[str, ...]:
        return self._site_ids

    @property
    def consumed(self) -> bool:
        return self._consumed

    def score_gradients(self, objective: Tensor) -> dict[str, Tensor]:
        if self._consumed:
            raise RuntimeError("event gradient tape was already consumed")
        if (
            not isinstance(objective, Tensor)
            or objective.numel() != 1
            or not objective.is_floating_point()
            or not objective.requires_grad
        ):
            raise ValueError("score objective must be one differentiable scalar")
        gradients = torch.autograd.grad(
            objective,
            self._tensors,
            allow_unused=False,
            retain_graph=False,
            create_graph=False,
        )
        self._consumed = True
        self._tensors = ()
        return {
            site_id: gradient.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .clone()
            for site_id, gradient in zip(self._site_ids, gradients, strict=True)
        }


@dataclass(frozen=True, slots=True)
class JointGraphWholeModelExecution:
    """Inspectable result of one complete joint graph traversal."""

    output: Tensor
    normalized_output: Tensor
    logits: Tensor
    event_ids: tuple[str, ...]
    mutation_receipts: tuple[ResidualMutationReceipt, ...]
    operator_receipts: tuple[JointGraphMutationRuntimeReceipt, ...]
    mutation_chart_input_sha256s: tuple[str, ...]
    mutation_delta_sha256s: tuple[str, ...]
    carrier_version: int
    transport_edge_count: int
    peak_live_transport_packets: int
    peak_live_transport_packet_bytes: int

    def __post_init__(self) -> None:
        if len(self.event_ids) != PINNED_GEMMA3_EVENT_COUNT:
            raise ValueError("joint graph did not execute exactly 39 events")
        if self.carrier_version != PINNED_GEMMA3_MUTATION_COUNT:
            raise ValueError("joint graph residual carrier did not close")
        if len(self.mutation_receipts) != self.carrier_version:
            raise ValueError("joint graph mutation receipts are incomplete")
        if len(self.operator_receipts) != self.carrier_version:
            raise ValueError("joint graph operator receipts are incomplete")
        if tuple(item.event_id for item in self.operator_receipts) != tuple(
            item.mutation_id for item in self.mutation_receipts
        ):
            raise ValueError("operator and carrier receipt order disagrees")
        for label, values in (
            ("chart input", self.mutation_chart_input_sha256s),
            ("delta", self.mutation_delta_sha256s),
        ):
            if (
                len(values) != self.carrier_version
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                    for value in values
                )
            ):
                raise ValueError(f"mutation {label} SHA-256 inventory is invalid")


@dataclass(frozen=True, slots=True)
class JointGraphObservedExecution:
    """Exact execution plus an optional one-use score-gradient tape."""

    execution: JointGraphWholeModelExecution
    gradient_tape: JointGraphEventGradientTape | None
    shadow_source_site_id: str | None = None
    shadow_target_site_id: str | None = None
    shadow_target_input: Tensor | None = None
    shadow_target_flow: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution, JointGraphWholeModelExecution):
            raise TypeError("observed execution result is invalid")
        if self.gradient_tape is not None and not isinstance(
            self.gradient_tape,
            JointGraphEventGradientTape,
        ):
            raise TypeError("observed execution gradient tape is invalid")
        shadow_values = (
            self.shadow_source_site_id,
            self.shadow_target_site_id,
            self.shadow_target_input,
            self.shadow_target_flow,
        )
        if any(value is not None for value in shadow_values):
            if (
                not isinstance(self.shadow_source_site_id, str)
                or not isinstance(self.shadow_target_site_id, str)
                or not isinstance(self.shadow_target_input, Tensor)
                or self.shadow_target_input.ndim != 3
                or not isinstance(self.shadow_target_flow, Tensor)
                or self.shadow_target_flow.ndim != 3
                or not self.shadow_target_flow.is_floating_point()
                or self.shadow_target_input.shape
                != self.shadow_target_flow.shape
            ):
                raise TypeError("shadow intervention receipt is incomplete")


class Gemma3JointCrossBlockChartExecutor(nn.Module):
    """Standalone 39-event executor for a fitted joint Gemma candidate."""

    def __init__(
        self,
        *,
        lexical: nn.Module,
        mutations: Sequence[JointGraphGeneratorMutation],
        transports: Sequence[JointGraphCrossBlockTransportEdge],
        terminal_norm: nn.Module,
        final_logit_softcap: float | None = None,
        arm_id: str = JOINT_CROSS_BLOCK_CHART_ARM_ID,
        carrier_factory: ResidualCarrierFactory | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(lexical, nn.Module) or not all(
            callable(getattr(lexical, method, None))
            for method in ("ingress", "egress")
        ):
            raise TypeError("lexical must expose tied ingress() and egress()")
        if not isinstance(terminal_norm, nn.Module):
            raise TypeError("terminal_norm must be an nn.Module")
        _require_id(arm_id, label="arm_id")
        if arm_id not in {
            JOINT_CROSS_BLOCK_CHART_ARM_ID,
            JOINT_EXACT_CONTROL_ARM_ID,
        }:
            raise ValueError("joint executor arm_id is unsupported")
        if final_logit_softcap is not None and (
            not isinstance(final_logit_softcap, (int, float))
            or isinstance(final_logit_softcap, bool)
            or not math.isfinite(float(final_logit_softcap))
            or float(final_logit_softcap) <= 0
        ):
            raise ValueError("final_logit_softcap must be positive when set")
        if carrier_factory is not None and not isinstance(
            carrier_factory,
            ResidualCarrierFactory,
        ):
            raise TypeError("carrier_factory does not implement the protocol")

        schedule = build_joint_cross_block_event_schedule()
        mutation_specs = schedule[1:-2]
        declared_mutations = tuple(mutations)
        if (
            len(declared_mutations) != PINNED_GEMMA3_MUTATION_COUNT
            or any(
                not isinstance(item, JointGraphGeneratorMutation)
                for item in declared_mutations
            )
            or tuple(item.event_spec for item in declared_mutations)
            != mutation_specs
        ):
            raise ValueError(
                "mutations must exactly cover the ordered 36-event body"
            )
        hidden_widths = {item.hidden_width for item in declared_mutations}
        if len(hidden_widths) != 1:
            raise ValueError("joint mutations must share one residual width")

        declared_edges = tuple(transports)
        if any(
            not isinstance(edge, JointGraphCrossBlockTransportEdge)
            for edge in declared_edges
        ):
            raise TypeError("transports must be joint graph edges")
        event_by_id = {event.event_id: event for event in mutation_specs}
        if len({edge.spec.transport_id for edge in declared_edges}) != len(
            declared_edges
        ):
            raise ValueError("transport ids must be unique")
        coordinates: set[tuple[str, int, str, int]] = set()
        cross_block_count = 0
        cross_family_count = 0
        cross_block_cross_family_count = 0
        for edge in declared_edges:
            source = event_by_id.get(edge.spec.source_event_id)
            target = event_by_id.get(edge.spec.target_event_id)
            if source is None or target is None:
                raise ValueError("transport references a non-mutation event")
            if source.event_index >= target.event_index:
                raise ValueError("transport edges must be strictly forward")
            source_mutation = declared_mutations[source.event_index - 1]
            target_mutation = declared_mutations[target.event_index - 1]
            if (
                edge.spec.source_chart_index
                >= source_mutation.emitted_chart_count
                or edge.spec.target_chart_index
                >= target_mutation.incoming_chart_count
                or edge.source_generator_rank
                != source_mutation.emitted_generator_rank
                or edge.target_generator_rank
                != target_mutation.incoming_generator_rank
            ):
                raise ValueError("transport does not match endpoint charts")
            coordinate = (
                edge.spec.source_event_id,
                edge.spec.source_chart_index,
                edge.spec.target_event_id,
                edge.spec.target_chart_index,
            )
            if coordinate in coordinates:
                raise ValueError("duplicate chart transport is forbidden")
            coordinates.add(coordinate)
            if source.block_ordinal != target.block_ordinal:
                cross_block_count += 1
            if source.component_family != target.component_family:
                cross_family_count += 1
            if (
                source.block_ordinal != target.block_ordinal
                and source.component_family != target.component_family
            ):
                cross_block_cross_family_count += 1

        if arm_id == JOINT_CROSS_BLOCK_CHART_ARM_ID and (
            not declared_edges
            or cross_block_cross_family_count == 0
        ):
            raise ValueError(
                "candidate requires a cross-block, cross-family transport"
            )
        lexical_metadata = getattr(lexical, "metadata", None)
        tie_receipt = (
            lexical_metadata() if callable(lexical_metadata) else None
        )
        observed_lexical_storage_count = len(
            _module_physical_spans(lexical)
        )
        if (
            not isinstance(tie_receipt, Mapping)
            or tie_receipt.get(
                "ingress_and_egress_share_the_same_storage"
            )
            is not True
            or tie_receipt.get("second_head_copy_present") is not False
            or tie_receipt.get("physical_storage_count")
            != observed_lexical_storage_count
            or observed_lexical_storage_count != 1
        ):
            raise ValueError(
                "lexical module did not authenticate one tied physical factor"
            )
        if arm_id == JOINT_CROSS_BLOCK_CHART_ARM_ID and any(
            item.source_checkpoint_coefficients_retained
            for item in declared_mutations
        ):
            raise ValueError(
                "joint candidate cannot retain source checkpoint mutations"
            )

        self.lexical = lexical
        self.mutations = nn.ModuleList(declared_mutations)
        self.transports = nn.ModuleList(declared_edges)
        self.terminal_norm = terminal_norm
        self.arm_id = arm_id
        self.final_logit_softcap = (
            None
            if final_logit_softcap is None
            else float(final_logit_softcap)
        )
        self._schedule = schedule
        self._carrier_factory = carrier_factory
        self._cross_block_transport_count = cross_block_count
        self._cross_family_transport_count = cross_family_count
        self._cross_block_cross_family_transport_count = (
            cross_block_cross_family_count
        )
        self.eval().requires_grad_(False)

    @property
    def ordered_event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self._schedule)

    @property
    def expected_mutation_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self._schedule[1:-2])

    @property
    def hidden_width(self) -> int:
        return self.mutations[0].hidden_width

    @property
    def deployed_scalar_count(self) -> int:
        return _module_scalar_count(self)

    def resource_inventory(self) -> dict[str, object]:
        groups = {
            "lexical": _module_physical_spans(self.lexical),
            "mutation": _module_physical_spans(self.mutations),
            "transport": _module_physical_spans(self.transports),
            "terminal": _module_physical_spans(self.terminal_norm),
        }
        ownership: dict[int, str] = {}
        for group, spans in groups.items():
            for pointer in spans:
                previous = ownership.setdefault(pointer, group)
                if previous != group:
                    raise RuntimeError(
                        "coefficient storage aliases across resource owners"
                    )
        lexical = sum(groups["lexical"].values())
        mutation = sum(groups["mutation"].values())
        transport = sum(groups["transport"].values())
        terminal = sum(groups["terminal"].values())
        total = lexical + mutation + transport + terminal
        if total != self.deployed_scalar_count:
            raise RuntimeError("joint executor resource inventory did not close")
        return {
            "lexical_shared_physical_scalars": lexical,
            "mutation_scalars": mutation,
            "transport_scalars": transport,
            "terminal_norm_scalars": terminal,
            "candidate_deployed_scalars": total,
            "inventory_closes": True,
            "lexical_ingress_egress_factor_counted_once": True,
            "serialized_artifact_written": False,
            "source_unique_parameter_reduction_claimed": False,
            "whole_model_mac_reduction_claimed": False,
            "latency_reduction_claimed": False,
        }

    def architecture_manifest(self) -> dict[str, object]:
        attention = [
            item
            for item in self.mutations
            if item.event_spec.event_kind == "attention_mutation"
        ]
        mlp = [
            item
            for item in self.mutations
            if item.event_spec.event_kind == "mlp_mutation"
        ]
        global_blocks = tuple(
            item.event_spec.block_ordinal
            for item in attention
            if item.event_spec.attention_kind == "global"
        )
        lexical_metadata = getattr(self.lexical, "metadata", None)
        lexical_manifest: object = (
            lexical_metadata()
            if callable(lexical_metadata)
            else {
                "kind": type(self.lexical).__qualname__,
                "stored_scalar_count": _module_scalar_count(self.lexical),
            }
        )
        return {
            "kind": "gemma3_joint_cross_block_chart_executor",
            "arm_id": self.arm_id,
            "candidate_arm": self.arm_id == JOINT_CROSS_BLOCK_CHART_ARM_ID,
            "exact_control_arm": self.arm_id == JOINT_EXACT_CONTROL_ARM_ID,
            "event_count": len(self._schedule),
            "ordered_event_ids": self.ordered_event_ids,
            "events": tuple(event.metadata() for event in self._schedule),
            "tied_lexical_ingress_and_egress_present": True,
            "lexical_factor_registered_once": True,
            "lexical": lexical_manifest,
            "attention_mutation_count": len(attention),
            "mlp_mutation_count": len(mlp),
            "global_attention_block_ordinals": global_blocks,
            "sliding_attention_mutation_count": sum(
                item.event_spec.attention_kind == "sliding_512"
                for item in attention
            ),
            "attention_sequence_operators_present": all(
                item.compiled_operator is not None
                or item.sequence_operator is not None
                for item in attention
            ),
            "direct_compiled_attention_mutation_count": sum(
                item.compiled_operator is not None for item in attention
            ),
            "generator_family_physical_sharing_present": len(
                {id(item._endpoint) for item in self.mutations}
            )
            < len(self.mutations),
            "unique_chart_generator_module_count": len(
                {id(item._endpoint) for item in self.mutations}
            ),
            "residual_carrier": {
                "mutation_count": len(self.mutations),
                "initial_version": 0,
                "final_version": len(self.mutations),
                "identity_preserved_by_ordered_delta_updates": True,
                "factory_kind": (
                    "DenseResidualCarrierFactory"
                    if self._carrier_factory is None
                    else type(self._carrier_factory).__qualname__
                ),
            },
            "transport_edge_count": len(self.transports),
            "cross_block_transport_count": (
                self._cross_block_transport_count
            ),
            "cross_component_family_transport_count": (
                self._cross_family_transport_count
            ),
            "cross_block_cross_component_transport_count": (
                self._cross_block_cross_family_transport_count
            ),
            "transports": tuple(
                edge.metadata() for edge in self.transports
            ),
            "strictly_forward_transports": True,
            "forward_skip_fanout_and_fanin_supported": True,
            "terminal_rmsnorm_present": True,
            "final_logit_softcap": self.final_logit_softcap,
            "live_teacher_reference_present": False,
            "native_fallback_present": False,
            "native_layer_objects_present": False,
            "ordered_per_mutation_operator_receipts_captured": True,
            "attention_source_call_tripwire_required": True,
            "source_checkpoint_coefficients_retained_by_mutations": any(
                item.source_checkpoint_coefficients_retained
                for item in self.mutations
            ),
            "compression_success_claimed": False,
            "source_free_artifact_established": False,
            "resource_inventory": self.resource_inventory(),
        }

    def _forward_components_impl(
        self,
        input_ids: Tensor,
        sequence: SequenceContext,
        *,
        observer: JointGraphEventObserver | None,
        retain_gradient_tape: bool,
        shadow_source_site_id: str | None = None,
        shadow_target_event_id: str | None = None,
        shadow_perturbation: Tensor | None = None,
    ) -> JointGraphObservedExecution:
        if observer is not None and not callable(observer):
            raise TypeError("event observer must be callable")
        if type(retain_gradient_tape) is not bool:
            raise TypeError("retain_gradient_tape must be bool")
        if retain_gradient_tape and torch.is_inference_mode_enabled():
            raise RuntimeError(
                "event score gradients cannot be captured in inference mode"
            )
        shadow_requested = any(
            value is not None
            for value in (
                shadow_source_site_id,
                shadow_target_event_id,
                shadow_perturbation,
            )
        )
        source_event: JointGraphEventSpec | None = None
        source_role: str | None = None
        target_event: JointGraphEventSpec | None = None
        if shadow_requested:
            if (
                self.arm_id != JOINT_EXACT_CONTROL_ARM_ID
                or not isinstance(shadow_source_site_id, str)
                or not isinstance(shadow_target_event_id, str)
                or not isinstance(shadow_perturbation, Tensor)
            ):
                raise ValueError(
                    "shadow intervention requires an exact control and complete endpoints"
                )
            event_by_id = {event.event_id: event for event in self._schedule}
            try:
                source_event_id, source_role = shadow_source_site_id.rsplit(
                    ".",
                    1,
                )
            except ValueError as error:
                raise ValueError("shadow source must be an event chart site") from error
            source_event = event_by_id.get(source_event_id)
            target_event = event_by_id.get(shadow_target_event_id)
            if (
                source_event is None
                or target_event is None
                or source_role not in {"input", "output"}
                or (
                    source_event.event_kind == "lexical_ingress"
                    and source_role != "output"
                )
                or (
                    source_event.event_kind
                    in {"attention_mutation", "mlp_mutation"}
                    and source_role not in {"input", "output"}
                )
                or source_event.event_kind
                not in {"lexical_ingress", "attention_mutation", "mlp_mutation"}
                or target_event.event_kind
                not in {"attention_mutation", "mlp_mutation"}
                or source_event.event_index >= target_event.event_index
            ):
                raise ValueError(
                    "shadow intervention requires a forward output-to-mutation edge"
                )
        if (
            not isinstance(input_ids, Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "input_ids must be an integer [batch, sequence] tensor"
            )
        if input_ids.shape != (
            sequence.batch_size,
            sequence.query_length,
        ):
            raise ValueError("input_ids do not match sequence context")
        context = ResidualGraphExecutionContext.bind(sequence)
        initial = self.lexical.ingress(input_ids)
        if (
            not isinstance(initial, Tensor)
            or initial.shape != (*input_ids.shape, self.hidden_width)
        ):
            raise ValueError("lexical ingress returned an invalid carrier")
        if shadow_requested:
            assert isinstance(shadow_perturbation, Tensor)
            if (
                shadow_perturbation.shape != initial.shape
                or shadow_perturbation.dtype != initial.dtype
                or shadow_perturbation.device != initial.device
                or not bool(torch.isfinite(shadow_perturbation).all())
            ):
                raise ValueError(
                    "shadow perturbation must match the exact residual carrier"
                )
        shadow_applied = False
        shadow_target_input: Tensor | None = None
        shadow_target_flow: Tensor | None = None
        if source_event is not None and source_event.event_kind == "lexical_ingress":
            assert isinstance(shadow_perturbation, Tensor)
            initial = initial + shadow_perturbation.masked_fill(
                ~context.query_valid_mask.unsqueeze(-1),
                0,
            )
            shadow_applied = True
        if retain_gradient_tape:
            initial = initial.detach().requires_grad_(True)
        live_observations: list[tuple[str, Tensor]] = []

        def observe(event_id: str, role: str, value: Tensor) -> None:
            site_id = f"{event_id}.{role}"
            if retain_gradient_tape:
                if not value.requires_grad:
                    raise RuntimeError(
                        f"observed site {site_id} is detached from the exact suffix"
                    )
                live_observations.append((site_id, value))
            if observer is not None:
                # The callback can mutate its private snapshot or return any
                # value without gaining a write path into exact execution.
                observer(
                    event_id=event_id,
                    role=role,
                    values=value.detach().clone(),
                )

        session = ResidualCarrierSession.begin(
            initial_state=initial,
            context=context,
            expected_mutation_ids=self.expected_mutation_ids,
            wire_id="gemma3_joint_cross_block_chart",
            factory=self._carrier_factory,
        )
        mutation_by_id = {
            mutation.event_spec.event_id: mutation
            for mutation in self.mutations
        }
        message_session = _JointGraphMessageSession(
            mutations=mutation_by_id,
            edges=tuple(self.transports),
            valid_positions=context.query_valid_mask,
            event_indexes={
                event.event_id: event.event_index
                for event in self._schedule[1:-2]
            },
            dtype=initial.dtype,
            device=initial.device,
        )
        try:
            observe("lexical.ingress", "output", initial)
            operator_receipts: list[JointGraphMutationRuntimeReceipt] = []
            mutation_chart_input_sha256s: list[str] = []
            mutation_delta_sha256s: list[str] = []
            for mutation in self.mutations:
                incoming = message_session.incoming_for(mutation)
                input_perturbation = (
                    shadow_perturbation
                    if (
                        source_event is not None
                        and source_role == "input"
                        and source_event.event_id
                        == mutation.event_spec.event_id
                    )
                    else None
                )
                execution = mutation.execute_with_incoming(
                    session.materialize(),
                    sequence,
                    incoming,
                    chart_input_perturbation=input_perturbation,
                )
                applied_delta = execution.delta
                if (
                    source_event is not None
                    and source_role == "output"
                    and source_event.event_id == mutation.event_spec.event_id
                ):
                    assert isinstance(shadow_perturbation, Tensor)
                    applied_delta = applied_delta + shadow_perturbation.masked_fill(
                        ~context.query_valid_mask.unsqueeze(-1),
                        0,
                    )
                    shadow_applied = True
                elif input_perturbation is not None:
                    shadow_applied = True
                mutation_chart_input_sha256s.append(
                    _tensor_value_sha256(execution.chart_input)
                )
                mutation_delta_sha256s.append(
                    _tensor_value_sha256(applied_delta)
                )
                if (
                    target_event is not None
                    and target_event.event_id == mutation.event_spec.event_id
                ):
                    shadow_target_input = execution.chart_input
                    shadow_target_flow = execution.delta
                observe(
                    mutation.event_spec.event_id,
                    "input",
                    execution.chart_input,
                )
                observe(
                    mutation.event_spec.event_id,
                    "output",
                    applied_delta,
                )
                session.apply_mutation(
                    mutation.event_spec.event_id,
                    applied_delta,
                )
                message_session.publish(
                    mutation,
                    execution.emitted_generator_latents,
                )
                operator_receipts.append(execution.runtime_receipt)
            message_session.finish(self.expected_mutation_ids)
            receipts = session.receipts
            carrier_version = session.version
            output = session.materialize()
            final_carrier = session.finish()
        except BaseException:
            if session.active:
                session.abort()
            if not message_session.closed:
                message_session.abort()
            raise
        if (
            carrier_version != PINNED_GEMMA3_MUTATION_COUNT
            or final_carrier.version != carrier_version
            or tuple(receipt.mutation_id for receipt in receipts)
            != self.expected_mutation_ids
        ):
            raise RuntimeError("joint graph residual carrier closure drifted")
        if shadow_requested and (
            not shadow_applied
            or shadow_target_input is None
            or shadow_target_flow is None
        ):
            raise RuntimeError("shadow intervention did not traverse its endpoints")
        normalized = self.terminal_norm(output)
        if not isinstance(normalized, Tensor) or normalized.shape != output.shape:
            raise ValueError("terminal normalization must preserve carrier shape")
        logits = self.lexical.egress(normalized)
        if (
            not isinstance(logits, Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
            or logits.shape[-1] <= 0
        ):
            raise ValueError("lexical egress returned invalid logits")
        if self.final_logit_softcap is not None:
            cap = self.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap
        execution = JointGraphWholeModelExecution(
            output=output,
            normalized_output=normalized,
            logits=logits,
            event_ids=self.ordered_event_ids,
            mutation_receipts=receipts,
            operator_receipts=tuple(operator_receipts),
            mutation_chart_input_sha256s=tuple(
                mutation_chart_input_sha256s
            ),
            mutation_delta_sha256s=tuple(mutation_delta_sha256s),
            carrier_version=carrier_version,
            transport_edge_count=len(self.transports),
            peak_live_transport_packets=(
                message_session.peak_live_packets
            ),
            peak_live_transport_packet_bytes=(
                message_session.peak_live_packet_bytes
            ),
        )
        tape = (
            JointGraphEventGradientTape(live_observations)
            if retain_gradient_tape
            else None
        )
        return JointGraphObservedExecution(
            execution=execution,
            gradient_tape=tape,
            shadow_source_site_id=(
                None
                if source_event is None or source_role is None
                else f"{source_event.event_id}.{source_role}"
            ),
            shadow_target_site_id=(
                None
                if target_event is None
                else f"{target_event.event_id}.output"
            ),
            shadow_target_input=shadow_target_input,
            shadow_target_flow=shadow_target_flow,
        )

    def forward_components(
        self,
        input_ids: Tensor,
        sequence: SequenceContext,
    ) -> JointGraphWholeModelExecution:
        return self._forward_components_impl(
            input_ids,
            sequence,
            observer=None,
            retain_gradient_tape=False,
            shadow_source_site_id=None,
            shadow_target_event_id=None,
            shadow_perturbation=None,
        ).execution

    def forward_components_with_observer(
        self,
        input_ids: Tensor,
        sequence: SequenceContext,
        *,
        observer: JointGraphEventObserver,
        retain_gradient_tape: bool = False,
    ) -> JointGraphObservedExecution:
        """Run exactly once while exporting detached 73-site snapshots.

        The observer is invoked in canonical site order with
        the keyword arguments ``event_id``, ``role``, and ``values``.  The
        value is a detached clone and the observer's return value is ignored.
        When requested, the result also carries a one-use tape for obtaining
        the exact loss gradient at those same sites.
        """

        return self._forward_components_impl(
            input_ids,
            sequence,
            observer=observer,
            retain_gradient_tape=retain_gradient_tape,
            shadow_source_site_id=None,
            shadow_target_event_id=None,
            shadow_perturbation=None,
        )

    def forward_components_with_shadow_intervention(
        self,
        input_ids: Tensor,
        sequence: SequenceContext,
        *,
        source_site_id: str,
        target_event_id: str,
        perturbation: Tensor,
    ) -> JointGraphObservedExecution:
        """Run an isolated exact shadow with one ephemeral source perturbation.

        The source may be lexical ingress or either chart site of a mutation;
        the target must be a strictly later mutation output.  This method is
        unavailable on compressed candidates.  It exists solely so edge-sidecar
        workers can evaluate JVPs and symmetric finite displacements without
        replacing the unperturbed baseline output.
        """

        return self._forward_components_impl(
            input_ids,
            sequence,
            observer=None,
            retain_gradient_tape=False,
            shadow_source_site_id=source_site_id,
            shadow_target_event_id=target_event_id,
            shadow_perturbation=perturbation,
        )

    execute = forward_components

    def forward(self, input_ids: Tensor, sequence: SequenceContext) -> Tensor:
        return self.forward_components(input_ids, sequence).logits


__all__ = [
    "ExactTokenLocalChartGenerator",
    "FactorizedHiddenChartExecution",
    "FactorizedHiddenChartGenerator",
    "Gemma3JointCrossBlockChartExecutor",
    "JOINT_CROSS_BLOCK_CHART_ARM_ID",
    "JOINT_EXACT_CONTROL_ARM_ID",
    "JointGraphCrossBlockTransportEdge",
    "JointGraphEventGradientTape",
    "JointGraphEventObserver",
    "JointGraphEventSpec",
    "JointGraphGeneratorMutation",
    "JointGraphMutationExecution",
    "JointGraphMutationRuntimeReceipt",
    "JointGraphObservedExecution",
    "JointGraphTransportSpec",
    "JointGraphWholeModelExecution",
    "PINNED_GEMMA3_BLOCK_COUNT",
    "PINNED_GEMMA3_EVENT_COUNT",
    "PINNED_GEMMA3_GLOBAL_ATTENTION_BLOCKS",
    "PINNED_GEMMA3_MUTATION_COUNT",
    "build_joint_cross_block_event_schedule",
]
