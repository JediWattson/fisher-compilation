"""Compiler support for the joint Gemma 3 cross-block chart arm.

The compiler has two explicit products:

* an exact 39-event control cloned from a :class:`Gemma3CausalLMAdapter`; and
* a joint candidate assembled from one tied lexical factor, 18 reduced
  attention generators, 18 MLP chart generators, and fitted forward
  transports.

This module does not load a checkpoint, open prompt data, fit coefficients, or
evaluate either executor.  It only binds caller-owned fitted artifacts to the
source-independent runtime and proves object/storage separation from an
already-loaded teacher.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass

import torch
from torch import nn

from .adapters import Gemma3CausalLMAdapter
from .gemma3_attention_generator import (
    compile_exact_attention_generator_from_structured_executor,
)
from .gemma3_full_body_residual_carrier_experiment import (
    _compile_complete_body,
)
from .gemma3_full_width_single_layer_experiment import (
    _assert_source_independence,
)
from .gemma3_joint_cross_block_chart_executor import (
    ExactTokenLocalChartGenerator,
    Gemma3JointCrossBlockChartExecutor,
    JOINT_CROSS_BLOCK_CHART_ARM_ID,
    JOINT_EXACT_CONTROL_ARM_ID,
    JointGraphCrossBlockTransportEdge,
    JointGraphGeneratorMutation,
    JointGraphTransportSpec,
    build_joint_cross_block_event_schedule,
)
from .gemma3_joint_cross_block_transport_fit import (
    FittedJointCrossBlockTransport,
)
from .gemma3_layer_agnostic_whole_model_compression_protocol import (
    PINNED_HIDDEN_WIDTH,
    PINNED_LAYER_COUNT,
    PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
)
from .gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
    _source_lexical_modules,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


GEMMA3_JOINT_CROSS_BLOCK_COMPILER_SCHEMA = (
    "fisher_graph.gemma3_joint_cross_block.compiler_receipt.v1"
)


class _ExactStructuredFeedForwardOperator(nn.Module):
    """Give the structured exact MLP its trace-free chart-call signature."""

    def __init__(self, operator: nn.Module, *, prefix: str) -> None:
        super().__init__()
        if not isinstance(operator, nn.Module):
            raise TypeError("exact feed-forward operator must be an nn.Module")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("exact feed-forward prefix must be nonempty")
        self.operator = operator
        self.prefix = prefix
        self.eval().requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.operator(
            values,
            trace=None,
            prefix=self.prefix,
        )


@dataclass(frozen=True, slots=True)
class Gemma3JointGraphCompilation:
    """One source-independent executor plus compile-time audit receipts."""

    executor: Gemma3JointCrossBlockChartExecutor
    compilation_kind: str
    source_model_parameter_count: int | None
    source_independence_audit: Mapping[str, object] | None
    source_model_fingerprint: str | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.executor,
            Gemma3JointCrossBlockChartExecutor,
        ):
            raise TypeError("compilation executor has the wrong type")
        if self.compilation_kind not in {
            "exact_39_event_control",
            "joint_cross_block_chart_candidate",
        }:
            raise ValueError("compilation_kind is invalid")
        if self.source_model_parameter_count is not None and (
            type(self.source_model_parameter_count) is not int
            or self.source_model_parameter_count <= 0
        ):
            raise ValueError("source model parameter count is invalid")
        if self.source_independence_audit is not None and (
            not isinstance(self.source_independence_audit, Mapping)
            or self.source_independence_audit.get("passed") is not True
        ):
            raise ValueError("source independence audit did not pass")
        if self.source_model_fingerprint is not None and (
            not isinstance(self.source_model_fingerprint, str)
            or len(self.source_model_fingerprint) != 64
        ):
            raise ValueError("source model fingerprint is invalid")

    def metadata(self) -> dict[str, object]:
        resources = self.executor.resource_inventory()
        candidate = int(resources["candidate_deployed_scalars"])
        source = self.source_model_parameter_count
        exact = self.compilation_kind == "exact_39_event_control"
        return {
            "schema": GEMMA3_JOINT_CROSS_BLOCK_COMPILER_SCHEMA,
            "compilation_kind": self.compilation_kind,
            "source_model_fingerprint": self.source_model_fingerprint,
            "source_model_parameter_count": source,
            "candidate_deployed_scalar_count": candidate,
            "exact_parameter_inventory_parity": (
                None if source is None or not exact else candidate == source
            ),
            "projected_parameter_difference": (
                None if source is None else source - candidate
            ),
            "source_independence_audit": (
                None
                if self.source_independence_audit is None
                else copy.deepcopy(dict(self.source_independence_audit))
            ),
            "executor_manifest": self.executor.architecture_manifest(),
            "resource_inventory": resources,
            "model_loaded_by_compiler": False,
            "prompt_data_opened": False,
            "tokenization_performed": False,
            "fitting_performed": False,
            "experiment_executed": False,
            "metrics_present": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
        }


def _manifest(module: nn.Module) -> Mapping[str, object]:
    callback = getattr(module, "architecture_manifest", None)
    value = callback() if callable(callback) else None
    if not isinstance(value, Mapping):
        raise ValueError("compiled module omitted an architecture manifest")
    return value


def _lexical_metadata(lexical: nn.Module) -> Mapping[str, object]:
    callback = getattr(lexical, "metadata", None)
    value = callback() if callable(callback) else None
    if not isinstance(value, Mapping):
        raise ValueError("lexical module omitted authenticated metadata")
    return value


def _validate_component_counts(
    *,
    attention_generators: Sequence[nn.Module],
    mlp_input_norms: Sequence[nn.Module],
    mlp_chart_generators: Sequence[nn.Module],
    mlp_output_norms: Sequence[nn.Module],
) -> tuple[
    tuple[nn.Module, ...],
    tuple[nn.Module, ...],
    tuple[nn.Module, ...],
    tuple[nn.Module, ...],
]:
    groups = tuple(
        tuple(values)
        for values in (
            attention_generators,
            mlp_input_norms,
            mlp_chart_generators,
            mlp_output_norms,
        )
    )
    if any(
        len(group) != PINNED_LAYER_COUNT
        or any(not isinstance(item, nn.Module) for item in group)
        for group in groups
    ):
        raise ValueError(
            "joint compiler requires 18 attention and 18 complete MLP parts"
        )
    return groups  # type: ignore[return-value]


def assemble_gemma3_joint_cross_block_executor(
    *,
    lexical: nn.Module,
    attention_generators: Sequence[nn.Module],
    mlp_input_norms: Sequence[nn.Module],
    mlp_chart_generators: Sequence[nn.Module],
    mlp_output_norms: Sequence[nn.Module],
    transports: Sequence[JointGraphCrossBlockTransportEdge],
    terminal_norm: nn.Module,
    final_logit_softcap: float | None,
    arm_id: str,
) -> Gemma3JointCrossBlockChartExecutor:
    """Assemble an exact or candidate executor from independent components."""

    if not isinstance(lexical, nn.Module):
        raise TypeError("lexical must be an nn.Module")
    if not isinstance(terminal_norm, nn.Module):
        raise TypeError("terminal_norm must be an nn.Module")
    if arm_id not in {
        JOINT_EXACT_CONTROL_ARM_ID,
        JOINT_CROSS_BLOCK_CHART_ARM_ID,
    }:
        raise ValueError("joint compiler arm_id is invalid")
    (
        attention,
        input_norms,
        mlps,
        output_norms,
    ) = _validate_component_counts(
        attention_generators=attention_generators,
        mlp_input_norms=mlp_input_norms,
        mlp_chart_generators=mlp_chart_generators,
        mlp_output_norms=mlp_output_norms,
    )
    lexical_receipt = _lexical_metadata(lexical)
    exact = arm_id == JOINT_EXACT_CONTROL_ARM_ID
    if lexical_receipt.get("is_exact_copied_matrix") is not exact:
        raise ValueError("lexical factor kind disagrees with compiler arm")

    schedule = build_joint_cross_block_event_schedule()
    mutations: list[JointGraphGeneratorMutation] = []
    for block in range(PINNED_LAYER_COUNT):
        attention_event = schedule[1 + 2 * block]
        mlp_event = schedule[2 + 2 * block]
        attention_manifest = _manifest(attention[block])
        expected_generator_kind = (
            "exact_full_rank" if exact else "reduced_coordinates"
        )
        if attention_manifest.get("generator_kind") != expected_generator_kind:
            raise ValueError("attention generator kind disagrees with arm")
        mutations.append(
            JointGraphGeneratorMutation(
                event_spec=attention_event,
                compiled_operator=attention[block],
                source_checkpoint_coefficients_retained=exact,
            )
        )
        chart = mlps[block]
        if getattr(chart, "hidden_width", None) != PINNED_HIDDEN_WIDTH:
            raise ValueError("MLP chart hidden width is not pinned Gemma width")
        if exact and not isinstance(chart, ExactTokenLocalChartGenerator):
            raise TypeError("exact arm requires exact token-local generators")
        if not exact and isinstance(chart, ExactTokenLocalChartGenerator):
            raise TypeError("candidate arm requires fitted MLP chart generators")
        layer_ordinal = getattr(chart, "layer_ordinal", block)
        if layer_ordinal != block:
            raise ValueError("MLP chart layer provenance order drifted")
        mutations.append(
            JointGraphGeneratorMutation(
                event_spec=mlp_event,
                input_norm=input_norms[block],
                chart_generator=chart,
                output_norm=output_norms[block],
                source_checkpoint_coefficients_retained=exact,
            )
        )

    executor = Gemma3JointCrossBlockChartExecutor(
        lexical=lexical,
        mutations=mutations,
        transports=transports,
        terminal_norm=terminal_norm,
        final_logit_softcap=final_logit_softcap,
        arm_id=arm_id,
    )
    manifest = executor.architecture_manifest()
    if (
        manifest["attention_mutation_count"] != PINNED_LAYER_COUNT
        or manifest["mlp_mutation_count"] != PINNED_LAYER_COUNT
        or manifest["direct_compiled_attention_mutation_count"]
        != PINNED_LAYER_COUNT
        or manifest["event_count"] != 39
    ):
        raise RuntimeError("assembled joint event inventory did not close")
    resources = executor.resource_inventory()
    if resources.get("inventory_closes") is not True:
        raise RuntimeError("assembled joint resource inventory did not close")
    return executor


def materialize_fitted_joint_cross_block_transports(
    *,
    target_event_id: str,
    fitted: Sequence[FittedJointCrossBlockTransport],
) -> tuple[JointGraphCrossBlockTransportEdge, ...]:
    """Turn one target event's fit-only blocks into runtime edges."""

    if not isinstance(target_event_id, str) or not target_event_id:
        raise ValueError("target_event_id must be nonempty")
    declared = tuple(fitted)
    if any(
        not isinstance(item, FittedJointCrossBlockTransport)
        for item in declared
    ):
        raise TypeError("fitted transports have the wrong type")
    ordered = tuple(
        sorted(
            declared,
            key=lambda item: (
                item.greedy_step,
                item.source_event_id,
                item.source_chart_index,
                item.target_chart_index,
            ),
        )
    )
    if tuple(item.greedy_step for item in ordered) != tuple(range(len(ordered))):
        raise ValueError("fitted transport greedy steps must be contiguous")
    return tuple(
        JointGraphCrossBlockTransportEdge(
            spec=JointGraphTransportSpec(
                transport_id=(
                    f"transport.{target_event_id}.{item.greedy_step}."
                    f"{item.source_chart_index}.{item.target_chart_index}"
                ),
                source_event_id=item.source_event_id,
                target_event_id=target_event_id,
                source_chart_index=item.source_chart_index,
                target_chart_index=item.target_chart_index,
            ),
            transport_weight=item.transport_weight,
        )
        for item in ordered
    )


def _pinned_adapter(adapter: Gemma3CausalLMAdapter) -> None:
    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be a Gemma3CausalLMAdapter")
    if (
        len(adapter.layers) != PINNED_LAYER_COUNT
        or any(
            layer.residual_width != PINNED_HIDDEN_WIDTH
            for layer in adapter.layers
        )
    ):
        raise ValueError("adapter is not the pinned 18-block Gemma shape")
    source_count = sum(
        parameter.numel() for parameter in adapter.module.parameters()
    )
    if source_count != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT:
        raise ValueError("adapter source parameter inventory is not pinned")


def _terminal_norm_and_softcap(
    adapter: Gemma3CausalLMAdapter,
) -> tuple[nn.Module, float | None]:
    backbone = getattr(adapter.module, "model", None)
    source_norm = getattr(backbone, "norm", None)
    if not isinstance(source_norm, nn.Module):
        raise TypeError("Gemma backbone omitted terminal RMSNorm")
    terminal = copy.deepcopy(source_norm).eval().requires_grad_(False)
    softcap = getattr(adapter.module.config, "final_logit_softcapping", None)
    return terminal, None if softcap is None else float(softcap)


def _exact_lexical_factor(
    adapter: Gemma3CausalLMAdapter,
) -> SharedTiedLexicalFactor:
    _, _, weight, embed_scale = _source_lexical_modules(adapter)
    return SharedTiedLexicalFactor(
        exact_weight=weight,
        embed_scale=embed_scale,
    )


def _structured_layers(
    adapter: Gemma3CausalLMAdapter,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[StructuredTransformerLayerExecutor, ...]:
    body = _compile_complete_body(
        adapter,
        dtype=dtype,
        device=device,
    )
    result = tuple(block.executor for block in body.blocks)
    if (
        len(result) != PINNED_LAYER_COUNT
        or any(
            not isinstance(item, StructuredTransformerLayerExecutor)
            for item in result
        )
    ):
        raise RuntimeError("exact body clone omitted structured layers")
    return result


def _compilation_receipt(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3JointCrossBlockChartExecutor,
    compilation_kind: str,
) -> Gemma3JointGraphCompilation:
    source_count = sum(
        parameter.numel() for parameter in adapter.module.parameters()
    )
    independence = _assert_source_independence(
        adapter.module,
        {compilation_kind: executor},
    )
    result = Gemma3JointGraphCompilation(
        executor=executor,
        compilation_kind=compilation_kind,
        source_model_parameter_count=source_count,
        source_independence_audit=independence,
        source_model_fingerprint=adapter.model_fingerprint(),
    )
    metadata = result.metadata()
    if compilation_kind == "exact_39_event_control" and (
        metadata["exact_parameter_inventory_parity"] is not True
    ):
        raise RuntimeError("exact joint control parameter inventory drifted")
    return result


def compile_gemma3_joint_exact_control(
    adapter: Gemma3CausalLMAdapter,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Gemma3JointGraphCompilation:
    """Clone the pinned Gemma into one exact, standalone 39-event executor."""

    _pinned_adapter(adapter)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("compiler dtype must be floating point")
    source_device = next(adapter.module.parameters()).device
    resolved_device = source_device if device is None else torch.device(device)
    if resolved_device != source_device:
        raise ValueError("compiler and loaded source must share a device")
    structured = _structured_layers(
        adapter,
        dtype=dtype,
        device=resolved_device,
    )
    schedule = build_joint_cross_block_event_schedule()
    attention = tuple(
        compile_exact_attention_generator_from_structured_executor(
            layer,
            invocation_id=schedule[1 + 2 * block].event_id,
        )
        for block, layer in enumerate(structured)
    )
    mlps = tuple(
        ExactTokenLocalChartGenerator(
            operator=_ExactStructuredFeedForwardOperator(
                layer.feed_forward,
                prefix=f"mutation.{2 + 2 * block}.mlp",
            ),
            hidden_width=PINNED_HIDDEN_WIDTH,
            chart_count=1,
            generator_rank=PINNED_HIDDEN_WIDTH,
        )
        for block, layer in enumerate(structured)
    )
    terminal, softcap = _terminal_norm_and_softcap(adapter)
    executor = assemble_gemma3_joint_cross_block_executor(
        lexical=_exact_lexical_factor(adapter),
        attention_generators=attention,
        mlp_input_norms=tuple(
            layer.feed_forward_input_norm for layer in structured
        ),
        mlp_chart_generators=mlps,
        mlp_output_norms=tuple(
            layer.feed_forward_output_norm for layer in structured
        ),
        transports=(),
        terminal_norm=terminal,
        final_logit_softcap=softcap,
        arm_id=JOINT_EXACT_CONTROL_ARM_ID,
    )
    return _compilation_receipt(
        adapter=adapter,
        executor=executor,
        compilation_kind="exact_39_event_control",
    )


def compile_gemma3_joint_cross_block_candidate(
    adapter: Gemma3CausalLMAdapter,
    *,
    lexical: nn.Module,
    attention_generators: Sequence[nn.Module],
    mlp_chart_generators: Sequence[nn.Module],
    transports: Sequence[JointGraphCrossBlockTransportEdge],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Gemma3JointGraphCompilation:
    """Bind caller-fitted joint components to one standalone candidate."""

    _pinned_adapter(adapter)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("compiler dtype must be floating point")
    source_device = next(adapter.module.parameters()).device
    resolved_device = source_device if device is None else torch.device(device)
    if resolved_device != source_device:
        raise ValueError("compiler and loaded source must share a device")
    structured = _structured_layers(
        adapter,
        dtype=dtype,
        device=resolved_device,
    )
    terminal, softcap = _terminal_norm_and_softcap(adapter)
    executor = assemble_gemma3_joint_cross_block_executor(
        lexical=lexical,
        attention_generators=attention_generators,
        mlp_input_norms=tuple(
            layer.feed_forward_input_norm for layer in structured
        ),
        mlp_chart_generators=mlp_chart_generators,
        mlp_output_norms=tuple(
            layer.feed_forward_output_norm for layer in structured
        ),
        transports=transports,
        terminal_norm=terminal,
        final_logit_softcap=softcap,
        arm_id=JOINT_CROSS_BLOCK_CHART_ARM_ID,
    )
    return _compilation_receipt(
        adapter=adapter,
        executor=executor,
        compilation_kind="joint_cross_block_chart_candidate",
    )


__all__ = [
    "GEMMA3_JOINT_CROSS_BLOCK_COMPILER_SCHEMA",
    "Gemma3JointGraphCompilation",
    "assemble_gemma3_joint_cross_block_executor",
    "compile_gemma3_joint_cross_block_candidate",
    "compile_gemma3_joint_exact_control",
    "materialize_fitted_joint_cross_block_transports",
]
