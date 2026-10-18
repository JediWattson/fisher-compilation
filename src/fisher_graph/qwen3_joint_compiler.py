"""Qwen 3 joint cross-block compiler: assemble a 59-event exact control.

Mirrors :func:`gemma3_joint_cross_block_compiler.compile_gemma3_joint_exact_control`
for :class:`Qwen3CausalLMAdapter`.  The Gemma compiler is source-pinned to
the 18-layer/640-width artifact, so the Qwen assembly path lives here.

The exact control clones every block into executor-owned modules: exact
attention generators (Qwen 3 GQA/QK-norm/RoPE semantics), the native MLP
behind an ``ExactTokenLocalChartGenerator`` control wrapper, the real
per-layer norm modules, the tied lexical factor, and the terminal RMSNorm.
No fitted or reduced components are admitted; the result must reproduce
the teacher bit-for-bit.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .adapters import Qwen3CausalLMAdapter
from .gemma3_joint_cross_block_chart_executor import (
    ExactTokenLocalChartGenerator,
    JointGraphCrossBlockTransportEdge,
    JointGraphGeneratorMutation,
    JointGraphTransportSpec,
)
from .gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)
from .qwen3_attention_generator import build_qwen3_exact_attention_generator
from .qwen3_joint_executor import (
    QWEN3_JOINT_BLOCK_COUNT,
    QWEN3_JOINT_CHART_ARM_ID,
    QWEN3_JOINT_EXACT_CONTROL_ARM_ID,
    Qwen3JointCrossBlockChartExecutor,
    build_qwen3_event_schedule,
)


def _qwen3_lexical_factor(adapter: Qwen3CausalLMAdapter) -> SharedTiedLexicalFactor:
    """Tied lexical factor holding the exact embedding matrix once.

    Qwen 3 ties ``embed_tokens`` to ``lm_head`` and applies no embedding
    scale, so the ingress and the egress share one physical weight at
    ``embed_scale == 1``.
    """

    backbone = getattr(adapter.module, "model", None)
    embed = getattr(backbone, "embed_tokens", None)
    head = getattr(adapter.module, "lm_head", None)
    if not isinstance(embed, nn.Embedding) or not isinstance(head, nn.Linear):
        raise TypeError("Qwen 3 backbone omitted tied lexical modules")
    if head.weight.data_ptr() != embed.weight.data_ptr():
        raise ValueError("Qwen 3 lexical modules are not storage-tied")
    return SharedTiedLexicalFactor(
        exact_weight=embed.weight.detach().clone(),
        embed_scale=1.0,
    )


def _qwen3_terminal_norm(adapter: Qwen3CausalLMAdapter) -> nn.Module:
    backbone = getattr(adapter.module, "model", None)
    source_norm = getattr(backbone, "norm", None)
    if not isinstance(source_norm, nn.Module):
        raise TypeError("Qwen 3 backbone omitted terminal RMSNorm")
    return copy.deepcopy(source_norm).eval().requires_grad_(False)


def _mlp_components(
    adapter: Qwen3CausalLMAdapter,
    layer_id: str,
    *,
    hidden_width: int,
) -> tuple[nn.Module, ExactTokenLocalChartGenerator, nn.Module]:
    """Clone one block's MLP path into executor-owned mutation components.

    Qwen 3 applies ``post_attention_layernorm`` before the MLP and has no
    output norm on the residual delta, so the pipeline is
    ``norm -> exact cloned MLP -> identity``.
    """

    module = adapter.source_module(layer_id)
    input_norm = getattr(module, "post_attention_layernorm", None)
    mlp = getattr(module, "mlp", None)
    if not isinstance(input_norm, nn.Module) or not isinstance(mlp, nn.Module):
        raise TypeError(
            "Qwen 3 layer must expose post_attention_layernorm and mlp"
        )
    chart = ExactTokenLocalChartGenerator(
        operator=copy.deepcopy(mlp).eval().requires_grad_(False),
        hidden_width=hidden_width,
        chart_count=1,
        generator_rank=hidden_width,
    )
    return (
        copy.deepcopy(input_norm).eval().requires_grad_(False),
        chart,
        nn.Identity(),
    )


def assemble_qwen3_joint_executor(
    adapter: Qwen3CausalLMAdapter,
    *,
    lexical: nn.Module,
    attention_generators: Sequence[nn.Module],
    mlp_input_norms: Sequence[nn.Module],
    mlp_chart_generators: Sequence[nn.Module],
    mlp_output_norms: Sequence[nn.Module],
    transports: Sequence[JointGraphCrossBlockTransportEdge],
    terminal_norm: nn.Module,
    arm_id: str,
    layer_count: int = QWEN3_JOINT_BLOCK_COUNT,
) -> Qwen3JointCrossBlockChartExecutor:
    """Bind Qwen components (59 events for the default 28 decoder layers)."""

    if not isinstance(adapter, Qwen3CausalLMAdapter):
        raise TypeError("adapter must be a Qwen3CausalLMAdapter")
    if len(adapter.layers) != layer_count:
        raise ValueError("adapter layer count does not match the schedule")
    counts = (
        len(tuple(attention_generators)),
        len(tuple(mlp_input_norms)),
        len(tuple(mlp_chart_generators)),
        len(tuple(mlp_output_norms)),
    )
    if any(count != layer_count for count in counts):
        raise ValueError("component inventory must cover every decoder block")

    schedule = build_qwen3_event_schedule(layer_count)
    exact = arm_id == QWEN3_JOINT_EXACT_CONTROL_ARM_ID
    mutations: list[JointGraphGeneratorMutation] = []
    for block in range(layer_count):
        attention_event = schedule[1 + 2 * block]
        mlp_event = schedule[2 + 2 * block]
        generator = attention_generators[block]
        manifest_method = getattr(generator, "architecture_manifest", None)
        manifest = manifest_method() if callable(manifest_method) else {}
        expected_kind = "exact_full_rank" if exact else "reduced_coordinates"
        if manifest.get("generator_kind") != expected_kind:
            raise ValueError("attention generator kind disagrees with arm")
        mutations.append(
            JointGraphGeneratorMutation(
                event_spec=attention_event,
                compiled_operator=generator,
                source_checkpoint_coefficients_retained=exact,
            )
        )
        chart = mlp_chart_generators[block]
        if exact and not isinstance(chart, ExactTokenLocalChartGenerator):
            raise TypeError("exact arm requires exact token-local generators")
        if not exact and isinstance(chart, ExactTokenLocalChartGenerator):
            raise TypeError(
                "candidate arm requires fitted MLP chart generators"
            )
        mutations.append(
            JointGraphGeneratorMutation(
                event_spec=mlp_event,
                input_norm=mlp_input_norms[block],
                chart_generator=chart,
                output_norm=mlp_output_norms[block],
                source_checkpoint_coefficients_retained=exact,
            )
        )

    return Qwen3JointCrossBlockChartExecutor(
        lexical=lexical,
        mutations=mutations,
        transports=transports,
        terminal_norm=terminal_norm,
        final_logit_softcap=None,
        arm_id=arm_id,
        layer_count=layer_count,
    )


def compile_qwen3_joint_exact_control(
    adapter: Qwen3CausalLMAdapter,
) -> Qwen3JointCrossBlockChartExecutor:
    """Clone Qwen 3 into an exact executor (59 events for 28 layers)."""

    if not isinstance(adapter, Qwen3CausalLMAdapter):
        raise TypeError("adapter must be a Qwen3CausalLMAdapter")
    layer_count = len(adapter.layers)
    hidden_width = adapter.layers[0].residual_width
    schedule = build_qwen3_event_schedule(layer_count)
    attention = tuple(
        build_qwen3_exact_attention_generator(
            adapter,
            f"layer.{block}",
            invocation_id=schedule[1 + 2 * block].event_id,
        )
        for block in range(layer_count)
    )
    input_norms: list[nn.Module] = []
    mlps: list[nn.Module] = []
    output_norms: list[nn.Module] = []
    for block in range(layer_count):
        input_norm, chart, output_norm = _mlp_components(
            adapter,
            f"layer.{block}",
            hidden_width=hidden_width,
        )
        input_norms.append(input_norm)
        mlps.append(chart)
        output_norms.append(output_norm)
    return assemble_qwen3_joint_executor(
        adapter,
        lexical=_qwen3_lexical_factor(adapter),
        attention_generators=attention,
        mlp_input_norms=tuple(input_norms),
        mlp_chart_generators=tuple(mlps),
        mlp_output_norms=tuple(output_norms),
        transports=(),
        terminal_norm=_qwen3_terminal_norm(adapter),
        arm_id=QWEN3_JOINT_EXACT_CONTROL_ARM_ID,
        layer_count=layer_count,
    )


__all__ = [
    "assemble_qwen3_joint_executor",
    "compile_qwen3_joint_exact_control",
]
