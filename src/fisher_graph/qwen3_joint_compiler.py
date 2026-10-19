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
import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Qwen3CausalLMAdapter
from .adapters.base import AttentionSpec, NormalizationSpec, RopeSpec
from .gemma3_attention_generator import (
    ExactAttentionGeneratorCoefficients,
    ReducedAttentionGeneratorCoefficients,
)
from .gemma3_cross_layer_chart_transport_executor import (
    Gemma3FactorizedChartGeneratorLayer,
)
from .gemma3_joint_cross_block_chart_executor import (
    ExactTokenLocalChartGenerator,
    JointGraphCrossBlockTransportEdge,
    JointGraphGeneratorMutation,
    JointGraphTransportSpec,
)
from .gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)
from .qwen3_attention_generator import (
    Qwen3AttentionGenerator,
    Qwen3AttentionGeneratorConfig,
    build_qwen3_exact_attention_generator,
    compile_qwen3_reduced_attention_generator,
)
from .qwen3_joint_executor import (
    QWEN3_JOINT_BLOCK_COUNT,
    QWEN3_JOINT_CHART_ARM_ID,
    QWEN3_JOINT_EXACT_CONTROL_ARM_ID,
    QWEN3_JOINT_MLX_HYBRID_ARM_ID,
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


class _Qwen3TorchRMSNorm(nn.Module):
    """Source-free RMSNorm module carrying a NormalizationSpec for round-trip."""

    def __init__(
        self,
        weight: Tensor,
        spec: NormalizationSpec,
        *,
        cast_before_scale: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().contiguous(), requires_grad=False)
        self.spec = spec
        self.variance_epsilon = float(spec.epsilon)
        self._cast_before_scale = bool(cast_before_scale)

    def forward(self, x: Tensor) -> Tensor:
        if self.spec.kind == "identity":
            return x
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + float(self.spec.epsilon))
        gain = self.weight.float()
        if self.spec.scale_parameterization == "unit_offset":
            gain = gain + 1.0
        if self._cast_before_scale:
            return y.to(dtype=x.dtype) * gain.to(dtype=x.dtype)
        return (y * gain).to(dtype=x.dtype)


class _Qwen3TorchSwiGLUMLP(nn.Module):
    """Source-free SwiGLU MLP matching Hugging Face projection names."""

    def __init__(self, gate_weight: Tensor, up_weight: Tensor, down_weight: Tensor) -> None:
        super().__init__()
        inter, hidden = gate_weight.shape
        self.gate_proj = nn.Linear(hidden, inter, bias=False, dtype=gate_weight.dtype, device=gate_weight.device)
        self.up_proj = nn.Linear(hidden, inter, bias=False, dtype=up_weight.dtype, device=up_weight.device)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=down_weight.dtype, device=down_weight.device)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate_weight.detach())
            self.up_proj.weight.copy_(up_weight.detach())
            self.down_proj.weight.copy_(down_weight.detach())
        self.eval().requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _norm_spec_from_dict(data: Mapping[str, object] | None, width: int) -> NormalizationSpec:
    if not data or data.get("kind") == "identity":
        return NormalizationSpec(
            kind="identity",
            width=width,
            epsilon=1e-6,
            affine=False,
            scale_parameterization="direct",
            compute_dtype="float32",
        )
    return NormalizationSpec(
        kind="rms_norm",
        width=width,
        epsilon=float(data.get("epsilon", 1e-6)),
        affine=bool(data.get("affine", True)),
        scale_parameterization=str(data.get("scale_parameterization", "direct")),
        compute_dtype=str(data.get("compute_dtype", "float32")),
    )


def compile_qwen3_joint_from_mlx(
    runtime: object,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Qwen3JointCrossBlockChartExecutor:
    """Convert an MLXQwen3 runtime (exact, partial-fit, or full-fit) into a Torch executor."""
    import mlx.core as mx

    p = runtime.parameters
    layers_cfg = runtime.layers
    layer_count = len(layers_cfg)
    schedule = build_qwen3_event_schedule(layer_count)

    def to_tensor(key: str) -> Tensor | None:
        if key not in p or p[key] is None:
            return None
        arr = np.asarray(p[key].astype(mx.float32))
        return torch.from_numpy(arr.copy()).to(device=device, dtype=dtype).contiguous()

    if "lexical.weight" in p:
        lex_w = to_tensor("lexical.weight")
        assert lex_w is not None
        hidden_width = int(lex_w.shape[1])
        lexical = SharedTiedLexicalFactor(
            exact_weight=lex_w,
            embed_scale=float(runtime.config.get("embed_scale", 1.0)),
        )
    else:
        cb, bs = to_tensor("lexical.codebook"), to_tensor("lexical.basis")
        assert cb is not None and bs is not None
        hidden_width = int(bs.shape[0])
        lexical = SharedTiedLexicalFactor(
            codebook=cb,
            basis=bs,
            embed_scale=float(runtime.config.get("embed_scale", 1.0)),
        )

    term_cfg = runtime.config.get("terminal_norm", {"kind": "identity"})
    if term_cfg.get("kind") == "identity" or "terminal.weight" not in p:
        terminal_norm: nn.Module = nn.Identity()
    else:
        tw = to_tensor("terminal.weight")
        assert tw is not None
        terminal_norm = _Qwen3TorchRMSNorm(tw, _norm_spec_from_dict(term_cfg, hidden_width))

    fingerprint = hashlib.sha256(
        json.dumps(runtime.config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    mutations: list[JointGraphGeneratorMutation] = []
    all_exact = True
    for i, layer in enumerate(layers_cfg):
        ac, mc = layer["attention"], layer["mlp"]
        a_event = schedule[1 + 2 * i]
        m_event = schedule[2 + 2 * i]
        a_pref, m_pref = f"attention.{i}.", f"mlp.{i}."
        qh = int(ac["query_heads"])
        kh = int(ac["key_value_heads"])
        hd = int(ac["head_dimension"])
        rope_factor = float(ac.get("rope_factor", 1.0))
        rope = RopeSpec(
            kind="rotary",
            theta=float(ac.get("theta", 1000000.0)),
            rotary_dimension=hd,
            scaling_kind="linear" if rope_factor != 1.0 else None,
            scaling_factor=rope_factor if rope_factor != 1.0 else None,
        )
        win = ac.get("window_size")
        attn_spec = AttentionSpec(
            kind="sliding_causal" if win is not None else "global_causal",
            query_heads=qh,
            key_value_heads=kh,
            head_dimension=hd,
            window_size=win,
            rope=rope,
            qk_norm=ac.get("qk_norm") is not None,
            query_scale=float(ac["query_scale"]),
            cache_kind="none",
        )
        in_norm_spec = _norm_spec_from_dict(ac["input_norm"], hidden_width)
        out_norm_spec = _norm_spec_from_dict(ac["output_norm"], hidden_width)
        qk_norm_spec = (
            _norm_spec_from_dict(ac["qk_norm"], hd)
            if ac.get("qk_norm") is not None
            else None
        )
        has_bias = any((a_pref + f"{k}_bias") in p for k in ("q", "k", "v", "o"))
        gen_cfg = Qwen3AttentionGeneratorConfig(
            invocation_id=a_event.event_id,
            residual_width=hidden_width,
            attention=attn_spec,
            input_norm=in_norm_spec,
            output_norm=out_norm_spec,
            qk_norm=qk_norm_spec,
            projection_bias=has_bias,
            dropout=0.0,
            logit_softcap=ac.get("logit_softcap"),
        )
        in_w = to_tensor(a_pref + "input_norm_weight")
        assert in_w is not None
        out_w = to_tensor(a_pref + "output_norm_weight")
        if out_w is None:
            out_w = torch.zeros(hidden_width, dtype=dtype, device=device)
        qw = to_tensor(a_pref + "q_weight")
        kw = to_tensor(a_pref + "k_weight")
        vw = to_tensor(a_pref + "v_weight")
        ow = to_tensor(a_pref + "output_decoder_weight")
        assert qw is not None and kw is not None and vw is not None and ow is not None
        qnw = to_tensor(a_pref + "q_norm_weight")
        knw = to_tensor(a_pref + "k_norm_weight")
        qb = to_tensor(a_pref + "q_bias")
        kb = to_tensor(a_pref + "k_bias")
        vb = to_tensor(a_pref + "v_bias")
        ob = to_tensor(a_pref + "o_bias")

        if (a_pref + "input_basis") in p or (a_pref + "output_encoder_weight") in p:
            all_exact = False
            ib = to_tensor(a_pref + "input_basis")
            if ib is None:
                ib = torch.eye(hidden_width, dtype=dtype, device=device)
            oew = to_tensor(a_pref + "output_encoder_weight")
            if oew is None:
                oew = torch.eye(qh * hd, dtype=dtype, device=device)
            attn_gen = compile_qwen3_reduced_attention_generator(
                gen_cfg,
                ReducedAttentionGeneratorCoefficients(
                    input_norm_weight=in_w,
                    input_basis=ib,
                    q_decoder_weight=qw,
                    k_decoder_weight=kw,
                    v_decoder_weight=vw,
                    output_encoder_weight=oew,
                    output_decoder_weight=ow,
                    output_norm_weight=out_w,
                    q_norm_weight=qnw,
                    k_norm_weight=knw,
                    q_bias=qb,
                    k_bias=kb,
                    v_bias=vb,
                    o_bias=ob,
                ),
            )
            a_retained = False
        else:
            attn_gen = Qwen3AttentionGenerator(
                gen_cfg,
                exact=ExactAttentionGeneratorCoefficients(
                    input_norm_weight=in_w,
                    q_weight=qw,
                    k_weight=kw,
                    v_weight=vw,
                    o_weight=ow,
                    output_norm_weight=out_w,
                    q_norm_weight=qnw,
                    k_norm_weight=knw,
                    q_bias=qb,
                    k_bias=kb,
                    v_bias=vb,
                    o_bias=ob,
                ),
                source_checkpoint_coefficients_retained=True,
            )
            a_retained = True

        mutations.append(
            JointGraphGeneratorMutation(
                event_spec=a_event,
                compiled_operator=attn_gen,
                source_checkpoint_coefficients_retained=a_retained,
            )
        )

        m_in_spec = _norm_spec_from_dict(mc["input_norm"], hidden_width)
        m_in_w = to_tensor(m_pref + "input_norm_weight")
        m_in_norm: nn.Module = (
            nn.Identity()
            if m_in_spec.kind == "identity" or m_in_w is None
            else _Qwen3TorchRMSNorm(m_in_w, m_in_spec)
        )
        m_out_spec = _norm_spec_from_dict(mc["output_norm"], hidden_width)
        m_out_w = to_tensor(m_pref + "output_norm_weight")
        m_out_norm: nn.Module = (
            nn.Identity()
            if m_out_spec.kind == "identity" or m_out_w is None
            else _Qwen3TorchRMSNorm(m_out_w, m_out_spec)
        )

        if (m_pref + "chart_centers") in p:
            all_exact = False
            aw = to_tensor(m_pref + "affine_weight")
            ab = to_tensor(m_pref + "affine_bias")
            cc = to_tensor(m_pref + "chart_centers")
            cb_m = to_tensor(m_pref + "chart_bases")
            cs = to_tensor(m_pref + "chart_distance_scales")
            lge = to_tensor(m_pref + "local_generator_encoders")
            cmd = to_tensor(m_pref + "chart_mode_decoders")
            assert (
                aw is not None
                and ab is not None
                and cc is not None
                and cb_m is not None
                and cs is not None
                and lge is not None
                and cmd is not None
            )
            mdw = to_tensor(m_pref + "mode_decoder_weight")
            if mdw is None:
                mdw = torch.eye(hidden_width, dtype=dtype, device=device)
            chart_gen: nn.Module = Gemma3FactorizedChartGeneratorLayer(
                affine_weight=aw,
                affine_bias=ab,
                mode_decoder_weight=mdw,
                chart_centers=cc,
                chart_bases=cb_m,
                chart_distance_scales=cs,
                local_generator_encoders=lge,
                chart_mode_decoders=cmd,
                chart_temperature=float(mc.get("temperature", 1.0)),
                model_fingerprint=fingerprint,
                layer_ordinal=i,
                fit_row_identity_sha256=fingerprint,
                compiled_prefix_catalog_sha256=fingerprint,
            )
            m_retained = False
        else:
            gw = to_tensor(m_pref + "gate_weight")
            uw = to_tensor(m_pref + "up_weight")
            dw = to_tensor(m_pref + "down_weight")
            assert gw is not None and uw is not None and dw is not None
            swiglu = _Qwen3TorchSwiGLUMLP(gw, uw, dw)
            chart_gen = ExactTokenLocalChartGenerator(
                operator=swiglu,
                hidden_width=hidden_width,
                chart_count=int(mc.get("emitted_chart_count", 1)),
                generator_rank=int(mc.get("emitted_generator_rank", hidden_width)),
            )
            m_retained = True

        mutations.append(
            JointGraphGeneratorMutation(
                event_spec=m_event,
                input_norm=m_in_norm,
                chart_generator=chart_gen,
                output_norm=m_out_norm,
                source_checkpoint_coefficients_retained=m_retained,
            )
        )

    event_id_by_site = {}
    for i in range(layer_count):
        event_id_by_site[f"attention.{i}"] = schedule[1 + 2 * i].event_id
        event_id_by_site[f"mlp.{i}"] = schedule[2 + 2 * i].event_id

    edges: list[JointGraphCrossBlockTransportEdge] = []
    for idx, edge in enumerate(runtime.config.get("transports", ())):
        tw_edge = to_tensor(edge["weight"])
        assert tw_edge is not None
        src_event = event_id_by_site[edge["source"]]
        dst_event = event_id_by_site[edge["target"]]
        edges.append(
            JointGraphCrossBlockTransportEdge(
                spec=JointGraphTransportSpec(
                    transport_id=f"transport.{idx}.{src_event}.to.{dst_event}",
                    source_event_id=src_event,
                    target_event_id=dst_event,
                    source_chart_index=int(edge["source_chart"]),
                    target_chart_index=int(edge["target_chart"]),
                ),
                transport_weight=tw_edge,
            )
        )

    src_arm = runtime.config.get("source")
    if src_arm in {QWEN3_JOINT_CHART_ARM_ID, QWEN3_JOINT_EXACT_CONTROL_ARM_ID, QWEN3_JOINT_MLX_HYBRID_ARM_ID}:
        arm_id = str(src_arm)
    elif all_exact and not edges:
        arm_id = QWEN3_JOINT_EXACT_CONTROL_ARM_ID
    else:
        arm_id = QWEN3_JOINT_MLX_HYBRID_ARM_ID

    return Qwen3JointCrossBlockChartExecutor(
        lexical=lexical,
        mutations=mutations,
        transports=edges,
        terminal_norm=terminal_norm,
        final_logit_softcap=runtime.config.get("final_logit_softcap"),
        arm_id=arm_id,
        layer_count=layer_count,
    )


__all__ = [
    "assemble_qwen3_joint_executor",
    "compile_qwen3_joint_exact_control",
    "compile_qwen3_joint_from_mlx",
]
