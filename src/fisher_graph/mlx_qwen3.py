"""Standalone MLX execution of dense Qwen3 and compiled Qwen3 chart graphs.

PyTorch is needed only by the two conversion entry points. Forward, prefill,
cached decode, captures, perturbation VJPs, and saved artifacts use MLX arrays.
Parameters are a flat tree so fitting can differentiate only selected leaves.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


def _core():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("Install fisher-graph[mlx] on Apple silicon") from exc
    return mx


def _array(value, dtype=None):
    mx = _core()
    # CPU float conversion handles torch bfloat16 without NumPy bf16 support.
    result = mx.array(value.detach().cpu().float().numpy())
    return result.astype(dtype) if dtype is not None else result


def _norm_spec(module):
    name = type(module).__name__
    if name == "Identity":
        return {"kind": "identity"}
    if hasattr(module, "spec"):
        return asdict(module.spec)
    epsilon = getattr(module, "variance_epsilon", getattr(module, "eps", None))
    if epsilon is None or not hasattr(module, "weight"):
        raise TypeError(f"Unsupported norm: {name}")
    if "Qwen3_5RMSNorm" in name and "Gated" not in name:
        return {"kind": "rms_norm", "epsilon": float(epsilon),
                "scale_parameterization": "unit_offset", "cast_before_scale": False}
    return {"kind": "rms_norm", "epsilon": float(epsilon),
            "scale_parameterization": "direct", "cast_before_scale": True}


def _rms(x, weight, spec, *, fast=False):
    if spec["kind"] == "identity":
        return x
    mx = _core()
    if fast and hasattr(mx, "fast"):
        w = weight.astype(x.dtype)
        if spec.get("scale_parameterization") == "unit_offset":
            w = w + 1.0
        return mx.fast.rms_norm(x, w, spec["epsilon"])
    y = x.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + spec["epsilon"])
    gain = weight.astype(mx.float32)
    if spec.get("scale_parameterization") == "unit_offset":
        gain = gain + 1.0
    if spec.get("cast_before_scale", False):
        return y.astype(x.dtype) * gain.astype(x.dtype)
    return (y * gain).astype(x.dtype)


_GDN_METAL_KERNEL = None


def _get_gdn_metal_kernel(mx):
    global _GDN_METAL_KERNEL
    if _GDN_METAL_KERNEL is not None:
        return _GDN_METAL_KERNEL
    if not hasattr(mx, "fast") or not hasattr(mx, "metal") or not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    """
    _GDN_METAL_KERNEL = mx.fast.metal_kernel(
        name="fisher_gated_delta_step",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )
    return _GDN_METAL_KERNEL


@dataclass(frozen=True)
class MLXQwen3Cache:
    """Keys/values plus per-row physical validity and logical RoPE positions."""
    key_values: tuple
    attention_mask: Any
    position_ids: Any
    all_valid: bool = False

    @property
    def length(self):
        return self.attention_mask.shape[1]


@dataclass(frozen=True)
class MLXQwen3Output:
    logits: Any
    hidden_states: Any
    captures: dict
    cache: MLXQwen3Cache


class MLXQwen3:
    """A dense Qwen3 or fitted chart graph with no live source-model calls.

    Default positions count valid tokens from zero, matching the adapter.
    Supply explicit position_ids to match HF's physical-position default on
    padded batches. Invalid query deltas are zero, as in the compiled executor.
    """

    def __init__(self, parameters: Mapping, config: Mapping):
        self.parameters = dict(parameters)
        self.config = copy.deepcopy(dict(config))
        self.layers = tuple(self.config["layers"])
        if not self.layers:
            raise ValueError("Qwen3 requires at least one decoder layer")
        self._fast_cache: dict[str, Any] = {}

    def with_parameters(self, updates, *, remove=(), layer_overrides=None):
        parameters = {k: v for k, v in self.parameters.items() if k not in remove}
        parameters.update(updates)
        config = copy.deepcopy(self.config)
        for index, parts in (layer_overrides or {}).items():
            for part, changes in parts.items():
                config["layers"][index][part].update(changes)
        return type(self)(parameters, config)

    @classmethod
    def from_hf(cls, model, *, dtype=None):
        """Convert a loaded HF model directly, without an intermediate clone."""
        config = model.config.to_dict()
        text_model = getattr(model.model, "language_model", model.model)
        embedding, head = text_model.embed_tokens.weight, model.lm_head.weight
        config["tie_word_embeddings"] = (
            embedding.data_ptr() == head.data_ptr()
            and embedding.shape == head.shape and embedding.stride() == head.stride()
        )
        if isinstance(config.get("text_config"), dict):
            config["text_config"]["tie_word_embeddings"] = config["tie_word_embeddings"]
        state = model.state_dict()
        runtime = cls._from_hf_weights(config, {k: _array(v, dtype) for k, v in state.items()
                                               if k != "lm_head.weight" or not config.get("tie_word_embeddings")})
        # Config values can lag behind deliberate live-module interventions.
        runtime.config["terminal_norm"] = _norm_spec(text_model.norm)
        for layer, source_layer in zip(runtime.layers, text_model.layers):
            layer["mlp"]["input_norm"] = _norm_spec(source_layer.post_attention_layernorm)
            if "deltanet" in layer:
                layer["deltanet"]["input_norm"] = _norm_spec(source_layer.input_layernorm)
                layer["deltanet"]["norm"] = _norm_spec(source_layer.linear_attn.norm)
            else:
                layer["attention"]["input_norm"] = _norm_spec(source_layer.input_layernorm)
                qnorm, knorm = (_norm_spec(n) for n in (source_layer.self_attn.q_norm, source_layer.self_attn.k_norm))
                if qnorm != knorm:
                    raise ValueError("Qwen3 MLX requires matching Q/K normalization semantics")
                layer["attention"]["qk_norm"] = qnorm
                layer["attention"]["query_scale"] = float(source_layer.self_attn.scaling)
        return runtime

    @classmethod
    def from_pretrained(cls, directory, *, dtype=None):
        """Read an existing local HF safetensors snapshot directly into MLX.

        No download or torch model allocation occurs. This is the low-memory
        path for 1.7B on a 16GB Mac. Default preserves checkpoint precision.
        """
        mx = _core()
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        index = directory / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            files = sorted({
                fname for key, fname in weight_map.items()
                if not key.startswith(("mtp.", "model.visual.", "visual.", "vision_tower."))
            })
        else:
            files = ["model.safetensors"]
        weights = {}
        for name in files:
            weights.update(mx.load(str(directory / name)))
        if dtype is not None:
            weights = {
                k: (v.astype(dtype) if v.dtype in (mx.float16, mx.bfloat16, mx.float32) else v)
                for k, v in weights.items()
            }
        return cls._from_hf_weights(config, weights)

    @classmethod
    def _from_hf_weights(cls, source, weights):
        quantization = source.get("quantization") or source.get("quantization_config")
        if source.get("model_type") == "qwen3_5" and isinstance(source.get("text_config"), Mapping):
            text_source = dict(source["text_config"])
            if "tie_word_embeddings" in source and "tie_word_embeddings" not in text_source:
                text_source["tie_word_embeddings"] = source["tie_word_embeddings"]
            source = text_source
        if not quantization:
            quantization = source.get("quantization") or source.get("quantization_config")
        model_type = source.get("model_type")
        is_qwen3_5 = model_type in ("qwen3_5", "qwen3_5_text")
        if (model_type != "qwen3" and not is_qwen3_5) or source.get("hidden_act", "silu") != "silu":
            raise ValueError("Only dense Qwen3 with SiLU is supported")
        if source.get("num_experts", 0) > 0:
            raise ValueError("Only dense Qwen3 with SiLU is supported")
        num_layers = int(source["num_hidden_layers"])
        if is_qwen3_5:
            interval = int(source.get("full_attention_interval", 4))
            layer_types = list(source.get("layer_types") or [
                "linear_attention" if (i + 1) % interval else "full_attention"
                for i in range(num_layers)
            ])
            if source.get("use_sliding_window") or len(layer_types) != num_layers or any(
                x not in ("full_attention", "linear_attention") for x in layer_types
            ):
                raise ValueError("This loader supports Qwen3.5 full_attention and linear_attention layers only")
        else:
            layer_types = ["full_attention"] * num_layers
            if source.get("use_sliding_window") or any(x != "full_attention" for x in source.get("layer_types", []) or []):
                raise ValueError("This loader supports global Qwen3 attention only")
        rope = source.get("rope_parameters") or source.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) not in ("default", "linear"):
            raise ValueError("Only default and linear RoPE are supported")
        factor = float(rope.get("factor", 1.0))
        heads = int(source["num_attention_heads"])
        kvheads = int(source["num_key_value_heads"])
        dim = int(source.get("head_dim") or (source["hidden_size"] // heads))
        partial_rotary = float(rope.get("partial_rotary_factor", source.get("partial_rotary_factor", 1.0)))
        rotary_dim = int(round(dim * partial_rotary))
        if heads % kvheads or dim % 2 or rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > dim or factor <= 0:
            raise ValueError("Invalid GQA or rotary dimensions")
        sanitized_mlx = is_qwen3_5 and any(
            "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] == 1 and v.shape[1] != 1
            for k, v in weights.items()
        )
        norm = {
            "kind": "rms_norm",
            "epsilon": float(source["rms_norm_eps"]),
            "scale_parameterization": ("direct" if sanitized_mlx else "unit_offset") if is_qwen3_5 else "direct",
            "cast_before_scale": not is_qwen3_5,
        }
        gated_norm = {
            "kind": "rms_norm",
            "epsilon": float(source["rms_norm_eps"]),
            "scale_parameterization": "direct",
            "cast_before_scale": True,
        }
        prefix = next(
            (cand for cand in ("model.language_model.", "language_model.model.", "model.") if cand + "embed_tokens.weight" in weights),
            "model.",
        )
        p = {"terminal.weight": weights[prefix + "norm.weight"]}

        def copy_linear(target_key, source_prefix):
            p[target_key] = weights[source_prefix + ".weight"]
            if source_prefix + ".scales" in weights:
                p[target_key + ".scales"] = weights[source_prefix + ".scales"]
            if source_prefix + ".biases" in weights:
                p[target_key + ".biases"] = weights[source_prefix + ".biases"]

        copy_linear("lexical.weight", prefix + "embed_tokens")
        if not source.get("tie_word_embeddings", False):
            head_prefix = next(
                (cand for cand in ("lm_head", "language_model.lm_head", prefix + "lm_head") if cand + ".weight" in weights),
                "lm_head",
            )
            copy_linear("lexical.head_weight", head_prefix)
        output_gate = bool(source.get("attn_output_gate", True)) if is_qwen3_5 else bool(source.get("attn_output_gate", False))
        layers = []
        for i, layer_type in enumerate(layer_types):
            s = f"{prefix}layers.{i}."
            m = f"mlp.{i}."
            p[m + "input_norm_weight"] = weights[s + "post_attention_layernorm.weight"]
            for name in ("gate", "up", "down"):
                copy_linear(m + name + "_weight", s + f"mlp.{name}_proj")
            mlp_cfg = {"input_norm": norm, "output_norm": {"kind": "identity"}, "temperature": 1.0}
            if layer_type == "linear_attention":
                d = f"deltanet.{i}."
                kh = int(source["linear_num_key_heads"])
                vh = int(source["linear_num_value_heads"])
                kd = int(source["linear_key_head_dim"])
                vd = int(source["linear_value_head_dim"])
                kernel_size = int(source.get("linear_conv_kernel_dim", 4))
                if vh % kh != 0:
                    raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
                p[d + "input_norm_weight"] = weights[s + "input_layernorm.weight"]
                copy_linear(d + "qkv_weight", s + "linear_attn.in_proj_qkv")
                copy_linear(d + "z_weight", s + "linear_attn.in_proj_z")
                copy_linear(d + "b_weight", s + "linear_attn.in_proj_b")
                copy_linear(d + "a_weight", s + "linear_attn.in_proj_a")
                conv_w = weights[s + "linear_attn.conv1d.weight"]
                p[d + "conv_weight"] = conv_w.reshape(conv_w.shape[0], kernel_size)
                if s + "linear_attn.conv1d.bias" in weights:
                    p[d + "conv_bias"] = weights[s + "linear_attn.conv1d.bias"]
                p[d + "A_log"] = weights[s + "linear_attn.A_log"]
                p[d + "dt_bias"] = weights[s + "linear_attn.dt_bias"]
                p[d + "norm_weight"] = weights[s + "linear_attn.norm.weight"]
                copy_linear(d + "output_decoder_weight", s + "linear_attn.out_proj")
                if s + "linear_attn.out_proj.bias" in weights:
                    p[d + "o_bias"] = weights[s + "linear_attn.out_proj.bias"]
                layers.append({
                    "deltanet": {
                        "input_norm": norm,
                        "output_norm": {"kind": "identity"},
                        "norm": gated_norm,
                        "key_heads": kh,
                        "value_heads": vh,
                        "key_head_dimension": kd,
                        "value_head_dimension": vd,
                        "carrier_rank": kd,
                        "conv_kernel_size": kernel_size,
                        "qk_l2_eps": 1e-6,
                    },
                    "mlp": mlp_cfg,
                })
            else:
                a = f"attention.{i}."
                p[a + "input_norm_weight"] = weights[s + "input_layernorm.weight"]
                for name in ("q", "k", "v", "o"):
                    target = "output_decoder_weight" if name == "o" else name + "_weight"
                    copy_linear(a + target, s + f"self_attn.{name}_proj")
                    bias = weights.get(s + f"self_attn.{name}_proj.bias")
                    if bias is not None:
                        p[a + name + "_bias"] = bias
                for name in ("q", "k"):
                    p[a + name + "_norm_weight"] = weights[s + f"self_attn.{name}_norm.weight"]
                attn_cfg = {
                    "input_norm": norm, "output_norm": {"kind": "identity"}, "qk_norm": norm,
                    "query_heads": heads, "key_value_heads": kvheads, "head_dimension": dim,
                    "query_scale": dim ** -0.5, "theta": rope.get("rope_theta", source.get("rope_theta", 1000000.0)),
                    "rope_factor": factor, "window_size": None, "logit_softcap": None,
                }
                if rotary_dim != dim:
                    attn_cfg["rotary_dimension"] = rotary_dim
                if output_gate:
                    attn_cfg["output_gate"] = True
                layers.append({"attention": attn_cfg, "mlp": mlp_cfg})
        cfg = {"format_version": 1, "layers": layers, "terminal_norm": norm,
               "embed_scale": 1.0, "final_logit_softcap": None,
               "source": "hf_qwen3_5" if is_qwen3_5 else "hf_qwen3"}
        if isinstance(quantization, Mapping) and any(k.endswith(".scales") for k in p):
            cfg["quantization"] = {
                "group_size": int(quantization.get("group_size", 64)),
                "bits": int(quantization.get("bits", 4)),
                "mode": str(quantization.get("mode", "affine")),
            }
        return cls(p, cfg)

    @classmethod
    def from_torch(cls, executor, *, dtype=None):
        """Convert an exact or reduced Qwen joint executor, including transports."""
        from .qwen3_joint_executor import Qwen3JointCrossBlockChartExecutor
        from .gemma3_joint_cross_block_chart_executor import ExactTokenLocalChartGenerator
        if not isinstance(executor, Qwen3JointCrossBlockChartExecutor):
            raise TypeError("Expected a Qwen3JointCrossBlockChartExecutor")
        p, layers = {}, []

        def put(key, value):
            if value is not None:
                p[key] = _array(value, dtype)

        def norm(module, key):
            put(key, getattr(module, "weight", None))
            return _norm_spec(module)

        lexical = executor.lexical
        put("lexical.weight", lexical.exact_weight)
        put("lexical.codebook", lexical.codebook)
        put("lexical.basis", lexical.basis)
        event_names = {}
        for i in range(len(executor.mutations) // 2):
            am, mm = executor.mutations[2*i:2*i+2]
            ag, mg = am.compiled_operator, mm.chart_generator
            a, m = f"attention.{i}.", f"mlp.{i}."
            event_names[am.event_spec.event_id] = a[:-1]
            event_names[mm.event_spec.event_id] = m[:-1]
            for name in ("input_norm_weight", "output_norm_weight", "q_norm_weight", "k_norm_weight",
                         "q_weight", "k_weight", "v_weight", "q_bias", "k_bias", "v_bias", "o_bias",
                         "input_basis", "output_encoder_weight", "output_decoder_weight"):
                put(a + name, getattr(ag, name, None))
            c, att = ag.config, ag.config.attention
            ac = {"input_norm": asdict(c.input_norm), "output_norm": asdict(c.output_norm),
                  "qk_norm": asdict(c.qk_norm) if c.qk_norm else None,
                  "query_heads": att.query_heads, "key_value_heads": att.key_value_heads,
                  "head_dimension": att.head_dimension, "query_scale": float(att.query_scale),
                  "theta": att.rope.theta, "rope_factor": att.rope.scaling_factor or 1.0,
                  "window_size": att.window_size, "logit_softcap": c.logit_softcap}
            mc = {"input_norm": norm(mm.input_norm, m + "input_norm_weight"),
                  "output_norm": norm(mm.output_norm, m + "output_norm_weight"),
                  "temperature": float(getattr(mg, "chart_temperature", 1.0))}
            if isinstance(mg, ExactTokenLocalChartGenerator):
                for name in ("gate", "up", "down"):
                    put(m + name + "_weight", getattr(mg.operator, name + "_proj").weight)
                mc["emitted_chart_count"] = mg.chart_count
                mc["emitted_generator_rank"] = mg.generator_rank
            else:
                for name in ("chart_centers", "chart_bases", "chart_distance_scales", "local_generator_encoders", "chart_mode_decoders"):
                    if not hasattr(mg, name):
                        raise TypeError("Unsupported MLP chart generator")
                    put(m + name, getattr(mg, name))
                put(m + "affine_weight", mg.affine_generators.weight)
                put(m + "affine_bias", mg.affine_generators.bias)
                put(m + "mode_decoder_weight", mg.mode_decoder.weight)
            layers.append({"attention": ac, "mlp": mc})
        edges = []
        for i, edge in enumerate(executor.transports):
            s = edge.spec
            key = f"transport.{i}.weight"
            put(key, edge.transport_weight)
            edges.append({"source": event_names[s.source_event_id], "target": event_names[s.target_event_id],
                          "source_chart": s.source_chart_index, "target_chart": s.target_chart_index, "weight": key})
        config = {"format_version": 1, "layers": layers, "transports": edges,
                  "terminal_norm": norm(executor.terminal_norm, "terminal.weight"),
                  "embed_scale": float(lexical.embed_scale), "final_logit_softcap": executor.final_logit_softcap,
                  "source": executor.arm_id}
        return cls(p, config)

    def to_torch(self, *, dtype=None, device="cpu"):
        """Convert this MLX Qwen3 runtime into a standalone Torch joint executor."""
        import torch
        from .qwen3_joint_compiler import compile_qwen3_joint_from_mlx

        torch_dtype = torch.float32 if dtype is None else dtype
        return compile_qwen3_joint_from_mlx(self, dtype=torch_dtype, device=device)

    def save(self, directory):
        mx = _core()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(directory / "weights.safetensors"), self.parameters)
        (directory / "mlx_qwen3.json").write_text(json.dumps(self.config, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        mx = _core()
        directory = Path(directory)
        config = json.loads((directory / "mlx_qwen3.json").read_text())
        if config.get("format_version") != 1:
            raise ValueError("Unsupported MLX Qwen3 artifact version")
        return cls(mx.load(str(directory / "weights.safetensors")), config)

    def __call__(self, input_ids, *, attention_mask=None, position_ids=None,
                 parameters=None, capture=False, perturbations=None,
                 muted_sites=None, muted_charts=None,
                 return_logits=True, cache=None):
        mx = _core()
        p = self.parameters if parameters is None else parameters
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("input_ids must be a nonempty integer [batch, length] array")
        lexical = p.get("lexical.weight", p.get("lexical.codebook"))
        if cache is None or not isinstance(input_ids, mx.array):
            if not bool(mx.all((ids >= 0) & (ids < lexical.shape[0])).item()):
                raise ValueError("input_ids must lie within the lexical vocabulary")
        b, t = ids.shape
        all_valid = attention_mask is None and (cache is None or getattr(cache, "all_valid", False))
        valid = mx.ones((b, t), dtype=mx.bool_) if attention_mask is None else mx.array(attention_mask).astype(mx.bool_)
        if valid.shape != ids.shape:
            raise ValueError("attention_mask must describe only the current input tokens")
        if cache is not None and (cache.attention_mask.shape[0] != b or len(cache.key_values) != len(self.layers)):
            raise ValueError("cache batch/layer count mismatch")
        if position_ids is None:
            if all_valid:
                offset_pos = cache.length if cache is not None else 0
                positions = mx.broadcast_to(mx.arange(offset_pos, offset_pos + t, dtype=mx.int32)[None, :], (b, t))
            else:
                start = (mx.max(mx.where(cache.attention_mask, cache.position_ids, -1), axis=1, keepdims=True) + 1
                         if cache is not None else mx.zeros((b, 1), dtype=mx.int32))
                positions = mx.maximum(start + mx.cumsum(valid.astype(mx.int32), axis=1) - 1, 0)
        else:
            positions = mx.array(position_ids)
            if positions.shape != ids.shape or positions.dtype not in (mx.int32, mx.int64):
                raise ValueError("position_ids must be integer and match input_ids")
        key_valid = mx.concatenate((cache.attention_mask, valid), axis=1) if cache else valid
        key_positions = mx.concatenate((cache.position_ids, positions), axis=1) if cache else positions
        captures, messages, new_kv = {}, {}, []
        perturbations = perturbations or {}
        muted_site_set = set(muted_sites or ())
        muted_chart_set = {(int(layer_idx), int(chart_idx)) for layer_idx, chart_idx in (muted_charts or ())}
        seen = set()
        capture_sites = None if capture is True else set(capture or ())
        has_transports = bool(self.config.get("transports"))
        use_fast_kernel = (
            all_valid
            and parameters is None
            and not capture
            and not perturbations
            and mx.default_device() == mx.gpu
        )
        gdn_kernel = _get_gdn_metal_kernel(mx) if use_fast_kernel else None
        fc = self._fast_cache if use_fast_kernel else None

        def fast_rms(x_val, key, spec):
            if spec["kind"] == "identity":
                return x_val
            if fc is not None and hasattr(mx, "fast"):
                ck = ("rms", key, x_val.dtype)
                w = fc.get(ck)
                if w is None:
                    w = p[key].astype(x_val.dtype)
                    if spec.get("scale_parameterization") == "unit_offset":
                        w = w + 1.0
                    mx.eval(w)
                    fc[ck] = w
                return mx.fast.rms_norm(x_val, w, spec["epsilon"])
            return _rms(x_val, p.get(key), spec)

        def mask_valid(val):
            return val if all_valid else mx.where(valid[..., None], val, 0)

        def observe(name, value):
            if name in muted_site_set:
                value = mx.zeros_like(value)
            if name in perturbations:
                if perturbations[name].shape != value.shape:
                    raise ValueError(f"Perturbation shape mismatch at {name}")
                value = value + perturbations[name]
                seen.add(name)
            if capture_sites is None or name in capture_sites:
                captures[name] = value
            return value

        quant = self.config.get("quantization") or {}
        q_group_size = int(quant.get("group_size", 64))
        q_bits = int(quant.get("bits", 4))
        q_mode = str(quant.get("mode", "affine"))

        def linear(x, key, bias=None):
            if key + ".scales" in p:
                y = mx.quantized_matmul(
                    x, p[key], scales=p[key + ".scales"], biases=p.get(key + ".biases"),
                    transpose=True, group_size=q_group_size, bits=q_bits, mode=q_mode,
                )
            else:
                y = x @ p[key].T
            return y + p[bias] if bias in p else y

        def incoming(name, count, rank, dtype):
            cols = [[] for _ in range(count)]
            for edge in self.config.get("transports", []):
                if edge["target"] == name:
                    cols[edge["target_chart"]].append(messages[edge["source"]][..., edge["source_chart"], :] @ p[edge["weight"]])
            return mx.stack([sum(c, mx.zeros((b, t, rank), dtype=dtype)) for c in cols], axis=-2)

        if "lexical.weight" in p:
            if "lexical.weight.scales" in p:
                x = mx.dequantize(
                    p["lexical.weight"][ids],
                    scales=p["lexical.weight.scales"][ids],
                    biases=p["lexical.weight.biases"][ids] if "lexical.weight.biases" in p else None,
                    group_size=q_group_size, bits=q_bits, mode=q_mode,
                )
            else:
                x = p["lexical.weight"][ids]
        else:
            x = p["lexical.codebook"][ids] @ p["lexical.basis"].T
        x = observe("embedding", x * self.config.get("embed_scale", 1.0))
        for i, layer in enumerate(self.layers):
            m = f"mlp.{i}"
            mc = layer["mlp"]
            if "deltanet" in layer:
                d = f"deltanet.{i}"
                dc = layer["deltanet"]
                state = observe(d + ".input", x)
                n = observe(d + ".normalized", fast_rms(state, d + ".input_norm_weight", dc["input_norm"]))
                n_masked = mask_valid(n)
                coords = n_masked @ p[d + ".input_basis"] if d + ".input_basis" in p else n_masked
                coords = mask_valid(coords)
                if has_transports:
                    coords = coords + incoming(d, 1, coords.shape[-1], coords.dtype)[..., 0, :]
                kh, vh = dc["key_heads"], dc["value_heads"]
                kd, vd = dc["key_head_dimension"], dc["value_head_dimension"]
                k_total, v_total = kh * kd, vh * vd
                kernel_size = dc["conv_kernel_size"]
                conv_dim = 2 * k_total + v_total
                qkv = linear(coords, d + ".qkv_weight", d + ".qkv_bias")
                z = linear(coords, d + ".z_weight", d + ".z_bias").reshape(b, t, vh, vd)
                b_gate = linear(n_masked, d + ".b_weight", d + ".b_bias")
                a_gate = linear(n_masked, d + ".a_weight", d + ".a_bias")
                if cache and cache.key_values[i] is not None:
                    conv_state, rec_state = cache.key_values[i]
                else:
                    conv_state = mx.zeros((b, kernel_size - 1, conv_dim), dtype=qkv.dtype)
                    rec_state = None
                conv_in = mx.concatenate((conv_state, qkv), axis=1)
                new_conv_state = conv_in[:, -(kernel_size - 1):, :] if kernel_size > 1 else conv_state
                if t == 1 and fc is not None:
                    ck_conv = ("conv_w_t", d, qkv.dtype)
                    conv_wt = fc.get(ck_conv)
                    if conv_wt is None:
                        conv_wt = mx.contiguous(p[d + ".conv_weight"].reshape(conv_dim, kernel_size).T[None, :, :].astype(qkv.dtype))
                        mx.eval(conv_wt)
                        fc[ck_conv] = conv_wt
                    conv_out = mx.sum(conv_in * conv_wt, axis=1, keepdims=True)
                else:
                    conv_w = p[d + ".conv_weight"].reshape(conv_dim, kernel_size)
                    conv_out = sum(conv_in[:, k_idx : k_idx + t, :] * conv_w[:, k_idx] for k_idx in range(kernel_size))
                if d + ".conv_bias" in p:
                    conv_out = conv_out + p[d + ".conv_bias"]
                conv_f = conv_out.astype(mx.float32)
                conv_act = (conv_f * mx.sigmoid(conv_f)).astype(qkv.dtype)
                q_raw = conv_act[..., :k_total].reshape(b, t, kh, kd)
                k_raw = conv_act[..., k_total : 2 * k_total].reshape(b, t, kh, kd)
                v = observe(d + ".v", conv_act[..., 2 * k_total:]).reshape(b, t, vh, vd)
                eps = dc.get("qk_l2_eps", 1e-6)
                if use_fast_kernel:
                    inv_scale = kd ** -0.5
                    rms_eps = eps * (inv_scale ** 2)
                    q = (inv_scale ** 2) * mx.fast.rms_norm(q_raw, None, rms_eps)
                    k = inv_scale * mx.fast.rms_norm(k_raw, None, rms_eps)
                else:
                    q_f, k_f = q_raw.astype(mx.float32), k_raw.astype(mx.float32)
                    q_norm = (q_f * mx.rsqrt(mx.sum(q_f * q_f, axis=-1, keepdims=True) + eps)) * (kd ** -0.5)
                    k_norm = k_f * mx.rsqrt(mx.sum(k_f * k_f, axis=-1, keepdims=True) + eps)
                    q = observe(d + ".q", q_norm.astype(qkv.dtype).reshape(b, t, k_total)).reshape(b, t, kh, kd)
                    k = observe(d + ".k", k_norm.astype(qkv.dtype).reshape(b, t, k_total)).reshape(b, t, kh, kd)
                if d + ".carrier_basis" in p:
                    if use_fast_kernel:
                        ck_b = ("carrier_basis", d, q.dtype)
                        basis = fc.get(ck_b)
                        if basis is None:
                            basis = mx.contiguous(p[d + ".carrier_basis"].astype(q.dtype))
                            mx.eval(basis)
                            fc[ck_b] = basis
                        if b == 1 and t == 1:
                            r_dim = basis.shape[-1]
                            q_rec = (q.reshape(kh, 1, kd) @ basis).reshape(1, 1, kh, r_dim)
                            k_rec = (k.reshape(kh, 1, kd) @ basis).reshape(1, 1, kh, r_dim)
                        else:
                            q_rec = mx.contiguous((q.transpose(0, 2, 1, 3) @ basis).transpose(0, 2, 1, 3))
                            k_rec = mx.contiguous((k.transpose(0, 2, 1, 3) @ basis).transpose(0, 2, 1, 3))
                    else:
                        basis = p[d + ".carrier_basis"].astype(mx.float32)
                        q_rec = mx.einsum("bthd,hdr->bthr", q.astype(mx.float32), basis)
                        k_rec = mx.einsum("bthd,hdr->bthr", k.astype(mx.float32), basis)
                else:
                    q_rec, k_rec = (q, k) if use_fast_kernel else (q.astype(mx.float32), k.astype(mx.float32))
                beta = mx.sigmoid(b_gate.astype(mx.float32))
                a_val = a_gate.astype(mx.float32) + p[d + ".dt_bias"].astype(mx.float32)
                softplus_a = mx.log1p(mx.exp(-mx.abs(a_val))) + mx.maximum(a_val, 0.0)
                if fc is not None:
                    ck_a = ("neg_exp_A", d)
                    neg_exp_a = fc.get(ck_a)
                    if neg_exp_a is None:
                        neg_exp_a = -mx.exp(p[d + ".A_log"].astype(mx.float32))
                        mx.eval(neg_exp_a)
                        fc[ck_a] = neg_exp_a
                    g_decay = mx.exp(neg_exp_a * softplus_a)
                else:
                    g_decay = mx.exp(-mx.exp(p[d + ".A_log"].astype(mx.float32)) * softplus_a)
                carrier_dim = q_rec.shape[-1]
                if rec_state is None:
                    rec_state = mx.zeros((b, vh, vd, carrier_dim), dtype=mx.float32)
                else:
                    rec_state = rec_state.astype(mx.float32)
                if gdn_kernel is not None and carrier_dim >= 32 and carrier_dim % 32 == 0:
                    y_kern, rec_state = gdn_kernel(
                        inputs=[q_rec.astype(qkv.dtype), k_rec.astype(qkv.dtype), v, g_decay, beta.astype(qkv.dtype), rec_state, t],
                        template=[
                            ("InT", qkv.dtype),
                            ("StT", rec_state.dtype),
                            ("Dk", carrier_dim),
                            ("Dv", vd),
                            ("Hk", kh),
                            ("Hv", vh),
                        ],
                        grid=(32, vd, b * vh),
                        threadgroup=(32, 4, 1),
                        output_shapes=[(b, t, vh, vd), rec_state.shape],
                        output_dtypes=[qkv.dtype, rec_state.dtype],
                    )
                    rec_out = observe(d + ".recurrent_out", y_kern.reshape(b, t, v_total)).reshape(b, t, vh, vd)
                else:
                    q_rec, k_rec = q_rec.astype(mx.float32), k_rec.astype(mx.float32)
                    if vh // kh > 1:
                        q_rec = mx.repeat(q_rec, vh // kh, axis=2)
                        k_rec = mx.repeat(k_rec, vh // kh, axis=2)
                    v_f = v.astype(mx.float32)
                    ys = []
                    for step in range(t):
                        q_t = q_rec[:, step]
                        k_t = k_rec[:, step]
                        v_t = v_f[:, step]
                        g_t = g_decay[:, step, :, None, None]
                        beta_t = beta[:, step, :, None]
                        rec_state = rec_state * g_t
                        kv_mem = mx.sum(rec_state * k_t[:, :, None, :], axis=-1)
                        delta_v = (v_t - kv_mem) * beta_t
                        rec_state = rec_state + delta_v[:, :, :, None] * k_t[:, :, None, :]
                        y_t = mx.sum(rec_state * q_t[:, :, None, :], axis=-1)
                        ys.append(y_t)
                    rec_out = observe(d + ".recurrent_out", mx.stack(ys, axis=1).astype(qkv.dtype).reshape(b, t, v_total)).reshape(b, t, vh, vd)
                new_kv.append((new_conv_state, rec_state))
                normed_out = fast_rms(rec_out, d + ".norm_weight", dc["norm"])
                z_f = z.astype(mx.float32)
                gated_out = (normed_out.astype(mx.float32) * (z_f * mx.sigmoid(z_f))).astype(qkv.dtype)
                context = observe(d + ".context", gated_out.reshape(b, t, v_total))
                if d + ".value_basis" in p:
                    v_basis = p[d + ".value_basis"].astype(gated_out.dtype)
                    if b == 1 and t == 1:
                        r_v = v_basis.shape[-1]
                        outcoords = (gated_out.reshape(vh, 1, vd) @ v_basis).reshape(1, 1, vh * r_v)
                    else:
                        outcoords = (gated_out.transpose(0, 2, 1, 3) @ v_basis).transpose(0, 2, 1, 3).reshape(b, t, -1)
                elif d + ".output_encoder_weight" in p:
                    outcoords = linear(context, d + ".output_encoder_weight")
                else:
                    outcoords = context
                raw = linear(outcoords, d + ".output_decoder_weight", d + ".o_bias")
                if d + ".chart_centers" in p:
                    d_centers = p[d + ".chart_centers"]
                    d_disp = context[..., None, :] - d_centers
                    d_scales = p[d + ".chart_distance_scales"]
                    d_memb = mx.softmax(-0.5 * mx.sum(d_disp ** 2, axis=-1) / (d_scales ** 2 * dc.get("temperature", 1.0)), axis=-1)
                    if muted_chart_set:
                        d_keep = [0.0 if (i, c_idx) in muted_chart_set else 1.0 for c_idx in range(d_centers.shape[0])]
                        if any(v == 0.0 for v in d_keep):
                            d_memb = d_memb * mx.array(d_keep, dtype=d_memb.dtype)
                    d_coords = mx.einsum("btch,chr->btcr", d_disp, p[d + ".chart_bases"])
                    d_aug = mx.concatenate((mx.ones_like(d_coords[..., :1]), d_coords), axis=-1)
                    d_enc = p[d + ".local_generator_encoders"]
                    d_local = mx.einsum("btcr,crg->btcg", d_aug, d_enc)
                    d_inc = incoming(d, d_centers.shape[0], d_enc.shape[-1], context.dtype) if has_transports else 0
                    d_packet = d_memb[..., None] * (d_local + d_inc)
                    raw = raw + mx.einsum("btcg,cgm->btm", d_packet, p[d + ".chart_mode_decoders"])
                raw = observe(d + ".output", raw)
                delta = observe(d + ".delta", fast_rms(raw, d + ".output_norm_weight", dc["output_norm"]))
                x = state + mask_valid(delta)
                if has_transports:
                    messages[d] = mask_valid(outcoords)[..., None, :]
            else:
                a = f"attention.{i}"
                ac = layer["attention"]
                state = observe(a + ".input", x)
                n = observe(a + ".normalized", fast_rms(state, a + ".input_norm_weight", ac["input_norm"]))
                coords = n @ p[a + ".input_basis"] if a + ".input_basis" in p else n
                coords = mask_valid(coords)
                if has_transports:
                    coords = coords + incoming(a, 1, coords.shape[-1], coords.dtype)[..., 0, :]
                qh, kh, d = ac["query_heads"], ac["key_value_heads"], ac["head_dimension"]
                use_gate = bool(ac.get("output_gate", False))
                if use_gate:
                    q_raw = linear(coords, a + ".q_weight", a + ".q_bias").reshape(b, t, qh, 2 * d)
                    q_proj, attn_gate = q_raw[..., :d], q_raw[..., d:].reshape(b, t, qh * d)
                else:
                    q_proj = linear(coords, a + ".q_weight", a + ".q_bias").reshape(b, t, qh, d)
                    attn_gate = None
                q = q_proj.transpose(0, 2, 1, 3)
                if ac["qk_norm"] is not None:
                    q = fast_rms(q, a + ".q_norm_weight", ac["qk_norm"])
                k = linear(coords, a + ".k_weight", a + ".k_bias").reshape(b, t, kh, d).transpose(0, 2, 1, 3)
                if ac["qk_norm"] is not None:
                    k = fast_rms(k, a + ".k_norm_weight", ac["qk_norm"])
                v = linear(coords, a + ".v_weight", a + ".v_bias").reshape(b, t, kh, d).transpose(0, 2, 1, 3)
                d_rot = int(ac.get("rotary_dimension", d))
                offset = cache.length if cache else 0
                if use_fast_kernel and position_ids is None and ac["rope_factor"] == 1.0:
                    q = mx.fast.rope(q, d_rot, traditional=False, base=ac["theta"], scale=1.0, offset=offset)
                    k = mx.fast.rope(k, d_rot, traditional=False, base=ac["theta"], scale=1.0, offset=offset)
                else:
                    inv = ac["theta"] ** (-mx.arange(0, d_rot, 2, dtype=mx.float32) / d_rot)
                    freqs = positions.astype(mx.float32)[..., None] * inv / ac["rope_factor"]
                    angles = mx.concatenate((freqs, freqs), axis=-1)[:, None, :, :]
                    cos, sin = mx.cos(angles).astype(q.dtype), mx.sin(angles).astype(q.dtype)

                    def rotary(y):
                        y_rot, y_pass = y[..., :d_rot], y[..., d_rot:]
                        y_rot = y_rot * cos + mx.concatenate((-y_rot[..., d_rot // 2:], y_rot[..., :d_rot // 2]), axis=-1) * sin
                        return mx.concatenate((y_rot, y_pass), axis=-1) if d_rot < d else y_rot

                    q, k = rotary(q), rotary(k)
                if cache:
                    k = mx.concatenate((cache.key_values[i][0], k), axis=2)
                    v = mx.concatenate((cache.key_values[i][1], v), axis=2)
                new_kv.append((k, v))
                if use_fast_kernel and ac["logit_softcap"] is None and ac["window_size"] is None:
                    sdpa_mask = "causal" if t > 1 else None
                    raw_context = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=ac["query_scale"], mask=sdpa_mask
                    ).transpose(0, 2, 1, 3).reshape(b, t, qh * d)
                else:
                    kr, vr = mx.repeat(k, qh // kh, axis=1), mx.repeat(v, qh // kh, axis=1)
                    scores = (q @ kr.transpose(0, 1, 3, 2)) * ac["query_scale"]
                    if ac["logit_softcap"] is not None:
                        cap = ac["logit_softcap"]
                        scores = mx.tanh(scores / cap) * cap
                    physical_causal = mx.arange(key_valid.shape[1])[None, :] <= (mx.arange(t) + offset)[:, None]
                    allowed = valid[:, :, None] & key_valid[:, None, :] & physical_causal[None, :, :]
                    if ac["window_size"] is not None:
                        allowed = allowed & ((positions[:, :, None] - key_positions[:, None, :]) < ac["window_size"])
                    scores = mx.where(allowed[:, None, :, :], scores.astype(mx.float32), -3.4028234663852886e38)
                    probabilities = mx.softmax(scores, axis=-1).astype(q.dtype)
                    raw_context = (probabilities @ vr).transpose(0, 2, 1, 3).reshape(b, t, qh * d)
                if attn_gate is not None:
                    raw_context = (raw_context.astype(mx.float32) * mx.sigmoid(attn_gate.astype(mx.float32))).astype(raw_context.dtype)
                context = observe(a + ".context", raw_context)
                outcoords = linear(context, a + ".output_encoder_weight") if a + ".output_encoder_weight" in p else context
                raw = observe(a + ".output", linear(outcoords, a + ".output_decoder_weight", a + ".o_bias"))
                delta = observe(a + ".delta", fast_rms(raw, a + ".output_norm_weight", ac["output_norm"]))
                x = state + mask_valid(delta)
                if has_transports:
                    messages[a] = mask_valid(outcoords)[..., None, :]
            state = observe(m + ".input", x)
            n = observe(m + ".normalized", fast_rms(state, m + ".input_norm_weight", mc["input_norm"]))
            if m + ".chart_centers" in p:
                centers = p[m + ".chart_centers"]
                displacement = n[..., None, :] - centers
                scales = p[m + ".chart_distance_scales"]
                memberships = mx.softmax(-0.5 * mx.sum(displacement ** 2, axis=-1) / (scales ** 2 * mc.get("temperature", 1.0)), axis=-1)
                if muted_chart_set:
                    keep = [0.0 if (i, c_idx) in muted_chart_set else 1.0 for c_idx in range(centers.shape[0])]
                    if any(v == 0.0 for v in keep):
                        memberships = memberships * mx.array(keep, dtype=memberships.dtype)
                coordinates = mx.einsum("btch,chr->btcr", displacement, p[m + ".chart_bases"])
                aug = mx.concatenate((mx.ones_like(coordinates[..., :1]), coordinates), axis=-1)
                enc = p[m + ".local_generator_encoders"]
                local = mx.einsum("btcr,crg->btcg", aug, enc)
                inc = incoming(m, centers.shape[0], enc.shape[-1], n.dtype) if has_transports else 0
                packet = memberships[..., None] * (local + inc)
                modal = linear(n, m + ".affine_weight", m + ".affine_bias") + mx.einsum("btcg,cgm->btm", packet, p[m + ".chart_mode_decoders"])
                raw = linear(modal, m + ".mode_decoder_weight") if m + ".mode_decoder_weight" in p else modal
            else:
                gate = linear(n, m + ".gate_weight")
                raw = linear((gate * mx.sigmoid(gate)) * linear(n, m + ".up_weight"), m + ".down_weight")
                packet = None
            raw = observe(m + ".output", raw)
            delta = observe(m + ".delta", fast_rms(raw, m + ".output_norm_weight", mc["output_norm"]))
            x = state + mask_valid(delta)
            if has_transports:
                if packet is None:
                    packet = mx.zeros((b, t, mc.get("emitted_chart_count", 1), mc.get("emitted_generator_rank", n.shape[-1])), dtype=n.dtype)
                messages[m] = mx.where(valid[..., None, None], packet, 0)
        x = observe("terminal", x)
        normalized = observe("terminal.normalized", fast_rms(x, "terminal.weight", self.config["terminal_norm"]))
        logits = None
        if return_logits:
            if "lexical.head_weight" in p:
                logits = linear(normalized, "lexical.head_weight")
            elif "lexical.weight" in p:
                logits = linear(normalized, "lexical.weight")
            else:
                logits = (normalized @ p["lexical.basis"]) @ p["lexical.codebook"].T
            if self.config.get("final_logit_softcap") is not None:
                cap = self.config["final_logit_softcap"]
                logits = mx.tanh(logits / cap) * cap
        if set(perturbations) - seen:
            raise ValueError(f"Unknown perturbation sites: {sorted(set(perturbations) - seen)}")
        if capture_sites and capture_sites - set(captures):
            raise ValueError(f"Unknown capture sites: {sorted(capture_sites - set(captures))}")
        return MLXQwen3Output(logits, x, captures, MLXQwen3Cache(tuple(new_kv), key_valid, key_positions, all_valid))

    def prefill(self, input_ids, **kwargs):
        return self(input_ids, **kwargs)

    def decode(self, input_ids, cache, **kwargs):
        return self(input_ids, cache=cache, **kwargs)
