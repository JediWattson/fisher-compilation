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
    if type(module).__name__ == "Identity":
        return {"kind": "identity"}
    if hasattr(module, "spec"):
        return asdict(module.spec)
    epsilon = getattr(module, "variance_epsilon", getattr(module, "eps", None))
    if epsilon is None or not hasattr(module, "weight"):
        raise TypeError(f"Unsupported norm: {type(module).__name__}")
    return {"kind": "rms_norm", "epsilon": float(epsilon),
            "scale_parameterization": "direct", "cast_before_scale": True}


def _rms(x, weight, spec):
    if spec["kind"] == "identity":
        return x
    mx = _core()
    y = x.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + spec["epsilon"])
    gain = weight.astype(mx.float32)
    if spec.get("scale_parameterization") == "unit_offset":
        gain = gain + 1.0
    if spec.get("cast_before_scale", False):
        return y.astype(x.dtype) * gain.astype(x.dtype)
    return (y * gain).astype(x.dtype)


@dataclass(frozen=True)
class MLXQwen3Cache:
    """Keys/values plus per-row physical validity and logical RoPE positions."""
    key_values: tuple
    attention_mask: Any
    position_ids: Any

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
        embedding, head = model.model.embed_tokens.weight, model.lm_head.weight
        config["tie_word_embeddings"] = (
            embedding.data_ptr() == head.data_ptr()
            and embedding.shape == head.shape and embedding.stride() == head.stride()
        )
        state = model.state_dict()
        runtime = cls._from_hf_weights(config, {k: _array(v, dtype) for k, v in state.items()
                                               if k != "lm_head.weight" or not config.get("tie_word_embeddings")})
        # Config values can lag behind deliberate live-module interventions.
        runtime.config["terminal_norm"] = _norm_spec(model.model.norm)
        for layer, source_layer in zip(runtime.layers, model.model.layers):
            layer["attention"]["input_norm"] = _norm_spec(source_layer.input_layernorm)
            layer["mlp"]["input_norm"] = _norm_spec(source_layer.post_attention_layernorm)
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
        files = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
                 if index.exists() else ["model.safetensors"])
        weights = {}
        for name in files:
            weights.update(mx.load(str(directory / name)))
        if dtype is not None:
            weights = {k: v.astype(dtype) for k, v in weights.items()}
        return cls._from_hf_weights(config, weights)

    @classmethod
    def _from_hf_weights(cls, source, weights):
        if source.get("model_type") != "qwen3" or source.get("hidden_act", "silu") != "silu":
            raise ValueError("Only dense Qwen3 with SiLU is supported")
        if source.get("use_sliding_window") or any(x != "full_attention" for x in source.get("layer_types", []) or []):
            raise ValueError("This loader supports global Qwen3 attention only")
        rope = source.get("rope_parameters") or source.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) not in ("default", "linear"):
            raise ValueError("Only default and linear RoPE are supported")
        factor = float(rope.get("factor", 1.0))
        heads = source["num_attention_heads"]
        kvheads = source["num_key_value_heads"]
        dim = source.get("head_dim", source["hidden_size"] // heads)
        if heads % kvheads or dim % 2 or factor <= 0:
            raise ValueError("Invalid GQA or rotary dimensions")
        norm = {"kind": "rms_norm", "epsilon": source["rms_norm_eps"],
                "scale_parameterization": "direct", "cast_before_scale": True}
        p = {"lexical.weight": weights["model.embed_tokens.weight"],
             "terminal.weight": weights["model.norm.weight"]}
        if not source.get("tie_word_embeddings", False):
            p["lexical.head_weight"] = weights["lm_head.weight"]
        layers = []
        for i in range(source["num_hidden_layers"]):
            s = f"model.layers.{i}."
            a, m = f"attention.{i}.", f"mlp.{i}."
            p[a + "input_norm_weight"] = weights[s + "input_layernorm.weight"]
            p[m + "input_norm_weight"] = weights[s + "post_attention_layernorm.weight"]
            for name in ("q", "k", "v", "o"):
                target = "output_decoder_weight" if name == "o" else name + "_weight"
                p[a + target] = weights[s + f"self_attn.{name}_proj.weight"]
                bias = weights.get(s + f"self_attn.{name}_proj.bias")
                if bias is not None:
                    p[a + name + "_bias"] = bias
            for name in ("q", "k"):
                p[a + name + "_norm_weight"] = weights[s + f"self_attn.{name}_norm.weight"]
            for name in ("gate", "up", "down"):
                p[m + name + "_weight"] = weights[s + f"mlp.{name}_proj.weight"]
            layers.append({"attention": {
                "input_norm": norm, "output_norm": {"kind": "identity"}, "qk_norm": norm,
                "query_heads": heads, "key_value_heads": kvheads, "head_dimension": dim,
                "query_scale": dim ** -0.5, "theta": rope.get("rope_theta", source.get("rope_theta", 1000000.0)),
                "rope_factor": factor, "window_size": None, "logit_softcap": None},
                "mlp": {"input_norm": norm, "output_norm": {"kind": "identity"}, "temperature": 1.0}})
        return cls(p, {"format_version": 1, "layers": layers, "terminal_norm": norm,
                       "embed_scale": 1.0, "final_logit_softcap": None, "source": "hf_qwen3"})

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
                 return_logits=True, cache=None):
        mx = _core()
        p = self.parameters if parameters is None else parameters
        ids = mx.array(input_ids)
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError("input_ids must be a nonempty integer [batch, length] array")
        lexical = p.get("lexical.weight", p.get("lexical.codebook"))
        if not bool(mx.all((ids >= 0) & (ids < lexical.shape[0])).item()):
            raise ValueError("input_ids must lie within the lexical vocabulary")
        b, t = ids.shape
        valid = mx.ones((b, t), dtype=mx.bool_) if attention_mask is None else mx.array(attention_mask).astype(mx.bool_)
        if valid.shape != ids.shape:
            raise ValueError("attention_mask must describe only the current input tokens")
        if cache is not None and (cache.attention_mask.shape[0] != b or len(cache.key_values) != len(self.layers)):
            raise ValueError("cache batch/layer count mismatch")
        if position_ids is None:
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
        seen = set()
        capture_sites = None if capture is True else set(capture or ())

        def observe(name, value):
            if name in perturbations:
                if perturbations[name].shape != value.shape:
                    raise ValueError(f"Perturbation shape mismatch at {name}")
                value = value + perturbations[name]
                seen.add(name)
            if capture_sites is None or name in capture_sites:
                captures[name] = value
            return value

        def linear(x, key, bias=None):
            y = x @ p[key].T
            return y + p[bias] if bias in p else y

        def incoming(name, count, rank, dtype):
            cols = [[] for _ in range(count)]
            for edge in self.config.get("transports", []):
                if edge["target"] == name:
                    cols[edge["target_chart"]].append(messages[edge["source"]][..., edge["source_chart"], :] @ p[edge["weight"]])
            return mx.stack([sum(c, mx.zeros((b, t, rank), dtype=dtype)) for c in cols], axis=-2)

        if "lexical.weight" in p:
            x = p["lexical.weight"][ids]
        else:
            x = p["lexical.codebook"][ids] @ p["lexical.basis"].T
        x = observe("embedding", x * self.config.get("embed_scale", 1.0))
        for i, layer in enumerate(self.layers):
            a, m = f"attention.{i}", f"mlp.{i}"
            ac, mc = layer["attention"], layer["mlp"]
            state = observe(a + ".input", x)
            n = observe(a + ".normalized", _rms(state, p.get(a + ".input_norm_weight"), ac["input_norm"]))
            coords = n @ p[a + ".input_basis"] if a + ".input_basis" in p else n
            coords = mx.where(valid[..., None], coords, 0)
            coords = coords + incoming(a, 1, coords.shape[-1], coords.dtype)[..., 0, :]
            qh, kh, d = ac["query_heads"], ac["key_value_heads"], ac["head_dimension"]
            qkv = []
            for name, heads in (("q", qh), ("k", kh), ("v", kh)):
                y = linear(coords, a + f".{name}_weight", a + f".{name}_bias").reshape(b, t, heads, d).transpose(0, 2, 1, 3)
                if name != "v" and ac["qk_norm"] is not None:
                    y = _rms(y, p[a + f".{name}_norm_weight"], ac["qk_norm"])
                qkv.append(y)
            q, k, v = qkv
            inv = ac["theta"] ** (-mx.arange(0, d, 2, dtype=mx.float32) / d)
            freqs = positions.astype(mx.float32)[..., None] * inv / ac["rope_factor"]
            angles = mx.concatenate((freqs, freqs), axis=-1)[:, None, :, :]
            cos, sin = mx.cos(angles).astype(q.dtype), mx.sin(angles).astype(q.dtype)

            def rotary(y):
                return y * cos + mx.concatenate((-y[..., d//2:], y[..., :d//2]), axis=-1) * sin

            q, k = rotary(q), rotary(k)
            if cache:
                k = mx.concatenate((cache.key_values[i][0], k), axis=2)
                v = mx.concatenate((cache.key_values[i][1], v), axis=2)
            new_kv.append((k, v))
            kr, vr = mx.repeat(k, qh // kh, axis=1), mx.repeat(v, qh // kh, axis=1)
            scores = (q @ kr.transpose(0, 1, 3, 2)) * ac["query_scale"]
            if ac["logit_softcap"] is not None:
                cap = ac["logit_softcap"]
                scores = mx.tanh(scores / cap) * cap
            offset = cache.length if cache else 0
            physical_causal = mx.arange(key_valid.shape[1])[None, :] <= (mx.arange(t) + offset)[:, None]
            allowed = valid[:, :, None] & key_valid[:, None, :] & physical_causal[None, :, :]
            if ac["window_size"] is not None:
                allowed = allowed & ((positions[:, :, None] - key_positions[:, None, :]) < ac["window_size"])
            scores = mx.where(allowed[:, None, :, :], scores.astype(mx.float32), -3.4028234663852886e38)
            probabilities = mx.softmax(scores, axis=-1).astype(q.dtype)
            context = observe(a + ".context", (probabilities @ vr).transpose(0, 2, 1, 3).reshape(b, t, qh*d))
            outcoords = linear(context, a + ".output_encoder_weight") if a + ".output_encoder_weight" in p else context
            raw = observe(a + ".output", linear(outcoords, a + ".output_decoder_weight", a + ".o_bias"))
            delta = observe(a + ".delta", _rms(raw, p.get(a + ".output_norm_weight"), ac["output_norm"]))
            x = state + mx.where(valid[..., None], delta, 0)
            messages[a] = mx.where(valid[..., None], outcoords, 0)[..., None, :]
            state = observe(m + ".input", x)
            n = observe(m + ".normalized", _rms(state, p.get(m + ".input_norm_weight"), mc["input_norm"]))
            if m + ".chart_centers" in p:
                centers = p[m + ".chart_centers"]
                displacement = n[..., None, :] - centers
                scales = p[m + ".chart_distance_scales"]
                memberships = mx.softmax(-0.5 * mx.sum(displacement ** 2, axis=-1) / (scales ** 2 * mc.get("temperature", 1.0)), axis=-1)
                coordinates = mx.einsum("btch,chr->btcr", displacement, p[m + ".chart_bases"])
                aug = mx.concatenate((mx.ones_like(coordinates[..., :1]), coordinates), axis=-1)
                enc = p[m + ".local_generator_encoders"]
                local = mx.einsum("btcr,crg->btcg", aug, enc)
                packet = memberships[..., None] * (local + incoming(m, centers.shape[0], enc.shape[-1], n.dtype))
                modal = linear(n, m + ".affine_weight", m + ".affine_bias") + mx.einsum("btcg,cgm->btm", packet, p[m + ".chart_mode_decoders"])
                raw = linear(modal, m + ".mode_decoder_weight") if m + ".mode_decoder_weight" in p else modal
            else:
                gate = linear(n, m + ".gate_weight")
                raw = linear((gate * mx.sigmoid(gate)) * linear(n, m + ".up_weight"), m + ".down_weight")
                packet = mx.zeros((b, t, mc.get("emitted_chart_count", 1), mc.get("emitted_generator_rank", n.shape[-1])), dtype=n.dtype)
            raw = observe(m + ".output", raw)
            delta = observe(m + ".delta", _rms(raw, p.get(m + ".output_norm_weight"), mc["output_norm"]))
            x = state + mx.where(valid[..., None], delta, 0)
            messages[m] = mx.where(valid[..., None, None], packet, 0)
        x = observe("terminal", x)
        normalized = observe("terminal.normalized", _rms(x, p.get("terminal.weight"), self.config["terminal_norm"]))
        logits = None
        if return_logits:
            if "lexical.head_weight" in p:
                logits = normalized @ p["lexical.head_weight"].T
            elif "lexical.weight" in p:
                logits = normalized @ p["lexical.weight"].T
            else:
                logits = (normalized @ p["lexical.basis"]) @ p["lexical.codebook"].T
            if self.config.get("final_logit_softcap") is not None:
                cap = self.config["final_logit_softcap"]
                logits = mx.tanh(logits / cap) * cap
        if set(perturbations) - seen:
            raise ValueError(f"Unknown perturbation sites: {sorted(set(perturbations) - seen)}")
        if capture_sites and capture_sites - set(captures):
            raise ValueError(f"Unknown capture sites: {sorted(capture_sites - set(captures))}")
        return MLXQwen3Output(logits, x, captures, MLXQwen3Cache(tuple(new_kv), key_valid, key_positions))

    def prefill(self, input_ids, **kwargs):
        return self(input_ids, **kwargs)

    def decode(self, input_ids, cache, **kwargs):
        return self(input_ids, cache=cache, **kwargs)
