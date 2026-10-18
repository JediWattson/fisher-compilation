"""Interactive TUI, X-Ray Debugger, and Equalizer Harness for Pullback-EWC Gemma 3 270M.

Usage:
  # Launch interactive TUI with /xray and /mute debugging commands:
  .venv/bin/python -m fisher_graph.gemma3_ewc_interactive_harness

  # Run a single prompt with the full X-Ray causal attribution panel:
  .venv/bin/python -m fisher_graph.gemma3_ewc_interactive_harness \
    --xray \
    --prompt "The capital of France is Paris, and the capital of Japan is"

  # Run a prompt with a live counterfactual route or layer mute:
  .venv/bin/python -m fisher_graph.gemma3_ewc_interactive_harness \
    --xray \
    --mute-route "Japan" \
    --prompt "The capital of France is Paris, and the capital of Japan is"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, SequenceContext
from .gemma3_attention_edge_block_ewc import (
    select_block_ewc_attention_route_mask,
    solve_block_ewc_sparse_attention_weights,
)
from .gemma3_cross_layer_chart_transport_model_experiment import (
    _ordered_example_ids,
    _tokenize_keyed_batches,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_joint_cross_block_compiler import (
    compile_gemma3_joint_cross_block_candidate,
    compile_gemma3_joint_exact_control,
)
from .gemma3_joint_cross_block_model_fit import (
    fit_ewc_joint_cross_block_whole_model,
    fit_pullback_ewc_shared_tied_lexical_factor,
)
from .gemma3_joint_cross_block_whole_model_experiment import (
    _evaluate_joint_executor,
)
from .gemma3_surgical_trajectory_refit import surgical_trajectory_refit
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    DEFAULT_PANEL_PATH,
    DEFAULT_REVISION,
    _load_prompt_split,
)
from .gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
    _source_lexical_modules,
)
from .gemma3_parts_first_attention_edge_fidelity import (
    _validate_generator,
    attention_route_tensors,
    operator_from_routed_probabilities,
)


CACHE_DIR = Path(".local-runs/google--gemma-3-270m")
VALID_MODES = (
    "hybrid-ewc",
    "block-ewc-k8",
    "39-event-fused640",
    "39-event",
)
PRESET_PROMPTS: tuple[str, ...] = (
    "Complete the category fact: A robin is a kind of",
    "Complete the category fact: Copper is a kind of",
    "Complete the category fact: A violin is a kind of",
    "Complete the work-location fact: A baker works in a",
    "Complete the tool-use fact: A telescope is used to",
    "Complete the opposite-word cue: The opposite of ancient is",
)

# ANSI styling
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"


@dataclass
class ModalEqualizerState:
    """Active counterfactual mutes/interventions for interactive debugging."""

    muted_attn_layers: set[int] = field(default_factory=set)
    muted_attn_heads: set[tuple[int, int]] = field(default_factory=set)
    muted_mlp_layers: set[int] = field(default_factory=set)
    muted_mlp_charts: set[tuple[int, int]] = field(default_factory=set)
    muted_route_substrings: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.muted_attn_layers
            or self.muted_attn_heads
            or self.muted_mlp_layers
            or self.muted_mlp_charts
            or self.muted_route_substrings
        )

    def clear(self) -> None:
        self.muted_attn_layers.clear()
        self.muted_attn_heads.clear()
        self.muted_mlp_layers.clear()
        self.muted_mlp_charts.clear()
        self.muted_route_substrings.clear()

    def summary(self) -> str:
        if self.is_empty():
            return "none (all clusters, charts, and routes active)"
        parts: list[str] = []
        if self.muted_attn_layers:
            parts.append(f"attn_layers={sorted(self.muted_attn_layers)}")
        if self.muted_attn_heads:
            parts.append(f"attn_heads={sorted(self.muted_attn_heads)}")
        if self.muted_mlp_layers:
            parts.append(f"mlp_layers={sorted(self.muted_mlp_layers)}")
        if self.muted_mlp_charts:
            parts.append(f"mlp_charts={sorted(self.muted_mlp_charts)}")
        if self.muted_route_substrings:
            parts.append(f"route_tokens={self.muted_route_substrings}")
        return ", ".join(parts)


class HybridPullbackEWCExecutor(nn.Module):
    """Whole-model 39-event executor combining Pullback-EWC Lex512 + All-18-Layer Block-EWC Sparse Attention."""

    def __init__(
        self,
        exact_executor: nn.Module,
        lexical: SharedTiedLexicalFactor,
        *,
        k: int = 16,
        min_per_head: int = 2,
        ewc_prior_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.lexical = lexical
        self.mutations = exact_executor.mutations
        self.terminal_norm = exact_executor.terminal_norm
        self.final_logit_softcap = getattr(exact_executor, "final_logit_softcap", None)
        self.k = k
        self.min_per_head = min_per_head
        self.ewc_prior_weight = ewc_prior_weight
        self.eval().requires_grad_(False)

    def _project_state_to_logits(self, state: Tensor) -> Tensor:
        norm_out = self.terminal_norm(state)
        logits = self.lexical.egress(norm_out)
        if self.final_logit_softcap is not None:
            cap = float(self.final_logit_softcap)
            logits = torch.tanh(logits / cap) * cap
        return logits

    def execute_logits(
        self,
        input_ids: Tensor,
        sequence: SequenceContext,
        *,
        equalizer: ModalEqualizerState | None = None,
        token_strings: list[str] | None = None,
    ) -> Tensor:
        state = self.lexical.ingress(input_ids)
        valid = sequence.query_valid_mask.unsqueeze(-1)
        muted_key_positions: set[int] = set()
        if equalizer is not None and equalizer.muted_route_substrings and token_strings:
            for pos, tok_s in enumerate(token_strings):
                for sub in equalizer.muted_route_substrings:
                    if sub.lower() in tok_s.lower():
                        muted_key_positions.add(pos)

        for mutation_index, mutation in enumerate(self.mutations):
            layer = mutation_index // 2
            if mutation_index % 2 == 0:
                if equalizer is not None and layer in equalizer.muted_attn_layers:
                    continue
                generator = getattr(mutation, "compiled_operator", None)
                _validate_generator(generator)
                normalized, _ = generator.encode_input(state, sequence)
                routes = attention_route_tensors(generator, normalized, sequence)
                mask = select_block_ewc_attention_route_mask(
                    routes.probabilities,
                    routes.projected_values,
                    routes.allowed_mask,
                    k=self.k,
                    min_per_head=self.min_per_head,
                )
                routed = solve_block_ewc_sparse_attention_weights(
                    routes.probabilities,
                    routes.projected_values,
                    routes.allowed_mask,
                    mask,
                    ewc_prior_weight=self.ewc_prior_weight,
                )
                if equalizer is not None:
                    for h in range(routed.shape[1]):
                        if (layer, h) in equalizer.muted_attn_heads:
                            routed[:, h, :, :] = 0.0
                    for pos in muted_key_positions:
                        if pos < routed.shape[-1]:
                            # Mute routes from `pos` into later queries (keep self-route at query == pos so key 0 doesn't zero out)
                            for q in range(pos + 1, routed.shape[-2]):
                                routed[:, :, q, pos] = 0.0
                routed = torch.where(
                    sequence.query_valid_mask[:, None, :, None],
                    routed,
                    routes.execution_probabilities,
                )
                _, delta = operator_from_routed_probabilities(
                    generator, routed, routes.repeated_values, sequence
                )
            else:
                if equalizer is not None and layer in equalizer.muted_mlp_layers:
                    continue
                incoming = torch.zeros(
                    *state.shape[:2],
                    mutation.incoming_chart_count,
                    mutation.incoming_generator_rank,
                    dtype=state.dtype,
                    device=state.device,
                )
                delta = mutation.execute_with_incoming(state, sequence, incoming).delta
            state = torch.where(valid, state + delta.masked_fill(~valid, 0), state)
        return self._project_state_to_logits(state)


def _cache_path(mode: str) -> Path:
    return CACHE_DIR / f"ewc-interactive-executor-{mode}-v1.pt"


def _lexical_cache_path() -> Path:
    return CACHE_DIR / "ewc-interactive-lex512-v1.pt"


def _chart_atlas_cache_path() -> Path:
    return CACHE_DIR / "ewc-interactive-chart-atlas-v1.pt"


def _load_or_fit_lex512(
    adapter: Gemma3CausalLMAdapter,
    exact_executor: nn.Module,
    tokenizer: object,
    *,
    rebuild: bool = False,
) -> SharedTiedLexicalFactor:
    lex_path = _lexical_cache_path()
    if lex_path.exists() and not rebuild:
        return torch.load(lex_path, map_location="cpu", weights_only=False)
    device = next(adapter.module.parameters()).device
    fit_prompts, _, _ = _load_prompt_split(Path(DEFAULT_PANEL_PATH))
    fit_ids = _ordered_example_ids(Path(DEFAULT_PANEL_PATH), fit_prompts)
    fit_keyed = _tokenize_keyed_batches(
        tokenizer, fit_prompts, fit_ids, batch_size=4, device=device
    )
    fit_batches = tuple(item.model_inputs for item in fit_keyed)
    _, _, source_weight, embed_scale = _source_lexical_modules(adapter)
    lexical = fit_pullback_ewc_shared_tied_lexical_factor(
        adapter=adapter,
        exact_executor=exact_executor,
        batches=fit_batches,
        source_weight=source_weight,
        lexical_rank=512,
        embed_scale=embed_scale,
    )
    lex_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lexical, lex_path)
    return lexical


def load_or_compile_executor(
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
    *,
    mode: str = "hybrid-ewc",
    rebuild: bool = False,
) -> object:
    """Load or compile the requested Pullback-EWC executor mode."""
    device = next(adapter.module.parameters()).device
    if mode in {"hybrid-ewc", "block-ewc-k8"}:
        exact = compile_gemma3_joint_exact_control(adapter, dtype=torch.float32, device=device)
        if mode == "block-ewc-k8":
            return exact.executor
        lexical = _load_or_fit_lex512(
            adapter, exact.executor, tokenizer, rebuild=rebuild
        )
        return HybridPullbackEWCExecutor(exact.executor, lexical, k=16, min_per_head=2)

    fuse_mlp_modes = mode == "39-event-fused640"
    cache_file = _cache_path(mode)
    if cache_file.exists() and not rebuild:
        print(f"{DIM}[harness] loading cached {mode} executor from {cache_file}...{RESET}", flush=True)
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    print(
        f"{YELLOW}[harness] compiling 39-event Pullback-EWC whole-model executor ({mode})...{RESET}",
        flush=True,
    )
    t0 = time.perf_counter()
    fit_prompts, _, _ = _load_prompt_split(Path(DEFAULT_PANEL_PATH))
    fit_ids = _ordered_example_ids(Path(DEFAULT_PANEL_PATH), fit_prompts)
    fit_keyed = _tokenize_keyed_batches(
        tokenizer,
        fit_prompts,
        fit_ids,
        batch_size=4,
        device=device,
    )
    fit_batches = tuple(item.model_inputs for item in fit_keyed)
    fit_row_keys = tuple(item.row_keys for item in fit_keyed)

    exact = compile_gemma3_joint_exact_control(adapter, dtype=torch.float32, device=device)
    exact_attention = tuple(
        m.compiled_operator
        for m in exact.executor.mutations
        if m.event_spec.event_kind == "attention_mutation"
    )
    lexical = _load_or_fit_lex512(
        adapter, exact.executor, tokenizer, rebuild=rebuild
    )
    reduced_attention, fit = fit_ewc_joint_cross_block_whole_model(
        adapter=adapter,
        exact_executor=exact.executor,
        lexical=lexical,
        exact_attention=exact_attention,
        batches=fit_batches,
        batch_row_keys=fit_row_keys,
        model_fingerprint=adapter.model_fingerprint(),
        attention_input_rank=256,
        attention_output_rank=64,
        generator_rank=4,
        maximum_incoming_attention_sources_per_target_chart=1,
        fuse_mlp_modes=fuse_mlp_modes,
        progress=lambda msg: print(f"{DIM}  [compile] {msg}{RESET}", flush=True),
    )
    compilation = compile_gemma3_joint_cross_block_candidate(
        adapter,
        lexical=lexical,
        attention_generators=reduced_attention,
        mlp_chart_generators=fit.mlp_generators,
        transports=fit.transports,
        dtype=torch.float32,
        device=device,
    )
    executor = compilation.executor
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(executor, cache_file)
    dt = time.perf_counter() - t0
    print(f"{GREEN}[harness] compiled & cached to {cache_file} in {dt:.1f}s{RESET}", flush=True)
    return executor


def _load_or_build_chart_atlas(
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
) -> tuple[object, dict[int, dict[int, list[str]]]]:
    """Load the 39-event chart executor and build emergent token-trigger profiles for each (layer, chart)."""
    chart_exec = load_or_compile_executor(
        adapter, tokenizer, mode="39-event-fused640", rebuild=False
    )
    atlas_path = _chart_atlas_cache_path()
    if atlas_path.exists():
        atlas = torch.load(atlas_path, map_location="cpu", weights_only=False)
        return chart_exec, atlas

    fit_prompts, eval_prompts, _ = _load_prompt_split(Path(DEFAULT_PANEL_PATH))
    atlas: dict[int, dict[int, list[str]]] = {
        b: {c: [] for c in range(8)} for b in range(18)
    }
    with torch.no_grad():
        for p in (*fit_prompts[:18], *eval_prompts[:18]):
            enc = tokenizer(p, return_tensors="pt")
            inp = enc["input_ids"]
            seq = adapter.prepare_sequence(
                {"input_ids": inp, "attention_mask": torch.ones_like(inp)}
            )
            state = chart_exec.lexical.ingress(inp)
            for m_idx, mut in enumerate(chart_exec.mutations):
                layer = m_idx // 2
                if m_idx % 2 == 1:
                    norm_in = mut.input_norm(state)
                    memb, _, _ = mut.chart_generator.encode_local_generator_latents(norm_in)
                    dom = memb[0].argmax(dim=-1).tolist()
                    for pos, c_id in enumerate(dom):
                        tok_s = tokenizer.decode([int(inp[0, pos].item())])
                        if tok_s != "<bos>" and tok_s not in atlas[layer][c_id] and len(atlas[layer][c_id]) < 6:
                            atlas[layer][c_id].append(tok_s)
                inc = torch.zeros(
                    *state.shape[:2],
                    mut.incoming_chart_count,
                    mut.incoming_generator_rank,
                )
                state = state + mut.execute_with_incoming(state, seq, inc).delta
    torch.save(atlas, atlas_path)
    return chart_exec, atlas


def _execute_mode_logits(
    executor: object,
    input_ids: Tensor,
    sequence: SequenceContext,
    mode: str,
    equalizer: ModalEqualizerState | None = None,
    token_strings: list[str] | None = None,
) -> Tensor:
    if isinstance(executor, HybridPullbackEWCExecutor):
        return executor.execute_logits(
            input_ids,
            sequence,
            equalizer=equalizer,
            token_strings=token_strings,
        )
    if mode == "block-ewc-k8":
        return replay_exact_with_block_ewc_attention(
            executor,
            input_ids,
            sequence,
            k=8,
            min_per_head=1,
            ewc_prior_weight=0.05,
        ).logits
    if equalizer is not None and not equalizer.is_empty():
        # Custom replay on 39-event chart executor with mutes applied
        state = executor.lexical.ingress(input_ids)
        valid = sequence.query_valid_mask.unsqueeze(-1)
        for m_idx, mut in enumerate(executor.mutations):
            layer = m_idx // 2
            if m_idx % 2 == 0 and layer in equalizer.muted_attn_layers:
                continue
            if m_idx % 2 == 1 and layer in equalizer.muted_mlp_layers:
                continue
            inc = torch.zeros(
                *state.shape[:2],
                mut.incoming_chart_count,
                mut.incoming_generator_rank,
                dtype=state.dtype,
                device=state.device,
            )
            if m_idx % 2 == 1 and any(l_ord == layer for l_ord, _ in equalizer.muted_mlp_charts):
                norm_in = mut.input_norm(state)
                memb, coords, latents = mut.chart_generator.encode_local_generator_latents(norm_in)
                for l_ord, c_id in equalizer.muted_mlp_charts:
                    if l_ord == layer and 0 <= c_id < memb.shape[-1]:
                        memb = memb.clone()
                        memb[..., c_id] = 0.0
                emitted = memb.unsqueeze(-1) * latents
                modal_states = mut.chart_generator.affine_generators(norm_in) + torch.einsum(
                    "...cg,cgk->...k",
                    emitted,
                    mut.chart_generator.chart_mode_decoders,
                )
                out_m = mut.chart_generator.mode_decoder(modal_states)
                delta = mut.output_norm(out_m)
            else:
                delta = mut.execute_with_incoming(state, sequence, inc).delta
            state = torch.where(valid, state + delta.masked_fill(~valid, 0), state)
        norm_out = executor.terminal_norm(state)
        logits = executor.lexical.egress(norm_out)
        cap = getattr(executor, "final_logit_softcap", None)
        if cap is not None:
            logits = torch.tanh(logits / float(cap)) * float(cap)
        return logits
    return executor.execute(input_ids, sequence).logits


def run_xray_analysis(
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
    executor: object,
    prompt: str,
    *,
    mode: str = "hybrid-ewc",
    equalizer: ModalEqualizerState | None = None,
) -> None:
    """Print a 3-part causal X-Ray breakdown for ``prompt``:
    1) Active MLP Charts per token + emergent semantic chart labels
    2) Top Block-EWC Attention Routes into the final query token & their counterfactual word push
    3) Top Layer / Component Logit-Lens contributions to the winning next token
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids: Tensor = encoded["input_ids"]
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    seq = adapter.prepare_sequence(batch)
    tok_strs = [tokenizer.decode([int(t)]) for t in input_ids[0].tolist()]

    # Ensure we have HybridPullbackEWCExecutor for exact route counterfactuals + chart atlas for chart labels
    hybrid_exec = (
        executor
        if isinstance(executor, HybridPullbackEWCExecutor)
        else load_or_compile_executor(adapter, tokenizer, mode="hybrid-ewc")
    )
    chart_exec, chart_atlas = _load_or_build_chart_atlas(adapter, tokenizer)

    with torch.no_grad():
        base_logits = hybrid_exec.execute_logits(
            input_ids, seq, equalizer=equalizer, token_strings=tok_strs
        )[0, -1]
        probs = F.softmax(base_logits.double(), dim=-1)
        win_id = int(base_logits.argmax().item())
        win_str = repr(tokenizer.decode([win_id]))
        win_prob = float(probs[win_id].item()) * 100.0

        print(f"\n{BOLD}{YELLOW}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓{RESET}")
        print(f"{BOLD}{YELLOW}┃  X-RAY MODAL DEBUGGER — Target Prediction: {GREEN}{win_str} ({win_prob:.1f}%){YELLOW}                 ┃{RESET}")
        if equalizer is not None and not equalizer.is_empty():
            print(f"{BOLD}{YELLOW}┃  Active Mutes: {RED}{equalizer.summary()}{RESET}")
        print(f"{BOLD}{YELLOW}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛{RESET}")

        # ── Part 1: Token -> MLP Chart Routing Map (Layers 10 & 17) ───────────────
        print(f"\n  {BOLD}{CYAN}[1] Token → Fisher MLP Chart Routing & Emergent Chart Identity (Layer 17):{RESET}")
        c_state = chart_exec.lexical.ingress(input_ids)
        l10_memb, l17_memb = None, None
        for m_idx, mut in enumerate(chart_exec.mutations):
            layer = m_idx // 2
            if m_idx % 2 == 1 and layer in (10, 17):
                norm_in = mut.input_norm(c_state)
                memb, _, _ = mut.chart_generator.encode_local_generator_latents(norm_in)
                if layer == 10:
                    l10_memb = memb[0]
                else:
                    l17_memb = memb[0]
            inc = torch.zeros(
                *c_state.shape[:2],
                mut.incoming_chart_count,
                mut.incoming_generator_rank,
            )
            c_state = c_state + mut.execute_with_incoming(c_state, seq, inc).delta

        if l17_memb is not None and l10_memb is not None:
            for pos, tok_s in enumerate(tok_strs):
                if tok_s == "<bos>":
                    continue
                c17 = int(l17_memb[pos].argmax().item())
                p17 = float(l17_memb[pos, c17].item()) * 100.0
                c10 = int(l10_memb[pos].argmax().item())
                triggers17 = chart_atlas[17].get(c17, [])
                trig_str = ", ".join(repr(x) for x in triggers17[:4])
                print(
                    f"    pos {pos:2d} {BOLD}{repr(tok_s):14s}{RESET} → "
                    f"L10:{GREEN}Chart {c10}{RESET} | L17:{GREEN}Chart {c17}{RESET} ({p17:4.0f}%) "
                    f"{DIM}[Chart {c17} vocabulary motif: {trig_str}]{RESET}"
                )

        # ── Part 2: Top Block-EWC Attention Routes into Final Token ───────────────
        print(f"\n  {BOLD}{CYAN}[2] Top Block-EWC Sparse Attention Routes into Final Token ({repr(tok_strs[-1])}):{RESET}")
        state = hybrid_exec.lexical.ingress(input_ids)
        valid = seq.query_valid_mask.unsqueeze(-1)
        route_records: list[tuple[float, int, int, int, float]] = []
        states_before_mut: list[Tensor] = []
        deltas_per_mut: list[Tensor] = []

        for m_idx, mut in enumerate(hybrid_exec.mutations):
            states_before_mut.append(state.detach().clone())
            layer = m_idx // 2
            if m_idx % 2 == 0:
                gen = getattr(mut, "compiled_operator", None)
                norm_in, _ = gen.encode_input(state, seq)
                routes = attention_route_tensors(gen, norm_in, seq)
                mask = select_block_ewc_attention_route_mask(
                    routes.probabilities,
                    routes.projected_values,
                    routes.allowed_mask,
                    k=hybrid_exec.k,
                    min_per_head=hybrid_exec.min_per_head,
                )
                routed = solve_block_ewc_sparse_attention_weights(
                    routes.probabilities,
                    routes.projected_values,
                    routes.allowed_mask,
                    mask,
                    ewc_prior_weight=hybrid_exec.ewc_prior_weight,
                )
                routed = torch.where(
                    seq.query_valid_mask[:, None, :, None],
                    routed,
                    routes.execution_probabilities,
                )
                _, delta = operator_from_routed_probabilities(
                    gen, routed, routes.repeated_values, seq
                )
                last_q = input_ids.shape[1] - 1
                for h in range(routed.shape[1]):
                    for k_pos in range(1, last_q):
                        w_hk = float(routed[0, h, last_q, k_pos].item())
                        if w_hk > 0.04:
                            # Score route importance by w_hk * ||projected_value||
                            v_norm = float(routes.projected_values[0, h, k_pos].norm().item())
                            route_records.append((w_hk * v_norm, layer, h, k_pos, w_hk))
            else:
                inc = torch.zeros(
                    *state.shape[:2],
                    mut.incoming_chart_count,
                    mut.incoming_generator_rank,
                )
                delta = mut.execute_with_incoming(state, seq, inc).delta
            deltas_per_mut.append(delta.detach().clone())
            state = torch.where(valid, state + delta.masked_fill(~valid, 0), state)

        final_state = state.detach().clone()
        route_records.sort(reverse=True)
        for _, layer, h, k_pos, w_hk in route_records[:6]:
            src_tok = repr(tok_strs[k_pos])
            dst_tok = repr(tok_strs[-1])
            print(
                f"    Layer {layer:2d} Head {h} | {BOLD}{GREEN}{src_tok:12s} ──(w={w_hk:.2f})──► {dst_tok}{RESET} "
                f"{DIM}(mute with: /mute route {tok_strs[k_pos].strip()}){RESET}"
            )

        # ── Part 3: Direct Logit-Lens Push of Top Layers on Winning Token ─────────
        print(f"\n  {BOLD}{CYAN}[3] Top Causal Layer Contributions to {GREEN}{win_str}{CYAN} (Direct Residual Knock-Out):{RESET}")
        layer_impacts: list[tuple[float, str, int, Tensor]] = []
        for m_idx, delta in enumerate(deltas_per_mut):
            layer = m_idx // 2
            kind = "Attention" if m_idx % 2 == 0 else "MLP      "
            # Counterfactual state without this single mutation's delta at the final position
            ablated_state = final_state - delta
            ablated_logits = hybrid_exec._project_state_to_logits(ablated_state)[0, -1]
            diff_logits = base_logits - ablated_logits
            win_push = float(diff_logits[win_id].item())
            layer_impacts.append((abs(win_push), f"Layer {layer:2d} {kind}", layer, diff_logits))

        layer_impacts.sort(reverse=True)
        for _, label, _, diff_logits in layer_impacts[:6]:
            win_push = float(diff_logits[win_id].item())
            top_up_ids = diff_logits.topk(3).indices.tolist()
            top_dn_ids = (-diff_logits).topk(2).indices.tolist()
            up_str = ", ".join(
                f"{repr(tokenizer.decode([int(i)]))} ({float(diff_logits[i]):+.1f})"
                for i in top_up_ids
            )
            dn_str = ", ".join(
                f"{repr(tokenizer.decode([int(i)]))} ({float(diff_logits[i]):+.1f})"
                for i in top_dn_ids
            )
            sign_color = GREEN if win_push >= 0 else RED
            print(
                f"    {BOLD}{label}{RESET} | push on {win_str}: {sign_color}{win_push:+.2f} logits{RESET}\n"
                f"      {DIM}├─ Pushes UP ↑ : {up_str}{RESET}\n"
                f"      {DIM}└─ Pushes DN ↓ : {dn_str}{RESET}"
            )


def _sample_next_token(logits_1d: Tensor, temperature: float) -> int:
    if temperature <= 0.0:
        return int(logits_1d.argmax(dim=-1).item())
    probs = F.softmax(logits_1d.double() / float(temperature), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _top_k_summary(tokenizer: object, logits_1d: Tensor, k: int = 5) -> str:
    probs = F.softmax(logits_1d.double(), dim=-1)
    vals, idxs = torch.topk(probs, k=k)
    parts: list[str] = []
    for p, idx in zip(vals.tolist(), idxs.tolist()):
        tok = repr(tokenizer.decode([int(idx)]))
        parts.append(f"{tok}: {p * 100:4.1f}%")
    return " | ".join(parts)


def run_prompt_comparison(
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
    executor: object,
    prompt: str,
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    compare_native: bool = True,
    mode_label: str = "hybrid-ewc",
    equalizer: ModalEqualizerState | None = None,
    xray: bool = False,
) -> None:
    """Run autoregressive generation on ``prompt`` and print a rich comparison panel."""
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids: Tensor = encoded["input_ids"]
    eos_id = getattr(tokenizer, "eos_token_id", None)

    print(f"\n{BOLD}{CYAN}┌──────────────────────────────────────────────────────────────────────────────┐{RESET}")
    print(f"{BOLD}{CYAN}│ Prompt:{RESET} {BOLD}{prompt}{RESET}")
    if equalizer is not None and not equalizer.is_empty():
        print(f"{BOLD}{CYAN}│ Mutes :{RESET} {RED}{equalizer.summary()}{RESET}")
    print(f"{BOLD}{CYAN}└──────────────────────────────────────────────────────────────────────────────┘{RESET}")

    # 1. Compiled EWC Generation
    cand_ids = prompt_ids.clone()
    cand_new_tokens: list[int] = []
    first_cand_logits: Tensor | None = None
    t0 = time.perf_counter()
    print(f"  {BOLD}{GREEN}[Compiled EWC ({mode_label})]{RESET} ", end="", flush=True)
    print(f"{DIM}{prompt}{RESET}", end="", flush=True)
    with torch.no_grad():
        for step in range(max_new_tokens):
            batch = {
                "input_ids": cand_ids,
                "attention_mask": torch.ones_like(cand_ids),
            }
            seq = adapter.prepare_sequence(batch)
            tok_strs = [tokenizer.decode([int(t)]) for t in cand_ids[0].tolist()]
            logits = _execute_mode_logits(
                executor,
                cand_ids,
                seq,
                mode_label,
                equalizer=equalizer,
                token_strings=tok_strs,
            )
            next_logits = logits[0, -1]
            if step == 0:
                first_cand_logits = next_logits.detach().clone()
            next_id = _sample_next_token(next_logits, temperature)
            cand_new_tokens.append(next_id)
            piece = tokenizer.decode([next_id])
            print(f"{BOLD}{GREEN}{piece}{RESET}", end="", flush=True)
            cand_ids = torch.cat(
                [cand_ids, torch.tensor([[next_id]], dtype=cand_ids.dtype)],
                dim=1,
            )
            if eos_id is not None and next_id == eos_id:
                break
    cand_dt = time.perf_counter() - t0
    print(f"  {DIM}({len(cand_new_tokens)} tok, {cand_dt:.2f}s){RESET}")

    if compare_native:
        nat_ids = prompt_ids.clone()
        nat_new_tokens: list[int] = []
        first_nat_logits: Tensor | None = None
        t1 = time.perf_counter()
        pad = " " * max(0, len(mode_label) - 4)
        print(f"  {BOLD}{MAGENTA}[Native Gemma 3 270M]{pad}{RESET} ", end="", flush=True)
        print(f"{DIM}{prompt}{RESET}", end="", flush=True)
        with torch.no_grad():
            for step in range(max_new_tokens):
                batch = {
                    "input_ids": nat_ids,
                    "attention_mask": torch.ones_like(nat_ids),
                }
                nat_out = adapter.forward(batch)
                next_logits = nat_out.logits[0, -1]
                if step == 0:
                    first_nat_logits = next_logits.detach().clone()
                next_id = _sample_next_token(next_logits, temperature)
                nat_new_tokens.append(next_id)
                piece = tokenizer.decode([next_id])
                print(f"{BOLD}{MAGENTA}{piece}{RESET}", end="", flush=True)
                nat_ids = torch.cat(
                    [nat_ids, torch.tensor([[next_id]], dtype=nat_ids.dtype)],
                    dim=1,
                )
                if eos_id is not None and next_id == eos_id:
                    break
        nat_dt = time.perf_counter() - t1
        print(f"  {DIM}({len(nat_new_tokens)} tok, {nat_dt:.2f}s){RESET}")

        if first_cand_logits is not None and first_nat_logits is not None:
            p_nat = F.log_softmax(first_nat_logits.double(), dim=-1)
            p_cand = F.log_softmax(first_cand_logits.double(), dim=-1)
            kl_first = float(
                F.kl_div(p_cand, p_nat, log_target=True, reduction="sum").item()
            )
            top1_match = int(first_nat_logits.argmax().item()) == int(
                first_cand_logits.argmax().item()
            )
            match_badge = f"{GREEN}MATCH{RESET}" if top1_match else f"{YELLOW}CLOSE/ALT{RESET}"
            print(f"  {DIM}├─ First-Token Top-1: {match_badge} {DIM}| Next-Token KL: {kl_first:.4f} nats{RESET}")
            print(f"  {DIM}├─ EWC Top-5:    {_top_k_summary(tokenizer, first_cand_logits)}{RESET}")
            print(f"  {DIM}└─ Native Top-5: {_top_k_summary(tokenizer, first_nat_logits)}{RESET}")
    elif first_cand_logits is not None:
        print(f"  {DIM}First-token Top-5 (EWC): {_top_k_summary(tokenizer, first_cand_logits)}{RESET}")

    if xray:
        run_xray_analysis(
            adapter,
            tokenizer,
            executor,
            prompt,
            mode=mode_label,
            equalizer=equalizer,
        )


def _print_banner(mode: str, xray: bool, equalizer: ModalEqualizerState) -> None:
    if mode == "hybrid-ewc":
        desc = "Pullback-EWC Lex512 (-33.2M params) + 18-Layer Block-EWC K=16 Sparse Attention"
    elif mode == "block-ewc-k8":
        desc = "All-18-Layer Block-EWC K=8 Sparse Attention (0% starved heads, 80% Top-1)"
    elif mode == "39-event-fused640":
        desc = "All-39-Event Fused-640 Surrogate (163.3M params, -39.1% params, -71.4% body MACs)"
    else:
        desc = "All-39-Event 2048-Mode Surrogate (196.6M params, -26.7% params, -38.3% body MACs)"

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   Fisher-Graph Gemma 3 270M — Pullback-EWC Interactive X-Ray & Equalizer     ║{RESET}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} Active Mode : {BOLD}{GREEN}{mode:62s}{RESET} {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} Description : {desc:62s} {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} X-Ray Mode  : {BOLD}{GREEN if xray else YELLOW}{'ON (prints chart/route/layer attribution on every prompt)' if xray else 'OFF (type /xray to toggle on)':62s}{RESET} {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} Active Mutes: {RED}{equalizer.summary():62s}{RESET} {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} Debugging & Equalizer Commands:                                              {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/xray{RESET}                      Toggle automatic X-Ray causal attribution      {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/mute route <word>{RESET}         Mute attention routes from <word> (e.g. Japan) {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/mute attn <layer> [head]{RESET}  Mute an attention layer (0..17) or head (0..3) {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/mute mlp <layer>{RESET}          Mute an MLP layer (0..17)                      {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/mute chart <layer> <id>{RESET}   Mute a specific MLP chart (0..7)               {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/optimize [lam a beta]{RESET}    Terminal-state EWC refit (default 0.3 1.0 10)  {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/unmute{RESET}                    Restore all muted routes, charts, and layers   {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/atlas [layer]{RESET}             Show emergent vocabulary motifs for charts 0..7{BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/preset{RESET} | {YELLOW}/mode <m>{RESET} | {YELLOW}/tokens <N>{RESET} | {YELLOW}/temp <T>{RESET} | {YELLOW}/compare{RESET} | {YELLOW}/quit{RESET}         {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{RESET}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive TUI / X-Ray Debugger for compiled Pullback-EWC Gemma 3 270M models."
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="hybrid-ewc",
        help="compiled EWC variant to run (default: hybrid-ewc)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="run a single prompt non-interactively and exit",
    )
    parser.add_argument(
        "--xray",
        action="store_true",
        help="enable the 3-part X-Ray causal attribution breakdown (charts, routes, and layer logit-push)",
    )
    parser.add_argument(
        "--mute-route",
        action="append",
        default=[],
        help="mute attention routes originating from tokens matching this substring (e.g. --mute-route Japan)",
    )
    parser.add_argument(
        "--mute-attn-layer",
        type=int,
        action="append",
        default=[],
        help="mute a specific attention layer (0..17)",
    )
    parser.add_argument(
        "--mute-mlp-layer",
        type=int,
        action="append",
        default=[],
        help="mute a specific MLP layer (0..17)",
    )
    parser.add_argument(
        "--presets",
        action="store_true",
        help="run the 6 preset prompts non-interactively and exit",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="maximum number of new tokens to generate per prompt (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature; 0.0 uses greedy argmax decoding (default: 0.0)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="disable side-by-side native Gemma comparison",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force recompilation instead of loading from .local-runs cache",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_torch_device("cpu")
    hub_cache = resolve_gemma3_huggingface_paths(args.cache_dir)["hub_cache"]
    print(f"{DIM}[harness] loading pinned Gemma 3 270M teacher & tokenizer...{RESET}", flush=True)
    tokenizer, model = load_gemma3(
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=hub_cache,
        device=device,
        dtype="float32",
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)

    mode = str(args.mode)
    executor = load_or_compile_executor(
        adapter,
        tokenizer,
        mode=mode,
        rebuild=bool(args.rebuild),
    )
    max_new_tokens = int(args.max_new_tokens)
    temperature = float(args.temperature)
    compare_native = not bool(args.no_compare)
    xray = bool(args.xray)
    equalizer = ModalEqualizerState(
        muted_attn_layers=set(args.mute_attn_layer),
        muted_mlp_layers=set(args.mute_mlp_layer),
        muted_route_substrings=list(args.mute_route),
    )

    if args.prompt is not None:
        run_prompt_comparison(
            adapter,
            tokenizer,
            executor,
            args.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            compare_native=compare_native,
            mode_label=mode,
            equalizer=equalizer,
            xray=xray,
        )
        return 0

    if args.presets:
        _print_banner(mode, xray, equalizer)
        for preset in PRESET_PROMPTS:
            run_prompt_comparison(
                adapter,
                tokenizer,
                executor,
                preset,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                compare_native=compare_native,
                mode_label=mode,
                equalizer=equalizer,
                xray=xray,
            )
        return 0

    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    _print_banner(mode, xray, equalizer)
    last_prompt: str | None = None
    prompt_history: list[str] = []
    while True:
        try:
            line = input(f"\n{BOLD}{CYAN}ewc-gemma ({mode})>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Exiting harness.{RESET}")
            break
        if not line:
            continue
        if line in {"/quit", "/exit", "quit", "exit"}:
            break
        if line == "/xray":
            xray = not xray
            print(f"{YELLOW}X-Ray causal debugger: {'ON' if xray else 'OFF'}{RESET}")
            if xray and last_prompt is not None:
                run_xray_analysis(
                    adapter, tokenizer, executor, last_prompt, mode=mode, equalizer=equalizer
                )
            continue
        if line == "/unmute":
            equalizer.clear()
            print(f"{GREEN}Cleared all mutes: {equalizer.summary()}{RESET}")
            continue
        if line.startswith("/optimize"):
            parts = line.split()
            if len(parts) > 1 and parts[1] == "revert":
                executor = load_or_compile_executor(
                    adapter,
                    tokenizer,
                    mode=mode,
                    rebuild=False,
                )
                print(f"{YELLOW}Executor reloaded from cache -- refit discarded.{RESET}")
                continue
            try:
                lam = float(parts[1]) if len(parts) > 1 else 0.3
                alpha = float(parts[2]) if len(parts) > 2 else 1.0
                beta = float(parts[3]) if len(parts) > 3 else 10.0
            except ValueError:
                print(f"{YELLOW}Usage: /optimize [anchor_lambda] [alpha] [pin_beta] | /optimize revert{RESET}")
                continue
            corpus = list(dict.fromkeys([*PRESET_PROMPTS, *prompt_history]))
            print(f"{DIM}Terminal-state EWC refit on {len(corpus)} prompts "
                  f"(lam={lam}, alpha={alpha}, beta={beta})...{RESET}", flush=True)
            keyed = _tokenize_keyed_batches(
                tokenizer,
                corpus,
                [f"opt{i}" for i in range(len(corpus))],
                batch_size=16,
                device=device,
            )
            fit_batches = tuple(item.model_inputs for item in keyed)
            report = surgical_trajectory_refit(
                adapter=adapter,
                executor=executor,
                batches=fit_batches,
                rounds=1,
                mode="terminal_last",
                mlp_anchor_relative=lam,
                terminal_alpha=alpha,
                terminal_pin_strength=beta,
                progress=lambda s: print(f"{DIM}  {s}{RESET}", flush=True),
            )
            layer = report["rounds"][-1]["layers"][-1]
            metrics, _ = _evaluate_joint_executor(
                adapter=adapter,
                executor=executor,
                batches=fit_batches,
            )
            cand = metrics["candidate"]
            print(f"{GREEN}Terminal refit applied:{RESET} "
                  f"delta_nrmse {layer['terminal_delta_nrmse_before']:.3f} -> "
                  f"{layer['terminal_delta_nrmse_after']:.3f}, "
                  f"drift {layer['weight_drift_fro']:.2f} Fro")
            print(f"  {DIM}On-corpus: KL {cand['native_to_candidate_kl_per_query']:.4f} nats | "
                  f"top1 {cand['top1_agreement_to_native'] * 100:.1f}% | "
                  f"logit_nrmse {cand['logit_nrmse']:.4f}{RESET}")
            print(f"  {DIM}(weights updated in-place; /optimize revert restores the cached artifact){RESET}")
            continue
        if line.startswith("/mute "):
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "route":
                word = " ".join(parts[2:])
                equalizer.muted_route_substrings.append(word)
                print(f"{RED}Muted attention routes from tokens matching {word!r}{RESET}")
            elif len(parts) == 3 and parts[1] == "attn":
                equalizer.muted_attn_layers.add(int(parts[2]))
                print(f"{RED}Muted Attention Layer {int(parts[2])}{RESET}")
            elif len(parts) == 4 and parts[1] == "attn":
                equalizer.muted_attn_heads.add((int(parts[2]), int(parts[3])))
                print(f"{RED}Muted Attention Layer {int(parts[2])} Head {int(parts[3])}{RESET}")
            elif len(parts) == 3 and parts[1] == "mlp":
                equalizer.muted_mlp_layers.add(int(parts[2]))
                print(f"{RED}Muted MLP Layer {int(parts[2])}{RESET}")
            elif len(parts) == 4 and parts[1] == "chart":
                equalizer.muted_mlp_charts.add((int(parts[2]), int(parts[3])))
                print(f"{RED}Muted MLP Layer {int(parts[2])} Chart {int(parts[3])}{RESET}")
            else:
                print(f"{YELLOW}Usage: /mute route <word> | /mute attn <layer> [head] | /mute mlp <layer> | /mute chart <layer> <id>{RESET}")
            continue
        if line.startswith("/atlas"):
            parts = line.split()
            target_l = int(parts[1]) if len(parts) > 1 else 17
            _, atlas = _load_or_build_chart_atlas(adapter, tokenizer)
            print(f"\n  {BOLD}{CYAN}Emergent Vocabulary Motifs for Layer {target_l} MLP Charts (0..7):{RESET}")
            for c_id in range(8):
                toks = atlas.get(target_l, {}).get(c_id, [])
                print(f"    {GREEN}Chart {c_id}{RESET}: {', '.join(repr(t) for t in toks) or '(unoccupied on calibration panel)'}")
            continue
        if line == "/preset":
            for preset in PRESET_PROMPTS:
                run_prompt_comparison(
                    adapter,
                    tokenizer,
                    executor,
                    preset,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    compare_native=compare_native,
                    mode_label=mode,
                    equalizer=equalizer,
                    xray=xray,
                )
            continue
        if line == "/compare":
            compare_native = not compare_native
            print(f"{YELLOW}Side-by-side native comparison: {'ON' if compare_native else 'OFF'}{RESET}")
            continue
        if line.startswith("/tokens "):
            try:
                max_new_tokens = max(1, int(line.split()[1]))
                print(f"{YELLOW}Max new tokens set to {max_new_tokens}{RESET}")
            except ValueError:
                print(f"{YELLOW}Usage: /tokens <int>{RESET}")
            continue
        if line.startswith("/temp "):
            try:
                temperature = max(0.0, float(line.split()[1]))
                print(f"{YELLOW}Temperature set to {temperature}{RESET}")
            except ValueError:
                print(f"{YELLOW}Usage: /temp <float>{RESET}")
            continue
        if line.startswith("/mode "):
            requested = line.split(maxsplit=1)[1].strip()
            if requested not in VALID_MODES:
                print(f"{YELLOW}Valid modes: {', '.join(VALID_MODES)}{RESET}")
                continue
            mode = requested
            executor = load_or_compile_executor(
                adapter,
                tokenizer,
                mode=mode,
                rebuild=False,
            )
            _print_banner(mode, xray, equalizer)
            continue

        last_prompt = line
        if line not in prompt_history:
            prompt_history.append(line)
        run_prompt_comparison(
            adapter,
            tokenizer,
            executor,
            line,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            compare_native=compare_native,
            mode_label=mode,
            equalizer=equalizer,
            xray=xray,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
