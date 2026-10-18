"""Interactive TUI and CLI harness for prompting compiled Pullback-EWC Gemma 3 270M executors.

Usage:
  # Launch interactive TUI:
  .venv/bin/python -m fisher_graph.gemma3_ewc_interactive_harness

  # Run the 6 preset factual prompts side-by-side:
  .venv/bin/python -m fisher_graph.gemma3_ewc_interactive_harness --presets

  # One-shot side-by-side prompt comparison:
  .venv/bin/python -m fisher_graph.gemma3_ewc_interactive_harness \
    --prompt "Complete the category fact: Copper is a kind of"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, SequenceContext
from .gemma3_attention_edge_block_ewc import (
    block_ewc_sparse_attention_from_normalized,
    replay_exact_with_block_ewc_attention,
    select_block_ewc_attention_route_mask,
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
RESET = "\033[0m"


class HybridPullbackEWCExecutor(nn.Module):
    """Whole-model 39-event executor combining Pullback-EWC Lex512 + All-18-Layer Block-EWC K=8 Sparse Attention."""

    def __init__(
        self,
        exact_executor: nn.Module,
        lexical: SharedTiedLexicalFactor,
        *,
        k: int = 8,
        min_per_head: int = 1,
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

    def execute_logits(self, input_ids: Tensor, sequence: SequenceContext) -> Tensor:
        state = self.lexical.ingress(input_ids)
        valid = sequence.query_valid_mask.unsqueeze(-1)
        for mutation_index, mutation in enumerate(self.mutations):
            if mutation_index % 2 == 0:
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
                sparse = block_ewc_sparse_attention_from_normalized(
                    generator,
                    normalized,
                    sequence,
                    mask,
                    ewc_prior_weight=self.ewc_prior_weight,
                )
                delta = sparse.residual_delta
            else:
                incoming = torch.zeros(
                    *state.shape[:2],
                    mutation.incoming_chart_count,
                    mutation.incoming_generator_rank,
                    dtype=state.dtype,
                    device=state.device,
                )
                delta = mutation.execute_with_incoming(state, sequence, incoming).delta
            state = torch.where(valid, state + delta.masked_fill(~valid, 0), state)
        normalized_output = self.terminal_norm(state)
        logits = self.lexical.egress(normalized_output)
        if self.final_logit_softcap is not None:
            cap = float(self.final_logit_softcap)
            logits = torch.tanh(logits / cap) * cap
        return logits


def _cache_path(mode: str) -> Path:
    return CACHE_DIR / f"ewc-interactive-executor-{mode}-v1.pt"


def _lexical_cache_path() -> Path:
    return CACHE_DIR / "ewc-interactive-lex512-v1.pt"


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
        return HybridPullbackEWCExecutor(exact.executor, lexical, k=8, min_per_head=1)

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


def _execute_mode_logits(
    executor: object,
    input_ids: Tensor,
    sequence: SequenceContext,
    mode: str,
) -> Tensor:
    if isinstance(executor, HybridPullbackEWCExecutor):
        return executor.execute_logits(input_ids, sequence)
    if mode == "block-ewc-k8":
        return replay_exact_with_block_ewc_attention(
            executor,
            input_ids,
            sequence,
            k=8,
            min_per_head=1,
            ewc_prior_weight=0.05,
        ).logits
    return executor.execute(input_ids, sequence).logits


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
) -> None:
    """Run autoregressive generation on ``prompt`` and print a rich comparison panel."""
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_ids: Tensor = encoded["input_ids"]
    eos_id = getattr(tokenizer, "eos_token_id", None)

    print(f"\n{BOLD}{CYAN}┌──────────────────────────────────────────────────────────────────────────────┐{RESET}")
    print(f"{BOLD}{CYAN}│ Prompt:{RESET} {BOLD}{prompt}{RESET}")
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
            logits = _execute_mode_logits(executor, cand_ids, seq, mode_label)
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

    if not compare_native:
        if first_cand_logits is not None:
            print(f"  {DIM}First-token Top-5 (EWC):    {_top_k_summary(tokenizer, first_cand_logits)}{RESET}")
        return

    # 2. Native Gemma 3 270M Generation
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

    # 3. Compute first-token KL & Top-5 comparison
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


def _print_banner(mode: str) -> None:
    if mode == "hybrid-ewc":
        desc = "Pullback-EWC Lex512 (-33.2M params) + 18-Layer Block-EWC K=8 Sparse Attention"
    elif mode == "block-ewc-k8":
        desc = "All-18-Layer Block-EWC K=8 Sparse Attention (0% starved heads, 80% Top-1)"
    elif mode == "39-event-fused640":
        desc = "All-39-Event Fused-640 Surrogate (163.3M params, -39.1% params, -71.4% body MACs)"
    else:
        desc = "All-39-Event 2048-Mode Surrogate (196.6M params, -26.7% params, -38.3% body MACs)"

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   Fisher-Graph Gemma 3 270M — Pullback-EWC Interactive Prompting Harness     ║{RESET}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} Active Mode : {BOLD}{GREEN}{mode:62s}{RESET} {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} Description : {desc:62s} {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} Commands:                                                                    {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • Type any prompt and press Enter to generate side-by-side                 {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/preset{RESET}          Run the 6 benchmark factual prompts                     {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/mode <name>{RESET}     Switch mode: {GREEN}hybrid-ewc{RESET}, {GREEN}block-ewc-k8{RESET},                  {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}                                   {GREEN}39-event-fused640{RESET}, {GREEN}39-event{RESET}                {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/tokens <N>{RESET}      Set max new tokens (default: 16)                        {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/temp <T>{RESET}        Set sampling temperature (0.0 = greedy)                 {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/compare{RESET}         Toggle side-by-side native Gemma comparison             {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}║{RESET}   • {YELLOW}/quit{RESET}            Exit the harness                                        {BOLD}{CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════════════╝{RESET}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive TUI / CLI harness for prompting compiled Pullback-EWC Gemma 3 270M models."
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
        )
        return 0

    if args.presets:
        _print_banner(mode)
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
            )
        return 0

    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    _print_banner(mode)
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
            _print_banner(mode)
            continue

        run_prompt_comparison(
            adapter,
            tokenizer,
            executor,
            line,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            compare_native=compare_native,
            mode_label=mode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
