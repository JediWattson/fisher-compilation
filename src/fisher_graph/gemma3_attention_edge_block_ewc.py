"""Block-EWC coalition-aware route selection and optimal EWC reweighting for sparse attention.

This module addresses the two structural failure modes of scalar diagonal-Fisher
attention pruning identified in the parts-first attention-edge fidelity study:

1. Head starvation & off-diagonal coalition loss:
   Instead of ranking routes ``(head, key)`` independently by scalar diagonal energy
   ``E = P^2 * sum((G * U)^2)`` (which ignores constructive/destructive interference
   between routes and can starve entire heads), ``select_block_ewc_attention_route_mask``
   guarantees per-head minimum support and greedily minimizes the full off-diagonal
   Block-EWC quadratic residual ``||sum_h (mu_{h,q} - sum_{k in S_h} P_{h,q,k} U_{h,k})||_W^2``
   at each query token.
2. Spurious renormalization vs. zero-fill truncation loss:
   Instead of leaving surviving probabilities unnormalized (losing collinear mass) or
   blindly dividing by ``sum_{k in S} P_{h,q,k}`` (which amplifies orthogonal value
   directions by ``1 / mass`` when ``mass`` is small), ``solve_block_ewc_sparse_attention_weights``
   solves the closed-form ``k_h x k_h`` Block-EWC normal equations anchored at native
   probabilities ``p_S``:
   ``w_S* = max(0, (G_S + lambda_EWC * bar_g * I)^{-1} (U_S mu_{h,q} + lambda_EWC * bar_g * p_S))``.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn

from .adapters import SequenceContext
from .gemma3_attention_generator import Gemma3AttentionGenerator
from .gemma3_joint_cross_block_chart_executor import (
    JOINT_EXACT_CONTROL_ARM_ID,
)
from .gemma3_parts_first_attention_edge_fidelity import (
    SparseAttentionExecution,
    SparseWholeModelExecution,
    _repeat_sequence,
    _validate_generator,
    attention_route_tensors,
    operator_from_routed_probabilities,
)


__all__ = [
    "block_ewc_sparse_attention_from_normalized",
    "replay_exact_with_block_ewc_attention",
    "select_block_ewc_attention_route_mask",
    "solve_block_ewc_sparse_attention_weights",
]


def select_block_ewc_attention_route_mask(
    probabilities: Tensor,
    projected_values: Tensor,
    allowed_mask: Tensor,
    *,
    k: int,
    min_per_head: int = 1,
    operator_gradients: Tensor | None = None,
) -> Tensor:
    """Select up to ``k`` routes per query using per-head minimums and off-diagonal Block-EWCOMP.

    Works both retrospectively (when ``operator_gradients`` ``[B, Q, D]`` is provided)
    and prospectively/target-free (when ``operator_gradients`` is ``None``).
    """
    if (
        not isinstance(probabilities, Tensor)
        or probabilities.ndim != 4
        or not probabilities.is_floating_point()
        or not bool(torch.isfinite(probabilities).all())
        or not isinstance(projected_values, Tensor)
        or projected_values.ndim != 4
        or projected_values.shape[:2] != probabilities.shape[:2]
        or projected_values.shape[2] != probabilities.shape[3]
        or not isinstance(allowed_mask, Tensor)
        or allowed_mask.dtype is not torch.bool
        or allowed_mask.shape != (probabilities.shape[0], probabilities.shape[2], probabilities.shape[3])
    ):
        raise ValueError("invalid tensors for Block-EWC attention route selection")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if type(min_per_head) is not int or min_per_head < 0:
        raise ValueError("min_per_head must be a nonnegative integer")

    batch, heads, queries, keys = probabilities.shape
    width = projected_values.shape[-1]
    if operator_gradients is not None:
        if (
            not isinstance(operator_gradients, Tensor)
            or operator_gradients.shape != (batch, queries, width)
            or not bool(torch.isfinite(operator_gradients).all())
        ):
            raise ValueError("operator_gradients must have finite shape [batch, queries, width]")

    probs64 = probabilities.detach().to(device="cpu", dtype=torch.float64)
    proj64 = projected_values.detach().to(device="cpu", dtype=torch.float64)
    allowed_cpu = allowed_mask.detach().to(device="cpu")
    grads64 = (
        None
        if operator_gradients is None
        else operator_gradients.detach().to(device="cpu", dtype=torch.float64)
    )

    mask = torch.zeros_like(probabilities, dtype=torch.bool, device="cpu")
    per_head_quota = min(min_per_head, k // max(heads, 1))

    for b in range(batch):
        for q in range(queries):
            valid_keys = torch.nonzero(allowed_cpu[b, q], as_tuple=False).reshape(-1)
            num_valid = int(valid_keys.numel())
            if num_valid == 0:
                continue

            if grads64 is not None:
                g_q = grads64[b, q].abs()
                g_rms = float(g_q.square().mean().sqrt().item())
                w_q = torch.sqrt(g_q.square() + (0.1 * max(g_rms, 1e-12)) ** 2)
            else:
                w_q = torch.ones(width, dtype=torch.float64)

            # Normalize value vectors by their per-head RMS norm so a single outlier-norm
            # token or head cannot monopolize the global route budget before post-attention RMSNorm.
            p_hq = probs64[b, :, q, :][:, valid_keys]  # [H, V]
            u_raw = proj64[b, :, valid_keys, :] * w_q.view(1, 1, width)  # [H, V, D]
            u_rms = u_raw.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            u_hk = u_raw / u_rms
            v_hk = p_hq.unsqueeze(-1) * u_hk  # [H, V, D]
            v_norm_sq = v_hk.square().sum(dim=-1)  # [H, V]

            head_residuals = v_hk.sum(dim=1)  # [H, D]
            selected_local = torch.zeros((heads, num_valid), dtype=torch.bool)
            selected_count = 0
            max_total = min(k, heads * num_valid)

            # Phase 1: Per-head minimum support to prevent head starvation
            for _ in range(min(per_head_quota, num_valid)):
                for h in range(heads):
                    if selected_count >= max_total:
                        break
                    avail = ~selected_local[h]
                    if not bool(avail.any()):
                        continue
                    dots = torch.mv(v_hk[h], head_residuals[h])
                    gains = 2.0 * dots - v_norm_sq[h]
                    gains = gains.masked_fill(~avail, float("-inf"))
                    best_v = int(gains.argmax().item())
                    selected_local[h, best_v] = True
                    head_residuals[h] = head_residuals[h] - v_hk[h, best_v]
                    selected_count += 1

            # Phase 2: Within-head off-diagonal Block-EWC greedy reduction across heads
            while selected_count < max_total:
                avail = ~selected_local
                if not bool(avail.any()):
                    break
                dots = torch.einsum("hvd,hd->hv", v_hk, head_residuals)
                gains = (2.0 * dots - v_norm_sq).masked_fill(~avail, float("-inf"))
                flat_idx = int(gains.reshape(-1).argmax().item())
                best_h = flat_idx // num_valid
                best_v = flat_idx % num_valid
                selected_local[best_h, best_v] = True
                head_residuals[best_h] = head_residuals[best_h] - v_hk[best_h, best_v]
                selected_count += 1

            for h in range(heads):
                chosen_v = valid_keys[selected_local[h]]
                mask[b, h, q, chosen_v] = True

    return mask.to(device=probabilities.device).contiguous()


def solve_block_ewc_sparse_attention_weights(
    probabilities: Tensor,
    projected_values: Tensor,
    allowed_mask: Tensor,
    selection_mask: Tensor,
    *,
    ewc_prior_weight: float = 0.05,
) -> Tensor:
    """Solve closed-form Block-EWC optimal route weights on surviving edges per (batch, head, query)."""
    if (
        not isinstance(probabilities, Tensor)
        or probabilities.ndim != 4
        or not isinstance(projected_values, Tensor)
        or projected_values.ndim != 4
        or not isinstance(allowed_mask, Tensor)
        or allowed_mask.dtype is not torch.bool
        or not isinstance(selection_mask, Tensor)
        or selection_mask.dtype is not torch.bool
        or selection_mask.shape != probabilities.shape
    ):
        raise ValueError("invalid tensors for Block-EWC sparse weight solve")
    if not math.isfinite(ewc_prior_weight) or ewc_prior_weight < 0.0:
        raise ValueError("ewc_prior_weight must be finite and nonnegative")

    batch, heads, queries, keys = probabilities.shape
    probs64 = probabilities.detach().to(device="cpu", dtype=torch.float64)
    proj64 = projected_values.detach().to(device="cpu", dtype=torch.float64)
    allowed_cpu = allowed_mask.detach().to(device="cpu")
    sel_cpu = selection_mask.detach().to(device="cpu")

    routed64 = torch.zeros_like(probs64)
    for b in range(batch):
        for q in range(queries):
            if not bool(allowed_cpu[b, q].any()):
                continue
            for h in range(heads):
                sel_keys = torch.nonzero(sel_cpu[b, h, q], as_tuple=False).reshape(-1)
                m = int(sel_keys.numel())
                if m == 0:
                    continue
                if bool((sel_cpu[b, h, q] == allowed_cpu[b, q]).all()):
                    routed64[b, h, q] = probs64[b, h, q]
                    continue

                p_all = probs64[b, h, q]  # [K]
                u_all = proj64[b, h]  # [K, D]
                target_mu = torch.mv(u_all.T, p_all)  # [D]

                u_sel = u_all.index_select(0, sel_keys)  # [m, D]
                p_sel = p_all.index_select(0, sel_keys)  # [m]
                mass = float(p_sel.sum().item())
                p_renorm = p_sel / max(mass, 1e-30) if mass > 0.0 else p_sel
                renorm_mu = torch.mv(u_sel.T, p_renorm)  # [D]
                delta_mu = target_mu - renorm_mu

                gram = u_sel @ u_sel.T  # [m, m]
                rhs = torch.mv(u_sel, delta_mu)  # [m]

                mean_diag = max(float(torch.diagonal(gram).mean().item()), 1e-12)
                reg = max(float(ewc_prior_weight), 1e-8) * mean_diag
                system = gram + reg * torch.eye(m, dtype=torch.float64)
                delta_w = torch.linalg.solve(system, rhs)
                w_opt = (p_renorm + delta_w).clamp_min(0.0)
                routed64[b, h, q, sel_keys] = w_opt

    return routed64.to(device=probabilities.device, dtype=probabilities.dtype).contiguous()


def block_ewc_sparse_attention_from_normalized(
    generator: Gemma3AttentionGenerator,
    normalized_input: Tensor,
    sequence: SequenceContext,
    selection_mask: Tensor,
    *,
    ewc_prior_weight: float = 0.05,
) -> SparseAttentionExecution:
    """Execute sparse attention using closed-form Block-EWC optimal route reweighting."""
    routes = attention_route_tensors(generator, normalized_input, sequence)
    if (
        not isinstance(selection_mask, Tensor)
        or selection_mask.dtype is not torch.bool
        or selection_mask.shape != routes.probabilities.shape
        or bool((selection_mask & ~routes.allowed_mask[:, None, :, :]).any())
    ):
        raise ValueError("sparse attention selection mask is invalid")

    routed = solve_block_ewc_sparse_attention_weights(
        routes.probabilities,
        routes.projected_values,
        routes.allowed_mask,
        selection_mask,
        ewc_prior_weight=ewc_prior_weight,
    )
    routed = torch.where(
        sequence.query_valid_mask[:, None, :, None],
        routed,
        routes.execution_probabilities,
    )
    operator, delta = operator_from_routed_probabilities(
        generator,
        routed,
        routes.repeated_values,
        sequence,
    )
    selected_mass = routes.probabilities.masked_fill(~selection_mask, 0).sum(dim=-1)
    return SparseAttentionExecution(
        operator_output=operator,
        residual_delta=delta,
        routed_probabilities=routed,
        selected_native_probability_mass=selected_mass,
        active_head_query_mask=selection_mask.any(dim=-1),
    )


def replay_exact_with_block_ewc_attention(
    executor: nn.Module,
    input_ids: Tensor,
    sequence: SequenceContext,
    *,
    k: int = 8,
    min_per_head: int = 1,
    ewc_prior_weight: float = 0.05,
    layer_selection_masks: tuple[Tensor, ...] | None = None,
    egress_query_ordinal: int | None = None,
) -> SparseWholeModelExecution:
    """Replay the 39-event whole-model executor with Block-EWC sparse attention across all 18 layers."""
    mutations = getattr(executor, "mutations", None)
    transports = getattr(executor, "transports", None)
    if (
        getattr(executor, "arm_id", None) != JOINT_EXACT_CONTROL_ARM_ID
        or not isinstance(mutations, nn.ModuleList)
        or len(mutations) != 36
        or not isinstance(transports, nn.ModuleList)
        or len(transports) != 0
        or (layer_selection_masks is not None and len(layer_selection_masks) != 18)
        or not isinstance(input_ids, Tensor)
        or input_ids.ndim != 2
        or input_ids.dtype not in (torch.int32, torch.int64)
        or input_ids.device.type != "cpu"
    ):
        raise ValueError("Block-EWC attention replay requires the exact transport-free executor")

    batch_size = int(input_ids.shape[0])
    if sequence.batch_size == 1 and batch_size != 1:
        active_sequence = _repeat_sequence(sequence, batch_size)
    elif sequence.batch_size == batch_size:
        active_sequence = sequence
    else:
        raise ValueError("input batch and sequence context disagree")

    state = executor.lexical.ingress(input_ids)
    valid = active_sequence.query_valid_mask.unsqueeze(-1)
    attention_operators: list[Tensor] = []
    attention_deltas: list[Tensor] = []
    mutation_deltas: list[Tensor] = []
    mutation_carriers: list[Tensor] = []

    for mutation_index, mutation in enumerate(mutations):
        if mutation_index % 2 == 0:
            layer = mutation_index // 2
            generator = getattr(mutation, "compiled_operator", None)
            _validate_generator(generator)
            normalized, _ = generator.encode_input(state, active_sequence)
            if layer_selection_masks is not None:
                mask = layer_selection_masks[layer]
                if mask.shape[0] == 1 and batch_size != 1:
                    mask = mask.repeat(batch_size, 1, 1, 1)
            else:
                routes = attention_route_tensors(generator, normalized, active_sequence)
                mask = select_block_ewc_attention_route_mask(
                    routes.probabilities,
                    routes.projected_values,
                    routes.allowed_mask,
                    k=k,
                    min_per_head=min_per_head,
                )
            sparse = block_ewc_sparse_attention_from_normalized(
                generator,
                normalized,
                active_sequence,
                mask,
                ewc_prior_weight=ewc_prior_weight,
            )
            delta = sparse.residual_delta
            attention_operators.append(sparse.operator_output.detach().clone())
            attention_deltas.append(delta.detach().clone())
        else:
            incoming = torch.zeros(
                *state.shape[:2],
                mutation.incoming_chart_count,
                mutation.incoming_generator_rank,
                dtype=state.dtype,
                device=state.device,
            )
            delta = mutation.execute_with_incoming(state, active_sequence, incoming).delta
        mutation_deltas.append(delta.detach().clone())
        state = torch.where(valid, state + delta.masked_fill(~valid, 0), state)
        mutation_carriers.append(state.detach().clone())

    normalized_output = executor.terminal_norm(state)
    if egress_query_ordinal is None:
        egress_input = normalized_output
    else:
        if (
            type(egress_query_ordinal) is not int
            or not 0 <= egress_query_ordinal < normalized_output.shape[1]
            or not bool(active_sequence.query_valid_mask[:, egress_query_ordinal].all())
        ):
            raise ValueError("egress_query_ordinal is not a valid shared query")
        egress_input = normalized_output[:, egress_query_ordinal : egress_query_ordinal + 1]
    logits = executor.lexical.egress(egress_input)
    cap = getattr(executor, "final_logit_softcap", None)
    if cap is not None:
        logits = torch.tanh(logits / float(cap)) * float(cap)
    return SparseWholeModelExecution(
        output=state,
        normalized_output=normalized_output,
        logits=logits,
        attention_operator_outputs=tuple(attention_operators),
        attention_residual_deltas=tuple(attention_deltas),
        mutation_residual_deltas=tuple(mutation_deltas),
        mutation_carriers=tuple(mutation_carriers),
    )

