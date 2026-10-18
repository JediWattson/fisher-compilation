"""Surgical closed-form trajectory refit for a compiled joint executor.

Given a fitted :class:`Gemma3JointCrossBlockChartExecutor` and the pinned
Gemma 3 270M teacher, this module performs one or more Gauss-Seidel rounds
of incumbent-anchored ridge refits against the teacher's *functional*
per-layer deltas evaluated at the candidate's own residual states:

* For every valid token on the fit corpus we record the candidate residual
  state entering each of the 36 body mutations, plus the candidate's fixed
  emitted transport latents.  We then call the *native* decoder layer's
  submodules (``input_layernorm``/``self_attn`` and
  ``pre_feedforward_layernorm``/``mlp``) on those same states, obtaining the
  teacher's *raw operator output* ``v`` for the actual compiled-trajectory
  input -- this removes the domino-drift mismatch of fitting on
  native-trajectory rows.
* Because every mutation's ``output_norm`` is an exact unit-offset RMSNorm
  clone of the source post-norm, the exact pre-norm regression target is
  ``y = v / rms(v)``: ``output_norm(y)`` then reproduces the teacher delta
  ``post_norm(v)`` bit-for-bit, with no gain division and no dead-channel
  information loss (``v`` carries full mass even where ``1 + w == 0``).
* MLP mutations are refit in mode space: ``s* = pinv(mode_decoder.T) @ y``,
  then one shared-design anchored ridge over features
  ``[n ; 1 ; emitted_1..8]`` updates ``affine_generators`` and
  ``chart_mode_decoders``.  Chart encoders, chart geometry, and every
  emitted-latent transport vocabulary remain bit-identical by construction.
* Attention mutations are refit in output-coordinate space: an anchored
  ridge from ``output_coordinates`` to ``y`` updates
  ``output_decoder_weight``.  Input coordinates, Q/K/V generators, RoPE,
  route probabilities, and the emitted packet are all untouched.

The incumbent anchor uses per-feature activation evidence (mean of squared
feature values) as the diagonal Fisher proxy -- the same family-aware EWC
schedule as the original fit: directions the calibration corpus exercises
are free to move; unexcercised directions stay pinned to the incumbent.

This is a development-time optimizer.  It calls teacher submodules only
while fitting; the resulting executor remains a source-free runtime.
Evaluation claims belong to the experiment driver that invokes it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import time
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, SequenceContext
from .gemma3_cross_layer_chart_transport_executor import (
    _LAYER_HASH_DOMAIN,
    _artifact_sha256,
)


DEFAULT_MLP_ANCHOR_RELATIVE = 0.10
DEFAULT_ATTENTION_ANCHOR_RELATIVE = 0.10
DEFAULT_REFIT_ROUNDS = 2
_SOLVE_EPSILON = 1.0e-8
_PINV_RTOL = 1.0e-4


@dataclass(frozen=True, slots=True)
class _MutationCapture:
    """Per-token rows captured for one mutation on one round."""

    state_rows: Tensor
    native_prenorm_rows: Tensor
    feature_rows: Tensor
    operator_output_rows: Tensor | None


def _require_executor(executor: object) -> Any:
    mutations = getattr(executor, "mutations", None)
    lexical = getattr(executor, "lexical", None)
    transports = getattr(executor, "transports", None)
    terminal_norm = getattr(executor, "terminal_norm", None)
    if not all(
        value is not None
        for value in (mutations, lexical, transports, terminal_norm)
    ):
        raise TypeError(
            "surgical refit requires a Gemma3JointCrossBlockChartExecutor "
            "with lexical/mutations/transports/terminal_norm"
        )
    return executor


def _native_attention_context(
    adapter: Gemma3CausalLMAdapter,
    batch: Mapping[str, Tensor],
    sequence: SequenceContext,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, tuple[Tensor, Tensor]]]:
    """Build the causal masks and RoPE tables the teacher itself would use."""

    from transformers.models.gemma3.modeling_gemma3 import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    text_config = getattr(adapter.module.config, "text_config", None)
    config = text_config if text_config is not None else adapter.module.config
    position_ids = sequence.logical_positions
    probe = torch.zeros(
        sequence.batch_size,
        sequence.query_length,
        1,
        dtype=dtype,
        device=device,
    )
    mask_kwargs = {
        "config": config,
        "inputs_embeds": probe,
        "attention_mask": batch["attention_mask"],
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    rotary = adapter.module.model.rotary_emb
    position_embeddings = {
        layer_type: rotary(probe, position_ids, layer_type)
        for layer_type in ("full_attention", "sliding_attention")
    }
    return masks, position_embeddings


def _native_attention_prenorm(
    layer: nn.Module,
    state: Tensor,
    *,
    layer_type: str,
    masks: Mapping[str, Tensor],
    position_embeddings: Mapping[str, tuple[Tensor, Tensor]],
    position_ids: Tensor,
) -> Tensor:
    """Raw teacher attention-operator output (pre ``post_attention_layernorm``)."""

    normalized = layer.input_layernorm(state)
    return layer.self_attn(
        hidden_states=normalized,
        position_embeddings=position_embeddings[layer_type],
        attention_mask=masks[layer_type],
        position_ids=position_ids,
    )[0]


def _native_mlp_prenorm(layer: nn.Module, state: Tensor) -> Tensor:
    """Raw teacher MLP output (pre ``post_feedforward_layernorm``)."""

    return layer.mlp(layer.pre_feedforward_layernorm(state))


def _row_rms_normalized(rows: Tensor) -> Tensor:
    return rows.to(torch.float64) / (
        rows.to(torch.float64)
        .square()
        .mean(dim=-1, keepdim=True)
        .add_(_SOLVE_EPSILON)
        .sqrt()
    )


def _unit_offset_gain(module: nn.Module, width: int) -> Tensor:
    """Measure an RMSNorm-style module's per-channel gain on a unit-RMS row."""

    probe = torch.ones(
        (1, 1, width),
        dtype=next(module.parameters()).dtype,
        device=next(module.parameters()).device,
    )
    with torch.no_grad():
        gain = module(probe).float() / probe.float()
    if not bool(torch.isfinite(gain).all()):
        raise RuntimeError("output-norm gain probe produced non-finite gain")
    return gain.reshape(width)


def _invert_norm_gain(delta: Tensor, gain: Tensor) -> Tensor:
    """[unused] kept for provenance: post-norm delta to pre-norm target."""

    gain64 = gain.to(dtype=torch.float64)
    floor = gain64.abs().median().clamp_min(1.0e-12) * 0.02
    dead = gain64.abs() < floor
    safe = torch.where(dead, torch.full_like(gain64, 1.0), gain64)
    target = delta.to(dtype=torch.float64) / safe.unsqueeze(0)
    target[:, dead] = 0.0
    return target


def _anchored_ridge(
    features: Tensor,
    targets: Tensor,
    incumbent: Tensor,
    *,
    anchor_relative: float,
) -> Tensor:
    """Solve ``min_W ||Phi W - Y||^2 + anchor||(W - W0) diag(f)||^2``.

    ``anchor_relative`` is the anchor's mass as a fraction of the observed
    token count; the per-feature diagonal Fisher proxy ``f`` is the mean
    squared feature activation, so well-exercised directions are free to
    move and nullspace directions stay pinned to the incumbent.
    """

    phi = features.to(dtype=torch.float64)
    y = targets.to(dtype=torch.float64)
    w0 = incumbent.to(dtype=torch.float64)
    count = phi.shape[0]
    evidence = phi.square().mean(dim=0)
    # Combined anchor: Fisher-scaled evidence plus a uniform floor, so even
    # near-nullspace features stay pinned to the incumbent with strength
    # ``anchor_relative`` pseudo-observations.
    anchor_diag = evidence + evidence.mean().clamp_min(_SOLVE_EPSILON)
    anchor = anchor_diag.mul(anchor_relative * count).add_(_SOLVE_EPSILON)
    lhs = phi.T @ phi
    lhs += torch.diag(anchor)
    rhs = phi.T @ y + anchor.unsqueeze(1) * w0
    try:
        solved = torch.linalg.solve(lhs, rhs)
    except torch.linalg.LinAlgError:
        solved = torch.linalg.lstsq(lhs, rhs).solution
    if not bool(torch.isfinite(solved).all()):
        solved = torch.linalg.lstsq(lhs, rhs).solution
    if not bool(torch.isfinite(solved).all()):
        raise RuntimeError("anchored ridge solve produced non-finite weights")
    return solved


def _anchored_decoded_ridge(
    features: Tensor,
    targets: Tensor,
    decoder_t: Tensor,
    incumbent: Tensor,
    *,
    anchor_lambda: float,
    anchor_weight: float,
) -> Tensor:
    r"""Solve ``min_W ||Phi W A - Y||^2 + lam||(W - W0) A||^2 + lamw||W - W0||^2``.

    The primary anchor is measured in *decoded* space (through the fixed
    right-factor ``A``), so ``anchor_lambda`` interpolates between the pure
    decoded least-squares refit and the incumbent *function*; a small
    weight-space anchor ``anchor_weight`` additionally pins decoder-null
    directions so weights stay finite where the data is blind.

    Normal equation
    ``(Phi^T Phi + lam I) W M + lamw W = Phi^T Y A^T + lam W0 M + lamw W0``
    with ``M = A A^T`` is solved exactly by joint eigendecomposition.
    """

    phi = features.to(dtype=torch.float64)
    y = targets.to(dtype=torch.float64)
    a = decoder_t.to(dtype=torch.float64)
    w0 = incumbent.to(dtype=torch.float64)
    # Whiten feature columns so ``anchor_lambda`` acts in pseudo-observation
    # units uniformly across heterogeneous feature blocks (normalized
    # inputs, bias, emitted latents).
    scale = phi.square().mean(dim=0).sqrt().clamp_min(_SOLVE_EPSILON)
    phiw = phi / scale.unsqueeze(0)
    w0w = w0 * scale.unsqueeze(1)
    gram = phiw.T @ phiw
    gram.diagonal().add_(anchor_lambda)
    mass = a @ a.T
    rhs = (
        phiw.T @ y @ a.T
        + anchor_lambda * (w0w @ mass)
        + anchor_weight * w0w
    )
    eval_g, evec_g = torch.linalg.eigh(gram)
    eval_m, evec_m = torch.linalg.eigh(mass)
    rhs_rot = evec_g.T @ rhs @ evec_m
    denom = (
        eval_g.unsqueeze(1) * eval_m.unsqueeze(0) + anchor_weight
    ).clamp_min(_SOLVE_EPSILON)
    solved_rot = rhs_rot / denom
    solved = evec_g @ solved_rot @ evec_m.T
    if not bool(torch.isfinite(solved).all()):
        raise RuntimeError("anchored decoded ridge produced non-finite weights")
    return solved / scale.unsqueeze(1)


def _capture_native_boundaries(
    adapter: Gemma3CausalLMAdapter,
    batch: Mapping[str, Tensor],
) -> dict[tuple[str, int], Tensor]:
    """Teacher residual stream at every mutation's output boundary.

    For block ``b`` the attention mutation's boundary is the residual input
    to ``pre_feedforward_layernorm`` and the MLP mutation's boundary is the
    residual input to the next block's ``input_layernorm`` (or
    ``model.norm`` for the last block).
    """

    layers = adapter.module.model.layers
    boundaries: dict[tuple[str, int], Tensor] = {}
    handles: list[Any] = []
    for block, layer in enumerate(layers):
        def _record(module, args, *, key):
            boundaries[key] = args[0]

        handles.append(
            layer.pre_feedforward_layernorm.register_forward_pre_hook(
                lambda m, a, b=block: _record(m, a, key=("attention", b))
            )
        )
        downstream = (
            layers[block + 1].input_layernorm
            if block + 1 < len(layers)
            else adapter.module.model.norm
        )
        handles.append(
            downstream.register_forward_pre_hook(
                lambda m, a, b=block: _record(m, a, key=("mlp", b))
            )
        )
    try:
        adapter.forward(batch)
    finally:
        for handle in handles:
            handle.remove()
    if len(boundaries) != 2 * len(layers):
        raise RuntimeError("native boundary capture missed layers")
    return boundaries


def _capture_round(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Any,
    batches: Sequence[Mapping[str, Tensor]],
) -> tuple[
    list[list[_MutationCapture]],
    list[Tensor],
    list[dict[str, Tensor]],
]:
    """Run the compiled trajectory once, collecting refit rows per mutation.

    The second return element carries, per batch, the teacher's final
    residual-stream rows (input to ``model.model.norm``) restricted to valid
    query positions -- the terminal-state matching target.  The third
    element carries, per batch, a mapping ``event_id -> boundary rows``:
    the teacher residual stream at each mutation's own output boundary
    (the trajectory-snapping target for that mutation).
    """

    device = next(adapter.module.parameters()).device
    dtype = next(adapter.module.parameters()).dtype
    backbone_layers = adapter.module.model.layers
    layer_types = tuple(
        "full_attention" if block in (5, 11, 17) else "sliding_attention"
        for block in range(len(backbone_layers))
    )
    captures: list[list[_MutationCapture]] = [
        [] for _ in range(len(executor.mutations))
    ]
    native_final_rows: list[Tensor] = []
    boundary_rows: list[dict[str, Tensor]] = []
    with torch.no_grad():
        for batch in batches:
            sequence = adapter.prepare_sequence(batch)
            valid = sequence.query_valid_mask
            boundaries = _capture_native_boundaries(adapter, batch)
            native_final_rows.append(
                boundaries[("mlp", len(backbone_layers) - 1)][valid]
                .detach()
                .cpu()
            )
            batch_boundaries: dict[str, Tensor] = {}
            for mutation in executor.mutations:
                spec = mutation.event_spec
                if spec.event_kind == "attention_mutation":
                    key = ("attention", spec.block_ordinal)
                else:
                    key = ("mlp", spec.block_ordinal)
                batch_boundaries[spec.event_id] = (
                    boundaries[key][valid].detach().cpu()
                )
            boundary_rows.append(batch_boundaries)
            masks, position_embeddings = _native_attention_context(
                adapter,
                batch,
                sequence,
                dtype=dtype,
                device=device,
            )
            state = executor.lexical.ingress(batch["input_ids"])
            emitted_packets: dict[str, Tensor] = {}
            for mutation_index, mutation in enumerate(executor.mutations):
                event_id = mutation.event_spec.event_id
                block = mutation.event_spec.block_ordinal
                incoming = torch.zeros(
                    *state.shape[:2],
                    mutation.incoming_chart_count,
                    mutation.incoming_generator_rank,
                    dtype=state.dtype,
                    device=state.device,
                )
                for edge in executor.transports:
                    if edge.spec.target_event_id == event_id:
                        source = emitted_packets[edge.spec.source_event_id]
                        incoming[..., edge.spec.target_chart_index, :] += edge(
                            source[..., edge.spec.source_chart_index, :]
                        )
                layer = backbone_layers[block]
                if mutation.event_spec.event_kind == "attention_mutation":
                    execution = mutation.compiled_operator.execute_with_incoming(
                        state,
                        sequence,
                        incoming,
                    )
                    native_prenorm = _native_attention_prenorm(
                        layer,
                        state,
                        layer_type=layer_types[block],
                        masks=masks,
                        position_embeddings=position_embeddings,
                        position_ids=sequence.logical_positions,
                    )
                    captures[mutation_index].append(
                        _MutationCapture(
                            state_rows=state[valid].detach().cpu(),
                            native_prenorm_rows=(
                                native_prenorm[valid].detach().cpu()
                            ),
                            feature_rows=(
                                execution.output_coordinates.values[valid]
                                .detach()
                                .cpu()
                            ),
                            operator_output_rows=(
                                execution.operator_output[valid].detach().cpu()
                            ),
                        )
                    )
                    delta = execution.residual_delta
                    emitted = execution.emitted_generator_latents
                else:
                    normalized = mutation.input_norm(state)
                    chart_execution = (
                        mutation.chart_generator.execute_with_incoming(
                            normalized,
                            incoming,
                        )
                    )
                    native_prenorm = _native_mlp_prenorm(layer, state)
                    emitted = chart_execution.emitted_generator_latents
                    emitted_rows = emitted[valid].detach().cpu()
                    normalized_rows = normalized[valid].detach().cpu()
                    captures[mutation_index].append(
                        _MutationCapture(
                            state_rows=state[valid].detach().cpu(),
                            native_prenorm_rows=(
                                native_prenorm[valid].detach().cpu()
                            ),
                            feature_rows=torch.cat(
                                (
                                    normalized_rows,
                                    torch.ones(
                                        normalized_rows.shape[0],
                                        1,
                                        dtype=normalized_rows.dtype,
                                    ),
                                    emitted_rows.flatten(start_dim=-2),
                                ),
                                dim=-1,
                            ),
                            operator_output_rows=(
                                chart_execution.output[valid].detach().cpu()
                            ),
                        )
                    )
                    delta = mutation.output_norm(chart_execution.output)
                emitted_packets[event_id] = emitted.masked_fill(
                    ~valid.unsqueeze(-1).unsqueeze(-1),
                    0,
                )
                delta = delta.masked_fill(~valid.unsqueeze(-1), 0)
                state = torch.where(
                    valid.unsqueeze(-1),
                    state + delta,
                    state,
                )
    return captures, native_final_rows, boundary_rows


def _apply_mlp_refit(
    mutation: nn.Module,
    rows: list[_MutationCapture],
    *,
    anchor_relative: float,
) -> dict[str, float]:
    generator = mutation.chart_generator
    mode_decoder_weight = generator.mode_decoder.weight.detach()
    hidden = mode_decoder_weight.shape[0]
    modes = mode_decoder_weight.shape[1]
    charts = generator.chart_count
    rank = generator.generator_rank
    features = torch.cat([item.feature_rows for item in rows], dim=0)
    native_prenorm = torch.cat(
        [item.native_prenorm_rows for item in rows], dim=0
    ).to(torch.float64)
    gain = _unit_offset_gain(mutation.output_norm, hidden).to(torch.float64)
    # Gain-weighted decoded target: matching ``pre * gain`` to ``v * gain``
    # is the linear surrogate for ``rms_norm(pre) * gain == teacher delta``.
    # Dead channels (gain ~ 0) correctly contribute ~0 loss instead of an
    # unidentifiable target.
    target_delta_space = native_prenorm * gain.unsqueeze(0)
    native = _row_rms_normalized(native_prenorm) * gain.unsqueeze(0)
    incumbent = torch.cat(
        (
            generator.affine_generators.weight.detach().T,
            generator.affine_generators.bias.detach().unsqueeze(0),
            generator.chart_mode_decoders.detach().reshape(
                charts * rank,
                modes,
            ),
        ),
        dim=0,
    )
    expected = hidden + 1 + charts * rank
    if features.shape[1] != expected or incumbent.shape[0] != expected:
        raise RuntimeError(
            f"MLP refit feature width drifted: "
            f"{features.shape[1]} vs {expected}"
        )
    # Decoded-space ridge in *delta* metric: right factor A_eff = W_dec^T G
    # folds the output-norm gain into the decoded coordinate system.
    a_eff = mode_decoder_weight.to(torch.float64).T * gain.unsqueeze(0)
    anchor_lambda = anchor_relative * features.shape[0]
    anchor_weight = max(anchor_lambda * 1.0e-3, _SOLVE_EPSILON)
    solved = _anchored_decoded_ridge(
        features,
        target_delta_space,
        a_eff,
        incumbent,
        anchor_lambda=anchor_lambda,
        anchor_weight=anchor_weight,
    ).to(dtype=generator.affine_generators.weight.dtype)
    # On-trajectory diagnostic: predicted modal states with solved weights,
    # pushed back through the frozen mode_decoder + output_norm.
    eps = 1.0e-8
    gain64 = gain
    solved_modes = features.to(torch.float64) @ solved.to(torch.float64)
    solved_prenorm = solved_modes @ mode_decoder_weight.to(torch.float64).T
    rms = solved_prenorm.square().mean(dim=-1, keepdim=True).add_(eps).sqrt()
    solved_delta = solved_prenorm / rms * gain64.unsqueeze(0)
    native64 = native
    nrmse_after = (
        (solved_delta - native64).square().mean().sqrt()
        / native64.square().mean().sqrt().clamp_min(1.0e-12)
    )
    incumbent_modes = features.to(torch.float64) @ incumbent.to(torch.float64)
    incumbent_prenorm = incumbent_modes @ mode_decoder_weight.to(torch.float64).T
    rms_i = incumbent_prenorm.square().mean(dim=-1, keepdim=True).add_(eps).sqrt()
    incumbent_delta = incumbent_prenorm / rms_i * gain64.unsqueeze(0)
    nrmse_before = (
        (incumbent_delta - native64).square().mean().sqrt()
        / native64.square().mean().sqrt().clamp_min(1.0e-12)
    )
    with torch.no_grad():
        generator.affine_generators.weight.copy_(
            solved[:hidden].T.contiguous()
        )
        generator.affine_generators.bias.copy_(solved[hidden].contiguous())
        generator.chart_mode_decoders.copy_(
            solved[hidden + 1 :]
            .reshape(charts, rank, modes)
            .contiguous()
        )
    generator._artifact_sha256 = _artifact_sha256(
        _LAYER_HASH_DOMAIN,
        generator._hash_payload(),
    )
    return {
        "row_count": float(features.shape[0]),
        "weight_drift_fro": float(
            (solved - incumbent.to(solved.dtype)).norm().item()
        ),
        "delta_nrmse_before": float(nrmse_before.item()),
        "delta_nrmse_after": float(nrmse_after.item()),
    }


def _apply_attention_refit(
    mutation: nn.Module,
    rows: list[_MutationCapture],
    *,
    anchor_relative: float,
) -> dict[str, float]:
    operator = mutation.compiled_operator
    decoder_weight = operator.output_decoder_weight.detach()
    spec = operator.config.output_norm
    if spec.scale_parameterization != "unit_offset":
        raise RuntimeError("attention output norm is not unit-offset RMSNorm")
    features = torch.cat([item.feature_rows for item in rows], dim=0)
    native_prenorm = torch.cat(
        [item.native_prenorm_rows for item in rows], dim=0
    ).to(torch.float64)
    gain = (
        1.0 + operator.output_norm_weight.detach().float()
    ).cpu().to(torch.float64)
    # Gain-weighted delta-space target (see _apply_mlp_refit).
    target_delta_space = native_prenorm * gain.unsqueeze(0)
    incumbent = decoder_weight.T  # [in_rank, residual_width]
    anchor_lambda = anchor_relative * features.shape[0]
    anchor_weight = max(anchor_lambda * 1.0e-3, _SOLVE_EPSILON)
    solved = _anchored_decoded_ridge(
        features,
        target_delta_space,
        torch.diag(gain),
        incumbent,
        anchor_lambda=anchor_lambda,
        anchor_weight=anchor_weight,
    ).to(dtype=decoder_weight.dtype)
    with torch.no_grad():
        operator.output_decoder_weight.copy_(solved.T.contiguous())
        if operator.o_bias is not None:
            operator.o_bias.mul_(0.0)
            raise RuntimeError(
                "attention output bias refit is not yet implemented; "
                "refusing to keep a stale bias"
            )
    native64 = _row_rms_normalized(native_prenorm) * gain.unsqueeze(0)
    eps = 1.0e-8
    denom = native64.square().mean().sqrt().clamp_min(1.0e-12)

    solved_prenorm = features.to(torch.float64) @ solved.to(torch.float64)
    rms = solved_prenorm.square().mean(dim=-1, keepdim=True).add_(eps).sqrt()
    nrmse_after = (
        (solved_prenorm / rms * gain.unsqueeze(0) - native64)
        .square().mean().sqrt() / denom
    )
    incumbent_prenorm = features.to(torch.float64) @ incumbent.to(torch.float64)
    rms_i = incumbent_prenorm.square().mean(dim=-1, keepdim=True).add_(eps).sqrt()
    nrmse_before = (
        (incumbent_prenorm / rms_i * gain.unsqueeze(0) - native64)
        .square().mean().sqrt() / denom
    )
    return {
        "row_count": float(features.shape[0]),
        "weight_drift_fro": float(
            (solved - incumbent.to(solved.dtype)).norm().item()
        ),
        "delta_nrmse_before": float(nrmse_before.item()),
        "delta_nrmse_after": float(nrmse_after.item()),
    }


def _apply_terminal_mlp_refit(
    mutation: nn.Module,
    rows: list[_MutationCapture],
    boundary_rows: Tensor,
    *,
    anchor_relative: float,
    alpha: float = 0.5,
    pin_strength: float = 1.0,
) -> dict[str, float]:
    """Refit the *last* mutation to absorb the full accumulated trajectory error.

    Rather than matching the teacher's per-layer delta, the solve targets the
    residual needed to land the compiled final state on the teacher's final
    hidden state: ``delta_t = h_native_final - state_pre``.  Since the
    mutation's own input is unaffected by its refit, the target is exact.

    ``post_norm(pre) = pre/rms(pre)*gain == delta_t`` is achieved exactly
    when ``pre * gain == delta_t * rms(pre)``.  We solve in the
    gain-weighted decoded metric ``q = pre * gain`` for a *blended* q-target
    ``q_t = q_inc + alpha * (delta_t * r_inc - q_inc)`` where
    ``q_inc = incumbent_pre * gain`` and ``r_inc`` is the incumbent per-row
    scale: ``alpha = 0`` reproduces the incumbent, ``alpha = 1`` requests
    the full terminal correction, and intermediate values bound the step.
    Dead channels (gain ~ 0) contribute ~0 loss through the gain-weighted
    right factor, so they correctly keep the incumbent's dead-channel
    behaviour; the remaining row-scale slack in ``post`` is absorbed by the
    row normalization.
    """

    generator = mutation.chart_generator
    mode_decoder_weight = generator.mode_decoder.weight.detach()
    hidden = mode_decoder_weight.shape[0]
    modes = mode_decoder_weight.shape[1]
    charts = generator.chart_count
    rank = generator.generator_rank
    features = torch.cat([item.feature_rows for item in rows], dim=0)
    state_pre = torch.cat(
        [item.state_rows for item in rows], dim=0
    ).to(torch.float64)
    incumbent_pre = torch.cat(
        [item.operator_output_rows for item in rows], dim=0
    ).to(torch.float64)
    native_final = boundary_rows.to(torch.float64)
    delta_target = native_final - state_pre
    gain = _unit_offset_gain(mutation.output_norm, hidden).to(torch.float64)
    incumbent = torch.cat(
        (
            generator.affine_generators.weight.detach().T,
            generator.affine_generators.bias.detach().unsqueeze(0),
            generator.chart_mode_decoders.detach().reshape(
                charts * rank,
                modes,
            ),
        ),
        dim=0,
    )
    expected = hidden + 1 + charts * rank
    if features.shape[1] != expected or incumbent.shape[0] != expected:
        raise RuntimeError(
            "terminal MLP refit feature width drifted: "
            f"{features.shape[1]} vs {expected}"
        )
    a_eff = mode_decoder_weight.to(torch.float64).T * gain.unsqueeze(0)
    anchor_lambda = anchor_relative * features.shape[0]
    anchor_weight = max(anchor_lambda * 1.0e-3, _SOLVE_EPSILON)
    w_dec_t = mode_decoder_weight.to(torch.float64).T
    q_inc = incumbent_pre * gain.unsqueeze(0)
    r_inc = incumbent_pre.square().mean(dim=-1, keepdim=True).add_(
        _SOLVE_EPSILON
    ).sqrt()
    target_q = q_inc + alpha * (delta_target * r_inc - q_inc)
    # The q-space target assumes rms(pre_solved) ~ r_inc; the gain-weighted
    # loss alone leaves dead/low-gain prenorm channels unconstrained, so the
    # solved row scale drifts and post-normalization lands off target.  A
    # second plain decoded-space incumbent pin
    # ``beta * ||Phi (W - W0) W_dec^T||^2`` keeps ``rms(pre)`` consistent --
    # folded into the same eigendecomposition solve by stacking the right
    # factors and targets.
    pin = pin_strength ** 0.5
    a_cat = torch.cat((a_eff, w_dec_t * pin), dim=-1)
    y_cat = torch.cat((target_q, incumbent_pre * pin), dim=-1)
    solved = _anchored_decoded_ridge(
        features,
        y_cat,
        a_cat,
        incumbent,
        anchor_lambda=anchor_lambda,
        anchor_weight=anchor_weight,
    )
    solved_pre = (features.to(torch.float64) @ solved) @ w_dec_t
    solved_pre = (features.to(torch.float64) @ solved) @ w_dec_t
    rms = solved_pre.square().mean(dim=-1, keepdim=True).add_(
        _SOLVE_EPSILON
    ).sqrt()
    solved_delta = solved_pre / rms * gain.unsqueeze(0)
    denom = delta_target.square().mean().sqrt().clamp_min(1.0e-12)
    nrmse_after = (solved_delta - delta_target).square().mean().sqrt() / denom
    incumbent_delta = incumbent_pre / (
        incumbent_pre.square().mean(dim=-1, keepdim=True)
        .add_(_SOLVE_EPSILON).sqrt()
    ) * gain.unsqueeze(0)
    nrmse_before = (
        (incumbent_delta - delta_target).square().mean().sqrt() / denom
    )
    with torch.no_grad():
        generator.affine_generators.weight.copy_(
            solved[:hidden].T.contiguous().to(
                dtype=generator.affine_generators.weight.dtype
            )
        )
        generator.affine_generators.bias.copy_(
            solved[hidden].contiguous().to(
                dtype=generator.affine_generators.bias.dtype
            )
        )
        generator.chart_mode_decoders.copy_(
            solved[hidden + 1 :]
            .reshape(charts, rank, modes)
            .contiguous()
            .to(dtype=generator.chart_mode_decoders.dtype)
        )
    generator._artifact_sha256 = _artifact_sha256(
        _LAYER_HASH_DOMAIN,
        generator._hash_payload(),
    )
    return {
        "row_count": float(features.shape[0]),
        "weight_drift_fro": float(
            (solved - incumbent.to(solved.dtype)).norm().item()
        ),
        "terminal_delta_nrmse_before": float(nrmse_before.item()),
        "terminal_delta_nrmse_after": float(nrmse_after.item()),
        "terminal_gap_rms": float(denom.item()),
    }


def _apply_attention_boundary_refit(
    mutation: nn.Module,
    rows: list[_MutationCapture],
    boundary_rows: Tensor,
    *,
    anchor_relative: float,
    alpha: float = 0.5,
    pin_strength: float = 1.0,
) -> dict[str, float]:
    """Attention analogue of :func:`_apply_terminal_mlp_refit`.

    ``pre = output_coords @ output_decoder_weight.T`` and the emitted delta
    is ``rms_norm(pre) * (1 + w)``; we fit ``q = pre * gain`` toward
    ``q_t = q_inc + alpha * (delta_t * r_inc - q_inc)`` with the same
    plain-decoded incumbent pin that keeps ``rms(pre)`` consistent.
    """

    operator = mutation.compiled_operator
    spec = operator.config.output_norm
    if spec.scale_parameterization != "unit_offset":
        raise RuntimeError("attention output norm is not unit-offset RMSNorm")
    features = torch.cat([item.feature_rows for item in rows], dim=0)
    state_pre = torch.cat(
        [item.state_rows for item in rows], dim=0
    ).to(torch.float64)
    incumbent_pre = torch.cat(
        [item.operator_output_rows for item in rows], dim=0
    ).to(torch.float64)
    boundary = boundary_rows.to(torch.float64)
    delta_target = boundary - state_pre
    gain = (
        1.0 + operator.output_norm_weight.detach().float()
    ).cpu().to(torch.float64)
    incumbent = operator.output_decoder_weight.detach().to(
        torch.float64
    ).T.contiguous()  # [in_rank, residual_width]
    hidden = incumbent.shape[1]
    eye = torch.eye(hidden, dtype=torch.float64)
    a_eff = torch.diag(gain)
    q_inc = incumbent_pre * gain.unsqueeze(0)
    r_inc = incumbent_pre.square().mean(dim=-1, keepdim=True).add_(
        _SOLVE_EPSILON
    ).sqrt()
    target_q = q_inc + alpha * (delta_target * r_inc - q_inc)
    pin = pin_strength ** 0.5
    a_cat = torch.cat((a_eff, eye * pin), dim=-1)
    y_cat = torch.cat((target_q, incumbent_pre * pin), dim=-1)
    anchor_lambda = anchor_relative * features.shape[0]
    anchor_weight = max(anchor_lambda * 1.0e-3, _SOLVE_EPSILON)
    solved = _anchored_decoded_ridge(
        features,
        y_cat,
        a_cat,
        incumbent,
        anchor_lambda=anchor_lambda,
        anchor_weight=anchor_weight,
    )
    solved_pre = features.to(torch.float64) @ solved
    rms = solved_pre.square().mean(dim=-1, keepdim=True).add_(
        _SOLVE_EPSILON
    ).sqrt()
    solved_delta = solved_pre / rms * gain.unsqueeze(0)
    denom = delta_target.square().mean().sqrt().clamp_min(1.0e-12)
    nrmse_after = (solved_delta - delta_target).square().mean().sqrt() / denom
    incumbent_delta = incumbent_pre / r_inc * gain.unsqueeze(0)
    nrmse_before = (
        (incumbent_delta - delta_target).square().mean().sqrt() / denom
    )
    with torch.no_grad():
        operator.output_decoder_weight.copy_(
            solved.T.contiguous().to(
                dtype=operator.output_decoder_weight.dtype
            )
        )
        if operator.o_bias is not None:
            operator.o_bias.mul_(0.0)
            raise RuntimeError(
                "attention output bias refit is not yet implemented; "
                "refusing to keep a stale bias"
            )
    return {
        "row_count": float(features.shape[0]),
        "weight_drift_fro": float(
            (solved - incumbent.to(solved.dtype)).norm().item()
        ),
        "terminal_delta_nrmse_before": float(nrmse_before.item()),
        "terminal_delta_nrmse_after": float(nrmse_after.item()),
        "terminal_gap_rms": float(denom.item()),
    }


def surgical_trajectory_refit(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Any,
    batches: Sequence[Mapping[str, Tensor]],
    rounds: int = DEFAULT_REFIT_ROUNDS,
    mlp_anchor_relative: float = DEFAULT_MLP_ANCHOR_RELATIVE,
    attention_anchor_relative: float = DEFAULT_ATTENTION_ANCHOR_RELATIVE,
    mode: str = "per_layer",
    terminal_alpha: float = 0.5,
    terminal_pin_strength: float = 1.0,
    snap_tail_count: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Gauss-Seidel closed-form refit of one compiled executor in place.

    ``mode="per_layer"`` refits every mutation against the teacher's
    per-layer raw operator output.  ``mode="terminal_last"`` instead refits
    only the final mutation so its emitted delta cancels the accumulated
    trajectory error (``h_native_final - state_pre``): the eval-visible
    objective applied at the single site whose input is unaffected by its
    own update.
    """

    _require_executor(executor)
    if mode not in ("per_layer", "terminal_last", "trajectory_snap"):
        raise ValueError(f"unknown refit mode: {mode}")
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    if not isinstance(mlp_anchor_relative, (int, float)) or not (
        0.0 < float(mlp_anchor_relative)
    ):
        raise ValueError("mlp_anchor_relative must be positive")
    if not isinstance(attention_anchor_relative, (int, float)) or not (
        0.0 < float(attention_anchor_relative)
    ):
        raise ValueError("attention_anchor_relative must be positive")

    report_rounds: list[dict[str, object]] = []
    for round_index in range(rounds):
        started = time.perf_counter()
        captures, native_final_rows, boundary_rows = _capture_round(
            adapter=adapter,
            executor=executor,
            batches=batches,
        )
        if mode == "trajectory_snap":
            layer_reports = []
            total = len(executor.mutations)
            start_index = (
                0
                if snap_tail_count is None
                else max(0, total - snap_tail_count)
            )
            for mutation_index, mutation in enumerate(executor.mutations):
                if mutation_index < start_index:
                    continue
                boundary = torch.cat(
                    [
                        item[mutation.event_spec.event_id]
                        for item in boundary_rows
                    ],
                    dim=0,
                )
                if mutation.event_spec.event_kind == "attention_mutation":
                    result = _apply_attention_boundary_refit(
                        mutation,
                        captures[mutation_index],
                        boundary,
                        anchor_relative=attention_anchor_relative,
                        alpha=terminal_alpha,
                        pin_strength=terminal_pin_strength,
                    )
                else:
                    result = _apply_terminal_mlp_refit(
                        mutation,
                        captures[mutation_index],
                        boundary,
                        anchor_relative=mlp_anchor_relative,
                        alpha=terminal_alpha,
                        pin_strength=terminal_pin_strength,
                    )
                layer_reports.append(
                    {
                        "event_id": mutation.event_spec.event_id,
                        "event_kind": mutation.event_spec.event_kind,
                        "block_ordinal": mutation.event_spec.block_ordinal,
                        **result,
                    }
                )
            report_rounds.append(
                {
                    "round_index": round_index,
                    "seconds": time.perf_counter() - started,
                    "layers": layer_reports,
                }
            )
            if progress is not None:
                progressed = [
                    report
                    for report in layer_reports
                    if report["terminal_delta_nrmse_after"]
                    < report["terminal_delta_nrmse_before"]
                ]
                progress(
                    f"trajectory-snap round {round_index + 1}/{rounds} "
                    f"({report_rounds[-1]['seconds']:.1f}s) "
                    f"{len(progressed)}/{len(layer_reports)} layers improved"
                )
            continue
        if mode == "terminal_last":
            last_index = len(executor.mutations) - 1
            last = executor.mutations[last_index]
            if last.event_spec.event_kind != "mlp_mutation":
                raise RuntimeError(
                    "terminal_last requires the final mutation to be an MLP"
                )
            result = _apply_terminal_mlp_refit(
                last,
                captures[last_index],
                torch.cat(native_final_rows, dim=0),
                anchor_relative=mlp_anchor_relative,
                alpha=terminal_alpha,
                pin_strength=terminal_pin_strength,
            )
            report_rounds.append(
                {
                    "round_index": round_index,
                    "seconds": time.perf_counter() - started,
                    "layers": [
                        {
                            "event_id": last.event_spec.event_id,
                            "event_kind": last.event_spec.event_kind,
                            "block_ordinal": last.event_spec.block_ordinal,
                            **result,
                        }
                    ],
                }
            )
            if progress is not None:
                progress(
                    f"terminal refit round {round_index + 1}/{rounds} "
                    f"({report_rounds[-1]['seconds']:.1f}s) "
                    f"delta_nrmse {result['terminal_delta_nrmse_before']:.3f}"
                    f" -> {result['terminal_delta_nrmse_after']:.3f}"
                )
            continue
        layer_reports: list[dict[str, object]] = []
        for mutation_index, mutation in enumerate(executor.mutations):
            rows = captures[mutation_index]
            if mutation.event_spec.event_kind == "attention_mutation":
                result = _apply_attention_refit(
                    mutation,
                    rows,
                    anchor_relative=attention_anchor_relative,
                )
            else:
                result = _apply_mlp_refit(
                    mutation,
                    rows,
                    anchor_relative=mlp_anchor_relative,
                )
            layer_reports.append(
                {
                    "event_id": mutation.event_spec.event_id,
                    "event_kind": mutation.event_spec.event_kind,
                    "block_ordinal": mutation.event_spec.block_ordinal,
                    **result,
                }
            )
        for mutation in executor.mutations:
            if (
                mutation.event_spec.event_kind == "mlp_mutation"
                and not bool(
                    torch.isfinite(
                        mutation.chart_generator.affine_generators.weight
                    ).all()
                )
            ):
                raise RuntimeError("non-finite MLP weights after refit")
        report_rounds.append(
            {
                "round_index": round_index,
                "seconds": time.perf_counter() - started,
                "layers": layer_reports,
            }
        )
        if progress is not None:
            progress(
                f"refit round {round_index + 1}/{rounds} complete "
                f"({report_rounds[-1]['seconds']:.1f}s)"
            )
    return {
        "refit_kind": "surgical_trajectory_anchored_ridge",
        "rounds": report_rounds,
        "mlp_anchor_relative": float(mlp_anchor_relative),
        "attention_anchor_relative": float(attention_anchor_relative),
        "refit_parameter_groups": [
            "mlp.affine_generators.weight",
            "mlp.affine_generators.bias",
            "mlp.chart_mode_decoders",
            "attention.output_decoder_weight",
        ],
        "frozen_parameter_groups": [
            "mlp.local_generator_encoders",
            "mlp.chart_centers/chart_bases/chart_distance_scales",
            "mlp.mode_decoder",
            "attention.q/k/v/output_encoder/input_basis",
            "lexical factor",
            "transports",
            "all normalization weights",
        ],
    }


def _report_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
