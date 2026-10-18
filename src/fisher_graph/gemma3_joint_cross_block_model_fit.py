"""Fit MLP charts and attention-to-MLP transports for the joint Gemma arm.

This is a development fitter, not a held-out authority.  It uses the pinned
prompt-disjoint fit panel to capture native MLP Fisher rows and sequence-aware
attention generator packets.  Every target MLP chart is fitted locally first;
only the remaining modal residual is exposed to strictly earlier attention
events from any prior block.  Layer distance is never a numerical feature.

The resulting artifacts are source-independent modules, but the first v1 fit
is intentionally one-shot on the native trajectory.  Candidate-trajectory
iteration and family-disjoint guard evaluation remain separate later rungs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .gemma3_attention_generator import (
    Gemma3AttentionGenerator,
    ReducedAttentionGeneratorCoefficients,
    compile_reduced_attention_generator,
)
from .gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
    compute_tied_lexical_gram,
    fit_shared_tied_lexical_factor,
)
from .gemma3_cross_layer_chart_transport_executor import (
    Gemma3FactorizedChartGeneratorLayer,
)
from .gemma3_cross_layer_chart_transport_fit import (
    factor_chart_edge_bank,
    fit_fisher_weighted_affine_base,
    fit_fisher_weighted_dense_chart_edge_bank,
)
from .gemma3_joint_cross_block_chart_executor import (
    JointGraphCrossBlockTransportEdge,
    JointGraphTransportSpec,
)
from .gemma3_joint_cross_block_transport_fit import (
    fit_greedy_joint_cross_block_transports,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    CHART_COUNT,
    CHART_EDGE_RIDGE,
    CHART_RANK,
    CHART_TEMPERATURE,
    _LayerCapture,
    _capture_fisher_rows,
    _chart_membership_coordinates,
    _deterministic_chart_geometry,
    _relative_rmse,
)


@dataclass(frozen=True, slots=True)
class JointCrossBlockFitResult:
    """Executable fit-only artifacts for all 18 MLP mutations."""

    mlp_generators: tuple[Gemma3FactorizedChartGeneratorLayer, ...]
    transports: tuple[JointGraphCrossBlockTransportEdge, ...]
    layer_metrics: Mapping[str, object]
    fit_row_identity_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _row_identity(
    batch_row_keys: Sequence[Sequence[tuple[str, int]]],
) -> tuple[tuple[tuple[str, int], ...], str]:
    if isinstance(batch_row_keys, (str, bytes)) or not isinstance(
        batch_row_keys, Sequence
    ):
        raise TypeError("batch_row_keys must be a materialized sequence")
    rows = tuple(row for batch in batch_row_keys for row in tuple(batch))
    if not rows or any(
        not isinstance(example_id, str)
        or not example_id
        or type(position) is not int
        or position < 0
        for example_id, position in rows
    ):
        raise ValueError("fit row identity is invalid")
    if len(set(rows)) != len(rows):
        raise ValueError("fit row identities must be unique")
    digest = hashlib.sha256(
        b"fisher_graph:joint-cross-block-fit-rows:v1\0"
        + _canonical_json_bytes(rows)
    ).hexdigest()
    return rows, digest


def _row_fisher_weights(capture: _LayerCapture) -> Tensor:
    weights = (capture.features * capture.score_gradients).square().mean(dim=1)
    mean = weights.mean()
    if not bool(torch.isfinite(mean)) or float(mean) <= 0.0:
        raise RuntimeError("MLP Fisher row weights are degenerate")
    return (weights / mean).clamp(min=0.1, max=10.0)


def _attention_event_id(block_ordinal: int) -> str:
    return f"mutation.{2 * block_ordinal + 1}"


def _mlp_event_id(block_ordinal: int) -> str:
    return f"mutation.{2 * block_ordinal + 2}"


def _capture_attention_packets(
    *,
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[Mapping[str, Tensor]],
    attention_generators: Sequence[Gemma3AttentionGenerator],
) -> dict[str, Tensor]:
    if len(attention_generators) != len(adapter.layers):
        raise ValueError("attention generator catalog is incomplete")
    capture_sites = tuple(layer.input_site for layer in adapter.layers)
    packets: dict[str, list[Tensor]] = {
        _attention_event_id(ordinal): []
        for ordinal in range(len(adapter.layers))
    }
    for batch in batches:
        with torch.inference_mode():
            observed = adapter.forward(batch, capture_sites=capture_sites)
            for ordinal, generator in enumerate(attention_generators):
                source_state = observed.activations[
                    adapter.layers[ordinal].input_site
                ]
                execution = generator(source_state, observed.sequence)
                packet = execution.emitted_generator_latents
                if packet.shape[:2] != source_state.shape[:2]:
                    raise RuntimeError("attention packet sequence shape drifted")
                packets[_attention_event_id(ordinal)].append(
                    packet[observed.sequence.query_valid_mask]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
    return {
        event_id: torch.cat(values, dim=0).contiguous()
        for event_id, values in packets.items()
    }


def fit_joint_cross_block_mlp_charts(
    *,
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[Mapping[str, Tensor]],
    batch_row_keys: Sequence[Sequence[tuple[str, int]]],
    attention_generators: Sequence[Gemma3AttentionGenerator],
    model_fingerprint: str,
    generator_rank: int = 4,
    maximum_incoming_attention_sources_per_target_chart: int = 1,
    transport_relative_ridge: float = 1.0e-4,
    progress: Callable[[str], None] | None = None,
) -> JointCrossBlockFitResult:
    """Fit all MLP charts plus non-adjacent attention-to-MLP edges."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be Gemma3CausalLMAdapter")
    if len(adapter.layers) != 18:
        raise ValueError("joint Gemma v1 requires all 18 blocks")
    materialized_batches = tuple(batches)
    if not materialized_batches:
        raise ValueError("fit batches cannot be empty")
    rows, row_sha256 = _row_identity(batch_row_keys)
    if sum(
        int(batch["attention_mask"].bool().sum().item())
        for batch in materialized_batches
    ) != len(rows):
        raise ValueError("fit row keys do not match valid token rows")
    if (
        not isinstance(model_fingerprint, str)
        or len(model_fingerprint) != 64
    ):
        raise ValueError("model_fingerprint must be a SHA-256 digest")
    if type(generator_rank) is not int or generator_rank <= 0:
        raise ValueError("generator_rank must be positive")

    layer_ordinals = tuple(range(len(adapter.layers)))
    captures: dict[int, _LayerCapture] = {}
    for ordinal in layer_ordinals:
        if progress is not None:
            progress(
                "capture full-native-suffix Fisher rows for MLP block "
                f"{ordinal}/17"
            )
        # The shared capture helper makes each requested modal feature a leaf.
        # Capturing all 18 targets together would therefore detach later MLP
        # branches from earlier targets and truncate their score gradients.
        # One target per backward pass preserves the complete native suffix for
        # every Fisher row while keeping the source checkpoint frozen.
        observed = _capture_fisher_rows(
            adapter,
            materialized_batches,
            target_layers=(ordinal,),
        )
        if tuple(observed) != (ordinal,):
            raise RuntimeError("single-target Fisher capture inventory drifted")
        captures[ordinal] = observed[ordinal]
    attention_packets = _capture_attention_packets(
        adapter=adapter,
        batches=materialized_batches,
        attention_generators=attention_generators,
    )
    if any(packet.shape[0] != len(rows) for packet in attention_packets.values()):
        raise RuntimeError("attention and MLP fit rows are not aligned")

    prefix_catalog = hashlib.sha256(
        b"fisher_graph:joint-native-fit-trajectory:v1\0"
        + model_fingerprint.encode("ascii")
        + row_sha256.encode("ascii")
    ).hexdigest()
    mlp_generators: list[Gemma3FactorizedChartGeneratorLayer] = []
    local_executions: list[object] = []
    row_weights: list[Tensor] = []
    layer_metrics: dict[str, object] = {}

    for ordinal in layer_ordinals:
        if progress is not None:
            progress(f"fit Fisher activation chart for MLP block {ordinal}/17")
        capture = captures[ordinal]
        if capture.inputs.shape[0] != len(rows):
            raise RuntimeError("MLP capture row count drifted")
        weights = _row_fisher_weights(capture)
        row_weights.append(weights)
        affine_weight, affine_bias, affine_metrics = (
            fit_fisher_weighted_affine_base(
                inputs=capture.inputs,
                modal_states=capture.features,
                row_fisher_weights=weights,
                relative_ridge=1.0e-3,
            )
        )
        centers, bases, scales, occupancy = _deterministic_chart_geometry(
            capture.inputs,
            weights,
            chart_count=CHART_COUNT,
            chart_rank=CHART_RANK,
        )
        memberships, coordinates = _chart_membership_coordinates(
            capture.inputs,
            centers,
            bases,
            scales,
            temperature=CHART_TEMPERATURE,
        )
        base = capture.inputs @ affine_weight.T + affine_bias
        dense, dense_metrics = fit_fisher_weighted_dense_chart_edge_bank(
            memberships=memberships,
            coordinates=coordinates,
            base_modal_residual=capture.features - base,
            row_fisher_weights=weights,
            relative_ridge=CHART_EDGE_RIDGE,
        )
        local_encoder, mode_decoder, factor_metrics = factor_chart_edge_bank(
            dense,
            generator_rank,
        )
        source_mlp = adapter.source_module(adapter.layers[ordinal].id).mlp
        layer = Gemma3FactorizedChartGeneratorLayer(
            affine_weight=affine_weight,
            affine_bias=affine_bias,
            mode_decoder_weight=source_mlp.down_proj.weight.detach(),
            chart_centers=centers,
            chart_bases=bases,
            chart_distance_scales=scales,
            local_generator_encoders=local_encoder,
            chart_mode_decoders=mode_decoder,
            chart_temperature=CHART_TEMPERATURE,
            model_fingerprint=model_fingerprint,
            layer_ordinal=ordinal,
            fit_row_identity_sha256=row_sha256,
            compiled_prefix_catalog_sha256=prefix_catalog,
        )
        with torch.no_grad():
            local = layer.execute_with_incoming(capture.inputs)
        mlp_generators.append(layer)
        local_executions.append(local)
        layer_metrics[str(ordinal)] = {
            "row_count": len(rows),
            "fit_trajectory": "one_shot_native_development_v1",
            "affine": affine_metrics,
            "dense_chart": dense_metrics,
            "factorization": factor_metrics,
            "hard_chart_occupancy": occupancy.tolist(),
            "local_modal_nrmse": _relative_rmse(
                local.modal_states,
                capture.features,
            ),
            "transport": {
                "source_event_count": 0,
                "selected_edge_count": 0,
            },
        }

    transports: list[JointGraphCrossBlockTransportEdge] = []
    for target_ordinal in range(1, len(adapter.layers)):
        if progress is not None:
            progress(
                "fit all-earlier attention transports into MLP block "
                f"{target_ordinal}/17"
            )
        capture = captures[target_ordinal]
        target = mlp_generators[target_ordinal]
        local = local_executions[target_ordinal]
        target_residual = capture.features - local.modal_states
        earlier_attention = {
            _attention_event_id(source): attention_packets[
                _attention_event_id(source)
            ]
            for source in range(target_ordinal)
        }
        fitted, metrics = fit_greedy_joint_cross_block_transports(
            source_packets=earlier_attention,
            target_memberships=local.memberships,
            target_mode_decoders=target.chart_mode_decoders,
            target_modal_residual=target_residual,
            row_fisher_weights=row_weights[target_ordinal],
            maximum_incoming_sources_per_target_chart=(
                maximum_incoming_attention_sources_per_target_chart
            ),
            relative_ridge=transport_relative_ridge,
        )
        for edge_index, edge in enumerate(fitted):
            spec = JointGraphTransportSpec(
                transport_id=(
                    f"transport.{edge.source_event_id}.to."
                    f"{_mlp_event_id(target_ordinal)}."
                    f"c{edge.source_chart_index}.d{edge.target_chart_index}."
                    f"e{edge_index}"
                ),
                source_event_id=edge.source_event_id,
                target_event_id=_mlp_event_id(target_ordinal),
                source_chart_index=edge.source_chart_index,
                target_chart_index=edge.target_chart_index,
            )
            transports.append(
                JointGraphCrossBlockTransportEdge(
                    spec=spec,
                    transport_weight=edge.transport_weight,
                )
            )
        layer_metrics[str(target_ordinal)]["transport"] = metrics

    if not transports:
        raise RuntimeError(
            "joint candidate fit found no positive cross-block transport"
        )
    return JointCrossBlockFitResult(
        mlp_generators=tuple(mlp_generators),
        transports=tuple(transports),
        layer_metrics=layer_metrics,
        fit_row_identity_sha256=row_sha256,
    )


def fit_pullback_ewc_shared_tied_lexical_factor(
    *,
    adapter: Gemma3CausalLMAdapter,
    exact_executor: nn.Module,
    batches: Sequence[Mapping[str, Tensor]],
    source_weight: Tensor,
    lexical_rank: int,
    embed_scale: float = 1.0,
    ewc_prior_weight: float = 0.05,
) -> SharedTiedLexicalFactor:
    """Fit a rank-``lexical_rank`` tied lexical factor under the ingress + egress + active-token Pullback-EWC metric."""
    h0_list: list[Tensor] = []
    hfinal_list: list[Tensor] = []
    active_ids: list[Tensor] = []
    with torch.no_grad():
        for batch in batches:
            seq = adapter.prepare_sequence(batch)
            valid = seq.query_valid_mask
            active_ids.append(batch["input_ids"][valid].reshape(-1))
            h0 = exact_executor.lexical.ingress(batch["input_ids"])[valid].double()
            ex_out = exact_executor.execute(batch["input_ids"], seq)
            hf = exact_executor.terminal_norm(ex_out.output)[valid].double()
            h0_list.append(h0)
            hfinal_list.append(hf)
    h0_all = torch.cat(h0_list, dim=0)
    hf_all = torch.cat(hfinal_list, dim=0)
    uniq_ids = torch.unique(torch.cat(active_ids, dim=0))
    e_act = source_weight.detach()[uniq_ids].double()
    g_act = e_act.T @ e_act
    g_in = h0_all.T @ h0_all
    g_out = hf_all.T @ hf_all
    g_emb = compute_tied_lexical_gram(source_weight)
    g_ewc = (
        2.0 * (g_in / torch.trace(g_in).clamp_min(1e-12))
        + 2.0 * (g_out / torch.trace(g_out).clamp_min(1e-12))
        + 2.0 * (g_act / torch.trace(g_act).clamp_min(1e-12))
        + float(ewc_prior_weight) * (g_emb / torch.trace(g_emb).clamp_min(1e-12))
    )
    return fit_shared_tied_lexical_factor(
        source_weight,
        lexical_rank,
        embed_scale=embed_scale,
        gram=g_ewc,
    )


def fit_pullback_ewc_reduced_attention_generator(
    exact_gen: Gemma3AttentionGenerator,
    norm_inputs: Tensor,
    contexts: Tensor,
    *,
    op_outputs: Tensor | None = None,
    row_weights: Tensor | None = None,
    input_rank: int = 256,
    output_rank: int = 64,
    ewc_prior_weight: float = 0.005,
) -> Gemma3AttentionGenerator:
    """Compile a reduced-coordinate attention generator using RMSNorm-Pullback + QK-norm EWC anchoring."""
    n = max(1, int(norm_inputs.shape[0]))
    if row_weights is not None:
        w = row_weights.double()
        w = w / w.mean().clamp_min(1e-12)
        root_w = w.sqrt().unsqueeze(1)
    else:
        root_w = torch.ones((n, 1), dtype=torch.float64)

    x = norm_inputs.double() * root_w
    c = contexts.double()
    if op_outputs is not None:
        y = op_outputs.double()
        s_y = y.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
        c_tilde = (c / s_y) * root_w
    else:
        c_tilde = c * root_w

    w_q = exact_gen.q_weight.detach().double()
    w_k = exact_gen.k_weight.detach().double()
    w_v = exact_gen.v_weight.detach().double()
    w_o = exact_gen.output_decoder_weight.detach().double()
    w_norm = (1.0 + exact_gen.output_norm_weight.detach().double()).abs().clamp_min(1e-3)
    if exact_gen.q_norm_weight is not None and exact_gen.k_norm_weight is not None:
        qn = (1.0 + exact_gen.q_norm_weight.detach().double()).abs().clamp_min(1e-3).repeat(
            w_q.shape[0] // exact_gen.q_norm_weight.shape[0]
        )
        kn = (1.0 + exact_gen.k_norm_weight.detach().double()).abs().clamp_min(1e-3).repeat(
            w_k.shape[0] // exact_gen.k_norm_weight.shape[0]
        )
        w_q_eff = w_q * qn.unsqueeze(1)
        w_k_eff = w_k * kn.unsqueeze(1)
    else:
        w_q_eff, w_k_eff = w_q, w_k
    repeats = w_o.shape[1] // w_v.shape[0]
    w_v_rep = w_v.repeat(repeats, 1)
    w_v_eff = (w_o * w_norm.unsqueeze(1)) @ w_v_rep

    gq = w_q_eff.T @ w_q_eff
    gk = w_k_eff.T @ w_k_eff
    gv = w_v_eff.T @ w_v_eff
    qkv_gram = (
        gq / torch.trace(gq).clamp_min(1e-12)
        + gk / torch.trace(gk).clamp_min(1e-12)
        + 2.0 * (gv / torch.trace(gv).clamp_min(1e-12))
    )

    cov_x = (x.T @ x) / float(n)
    mean_dx = float(torch.diagonal(cov_x).mean().clamp_min(1e-12).item())
    cov_x_reg = cov_x + float(ewc_prior_weight) * mean_dx * torch.eye(
        x.shape[1],
        dtype=torch.float64,
    )
    pb_in = cov_x_reg @ qkv_gram @ cov_x_reg
    _, evecs_in = torch.linalg.eigh(0.5 * (pb_in + pb_in.T))
    input_basis = evecs_in[:, -input_rank:].flip(1).contiguous()
    max_in = input_basis.abs().argmax(dim=0)
    signs_in = torch.where(
        input_basis[max_in, torch.arange(input_rank)] < 0,
        -1.0,
        1.0,
    )
    input_basis = (input_basis * signs_in.unsqueeze(0)).contiguous()

    z = x @ input_basis
    gram_z = (z.T @ z) / float(n)
    mean_dz = float(torch.diagonal(gram_z).mean().clamp_min(1e-12).item())
    reg_z = float(ewc_prior_weight) * mean_dz
    sys_z = gram_z + reg_z * torch.eye(input_rank, dtype=torch.float64)
    rhs_x = (z.T @ x) / float(n) + reg_z * input_basis.T
    recon_proj = torch.linalg.solve(sys_z, rhs_x)
    q_dec = (w_q @ recon_proj.T).contiguous()
    k_dec = (w_k @ recon_proj.T).contiguous()
    v_dec = (w_v @ recon_proj.T).contiguous()

    cov_c = (c_tilde.T @ c_tilde) / float(n)
    mean_dc = float(torch.diagonal(cov_c).mean().clamp_min(1e-12).item())
    cov_c_reg = cov_c + float(ewc_prior_weight) * mean_dc * torch.eye(
        c.shape[1],
        dtype=torch.float64,
    )
    evals_c, evecs_c = torch.linalg.eigh(0.5 * (cov_c_reg + cov_c_reg.T))
    sqrt_c = evecs_c @ torch.diag(evals_c.clamp_min(1e-12).sqrt()) @ evecs_c.T
    inv_sqrt_c = evecs_c @ torch.diag(evals_c.clamp_min(1e-12).rsqrt()) @ evecs_c.T
    m_mat = sqrt_c @ w_o.T * w_norm.unsqueeze(0)
    u_m, s_m, vh_m = torch.linalg.svd(m_mat, full_matrices=False)
    root_s = s_m[:output_rank].sqrt()
    out_enc = (inv_sqrt_c @ (u_m[:, :output_rank] * root_s.unsqueeze(0))).T.contiguous()
    out_dec = ((root_s.unsqueeze(1) * vh_m[:output_rank, :]) / w_norm.unsqueeze(0)).T.contiguous()

    return compile_reduced_attention_generator(
        exact_gen.config,
        ReducedAttentionGeneratorCoefficients(
            input_norm_weight=exact_gen.input_norm_weight,
            input_basis=input_basis.to(dtype=exact_gen.dtype, device=exact_gen.device),
            q_decoder_weight=q_dec.to(dtype=exact_gen.dtype, device=exact_gen.device),
            k_decoder_weight=k_dec.to(dtype=exact_gen.dtype, device=exact_gen.device),
            v_decoder_weight=v_dec.to(dtype=exact_gen.dtype, device=exact_gen.device),
            output_encoder_weight=out_enc.to(dtype=exact_gen.dtype, device=exact_gen.device),
            output_decoder_weight=out_dec.to(dtype=exact_gen.dtype, device=exact_gen.device),
            output_norm_weight=exact_gen.output_norm_weight,
            q_norm_weight=exact_gen.q_norm_weight,
            k_norm_weight=exact_gen.k_norm_weight,
            q_bias=exact_gen.q_bias,
            k_bias=exact_gen.k_bias,
            v_bias=exact_gen.v_bias,
            o_bias=exact_gen.o_bias,
        ),
    )


def _ewc_pullback_chart_geometry(
    inputs: Tensor,
    residual_targets: Tensor,
    weights: Tensor,
    *,
    chart_count: int = CHART_COUNT,
    chart_rank: int = CHART_RANK,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute chart centers and scales, then refine chart bases via EWC Pullback-Fisher."""
    centers, _, scales, occupancy = _deterministic_chart_geometry(
        inputs,
        weights,
        chart_count=chart_count,
        chart_rank=chart_rank,
    )
    x = inputs.double()
    r = residual_targets.double()
    w = weights.double()
    w = w / w.sum().clamp_min(1e-12)
    x_c = x - (w.unsqueeze(1) * x).sum(dim=0)
    r_c = r - (w.unsqueeze(1) * r).sum(dim=0)
    cov_global = x_c.T @ (w.unsqueeze(1) * x_c)
    cross_xr = x_c.T @ (w.unsqueeze(1) * r_c)
    pb_global = cross_xr @ cross_xr.T
    pb_global = pb_global * (
        torch.trace(cov_global) / torch.trace(pb_global).clamp_min(1e-12)
    )
    prior_gram = 0.5 * cov_global + 0.5 * pb_global

    disp = x.unsqueeze(1) - centers.double().unsqueeze(0)
    dist_sq = disp.square().sum(dim=-1)
    logits = -0.5 * dist_sq / (
        scales.double().square().unsqueeze(0) * CHART_TEMPERATURE
    )
    memb = logits.softmax(dim=-1)

    new_bases = torch.empty(
        (chart_count, x.shape[1], chart_rank),
        dtype=torch.float64,
    )
    for c in range(chart_count):
        wc = w * memb[:, c]
        wc_sum = float(wc.sum().item())
        if wc_sum > 1e-12:
            wc_norm = wc / wc_sum
            dc = x - centers[c].double()
            cov_c = dc.T @ (wc_norm.unsqueeze(1) * dc)
            rc = r - (wc_norm.unsqueeze(1) * r).sum(dim=0)
            cxr_c = dc.T @ (wc_norm.unsqueeze(1) * rc)
            pb_c = cxr_c @ cxr_c.T
            if float(torch.trace(pb_c).item()) > 1e-12:
                pb_c = pb_c * (torch.trace(cov_c) / torch.trace(pb_c))
            gram_c = 0.5 * cov_c + 0.5 * pb_c + 0.25 * prior_gram
        else:
            gram_c = prior_gram
        _, evecs = torch.linalg.eigh(0.5 * (gram_c + gram_c.T))
        b_c = evecs[:, -chart_rank:].flip(1).contiguous()
        piv = b_c.abs().argmax(dim=0)
        sgn = torch.where(
            b_c[piv, torch.arange(chart_rank)] < 0,
            -1.0,
            1.0,
        )
        new_bases[c] = b_c * sgn.unsqueeze(0)
    return (
        centers,
        new_bases.to(dtype=inputs.dtype).contiguous(),
        scales,
        occupancy,
    )


def fit_ewc_joint_cross_block_whole_model(
    *,
    adapter: Gemma3CausalLMAdapter,
    exact_executor: nn.Module,
    lexical: SharedTiedLexicalFactor,
    exact_attention: Sequence[Gemma3AttentionGenerator],
    batches: Sequence[Mapping[str, Tensor]],
    batch_row_keys: Sequence[Sequence[tuple[str, int]]],
    model_fingerprint: str,
    attention_input_rank: int = 256,
    attention_output_rank: int = 64,
    generator_rank: int = 4,
    maximum_incoming_attention_sources_per_target_chart: int = 1,
    fuse_mlp_modes: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[tuple[Gemma3AttentionGenerator, ...], JointCrossBlockFitResult]:
    """Sequentially fit all 18 reduced attention generators, MLP charts, and EWC transports."""
    materialized_batches = tuple(batches)
    rows, row_sha256 = _row_identity(batch_row_keys)
    layer_ordinals = tuple(range(len(adapter.layers)))
    captures: dict[int, _LayerCapture] = {}
    for ordinal in layer_ordinals:
        if progress is not None:
            progress(f"capture full-native-suffix Fisher rows for MLP block {ordinal}/17")
        observed = _capture_fisher_rows(
            adapter,
            materialized_batches,
            target_layers=(ordinal,),
        )
        captures[ordinal] = observed[ordinal]

    exact_mlp_muts = tuple(
        m for m in exact_executor.mutations if m.event_spec.event_kind == "mlp_mutation"
    )
    seqs_per_batch = [adapter.prepare_sequence(b) for b in materialized_batches]
    states_per_batch = [lexical.ingress(b["input_ids"]).detach() for b in materialized_batches]

    prefix_catalog = hashlib.sha256(
        b"fisher_graph:joint-ewc-fit-trajectory:v1\0"
        + model_fingerprint.encode("ascii")
        + row_sha256.encode("ascii")
    ).hexdigest()

    ewc_attn_generators: list[Gemma3AttentionGenerator] = []
    mlp_generators: list[Gemma3FactorizedChartGeneratorLayer] = []
    transports: list[JointGraphCrossBlockTransportEdge] = []
    attention_packets: dict[str, Tensor] = {}
    layer_metrics: dict[str, object] = {}
    probe_gen = torch.Generator().manual_seed(1729)

    for b in layer_ordinals:
        if progress is not None:
            progress(f"fit Pullback-EWC attention and MLP chart for block {b}/17")
        ex_gen = exact_attention[b]
        norm_list: list[Tensor] = []
        ctx_list: list[Tensor] = []
        op_list: list[Tensor] = []
        with torch.no_grad():
            for s_state, seq in zip(states_per_batch, seqs_per_batch):
                valid = seq.query_valid_mask
                ex_run = ex_gen(s_state, seq)
                norm_list.append(ex_run.normalized_input[valid])
                ctx_list.append(ex_run.output_coordinates.values[valid])
                op_list.append(ex_run.operator_output[valid])

        raw_weights = _row_fisher_weights(captures[b])
        red_gen = fit_pullback_ewc_reduced_attention_generator(
            ex_gen,
            torch.cat(norm_list, dim=0),
            torch.cat(ctx_list, dim=0),
            op_outputs=torch.cat(op_list, dim=0),
            row_weights=raw_weights,
            input_rank=attention_input_rank,
            output_rank=attention_output_rank,
        )
        ewc_attn_generators.append(red_gen)

        mlp_in_list: list[Tensor] = []
        mlp_feat_list: list[Tensor] = []
        pkt_list: list[Tensor] = []
        mlp_mut_exact = exact_mlp_muts[b]
        source_mlp = adapter.source_module(adapter.layers[b].id).mlp
        w_down = source_mlp.down_proj.weight.detach()
        w_post = (1.0 + mlp_mut_exact.output_norm.weight.detach()).abs().clamp_min(0.05)
        mode_decoder_weight = (
            torch.diag(1.0 / w_post).contiguous()
            if fuse_mlp_modes
            else w_down
        )

        with torch.no_grad():
            new_states: list[Tensor] = []
            for s_state, seq in zip(states_per_batch, seqs_per_batch):
                v_mask = seq.query_valid_mask.unsqueeze(-1)
                valid = seq.query_valid_mask
                run_a = red_gen(s_state, seq)
                pkt_list.append(
                    run_a.emitted_generator_latents[valid].cpu().float().contiguous()
                )
                s_state = torch.where(
                    v_mask,
                    s_state + run_a.residual_delta.masked_fill(~v_mask, 0),
                    s_state,
                )
                new_states.append(s_state)
                norm_mlp_in = mlp_mut_exact.input_norm(s_state)[valid].cpu().float().contiguous()
                mlp_in_list.append(norm_mlp_in)
                gate = F.gelu(
                    F.linear(norm_mlp_in, source_mlp.gate_proj.weight),
                    approximate="tanh",
                )
                up = F.linear(norm_mlp_in, source_mlp.up_proj.weight)
                feat = gate * up
                if fuse_mlp_modes:
                    feat = F.linear(feat, w_down) * w_post.unsqueeze(0)
                mlp_feat_list.append(feat.cpu().float().contiguous())
            states_per_batch = new_states

        attention_packets[_attention_event_id(b)] = torch.cat(pkt_list, dim=0).contiguous()
        cand_inputs = torch.cat(mlp_in_list, dim=0).contiguous()
        cand_features = torch.cat(mlp_feat_list, dim=0).contiguous()
        nat_raw = captures[b].features.cpu().float()
        nat_features = (
            (F.linear(nat_raw, w_down) * w_post.unsqueeze(0)).contiguous()
            if fuse_mlp_modes
            else nat_raw
        )
        blended_features = (0.85 * cand_features + 0.15 * nat_features).contiguous()

        with torch.no_grad():
            y_raw = F.linear(nat_raw, w_down)
            s_mlp_sq = y_raw.square().mean(dim=1).clamp_min(1e-4)
            inv_s_sq = (s_mlp_sq.median() / s_mlp_sq).clamp(min=0.05, max=5.0)
        weights = (raw_weights * inv_s_sq).clamp(min=0.05, max=20.0)

        in_std = cand_inputs.std(dim=0, keepdim=True).clamp_min(1e-3)
        aug_inputs = [cand_inputs]
        aug_features = [blended_features]
        aug_weights = [weights]
        with torch.no_grad():
            for sigma in (0.04, 0.08, 0.12, 0.18, 0.25, 0.35):
                noise = torch.randn(
                    cand_inputs.shape,
                    generator=probe_gen,
                    dtype=torch.float32,
                ) * (sigma * in_std)
                x_plus = (cand_inputs + noise).contiguous()
                x_minus = (cand_inputs - noise).contiguous()
                f_plus = F.gelu(
                    F.linear(x_plus, source_mlp.gate_proj.weight),
                    approximate="tanh",
                ) * F.linear(x_plus, source_mlp.up_proj.weight)
                f_minus = F.gelu(
                    F.linear(x_minus, source_mlp.gate_proj.weight),
                    approximate="tanh",
                ) * F.linear(x_minus, source_mlp.up_proj.weight)
                if fuse_mlp_modes:
                    f_plus = F.linear(f_plus, w_down) * w_post.unsqueeze(0)
                    f_minus = F.linear(f_minus, w_down) * w_post.unsqueeze(0)
                df = 0.5 * (f_plus - f_minus)
                aug_inputs.append(x_plus)
                aug_features.append((blended_features + df).contiguous())
                aug_weights.append(0.30 * weights)
                aug_inputs.append(x_minus)
                aug_features.append((blended_features - df).contiguous())
                aug_weights.append(0.30 * weights)

        all_inputs = torch.cat(aug_inputs, dim=0)
        all_features = torch.cat(aug_features, dim=0)
        all_weights = torch.cat(aug_weights, dim=0)

        affine_weight, affine_bias, affine_metrics = fit_fisher_weighted_affine_base(
            inputs=all_inputs,
            modal_states=all_features,
            row_fisher_weights=all_weights,
            relative_ridge=0.015,
        )
        base_cand = cand_inputs @ affine_weight.T + affine_bias
        res_cand = (blended_features - base_cand).contiguous()
        centers, bases, scales, occupancy = _ewc_pullback_chart_geometry(
            cand_inputs,
            res_cand,
            weights,
            chart_count=CHART_COUNT,
            chart_rank=CHART_RANK,
        )
        memberships_0, coordinates_0 = _chart_membership_coordinates(
            cand_inputs,
            centers,
            bases,
            scales,
            temperature=CHART_TEMPERATURE,
        )
        dom_chart = memberships_0.argmax(dim=1)
        dom_bases = bases[dom_chart]
        coord_std = coordinates_0.std(dim=0, keepdim=True).clamp_min(1e-2)
        dom_std = coord_std[0, dom_chart, :]

        c_aug_in = [cand_inputs]
        c_aug_res = [res_cand]
        c_aug_w = [weights]
        with torch.no_grad():
            for c_sigma in (0.10, 0.20, 0.35):
                z_noise = torch.randn(
                    (cand_inputs.shape[0], CHART_RANK),
                    generator=probe_gen,
                    dtype=torch.float32,
                ) * (c_sigma * dom_std)
                dx_chart = torch.einsum("nr,nkr->nk", z_noise, dom_bases).contiguous()
                xp = (cand_inputs + dx_chart).contiguous()
                xm = (cand_inputs - dx_chart).contiguous()
                fp = F.gelu(
                    F.linear(xp, source_mlp.gate_proj.weight),
                    approximate="tanh",
                ) * F.linear(xp, source_mlp.up_proj.weight)
                fm = F.gelu(
                    F.linear(xm, source_mlp.gate_proj.weight),
                    approximate="tanh",
                ) * F.linear(xm, source_mlp.up_proj.weight)
                if fuse_mlp_modes:
                    fp = F.linear(fp, w_down) * w_post.unsqueeze(0)
                    fm = F.linear(fm, w_down) * w_post.unsqueeze(0)
                rp = fp - (xp @ affine_weight.T + affine_bias)
                rm = fm - (xm @ affine_weight.T + affine_bias)
                c_aug_in.extend([xp, xm])
                c_aug_res.extend([rp.contiguous(), rm.contiguous()])
                c_aug_w.extend([0.25 * weights, 0.25 * weights])

        chart_inputs_all = torch.cat(c_aug_in, dim=0)
        chart_res_all = torch.cat(c_aug_res, dim=0)
        chart_w_all = torch.cat(c_aug_w, dim=0)
        memb_chart_all, coord_chart_all = _chart_membership_coordinates(
            chart_inputs_all,
            centers,
            bases,
            scales,
            temperature=CHART_TEMPERATURE,
        )
        dense, dense_metrics = fit_fisher_weighted_dense_chart_edge_bank(
            memberships=memb_chart_all,
            coordinates=coord_chart_all,
            base_modal_residual=chart_res_all,
            row_fisher_weights=chart_w_all,
            relative_ridge=0.05,
        )
        local_encoder, mode_decoder, factor_metrics = factor_chart_edge_bank(
            dense,
            generator_rank,
        )
        local_encoder = (0.65 * local_encoder).contiguous()

        temp_layer = Gemma3FactorizedChartGeneratorLayer(
            affine_weight=affine_weight,
            affine_bias=affine_bias,
            mode_decoder_weight=mode_decoder_weight,
            chart_centers=centers,
            chart_bases=bases,
            chart_distance_scales=scales,
            local_generator_encoders=local_encoder,
            chart_mode_decoders=mode_decoder,
            chart_temperature=CHART_TEMPERATURE,
            model_fingerprint=model_fingerprint,
            layer_ordinal=b,
            fit_row_identity_sha256=row_sha256,
            compiled_prefix_catalog_sha256=prefix_catalog,
        )
        with torch.no_grad():
            local = temp_layer.execute_with_incoming(cand_inputs)

        t_metrics: dict[str, object] = {
            "source_event_count": 0,
            "selected_edge_count": 0,
        }
        layer_transports: list[JointGraphCrossBlockTransportEdge] = []
        if b >= 1:
            target_residual = blended_features - local.modal_states
            earlier_attention = {
                _attention_event_id(src): attention_packets[_attention_event_id(src)]
                for src in range(b)
            }
            fitted, t_metrics = fit_greedy_joint_cross_block_transports(
                source_packets=earlier_attention,
                target_memberships=local.memberships,
                target_mode_decoders=temp_layer.chart_mode_decoders,
                target_modal_residual=target_residual,
                row_fisher_weights=weights,
                maximum_incoming_sources_per_target_chart=(
                    maximum_incoming_attention_sources_per_target_chart
                ),
                relative_ridge=0.25,
                refinement_sweeps=0,
            )
            for edge_index, edge in enumerate(fitted):
                spec = JointGraphTransportSpec(
                    transport_id=(
                        f"transport.{edge.source_event_id}.to."
                        f"{_mlp_event_id(b)}."
                        f"c{edge.source_chart_index}.d{edge.target_chart_index}."
                        f"e{edge_index}"
                    ),
                    source_event_id=edge.source_event_id,
                    target_event_id=_mlp_event_id(b),
                    source_chart_index=edge.source_chart_index,
                    target_chart_index=edge.target_chart_index,
                )
                edge_obj = JointGraphCrossBlockTransportEdge(
                    spec=spec,
                    transport_weight=(0.20 * edge.transport_weight).contiguous(),
                )
                transports.append(edge_obj)
                layer_transports.append(edge_obj)

        # 640-D Linear Orthogonalization of chart + transport contribution
        with torch.no_grad():
            inc_valid = torch.zeros(
                (cand_inputs.shape[0], CHART_COUNT, generator_rank),
                dtype=torch.float32,
            )
            for edge_obj in layer_transports:
                src_pkt = attention_packets[edge_obj.spec.source_event_id][
                    :, edge_obj.spec.source_chart_index, :
                ]
                msg = edge_obj(src_pkt)
                inc_valid[:, edge_obj.spec.target_chart_index, :] += msg
            full_exec = temp_layer.execute_with_incoming(cand_inputs, inc_valid)
            nonlin_contrib = (full_exec.modal_states - base_cand).contiguous()
            da_w, da_b, _ = fit_fisher_weighted_affine_base(
                inputs=cand_inputs,
                modal_states=nonlin_contrib,
                row_fisher_weights=weights,
                relative_ridge=0.10,
            )
            affine_weight = (affine_weight - da_w).contiguous()
            affine_bias = (affine_bias - da_b).contiguous()

        layer = Gemma3FactorizedChartGeneratorLayer(
            affine_weight=affine_weight,
            affine_bias=affine_bias,
            mode_decoder_weight=mode_decoder_weight,
            chart_centers=centers,
            chart_bases=bases,
            chart_distance_scales=scales,
            local_generator_encoders=local_encoder,
            chart_mode_decoders=mode_decoder,
            chart_temperature=CHART_TEMPERATURE,
            model_fingerprint=model_fingerprint,
            layer_ordinal=b,
            fit_row_identity_sha256=row_sha256,
            compiled_prefix_catalog_sha256=prefix_catalog,
        )
        with torch.no_grad():
            local_final = layer.execute_with_incoming(cand_inputs)
        mlp_generators.append(layer)

        layer_metrics[str(b)] = {
            "row_count": len(rows),
            "fit_trajectory": "sequential_pullback_ewc_development_v1",
            "affine": affine_metrics,
            "dense_chart": dense_metrics,
            "factorization": factor_metrics,
            "hard_chart_occupancy": occupancy.tolist(),
            "local_modal_nrmse": _relative_rmse(
                local_final.modal_states,
                cand_features,
            ),
            "transport": t_metrics,
        }

        with torch.no_grad():
            offset = 0
            new_states_after_mlp: list[Tensor] = []
            for s_state, seq in zip(states_per_batch, seqs_per_batch):
                v_mask = seq.query_valid_mask.unsqueeze(-1)
                valid = seq.query_valid_mask
                n_tok = int(valid.sum().item())
                norm_m = mlp_mut_exact.input_norm(s_state)
                inc_batch = torch.zeros(
                    (*s_state.shape[:2], CHART_COUNT, generator_rank),
                    dtype=s_state.dtype,
                )
                inc_batch[valid] = inc_valid[offset : offset + n_tok]
                offset += n_tok
                out_m = layer.execute_with_incoming(norm_m, inc_batch).output
                d_mlp = mlp_mut_exact.output_norm(out_m)
                s_state = torch.where(
                    v_mask,
                    s_state + d_mlp.masked_fill(~v_mask, 0),
                    s_state,
                )
                new_states_after_mlp.append(s_state)
            states_per_batch = new_states_after_mlp

    if not transports:
        raise RuntimeError(
            "joint candidate fit found no positive cross-block transport"
        )
    return (
        tuple(ewc_attn_generators),
        JointCrossBlockFitResult(
            mlp_generators=tuple(mlp_generators),
            transports=tuple(transports),
            layer_metrics=layer_metrics,
            fit_row_identity_sha256=row_sha256,
        ),
    )


__all__ = [
    "JointCrossBlockFitResult",
    "fit_ewc_joint_cross_block_whole_model",
    "fit_joint_cross_block_mlp_charts",
    "fit_pullback_ewc_reduced_attention_generator",
    "fit_pullback_ewc_shared_tied_lexical_factor",
]
