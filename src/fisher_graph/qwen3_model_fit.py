"""Offline reduced-attention fitting with Qwen 3 execution semantics.

The input chart uses activation-weighted Q/K/V sensitivities and an anchored
reconstruction fit. The output factor minimizes weighted operator error.
Qwen has no post-attention RMSNorm, so its context rows must not be divided
by an operator-output RMS as in the Gemma-specific fitting path.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .gemma3_attention_generator import ReducedAttentionGeneratorCoefficients
from .qwen3_attention_generator import (
    Qwen3AttentionGenerator,
    compile_qwen3_reduced_attention_generator,
)


def _matrix(value: Tensor, *, name: str, width: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] == 0
        or value.shape[1] != width
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a finite nonempty matrix of width {width}")
    return value.detach().to(device="cpu", dtype=torch.float64)


@torch.no_grad()
def fit_qwen3_pullback_ewc_reduced_attention_generator(
    exact_gen: Qwen3AttentionGenerator,
    norm_inputs: Tensor,
    contexts: Tensor,
    *,
    op_outputs: Tensor | None = None,
    row_weights: Tensor | None = None,
    input_rank: int = 256,
    output_rank: int = 64,
    ewc_prior_weight: float = 0.005,
) -> Qwen3AttentionGenerator:
    """Fit a Qwen generator, preserving identity residual-delta normalization.

    ``contexts`` are the exact generator's pre-output-projection coordinates.
    ``op_outputs`` is accepted for capture-pipeline compatibility and validated
    when supplied; Qwen's identity delta norm needs no output-scale weighting.
    All fit algebra runs on detached CPU float64 rows; the returned coefficients
    use the exact generator's device and dtype.
    """

    if (
        not isinstance(exact_gen, Qwen3AttentionGenerator)
        or exact_gen.generator_kind != "exact_full_rank"
    ):
        raise TypeError("exact_gen must be an exact Qwen3AttentionGenerator")
    config = exact_gen.config
    if config.output_norm.kind != "identity":
        raise ValueError("Qwen reduced fitting requires identity output normalization")
    d, q_width = config.residual_width, config.query_width
    for name, rank, maximum in (
        ("input_rank", input_rank, d),
        ("output_rank", output_rank, min(q_width, d)),
    ):
        if type(rank) is not int or not 1 <= rank <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    if (
        isinstance(ewc_prior_weight, bool)
        or not isinstance(ewc_prior_weight, (int, float))
        or not math.isfinite(ewc_prior_weight)
        or ewc_prior_weight <= 0
    ):
        raise ValueError("ewc_prior_weight must be finite and positive")

    x = _matrix(norm_inputs, name="norm_inputs", width=d)
    c = _matrix(contexts, name="contexts", width=q_width)
    n = x.shape[0]
    if c.shape[0] != n:
        raise ValueError("norm_inputs and contexts must have the same row count")
    if op_outputs is not None:
        output = _matrix(op_outputs, name="op_outputs", width=d)
        if output.shape[0] != n:
            raise ValueError("op_outputs must have the same row count as norm_inputs")
    if row_weights is None:
        root_w = torch.ones((n, 1), dtype=torch.float64)
    else:
        if (
            not isinstance(row_weights, Tensor)
            or row_weights.shape != (n,)
            or not row_weights.is_floating_point()
            or not bool(torch.isfinite(row_weights).all())
            or bool((row_weights < 0).any())
            or float(row_weights.double().sum()) <= 0
        ):
            raise ValueError("row_weights must be finite nonnegative row-aligned weights with positive mass")
        weights = row_weights.detach().to(device="cpu", dtype=torch.float64)
        root_w = (weights / weights.mean()).sqrt().unsqueeze(1)
    x, c = x * root_w, c * root_w

    def cpu(value: Tensor) -> Tensor:
        return value.detach().to(device="cpu", dtype=torch.float64)

    w_q, w_k, w_v = (cpu(value) for value in (exact_gen.q_weight, exact_gen.k_weight, exact_gen.v_weight))
    w_o = cpu(exact_gen.output_decoder_weight)
    if config.qk_norm is not None:
        assert exact_gen.q_norm_weight is not None and exact_gen.k_norm_weight is not None
        offset = 1.0 if config.qk_norm.scale_parameterization == "unit_offset" else 0.0
        q_scale = (cpu(exact_gen.q_norm_weight) + offset).repeat(config.attention.query_heads)
        k_scale = (cpu(exact_gen.k_norm_weight) + offset).repeat(config.attention.key_value_heads)
        w_q_eff, w_k_eff = w_q * q_scale[:, None], w_k * k_scale[:, None]
    else:
        w_q_eff, w_k_eff = w_q, w_k
    # Each KV head serves a consecutive group of query heads. Tiling the
    # entire KV matrix instead would assign values to the wrong O head slice.
    repeats = config.attention.query_heads // config.attention.key_value_heads
    w_v_repeated = w_v.reshape(
        config.attention.key_value_heads, config.attention.head_dimension, d
    ).repeat_interleave(repeats, dim=0).reshape(q_width, d)
    w_v_eff = w_o @ w_v_repeated
    sensitivity = torch.zeros((d, d), dtype=torch.float64)
    for factor, multiplier in ((w_q_eff, 1.0), (w_k_eff, 1.0), (w_v_eff, 2.0)):
        gram = factor.T @ factor
        sensitivity += multiplier * gram / torch.trace(gram).clamp_min(1e-30)

    cov_x = x.T @ x / n
    reg_x = ewc_prior_weight * float(cov_x.diagonal().mean().clamp_min(1e-12))
    cov_x_reg = cov_x + reg_x * torch.eye(d, dtype=torch.float64)
    chart_metric = cov_x_reg @ sensitivity @ cov_x_reg
    _, vectors = torch.linalg.eigh((chart_metric + chart_metric.T) * 0.5)
    basis = vectors[:, -input_rank:].flip(1).contiguous()
    pivots = basis.abs().argmax(dim=0)
    signs = torch.where(basis[pivots, torch.arange(input_rank)] < 0, -1.0, 1.0)
    basis *= signs.unsqueeze(0)
    z = x @ basis
    cov_z = z.T @ z / n
    reg_z = ewc_prior_weight * float(cov_z.diagonal().mean().clamp_min(1e-12))
    reconstruction = torch.linalg.solve(
        cov_z + reg_z * torch.eye(input_rank, dtype=torch.float64),
        z.T @ x / n + reg_z * basis.T,
    )

    cov_c = c.T @ c / n
    reg_c = ewc_prior_weight * float(cov_c.diagonal().mean().clamp_min(1e-12))
    cov_c += reg_c * torch.eye(q_width, dtype=torch.float64)
    values, vectors = torch.linalg.eigh((cov_c + cov_c.T) * 0.5)
    roots = values.clamp_min(1e-30).sqrt()
    sqrt_c = (vectors * roots.unsqueeze(0)) @ vectors.T
    inv_sqrt_c = (vectors / roots.unsqueeze(0)) @ vectors.T
    u, singular, vh = torch.linalg.svd(sqrt_c @ w_o.T, full_matrices=False)
    root_s = singular[:output_rank].sqrt()
    output_encoder = (inv_sqrt_c @ (u[:, :output_rank] * root_s.unsqueeze(0))).T
    output_decoder = (root_s.unsqueeze(1) * vh[:output_rank]).T

    def source_format(value: Tensor) -> Tensor:
        return value.to(device=exact_gen.device, dtype=exact_gen.dtype).contiguous()

    return compile_qwen3_reduced_attention_generator(
        config,
        ReducedAttentionGeneratorCoefficients(
            input_norm_weight=exact_gen.input_norm_weight,
            input_basis=source_format(basis),
            q_decoder_weight=source_format(w_q @ reconstruction.T),
            k_decoder_weight=source_format(w_k @ reconstruction.T),
            v_decoder_weight=source_format(w_v @ reconstruction.T),
            output_encoder_weight=source_format(output_encoder),
            output_decoder_weight=source_format(output_decoder),
            output_norm_weight=exact_gen.output_norm_weight,
            q_norm_weight=exact_gen.q_norm_weight,
            k_norm_weight=exact_gen.k_norm_weight,
            q_bias=exact_gen.q_bias,
            k_bias=exact_gen.k_bias,
            v_bias=exact_gen.v_bias,
            o_bias=exact_gen.o_bias,
        ),
    )


__all__ = ["fit_qwen3_pullback_ewc_reduced_attention_generator"]
