"""Measure the first live L3-to-L4 hierarchy rung on frozen Gemma generators.

The existing causal map nominates layers 3 and 4 from finite suppression
responses.  Those responses are useful topology evidence, but they are not an
executable edge.  This development experiment collects the missing local
linear evidence without mutating or training the source model:

* activation moments and valid-position score-gradient Fisher blocks induced
  by summed prompt NLL for the L3/L4 generator inputs and outputs;
* Fisher-balanced low-rank factors for the exact frozen affine generators;
* an exact literal-zero topology tear between the L3 generated residual and
  the L4 MLP input;
* the prompt-conditioned mean-source reference path required by the centered
  modal coordinates; and
* signed, logical-lag-aware projected JVP kernels on that torn boundary.

The output is deliberately analysis-only.  It does not install the measured
edge into the ordinary L4 path, because that would count L3's influence twice.
The literal-zero path is topology evidence, not the compatible execution base
for a centered JVP: a future executor must supply the prompt-conditioned L4
input produced by the fitted mean L3 source.  It also does not claim a
cache-safe decoder replacement: the initial tear is cache-free prefill only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .causal_edge_jvp import CausalEdgeJVPFit, estimate_causal_edge_jvp
from .causal_modal_pair import bind_causal_modal_pair_plan
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_MAX_LENGTH,
    _safe_tokenized_stream_metadata,
    load_development_prompt_export,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    _bind_batch_example_ids,
)
from .generator_message_analysis import (
    GeneratorMessageCapturePlan,
    StreamingJointMessageMoments,
    iter_generator_message_score_gradient_rows,
)
from .modal_connectivity_modes import (
    CausalBoundaryTransfer,
    MessageMoments,
    ModalBoundaryPort,
    ModalConnectivityFactor,
    factor_modal_connectivity,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_hierarchy_experiment",
]


DEFAULT_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
DEFAULT_FIT_EXPORT = Path(
    ".local-runs/google--gemma-3-270m/dev-v9-a-fit-first40-export.json"
)
DEFAULT_PROBE_EXPORT = Path(
    ".local-runs/google--gemma-3-270m/dev-v9-a-fit-40-79-export.json"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-hierarchy-measurement-dev-v1.pt"
)
DEFAULT_RANKS = (32, 64, 128, 256, 320)
DEFAULT_EDGE_RANK = 64
DEFAULT_MAX_LAG = 4
DEFAULT_PROBE_COUNT = 16
DEFAULT_PROBE_SEQUENCES = 2
DEFAULT_RIDGE = 1e-6
_SCHEMA = "fisher_graph.gemma3_l3_l4_hierarchy_measurement_development"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-hierarchy-report:v1\0"
_SOURCE_SCOPE = "factorized_refit"
_LAYER_3 = 3
_LAYER_4 = 4
_X3 = "layer.3.mlp.normalized_input"
_Y3 = "layer.3.mlp.operator_output"
_L3_POST_ATTENTION = "layer.3.post_attention"
_X4 = "layer.4.mlp.normalized_input"
_Y4 = "layer.4.mlp.operator_output"
_SITES = (_X3, _Y3, _X4, _Y4)
_METRIC_RELATIVE_TRACE_FLOOR = 1e-6


def _progress(message: str) -> None:
    print(f"[gemma-l3-l4-hierarchy] {message}", file=sys.stderr, flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(b"fisher-graph:gemma3-l3-l4-tensor:v1\0")
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("ratio inputs must be finite")
    if denominator <= 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _cosine(left: Tensor, right: Tensor) -> float:
    left64 = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right64 = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if left64.shape != right64.shape:
        raise ValueError("cosine tensors must share shape")
    denominator = (
        torch.linalg.vector_norm(left64)
        * torch.linalg.vector_norm(right64)
    )
    if float(denominator.item()) == 0.0:
        return 1.0 if bool(torch.equal(left64, right64)) else 0.0
    return float(torch.dot(left64, right64).item() / denominator.item())


def _regularized_metric(value: Tensor) -> tuple[Tensor, float]:
    matrix = value.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("metric must be square")
    matrix = (matrix + matrix.T) * 0.5
    scale = max(
        float(matrix.trace().item()) / matrix.shape[0],
        float(matrix.abs().max().item())
        * torch.finfo(torch.float64).eps,
        torch.finfo(torch.float64).tiny,
    )
    floor = scale * _METRIC_RELATIVE_TRACE_FLOOR
    regularized = matrix + torch.eye(
        matrix.shape[0],
        dtype=torch.float64,
    ) * floor
    return regularized.contiguous(), floor


@dataclass(frozen=True, slots=True)
class _SiteMoments:
    mean: Tensor
    covariance: Tensor
    fisher: Tensor


@dataclass(frozen=True, slots=True)
class _CollectedMoments:
    sites: Mapping[str, _SiteMoments]
    activation_cross_y3_x4: Tensor
    fisher_cross_y3_x4: Tensor
    observations: int
    sequences: int
    mean_sequence_nll: float
    example_ids: tuple[str, ...]


def _collect_moments(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
) -> _CollectedMoments:
    widths = {
        name: adapter.activation_site(name).width
        for name in _SITES
    }
    if any(width is None for width in widths.values()):
        raise ValueError("L3/L4 sites must declare widths")
    canonical_widths = {
        name: int(width) for name, width in widths.items()
    }
    plan = GeneratorMessageCapturePlan(
        value_sites=_SITES,
        gradient_sites=_SITES,
        leaf_site=_X3,
    )
    joint = StreamingJointMessageMoments(
        _SITES,
        site_widths=canonical_widths,
    )
    rows = iter_generator_message_score_gradient_rows(
        adapter,
        batches,
        plan=plan,
        score_objective=CausalLanguageModelNLL(),
    )
    losses: list[float] = []
    example_ids: list[str] = []
    try:
        for row in rows:
            joint.update(row)
            losses.append(row.loss)
            example_ids.append(row.example_id)
            _progress(
                "moments: "
                f"{len(example_ids)}/{sum(b.batch_size for b in batches)} "
                "sequences"
            )
    finally:
        close = getattr(rows, "close", None)
        if callable(close):
            close()
    result = joint.finalize()
    sites = {
        _X3: _SiteMoments(
            mean=result.per_site_means[_X3],
            covariance=result.covariance_block(_X3, _X3),
            fisher=result.fisher_block(_X3, _X3),
        ),
        _Y3: _SiteMoments(
            mean=result.per_site_means[_Y3],
            covariance=result.covariance_block(_Y3, _Y3),
            fisher=result.fisher_block(_Y3, _Y3),
        ),
        _X4: _SiteMoments(
            mean=result.per_site_means[_X4],
            covariance=result.covariance_block(_X4, _X4),
            fisher=result.fisher_block(_X4, _X4),
        ),
        _Y4: _SiteMoments(
            mean=result.per_site_means[_Y4],
            covariance=result.covariance_block(_Y4, _Y4),
            fisher=result.fisher_block(_Y4, _Y4),
        ),
    }
    return _CollectedMoments(
        sites=sites,
        activation_cross_y3_x4=result.covariance_block(_Y3, _X4),
        fisher_cross_y3_x4=result.fisher_block(_Y3, _X4),
        observations=result.observations,
        sequences=result.sequences,
        mean_sequence_nll=math.fsum(losses) / len(losses),
        example_ids=tuple(example_ids),
    )


def _prepared_affine(
    adapter: Gemma3CausalLMAdapter,
    layer_ordinal: int,
) -> tuple[Tensor, Tensor, int, int]:
    layer = adapter.source_module(f"layer.{layer_ordinal}")
    mlp = getattr(layer, "mlp", None)
    input_projection = getattr(mlp, "input_projection", None)
    output_projection = getattr(mlp, "output_projection", None)
    if not isinstance(input_projection, nn.Linear) or not isinstance(
        output_projection,
        nn.Linear,
    ):
        raise TypeError(
            "active Gemma source must be the prepared full generator stack"
        )
    if (
        input_projection.bias is not None
        or input_projection.out_features != output_projection.in_features
        or input_projection.in_features != output_projection.out_features
        or output_projection.bias is None
    ):
        raise ValueError("prepared generator affine geometry drifted")
    input_weight = input_projection.weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    output_weight = output_projection.weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    matrix = (output_weight @ input_weight).contiguous()
    bias = output_projection.bias.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    return (
        matrix,
        bias,
        sum(parameter.numel() for parameter in mlp.parameters()),
        sum(
            child.weight.numel()
            for child in mlp.modules()
            if isinstance(child, nn.Linear)
        ),
    )


def _factor_generator(
    *,
    layer_ordinal: int,
    matrix: Tensor,
    bias: Tensor,
    input_site: str,
    output_site: str,
    moments: _CollectedMoments,
    source_level_sha256: str,
) -> ModalConnectivityFactor:
    width = int(matrix.shape[0])
    if matrix.shape != (width, width) or bias.shape != (width,):
        raise ValueError("generator affine must be square")
    input_port = ModalBoundaryPort(
        name=input_site,
        direction="input",
        causal_order=layer_ordinal,
        width=width,
        owner_id=f"gemma3.layer.{layer_ordinal}.generator",
    )
    output_port = ModalBoundaryPort(
        name=output_site,
        direction="output",
        causal_order=layer_ordinal,
        width=width,
        owner_id=f"gemma3.layer.{layer_ordinal}.generator",
    )
    transfer = CausalBoundaryTransfer(
        source_level_sha256=source_level_sha256,
        input_ports=(input_port,),
        output_ports=(output_port,),
        input_prefixes=((input_site,),),
        transfer_matrices=(matrix,),
        affine_offsets=(bias,),
    )
    reduction_id = "gemma3-l3-l4-prompt-nll-valid-position-rows"
    input_values = moments.sites[input_site]
    output_values = moments.sites[output_site]
    input_covariance, _ = _regularized_metric(input_values.covariance)
    input_fisher, _ = _regularized_metric(input_values.fisher)
    output_covariance, _ = _regularized_metric(output_values.covariance)
    output_fisher, _ = _regularized_metric(output_values.fisher)
    input_moment = MessageMoments(
        port=input_port,
        source_level_sha256=source_level_sha256,
        reduction_id=reduction_id,
        sample_count=moments.observations,
        mean=input_values.mean,
        covariance=input_covariance,
        fisher=input_fisher,
    )
    output_moment = MessageMoments(
        port=output_port,
        source_level_sha256=source_level_sha256,
        reduction_id=reduction_id,
        sample_count=moments.observations,
        mean=output_values.mean,
        covariance=output_covariance,
        fisher=output_fisher,
    )
    decomposition = factor_modal_connectivity(
        transfer,
        (input_moment,),
        (output_moment,),
        retained_ranks=width,
    )
    return decomposition.factors[0]


def _moment_summary(moments: _CollectedMoments) -> dict[str, object]:
    sites: dict[str, object] = {}
    for name in _SITES:
        values = moments.sites[name]
        covariance_norm = float(
            torch.linalg.matrix_norm(values.covariance).item()
        )
        fisher_norm = float(torch.linalg.matrix_norm(values.fisher).item())
        sites[name] = {
            "width": values.mean.numel(),
            "mean_l2": float(torch.linalg.vector_norm(values.mean).item()),
            "covariance_trace": float(values.covariance.trace().item()),
            "covariance_frobenius": covariance_norm,
            "raw_covariance_rank_upper_bound": min(
                max(moments.observations - 1, 0),
                values.mean.numel(),
            ),
            "fisher_trace": float(values.fisher.trace().item()),
            "fisher_frobenius": fisher_norm,
            "raw_fisher_rank_upper_bound": min(
                moments.observations,
                values.mean.numel(),
            ),
        }
    y3_cov = moments.sites[_Y3].covariance
    x4_cov = moments.sites[_X4].covariance
    y3_fisher = moments.sites[_Y3].fisher
    x4_fisher = moments.sites[_X4].fisher
    activation_cross_norm = float(
        torch.linalg.matrix_norm(moments.activation_cross_y3_x4).item()
    )
    fisher_cross_norm = float(
        torch.linalg.matrix_norm(moments.fisher_cross_y3_x4).item()
    )
    return {
        "observations": moments.observations,
        "independent_sequences": moments.sequences,
        "mean_summed_sequence_nll": moments.mean_sequence_nll,
        "sites": sites,
        "cross_port": {
            "ports": (_Y3, _X4),
            "activation_cross_covariance_frobenius": activation_cross_norm,
            "activation_cross_covariance_normalized_frobenius": _finite_ratio(
                activation_cross_norm,
                math.sqrt(
                    float(torch.linalg.matrix_norm(y3_cov).item())
                    * float(torch.linalg.matrix_norm(x4_cov).item())
                ),
            ),
            "score_gradient_cross_moment_frobenius": fisher_cross_norm,
            "score_gradient_cross_moment_normalized_frobenius": _finite_ratio(
                fisher_cross_norm,
                math.sqrt(
                    float(torch.linalg.matrix_norm(y3_fisher).item())
                    * float(torch.linalg.matrix_norm(x4_fisher).item())
                ),
            ),
            "block_diagonal_input_assumption_permitted": False,
        },
    }


def _factor_summary(factor: ModalConnectivityFactor) -> dict[str, object]:
    measured_mean = factor.output_mean
    return {
        "artifact_sha256": factor.artifact_sha256,
        "input_site": factor.input_ports[0].name,
        "output_site": factor.output_port.name,
        "width": factor.output_port.width,
        "effective_rank": factor.effective_rank,
        "input_support_rank": factor.input_support_ranks[0],
        "output_support_rank": factor.output_support_rank,
        "support_ranks_include_isotropic_metric_regularization": True,
        "total_weighted_energy": factor.total_weighted_energy,
        "singular_tolerance": factor.singular_tolerance,
        "output_mean_l2": float(torch.linalg.vector_norm(measured_mean).item()),
    }


class _TornL3L4Boundary:
    """Prompt-local exact prefill function from L3 MLP output to L4 MLP input."""

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        batch: CalibrationBatch,
    ) -> None:
        if batch.batch_size != 1:
            raise ValueError("torn boundary requires batch size one")
        with torch.no_grad():
            run = adapter.forward(
                batch.model_inputs,
                capture_sites=(
                    _X3,
                    _Y3,
                    _L3_POST_ATTENTION,
                    _X4,
                ),
            )
        if (
            run.sequence.phase != "prefill"
            or run.sequence.cache_state is not None
            or run.sequence.batch_size != 1
        ):
            raise ValueError("initial torn boundary is prefill-only")
        self.adapter = adapter
        self.sequence = run.sequence
        self.x3 = run.activations[_X3].detach()
        self.y3 = run.activations[_Y3].detach()
        self.l3_post_attention = run.activations[
            _L3_POST_ATTENTION
        ].detach()
        self.x4 = run.activations[_X4].detach()
        self.valid_mask = run.sequence.query_valid_mask.detach()
        self.logical_positions = run.sequence.logical_positions.detach()
        self._layer3 = adapter.source_module("layer.3")
        self._layer4 = adapter.source_module("layer.4")
        self._segment4 = adapter.segment("layer.4")
        for name in ("post_feedforward_layernorm",):
            if not isinstance(getattr(self._layer3, name, None), nn.Module):
                raise TypeError(f"Gemma L3 is missing {name}")
        if not isinstance(
            getattr(self._layer4, "pre_feedforward_layernorm", None),
            nn.Module,
        ):
            raise TypeError("Gemma L4 is missing pre_feedforward_layernorm")

        with torch.no_grad():
            replay = self(self.y3)
        difference = replay.float() - self.x4.float()
        scale = max(float(self.x4.float().abs().max().item()), 1.0)
        maximum = float(difference.abs().max().item())
        relative_l2 = _finite_ratio(
            float(torch.linalg.vector_norm(difference).item()),
            float(torch.linalg.vector_norm(self.x4.float()).item()),
        )
        tolerance = 2e-5 * scale
        if maximum > tolerance:
            raise RuntimeError(
                "local L3-to-L4 boundary replay differs from the full model: "
                f"max_abs={maximum:.6e}, tolerance={tolerance:.6e}"
            )
        self.identity_max_abs = maximum
        self.identity_relative_l2 = relative_l2

    def __call__(self, generated_l3: Tensor) -> Tensor:
        if (
            not isinstance(generated_l3, Tensor)
            or generated_l3.shape != self.y3.shape
            or generated_l3.dtype != self.y3.dtype
            or generated_l3.device != self.y3.device
            or not bool(torch.isfinite(generated_l3).all())
        ):
            raise ValueError("L3 generated residual differs from torn boundary")
        hidden3 = self.l3_post_attention + (
            self._layer3.post_feedforward_layernorm(generated_l3)
        )
        captures: list[Tensor] = []

        def capture(
            _module: nn.Module,
            _args: tuple[object, ...],
            output: object,
        ) -> object:
            if not isinstance(output, Tensor):
                raise TypeError("L4 pre-feedforward norm must return a Tensor")
            captures.append(output)
            return output

        handle = self._layer4.pre_feedforward_layernorm.register_forward_hook(
            capture
        )
        try:
            self.adapter.run_segment(
                self._segment4,
                hidden3,
                self.sequence,
            )
        finally:
            handle.remove()
        if len(captures) != 1:
            raise RuntimeError("L4 MLP input boundary was not captured once")
        return captures[0]


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    difference = actual.detach().to(device="cpu", dtype=torch.float64) - (
        expected.detach().to(device="cpu", dtype=torch.float64)
    )
    denominator = float(
        torch.linalg.vector_norm(
            expected.detach().to(device="cpu", dtype=torch.float64)
        ).item()
    )
    return _finite_ratio(
        float(torch.linalg.vector_norm(difference).item()),
        denominator,
    )


def _edge_measurement(
    boundary: _TornL3L4Boundary,
    *,
    factor3: ModalConnectivityFactor,
    factor4: ModalConnectivityFactor,
    rank: int,
    max_lag: int,
    probe_count: int,
    probe_seed: int,
    ridge: float,
) -> tuple[CausalEdgeJVPFit, dict[str, object]]:
    stage3 = factor3.truncate(rank)
    stage4 = factor4.truncate(rank)
    device = boundary.y3.device
    dtype = boundary.y3.dtype
    source_mean = stage3.output_mean.to(
        device=device,
        dtype=dtype,
    ).view(1, 1, -1).expand_as(boundary.y3)
    source_zero = torch.zeros_like(boundary.y3)
    source_modes = (
        boundary.x3
        - stage3.input_mean.to(device=device, dtype=dtype)
    ) @ stage3.restriction.to(device=device, dtype=dtype).T
    source_reconstruction = source_mean + (
        source_modes
        @ stage3.prolongation.to(device=device, dtype=dtype).T
    )
    with torch.no_grad():
        target_zero = boundary(source_zero)
        target_mean = boundary(source_mean)
        target_reconstruction = boundary(source_reconstruction)
    fit = estimate_causal_edge_jvp(
        boundary,
        baseline_source=source_mean,
        logical_positions=boundary.logical_positions,
        valid_mask=boundary.valid_mask,
        source_decoder=stage3.prolongation,
        target_encoder=stage4.restriction.T,
        max_lag=max_lag,
        probe_count=probe_count,
        probe_seed=probe_seed,
        ridge=ridge,
    )
    pair_plan = bind_causal_modal_pair_plan(
        stage3,
        stage4,
        fit,
        baseline_source=source_mean,
        logical_positions=boundary.logical_positions,
        valid_mask=boundary.valid_mask,
        x4_reference_torn_base_name=(
            "layer.4.mean_source.reference.torn_base"
        ),
    )
    pair_y3, pair_y4 = pair_plan.execute_factorized(
        boundary.x3,
        x4_mean_source_reference_torn_base=target_mean,
        logical_positions=boundary.logical_positions,
        valid_mask=boundary.valid_mask,
    )
    dense_y3, dense_y4 = pair_plan.execute_dense(
        boundary.x3,
        x4_mean_source_reference_torn_base=target_mean,
        logical_positions=boundary.logical_positions,
        valid_mask=boundary.valid_mask,
    )
    with torch.no_grad():
        control_y3 = stage3.execute((boundary.x3,))
        control_x4 = boundary(control_y3)
        control_y4 = stage4.execute((control_x4,))
    predicted_variable = fit.execute(
        source_modes,
        logical_positions=boundary.logical_positions,
        valid_mask=boundary.valid_mask,
    )
    target_variable = (
        target_reconstruction - target_mean
    ) @ stage4.restriction.to(device=device, dtype=dtype).T
    valid = boundary.valid_mask.to(device=device, dtype=torch.bool)
    predicted_valid = predicted_variable[valid]
    target_valid = target_variable[valid]
    affine_modal = (
        target_mean - target_zero
    ) @ stage4.restriction.to(device=device, dtype=dtype).T
    native_modal = (
        boundary.x4 - target_mean
    ) @ stage4.restriction.to(device=device, dtype=dtype).T
    kernel_energy = fit.kernel.square().sum(dim=(1, 2))
    total_kernel_energy = float(kernel_energy.sum().item())
    lag_fractions = tuple(
        _finite_ratio(float(value.item()), total_kernel_energy)
        for value in kernel_energy
    )
    diagnostics = {
        "edge_fit": fit.metadata(),
        "pair_binding": {
            "stage3_factor_sha256": stage3.artifact_sha256,
            "stage4_factor_sha256": stage4.artifact_sha256,
            "edge_fit_sha256": fit.artifact_sha256,
            "boundary_contract_sha256": (
                pair_plan.boundary_contract.artifact_sha256
            ),
            "pair_plan_sha256": pair_plan.artifact_sha256,
            "mean_source_reference_sha256": _tensor_sha256(target_mean),
            "reference_provider": "live_frozen_source_boundary_oracle",
            "reference_provider_compiled": False,
            "reference_provider_authenticated_at_runtime": False,
            "full_model_executor": False,
        },
        "boundary_identity_max_abs": boundary.identity_max_abs,
        "boundary_identity_relative_l2": boundary.identity_relative_l2,
        "topology_tear_reference": "literal_zero_L3_generated_residual",
        "centered_edge_reference_base": (
            "prompt_conditioned_fit_mean_L3_generated_residual"
        ),
        "linearization_reference": "fit_split_mean_L3_generated_residual",
        "source_rank_reconstruction_relative_l2": _relative_error(
            source_reconstruction,
            boundary.y3,
        ),
        "target_rank_reconstruction_path_relative_l2": _relative_error(
            target_reconstruction,
            boundary.x4,
        ),
        "pair_stage3_binding_relative_error": _relative_error(
            pair_y3,
            control_y3,
        ),
        "pair_stage3_binding_max_abs": float(
            (pair_y3.float() - control_y3.float()).abs().max().item()
        ),
        "pair_dense_factorized_stage3_max_abs": float(
            (pair_y3.float() - dense_y3.float()).abs().max().item()
        ),
        "pair_dense_factorized_stage4_max_abs": float(
            (pair_y4.float() - dense_y4.float()).abs().max().item()
        ),
        "pair_comparison_target": (
            "local_truncated_factor_control_with_live_frozen_boundary"
        ),
        "oracle_base_pair_vs_local_factor_control_y4_relative_error": (
            _relative_error(
                pair_y4[valid],
                control_y4[valid],
            )
        ),
        "oracle_base_pair_vs_local_factor_control_y4_cosine": _cosine(
            pair_y4[valid],
            control_y4[valid],
        ),
        "oracle_base_pair_vs_local_factor_control_y4_max_abs": float(
            (
                pair_y4[valid].float()
                - control_y4[valid].float()
            ).abs().max().item()
        ),
        "finite_variable_modal_relative_error": _relative_error(
            predicted_valid,
            target_valid,
        ),
        "finite_variable_modal_cosine": _cosine(
            predicted_valid,
            target_valid,
        ),
        "zero_to_mean_reference_modal_l2": float(
            torch.linalg.vector_norm(affine_modal[valid]).item()
        ),
        "native_minus_mean_target_modal_l2": float(
            torch.linalg.vector_norm(native_modal[valid]).item()
        ),
        "kernel_squared_energy_by_lag": tuple(
            float(value.item()) for value in kernel_energy
        ),
        "kernel_squared_energy_fraction_by_lag": lag_fractions,
        "lag_zero_energy_fraction": lag_fractions[0],
        "positive_lag_energy_fraction": sum(lag_fractions[1:]),
        "ordinary_L4_path_used_as_reference_base": False,
        "literal_zero_path_used_as_centered_edge_base": False,
        "mean_source_reference_path_used_as_centered_edge_base": True,
        "edge_added_to_ordinary_L4_path": False,
    }
    return fit, diagnostics


def _rank_curve(
    *,
    factor3: ModalConnectivityFactor,
    factor4: ModalConnectivityFactor,
    ranks: Sequence[int],
    max_lag: int,
    flat_pair_parameters: int,
    flat_pair_macs: int,
    full_stack_parameters: int,
    full_stack_macs: int,
    whole_model_parameters: int,
) -> tuple[dict[str, object], ...]:
    rows = []
    for rank in ranks:
        stage3 = factor3.truncate(rank)
        stage4 = factor4.truncate(rank)
        local_width_sum = (
            stage3.input_mean.numel()
            + stage3.output_mean.numel()
            + stage4.input_mean.numel()
            + stage4.output_mean.numel()
        )
        edge_scalars = (max_lag + 1) * rank * rank
        factor_scalars = rank * local_width_sum
        stored_mean_scalars = local_width_sum
        candidate_parameters = (
            factor_scalars + edge_scalars + stored_mean_scalars
        )
        candidate_macs = factor_scalars + edge_scalars
        candidate_full_stack_parameters = (
            full_stack_parameters
            - flat_pair_parameters
            + candidate_parameters
        )
        candidate_full_stack_macs = (
            full_stack_macs - flat_pair_macs + candidate_macs
        )
        candidate_whole_model_parameters = (
            whole_model_parameters
            - flat_pair_parameters
            + candidate_parameters
        )
        energy3 = _finite_ratio(
            stage3.retained_weighted_energy,
            stage3.total_weighted_energy,
        )
        energy4 = _finite_ratio(
            stage4.retained_weighted_energy,
            stage4.total_weighted_energy,
        )
        rows.append(
            {
                "rank": rank,
                "lag_count": max_lag + 1,
                "l3_retained_weighted_energy_fraction": energy3,
                "l4_retained_weighted_energy_fraction": energy4,
                "minimum_retained_weighted_energy_fraction": min(
                    energy3,
                    energy4,
                ),
                "weighted_energy_is_fidelity_metric": False,
                "centering_folded_into_bias": False,
                "stored_mean_scalars": stored_mean_scalars,
                "candidate_pair_parameters": candidate_parameters,
                "candidate_pair_parameter_fraction_of_flat": _finite_ratio(
                    candidate_parameters,
                    flat_pair_parameters,
                ),
                "candidate_pair_macs_per_token": candidate_macs,
                "candidate_pair_mac_fraction_of_flat": _finite_ratio(
                    candidate_macs,
                    flat_pair_macs,
                ),
                "lag_edge_parameters": edge_scalars,
                "peak_live_modal_scalars_per_token": rank,
                "candidate_full_stack_parameters": (
                    candidate_full_stack_parameters
                ),
                "candidate_full_stack_parameter_fraction": _finite_ratio(
                    candidate_full_stack_parameters,
                    full_stack_parameters,
                ),
                "candidate_full_stack_macs_per_token": (
                    candidate_full_stack_macs
                ),
                "candidate_full_stack_mac_fraction": _finite_ratio(
                    candidate_full_stack_macs,
                    full_stack_macs,
                ),
                "candidate_whole_model_parameters": (
                    candidate_whole_model_parameters
                ),
                "whole_model_parameter_fraction_of_flat_source": _finite_ratio(
                    candidate_whole_model_parameters,
                    whole_model_parameters,
                ),
                "accounting_status": (
                    "logical_shape_only_until_reference_base_executor_is_built"
                ),
            }
        )
    return tuple(rows)


def _factor_state(factor: ModalConnectivityFactor) -> dict[str, object]:
    factor.validate_integrity()
    return {
        "artifact_sha256": factor.artifact_sha256,
        "input_site": factor.input_ports[0].name,
        "output_site": factor.output_port.name,
        "singular_values": factor.singular_values.clone(),
        "singular_tolerance": factor.singular_tolerance,
        "restriction": factor.restriction.clone(),
        "prolongation": factor.prolongation.clone(),
        "input_mean": factor.input_mean.clone(),
        "output_mean": factor.output_mean.clone(),
        "input_support_rank": factor.input_support_ranks[0],
        "output_support_rank": factor.output_support_rank,
    }


def _moments_state(moments: _CollectedMoments) -> dict[str, object]:
    return {
        "observations": moments.observations,
        "sequences": moments.sequences,
        "mean_sequence_nll": moments.mean_sequence_nll,
        "example_ids": moments.example_ids,
        "sites": {
            name: {
                "mean": moments.sites[name].mean.clone(),
                "covariance": moments.sites[name].covariance.clone(),
                "fisher": moments.sites[name].fisher.clone(),
            }
            for name in _SITES
        },
        "activation_cross_y3_x4": (
            moments.activation_cross_y3_x4.clone()
        ),
        "fisher_cross_y3_x4": moments.fisher_cross_y3_x4.clone(),
    }


def _atomic_torch_save(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(value: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_configuration(
    *,
    output: Path,
    ranks: Sequence[int],
    edge_rank: int,
    max_lag: int,
    probe_count: int,
    probe_sequences: int,
    ridge: float,
    fit_limit: int | None,
) -> tuple[int, ...]:
    if output.suffix != ".pt":
        raise ValueError("hierarchy measurement output must use .pt")
    report = output.with_suffix(".json")
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite hierarchy measurement")
    rank_tuple = tuple(ranks)
    if (
        not rank_tuple
        or any(type(rank) is not int or not 1 <= rank <= 640 for rank in rank_tuple)
        or rank_tuple != tuple(sorted(set(rank_tuple)))
    ):
        raise ValueError("ranks must be unique increasing integers in [1, 640]")
    if type(edge_rank) is not int or edge_rank not in rank_tuple:
        raise ValueError("edge_rank must be one of ranks")
    for value, label, minimum in (
        (max_lag, "max_lag", 0),
        (probe_count, "probe_count", 1),
        (probe_sequences, "probe_sequences", 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    if (
        isinstance(ridge, bool)
        or not isinstance(ridge, (int, float))
        or not math.isfinite(float(ridge))
        or float(ridge) < 0.0
    ):
        raise ValueError("ridge must be finite and nonnegative")
    if fit_limit is not None and (
        type(fit_limit) is not int or fit_limit <= 0
    ):
        raise ValueError("fit_limit must be positive when supplied")
    return rank_tuple


def run_gemma3_l3_l4_hierarchy_experiment(
    *,
    revision: str = DEFAULT_REVISION,
    fit_export_path: Path | str = DEFAULT_FIT_EXPORT,
    probe_export_path: Path | str = DEFAULT_PROBE_EXPORT,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = 1,
    fit_limit: int | None = None,
    ranks: Sequence[int] = DEFAULT_RANKS,
    edge_rank: int = DEFAULT_EDGE_RANK,
    max_lag: int = DEFAULT_MAX_LAG,
    probe_count: int = DEFAULT_PROBE_COUNT,
    probe_sequences: int = DEFAULT_PROBE_SEQUENCES,
    ridge: float = DEFAULT_RIDGE,
    probe_seed: int = 20260727,
) -> dict[str, object]:
    """Run a development-only Fisher/JVP rung on frozen L3/L4 generators."""

    destination = Path(output)
    rank_tuple = _validate_configuration(
        output=destination,
        ranks=ranks,
        edge_rank=edge_rank,
        max_lag=max_lag,
        probe_count=probe_count,
        probe_sequences=probe_sequences,
        ridge=ridge,
        fit_limit=fit_limit,
    )
    if (
        not isinstance(revision, str)
        or not revision
        or type(max_length) is not int
        or max_length < 2
        or type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
        or type(probe_seed) is not int
        or probe_seed < 0
    ):
        raise ValueError("model/tokenization/probe configuration is invalid")

    base_path = Path(base_artifact_path)
    refit_path = Path(refit_artifact_path)
    _progress("artifacts: authenticate the frozen full-stack refit")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_path,
        refit_path,
    )
    replacements = catalog.replacements
    source_level_sha256 = _require_sha256(
        catalog.refit_scientific_payload_sha256,
        label="refit scientific payload",
    )
    model_metadata = dict(catalog.model_metadata)
    catalog_binding = {
        "base_tensor_file": str(base_path),
        "base_tensor_file_sha256": catalog.base_artifact_file_sha256,
        "base_scientific_payload_sha256": (
            catalog.base_scientific_payload_sha256
        ),
        "refit_tensor_file": str(refit_path),
        "refit_tensor_file_sha256": catalog.refit_artifact_file_sha256,
        "refit_scientific_payload_sha256": (
            catalog.refit_scientific_payload_sha256
        ),
        "source_model_sha256": catalog.source_model_sha256,
        "generator_plan_sha256s": catalog.generator_plan_sha256s,
    }
    if (
        model_metadata.get("model_id") != model_id
        or model_metadata.get("requested_revision") != revision
        or model_metadata.get("resolved_commit") != revision
        or model_metadata.get("local_files_only") is not True
    ):
        raise ValueError("requested model differs from the frozen refit")

    fit_export = load_development_prompt_export(fit_export_path)
    probe_export = load_development_prompt_export(probe_export_path)
    fit_count = (
        len(fit_export.prompts)
        if fit_limit is None
        else min(fit_limit, len(fit_export.prompts))
    )
    if fit_count <= 0 or probe_sequences > len(probe_export.prompts):
        raise ValueError("development exports do not cover requested prompts")
    fit_prompts = fit_export.prompts[:fit_count]
    fit_ids = fit_export.prompt_sha256s[:fit_count]
    probe_prompts = probe_export.prompts[:probe_sequences]
    probe_ids = probe_export.prompt_sha256s[:probe_sequences]
    overlap = set(fit_ids) & set(probe_ids)
    if overlap:
        raise ValueError("fit and JVP probe prompts must be content-disjoint")

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load the pinned local Gemma checkpoint")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    source_model_sha256 = adapter.model_fingerprint()
    if source_model_sha256 != catalog_binding["source_model_sha256"]:
        raise ValueError("live Gemma fingerprint differs from the refit")

    _progress("runtime: prepare the exact frozen 18-generator source")
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_SOURCE_SCOPE: replacements},
    )
    del replacements, catalog
    gc.collect()

    try:
        switcher.switch(_SOURCE_SCOPE)
        full_stack_accounting = switcher.scope_accounting[_SOURCE_SCOPE]
        whole_model_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        fit_batches, fit_stream = _materialize_split(
            tokenizer,
            fit_prompts,
            split_name="l3_l4_hierarchy_fit",
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            device=device,
        )
        fit_batches = _bind_batch_example_ids(fit_batches, fit_ids)
        probe_batches, probe_stream = _materialize_split(
            tokenizer,
            probe_prompts,
            split_name="l3_l4_hierarchy_jvp_probe",
            max_length=max_length,
            tokenization_batch_size=1,
            device=device,
        )
        probe_batches = _bind_batch_example_ids(probe_batches, probe_ids)

        _progress("Fisher: stream joint L3/L4 activation and score moments")
        moments = _collect_moments(adapter, fit_batches)
        if moments.example_ids != tuple(fit_ids):
            raise RuntimeError("fit example order drifted")

        _progress("factors: balance the two exact affine generators")
        matrix3, bias3, pair_params3, pair_macs3 = _prepared_affine(
            adapter,
            _LAYER_3,
        )
        matrix4, bias4, pair_params4, pair_macs4 = _prepared_affine(
            adapter,
            _LAYER_4,
        )
        factor3 = _factor_generator(
            layer_ordinal=_LAYER_3,
            matrix=matrix3,
            bias=bias3,
            input_site=_X3,
            output_site=_Y3,
            moments=moments,
            source_level_sha256=source_level_sha256,
        )
        factor4 = _factor_generator(
            layer_ordinal=_LAYER_4,
            matrix=matrix4,
            bias=bias4,
            input_site=_X4,
            output_site=_Y4,
            moments=moments,
            source_level_sha256=source_level_sha256,
        )
        del matrix3, matrix4, bias3, bias4
        flat_pair_parameters = pair_params3 + pair_params4
        flat_pair_macs = pair_macs3 + pair_macs4
        curve = _rank_curve(
            factor3=factor3,
            factor4=factor4,
            ranks=rank_tuple,
            max_lag=max_lag,
            flat_pair_parameters=flat_pair_parameters,
            flat_pair_macs=flat_pair_macs,
            full_stack_parameters=(
                full_stack_accounting.learned_parameter_count
            ),
            full_stack_macs=full_stack_accounting.linear_macs_per_token,
            whole_model_parameters=whole_model_parameters,
        )

        _progress(
            "JVP: fit signed logical-lag edges around mean-source references"
        )
        edge_fits: list[CausalEdgeJVPFit] = []
        edge_diagnostics: list[dict[str, object]] = []
        for index, batch in enumerate(probe_batches):
            boundary = _TornL3L4Boundary(adapter, batch)
            fit, diagnostic = _edge_measurement(
                boundary,
                factor3=factor3,
                factor4=factor4,
                rank=edge_rank,
                max_lag=max_lag,
                probe_count=probe_count,
                probe_seed=probe_seed + index,
                ridge=float(ridge),
            )
            edge_fits.append(fit)
            diagnostic["example_id"] = probe_ids[index]
            edge_diagnostics.append(diagnostic)
            _progress(
                f"JVP: {index + 1}/{len(probe_batches)} prompt-local kernels"
            )

        kernel_cosines = tuple(
            _cosine(edge_fits[left].kernel, edge_fits[right].kernel)
            for left in range(len(edge_fits))
            for right in range(left + 1, len(edge_fits))
        )
        mean_kernel = torch.stack(
            tuple(fit.kernel for fit in edge_fits),
            dim=0,
        ).mean(dim=0)
        mean_centered = torch.stack(
            tuple(fit.kernel - mean_kernel for fit in edge_fits),
            dim=0,
        )
        between_prompt_variation = (
            None
            if len(edge_fits) < 2
            else _finite_ratio(
                float(torch.linalg.vector_norm(mean_centered).item()),
                float(
                    torch.linalg.vector_norm(
                        torch.stack(tuple(fit.kernel for fit in edge_fits))
                    ).item()
                ),
            )
        )

        safe_moments = _moment_summary(moments)
        safe_factors = {
            "layer_3": _factor_summary(factor3),
            "layer_4": _factor_summary(factor4),
        }
        pair_control_relative_errors = tuple(
            float(
                diagnostic[
                    "oracle_base_pair_vs_local_factor_control_y4_relative_error"
                ]
            )
            for diagnostic in edge_diagnostics
        )
        pair_control_cosines = tuple(
            float(
                diagnostic[
                    "oracle_base_pair_vs_local_factor_control_y4_cosine"
                ]
            )
            for diagnostic in edge_diagnostics
        )
        safe_edge = {
            "selected_rank": edge_rank,
            "max_lag": max_lag,
            "probe_count_per_sequence": probe_count,
            "jvp_fit_residual_is_in_sample": True,
            "heldout_jvp_direction_validation": False,
            "prompt_local_fits": tuple(edge_diagnostics),
            "pairwise_kernel_cosines": kernel_cosines,
            "minimum_pairwise_kernel_cosine": (
                None if not kernel_cosines else min(kernel_cosines)
            ),
            "mean_pairwise_kernel_cosine": (
                None
                if not kernel_cosines
                else math.fsum(kernel_cosines) / len(kernel_cosines)
            ),
            "between_prompt_kernel_variation_fraction": (
                between_prompt_variation
            ),
            "oracle_base_pair_control": {
                "comparison_target": (
                    "local_truncated_factor_control_with_live_frozen_boundary"
                ),
                "probe_count": len(edge_diagnostics),
                "mean_relative_error": (
                    math.fsum(pair_control_relative_errors)
                    / len(pair_control_relative_errors)
                ),
                "maximum_relative_error": max(
                    pair_control_relative_errors
                ),
                "mean_cosine": (
                    math.fsum(pair_control_cosines)
                    / len(pair_control_cosines)
                ),
                "minimum_cosine": min(pair_control_cosines),
                "deployment_fidelity_accepted": False,
                "acceptance_reason": (
                    "development diagnostic only; no predeclared fidelity "
                    "gate or source-authoritative full-model comparison"
                ),
            },
            "global_constant_edge_accepted": False,
            "acceptance_reason": (
                "prompt-local analysis-plan evidence only; no compiled "
                "reference-base provider, held-out direction validation, "
                "finite-displacement fidelity gate, or family-disjoint guard"
            ),
        }
        artifact: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "scientific_status": {
                "outcome": (
                    "l3_l4_hierarchy_measurement_with_oracle_pair_control"
                ),
                "development_only": True,
                "source_weights_changed": False,
                "joint_cross_port_moments_measured": True,
                "zero_source_edge_tear_measured": True,
                "mean_source_reference_path_measured": True,
                "signed_lag_aware_jvp_measured": True,
                "analysis_pair_plan_bound_and_executed": True,
                "oracle_base_pair_local_factor_control_built": True,
                "compiled_reference_base_provider_built": False,
                "full_model_replacement_candidate_built": False,
                "authorizes_replacement_execution": False,
                "authorizes_compilation": False,
                "compression_claim": False,
                "latency_claim": False,
                "cached_decode_claim": False,
            },
            "binding": catalog_binding,
            "protocol": {
                "source_scope": _SOURCE_SCOPE,
                "fit_sites": _SITES,
                "detached_leaf_site": _X3,
                "score": "summed_hard_target_next_token_nll",
                "row_normalizer": "valid_activation_positions",
                "fit_prompt_ids": fit_ids,
                "probe_prompt_ids": probe_ids,
                "fit_family_ids": fit_export.family_ids[:fit_count],
                "probe_family_ids": probe_export.family_ids[:probe_sequences],
                "fit_and_probe_content_disjoint": True,
                "fit_and_probe_family_disjoint": (
                    not bool(
                        set(fit_export.family_ids[:fit_count])
                        & set(
                            probe_export.family_ids[:probe_sequences]
                        )
                    )
                ),
                "prefill_only": True,
                "cache_state": "none",
                "tear_source_site": _Y3,
                "tear_target_site": _X4,
                "topology_tear_base": "literal_zero_source_path",
                "centered_edge_reference_base": (
                    "prompt_conditioned_fit_mean_source_path"
                ),
                "rank_ladder": rank_tuple,
                "edge_rank": edge_rank,
                "factor_metric_regularization": {
                    "kind": "isotropic_absolute_floor",
                    "floor": (
                        "1e-6_times_mean_diagonal_with_machine_floor"
                    ),
                    "relative_trace_floor": (
                        _METRIC_RELATIVE_TRACE_FLOOR
                    ),
                    "raw_moments_preserved_separately": True,
                },
                "logical_lags": tuple(range(max_lag + 1)),
                "randomized_exact_jvp_probes_per_sequence": probe_count,
                "ridge": float(ridge),
            },
            "moments": _moments_state(moments),
            "factors": {
                "layer_3": _factor_state(factor3),
                "layer_4": _factor_state(factor4),
            },
            "edge_jvp_states": tuple(fit.state_dict() for fit in edge_fits),
            "mean_prompt_local_kernel": mean_kernel,
            "safe_analysis": {
                "moments": safe_moments,
                "factors": safe_factors,
                "edge": safe_edge,
                "rate_curve": curve,
            },
            "safety": {
                "contains_source_model_state_dict": False,
                "contains_tokenizer": False,
                "contains_prompt_text": False,
                "contains_token_ids": False,
                "contains_activation_rows": False,
                "contains_score_gradient_rows": False,
                "contains_executable_low_rank_factors": True,
                "artifact_must_remain_outside_git": True,
            },
        }

        _progress("artifact: write ignored tensor state and source-safe report")
        _atomic_torch_save(artifact, destination)
        tensor_file_sha256 = _file_sha256(destination)
        report_payload: dict[str, object] = {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "scientific_status": artifact["scientific_status"],
            "model": {
                "model_id": model_id,
                "requested_revision": revision,
                "resolved_commit": revision,
                "source_model_sha256": source_model_sha256,
                "hidden_width": factor3.output_port.width,
                "transformer_layers": len(adapter.layers),
                "local_files_only": True,
            },
            "binding": catalog_binding,
            "protocol": artifact["protocol"],
            "tokenized_streams": {
                "fit": _safe_tokenized_stream_metadata(fit_stream),
                "probe": _safe_tokenized_stream_metadata(probe_stream),
            },
            "analysis": {
                "moments": safe_moments,
                "factors": safe_factors,
                "edge": safe_edge,
                "rate_curve": curve,
                "resource_baseline": {
                    "flat_l3_l4_parameters": flat_pair_parameters,
                    "flat_l3_l4_macs_per_token": flat_pair_macs,
                    "flat_full_stack_parameters": (
                        full_stack_accounting.learned_parameter_count
                    ),
                    "flat_full_stack_macs_per_token": (
                        full_stack_accounting.linear_macs_per_token
                    ),
                    "flat_whole_model_parameters": whole_model_parameters,
                },
            },
            "artifact": {
                "tensor_file": str(destination),
                "tensor_file_sha256": tensor_file_sha256,
                "tensor_file_bytes": destination.stat().st_size,
                "report_file": str(destination.with_suffix(".json")),
                "committable": False,
                "contains_executable_low_rank_factors": True,
            },
            "interpretation": {
                "what_is_proven": (
                    "joint Fisher/covariance and signed prompt-local causal "
                    "transport can be measured, bound to exact factors and "
                    "a JVP artifact, and executed against a frozen-model "
                    "mean-source oracle as a local analysis control"
                ),
                "what_is_not_proven": (
                    "no global edge, compiled mean-source reference-base "
                    "provider, full-model executor, downstream fidelity "
                    "guard, compression, or speed claim exists yet"
                ),
                "next_gate": (
                    "validate unseen JVP directions, replace or augment the "
                    "first-order edge with a finite-displacement conditional "
                    "correction, compile and authenticate the prompt-"
                    "conditioned mean-source provider, then freeze rank on "
                    "disjoint selection and evaluate a family-disjoint "
                    "source-authoritative shadow"
                ),
            },
        }
        report_payload["report_sha256"] = _report_sha256(report_payload)
        _atomic_json_save(report_payload, destination.with_suffix(".json"))
    finally:
        switcher.close()

    if adapter.model_fingerprint() != source_model_sha256:
        raise RuntimeError("hierarchy experiment mutated the source model")
    if (
        _file_sha256(base_path)
        != catalog_binding["base_tensor_file_sha256"]
        or _file_sha256(refit_path)
        != catalog_binding["refit_tensor_file_sha256"]
    ):
        raise RuntimeError("frozen generator artifacts changed during analysis")
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure joint Fisher modes and a signed torn L3-to-L4 edge on "
            "the frozen Gemma generator stack."
        )
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--fit-export", type=Path, default=DEFAULT_FIT_EXPORT)
    parser.add_argument(
        "--probe-export",
        type=Path,
        default=DEFAULT_PROBE_EXPORT,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--tokenization-batch-size", type=int, default=1)
    parser.add_argument("--fit-limit", type=int)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=DEFAULT_RANKS,
    )
    parser.add_argument("--edge-rank", type=int, default=DEFAULT_EDGE_RANK)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument(
        "--probe-count",
        type=int,
        default=DEFAULT_PROBE_COUNT,
    )
    parser.add_argument(
        "--probe-sequences",
        type=int,
        default=DEFAULT_PROBE_SEQUENCES,
    )
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--probe-seed", type=int, default=20260727)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_hierarchy_experiment(
        revision=arguments.revision,
        fit_export_path=arguments.fit_export,
        probe_export_path=arguments.probe_export,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        output=arguments.output,
        model_id=arguments.model,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        fit_limit=arguments.fit_limit,
        ranks=arguments.ranks,
        edge_rank=arguments.edge_rank,
        max_lag=arguments.max_lag,
        probe_count=arguments.probe_count,
        probe_sequences=arguments.probe_sequences,
        ridge=arguments.ridge,
        probe_seed=arguments.probe_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
