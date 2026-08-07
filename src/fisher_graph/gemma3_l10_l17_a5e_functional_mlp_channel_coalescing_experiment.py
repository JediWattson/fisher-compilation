"""Development A5e Fisher/Jacobian physical-channel coalescing ladder.

The experiment keeps the native Gemma checkpoint immutable, learns disjoint
donor/survivor channel pairs from fit-only activation Fisher signatures, and
materializes genuinely narrower gate/up/down projections for layers 10 and 17.
It compares those candidates with deletion of the exact same donor channels.

The maximum-rate coalesced error also supplies one frozen low-rank post-RMSNorm
residual.  That identical residual is evaluated on both the compact candidate
and the fully intact native MLP.  The native control executes every original
gate/up/down channel and receives no compression credit.

This is a prompt-disjoint development experiment, not held-out confirmation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .adapters import Gemma3CausalLMAdapter
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_all_mode_generator_graph_executor import (
    Gemma3AllModeGeneratorGraphMLP,
    compile_chart_conditioned_all_mode_generator_graph,
    compile_exact_all_mode_generator_graph,
    compile_fitted_affine_all_mode_generator_graph,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_protocol import (
    A5E_MERGE_RATE_LADDER,
    build_a5e_functional_mlp_channel_coalescing_protocol,
)


A5E_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5e_functional_mlp_channel_"
    "coalescing.experiment.v4"
)
A5E_EXPERIMENT_FORMAT_VERSION = 4
DEFAULT_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
DEFAULT_PANEL_PATH = Path("examples/gemma3_downstream_retention_v1.json")
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a5e-functional-channel-coalescing-dev-v4.json"
)
TARGET_LAYERS = (10, 17)
FIT_EXAMPLES_PER_FAMILY = 5
EVALUATION_EXAMPLES_PER_FAMILY = 5
GENERATOR_STEPS = 32
GENERATOR_MINIBATCH_ROWS = 128
GENERATOR_LEARNING_RATE = 2.0e-3
RESIDUAL_RANK = 16
RESIDUAL_RIDGE = 1.0e-3
CHART_COUNT = 8
CHART_RANK = 16
CHART_TEMPERATURE = 1.0
CHART_EDGE_RIDGE = 1.0e-2

__all__ = [
    "A5E_EXPERIMENT_FORMAT_VERSION",
    "A5E_EXPERIMENT_SCHEMA",
    "DEFAULT_OUTPUT",
    "DEFAULT_PANEL_PATH",
    "DEFAULT_REVISION",
    "Gemma3PhysicalCompactMLP",
    "run_gemma3_l10_l17_a5e_functional_mlp_channel_coalescing",
]


def _progress(message: str) -> None:
    print(f"[a5e-coalescing] {message}", flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        b"fisher_graph:a5e-functional-coalescing-report:v4\0"
        + _canonical_json_bytes(value)
    ).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _module_storage_pointers(module: nn.Module) -> set[int]:
    return {
        int(value.untyped_storage().data_ptr())
        for value in (*module.parameters(), *module.buffers())
        if value.numel()
    }


class Gemma3PhysicalCompactMLP(nn.Module):
    """Bias-free, physically narrower Gemma gated MLP."""

    def __init__(
        self,
        *,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        if (
            gate_weight.ndim != 2
            or up_weight.shape != gate_weight.shape
            or down_weight.ndim != 2
            or down_weight.shape[1] != gate_weight.shape[0]
            or down_weight.shape[0] != gate_weight.shape[1]
        ):
            raise ValueError("compact gated-MLP weight shapes are inconsistent")
        intermediate, hidden = gate_weight.shape
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(gate_weight.detach())
            self.up_proj.weight.copy_(up_weight.detach())
            self.down_proj.weight.copy_(down_weight.detach())
        self.act_fn = copy.deepcopy(activation)
        self.eval()
        self.requires_grad_(False)

    @property
    def intermediate_width(self) -> int:
        return self.gate_proj.out_features

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


@dataclass(frozen=True, slots=True)
class _LayerCapture:
    inputs: Tensor
    features: Tensor
    score_gradients: Tensor

    def __post_init__(self) -> None:
        if (
            self.inputs.ndim != 2
            or self.features.ndim != 2
            or self.score_gradients.shape != self.features.shape
            or self.inputs.shape[0] != self.features.shape[0]
            or not self.inputs.shape[0]
        ):
            raise ValueError("invalid A5e layer capture")


@dataclass(frozen=True, slots=True)
class _PairFit:
    donors: Tensor
    survivors: Tensor
    pair_scores: Tensor
    gate_weight: Tensor
    up_weight: Tensor
    down_weight: Tensor
    fit_metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _FrozenResidual:
    input_mean: Tensor
    output_mean: Tensor
    input_to_latent: Tensor
    latent_to_output: Tensor

    @property
    def rank(self) -> int:
        return self.input_to_latent.shape[1]

    @property
    def parameter_count(self) -> int:
        return (
            self.input_to_latent.numel()
            + self.latent_to_output.numel()
            + self.output_mean.numel()
        )

    def predict(self, inputs: Tensor) -> Tensor:
        return (
            (inputs - self.input_mean)
            @ self.input_to_latent
            @ self.latent_to_output
            + self.output_mean
        )


class _ResidualBuffer:
    def __init__(self) -> None:
        self.value: Tensor | None = None

    def store(self, value: Tensor) -> None:
        if self.value is not None:
            raise RuntimeError("A5e residual buffer was not consumed")
        self.value = value

    def consume(self, reference: Tensor) -> Tensor:
        value = self.value
        self.value = None
        if value is None or value.shape != reference.shape:
            raise RuntimeError("A5e residual buffer shape drifted")
        return reference + value.to(device=reference.device, dtype=reference.dtype)


class _ResidualMLPOverlay(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        residual: _FrozenResidual,
        buffer: _ResidualBuffer,
    ) -> None:
        super().__init__()
        self.source = source
        self.register_buffer("input_mean", residual.input_mean.detach().clone())
        self.register_buffer(
            "output_mean", residual.output_mean.detach().clone()
        )
        self.register_buffer(
            "input_to_latent", residual.input_to_latent.detach().clone()
        )
        self.register_buffer(
            "latent_to_output", residual.latent_to_output.detach().clone()
        )
        object.__setattr__(self, "_buffer", buffer)
        self.eval()

    @property
    def gate_proj(self) -> nn.Module:
        return self.source.gate_proj  # type: ignore[no-any-return]

    @property
    def up_proj(self) -> nn.Module:
        return self.source.up_proj  # type: ignore[no-any-return]

    @property
    def down_proj(self) -> nn.Module:
        return self.source.down_proj  # type: ignore[no-any-return]

    def forward(self, inputs: Tensor) -> Tensor:
        correction = (
            (inputs - self.input_mean)
            @ self.input_to_latent
            @ self.latent_to_output
            + self.output_mean
        )
        self._buffer.store(correction)
        return self.source(inputs)


class _ResidualNormOverlay(nn.Module):
    def __init__(self, source: nn.Module, buffer: _ResidualBuffer) -> None:
        super().__init__()
        self.source = source
        object.__setattr__(self, "_buffer", buffer)
        self.eval()

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.source(inputs)
        if not isinstance(result, Tensor):
            raise TypeError("Gemma post-feed-forward norm must return a Tensor")
        return self._buffer.consume(result)


@contextmanager
def _temporary_mlp_overlay(
    adapter: Gemma3CausalLMAdapter,
    replacements: Mapping[int, nn.Module],
    *,
    residuals: Mapping[int, _FrozenResidual] | None = None,
) -> Iterator[None]:
    layers = adapter.module.model.layers
    original_mlps: dict[int, nn.Module] = {}
    original_norms: dict[int, nn.Module] = {}
    buffers: dict[int, _ResidualBuffer] = {}
    try:
        for ordinal, replacement in replacements.items():
            layer = layers[ordinal]
            source_mlp = layer.mlp
            if not isinstance(source_mlp, nn.Module):
                raise TypeError("Gemma layer MLP is invalid")
            original_mlps[ordinal] = source_mlp
            if residuals is None or ordinal not in residuals:
                layer.mlp = replacement
                continue
            source_norm = layer.post_feedforward_layernorm
            if not isinstance(source_norm, nn.Module):
                raise TypeError("Gemma post-feed-forward norm is invalid")
            original_norms[ordinal] = source_norm
            buffer = _ResidualBuffer()
            buffers[ordinal] = buffer
            layer.mlp = _ResidualMLPOverlay(
                replacement,
                residuals[ordinal],
                buffer,
            )
            layer.post_feedforward_layernorm = _ResidualNormOverlay(
                source_norm,
                buffer,
            )
        yield
        if any(buffer.value is not None for buffer in buffers.values()):
            raise RuntimeError("A5e residual buffer remained live")
    finally:
        for ordinal, norm in original_norms.items():
            layers[ordinal].post_feedforward_layernorm = norm
        for ordinal, mlp in original_mlps.items():
            layers[ordinal].mlp = mlp


def _load_prompt_split(
    path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("A5e panel examples are unavailable")
    by_family: dict[str, list[dict[str, object]]] = {}
    for example in examples:
        if not isinstance(example, dict):
            raise TypeError("A5e panel example must be an object")
        family = example.get("family_id")
        prompt = example.get("prompt")
        example_id = example.get("example_id")
        if not all(
            isinstance(value, str) and value
            for value in (family, prompt, example_id)
        ):
            raise ValueError("A5e panel example identity is invalid")
        by_family.setdefault(family, []).append(example)
    fit: list[str] = []
    evaluation: list[str] = []
    fit_ids: list[str] = []
    evaluation_ids: list[str] = []
    for family in sorted(by_family):
        rows = by_family[family]
        needed = FIT_EXAMPLES_PER_FAMILY + EVALUATION_EXAMPLES_PER_FAMILY
        if len(rows) < needed:
            raise ValueError("each A5e family must supply ten examples")
        for row in rows[:FIT_EXAMPLES_PER_FAMILY]:
            fit.append(str(row["prompt"]))
            fit_ids.append(str(row["example_id"]))
        for row in rows[FIT_EXAMPLES_PER_FAMILY:needed]:
            evaluation.append(str(row["prompt"]))
            evaluation_ids.append(str(row["example_id"]))
    if set(fit_ids) & set(evaluation_ids):
        raise RuntimeError("A5e fit/evaluation prompts overlap")
    split = {
        "panel_file": path.name,
        "panel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "family_count": len(by_family),
        "fit_example_count": len(fit),
        "evaluation_example_count": len(evaluation),
        "fit_example_ids_sha256": hashlib.sha256(
            _canonical_json_bytes(fit_ids)
        ).hexdigest(),
        "evaluation_example_ids_sha256": hashlib.sha256(
            _canonical_json_bytes(evaluation_ids)
        ).hexdigest(),
        "prompt_disjoint": True,
        "family_disjoint": False,
    }
    return tuple(fit), tuple(evaluation), split


def _tokenize_batches(
    tokenizer: object,
    prompts: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], ...]:
    batches: list[dict[str, Tensor]] = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            list(prompts[start : start + batch_size]),
            padding=True,
            return_tensors="pt",
        )
        batch = {
            key: value.to(device=device)
            for key, value in encoded.items()
            if isinstance(value, Tensor)
        }
        if "input_ids" not in batch or "attention_mask" not in batch:
            raise ValueError("tokenizer omitted input_ids or attention_mask")
        batches.append(batch)
    return tuple(batches)


def _capture_fisher_rows(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[Mapping[str, Tensor]],
) -> dict[int, _LayerCapture]:
    rows: dict[int, dict[str, list[Tensor]]] = {
        ordinal: {"inputs": [], "features": [], "score_gradients": []}
        for ordinal in TARGET_LAYERS
    }
    source_mlps = {
        ordinal: adapter.source_module(adapter.layers[ordinal].id).mlp
        for ordinal in TARGET_LAYERS
    }
    for batch in batches:
        current_inputs: dict[int, Tensor] = {}
        current_features: dict[int, Tensor] = {}
        handles = []
        for ordinal, mlp in source_mlps.items():
            def capture_input(
                _module: nn.Module,
                args: tuple[Tensor, ...],
                *,
                layer_ordinal: int = ordinal,
            ) -> None:
                current_inputs[layer_ordinal] = args[0].detach()

            def detach_feature(
                _module: nn.Module,
                args: tuple[Tensor, ...],
                *,
                layer_ordinal: int = ordinal,
            ) -> tuple[Tensor]:
                value = args[0].detach().requires_grad_(True)
                current_features[layer_ordinal] = value
                return (value,)

            handles.append(mlp.register_forward_pre_hook(capture_input))
            handles.append(mlp.down_proj.register_forward_pre_hook(detach_feature))
        try:
            adapter.module.zero_grad(set_to_none=True)
            output = adapter.module(**batch, use_cache=False, return_dict=True)
            logits = output.logits[:, :-1, :].float()
            labels = batch["input_ids"][:, 1:]
            valid = batch["attention_mask"][:, 1:].bool()
            loss = F.cross_entropy(logits[valid], labels[valid], reduction="sum")
            loss.backward()
            token_mask = batch["attention_mask"].bool()
            for ordinal in TARGET_LAYERS:
                inputs = current_inputs[ordinal]
                features = current_features[ordinal]
                gradients = features.grad
                if gradients is None:
                    raise RuntimeError("A5e feature score gradient is unavailable")
                rows[ordinal]["inputs"].append(
                    inputs[token_mask].detach().to(device="cpu", dtype=torch.float32)
                )
                rows[ordinal]["features"].append(
                    features[token_mask].detach().to(device="cpu", dtype=torch.float32)
                )
                rows[ordinal]["score_gradients"].append(
                    gradients[token_mask].detach().to(device="cpu", dtype=torch.float32)
                )
            del output, logits, loss
        finally:
            for handle in handles:
                handle.remove()
            adapter.module.zero_grad(set_to_none=True)
    return {
        ordinal: _LayerCapture(
            inputs=torch.cat(value["inputs"], dim=0).contiguous(),
            features=torch.cat(value["features"], dim=0).contiguous(),
            score_gradients=torch.cat(
                value["score_gradients"], dim=0
            ).contiguous(),
        )
        for ordinal, value in rows.items()
    }


def _normalize_columns(value: Tensor) -> Tensor:
    return value / value.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1.0e-12)


def _fit_functional_pairs(
    source_mlp: nn.Module,
    capture: _LayerCapture,
    *,
    removed_count: int,
    seed: int,
) -> _PairFit:
    features = capture.features
    influence = features * capture.score_gradients
    down_columns = source_mlp.down_proj.weight.detach().float().cpu().T.contiguous()
    importance = influence.square().mean(dim=0)
    donors = torch.argsort(importance)[:removed_count].contiguous()
    donor_set = set(int(value) for value in donors.tolist())

    influence_unit = _normalize_columns(influence)
    feature_unit = _normalize_columns(features)
    down_unit = _normalize_columns(down_columns.T).T
    coupling = (
        (influence_unit[:, donors].T @ influence_unit).abs()
        * (feature_unit[:, donors].T @ feature_unit).abs()
        * (down_unit[donors] @ down_unit.T).abs()
    )
    coupling[:, donors] = -1.0
    survivors: list[int] = []
    pair_scores: list[float] = []
    used: set[int] = set()
    for row in range(removed_count):
        scores = coupling[row].clone()
        if used:
            scores[list(used)] = -1.0
        survivor = int(torch.argmax(scores).item())
        if survivor in donor_set or survivor in used or scores[survivor] < 0.0:
            raise RuntimeError("A5e could not assign a unique survivor")
        used.add(survivor)
        survivors.append(survivor)
        pair_scores.append(float(scores[survivor].item()))
    survivor_tensor = torch.tensor(survivors, dtype=torch.long)

    target_coordinates: list[Tensor] = []
    target_down: list[Tensor] = []
    for donor, survivor in zip(donors.tolist(), survivors, strict=True):
        left = features[:, [survivor, donor]].double()
        right = down_columns[[survivor, donor], :].double().T
        q_left, r_left = torch.linalg.qr(left, mode="reduced")
        q_right, r_right = torch.linalg.qr(right, mode="reduced")
        u, singular, vh = torch.linalg.svd(
            r_left @ r_right.T,
            full_matrices=False,
        )
        scale = singular[0].clamp_min(0.0).sqrt()
        target_coordinates.append((q_left @ u[:, 0] * scale).float())
        target_down.append((q_right @ vh[0] * scale).float())
    coordinates = torch.stack(target_coordinates, dim=1).contiguous()
    fitted_down = torch.stack(target_down, dim=1).contiguous()

    gate = nn.Parameter(
        source_mlp.gate_proj.weight.detach()[survivor_tensor]
        .float()
        .cpu()
        .clone()
    )
    up = nn.Parameter(
        source_mlp.up_proj.weight.detach()[survivor_tensor]
        .float()
        .cpu()
        .clone()
    )
    inputs = capture.inputs
    scale = coordinates.square().mean(dim=0).sqrt().clamp_min(1.0e-5)
    optimizer = torch.optim.Adam((gate, up), lr=GENERATOR_LEARNING_RATE)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def coordinate_nrmse(gate_weight: Tensor, up_weight: Tensor) -> float:
        with torch.no_grad():
            predicted = source_mlp.act_fn(inputs @ gate_weight.T) * (
                inputs @ up_weight.T
            )
            return float(
                (predicted - coordinates).square().sum().sqrt()
                / coordinates.square().sum().sqrt().clamp_min(1.0e-12)
            )

    initial_nrmse = coordinate_nrmse(gate.detach(), up.detach())
    for _ in range(GENERATOR_STEPS):
        count = min(GENERATOR_MINIBATCH_ROWS, inputs.shape[0])
        indices = torch.randperm(inputs.shape[0], generator=generator)[:count]
        batch_inputs = inputs[indices]
        target = coordinates[indices]
        predicted = source_mlp.act_fn(batch_inputs @ gate.T) * (
            batch_inputs @ up.T
        )
        loss = ((predicted - target) / scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((gate, up), 1.0)
        optimizer.step()
    final_nrmse = coordinate_nrmse(gate.detach(), up.detach())
    return _PairFit(
        donors=donors,
        survivors=survivor_tensor,
        pair_scores=torch.tensor(pair_scores, dtype=torch.float32),
        gate_weight=gate.detach().contiguous(),
        up_weight=up.detach().contiguous(),
        down_weight=fitted_down,
        fit_metrics={
            "initial_coordinate_nrmse": initial_nrmse,
            "final_coordinate_nrmse": final_nrmse,
            "relative_improvement": (
                0.0
                if initial_nrmse == 0.0
                else (initial_nrmse - final_nrmse) / initial_nrmse
            ),
            "mean_pair_coupling": float(
                torch.tensor(pair_scores).mean().item()
            ),
        },
    )


def _materialize_candidate(
    source_mlp: nn.Module,
    fit: _PairFit,
    *,
    removed_count: int,
    functional: bool,
) -> Gemma3PhysicalCompactMLP:
    donors = fit.donors[:removed_count]
    removed = set(int(value) for value in donors.tolist())
    source_width = source_mlp.gate_proj.weight.shape[0]
    kept = torch.tensor(
        [index for index in range(source_width) if index not in removed],
        dtype=torch.long,
    )
    gate = source_mlp.gate_proj.weight.detach()[kept].clone()
    up = source_mlp.up_proj.weight.detach()[kept].clone()
    down = source_mlp.down_proj.weight.detach()[:, kept].clone()
    if functional:
        position = {int(index): offset for offset, index in enumerate(kept.tolist())}
        for pair_index, survivor in enumerate(
            fit.survivors[:removed_count].tolist()
        ):
            offset = position[int(survivor)]
            gate[offset].copy_(fit.gate_weight[pair_index])
            up[offset].copy_(fit.up_weight[pair_index])
            down[:, offset].copy_(fit.down_weight[:, pair_index])
    candidate = Gemma3PhysicalCompactMLP(
        gate_weight=gate.contiguous(),
        up_weight=up.contiguous(),
        down_weight=down.contiguous(),
        activation=source_mlp.act_fn,
    )
    if _module_storage_pointers(source_mlp) & _module_storage_pointers(candidate):
        raise RuntimeError("A5e compact MLP aliases native tensor storage")
    return candidate


def _materialize_all_modes(
    source_mlp: nn.Module,
    *,
    model_fingerprint: str = "0" * 64,
    layer_ordinal: int = 0,
) -> Gemma3AllModeGeneratorGraphMLP:
    """Compile every native channel into an exact generator graph node."""

    return compile_exact_all_mode_generator_graph(
        source_mlp,
        model_fingerprint=model_fingerprint,
        layer_ordinal=layer_ordinal,
    )


def _fit_affine_all_mode_graph(
    source_mlp: nn.Module,
    capture: _LayerCapture,
    *,
    model_fingerprint: str,
    layer_ordinal: int,
    fit_split_sha256: str,
) -> tuple[Gemma3AllModeGeneratorGraphMLP, dict[str, float]]:
    """Fit all 2,048 modal states without deleting or merging a node."""

    inputs = capture.inputs
    states = capture.features
    input_mean = inputs.mean(dim=0, keepdim=True)
    state_mean = states.mean(dim=0, keepdim=True)
    centered_inputs = inputs - input_mean
    centered_states = states - state_mean
    gram = centered_inputs @ centered_inputs.T
    ridge = 1.0e-3 * float(gram.diag().mean().clamp_min(1.0e-12))
    solution = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0]),
        centered_states,
    )
    input_factor = centered_inputs.T @ solution
    bias = (state_mean - input_mean @ input_factor).squeeze(0)
    candidate = compile_fitted_affine_all_mode_generator_graph(
        affine_weight=input_factor.T,
        affine_bias=bias,
        decoder_weight=source_mlp.down_proj.weight.detach(),
        model_fingerprint=model_fingerprint,
        layer_ordinal=layer_ordinal,
        fit_split_sha256=fit_split_sha256,
    )
    with torch.no_grad():
        predicted_states = candidate.affine_generators(inputs)
        predicted_output = candidate(inputs)
        native_output = source_mlp(inputs)
    return candidate, {
        "fit_modal_state_nrmse": _relative_rmse(
            predicted_states,
            states,
        ),
        "fit_mlp_output_nrmse": _relative_rmse(
            predicted_output,
            native_output,
        ),
    }


def _chart_membership_coordinates(
    inputs: Tensor,
    centers: Tensor,
    bases: Tensor,
    distance_scales: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    displacement = inputs.unsqueeze(1) - centers
    distance = displacement.square().sum(dim=-1)
    logits = -0.5 * distance / (
        distance_scales.square() * temperature
    )
    memberships = logits.softmax(dim=-1)
    coordinates = torch.einsum(
        "nch,chr->ncr",
        displacement,
        bases,
    )
    return memberships, coordinates


def _deterministic_chart_geometry(
    inputs: Tensor,
    row_weights: Tensor,
    *,
    chart_count: int,
    chart_rank: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if (
        row_weights.shape != (inputs.shape[0],)
        or not bool(torch.isfinite(row_weights).all())
        or bool((row_weights <= 0.0).any())
    ):
        raise ValueError("chart row weights must be finite and positive")
    normalized_weights = row_weights / row_weights.sum()
    global_mean = (normalized_weights.unsqueeze(1) * inputs).sum(
        dim=0,
        keepdim=True,
    )
    centered = inputs - global_mean
    weighted_centered = centered * normalized_weights.sqrt().unsqueeze(1)
    _, _, global_vh = torch.linalg.svd(
        weighted_centered,
        full_matrices=False,
    )
    global_basis = global_vh[:chart_rank].T.contiguous()
    routing = centered @ global_vh[: min(chart_rank, 8)].T
    chosen = [int(torch.argmin(routing[:, 0]).item())]
    minimum_distance = torch.full((inputs.shape[0],), float("inf"))
    for _ in range(1, chart_count):
        current = routing[chosen[-1]]
        distance = (routing - current).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        chosen.append(int(torch.argmax(minimum_distance).item()))
    routing_centers = routing[chosen].clone()
    assignments = torch.full(
        (inputs.shape[0],),
        -1,
        dtype=torch.long,
    )
    for _ in range(12):
        distance = torch.cdist(routing, routing_centers).square()
        updated = distance.argmin(dim=1)
        if torch.equal(updated, assignments):
            break
        assignments = updated
        for chart in range(chart_count):
            selected = routing[assignments == chart]
            if selected.numel():
                selected_weights = row_weights[assignments == chart]
                routing_centers[chart] = (
                    selected_weights.unsqueeze(1) * selected
                ).sum(dim=0) / selected_weights.sum()

    centers: list[Tensor] = []
    bases: list[Tensor] = []
    scales: list[Tensor] = []
    occupancies: list[int] = []
    for chart in range(chart_count):
        selected = inputs[assignments == chart]
        selected_weights = row_weights[assignments == chart]
        if not selected.shape[0]:
            raise RuntimeError("deterministic chart fitting produced an empty chart")
        normalized_local_weights = selected_weights / selected_weights.sum()
        center = (
            normalized_local_weights.unsqueeze(1) * selected
        ).sum(dim=0)
        local = selected - center
        _, _, local_vh = torch.linalg.svd(
            local * normalized_local_weights.sqrt().unsqueeze(1),
            full_matrices=False,
        )
        candidates = torch.cat(
            (local_vh[:chart_rank].T, global_basis),
            dim=1,
        )
        basis, _ = torch.linalg.qr(candidates, mode="reduced")
        local_basis = basis[:, :chart_rank]
        local_coordinates = local @ local_basis
        coordinate_scales = (
            normalized_local_weights.unsqueeze(1)
            * local_coordinates.square()
        ).sum(dim=0).sqrt().clamp_min(1.0e-4)
        centers.append(center)
        bases.append(local_basis / coordinate_scales.unsqueeze(0))
        scales.append(
            (
                normalized_local_weights
                * local.square().sum(dim=1)
            ).sum().sqrt().clamp_min(1.0e-5)
        )
        occupancies.append(int(selected.shape[0]))
    return (
        torch.stack(centers).contiguous(),
        torch.stack(bases).contiguous(),
        torch.stack(scales).contiguous(),
        torch.tensor(occupancies, dtype=torch.long),
    )


def _fit_chart_conditioned_all_mode_graph(
    source_mlp: nn.Module,
    base_graph: Gemma3AllModeGeneratorGraphMLP,
    capture: _LayerCapture,
    *,
    model_fingerprint: str,
    layer_ordinal: int,
    fit_split_sha256: str,
) -> tuple[Gemma3AllModeGeneratorGraphMLP, dict[str, object]]:
    if base_graph.affine_generators is None:
        raise ValueError("chart-conditioned fit requires the affine base graph")
    inputs = capture.inputs
    states = capture.features
    row_fisher = (states * capture.score_gradients).square().mean(dim=1)
    row_fisher = (row_fisher / row_fisher.mean().clamp_min(1.0e-12)).clamp(
        min=0.1,
        max=10.0,
    )
    centers, bases, scales, occupancy = _deterministic_chart_geometry(
        inputs,
        row_fisher,
        chart_count=CHART_COUNT,
        chart_rank=CHART_RANK,
    )
    memberships, coordinates = _chart_membership_coordinates(
        inputs,
        centers,
        bases,
        scales,
        temperature=CHART_TEMPERATURE,
    )
    augmented = torch.cat(
        (torch.ones_like(coordinates[..., :1]), coordinates),
        dim=-1,
    )
    design = (memberships.unsqueeze(-1) * augmented).reshape(
        inputs.shape[0],
        CHART_COUNT * (CHART_RANK + 1),
    )
    with torch.no_grad():
        base_states = base_graph.affine_generators(inputs)
    target = states - base_states
    root_weight = row_fisher.sqrt().unsqueeze(1)
    weighted_design = design * root_weight
    weighted_target = target * root_weight
    gram = weighted_design.T @ weighted_design
    ridge = CHART_EDGE_RIDGE * float(
        gram.diag().mean().clamp_min(1.0e-12)
    )
    coefficients = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0]),
        weighted_design.T @ weighted_target,
    ).reshape(CHART_COUNT, CHART_RANK + 1, states.shape[1])
    graph = compile_chart_conditioned_all_mode_generator_graph(
        affine_weight=base_graph.affine_generators.weight.detach(),
        affine_bias=base_graph.affine_generators.bias.detach(),
        decoder_weight=source_mlp.down_proj.weight.detach(),
        chart_centers=centers,
        chart_bases=bases,
        chart_distance_scales=scales,
        chart_edge_weight=coefficients,
        chart_temperature=CHART_TEMPERATURE,
        model_fingerprint=model_fingerprint,
        layer_ordinal=layer_ordinal,
        fit_split_sha256=fit_split_sha256,
    )
    with torch.no_grad():
        execution = graph.execute_graph(
            inputs,
            capture_modal_states=True,
            capture_chart_states=True,
        )
        native_output = source_mlp(inputs)
    if execution.modal_states is None or execution.chart_memberships is None:
        raise RuntimeError("chart-conditioned fit instrumentation is incomplete")
    entropy = -(
        execution.chart_memberships
        * execution.chart_memberships.clamp_min(1.0e-12).log()
    ).sum(dim=1)
    return graph, {
        "fit_modal_state_nrmse": _relative_rmse(
            execution.modal_states,
            states,
        ),
        "fit_mlp_output_nrmse": _relative_rmse(
            execution.output,
            native_output,
        ),
        "fit_mean_membership_entropy": float(entropy.mean().item()),
        "fit_mean_max_membership": float(
            execution.chart_memberships.max(dim=1).values.mean().item()
        ),
        "hard_chart_occupancy": occupancy.tolist(),
        "row_fisher_minimum": float(row_fisher.min().item()),
        "row_fisher_maximum": float(row_fisher.max().item()),
    }


def _relative_rmse(actual: Tensor, expected: Tensor) -> float:
    return float(
        (actual.double() - expected.double()).square().sum().sqrt()
        / expected.double().square().sum().sqrt().clamp_min(1.0e-12)
    )


def _layer_metrics(
    source_mlp: nn.Module,
    candidate: nn.Module,
    capture: _LayerCapture,
    fit: _PairFit,
    *,
    removed_count: int,
) -> dict[str, float]:
    with torch.no_grad():
        expected = source_mlp(capture.inputs)
        actual = candidate(capture.inputs)
    donors = fit.donors[:removed_count]
    survivors = fit.survivors[:removed_count]
    expected_jvp = (
        capture.features[:, donors] * capture.score_gradients[:, donors]
        + capture.features[:, survivors]
        * capture.score_gradients[:, survivors]
    )
    compact_features = (
        candidate.act_fn(candidate.gate_proj(capture.inputs))
        * candidate.up_proj(capture.inputs)
    )
    kept = [
        index
        for index in range(source_mlp.gate_proj.out_features)
        if index not in set(int(value) for value in donors.tolist())
    ]
    positions = {value: index for index, value in enumerate(kept)}
    predicted_columns = torch.stack(
        [compact_features[:, positions[int(value)]] for value in survivors],
        dim=1,
    )
    predicted_jvp = predicted_columns * (
        capture.score_gradients[:, survivors]
    )
    return {
        "mlp_output_nrmse": _relative_rmse(actual, expected),
        "channel_jvp_relative_error": _relative_rmse(
            predicted_jvp,
            expected_jvp,
        ),
    }


def _fit_residual(
    source_mlp: nn.Module,
    source_norm: nn.Module,
    compact_mlp: nn.Module,
    inputs: Tensor,
) -> tuple[_FrozenResidual, dict[str, float]]:
    with torch.no_grad():
        native_delta = source_norm(source_mlp(inputs))
        compact_delta = source_norm(compact_mlp(inputs))
        target = (native_delta - compact_delta).float().cpu()
    input_mean = inputs.mean(dim=0, keepdim=True)
    output_mean = target.mean(dim=0, keepdim=True)
    centered_inputs = inputs - input_mean
    centered_target = target - output_mean
    u, singular, vh = torch.linalg.svd(centered_target, full_matrices=False)
    rank = min(RESIDUAL_RANK, singular.numel())
    coordinates = u[:, :rank] * singular[:rank]
    gram = centered_inputs @ centered_inputs.T
    ridge = RESIDUAL_RIDGE * float(gram.diag().mean().clamp_min(1.0e-12))
    solution = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0]),
        coordinates,
    )
    input_to_latent = centered_inputs.T @ solution
    residual = _FrozenResidual(
        input_mean=input_mean,
        output_mean=output_mean,
        input_to_latent=input_to_latent.contiguous(),
        latent_to_output=vh[:rank].contiguous(),
    )
    prediction = residual.predict(inputs)
    return residual, {
        "fit_target_nrmse": _relative_rmse(prediction, target),
        "target_energy_fraction": float(
            target.square().sum()
            / native_delta.float().cpu().square().sum().clamp_min(1.0e-12)
        ),
    }


class _MetricAccumulator:
    def __init__(self) -> None:
        self.tokens = 0
        self.nll = 0.0
        self.kl = 0.0
        self.top1 = 0
        self.square_error = 0.0
        self.native_square = 0.0
        self.max_abs_error = 0.0

    def update(
        self,
        native_logits: Tensor,
        candidate_logits: Tensor,
        batch: Mapping[str, Tensor],
    ) -> None:
        valid = batch["attention_mask"][:, 1:].bool()
        labels = batch["input_ids"][:, 1:][valid]
        native = native_logits[:, :-1, :][valid].float()
        candidate = candidate_logits[:, :-1, :][valid].float()
        self.tokens += int(labels.numel())
        self.nll += float(F.cross_entropy(candidate, labels, reduction="sum"))
        native_log = native.log_softmax(dim=-1)
        candidate_log = candidate.log_softmax(dim=-1)
        self.kl += float(
            (native_log.exp() * (native_log - candidate_log)).sum().item()
        )
        self.top1 += int(
            (native.argmax(dim=-1) == candidate.argmax(dim=-1)).sum().item()
        )
        self.square_error += float((candidate - native).square().sum().item())
        self.native_square += float(native.square().sum().item())
        self.max_abs_error = max(
            self.max_abs_error,
            float((candidate - native).abs().max().item()),
        )

    def result(self, *, native_nll: float) -> dict[str, float | int]:
        if self.tokens <= 0:
            raise RuntimeError("A5e evaluation has no supervised tokens")
        nll = self.nll / self.tokens
        return {
            "supervised_tokens": self.tokens,
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": self.kl / self.tokens,
            "top1_agreement_to_native": self.top1 / self.tokens,
            "logit_nrmse": math.sqrt(
                self.square_error / max(self.native_square, 1.0e-30)
            ),
            "maximum_absolute_logit_error": self.max_abs_error,
        }


def _native_nll(logits: Tensor, batch: Mapping[str, Tensor]) -> tuple[float, int]:
    valid = batch["attention_mask"][:, 1:].bool()
    labels = batch["input_ids"][:, 1:][valid]
    value = F.cross_entropy(logits[:, :-1, :][valid].float(), labels, reduction="sum")
    return float(value.item()), int(labels.numel())


def run_gemma3_l10_l17_a5e_functional_mlp_channel_coalescing(
    *,
    revision: str,
    output: Path | str = DEFAULT_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 4,
) -> dict[str, object]:
    """Run the prompt-disjoint A5e physical-compaction development ladder."""

    if revision != DEFAULT_REVISION or model_id != DEFAULT_MODEL_ID:
        raise ValueError("A5e requires the canonical pinned Gemma checkpoint")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("A5e currently requires deterministic CPU float32")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite an A5e experiment")
    fit_prompts, evaluation_prompts, split = _load_prompt_split(Path(panel_path))
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("load pinned local Gemma")
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
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    source_fingerprint = adapter.model_fingerprint()
    execution_fingerprint = adapter.execution_fingerprint()
    fit_batches = _tokenize_batches(
        tokenizer,
        fit_prompts,
        batch_size=batch_size,
        device=device,
    )
    evaluation_batches = _tokenize_batches(
        tokenizer,
        evaluation_prompts,
        batch_size=batch_size,
        device=device,
    )

    _progress("fit: capture L10/L17 activations and score gradients")
    started = time.monotonic()
    fit_capture = _capture_fisher_rows(adapter, fit_batches)
    _progress("evaluation: capture frozen native JVP rows")
    evaluation_capture = _capture_fisher_rows(adapter, evaluation_batches)
    capture_seconds = time.monotonic() - started

    source_mlps = {
        ordinal: adapter.source_module(adapter.layers[ordinal].id).mlp
        for ordinal in TARGET_LAYERS
    }
    compiled_all_modes = {
        ordinal: _materialize_all_modes(
            source_mlps[ordinal],
            model_fingerprint=source_fingerprint,
            layer_ordinal=ordinal,
        )
        for ordinal in TARGET_LAYERS
    }
    protocol = build_a5e_functional_mlp_channel_coalescing_protocol()
    max_removed = protocol.trials[-1].removed_channel_count
    fits: dict[int, _PairFit] = {}
    _progress(f"fit: select and synthesize {max_removed} disjoint pairs per layer")
    fit_started = time.monotonic()
    for ordinal in TARGET_LAYERS:
        fits[ordinal] = _fit_functional_pairs(
            source_mlps[ordinal],
            fit_capture[ordinal],
            removed_count=max_removed,
            seed=17_000 + ordinal,
        )
    affine_all_modes: dict[int, Gemma3AllModeGeneratorGraphMLP] = {}
    affine_fit_metrics: dict[str, object] = {}
    chart_all_modes: dict[int, Gemma3AllModeGeneratorGraphMLP] = {}
    chart_fit_metrics: dict[str, object] = {}
    for ordinal in TARGET_LAYERS:
        graph, metrics = _fit_affine_all_mode_graph(
            source_mlps[ordinal],
            fit_capture[ordinal],
            model_fingerprint=source_fingerprint,
            layer_ordinal=ordinal,
            fit_split_sha256=str(split["fit_example_ids_sha256"]),
        )
        affine_all_modes[ordinal] = graph
        with torch.no_grad():
            held_states = graph.affine_generators(
                evaluation_capture[ordinal].inputs
            )
            held_output = graph(evaluation_capture[ordinal].inputs)
            native_held_output = source_mlps[ordinal](
                evaluation_capture[ordinal].inputs
            )
        affine_fit_metrics[str(ordinal)] = {
            **metrics,
            "evaluation_modal_state_nrmse": _relative_rmse(
                held_states,
                evaluation_capture[ordinal].features,
            ),
            "evaluation_mlp_output_nrmse": _relative_rmse(
                held_output,
                native_held_output,
            ),
        }
        chart_graph, chart_metrics = _fit_chart_conditioned_all_mode_graph(
            source_mlps[ordinal],
            graph,
            fit_capture[ordinal],
            model_fingerprint=source_fingerprint,
            layer_ordinal=ordinal,
            fit_split_sha256=str(split["fit_example_ids_sha256"]),
        )
        chart_all_modes[ordinal] = chart_graph
        with torch.no_grad():
            chart_execution = chart_graph.execute_graph(
                evaluation_capture[ordinal].inputs,
                capture_modal_states=True,
                capture_chart_states=True,
            )
            native_chart_output = source_mlps[ordinal](
                evaluation_capture[ordinal].inputs
            )
        if (
            chart_execution.modal_states is None
            or chart_execution.chart_memberships is None
        ):
            raise RuntimeError("chart evaluation instrumentation is incomplete")
        chart_memberships = chart_execution.chart_memberships
        chart_entropy = -(
            chart_memberships * chart_memberships.clamp_min(1.0e-12).log()
        ).sum(dim=1)
        chart_fit_metrics[str(ordinal)] = {
            **chart_metrics,
            "evaluation_modal_state_nrmse": _relative_rmse(
                chart_execution.modal_states,
                evaluation_capture[ordinal].features,
            ),
            "evaluation_mlp_output_nrmse": _relative_rmse(
                chart_execution.output,
                native_chart_output,
            ),
            "evaluation_mean_membership_entropy": float(
                chart_entropy.mean().item()
            ),
            "evaluation_mean_max_membership": float(
                chart_memberships.max(dim=1).values.mean().item()
            ),
            "evaluation_hard_chart_occupancy": torch.bincount(
                chart_memberships.argmax(dim=1),
                minlength=CHART_COUNT,
            ).tolist(),
        }
    fit_seconds = time.monotonic() - fit_started

    candidates: dict[str, dict[float, dict[int, Gemma3PhysicalCompactMLP]]] = {
        "matched_naive_deletion": {},
        "fisher_jacobian_functional_coalescing": {},
    }
    local_metrics: dict[str, dict[str, dict[str, object]]] = {
        "matched_naive_deletion": {},
        "fisher_jacobian_functional_coalescing": {},
    }
    resources: dict[str, dict[str, object]] = {}
    for trial in protocol.trials:
        rate = trial.merge_rate
        count = trial.removed_channel_count
        rate_key = f"{rate:.2f}"
        resources[rate_key] = {
            **trial.state_dict(),
            "target_layer_count": len(TARGET_LAYERS),
            "removed_learned_parameters": (
                trial.native_parameter_savings * len(TARGET_LAYERS)
            ),
            "candidate_whole_model_learned_parameters": (
                source_parameters
                - trial.native_parameter_savings * len(TARGET_LAYERS)
            ),
            "whole_model_parameter_reduction_fraction": (
                trial.native_parameter_savings
                * len(TARGET_LAYERS)
                / source_parameters
            ),
        }
        for arm, functional in (
            ("matched_naive_deletion", False),
            ("fisher_jacobian_functional_coalescing", True),
        ):
            by_layer: dict[int, Gemma3PhysicalCompactMLP] = {}
            layer_results: dict[str, object] = {}
            for ordinal in TARGET_LAYERS:
                candidate = _materialize_candidate(
                    source_mlps[ordinal],
                    fits[ordinal],
                    removed_count=count,
                    functional=functional,
                )
                by_layer[ordinal] = candidate
                layer_results[str(ordinal)] = _layer_metrics(
                    source_mlps[ordinal],
                    candidate,
                    evaluation_capture[ordinal],
                    fits[ordinal],
                    removed_count=count,
                )
            candidates[arm][rate] = by_layer
            local_metrics[arm][rate_key] = layer_results

    _progress("fit: freeze max-rate low-rank post-RMSNorm repair")
    residuals: dict[int, _FrozenResidual] = {}
    residual_fit_metrics: dict[str, object] = {}
    max_rate = A5E_MERGE_RATE_LADDER[-1]
    max_compact = candidates["fisher_jacobian_functional_coalescing"][max_rate]
    for ordinal in TARGET_LAYERS:
        layer = adapter.source_module(adapter.layers[ordinal].id)
        residual, metrics = _fit_residual(
            source_mlps[ordinal],
            layer.post_feedforward_layernorm,
            max_compact[ordinal],
            fit_capture[ordinal].inputs,
        )
        residuals[ordinal] = residual
        residual_fit_metrics[str(ordinal)] = {
            **metrics,
            "rank": residual.rank,
            "learned_parameter_count": residual.parameter_count,
        }
    residual_parameters = sum(value.parameter_count for value in residuals.values())

    accumulators: dict[str, _MetricAccumulator] = {
        "compiled_all_modes_intact": _MetricAccumulator(),
        "approximated_all_modes_graph": _MetricAccumulator(),
        "chart_conditioned_all_modes_graph": _MetricAccumulator(),
        "native_intact_plus_residual_diagnostic": _MetricAccumulator(),
        "max_rate_coalescing_plus_identical_residual": _MetricAccumulator(),
    }
    for rate in A5E_MERGE_RATE_LADDER:
        rate_key = f"{rate:.2f}"
        accumulators[f"matched_naive_deletion@{rate_key}"] = _MetricAccumulator()
        accumulators[
            f"fisher_jacobian_functional_coalescing@{rate_key}"
        ] = _MetricAccumulator()

    native_projection_calls = {
        ordinal: {"gate_proj": 0, "up_proj": 0, "down_proj": 0}
        for ordinal in TARGET_LAYERS
    }
    compiled_all_modes_source_calls = {
        ordinal: {"gate_proj": 0, "up_proj": 0, "down_proj": 0}
        for ordinal in TARGET_LAYERS
    }
    compiled_all_modes_runtime_calls = copy.deepcopy(
        compiled_all_modes_source_calls
    )
    affine_graph_runtime_calls = {
        ordinal: {"affine_generators": 0, "mode_decoder": 0}
        for ordinal in TARGET_LAYERS
    }
    chart_graph_runtime_calls = {
        ordinal: {
            "graph_forward": 0,
            "affine_generators": 0,
            "mode_decoder": 0,
        }
        for ordinal in TARGET_LAYERS
    }
    chart_graph_source_calls = {
        ordinal: {"gate_proj": 0, "up_proj": 0, "down_proj": 0}
        for ordinal in TARGET_LAYERS
    }
    native_nll_sum = 0.0
    native_tokens = 0
    _progress("evaluate: full-model native, intact-control, deletion, and coalescing")
    evaluation_started = time.monotonic()
    with torch.no_grad():
        for batch_index, batch in enumerate(evaluation_batches, start=1):
            _progress(f"evaluate batch {batch_index}/{len(evaluation_batches)}")
            native_output = model(**batch, use_cache=False, return_dict=True)
            native_logits = native_output.logits
            batch_nll, batch_tokens = _native_nll(native_logits, batch)
            native_nll_sum += batch_nll
            native_tokens += batch_tokens

            handles = []
            for ordinal in TARGET_LAYERS:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    def count_source_call(
                        _module: nn.Module,
                        _args: tuple[Tensor, ...],
                        _output: Tensor,
                        *,
                        layer_ordinal: int = ordinal,
                        projection: str = name,
                    ) -> None:
                        compiled_all_modes_source_calls[layer_ordinal][
                            projection
                        ] += 1

                    def count_runtime_call(
                        _module: nn.Module,
                        _args: tuple[Tensor, ...],
                        _output: Tensor,
                        *,
                        layer_ordinal: int = ordinal,
                        projection: str = name,
                    ) -> None:
                        compiled_all_modes_runtime_calls[layer_ordinal][
                            projection
                        ] += 1

                    handles.append(
                        getattr(source_mlps[ordinal], name).register_forward_hook(
                            count_source_call
                        )
                    )
                    handles.append(
                        getattr(
                            compiled_all_modes[ordinal], name
                        ).register_forward_hook(count_runtime_call)
                    )
            try:
                with _temporary_mlp_overlay(adapter, compiled_all_modes):
                    output = model(**batch, use_cache=False, return_dict=True)
                accumulators["compiled_all_modes_intact"].update(
                    native_logits,
                    output.logits,
                    batch,
                )
                del output
            finally:
                for handle in handles:
                    handle.remove()

            handles = []
            for ordinal, graph in chart_all_modes.items():
                def count_chart_graph_call(
                    _module: nn.Module,
                    _args: tuple[Tensor, ...],
                    _output: Tensor,
                    *,
                    layer_ordinal: int = ordinal,
                ) -> None:
                    chart_graph_runtime_calls[layer_ordinal][
                        "graph_forward"
                    ] += 1

                handles.append(graph.register_forward_hook(count_chart_graph_call))
                for name in ("affine_generators", "mode_decoder"):
                    def count_chart_operation(
                        _module: nn.Module,
                        _args: tuple[Tensor, ...],
                        _output: Tensor,
                        *,
                        layer_ordinal: int = ordinal,
                        operation: str = name,
                    ) -> None:
                        chart_graph_runtime_calls[layer_ordinal][operation] += 1

                    handles.append(
                        getattr(graph, name).register_forward_hook(
                            count_chart_operation
                        )
                    )
                for name in ("gate_proj", "up_proj", "down_proj"):
                    def count_chart_source_call(
                        _module: nn.Module,
                        _args: tuple[Tensor, ...],
                        _output: Tensor,
                        *,
                        layer_ordinal: int = ordinal,
                        projection: str = name,
                    ) -> None:
                        chart_graph_source_calls[layer_ordinal][projection] += 1

                    handles.append(
                        getattr(
                            source_mlps[ordinal], name
                        ).register_forward_hook(count_chart_source_call)
                    )
            try:
                with _temporary_mlp_overlay(adapter, chart_all_modes):
                    output = model(**batch, use_cache=False, return_dict=True)
                accumulators["chart_conditioned_all_modes_graph"].update(
                    native_logits,
                    output.logits,
                    batch,
                )
                del output
            finally:
                for handle in handles:
                    handle.remove()

            handles = []
            for ordinal, graph in affine_all_modes.items():
                for name in ("affine_generators", "mode_decoder"):
                    def count_affine_graph_call(
                        _module: nn.Module,
                        _args: tuple[Tensor, ...],
                        _output: Tensor,
                        *,
                        layer_ordinal: int = ordinal,
                        operation: str = name,
                    ) -> None:
                        affine_graph_runtime_calls[layer_ordinal][operation] += 1

                    handles.append(
                        getattr(graph, name).register_forward_hook(
                            count_affine_graph_call
                        )
                    )
            try:
                with _temporary_mlp_overlay(adapter, affine_all_modes):
                    output = model(**batch, use_cache=False, return_dict=True)
                accumulators["approximated_all_modes_graph"].update(
                    native_logits,
                    output.logits,
                    batch,
                )
                del output
            finally:
                for handle in handles:
                    handle.remove()

            handles = []
            for ordinal, mlp in source_mlps.items():
                for name in ("gate_proj", "up_proj", "down_proj"):
                    def count_call(
                        _module: nn.Module,
                        _args: tuple[Tensor, ...],
                        _output: Tensor,
                        *,
                        layer_ordinal: int = ordinal,
                        projection: str = name,
                    ) -> None:
                        native_projection_calls[layer_ordinal][projection] += 1

                    handles.append(getattr(mlp, name).register_forward_hook(count_call))
            try:
                with _temporary_mlp_overlay(
                    adapter,
                    source_mlps,
                    residuals=residuals,
                ):
                    output = model(**batch, use_cache=False, return_dict=True)
                accumulators["native_intact_plus_residual_diagnostic"].update(
                    native_logits,
                    output.logits,
                    batch,
                )
                del output
            finally:
                for handle in handles:
                    handle.remove()

            for rate in A5E_MERGE_RATE_LADDER:
                rate_key = f"{rate:.2f}"
                for arm in (
                    "matched_naive_deletion",
                    "fisher_jacobian_functional_coalescing",
                ):
                    with _temporary_mlp_overlay(
                        adapter,
                        candidates[arm][rate],
                    ):
                        output = model(**batch, use_cache=False, return_dict=True)
                    accumulators[f"{arm}@{rate_key}"].update(
                        native_logits,
                        output.logits,
                        batch,
                    )
                    del output
            with _temporary_mlp_overlay(
                adapter,
                max_compact,
                residuals=residuals,
            ):
                output = model(**batch, use_cache=False, return_dict=True)
            accumulators["max_rate_coalescing_plus_identical_residual"].update(
                native_logits,
                output.logits,
                batch,
            )
            del output, native_output, native_logits
    evaluation_seconds = time.monotonic() - evaluation_started
    native_nll = native_nll_sum / native_tokens
    full_model_metrics = {
        name: accumulator.result(native_nll=native_nll)
        for name, accumulator in accumulators.items()
    }
    expected_calls = len(evaluation_batches)
    if any(
        count != expected_calls
        for layer in native_projection_calls.values()
        for count in layer.values()
    ):
        raise RuntimeError("native intact residual control skipped a projection")
    if any(
        count != 0
        for layer in compiled_all_modes_source_calls.values()
        for count in layer.values()
    ):
        raise RuntimeError("full-width compiled control called a native projection")
    if any(
        count != expected_calls
        for layer in compiled_all_modes_runtime_calls.values()
        for count in layer.values()
    ):
        raise RuntimeError("full-width compiled control skipped a projection")
    if any(
        count != expected_calls
        for layer in affine_graph_runtime_calls.values()
        for count in layer.values()
    ):
        raise RuntimeError("affine all-mode graph skipped a graph operation")
    if any(
        count != expected_calls
        for layer in chart_graph_runtime_calls.values()
        for count in layer.values()
    ):
        raise RuntimeError("chart-conditioned graph skipped a graph traversal")
    if any(
        count != 0
        for layer in chart_graph_source_calls.values()
        for count in layer.values()
    ):
        raise RuntimeError("chart-conditioned graph called a native projection")
    if (
        adapter.model_fingerprint() != source_fingerprint
        or adapter.execution_fingerprint() != execution_fingerprint
        or sum(parameter.numel() for parameter in model.parameters())
        != source_parameters
    ):
        raise RuntimeError("A5e failed to restore the native model exactly")

    pair_reports: dict[str, object] = {}
    for ordinal, fit in fits.items():
        pair_reports[str(ordinal)] = {
            "pair_count": int(fit.donors.numel()),
            "donor_sha256": _tensor_sha256(fit.donors),
            "survivor_sha256": _tensor_sha256(fit.survivors),
            "pair_score_sha256": _tensor_sha256(fit.pair_scores),
            "donors_and_survivors_disjoint": not bool(
                set(fit.donors.tolist()) & set(fit.survivors.tolist())
            ),
            "survivors_unique": len(set(fit.survivors.tolist()))
            == fit.survivors.numel(),
            "selection": (
                "fit_only_activation_fisher_times_feature_correlation_"
                "times_down_jacobian_alignment"
            ),
            "functional_refit": dict(fit.fit_metrics),
        }

    report: dict[str, object] = {
        "schema": A5E_EXPERIMENT_SCHEMA,
        "format_version": A5E_EXPERIMENT_FORMAT_VERSION,
        "scientific_status": {
            "development_only": True,
            "prompt_disjoint_evaluation": True,
            "family_disjoint_evaluation": False,
            "heldout_confirmation": False,
            "compression_success_claimed": False,
            "latency_or_kernel_speed_claimed": False,
        },
        "model": {
            "model_id": model_id,
            "revision": revision,
            "model_fingerprint": source_fingerprint,
            "device": device_name,
            "dtype": dtype,
            "source_whole_model_learned_parameters": source_parameters,
            "target_layers": list(TARGET_LAYERS),
        },
        "data": split,
        "protocol": {
            **protocol.state_dict(),
            "artifact_role": "a5e_executable_development_ladder",
            "actual_model_experiment_implemented": True,
            "model_weights_loaded": True,
            "tensor_values_stored": False,
        },
        "capture": {
            "fit_rows_by_layer": {
                str(key): value.inputs.shape[0]
                for key, value in fit_capture.items()
            },
            "evaluation_rows_by_layer": {
                str(key): value.inputs.shape[0]
                for key, value in evaluation_capture.items()
            },
            "capture_elapsed_seconds": capture_seconds,
        },
        "pairing_and_refit": pair_reports,
        "local_evaluation": local_metrics,
        "full_model_evaluation": {
            "native": {
                "supervised_tokens": native_tokens,
                "nll_per_token": native_nll,
            },
            "conditions": full_model_metrics,
        },
        "native_intact_control": {
            "all_native_channels_executed": True,
            "source_mlp_identity_preserved": True,
            "source_model_restored_exactly": True,
            "projection_call_counts": {
                str(key): value for key, value in native_projection_calls.items()
            },
            "identical_residual_candidate_reused_without_native_refit": True,
            "residual_application_boundary": "layer.L.mlp.delta",
            "post_feedforward_rmsnorm_attached": True,
            "compression_credit_allowed": False,
        },
        "compiled_all_modes_control": {
            "all_native_mode_triplets_materialized": True,
            "mode_count_per_layer": 2048,
            "removed_channel_count": 0,
            "source_free_runtime": True,
            "native_source_projection_call_counts": {
                str(key): value
                for key, value in compiled_all_modes_source_calls.items()
            },
            "compiled_projection_call_counts": {
                str(key): value
                for key, value in compiled_all_modes_runtime_calls.items()
            },
            "target_mlp_learned_parameter_savings": 0,
            "target_mlp_matrix_mac_savings_per_token": 0,
            "compression_credit_allowed": False,
            "graph_metadata_by_layer": {
                str(key): value.graph_metadata()
                for key, value in compiled_all_modes.items()
            },
        },
        "approximated_all_modes_graph": {
            "all_native_modes_retained": True,
            "removed_mode_count": 0,
            "merged_mode_count": 0,
            "generator_family": "fit_only_affine_per_mode",
            "runtime_call_counts": {
                str(key): value
                for key, value in affine_graph_runtime_calls.items()
            },
            "fit_and_evaluation_metrics_by_layer": affine_fit_metrics,
            "graph_metadata_by_layer": {
                str(key): value.graph_metadata()
                for key, value in affine_all_modes.items()
            },
            "compression_credit_allowed": False,
        },
        "chart_conditioned_all_modes_graph": {
            "all_native_modes_retained": True,
            "removed_mode_count": 0,
            "merged_mode_count": 0,
            "chart_count": CHART_COUNT,
            "chart_rank": CHART_RANK,
            "chart_temperature": CHART_TEMPERATURE,
            "edge_ridge": CHART_EDGE_RIDGE,
            "routing_input": "normalized_hidden_state",
            "chart_coordinates": "B_c_transpose_times_h_minus_mu_c",
            "edge_gate": "softmax_negative_scaled_chart_distance",
            "edge_message": "affine_function_of_local_chart_coordinates",
            "runtime_call_counts": {
                str(key): value
                for key, value in chart_graph_runtime_calls.items()
            },
            "native_source_projection_call_counts": {
                str(key): value
                for key, value in chart_graph_source_calls.items()
            },
            "fit_and_evaluation_metrics_by_layer": chart_fit_metrics,
            "graph_metadata_by_layer": {
                str(key): value.graph_metadata()
                for key, value in chart_all_modes.items()
            },
            "compression_credit_allowed": False,
        },
        "residual_diagnostic": {
            "fit_source": "maximum_rate_functional_coalescing_error",
            "fit_metrics_by_layer": residual_fit_metrics,
            "learned_parameter_count": residual_parameters,
            "identical_candidate_applied_to_native_and_compact": True,
        },
        "resources": resources,
        "timing": {
            "pairing_and_refit_seconds": fit_seconds,
            "full_model_evaluation_seconds": evaluation_seconds,
        },
    }
    report["report_sha256"] = _json_sha256(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _progress(f"published: {destination}")
    return copy.deepcopy(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Gemma L10/L17 A5e physical coalescing ladder."
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    run_gemma3_l10_l17_a5e_functional_mlp_channel_coalescing(
        revision=args.revision,
        output=args.output,
        panel_path=args.panel_path,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
