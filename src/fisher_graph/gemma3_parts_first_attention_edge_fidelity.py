"""Exact numerical kernel for attention-edge fidelity interventions.

The kernel has no model loader or artifact writer.  It reconstructs the exact
Gemma 3 attention route tensor from committed generator coefficients, applies
a caller-frozen global route mask, and can replay the standalone exact
39-event executor because that control has no transport edges.  Every sparse
attention layer recomputes Q/K/V and native probabilities from its current
carrier state; only the discrete baseline-selected mask is frozen.

An all-visible, non-renormalized mask is the mandatory control.  It must be
bitwise equal to the public generator's operator output and residual delta.
The experiment runner additionally compares every layer and final endpoint to
the committed whole-model executor before accepting any sparse result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import SequenceContext
from .gemma3_attention_generator import (
    AttentionCoordinatePacket,
    Gemma3AttentionGenerator,
)
from .gemma3_joint_cross_block_chart_executor import (
    JOINT_EXACT_CONTROL_ARM_ID,
)


ATTENTION_EDGE_FIDELITY_K_VALUES: Final = (1, 2, 4, 8, 16)
ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS: Final = (
    "fisher",
    "probability",
    "recency",
    "deterministic_random",
)
ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS: Final = (
    "route_output_energy",
)
ATTENTION_EDGE_FIDELITY_RANKERS: Final = (
    *ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS,
    *ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS,
)
ATTENTION_EDGE_FIDELITY_VARIANTS: Final = (
    "unnormalized_contribution",
    "renormalized_sparse_attention",
)
ATTENTION_EDGE_FIDELITY_RANDOM_SALTS: Final = (
    "primary",
    "replicate_1",
    "replicate_2",
    "replicate_3",
)

_RANDOM_DOMAIN = b"fisher_graph:parts-first-attention-edge-random:v1\0"


__all__ = [
    "ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS",
    "ATTENTION_EDGE_FIDELITY_K_VALUES",
    "ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS",
    "ATTENTION_EDGE_FIDELITY_RANDOM_SALTS",
    "ATTENTION_EDGE_FIDELITY_RANKERS",
    "ATTENTION_EDGE_FIDELITY_VARIANTS",
    "AttentionRouteTensors",
    "SparseAttentionExecution",
    "SparseWholeModelExecution",
    "attention_route_tensors",
    "operator_from_routed_probabilities",
    "replay_exact_with_attention_masks",
    "select_global_attention_route_mask",
    "sparse_attention_from_normalized",
]


def _rms_norm(values: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(dim=-1, keepdim=True) + float(epsilon)
    )
    return (normalized * (1.0 + weight.float())).to(dtype=values.dtype)


def _rotate_half(values: Tensor) -> Tensor:
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def _repeat_key_value(values: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return values
    batch, heads, length, width = values.shape
    return (
        values[:, :, None, :, :]
        .expand(batch, heads, repeats, length, width)
        .reshape(batch, heads * repeats, length, width)
    )


def _allowed_mask(sequence: SequenceContext, window_size: int | None) -> Tensor:
    query = torch.arange(
        sequence.query_length, dtype=torch.int64, device=sequence.device
    ).view(1, sequence.query_length, 1)
    key = torch.arange(
        sequence.key_length, dtype=torch.int64, device=sequence.device
    ).view(1, 1, sequence.key_length)
    allowed = (
        sequence.query_valid_mask[:, :, None]
        & sequence.key_valid_mask[:, None, :]
        & (key <= query)
    )
    if window_size is not None:
        allowed &= query - key < window_size
    return allowed.contiguous()


def _validate_generator(generator: Gemma3AttentionGenerator) -> None:
    if not isinstance(generator, Gemma3AttentionGenerator):
        raise TypeError("attention fidelity requires Gemma3AttentionGenerator")
    if (
        generator.generator_kind != "exact_full_rank"
        or generator.device.type != "cpu"
        or generator.dtype != torch.float32
        or generator.config.dropout != 0.0
        or generator.input_basis is not None
        or generator.output_encoder_weight is not None
        or any(
            value is not None
            for value in (
                generator.q_bias,
                generator.k_bias,
                generator.v_bias,
                generator.o_bias,
            )
        )
    ):
        raise ValueError(
            "attention fidelity requires exact CPU-float32 bias-free attention"
        )


@dataclass(frozen=True, slots=True)
class AttentionRouteTensors:
    probabilities: Tensor
    execution_probabilities: Tensor
    repeated_values: Tensor
    projected_values: Tensor
    route_output_energies: Tensor
    allowed_mask: Tensor
    reference_operator_output: Tensor
    reference_residual_delta: Tensor
    all_routes_operator_literal_parity_passed: bool
    all_routes_delta_literal_parity_passed: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.probabilities, Tensor)
            or self.probabilities.ndim != 4
            or self.probabilities.dtype != torch.float32
            or self.probabilities.device.type != "cpu"
            or not bool(torch.isfinite(self.probabilities).all())
            or bool((self.probabilities < 0).any())
        ):
            raise ValueError("route probabilities are invalid")
        batch, heads, queries, keys = self.probabilities.shape
        if (
            not isinstance(self.execution_probabilities, Tensor)
            or self.execution_probabilities.shape != self.probabilities.shape
            or self.execution_probabilities.dtype != torch.float32
            or self.execution_probabilities.device.type != "cpu"
            or not bool(torch.isfinite(self.execution_probabilities).all())
            or bool((self.execution_probabilities < 0).any())
            or not isinstance(self.repeated_values, Tensor)
            or self.repeated_values.shape[:3] != (batch, heads, keys)
            or self.repeated_values.dtype != torch.float32
            or self.repeated_values.device.type != "cpu"
            or not isinstance(self.projected_values, Tensor)
            or self.projected_values.shape[:3] != (batch, heads, keys)
            or self.projected_values.dtype != torch.float32
            or self.projected_values.device.type != "cpu"
            or not isinstance(self.route_output_energies, Tensor)
            or self.route_output_energies.shape != self.probabilities.shape
            or self.route_output_energies.dtype != torch.float64
            or self.route_output_energies.device.type != "cpu"
            or not bool(torch.isfinite(self.route_output_energies).all())
            or bool((self.route_output_energies < 0).any())
            or not isinstance(self.allowed_mask, Tensor)
            or self.allowed_mask.dtype is not torch.bool
            or self.allowed_mask.shape != (batch, queries, keys)
            or self.allowed_mask.device.type != "cpu"
        ):
            raise ValueError("attention route factor geometry is invalid")
        expected_output_shape = (
            batch,
            queries,
            self.projected_values.shape[-1],
        )
        if (
            self.reference_operator_output.shape != expected_output_shape
            or self.reference_residual_delta.shape != expected_output_shape
            or self.reference_operator_output.dtype != torch.float32
            or self.reference_residual_delta.dtype != torch.float32
            or self.reference_operator_output.device.type != "cpu"
            or self.reference_residual_delta.device.type != "cpu"
        ):
            raise ValueError("attention route references are invalid")
        if (
            self.all_routes_operator_literal_parity_passed is not True
            or self.all_routes_delta_literal_parity_passed is not True
        ):
            raise ValueError("all-routes attention control failed")


def _native_route_factors(
    generator: Gemma3AttentionGenerator,
    normalized_input: Tensor,
    sequence: SequenceContext,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _validate_generator(generator)
    expected = (
        sequence.batch_size,
        sequence.query_length,
        generator.config.residual_width,
    )
    if (
        not isinstance(normalized_input, Tensor)
        or tuple(normalized_input.shape) != expected
        or normalized_input.dtype != torch.float32
        or normalized_input.device.type != "cpu"
        or not bool(torch.isfinite(normalized_input).all())
    ):
        raise ValueError("normalized attention input is invalid")
    if sequence.key_length != sequence.query_length:
        raise ValueError("attention fidelity supports prefill self-attention only")
    effective = normalized_input.masked_fill(
        ~sequence.query_valid_mask.unsqueeze(-1), 0
    )
    attention = generator.config.attention
    batch = sequence.batch_size
    length = sequence.query_length
    heads = attention.query_heads
    kv_heads = attention.key_value_heads
    dimension = attention.head_dimension
    query = F.linear(effective, generator.q_weight).view(
        batch, length, heads, dimension
    ).transpose(1, 2)
    key = F.linear(effective, generator.k_weight).view(
        batch, length, kv_heads, dimension
    ).transpose(1, 2)
    value = F.linear(effective, generator.v_weight).view(
        batch, length, kv_heads, dimension
    ).transpose(1, 2)
    if attention.qk_norm:
        if (
            generator.q_norm_weight is None
            or generator.k_norm_weight is None
            or generator.config.qk_norm is None
        ):
            raise RuntimeError("exact attention omitted Q/K norm")
        query = _rms_norm(
            query,
            generator.q_norm_weight,
            generator.config.qk_norm.epsilon,
        )
        key = _rms_norm(
            key,
            generator.k_norm_weight,
            generator.config.qk_norm.epsilon,
        )
    rope = attention.rope
    if rope is None or rope.theta is None:
        raise RuntimeError("exact attention omitted RoPE")
    inverse_frequency = 1.0 / (
        float(rope.theta)
        ** (
            torch.arange(0, dimension, 2, dtype=torch.float32) / dimension
        )
    )
    positions = sequence.logical_positions.float()
    if rope.scaling_kind == "linear":
        if rope.scaling_factor is None:
            raise RuntimeError("linear RoPE omitted scaling factor")
        positions = positions / float(rope.scaling_factor)
    frequencies = torch.einsum("bt,d->btd", positions, inverse_frequency)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos()
    sin = embedding.sin()
    query = query * cos.unsqueeze(1) + _rotate_half(query) * sin.unsqueeze(1)
    key = key * cos.unsqueeze(1) + _rotate_half(key) * sin.unsqueeze(1)
    repeats = heads // kv_heads
    repeated_key = _repeat_key_value(key, repeats)
    repeated_value = _repeat_key_value(value, repeats)
    scores = torch.matmul(query, repeated_key.transpose(-2, -1)) * float(
        attention.query_scale
    )
    if generator.config.logit_softcap is not None:
        cap = float(generator.config.logit_softcap)
        scores = torch.tanh(scores / cap) * cap
    allowed = _allowed_mask(sequence, attention.window_size)
    expanded = allowed[:, None, :, :].expand(batch, heads, length, length)
    scores = scores.masked_fill(~expanded, torch.finfo(torch.float32).min)
    execution_probabilities = F.softmax(
        scores, dim=-1, dtype=torch.float32
    ).contiguous()
    probabilities = execution_probabilities.masked_fill(
        ~expanded, 0
    ).contiguous()
    output_slices = generator.output_decoder_weight.detach().reshape(
        generator.config.residual_width, heads, dimension
    ).permute(1, 2, 0).contiguous()
    projected_values = torch.einsum(
        "bhkd,hdo->bhko", repeated_value, output_slices
    ).contiguous()
    return (
        probabilities,
        execution_probabilities,
        repeated_value,
        projected_values,
        allowed,
    )


def _operator_from_probabilities(
    generator: Gemma3AttentionGenerator,
    routed_probabilities: Tensor,
    repeated_values: Tensor,
    sequence: SequenceContext,
) -> tuple[Tensor, Tensor]:
    batch, heads, length, _ = routed_probabilities.shape
    context = torch.matmul(routed_probabilities, repeated_values)
    context = context.transpose(1, 2).contiguous().view(
        batch, length, generator.config.query_width
    )
    operator = F.linear(context, generator.output_decoder_weight)
    delta = _rms_norm(
        operator,
        generator.output_norm_weight,
        generator.config.output_norm.epsilon,
    ).masked_fill(~sequence.query_valid_mask.unsqueeze(-1), 0)
    return operator, delta


def operator_from_routed_probabilities(
    generator: Gemma3AttentionGenerator,
    routed_probabilities: Tensor,
    repeated_values: Tensor,
    sequence: SequenceContext,
) -> tuple[Tensor, Tensor]:
    """Public exact output projection for already-routed probabilities."""

    _validate_generator(generator)
    if (
        not isinstance(routed_probabilities, Tensor)
        or routed_probabilities.ndim != 4
        or routed_probabilities.dtype != torch.float32
        or routed_probabilities.device.type != "cpu"
        or not bool(torch.isfinite(routed_probabilities).all())
        or bool((routed_probabilities < 0).any())
        or not isinstance(repeated_values, Tensor)
        or repeated_values.dtype != torch.float32
        or repeated_values.device.type != "cpu"
        or repeated_values.shape[:2] != routed_probabilities.shape[:2]
        or repeated_values.shape[2] != routed_probabilities.shape[3]
    ):
        raise ValueError("routed attention factors are invalid")
    return _operator_from_probabilities(
        generator, routed_probabilities, repeated_values, sequence
    )


def attention_route_tensors(
    generator: Gemma3AttentionGenerator,
    normalized_input: Tensor,
    sequence: SequenceContext,
) -> AttentionRouteTensors:
    """Return exact native route factors and prove the all-route closure."""

    (
        probabilities,
        execution_probabilities,
        values,
        projected,
        allowed,
    ) = _native_route_factors(generator, normalized_input, sequence)
    packet = AttentionCoordinatePacket(
        values=normalized_input.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1), 0
        ),
        valid_token_mask=sequence.query_valid_mask,
        coordinate_system_sha256=generator.input_coordinate_system_sha256,
        semantic_role="normalized_input",
        source_invocation_id=generator.config.invocation_id,
    )
    reference = generator.forward_from_coordinates(packet, sequence)
    operator, delta = _operator_from_probabilities(
        generator, execution_probabilities, values, sequence
    )
    route_output_energies = (
        probabilities.to(torch.float64).square()
        * projected.to(torch.float64).square().sum(dim=-1)[:, :, None, :]
    ).masked_fill(~allowed[:, None, :, :], 0).contiguous()
    return AttentionRouteTensors(
        probabilities=probabilities,
        execution_probabilities=execution_probabilities,
        repeated_values=values,
        projected_values=projected,
        route_output_energies=route_output_energies,
        allowed_mask=allowed,
        reference_operator_output=reference.operator_output.detach().clone(),
        reference_residual_delta=reference.residual_delta.detach().clone(),
        all_routes_operator_literal_parity_passed=torch.equal(
            operator, reference.operator_output
        ),
        all_routes_delta_literal_parity_passed=torch.equal(
            delta, reference.residual_delta
        ),
    )


def select_global_attention_route_mask(
    probabilities: Tensor,
    intrinsic_fisher_energies: Tensor,
    route_output_energies: Tensor,
    allowed_mask: Tensor,
    *,
    ranker: str,
    k: int,
    row_catalog_id: str,
    layer_ordinal: int,
    random_salt: str = "primary",
) -> Tensor:
    """Return one deterministic global-head/key top-K prefix per query."""

    if ranker not in ATTENTION_EDGE_FIDELITY_RANKERS:
        raise ValueError("attention fidelity ranker is invalid")
    if k not in ATTENTION_EDGE_FIDELITY_K_VALUES:
        raise ValueError("attention fidelity K is not frozen")
    if random_salt not in ATTENTION_EDGE_FIDELITY_RANDOM_SALTS:
        raise ValueError("attention fidelity random salt is invalid")
    if ranker != "deterministic_random" and random_salt != "primary":
        raise ValueError("random replicates apply only to random ranking")
    if not isinstance(row_catalog_id, str) or not row_catalog_id:
        raise ValueError("row_catalog_id must be nonempty")
    if type(layer_ordinal) is not int or layer_ordinal < 0:
        raise ValueError("layer_ordinal must be nonnegative")
    if (
        not isinstance(probabilities, Tensor)
        or probabilities.ndim != 4
        or probabilities.shape[0] != 1
        or probabilities.device.type != "cpu"
        or not probabilities.is_floating_point()
        or not bool(torch.isfinite(probabilities).all())
        or not isinstance(intrinsic_fisher_energies, Tensor)
        or intrinsic_fisher_energies.shape != probabilities.shape
        or intrinsic_fisher_energies.device.type != "cpu"
        or not bool(torch.isfinite(intrinsic_fisher_energies).all())
        or not isinstance(route_output_energies, Tensor)
        or route_output_energies.shape != probabilities.shape
        or route_output_energies.device.type != "cpu"
        or not bool(torch.isfinite(route_output_energies).all())
        or not isinstance(allowed_mask, Tensor)
        or allowed_mask.dtype is not torch.bool
        or allowed_mask.device.type != "cpu"
        or allowed_mask.shape
        != (1, probabilities.shape[2], probabilities.shape[3])
    ):
        raise ValueError("attention route-selection tensors are invalid")
    heads = int(probabilities.shape[1])
    queries = int(probabilities.shape[2])
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    for query in range(queries):
        keys = tuple(
            int(item)
            for item in torch.nonzero(
                allowed_mask[0, query], as_tuple=False
            ).reshape(-1).tolist()
        )
        candidates: list[tuple[object, int, int]] = []
        for head in range(heads):
            for key in keys:
                if ranker == "fisher":
                    primary: object = -float(
                        intrinsic_fisher_energies[0, head, query, key].item()
                    )
                elif ranker == "probability":
                    primary = -float(probabilities[0, head, query, key].item())
                elif ranker == "route_output_energy":
                    primary = -float(
                        route_output_energies[0, head, query, key].item()
                    )
                elif ranker == "recency":
                    primary = query - key
                else:
                    primary = hashlib.sha256(
                        _RANDOM_DOMAIN
                        + random_salt.encode("ascii")
                        + b"\0"
                        + row_catalog_id.encode("utf-8")
                        + b"\0"
                        + str(layer_ordinal).encode("ascii")
                        + b"\0"
                        + str(query).encode("ascii")
                        + b"\0"
                        + str(head).encode("ascii")
                        + b"\0"
                        + str(key).encode("ascii")
                    ).digest()
                candidates.append((primary, head, key))
        candidates.sort()
        for _, head, key in candidates[:k]:
            mask[0, head, query, key] = True
    return mask.contiguous()


@dataclass(frozen=True, slots=True)
class SparseAttentionExecution:
    operator_output: Tensor
    residual_delta: Tensor
    routed_probabilities: Tensor
    selected_native_probability_mass: Tensor
    active_head_query_mask: Tensor


def sparse_attention_from_normalized(
    generator: Gemma3AttentionGenerator,
    normalized_input: Tensor,
    sequence: SequenceContext,
    selection_mask: Tensor,
    *,
    renormalize_rows: Tensor,
) -> SparseAttentionExecution:
    """Apply frozen masks to native probabilities from the current state."""

    (
        probabilities,
        execution_probabilities,
        values,
        _,
        allowed,
    ) = _native_route_factors(generator, normalized_input, sequence)
    if (
        not isinstance(selection_mask, Tensor)
        or selection_mask.dtype is not torch.bool
        or selection_mask.device.type != "cpu"
        or selection_mask.shape != probabilities.shape
        or bool((selection_mask & ~allowed[:, None, :, :]).any())
    ):
        raise ValueError("sparse attention selection mask is invalid")
    if (
        not isinstance(renormalize_rows, Tensor)
        or renormalize_rows.dtype is not torch.bool
        or renormalize_rows.device.type != "cpu"
        or tuple(renormalize_rows.shape) != (probabilities.shape[0],)
    ):
        raise ValueError("renormalize_rows must be CPU bool [batch]")
    selected = probabilities.masked_fill(~selection_mask, 0)
    mass = selected.sum(dim=-1, keepdim=True)
    renormalized = torch.where(mass > 0, selected / mass.clamp_min(1e-45), 0)
    routed = torch.where(
        renormalize_rows[:, None, None, None], renormalized, selected
    )
    # The public generator leaves a deterministic all-masked softmax row in
    # ``operator_output`` for an invalid query, then zeroes only the residual
    # delta.  Preserve those otherwise semantically inert bytes so the dense
    # all-route control can be literally equal even for padded batches.
    routed = torch.where(
        sequence.query_valid_mask[:, None, :, None],
        routed,
        execution_probabilities,
    )
    operator, delta = _operator_from_probabilities(
        generator, routed, values, sequence
    )
    return SparseAttentionExecution(
        operator_output=operator,
        residual_delta=delta,
        routed_probabilities=routed,
        selected_native_probability_mass=mass.squeeze(-1),
        active_head_query_mask=selection_mask.any(dim=-1),
    )


def _repeat_sequence(sequence: SequenceContext, batch_size: int) -> SequenceContext:
    if sequence.batch_size != 1 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("fidelity sequence repeat requires one source row")
    if sequence.cache_state is not None or sequence.phase != "prefill":
        raise ValueError("fidelity replay requires cache-free prefill")

    def repeated(value: Tensor) -> Tensor:
        return value.repeat(batch_size, *([1] * (value.ndim - 1)))

    cache_positions = sequence.cache_positions
    if cache_positions is not None and cache_positions.ndim == 2:
        cache_positions = repeated(cache_positions)
    return SequenceContext(
        query_valid_mask=repeated(sequence.query_valid_mask),
        key_valid_mask=repeated(sequence.key_valid_mask),
        logical_positions=repeated(sequence.logical_positions),
        key_logical_positions=repeated(sequence.key_logical_positions),
        cache_positions=cache_positions,
        phase=sequence.phase,
        input_origin=sequence.input_origin,
        cache_state=None,
        adapter_payload=None,
    )


@dataclass(frozen=True, slots=True)
class SparseWholeModelExecution:
    output: Tensor
    normalized_output: Tensor
    logits: Tensor
    attention_operator_outputs: tuple[Tensor, ...]
    attention_residual_deltas: tuple[Tensor, ...]
    mutation_residual_deltas: tuple[Tensor, ...]
    mutation_carriers: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if (
            len(self.attention_operator_outputs) != 18
            or len(self.attention_residual_deltas) != 18
            or len(self.mutation_residual_deltas) != 36
            or len(self.mutation_carriers) != 36
        ):
            raise ValueError("sparse replay omitted a mutation boundary")


def replay_exact_with_attention_masks(
    executor: nn.Module,
    input_ids: Tensor,
    sequence: SequenceContext,
    layer_selection_masks: tuple[Tensor, ...],
    *,
    renormalize_rows: Tensor,
    egress_query_ordinal: int | None = None,
) -> SparseWholeModelExecution:
    """Replay the transport-free exact executor under frozen attention masks."""

    mutations = getattr(executor, "mutations", None)
    transports = getattr(executor, "transports", None)
    if (
        getattr(executor, "arm_id", None) != JOINT_EXACT_CONTROL_ARM_ID
        or not isinstance(mutations, nn.ModuleList)
        or len(mutations) != 36
        or not isinstance(transports, nn.ModuleList)
        or len(transports) != 0
        or len(layer_selection_masks) != 18
        or not isinstance(input_ids, Tensor)
        or input_ids.ndim != 2
        or input_ids.dtype not in (torch.int32, torch.int64)
        or input_ids.device.type != "cpu"
    ):
        raise ValueError("attention fidelity requires the exact transport-free executor")
    batch_size = int(input_ids.shape[0])
    if sequence.batch_size == 1 and batch_size != 1:
        active_sequence = _repeat_sequence(sequence, batch_size)
    elif sequence.batch_size == batch_size:
        active_sequence = sequence
    else:
        raise ValueError("input batch and sequence context disagree")
    if (
        not isinstance(renormalize_rows, Tensor)
        or renormalize_rows.dtype is not torch.bool
        or renormalize_rows.device.type != "cpu"
        or tuple(renormalize_rows.shape) != (batch_size,)
    ):
        raise ValueError("renormalize_rows must align with replay batch")
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
            mask = layer_selection_masks[layer]
            if mask.shape[0] == 1 and batch_size != 1:
                mask = mask.repeat(batch_size, 1, 1, 1)
            sparse = sparse_attention_from_normalized(
                generator,
                normalized,
                active_sequence,
                mask,
                renormalize_rows=renormalize_rows,
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
            delta = mutation.execute_with_incoming(
                state, active_sequence, incoming
            ).delta
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
            or not bool(
                active_sequence.query_valid_mask[:, egress_query_ordinal].all()
            )
        ):
            raise ValueError("egress_query_ordinal is not a valid shared query")
        egress_input = normalized_output[
            :, egress_query_ordinal : egress_query_ordinal + 1
        ]
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
