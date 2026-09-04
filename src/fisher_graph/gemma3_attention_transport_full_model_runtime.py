"""Live all-layer execution seam for shared Gemma attention V/O transport.

The capture diagnostics in :mod:`gemma3_attention_transport_compiler` operate
on native teacher hidden states.  This module is deliberately different: each
attention wrapper receives the *current* residual carrier from the candidate
executor, recomputes the native Q/K route on that carrier, and evaluates a
shared source-free V/O dictionary/router on the resulting head contexts.

Two backends use the same seam:

* :class:`NativeHeadCompositeRuntime` stores the exact per-head ``O @ V``
  composites.  It is an uncompressed parity control used to authenticate the
  all-18 replacement seam.
* :class:`FrozenCrossLayerTransportRuntime` stores one shared fitted bank and
  one shared Top-2 router.  It contains no native V or O projection tensor.

The surrounding joint executor keeps lexical ingress/readout, MLPs, terminal
normalization, residual order, native Q/K, RoPE, masking, and attention
softmax unchanged.  Consequently early V/O errors propagate into all later
layers during the same forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .adapters import SequenceContext
from .gemma3_attention_generator import (
    Gemma3AttentionGenerator,
    Gemma3AttentionGeneratorReceipt,
)
from .gemma3_attention_transport_full_model_cross_layer import (
    DYNAMIC_ROUTER_FEATURE_NAMES,
    FIXED_INVOCATION_FEATURE_NAMES,
    FullModelCrossLayerAttentionTransport,
    INVOCATION_IDS,
    fixed_invocation_encoding,
)
from .gemma3_attention_transport_full_model_partition import (
    MIDDLE_HOLDOUT_INVOCATION_IDS,
)
from .gemma3_joint_cross_block_chart_executor import (
    Gemma3JointCrossBlockChartExecutor,
    JointGraphGeneratorMutation,
)


ControlMode = Literal["normal", "zero_held", "permuted_held"]
CONTROL_MODES: tuple[ControlMode, ...] = (
    "normal",
    "zero_held",
    "permuted_held",
)

# A one-step cyclic permutation has no fixed point and moves the held global
# site onto a local structural encoding.  It is fixed before any model read.
HELD_INVOCATION_PERMUTATION = dict(
    zip(
        MIDDLE_HOLDOUT_INVOCATION_IDS,
        MIDDLE_HOLDOUT_INVOCATION_IDS[1:]
        + MIDDLE_HOLDOUT_INVOCATION_IDS[:1],
        strict=True,
    )
)

_WRAPPER_HASH_DOMAIN = b"fisher_graph:live-native-qk-shared-vo-wrapper:v1\0"
_RUNTIME_HASH_DOMAIN = b"fisher_graph:live-cross-layer-vo-runtime:v1\0"
_COMPOSITE_HASH_DOMAIN = b"fisher_graph:native-head-composite-runtime:v1\0"


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().to(device="cpu").contiguous()
    return tensor.view(torch.uint8).numpy().tobytes(order="C")


def _update_tensor_hash(digest: object, name: str, value: Tensor) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_tensor_bytes(value))
    digest.update(b"\0")


@dataclass(frozen=True, slots=True)
class LiveTransportForward:
    """Inspectable result from a V/O backend on flattened live rows."""

    invocation_id: str
    routing_invocation_id: str
    operator_outputs: Tensor
    generator_coefficients: Tensor | None
    active_mask: Tensor | None


class FrozenCrossLayerTransportRuntime(nn.Module):
    """Differentiable executor-local copy of an immutable fitted artifact."""

    def __init__(
        self,
        artifact: FullModelCrossLayerAttentionTransport,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if not isinstance(artifact, FullModelCrossLayerAttentionTransport):
            raise TypeError("artifact must be a full-model cross-layer transport")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("runtime dtype must be floating point")
        resolved_device = torch.device(device)
        bank = artifact.bank
        router = artifact.router
        for name, value in (
            ("left_factors", bank.left_factors),
            ("base_singular_diagonal", bank.base_singular_diagonal),
            ("depth_singular_slopes", bank.depth_singular_slopes),
            ("site_singular_offsets", bank.site_singular_offsets),
            ("right_factors", bank.right_factors),
            ("site_head_coefficients", bank.site_head_coefficients),
            ("selection_weight", router.selection_weight),
            ("selection_bias", router.selection_bias),
            ("amplitude_weight", router.amplitude_weight),
            ("amplitude_bias", router.amplitude_bias),
        ):
            self.register_buffer(
                name,
                value.detach().to(
                    device=resolved_device,
                    dtype=dtype,
                    copy=True,
                ).contiguous(),
            )
        self.register_buffer(
            "fixed_invocation_encodings",
            torch.stack(tuple(fixed_invocation_encoding(item) for item in INVOCATION_IDS))
            .to(device=resolved_device, dtype=dtype)
            .contiguous(),
        )
        self.source_artifact_sha256 = artifact.artifact_sha256
        self.generator_count = bank.generator_count
        self.generator_rank = bank.generator_rank
        self.hidden_width = bank.hidden_width
        self.query_head_count = bank.query_head_count
        self.active_count = router.active_count
        self.artifact_sha256 = self._compute_artifact_sha256()
        self.eval().requires_grad_(False)

    def _compute_artifact_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(_RUNTIME_HASH_DOMAIN)
        digest.update(self.source_artifact_sha256.encode("ascii"))
        digest.update(b"\0")
        for name, value in sorted(self.named_buffers()):
            _update_tensor_hash(digest, name, value)
        return digest.hexdigest()

    @property
    def stored_parameter_count(self) -> int:
        # Fixed invocation encodings are deterministic constants, not fitted
        # or stored artifact parameters.
        return int(
            sum(
                value.numel()
                for name, value in self.named_buffers()
                if name != "fixed_invocation_encodings"
            )
        )

    @staticmethod
    def routing_invocation_id(
        invocation_id: str,
        control_mode: ControlMode,
    ) -> str:
        if invocation_id not in INVOCATION_IDS:
            raise ValueError("invocation is outside the fixed attention catalog")
        if control_mode not in CONTROL_MODES:
            raise ValueError("unsupported full-model control mode")
        if control_mode == "permuted_held":
            return HELD_INVOCATION_PERMUTATION.get(invocation_id, invocation_id)
        return invocation_id

    def forward(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
        control_mode: ControlMode = "normal",
    ) -> LiveTransportForward:
        if (
            not isinstance(head_contexts, Tensor)
            or head_contexts.ndim != 3
            or head_contexts.shape[1:] != (
                self.query_head_count,
                self.hidden_width,
            )
            or head_contexts.dtype != self.left_factors.dtype
            or head_contexts.device != self.left_factors.device
            or not bool(torch.isfinite(head_contexts).all())
        ):
            raise ValueError("live head contexts do not match the shared runtime")
        routing_id = self.routing_invocation_id(invocation_id, control_mode)
        if (
            control_mode == "zero_held"
            and invocation_id in MIDDLE_HOLDOUT_INVOCATION_IDS
        ):
            rows = int(head_contexts.shape[0])
            return LiveTransportForward(
                invocation_id=invocation_id,
                routing_invocation_id=invocation_id,
                # Preserve the exact zero intervention while keeping the
                # executor's diagnostic gradient tape connected to the live
                # candidate state.
                operator_outputs=head_contexts.sum(dim=1).mul(0.0),
                generator_coefficients=torch.zeros(
                    rows,
                    self.generator_count,
                    dtype=head_contexts.dtype,
                    device=head_contexts.device,
                ),
                active_mask=torch.zeros(
                    rows,
                    self.generator_count,
                    dtype=torch.bool,
                    device=head_contexts.device,
                ),
            )

        ordinal = INVOCATION_IDS.index(routing_id)
        site_index = ordinal % 6
        fixed = self.fixed_invocation_encodings[ordinal]
        contextual = torch.einsum(
            "nah,ak->nkh",
            head_contexts,
            self.site_head_coefficients[site_index],
        )
        latent = torch.einsum(
            "nkh,khr->nkr",
            contextual,
            self.right_factors,
        )
        depth = fixed[0]
        diagonal = (
            self.base_singular_diagonal
            + depth * self.depth_singular_slopes
            + self.site_singular_offsets[site_index]
        )
        scaled = latent * diagonal.unsqueeze(0)
        candidates = torch.einsum("nkr,khr->nkh", scaled, self.left_factors)

        contextual_rms = contextual.float().square().mean(dim=-1).sqrt()
        candidate_rms = candidates.float().square().mean(dim=-1).sqrt()
        alignment_numerator = (latent.float() * diagonal.float().unsqueeze(0)).sum(
            dim=-1
        )
        alignment_denominator = (
            torch.linalg.vector_norm(latent.float(), dim=-1)
            * torch.linalg.vector_norm(diagonal.float(), dim=-1).unsqueeze(0)
        ).clamp_min(1.0e-12)
        alignment = (alignment_numerator / alignment_denominator).clamp(-1.0, 1.0)
        dynamic = torch.stack(
            (
                torch.log1p(contextual_rms),
                alignment,
                torch.log1p(candidate_rms),
            ),
            dim=-1,
        ).to(dtype=head_contexts.dtype)
        if dynamic.shape[-1] != len(DYNAMIC_ROUTER_FEATURE_NAMES):
            raise RuntimeError("dynamic router feature inventory drifted")
        fixed_rows = fixed.view(1, 1, -1).expand(
            dynamic.shape[0],
            dynamic.shape[1],
            -1,
        )
        if fixed_rows.shape[-1] != len(FIXED_INVOCATION_FEATURE_NAMES):
            raise RuntimeError("fixed router feature inventory drifted")
        features = torch.cat((fixed_rows, dynamic), dim=-1)
        logits = (
            torch.einsum("nkf,kf->nk", features, self.selection_weight)
            + self.selection_bias.unsqueeze(0)
        )
        probabilities = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        amplitudes = (
            torch.einsum("nkf,kf->nk", features, self.amplitude_weight)
            + self.amplitude_bias.unsqueeze(0)
        )
        indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[
            :, : self.active_count
        ]
        active = torch.zeros_like(probabilities, dtype=torch.bool)
        active.scatter_(-1, indices, True)
        sparse = probabilities.masked_fill(~active, 0)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(sparse.dtype).tiny
        )
        coefficients = sparse * amplitudes
        outputs = torch.einsum("nk,nkh->nh", coefficients, candidates)
        return LiveTransportForward(
            invocation_id=invocation_id,
            routing_invocation_id=routing_id,
            operator_outputs=outputs,
            generator_coefficients=coefficients,
            active_mask=active,
        )


class NativeHeadCompositeRuntime(nn.Module):
    """Exact per-head ``O @ V`` composite used only to prove seam parity."""

    def __init__(self, invocation_id: str, head_operators: Tensor) -> None:
        super().__init__()
        if invocation_id not in INVOCATION_IDS:
            raise ValueError("composite invocation is outside the fixed catalog")
        if (
            not isinstance(head_operators, Tensor)
            or head_operators.ndim != 3
            or head_operators.shape[1] != head_operators.shape[2]
            or not head_operators.is_floating_point()
            or not bool(torch.isfinite(head_operators).all())
        ):
            raise ValueError("head operators must be finite [head, hidden, hidden]")
        self.invocation_id = invocation_id
        self.query_head_count = int(head_operators.shape[0])
        self.hidden_width = int(head_operators.shape[1])
        self.register_buffer(
            "head_operators",
            head_operators.detach().clone().contiguous(),
        )
        digest = hashlib.sha256()
        digest.update(_COMPOSITE_HASH_DOMAIN)
        digest.update(invocation_id.encode("ascii"))
        digest.update(b"\0")
        _update_tensor_hash(digest, "head_operators", self.head_operators)
        self.artifact_sha256 = digest.hexdigest()
        self.eval().requires_grad_(False)

    def forward(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
        control_mode: ControlMode = "normal",
    ) -> LiveTransportForward:
        if control_mode != "normal":
            raise ValueError("native composite parity control supports normal mode only")
        if invocation_id != self.invocation_id:
            raise ValueError("native composite invoked at the wrong attention site")
        if head_contexts.shape[1:] != (
            self.query_head_count,
            self.hidden_width,
        ):
            raise ValueError("head contexts do not match native composites")
        output = torch.einsum(
            "aoh,nah->no",
            self.head_operators,
            head_contexts,
        )
        return LiveTransportForward(
            invocation_id=invocation_id,
            routing_invocation_id=invocation_id,
            operator_outputs=output,
            generator_coefficients=None,
            active_mask=None,
        )


@dataclass(frozen=True, slots=True)
class LiveSharedVOAttentionExecution:
    normalized_input: Tensor
    operator_output: Tensor
    residual_delta: Tensor
    emitted_generator_latents: Tensor
    receipt: Gemma3AttentionGeneratorReceipt
    transport_forward: LiveTransportForward
    route_factor_snapshot: None = None

    @property
    def delta(self) -> Tensor:
        return self.residual_delta


class Gemma3NativeQKSharedVOAttentionOperator(nn.Module):
    """Native Q/K attention whose V/O boundary is supplied by a live backend."""

    generator_kind = "native_qk_shared_vo"
    chart_count = 1
    incoming_chart_count = 1
    emitted_chart_count = 1
    incoming_generator_rank = 1
    emitted_generator_rank = 1
    supports_route_factor_capture = False
    source_checkpoint_coefficients_retained = True

    def __init__(
        self,
        exact: Gemma3AttentionGenerator,
        *,
        backend: FrozenCrossLayerTransportRuntime | NativeHeadCompositeRuntime,
        control_mode: ControlMode = "normal",
    ) -> None:
        super().__init__()
        if (
            not isinstance(exact, Gemma3AttentionGenerator)
            or exact.generator_kind != "exact_full_rank"
        ):
            raise TypeError("live wrapper requires an exact attention generator")
        if not isinstance(
            backend,
            (FrozenCrossLayerTransportRuntime, NativeHeadCompositeRuntime),
        ):
            raise TypeError("live wrapper backend is unsupported")
        if control_mode not in CONTROL_MODES:
            raise ValueError("live wrapper control mode is unsupported")
        if isinstance(backend, NativeHeadCompositeRuntime) and control_mode != "normal":
            raise ValueError("native composite wrapper must use normal mode")
        if exact.v_bias is not None or exact.o_bias is not None:
            raise ValueError("live V/O replacement requires bias-free native V/O")
        if backend.hidden_width != exact.hidden_width:
            raise ValueError("live V/O backend hidden width drifted")
        if backend.query_head_count != exact.config.attention.query_heads:
            raise ValueError("live V/O backend query-head count drifted")

        self.config = exact.config
        self.backend = backend
        self.control_mode = control_mode
        self.input_norm_weight = nn.Parameter(
            exact.input_norm_weight.detach().clone(), requires_grad=False
        )
        self.output_norm_weight = nn.Parameter(
            exact.output_norm_weight.detach().clone(), requires_grad=False
        )
        self.q_weight = nn.Parameter(exact.q_weight.detach().clone(), requires_grad=False)
        self.k_weight = nn.Parameter(exact.k_weight.detach().clone(), requires_grad=False)
        self.q_bias = (
            None
            if exact.q_bias is None
            else nn.Parameter(exact.q_bias.detach().clone(), requires_grad=False)
        )
        self.k_bias = (
            None
            if exact.k_bias is None
            else nn.Parameter(exact.k_bias.detach().clone(), requires_grad=False)
        )
        self.q_norm_weight = (
            None
            if exact.q_norm_weight is None
            else nn.Parameter(exact.q_norm_weight.detach().clone(), requires_grad=False)
        )
        self.k_norm_weight = (
            None
            if exact.k_norm_weight is None
            else nn.Parameter(exact.k_norm_weight.detach().clone(), requires_grad=False)
        )
        self.register_buffer(
            "inverse_frequency",
            exact.inverse_frequency.detach().clone(),
            persistent=False,
        )
        self.execution_count = 0
        self.native_q_projection_call_count = 0
        self.native_k_projection_call_count = 0
        self.native_v_projection_call_count = 0
        self.native_o_projection_call_count = 0
        self.shared_backend_call_count = 0
        self._artifact_sha256 = self._compute_artifact_sha256()
        self.eval().requires_grad_(False)

    @property
    def hidden_width(self) -> int:
        return int(self.config.residual_width)

    @property
    def input_coordinate_rank(self) -> int:
        return 1

    @property
    def output_coordinate_rank(self) -> int:
        return 1

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    def _compute_artifact_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(_WRAPPER_HASH_DOMAIN)
        digest.update(self.config.invocation_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.control_mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.backend.artifact_sha256.encode("ascii"))
        digest.update(b"\0")
        for name, value in sorted(self.named_parameters()):
            _update_tensor_hash(digest, name, value)
        _update_tensor_hash(digest, "inverse_frequency", self.inverse_frequency)
        return digest.hexdigest()

    def architecture_manifest(self) -> dict[str, object]:
        attention = self.config.attention
        rope = attention.rope
        assert rope is not None
        return {
            "kind": "gemma3_attention_generator",
            "generator_kind": self.generator_kind,
            "invocation_id": self.config.invocation_id,
            "source_independent_runtime": True,
            "source_fallback": False,
            "source_checkpoint_coefficients_retained": True,
            "native_qk_retained": True,
            "native_output_rmsnorm_retained": True,
            "native_v_projection_present": False,
            "native_o_projection_present": False,
            "contains_native_value_or_output_weights": False,
            "input_coordinate_rank": 1,
            "output_coordinate_rank": 1,
            "attention_contract": {
                "kind": attention.kind,
                "query_heads": attention.query_heads,
                "key_value_heads": attention.key_value_heads,
                "head_dimension": attention.head_dimension,
                "qk_norm": attention.qk_norm,
                "rope_theta": float(rope.theta),
                "rope_scaling_kind": rope.scaling_kind,
                "rope_scaling_factor": rope.scaling_factor,
                "window_size": attention.window_size,
                "logit_softcap": self.config.logit_softcap,
            },
            "control_mode": self.control_mode,
            "backend_kind": type(self.backend).__qualname__,
            "backend_artifact_sha256": self.backend.artifact_sha256,
            "artifact_sha256": self.artifact_sha256,
        }

    def _validate_sequence(self, sequence: SequenceContext) -> None:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill" or sequence.cache_state is not None:
            raise ValueError("live attention supports cache-free prefill only")
        if sequence.cache_positions is not None:
            raise ValueError("live attention rejects cache positions")
        if sequence.query_length != sequence.key_length:
            raise ValueError("live self-attention requires equal Q/K lengths")
        if not torch.equal(sequence.logical_positions, sequence.key_logical_positions):
            raise ValueError("query/key logical positions drifted")
        if not torch.equal(sequence.query_valid_mask, sequence.key_valid_mask):
            raise ValueError("query/key validity drifted")
        if sequence.device != self.q_weight.device:
            raise ValueError("sequence and live attention must share a device")

    def _allowed_pairs(self, sequence: SequenceContext) -> Tensor:
        query = torch.arange(sequence.query_length, device=sequence.device).view(
            1, sequence.query_length, 1
        )
        key = torch.arange(sequence.key_length, device=sequence.device).view(
            1, 1, sequence.key_length
        )
        allowed = key <= query
        allowed = allowed & sequence.key_valid_mask.unsqueeze(1)
        allowed = allowed & sequence.query_valid_mask.unsqueeze(2)
        if self.config.attention.window_size is not None:
            allowed = allowed & (query - key < self.config.attention.window_size)
        return allowed

    @staticmethod
    def _rotate_half(values: Tensor) -> Tensor:
        half = values.shape[-1] // 2
        return torch.cat((-values[..., half:], values[..., :half]), dim=-1)

    @staticmethod
    def _rms_norm(values: Tensor, weight: Tensor, spec: object) -> Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True)
            + float(getattr(spec, "epsilon"))
        )
        return (normalized * (1.0 + weight.float())).to(dtype=values.dtype)

    def _rotary_cos_sin(
        self,
        positions: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        rope = self.config.attention.rope
        assert rope is not None
        position_values = positions.float()
        if rope.scaling_kind == "linear":
            assert rope.scaling_factor is not None
            position_values = position_values / float(rope.scaling_factor)
        frequencies = torch.einsum(
            "bt,d->btd",
            position_values,
            self.inverse_frequency.float(),
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)

    def execute_with_incoming(
        self,
        residual_state: Tensor,
        sequence: SequenceContext,
        incoming_generator_latents: Tensor,
        *,
        normalized_input_perturbation: Tensor | None = None,
        capture_route_factors: bool = False,
    ) -> LiveSharedVOAttentionExecution:
        if capture_route_factors:
            raise ValueError("hybrid attention cannot export native V/O route factors")
        self._validate_sequence(sequence)
        expected_state = (
            sequence.batch_size,
            sequence.query_length,
            self.hidden_width,
        )
        if (
            not isinstance(residual_state, Tensor)
            or residual_state.shape != expected_state
            or residual_state.dtype != self.q_weight.dtype
            or residual_state.device != self.q_weight.device
        ):
            raise ValueError("live residual state does not match attention")
        expected_incoming = (*expected_state[:2], 1, 1)
        if (
            not isinstance(incoming_generator_latents, Tensor)
            or incoming_generator_latents.shape != expected_incoming
            or incoming_generator_latents.dtype != residual_state.dtype
            or incoming_generator_latents.device != residual_state.device
            or bool(torch.count_nonzero(incoming_generator_latents))
        ):
            raise ValueError("hybrid attention requires a zero rank-one packet")

        normalized = self._rms_norm(
            residual_state,
            self.input_norm_weight,
            self.config.input_norm,
        )
        if normalized_input_perturbation is not None:
            if (
                normalized_input_perturbation.shape != normalized.shape
                or normalized_input_perturbation.dtype != normalized.dtype
                or normalized_input_perturbation.device != normalized.device
            ):
                raise ValueError("normalized input perturbation does not align")
            normalized = normalized + normalized_input_perturbation.masked_fill(
                ~sequence.query_valid_mask.unsqueeze(-1), 0
            )

        attention = self.config.attention
        batch = sequence.batch_size
        length = sequence.query_length
        head_dimension = attention.head_dimension
        self.native_q_projection_call_count += 1
        query = F.linear(normalized, self.q_weight, self.q_bias).view(
            batch, length, attention.query_heads, head_dimension
        ).transpose(1, 2)
        self.native_k_projection_call_count += 1
        key = F.linear(normalized, self.k_weight, self.k_bias).view(
            batch, length, attention.key_value_heads, head_dimension
        ).transpose(1, 2)
        if attention.qk_norm:
            assert self.q_norm_weight is not None
            assert self.k_norm_weight is not None
            assert self.config.qk_norm is not None
            query = self._rms_norm(query, self.q_norm_weight, self.config.qk_norm)
            key = self._rms_norm(key, self.k_norm_weight, self.config.qk_norm)
        cos, sin = self._rotary_cos_sin(sequence.logical_positions, dtype=query.dtype)
        query = query * cos.unsqueeze(1) + self._rotate_half(query) * sin.unsqueeze(1)
        key = key * cos.unsqueeze(1) + self._rotate_half(key) * sin.unsqueeze(1)
        repeats = attention.query_heads // attention.key_value_heads
        if repeats != 1:
            key = (
                key[:, :, None, :, :]
                .expand(batch, attention.key_value_heads, repeats, length, head_dimension)
                .reshape(batch, attention.query_heads, length, head_dimension)
            )
        scores = torch.matmul(query, key.transpose(-2, -1)) * float(
            attention.query_scale
        )
        if self.config.logit_softcap is not None:
            cap = float(self.config.logit_softcap)
            scores = torch.tanh(scores / cap) * cap
        allowed = self._allowed_pairs(sequence)
        scores = scores.masked_fill(
            ~allowed.unsqueeze(1),
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            dtype=query.dtype
        )
        head_contexts = torch.einsum(
            "baqs,bsh->bqah",
            probabilities,
            normalized,
        )
        self.shared_backend_call_count += 1
        transport = self.backend(
            invocation_id=self.config.invocation_id,
            head_contexts=head_contexts.reshape(
                batch * length,
                attention.query_heads,
                self.hidden_width,
            ),
            control_mode=self.control_mode,
        )
        operator_output = transport.operator_outputs.reshape(
            batch, length, self.hidden_width
        )
        residual_delta = self._rms_norm(
            operator_output,
            self.output_norm_weight,
            self.config.output_norm,
        ).masked_fill(~sequence.query_valid_mask.unsqueeze(-1), 0)
        emitted = torch.zeros(
            batch,
            length,
            1,
            1,
            dtype=residual_state.dtype,
            device=residual_state.device,
        )
        self.execution_count += 1
        receipt = Gemma3AttentionGeneratorReceipt(
            invocation_id=self.config.invocation_id,
            generator_kind=self.generator_kind,
            attention_kind=attention.kind,
            input_coordinate_rank=1,
            output_coordinate_rank=1,
            logical_causal_key_pairs=int(allowed.sum().item()),
            source_projection_call_count=0,
            source_fallback_call_count=0,
        )
        return LiveSharedVOAttentionExecution(
            normalized_input=normalized,
            operator_output=operator_output,
            residual_delta=residual_delta,
            emitted_generator_latents=emitted,
            receipt=receipt,
            transport_forward=transport,
        )


def native_head_composites_from_exact_attention(
    exact: Gemma3AttentionGenerator,
) -> Tensor:
    """Build the exact per-query-head ``O_head @ V_kv_head`` controls."""

    if (
        not isinstance(exact, Gemma3AttentionGenerator)
        or exact.generator_kind != "exact_full_rank"
    ):
        raise TypeError("native head composites require exact attention")
    if exact.v_bias is not None or exact.o_bias is not None:
        raise ValueError("native head composite parity requires bias-free V/O")
    attention = exact.config.attention
    repeats = attention.query_heads // attention.key_value_heads
    operators = []
    for query_head in range(attention.query_heads):
        kv_head = query_head // repeats
        query_slice = slice(
            query_head * attention.head_dimension,
            (query_head + 1) * attention.head_dimension,
        )
        kv_slice = slice(
            kv_head * attention.head_dimension,
            (kv_head + 1) * attention.head_dimension,
        )
        operators.append(
            exact.output_decoder_weight[:, query_slice]
            @ exact.v_weight[kv_slice]
        )
    return torch.stack(operators).detach().contiguous()


def build_native_qk_shared_vo_executor(
    exact_executor: Gemma3JointCrossBlockChartExecutor,
    *,
    shared_runtime: FrozenCrossLayerTransportRuntime,
    control_mode: ControlMode = "normal",
) -> Gemma3JointCrossBlockChartExecutor:
    """Replace every exact attention V/O path with one shared live runtime."""

    if not isinstance(exact_executor, Gemma3JointCrossBlockChartExecutor):
        raise TypeError("exact_executor must be a joint Gemma executor")
    if control_mode not in CONTROL_MODES:
        raise ValueError("shared V/O control mode is unsupported")
    from .gemma3_joint_cross_block_chart_executor import (
        JOINT_EXACT_CONTROL_ARM_ID,
    )

    if exact_executor.arm_id != JOINT_EXACT_CONTROL_ARM_ID:
        raise ValueError("shared V/O build requires the exact control executor")
    mutations: list[JointGraphGeneratorMutation] = []
    for mutation in exact_executor.mutations:
        if mutation.event_spec.event_kind == "attention_mutation":
            exact = mutation.compiled_operator
            if not isinstance(exact, Gemma3AttentionGenerator):
                raise TypeError("exact attention inventory drifted")
            wrapper = Gemma3NativeQKSharedVOAttentionOperator(
                exact,
                backend=shared_runtime,
                control_mode=control_mode,
            )
            mutations.append(
                JointGraphGeneratorMutation(
                    event_spec=mutation.event_spec,
                    compiled_operator=wrapper,
                    source_checkpoint_coefficients_retained=True,
                )
            )
        else:
            mutations.append(mutation)
    return Gemma3JointCrossBlockChartExecutor(
        lexical=exact_executor.lexical,
        mutations=mutations,
        transports=(),
        terminal_norm=exact_executor.terminal_norm,
        final_logit_softcap=exact_executor.final_logit_softcap,
        # The sealed executor exposes no third arm.  Reuse its zero-transport
        # container while the wrapper manifests and experiment report carry
        # the explicit hybrid semantics; no exact-attention claim is made.
        arm_id=JOINT_EXACT_CONTROL_ARM_ID,
    )


def build_native_head_composite_parity_executor(
    exact_executor: Gemma3JointCrossBlockChartExecutor,
    *,
    head_operators_by_invocation: Mapping[str, Tensor] | None = None,
) -> Gemma3JointCrossBlockChartExecutor:
    """Replace all V/O paths by exact dense head composites for seam parity."""

    if not isinstance(exact_executor, Gemma3JointCrossBlockChartExecutor):
        raise TypeError("exact_executor must be a joint Gemma executor")
    from .gemma3_joint_cross_block_chart_executor import (
        JOINT_EXACT_CONTROL_ARM_ID,
    )

    if exact_executor.arm_id != JOINT_EXACT_CONTROL_ARM_ID:
        raise ValueError("composite parity build requires exact control executor")
    mutations: list[JointGraphGeneratorMutation] = []
    for mutation in exact_executor.mutations:
        if mutation.event_spec.event_kind == "attention_mutation":
            exact = mutation.compiled_operator
            if not isinstance(exact, Gemma3AttentionGenerator):
                raise TypeError("exact attention inventory drifted")
            head_operators = (
                native_head_composites_from_exact_attention(exact)
                if head_operators_by_invocation is None
                else head_operators_by_invocation[exact.config.invocation_id]
                .detach()
                .to(device=exact.device, dtype=exact.dtype)
                .contiguous()
            )
            backend = NativeHeadCompositeRuntime(
                exact.config.invocation_id,
                head_operators,
            )
            wrapper = Gemma3NativeQKSharedVOAttentionOperator(
                exact,
                backend=backend,
                control_mode="normal",
            )
            mutations.append(
                JointGraphGeneratorMutation(
                    event_spec=mutation.event_spec,
                    compiled_operator=wrapper,
                    source_checkpoint_coefficients_retained=True,
                )
            )
        else:
            mutations.append(mutation)
    return Gemma3JointCrossBlockChartExecutor(
        lexical=exact_executor.lexical,
        mutations=mutations,
        transports=(),
        terminal_norm=exact_executor.terminal_norm,
        final_logit_softcap=exact_executor.final_logit_softcap,
        arm_id=JOINT_EXACT_CONTROL_ARM_ID,
    )


def attention_wrapper_tripwire_payload(
    executor: Gemma3JointCrossBlockChartExecutor,
) -> dict[str, object]:
    """Return aggregate proof that the hybrid arm never called native V/O."""

    wrappers = tuple(
        mutation.compiled_operator
        for mutation in executor.mutations
        if mutation.event_spec.event_kind == "attention_mutation"
    )
    if len(wrappers) != 18 or any(
        not isinstance(item, Gemma3NativeQKSharedVOAttentionOperator)
        for item in wrappers
    ):
        raise ValueError("executor does not contain 18 hybrid attention wrappers")
    typed = tuple(item for item in wrappers if isinstance(item, Gemma3NativeQKSharedVOAttentionOperator))
    return {
        "attention_wrapper_count": len(typed),
        "execution_count": sum(item.execution_count for item in typed),
        "native_q_projection_call_count": sum(
            item.native_q_projection_call_count for item in typed
        ),
        "native_k_projection_call_count": sum(
            item.native_k_projection_call_count for item in typed
        ),
        "native_v_projection_call_count": sum(
            item.native_v_projection_call_count for item in typed
        ),
        "native_o_projection_call_count": sum(
            item.native_o_projection_call_count for item in typed
        ),
        "shared_backend_call_count": sum(item.shared_backend_call_count for item in typed),
        "native_v_o_calls_zero": all(
            item.native_v_projection_call_count == 0
            and item.native_o_projection_call_count == 0
            for item in typed
        ),
        "native_v_o_state_names_absent": all(
            not any(
                name.endswith("v_weight")
                or name.endswith("o_weight")
                or "output_decoder_weight" in name
                for name, _ in item.named_parameters()
            )
            for item in typed
        ),
    }


__all__ = [
    "CONTROL_MODES",
    "ControlMode",
    "FrozenCrossLayerTransportRuntime",
    "Gemma3NativeQKSharedVOAttentionOperator",
    "HELD_INVOCATION_PERMUTATION",
    "LiveSharedVOAttentionExecution",
    "LiveTransportForward",
    "NativeHeadCompositeRuntime",
    "attention_wrapper_tripwire_payload",
    "build_native_head_composite_parity_executor",
    "build_native_qk_shared_vo_executor",
    "native_head_composites_from_exact_attention",
]
