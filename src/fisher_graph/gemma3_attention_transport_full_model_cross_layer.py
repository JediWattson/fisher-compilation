"""Frozen full-model attention transport with a shared cross-layer router.

The artifact in this module deliberately has no learned layer-id table.  The
eighteen Gemma attention invocations are mapped onto a fixed crossed layout of
three macroblocks and six sites.  A single generator dictionary is reused at
every invocation, with the only structured variation given by

``q[k, r] + depth[l] * beta[k, r] + gamma[site, k, r]``.

``gamma`` is centered across sites on construction.  Head mixing is shared by
site across all three macroblocks, and the router is one shared Top-2 function
conditioned on fixed depth/site/local-global coordinates and source-free live
context features.  Thus neither the bank nor the router contains an 18-row
free layer embedding.

The classes are frozen, source-free execution artifacts rather than training
modules.  Inputs are cloned to CPU float64, tensor digests are sealed, and all
public execution/hash/state operations fail closed if a retained tensor is
mutated after construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math

import torch
from torch import Tensor


INVOCATION_IDS = tuple(f"mutation.{2 * index + 1}" for index in range(18))
GLOBAL_ATTENTION_INVOCATION_IDS = (
    "mutation.11",
    "mutation.23",
    "mutation.35",
)
MACROBLOCK_COUNT = 3
SITE_COUNT = 6
TOP2_ACTIVE_COUNT = 2

FIXED_INVOCATION_FEATURE_NAMES = (
    "normalized_depth",
    "repeated_site_0",
    "repeated_site_1",
    "repeated_site_2",
    "repeated_site_3",
    "repeated_site_4",
    "repeated_site_5",
    "local_attention",
    "global_attention",
)
DYNAMIC_ROUTER_FEATURE_NAMES = (
    "log1p_contextual_rms",
    "signed_diagonal_alignment",
    "log1p_candidate_rms",
)
ROUTER_FEATURE_NAMES = (
    FIXED_INVOCATION_FEATURE_NAMES + DYNAMIC_ROUTER_FEATURE_NAMES
)

SHARED_BANK_SEMANTICS = (
    "shared_K_r_dictionary_with_site_shared_head_mixing_and_"
    "q_plus_normalized_continuous_depth_beta_plus_centered_site_gamma_v1"
)
SHARED_ROUTER_SEMANTICS = (
    "single_full_model_top2_router_fixed_depth_site_global_bit_v1"
)
FULL_MODEL_ARTIFACT_SEMANTICS = (
    "gemma3_270m_all_18_attention_invocations_cross_layer_transport_v1"
)
SITE_HEAD_MIXING_SEMANTICS = (
    "c_site_query_head_generator_shared_across_three_macroblocks_v1"
)
SINGULAR_DIAGONAL_SEMANTICS = (
    "q_k_r_plus_normalized_continuous_layer_depth_times_beta_k_r_plus_"
    "zero_sum_gamma_site_k_r_v1"
)

_TENSOR_DOMAIN = b"fisher_graph:full-model-cross-layer:tensor:v1\0"
_BANK_DOMAIN = b"fisher_graph:full-model-cross-layer:bank:v1\0"
_ROUTER_DOMAIN = b"fisher_graph:full-model-cross-layer:router:v1\0"
_ARTIFACT_DOMAIN = b"fisher_graph:full-model-cross-layer:artifact:v1\0"
_ALIGNMENT_FLOOR = 1.0e-12


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _cpu_float64(value: object, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or any(int(size) <= 0 for size in value.shape)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating rank-{ndim} tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _invocation_ordinal(invocation_id: str) -> int:
    if invocation_id not in INVOCATION_IDS:
        raise ValueError("invocation_id is absent from the fixed 18-site catalog")
    return INVOCATION_IDS.index(invocation_id)


@dataclass(frozen=True, slots=True)
class FullModelCrossLayerInvocationCoordinates:
    """Fixed, parameter-free coordinates for one attention invocation."""

    invocation_id: str
    ordinal: int
    macroblock_index: int
    site_index: int
    normalized_depth: float
    is_global_attention: bool

    def __post_init__(self) -> None:
        expected_ordinal = _invocation_ordinal(self.invocation_id)
        expected_block, expected_site = divmod(expected_ordinal, SITE_COUNT)
        expected_depth = float((2 * expected_ordinal - 17) / 17)
        expected_global = self.invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
        if (
            type(self.ordinal) is not int
            or self.ordinal != expected_ordinal
            or type(self.macroblock_index) is not int
            or self.macroblock_index != expected_block
            or type(self.site_index) is not int
            or self.site_index != expected_site
            or isinstance(self.normalized_depth, bool)
            or not isinstance(self.normalized_depth, (int, float))
            or float(self.normalized_depth) != expected_depth
            or type(self.is_global_attention) is not bool
            or self.is_global_attention is not expected_global
            or expected_global is not (expected_site == SITE_COUNT - 1)
        ):
            raise ValueError("invocation coordinates do not match the fixed layout")


def fixed_invocation_coordinates(
    invocation_id: str,
) -> FullModelCrossLayerInvocationCoordinates:
    """Return the fixed 3x6 coordinate of an invocation."""

    ordinal = _invocation_ordinal(invocation_id)
    macroblock_index, site_index = divmod(ordinal, SITE_COUNT)
    return FullModelCrossLayerInvocationCoordinates(
        invocation_id=invocation_id,
        ordinal=ordinal,
        macroblock_index=macroblock_index,
        site_index=site_index,
        normalized_depth=float((2 * ordinal - 17) / 17),
        is_global_attention=invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS,
    )


def fixed_invocation_encoding(invocation_id: str) -> Tensor:
    """Encode depth, site, and attention kind without learned layer state."""

    coordinates = fixed_invocation_coordinates(invocation_id)
    result = torch.zeros(
        len(FIXED_INVOCATION_FEATURE_NAMES),
        dtype=torch.float64,
    )
    result[0] = coordinates.normalized_depth
    result[1 + coordinates.site_index] = 1.0
    result[-2] = 0.0 if coordinates.is_global_attention else 1.0
    result[-1] = 1.0 if coordinates.is_global_attention else 0.0
    return result


@dataclass(frozen=True, slots=True)
class SharedCrossLayerBankExecution:
    """Inspectable source-free evaluation of the shared generator bank."""

    invocation_coordinates: FullModelCrossLayerInvocationCoordinates
    contextual_inputs: Tensor
    latent_coordinates: Tensor
    effective_singular_diagonal: Tensor
    candidate_outputs: Tensor
    dynamic_router_features: Tensor


@dataclass(frozen=True, slots=True, eq=False)
class SharedCrossLayerGeneratorBank:
    """One K/r dictionary with site-shared head mixing across all 18 sites."""

    left_factors: Tensor
    base_singular_diagonal: Tensor
    depth_singular_slopes: Tensor
    site_singular_offsets: Tensor
    right_factors: Tensor
    site_head_coefficients: Tensor
    _tensor_sha256s: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _artifact_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        left = _cpu_float64(self.left_factors, label="left_factors", ndim=3)
        base = _cpu_float64(
            self.base_singular_diagonal,
            label="base_singular_diagonal",
            ndim=2,
        )
        slopes = _cpu_float64(
            self.depth_singular_slopes,
            label="depth_singular_slopes",
            ndim=2,
        )
        offsets = _cpu_float64(
            self.site_singular_offsets,
            label="site_singular_offsets",
            ndim=3,
        )
        right = _cpu_float64(self.right_factors, label="right_factors", ndim=3)
        head_coefficients = _cpu_float64(
            self.site_head_coefficients,
            label="site_head_coefficients",
            ndim=3,
        )
        generator_count, hidden_width, generator_rank = left.shape
        if (
            generator_count < TOP2_ACTIVE_COUNT
            or generator_rank > hidden_width
            or right.shape != left.shape
            or base.shape != (generator_count, generator_rank)
            or slopes.shape != base.shape
            or offsets.shape != (SITE_COUNT, generator_count, generator_rank)
            or head_coefficients.shape[0] != SITE_COUNT
            or head_coefficients.shape[2] != generator_count
            or bool((base < 0.0).any())
        ):
            raise ValueError("shared generator bank shapes are invalid")
        identity = torch.eye(generator_rank, dtype=torch.float64)
        for label, factors in (("left", left), ("right", right)):
            for generator in factors:
                if not torch.allclose(
                    generator.T @ generator,
                    identity,
                    atol=1.0e-8,
                    rtol=1.0e-8,
                ):
                    raise ValueError(f"{label} generator factors must be orthonormal")

        # Center on construction so gamma has no intercept or free layer table.
        # Preserve already-centered serialized values bit-for-bit on reload.
        if not torch.allclose(
            offsets.sum(dim=0),
            torch.zeros_like(base),
            atol=1.0e-12,
            rtol=0.0,
        ):
            offsets = (offsets - offsets.mean(dim=0, keepdim=True)).contiguous()
        for name, value in (
            ("left_factors", left),
            ("base_singular_diagonal", base),
            ("depth_singular_slopes", slopes),
            ("site_singular_offsets", offsets),
            ("right_factors", right),
            ("site_head_coefficients", head_coefficients),
        ):
            object.__setattr__(self, name, value)
        tensor_sha256s = self._current_tensor_sha256s()
        object.__setattr__(self, "_tensor_sha256s", tensor_sha256s)
        object.__setattr__(
            self,
            "_artifact_sha256",
            hashlib.sha256(
                _BANK_DOMAIN + _canonical_json_bytes(self._hash_payload(tensor_sha256s))
            ).hexdigest(),
        )

    def _current_tensor_sha256s(self) -> tuple[str, ...]:
        return tuple(
            _tensor_sha256(value)
            for value in (
                self.left_factors,
                self.base_singular_diagonal,
                self.depth_singular_slopes,
                self.site_singular_offsets,
                self.right_factors,
                self.site_head_coefficients,
            )
        )

    def _hash_payload(self, tensor_sha256s: tuple[str, ...]) -> dict[str, object]:
        return {
            "bank_semantics": SHARED_BANK_SEMANTICS,
            "base_singular_diagonal_sha256": tensor_sha256s[1],
            "contains_native_value_or_output_weights": False,
            "contains_teacher_outputs_or_gradients": False,
            "depth_singular_slopes_sha256": tensor_sha256s[2],
            "generator_count": int(self.left_factors.shape[0]),
            "generator_rank": int(self.left_factors.shape[2]),
            "global_attention_invocation_ids": list(
                GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            "hidden_width": int(self.left_factors.shape[1]),
            "invocation_ids": list(INVOCATION_IDS),
            "left_factors_sha256": tensor_sha256s[0],
            "macroblock_count": MACROBLOCK_COUNT,
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "query_head_count": int(self.site_head_coefficients.shape[1]),
            "right_factors_sha256": tensor_sha256s[4],
            "singular_diagonal_semantics": SINGULAR_DIAGONAL_SEMANTICS,
            "site_count": SITE_COUNT,
            "site_head_coefficients_sha256": tensor_sha256s[5],
            "site_head_mixing_semantics": SITE_HEAD_MIXING_SEMANTICS,
            "site_singular_offsets_sha256": tensor_sha256s[3],
        }

    def _validate_unchanged(self) -> None:
        if self._current_tensor_sha256s() != self._tensor_sha256s:
            raise RuntimeError("shared cross-layer generator bank changed after construction")
        if not torch.allclose(
            self.site_singular_offsets.sum(dim=0),
            torch.zeros_like(self.base_singular_diagonal),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise RuntimeError("shared cross-layer site diagonal offsets lost zero-sum gauge")

    @property
    def generator_count(self) -> int:
        self._validate_unchanged()
        return int(self.left_factors.shape[0])

    @property
    def hidden_width(self) -> int:
        self._validate_unchanged()
        return int(self.left_factors.shape[1])

    @property
    def generator_rank(self) -> int:
        self._validate_unchanged()
        return int(self.left_factors.shape[2])

    @property
    def query_head_count(self) -> int:
        self._validate_unchanged()
        return int(self.site_head_coefficients.shape[1])

    @property
    def stored_parameter_count(self) -> int:
        self._validate_unchanged()
        return int(
            self.left_factors.numel()
            + self.base_singular_diagonal.numel()
            + self.depth_singular_slopes.numel()
            + self.site_singular_offsets.numel()
            + self.right_factors.numel()
            + self.site_head_coefficients.numel()
        )

    @property
    def artifact_sha256(self) -> str:
        self._validate_unchanged()
        return self._artifact_sha256

    def __hash__(self) -> int:
        return hash((SharedCrossLayerGeneratorBank, self.artifact_sha256))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SharedCrossLayerGeneratorBank)
            and self.artifact_sha256 == other.artifact_sha256
        )

    def effective_singular_diagonal(self, invocation_id: str) -> Tensor:
        self._validate_unchanged()
        coordinates = fixed_invocation_coordinates(invocation_id)
        return (
            self.base_singular_diagonal
            + coordinates.normalized_depth * self.depth_singular_slopes
            + self.site_singular_offsets[coordinates.site_index]
        ).contiguous()

    def execute(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> SharedCrossLayerBankExecution:
        """Evaluate all K candidates from live ``[row, head, hidden]`` contexts."""

        self._validate_unchanged()
        contexts = _cpu_float64(head_contexts, label="head_contexts", ndim=3)
        if contexts.shape[1:] != (self.query_head_count, self.hidden_width):
            raise ValueError(
                "head_contexts must match the bank query-head count and hidden width"
            )
        coordinates = fixed_invocation_coordinates(invocation_id)
        head_coefficients = self.site_head_coefficients[coordinates.site_index]
        contextual = torch.einsum(
            "nah,ak->nkh",
            contexts,
            head_coefficients,
        ).contiguous()
        latent = torch.einsum(
            "nkh,khr->nkr",
            contextual,
            self.right_factors,
        ).contiguous()
        diagonal = self.effective_singular_diagonal(invocation_id)
        scaled_latent = latent * diagonal[None]
        candidates = torch.einsum(
            "nkr,khr->nkh",
            scaled_latent,
            self.left_factors,
        ).contiguous()

        contextual_rms = contextual.square().mean(dim=-1).sqrt()
        candidate_rms = candidates.square().mean(dim=-1).sqrt()
        alignment_numerator = (latent * diagonal[None]).sum(dim=-1)
        alignment_denominator = (
            torch.linalg.vector_norm(latent, dim=-1)
            * torch.linalg.vector_norm(diagonal, dim=-1)[None]
        ).clamp_min(_ALIGNMENT_FLOOR)
        alignment = (alignment_numerator / alignment_denominator).clamp(-1.0, 1.0)
        dynamic_features = torch.stack(
            (
                torch.log1p(contextual_rms),
                alignment,
                torch.log1p(candidate_rms),
            ),
            dim=-1,
        ).contiguous()
        return SharedCrossLayerBankExecution(
            invocation_coordinates=coordinates,
            contextual_inputs=contextual,
            latent_coordinates=latent,
            effective_singular_diagonal=diagonal,
            candidate_outputs=candidates,
            dynamic_router_features=dynamic_features,
        )

    def candidate_outputs_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        return self.execute(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        ).candidate_outputs

    def state_dict(self) -> dict[str, object]:
        self._validate_unchanged()
        return {
            "artifact_sha256": self._artifact_sha256,
            "bank_semantics": SHARED_BANK_SEMANTICS,
            "base_singular_diagonal": self.base_singular_diagonal.clone(),
            "contains_native_value_or_output_weights": False,
            "contains_teacher_outputs_or_gradients": False,
            "depth_singular_slopes": self.depth_singular_slopes.clone(),
            "global_attention_invocation_ids": list(
                GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            "invocation_ids": list(INVOCATION_IDS),
            "left_factors": self.left_factors.clone(),
            "macroblock_count": MACROBLOCK_COUNT,
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "right_factors": self.right_factors.clone(),
            "singular_diagonal_semantics": SINGULAR_DIAGONAL_SEMANTICS,
            "site_count": SITE_COUNT,
            "site_head_coefficients": self.site_head_coefficients.clone(),
            "site_head_mixing_semantics": SITE_HEAD_MIXING_SEMANTICS,
            "site_singular_offsets": self.site_singular_offsets.clone(),
        }


_BANK_STATE_FIELDS = {
    "artifact_sha256",
    "bank_semantics",
    "base_singular_diagonal",
    "contains_native_value_or_output_weights",
    "contains_teacher_outputs_or_gradients",
    "depth_singular_slopes",
    "global_attention_invocation_ids",
    "invocation_ids",
    "left_factors",
    "macroblock_count",
    "native_output_rmsnorm_retained",
    "native_qk_retained",
    "right_factors",
    "singular_diagonal_semantics",
    "site_count",
    "site_head_coefficients",
    "site_head_mixing_semantics",
    "site_singular_offsets",
}


def shared_cross_layer_generator_bank_from_state_dict(
    state: Mapping[str, object],
) -> SharedCrossLayerGeneratorBank:
    """Reload and independently authenticate a shared generator bank."""

    if not isinstance(state, Mapping) or set(state) != _BANK_STATE_FIELDS:
        raise ValueError("shared cross-layer bank state fields are invalid")
    if (
        not _is_sha256(state["artifact_sha256"])
        or state["bank_semantics"] != SHARED_BANK_SEMANTICS
        or state["contains_native_value_or_output_weights"] is not False
        or state["contains_teacher_outputs_or_gradients"] is not False
        or state["global_attention_invocation_ids"]
        != list(GLOBAL_ATTENTION_INVOCATION_IDS)
        or state["invocation_ids"] != list(INVOCATION_IDS)
        or state["macroblock_count"] != MACROBLOCK_COUNT
        or state["native_output_rmsnorm_retained"] is not True
        or state["native_qk_retained"] is not True
        or state["singular_diagonal_semantics"]
        != SINGULAR_DIAGONAL_SEMANTICS
        or state["site_count"] != SITE_COUNT
        or state["site_head_mixing_semantics"] != SITE_HEAD_MIXING_SEMANTICS
    ):
        raise ValueError("shared cross-layer bank state semantics are invalid")
    raw_offsets = _cpu_float64(
        state["site_singular_offsets"],
        label="state site_singular_offsets",
        ndim=3,
    )
    if not torch.allclose(
        raw_offsets.sum(dim=0),
        torch.zeros_like(raw_offsets[0]),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("state site_singular_offsets are not zero-sum")
    bank = SharedCrossLayerGeneratorBank(
        left_factors=state["left_factors"],
        base_singular_diagonal=state["base_singular_diagonal"],
        depth_singular_slopes=state["depth_singular_slopes"],
        site_singular_offsets=raw_offsets,
        right_factors=state["right_factors"],
        site_head_coefficients=state["site_head_coefficients"],
    )
    if state["artifact_sha256"] != bank.artifact_sha256:
        raise ValueError("shared cross-layer bank artifact hash drifted")
    return bank


@dataclass(frozen=True, slots=True)
class SharedTop2CrossLayerRouterExecution:
    fixed_invocation_features: Tensor
    dynamic_features: Tensor
    router_features: Tensor
    selection_logits: Tensor
    selection_probabilities: Tensor
    sparse_selection_probabilities: Tensor
    signed_amplitudes: Tensor
    active_mask: Tensor
    generator_coefficients: Tensor


@dataclass(frozen=True, slots=True, eq=False)
class SharedTop2CrossLayerRouter:
    """A single Top-2 router shared by every full-model invocation."""

    generator_bank_sha256: str
    selection_weight: Tensor
    selection_bias: Tensor
    amplitude_weight: Tensor
    amplitude_bias: Tensor
    active_count: int = TOP2_ACTIVE_COUNT
    _tensor_sha256s: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _artifact_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not _is_sha256(self.generator_bank_sha256):
            raise ValueError("router generator-bank hash is invalid")
        selection_weight = _cpu_float64(
            self.selection_weight,
            label="selection_weight",
            ndim=2,
        )
        selection_bias = _cpu_float64(
            self.selection_bias,
            label="selection_bias",
            ndim=1,
        )
        amplitude_weight = _cpu_float64(
            self.amplitude_weight,
            label="amplitude_weight",
            ndim=2,
        )
        amplitude_bias = _cpu_float64(
            self.amplitude_bias,
            label="amplitude_bias",
            ndim=1,
        )
        if (
            selection_weight.shape[0] < TOP2_ACTIVE_COUNT
            or selection_weight.shape[1] != len(ROUTER_FEATURE_NAMES)
            or amplitude_weight.shape != selection_weight.shape
            or selection_bias.shape != (selection_weight.shape[0],)
            or amplitude_bias.shape != selection_bias.shape
            or type(self.active_count) is not int
            or self.active_count != TOP2_ACTIVE_COUNT
        ):
            raise ValueError("shared Top-2 router parameter shapes are invalid")
        for name, value in (
            ("selection_weight", selection_weight),
            ("selection_bias", selection_bias),
            ("amplitude_weight", amplitude_weight),
            ("amplitude_bias", amplitude_bias),
        ):
            object.__setattr__(self, name, value)
        tensor_sha256s = self._current_tensor_sha256s()
        object.__setattr__(self, "_tensor_sha256s", tensor_sha256s)
        payload = self._hash_payload(tensor_sha256s)
        object.__setattr__(
            self,
            "_artifact_sha256",
            hashlib.sha256(
                _ROUTER_DOMAIN + _canonical_json_bytes(payload)
            ).hexdigest(),
        )

    def _current_tensor_sha256s(self) -> tuple[str, ...]:
        return tuple(
            _tensor_sha256(value)
            for value in (
                self.selection_weight,
                self.selection_bias,
                self.amplitude_weight,
                self.amplitude_bias,
            )
        )

    def _hash_payload(self, tensor_sha256s: tuple[str, ...]) -> dict[str, object]:
        return {
            "active_count": self.active_count,
            "amplitude_bias_sha256": tensor_sha256s[3],
            "amplitude_weight_sha256": tensor_sha256s[2],
            "contains_oracle_outputs_or_coefficients": False,
            "contains_teacher_outputs_or_gradients": False,
            "dynamic_router_feature_names": list(DYNAMIC_ROUTER_FEATURE_NAMES),
            "fixed_invocation_feature_names": list(
                FIXED_INVOCATION_FEATURE_NAMES
            ),
            "generator_bank_sha256": self.generator_bank_sha256,
            "generator_count": int(self.selection_bias.shape[0]),
            "invocation_ids": list(INVOCATION_IDS),
            "router_semantics": SHARED_ROUTER_SEMANTICS,
            "selection_bias_sha256": tensor_sha256s[1],
            "selection_weight_sha256": tensor_sha256s[0],
        }

    def _validate_unchanged(self) -> None:
        if self._current_tensor_sha256s() != self._tensor_sha256s:
            raise RuntimeError("shared Top-2 router changed after construction")

    @property
    def generator_count(self) -> int:
        self._validate_unchanged()
        return int(self.selection_bias.shape[0])

    @property
    def feature_count(self) -> int:
        self._validate_unchanged()
        return int(self.selection_weight.shape[1])

    @property
    def stored_parameter_count(self) -> int:
        self._validate_unchanged()
        return int(
            self.selection_weight.numel()
            + self.selection_bias.numel()
            + self.amplitude_weight.numel()
            + self.amplitude_bias.numel()
        )

    @property
    def artifact_sha256(self) -> str:
        self._validate_unchanged()
        return self._artifact_sha256

    def __hash__(self) -> int:
        return hash((SharedTop2CrossLayerRouter, self.artifact_sha256))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SharedTop2CrossLayerRouter)
            and self.artifact_sha256 == other.artifact_sha256
        )

    def execute(
        self,
        *,
        invocation_id: str,
        dynamic_features: Tensor,
    ) -> SharedTop2CrossLayerRouterExecution:
        """Route source-free ``[row, generator, 3]`` live bank features."""

        self._validate_unchanged()
        dynamic = _cpu_float64(
            dynamic_features,
            label="dynamic_features",
            ndim=3,
        )
        if dynamic.shape[1:] != (
            self.generator_count,
            len(DYNAMIC_ROUTER_FEATURE_NAMES),
        ):
            raise ValueError("dynamic router features do not match the shared router")
        fixed = fixed_invocation_encoding(invocation_id)
        fixed_rows = fixed[None, None, :].expand(
            dynamic.shape[0],
            dynamic.shape[1],
            -1,
        )
        features = torch.cat((fixed_rows, dynamic), dim=-1).contiguous()
        logits = (
            torch.einsum("nkf,kf->nk", features, self.selection_weight)
            + self.selection_bias[None]
        )
        probabilities = torch.softmax(logits, dim=-1)
        amplitudes = (
            torch.einsum("nkf,kf->nk", features, self.amplitude_weight)
            + self.amplitude_bias[None]
        )

        # Stable descending argsort makes equal-logit ties select lower K indices.
        indices = torch.argsort(logits, dim=-1, descending=True, stable=True)[
            :, :TOP2_ACTIVE_COUNT
        ]
        active = torch.zeros_like(probabilities, dtype=torch.bool)
        active.scatter_(-1, indices, True)
        sparse = probabilities.masked_fill(~active, 0.0)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        coefficients = (sparse * amplitudes).contiguous()
        return SharedTop2CrossLayerRouterExecution(
            fixed_invocation_features=fixed,
            dynamic_features=dynamic,
            router_features=features,
            selection_logits=logits,
            selection_probabilities=probabilities,
            sparse_selection_probabilities=sparse,
            signed_amplitudes=amplitudes,
            active_mask=active,
            generator_coefficients=coefficients,
        )

    def state_dict(self) -> dict[str, object]:
        self._validate_unchanged()
        return {
            "active_count": self.active_count,
            "amplitude_bias": self.amplitude_bias.clone(),
            "amplitude_weight": self.amplitude_weight.clone(),
            "artifact_sha256": self._artifact_sha256,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_teacher_outputs_or_gradients": False,
            "dynamic_router_feature_names": list(DYNAMIC_ROUTER_FEATURE_NAMES),
            "fixed_invocation_feature_names": list(
                FIXED_INVOCATION_FEATURE_NAMES
            ),
            "generator_bank_sha256": self.generator_bank_sha256,
            "invocation_ids": list(INVOCATION_IDS),
            "router_semantics": SHARED_ROUTER_SEMANTICS,
            "selection_bias": self.selection_bias.clone(),
            "selection_weight": self.selection_weight.clone(),
        }


_ROUTER_STATE_FIELDS = {
    "active_count",
    "amplitude_bias",
    "amplitude_weight",
    "artifact_sha256",
    "contains_oracle_outputs_or_coefficients",
    "contains_teacher_outputs_or_gradients",
    "dynamic_router_feature_names",
    "fixed_invocation_feature_names",
    "generator_bank_sha256",
    "invocation_ids",
    "router_semantics",
    "selection_bias",
    "selection_weight",
}


def shared_top2_cross_layer_router_from_state_dict(
    state: Mapping[str, object],
) -> SharedTop2CrossLayerRouter:
    """Reload and independently authenticate the single shared router."""

    if not isinstance(state, Mapping) or set(state) != _ROUTER_STATE_FIELDS:
        raise ValueError("shared Top-2 router state fields are invalid")
    if (
        state["active_count"] != TOP2_ACTIVE_COUNT
        or not _is_sha256(state["artifact_sha256"])
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_teacher_outputs_or_gradients"] is not False
        or state["dynamic_router_feature_names"]
        != list(DYNAMIC_ROUTER_FEATURE_NAMES)
        or state["fixed_invocation_feature_names"]
        != list(FIXED_INVOCATION_FEATURE_NAMES)
        or not _is_sha256(state["generator_bank_sha256"])
        or state["invocation_ids"] != list(INVOCATION_IDS)
        or state["router_semantics"] != SHARED_ROUTER_SEMANTICS
    ):
        raise ValueError("shared Top-2 router state semantics are invalid")
    router = SharedTop2CrossLayerRouter(
        generator_bank_sha256=state["generator_bank_sha256"],
        selection_weight=state["selection_weight"],
        selection_bias=state["selection_bias"],
        amplitude_weight=state["amplitude_weight"],
        amplitude_bias=state["amplitude_bias"],
        active_count=state["active_count"],
    )
    if state["artifact_sha256"] != router.artifact_sha256:
        raise ValueError("shared Top-2 router artifact hash drifted")
    return router


@dataclass(frozen=True, slots=True)
class FullModelCrossLayerParameterAccounting:
    """Stored-parameter accounting; fixed coordinates cost zero parameters."""

    generator_factor_parameters: int
    singular_diagonal_parameters: int
    site_head_mixing_parameters: int
    bank_parameters: int
    router_parameters: int
    fixed_encoding_parameters: int
    free_layer_id_parameters: int
    compiled_parameters: int
    native_v_o_parameter_count: int | None
    parameter_reduction_fraction: float | None
    parameter_compression_eligible: bool | None

    def __post_init__(self) -> None:
        integer_fields = (
            self.generator_factor_parameters,
            self.singular_diagonal_parameters,
            self.site_head_mixing_parameters,
            self.bank_parameters,
            self.router_parameters,
            self.fixed_encoding_parameters,
            self.free_layer_id_parameters,
            self.compiled_parameters,
        )
        if (
            any(type(value) is not int or value < 0 for value in integer_fields)
            or self.generator_factor_parameters <= 0
            or self.singular_diagonal_parameters <= 0
            or self.site_head_mixing_parameters <= 0
            or self.router_parameters <= 0
            or self.fixed_encoding_parameters != 0
            or self.free_layer_id_parameters != 0
            or self.bank_parameters
            != self.generator_factor_parameters
            + self.singular_diagonal_parameters
            + self.site_head_mixing_parameters
            or self.compiled_parameters
            != self.bank_parameters + self.router_parameters
        ):
            raise ValueError("full-model parameter accounting does not close")
        if self.native_v_o_parameter_count is None:
            if (
                self.parameter_reduction_fraction is not None
                or self.parameter_compression_eligible is not None
            ):
                raise ValueError("native-free accounting cannot claim compression")
        elif (
            type(self.native_v_o_parameter_count) is not int
            or self.native_v_o_parameter_count <= 0
            or isinstance(self.parameter_reduction_fraction, bool)
            or not isinstance(self.parameter_reduction_fraction, float)
            or not math.isfinite(self.parameter_reduction_fraction)
            or type(self.parameter_compression_eligible) is not bool
            or not math.isclose(
                self.parameter_reduction_fraction,
                1.0 - self.compiled_parameters / self.native_v_o_parameter_count,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or self.parameter_compression_eligible
            is not (self.compiled_parameters < self.native_v_o_parameter_count)
        ):
            raise ValueError("native parameter comparison does not close")


@dataclass(frozen=True, slots=True)
class FullModelCrossLayerForward:
    invocation_id: str
    bank_execution: SharedCrossLayerBankExecution
    router_execution: SharedTop2CrossLayerRouterExecution
    operator_outputs: Tensor


@dataclass(frozen=True, slots=True, eq=False)
class FullModelCrossLayerAttentionTransport:
    """One immutable dictionary/router pair executable at all 18 invocations."""

    bank: SharedCrossLayerGeneratorBank
    router: SharedTop2CrossLayerRouter
    _bank_sha256: str = field(init=False, repr=False, compare=False)
    _router_sha256: str = field(init=False, repr=False, compare=False)
    _artifact_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bank, SharedCrossLayerGeneratorBank):
            raise TypeError("bank must be a SharedCrossLayerGeneratorBank")
        if not isinstance(self.router, SharedTop2CrossLayerRouter):
            raise TypeError("router must be a SharedTop2CrossLayerRouter")
        bank_sha256 = self.bank.artifact_sha256
        router_sha256 = self.router.artifact_sha256
        if (
            self.router.generator_bank_sha256 != bank_sha256
            or self.router.generator_count != self.bank.generator_count
        ):
            raise ValueError("shared router is not bound to the shared generator bank")
        object.__setattr__(self, "_bank_sha256", bank_sha256)
        object.__setattr__(self, "_router_sha256", router_sha256)
        object.__setattr__(
            self,
            "_artifact_sha256",
            hashlib.sha256(
                _ARTIFACT_DOMAIN
                + _canonical_json_bytes(self._hash_payload(bank_sha256, router_sha256))
            ).hexdigest(),
        )

    @staticmethod
    def _hash_payload(bank_sha256: str, router_sha256: str) -> dict[str, object]:
        return {
            "artifact_semantics": FULL_MODEL_ARTIFACT_SEMANTICS,
            "bank_sha256": bank_sha256,
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_teacher_outputs_or_gradients": False,
            "global_attention_invocation_ids": list(
                GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            "invocation_ids": list(INVOCATION_IDS),
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "router_sha256": router_sha256,
        }

    def _validate_unchanged(self) -> None:
        if (
            self.bank.artifact_sha256 != self._bank_sha256
            or self.router.artifact_sha256 != self._router_sha256
            or self.router.generator_bank_sha256 != self._bank_sha256
            or self.router.generator_count != self.bank.generator_count
        ):
            raise RuntimeError("full-model cross-layer artifact lineage changed")

    @property
    def artifact_sha256(self) -> str:
        self._validate_unchanged()
        return self._artifact_sha256

    def __hash__(self) -> int:
        return hash((FullModelCrossLayerAttentionTransport, self.artifact_sha256))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FullModelCrossLayerAttentionTransport)
            and self.artifact_sha256 == other.artifact_sha256
        )

    def parameter_accounting(
        self,
        *,
        native_v_o_parameter_count: int | None = None,
    ) -> FullModelCrossLayerParameterAccounting:
        self._validate_unchanged()
        if native_v_o_parameter_count is not None and (
            type(native_v_o_parameter_count) is not int
            or native_v_o_parameter_count <= 0
        ):
            raise ValueError("native_v_o_parameter_count must be positive or None")
        generator_factors = int(
            self.bank.left_factors.numel() + self.bank.right_factors.numel()
        )
        diagonals = int(
            self.bank.base_singular_diagonal.numel()
            + self.bank.depth_singular_slopes.numel()
            + self.bank.site_singular_offsets.numel()
        )
        head_mixing = int(self.bank.site_head_coefficients.numel())
        bank_parameters = generator_factors + diagonals + head_mixing
        router_parameters = self.router.stored_parameter_count
        compiled = bank_parameters + router_parameters
        reduction = (
            None
            if native_v_o_parameter_count is None
            else float(1.0 - compiled / native_v_o_parameter_count)
        )
        return FullModelCrossLayerParameterAccounting(
            generator_factor_parameters=generator_factors,
            singular_diagonal_parameters=diagonals,
            site_head_mixing_parameters=head_mixing,
            bank_parameters=bank_parameters,
            router_parameters=router_parameters,
            fixed_encoding_parameters=0,
            free_layer_id_parameters=0,
            compiled_parameters=compiled,
            native_v_o_parameter_count=native_v_o_parameter_count,
            parameter_reduction_fraction=reduction,
            parameter_compression_eligible=(
                None
                if native_v_o_parameter_count is None
                else compiled < native_v_o_parameter_count
            ),
        )

    def forward(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> FullModelCrossLayerForward:
        """Run the frozen synthetic/live-executor seam for one invocation."""

        self._validate_unchanged()
        bank_execution = self.bank.execute(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        router_execution = self.router.execute(
            invocation_id=invocation_id,
            dynamic_features=bank_execution.dynamic_router_features,
        )
        operator_outputs = torch.einsum(
            "nk,nkh->nh",
            router_execution.generator_coefficients,
            bank_execution.candidate_outputs,
        ).contiguous()
        return FullModelCrossLayerForward(
            invocation_id=invocation_id,
            bank_execution=bank_execution,
            router_execution=router_execution,
            operator_outputs=operator_outputs,
        )

    def __call__(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> FullModelCrossLayerForward:
        return self.forward(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )

    def state_dict(self) -> dict[str, object]:
        self._validate_unchanged()
        return {
            "artifact_semantics": FULL_MODEL_ARTIFACT_SEMANTICS,
            "artifact_sha256": self._artifact_sha256,
            "bank": self.bank.state_dict(),
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_teacher_outputs_or_gradients": False,
            "global_attention_invocation_ids": list(
                GLOBAL_ATTENTION_INVOCATION_IDS
            ),
            "invocation_ids": list(INVOCATION_IDS),
            "native_output_rmsnorm_retained": True,
            "native_qk_retained": True,
            "router": self.router.state_dict(),
        }


_ARTIFACT_STATE_FIELDS = {
    "artifact_semantics",
    "artifact_sha256",
    "bank",
    "contains_native_value_or_output_weights",
    "contains_oracle_outputs_or_coefficients",
    "contains_teacher_outputs_or_gradients",
    "global_attention_invocation_ids",
    "invocation_ids",
    "native_output_rmsnorm_retained",
    "native_qk_retained",
    "router",
}


def full_model_cross_layer_attention_transport_from_state_dict(
    state: Mapping[str, object],
) -> FullModelCrossLayerAttentionTransport:
    """Reload and authenticate the complete source-free execution artifact."""

    if not isinstance(state, Mapping) or set(state) != _ARTIFACT_STATE_FIELDS:
        raise ValueError("full-model cross-layer artifact state fields are invalid")
    if (
        state["artifact_semantics"] != FULL_MODEL_ARTIFACT_SEMANTICS
        or not _is_sha256(state["artifact_sha256"])
        or state["contains_native_value_or_output_weights"] is not False
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_teacher_outputs_or_gradients"] is not False
        or state["global_attention_invocation_ids"]
        != list(GLOBAL_ATTENTION_INVOCATION_IDS)
        or state["invocation_ids"] != list(INVOCATION_IDS)
        or state["native_output_rmsnorm_retained"] is not True
        or state["native_qk_retained"] is not True
        or not isinstance(state["bank"], Mapping)
        or not isinstance(state["router"], Mapping)
    ):
        raise ValueError("full-model cross-layer artifact safety semantics are invalid")
    bank = shared_cross_layer_generator_bank_from_state_dict(state["bank"])
    router = shared_top2_cross_layer_router_from_state_dict(state["router"])
    artifact = FullModelCrossLayerAttentionTransport(bank=bank, router=router)
    if state["artifact_sha256"] != artifact.artifact_sha256:
        raise ValueError("full-model cross-layer artifact hash drifted")
    return artifact


def check_full_model_cross_layer_attention_transport_state_dict(
    state: Mapping[str, object],
) -> FullModelCrossLayerAttentionTransport:
    """Fail closed on drift and return the independently reconstructed artifact."""

    return full_model_cross_layer_attention_transport_from_state_dict(state)


__all__ = [
    "DYNAMIC_ROUTER_FEATURE_NAMES",
    "FIXED_INVOCATION_FEATURE_NAMES",
    "FULL_MODEL_ARTIFACT_SEMANTICS",
    "FullModelCrossLayerAttentionTransport",
    "FullModelCrossLayerForward",
    "FullModelCrossLayerInvocationCoordinates",
    "FullModelCrossLayerParameterAccounting",
    "GLOBAL_ATTENTION_INVOCATION_IDS",
    "INVOCATION_IDS",
    "MACROBLOCK_COUNT",
    "ROUTER_FEATURE_NAMES",
    "SHARED_BANK_SEMANTICS",
    "SHARED_ROUTER_SEMANTICS",
    "SITE_COUNT",
    "SITE_HEAD_MIXING_SEMANTICS",
    "SINGULAR_DIAGONAL_SEMANTICS",
    "SharedCrossLayerBankExecution",
    "SharedCrossLayerGeneratorBank",
    "SharedTop2CrossLayerRouter",
    "SharedTop2CrossLayerRouterExecution",
    "TOP2_ACTIVE_COUNT",
    "check_full_model_cross_layer_attention_transport_state_dict",
    "fixed_invocation_coordinates",
    "fixed_invocation_encoding",
    "full_model_cross_layer_attention_transport_from_state_dict",
    "shared_cross_layer_generator_bank_from_state_dict",
    "shared_top2_cross_layer_router_from_state_dict",
]
