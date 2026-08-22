"""Additive raw-capture seam for the attention transport A/B/C experiment.

The chart-native worker remains byte-for-byte frozen.  This subclass calls its
public capture path exactly once, observes the already-detached execution
factors and the same single-VJP gradient bundle, authenticates each native V/O
decomposition, and emits one invocation at a time to a bounded sink.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .gemma3_attention_generator import (
    Gemma3AttentionGenerator,
    Gemma3AttentionRouteFactorSnapshot,
)
from .gemma3_attention_transport_compiler import (
    AttentionTransportCapture,
    AttentionTransportHeadOperatorCatalog,
    AttentionTransportRowBatch,
    attention_transport_capture_from_exact_attention,
    attention_transport_row_batch_from_capture,
)
from .gemma3_chart_native_neighbor_experiment import (
    CapturedBatch,
    ExecutionPreflightReceipt,
)
from .gemma3_chart_native_neighbor_runtime import (
    Gemma3ChartNativeCaptureRuntime,
)
from .gemma3_joint_cross_block_chart_executor import (
    JointGraphScoreGradientBundle,
)


_OUTPUT_NORM_DOMAIN = b"fisher_graph:attention-transport-output-norm:v1\0"


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _shifted_nll_route_factors(
    route_factors: Gemma3AttentionRouteFactorSnapshot,
) -> Gemma3AttentionRouteFactorSnapshot:
    """Restrict an exact route snapshot to shifted-NLL query rows.

    Native prefill routes include the final real query token, while the
    next-token objective has no label after that query.  Its score cotangent
    is consequently zero, but retaining the row would still charge the
    compiler's Euclidean Fisher floor.  Clone the snapshot, clear each
    example's final valid query, and zero only those now-invalid probability
    rows.  Every Q/K/V/O lineage field remains unchanged.
    """

    if not isinstance(route_factors, Gemma3AttentionRouteFactorSnapshot):
        raise TypeError("route factors must be a native attention snapshot")
    query_valid = route_factors.query_valid_mask.detach().clone()
    for batch_index in range(int(query_valid.shape[0])):
        positions = torch.nonzero(
            query_valid[batch_index],
            as_tuple=False,
        ).flatten()
        if positions.numel() < 2:
            raise ValueError(
                "shifted-NLL attention capture requires two valid tokens"
            )
        query_valid[batch_index, int(positions[-1])] = False
    probabilities = route_factors.probabilities.masked_fill(
        ~query_valid[:, None, :, None],
        0,
    )
    return replace(
        route_factors,
        probabilities=probabilities,
        query_valid_mask=query_valid,
    )


@dataclass(frozen=True, slots=True)
class AttentionTransportOutputNormSpec:
    invocation_id: str
    weight: Tensor
    epsilon: float
    scale_parameterization: str
    compute_dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("output norm invocation_id must be nonempty")
        if (
            not isinstance(self.weight, Tensor)
            or self.weight.ndim != 1
            or not self.weight.is_floating_point()
            or not bool(torch.isfinite(self.weight).all())
        ):
            raise ValueError("output norm weight must be a finite vector")
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (int, float))
            or not 0.0 < float(self.epsilon) < 1.0
            or self.scale_parameterization != "unit_offset"
            or self.compute_dtype != "float32"
        ):
            raise ValueError("output norm semantics are not Gemma RMSNorm")
        object.__setattr__(
            self,
            "weight",
            self.weight.detach().to(device="cpu", dtype=torch.float64).clone(),
        )

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "compute_dtype": self.compute_dtype,
            "epsilon": float(self.epsilon),
            "invocation_id": self.invocation_id,
            "scale_parameterization": self.scale_parameterization,
            "weight_sha256": _tensor_sha256(self.weight),
        }
        return hashlib.sha256(
            _OUTPUT_NORM_DOMAIN
            + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def apply(self, values: Tensor) -> Tensor:
        if (
            not isinstance(values, Tensor)
            or values.ndim != 2
            or values.shape[1] != self.weight.shape[0]
            or not values.is_floating_point()
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("output norm input must be a finite aligned matrix")
        source_dtype = values.dtype
        source_device = values.device
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True)
            + float(self.epsilon)
        )
        weight = self.weight.to(device=source_device, dtype=torch.float32)
        return (normalized * (1.0 + weight)).to(dtype=source_dtype)


@dataclass(frozen=True, slots=True)
class AttentionTransportCapturedInvocation:
    role: str
    base_capture_receipt_sha256: str
    capture: AttentionTransportCapture
    postnorm_targets: Tensor
    output_norm: AttentionTransportOutputNormSpec
    head_operator_catalog: AttentionTransportHeadOperatorCatalog | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("transport capture role must be nonempty")
        if (
            not isinstance(self.base_capture_receipt_sha256, str)
            or len(self.base_capture_receipt_sha256) != 64
        ):
            raise ValueError("base capture receipt must be a SHA-256")
        if not isinstance(self.capture, AttentionTransportCapture):
            raise TypeError("transport capture is invalid")
        if not isinstance(self.output_norm, AttentionTransportOutputNormSpec):
            raise TypeError("transport output norm spec is invalid")
        catalog = self.head_operator_catalog
        if catalog is None:
            catalog = AttentionTransportHeadOperatorCatalog(
                invocation_id=self.capture.invocation_id,
                head_operators=self.capture.head_operators,
                native_v_o_parameter_count=(
                    self.capture.native_v_o_parameter_count
                ),
            )
        elif not isinstance(catalog, AttentionTransportHeadOperatorCatalog):
            raise TypeError("transport head-operator catalog is invalid")
        catalog.validate_capture(self.capture)
        postnorm = self.postnorm_targets
        if (
            not isinstance(postnorm, Tensor)
            or postnorm.ndim != 2
            or postnorm.shape != self.capture.valid_operator_outputs.shape
            or not postnorm.is_floating_point()
            or not bool(torch.isfinite(postnorm).all())
            or self.output_norm.invocation_id != self.capture.invocation_id
        ):
            raise ValueError("postnorm targets do not align with transport rows")
        postnorm = postnorm.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).clone().contiguous()
        exact = self.output_norm.apply(self.capture.valid_operator_outputs)
        if not torch.allclose(exact, postnorm, atol=2.0e-5, rtol=2.0e-5):
            maximum = float((exact - postnorm).abs().max())
            raise ValueError(
                "captured postnorm delta does not close over operator output "
                f"(maximum absolute error {maximum:.6g})"
            )
        object.__setattr__(self, "postnorm_targets", postnorm)
        object.__setattr__(self, "head_operator_catalog", catalog)


@dataclass(frozen=True, slots=True)
class AttentionTransportCapturedRowBatch:
    """Bounded prompt-free rows retained after full capture authentication."""

    role: str
    base_capture_receipt_sha256: str
    capture: AttentionTransportRowBatch
    postnorm_targets: Tensor
    output_norm: AttentionTransportOutputNormSpec

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("transport row-batch role must be nonempty")
        if (
            not isinstance(self.base_capture_receipt_sha256, str)
            or len(self.base_capture_receipt_sha256) != 64
        ):
            raise ValueError("base capture receipt must be a SHA-256")
        if not isinstance(self.capture, AttentionTransportRowBatch):
            raise TypeError("transport row batch is invalid")
        if not isinstance(self.output_norm, AttentionTransportOutputNormSpec):
            raise TypeError("transport output norm spec is invalid")
        postnorm = self.postnorm_targets
        if (
            not isinstance(postnorm, Tensor)
            or postnorm.ndim != 2
            or postnorm.shape != self.capture.valid_operator_outputs.shape
            or not postnorm.is_floating_point()
            or not bool(torch.isfinite(postnorm).all())
            or self.output_norm.invocation_id != self.capture.invocation_id
        ):
            raise ValueError("postnorm targets do not align with transport rows")
        postnorm = postnorm.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).clone().contiguous()
        exact = self.output_norm.apply(self.capture.valid_operator_outputs)
        if not torch.allclose(exact, postnorm, atol=2.0e-5, rtol=2.0e-5):
            maximum = float((exact - postnorm).abs().max())
            raise ValueError(
                "sampled postnorm delta does not close over operator output "
                f"(maximum absolute error {maximum:.6g})"
            )
        object.__setattr__(self, "postnorm_targets", postnorm)


def sample_attention_transport_captured_invocation(
    captured: AttentionTransportCapturedInvocation,
    *,
    maximum_rows: int,
) -> AttentionTransportCapturedRowBatch:
    """Select content-independent, evenly spaced rows from one capture."""

    if not isinstance(captured, AttentionTransportCapturedInvocation):
        raise TypeError("captured must be an authenticated transport invocation")
    if type(maximum_rows) is not int or maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    count = min(maximum_rows, captured.capture.row_count)
    ordinal = torch.arange(count, dtype=torch.int64)
    indices = torch.div(
        (2 * ordinal + 1) * captured.capture.row_count,
        2 * count,
        rounding_mode="floor",
    )
    rows = attention_transport_row_batch_from_capture(
        captured.capture,
        row_indices=indices,
        head_operator_catalog=captured.head_operator_catalog,
    )
    return AttentionTransportCapturedRowBatch(
        role=captured.role,
        base_capture_receipt_sha256=captured.base_capture_receipt_sha256,
        capture=rows,
        postnorm_targets=captured.postnorm_targets.index_select(0, indices),
        output_norm=captured.output_norm,
    )


AttentionTransportCaptureSink = Callable[
    [AttentionTransportCapturedInvocation],
    None,
]


class Gemma3AttentionTransportCaptureRuntime(Gemma3ChartNativeCaptureRuntime):
    """Frozen chart-native runtime plus an ephemeral transport tensor sink."""

    def __init__(
        self,
        preflight: ExecutionPreflightReceipt | object,
        *,
        adapter: Gemma3CausalLMAdapter | object,
        tokenizer: object,
        sink: AttentionTransportCaptureSink,
        compilation: object | None = None,
    ) -> None:
        if not callable(sink):
            raise TypeError("attention transport sink must be callable")
        super().__init__(
            preflight,
            adapter=adapter,
            tokenizer=tokenizer,
            compilation=compilation,
        )
        self._transport_sink = sink
        self._transport_capture_active = False
        self._transport_backbone: (
            tuple[
                object,
                dict[str, Tensor],
                dict[str, object],
                object,
            ]
            | None
        ) = None
        self._transport_gradient_bundle: JointGraphScoreGradientBundle | None = None
        self._transport_head_operator_catalogs: dict[
            str,
            AttentionTransportHeadOperatorCatalog,
        ] = {}

    def _run_backbone(
        self,
        *,
        input_ids: Tensor,
        capture_routes: bool,
    ) -> tuple[object, dict[str, Tensor], dict[str, object], object]:
        result = super()._run_backbone(
            input_ids=input_ids,
            capture_routes=capture_routes,
        )
        if self._transport_capture_active:
            if self._transport_backbone is not None:
                raise RuntimeError("transport capture observed two backbones")
            self._transport_backbone = result
        return result

    def _validate_gradient_bundle(
        self,
        bundle: JointGraphScoreGradientBundle,
    ) -> None:
        Gemma3ChartNativeCaptureRuntime._validate_gradient_bundle(bundle)
        if self._transport_capture_active:
            if self._transport_gradient_bundle is not None:
                raise RuntimeError("transport capture observed two VJP bundles")
            self._transport_gradient_bundle = bundle

    def capture_batch(
        self,
        *,
        preflight: ExecutionPreflightReceipt,
        role: str,
        raw_examples: Sequence[Mapping[str, object]],
        role_manifest: Mapping[str, object],
    ) -> CapturedBatch:
        if self._transport_capture_active:
            raise RuntimeError("attention transport capture is not reentrant")
        self._transport_capture_active = True
        self._transport_backbone = None
        self._transport_gradient_bundle = None
        try:
            captured = super().capture_batch(
                preflight=preflight,
                role=role,
                raw_examples=raw_examples,
                role_manifest=role_manifest,
            )
            if (
                self._transport_backbone is None
                or self._transport_gradient_bundle is None
            ):
                raise RuntimeError("base runtime omitted transport capture state")
            observed, observations, routes, _sequence = self._transport_backbone
            execution = observed.execution
            gradients = self._transport_gradient_bundle
            operator_outputs = execution.attention_operator_outputs
            operator_gradients = gradients.attention_operator_output_gradients
            for mutation_index in range(1, 37, 2):
                event_id = f"mutation.{mutation_index}"
                operator_site = f"{event_id}.operator_output"
                mutation = self._executor.mutations[mutation_index - 1]
                generator = mutation.compiled_operator
                if not isinstance(generator, Gemma3AttentionGenerator):
                    raise TypeError("exact attention mutation generator drifted")
                transport = attention_transport_capture_from_exact_attention(
                    generator,
                    normalized_inputs=observations[f"{event_id}.input"],
                    route_factors=_shifted_nll_route_factors(
                        routes[event_id]
                    ),
                    operator_outputs=operator_outputs[operator_site],
                    operator_output_score_gradients=operator_gradients[
                        operator_site
                    ],
                )
                norm = generator.config.output_norm
                norm_spec = AttentionTransportOutputNormSpec(
                    invocation_id=transport.invocation_id,
                    weight=generator.output_norm_weight,
                    epsilon=float(norm.epsilon),
                    scale_parameterization=norm.scale_parameterization,
                    compute_dtype=norm.compute_dtype,
                )
                catalogs = getattr(
                    self,
                    "_transport_head_operator_catalogs",
                    None,
                )
                if not isinstance(catalogs, dict):
                    catalogs = {}
                    self._transport_head_operator_catalogs = catalogs
                operator_catalog = catalogs.get(transport.invocation_id)
                if operator_catalog is None:
                    operator_catalog = AttentionTransportHeadOperatorCatalog(
                        invocation_id=transport.invocation_id,
                        head_operators=transport.head_operators,
                        native_v_o_parameter_count=(
                            transport.native_v_o_parameter_count
                        ),
                    )
                    catalogs[transport.invocation_id] = operator_catalog
                postnorm = observations[f"{event_id}.output"][
                    transport.query_valid_mask
                ]
                self._transport_sink(
                    AttentionTransportCapturedInvocation(
                        role=role,
                        base_capture_receipt_sha256=(
                            captured.execution_receipt_sha256
                        ),
                        capture=transport,
                        postnorm_targets=postnorm,
                        output_norm=norm_spec,
                        head_operator_catalog=operator_catalog,
                    )
                )
                del transport
            return captured
        finally:
            self._transport_backbone = None
            self._transport_gradient_bundle = None
            self._transport_capture_active = False


def build_pinned_gemma3_attention_transport_capture_runtime(
    preflight: ExecutionPreflightReceipt,
    *,
    sink: AttentionTransportCaptureSink,
    cache_dir: Path | str | None = None,
) -> Gemma3AttentionTransportCaptureRuntime:
    """Load the setup-pinned CPU float32 Gemma into the additive runtime."""

    preflight.validate_receipt()
    if preflight.binding.test_only_allow_nonproduction_shape:
        raise ValueError("test-only runtimes require an injected constructor")
    source = preflight.setup["source_model"]
    model_id = str(source["model_id"])
    revision = str(source["model_revision"])
    tokenizer_revision = str(source.get("tokenizer_revision", ""))
    from .gemma3_chart_native_neighbor_setup import (
        PINNED_MODEL_ID,
        PINNED_MODEL_REVISION,
    )

    if (
        model_id != PINNED_MODEL_ID
        or revision != PINNED_MODEL_REVISION
        or tokenizer_revision != PINNED_MODEL_REVISION
    ):
        raise ValueError("runtime setup model/tokenizer pin drifted")
    from .gemma3_chart_native_neighbor_fresh_panels import (
        authenticate_tokenizer_assets,
    )
    from .gemma3_experiment import load_gemma3, resolve_gemma3_huggingface_paths

    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    snapshot = (
        cache
        / "models--google--gemma-3-270m"
        / "snapshots"
        / revision
    )
    authenticate_tokenizer_assets(snapshot)
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=torch.device("cpu"),
        dtype="float32",
        local_files_only=True,
    )
    if not isinstance(model, nn.Module):
        raise TypeError("pinned Gemma loader returned an invalid model")
    model.to(device="cpu", dtype=torch.float32)
    model.eval().requires_grad_(False)
    return Gemma3AttentionTransportCaptureRuntime(
        preflight,
        adapter=Gemma3CausalLMAdapter(model),
        tokenizer=tokenizer,
        sink=sink,
    )


__all__ = [
    "AttentionTransportCapturedInvocation",
    "AttentionTransportCapturedRowBatch",
    "AttentionTransportCaptureSink",
    "AttentionTransportOutputNormSpec",
    "Gemma3AttentionTransportCaptureRuntime",
    "build_pinned_gemma3_attention_transport_capture_runtime",
    "sample_attention_transport_captured_invocation",
]
