"""Source-free materialization of the Gemma 3 layer-multimode MLP map.

The fitted map predicts the teacher delta *after* the native MLP output
RMSNorm.  A compiled decoder layer therefore has the following boundary::

    x = pre_feedforward_layernorm(residual)
    delta = compiled_multimode_mlp(x)
    residual = residual + delta

The native gate/up/down matrices and ``post_feedforward_layernorm`` are not
part of the deployed graph.  Attention, lexical weights, input RMSNorms, and
the terminal path are deep-copied exactly.  This module deliberately contains
no native-MLP fallback and never calls a teacher model during compilation or
loading.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import InitVar, dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Final
import weakref

import torch
from torch import Tensor, nn

from .adapters.gemma3 import Gemma3CausalLMAdapter
from .gemma3_layer_multimode_capture import ADevelopmentPartitionReceipt
from .gemma3_layer_multimode_materialization import (
    LayerMLPFlowMaterializationInventoryReceipt,
)
from .gemma3_layer_multimode_selection import LayerMultimodeSelectionReceipt


__all__ = [
    "COMPILED_LAYER_MULTIMODE_EQUATION_ID",
    "CompiledGemma3LayerMultimodeModel",
    "Float32LayerMultimodeMLP",
    "FrozenLayerMultimodeCoefficients",
    "Gemma3LayerMultimodeCompilationReceipt",
    "GEMMA3_PRODUCTION_HIDDEN_WIDTH",
    "GEMMA3_PRODUCTION_INTERMEDIATE_WIDTH",
    "LayerMultimodeBundleReceipt",
    "LayerMultimodeCompilationError",
    "LayerMultimodeLayerAccounting",
    "MAX_LAYER_MULTIMODE_BUNDLE_BYTES",
    "SelectedLayerMultimodeCoefficientInventory",
    "build_selected_layer_multimode_coefficient_inventory",
    "compile_gemma3_layer_multimode_model",
    "load_compiled_layer_multimode_bundle",
    "save_compiled_layer_multimode_bundle",
]


GEMMA3_LAYER_COUNT: Final = 18
GEMMA3_PRODUCTION_HIDDEN_WIDTH: Final = 640
GEMMA3_PRODUCTION_INTERMEDIATE_WIDTH: Final = 2048
COMPILED_LAYER_MULTIMODE_EQUATION_ID: Final = (
    "normal_squared_chart_softmax.affine_z_router."
    "fused_mode_coordinate.single_basis_decode.direct_post_norm_delta.v1"
)
_COMPILATION_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode_compilation.v1"
)
_BUNDLE_SCHEMA: Final = "fisher_graph.gemma3_layer_multimode_bundle.v1"
_FORMAT_VERSION: Final = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COEFFICIENT_DOMAIN = b"fisher_graph:gemma3-layer-multimode-coefficients:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:gemma3-layer-multimode-compiled-tensor:v1\0"
_STATE_DOMAIN = b"fisher_graph:gemma3-layer-multimode-compiled-state:v1\0"
_ARCHITECTURE_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode-protected-architecture:v1\0"
)
_RECEIPT_DOMAIN = b"fisher_graph:gemma3-layer-multimode-compilation-receipt:v1\0"
_MANIFEST_DOMAIN = b"fisher_graph:gemma3-layer-multimode-bundle-manifest:v1\0"
_INVENTORY_DOMAIN = b"fisher_graph:gemma3-layer-multimode-selected-inventory:v1\0"
_INVENTORY_BUILDER_AUTHORITY = object()
_LIVE_SELECTED_INVENTORIES: dict[int, weakref.ReferenceType[object]] = {}
_FLOAT32_ORTHONORMAL_ATOL = 5.0e-5
MAX_LAYER_MULTIMODE_BUNDLE_BYTES: Final = 2_500_000_000
_READ_CHUNK_BYTES: Final = 1024 * 1024
_COEFFICIENT_NAMES = (
    "bases",
    "centers",
    "generator_bias",
    "generator_weight",
    "log_chart_temperature",
    "router_bias",
    "router_weight",
)


class LayerMultimodeCompilationError(ValueError):
    """A model or bundle cannot satisfy the frozen deployment boundary."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LayerMultimodeCompilationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LayerMultimodeCompilationError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return value


def _strict_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise LayerMultimodeCompilationError(f"{label} must be an exact bool")
    return value


def _finite_float(value: object, *, label: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise LayerMultimodeCompilationError(f"{label} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise LayerMultimodeCompilationError(f"{label} must be positive")
    return result


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout != torch.strided:
        raise TypeError("only dense strided tensors can be authenticated")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        ).encode("utf-8")
    )
    # ``view(dtype)`` rejects rank-zero tensors when element sizes differ.
    digest.update(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


def _state_inventory(state: Mapping[str, Tensor]) -> tuple[tuple[str, str], ...]:
    if not isinstance(state, Mapping):
        raise TypeError("state must be a tensor mapping")
    pairs: list[tuple[str, str]] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise TypeError("state entries must be named tensors")
        pairs.append((name, _tensor_sha256(value)))
    return tuple(pairs)


def _state_sha256(state: Mapping[str, Tensor]) -> str:
    return _digest(_STATE_DOMAIN, [list(pair) for pair in _state_inventory(state)])


def _cpu_float32_clone(value: object, *, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise LayerMultimodeCompilationError(
            f"{label} must be a CPU torch.float32 tensor"
        )
    if value.requires_grad:
        raise LayerMultimodeCompilationError(f"{label} must be detached")
    if value.ndim != ndim or any(int(width) <= 0 for width in value.shape):
        raise LayerMultimodeCompilationError(
            f"{label} must be a nonempty rank-{ndim} tensor"
        )
    if not bool(torch.isfinite(value).all()):
        raise LayerMultimodeCompilationError(f"{label} must be finite")
    return value.detach().clone().contiguous()


@dataclass(frozen=True, slots=True)
class FrozenLayerMultimodeCoefficients:
    """Independent float32 coefficients for one selected layer candidate."""

    layer_ordinal: int
    centers: Tensor
    bases: Tensor
    router_weight: Tensor
    router_bias: Tensor
    generator_weight: Tensor
    generator_bias: Tensor
    log_chart_temperature: Tensor
    source_flow_model_parameter_sha256: str
    source_fit_receipt_sha256: str
    selection_receipt_sha256: str
    mode_temperature: float = 1.0
    coefficient_sha256: str = ""

    def __post_init__(self) -> None:
        ordinal = _strict_int(self.layer_ordinal, label="layer_ordinal")
        if ordinal >= GEMMA3_LAYER_COUNT:
            raise LayerMultimodeCompilationError(
                "layer_ordinal must be in the 18-layer stack"
            )
        centers = _cpu_float32_clone(self.centers, label="centers", ndim=2)
        bases = _cpu_float32_clone(self.bases, label="bases", ndim=3)
        router_weight = _cpu_float32_clone(
            self.router_weight, label="router_weight", ndim=3
        )
        router_bias = _cpu_float32_clone(
            self.router_bias, label="router_bias", ndim=2
        )
        generator_weight = _cpu_float32_clone(
            self.generator_weight, label="generator_weight", ndim=4
        )
        generator_bias = _cpu_float32_clone(
            self.generator_bias, label="generator_bias", ndim=3
        )
        log_temperature = _cpu_float32_clone(
            self.log_chart_temperature,
            label="log_chart_temperature",
            ndim=0,
        )
        c, h = (int(v) for v in centers.shape)
        if c not in (1, 2):
            raise LayerMultimodeCompilationError("chart count must be C=1 or C=2")
        if bases.shape[:2] != centers.shape:
            raise LayerMultimodeCompilationError("basis C/H axes must match centers")
        r = int(bases.shape[2])
        if r > h:
            raise LayerMultimodeCompilationError("tangent rank cannot exceed H")
        if router_weight.shape[0] != c or router_weight.shape[2] != r:
            raise LayerMultimodeCompilationError("router shape must be [C,M,R]")
        m = int(router_weight.shape[1])
        if m not in (1, 2, 4):
            raise LayerMultimodeCompilationError("mode count must be M=1, M=2, or M=4")
        if tuple(router_bias.shape) != (c, m):
            raise LayerMultimodeCompilationError("router bias shape must be [C,M]")
        if tuple(generator_weight.shape) != (c, m, r, r):
            raise LayerMultimodeCompilationError(
                "generator weight shape must be [C,M,R,R]"
            )
        if tuple(generator_bias.shape) != (c, m, r):
            raise LayerMultimodeCompilationError(
                "generator bias shape must be [C,M,R]"
            )
        gram = torch.einsum("chr,chs->crs", bases, bases)
        identity = torch.eye(r, dtype=torch.float32).expand_as(gram)
        if float((gram - identity).abs().max()) > _FLOAT32_ORTHONORMAL_ATOL:
            raise LayerMultimodeCompilationError(
                "float32 chart basis columns must remain orthonormal"
            )
        chart_temperature = float(log_temperature.exp())
        if not math.isfinite(chart_temperature) or chart_temperature <= 0.0:
            raise LayerMultimodeCompilationError(
                "serialized chart temperature must be positive"
            )
        if c == 1 and float(log_temperature) != 0.0:
            raise LayerMultimodeCompilationError("C=1 requires tau_c exactly 1.0")
        if type(self.mode_temperature) is not float:
            raise LayerMultimodeCompilationError(
                "mode_temperature must be an exact JSON float"
            )
        mode_temperature = _finite_float(
            self.mode_temperature, label="mode_temperature", positive=True
        )
        if mode_temperature != 1.0:
            raise LayerMultimodeCompilationError(
                "deployment mode temperature is frozen exactly 1.0"
            )
        _require_sha256(
            self.source_flow_model_parameter_sha256,
            label="source_flow_model_parameter_sha256",
        )
        _require_sha256(
            self.source_fit_receipt_sha256, label="source_fit_receipt_sha256"
        )
        _require_sha256(
            self.selection_receipt_sha256, label="selection_receipt_sha256"
        )
        for name, tensor in (
            ("centers", centers),
            ("bases", bases),
            ("router_weight", router_weight),
            ("router_bias", router_bias),
            ("generator_weight", generator_weight),
            ("generator_bias", generator_bias),
            ("log_chart_temperature", log_temperature),
        ):
            object.__setattr__(self, name, tensor)
        object.__setattr__(self, "mode_temperature", mode_temperature)
        computed = _digest(_COEFFICIENT_DOMAIN, self._hash_payload())
        if self.coefficient_sha256 == "":
            object.__setattr__(self, "coefficient_sha256", computed)
        elif _require_sha256(
            self.coefficient_sha256, label="coefficient_sha256"
        ) != computed:
            raise LayerMultimodeCompilationError("coefficient hash mismatch")

    @classmethod
    def from_float64_flow(
        cls,
        flow: nn.Module,
        *,
        source_fit_receipt_sha256: str,
        selection_receipt_sha256: str,
    ) -> "FrozenLayerMultimodeCoefficients":
        """Clone a fitted float64 flow without retaining any source aliases."""

        from .gemma3_layer_multimode_flow import AffineLocalLayerMultimodeFlow

        if type(flow) is not AffineLocalLayerMultimodeFlow:
            raise TypeError("flow must be an AffineLocalLayerMultimodeFlow")
        if flow.config.mode_temperature != 1.0:
            raise LayerMultimodeCompilationError(
                "selected flow must use frozen mode temperature 1.0"
            )
        if flow.atlas.chart_count == 2 and not flow.config.fit_chart_temperature:
            raise LayerMultimodeCompilationError(
                "selected C=2 flow must A-fit its chart temperature"
            )
        return cls(
            layer_ordinal=flow.config.layer_ordinal,
            centers=flow.centers.detach().to(torch.float32).cpu(),
            bases=flow.bases.detach().to(torch.float32).cpu(),
            router_weight=flow.router_weight.detach().to(torch.float32).cpu(),
            router_bias=flow.router_bias.detach().to(torch.float32).cpu(),
            generator_weight=flow.generator_weight.detach().to(torch.float32).cpu(),
            generator_bias=flow.generator_bias.detach().to(torch.float32).cpu(),
            log_chart_temperature=(
                flow.log_chart_temperature.detach().to(torch.float32).cpu()
            ),
            source_flow_model_parameter_sha256=flow.model_parameter_sha256,
            source_fit_receipt_sha256=source_fit_receipt_sha256,
            selection_receipt_sha256=selection_receipt_sha256,
            mode_temperature=flow.config.mode_temperature,
        )

    @property
    def chart_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def hidden_width(self) -> int:
        return int(self.centers.shape[1])

    @property
    def tangent_rank(self) -> int:
        return int(self.bases.shape[2])

    @property
    def modes_per_chart(self) -> int:
        return int(self.router_weight.shape[1])

    @property
    def owned_parameter_count(self) -> int:
        c = self.chart_count
        h = self.hidden_width
        r = self.tangent_rank
        m = self.modes_per_chart
        return c * h + c * h * r + 1 + c * m * (r + 1) + c * m * (r * r + r)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "chart_count": self.chart_count,
            "equation_id": COMPILED_LAYER_MULTIMODE_EQUATION_ID,
            "hidden_width": self.hidden_width,
            "layer_ordinal": self.layer_ordinal,
            "mode_temperature": self.mode_temperature,
            "modes_per_chart": self.modes_per_chart,
            "owned_parameter_count": self.owned_parameter_count,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "source_flow_model_parameter_sha256": (
                self.source_flow_model_parameter_sha256
            ),
            "source_fit_receipt_sha256": self.source_fit_receipt_sha256,
            "tangent_rank": self.tangent_rank,
            "tensor_sha256": {
                name: _tensor_sha256(getattr(self, name))
                for name in _COEFFICIENT_NAMES
            },
        }

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe, tensor-free deployment metadata."""

        return {**self._hash_payload(), "coefficient_sha256": self.coefficient_sha256}

    def assert_integrity(self) -> None:
        if _digest(_COEFFICIENT_DOMAIN, self._hash_payload()) != self.coefficient_sha256:
            raise LayerMultimodeCompilationError(
                "frozen coefficient tensors changed after construction"
            )


def _register_live_selected_inventory(value: object) -> None:
    identity = id(value)

    def _discard(reference: weakref.ReferenceType[object]) -> None:
        if _LIVE_SELECTED_INVENTORIES.get(identity) is reference:
            _LIVE_SELECTED_INVENTORIES.pop(identity, None)

    _LIVE_SELECTED_INVENTORIES[identity] = weakref.ref(value, _discard)


def _is_live_selected_inventory(value: object) -> bool:
    reference = _LIVE_SELECTED_INVENTORIES.get(id(value))
    return reference is not None and reference() is value


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SelectedLayerMultimodeCoefficientInventory:
    """Typed cross-binding from the A selection to 18 frozen fit tensors."""

    a_development_partition_receipt: ADevelopmentPartitionReceipt
    selection_receipt: LayerMultimodeSelectionReceipt
    coefficients: tuple[FrozenLayerMultimodeCoefficients, ...]
    selected_candidate_artifact_sha256s: tuple[str, ...]
    selected_fit_receipt_sha256s: tuple[str, ...]
    selected_source_rows_receipt_sha256s: tuple[str, ...]
    selected_fit_source_sha256s: tuple[str, ...]
    selected_validation_source_sha256s: tuple[str, ...]
    selected_atlas_sha256s: tuple[str, ...]
    selected_model_parameter_sha256s: tuple[str, ...]
    split_refit_evidence_sha256s: tuple[str | None, ...]
    _builder_authority: InitVar[object] = None
    layer_materialization_inventory: (
        LayerMLPFlowMaterializationInventoryReceipt | None
    ) = None
    production_provenance_authorized: bool = field(init=False)
    inventory_sha256: str = ""
    _builder_validation: object = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self, _builder_authority: object) -> None:
        if _builder_authority is not _INVENTORY_BUILDER_AUTHORITY:
            raise TypeError(
                "selected inventories must be constructed by the typed builder"
            )
        object.__setattr__(
            self,
            "_builder_validation",
            _INVENTORY_BUILDER_AUTHORITY,
        )
        if type(self.a_development_partition_receipt) is not (
            ADevelopmentPartitionReceipt
        ):
            raise TypeError(
                "a_development_partition_receipt must be a typed A receipt"
            )
        self.a_development_partition_receipt.assert_integrity()
        restored_partition = ADevelopmentPartitionReceipt.from_state_dict(
            json.loads(
                json.dumps(
                    self.a_development_partition_receipt.state_dict()
                )
            ),
            expected_receipt_sha256=(
                self.a_development_partition_receipt.receipt_sha256
            ),
        )
        if restored_partition != self.a_development_partition_receipt:
            raise LayerMultimodeCompilationError(
                "A development partition receipt did not strictly replay"
            )
        if type(self.selection_receipt) is not LayerMultimodeSelectionReceipt:
            raise TypeError("selection_receipt must be a typed selection receipt")
        self.selection_receipt.validate_integrity()
        # Exercise the strict JSON parser as part of the production boundary.
        restored = LayerMultimodeSelectionReceipt.from_state_dict(
            json.loads(json.dumps(self.selection_receipt.state_dict()))
        )
        if restored.artifact_sha256 != self.selection_receipt.artifact_sha256:
            raise LayerMultimodeCompilationError(
                "selection receipt did not survive strict JSON replay"
            )
        if (
            self.a_development_partition_receipt.fit_partition_sha256
            != self.selection_receipt.fit_a_sha256
            or self.a_development_partition_receipt.validation_partition_sha256
            != self.selection_receipt.validation_a_sha256
        ):
            raise LayerMultimodeCompilationError(
                "selection A partitions differ from the authenticated shared-A receipt"
            )
        materialization = self.layer_materialization_inventory
        if materialization is None:
            object.__setattr__(self, "production_provenance_authorized", False)
        elif type(materialization) is LayerMLPFlowMaterializationInventoryReceipt:
            _require_live_full_grid_selection_authority(self.selection_receipt)
            if self.selection_receipt.production_fit_authorized is not True:
                raise LayerMultimodeCompilationError(
                    "production materialization cannot authorize a test-only fit "
                    "selection"
                )
            object.__setattr__(self, "production_provenance_authorized", True)
        else:
            raise TypeError(
                "layer_materialization_inventory must be the exact typed "
                "materialization inventory or None"
            )
        if (
            type(self.coefficients) is not tuple
            or len(self.coefficients) != GEMMA3_LAYER_COUNT
            or any(
                type(value) is not FrozenLayerMultimodeCoefficients
                for value in self.coefficients
            )
        ):
            raise LayerMultimodeCompilationError(
                "selected inventory requires 18 frozen coefficient rows"
            )
        for name in (
            "selected_candidate_artifact_sha256s",
            "selected_fit_receipt_sha256s",
            "selected_source_rows_receipt_sha256s",
            "selected_fit_source_sha256s",
            "selected_validation_source_sha256s",
            "selected_atlas_sha256s",
            "selected_model_parameter_sha256s",
            "split_refit_evidence_sha256s",
        ):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != GEMMA3_LAYER_COUNT:
                raise LayerMultimodeCompilationError(
                    f"{name} must be an ordered 18-row tuple"
                )
        selection_sha256 = self.selection_receipt.artifact_sha256
        candidates_by_sha256 = {
            value.artifact_sha256: value
            for value in self.selection_receipt.candidates
        }
        for ordinal, (layer, coefficient) in enumerate(
            zip(
                self.selection_receipt.layers,
                self.coefficients,
                strict=True,
            )
        ):
            coefficient.assert_integrity()
            candidate_sha256 = self.selected_candidate_artifact_sha256s[ordinal]
            selected = candidates_by_sha256.get(candidate_sha256)
            if selected is None:
                raise LayerMultimodeCompilationError(
                    f"layer {ordinal} selected candidate is absent"
                )
            fit_sha256 = self.selected_fit_receipt_sha256s[ordinal]
            rows_sha256 = self.selected_source_rows_receipt_sha256s[ordinal]
            fit_source_sha256 = self.selected_fit_source_sha256s[ordinal]
            validation_source_sha256 = (
                self.selected_validation_source_sha256s[ordinal]
            )
            atlas_sha256 = self.selected_atlas_sha256s[ordinal]
            model_sha256 = self.selected_model_parameter_sha256s[ordinal]
            split_sha256 = self.split_refit_evidence_sha256s[ordinal]
            for label, digest in (
                ("selected candidate", candidate_sha256),
                ("selected fit", fit_sha256),
                ("selected source rows", rows_sha256),
                ("selected fit source", fit_source_sha256),
                ("selected validation source", validation_source_sha256),
                ("selected atlas", atlas_sha256),
                ("selected model", model_sha256),
            ):
                _require_sha256(digest, label=f"layer {ordinal} {label}")
            if split_sha256 is not None:
                _require_sha256(
                    split_sha256,
                    label=f"layer {ordinal} split-refit evidence",
                )
            if (
                layer.layer_ordinal != ordinal
                or coefficient.layer_ordinal != ordinal
                or layer.selected_candidate_sha256 != candidate_sha256
                or selected.source_rows_receipt_sha256 != rows_sha256
                or selected.fit_source_sha256 != fit_source_sha256
                or selected.validation_source_sha256
                != validation_source_sha256
                or selected.atlas_sha256 != atlas_sha256
                or coefficient.chart_count != layer.selected_chart_count
                or coefficient.modes_per_chart
                != layer.selected_dynamic_mode_count
                or coefficient.tangent_rank != layer.selected_tangent_rank
                or coefficient.source_fit_receipt_sha256 != fit_sha256
                or coefficient.source_flow_model_parameter_sha256
                != model_sha256
                or coefficient.selection_receipt_sha256 != selection_sha256
            ):
                raise LayerMultimodeCompilationError(
                    f"layer {ordinal} selection/fit/coefficient binding drifted"
                )
            if (coefficient.modes_per_chart == 1) is not (split_sha256 is None):
                raise LayerMultimodeCompilationError(
                    f"layer {ordinal} split-refit evidence boundary drifted"
                )
        self._assert_materialization_binding()
        computed = _digest(_INVENTORY_DOMAIN, self._payload())
        if self.inventory_sha256 == "":
            object.__setattr__(self, "inventory_sha256", computed)
        elif _require_sha256(
            self.inventory_sha256, label="inventory_sha256"
        ) != computed:
            raise LayerMultimodeCompilationError(
                "selected coefficient inventory hash mismatch"
            )
        _register_live_selected_inventory(self)

    @property
    def source_model_sha256(self) -> str:
        return self.selection_receipt.source_model_sha256

    @property
    def source_execution_fingerprint_sha256(self) -> str | None:
        materialization = self.layer_materialization_inventory
        return (
            None
            if materialization is None
            else materialization.source_execution_fingerprint_sha256
        )

    @property
    def selection_receipt_sha256(self) -> str:
        return self.selection_receipt.artifact_sha256

    @property
    def layer_materialization_inventory_receipt_sha256(self) -> str | None:
        value = self.layer_materialization_inventory
        return None if value is None else value.receipt_sha256

    def _assert_materialization_binding(self) -> None:
        materialization = self.layer_materialization_inventory
        if materialization is None:
            if self.production_provenance_authorized is not False:
                raise LayerMultimodeCompilationError(
                    "unbound selected inventory cannot claim production provenance"
                )
            return
        if type(materialization) is not LayerMLPFlowMaterializationInventoryReceipt:
            raise TypeError(
                "layer_materialization_inventory must remain the exact typed "
                "materialization inventory"
            )
        if self.production_provenance_authorized is not True:
            raise LayerMultimodeCompilationError(
                "bound selected inventory lost production authorization"
            )
        _require_live_full_grid_selection_authority(self.selection_receipt)
        try:
            materialization.assert_integrity()
            replayed = LayerMLPFlowMaterializationInventoryReceipt.from_state_dict(
                json.loads(json.dumps(materialization.state_dict())),
                expected_receipt_sha256=materialization.receipt_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise LayerMultimodeCompilationError(
                "layer materialization inventory failed strict replay"
            ) from exc
        if replayed.state_dict() != materialization.state_dict():
            raise LayerMultimodeCompilationError(
                "layer materialization inventory did not replay exactly"
            )
        # Parsed receipts are durable audit evidence only.  Deployment authority
        # exists solely during the live capture/materialization transaction.
        if materialization.live_materialization_authorized is not True:
            raise LayerMultimodeCompilationError(
                "production compilation requires live materialization authority"
            )
        partition = self.a_development_partition_receipt
        if (
            materialization.a_development_partition_receipt_sha256
            != partition.receipt_sha256
            or materialization.source_full_A_bridge_receipt_sha256
            != partition.source_full_A_bridge_receipt_sha256
            or materialization.source_runtime_session_binding_sha256
            != partition.source_runtime_session_binding_sha256
            or materialization.fit_partition_sha256
            != partition.fit_partition_sha256
            or materialization.validation_partition_sha256
            != partition.validation_partition_sha256
        ):
            raise LayerMultimodeCompilationError(
                "materialization inventory differs from the authenticated A partition"
            )
        if (
            materialization.source_model_fingerprint_sha256
            != self.source_model_sha256
        ):
            raise LayerMultimodeCompilationError(
                "materialization source-model fingerprint differs from selection"
            )
        if (
            materialization.layer_flow_rows_receipt_sha256s
            != self.selected_source_rows_receipt_sha256s
        ):
            raise LayerMultimodeCompilationError(
                "materialization flow-row inventory differs from selected fits"
            )

    def _payload(self) -> dict[str, object]:
        materialization = self.layer_materialization_inventory
        materialization_bound = materialization is not None
        return {
            "a_development_partition_receipt_sha256": (
                self.a_development_partition_receipt.receipt_sha256
            ),
            "coefficient_sha256s": [
                value.coefficient_sha256 for value in self.coefficients
            ],
            "equation_id": COMPILED_LAYER_MULTIMODE_EQUATION_ID,
            "format_version": _FORMAT_VERSION,
            "capture_time_layer_materialization_receipt_bound": (
                materialization_bound
            ),
            "capture_time_layer_materialization_receipt_required": True,
            "layer_materialization_inventory_receipt_sha256": (
                self.layer_materialization_inventory_receipt_sha256
            ),
            "materialization_predecessor_runtime_qualification_receipt_sha256": (
                None
                if materialization is None
                else materialization.predecessor_runtime_qualification_receipt_sha256
            ),
            "materialization_source_full_A_bridge_receipt_sha256": (
                None
                if materialization is None
                else materialization.source_full_A_bridge_receipt_sha256
            ),
            "materialization_source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "materialization_source_runtime_session_binding_sha256": (
                None
                if materialization is None
                else materialization.source_runtime_session_binding_sha256
            ),
            "partition_to_flow_rows_proof_claimed": materialization_bound,
            "production_provenance_authorized": (
                self.production_provenance_authorized
            ),
            "selection_production_fit_authorized": (
                self.selection_receipt.production_fit_authorized
            ),
            "selected_candidate_artifact_sha256s": list(
                self.selected_candidate_artifact_sha256s
            ),
            "selected_fit_receipt_sha256s": list(
                self.selected_fit_receipt_sha256s
            ),
            "selected_source_rows_receipt_sha256s": list(
                self.selected_source_rows_receipt_sha256s
            ),
            "selected_fit_source_sha256s": list(
                self.selected_fit_source_sha256s
            ),
            "selected_validation_source_sha256s": list(
                self.selected_validation_source_sha256s
            ),
            "selected_atlas_sha256s": list(self.selected_atlas_sha256s),
            "selected_model_parameter_sha256s": list(
                self.selected_model_parameter_sha256s
            ),
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "source_model_sha256": self.source_model_sha256,
            "split_refit_evidence_sha256s": list(
                self.split_refit_evidence_sha256s
            ),
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "inventory_sha256": self.inventory_sha256}

    def assert_integrity(self) -> None:
        if not _is_live_selected_inventory(self):
            raise LayerMultimodeCompilationError(
                "selected inventory is not the exact live typed-builder result"
            )
        if self._builder_validation is not _INVENTORY_BUILDER_AUTHORITY:
            raise LayerMultimodeCompilationError(
                "selected inventory lost typed-builder authority"
            )
        if type(self.production_provenance_authorized) is not bool:
            raise LayerMultimodeCompilationError(
                "selected inventory production authorization must be an exact bool"
            )
        self.a_development_partition_receipt.assert_integrity()
        self.selection_receipt.validate_integrity()
        if (
            self.a_development_partition_receipt.fit_partition_sha256
            != self.selection_receipt.fit_a_sha256
            or self.a_development_partition_receipt.validation_partition_sha256
            != self.selection_receipt.validation_a_sha256
        ):
            raise LayerMultimodeCompilationError(
                "selected inventory A partition binding drifted"
            )
        for coefficient in self.coefficients:
            coefficient.assert_integrity()
        self._assert_materialization_binding()
        if _digest(_INVENTORY_DOMAIN, self._payload()) != self.inventory_sha256:
            raise LayerMultimodeCompilationError(
                "selected coefficient inventory drifted"
            )


def build_selected_layer_multimode_coefficient_inventory(
    selection_receipt: LayerMultimodeSelectionReceipt,
    selected_candidate_fits: tuple[object, ...],
    *,
    a_development_partition_receipt: ADevelopmentPartitionReceipt,
    layer_materialization_inventory: (
        LayerMLPFlowMaterializationInventoryReceipt | None
    ),
    split_refit_evidence: tuple[object | None, ...],
    allow_test_unbound_materialization: bool = False,
) -> SelectedLayerMultimodeCoefficientInventory:
    """Regenerate all selected A evaluations before freezing deploy tensors."""

    from .gemma3_layer_multimode_fit import (
        LayerMultimodeCandidateFit,
        LayerMultimodeSplitRefitEvidence,
        layer_multimode_candidate_evaluation_from_fit,
    )

    allow_test_unbound_materialization = _strict_bool(
        allow_test_unbound_materialization,
        label="allow_test_unbound_materialization",
    )
    if layer_materialization_inventory is None:
        if not allow_test_unbound_materialization:
            raise LayerMultimodeCompilationError(
                "production inventory requires a live typed layer-materialization "
                "inventory; the unbound path requires explicit test-only opt-in"
            )
    elif type(layer_materialization_inventory) is not (
        LayerMLPFlowMaterializationInventoryReceipt
    ):
        raise TypeError(
            "layer_materialization_inventory must be the exact typed "
            "materialization inventory or None"
        )
    elif allow_test_unbound_materialization:
        raise LayerMultimodeCompilationError(
            "test-only unbound opt-in is forbidden with a bound materialization "
            "inventory"
        )
    if type(selection_receipt) is not LayerMultimodeSelectionReceipt:
        raise TypeError("selection_receipt must be a typed selection receipt")
    selection_receipt.validate_integrity()
    if layer_materialization_inventory is not None:
        _require_live_full_grid_selection_authority(selection_receipt)
    if (
        layer_materialization_inventory is not None
        and selection_receipt.production_fit_authorized is not True
    ):
        raise LayerMultimodeCompilationError(
            "production materialization requires an exact production-authorized "
            "fit selection"
        )
    if type(a_development_partition_receipt) is not ADevelopmentPartitionReceipt:
        raise TypeError(
            "a_development_partition_receipt must be a typed A receipt"
        )
    a_development_partition_receipt.assert_integrity()
    replayed_partition = ADevelopmentPartitionReceipt.from_state_dict(
        json.loads(json.dumps(a_development_partition_receipt.state_dict())),
        expected_receipt_sha256=(
            a_development_partition_receipt.receipt_sha256
        ),
    )
    if replayed_partition != a_development_partition_receipt:
        raise LayerMultimodeCompilationError(
            "A development partition receipt did not strictly replay"
        )
    if (
        a_development_partition_receipt.fit_partition_sha256
        != selection_receipt.fit_a_sha256
        or a_development_partition_receipt.validation_partition_sha256
        != selection_receipt.validation_a_sha256
    ):
        raise LayerMultimodeCompilationError(
            "selection A partitions differ from the authenticated shared-A receipt"
        )
    if (
        type(selected_candidate_fits) is not tuple
        or len(selected_candidate_fits) != GEMMA3_LAYER_COUNT
        or any(
            type(value) is not LayerMultimodeCandidateFit
            for value in selected_candidate_fits
        )
    ):
        raise LayerMultimodeCompilationError(
            "selected_candidate_fits must be 18 ordered typed fits"
        )
    if (
        type(split_refit_evidence) is not tuple
        or len(split_refit_evidence) != GEMMA3_LAYER_COUNT
    ):
        raise LayerMultimodeCompilationError(
            "split_refit_evidence must be an ordered 18-row tuple"
        )
    if layer_materialization_inventory is not None:
        _require_live_selected_fit_authority(
            selected_candidate_fits,
            split_refit_evidence,
        )
    coefficients: list[FrozenLayerMultimodeCoefficients] = []
    candidate_sha256s: list[str] = []
    fit_sha256s: list[str] = []
    source_rows_sha256s: list[str] = []
    fit_source_sha256s: list[str] = []
    validation_source_sha256s: list[str] = []
    atlas_sha256s: list[str] = []
    model_sha256s: list[str] = []
    split_sha256s: list[str | None] = []
    candidates_by_sha256 = {
        value.artifact_sha256: value for value in selection_receipt.candidates
    }
    for ordinal, (layer, fit, split) in enumerate(
        zip(
            selection_receipt.layers,
            selected_candidate_fits,
            split_refit_evidence,
            strict=True,
        )
    ):
        assert isinstance(fit, LayerMultimodeCandidateFit)
        fit.assert_integrity()
        selected = candidates_by_sha256[layer.selected_candidate_sha256]
        if fit.receipt.modes_per_chart == 1:
            if split is not None:
                raise LayerMultimodeCompilationError(
                    f"layer {ordinal} M=1 must not supply split-refit evidence"
                )
            typed_split = None
            split_sha256 = None
        else:
            if type(split) is not LayerMultimodeSplitRefitEvidence:
                raise LayerMultimodeCompilationError(
                    f"layer {ordinal} M>1 requires typed split-refit evidence"
                )
            typed_split = split
            split_sha256 = split.evidence_sha256
        regenerated = layer_multimode_candidate_evaluation_from_fit(
            fit,
            source_model_sha256=selection_receipt.source_model_sha256,
            fit_a_partition_sha256=selection_receipt.fit_a_sha256,
            validation_a_partition_sha256=(
                selection_receipt.validation_a_sha256
            ),
            evaluation_protocol_sha256=(
                selection_receipt.evaluation_protocol_sha256
            ),
            split_refit=typed_split,
            allow_test_only_nonlive_authority=(
                allow_test_unbound_materialization
            ),
        )
        if regenerated.state_dict() != selected.state_dict():
            raise LayerMultimodeCompilationError(
                f"layer {ordinal} selected evaluation does not regenerate from fit"
            )
        if (
            fit.receipt.layer_ordinal != ordinal
            or fit.receipt.chart_count != layer.selected_chart_count
            or fit.receipt.modes_per_chart
            != layer.selected_dynamic_mode_count
            or fit.receipt.tangent_rank != layer.selected_tangent_rank
            or fit.receipt.final_chart_temperature
            != selected.chart_temperature
            or fit.receipt.source_rows_receipt_sha256
            != selected.source_rows_receipt_sha256
            or fit.receipt.fit_source_sha256 != selected.fit_source_sha256
            or fit.receipt.validation_source_sha256
            != selected.validation_source_sha256
            or fit.receipt.atlas_fit.atlas_sha256 != selected.atlas_sha256
            or fit.receipt.model_parameter_sha256
            != fit.model.model_parameter_sha256
        ):
            raise LayerMultimodeCompilationError(
                f"layer {ordinal} selected fit provenance/dimensions drifted"
            )
        coefficient = FrozenLayerMultimodeCoefficients.from_float64_flow(
            fit.model,
            source_fit_receipt_sha256=fit.receipt.receipt_sha256,
            selection_receipt_sha256=selection_receipt.artifact_sha256,
        )
        coefficients.append(coefficient)
        candidate_sha256s.append(selected.artifact_sha256)
        fit_sha256s.append(fit.receipt.receipt_sha256)
        source_rows_sha256s.append(
            fit.receipt.source_rows_receipt_sha256
        )
        fit_source_sha256s.append(fit.receipt.fit_source_sha256)
        validation_source_sha256s.append(
            fit.receipt.validation_source_sha256
        )
        atlas_sha256s.append(fit.receipt.atlas_fit.atlas_sha256)
        model_sha256s.append(fit.receipt.model_parameter_sha256)
        split_sha256s.append(split_sha256)
    return SelectedLayerMultimodeCoefficientInventory(
        a_development_partition_receipt=a_development_partition_receipt,
        selection_receipt=selection_receipt,
        coefficients=tuple(coefficients),
        selected_candidate_artifact_sha256s=tuple(candidate_sha256s),
        selected_fit_receipt_sha256s=tuple(fit_sha256s),
        selected_source_rows_receipt_sha256s=tuple(source_rows_sha256s),
        selected_fit_source_sha256s=tuple(fit_source_sha256s),
        selected_validation_source_sha256s=tuple(
            validation_source_sha256s
        ),
        selected_atlas_sha256s=tuple(atlas_sha256s),
        selected_model_parameter_sha256s=tuple(model_sha256s),
        split_refit_evidence_sha256s=tuple(split_sha256s),
        layer_materialization_inventory=layer_materialization_inventory,
        _builder_authority=_INVENTORY_BUILDER_AUTHORITY,
    )


def _require_live_selected_fit_authority(
    selected_candidate_fits: tuple[object, ...],
    split_refit_evidence: tuple[object | None, ...],
) -> None:
    """Require the original trusted fitter/evaluator results for production."""

    from .gemma3_layer_multimode_fit import (
        LayerMultimodeCandidateFit,
        LayerMultimodeSplitRefitEvidence,
    )

    if (
        type(selected_candidate_fits) is not tuple
        or len(selected_candidate_fits) != GEMMA3_LAYER_COUNT
        or type(split_refit_evidence) is not tuple
        or len(split_refit_evidence) != GEMMA3_LAYER_COUNT
    ):
        raise LayerMultimodeCompilationError(
            "live fitter authority requires ordered 18-row inventories"
        )
    for ordinal, (fit, split) in enumerate(
        zip(selected_candidate_fits, split_refit_evidence, strict=True)
    ):
        if type(fit) is not LayerMultimodeCandidateFit:
            raise LayerMultimodeCompilationError(
                f"layer {ordinal} fit does not have the exact trusted type"
            )
        fit.assert_integrity()
        if fit.live_fitter_authorized is not True:
            raise LayerMultimodeCompilationError(
                f"layer {ordinal} production coefficient freezing requires "
                "the original live fitter result"
            )
        if fit.receipt.modes_per_chart == 1:
            if split is not None:
                raise LayerMultimodeCompilationError(
                    f"layer {ordinal} M=1 must not supply split-refit evidence"
                )
            continue
        if type(split) is not LayerMultimodeSplitRefitEvidence:
            raise LayerMultimodeCompilationError(
                f"layer {ordinal} production M>1 freezing requires exact "
                "split-refit evidence"
            )
        split.assert_integrity()
        if split.live_split_evaluator_authorized is not True:
            raise LayerMultimodeCompilationError(
                f"layer {ordinal} production M>1 freezing requires the "
                "original live split-evaluator evidence"
            )


def _require_live_full_grid_selection_authority(
    selection_receipt: LayerMultimodeSelectionReceipt,
) -> None:
    """Require the original selector and all original 324 candidate inputs."""

    if type(selection_receipt) is not LayerMultimodeSelectionReceipt:
        raise TypeError("selection_receipt must be the exact typed selection")
    selection_receipt.validate_integrity()
    if selection_receipt.live_full_grid_selection_authorized is not True:
        raise LayerMultimodeCompilationError(
            "production coefficient freezing requires the original live-authorized "
            "full-grid selection receipt"
        )


class Float32LayerMultimodeMLP(nn.Module):
    """Float32 direct-delta implementation of the frozen multimode equation."""

    source_free: Final = True
    native_mlp_fallback_present: Final = False
    returns_post_output_rmsnorm_teacher_delta: Final = True

    def __init__(self, coefficients: FrozenLayerMultimodeCoefficients) -> None:
        super().__init__()
        if type(coefficients) is not FrozenLayerMultimodeCoefficients:
            raise TypeError("coefficients must be frozen multimode coefficients")
        coefficients.assert_integrity()
        self.layer_ordinal = coefficients.layer_ordinal
        self.source_flow_model_parameter_sha256 = (
            coefficients.source_flow_model_parameter_sha256
        )
        self.source_fit_receipt_sha256 = coefficients.source_fit_receipt_sha256
        self.selection_receipt_sha256 = coefficients.selection_receipt_sha256
        self.coefficient_sha256 = coefficients.coefficient_sha256
        self.mode_temperature = coefficients.mode_temperature
        for name in _COEFFICIENT_NAMES:
            value = getattr(coefficients, name).detach().clone().contiguous()
            setattr(self, name, nn.Parameter(value, requires_grad=False))

    @property
    def chart_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def hidden_width(self) -> int:
        return int(self.centers.shape[1])

    @property
    def tangent_rank(self) -> int:
        return int(self.bases.shape[2])

    @property
    def modes_per_chart(self) -> int:
        return int(self.router_weight.shape[1])

    @property
    def owned_parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def coefficient_metadata(self) -> dict[str, object]:
        return {
            "chart_count": self.chart_count,
            "coefficient_sha256": self.coefficient_sha256,
            "equation_id": COMPILED_LAYER_MULTIMODE_EQUATION_ID,
            "hidden_width": self.hidden_width,
            "layer_ordinal": self.layer_ordinal,
            "mode_temperature": self.mode_temperature,
            "modes_per_chart": self.modes_per_chart,
            "owned_parameter_count": self.owned_parameter_count,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "source_flow_model_parameter_sha256": (
                self.source_flow_model_parameter_sha256
            ),
            "source_fit_receipt_sha256": self.source_fit_receipt_sha256,
            "tangent_rank": self.tangent_rank,
            "tensor_sha256": {
                name: _tensor_sha256(getattr(self, name))
                for name in _COEFFICIENT_NAMES
            },
        }

    def assert_integrity(self) -> None:
        payload = dict(self.coefficient_metadata())
        payload.pop("coefficient_sha256")
        if _digest(_COEFFICIENT_DOMAIN, payload) != self.coefficient_sha256:
            raise LayerMultimodeCompilationError("deployed coefficients drifted")
        for name, parameter in self.named_parameters():
            if parameter.dtype != torch.float32 or parameter.requires_grad:
                raise LayerMultimodeCompilationError(
                    f"deployed coefficient {name} must be frozen float32"
                )

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("compiled MLP input must be a tensor")
        if x.dtype != torch.float32:
            raise TypeError("compiled MLP input must have dtype torch.float32")
        if x.ndim < 1 or int(x.shape[-1]) != self.hidden_width:
            raise LayerMultimodeCompilationError(
                "compiled MLP input must end in hidden width H"
            )
        if x.device != self.centers.device:
            raise LayerMultimodeCompilationError(
                "compiled MLP input and coefficients must share a device"
            )
        if not bool(torch.isfinite(x).all()):
            raise LayerMultimodeCompilationError("compiled MLP input must be finite")
        original_shape = x.shape
        flat = x.reshape(-1, self.hidden_width)
        centered = flat[:, None, :] - self.centers[None, :, :]
        z = torch.einsum("nch,chr->ncr", centered, self.bases)
        normal_squared = (
            centered.square().sum(dim=-1) - z.square().sum(dim=-1)
        ).clamp_min(0.0)
        chart_temperature = self.log_chart_temperature.exp()
        q = torch.softmax(-normal_squared / chart_temperature, dim=-1)
        router_logits = (
            torch.einsum("ncr,cmr->ncm", z, self.router_weight)
            + self.router_bias[None, :, :]
        )
        pi = torch.softmax(router_logits, dim=-1)
        per_mode_coordinates = (
            torch.einsum("ncr,cmsr->ncms", z, self.generator_weight)
            + self.generator_bias[None, :, :, :]
        )
        fused_coordinates = torch.einsum(
            "ncm,ncmr->ncr", pi, per_mode_coordinates
        )
        chart_velocities = torch.einsum(
            "ncr,chr->nch", fused_coordinates, self.bases
        )
        delta = torch.einsum("nc,nch->nh", q, chart_velocities)
        return delta.reshape(original_shape)


@dataclass(frozen=True, slots=True)
class LayerMultimodeLayerAccounting:
    layer_ordinal: int
    hidden_width: int
    source_intermediate_width: int
    chart_count: int
    modes_per_chart: int
    tangent_rank: int
    retained_input_rmsnorm_parameters: int
    removed_output_rmsnorm_parameters: int
    removed_native_mlp_parameters: int
    replacement_owned_parameters: int
    native_boundary_parameters: int
    compiled_boundary_parameters: int
    boundary_parameter_savings: int
    replacement_float32_bytes: int
    coefficient_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "layer_ordinal",
            "hidden_width",
            "source_intermediate_width",
            "chart_count",
            "modes_per_chart",
            "tangent_rank",
            "retained_input_rmsnorm_parameters",
            "removed_output_rmsnorm_parameters",
            "removed_native_mlp_parameters",
            "replacement_owned_parameters",
            "native_boundary_parameters",
            "compiled_boundary_parameters",
            "boundary_parameter_savings",
            "replacement_float32_bytes",
        ):
            minimum = 0 if name in {"layer_ordinal", "boundary_parameter_savings"} else 1
            _strict_int(getattr(self, name), label=name, minimum=minimum)
        if self.layer_ordinal >= GEMMA3_LAYER_COUNT:
            raise LayerMultimodeCompilationError("invalid layer ordinal")
        _require_sha256(self.coefficient_sha256, label="coefficient_sha256")
        h, i, c, m, r = (
            self.hidden_width,
            self.source_intermediate_width,
            self.chart_count,
            self.modes_per_chart,
            self.tangent_rank,
        )
        expected_owned = c * h + c * h * r + 1 + c * m * (r + 1) + c * m * (r * r + r)
        if (
            self.retained_input_rmsnorm_parameters != h
            or self.removed_output_rmsnorm_parameters != h
            or self.removed_native_mlp_parameters != 3 * h * i
            or self.replacement_owned_parameters != expected_owned
            or self.native_boundary_parameters != 2 * h + 3 * h * i
            or self.compiled_boundary_parameters != h + expected_owned
            or self.boundary_parameter_savings
            != self.native_boundary_parameters - self.compiled_boundary_parameters
            or self.replacement_float32_bytes != 4 * expected_owned
        ):
            raise LayerMultimodeCompilationError("layer accounting formula mismatch")

    def state_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_state_dict(cls, state: object) -> "LayerMultimodeLayerAccounting":
        if not isinstance(state, dict) or set(state) != set(cls.__dataclass_fields__):
            raise LayerMultimodeCompilationError("invalid layer accounting state")
        return cls(**state)


@dataclass(frozen=True, slots=True)
class Gemma3LayerMultimodeCompilationReceipt:
    source_model_sha256: str
    recomputed_source_model_sha256: str | None
    source_execution_fingerprint_sha256: str | None
    recomputed_source_execution_fingerprint_sha256: str | None
    compiled_execution_fingerprint_sha256: str
    selection_receipt_sha256: str
    selected_inventory_sha256: str
    source_state_sha256: str
    protected_architecture_sha256: str
    protected_compiled_state_sha256: str
    compiled_state_sha256: str
    layer_accounting: tuple[LayerMultimodeLayerAccounting, ...]
    source_parameter_count: int
    compiled_parameter_count: int
    source_parameter_storage_bytes: int
    compiled_parameter_storage_bytes: int
    source_persistent_storage_bytes: int
    compiled_persistent_storage_bytes: int
    source_state_tensor_payload_bytes: int
    compiled_state_tensor_payload_bytes: int
    tied_embedding_parameter_identity: bool
    production_shape_enforced: bool
    production_provenance_authorized: bool
    replacement_count: int
    removed_output_rmsnorm_count: int
    native_mlp_fallback_present: bool
    source_parameter_alias_present: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "source_model_sha256",
            "selection_receipt_sha256",
            "selected_inventory_sha256",
            "source_state_sha256",
            "protected_architecture_sha256",
            "protected_compiled_state_sha256",
            "compiled_state_sha256",
            "compiled_execution_fingerprint_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _strict_bool(
            self.production_provenance_authorized,
            label="production_provenance_authorized",
        )
        if self.production_provenance_authorized:
            if (
                _require_sha256(
                    self.recomputed_source_model_sha256,
                    label="recomputed_source_model_sha256",
                )
                != self.source_model_sha256
            ):
                raise LayerMultimodeCompilationError(
                    "authorized source-model fingerprint does not close"
                )
            if (
                _require_sha256(
                    self.recomputed_source_execution_fingerprint_sha256,
                    label="recomputed_source_execution_fingerprint_sha256",
                )
                != _require_sha256(
                    self.source_execution_fingerprint_sha256,
                    label="source_execution_fingerprint_sha256",
                )
            ):
                raise LayerMultimodeCompilationError(
                    "authorized source execution fingerprint does not close"
                )
        elif (
            self.recomputed_source_model_sha256 is not None
            or self.source_execution_fingerprint_sha256 is not None
            or self.recomputed_source_execution_fingerprint_sha256 is not None
        ):
            raise LayerMultimodeCompilationError(
                "unbound test compilations cannot claim recomputed source fingerprints"
            )
        if (
            type(self.layer_accounting) is not tuple
            or len(self.layer_accounting) != GEMMA3_LAYER_COUNT
        ):
            raise LayerMultimodeCompilationError("receipt requires 18 layer accounting rows")
        if tuple(row.layer_ordinal for row in self.layer_accounting) != tuple(
            range(GEMMA3_LAYER_COUNT)
        ):
            raise LayerMultimodeCompilationError("accounting rows must be ordered 0..17")
        for name in (
            "source_parameter_count",
            "compiled_parameter_count",
            "source_parameter_storage_bytes",
            "compiled_parameter_storage_bytes",
            "source_persistent_storage_bytes",
            "compiled_persistent_storage_bytes",
            "source_state_tensor_payload_bytes",
            "compiled_state_tensor_payload_bytes",
            "replacement_count",
            "removed_output_rmsnorm_count",
        ):
            _strict_int(getattr(self, name), label=name, minimum=1)
        if (
            self.replacement_count != GEMMA3_LAYER_COUNT
            or self.removed_output_rmsnorm_count != GEMMA3_LAYER_COUNT
        ):
            raise LayerMultimodeCompilationError("receipt must cover exactly 18 layers")
        if (
            _strict_bool(
                self.tied_embedding_parameter_identity,
                label="tied_embedding_parameter_identity",
            )
            is not True
            or _strict_bool(
                self.native_mlp_fallback_present,
                label="native_mlp_fallback_present",
            )
            is not False
            or _strict_bool(
                self.source_parameter_alias_present,
                label="source_parameter_alias_present",
            )
            is not False
        ):
            raise LayerMultimodeCompilationError("compiler safety assertions failed")
        _strict_bool(
            self.production_shape_enforced,
            label="production_shape_enforced",
        )
        if self.production_shape_enforced and any(
            row.hidden_width != GEMMA3_PRODUCTION_HIDDEN_WIDTH
            or row.source_intermediate_width
            != GEMMA3_PRODUCTION_INTERMEDIATE_WIDTH
            for row in self.layer_accounting
        ):
            raise LayerMultimodeCompilationError(
                "production receipt must use Gemma 3 H=640/I=2048"
            )
        expected_compiled = self.source_parameter_count + sum(
            row.compiled_boundary_parameters - row.native_boundary_parameters
            for row in self.layer_accounting
        )
        if self.compiled_parameter_count != expected_compiled:
            raise LayerMultimodeCompilationError("whole-model parameter count does not close")
        computed = _digest(_RECEIPT_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(self.receipt_sha256, label="receipt_sha256") != computed:
            raise LayerMultimodeCompilationError("compilation receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "compiled_parameter_count": self.compiled_parameter_count,
            "compiled_parameter_storage_bytes": self.compiled_parameter_storage_bytes,
            "compiled_persistent_storage_bytes": self.compiled_persistent_storage_bytes,
            "compiled_execution_fingerprint_sha256": (
                self.compiled_execution_fingerprint_sha256
            ),
            "compiled_state_sha256": self.compiled_state_sha256,
            "compiled_state_tensor_payload_bytes": self.compiled_state_tensor_payload_bytes,
            "equation_id": COMPILED_LAYER_MULTIMODE_EQUATION_ID,
            "format_version": _FORMAT_VERSION,
            "layer_accounting": [row.state_dict() for row in self.layer_accounting],
            "native_mlp_fallback_present": self.native_mlp_fallback_present,
            "protected_architecture_sha256": self.protected_architecture_sha256,
            "protected_compiled_state_sha256": self.protected_compiled_state_sha256,
            "production_shape_enforced": self.production_shape_enforced,
            "production_provenance_authorized": (
                self.production_provenance_authorized
            ),
            "removed_output_rmsnorm_count": self.removed_output_rmsnorm_count,
            "recomputed_source_model_sha256": (
                self.recomputed_source_model_sha256
            ),
            "recomputed_source_execution_fingerprint_sha256": (
                self.recomputed_source_execution_fingerprint_sha256
            ),
            "replacement_count": self.replacement_count,
            "schema": _COMPILATION_SCHEMA,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "selected_inventory_sha256": self.selected_inventory_sha256,
            "source_model_sha256": self.source_model_sha256,
            "source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "source_parameter_alias_present": self.source_parameter_alias_present,
            "source_parameter_count": self.source_parameter_count,
            "source_parameter_storage_bytes": self.source_parameter_storage_bytes,
            "source_persistent_storage_bytes": self.source_persistent_storage_bytes,
            "source_state_sha256": self.source_state_sha256,
            "source_state_tensor_payload_bytes": self.source_state_tensor_payload_bytes,
            "tied_embedding_parameter_identity": self.tied_embedding_parameter_identity,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_state_dict(
        cls, state: object
    ) -> "Gemma3LayerMultimodeCompilationReceipt":
        if not isinstance(state, dict):
            raise TypeError("compilation receipt state must be a dict")
        expected = {
            "compiled_parameter_count",
            "compiled_parameter_storage_bytes",
            "compiled_persistent_storage_bytes",
            "compiled_execution_fingerprint_sha256",
            "compiled_state_sha256",
            "compiled_state_tensor_payload_bytes",
            "equation_id",
            "format_version",
            "layer_accounting",
            "native_mlp_fallback_present",
            "protected_architecture_sha256",
            "protected_compiled_state_sha256",
            "production_shape_enforced",
            "production_provenance_authorized",
            "receipt_sha256",
            "recomputed_source_execution_fingerprint_sha256",
            "recomputed_source_model_sha256",
            "removed_output_rmsnorm_count",
            "replacement_count",
            "schema",
            "selection_receipt_sha256",
            "selected_inventory_sha256",
            "source_model_sha256",
            "source_execution_fingerprint_sha256",
            "source_parameter_alias_present",
            "source_parameter_count",
            "source_parameter_storage_bytes",
            "source_persistent_storage_bytes",
            "source_state_sha256",
            "source_state_tensor_payload_bytes",
            "tied_embedding_parameter_identity",
        }
        if set(state) != expected:
            raise LayerMultimodeCompilationError("compilation receipt fields differ")
        if (
            state["schema"] != _COMPILATION_SCHEMA
            or type(state["format_version"]) is not int
            or state["format_version"] != _FORMAT_VERSION
            or state["equation_id"] != COMPILED_LAYER_MULTIMODE_EQUATION_ID
            or type(state["layer_accounting"]) is not list
        ):
            raise LayerMultimodeCompilationError("compilation receipt header differs")
        rows = tuple(
            LayerMultimodeLayerAccounting.from_state_dict(row)
            for row in state["layer_accounting"]
        )
        kwargs = {
            key: value
            for key, value in state.items()
            if key not in {"schema", "format_version", "equation_id", "layer_accounting"}
        }
        return cls(layer_accounting=rows, **kwargs)


@dataclass(frozen=True, slots=True)
class CompiledGemma3LayerMultimodeModel:
    model: nn.Module
    receipt: Gemma3LayerMultimodeCompilationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("compiled model must be an nn.Module")
        if type(self.receipt) is not Gemma3LayerMultimodeCompilationReceipt:
            raise TypeError("receipt must be a compilation receipt")

    def assert_integrity(self) -> None:
        _require_entirely_eval(self.model, label="compiled model")
        layers = _require_layers(self.model)
        if any(type(layer.mlp) is not Float32LayerMultimodeMLP for layer in layers):
            raise LayerMultimodeCompilationError("compiled MLP replacement drifted")
        if any(type(layer.post_feedforward_layernorm) is not nn.Identity for layer in layers):
            raise LayerMultimodeCompilationError("output RMSNorm was reintroduced")
        for layer, row in zip(layers, self.receipt.layer_accounting, strict=True):
            layer.mlp.assert_integrity()
            if layer.mlp.coefficient_sha256 != row.coefficient_sha256:
                raise LayerMultimodeCompilationError("layer coefficient receipt drifted")
        input_weight, output_weight = _embedding_weights(self.model)
        if input_weight is not output_weight:
            raise LayerMultimodeCompilationError("compiled embeddings are no longer tied")
        state = self.model.state_dict()
        if _state_sha256(state) != self.receipt.compiled_state_sha256:
            raise LayerMultimodeCompilationError("compiled model state drifted")
        protected = _protected_state(state)
        if _state_sha256(protected) != self.receipt.protected_compiled_state_sha256:
            raise LayerMultimodeCompilationError("protected exact path drifted")
        if (
            _protected_architecture_sha256(self.model)
            != self.receipt.protected_architecture_sha256
        ):
            raise LayerMultimodeCompilationError(
                "protected model architecture drifted"
            )
        if (
            _gemma3_execution_fingerprint(
                self.model,
                label="compiled model",
            )
            != self.receipt.compiled_execution_fingerprint_sha256
        ):
            raise LayerMultimodeCompilationError(
                "compiled model execution fingerprint drifted"
            )


@dataclass(frozen=True, slots=True)
class LayerMultimodeBundleReceipt:
    bundle_sha256: str
    byte_count: int
    manifest_sha256: str
    compilation_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "bundle_sha256",
            "manifest_sha256",
            "compilation_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _strict_int(self.byte_count, label="byte_count", minimum=1)


@dataclass(frozen=True, slots=True)
class _SourceAnalysis:
    source_state_sha256: str
    source_parameter_count: int
    source_parameter_storage_bytes: int
    source_persistent_storage_bytes: int
    source_state_tensor_payload_bytes: int
    protected_architecture_sha256: str
    production_shape_enforced: bool
    layer_accounting: tuple[LayerMultimodeLayerAccounting, ...]


def _require_entirely_eval(model: nn.Module, *, label: str) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError(f"{label} must be an nn.Module")
    training_paths = tuple(
        name if name else "<root>"
        for name, module in model.named_modules()
        if module.training
    )
    if training_paths:
        preview = ", ".join(training_paths[:3])
        suffix = "" if len(training_paths) <= 3 else ", ..."
        raise LayerMultimodeCompilationError(
            f"{label} must be entirely in eval mode; training modules: "
            f"{preview}{suffix}"
        )


def _gemma3_execution_fingerprint(model: nn.Module, *, label: str) -> str:
    try:
        fingerprint = Gemma3CausalLMAdapter(model).execution_fingerprint()
    except Exception as error:
        raise LayerMultimodeCompilationError(
            f"{label} does not expose exact Gemma 3 execution semantics"
        ) from error
    return _require_sha256(
        fingerprint,
        label=f"{label} execution fingerprint",
    )


def _require_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(backbone, nn.Module) or not isinstance(layers, nn.ModuleList):
        raise LayerMultimodeCompilationError(
            "model must expose the HF text seam model.layers"
        )
    result = tuple(layers)
    if len(result) != GEMMA3_LAYER_COUNT:
        raise LayerMultimodeCompilationError("Gemma compiler requires exactly 18 layers")
    for ordinal, layer in enumerate(result):
        if not isinstance(layer, nn.Module):
            raise LayerMultimodeCompilationError(f"layer {ordinal} is not a module")
        for name in (
            "mlp",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            if not isinstance(getattr(layer, name, None), nn.Module):
                raise LayerMultimodeCompilationError(
                    f"layer {ordinal} lacks module {name}"
                )
    return result


def _embedding_weights(model: nn.Module) -> tuple[nn.Parameter, nn.Parameter]:
    input_module = None
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        input_module = getter()
    if input_module is None:
        input_module = getattr(getattr(model, "model", None), "embed_tokens", None)
    output_module = None
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        output_module = getter()
    if output_module is None:
        output_module = getattr(model, "lm_head", None)
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if not isinstance(input_weight, nn.Parameter) or not isinstance(output_weight, nn.Parameter):
        raise LayerMultimodeCompilationError(
            "model must expose input and output embedding Parameters"
        )
    return input_weight, output_weight


def _unique_parameter_count(model: nn.Module) -> int:
    return sum(int(value.numel()) for value in model.parameters())


def _storage_key(value: Tensor) -> tuple[str, int | None, int, int]:
    storage = value.untyped_storage()
    return (
        value.device.type,
        value.device.index,
        int(storage.data_ptr()),
        int(storage.nbytes()),
    )


def _unique_storage_bytes(values: Sequence[Tensor]) -> int:
    storages: dict[tuple[str, int | None, int, int], int] = {}
    for value in values:
        if value.device.type == "meta":
            raise LayerMultimodeCompilationError("meta tensors cannot be materialized")
        key = _storage_key(value)
        storages[key] = key[-1]
    return sum(storages.values())


def _parameter_storage_bytes(model: nn.Module) -> int:
    return _unique_storage_bytes(tuple(model.parameters()))


def _persistent_storage_bytes(model: nn.Module) -> int:
    state_values = tuple(model.state_dict().values())
    return _unique_storage_bytes(state_values)


def _state_tensor_payload_bytes(state: Mapping[str, Tensor]) -> int:
    return sum(int(value.numel()) * int(value.element_size()) for value in state.values())


def _protected_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if _is_replaced_module_path(name):
            continue
        result[name] = value
    return result


def _is_replaced_module_path(name: str) -> bool:
    for ordinal in range(GEMMA3_LAYER_COUNT):
        for leaf in ("mlp", "post_feedforward_layernorm"):
            prefix = f"model.layers.{ordinal}.{leaf}"
            if name == prefix or name.startswith(prefix + "."):
                return True
    return False


def _architecture_json_value(value: object) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise LayerMultimodeCompilationError(
                "model architecture config contains a nonfinite float"
            )
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _architecture_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_architecture_json_value(item) for item in value]
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!s}"


def _protected_architecture_sha256(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    config_to_dict = getattr(config, "to_dict", None)
    if callable(config_to_dict):
        raw_config = config_to_dict()
        if not isinstance(raw_config, Mapping):
            raise LayerMultimodeCompilationError("model config.to_dict() must return a mapping")
        config_payload: object = _architecture_json_value(raw_config)
    else:
        config_payload = None
    protected = _protected_state(model.state_dict())
    payload = {
        "config": config_payload,
        "modules": [
            [
                name,
                f"{type(module).__module__}.{type(module).__qualname__}",
            ]
            for name, module in model.named_modules()
            if not _is_replaced_module_path(name)
        ],
        "state_shapes": [
            {
                "dtype": str(protected[name].dtype),
                "name": name,
                "shape": list(protected[name].shape),
            }
            for name in sorted(protected)
        ],
    }
    return _digest(_ARCHITECTURE_DOMAIN, payload)


def _source_layer_shapes(layer: nn.Module, ordinal: int) -> tuple[int, int]:
    mlp = layer.mlp
    gate = getattr(getattr(mlp, "gate_proj", None), "weight", None)
    up = getattr(getattr(mlp, "up_proj", None), "weight", None)
    down = getattr(getattr(mlp, "down_proj", None), "weight", None)
    if not all(isinstance(value, nn.Parameter) for value in (gate, up, down)):
        raise LayerMultimodeCompilationError(
            f"layer {ordinal} native MLP must expose gate/up/down weights"
        )
    assert isinstance(gate, nn.Parameter)
    assert isinstance(up, nn.Parameter)
    assert isinstance(down, nn.Parameter)
    if gate.ndim != 2:
        raise LayerMultimodeCompilationError("native gate projection must be rank 2")
    intermediate, hidden = (int(v) for v in gate.shape)
    if tuple(up.shape) != (intermediate, hidden) or tuple(down.shape) != (hidden, intermediate):
        raise LayerMultimodeCompilationError(
            f"layer {ordinal} native projection shapes do not close"
        )
    native_parameters = tuple(mlp.parameters())
    if len(native_parameters) != 3 or sum(
        int(value.numel()) for value in native_parameters
    ) != 3 * hidden * intermediate:
        raise LayerMultimodeCompilationError(
            f"layer {ordinal} native MLP must contain only three bias-free matrices"
        )
    for norm_name in (
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        norm = getattr(layer, norm_name)
        parameters = tuple(norm.parameters())
        if len(parameters) != 1 or tuple(parameters[0].shape) != (hidden,):
            raise LayerMultimodeCompilationError(
                f"layer {ordinal} {norm_name} must own exactly one H-vector"
            )
    return hidden, intermediate


def _validate_float32_source(model: nn.Module) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise LayerMultimodeCompilationError(
                f"source state {name} must be float32 for this deployment ABI"
            )
        if value.device.type == "meta":
            raise LayerMultimodeCompilationError("source model may not contain meta tensors")


def _analyze_source(
    model: nn.Module,
    coefficients: tuple[FrozenLayerMultimodeCoefficients, ...],
    *,
    allow_test_nonproduction_shape: bool,
) -> _SourceAnalysis:
    layers = _require_layers(model)
    _validate_float32_source(model)
    input_weight, output_weight = _embedding_weights(model)
    if input_weight is not output_weight:
        raise LayerMultimodeCompilationError(
            "source input/output embedding Parameter must already be tied"
        )
    seen_replaced: set[int] = set()
    replaced_storage_keys: set[tuple[str, int | None, int, int]] = set()
    rows: list[LayerMultimodeLayerAccounting] = []
    for ordinal, (layer, spec) in enumerate(zip(layers, coefficients, strict=True)):
        spec.assert_integrity()
        if spec.layer_ordinal != ordinal:
            raise LayerMultimodeCompilationError("coefficient inventory must be ordered 0..17")
        hidden, intermediate = _source_layer_shapes(layer, ordinal)
        if not allow_test_nonproduction_shape and (
            hidden != GEMMA3_PRODUCTION_HIDDEN_WIDTH
            or intermediate != GEMMA3_PRODUCTION_INTERMEDIATE_WIDTH
        ):
            raise LayerMultimodeCompilationError(
                "production Gemma 3 compilation requires H=640 and I=2048; "
                "small shells require allow_test_nonproduction_shape=True"
            )
        if spec.hidden_width != hidden:
            raise LayerMultimodeCompilationError(
                f"layer {ordinal} coefficient hidden width differs from source"
            )
        replaced = tuple(layer.mlp.parameters()) + tuple(
            layer.post_feedforward_layernorm.parameters()
        )
        if any(id(value) in seen_replaced for value in replaced):
            raise LayerMultimodeCompilationError(
                "native MLP/output-norm Parameters may not be shared across layers"
            )
        seen_replaced.update(id(value) for value in replaced)
        for value in replaced:
            key = _storage_key(value)
            if key in replaced_storage_keys:
                raise LayerMultimodeCompilationError(
                    "removed native Parameters may not share tensor storage"
                )
            replaced_storage_keys.add(key)
        owned = spec.owned_parameter_count
        native_boundary = 2 * hidden + 3 * hidden * intermediate
        compiled_boundary = hidden + owned
        rows.append(
            LayerMultimodeLayerAccounting(
                layer_ordinal=ordinal,
                hidden_width=hidden,
                source_intermediate_width=intermediate,
                chart_count=spec.chart_count,
                modes_per_chart=spec.modes_per_chart,
                tangent_rank=spec.tangent_rank,
                retained_input_rmsnorm_parameters=hidden,
                removed_output_rmsnorm_parameters=hidden,
                removed_native_mlp_parameters=3 * hidden * intermediate,
                replacement_owned_parameters=owned,
                native_boundary_parameters=native_boundary,
                compiled_boundary_parameters=compiled_boundary,
                boundary_parameter_savings=native_boundary - compiled_boundary,
                replacement_float32_bytes=4 * owned,
                coefficient_sha256=spec.coefficient_sha256,
            )
        )
    # Replaced tensors must not be tied to a supposedly retained path.
    all_occurrences: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        all_occurrences.setdefault(id(parameter), []).append(name)
    for parameter_id in seen_replaced:
        if len(all_occurrences.get(parameter_id, ())) != 1:
            raise LayerMultimodeCompilationError(
                "a removed native Parameter aliases another model path"
            )
    storage_occurrences: dict[
        tuple[str, int | None, int, int], list[str]
    ] = {}
    for name, value in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        storage_occurrences.setdefault(_storage_key(value), []).append(name)
    for key in replaced_storage_keys:
        if len(storage_occurrences.get(key, ())) != 1:
            raise LayerMultimodeCompilationError(
                "a removed native tensor storage aliases another model path"
            )
    state = model.state_dict()
    return _SourceAnalysis(
        source_state_sha256=_state_sha256(state),
        source_parameter_count=_unique_parameter_count(model),
        source_parameter_storage_bytes=_parameter_storage_bytes(model),
        source_persistent_storage_bytes=_persistent_storage_bytes(model),
        source_state_tensor_payload_bytes=_state_tensor_payload_bytes(state),
        protected_architecture_sha256=_protected_architecture_sha256(model),
        production_shape_enforced=not allow_test_nonproduction_shape,
        layer_accounting=tuple(rows),
    )


def _materialize_structure(
    source_model: nn.Module,
    coefficients: tuple[FrozenLayerMultimodeCoefficients, ...],
    *,
    allow_test_nonproduction_shape: bool,
) -> tuple[nn.Module, _SourceAnalysis]:
    _require_entirely_eval(source_model, label="source model")
    analysis = _analyze_source(
        source_model,
        coefficients,
        allow_test_nonproduction_shape=allow_test_nonproduction_shape,
    )
    source_state_before = analysis.source_state_sha256
    source_storage_keys = {
        _storage_key(value)
        for value in tuple(source_model.parameters()) + tuple(source_model.buffers())
    }
    try:
        compiled = copy.deepcopy(source_model)
    except BaseException as error:
        raise LayerMultimodeCompilationError("source model cannot be deep-copied") from error
    layers = _require_layers(compiled)
    for layer, spec in zip(layers, coefficients, strict=True):
        replacement = Float32LayerMultimodeMLP(spec)
        target_device = next(layer.pre_feedforward_layernorm.parameters()).device
        replacement.to(device=target_device)
        replacement.train(layer.training)
        identity = nn.Identity()
        identity.train(layer.training)
        layer.mlp = replacement
        layer.post_feedforward_layernorm = identity
    input_weight, output_weight = _embedding_weights(compiled)
    if input_weight is not output_weight:
        raise LayerMultimodeCompilationError(
            "deep-copy operation did not preserve tied embeddings"
        )
    compiled_storage_keys = {
        _storage_key(value)
        for value in tuple(compiled.parameters()) + tuple(compiled.buffers())
    }
    if source_storage_keys.intersection(compiled_storage_keys):
        raise LayerMultimodeCompilationError(
            "compiled model retains source tensor storage aliases"
        )
    if _state_sha256(source_model.state_dict()) != source_state_before:
        raise LayerMultimodeCompilationError("compilation mutated the source model")
    source_protected = _protected_state(source_model.state_dict())
    compiled_protected = _protected_state(compiled.state_dict())
    if set(source_protected) != set(compiled_protected):
        raise LayerMultimodeCompilationError("protected exact-path inventory changed")
    for name in source_protected:
        if not torch.equal(
            source_protected[name].detach().cpu(),
            compiled_protected[name].detach().cpu(),
        ):
            raise LayerMultimodeCompilationError(
                f"protected exact-path tensor {name} changed"
            )
    if (
        _protected_architecture_sha256(compiled)
        != analysis.protected_architecture_sha256
    ):
        raise LayerMultimodeCompilationError(
            "protected exact-path module architecture changed"
        )
    return compiled, analysis


def _build_receipt(
    *,
    compiled: nn.Module,
    source_analysis: _SourceAnalysis,
    source_model_sha256: str,
    selection_receipt_sha256: str,
    selected_inventory_sha256: str,
    production_provenance_authorized: bool,
    recomputed_source_model_sha256: str | None,
    source_execution_fingerprint_sha256: str | None,
    recomputed_source_execution_fingerprint_sha256: str | None,
    source_state_sha256: str | None = None,
) -> Gemma3LayerMultimodeCompilationReceipt:
    _require_entirely_eval(compiled, label="compiled model")
    layers = _require_layers(compiled)
    if any(type(layer.mlp) is not Float32LayerMultimodeMLP for layer in layers):
        raise LayerMultimodeCompilationError("not every MLP was replaced")
    if any(type(layer.post_feedforward_layernorm) is not nn.Identity for layer in layers):
        raise LayerMultimodeCompilationError("not every output RMSNorm was removed")
    input_weight, output_weight = _embedding_weights(compiled)
    state = compiled.state_dict()
    receipt = Gemma3LayerMultimodeCompilationReceipt(
        source_model_sha256=_require_sha256(
            source_model_sha256, label="source_model_sha256"
        ),
        recomputed_source_model_sha256=(
            recomputed_source_model_sha256
        ),
        source_execution_fingerprint_sha256=(
            source_execution_fingerprint_sha256
        ),
        recomputed_source_execution_fingerprint_sha256=(
            recomputed_source_execution_fingerprint_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            _gemma3_execution_fingerprint(
                compiled,
                label="compiled model",
            )
        ),
        selection_receipt_sha256=_require_sha256(
            selection_receipt_sha256, label="selection_receipt_sha256"
        ),
        selected_inventory_sha256=_require_sha256(
            selected_inventory_sha256,
            label="selected_inventory_sha256",
        ),
        source_state_sha256=(
            source_analysis.source_state_sha256
            if source_state_sha256 is None
            else _require_sha256(source_state_sha256, label="source_state_sha256")
        ),
        protected_architecture_sha256=_protected_architecture_sha256(compiled),
        protected_compiled_state_sha256=_state_sha256(_protected_state(state)),
        compiled_state_sha256=_state_sha256(state),
        layer_accounting=source_analysis.layer_accounting,
        source_parameter_count=source_analysis.source_parameter_count,
        compiled_parameter_count=_unique_parameter_count(compiled),
        source_parameter_storage_bytes=source_analysis.source_parameter_storage_bytes,
        compiled_parameter_storage_bytes=_parameter_storage_bytes(compiled),
        source_persistent_storage_bytes=source_analysis.source_persistent_storage_bytes,
        compiled_persistent_storage_bytes=_persistent_storage_bytes(compiled),
        source_state_tensor_payload_bytes=source_analysis.source_state_tensor_payload_bytes,
        compiled_state_tensor_payload_bytes=_state_tensor_payload_bytes(state),
        tied_embedding_parameter_identity=input_weight is output_weight,
        production_shape_enforced=source_analysis.production_shape_enforced,
        production_provenance_authorized=_strict_bool(
            production_provenance_authorized,
            label="production_provenance_authorized",
        ),
        replacement_count=GEMMA3_LAYER_COUNT,
        removed_output_rmsnorm_count=GEMMA3_LAYER_COUNT,
        native_mlp_fallback_present=False,
        source_parameter_alias_present=False,
    )
    return receipt


def _coefficient_tuple(
    coefficients: Sequence[FrozenLayerMultimodeCoefficients],
    *,
    selection_receipt_sha256: str,
) -> tuple[FrozenLayerMultimodeCoefficients, ...]:
    if isinstance(coefficients, (str, bytes)) or not isinstance(coefficients, Sequence):
        raise TypeError("layer_coefficients must be a sequence")
    result = tuple(coefficients)
    if len(result) != GEMMA3_LAYER_COUNT or any(
        type(value) is not FrozenLayerMultimodeCoefficients for value in result
    ):
        raise LayerMultimodeCompilationError(
            "layer_coefficients must contain exactly 18 frozen rows"
        )
    if tuple(value.layer_ordinal for value in result) != tuple(range(GEMMA3_LAYER_COUNT)):
        raise LayerMultimodeCompilationError("coefficient rows must be ordered 0..17")
    if any(value.selection_receipt_sha256 != selection_receipt_sha256 for value in result):
        raise LayerMultimodeCompilationError(
            "every coefficient row must bind the selected inventory receipt"
        )
    return result


def _compile_from_frozen_coefficients(
    source_model: nn.Module,
    layer_coefficients: Sequence[FrozenLayerMultimodeCoefficients],
    *,
    source_model_sha256: str,
    selection_receipt_sha256: str,
    selected_inventory_sha256: str,
    production_provenance_authorized: bool,
    recomputed_source_model_sha256: str | None,
    source_execution_fingerprint_sha256: str | None,
    recomputed_source_execution_fingerprint_sha256: str | None,
    allow_test_nonproduction_shape: bool,
) -> CompiledGemma3LayerMultimodeModel:
    if not isinstance(source_model, nn.Module):
        raise TypeError("source_model must be an nn.Module")
    _strict_bool(
        allow_test_nonproduction_shape,
        label="allow_test_nonproduction_shape",
    )
    source_model_sha256 = _require_sha256(
        source_model_sha256, label="source_model_sha256"
    )
    selection_receipt_sha256 = _require_sha256(
        selection_receipt_sha256, label="selection_receipt_sha256"
    )
    selected_inventory_sha256 = _require_sha256(
        selected_inventory_sha256,
        label="selected_inventory_sha256",
    )
    _strict_bool(
        production_provenance_authorized,
        label="production_provenance_authorized",
    )
    if production_provenance_authorized:
        if (
            _require_sha256(
                recomputed_source_model_sha256,
                label="recomputed_source_model_sha256",
            )
            != source_model_sha256
        ):
            raise LayerMultimodeCompilationError(
                "actual source-model fingerprint differs from selection"
            )
        if (
            _require_sha256(
                recomputed_source_execution_fingerprint_sha256,
                label="recomputed_source_execution_fingerprint_sha256",
            )
            != _require_sha256(
                source_execution_fingerprint_sha256,
                label="source_execution_fingerprint_sha256",
            )
        ):
            raise LayerMultimodeCompilationError(
                "actual source execution fingerprint differs from materialization"
            )
    elif (
        recomputed_source_model_sha256 is not None
        or source_execution_fingerprint_sha256 is not None
        or recomputed_source_execution_fingerprint_sha256 is not None
    ):
        raise LayerMultimodeCompilationError(
            "unbound test compilation cannot supply source fingerprints"
        )
    coefficients = _coefficient_tuple(
        layer_coefficients,
        selection_receipt_sha256=selection_receipt_sha256,
    )
    compiled, analysis = _materialize_structure(
        source_model,
        coefficients,
        allow_test_nonproduction_shape=allow_test_nonproduction_shape,
    )
    receipt = _build_receipt(
        compiled=compiled,
        source_analysis=analysis,
        source_model_sha256=source_model_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        selected_inventory_sha256=selected_inventory_sha256,
        production_provenance_authorized=(
            production_provenance_authorized
        ),
        recomputed_source_model_sha256=(
            recomputed_source_model_sha256
        ),
        source_execution_fingerprint_sha256=(
            source_execution_fingerprint_sha256
        ),
        recomputed_source_execution_fingerprint_sha256=(
            recomputed_source_execution_fingerprint_sha256
        ),
    )
    result = CompiledGemma3LayerMultimodeModel(model=compiled, receipt=receipt)
    result.assert_integrity()
    return result


def compile_gemma3_layer_multimode_model(
    source_model: nn.Module,
    selected_inventory: SelectedLayerMultimodeCoefficientInventory,
    *,
    allow_test_unbound_materialization: bool = False,
) -> CompiledGemma3LayerMultimodeModel:
    """Compile H=640/I=2048 only after provenance is production-authorized."""

    if type(selected_inventory) is not SelectedLayerMultimodeCoefficientInventory:
        raise TypeError(
            "production compilation requires a typed selected coefficient inventory"
        )
    _strict_bool(
        allow_test_unbound_materialization,
        label="allow_test_unbound_materialization",
    )
    selected_inventory.assert_integrity()
    if (
        not selected_inventory.production_provenance_authorized
        and not allow_test_unbound_materialization
    ):
        raise LayerMultimodeCompilationError(
            "production compilation is unauthorized until the typed 18-layer "
            "capture-time materialization inventory is bound; tests must set "
            "allow_test_unbound_materialization=True explicitly"
        )
    if (
        selected_inventory.production_provenance_authorized
        and allow_test_unbound_materialization
    ):
        raise LayerMultimodeCompilationError(
            "the unbound-materialization test override is forbidden for an "
            "authorized production inventory"
        )
    recomputed_source_model_sha256: str | None = None
    source_execution_fingerprint_sha256: str | None = None
    recomputed_source_execution_fingerprint_sha256: str | None = None
    if selected_inventory.production_provenance_authorized:
        source_execution_fingerprint_sha256 = (
            selected_inventory.source_execution_fingerprint_sha256
        )
        (
            recomputed_source_model_sha256,
            recomputed_source_execution_fingerprint_sha256,
        ) = _require_source_fingerprints(
            source_model,
            expected_model_sha256=selected_inventory.source_model_sha256,
            expected_execution_sha256=(
                source_execution_fingerprint_sha256
            ),
        )
    return _compile_from_frozen_coefficients(
        source_model,
        selected_inventory.coefficients,
        source_model_sha256=selected_inventory.source_model_sha256,
        selection_receipt_sha256=(
            selected_inventory.selection_receipt_sha256
        ),
        selected_inventory_sha256=selected_inventory.inventory_sha256,
        production_provenance_authorized=(
            selected_inventory.production_provenance_authorized
        ),
        recomputed_source_model_sha256=recomputed_source_model_sha256,
        source_execution_fingerprint_sha256=(
            source_execution_fingerprint_sha256
        ),
        recomputed_source_execution_fingerprint_sha256=(
            recomputed_source_execution_fingerprint_sha256
        ),
        allow_test_nonproduction_shape=False,
    )


def _require_source_fingerprints(
    source_model: nn.Module,
    *,
    expected_model_sha256: str,
    expected_execution_sha256: str,
) -> tuple[str, str]:
    """Recompute exact tensor/semantic and live execution source identities."""

    _require_entirely_eval(source_model, label="authorized source model")
    expected_model_sha256 = _require_sha256(
        expected_model_sha256,
        label="expected source-model fingerprint",
    )
    expected_execution_sha256 = _require_sha256(
        expected_execution_sha256,
        label="expected source execution fingerprint",
    )
    try:
        adapter = Gemma3CausalLMAdapter(source_model)
        recomputed_model = adapter.model_fingerprint()
        recomputed_execution = adapter.execution_fingerprint()
    except Exception as error:
        raise LayerMultimodeCompilationError(
            "authorized compilation requires the exact Gemma 3 causal LM source"
        ) from error
    if recomputed_model != expected_model_sha256:
        raise LayerMultimodeCompilationError(
            "actual source-model fingerprint differs from the selected inventory"
        )
    if recomputed_execution != expected_execution_sha256:
        raise LayerMultimodeCompilationError(
            "actual source execution fingerprint differs from materialization"
        )
    return recomputed_model, recomputed_execution


def _require_source_model_fingerprint(
    source_model: nn.Module,
    *,
    expected_sha256: str,
) -> str:
    """Recompute the exact v2 adapter fingerprint without mutating the source."""

    expected_sha256 = _require_sha256(
        expected_sha256,
        label="expected source-model fingerprint",
    )
    _require_entirely_eval(source_model, label="authorized source model")
    try:
        recomputed = Gemma3CausalLMAdapter(source_model).model_fingerprint()
    except Exception as error:
        raise LayerMultimodeCompilationError(
            "authorized compilation requires the exact Gemma 3 causal LM source"
        ) from error
    if recomputed != expected_sha256:
        raise LayerMultimodeCompilationError(
            "actual source-model fingerprint differs from the selected inventory"
        )
    return recomputed


def _compile_gemma3_layer_multimode_model_for_test(
    source_model: nn.Module,
    layer_coefficients: Sequence[FrozenLayerMultimodeCoefficients],
    *,
    source_model_sha256: str,
    selection_receipt_sha256: str,
) -> CompiledGemma3LayerMultimodeModel:
    """Loudly private small-shape seam used only by model-free tests."""

    coefficient_tuple = tuple(layer_coefficients)
    test_inventory_sha256 = _digest(
        _INVENTORY_DOMAIN,
        {
            "coefficient_sha256s": [
                value.coefficient_sha256 for value in coefficient_tuple
            ],
            "selection_receipt_sha256": selection_receipt_sha256,
            "source_model_sha256": source_model_sha256,
            "test_only_nonproduction_inventory": True,
        },
    )
    return _compile_from_frozen_coefficients(
        source_model,
        coefficient_tuple,
        source_model_sha256=source_model_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        selected_inventory_sha256=test_inventory_sha256,
        production_provenance_authorized=False,
        recomputed_source_model_sha256=None,
        source_execution_fingerprint_sha256=None,
        recomputed_source_execution_fingerprint_sha256=None,
        allow_test_nonproduction_shape=True,
    )


def _tensor_descriptor(name: str, value: Tensor) -> dict[str, object]:
    return {
        "dtype": str(value.dtype),
        "name": name,
        "shape": list(value.shape),
        "tensor_sha256": _tensor_sha256(value),
    }


def _bundle_payload(
    compiled: CompiledGemma3LayerMultimodeModel,
) -> tuple[dict[str, object], str]:
    compiled.assert_integrity()
    state = {
        name: value.detach().cpu().clone().contiguous()
        for name, value in compiled.model.state_dict().items()
    }
    layers = _require_layers(compiled.model)
    manifest_without_hash: dict[str, object] = {
        "compilation_receipt": compiled.receipt.state_dict(),
        "equation_id": COMPILED_LAYER_MULTIMODE_EQUATION_ID,
        "format_version": _FORMAT_VERSION,
        "layer_coefficients": [layer.mlp.coefficient_metadata() for layer in layers],
        "schema": _BUNDLE_SCHEMA,
        "state_tensors": [
            _tensor_descriptor(name, state[name]) for name in sorted(state)
        ],
    }
    manifest_sha256 = _digest(_MANIFEST_DOMAIN, manifest_without_hash)
    manifest = {**manifest_without_hash, "manifest_sha256": manifest_sha256}
    return {
        "manifest_json": _canonical_json(manifest),
        "state_dict": state,
    }, manifest_sha256


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _validate_bundle_stat(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise LayerMultimodeCompilationError("bundle must be a regular file")
    if value.st_nlink != 1:
        raise LayerMultimodeCompilationError(
            "bundle must have exactly one filesystem link"
        )
    if stat.S_IMODE(value.st_mode) != 0o600:
        raise LayerMultimodeCompilationError("bundle mode must be exactly 0600")
    if value.st_size <= 0 or value.st_size > MAX_LAYER_MULTIMODE_BUNDLE_BYTES:
        raise LayerMultimodeCompilationError(
            "bundle size is outside the frozen deployment cap"
        )


def _strict_read_bundle_bytes(path: Path) -> tuple[bytes, str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LayerMultimodeCompilationError(
            "bundle could not be opened as a non-symlink file"
        ) from error
    try:
        before = os.fstat(descriptor)
        _validate_bundle_stat(before)
        payload = bytearray(before.st_size)
        payload_view = memoryview(payload)
        byte_count = 0
        digest = hashlib.sha256()
        while byte_count < before.st_size:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, before.st_size - byte_count),
            )
            if not chunk:
                break
            payload_view[byte_count : byte_count + len(chunk)] = chunk
            digest.update(chunk)
            byte_count += len(chunk)
        if byte_count == before.st_size and os.read(descriptor, 1):
            raise LayerMultimodeCompilationError(
                "bundle grew during its descriptor-pinned read"
            )
        after = os.fstat(descriptor)
        _validate_bundle_stat(after)
        if (
            _stable_stat_identity(before) != _stable_stat_identity(after)
            or byte_count != before.st_size
        ):
            raise LayerMultimodeCompilationError(
                "bundle changed during its descriptor-pinned read"
            )
    finally:
        os.close(descriptor)
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise LayerMultimodeCompilationError(
            "bundle pathname changed during read"
        ) from error
    _validate_bundle_stat(named)
    if _stable_stat_identity(named) != _stable_stat_identity(after):
        raise LayerMultimodeCompilationError(
            "bundle pathname no longer names the descriptor-pinned inode"
        )
    return bytes(payload), digest.hexdigest(), after


def _unlink_if_same_inode(
    path: Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if int(current.st_dev) == expected_device and int(current.st_ino) == expected_inode:
        path.unlink()


def save_compiled_layer_multimode_bundle(
    compiled: CompiledGemma3LayerMultimodeModel,
    path: Path | str,
    *,
    allow_test_unbound_materialization: bool = False,
) -> LayerMultimodeBundleReceipt:
    """Atomically save a strict, source-free, weights-only deployment bundle."""

    if type(compiled) is not CompiledGemma3LayerMultimodeModel:
        raise TypeError("compiled must be a compiled multimode model")
    _strict_bool(
        allow_test_unbound_materialization,
        label="allow_test_unbound_materialization",
    )
    if (
        not compiled.receipt.production_provenance_authorized
        and not allow_test_unbound_materialization
    ):
        raise LayerMultimodeCompilationError(
            "saving an unbound materialization requires the explicit test-only opt-in"
        )
    if (
        compiled.receipt.production_provenance_authorized
        and allow_test_unbound_materialization
    ):
        raise LayerMultimodeCompilationError(
            "the unbound-materialization test override is forbidden for an "
            "authorized production bundle"
        )
    destination = Path(path)
    if destination.suffix != ".pt":
        raise LayerMultimodeCompilationError("deployment bundle must use .pt")
    if os.path.lexists(destination):
        raise FileExistsError("refusing to overwrite deployment bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload, manifest_sha256 = _bundle_payload(compiled)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    published_device: int | None = None
    published_inode: int | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_bytes, temporary_sha256, temporary_stat = (
            _strict_read_bundle_bytes(temporary)
        )
        temporary_manifest, temporary_receipt, _, _ = _decode_bundle_bytes(
            temporary_bytes
        )
        if (
            temporary_manifest["manifest_sha256"] != manifest_sha256
            or temporary_receipt.state_dict() != compiled.receipt.state_dict()
        ):
            raise LayerMultimodeCompilationError(
                "serialized temporary bundle did not strict-reload"
            )
        try:
            os.link(temporary, destination)
            published_device = int(temporary_stat.st_dev)
            published_inode = int(temporary_stat.st_ino)
        except FileExistsError as error:
            raise FileExistsError("refusing to overwrite deployment bundle") from error
        temporary.unlink()
        _fsync_directory(destination.parent)
        published_bytes, bundle_sha256, published_stat = (
            _strict_read_bundle_bytes(destination)
        )
        published_manifest, published_receipt, _, _ = _decode_bundle_bytes(
            published_bytes
        )
        if (
            bundle_sha256 != temporary_sha256
            or published_stat.st_size != temporary_stat.st_size
            or published_manifest != temporary_manifest
            or published_receipt.state_dict() != compiled.receipt.state_dict()
        ):
            raise LayerMultimodeCompilationError(
                "published bundle did not reproduce its strict temporary reload"
            )
        receipt = LayerMultimodeBundleReceipt(
            bundle_sha256=bundle_sha256,
            byte_count=published_stat.st_size,
            manifest_sha256=manifest_sha256,
            compilation_receipt_sha256=compiled.receipt.receipt_sha256,
        )
        return receipt
    except BaseException:
        if published_device is not None and published_inode is not None:
            try:
                _unlink_if_same_inode(
                    destination,
                    expected_device=published_device,
                    expected_inode=published_inode,
                )
            except BaseException:
                pass
        try:
            temporary.unlink(missing_ok=True)
        except BaseException:
            pass
        try:
            _fsync_directory(destination.parent)
        except BaseException:
            pass
        raise


def _parse_manifest(raw_json: object) -> dict[str, object]:
    if not isinstance(raw_json, str):
        raise LayerMultimodeCompilationError("bundle manifest_json must be a string")
    try:
        manifest = json.loads(raw_json)
    except (TypeError, ValueError) as error:
        raise LayerMultimodeCompilationError("bundle manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or _canonical_json(manifest) != raw_json:
        raise LayerMultimodeCompilationError("bundle manifest must be canonical JSON")
    expected = {
        "compilation_receipt",
        "equation_id",
        "format_version",
        "layer_coefficients",
        "manifest_sha256",
        "schema",
        "state_tensors",
    }
    if set(manifest) != expected:
        raise LayerMultimodeCompilationError("bundle manifest fields differ")
    if (
        manifest["schema"] != _BUNDLE_SCHEMA
        or type(manifest["format_version"]) is not int
        or manifest["format_version"] != _FORMAT_VERSION
        or manifest["equation_id"] != COMPILED_LAYER_MULTIMODE_EQUATION_ID
        or type(manifest["layer_coefficients"]) is not list
        or type(manifest["state_tensors"]) is not list
    ):
        raise LayerMultimodeCompilationError("bundle manifest header differs")
    supplied = _require_sha256(manifest["manifest_sha256"], label="manifest_sha256")
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if _digest(_MANIFEST_DOMAIN, without_hash) != supplied:
        raise LayerMultimodeCompilationError("bundle manifest hash mismatch")
    return manifest


def _validate_state_dict(
    state: object,
    descriptors: object,
) -> dict[str, Tensor]:
    if not isinstance(state, dict) or type(descriptors) is not list:
        raise LayerMultimodeCompilationError("bundle state inventory must be exact")
    parsed: dict[str, Tensor] = {}
    descriptor_names: list[str] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "dtype",
            "name",
            "shape",
            "tensor_sha256",
        }:
            raise LayerMultimodeCompilationError("invalid state tensor descriptor")
        name = descriptor["name"]
        shape = descriptor["shape"]
        if (
            not isinstance(name, str)
            or not name
            or type(shape) is not list
            or any(type(width) is not int or width < 0 for width in shape)
            or not isinstance(descriptor["dtype"], str)
        ):
            raise LayerMultimodeCompilationError("invalid state tensor metadata types")
        descriptor_names.append(name)
        value = state.get(name)
        if not isinstance(value, Tensor) or value.device.type != "cpu":
            raise LayerMultimodeCompilationError("bundle state tensors must be on CPU")
        if (
            list(value.shape) != shape
            or str(value.dtype) != descriptor["dtype"]
            or _tensor_sha256(value)
            != _require_sha256(descriptor["tensor_sha256"], label="tensor_sha256")
        ):
            raise LayerMultimodeCompilationError(f"state tensor {name} failed integrity")
        parsed[name] = value.detach().clone().contiguous()
    if descriptor_names != sorted(descriptor_names) or len(
        set(descriptor_names)
    ) != len(descriptor_names):
        raise LayerMultimodeCompilationError("state descriptors must be sorted and unique")
    if set(state) != set(descriptor_names):
        raise LayerMultimodeCompilationError("state tensor inventory differs")
    return parsed


def _specs_from_bundle(
    state: Mapping[str, Tensor],
    metadata_rows: object,
    selection_receipt_sha256: str,
) -> tuple[FrozenLayerMultimodeCoefficients, ...]:
    if type(metadata_rows) is not list or len(metadata_rows) != GEMMA3_LAYER_COUNT:
        raise LayerMultimodeCompilationError("bundle requires 18 coefficient metadata rows")
    specs: list[FrozenLayerMultimodeCoefficients] = []
    for ordinal, metadata in enumerate(metadata_rows):
        if not isinstance(metadata, dict):
            raise LayerMultimodeCompilationError("coefficient metadata must be dicts")
        expected = {
            "chart_count",
            "coefficient_sha256",
            "equation_id",
            "hidden_width",
            "layer_ordinal",
            "mode_temperature",
            "modes_per_chart",
            "owned_parameter_count",
            "selection_receipt_sha256",
            "source_flow_model_parameter_sha256",
            "source_fit_receipt_sha256",
            "tangent_rank",
            "tensor_sha256",
        }
        if set(metadata) != expected:
            raise LayerMultimodeCompilationError("coefficient metadata fields differ")
        if (
            type(metadata["layer_ordinal"]) is not int
            or metadata["layer_ordinal"] != ordinal
            or metadata["equation_id"] != COMPILED_LAYER_MULTIMODE_EQUATION_ID
            or metadata["selection_receipt_sha256"] != selection_receipt_sha256
            or not isinstance(metadata["tensor_sha256"], dict)
            or set(metadata["tensor_sha256"]) != set(_COEFFICIENT_NAMES)
        ):
            raise LayerMultimodeCompilationError("coefficient metadata header differs")
        for name in (
            "chart_count",
            "hidden_width",
            "modes_per_chart",
            "owned_parameter_count",
            "tangent_rank",
        ):
            _strict_int(metadata[name], label=f"coefficient metadata {name}", minimum=1)
        if type(metadata["mode_temperature"]) is not float:
            raise LayerMultimodeCompilationError(
                "coefficient metadata mode_temperature must be a JSON float"
            )
        _finite_float(
            metadata["mode_temperature"],
            label="coefficient metadata mode_temperature",
            positive=True,
        )
        _require_sha256(
            metadata["coefficient_sha256"], label="coefficient_sha256"
        )
        _require_sha256(
            metadata["source_flow_model_parameter_sha256"],
            label="source_flow_model_parameter_sha256",
        )
        _require_sha256(
            metadata["source_fit_receipt_sha256"],
            label="source_fit_receipt_sha256",
        )
        prefix = f"model.layers.{ordinal}.mlp."
        try:
            tensors = {name: state[prefix + name] for name in _COEFFICIENT_NAMES}
        except KeyError as error:
            raise LayerMultimodeCompilationError(
                "bundle is missing replacement coefficients"
            ) from error
        for name, value in tensors.items():
            if _tensor_sha256(value) != metadata["tensor_sha256"][name]:
                raise LayerMultimodeCompilationError("coefficient tensor hash differs")
        spec = FrozenLayerMultimodeCoefficients(
            layer_ordinal=ordinal,
            source_flow_model_parameter_sha256=metadata[
                "source_flow_model_parameter_sha256"
            ],
            source_fit_receipt_sha256=metadata["source_fit_receipt_sha256"],
            selection_receipt_sha256=metadata["selection_receipt_sha256"],
            mode_temperature=metadata["mode_temperature"],
            coefficient_sha256=metadata["coefficient_sha256"],
            **tensors,
        )
        if (
            spec.chart_count != metadata["chart_count"]
            or spec.hidden_width != metadata["hidden_width"]
            or spec.modes_per_chart != metadata["modes_per_chart"]
            or spec.tangent_rank != metadata["tangent_rank"]
            or spec.owned_parameter_count != metadata["owned_parameter_count"]
        ):
            raise LayerMultimodeCompilationError("coefficient dimensions differ")
        specs.append(spec)
    return tuple(specs)


def _decode_bundle_bytes(
    bundle_bytes: bytes,
) -> tuple[
    dict[str, object],
    Gemma3LayerMultimodeCompilationReceipt,
    dict[str, Tensor],
    tuple[FrozenLayerMultimodeCoefficients, ...],
]:
    if type(bundle_bytes) is not bytes or not bundle_bytes:
        raise LayerMultimodeCompilationError("bundle bytes must be nonempty")
    try:
        raw = torch.load(
            io.BytesIO(bundle_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except BaseException as error:
        raise LayerMultimodeCompilationError(
            "bundle is not a valid weights-only torch archive"
        ) from error
    if not isinstance(raw, dict) or set(raw) != {"manifest_json", "state_dict"}:
        raise LayerMultimodeCompilationError("bundle top-level fields differ")
    manifest = _parse_manifest(raw["manifest_json"])
    receipt = Gemma3LayerMultimodeCompilationReceipt.from_state_dict(
        manifest["compilation_receipt"]
    )
    state = _validate_state_dict(raw["state_dict"], manifest["state_tensors"])
    if _state_sha256(state) != receipt.compiled_state_sha256:
        raise LayerMultimodeCompilationError(
            "bundle state does not reproduce the compilation receipt"
        )
    specs = _specs_from_bundle(
        state,
        manifest["layer_coefficients"],
        receipt.selection_receipt_sha256,
    )
    if tuple(spec.coefficient_sha256 for spec in specs) != tuple(
        row.coefficient_sha256 for row in receipt.layer_accounting
    ):
        raise LayerMultimodeCompilationError(
            "bundle coefficient inventory differs from compilation receipt"
        )
    return manifest, receipt, state, specs


def load_compiled_layer_multimode_bundle(
    path: Path | str,
    *,
    expected_bundle_sha256: str,
    model_factory: Callable[[], nn.Module],
    allow_test_nonproduction_shape: bool = False,
    allow_test_unbound_materialization: bool = False,
) -> CompiledGemma3LayerMultimodeModel:
    """Strict-load a source-free bundle into an architecture-only model shell."""

    source = Path(path)
    expected_bundle_sha256 = _require_sha256(
        expected_bundle_sha256, label="expected_bundle_sha256"
    )
    _strict_bool(
        allow_test_nonproduction_shape,
        label="allow_test_nonproduction_shape",
    )
    _strict_bool(
        allow_test_unbound_materialization,
        label="allow_test_unbound_materialization",
    )
    bundle_bytes, bundle_sha256, _ = _strict_read_bundle_bytes(source)
    if bundle_sha256 != expected_bundle_sha256:
        raise LayerMultimodeCompilationError("external bundle hash mismatch")
    _, receipt, state, specs = _decode_bundle_bytes(bundle_bytes)
    if (
        not receipt.production_provenance_authorized
        and not allow_test_unbound_materialization
    ):
        raise LayerMultimodeCompilationError(
            "loading an unbound materialization requires the explicit test-only opt-in"
        )
    if receipt.production_provenance_authorized and allow_test_unbound_materialization:
        raise LayerMultimodeCompilationError(
            "the unbound-materialization test override is forbidden for an "
            "authorized production bundle"
        )
    if receipt.production_shape_enforced == allow_test_nonproduction_shape:
        raise LayerMultimodeCompilationError(
            "bundle production-shape policy does not match the explicit load policy"
        )
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    shell = model_factory()
    if not isinstance(shell, nn.Module):
        raise TypeError("model_factory must return an nn.Module")
    materialized, shell_analysis = _materialize_structure(
        shell,
        specs,
        allow_test_nonproduction_shape=allow_test_nonproduction_shape,
    )
    try:
        materialized.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise LayerMultimodeCompilationError(
            "bundle state does not match the architecture-only model shell"
        ) from error
    loaded_receipt = _build_receipt(
        compiled=materialized,
        source_analysis=shell_analysis,
        source_model_sha256=receipt.source_model_sha256,
        selection_receipt_sha256=receipt.selection_receipt_sha256,
        selected_inventory_sha256=receipt.selected_inventory_sha256,
        production_provenance_authorized=(
            receipt.production_provenance_authorized
        ),
        recomputed_source_model_sha256=(
            receipt.recomputed_source_model_sha256
        ),
        source_execution_fingerprint_sha256=(
            receipt.source_execution_fingerprint_sha256
        ),
        recomputed_source_execution_fingerprint_sha256=(
            receipt.recomputed_source_execution_fingerprint_sha256
        ),
        source_state_sha256=receipt.source_state_sha256,
    )
    if loaded_receipt.state_dict() != receipt.state_dict():
        raise LayerMultimodeCompilationError(
            "loaded compilation receipt does not reproduce exactly"
        )
    result = CompiledGemma3LayerMultimodeModel(
        model=materialized,
        receipt=loaded_receipt,
    )
    result.assert_integrity()
    return result
