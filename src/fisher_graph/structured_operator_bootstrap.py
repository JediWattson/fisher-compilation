"""Activation-only system identification for structured transformer operators.

The compiler-side API in this module receives captured activations and a
destination executor.  It never receives a source module or a source
parameter, and it does not serialize teacher activations or sufficient
statistics.  Linear operators are recovered with deterministic streaming
ridge fits; Gemma-style RMSNorm gains are recovered by coordinate least
squares.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn

from .adapters.base import LayerSpec, StructuredOperatorSites
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)


STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM = (
    "calibration_a_internal_activation_active_support_ridge_v2"
)
STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA = (
    "fisher_graph.structured_operator_bootstrap"
)
STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION = 2
DEFAULT_STRUCTURED_OPERATOR_BOOTSTRAP_ROWS = 8_192
DEFAULT_STRUCTURED_OPERATOR_RIDGE_RELATIVE = 1e-10
DEFAULT_STRUCTURED_OPERATOR_RANK_RTOL = 1e-12
DEFAULT_STRUCTURED_OPERATOR_MAX_CONDITION = 1e12
DEFAULT_STRUCTURED_OPERATOR_MAXIMUM_NULLITY = 1
STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY = (
    "maximum_structural_nullity_active_support_v1"
)

_ROW_DOMAIN = b"fisher_graph.structured_operator_bootstrap.row.v1\0"
_ROW_SET_DOMAIN = (
    b"fisher_graph.structured_operator_bootstrap.row_set.v1\0"
)
_SITE_SCHEMA_DOMAIN = (
    b"fisher_graph.structured_operator_bootstrap.site_schema.v1\0"
)
_COEFFICIENT_DOMAIN = (
    b"fisher_graph.structured_operator_bootstrap.coefficients.v1\0"
)
_BOOTSTRAPPED_PARAMETER_NAMES = (
    "attention.q_proj.weight",
    "attention.k_proj.weight",
    "attention.v_proj.weight",
    "attention.o_proj.weight",
    "feed_forward.gate_proj.weight",
    "feed_forward.up_proj.weight",
    "feed_forward.down_proj.weight",
    "attention_input_norm.weight",
    "attention_output_norm.weight",
    "feed_forward_input_norm.weight",
    "feed_forward_output_norm.weight",
    "attention.q_norm.weight",
    "attention.k_norm.weight",
)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _tensor_digest_items(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(_COEFFICIENT_DOMAIN)
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def structured_operator_coefficient_sha256(
    executor: StructuredTransformerLayerExecutor,
) -> str:
    """Digest exactly the coefficients identified by the bootstrap."""

    if not isinstance(executor, StructuredTransformerLayerExecutor):
        raise TypeError(
            "executor must be StructuredTransformerLayerExecutor"
        )
    parameters = dict(executor.named_parameters())
    expected = set(_BOOTSTRAPPED_PARAMETER_NAMES)
    if set(parameters) != expected:
        raise ValueError(
            "structured executor coefficient schema drifted: "
            f"missing={sorted(expected - set(parameters))}, "
            f"unexpected={sorted(set(parameters) - expected)}"
        )
    return _tensor_digest_items(
        {
            name: parameters[name]
            for name in _BOOTSTRAPPED_PARAMETER_NAMES
        }
    )


@dataclass(frozen=True, slots=True)
class StructuredOperatorIdentityBatch:
    """Cheap row identities used to select calibration tokens before capture."""

    valid_positions: Tensor
    logical_positions: Tensor
    example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.valid_positions, Tensor)
            or self.valid_positions.dtype is not torch.bool
            or self.valid_positions.ndim != 2
            or not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.dtype
            not in (torch.int32, torch.int64)
            or self.logical_positions.shape != self.valid_positions.shape
            or len(self.example_ids) != self.valid_positions.shape[0]
            or any(
                not isinstance(example_id, str) or not example_id
                for example_id in self.example_ids
            )
        ):
            raise ValueError(
                "operator row masks, positions, and example ids are "
                "inconsistent"
            )

    @property
    def batch_size(self) -> int:
        return int(self.valid_positions.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.valid_positions.shape[1])


@dataclass(frozen=True, slots=True)
class StructuredOperatorCaptureBatch(StructuredOperatorIdentityBatch):
    """One compact activation-only batch used by the operator bootstrap."""

    activations: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        super(StructuredOperatorCaptureBatch, self).__post_init__()
        if (
            not isinstance(self.activations, Mapping)
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, Tensor)
                or not value.is_floating_point()
                for name, value in self.activations.items()
            )
        ):
            raise TypeError(
                "operator capture activations must map names to floating "
                "tensors"
            )


@dataclass(frozen=True, slots=True)
class StructuredOperatorRowSelection:
    """Exact lowest-hash token rows chosen before activation capture.

    The selection contains identities only, so a runner can compute it over
    inexpensive prompt metadata, use :meth:`mask_for` while each source batch
    is live, and retain only the selected activation rows.
    """

    layer_id: str
    calibration_split_sha256: str
    requested_rows: int
    valid_rows: int
    selected_identities: tuple[tuple[str, int], ...]
    selected_rows_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id:
            raise ValueError("row selection layer_id must be nonempty")
        _require_sha256(
            self.calibration_split_sha256,
            label="calibration_split_sha256",
        )
        _require_sha256(
            self.selected_rows_sha256,
            label="selected_rows_sha256",
        )
        if (
            type(self.requested_rows) is not int
            or self.requested_rows <= 0
            or type(self.valid_rows) is not int
            or self.valid_rows < len(self.selected_identities)
            or len(self.selected_identities)
            != min(self.requested_rows, self.valid_rows)
        ):
            raise ValueError("row selection counts are inconsistent")
        if (
            any(
                not isinstance(example_id, str)
                or not example_id
                or type(logical_position) is not int
                for example_id, logical_position in self.selected_identities
            )
            or len(set(self.selected_identities))
            != len(self.selected_identities)
        ):
            raise ValueError(
                "selected row identities must be unique (example, position) "
                "pairs"
            )
        expected_order = tuple(
            identity
            for _, identity in sorted(
                (
                    (
                        _row_key(
                            calibration_split_sha256=(
                                self.calibration_split_sha256
                            ),
                            layer_id=self.layer_id,
                            example_id=example_id,
                            logical_position=logical_position,
                        ),
                        (example_id, logical_position),
                    )
                    for example_id, logical_position
                    in self.selected_identities
                ),
                key=lambda record: (
                    record[0],
                    record[1][0],
                    record[1][1],
                ),
            )
        )
        if expected_order != self.selected_identities:
            raise ValueError(
                "selected row identities must be in lowest-hash order"
            )
        if (
            _selected_rows_digest(
                self.selected_identities,
                calibration_split_sha256=self.calibration_split_sha256,
                layer_id=self.layer_id,
            )
            != self.selected_rows_sha256
        ):
            raise ValueError("selected row digest does not match identities")

    @property
    def selected_rows(self) -> int:
        return len(self.selected_identities)

    def mask_for(
        self,
        batch: StructuredOperatorIdentityBatch,
    ) -> Tensor:
        """Return the selected-token mask for one uncaptured source batch."""

        if not isinstance(batch, StructuredOperatorIdentityBatch):
            raise TypeError(
                "batch must be StructuredOperatorIdentityBatch"
            )
        selected = set(self.selected_identities)
        valid = batch.valid_positions.detach().cpu()
        positions = batch.logical_positions.detach().cpu()
        mask = torch.zeros_like(valid)
        for row, column in valid.nonzero(as_tuple=False).tolist():
            identity = (
                batch.example_ids[row],
                int(positions[row, column].item()),
            )
            if identity in selected:
                mask[row, column] = True
        return mask.to(device=batch.valid_positions.device)

    def report(self) -> dict[str, object]:
        """Return the serializable identity-free selection report."""

        return {
            "algorithm": "lowest_sha256_valid_token_rows_v1",
            "hash_domain": _ROW_DOMAIN.decode("ascii").rstrip("\0"),
            "requested_rows": self.requested_rows,
            "valid_rows": self.valid_rows,
            "selected_rows": self.selected_rows,
            "selected_rows_sha256": self.selected_rows_sha256,
            "selection_depends_on_activation_values": False,
            "selection_depends_on_teacher_targets": False,
        }


@dataclass(slots=True)
class _RidgeGroup:
    input_width: int
    bias: bool
    output_widths: Mapping[str, int]
    gram: Tensor = field(init=False)
    cross: dict[str, Tensor] = field(init=False)
    target_energy: dict[str, Tensor] = field(init=False)
    rows: int = field(init=False)

    def __post_init__(self) -> None:
        augmented = self.input_width + int(self.bias)
        self.gram = torch.zeros(
            augmented,
            augmented,
            dtype=torch.float64,
        )
        self.cross = {
            name: torch.zeros(
                augmented,
                width,
                dtype=torch.float64,
            )
            for name, width in self.output_widths.items()
        }
        self.target_energy = {
            name: torch.zeros((), dtype=torch.float64)
            for name in self.output_widths
        }
        self.rows = 0

    def add(self, inputs: Tensor, outputs: Mapping[str, Tensor]) -> None:
        x = inputs.detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != self.input_width:
            raise ValueError("ridge input rows have an incompatible shape")
        if self.bias:
            x = torch.cat(
                (x, torch.ones(x.shape[0], 1, dtype=torch.float64)),
                dim=1,
            )
        if set(outputs) != set(self.output_widths):
            raise ValueError("ridge output fields are inconsistent")
        self.gram.add_(x.T @ x)
        for name, width in self.output_widths.items():
            y = outputs[name].detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            if y.ndim != 2 or y.shape != (x.shape[0], width):
                raise ValueError(
                    f"ridge output {name!r} has an incompatible shape"
                )
            self.cross[name].add_(x.T @ y)
            self.target_energy[name].add_(y.square().sum())
        self.rows += int(x.shape[0])

    def solve(
        self,
        *,
        ridge_relative: float,
        rank_rtol: float,
        maximum_condition: float,
        maximum_nullity: int,
    ) -> tuple[
        dict[str, tuple[Tensor, Tensor | None]],
        dict[str, dict[str, object]],
    ]:
        dimension = int(self.gram.shape[0])
        if self.rows < dimension:
            raise ValueError(
                "operator ridge fit has fewer selected rows than input "
                f"dimensions: rows={self.rows}, dimensions={dimension}"
            )
        gram = (self.gram + self.gram.T) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        maximum = float(eigenvalues.max().item())
        if maximum <= torch.finfo(torch.float64).tiny:
            raise ValueError("operator ridge design has zero energy")
        threshold = rank_rtol * maximum
        positive = eigenvalues[eigenvalues > threshold]
        effective_rank = int(positive.numel())
        nullity = dimension - effective_rank
        if nullity > maximum_nullity:
            smallest = [
                float(value)
                for value in eigenvalues[: min(5, dimension)].tolist()
            ]
            raise ValueError(
                "operator ridge design is rank deficient beyond the "
                "active-support policy: "
                f"rank={effective_rank}, dimensions={dimension}, "
                f"nullity={nullity}, maximum_nullity={maximum_nullity}, "
                f"minimum_eigenvalue={float(eigenvalues.min().item())}, "
                f"maximum_eigenvalue={maximum}, threshold={threshold}, "
                f"smallest_eigenvalues={smallest}"
            )
        minimum = float(positive.min().item())
        active_condition = maximum / minimum
        if (
            not math.isfinite(active_condition)
            or active_condition > maximum_condition
        ):
            raise ValueError(
                "operator ridge active support exceeds the condition limit: "
                f"condition={active_condition}, limit={maximum_condition}"
            )
        ridge = (
            ridge_relative
            * float(torch.diagonal(gram).mean().item())
        )
        regularizer = torch.eye(dimension, dtype=torch.float64)
        if self.bias:
            regularizer[-1, -1] = 0
        active_vectors = eigenvectors[:, eigenvalues > threshold]
        active_system = (
            torch.diag(positive)
            + ridge * (active_vectors.T @ regularizer @ active_vectors)
        )
        fitted = {}
        reports = {}
        for name, cross in self.cross.items():
            active_cross = active_vectors.T @ cross
            active_solution = torch.linalg.solve(
                active_system,
                active_cross,
            )
            solution = active_vectors @ active_solution
            weight = solution[: self.input_width].T.contiguous()
            bias = (
                solution[self.input_width].contiguous()
                if self.bias
                else None
            )
            quadratic = float(
                (solution * (gram @ solution)).sum().item()
            )
            alignment = float((solution * cross).sum().item())
            target_energy = float(self.target_energy[name].item())
            squared_error = max(
                0.0,
                target_energy - 2.0 * alignment + quadratic,
            )
            fit_rmse = math.sqrt(
                squared_error
                / max(
                    self.rows * int(cross.shape[1]),
                    1,
                )
            )
            fit_nrmse = (
                math.sqrt(squared_error / target_energy)
                if target_energy > torch.finfo(torch.float64).tiny
                else (0.0 if squared_error == 0 else math.inf)
            )
            fitted[name] = (weight, bias)
            reports[name] = {
                "rows": self.rows,
                "input_width": self.input_width,
                "output_width": int(cross.shape[1]),
                "bias": self.bias,
                "dimension": dimension,
                "effective_rank": effective_rank,
                "nullity": nullity,
                "full_column_rank": nullity == 0,
                "active_condition_number": active_condition,
                "rank_policy": STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY,
                "maximum_nullity": maximum_nullity,
                "rank_relative_tolerance": rank_rtol,
                "maximum_condition_number": maximum_condition,
                "ridge_relative_to_mean_gram_diagonal": (
                    ridge_relative
                ),
                "ridge_absolute": ridge,
                "fit_rmse": fit_rmse,
                "fit_nrmse": fit_nrmse,
            }
        return fitted, reports


@dataclass(slots=True)
class _NormFit:
    width: int
    epsilon: float
    numerator: Tensor = field(init=False)
    denominator: Tensor = field(init=False)
    target_energy: Tensor = field(init=False)
    rows: int = field(init=False)

    def __post_init__(self) -> None:
        self.numerator = torch.zeros(self.width, dtype=torch.float64)
        self.denominator = torch.zeros(self.width, dtype=torch.float64)
        self.target_energy = torch.zeros(self.width, dtype=torch.float64)
        self.rows = 0

    def add(self, inputs: Tensor, targets: Tensor) -> None:
        x = inputs.detach().to(device="cpu", dtype=torch.float64)
        y = targets.detach().to(device="cpu", dtype=torch.float64)
        if (
            x.shape != y.shape
            or x.ndim < 2
            or x.shape[-1] != self.width
        ):
            raise ValueError("RMSNorm activation pairs have incompatible shapes")
        x = x.reshape(-1, self.width)
        y = y.reshape(-1, self.width)
        base = x * torch.rsqrt(
            x.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        self.numerator.add_((base * y).sum(dim=0))
        self.denominator.add_(base.square().sum(dim=0))
        self.target_energy.add_(y.square().sum(dim=0))
        self.rows += int(x.shape[0])

    def solve(self) -> tuple[Tensor, dict[str, object]]:
        identifiable = self.denominator > torch.finfo(torch.float64).tiny
        if not bool(identifiable.all()):
            raise ValueError(
                "RMSNorm activation pairs do not identify every coordinate"
            )
        gain = self.numerator / self.denominator
        if not bool(torch.isfinite(gain).all()):
            raise ValueError("RMSNorm fitted gains are nonfinite")
        weight = gain - 1.0
        residual = (
            self.target_energy
            - self.numerator.square() / self.denominator
        ).clamp_min(0)
        target_energy = float(self.target_energy.sum().item())
        squared_error = float(residual.sum().item())
        report = {
            "rows": self.rows,
            "width": self.width,
            "identified_coordinates": int(identifiable.sum().item()),
            "fit_nrmse": (
                math.sqrt(squared_error / target_energy)
                if target_energy > torch.finfo(torch.float64).tiny
                else (0.0 if squared_error == 0 else math.inf)
            ),
            "weight_minimum": float(weight.min().item()),
            "weight_median": float(weight.median().item()),
            "weight_maximum": float(weight.max().item()),
            "weight_rms": float(weight.square().mean().sqrt().item()),
        }
        return weight, report


def _site_schema(layer: LayerSpec) -> dict[str, object]:
    transformer = layer.transformer
    attention = layer.attention
    if (
        transformer is None
        or attention is None
        or not isinstance(
            transformer.operator_sites,
            StructuredOperatorSites,
        )
        or transformer.qk_norm is None
    ):
        raise ValueError(
            "operator bootstrap requires complete structured layer semantics"
        )
    attention_stage, feed_forward_stage = transformer.stages
    sites = transformer.operator_sites
    return {
        "layer_id": layer.id,
        "residual_width": layer.residual_width,
        "query_heads": attention.query_heads,
        "key_value_heads": attention.key_value_heads,
        "head_dimension": attention.head_dimension,
        "query_width": attention.query_heads * attention.head_dimension,
        "key_value_width": (
            attention.key_value_heads * attention.head_dimension
        ),
        "feed_forward_width": transformer.feed_forward.intermediate_width,
        "projection_bias": {
            "attention": transformer.attention_projection_bias,
            "feed_forward": transformer.feed_forward.projection_bias,
        },
        "residual_sites": {
            "layer_input": layer.input_site,
            "attention_normalized_input": (
                attention_stage.normalized_input_site
            ),
            "attention_operator_output": (
                attention_stage.operator_output_site
            ),
            "attention_delta": attention_stage.delta_site,
            "post_attention": attention_stage.output_site,
            "feed_forward_normalized_input": (
                feed_forward_stage.normalized_input_site
            ),
            "feed_forward_operator_output": (
                feed_forward_stage.operator_output_site
            ),
            "feed_forward_delta": feed_forward_stage.delta_site,
        },
        "operator_sites": asdict(sites),
    }


def structured_operator_site_schema_sha256(layer: LayerSpec) -> str:
    """Digest the activation-site schema required by the bootstrap."""

    if not isinstance(layer, LayerSpec):
        raise TypeError("layer must be a LayerSpec")
    return _json_sha256(
        _site_schema(layer),
        domain=_SITE_SCHEMA_DOMAIN,
    )


def structured_operator_site_schema(layer: LayerSpec) -> dict[str, object]:
    """Return the JSON-safe activation-site schema for strict replay."""

    if not isinstance(layer, LayerSpec):
        raise TypeError("layer must be a LayerSpec")
    return _site_schema(layer)


def _required_shapes(
    schema: Mapping[str, object],
) -> dict[str, tuple[int | None, ...]]:
    residual = int(schema["residual_width"])
    query_heads = int(schema["query_heads"])
    key_value_heads = int(schema["key_value_heads"])
    head_dimension = int(schema["head_dimension"])
    query_width = int(schema["query_width"])
    feed_forward_width = int(schema["feed_forward_width"])
    residual_sites = schema["residual_sites"]
    operator_sites = schema["operator_sites"]
    assert isinstance(residual_sites, Mapping)
    assert isinstance(operator_sites, Mapping)
    result = {
        str(site): (None, None, residual)
        for site in residual_sites.values()
    }
    result.update(
        {
            str(operator_sites["attention_query_projection"]): (
                None,
                None,
                query_heads,
                head_dimension,
            ),
            str(operator_sites["attention_query_normalized"]): (
                None,
                None,
                query_heads,
                head_dimension,
            ),
            str(operator_sites["attention_key_projection"]): (
                None,
                None,
                key_value_heads,
                head_dimension,
            ),
            str(operator_sites["attention_key_normalized"]): (
                None,
                None,
                key_value_heads,
                head_dimension,
            ),
            str(operator_sites["attention_value_projection"]): (
                None,
                None,
                key_value_heads,
                head_dimension,
            ),
            str(operator_sites["attention_context"]): (
                None,
                None,
                query_width,
            ),
            str(operator_sites["feed_forward_gate_projection"]): (
                None,
                None,
                feed_forward_width,
            ),
            str(operator_sites["feed_forward_up_projection"]): (
                None,
                None,
                feed_forward_width,
            ),
            str(operator_sites["feed_forward_down_input"]): (
                None,
                None,
                feed_forward_width,
            ),
        }
    )
    return result


def _validate_capture_batches(
    batches: Sequence[StructuredOperatorCaptureBatch],
    *,
    schema: Mapping[str, object],
) -> None:
    if not batches:
        raise ValueError("operator bootstrap capture batches cannot be empty")
    shapes = _required_shapes(schema)
    for batch in batches:
        if not isinstance(batch, StructuredOperatorCaptureBatch):
            raise TypeError(
                "operator bootstrap batches must be "
                "StructuredOperatorCaptureBatch values"
            )
        for site, expected in shapes.items():
            value = batch.activations.get(site)
            if not isinstance(value, Tensor):
                raise ValueError(
                    f"operator bootstrap activation {site!r} is missing"
                )
            actual = tuple(value.shape)
            if (
                len(actual) != len(expected)
                or actual[:2]
                != (batch.batch_size, batch.sequence_length)
                or any(
                    required is not None and actual[index] != required
                    for index, required in enumerate(expected)
                )
            ):
                raise ValueError(
                    f"operator bootstrap activation {site!r} has shape "
                    f"{actual}, expected {expected}"
                )


def _row_key(
    *,
    calibration_split_sha256: str,
    layer_id: str,
    example_id: str,
    logical_position: int,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(_ROW_DOMAIN)
    digest.update(bytes.fromhex(calibration_split_sha256))
    digest.update(layer_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(example_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(struct.pack(">q", logical_position))
    return digest.digest()


def _selected_rows_digest(
    identities: Sequence[tuple[str, int]],
    *,
    calibration_split_sha256: str,
    layer_id: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_ROW_SET_DOMAIN)
    for example_id, logical_position in identities:
        digest.update(
            _row_key(
                calibration_split_sha256=calibration_split_sha256,
                layer_id=layer_id,
                example_id=example_id,
                logical_position=logical_position,
            )
        )
        digest.update(example_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack(">q", logical_position))
    return digest.hexdigest()


def select_structured_operator_rows(
    batches: Sequence[StructuredOperatorIdentityBatch],
    *,
    calibration_split_sha256: str,
    layer_id: str,
    requested_rows: int = DEFAULT_STRUCTURED_OPERATOR_BOOTSTRAP_ROWS,
) -> StructuredOperatorRowSelection:
    """Select exact lowest-hash valid token identities before source capture."""

    calibration_split_sha256 = _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    if not isinstance(layer_id, str) or not layer_id:
        raise ValueError("layer_id must be nonempty")
    if type(requested_rows) is not int or requested_rows <= 0:
        raise ValueError("requested_rows must be positive")
    if not batches:
        raise ValueError("operator row identity batches cannot be empty")
    records: list[tuple[bytes, str, int]] = []
    identities: set[tuple[str, int]] = set()
    for batch in batches:
        if not isinstance(batch, StructuredOperatorIdentityBatch):
            raise TypeError(
                "row selection batches must be "
                "StructuredOperatorIdentityBatch values"
            )
        valid = batch.valid_positions.detach().cpu()
        positions = batch.logical_positions.detach().cpu()
        for row, column in valid.nonzero(as_tuple=False).tolist():
            identity = (
                batch.example_ids[row],
                int(positions[row, column].item()),
            )
            if identity in identities:
                raise ValueError(
                    "operator bootstrap row identities must be unique"
                )
            identities.add(identity)
            key = _row_key(
                calibration_split_sha256=calibration_split_sha256,
                layer_id=layer_id,
                example_id=identity[0],
                logical_position=identity[1],
            )
            records.append((key, identity[0], identity[1]))
    valid_rows = len(records)
    if valid_rows == 0:
        raise ValueError("operator row selection has no valid rows")
    selected_records = sorted(
        records,
        key=lambda record: (record[0], record[1], record[2]),
    )[
        : min(requested_rows, valid_rows)
    ]
    selected_identities = tuple(
        (example_id, logical_position)
        for _, example_id, logical_position in selected_records
    )
    return StructuredOperatorRowSelection(
        layer_id=layer_id,
        calibration_split_sha256=calibration_split_sha256,
        requested_rows=requested_rows,
        valid_rows=valid_rows,
        selected_identities=selected_identities,
        selected_rows_sha256=_selected_rows_digest(
            selected_identities,
            calibration_split_sha256=calibration_split_sha256,
            layer_id=layer_id,
        ),
    )


def _indices_for_selection(
    batches: Sequence[StructuredOperatorCaptureBatch],
    *,
    selection: StructuredOperatorRowSelection,
    require_compact: bool,
) -> dict[int, tuple[Tensor, Tensor]]:
    wanted = set(selection.selected_identities)
    found: set[tuple[str, int]] = set()
    grouped: dict[int, list[tuple[int, int]]] = {}
    for batch_index, batch in enumerate(batches):
        valid = batch.valid_positions.detach().cpu()
        positions = batch.logical_positions.detach().cpu()
        for row, column in valid.nonzero(as_tuple=False).tolist():
            identity = (
                batch.example_ids[row],
                int(positions[row, column].item()),
            )
            if identity not in wanted:
                if require_compact:
                    raise ValueError(
                        "preselected capture batches contain an unselected "
                        f"row identity: {identity!r}"
                    )
                continue
            if identity in found:
                raise ValueError(
                    "operator bootstrap selected row identities must occur "
                    "exactly once"
                )
            found.add(identity)
            grouped.setdefault(batch_index, []).append((row, column))
    missing = wanted - found
    if missing:
        raise ValueError(
            "operator bootstrap is missing preselected activation rows: "
            f"{sorted(missing)[:3]}"
        )
    return {
        batch_index: (
            torch.tensor(
                [row for row, _ in rows],
                dtype=torch.long,
            ),
            torch.tensor(
                [column for _, column in rows],
                dtype=torch.long,
            ),
        )
        for batch_index, rows in grouped.items()
    }


def _selected(
    batch: StructuredOperatorCaptureBatch,
    site: str,
    indices: tuple[Tensor, Tensor],
) -> Tensor:
    value = batch.activations[site]
    rows, columns = indices
    selected = value[
        rows.to(device=value.device),
        columns.to(device=value.device),
    ]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError(
            f"operator bootstrap activation {site!r} contains nonfinite "
            "selected rows"
        )
    return selected


def _linear_assignment(
    assignments: dict[str, tuple[nn.Parameter, Tensor]],
    *,
    name: str,
    module: nn.Linear,
    fitted: tuple[Tensor, Tensor | None],
) -> None:
    weight, bias = fitted
    if weight.shape != module.weight.shape:
        raise ValueError(
            f"fitted operator {name!r} weight has an incompatible shape"
        )
    assignments[f"{name}.weight"] = (module.weight, weight)
    if module.bias is None:
        if bias is not None:
            raise ValueError(
                f"fitted operator {name!r} unexpectedly contains a bias"
            )
    else:
        if bias is None or bias.shape != module.bias.shape:
            raise ValueError(
                f"fitted operator {name!r} bias has an incompatible shape"
            )
        assignments[f"{name}.bias"] = (module.bias, bias)


def _norm_assignment(
    assignments: dict[str, tuple[nn.Parameter, Tensor]],
    *,
    name: str,
    module: nn.Module,
    weight: Tensor,
) -> None:
    parameter = getattr(module, "weight", None)
    if (
        not isinstance(parameter, nn.Parameter)
        or parameter.shape != weight.shape
    ):
        raise ValueError(
            f"fitted normalization {name!r} has an incompatible destination"
        )
    assignments[f"{name}.weight"] = (parameter, weight)


def bootstrap_structured_operator_executor_(
    destination: StructuredTransformerLayerExecutor,
    batches: Sequence[StructuredOperatorCaptureBatch],
    *,
    layer: LayerSpec,
    calibration_split_sha256: str,
    source_segment_fingerprint: str,
    requested_rows: int = DEFAULT_STRUCTURED_OPERATOR_BOOTSTRAP_ROWS,
    ridge_relative: float = DEFAULT_STRUCTURED_OPERATOR_RIDGE_RELATIVE,
    rank_relative_tolerance: float = DEFAULT_STRUCTURED_OPERATOR_RANK_RTOL,
    maximum_condition_number: float = (
        DEFAULT_STRUCTURED_OPERATOR_MAX_CONDITION
    ),
    maximum_nullity: int = DEFAULT_STRUCTURED_OPERATOR_MAXIMUM_NULLITY,
    row_selection: StructuredOperatorRowSelection | None = None,
) -> dict[str, object]:
    """Fit a full-width executor from A-only internal activation pairs.

    The destination remains source-free: only independently solved
    coefficients are written and the executor's source-weight-origin marker is
    required to remain clear.  Supplying ``row_selection`` requires the input
    batches to contain exactly those selected valid rows; this is the
    memory-bounded runner path.
    """

    if not isinstance(destination, StructuredTransformerLayerExecutor):
        raise TypeError(
            "destination must be StructuredTransformerLayerExecutor"
        )
    if destination.owns_source_model_weights:
        raise ValueError(
            "operator bootstrap refuses a source-weight-contaminated "
            "destination"
        )
    if not isinstance(layer, LayerSpec):
        raise TypeError("layer must be a LayerSpec")
    calibration_split_sha256 = _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    source_segment_fingerprint = _require_sha256(
        source_segment_fingerprint,
        label="source_segment_fingerprint",
    )
    if type(requested_rows) is not int or requested_rows <= 0:
        raise ValueError("requested_rows must be positive")
    if type(maximum_nullity) is not int or maximum_nullity < 0:
        raise ValueError("maximum_nullity must be a nonnegative integer")
    for name, value, minimum in (
        ("ridge_relative", ridge_relative, 0.0),
        ("maximum_condition_number", maximum_condition_number, 1.0),
    ):
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= minimum
        ):
            raise ValueError(f"{name} must be finite and greater than {minimum}")
    if (
        not isinstance(rank_relative_tolerance, (float, int))
        or isinstance(rank_relative_tolerance, bool)
        or not math.isfinite(float(rank_relative_tolerance))
        or not 0.0 < float(rank_relative_tolerance) < 1.0
    ):
        raise ValueError(
            "rank_relative_tolerance must be finite and strictly between "
            "zero and one"
        )
    if (
        destination.config.attention != layer.attention
        or destination.config.transformer != layer.transformer
        or destination.width != layer.residual_width
    ):
        raise ValueError(
            "destination executor does not match the structured source layer"
        )

    schema = _site_schema(layer)
    _validate_capture_batches(batches, schema=schema)
    selection_applied_before_capture = row_selection is not None
    if row_selection is None:
        row_selection = select_structured_operator_rows(
            tuple(
                StructuredOperatorIdentityBatch(
                    valid_positions=batch.valid_positions,
                    logical_positions=batch.logical_positions,
                    example_ids=batch.example_ids,
                )
                for batch in batches
            ),
            calibration_split_sha256=calibration_split_sha256,
            layer_id=layer.id,
            requested_rows=requested_rows,
        )
    else:
        if not isinstance(row_selection, StructuredOperatorRowSelection):
            raise TypeError(
                "row_selection must be StructuredOperatorRowSelection"
            )
        if (
            row_selection.layer_id != layer.id
            or row_selection.calibration_split_sha256
            != calibration_split_sha256
            or row_selection.requested_rows != requested_rows
        ):
            raise ValueError(
                "precomputed row selection does not match the bootstrap "
                "layer, split, or requested row count"
            )
    row_indices = _indices_for_selection(
        batches,
        selection=row_selection,
        require_compact=selection_applied_before_capture,
    )
    selection_report = row_selection.report()
    selection_report["selection_applied_before_activation_capture"] = (
        selection_applied_before_capture
    )
    selection_report["capture_contains_only_selected_rows"] = (
        selection_applied_before_capture
    )
    residual_width = int(schema["residual_width"])
    query_heads = int(schema["query_heads"])
    key_value_heads = int(schema["key_value_heads"])
    head_dimension = int(schema["head_dimension"])
    query_width = int(schema["query_width"])
    key_value_width = int(schema["key_value_width"])
    feed_forward_width = int(schema["feed_forward_width"])
    residual_sites = schema["residual_sites"]
    operator_sites = schema["operator_sites"]
    assert isinstance(residual_sites, Mapping)
    assert isinstance(operator_sites, Mapping)

    attention_bias = destination.attention.q_proj.bias is not None
    feed_forward_bias = destination.feed_forward.gate_proj.bias is not None
    attention_inputs = _RidgeGroup(
        residual_width,
        attention_bias,
        {
            "attention.q_proj": query_width,
            "attention.k_proj": key_value_width,
            "attention.v_proj": key_value_width,
        },
    )
    attention_output = _RidgeGroup(
        query_width,
        destination.attention.o_proj.bias is not None,
        {"attention.o_proj": residual_width},
    )
    feed_forward_inputs = _RidgeGroup(
        residual_width,
        feed_forward_bias,
        {
            "feed_forward.gate_proj": feed_forward_width,
            "feed_forward.up_proj": feed_forward_width,
        },
    )
    feed_forward_output = _RidgeGroup(
        feed_forward_width,
        destination.feed_forward.down_proj.bias is not None,
        {"feed_forward.down_proj": residual_width},
    )
    transformer = layer.transformer
    assert transformer is not None and transformer.qk_norm is not None
    norm_fits = {
        "attention_input_norm": _NormFit(
            residual_width,
            float(transformer.attention_input_norm.epsilon),
        ),
        "attention_output_norm": _NormFit(
            residual_width,
            float(transformer.attention_output_norm.epsilon),
        ),
        "feed_forward_input_norm": _NormFit(
            residual_width,
            float(transformer.feed_forward_input_norm.epsilon),
        ),
        "feed_forward_output_norm": _NormFit(
            residual_width,
            float(transformer.feed_forward_output_norm.epsilon),
        ),
        "attention.q_norm": _NormFit(
            head_dimension,
            float(transformer.qk_norm.epsilon),
        ),
        "attention.k_norm": _NormFit(
            head_dimension,
            float(transformer.qk_norm.epsilon),
        ),
    }

    for batch_index, batch in enumerate(batches):
        indices = row_indices.get(batch_index)
        if indices is None:
            continue
        attention_normalized = _selected(
            batch,
            str(residual_sites["attention_normalized_input"]),
            indices,
        )
        query_projection = _selected(
            batch,
            str(operator_sites["attention_query_projection"]),
            indices,
        )
        key_projection = _selected(
            batch,
            str(operator_sites["attention_key_projection"]),
            indices,
        )
        value_projection = _selected(
            batch,
            str(operator_sites["attention_value_projection"]),
            indices,
        )
        attention_inputs.add(
            attention_normalized,
            {
                "attention.q_proj": query_projection.reshape(
                    query_projection.shape[0],
                    query_width,
                ),
                "attention.k_proj": key_projection.reshape(
                    key_projection.shape[0],
                    key_value_width,
                ),
                "attention.v_proj": value_projection.reshape(
                    value_projection.shape[0],
                    key_value_width,
                ),
            },
        )
        attention_output.add(
            _selected(
                batch,
                str(operator_sites["attention_context"]),
                indices,
            ),
            {
                "attention.o_proj": _selected(
                    batch,
                    str(residual_sites["attention_operator_output"]),
                    indices,
                )
            },
        )
        feed_forward_normalized = _selected(
            batch,
            str(residual_sites["feed_forward_normalized_input"]),
            indices,
        )
        feed_forward_inputs.add(
            feed_forward_normalized,
            {
                "feed_forward.gate_proj": _selected(
                    batch,
                    str(
                        operator_sites[
                            "feed_forward_gate_projection"
                        ]
                    ),
                    indices,
                ),
                "feed_forward.up_proj": _selected(
                    batch,
                    str(
                        operator_sites[
                            "feed_forward_up_projection"
                        ]
                    ),
                    indices,
                ),
            },
        )
        feed_forward_output.add(
            _selected(
                batch,
                str(operator_sites["feed_forward_down_input"]),
                indices,
            ),
            {
                "feed_forward.down_proj": _selected(
                    batch,
                    str(residual_sites["feed_forward_operator_output"]),
                    indices,
                )
            },
        )
        norm_fits["attention_input_norm"].add(
            _selected(
                batch,
                str(residual_sites["layer_input"]),
                indices,
            ),
            attention_normalized,
        )
        norm_fits["attention_output_norm"].add(
            _selected(
                batch,
                str(residual_sites["attention_operator_output"]),
                indices,
            ),
            _selected(
                batch,
                str(residual_sites["attention_delta"]),
                indices,
            ),
        )
        norm_fits["feed_forward_input_norm"].add(
            _selected(
                batch,
                str(residual_sites["post_attention"]),
                indices,
            ),
            feed_forward_normalized,
        )
        norm_fits["feed_forward_output_norm"].add(
            _selected(
                batch,
                str(residual_sites["feed_forward_operator_output"]),
                indices,
            ),
            _selected(
                batch,
                str(residual_sites["feed_forward_delta"]),
                indices,
            ),
        )
        norm_fits["attention.q_norm"].add(
            query_projection,
            _selected(
                batch,
                str(operator_sites["attention_query_normalized"]),
                indices,
            ),
        )
        norm_fits["attention.k_norm"].add(
            key_projection,
            _selected(
                batch,
                str(operator_sites["attention_key_normalized"]),
                indices,
            ),
        )

    solve_arguments = {
        "ridge_relative": float(ridge_relative),
        "rank_rtol": float(rank_relative_tolerance),
        "maximum_condition": float(maximum_condition_number),
        "maximum_nullity": maximum_nullity,
    }
    fitted_operators: dict[str, tuple[Tensor, Tensor | None]] = {}
    operator_reports: dict[str, dict[str, object]] = {}
    for group in (
        attention_inputs,
        attention_output,
        feed_forward_inputs,
        feed_forward_output,
    ):
        fitted, reports = group.solve(**solve_arguments)
        fitted_operators.update(fitted)
        operator_reports.update(reports)
    fitted_norms = {}
    norm_reports = {}
    for name, fit in norm_fits.items():
        weight, report = fit.solve()
        fitted_norms[name] = weight
        norm_reports[name] = report

    assignments: dict[str, tuple[nn.Parameter, Tensor]] = {}
    _linear_assignment(
        assignments,
        name="attention.q_proj",
        module=destination.attention.q_proj,
        fitted=fitted_operators["attention.q_proj"],
    )
    _linear_assignment(
        assignments,
        name="attention.k_proj",
        module=destination.attention.k_proj,
        fitted=fitted_operators["attention.k_proj"],
    )
    _linear_assignment(
        assignments,
        name="attention.v_proj",
        module=destination.attention.v_proj,
        fitted=fitted_operators["attention.v_proj"],
    )
    _linear_assignment(
        assignments,
        name="attention.o_proj",
        module=destination.attention.o_proj,
        fitted=fitted_operators["attention.o_proj"],
    )
    _linear_assignment(
        assignments,
        name="feed_forward.gate_proj",
        module=destination.feed_forward.gate_proj,
        fitted=fitted_operators["feed_forward.gate_proj"],
    )
    _linear_assignment(
        assignments,
        name="feed_forward.up_proj",
        module=destination.feed_forward.up_proj,
        fitted=fitted_operators["feed_forward.up_proj"],
    )
    _linear_assignment(
        assignments,
        name="feed_forward.down_proj",
        module=destination.feed_forward.down_proj,
        fitted=fitted_operators["feed_forward.down_proj"],
    )
    for name, module in (
        ("attention_input_norm", destination.attention_input_norm),
        ("attention_output_norm", destination.attention_output_norm),
        ("feed_forward_input_norm", destination.feed_forward_input_norm),
        ("feed_forward_output_norm", destination.feed_forward_output_norm),
        ("attention.q_norm", destination.attention.q_norm),
        ("attention.k_norm", destination.attention.k_norm),
    ):
        _norm_assignment(
            assignments,
            name=name,
            module=module,
            weight=fitted_norms[name],
        )

    applied = {}
    with torch.no_grad():
        for name, (parameter, value) in assignments.items():
            converted = value.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            if not bool(torch.isfinite(converted).all()):
                raise ValueError(
                    f"fitted coefficient {name!r} is nonfinite"
                )
            parameter.copy_(converted)
            applied[name] = parameter.detach().cpu().clone()
    if destination.owns_source_model_weights:
        raise RuntimeError(
            "operator bootstrap contaminated the destination source marker"
        )
    destination.eval()
    site_schema_sha256 = structured_operator_site_schema_sha256(layer)
    return {
        "schema": STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA,
        "format_version": STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION,
        "algorithm": STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM,
        "layer_id": layer.id,
        "calibration_split_sha256": calibration_split_sha256,
        "source_segment_fingerprint": source_segment_fingerprint,
        "site_schema": schema,
        "site_schema_sha256": site_schema_sha256,
        "row_selection": selection_report,
        "solver": {
            "accumulation_dtype": "torch.float64",
            "solve_dtype": "torch.float64",
            "ridge_relative_to_mean_gram_diagonal": float(
                ridge_relative
            ),
            "rank_relative_tolerance": float(
                rank_relative_tolerance
            ),
            "maximum_condition_number": float(
                maximum_condition_number
            ),
            "rank_policy": STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY,
            "maximum_nullity": maximum_nullity,
            "bias_regularized": False,
        },
        "operators": operator_reports,
        "normalizations": norm_reports,
        "coefficient_sha256": structured_operator_coefficient_sha256(
            destination
        ),
        "source_module_or_parameter_read": False,
        "direct_source_tensor_copy": False,
        "activation_targets_serialized": False,
        "sufficient_statistics_serialized": False,
        "destination_source_weight_contamination": False,
        "destination_executor_local_source_free": (
            destination.executor_local_source_free
        ),
    }


__all__ = [
    "DEFAULT_STRUCTURED_OPERATOR_BOOTSTRAP_ROWS",
    "DEFAULT_STRUCTURED_OPERATOR_MAX_CONDITION",
    "DEFAULT_STRUCTURED_OPERATOR_MAXIMUM_NULLITY",
    "DEFAULT_STRUCTURED_OPERATOR_RANK_RTOL",
    "DEFAULT_STRUCTURED_OPERATOR_RIDGE_RELATIVE",
    "STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM",
    "STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION",
    "STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA",
    "STRUCTURED_OPERATOR_ACTIVE_SUPPORT_POLICY",
    "StructuredOperatorCaptureBatch",
    "StructuredOperatorIdentityBatch",
    "StructuredOperatorRowSelection",
    "bootstrap_structured_operator_executor_",
    "select_structured_operator_rows",
    "structured_operator_coefficient_sha256",
    "structured_operator_site_schema",
    "structured_operator_site_schema_sha256",
]
