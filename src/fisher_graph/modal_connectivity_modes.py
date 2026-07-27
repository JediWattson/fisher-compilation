"""Fisher-weighted connectivity modes for causal multi-port systems.

The objects in this module describe the *boundary transfer* of a causal
subgraph.  They deliberately do not infer that transfer from ablations.  A
caller must first provide a signed direct-Jacobian graph (or another exact
linearization) and reduce it to :class:`CausalBoundaryTransfer`.

Each output port is factored independently over only the input ports that are
causally available to it:

```
M_o = F_o^(1/2) H_o C_o^(1/2) = U Sigma V^T
R_o = Sigma_r^(1/2) V_r^T C_o^(-1/2)
P_o = F_o^(-1/2) U_r Sigma_r^(1/2)
```

The balanced factors ``R_o`` and ``P_o`` are respectively the restriction and
prolongation maps of one higher-level connectivity mode.  Its covariance and
Fisher are both ``diag(Sigma_r)``, so its modal salience is
``Sigma_r ** 2``.  Factoring output ports separately is structural: a factor
for an early output has no storage slot through which a later input could be
read.  Version 1 consumes per-port input covariance and per-output Fisher
blocks.  A prefix containing multiple inputs therefore requires an explicit
block-diagonal covariance assumption; multiple outputs likewise imply a
block-diagonal output-Fisher assumption.  The artifact records both limits,
and the API never infers input independence silently.

Full rank reconstructs ``P_F H P_C`` on the measured PSD supports.  It does
not make a claim about covariance- or Fisher-null directions.  Reduced-rank
weighted error is exactly the squared discarded singular-value mass under the
recorded block-diagonal input-covariance and independent-output-Fisher model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .weighted_jacobian import factor_psd_support


__all__ = [
    "CausalBoundaryTransfer",
    "ConnectivityModeMoments",
    "MessageMoments",
    "ModalBoundaryPort",
    "ModalConnectivityDecomposition",
    "ModalConnectivityFactor",
    "factor_modal_connectivity",
]


_FORMAT_VERSION = 1
_TRANSFER_KIND = "fisher_graph.causal_boundary_transfer"
_MOMENTS_KIND = "fisher_graph.modal_message_moments"
_MODE_MOMENTS_KIND = "fisher_graph.connectivity_mode_moments"
_FACTOR_KIND = "fisher_graph.modal_connectivity_factor"
_DECOMPOSITION_KIND = "fisher_graph.modal_connectivity_decomposition"
_ALGORITHM = "causal_multiport_fisher_weighted_svd"
_TRANSFER_DOMAIN = b"fisher_graph.causal_boundary_transfer.v1\0"
_PORT_DOMAIN = b"fisher_graph.modal_boundary_port.v1\0"
_MOMENTS_DOMAIN = b"fisher_graph.modal_message_moments.v1\0"
_MODE_MOMENTS_DOMAIN = b"fisher_graph.connectivity_mode_moments.v1\0"
_FACTOR_DOMAIN = b"fisher_graph.modal_connectivity_factor.v1\0"
_DECOMPOSITION_DOMAIN = (
    b"fisher_graph.modal_connectivity_decomposition.v1\0"
)
_TENSOR_DOMAIN = b"fisher_graph.modal_connectivity_tensor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_probability_cutoff(value: object, *, label: str) -> float:
    if (
        not isinstance(value, float)
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(f"{label} must lie in (0, 1)")
    return value


def _cpu_float64(
    value: Tensor,
    *,
    label: str,
    ndim: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions")
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result.clone()


def _tensor_sha256(value: Tensor, *, label: str) -> str:
    tensor = _cpu_float64(value, label=label)
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _close(actual: Tensor, expected: Tensor, *, scale: float = 1.0) -> bool:
    return torch.allclose(
        actual,
        expected,
        rtol=2e-10,
        atol=2e-11 * max(scale, 1.0),
    )


def _canonicalize_svd_signs(
    left: Tensor,
    right_h: Tensor,
) -> tuple[Tensor, Tensor]:
    """Remove independent SVD sign ambiguity without changing the product."""

    canonical_left = left.clone()
    canonical_right = right_h.clone()
    for index in range(canonical_left.shape[1]):
        joined = torch.cat(
            (canonical_left[:, index], canonical_right[index]),
            dim=0,
        )
        pivot = int(torch.argmax(joined.abs()).item())
        if float(joined[pivot].item()) < 0.0:
            canonical_left[:, index].neg_()
            canonical_right[index].neg_()
    return canonical_left, canonical_right


def _canonical_basis_from_projector(
    projector: Tensor,
    *,
    dimension: int,
) -> Tensor:
    """Choose a coordinate-ordered basis from a subspace projector."""

    width = projector.shape[0]
    if (
        projector.shape != (width, width)
        or projector.dtype != torch.float64
        or projector.device.type != "cpu"
        or not 0 < dimension <= width
    ):
        raise ValueError("canonical projector basis inputs are invalid")
    tolerance = (
        64.0 * torch.finfo(torch.float64).eps * max(width, 1)
    )
    vectors: list[Tensor] = []
    for coordinate in range(width):
        candidate = projector[:, coordinate].clone()
        for _ in range(2):
            for prior in vectors:
                candidate -= prior * torch.dot(prior, candidate)
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= tolerance:
            continue
        vectors.append(candidate / norm)
        if len(vectors) == dimension:
            break
    if len(vectors) != dimension:
        raise RuntimeError(
            "could not derive the requested deterministic projector basis"
        )
    basis = torch.stack(vectors, dim=1).contiguous()
    for column in range(basis.shape[1]):
        pivot = int(torch.argmax(basis[:, column].abs()).item())
        if float(basis[pivot, column].item()) < 0.0:
            basis[:, column].neg_()
    return basis


def _canonicalize_svd_subspaces(
    weighted: Tensor,
    left: Tensor,
    singular_values: Tensor,
    right_h: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Remove rotations from numerically tied and null SVD subspaces.

    Distinct singular directions remain the SVD directions.  Exactly repeated
    or numerically inseparable positive values are oriented from their
    invariant left projector and then paired through the signed matrix.
    Numerical-null tails use independent coordinate-ordered complements
    because their pairing has no effect on the represented operator.  Finally,
    ordinary column signs are canonicalized.
    """

    rank = singular_values.numel()
    if rank == 0:
        return left.clone(), singular_values.clone(), right_h.clone()
    singular_scale = float(singular_values[0].item())
    relative_tolerance = (
        torch.finfo(torch.float64).eps
        * max(weighted.shape)
        * 512
    )
    absolute_tolerance = singular_scale * relative_tolerance
    right = right_h.T
    left_blocks: list[Tensor] = []
    right_blocks: list[Tensor] = []
    value_blocks: list[Tensor] = []
    start = 0
    while start < rank:
        value = float(singular_values[start].item())
        if value <= absolute_tolerance:
            remaining = rank - start
            left_projector = torch.eye(
                left.shape[0],
                dtype=torch.float64,
            )
            right_projector = torch.eye(
                right.shape[0],
                dtype=torch.float64,
            )
            if start:
                retained_left = torch.cat(left_blocks, dim=1)
                retained_right = torch.cat(right_blocks, dim=1)
                left_projector -= retained_left @ retained_left.T
                right_projector -= retained_right @ retained_right.T
            left_blocks.append(
                _canonical_basis_from_projector(
                    left_projector,
                    dimension=remaining,
                )
            )
            right_blocks.append(
                _canonical_basis_from_projector(
                    right_projector,
                    dimension=remaining,
                )
            )
            value_blocks.append(
                torch.zeros(remaining, dtype=torch.float64)
            )
            break

        stop = start + 1
        while (
            stop < rank
            and abs(float(singular_values[stop].item()) - value)
            <= absolute_tolerance
        ):
            stop += 1
        block_width = stop - start
        if block_width == 1:
            left_blocks.append(left[:, start:stop].clone())
            right_blocks.append(right[:, start:stop].clone())
            value_blocks.append(singular_values[start:stop].clone())
        else:
            source_left = left[:, start:stop]
            canonical_left = _canonical_basis_from_projector(
                source_left @ source_left.T,
                dimension=block_width,
            )
            representative = math.fsum(
                float(item)
                for item in singular_values[start:stop].tolist()
            ) / block_width
            canonical_right = weighted.T @ canonical_left / representative
            if not _close(
                canonical_right.T @ canonical_right,
                torch.eye(block_width, dtype=torch.float64),
            ):
                raise RuntimeError(
                    "canonical tied SVD basis lost right orthonormality"
                )
            left_blocks.append(canonical_left)
            right_blocks.append(canonical_right)
            value_blocks.append(
                torch.full(
                    (block_width,),
                    representative,
                    dtype=torch.float64,
                )
            )
        start = stop
    canonical_left = torch.cat(left_blocks, dim=1).contiguous()
    canonical_values = torch.cat(value_blocks, dim=0).contiguous()
    canonical_right_h = torch.cat(right_blocks, dim=1).T.contiguous()
    canonical_left, canonical_right_h = _canonicalize_svd_signs(
        canonical_left,
        canonical_right_h,
    )
    return canonical_left, canonical_values, canonical_right_h


def _block_diag(values: Sequence[Tensor]) -> Tensor:
    if not values:
        return torch.empty((0, 0), dtype=torch.float64)
    return torch.block_diag(*values)


@dataclass(frozen=True, slots=True)
class ModalBoundaryPort:
    """One typed input or output on a causal subgraph boundary."""

    name: str
    direction: str
    causal_order: int
    width: int
    owner_id: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_name(self.name, label="port name")
        _require_name(self.owner_id, label="port owner_id")
        if self.direction not in {"input", "output"}:
            raise ValueError("port direction must be input or output")
        _require_int(
            self.causal_order,
            label="port causal_order",
            minimum=0,
        )
        _require_int(self.width, label="port width", minimum=1)
        computed = _json_sha256(
            {
                "name": self.name,
                "direction": self.direction,
                "causal_order": self.causal_order,
                "width": self.width,
                "owner_id": self.owner_id,
            },
            domain=_PORT_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="port artifact_sha256",
                )
                != computed
            ):
                raise ValueError("port artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def validate_integrity(self) -> None:
        if (
            _json_sha256(
                {
                    "name": self.name,
                    "direction": self.direction,
                    "causal_order": self.causal_order,
                    "width": self.width,
                    "owner_id": self.owner_id,
                },
                domain=_PORT_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("port artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": self.direction,
            "causal_order": self.causal_order,
            "width": self.width,
            "owner_id": self.owner_id,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class MessageMoments:
    """Authenticated mean, covariance, and activation Fisher for one port."""

    port: ModalBoundaryPort
    source_level_sha256: str
    reduction_id: str
    sample_count: int
    mean: Tensor
    covariance: Tensor
    fisher: Tensor
    mean_sha256: str = ""
    covariance_sha256: str = ""
    fisher_sha256: str = ""
    artifact_sha256: str = ""
    artifact_kind: str = _MOMENTS_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.port, ModalBoundaryPort):
            raise TypeError("port must be a ModalBoundaryPort")
        self.port.validate_integrity()
        _require_sha256(
            self.source_level_sha256,
            label="moments source_level_sha256",
        )
        _require_name(self.reduction_id, label="moments reduction_id")
        _require_int(
            self.sample_count,
            label="moments sample_count",
            minimum=1,
        )
        if (
            self.artifact_kind != _MOMENTS_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("message moments artifact header is invalid")

        mean = _cpu_float64(self.mean, label="moments mean", ndim=1)
        covariance = _cpu_float64(
            self.covariance,
            label="moments covariance",
            ndim=2,
        )
        fisher = _cpu_float64(
            self.fisher,
            label="moments fisher",
            ndim=2,
        )
        expected_square = (self.port.width, self.port.width)
        if mean.shape != (self.port.width,):
            raise ValueError("moments mean width does not match port")
        if covariance.shape != expected_square:
            raise ValueError("moments covariance width does not match port")
        if fisher.shape != expected_square:
            raise ValueError("moments fisher width does not match port")
        covariance = factor_psd_support(covariance).matrix
        fisher = factor_psd_support(fisher).matrix
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "fisher", fisher)

        actual_hashes = {
            "mean_sha256": _tensor_sha256(mean, label="moments mean"),
            "covariance_sha256": _tensor_sha256(
                covariance,
                label="moments covariance",
            ),
            "fisher_sha256": _tensor_sha256(
                fisher,
                label="moments fisher",
            ),
        }
        for name, actual in actual_hashes.items():
            supplied = getattr(self, name)
            if supplied:
                if _require_sha256(supplied, label=name) != actual:
                    raise ValueError(f"{name} does not match tensor")
            else:
                object.__setattr__(self, name, actual)

        computed = _json_sha256(self._hash_payload(), domain=_MOMENTS_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="moments artifact_sha256",
                )
                != computed
            ):
                raise ValueError("message moments artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "port_sha256": self.port.artifact_sha256,
            "source_level_sha256": self.source_level_sha256,
            "reduction_id": self.reduction_id,
            "sample_count": self.sample_count,
            "mean_sha256": self.mean_sha256,
            "covariance_sha256": self.covariance_sha256,
            "fisher_sha256": self.fisher_sha256,
        }

    def validate_integrity(self) -> None:
        self.port.validate_integrity()
        actual = {
            "mean_sha256": _tensor_sha256(self.mean, label="moments mean"),
            "covariance_sha256": _tensor_sha256(
                self.covariance,
                label="moments covariance",
            ),
            "fisher_sha256": _tensor_sha256(
                self.fisher,
                label="moments fisher",
            ),
        }
        if any(getattr(self, name) != value for name, value in actual.items()):
            raise ValueError("message moments tensor hash mismatch")
        if (
            _json_sha256(self._hash_payload(), domain=_MOMENTS_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("message moments artifact hash mismatch")


@dataclass(frozen=True, slots=True)
class CausalBoundaryTransfer:
    """Signed affine transfer with a structurally omitted future prefix."""

    source_level_sha256: str
    input_ports: tuple[ModalBoundaryPort, ...]
    output_ports: tuple[ModalBoundaryPort, ...]
    input_prefixes: tuple[tuple[str, ...], ...]
    transfer_matrices: tuple[Tensor, ...]
    affine_offsets: tuple[Tensor, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _TRANSFER_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_level_sha256,
            label="transfer source_level_sha256",
        )
        if (
            self.artifact_kind != _TRANSFER_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("boundary transfer artifact header is invalid")
        if (
            type(self.input_ports) is not tuple
            or not self.input_ports
            or any(
                not isinstance(port, ModalBoundaryPort)
                for port in self.input_ports
            )
        ):
            raise ValueError("input_ports must be a nonempty port tuple")
        if (
            type(self.output_ports) is not tuple
            or not self.output_ports
            or any(
                not isinstance(port, ModalBoundaryPort)
                for port in self.output_ports
            )
        ):
            raise ValueError("output_ports must be a nonempty port tuple")
        expected_inputs = tuple(
            sorted(
                self.input_ports,
                key=lambda port: (port.causal_order, port.name),
            )
        )
        expected_outputs = tuple(
            sorted(
                self.output_ports,
                key=lambda port: (port.causal_order, port.name),
            )
        )
        if self.input_ports != expected_inputs:
            raise ValueError("input_ports must be in canonical causal order")
        if self.output_ports != expected_outputs:
            raise ValueError("output_ports must be in canonical causal order")
        if any(port.direction != "input" for port in self.input_ports):
            raise ValueError("input_ports contain a non-input port")
        if any(port.direction != "output" for port in self.output_ports):
            raise ValueError("output_ports contain a non-output port")
        all_names = tuple(
            port.name for port in self.input_ports + self.output_ports
        )
        if len(all_names) != len(set(all_names)):
            raise ValueError("boundary port names must be globally unique")
        for port in self.input_ports + self.output_ports:
            port.validate_integrity()

        count = len(self.output_ports)
        if (
            type(self.input_prefixes) is not tuple
            or len(self.input_prefixes) != count
            or type(self.transfer_matrices) is not tuple
            or len(self.transfer_matrices) != count
            or type(self.affine_offsets) is not tuple
            or len(self.affine_offsets) != count
        ):
            raise ValueError(
                "each output requires one prefix, transfer matrix, and offset"
            )
        by_input = {port.name: port for port in self.input_ports}
        matrices: list[Tensor] = []
        offsets: list[Tensor] = []
        for output, prefix, matrix, offset in zip(
            self.output_ports,
            self.input_prefixes,
            self.transfer_matrices,
            self.affine_offsets,
            strict=True,
        ):
            expected_prefix = tuple(
                port.name
                for port in self.input_ports
                if port.causal_order <= output.causal_order
            )
            if prefix != expected_prefix:
                raise ValueError(
                    f"output {output.name} must store exactly its legal "
                    "causal input prefix"
                )
            prefix_width = sum(by_input[name].width for name in prefix)
            canonical_matrix = _cpu_float64(
                matrix,
                label=f"transfer matrix {output.name}",
                ndim=2,
            )
            canonical_offset = _cpu_float64(
                offset,
                label=f"affine offset {output.name}",
                ndim=1,
            )
            if canonical_matrix.shape != (output.width, prefix_width):
                raise ValueError(
                    f"transfer matrix {output.name} has the wrong shape"
                )
            if canonical_offset.shape != (output.width,):
                raise ValueError(
                    f"affine offset {output.name} has the wrong shape"
                )
            matrices.append(canonical_matrix)
            offsets.append(canonical_offset)
        object.__setattr__(self, "transfer_matrices", tuple(matrices))
        object.__setattr__(self, "affine_offsets", tuple(offsets))

        computed = _json_sha256(
            self._hash_payload(),
            domain=_TRANSFER_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="transfer artifact_sha256",
                )
                != computed
            ):
                raise ValueError("boundary transfer artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def input_widths(self) -> dict[str, int]:
        return {port.name: port.width for port in self.input_ports}

    @property
    def output_widths(self) -> dict[str, int]:
        return {port.name: port.width for port in self.output_ports}

    def output_index(self, name: str) -> int:
        for index, port in enumerate(self.output_ports):
            if port.name == name:
                return index
        raise KeyError(name)

    def block(self, output_name: str, input_name: str) -> Tensor:
        self.validate_integrity()
        index = self.output_index(output_name)
        prefix = self.input_prefixes[index]
        if input_name not in prefix:
            raise KeyError(
                f"{input_name} is not in the causal prefix of {output_name}"
            )
        by_input = {port.name: port for port in self.input_ports}
        start = sum(
            by_input[name].width
            for name in prefix[: prefix.index(input_name)]
        )
        width = by_input[input_name].width
        return self.transfer_matrices[index][:, start : start + width].clone()

    def execute(self, inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.validate_integrity()
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping")
        expected = {port.name for port in self.input_ports}
        if set(inputs) != expected:
            raise ValueError("boundary input names do not match transfer")
        leading_shape: tuple[int, ...] | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        canonical: dict[str, Tensor] = {}
        for port in self.input_ports:
            value = inputs[port.name]
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError("boundary inputs must be floating Tensors")
            if value.shape[-1:] != (port.width,):
                raise ValueError(
                    f"boundary input {port.name} has the wrong width"
                )
            if not torch.isfinite(value).all():
                raise ValueError("boundary inputs must be finite")
            current_leading = tuple(value.shape[:-1])
            if leading_shape is None:
                leading_shape = current_leading
                device = value.device
                dtype = value.dtype
            elif (
                current_leading != leading_shape
                or value.device != device
                or value.dtype != dtype
            ):
                raise ValueError(
                    "boundary inputs must share leading shape, device, and dtype"
                )
            canonical[port.name] = value
        assert device is not None
        assert dtype is not None

        outputs: dict[str, Tensor] = {}
        for output, prefix, matrix, offset in zip(
            self.output_ports,
            self.input_prefixes,
            self.transfer_matrices,
            self.affine_offsets,
            strict=True,
        ):
            joined = torch.cat(
                tuple(canonical[name] for name in prefix),
                dim=-1,
            )
            result = joined @ matrix.to(device=device, dtype=dtype).T
            outputs[output.name] = result + offset.to(
                device=device,
                dtype=dtype,
            )
        return outputs

    @property
    def stored_scalar_count(self) -> int:
        return sum(
            matrix.numel() + offset.numel()
            for matrix, offset in zip(
                self.transfer_matrices,
                self.affine_offsets,
                strict=True,
            )
        )

    @property
    def macs_per_row(self) -> int:
        return sum(matrix.numel() for matrix in self.transfer_matrices)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_level_sha256": self.source_level_sha256,
            "input_port_sha256s": tuple(
                port.artifact_sha256 for port in self.input_ports
            ),
            "output_port_sha256s": tuple(
                port.artifact_sha256 for port in self.output_ports
            ),
            "input_prefixes": self.input_prefixes,
            "transfer_matrix_sha256s": tuple(
                _tensor_sha256(value, label="transfer matrix")
                for value in self.transfer_matrices
            ),
            "affine_offset_sha256s": tuple(
                _tensor_sha256(value, label="affine offset")
                for value in self.affine_offsets
            ),
        }

    def validate_integrity(self) -> None:
        for port in self.input_ports + self.output_ports:
            port.validate_integrity()
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_TRANSFER_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("boundary transfer artifact hash mismatch")


@dataclass(frozen=True, slots=True)
class ConnectivityModeMoments:
    """Authenticated moments of retained higher-level coordinates."""

    mode_id: str
    causal_order: int
    source_factor_sha256: str
    reduction_id: str
    sample_count: int
    covariance: Tensor
    fisher: Tensor
    salience: Tensor
    artifact_sha256: str = ""
    artifact_kind: str = _MODE_MOMENTS_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.mode_id, label="mode_id")
        _require_int(
            self.causal_order,
            label="mode causal_order",
            minimum=0,
        )
        _require_sha256(
            self.source_factor_sha256,
            label="mode source_factor_sha256",
        )
        _require_name(self.reduction_id, label="mode reduction_id")
        _require_int(
            self.sample_count,
            label="mode sample_count",
            minimum=1,
        )
        if (
            self.artifact_kind != _MODE_MOMENTS_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("connectivity mode moments header is invalid")
        covariance = _cpu_float64(
            self.covariance,
            label="mode covariance",
            ndim=2,
        )
        fisher = _cpu_float64(
            self.fisher,
            label="mode fisher",
            ndim=2,
        )
        salience = _cpu_float64(
            self.salience,
            label="mode salience",
            ndim=1,
        )
        rank = salience.numel()
        if covariance.shape != (rank, rank) or fisher.shape != (rank, rank):
            raise ValueError("mode moment shapes do not match salience")
        if not _close(covariance, fisher):
            raise ValueError("balanced mode covariance and Fisher must match")
        if not _close(torch.diag(covariance).square(), salience):
            raise ValueError("mode salience must equal covariance diagonal squared")
        if rank and (
            not _close(covariance, torch.diag(torch.diag(covariance)))
            or (torch.diag(covariance) < 0).any()
        ):
            raise ValueError("mode moments must be nonnegative diagonal matrices")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "fisher", fisher)
        object.__setattr__(self, "salience", salience)
        computed = _json_sha256(
            self._hash_payload(),
            domain=_MODE_MOMENTS_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="mode moments artifact_sha256",
                )
                != computed
            ):
                raise ValueError("connectivity mode moments hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "mode_id": self.mode_id,
            "causal_order": self.causal_order,
            "source_factor_sha256": self.source_factor_sha256,
            "reduction_id": self.reduction_id,
            "sample_count": self.sample_count,
            "covariance_sha256": _tensor_sha256(
                self.covariance,
                label="mode covariance",
            ),
            "fisher_sha256": _tensor_sha256(
                self.fisher,
                label="mode fisher",
            ),
            "salience_sha256": _tensor_sha256(
                self.salience,
                label="mode salience",
            ),
        }

    def validate_integrity(self) -> None:
        diagonal = torch.diag(self.covariance)
        if (
            self.covariance.shape != self.fisher.shape
            or self.covariance.shape
            != (self.salience.numel(), self.salience.numel())
            or not _close(self.covariance, self.fisher)
            or not _close(diagonal.square(), self.salience)
            or not _close(self.covariance, torch.diag(diagonal))
            or (diagonal < 0).any()
        ):
            raise ValueError("connectivity mode moments are inconsistent")
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_MODE_MOMENTS_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("connectivity mode moments hash mismatch")

    def bind_port(
        self,
        port: ModalBoundaryPort,
        *,
        source_level_sha256: str,
    ) -> MessageMoments:
        """Bind these modal statistics to a concrete graph boundary port."""

        self.validate_integrity()
        if not isinstance(port, ModalBoundaryPort):
            raise TypeError("mode moments port must be a ModalBoundaryPort")
        if (
            port.width != self.salience.numel()
            or port.causal_order != self.causal_order
        ):
            raise ValueError("mode moments do not match boundary port")
        return MessageMoments(
            port=port,
            source_level_sha256=source_level_sha256,
            reduction_id=self.reduction_id,
            sample_count=self.sample_count,
            mean=torch.zeros(port.width, dtype=torch.float64),
            covariance=self.covariance,
            fisher=self.fisher,
        )


@dataclass(frozen=True, slots=True)
class ModalConnectivityFactor:
    """One balanced Fisher-weighted factor for one causal output prefix."""

    source_transfer_sha256: str
    output_port: ModalBoundaryPort
    input_ports: tuple[ModalBoundaryPort, ...]
    input_moment_sha256s: tuple[str, ...]
    output_moment_sha256: str
    reduction_id: str
    sample_count: int
    retained_rank: int
    singular_values: Tensor
    weighted_left_vectors: Tensor
    weighted_right_vectors: Tensor
    input_inverse_square_roots: tuple[Tensor, ...]
    output_inverse_square_root: Tensor
    restriction: Tensor
    prolongation: Tensor
    input_mean: Tensor
    output_mean: Tensor
    affine_offset: Tensor
    source_block_energy: Tensor
    input_support_ranks: tuple[int, ...]
    output_support_rank: int
    singular_tolerance: float
    artifact_sha256: str = ""
    artifact_kind: str = _FACTOR_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_transfer_sha256,
            label="factor source_transfer_sha256",
        )
        if not isinstance(self.output_port, ModalBoundaryPort):
            raise TypeError("output_port must be a ModalBoundaryPort")
        if self.output_port.direction != "output":
            raise ValueError("factor output_port must be an output")
        if (
            type(self.input_ports) is not tuple
            or not self.input_ports
            or any(
                not isinstance(port, ModalBoundaryPort)
                for port in self.input_ports
            )
            or any(port.direction != "input" for port in self.input_ports)
        ):
            raise ValueError("factor input_ports must be a nonempty input tuple")
        if any(
            port.causal_order > self.output_port.causal_order
            for port in self.input_ports
        ):
            raise ValueError("factor contains an illegal future input port")
        if len({port.name for port in self.input_ports}) != len(
            self.input_ports
        ):
            raise ValueError("factor input port names must be unique")
        for port in self.input_ports + (self.output_port,):
            port.validate_integrity()
        if (
            type(self.input_moment_sha256s) is not tuple
            or len(self.input_moment_sha256s) != len(self.input_ports)
        ):
            raise ValueError("input moment digests must match input ports")
        for digest in self.input_moment_sha256s:
            _require_sha256(digest, label="input moment digest")
        _require_sha256(
            self.output_moment_sha256,
            label="output moment digest",
        )
        _require_name(self.reduction_id, label="factor reduction_id")
        _require_int(
            self.sample_count,
            label="factor sample_count",
            minimum=1,
        )
        if (
            self.artifact_kind != _FACTOR_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("connectivity factor artifact header is invalid")

        prefix_width = sum(port.width for port in self.input_ports)
        output_width = self.output_port.width
        spectrum_rank = min(prefix_width, output_width)
        _require_int(
            self.retained_rank,
            label="retained_rank",
            minimum=0,
        )
        if self.retained_rank > spectrum_rank:
            raise ValueError("retained_rank exceeds the output-prefix spectrum")
        _require_int(
            self.output_support_rank,
            label="output_support_rank",
            minimum=0,
        )
        if self.output_support_rank > output_width:
            raise ValueError("output_support_rank exceeds output width")
        if (
            type(self.input_support_ranks) is not tuple
            or len(self.input_support_ranks) != len(self.input_ports)
            or any(
                type(rank) is not int
                or not 0 <= rank <= port.width
                for rank, port in zip(
                    self.input_support_ranks,
                    self.input_ports,
                    strict=True,
                )
            )
        ):
            raise ValueError("input_support_ranks do not match input ports")
        if (
            not isinstance(self.singular_tolerance, float)
            or not math.isfinite(self.singular_tolerance)
            or self.singular_tolerance < 0.0
        ):
            raise ValueError("singular_tolerance must be finite and nonnegative")

        tensors = {
            "singular_values": (
                self.singular_values,
                (spectrum_rank,),
            ),
            "weighted_left_vectors": (
                self.weighted_left_vectors,
                (output_width, spectrum_rank),
            ),
            "weighted_right_vectors": (
                self.weighted_right_vectors,
                (prefix_width, spectrum_rank),
            ),
            "output_inverse_square_root": (
                self.output_inverse_square_root,
                (output_width, output_width),
            ),
            "restriction": (
                self.restriction,
                (self.retained_rank, prefix_width),
            ),
            "prolongation": (
                self.prolongation,
                (output_width, self.retained_rank),
            ),
            "input_mean": (self.input_mean, (prefix_width,)),
            "output_mean": (self.output_mean, (output_width,)),
            "affine_offset": (self.affine_offset, (output_width,)),
            "source_block_energy": (
                self.source_block_energy,
                (len(self.input_ports),),
            ),
        }
        canonical_tensors: dict[str, Tensor] = {}
        for name, (value, shape) in tensors.items():
            canonical = _cpu_float64(value, label=name)
            if canonical.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            canonical_tensors[name] = canonical
            object.__setattr__(self, name, canonical)
        if (
            type(self.input_inverse_square_roots) is not tuple
            or len(self.input_inverse_square_roots) != len(self.input_ports)
        ):
            raise ValueError(
                "input inverse square roots must match input ports"
            )
        inverse_roots: list[Tensor] = []
        for port, value in zip(
            self.input_ports,
            self.input_inverse_square_roots,
            strict=True,
        ):
            canonical = _cpu_float64(
                value,
                label=f"input inverse square root {port.name}",
                ndim=2,
            )
            if canonical.shape != (port.width, port.width):
                raise ValueError(
                    "input inverse square root width does not match port"
                )
            if not _close(
                canonical,
                canonical.T,
                scale=float(canonical.abs().max().item()),
            ):
                raise ValueError("input inverse square roots must be symmetric")
            inverse_roots.append(canonical)
        object.__setattr__(
            self,
            "input_inverse_square_roots",
            tuple(inverse_roots),
        )

        singular_values = canonical_tensors["singular_values"]
        if (singular_values < 0).any():
            raise ValueError("singular values cannot be negative")
        if singular_values.numel() > 1 and (
            singular_values[1:] > singular_values[:-1]
        ).any():
            raise ValueError("singular values must be sorted descending")
        if (canonical_tensors["source_block_energy"] < 0).any():
            raise ValueError("source block energy cannot be negative")
        identity = torch.eye(spectrum_rank, dtype=torch.float64)
        if not _close(
            canonical_tensors["weighted_left_vectors"].T
            @ canonical_tensors["weighted_left_vectors"],
            identity,
        ):
            raise ValueError("weighted left vectors must be orthonormal")
        if not _close(
            canonical_tensors["weighted_right_vectors"].T
            @ canonical_tensors["weighted_right_vectors"],
            identity,
        ):
            raise ValueError("weighted right vectors must be orthonormal")
        output_inverse = canonical_tensors["output_inverse_square_root"]
        if not _close(
            output_inverse,
            output_inverse.T,
            scale=float(output_inverse.abs().max().item()),
        ):
            raise ValueError("output inverse square root must be symmetric")

        root = singular_values[: self.retained_rank].sqrt()
        expected_prolongation = output_inverse @ (
            canonical_tensors["weighted_left_vectors"][
                :, : self.retained_rank
            ]
            * root.unsqueeze(0)
        )
        prefix_inverse = _block_diag(inverse_roots)
        expected_restriction = (
            root.unsqueeze(1)
            * canonical_tensors["weighted_right_vectors"][
                :, : self.retained_rank
            ].T
        ) @ prefix_inverse
        scale = max(
            float(expected_prolongation.abs().max().item())
            if expected_prolongation.numel()
            else 0.0,
            float(expected_restriction.abs().max().item())
            if expected_restriction.numel()
            else 0.0,
            1.0,
        )
        if not _close(
            canonical_tensors["prolongation"],
            expected_prolongation,
            scale=scale,
        ) or not _close(
            canonical_tensors["restriction"],
            expected_restriction,
            scale=scale,
        ):
            raise ValueError(
                "restriction/prolongation do not match the balanced SVD"
            )

        computed = _json_sha256(self._hash_payload(), domain=_FACTOR_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="factor artifact_sha256",
                )
                != computed
            ):
                raise ValueError("connectivity factor artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def prefix_width(self) -> int:
        return sum(port.width for port in self.input_ports)

    @property
    def spectrum_rank(self) -> int:
        return self.singular_values.numel()

    @property
    def effective_rank(self) -> int:
        return int(
            (self.singular_values > self.singular_tolerance).sum().item()
        )

    @property
    def total_weighted_energy(self) -> float:
        return float(self.singular_values.square().sum().item())

    @property
    def retained_weighted_energy(self) -> float:
        return float(
            self.singular_values[: self.retained_rank].square().sum().item()
        )

    @property
    def discarded_weighted_energy(self) -> float:
        return float(
            self.singular_values[self.retained_rank :].square().sum().item()
        )

    @property
    def reconstructed_matrix(self) -> Tensor:
        return self.prolongation @ self.restriction

    @property
    def mode_moments(self) -> ConnectivityModeMoments:
        self.validate_integrity()
        diagonal = self.singular_values[: self.retained_rank]
        return ConnectivityModeMoments(
            mode_id=f"connectivity.{self.output_port.name}",
            causal_order=self.output_port.causal_order,
            source_factor_sha256=self.artifact_sha256,
            reduction_id=self.reduction_id,
            sample_count=self.sample_count,
            covariance=torch.diag(diagonal),
            fisher=torch.diag(diagonal),
            salience=diagonal.square(),
        )

    @property
    def stored_scalar_count(self) -> int:
        return (
            self.restriction.numel()
            + self.prolongation.numel()
            + self.input_mean.numel()
            + self.output_mean.numel()
        )

    @property
    def macs_per_row(self) -> int:
        return self.restriction.numel() + self.prolongation.numel()

    def execute(self, inputs: Sequence[Tensor]) -> Tensor:
        self.validate_integrity()
        if len(inputs) != len(self.input_ports):
            raise ValueError("factor inputs do not match the causal prefix")
        leading_shape: tuple[int, ...] | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        values: list[Tensor] = []
        for port, value in zip(self.input_ports, inputs, strict=True):
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError("factor inputs must be floating Tensors")
            if value.shape[-1:] != (port.width,):
                raise ValueError("factor input width does not match port")
            if not torch.isfinite(value).all():
                raise ValueError("factor inputs must be finite")
            if leading_shape is None:
                leading_shape = tuple(value.shape[:-1])
                device = value.device
                dtype = value.dtype
            elif (
                tuple(value.shape[:-1]) != leading_shape
                or value.device != device
                or value.dtype != dtype
            ):
                raise ValueError(
                    "factor inputs must share leading shape, device, and dtype"
                )
            values.append(value)
        assert device is not None
        assert dtype is not None
        joined = torch.cat(tuple(values), dim=-1)
        centered = joined - self.input_mean.to(device=device, dtype=dtype)
        modes = centered @ self.restriction.to(
            device=device,
            dtype=dtype,
        ).T
        return (
            modes
            @ self.prolongation.to(device=device, dtype=dtype).T
            + self.output_mean.to(device=device, dtype=dtype)
        )

    def truncate(self, retained_rank: int) -> ModalConnectivityFactor:
        self.validate_integrity()
        if type(retained_rank) is not int or not (
            0 <= retained_rank <= self.spectrum_rank
        ):
            raise ValueError("retained_rank is outside the factor spectrum")
        root = self.singular_values[:retained_rank].sqrt()
        prolongation = self.output_inverse_square_root @ (
            self.weighted_left_vectors[:, :retained_rank]
            * root.unsqueeze(0)
        )
        prefix_inverse = _block_diag(self.input_inverse_square_roots)
        restriction = (
            root.unsqueeze(1)
            * self.weighted_right_vectors[:, :retained_rank].T
        ) @ prefix_inverse
        return ModalConnectivityFactor(
            source_transfer_sha256=self.source_transfer_sha256,
            output_port=self.output_port,
            input_ports=self.input_ports,
            input_moment_sha256s=self.input_moment_sha256s,
            output_moment_sha256=self.output_moment_sha256,
            reduction_id=self.reduction_id,
            sample_count=self.sample_count,
            retained_rank=retained_rank,
            singular_values=self.singular_values,
            weighted_left_vectors=self.weighted_left_vectors,
            weighted_right_vectors=self.weighted_right_vectors,
            input_inverse_square_roots=self.input_inverse_square_roots,
            output_inverse_square_root=self.output_inverse_square_root,
            restriction=restriction,
            prolongation=prolongation,
            input_mean=self.input_mean,
            output_mean=self.output_mean,
            affine_offset=self.affine_offset,
            source_block_energy=self.source_block_energy,
            input_support_ranks=self.input_support_ranks,
            output_support_rank=self.output_support_rank,
            singular_tolerance=self.singular_tolerance,
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_transfer_sha256": self.source_transfer_sha256,
            "output_port_sha256": self.output_port.artifact_sha256,
            "input_port_sha256s": tuple(
                port.artifact_sha256 for port in self.input_ports
            ),
            "input_moment_sha256s": self.input_moment_sha256s,
            "output_moment_sha256": self.output_moment_sha256,
            "reduction_id": self.reduction_id,
            "sample_count": self.sample_count,
            "retained_rank": self.retained_rank,
            "singular_values_sha256": _tensor_sha256(
                self.singular_values,
                label="singular_values",
            ),
            "weighted_left_vectors_sha256": _tensor_sha256(
                self.weighted_left_vectors,
                label="weighted_left_vectors",
            ),
            "weighted_right_vectors_sha256": _tensor_sha256(
                self.weighted_right_vectors,
                label="weighted_right_vectors",
            ),
            "input_inverse_square_root_sha256s": tuple(
                _tensor_sha256(value, label="input_inverse_square_root")
                for value in self.input_inverse_square_roots
            ),
            "output_inverse_square_root_sha256": _tensor_sha256(
                self.output_inverse_square_root,
                label="output_inverse_square_root",
            ),
            "restriction_sha256": _tensor_sha256(
                self.restriction,
                label="restriction",
            ),
            "prolongation_sha256": _tensor_sha256(
                self.prolongation,
                label="prolongation",
            ),
            "input_mean_sha256": _tensor_sha256(
                self.input_mean,
                label="input_mean",
            ),
            "output_mean_sha256": _tensor_sha256(
                self.output_mean,
                label="output_mean",
            ),
            "affine_offset_sha256": _tensor_sha256(
                self.affine_offset,
                label="affine_offset",
            ),
            "source_block_energy_sha256": _tensor_sha256(
                self.source_block_energy,
                label="source_block_energy",
            ),
            "input_support_ranks": self.input_support_ranks,
            "output_support_rank": self.output_support_rank,
            "singular_tolerance": self.singular_tolerance,
        }

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._hash_payload(), domain=_FACTOR_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("connectivity factor artifact hash mismatch")


def _shared_measurement_lineage(
    moments: Sequence[MessageMoments],
) -> tuple[str, int]:
    lineage = {
        (moment.reduction_id, moment.sample_count)
        for moment in moments
    }
    if len(lineage) != 1:
        raise ValueError(
            "all connectivity moments must share reduction_id and "
            "sample_count"
        )
    return next(iter(lineage))


def _validate_factor_against_measurements(
    factor: ModalConnectivityFactor,
    *,
    source_matrix: Tensor,
    affine_offset: Tensor,
    input_moments: tuple[MessageMoments, ...],
    output_moment: MessageMoments,
    relative_eigenvalue_cutoff: float,
    relative_singular_value_cutoff: float,
) -> None:
    """Recompute the measured operator that an authenticated factor names."""

    reduction_id, sample_count = _shared_measurement_lineage(
        input_moments + (output_moment,)
    )
    if (
        factor.reduction_id != reduction_id
        or factor.sample_count != sample_count
    ):
        raise ValueError("factor measurement lineage is stale")
    covariance_factors = tuple(
        factor_psd_support(
            moment.covariance,
            relative_cutoff=relative_eigenvalue_cutoff,
        )
        for moment in input_moments
    )
    output_psd = factor_psd_support(
        output_moment.fisher,
        relative_cutoff=relative_eigenvalue_cutoff,
    )
    if factor.input_support_ranks != tuple(
        value.support_rank for value in covariance_factors
    ) or factor.output_support_rank != output_psd.support_rank:
        raise ValueError("factor PSD support ranks do not match moments")
    for actual, expected in zip(
        factor.input_inverse_square_roots,
        (
            value.inverse_square_root
            for value in covariance_factors
        ),
        strict=True,
    ):
        if not _close(
            actual,
            expected,
            scale=float(expected.abs().max().item()),
        ):
            raise ValueError(
                "factor input inverse root does not match covariance"
            )
    if not _close(
        factor.output_inverse_square_root,
        output_psd.inverse_square_root,
        scale=float(output_psd.inverse_square_root.abs().max().item()),
    ):
        raise ValueError("factor output inverse root does not match Fisher")

    covariance_root = _block_diag(
        tuple(value.square_root for value in covariance_factors)
    )
    weighted = output_psd.square_root @ source_matrix @ covariance_root
    represented = (
        factor.weighted_left_vectors
        @ torch.diag(factor.singular_values)
        @ factor.weighted_right_vectors.T
    )
    if not _close(
        represented,
        weighted,
        scale=float(weighted.abs().max().item()),
    ):
        raise ValueError(
            "factor weighted SVD does not reconstruct linked transfer and "
            "moments"
        )

    expected_input_mean = torch.cat(
        tuple(moment.mean for moment in input_moments),
        dim=0,
    )
    expected_output_mean = affine_offset + source_matrix @ expected_input_mean
    if (
        not _close(factor.input_mean, expected_input_mean)
        or not _close(factor.output_mean, expected_output_mean)
        or not _close(factor.affine_offset, affine_offset)
    ):
        raise ValueError("factor affine means do not match linked transfer")

    expected_block_energies: list[Tensor] = []
    start = 0
    for moment in input_moments:
        stop = start + moment.port.width
        expected_block_energies.append(
            weighted[:, start:stop].square().sum()
        )
        start = stop
    if not _close(
        factor.source_block_energy,
        torch.stack(expected_block_energies),
        scale=float(weighted.square().sum().item()),
    ):
        raise ValueError("factor source-block energy does not match transfer")

    singular_scale = (
        float(factor.singular_values[0].item())
        if factor.singular_values.numel()
        else 0.0
    )
    numerical_relative = (
        torch.finfo(torch.float64).eps
        * max(weighted.shape)
        * 128
    )
    expected_tolerance = singular_scale * max(
        relative_singular_value_cutoff,
        numerical_relative,
    )
    if not math.isclose(
        factor.singular_tolerance,
        expected_tolerance,
        rel_tol=2e-12,
        abs_tol=2e-15 * max(singular_scale, 1.0),
    ):
        raise ValueError("factor singular tolerance is stale")


@dataclass(frozen=True, slots=True)
class ModalConnectivityDecomposition:
    """All independently causal output factors for one boundary transfer."""

    source_transfer: CausalBoundaryTransfer
    input_moments: tuple[MessageMoments, ...]
    output_moments: tuple[MessageMoments, ...]
    factors: tuple[ModalConnectivityFactor, ...]
    relative_eigenvalue_cutoff: float
    relative_singular_value_cutoff: float
    assumes_block_diagonal_input_covariance: bool
    assumes_block_diagonal_output_fisher: bool
    artifact_sha256: str = ""
    artifact_kind: str = _DECOMPOSITION_KIND
    format_version: int = _FORMAT_VERSION
    algorithm: str = _ALGORITHM
    algorithm_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.source_transfer, CausalBoundaryTransfer):
            raise TypeError("source_transfer must be a CausalBoundaryTransfer")
        self.source_transfer.validate_integrity()
        if (
            type(self.input_moments) is not tuple
            or type(self.output_moments) is not tuple
            or type(self.factors) is not tuple
        ):
            raise TypeError("moments and factors must be tuples")
        if len(self.input_moments) != len(self.source_transfer.input_ports):
            raise ValueError("input moments do not match input ports")
        if len(self.output_moments) != len(self.source_transfer.output_ports):
            raise ValueError("output moments do not match output ports")
        if len(self.factors) != len(self.source_transfer.output_ports):
            raise ValueError("factors do not match output ports")
        if (
            type(self.assumes_block_diagonal_input_covariance) is not bool
            or type(self.assumes_block_diagonal_output_fisher) is not bool
        ):
            raise TypeError("connectivity metric assumptions must be booleans")
        expected_input_assumption = any(
            len(prefix) > 1
            for prefix in self.source_transfer.input_prefixes
        )
        expected_output_assumption = (
            len(self.source_transfer.output_ports) > 1
        )
        if (
            self.assumes_block_diagonal_input_covariance
            != expected_input_assumption
            or self.assumes_block_diagonal_output_fisher
            != expected_output_assumption
        ):
            raise ValueError(
                "connectivity metric assumptions do not match the boundary"
            )
        for moment, port in zip(
            self.input_moments,
            self.source_transfer.input_ports,
            strict=True,
        ):
            if not isinstance(moment, MessageMoments):
                raise TypeError("input moments must be MessageMoments")
            moment.validate_integrity()
            if moment.port.artifact_sha256 != port.artifact_sha256:
                raise ValueError("input moment port does not match transfer")
            if (
                moment.source_level_sha256
                != self.source_transfer.source_level_sha256
            ):
                raise ValueError(
                    "input moment source level does not match transfer"
                )
        for moment, port in zip(
            self.output_moments,
            self.source_transfer.output_ports,
            strict=True,
        ):
            if not isinstance(moment, MessageMoments):
                raise TypeError("output moments must be MessageMoments")
            moment.validate_integrity()
            if moment.port.artifact_sha256 != port.artifact_sha256:
                raise ValueError("output moment port does not match transfer")
            if (
                moment.source_level_sha256
                != self.source_transfer.source_level_sha256
            ):
                raise ValueError(
                    "output moment source level does not match transfer"
                )
        input_moments_by_name = {
            moment.port.name: moment for moment in self.input_moments
        }
        output_moments_by_name = {
            moment.port.name: moment for moment in self.output_moments
        }
        _require_probability_cutoff(
            self.relative_eigenvalue_cutoff,
            label="relative_eigenvalue_cutoff",
        )
        _require_probability_cutoff(
            self.relative_singular_value_cutoff,
            label="relative_singular_value_cutoff",
        )
        reduction_id, sample_count = _shared_measurement_lineage(
            self.input_moments + self.output_moments
        )
        for factor, output, prefix in zip(
            self.factors,
            self.source_transfer.output_ports,
            self.source_transfer.input_prefixes,
            strict=True,
        ):
            if not isinstance(factor, ModalConnectivityFactor):
                raise TypeError("factors must be ModalConnectivityFactor values")
            factor.validate_integrity()
            if factor.source_transfer_sha256 != self.source_transfer.artifact_sha256:
                raise ValueError("factor source transfer digest is stale")
            if factor.output_port.artifact_sha256 != output.artifact_sha256:
                raise ValueError("factor output does not match transfer")
            if tuple(port.name for port in factor.input_ports) != prefix:
                raise ValueError("factor causal prefix does not match transfer")
            expected_input_moments = tuple(
                input_moments_by_name[name].artifact_sha256
                for name in prefix
            )
            if factor.input_moment_sha256s != expected_input_moments:
                raise ValueError(
                    "factor input moments do not match decomposition"
                )
            if (
                factor.output_moment_sha256
                != output_moments_by_name[
                    output.name
                ].artifact_sha256
            ):
                raise ValueError(
                    "factor output moment does not match decomposition"
                )
            if (
                factor.reduction_id != reduction_id
                or factor.sample_count != sample_count
            ):
                raise ValueError(
                    "factor lineage does not match decomposition moments"
                )
            prefix_moments = tuple(
                input_moments_by_name[name] for name in prefix
            )
            output_index = self.source_transfer.output_ports.index(output)
            _validate_factor_against_measurements(
                factor,
                source_matrix=(
                    self.source_transfer.transfer_matrices[output_index]
                ),
                affine_offset=(
                    self.source_transfer.affine_offsets[output_index]
                ),
                input_moments=prefix_moments,
                output_moment=output_moments_by_name[output.name],
                relative_eigenvalue_cutoff=(
                    self.relative_eigenvalue_cutoff
                ),
                relative_singular_value_cutoff=(
                    self.relative_singular_value_cutoff
                ),
            )
        if (
            self.artifact_kind != _DECOMPOSITION_KIND
            or self.format_version != _FORMAT_VERSION
            or self.algorithm != _ALGORITHM
            or self.algorithm_version != 1
        ):
            raise ValueError("connectivity decomposition header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_DECOMPOSITION_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="decomposition artifact_sha256",
                )
                != computed
            ):
                raise ValueError("connectivity decomposition hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def total_weighted_energy(self) -> float:
        return sum(factor.total_weighted_energy for factor in self.factors)

    @property
    def retained_weighted_energy(self) -> float:
        return sum(factor.retained_weighted_energy for factor in self.factors)

    @property
    def discarded_weighted_energy(self) -> float:
        return sum(factor.discarded_weighted_energy for factor in self.factors)

    @property
    def mode_moments(self) -> tuple[ConnectivityModeMoments, ...]:
        return tuple(factor.mode_moments for factor in self.factors)

    @property
    def candidate_stored_scalar_count(self) -> int:
        return sum(factor.stored_scalar_count for factor in self.factors)

    @property
    def candidate_macs_per_row(self) -> int:
        return sum(factor.macs_per_row for factor in self.factors)

    def execute_candidate(
        self,
        inputs: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        self.validate_integrity()
        expected = {port.name for port in self.source_transfer.input_ports}
        if set(inputs) != expected:
            raise ValueError("candidate inputs do not match transfer")
        return {
            factor.output_port.name: factor.execute(
                tuple(inputs[port.name] for port in factor.input_ports)
            )
            for factor in self.factors
        }

    def candidate_transfer(self) -> CausalBoundaryTransfer:
        self.validate_integrity()
        matrices = tuple(
            factor.reconstructed_matrix for factor in self.factors
        )
        offsets = tuple(
            factor.output_mean
            - factor.reconstructed_matrix @ factor.input_mean
            for factor in self.factors
        )
        return CausalBoundaryTransfer(
            source_level_sha256=self.artifact_sha256,
            input_ports=self.source_transfer.input_ports,
            output_ports=self.source_transfer.output_ports,
            input_prefixes=self.source_transfer.input_prefixes,
            transfer_matrices=matrices,
            affine_offsets=offsets,
        )

    def truncate(
        self,
        retained_ranks: int | Sequence[int] | Mapping[str, int],
    ) -> ModalConnectivityDecomposition:
        self.validate_integrity()
        ranks = _resolve_retained_ranks(
            retained_ranks,
            outputs=self.source_transfer.output_ports,
            spectrum_ranks=tuple(
                factor.spectrum_rank for factor in self.factors
            ),
        )
        return ModalConnectivityDecomposition(
            source_transfer=self.source_transfer,
            input_moments=self.input_moments,
            output_moments=self.output_moments,
            factors=tuple(
                factor.truncate(rank)
                for factor, rank in zip(self.factors, ranks, strict=True)
            ),
            relative_eigenvalue_cutoff=self.relative_eigenvalue_cutoff,
            relative_singular_value_cutoff=(
                self.relative_singular_value_cutoff
            ),
            assumes_block_diagonal_input_covariance=(
                self.assumes_block_diagonal_input_covariance
            ),
            assumes_block_diagonal_output_fisher=(
                self.assumes_block_diagonal_output_fisher
            ),
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "source_transfer_sha256": self.source_transfer.artifact_sha256,
            "input_moment_sha256s": tuple(
                moment.artifact_sha256 for moment in self.input_moments
            ),
            "output_moment_sha256s": tuple(
                moment.artifact_sha256 for moment in self.output_moments
            ),
            "factor_sha256s": tuple(
                factor.artifact_sha256 for factor in self.factors
            ),
            "relative_eigenvalue_cutoff": self.relative_eigenvalue_cutoff,
            "relative_singular_value_cutoff": (
                self.relative_singular_value_cutoff
            ),
            "assumes_block_diagonal_input_covariance": (
                self.assumes_block_diagonal_input_covariance
            ),
            "assumes_block_diagonal_output_fisher": (
                self.assumes_block_diagonal_output_fisher
            ),
            "total_weighted_energy": self.total_weighted_energy,
            "retained_weighted_energy": self.retained_weighted_energy,
            "discarded_weighted_energy": self.discarded_weighted_energy,
        }

    def validate_integrity(self) -> None:
        self.__post_init__()


def _resolve_retained_ranks(
    retained_ranks: int | Sequence[int] | Mapping[str, int] | None,
    *,
    outputs: tuple[ModalBoundaryPort, ...],
    spectrum_ranks: tuple[int, ...],
) -> tuple[int, ...]:
    if retained_ranks is None:
        ranks = spectrum_ranks
    elif type(retained_ranks) is int:
        ranks = (retained_ranks,) * len(outputs)
    elif isinstance(retained_ranks, Mapping):
        if set(retained_ranks) != {port.name for port in outputs}:
            raise ValueError(
                "retained rank mapping must name every output exactly once"
            )
        ranks = tuple(retained_ranks[port.name] for port in outputs)
    elif isinstance(retained_ranks, Sequence) and not isinstance(
        retained_ranks,
        (str, bytes),
    ):
        ranks = tuple(retained_ranks)
        if len(ranks) != len(outputs):
            raise ValueError("retained ranks must match output ports")
    else:
        raise TypeError(
            "retained_ranks must be an integer, mapping, sequence, or None"
        )
    for port, rank, maximum in zip(
        outputs,
        ranks,
        spectrum_ranks,
        strict=True,
    ):
        if type(rank) is not int or not 0 <= rank <= maximum:
            raise ValueError(
                f"retained rank for {port.name} must lie in [0, {maximum}]"
            )
    return ranks


def factor_modal_connectivity(
    transfer: CausalBoundaryTransfer,
    input_moments: Sequence[MessageMoments],
    output_moments: Sequence[MessageMoments],
    *,
    retained_ranks: int | Sequence[int] | Mapping[str, int] | None = None,
    relative_eigenvalue_cutoff: float = 1e-12,
    relative_singular_value_cutoff: float = 1e-12,
    assume_block_diagonal_input_covariance: bool = False,
) -> ModalConnectivityDecomposition:
    """Factor one exact causal multi-port boundary transfer.

    The v1 input covariance model is block-local by boundary input port.
    Prefixes with multiple ports are rejected unless the caller explicitly
    declares their cross-covariance to be zero.  Output ports are factored
    independently, so a multi-output decomposition also records a
    block-diagonal output-Fisher limitation.
    """

    if not isinstance(transfer, CausalBoundaryTransfer):
        raise TypeError("transfer must be a CausalBoundaryTransfer")
    transfer.validate_integrity()
    eigen_cutoff = _require_probability_cutoff(
        relative_eigenvalue_cutoff,
        label="relative_eigenvalue_cutoff",
    )
    singular_cutoff = _require_probability_cutoff(
        relative_singular_value_cutoff,
        label="relative_singular_value_cutoff",
    )
    if type(assume_block_diagonal_input_covariance) is not bool:
        raise TypeError(
            "assume_block_diagonal_input_covariance must be a boolean"
        )
    needs_input_independence = any(
        len(prefix) > 1 for prefix in transfer.input_prefixes
    )
    if (
        needs_input_independence
        and not assume_block_diagonal_input_covariance
    ):
        raise ValueError(
            "multi-port prefixes require measured joint covariance; v1 only "
            "supports an explicit block-diagonal input covariance assumption"
        )
    input_tuple = tuple(input_moments)
    output_tuple = tuple(output_moments)
    if len(input_tuple) != len(transfer.input_ports):
        raise ValueError("input moments do not match input ports")
    if len(output_tuple) != len(transfer.output_ports):
        raise ValueError("output moments do not match output ports")
    input_by_name: dict[str, MessageMoments] = {}
    output_by_name: dict[str, MessageMoments] = {}
    for moment, port in zip(
        input_tuple,
        transfer.input_ports,
        strict=True,
    ):
        if not isinstance(moment, MessageMoments):
            raise TypeError("input moments must be MessageMoments")
        moment.validate_integrity()
        if moment.port.artifact_sha256 != port.artifact_sha256:
            raise ValueError("input moment port does not match transfer")
        input_by_name[port.name] = moment
    for moment, port in zip(
        output_tuple,
        transfer.output_ports,
        strict=True,
    ):
        if not isinstance(moment, MessageMoments):
            raise TypeError("output moments must be MessageMoments")
        moment.validate_integrity()
        if moment.port.artifact_sha256 != port.artifact_sha256:
            raise ValueError("output moment port does not match transfer")
        output_by_name[port.name] = moment
    reduction_id, sample_count = _shared_measurement_lineage(
        input_tuple + output_tuple
    )

    spectrum_ranks = tuple(
        min(
            output.width,
            sum(
                input_by_name[name].port.width
                for name in prefix
            ),
        )
        for output, prefix in zip(
            transfer.output_ports,
            transfer.input_prefixes,
            strict=True,
        )
    )
    ranks = _resolve_retained_ranks(
        retained_ranks,
        outputs=transfer.output_ports,
        spectrum_ranks=spectrum_ranks,
    )

    factors: list[ModalConnectivityFactor] = []
    for output_index, (output, prefix_names, retained_rank) in enumerate(
        zip(
            transfer.output_ports,
            transfer.input_prefixes,
            ranks,
            strict=True,
        )
    ):
        prefix_moments = tuple(input_by_name[name] for name in prefix_names)
        covariance_factors = tuple(
            factor_psd_support(
                moment.covariance,
                relative_cutoff=eigen_cutoff,
            )
            for moment in prefix_moments
        )
        output_psd = factor_psd_support(
            output_by_name[output.name].fisher,
            relative_cutoff=eigen_cutoff,
        )
        covariance_root = _block_diag(
            tuple(factor.square_root for factor in covariance_factors)
        )
        covariance_inverse = _block_diag(
            tuple(
                factor.inverse_square_root
                for factor in covariance_factors
            )
        )
        source_matrix = transfer.transfer_matrices[output_index]
        weighted = (
            output_psd.square_root @ source_matrix @ covariance_root
        )
        left, singular_values, right_h = torch.linalg.svd(
            weighted,
            full_matrices=False,
        )
        left, singular_values, right_h = _canonicalize_svd_subspaces(
            weighted,
            left,
            singular_values,
            right_h,
        )
        singular_scale = (
            float(singular_values[0].item())
            if singular_values.numel()
            else 0.0
        )
        numerical_relative = (
            torch.finfo(torch.float64).eps
            * max(weighted.shape)
            * 128
        )
        singular_tolerance = singular_scale * max(
            singular_cutoff,
            numerical_relative,
        )
        root = singular_values[:retained_rank].sqrt()
        restriction = (
            root.unsqueeze(1) * right_h[:retained_rank]
        ) @ covariance_inverse
        prolongation = output_psd.inverse_square_root @ (
            left[:, :retained_rank] * root.unsqueeze(0)
        )
        input_mean = torch.cat(
            tuple(moment.mean for moment in prefix_moments),
            dim=0,
        )
        affine_offset = transfer.affine_offsets[output_index]
        output_mean = affine_offset + source_matrix @ input_mean
        block_energies: list[Tensor] = []
        start = 0
        for moment in prefix_moments:
            stop = start + moment.port.width
            block_energies.append(weighted[:, start:stop].square().sum())
            start = stop
        factors.append(
            ModalConnectivityFactor(
                source_transfer_sha256=transfer.artifact_sha256,
                output_port=output,
                input_ports=tuple(
                    moment.port for moment in prefix_moments
                ),
                input_moment_sha256s=tuple(
                    moment.artifact_sha256 for moment in prefix_moments
                ),
                output_moment_sha256=output_by_name[
                    output.name
                ].artifact_sha256,
                reduction_id=reduction_id,
                sample_count=sample_count,
                retained_rank=retained_rank,
                singular_values=singular_values,
                weighted_left_vectors=left,
                weighted_right_vectors=right_h.T,
                input_inverse_square_roots=tuple(
                    factor.inverse_square_root
                    for factor in covariance_factors
                ),
                output_inverse_square_root=(
                    output_psd.inverse_square_root
                ),
                restriction=restriction,
                prolongation=prolongation,
                input_mean=input_mean,
                output_mean=output_mean,
                affine_offset=affine_offset,
                source_block_energy=torch.stack(block_energies),
                input_support_ranks=tuple(
                    factor.support_rank
                    for factor in covariance_factors
                ),
                output_support_rank=output_psd.support_rank,
                singular_tolerance=float(singular_tolerance),
            )
        )
    return ModalConnectivityDecomposition(
        source_transfer=transfer,
        input_moments=input_tuple,
        output_moments=output_tuple,
        factors=tuple(factors),
        relative_eigenvalue_cutoff=eigen_cutoff,
        relative_singular_value_cutoff=singular_cutoff,
        assumes_block_diagonal_input_covariance=(
            needs_input_independence
        ),
        assumes_block_diagonal_output_fisher=(
            len(transfer.output_ports) > 1
        ),
    )
