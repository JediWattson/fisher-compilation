"""Authenticated local sidecar for the selected complete-H4 D320 basis.

The successful A16 tail-informed factorial intentionally published no basis
coefficients.  This module defines the narrow tensor boundary needed by the
next one-pass carrier-transfer oracle: one CPU float64 ``[320, 640]`` matrix,
bound to the exact successful parent report and selected-arm lineage.

The sidecar is local research state, not a model artifact.  It must remain
under ``.local-runs``, is never committable, and contains no prompts, token
IDs, logits, gradients, or activation rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile

import torch
from torch import Tensor

from .gemma3_l3_l4_complete_h4_projection import (
    ImmutableFloat64Matrix,
    MatrixInput,
)
from .gemma3_l3_l4_complete_h4_tail_informed_projection import (
    TAIL_INFORMED_PROJECTION_ORDERING,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
    gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256,
)


__all__ = [
    "DEFAULT_LOCAL_ROOT",
    "DEFAULT_OUTPUT",
    "CompleteH4Rank320BasisSidecar",
    "build_complete_h4_rank320_basis_sidecar",
    "load_complete_h4_rank320_basis_sidecar",
    "save_complete_h4_rank320_basis_sidecar",
]


DEFAULT_LOCAL_ROOT = Path(".local-runs")
DEFAULT_OUTPUT = DEFAULT_LOCAL_ROOT / "google--gemma-3-270m" / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-"
    "tail-informed-rank320-basis-a-fit16-dev-v1.pt"
)

SCHEMA = "fisher_graph.gemma3_l3_l4_complete_h4_rank320_basis_sidecar"
FORMAT_VERSION = 1
PARENT_FILE_SHA256 = (
    "30246112335e5fa6ab62f299405d1430c9e182684d4d13e4c6dd6decd0d82a0d"
)
PARENT_REPORT_SHA256 = (
    "dd325816e21a095921b9aefc955e396c8d765e7c193cb3838e305540db4b68e6"
)
SELECTED_ARM = "tail_informed.rank320"
TAIL_INFORMED_FIT_ARTIFACT_SHA256 = (
    "0a5328c5c2a5ec7f5dd39799c7cd349b7dc5a5d4f625d0d492ab552df48aecb7"
)
FIT_TO_PREFIX_LINEAGE_SHA256 = (
    "ad19b3a896c05e4d4f2de36ddd778020e1e4a1cf0733488ae3e919c041e67fad"
)
EXPECTED_BASIS_MATRIX_SHA256 = (
    "93de87e8abd6242fa0ca2ab1683607aa7f7b89617c66a5c673d218d04a74097f"
)
EXPECTED_RUNTIME_TENSOR_SHA256 = (
    "e345e71394a65f55f5fd9f96196d830f9dec1df7962216dd4c28f3835793d0cd"
)
EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256 = (
    "5ad248ba6c0e4b5a208f2ad05517649ee728ad95a00d64ba8e95917e3a45fbb4"
)
PROJECTION_RANK = 320
RESIDUAL_WIDTH = 640

_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-rank320-basis-sidecar:v1\0"
)
_MAX_SIDECAR_FILE_BYTES = 8 * 1024 * 1024
_STATE_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "basis",
        "parent_file_sha256",
        "parent_report_sha256",
        "selected_arm",
        "tail_informed_fit_artifact_sha256",
        "fit_to_prefix_lineage_sha256",
        "projection_rank",
        "residual_width",
        "ordering",
        "basis_matrix_sha256",
        "runtime_tensor_sha256",
        "projection_basis_artifact_sha256",
        "orthonormal_max_abs_error_hex",
        "contains_basis_coefficients",
        "contains_prompt_text",
        "contains_token_ids",
        "contains_logits",
        "contains_activation_rows",
        "contains_gradient_rows",
        "committable",
        "artifact_sha256",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _logical_artifact_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _ARTIFACT_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()


def _file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contract_payload(*, orthonormal_error: float) -> dict[str, object]:
    if not math.isfinite(orthonormal_error) or orthonormal_error < 0.0:
        raise ValueError("basis orthonormality error must be finite and nonnegative")
    return {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "parent_file_sha256": PARENT_FILE_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "selected_arm": SELECTED_ARM,
        "tail_informed_fit_artifact_sha256": (
            TAIL_INFORMED_FIT_ARTIFACT_SHA256
        ),
        "fit_to_prefix_lineage_sha256": FIT_TO_PREFIX_LINEAGE_SHA256,
        "projection_rank": PROJECTION_RANK,
        "residual_width": RESIDUAL_WIDTH,
        "ordering": TAIL_INFORMED_PROJECTION_ORDERING,
        "dtype": "cpu_float64",
        "basis_matrix_sha256": EXPECTED_BASIS_MATRIX_SHA256,
        "runtime_tensor_sha256": EXPECTED_RUNTIME_TENSOR_SHA256,
        "projection_basis_artifact_sha256": (
            EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256
        ),
        "orthonormal_max_abs_error_hex": orthonormal_error.hex(),
        "contains_basis_coefficients": True,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_logits": False,
        "contains_activation_rows": False,
        "contains_gradient_rows": False,
        "committable": False,
    }


def _as_immutable_basis(value: MatrixInput) -> ImmutableFloat64Matrix:
    if isinstance(value, ImmutableFloat64Matrix):
        immutable = ImmutableFloat64Matrix(
            row_count=value.row_count,
            width=value.width,
            _little_endian_bytes=value._little_endian_bytes,
        )
        if immutable.shape != (PROJECTION_RANK, RESIDUAL_WIDTH):
            raise ValueError("rank-320 immutable basis must have shape [320, 640]")
        return immutable
    if not isinstance(value, Tensor):
        raise TypeError("rank-320 basis must be a torch.Tensor")
    if (
        value.shape != (PROJECTION_RANK, RESIDUAL_WIDTH)
        or value.dtype != torch.float64
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
    ):
        raise ValueError(
            "rank-320 basis must be contiguous, gradient-free CPU float64 "
            "with shape [320, 640]"
        )
    return ImmutableFloat64Matrix.from_tensor(value, label="rank-320 basis")


@dataclass(frozen=True, slots=True)
class CompleteH4Rank320BasisSidecar:
    """Immutable D320 coefficients and their successful-parent receipt."""

    basis: MatrixInput = field(repr=False)
    artifact_sha256: str = ""
    basis_matrix_sha256: str = field(init=False)
    runtime_tensor_sha256: str = field(init=False)
    projection_basis_artifact_sha256: str = field(init=False)
    orthonormal_max_abs_error: float = field(init=False)

    def __post_init__(self) -> None:
        immutable = _as_immutable_basis(self.basis)
        if immutable.matrix_sha256 != EXPECTED_BASIS_MATRIX_SHA256:
            raise ValueError("rank-320 basis matrix differs from the parent")
        basis = immutable.to_tensor()
        runtime_sha256 = _runtime_tensor_sha256(basis)
        if runtime_sha256 != EXPECTED_RUNTIME_TENSOR_SHA256:
            raise ValueError("rank-320 runtime tensor differs from the parent")
        projection_artifact = (
            gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
                basis,
                projection_rank=PROJECTION_RANK,
                projection_ordering=TAIL_INFORMED_PROJECTION_ORDERING,
            )
        )
        gram = basis @ basis.T
        identity = torch.eye(PROJECTION_RANK, dtype=torch.float64)
        orthonormal_error = float((gram - identity).abs().max().item())
        if projection_artifact != EXPECTED_PROJECTION_BASIS_ARTIFACT_SHA256:
            raise ValueError("rank-320 projection artifact differs from the parent")
        payload = _contract_payload(orthonormal_error=orthonormal_error)
        computed = _logical_artifact_sha256(payload)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("rank-320 sidecar logical artifact hash differs")
        else:
            object.__setattr__(self, "artifact_sha256", computed)
        object.__setattr__(self, "basis", immutable)
        object.__setattr__(
            self,
            "basis_matrix_sha256",
            immutable.matrix_sha256,
        )
        object.__setattr__(self, "runtime_tensor_sha256", runtime_sha256)
        object.__setattr__(
            self,
            "projection_basis_artifact_sha256",
            projection_artifact,
        )
        object.__setattr__(
            self,
            "orthonormal_max_abs_error",
            orthonormal_error,
        )

    def validate_integrity(self) -> None:
        immutable = self.basis
        if not isinstance(immutable, ImmutableFloat64Matrix):
            raise RuntimeError("rank-320 sidecar lost its immutable basis")
        rebuilt = CompleteH4Rank320BasisSidecar(
            basis=ImmutableFloat64Matrix(
                row_count=immutable.row_count,
                width=immutable.width,
                _little_endian_bytes=immutable._little_endian_bytes,
            ),
            artifact_sha256=self.artifact_sha256,
        )
        fields = (
            "basis_matrix_sha256",
            "runtime_tensor_sha256",
            "projection_basis_artifact_sha256",
            "orthonormal_max_abs_error",
            "artifact_sha256",
        )
        if any(getattr(rebuilt, name) != getattr(self, name) for name in fields):
            raise RuntimeError("rank-320 sidecar integrity fields drifted")

    def basis_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.basis.to_tensor()  # type: ignore[union-attr]

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **_contract_payload(
                orthonormal_error=self.orthonormal_max_abs_error
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        metadata = self.metadata()
        metadata.pop("dtype")
        return {**metadata, "basis": self.basis_tensor()}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "CompleteH4Rank320BasisSidecar":
        if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
            raise ValueError("rank-320 sidecar state has an unexpected keyset")
        basis = state.get("basis")
        if not isinstance(basis, Tensor):
            raise TypeError("rank-320 sidecar basis state must be a tensor")
        candidate = cls(
            basis=basis,
            artifact_sha256=str(state.get("artifact_sha256")),
        )
        expected = candidate.state_dict()
        for key in _STATE_KEYS - {"basis"}:
            if state.get(key) != expected[key]:
                raise ValueError(f"rank-320 sidecar state field {key} differs")
        if not torch.equal(basis, expected["basis"]):
            raise ValueError("rank-320 sidecar basis state differs")
        return candidate


def build_complete_h4_rank320_basis_sidecar(
    basis: MatrixInput,
) -> CompleteH4Rank320BasisSidecar:
    """Bind one exact selected D320 tensor to the frozen parent contract."""

    return CompleteH4Rank320BasisSidecar(basis=basis)


def _validated_local_path(
    value: Path | str,
    *,
    local_root: Path | str,
) -> Path:
    raw_root = Path(local_root)
    if raw_root.name != ".local-runs":
        raise ValueError("rank-320 sidecar root must be named .local-runs")
    root = raw_root.absolute().resolve(strict=False)
    raw_path = Path(value)
    path = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
    path = path.absolute()
    if path.suffix != ".pt":
        raise ValueError("rank-320 sidecar output must use the .pt suffix")
    resolved = path.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError("rank-320 sidecar must remain under .local-runs")
    if path.is_symlink():
        raise ValueError("rank-320 sidecar path cannot be a symbolic link")
    return path


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("rank-320 sidecar must be a regular file")
        if info.st_size <= 0 or info.st_size > _MAX_SIDECAR_FILE_BYTES:
            raise ValueError("rank-320 sidecar file size is invalid")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OSError("rank-320 sidecar ended before its stated size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise OSError("rank-320 sidecar grew during its authenticated read")
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            after.st_dev != info.st_dev
            or after.st_ino != info.st_ino
            or after.st_size != info.st_size
        ):
            raise OSError("rank-320 sidecar changed during its authenticated read")
        return payload
    finally:
        os.close(descriptor)


def _load_payload(payload: bytes) -> CompleteH4Rank320BasisSidecar:
    try:
        state = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError("rank-320 sidecar tensor payload is invalid") from error
    if not isinstance(state, Mapping):
        raise ValueError("rank-320 sidecar payload must contain a state mapping")
    return CompleteH4Rank320BasisSidecar.from_state_dict(state)


def save_complete_h4_rank320_basis_sidecar(
    sidecar: CompleteH4Rank320BasisSidecar,
    output: Path | str = DEFAULT_OUTPUT,
    *,
    local_root: Path | str = DEFAULT_LOCAL_ROOT,
) -> dict[str, object]:
    """Atomically publish one authenticated local sidecar without overwrite."""

    if not isinstance(sidecar, CompleteH4Rank320BasisSidecar):
        raise TypeError("sidecar must be CompleteH4Rank320BasisSidecar")
    sidecar.validate_integrity()
    destination = _validated_local_path(output, local_root=local_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("refusing to overwrite rank-320 basis sidecar")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    stage = Path(temporary_name)
    published = False
    try:
        torch.save(sidecar.state_dict(), stage)
        sync_descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(sync_descriptor)
        finally:
            os.close(sync_descriptor)
        payload = _read_regular_file(stage)
        restored = _load_payload(payload)
        if restored.artifact_sha256 != sidecar.artifact_sha256:
            raise RuntimeError("rank-320 sidecar staged roundtrip drifted")
        try:
            os.link(stage, destination)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite rank-320 basis sidecar"
            ) from error
        published = True
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return {
            "file": str(destination),
            "file_sha256": _file_sha256(payload),
            "file_bytes": len(payload),
            "logical_artifact_sha256": sidecar.artifact_sha256,
            "committable": False,
        }
    except BaseException:
        if published:
            # Publication has already become visible and durable enough that
            # silently replacing or rolling it back would weaken provenance.
            raise RuntimeError(
                "rank-320 sidecar publication durability is uncertain"
            )
        raise
    finally:
        stage.unlink(missing_ok=True)


def load_complete_h4_rank320_basis_sidecar(
    path: Path | str = DEFAULT_OUTPUT,
    *,
    local_root: Path | str = DEFAULT_LOCAL_ROOT,
) -> CompleteH4Rank320BasisSidecar:
    """Load and fully reauthenticate one local D320 sidecar."""

    source = _validated_local_path(path, local_root=local_root)
    payload = _read_regular_file(source)
    sidecar = _load_payload(payload)
    sidecar.validate_integrity()
    return sidecar
