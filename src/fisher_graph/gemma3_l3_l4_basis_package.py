"""Prompt-blind Fisher basis package for the frozen Gemma L3/L4 boundary.

The historical hierarchy measurement contains both reusable Fisher factors and
prompt-local diagnostic state.  A prompt-blind compiler must not be able to
open the latter.  This module therefore provides a one-way export boundary:

* the exporter authenticates the frozen hierarchy and refit lineage;
* only means, Fisher bases, the L4 balanced modal singular spectrum, and the
  L3 input covariance are copied;
* the resulting package contains no prompt identifiers, token data,
  activation rows, or prompt-local edge kernels;
* downstream compilers strict-load only this reduced package.

The package is an ignored research artifact.  It is not a model checkpoint and
does not authorize a compression, fidelity, or speed claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

import torch
from torch import Tensor

from .external_models import find_git_worktree
from .gemma3_full_mlp_stack_refit_runtime import Gemma3RefitRuntimeCatalog
from .gemma3_l3_l4_spectral_mapping_experiment import (
    DEFAULT_HIERARCHY_ARTIFACT,
    DEFAULT_HIERARCHY_ARTIFACT_SHA256,
    Gemma3L3L4SpectralReference,
    load_gemma3_l3_l4_spectral_reference,
)


__all__ = [
    "DEFAULT_BASIS_PACKAGE",
    "Gemma3L3L4BasisPackage",
    "export_gemma3_l3_l4_basis_package",
    "load_gemma3_l3_l4_basis_package",
]


DEFAULT_BASIS_PACKAGE = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-prompt-blind-basis-v3.pt"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_prompt_blind_basis_v3"
_FORMAT_VERSION = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_DOMAIN = b"fisher-graph:gemma3-l3-l4-basis-payload:v3\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-basis-marker:v3\0"
_PROJECTION_BINDING_FIELDS = (
    "source_model_sha256",
    "generator_plan_sha256s",
    "layer3_factor_sha256",
    "layer4_factor_sha256",
)
_TENSOR_NAMES = (
    "x3_mean",
    "y3_mean",
    "x4_mean",
    "y4_mean",
    "R3",
    "P3",
    "R4",
    "P4",
    "S4",
    "x3_covariance",
)
_TENSOR_NDIMS = {
    "x3_mean": 1,
    "y3_mean": 1,
    "x4_mean": 1,
    "y4_mean": 1,
    "R3": 2,
    "P3": 2,
    "R4": 2,
    "P4": 2,
    "S4": 1,
    "x3_covariance": 2,
}
_SOURCE_SITES = {
    "x3": "layer.3.mlp.normalized_input",
    "y3": "layer.3.mlp.operator_output",
    "x4": "layer.4.mlp.normalized_input",
    "y4": "layer.4.mlp.operator_output",
}
_SCIENTIFIC_STATUS = {
    "scope": "prompt_blind_after_frozen_fisher_aggregate_export",
    "authorizes_hash_bound_synthetic_provider_protocol": True,
    "contains_prompt_derived_aggregate_statistics": True,
    "prompt_membership_privacy_proven": False,
    "prompt_distribution_fidelity_claim": False,
    "compression_claim": False,
    "latency_or_speed_claim": False,
    "development_only": True,
}
_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_identifiers": False,
    "contains_prompt_families": False,
    "contains_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_prompt_local_edge_kernels": False,
    "contains_fisher_bases_and_moments": True,
    "contains_balanced_modal_singular_spectra": True,
    "contains_prompt_derived_aggregate_statistics": True,
    "artifact_must_remain_outside_git": True,
}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode):
            raise ValueError(f"{path.name} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_tensor(value: object, *, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    result = result.contiguous().clone()
    if result.ndim != ndim or any(int(size) <= 0 for size in result.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": "float64",
            "shape": tuple(int(size) for size in tensor.shape),
        }
    )
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _payload_sha256(
    *,
    binding: Mapping[str, object],
    tensors: Mapping[str, Tensor],
) -> str:
    logical = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "binding": dict(binding),
        "source_sites": dict(_SOURCE_SITES),
        "scientific_status": dict(_SCIENTIFIC_STATUS),
        "tensor_sha256s": {
            name: _tensor_sha256(tensors[name]) for name in _TENSOR_NAMES
        },
        "safety": dict(_SAFETY),
    }
    return hashlib.sha256(
        _PAYLOAD_DOMAIN + _canonical_json_bytes(logical)
    ).hexdigest()


def _report_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("report_sha256", None)
    return hashlib.sha256(
        _REPORT_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the frozen format")


@dataclass(frozen=True, slots=True)
class Gemma3L3L4BasisPackage:
    """Strict in-memory view of the prompt-blind basis artifact."""

    basis_payload_sha256: str
    source_model_sha256: str
    generator_plan_sha256s: tuple[str, ...]
    layer3_factor_sha256: str
    layer4_factor_sha256: str
    x3_mean: Tensor
    y3_mean: Tensor
    x4_mean: Tensor
    y4_mean: Tensor
    R3: Tensor
    P3: Tensor
    R4: Tensor
    P4: Tensor
    S4: Tensor
    x3_covariance: Tensor

    def __post_init__(self) -> None:
        for name in (
            "basis_payload_sha256",
            "source_model_sha256",
            "layer3_factor_sha256",
            "layer4_factor_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            type(self.generator_plan_sha256s) is not tuple
            or not self.generator_plan_sha256s
        ):
            raise ValueError("generator plan hashes must be a nonempty tuple")
        for digest in self.generator_plan_sha256s:
            _require_sha256(digest, label="generator plan")
        for name in ("x3_mean", "y3_mean", "x4_mean", "y4_mean", "S4"):
            object.__setattr__(
                self,
                name,
                _canonical_tensor(getattr(self, name), label=name, ndim=1),
            )
        for name in ("R3", "P3", "R4", "P4", "x3_covariance"):
            object.__setattr__(
                self,
                name,
                _canonical_tensor(getattr(self, name), label=name, ndim=2),
            )
        width = int(self.x3_mean.numel())
        if any(
            int(value.numel()) != width
            for value in (self.y3_mean, self.x4_mean, self.y4_mean)
        ):
            raise ValueError("L3/L4 means must share the residual width")
        if any(
            tuple(value.shape) != (width, width)
            for value in (
                self.R3,
                self.P3,
                self.R4,
                self.P4,
                self.x3_covariance,
            )
        ):
            raise ValueError("L3/L4 basis geometry is invalid")
        if self.S4.shape != (width,):
            raise ValueError("L4 modal singular spectrum geometry is invalid")
        singular_tolerance = max(float(self.S4.abs().max()), 1.0) * 1e-12
        if (
            bool((self.S4 < 0.0).any())
            or bool((self.S4[1:] > self.S4[:-1] + singular_tolerance).any())
        ):
            raise ValueError(
                "L4 balanced modal singular spectrum is invalid"
            )
        if not torch.allclose(
            self.x3_covariance,
            self.x3_covariance.T,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("x3 covariance must be symmetric")
        covariance_scale = float(self.x3_covariance.abs().max())
        if not math.isfinite(covariance_scale) or covariance_scale < 0.0:
            raise ValueError("x3 covariance scale must be finite")
        scaled_covariance = (
            self.x3_covariance.clone()
            if covariance_scale == 0.0
            else self.x3_covariance / covariance_scale
        )
        # Average only after scaling.  Directly adding two finite values near
        # float64's maximum can overflow before the eigensolver sees them.
        scaled_covariance = (
            scaled_covariance * 0.5
            + scaled_covariance.T * 0.5
        )
        if not bool(torch.isfinite(scaled_covariance).all()):
            raise ValueError("x3 covariance scaling must remain finite")
        try:
            eigenvalues = torch.linalg.eigvalsh(scaled_covariance)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                "x3 covariance eigenspectrum could not be computed"
            ) from error
        if (
            tuple(eigenvalues.shape) != (width,)
            or not bool(torch.isfinite(eigenvalues).all())
        ):
            raise ValueError("x3 covariance eigenspectrum must be finite")
        tolerance = max(float(eigenvalues.abs().max()), 1.0) * 1e-9
        if float(eigenvalues.min()) < -tolerance:
            raise ValueError("x3 covariance must be positive semidefinite")
        expected = _payload_sha256(
            binding=self.binding(),
            tensors=self.tensors(),
        )
        if expected != self.basis_payload_sha256:
            raise ValueError("basis logical payload hash is invalid")

    @property
    def residual_width(self) -> int:
        return int(self.x3_mean.numel())

    def binding(self) -> dict[str, object]:
        return {
            "source_model_sha256": self.source_model_sha256,
            "generator_plan_sha256s": self.generator_plan_sha256s,
            "layer3_factor_sha256": self.layer3_factor_sha256,
            "layer4_factor_sha256": self.layer4_factor_sha256,
        }

    def tensors(self) -> dict[str, Tensor]:
        return {name: getattr(self, name).clone() for name in _TENSOR_NAMES}

    def source_mode_standard_deviations(self, rank: int) -> Tensor:
        if type(rank) is not int or rank <= 0 or rank > self.residual_width:
            raise ValueError("rank is outside the frozen L3 basis")
        restriction = self.R3[:rank]
        variances = torch.diagonal(
            restriction @ self.x3_covariance @ restriction.T
        )
        tolerance = max(float(variances.abs().max()), 1.0) * 1e-10
        if float(variances.min()) < -tolerance:
            raise ValueError("projected source covariance has negative energy")
        result = variances.clamp_min(0.0).sqrt()
        if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
            raise ValueError("source modal standard deviations are degenerate")
        return result.contiguous()

    def metadata(self) -> dict[str, object]:
        return {
            **self.binding(),
            "basis_payload_sha256": self.basis_payload_sha256,
            "residual_width": self.residual_width,
            "source_sites": dict(_SOURCE_SITES),
            "prompt_blind_after_frozen_export": True,
            "contains_prompt_derived_aggregate_statistics": True,
            "contains_balanced_modal_singular_spectrum": True,
        }


def _reference_binding(reference: Gemma3L3L4SpectralReference) -> dict[str, object]:
    return {
        "source_model_sha256": reference.source_model_sha256,
        "generator_plan_sha256s": reference.generator_plan_sha256s,
        "layer3_factor_sha256": reference.layer3_factor_sha256,
        "layer4_factor_sha256": reference.layer4_factor_sha256,
    }


def _build_package(
    reference: Gemma3L3L4SpectralReference,
) -> Gemma3L3L4BasisPackage:
    tensors = {
        name: getattr(reference, name).clone() for name in _TENSOR_NAMES
    }
    binding = _reference_binding(reference)
    payload = _payload_sha256(binding=binding, tensors=tensors)
    return Gemma3L3L4BasisPackage(
        basis_payload_sha256=payload,
        **binding,
        **tensors,
    )


def _artifact(package: Gemma3L3L4BasisPackage) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "scientific_status": dict(_SCIENTIFIC_STATUS),
        "binding": package.binding(),
        "source_sites": dict(_SOURCE_SITES),
        "basis_payload_sha256": package.basis_payload_sha256,
        "tensors": package.tensors(),
        "safety": dict(_SAFETY),
    }


def _package_from_artifact(raw: object) -> Gemma3L3L4BasisPackage:
    if not isinstance(raw, Mapping):
        raise TypeError("basis package must contain a mapping")
    _strict_keys(
        raw,
        expected={
            "schema",
            "format_version",
            "scientific_status",
            "binding",
            "source_sites",
            "basis_payload_sha256",
            "tensors",
            "safety",
        },
        label="basis package",
    )
    if raw["schema"] != _SCHEMA or raw["format_version"] != _FORMAT_VERSION:
        raise ValueError("basis package schema or version drifted")
    if (
        raw["source_sites"] != _SOURCE_SITES
        or raw["safety"] != _SAFETY
        or raw["scientific_status"] != _SCIENTIFIC_STATUS
    ):
        raise ValueError("basis package source sites or safety drifted")
    binding = raw["binding"]
    tensors = raw["tensors"]
    if not isinstance(binding, Mapping) or not isinstance(tensors, Mapping):
        raise TypeError("basis binding and tensors must be mappings")
    _strict_keys(
        binding,
        expected=set(_PROJECTION_BINDING_FIELDS),
        label="basis binding",
    )
    _strict_keys(
        tensors,
        expected=set(_TENSOR_NAMES),
        label="basis tensors",
    )
    plans = binding["generator_plan_sha256s"]
    if not isinstance(plans, (tuple, list)):
        raise TypeError("basis generator plans must be a sequence")
    return Gemma3L3L4BasisPackage(
        basis_payload_sha256=_require_sha256(
            raw["basis_payload_sha256"],
            label="basis payload",
        ),
        source_model_sha256=_require_sha256(
            binding["source_model_sha256"],
            label="source model",
        ),
        generator_plan_sha256s=tuple(
            _require_sha256(value, label="generator plan") for value in plans
        ),
        layer3_factor_sha256=_require_sha256(
            binding["layer3_factor_sha256"],
            label="layer3 factor",
        ),
        layer4_factor_sha256=_require_sha256(
            binding["layer4_factor_sha256"],
            label="layer4 factor",
        ),
        **{
            name: _canonical_tensor(
                tensors[name],
                label=name,
                ndim=_TENSOR_NDIMS[name],
            )
            for name in _TENSOR_NAMES
        },
    )


def _basis_report(
    package: Gemma3L3L4BasisPackage,
    *,
    tensor_file_sha256: str,
    tensor_file_bytes: int,
) -> dict[str, object]:
    if (
        type(tensor_file_bytes) is not int
        or tensor_file_bytes <= 0
    ):
        raise ValueError("tensor_file_bytes must be positive")
    report: dict[str, object] = {
        "schema": f"{_SCHEMA}.report",
        "format_version": _FORMAT_VERSION,
        "basis_payload_sha256": package.basis_payload_sha256,
        "binding": package.binding(),
        "artifact": {
            "tensor_role": "prompt_blind_basis_package",
            "tensor_file_sha256": _require_sha256(
                tensor_file_sha256,
                label="basis tensor file",
            ),
            "tensor_file_bytes": tensor_file_bytes,
            "report_role": "prompt_blind_basis_commit_marker",
            "committable": False,
        },
        "safety": dict(_SAFETY),
    }
    report["report_sha256"] = _report_sha256(report)
    return report


def _export_receipt(
    reference: Gemma3L3L4SpectralReference,
    *,
    basis_payload_sha256: str,
    tensor_file_sha256: str,
    marker_sha256: str,
) -> dict[str, object]:
    return {
        "schema": f"{_SCHEMA}.quarantined_export_receipt",
        "format_version": _FORMAT_VERSION,
        "basis_payload_sha256": _require_sha256(
            basis_payload_sha256,
            label="basis payload",
        ),
        "tensor_file_sha256": _require_sha256(
            tensor_file_sha256,
            label="basis tensor",
        ),
        "marker_sha256": _require_sha256(
            marker_sha256,
            label="basis marker",
        ),
        "hierarchy_artifact_sha256": (
            reference.hierarchy_artifact_sha256
        ),
        "base_artifact_file_sha256": (
            reference.base_artifact_file_sha256
        ),
        "base_scientific_payload_sha256": (
            reference.base_scientific_payload_sha256
        ),
        "refit_artifact_file_sha256": (
            reference.refit_artifact_file_sha256
        ),
        "refit_scientific_payload_sha256": (
            reference.refit_scientific_payload_sha256
        ),
        "excluded_from_compiler_visible_marker": True,
        "exporter_process_opened_frozen_hierarchy": True,
    }


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("basis output must use a .pt suffix")
    report = destination.with_suffix(".json")
    if destination.exists() or report.exists():
        raise FileExistsError("refusing to overwrite basis package output")
    worktree = find_git_worktree(Path(__file__))
    resolved = destination.expanduser().resolve()
    if worktree is not None:
        root = worktree.resolve()
        if resolved == root or root in resolved.parents:
            relative = resolved.relative_to(root)
            if not relative.parts or relative.parts[0] not in (
                ".local-runs",
                "local-runs",
            ):
                raise ValueError(
                    "basis output inside the worktree must remain under "
                    "an ignored .local-runs directory"
                )
    return destination


def _atomic_publish_bytes(
    *,
    destination: Path,
    writer: Any,
) -> tuple[int, int]:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(name)
    published_identity: tuple[int, int] | None = None
    staging_identity: tuple[int, int] | None = None
    staging_cleanup_complete = False
    try:
        writer(temporary)
        staged = os.fstat(descriptor)
        staging_identity = (staged.st_dev, staged.st_ino)
        if (
            not stat.S_ISREG(staged.st_mode)
            or _path_identity(temporary) != staging_identity
        ):
            raise RuntimeError("basis publication staging inode was replaced")
        os.fsync(descriptor)
        os.link(temporary, destination, follow_symlinks=False)
        published_identity = staging_identity
        staging_cleanup_complete = _best_effort_unlink_owned(
            temporary,
            staging_identity,
        )
        try:
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            # The complete destination became observable at os.link().  Its
            # durability is now uncertain, so retracting it would let a
            # concurrent observer watch a published file disappear.  Preserve
            # the inode and let the pair-level reservation remain fail-closed.
            raise _PublicationDurabilityUncertain(
                destination=destination,
                identity=published_identity,
            ) from error
        return published_identity
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite {destination.name}"
        ) from error
    finally:
        try:
            os.close(descriptor)
        except OSError:
            # Closing the staging descriptor does not revoke a linked
            # destination.  A preceding fsync is the durability boundary.
            pass
        cleanup_identity = staging_identity
        if cleanup_identity is None:
            cleanup_identity = _path_identity(temporary)
        try:
            if cleanup_identity is not None and not staging_cleanup_complete:
                removed = _best_effort_unlink_owned(
                    temporary,
                    cleanup_identity,
                )
                if removed:
                    _best_effort_fsync_directory(destination.parent)
        except OSError:
            # A hidden staging-file cleanup failure does not invalidate a
            # destination whose complete hard link is already observable.
            # Reporting failure here would invite an unsafe caller rollback
            # after another reader may have observed the committed file.
            pass


class _PublicationDurabilityUncertain(OSError):
    """A complete path is visible, but its directory fsync did not finish."""

    def __init__(
        self,
        *,
        destination: Path,
        identity: tuple[int, int],
    ) -> None:
        super().__init__(
            f"{destination.name} is published but directory durability "
            "is uncertain; its reservation must remain in place"
        )
        self.destination = destination
        self.identity = identity


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        state = path.lstat()
    except FileNotFoundError:
        return None
    return state.st_dev, state.st_ino


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> bool:
    if _path_identity(path) == identity:
        # Do not perform another fallible path lookup after unlink succeeds:
        # for claim removal, that unlink is the publication commit point.
        # missing_ok=False also turns a check/unlink race into failure instead
        # of falsely reporting that our inode was removed.
        path.unlink()
        return True
    return False


def _best_effort_unlink_owned(
    path: Path,
    identity: tuple[int, int],
) -> bool:
    """Retry cleanup of a path only while it names the inode we created."""

    for _ in range(2):
        try:
            if _unlink_if_identity(path, identity):
                return True
            return _path_identity(path) is None
        except OSError:
            continue
    return _path_identity(path) is None


def _best_effort_fsync_directory(path: Path) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(descriptor)
    except OSError:
        return False
    finally:
        os.close(descriptor)
    return True


def export_gemma3_l3_l4_basis_package(
    *,
    catalog: Gemma3RefitRuntimeCatalog,
    hierarchy_artifact_path: Path | str = DEFAULT_HIERARCHY_ARTIFACT,
    hierarchy_artifact_sha256: str = DEFAULT_HIERARCHY_ARTIFACT_SHA256,
    output: Path | str = DEFAULT_BASIS_PACKAGE,
) -> dict[str, object]:
    """Authenticate hierarchy v3 and publish its prompt-blind subset."""

    if not isinstance(catalog, Gemma3RefitRuntimeCatalog):
        raise TypeError("catalog must be a Gemma3RefitRuntimeCatalog")
    destination = _validate_output_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim = destination.with_name(f".{destination.name}.publish.lock")
    try:
        descriptor = os.open(
            claim,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            "basis package output pair is already reserved"
        ) from error
    try:
        claim_state = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    claim_identity = (claim_state.st_dev, claim_state.st_ino)
    try:
        claim_directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(claim_directory)
        finally:
            os.close(claim_directory)
    except OSError:
        _best_effort_unlink_owned(claim, claim_identity)
        _best_effort_fsync_directory(destination.parent)
        raise
    claim_active = True
    preserve_claim = False
    tensor_identity: tuple[int, int] | None = None
    report_identity: tuple[int, int] | None = None
    pair_published = False
    try:
        if destination.exists() or destination.with_suffix(".json").exists():
            raise FileExistsError(
                "refusing to overwrite basis package output"
            )
        reference = load_gemma3_l3_l4_spectral_reference(
            hierarchy_artifact_path,
            expected_file_sha256=hierarchy_artifact_sha256,
            catalog=catalog,
        )
        package = _build_package(reference)
        artifact = _artifact(package)
        try:
            tensor_identity = _atomic_publish_bytes(
                destination=destination,
                writer=lambda path: torch.save(artifact, path),
            )
        except _PublicationDurabilityUncertain as error:
            tensor_identity = error.identity
            preserve_claim = True
            raise
        tensor_bytes = _read_regular_file(destination)
        tensor_file_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
        restored = _package_from_artifact(
            torch.load(
                io.BytesIO(tensor_bytes),
                map_location="cpu",
                weights_only=True,
            )
        )
        if restored.basis_payload_sha256 != package.basis_payload_sha256:
            raise RuntimeError("staged basis package roundtrip drifted")
        report = _basis_report(
            package,
            tensor_file_sha256=tensor_file_sha256,
            tensor_file_bytes=len(tensor_bytes),
        )
        report_path = destination.with_suffix(".json")

        def write_report(path: Path) -> None:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    report,
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")

        receipt = _export_receipt(
            reference,
            basis_payload_sha256=package.basis_payload_sha256,
            tensor_file_sha256=tensor_file_sha256,
            marker_sha256=report["report_sha256"],
        )
        result = {
            **report,
            "quarantined_export_receipt_not_in_marker": receipt,
        }
        # The marker is linked and directory-fsynced while the reservation is
        # still present.  Public loaders reject that reservation, so no reader
        # can accept a pair which this transaction may still roll back.
        try:
            report_identity = _atomic_publish_bytes(
                destination=report_path,
                writer=write_report,
            )
        except _PublicationDurabilityUncertain as error:
            report_identity = error.identity
            preserve_claim = True
            raise
        pair_published = True
        # Releasing the claim is the visibility/commit action.  From this point
        # onward the durable tensor/marker pair is never rolled back.
        if not _unlink_if_identity(claim, claim_identity):
            preserve_claim = True
            raise RuntimeError("basis publication reservation was not released")
        claim_active = False
        # Claim removal is the commit point.  A subsequent directory-fsync
        # error cannot safely be reported as transaction failure because a
        # strict loader may already have accepted the complete pair.
        _best_effort_fsync_directory(destination.parent)
        return result
    except BaseException:
        if not pair_published and not preserve_claim:
            if report_identity is not None:
                _unlink_if_identity(
                    destination.with_suffix(".json"),
                    report_identity,
                )
            if tensor_identity is not None:
                _unlink_if_identity(destination, tensor_identity)
            _best_effort_fsync_directory(destination.parent)
        raise
    finally:
        if claim_active and not preserve_claim and not pair_published:
            try:
                if _unlink_if_identity(claim, claim_identity):
                    _best_effort_fsync_directory(destination.parent)
            except OSError:
                pass


def _load_basis_report(
    source: Path,
    *,
    tensor_file_sha256: str,
    tensor_file_bytes: int,
) -> Mapping[str, object]:
    report_path = source.with_suffix(".json")
    if not report_path.is_file():
        raise FileNotFoundError(
            "basis package commit-marker report is missing"
        )
    report_bytes = _read_regular_file(report_path)
    report = json.loads(report_bytes.decode("utf-8"))
    if not isinstance(report, Mapping):
        raise TypeError("basis package report must contain a mapping")
    _strict_keys(
        report,
        expected={
            "schema",
            "format_version",
            "basis_payload_sha256",
            "binding",
            "artifact",
            "safety",
            "report_sha256",
        },
        label="basis package report",
    )
    if (
        report["schema"] != f"{_SCHEMA}.report"
        or report["format_version"] != _FORMAT_VERSION
        or report["safety"] != _SAFETY
        or report["report_sha256"] != _report_sha256(report)
    ):
        raise ValueError("basis package report authentication failed")
    artifact = report["artifact"]
    if not isinstance(artifact, Mapping):
        raise TypeError("basis report artifact must be a mapping")
    _strict_keys(
        artifact,
        expected={
            "tensor_role",
            "tensor_file_sha256",
            "tensor_file_bytes",
            "report_role",
            "committable",
        },
        label="basis report artifact",
    )
    if dict(artifact) != {
        "tensor_role": "prompt_blind_basis_package",
        "tensor_file_sha256": tensor_file_sha256,
        "tensor_file_bytes": tensor_file_bytes,
        "report_role": "prompt_blind_basis_commit_marker",
        "committable": False,
    }:
        raise ValueError("basis report does not commit the tensor file")
    return report


def load_gemma3_l3_l4_basis_package(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str | None = None,
) -> Gemma3L3L4BasisPackage:
    """Strict-load a prompt-blind package and authenticate every tensor."""

    source = Path(path)
    expected_file = _require_sha256(
        expected_file_sha256,
        label="expected basis file",
    )
    claim = source.with_name(f".{source.name}.publish.lock")
    if claim.exists():
        raise RuntimeError("basis package publication is still in progress")
    tensor_bytes = _read_regular_file(source)
    actual_file = hashlib.sha256(tensor_bytes).hexdigest()
    if actual_file != expected_file:
        raise ValueError("basis package file hash differs from expectation")
    report = _load_basis_report(
        source,
        tensor_file_sha256=actual_file,
        tensor_file_bytes=len(tensor_bytes),
    )
    raw = torch.load(
        io.BytesIO(tensor_bytes),
        map_location="cpu",
        weights_only=True,
    )
    package = _package_from_artifact(raw)
    if (
        expected_payload_sha256 is not None
        and package.basis_payload_sha256
        != _require_sha256(
            expected_payload_sha256,
            label="expected basis payload",
        )
    ):
        raise ValueError("basis logical payload hash differs from expectation")
    if (
        report["basis_payload_sha256"] != package.basis_payload_sha256
        or _canonical_json_bytes(report["binding"])
        != _canonical_json_bytes(package.binding())
    ):
        raise ValueError("basis report and tensor logical contents differ")
    return package
