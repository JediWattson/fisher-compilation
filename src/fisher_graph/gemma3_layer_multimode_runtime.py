"""Qualified model-facing runtime for the layer-multimode v3 experiment.

This source is model-runtime-inert: its top-level imports are standard-library
modules plus the orchestration ABI.  The package's legacy eager ``__init__``
may already import torch, so this does not claim that normal package import
leaves torch absent.  It does guarantee that transformers, the pinned Gemma
loader, model tensors, and predecessor capture primitives are untouched until
runtime methods reached after strict static preflight.

The first production boundary implemented here is the only authorized reuse of
the predecessor protocol: re-execute its already exposed A role once, through
its original exact-control runtime and panel authority, into new temporary
spools.  No predecessor B/C role is exposed by this API.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
from typing import Mapping
import weakref

from . import gemma3_layer_multimode_experiment as experiment


RUNTIME_QUALIFICATION_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.runtime_qualification.v1"
)
RUNTIME_QUALIFICATION_DOMAIN = (
    b"fisher_graph:layer-multimode:runtime-qualification:v1\0"
)
A_RECAPTURE_COMPARISON_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.predecessor_A_recapture_comparison.v1"
)
A_RECAPTURE_COMPARISON_DOMAIN = (
    b"fisher_graph:layer-multimode:predecessor-A-comparison:v1\0"
)
_DEVELOPMENT_RELEASE_DOMAIN = (
    b"fisher_graph:layer-multimode:development-release-token:v1\0"
)
_FRESH_EXAMPLE_IDENTITY_DOMAIN = (
    b"fisher_graph:layer-multimode:fresh-example-identity:v1\0"
)
_FRESH_EXAMPLE_EXECUTION_DOMAIN = (
    b"fisher_graph:layer-multimode:fresh-example-execution:v1\0"
)
_D3_SERIALIZATION_PROTOCOL_DOMAIN = (
    b"fisher_graph:layer-multimode:d3-serialization-protocol:v1\0"
)
_D3_PARAMETER_FORMULA_DOMAIN = (
    b"fisher_graph:layer-multimode:d3-parameter-formula:v1\0"
)
_D3_TRACE_PROTOCOL_DOMAIN = (
    b"fisher_graph:layer-multimode:d3-trace-protocol:v1\0"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _detached_atlas_support_error(error: BaseException) -> BaseException:
    """Rebuild a scalar support error without retaining fitter frames."""

    state_method = getattr(error, "state_dict", None)
    if not callable(state_method):
        raise TypeError("atlas support error lacks structured scalar state")
    state = state_method()
    if type(state) is not dict or "support_failure_stage" not in state:
        raise TypeError("atlas support error state is malformed")
    expected_stage = state.pop("support_failure_stage")
    detached = type(error)(**state)
    if getattr(detached, "support_failure_stage", None) != expected_stage:
        raise RuntimeError("detached atlas support stage drifted")
    if any(
        value is not None
        for value in (
            detached.__traceback__,
            detached.__context__,
            detached.__cause__,
        )
    ):
        raise RuntimeError("detached atlas support error retained exception frames")
    return detached


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_catalog(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _encode_prompt_ids(tokenizer: object, prompt: str) -> tuple[int, ...]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    encoder = getattr(backend if backend is not None else tokenizer, "encode", None)
    if not callable(encoder):
        raise TypeError("tokenizer must expose encode()")
    encoded = encoder(prompt, add_special_tokens=True)
    values = getattr(encoded, "ids", encoded)
    if not isinstance(values, (tuple, list)) or not values or any(
        type(value) is not int or value < 0 for value in values
    ):
        raise TypeError("tokenizer returned invalid token ids")
    return tuple(values)


def _fresh_batches(
    panel: Mapping[str, object],
    *,
    tokenizer: object,
) -> tuple[tuple[tuple[Mapping[str, object], ...], object], ...]:
    """Return eight equal-length ephemeral token batches; persist no ids."""

    import torch

    raw = panel.get("examples")
    if not isinstance(raw, list) or len(raw) != 48:
        raise ValueError("fresh role must contain exactly 48 examples")
    by_length: dict[int, list[tuple[Mapping[str, object], tuple[int, ...]]]] = {}
    for value in raw:
        if not isinstance(value, Mapping):
            raise TypeError("fresh example is invalid")
        prompt = value.get("prompt")
        target = value.get("target_token_count")
        if not isinstance(prompt, str) or type(target) is not int:
            raise TypeError("fresh example prompt/length is invalid")
        ids = _encode_prompt_ids(tokenizer, prompt)
        if len(ids) != target:
            raise RuntimeError("evaluated tokenization differs from authenticated role")
        by_length.setdefault(target, []).append((value, ids))
    schedule = (96, 511, 512, 513, 514, 576, 640, 768)
    if tuple(sorted(by_length)) != schedule:
        raise ValueError("fresh target-length axis drifted")
    result: list[tuple[tuple[Mapping[str, object], ...], object]] = []
    for target in schedule:
        rows = by_length[target]
        if len(rows) != 6:
            raise ValueError("fresh length batch must contain six families")
        result.append(
            (
                tuple(value[0] for value in rows),
                torch.tensor(
                    [value[1] for value in rows],
                    dtype=torch.long,
                    device="cpu",
                ),
            )
        )
    return tuple(result)


def _backbone_last_hidden(model: object, input_ids: object) -> object:
    import torch

    backbone = getattr(model, "model", None)
    if backbone is None:
        raise TypeError("Gemma model omitted its backbone")
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = torch.arange(
        input_ids.shape[1], dtype=torch.long, device=input_ids.device
    )[None, :].expand(input_ids.shape[0], -1)
    with torch.no_grad():
        output = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
    hidden = getattr(output, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor):
        if isinstance(output, (tuple, list)) and output:
            hidden = output[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("Gemma backbone did not return terminal hidden states")
    return hidden


def _tensor_output(value: object) -> object:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(
        value[0], torch.Tensor
    ):
        return value[0]
    raise TypeError("module output did not expose a tensor")


def _readout(model: object) -> tuple[object, float | None]:
    import torch

    callback = getattr(model, "get_output_embeddings", None)
    module = callback() if callable(callback) else getattr(model, "lm_head", None)
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("Gemma model omitted its tied readout weight")
    raw = getattr(getattr(model, "config", None), "final_logit_softcapping", None)
    softcap = None if raw is None else float(raw)
    if softcap is not None and softcap <= 0.0:
        raise ValueError("final logit softcap must be positive")
    return weight, softcap


def _apply_softcap(logits: object, softcap: float | None) -> object:
    if softcap is None:
        return logits
    import torch

    return torch.tanh(logits / softcap) * softcap


def _chunked_full_vocab_pair_metric_sums(
    *,
    native_hidden: object,
    candidate_hidden: object,
    native_weight: object,
    candidate_weight: object,
    target_token_ids: object,
    example_row_counts: tuple[int, ...],
    native_softcap: float | None,
    candidate_softcap: float | None,
    vocabulary_chunk_size: int = 1024,
) -> tuple[tuple[float, float, int], ...]:
    """Exact full-vocabulary pair metrics with float64 accumulation.

    The vocabulary is chunked only as an algebraically exact partition of the
    full support.  Ties retain the first (therefore lowest) vocabulary index.
    Returned values are whole-example additive sums, never token-level data.
    """

    import torch

    tensors = (
        native_hidden,
        candidate_hidden,
        native_weight,
        candidate_weight,
        target_token_ids,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("full-vocabulary metric inputs must be tensors")
    if (
        native_hidden.ndim != 2
        or candidate_hidden.shape != native_hidden.shape
        or native_weight.ndim != 2
        or candidate_weight.shape != native_weight.shape
        or native_weight.shape[1] != native_hidden.shape[1]
        or target_token_ids.shape != (native_hidden.shape[0],)
        or sum(example_row_counts) != native_hidden.shape[0]
        or type(vocabulary_chunk_size) is not int
        or vocabulary_chunk_size <= 0
    ):
        raise ValueError("full-vocabulary metric tensor axes drifted")
    vocabulary = int(native_weight.shape[0])
    if bool((target_token_ids < 0).any()) or bool(
        (target_token_ids >= vocabulary).any()
    ):
        raise ValueError("target token is outside the full vocabulary")
    rows = int(native_hidden.shape[0])
    device = native_hidden.device
    native64 = native_hidden.to(dtype=torch.float64)
    candidate64 = candidate_hidden.to(dtype=torch.float64)
    logz_native = torch.full((rows,), -torch.inf, dtype=torch.float64, device=device)
    logz_candidate = torch.full_like(logz_native, -torch.inf)
    target_native = torch.empty_like(logz_native)
    target_candidate = torch.empty_like(logz_native)
    native_max = torch.full_like(logz_native, -torch.inf)
    candidate_max = torch.full_like(logz_native, -torch.inf)
    native_argmax = torch.zeros((rows,), dtype=torch.long, device=device)
    candidate_argmax = torch.zeros_like(native_argmax)
    targets = target_token_ids.to(device=device, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, vocabulary, vocabulary_chunk_size):
            end = min(vocabulary, start + vocabulary_chunk_size)
            native_logits = _apply_softcap(
                native64 @ native_weight[start:end].to(dtype=torch.float64).T,
                native_softcap,
            )
            candidate_logits = _apply_softcap(
                candidate64
                @ candidate_weight[start:end].to(dtype=torch.float64).T,
                candidate_softcap,
            )
            logz_native = torch.logaddexp(
                logz_native,
                torch.logsumexp(native_logits, dim=-1),
            )
            logz_candidate = torch.logaddexp(
                logz_candidate,
                torch.logsumexp(candidate_logits, dim=-1),
            )
            native_values, native_indices = native_logits.max(dim=-1)
            candidate_values, candidate_indices = candidate_logits.max(dim=-1)
            native_better = native_values > native_max
            candidate_better = candidate_values > candidate_max
            native_max = torch.where(native_better, native_values, native_max)
            candidate_max = torch.where(
                candidate_better,
                candidate_values,
                candidate_max,
            )
            native_argmax = torch.where(
                native_better,
                native_indices + start,
                native_argmax,
            )
            candidate_argmax = torch.where(
                candidate_better,
                candidate_indices + start,
                candidate_argmax,
            )
            in_chunk = (targets >= start) & (targets < end)
            if bool(in_chunk.any()):
                local = targets[in_chunk] - start
                target_native[in_chunk] = native_logits[in_chunk, local]
                target_candidate[in_chunk] = candidate_logits[in_chunk, local]

        weighted_logit_difference = torch.zeros_like(logz_native)
        for start in range(0, vocabulary, vocabulary_chunk_size):
            end = min(vocabulary, start + vocabulary_chunk_size)
            native_logits = _apply_softcap(
                native64 @ native_weight[start:end].to(dtype=torch.float64).T,
                native_softcap,
            )
            candidate_logits = _apply_softcap(
                candidate64
                @ candidate_weight[start:end].to(dtype=torch.float64).T,
                candidate_softcap,
            )
            probability = torch.exp(native_logits - logz_native[:, None])
            weighted_logit_difference += (
                probability * (native_logits - candidate_logits)
            ).sum(dim=-1)
        delta_nll = (
            logz_candidate - target_candidate - logz_native + target_native
        )
        kl = weighted_logit_difference - logz_native + logz_candidate
        if bool((kl < -1e-10).any()):
            raise RuntimeError("full-vocabulary KL became materially negative")
        kl = kl.clamp_min(0.0)
        agreement = native_argmax == candidate_argmax

    result: list[tuple[float, float, int]] = []
    offset = 0
    for count in example_row_counts:
        end = offset + count
        result.append(
            (
                float(delta_nll[offset:end].sum().item()),
                float(kl[offset:end].sum().item()),
                int(agreement[offset:end].sum().item()),
            )
        )
        offset = end
    return tuple(result)


def _multimode_routing_additive(
    module: object,
    rows: object,
) -> tuple[object, object, object]:
    """Return additive sum(q), sum(q*pi), and sum(q*H(pi)) in float64."""

    import torch

    required = (
        "centers",
        "bases",
        "log_chart_temperature",
        "router_weight",
        "router_bias",
    )
    if not isinstance(rows, torch.Tensor) or any(
        not isinstance(getattr(module, name, None), torch.Tensor)
        for name in required
    ):
        raise TypeError("multimode routing inputs are invalid")
    with torch.no_grad():
        flat = rows.reshape(-1, rows.shape[-1])
        centered = flat[:, None, :] - module.centers[None, :, :]
        z = torch.einsum("nch,chr->ncr", centered, module.bases)
        normal_squared = (
            centered.square().sum(dim=-1) - z.square().sum(dim=-1)
        ).clamp_min(0.0)
        q = torch.softmax(
            -normal_squared / module.log_chart_temperature.exp(),
            dim=-1,
        )
        logits = (
            torch.einsum("ncr,cmr->ncm", z, module.router_weight)
            + module.router_bias[None, :, :]
        )
        pi = torch.softmax(logits, dim=-1)
        entropy = -(pi * torch.where(pi > 0, pi.log(), torch.zeros_like(pi))).sum(
            dim=-1
        )
        q64 = q.to(dtype=torch.float64)
        pi64 = pi.to(dtype=torch.float64)
        return (
            q64.sum(dim=0),
            (q64[:, :, None] * pi64).sum(dim=0),
            (q64 * entropy.to(dtype=torch.float64)).sum(dim=0),
        )


@dataclass(frozen=True, slots=True)
class _D3TemporaryArtifact:
    name: str
    device: int
    inode: int


@dataclass(slots=True)
class _D3ResourceDirectory:
    path: Path
    parent_descriptor: int
    directory_descriptor: int
    device: int
    inode: int
    artifacts: list[_D3TemporaryArtifact]


def _open_D3_resource_directory(directory: Path) -> _D3ResourceDirectory:
    """Anchor a fresh private directory by parent and child descriptors."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_descriptor = -1
    directory_descriptor = -1
    primary: BaseException | None = None
    try:
        parent_descriptor = os.open(directory.parent, flags)
        directory_descriptor = os.open(
            directory.name,
            flags,
            dir_fd=parent_descriptor,
        )
        observed = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError("D3 resource workspace is not a directory")
        return _D3ResourceDirectory(
            path=directory,
            parent_descriptor=parent_descriptor,
            directory_descriptor=directory_descriptor,
            device=observed.st_dev,
            inode=observed.st_ino,
            artifacts=[],
        )
    except BaseException as error:
        primary = error
    for descriptor in (directory_descriptor, parent_descriptor):
        if descriptor < 0:
            continue
        try:
            os.close(descriptor)
        except BaseException as close_error:
            assert primary is not None
            primary.add_note(
                "D3 resource workspace descriptor close also failed: "
                f"{type(close_error).__name__}: {close_error}"
            )
    assert primary is not None
    try:
        directory.rmdir()
    except BaseException as cleanup_error:
        primary.add_note(
            "D3 unopened resource workspace cleanup also failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
    raise primary


def _serialize_model_state_same_protocol(
    model: object,
    workspace: _D3ResourceDirectory,
    filename: str,
) -> tuple[str, int]:
    """Serialize through one exclusive dirfd-anchored regular file."""

    import torch

    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("D3 serialization filename must be one safe basename")
    source_state = model.state_dict()  # type: ignore[union-attr]
    state = {
        name: value.detach().cpu().clone().contiguous()
        for name, value in sorted(source_state.items())
    }
    del source_state
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    primary: BaseException | None = None
    result: tuple[str, int] | None = None
    try:
        descriptor = os.open(
            filename,
            flags,
            0o600,
            dir_fd=workspace.directory_descriptor,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError("D3 state target is not a single-link regular file")
        workspace.artifacts.append(
            _D3TemporaryArtifact(filename, opened.st_dev, opened.st_ino)
        )
        with os.fdopen(os.dup(descriptor), "w+b") as stream:
            torch.save(
                {"format_version": 1, "state_dict": state},
                stream,
                _use_new_zipfile_serialization=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        serialized = os.fstat(descriptor)
        if (
            not stat.S_ISREG(serialized.st_mode)
            or serialized.st_nlink != 1
            or (serialized.st_dev, serialized.st_ino)
            != (opened.st_dev, opened.st_ino)
            or serialized.st_size <= 0
        ):
            raise RuntimeError("D3 state serialization identity changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = serialized.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("D3 state serialization was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("D3 state serialization grew while hashing")
        result = (digest.hexdigest(), serialized.st_size)
    except BaseException as error:
        primary = error
    finally:
        del state
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as close_error:
                if primary is None:
                    primary = close_error
                else:
                    primary.add_note(
                        "D3 state descriptor close also failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
    if primary is not None:
        raise primary
    if result is None:
        raise AssertionError("D3 state serialization produced no result")
    return result


def _cleanup_D3_resource_directory(
    workspace: _D3ResourceDirectory,
) -> tuple[BaseException, ...]:
    """Identity-check and attempt every anchored D3 cleanup operation."""

    errors: list[BaseException] = []
    for artifact in workspace.artifacts:
        guard_descriptor = -1
        artifact_error: BaseException | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            named = os.stat(
                artifact.name,
                dir_fd=workspace.directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(named.st_mode)
                or named.st_nlink != 1
                or (named.st_dev, named.st_ino)
                != (artifact.device, artifact.inode)
            ):
                raise RuntimeError(
                    f"D3 cleanup preserved replaced artifact: {artifact.name}"
                )
            guard_descriptor = os.open(
                artifact.name,
                flags,
                dir_fd=workspace.directory_descriptor,
            )
            guarded = os.fstat(guard_descriptor)
            named_after_open = os.stat(
                artifact.name,
                dir_fd=workspace.directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(guarded.st_mode)
                or guarded.st_nlink != 1
                or (guarded.st_dev, guarded.st_ino)
                != (artifact.device, artifact.inode)
                or (
                    named_after_open.st_dev,
                    named_after_open.st_ino,
                    named_after_open.st_nlink,
                )
                != (guarded.st_dev, guarded.st_ino, guarded.st_nlink)
            ):
                raise RuntimeError(
                    f"D3 cleanup preserved replaced artifact: {artifact.name}"
                )
            quarantine_name = (
                f".{artifact.name}.{secrets.token_hex(16)}.cleanup"
            )
            os.rename(
                artifact.name,
                quarantine_name,
                src_dir_fd=workspace.directory_descriptor,
                dst_dir_fd=workspace.directory_descriptor,
            )
            moved = os.stat(
                quarantine_name,
                dir_fd=workspace.directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(moved.st_mode)
                or (moved.st_dev, moved.st_ino, moved.st_nlink)
                != (guarded.st_dev, guarded.st_ino, guarded.st_nlink)
            ):
                raise RuntimeError(
                    "D3 cleanup quarantined a replaced artifact: "
                    f"{artifact.name}"
                )
            os.unlink(
                quarantine_name,
                dir_fd=workspace.directory_descriptor,
            )
            removed = os.fstat(guard_descriptor)
            if (
                (removed.st_dev, removed.st_ino)
                != (guarded.st_dev, guarded.st_ino)
                or removed.st_nlink != guarded.st_nlink - 1
            ):
                raise RuntimeError(
                    f"D3 artifact unlink did not close: {artifact.name}"
                )
        except BaseException as error:
            artifact_error = error
        finally:
            if guard_descriptor >= 0:
                try:
                    os.close(guard_descriptor)
                except BaseException as close_error:
                    if artifact_error is None:
                        artifact_error = close_error
                    else:
                        artifact_error.add_note(
                            "D3 artifact guard close also failed: "
                            f"{type(close_error).__name__}: {close_error}"
                        )
        if artifact_error is not None:
            errors.append(artifact_error)
    try:
        os.fsync(workspace.directory_descriptor)
    except BaseException as error:
        errors.append(error)
    if errors:
        errors.append(
            RuntimeError("D3 resource workspace retained after cleanup failure")
        )
    else:
        try:
            guarded_directory = os.fstat(workspace.directory_descriptor)
            named_directory = os.stat(
                workspace.path.name,
                dir_fd=workspace.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(guarded_directory.st_mode)
                or (guarded_directory.st_dev, guarded_directory.st_ino)
                != (workspace.device, workspace.inode)
                or (named_directory.st_dev, named_directory.st_ino)
                != (guarded_directory.st_dev, guarded_directory.st_ino)
            ):
                raise RuntimeError(
                    "D3 cleanup preserved a replaced workspace"
                )
            quarantine_directory_name = (
                f".{workspace.path.name}.{secrets.token_hex(16)}.cleanup"
            )
            os.rename(
                workspace.path.name,
                quarantine_directory_name,
                src_dir_fd=workspace.parent_descriptor,
                dst_dir_fd=workspace.parent_descriptor,
            )
            moved_directory = os.stat(
                quarantine_directory_name,
                dir_fd=workspace.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(moved_directory.st_mode)
                or (moved_directory.st_dev, moved_directory.st_ino)
                != (guarded_directory.st_dev, guarded_directory.st_ino)
            ):
                raise RuntimeError(
                    "D3 cleanup quarantined a replaced workspace"
                )
            os.rmdir(
                quarantine_directory_name,
                dir_fd=workspace.parent_descriptor,
            )
            try:
                os.stat(
                    quarantine_directory_name,
                    dir_fd=workspace.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("D3 workspace name survived rmdir")
            removed_directory = os.fstat(workspace.directory_descriptor)
            if (
                (removed_directory.st_dev, removed_directory.st_ino)
                != (guarded_directory.st_dev, guarded_directory.st_ino)
                or not stat.S_ISDIR(removed_directory.st_mode)
                or removed_directory.st_nlink
                not in (0, guarded_directory.st_nlink)
            ):
                raise RuntimeError("D3 workspace rmdir did not close")
            os.fsync(workspace.parent_descriptor)
        except BaseException as error:
            errors.append(error)
    try:
        os.close(workspace.directory_descriptor)
        workspace.directory_descriptor = -1
    except BaseException as close_error:
        if errors:
            errors[-1].add_note(
                "D3 workspace guard close also failed: "
                f"{type(close_error).__name__}: {close_error}"
            )
        else:
            errors.append(close_error)
    try:
        os.close(workspace.parent_descriptor)
        workspace.parent_descriptor = -1
    except BaseException as close_error:
        if errors:
            errors[-1].add_note(
                "D3 workspace parent close also failed: "
                f"{type(close_error).__name__}: {close_error}"
            )
        else:
            errors.append(close_error)
    return tuple(errors)


def _execution_environment_sha256() -> str:
    """Fingerprint the concrete CPU execution policy used by v3."""

    import platform
    import sys

    import torch

    return _digest(
        b"fisher_graph:layer-multimode:execution-environment:v1\0",
        {
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "device": "cpu",
            "interop_thread_count": torch.get_num_interop_threads(),
            "intraop_thread_count": torch.get_num_threads(),
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
        },
    )


def _fresh_role_catalog_sha256s(
    *,
    manifest: Mapping[str, object],
    role: str,
) -> tuple[str, str, str, str]:
    """Hash only prompt-free frozen axes needed to replay one fresh role."""

    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise TypeError("fresh manifest role inventory is invalid")
    descriptor = roles.get(role)
    if not isinstance(descriptor, Mapping):
        raise TypeError("fresh role descriptor is invalid")
    members = descriptor.get("ordered_members")
    if not isinstance(members, list) or len(members) != 48:
        raise ValueError("fresh role member inventory drifted")
    row_offset = 0
    row_catalog: list[dict[str, object]] = []
    for member in members:
        if not isinstance(member, Mapping):
            raise TypeError("fresh role member is invalid")
        predicted = member.get("predicted_token_count")
        example_id = member.get("example_id")
        if type(predicted) is not int or predicted <= 0 or not isinstance(
            example_id,
            str,
        ):
            raise TypeError("fresh role row identity is invalid")
        row_catalog.append(
            {
                "example_id": example_id,
                "predicted_token_count": predicted,
                "row_start": row_offset,
                "row_stop": row_offset + predicted,
            }
        )
        row_offset += predicted
    if row_offset != 24_732:
        raise ValueError("fresh role row catalog does not cover 24,732 rows")
    family_ids = manifest.get("family_ids")
    target_counts = manifest.get("target_token_counts")
    if not isinstance(family_ids, list) or not isinstance(target_counts, list):
        raise TypeError("fresh family/batch axes are invalid")
    family_axis_sha256 = _sha256_catalog(family_ids)
    batch_axis_sha256 = _sha256_catalog(
        {
            "batch_order": target_counts,
            "example_order_within_batch": family_ids,
            "examples_per_batch": len(family_ids),
            "predicted_rows_per_example": [
                int(value) - 1 for value in target_counts
            ],
        }
    )
    return (
        _sha256_catalog(members),
        _sha256_catalog(row_catalog),
        family_axis_sha256,
        batch_axis_sha256,
    )


@dataclass(frozen=True, slots=True)
class _PreparedDevelopmentRelease:
    runtime_implementation_sha256: str
    source_model_sha256: str
    source_execution_fingerprint_sha256: str
    compiled_candidate_present: bool
    compilation_receipt_sha256: str | None
    compiled_model_fingerprint_sha256: str | None
    nonce_sha256: str
    token_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "runtime_implementation_sha256",
            "source_model_sha256",
            "source_execution_fingerprint_sha256",
            "nonce_sha256",
        ):
            experiment._require_sha256(getattr(self, name), label=name)
        if self.compiled_candidate_present:
            experiment._require_sha256(
                self.compilation_receipt_sha256,
                label="compilation_receipt_sha256",
            )
            experiment._require_sha256(
                self.compiled_model_fingerprint_sha256,
                label="compiled_model_fingerprint_sha256",
            )
        elif self.compilation_receipt_sha256 is not None or (
            self.compiled_model_fingerprint_sha256 is not None
        ):
            raise ValueError("A-negative release token claimed compilation")
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "token_sha256"
        }
        expected = _digest(_DEVELOPMENT_RELEASE_DOMAIN, payload)
        if self.token_sha256 == "":
            object.__setattr__(self, "token_sha256", expected)
        elif self.token_sha256 != expected:
            raise ValueError("development release token digest mismatch")


class _NativeBaselineEvaluator:
    """One immutable eval-only teacher owner with no compiler references."""

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        expected_model_sha256: str,
        expected_execution_sha256: str,
    ) -> None:
        from .adapters.gemma3 import Gemma3CausalLMAdapter

        self._model = model
        self._tokenizer = tokenizer
        self._adapter = Gemma3CausalLMAdapter(model)  # type: ignore[arg-type]
        self.model_sha256 = expected_model_sha256
        self.execution_sha256 = expected_execution_sha256
        self.forward_counts = {
            experiment.B3_ROLE: 0,
            experiment.C3_ROLE: 0,
            experiment.D3_ROLE: 0,
        }
        self.parameter_mutation_count = 0
        self.optimizer_reference_count = 0
        self._closed = False
        self.assert_unchanged()

    @property
    def model(self) -> object:
        if self._closed:
            raise RuntimeError("native baseline evaluator is closed")
        return self._model

    @property
    def tokenizer(self) -> object:
        if self._closed:
            raise RuntimeError("native baseline evaluator is closed")
        return self._tokenizer

    @property
    def adapter(self) -> object:
        if self._closed:
            raise RuntimeError("native baseline evaluator is closed")
        return self._adapter

    def assert_unchanged(self) -> None:
        if self._closed:
            raise RuntimeError("native baseline evaluator is closed")
        modules = tuple(self._model.modules())  # type: ignore[union-attr]
        parameters = tuple(self._model.parameters())  # type: ignore[union-attr]
        if any(module.training for module in modules) or any(
            parameter.requires_grad for parameter in parameters
        ):
            raise RuntimeError("native baseline left immutable eval mode")
        if (
            self._adapter.model_fingerprint() != self.model_sha256
            or self._adapter.execution_fingerprint() != self.execution_sha256
        ):
            self.parameter_mutation_count += 1
            raise RuntimeError("native baseline fingerprint changed")
        if self.optimizer_reference_count != 0:
            raise RuntimeError("native baseline acquired an optimizer reference")

    def record_forward(self, role: str) -> None:
        self.assert_unchanged()
        if role not in self.forward_counts:
            raise ValueError("native baseline role is invalid")
        self.forward_counts[role] += 1
        self.assert_unchanged()

    def close(self) -> None:
        if self._closed:
            return
        self.assert_unchanged()
        self._adapter = None
        self._model = None
        self._tokenizer = None
        self._closed = True


class _CompiledCandidateEvaluator:
    """Immutable source-free heldout evaluator with pre/post integrity checks."""

    def __init__(self, frozen: experiment.LayerMultimodeFrozenCandidate) -> None:
        from .adapters.gemma3 import Gemma3CausalLMAdapter
        from .gemma3_layer_multimode_compiler import (
            CompiledGemma3LayerMultimodeModel,
            Float32LayerMultimodeMLP,
        )

        self._model = frozen.compiled_model
        self._receipt = frozen.compilation_receipt
        self._adapter = Gemma3CausalLMAdapter(self._model)
        self.model_sha256 = self._receipt.compiled_state_sha256
        self.execution_sha256 = self._receipt.compiled_execution_fingerprint_sha256
        self.forward_counts = {
            experiment.C3_ROLE: 0,
            experiment.D3_ROLE: 0,
        }
        self.parameter_mutation_count = 0
        self._closed = False
        layers = tuple(self._model.model.layers)
        self.native_mlp_module_count = sum(
            type(layer.mlp) is not Float32LayerMultimodeMLP for layer in layers
        )
        self.native_mlp_projection_inventory_count = sum(
            name.rsplit(".", 1)[-1] in {"gate_proj", "up_proj", "down_proj"}
            for layer in layers
            for name, _module in layer.mlp.named_modules()
            if name
        )
        self.native_mlp_fallback_inventory_count = sum(
            getattr(layer.mlp, "native_mlp_fallback_present", None) is not False
            for layer in layers
        )
        if len(layers) != 18 or self.native_mlp_module_count != 0:
            raise RuntimeError("compiled candidate contains a native MLP")
        if (
            self.native_mlp_projection_inventory_count != 0
            or self.native_mlp_fallback_inventory_count != 0
        ):
            raise RuntimeError("compiled candidate retains a native MLP path")
        self._integrity_wrapper = CompiledGemma3LayerMultimodeModel(
            self._model,
            self._receipt,
        )
        self.assert_unchanged()

    @property
    def model(self) -> object:
        if self._closed:
            raise RuntimeError("compiled candidate evaluator is closed")
        return self._model

    def assert_unchanged(self) -> None:
        if self._closed:
            raise RuntimeError("compiled candidate evaluator is closed")
        try:
            self._integrity_wrapper.assert_integrity()
        except BaseException:
            self.parameter_mutation_count += 1
            raise
        if self._adapter.execution_fingerprint() != self.execution_sha256:
            self.parameter_mutation_count += 1
            raise RuntimeError("compiled candidate execution fingerprint changed")

    def record_forward(self, role: str) -> None:
        self.assert_unchanged()
        if role not in self.forward_counts:
            raise ValueError("compiled candidate role is invalid")
        self.forward_counts[role] += 1
        self.assert_unchanged()

    def close(self) -> None:
        if self._closed:
            return
        self.assert_unchanged()
        self._integrity_wrapper = None
        self._adapter = None
        self._receipt = None
        self._model = None
        self._closed = True


@dataclass(frozen=True, slots=True)
class LayerMultimodeRuntimeQualificationReceipt:
    execution_preflight_receipt_sha256: str
    implementation_qualification_sha256: str
    implementation_qualification_file_sha256: str
    runtime_implementation_sha256: str
    qualified_runtime_source_sha256: str
    predecessor_preflight_receipt_sha256: str
    predecessor_runtime_qualification_receipt_sha256: str
    source_model_fingerprint_before_sha256: str
    source_model_fingerprint_after_sha256: str
    source_execution_fingerprint_before_sha256: str
    source_execution_fingerprint_after_sha256: str
    tokenizer_asset_catalog_sha256: str
    evaluation_protocol_sha256: str
    source_model_mutated: bool
    full_vocab_accumulator_dtype: str = "float64"
    top1_tie_break: str = "lowest_vocabulary_index"
    predecessor_A_exact_control_batch_count: int = 45
    predecessor_A_dense_parity_count: int = 1
    fresh_batches_per_opened_role: int = 8
    logits_persisted: bool = False
    token_ids_persisted: bool = False
    prompt_text_persisted: bool = False
    receipt_sha256: str = ""
    schema: str = RUNTIME_QUALIFICATION_SCHEMA
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_QUALIFICATION_SCHEMA or self.format_version != 1:
            raise ValueError("runtime qualification header is invalid")
        for name in (
            "execution_preflight_receipt_sha256",
            "implementation_qualification_sha256",
            "implementation_qualification_file_sha256",
            "runtime_implementation_sha256",
            "qualified_runtime_source_sha256",
            "predecessor_preflight_receipt_sha256",
            "predecessor_runtime_qualification_receipt_sha256",
            "source_model_fingerprint_before_sha256",
            "source_model_fingerprint_after_sha256",
            "source_execution_fingerprint_before_sha256",
            "source_execution_fingerprint_after_sha256",
            "tokenizer_asset_catalog_sha256",
            "evaluation_protocol_sha256",
        ):
            experiment._require_sha256(getattr(self, name), label=name)
        if self.source_model_mutated is not False:
            raise ValueError("runtime qualification requires an immutable source")
        if (
            self.qualified_runtime_source_sha256
            != self.runtime_implementation_sha256
        ):
            raise ValueError(
                "runtime source is not bound to implementation qualification"
            )
        if (
            self.source_model_fingerprint_before_sha256
            != self.source_model_fingerprint_after_sha256
            or self.source_execution_fingerprint_before_sha256
            != self.source_execution_fingerprint_after_sha256
        ):
            raise ValueError("runtime qualification source fingerprints changed")
        if (
            self.full_vocab_accumulator_dtype != "float64"
            or self.top1_tie_break != "lowest_vocabulary_index"
            or self.predecessor_A_exact_control_batch_count != 45
            or self.predecessor_A_dense_parity_count != 1
            or self.fresh_batches_per_opened_role != 8
            or self.logits_persisted is not False
            or self.token_ids_persisted is not False
            or self.prompt_text_persisted is not False
        ):
            raise ValueError("runtime qualification numeric/privacy policy drifted")
        expected = _digest(RUNTIME_QUALIFICATION_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif self.receipt_sha256 != expected:
            raise ValueError("runtime qualification digest mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "receipt_sha256"
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PredecessorARecaptureComparisonReceipt:
    predecessor_capture_receipt_sha256: str
    recaptured_capture_receipt_sha256: str
    predecessor_runtime_execution_receipt_sha256: str
    predecessor_runtime_session_binding_sha256: str
    recaptured_runtime_session_binding_sha256: str
    role: str
    batch_count: int
    row_count: int
    role_panel_binding_sha256: str
    ordered_objective_row_catalog_sha256: str
    score_objective_receipt_sha256: str
    dense_lexical_parity_receipt_sha256: str
    mode_raw_artifact_catalog_sha256: str
    mode_semantic_projection_catalog_sha256: str
    lexical_nll_receipt_catalog_sha256: str
    supplemental_route_projection_catalog_sha256: str
    compared_field_inventory: tuple[str, ...]
    intentionally_session_dependent_field_inventory: tuple[str, ...]
    runtime_sessions_expected_different: bool
    all_comparable_fields_matched: bool
    artifact_sha256: str = ""
    schema: str = A_RECAPTURE_COMPARISON_SCHEMA
    format_version: int = 1

    @property
    def receipt_sha256(self) -> str:
        return self.artifact_sha256

    def __post_init__(self) -> None:
        for name in (
            "predecessor_capture_receipt_sha256",
            "recaptured_capture_receipt_sha256",
            "predecessor_runtime_execution_receipt_sha256",
            "predecessor_runtime_session_binding_sha256",
            "recaptured_runtime_session_binding_sha256",
            "role_panel_binding_sha256",
            "ordered_objective_row_catalog_sha256",
            "score_objective_receipt_sha256",
            "dense_lexical_parity_receipt_sha256",
            "mode_raw_artifact_catalog_sha256",
            "mode_semantic_projection_catalog_sha256",
            "lexical_nll_receipt_catalog_sha256",
            "supplemental_route_projection_catalog_sha256",
        ):
            experiment._require_sha256(getattr(self, name), label=name)
        expected_compared = (
            "batch_count",
            "dense_lexical_parity_receipt_sha256",
            "lexical_nll_receipt_sha256s",
            "mode_raw_artifacts",
            "mode_semantic_projection",
            "ordered_objective_row_catalog_sha256",
            "role",
            "role_panel_binding_sha256",
            "row_count",
            "score_objective_receipt_sha256",
            "supplemental_route_projection",
        )
        expected_excluded = (
            "batch_axis_catalog_sha256",
            "example_capture_receipt_sha256s",
            "execution_receipt_sha256s",
            "receipt_sha256",
            "runtime_session_binding_sha256",
            "supplemental_route_receipt_sha256s",
        )
        if (
            self.role != "A_chart_transport_fit"
            or self.batch_count != 45
            or self.row_count != 24_732
            or self.compared_field_inventory != expected_compared
            or self.intentionally_session_dependent_field_inventory
            != expected_excluded
            or self.runtime_sessions_expected_different is not True
            or self.predecessor_runtime_session_binding_sha256
            == self.recaptured_runtime_session_binding_sha256
            or self.all_comparable_fields_matched is not True
        ):
            raise ValueError("predecessor A recapture comparison did not close")
        expected = _digest(A_RECAPTURE_COMPARISON_DOMAIN, self._payload())
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", expected)
        elif self.artifact_sha256 != expected:
            raise ValueError("predecessor A comparison digest mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "artifact_sha256"
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _compare_predecessor_A(
    *,
    frozen: Mapping[str, object],
    recaptured: object,
    predecessor_runtime_execution_receipt_sha256: str,
) -> PredecessorARecaptureComparisonReceipt:
    current = recaptured.payload_without_sha256()
    stable_scalars = (
        "role",
        "batch_count",
        "row_count",
        "role_panel_binding_sha256",
        "ordered_objective_row_catalog_sha256",
        "score_objective_receipt_sha256",
        "dense_lexical_parity_receipt_sha256",
    )
    mismatched = tuple(
        name for name in stable_scalars if frozen.get(name) != current.get(name)
    )
    def semantic_mode_projection(value: object) -> tuple[dict[str, object], ...]:
        if not isinstance(value, (list, tuple)) or len(value) != 38:
            raise ValueError("predecessor A mode inventory must contain 38 rows")
        rows: list[dict[str, object]] = []
        for item in value:
            if isinstance(item, Mapping):
                mode = item.get("mode_id")
                row_catalog = item.get("row_catalog_sha256")
                tensor_catalog = item.get("tensor_catalog_sha256")
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                mode, _artifact, row_catalog, tensor_catalog = item
            else:
                raise TypeError("predecessor A mode artifact row is invalid")
            if not isinstance(mode, str) or not mode:
                raise TypeError("predecessor A mode id is invalid")
            experiment._require_sha256(row_catalog, label=f"{mode} row catalog")
            experiment._require_sha256(
                tensor_catalog,
                label=f"{mode} tensor catalog",
            )
            rows.append(
                {
                    "mode_id": mode,
                    "row_catalog_sha256": row_catalog,
                    "tensor_catalog_sha256": tensor_catalog,
                }
            )
        expected_modes = (
            "mode.lexical.ingress",
            *(f"mode.mutation.{index}" for index in range(1, 37)),
            "mode.lexical.egress",
        )
        if tuple(row["mode_id"] for row in rows) != expected_modes:
            raise ValueError("predecessor A mode inventory axis drifted")
        return tuple(rows)

    old_projection = semantic_mode_projection(frozen.get("mode_raw_artifacts"))
    new_projection = semantic_mode_projection(current.get("mode_raw_artifacts"))
    if old_projection != new_projection:
        mismatched += ("mode_semantic_projection",)
    old_mode_artifacts = frozen.get("mode_raw_artifacts")
    new_mode_artifacts = current.get("mode_raw_artifacts")
    if old_mode_artifacts != new_mode_artifacts:
        mismatched += ("mode_raw_artifacts",)
    old_lexical = frozen.get("lexical_nll_receipt_sha256s")
    new_lexical = current.get("lexical_nll_receipt_sha256s")
    if old_lexical != new_lexical:
        mismatched += ("lexical_nll_receipt_sha256s",)

    def route_projection(value: object) -> tuple[tuple[str, str, str], ...]:
        if not isinstance(value, (list, tuple)) or len(value) != 864:
            raise ValueError("predecessor A route inventory must contain 864 rows")
        rows: list[tuple[str, str, str]] = []
        for item in value:
            if isinstance(item, Mapping):
                mode = item.get("mode_id")
                example = item.get("example_id")
                catalog = item.get("route_catalog_sha256")
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                mode, example, catalog, _receipt = item
            else:
                raise TypeError("predecessor A route evidence row is invalid")
            if not isinstance(mode, str) or not isinstance(example, str):
                raise TypeError("predecessor A route identity is invalid")
            experiment._require_sha256(catalog, label="route catalog")
            rows.append((mode, example, catalog))
        return tuple(rows)

    old_routes = route_projection(frozen.get("supplemental_route_evidence"))
    new_routes = route_projection(current.get("supplemental_route_evidence"))
    if old_routes != new_routes:
        mismatched += ("supplemental_route_projection",)
    if mismatched:
        raise RuntimeError(
            "predecessor A recapture scalar lineage drifted: "
            + ", ".join(mismatched)
        )
    return PredecessorARecaptureComparisonReceipt(
        predecessor_capture_receipt_sha256=str(frozen["receipt_sha256"]),
        recaptured_capture_receipt_sha256=recaptured.receipt_sha256,
        predecessor_runtime_execution_receipt_sha256=(
            predecessor_runtime_execution_receipt_sha256
        ),
        predecessor_runtime_session_binding_sha256=str(
            frozen["runtime_session_binding_sha256"]
        ),
        recaptured_runtime_session_binding_sha256=str(
            current["runtime_session_binding_sha256"]
        ),
        role=str(current["role"]),
        batch_count=int(current["batch_count"]),
        row_count=int(current["row_count"]),
        role_panel_binding_sha256=str(current["role_panel_binding_sha256"]),
        ordered_objective_row_catalog_sha256=str(
            current["ordered_objective_row_catalog_sha256"]
        ),
        score_objective_receipt_sha256=str(
            current["score_objective_receipt_sha256"]
        ),
        dense_lexical_parity_receipt_sha256=str(
            current["dense_lexical_parity_receipt_sha256"]
        ),
        mode_raw_artifact_catalog_sha256=_sha256_catalog(new_mode_artifacts),
        mode_semantic_projection_catalog_sha256=_sha256_catalog(new_projection),
        lexical_nll_receipt_catalog_sha256=_sha256_catalog(new_lexical),
        supplemental_route_projection_catalog_sha256=_sha256_catalog(new_routes),
        compared_field_inventory=(
            "batch_count",
            "dense_lexical_parity_receipt_sha256",
            "lexical_nll_receipt_sha256s",
            "mode_raw_artifacts",
            "mode_semantic_projection",
            "ordered_objective_row_catalog_sha256",
            "role",
            "role_panel_binding_sha256",
            "row_count",
            "score_objective_receipt_sha256",
            "supplemental_route_projection",
        ),
        intentionally_session_dependent_field_inventory=(
            "batch_axis_catalog_sha256",
            "example_capture_receipt_sha256s",
            "execution_receipt_sha256s",
            "receipt_sha256",
            "runtime_session_binding_sha256",
            "supplemental_route_receipt_sha256s",
        ),
        runtime_sessions_expected_different=True,
        all_comparable_fields_matched=True,
    )


class Gemma3LayerMultimodeRuntime(experiment.LayerMultimodeV3Runtime):
    """Production runtime; construction itself does not load a model."""

    test_only_nonproduction = False

    def __init__(
        self,
        preflight: experiment.LayerMultimodeExecutionPreflightReceipt,
        *,
        repository_root: Path | str,
        cache_dir: Path | str | None = None,
    ) -> None:
        if type(preflight) is not experiment.LayerMultimodeExecutionPreflightReceipt:
            raise TypeError("runtime requires the exact v3 preflight receipt")
        if preflight.test_only_nonproduction or not preflight.live_preflight_authorized:
            raise RuntimeError("production runtime requires original live authority")
        self.preflight = preflight
        self.repository_root = Path(repository_root).resolve()
        if not self.repository_root.is_dir() or self.repository_root.is_symlink():
            raise ValueError("runtime repository root is missing or unsafe")
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.runtime_implementation_sha256 = _sha256_file(Path(__file__))
        # These prompt-free authorities are strict-loaded after v3 preflight
        # but before any old runtime/model import.  They are never injectable.
        from . import gemma3_chart_native_neighbor_v2_report as predecessor
        from . import gemma3_layer_multimode_implementation_qualification as iq
        from . import gemma3_layer_multimode_setup as setup

        report_path = self.repository_root / setup.PREDECESSOR_V2_REPORT
        if _sha256_file(report_path) != setup.PREDECESSOR_V2_REPORT_FILE_SHA256:
            raise RuntimeError("predecessor v2 report file differs from setup pin")
        predecessor_report = predecessor.strict_load_v2_experiment_report(
            report_path,
            repository_root=self.repository_root,
        )
        runtime_state = predecessor_report["runtime_execution_receipt"]
        capture_state = predecessor_report["A_fit"]["capture_receipt"]
        if (
            predecessor_report["report_sha256"]
            != preflight.predecessor_v2_report_sha256
            or capture_state["receipt_sha256"]
            != preflight.predecessor_A_capture_receipt_sha256
            or runtime_state["runtime_receipt_sha256"]
            != preflight.predecessor_runtime_execution_receipt_sha256
            or runtime_state["source_model_fingerprint_before_sha256"]
            != preflight.source_model_sha256
            or runtime_state["source_model_fingerprint_after_sha256"]
            != preflight.source_model_sha256
            or predecessor_report["v2_preflight"]["v2_setup_sha256"]
            != preflight.predecessor_v2_setup_sha256
        ):
            raise RuntimeError(
                "predecessor v2 report lineage differs from v3 preflight"
            )
        qualification_path = self.repository_root / iq.DEFAULT_OUTPUT
        qualification_raw_sha = _sha256_file(qualification_path)
        qualification = iq.load_qualification(
            qualification_path,
            repository_root=self.repository_root,
        )
        runtime_rows = tuple(
            row
            for row in qualification["source_closure"]
            if row["module"]
            == "fisher_graph.gemma3_layer_multimode_runtime"
        )
        if (
            qualification_raw_sha
            != preflight.implementation_qualification_file_sha256
            or qualification["qualification_sha256"]
            != preflight.implementation_qualification_sha256
            or len(runtime_rows) != 1
            or runtime_rows[0]["file_sha256"]
            != self.runtime_implementation_sha256
        ):
            raise RuntimeError("runtime source is outside implementation qualification")
        self.predecessor_preflight: object | None = None
        self.predecessor_report = predecessor_report
        self.implementation_qualification = qualification
        self._capture_runtime: object | None = None
        self._native_baseline: _NativeBaselineEvaluator | None = None
        self._compiled_evaluator: _CompiledCandidateEvaluator | None = None
        self._native_model_weakref: weakref.ReferenceType[object] | None = None
        self._native_tokenizer_weakref: weakref.ReferenceType[object] | None = None
        self._compiled_model_weakref: weakref.ReferenceType[object] | None = None
        self._predecessor_qualification: object | None = None
        self._runtime_qualification: (
            LayerMultimodeRuntimeQualificationReceipt | None
        ) = None
        self._A_runtime_receipt: (
            experiment.LayerMultimodePredecessorARuntimeReceipt | None
        ) = None
        self._A_comparison_receipt: (
            PredecessorARecaptureComparisonReceipt | None
        ) = None
        self._A_raw_directory: Path | None = None
        self._captured_A: object | None = None
        self._bridge: object | None = None
        self._partition: object | None = None
        self._frozen_candidate: experiment.LayerMultimodeFrozenCandidate | None = None
        self._pending_development_release: _PreparedDevelopmentRelease | None = None
        self._development_teardown_confirmed = False
        self._tokenization_receipts: dict[str, object] = {}
        self._authenticated_panel_ids: dict[str, int] = {}
        self._fresh_manifests: dict[str, Mapping[str, object]] = {}
        self._fresh_role_capture_receipts: dict[
            str, experiment.LayerMultimodeFreshRoleCaptureReceipt
        ] = {}
        self._execution_environment_sha256: str | None = None
        self._runtime_session_binding_sha256: str | None = None
        self._compiled_model_fingerprint_before_sha256: str | None = None
        self._compiled_execution_fingerprint_before_sha256: str | None = None
        self._selected_candidate_sha256s: tuple[str, ...] | None = None
        self._B3_candidate_layer_velocity_forward_count = 0
        self._fresh_batch_counts = {
            experiment.B3_ROLE: 0,
            experiment.C3_ROLE: 0,
            experiment.D3_ROLE: 0,
        }
        self._full_stack_example_aggregates: dict[str, tuple[object, ...]] = {}
        self._D3_compiled_layer_call_counts = [0] * 18
        self._D3_trace_events: list[tuple[int, int]] = []
        self._A_recapture_attempt_count = 0
        self._A_recapture_completed = False
        self._issued_runtime_execution_receipt: (
            experiment.LayerMultimodeRuntimeExecutionReceipt | None
        ) = None
        self._closed = False

    def recapture_exposed_predecessor_A(
        self,
        *,
        preflight: experiment.LayerMultimodeExecutionPreflightReceipt,
        workspace: Path,
    ) -> experiment.ADevelopmentRecapture:
        if (
            self._closed
            or preflight is not self.preflight
            or self._A_recapture_attempt_count != 0
        ):
            raise RuntimeError("predecessor A may be recaptured exactly once")
        self._A_recapture_attempt_count = 1
        try:
            result = self._recapture_exposed_predecessor_A_once(
                workspace=workspace,
            )
        except BaseException as primary_error:
            cleanup_errors: list[BaseException] = []
            if self._capture_runtime is not None:
                try:
                    self._capture_runtime.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            self._capture_runtime = None
            if self._native_baseline is not None:
                try:
                    self._native_baseline.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            self._native_baseline = None
            self._native_model_weakref = None
            self._native_tokenizer_weakref = None
            self._compiled_model_weakref = None
            self._predecessor_qualification = None
            self._runtime_qualification = None
            self._A_runtime_receipt = None
            self._A_comparison_receipt = None
            self._captured_A = None
            self._bridge = None
            self._partition = None
            if self._A_raw_directory is not None:
                try:
                    if self._A_raw_directory.exists():
                        shutil.rmtree(self._A_raw_directory)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            self._A_raw_directory = None
            for cleanup_error in cleanup_errors:
                primary_error.add_note(
                    "A recapture cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        self._A_recapture_completed = True
        return result

    def _recapture_exposed_predecessor_A_once(
        self,
        *,
        workspace: Path,
    ) -> experiment.ADevelopmentRecapture:
        from . import gemma3_chart_native_neighbor_experiment as predecessor

        binding = predecessor.build_default_execution_binding(
            repository_root=self.repository_root,
            output_directory=(
                self.repository_root
                / ".local-runs/google--gemma-3-270m"
            ),
            temp_directory=workspace,
        )
        predecessor_preflight = predecessor.static_preflight(binding)
        predecessor_preflight.validate_receipt()
        if (
            predecessor_preflight.setup["setup_sha256"]
            != self.predecessor_report["preflight"]["setup_sha256"]
            or predecessor_preflight.fresh_manifest["manifest_sha256"]
            != self.predecessor_report["preflight"]["fresh_manifest_sha256"]
            or predecessor_preflight.fresh_receipt["receipt_sha256"]
            != self.predecessor_report["preflight"]["fresh_receipt_sha256"]
        ):
            raise RuntimeError(
                "live predecessor A authority differs from frozen report"
            )
        self.predecessor_preflight = predecessor_preflight

        from .gemma3_chart_native_neighbor_fresh_panels import (
            SequentialPanelSource,
            authenticate_tokenizer_assets,
        )
        from .gemma3_chart_native_neighbor_runtime import (
            build_pinned_gemma3_chart_native_capture_runtime,
        )
        from .gemma3_experiment import resolve_gemma3_huggingface_paths
        from .gemma3_layer_multimode_capture import (
            build_A_development_partition_receipt,
            open_full_A_layer_mlp_spool_bridge,
        )

        revision = str(predecessor_preflight.setup["source_model"]["model_revision"])
        cache = resolve_gemma3_huggingface_paths(self.cache_dir)["hub_cache"]
        snapshot = (
            cache
            / "models--google--gemma-3-270m"
            / "snapshots"
            / revision
        )
        tokenizer_assets = authenticate_tokenizer_assets(snapshot)
        runtime = build_pinned_gemma3_chart_native_capture_runtime(
            predecessor_preflight,
            cache_dir=self.cache_dir,
        )
        # Acquire ownership immediately; every later failure is cleaned by the
        # one-shot wrapper and can never authorize another A attempt.
        self._capture_runtime = runtime
        qualification = runtime.runtime_qualification_receipt()
        predecessor.validate_runtime_qualification(
            preflight=predecessor_preflight,
            receipt=qualification,
        )
        panel_source = SequentialPanelSource(
            repository_root=predecessor_preflight.binding.repository_root,
            manifest=predecessor_preflight.fresh_manifest,
            receipt=predecessor_preflight.fresh_receipt,
            preflight=predecessor_preflight.opaque_panel_receipt,
        )
        A_corpus = panel_source.open_role(predecessor.FIT_ROLE)
        A_manifest = predecessor_preflight.fresh_manifest["roles"][
            predecessor.FIT_ROLE
        ]
        parity_sha = runtime.run_dense_lexical_parity_smoke(
            preflight=predecessor_preflight,
            raw_example=A_corpus["examples"][0],
            role_manifest=A_manifest,
        )
        raw_directory = workspace / "A-recaptured-raw"
        self._A_raw_directory = raw_directory
        raw_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        captured = predecessor.capture_role_to_raw_spools(
            preflight=self.predecessor_preflight,
            runtime=runtime,
            role=predecessor.FIT_ROLE,
            raw_corpus=A_corpus,
            destination_directory=raw_directory,
            dense_lexical_parity_receipt_sha256=parity_sha,
        )
        frozen_A = self.predecessor_report["A_fit"]["capture_receipt"]
        comparison = _compare_predecessor_A(
            frozen=frozen_A,
            recaptured=captured.receipt,
            predecessor_runtime_execution_receipt_sha256=(
                self.preflight.predecessor_runtime_execution_receipt_sha256
            ),
        )
        mlp_mode_ids = tuple(
            f"mode.mutation.{2 + 2 * layer}" for layer in range(18)
        )
        by_mode = {spool.mode_id: spool for spool in captured.raw_spools}
        bridge = open_full_A_layer_mlp_spool_bridge(
            capture_receipt=captured.receipt,
            mlp_spools=tuple(by_mode[value] for value in mlp_mode_ids),
            role_manifest=A_manifest,
            expected_capture_receipt_sha256=captured.receipt.receipt_sha256,
            expected_runtime_session_binding_sha256=(
                runtime.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                captured.receipt.role_panel_binding_sha256
            ),
        )
        partition = build_A_development_partition_receipt(
            bridge_receipt=bridge.receipt,
            role_manifest=A_manifest,
            expected_bridge_receipt_sha256=bridge.receipt.receipt_sha256,
        )
        adapter = getattr(runtime, "_adapter", None)
        native_model = getattr(adapter, "module", None)
        tokenizer = getattr(runtime, "_tokenizer", None)
        if native_model is None or tokenizer is None:
            raise RuntimeError(
                "predecessor runtime did not expose its pinned evaluator"
            )
        baseline = _NativeBaselineEvaluator(
            model=native_model,
            tokenizer=tokenizer,
            expected_model_sha256=(
                qualification.source_model_fingerprint_after_compilation_sha256
            ),
            expected_execution_sha256=(
                qualification.source_execution_fingerprint_after_compilation_sha256
            ),
        )
        self._native_baseline = baseline
        self._native_model_weakref = weakref.ref(native_model)
        self._native_tokenizer_weakref = weakref.ref(tokenizer)
        self._runtime_qualification = LayerMultimodeRuntimeQualificationReceipt(
            execution_preflight_receipt_sha256=self.preflight.receipt_sha256,
            implementation_qualification_sha256=(
                self.preflight.implementation_qualification_sha256
            ),
            implementation_qualification_file_sha256=(
                self.preflight.implementation_qualification_file_sha256
            ),
            runtime_implementation_sha256=self.runtime_implementation_sha256,
            qualified_runtime_source_sha256=self.runtime_implementation_sha256,
            predecessor_preflight_receipt_sha256=(
                predecessor_preflight.receipt_sha256
            ),
            predecessor_runtime_qualification_receipt_sha256=(
                qualification.qualification_receipt_sha256
            ),
            source_model_fingerprint_before_sha256=(
                qualification.source_model_fingerprint_before_compilation_sha256
            ),
            source_model_fingerprint_after_sha256=(
                qualification.source_model_fingerprint_after_compilation_sha256
            ),
            source_execution_fingerprint_before_sha256=(
                qualification.source_execution_fingerprint_before_compilation_sha256
            ),
            source_execution_fingerprint_after_sha256=(
                qualification.source_execution_fingerprint_after_compilation_sha256
            ),
            tokenizer_asset_catalog_sha256=_sha256_catalog(tokenizer_assets),
            evaluation_protocol_sha256=self.preflight.evaluation_protocol_sha256,
            source_model_mutated=False,
        )
        observed = runtime.counters
        capture_receipt = captured.receipt
        self._A_runtime_receipt = (
            experiment.LayerMultimodePredecessorARuntimeReceipt(
            execution_preflight_receipt_sha256=self.preflight.receipt_sha256,
            predecessor_preflight_receipt_sha256=(
                predecessor_preflight.receipt_sha256
            ),
            predecessor_runtime_qualification_receipt_sha256=(
                qualification.qualification_receipt_sha256
            ),
            source_capture_receipt_sha256=capture_receipt.receipt_sha256,
            runtime_session_binding_sha256=runtime.runtime_session_binding_sha256,
            source_model_fingerprint_before_sha256=(
                qualification.source_model_fingerprint_before_compilation_sha256
            ),
            source_model_fingerprint_after_sha256=(
                qualification.source_model_fingerprint_after_compilation_sha256
            ),
            source_execution_fingerprint_before_sha256=(
                qualification.source_execution_fingerprint_before_compilation_sha256
            ),
            source_execution_fingerprint_after_sha256=(
                qualification.source_execution_fingerprint_after_compilation_sha256
            ),
            role_batch_counts=observed.role_batch_counts,
            exact_control_forward_count=observed.chart_backbone_forward_count,
            exact_control_vjp_count=(
                observed.chart_terminal_cotangent_vjp_count
            ),
            dense_parity_smoke_count=observed.dense_parity_smoke_count,
            dense_parity_forward_count=observed.parity_backbone_forward_count,
            dense_parity_vjp_count=(
                observed.parity_terminal_cotangent_vjp_count
            ),
            total_backbone_forward_count=(
                observed.chart_backbone_forward_count
                + observed.parity_backbone_forward_count
            ),
            total_terminal_cotangent_vjp_count=(
                observed.chart_terminal_cotangent_vjp_count
                + observed.parity_terminal_cotangent_vjp_count
            ),
            captured_example_count=observed.captured_example_count,
            supplemental_route_receipt_count=(
                observed.supplemental_route_receipt_count
            ),
            mode_raw_artifact_count=len(capture_receipt.mode_raw_artifacts),
            lexical_nll_receipt_count=len(
                capture_receipt.lexical_nll_receipt_sha256s
            ),
            execution_receipt_count=len(
                capture_receipt.execution_receipt_sha256s
            ),
            example_capture_receipt_count=len(
                capture_receipt.example_capture_receipt_sha256s
            ),
            canonical_gradient_targets_per_vjp=(
                observed.canonical_gradient_targets_per_vjp
            ),
            auxiliary_gradient_targets_per_vjp=(
                observed.auxiliary_gradient_targets_per_vjp
            ),
                gradient_targets_per_vjp=observed.gradient_targets_per_vjp,
            )
        )
        self._A_comparison_receipt = comparison
        self._predecessor_qualification = qualification
        self._captured_A = captured
        self._bridge = bridge
        self._partition = partition
        self.predecessor_report = {}
        return experiment.ADevelopmentRecapture(
            bridge=bridge,
            partition_receipt=partition,
            source_model=baseline.model,
            source_model_sha256=(
                qualification.source_model_fingerprint_after_compilation_sha256
            ),
            source_capture_receipt_sha256=captured.receipt.receipt_sha256,
            predecessor_capture_receipt_sha256=(
                self.preflight.predecessor_A_capture_receipt_sha256
            ),
            predecessor_lineage_comparison_receipt=comparison,
            predecessor_A_runtime_receipt=self._A_runtime_receipt,
        )

    def materialize_A_layer(self, recapture, layer_ordinal):
        from .gemma3_layer_multimode_materialization import (
            materialize_layer_mlp_flow_rows,
        )

        result = materialize_layer_mlp_flow_rows(
            recapture.bridge.open_layer(layer_ordinal),
            a_development_partition_receipt=recapture.partition_receipt,
            runtime_qualification_receipt=self._predecessor_qualification,
            predecessor_preflight_receipt_sha256=(
                self.predecessor_preflight.receipt_sha256
            ),
        )
        return experiment.LayerMaterialization(
            layer_ordinal=layer_ordinal,
            rows=result.rows,
            receipt=result.receipt,
        )

    def build_materialization_inventory(self, recapture, layers):
        from .gemma3_layer_multimode_materialization import (
            build_layer_mlp_flow_materialization_inventory,
        )

        return build_layer_mlp_flow_materialization_inventory(
            tuple(value.receipt for value in layers),
            a_development_partition_receipt=recapture.partition_receipt,
        )

    def fit_chart_atlas(self, materialization, *, chart_count, tangent_rank):
        from .gemma3_layer_multimode_fit import (
            AtlasSupportError,
            fit_layer_chart_atlas,
        )

        try:
            fitted = fit_layer_chart_atlas(
                rows=materialization.rows,
                chart_count=chart_count,
                tangent_rank=tangent_rank,
            )
        except AtlasSupportError as error:
            if type(error) is not AtlasSupportError:
                raise
            return experiment.LayerMultimodePreoptimizerAttempt(
                result=None,
                support_error=_detached_atlas_support_error(error),
            )
        return experiment.LayerMultimodePreoptimizerAttempt(
            result=fitted,
            support_error=None,
        )

    def fit_candidate(
        self,
        materialization,
        *,
        chart_count,
        mode_count,
        tangent_rank,
        atlas,
    ):
        from .gemma3_layer_multimode_fit import fit_layer_multimode_candidate

        return fit_layer_multimode_candidate(
            rows=materialization.rows,
            chart_count=chart_count,
            modes_per_chart=mode_count,
            tangent_rank=tangent_rank,
            atlas_fit=atlas,
        )

    def evaluate_split_refit(
        self,
        materialization,
        *,
        chart_count,
        mode_count,
        tangent_rank,
    ):
        from .gemma3_layer_multimode_fit import (
            AtlasSupportError,
            evaluate_layer_multimode_split_refit,
        )

        try:
            evidence = evaluate_layer_multimode_split_refit(
                rows=materialization.rows,
                chart_count=chart_count,
                modes_per_chart=mode_count,
                tangent_rank=tangent_rank,
            )
        except AtlasSupportError as error:
            if type(error) is not AtlasSupportError:
                raise
            return experiment.LayerMultimodePreoptimizerAttempt(
                result=None,
                support_error=_detached_atlas_support_error(error),
            )
        return experiment.LayerMultimodePreoptimizerAttempt(
            result=evidence,
            support_error=None,
        )

    def lower_candidate_support_rejection(
        self,
        *,
        materialization,
        chart_count,
        mode_count,
        tangent_rank,
        candidate_fit,
        support_error,
        recapture,
    ):
        from .gemma3_layer_multimode_fit import (
            layer_multimode_candidate_support_rejection_from_error,
        )

        partition = recapture.partition_receipt
        return layer_multimode_candidate_support_rejection_from_error(
            rows=materialization.rows,
            chart_count=chart_count,
            modes_per_chart=mode_count,
            tangent_rank=tangent_rank,
            candidate_fit=candidate_fit,
            source_model_sha256=recapture.source_model_sha256,
            fit_a_partition_sha256=partition.fit_partition_sha256,
            validation_a_partition_sha256=partition.validation_partition_sha256,
            evaluation_protocol_sha256=self.preflight.evaluation_protocol_sha256,
            support_error=support_error,
        )

    def lower_candidate_evaluation(self, *, candidate_fit, split_refit, recapture):
        from .gemma3_layer_multimode_fit import (
            layer_multimode_candidate_evaluation_from_fit,
        )

        partition = recapture.partition_receipt
        return layer_multimode_candidate_evaluation_from_fit(
            candidate_fit,
            source_model_sha256=recapture.source_model_sha256,
            fit_a_partition_sha256=partition.fit_partition_sha256,
            validation_a_partition_sha256=partition.validation_partition_sha256,
            evaluation_protocol_sha256=self.preflight.evaluation_protocol_sha256,
            split_refit=split_refit,
        )

    def select_full_grid(self, evaluations):
        from .gemma3_layer_multimode_selection import (
            select_layer_multimode_candidates,
        )

        return select_layer_multimode_candidates(evaluations)

    def build_selected_inventory(
        self,
        *,
        recapture,
        materialization_inventory,
        selection,
        selected_fits,
        selected_splits,
    ):
        from .gemma3_layer_multimode_compiler import (
            build_selected_layer_multimode_coefficient_inventory,
        )

        return build_selected_layer_multimode_coefficient_inventory(
            selection,
            selected_fits,
            a_development_partition_receipt=recapture.partition_receipt,
            layer_materialization_inventory=materialization_inventory,
            split_refit_evidence=selected_splits,
        )

    def compile_source_free(self, *, source_model, selected_inventory):
        from .gemma3_layer_multimode_compiler import (
            compile_gemma3_layer_multimode_model,
        )

        return compile_gemma3_layer_multimode_model(
            source_model,
            selected_inventory,
        )

    def freeze_compiled_candidate(self, *, compiled, selection, workspace):
        from .gemma3_layer_multimode_compiler import (
            load_compiled_layer_multimode_bundle,
            save_compiled_layer_multimode_bundle,
        )
        from .gemma3_layer_multimode_evaluation import (
            CompiledCandidateIdentityReceipt,
        )

        path = workspace / "layer-multimode-candidate.pt"
        bundle = save_compiled_layer_multimode_bundle(compiled, path)
        loaded = load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=bundle.bundle_sha256,
            model_factory=lambda: copy.deepcopy(self._native_baseline.model),
        )
        compiler_protocol = hashlib.sha256(
            b"fisher_graph:layer-multimode:compiler-source:v1\0"
            + Path(__file__)
            .with_name("gemma3_layer_multimode_compiler.py")
            .read_bytes()
        ).hexdigest()
        identity = CompiledCandidateIdentityReceipt(
            selection_receipt_sha256=selection.receipt_sha256,
            compiler_protocol_sha256=compiler_protocol,
            compile_receipt_sha256=loaded.receipt.receipt_sha256,
            bundle_semantic_sha256=bundle.manifest_sha256,
            compiled_model_state_sha256=loaded.receipt.compiled_state_sha256,
        )
        frozen = experiment.LayerMultimodeFrozenCandidate(
            compiled_model=loaded.model,
            candidate_identity=identity,
            bundle_receipt=bundle,
            compilation_receipt=loaded.receipt,
            bundle_path=str(path),
            bundle_file_sha256=bundle.bundle_sha256,
        )
        selected_hashes = tuple(
            str(value.selected_candidate_sha256) for value in selection.layers
        )
        if len(selected_hashes) != 18:
            raise RuntimeError("frozen selection does not cover 18 layers")
        self._selected_candidate_sha256s = selected_hashes
        self._frozen_candidate = frozen
        self._compiled_evaluator = _CompiledCandidateEvaluator(frozen)
        self._compiled_model_weakref = weakref.ref(frozen.compiled_model)
        self._seal_runtime_session()
        return frozen

    def _seal_runtime_session(self) -> str:
        if self._runtime_session_binding_sha256 is not None:
            return self._runtime_session_binding_sha256
        if (
            self._runtime_qualification is None
            or self._native_baseline is None
            or self._compiled_evaluator is None
        ):
            raise RuntimeError("runtime session cannot seal before compilation")
        self._native_baseline.assert_unchanged()
        self._compiled_evaluator.assert_unchanged()
        self._execution_environment_sha256 = _execution_environment_sha256()
        self._compiled_model_fingerprint_before_sha256 = (
            self._compiled_evaluator.model_sha256
        )
        self._compiled_execution_fingerprint_before_sha256 = (
            self._compiled_evaluator.execution_sha256
        )
        qualification = self._runtime_qualification
        self._runtime_session_binding_sha256 = (
            experiment.layer_multimode_runtime_session_binding_sha256(
                runtime_implementation_sha256=(
                    self.runtime_implementation_sha256
                ),
                implementation_qualification_sha256=(
                    self.preflight.implementation_qualification_sha256
                ),
                execution_environment_sha256=(
                    self._execution_environment_sha256
                ),
                tokenizer_fingerprint_sha256=(
                    qualification.tokenizer_asset_catalog_sha256
                ),
                runtime_qualification_receipt_sha256=(
                    qualification.receipt_sha256
                ),
                source_model_fingerprint_before_sha256=(
                    qualification.source_model_fingerprint_before_sha256
                ),
                source_model_fingerprint_after_sha256=(
                    qualification.source_model_fingerprint_after_sha256
                ),
                source_execution_fingerprint_before_sha256=(
                    qualification.source_execution_fingerprint_before_sha256
                ),
                source_execution_fingerprint_after_sha256=(
                    qualification.source_execution_fingerprint_after_sha256
                ),
                compiled_model_fingerprint_before_sha256=(
                    self._compiled_model_fingerprint_before_sha256
                ),
                compiled_model_fingerprint_after_sha256=(
                    self._compiled_model_fingerprint_before_sha256
                ),
                compiled_execution_fingerprint_before_sha256=(
                    self._compiled_execution_fingerprint_before_sha256
                ),
                compiled_execution_fingerprint_after_sha256=(
                    self._compiled_execution_fingerprint_before_sha256
                ),
                compiled_candidate_mutated=False,
            )
        )
        return self._runtime_session_binding_sha256

    def release_development_state(
        self,
        *,
        recapture,
        materializations,
        atlases,
        candidate_fits,
        split_refits,
        compiled,
    ):
        if self._closed or self._pending_development_release is not None or (
            self._development_teardown_confirmed
        ):
            raise RuntimeError("development state may be released exactly once")
        del materializations, atlases, candidate_fits, split_refits
        source_execution = (
            self._runtime_qualification.source_execution_fingerprint_after_sha256
        )
        compiled_present = compiled is not None
        compilation_sha = (
            None if not compiled_present else compiled.receipt.receipt_sha256
        )
        compiled_model_sha = (
            None if not compiled_present else compiled.receipt.compiled_state_sha256
        )
        if self._capture_runtime is not None:
            self._capture_runtime.close()
        self._capture_runtime = None
        self._captured_A = None
        self._bridge = None
        self._partition = None
        if self._A_raw_directory is not None and self._A_raw_directory.exists():
            shutil.rmtree(self._A_raw_directory)
        self._A_raw_directory = None
        prepared = _PreparedDevelopmentRelease(
            runtime_implementation_sha256=self.runtime_implementation_sha256,
            source_model_sha256=recapture.source_model_sha256,
            source_execution_fingerprint_sha256=source_execution,
            compiled_candidate_present=compiled_present,
            compilation_receipt_sha256=compilation_sha,
            compiled_model_fingerprint_sha256=compiled_model_sha,
            nonce_sha256=hashlib.sha256(os.urandom(32)).hexdigest(),
        )
        self._pending_development_release = prepared
        return prepared

    def confirm_development_teardown(self, prepared_release):
        if (
            type(prepared_release) is not _PreparedDevelopmentRelease
            or prepared_release is not self._pending_development_release
            or self._development_teardown_confirmed
        ):
            raise RuntimeError("development teardown token is not live")
        if any(
            value is not None
            for value in (
                self._capture_runtime,
                self._captured_A,
                self._bridge,
                self._partition,
                self._A_raw_directory,
            )
        ):
            raise RuntimeError("runtime still owns A development aliases")
        self._pending_development_release = None
        self._development_teardown_confirmed = True
        return experiment.LayerMultimodeDevelopmentTeardownReceipt(
            source_model_sha256=prepared_release.source_model_sha256,
            source_execution_fingerprint_sha256=(
                prepared_release.source_execution_fingerprint_sha256
            ),
            compiled_candidate_present=(
                prepared_release.compiled_candidate_present
            ),
            compilation_receipt_sha256=(
                prepared_release.compilation_receipt_sha256
            ),
            compiled_model_fingerprint_sha256=(
                prepared_release.compiled_model_fingerprint_sha256
            ),
            source_compilation_aliases_released=True,
            native_baseline_retained_for_heldout_evaluation=True,
            A_activation_rows_released=True,
            A_fit_models_released=True,
            A_optimizer_state_released=True,
            source_compilation_native_mlp_alias_count=0,
            native_baseline_instance_count=1,
        )

    def authenticate_fresh_panel_tokenization(
        self,
        *,
        panel_source,
        panel,
        role,
        selection_receipt_sha256,
        frozen_candidate_identity_sha256,
    ):
        from .gemma3_layer_multimode_fresh_panels import (
            SequentialPanelSource,
            authenticate_open_role_tokenization,
        )

        if (
            self._closed
            or not self._development_teardown_confirmed
            or role in self._tokenization_receipts
            or type(panel_source) is not SequentialPanelSource
            or self._native_baseline is None
            or self._compiled_evaluator is None
            or self._frozen_candidate is None
        ):
            raise RuntimeError("fresh role tokenization may authenticate once")
        expected_order = (
            experiment.B3_ROLE,
            experiment.C3_ROLE,
            experiment.D3_ROLE,
        )
        if role != expected_order[len(self._tokenization_receipts)]:
            raise RuntimeError("fresh tokenization role order drifted")
        candidate_identity = self._frozen_candidate.candidate_identity
        expected_candidate_identity_sha256 = experiment._artifact_sha256(
            candidate_identity,
            label="frozen candidate identity",
        )
        if (
            panel_source.current_role != role
            or panel_source.opened_roles[-1:] != (role,)
            or panel_source.selection_receipt_sha256
            != selection_receipt_sha256
            or candidate_identity.selection_receipt_sha256
            != selection_receipt_sha256
            or frozen_candidate_identity_sha256
            != expected_candidate_identity_sha256
            or panel_source.panel_manifest_sha256
            != self.preflight.panel_manifest_sha256
        ):
            raise RuntimeError("fresh role panel/candidate authority drifted")
        runtime_session_sha256 = self._seal_runtime_session()
        receipt = authenticate_open_role_tokenization(
            panel,
            role=role,
            manifest=panel_source.manifest,
            tokenizer=self._native_baseline.tokenizer,
        )
        if (
            receipt.role != role
            or receipt.example_count != 48
            or receipt.predicted_token_count != 24_732
        ):
            raise RuntimeError("fresh tokenization receipt inventory drifted")
        manifest = panel_source.manifest
        roles = manifest.get("roles")
        descriptor = None if not isinstance(roles, Mapping) else roles.get(role)
        if not isinstance(descriptor, Mapping):
            raise TypeError("fresh role descriptor is invalid")
        role_file_sha256 = descriptor.get("file_sha256")
        experiment._require_sha256(
            role_file_sha256,
            label=f"{role} role file",
        )
        (
            example_catalog_sha256,
            row_catalog_sha256,
            family_axis_sha256,
            batch_axis_sha256,
        ) = _fresh_role_catalog_sha256s(manifest=manifest, role=role)
        capture = experiment.LayerMultimodeFreshRoleCaptureReceipt(
            role=role,
            stage_kind=experiment.FRESH_STAGE_KIND_BY_ROLE[role],
            panel_manifest_sha256=self.preflight.panel_manifest_sha256,
            panel_manifest_file_sha256=(
                self.preflight.panel_manifest_file_sha256
            ),
            panel_receipt_sha256=self.preflight.panel_receipt_sha256,
            panel_receipt_file_sha256=(
                self.preflight.panel_receipt_file_sha256
            ),
            panel_bundle_sha256=self.preflight.panel_bundle_sha256,
            role_file_sha256=role_file_sha256,
            selection_receipt_sha256=selection_receipt_sha256,
            frozen_candidate_identity_sha256=(
                frozen_candidate_identity_sha256
            ),
            runtime_session_binding_sha256=runtime_session_sha256,
            ordered_tokenization_receipt_sha256=(
                receipt.ordered_example_tokenization_sha256
            ),
            ordered_example_identity_catalog_sha256=(
                example_catalog_sha256
            ),
            ordered_row_identity_catalog_sha256=row_catalog_sha256,
            family_axis_sha256=family_axis_sha256,
            batch_axis_sha256=batch_axis_sha256,
            example_count=receipt.example_count,
            batch_count=8,
            predicted_next_token_position_count=(
                receipt.predicted_token_count
            ),
        )
        self._tokenization_receipts[role] = receipt
        self._authenticated_panel_ids[role] = id(panel)
        self._fresh_manifests[role] = manifest
        self._fresh_role_capture_receipts[role] = capture
        return capture

    def _require_authenticated_panel(
        self,
        panel: object,
        role: str,
        *,
        candidate: experiment.LayerMultimodeFrozenCandidate,
        fresh_role_capture_receipt: (
            experiment.LayerMultimodeFreshRoleCaptureReceipt
        ),
    ) -> object:
        receipt = self._tokenization_receipts.get(role)
        capture = self._fresh_role_capture_receipts.get(role)
        if (
            receipt is None
            or self._authenticated_panel_ids.get(role) != id(panel)
            or type(fresh_role_capture_receipt)
            is not experiment.LayerMultimodeFreshRoleCaptureReceipt
            or capture is not fresh_role_capture_receipt
            or candidate is not self._frozen_candidate
            or fresh_role_capture_receipt.role != role
            or fresh_role_capture_receipt.runtime_session_binding_sha256
            != self._runtime_session_binding_sha256
            or fresh_role_capture_receipt.selection_receipt_sha256
            != candidate.candidate_identity.selection_receipt_sha256
            or fresh_role_capture_receipt.frozen_candidate_identity_sha256
            != experiment._artifact_sha256(
                candidate.candidate_identity,
                label="frozen candidate identity",
            )
            or fresh_role_capture_receipt.ordered_tokenization_receipt_sha256
            != receipt.ordered_example_tokenization_sha256
        ):
            raise RuntimeError("evaluator did not receive its authenticated panel")
        from .gemma3_layer_multimode_fresh_panels import (
            authenticate_open_role_tokenization,
        )

        replayed = authenticate_open_role_tokenization(
            panel,
            role=role,
            manifest=self._fresh_manifests[role],
            tokenizer=self._native_baseline.tokenizer,
        )
        if replayed != receipt:
            raise RuntimeError("authenticated fresh panel mutated before evaluation")
        return receipt

    def evaluate_b3_teacher_forced(
        self,
        *,
        panel,
        candidate,
        fresh_role_capture_receipt,
    ):
        import math
        import torch

        from .gemma3_layer_multimode_evaluation import (
            LayerTeacherForcedAdditiveEvidence,
            LayerTeacherForcedFamilyAggregate,
            LayerTeacherForcedModeAggregate,
            evaluate_b3_teacher_forced,
        )

        role = experiment.B3_ROLE
        self._require_authenticated_panel(
            panel,
            role,
            candidate=candidate,
            fresh_role_capture_receipt=fresh_role_capture_receipt,
        )
        capture = self._fresh_role_capture_receipts.get(role)
        if capture is None:
            raise RuntimeError("B3 role capture receipt was not sealed")
        if candidate is not self._frozen_candidate:
            raise RuntimeError("B3 candidate differs from the frozen deployment")
        if self._native_baseline is None or self._compiled_evaluator is None:
            raise RuntimeError("heldout evaluators are unavailable")
        self._native_baseline.assert_unchanged()
        self._compiled_evaluator.assert_unchanged()
        model = self._native_baseline.model
        native_layers = tuple(model.model.layers)
        compiled_layers = tuple(candidate.compiled_model.model.layers)
        if len(native_layers) != 18 or len(compiled_layers) != 18:
            raise RuntimeError("B3 requires the exact 18-layer axis")
        family_stats: list[dict[str, list[float | int]]] = [
            {} for _ in range(18)
        ]
        chart_sums: list[object | None] = [None] * 18
        mode_sums: list[object | None] = [None] * 18
        entropy_sums: list[object | None] = [None] * 18
        row_counts = [0] * 18
        batches = _fresh_batches(
            panel,
            tokenizer=self._native_baseline.tokenizer,
        )
        for examples, input_ids in batches:
            inputs: list[object | None] = [None] * 18
            teachers: list[object | None] = [None] * 18
            handles: list[object] = []
            for ordinal, layer in enumerate(native_layers):
                handles.append(
                    layer.pre_feedforward_layernorm.register_forward_hook(
                        lambda _module, _args, output, current=ordinal: (
                            inputs.__setitem__(
                                current,
                                _tensor_output(output).detach(),
                            )
                        )
                    )
                )
                handles.append(
                    layer.post_feedforward_layernorm.register_forward_hook(
                        lambda _module, _args, output, current=ordinal: (
                            teachers.__setitem__(
                                current,
                                _tensor_output(output).detach(),
                            )
                        )
                    )
                )
            try:
                _backbone_last_hidden(model, input_ids)
            finally:
                for handle in reversed(handles):
                    handle.remove()
            self._native_baseline.record_forward(role)
            self._fresh_batch_counts[role] += 1
            rows_per_example = int(input_ids.shape[1]) - 1
            for ordinal, (x_value, teacher_value, compiled_layer) in enumerate(
                zip(inputs, teachers, compiled_layers, strict=True)
            ):
                if not isinstance(x_value, torch.Tensor) or not isinstance(
                    teacher_value,
                    torch.Tensor,
                ):
                    raise RuntimeError("B3 native hook inventory did not close")
                x = x_value[:, :-1, :].contiguous()
                teacher = teacher_value[:, :-1, :].contiguous()
                with torch.no_grad():
                    prediction = compiled_layer.mlp(x)
                self._B3_candidate_layer_velocity_forward_count += 1
                q_sum, qpi_sum, entropy_sum = _multimode_routing_additive(
                    compiled_layer.mlp,
                    x,
                )
                chart_sums[ordinal] = (
                    q_sum
                    if chart_sums[ordinal] is None
                    else chart_sums[ordinal] + q_sum
                )
                mode_sums[ordinal] = (
                    qpi_sum
                    if mode_sums[ordinal] is None
                    else mode_sums[ordinal] + qpi_sum
                )
                entropy_sums[ordinal] = (
                    entropy_sum
                    if entropy_sums[ordinal] is None
                    else entropy_sums[ordinal] + entropy_sum
                )
                row_counts[ordinal] += int(x.shape[0] * x.shape[1])
                for example_index, example in enumerate(examples):
                    family = str(example["family_id"])
                    current = family_stats[ordinal].setdefault(
                        family,
                        [0, 0.0, 0.0, 0.0, 0.0],
                    )
                    predicted_rows = prediction[example_index].to(torch.float64)
                    teacher_rows = teacher[example_index].to(torch.float64)
                    difference = predicted_rows - teacher_rows
                    current[0] = int(current[0]) + rows_per_example * int(
                        teacher_rows.shape[-1]
                    )
                    current[1] = float(current[1]) + float(
                        difference.square().sum().item()
                    )
                    current[2] = float(current[2]) + float(
                        teacher_rows.square().sum().item()
                    )
                    current[3] = float(current[3]) + float(
                        predicted_rows.square().sum().item()
                    )
                    current[4] = float(current[4]) + float(
                        (predicted_rows * teacher_rows).sum().item()
                    )
        if self._fresh_batch_counts[role] != 8:
            raise RuntimeError("B3 did not execute exactly eight native batches")
        capture_sha = experiment._artifact_sha256(
            capture,
            label="B3 role capture",
        )
        evidence: list[object] = []
        for ordinal, compiled_layer in enumerate(compiled_layers):
            families = tuple(
                LayerTeacherForcedFamilyAggregate(
                    family_id=family,
                    scalar_count=int(values[0]),
                    prediction_sse=float(values[1]),
                    teacher_energy=float(values[2]),
                    prediction_energy=float(values[3]),
                    prediction_teacher_dot=float(values[4]),
                )
                for family, values in sorted(family_stats[ordinal].items())
            )
            q_sum = chart_sums[ordinal]
            qpi_sum = mode_sums[ordinal]
            entropy_sum = entropy_sums[ordinal]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (q_sum, qpi_sum, entropy_sum)
            ):
                raise RuntimeError("B3 routing aggregates did not close")
            modes = tuple(
                LayerTeacherForcedModeAggregate(
                    chart_ordinal=chart,
                    chart_weight_sum=float(q_sum[chart].item()),
                    chart_mode_weight_sums=tuple(
                        float(value)
                        for value in qpi_sum[chart].tolist()
                    ),
                    weighted_router_entropy_sum_nats=(
                        0.0
                        if compiled_layer.mlp.modes_per_chart == 1
                        else float(entropy_sum[chart].item())
                    ),
                )
                for chart in range(compiled_layer.mlp.chart_count)
            )
            evidence.append(
                LayerTeacherForcedAdditiveEvidence(
                    layer_ordinal=ordinal,
                    chart_count=compiled_layer.mlp.chart_count,
                    dynamic_mode_count=compiled_layer.mlp.modes_per_chart,
                    tangent_rank=compiled_layer.mlp.tangent_rank,
                    selected_candidate_sha256=(
                        self._selected_candidate_sha256s[ordinal]
                    ),
                    role_capture_receipt_sha256=capture_sha,
                    family_aggregates=families,
                    mode_aggregates=modes,
                    row_weight_sum=float(row_counts[ordinal]),
                )
            )
        del batches
        self._native_baseline.assert_unchanged()
        self._compiled_evaluator.assert_unchanged()
        return evaluate_b3_teacher_forced(
            evidence,
            source_model_sha256=self._native_baseline.model_sha256,
            selection_receipt_sha256=(
                candidate.candidate_identity.selection_receipt_sha256
            ),
            panel_manifest_sha256=self._fresh_manifests[role]["manifest_sha256"],
            evaluation_protocol_sha256=self.preflight.evaluation_protocol_sha256,
        )

    def evaluate_c3_full_stack(
        self,
        *,
        panel,
        candidate,
        b3_evaluation,
        fresh_role_capture_receipt,
    ):
        from .gemma3_layer_multimode_evaluation import evaluate_c3_full_stack

        role = experiment.C3_ROLE
        records = self._evaluate_full_stack_examples(
            panel=panel,
            candidate=candidate,
            role=role,
            fresh_role_capture_receipt=fresh_role_capture_receipt,
        )
        result = evaluate_c3_full_stack(
            records,
            b3_evaluation=b3_evaluation,
            source_model_sha256=self._native_baseline.model_sha256,
            panel_manifest_sha256=self._fresh_manifests[role]["manifest_sha256"],
            role_capture_receipt_sha256=experiment._artifact_sha256(
                self._fresh_role_capture_receipts[role],
                label="C3 role capture",
            ),
            evaluation_protocol_sha256=self.preflight.evaluation_protocol_sha256,
            candidate_identity=candidate.candidate_identity,
        )
        self._full_stack_example_aggregates[role] = tuple(records)
        return result

    def _evaluate_full_stack_examples(
        self,
        *,
        panel,
        candidate,
        role,
        fresh_role_capture_receipt,
    ):
        import torch

        from .gemma3_layer_multimode_evaluation import FullStackExampleAggregate

        self._require_authenticated_panel(
            panel,
            role,
            candidate=candidate,
            fresh_role_capture_receipt=fresh_role_capture_receipt,
        )
        capture = self._fresh_role_capture_receipts.get(role)
        if capture is None:
            raise RuntimeError(f"{role} capture receipt was not sealed")
        if (
            candidate is not self._frozen_candidate
            or self._native_baseline is None
            or self._compiled_evaluator is None
        ):
            raise RuntimeError("full-stack evaluator identity drifted")
        self._native_baseline.assert_unchanged()
        self._compiled_evaluator.assert_unchanged()
        native_weight, native_softcap = _readout(self._native_baseline.model)
        candidate_weight, candidate_softcap = _readout(
            self._compiled_evaluator.model
        )
        capture_sha = experiment._artifact_sha256(
            capture,
            label=f"{role} role capture",
        )
        records: list[object] = []
        batches = _fresh_batches(
            panel,
            tokenizer=self._native_baseline.tokenizer,
        )
        for batch_index, (examples, input_ids) in enumerate(batches):
            native_hidden = _backbone_last_hidden(
                self._native_baseline.model,
                input_ids,
            )
            self._native_baseline.record_forward(role)
            handles: list[object] = []
            if role == experiment.D3_ROLE:
                for ordinal, layer in enumerate(
                    self._compiled_evaluator.model.model.layers
                ):
                    handles.append(
                        layer.mlp.register_forward_hook(
                            lambda _module, _args, _output, current=ordinal: (
                                self._D3_compiled_layer_call_counts.__setitem__(
                                    current,
                                    self._D3_compiled_layer_call_counts[current] + 1,
                                )
                            )
                        )
                    )
            try:
                candidate_hidden = _backbone_last_hidden(
                    self._compiled_evaluator.model,
                    input_ids,
                )
            finally:
                for handle in reversed(handles):
                    handle.remove()
            self._compiled_evaluator.record_forward(role)
            self._fresh_batch_counts[role] += 1
            if role == experiment.D3_ROLE:
                self._D3_trace_events.append((batch_index, int(input_ids.shape[0])))
            row_count = int(input_ids.shape[1]) - 1
            native_rows = native_hidden[:, :-1, :].reshape(
                -1,
                native_hidden.shape[-1],
            )
            candidate_rows = candidate_hidden[:, :-1, :].reshape(
                -1,
                candidate_hidden.shape[-1],
            )
            targets = input_ids[:, 1:].reshape(-1)
            sums = _chunked_full_vocab_pair_metric_sums(
                native_hidden=native_rows,
                candidate_hidden=candidate_rows,
                native_weight=native_weight,
                candidate_weight=candidate_weight,
                target_token_ids=targets,
                example_row_counts=(row_count,) * len(examples),
                native_softcap=native_softcap,
                candidate_softcap=candidate_softcap,
            )
            for example, (delta, kl, agreement) in zip(
                examples,
                sums,
                strict=True,
            ):
                identity = _digest(
                    _FRESH_EXAMPLE_IDENTITY_DOMAIN,
                    {
                        "role": role,
                        "example_id": example["example_id"],
                        "family_id": example["family_id"],
                        "hard_long": example["hard_long"],
                        "target_token_count": example["target_token_count"],
                        "predicted_token_count": example["predicted_token_count"],
                        "token_ids_sha256": example["token_ids_sha256"],
                        "semantic_lineage_sha256": (
                            example["semantic_lineage_sha256"]
                        ),
                    },
                )
                execution_sha = _digest(
                    _FRESH_EXAMPLE_EXECUTION_DOMAIN,
                    {
                        "candidate_identity_sha256": (
                            candidate.candidate_identity.artifact_sha256
                        ),
                        "delta_nll_sum_nats": delta,
                        "example_identity_sha256": identity,
                        "native_to_candidate_kl_sum_nats": kl,
                        "predicted_token_count": row_count,
                        "role_capture_receipt_sha256": capture_sha,
                        "runtime_session_binding_sha256": (
                            capture.runtime_session_binding_sha256
                        ),
                        "top1_agreement_count": agreement,
                    },
                )
                records.append(
                    FullStackExampleAggregate(
                        role=role,
                        example_identity_sha256=identity,
                        family_id=str(example["family_id"]),
                        hard_long=bool(example["hard_long"]),
                        predicted_token_count=row_count,
                        delta_nll_sum_nats=delta,
                        native_to_candidate_kl_sum_nats=kl,
                        top1_agreement_count=agreement,
                        example_execution_receipt_sha256=execution_sha,
                    )
                )
            del native_hidden, candidate_hidden, native_rows, candidate_rows, targets
        if self._fresh_batch_counts[role] != 8 or len(records) != 48:
            raise RuntimeError(f"{role} full-stack ledger did not close")
        self._native_baseline.assert_unchanged()
        self._compiled_evaluator.assert_unchanged()
        return tuple(
            sorted(
                records,
                key=lambda value: value.example_identity_sha256,
            )
        )

    def evaluate_d3_final(
        self,
        *,
        panel,
        candidate,
        c3_evaluation,
        fresh_role_capture_receipt,
    ):
        import torch

        from .gemma3_layer_multimode_evaluation import (
            D3NoNativeMLPTraceReceipt,
            D3ResourceComparisonReceipt,
            evaluate_d3_final,
        )

        role = experiment.D3_ROLE
        records = self._evaluate_full_stack_examples(
            panel=panel,
            candidate=candidate,
            role=role,
            fresh_role_capture_receipt=fresh_role_capture_receipt,
        )
        compilation = candidate.compilation_receipt
        parameter_formula_sha = _digest(
            _D3_PARAMETER_FORMULA_DOMAIN,
            {
                "compilation_receipt_sha256": compilation.receipt_sha256,
                "layer_accounting": [
                    value.state_dict() for value in compilation.layer_accounting
                ],
            },
        )
        serialization_protocol_sha = _digest(
            _D3_SERIALIZATION_PROTOCOL_DOMAIN,
            {
                "container": (
                    "torch.save.new_zipfile_serialization_via_"
                    "exclusive_dirfd_regular_file"
                ),
                "dtype_policy": "unchanged_CPU_contiguous",
                "metadata": "format_version_plus_sorted_state_dict",
                "compression": "torch_save_default_uncompressed_storage_records",
                "sharding": "single_regular_file",
                "native_and_compiled_protocol_identical": True,
                "published_compiler_bundle_is_separate_evidence": True,
                "temporary_artifact_cleanup_is_inode_guarded": True,
            },
        )
        import tempfile

        directory = Path(
            tempfile.mkdtemp(
                prefix="D3-resource-comparison-",
                dir=Path(candidate.bundle_path).parent,
            )
        )
        resource_workspace = _open_D3_resource_directory(directory)
        # Equal-length archive basenames avoid a ZIP metadata length bias in
        # the same-protocol serialized-byte comparison.
        resource_error: BaseException | None = None
        try:
            native_file_sha, native_bytes = _serialize_model_state_same_protocol(
                self._native_baseline.model,
                resource_workspace,
                "native00.pt",
            )
            compiled_file_sha, compiled_bytes = _serialize_model_state_same_protocol(
                self._compiled_evaluator.model,
                resource_workspace,
                "compile0.pt",
            )
            native_boundary = sum(
                value.native_boundary_parameters
                for value in compilation.layer_accounting
            )
            compiled_boundary = sum(
                value.compiled_boundary_parameters
                for value in compilation.layer_accounting
            )
            resource = D3ResourceComparisonReceipt(
                candidate_identity_sha256=(
                    candidate.candidate_identity.artifact_sha256
                ),
                parameter_formula_receipt_sha256=parameter_formula_sha,
                serialization_protocol_sha256=serialization_protocol_sha,
                native_model_file_sha256=native_file_sha,
                compiled_model_file_sha256=compiled_file_sha,
                native_full_model_parameter_count=(
                    compilation.source_parameter_count
                ),
                native_replaced_boundary_parameter_count=native_boundary,
                compiled_replacement_boundary_parameter_count=compiled_boundary,
                expected_compiled_full_model_parameter_count=(
                    compilation.source_parameter_count
                    - native_boundary
                    + compiled_boundary
                ),
                measured_compiled_full_model_parameter_count=(
                    compilation.compiled_parameter_count
                ),
                native_serialized_model_bytes=native_bytes,
                compiled_serialized_model_bytes=compiled_bytes,
                formula_parameter_reduction_passed=(
                    compiled_boundary < native_boundary
                ),
                measured_parameter_reduction_passed=(
                    compilation.compiled_parameter_count
                    == compilation.source_parameter_count
                    - native_boundary
                    + compiled_boundary
                    and compilation.compiled_parameter_count
                    < compilation.source_parameter_count
                ),
                serialized_byte_reduction_passed=compiled_bytes < native_bytes,
                passed=(
                    compiled_boundary < native_boundary
                    and compilation.compiled_parameter_count
                    == compilation.source_parameter_count
                    - native_boundary
                    + compiled_boundary
                    and compilation.compiled_parameter_count
                    < compilation.source_parameter_count
                    and compiled_bytes < native_bytes
                ),
            )
        except BaseException as error:
            resource_error = error
        cleanup_errors = _cleanup_D3_resource_directory(resource_workspace)
        if resource_error is not None:
            for error in cleanup_errors:
                resource_error.add_note(
                    "D3 resource cleanup also failed: "
                    f"{type(error).__name__}: {error}"
                )
            raise resource_error.with_traceback(resource_error.__traceback__)
        if cleanup_errors:
            cleanup_primary = cleanup_errors[0]
            for error in cleanup_errors[1:]:
                cleanup_primary.add_note(
                    "additional D3 resource cleanup failure: "
                    f"{type(error).__name__}: {error}"
                )
            raise cleanup_primary
        compiled_layers = tuple(self._compiled_evaluator.model.model.layers)
        module_inventory = [
            {
                "layer_ordinal": ordinal,
                "mlp_type": (
                    f"{type(layer.mlp).__module__}.{type(layer.mlp).__qualname__}"
                ),
                "coefficient_sha256": layer.mlp.coefficient_sha256,
                "native_mlp_fallback_present": (
                    layer.mlp.native_mlp_fallback_present
                ),
                "ordered_submodules": [
                    [
                        name,
                        f"{type(module).__module__}.{type(module).__qualname__}",
                    ]
                    for name, module in layer.mlp.named_modules()
                ],
            }
            for ordinal, layer in enumerate(compiled_layers)
        ]
        native_mlp_storages = {
            (
                parameter.device.type,
                parameter.device.index,
                int(parameter.untyped_storage().data_ptr()),
                int(parameter.untyped_storage().nbytes()),
            )
            for layer in self._native_baseline.model.model.layers
            for parameter in layer.mlp.parameters()
        }
        candidate_storages = {
            (
                parameter.device.type,
                parameter.device.index,
                int(parameter.untyped_storage().data_ptr()),
                int(parameter.untyped_storage().nbytes()),
            )
            for parameter in self._compiled_evaluator.model.parameters()
        }
        alias_count = len(native_mlp_storages.intersection(candidate_storages))
        trace_protocol_sha = _digest(
            _D3_TRACE_PROTOCOL_DOMAIN,
            {
                "backend": "qualified_exact_module_inventory_plus_"
                "all_replacement_forward_hooks_v1",
                "compiled_layer_count": 18,
                "required_compiled_module_type": (
                    "fisher_graph.gemma3_layer_multimode_compiler."
                    "Float32LayerMultimodeMLP"
                ),
                "native_projection_names_rejected": [
                    "down_proj",
                    "gate_proj",
                    "up_proj",
                ],
                "native_fallback_flag_must_be_false": True,
                "storage_alias_intersection_checked": True,
                "generic_operator_absence_claimed": False,
                "MAC_reduction_claimed": False,
                "latency_reduction_claimed": False,
            },
        )
        native_module_inventory_count = (
            self._compiled_evaluator.native_mlp_module_count
        )
        native_projection_inventory_count = (
            self._compiled_evaluator.native_mlp_projection_inventory_count
        )
        native_fallback_inventory_count = (
            self._compiled_evaluator.native_mlp_fallback_inventory_count
            + int(compilation.native_mlp_fallback_present is not False)
        )
        trace = D3NoNativeMLPTraceReceipt(
            candidate_identity_sha256=candidate.candidate_identity.artifact_sha256,
            role_capture_receipt_sha256=experiment._artifact_sha256(
                self._fresh_role_capture_receipts[role],
                label="D3 role capture",
            ),
            trace_protocol_sha256=trace_protocol_sha,
            module_inventory_sha256=_sha256_catalog(module_inventory),
            trace_event_stream_sha256=_sha256_catalog(self._D3_trace_events),
            proof_backend=(
                "qualified_exact_module_inventory_plus_"
                "all_replacement_forward_hooks_v1"
            ),
            traced_forward_call_count=self._fresh_batch_counts[role],
            traced_example_count=len(records),
            traced_predicted_token_count=sum(
                value.predicted_token_count for value in records
            ),
            compiled_layer_call_counts=tuple(
                self._D3_compiled_layer_call_counts
            ),
            native_mlp_module_inventory_count=(
                native_module_inventory_count
            ),
            native_mlp_projection_inventory_count=(
                native_projection_inventory_count
            ),
            native_mlp_fallback_inventory_count=(
                native_fallback_inventory_count
            ),
            native_mlp_parameter_storage_alias_count=alias_count,
            passed=(
                self._D3_compiled_layer_call_counts == [8] * 18
                and native_module_inventory_count == 0
                and native_projection_inventory_count == 0
                and native_fallback_inventory_count == 0
                and alias_count == 0
            ),
        )
        result = evaluate_d3_final(
            records,
            c3_evaluation=c3_evaluation,
            source_model_sha256=self._native_baseline.model_sha256,
            panel_manifest_sha256=self._fresh_manifests[role]["manifest_sha256"],
            role_capture_receipt_sha256=trace.role_capture_receipt_sha256,
            evaluation_protocol_sha256=self.preflight.evaluation_protocol_sha256,
            candidate_identity=candidate.candidate_identity,
            resource_comparison=resource,
            no_native_mlp_trace=trace,
        )
        self._full_stack_example_aggregates[role] = tuple(records)
        return result

    def build_runtime_execution_receipt(self, *, terminal_status):
        expected_roles_by_status = {
            experiment.TERMINAL_B3_FAILED: (experiment.B3_ROLE,),
            experiment.TERMINAL_C3_FAILED: (
                experiment.B3_ROLE,
                experiment.C3_ROLE,
            ),
            experiment.TERMINAL_D3_COMPLETED: (
                experiment.B3_ROLE,
                experiment.C3_ROLE,
                experiment.D3_ROLE,
            ),
        }
        expected_roles = expected_roles_by_status.get(terminal_status)
        if (
            expected_roles is None
            or self._closed
            or not self._development_teardown_confirmed
            or self._issued_runtime_execution_receipt is not None
            or self._runtime_qualification is None
            or self._A_runtime_receipt is None
            or self._A_comparison_receipt is None
            or self._native_baseline is None
            or self._compiled_evaluator is None
            or self._runtime_session_binding_sha256 is None
            or self._execution_environment_sha256 is None
            or self._compiled_model_fingerprint_before_sha256 is None
            or self._compiled_execution_fingerprint_before_sha256 is None
            or tuple(self._fresh_role_capture_receipts) != expected_roles
            or tuple(self._tokenization_receipts) != expected_roles
        ):
            raise RuntimeError("runtime execution ledger is not terminal")
        self._native_baseline.assert_unchanged()
        self._compiled_evaluator.assert_unchanged()
        if tuple(self._fresh_batch_counts[role] for role in expected_roles) != (
            8,
        ) * len(expected_roles):
            raise RuntimeError("fresh batch ledger did not close")
        qualification = self._runtime_qualification
        A_runtime = self._A_runtime_receipt
        captures = tuple(
            self._fresh_role_capture_receipts[role]
            for role in expected_roles
        )
        native_eval_mode = not any(
            module.training
            for module in self._native_baseline.model.modules()
        )
        receipt = experiment.LayerMultimodeRuntimeExecutionReceipt(
            terminal_status=terminal_status,
            runtime_implementation_sha256=self.runtime_implementation_sha256,
            implementation_qualification_sha256=(
                self.preflight.implementation_qualification_sha256
            ),
            runtime_qualification_receipt=qualification,
            runtime_qualification_receipt_sha256=(
                qualification.receipt_sha256
            ),
            execution_environment_sha256=(
                self._execution_environment_sha256
            ),
            tokenizer_fingerprint_sha256=(
                qualification.tokenizer_asset_catalog_sha256
            ),
            predecessor_A_capture_receipt_sha256=(
                self.preflight.predecessor_A_capture_receipt_sha256
            ),
            predecessor_runtime_execution_receipt_sha256=(
                self.preflight.predecessor_runtime_execution_receipt_sha256
            ),
            predecessor_A_runtime_receipt=A_runtime,
            predecessor_A_runtime_receipt_sha256=A_runtime.receipt_sha256,
            predecessor_A_lineage_comparison_receipt=(
                self._A_comparison_receipt
            ),
            predecessor_A_lineage_comparison_receipt_sha256=(
                self._A_comparison_receipt.artifact_sha256
            ),
            predecessor_A_recaptured_capture_receipt_sha256=(
                A_runtime.source_capture_receipt_sha256
            ),
            predecessor_A_current_runtime_session_binding_sha256=(
                A_runtime.runtime_session_binding_sha256
            ),
            source_model_fingerprint_before_sha256=(
                qualification.source_model_fingerprint_before_sha256
            ),
            source_model_fingerprint_after_sha256=(
                qualification.source_model_fingerprint_after_sha256
            ),
            source_model_mutated=False,
            source_execution_fingerprint_before_sha256=(
                qualification.source_execution_fingerprint_before_sha256
            ),
            source_execution_fingerprint_after_sha256=(
                qualification.source_execution_fingerprint_after_sha256
            ),
            source_execution_mutated=False,
            compiled_model_fingerprint_before_sha256=(
                self._compiled_model_fingerprint_before_sha256
            ),
            compiled_model_fingerprint_after_sha256=(
                self._compiled_evaluator.model_sha256
            ),
            compiled_execution_fingerprint_before_sha256=(
                self._compiled_execution_fingerprint_before_sha256
            ),
            compiled_execution_fingerprint_after_sha256=(
                self._compiled_evaluator.execution_sha256
            ),
            compiled_candidate_mutated=False,
            runtime_session_binding_sha256=(
                self._runtime_session_binding_sha256
            ),
            fresh_role_capture_receipts=captures,
            predecessor_A_batch_count=A_runtime.role_batch_counts[0][1],
            predecessor_A_exact_control_forward_count=(
                A_runtime.exact_control_forward_count
            ),
            predecessor_A_exact_control_vjp_count=(
                A_runtime.exact_control_vjp_count
            ),
            predecessor_A_dense_parity_forward_count=(
                A_runtime.dense_parity_forward_count
            ),
            predecessor_A_dense_parity_vjp_count=(
                A_runtime.dense_parity_vjp_count
            ),
            B3_batch_count=self._fresh_batch_counts[experiment.B3_ROLE],
            B3_native_forward_count=(
                self._native_baseline.forward_counts[experiment.B3_ROLE]
            ),
            B3_candidate_layer_velocity_forward_count=(
                self._B3_candidate_layer_velocity_forward_count
            ),
            C3_batch_count=self._fresh_batch_counts[experiment.C3_ROLE],
            C3_native_forward_count=(
                self._native_baseline.forward_counts[experiment.C3_ROLE]
            ),
            C3_compiled_forward_count=(
                self._compiled_evaluator.forward_counts[experiment.C3_ROLE]
            ),
            D3_batch_count=self._fresh_batch_counts[experiment.D3_ROLE],
            D3_native_forward_count=(
                self._native_baseline.forward_counts[experiment.D3_ROLE]
            ),
            D3_compiled_forward_count=(
                self._compiled_evaluator.forward_counts[experiment.D3_ROLE]
            ),
            fresh_tokenization_authentication_count=len(captures),
            fresh_example_count=sum(value.example_count for value in captures),
            fresh_next_token_position_count=sum(
                value.predicted_next_token_position_count
                for value in captures
            ),
            native_baseline_evaluator_count=1,
            native_baseline_eval_mode=native_eval_mode,
            native_baseline_parameter_mutation_count=(
                self._native_baseline.parameter_mutation_count
            ),
            native_baseline_optimizer_reference_count=(
                self._native_baseline.optimizer_reference_count
            ),
            compiled_candidate_native_mlp_module_count=(
                self._compiled_evaluator.native_mlp_module_count
            ),
            reserve_open_count=0,
        )
        self._issued_runtime_execution_receipt = receipt
        return receipt

    def close(
        self,
        *,
        terminal_status,
        runtime_execution_receipt_sha256,
        runtime_session_binding_sha256,
        fresh_panel_execution_receipt_sha256,
    ):
        if self._closed or not self._development_teardown_confirmed:
            raise RuntimeError("runtime may close exactly once after teardown")
        A_negative = terminal_status == experiment.TERMINAL_A_NO_CANDIDATE
        if A_negative:
            if any(
                value is not None
                for value in (
                    runtime_execution_receipt_sha256,
                    runtime_session_binding_sha256,
                    fresh_panel_execution_receipt_sha256,
                )
            ) or self._issued_runtime_execution_receipt is not None:
                raise RuntimeError("A-negative close claimed fresh execution")
        else:
            issued = self._issued_runtime_execution_receipt
            if (
                issued is None
                or issued.terminal_status != terminal_status
                or issued.receipt_sha256
                != runtime_execution_receipt_sha256
                or issued.runtime_session_binding_sha256
                != runtime_session_binding_sha256
            ):
                raise RuntimeError("terminal close differs from runtime receipt")
            experiment._require_sha256(
                fresh_panel_execution_receipt_sha256,
                label="fresh panel execution receipt",
            )
        if self._capture_runtime is not None:
            raise RuntimeError("predecessor capture runtime survived teardown")
        if self._A_raw_directory is not None:
            if self._A_raw_directory.exists():
                if self._A_raw_directory.is_symlink():
                    raise RuntimeError("temporary A path became a symlink")
                shutil.rmtree(self._A_raw_directory)
            self._A_raw_directory = None
        bundle_retained = False
        if not A_negative:
            if self._frozen_candidate is None:
                raise RuntimeError("compiled candidate disappeared before publication")
            bundle_path = Path(self._frozen_candidate.bundle_path)
            if (
                not bundle_path.is_file()
                or bundle_path.is_symlink()
                or _sha256_file(bundle_path)
                != self._frozen_candidate.bundle_file_sha256
            ):
                raise RuntimeError("frozen bundle changed before publication")
            bundle_retained = True
        if self._native_baseline is not None:
            self._native_baseline.close()
        if self._compiled_evaluator is not None:
            self._compiled_evaluator.close()
        native_model_weakref = self._native_model_weakref
        native_tokenizer_weakref = self._native_tokenizer_weakref
        compiled_model_weakref = self._compiled_model_weakref
        self._native_baseline = None
        self._compiled_evaluator = None
        self._native_model_weakref = None
        self._native_tokenizer_weakref = None
        self._compiled_model_weakref = None
        self._frozen_candidate = None
        self._captured_A = None
        self._bridge = None
        self._partition = None
        self._predecessor_qualification = None
        self._runtime_qualification = None
        self._A_runtime_receipt = None
        self._A_comparison_receipt = None
        self._pending_development_release = None
        self._tokenization_receipts.clear()
        self._authenticated_panel_ids.clear()
        self._fresh_manifests.clear()
        self._fresh_role_capture_receipts.clear()
        self._selected_candidate_sha256s = None
        self._full_stack_example_aggregates.clear()
        self._D3_trace_events.clear()
        self._issued_runtime_execution_receipt = None
        self._execution_environment_sha256 = None
        self._runtime_session_binding_sha256 = None
        self._compiled_model_fingerprint_before_sha256 = None
        self._compiled_execution_fingerprint_before_sha256 = None
        self.predecessor_preflight = None
        self.predecessor_report = {}
        self.implementation_qualification = {}
        gc.collect()
        native_model_count = int(
            native_model_weakref is not None
            and native_model_weakref() is not None
        )
        compiled_model_count = int(
            compiled_model_weakref is not None
            and compiled_model_weakref() is not None
        )
        tokenizer_count = int(
            native_tokenizer_weakref is not None
            and native_tokenizer_weakref() is not None
        )
        self._native_model_weakref = None
        self._native_tokenizer_weakref = None
        self._compiled_model_weakref = None
        self._closed = True
        if native_model_count or compiled_model_count or tokenizer_count:
            raise RuntimeError(
                "terminal cleanup retained an external model/tokenizer alias"
            )
        return experiment.LayerMultimodePostTerminalCleanupReceipt(
            terminal_status=terminal_status,
            runtime_execution_receipt_sha256=(
                runtime_execution_receipt_sha256
            ),
            runtime_session_binding_sha256=runtime_session_binding_sha256,
            fresh_panel_execution_receipt_sha256=(
                fresh_panel_execution_receipt_sha256
            ),
            runtime_closed=True,
            native_baseline_released=True,
            native_baseline_instance_count_after_close=native_model_count,
            compiled_candidate_released=True,
            model_reference_count_after_close=(
                native_model_count + compiled_model_count
            ),
            tokenizer_reference_count_after_close=tokenizer_count,
            temporary_A_spools_removed=True,
            temporary_materializations_removed=True,
            temporary_fit_state_removed=True,
            frozen_bundle_retained_for_publication=bundle_retained,
            reserve_open_count=0,
        )

    def abort(self) -> None:
        if self._closed:
            return
        cleanup_errors: list[BaseException] = []
        if self._capture_runtime is not None:
            try:
                self._capture_runtime.close()
            except BaseException as error:
                cleanup_errors.append(error)
        self._capture_runtime = None
        if self._native_baseline is not None:
            try:
                self._native_baseline.close()
            except BaseException as error:
                cleanup_errors.append(error)
        self._native_baseline = None
        if self._compiled_evaluator is not None:
            try:
                self._compiled_evaluator.close()
            except BaseException as error:
                cleanup_errors.append(error)
        self._compiled_evaluator = None
        self._frozen_candidate = None
        self._captured_A = None
        self._bridge = None
        self._partition = None
        self._predecessor_qualification = None
        self._runtime_qualification = None
        self._A_runtime_receipt = None
        self._A_comparison_receipt = None
        self._pending_development_release = None
        self._issued_runtime_execution_receipt = None
        self._execution_environment_sha256 = None
        self._runtime_session_binding_sha256 = None
        self._compiled_model_fingerprint_before_sha256 = None
        self._compiled_execution_fingerprint_before_sha256 = None
        self._tokenization_receipts.clear()
        self._authenticated_panel_ids.clear()
        self._fresh_manifests.clear()
        self._fresh_role_capture_receipts.clear()
        self._selected_candidate_sha256s = None
        self._full_stack_example_aggregates.clear()
        self._D3_trace_events.clear()
        self.predecessor_preflight = None
        self.predecessor_report = {}
        self.implementation_qualification = {}
        if self._A_raw_directory is not None:
            try:
                if self._A_raw_directory.exists():
                    shutil.rmtree(self._A_raw_directory)
            except BaseException as error:
                cleanup_errors.append(error)
        self._A_raw_directory = None
        self._closed = True
        if cleanup_errors:
            primary = cleanup_errors[0]
            for error in cleanup_errors[1:]:
                primary.add_note(
                    "additional abort cleanup failure: "
                    f"{type(error).__name__}: {error}"
                )
            raise primary


def build_production_layer_multimode_runtime(
    preflight: experiment.LayerMultimodeExecutionPreflightReceipt,
    *,
    repository_root: Path | str,
    cache_dir: Path | str | None = None,
) -> Gemma3LayerMultimodeRuntime:
    """Build from a live v3 authority; no predecessor evidence is injectable."""

    return Gemma3LayerMultimodeRuntime(
        preflight,
        repository_root=repository_root,
        cache_dir=cache_dir,
    )


__all__ = [
    "Gemma3LayerMultimodeRuntime",
    "LayerMultimodeRuntimeQualificationReceipt",
    "PredecessorARecaptureComparisonReceipt",
    "build_production_layer_multimode_runtime",
]
