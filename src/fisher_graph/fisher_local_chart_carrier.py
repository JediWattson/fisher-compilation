"""Authoritative rank-64 Fisher carrier for the layer-4 local-chart probe.

The source experiment contains several unrelated calibration products.  This
module intentionally follows one, and only one, data path through that file::

    calibration_a -> fisher -> bases -> layer.4.output -> fisher

It republishes the leading Fisher eigensystem as a small, prompt-free,
weights-only sidecar.  Calibration-B selections, codecs, and results are not
read and cannot enter the sidecar receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
from torch import Tensor

from .fisher_local_chart import Rank64FisherCarrier


__all__ = [
    "DEFAULT_CARRIER_SIDECAR",
    "DEFAULT_FISHER_SOURCE",
    "DEFAULT_FISHER_CARRIER_PATH",
    "DEFAULT_WEIGHTED_JACOBIAN_PATH",
    "FisherLocalChartCarrierArtifact",
    "extract_fisher_local_chart_carrier",
    "load_fisher_local_chart_carrier",
]


DEFAULT_WEIGHTED_JACOBIAN_PATH = Path(
    ".local-runs/google--gemma-3-270m/layers-4-6-weighted-jacobian.pt"
)
DEFAULT_FISHER_CARRIER_PATH = Path(
    ".local-runs/google--gemma-3-270m/"
    "fisher-local-chart-layer4-output-rank64-carrier-v1.pt"
)
# Runner-facing names.  Keep the longer aliases for callers that benefit from
# seeing what kind of artifact each path denotes.
DEFAULT_FISHER_SOURCE = DEFAULT_WEIGHTED_JACOBIAN_PATH
DEFAULT_CARRIER_SIDECAR = DEFAULT_FISHER_CARRIER_PATH

_SOURCE_SCHEMA = "fisher_graph.gemma3_weighted_jacobian"
_SOURCE_FORMAT_VERSION = 2
_ARTIFACT_SCHEMA = "fisher_graph.fisher_local_chart.output_fisher_carrier"
_ARTIFACT_FORMAT_VERSION = 1
_SITE = "layer.4.output"
_DEFAULT_MODEL_ID = "google/gemma-3-270m"
_DEFAULT_RESOLVED_COMMIT = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_DEFAULT_CONFIG_SHA256 = (
    "588ffd9caad89c7bcb29c981e6836a16812c7b5533ffc4d8e0448e1fab3e1924"
)
_CARRIER_DOMAIN = b"fisher-graph:fisher-local-chart:output-carrier:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:fisher-local-chart:carrier-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFETY = {
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_tokenizer_state": False,
    "contains_model_weights": False,
    "contains_calibration_b": False,
}
_QUALIFICATION = {
    "metric": "shared_output_empirical_fisher_vjp",
    "metric_site": _SITE,
    "shared_post_branch_metric": True,
    "exact_local_fisher_at_attention_site": False,
    "exact_local_fisher_at_mlp_site": False,
    "claim": (
        "shared output-Fisher sensitivity metric only; it is not an exact "
        "local Fisher at either earlier branch site"
    ),
}


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} keys do not match the authenticated schema "
            f"(missing={missing}, extra={extra})"
        )


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float(value: object, *, label: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{label} must be a finite {qualifier}number")
    return float(value)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("carrier tensors must be finite detached CPU tensors")
    tensor = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _carrier_digest(
    *,
    source: Mapping[str, object],
    estimator: Mapping[str, object],
    tensor_sha256s: Mapping[str, str],
) -> str:
    receipt = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "site": _SITE,
        "rank": estimator["retained_rank"],
        "source": dict(source),
        "estimator": dict(estimator),
        "semantic_qualification": dict(_QUALIFICATION),
        "safety": dict(_SAFETY),
        "tensor_sha256s": dict(tensor_sha256s),
    }
    return hashlib.sha256(
        _CARRIER_DOMAIN + _canonical_json_bytes(receipt)
    ).hexdigest()


def _load_weights_only(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    before = path.stat()
    with path.open("rb") as handle:
        raw = torch.load(handle, map_location="cpu", weights_only=True)
        after_fd = os.fstat(handle.fileno())
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    identity_fd = (
        after_fd.st_dev,
        after_fd.st_ino,
        after_fd.st_size,
        after_fd.st_mtime_ns,
    )
    if identity_before != identity_after or identity_before != identity_fd:
        raise RuntimeError(f"{label} changed while it was being loaded")
    return _mapping(raw, label=label)


def _exclusive_atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link gives us atomic O_EXCL semantics without replacing an
        # independently produced artifact at the requested path.
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing carrier sidecar: {path}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class FisherLocalChartCarrierArtifact:
    """Authenticated sidecar plus its executable low-rank carrier."""

    carrier: Rank64FisherCarrier
    source_file_sha256: str
    source_scientific_payload_sha256: str
    model_id: str
    resolved_commit: str
    config_sha256: str
    observations: int
    nonzero_observations: int
    sequences: int
    sketch_rows: int
    source_fisher_trace: float
    retained_rank_trace: float
    retained_trace_fraction: float
    q_sha256: str
    eigenvalues_sha256: str
    carrier_sha256: str

    @property
    def q(self) -> Tensor:
        return self.carrier.q

    @property
    def eigenvalues(self) -> Tensor:
        return self.carrier.eigenvalues

    @property
    def rank(self) -> int:
        return self.carrier.rank

    @property
    def hidden_width(self) -> int:
        return self.carrier.hidden_width

    def metadata(self) -> dict[str, object]:
        return {
            "schema": _ARTIFACT_SCHEMA,
            "format_version": _ARTIFACT_FORMAT_VERSION,
            "site": _SITE,
            "rank": self.rank,
            "hidden_width": self.hidden_width,
            "source_file_sha256": self.source_file_sha256,
            "source_scientific_payload_sha256": (
                self.source_scientific_payload_sha256
            ),
            "model_id": self.model_id,
            "resolved_commit": self.resolved_commit,
            "config_sha256": self.config_sha256,
            "observations": self.observations,
            "nonzero_observations": self.nonzero_observations,
            "sequences": self.sequences,
            "sketch_rows": self.sketch_rows,
            "source_fisher_trace": self.source_fisher_trace,
            "retained_rank_trace": self.retained_rank_trace,
            "retained_trace_fraction": self.retained_trace_fraction,
            "semantic_qualification": dict(_QUALIFICATION),
            "safety": dict(_SAFETY),
            "tensor_sha256s": {
                "q": self.q_sha256,
                "eigenvalues": self.eigenvalues_sha256,
            },
            "carrier_sha256": self.carrier_sha256,
        }


def _source_factors(
    raw: Mapping[str, Any],
    *,
    rank: int,
    expected_hidden_width: int,
    expected_model_id: str,
    expected_resolved_commit: str,
    expected_config_sha256: str,
) -> tuple[Rank64FisherCarrier, dict[str, object], dict[str, object]]:
    """Read the sole authorized source path and validate its provenance."""

    if raw.get("schema") != _SOURCE_SCHEMA:
        raise ValueError("weighted-Jacobian source schema mismatch")
    if raw.get("format_version") != _SOURCE_FORMAT_VERSION:
        raise ValueError("weighted-Jacobian source format version mismatch")
    for key in (
        "contains_model_weights",
        "contains_prompt_text",
        "contains_tokenizer_state",
    ):
        if raw.get(key) is not False:
            raise ValueError(f"unsafe weighted-Jacobian source flag: {key}")

    model = _mapping(raw.get("model"), label="source model metadata")
    if model.get("model_id") != expected_model_id:
        raise ValueError("source model ID mismatch")
    if model.get("resolved_commit") != expected_resolved_commit:
        raise ValueError("source resolved model commit mismatch")
    if model.get("config_sha256") != expected_config_sha256:
        raise ValueError("source model config SHA-256 mismatch")
    if model.get("hidden_size") != expected_hidden_width:
        raise ValueError("source hidden width mismatch")
    if model.get("weights_in_artifact") is not False:
        raise ValueError("source unexpectedly declares embedded model weights")

    protocol = _mapping(raw.get("protocol"), label="source protocol")
    if protocol.get("residual_width") != expected_hidden_width:
        raise ValueError("source residual-width protocol mismatch")
    if protocol.get("fisher_scope") != "width_pooled":
        raise ValueError("source Fisher scope is not width_pooled")
    if protocol.get("normalizer") != "valid_activation_positions":
        raise ValueError("source Fisher normalizer mismatch")
    if protocol.get("score") != "summed_hard_target_next_token_nll":
        raise ValueError("source Fisher score mismatch")
    if protocol.get("fit_policy") != "calibration_a_only":
        raise ValueError("source Fisher was not fit on Calibration A only")
    canonical = protocol.get("canonical_boundaries")
    gated = protocol.get("gated_output_sites")
    if (
        not isinstance(canonical, (tuple, list))
        or _SITE not in canonical
        or not isinstance(gated, (tuple, list))
        or _SITE not in gated
    ):
        raise ValueError("source protocol does not bind layer.4.output")
    if _positive_int(protocol.get("extraction_rank"), label="extraction rank") < rank:
        raise ValueError("source extraction rank is below requested carrier rank")

    # Do not inspect ``calibration_b`` (or any codec/selection field).  The
    # exact path below is the experiment's only authorized numerical source.
    calibration_a = _mapping(raw.get("calibration_a"), label="Calibration A")
    fisher_root = _mapping(calibration_a.get("fisher"), label="Calibration-A Fisher")
    if fisher_root.get("format_version") != 1:
        raise ValueError("Calibration-A Fisher format version mismatch")
    bases = _mapping(fisher_root.get("bases"), label="Calibration-A Fisher bases")
    basis = _mapping(bases.get(_SITE), label=f"Calibration-A {_SITE} basis")
    if basis.get("activation_name") != _SITE:
        raise ValueError("source basis activation mismatch")
    if basis.get("sequences") != fisher_root.get("sequences"):
        raise ValueError("source basis sequence count mismatch")
    fisher = _mapping(basis.get("fisher"), label=f"Calibration-A {_SITE} Fisher")

    if fisher.get("activation_name") != _SITE:
        raise ValueError("source Fisher activation mismatch")
    if fisher.get("width") != expected_hidden_width:
        raise ValueError("source Fisher width mismatch")
    if fisher.get("scope") != "width_pooled":
        raise ValueError("source Fisher has unexpected scope")
    if fisher.get("score_reduction") != "sum":
        raise ValueError("source Fisher has unexpected score reduction")
    if fisher.get("normalizer") != "valid_activation_positions":
        raise ValueError("source Fisher has unexpected normalizer")
    if fisher.get("accumulation_dtype") != "float64":
        raise ValueError("source Fisher was not accumulated in float64")
    if fisher.get("algorithm") != "frequent_directions":
        raise ValueError("source Fisher algorithm mismatch")
    if fisher.get("algorithm_version") != 1:
        raise ValueError("source Fisher algorithm version mismatch")

    observations = _positive_int(fisher.get("observations"), label="observations")
    nonzero = _positive_int(
        fisher.get("nonzero_observations"), label="nonzero observations"
    )
    rows_seen = _positive_int(fisher.get("rows_seen"), label="rows seen")
    requested_rank = _positive_int(
        fisher.get("requested_rank"), label="requested rank"
    )
    sketch_rows = _positive_int(fisher.get("sketch_rows"), label="sketch rows")
    sequences = _positive_int(basis.get("sequences"), label="sequence count")
    if rows_seen != observations or nonzero > observations or nonzero < rank:
        raise ValueError("source Fisher observation metadata is inconsistent")
    if requested_rank < rank or sketch_rows <= requested_rank:
        raise ValueError("source Fisher sketch cannot support the requested rank")
    source_trace = _finite_float(
        fisher.get("fisher_trace"), label="source Fisher trace", positive=True
    )

    vectors = fisher.get("vectors")
    eigenvalues = fisher.get("eigenvalues")
    if (
        not isinstance(vectors, Tensor)
        or vectors.device.type != "cpu"
        or vectors.ndim != 2
        or vectors.shape[0] != expected_hidden_width
        or vectors.shape[1] < rank
        or not vectors.is_floating_point()
        or not bool(torch.isfinite(vectors).all())
    ):
        raise ValueError("source Fisher vectors are invalid")
    if (
        not isinstance(eigenvalues, Tensor)
        or eigenvalues.device.type != "cpu"
        or eigenvalues.ndim != 1
        or eigenvalues.shape[0] < rank
        or not eigenvalues.is_floating_point()
        or not bool(torch.isfinite(eigenvalues).all())
    ):
        raise ValueError("source Fisher eigenvalues are invalid")
    q = vectors[:, :rank].detach().to(dtype=torch.float64).clone().contiguous()
    retained_eigenvalues = (
        eigenvalues[:rank].detach().to(dtype=torch.float64).clone().contiguous()
    )
    if bool((retained_eigenvalues <= 0.0).any()):
        raise ValueError("requested source Fisher modes include a zero mode")
    if bool((retained_eigenvalues[1:] > retained_eigenvalues[:-1]).any()):
        raise ValueError("source Fisher eigenvalues are not descending")
    carrier = Rank64FisherCarrier(q=q, eigenvalues=retained_eigenvalues)
    retained_trace = float(retained_eigenvalues.sum().item())
    retained_fraction = retained_trace / source_trace
    if not 0.0 < retained_fraction <= 1.0 + 1.0e-12:
        raise ValueError("retained Fisher trace fraction is invalid")

    source = {
        "file_sha256": "",  # filled only after immutable file authentication
        "schema": _SOURCE_SCHEMA,
        "format_version": _SOURCE_FORMAT_VERSION,
        "scientific_payload_sha256": _sha256(
            raw.get("scientific_payload_sha256"),
            label="source scientific payload",
        ),
        "source_role": "calibration_a",
        "source_path": (
            "calibration_a.fisher.bases.layer.4.output.fisher"
        ),
        "model_id": expected_model_id,
        "resolved_commit": expected_resolved_commit,
        "config_sha256": expected_config_sha256,
    }
    estimator = {
        "activation_name": _SITE,
        "hidden_width": expected_hidden_width,
        "retained_rank": rank,
        "observations": observations,
        "nonzero_observations": nonzero,
        "sequences": sequences,
        "requested_rank": requested_rank,
        "sketch_rows": sketch_rows,
        "scope": "width_pooled",
        "score_reduction": "sum",
        "normalizer": "valid_activation_positions",
        "accumulation_dtype": "float64",
        "algorithm": "frequent_directions",
        "algorithm_version": 1,
        "source_fisher_trace": source_trace,
        "retained_rank_trace": retained_trace,
        "retained_trace_fraction": min(retained_fraction, 1.0),
    }
    return carrier, source, estimator


def _payload_from_parts(
    *,
    carrier: Rank64FisherCarrier,
    source: Mapping[str, object],
    estimator: Mapping[str, object],
) -> dict[str, object]:
    tensor_sha256s = {
        "q": _tensor_sha256(carrier.q),
        "eigenvalues": _tensor_sha256(carrier.eigenvalues),
    }
    digest = _carrier_digest(
        source=source,
        estimator=estimator,
        tensor_sha256s=tensor_sha256s,
    )
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "site": _SITE,
        "rank": carrier.rank,
        "source": dict(source),
        "estimator": dict(estimator),
        "semantic_qualification": dict(_QUALIFICATION),
        "safety": dict(_SAFETY),
        "tensors": {
            "q": carrier.q.detach().clone(),
            "eigenvalues": carrier.eigenvalues.detach().clone(),
        },
        "tensor_sha256s": tensor_sha256s,
        "carrier_sha256": digest,
    }


def extract_fisher_local_chart_carrier(
    source_path: str | Path = DEFAULT_WEIGHTED_JACOBIAN_PATH,
    output_path: str | Path = DEFAULT_FISHER_CARRIER_PATH,
    *,
    rank: int = 64,
    expected_hidden_width: int = 640,
    expected_model_id: str = _DEFAULT_MODEL_ID,
    expected_resolved_commit: str = _DEFAULT_RESOLVED_COMMIT,
    expected_config_sha256: str = _DEFAULT_CONFIG_SHA256,
) -> FisherLocalChartCarrierArtifact:
    """Extract and exclusively publish the authorized output-Fisher carrier."""

    rank = _positive_int(rank, label="rank")
    expected_hidden_width = _positive_int(
        expected_hidden_width, label="expected hidden width"
    )
    _sha256(expected_config_sha256, label="expected config")
    source_file = Path(source_path)
    output_file = Path(output_path)
    if source_file.resolve() == output_file.resolve():
        raise ValueError("carrier output may not overwrite its source")

    before_sha256 = _file_sha256(source_file)
    raw = _load_weights_only(source_file, label="weighted-Jacobian source")
    carrier, source, estimator = _source_factors(
        raw,
        rank=rank,
        expected_hidden_width=expected_hidden_width,
        expected_model_id=expected_model_id,
        expected_resolved_commit=expected_resolved_commit,
        expected_config_sha256=expected_config_sha256,
    )
    after_sha256 = _file_sha256(source_file)
    if before_sha256 != after_sha256:
        raise RuntimeError("weighted-Jacobian source changed during extraction")
    source["file_sha256"] = before_sha256
    payload = _payload_from_parts(
        carrier=carrier,
        source=source,
        estimator=estimator,
    )
    _exclusive_atomic_torch_save(payload, output_file)
    return load_fisher_local_chart_carrier(
        output_file,
        expected_rank=rank,
        expected_hidden_width=expected_hidden_width,
        expected_model_id=expected_model_id,
        expected_resolved_commit=expected_resolved_commit,
        expected_config_sha256=expected_config_sha256,
        expected_source_file_sha256=before_sha256,
    )


def load_fisher_local_chart_carrier(
    path: str | Path = DEFAULT_FISHER_CARRIER_PATH,
    *,
    expected_rank: int = 64,
    expected_hidden_width: int = 640,
    expected_model_id: str = _DEFAULT_MODEL_ID,
    expected_resolved_commit: str = _DEFAULT_RESOLVED_COMMIT,
    expected_config_sha256: str = _DEFAULT_CONFIG_SHA256,
    expected_source_file_sha256: str | None = None,
) -> FisherLocalChartCarrierArtifact:
    """Strictly authenticate a prompt-free weights-only carrier sidecar."""

    expected_rank = _positive_int(expected_rank, label="expected rank")
    expected_hidden_width = _positive_int(
        expected_hidden_width, label="expected hidden width"
    )
    _sha256(expected_config_sha256, label="expected config")
    if expected_source_file_sha256 is not None:
        _sha256(expected_source_file_sha256, label="expected source file")
    raw = _load_weights_only(Path(path), label="Fisher carrier sidecar")
    _exact_keys(
        raw,
        {
            "schema",
            "format_version",
            "site",
            "rank",
            "source",
            "estimator",
            "semantic_qualification",
            "safety",
            "tensors",
            "tensor_sha256s",
            "carrier_sha256",
        },
        label="carrier sidecar",
    )
    if raw["schema"] != _ARTIFACT_SCHEMA or raw["format_version"] != 1:
        raise ValueError("carrier sidecar schema/version mismatch")
    if raw["site"] != _SITE or raw["rank"] != expected_rank:
        raise ValueError("carrier sidecar site/rank mismatch")

    safety = _mapping(raw["safety"], label="carrier safety flags")
    qualification = _mapping(
        raw["semantic_qualification"], label="semantic qualification"
    )
    if dict(safety) != _SAFETY:
        raise ValueError("carrier safety flags mismatch")
    if dict(qualification) != _QUALIFICATION:
        raise ValueError("carrier semantic qualification mismatch")

    source = _mapping(raw["source"], label="carrier source receipt")
    _exact_keys(
        source,
        {
            "file_sha256",
            "schema",
            "format_version",
            "scientific_payload_sha256",
            "source_role",
            "source_path",
            "model_id",
            "resolved_commit",
            "config_sha256",
        },
        label="carrier source receipt",
    )
    source_sha256 = _sha256(source["file_sha256"], label="source file")
    scientific_sha256 = _sha256(
        source["scientific_payload_sha256"], label="source scientific payload"
    )
    if expected_source_file_sha256 is not None and (
        source_sha256 != expected_source_file_sha256
    ):
        raise ValueError("carrier source file SHA-256 mismatch")
    if (
        source["schema"] != _SOURCE_SCHEMA
        or source["format_version"] != _SOURCE_FORMAT_VERSION
        or source["source_role"] != "calibration_a"
        or source["source_path"]
        != "calibration_a.fisher.bases.layer.4.output.fisher"
        or source["model_id"] != expected_model_id
        or source["resolved_commit"] != expected_resolved_commit
        or source["config_sha256"] != expected_config_sha256
    ):
        raise ValueError("carrier source receipt mismatch")

    estimator = _mapping(raw["estimator"], label="carrier estimator receipt")
    _exact_keys(
        estimator,
        {
            "activation_name",
            "hidden_width",
            "retained_rank",
            "observations",
            "nonzero_observations",
            "sequences",
            "requested_rank",
            "sketch_rows",
            "scope",
            "score_reduction",
            "normalizer",
            "accumulation_dtype",
            "algorithm",
            "algorithm_version",
            "source_fisher_trace",
            "retained_rank_trace",
            "retained_trace_fraction",
        },
        label="carrier estimator receipt",
    )
    if (
        estimator["activation_name"] != _SITE
        or estimator["hidden_width"] != expected_hidden_width
        or estimator["retained_rank"] != expected_rank
        or estimator["scope"] != "width_pooled"
        or estimator["score_reduction"] != "sum"
        or estimator["normalizer"] != "valid_activation_positions"
        or estimator["accumulation_dtype"] != "float64"
        or estimator["algorithm"] != "frequent_directions"
        or estimator["algorithm_version"] != 1
    ):
        raise ValueError("carrier estimator metadata mismatch")
    observations = _positive_int(estimator["observations"], label="observations")
    nonzero = _positive_int(
        estimator["nonzero_observations"], label="nonzero observations"
    )
    sequences = _positive_int(estimator["sequences"], label="sequences")
    requested_rank = _positive_int(
        estimator["requested_rank"], label="requested rank"
    )
    sketch_rows = _positive_int(estimator["sketch_rows"], label="sketch rows")
    if (
        nonzero > observations
        or nonzero < expected_rank
        or requested_rank < expected_rank
        or sketch_rows <= requested_rank
    ):
        raise ValueError("carrier estimator counts are inconsistent")
    source_trace = _finite_float(
        estimator["source_fisher_trace"], label="source trace", positive=True
    )
    retained_trace = _finite_float(
        estimator["retained_rank_trace"], label="retained trace", positive=True
    )
    retained_fraction = _finite_float(
        estimator["retained_trace_fraction"],
        label="retained trace fraction",
        positive=True,
    )

    tensors = _mapping(raw["tensors"], label="carrier tensors")
    hashes = _mapping(raw["tensor_sha256s"], label="carrier tensor hashes")
    _exact_keys(tensors, {"q", "eigenvalues"}, label="carrier tensors")
    _exact_keys(hashes, {"q", "eigenvalues"}, label="carrier tensor hashes")
    q = tensors["q"]
    eigenvalues = tensors["eigenvalues"]
    if not isinstance(q, Tensor) or not isinstance(eigenvalues, Tensor):
        raise ValueError("carrier factors must be tensors")
    carrier = Rank64FisherCarrier(q=q, eigenvalues=eigenvalues)
    if carrier.hidden_width != expected_hidden_width or carrier.rank != expected_rank:
        raise ValueError("carrier tensor dimensions mismatch")
    if bool((carrier.eigenvalues[1:] > carrier.eigenvalues[:-1]).any()):
        raise ValueError("carrier eigenvalues are not descending")
    q_sha256 = _sha256(hashes["q"], label="q tensor")
    eigenvalues_sha256 = _sha256(
        hashes["eigenvalues"], label="eigenvalues tensor"
    )
    if q_sha256 != _tensor_sha256(carrier.q):
        raise ValueError("carrier q tensor SHA-256 mismatch")
    if eigenvalues_sha256 != _tensor_sha256(carrier.eigenvalues):
        raise ValueError("carrier eigenvalue tensor SHA-256 mismatch")
    measured_trace = float(carrier.eigenvalues.sum().item())
    measured_fraction = measured_trace / source_trace
    if not math.isclose(retained_trace, measured_trace, rel_tol=1e-14, abs_tol=1e-15):
        raise ValueError("carrier retained trace does not match eigenvalues")
    if not math.isclose(
        retained_fraction,
        measured_fraction,
        rel_tol=1e-14,
        abs_tol=1e-15,
    ):
        raise ValueError("carrier retained trace fraction mismatch")
    if not 0.0 < retained_fraction <= 1.0 + 1.0e-12:
        raise ValueError("carrier retained trace fraction is invalid")

    carrier_sha256 = _sha256(raw["carrier_sha256"], label="carrier")
    expected_digest = _carrier_digest(
        source=source,
        estimator=estimator,
        tensor_sha256s={
            "q": q_sha256,
            "eigenvalues": eigenvalues_sha256,
        },
    )
    if carrier_sha256 != expected_digest:
        raise ValueError("carrier content SHA-256 mismatch")

    return FisherLocalChartCarrierArtifact(
        carrier=carrier,
        source_file_sha256=source_sha256,
        source_scientific_payload_sha256=scientific_sha256,
        model_id=expected_model_id,
        resolved_commit=expected_resolved_commit,
        config_sha256=expected_config_sha256,
        observations=observations,
        nonzero_observations=nonzero,
        sequences=sequences,
        sketch_rows=sketch_rows,
        source_fisher_trace=source_trace,
        retained_rank_trace=retained_trace,
        retained_trace_fraction=min(retained_fraction, 1.0),
        q_sha256=q_sha256,
        eigenvalues_sha256=eigenvalues_sha256,
        carrier_sha256=carrier_sha256,
    )
