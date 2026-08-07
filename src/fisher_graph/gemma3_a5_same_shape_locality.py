"""Strict same-shape off-row locality audit for Gemma final heads.

For every source row this audit deterministically changes every coordinate of
only that row, replays the final head at the identical
``[1, rows, hidden]`` shape, and compares every preserved target row.  The
resulting ``source -> target`` catalog covers all directed off-diagonal pairs
without allowing simultaneous peer changes to cancel.  Nonlinear token-local
heads pass while genuine off-row dependence is exposed without changing
GEMM/GEMV shape.  Callers must explicitly say when the receipt authorizes a
solver; diagnostic-only remains the default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Protocol

import torch
from torch import Tensor

from .gemma3_l10_l17_a5_frozen_affine_capacity_oracle import (
    _contains_tensor,
    _head_only_context,
    _tensor_sha256,
)


_SCHEMA = "fisher_graph.a5_same_shape_off_row_locality.v2"
_RECEIPT_DOMAIN = b"fisher-graph:a5-same-shape-off-row-locality:v2\0"
_METHOD = (
    "mutate_each_source_row_alone_and_compare_every_preserved_target_at_"
    "identical_1_by_rows_by_hidden_shape"
)
_PAIR_ORDER = (
    "source_row_ascending_then_target_row_ascending_excluding_diagonal"
)
_COUNTERFACTUAL_POLICY = {
    "mutated_rows_per_call": 1,
    "source_transform": (
        "opposite_sign_half_chunk_max_abs_clamped_to_at_least_one"
    ),
    "minimum_coordinate_change_in_chunk_scale_units": 0.5,
    "every_source_coordinate_verified_changed": True,
    "every_nonsource_target_row_verified_exact": True,
}
_RECEIPT_BODY_FIELDS = {
    "schema",
    "scientific_role",
    "changes_solver_authorization",
    "probe_name",
    "method",
    "counterfactual_policy",
    "row_count",
    "hidden_width",
    "vocabulary_width",
    "projection_input_shape",
    "projection_output_shape",
    "projection_call_count",
    "baseline_call_count",
    "counterfactual_call_count",
    "directed_pair_order",
    "directed_pair_count",
    "mutation_scale",
    "absolute_tolerance",
    "relative_tolerance",
    "input_rows_sha256",
    "baseline_logits_sha256",
    "source_counterfactuals",
    "directed_pair_checks",
    "failing_directed_pair_count",
    "failing_directed_pairs",
    "worst_source_row_index",
    "worst_target_row_index",
    "worst_max_abs",
    "worst_rms",
    "worst_max_abs_over_allowed",
    "passed",
    "contains_tensor_payloads",
}
_SOURCE_COUNTERFACTUAL_FIELDS = {
    "source_row_index",
    "source_row_sha256",
    "mutated_source_row_sha256",
    "counterfactual_input_sha256",
    "counterfactual_logits_sha256",
    "mutated_source_element_count",
    "preserved_target_row_count",
    "minimum_absolute_coordinate_change",
    "maximum_absolute_coordinate_change",
    "source_mutation_l2",
}
_DIRECTED_PAIR_FIELDS = {
    "source_row_index",
    "target_row_index",
    "max_abs",
    "rms",
    "target_reference_max_abs",
    "allowed",
    "max_abs_over_allowed",
    "passed",
}


class _FinalHead(Protocol):
    def project_logits(
        self, hidden_states: Tensor, sequence: object, *, trace: object = None
    ) -> Tensor: ...


@dataclass(frozen=True)
class SameShapeOffRowLocalityAudit:
    """Ephemeral baseline logits plus a tensor-free authenticated receipt."""

    baseline_logits: Tensor
    receipt: Mapping[str, object]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _receipt_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_RECEIPT_DOMAIN + _canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_counterfactual_policy(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(
        _COUNTERFACTUAL_POLICY
    ):
        return False
    return bool(
        type(value.get("mutated_rows_per_call")) is int
        and value["mutated_rows_per_call"] == 1
        and type(value.get("source_transform")) is str
        and value.get("source_transform")
        == _COUNTERFACTUAL_POLICY["source_transform"]
        and type(
            value.get("minimum_coordinate_change_in_chunk_scale_units")
        )
        is float
        and value["minimum_coordinate_change_in_chunk_scale_units"] == 0.5
        and value.get("every_source_coordinate_verified_changed") is True
        and value.get("every_nonsource_target_row_verified_exact") is True
    )


def _exact_int_shape(value: object, expected: list[int]) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(expected)
        and all(
            type(observed) is int and observed == wanted
            for observed, wanted in zip(value, expected, strict=True)
        )
    )


def _finite_state_rows(rows: object) -> bool:
    return bool(
        isinstance(rows, Tensor)
        and rows.layout is torch.strided
        and rows.ndim == 2
        and rows.is_floating_point()
        and rows.shape[0] >= 2
        and rows.shape[1] >= 1
        and torch.isfinite(rows.detach()).all().item()
    )


def _project_same_shape(adapter: _FinalHead, rows: Tensor) -> Tensor:
    context = _head_only_context(int(rows.shape[0]), rows.device)
    logits = adapter.project_logits(rows[None, :, :], context)
    if (
        not isinstance(logits, Tensor)
        or logits.layout is not torch.strided
        or logits.ndim != 3
        or logits.shape[:2] != (1, rows.shape[0])
        or logits.shape[2] < 2
        or not logits.is_floating_point()
        or logits.device != rows.device
        or not bool(torch.isfinite(logits.detach()).all().item())
    ):
        raise RuntimeError(
            "adapter.project_logits must return finite [1, rows, vocab] logits"
        )
    return logits[0].detach()


def _counterfactual_rows(rows: Tensor, *, source_row: int) -> tuple[Tensor, Tensor]:
    """Materially replace one source row while keeping every target exact."""

    row_count = int(rows.shape[0])
    if source_row < 0 or source_row >= row_count:
        raise ValueError("source_row is outside the counterfactual chunk")
    source = rows[source_row]
    scale = rows.detach().abs().amax().clamp_min(1.0)
    magnitude = 0.5 * scale
    # Every nonnegative source coordinate becomes materially negative; every
    # negative coordinate becomes materially positive.  Thus each coordinate
    # changes by at least half the chunk scale, without overflow at finite
    # floating-point maxima.
    replacement = torch.where(
        source >= 0,
        -torch.ones_like(source) * magnitude,
        torch.ones_like(source) * magnitude,
    ).contiguous()
    counterfactual = rows.clone()
    counterfactual[source_row].copy_(replacement)
    preserved_mask = torch.ones(
        row_count, dtype=torch.bool, device=rows.device
    )
    preserved_mask[source_row] = False
    if not torch.equal(counterfactual[preserved_mask], rows[preserved_mask]):
        raise RuntimeError("same-shape counterfactual changed a target row")
    if not bool((replacement != source).all().item()):
        raise RuntimeError("same-shape counterfactual left a source coordinate")
    if not bool(torch.isfinite(counterfactual).all().item()):
        raise RuntimeError("same-shape counterfactual became non-finite")
    delta = (replacement.detach().double() - source.detach().double()).abs()
    if float(delta.min().item()) < 0.5 * float(scale.detach().double().item()):
        raise RuntimeError("same-shape source mutation was not material")
    return counterfactual, delta


def audit_same_shape_off_row_locality(
    *,
    adapter: _FinalHead,
    rows: Tensor,
    probe_name: str,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 2.0e-6,
    solver_authorization: bool = False,
) -> SameShapeOffRowLocalityAudit:
    """Audit off-row dependence without changing projection shape.

    Projection cost is exactly ``1 + rows`` calls: one baseline call and one
    same-shape counterfactual call for each mutated source row.  Each call
    compares all ``rows - 1`` preserved targets, covering ``rows * (rows - 1)``
    directed pairs.
    """

    if not _finite_state_rows(rows):
        raise ValueError(
            "rows must be a finite floating [rows>=2, hidden>=1] matrix"
        )
    if not isinstance(probe_name, str) or not probe_name:
        raise ValueError("probe_name must be a non-empty string")
    if not isinstance(solver_authorization, bool):
        raise TypeError("solver_authorization must be a bool")
    for label, value in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be finite and nonnegative")
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    row_count, hidden_width = (int(value) for value in rows.shape)
    mutation_scale = float(
        rows.detach().double().abs().amax().clamp_min(1.0).item()
    )

    with torch.no_grad():
        baseline_logits = _project_same_shape(adapter, rows)
        source_counterfactuals: list[dict[str, object]] = []
        directed_pair_checks: list[dict[str, object]] = []
        for source_row in range(row_count):
            counterfactual, source_delta = _counterfactual_rows(
                rows, source_row=source_row
            )
            changed_logits = _project_same_shape(adapter, counterfactual)
            source_counterfactuals.append(
                {
                    "source_row_index": source_row,
                    "source_row_sha256": _tensor_sha256(rows[source_row]),
                    "mutated_source_row_sha256": _tensor_sha256(
                        counterfactual[source_row]
                    ),
                    "counterfactual_input_sha256": _tensor_sha256(
                        counterfactual
                    ),
                    "counterfactual_logits_sha256": _tensor_sha256(
                        changed_logits
                    ),
                    "mutated_source_element_count": hidden_width,
                    "preserved_target_row_count": row_count - 1,
                    "minimum_absolute_coordinate_change": float(
                        source_delta.min().item()
                    ),
                    "maximum_absolute_coordinate_change": float(
                        source_delta.max().item()
                    ),
                    "source_mutation_l2": float(
                        torch.linalg.vector_norm(source_delta).item()
                    ),
                }
            )
            for target_row in range(row_count):
                if target_row == source_row:
                    continue
                baseline_row = baseline_logits[target_row].double()
                changed_row = changed_logits[target_row].double()
                difference = baseline_row - changed_row
                max_abs = float(difference.abs().max().item())
                rms = float(torch.sqrt(difference.square().mean()).item())
                reference_max_abs = float(baseline_row.abs().max().item())
                allowed = atol + rtol * reference_max_abs
                directed_pair_checks.append(
                    {
                        "source_row_index": source_row,
                        "target_row_index": target_row,
                        "max_abs": max_abs,
                        "rms": rms,
                        "target_reference_max_abs": reference_max_abs,
                        "allowed": allowed,
                        "max_abs_over_allowed": (
                            max_abs / allowed
                            if allowed > 0.0
                            else (0.0 if max_abs == 0.0 else None)
                        ),
                        "passed": max_abs <= allowed,
                    }
                )

    failing = [
        {
            "source_row_index": int(check["source_row_index"]),
            "target_row_index": int(check["target_row_index"]),
        }
        for check in directed_pair_checks
        if check["passed"] is not True
    ]
    worst = max(
        directed_pair_checks,
        key=lambda check: (
            float(check["max_abs_over_allowed"])
            if check["max_abs_over_allowed"] is not None
            else math.inf
        ),
    )
    body: dict[str, object] = {
        "schema": _SCHEMA,
        "scientific_role": (
            "solver_authorization"
            if solver_authorization
            else "diagnostic_only_not_solver_authorization"
        ),
        "changes_solver_authorization": solver_authorization,
        "probe_name": probe_name,
        "method": _METHOD,
        "counterfactual_policy": dict(_COUNTERFACTUAL_POLICY),
        "row_count": row_count,
        "hidden_width": hidden_width,
        "vocabulary_width": int(baseline_logits.shape[1]),
        "projection_input_shape": [1, row_count, hidden_width],
        "projection_output_shape": [1, row_count, int(baseline_logits.shape[1])],
        "projection_call_count": 1 + row_count,
        "baseline_call_count": 1,
        "counterfactual_call_count": row_count,
        "directed_pair_order": _PAIR_ORDER,
        "directed_pair_count": row_count * (row_count - 1),
        "mutation_scale": mutation_scale,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "input_rows_sha256": _tensor_sha256(rows),
        "baseline_logits_sha256": _tensor_sha256(baseline_logits),
        "source_counterfactuals": source_counterfactuals,
        "directed_pair_checks": directed_pair_checks,
        "failing_directed_pair_count": len(failing),
        "failing_directed_pairs": failing,
        "worst_source_row_index": int(worst["source_row_index"]),
        "worst_target_row_index": int(worst["target_row_index"]),
        "worst_max_abs": float(worst["max_abs"]),
        "worst_rms": float(worst["rms"]),
        "worst_max_abs_over_allowed": worst["max_abs_over_allowed"],
        "passed": not failing,
        "contains_tensor_payloads": False,
    }
    body["receipt_sha256"] = _receipt_sha256(body)
    if _contains_tensor(body):
        raise RuntimeError("same-shape locality receipt contains a tensor")
    return SameShapeOffRowLocalityAudit(
        baseline_logits=baseline_logits,
        receipt=body,
    )


def validate_same_shape_off_row_locality_receipt(
    receipt: Mapping[str, object],
) -> None:
    """Fail closed on structural or authenticated receipt drift."""

    if not isinstance(receipt, Mapping):
        raise TypeError("same-shape locality receipt must be a mapping")
    body = dict(receipt)
    digest = body.pop("receipt_sha256", None)
    if not isinstance(digest, str) or digest != _receipt_sha256(body):
        raise ValueError("same-shape locality receipt hash mismatch")
    row_count = body.get("row_count")
    hidden_width = body.get("hidden_width")
    vocabulary_width = body.get("vocabulary_width")
    source_counterfactuals = body.get("source_counterfactuals")
    directed_pair_checks = body.get("directed_pair_checks")
    mutation_scale = body.get("mutation_scale")
    atol = body.get("absolute_tolerance")
    rtol = body.get("relative_tolerance")
    scientific_role = body.get("scientific_role")
    changes_solver_authorization = body.get("changes_solver_authorization")
    if (
        set(body) != _RECEIPT_BODY_FIELDS
        or body.get("schema") != _SCHEMA
        or not (
            (
                scientific_role
                == "diagnostic_only_not_solver_authorization"
                and changes_solver_authorization is False
            )
            or (
                scientific_role == "solver_authorization"
                and changes_solver_authorization is True
            )
        )
        or not isinstance(body.get("probe_name"), str)
        or not body["probe_name"]
        or body.get("method") != _METHOD
        or not _valid_counterfactual_policy(
            body.get("counterfactual_policy")
        )
        or body.get("directed_pair_order") != _PAIR_ORDER
        or type(row_count) is not int
        or row_count < 2
        or type(hidden_width) is not int
        or hidden_width < 1
        or type(vocabulary_width) is not int
        or vocabulary_width < 2
        or not isinstance(mutation_scale, float)
        or not math.isfinite(mutation_scale)
        or mutation_scale < 1.0
        or not isinstance(atol, float)
        or not math.isfinite(atol)
        or atol < 0.0
        or not isinstance(rtol, float)
        or not math.isfinite(rtol)
        or rtol < 0.0
        or not _exact_int_shape(
            body.get("projection_input_shape"),
            [1, row_count, hidden_width],
        )
        or not _exact_int_shape(
            body.get("projection_output_shape"),
            [1, row_count, vocabulary_width],
        )
        or type(body.get("projection_call_count")) is not int
        or body.get("projection_call_count") != row_count + 1
        or type(body.get("baseline_call_count")) is not int
        or body.get("baseline_call_count") != 1
        or type(body.get("counterfactual_call_count")) is not int
        or body.get("counterfactual_call_count") != row_count
        or type(body.get("directed_pair_count")) is not int
        or body.get("directed_pair_count") != row_count * (row_count - 1)
        or not isinstance(source_counterfactuals, list)
        or len(source_counterfactuals) != row_count
        or not isinstance(directed_pair_checks, list)
        or len(directed_pair_checks) != row_count * (row_count - 1)
        or body.get("contains_tensor_payloads") is not False
        or not _is_sha256(body.get("input_rows_sha256"))
        or not _is_sha256(body.get("baseline_logits_sha256"))
        or _contains_tensor(body)
    ):
        raise ValueError("same-shape locality receipt structure drifted")
    for index, source in enumerate(source_counterfactuals):
        if (
            not isinstance(source, Mapping)
            or set(source) != _SOURCE_COUNTERFACTUAL_FIELDS
            or type(source.get("source_row_index")) is not int
            or source.get("source_row_index") != index
            or type(source.get("mutated_source_element_count")) is not int
            or source.get("mutated_source_element_count") != hidden_width
            or type(source.get("preserved_target_row_count")) is not int
            or source.get("preserved_target_row_count") != row_count - 1
            or not _is_sha256(source.get("source_row_sha256"))
            or not _is_sha256(source.get("mutated_source_row_sha256"))
            or not _is_sha256(source.get("counterfactual_input_sha256"))
            or not _is_sha256(source.get("counterfactual_logits_sha256"))
            or source.get("source_row_sha256")
            == source.get("mutated_source_row_sha256")
            or not isinstance(
                source.get("minimum_absolute_coordinate_change"), float
            )
            or not isinstance(
                source.get("maximum_absolute_coordinate_change"), float
            )
            or not isinstance(source.get("source_mutation_l2"), float)
            or not all(
                math.isfinite(float(source[name]))
                for name in (
                    "minimum_absolute_coordinate_change",
                    "maximum_absolute_coordinate_change",
                    "source_mutation_l2",
                )
            )
            or float(source["minimum_absolute_coordinate_change"])
            < 0.5 * mutation_scale
            or float(source["maximum_absolute_coordinate_change"])
            < float(source["minimum_absolute_coordinate_change"])
            or float(source["source_mutation_l2"])
            < float(source["maximum_absolute_coordinate_change"])
            or float(source["source_mutation_l2"])
            < math.sqrt(hidden_width)
            * float(source["minimum_absolute_coordinate_change"])
        ):
            raise ValueError("same-shape source catalog drifted")
    failures: list[dict[str, int]] = []
    expected_pair_index = 0
    for source_row in range(row_count):
        for target_row in range(row_count):
            if source_row == target_row:
                continue
            check = directed_pair_checks[expected_pair_index]
            expected_pair_index += 1
            if (
                not isinstance(check, Mapping)
                or set(check) != _DIRECTED_PAIR_FIELDS
                or type(check.get("source_row_index")) is not int
                or check.get("source_row_index") != source_row
                or type(check.get("target_row_index")) is not int
                or check.get("target_row_index") != target_row
                or not isinstance(check.get("passed"), bool)
            ):
                raise ValueError("same-shape directed-pair catalog drifted")
            max_abs = check.get("max_abs")
            rms = check.get("rms")
            reference = check.get("target_reference_max_abs")
            allowed = check.get("allowed")
            expected_allowed = float(atol) + float(rtol) * float(reference)
            expected_ratio = (
                float(max_abs) / expected_allowed
                if expected_allowed > 0.0
                else (0.0 if float(max_abs) == 0.0 else None)
            )
            observed_ratio = check.get("max_abs_over_allowed")
            if (
                not all(
                    isinstance(value, float) and math.isfinite(value)
                    for value in (max_abs, rms, reference, allowed)
                )
                or any(
                    float(value) < 0.0
                    for value in (max_abs, rms, reference, allowed)
                )
                or float(allowed) != expected_allowed
                or (
                    observed_ratio is not None
                    and not isinstance(observed_ratio, float)
                )
                or observed_ratio != expected_ratio
                or check["passed"] is not (float(max_abs) <= float(allowed))
            ):
                raise ValueError("same-shape directed-pair metric drifted")
            if check["passed"] is not True:
                failures.append(
                    {
                        "source_row_index": source_row,
                        "target_row_index": target_row,
                    }
                )
    worst = max(
        directed_pair_checks,
        key=lambda check: (
            float(check["max_abs_over_allowed"])
            if check["max_abs_over_allowed"] is not None
            else math.inf
        ),
    )
    reported_failures = body.get("failing_directed_pairs")
    worst_ratio = body.get("worst_max_abs_over_allowed")
    if (
        not isinstance(reported_failures, list)
        or any(
            not isinstance(failure, Mapping)
            or set(failure) != {"source_row_index", "target_row_index"}
            or type(failure.get("source_row_index")) is not int
            or type(failure.get("target_row_index")) is not int
            for failure in reported_failures
        )
        or reported_failures != failures
        or type(body.get("failing_directed_pair_count")) is not int
        or body.get("failing_directed_pair_count") != len(failures)
        or body.get("passed") is not (not failures)
        or type(body.get("worst_source_row_index")) is not int
        or body.get("worst_source_row_index") != worst["source_row_index"]
        or type(body.get("worst_target_row_index")) is not int
        or body.get("worst_target_row_index") != worst["target_row_index"]
        or not isinstance(body.get("worst_max_abs"), float)
        or body.get("worst_max_abs") != worst["max_abs"]
        or not isinstance(body.get("worst_rms"), float)
        or body.get("worst_rms") != worst["rms"]
        or (worst_ratio is not None and not isinstance(worst_ratio, float))
        or worst_ratio != worst["max_abs_over_allowed"]
    ):
        raise ValueError("same-shape locality decision drifted")


__all__ = [
    "SameShapeOffRowLocalityAudit",
    "audit_same_shape_off_row_locality",
    "validate_same_shape_off_row_locality_receipt",
]
