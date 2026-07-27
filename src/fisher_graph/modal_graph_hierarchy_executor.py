"""Reference source/candidate/shadow execution for hierarchical generators.

The v1 hierarchy is deliberately conservative.  Its immediate child graph
remains resident as a fallback, so candidate-only active MAC reductions are
*logical* reductions and not deployed storage savings.  ``shadow`` executes
both paths, returns the immediate-child path byte-for-byte, and records
Fisher-weighted candidate error.  ``adaptive_validation`` may return the
candidate below a threshold and expand to that child above it.  This executor
does not claim a transitive fallback to the original fine leaf graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .modal_graph_hierarchy import HierarchicalModalGenerator


__all__ = [
    "HierarchyExecution",
    "HierarchyExecutionAccounting",
    "HierarchySourceResources",
    "ModalGraphHierarchyExecutor",
]


@dataclass(frozen=True, slots=True)
class HierarchySourceResources:
    """External accounting for the exact immediate child graph."""

    source_artifact_sha256: str
    learned_parameter_count: int
    linear_macs_per_row: int
    storage_bytes: int

    def __post_init__(self) -> None:
        from .modal_graph_hierarchy import _require_sha256

        _require_sha256(
            self.source_artifact_sha256,
            label="source resources artifact digest",
        )
        for label, value in (
            ("learned_parameter_count", self.learned_parameter_count),
            ("linear_macs_per_row", self.linear_macs_per_row),
            ("storage_bytes", self.storage_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class HierarchyExecutionAccounting:
    """Separate logical active costs from immediate-child residency."""

    execution_mode: str
    source_learned_parameter_count: int
    source_linear_macs_per_row: int
    source_storage_bytes: int
    candidate_stored_scalar_count: int
    candidate_linear_macs_per_row: int
    candidate_storage_bytes_float64: int
    validation_metric_macs_per_row: int
    validation_metric_storage_bytes_float64: int
    active_linear_macs_per_row: int
    resident_storage_bytes: int
    exact_child_fallback_resident: bool
    transitive_leaf_fallback_authorized: bool
    logical_candidate_parameter_delta: int
    logical_candidate_mac_delta: int
    deployed_storage_reduction_authorized: bool
    latency_reduction_claim: bool

    def __post_init__(self) -> None:
        if self.execution_mode not in {
            "source",
            "candidate",
            "shadow",
            "adaptive_validation",
        }:
            raise ValueError("unsupported hierarchy execution mode")
        nonnegative = (
            "source_learned_parameter_count",
            "source_linear_macs_per_row",
            "source_storage_bytes",
            "candidate_stored_scalar_count",
            "candidate_linear_macs_per_row",
            "candidate_storage_bytes_float64",
            "validation_metric_macs_per_row",
            "validation_metric_storage_bytes_float64",
            "active_linear_macs_per_row",
            "resident_storage_bytes",
        )
        for label in nonnegative:
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if (
            self.exact_child_fallback_resident is not True
            or self.transitive_leaf_fallback_authorized is not False
            or self.deployed_storage_reduction_authorized is not False
            or self.latency_reduction_claim is not False
        ):
            raise ValueError(
                "v1 hierarchy accounting must retain the immediate child, "
                "deny transitive fallback authority, and make no deployed "
                "storage or latency claim"
            )
        validation_storage = (
            self.validation_metric_storage_bytes_float64
            if self.execution_mode in {"shadow", "adaptive_validation"}
            else 0
        )
        expected_resident = (
            self.source_storage_bytes
            + self.candidate_storage_bytes_float64
            + validation_storage
        )
        if self.resident_storage_bytes != expected_resident:
            raise ValueError("resident storage must include source and candidate")
        expected_active = {
            "source": self.source_linear_macs_per_row,
            "candidate": self.candidate_linear_macs_per_row,
            "shadow": (
                self.source_linear_macs_per_row
                + self.candidate_linear_macs_per_row
                + self.validation_metric_macs_per_row
            ),
            "adaptive_validation": (
                self.source_linear_macs_per_row
                + self.candidate_linear_macs_per_row
                + self.validation_metric_macs_per_row
            ),
        }[self.execution_mode]
        if self.active_linear_macs_per_row != expected_active:
            raise ValueError("active MAC accounting does not match mode")
        if self.logical_candidate_parameter_delta != (
            self.candidate_stored_scalar_count
            - self.source_learned_parameter_count
        ):
            raise ValueError("logical candidate parameter delta is inconsistent")
        if self.logical_candidate_mac_delta != (
            self.candidate_linear_macs_per_row
            - self.source_linear_macs_per_row
        ):
            raise ValueError("logical candidate MAC delta is inconsistent")


@dataclass(frozen=True, slots=True)
class HierarchyExecution:
    """Authoritative outputs plus optional validation instrumentation."""

    outputs: dict[str, Tensor]
    execution_mode: str
    authoritative_path: str
    expanded_to_source: bool
    accounting: HierarchyExecutionAccounting
    candidate_outputs: dict[str, Tensor] | None = None
    weighted_error_per_row: Tensor | None = None
    maximum_weighted_error: float | None = None

    def __post_init__(self) -> None:
        if self.execution_mode != self.accounting.execution_mode:
            raise ValueError("execution and accounting modes differ")
        expected_paths = {
            "source": {"source"},
            "candidate": {"candidate"},
            "shadow": {"source_shadow"},
            "adaptive_validation": {"candidate", "source_expansion"},
        }
        if self.authoritative_path not in expected_paths[self.execution_mode]:
            raise ValueError("authoritative path does not match execution mode")
        if (
            self.execution_mode in {"source", "candidate"}
            and self.expanded_to_source
        ):
            raise ValueError("source/candidate mode cannot report expansion")
        if self.execution_mode == "shadow":
            if self.authoritative_path != "source_shadow":
                raise ValueError("shadow must return source-authoritative output")
            if (
                self.candidate_outputs is None
                or self.weighted_error_per_row is None
                or self.maximum_weighted_error is None
            ):
                raise ValueError("shadow execution requires candidate metrics")
        if self.execution_mode == "adaptive_validation":
            if (
                self.candidate_outputs is None
                or self.weighted_error_per_row is None
                or self.maximum_weighted_error is None
            ):
                raise ValueError(
                    "adaptive validation requires candidate metrics"
                )
            if self.expanded_to_source != (
                self.authoritative_path == "source_expansion"
            ):
                raise ValueError("adaptive expansion metadata is inconsistent")
        if self.weighted_error_per_row is not None:
            if (
                not isinstance(self.weighted_error_per_row, Tensor)
                or not self.weighted_error_per_row.is_floating_point()
                or not torch.isfinite(self.weighted_error_per_row).all()
                or (self.weighted_error_per_row < 0).any()
            ):
                raise ValueError(
                    "weighted_error_per_row must be finite and nonnegative"
                )
            actual_max = float(
                self.weighted_error_per_row.max().item()
                if self.weighted_error_per_row.numel()
                else 0.0
            )
            if (
                self.maximum_weighted_error is None
                or not math.isclose(
                    self.maximum_weighted_error,
                    actual_max,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("maximum weighted error is inconsistent")


class ModalGraphHierarchyExecutor:
    """Execute one hierarchical generator in validation-safe modes."""

    def __init__(
        self,
        generator: HierarchicalModalGenerator,
        *,
        source_resources: HierarchySourceResources | None = None,
    ) -> None:
        if not isinstance(generator, HierarchicalModalGenerator):
            raise TypeError("generator must be HierarchicalModalGenerator")
        generator.validate_integrity()
        if source_resources is None:
            source_resources = HierarchySourceResources(
                source_artifact_sha256=(
                    generator.exact_child_fallback_sha256
                ),
                learned_parameter_count=(
                    generator.child_graph.stored_scalar_count
                ),
                linear_macs_per_row=generator.child_graph.macs_per_row,
                storage_bytes=8 * generator.child_graph.stored_scalar_count,
            )
        elif (
            source_resources.source_artifact_sha256
            != generator.exact_child_fallback_sha256
        ):
            raise ValueError(
                "source resource accounting is stale for child fallback"
            )
        self.generator = generator
        self.source_resources = source_resources

    def accounting(self, mode: str) -> HierarchyExecutionAccounting:
        self.generator.validate_integrity()
        if mode not in {
            "source",
            "candidate",
            "shadow",
            "adaptive_validation",
        }:
            raise ValueError("unsupported hierarchy execution mode")
        candidate_scalars = (
            self.generator.decomposition.candidate_stored_scalar_count
        )
        candidate_macs = (
            self.generator.decomposition.candidate_macs_per_row
        )
        validation_metric_macs = sum(
            port.width * port.width + port.width
            for port in (
                self.generator.decomposition.source_transfer.output_ports
            )
        )
        validation_metric_storage = 8 * sum(
            moments.fisher.numel()
            for moments in self.generator.decomposition.output_moments
        )
        active_macs = {
            "source": self.source_resources.linear_macs_per_row,
            "candidate": candidate_macs,
            "shadow": (
                self.source_resources.linear_macs_per_row
                + candidate_macs
                + validation_metric_macs
            ),
            "adaptive_validation": (
                self.source_resources.linear_macs_per_row
                + candidate_macs
                + validation_metric_macs
            ),
        }[mode]
        candidate_bytes = 8 * candidate_scalars
        return HierarchyExecutionAccounting(
            execution_mode=mode,
            source_learned_parameter_count=(
                self.source_resources.learned_parameter_count
            ),
            source_linear_macs_per_row=(
                self.source_resources.linear_macs_per_row
            ),
            source_storage_bytes=self.source_resources.storage_bytes,
            candidate_stored_scalar_count=candidate_scalars,
            candidate_linear_macs_per_row=candidate_macs,
            candidate_storage_bytes_float64=candidate_bytes,
            validation_metric_macs_per_row=validation_metric_macs,
            validation_metric_storage_bytes_float64=(
                validation_metric_storage
            ),
            active_linear_macs_per_row=active_macs,
            resident_storage_bytes=(
                self.source_resources.storage_bytes
                + candidate_bytes
                + (
                    validation_metric_storage
                    if mode in {"shadow", "adaptive_validation"}
                    else 0
                )
            ),
            exact_child_fallback_resident=True,
            transitive_leaf_fallback_authorized=False,
            logical_candidate_parameter_delta=(
                candidate_scalars
                - self.source_resources.learned_parameter_count
            ),
            logical_candidate_mac_delta=(
                candidate_macs
                - self.source_resources.linear_macs_per_row
            ),
            deployed_storage_reduction_authorized=False,
            latency_reduction_claim=False,
        )

    def _weighted_error(
        self,
        source: Mapping[str, Tensor],
        candidate: Mapping[str, Tensor],
    ) -> Tensor:
        result: Tensor | None = None
        output_moments = {
            value.port.name: value
            for value in self.generator.decomposition.output_moments
        }
        for port in self.generator.decomposition.source_transfer.output_ports:
            source_value = source[port.name]
            candidate_value = candidate[port.name]
            difference = candidate_value - source_value
            fisher = output_moments[port.name].fisher.to(
                device=difference.device,
                dtype=difference.dtype,
            )
            contribution = torch.einsum(
                "...i,ij,...j->...",
                difference,
                fisher,
                difference,
            ).clamp_min(0)
            result = contribution if result is None else result + contribution
        assert result is not None
        return result

    def execute(
        self,
        inputs: Mapping[str, Tensor],
        *,
        mode: str,
        expansion_weighted_error_threshold: float | None = None,
    ) -> HierarchyExecution:
        # Frozen dataclasses do not make their contained tensors immutable.
        # Re-authenticate immediately before every execution so a mutated
        # restriction, prolongation, edge, or source matrix fails closed.
        self.generator.validate_integrity()
        if mode not in {
            "source",
            "candidate",
            "shadow",
            "adaptive_validation",
        }:
            raise ValueError("unsupported hierarchy execution mode")
        if mode in {"shadow", "adaptive_validation"}:
            if (
                not isinstance(expansion_weighted_error_threshold, float)
                or not math.isfinite(expansion_weighted_error_threshold)
                or expansion_weighted_error_threshold < 0.0
            ):
                raise ValueError(
                    "validation modes require a finite nonnegative "
                    "expansion threshold"
                )
        elif expansion_weighted_error_threshold is not None:
            raise ValueError(
                "expansion threshold is only valid for validation modes"
            )

        accounting = self.accounting(mode)
        if mode == "source":
            source = self.generator.child_graph.execute(inputs)
            return HierarchyExecution(
                outputs=source,
                execution_mode=mode,
                authoritative_path="source",
                expanded_to_source=False,
                accounting=accounting,
            )
        if mode == "candidate":
            candidate = (
                self.generator.decomposition.execute_candidate(inputs)
            )
            return HierarchyExecution(
                outputs=candidate,
                execution_mode=mode,
                authoritative_path="candidate",
                expanded_to_source=False,
                accounting=accounting,
            )

        source = self.generator.child_graph.execute(inputs)
        candidate = self.generator.decomposition.execute_candidate(inputs)
        weighted_error = self._weighted_error(source, candidate)
        maximum = float(
            weighted_error.max().item() if weighted_error.numel() else 0.0
        )
        assert expansion_weighted_error_threshold is not None
        expanded = maximum > expansion_weighted_error_threshold
        if mode == "shadow":
            outputs = source
            path = "source_shadow"
        elif expanded:
            outputs = source
            path = "source_expansion"
        else:
            outputs = candidate
            path = "candidate"
        return HierarchyExecution(
            outputs=outputs,
            execution_mode=mode,
            authoritative_path=path,
            expanded_to_source=expanded,
            accounting=accounting,
            candidate_outputs=candidate,
            weighted_error_per_row=weighted_error,
            maximum_weighted_error=maximum,
        )
