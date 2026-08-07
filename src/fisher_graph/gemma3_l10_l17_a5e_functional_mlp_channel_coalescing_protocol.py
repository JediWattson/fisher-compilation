"""Declarative A5e plan for functional Gemma MLP channel coalescing.

This is a test scaffold, not an experiment runner.  It loads no model, reads
no prompts, stores no tensor values, and mutates no weights.

One natural gated-MLP channel owns the matching ``gate_proj`` row,
``up_proj`` row, and ``down_proj`` column.  A future A5e runner may use
fit-only grouped Fisher and channel-contribution Jacobians to assign donor
channels to survivors.  Because the gated activation is nonlinear, the plan
requires a functional survivor refit; it never treats row averaging as an
exact merge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math


__all__ = [
    "A5E_ARM_IDS",
    "A5E_MERGE_RATE_LADDER",
    "A5E_PROTOCOL_FORMAT_VERSION",
    "A5E_PROTOCOL_SCHEMA",
    "A5eFunctionalMlpChannelCoalescingProtocol",
    "A5eMergeRateTrial",
    "ChannelMerge",
    "GatedMlpProjectionShapes",
    "PhysicalChannelCoalescingContract",
    "build_a5e_functional_mlp_channel_coalescing_protocol",
    "validate_compacted_weight_mocks",
    "validate_matched_naive_deletion",
]


A5E_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5e_functional_mlp_channel_"
    "coalescing.protocol.v2"
)
A5E_PROTOCOL_FORMAT_VERSION = 2
A5E_MERGE_RATE_LADDER = (0.01, 0.02, 0.05, 0.10)
A5E_ARM_IDS = (
    "native_intact",
    "native_intact_plus_residual_diagnostic",
    "compiled_all_modes_intact",
    "approximated_all_modes_graph",
    "chart_conditioned_all_modes_graph",
    "matched_naive_deletion",
    "fisher_jacobian_functional_coalescing",
)

_GROUPING_DEFINITION = (
    "gate_proj_row_j+up_proj_row_j+down_proj_column_j"
)
_PHYSICAL_AXES = {
    "gate_proj": "compact_matching_rows",
    "up_proj": "compact_matching_rows",
    "down_proj": "compact_matching_columns",
}


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _matrix_shape(value: object, *, label: str) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if (
        isinstance(shape, (str, bytes))
        or not isinstance(shape, Sequence)
        or len(shape) != 2
    ):
        raise ValueError(f"{label} must expose a two-dimensional shape")
    try:
        result = (int(shape[0]), int(shape[1]))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} shape must contain integers") from error
    if any(dimension <= 0 for dimension in result):
        raise ValueError(f"{label} shape must be strictly positive")
    return result


@dataclass(frozen=True, slots=True)
class GatedMlpProjectionShapes:
    """Shape-only contract for one bias-free gated MLP."""

    gate_proj: tuple[int, int]
    up_proj: tuple[int, int]
    down_proj: tuple[int, int]

    def __post_init__(self) -> None:
        for label, shape in (
            ("gate_proj", self.gate_proj),
            ("up_proj", self.up_proj),
            ("down_proj", self.down_proj),
        ):
            if (
                type(shape) is not tuple
                or len(shape) != 2
                or any(type(value) is not int or value <= 0 for value in shape)
            ):
                raise ValueError(
                    f"{label} must be a positive two-dimensional shape tuple"
                )
        if self.gate_proj != self.up_proj:
            raise ValueError("gate_proj and up_proj shapes must match")
        if self.down_proj[1] != self.gate_proj[0]:
            raise ValueError(
                "down_proj columns must match gate/up intermediate rows"
            )

    @classmethod
    def from_weight_mocks(
        cls,
        *,
        gate_weight: object,
        up_weight: object,
        down_weight: object,
    ) -> GatedMlpProjectionShapes:
        """Read tensor-like shapes without importing a tensor backend."""

        return cls(
            gate_proj=_matrix_shape(gate_weight, label="gate_weight"),
            up_proj=_matrix_shape(up_weight, label="up_weight"),
            down_proj=_matrix_shape(down_weight, label="down_weight"),
        )

    @property
    def intermediate_width(self) -> int:
        return self.gate_proj[0]

    @property
    def input_width(self) -> int:
        return self.gate_proj[1]

    @property
    def output_width(self) -> int:
        return self.down_proj[0]

    @property
    def learned_parameter_count(self) -> int:
        return self.intermediate_width * (
            2 * self.input_width + self.output_width
        )

    @property
    def matrix_macs_per_token(self) -> int:
        return self.learned_parameter_count

    def compacted(self, removed_channel_count: int) -> GatedMlpProjectionShapes:
        removed = _positive_int(
            removed_channel_count,
            label="removed_channel_count",
        )
        remaining = self.intermediate_width - removed
        if remaining <= 0:
            raise ValueError("physical compaction must retain a channel")
        return type(self)(
            gate_proj=(remaining, self.input_width),
            up_proj=(remaining, self.input_width),
            down_proj=(self.output_width, remaining),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "gate_proj": list(self.gate_proj),
            "up_proj": list(self.up_proj),
            "down_proj": list(self.down_proj),
            "learned_parameter_count": self.learned_parameter_count,
            "matrix_macs_per_token": self.matrix_macs_per_token,
            "bias_policy": "weight_matrices_only_no_bias",
        }


@dataclass(frozen=True, slots=True)
class ChannelMerge:
    """A donor channel assigned to one retained survivor."""

    donor_channel: int
    survivor_channel: int

    def __post_init__(self) -> None:
        if (
            type(self.donor_channel) is not int
            or self.donor_channel < 0
            or type(self.survivor_channel) is not int
            or self.survivor_channel < 0
        ):
            raise ValueError("merge channel indices must be nonnegative integers")
        if self.donor_channel == self.survivor_channel:
            raise ValueError("a channel cannot merge into itself")


@dataclass(frozen=True, slots=True)
class PhysicalChannelCoalescingContract:
    """Topology and physical shape contract for a future functional refit."""

    original_shapes: GatedMlpProjectionShapes
    merges: tuple[ChannelMerge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.original_shapes, GatedMlpProjectionShapes):
            raise TypeError("original_shapes must be GatedMlpProjectionShapes")
        if (
            type(self.merges) is not tuple
            or not self.merges
            or any(not isinstance(value, ChannelMerge) for value in self.merges)
        ):
            raise ValueError("merges must be a nonempty ChannelMerge tuple")
        donors = tuple(value.donor_channel for value in self.merges)
        survivors = tuple(value.survivor_channel for value in self.merges)
        if len(donors) != len(set(donors)):
            raise ValueError("each donor may be removed only once")
        if set(donors) & set(survivors):
            raise ValueError("a removed donor cannot also be a survivor")
        if max((*donors, *survivors)) >= self.original_shapes.intermediate_width:
            raise ValueError("merge channel is outside the MLP width")
        if len(donors) >= self.original_shapes.intermediate_width:
            raise ValueError("coalescing must retain a channel")

    @property
    def donor_channels(self) -> tuple[int, ...]:
        return tuple(sorted(value.donor_channel for value in self.merges))

    @property
    def compacted_shapes(self) -> GatedMlpProjectionShapes:
        return self.original_shapes.compacted(len(self.merges))

    @property
    def removed_parameter_count(self) -> int:
        return (
            self.original_shapes.learned_parameter_count
            - self.compacted_shapes.learned_parameter_count
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "donor_channels": list(self.donor_channels),
            "survivor_channels": [
                value.survivor_channel for value in self.merges
            ],
            "compacted_shapes": self.compacted_shapes.state_dict(),
            "natural_channel_grouping": _GROUPING_DEFINITION,
            "physical_compaction_axes": dict(_PHYSICAL_AXES),
            "functional_survivor_refit_required": True,
            "direct_weight_averaging_is_exact": False,
            "removed_parameter_count": self.removed_parameter_count,
            "removed_matrix_macs_per_token": self.removed_parameter_count,
        }


def validate_compacted_weight_mocks(
    *,
    contract: PhysicalChannelCoalescingContract,
    gate_weight: object,
    up_weight: object,
    down_weight: object,
) -> None:
    """Require physical gate/up row and down column compaction."""

    try:
        observed = GatedMlpProjectionShapes.from_weight_mocks(
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=down_weight,
        )
    except ValueError as error:
        raise ValueError(
            "materialized gate/up rows and down columns do not match the plan"
        ) from error
    if observed != contract.compacted_shapes:
        raise ValueError(
            "materialized gate/up rows and down columns do not match the plan"
        )


def validate_matched_naive_deletion(
    *,
    contract: PhysicalChannelCoalescingContract,
    naive_deleted_channels: tuple[int, ...],
    expected_removed_channel_count: int,
) -> None:
    """Match naive deletion to the guided arm's donors and compact shape."""

    expected = _positive_int(
        expected_removed_channel_count,
        label="expected_removed_channel_count",
    )
    if naive_deleted_channels != contract.donor_channels:
        raise ValueError("naive deletion must remove the guided arm's donors")
    if len(contract.donor_channels) != expected:
        raise ValueError("comparison does not implement the requested rate")


@dataclass(frozen=True, slots=True)
class A5eMergeRateTrial:
    """Shape and resource expectation at one requested merge rate."""

    merge_rate: float
    original_shapes: GatedMlpProjectionShapes

    def __post_init__(self) -> None:
        if (
            type(self.merge_rate) is not float
            or not math.isfinite(self.merge_rate)
            or not 0.0 < self.merge_rate < 1.0
        ):
            raise ValueError("merge_rate must be a finite float in (0, 1)")
        if not isinstance(self.original_shapes, GatedMlpProjectionShapes):
            raise TypeError("original_shapes must be GatedMlpProjectionShapes")
        if self.removed_channel_count <= 0:
            raise ValueError("MLP width is too small for the rate ladder")

    @property
    def removed_channel_count(self) -> int:
        return math.floor(
            self.original_shapes.intermediate_width * self.merge_rate
        )

    @property
    def compacted_shapes(self) -> GatedMlpProjectionShapes:
        return self.original_shapes.compacted(self.removed_channel_count)

    @property
    def native_parameter_savings(self) -> int:
        return (
            self.original_shapes.learned_parameter_count
            - self.compacted_shapes.learned_parameter_count
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "merge_rate": self.merge_rate,
            "removed_channel_count": self.removed_channel_count,
            "actual_removed_fraction": (
                self.removed_channel_count
                / self.original_shapes.intermediate_width
            ),
            "compacted_shapes": self.compacted_shapes.state_dict(),
            "native_parameter_savings": self.native_parameter_savings,
            "native_matrix_macs_saved_per_token": (
                self.native_parameter_savings
            ),
            "matched_compressed_arms": list(A5E_ARM_IDS[-2:]),
        }


def _arm_specs() -> list[dict[str, object]]:
    return [
        {
            "arm_id": "native_intact",
            "native_mlp_intact": True,
            "residual_diagnostic": False,
            "physically_compacted": False,
            "compression_credit_allowed": False,
        },
        {
            "arm_id": "native_intact_plus_residual_diagnostic",
            "native_mlp_intact": True,
            "native_mlp_identity_preserved": True,
            "native_mlp_call_preserved": True,
            "residual_diagnostic": True,
            "residual_candidate_policy": (
                "reuse_identical_frozen_fit_only_residual_candidate"
            ),
            "native_specific_residual_refit_allowed": False,
            "residual_application_boundary": "layer.L.mlp.delta",
            "post_feedforward_rmsnorm_attached": True,
            "physically_compacted": False,
            "compression_credit_allowed": False,
        },
        {
            "arm_id": "compiled_all_modes_intact",
            "native_mlp_intact": False,
            "all_native_mode_triplets_materialized": True,
            "source_free_runtime": True,
            "source_projection_calls_expected": 0,
            "compiled_projection_calls_preserved": True,
            "residual_diagnostic": False,
            "physically_compacted": False,
            "removed_channel_count": 0,
            "compression_credit_allowed": False,
        },
        {
            "arm_id": "approximated_all_modes_graph",
            "native_mlp_intact": False,
            "all_native_modes_retained": True,
            "generator_family": "fit_only_affine_per_mode",
            "source_free_runtime": True,
            "residual_diagnostic": False,
            "physically_compacted": False,
            "removed_channel_count": 0,
            "compression_credit_allowed": False,
        },
        {
            "arm_id": "chart_conditioned_all_modes_graph",
            "native_mlp_intact": False,
            "all_native_modes_retained": True,
            "generator_family": "global_affine_plus_local_chart_edges",
            "routing_input": "normalized_hidden_state",
            "source_free_runtime": True,
            "residual_diagnostic": False,
            "physically_compacted": False,
            "removed_channel_count": 0,
            "compression_credit_allowed": False,
        },
        {
            "arm_id": "matched_naive_deletion",
            "native_mlp_intact": False,
            "residual_diagnostic": False,
            "physically_compacted": True,
            "functional_survivor_refit": False,
            "donors_match_guided_arm": True,
        },
        {
            "arm_id": "fisher_jacobian_functional_coalescing",
            "native_mlp_intact": False,
            "residual_diagnostic": False,
            "physically_compacted": True,
            "functional_survivor_refit": True,
            "selection": "fit_only_grouped_fisher_plus_channel_jacobian",
        },
    ]


@dataclass(frozen=True, slots=True)
class A5eFunctionalMlpChannelCoalescingProtocol:
    """A5e L10/L17 non-executable test plan."""

    original_shapes: GatedMlpProjectionShapes
    target_layer_ordinals: tuple[int, ...] = (10, 17)
    merge_rates: tuple[float, ...] = A5E_MERGE_RATE_LADDER

    def __post_init__(self) -> None:
        if not isinstance(self.original_shapes, GatedMlpProjectionShapes):
            raise TypeError("original_shapes must be GatedMlpProjectionShapes")
        if self.target_layer_ordinals != (10, 17):
            raise ValueError("A5e must preserve the frozen L10/L17 scope")
        if self.merge_rates != A5E_MERGE_RATE_LADDER:
            raise ValueError("merge-rate ladder must remain 1/2/5/10 percent")
        self.trials

    @property
    def trials(self) -> tuple[A5eMergeRateTrial, ...]:
        return tuple(
            A5eMergeRateTrial(rate, self.original_shapes)
            for rate in self.merge_rates
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": A5E_PROTOCOL_SCHEMA,
            "format_version": A5E_PROTOCOL_FORMAT_VERSION,
            "artifact_role": "a5e_test_scaffold_only",
            "actual_model_experiment_implemented": False,
            "model_weights_loaded": False,
            "tensor_values_stored": False,
            "target_layer_ordinals": list(self.target_layer_ordinals),
            "original_shapes": self.original_shapes.state_dict(),
            "natural_channel_grouping": _GROUPING_DEFINITION,
            "merge_rate_ladder": list(self.merge_rates),
            "arms": _arm_specs(),
            "trials": [trial.state_dict() for trial in self.trials],
            "scientific_order": [
                "fit_only_compute_grouped_fisher_and_channel_jacobians",
                "fit_only_construct_fisher_weighted_hidden_state_charts",
                "fit_only_jointly_fit_chart_to_mode_edge_bank",
                "fit_only_rank_and_assign_donors_to_survivors",
                "fit_only_refit_functional_survivor_triplets",
                "freeze_topology_weights_and_diagnostic_artifacts",
                "held_only_score_each_frozen_arm_once",
            ],
            "held_data_may_select_pairs": False,
            "held_data_may_refit_survivors": False,
            "residual_control_contract": {
                "candidate_reused_without_refit": True,
                "candidate_fit_role": "fit_only_then_frozen",
                "application_boundary": "layer.L.mlp.delta",
                "post_feedforward_rmsnorm_attached": True,
                "native_mlp_identity_and_call_preserved": True,
                "compression_credit_allowed": False,
            },
            "compiled_all_modes_control_contract": {
                "source_free_runtime": True,
                "all_native_mode_triplets_materialized": True,
                "source_projection_calls_expected": 0,
                "removed_channel_count": 0,
                "parameter_and_matrix_mac_savings": 0,
                "compression_credit_allowed": False,
            },
            "approximated_all_modes_graph_contract": {
                "all_native_modes_retained": True,
                "removed_channel_count": 0,
                "merged_channel_count": 0,
                "fit_role": "fit_only_then_frozen",
                "generator_family": "affine_per_mode",
                "held_data_may_refit_generators": False,
                "compression_credit_allowed": False,
            },
            "chart_conditioned_all_modes_graph_contract": {
                "all_native_modes_retained": True,
                "removed_channel_count": 0,
                "merged_channel_count": 0,
                "fit_role": "fit_only_then_frozen",
                "chart_coordinates": "B_c_transpose_times_h_minus_mu_c",
                "edge_gate": "hidden_state_conditioned_chart_membership",
                "edge_message": "affine_local_coordinate_residual",
                "held_data_may_refit_charts_or_edges": False,
                "compression_credit_allowed": False,
            },
            "nonlinear_merge_contract": {
                "direct_weight_averaging_is_exact": False,
                "fit_target": (
                    "native_donor_plus_survivor_channel_contribution_"
                    "at_mlp_output"
                ),
            },
            "required_metrics": [
                "held_mlp_output_nrmse",
                "held_channel_jvp_relative_error",
                "held_full_model_delta_nll_per_token",
                "held_native_to_candidate_kl_per_token",
                "held_top1_agreement_to_native",
                "physically_instantiated_learned_parameters",
                "physically_executed_matrix_macs_per_token",
                "chart_membership_entropy_and_occupancy",
                "chart_to_mode_interaction_parameter_and_mac_count",
            ],
            "claim_boundary": (
                "no_compression_fidelity_or_speed_claim_from_scaffold"
            ),
        }


def build_a5e_functional_mlp_channel_coalescing_protocol(
    *,
    hidden_width: int = 640,
    intermediate_width: int = 2048,
) -> A5eFunctionalMlpChannelCoalescingProtocol:
    """Build the default Gemma-3 A5e shape-only protocol."""

    hidden = _positive_int(hidden_width, label="hidden_width")
    intermediate = _positive_int(
        intermediate_width,
        label="intermediate_width",
    )
    return A5eFunctionalMlpChannelCoalescingProtocol(
        original_shapes=GatedMlpProjectionShapes(
            gate_proj=(intermediate, hidden),
            up_proj=(intermediate, hidden),
            down_proj=(hidden, intermediate),
        )
    )
