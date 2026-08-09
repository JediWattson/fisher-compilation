"""Preregister Gemma's heterogeneous whole-model generator-family ladder.

This setup-only protocol lifts the authenticated exact carrier runtime through
increasingly general generator-family representations before it permits any
compression fit.  It keeps an invocation, a reusable family, a physical atom
bank, and an edge occurrence as four distinct objects.

The module is deliberately standard-library only.  Building, validating, or
writing this protocol cannot load Gemma, open prompt data, tokenize text, fit a
coefficient, compile a model, or execute an experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA = (
    "fisher_graph.gemma3_heterogeneous_whole_model_generator_family."
    "protocol.v1"
)
GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION = 1
DEFAULT_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-heterogeneous-whole-model-generator-family-protocol-v1.json"
)
_HASH_DOMAIN = (
    b"fisher_graph:gemma3-heterogeneous-whole-model-generator-family:v1\0"
)

PINNED_MODEL_ID = "google/gemma-3-270m"
PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_SOURCE_SCALARS = 268_098_176
PINNED_LEXICAL_SCALARS = 167_772_160
PINNED_ATTENTION_SCALARS = 29_523_456
PINNED_MLP_SCALARS = 70_801_920
PINNED_TERMINAL_NORM_SCALARS = 640
PINNED_HIDDEN_WIDTH = 640
PINNED_MLP_WIDTH = 2_048
PINNED_VOCABULARY_SIZE = 262_144
PINNED_LAYER_COUNT = 18
PINNED_MUTATION_COUNT = 36
PINNED_EVENT_COUNT = 39
PINNED_EDGE_OCCURRENCE_COUNT = 75
PINNED_BODY_ATOM_COUNT = 36
PINNED_HETEROGENEOUS_ATOM_COUNT = 37
PINNED_GLOBAL_ATTENTION_BLOCKS = (5, 11, 17)
PINNED_WHOLE_MODEL_REDUCTION_TARGET = 0.20
PINNED_MAXIMUM_TARGET_CANDIDATE_SCALARS = 214_478_540
PINNED_MINIMUM_TARGET_REMOVED_SCALARS = 53_619_636

_MLP_INVOCATION_SCALARS = 3_933_440
_ATTENTION_INVOCATION_SCALARS = 1_640_192
_MLP_RESIDUAL_FIXED_SCALARS = 1_280
_MLP_RESIDUAL_RANK_SLOPE = 8_064
_ATTENTION_RESIDUAL_FIXED_SCALARS = 1_792
_ATTENTION_RESIDUAL_RANK_SLOPE = 3_840
_RESIDUAL_RANKS = (256, 128, 64, 32)
_HETEROGENEOUS_HIDDEN_RANKS = (640, 608, 576, 544, 512, 496)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _protocol_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_HASH_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _attention_contract(block_ordinal: int) -> str:
    return (
        "global_causal_q4_kv1_d256_qk_norm_rope_global"
        if block_ordinal in PINNED_GLOBAL_ATTENTION_BLOCKS
        else "sliding_causal_512_q4_kv1_d256_qk_norm_rope_local"
    )


def _family_id(component_kind: str, block_ordinal: int) -> str:
    if component_kind == "mlp":
        return "family.mlp.gated_gelu_h640_i2048"
    if component_kind != "attention":
        raise ValueError("unknown generator component kind")
    return (
        "family.attention.global_causal"
        if block_ordinal in PINNED_GLOBAL_ATTENTION_BLOCKS
        else "family.attention.sliding_causal_512"
    )


def _atom_bank_id(component_kind: str, block_ordinal: int) -> str:
    if component_kind == "mlp":
        return "atom_bank.mlp.full_rank"
    if component_kind != "attention":
        raise ValueError("unknown generator component kind")
    return (
        "atom_bank.attention.global.full_rank"
        if block_ordinal in PINNED_GLOBAL_ATTENTION_BLOCKS
        else "atom_bank.attention.sliding.full_rank"
    )


def _body_invocations() -> list[dict[str, object]]:
    invocations: list[dict[str, object]] = []
    for block in range(PINNED_LAYER_COUNT):
        for component_kind in ("attention", "mlp"):
            mutation_index = len(invocations) + 1
            invocation_id = f"mutation.{mutation_index}"
            invocations.append(
                {
                    "invocation_id": invocation_id,
                    "invocation_ordinal": mutation_index,
                    "event_kind": "body_mutation",
                    "block_ordinal": block,
                    "component_kind": component_kind,
                    "operator_contract": (
                        _attention_contract(block)
                        if component_kind == "attention"
                        else "token_local_gated_gelu_h640_i2048"
                    ),
                    "family_id": _family_id(component_kind, block),
                    "atom_bank_id": _atom_bank_id(component_kind, block),
                    "atom_id": f"atom.{component_kind}.layer.{block}",
                    "invocation_adapter_id": f"adapter.{invocation_id}",
                    "primary_edge_occurrence_id": (
                        f"occurrence.primary.{invocation_id}"
                    ),
                    "input_carrier_version": mutation_index - 1,
                    "output_carrier_version": mutation_index,
                    "provenance_only_source_mutation_id": (
                        f"layer.{block}."
                        + (
                            "attention"
                            if component_kind == "attention"
                            else "feed_forward"
                        )
                    ),
                }
            )
    return invocations


def _full_invocations() -> list[dict[str, object]]:
    ingress = {
        "invocation_id": "lexical.ingress",
        "invocation_ordinal": 0,
        "event_kind": "lexical_ingress",
        "block_ordinal": None,
        "component_kind": "lexical",
        "operator_contract": "tied_token_embedding_v262144_h640_ingress",
        "family_id": "family.lexical.tied",
        "atom_bank_id": "atom_bank.lexical.tied.full_rank",
        "atom_id": "atom.lexical.tied",
        "invocation_adapter_id": "adapter.lexical.ingress",
        "primary_edge_occurrence_id": "occurrence.lexical.ingress",
        "input_carrier_version": None,
        "output_carrier_version": 0,
        "provenance_only_source_mutation_id": "model.embed_tokens",
    }
    terminal = {
        "invocation_id": "terminal.norm_softcap",
        "invocation_ordinal": 37,
        "event_kind": "terminal_exact",
        "block_ordinal": None,
        "component_kind": "terminal",
        "operator_contract": "terminal_RMSNorm_exact_then_logit_softcap_exact",
        "family_id": None,
        "atom_bank_id": "atom_bank.terminal.exact",
        "atom_id": "atom.terminal.RMSNorm",
        "invocation_adapter_id": "adapter.terminal.norm_softcap",
        "primary_edge_occurrence_id": "occurrence.terminal.norm_softcap",
        "input_carrier_version": PINNED_MUTATION_COUNT,
        "output_carrier_version": PINNED_MUTATION_COUNT,
        "provenance_only_source_mutation_id": "model.norm_and_logit_softcap",
    }
    egress = {
        "invocation_id": "lexical.egress",
        "invocation_ordinal": 38,
        "event_kind": "lexical_egress",
        "block_ordinal": None,
        "component_kind": "lexical",
        "operator_contract": "tied_token_embedding_v262144_h640_egress",
        "family_id": "family.lexical.tied",
        "atom_bank_id": "atom_bank.lexical.tied.full_rank",
        "atom_id": "atom.lexical.tied",
        "invocation_adapter_id": "adapter.lexical.egress",
        "primary_edge_occurrence_id": "occurrence.lexical.egress",
        "input_carrier_version": PINNED_MUTATION_COUNT,
        "output_carrier_version": None,
        "provenance_only_source_mutation_id": "model.embed_tokens_tied_lm_head",
    }
    return [ingress, *_body_invocations(), terminal, egress]


def _edge_occurrences(
    invocations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    occurrences: list[dict[str, object]] = []
    for invocation in invocations:
        event_kind = invocation["event_kind"]
        if event_kind == "body_mutation":
            mutation_index = invocation["invocation_ordinal"]
            occurrences.append(
                {
                    "edge_occurrence_id": f"occurrence.carry.mutation.{mutation_index}",
                    "occurrence_kind": "identity_carry",
                    "invocation_id": invocation["invocation_id"],
                    "atom_id": None,
                    "stored_scalars": 0,
                    "input_carrier_version": invocation["input_carrier_version"],
                    "output_carrier_version": invocation["output_carrier_version"],
                }
            )
            occurrence_kind = "primary_generator"
        else:
            occurrence_kind = str(event_kind)
        occurrences.append(
            {
                "edge_occurrence_id": invocation["primary_edge_occurrence_id"],
                "occurrence_kind": occurrence_kind,
                "invocation_id": invocation["invocation_id"],
                "atom_id": invocation["atom_id"],
                "stored_scalars": 0,
                "input_carrier_version": invocation["input_carrier_version"],
                "output_carrier_version": invocation["output_carrier_version"],
            }
        )
    return occurrences


def _family_catalog(
    invocations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    family_rows = (
        (
            "family.lexical.tied",
            "lexical",
            "tied_token_embedding_v262144_h640_ingress_egress",
            "atom_bank.lexical.tied.full_rank",
            1,
            PINNED_LEXICAL_SCALARS,
        ),
        (
            "family.mlp.gated_gelu_h640_i2048",
            "mlp",
            "token_local_gated_gelu_h640_i2048",
            "atom_bank.mlp.full_rank",
            18,
            PINNED_MLP_SCALARS,
        ),
        (
            "family.attention.sliding_causal_512",
            "attention",
            "sliding_causal_512_q4_kv1_d256_qk_norm_rope_local",
            "atom_bank.attention.sliding.full_rank",
            15,
            24_602_880,
        ),
        (
            "family.attention.global_causal",
            "attention",
            "global_causal_q4_kv1_d256_qk_norm_rope_global",
            "atom_bank.attention.global.full_rank",
            3,
            4_920_576,
        ),
    )
    catalog: list[dict[str, object]] = []
    for (
        family_id,
        component_kind,
        compatibility_key,
        atom_bank_id,
        atom_count,
        stored_scalars,
    ) in family_rows:
        member_ids = [
            row["invocation_id"]
            for row in invocations
            if row["family_id"] == family_id
        ]
        catalog.append(
            {
                "family_id": family_id,
                "component_kind": component_kind,
                "compatibility_key": compatibility_key,
                "member_invocation_ids": member_ids,
                "member_invocation_count": len(member_ids),
                "atom_bank_id": atom_bank_id,
                "full_rank_physical_atom_count": atom_count,
                "exact_stored_scalars": stored_scalars,
                "cross_block_membership": component_kind in {"mlp", "attention"},
                "tied_physical_storage_alias_required": component_kind == "lexical",
            }
        )
    return catalog


def _resource_point(
    point_id: str,
    candidate_stored_scalars: int,
    **fields: object,
) -> dict[str, object]:
    removed = PINNED_SOURCE_SCALARS - candidate_stored_scalars
    return {
        "point_id": point_id,
        **fields,
        "learned_correction_edge_count": 0,
        "candidate_stored_scalars": candidate_stored_scalars,
        "removed_scalars": removed,
        "whole_model_reduction_fraction": removed / PINNED_SOURCE_SCALARS,
        "meets_twenty_percent_resource_target": (
            candidate_stored_scalars
            <= PINNED_MAXIMUM_TARGET_CANDIDATE_SCALARS
        ),
    }


def _residual_adapter_scalars(
    component_kind: str,
    residual_rank: int | None,
) -> int:
    if component_kind == "mlp":
        if residual_rank is None:
            return _MLP_INVOCATION_SCALARS
        return _MLP_RESIDUAL_FIXED_SCALARS + (
            _MLP_RESIDUAL_RANK_SLOPE * residual_rank
        )
    if component_kind == "attention":
        if residual_rank is None:
            return _ATTENTION_INVOCATION_SCALARS
        return _ATTENTION_RESIDUAL_FIXED_SCALARS + (
            _ATTENTION_RESIDUAL_RANK_SLOPE * residual_rank
        )
    raise ValueError("residual adapters exist only for MLP or attention")


def _family_curve(
    curve_id: str,
    family_id: str,
    component_kind: str,
    member_count: int,
    eligible_member_count: int,
) -> list[dict[str, object]]:
    source_per_member = (
        _MLP_INVOCATION_SCALARS
        if component_kind == "mlp"
        else _ATTENTION_INVOCATION_SCALARS
    )
    points: list[dict[str, object]] = []
    for residual_rank in (None, *_RESIDUAL_RANKS):
        representation = (
            "exact_full_rank_dense_residual"
            if residual_rank is None
            else f"low_rank_residual_r{residual_rank}"
        )
        per_adapter = _residual_adapter_scalars(
            component_kind,
            residual_rank,
        )
        adapter_count = member_count - 1
        family_candidate = source_per_member + adapter_count * per_adapter
        candidate = (
            PINNED_SOURCE_SCALARS
            - member_count * source_per_member
            + family_candidate
        )
        points.append(
            _resource_point(
                f"{curve_id}.{representation}",
                candidate,
                primary_ladder=True,
                family_id=family_id,
                component_kind=component_kind,
                compressed_family_member_count=member_count,
                exact_unassigned_compatible_invocation_count=(
                    eligible_member_count - member_count
                ),
                residual_representation=representation,
                residual_rank=residual_rank,
                shared_core_count=1,
                shared_core_stored_scalars=source_per_member,
                shared_core_counted_once=True,
                residual_adapter_count=adapter_count,
                per_residual_adapter_stored_scalars=per_adapter,
                family_candidate_stored_scalars=family_candidate,
                exact_dense_residual_control=(residual_rank is None),
            )
        )
    return points


def _joint_full_family_curve() -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for residual_rank in (None, *_RESIDUAL_RANKS):
        representation = (
            "exact_full_rank_dense_residual"
            if residual_rank is None
            else f"low_rank_residual_r{residual_rank}"
        )
        mlp_adapter = _residual_adapter_scalars("mlp", residual_rank)
        attention_adapter = _residual_adapter_scalars(
            "attention",
            residual_rank,
        )
        family_breakdown = {
            "family.mlp.gated_gelu_h640_i2048": (
                _MLP_INVOCATION_SCALARS + 17 * mlp_adapter
            ),
            "family.attention.sliding_causal_512": (
                _ATTENTION_INVOCATION_SCALARS + 14 * attention_adapter
            ),
            "family.attention.global_causal": (
                _ATTENTION_INVOCATION_SCALARS + 2 * attention_adapter
            ),
        }
        candidate = (
            PINNED_LEXICAL_SCALARS
            + sum(family_breakdown.values())
            + PINNED_TERMINAL_NORM_SCALARS
        )
        points.append(
            _resource_point(
                f"shared_family.full_body.{representation}",
                candidate,
                primary_ladder=True,
                family_ids=list(family_breakdown),
                residual_representation=representation,
                residual_rank=residual_rank,
                shared_core_count=3,
                shared_core_stored_scalars=(
                    _MLP_INVOCATION_SCALARS
                    + 2 * _ATTENTION_INVOCATION_SCALARS
                ),
                shared_core_counted_once_per_family=True,
                residual_adapter_count=33,
                mlp_residual_adapter_count=17,
                compatible_attention_residual_adapter_count=16,
                per_MLP_residual_adapter_stored_scalars=mlp_adapter,
                per_attention_residual_adapter_stored_scalars=attention_adapter,
                body_family_stored_scalars=sum(family_breakdown.values()),
                family_stored_scalar_breakdown=family_breakdown,
                tied_lexical_remains_exact=True,
                terminal_remains_exact=True,
                exact_dense_residual_control=(residual_rank is None),
            )
        )
    return points


def _shared_family_ladder() -> dict[str, object]:
    mlp_pair = _family_curve(
        "shared_family.mlp_nonadjacent_pair",
        "family.mlp.gated_gelu_h640_i2048",
        "mlp",
        2,
        18,
    )
    sliding_pair = _family_curve(
        "shared_family.sliding_attention_compatible_pair",
        "family.attention.sliding_causal_512",
        "attention",
        2,
        15,
    )
    global_pair = _family_curve(
        "shared_family.global_attention_compatible_pair",
        "family.attention.global_causal",
        "attention",
        2,
        3,
    )
    growth_specs = (
        (
            "family.mlp.gated_gelu_h640_i2048",
            "mlp",
            18,
            (4, 8, 12, 18),
        ),
        (
            "family.attention.sliding_causal_512",
            "attention",
            15,
            (4, 8, 15),
        ),
        (
            "family.attention.global_causal",
            "attention",
            3,
            (3,),
        ),
    )
    growth_curves: list[dict[str, object]] = []
    for family_id, component_kind, eligible_count, sizes in growth_specs:
        growth_curves.append(
            {
                "family_id": family_id,
                "eligible_invocation_count": eligible_count,
                "pair_seed_member_count": 2,
                "ordered_growth_member_counts": list(sizes),
                "ordered_curves": [
                    {
                        "member_count": size,
                        "ordered_points": _family_curve(
                            f"shared_family.growth.{family_id}.n{size}",
                            family_id,
                            component_kind,
                            size,
                            eligible_count,
                        ),
                    }
                    for size in sizes
                ],
            }
        )
    return {
        "ladder_id": "primary.shared_family_core_residual_adapter_ladder",
        "primary_ladder": True,
        "phase_order": [
            "one_nonadjacent_MLP_pair",
            "compatible_attention_pairs_separately",
            "family_size_growth",
            "disjoint_full_body_family_joint",
        ],
        "residual_representation_order": [
            "exact_full_rank_dense_residual",
            "low_rank_residual_r256",
            "low_rank_residual_r128",
            "low_rank_residual_r64",
            "low_rank_residual_r32",
        ],
        "full_rank_control_is_dense_not_low_rank_r640": True,
        "shared_core_counted_once_and_each_nonanchor_has_one_adapter": True,
        "pair_selection_contract": {
            "selected_on": "A_fit",
            "MLP_pair_nonadjacent_required": True,
            "attention_pair_same_compatibility_key_required": True,
            "attention_sliding_and_global_pairs_are_separate_candidates": True,
            "activation_and_Fisher_criteria_only": True,
            "block_ordinal_or_distance_used_as_numerical_feature": False,
            "nonadjacency_is_an_eligibility_constraint_only": True,
            "members_and_anchor_frozen_before_A_select": True,
        },
        "one_nonadjacent_MLP_pair": {
            "family_id": "family.mlp.gated_gelu_h640_i2048",
            "member_count": 2,
            "ordered_points": mlp_pair,
        },
        "compatible_attention_pairs_separately": [
            {
                "family_id": "family.attention.sliding_causal_512",
                "member_count": 2,
                "ordered_points": sliding_pair,
            },
            {
                "family_id": "family.attention.global_causal",
                "member_count": 2,
                "ordered_points": global_pair,
            },
        ],
        "family_size_growth": {
            "each_candidate_has_disjoint_family_membership": True,
            "unassigned_compatible_invocations_remain_exact_singletons": True,
            "ordered_family_curves": growth_curves,
        },
        "disjoint_full_body_family_joint": {
            "family_member_counts": {
                "family.mlp.gated_gelu_h640_i2048": 18,
                "family.attention.sliding_causal_512": 15,
                "family.attention.global_causal": 3,
            },
            "duplicate_invocation_membership_allowed": False,
            "ordered_points": _joint_full_family_curve(),
        },
    }


def _heterogeneous_atom_bank_ladder() -> dict[str, object]:
    points: list[dict[str, object]] = []
    for rank in _HETEROGENEOUS_HIDDEN_RANKS:
        if rank == PINNED_HIDDEN_WIDTH:
            candidate = PINNED_SOURCE_SCALARS
            basis_scalars = 0
            lexical_codebook_scalars = PINNED_LEXICAL_SCALARS
            representation = "exact_identity_bank_implicit"
        else:
            basis_scalars = PINNED_HIDDEN_WIDTH * rank
            lexical_codebook_scalars = PINNED_VOCABULARY_SIZE * rank
            candidate = 419_456 * rank + 55_936
            representation = f"shared_hidden_atom_bank_r{rank}"
        points.append(
            _resource_point(
                f"heterogeneous_atom_bank.rank_{rank}",
                candidate,
                primary_ladder=True,
                hidden_rank=rank,
                representation=representation,
                exact_rank_control=(rank == PINNED_HIDDEN_WIDTH),
                shared_hidden_basis_stored_scalars=basis_scalars,
                shared_hidden_basis_counted_once=(rank < PINNED_HIDDEN_WIDTH),
                shared_hidden_basis_transpose_is_nonowning_view=True,
                tied_lexical_codebook_stored_scalars=lexical_codebook_scalars,
                tied_lexical_codebook_counted_once=True,
                MLP_invocation_coordinate_scalars=(
                    _MLP_INVOCATION_SCALARS
                    if rank == PINNED_HIDDEN_WIDTH
                    else 6_144 * rank + 1_280
                ),
                attention_invocation_coordinate_scalars=(
                    _ATTENTION_INVOCATION_SCALARS
                    if rank == PINNED_HIDDEN_WIDTH
                    else 2_560 * rank + 1_792
                ),
                MLP_invocation_count=18,
                attention_invocation_count=18,
                terminal_exact_stored_scalars=PINNED_TERMINAL_NORM_SCALARS,
            )
        )
    return {
        "ladder_id": "primary.heterogeneous_shared_hidden_atom_bank_ladder",
        "primary_ladder": True,
        "predecessor_ladder_id": (
            "primary.shared_family_core_residual_adapter_ladder"
        ),
        "typed_subbank_ids": [
            "atom_bank.lexical.tied.full_rank",
            "atom_bank.mlp.full_rank",
            "atom_bank.attention.sliding.full_rank",
            "atom_bank.attention.global.full_rank",
        ],
        "typed_subbank_count": 4,
        "rank_order": list(_HETEROGENEOUS_HIDDEN_RANKS),
        "rank_640_is_exact_identity_control_with_no_owned_basis_storage": True,
        "rank_below_640_formula": (
            "640*r_shared_basis_once + 262144*r_tied_lexical_once + "
            "18*(6144*r+1280)_MLP + 18*(2560*r+1792)_attention + "
            "640_terminal = 419456*r+55936"
        ),
        "terminal_norm_is_typed_exact_and_outside_shareable_family_bank": True,
        "ordered_points": points,
        "rank_512_misses_twenty_percent_target_by_scalars": 338_868,
        "largest_integer_rank_meeting_target_under_formula": 511,
        "rank_511_is_not_a_preregistered_execution_point": True,
        "first_preregistered_point_meeting_target": 496,
    }


def _matched_controls() -> list[dict[str, object]]:
    specifications = (
        (
            "control.independent.lexical_r512_exact_body",
            234_871_424,
            512,
            "exact_640_without_explicit_basis",
            "exact_1024_without_explicit_basis",
            "exact_checkpoint_operator",
        ),
        (
            "control.independent.reduced_body_exact_lexical",
            229_751_120,
            "exact_tied_640",
            256,
            64,
            4,
        ),
        (
            "control.independent.lexical_r512_mlp_g4",
            214_071_632,
            512,
            "exact_640_without_explicit_basis",
            "exact_1024_without_explicit_basis",
            4,
        ),
        (
            "control.independent.joint_lexical_r576_attention_256_64_mlp_g4",
            213_342_544,
            576,
            256,
            64,
            4,
        ),
        (
            "control.independent.joint_lexical_r544_attention_256_64_mlp_g4",
            204_933_456,
            544,
            256,
            64,
            4,
        ),
        (
            "control.independent.joint_lexical_r512_attention_256_64_mlp_g4",
            196_524_368,
            512,
            256,
            64,
            4,
        ),
    )
    return [
        _resource_point(
            point_id,
            candidate,
            primary_ladder=False,
            control_only=True,
            matched_control_kind="independent_factorization_negative_prior",
            lexical_rank=lexical_rank,
            attention_input_rank=attention_input_rank,
            attention_output_rank=attention_output_rank,
            MLP_generator_rank=mlp_rank,
            recommended_primary=False,
        )
        for (
            point_id,
            candidate,
            lexical_rank,
            attention_input_rank,
            attention_output_rank,
            mlp_rank,
        ) in specifications
    ]


def _exact_arms() -> list[dict[str, object]]:
    return [
        {
            "arm_id": "exact.causal_carrier",
            "purpose": "authenticated_exact_39_invocation_75_occurrence_control",
            "predecessor_arm_id": None,
            "family_lift_scope": "none_occurrence_local_exact_generators",
            "candidate_stored_scalars": PINNED_SOURCE_SCALARS,
        },
        {
            "arm_id": "exact.cross_block_mlp_family_lift",
            "purpose": "lift_18_exact_MLP_invocations_into_one_family_contract",
            "predecessor_arm_id": "exact.causal_carrier",
            "family_lift_scope": "all_18_MLP_invocations",
            "candidate_stored_scalars": PINNED_SOURCE_SCALARS,
        },
        {
            "arm_id": "exact.compatible_attention_family_lift",
            "purpose": (
                "lift_attention_into_separate_sliding_and_global_compatible_"
                "families"
            ),
            "predecessor_arm_id": "exact.cross_block_mlp_family_lift",
            "family_lift_scope": "15_sliding_and_3_global_attention_invocations",
            "candidate_stored_scalars": PINNED_SOURCE_SCALARS,
        },
        {
            "arm_id": "exact.heterogeneous_full_rank_atom_bank_lift",
            "purpose": (
                "bind_two_tied_lexical_endpoints_and_36_body_invocations_to_"
                "four_typed_full_rank_subbanks_while_terminal_stays_exact"
            ),
            "predecessor_arm_id": "exact.compatible_attention_family_lift",
            "family_lift_scope": (
                "tied_lexical_MLP_sliding_attention_global_attention"
            ),
            "physical_family_atom_count": PINNED_HETEROGENEOUS_ATOM_COUNT,
            "terminal_exact_atom_count": 1,
            "candidate_stored_scalars": PINNED_SOURCE_SCALARS,
        },
    ]


def _build_without_hash() -> dict[str, object]:
    invocations = _full_invocations()
    edge_occurrences = _edge_occurrences(invocations)
    families = _family_catalog(invocations)
    exact_arms = _exact_arms()
    shared_family_ladder = _shared_family_ladder()
    heterogeneous_ladder = _heterogeneous_atom_bank_ladder()
    matched_controls = _matched_controls()
    compression_arms = [
        {
            "arm_id": "primary.shared_family_core_residual_adapter_ladder",
            "purpose": (
                "share_one_core_per_compatible_family_then_reduce_only_"
                "invocation_residual_adapters"
            ),
            "predecessor_arm_id": (
                "exact.heterogeneous_full_rank_atom_bank_lift"
            ),
            "primary_ladder": True,
        },
        {
            "arm_id": "primary.heterogeneous_shared_hidden_atom_bank_ladder",
            "purpose": (
                "share_one_hidden_basis_across_tied_lexical_and_typed_body_"
                "subbanks"
            ),
            "predecessor_arm_id": (
                "primary.shared_family_core_residual_adapter_ladder"
            ),
            "primary_ladder": True,
        },
        {
            "arm_id": "control.matched_independent_factorizations",
            "purpose": "matched_naive_deletion_SVD_and_joint_negative_controls",
            "predecessor_arm_id": None,
            "primary_ladder": False,
            "control_only": True,
        },
    ]
    arms = exact_arms + compression_arms
    family_stored_scalars = sum(
        int(family["exact_stored_scalars"]) for family in families
    )
    return {
        "schema": GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA,
        "format_version": (
            GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION
        ),
        "artifact_role": (
            "setup_only_preregistered_exact_lifts_then_compression_ladders"
        ),
        "setup_receipt": {
            "model_weights_loaded": False,
            "prompt_manifests_opened": False,
            "prompt_text_opened": False,
            "tokenization_performed": False,
            "family_assignment_performed": False,
            "coefficient_fitting_performed": False,
            "compiler_executed": False,
            "model_forward_executed": False,
            "new_experiment_metrics_present": False,
            "candidate_artifact_written": False,
        },
        "model_contract": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "hidden_width": PINNED_HIDDEN_WIDTH,
            "mlp_width": PINNED_MLP_WIDTH,
            "vocabulary_size": PINNED_VOCABULARY_SIZE,
            "layer_count": PINNED_LAYER_COUNT,
            "event_count": PINNED_EVENT_COUNT,
            "mutation_count": PINNED_MUTATION_COUNT,
            "edge_occurrence_count": PINNED_EDGE_OCCURRENCE_COUNT,
            "global_attention_blocks": list(PINNED_GLOBAL_ATTENTION_BLOCKS),
            "tied_lexical_ingress_egress_required": True,
            "terminal_RMSNorm_and_logit_softcap_exact": True,
        },
        "ontology_contract": {
            "invocation_definition": (
                "one_ordered_temporal_use_of_one_typed_operator"
            ),
            "family_definition": (
                "reusable_shape_and_semantics_compatible_generator_contract"
            ),
            "atom_bank_definition": (
                "physical_typed_coefficient_inventory_owned_by_one_family"
            ),
            "edge_occurrence_definition": (
                "one_logical_edge_application_at_one_carrier_transition"
            ),
            "invocation_is_not_family": True,
            "family_is_not_atom_bank": True,
            "atom_bank_is_not_edge_occurrence": True,
            "repeated_family_use_does_not_collapse_invocation_order": True,
            "edge_occurrence_count_does_not_imply_physical_coefficient_count": (
                True
            ),
            "family_schema_sharing_receives_no_parameter_credit": True,
            "coefficient_sharing_credit_requires_authenticated_storage_alias": (
                True
            ),
            "provenance_labels_are_not_numerical_features": True,
        },
        "execution_graph": {
            "carrier_initial_version": 0,
            "carrier_final_version": PINNED_MUTATION_COUNT,
            "invocation_count": len(invocations),
            "body_invocation_count": PINNED_MUTATION_COUNT,
            "lexical_invocation_count": 2,
            "terminal_invocation_count": 1,
            "edge_occurrence_count": len(edge_occurrences),
            "lexical_edge_occurrence_count": 2,
            "identity_carry_occurrence_count": PINNED_MUTATION_COUNT,
            "primary_generator_edge_occurrence_count": PINNED_MUTATION_COUNT,
            "terminal_edge_occurrence_count": 1,
            "learned_correction_edge_count": 0,
            "optional_learned_correction_edges_allowed": False,
            "invocations": invocations,
            "ordered_invocation_ids": [
                invocation["invocation_id"] for invocation in invocations
            ],
            "edge_occurrences": edge_occurrences,
            "ordered_edge_occurrence_ids": [
                occurrence["edge_occurrence_id"]
                for occurrence in edge_occurrences
            ],
            "every_invocation_has_one_unique_adapter": True,
            "every_invocation_selects_exactly_one_atom": True,
            "lexical_ingress_and_egress_select_same_physical_atom": True,
            "every_mutation_aggregates_one_identity_carry_and_one_generator": (
                True
            ),
            "identity_carries_store_zero_scalars": True,
            "future_token_cross_example_reverse_and_cyclic_edges_forbidden": (
                True
            ),
        },
        "family_and_atom_bank_contract": {
            "family_count": len(families),
            "families": families,
            "family_assignment_contract": {
                "architectural_compatibility_catalog_frozen_before_model_load": (
                    True
                ),
                "compressed_members_and_shared_core_anchor_selected_on": "A_fit",
                "selection_uses_activation_and_Fisher_statistics_only": True,
                "block_ordinal_used_as_numerical_feature": False,
                "layer_distance_used_as_numerical_feature": False,
                "nonadjacency_may_be_used_only_as_boolean_eligibility": True,
                "family_assignment_frozen_before_A_select": True,
                "family_assignment_manifest_SHA256_required": True,
                "duplicate_invocation_membership_allowed": False,
                "unassigned_invocations_remain_exact_singletons": True,
            },
            "heterogeneous_atom_bank": {
                "atom_bank_id": "atom_bank.heterogeneous.full_rank",
                "typed_subbank_ids": [
                    "atom_bank.lexical.tied.full_rank",
                    "atom_bank.mlp.full_rank",
                    "atom_bank.attention.sliding.full_rank",
                    "atom_bank.attention.global.full_rank",
                ],
                "typed_subbank_count": 4,
                "physical_atom_count": PINNED_HETEROGENEOUS_ATOM_COUNT,
                "stored_scalars": family_stored_scalars,
                "lexical_endpoint_invocation_count": 2,
                "lexical_physical_atom_count": 1,
                "body_physical_atom_count": PINNED_BODY_ATOM_COUNT,
                "terminal_exact_atom_is_outside_bank": True,
                "incompatible_atom_shapes_are_never_flattened_or_averaged": (
                    True
                ),
                "atom_to_source_tensor_bijection_required_for_exact_lift": (
                    True
                ),
            },
            "MLP_family_may_span_all_blocks": True,
            "sliding_and_global_attention_share_a_family": False,
            "attention_compatibility_includes_mask_RoPE_heads_and_QK_norm": True,
            "exact_full_rank_body_atoms_remain_invocation_local_coefficients": (
                True
            ),
            "tied_lexical_storage_counted_once": True,
            "exact_lifts_claim_no_parameter_reduction": True,
        },
        "arms": arms,
        "arm_order": [arm["arm_id"] for arm in arms],
        "exact_lift_gate_chain": {
            "exact_arm_order": [arm["arm_id"] for arm in exact_arms],
            "each_arm_must_pass_before_successor_executes": True,
            "all_exact_arms_must_pass_before_any_compression_fit": True,
            "native_reference_required": True,
            "compared_rows": "valid_query_rows_only",
            "maximum_absolute_logit_error": 0.0,
            "absolute_delta_nll_per_token": 0.0,
            "native_to_candidate_KL_per_query": 0.0,
            "top1_agreement": 1.0,
            "ordered_invocation_input_hashes_exact": True,
            "ordered_invocation_delta_hashes_exact": True,
            "all_75_ordered_edge_occurrence_hashes_exact": True,
            "final_carrier_hash_exact": True,
            "terminal_normalized_state_hash_exact": True,
            "pre_and_post_softcap_logit_hashes_exact": True,
            "tied_lexical_ingress_egress_storage_alias_exact": True,
            "atom_bank_source_tensor_bijection_exact": True,
            "source_module_and_native_fallback_calls_for_lifted_arms": 0,
            "source_and_candidate_fingerprints_unchanged": True,
            "exact_candidate_stored_scalars": PINNED_SOURCE_SCALARS,
            "exact_invocation_adapter_stored_scalars": 0,
            "exact_learned_correction_edge_count": 0,
        },
        "compression_ladders": {
            "fit_one_artifact_per_ordered_point_not_one_supernet": True,
            "primary_ladder_order": [
                "primary.shared_family_core_residual_adapter_ladder",
                "primary.heterogeneous_shared_hidden_atom_bank_ladder",
            ],
            "shared_family_core_residual_adapters": shared_family_ladder,
            "heterogeneous_shared_hidden_atom_bank": heterogeneous_ladder,
            "matched_controls": {
                "arm_id": "control.matched_independent_factorizations",
                "control_only": True,
                "points_may_not_be_promoted_to_primary_without_new_protocol": (
                    True
                ),
                "ordered_points": matched_controls,
            },
            "excluded_historical_negative_prior": {
                "configuration_id": (
                    "excluded.joint_lexical_r512_attention_256_64_mlp_g4_"
                    "with_136_learned_transports"
                ),
                "candidate_stored_scalars": 196_559_184,
                "learned_correction_edge_count": 136,
                "historical_outcome_label": "known_behavioral_collapse",
                "historical_metric_values_imported_into_protocol": False,
                "recommended_primary": False,
                "execution_allowed_under_this_protocol": False,
                "reason": "learned_correction_edges_are_forbidden",
            },
            "learned_correction_edge_count_ladder": [0],
            "resource_point_order_is_not_a_behavioral_ranking": True,
            "existing_or_prior_results_may_not_remove_points_or_change_order": (
                True
            ),
        },
        "fit_selection_guard_assessment_firewall": {
            "split_unit": "prompt_or_paraphrase_family_is_atomic",
            "required_manifests": [
                "A_fit",
                "A_select",
                "B_guard",
                "C_assessment",
            ],
            "split_manifest_SHA256_values_required_before_model_load": True,
            "prompt_example_and_family_overlap_across_roles_allowed": False,
            "A_fit": (
                "select_and_fit_family_members_shared_core_anchors_residual_"
                "adapters_and_frozen_atom_bank_coefficients"
            ),
            "A_fit_uses_actual_frozen_candidate_prefix_trajectory": True,
            "A_fit_uses_activation_and_Fisher_statistics_for_assignment": True,
            "A_fit_may_use_block_ordinal_or_layer_distance_as_feature": False,
            "A_fit_may_open_A_select_B_guard_or_C_assessment": False,
            "family_assignment_manifest_frozen_and_hashed_before_A_select": (
                True
            ),
            "A_select": "select_only_among_fitted_frozen_artifacts",
            "A_select_may_change_family_membership_or_shared_core_anchor": False,
            "A_select_may_refit_change_topology_or_change_resource_points": (
                False
            ),
            "pareto_frontier_frozen_before_B_guard": True,
            "B_guard_opened_once": True,
            "B_guard_scores_entire_frozen_frontier_once": True,
            "B_guard_may_refit_reorder_prune_or_change_thresholds": False,
            "B_guard_may_change_family_assignment": False,
            "B_guard_multiplicity_control": (
                "simultaneous_family_clustered_bootstrap_intervals"
            ),
            "B_guard_selection_rule": (
                "highest_compression_point_passing_every_hard_behavior_"
                "causality_and_resource_gate"
            ),
            "C_assessment_opened_only_after_guard_pass": True,
            "C_assessment_candidate_count": 1,
            "C_assessment_opened_once": True,
            "C_assessment_may_refit_select_reorder_or_change_thresholds": False,
            "C_assessment_may_change_family_assignment": False,
            "failed_guard_or_assessment_requires_new_protocol_and_fresh_splits": (
                True
            ),
            "raw_prompt_text_or_token_ids_written_to_report": False,
            "row_identity_hash_required_at_every_capture_site": True,
        },
        "resource_accounting": {
            "source_unique_stored_scalars": PINNED_SOURCE_SCALARS,
            "source_pools": {
                "tied_lexical_counted_once": PINNED_LEXICAL_SCALARS,
                "attention_all_18": PINNED_ATTENTION_SCALARS,
                "MLP_all_18": PINNED_MLP_SCALARS,
                "terminal_RMSNorm": PINNED_TERMINAL_NORM_SCALARS,
            },
            "source_inventory_closes": (
                PINNED_LEXICAL_SCALARS
                + PINNED_ATTENTION_SCALARS
                + PINNED_MLP_SCALARS
                + PINNED_TERMINAL_NORM_SCALARS
                == PINNED_SOURCE_SCALARS
            ),
            "exact_family_bank_plus_terminal_closes": (
                family_stored_scalars + PINNED_TERMINAL_NORM_SCALARS
                == PINNED_SOURCE_SCALARS
            ),
            "twenty_percent_reduction_target": (
                PINNED_WHOLE_MODEL_REDUCTION_TARGET
            ),
            "maximum_candidate_scalars_to_meet_target": (
                PINNED_MAXIMUM_TARGET_CANDIDATE_SCALARS
            ),
            "minimum_removed_scalars_to_meet_target": (
                PINNED_MINIMUM_TARGET_REMOVED_SCALARS
            ),
            "shared_family_formulas": {
                "MLP_source_scalars_per_invocation": _MLP_INVOCATION_SCALARS,
                "MLP_low_rank_residual_adapter": "1280+8064*r",
                "attention_source_scalars_per_invocation": (
                    _ATTENTION_INVOCATION_SCALARS
                ),
                "attention_low_rank_residual_adapter": "1792+3840*r",
                "family_total": (
                    "one_shared_core_once + (member_count-1)*residual_adapter"
                ),
                "whole_model_candidate": (
                    "source - member_count*source_per_invocation + family_total"
                ),
                "exact_full_rank_control_uses_dense_source_sized_residual": True,
            },
            "heterogeneous_rank_below_640_formula": "419456*r+55936",
            "heterogeneous_rank_512_candidate_scalars": 214_817_408,
            "heterogeneous_rank_512_target_shortfall_scalars": 338_868,
            "largest_heterogeneous_integer_rank_meeting_target": 511,
            "heterogeneous_rank_496_candidate_scalars": 208_106_112,
            "heterogeneous_rank_496_meets_target": True,
            "invocation_adapter_parameters_and_persistent_buffers_count": True,
            "atom_bank_parameters_and_persistent_buffers_count_once_per_"
            "authenticated_physical_storage": True,
            "shared_basis_transpose_view_receives_no_second_storage_charge": (
                True
            ),
            "edge_occurrence_metadata_is_not_a_free_coefficient_alias": True,
            "chart_and_transport_overhead_policy": {
                "primary_chart_count": 0,
                "primary_learned_transport_count": 0,
                "charts_centers_bases_scales_encoders_and_decoders_if_added_"
                "must_count": True,
                "learned_transport_parameters_buffers_and_occurrences_if_added_"
                "must_count": True,
                "adding_either_requires_new_protocol": True,
            },
            "learned_correction_edge_scalars": 0,
            "optional_learned_correction_edge_budget_present": False,
            "logical_MACs_serialized_bytes_peak_dynamic_bytes_and_latency_"
            "reported_separately": True,
            "source_plus_candidate_development_residency_reported": True,
            "parameter_reduction_does_not_authorize_behavioral_compression": (
                True
            ),
        },
        "claim_firewall": {
            "setup_authenticates_only_a_frozen_protocol": True,
            "exact_lifts_establish_execution_ontology_not_compression": True,
            "resource_target_pass_is_not_a_fidelity_pass": True,
            "source_free_serialized_candidate_required_for_deployment_claim": (
                True
            ),
            "parameter_byte_MAC_memory_and_latency_claims_are_independent": (
                True
            ),
            "cached_decode_long_context_and_padding_require_separate_gates": (
                True
            ),
            "learned_optional_correction_edges_require_a_new_protocol": True,
            "no_compression_or_speed_claim_authorized_by_this_setup": True,
        },
    }


def build_gemma3_heterogeneous_generator_family_protocol() -> dict[str, object]:
    protocol = _build_without_hash()
    protocol["protocol_sha256"] = _protocol_sha256(protocol)
    return protocol


def validate_gemma3_heterogeneous_generator_family_protocol(
    value: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("heterogeneous generator-family protocol must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    if materialized != build_gemma3_heterogeneous_generator_family_protocol():
        raise ValueError("heterogeneous generator-family protocol is not canonical")


def load_gemma3_heterogeneous_generator_family_protocol(
    source: Path | str,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    validate_gemma3_heterogeneous_generator_family_protocol(value)
    return value


def write_gemma3_heterogeneous_generator_family_protocol(
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    target = Path(destination)
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite a heterogeneous generator-family protocol"
        ) from error
    return protocol


@dataclass(frozen=True, slots=True)
class Gemma3HeterogeneousGeneratorFamilyProtocol:
    """Typed wrapper around the immutable setup artifact."""

    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_gemma3_heterogeneous_generator_family_protocol(self.state)

    @classmethod
    def build(cls) -> "Gemma3HeterogeneousGeneratorFamilyProtocol":
        return cls(build_gemma3_heterogeneous_generator_family_protocol())

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(self.state, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the existing canonical protocol without writing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        load_gemma3_heterogeneous_generator_family_protocol(args.output)
    else:
        write_gemma3_heterogeneous_generator_family_protocol(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
