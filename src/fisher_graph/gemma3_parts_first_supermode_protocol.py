"""Prospective protocol for parts-first ordered Gemma 3 supermodes.

The predecessor whole-model lift treated each attention or MLP layer operator
as one atom.  This protocol refines those operators into tied lexical hidden
columns, query heads, and MLP intermediate channels.  A layer is an execution
container; it is never itself a generator identity.

The module is intentionally model-free and writes no artifact.  It freezes the
ontology, checkpoint accounting, event order, and role firewall needed before
any data-dependent grouping or intervention is allowed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Mapping


SCHEMA = "fisher_graph.gemma3_parts_first_supermode_protocol"
FORMAT_VERSION = 1
SEMANTIC_DOMAIN = b"fisher_graph.gemma3_parts_first_supermode_protocol.v1\0"

MODEL_ID = "google/gemma-3-270m"
SOURCE_MODEL_FINGERPRINT_SHA256 = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
PREDECESSOR_PROTOCOL_PATH = (
    "artifacts/research/"
    "gemma3-heterogeneous-whole-model-generator-family-protocol-v1.json"
)
PREDECESSOR_PROTOCOL_SHA256 = (
    "32747a59ce2aa7ad0633904b54680d19b7f2da1545bb136c4816d025667f61f2"
)
PREDECESSOR_PROTOCOL_FILE_SHA256 = (
    "b2caaae36742864f7a0bb491ed80ab08f8c94029d6d554d7d68cdbd1f7b1d243"
)
PREDECESSOR_REPORT_PATH = (
    ".local-runs/google--gemma-3-270m/"
    "heterogeneous-generator-family-exact-v1.json"
)
PREDECESSOR_REPORT_SHA256 = (
    "7f062b40a2af2fe61364b208189773b82569bbe88902c0b8f3e7bbe607401926"
)
PREDECESSOR_REPORT_FILE_SHA256 = (
    "2ad135209ff1c780944c7d0efd5dbd7b52724266dc880345abb31cfc9bbcf64a"
)

LAYER_COUNT = 18
HIDDEN_WIDTH = 640
VOCABULARY_SIZE = 262_144
QUERY_HEAD_COUNT = 4
KEY_VALUE_HEAD_COUNT = 1
HEAD_DIMENSION = 256
MLP_WIDTH = 2_048

LEXICAL_ATOM_COUNT = HIDDEN_WIDTH
ATTENTION_ATOM_COUNT = LAYER_COUNT * QUERY_HEAD_COUNT
MLP_ATOM_COUNT = LAYER_COUNT * MLP_WIDTH
PHYSICAL_ATOM_COUNT = (
    LEXICAL_ATOM_COUNT + ATTENTION_ATOM_COUNT + MLP_ATOM_COUNT
)

LEXICAL_ATOM_SCALARS = VOCABULARY_SIZE
ATTENTION_ATOM_SCALARS = 2 * HIDDEN_WIDTH * HEAD_DIMENSION
MLP_ATOM_SCALARS = 3 * HIDDEN_WIDTH
ATOM_OWNED_SCALARS = (
    LEXICAL_ATOM_COUNT * LEXICAL_ATOM_SCALARS
    + ATTENTION_ATOM_COUNT * ATTENTION_ATOM_SCALARS
    + MLP_ATOM_COUNT * MLP_ATOM_SCALARS
)
ATTENTION_SHARED_SCALARS_PER_LAYER = (
    2 * HEAD_DIMENSION * HIDDEN_WIDTH  # shared K and V projections
    + 2 * HEAD_DIMENSION  # shared q_norm and k_norm
    + 2 * HIDDEN_WIDTH  # input and post-attention RMSNorms
)
MLP_SHARED_SCALARS_PER_LAYER = 2 * HIDDEN_WIDTH
TERMINAL_SHARED_SCALARS = HIDDEN_WIDTH
SHARED_SUBSTRATE_SCALARS = (
    LAYER_COUNT * ATTENTION_SHARED_SCALARS_PER_LAYER
    + LAYER_COUNT * MLP_SHARED_SCALARS_PER_LAYER
    + TERMINAL_SHARED_SCALARS
)
SOURCE_SCALARS = ATOM_OWNED_SCALARS + SHARED_SUBSTRATE_SCALARS


def _event_schedule() -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "event_index": 0,
            "event_id": "lexical.ingress",
            "physical_atom_family_id": "tied_lexical_hidden_column",
            "functional_role_id": "lexical_ingress",
            "layer_ordinal": None,
            "atom_emitting": True,
            "additive_boundary": "embedding_output_h640",
            "post_boundary": "layer_0_attention_input_rmsnorm",
        }
    ]
    for layer in range(LAYER_COUNT):
        events.extend(
            (
                {
                    "event_index": 1 + 2 * layer,
                    "event_id": f"layer.{layer}.attention",
                    "physical_atom_family_id": "attention_qo_head",
                    "functional_role_id": "attention_qo_contribution",
                    "layer_ordinal": layer,
                    "atom_emitting": True,
                    "additive_boundary": "attention_operator_output_h640",
                    "post_boundary": "post_attention_rmsnorm_and_residual",
                },
                {
                    "event_index": 2 + 2 * layer,
                    "event_id": f"layer.{layer}.mlp",
                    "physical_atom_family_id": "mlp_channel",
                    "functional_role_id": "mlp_channel_contribution",
                    "layer_ordinal": layer,
                    "atom_emitting": True,
                    "additive_boundary": "mlp_operator_output_h640",
                    "post_boundary": "post_feedforward_rmsnorm_and_residual",
                },
            )
        )
    events.extend(
        (
            {
                "event_index": 37,
                "event_id": "terminal.rmsnorm",
                "physical_atom_family_id": None,
                "functional_role_id": "shared_substrate_only",
                "layer_ordinal": None,
                "atom_emitting": False,
                "additive_boundary": None,
                "post_boundary": "normalized_terminal_hidden_h640",
            },
            {
                "event_index": 38,
                "event_id": "lexical.raw_egress",
                "physical_atom_family_id": "tied_lexical_hidden_column",
                "functional_role_id": "lexical_readout",
                "layer_ordinal": None,
                "atom_emitting": True,
                "additive_boundary": "raw_logits_v262144",
                "post_boundary": "final_logit_softcap",
            },
        )
    )
    return events


def _protocol_without_sha256() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "status": "prospective_not_run",
        "objective": (
            "discover chart-conditioned ordered supermodes made from grouped "
            "embedding columns, attention heads, and MLP channels"
        ),
        "predecessor": {
            "protocol_path": PREDECESSOR_PROTOCOL_PATH,
            "protocol_sha256": PREDECESSOR_PROTOCOL_SHA256,
            "protocol_file_sha256": PREDECESSOR_PROTOCOL_FILE_SHA256,
            "report_path": PREDECESSOR_REPORT_PATH,
            "report_sha256": PREDECESSOR_REPORT_SHA256,
            "report_file_sha256": PREDECESSOR_REPORT_FILE_SHA256,
            "accepted_fact": (
                "the exact 39-event heterogeneous full-rank lift closes"
            ),
            "rejected_inference": (
                "a whole attention or MLP layer is a primitive generator"
            ),
        },
        "model_inventory": {
            "model_id": MODEL_ID,
            "source_model_fingerprint_sha256": (
                SOURCE_MODEL_FINGERPRINT_SHA256
            ),
            "layer_count": LAYER_COUNT,
            "hidden_width": HIDDEN_WIDTH,
            "vocabulary_size": VOCABULARY_SIZE,
            "query_head_count_per_layer": QUERY_HEAD_COUNT,
            "key_value_head_count_per_layer": KEY_VALUE_HEAD_COUNT,
            "head_dimension": HEAD_DIMENSION,
            "mlp_intermediate_width": MLP_WIDTH,
            "attention_projection_bias_present": False,
            "mlp_projection_bias_present": False,
            "embedding_and_lm_head_storage_tied": True,
        },
        "physical_atom_universe": {
            "universe_id": "gemma3_parts_first_physical_atom_u37576.v1",
            "physical_atom_count": PHYSICAL_ATOM_COUNT,
            "families": [
                {
                    "family_id": "tied_lexical_hidden_column",
                    "ordinal_start": 0,
                    "ordinal_end_exclusive": 640,
                    "atom_count": LEXICAL_ATOM_COUNT,
                    "atom_id_formula": "g",
                    "owned_slice": "E[:,g]",
                    "stored_scalars_per_atom": LEXICAL_ATOM_SCALARS,
                    "role_invocations": [
                        "lexical.ingress",
                        "lexical.raw_egress",
                    ],
                    "storage_counted_once_across_roles": True,
                },
                {
                    "family_id": "attention_qo_head",
                    "functional_role_id": "attention_qo_contribution",
                    "ordinal_start": 640,
                    "ordinal_end_exclusive": 712,
                    "atom_count": ATTENTION_ATOM_COUNT,
                    "atom_id_formula": "640 + 4*layer + query_head",
                    "owned_slice": "Q[head_rows,:] plus O[:,head_columns]",
                    "stored_scalars_per_atom": ATTENTION_ATOM_SCALARS,
                    "shared_not_owned": [
                        "K_projection",
                        "V_projection",
                        "q_norm",
                        "k_norm",
                        "RoPE",
                        "causal_or_sliding_mask",
                        "GQA_value_repetition",
                        "float32_softmax_router",
                        "input_and_output_RMSNorm",
                    ],
                },
                {
                    "family_id": "mlp_channel",
                    "functional_role_id": "mlp_channel_contribution",
                    "ordinal_start": 712,
                    "ordinal_end_exclusive": 37_576,
                    "atom_count": MLP_ATOM_COUNT,
                    "atom_id_formula": "712 + 2048*layer + channel",
                    "owned_slice": (
                        "gate[channel,:] plus up[channel,:] plus "
                        "down[:,channel]"
                    ),
                    "stored_scalars_per_atom": MLP_ATOM_SCALARS,
                    "shared_not_owned": [
                        "input_RMSNorm",
                        "output_RMSNorm",
                    ],
                },
            ],
            "vocabulary_rows_are_semantic_views_only": True,
            "vocabulary_row_views_receive_storage_credit": False,
            "vocabulary_row_views_may_be_supermode_members": False,
        },
        "scalar_accounting": {
            "lexical_atom_owned_scalars": (
                LEXICAL_ATOM_COUNT * LEXICAL_ATOM_SCALARS
            ),
            "attention_atom_owned_scalars": (
                ATTENTION_ATOM_COUNT * ATTENTION_ATOM_SCALARS
            ),
            "mlp_atom_owned_scalars": MLP_ATOM_COUNT * MLP_ATOM_SCALARS,
            "atom_owned_scalars": ATOM_OWNED_SCALARS,
            "attention_shared_scalars_per_layer": (
                ATTENTION_SHARED_SCALARS_PER_LAYER
            ),
            "mlp_shared_scalars_per_layer": MLP_SHARED_SCALARS_PER_LAYER,
            "terminal_shared_scalars": TERMINAL_SHARED_SCALARS,
            "shared_substrate_scalars": SHARED_SUBSTRATE_SCALARS,
            "source_unique_stored_scalars": SOURCE_SCALARS,
            "disjoint_and_exhaustive_closure_required": True,
        },
        "group_ontology": {
            "physical_ownership": (
                "the fixed disjoint atom and shared-substrate ledger"
            ),
            "exact_closure_partition": (
                "one disjoint exhaustive grouping used to prove an operator sum"
            ),
            "logical_coalition_group": (
                "a possibly overlapping discovery hypothesis over atom IDs"
            ),
            "functional_occurrence": (
                "one physical atom or group invoked at one event and role"
            ),
            "overlapping_groups_receive_storage_or_reuse_credit": False,
            "selected_same_boundary_groups_are_canonicalized_to_one_union": True,
            "singleton_parts_use_singleton_groups": True,
            "raw_atom_references_in_supermode_stages_allowed": False,
            "membership_encoding": [
                "sorted_unique_uint32_le.v1",
            ],
            "membership_sidecars_must_bind_physical_universe": True,
        },
        "exact_operator_equations": {
            "lexical_ingress": (
                "h0(t)=sum_g embed_scale*E[t,g]*basis_vector(g)"
            ),
            "attention": (
                "operator_output=sum_query_heads routed_shared_V_head_O_slice"
            ),
            "mlp": (
                "operator_output=sum_j "
                "activation(gate_j*x)*(up_j*x)*down_column_j"
            ),
            "lexical_raw_egress": (
                "raw_logits(h)=sum_g h[g]*E[:,g]"
            ),
            "additivity_stops_before_nonlinear_post_boundary": True,
            "cross_event_outputs_may_be_raw_summed": False,
        },
        "event_schedule": _event_schedule(),
        "supermode_semantics": {
            "stage": (
                "one chart-conditioned logical group invocation at one event"
            ),
            "ordered_supermode": (
                "a strictly event-ordered tuple of two or more stages"
            ),
            "layer_is_generator_identity": False,
            "stage_members_are_homogeneous_at_one_additive_boundary": True,
            "cross_layer_claim_requires_two_distinct_layer_provenances": True,
            "finite_gate_equation": "o_e(alpha)=o_e+(alpha-1)*group_emission_e",
            "native_post_boundary_and_later_events_rerun_after_each_gate": True,
            "attention_chart_state_binds_sequence_KV_mask_position_and_window": (
                True
            ),
            "primary_intervention_freezes_baseline_chart_activation_mask": True,
            "online_router_rerun_is_reported_separately": True,
            "common_response_coordinate": (
                "dScore/dalpha_at_1 = inner_product("
                "operator_score_gradient, part_emission)"
            ),
            "response_similarity_implies_operator_interchangeability": False,
        },
        "experiment_roles": {
            "phase_0_exact_lift": {
                "data_role": "predecessor_replay_only",
                "requirements": [
                    "all 37576 physical atoms and shared slices close",
                    "all 38 atom-emitting role invocations close",
                    (
                        "every provided operator reference binds a typed native "
                        "observer receipt"
                    ),
                    "tied lexical storage identity closes across both roles",
                    "source fingerprint and source state are unchanged",
                ],
            },
            "A_exposed_discovery": {
                "may_fit": [
                    "state charts",
                    "logical part groups",
                    "ordered supermode paths",
                    "all thresholds and multiplicity inventory",
                ],
                "forbidden_features": [
                    "layer label as a generator identity",
                    "heldout B or C rows",
                ],
                "freeze_before_B": True,
            },
            "B_fresh_confirmation": {
                "may_fit": [],
                "tests": [
                    "frozen path vs target-local baseline",
                    "zero control",
                    "matched-size shuffled-membership control",
                    "matched-size shuffled-row control",
                    "controls matched by role event response mass and Fisher mass",
                    "family-block bootstrap with simultaneous correction",
                    "rare-support and chart-occupancy gates",
                ],
                "nomination_or_refit_allowed": False,
            },
            "C_fresh_intervention": {
                "selection": "exactly one frozen B-passing ordered path",
                "intervention": (
                    "finite same-boundary group gates applied in event order"
                ),
                "tests": [
                    "JVP versus finite-difference agreement",
                    "matched random ordered paths",
                    "chart-router stability under intervention",
                    "native parity control",
                    "delta-NLL KL and top1 fidelity",
                ],
                "later_part_emissions_recomputed_on_intervened_state": True,
                "reserve_may_be_opened": False,
            },
            "reserve": {
                "sealed": True,
                "required_if_C_selects_among_interventions_or_gate_rates": True,
            },
        },
        "claim_boundary": {
            "may_claim_after_success": (
                "exact parts-first decomposition and heldout evidence for "
                "state-conditioned ordered coalitions"
            ),
            "compression_claim_allowed": False,
            "parameter_savings_claim_allowed": False,
            "MAC_savings_claim_allowed": False,
            "latency_claim_allowed": False,
            "interchangeable_generator_claim_allowed": False,
            "invariant_circuit_ontology_claim_allowed": False,
            "attention_shared_KV_held_fixed": True,
            "first_order_response_is_causal_sufficiency": False,
            "column_atoms_are_checkpoint_basis_dependent": True,
            "egress_capture_must_be_vocab_chunked": True,
            "atoms_by_vocabulary_tensor_may_be_materialized": False,
        },
    }


def _canonical_bytes(state: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def protocol_semantic_sha256(state: Mapping[str, object]) -> str:
    payload = dict(state)
    payload.pop("protocol_sha256", None)
    return hashlib.sha256(SEMANTIC_DOMAIN + _canonical_bytes(payload)).hexdigest()


def build_parts_first_supermode_protocol() -> dict[str, object]:
    state = _protocol_without_sha256()
    state["protocol_sha256"] = protocol_semantic_sha256(state)
    validate_parts_first_supermode_protocol(state)
    return state


def render_parts_first_supermode_protocol(
    state: Mapping[str, object] | None = None,
) -> bytes:
    value = build_parts_first_supermode_protocol() if state is None else state
    validate_parts_first_supermode_protocol(value)
    return _canonical_bytes(value)


def validate_parts_first_supermode_protocol(
    state: Mapping[str, object],
) -> None:
    if type(state) is not dict:
        raise TypeError("protocol state must be an exact dict")
    expected = _protocol_without_sha256()
    if set(state) != set(expected) | {"protocol_sha256"}:
        raise ValueError("protocol top-level field inventory is invalid")
    candidate = deepcopy(state)
    claimed_sha256 = candidate.pop("protocol_sha256")
    if candidate != expected:
        raise ValueError("protocol state differs from the frozen contract")
    if claimed_sha256 != protocol_semantic_sha256(state):
        raise ValueError("protocol semantic hash does not close")
    if PHYSICAL_ATOM_COUNT != 37_576:
        raise AssertionError("physical atom accounting drifted")
    if ATOM_OWNED_SCALARS != 262_144_000:
        raise AssertionError("atom-owned scalar accounting drifted")
    if SHARED_SUBSTRATE_SCALARS != 5_954_176:
        raise AssertionError("shared-substrate accounting drifted")
    if SOURCE_SCALARS != 268_098_176:
        raise AssertionError("source scalar accounting drifted")
    schedule = state["event_schedule"]
    if not isinstance(schedule, list) or len(schedule) != 39:
        raise ValueError("exactly 39 ordered events are required")
    if [event["event_index"] for event in schedule] != list(range(39)):
        raise ValueError("event indices must be contiguous and ordered")
    if sum(bool(event["atom_emitting"]) for event in schedule) != 38:
        raise ValueError("exactly 38 atom-emitting role events are required")
