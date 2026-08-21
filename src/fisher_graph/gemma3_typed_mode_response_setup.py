"""Freeze Gemma's typed native-mode and common-response experiment.

The thirty-eight response modes are eighteen exact attention mutations,
eighteen exact MLP mutations, and the two logical endpoints of one tied
lexical state.  Native comparisons remain partitioned by four exact execution
ABIs.  A separate terminal-normalization mode exists only as exact baseline
infrastructure and is never admitted to the response atlas.

The common comparison is a finite, exact native-output ablation from scale one
to scale zero.  Its response is the four restricted-choice logits after
removing their mean.  A tangent may be reported as a diagnostic, but it cannot
replace or modify that finite response.

This module is standard-library-only and import inert.  Building, validating,
loading, or writing the setup does not open a predecessor, a prompt panel, or
model weights; tokenize text; import a tensor library; run an operator; or
compute an atlas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


GEMMA3_TYPED_MODE_RESPONSE_SETUP_SCHEMA = (
    "fisher_graph.gemma3_typed_mode_response.setup.v1"
)
GEMMA3_TYPED_MODE_RESPONSE_SETUP_FORMAT_VERSION = 1
SETUP_ID = "gemma3-typed-mode-response-v1"
DEFAULT_OUTPUT = Path(
    "artifacts/research/gemma3-typed-mode-response-setup-v1.json"
)
PINNED_SETUP_SHA256 = (
    "189887111e12acd8e86bb56574ac69ebaf44660491a99a333ebcc75f7942de0f"
)
PINNED_SETUP_FILE_SHA256 = (
    "a821b963c491b710784f4bf7745321208dfee8ba23201f4ece4f017d1be00aa0"
)
PINNED_TYPED_MODE_RESPONSE_SETUP_SHA256 = PINNED_SETUP_SHA256
PINNED_TYPED_MODE_RESPONSE_SETUP_FILE_SHA256 = PINNED_SETUP_FILE_SHA256

_SETUP_HASH_DOMAIN = b"fisher_graph:gemma3-typed-mode-response-setup:v1\0"
_NATIVE_ABI_HASH_DOMAIN = b"fisher_graph.mode_similarity_atlas.abi.v1\0"
_RESPONSE_ABI_HASH_DOMAIN = (
    b"fisher_graph.multi_abi_mode_atlas.downstream_response_abi.v1\0"
)
_ROW_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-response-row-catalog:v1\0"
)
_WEIGHTING_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-typed-mode-response-weighting-catalog:v1\0"
)
_BOOTSTRAP_DRAW_HASH_DOMAIN = (
    b"fisher_graph:gemma3-mode-atlas-bootstrap-draw:v1\0"
)
_BOOTSTRAP_CATALOG_HASH_DOMAIN = (
    b"fisher_graph.mode_similarity_atlas.bootstrap_catalog.v1\0"
)

PINNED_MODEL_ID = "google/gemma-3-270m"
PINNED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_SOURCE_PARAMETER_COUNT = 268_098_176
PINNED_SOURCE_MODEL_FINGERPRINT_SHA256 = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256 = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
PINNED_HIDDEN_WIDTH = 640
PINNED_VOCABULARY_SIZE = 262_144
PINNED_LAYER_COUNT = 18
PINNED_QUERY_HEADS = 4
PINNED_KEY_VALUE_HEADS = 1
PINNED_HEAD_DIMENSION = 256
PINNED_SLIDING_WINDOW = 512
PINNED_GLOBAL_ATTENTION_BLOCK_ORDINALS = (5, 11, 17)

PINNED_PANEL_PATH = Path("examples/gemma3_downstream_retention_v1.json")
PINNED_PANEL_FILE_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PINNED_PANEL_ID = "gemma3-single-token-forced-choice-v1"
PINNED_CHOICE_TOKENIZATION_CATALOG_SHA256 = (
    "8d9a2eed7698a210e21fde198477f561bf844ee3b0f35c7a651dbd0f5ab57840"
)
PINNED_LAST_QUERY_CATALOG_SHA256 = (
    "3a3fc3a3f021a60129d4fc3c02cce826a3ec22006462c82a2329397e7d791e8d"
)
PINNED_VALID_PROMPT_TOKEN_COUNT_MINIMUM = 11
PINNED_VALID_PROMPT_TOKEN_COUNT_MAXIMUM = 26
PINNED_VALID_PROMPT_ROW_COUNT = 814

ORDERED_FAMILY_IDS = (
    "integer_addition",
    "integer_subtraction",
    "integer_comparison",
    "lexical_antonym",
    "category_completion",
    "world_capitals",
)
_FAMILY_PREFIXES = {
    "integer_addition": "addition",
    "integer_subtraction": "subtraction",
    "integer_comparison": "comparison",
    "lexical_antonym": "antonym",
    "category_completion": "category",
    "world_capitals": "capital",
}

ATTENTION_MODE_IDS = tuple(
    f"mode.mutation.{2 * block + 1}" for block in range(PINNED_LAYER_COUNT)
)
MLP_MODE_IDS = tuple(
    f"mode.mutation.{2 * block + 2}" for block in range(PINNED_LAYER_COUNT)
)
LEXICAL_MODE_IDS = ("mode.lexical.ingress", "mode.lexical.egress")
RESPONSE_MODE_IDS = tuple(
    mode_id
    for pair in zip(ATTENTION_MODE_IDS, MLP_MODE_IDS, strict=True)
    for mode_id in pair
) + LEXICAL_MODE_IDS
TERMINAL_NORM_MODE_ID = "mode.terminal.norm"

RESPONSE_MODE_COUNT = 38
TOTAL_TYPED_MODE_COUNT = 39
RESPONSE_UNORDERED_PAIR_COUNT = 703
RESPONSE_FAMILY_PAIR_STAT_COUNT = 4_218
RESPONSE_EXAMPLE_PAIR_STAT_COUNT = 42_180
RESPONSE_TOKEN_ID_BASELINE_FORWARD_COUNT = 18
RESPONSE_INPUTS_EMBEDS_BASELINE_FORWARD_COUNT = 18
RESPONSE_INTERVENTION_FORWARD_COUNT = 684
RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT = 720

PROBES_PER_MODE_EXAMPLE = 2
ATTENTION_PROBE_COUNT = 2_160
ATTENTION_SELECTED_CANDIDATE_OUTPUT_COUNT = 38_880
ATTENTION_CAPTURE_BATCH_COUNT = 18
ATTENTION_CAPTURE_FORWARD_COUNT = 18
ATTENTION_CAPTURE_BACKWARD_COUNT = 18
ATTENTION_NATIVE_PREFIX_CALL_COUNT = 324
ATTENTION_FULL_SEQUENCE_VALID_TOKEN_EXECUTION_COUNT = 263_736
ATTENTION_LOGICAL_MAC_ESTIMATE = 436_467_253_248
RESIDUAL_PROFILE_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 640)

PINNED_MLP_SETUP_PATH = Path(
    "artifacts/research/gemma3-mlp-mode-similarity-atlas-setup-v1.json"
)
PINNED_MLP_SETUP_SHA256 = (
    "7edb27bc2a2778459764a855bba0d0f358b55b06b547e2a4b6c1029ee0110dd0"
)
PINNED_MLP_SETUP_FILE_SHA256 = (
    "4c3f39fdf18917be77f246ecdca852d001810915f9fdecc516e1e2860f088290"
)
PINNED_MLP_REPORT_PATH = Path(
    ".local-runs/google--gemma-3-270m/mlp-mode-similarity-atlas-v1.json"
)
PINNED_MLP_REPORT_SHA256 = (
    "0fc453081460885c2d07015ad4cf3ffae47970bec47855a34f1211a188da5161"
)
PINNED_MLP_REPORT_FILE_SHA256 = (
    "82a15b23b4a073a8874206a769ea5e8fbc05da082808e79ba13b2b78e7a69fb2"
)

DEFAULT_ATTENTION_REPORT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/attention-mode-similarity-atlas-v1.json"
)
DEFAULT_ATTENTION_SVG_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/attention-mode-similarity-atlas-v1-heatmap.svg"
)
DEFAULT_RESPONSE_REPORT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/typed-mode-response-atlas-v1.json"
)
DEFAULT_RESPONSE_SVG_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/typed-mode-response-atlas-v1-heatmap.svg"
)

ATTENTION_NATIVE_ABI = {
    "abi_name": "carrier_mutation.sequence_routed.h640",
    "abi_version": 1,
    "input_signature": "raw_carrier_sequence.float.h640",
    "output_signature": "additive_full_atom_delta_sequence.float.h640",
    "context_signature": "causal_mask.positions.rope_and_routing.exact_prompt",
}
MLP_NATIVE_ABI = {
    "abi_name": "carrier_mutation.token_local.h640",
    "abi_version": 1,
    "input_signature": "raw_carrier_row.float.h640",
    "output_signature": "additive_full_atom_delta.float.h640",
    "context_signature": "token_local.no_external_context",
}
LEXICAL_INGRESS_NATIVE_ABI = {
    "abi_name": "lexical_boundary.ingress.vocab_to_h640",
    "abi_version": 1,
    "input_signature": "token_identity.int64",
    "output_signature": "initial_carrier_row.float.h640",
    "context_signature": "tied_lexical_state.ingress_lookup",
}
LEXICAL_EGRESS_NATIVE_ABI = {
    "abi_name": "lexical_boundary.egress.h640_to_vocab",
    "abi_version": 1,
    "input_signature": "terminal_normalized_carrier_row.float.h640",
    "output_signature": "vocabulary_logits.float.v262144",
    "context_signature": "tied_lexical_state.egress_projection",
}
TERMINAL_NORM_NATIVE_ABI = {
    "abi_name": "terminal_boundary.rmsnorm.h640",
    "abi_version": 1,
    "input_signature": "pre_terminal_carrier_row.float.h640",
    "output_signature": "terminal_normalized_carrier_row.float.h640",
    "context_signature": "token_local.exact_baseline_only",
}

ATLAS_CONFIG = {
    "energy_epsilon": 1.0e-12,
    "sufficient_stat_relative_tolerance": 1.0e-7,
    "connectivity_tolerance": 1.0e-12,
    "fiedler_eigengap_relative_tolerance": 1.0e-8,
    "fiedler_coordinate_tie_tolerance": 1.0e-10,
    "nearest_neighbor_tie_tolerance": 1.0e-10,
    "affinity_max_exponent": 700.0,
}
BOOTSTRAP_SEED = 2_311
BOOTSTRAP_RESAMPLES = 10_000


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _setup_sha256(value: Mapping[str, object]) -> str:
    return _sha256(_SETUP_HASH_DOMAIN, value)


def _native_abi_sha256(value: Mapping[str, object]) -> str:
    return _sha256(_NATIVE_ABI_HASH_DOMAIN, value)


def _example_ids(family_id: str) -> list[str]:
    prefix = _FAMILY_PREFIXES[family_id]
    return [f"{prefix}-{ordinal:02d}" for ordinal in range(1, 11)]


def _ordered_example_ids() -> list[str]:
    return [
        example_id
        for family_id in ORDERED_FAMILY_IDS
        for example_id in _example_ids(family_id)
    ]


def _atlas_config_payload() -> dict[str, object]:
    return {
        "affinity_kernel": "exp_neg_distance_over_positive_median_v1",
        "affinity_max_exponent": ATLAS_CONFIG["affinity_max_exponent"],
        "connectivity_tolerance": ATLAS_CONFIG["connectivity_tolerance"],
        "cosine_role": "diagnostic_only",
        "distance_metric": "bidirectional_equal_family_mean_relative_sse_v1",
        "energy_epsilon": ATLAS_CONFIG["energy_epsilon"],
        "fiedler_coordinate_tie_tolerance": ATLAS_CONFIG[
            "fiedler_coordinate_tie_tolerance"
        ],
        "fiedler_eigengap_relative_tolerance": ATLAS_CONFIG[
            "fiedler_eigengap_relative_tolerance"
        ],
        "laplacian": "symmetric_normalized_v1",
        "mode_id_role": "receipt_label_only",
        "mode_kind_role": "receipt_label_only",
        "nearest_neighbor_tie_tolerance": ATLAS_CONFIG[
            "nearest_neighbor_tie_tolerance"
        ],
        "ordering_quality_random_permutation_count": 10_000,
        "ordering_quality_random_permutation_seed": 0,
        "state_fingerprint_role": "canonical_axis_and_global_reversal_only",
        "sufficient_stat_relative_tolerance": ATLAS_CONFIG[
            "sufficient_stat_relative_tolerance"
        ],
    }


def _within_family_bootstrap_config() -> dict[str, object]:
    return {
        "draw_catalog_digest_encoding": (
            "sha256(catalog_domain||canonical_config||"
            "repeated(resample_u64be,family_u32be,draw_u32be,example_u32be))"
        ),
        "draw_catalog_digest_domain_hex": _BOOTSTRAP_CATALOG_HASH_DOMAIN.hex(),
        "draw_digest_selection": "first_8_sha256_bytes_u64be_mod_example_count",
        "draw_preimage_encoding": (
            "domain||seed_u64be||resample_u64be||family_u32be||draw_u32be"
        ),
        "draw_protocol": "sha256_counter_modulo_example_count_v1",
        "draw_protocol_domain_hex": _BOOTSTRAP_DRAW_HASH_DOMAIN.hex(),
        "draws_per_family": 10,
        "families": [
            {
                "family_id": family_id,
                "example_cluster_ids": _example_ids(family_id),
            }
            for family_id in ORDERED_FAMILY_IDS
        ],
        "maximum_degenerate_fraction": 0.01,
        "resample_count": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "sampling_unit": "example_plus_selected_probes_cluster",
        "with_replacement": True,
    }


def _common_response_bootstrap_config() -> dict[str, object]:
    """Freeze example-cluster resampling for four-coordinate responses.

    This is intentionally a distinct semantic sampling unit from the selected
    native-probe clusters used by the attention atlas, even though both use the
    same deterministic counter draw protocol and family manifests.
    """

    config = _within_family_bootstrap_config()
    config["sampling_unit"] = "example_plus_centered_logits4_response_cluster"
    config["response_coordinates_per_example"] = 4
    config["all_four_coordinates_resampled_together"] = True
    return config


def _response_weighting_policy() -> dict[str, object]:
    return {
        "coordinate_weight": (
            "square_of_analytic_baseline_restricted_choice_cross_entropy_"
            "gradient_for_each_of_four_choice_logits"
        ),
        "analytic_gradient_formula": "g_k=softmax(z_scale1)_k-one[k=gold]",
        "output_Fisher_formula": "w_k=g_k^2_computed_in_float64",
        "baseline_logits_source": (
            "exact_scale1_restricted_choice_logits4_before_centering"
        ),
        "example_weight": "equal_within_family",
        "family_weight": "equal_one_sixth",
        "family_order": list(ORDERED_FAMILY_IDS),
        "gold_choice_or_Fisher_weighting_used": True,
        "runtime_value_catalog_encoding": (
            "ordered_family_example_choice_coordinate_and_canonical_float64_"
            "output_Fisher_weight"
        ),
        "runtime_value_catalog_SHA256_required_before_atlas_construction": True,
        "projection": "C4=I4-(1/4)*ones4x4",
    }


def _response_row_catalog_payload() -> dict[str, object]:
    return {
        "choice_tokenization_catalog_sha256": (
            PINNED_CHOICE_TOKENIZATION_CATALOG_SHA256
        ),
        "last_query_catalog_sha256": PINNED_LAST_QUERY_CATALOG_SHA256,
        "ordered_example_ids": _ordered_example_ids(),
        "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
        "response_coordinates_per_example": 4,
    }


def _common_response_abi() -> dict[str, object]:
    row_sha = _sha256(_ROW_CATALOG_HASH_DOMAIN, _response_row_catalog_payload())
    weight_sha = _sha256(
        _WEIGHTING_CATALOG_HASH_DOMAIN,
        _response_weighting_policy(),
    )
    payload: dict[str, object] = {
        "abi_id": "downstream_response.restricted_choice_logits4.v1",
        "abi_name": "downstream_response.restricted_choice_logits4",
        "abi_version": 1,
        "baseline_signature": (
            "exact_model.scale1.terminal_norm.lexical_readout.logit_softcap"
        ),
        "comparison_definition": "aligned_weighted_downstream_response_delta_v1",
        "intervention_signature": (
            "exact_native_occurrence_output_scale.1_to_0.one_logical_mode"
        ),
        "native_operator_interchangeability_implied": False,
        "response_coordinate_signature": (
            "centered_restricted_choice_logits4.delta.z1_minus_z0"
        ),
        "response_width": 4,
        "row_catalog_sha256": row_sha,
        "weighting_catalog_sha256": weight_sha,
    }
    payload["sha256"] = _sha256(_RESPONSE_ABI_HASH_DOMAIN, payload)
    return payload


def _mode_catalog() -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for block_ordinal, (attention_id, mlp_id) in enumerate(
        zip(ATTENTION_MODE_IDS, MLP_MODE_IDS, strict=True)
    ):
        attention_mutation = 2 * block_ordinal + 1
        mlp_mutation = attention_mutation + 1
        catalog.append(
            {
                "mode_id": attention_id,
                "native_abi_id": (
                    "abi.carrier_mutation.sequence_routed.h640.v1"
                ),
                "source_invocation_id": f"mutation.{attention_mutation}",
                "source_atom_id": f"atom.attention.layer.{block_ordinal}",
                "source_block_ordinal": block_ordinal,
                "operator_kind": "attention",
                "source_implementation_label": "attention",
                "generic_core_mode_kind": (
                    "carrier_mutation.sequence_routed.h640"
                ),
                "native_source_state_sha256": (
                    "captured_and_authenticated_at_runtime"
                ),
                "response_intervention_id": f"intervention.{attention_id}",
                "eligible_for_native_atlas": True,
                "eligible_for_common_response": True,
                "provenance_fields_allowed_as_numerical_features": False,
            }
        )
        catalog.append(
            {
                "mode_id": mlp_id,
                "native_abi_id": "abi.carrier_mutation.token_local.h640.v1",
                "source_invocation_id": f"mutation.{mlp_mutation}",
                "source_atom_id": f"atom.mlp.layer.{block_ordinal}",
                "source_block_ordinal": block_ordinal,
                "operator_kind": "mlp",
                "source_implementation_label": "MLP",
                "generic_core_mode_kind": "carrier_mutation.token_local.h640",
                "native_source_state_sha256": (
                    "authenticated_by_existing_MLP_native_atlas"
                ),
                "response_intervention_id": f"intervention.{mlp_id}",
                "eligible_for_native_atlas": True,
                "eligible_for_common_response": True,
                "provenance_fields_allowed_as_numerical_features": False,
            }
        )
    catalog.extend(
        (
            {
                "mode_id": LEXICAL_MODE_IDS[0],
                "native_abi_id": (
                    "abi.lexical_boundary.ingress.vocab_to_h640.v1"
                ),
                "source_invocation_id": "lexical.ingress",
                "source_atom_id": "atom.tied_lexical.embedding_lm_head",
                "source_block_ordinal": None,
                "operator_kind": "lexical_ingress",
                "source_implementation_label": "embedding_ingress",
                "generic_core_mode_kind": "lexical_boundary.ingress",
                "native_source_state_sha256": (
                    "captured_once_for_the_shared_tied_state_at_runtime"
                ),
                "shared_physical_state_id": (
                    "state.tied_lexical.embedding_lm_head"
                ),
                "response_intervention_id": (
                    "intervention.mode.lexical.ingress.native_occurrence."
                    "scale1_to0"
                ),
                "eligible_for_native_atlas": False,
                "eligible_for_common_response": True,
                "provenance_fields_allowed_as_numerical_features": False,
            },
            {
                "mode_id": LEXICAL_MODE_IDS[1],
                "native_abi_id": (
                    "abi.lexical_boundary.egress.h640_to_vocab.v1"
                ),
                "source_invocation_id": "lexical.egress",
                "source_atom_id": "atom.tied_lexical.embedding_lm_head",
                "source_block_ordinal": None,
                "operator_kind": "lexical_egress",
                "source_implementation_label": "tied_lexical_egress",
                "generic_core_mode_kind": "lexical_boundary.egress",
                "native_source_state_sha256": (
                    "captured_once_for_the_shared_tied_state_at_runtime"
                ),
                "shared_physical_state_id": (
                    "state.tied_lexical.embedding_lm_head"
                ),
                "response_intervention_id": (
                    "intervention.mode.lexical.egress.native_occurrence."
                    "scale1_to0"
                ),
                "eligible_for_native_atlas": False,
                "eligible_for_common_response": True,
                "provenance_fields_allowed_as_numerical_features": False,
            },
        )
    )
    return catalog


def _build_without_hash() -> dict[str, object]:
    native_abis = {
        "attention": dict(ATTENTION_NATIVE_ABI),
        "mlp": dict(MLP_NATIVE_ABI),
        "lexical_ingress": dict(LEXICAL_INGRESS_NATIVE_ABI),
        "lexical_egress": dict(LEXICAL_EGRESS_NATIVE_ABI),
    }
    common_response_abi = _common_response_abi()
    ordered_examples = _ordered_example_ids()
    return {
        "schema": GEMMA3_TYPED_MODE_RESPONSE_SETUP_SCHEMA,
        "format_version": GEMMA3_TYPED_MODE_RESPONSE_SETUP_FORMAT_VERSION,
        "setup_id": SETUP_ID,
        "artifact_role": (
            "setup_only_typed_native_ABI_atlases_and_exact_common_response"
        ),
        "setup_receipt": {
            "predecessor_or_panel_opened": False,
            "tensor_library_imported": False,
            "model_or_tokenizer_loaded": False,
            "prompt_text_or_token_ids_read": False,
            "native_operator_executed": False,
            "finite_ablation_or_tangent_executed": False,
            "distance_order_neighbor_cluster_merge_or_compression_computed": False,
            "experiment_report_written": False,
        },
        "model_contract": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "source_parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
            "source_model_fingerprint_sha256": (
                PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
            ),
            "source_execution_fingerprint_sha256": (
                PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            ),
            "dtype": "float32",
            "device": "cpu",
            "hidden_width": PINNED_HIDDEN_WIDTH,
            "vocabulary_size": PINNED_VOCABULARY_SIZE,
            "layer_count": PINNED_LAYER_COUNT,
            "query_head_count": PINNED_QUERY_HEADS,
            "key_value_head_count": PINNED_KEY_VALUE_HEADS,
            "head_dimension": PINNED_HEAD_DIMENSION,
            "sliding_window": PINNED_SLIDING_WINDOW,
            "global_attention_block_ordinals": list(
                PINNED_GLOBAL_ATTENTION_BLOCK_ORDINALS
            ),
            "parameters_training_mode_and_persistent_buffers_immutable": True,
        },
        "discovery_data_firewall": {
            "panel_path": str(PINNED_PANEL_PATH),
            "panel_id": PINNED_PANEL_ID,
            "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
            "ordered_family_ids": list(ORDERED_FAMILY_IDS),
            "ordered_example_ids": ordered_examples,
            "examples_per_family": 10,
            "prompt_count": 60,
            "choice_tokenization_catalog_sha256": (
                PINNED_CHOICE_TOKENIZATION_CATALOG_SHA256
            ),
            "last_query_catalog_sha256": PINNED_LAST_QUERY_CATALOG_SHA256,
            "all_prompts_spent": True,
            "one_role_only": "spent_descriptive_development",
            "native_MLP_attention_and_common_response_may_describe_this_panel": (
                True
            ),
            "leave_one_family_out_is_stability_not_clean_assessment": True,
            "no_prompt_family_or_paraphrase_may_support_a_future_selection_"
            "merge_or_compression_claim": True,
            "fresh_panel_requirements_for_any_future_claim": [
                "new_authenticated_panel_SHA256",
                "zero_example_id_overlap",
                "zero_exact_prompt_text_overlap",
                "zero_paraphrase_or_template_instance_overlap",
                "overlap_audit_before_model_load",
                "atlas_blind_authoring",
            ],
        },
        "short_context_limitation": {
            "observed_valid_prompt_token_count_minimum": (
                PINNED_VALID_PROMPT_TOKEN_COUNT_MINIMUM
            ),
            "observed_valid_prompt_token_count_maximum": (
                PINNED_VALID_PROMPT_TOKEN_COUNT_MAXIMUM
            ),
            "observed_valid_prompt_rows": PINNED_VALID_PROMPT_ROW_COUNT,
            "sliding_window_token_count": PINNED_SLIDING_WINDOW,
            "sliding_window_cutoff_exercised": False,
            "global_versus_sliding_attention_long_context_behavior_tested": False,
            "no_result_may_be_generalized_beyond_the_observed_11_to_26_token_"
            "range": True,
            "future_attention_neighbor_cluster_merge_or_compression_claim_"
            "requires": {
                "fresh_prompt_and_paraphrase_disjoint_panel": True,
                "at_least_one_valid_prompt_token_count_greater_than": 512,
                "minimum_qualifying_valid_prompt_token_count": 513,
                "must_include_queries_with_keys_outside_the_512_token_window": (
                    True
                ),
            },
        },
        "mode_ontology": {
            "definition": (
                "one_parameterized_state_mutation_at_one_typed_causal_"
                "occurrence_or_one_logical_endpoint_of_an_explicitly_tied_state"
            ),
            "implementation_names_are_provenance_not_ontology": True,
            "source_block_invocation_and_component_labels_are_not_features": (
                True
            ),
            "native_ABI_count_for_response_modes": 4,
            "native_comparison_requires_exact_ABI_equality": True,
            "cross_ABI_native_execution_forbidden": True,
            "common_response_similarity_does_not_imply_native_"
            "interchangeability": True,
            "response_mode_count": RESPONSE_MODE_COUNT,
            "total_typed_mode_count_including_terminal_baseline": (
                TOTAL_TYPED_MODE_COUNT
            ),
        },
        "native_abis": native_abis,
        "native_abi_sha256s": {
            name: _native_abi_sha256(abi) for name, abi in native_abis.items()
        },
        "typed_mode_catalog": _mode_catalog(),
        "scope_freeze": {
            "ordered_response_mode_ids": list(RESPONSE_MODE_IDS),
            "attention_mode_ids": list(ATTENTION_MODE_IDS),
            "mlp_mode_ids": list(MLP_MODE_IDS),
            "lexical_mode_ids": list(LEXICAL_MODE_IDS),
            "response_mode_count": RESPONSE_MODE_COUNT,
            "attention_mode_count": 18,
            "mlp_mode_count": 18,
            "lexical_logical_mode_count": 2,
            "native_ABI_count": 4,
            "unordered_response_pair_count": RESPONSE_UNORDERED_PAIR_COUNT,
            "per_family_unordered_pair_stat_count": (
                RESPONSE_FAMILY_PAIR_STAT_COUNT
            ),
            "per_example_unordered_pair_stat_count": (
                RESPONSE_EXAMPLE_PAIR_STAT_COUNT
            ),
            "terminal_norm_in_response_mode_ids": False,
            "source_ordinal_or_operator_label_used_as_numerical_feature": False,
            "source_parameter_or_edge_changed": False,
            "fit_merge_cluster_or_compression_performed": False,
        },
        "tied_lexical_state_contract": {
            "physical_state_id": "state.tied_lexical.embedding_lm_head",
            "logical_mode_ids": list(LEXICAL_MODE_IDS),
            "logical_native_ABIs_remain_distinct": True,
            "one_physical_storage_span_and_one_state_fingerprint_required": True,
            "physical_parameter_state_group_count": 1,
            "native_occurrence_intervention_count": 2,
            "native_occurrence_intervention_ids_are_distinct": True,
            "ingress_intervention": (
                "scale_only_the_exact_native_ingress_occurrence_output_from_"
                "one_to_zero"
            ),
            "egress_intervention": (
                "scale_only_the_exact_native_egress_occurrence_output_from_"
                "one_to_zero"
            ),
            "both_logical_modes_are_executed_independently": True,
            "both_finite_responses_must_be_exactly_equal": True,
            "pairwise_family_and_example_SSE_must_be_exactly_zero": True,
            "energy_ingress_equals_dot_equals_energy_egress_exactly": True,
            "logical_modes_remain_distinct_after_response_equivalence_collapse": (
                True
            ),
            "tied_identity_does_not_make_their_native_ABIs_equal": True,
        },
        "terminal_norm_baseline": {
            "mode_id": TERMINAL_NORM_MODE_ID,
            "operator_kind": "terminal_norm",
            "native_abi": dict(TERMINAL_NORM_NATIVE_ABI),
            "native_abi_sha256": _native_abi_sha256(TERMINAL_NORM_NATIVE_ABI),
            "source_invocation_id": "terminal.rmsnorm",
            "role": "exact_common_response_baseline_infrastructure_only",
            "included_in_38_response_modes": False,
            "native_atlas_built": False,
            "finite_ablation_response_measured": False,
            "generator_family_neighbor_cluster_merge_or_compression_eligible": (
                False
            ),
            "terminal_RMSNorm_tied_lexical_readout_and_logit_softcap_exact": (
                True
            ),
        },
        "predecessors": {
            "mlp_native_atlas": {
                "setup_path": str(PINNED_MLP_SETUP_PATH),
                "setup_schema": (
                    "fisher_graph.gemma3_mlp_mode_similarity_atlas.setup.v1"
                ),
                "setup_sha256": PINNED_MLP_SETUP_SHA256,
                "setup_file_sha256": PINNED_MLP_SETUP_FILE_SHA256,
                "report_path": str(PINNED_MLP_REPORT_PATH),
                "report_schema": (
                    "fisher_graph.gemma3_mlp_mode_similarity_atlas.experiment.v1"
                ),
                "report_sha256": PINNED_MLP_REPORT_SHA256,
                "report_file_sha256": PINNED_MLP_REPORT_FILE_SHA256,
                "required_diagnostic_outcome": "mode_similarity_discovery_only",
                "ordered_mode_ids": list(MLP_MODE_IDS),
                "compatibility_abi_sha256": _native_abi_sha256(MLP_NATIVE_ABI),
                "internal_and_file_SHA256_values_verified_before_model_load": (
                    True
                ),
                "native_distances_may_not_be_reused_as_common_response_stats": (
                    True
                ),
                "stable_neighbor_hypotheses_are_not_inherited_as_claims": True,
            }
        },
        "required_native_atlases": {
            "attention": {
                "required_before_common_response_atlas": True,
                "must_be_newly_executed": True,
                "report_path": str(DEFAULT_ATTENTION_REPORT_OUTPUT),
                "report_schema": (
                    "fisher_graph.gemma3_native_attention_mode_atlas."
                    "experiment.v1"
                ),
                "report_SHA256_status": "captured_and_authenticated_at_runtime",
                "file_SHA256_status": "captured_and_authenticated_at_runtime",
                "ordered_mode_ids": list(ATTENTION_MODE_IDS),
                "mode_count": 18,
                "compatibility_abi_sha256": _native_abi_sha256(
                    ATTENTION_NATIVE_ABI
                ),
                "required_diagnostic_outcome": "mode_similarity_discovery_only",
                "full_native_sequence_operator_required": True,
                "short_context_limitation_must_be_carried_into_report": True,
            },
            "lexical_ingress": {
                "mode_count": 1,
                "status": "insufficient_modes_for_native_seriation",
            },
            "lexical_egress": {
                "mode_count": 1,
                "status": "insufficient_modes_for_native_seriation",
            },
        },
        "attention_atlas_config": _atlas_config_payload(),
        "attention_within_family_bootstrap_config": (
            _within_family_bootstrap_config()
        ),
        "attention_experiment_contract": {
            "operator_kind": "attention",
            "ordered_mode_ids": list(ATTENTION_MODE_IDS),
            "full_atom_ownership": [
                "input_RMSNorm",
                "Q_K_V_projections",
                "Q_and_K_normalization",
                "RoPE",
                "causal_grouped_query_routing",
                "output_projection",
                "post_attention_RMSNorm",
            ],
            "native_input_unit": (
                "full_exact_prompt_carrier_sequence_with_mask_positions_and_"
                "routing_context"
            ),
            "native_output_unit": "full_additive_carrier_sequence_delta_h640",
            "native_prefix_replay": (
                "run_each_candidate_full_attention_atom_on_each_origin_full_"
                "sequence_before_selecting_query_rows"
            ),
            "rowwise_attention_cross_execution_forbidden": True,
            "Fisher_objective": (
                "summed_gold_cross_entropy_over_restricted_four_choice_"
                "softmax_at_last_valid_prompt_query"
            ),
            "top_probe_rows_per_mode_example": PROBES_PER_MODE_EXAMPLE,
            "probe_selection_occurs_after_full_sequence_execution": True,
            "capture_batch_count": ATTENTION_CAPTURE_BATCH_COUNT,
            "capture_forward_count": ATTENTION_CAPTURE_FORWARD_COUNT,
            "capture_backward_count": ATTENTION_CAPTURE_BACKWARD_COUNT,
            "expected_calls_per_attention_full_atom_during_capture": 18,
            "expected_full_model_embedding_and_lexical_head_calls_during_"
            "capture": 18,
            "valid_prompt_rows_per_mode": PINNED_VALID_PROMPT_ROW_COUNT,
            "selected_probe_count": ATTENTION_PROBE_COUNT,
            "selected_candidate_row_output_count": (
                ATTENTION_SELECTED_CANDIDATE_OUTPUT_COUNT
            ),
            "native_prefix_call_count": ATTENTION_NATIVE_PREFIX_CALL_COUNT,
            "full_sequence_valid_token_execution_count": (
                ATTENTION_FULL_SEQUENCE_VALID_TOKEN_EXECUTION_COUNT
            ),
            "cross_execution_per_candidate": {
                "run_attention_prefix_calls": 18,
                "each_owned_attention_component_calls": 18,
                "all_foreign_attention_MLP_post_feedforward_full_model_"
                "embedding_and_lexical_head_calls": 0,
            },
            "logical_MAC_estimate": ATTENTION_LOGICAL_MAC_ESTIMATE,
            "logical_MAC_formula": (
                "sum_over_18_candidates_and_18_origin_batches_of_exact_Q_K_V_"
                "O_projection_MACs_plus_routed_attention_score_and_value_"
                "product_MACs_over_the_frozen_batch_shapes"
            ),
            "selected_38880_rows_are_not_the_compute_unit": True,
            "sufficient_statistic_accumulation_dtype": "float64",
            "self_replay_numerical_parity": {
                "maximum_absolute_delta_error": 1.0e-6,
                "delta_NRMSE_maximum": 1.0e-7,
                "every_mode_by_family_receipt_must_pass": True,
            },
            "weighted_residual_rank_profile_ranks": list(
                RESIDUAL_PROFILE_RANKS
            ),
            "rank_profiles_are_diagnostic_only": True,
            "source_state_and_execution_fingerprints_unchanged_required": True,
            "compression_fit_merge_or_edge_change_forbidden": True,
            "short_context_limitation": "top_level_short_context_limitation",
        },
        "common_response_abi": common_response_abi,
        "common_response_atlas_config": _atlas_config_payload(),
        "common_response_within_family_bootstrap_config": (
            _common_response_bootstrap_config()
        ),
        "common_response_contract": {
            "ordered_mode_ids": list(RESPONSE_MODE_IDS),
            "mode_count": RESPONSE_MODE_COUNT,
            "physical_parameter_state_group_count": 37,
            "logical_intervention_count": 38,
            "native_occurrence_intervention_count": 38,
            "token_id_baseline_parity_forward_count": (
                RESPONSE_TOKEN_ID_BASELINE_FORWARD_COUNT
            ),
            "canonical_inputs_embeds_baseline_forward_count": (
                RESPONSE_INPUTS_EMBEDS_BASELINE_FORWARD_COUNT
            ),
            "logical_intervention_forward_count": (
                RESPONSE_INTERVENTION_FORWARD_COUNT
            ),
            "total_full_model_forward_count": (
                RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT
            ),
            "full_model_forward_count_formula": "18+18+(38*18)=720",
            "backward_count_for_primary_response": 0,
            "restricted_choice_cross_entropy_gradient_for_optional_tangent": (
                "analytic_no_backward_required_for_primary_weights"
            ),
            "primary_measurement": (
                "exact_native_output_scale_one_to_zero_finite_ablation"
            ),
            "native_baseline_scale": 1,
            "ablated_target_scale": 0,
            "response_formula": (
                "r_m=C4*(restricted_logits4_at_scale1-"
                "restricted_logits4_at_scale0)"
            ),
            "centering_matrix": "C4=I4-(1/4)*ones4x4",
            "center_each_example_before_any_pair_statistic": True,
            "response_width": 4,
            "response_effective_rank_after_centering": 3,
            "baseline_execution": (
                "exact_source_model_with_all_native_outputs_at_scale1"
            ),
            "all_non_target_modes_terminal_RMSNorm_tied_lexical_readout_and_"
            "logit_softcap_remain_exact": True,
            "mutation_intervention_scope": (
                "scale_only_the_target_full_native_occurrence_output"
            ),
            "tied_lexical_intervention_scope": (
                "execute_distinct_ingress_zero_and_egress_zero_native_"
                "occurrence_interventions_over_one_tied_parameter_state_group"
            ),
            "tied_lexical_execution_receipt": (
                "execute_both_logical_labels_and_require_exact_identical_"
                "centered_logits4_without_reusing_an_intervention_ID"
            ),
            "row_catalog": _response_row_catalog_payload(),
            "weighting_catalog": _response_weighting_policy(),
            "unordered_pair_count": RESPONSE_UNORDERED_PAIR_COUNT,
            "family_pair_sufficient_stat_count": (
                RESPONSE_FAMILY_PAIR_STAT_COUNT
            ),
            "example_pair_sufficient_stat_count": (
                RESPONSE_EXAMPLE_PAIR_STAT_COUNT
            ),
            "per_family_pair_row_count": 40,
            "per_example_pair_row_count": 4,
            "output_Fisher_weight_catalog_binding": {
                "protocol_catalog_sha256": common_response_abi[
                    "weighting_catalog_sha256"
                ],
                "runtime_float64_value_catalog_SHA256_required": True,
                "runtime_catalog_must_be_authenticated_before_pair_stats": True,
                "all_pair_statistics_use_the_same_baseline_weight_catalog": True,
            },
            "distance_normalization": {
                "coordinate_weights": "raw_baseline_output_Fisher_g_squared",
                "per_example_weight_renormalization": False,
                "per_family_weighted_SSE": "sum(w*(r_a-r_b)^2)",
                "per_family_mode_a_energy": "sum(w*r_a^2)",
                "per_family_mode_b_energy": "sum(w*r_b^2)",
                "directed_a_to_b": (
                    "SSE/max(mode_a_energy,atlas_config.energy_epsilon)"
                ),
                "directed_b_to_a": (
                    "SSE/max(mode_b_energy,atlas_config.energy_epsilon)"
                ),
                "directed_distance": (
                    "equal_arithmetic_mean_of_six_family_relative_SSE_values"
                ),
                "symmetric_distance": "0.5*(d(a_to_b)+d(b_to_a))",
                "family_row_weighting_used": False,
            },
            "pair_sufficient_statistics": [
                "weighted_SSE",
                "weighted_mode_a_energy",
                "weighted_dot",
                "weighted_mode_b_energy",
                "row_count",
            ],
            "tied_lexical_pair_is_an_exact_response_equivalence_class": True,
            "common_response_geometry_is_not_native_interchangeability": True,
            "optional_tangent_diagnostic": {
                "allowed": True,
                "definition": (
                    "C4*d_restricted_logits4_d_scale_at_scale1_with_the_same_"
                    "target_scope"
                ),
                "finite_ablation_remains_primary": True,
                "may_enter_primary_pair_statistics_distance_affinity_order_"
                "or_claims": False,
                "absence_is_valid": True,
            },
        },
        "execution_order": [
            "authenticate_setup_panel_model_and_existing_MLP_native_atlas",
            "run_and_authenticate_new_attention_native_atlas",
            "capture_exact_scale1_baselines_and_38_occurrence_scale0_"
            "interventions",
            "materialize_38_logical_centered_logits4_response_modes",
            "build_descriptive_common_response_atlas_on_exact_equivalence_"
            "classes",
            "write_authenticated_reports_without_clobber",
        ],
        "output_contract": {
            "attention_report_schema": (
                "fisher_graph.gemma3_native_attention_mode_atlas.experiment.v1"
            ),
            "attention_report_path": str(DEFAULT_ATTENTION_REPORT_OUTPUT),
            "attention_SVG_path": str(DEFAULT_ATTENTION_SVG_OUTPUT),
            "attention_report_maximum_bytes": 64 * 1024 * 1024,
            "common_response_report_schema": (
                "fisher_graph.gemma3_typed_mode_response.experiment.v1"
            ),
            "common_response_report_path": str(DEFAULT_RESPONSE_REPORT_OUTPUT),
            "common_response_SVG_path": str(DEFAULT_RESPONSE_SVG_OUTPUT),
            "common_response_report_maximum_bytes": 128 * 1024 * 1024,
            "atomic_publish": True,
            "refuse_existing_JSON_or_SVG_path": True,
            "reject_symlink_path_components": True,
            "strict_internal_and_file_SHA256_required": True,
            "contains_model_weights_raw_prompts_token_ids_activations_"
            "gradients_or_raw_responses": False,
        },
        "claim_boundary": {
            "setup_only": True,
            "new_model_prompt_or_metric_access_occurred": False,
            "descriptive_native_and_common_response_geometry_only": True,
            "native_attention_or_common_response_result_claimed": False,
            "neighbor_hypothesis_claimed": False,
            "cluster_claimed": False,
            "generator_family_claimed": False,
            "merge_or_compression_claimed": False,
            "causal_or_functional_interchangeability_claimed": False,
            "long_context_attention_behavior_claimed": False,
            "parameter_MAC_byte_memory_latency_or_quality_reduction_claimed": (
                False
            ),
            "future_selection_requires_fresh_prompt_and_paraphrase_disjoint_"
            "data": True,
        },
    }


def _validate_built_setup(value: Mapping[str, object]) -> None:
    if value["schema"] != GEMMA3_TYPED_MODE_RESPONSE_SETUP_SCHEMA:
        raise ValueError("typed-mode setup schema drifted")
    catalog = value["typed_mode_catalog"]
    if len(catalog) != RESPONSE_MODE_COUNT:
        raise ValueError("typed response mode count drifted")
    mode_ids = [entry["mode_id"] for entry in catalog]
    if mode_ids != list(RESPONSE_MODE_IDS) or len(set(mode_ids)) != 38:
        raise ValueError("typed response mode order or identity drifted")
    if set(value["native_abis"]) != {
        "attention",
        "mlp",
        "lexical_ingress",
        "lexical_egress",
    }:
        raise ValueError("the four response-mode native ABIs drifted")
    if value["scope_freeze"]["terminal_norm_in_response_mode_ids"]:
        raise ValueError("terminal norm entered the response atlas")
    if value["terminal_norm_baseline"]["included_in_38_response_modes"]:
        raise ValueError("terminal norm entered the response inventory")
    if value["common_response_contract"]["response_formula"] != (
        "r_m=C4*(restricted_logits4_at_scale1-"
        "restricted_logits4_at_scale0)"
    ):
        raise ValueError("primary finite response formula drifted")
    if not value["discovery_data_firewall"]["all_prompts_spent"]:
        raise ValueError("the development panel must remain spent")
    if value["short_context_limitation"]["sliding_window_cutoff_exercised"]:
        raise ValueError("the short panel cannot exercise the 512-token cutoff")
    if any(
        value["claim_boundary"][field]
        for field in (
            "neighbor_hypothesis_claimed",
            "cluster_claimed",
            "generator_family_claimed",
            "merge_or_compression_claimed",
            "long_context_attention_behavior_claimed",
        )
    ):
        raise ValueError("a forbidden scientific claim entered the setup")
    if value["attention_atlas_config"] != _atlas_config_payload():
        raise ValueError("attention atlas numerical config drifted")
    if value["attention_within_family_bootstrap_config"] != (
        _within_family_bootstrap_config()
    ):
        raise ValueError("attention bootstrap config drifted")
    if value["common_response_atlas_config"] != _atlas_config_payload():
        raise ValueError("common response atlas numerical config drifted")
    if value["common_response_within_family_bootstrap_config"] != (
        _common_response_bootstrap_config()
    ):
        raise ValueError("common response bootstrap config drifted")


def build_gemma3_typed_mode_response_setup() -> dict[str, object]:
    setup = _build_without_hash()
    _validate_built_setup(setup)
    setup["setup_sha256"] = _setup_sha256(setup)
    return setup


def validate_gemma3_typed_mode_response_setup(
    value: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("typed-mode response setup must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    if materialized != build_gemma3_typed_mode_response_setup():
        raise ValueError("typed-mode response setup is not canonical")


def load_gemma3_typed_mode_response_setup(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    validate_gemma3_typed_mode_response_setup(value)
    return value


def write_gemma3_typed_mode_response_setup(
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    target = Path(destination)
    setup = build_gemma3_typed_mode_response_setup()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(setup, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite a typed-mode response setup"
        ) from error
    return setup


@dataclass(frozen=True, slots=True)
class Gemma3TypedModeResponseSetup:
    """Typed wrapper around the immutable setup artifact."""

    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_gemma3_typed_mode_response_setup(self.state)

    @classmethod
    def build(cls) -> "Gemma3TypedModeResponseSetup":
        return cls(build_gemma3_typed_mode_response_setup())

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(self.state, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the existing canonical setup without writing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        load_gemma3_typed_mode_response_setup(args.output)
    else:
        write_gemma3_typed_mode_response_setup(args.output)
    return 0


__all__ = [
    "ATLAS_CONFIG",
    "ATTENTION_FULL_SEQUENCE_VALID_TOKEN_EXECUTION_COUNT",
    "ATTENTION_LOGICAL_MAC_ESTIMATE",
    "ATTENTION_MODE_IDS",
    "ATTENTION_NATIVE_ABI",
    "ATTENTION_NATIVE_PREFIX_CALL_COUNT",
    "ATTENTION_PROBE_COUNT",
    "ATTENTION_SELECTED_CANDIDATE_OUTPUT_COUNT",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DEFAULT_ATTENTION_REPORT_OUTPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_RESPONSE_REPORT_OUTPUT",
    "GEMMA3_TYPED_MODE_RESPONSE_SETUP_FORMAT_VERSION",
    "GEMMA3_TYPED_MODE_RESPONSE_SETUP_SCHEMA",
    "Gemma3TypedModeResponseSetup",
    "LEXICAL_EGRESS_NATIVE_ABI",
    "LEXICAL_INGRESS_NATIVE_ABI",
    "LEXICAL_MODE_IDS",
    "MLP_MODE_IDS",
    "MLP_NATIVE_ABI",
    "ORDERED_FAMILY_IDS",
    "PINNED_MODEL_ID",
    "PINNED_MODEL_REVISION",
    "PINNED_PANEL_FILE_SHA256",
    "PINNED_SETUP_FILE_SHA256",
    "PINNED_SETUP_SHA256",
    "PINNED_TYPED_MODE_RESPONSE_SETUP_FILE_SHA256",
    "PINNED_TYPED_MODE_RESPONSE_SETUP_SHA256",
    "RESPONSE_EXAMPLE_PAIR_STAT_COUNT",
    "RESPONSE_FAMILY_PAIR_STAT_COUNT",
    "RESPONSE_MODE_COUNT",
    "RESPONSE_MODE_IDS",
    "RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT",
    "RESPONSE_UNORDERED_PAIR_COUNT",
    "SETUP_ID",
    "TERMINAL_NORM_MODE_ID",
    "TERMINAL_NORM_NATIVE_ABI",
    "TOTAL_TYPED_MODE_COUNT",
    "build_gemma3_typed_mode_response_setup",
    "load_gemma3_typed_mode_response_setup",
    "main",
    "validate_gemma3_typed_mode_response_setup",
    "write_gemma3_typed_mode_response_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
