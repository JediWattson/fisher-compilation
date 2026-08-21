"""Freeze Gemma's confirmatory stable mutual-kNN graph protocol.

The spent typed-response atlas supplies fourteen discovery-only candidate
edges on thirty-seven exact response-equivalence classes.  This setup freezes
their identities and a fresh-data confirmation protocol before any successor
panel, tokenizer, or model weights are opened.  Exact equivalence classes are
atomic graph nodes; the tied lexical ingress/egress class therefore occupies
one neighbor slot and expands back to two logical modes only for reporting.

This module is standard-library-only and import inert.  Building, validating,
loading, or writing the setup does not open a predecessor or panel, tokenize
text, import a tensor library, run a model, fit a graph, or emit a scientific
result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


GEMMA3_STABLE_MUTUAL_KNN_SETUP_SCHEMA = (
    "fisher_graph.gemma3_stable_mutual_knn.setup.v1"
)
GEMMA3_STABLE_MUTUAL_KNN_SETUP_FORMAT_VERSION = 1
SETUP_ID = "gemma3-stable-mutual-knn-v1"
DEFAULT_OUTPUT = Path(
    "artifacts/research/gemma3-stable-mutual-knn-setup-v1.json"
)

# Filled only after the canonical artifact is generated.  They are public
# release pins, not inputs to the setup hash, so updating them cannot create a
# self-referential hash cycle.
PINNED_SETUP_SHA256 = (
    "feeb55f74882093fc29415621f9b5c99b45765efb3d3c72bb16dbbfc9debfc9a"
)
PINNED_SETUP_FILE_SHA256 = (
    "50f73436c34e3e85211e57090b6160a7c5ac936fc136c68fad1b890af716eb5c"
)
PINNED_CONFIRMATION_WORKER_SOURCE_PATH = Path(
    "src/fisher_graph/"
    "gemma3_stable_mutual_knn_confirmation_experiment.py"
)
# Patched to the frozen worker bytes after the worker-side provenance check is
# complete; unlike the setup's own public pins, this value is part of the
# authenticated setup payload.
PINNED_CONFIRMATION_WORKER_SOURCE_SHA256 = (
    "caf874c060058340e7eebfb95ce2e6266125b5737613fae769ad800b50d8b2a7"
)

_SETUP_HASH_DOMAIN = b"fisher_graph:gemma3-stable-mutual-knn-setup:v1\0"
_NODE_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-node-catalog:v1\0"
)
_CANDIDATE_CATALOG_HASH_DOMAIN = (
    b"fisher_graph.stable_mutual_knn_graph.candidate_catalog.v1\0"
)
_DISCOVERY_EVIDENCE_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-discovery-evidence:v1\0"
)
_DISCOVERY_GRAPH_HASH_DOMAIN = (
    b"fisher_graph:gemma3-stable-mutual-knn-discovery-graph:v1\0"
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

PINNED_TYPED_RESPONSE_SETUP_PATH = Path(
    "artifacts/research/gemma3-typed-mode-response-setup-v1.json"
)
PINNED_TYPED_RESPONSE_SETUP_SHA256 = (
    "189887111e12acd8e86bb56574ac69ebaf44660491a99a333ebcc75f7942de0f"
)
PINNED_TYPED_RESPONSE_SETUP_FILE_SHA256 = (
    "a821b963c491b710784f4bf7745321208dfee8ba23201f4ece4f017d1be00aa0"
)
PINNED_TYPED_RESPONSE_REPORT_PATH = Path(
    ".local-runs/google--gemma-3-270m/typed-mode-response-atlas-v1.json"
)
PINNED_TYPED_RESPONSE_REPORT_SHA256 = (
    "1ae47951899bf7f3ff4ddae4e3de69ce1bd5a798e2897ffe42931f3ed7005faa"
)
PINNED_TYPED_RESPONSE_REPORT_FILE_SHA256 = (
    "eeb4d735263309f9050c7ae3a61a49dfe9830e7b1d1ac23b47266a3cc3186996"
)
PINNED_MULTI_ABI_RECEIPT_SHA256 = (
    "502772681a731e08cc5b9bf6a3b5e4b2c66b258707d08c0338c9cd9fd19b4966"
)
PINNED_EQUIVALENCE_ATLAS_RECEIPT_SHA256 = (
    "4572fbd621eef5345788a96fcaa4b0ecec076f0ce15f735d157af47439c4e22a"
)
PINNED_INPUT_STATISTICS_SHA256 = (
    "f6d9a4d423627a5aa9a257ade07dc39767c7cf0bf34813468399aa512a2b5a0f"
)
PINNED_MEASUREMENT_INPUT_SHA256 = (
    "cae68825f4d8f81e5fe106495bcab2eaa8f0f167ccc40f3fe9daf9487e7cad9f"
)
PINNED_RESPONSE_ABI_SHA256 = (
    "a55b5fc1c50c7d435a8b26d22b0da81df32914685a2fff4f9b1ea69e6cd72367"
)
PINNED_RUNTIME_FISHER_WEIGHT_CATALOG_SHA256 = (
    "b99b42a17ccd3dcf4082ba8a8dba4ff943c65c3ef7456dd8e65440ff50b1b1b0"
)
PINNED_DISCOVERY_PANEL_PATH = Path(
    "examples/gemma3_downstream_retention_v1.json"
)
PINNED_DISCOVERY_PANEL_FILE_SHA256 = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PINNED_DISCOVERY_PANEL_ID = "gemma3-single-token-forced-choice-v1"

DISCOVERY_ORDERED_FAMILY_IDS = (
    "integer_addition",
    "integer_subtraction",
    "integer_comparison",
    "lexical_antonym",
    "category_completion",
    "world_capitals",
)
FRESH_ORDERED_FAMILY_IDS = (
    "C_confirmation.observational_diagnosis",
    "C_confirmation.quantitative_reasoning",
    "C_confirmation.technical_explanation",
    "C_confirmation.procedural_planning",
    "C_confirmation.formal_logic",
    "C_confirmation.causal_evidence_synthesis",
)
RESERVE_ORDERED_FAMILY_IDS = (
    "reserve.observational_diagnosis",
    "reserve.quantitative_reasoning",
    "reserve.technical_explanation",
    "reserve.procedural_planning",
    "reserve.formal_logic",
    "reserve.causal_evidence_synthesis",
)

GRAPH_NODE_COUNT = 37
LOGICAL_MODE_COUNT = 38
DISCOVERY_CANDIDATE_EDGE_COUNT = 14
MUTUAL_K = 2
RANK_BOUNDARY_TIE_ABSOLUTE_TOLERANCE = 1.0e-10
MAX_DIRECTIONAL_RELATIVE_SSE = 1.0
MAX_ABSOLUTE_WEIGHTED_RMSE = 0.125
BOOTSTRAP_SEED = 2_311
BOOTSTRAP_RESAMPLE_COUNT = 10_000
BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY = 0.80
BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION = 0.01
BOOTSTRAP_DRAW_PROTOCOL = "sha256_counter_modulo_example_count_v1"
BOOTSTRAP_DRAW_PROTOCOL_DOMAIN = (
    b"fisher_graph:gemma3-mode-atlas-bootstrap-draw:v1\0"
)
BOOTSTRAP_DRAW_PREIMAGE_ENCODING = (
    "domain||seed_u64be||resample_u64be||family_u32be||draw_u32be"
)
BOOTSTRAP_DRAW_DIGEST_SELECTION = (
    "first_8_sha256_bytes_u64be_mod_example_count"
)
BOOTSTRAP_CATALOG_DIGEST_DOMAIN = (
    b"fisher_graph.mode_similarity_atlas.bootstrap_catalog.v1\0"
)
BOOTSTRAP_CATALOG_DIGEST_ENCODING = (
    "sha256(catalog_domain||canonical_config||"
    "repeated(resample_u64be,family_u32be,draw_u32be,example_u32be))"
)
LOFO_REPLICATE_COUNT = 6
LOFO_MINIMUM_MUTUAL_COUNT = 5

FRESH_EXAMPLES_PER_FAMILY = 10
FRESH_SHORT_EXAMPLES_PER_FAMILY = 3
FRESH_BOUNDARY_EXAMPLES_PER_FAMILY = 4
FRESH_LONG_EXAMPLES_PER_FAMILY = 3
FRESH_EXAMPLE_COUNT = 60
RESERVE_EXAMPLES_PER_FAMILY = 2
RESERVE_EXAMPLE_COUNT = 12
FRESH_VALID_TOKEN_TARGETS = (32, 64, 96, 511, 512, 513, 514, 576, 640, 768)
SHORT_VALID_TOKEN_TARGETS = (32, 64, 96)
BOUNDARY_VALID_TOKEN_TARGETS = (511, 512, 513, 514)
LONG_VALID_TOKEN_TARGETS = (576, 640, 768)
LONG_VALID_TOKEN_MINIMUM = 576
LONG_VALID_TOKEN_MAXIMUM = 768

PINNED_DISCOVERY_RAW_PROMPT_CATALOG_SHA256 = (
    "a0df1c0d189562a01b8429a77a60a36dddcceb47f5d95e7a61199e718cf1222f"
)
PINNED_DISCOVERY_CANONICAL_PROMPT_CATALOG_SHA256 = (
    "07c629206de7c868b7789c9e739fca8392b19b9453ecfccbb9f4296120f75a1f"
)
PINNED_DISCOVERY_SEMANTIC_INSTANCE_CATALOG_SHA256 = (
    "5a31cfc3b5495af925b96f56a85a70ec1375d173f8875dde3067db1409bec6d2"
)
PINNED_DISCOVERY_TEMPLATE_SIGNATURE_CATALOG_SHA256 = (
    "7b91b2a14286dd5becf38715394b5b3dbc17a5057c5bcf38774a7d6ada9e2ea1"
)

# Canonical class order is the exact order authenticated by the predecessor.
_NODE_TUPLES = (
    ("response_class.0279b2a8f0931bdc17fdff17b5333cc3627c841f3f63a3cd07782d3437e29a29", ("mode.mutation.21",)),
    ("response_class.030e85671a21944e892509f687eaccb87cae5aa746730248526b43c6138a7a58", ("mode.mutation.15",)),
    ("response_class.18b04a725ceaaa654e60f4dfa45ade2b70b6eff0a9bf2405967ae93f230cd1fe", ("mode.mutation.30",)),
    ("response_class.21f6019b8b6571a8be1dcf1843170e69a618c29daff30e02dc4a4d569a316620", ("mode.mutation.22",)),
    ("response_class.281da9a967ba0eddbe88fae58d3593e63090e9036ba00ad0f94bb002d07feb63", ("mode.mutation.27",)),
    ("response_class.313860638f6f89edbc94a6e490b1bdf0d2c049729308407fbaf663fb1d554f27", ("mode.mutation.29",)),
    ("response_class.3215b347d3f8145cc6cd5d309b6b73c944670b621ebe3f52a881b714af461e40", ("mode.mutation.10",)),
    ("response_class.3bb9a9b465aa135c3d9f520aa89842b228f50c076d5d37f53b5f84d7298c5e73", ("mode.mutation.17",)),
    ("response_class.3cfa05a5cade7177cd9dd152f7317e0343dbf4a0cec23f86b290ad50d26d7ede", ("mode.mutation.13",)),
    ("response_class.3ebdd1d3925bda25afd09cb6494da7a49d186318f60cb496165662729b6f365c", ("mode.mutation.19",)),
    ("response_class.47e7d574473ec0a15e6df58ed83164db0d8edffab014884cb2eb1bbfbd51ae66", ("mode.mutation.8",)),
    ("response_class.4c7543a57d333b728c3cf83621dbba5ec7954bd19c4fe8d224c606ecdd99001d", ("mode.mutation.9",)),
    ("response_class.5fcc6e670d5d477ceb7c9b51adce9854397ba6b1bdd5810b58e58bb077573b16", ("mode.mutation.33",)),
    ("response_class.62bf485f7f88d09061a6c9e217d2c54e62df360ae5e71351ec18efe3b854c696", ("mode.mutation.35",)),
    ("response_class.64999e8947294bead3edb9e5afdafeb3deec4451a1a8377dd034c5850e322e6e", ("mode.mutation.1",)),
    ("response_class.67db4902796b165c67fbf840d2f5c70a778dd88a4929ee06431598014e22a4dc", ("mode.mutation.11",)),
    ("response_class.6ee61238904e8ad4cc64585e39d6a2e82d370fcb145678804f6fe759ae2f46fb", ("mode.mutation.28",)),
    ("response_class.7066d642ba230ebc5813d58e9b3aa409801c50ce3ef84df0896f59b0b97d557d", ("mode.mutation.7",)),
    ("response_class.78db0226061c87a1c7583aa73eb620fe7f5157169e97f25d9b58faaf323dbc7a", ("mode.mutation.12",)),
    ("response_class.7bda3f9efb0d7df269d74d6bf27b801bcb2b40eff6282bc7153247abbe51fc38", ("mode.mutation.6",)),
    ("response_class.7c64262e52b2d919cf223124c320fe352f6c396dbdf299f52b73ee0dacc68bb2", ("mode.mutation.20",)),
    ("response_class.86db586a5ac58bb844ad26393c3e853ae42d2c9d5d6a8b3c67e6f697f05f974a", ("mode.mutation.34",)),
    ("response_class.9878790727773903a7d6561dffa96dc61476edea1519c37ba21c812792a1400b", ("mode.mutation.2",)),
    ("response_class.b111c3a4516d6f21c9051cb92ebab69230a1fdbfc5dbdf41d1578d68304019a2", ("mode.mutation.36",)),
    ("response_class.b536e2db2cfc5117f9aaeb80e708bbfb1d68e8e990c5ed1afd10ccd2c0b340", ("mode.mutation.26",)),
    ("response_class.be7aae74b527794d493c67d8718290cda375055e279f72fbe160a0ee25bb8008", ("mode.mutation.32",)),
    ("response_class.c45b410145c1cee200f32b2e92cc89940e6a8e427bef765a36cc50918a8f4f32", ("mode.mutation.23",)),
    ("response_class.c76b0821ec24b40a43fefb422230fafca0e45fb066ac96c282197a71625de2d6", ("mode.mutation.24",)),
    ("response_class.c85512439f03a85572af0f7d9f5d14310ce5877cd1ad60fe0f5ca5dd4d456fc0", ("mode.mutation.31",)),
    ("response_class.d2c4ed41acd9890238b762276f6d3f4fe020918849a6545e30276eb7ae8b1e50", ("mode.mutation.18",)),
    ("response_class.d364d5410eceed7fbf3d5a4cadd4abf9717e65582ba7352fc91f620ae0c0a3e6", ("mode.mutation.14",)),
    ("response_class.d720edfaef364e73134049861dffaa9db2a911e08e2325abafb40b3387638c82", ("mode.mutation.25",)),
    ("response_class.e74b48dc469c2d658967958c7bef056cca82a7af1ba2845f46e37295555e1d91", ("mode.lexical.egress", "mode.lexical.ingress")),
    ("response_class.eb0e4812ee6f0bead5a403d3b7acadae7d5db8a8f708ba9eea2accaeb4e5ebb1", ("mode.mutation.5",)),
    ("response_class.f225f02c55bb5bbfb58f9b79a8e3a4925f5a24744ff08e925d17bb1ebdcc4ed9", ("mode.mutation.16",)),
    ("response_class.f7a5391b4cd047a4efdc05e21138531ee78f0b5dd3336650c0a60f2fc4eca26d", ("mode.mutation.4",)),
    ("response_class.fa8ee5755e59963e35b61f100a1d63320ee1fcdd4471b36432d80080210b714f", ("mode.mutation.3",)),
)

# mode_a, mode_b, symmetric distance, a->b relative SSE, b->a relative
# SSE, absolute Fisher-weighted RMSE, bootstrap mutual count, LOFO count.
_CANDIDATE_DATA = (
    ("mode.mutation.21", "mode.mutation.19", 0.9994406746300135, 0.919377302273929, 1.0795040469860981, 0.08281651576099282, 6315, 4),
    ("mode.mutation.21", "mode.mutation.18", 1.0906713153673586, 1.1872690840950557, 0.9940735466396614, 0.0922131140176002, 3655, 5),
    ("mode.mutation.15", "mode.mutation.33", 2.1399908362171622, 2.0204379994760737, 2.259543672958251, 0.0506755537486389, 2891, 3),
    ("mode.mutation.17", "mode.mutation.19", 0.9865616577794256, 0.9249753224808366, 1.0481479930780147, 0.07295875005645168, 7009, 5),
    ("mode.mutation.17", "mode.mutation.7", 1.1200344172740004, 1.3162343890696964, 0.9238344454783046, 0.09058264474727452, 4906, 2),
    ("mode.mutation.8", "mode.mutation.3", 1.2809238970735706, 2.0410470446508326, 0.5208007494963087, 0.48867159850937725, 5460, 2),
    ("mode.mutation.9", "mode.mutation.6", 0.45370462235807674, 0.5965177367603002, 0.31089150795585324, 0.1021818022967684, 7609, 5),
    ("mode.mutation.33", "mode.mutation.31", 1.9582825707557863, 2.2087261444083683, 1.7078389971032042, 0.05104194502015226, 3824, 3),
    ("mode.mutation.6", "mode.mutation.14", 0.5214591038643943, 0.26593512976754013, 0.7769830779612485, 0.0942355878213215, 3956, 2),
    ("mode.mutation.20", "mode.mutation.18", 1.0668071790664015, 0.5734112009095815, 1.5602031572232216, 0.09906232033079143, 4400, 4),
    ("mode.mutation.34", "mode.mutation.32", 1.7792829363949387, 1.5270843448895015, 2.0314815279003757, 0.1243398278839728, 3070, 3),
    ("mode.mutation.26", "mode.mutation.24", 2.0168747185531655, 2.487732295522701, 1.54601714158363, 0.12696489506776978, 3368, 3),
    ("mode.mutation.14", "mode.mutation.16", 0.20248495781919296, 0.23329272610883367, 0.17167718952955222, 0.04821618592346871, 9997, 6),
    ("mode.lexical.egress", "mode.mutation.3", 1.2485436711412734, 1.6679938882609378, 0.8290934540216089, 0.5317367830432929, 9155, 6),
)

_K1_MODE_EDGES = (
    ("mode.mutation.17", "mode.mutation.19"),
    ("mode.mutation.9", "mode.mutation.6"),
    ("mode.mutation.33", "mode.mutation.31"),
    ("mode.mutation.26", "mode.mutation.24"),
    ("mode.mutation.14", "mode.mutation.16"),
    ("mode.lexical.egress", "mode.mutation.3"),
)
_K2_MODE_EDGES = tuple((row[0], row[1]) for row in _CANDIDATE_DATA)
_K3_MODE_EDGES = _K2_MODE_EDGES + (
    ("mode.mutation.30", "mode.mutation.2"),
    ("mode.mutation.10", "mode.mutation.20"),
    ("mode.mutation.10", "mode.mutation.16"),
    ("mode.mutation.13", "mode.mutation.26"),
    ("mode.mutation.9", "mode.mutation.14"),
    ("mode.mutation.6", "mode.mutation.16"),
)


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


def _operator_kind(mode_id: str) -> str:
    if mode_id == "mode.lexical.ingress":
        return "lexical_ingress"
    if mode_id == "mode.lexical.egress":
        return "lexical_egress"
    ordinal = int(mode_id.rsplit(".", 1)[1])
    return "attention" if ordinal % 2 else "mlp"


def _block_ordinal(mode_id: str) -> int | None:
    if not mode_id.startswith("mode.mutation."):
        return None
    ordinal = int(mode_id.rsplit(".", 1)[1])
    return (ordinal - 1) // 2


def _node_catalog() -> list[dict[str, object]]:
    return [
        {
            "node_id": node_id,
            "member_mode_ids": list(member_ids),
            "member_count": len(member_ids),
            "representative_mode_id": member_ids[0],
            "member_operator_kinds": [
                _operator_kind(mode_id) for mode_id in member_ids
            ],
            "member_block_ordinals": [
                _block_ordinal(mode_id) for mode_id in member_ids
            ],
            "equivalence_definition": (
                "exact_zero_sse_and_identical_quadratics_on_every_row_v1"
            ),
            "native_ABI_equivalence_implied": False,
        }
        for node_id, member_ids in _NODE_TUPLES
    ]


def _member_to_node() -> dict[str, str]:
    return {
        mode_id: node_id
        for node_id, member_ids in _NODE_TUPLES
        for mode_id in member_ids
    }


def _edge_from_modes(mode_a_id: str, mode_b_id: str) -> list[str]:
    by_member = _member_to_node()
    return sorted((by_member[mode_a_id], by_member[mode_b_id]))


def _edge_catalog(mode_edges: Sequence[tuple[str, str]]) -> list[list[str]]:
    return sorted(_edge_from_modes(a, b) for a, b in mode_edges)


def _graph_summary(mode_edges: Sequence[tuple[str, str]]) -> dict[str, object]:
    node_ids = [node_id for node_id, _ in _NODE_TUPLES]
    edges = _edge_catalog(mode_edges)
    adjacency = {node_id: set() for node_id in node_ids}
    for node_a_id, node_b_id in edges:
        adjacency[node_a_id].add(node_b_id)
        adjacency[node_b_id].add(node_a_id)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node_id in node_ids:
        if node_id in seen or not adjacency[node_id]:
            continue
        pending = [node_id]
        seen.add(node_id)
        component: list[str] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda value: (-len(value), value))
    covered = sum(bool(adjacency[node_id]) for node_id in node_ids)
    return {
        "edge_count": len(edges),
        "covered_node_count": covered,
        "isolated_node_count": len(node_ids) - covered,
        "nonisolated_connected_component_count": len(components),
        "nonisolated_connected_component_sizes": [
            len(component) for component in components
        ],
        "maximum_degree": max(len(value) for value in adjacency.values()),
        "edge_catalog": edges,
        "edge_catalog_sha256": _sha256(_DISCOVERY_GRAPH_HASH_DOMAIN, edges),
    }


def _candidate_catalog() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for (
        mode_a_id,
        mode_b_id,
        symmetric_distance,
        relative_a_to_b,
        relative_b_to_a,
        absolute_rmse,
        bootstrap_count,
        lofo_count,
    ) in _CANDIDATE_DATA:
        node_a_id, node_b_id = _edge_from_modes(mode_a_id, mode_b_id)
        # The frozen rows were authored in canonical node order.  Fail loudly
        # if a future catalog edit reverses that relationship.
        if _member_to_node()[mode_a_id] != node_a_id:
            raise ValueError("candidate directional statistics changed order")
        candidates.append(
            {
                "node_a_id": node_a_id,
                "node_b_id": node_b_id,
                "member_label_a": mode_a_id,
                "member_label_b": mode_b_id,
                "discovery_symmetric_distance": symmetric_distance,
                "discovery_directed_relative_sse_a_to_b": relative_a_to_b,
                "discovery_directed_relative_sse_b_to_a": relative_b_to_a,
                "discovery_absolute_weighted_RMSE_a_to_b": absolute_rmse,
                "discovery_absolute_weighted_RMSE_b_to_a": absolute_rmse,
                "discovery_bootstrap_mutual_count": bootstrap_count,
                "discovery_bootstrap_mutual_frequency": (
                    bootstrap_count / BOOTSTRAP_RESAMPLE_COUNT
                ),
                "discovery_LOFO_mutual_count": lofo_count,
                "discovery_LOFO_mutual_frequency": (
                    lofo_count / LOFO_REPLICATE_COUNT
                ),
                "discovery_only_not_confirmed": True,
            }
        )
    return sorted(
        candidates,
        key=lambda row: (row["node_a_id"], row["node_b_id"]),
    )


def _build_without_hash() -> dict[str, object]:
    node_catalog = _node_catalog()
    candidates = _candidate_catalog()
    node_catalog_sha256 = _sha256(_NODE_CATALOG_HASH_DOMAIN, node_catalog)
    candidate_catalog_sha256 = _sha256(
        _CANDIDATE_CATALOG_HASH_DOMAIN, _edge_catalog(_K2_MODE_EDGES)
    )
    discovery_evidence_catalog_sha256 = _sha256(
        _DISCOVERY_EVIDENCE_CATALOG_HASH_DOMAIN, candidates
    )
    return {
        "schema": GEMMA3_STABLE_MUTUAL_KNN_SETUP_SCHEMA,
        "format_version": GEMMA3_STABLE_MUTUAL_KNN_SETUP_FORMAT_VERSION,
        "setup_id": SETUP_ID,
        "setup_receipt": {
            "predecessor_opened": False,
            "fresh_panel_opened": False,
            "tokenizer_loaded": False,
            "model_loaded": False,
            "model_forward_count": 0,
            "model_backward_count": 0,
            "graph_fitted": False,
            "scientific_result_emitted": False,
        },
        "confirmation_worker_source": {
            "path": str(PINNED_CONFIRMATION_WORKER_SOURCE_PATH),
            "sha256": PINNED_CONFIRMATION_WORKER_SOURCE_SHA256,
        },
        "model_freeze": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
            "source_model_fingerprint_sha256": (
                PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
            ),
            "source_execution_fingerprint_sha256": (
                PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            ),
            "source_parameters_frozen": True,
        },
        "exact_predecessor": {
            "typed_response_setup_path": str(PINNED_TYPED_RESPONSE_SETUP_PATH),
            "typed_response_setup_sha256": (
                PINNED_TYPED_RESPONSE_SETUP_SHA256
            ),
            "typed_response_setup_file_sha256": (
                PINNED_TYPED_RESPONSE_SETUP_FILE_SHA256
            ),
            "typed_response_report_path": str(
                PINNED_TYPED_RESPONSE_REPORT_PATH
            ),
            "typed_response_report_schema": (
                "fisher_graph.gemma3_typed_mode_response.experiment.v1"
            ),
            "typed_response_report_sha256": (
                PINNED_TYPED_RESPONSE_REPORT_SHA256
            ),
            "typed_response_report_file_sha256": (
                PINNED_TYPED_RESPONSE_REPORT_FILE_SHA256
            ),
            "multi_ABI_receipt_sha256": PINNED_MULTI_ABI_RECEIPT_SHA256,
            "equivalence_atlas_receipt_sha256": (
                PINNED_EQUIVALENCE_ATLAS_RECEIPT_SHA256
            ),
            "input_statistics_sha256": PINNED_INPUT_STATISTICS_SHA256,
            "measurement_input_sha256": PINNED_MEASUREMENT_INPUT_SHA256,
            "response_ABI_sha256": PINNED_RESPONSE_ABI_SHA256,
            "runtime_output_Fisher_weight_catalog_sha256": (
                PINNED_RUNTIME_FISHER_WEIGHT_CATALOG_SHA256
            ),
            "diagnostic_outcome": (
                "typed_mode_response_geometry_discovery_only"
            ),
            "all_predecessor_digests_required_before_fresh_panel_or_live_"
            "dependencies": True,
        },
        "node_ontology": {
            "graph_fit_unit": "exact_response_equivalence_class",
            "node_count": GRAPH_NODE_COUNT,
            "logical_mode_count": LOGICAL_MODE_COUNT,
            "logical_member_expansion_reporting_only": True,
            "exact_equivalence_class_is_atomic_for_ranking": True,
            "tied_lexical_class_member_mode_ids": [
                "mode.lexical.egress",
                "mode.lexical.ingress",
            ],
            "tied_lexical_class_consumes_one_neighbor_slot": True,
            "predecessor_37_node_universe_is_immutable_for_fresh_graph": True,
            "fresh_tied_lexical_ingress_egress_exact_equality_must_be_"
            "reproved": True,
            "failed_fresh_tied_lexical_equality_invalidates_the_run": True,
            "new_fresh_exact_equalities_may_not_collapse_nodes_or_change_"
            "ranking_universe": True,
            "new_fresh_exact_equalities_are_diagnostic_only": True,
            "response_equivalence_does_not_imply_native_ABI_equivalence": True,
            "operator_kind_and_block_ordinal_are_provenance_only": True,
            "operator_kind_or_block_ordinal_used_as_graph_features": False,
            "ordered_node_catalog": node_catalog,
            "node_catalog_sha256": node_catalog_sha256,
        },
        "discovery_k_sweep": {
            "source": "spent_exact_response_equivalence_class_atlas",
            "metric": (
                "equal_family_mean_bidirectional_relative_SSE_symmetric_"
                "distance"
            ),
            "tie_policy": (
                "fail_closed_when_kth_and_kplus1_distances_are_within_"
                "absolute_tolerance"
            ),
            "rank_boundary_tie_absolute_tolerance": (
                RANK_BOUNDARY_TIE_ABSOLUTE_TOLERANCE
            ),
            "k_1": _graph_summary(_K1_MODE_EDGES),
            "k_2": _graph_summary(_K2_MODE_EDGES),
            "k_3": _graph_summary(_K3_MODE_EDGES),
            "selected_k": MUTUAL_K,
            "selection_was_frozen_before_fresh_panel": True,
            "selection_reason": (
                "k1_leaves_25_of_37_nodes_isolated_while_k3_already_has_an_"
                "11_node_component;_k2_is_the_last_sparse_middle_rung"
            ),
            "k_selection_may_not_be_retuned_on_fresh_data": True,
        },
        "discovery_candidate_graph": {
            "candidate_edge_count": DISCOVERY_CANDIDATE_EDGE_COUNT,
            "candidate_edges": candidates,
            "candidate_catalog_sha256": candidate_catalog_sha256,
            "candidate_catalog_hash_payload": (
                "canonical_sorted_JSON_list_of_lexically_canonical_node_"
                "ID_pairs_only"
            ),
            "discovery_evidence_catalog_sha256": (
                discovery_evidence_catalog_sha256
            ),
            "candidate_universe_is_exactly_discovery_mutual_k2": True,
            "all_14_edges_retained_irrespective_of_discovery_stability_or_"
            "error_gates": True,
            "discovery_bootstrap_protocol": (
                "within_family_example_cluster_resampling_10000_seed2311"
            ),
            "ordered_discovery_family_ids": list(
                DISCOVERY_ORDERED_FAMILY_IDS
            ),
            "discovery_LOFO_protocol": "omit_each_of_six_families_once",
            "discovery_values_are_descriptive_and_spent": True,
            "no_discovery_edge_is_confirmed": True,
        },
        "ranking_protocol": {
            "k": MUTUAL_K,
            "rank_metric": "symmetric_distance",
            "symmetric_distance_formula": (
                "0.5*(directed_relative_SSE_a_to_b+"
                "directed_relative_SSE_b_to_a)"
            ),
            "directed_relative_SSE_formula": (
                "equal_family_mean(weighted_SSE_f/"
                "max(source_energy_f,1e-12))"
            ),
            "pool_family_sufficient_statistics_before_each_family_ratio": (
                True
            ),
            "rank_boundary_tie_absolute_tolerance": (
                RANK_BOUNDARY_TIE_ABSOLUTE_TOLERANCE
            ),
            "rank_boundary_tie_policy": (
                "fail_closed_undefined_node_neighbors"
            ),
            "mode_or_node_label_tie_break_forbidden": True,
            "mutual_definition": (
                "a_is_in_exact_top2_of_b_and_b_is_in_exact_top2_of_a"
            ),
            "equivalence_classes_collapsed_before_ranking": True,
        },
        "fresh_confirmation_gates": {
            "point_mutual_k2_required": True,
            "point_error_gates": {
                "both_directions_required": True,
                "maximum_directional_relative_SSE": (
                    MAX_DIRECTIONAL_RELATIVE_SSE
                ),
                "absolute_weighted_RMSE_formula": (
                    "sqrt(equal_family_mean(weighted_SSE_f/"
                    "coordinate_row_count_f))"
                ),
                "absolute_weighted_RMSE_units": (
                    "Fisher_weighted_centered_restricted_logit"
                ),
                "maximum_absolute_weighted_RMSE": (
                    MAX_ABSOLUTE_WEIGHTED_RMSE
                ),
                "both_absolute_directions_recorded_even_though_primary_SSE_"
                "is_symmetric": True,
                "asymmetry_ratio_formula": (
                    "max(relative_a_to_b,relative_b_to_a)/"
                    "min(relative_a_to_b,relative_b_to_a)"
                ),
                "asymmetry_ratio_both_zero_value": 1,
                "asymmetry_ratio_one_zero_value": "positive_infinity",
                "asymmetry_ratio_is_diagnostic_only": True,
            },
            "within_family_bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "resample_count": BOOTSTRAP_RESAMPLE_COUNT,
                "sampling_unit": (
                    "example_plus_centered_logits4_response_cluster"
                ),
                "draw_protocol": BOOTSTRAP_DRAW_PROTOCOL,
                "draw_protocol_domain_hex": (
                    BOOTSTRAP_DRAW_PROTOCOL_DOMAIN.hex()
                ),
                "draw_preimage_encoding": (
                    BOOTSTRAP_DRAW_PREIMAGE_ENCODING
                ),
                "draw_digest_selection": BOOTSTRAP_DRAW_DIGEST_SELECTION,
                "draw_catalog_digest_domain_hex": (
                    BOOTSTRAP_CATALOG_DIGEST_DOMAIN.hex()
                ),
                "draw_catalog_digest_encoding": (
                    BOOTSTRAP_CATALOG_DIGEST_ENCODING
                ),
                "draw_catalog_iteration_order": (
                    "resample_then_configured_family_then_draw"
                ),
                "draws_per_family": FRESH_EXAMPLES_PER_FAMILY,
                "with_replacement": True,
                "mutual_k2_topology_recomputed_per_resample": True,
                "point_error_gates_recomputed_per_resample": False,
                "minimum_mutual_frequency": (
                    BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY
                ),
                "maximum_degenerate_fraction": (
                    BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION
                ),
                "frequency_denominator": "valid_resamples_only",
            },
            "leave_one_family_out": {
                "replicate_count": LOFO_REPLICATE_COUNT,
                "ordered_omitted_family_ids": list(
                    FRESH_ORDERED_FAMILY_IDS
                ),
                "mutual_k2_topology_recomputed_per_replicate": True,
                "point_error_gates_recomputed_per_replicate": False,
                "minimum_mutual_count": LOFO_MINIMUM_MUTUAL_COUNT,
                "minimum_mutual_frequency": (
                    LOFO_MINIMUM_MUTUAL_COUNT / LOFO_REPLICATE_COUNT
                ),
            },
            "long_context_robustness_veto": {
                "applies_to_every_candidate_edge": True,
                "long_stratum_point_mutual_k2_required": True,
                "same_bidirectional_relative_and_absolute_error_gates_"
                "required": True,
                "failure_vetoes_confirmation": True,
                "long_stratum_bootstrap_or_LOFO_claim_required": False,
            },
            "confirmation_conjunction": [
                "edge_is_in_frozen_14_edge_candidate_catalog",
                "fresh_pooled_point_mutual_k2",
                "fresh_pooled_both_directional_relative_SSE_at_most_1",
                "fresh_pooled_both_absolute_weighted_RMSE_at_most_0.125",
                "fresh_bootstrap_mutual_frequency_at_least_0.80",
                "fresh_LOFO_mutual_count_at_least_5_of_6",
                "fresh_long_stratum_point_mutual_k2",
                "fresh_long_stratum_same_point_error_gates",
            ],
            "partial_gate_pass_never_confirms_an_edge": True,
            "thresholds_may_not_be_retuned_after_fresh_data_access": True,
        },
            "fresh_panel_contract": {
            "panel_not_authored_by_this_setup": True,
            "manifest_schema": (
                "fisher_graph.gemma3_stable_mutual_knn_confirmation_panel"
            ),
            "manifest_format_version": 1,
            "panel_id": "gemma3_stable_mutual_knn_confirmation_v1",
            "corpus_schema": (
                "fisher_graph.gemma3_stable_mutual_knn_confirmation_corpus"
            ),
            "corpus_format_version": 1,
            "default_corpus_path": (
                "examples/gemma3_stable_mutual_knn_confirmation_corpus.json"
            ),
            "default_manifest_path": (
                "examples/gemma3_stable_mutual_knn_confirmation_panel.json"
            ),
            "default_audit_path": (
                "artifacts/research/gemma3-stable-mutual-knn-confirmation-"
                "panel-audit-v1.json"
            ),
            "corpus_top_level_roles_exactly": [
                "C_confirmation",
                "reserve",
            ],
            "manifest_role_keys_exactly": [
                "C_confirmation",
                "reserve",
            ],
            "panel_id_must_differ_from_discovery_panel_id": True,
            "ordered_family_ids": list(FRESH_ORDERED_FAMILY_IDS),
            "ordered_reserve_family_ids": list(RESERVE_ORDERED_FAMILY_IDS),
            "fresh_semantic_families_are_disjoint_from_discovery_families": (
                True
            ),
            "example_count": FRESH_EXAMPLE_COUNT,
            "examples_per_family": FRESH_EXAMPLES_PER_FAMILY,
            "short_examples_per_family": FRESH_SHORT_EXAMPLES_PER_FAMILY,
            "boundary_examples_per_family": (
                FRESH_BOUNDARY_EXAMPLES_PER_FAMILY
            ),
            "long_examples_per_family": FRESH_LONG_EXAMPLES_PER_FAMILY,
            "exact_valid_prompt_token_targets_per_family": list(
                FRESH_VALID_TOKEN_TARGETS
            ),
            "canonical_confirmation_stratum_ids": [
                "short_32",
                "short_64",
                "short_96",
                "boundary_511",
                "boundary_512",
                "transition_513",
                "transition_514",
                "outside_window_576",
                "outside_window_640",
                "outside_window_768",
            ],
            "context_strata": {
                "short": {
                    "exact_valid_prompt_token_targets": list(
                        SHORT_VALID_TOKEN_TARGETS
                    ),
                    "global_example_count": 18,
                    "role": "diagnostic_control",
                },
                "boundary": {
                    "exact_valid_prompt_token_targets": list(
                        BOUNDARY_VALID_TOKEN_TARGETS
                    ),
                    "global_example_count": 24,
                    "role": "diagnostic_window_transition",
                    "token_513_is_diagnostic_only_under_BOS_semantics": True,
                    "token_514_is_diagnostic_only_and_carries_no_long_"
                    "context_evidence_claim": True,
                },
                "long": {
                    "exact_valid_prompt_token_targets": list(
                        LONG_VALID_TOKEN_TARGETS
                    ),
                    "global_example_count": 18,
                    "role": "hard_robustness_veto",
                    "crosses_attention_sliding_window_cutoff_512": True,
                    "minimum_valid_prompt_tokens": LONG_VALID_TOKEN_MINIMUM,
                    "full_minimal_sufficient_evidence_span_outside_final_"
                    "512_token_window_required": True,
                    "visible_suffix_must_be_answer_neutral": True,
                    "answer_relevant_evidence_outside_final_512_token_window_"
                    "required": True,
                    "evidence_span_token_indices_required": True,
                    "last_query_token_index_minus_evidence_end_at_least": 512,
                },
            },
            "stratum_outputs_required": [
                "pooled",
                "short",
                "boundary",
                "long",
            ],
            "pooled_is_primary": True,
            "short_and_boundary_are_diagnostic_only": True,
            "row_manifest_required_fields": [
                "prompt_id",
                "corpus_index",
                "choices",
                "correct_choice",
                "template_id",
                "instance_id",
                "domain_template",
                "kind",
                "stratum_id",
                "text_sha256",
                "target_valid_tokens",
                "input_ids_sha256",
                "choices_sha256",
                "choice_token_ids",
                "answer_evidence",
                "canonical_sha256",
                "semantic_lineage_id",
                "paraphrase_lineage_id",
                "template_signature_sha256",
            ],
            "choice_count": 4,
            "all_choices_have_one_leading_ASCII_space": True,
            "all_choices_single_token_under_pinned_tokenizer": True,
            "all_four_choice_token_ids_distinct_per_example": True,
            "same_finite_response_and_Fisher_weighting_ABI_as_predecessor": (
                True
            ),
            "fresh_panel_file_SHA256_frozen_into_execution_manifest_before_"
            "model_weights_load": True,
            "choice_tokenization_and_prompt_token_count_catalogs_frozen_"
            "before_model_weights_load": True,
            "all_60_confirmation_input_ids_attention_masks_and_valid_lengths_"
            "authenticated_before_model_weights_load": True,
            "reserve_contract": {
                "example_count": RESERVE_EXAMPLE_COUNT,
                "examples_per_family": RESERVE_EXAMPLES_PER_FAMILY,
                "ordered_family_ids": list(RESERVE_ORDERED_FAMILY_IDS),
                "same_overlap_and_manifest_audit_required": True,
                "parse_and_hash_audit_only": True,
                "executable_stratum_forbidden": True,
                "target_valid_tokens_must_be_null": True,
                "input_ids_sha256_must_be_null": True,
                "choice_token_ids_must_be_null": True,
                "choices_choices_sha256_and_correct_choice_required": True,
                "tokenization_receipt_forbidden": True,
                "tokenizer_access_forbidden": True,
                "model_access_forbidden": True,
                "unreachable_by_primary_experiment": True,
                "future_use_requires_a_new_frozen_protocol": True,
            },
        },
        "discovery_data_firewall": {
            "discovery_panel_path": str(PINNED_DISCOVERY_PANEL_PATH),
            "discovery_panel_id": PINNED_DISCOVERY_PANEL_ID,
            "discovery_panel_file_sha256": (
                PINNED_DISCOVERY_PANEL_FILE_SHA256
            ),
            "discovery_example_count": 60,
            "all_discovery_rows_spent": True,
            "raw_prompt_catalog_sha256": (
                PINNED_DISCOVERY_RAW_PROMPT_CATALOG_SHA256
            ),
            "canonical_prompt_catalog_sha256": (
                PINNED_DISCOVERY_CANONICAL_PROMPT_CATALOG_SHA256
            ),
            "semantic_instance_catalog_sha256": (
                PINNED_DISCOVERY_SEMANTIC_INSTANCE_CATALOG_SHA256
            ),
            "template_signature_catalog_sha256": (
                PINNED_DISCOVERY_TEMPLATE_SIGNATURE_CATALOG_SHA256
            ),
            "normalization": (
                "Unicode_NFKC_then_casefold_then_join_regex_[a-z0-9]+_"
                "tokens_with_single_ASCII_spaces"
            ),
            "semantic_instance_definition": (
                "family_specific_canonical_task_arguments_independent_of_"
                "surface_wording"
            ),
            "template_signature_definition": (
                "normalized_prompt_with_family_specific_semantic_slots_"
                "replaced_by_typed_placeholders"
            ),
            "required_zero_intersections": [
                "family_id",
                "text_sha256",
                "canonical_sha256",
                "instance_id",
                "template_id",
                "domain_template",
                "semantic_lineage_id",
                "paraphrase_lineage_id",
                "template_signature_sha256",
            ],
            "instance_id_is_canonical_semantic_task_key_not_opaque_nonce": (
                True
            ),
            "instance_id_disjointness_is_the_paraphrase_firewall": True,
            "template_id_and_domain_template_are_content_derived_not_"
            "arbitrary_nonces": True,
            "model_free_paraphrase_overlap_check_required": True,
            "ordered_source_reference_inventory_paths_file_SHA256s_counts_"
            "and_catalog_hashes_required": True,
            "all_duplicate_and_overlap_counts_must_be_zero": True,
            "fresh_overlap_audit_receipt_required_before_tokenization": True,
            "fresh_token_count_and_long_evidence_audit_required_before_"
            "model_weights_load": True,
            "any_overlap_or_missing_lineage_metadata_is_fail_closed": True,
        },
        "graph_extraction_and_reporting": {
            "surviving_edge_set": (
                "only_frozen_candidates_passing_every_confirmation_gate"
            ),
            "connected_components_reported": True,
            "connected_components_role": "diagnostic_topology_only",
            "connected_components_are_not_clusters": True,
            "maximal_cliques_reported": True,
            "maximal_clique_minimum_node_count": 2,
            "maximal_cliques_role": "stable_neighbor_groups_only",
            "maximal_cliques_may_overlap": True,
            "overlapping_cliques_must_not_be_forced_into_a_partition": True,
            "singletons_reported": True,
            "fresh_non_candidate_mutual_edges_reported": True,
            "fresh_non_candidate_mutual_edges_role": "exploratory_only",
            "new_edges_cannot_enter_confirmatory_graph": True,
            "operator_kind_reported_post_hoc_only": True,
        },
        "claim_taxonomy": {
            "allowed_after_all_gates": {
                "confirmed_neighborhood_edge": True,
                "stable_neighbor_group_for_maximal_clique": True,
                "response_geometry_only": True,
            },
            "diagnostic_only": {
                "connected_component": True,
                "fresh_non_candidate_mutual_edge": True,
                "operator_kind_or_depth_enrichment": True,
                "edge_asymmetry_ratio": True,
            },
            "never_authorized_by_this_protocol": {
                "cluster": True,
                "generator_family": True,
                "native_ABI_interchangeability": True,
                "causal_interchangeability": True,
                "merge": True,
                "compression": True,
                "deployment": True,
                "parameter_MAC_byte_memory_or_latency_reduction": True,
            },
            "later_compression_requires_joint_composition_and_quality_"
            "experiments": True,
        },
        "execution_order": [
            "strict_load_this_setup_and_authenticate_every_predecessor_digest",
            "strict_load_and_hash_fresh_panel_without_model_weights",
            "audit_all_overlap_semantic_template_and_lineage_firewalls",
            "load_only_pinned_tokenizer_and_freeze_choice_token_and_prompt_"
            "length_catalogs",
            "audit_short_long_counts_and_distant_evidence_spans",
            "only_then_load_pinned_model_weights",
            "measure_exact_38_logical_responses_and_37_atomic_graph_nodes",
            "evaluate_only_the_frozen_14_edges_confirmatorily",
            "report_components_cliques_singletons_and_exploratory_new_edges_"
            "under_the_claim_taxonomy",
        ],
        "claim_boundary": {
            "setup_only": True,
            "fresh_panel_exists": False,
            "fresh_measurement_exists": False,
            "confirmed_neighborhood_edge_claimed": False,
            "stable_neighbor_group_claimed": False,
            "cluster_claimed": False,
            "generator_family_claimed": False,
            "merge_or_compression_claimed": False,
            "long_context_result_claimed": False,
            "deployment_claimed": False,
        },
    }


def _validate_built_setup(value: Mapping[str, object]) -> None:
    if value["schema"] != GEMMA3_STABLE_MUTUAL_KNN_SETUP_SCHEMA:
        raise ValueError("stable mutual-kNN setup schema drifted")
    if value["confirmation_worker_source"] != {
        "path": str(PINNED_CONFIRMATION_WORKER_SOURCE_PATH),
        "sha256": PINNED_CONFIRMATION_WORKER_SOURCE_SHA256,
    }:
        raise ValueError("confirmation worker source binding drifted")
    ontology = value["node_ontology"]
    nodes = ontology["ordered_node_catalog"]
    if len(nodes) != GRAPH_NODE_COUNT:
        raise ValueError("stable mutual-kNN node count drifted")
    if len({node["node_id"] for node in nodes}) != GRAPH_NODE_COUNT:
        raise ValueError("stable mutual-kNN node identities are not unique")
    logical_modes = [
        mode_id for node in nodes for mode_id in node["member_mode_ids"]
    ]
    if len(logical_modes) != LOGICAL_MODE_COUNT or len(set(logical_modes)) != 38:
        raise ValueError("logical mode expansion drifted")
    lexical = [node for node in nodes if node["member_count"] == 2]
    if len(lexical) != 1 or lexical[0]["member_mode_ids"] != [
        "mode.lexical.egress",
        "mode.lexical.ingress",
    ]:
        raise ValueError("tied lexical atomic node drifted")
    if ontology["node_catalog_sha256"] != _sha256(
        _NODE_CATALOG_HASH_DOMAIN, nodes
    ):
        raise ValueError("stable mutual-kNN node catalog digest drifted")
    sweep = value["discovery_k_sweep"]
    expected_summaries = {
        "k_1": (6, 12, 25, [2, 2, 2, 2, 2, 2], 1),
        "k_2": (14, 20, 17, [6, 4, 3, 3, 2, 2], 2),
        "k_3": (20, 24, 13, [11, 3, 3, 3, 2, 2], 3),
    }
    for key, expected in expected_summaries.items():
        summary = sweep[key]
        actual = (
            summary["edge_count"],
            summary["covered_node_count"],
            summary["isolated_node_count"],
            summary["nonisolated_connected_component_sizes"],
            summary["maximum_degree"],
        )
        if actual != expected:
            raise ValueError(f"discovery {key} graph summary drifted")
    if sweep["selected_k"] != MUTUAL_K:
        raise ValueError("selected mutual-k value drifted")
    candidates = value["discovery_candidate_graph"]["candidate_edges"]
    if len(candidates) != DISCOVERY_CANDIDATE_EDGE_COUNT:
        raise ValueError("discovery candidate edge count drifted")
    candidate_pairs = sorted(
        [row["node_a_id"], row["node_b_id"]] for row in candidates
    )
    if candidate_pairs != sweep["k_2"]["edge_catalog"]:
        raise ValueError("candidate catalog is not exactly discovery mutual-k2")
    if value["discovery_candidate_graph"][
        "candidate_catalog_sha256"
    ] != _sha256(_CANDIDATE_CATALOG_HASH_DOMAIN, candidate_pairs):
        raise ValueError("discovery candidate catalog digest drifted")
    if value["discovery_candidate_graph"][
        "discovery_evidence_catalog_sha256"
    ] != _sha256(_DISCOVERY_EVIDENCE_CATALOG_HASH_DOMAIN, candidates):
        raise ValueError("discovery evidence catalog digest drifted")
    ranking = value["ranking_protocol"]
    if ranking["k"] != 2 or ranking["rank_boundary_tie_policy"] != (
        "fail_closed_undefined_node_neighbors"
    ):
        raise ValueError("mutual-kNN ranking contract drifted")
    gates = value["fresh_confirmation_gates"]
    if gates["point_error_gates"]["maximum_directional_relative_SSE"] != 1.0:
        raise ValueError("directional relative-SSE gate drifted")
    if gates["point_error_gates"]["maximum_absolute_weighted_RMSE"] != 0.125:
        raise ValueError("absolute weighted-RMSE gate drifted")
    if gates["within_family_bootstrap"]["minimum_mutual_frequency"] != 0.80:
        raise ValueError("bootstrap mutuality threshold drifted")
    bootstrap = gates["within_family_bootstrap"]
    expected_bootstrap_protocol = {
        "draw_catalog_digest_domain_hex": (
            BOOTSTRAP_CATALOG_DIGEST_DOMAIN.hex()
        ),
        "draw_catalog_digest_encoding": (
            BOOTSTRAP_CATALOG_DIGEST_ENCODING
        ),
        "draw_catalog_iteration_order": (
            "resample_then_configured_family_then_draw"
        ),
        "draw_digest_selection": BOOTSTRAP_DRAW_DIGEST_SELECTION,
        "draw_preimage_encoding": BOOTSTRAP_DRAW_PREIMAGE_ENCODING,
        "draw_protocol": BOOTSTRAP_DRAW_PROTOCOL,
        "draw_protocol_domain_hex": BOOTSTRAP_DRAW_PROTOCOL_DOMAIN.hex(),
    }
    if any(
        bootstrap.get(field) != expected
        for field, expected in expected_bootstrap_protocol.items()
    ):
        raise ValueError("bootstrap draw protocol drifted")
    if gates["leave_one_family_out"]["minimum_mutual_count"] != 5:
        raise ValueError("leave-one-family-out threshold drifted")
    panel = value["fresh_panel_contract"]
    if panel["example_count"] != 60:
        raise ValueError("fresh panel size drifted")
    if panel["exact_valid_prompt_token_targets_per_family"] != [
        32,
        64,
        96,
        511,
        512,
        513,
        514,
        576,
        640,
        768,
    ]:
        raise ValueError("fresh prompt token schedule drifted")
    if min(panel["context_strata"]["long"][
        "exact_valid_prompt_token_targets"
    ]) <= 513:
        raise ValueError("long-context stratum does not cross the cutoff")
    if not value["discovery_data_firewall"][
        "any_overlap_or_missing_lineage_metadata_is_fail_closed"
    ]:
        raise ValueError("fresh-data firewall stopped failing closed")
    if value["graph_extraction_and_reporting"][
        "connected_components_are_not_clusters"
    ] is not True:
        raise ValueError("components were promoted to clusters")
    forbidden = value["claim_taxonomy"]["never_authorized_by_this_protocol"]
    if not all(forbidden.values()):
        raise ValueError("a forbidden downstream claim was authorized")
    claims = value["claim_boundary"]
    if any(
        claims[field]
        for field in (
            "confirmed_neighborhood_edge_claimed",
            "stable_neighbor_group_claimed",
            "cluster_claimed",
            "generator_family_claimed",
            "merge_or_compression_claimed",
            "long_context_result_claimed",
            "deployment_claimed",
        )
    ):
        raise ValueError("a scientific result entered the setup artifact")


def build_gemma3_stable_mutual_knn_setup() -> dict[str, object]:
    setup = _build_without_hash()
    _validate_built_setup(setup)
    setup["setup_sha256"] = _setup_sha256(setup)
    return setup


def validate_gemma3_stable_mutual_knn_setup(
    value: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("stable mutual-kNN setup must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    if materialized != build_gemma3_stable_mutual_knn_setup():
        raise ValueError("stable mutual-kNN setup is not canonical")


def load_gemma3_stable_mutual_knn_setup(
    source: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    validate_gemma3_stable_mutual_knn_setup(value)
    return value


def write_gemma3_stable_mutual_knn_setup(
    destination: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    target = Path(destination)
    setup = build_gemma3_stable_mutual_knn_setup()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(setup, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to overwrite a stable mutual-kNN setup"
        ) from error
    return setup


@dataclass(frozen=True, slots=True)
class Gemma3StableMutualKNNSetup:
    """Typed wrapper around the immutable successor setup artifact."""

    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_gemma3_stable_mutual_knn_setup(self.state)

    @classmethod
    def build(cls) -> "Gemma3StableMutualKNNSetup":
        return cls(build_gemma3_stable_mutual_knn_setup())

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
        load_gemma3_stable_mutual_knn_setup(args.output)
    else:
        write_gemma3_stable_mutual_knn_setup(args.output)
    return 0


__all__ = [
    "BOOTSTRAP_CATALOG_DIGEST_DOMAIN",
    "BOOTSTRAP_CATALOG_DIGEST_ENCODING",
    "BOOTSTRAP_DRAW_DIGEST_SELECTION",
    "BOOTSTRAP_DRAW_PREIMAGE_ENCODING",
    "BOOTSTRAP_DRAW_PROTOCOL",
    "BOOTSTRAP_DRAW_PROTOCOL_DOMAIN",
    "BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY",
    "BOOTSTRAP_RESAMPLE_COUNT",
    "DEFAULT_OUTPUT",
    "DISCOVERY_CANDIDATE_EDGE_COUNT",
    "FRESH_EXAMPLE_COUNT",
    "GEMMA3_STABLE_MUTUAL_KNN_SETUP_FORMAT_VERSION",
    "GEMMA3_STABLE_MUTUAL_KNN_SETUP_SCHEMA",
    "GRAPH_NODE_COUNT",
    "Gemma3StableMutualKNNSetup",
    "LOFO_MINIMUM_MUTUAL_COUNT",
    "LOGICAL_MODE_COUNT",
    "LONG_VALID_TOKEN_MINIMUM",
    "MAX_ABSOLUTE_WEIGHTED_RMSE",
    "MAX_DIRECTIONAL_RELATIVE_SSE",
    "MUTUAL_K",
    "PINNED_SETUP_FILE_SHA256",
    "PINNED_SETUP_SHA256",
    "PINNED_CONFIRMATION_WORKER_SOURCE_PATH",
    "PINNED_CONFIRMATION_WORKER_SOURCE_SHA256",
    "RANK_BOUNDARY_TIE_ABSOLUTE_TOLERANCE",
    "SETUP_ID",
    "build_gemma3_stable_mutual_knn_setup",
    "load_gemma3_stable_mutual_knn_setup",
    "main",
    "validate_gemma3_stable_mutual_knn_setup",
    "write_gemma3_stable_mutual_knn_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
