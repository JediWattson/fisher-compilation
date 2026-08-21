"""Freeze the chart-native Gemma mode-neighbor protocol.

This setup makes shared Fisher/activation-chart geometry the primary neighbor
measurement.  Whole-model response ablations are deliberately downstream
validation only.  The module is standard-library-only and import inert: it
does not open prompts or result reports, import a tensor library, load a
tokenizer or model, fit charts, or claim a scientific result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence


GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.setup.v1"
)
GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_FORMAT_VERSION = 1
SETUP_ID = "gemma3-chart-native-neighbor-v1"
DEFAULT_OUTPUT = Path(
    "artifacts/research/gemma3-chart-native-neighbor-setup-v1.json"
)

PINNED_SETUP_SHA256 = (
    "5ebc02aa5800add5d16f9b3f506224029adcf1fa257ac80a7d46155ce0a1e484"
)
PINNED_SETUP_FILE_SHA256 = (
    "d597c9210f0fddae71a514b1a1738612eda22e99978f94d109ef61b489fee042"
)

_SETUP_HASH_DOMAIN = b"fisher_graph:gemma3-chart-native-neighbor-setup:v1\0"
_EVENT_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-events:v1\0"
)
_SITE_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-sites:v1\0"
)
_MODE_CATALOG_HASH_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-modes:v1\0"
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

PINNED_EVENT_EXECUTOR_SOURCE_PATH = Path(
    "src/fisher_graph/gemma3_joint_cross_block_chart_executor.py"
)
PINNED_EVENT_EXECUTOR_SOURCE_SHA256 = (
    "d7c397a694abb3d22b1f1289d7b79995cbf652bb824dc45a838b23ee4e847324"
)
PINNED_UNIFIED_CHART_SIDECAR_SOURCE_PATH = Path(
    "src/fisher_graph/gemma3_unified_event_chart_sidecar.py"
)
PINNED_UNIFIED_CHART_SIDECAR_SOURCE_SHA256 = (
    "ac0438c992d8909a67877a20756ce87c7021283cceb5f6067802f51b8c6fc0b4"
)
PINNED_TRANSPORT_PROTOCOL_PATH = Path(
    "artifacts/research/gemma3-cross-layer-chart-transport-protocol-v1.json"
)
PINNED_TRANSPORT_PROTOCOL_FILE_SHA256 = (
    "d01c1803e9b5eed28efa732b252a271bd12bbe80434194f306c288aa8ccaf1cf"
)
PINNED_TRANSPORT_PROTOCOL_SHA256 = (
    "b373f855076c73aceb9fea9adf404a60cf7406718e4b7a3c96dc295b8a64ab5b"
)
PINNED_EXACT_EDGE_PROTOCOL_PATH = Path(
    "artifacts/research/gemma3-exact-edge-sidecar-protocol-v1.json"
)
PINNED_EXACT_EDGE_PROTOCOL_FILE_SHA256 = (
    "f04662b3e24407799c9a87fde34ac97c91a6374500ee5eb97280abd23276e33c"
)
PINNED_CROSS_LAYER_RESULT_PATH = Path(
    ".local-runs/google--gemma-3-270m/"
    "cross-layer-chart-transport-dev-v1.json"
)
PINNED_CROSS_LAYER_RESULT_FILE_SHA256 = (
    "27c6f2ea1a0079c136f6a5d0ecbdbb7319c86b2e603a7a9c3edd2048569529b6"
)
PINNED_EXACT_EDGE_RESULT_PATH = Path(
    ".local-runs/google--gemma-3-270m/exact-edge-sidecar-v1.json"
)
PINNED_EXACT_EDGE_RESULT_FILE_SHA256 = (
    "21201e970c08a28ce3c69871f8d44d64847a1b0c1acae3184f345cdf4c34f463"
)

EVENT_COUNT = 39
MUTATION_COUNT = 36
CARRIER_SITE_COUNT = 73
LOGICAL_MODE_COUNT = 38
HIDDEN_WIDTH = 640
CHART_COUNT = 8
CHART_RANK = 16
COMMON_COMPARISON_ABI = "carrier_chart_energy.h640.v1"
MUTUAL_K = 2
RANK_BOUNDARY_TIE_ABSOLUTE_TOLERANCE = 1.0e-10
MAX_DIRECTIONAL_NORMALIZED_RESIDUAL = 1.0
BOOTSTRAP_SEED = 7_901
BOOTSTRAP_RESAMPLE_COUNT = 10_000
BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY = 0.80
BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION = 0.01
BOOTSTRAP_DRAW_PROTOCOL = "sha256_counter_modulo_example_count_v1"
BOOTSTRAP_DRAW_DOMAIN = (
    b"fisher_graph:chart-energy-neighbor-bootstrap-draw:v1\0"
)
BOOTSTRAP_CATALOG_DOMAIN = (
    b"fisher_graph.chart_energy_neighbor_graph.bootstrap_catalog.v1\0"
)
CONFIRMATION_FAMILY_COUNT = 6
LOFO_MINIMUM_MUTUAL_COUNT = 5
EXAMPLES_PER_FAMILY = 8
PRIMARY_ELIGIBLE_EXAMPLES_PER_FAMILY = 7
TARGET_TOKEN_COUNTS = (96, 511, 512, 513, 514, 576, 640, 768)
PRIMARY_ELIGIBLE_TOKEN_COUNTS = (96, 511, 512, 513, 576, 640, 768)
TRANSITION_DIAGNOSTIC_TOKEN_COUNT = 514
HARD_LONG_TOKEN_COUNTS = (576, 640, 768)
ROW_SHUFFLE_SEED = 6_607
ROW_SHUFFLE_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-row-shuffle-per-example:v2\0"
)
ROW_SHUFFLE_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-row-shuffle-per-example-catalog:v2\0"
)

PINNED_FRESH_PANEL_SOURCE_PATH = Path(
    "src/fisher_graph/gemma3_chart_native_neighbor_fresh_panels.py"
)
PINNED_FRESH_PANEL_SOURCE_SHA256 = (
    "c7951e12733842fe7a1abae29b0a2465da349ee40f83eebfb386be0fe366e852"
)
PINNED_FRESH_MANIFEST_PATH = Path(
    "examples/gemma3_chart_native_neighbor_fresh_manifest_v1.json"
)
PINNED_FRESH_MANIFEST_FILE_SHA256 = (
    "5031f44454873a369ba7968c5bbfd6cf7e4a7340feaf0eddf6aee51398c948d7"
)
PINNED_FRESH_MANIFEST_SHA256 = (
    "fe045222c8d4f7ec21b14d158da63878c3462a1c38c6b9f9dcecf2b4d1d5e57f"
)
PINNED_FRESH_RECEIPT_PATH = Path(
    "artifacts/research/gemma3-chart-native-neighbor-fresh-panels-v1.json"
)
PINNED_FRESH_RECEIPT_FILE_SHA256 = (
    "65c5f6ea2e1241ad9bdd389da63304962d966fd988a2323ef6baf5dee56410d5"
)
PINNED_FRESH_RECEIPT_SHA256 = (
    "a1586d6601323390573e750e9732bc7db4e415eefc0f15bf0a2fa8ef820cf1f0"
)
PINNED_RUNTIME_QUALIFICATION_PATH = Path(
    "artifacts/research/"
    "gemma3-chart-native-neighbor-runtime-qualification-v1.json"
)
PINNED_RUNTIME_QUALIFICATION_FILE_SHA256 = (
    "3e3603eae4c6d0ba126e293152b213dcecdc343408faff8769b3e6755f1ff600"
)
PINNED_RUNTIME_QUALIFICATION_RECEIPT_SHA256 = (
    "d430889be9f91fdbc76b8c59d4ced252c860b9fc3d7ea6c655f43b25c8f693ca"
)
PINNED_FRESH_ROLE_FILE_SHA256S = {
    "A_chart_transport_fit": (
        "112023d40793f1ff1c9952cc8b44e7d02b1f24a3473a72c941c4a494c2d6d0bf"
    ),
    "B_neighbor_discovery": (
        "f0c0b60e2e42b69703a4c81fc9a35c906a49d7a3e604b186c8d3fbc9c7755cc1"
    ),
    "C_neighbor_confirmation": (
        "ea8e533f6a4fe6c6461577211bd26d25ea0211a43b2847ab5afc75d90550ebc4"
    ),
    "D_response_validation": (
        "f16f4a3c04d04c403db8e50f9c2ed6db8c951b78065b5bb18f68956fb729a51a"
    ),
    "reserve_unopened": (
        "687824c48689a7a54698e3d1aa6d6d685e1bdfcddc703675b03ee5b68d429004"
    ),
}

# The setup artifact authenticates the conservative first-party functional
# dependency closure of the live worker.  The root package is recorded because
# it executes on normal package import, but that eager execution happens before
# static preflight and is explicitly outside the pre-execution authentication
# boundary.  This setup module is the reviewed trust anchor and cannot contain
# its own byte hash; excluding only it avoids a source-hash cycle.
PINNED_WORKER_IMPLEMENTATION_SOURCES = {
    "package_eager_import_boundary": {
        "path": "src/fisher_graph/__init__.py",
        "file_sha256": (
            "dca582dde3ae6d5d09ea7a30c54f5aeff8fc33758b9cab6440a1e59834fbf672"
        ),
    },
    "activations": {
        "path": "src/fisher_graph/activations.py",
        "file_sha256": (
            "887114e91abf0287d64b2d32eeca9e8979f79c76424653ddc34691442a65dcc5"
        ),
    },
    "attention_generator": {
        "path": "src/fisher_graph/gemma3_attention_generator.py",
        "file_sha256": (
            "ba36d1e48e262e8aeb3c97f2acf9051587ff10c1ab8bc89581a8dc83fb51fc19"
        ),
    },
    "chart_energy_neighbor_graph": {
        "path": "src/fisher_graph/chart_energy_neighbor_graph.py",
        "file_sha256": (
            "ae6fbf9235ec0bf2e69a0e83a28980bd30c2a447dc1900c5de64059b0058e1ee"
        ),
    },
    "chart_native_neighbor_runtime": {
        "path": "src/fisher_graph/gemma3_chart_native_neighbor_runtime.py",
        "file_sha256": (
            "42046db14cf4bf8d3ff872ed288b7c3804a11a4633bcec27ae53d13eb6ac1446"
        ),
    },
    "chart_native_neighbor_worker": {
        "path": "src/fisher_graph/gemma3_chart_native_neighbor_experiment.py",
        "file_sha256": (
            "442139eae492752f1bc9d2014502b374cf7f4986a904b26a1075775478bfdd3a"
        ),
    },
    "chunked_tied_lexical_nll": {
        "path": "src/fisher_graph/chunked_tied_lexical_nll.py",
        "file_sha256": (
            "9ae44b700cee330c7b841d98b1d087343fab3cfa0a97d8f2e9ea756a03c587bf"
        ),
    },
    "event_executor": {
        "path": "src/fisher_graph/gemma3_joint_cross_block_chart_executor.py",
        "file_sha256": (
            "d7c397a694abb3d22b1f1289d7b79995cbf652bb824dc45a838b23ee4e847324"
        ),
    },
    "gemma3_chart_mode_surfaces": {
        "path": "src/fisher_graph/gemma3_chart_mode_surfaces.py",
        "file_sha256": (
            "215d2ae925dadccc99c54ef29d11522f558a2fa8cd4d7d4579d2edff1ca9ead8"
        ),
    },
    "gemma3_chart_native_neighbor_fresh_panels": {
        "path": "src/fisher_graph/gemma3_chart_native_neighbor_fresh_panels.py",
        "file_sha256": (
            "c7951e12733842fe7a1abae29b0a2465da349ee40f83eebfb386be0fe366e852"
        ),
    },
    "joint_exact_compiler": {
        "path": "src/fisher_graph/gemma3_joint_cross_block_compiler.py",
        "file_sha256": (
            "142bb374e024cfcbabbd3855b15de49320ba3a0b4c57442c8933b4db8f339f47"
        ),
    },
    "runtime_adapter": {
        "path": "src/fisher_graph/adapters/gemma3.py",
        "file_sha256": (
            "9eba131a4086e884a9d099036bade50ee95760a5febe8f5bff9f1237ee6c378d"
        ),
    },
    "runtime_adapter_base": {
        "path": "src/fisher_graph/adapters/base.py",
        "file_sha256": (
            "6f8753da109d67a8adb825f54d500384899f713a2cd4e0c59bde2a980836c3b4"
        ),
    },
    "runtime_adapter_package": {
        "path": "src/fisher_graph/adapters/__init__.py",
        "file_sha256": (
            "6cfa928e96a02abb1693437532959e2ef83eff894d6c1ed728f1a0cfafad5e1c"
        ),
    },
    "runtime_adapter_toy": {
        "path": "src/fisher_graph/adapters/toy.py",
        "file_sha256": (
            "2e64ba83225360273d5ab96f2037e86014ba093388b4d50b6430445762190387"
        ),
    },
    "codimension_projection": {
        "path": "src/fisher_graph/codimension_projection.py",
        "file_sha256": (
            "4af5306a7b35fe4639e3d6971b5617a18ae5216f24f8d792c1ed2dd182733c39"
        ),
    },
    "compiler_calibration": {
        "path": "src/fisher_graph/compiler/calibration.py",
        "file_sha256": (
            "2d9f510b4871834d8066b43e7ae32005a68695c6a960fec92ce44072b501dce8"
        ),
    },
    "compiler_capabilities": {
        "path": "src/fisher_graph/compiler/capabilities.py",
        "file_sha256": (
            "b74d6a683cf43b69250c8d5e969d84ab6e9e240031a3118caa42e3623a7e625e"
        ),
    },
    "compiler_manifest": {
        "path": "src/fisher_graph/compiler/manifest.py",
        "file_sha256": (
            "18e63013f654e455cd9915e72363646304c71f7bc39ca1a963c2ce6102d0a852"
        ),
    },
    "complete_block_generator_stack": {
        "path": "src/fisher_graph/complete_block_generator_stack.py",
        "file_sha256": (
            "3dc37cc26e1bcfb6feeeb9e43cc6798bddf19035c3b9cd9e00e7fcb1ffa3c582"
        ),
    },
    "complete_block_residual_forms": {
        "path": "src/fisher_graph/complete_block_residual_forms.py",
        "file_sha256": (
            "ef7e59d6a9c474db26955d24748e4c044b27759e2dd550ee4b79e064025f17d8"
        ),
    },
    "config": {
        "path": "src/fisher_graph/config.py",
        "file_sha256": (
            "4b37faa008d2a32713e67549af80d25de705da4e4de97c269942bc5a34015ee0"
        ),
    },
    "external_models": {
        "path": "src/fisher_graph/external_models.py",
        "file_sha256": (
            "0a0be1fd3ca84aa715eb8e4f337f72fb7cea0b3a6228528cb678018376fddbc0"
        ),
    },
    "full_width_single_layer_executor": {
        "path": "src/fisher_graph/full_width_single_layer_executor.py",
        "file_sha256": (
            "c4d33bdd3154f08ad1c652d67164e669b263ddba0f27f34a6b30cafa0d4912a5"
        ),
    },
    "gated_executor": {
        "path": "src/fisher_graph/gated_executor.py",
        "file_sha256": (
            "0aa12b701203f31c261208ea552e4d823b10bf6af5a4a72f872cdfc67c2319e5"
        ),
    },
    "gemma3_ablation_experiment": {
        "path": "src/fisher_graph/gemma3_ablation_experiment.py",
        "file_sha256": (
            "0d20be7b96f2918ca7bf3ca8344138c982c9ec38f480ac1668280e9abd5622c3"
        ),
    },
    "gemma3_all_mode_generator_graph_executor": {
        "path": "src/fisher_graph/gemma3_all_mode_generator_graph_executor.py",
        "file_sha256": (
            "686122b422e41392896e9d1ccaf79745857528fbb42f6349db0b1e063ee72c96"
        ),
    },
    "gemma3_codimension_rotation_experiment": {
        "path": "src/fisher_graph/gemma3_codimension_rotation_experiment.py",
        "file_sha256": (
            "538b1d0d353dcfa9354a3964b27f254ce6fc743d3fd8f5b77ee2c015f8f8e6ee"
        ),
    },
    "gemma3_experiment": {
        "path": "src/fisher_graph/gemma3_experiment.py",
        "file_sha256": (
            "35b36855b232f83018313f14a9b6fd035e8d451a857be88a4948f290d3edf0ca"
        ),
    },
    "gemma3_full_body_residual_carrier_experiment": {
        "path": (
            "src/fisher_graph/gemma3_full_body_residual_carrier_experiment.py"
        ),
        "file_sha256": (
            "33042626bb2d0c1caad41ec7d0127829f91f33ae57b22ad27848a6e247bdd8e5"
        ),
    },
    "gemma3_full_width_single_layer_experiment": {
        "path": "src/fisher_graph/gemma3_full_width_single_layer_experiment.py",
        "file_sha256": (
            "2416ab7b9c0cdb381ff916feefe1a7425aef720fda02d7b6b1ccd2681b4b65ed"
        ),
    },
    "gemma3_gated_executor_experiment": {
        "path": "src/fisher_graph/gemma3_gated_executor_experiment.py",
        "file_sha256": (
            "a767986f894f49a6aaf4d0bba6ef4bda94f9546c1eb2535ebb0195935243f599"
        ),
    },
    "gemma3_joint_cross_block_transport_fit": {
        "path": "src/fisher_graph/gemma3_joint_cross_block_transport_fit.py",
        "file_sha256": (
            "4909366ca27653099731a685f8e6389c194baf40336c72f38f6c0598b6a93256"
        ),
    },
    "gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment": {
        "path": (
            "src/fisher_graph/"
            "gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment.py"
        ),
        "file_sha256": (
            "31d507c300283b4822b41c388b23eae8efe9a9caaf947cc5918973029363b6e3"
        ),
    },
    "gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_protocol": {
        "path": (
            "src/fisher_graph/"
            "gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_protocol.py"
        ),
        "file_sha256": (
            "a2bebec1284fcdfc3a2aba16acacf96a1000c82fe15399c1f2e447ad48c4f1f9"
        ),
    },
    "gemma3_layer_agnostic_whole_model_compression_protocol": {
        "path": (
            "src/fisher_graph/"
            "gemma3_layer_agnostic_whole_model_compression_protocol.py"
        ),
        "file_sha256": (
            "6a14f878e11d55d7747d70636b0f4ee780a695c69d0e0d2640cb629d02e9e72f"
        ),
    },
    "gemma3_layer_agnostic_whole_model_compression_worker": {
        "path": (
            "src/fisher_graph/"
            "gemma3_layer_agnostic_whole_model_compression_worker.py"
        ),
        "file_sha256": (
            "ed096118906df0f68dd30d71b5c70979a5a1ea725156174a6178ddd99dfa4ec8"
        ),
    },
    "gemma3_one_block_residual_form_experiment": {
        "path": "src/fisher_graph/gemma3_one_block_residual_form_experiment.py",
        "file_sha256": (
            "57bfa029b7d4595ed22dc88d7ae70e0148753b3413a9ba601560781e5a87bebf"
        ),
    },
    "gemma3_projection_ladder_experiment": {
        "path": "src/fisher_graph/gemma3_projection_ladder_experiment.py",
        "file_sha256": (
            "23e4212b43fcb624cbe64a7bdc50a6a8c0f8be5b783cf6ead1fb0dcbd6b895d9"
        ),
    },
    "gemma3_rotated_span_executor_experiment": {
        "path": "src/fisher_graph/gemma3_rotated_span_executor_experiment.py",
        "file_sha256": (
            "f9e8e38d00cfc5a5f539be3202a9eefe97aaa906e1d7f9cac9713695c3d8555b"
        ),
    },
    "gemma3_stability_experiment": {
        "path": "src/fisher_graph/gemma3_stability_experiment.py",
        "file_sha256": (
            "40fb038128e58769cf3acb7160f80ef784de317d6de9a15297303108e6d76ed2"
        ),
    },
    "gemma3_weighted_jacobian_experiment": {
        "path": "src/fisher_graph/gemma3_weighted_jacobian_experiment.py",
        "file_sha256": (
            "3cf243e11081180408a74bf188122d21d3350e2ed872bfbfafc550582bc373b5"
        ),
    },
    "instrumentation": {
        "path": "src/fisher_graph/instrumentation.py",
        "file_sha256": (
            "ed9f1373ed52b6275c83edb8359137850e00ba48fc330b9111bd01e0c4f7fa92"
        ),
    },
    "jacobian_probe": {
        "path": "src/fisher_graph/jacobian_probe.py",
        "file_sha256": (
            "f4bd73883c19559872b54f6b5c17cac51171677dfa78eb46432ea1dc01185e50"
        ),
    },
    "layers": {
        "path": "src/fisher_graph/layers.py",
        "file_sha256": (
            "7c6c0ce32c77619acd26e2b8c6015242f497b6ef65ac1e8df0cf2fd027913f25"
        ),
    },
    "linear_codec": {
        "path": "src/fisher_graph/linear_codec.py",
        "file_sha256": (
            "5277ca6b2f14966f7dbd65e69f55520a9755e1b57532764115a5f592751a036b"
        ),
    },
    "modal_ablation": {
        "path": "src/fisher_graph/modal_ablation.py",
        "file_sha256": (
            "9fbc22b52d9cef0eeeb8d327ed5c4c02d91413c7fa05718e21b2218f0b9bc3fb"
        ),
    },
    "model": {
        "path": "src/fisher_graph/model.py",
        "file_sha256": (
            "0d6aad9881267789055f531ca7de4b4dbba46c2a3b09b28eccdde9ace5a231d8"
        ),
    },
    "modes": {
        "path": "src/fisher_graph/modes.py",
        "file_sha256": (
            "aab97603973aee7a7596c41a4facb2d3cbc555e2d5172f08f26cf3afb383e7a1"
        ),
    },
    "residual_carrier": {
        "path": "src/fisher_graph/residual_carrier.py",
        "file_sha256": (
            "9b9dd18eb66189820624b11db531293f01c11dfd2e4433ddfcbe9ece98b5b869"
        ),
    },
    "rotated_span_executor": {
        "path": "src/fisher_graph/rotated_span_executor.py",
        "file_sha256": (
            "076c9e98de1f5f15c4970649bffc8b115a3976ef6ddcf2a0f410747d3a650b2e"
        ),
    },
    "shared_chart_energy_transport": {
        "path": "src/fisher_graph/shared_chart_energy_transport.py",
        "file_sha256": (
            "991e0f8de9597150e2d17e0cbec61f5a4ca473e7bf8dd8a54534024900c466f7"
        ),
    },
    "streaming_shared_chart_energy_transport": {
        "path": "src/fisher_graph/streaming_shared_chart_energy_transport.py",
        "file_sha256": (
            "fa9fcf05eb41cb5291eb57162a6c0160d96a319d494404731f888bf84e985b3f"
        ),
    },
    "static_transformer_span_executor": {
        "path": "src/fisher_graph/static_transformer_span_executor.py",
        "file_sha256": (
            "4f4cd73f50dcc8fcceca38d3577055813bb05a25c0c36f4a1909115378480f57"
        ),
    },
    "streaming_analysis": {
        "path": "src/fisher_graph/streaming_analysis.py",
        "file_sha256": (
            "42d3917b2ff6813dda7bcbdbf950b83064ab0ba34eccc618adcad3078baefb2f"
        ),
    },
    "streaming_block_validation": {
        "path": "src/fisher_graph/streaming_block_validation.py",
        "file_sha256": (
            "fc97d16cbe59efad771aceb67e551f20c291fd7e93284fdeb737ffb10c03c340"
        ),
    },
    "streaming_fisher": {
        "path": "src/fisher_graph/streaming_fisher.py",
        "file_sha256": (
            "a345abbaefffd20dbd4d2dfeeb26d557310354c523272a429dcceae733637dde"
        ),
    },
    "streaming_validation": {
        "path": "src/fisher_graph/streaming_validation.py",
        "file_sha256": (
            "02f6c3387fda79236eb1a46fe4ac563ef678d236c71eb7362988a6070e855ce9"
        ),
    },
    "structured_transformer_layer_executor": {
        "path": "src/fisher_graph/structured_transformer_layer_executor.py",
        "file_sha256": (
            "79352ed5b48871d9ab9a5e8e321cf60b517e2d5a37418ab6ed2e5af239092570"
        ),
    },
    "weighted_jacobian": {
        "path": "src/fisher_graph/weighted_jacobian.py",
        "file_sha256": (
            "55e7b695774aac41feda7225a714da87bd83d4c9fbf890ec754fc545311af0e1"
        ),
    },
}


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical JSON representation used by every digest."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def setup_sha256(value: Mapping[str, object]) -> str:
    """Hash a setup payload, ignoring its embedded digest field."""

    if not isinstance(value, Mapping):
        raise TypeError("chart-native neighbor setup must be a mapping")
    payload = dict(value)
    payload.pop("setup_sha256", None)
    return _sha256(_SETUP_HASH_DOMAIN, payload)


def _event_catalog() -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {
            "event_index": 0,
            "event_id": "lexical.ingress",
            "event_kind": "lexical_ingress",
            "component_family": "lexical",
            "block_ordinal": None,
            "attention_kind": None,
            "token_dependency": "vocabulary_boundary",
        }
    ]
    for block_ordinal in range(18):
        attention_index = len(events)
        attention_kind = (
            "global" if block_ordinal in (5, 11, 17) else "sliding_512"
        )
        events.append(
            {
                "event_index": attention_index,
                "event_id": f"mutation.{attention_index}",
                "event_kind": "attention_mutation",
                "component_family": "attention",
                "block_ordinal": block_ordinal,
                "attention_kind": attention_kind,
                "token_dependency": "causal_prefix_mixer",
            }
        )
        mlp_index = len(events)
        events.append(
            {
                "event_index": mlp_index,
                "event_id": f"mutation.{mlp_index}",
                "event_kind": "mlp_mutation",
                "component_family": "mlp",
                "block_ordinal": block_ordinal,
                "attention_kind": None,
                "token_dependency": "token_local",
            }
        )
    events.extend(
        (
            {
                "event_index": 37,
                "event_id": "terminal.rmsnorm",
                "event_kind": "terminal_rmsnorm",
                "component_family": "terminal",
                "block_ordinal": None,
                "attention_kind": None,
                "token_dependency": "token_local",
            },
            {
                "event_index": 38,
                "event_id": "lexical.egress",
                "event_kind": "lexical_egress",
                "component_family": "lexical",
                "block_ordinal": None,
                "attention_kind": None,
                "token_dependency": "vocabulary_boundary",
            },
        )
    )
    return events


def _carrier_site_catalog(
    events: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    sites: list[dict[str, object]] = []
    for event in events:
        kind = event["event_kind"]
        if kind == "lexical_ingress":
            sites.append(
                {
                    "site_id": "lexical.ingress.output",
                    "event_id": "lexical.ingress",
                    "event_index": 0,
                    "role": "output",
                    "value_semantics": "lexical_hidden_state_h640",
                }
            )
        elif kind in {"attention_mutation", "mlp_mutation"}:
            event_id = str(event["event_id"])
            sites.extend(
                (
                    {
                        "site_id": f"{event_id}.input",
                        "event_id": event_id,
                        "event_index": event["event_index"],
                        "role": "input",
                        "value_semantics": "normalized_mutation_input_h640",
                    },
                    {
                        "site_id": f"{event_id}.output",
                        "event_id": event_id,
                        "event_index": event["event_index"],
                        "role": "output",
                        "value_semantics": "mutation_residual_delta_h640",
                    },
                )
            )
    return sites


def _mode_catalog(
    events: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    modes: list[dict[str, object]] = [
        {
            "mode_id": "mode.lexical.ingress",
            "mode_index": 0,
            "event_id": "lexical.ingress",
            "event_index": 0,
            "mode_kind": "lexical_ingress",
            "native_ABI": "lexical_ingress.lookup_to_h640.v1",
            "comparison_ABI": COMMON_COMPARISON_ABI,
            "context_site_id": "lexical.ingress.output",
            "value_site_id": "lexical.ingress.output",
            "score_gradient_site_id": "lexical.ingress.output",
            "contribution_site_id": "lexical.ingress.output",
            "contribution_rule": "carrier_value_elementwise_score_gradient",
            "requires_cross_token_route_receipt": False,
        }
    ]
    for event in events:
        if event["event_kind"] not in {
            "attention_mutation",
            "mlp_mutation",
        }:
            continue
        event_id = str(event["event_id"])
        kind = str(event["event_kind"])
        modes.append(
            {
                "mode_id": f"mode.{event_id}",
                "mode_index": len(modes),
                "event_id": event_id,
                "event_index": event["event_index"],
                "mode_kind": kind.removesuffix("_mutation"),
                "native_ABI": (
                    "attention.residual_h640.causal_sequence.v1"
                    if kind == "attention_mutation"
                    else "mlp.residual_h640.token_local.v1"
                ),
                "comparison_ABI": COMMON_COMPARISON_ABI,
                "context_site_id": f"{event_id}.input",
                "value_site_id": f"{event_id}.output",
                "score_gradient_site_id": f"{event_id}.output",
                "contribution_site_id": f"{event_id}.output",
                "contribution_rule": (
                    "residual_delta_elementwise_output_score_gradient"
                ),
                "requires_cross_token_route_receipt": (
                    kind == "attention_mutation"
                ),
                "attention_kind": event["attention_kind"],
            }
        )
    modes.append(
        {
            "mode_id": "mode.lexical.egress",
            "mode_index": len(modes),
            "event_id": "lexical.egress",
            "event_index": 38,
            "mode_kind": "lexical_egress",
            "native_ABI": "lexical_egress.tied_vocab_softcap_readout.v1",
            "comparison_ABI": COMMON_COMPARISON_ABI,
            "context_site_id": "terminal.rmsnorm.output",
            "value_site_id": "terminal.rmsnorm.output",
            "score_gradient_site_id": "lexical.egress.readout_pullback",
            "contribution_site_id": "lexical.egress.readout_pullback",
            "contribution_rule": (
                "terminal_carrier_value_elementwise_exact_softcap_readout_"
                "pulled_NLL_score_gradient"
            ),
            "requires_cross_token_route_receipt": False,
        }
    )
    return modes


def _build_state() -> dict[str, object]:
    events = _event_catalog()
    sites = _carrier_site_catalog(events)
    modes = _mode_catalog(events)
    state: dict[str, object] = {
        "schema": GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_SCHEMA,
        "format_version": GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_FORMAT_VERSION,
        "setup_id": SETUP_ID,
        "artifact_role": "frozen_execution_setup_without_measurement",
        "setup_receipt": {
            "prompt_data_opened": False,
            "predecessor_result_opened": False,
            "tokenizer_loaded": False,
            "model_weights_loaded": False,
            "tensor_library_imported": False,
            "activations_captured": False,
            "score_gradients_computed": False,
            "charts_fitted": False,
            "transports_fitted": False,
            "neighbor_graph_built": False,
            "response_ablation_executed": False,
            "scientific_result_present": False,
        },
        "source_model": {
            "model_id": PINNED_MODEL_ID,
            "model_revision": PINNED_MODEL_REVISION,
            "tokenizer_revision": PINNED_MODEL_REVISION,
            "parameter_count": PINNED_SOURCE_PARAMETER_COUNT,
            "source_model_fingerprint_sha256": (
                PINNED_SOURCE_MODEL_FINGERPRINT_SHA256
            ),
            "source_execution_fingerprint_sha256": (
                PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            ),
            "hidden_width": HIDDEN_WIDTH,
            "tied_lexical_physical_storage_required": True,
            "tied_storage_does_not_collapse_ingress_and_egress_modes": True,
        },
        "frozen_runtime_qualification": {
            "authenticated_present": True,
            "path": str(PINNED_RUNTIME_QUALIFICATION_PATH),
            "file_sha256": PINNED_RUNTIME_QUALIFICATION_FILE_SHA256,
            "qualification_receipt_sha256": (
                PINNED_RUNTIME_QUALIFICATION_RECEIPT_SHA256
            ),
            "panel_corpus_bytes_authenticated_opaque": True,
            "prompt_text_decoded_or_semantically_opened": False,
            "panel_semantic_access_performed": False,
            "tokenizer_and_model_loaded_only_after_static_preflight": True,
            "qualifies_compiled_runtime_before_A_access": True,
        },
        "frozen_worker_implementation_sources": {
            name: dict(receipt)
            for name, receipt in sorted(
                PINNED_WORKER_IMPLEMENTATION_SOURCES.items()
            )
        },
        "frozen_predecessor_lineage": {
            "event_executor_source": {
                "path": str(PINNED_EVENT_EXECUTOR_SOURCE_PATH),
                "file_sha256": PINNED_EVENT_EXECUTOR_SOURCE_SHA256,
            },
            "unified_chart_sidecar_source": {
                "path": str(PINNED_UNIFIED_CHART_SIDECAR_SOURCE_PATH),
                "file_sha256": PINNED_UNIFIED_CHART_SIDECAR_SOURCE_SHA256,
            },
            "cross_layer_transport_protocol": {
                "path": str(PINNED_TRANSPORT_PROTOCOL_PATH),
                "file_sha256": PINNED_TRANSPORT_PROTOCOL_FILE_SHA256,
                "protocol_sha256": PINNED_TRANSPORT_PROTOCOL_SHA256,
            },
            "exact_edge_protocol": {
                "path": str(PINNED_EXACT_EDGE_PROTOCOL_PATH),
                "file_sha256": PINNED_EXACT_EDGE_PROTOCOL_FILE_SHA256,
            },
            "spent_development_result_receipts": [
                {
                    "path": str(PINNED_CROSS_LAYER_RESULT_PATH),
                    "file_sha256": PINNED_CROSS_LAYER_RESULT_FILE_SHA256,
                    "outcome": "heldout_edge_gain_gate_failed",
                    "inherited_as_neighbor_truth": False,
                },
                {
                    "path": str(PINNED_EXACT_EDGE_RESULT_PATH),
                    "file_sha256": PINNED_EXACT_EDGE_RESULT_FILE_SHA256,
                    "outcome": "heldout_predictive_or_causal_edge_gate_failed",
                    "inherited_as_neighbor_truth": False,
                },
            ],
            "old_adjacent_same_token_edge_catalog_inherited": False,
            "old_exact_edge_catalog_inherited": False,
            "old_response_neighbor_catalog_inherited": False,
            "predecessors_supply_lineage_and_negative_evidence_only": True,
        },
        "frozen_fresh_panel_lineage": {
            "panel_source": {
                "path": str(PINNED_FRESH_PANEL_SOURCE_PATH),
                "file_sha256": PINNED_FRESH_PANEL_SOURCE_SHA256,
            },
            "manifest": {
                "path": str(PINNED_FRESH_MANIFEST_PATH),
                "file_sha256": PINNED_FRESH_MANIFEST_FILE_SHA256,
                "manifest_sha256": PINNED_FRESH_MANIFEST_SHA256,
            },
            "receipt": {
                "path": str(PINNED_FRESH_RECEIPT_PATH),
                "file_sha256": PINNED_FRESH_RECEIPT_FILE_SHA256,
                "receipt_sha256": PINNED_FRESH_RECEIPT_SHA256,
            },
            "role_files": {
                role: {
                    "path": (
                        "examples/gemma3_chart_native_neighbor_"
                        f"{role}_corpus_v1.json"
                    ),
                    "file_sha256": digest,
                }
                for role, digest in PINNED_FRESH_ROLE_FILE_SHA256S.items()
            },
            "family_count": CONFIRMATION_FAMILY_COUNT,
            "examples_per_family": EXAMPLES_PER_FAMILY,
            "ordered_target_token_counts": list(TARGET_TOKEN_COUNTS),
            "primary_eligible_token_counts_B_C": list(
                PRIMARY_ELIGIBLE_TOKEN_COUNTS
            ),
            "primary_eligible_examples_per_family_B_C": (
                PRIMARY_ELIGIBLE_EXAMPLES_PER_FAMILY
            ),
            "transition_diagnostic_only_token_count": (
                TRANSITION_DIAGNOSTIC_TOKEN_COUNT
            ),
            "hard_long_veto_token_counts": list(HARD_LONG_TOKEN_COUNTS),
            "all_five_role_files_are_authenticated_and_present": True,
            "one_authoritative_manifest_only": True,
        },
        "exact_execution_lineage": {
            "ordered_event_count": EVENT_COUNT,
            "ordered_events": events,
            "event_catalog_sha256": _sha256(
                _EVENT_CATALOG_HASH_DOMAIN, events
            ),
            "mutation_count": MUTATION_COUNT,
            "attention_mutation_count": 18,
            "mlp_mutation_count": 18,
            "terminal_event_is_not_a_logical_mode": True,
            "event_order_may_not_be_relabelled_or_permuted": True,
        },
        "carrier_observation_lineage": {
            "ordered_site_count": CARRIER_SITE_COUNT,
            "ordered_sites": sites,
            "site_catalog_sha256": _sha256(
                _SITE_CATALOG_HASH_DOMAIN, sites
            ),
            "site_count_formula": "1_lexical_ingress_output_plus_2_times_36_mutations",
            "all_sites_are_detached_observations_of_one_exact_execution": True,
            "all_sites_share_example_position_rows_and_execution_fingerprint": (
                True
            ),
            "terminal_and_egress_are_not_silently_added_to_the_73_sites": True,
        },
        "logical_mode_ontology": {
            "logical_mode_count": LOGICAL_MODE_COUNT,
            "ordered_modes": modes,
            "mode_catalog_sha256": _sha256(_MODE_CATALOG_HASH_DOMAIN, modes),
            "all_modes_retain_distinct_native_ABI_provenance": True,
            "all_modes_expose_exact_common_comparison_ABI": (
                COMMON_COMPARISON_ABI
            ),
            "native_ABI_equality_required_for_graph_comparison": False,
            "native_ABI_partition_is_diagnostic_only": True,
            "cross_attention_MLP_lexical_candidates_are_allowed": True,
            "tied_lexical_weight_equality_is_not_chart_equivalence": True,
            "context_and_contribution_site_bindings_are_exact": True,
        },
        "common_comparison_ABI": {
            "abi_id": COMMON_COMPARISON_ABI,
            "abi_version": 1,
            "hidden_width": HIDDEN_WIDTH,
            "row_key": [
                "example_id",
                "prediction_query_position",
                "target_label_position",
            ],
            "row_semantics": (
                "aligned_example_prediction_query_and_target_label_positions"
            ),
            "row_order": (
                "manifest_example_order_then_ascending_prediction_query_"
                "position_then_target_label_position"
            ),
            "row_axis_contract": {
                "prediction_query_positions": "q=0_through_L_minus_2",
                "target_label_positions": "t=1_through_L_minus_1",
                "exact_relation": "t=q+1",
                "context_value_and_contribution_carriers_are_captured_at": "q",
                "next_token_NLL_label_is_read_at": "t",
                "ordered_role_family_example_q_t_catalog_sha256_required": True,
                "off_by_one_or_position_omission_invalidates_run": True,
            },
            "all_mode_row_catalogs_must_match_exactly_within_each_role": True,
            "padding_rows_forbidden": True,
            "masked_or_invalid_rows_forbidden": True,
            "finite_float64_accumulation_required": True,
            "generalized_contribution": {
                "symbol": "a_i_r",
                "semantics": (
                    "value_hadamard_summed_next_token_nll_score_gradient"
                ),
                "formula": (
                    "comparison_value_i_r elementwise_mul "
                    "summed_next_token_NLL_score_gradient_i_r"
                ),
                "summed_objective": (
                    "sum_of_next_token_NLL_over_all_valid_prediction_rows"
                ),
                "output_shape": "h640",
                "signed_values_retained": True,
                "absolute_value_or_squared_value_before_inner_product_forbidden": (
                    True
                ),
                "full_width_vector_retained": True,
            },
            "mode_surface_rules": {
                "lexical_ingress": (
                    "lexical_carrier_output_value_and_its_NLL_score_gradient"
                ),
                "attention": (
                    "mutation_output_residual_delta_and_output_NLL_score_gradient"
                ),
                "mlp": (
                    "mutation_output_residual_delta_and_output_NLL_score_gradient"
                ),
                "lexical_egress": (
                    "terminal_normalized_carrier_value_and_exact_softcap_"
                    "readout_pulled_NLL_score_gradient"
                ),
            },
        },
        "shared_global_chart_atlas": {
            "one_atlas_shared_by_all_38_modes": True,
            "mode_specific_chart_axes_forbidden": True,
            "fit_role": "A_chart_transport_fit",
            "chart_count": CHART_COUNT,
            "tangent_rank": CHART_RANK,
            "hidden_width": HIDDEN_WIDTH,
            "random_seed_required": False,
            "determinism_source": (
                "canonical_fit_pool_order_plus_weighted_farthest_point_"
                "initialization_and_fixed_Lloyd_iterations"
            ),
            "fit_pool_balancing": (
                "equal_mode_then_equal_family_then_equal_example_weight"
            ),
            "fit_pool_canonical_order": (
                "ascending_mode_id_then_manifest_example_order_then_"
                "ascending_prediction_query_position_then_target_label_position"
            ),
            "balanced_row_weight_formula": (
                "w_i_f_e_r=1/(mode_count*family_count*"
                "examples_in_family_f*rows_in_example_f_e)"
            ),
            "balanced_row_weights_sum_over_all_modes_and_rows": 1.0,
            "router_input": "authenticated_mode_context_site_h640",
            "membership_symbol": "q_i_r_c",
            "membership_shape": "rows_by_8",
            "membership_contract": "finite_nonnegative_row_simplex",
            "membership_temperature": 1.0,
            "membership_formula": (
                "q_r_c=softmax_c(-0.5*||x_r-center_c||^2/"
                "(radius_squared_c*temperature))"
            ),
            "center_fit": {
                "algorithm": "balanced_weighted_kmeans_float64",
                "initialization": "deterministic_weighted_farthest_point",
                "first_center": (
                    "pool_row_argmax_w_r_times_squared_distance_to_"
                    "balanced_global_context_mean"
                ),
                "later_center": (
                    "pool_row_argmax_w_r_times_minimum_squared_distance_"
                    "to_previously_selected_centers"
                ),
                "argmax_tie_break": "earliest_row_in_canonical_fit_pool",
                "lloyd_iteration_count": 12,
                "assignment": (
                    "minimum_squared_Euclidean_distance_then_lower_chart_"
                    "index_on_exact_tie"
                ),
                "center_update": (
                    "balanced_weighted_mean_of_hard_assigned_rows"
                ),
                "empty_chart_fails_closed": True,
                "canonical_final_chart_order": (
                    "lexicographic_ascending_float64_center_coordinates"
                ),
                "minimum_hard_rows_per_chart": CHART_RANK + 1,
                "minimum_balanced_hard_occupancy_weight": 0.005,
            },
            "radius_squared": {
                "context_reference": (
                    "max(sum_pool_w_r*||x_r||^2,1e-30)"
                ),
                "floor": "1e-8*context_reference",
                "per_chart": (
                    "max(balanced_weighted_mean_hard_assigned_"
                    "||x_r-center_c||^2,floor)"
                ),
            },
            "tangent_fit": {
                "input": "centered_context_x_r_minus_center_c",
                "soft_weight": (
                    "balanced_row_weight_w_r_times_membership_q_r_c"
                ),
                "covariance": (
                    "sum_r(w_r*q_r_c*(x_r-center_c)(x_r-center_c)^T)/"
                    "sum_r(w_r*q_r_c)"
                ),
                "solver": (
                    "symmetric_eigh_float64_descending_eigenvalue"
                ),
                "selected_rank": CHART_RANK,
                "eigenvalue_tie_gate": (
                    "every_selected_eigenvalue_minus_following_"
                    "eigenvalue_absolute_gap_gt_1e-12*max(1,abs(each))"
                ),
                "canonical_sign": (
                    "largest_absolute_loading_coordinate_positive; "
                    "lowest_coordinate_index_breaks_absolute_tie"
                ),
                "context_variance_reference": (
                    "max(sum_pool_w_r*||x_r-global_context_mean||^2/"
                    "640,1e-30)"
                ),
                "variance_floor": "1e-8*context_variance_reference",
                "tangent_variance": (
                    "max(selected_eigenvalue,variance_floor)"
                ),
                "orthogonal_variance": (
                    "sum_r(w_r*q_r_c*||(I-B_cB_c^T)(x_r-center_c)||^2)/"
                    "(sum_r(w_r*q_r_c)*(640-16))"
                ),
                "orthogonal_alpha": (
                    "1/max(orthogonal_variance,variance_floor)"
                ),
                "coordinate_formula": (
                    "z_r_c=(x_r-center_c)^T*B_c/"
                    "sqrt(tangent_variance_c)"
                ),
                "generalized_contributions_do_not_fit_router_or_metric": (
                    True
                ),
            },
            "shared_coordinate_signature_required_for_every_mode": True,
            "chart_artifact_sha256_required": True,
            "center_tangent_metric_and_router_receipts_required": True,
            "heldout_roles_apply_frozen_atlas_without_refit": True,
            "production_streaming_execution": {
                "required_for_live_h640_run": True,
                "in_memory_fit_shared_chart_atlas_is_fixture_reference_only": (
                    True
                ),
                "raw_capture_storage_dtype": "float32_little_endian",
                "all_numeric_reductions_and_solvers_dtype": "float64_CPU",
                "raw_mode_spool_fields": [
                    "context_values_h640",
                    "comparison_values_h640",
                    "signed_generalized_contributions_h640",
                ],
                "raw_mode_spool_contains_prompt_text_or_token_ids": False,
                "raw_mode_spool_is_strict_no_clobber_and_hash_sealed": True,
                "fixed_row_chunk_size": 256,
                "fit_pool_row_count": 939_816,
                "fit_pool_row_count_formula": "38_modes_times_24732_A_rows",
                "full_context_pool_torch_cat_forbidden": True,
                "broadcast_N_by_8_by_640_distance_tensor_forbidden": True,
                "chunk_distance_formula": (
                    "squared_distance=||x||^2+||center||^2-2*x*center^T"
                ),
                "streaming_passes": [
                    "balanced_global_context_mean_and_reference",
                    "deterministic_weighted_farthest_point_scans",
                    "12_fixed_weighted_Lloyd_assignment_sum_count_scans",
                    "final_hard_occupancy_and_radius_scan",
                    "soft_membership_weighted_context_covariance_scan",
                ],
                "scan_order": (
                    "ascending_mode_id_then_canonical_row_order_in_fixed_"
                    "256_row_chunks"
                ),
                "within_chunk_distance_backend": "torch_mm_CPU_float64",
                "chunk_schedule_and_backend_build_receipt_sha256_required": (
                    True
                ),
                "streaming_and_fixture_reference_parity_required_on_"
                "preregistered_small_case": True,
                "raw_spool_bytes_per_production_mode": 189_941_760,
                "raw_spool_bytes_per_production_role": 7_217_786_880,
                "default_streaming_kernel_receipt_sha256": (
                    "c87ca7f273bd224cc81eb49eb4e6cb22a98514ae85b293dcbe510bb004f6524e"
                ),
                "production_resource_estimate_receipt_sha256": (
                    "476029903c44172d7e73e47cb83b751aad2558cd2a6e94a32d4b31016e17d1f0"
                ),
                "estimated_peak_A_payload_plus_atomic_temp_bytes": (
                    10_829_054_592
                ),
                "maximum_live_resident_set_bytes": 8 * 1024**3,
                "resident_set_cap_is_fail_closed_and_phase_logged": True,
            },
            "local_SPD_metric": {
                "symbol": "M_c",
                "semantics": (
                    "shared_membership_spd_tangent_inverse_scale_plus_"
                    "isotropic_orthogonal"
                ),
                "formula": (
                    "B_c diag(inverse_scale_c_squared) B_c_transpose plus "
                    "alpha_c times (I minus B_c B_c_transpose)"
                ),
                "B_c_shape": "640_by_16",
                "B_c_columns_orthonormal": True,
                "all_tangent_scales_finite_positive": True,
                "alpha_c_finite_strictly_positive": True,
                "metric_is_full_rank_SPD": True,
                "isotropic_orthogonal_residual_may_not_be_zeroed": True,
            },
            "per_row_required_fields": [
                "family_id",
                "example_id",
                "prediction_query_position",
                "target_label_position",
                "mode_id",
                "native_ABI",
                "comparison_ABI",
                "context_site_id",
                "contribution_site_id",
                "row_catalog_sha256",
                "shared_chart_artifact_sha256",
                "membership_q_8",
                "tangent_coordinates_z_8_by_16",
                "full_width_signed_contribution_a_h640",
                "weighted_activation_energy_diagnostic_only",
                "weighted_chart_tangent_energy",
                "weighted_chart_orthogonal_energy",
                "weighted_fisher_energy",
            ],
        },
        "Fisher_chart_energy": {
            "semantics": "membership_weighted_spd_self_cross_mismatch",
            "weighted_activation_energy_formula": (
                "sum_c q_i_r_c times comparison_value_i_r^T M_c "
                "comparison_value_i_r"
            ),
            "weighted_activation_energy_is_diagnostic_only": True,
            "self_energy_formula": "E_i_r=sum_c q_i_r_c times a_i_r^T M_c a_i_r",
            "primary_self_energy_field": "weighted_fisher_energy",
            "weighted_chart_tangent_energy_formula": (
                "sum_c q_i_r_c times squared_norm(diag(inverse_scale_c) "
                "B_c_transpose a_i_r)"
            ),
            "weighted_chart_orthogonal_energy_formula": (
                "sum_c q_i_r_c alpha_c times squared_norm((I minus "
                "B_c B_c_transpose) a_i_r)"
            ),
            "full_width_energy_closure": (
                "weighted_chart_tangent_energy plus "
                "weighted_chart_orthogonal_energy equals "
                "weighted_fisher_energy"
            ),
            "cross_inner_product_formula": (
                "C_ij_r=sum_c sqrt(q_i_r_c q_j_r_c) times "
                "a_i_r^T M_c a_j_r"
            ),
            "mismatch_formula": "Delta_ij_r=E_i_r+E_j_r-2C_ij_r",
            "mismatch_nonnegative_up_to_frozen_float64_tolerance": True,
            "sign_and_cross_terms_retained": True,
            "individual_authenticated_row_Fisher_energy_may_be_exactly_zero": (
                True
            ),
            "minimum_row_Fisher_energy_is_finite_and_nonnegative": True,
            "negative_or_nonfinite_row_Fisher_energy_fails_closed": True,
            "zero_energy_rows_are_never_dropped": True,
            "per_family_directed_target_energy_at_or_below_epsilon_fails_closed": (
                True
            ),
            "coordinate_only_energy_without_orthogonal_closure_forbidden": True,
            "raw_activation_norm_without_score_gradient_forbidden": True,
            "raw_weight_similarity_is_diagnostic_only": True,
        },
        "directed_chart_transport": {
            "fit_role": "A_chart_transport_fit",
            "fit_all_ordered_distinct_mode_pairs": True,
            "ordered_pair_count": LOGICAL_MODE_COUNT * (LOGICAL_MODE_COUNT - 1),
            "source_feature": "concat_c(q_source_c times [1,z_source_c])",
            "source_feature_width": CHART_COUNT * (CHART_RANK + 1),
            "target": "full_width_signed_generalized_contribution_a_target_h640",
            "solver": "target_Fisher_weighted_ridge_float64",
            "relative_ridge": 1.0e-3,
            "ridge_scale_floor": 1.0e-12,
            "target_Fisher_energy_floor": 1.0e-12,
            "canonical_WLS": {
                "X": (
                    "concat_c(q_source_c*[1,z_source_c])_shape_N_by_136"
                ),
                "Y": (
                    "target_signed_generalized_contribution_shape_N_by_640"
                ),
                "base_row_weight": (
                    "1/(mode_count*family_count*examples_in_family*"
                    "rows_in_example)"
                ),
                "unnormalized_W_diagonal": (
                    "base_row_weight*max(target_row_weighted_fisher_"
                    "energy,1e-12)"
                ),
                "W_normalization": (
                    "divide_all_diagonal_entries_by_their_sum"
                ),
                "zero_Fisher_rows_retained_with_floor": True,
                "G": "X^T*W*X",
                "ridge_scale": "trace(G)/136",
                "absolute_ridge": "1e-3*max(ridge_scale,1e-12)",
                "solve": (
                    "Beta=solve(G+absolute_ridge*I_136,X^T*W*Y)"
                ),
                "all_136_coefficients_penalized": True,
                "separate_intercept": False,
                "float64_CPU_required": True,
            },
            "bias_is_represented_by_chart_membership_features": True,
            "fit_tie_and_SVD_orientation_rules_are_deterministic": True,
            "B_discovery_C_confirmation_D_response_rows_never_enter_fit": True,
            "frozen_transport_applied_without_refit_on_heldout_roles": True,
            "source_to_target_semantics": (
                "transport_source_chart_packet_to_predict_target_h640_"
                "generalized_contribution"
            ),
            "transported_membership_for_residual": "source_membership_q",
            "exact_target_membership_for_residual": "target_membership_q",
            "required_directed_evaluation_fields": [
                "source_mode_id",
                "target_mode_id",
                "comparison_ABI",
                "evaluation_panel_catalog_sha256",
                "aligned_row_catalog_sha256",
                "weighted_predicted_fisher_energy",
                "weighted_target_fisher_energy",
                "weighted_predicted_target_cross_inner_product",
                "weighted_transport_residual_energy",
                "weighted_zero_prediction_residual_energy",
                "weighted_row_shuffled_transport_residual_energy",
                "real_gain_over_zero",
                "row_shuffle_gain_over_zero",
            ],
            "weighted_transport_residual_energy_formula": (
                "weighted_predicted_fisher_energy plus "
                "weighted_target_fisher_energy minus 2 times "
                "weighted_predicted_target_cross_inner_product"
            ),
            "weighted_zero_prediction_residual_energy_formula": (
                "weighted_target_fisher_energy"
            ),
            "real_gain_over_zero_formula": (
                "1 minus weighted_transport_residual_energy divided_by "
                "weighted_zero_prediction_residual_energy"
            ),
            "row_shuffle_gain_over_zero_formula": (
                "1 minus weighted_row_shuffled_transport_residual_energy "
                "divided_by weighted_zero_prediction_residual_energy"
            ),
            "directed_residual_energy": (
                "sum_rows mismatch(transported_source_contribution_and_"
                "exact_target_contribution_under_shared_SPD_chart_metric)"
            ),
            "directional_normalization": (
                "source_to_target_residual_energy_divided_by_exact_target_"
                "Fisher_self_energy"
            ),
            "within_family_directional_distance": (
                "sum_family_row_residual_energy_divided_by_"
                "sum_family_row_exact_target_self_energy"
            ),
            "pooled_directional_distance": (
                "equal_family_mean_of_within_family_directional_distances"
            ),
            "global_sum_residual_over_sum_target_energy_is_diagnostic_only": True,
            "symmetric_neighbor_distance": (
                "0.5 times (distance_a_to_b plus distance_b_to_a)"
            ),
            "both_directed_receipts_required": True,
            "zero_prediction_control_ratio": 1.0,
            "zero_prediction_control_required_per_family_and_direction": True,
            "within_example_source_row_shuffle_control_required_per_direction": (
                True
            ),
            "row_shuffle_control": {
                "real_A_fitted_transport_reused_without_refit": True,
                "separately_fitted_shuffled_transport_forbidden": True,
                "applies_to_roles": [
                    "B_neighbor_discovery",
                    "C_neighbor_confirmation",
                ],
                "source_feature_rows_permuted_within_each_family_example": True,
                "cross_example_source_rows_forbidden": True,
                "diagnostic_514_rows_cannot_enter_primary_control": True,
                "target_rows_unchanged": True,
                "seed": ROW_SHUFFLE_SEED,
                "domain_hex": ROW_SHUFFLE_DOMAIN.hex(),
                "preimage_encoding": (
                    "domain||seed_u64be||role_utf8_length_u32be||role_utf8||"
                    "family_utf8_length_u32be||family_utf8||"
                    "example_utf8_length_u32be||example_utf8"
                ),
                "digest_selection": (
                    "first_8_sha256_bytes_u64be_mod_"
                    "(family_example_row_count_minus_1)"
                ),
                "permutation": (
                    "nonzero_cyclic_offset=1+digest_selection; "
                    "permuted_source_at_target_row_r=source_row_"
                    "((r+offset)_mod_family_example_row_count)"
                ),
                "minimum_family_example_row_count": 2,
                "derangement_guaranteed_within_each_family_example": True,
                "one_role_family_example_permutation_catalog_reused_for_all_"
                "directed_pairs_point_bootstrap_and_LOFO_replicates": True,
                "subset_stable_under_whole_example_selection": True,
                "permutation_applied_before_example_resampling": True,
                "catalog_digest_domain_hex": (
                    ROW_SHUFFLE_CATALOG_DOMAIN.hex()
                ),
                "catalog_digest_encoding": (
                    "sha256(catalog_domain||canonical_JSON_of_ordered_role_"
                    "family_source_row_to_permuted_source_row_catalog)"
                ),
                "permutation_catalog_sha256_required": True,
            },
            "transport_artifact_and_fit_row_catalog_sha256_required": True,
            "per_pair_fit_receipts_bind": [
                "source_mode_id",
                "target_mode_id",
                "atlas_sha256",
                "fit_panel_and_row_catalog_sha256",
                "source_design_X_sha256",
                "target_Y_sha256",
                "effective_normalized_W_sha256",
                "solver_id",
                "relative_and_absolute_ridge",
                "coefficient_sha256",
            ],
            "fit_and_evaluation_panel_catalog_sha256s_required": True,
            "production_streaming_execution": {
                "compact_applied_mode_store_fields": [
                    "source_features_N_by_136",
                    "memberships_N_by_8",
                    "signed_generalized_contributions_N_by_640",
                    "weighted_activation_energy_N",
                    "weighted_chart_tangent_energy_N",
                    "weighted_chart_orthogonal_energy_N",
                    "weighted_fisher_energy_N",
                ],
                "compact_store_field_dtypes": {
                    "source_features_N_by_136": "float64_little_endian",
                    "memberships_N_by_8": "float64_little_endian",
                    "signed_generalized_contributions_N_by_640": (
                        "float32_little_endian"
                    ),
                    "weighted_activation_energy_N": "float64_little_endian",
                    "weighted_chart_tangent_energy_N": (
                        "float64_little_endian"
                    ),
                    "weighted_chart_orthogonal_energy_N": (
                        "float64_little_endian"
                    ),
                    "weighted_fisher_energy_N": "float64_little_endian",
                },
                "compact_store_contains_prompt_text_or_token_ids": False,
                "raw_role_spool_transformed_once_then_deleted_only_after_"
                "strict_compact_store_reload": True,
                "compact_bytes_per_production_mode": 92_596_608,
                "compact_bytes_per_production_role": 3_518_671_104,
                "A_fit_accumulates_XTWX_and_XTWY_in_fixed_row_chunks": True,
                "fit_target_outer_source_inner_schedule": True,
                "B_C_evaluation_target_outer_source_inner_schedule": True,
                "one_target_contribution_and_membership_plus_one_source_"
                "feature_and_membership_store_resident": True,
                "prediction_kernel_id": (
                    "torch_mm_cpu_float64_streamed_fixed_rows_v1"
                ),
                "prediction_row_chunk_size": 256,
                "prediction_backend_torch_version_and_build_receipt_required": (
                    True
                ),
                "one_python_matrix_vector_call_per_row_forbidden": True,
                "ordered_pair_checkpoint_and_progress_receipts_required": True,
                "all_1406_coefficient_unique_float64_bytes": 979_025_920,
                "report_may_not_embed_coefficient_tensors": True,
            },
        },
        "attention_cross_token_route_chart": {
            "required_for_every_attention_mode": True,
            "supplemental_to_common_axis_neighbor_distance": True,
            "may_not_replace_h640_contribution_energy": True,
            "route_row_key": [
                "example_id",
                "context_fingerprint_sha256",
                "prediction_query_position",
                "query_head",
                "source_token_position",
            ],
            "causal_constraint": (
                "source_token_position_le_prediction_query_position"
            ),
            "sliding_512_constraint": (
                "max(0,prediction_query_position-511)_le_source_token_position_"
                "le_prediction_query_position"
            ),
            "global_constraint": (
                "0_le_source_token_position_le_prediction_query_position"
            ),
            "no_top_k_or_energy_threshold_route_truncation": True,
            "route_value": (
                "exact_native_attention_source_to_target_value_contribution_"
                "after_output_projection_before_post_attention_RMSNorm_in_h640"
            ),
            "route_generalized_contribution": (
                "route_value_elementwise_mul_pre_postnorm_operator_output_"
                "NLL_score_gradient"
            ),
            "route_sum_closure_target": (
                "pre_postnorm_attention_operator_output_elementwise_mul_"
                "operator_output_NLL_score_gradient"
            ),
            "route_sum_must_close_to_pre_postnorm_operator_contribution": True,
            "route_sum_does_not_equal_common_postnorm_delta_contribution": True,
            "nonzero_operator_closure_remainder_invalidates_route_receipt": True,
            "production_factorized_aggregation": {
                "kernel_id": (
                    "factorized_attention_route_energy.p_u_g.float64_chunked.v1"
                ),
                "native_probability_factor_P_shape": "B_by_H_by_Q_by_K",
                "repeated_native_value_factor_V_shape": "B_by_H_by_K_by_64",
                "head_output_projection_slice_W_shape": "H_by_640_by_64",
                "projected_value_factor_formula": (
                    "U_b_h_k=V_b_h_k times transpose(W_h)"
                ),
                "projected_value_factor_U_shape": "B_by_H_by_K_by_640",
                "operator_score_gradient_G_shape": "B_by_Q_by_640",
                "route_value_formula": "R_b_h_q_k=P_b_h_q_k times U_b_h_k",
                "route_coordinate_formula": (
                    "a_b_h_q_k=R_b_h_q_k elementwise_mul G_b_q"
                ),
                "intrinsic_route_energy_formula": (
                    "E_b_h_q_k=P_b_h_q_k_squared times sum_d(" 
                    "U_b_h_k_d_squared times G_b_q_d_squared)"
                ),
                "operator_reconstruction_formula": (
                    "Y_b_q=sum_h_k(P_b_h_q_k times U_b_h_k)"
                ),
                "head_coordinate_sum_formula": (
                    "sum_q G_b_q elementwise_mul sum_k(P_b_h_q_k*U_b_h_k)"
                ),
                "head_coordinate_second_moment_sum_formula": (
                    "sum_q G_b_q_squared elementwise_mul "
                    "sum_k(P_b_h_q_k_squared*U_b_h_k_squared)"
                ),
                "fixed_q_and_k_chunk_schedule_required": True,
                "prediction_query_chunk_size": 32,
                "source_key_chunk_size": 128,
                "CPU_float64_reduction_required": True,
                "stored_native_probability_bytes_preserved_without_"
                "renormalization": True,
                "source_softmax_dtype": "torch.float32",
                "source_softmax_device": "cpu",
                "probability_simplex_audit_accumulation_dtype": (
                    "torch.float64"
                ),
                "float32_probability_sum_then_float64_cast_forbidden": True,
                "probability_simplex_fixed_tolerance_forbidden": True,
                "probability_simplex_forward_error_formula_id": (
                    "float32_softmax_simplex."
                    "gamma32_support_plus_2_gamma64_audit_plus_ftz.v1"
                ),
                "probability_simplex_support_derivation_id": (
                    "query_key_valid_masks_logical_causal_optional_"
                    "strict_window.v1"
                ),
                "probability_simplex_support_minimum": 1,
                "probability_simplex_support_maximum": 768,
                "probability_simplex_float32_unit_roundoff_power": -24,
                "probability_simplex_float64_unit_roundoff_power": -53,
                "probability_simplex_float32_FTZ_safety_power": -126,
                "probability_simplex_bound_formula": (
                    "gamma32(k+2)+gamma64(k-1)*(1+gamma32(k+2))+"
                    "k*2^-126"
                ),
                "finite_probability_in_closed_unit_interval_required": True,
                "masked_and_invalid_route_probabilities_exact_zero_required": (
                    True
                ),
                "native_dense_support_equals_authenticated_logical_support_"
                "at_capture_required": True,
                "per_snapshot_and_per_example_simplex_audit_receipt_required": (
                    True
                ),
                "torch_version_build_and_kernel_receipt_required": True,
                "frozen_kernel_receipt_sha256": (
                    "228433c4960c61660e9e0cb6e3f08d1a2a0fbf54f6bd1f9f6175d8beac23020e"
                ),
                "per_example_receipt_required": True,
                "authenticated_factor_hashes_required": [
                    "P_probability_tensor_sha256",
                    "repeated_V_tensor_sha256",
                    "O_head_slice_tensor_sha256",
                    "projected_U_tensor_sha256",
                    "operator_gradient_G_tensor_sha256",
                    "query_and_key_mask_position_catalog_sha256",
                    "context_and_objective_receipt_sha256",
                    "attention_config_snapshot_and_execution_sha256",
                ],
                "materialized_h640_route_coordinate_scalar_count": 0,
                "literal_full_route_coordinate_stream_hash_forbidden": True,
                "top64_and_distance_bins_are_exact_not_truncated_input": True,
                "small_exhaustive_factorized_parity_required": True,
                "operator_intrinsic_and_common_postnorm_energy_zero_support_"
                "equivalence_not_required": True,
            },
            "operator_output_to_common_postnorm_delta_bridge": {
                "required": True,
                "reason": "post_attention_RMSNorm_is_nonlinear",
                "source_surface": "pre_postnorm_attention_operator_output_h640",
                "target_surface": "mutation_output_postnorm_residual_delta_h640",
                "authenticated_exact_native_transform_receipt_required": True,
                "source_target_row_catalogs_must_match": True,
                "source_and_target_tensor_catalog_sha256s_required": True,
                "native_transform_config_and_source_sha256s_required": True,
                "bridge_is_provenance_not_neighbor_distance": True,
            },
            "attention_mask_window_kind_and_position_catalog_sha256_required": True,
            "factorized_route_catalog_and_energy_signature_sha256_required": (
                True
            ),
            "route_similarity_is_diagnostic_only_in_v1": True,
            "same_token_only_route_receipt_is_invalid": True,
        },
        "lexical_egress_readout_chart": {
            "required": True,
            "separate_from_ordered_73_carrier_sites": True,
            "event_id": "lexical.egress",
            "event_index": 38,
            "context_and_value_site_id": "terminal.rmsnorm.output",
            "pullback_site_id": "lexical.egress.readout_pullback",
            "native_tied_embedding_projection_authenticated": True,
            "exact_model_config_logit_softcap_authenticated": True,
            "softcap_derivative_included_in_pullback": True,
            "summed_next_token_NLL_logit_score_is_pulled_back_to_h640": True,
            "pullback_formula": (
                "dNLL_dterminal_h = J_exact_softcapped_tied_readout^T "
                "dNLL_dsoftcapped_logits"
            ),
            "comparison_contribution": (
                "terminal_normalized_carrier_value_elementwise_mul_"
                "dNLL_dterminal_h"
            ),
            "raw_vocabulary_logits_as_chart_coordinates_forbidden": True,
            "pre_softcap_or_softcap_omitting_pullback_forbidden": True,
            "vocabulary_width_never_enters_common_comparison_ABI": True,
            "readout_config_tensor_and_pullback_catalog_sha256_required": True,
            "chunked_full_vocabulary_NLL_VJP": {
                "kernel_id": (
                    "gemma3_full_vocab_softcap_shifted_nll_vjp.chunked.v1"
                ),
                "kernel_receipt_sha256": (
                    "e79c953bd99483d87e8c09406388deecf2474a9dc20c468c0fd68c38d9deffdb"
                ),
                "prediction_position_chunk_size": 32,
                "vocabulary_chunk_size": 4096,
                "float64_reduction_required": True,
                "first_A_dense_parity_selected_prediction_rows": 2,
                "dense_reference_dtype": "torch.float64",
                "float32_dense_reference_forbidden": True,
                "parity_absolute_tolerance": 2.0e-5,
                "parity_relative_tolerance": 2.0e-5,
            },
        },
        "neighbor_graph_protocol": {
            "node_count": LOGICAL_MODE_COUNT,
            "complete_cross_kind_ranking_universe": True,
            "unordered_pair_count": LOGICAL_MODE_COUNT * (LOGICAL_MODE_COUNT - 1) // 2,
            "rank_metric": "symmetric_bidirectional_normalized_transport_residual",
            "k": MUTUAL_K,
            "mutual_neighbor_required": True,
            "self_neighbors_forbidden": True,
            "rank_boundary_tie_absolute_tolerance": (
                RANK_BOUNDARY_TIE_ABSOLUTE_TOLERANCE
            ),
            "rank_boundary_tie_policy": "fail_closed_undefined_node_neighbors",
            "fisher_energy_epsilon": 1.0e-12,
            "mode_kind_or_event_index_tie_break_forbidden": True,
            "maximum_each_directional_normalized_residual": (
                MAX_DIRECTIONAL_NORMALIZED_RESIDUAL
            ),
            "nonpositive_real_gain_over_zero_prediction_fails_edge": True,
            "real_transport_must_outperform_row_shuffle_in_both_directions": True,
            "B_discovery_nominates_only_mutual_k2_candidates": True,
            "C_confirmation_may_test_only_frozen_B_candidates": True,
            "new_C_edges_are_exploratory_only": True,
            "connected_components_are_diagnostic_not_clusters": True,
            "maximal_cliques_may_overlap_and_are_diagnostic": True,
        },
        "fresh_data_firewall": {
            "ordered_roles": [
                "A_chart_transport_fit",
                "B_neighbor_discovery",
                "C_neighbor_confirmation",
                "D_response_validation",
                "reserve_unopened",
            ],
            "all_five_role_files_are_authenticated_and_present": True,
            "one_authoritative_frozen_fresh_manifest_and_receipt": True,
            "manifest_file_sha256": PINNED_FRESH_MANIFEST_FILE_SHA256,
            "manifest_semantic_sha256": PINNED_FRESH_MANIFEST_SHA256,
            "receipt_file_sha256": PINNED_FRESH_RECEIPT_FILE_SHA256,
            "receipt_semantic_sha256": PINNED_FRESH_RECEIPT_SHA256,
            "role_file_sha256s": dict(PINNED_FRESH_ROLE_FILE_SHA256S),
            "exact_role_file_and_content_catalog_hashes_required_before_access": True,
            "static_preflight_may_read_raw_file_bytes_only_to_verify_pinned_"
            "SHA256_for_A_B_C_D_and_reserve": True,
            "static_preflight_may_not_JSON_decode_tokenize_or_semantically_"
            "inspect_any_role": True,
            "prompt_example_template_instance_paraphrase_and_semantic_lineage_"
            "intersections_between_roles_must_be_zero": True,
            "family_catalogs": {
                "A_chart_transport_fit": "fit_only_never_evaluation",
                "B_neighbor_discovery": "candidate_nomination_only_spent_after_use",
                "C_neighbor_confirmation": (
                    "exactly_6_families_confirmation_only_spent_after_use"
                ),
                "D_response_validation": (
                    "opened_only_after_C_chart_graph_and_receipt_are_frozen"
                ),
                "reserve_unopened": "parse_and_hash_forbidden_until_new_protocol",
            },
            "equal_examples_per_family_within_each_executable_role": True,
            "exact_examples_per_family": EXAMPLES_PER_FAMILY,
            "ordered_target_token_counts": list(TARGET_TOKEN_COUNTS),
            "short_boundary_and_long_context_strata_required_in_B_C_and_D": True,
            "B_C_primary_eligibility": {
                "eligible_token_counts": list(PRIMARY_ELIGIBLE_TOKEN_COUNTS),
                "eligible_examples_per_family": (
                    PRIMARY_ELIGIBLE_EXAMPLES_PER_FAMILY
                ),
                "pooled_point_topology_bootstrap_and_LOFO_use_only_eligible_"
                "examples": True,
                "token_count_514_is_excluded_from_primary_point_topology_"
                "bootstrap_LOFO_and_hard_gates": True,
            },
            "transition_514_reporting": {
                "token_count": TRANSITION_DIAGNOSTIC_TOKEN_COUNT,
                "stratum_id": "boundary_transition_514",
                "diagnostic_only": True,
                "authorizes_or_vetoes_edges": False,
                "non_authorizing_core_subset_receipt_required": True,
            },
            "hard_long_confirmation_veto": {
                "token_counts": list(HARD_LONG_TOKEN_COUNTS),
                "C_non_authorizing_core_subset_receipt_required": True,
                "same_point_distance_zero_and_shuffle_control_gates_as_core": (
                    True
                ),
                "frozen_B_and_C_confirmed_edge_must_pass_hard_long_subset": (
                    True
                ),
                "hard_long_subset_may_not_nominate_new_edges": True,
            },
            "context_stratum_metrics_reported_separately": True,
            "A_only_may_fit_shared_atlas_or_transport_tensors": True,
            "B_C_D_may_not_refit_or_select_atlas_transport_or_thresholds": True,
            "C_may_not_change_k_candidate_catalog_or_gate_thresholds": True,
            "D_may_not_add_remove_or_rerank_chart_neighbor_edges": True,
            "semantic_access_sequence": [
                "parse_and_tokenize_A_only",
                "freeze_shared_atlas_and_all_1406_A_transports",
                "parse_and_tokenize_B_only",
                "freeze_B_candidate_catalog",
                "parse_and_tokenize_C_only",
                "freeze_C_confirmed_chart_graph",
                "optionally_parse_and_tokenize_D_only_after_C_freeze",
            ],
            "later_role_semantic_access_before_prior_freeze_invalidates_run": (
                True
            ),
            "reserve_opaque_byte_hash_access_allowed": True,
            "reserve_JSON_decode_tokenizer_or_model_access_invalidates_run": (
                True
            ),
        },
        "fresh_confirmation_gates": {
            "candidate_must_be_in_frozen_B_catalog": True,
            "C_pooled_mutual_k2_required": True,
            "C_both_directional_residual_ratios_at_most_1": True,
            "C_real_transport_outperforms_zero_and_shuffle_both_directions": True,
            "attention_endpoint_route_receipts_must_pass_if_applicable": True,
            "within_family_bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "resample_count": BOOTSTRAP_RESAMPLE_COUNT,
                "maximum_degenerate_fraction": (
                    BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION
                ),
                "sampling_unit": "whole_example_with_all_its_target_token_rows",
                "draw_protocol": BOOTSTRAP_DRAW_PROTOCOL,
                "draw_protocol_domain_hex": BOOTSTRAP_DRAW_DOMAIN.hex(),
                "draw_preimage_encoding": (
                    "domain||seed_u64be||resample_u64be||family_u32be||"
                    "draw_u32be"
                ),
                "draw_digest_selection": (
                    "first_8_sha256_bytes_u64be_mod_example_count"
                ),
                "draw_catalog_digest_domain_hex": (
                    BOOTSTRAP_CATALOG_DOMAIN.hex()
                ),
                "draw_catalog_digest_encoding": (
                    "sha256(catalog_domain||canonical_config||repeated("
                    "resample_u64be,family_u32be,draw_u32be,example_u32be))"
                ),
                "draw_catalog_iteration_order": (
                    "resample_then_configured_family_then_draw"
                ),
                "all_eight_diagnostic_source_draws_per_family": (
                    EXAMPLES_PER_FAMILY
                ),
                "primary_authorizing_draws_per_family": (
                    PRIMARY_ELIGIBLE_EXAMPLES_PER_FAMILY
                ),
                "draws_per_family": PRIMARY_ELIGIBLE_EXAMPLES_PER_FAMILY,
                "configured_family_example_catalog_contains_exactly_7_"
                "eligible_examples_per_family": True,
                "resample_within_each_family_with_replacement": True,
                "one_draw_catalog_reused_for_every_pair_direction_control_"
                "and_topology_calculation": True,
                "mutual_k2_recomputed_per_resample": True,
                "minimum_mutual_frequency": BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY,
                "invalid_or_degenerate_resamples_fail_closed": True,
            },
            "leave_one_family_out": {
                "family_count": CONFIRMATION_FAMILY_COUNT,
                "replicate_count": CONFIRMATION_FAMILY_COUNT,
                "mutual_k2_recomputed_per_replicate": True,
                "minimum_mutual_count": LOFO_MINIMUM_MUTUAL_COUNT,
            },
            "long_context_stratum_repeats_point_topology_and_error_gates": True,
            "long_context_stratum_is_a_veto_and_cannot_nominate_edges": True,
            "every_gate_is_a_strict_conjunction": True,
            "partial_gate_pass_never_confirms_an_edge": True,
        },
        "response_ablation_validation_only": {
            "role": "D_response_validation",
            "opened_after_chart_graph_freeze": True,
            "whole_model_native_location_interventions_required": True,
            "restricted_choice_or_terminal_response_ABI_may_be_used": True,
            "response_measurements_may_label_chart_edges": [
                "downstream_response_supported",
                "downstream_response_mismatched",
                "downstream_response_indeterminate",
            ],
            "response_measurements_may_nominate_edges": False,
            "response_measurements_may_remove_chart_edges": False,
            "response_measurements_may_rerank_chart_edges": False,
            "response_only_neighbor_claim_forbidden": True,
            "compression_followup_requires_downstream_response_support": True,
        },
        "required_worker_receipts": {
            "pre_model_static_preflight": [
                "setup_sha256",
                "all_predecessor_file_sha256s",
                "event_catalog_sha256",
                "carrier_site_catalog_sha256",
                "mode_catalog_sha256",
                "fresh_panel_source_manifest_and_receipt_file_sha256s",
                "all_role_firewall_catalog_sha256s",
            ],
            "post_model_lineage": [
                "source_model_and_execution_fingerprints",
                "exact_event_and_site_runtime_catalog_sha256s",
                "per_mode_context_value_gradient_and_contribution_site_receipts",
                "shared_chart_atlas_artifact_sha256",
                "common_coordinate_signature_sha256",
                "per_role_panel_catalog_sha256",
                "per_role_aligned_row_catalog_sha256",
                "per_mode_finite_energy_catalog_sha256",
                "attention_cross_token_route_catalog_sha256s",
                "lexical_egress_readout_pullback_catalog_sha256",
                "ordered_transport_artifact_catalog_sha256",
            ],
            "graph_receipts": [
                "B_frozen_candidate_catalog_sha256",
                "C_confirmed_edge_catalog_sha256",
                "B_C_exact_7_per_family_primary_eligibility_catalog_sha256",
                "short96_boundary511_513_transition514_and_hard_long_"
                "non_authorizing_subset_receipts",
                "per_edge_both_directional_family_distances",
                "bootstrap_draw_and_frequency_catalog_sha256",
                "row_shuffle_permutation_catalog_sha256",
                "LOFO_receipts",
                "short_boundary_long_stratum_receipts",
                "D_response_validation_labels_if_D_was_opened",
            ],
            "raw_prompts_token_ids_full_width_vectors_gradients_and_routes_"
            "must_not_be_embedded_in_summary_report": True,
        },
        "claim_taxonomy": {
            "allowed_only_after_C_all_gates": {
                "chart_native_neighbor_edge": True,
                "stable_shared_chart_geometry": True,
                "cross_native_ABI_chart_affinity": True,
            },
            "allowed_only_after_separate_D_support": {
                "downstream_response_supported_chart_neighbor": True,
            },
            "diagnostic_only": {
                "native_ABI_partition": True,
                "operator_kind_or_depth_enrichment": True,
                "attention_route_similarity": True,
                "connected_component": True,
                "maximal_clique": True,
                "new_C_neighbor": True,
                "raw_weight_similarity": True,
            },
            "never_authorized_by_this_protocol": {
                "cluster_or_generator_family": True,
                "native_ABI_interchangeability": True,
                "causal_interchangeability": True,
                "parameter_sharing": True,
                "merge_or_deletion": True,
                "compression": True,
                "deployment_or_latency_improvement": True,
                "attention_route_causality_beyond_exact_decomposition": True,
            },
        },
        "implementation_status": {
            "worker_entrypoint_implemented_and_source_pinned": True,
            "concrete_runtime_API_implemented_and_source_pinned": True,
            "conservative_first_party_functional_closure_source_pinned": True,
            "setup_module_is_reviewed_non_self_hashed_trust_anchor": True,
            "package_eager_import_occurs_before_static_preflight": True,
            "package_eager_import_is_outside_preexecution_authentication_boundary": (
                True
            ),
            "whole_process_or_every_executable_tamper_proof_claimed": False,
            "implemented_runtime_surfaces": [
                "shared_global_full_rank_SPD_chart_atlas",
                "per_mode_generalized_contribution_capture",
                "attention_source_to_target_route_chart_receipts",
                "attention_operator_output_to_postnorm_delta_bridge_receipts",
                "lexical_egress_softcap_readout_pullback_chart",
                "chart_native_bidirectional_transport_graph_worker",
            ],
        },
        "claim_boundary": {
            "setup_artifact_contains_no_measurement": True,
            "fresh_panels_exist": True,
            "fresh_panels_are_frozen_and_byte_authenticated": True,
            "chart_native_neighbor_measured": False,
            "attention_cross_token_route_measured": False,
            "lexical_egress_readout_chart_measured": False,
            "neighbor_edge_claimed": False,
            "response_validation_claimed": False,
            "cluster_family_merge_or_compression_claimed": False,
        },
    }
    return state


def build_setup() -> dict[str, object]:
    """Build the canonical setup without opening any external artifact."""

    state = _build_state()
    state["setup_sha256"] = setup_sha256(state)
    _validate_internal_contract(state)
    return state


def _validate_internal_contract(value: Mapping[str, object]) -> None:
    if value.get("schema") != GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_SCHEMA:
        raise ValueError("chart-native neighbor schema drifted")
    events = value["exact_execution_lineage"]["ordered_events"]
    sites = value["carrier_observation_lineage"]["ordered_sites"]
    modes = value["logical_mode_ontology"]["ordered_modes"]
    if len(events) != EVENT_COUNT or [row["event_index"] for row in events] != list(
        range(EVENT_COUNT)
    ):
        raise ValueError("exact 39-event lineage drifted")
    if len(sites) != CARRIER_SITE_COUNT or len(
        {row["site_id"] for row in sites}
    ) != CARRIER_SITE_COUNT:
        raise ValueError("exact 73-site lineage drifted")
    if len(modes) != LOGICAL_MODE_COUNT or len(
        {row["mode_id"] for row in modes}
    ) != LOGICAL_MODE_COUNT:
        raise ValueError("exact 38-mode ontology drifted")
    if any(row["comparison_ABI"] != COMMON_COMPARISON_ABI for row in modes):
        raise ValueError("common chart-energy ABI drifted")
    if sum(row["requires_cross_token_route_receipt"] for row in modes) != 18:
        raise ValueError("attention route requirements drifted")
    if value.get("frozen_runtime_qualification") != {
        "authenticated_present": True,
        "path": str(PINNED_RUNTIME_QUALIFICATION_PATH),
        "file_sha256": PINNED_RUNTIME_QUALIFICATION_FILE_SHA256,
        "qualification_receipt_sha256": (
            PINNED_RUNTIME_QUALIFICATION_RECEIPT_SHA256
        ),
        "panel_corpus_bytes_authenticated_opaque": True,
        "prompt_text_decoded_or_semantically_opened": False,
        "panel_semantic_access_performed": False,
        "tokenizer_and_model_loaded_only_after_static_preflight": True,
        "qualifies_compiled_runtime_before_A_access": True,
    }:
        raise ValueError("runtime qualification pin drifted")
    implementations = value["frozen_worker_implementation_sources"]
    if (
        not isinstance(implementations, Mapping)
        or implementations != PINNED_WORKER_IMPLEMENTATION_SOURCES
        or "gemma3_chart_native_neighbor_setup" in implementations
        or len(implementations) != 60
        or implementations.get("package_eager_import_boundary", {}).get(
            "path"
        )
        != "src/fisher_graph/__init__.py"
        or any(
            receipt.get("path")
            == "src/fisher_graph/gemma3_chart_native_neighbor_setup.py"
            for receipt in implementations.values()
        )
        or any(
            receipt.get("path")
            == "src/fisher_graph/gemma3_unified_event_chart_sidecar.py"
            for receipt in implementations.values()
        )
    ):
        raise ValueError("worker implementation source catalog drifted")
    implementation_status = value["implementation_status"]
    if (
        not isinstance(implementation_status, Mapping)
        or implementation_status.get(
            "worker_entrypoint_implemented_and_source_pinned"
        )
        is not True
        or implementation_status.get(
            "concrete_runtime_API_implemented_and_source_pinned"
        )
        is not True
        or implementation_status.get(
            "conservative_first_party_functional_closure_source_pinned"
        )
        is not True
        or implementation_status.get(
            "setup_module_is_reviewed_non_self_hashed_trust_anchor"
        )
        is not True
        or implementation_status.get(
            "package_eager_import_occurs_before_static_preflight"
        )
        is not True
        or implementation_status.get(
            "package_eager_import_is_outside_preexecution_authentication_boundary"
        )
        is not True
        or implementation_status.get(
            "whole_process_or_every_executable_tamper_proof_claimed"
        )
        is not False
    ):
        raise ValueError("worker implementation status boundary drifted")
    for name, receipt in implementations.items():
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"path", "file_sha256"}
            or not isinstance(receipt["path"], str)
            or not receipt["path"]
            or not isinstance(receipt["file_sha256"], str)
            or len(receipt["file_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in receipt["file_sha256"]
            )
        ):
            raise ValueError(f"implementation source {name} receipt drifted")
    fresh = value["frozen_fresh_panel_lineage"]
    if (
        fresh["examples_per_family"] != EXAMPLES_PER_FAMILY
        or fresh["primary_eligible_examples_per_family_B_C"]
        != PRIMARY_ELIGIBLE_EXAMPLES_PER_FAMILY
        or fresh["ordered_target_token_counts"] != list(TARGET_TOKEN_COUNTS)
        or fresh["primary_eligible_token_counts_B_C"]
        != list(PRIMARY_ELIGIBLE_TOKEN_COUNTS)
        or set(fresh["role_files"]) != set(PINNED_FRESH_ROLE_FILE_SHA256S)
    ):
        raise ValueError("frozen fresh-panel eligibility lineage drifted")
    if any(
        fresh["role_files"][role]["file_sha256"] != digest
        for role, digest in PINNED_FRESH_ROLE_FILE_SHA256S.items()
    ):
        raise ValueError("frozen fresh-panel role file pins drifted")
    streaming = value["shared_global_chart_atlas"][
        "production_streaming_execution"
    ]
    if (
        streaming["fixed_row_chunk_size"] != 256
        or streaming["fit_pool_row_count"] != 939_816
        or not streaming["full_context_pool_torch_cat_forbidden"]
        or not streaming[
            "broadcast_N_by_8_by_640_distance_tensor_forbidden"
        ]
    ):
        raise ValueError("production streaming atlas contract drifted")
    route = value["attention_cross_token_route_chart"][
        "production_factorized_aggregation"
    ]
    if (
        route["kernel_id"]
        != "factorized_attention_route_energy.p_u_g.float64_chunked.v1"
        or route["materialized_h640_route_coordinate_scalar_count"] != 0
        or route["stored_native_probability_bytes_preserved_without_renormalization"]
        is not True
        or route["probability_simplex_audit_accumulation_dtype"]
        != "torch.float64"
        or route["float32_probability_sum_then_float64_cast_forbidden"]
        is not True
        or route["probability_simplex_fixed_tolerance_forbidden"] is not True
        or route["source_softmax_dtype"] != "torch.float32"
        or route["source_softmax_device"] != "cpu"
        or route["probability_simplex_forward_error_formula_id"]
        != (
            "float32_softmax_simplex."
            "gamma32_support_plus_2_gamma64_audit_plus_ftz.v1"
        )
        or route["probability_simplex_support_derivation_id"]
        != (
            "query_key_valid_masks_logical_causal_optional_strict_window.v1"
        )
        or route["probability_simplex_support_minimum"] != 1
        or route["probability_simplex_support_maximum"] != 768
        or route["probability_simplex_float32_unit_roundoff_power"] != -24
        or route["probability_simplex_float64_unit_roundoff_power"] != -53
        or route["probability_simplex_float32_FTZ_safety_power"] != -126
        or route["probability_simplex_bound_formula"]
        != (
            "gamma32(k+2)+gamma64(k-1)*(1+gamma32(k+2))+k*2^-126"
        )
        or route["finite_probability_in_closed_unit_interval_required"]
        is not True
        or route[
            "masked_and_invalid_route_probabilities_exact_zero_required"
        ]
        is not True
        or route[
            "native_dense_support_equals_authenticated_logical_support_"
            "at_capture_required"
        ]
        is not True
        or route[
            "per_snapshot_and_per_example_simplex_audit_receipt_required"
        ]
        is not True
        or route["frozen_kernel_receipt_sha256"]
        != "228433c4960c61660e9e0cb6e3f08d1a2a0fbf54f6bd1f9f6175d8beac23020e"
    ):
        raise ValueError("factorized attention route contract drifted")
    bootstrap = value["fresh_confirmation_gates"][
        "within_family_bootstrap"
    ]
    if bootstrap["draws_per_family"] != PRIMARY_ELIGIBLE_EXAMPLES_PER_FAMILY:
        raise ValueError("primary bootstrap eligibility drifted")
    lexical_kernel = value["lexical_egress_readout_chart"][
        "chunked_full_vocabulary_NLL_VJP"
    ]
    if lexical_kernel != {
        "kernel_id": "gemma3_full_vocab_softcap_shifted_nll_vjp.chunked.v1",
        "kernel_receipt_sha256": (
            "e79c953bd99483d87e8c09406388deecf2474a9dc20c468c0fd68c38d9deffdb"
        ),
        "prediction_position_chunk_size": 32,
        "vocabulary_chunk_size": 4096,
        "float64_reduction_required": True,
        "first_A_dense_parity_selected_prediction_rows": 2,
        "dense_reference_dtype": "torch.float64",
        "float32_dense_reference_forbidden": True,
        "parity_absolute_tolerance": 2.0e-5,
        "parity_relative_tolerance": 2.0e-5,
    }:
        raise ValueError("chunked lexical NLL/parity contract drifted")
    egress = modes[-1]
    if egress["context_site_id"] != "terminal.rmsnorm.output" or egress[
        "score_gradient_site_id"
    ] != "lexical.egress.readout_pullback":
        raise ValueError("lexical egress pullback surface drifted")
    if value["exact_execution_lineage"]["event_catalog_sha256"] != _sha256(
        _EVENT_CATALOG_HASH_DOMAIN, events
    ):
        raise ValueError("event catalog digest drifted")
    if value["carrier_observation_lineage"]["site_catalog_sha256"] != _sha256(
        _SITE_CATALOG_HASH_DOMAIN, sites
    ):
        raise ValueError("site catalog digest drifted")
    if value["logical_mode_ontology"]["mode_catalog_sha256"] != _sha256(
        _MODE_CATALOG_HASH_DOMAIN, modes
    ):
        raise ValueError("mode catalog digest drifted")
    if value.get("setup_sha256") != setup_sha256(value):
        raise ValueError("chart-native neighbor setup digest drifted")


def validate_setup(value: Mapping[str, object]) -> dict[str, object]:
    """Fail closed unless ``value`` is byte-semantically the frozen setup."""

    if not isinstance(value, Mapping):
        raise TypeError("chart-native neighbor setup must be a mapping")
    try:
        observed = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("chart-native neighbor setup is not canonical JSON") from error
    _validate_internal_contract(observed)
    expected = build_setup()
    if observed != expected:
        raise ValueError("chart-native neighbor setup does not match frozen v1")
    return observed


def _strict_json_loads(raw: bytes) -> object:
    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )


def _read_regular_setup(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError("chart-native neighbor setup is inaccessible") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("chart-native neighbor setup must be a regular file")
    if before.st_size > 16 * 1024 * 1024:
        raise ValueError("chart-native neighbor setup exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError("chart-native neighbor setup changed during open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("chart-native neighbor setup was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError("chart-native neighbor setup changed during read")
    return b"".join(chunks)


def load_setup(source: Path | str) -> dict[str, object]:
    path = Path(source)
    try:
        payload = _strict_json_loads(_read_regular_setup(path))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("chart-native neighbor setup is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("chart-native neighbor setup must be a JSON object")
    return validate_setup(payload)


def write_setup(destination: Path | str = DEFAULT_OUTPUT) -> dict[str, object]:
    path = Path(destination)
    payload = build_setup()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        os.unlink(temporary)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if load_setup(path) != payload:
            raise RuntimeError("strict setup reload drifted after publication")
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        if published:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            try:
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass
        raise
    return payload


def check_setup(source: Path | str = DEFAULT_OUTPUT) -> dict[str, object]:
    path = Path(source)
    payload = load_setup(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED_SETUP_FILE_SHA256:
        raise ValueError("chart-native neighbor setup file bytes drifted")
    if payload["setup_sha256"] != PINNED_SETUP_SHA256:
        raise ValueError("chart-native neighbor setup public digest drifted")
    return payload


# Descriptive aliases keep call sites clear while the short public API remains
# convenient for the standalone setup shim.
build_gemma3_chart_native_neighbor_setup = build_setup
validate_gemma3_chart_native_neighbor_setup = validate_setup
load_gemma3_chart_native_neighbor_setup = load_setup
write_gemma3_chart_native_neighbor_setup = write_setup


@dataclass(frozen=True, slots=True)
class Gemma3ChartNativeNeighborSetup:
    state: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_setup(self.state)

    @classmethod
    def build(cls) -> "Gemma3ChartNativeNeighborSetup":
        return cls(build_setup())

    def to_dict(self) -> dict[str, object]:
        return json.loads(canonical_json_bytes(self.state))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="authenticate an existing canonical setup instead of writing",
    )
    args = parser.parse_args(argv)
    if args.check:
        check_setup(args.output)
    else:
        write_setup(args.output)
    return 0


__all__ = [
    "BOOTSTRAP_CATALOG_DOMAIN",
    "BOOTSTRAP_DRAW_DOMAIN",
    "BOOTSTRAP_DRAW_PROTOCOL",
    "BOOTSTRAP_MAXIMUM_DEGENERATE_FRACTION",
    "BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY",
    "BOOTSTRAP_RESAMPLE_COUNT",
    "CARRIER_SITE_COUNT",
    "CHART_COUNT",
    "CHART_RANK",
    "COMMON_COMPARISON_ABI",
    "DEFAULT_OUTPUT",
    "EVENT_COUNT",
    "GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_FORMAT_VERSION",
    "GEMMA3_CHART_NATIVE_NEIGHBOR_SETUP_SCHEMA",
    "Gemma3ChartNativeNeighborSetup",
    "HIDDEN_WIDTH",
    "LOGICAL_MODE_COUNT",
    "MAX_DIRECTIONAL_NORMALIZED_RESIDUAL",
    "MUTUAL_K",
    "PINNED_RUNTIME_QUALIFICATION_FILE_SHA256",
    "PINNED_RUNTIME_QUALIFICATION_PATH",
    "PINNED_RUNTIME_QUALIFICATION_RECEIPT_SHA256",
    "PINNED_SETUP_FILE_SHA256",
    "PINNED_SETUP_SHA256",
    "ROW_SHUFFLE_CATALOG_DOMAIN",
    "ROW_SHUFFLE_DOMAIN",
    "ROW_SHUFFLE_SEED",
    "SETUP_ID",
    "build_gemma3_chart_native_neighbor_setup",
    "build_setup",
    "canonical_json_bytes",
    "check_setup",
    "load_gemma3_chart_native_neighbor_setup",
    "load_setup",
    "main",
    "setup_sha256",
    "validate_gemma3_chart_native_neighbor_setup",
    "validate_setup",
    "write_gemma3_chart_native_neighbor_setup",
    "write_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
