"""V20n nested continuous simplex-shrinkage campaign for Gemma 3.

V20n authenticates the completed V20m report and all eight V20m folds before
model construction.  It first reproduces V20m's nineteen-response conditional
inner-LOFO selection, then fits one continuous shrinkage scalar for the
selected response:

``q_lambda(z) = (1-lambda*u*z^2)*tanh(r*z) + lambda*v*z^2``.

For each outer fold, exact lambda-zero and lambda-one inner scores are reused
from the authenticated V20m response screen.  All seven lambda-half providers
are frozen before any half score; their family-equal score and the endpoint
scores define the precommitted quadratic proposal.  All seven proposed-vertex
providers are then frozen before any exact vertex score.  Final selection uses
only the exact objectives at zero, the frozen vertex, and one.  The genuinely
endpoint-disjoint outer family remains unopened until the selected response,
lambda, all nine outer providers, and all traces are frozen.  There is no
all-eight refit, Calibration-B access, compression claim, or serving authority.
Reports and resumable fold fragments contain scalar/hash evidence only and are
mode 0600.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_finite_joint_pedal_development as _v19
from . import gemma3_l3_l4_complete_h4_finite_microstep_nested_validation as _v20b
from . import gemma3_l3_l4_complete_h4_finite_microstep_preflight as _v20a
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development
    as _v20g,
)
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_reflection_nested_development
    as _v20i,
)
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_confidence_nested_development
    as _v20j,
)
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_log_response_nested_development
    as _v20k,
)
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_signed_stack_nested_development
    as _v20l,
)
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_simplex_response_nested_development
    as _v20m,
)
from . import complete_h4_fisher_soft_polarity_reflection_fit as _reflection
from . import complete_h4_fisher_soft_polarity_simplex_response_fit as _simplex_response_fit
from . import complete_h4_fisher_soft_polarity_simplex_shrinkage_fit as _shrinkage_fit
from .complete_h4_fisher_conditional_residual import _training_parent_modal
from .complete_h4_fisher_soft_polarity import (
    build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control,
)
from .complete_h4_fisher_soft_polarity_simplex_response import (
    AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_simplex_response,
    fisher_soft_polarity_simplex_response_box_certificate,
    fisher_soft_polarity_simplex_response_direction_sha256,
    validate_fisher_soft_polarity_simplex_response_provider_evidence,
)
from .complete_h4_fisher_soft_polarity_simplex_shrinkage import (
    AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider,
    FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256,
    build_autonomous_complete_h4_fisher_soft_polarity_simplex_shrinkage,
    fisher_soft_polarity_simplex_shrinkage_box_certificate,
    fisher_soft_polarity_simplex_shrinkage_direction_sha256,
    validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_soft_polarity_simplex_shrinkage_nested_development",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-soft-polarity-simplex-shrinkage-nested-"
    "r16-k256-a-fit16-dev-v20n.json"
)

_V20M_OUTPUT = _v20m.DEFAULT_OUTPUT
_V20M_LOGICAL_SHA256 = (
    "973e70987b54b9b898766e320c34d182efdbe6c27b107a3de01c8498edc38bcb"
)
_V20M_FILE_SHA256 = (
    "50948fe72bfa9cbd6394b1b57ec0f4fe2e2cf3f8e6f2c64ac168e580b60d2e86"
)
_V20M_SOURCE_SHA256 = (
    "88d1f8c9e37de2a5421c8cb81232ba6cdc040beb400ded91397b8e6c4c01536b"
)
_V20M_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "44ebd3364d9319e1075016b3e824d10fea2fac098bfc4e59af50f1fb4d1f2246"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "e2888f9af7e5eba6fbb41738ed75b8259368a538c9c69a8bbc39db239fc3264a"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "0ef11ca449c5fe649d5f122e4b82ef9a3bca99bc994dd7de357b5b1ea32de867"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "04b8abb3c00b54d498c69011243342ea075c6bd86bbc5c2c16d825ccbc57f864"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "bc08094aa86c158e17f381c4efce67a47ec3b8ec87d0e98dedf82da4ee89b8c1"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "4b34f6e818522cbc5b4545be5066ddee1b61691caeeeaf4ab923438055c55450"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "6cd0b9014d7f48d599a08abe97f30de01dde7ea379bb4510b8cad2ea0a269db3"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "4518e71729481168858c82293f62418d359ec4609eb4197aca80c527ff1a3d3b"
    ),
}

_V20L_OUTPUT = _v20l.DEFAULT_OUTPUT
_V20L_LOGICAL_SHA256 = (
    "803fc80586266936ee67601e85caee7b4357e5c3c7923805c0615b6ce96ad849"
)
_V20L_FILE_SHA256 = (
    "71f5d41b4bef4e2c55d674e105b2093a0e168d42ffdebf5f7fbe2c4fb57a4d38"
)
_V20L_SOURCE_SHA256 = (
    "d43273ea0a4555be88bb8afef20bff410dbbc53c148c063b148c600a78513094"
)
_V20L_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "bbeb5911d87ffb10c4a44c4ee3ff73ee1e818c615fe82946229c4c939d52ddb9"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "6a5b837986b66494baea3940efb44a0f4cf182a00a2076ee7026d2ea17cedc30"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "dddf37ff1bb9e465a28efc6e09b3d9ff75b8932804c8a3e955ba227cd9547892"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "8bc13eaceb346076910b8cb4dbdd693038378e94cd9e2e2d76c1c2d1725d4d05"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "89fa826b9697c8198514b75d90e5413ffe9e2f6626335ee0c78279d932c0ab0f"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "25cfc3d28a61efd81475c81a586ed04bebdef36223e9f4ac730d19461a0f7dc3"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "d0f9d05161a28a2af7158adb71bd88bc771a608c598dee608080988c1594643c"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "0b295253e2f3e7593e77388da55c46970da2d50a99a0c66fc8fc0f12702e2778"
    ),
}

_V20K_OUTPUT = _v20k.DEFAULT_OUTPUT
_V20K_LOGICAL_SHA256 = (
    "982f29baf5df78db7171d11d1bc56862333aa7967ed1a5f5b66a84b504f61deb"
)
_V20K_FILE_SHA256 = (
    "33b4f95113a92309a304b2cf8c5e6aa6ae7cee3d3868be1618a83e17cbfdc6f4"
)
_V20K_SOURCE_SHA256 = (
    "00acbbec07cf9c1902f0be24f1cc76f0b5fdcc3d80492bc5a32bbe3763dd6350"
)
_V20K_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "87d431a371b2070704b84de95fd8dfba0782e23847145ea38ebe21de79d32144"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "f579415119f5d48f47943eeb65907c22088c2a5cc5b6dff03b6ca3be40aeaf4b"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "22810de59044cbc72572a9ff93653a4f5f9b8eed4973c6b9fafe056ba6d1913f"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "8be49c8de233c99bf3f6b3f81ec645ccdf0056c2af7c9029136142b7e3d64752"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "11529eac81fd5c61a1d8fcfa3645c77dd5cd313b86c0ac7df2a3696792ced34b"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "3407ef07c90eea0b31677870214335f1405c8669a73fe49eb24b30c0dc3e939a"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "7ca6baf4d96e8de436d4971aa286429f97994ba28cf61cc94dd6188ecdfeee83"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "0e6333c543dab264ccfc6e7d2ee34226fa6bb92a06cacd69f544a5a431c8d7c6"
    ),
}

_V20J_OUTPUT = _v20j.DEFAULT_OUTPUT
_V20J_LOGICAL_SHA256 = (
    "2dd3be4a6bf3a30596bdecbf760d8e8fb6c518cdeaf87918000be2b1e3cfd28b"
)
_V20J_FILE_SHA256 = (
    "dea76949766ee93c48154f398fb3ac776287354bbcd628fc699cd6975537c2ce"
)
_V20J_SOURCE_SHA256 = (
    "eb9ed4076957e3a5986dacfe9a1b5a3886dabc359a015b20a41ed4f19916c36d"
)
_V20J_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "04d662e3e8d62e4a67a72f95bd2ffa8c00c8ad5cfe0825d186f33d8eef4c9c03"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "cd93bb8d652b53b3c56440e816a0905ae2bfcf962e6e66a5bd77402b18b66ecb"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "5c539e4a674cf122d957444b266ed6ff79691dc9dd6745bb6547ca293f90d60b"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "508e03d845615e0d3217a0a658238328a66c50ea20094e7b5f0fa2704c7dc3c8"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "a81d998dbbf3f0e15dbcc8f07c6f1da5c38a2631cb39c84aa9b444e4617918f2"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "a47dd589d3d7fde7aee42f4aae9a619a5ae7dcdd41db6b8196a21adf379e52d3"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "329233151edb0f813d6821009fd304f850f9dbbb289c5fb5c634c5b15cfc0e98"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "761e8dd07097962103304be364e98a584f8b1bcb7f3d2f7220b964b60e41f9b0"
    ),
}

_V20I_OUTPUT = _v20i.DEFAULT_OUTPUT
_V20I_LOGICAL_SHA256 = (
    "14618913ff620c67000213aa765f45d8587e8ba4dcb0f8dcb634d7ab3490ecdd"
)
_V20I_FILE_SHA256 = (
    "cc044795b3ad3e243eb2b091c17fc491aedbb8f905f92000c4c3f95002c8f356"
)
_V20I_SOURCE_SHA256 = (
    "39f6808053e8121294414db5807a7e3716a35c0b77404157eb0c4b282ca593a1"
)
_V20I_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "e2cd0922a260c198a3cf1fcc27ac5acb1029c7bffd5c60ff9044ff61a1a14324"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "aa19f13d016d410d6716fc2837b6285ed3e556fe34a400870d30411b80814891"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "1dc93c8a8ad1ebeb8a73d98ab77601503f048ab64ec9f928eb1b0b68c8973da3"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "5def19b3c650ed171a6162677e7c0f18ce77a91bfdcd1be561520e40e0c86818"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "8af8692ab94f50d306af7adc3048a53415301da7010537514562b63ab880b990"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "3f467552e75b33d6eeb79c4c7a1393bc390760b533a18e08d529a3acae761fc8"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "f7182ed7b8202443d06cfa87d5d15b6ffc4e947c63a7d5a6a4204647c9165ef0"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "f1776c84fa0b0f275616bce60346bca90aeff708f07bc7658be52c6e4e97ff46"
    ),
}

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_soft_polarity_simplex_shrinkage_nested.v20n"
)
_FOLD_SCHEMA = (
    "fisher_graph.complete_h4_soft_polarity_simplex_shrinkage_nested_outer_fold.v20n"
)
_FORMAT_VERSION = 30
_REPORT_DOMAIN = b"fisher-graph:soft-polarity-simplex_shrinkage-nested-report:v20n\0"
_SOURCE_DOMAIN = b"fisher-graph:soft-polarity-simplex_shrinkage-nested-source:v20n\0"
_FOLD_DOMAIN = b"fisher-graph:soft-polarity-simplex_shrinkage-nested-fold:v20n\0"
_INNER_FIT_DOMAIN = b"fisher-graph:soft-polarity-simplex_shrinkage-inner-fit:v20n\0"
_INNER_MANIFEST_DOMAIN = (
    b"fisher-graph:soft-polarity-simplex_shrinkage-inner-manifest:v20n\0"
)
_INNER_EXECUTION_DOMAIN = (
    b"fisher-graph:soft-polarity-simplex_shrinkage-inner-execution:v20n\0"
)
_RESPONSE_SELECTION_DOMAIN = (
    b"fisher-graph:soft-polarity-simplex_shrinkage-response-selection:v20n\0"
)
_SHRINKAGE_SELECTION_DOMAIN = (
    b"fisher-graph:soft-polarity-simplex_shrinkage-lambda-selection:v20n\0"
)
_OUTER_MANIFEST_DOMAIN = (
    b"fisher-graph:soft-polarity-simplex_shrinkage-outer-manifest:v20n\0"
)
_OUTER_EXECUTION_DOMAIN = (
    b"fisher-graph:soft-polarity-simplex_shrinkage-outer-execution:v20n\0"
)
_PROVIDER_DOMAIN = b"fisher-graph:soft-polarity-simplex_shrinkage-provider:v20n\0"
_TRACE_DOMAIN = b"fisher-graph:soft-polarity-simplex_shrinkage-trace:v20n\0"
_DECISION_DOMAIN = b"fisher-graph:soft-polarity-simplex_shrinkage-decision:v20n\0"

_FAMILY_COUNT = 8
_PROMPTS_PER_FAMILY = 2
_INNER_FAMILY_COUNT = 7
_INNER_TRAINING_FAMILY_COUNT = 6
_CONDITIONAL_RANK = 16
_SHRINKAGE_ANCHOR_LAMBDAS = (0.0, 0.5, 1.0)
_RESPONSES: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0 / 8.0, 0.0, 0.0),
    (1.0 / 8.0, 1.0 / 8.0, -1.0 / 8.0),
    (1.0 / 8.0, 1.0 / 8.0, -1.0 / 16.0),
    (1.0 / 8.0, 1.0 / 8.0, 0.0),
    (1.0 / 8.0, 1.0 / 8.0, 1.0 / 16.0),
    (1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0),
    (1.0 / 8.0, 1.0 / 4.0, -1.0 / 8.0),
    (1.0 / 8.0, 1.0 / 4.0, 0.0),
    (1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0),
    (1.0 / 4.0, 0.0, 0.0),
    (1.0 / 4.0, 1.0 / 8.0, -1.0 / 8.0),
    (1.0 / 4.0, 1.0 / 8.0, -1.0 / 16.0),
    (1.0 / 4.0, 1.0 / 8.0, 0.0),
    (1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0),
    (1.0 / 4.0, 1.0 / 8.0, 1.0 / 8.0),
    (1.0 / 4.0, 1.0 / 4.0, -1.0 / 8.0),
    (1.0 / 4.0, 1.0 / 4.0, 0.0),
    (1.0 / 4.0, 1.0 / 4.0, 1.0 / 8.0),
)
_RESPONSE_KEYS = tuple(
    f"radius={radius.hex()};u={u.hex()};v={v.hex()}"
    for radius, u, v in _RESPONSES
)
if _RESPONSES != tuple(_simplex_response_fit.SOFT_POLARITY_SIMPLEX_RESPONSE_LADDER):
    raise RuntimeError("V20n runner response ladder differs from core protocol")
_SIMPLEX_RESPONSE_LADDER_RECEIPT = (
    _simplex_response_fit.build_soft_polarity_simplex_response_ladder_receipt()
)
_SIMPLEX_RESPONSE_LADDER_RECEIPT_SHA256 = str(
    _SIMPLEX_RESPONSE_LADDER_RECEIPT["artifact_sha256"]
)
_ARMS = (
    "base",
    "fixed_plus",
    "fixed_minus",
    "matched_linear_reflected",
    "matched_v20l_boundary_reflected",
    "same_simplex_response_unreflected",
    "simplex_shrinkage_reflected",
    "simplex_response_reflected_exact_mirror",
    "matched_v20m_simplex_reflected",
)
_PRIMARY_ARM = "simplex_shrinkage_reflected"
_PROVIDER_RECEIPT_KEYS = frozenset(
    {
        "role",
        "response",
        "response_key",
        "radius",
        "u",
        "v",
        "direction",
        "direction_box_corner_scores",
        "box_certificate",
        "provider_artifact_sha256",
        "provider_metadata",
        "provider_metadata_sha256",
        "provider_payload",
        "transfer_protocol_sha256",
        "transfer_evidence_sha256",
        "rank",
        "conditional_rank",
        "prepared_float_scalar_count",
        "logical_macs_per_token_upper_bound",
        "analysis_only",
        "raw_provider_tensors_serialized",
        "artifact_sha256",
    }
)
_SHRINKAGE_PROVIDER_RECEIPT_KEYS = frozenset(
    {
        "role",
        "source_response",
        "source_response_key",
        "lambda",
        "lambda_hex",
        "effective_response",
        "direction",
        "direction_box_corner_scores",
        "box_certificate",
        "provider_artifact_sha256",
        "runtime_provider_artifact_sha256",
        "lineage_wrapper_not_inference_executor",
        "provider_metadata",
        "provider_metadata_sha256",
        "provider_payload",
        "transfer_protocol_sha256",
        "transfer_evidence_sha256",
        "rank",
        "conditional_rank",
        "prepared_float_scalar_count",
        "logical_macs_per_token_upper_bound",
        "analysis_only",
        "raw_provider_tensors_serialized",
        "artifact_sha256",
    }
)

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "v20n_nested_continuous_simplex_shrinkage_fit",
    "scientific_status": (
        "posthoc_after_v20m_reused_A16_development_hypothesis_only"
    ),
    "source": (
        "pinned_V20m_report_and_folds_with_complete_V20l_through_V20g_lineage"
    ),
    "outer_validation": "eight_leave_one_whole_development_family_out_folds",
    "inner_validation": (
        "seven_conditional_leave_one_outer_training_family_out_exact_"
        "simplex_response_folds_on_one_fixed_seven_family_endpoint"
    ),
    "inner_endpoint_scope": (
        "fixed_endpoint_fit_on_all_seven_outer_training_families_not_retrained_"
        "per_inner_fold"
    ),
    "inner_held_family_used_for_endpoint_fit": True,
    "inner_held_family_used_for_direction_or_reflection_fit": False,
    "inner_claim_scope": (
        "conditional_response_LOFO_not_fully_nested_model_cross_validation"
    ),
    "inner_direction": (
        "six_family_masked_Fisher_natural_direction_then_training_only_CVaR2_"
        "one_coordinate_reflection"
    ),
    "response_order": _RESPONSES,
    "simplex_response_ladder_receipt_sha256": _SIMPLEX_RESPONSE_LADDER_RECEIPT_SHA256,
    "simplex_response_ladder_receipt_constructed_before_provider_freeze": True,
    "response_selection": (
        "reproduce_V20m_nineteen_response_selection_before_lambda_fit"
    ),
    "simplex_response_formula": (
        "one_minus_u_times_z_squared_times_tanh_radius_z_plus_v_times_z_squared"
    ),
    "simplex_response_constraints": (
        "radius_in_zero_one_fourth_and_zero_less_equal_abs_v_less_equal_u_"
        "less_equal_one_fourth_finite_bounded_on_normalized_box"
    ),
    "shrinkage_anchor_lambdas": _SHRINKAGE_ANCHOR_LAMBDAS,
    "shrinkage_fit_protocol_sha256": (
        _shrinkage_fit.SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
    ),
    "simplex_shrinkage_provider_protocol_sha256": (
        FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256
    ),
    "shrinkage_fit": (
        "reuse_exact_lambda_zero_and_one_scores_then_freeze_and_score_all_"
        "seven_lambda_half_providers_then_freeze_quadratic_proposal_then_"
        "freeze_and_exact_score_all_seven_vertex_providers_then_select_exact_"
        "minimum_zero_vertex_one_ties_smaller_lambda_then_candidate_hash"
    ),
    "inner_freeze_barrier": (
        "V20m_133_provider_freeze_then_all_seven_half_provider_freeze_then_"
        "all_seven_vertex_provider_freeze_each_before_corresponding_score"
    ),
    "outer_arms": _ARMS,
    "outer_freeze_barrier": "all_nine_providers_and_traces_before_outer_capability",
    "primary_gate": (
        "candidate_macro_below_base_and_fixed_plus_and_at_least_six_of_eight_"
        "wins_against_each"
    ),
    "mechanism_gate": (
        "candidate_macro_below_same_simplex_response_unreflected_with_five_wins_"
        "below_exact_mirror_with_six_wins_below_matched_linear_with_five_wins_"
        "below_precommitted_matched_V20l_boundary_with_five_wins_and_below_"
        "matched_V20m_simplex_with_five_wins"
    ),
    "positive_changed_gate": (
        "all_selected_radius_positive_and_candidate_changed_exact"
    ),
    "simplex_response_evidence_gate": (
        "at_least_five_outer_folds_select_nontrivial_u_positive_response"
    ),
    "continuous_shrinkage_evidence_gate": (
        "at_least_five_outer_folds_select_lambda_strictly_between_zero_and_one"
    ),
    "interior_response_definition": "zero_less_than_abs_v_less_than_u",
    "matched_v20l_boundary_control": (
        "authenticated_prior_V20l_outer_selected_radius_and_signed_mix_mapped_"
        "to_radius_u_abs_signed_mix_v_signed_mix_and_exactly_reproduced"
    ),
    "matched_v20m_simplex_control": (
        "authenticated_prior_V20m_selected_simplex_response_exact_output_"
        "reproduction"
    ),
    "fixed_minus": "diagnostic_only",
    "matched_linear_reflected_control": True,
    "all_eight_final_refit_in_this_rung": False,
    "calibration_b_eligible": False,
    "serving_authorized": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
}
_RUNNER_PROTOCOL_SHA256 = _v14._sha256(_FIXED_PROTOCOL, domain=_SOURCE_DOMAIN)
_TRANSFER_PROTOCOL_SHA256 = _v14._sha256(
    {
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "reflection_fit_protocol_sha256": (
            _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
        ),
        "simplex_response_fit_protocol_sha256": (
            _simplex_response_fit.SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
        ),
        "simplex_shrinkage_fit_protocol_sha256": (
            _shrinkage_fit.SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
        ),
        "simplex_shrinkage_provider_protocol_sha256": (
            FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256
        ),
        "operation": "V20n_domain_separated_simplex_shrinkage_materialization",
        "held_rows_used": False,
    },
    domain=_PROVIDER_DOMAIN,
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _identifier(value: object, *, label: str) -> str:
    return _v20i._identifier(value, label=label)


def _sha(value: object, *, label: str) -> str:
    return _v20i._sha(value, label=label)


def _hashed(payload: Mapping[str, object], *, domain: bytes) -> dict[str, object]:
    return _v20i._hashed(payload, domain=domain)


def _validate_hashed(
    value: Mapping[str, object], *, domain: bytes, label: str
) -> Mapping[str, object]:
    return _v20i._validate_hashed(value, domain=domain, label=label)


def _simplex_parameters(value: object) -> tuple[float, float, float]:
    raw = _sequence(value, label="V20n simplex parameters")
    if len(raw) != 3:
        raise ValueError("V20n response must contain exactly radius, u, and v")
    if any(type(item) not in (int, float) for item in raw):
        raise ValueError("V20n response values must be JSON numbers")
    selected = tuple(float(item) for item in raw)
    if (
        any(
            not math.isfinite(item)
            or (item == 0.0 and math.copysign(1.0, item) < 0.0)
            for item in selected
        )
        or selected[0] < 0.0
        or selected[0] > 0.25
        or selected[1] < 0.0
        or selected[1] > 0.5
        or abs(selected[2]) > selected[1]
    ):
        raise ValueError("V20n simplex parameters are outside the certified domain")
    return selected[0], selected[1], selected[2]


def _response_tuple(value: object) -> tuple[float, float, float]:
    selected = _simplex_parameters(value)
    if selected not in _RESPONSES:
        raise ValueError("V20n response is outside the fixed ladder")
    return selected


def _parameters_key(value: object) -> str:
    radius, u, v = _simplex_parameters(value)
    return f"radius={radius.hex()};u={u.hex()};v={v.hex()}"


def _response_key(value: object) -> str:
    return _parameters_key(_response_tuple(value))


def _response_order(value: object) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        _response_tuple(item)
        for item in _sequence(value, label="V20n response order")
    )


def _validate_v20i_reflection_lineage(
    *,
    inner_receipt: Mapping[str, object],
    outer_reflection_fit: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
) -> None:
    """Bind every reused reflection decision to the pinned V20i authority."""

    inherited_outer = _mapping(
        authenticated_v20i_fold.get("outer_reflection_fit_receipt"),
        label="V20n inherited V20i outer reflection fit",
    )
    if _v14._canonical_json_bytes(outer_reflection_fit) != (
        _v14._canonical_json_bytes(inherited_outer)
    ):
        raise ValueError("V20n outer reflection lineage differs from pinned V20i")

    current_inner = _mapping(
        inner_receipt.get("inner_evidence_by_family"),
        label="V20n current inner reflection evidence",
    )
    inherited_inner_receipt = _mapping(
        authenticated_v20i_fold.get("inner_receipt"),
        label="V20n inherited V20i inner receipt",
    )
    inherited_inner = _mapping(
        inherited_inner_receipt.get("inner_evidence_by_family"),
        label="V20n inherited V20i inner reflection evidence",
    )
    if set(current_inner) != set(inherited_inner):
        raise ValueError("V20n inner reflection family lineage differs from V20i")
    for family in sorted(current_inner):
        current = _mapping(
            current_inner[family], label="V20n current inner reflection family"
        )
        inherited = _mapping(
            inherited_inner[family], label="V20n inherited inner reflection family"
        )
        for field in ("masked_direction_receipt", "reflection_fit_receipt"):
            if _v14._canonical_json_bytes(current.get(field)) != (
                _v14._canonical_json_bytes(inherited.get(field))
            ):
                raise ValueError(
                    f"V20n {field} lineage differs from pinned V20i"
                )


def _validate_output(path: Path | str) -> Path:
    destination = Path(path).resolve(strict=False)
    local_root = _LOCAL_ROOT.resolve(strict=False)
    protected = {
        candidate.resolve(strict=False)
        for candidate in (
            _v20g.DEFAULT_OUTPUT,
            _V20I_OUTPUT,
            _V20J_OUTPUT,
            _V20K_OUTPUT,
            _V20L_OUTPUT,
            _V20M_OUTPUT,
            *(
                _v20m._fold_path(_V20M_OUTPUT, family)
                for family in sorted(_V20M_FOLD_SHA256S)
            ),
            *(
                _v20l._fold_path(_V20L_OUTPUT, family)
                for family in sorted(_V20L_FOLD_SHA256S)
            ),
            *getattr(_v20g, "_PROTECTED_PREREQUISITE_PATHS", ()),
        )
    }
    if destination in protected:
        raise ValueError("V20n output must preserve immutable prerequisite artifacts")
    if destination.parent != local_root:
        raise ValueError("V20n output must remain directly under .local-runs")
    return destination


def _family_suffix(family_id: str) -> str:
    return _v20i._family_suffix(family_id)


def _fold_path(output: Path | str, family_id: str) -> Path:
    destination = _validate_output(output)
    return destination.with_name(
        f"{destination.stem}.fold-{_family_suffix(family_id)}.json"
    )


@dataclass(slots=True)
class _FoldLive:
    endpoint: _v20g._EndpointLive
    inner_receipt: dict[str, object]
    outer_reflection_fit: dict[str, object]
    response_selection: dict[str, object]
    shrinkage_selection: dict[str, object]
    provider_manifest: dict[str, object]
    held_evidence: dict[str, object]
    fold_receipt: dict[str, object]


def _load_prerequisites() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    """Authenticate V20m and its complete lineage before construction."""

    if (
        _V20M_LOGICAL_SHA256 is None
        or _V20M_FILE_SHA256 is None
        or _V20M_SOURCE_SHA256 is None
        or _V20M_FOLD_SHA256S is None
    ):
        raise RuntimeError(
            "V20n is fail-closed until the completed V20m report and all eight "
            "fold hashes are authenticated and pinned"
        )

    (
        prerequisite,
        authenticated_v20a_folds,
        v20g_report,
        authenticated_v20g_folds,
        v20i_report,
        authenticated_v20i_folds,
        v20l_report,
        authenticated_v20l_folds,
        v20m_source,
    ) = _v20m._load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"),
            label="V20n inherited panel receipt",
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20n inherited bridge binding",
    )
    if _v14._file_sha256(_V20M_OUTPUT) != _V20M_FILE_SHA256:
        raise RuntimeError("pinned V20m report file hash drifted")
    v20m_report = _v20m._load_existing_report(
        _V20M_OUTPUT,
        source=v20m_source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_v20g_folds=authenticated_v20g_folds,
        authenticated_v20i_folds=authenticated_v20i_folds,
        authenticated_v20l_folds=authenticated_v20l_folds,
    )
    observed_fold_hashes = {
        _identifier(family, label="V20n V20m fold family"): _sha(
            value, label="V20n V20m fold hash"
        )
        for family, value in _mapping(
            v20m_report.get("fold_fragment_sha256s_by_family"),
            label="V20n V20m fold hashes",
        ).items()
    }
    if (
        v20m_report.get("report_sha256") != _V20M_LOGICAL_SHA256
        or v20m_report.get("all_eight_outer_folds_completed") is not True
        or _mapping(
            v20m_report.get("decision"), label="V20n V20m decision"
        ).get("integrity_passed")
        is not True
        or v20m_report.get("final_refit") is not None
        or v20m_report.get("calibration_b_opened") is not False
        or _mapping(
            v20m_report.get("source_receipt"), label="V20n V20m source"
        ).get("artifact_sha256")
        != _V20M_SOURCE_SHA256
        or observed_fold_hashes != _V20M_FOLD_SHA256S
    ):
        raise RuntimeError("pinned V20m development authority differs")

    families = tuple(sorted(_V20M_FOLD_SHA256S))
    authenticated_v20m_folds = {
        family: _v20m._load_fold_fragment(
            output=_V20M_OUTPUT,
            source=v20m_source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_fold=authenticated_v20g_folds[family],
            authenticated_v20i_fold=authenticated_v20i_folds[family],
            authenticated_v20l_fold=authenticated_v20l_folds[family],
        )
        for family in families
    }
    if {
        family: fragment["fragment_sha256"]
        for family, fragment in authenticated_v20m_folds.items()
    } != _V20M_FOLD_SHA256S:
        raise RuntimeError("pinned V20m fold authority differs")
    inherited_source = {
        key: value
        for key, value in v20m_source.items()
        if key != "artifact_sha256"
    }
    source = _hashed(
        {
            **inherited_source,
            "v20m_parent_source_receipt_sha256": v20m_source[
                "artifact_sha256"
            ],
            "v20m_report_sha256": _V20M_LOGICAL_SHA256,
            "v20m_file_sha256": _V20M_FILE_SHA256,
            "v20m_source_receipt_sha256": _V20M_SOURCE_SHA256,
            "v20m_classification": v20m_report.get("classification"),
            "v20m_passed": v20m_report.get("passed"),
            "v20m_rollback_to_base": v20m_report.get("rollback_to_base"),
            "v20m_fold_fragment_sha256s_by_family": dict(
                sorted(_V20M_FOLD_SHA256S.items())
            ),
            "reflection_fit_protocol_sha256": (
                _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
            ),
            "masked_direction_protocol_sha256": (
                _reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
            ),
            "simplex_response_fit_protocol_sha256": (
                _simplex_response_fit.SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
            ),
            "simplex_shrinkage_fit_protocol_sha256": (
                _shrinkage_fit.SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
            ),
            "simplex_shrinkage_provider_protocol_sha256": (
                FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256
            ),
            "response_order": _RESPONSES,
            "shrinkage_anchor_lambdas": _SHRINKAGE_ANCHOR_LAMBDAS,
            "exact_objective_kind": (
                "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
            ),
            "authenticated_before_model_construction": True,
            "historically_reused_A16_only": True,
            "held_scores_used_before_direction_response_or_lambda_freeze": False,
            "calibration_b_manifest_read": False,
            "calibration_b_tokenized": False,
        },
        domain=_SOURCE_DOMAIN,
    )
    return (
        prerequisite,
        authenticated_v20a_folds,
        dict(v20g_report),
        authenticated_v20g_folds,
        dict(v20i_report),
        authenticated_v20i_folds,
        dict(v20l_report),
        authenticated_v20l_folds,
        dict(v20m_report),
        authenticated_v20m_folds,
        source,
    )


def _selected_direction(
    reflection_fit: Mapping[str, object],
) -> tuple[float, float, float, float]:
    if reflection_fit.get("selected_variant_available") is not True:
        raise RuntimeError("V20n reflection fit has no admissible direction")
    raw = tuple(
        float(item)
        for item in _sequence(
            reflection_fit.get("selected_normalized_direction"),
            label="V20n selected reflection direction",
        )
    )
    if len(raw) != 4 or not all(math.isfinite(item) for item in raw):
        raise RuntimeError("V20n reflection direction is not a finite four-vector")
    return raw  # type: ignore[return-value]


def _unreflected_direction(
    direction_receipt: Mapping[str, object],
) -> tuple[float, float, float, float]:
    raw = tuple(
        float(item)
        for item in _sequence(
            direction_receipt.get("natural_direction"),
            label="V20n unreflected direction",
        )
    )
    if len(raw) != 4 or not all(math.isfinite(item) for item in raw):
        raise RuntimeError("V20n unreflected direction is not finite")
    return raw  # type: ignore[return-value]


def _box_corner_scores(values: Sequence[float] | Tensor) -> tuple[float, ...]:
    direction = tuple(float(item) for item in _v20g._eta_tensor(values).tolist())
    return tuple(
        direction[0]
        + direction[1] * c1
        + direction[2] * c2
        + direction[3] * c1 * c2
        for c1, c2 in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    )


def _provider_seed(
    *,
    endpoint_receipt_sha256: str,
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    direction: Sequence[float],
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> str:
    radius, u, v = _simplex_parameters(response)
    selected_direction = tuple(float(item) for item in direction)
    return _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": _sha(
                endpoint_receipt_sha256, label="V20n provider endpoint"
            ),
            "direction_artifact_sha256": _sha(
                direction_artifact_sha256, label="V20n provider direction"
            ),
            "reflection_fit_sha256": _sha(
                reflection_fit_sha256, label="V20n provider reflection fit"
            ),
            "response": (radius, u, v),
            "response_key": _parameters_key((radius, u, v)),
            "direction": selected_direction,
            "direction_box_corner_scores": _box_corner_scores(selected_direction),
            "outer_held_family_id": _identifier(
                outer_family_id, label="V20n provider outer family"
            ),
            "inner_held_family_id": inner_family_id,
            "role": role,
            "held_rows_used": False,
        },
        domain=_PROVIDER_DOMAIN,
    )


def _materialize_provider(
    endpoint: _v20g._EndpointLive,
    *,
    direction: Sequence[float],
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> tuple[AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider, str]:
    radius, u, v = _simplex_parameters(response)
    seed = _provider_seed(
        endpoint_receipt_sha256=str(endpoint.receipt["artifact_sha256"]),
        direction_artifact_sha256=direction_artifact_sha256,
        reflection_fit_sha256=reflection_fit_sha256,
        response=response,
        direction=direction,
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        role=role,
    )
    provider = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        endpoint.base_provider,
        endpoint.proposal_provider,
        direction=_v20g._eta_tensor(direction),
        radius=radius,
        shrink_mass=u,
        polarity_bias=v,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
        transfer_evidence_sha256=seed,
    )
    return provider, seed


def _shrinkage_provider_seed(
    *,
    endpoint_receipt_sha256: str,
    direction: Sequence[float],
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    lambda_: float,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> str:
    radius, u, v = _simplex_parameters(response)
    shrinkage = float(lambda_)
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("V20n shrinkage lambda must be inside [0,1]")
    return _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": _sha(
                endpoint_receipt_sha256,
                label="V20n shrinkage provider endpoint",
            ),
            "direction_artifact_sha256": _sha(
                direction_artifact_sha256,
                label="V20n shrinkage provider direction",
            ),
            "reflection_fit_sha256": _sha(
                reflection_fit_sha256,
                label="V20n shrinkage provider reflection fit",
            ),
            "source_response": (radius, u, v),
            "source_response_key": _parameters_key((radius, u, v)),
            "lambda": shrinkage,
            "lambda_hex": shrinkage.hex(),
            "direction": tuple(float(item) for item in direction),
            "direction_box_corner_scores": _box_corner_scores(direction),
            "outer_held_family_id": _identifier(
                outer_family_id, label="V20n shrinkage provider outer family"
            ),
            "inner_held_family_id": inner_family_id,
            "role": role,
            "held_rows_used": False,
        },
        domain=_PROVIDER_DOMAIN,
    )


def _materialize_shrinkage_provider(
    endpoint: _v20g._EndpointLive,
    *,
    direction: Sequence[float],
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    lambda_: float,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> tuple[AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider, str]:
    radius, u, v = _simplex_parameters(response)
    shrinkage = float(lambda_)
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("V20n shrinkage lambda must be inside [0,1]")
    seed = _shrinkage_provider_seed(
        endpoint_receipt_sha256=str(endpoint.receipt["artifact_sha256"]),
        direction=direction,
        direction_artifact_sha256=direction_artifact_sha256,
        reflection_fit_sha256=reflection_fit_sha256,
        response=response,
        lambda_=shrinkage,
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        role=role,
    )
    provider = build_autonomous_complete_h4_fisher_soft_polarity_simplex_shrinkage(
        endpoint.base_provider,
        endpoint.proposal_provider,
        direction=_v20g._eta_tensor(direction),
        radius=radius,
        shrink_mass=u,
        polarity_bias=v,
        lambda_=shrinkage,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
        transfer_evidence_sha256=seed,
    )
    return provider, seed


def _shrinkage_provider_receipt(
    provider: AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider,
    *,
    role: str,
    response: tuple[float, float, float],
    lambda_: float,
    direction: Sequence[float],
) -> dict[str, object]:
    source = _simplex_parameters(response)
    shrinkage = float(lambda_)
    selected_direction = tuple(float(item) for item in provider.direction.tolist())
    expected_direction = tuple(float(item) for item in direction)
    if selected_direction != expected_direction:
        raise RuntimeError("V20n shrinkage provider differs from frozen direction")
    if float(provider.lambda_) != shrinkage:
        raise RuntimeError("V20n shrinkage provider differs from frozen lambda")
    effective = (
        float(provider.effective_radius),
        float(provider.effective_shrink_mass),
        float(provider.effective_polarity_bias),
    )
    metadata = _mapping(provider.metadata(), label=f"V20n {role} metadata")
    payload = provider.artifact_payload()
    receipt = {
        "role": role,
        "source_response": source,
        "source_response_key": _parameters_key(source),
        "lambda": shrinkage,
        "lambda_hex": shrinkage.hex(),
        "effective_response": effective,
        "direction": selected_direction,
        "direction_box_corner_scores": _box_corner_scores(selected_direction),
        "box_certificate": fisher_soft_polarity_simplex_shrinkage_box_certificate(
            provider.direction,
            radius=source[0],
            shrink_mass=source[1],
            polarity_bias=source[2],
            lambda_=shrinkage,
        ),
        "provider_artifact_sha256": _sha(
            provider.artifact_sha256, label=f"V20n {role} provider artifact"
        ),
        "runtime_provider_artifact_sha256": _sha(
            provider.runtime_provider.artifact_sha256,
            label=f"V20n {role} runtime provider artifact",
        ),
        "lineage_wrapper_not_inference_executor": True,
        "provider_metadata": dict(metadata),
        "provider_metadata_sha256": _v14._sha256(
            metadata, domain=_PROVIDER_DOMAIN
        ),
        "provider_payload": dict(payload),
        "transfer_protocol_sha256": metadata.get("transfer_protocol_sha256"),
        "transfer_evidence_sha256": metadata.get("transfer_evidence_sha256"),
        "rank": int(provider.rank),
        "conditional_rank": int(provider.conditional_rank),
        "prepared_float_scalar_count": int(provider.prepared_float_scalar_count),
        "logical_macs_per_token_upper_bound": int(
            provider.logical_macs_per_token_upper_bound
        ),
        "analysis_only": True,
        "raw_provider_tensors_serialized": False,
    }
    return _hashed(receipt, domain=_PROVIDER_DOMAIN)


def _provider_receipt(
    provider: object,
    *,
    role: str,
    response: tuple[float, float, float] | None = None,
    direction: Sequence[float] | None = None,
) -> dict[str, object]:
    metadata = _mapping(provider.metadata(), label=f"V20n {role} metadata")
    provider_payload: Mapping[str, object] | None = None
    selected_direction: tuple[float, ...] | None = None
    corners: tuple[float, ...] | None = None
    radius: float | None = None
    u: float | None = None
    v: float | None = None
    if isinstance(
        provider, AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider
    ):
        provider_payload = provider.artifact_payload()
        selected_direction = tuple(float(item) for item in provider.direction.tolist())
        corners = _box_corner_scores(selected_direction)
        if response is None or direction is None:
            raise ValueError(
                "V20n simplex_response provider receipt needs response and direction"
            )
        radius, u, v = _simplex_parameters(response)
        expected = tuple(float(item) for item in direction)
        if selected_direction != expected:
            raise RuntimeError("V20n provider differs from its frozen direction")
        if (
            float(provider.radius) != radius
            or float(provider.shrink_mass) != u
            or float(provider.polarity_bias) != v
        ):
            raise RuntimeError("V20n provider coefficients differ from response")
        bound = max(abs(item) for item in corners)
        if abs(bound - 1.0) > 1.0e-12:
            raise RuntimeError("V20n provider direction is not box normalized")
    payload = {
        "role": role,
        "response": response,
        "response_key": (
            None if response is None else _parameters_key(response)
        ),
        "radius": radius,
        "u": u,
        "v": v,
        "direction": None if direction is None else tuple(float(x) for x in direction),
        "direction_box_corner_scores": corners,
        "box_certificate": (
            None
            if not isinstance(
                provider, AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider
            )
            else fisher_soft_polarity_simplex_response_box_certificate(
                provider.direction,
                radius=float(provider.radius),
                shrink_mass=float(provider.shrink_mass),
                polarity_bias=float(provider.polarity_bias),
            )
        ),
        "provider_artifact_sha256": _sha(
            provider.artifact_sha256, label=f"V20n {role} provider artifact"
        ),
        "provider_metadata": dict(metadata),
        "provider_metadata_sha256": _v14._sha256(
            metadata, domain=_PROVIDER_DOMAIN
        ),
        "provider_payload": (
            None if provider_payload is None else dict(provider_payload)
        ),
        "transfer_protocol_sha256": metadata.get("transfer_protocol_sha256"),
        "transfer_evidence_sha256": metadata.get("transfer_evidence_sha256"),
        "rank": int(provider.rank),
        "conditional_rank": int(provider.conditional_rank),
        "prepared_float_scalar_count": int(provider.prepared_float_scalar_count),
        "logical_macs_per_token_upper_bound": int(
            provider.logical_macs_per_token_upper_bound
        ),
        "analysis_only": role != "base",
        "raw_provider_tensors_serialized": False,
    }
    return _hashed(payload, domain=_PROVIDER_DOMAIN)


def _strict_receipt_integer(
    value: Mapping[str, object], key: str, *, label: str
) -> int:
    selected = value.get(key)
    if type(selected) is not int or selected < 0:
        raise ValueError(f"{label} {key} must be a nonnegative integer")
    return selected


def _expected_provider_accounting_from_v20i(
    authenticated_v20i_fold: Mapping[str, object], *, role: str
) -> tuple[int, int, int, int]:
    manifest = _mapping(
        authenticated_v20i_fold.get("provider_manifest"),
        label="V20n inherited V20i provider manifest",
    )
    receipts = _mapping(
        manifest.get("provider_receipts"),
        label="V20n inherited V20i provider receipts",
    )
    reference_role = {
        "base": "base",
        "fixed_plus": "fixed_plus",
        "fixed_minus": "fixed_minus",
    }.get(role, "fixed_plus")
    authority = _mapping(
        receipts.get(reference_role),
        label=f"V20n inherited V20i {reference_role} provider receipt",
    )
    rank = _strict_receipt_integer(authority, "rank", label="V20i provider")
    conditional_rank = _strict_receipt_integer(
        authority, "conditional_rank", label="V20i provider"
    )
    prepared = _strict_receipt_integer(
        authority, "prepared_float_scalar_count", label="V20i provider"
    )
    macs = _strict_receipt_integer(
        authority,
        "logical_macs_per_token_upper_bound",
        label="V20i provider",
    )
    if role not in ("base", "fixed_plus", "fixed_minus"):
        # The fixed signed-log axis control stores three response scalars.
        # SimplexResponse stores four direction values plus radius, shrink mass,
        # and polarity bias.  Relative to the fixed signed-log control this is
        # four additional prepared scalars and one dense projection MAC; the
        # simplex arithmetic itself remains outside the dense-MAC total.
        prepared += 4
        macs += 1
    return rank, conditional_rank, prepared, macs


def _validate_provider_receipt_evidence(
    receipt: Mapping[str, object],
    *,
    expected_role: str,
    expected_provider_artifact_sha256: str,
    expected_endpoint_receipt: Mapping[str, object],
    expected_bridge_binding_sha256: str,
    authenticated_v20i_fold: Mapping[str, object],
    expected_response: tuple[float, float, float] | None = None,
    expected_direction: Sequence[float] | None = None,
    expected_transfer_evidence_sha256: str | None = None,
) -> None:
    """Replay one scalar/hash provider claim against independent authority."""

    if set(receipt) != _PROVIDER_RECEIPT_KEYS:
        raise ValueError("V20n provider receipt key set differs")
    provider_artifact = _sha(
        expected_provider_artifact_sha256,
        label="V20n expected provider artifact",
    )
    if (
        receipt.get("role") != expected_role
        or receipt.get("provider_artifact_sha256") != provider_artifact
        or receipt.get("analysis_only") is not (expected_role != "base")
        or receipt.get("raw_provider_tensors_serialized") is not False
    ):
        raise ValueError("V20n provider receipt identity differs")

    metadata = _mapping(
        receipt.get("provider_metadata"), label="V20n provider metadata"
    )
    metadata_sha = _sha(
        receipt.get("provider_metadata_sha256"),
        label="V20n provider metadata hash",
    )
    if (
        _v14._sha256(metadata, domain=_PROVIDER_DOMAIN) != metadata_sha
        or metadata.get("artifact_sha256") != provider_artifact
    ):
        raise ValueError("V20n provider metadata authentication differs")

    receipt_accounting = tuple(
        _strict_receipt_integer(receipt, key, label="V20n provider receipt")
        for key in (
            "rank",
            "conditional_rank",
            "prepared_float_scalar_count",
            "logical_macs_per_token_upper_bound",
        )
    )
    metadata_accounting = tuple(
        _strict_receipt_integer(metadata, key, label="V20n provider metadata")
        for key in (
            "rank",
            "conditional_rank",
            "prepared_float_scalar_count",
            "logical_macs_per_token_upper_bound",
        )
    )
    expected_accounting = _expected_provider_accounting_from_v20i(
        authenticated_v20i_fold, role=expected_role
    )
    if (
        receipt_accounting != metadata_accounting
        or receipt_accounting != expected_accounting
    ):
        raise ValueError("V20n provider accounting differs from pinned V20i")

    if metadata.get("bridge_binding_sha256") not in (
        None,
        expected_bridge_binding_sha256,
    ):
        raise ValueError("V20n provider bridge metadata differs")

    simplex_response_role = expected_response is not None or expected_direction is not None
    if simplex_response_role:
        if expected_response is None or expected_direction is None:
            raise ValueError("V20n simplex_response provider expectation is incomplete")
        response = _simplex_parameters(expected_response)
        direction = tuple(float(item) for item in expected_direction)
        if len(direction) != 4 or not all(math.isfinite(item) for item in direction):
            raise ValueError("V20n expected simplex_response direction differs")
        payload = _mapping(
            receipt.get("provider_payload"), label="V20n simplex_response provider payload"
        )
        validated = validate_fisher_soft_polarity_simplex_response_provider_evidence(
            payload, metadata
        )
        if _v14._canonical_json_bytes(
            validated.metadata.get("box_certificate")
        ) != _v14._canonical_json_bytes(receipt.get("box_certificate")):
            raise ValueError("V20n simplex_response provider box certificate differs")
        expected_direction_sha = fisher_soft_polarity_simplex_response_direction_sha256(
            _v20g._eta_tensor(direction)
        )
        endpoint_base = _sha(
            expected_endpoint_receipt.get("base_provider_artifact_sha256"),
            label="V20n endpoint base provider",
        )
        endpoint_proposal = _sha(
            expected_endpoint_receipt.get("proposal_provider_artifact_sha256"),
            label="V20n endpoint proposal provider",
        )
        expected_transfer = _sha(
            expected_transfer_evidence_sha256,
            label="V20n simplex_response transfer evidence",
        )
        expected_bindings = {
            "bridge_binding_sha256": expected_bridge_binding_sha256,
            "base_provider_artifact_sha256": endpoint_base,
            "proposal_provider_artifact_sha256": endpoint_proposal,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "transfer_evidence_sha256": expected_transfer,
            "direction_sha256": expected_direction_sha,
            "radius": response[0],
            "shrink_mass": response[1],
            "polarity_bias": response[2],
        }
        for key, expected in expected_bindings.items():
            if validated.payload.get(key) != expected:
                raise ValueError(f"V20n simplex_response provider {key} differs")
        for key in (
            "parent_provider_artifact_sha256",
            "start_provider_artifact_sha256",
        ):
            inherited = expected_endpoint_receipt.get(key)
            if inherited is not None and validated.payload.get(key) != inherited:
                raise ValueError(f"V20n simplex_response provider {key} differs")
        if validated.artifact_sha256 != provider_artifact:
            raise ValueError("V20n simplex_response provider artifact replay differs")
    else:
        if receipt.get("provider_payload") is not None:
            raise ValueError("V20n non-simplex_response provider serialized a payload")
        if expected_role in ("fixed_plus", "fixed_minus"):
            expected_transfer = _sha(
                expected_transfer_evidence_sha256,
                label="V20n fixed-control transfer evidence",
            )
            expected_bindings = {
                "base_provider_artifact_sha256": _sha(
                    expected_endpoint_receipt.get(
                        "base_provider_artifact_sha256"
                    ),
                    label="V20n fixed-control endpoint base",
                ),
                "proposal_provider_artifact_sha256": _sha(
                    expected_endpoint_receipt.get(
                        "proposal_provider_artifact_sha256"
                    ),
                    label="V20n fixed-control endpoint proposal",
                ),
                "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
                "transfer_evidence_sha256": expected_transfer,
            }
            for key, expected in expected_bindings.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"V20n fixed-control {key} differs")


def _validate_shrinkage_provider_receipt_evidence(
    receipt: Mapping[str, object],
    *,
    expected_role: str,
    expected_provider_artifact_sha256: str,
    expected_endpoint_receipt: Mapping[str, object],
    expected_bridge_binding_sha256: str,
    authenticated_v20i_fold: Mapping[str, object],
    expected_response: tuple[float, float, float],
    expected_lambda: float,
    expected_direction: Sequence[float],
    expected_transfer_evidence_sha256: str,
) -> None:
    if set(receipt) != _SHRINKAGE_PROVIDER_RECEIPT_KEYS:
        raise ValueError("V20n shrinkage provider receipt key set differs")
    provider_artifact = _sha(
        expected_provider_artifact_sha256,
        label="V20n expected shrinkage provider artifact",
    )
    response = _simplex_parameters(expected_response)
    shrinkage = float(expected_lambda)
    direction = tuple(float(item) for item in expected_direction)
    if (
        receipt.get("role") != expected_role
        or receipt.get("provider_artifact_sha256") != provider_artifact
        or _simplex_parameters(receipt.get("source_response")) != response
        or receipt.get("source_response_key") != _parameters_key(response)
        or float(receipt.get("lambda", math.nan)) != shrinkage
        or receipt.get("lambda_hex") != shrinkage.hex()
        or tuple(receipt.get("direction", ())) != direction
        or receipt.get("lineage_wrapper_not_inference_executor") is not True
        or receipt.get("analysis_only") is not True
        or receipt.get("raw_provider_tensors_serialized") is not False
    ):
        raise ValueError("V20n shrinkage provider receipt identity differs")
    metadata = _mapping(
        receipt.get("provider_metadata"),
        label="V20n shrinkage provider metadata",
    )
    if (
        _v14._sha256(metadata, domain=_PROVIDER_DOMAIN)
        != receipt.get("provider_metadata_sha256")
        or metadata.get("artifact_sha256") != provider_artifact
    ):
        raise ValueError("V20n shrinkage provider metadata authentication differs")
    payload = _mapping(
        receipt.get("provider_payload"),
        label="V20n shrinkage provider payload",
    )
    validated = validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence(
        payload, metadata
    )
    expected_transfer = _sha(
        expected_transfer_evidence_sha256,
        label="V20n shrinkage transfer evidence",
    )
    if (
        validated.artifact_sha256 != provider_artifact
        or validated.payload.get("protocol_sha256")
        != FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256
        or tuple(receipt.get("direction_box_corner_scores", ()))
        != _box_corner_scores(direction)
        or _v14._canonical_json_bytes(
            _mapping(
                receipt.get("box_certificate"),
                label="V20n shrinkage receipt box certificate",
            )
        )
        != _v14._canonical_json_bytes(
            _mapping(
                validated.metadata.get("box_certificate"),
                label="V20n shrinkage metadata box certificate",
            )
        )
        or receipt.get("transfer_protocol_sha256") != _TRANSFER_PROTOCOL_SHA256
        or receipt.get("transfer_evidence_sha256") != expected_transfer
    ):
        raise ValueError("V20n shrinkage provider artifact replay differs")
    expected_effective = (response[0], shrinkage * response[1], shrinkage * response[2])
    if tuple(receipt.get("effective_response", ())) != expected_effective:
        raise ValueError("V20n shrinkage provider effective response differs")
    if receipt.get("runtime_provider_artifact_sha256") != validated.payload.get(
        "compiled_simplex_response_provider_artifact_sha256"
    ):
        raise ValueError("V20n shrinkage runtime provider artifact differs")
    expected_bindings = {
        "bridge_binding_sha256": expected_bridge_binding_sha256,
        "base_provider_artifact_sha256": _sha(
            expected_endpoint_receipt.get("base_provider_artifact_sha256"),
            label="V20n shrinkage endpoint base provider",
        ),
        "proposal_provider_artifact_sha256": _sha(
            expected_endpoint_receipt.get("proposal_provider_artifact_sha256"),
            label="V20n shrinkage endpoint proposal provider",
        ),
        "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
        "transfer_evidence_sha256": expected_transfer,
        "direction_sha256": fisher_soft_polarity_simplex_shrinkage_direction_sha256(
            _v20g._eta_tensor(direction)
        ),
        "source_radius": response[0],
        "source_shrink_mass": response[1],
        "source_polarity_bias": response[2],
        "shrinkage_lambda": shrinkage,
    }
    for key, expected in expected_bindings.items():
        if validated.payload.get(key) != expected:
            raise ValueError(f"V20n shrinkage provider {key} differs")
    receipt_accounting = tuple(
        _strict_receipt_integer(
            receipt, key, label="V20n shrinkage provider receipt"
        )
        for key in (
            "rank",
            "conditional_rank",
            "prepared_float_scalar_count",
            "logical_macs_per_token_upper_bound",
        )
    )
    if receipt_accounting != _expected_provider_accounting_from_v20i(
        authenticated_v20i_fold, role=expected_role
    ):
        raise ValueError("V20n shrinkage provider accounting differs")


def _provider_trace(
    provider: object, records: Sequence[object], *, role: str
) -> dict[str, object]:
    return _v20g._provider_trace(
        provider,
        records,
        arm="base" if role == "base" else role,
        artifact_domain=_TRACE_DOMAIN,
    )


def _runtime_provider_artifact_sha256(provider: object) -> str:
    runtime_provider = (
        provider.runtime_provider
        if isinstance(
            provider,
            AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider,
        )
        else provider
    )
    return _sha(
        runtime_provider.artifact_sha256,
        label="V20n runtime provider artifact",
    )


def _execution_sha256(
    *,
    phase: str,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
    provider_artifact_sha256: str,
    example_id: str,
    family_id: str,
    objective: float,
    h4_sha256: str,
    logits_sha256: str,
    evidence_sha256: str,
    domain: bytes,
) -> str:
    return _v14._sha256(
        {
            "phase": phase,
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "role": role,
            "provider_artifact_sha256": provider_artifact_sha256,
            "example_id": example_id,
            "family_id": family_id,
            "objective": float(objective),
            "post_cast_h4_sha256": h4_sha256,
            "supervised_full_vocab_logits_sha256": logits_sha256,
            "evidence_sha256": evidence_sha256,
        },
        domain=domain,
    )


def _score_exact_provider(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: object,
    phase: str,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
    evidence_sha256: str,
    domain: bytes,
) -> tuple[dict[str, float], dict[str, str], dict[str, str], dict[str, str]]:
    objectives: dict[str, float] = {}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    execution_hashes: dict[str, str] = {}
    for record in _v20b._ordered_records(records):
        model_inputs, supervised_indices, _targets = _v20a._verified_model_inputs(
            context, record
        )
        teacher = capability.get(
            record.sequence.example_id, family_id=record.sequence.family_id
        )
        runtime_provider = (
            provider.runtime_provider
            if isinstance(
                provider,
                AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider,
            )
            else provider
        )
        execution = context.bridge.execute(
            context.adapter, model_inputs, h4_head=runtime_provider
        )
        score, h4_sha, logits_sha = _v20a._execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=runtime_provider.artifact_sha256,
        )
        example = record.sequence.example_id
        objectives[example] = score
        h4_hashes[example] = h4_sha
        logits_hashes[example] = logits_sha
        execution_hashes[example] = _execution_sha256(
            phase=phase,
            outer_family_id=outer_family_id,
            inner_family_id=inner_family_id,
            role=role,
            provider_artifact_sha256=runtime_provider.artifact_sha256,
            example_id=example,
            family_id=record.sequence.family_id,
            objective=score,
            h4_sha256=h4_sha,
            logits_sha256=logits_sha,
            evidence_sha256=evidence_sha256,
            domain=domain,
        )
        del model_inputs, teacher, execution
    return objectives, h4_hashes, logits_hashes, execution_hashes


def _freeze_inner_providers(
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    records: Sequence[object],
    *,
    outer_family_id: str,
) -> tuple[
    dict[
        str,
        dict[
            tuple[float, float, float],
            AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider,
        ],
    ],
    dict[str, object],
    dict[str, dict[tuple[float, float, float], dict[str, object]]],
    dict[str, dict[str, object]],
]:
    """Freeze all 7x19 inner providers and traces before any capability."""

    outer = _identifier(outer_family_id, label="V20n inner outer family")
    ordered = _v20b._ordered_records(records)
    training_families = tuple(
        sorted({record.sequence.family_id for record in ordered})
    )
    if (
        len(training_families) != _INNER_FAMILY_COUNT
        or outer in training_families
        or tuple(source_direction_receipt.get("training_family_ids", ()))
        != training_families
    ):
        raise RuntimeError("V20n inner family geometry differs")

    providers: dict[
        str,
        dict[
            tuple[float, float, float],
            AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider,
        ],
    ] = {}
    traces: dict[str, dict[tuple[float, float, float], dict[str, object]]] = {}
    fits: dict[str, dict[str, object]] = {}
    provider_hashes: dict[str, dict[str, str]] = {}
    trace_hashes: dict[str, dict[str, str]] = {}
    provider_receipts: dict[str, dict[str, dict[str, object]]] = {}
    transfer_evidence: dict[str, dict[str, str]] = {}

    for inner in training_families:
        masked = _reflection.build_soft_polarity_masked_direction_receipt(
            source_direction_receipt=source_direction_receipt,
            excluded_training_family_id=inner,
        )
        reflection_fit = _reflection.build_soft_polarity_reflection_fit_receipt(
            direction_receipt=masked
        )
        selected = _selected_direction(reflection_fit)
        selected_artifact = _sha(
            reflection_fit.get("selected_variant_artifact_sha256"),
            label="V20n inner selected reflection variant",
        )
        held_records = tuple(
            record for record in ordered if record.sequence.family_id == inner
        )
        if len(held_records) != _PROMPTS_PER_FAMILY:
            raise RuntimeError("V20n inner-held prompt geometry differs")

        providers[inner] = {}
        traces[inner] = {}
        provider_hashes[inner] = {}
        trace_hashes[inner] = {}
        provider_receipts[inner] = {}
        transfer_evidence[inner] = {}
        for response in _RESPONSES:
            key = _response_key(response)
            provider, seed = _materialize_provider(
                endpoint,
                direction=selected,
                direction_artifact_sha256=selected_artifact,
                reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
                response=response,
                outer_family_id=outer,
                inner_family_id=inner,
                role="inner_reflected_response_candidate",
            )
            providers[inner][response] = provider
            radius, u, v = response
            traces[inner][response] = _provider_trace(
                provider,
                held_records,
                role=(
                    f"inner_{inner}_radius_{radius.hex()}_u_{u.hex()}_v_{v.hex()}"
                ),
            )
            provider_hashes[inner][key] = provider.artifact_sha256
            trace_hashes[inner][key] = str(
                traces[inner][response]["artifact_sha256"]
            )
            provider_receipts[inner][key] = _provider_receipt(
                provider,
                role="inner_reflected_response_candidate",
                response=response,
                direction=selected,
            )
            transfer_evidence[inner][key] = seed
        fits[inner] = {
            "masked_direction_receipt": masked,
            "reflection_fit_receipt": reflection_fit,
            "selected_variant_artifact_sha256": selected_artifact,
            "selected_normalized_direction": selected,
            "inner_held_family_id": inner,
            "inner_training_family_ids": tuple(
                family for family in training_families if family != inner
            ),
        }

    flat_hashes = tuple(
        provider_hashes[inner][_response_key(response)]
        for inner in training_families
        for response in _RESPONSES
    )
    if len(flat_hashes) != _INNER_FAMILY_COUNT * len(_RESPONSES) or len(
        set(flat_hashes)
    ) != len(flat_hashes):
        raise RuntimeError("V20n inner provider artifacts are not all distinct")
    manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "inner_family_order": training_families,
            "response_order": _RESPONSES,
            "simplex_response_ladder_receipt_sha256": (
                _SIMPLEX_RESPONSE_LADDER_RECEIPT_SHA256
            ),
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "source_direction_receipt_sha256": source_direction_receipt[
                "artifact_sha256"
            ],
            "masked_direction_receipt_sha256s_by_inner_family": {
                inner: fits[inner]["masked_direction_receipt"]["artifact_sha256"]
                for inner in training_families
            },
            "reflection_fit_receipt_sha256s_by_inner_family": {
                inner: fits[inner]["reflection_fit_receipt"]["artifact_sha256"]
                for inner in training_families
            },
            "selected_variant_artifact_sha256s_by_inner_family": {
                inner: fits[inner]["selected_variant_artifact_sha256"]
                for inner in training_families
            },
            "provider_artifact_sha256s_by_inner_family_and_response": provider_hashes,
            "provider_transfer_evidence_sha256s_by_inner_family_and_response": (
                transfer_evidence
            ),
            "provider_receipts_by_inner_family_and_response": provider_receipts,
            "response_trace_sha256s_by_inner_family_and_response": trace_hashes,
            "all_seven_times_nineteen_providers_frozen_before_any_inner_capability": True,
            "all_seven_times_nineteen_traces_frozen_before_any_inner_capability": True,
            "inner_capability_count_at_freeze": 0,
            "inner_objectives_or_teacher_rows_used_at_freeze": False,
            "inner_endpoint_retrained_per_fold": False,
            "inner_held_family_used_for_endpoint_fit": True,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=_INNER_MANIFEST_DOMAIN,
    )
    return providers, manifest, traces, fits


def _aggregate_response_selection(
    inner_evidence_by_family: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    families = tuple(sorted(inner_evidence_by_family))
    if len(families) != _INNER_FAMILY_COUNT:
        raise ValueError("V20n response selection requires seven inner OOF families")
    outer_ids = {
        _identifier(
            inner_evidence_by_family[family].get("outer_held_family_id"),
            label="V20n response selection outer family",
        )
        for family in families
    }
    if len(outer_ids) != 1 or next(iter(outer_ids)) in families:
        raise ValueError("V20n response selection outer family geometry differs")
    outer = next(iter(outer_ids))
    all_families = tuple(sorted((*families, outer)))
    objectives_by_response: dict[str, float] = {}
    aggregate_artifacts: dict[str, str] = {}
    objectives_by_family_and_response: dict[str, dict[str, float]] = {}
    for family in families:
        raw = _mapping(
            inner_evidence_by_family[family].get("objective_by_response"),
            label="V20n inner objective ladder",
        )
        if set(raw) != set(_RESPONSE_KEYS):
            raise ValueError("V20n inner objective response geometry differs")
        objectives_by_family_and_response[family] = {
            key: float(raw[key]) for key in _RESPONSE_KEYS
        }
    for response in _RESPONSES:
        key = _response_key(response)
        objectives_by_response[key] = math.fsum(
            objectives_by_family_and_response[family][key] for family in families
        ) / len(families)
        aggregate_artifacts[key] = _v14._sha256(
            {
                "response": response,
                "inner_family_order": families,
                "inner_evidence_sha256s": {
                    family: inner_evidence_by_family[family]["artifact_sha256"]
                    for family in families
                },
                "family_objectives": {
                    family: objectives_by_family_and_response[family][key]
                    for family in families
                },
                "family_equal_objective": objectives_by_response[key],
            },
            domain=_RESPONSE_SELECTION_DOMAIN,
        )
    ladder_receipt = dict(_SIMPLEX_RESPONSE_LADDER_RECEIPT)
    exact_by_family_and_candidate = {
        family: {
            candidate_id: objectives_by_family_and_response[family][
                _response_key(response)
            ]
            for candidate_id, response in zip(
                _simplex_response_fit.SOFT_POLARITY_SIMPLEX_RESPONSE_CANDIDATE_IDS,
                _RESPONSES,
                strict=True,
            )
        }
        for family in families
    }
    core_selection = (
        _simplex_response_fit.build_soft_polarity_simplex_response_inner_oof_selection_receipt(
            ladder_receipt=ladder_receipt,
            all_development_family_ids=all_families,
            outer_held_family_id=outer,
            exact_objectives_by_family_and_candidate=exact_by_family_and_candidate,
        )
    )
    selected = (
        float(core_selection["selected_r"]),
        float(core_selection["selected_u"]),
        float(core_selection["selected_v"]),
    )
    return _hashed(
        {
            "inner_family_order": families,
            "response_order": _RESPONSES,
            "objectives_by_inner_family_and_response": (
                objectives_by_family_and_response
            ),
            "family_equal_objective_by_response": objectives_by_response,
            "aggregate_artifact_sha256_by_response": aggregate_artifacts,
            "simplex_response_fit_protocol_sha256": (
                _simplex_response_fit.SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
            ),
            "simplex_response_ladder_receipt": ladder_receipt,
            "simplex_response_selection_receipt": core_selection,
            "selection_rule": (
                "minimum_inner_OOF_family_equal_token_mean_exact_float64_full_"
                "vocabulary_KL_teacher_to_candidate_then_smaller_u_then_"
                "smaller_abs_v_then_smaller_radius_then_fixed_index_then_"
                "candidate_artifact_sha256"
            ),
            "selected_response": selected,
            "selected_family_equal_objective": objectives_by_response[
                _response_key(selected)
            ],
            "selected_aggregate_artifact_sha256": aggregate_artifacts[
                _response_key(selected)
            ],
            "all_inner_providers_frozen_before_any_inner_score": True,
            "same_family_used_for_direction_fit_and_inner_score": False,
            "inner_endpoint_retrained_per_fold": False,
            "inner_held_family_used_for_endpoint_fit": True,
            "inner_claim_scope": (
                "conditional_response_LOFO_not_fully_nested_model_cross_"
                "validation"
            ),
            "outer_held_family_used_for_selection": False,
        },
        domain=_RESPONSE_SELECTION_DOMAIN,
    )


def _fit_inner_response(
    context: object,
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    outer = _identifier(outer_family_id, label="V20n inner-response outer family")
    training = _v20b._ordered_records(endpoint.training_records)
    providers, manifest, traces, fits = _freeze_inner_providers(
        endpoint,
        source_direction_receipt,
        training,
        outer_family_id=outer,
    )
    if (
        manifest.get(
            "all_seven_times_nineteen_providers_frozen_before_any_inner_capability"
        )
        is not True
        or manifest.get(
            "all_seven_times_nineteen_traces_frozen_before_any_inner_capability"
        )
        is not True
        or manifest.get("inner_capability_count_at_freeze") != 0
        or manifest.get("inner_objectives_or_teacher_rows_used_at_freeze")
        is not False
    ):
        raise PermissionError("V20n inner freeze barrier is not satisfied")

    inner_evidence: dict[str, dict[str, object]] = {}
    gradient_evidence = _mapping(
        _mapping(
            authenticated_v20g_fold.get("fit_training_evidence"),
            label="V20n inherited fit evidence",
        ).get("gradient_evidence"),
        label="V20n inherited gradient evidence",
    )
    eta_zero_objectives = _mapping(
        gradient_evidence.get("eta_zero_objectives_by_family"),
        label="V20n inherited eta-zero objectives",
    )
    eta_zero_h4 = _mapping(
        gradient_evidence.get("post_cast_h4_sha256s"),
        label="V20n inherited eta-zero H4 hashes",
    )
    eta_zero_logits = _mapping(
        gradient_evidence.get("supervised_full_vocab_logits_sha256s"),
        label="V20n inherited eta-zero logits hashes",
    )
    for inner in tuple(manifest["inner_family_order"]):
        held = _v20b._ordered_records(
            tuple(record for record in training if record.sequence.family_id == inner)
        )
        trace_bundle_sha = _v14._sha256(
            {
                _response_key(response): traces[inner][response]["artifact_sha256"]
                for response in _RESPONSES
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
        capability = teacher_vault.capability(
            tuple(record.sequence.example_id for record in held),
            held_family_id=outer,
        )
        objective_by_response: dict[str, float] = {}
        evidence_by_response: dict[str, dict[str, object]] = {}
        for response in _RESPONSES:
            key = _response_key(response)
            seed = _v14._sha256(
                {
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "trace_bundle_sha256": trace_bundle_sha,
                    "outer_held_family_id": outer,
                    "inner_held_family_id": inner,
                    "response": response,
                    "provider_artifact_sha256": providers[inner][
                        response
                    ].artifact_sha256,
                    "all_inner_candidates_frozen": True,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
            objectives, h4_hashes, logits_hashes, execution_hashes = (
                _score_exact_provider(
                    context,
                    held,
                    capability,
                    provider=providers[inner][response],
                    phase="inner_conditional_leave_one_family_out_response_score",
                    outer_family_id=outer,
                    inner_family_id=inner,
                    role="inner_reflected_response_candidate",
                    evidence_sha256=seed,
                    domain=_INNER_EXECUTION_DOMAIN,
                )
            )
            macro, family_scores = _v19._family_equal_mean(objectives, held)
            if set(family_scores) != {inner}:
                raise RuntimeError("V20n inner score family geometry differs")
            objective_by_response[key] = macro
            evidence_by_response[key] = _hashed(
                {
                    "outer_held_family_id": outer,
                    "inner_held_family_id": inner,
                    "response": response,
                    "provider_artifact_sha256": providers[inner][
                        response
                    ].artifact_sha256,
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "response_trace": traces[inner][response],
                    "objective": macro,
                    "objectives_by_example": dict(sorted(objectives.items())),
                    "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                    "supervised_full_vocab_logits_sha256s": dict(
                        sorted(logits_hashes.items())
                    ),
                    "execution_sha256s": dict(sorted(execution_hashes.items())),
                    "exact_execution": True,
                    "finite": True,
                    "inner_family_absent_from_direction_and_reflection_fit": True,
                    "outer_family_absent_from_endpoint_direction_and_score": True,
                    "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
        capability_receipt = capability.receipt()
        _v20b._validate_capability_receipt(
            capability_receipt,
            expected_example_ids=tuple(
                record.sequence.example_id for record in held
            ),
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=len(_RESPONSES),
            label="V20n inner-held capability",
        )
        zero = evidence_by_response[_response_key((0.0, 0.0, 0.0))]
        zero_objectives = _mapping(
            zero.get("objectives_by_example"),
            label="V20n inner eta-zero objectives",
        )
        zero_h4 = _mapping(
            zero.get("post_cast_h4_sha256s"),
            label="V20n inner eta-zero H4 hashes",
        )
        zero_logits = _mapping(
            zero.get("supervised_full_vocab_logits_sha256s"),
            label="V20n inner eta-zero logits hashes",
        )
        expected_zero_objectives = _mapping(
            eta_zero_objectives.get(inner),
            label="V20n inherited family eta-zero objectives",
        )
        zero_anchor = (
            dict(zero_objectives) == dict(expected_zero_objectives)
            and dict(zero_h4)
            == {
                example: eta_zero_h4[example]
                for example in sorted(expected_zero_objectives)
            }
            and dict(zero_logits)
            == {
                example: eta_zero_logits[example]
                for example in sorted(expected_zero_objectives)
            }
        )
        if not zero_anchor:
            raise RuntimeError("V20n inner eta-zero output anchor differs from V20g")
        inner_evidence[inner] = _hashed(
            {
                "outer_held_family_id": outer,
                "inner_held_family_id": inner,
                "inner_training_family_ids": fits[inner][
                    "inner_training_family_ids"
                ],
                "masked_direction_receipt": fits[inner][
                    "masked_direction_receipt"
                ],
                "reflection_fit_receipt": fits[inner]["reflection_fit_receipt"],
                "selected_variant_artifact_sha256": fits[inner][
                    "selected_variant_artifact_sha256"
                ],
                "response_order": _RESPONSES,
                "objective_by_response": objective_by_response,
                "response_evidence": evidence_by_response,
                "capability_receipt": capability_receipt,
                "exact_execution_count": len(_RESPONSES) * len(held),
                "zero_response_exact_v20g_eta_zero_output_anchor": zero_anchor,
                "all_inner_candidates_frozen_before_capability": True,
                "held_family_used_for_direction_or_reflection_fit": False,
                "held_family_used_for_endpoint_fit": True,
                "endpoint_retrained_without_held_inner_family": False,
                "raw_prompts_tokens_logits_h4_gradients_or_teacher_rows_"
                "serialized": False,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )

    selection = _aggregate_response_selection(inner_evidence)
    receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "source_direction_receipt_sha256": source_direction_receipt[
                "artifact_sha256"
            ],
            "inner_provider_manifest": manifest,
            "inner_evidence_by_family": inner_evidence,
            "response_selection_receipt": selection,
            "inner_family_order": manifest["inner_family_order"],
            "response_order": _RESPONSES,
            "all_inner_fits_and_providers_frozen_before_any_inner_capability": True,
            "exact_inner_execution_count": (
                _INNER_FAMILY_COUNT * len(_RESPONSES) * _PROMPTS_PER_FAMILY
            ),
            "inner_endpoint_retrained_per_fold": False,
            "inner_held_family_used_for_endpoint_fit": True,
            "inner_claim_scope": (
                "conditional_response_LOFO_not_fully_nested_model_cross_"
                "validation"
            ),
            "outer_held_family_used_for_fit_or_selection": False,
            "raw_provider_gradient_logits_h4_or_teacher_tensors_serialized": False,
        },
        domain=_INNER_FIT_DOMAIN,
    )
    return receipt, selection


def _fit_inner_shrinkage(
    context: object,
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Reproduce V20m, then freeze/score half and proposed-vertex stages."""

    outer = _identifier(outer_family_id, label="V20n shrinkage outer family")
    v20m_inner, response_selection = _fit_inner_response(
        context,
        endpoint,
        source_direction_receipt,
        teacher_vault,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
    )
    inherited_response_selection = _mapping(
        authenticated_v20m_fold.get("response_selection_receipt"),
        label="V20n authenticated V20m response selection",
    )
    for key in (
        "objectives_by_inner_family_and_response",
        "family_equal_objective_by_response",
        "simplex_response_selection_receipt",
        "selected_response",
    ):
        if _v14._canonical_json_bytes(response_selection.get(key)) != (
            _v14._canonical_json_bytes(inherited_response_selection.get(key))
        ):
            raise RuntimeError(
                f"V20n live V20m response selection {key} reproduction differs"
            )
    inherited_inner_evidence = _mapping(
        _mapping(
            authenticated_v20m_fold.get("inner_receipt"),
            label="V20n authenticated V20m inner receipt",
        ).get("inner_evidence_by_family"),
        label="V20n authenticated V20m inner evidence",
    )
    live_inner_evidence = _mapping(
        v20m_inner.get("inner_evidence_by_family"),
        label="V20n live V20m inner evidence",
    )
    if set(live_inner_evidence) != set(inherited_inner_evidence):
        raise RuntimeError("V20n live V20m inner family reproduction differs")
    for family in sorted(live_inner_evidence):
        live_responses = _mapping(
            live_inner_evidence[family].get("response_evidence"),
            label="V20n live V20m response evidence",
        )
        inherited_responses = _mapping(
            inherited_inner_evidence[family].get("response_evidence"),
            label="V20n authenticated V20m response evidence",
        )
        if set(live_responses) != set(inherited_responses):
            raise RuntimeError("V20n live V20m response geometry differs")
        for response_key in sorted(live_responses):
            for evidence_key in (
                "objectives_by_example",
                "post_cast_h4_sha256s",
                "supervised_full_vocab_logits_sha256s",
            ):
                if _v14._canonical_json_bytes(
                    live_responses[response_key].get(evidence_key)
                ) != _v14._canonical_json_bytes(
                    inherited_responses[response_key].get(evidence_key)
                ):
                    raise RuntimeError(
                        "V20n live V20m inner exact-output reproduction differs"
                    )
    response = _response_tuple(response_selection["selected_response"])
    linear_response = (response[0], 0.0, 0.0)
    if linear_response not in _RESPONSES:
        raise RuntimeError("V20n selected response has no exact lambda-zero anchor")
    training = _v20b._ordered_records(endpoint.training_records)
    inner_evidence = _mapping(
        v20m_inner.get("inner_evidence_by_family"),
        label="V20n inherited V20m inner evidence",
    )
    families = tuple(sorted(inner_evidence))
    if len(families) != _INNER_FAMILY_COUNT or outer in families:
        raise RuntimeError("V20n shrinkage inner family geometry differs")

    half_providers: dict[
        str, AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider
    ] = {}
    half_traces: dict[str, dict[str, object]] = {}
    half_receipts: dict[str, dict[str, object]] = {}
    half_seeds: dict[str, str] = {}
    directions: dict[str, tuple[float, float, float, float]] = {}
    fits: dict[str, Mapping[str, object]] = {}
    for inner in families:
        evidence = _mapping(
            inner_evidence[inner], label="V20n V20m inner family evidence"
        )
        fit = _mapping(
            evidence.get("reflection_fit_receipt"),
            label="V20n inner reflection fit",
        )
        direction = _selected_direction(fit)
        fits[inner] = fit
        directions[inner] = direction
        provider, seed = _materialize_shrinkage_provider(
            endpoint,
            direction=direction,
            direction_artifact_sha256=_sha(
                fit.get("selected_variant_artifact_sha256"),
                label="V20n half selected direction artifact",
            ),
            reflection_fit_sha256=_sha(
                fit.get("artifact_sha256"),
                label="V20n half reflection fit artifact",
            ),
            response=response,
            lambda_=0.5,
            outer_family_id=outer,
            inner_family_id=inner,
            role="inner_simplex_shrinkage_lambda_half",
        )
        held = tuple(
            record for record in training if record.sequence.family_id == inner
        )
        half_providers[inner] = provider
        half_seeds[inner] = seed
        half_traces[inner] = _provider_trace(
            provider, held, role="inner_simplex_shrinkage_lambda_half"
        )
        half_receipts[inner] = _shrinkage_provider_receipt(
            provider,
            role="inner_simplex_shrinkage_lambda_half",
            response=response,
            lambda_=0.5,
            direction=direction,
        )
    if len({provider.artifact_sha256 for provider in half_providers.values()}) != (
        _INNER_FAMILY_COUNT
    ):
        raise RuntimeError("V20n half-provider artifacts are not all distinct")
    half_manifest = _hashed(
        {
            "stage": "lambda_half",
            "outer_held_family_id": outer,
            "inner_family_order": families,
            "source_response": response,
            "lambda": 0.5,
            "lambda_hex": (0.5).hex(),
            "provider_artifact_sha256s_by_inner_family": {
                family: half_providers[family].artifact_sha256
                for family in families
            },
            "runtime_provider_artifact_sha256s_by_inner_family": {
                family: _runtime_provider_artifact_sha256(
                    half_providers[family]
                )
                for family in families
            },
            "provider_transfer_evidence_sha256s_by_inner_family": half_seeds,
            "provider_receipts_by_inner_family": half_receipts,
            "trace_sha256s_by_inner_family": {
                family: half_traces[family]["artifact_sha256"]
                for family in families
            },
            "all_seven_half_providers_and_traces_frozen_before_any_half_"
            "capability": True,
            "half_capability_count_at_freeze": 0,
            "half_objectives_or_teacher_rows_used_at_freeze": False,
            "outer_held_family_used": False,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=_INNER_MANIFEST_DOMAIN,
    )

    half_evidence: dict[str, dict[str, object]] = {}
    half_objectives: dict[str, float] = {}
    for inner in families:
        held = _v20b._ordered_records(
            tuple(
                record
                for record in training
                if record.sequence.family_id == inner
            )
        )
        capability = teacher_vault.capability(
            tuple(record.sequence.example_id for record in held),
            held_family_id=outer,
        )
        seed = _v14._sha256(
            {
                "half_manifest_sha256": half_manifest["artifact_sha256"],
                "outer_held_family_id": outer,
                "inner_held_family_id": inner,
                "provider_artifact_sha256": half_providers[
                    inner
                ].artifact_sha256,
                "runtime_provider_artifact_sha256": (
                    _runtime_provider_artifact_sha256(half_providers[inner])
                ),
                "lineage_wrapper_not_inference_executor": True,
                "all_half_providers_frozen": True,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
        objectives, h4_hashes, logits_hashes, execution_hashes = (
            _score_exact_provider(
                context,
                held,
                capability,
                provider=half_providers[inner],
                phase="inner_simplex_shrinkage_lambda_half_score",
                outer_family_id=outer,
                inner_family_id=inner,
                role="inner_simplex_shrinkage_lambda_half",
                evidence_sha256=seed,
                domain=_INNER_EXECUTION_DOMAIN,
            )
        )
        macro, family_scores = _v19._family_equal_mean(objectives, held)
        if set(family_scores) != {inner}:
            raise RuntimeError("V20n half score family geometry differs")
        half_objectives[inner] = macro
        capability_receipt = capability.receipt()
        _v20b._validate_capability_receipt(
            capability_receipt,
            expected_example_ids=tuple(
                record.sequence.example_id for record in held
            ),
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=1,
            label="V20n half-score capability",
        )
        half_evidence[inner] = _hashed(
            {
                "stage": "lambda_half",
                "outer_held_family_id": outer,
                "inner_held_family_id": inner,
                "lambda": 0.5,
                "provider_artifact_sha256": half_providers[
                    inner
                ].artifact_sha256,
                "runtime_provider_artifact_sha256": (
                    _runtime_provider_artifact_sha256(half_providers[inner])
                ),
                "lineage_wrapper_not_inference_executor": True,
                "manifest_sha256": half_manifest["artifact_sha256"],
                "response_trace": half_traces[inner],
                "objective": macro,
                "objectives_by_example": dict(sorted(objectives.items())),
                "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                "supervised_full_vocab_logits_sha256s": dict(
                    sorted(logits_hashes.items())
                ),
                "execution_sha256s": dict(sorted(execution_hashes.items())),
                "capability_receipt": capability_receipt,
                "all_half_providers_frozen_before_score": True,
                "outer_family_absent_from_fit_and_score": True,
                "exact_execution": True,
                "finite": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )

    anchor_objectives: dict[str, dict[str, float]] = {}
    for family in families:
        response_scores = _mapping(
            inner_evidence[family].get("objective_by_response"),
            label="V20n inherited V20m response objectives",
        )
        anchor_objectives[family] = {
            "lambda_0": float(response_scores[_response_key(linear_response)]),
            "lambda_half": half_objectives[family],
            "lambda_1": float(response_scores[_response_key(response)]),
        }
    all_families = tuple(sorted((*families, outer)))
    anchor_receipt = (
        _shrinkage_fit.build_soft_polarity_simplex_shrinkage_anchor_receipt(
            all_development_family_ids=all_families,
            outer_held_family_id=outer,
            exact_anchor_objectives_by_family_and_anchor=anchor_objectives,
        )
    )
    proposal_receipt = (
        _shrinkage_fit.build_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
            anchor_receipt=anchor_receipt
        )
    )
    proposed_lambda = float(proposal_receipt["proposed_lambda"])

    vertex_providers: dict[
        str, AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider
    ] = {}
    vertex_traces: dict[str, dict[str, object]] = {}
    vertex_receipts: dict[str, dict[str, object]] = {}
    vertex_seeds: dict[str, str] = {}
    for inner in families:
        fit = fits[inner]
        provider, seed = _materialize_shrinkage_provider(
            endpoint,
            direction=directions[inner],
            direction_artifact_sha256=_sha(
                fit.get("selected_variant_artifact_sha256"),
                label="V20n vertex selected direction artifact",
            ),
            reflection_fit_sha256=_sha(
                fit.get("artifact_sha256"),
                label="V20n vertex reflection fit artifact",
            ),
            response=response,
            lambda_=proposed_lambda,
            outer_family_id=outer,
            inner_family_id=inner,
            role="inner_simplex_shrinkage_vertex",
        )
        held = tuple(
            record for record in training if record.sequence.family_id == inner
        )
        vertex_providers[inner] = provider
        vertex_seeds[inner] = seed
        vertex_traces[inner] = _provider_trace(
            provider, held, role="inner_simplex_shrinkage_vertex"
        )
        vertex_receipts[inner] = _shrinkage_provider_receipt(
            provider,
            role="inner_simplex_shrinkage_vertex",
            response=response,
            lambda_=proposed_lambda,
            direction=directions[inner],
        )
    if len(
        {provider.artifact_sha256 for provider in vertex_providers.values()}
    ) != _INNER_FAMILY_COUNT:
        raise RuntimeError("V20n vertex-provider artifacts are not all distinct")
    vertex_manifest = _hashed(
        {
            "stage": "quadratic_vertex",
            "outer_held_family_id": outer,
            "inner_family_order": families,
            "source_response": response,
            "lambda": proposed_lambda,
            "lambda_hex": proposed_lambda.hex(),
            "anchor_receipt_sha256": anchor_receipt["artifact_sha256"],
            "proposal_receipt_sha256": proposal_receipt["artifact_sha256"],
            "provider_artifact_sha256s_by_inner_family": {
                family: vertex_providers[family].artifact_sha256
                for family in families
            },
            "runtime_provider_artifact_sha256s_by_inner_family": {
                family: _runtime_provider_artifact_sha256(
                    vertex_providers[family]
                )
                for family in families
            },
            "provider_transfer_evidence_sha256s_by_inner_family": vertex_seeds,
            "provider_receipts_by_inner_family": vertex_receipts,
            "trace_sha256s_by_inner_family": {
                family: vertex_traces[family]["artifact_sha256"]
                for family in families
            },
            "all_seven_vertex_providers_and_traces_frozen_before_any_vertex_"
            "capability": True,
            "vertex_capability_count_at_freeze": 0,
            "vertex_objectives_or_teacher_rows_used_at_freeze": False,
            "outer_held_family_used": False,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=_INNER_MANIFEST_DOMAIN,
    )
    vertex_evidence: dict[str, dict[str, object]] = {}
    vertex_objectives: dict[str, float] = {}
    endpoint_vertex_anchor_by_family: dict[str, bool] = {}
    for inner in families:
        held = _v20b._ordered_records(
            tuple(
                record
                for record in training
                if record.sequence.family_id == inner
            )
        )
        capability = teacher_vault.capability(
            tuple(record.sequence.example_id for record in held),
            held_family_id=outer,
        )
        seed = _v14._sha256(
            {
                "vertex_manifest_sha256": vertex_manifest["artifact_sha256"],
                "outer_held_family_id": outer,
                "inner_held_family_id": inner,
                "provider_artifact_sha256": vertex_providers[
                    inner
                ].artifact_sha256,
                "runtime_provider_artifact_sha256": (
                    _runtime_provider_artifact_sha256(vertex_providers[inner])
                ),
                "lineage_wrapper_not_inference_executor": True,
                "all_vertex_providers_frozen": True,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
        objectives, h4_hashes, logits_hashes, execution_hashes = (
            _score_exact_provider(
                context,
                held,
                capability,
                provider=vertex_providers[inner],
                phase="inner_simplex_shrinkage_vertex_score",
                outer_family_id=outer,
                inner_family_id=inner,
                role="inner_simplex_shrinkage_vertex",
                evidence_sha256=seed,
                domain=_INNER_EXECUTION_DOMAIN,
            )
        )
        macro, family_scores = _v19._family_equal_mean(objectives, held)
        if set(family_scores) != {inner}:
            raise RuntimeError("V20n vertex score family geometry differs")
        vertex_objectives[inner] = macro
        endpoint_anchor = True
        if proposed_lambda in (0.0, 1.0):
            endpoint_response = linear_response if proposed_lambda == 0.0 else response
            inherited_endpoint = _mapping(
                _mapping(
                    inner_evidence[inner].get("response_evidence"),
                    label="V20n inherited V20m response evidence",
                ).get(_response_key(endpoint_response)),
                label="V20n inherited V20m endpoint response evidence",
            )
            endpoint_anchor = (
                macro == float(inherited_endpoint["objective"])
                and dict(sorted(objectives.items()))
                == dict(
                    _mapping(
                        inherited_endpoint.get("objectives_by_example"),
                        label="V20n inherited endpoint objectives",
                    )
                )
                and dict(sorted(h4_hashes.items()))
                == dict(
                    _mapping(
                        inherited_endpoint.get("post_cast_h4_sha256s"),
                        label="V20n inherited endpoint H4 hashes",
                    )
                )
                and dict(sorted(logits_hashes.items()))
                == dict(
                    _mapping(
                        inherited_endpoint.get(
                            "supervised_full_vocab_logits_sha256s"
                        ),
                        label="V20n inherited endpoint logits hashes",
                    )
                )
            )
            if not endpoint_anchor:
                raise RuntimeError(
                    "V20n endpoint vertex failed exact V20m output reproduction"
                )
        endpoint_vertex_anchor_by_family[inner] = endpoint_anchor
        capability_receipt = capability.receipt()
        _v20b._validate_capability_receipt(
            capability_receipt,
            expected_example_ids=tuple(
                record.sequence.example_id for record in held
            ),
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=1,
            label="V20n vertex-score capability",
        )
        vertex_evidence[inner] = _hashed(
            {
                "stage": "quadratic_vertex",
                "outer_held_family_id": outer,
                "inner_held_family_id": inner,
                "lambda": proposed_lambda,
                "provider_artifact_sha256": vertex_providers[
                    inner
                ].artifact_sha256,
                "runtime_provider_artifact_sha256": (
                    _runtime_provider_artifact_sha256(vertex_providers[inner])
                ),
                "lineage_wrapper_not_inference_executor": True,
                "manifest_sha256": vertex_manifest["artifact_sha256"],
                "response_trace": vertex_traces[inner],
                "objective": macro,
                "objectives_by_example": dict(sorted(objectives.items())),
                "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                "supervised_full_vocab_logits_sha256s": dict(
                    sorted(logits_hashes.items())
                ),
                "execution_sha256s": dict(sorted(execution_hashes.items())),
                "capability_receipt": capability_receipt,
                "all_vertex_providers_frozen_before_score": True,
                "endpoint_vertex_exact_v20m_output_anchor": endpoint_anchor,
                "outer_family_absent_from_fit_and_score": True,
                "exact_execution": True,
                "finite": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
    vertex_score_receipt = (
        _shrinkage_fit.build_soft_polarity_simplex_shrinkage_vertex_score_receipt(
            anchor_receipt=anchor_receipt,
            proposal_receipt=proposal_receipt,
            exact_vertex_objectives_by_family=vertex_objectives,
        )
    )
    selection_receipt = (
        _shrinkage_fit.build_soft_polarity_simplex_shrinkage_selection_receipt(
            anchor_receipt=anchor_receipt,
            proposal_receipt=proposal_receipt,
            vertex_score_receipt=vertex_score_receipt,
        )
    )
    selected_lambda = float(selection_receipt["selected_lambda"])
    shrinkage_selection = _hashed(
        {
            "outer_held_family_id": outer,
            "source_response": response,
            "matched_linear_response": linear_response,
            "v20m_response_selection_receipt_sha256": response_selection[
                "artifact_sha256"
            ],
            "half_provider_manifest": half_manifest,
            "half_evidence_by_family": half_evidence,
            "anchor_objectives_by_family_and_anchor": anchor_objectives,
            "core_anchor_receipt": anchor_receipt,
            "core_quadratic_proposal_receipt": proposal_receipt,
            "vertex_provider_manifest": vertex_manifest,
            "vertex_evidence_by_family": vertex_evidence,
            "core_vertex_score_receipt": vertex_score_receipt,
            "core_selection_receipt": selection_receipt,
            "endpoint_vertex_exact_v20m_output_anchor_by_family": (
                endpoint_vertex_anchor_by_family
            ),
            "all_endpoint_vertex_exact_v20m_output_anchors_passed": all(
                endpoint_vertex_anchor_by_family.values()
            ),
            "selected_lambda": selected_lambda,
            "selected_lambda_hex": selected_lambda.hex(),
            "selected_lambda_interior": 0.0 < selected_lambda < 1.0,
            "all_half_providers_frozen_before_any_half_score": True,
            "proposal_frozen_before_any_vertex_provider_or_score": True,
            "all_vertex_providers_frozen_before_any_vertex_score": True,
            "exact_additional_inner_execution_count": (
                2 * _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY
            ),
            "outer_held_family_used_for_fit_or_selection": False,
            "final_refit_or_calibration_b_used": False,
            "raw_provider_prompt_token_logit_h4_or_teacher_tensors_serialized": False,
        },
        domain=_SHRINKAGE_SELECTION_DOMAIN,
    )
    return v20m_inner, response_selection, shrinkage_selection


def _matched_v20l_boundary_response(
    authenticated_v20l_fold: Mapping[str, object], *, outer_family_id: str
) -> tuple[float, float, float]:
    """Map the precommitted V20l winner into the simplex boundary exactly."""

    fold = _mapping(
        authenticated_v20l_fold.get("fold_receipt"),
        label="V20n authenticated V20l fold receipt",
    )
    outer = _identifier(outer_family_id, label="V20n V20l boundary family")
    if fold.get("outer_held_family_id") != outer:
        raise ValueError("V20n V20l boundary family differs")
    radius, signed_mix = _v20l._response_pair(fold.get("selected_response"))
    return _simplex_parameters((radius, abs(signed_mix), signed_mix))


def _freeze_outer_providers(
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    outer_reflection_fit: Mapping[str, object],
    held_records: Sequence[object],
    *,
    selected_response: tuple[float, float, float],
    selected_lambda: float,
    outer_family_id: str,
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    outer = _identifier(outer_family_id, label="V20n outer provider family")
    response = _response_tuple(selected_response)
    shrinkage = float(selected_lambda)
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("V20n selected lambda must be inside [0,1]")
    linear_response = (response[0], 0.0, 0.0)
    v20l_boundary_response = _matched_v20l_boundary_response(
        authenticated_v20l_fold, outer_family_id=outer
    )
    reflected = _selected_direction(outer_reflection_fit)
    unreflected = _unreflected_direction(source_direction_receipt)
    mirror = tuple(-item for item in reflected)
    selected_variant_artifact = _sha(
        outer_reflection_fit.get("selected_variant_artifact_sha256"),
        label="V20n outer reflection variant",
    )
    source_direction_artifact = _sha(
        source_direction_receipt.get("artifact_sha256"),
        label="V20n outer unreflected direction",
    )
    reflection_fit_artifact = _sha(
        outer_reflection_fit.get("artifact_sha256"),
        label="V20n outer reflection fit",
    )

    reflected_provider, reflected_seed = _materialize_shrinkage_provider(
        endpoint,
        direction=reflected,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        lambda_=shrinkage,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_simplex_shrinkage_reflected",
    )
    unreflected_provider, unreflected_seed = _materialize_shrinkage_provider(
        endpoint,
        direction=unreflected,
        direction_artifact_sha256=source_direction_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        lambda_=shrinkage,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_same_simplex_response_unreflected",
    )
    mirror_provider, mirror_seed = _materialize_shrinkage_provider(
        endpoint,
        direction=mirror,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        lambda_=shrinkage,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_simplex_response_reflected_exact_mirror",
    )
    matched_v20m_provider, matched_v20m_seed = _materialize_provider(
        endpoint,
        direction=reflected,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_matched_v20m_simplex_reflected",
    )
    linear_provider, linear_seed = _materialize_provider(
        endpoint,
        direction=reflected,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=linear_response,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_matched_linear_reflected",
    )
    boundary_provider, boundary_seed = _materialize_provider(
        endpoint,
        direction=reflected,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=v20l_boundary_response,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_matched_v20l_boundary_reflected",
    )
    control_seed = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "outer_held_family_id": outer,
            "reflection_fit_sha256": reflection_fit_artifact,
            "selected_response": response,
            "role": "outer_fixed_controls",
            "held_rows_used": False,
        },
        domain=_OUTER_MANIFEST_DOMAIN,
    )
    providers: dict[str, object] = {
        "base": endpoint.base_provider,
        "fixed_plus": (
            build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
                endpoint.base_provider,
                endpoint.proposal_provider,
                polarity=1,
                transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
                transfer_evidence_sha256=control_seed,
            )
        ),
        "fixed_minus": (
            build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
                endpoint.base_provider,
                endpoint.proposal_provider,
                polarity=-1,
                transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
                transfer_evidence_sha256=control_seed,
            )
        ),
        "matched_linear_reflected": linear_provider,
        "matched_v20l_boundary_reflected": boundary_provider,
        "same_simplex_response_unreflected": unreflected_provider,
        "simplex_shrinkage_reflected": reflected_provider,
        "simplex_response_reflected_exact_mirror": mirror_provider,
        "matched_v20m_simplex_reflected": matched_v20m_provider,
    }
    if tuple(providers) != _ARMS or len(
        {provider.artifact_sha256 for provider in providers.values()}
    ) != len(_ARMS):
        raise RuntimeError("V20n outer provider arm artifacts are not distinct")

    receipts = {
        "base": _provider_receipt(providers["base"], role="base"),
        "fixed_plus": _provider_receipt(
            providers["fixed_plus"], role="fixed_plus"
        ),
        "fixed_minus": _provider_receipt(
            providers["fixed_minus"], role="fixed_minus"
        ),
        "matched_linear_reflected": _provider_receipt(
            providers["matched_linear_reflected"],
            role="matched_linear_reflected",
            response=linear_response,
            direction=reflected,
        ),
        "matched_v20l_boundary_reflected": _provider_receipt(
            providers["matched_v20l_boundary_reflected"],
            role="matched_v20l_boundary_reflected",
            response=v20l_boundary_response,
            direction=reflected,
        ),
        "same_simplex_response_unreflected": _shrinkage_provider_receipt(
            providers["same_simplex_response_unreflected"],
            role="same_simplex_response_unreflected",
            response=response,
            lambda_=shrinkage,
            direction=unreflected,
        ),
        "simplex_shrinkage_reflected": _shrinkage_provider_receipt(
            providers["simplex_shrinkage_reflected"],
            role="simplex_shrinkage_reflected",
            response=response,
            lambda_=shrinkage,
            direction=reflected,
        ),
        "simplex_response_reflected_exact_mirror": _shrinkage_provider_receipt(
            providers["simplex_response_reflected_exact_mirror"],
            role="simplex_response_reflected_exact_mirror",
            response=response,
            lambda_=shrinkage,
            direction=mirror,
        ),
        "matched_v20m_simplex_reflected": _provider_receipt(
            providers["matched_v20m_simplex_reflected"],
            role="matched_v20m_simplex_reflected",
            response=response,
            direction=reflected,
        ),
    }
    traces = {
        arm: _provider_trace(providers[arm], held_records, role=arm)
        for arm in _ARMS
    }
    manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "source_direction_receipt_sha256": source_direction_artifact,
            "outer_reflection_fit_receipt_sha256": reflection_fit_artifact,
            "selected_variant_artifact_sha256": selected_variant_artifact,
            "selected_response": response,
            "selected_lambda": shrinkage,
            "selected_lambda_hex": shrinkage.hex(),
            "matched_linear_response": linear_response,
            "matched_v20l_boundary_response": v20l_boundary_response,
            "matched_v20l_boundary_source_fold_sha256": (
                authenticated_v20l_fold["fragment_sha256"]
            ),
            "matched_v20m_source_fold_sha256": authenticated_v20m_fold[
                "fragment_sha256"
            ],
            "arm_order": _ARMS,
            "provider_artifact_sha256s": {
                arm: providers[arm].artifact_sha256 for arm in _ARMS
            },
            "runtime_provider_artifact_sha256s": {
                arm: _runtime_provider_artifact_sha256(providers[arm])
                for arm in _ARMS
            },
            "provider_receipts": receipts,
            "response_trace_sha256s": {
                arm: traces[arm]["artifact_sha256"] for arm in _ARMS
            },
            "soft_provider_transfer_evidence_sha256s": {
                "matched_linear_reflected": linear_seed,
                "matched_v20l_boundary_reflected": boundary_seed,
                "same_simplex_response_unreflected": unreflected_seed,
                "simplex_shrinkage_reflected": reflected_seed,
                "simplex_response_reflected_exact_mirror": mirror_seed,
                "matched_v20m_simplex_reflected": matched_v20m_seed,
            },
            "fixed_control_transfer_evidence_sha256": control_seed,
            "all_nine_providers_frozen_before_outer_capability": True,
            "all_nine_traces_frozen_before_outer_capability": True,
            "outer_capability_count_at_freeze": 0,
            "outer_objectives_or_teacher_rows_used_at_freeze": False,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=_OUTER_MANIFEST_DOMAIN,
    )
    return providers, manifest, traces


def _score_outer_arms(
    context: object,
    endpoint: _v20g._EndpointLive,
    records: Sequence[object],
    teacher_vault: object,
    source_direction_receipt: Mapping[str, object],
    outer_reflection_fit: Mapping[str, object],
    *,
    selected_response: tuple[float, float, float],
    selected_lambda: float,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    outer = _identifier(outer_family_id, label="V20n held outer family")
    held = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id == outer)
    )
    if len(held) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20n outer-held prompt geometry differs")
    providers, manifest, traces = _freeze_outer_providers(
        endpoint,
        source_direction_receipt,
        outer_reflection_fit,
        held,
        selected_response=selected_response,
        selected_lambda=selected_lambda,
        outer_family_id=outer,
        authenticated_v20l_fold=authenticated_v20l_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
    )
    if (
        manifest.get("all_nine_providers_frozen_before_outer_capability") is not True
        or manifest.get("all_nine_traces_frozen_before_outer_capability") is not True
        or manifest.get("outer_capability_count_at_freeze") != 0
        or manifest.get("outer_objectives_or_teacher_rows_used_at_freeze")
        is not False
    ):
        raise PermissionError("V20n outer freeze barrier is not satisfied")

    trace_bundle_sha = _v14._sha256(
        {arm: traces[arm]["artifact_sha256"] for arm in _ARMS},
        domain=_OUTER_EXECUTION_DOMAIN,
    )
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in held), held_family_id=None
    )
    objective_by_arm: dict[str, float] = {}
    evidence_by_arm: dict[str, dict[str, object]] = {}
    for arm in _ARMS:
        seed = _v14._sha256(
            {
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle_sha,
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": providers[arm].artifact_sha256,
                "runtime_provider_artifact_sha256": (
                    _runtime_provider_artifact_sha256(providers[arm])
                ),
                "lineage_wrapper_not_inference_executor": isinstance(
                    providers[arm],
                    AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider,
                ),
                "all_outer_arms_frozen": True,
            },
            domain=_OUTER_EXECUTION_DOMAIN,
        )
        objectives, h4_hashes, logits_hashes, execution_hashes = (
            _score_exact_provider(
                context,
                held,
                capability,
                provider=providers[arm],
                phase="outer_family_disjoint_mechanism_score",
                outer_family_id=outer,
                inner_family_id=None,
                role=arm,
                evidence_sha256=seed,
                domain=_OUTER_EXECUTION_DOMAIN,
            )
        )
        macro, family_scores = _v19._family_equal_mean(objectives, held)
        if set(family_scores) != {outer}:
            raise RuntimeError("V20n outer score family geometry differs")
        objective_by_arm[arm] = macro
        evidence_by_arm[arm] = _hashed(
            {
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": providers[arm].artifact_sha256,
                "runtime_provider_artifact_sha256": (
                    _runtime_provider_artifact_sha256(providers[arm])
                ),
                "lineage_wrapper_not_inference_executor": isinstance(
                    providers[arm],
                    AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider,
                ),
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "response_trace": traces[arm],
                "objective": macro,
                "objectives_by_example": dict(sorted(objectives.items())),
                "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                "supervised_full_vocab_logits_sha256s": dict(
                    sorted(logits_hashes.items())
                ),
                "execution_sha256s": dict(sorted(execution_hashes.items())),
                "exact_execution": True,
                "finite": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=_OUTER_EXECUTION_DOMAIN,
        )
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(record.sequence.example_id for record in held),
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ARMS),
        label="V20n outer-held capability",
    )
    inherited_arms = _mapping(
        _mapping(
            authenticated_v20g_fold.get("held_evidence"),
            label="V20n inherited V20g held evidence",
        ).get("arm_evidence"),
        label="V20n inherited V20g held arms",
    )
    control_anchors: dict[str, bool] = {}
    for arm in ("base", "fixed_plus", "fixed_minus"):
        inherited = _mapping(
            inherited_arms.get(arm), label=f"V20n inherited V20g {arm} arm"
        )
        current = evidence_by_arm[arm]
        control_anchors[arm] = (
            float(current["objective"]) == float(inherited["objective"])
            and dict(
                _mapping(
                    current.get("objectives_by_example"),
                    label=f"V20n current {arm} objectives",
                )
            )
            == dict(
                _mapping(
                    inherited.get("objectives_by_example"),
                    label=f"V20n inherited {arm} objectives",
                )
            )
            and dict(
                _mapping(
                    current.get("post_cast_h4_sha256s"),
                    label=f"V20n current {arm} H4 hashes",
                )
            )
            == dict(
                _mapping(
                    inherited.get("post_cast_h4_sha256s"),
                    label=f"V20n inherited {arm} H4 hashes",
                )
            )
            and dict(
                _mapping(
                    current.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20n current {arm} logits hashes",
                )
            )
            == dict(
                _mapping(
                    inherited.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20n inherited {arm} logits hashes",
                )
            )
        )
    if not all(control_anchors.values()):
        raise RuntimeError("V20n outer control output anchor differs from V20g")
    inherited_v20l_arm = _mapping(
        _mapping(
            _mapping(
                authenticated_v20l_fold.get("held_evidence"),
                label="V20n inherited V20l held evidence",
            ).get("arm_evidence"),
            label="V20n inherited V20l held arms",
        ).get("signed_stack_reflected"),
        label="V20n inherited V20l selected boundary arm",
    )
    current_boundary = evidence_by_arm["matched_v20l_boundary_reflected"]
    v20l_boundary_anchor = (
        float(current_boundary["objective"])
        == float(inherited_v20l_arm["objective"])
        and dict(
            _mapping(
                current_boundary.get("objectives_by_example"),
                label="V20n current V20l boundary objectives",
            )
        )
        == dict(
            _mapping(
                inherited_v20l_arm.get("objectives_by_example"),
                label="V20n inherited V20l boundary objectives",
            )
        )
        and dict(
            _mapping(
                current_boundary.get("post_cast_h4_sha256s"),
                label="V20n current V20l boundary H4 hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20l_arm.get("post_cast_h4_sha256s"),
                label="V20n inherited V20l boundary H4 hashes",
            )
        )
        and dict(
            _mapping(
                current_boundary.get("supervised_full_vocab_logits_sha256s"),
                label="V20n current V20l boundary logits hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20l_arm.get("supervised_full_vocab_logits_sha256s"),
                label="V20n inherited V20l boundary logits hashes",
            )
        )
    )
    if not v20l_boundary_anchor:
        raise RuntimeError(
            "V20n matched boundary failed exact V20l objective/output reproduction"
        )
    inherited_v20m_arm = _mapping(
        _mapping(
            _mapping(
                authenticated_v20m_fold.get("held_evidence"),
                label="V20n inherited V20m held evidence",
            ).get("arm_evidence"),
            label="V20n inherited V20m held arms",
        ).get("simplex_response_reflected"),
        label="V20n inherited V20m selected simplex arm",
    )
    current_v20m = evidence_by_arm["matched_v20m_simplex_reflected"]
    v20m_anchor = (
        float(current_v20m["objective"]) == float(inherited_v20m_arm["objective"])
        and dict(
            _mapping(
                current_v20m.get("objectives_by_example"),
                label="V20n current V20m objectives",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_arm.get("objectives_by_example"),
                label="V20n inherited V20m objectives",
            )
        )
        and dict(
            _mapping(
                current_v20m.get("post_cast_h4_sha256s"),
                label="V20n current V20m H4 hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_arm.get("post_cast_h4_sha256s"),
                label="V20n inherited V20m H4 hashes",
            )
        )
        and dict(
            _mapping(
                current_v20m.get("supervised_full_vocab_logits_sha256s"),
                label="V20n current V20m logits hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_arm.get(
                    "supervised_full_vocab_logits_sha256s"
                ),
                label="V20n inherited V20m logits hashes",
            )
        )
    )
    if not v20m_anchor:
        raise RuntimeError(
            "V20n matched V20m arm failed exact objective/output reproduction"
        )
    base_logits = _mapping(
        evidence_by_arm["base"].get("supervised_full_vocab_logits_sha256s"),
        label="V20n base output hashes",
    )
    candidate_logits = _mapping(
        evidence_by_arm[_PRIMARY_ARM].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20n candidate output hashes",
    )
    candidate_changed = any(
        candidate_logits[example] != base_logits[example] for example in base_logits
    )
    boundary_logits = _mapping(
        evidence_by_arm["matched_v20l_boundary_reflected"].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20n matched V20l boundary output hashes",
    )
    candidate_changed_from_v20l_boundary = any(
        candidate_logits[example] != boundary_logits[example]
        for example in candidate_logits
    )
    v20m_logits = _mapping(
        evidence_by_arm["matched_v20m_simplex_reflected"].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20n matched V20m output hashes",
    )
    linear_logits = _mapping(
        evidence_by_arm["matched_linear_reflected"].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20n matched linear output hashes",
    )
    candidate_changed_from_v20m = any(
        candidate_logits[example] != v20m_logits[example]
        for example in candidate_logits
    )
    candidate_changed_from_linear = any(
        candidate_logits[example] != linear_logits[example]
        for example in candidate_logits
    )
    selected_lambda_interior = 0.0 < float(selected_lambda) < 1.0
    interior_exact_distinct_from_both_endpoints = (
        not selected_lambda_interior
        or (candidate_changed_from_v20m and candidate_changed_from_linear)
    )
    health = all(
        evidence_by_arm[arm]["finite"] is True
        and traces[arm]["finite"] is True
        and traces[arm]["pointwise_trust_passed"] is True
        and traces[arm]["endpoint_conditional_ranks_are_16"] is True
        for arm in _ARMS
    )
    held_evidence = _hashed(
        {
            "outer_held_family_id": outer,
            "outer_manifest_sha256": manifest["artifact_sha256"],
            "arm_evidence": evidence_by_arm,
            "capability_receipt": capability_receipt,
            "all_nine_providers_and_traces_frozen_before_outer_capability": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_outer_execution_count": len(_ARMS) * len(held),
            "v20g_control_output_anchors": control_anchors,
            "all_v20g_control_output_anchors_passed": all(
                control_anchors.values()
            ),
            "matched_v20l_boundary_exact_output_anchor_passed": (
                v20l_boundary_anchor
            ),
            "matched_v20m_exact_output_anchor_passed": v20m_anchor,
            "raw_prompts_tokens_logits_h4_or_teacher_rows_serialized": False,
        },
        domain=_OUTER_EXECUTION_DOMAIN,
    )
    fold_receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "selected_response": _response_tuple(selected_response),
            "selected_response_key": _response_key(selected_response),
            "selected_lambda": float(selected_lambda),
            "selected_lambda_hex": float(selected_lambda).hex(),
            "selected_lambda_interior": selected_lambda_interior,
            "selected_variant_id": outer_reflection_fit["selected_variant_id"],
            "selected_variant_artifact_sha256": outer_reflection_fit[
                "selected_variant_artifact_sha256"
            ],
            "arm_order": _ARMS,
            "held_objective_by_arm": objective_by_arm,
            "candidate_provider_artifact_sha256": manifest[
                "provider_artifact_sha256s"
            ][_PRIMARY_ARM],
            "base_provider_artifact_sha256": manifest[
                "provider_artifact_sha256s"
            ]["base"],
            "candidate_provider_distinct_from_base": (
                manifest["provider_artifact_sha256s"][_PRIMARY_ARM]
                != manifest["provider_artifact_sha256s"]["base"]
            ),
            "candidate_exact_execution_changed_from_base": candidate_changed,
            "candidate_exact_execution_changed_from_matched_v20l_boundary": (
                candidate_changed_from_v20l_boundary
            ),
            "candidate_exact_execution_changed_from_matched_v20m": (
                candidate_changed_from_v20m
            ),
            "candidate_exact_execution_changed_from_matched_linear": (
                candidate_changed_from_linear
            ),
            "interior_candidate_exact_distinct_from_both_endpoints": (
                interior_exact_distinct_from_both_endpoints
            ),
            "selected_radius_positive": _response_tuple(selected_response)[0] > 0.0,
            "selected_u_positive": _response_tuple(selected_response)[1] > 0.0,
            "selected_interior_simplex_response": (
                0.0 < abs(_response_tuple(selected_response)[2])
                < _response_tuple(selected_response)[1]
            ),
            "selected_boundary_simplex_response": (
                _response_tuple(selected_response)[1] > 0.0
                and abs(_response_tuple(selected_response)[2])
                == _response_tuple(selected_response)[1]
            ),
            "selected_zero_bias_simplex_response": (
                _response_tuple(selected_response)[1] > 0.0
                and _response_tuple(selected_response)[2] == 0.0
            ),
            "all_runtime_health_passed": health,
            "all_v20g_control_output_anchors_passed": all(
                control_anchors.values()
            ),
            "matched_v20l_boundary_exact_output_anchor_passed": (
                v20l_boundary_anchor
            ),
            "matched_v20m_exact_output_anchor_passed": v20m_anchor,
            "selection_frozen_before_outer_score": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_execution": True,
        },
        domain=_DECISION_DOMAIN,
    )
    return manifest, held_evidence, fold_receipt


def _execute_outer_fold(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    family_ids: Sequence[str],
    outer_family_id: str,
    panel_receipt: Mapping[str, object],
    authenticated_v20a_fold: Mapping[str, object],
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> _FoldLive:
    outer = _identifier(outer_family_id, label="V20n outer family")
    endpoint = _v20g._outer_endpoint(
        context,
        records,
        teacher_vault,
        family_ids=family_ids,
        outer_family_id=outer,
        panel_receipt=panel_receipt,
        authenticated_v20a_fold=authenticated_v20a_fold,
    )
    inherited_endpoint = _mapping(
        authenticated_v20g_fold.get("endpoint_receipt"),
        label="V20n inherited endpoint receipt",
    )
    inherited_evidence = _mapping(
        authenticated_v20g_fold.get("endpoint_evidence"),
        label="V20n inherited endpoint evidence",
    )
    if (
        _v14._canonical_json_bytes(endpoint.receipt)
        != _v14._canonical_json_bytes(inherited_endpoint)
        or _v14._canonical_json_bytes(endpoint.evidence)
        != _v14._canonical_json_bytes(inherited_evidence)
    ):
        raise RuntimeError("V20n reconstructed endpoint differs from pinned V20g")
    fit = _mapping(
        authenticated_v20g_fold.get("fit_receipt"),
        label="V20n inherited V20g fit receipt",
    )
    source_direction = _mapping(
        fit.get("direction_receipt"), label="V20n inherited V20g direction"
    )
    _v20g._core.validate_soft_polarity_direction_receipt(source_direction)
    if source_direction.get("held_family_id") != outer:
        raise RuntimeError("V20n inherited direction held family differs")

    inner_receipt, response_selection, shrinkage_selection = (
        _fit_inner_shrinkage(
        context,
        endpoint,
        source_direction,
        teacher_vault,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
        )
    )
    outer_reflection_fit = _reflection.build_soft_polarity_reflection_fit_receipt(
        direction_receipt=source_direction
    )
    _validate_v20i_reflection_lineage(
        inner_receipt=inner_receipt,
        outer_reflection_fit=outer_reflection_fit,
        authenticated_v20i_fold=authenticated_v20i_fold,
    )
    selected_response = _response_tuple(
        response_selection["selected_response"]
    )
    selected_lambda = float(shrinkage_selection["selected_lambda"])
    provider_manifest, held_evidence, fold_receipt = _score_outer_arms(
        context,
        endpoint,
        records,
        teacher_vault,
        source_direction,
        outer_reflection_fit,
        selected_response=selected_response,
        selected_lambda=selected_lambda,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20l_fold=authenticated_v20l_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
    )
    return _FoldLive(
        endpoint=endpoint,
        inner_receipt=inner_receipt,
        outer_reflection_fit=outer_reflection_fit,
        response_selection=response_selection,
        shrinkage_selection=shrinkage_selection,
        provider_manifest=provider_manifest,
        held_evidence=held_evidence,
        fold_receipt=fold_receipt,
    )


_FOLD_FRAGMENT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "masked_direction_protocol_sha256",
        "simplex_response_fit_protocol_sha256",
        "simplex_shrinkage_fit_protocol_sha256",
        "simplex_shrinkage_provider_protocol_sha256",
        "exact_objective_kind",
        "source_artifact_sha256",
        "panel_receipt_sha256",
        "bridge_binding_sha256",
        "v20g_fold_fragment_sha256",
        "v20i_fold_fragment_sha256",
        "v20j_fold_fragment_sha256",
        "v20k_fold_fragment_sha256",
        "v20l_fold_fragment_sha256",
        "v20m_fold_fragment_sha256",
        "outer_held_family_id",
        "endpoint_receipt",
        "endpoint_evidence",
        "inner_receipt",
        "outer_reflection_fit_receipt",
        "response_selection_receipt",
        "shrinkage_selection_receipt",
        "provider_manifest",
        "held_evidence",
        "fold_receipt",
        "fixed_schedule_completed",
        "candidate",
        "provider_sidecar",
        "fragment_sha256",
    }
)


def _fold_payload(
    live: _FoldLive,
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": (
            _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
        ),
        "masked_direction_protocol_sha256": (
            _reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
        ),
        "simplex_response_fit_protocol_sha256": (
            _simplex_response_fit.SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
        ),
        "simplex_shrinkage_fit_protocol_sha256": (
            _shrinkage_fit.SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
        ),
        "simplex_shrinkage_provider_protocol_sha256": (
            FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256
        ),
        "exact_objective_kind": (
            "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
        ),
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20g_fold_fragment_sha256": authenticated_v20g_fold[
            "fragment_sha256"
        ],
        "v20i_fold_fragment_sha256": authenticated_v20i_fold[
            "fragment_sha256"
        ],
        "v20j_fold_fragment_sha256": source[
            "v20j_fold_fragment_sha256s_by_family"
        ][outer_family_id],
        "v20k_fold_fragment_sha256": source[
            "v20k_fold_fragment_sha256s_by_family"
        ][outer_family_id],
        "v20l_fold_fragment_sha256": authenticated_v20l_fold[
            "fragment_sha256"
        ],
        "v20m_fold_fragment_sha256": authenticated_v20m_fold[
            "fragment_sha256"
        ],
        "outer_held_family_id": outer_family_id,
        "endpoint_receipt": live.endpoint.receipt,
        "endpoint_evidence": live.endpoint.evidence,
        "inner_receipt": live.inner_receipt,
        "outer_reflection_fit_receipt": live.outer_reflection_fit,
        "response_selection_receipt": live.response_selection,
        "shrinkage_selection_receipt": live.shrinkage_selection,
        "provider_manifest": live.provider_manifest,
        "held_evidence": live.held_evidence,
        "fold_receipt": live.fold_receipt,
        "fixed_schedule_completed": True,
        "candidate": None,
        "provider_sidecar": None,
    }


def _validate_exact_score_bundle(
    evidence: Mapping[str, object],
    *,
    expected_example_ids: Sequence[str],
    label: str,
) -> tuple[dict[str, float], dict[str, str], dict[str, str], dict[str, str]]:
    examples = tuple(
        sorted(
            _identifier(item, label=f"{label} example")
            for item in expected_example_ids
        )
    )
    if len(examples) != _PROMPTS_PER_FAMILY or len(set(examples)) != len(examples):
        raise ValueError(f"{label} example geometry differs")
    objectives = {
        _identifier(example, label=f"{label} objective example"): float(value)
        for example, value in _mapping(
            evidence.get("objectives_by_example"),
            label=f"{label} objectives",
        ).items()
    }
    hashes: list[dict[str, str]] = []
    for field, field_label in (
        ("post_cast_h4_sha256s", "H4"),
        ("supervised_full_vocab_logits_sha256s", "logits"),
        ("execution_sha256s", "execution"),
    ):
        values = {
            _identifier(example, label=f"{label} {field_label} example"): _sha(
                value, label=f"{label} {field_label} hash"
            )
            for example, value in _mapping(
                evidence.get(field), label=f"{label} {field_label} hashes"
            ).items()
        }
        hashes.append(values)
    h4_hashes, logits_hashes, execution_hashes = hashes
    if (
        set(objectives) != set(examples)
        or set(h4_hashes) != set(examples)
        or set(logits_hashes) != set(examples)
        or set(execution_hashes) != set(examples)
        or not all(math.isfinite(value) for value in objectives.values())
    ):
        raise ValueError(f"{label} exact output geometry differs")
    macro = math.fsum(objectives[example] for example in examples) / len(examples)
    if not math.isfinite(macro) or float(evidence.get("objective", math.nan)) != macro:
        raise ValueError(f"{label} exact objective replay differs")
    return objectives, h4_hashes, logits_hashes, execution_hashes


def _validate_inner_receipt(
    value: Mapping[str, object],
    *,
    source_direction: Mapping[str, object],
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> Mapping[str, object]:
    receipt = _validate_hashed(
        value, domain=_INNER_FIT_DOMAIN, label="V20n inner receipt"
    )
    inherited_v20m_inner = _mapping(
        authenticated_v20m_fold.get("inner_receipt"),
        label="V20n authenticated V20m inner receipt",
    )
    inherited_v20m_inner_evidence = _mapping(
        inherited_v20m_inner.get("inner_evidence_by_family"),
        label="V20n authenticated V20m inner evidence",
    )
    inherited_v20m_selection = _mapping(
        authenticated_v20m_fold.get("response_selection_receipt"),
        label="V20n authenticated V20m response selection",
    )
    endpoint_receipt = _mapping(
        authenticated_v20g_fold.get("endpoint_receipt"),
        label="V20n inherited endpoint receipt",
    )
    inherited_bridge = authenticated_v20g_fold.get("bridge_binding_sha256")
    if inherited_bridge is None:
        inherited_bridge = endpoint_receipt.get("bridge_binding_sha256")
    if inherited_bridge is None:
        inherited_bridge = source_direction.get("bridge_binding_sha256")
    expected_bridge_binding = _sha(
        inherited_bridge,
        label="V20n inherited bridge binding",
    )
    if (
        receipt.get("outer_held_family_id") != outer_family_id
        or receipt.get("source_direction_receipt_sha256")
        != source_direction.get("artifact_sha256")
        or receipt.get(
            "all_inner_fits_and_providers_frozen_before_any_inner_capability"
        )
        is not True
        or _response_order(receipt.get("response_order", ())) != _RESPONSES
        or int(receipt.get("exact_inner_execution_count", -1))
        != _INNER_FAMILY_COUNT * len(_RESPONSES) * _PROMPTS_PER_FAMILY
        or receipt.get("inner_endpoint_retrained_per_fold") is not False
        or receipt.get("inner_held_family_used_for_endpoint_fit") is not True
        or receipt.get("inner_claim_scope")
        != "conditional_response_LOFO_not_fully_nested_model_cross_validation"
        or receipt.get("outer_held_family_used_for_fit_or_selection") is not False
        or receipt.get("raw_provider_gradient_logits_h4_or_teacher_tensors_serialized")
        is not False
    ):
        raise ValueError("V20n inner receipt boundary differs")
    manifest = _validate_hashed(
        _mapping(
            receipt.get("inner_provider_manifest"),
            label="V20n inner provider manifest",
        ),
        domain=_INNER_MANIFEST_DOMAIN,
        label="V20n inner provider manifest",
    )
    families = tuple(
        _identifier(item, label="V20n inner family")
        for item in _sequence(
            manifest.get("inner_family_order"), label="V20n inner family order"
        )
    )
    source_families = tuple(
        _identifier(item, label="V20n source training family")
        for item in _sequence(
            source_direction.get("training_family_ids"),
            label="V20n source training families",
        )
    )
    training_ids_by_family = _mapping(
        source_direction.get("training_example_ids_by_family"),
        label="V20n source training example ids",
    )
    if (
        len(families) != _INNER_FAMILY_COUNT
        or len(set(families)) != len(families)
        or families != source_families
        or manifest.get("outer_held_family_id") != outer_family_id
        or _response_order(manifest.get("response_order", ())) != _RESPONSES
        or manifest.get("simplex_response_ladder_receipt_sha256")
        != _SIMPLEX_RESPONSE_LADDER_RECEIPT_SHA256
        or manifest.get("endpoint_receipt_sha256")
        != endpoint_receipt.get("artifact_sha256")
        or manifest.get("source_direction_receipt_sha256")
        != source_direction.get("artifact_sha256")
        or manifest.get(
            "all_seven_times_nineteen_providers_frozen_before_any_inner_capability"
        )
        is not True
        or manifest.get(
            "all_seven_times_nineteen_traces_frozen_before_any_inner_capability"
        )
        is not True
        or manifest.get("inner_capability_count_at_freeze") != 0
        or manifest.get("inner_objectives_or_teacher_rows_used_at_freeze")
        is not False
        or manifest.get("inner_endpoint_retrained_per_fold") is not False
        or manifest.get("inner_held_family_used_for_endpoint_fit") is not True
        or manifest.get("raw_provider_or_response_tensors_serialized") is not False
    ):
        raise ValueError("V20n inner manifest freeze geometry differs")
    if tuple(receipt.get("inner_family_order", ())) != families:
        raise ValueError("V20n inner receipt family order differs")
    if set(inherited_v20m_inner_evidence) != set(families):
        raise ValueError("V20n authenticated V20m inner family geometry differs")
    masked_hashes = _mapping(
        manifest.get("masked_direction_receipt_sha256s_by_inner_family"),
        label="V20n inner masked receipt hashes",
    )
    fit_hashes = _mapping(
        manifest.get("reflection_fit_receipt_sha256s_by_inner_family"),
        label="V20n inner reflection fit hashes",
    )
    variant_hashes = _mapping(
        manifest.get("selected_variant_artifact_sha256s_by_inner_family"),
        label="V20n inner selected variant hashes",
    )
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s_by_inner_family_and_response"),
        label="V20n inner provider hashes",
    )
    provider_receipts = _mapping(
        manifest.get("provider_receipts_by_inner_family_and_response"),
        label="V20n inner provider receipts",
    )
    transfer_hashes = _mapping(
        manifest.get(
            "provider_transfer_evidence_sha256s_by_inner_family_and_response"
        ),
        label="V20n inner provider transfer hashes",
    )
    trace_hashes = _mapping(
        manifest.get("response_trace_sha256s_by_inner_family_and_response"),
        label="V20n inner trace hashes",
    )
    family_set = set(families)
    if any(
        set(values) != family_set
        for values in (
            masked_hashes,
            fit_hashes,
            variant_hashes,
            provider_hashes,
            provider_receipts,
            transfer_hashes,
            trace_hashes,
        )
    ):
        raise ValueError("V20n inner manifest family bindings differ")
    gradient_evidence = _mapping(
        _mapping(
            authenticated_v20g_fold.get("fit_training_evidence"),
            label="V20n inherited fit evidence",
        ).get("gradient_evidence"),
        label="V20n inherited gradient evidence",
    )
    inherited_zero_objectives = _mapping(
        gradient_evidence.get("eta_zero_objectives_by_family"),
        label="V20n inherited eta-zero objectives",
    )
    inherited_zero_h4 = _mapping(
        gradient_evidence.get("post_cast_h4_sha256s"),
        label="V20n inherited eta-zero H4 hashes",
    )
    inherited_zero_logits = _mapping(
        gradient_evidence.get("supervised_full_vocab_logits_sha256s"),
        label="V20n inherited eta-zero logits hashes",
    )
    if set(inherited_zero_objectives) != family_set:
        raise ValueError("V20n inherited eta-zero family geometry differs")
    raw_inner = _mapping(
        receipt.get("inner_evidence_by_family"),
        label="V20n inner evidence map",
    )
    if set(raw_inner) != set(families):
        raise ValueError("V20n inner evidence family geometry differs")
    validated_inner: dict[str, Mapping[str, object]] = {}
    for family in families:
        evidence = _validate_hashed(
            _mapping(raw_inner[family], label="V20n inner family evidence"),
            domain=_INNER_EXECUTION_DOMAIN,
            label="V20n inner family evidence",
        )
        masked = _mapping(
            evidence.get("masked_direction_receipt"),
            label="V20n masked direction receipt",
        )
        _reflection.validate_soft_polarity_masked_direction_receipt(
            masked,
            source_direction_receipt=source_direction,
            expected_excluded_training_family_id=family,
        )
        reflection_fit = _mapping(
            evidence.get("reflection_fit_receipt"),
            label="V20n inner reflection fit",
        )
        _reflection.validate_soft_polarity_reflection_fit_receipt(
            reflection_fit, direction_receipt=masked
        )
        expected_inner_training = tuple(item for item in families if item != family)
        expected_examples = tuple(
            _identifier(item, label="V20n inner expected example")
            for item in _sequence(
                training_ids_by_family.get(family),
                label="V20n inner expected examples",
            )
        )
        objectives = _mapping(
            evidence.get("objective_by_response"),
            label="V20n inner objectives",
        )
        response_evidence = _mapping(
            evidence.get("response_evidence"),
            label="V20n inner response evidence",
        )
        inherited_v20m_responses = _mapping(
            _mapping(
                inherited_v20m_inner_evidence[family],
                label="V20n authenticated V20m inner family evidence",
            ).get("response_evidence"),
            label="V20n authenticated V20m response evidence",
        )
        if (
            set(objectives) != set(_RESPONSE_KEYS)
            or set(response_evidence) != set(_RESPONSE_KEYS)
            or set(inherited_v20m_responses) != set(_RESPONSE_KEYS)
            or evidence.get("outer_held_family_id") != outer_family_id
            or evidence.get("inner_held_family_id") != family
            or tuple(evidence.get("inner_training_family_ids", ()))
            != expected_inner_training
            or masked.get("artifact_sha256") != masked_hashes[family]
            or reflection_fit.get("artifact_sha256") != fit_hashes[family]
            or evidence.get("selected_variant_artifact_sha256")
            != variant_hashes[family]
            or evidence.get("held_family_used_for_direction_or_reflection_fit")
            is not False
            or evidence.get("held_family_used_for_endpoint_fit") is not True
            or evidence.get("endpoint_retrained_without_held_inner_family")
            is not False
            or evidence.get("all_inner_candidates_frozen_before_capability")
            is not True
            or evidence.get("zero_response_exact_v20g_eta_zero_output_anchor")
            is not True
            or int(evidence.get("exact_execution_count", -1))
            != len(_RESPONSES) * _PROMPTS_PER_FAMILY
            or evidence.get(
                "raw_prompts_tokens_logits_h4_gradients_or_teacher_rows_serialized"
            )
            is not False
        ):
            raise ValueError("V20n inner evidence schedule differs")
        family_provider_hashes = _mapping(
            provider_hashes[family], label="V20n inner family provider hashes"
        )
        family_provider_receipts = _mapping(
            provider_receipts[family], label="V20n inner family provider receipts"
        )
        family_transfer_hashes = _mapping(
            transfer_hashes[family],
            label="V20n inner family provider transfer hashes",
        )
        family_trace_hashes = _mapping(
            trace_hashes[family], label="V20n inner family trace hashes"
        )
        if any(
            set(values) != set(_RESPONSE_KEYS)
            for values in (
                family_provider_hashes,
                family_provider_receipts,
                family_transfer_hashes,
                family_trace_hashes,
            )
        ):
            raise ValueError("V20n inner manifest response bindings differ")
        trace_bundle_sha = _v14._sha256(
            {
                _response_key(response): family_trace_hashes[_response_key(response)]
                for response in _RESPONSES
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
        selected_direction = _selected_direction(reflection_fit)
        for response in _RESPONSES:
            key = _response_key(response)
            arm = _validate_hashed(
                _mapping(
                    response_evidence[key], label="V20n inner response arm evidence"
                ),
                domain=_INNER_EXECUTION_DOMAIN,
                label="V20n inner response arm evidence",
            )
            trace = _validate_hashed(
                _mapping(arm.get("response_trace"), label="V20n inner trace"),
                domain=_TRACE_DOMAIN,
                label="V20n inner trace",
            )
            provider_receipt = _validate_hashed(
                _mapping(
                    family_provider_receipts[key],
                    label="V20n inner provider receipt",
                ),
                domain=_PROVIDER_DOMAIN,
                label="V20n inner provider receipt",
            )
            score_bundle = _validate_exact_score_bundle(
                arm,
                expected_example_ids=expected_examples,
                label=f"V20n inner {family} response {key}",
            )
            arm_objectives, arm_h4, arm_logits, arm_executions = score_bundle
            inherited_v20m_arm = _mapping(
                inherited_v20m_responses[key],
                label="V20n authenticated V20m response arm",
            )
            for evidence_key in (
                "objectives_by_example",
                "post_cast_h4_sha256s",
                "supervised_full_vocab_logits_sha256s",
            ):
                if _v14._canonical_json_bytes(arm.get(evidence_key)) != (
                    _v14._canonical_json_bytes(
                        inherited_v20m_arm.get(evidence_key)
                    )
                ):
                    raise ValueError(
                        "V20n inner exact output differs from authenticated V20m"
                    )
            provider_artifact = _sha(
                family_provider_hashes[key], label="V20n inner provider hash"
            )
            transfer_artifact = _sha(
                family_transfer_hashes[key],
                label="V20n inner provider transfer hash",
            )
            expected_transfer_artifact = _provider_seed(
                endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
                direction_artifact_sha256=str(
                    reflection_fit["selected_variant_artifact_sha256"]
                ),
                reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
                response=response,
                direction=selected_direction,
                outer_family_id=outer_family_id,
                inner_family_id=family,
                role="inner_reflected_response_candidate",
            )
            _validate_provider_receipt_evidence(
                provider_receipt,
                expected_role="inner_reflected_response_candidate",
                expected_provider_artifact_sha256=provider_artifact,
                expected_endpoint_receipt=endpoint_receipt,
                expected_bridge_binding_sha256=expected_bridge_binding,
                authenticated_v20i_fold=authenticated_v20i_fold,
                expected_response=response,
                expected_direction=selected_direction,
                expected_transfer_evidence_sha256=expected_transfer_artifact,
            )
            trace_artifact = _sha(
                family_trace_hashes[key], label="V20n inner trace hash"
            )
            seed = _v14._sha256(
                {
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "trace_bundle_sha256": trace_bundle_sha,
                    "outer_held_family_id": outer_family_id,
                    "inner_held_family_id": family,
                    "response": response,
                    "provider_artifact_sha256": provider_artifact,
                    "all_inner_candidates_frozen": True,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
            expected_executions = {
                example: _execution_sha256(
                    phase="inner_conditional_leave_one_family_out_response_score",
                    outer_family_id=outer_family_id,
                    inner_family_id=family,
                    role="inner_reflected_response_candidate",
                    provider_artifact_sha256=provider_artifact,
                    example_id=example,
                    family_id=family,
                    objective=arm_objectives[example],
                    h4_sha256=arm_h4[example],
                    logits_sha256=arm_logits[example],
                    evidence_sha256=seed,
                    domain=_INNER_EXECUTION_DOMAIN,
                )
                for example in expected_examples
            }
            response_gain_hashes = _mapping(
                trace.get("response_gain_sha256s"),
                label="V20n inner response gain hashes",
            )
            for value in response_gain_hashes.values():
                _sha(value, label="V20n inner response gain hash")
            expected_corners = _box_corner_scores(selected_direction)
            expected_certificate = fisher_soft_polarity_simplex_response_box_certificate(
                _v20g._eta_tensor(selected_direction),
                radius=response[0],
                shrink_mass=response[1],
                polarity_bias=response[2],
            )
            expected_trace_arm = (
                f"inner_{family}_radius_{response[0].hex()}_u_{response[1].hex()}_"
                f"v_{response[2].hex()}"
            )
            if (
                _response_tuple(arm.get("response")) != response
                or float(arm.get("objective", math.nan))
                != float(objectives[key])
                or arm.get("provider_artifact_sha256") != provider_artifact
                or arm.get("inner_manifest_sha256")
                != manifest.get("artifact_sha256")
                or trace.get("artifact_sha256") != trace_artifact
                or trace.get("provider_artifact_sha256") != provider_artifact
                or trace.get("arm") != expected_trace_arm
                or tuple(trace.get("scored_family_ids", ())) != (family,)
                or set(response_gain_hashes) != set(expected_examples)
                or provider_receipt.get("provider_artifact_sha256")
                != provider_artifact
                or transfer_artifact != expected_transfer_artifact
                or provider_receipt.get("transfer_protocol_sha256")
                != _TRANSFER_PROTOCOL_SHA256
                or provider_receipt.get("transfer_evidence_sha256")
                != transfer_artifact
                or provider_receipt.get("role")
                != "inner_reflected_response_candidate"
                or _response_tuple(provider_receipt.get("response"))
                != response
                or provider_receipt.get("response_key") != key
                or float(
                    provider_receipt.get("radius", math.nan)
                )
                != response[0]
                or float(
                    provider_receipt.get("u", math.nan)
                )
                != response[1]
                or float(provider_receipt.get("v", math.nan)) != response[2]
                or tuple(provider_receipt.get("direction", ()))
                != selected_direction
                or tuple(
                    provider_receipt.get("direction_box_corner_scores", ())
                )
                != expected_corners
                or _v14._canonical_json_bytes(
                    _mapping(
                        provider_receipt.get("box_certificate"),
                        label="V20n simplex_response box certificate",
                    )
                )
                != _v14._canonical_json_bytes(expected_certificate)
                or int(provider_receipt.get("conditional_rank", -1))
                != _CONDITIONAL_RANK
                or provider_receipt.get("analysis_only") is not True
                or provider_receipt.get("raw_provider_tensors_serialized")
                is not False
                or arm_executions != expected_executions
                or arm.get("exact_execution") is not True
                or arm.get("finite") is not True
                or arm.get("raw_logits_h4_teacher_rows_or_tensors_serialized")
                is not False
                or trace.get("finite") is not True
                or trace.get("pointwise_trust_passed") is not True
                or trace.get("endpoint_conditional_ranks_are_16") is not True
                or trace.get("raw_response_or_modal_tensors_serialized")
                is not False
            ):
                raise ValueError("V20n inner response evidence differs")
        _v20b._validate_capability_receipt(
            evidence.get("capability_receipt"),
            expected_example_ids=expected_examples,
            expected_family_count=1,
            expected_held_family_id=outer_family_id,
            expected_accesses_per_example=len(_RESPONSES),
            label="V20n resumed inner-held capability",
        )
        zero = _mapping(
            response_evidence[_response_key((0.0, 0.0, 0.0))],
            label="V20n inner zero-response evidence",
        )
        expected_zero_objectives = {
            _identifier(example, label="V20n inherited zero objective example"): float(
                objective
            )
            for example, objective in _mapping(
                inherited_zero_objectives[family],
                label="V20n inherited family zero objectives",
            ).items()
        }
        zero_anchor = (
            dict(
                _mapping(
                    zero.get("objectives_by_example"),
                    label="V20n zero objectives",
                )
            )
            == expected_zero_objectives
            and dict(
                _mapping(
                    zero.get("post_cast_h4_sha256s"),
                    label="V20n zero H4 hashes",
                )
            )
            == {
                example: inherited_zero_h4[example]
                for example in expected_examples
            }
            and dict(
                _mapping(
                    zero.get("supervised_full_vocab_logits_sha256s"),
                    label="V20n zero logits hashes",
                )
            )
            == {
                example: inherited_zero_logits[example]
                for example in expected_examples
            }
        )
        if (
            not zero_anchor
            or evidence.get("zero_response_exact_v20g_eta_zero_output_anchor")
            is not zero_anchor
        ):
            raise ValueError("V20n inner zero-response V20g output anchor differs")
        validated_inner[family] = evidence
    selection = _aggregate_response_selection(validated_inner)
    persisted_selection = _mapping(
        receipt.get("response_selection_receipt"),
        label="V20n persisted response selection",
    )
    if _v14._canonical_json_bytes(selection) != _v14._canonical_json_bytes(
        persisted_selection
    ):
        raise ValueError("V20n inner response selection replay differs")
    for key in (
        "objectives_by_inner_family_and_response",
        "family_equal_objective_by_response",
        "simplex_response_selection_receipt",
        "selected_response",
    ):
        if _v14._canonical_json_bytes(selection.get(key)) != (
            _v14._canonical_json_bytes(inherited_v20m_selection.get(key))
        ):
            raise ValueError(
                f"V20n response selection {key} differs from authenticated V20m"
            )
    return receipt


def _validate_shrinkage_selection(
    value: Mapping[str, object],
    *,
    response_selection: Mapping[str, object],
    inner_receipt: Mapping[str, object],
    endpoint_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20i_fold: Mapping[str, object],
) -> Mapping[str, object]:
    receipt = _validate_hashed(
        value,
        domain=_SHRINKAGE_SELECTION_DOMAIN,
        label="V20n shrinkage selection",
    )
    outer = _identifier(outer_family_id, label="V20n shrinkage outer family")
    response = _response_tuple(response_selection.get("selected_response"))
    linear_response = (response[0], 0.0, 0.0)
    inner_evidence = _mapping(
        inner_receipt.get("inner_evidence_by_family"),
        label="V20n shrinkage V20m inner evidence",
    )
    families = tuple(sorted(inner_evidence))
    if (
        receipt.get("outer_held_family_id") != outer
        or _response_tuple(receipt.get("source_response")) != response
        or _response_tuple(receipt.get("matched_linear_response"))
        != linear_response
        or receipt.get("v20m_response_selection_receipt_sha256")
        != response_selection.get("artifact_sha256")
        or receipt.get("all_half_providers_frozen_before_any_half_score")
        is not True
        or receipt.get("proposal_frozen_before_any_vertex_provider_or_score")
        is not True
        or receipt.get("all_vertex_providers_frozen_before_any_vertex_score")
        is not True
        or receipt.get("outer_held_family_used_for_fit_or_selection") is not False
        or receipt.get("final_refit_or_calibration_b_used") is not False
        or receipt.get(
            "raw_provider_prompt_token_logit_h4_or_teacher_tensors_serialized"
        )
        is not False
        or int(receipt.get("exact_additional_inner_execution_count", -1))
        != 2 * _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY
    ):
        raise ValueError("V20n shrinkage selection boundary differs")

    def validate_stage(
        *,
        stage: str,
        expected_lambda: float,
        manifest_key: str,
        evidence_key: str,
        role: str,
        freeze_flag: str,
    ) -> tuple[dict[str, float], dict[str, bool]]:
        manifest = _validate_hashed(
            _mapping(receipt.get(manifest_key), label=f"V20n {stage} manifest"),
            domain=_INNER_MANIFEST_DOMAIN,
            label=f"V20n {stage} manifest",
        )
        evidence_by_family = _mapping(
            receipt.get(evidence_key), label=f"V20n {stage} evidence"
        )
        provider_hashes = _mapping(
            manifest.get("provider_artifact_sha256s_by_inner_family"),
            label=f"V20n {stage} provider hashes",
        )
        runtime_hashes = _mapping(
            manifest.get("runtime_provider_artifact_sha256s_by_inner_family"),
            label=f"V20n {stage} runtime provider hashes",
        )
        receipts = _mapping(
            manifest.get("provider_receipts_by_inner_family"),
            label=f"V20n {stage} provider receipts",
        )
        traces = _mapping(
            manifest.get("trace_sha256s_by_inner_family"),
            label=f"V20n {stage} trace hashes",
        )
        transfers = _mapping(
            manifest.get(
                "provider_transfer_evidence_sha256s_by_inner_family"
            ),
            label=f"V20n {stage} transfer hashes",
        )
        stage_prefix = "half" if stage == "lambda_half" else "vertex"
        if (
            manifest.get("stage") != stage
            or manifest.get("outer_held_family_id") != outer
            or tuple(manifest.get("inner_family_order", ())) != families
            or _response_tuple(manifest.get("source_response")) != response
            or float(manifest.get("lambda", math.nan)) != expected_lambda
            or manifest.get("lambda_hex") != expected_lambda.hex()
            or manifest.get(freeze_flag) is not True
            or int(
                manifest.get(f"{stage_prefix}_capability_count_at_freeze", -1)
            )
            != 0
            or manifest.get(
                f"{stage_prefix}_objectives_or_teacher_rows_used_at_freeze"
            )
            is not False
            or manifest.get("outer_held_family_used") is not False
            or manifest.get("raw_provider_or_response_tensors_serialized")
            is not False
            or any(
                set(mapping) != set(families)
                for mapping in (
                    evidence_by_family,
                    provider_hashes,
                    runtime_hashes,
                    receipts,
                    traces,
                    transfers,
                )
            )
        ):
            raise ValueError(f"V20n {stage} freeze manifest differs")
        objectives: dict[str, float] = {}
        endpoint_anchors: dict[str, bool] = {}
        for family in families:
            family_evidence = _validate_hashed(
                _mapping(
                    evidence_by_family[family],
                    label=f"V20n {stage} family evidence",
                ),
                domain=_INNER_EXECUTION_DOMAIN,
                label=f"V20n {stage} family evidence",
            )
            fit = _mapping(
                inner_evidence[family].get("reflection_fit_receipt"),
                label=f"V20n {stage} reflection fit",
            )
            direction = _selected_direction(fit)
            expected_transfer = _shrinkage_provider_seed(
                endpoint_receipt_sha256=str(
                    endpoint_receipt["artifact_sha256"]
                ),
                direction=direction,
                direction_artifact_sha256=str(
                    fit["selected_variant_artifact_sha256"]
                ),
                reflection_fit_sha256=str(fit["artifact_sha256"]),
                response=response,
                lambda_=expected_lambda,
                outer_family_id=outer,
                inner_family_id=family,
                role=role,
            )
            provider_receipt = _validate_hashed(
                _mapping(
                    receipts[family], label=f"V20n {stage} provider receipt"
                ),
                domain=_PROVIDER_DOMAIN,
                label=f"V20n {stage} provider receipt",
            )
            _validate_shrinkage_provider_receipt_evidence(
                provider_receipt,
                expected_role=role,
                expected_provider_artifact_sha256=str(provider_hashes[family]),
                expected_endpoint_receipt=endpoint_receipt,
                expected_bridge_binding_sha256=bridge_binding_sha256,
                authenticated_v20i_fold=authenticated_v20i_fold,
                expected_response=response,
                expected_lambda=expected_lambda,
                expected_direction=direction,
                expected_transfer_evidence_sha256=expected_transfer,
            )
            exact = _validate_exact_score_bundle(
                family_evidence,
                expected_example_ids=tuple(
                    sorted(
                        _mapping(
                            _mapping(
                                inner_evidence[family].get("response_evidence"),
                                label="V20n V20m response evidence",
                            )[_response_key(response)].get(
                                "objectives_by_example"
                            ),
                            label="V20n V20m response objectives",
                        )
                    )
                ),
                label=f"V20n {stage} exact score",
            )
            (
                family_objectives,
                family_h4_hashes,
                family_logits_hashes,
                family_execution_hashes,
            ) = exact
            objective = math.fsum(family_objectives.values()) / len(
                family_objectives
            )
            trace = _validate_hashed(
                _mapping(
                    family_evidence.get("response_trace"),
                    label=f"V20n {stage} trace",
                ),
                domain=_TRACE_DOMAIN,
                label=f"V20n {stage} trace",
            )
            provider_artifact = _sha(
                provider_hashes[family],
                label=f"V20n {stage} provider artifact",
            )
            runtime_provider_artifact = _sha(
                runtime_hashes[family],
                label=f"V20n {stage} runtime provider artifact",
            )
            execution_seed = _v14._sha256(
                {
                    f"{stage_prefix}_manifest_sha256": manifest[
                        "artifact_sha256"
                    ],
                    "outer_held_family_id": outer,
                    "inner_held_family_id": family,
                    "provider_artifact_sha256": provider_artifact,
                    "runtime_provider_artifact_sha256": (
                        runtime_provider_artifact
                    ),
                    "lineage_wrapper_not_inference_executor": True,
                    f"all_{stage_prefix}_providers_frozen": True,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
            phase = (
                "inner_simplex_shrinkage_lambda_half_score"
                if stage == "lambda_half"
                else "inner_simplex_shrinkage_vertex_score"
            )
            expected_execution_hashes = {
                example: _execution_sha256(
                    phase=phase,
                    outer_family_id=outer,
                    inner_family_id=family,
                    role=role,
                    provider_artifact_sha256=runtime_provider_artifact,
                    example_id=example,
                    family_id=family,
                    objective=family_objectives[example],
                    h4_sha256=family_h4_hashes[example],
                    logits_sha256=family_logits_hashes[example],
                    evidence_sha256=execution_seed,
                    domain=_INNER_EXECUTION_DOMAIN,
                )
                for example in sorted(family_objectives)
            }
            response_gain_hashes = _mapping(
                trace.get("response_gain_sha256s"),
                label=f"V20n {stage} response gain hashes",
            )
            for value in response_gain_hashes.values():
                _sha(value, label=f"V20n {stage} response gain hash")
            if (
                family_evidence.get("stage") != stage
                or family_evidence.get("inner_held_family_id") != family
                or family_evidence.get("outer_held_family_id") != outer
                or float(family_evidence.get("lambda", math.nan))
                != expected_lambda
                or float(family_evidence.get("objective", math.nan))
                != objective
                or family_evidence.get("provider_artifact_sha256")
                != provider_hashes[family]
                or family_evidence.get("runtime_provider_artifact_sha256")
                != runtime_hashes[family]
                or family_evidence.get("lineage_wrapper_not_inference_executor")
                is not True
                or family_evidence.get("manifest_sha256")
                != manifest.get("artifact_sha256")
                or trace.get("artifact_sha256") != traces[family]
                or trace.get("provider_artifact_sha256")
                != provider_hashes[family]
                or trace.get("arm") != role
                or tuple(trace.get("scored_family_ids", ())) != (family,)
                or set(response_gain_hashes) != set(family_objectives)
                or provider_receipt.get("runtime_provider_artifact_sha256")
                != runtime_hashes[family]
                or transfers[family] != expected_transfer
                or family_execution_hashes != expected_execution_hashes
                or family_evidence.get(
                    f"all_{stage_prefix}_providers_frozen_before_score"
                )
                is not True
                or family_evidence.get("outer_family_absent_from_fit_and_score")
                is not True
                or family_evidence.get("exact_execution") is not True
                or family_evidence.get("finite") is not True
                or family_evidence.get(
                    "raw_logits_h4_teacher_rows_or_tensors_serialized"
                )
                is not False
                or trace.get("finite") is not True
                or trace.get("pointwise_trust_passed") is not True
                or trace.get("endpoint_conditional_ranks_are_16") is not True
                or trace.get("raw_response_or_modal_tensors_serialized")
                is not False
            ):
                raise ValueError(f"V20n {stage} evidence differs")
            _v20b._validate_capability_receipt(
                family_evidence.get("capability_receipt"),
                expected_example_ids=tuple(sorted(family_objectives)),
                expected_family_count=1,
                expected_held_family_id=outer,
                expected_accesses_per_example=1,
                label=f"V20n {stage} capability",
            )
            objectives[family] = objective
            if stage == "quadratic_vertex":
                endpoint_anchor = True
                if expected_lambda in (0.0, 1.0):
                    endpoint_response = (
                        linear_response if expected_lambda == 0.0 else response
                    )
                    inherited_endpoint = _mapping(
                        _mapping(
                            inner_evidence[family].get("response_evidence"),
                            label="V20n inherited V20m response evidence",
                        ).get(_response_key(endpoint_response)),
                        label="V20n inherited V20m endpoint response evidence",
                    )
                    endpoint_anchor = (
                        objective == float(inherited_endpoint["objective"])
                        and dict(family_objectives)
                        == dict(
                            _mapping(
                                inherited_endpoint.get("objectives_by_example"),
                                label="V20n inherited endpoint objectives",
                            )
                        )
                        and dict(family_h4_hashes)
                        == dict(
                            _mapping(
                                inherited_endpoint.get("post_cast_h4_sha256s"),
                                label="V20n inherited endpoint H4 hashes",
                            )
                        )
                        and dict(family_logits_hashes)
                        == dict(
                            _mapping(
                                inherited_endpoint.get(
                                    "supervised_full_vocab_logits_sha256s"
                                ),
                                label="V20n inherited endpoint logits hashes",
                            )
                        )
                    )
                if (
                    family_evidence.get(
                        "endpoint_vertex_exact_v20m_output_anchor"
                    )
                    is not endpoint_anchor
                ):
                    raise ValueError("V20n endpoint vertex anchor differs")
                endpoint_anchors[family] = endpoint_anchor
        return objectives, endpoint_anchors

    half_objectives, half_endpoint_anchors = validate_stage(
        stage="lambda_half",
        expected_lambda=0.5,
        manifest_key="half_provider_manifest",
        evidence_key="half_evidence_by_family",
        role="inner_simplex_shrinkage_lambda_half",
        freeze_flag=(
            "all_seven_half_providers_and_traces_frozen_before_any_half_capability"
        ),
    )
    if half_endpoint_anchors:
        raise ValueError("V20n lambda-half stage emitted endpoint anchors")
    persisted_anchors = _mapping(
        receipt.get("anchor_objectives_by_family_and_anchor"),
        label="V20n persisted anchor objectives",
    )
    expected_anchors = {}
    for family in families:
        scores = _mapping(
            inner_evidence[family].get("objective_by_response"),
            label="V20n V20m response objectives",
        )
        expected_anchors[family] = {
            "lambda_0": float(scores[_response_key(linear_response)]),
            "lambda_half": half_objectives[family],
            "lambda_1": float(scores[_response_key(response)]),
        }
    if _v14._canonical_json_bytes(persisted_anchors) != _v14._canonical_json_bytes(
        expected_anchors
    ):
        raise ValueError("V20n anchor objectives differ from provider evidence")
    anchor_receipt = _mapping(
        receipt.get("core_anchor_receipt"), label="V20n core anchor receipt"
    )
    _shrinkage_fit.validate_soft_polarity_simplex_shrinkage_anchor_receipt(
        anchor_receipt,
        all_development_family_ids=tuple(sorted((*families, outer))),
        outer_held_family_id=outer,
        exact_anchor_objectives_by_family_and_anchor=expected_anchors,
    )
    proposal_receipt = _mapping(
        receipt.get("core_quadratic_proposal_receipt"),
        label="V20n core proposal receipt",
    )
    _shrinkage_fit.validate_soft_polarity_simplex_shrinkage_quadratic_proposal_receipt(
        proposal_receipt, anchor_receipt=anchor_receipt
    )
    proposed_lambda = float(proposal_receipt["proposed_lambda"])
    vertex_manifest = _mapping(
        receipt.get("vertex_provider_manifest"),
        label="V20n quadratic_vertex manifest",
    )
    if (
        vertex_manifest.get("anchor_receipt_sha256")
        != anchor_receipt.get("artifact_sha256")
        or vertex_manifest.get("proposal_receipt_sha256")
        != proposal_receipt.get("artifact_sha256")
    ):
        raise ValueError("V20n vertex manifest fit lineage differs")
    vertex_objectives, endpoint_anchors = validate_stage(
        stage="quadratic_vertex",
        expected_lambda=proposed_lambda,
        manifest_key="vertex_provider_manifest",
        evidence_key="vertex_evidence_by_family",
        role="inner_simplex_shrinkage_vertex",
        freeze_flag=(
            "all_seven_vertex_providers_and_traces_frozen_before_any_vertex_capability"
        ),
    )
    persisted_endpoint_anchors = _mapping(
        receipt.get("endpoint_vertex_exact_v20m_output_anchor_by_family"),
        label="V20n persisted endpoint vertex anchors",
    )
    if (
        set(persisted_endpoint_anchors) != set(families)
        or _v14._canonical_json_bytes(persisted_endpoint_anchors)
        != _v14._canonical_json_bytes(endpoint_anchors)
    ):
        raise ValueError("V20n persisted endpoint vertex anchors differ")
    vertex_receipt = _mapping(
        receipt.get("core_vertex_score_receipt"),
        label="V20n core vertex score receipt",
    )
    _shrinkage_fit.validate_soft_polarity_simplex_shrinkage_vertex_score_receipt(
        vertex_receipt,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=vertex_objectives,
    )
    selection_receipt = _mapping(
        receipt.get("core_selection_receipt"),
        label="V20n core selection receipt",
    )
    _shrinkage_fit.validate_soft_polarity_simplex_shrinkage_selection_receipt(
        selection_receipt,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_receipt,
    )
    selected_lambda = float(selection_receipt["selected_lambda"])
    if (
        float(receipt.get("selected_lambda", math.nan)) != selected_lambda
        or receipt.get("selected_lambda_hex") != selected_lambda.hex()
        or receipt.get("selected_lambda_interior")
        is not (0.0 < selected_lambda < 1.0)
        or not endpoint_anchors
        or not all(endpoint_anchors.values())
        or receipt.get("all_endpoint_vertex_exact_v20m_output_anchors_passed")
        is not True
    ):
        raise ValueError("V20n selected shrinkage lambda differs")
    return receipt


def _validate_fold_fragment(
    value: Mapping[str, object],
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> None:
    fragment = _mapping(value, label="V20n fold fragment")
    if set(fragment) != _FOLD_FRAGMENT_KEYS:
        raise ValueError("V20n fold fragment key set differs")
    outer = _identifier(outer_family_id, label="V20n fold outer family")
    expected_header = {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": (
            _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
        ),
        "masked_direction_protocol_sha256": (
            _reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
        ),
        "simplex_response_fit_protocol_sha256": (
            _simplex_response_fit.SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
        ),
        "simplex_shrinkage_fit_protocol_sha256": (
            _shrinkage_fit.SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
        ),
        "simplex_shrinkage_provider_protocol_sha256": (
            FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256
        ),
        "exact_objective_kind": (
            "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
        ),
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20g_fold_fragment_sha256": authenticated_v20g_fold[
            "fragment_sha256"
        ],
        "v20i_fold_fragment_sha256": authenticated_v20i_fold[
            "fragment_sha256"
        ],
        "v20j_fold_fragment_sha256": source[
            "v20j_fold_fragment_sha256s_by_family"
        ][outer],
        "v20k_fold_fragment_sha256": source[
            "v20k_fold_fragment_sha256s_by_family"
        ][outer],
        "v20l_fold_fragment_sha256": source[
            "v20l_fold_fragment_sha256s_by_family"
        ][outer],
        "v20m_fold_fragment_sha256": authenticated_v20m_fold[
            "fragment_sha256"
        ],
        "outer_held_family_id": outer,
    }
    if any(fragment.get(key) != expected for key, expected in expected_header.items()):
        raise ValueError("V20n fold fragment header differs")
    if (
        fragment.get("fixed_schedule_completed") is not True
        or fragment.get("candidate") is not None
        or fragment.get("provider_sidecar") is not None
    ):
        raise ValueError("V20n fold scalar-only boundary differs")
    for key in ("endpoint_receipt", "endpoint_evidence"):
        if _v14._canonical_json_bytes(fragment.get(key)) != _v14._canonical_json_bytes(
            authenticated_v20g_fold.get(key)
        ):
            raise ValueError("V20n fold endpoint lineage differs")

    fit = _mapping(
        authenticated_v20g_fold.get("fit_receipt"),
        label="V20n validation V20g fit",
    )
    source_direction = _mapping(
        fit.get("direction_receipt"), label="V20n validation source direction"
    )
    inner = _validate_inner_receipt(
        _mapping(fragment.get("inner_receipt"), label="V20n inner receipt"),
        source_direction=source_direction,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20i_fold=authenticated_v20i_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
    )
    response_selection = _validate_hashed(
        _mapping(
            fragment.get("response_selection_receipt"),
            label="V20n response selection",
        ),
        domain=_RESPONSE_SELECTION_DOMAIN,
        label="V20n response selection",
    )
    if _v14._canonical_json_bytes(response_selection) != _v14._canonical_json_bytes(
        inner.get("response_selection_receipt")
    ):
        raise ValueError("V20n duplicated response selection differs")
    endpoint_receipt = _mapping(
        authenticated_v20g_fold.get("endpoint_receipt"),
        label="V20n outer inherited endpoint receipt",
    )
    shrinkage_selection = _validate_shrinkage_selection(
        _mapping(
            fragment.get("shrinkage_selection_receipt"),
            label="V20n shrinkage selection receipt",
        ),
        response_selection=response_selection,
        inner_receipt=inner,
        endpoint_receipt=endpoint_receipt,
        outer_family_id=outer,
        bridge_binding_sha256=bridge_binding_sha256,
        authenticated_v20i_fold=authenticated_v20i_fold,
    )

    reflection_fit = _mapping(
        fragment.get("outer_reflection_fit_receipt"),
        label="V20n outer reflection fit",
    )
    _reflection.validate_soft_polarity_reflection_fit_receipt(
        reflection_fit, direction_receipt=source_direction
    )
    _validate_v20i_reflection_lineage(
        inner_receipt=inner,
        outer_reflection_fit=reflection_fit,
        authenticated_v20i_fold=authenticated_v20i_fold,
    )
    manifest = _validate_hashed(
        _mapping(
            fragment.get("provider_manifest"), label="V20n outer manifest"
        ),
        domain=_OUTER_MANIFEST_DOMAIN,
        label="V20n outer manifest",
    )
    if (
        tuple(manifest.get("arm_order", ())) != _ARMS
        or manifest.get("outer_held_family_id") != outer
        or _response_tuple(manifest.get("selected_response"))
        != _response_tuple(response_selection["selected_response"])
        or float(manifest.get("selected_lambda", math.nan))
        != float(shrinkage_selection["selected_lambda"])
        or manifest.get("selected_lambda_hex")
        != float(shrinkage_selection["selected_lambda"]).hex()
        or _response_tuple(manifest.get("matched_linear_response"))
        != (
            _response_tuple(response_selection["selected_response"])[0],
            0.0,
            0.0,
        )
        or _simplex_parameters(manifest.get("matched_v20l_boundary_response"))
        != _matched_v20l_boundary_response(
            authenticated_v20l_fold, outer_family_id=outer
        )
        or manifest.get("matched_v20l_boundary_source_fold_sha256")
        != authenticated_v20l_fold.get("fragment_sha256")
        or manifest.get("matched_v20m_source_fold_sha256")
        != authenticated_v20m_fold.get("fragment_sha256")
        or manifest.get("endpoint_receipt_sha256")
        != endpoint_receipt.get("artifact_sha256")
        or manifest.get("source_direction_receipt_sha256")
        != source_direction.get("artifact_sha256")
        or manifest.get("outer_reflection_fit_receipt_sha256")
        != reflection_fit.get("artifact_sha256")
        or manifest.get("selected_variant_artifact_sha256")
        != reflection_fit.get("selected_variant_artifact_sha256")
        or manifest.get("all_nine_providers_frozen_before_outer_capability")
        is not True
        or manifest.get("all_nine_traces_frozen_before_outer_capability") is not True
        or manifest.get("outer_capability_count_at_freeze") != 0
        or manifest.get("outer_objectives_or_teacher_rows_used_at_freeze")
        is not False
        or manifest.get("raw_provider_or_response_tensors_serialized") is not False
    ):
        raise ValueError("V20n outer provider manifest differs")
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s"),
        label="V20n outer provider hashes",
    )
    provider_receipts = _mapping(
        manifest.get("provider_receipts"),
        label="V20n outer provider receipts",
    )
    runtime_provider_hashes = _mapping(
        manifest.get("runtime_provider_artifact_sha256s"),
        label="V20n outer runtime provider hashes",
    )
    trace_hashes = _mapping(
        manifest.get("response_trace_sha256s"),
        label="V20n outer trace hashes",
    )
    soft_transfer_hashes = _mapping(
        manifest.get("soft_provider_transfer_evidence_sha256s"),
        label="V20n outer soft provider transfer hashes",
    )
    expected_soft_transfer_arms = (
        "matched_linear_reflected",
        "matched_v20l_boundary_reflected",
        "same_simplex_response_unreflected",
        "simplex_shrinkage_reflected",
        "simplex_response_reflected_exact_mirror",
        "matched_v20m_simplex_reflected",
    )
    if set(soft_transfer_hashes) != set(expected_soft_transfer_arms):
        raise ValueError("V20n outer soft transfer arm bindings differ")
    fixed_control_transfer_hash = _sha(
        manifest.get("fixed_control_transfer_evidence_sha256"),
        label="V20n outer fixed-control transfer hash",
    )
    if any(
        set(values) != set(_ARMS)
        for values in (
            provider_hashes,
            runtime_provider_hashes,
            provider_receipts,
            trace_hashes,
        )
    ):
        raise ValueError("V20n outer manifest arm bindings differ")
    inherited_arms = _mapping(
        _mapping(
            authenticated_v20g_fold.get("held_evidence"),
            label="V20n inherited V20g held evidence",
        ).get("arm_evidence"),
        label="V20n inherited V20g held arms",
    )
    if not all(arm in inherited_arms for arm in ("base", "fixed_plus", "fixed_minus")):
        raise ValueError("V20n inherited V20g control arms differ")
    expected_examples = tuple(
        sorted(
            _identifier(example, label="V20n outer expected example")
            for example in _mapping(
                _mapping(
                    inherited_arms["base"],
                    label="V20n inherited V20g base arm",
                ).get("objectives_by_example"),
                label="V20n inherited V20g base objectives",
            )
        )
    )
    held = _validate_hashed(
        _mapping(fragment.get("held_evidence"), label="V20n held evidence"),
        domain=_OUTER_EXECUTION_DOMAIN,
        label="V20n held evidence",
    )
    arms = _mapping(held.get("arm_evidence"), label="V20n held arm evidence")
    if (
        set(arms) != set(_ARMS)
        or held.get("outer_held_family_id") != outer
        or held.get("outer_manifest_sha256") != manifest.get("artifact_sha256")
        or held.get("all_nine_providers_and_traces_frozen_before_outer_capability")
        is not True
        or held.get("outer_family_used_for_fit_or_selection") is not False
        or held.get("all_v20g_control_output_anchors_passed") is not True
        or held.get("matched_v20l_boundary_exact_output_anchor_passed") is not True
        or held.get("matched_v20m_exact_output_anchor_passed") is not True
        or int(held.get("exact_outer_execution_count", -1))
        != len(_ARMS) * _PROMPTS_PER_FAMILY
        or held.get("raw_prompts_tokens_logits_h4_or_teacher_rows_serialized")
        is not False
    ):
        raise ValueError("V20n outer held schedule differs")
    objectives: dict[str, float] = {}
    exact_outputs: dict[
        str, tuple[dict[str, float], dict[str, str], dict[str, str]]
    ] = {}
    trace_bundle_sha = _v14._sha256(
        {arm: trace_hashes[arm] for arm in _ARMS},
        domain=_OUTER_EXECUTION_DOMAIN,
    )
    selected_response = _response_tuple(
        response_selection["selected_response"]
    )
    selected_lambda = float(shrinkage_selection["selected_lambda"])
    matched_linear_response = (selected_response[0], 0.0, 0.0)
    matched_v20l_boundary_response = _matched_v20l_boundary_response(
        authenticated_v20l_fold, outer_family_id=outer
    )
    selected_direction = _selected_direction(reflection_fit)
    unreflected_direction = _unreflected_direction(source_direction)
    mirror_direction = tuple(-item for item in selected_direction)
    expected_soft_transfers = {
        "matched_linear_reflected": _provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=matched_linear_response,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_matched_linear_reflected",
        ),
        "matched_v20l_boundary_reflected": _provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=matched_v20l_boundary_response,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_matched_v20l_boundary_reflected",
        ),
        "same_simplex_response_unreflected": _shrinkage_provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            lambda_=selected_lambda,
            direction_artifact_sha256=str(source_direction["artifact_sha256"]),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            direction=unreflected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_same_simplex_response_unreflected",
        ),
        "simplex_shrinkage_reflected": _shrinkage_provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            lambda_=selected_lambda,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_simplex_shrinkage_reflected",
        ),
        "simplex_response_reflected_exact_mirror": _shrinkage_provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            lambda_=selected_lambda,
            direction=mirror_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_simplex_response_reflected_exact_mirror",
        ),
        "matched_v20m_simplex_reflected": _provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_matched_v20m_simplex_reflected",
        ),
    }
    expected_soft_directions = {
        "matched_linear_reflected": selected_direction,
        "matched_v20l_boundary_reflected": selected_direction,
        "same_simplex_response_unreflected": unreflected_direction,
        "simplex_shrinkage_reflected": selected_direction,
        "simplex_response_reflected_exact_mirror": mirror_direction,
        "matched_v20m_simplex_reflected": selected_direction,
    }
    expected_soft_responses = {
        "matched_linear_reflected": matched_linear_response,
        "matched_v20l_boundary_reflected": matched_v20l_boundary_response,
        "same_simplex_response_unreflected": selected_response,
        "simplex_shrinkage_reflected": selected_response,
        "simplex_response_reflected_exact_mirror": selected_response,
        "matched_v20m_simplex_reflected": selected_response,
    }
    shrinkage_arms = {
        "same_simplex_response_unreflected",
        "simplex_shrinkage_reflected",
        "simplex_response_reflected_exact_mirror",
    }
    expected_fixed_control_transfer = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": manifest["endpoint_receipt_sha256"],
            "outer_held_family_id": outer,
            "reflection_fit_sha256": reflection_fit["artifact_sha256"],
            "selected_response": selected_response,
            "role": "outer_fixed_controls",
            "held_rows_used": False,
        },
        domain=_OUTER_MANIFEST_DOMAIN,
    )
    if (
        {
            arm: _sha(
                soft_transfer_hashes[arm],
                label=f"V20n {arm} soft transfer hash",
            )
            for arm in expected_soft_transfer_arms
        }
        != expected_soft_transfers
        or fixed_control_transfer_hash != expected_fixed_control_transfer
    ):
        raise ValueError("V20n outer provider transfer lineage differs")
    health_by_arm: dict[str, bool] = {}
    for arm in _ARMS:
        evidence = _validate_hashed(
            _mapping(arms[arm], label=f"V20n {arm} arm evidence"),
            domain=_OUTER_EXECUTION_DOMAIN,
            label=f"V20n {arm} arm evidence",
        )
        trace = _validate_hashed(
            _mapping(evidence.get("response_trace"), label=f"V20n {arm} trace"),
            domain=_TRACE_DOMAIN,
            label=f"V20n {arm} trace",
        )
        provider_receipt = _validate_hashed(
            _mapping(
                provider_receipts[arm],
                label=f"V20n {arm} provider receipt",
            ),
            domain=_PROVIDER_DOMAIN,
            label=f"V20n {arm} provider receipt",
        )
        score_bundle = _validate_exact_score_bundle(
            evidence,
            expected_example_ids=expected_examples,
            label=f"V20n outer {arm}",
        )
        arm_objectives, arm_h4, arm_logits, arm_executions = score_bundle
        provider_artifact = _sha(
            provider_hashes[arm], label=f"V20n {arm} provider hash"
        )
        if arm in shrinkage_arms:
            _validate_shrinkage_provider_receipt_evidence(
                provider_receipt,
                expected_role=arm,
                expected_provider_artifact_sha256=provider_artifact,
                expected_endpoint_receipt=endpoint_receipt,
                expected_bridge_binding_sha256=bridge_binding_sha256,
                authenticated_v20i_fold=authenticated_v20i_fold,
                expected_response=expected_soft_responses[arm],
                expected_lambda=selected_lambda,
                expected_direction=expected_soft_directions[arm],
                expected_transfer_evidence_sha256=expected_soft_transfers[arm],
            )
        else:
            _validate_provider_receipt_evidence(
                provider_receipt,
                expected_role=arm,
                expected_provider_artifact_sha256=provider_artifact,
                expected_endpoint_receipt=endpoint_receipt,
                expected_bridge_binding_sha256=bridge_binding_sha256,
                authenticated_v20i_fold=authenticated_v20i_fold,
                expected_response=(
                    expected_soft_responses[arm]
                    if arm in expected_soft_transfer_arms
                    else None
                ),
                expected_direction=(
                    expected_soft_directions[arm]
                    if arm in expected_soft_transfer_arms
                    else None
                ),
                expected_transfer_evidence_sha256=(
                    expected_soft_transfers[arm]
                    if arm in expected_soft_transfer_arms
                    else (
                        expected_fixed_control_transfer
                        if arm in ("fixed_plus", "fixed_minus")
                        else None
                    )
                ),
            )
        trace_artifact = _sha(
            trace_hashes[arm], label=f"V20n {arm} trace hash"
        )
        seed = _v14._sha256(
            {
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle_sha,
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": provider_artifact,
                "runtime_provider_artifact_sha256": runtime_provider_hashes[
                    arm
                ],
                "lineage_wrapper_not_inference_executor": arm
                in shrinkage_arms,
                "all_outer_arms_frozen": True,
            },
            domain=_OUTER_EXECUTION_DOMAIN,
        )
        expected_executions = {
            example: _execution_sha256(
                phase="outer_family_disjoint_mechanism_score",
                outer_family_id=outer,
                inner_family_id=None,
                role=arm,
                provider_artifact_sha256=str(runtime_provider_hashes[arm]),
                example_id=example,
                family_id=outer,
                objective=arm_objectives[example],
                h4_sha256=arm_h4[example],
                logits_sha256=arm_logits[example],
                evidence_sha256=seed,
                domain=_OUTER_EXECUTION_DOMAIN,
            )
            for example in expected_examples
        }
        response_gain_hashes = _mapping(
            trace.get("response_gain_sha256s"),
            label=f"V20n {arm} response gain hashes",
        )
        for value in response_gain_hashes.values():
            _sha(value, label=f"V20n {arm} response gain hash")
        health_by_arm[arm] = bool(
            evidence.get("finite") is True
            and trace.get("finite") is True
            and trace.get("pointwise_trust_passed") is True
            and trace.get("endpoint_conditional_ranks_are_16") is True
        )
        if (
            evidence.get("arm") != arm
            or evidence.get("provider_artifact_sha256") != provider_artifact
            or evidence.get("runtime_provider_artifact_sha256")
            != runtime_provider_hashes[arm]
            or evidence.get("lineage_wrapper_not_inference_executor")
            is not (arm in shrinkage_arms)
            or evidence.get("outer_manifest_sha256")
            != manifest.get("artifact_sha256")
            or trace.get("artifact_sha256") != trace_artifact
            or trace.get("provider_artifact_sha256") != provider_artifact
            or trace.get("arm") != arm
            or tuple(trace.get("scored_family_ids", ())) != (outer,)
            or set(response_gain_hashes) != set(expected_examples)
            or provider_receipt.get("provider_artifact_sha256")
            != provider_artifact
            or provider_receipt.get("role") != arm
            or provider_receipt.get("raw_provider_tensors_serialized")
            is not False
            or int(provider_receipt.get("conditional_rank", -1))
            != _CONDITIONAL_RANK
            or provider_receipt.get("analysis_only") is not (arm != "base")
            or (
                arm in expected_soft_transfer_arms
                and arm not in shrinkage_arms
                and (
                    provider_receipt.get("transfer_protocol_sha256")
                    != _TRANSFER_PROTOCOL_SHA256
                    or provider_receipt.get("transfer_evidence_sha256")
                    != expected_soft_transfers[arm]
                    or tuple(provider_receipt.get("direction", ()))
                    != expected_soft_directions[arm]
                    or _simplex_parameters(provider_receipt.get("response"))
                    != expected_soft_responses[arm]
                    or provider_receipt.get("response_key")
                    != _parameters_key(expected_soft_responses[arm])
                    or float(
                        provider_receipt.get("radius", math.nan)
                    )
                    != expected_soft_responses[arm][0]
                    or float(
                        provider_receipt.get("u", math.nan)
                    )
                    != expected_soft_responses[arm][1]
                    or float(provider_receipt.get("v", math.nan))
                    != expected_soft_responses[arm][2]
                    or tuple(
                        provider_receipt.get("direction_box_corner_scores", ())
                    )
                    != _box_corner_scores(expected_soft_directions[arm])
                    or _v14._canonical_json_bytes(
                        _mapping(
                            provider_receipt.get("box_certificate"),
                            label=f"V20n {arm} simplex_response certificate",
                        )
                    )
                    != _v14._canonical_json_bytes(
                        fisher_soft_polarity_simplex_response_box_certificate(
                            _v20g._eta_tensor(expected_soft_directions[arm]),
                            radius=expected_soft_responses[arm][0],
                            shrink_mass=expected_soft_responses[arm][1],
                            polarity_bias=expected_soft_responses[arm][2],
                        )
                    )
                )
            )
            or (
                arm in ("fixed_plus", "fixed_minus")
                and (
                    provider_receipt.get("transfer_protocol_sha256")
                    != _TRANSFER_PROTOCOL_SHA256
                    or provider_receipt.get("transfer_evidence_sha256")
                    != expected_fixed_control_transfer
                )
            )
            or arm_executions != expected_executions
            or evidence.get("exact_execution") is not True
            or evidence.get("raw_logits_h4_teacher_rows_or_tensors_serialized")
            is not False
            or trace.get("raw_response_or_modal_tensors_serialized") is not False
            or not health_by_arm[arm]
        ):
            raise ValueError("V20n outer arm health differs")
        objectives[arm] = float(evidence["objective"])
        exact_outputs[arm] = (arm_objectives, arm_h4, arm_logits)
    _v20b._validate_capability_receipt(
        held.get("capability_receipt"),
        expected_example_ids=expected_examples,
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ARMS),
        label="V20n resumed outer-held capability",
    )
    control_anchors: dict[str, bool] = {}
    for arm in ("base", "fixed_plus", "fixed_minus"):
        inherited = _mapping(
            inherited_arms[arm], label=f"V20n inherited V20g {arm} arm"
        )
        current_objectives, current_h4, current_logits = exact_outputs[arm]
        control_anchors[arm] = bool(
            objectives[arm] == float(inherited.get("objective", math.nan))
            and current_objectives
            == {
                _identifier(example, label=f"V20n inherited {arm} example"): float(
                    objective
                )
                for example, objective in _mapping(
                    inherited.get("objectives_by_example"),
                    label=f"V20n inherited {arm} objectives",
                ).items()
            }
            and current_h4
            == dict(
                _mapping(
                    inherited.get("post_cast_h4_sha256s"),
                    label=f"V20n inherited {arm} H4 hashes",
                )
            )
            and current_logits
            == dict(
                _mapping(
                    inherited.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20n inherited {arm} logits hashes",
                )
            )
        )
    persisted_control_anchors = _mapping(
        held.get("v20g_control_output_anchors"),
        label="V20n persisted control anchors",
    )
    if (
        not all(control_anchors.values())
        or dict(persisted_control_anchors) != control_anchors
        or held.get("all_v20g_control_output_anchors_passed")
        is not all(control_anchors.values())
    ):
        raise ValueError("V20n outer V20g control output anchor differs")
    inherited_v20l_arm = _mapping(
        _mapping(
            _mapping(
                authenticated_v20l_fold.get("held_evidence"),
                label="V20n resumed V20l held evidence",
            ).get("arm_evidence"),
            label="V20n resumed V20l held arms",
        ).get("signed_stack_reflected"),
        label="V20n resumed V20l selected boundary arm",
    )
    boundary_objectives, boundary_h4, boundary_logits = exact_outputs[
        "matched_v20l_boundary_reflected"
    ]
    v20l_boundary_anchor = bool(
        objectives["matched_v20l_boundary_reflected"]
        == float(inherited_v20l_arm.get("objective", math.nan))
        and boundary_objectives
        == {
            _identifier(example, label="V20n inherited V20l boundary example"): float(
                objective
            )
            for example, objective in _mapping(
                inherited_v20l_arm.get("objectives_by_example"),
                label="V20n inherited V20l boundary objectives",
            ).items()
        }
        and boundary_h4
        == dict(
            _mapping(
                inherited_v20l_arm.get("post_cast_h4_sha256s"),
                label="V20n inherited V20l boundary H4 hashes",
            )
        )
        and boundary_logits
        == dict(
            _mapping(
                inherited_v20l_arm.get("supervised_full_vocab_logits_sha256s"),
                label="V20n inherited V20l boundary logits hashes",
            )
        )
    )
    if (
        not v20l_boundary_anchor
        or held.get("matched_v20l_boundary_exact_output_anchor_passed") is not True
    ):
        raise ValueError("V20n exact V20l boundary output anchor differs")
    inherited_v20m_arm = _mapping(
        _mapping(
            _mapping(
                authenticated_v20m_fold.get("held_evidence"),
                label="V20n resumed V20m held evidence",
            ).get("arm_evidence"),
            label="V20n resumed V20m held arms",
        ).get("simplex_response_reflected"),
        label="V20n resumed V20m selected simplex arm",
    )
    v20m_objectives, v20m_h4, v20m_logits = exact_outputs[
        "matched_v20m_simplex_reflected"
    ]
    v20m_anchor = bool(
        objectives["matched_v20m_simplex_reflected"]
        == float(inherited_v20m_arm.get("objective", math.nan))
        and v20m_objectives
        == {
            _identifier(example, label="V20n inherited V20m example"): float(
                objective
            )
            for example, objective in _mapping(
                inherited_v20m_arm.get("objectives_by_example"),
                label="V20n inherited V20m objectives",
            ).items()
        }
        and v20m_h4
        == dict(
            _mapping(
                inherited_v20m_arm.get("post_cast_h4_sha256s"),
                label="V20n inherited V20m H4 hashes",
            )
        )
        and v20m_logits
        == dict(
            _mapping(
                inherited_v20m_arm.get(
                    "supervised_full_vocab_logits_sha256s"
                ),
                label="V20n inherited V20m logits hashes",
            )
        )
    )
    if (
        not v20m_anchor
        or held.get("matched_v20m_exact_output_anchor_passed") is not True
    ):
        raise ValueError("V20n exact V20m output anchor differs")
    base_logits = exact_outputs["base"][2]
    candidate_logits = exact_outputs[_PRIMARY_ARM][2]
    candidate_changed = any(
        candidate_logits[example] != base_logits[example]
        for example in expected_examples
    )
    boundary_logits = exact_outputs["matched_v20l_boundary_reflected"][2]
    candidate_changed_from_v20l_boundary = any(
        candidate_logits[example] != boundary_logits[example]
        for example in expected_examples
    )
    candidate_changed_from_v20m = any(
        candidate_logits[example] != v20m_logits[example]
        for example in expected_examples
    )
    linear_logits = exact_outputs["matched_linear_reflected"][2]
    candidate_changed_from_linear = any(
        candidate_logits[example] != linear_logits[example]
        for example in expected_examples
    )
    selected_lambda_interior = 0.0 < selected_lambda < 1.0
    interior_exact_distinct = (
        not selected_lambda_interior
        or (candidate_changed_from_v20m and candidate_changed_from_linear)
    )
    candidate_distinct = (
        provider_hashes[_PRIMARY_ARM] != provider_hashes["base"]
    )
    all_healthy = all(health_by_arm.values())
    fold = _validate_hashed(
        _mapping(fragment.get("fold_receipt"), label="V20n fold receipt"),
        domain=_DECISION_DOMAIN,
        label="V20n fold receipt",
    )
    if (
        fold.get("outer_held_family_id") != outer
        or tuple(fold.get("arm_order", ())) != _ARMS
        or dict(fold.get("held_objective_by_arm", {})) != objectives
        or _response_tuple(fold.get("selected_response"))
        != selected_response
        or fold.get("selected_response_key")
        != _response_key(selected_response)
        or float(fold.get("selected_lambda", math.nan)) != selected_lambda
        or fold.get("selected_lambda_hex") != selected_lambda.hex()
        or fold.get("selected_lambda_interior") is not selected_lambda_interior
        or fold.get("selected_variant_artifact_sha256")
        != reflection_fit.get("selected_variant_artifact_sha256")
        or fold.get("selected_variant_id")
        != reflection_fit.get("selected_variant_id")
        or fold.get("candidate_provider_artifact_sha256")
        != provider_hashes[_PRIMARY_ARM]
        or fold.get("base_provider_artifact_sha256")
        != provider_hashes["base"]
        or fold.get("candidate_provider_distinct_from_base")
        is not candidate_distinct
        or fold.get("candidate_exact_execution_changed_from_base")
        is not candidate_changed
        or fold.get(
            "candidate_exact_execution_changed_from_matched_v20l_boundary"
        )
        is not candidate_changed_from_v20l_boundary
        or fold.get("candidate_exact_execution_changed_from_matched_v20m")
        is not candidate_changed_from_v20m
        or fold.get("candidate_exact_execution_changed_from_matched_linear")
        is not candidate_changed_from_linear
        or fold.get("interior_candidate_exact_distinct_from_both_endpoints")
        is not interior_exact_distinct
        or fold.get("selected_radius_positive")
        is not (selected_response[0] > 0.0)
        or fold.get("selected_u_positive")
        is not (selected_response[1] > 0.0)
        or fold.get("selected_interior_simplex_response")
        is not (0.0 < abs(selected_response[2]) < selected_response[1])
        or fold.get("selected_boundary_simplex_response")
        is not (
            selected_response[1] > 0.0
            and abs(selected_response[2]) == selected_response[1]
        )
        or fold.get("selected_zero_bias_simplex_response")
        is not (
            selected_response[1] > 0.0 and selected_response[2] == 0.0
        )
        or fold.get("all_runtime_health_passed") is not all_healthy
        or fold.get("all_v20g_control_output_anchors_passed")
        is not all(control_anchors.values())
        or fold.get("matched_v20l_boundary_exact_output_anchor_passed")
        is not v20l_boundary_anchor
        or fold.get("matched_v20m_exact_output_anchor_passed")
        is not v20m_anchor
        or fold.get("selection_frozen_before_outer_score") is not True
        or fold.get("outer_family_used_for_fit_or_selection") is not False
        or fold.get("exact_execution") is not True
    ):
        raise ValueError("V20n fold decision receipt differs")


def _publish_fold_fragment(
    payload: Mapping[str, object], *, output: Path | str, outer_family_id: str
) -> dict[str, object]:
    path = _fold_path(output, outer_family_id)
    _v20b._publish_scalar_fragment(
        payload,
        path=path,
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20n simplex-shrinkage fold fragment",
    )
    return _v20b._load_scalar_fragment(
        path=path,
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20n simplex-shrinkage fold fragment",
    )


def _load_fold_fragment(
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> dict[str, object]:
    fragment = _v20b._load_scalar_fragment(
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20n simplex-shrinkage fold fragment",
    )
    _validate_fold_fragment(
        fragment,
        output=output,
        source=source,
        panel_receipt=panel_receipt,
        outer_family_id=outer_family_id,
        bridge_binding_sha256=bridge_binding_sha256,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20i_fold=authenticated_v20i_fold,
        authenticated_v20l_fold=authenticated_v20l_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
    )
    return fragment


def _fold_receipt_map(
    fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    return {
        family: _mapping(
            fragments[family].get("fold_receipt"), label="V20n aggregate fold"
        )
        for family in sorted(fragments)
    }


def _aggregate_decision(
    fold_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    families = tuple(sorted(fold_receipts))
    if len(families) != _FAMILY_COUNT:
        raise ValueError("V20n decision requires all eight outer folds")
    scores: dict[str, dict[str, float]] = {}
    responses: dict[str, tuple[float, float, float]] = {}
    lambdas: dict[str, float] = {}
    variants: dict[str, str] = {}
    changed: dict[str, bool] = {}
    changed_from_v20l_boundary: dict[str, bool] = {}
    changed_from_v20m: dict[str, bool] = {}
    changed_from_linear: dict[str, bool] = {}
    interior_exact_distinct: dict[str, bool] = {}
    health: dict[str, bool] = {}
    v20g_anchors: dict[str, bool] = {}
    v20l_boundary_anchors: dict[str, bool] = {}
    v20m_anchors: dict[str, bool] = {}
    for family in families:
        fold = fold_receipts[family]
        if tuple(fold.get("arm_order", ())) != _ARMS:
            raise ValueError("V20n aggregate arm order differs")
        raw_scores = _mapping(
            fold.get("held_objective_by_arm"), label="V20n aggregate arm scores"
        )
        if set(raw_scores) != set(_ARMS):
            raise ValueError("V20n aggregate arm geometry differs")
        scores[family] = {arm: float(raw_scores[arm]) for arm in _ARMS}
        if not all(math.isfinite(value) for value in scores[family].values()):
            raise ValueError("V20n aggregate score became nonfinite")
        responses[family] = _response_tuple(fold["selected_response"])
        lambdas[family] = float(fold["selected_lambda"])
        if not math.isfinite(lambdas[family]) or not 0.0 <= lambdas[family] <= 1.0:
            raise ValueError("V20n aggregate selected lambda differs")
        variants[family] = str(fold["selected_variant_id"])
        changed[family] = (
            fold.get("candidate_provider_distinct_from_base") is True
            and fold.get("candidate_exact_execution_changed_from_base") is True
        )
        changed_from_v20l_boundary[family] = fold.get(
            "candidate_exact_execution_changed_from_matched_v20l_boundary"
        ) is True
        changed_from_v20m[family] = fold.get(
            "candidate_exact_execution_changed_from_matched_v20m"
        ) is True
        changed_from_linear[family] = fold.get(
            "candidate_exact_execution_changed_from_matched_linear"
        ) is True
        interior_exact_distinct[family] = fold.get(
            "interior_candidate_exact_distinct_from_both_endpoints"
        ) is True
        health[family] = fold.get("all_runtime_health_passed") is True
        v20g_anchors[family] = (
            fold.get("all_v20g_control_output_anchors_passed") is True
        )
        v20l_boundary_anchors[family] = (
            fold.get("matched_v20l_boundary_exact_output_anchor_passed") is True
        )
        v20m_anchors[family] = (
            fold.get("matched_v20m_exact_output_anchor_passed") is True
        )
    macro = {
        arm: math.fsum(scores[family][arm] for family in families) / len(families)
        for arm in _ARMS
    }

    def wins(reference: str) -> int:
        return sum(
            scores[family][_PRIMARY_ARM] < scores[family][reference]
            for family in families
        )

    wins_vs = {
        arm: wins(arm)
        for arm in (
            "base",
            "fixed_plus",
            "fixed_minus",
            "matched_linear_reflected",
            "matched_v20l_boundary_reflected",
            "same_simplex_response_unreflected",
            "simplex_response_reflected_exact_mirror",
            "matched_v20m_simplex_reflected",
        )
    }
    positive_changed = all(
        responses[family][0] > 0.0 and changed[family]
        for family in families
    )
    nontrivial_count = sum(
        responses[family][1] > 0.0 for family in families
    )
    interior = {
        family: 0.0 < abs(responses[family][2]) < responses[family][1]
        for family in families
    }
    boundary = {
        family: (
            responses[family][1] > 0.0
            and abs(responses[family][2]) == responses[family][1]
        )
        for family in families
    }
    zero_bias = {
        family: responses[family][1] > 0.0 and responses[family][2] == 0.0
        for family in families
    }
    interior_distinguishable = all(
        not interior[family] or changed_from_v20l_boundary[family]
        for family in families
    )
    simplex_response_evidence = nontrivial_count >= 5
    interior_lambda = {
        family: 0.0 < lambdas[family] < 1.0 for family in families
    }
    interior_lambda_count = sum(interior_lambda.values())
    continuous_shrinkage_evidence = interior_lambda_count >= 5
    all_interior_exact_distinct = all(interior_exact_distinct.values())
    integrity = (
        all(health.values())
        and all(v20g_anchors.values())
        and all(v20l_boundary_anchors.values())
        and all(v20m_anchors.values())
        and all_interior_exact_distinct
    )
    primary_gate = (
        integrity
        and positive_changed
        and macro[_PRIMARY_ARM] < macro["base"]
        and macro[_PRIMARY_ARM] < macro["fixed_plus"]
        and wins_vs["base"] >= 6
        and wins_vs["fixed_plus"] >= 6
    )
    mechanism_gate = (
        integrity
        and positive_changed
        and simplex_response_evidence
        and continuous_shrinkage_evidence
        and interior_distinguishable
        and macro[_PRIMARY_ARM] < macro["same_simplex_response_unreflected"]
        and wins_vs["same_simplex_response_unreflected"] >= 5
        and macro[_PRIMARY_ARM] < macro["simplex_response_reflected_exact_mirror"]
        and wins_vs["simplex_response_reflected_exact_mirror"] >= 6
        and macro[_PRIMARY_ARM] < macro["matched_linear_reflected"]
        and wins_vs["matched_linear_reflected"] >= 5
        and macro[_PRIMARY_ARM] < macro["matched_v20l_boundary_reflected"]
        and wins_vs["matched_v20l_boundary_reflected"] >= 5
        and macro[_PRIMARY_ARM] < macro["matched_v20m_simplex_reflected"]
        and wins_vs["matched_v20m_simplex_reflected"] >= 5
    )
    passed = primary_gate and mechanism_gate
    return _hashed(
        {
            "family_ids": families,
            "selected_response_by_family": responses,
            "selected_lambda_by_family": lambdas,
            "selected_variant_id_by_family": variants,
            "held_objective_by_family_and_arm": scores,
            "macro_objective_by_arm": macro,
            "candidate_win_count_by_reference_arm": wins_vs,
            "candidate_changed_exact_by_family": changed,
            "candidate_changed_exact_from_matched_v20l_boundary_by_family": (
                changed_from_v20l_boundary
            ),
            "candidate_changed_exact_from_matched_v20m_by_family": (
                changed_from_v20m
            ),
            "candidate_changed_exact_from_matched_linear_by_family": (
                changed_from_linear
            ),
            "interior_candidate_exact_distinct_from_both_endpoints_by_family": (
                interior_exact_distinct
            ),
            "runtime_health_by_family": health,
            "v20g_control_output_anchor_by_family": v20g_anchors,
            "matched_v20l_boundary_output_anchor_by_family": (
                v20l_boundary_anchors
            ),
            "matched_v20m_output_anchor_by_family": v20m_anchors,
            "all_selected_radii_positive_and_candidates_changed_exact": (
                positive_changed
            ),
            "selected_nontrivial_simplex_response_by_family": {
                family: responses[family][1] > 0.0 for family in families
            },
            "selected_nontrivial_simplex_response_count": nontrivial_count,
            "selected_interior_simplex_response_by_family": interior,
            "selected_interior_simplex_response_count": sum(interior.values()),
            "selected_boundary_simplex_response_by_family": boundary,
            "selected_boundary_simplex_response_count": sum(boundary.values()),
            "selected_zero_bias_simplex_response_by_family": zero_bias,
            "selected_zero_bias_simplex_response_count": sum(zero_bias.values()),
            "all_selected_interior_responses_distinguishable_from_matched_"
            "v20l_boundary": interior_distinguishable,
            "simplex_response_evidence_gate_passed": simplex_response_evidence,
            "selected_interior_lambda_by_family": interior_lambda,
            "selected_interior_lambda_count": interior_lambda_count,
            "continuous_shrinkage_evidence_gate_passed": (
                continuous_shrinkage_evidence
            ),
            "all_interior_candidates_exact_distinct_from_both_endpoints": (
                all_interior_exact_distinct
            ),
            "integrity_passed": integrity,
            "primary_development_gate_passed": primary_gate,
            "mechanism_gate_passed": mechanism_gate,
            "development_oof_passed": passed,
            "fixed_minus_diagnostic_only": True,
            "strict_win_comparison": True,
        },
        domain=_DECISION_DOMAIN,
    )


def _runner_work_accounting() -> dict[str, object]:
    """Return the fixed canonical V20n one-shot schedule.

    Authentication and resume attempts are deliberately excluded.  V20n
    consumes serialized V20g Fisher/gradient summaries for its 56 masked
    solves, but live authority collection and endpoint reconstruction retain
    the same V20i backward/contraction accounting.
    """

    authority_forwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY * 2
    endpoint_forwards = (
        _FAMILY_COUNT * (_FAMILY_COUNT - 1) * _PROMPTS_PER_FAMILY
    )
    inner_response_forwards = (
        _FAMILY_COUNT
        * _INNER_FAMILY_COUNT
        * _PROMPTS_PER_FAMILY
        * len(_RESPONSES)
    )
    inner_half_forwards = (
        _FAMILY_COUNT * _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY
    )
    inner_vertex_forwards = inner_half_forwards
    inner_forwards = (
        inner_response_forwards + inner_half_forwards + inner_vertex_forwards
    )
    outer_forwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY * len(_ARMS)
    inner_response_providers = (
        _FAMILY_COUNT * _INNER_FAMILY_COUNT * len(_RESPONSES)
    )
    inner_half_providers = _FAMILY_COUNT * _INNER_FAMILY_COUNT
    inner_vertex_providers = inner_half_providers
    inner_providers = (
        inner_response_providers + inner_half_providers + inner_vertex_providers
    )
    inner_providers_per_outer_fold = (
        _INNER_FAMILY_COUNT * (len(_RESPONSES) + 2)
    )
    outer_providers = _FAMILY_COUNT * len(_ARMS)
    total_forwards = (
        authority_forwards + endpoint_forwards + inner_forwards + outer_forwards
    )
    authority_backwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY
    total_backwards = authority_backwards + endpoint_forwards
    teacher_accesses = endpoint_forwards + inner_forwards + outer_forwards
    if (
        total_forwards != 2640
        or total_backwards != 128
        or endpoint_forwards != 112
        or inner_forwards != 2352
        or inner_response_forwards != 2128
        or inner_half_forwards != 112
        or inner_vertex_forwards != 112
        or outer_forwards != 144
        or teacher_accesses != 2608
        or inner_providers != 1176
        or inner_response_providers != 1064
        or inner_half_providers != 56
        or inner_vertex_providers != 56
        or inner_providers_per_outer_fold != 147
        or outer_providers != 72
    ):
        raise RuntimeError("V20n canonical work schedule drifted")
    return {
        "accounting_scope": "canonical_one_shot_schedule",
        "resume_and_authentication_overhead_excluded": True,
        "live_authority_collection_model_forward_count": authority_forwards,
        "endpoint_reconstruction_model_forward_count": endpoint_forwards,
        "inner_conditional_leave_one_family_out_model_forward_count": (
            inner_forwards
        ),
        "inner_original_response_model_forward_count": inner_response_forwards,
        "inner_simplex_shrinkage_lambda_half_model_forward_count": (
            inner_half_forwards
        ),
        "inner_simplex_shrinkage_vertex_model_forward_count": (
            inner_vertex_forwards
        ),
        "inner_endpoint_retrained_per_fold": False,
        "outer_held_model_forward_count": outer_forwards,
        "canonical_model_forward_count": total_forwards,
        "total_model_forward_count": total_forwards,
        "canonical_teacher_access_count": teacher_accesses,
        "total_teacher_access_count": teacher_accesses,
        "live_authority_collection_suffix_backward_count": authority_backwards,
        "endpoint_reconstruction_suffix_backward_count": endpoint_forwards,
        "canonical_suffix_backward_count": total_backwards,
        "total_suffix_backward_count": total_backwards,
        "endpoint_reconstruction_local_autograd_contraction_count": (
            endpoint_forwards
        ),
        "canonical_local_autograd_contraction_count": endpoint_forwards,
        "total_local_autograd_contraction_count": endpoint_forwards,
        "masked_fisher_solve_count": (
            _FAMILY_COUNT * _INNER_FAMILY_COUNT
        ),
        "reflection_fit_count": (
            _FAMILY_COUNT * (_INNER_FAMILY_COUNT + 1)
        ),
        "reflection_variant_receipt_count": (
            _FAMILY_COUNT * (_INNER_FAMILY_COUNT + 1) * 5
        ),
        "simplex_response_candidate_count": (
            inner_response_providers
        ),
        "simplex_shrinkage_lambda_half_candidate_count": (
            inner_half_providers
        ),
        "simplex_shrinkage_vertex_candidate_count": (
            inner_vertex_providers
        ),
        "inner_provider_candidate_count": (
            inner_providers
        ),
        "inner_providers_and_traces_staged_per_outer_fold": (
            inner_providers_per_outer_fold
        ),
        "inner_providers_and_traces_staged_global_count": inner_providers,
        "outer_arm_provider_count": outer_providers,
        "inner_response_trace_example_count": inner_response_forwards,
        "inner_simplex_shrinkage_lambda_half_trace_example_count": (
            inner_half_forwards
        ),
        "inner_simplex_shrinkage_vertex_trace_example_count": (
            inner_vertex_forwards
        ),
        "outer_response_trace_example_count": outer_forwards,
        "endpoint_health_trace_example_count": endpoint_forwards,
        "all_eight_final_refit_model_forward_count": 0,
        "calibration_b_forward_or_tokenization_count": 0,
    }


def _build_report(
    *,
    output: Path,
    source: Mapping[str, object],
    v20g_report: Mapping[str, object],
    v20i_report: Mapping[str, object],
    v20m_report: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    fold_fragments: Mapping[str, Mapping[str, object]],
    decision: Mapping[str, object] | None = None,
) -> dict[str, object]:
    folds = _fold_receipt_map(fold_fragments)
    aggregate = (
        _aggregate_decision(folds)
        if decision is None
        else _validate_hashed(
            decision, domain=_DECISION_DOMAIN, label="V20n aggregate decision"
        )
    )
    families = tuple(
        _identifier(item, label="V20n report family")
        for item in _sequence(
            aggregate.get("family_ids"), label="V20n report families"
        )
    )
    if (
        len(families) != _FAMILY_COUNT
        or set(fold_fragments) != set(families)
        or set(folds) != set(families)
    ):
        raise RuntimeError("V20n report requires all eight authenticated folds")
    replayed = _aggregate_decision(folds)
    if _v14._canonical_json_bytes(replayed) != _v14._canonical_json_bytes(
        aggregate
    ):
        raise ValueError("V20n supplied decision differs from fold replay")
    passed = aggregate.get("development_oof_passed") is True
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": (
            _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
        ),
        "masked_direction_protocol_sha256": (
            _reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
        ),
        "simplex_response_fit_protocol_sha256": (
            _simplex_response_fit.SOFT_POLARITY_SIMPLEX_RESPONSE_FIT_PROTOCOL_SHA256
        ),
        "simplex_shrinkage_fit_protocol_sha256": (
            _shrinkage_fit.SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_PROTOCOL_SHA256
        ),
        "simplex_shrinkage_provider_protocol_sha256": (
            FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256
        ),
        "exact_objective_kind": (
            "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
        ),
        "fixed_protocol": _FIXED_PROTOCOL,
        "source_receipt": dict(source),
        "v20g_authority": {
            "report_sha256": v20g_report.get("report_sha256"),
            "classification": v20g_report.get("classification"),
            "passed": v20g_report.get("passed"),
            "rollback_to_base": v20g_report.get("rollback_to_base"),
            "fold_fragment_sha256s_by_family": {
                family: source["v20g_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "v20i_authority": {
            "report_sha256": v20i_report.get("report_sha256"),
            "classification": v20i_report.get("classification"),
            "development_oof_passed": v20i_report.get("development_oof_passed"),
            "primary_development_gate_passed": v20i_report.get(
                "primary_development_gate_passed"
            ),
            "mechanism_gate_passed": v20i_report.get("mechanism_gate_passed"),
            "passed": v20i_report.get("passed"),
            "rollback_to_base": v20i_report.get("rollback_to_base"),
            "fold_fragment_sha256s_by_family": {
                family: source["v20i_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "v20j_authority": {
            "report_sha256": source["v20j_report_sha256"],
            "file_sha256": source["v20j_file_sha256"],
            "source_receipt_sha256": source["v20j_source_receipt_sha256"],
            "classification": (
                "soft_polarity_confidence_nested_oof_failed_rollback_to_base"
            ),
            "development_oof_passed": False,
            "primary_development_gate_passed": False,
            "mechanism_gate_passed": False,
            "passed": False,
            "rollback_to_base": True,
            "fold_fragment_sha256s_by_family": {
                family: source["v20j_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "v20k_authority": {
            "report_sha256": source["v20k_report_sha256"],
            "file_sha256": source["v20k_file_sha256"],
            "source_receipt_sha256": source["v20k_source_receipt_sha256"],
            "classification": (
                "soft_polarity_log_response_nested_oof_failed_rollback_to_base"
            ),
            "development_oof_passed": False,
            "primary_development_gate_passed": False,
            "mechanism_gate_passed": False,
            "integrity_passed": True,
            "passed": False,
            "rollback_to_base": True,
            "final_refit": None,
            "calibration_b_opened": False,
            "fold_fragment_sha256s_by_family": {
                family: source["v20k_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "v20l_authority": {
            "report_sha256": source["v20l_report_sha256"],
            "file_sha256": source["v20l_file_sha256"],
            "source_receipt_sha256": source["v20l_source_receipt_sha256"],
            "classification": source["v20l_classification"],
            "passed": source["v20l_passed"],
            "rollback_to_base": source["v20l_rollback_to_base"],
            "integrity_passed": True,
            "final_refit": None,
            "calibration_b_opened": False,
            "fold_fragment_sha256s_by_family": {
                family: source["v20l_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "v20m_authority": {
            "report_sha256": _V20M_LOGICAL_SHA256,
            "file_sha256": _V20M_FILE_SHA256,
            "source_receipt_sha256": _V20M_SOURCE_SHA256,
            "classification": v20m_report.get("classification"),
            "development_oof_passed": v20m_report.get(
                "development_oof_passed"
            ),
            "primary_development_gate_passed": v20m_report.get(
                "primary_development_gate_passed"
            ),
            "mechanism_gate_passed": v20m_report.get(
                "mechanism_gate_passed"
            ),
            "passed": v20m_report.get("passed"),
            "rollback_to_base": v20m_report.get("rollback_to_base"),
            "integrity_passed": True,
            "final_refit": None,
            "calibration_b_opened": False,
            "fold_fragment_sha256s_by_family": {
                family: source["v20m_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "panel_receipt": dict(panel_receipt),
        "bridge_binding_sha256": bridge_binding_sha256,
        "fold_fragment_sha256s_by_family": {
            family: fold_fragments[family]["fragment_sha256"]
            for family in families
        },
        "fold_receipts_by_family": {
            family: dict(folds[family]) for family in families
        },
        "decision": dict(aggregate),
        "classification": (
            "soft_polarity_simplex_shrinkage_nested_oof_passed_fresh_shadow_eligible"
            if passed
            else "soft_polarity_simplex_shrinkage_nested_oof_failed_rollback_to_base"
        ),
        "passed": passed,
        "development_oof_passed": passed,
        "primary_development_gate_passed": (
            aggregate.get("primary_development_gate_passed") is True
        ),
        "mechanism_gate_passed": aggregate.get("mechanism_gate_passed") is True,
        "continuous_shrinkage_evidence_gate_passed": aggregate.get(
            "continuous_shrinkage_evidence_gate_passed"
        )
        is True,
        "all_eight_outer_folds_completed": True,
        "all_eight_final_refit_completed": False,
        "full_refit_performed": False,
        "final_refit_authorized_for_next_fresh_shadow": passed,
        "fresh_family_disjoint_shadow_eligible": passed,
        "fresh_family_disjoint_scoring_performed": False,
        "final_refit": None,
        "final_provider_frozen": False,
        "rollback_to_base": not passed,
        "calibration_b_eligibility_gate_passed": False,
        "calibration_b_eligible": False,
        "calibration_b_authorized": False,
        "calibration_b_manifest_read": False,
        "calibration_b_opened": False,
        "calibration_b_tokenized": False,
        "calibration_b_scored": False,
        "validation_opened": False,
        "test_opened": False,
        "serving_claim_authorized": False,
        "compression_claim_authorized": False,
        "speed_claim_authorized": False,
        "fixed_minus_is_diagnostic_only": True,
        "candidate": None,
        "provider_sidecar": None,
        "next_rung": (
            "fresh_family_disjoint_shadow_then_all_eight_refit"
            if passed
            else "revise_continuous_shrinkage_fit_then_repeat_nested_OOF"
        ),
        "work_accounting": _runner_work_accounting(),
        "integrity": {
            "V20g_through_V20m_reports_and_all_fragments_authenticated_before_model_"
            "construction": True,
            "all_56_masked_directions_use_only_six_training_family_summaries": True,
            "inner_response_selection_is_conditional_on_each_fixed_seven_"
            "family_endpoint_not_full_inner_model_cross_validation": True,
            "all_eight_outer_held_families_absent_from_endpoint_direction_"
            "reflection_response_and_lambda_selection": True,
            "all_147_inner_providers_and_traces_staged_before_each_outer_fold_"
            "inner_scoring": True,
            "all_1176_inner_providers_and_traces_staged_across_eight_outer_"
            "folds_before_corresponding_inner_scoring": True,
            "all_nine_outer_providers_and_traces_frozen_before_each_outer_"
            "capability": True,
            "all_inner_zero_response_V20g_eta_zero_output_anchors_passed": True,
            "all_outer_base_fixed_plus_fixed_minus_V20g_output_anchors_"
            "passed": True,
            "all_matched_V20l_boundary_exact_output_anchors_passed": True,
            "all_matched_V20m_response_and_outer_output_anchors_passed": True,
            "lineage_wrappers_not_used_as_inference_executors": True,
            "no_all_eight_refit_or_calibration_b_access_in_this_rung": True,
            "raw_prompts_tokens_logits_h4_gradients_or_provider_tensors_"
            "serialized": False,
        },
        "artifact": None,
    }
    _v14._scalar_report(report)
    return report


def _load_existing_report(
    output: Path,
    *,
    source: Mapping[str, object],
    v20g_report: Mapping[str, object],
    v20i_report: Mapping[str, object],
    v20m_report: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    authenticated_v20g_folds: Mapping[str, Mapping[str, object]],
    authenticated_v20i_folds: Mapping[str, Mapping[str, object]],
    authenticated_v20l_folds: Mapping[str, Mapping[str, object]],
    authenticated_v20m_folds: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20n simplex-shrinkage nested report",
    )
    families = tuple(sorted(authenticated_v20g_folds))
    if (
        set(authenticated_v20i_folds) != set(families)
        or set(authenticated_v20l_folds) != set(families)
        or set(authenticated_v20m_folds) != set(families)
    ):
        raise ValueError("V20n report authority family geometry differs")
    folds = {
        family: _load_fold_fragment(
            output=output,
            source=source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding_sha256,
            authenticated_v20g_fold=authenticated_v20g_folds[family],
            authenticated_v20i_fold=authenticated_v20i_folds[family],
            authenticated_v20l_fold=authenticated_v20l_folds[family],
            authenticated_v20m_fold=authenticated_v20m_folds[family],
        )
        for family in families
    }
    rebuilt = _build_report(
        output=output,
        source=source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        v20m_report=v20m_report,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding_sha256,
        fold_fragments=folds,
    )
    supplied = dict(value)
    report_sha = supplied.pop("report_sha256", None)
    if (
        _v14._canonical_json_bytes(supplied)
        != _v14._canonical_json_bytes(rebuilt)
        or report_sha != _v14._sha256(rebuilt, domain=_REPORT_DOMAIN)
    ):
        raise ValueError("V20n report reconstruction differs")
    return dict(value)


def run_gemma3_l3_l4_complete_h4_soft_polarity_simplex_shrinkage_nested_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or resume the V20n continuous simplex-shrinkage development screen."""

    destination = _validate_output(output)
    (
        prerequisite,
        authenticated_v20a_folds,
        v20g_report,
        authenticated_v20g_folds,
        v20i_report,
        authenticated_v20i_folds,
        _v20l_report,
        authenticated_v20l_folds,
        v20m_report,
        authenticated_v20m_folds,
        source,
    ) = _load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"), label="V20n panel receipt"
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20n bridge binding",
    )
    if destination.exists():
        return _load_existing_report(
            destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            v20m_report=v20m_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_folds=authenticated_v20g_folds,
            authenticated_v20i_folds=authenticated_v20i_folds,
            authenticated_v20l_folds=authenticated_v20l_folds,
            authenticated_v20m_folds=authenticated_v20m_folds,
        )

    family_ids = tuple(sorted(authenticated_v20g_folds))
    if (
        len(family_ids) != _FAMILY_COUNT
        or set(authenticated_v20a_folds) != set(family_ids)
        or set(authenticated_v20i_folds) != set(family_ids)
        or set(authenticated_v20l_folds) != set(family_ids)
        or set(authenticated_v20m_folds) != set(family_ids)
        or set(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="V20n panel families",
            )
        )
        != set(family_ids)
    ):
        raise RuntimeError("V20n authenticated family geometry differs")

    # The final aggregation is model-free.  Completed fold fragments remain
    # authoritative after an interruption and must never trigger a second
    # Gemma construction or a hidden all-eight refit.
    if all(_fold_path(destination, family).exists() for family in family_ids):
        completed = {
            family: _load_fold_fragment(
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                outer_family_id=family,
                bridge_binding_sha256=bridge_binding,
                authenticated_v20g_fold=authenticated_v20g_folds[family],
                authenticated_v20i_fold=authenticated_v20i_folds[family],
                authenticated_v20l_fold=authenticated_v20l_folds[family],
                authenticated_v20m_fold=authenticated_v20m_folds[family],
            )
            for family in family_ids
        }
        report = _build_report(
            output=destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            v20m_report=v20m_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            fold_fragments=completed,
        )
        try:
            _v20b._publish_scalar_fragment(
                report,
                path=destination,
                domain=_REPORT_DOMAIN,
                hash_key="report_sha256",
                label="V20n simplex-shrinkage nested report",
            )
        except FileExistsError:
            pass
        return _load_existing_report(
            destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            v20m_report=v20m_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_folds=authenticated_v20g_folds,
            authenticated_v20i_folds=authenticated_v20i_folds,
            authenticated_v20l_folds=authenticated_v20l_folds,
            authenticated_v20m_folds=authenticated_v20m_folds,
        )

    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        if context.bridge.bridge_binding_sha256 != bridge_binding:
            raise RuntimeError("V20n live bridge differs from authenticated authority")
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=prerequisite
        )
        if tuple(live_families) != family_ids:
            raise RuntimeError("V20n live family order differs from authenticated A16")
        fragments: dict[str, dict[str, object]] = {}
        for family in family_ids:
            if _fold_path(destination, family).exists():
                fragments[family] = _load_fold_fragment(
                    output=destination,
                    source=source,
                    panel_receipt=panel_receipt,
                    outer_family_id=family,
                    bridge_binding_sha256=bridge_binding,
                    authenticated_v20g_fold=authenticated_v20g_folds[family],
                    authenticated_v20i_fold=authenticated_v20i_folds[family],
                    authenticated_v20l_fold=authenticated_v20l_folds[family],
                    authenticated_v20m_fold=authenticated_v20m_folds[family],
                )
                continue
            live = _execute_outer_fold(
                context,
                records,
                teacher_vault,
                family_ids=family_ids,
                outer_family_id=family,
                panel_receipt=panel_receipt,
                authenticated_v20a_fold=authenticated_v20a_folds[family],
                authenticated_v20g_fold=authenticated_v20g_folds[family],
                authenticated_v20i_fold=authenticated_v20i_folds[family],
                authenticated_v20l_fold=authenticated_v20l_folds[family],
                authenticated_v20m_fold=authenticated_v20m_folds[family],
            )
            payload = _fold_payload(
                live,
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                bridge_binding_sha256=bridge_binding,
                outer_family_id=family,
                authenticated_v20g_fold=authenticated_v20g_folds[family],
                authenticated_v20i_fold=authenticated_v20i_folds[family],
                authenticated_v20l_fold=authenticated_v20l_folds[family],
                authenticated_v20m_fold=authenticated_v20m_folds[family],
            )
            try:
                _publish_fold_fragment(
                    payload, output=destination, outer_family_id=family
                )
            except FileExistsError:
                pass
            fragments[family] = _load_fold_fragment(
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                outer_family_id=family,
                bridge_binding_sha256=bridge_binding,
                authenticated_v20g_fold=authenticated_v20g_folds[family],
                authenticated_v20i_fold=authenticated_v20i_folds[family],
                authenticated_v20l_fold=authenticated_v20l_folds[family],
                authenticated_v20m_fold=authenticated_v20m_folds[family],
            )
        report = _build_report(
            output=destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            v20m_report=v20m_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            fold_fragments=fragments,
        )
    finally:
        context.validate_immutable_inputs()
        context.close()

    try:
        _v20b._publish_scalar_fragment(
            report,
            path=destination,
            domain=_REPORT_DOMAIN,
            hash_key="report_sha256",
            label="V20n simplex-shrinkage nested report",
        )
    except FileExistsError:
        pass
    return _load_existing_report(
        destination,
        source=source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        v20m_report=v20m_report,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_v20g_folds=authenticated_v20g_folds,
        authenticated_v20i_folds=authenticated_v20i_folds,
        authenticated_v20l_folds=authenticated_v20l_folds,
        authenticated_v20m_folds=authenticated_v20m_folds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V20n nested soft-polarity simplex-shrinkage development "
            "screen"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = (
        run_gemma3_l3_l4_complete_h4_soft_polarity_simplex_shrinkage_nested_development(
            output=arguments.output,
            cache_dir=arguments.cache_dir,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
