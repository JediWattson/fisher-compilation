"""V20o nested signed-continuum campaign for Gemma 3.

V20o authenticates the completed V20n report and all eight V20n folds before
model construction, while retaining V20m as the exact response/output
authority.  It reproduces V20m's nineteen-response conditional inner-LOFO
selection and then fits one signed scalar ``s`` for the selected response::

    t = abs(s)
    sigma = +1 if s >= 0 else -1
    g_s = e * ((1 - t) + t * q(sigma * z))

Thus ``s=-1`` is the exact V20m mirror, ``s=0`` is fixed-plus, and ``s=+1``
is the exact V20m reflected response.  For each outer fold, all fourteen
missing signed-anchor providers (seven mirror and seven fixed-plus) are frozen
before either missing anchor is scored.  Those exact anchors plus the reused
V20m ``s=+1`` score define a quadratic proposal.  All seven proposed-vertex
providers are then frozen before exact vertex scoring, and the deterministic
core fitter selects the exact minimum among the three anchors and vertex.

The outer-held family remains unopened until the response, signed scalar, all
nine outer providers, and all traces are frozen.  The lineage wrapper is never
installed for execution; its materialized ``runtime_provider`` is used instead.
There is no all-eight refit, Calibration-B access, compression, speed, serving,
or held-shadow claim.  Reports and resumable folds contain scalar/hash evidence
only and are mode 0600.
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
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_simplex_shrinkage_nested_development
    as _v20n,
)
from . import complete_h4_fisher_soft_polarity_reflection_fit as _reflection
from . import complete_h4_fisher_soft_polarity_simplex_response_fit as _simplex_response_fit
from . import complete_h4_fisher_soft_polarity_signed_continuum_fit as _signed_continuum_fit
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
from .complete_h4_fisher_soft_polarity_signed_continuum import (
    AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider,
    FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256,
    build_autonomous_complete_h4_fisher_soft_polarity_signed_continuum,
    fisher_soft_polarity_signed_continuum_box_certificate,
    fisher_soft_polarity_signed_continuum_direction_sha256,
    validate_fisher_soft_polarity_signed_continuum_provider_evidence,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_soft_polarity_signed_continuum_nested_development",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-soft-polarity-signed-continuum-nested-"
    "r16-k256-a-fit16-dev-v20o.json"
)

_V20N_OUTPUT = _v20n.DEFAULT_OUTPUT
_V20N_LOGICAL_SHA256 = (
    "af0eccc872b254d33504d3c54ae05a7b519dd4a25e3295aa83f2c780bbf15a97"
)
_V20N_FILE_SHA256 = (
    "847603d36325ee3115ec6d298a67e43f86c7f5995d69f6ac7e63721045006d8d"
)
_V20N_SOURCE_SHA256 = (
    "62f3270128ccde40704107eedb2991f8b3aee8d36fceee10b5d3c31f078d43ce"
)
_V20N_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "dd060f2ba5f5ca36eef1dacf2614848453953d223d6f852346c2ed85f6bd930f"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "9e9cfe0c49814621a45b1f21d615c77a6f0928b0a6bdb5d8b9ab4ea7ebda2824"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "3a939f11ae7cf176d89065dabd071cdd05603091238151cb42046c107b314442"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "f94a65dd2d110903b649d38b530c6b992b5530c318371fe544dacc45504ead2f"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "ef3ac375cee1cfcb4bcd9c94e44b10cfc54d11d72e464d4617318af583390690"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "d4bd0db22022d02a1737bc1c10db6cb6dc7ab25b36a7a03b59dbe1f985704683"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "8f916accc9835819e8d5a21d2b71e34466580bdda900ac70bc5c25d4edb631a3"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "675af5d7d9f7a427a4155f9f62f7cde13ad112e231af6266c33f92e9b3d3489d"
    ),
}

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
    "fisher_graph.gemma3_l3_l4.complete_h4_soft_polarity_signed_continuum_nested.v20o"
)
_FOLD_SCHEMA = (
    "fisher_graph.complete_h4_soft_polarity_signed_continuum_nested_outer_fold.v20o"
)
_FORMAT_VERSION = 31
_REPORT_DOMAIN = b"fisher-graph:soft-polarity-signed_continuum-nested-report:v20o\0"
_SOURCE_DOMAIN = b"fisher-graph:soft-polarity-signed_continuum-nested-source:v20o\0"
_FOLD_DOMAIN = b"fisher-graph:soft-polarity-signed_continuum-nested-fold:v20o\0"
_INNER_FIT_DOMAIN = b"fisher-graph:soft-polarity-signed_continuum-inner-fit:v20o\0"
_INNER_MANIFEST_DOMAIN = (
    b"fisher-graph:soft-polarity-signed_continuum-inner-manifest:v20o\0"
)
_INNER_EXECUTION_DOMAIN = (
    b"fisher-graph:soft-polarity-signed_continuum-inner-execution:v20o\0"
)
_RESPONSE_SELECTION_DOMAIN = (
    b"fisher-graph:soft-polarity-signed_continuum-response-selection:v20o\0"
)
_SIGNED_CONTINUUM_SELECTION_DOMAIN = (
    b"fisher-graph:soft-polarity-signed_continuum-selection:v20o\0"
)
_OUTER_MANIFEST_DOMAIN = (
    b"fisher-graph:soft-polarity-signed_continuum-outer-manifest:v20o\0"
)
_OUTER_EXECUTION_DOMAIN = (
    b"fisher-graph:soft-polarity-signed_continuum-outer-execution:v20o\0"
)
_PROVIDER_DOMAIN = b"fisher-graph:soft-polarity-signed_continuum-provider:v20o\0"
_TRACE_DOMAIN = b"fisher-graph:soft-polarity-signed_continuum-trace:v20o\0"
_DECISION_DOMAIN = b"fisher-graph:soft-polarity-signed_continuum-decision:v20o\0"

_FAMILY_COUNT = 8
_PROMPTS_PER_FAMILY = 2
_INNER_FAMILY_COUNT = 7
_INNER_TRAINING_FAMILY_COUNT = 6
_CONDITIONAL_RANK = 16
_SIGNED_CONTINUUM_ANCHOR_VALUES = (-1.0, 0.0, 1.0)
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
    raise RuntimeError("V20o runner response ladder differs from core protocol")
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
    "signed_continuum_reflected",
    "simplex_response_reflected_exact_mirror",
    "matched_v20m_simplex_reflected",
)
_PRIMARY_ARM = "signed_continuum_reflected"
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
_SIGNED_CONTINUUM_PROVIDER_RECEIPT_KEYS = frozenset(
    {
        "role",
        "source_response",
        "source_response_key",
        "signed_scalar",
        "signed_scalar_hex",
        "compiled_direction_sign",
        "compiled_mix",
        "compiled_mix_hex",
        "source_direction",
        "compiled_direction",
        "source_direction_box_corner_scores",
        "compiled_direction_box_corner_scores",
        "source_direction_sha256",
        "compiled_direction_sha256",
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
    "protocol": "v20o_nested_signed_continuum_fit",
    "scientific_status": (
        "predeclared_after_completed_v20n_reused_A16_development_hypothesis_only"
    ),
    "source": (
        "pinned_V20n_report_and_folds_with_exact_pinned_V20m_output_authority"
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
        "reproduce_V20m_nineteen_response_selection_before_signed_fit"
    ),
    "simplex_response_formula": (
        "one_minus_u_times_z_squared_times_tanh_radius_z_plus_v_times_z_squared"
    ),
    "simplex_response_constraints": (
        "radius_in_zero_one_fourth_and_zero_less_equal_abs_v_less_equal_u_"
        "less_equal_one_fourth_finite_bounded_on_normalized_box"
    ),
    "signed_continuum_anchor_values": _SIGNED_CONTINUUM_ANCHOR_VALUES,
    "signed_continuum_fit_protocol_sha256": (
        _signed_continuum_fit.SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256
    ),
    "signed_continuum_provider_protocol_sha256": (
        FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256
    ),
    "signed_continuum_fit": (
        "reuse_exact_signed_plus_one_V20m_scores_then_freeze_all_fourteen_"
        "signed_minus_one_mirror_and_signed_zero_fixed_plus_providers_before_"
        "either_missing_anchor_score_then_freeze_quadratic_proposal_then_"
        "freeze_and_exact_score_all_seven_vertex_providers_then_select_exact_"
        "minimum_minus_one_zero_plus_one_vertex_by_core_deterministic_order"
    ),
    "inner_freeze_barrier": (
        "V20m_133_provider_freeze_then_all_fourteen_missing_anchor_providers_"
        "and_traces_before_any_anchor_capability_then_all_seven_vertex_"
        "providers_and_traces_before_any_vertex_capability"
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
    "signed_continuum_evidence_gate": (
        "at_least_five_outer_folds_select_s_strictly_inside_either_"
        "minus_one_zero_or_zero_plus_one_segment_"
        "with_at_least_one_negative_non_anchor_interior_and_one_positive_"
        "non_anchor_interior"
    ),
    "interior_exact_distinct_gate": (
        "every_strict_interior_candidate_exact_output_distinct_from_signed_"
        "minus_one_mirror_signed_zero_fixed_plus_and_signed_plus_one_V20m"
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
        "signed_continuum_fit_protocol_sha256": (
            _signed_continuum_fit.SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256
        ),
        "signed_continuum_provider_protocol_sha256": (
            FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256
        ),
        "operation": "V20o_domain_separated_signed_continuum_materialization",
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
    raw = _sequence(value, label="V20o simplex parameters")
    if len(raw) != 3:
        raise ValueError("V20o response must contain exactly radius, u, and v")
    if any(type(item) not in (int, float) for item in raw):
        raise ValueError("V20o response values must be JSON numbers")
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
        raise ValueError("V20o simplex parameters are outside the certified domain")
    return selected[0], selected[1], selected[2]


def _response_tuple(value: object) -> tuple[float, float, float]:
    selected = _simplex_parameters(value)
    if selected not in _RESPONSES:
        raise ValueError("V20o response is outside the fixed ladder")
    return selected


def _parameters_key(value: object) -> str:
    radius, u, v = _simplex_parameters(value)
    return f"radius={radius.hex()};u={u.hex()};v={v.hex()}"


def _response_key(value: object) -> str:
    return _parameters_key(_response_tuple(value))


def _response_order(value: object) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        _response_tuple(item)
        for item in _sequence(value, label="V20o response order")
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
        label="V20o inherited V20i outer reflection fit",
    )
    if _v14._canonical_json_bytes(outer_reflection_fit) != (
        _v14._canonical_json_bytes(inherited_outer)
    ):
        raise ValueError("V20o outer reflection lineage differs from pinned V20i")

    current_inner = _mapping(
        inner_receipt.get("inner_evidence_by_family"),
        label="V20o current inner reflection evidence",
    )
    inherited_inner_receipt = _mapping(
        authenticated_v20i_fold.get("inner_receipt"),
        label="V20o inherited V20i inner receipt",
    )
    inherited_inner = _mapping(
        inherited_inner_receipt.get("inner_evidence_by_family"),
        label="V20o inherited V20i inner reflection evidence",
    )
    if set(current_inner) != set(inherited_inner):
        raise ValueError("V20o inner reflection family lineage differs from V20i")
    for family in sorted(current_inner):
        current = _mapping(
            current_inner[family], label="V20o current inner reflection family"
        )
        inherited = _mapping(
            inherited_inner[family], label="V20o inherited inner reflection family"
        )
        for field in ("masked_direction_receipt", "reflection_fit_receipt"):
            if _v14._canonical_json_bytes(current.get(field)) != (
                _v14._canonical_json_bytes(inherited.get(field))
            ):
                raise ValueError(
                    f"V20o {field} lineage differs from pinned V20i"
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
            _V20N_OUTPUT,
            *(
                _v20n._fold_path(_V20N_OUTPUT, family)
                for family in sorted(_V20N_FOLD_SHA256S)
            ),
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
        raise ValueError("V20o output must preserve immutable prerequisite artifacts")
    if destination.parent != local_root:
        raise ValueError("V20o output must remain directly under .local-runs")
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
    signed_continuum_selection: dict[str, object]
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
    dict[str, dict[str, object]],
    dict[str, object],
]:
    """Authenticate V20n, V20m, and their complete lineage pre-model."""

    if (
        _V20N_LOGICAL_SHA256 is None
        or _V20N_FILE_SHA256 is None
        or _V20N_SOURCE_SHA256 is None
        or _V20N_FOLD_SHA256S is None
    ):
        raise RuntimeError(
            "V20o is fail-closed until the completed V20n report and all eight "
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
        v20m_report,
        authenticated_v20m_folds,
        v20n_source,
    ) = _v20n._load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"),
            label="V20o inherited panel receipt",
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20o inherited bridge binding",
    )
    if _v14._file_sha256(_V20N_OUTPUT) != _V20N_FILE_SHA256:
        raise RuntimeError("pinned V20n report file hash drifted")
    v20n_report = _v20n._load_existing_report(
        _V20N_OUTPUT,
        source=v20n_source,
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
    observed_fold_hashes = {
        _identifier(family, label="V20o V20n fold family"): _sha(
            value, label="V20o V20n fold hash"
        )
        for family, value in _mapping(
            v20n_report.get("fold_fragment_sha256s_by_family"),
            label="V20o V20n fold hashes",
        ).items()
    }
    if (
        v20n_report.get("report_sha256") != _V20N_LOGICAL_SHA256
        or v20n_report.get("all_eight_outer_folds_completed") is not True
        or _mapping(
            v20n_report.get("decision"), label="V20o V20n decision"
        ).get("integrity_passed")
        is not True
        or v20n_report.get("final_refit") is not None
        or v20n_report.get("calibration_b_opened") is not False
        or _mapping(
            v20n_report.get("source_receipt"), label="V20o V20n source"
        ).get("artifact_sha256")
        != _V20N_SOURCE_SHA256
        or observed_fold_hashes != _V20N_FOLD_SHA256S
    ):
        raise RuntimeError("pinned V20n development authority differs")

    families = tuple(sorted(_V20N_FOLD_SHA256S))
    authenticated_v20n_folds = {
        family: _v20n._load_fold_fragment(
            output=_V20N_OUTPUT,
            source=v20n_source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_fold=authenticated_v20g_folds[family],
            authenticated_v20i_fold=authenticated_v20i_folds[family],
            authenticated_v20l_fold=authenticated_v20l_folds[family],
            authenticated_v20m_fold=authenticated_v20m_folds[family],
        )
        for family in families
    }
    if {
        family: fragment["fragment_sha256"]
        for family, fragment in authenticated_v20n_folds.items()
    } != _V20N_FOLD_SHA256S:
        raise RuntimeError("pinned V20n fold authority differs")
    inherited_source = {
        key: value
        for key, value in v20n_source.items()
        if key != "artifact_sha256"
    }
    source = _hashed(
        {
            **inherited_source,
            "v20n_parent_source_receipt_sha256": v20n_source[
                "artifact_sha256"
            ],
            "v20n_report_sha256": _V20N_LOGICAL_SHA256,
            "v20n_file_sha256": _V20N_FILE_SHA256,
            "v20n_source_receipt_sha256": _V20N_SOURCE_SHA256,
            "v20n_classification": v20n_report.get("classification"),
            "v20n_passed": v20n_report.get("passed"),
            "v20n_rollback_to_base": v20n_report.get("rollback_to_base"),
            "v20n_fold_fragment_sha256s_by_family": dict(
                sorted(_V20N_FOLD_SHA256S.items())
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
            "signed_continuum_fit_protocol_sha256": (
                _signed_continuum_fit.SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256
            ),
            "signed_continuum_provider_protocol_sha256": (
                FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256
            ),
            "response_order": _RESPONSES,
            "signed_continuum_anchor_values": _SIGNED_CONTINUUM_ANCHOR_VALUES,
            "exact_objective_kind": (
                "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
            ),
            "authenticated_before_model_construction": True,
            "historically_reused_A16_only": True,
            "held_scores_used_before_direction_response_or_signed_scalar_freeze": False,
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
        dict(v20n_report),
        authenticated_v20n_folds,
        source,
    )


def _selected_direction(
    reflection_fit: Mapping[str, object],
) -> tuple[float, float, float, float]:
    if reflection_fit.get("selected_variant_available") is not True:
        raise RuntimeError("V20o reflection fit has no admissible direction")
    raw = tuple(
        float(item)
        for item in _sequence(
            reflection_fit.get("selected_normalized_direction"),
            label="V20o selected reflection direction",
        )
    )
    if len(raw) != 4 or not all(math.isfinite(item) for item in raw):
        raise RuntimeError("V20o reflection direction is not a finite four-vector")
    return raw  # type: ignore[return-value]


def _unreflected_direction(
    direction_receipt: Mapping[str, object],
) -> tuple[float, float, float, float]:
    raw = tuple(
        float(item)
        for item in _sequence(
            direction_receipt.get("natural_direction"),
            label="V20o unreflected direction",
        )
    )
    if len(raw) != 4 or not all(math.isfinite(item) for item in raw):
        raise RuntimeError("V20o unreflected direction is not finite")
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
                endpoint_receipt_sha256, label="V20o provider endpoint"
            ),
            "direction_artifact_sha256": _sha(
                direction_artifact_sha256, label="V20o provider direction"
            ),
            "reflection_fit_sha256": _sha(
                reflection_fit_sha256, label="V20o provider reflection fit"
            ),
            "response": (radius, u, v),
            "response_key": _parameters_key((radius, u, v)),
            "direction": selected_direction,
            "direction_box_corner_scores": _box_corner_scores(selected_direction),
            "outer_held_family_id": _identifier(
                outer_family_id, label="V20o provider outer family"
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


def _signed_continuum_provider_seed(
    *,
    endpoint_receipt_sha256: str,
    direction: Sequence[float],
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    signed_scalar: float,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> str:
    radius, u, v = _simplex_parameters(response)
    signed_continuum = float(signed_scalar)
    if not math.isfinite(signed_continuum) or not -1.0 <= signed_continuum <= 1.0:
        raise ValueError("V20o signed continuum scalar must be inside [-1,1]")
    return _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": _sha(
                endpoint_receipt_sha256,
                label="V20o signed_continuum provider endpoint",
            ),
            "direction_artifact_sha256": _sha(
                direction_artifact_sha256,
                label="V20o signed_continuum provider direction",
            ),
            "reflection_fit_sha256": _sha(
                reflection_fit_sha256,
                label="V20o signed_continuum provider reflection fit",
            ),
            "source_response": (radius, u, v),
            "source_response_key": _parameters_key((radius, u, v)),
            "signed_scalar": signed_continuum,
            "signed_scalar_hex": signed_continuum.hex(),
            "direction": tuple(float(item) for item in direction),
            "direction_box_corner_scores": _box_corner_scores(direction),
            "outer_held_family_id": _identifier(
                outer_family_id, label="V20o signed_continuum provider outer family"
            ),
            "inner_held_family_id": inner_family_id,
            "role": role,
            "held_rows_used": False,
        },
        domain=_PROVIDER_DOMAIN,
    )


def _materialize_signed_continuum_provider(
    endpoint: _v20g._EndpointLive,
    *,
    direction: Sequence[float],
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    signed_scalar: float,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> tuple[AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider, str]:
    radius, u, v = _simplex_parameters(response)
    signed_continuum = float(signed_scalar)
    if not math.isfinite(signed_continuum) or not -1.0 <= signed_continuum <= 1.0:
        raise ValueError("V20o signed continuum scalar must be inside [-1,1]")
    seed = _signed_continuum_provider_seed(
        endpoint_receipt_sha256=str(endpoint.receipt["artifact_sha256"]),
        direction=direction,
        direction_artifact_sha256=direction_artifact_sha256,
        reflection_fit_sha256=reflection_fit_sha256,
        response=response,
        signed_scalar=signed_continuum,
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        role=role,
    )
    provider = build_autonomous_complete_h4_fisher_soft_polarity_signed_continuum(
        endpoint.base_provider,
        endpoint.proposal_provider,
        direction=_v20g._eta_tensor(direction),
        radius=radius,
        shrink_mass=u,
        polarity_bias=v,
        signed_scalar=signed_continuum,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
        transfer_evidence_sha256=seed,
    )
    return provider, seed


def _signed_continuum_provider_receipt(
    provider: AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider,
    *,
    role: str,
    response: tuple[float, float, float],
    signed_scalar: float,
    direction: Sequence[float],
) -> dict[str, object]:
    source = _simplex_parameters(response)
    signed_continuum = float(signed_scalar)
    source_direction = tuple(float(item) for item in provider.direction.tolist())
    expected_direction = tuple(float(item) for item in direction)
    if source_direction != expected_direction:
        raise RuntimeError("V20o signed_continuum provider differs from frozen direction")
    if float(provider.signed_scalar) != signed_continuum:
        raise RuntimeError("V20o signed continuum provider differs from frozen scalar")
    compiled_direction = tuple(
        float(item) for item in provider.runtime_provider.direction.tolist()
    )
    expected_sign = 1 if signed_continuum >= 0.0 else -1
    expected_mix = abs(signed_continuum)
    expected_compiled = tuple(expected_sign * item for item in expected_direction)
    if (
        provider.compiled_direction_sign != expected_sign
        or float(provider.compiled_mix) != expected_mix
        or compiled_direction != expected_compiled
    ):
        raise RuntimeError("V20o signed continuum materialization differs")
    metadata = _mapping(provider.metadata(), label=f"V20o {role} metadata")
    payload = provider.artifact_payload()
    receipt = {
        "role": role,
        "source_response": source,
        "source_response_key": _parameters_key(source),
        "signed_scalar": signed_continuum,
        "signed_scalar_hex": signed_continuum.hex(),
        "compiled_direction_sign": expected_sign,
        "compiled_mix": expected_mix,
        "compiled_mix_hex": expected_mix.hex(),
        "source_direction": source_direction,
        "compiled_direction": compiled_direction,
        "source_direction_box_corner_scores": _box_corner_scores(source_direction),
        "compiled_direction_box_corner_scores": _box_corner_scores(compiled_direction),
        "source_direction_sha256": payload.get("source_direction_sha256"),
        "compiled_direction_sha256": payload.get("compiled_direction_sha256"),
        "box_certificate": fisher_soft_polarity_signed_continuum_box_certificate(
            provider.direction,
            radius=source[0],
            shrink_mass=source[1],
            polarity_bias=source[2],
            signed_scalar=signed_continuum,
        ),
        "provider_artifact_sha256": _sha(
            provider.artifact_sha256, label=f"V20o {role} provider artifact"
        ),
        "runtime_provider_artifact_sha256": _sha(
            provider.runtime_provider.artifact_sha256,
            label=f"V20o {role} runtime provider artifact",
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
    metadata = _mapping(provider.metadata(), label=f"V20o {role} metadata")
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
                "V20o simplex_response provider receipt needs response and direction"
            )
        radius, u, v = _simplex_parameters(response)
        expected = tuple(float(item) for item in direction)
        if selected_direction != expected:
            raise RuntimeError("V20o provider differs from its frozen direction")
        if (
            float(provider.radius) != radius
            or float(provider.shrink_mass) != u
            or float(provider.polarity_bias) != v
        ):
            raise RuntimeError("V20o provider coefficients differ from response")
        bound = max(abs(item) for item in corners)
        if abs(bound - 1.0) > 1.0e-12:
            raise RuntimeError("V20o provider direction is not box normalized")
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
            provider.artifact_sha256, label=f"V20o {role} provider artifact"
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
        label="V20o inherited V20i provider manifest",
    )
    receipts = _mapping(
        manifest.get("provider_receipts"),
        label="V20o inherited V20i provider receipts",
    )
    reference_role = {
        "base": "base",
        "fixed_plus": "fixed_plus",
        "fixed_minus": "fixed_minus",
    }.get(role, "fixed_plus")
    authority = _mapping(
        receipts.get(reference_role),
        label=f"V20o inherited V20i {reference_role} provider receipt",
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
        if (
            "signed_continuum" in role
            or role == "same_simplex_response_unreflected"
        ):
            # The V20o runtime retains V20m's four direction/response additions
            # and materializes one additional nonnegative mixture scalar.
            prepared += 1
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
        raise ValueError("V20o provider receipt key set differs")
    provider_artifact = _sha(
        expected_provider_artifact_sha256,
        label="V20o expected provider artifact",
    )
    if (
        receipt.get("role") != expected_role
        or receipt.get("provider_artifact_sha256") != provider_artifact
        or receipt.get("analysis_only") is not (expected_role != "base")
        or receipt.get("raw_provider_tensors_serialized") is not False
    ):
        raise ValueError("V20o provider receipt identity differs")

    metadata = _mapping(
        receipt.get("provider_metadata"), label="V20o provider metadata"
    )
    metadata_sha = _sha(
        receipt.get("provider_metadata_sha256"),
        label="V20o provider metadata hash",
    )
    if (
        _v14._sha256(metadata, domain=_PROVIDER_DOMAIN) != metadata_sha
        or metadata.get("artifact_sha256") != provider_artifact
    ):
        raise ValueError("V20o provider metadata authentication differs")

    receipt_accounting = tuple(
        _strict_receipt_integer(receipt, key, label="V20o provider receipt")
        for key in (
            "rank",
            "conditional_rank",
            "prepared_float_scalar_count",
            "logical_macs_per_token_upper_bound",
        )
    )
    metadata_accounting = tuple(
        _strict_receipt_integer(metadata, key, label="V20o provider metadata")
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
        raise ValueError("V20o provider accounting differs from pinned V20i")

    if metadata.get("bridge_binding_sha256") not in (
        None,
        expected_bridge_binding_sha256,
    ):
        raise ValueError("V20o provider bridge metadata differs")

    simplex_response_role = expected_response is not None or expected_direction is not None
    if simplex_response_role:
        if expected_response is None or expected_direction is None:
            raise ValueError("V20o simplex_response provider expectation is incomplete")
        response = _simplex_parameters(expected_response)
        direction = tuple(float(item) for item in expected_direction)
        if len(direction) != 4 or not all(math.isfinite(item) for item in direction):
            raise ValueError("V20o expected simplex_response direction differs")
        payload = _mapping(
            receipt.get("provider_payload"), label="V20o simplex_response provider payload"
        )
        validated = validate_fisher_soft_polarity_simplex_response_provider_evidence(
            payload, metadata
        )
        if _v14._canonical_json_bytes(
            validated.metadata.get("box_certificate")
        ) != _v14._canonical_json_bytes(receipt.get("box_certificate")):
            raise ValueError("V20o simplex_response provider box certificate differs")
        expected_direction_sha = fisher_soft_polarity_simplex_response_direction_sha256(
            _v20g._eta_tensor(direction)
        )
        endpoint_base = _sha(
            expected_endpoint_receipt.get("base_provider_artifact_sha256"),
            label="V20o endpoint base provider",
        )
        endpoint_proposal = _sha(
            expected_endpoint_receipt.get("proposal_provider_artifact_sha256"),
            label="V20o endpoint proposal provider",
        )
        expected_transfer = _sha(
            expected_transfer_evidence_sha256,
            label="V20o simplex_response transfer evidence",
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
                raise ValueError(f"V20o simplex_response provider {key} differs")
        for key in (
            "parent_provider_artifact_sha256",
            "start_provider_artifact_sha256",
        ):
            inherited = expected_endpoint_receipt.get(key)
            if inherited is not None and validated.payload.get(key) != inherited:
                raise ValueError(f"V20o simplex_response provider {key} differs")
        if validated.artifact_sha256 != provider_artifact:
            raise ValueError("V20o simplex_response provider artifact replay differs")
    else:
        if receipt.get("provider_payload") is not None:
            raise ValueError("V20o non-simplex_response provider serialized a payload")
        if expected_role in ("fixed_plus", "fixed_minus"):
            expected_transfer = _sha(
                expected_transfer_evidence_sha256,
                label="V20o fixed-control transfer evidence",
            )
            expected_bindings = {
                "base_provider_artifact_sha256": _sha(
                    expected_endpoint_receipt.get(
                        "base_provider_artifact_sha256"
                    ),
                    label="V20o fixed-control endpoint base",
                ),
                "proposal_provider_artifact_sha256": _sha(
                    expected_endpoint_receipt.get(
                        "proposal_provider_artifact_sha256"
                    ),
                    label="V20o fixed-control endpoint proposal",
                ),
                "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
                "transfer_evidence_sha256": expected_transfer,
            }
            for key, expected in expected_bindings.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"V20o fixed-control {key} differs")


def _validate_signed_continuum_provider_receipt_evidence(
    receipt: Mapping[str, object],
    *,
    expected_role: str,
    expected_provider_artifact_sha256: str,
    expected_endpoint_receipt: Mapping[str, object],
    expected_bridge_binding_sha256: str,
    authenticated_v20i_fold: Mapping[str, object],
    expected_response: tuple[float, float, float],
    expected_signed_scalar: float,
    expected_direction: Sequence[float],
    expected_transfer_evidence_sha256: str,
) -> None:
    if set(receipt) != _SIGNED_CONTINUUM_PROVIDER_RECEIPT_KEYS:
        raise ValueError("V20o signed_continuum provider receipt key set differs")
    provider_artifact = _sha(
        expected_provider_artifact_sha256,
        label="V20o expected signed_continuum provider artifact",
    )
    response = _simplex_parameters(expected_response)
    signed_continuum = float(expected_signed_scalar)
    direction = tuple(float(item) for item in expected_direction)
    if not math.isfinite(signed_continuum) or not -1.0 <= signed_continuum <= 1.0:
        raise ValueError("V20o expected signed scalar is outside [-1,1]")
    compiled_sign = 1 if signed_continuum >= 0.0 else -1
    compiled_mix = abs(signed_continuum)
    compiled_direction = tuple(compiled_sign * item for item in direction)
    if (
        receipt.get("role") != expected_role
        or receipt.get("provider_artifact_sha256") != provider_artifact
        or _simplex_parameters(receipt.get("source_response")) != response
        or receipt.get("source_response_key") != _parameters_key(response)
        or float(receipt.get("signed_scalar", math.nan)) != signed_continuum
        or receipt.get("signed_scalar_hex") != signed_continuum.hex()
        or receipt.get("compiled_direction_sign") != compiled_sign
        or float(receipt.get("compiled_mix", math.nan)) != compiled_mix
        or receipt.get("compiled_mix_hex") != compiled_mix.hex()
        or tuple(receipt.get("source_direction", ())) != direction
        or tuple(receipt.get("compiled_direction", ())) != compiled_direction
        or tuple(receipt.get("source_direction_box_corner_scores", ()))
        != _box_corner_scores(direction)
        or tuple(receipt.get("compiled_direction_box_corner_scores", ()))
        != _box_corner_scores(compiled_direction)
        or receipt.get("lineage_wrapper_not_inference_executor") is not True
        or receipt.get("analysis_only") is not True
        or receipt.get("raw_provider_tensors_serialized") is not False
    ):
        raise ValueError("V20o signed_continuum provider receipt identity differs")
    metadata = _mapping(
        receipt.get("provider_metadata"),
        label="V20o signed_continuum provider metadata",
    )
    if (
        _v14._sha256(metadata, domain=_PROVIDER_DOMAIN)
        != receipt.get("provider_metadata_sha256")
        or metadata.get("artifact_sha256") != provider_artifact
    ):
        raise ValueError("V20o signed_continuum provider metadata authentication differs")
    payload = _mapping(
        receipt.get("provider_payload"),
        label="V20o signed_continuum provider payload",
    )
    validated = validate_fisher_soft_polarity_signed_continuum_provider_evidence(
        payload, metadata
    )
    expected_transfer = _sha(
        expected_transfer_evidence_sha256,
        label="V20o signed_continuum transfer evidence",
    )
    if (
        validated.artifact_sha256 != provider_artifact
        or validated.payload.get("protocol_sha256")
        != FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256
        or _v14._canonical_json_bytes(
            _mapping(
                receipt.get("box_certificate"),
                label="V20o signed_continuum receipt box certificate",
            )
        )
        != _v14._canonical_json_bytes(
            _mapping(
                validated.metadata.get("box_certificate"),
                label="V20o signed_continuum metadata box certificate",
            )
        )
        or receipt.get("transfer_protocol_sha256") != _TRANSFER_PROTOCOL_SHA256
        or receipt.get("transfer_evidence_sha256") != expected_transfer
    ):
        raise ValueError("V20o signed_continuum provider artifact replay differs")
    if receipt.get("runtime_provider_artifact_sha256") != validated.payload.get(
        "compiled_runtime_provider_artifact_sha256"
    ):
        raise ValueError("V20o signed_continuum runtime provider artifact differs")
    expected_bindings = {
        "bridge_binding_sha256": expected_bridge_binding_sha256,
        "base_provider_artifact_sha256": _sha(
            expected_endpoint_receipt.get("base_provider_artifact_sha256"),
            label="V20o signed_continuum endpoint base provider",
        ),
        "proposal_provider_artifact_sha256": _sha(
            expected_endpoint_receipt.get("proposal_provider_artifact_sha256"),
            label="V20o signed_continuum endpoint proposal provider",
        ),
        "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
        "transfer_evidence_sha256": expected_transfer,
        "source_direction_sha256": fisher_soft_polarity_signed_continuum_direction_sha256(
            _v20g._eta_tensor(direction)
        ),
        "compiled_direction_sha256": fisher_soft_polarity_signed_continuum_direction_sha256(
            _v20g._eta_tensor(compiled_direction)
        ),
        "compiled_direction_sign": compiled_sign,
        "source_radius": response[0],
        "source_shrink_mass": response[1],
        "source_polarity_bias": response[2],
        "signed_scalar": signed_continuum,
        "compiled_mix": compiled_mix,
    }
    for key, expected in expected_bindings.items():
        if validated.payload.get(key) != expected:
            raise ValueError(f"V20o signed_continuum provider {key} differs")
    if (
        receipt.get("source_direction_sha256")
        != validated.payload.get("source_direction_sha256")
        or receipt.get("compiled_direction_sha256")
        != validated.payload.get("compiled_direction_sha256")
    ):
        raise ValueError("V20o signed continuum direction hash receipt differs")
    receipt_accounting = tuple(
        _strict_receipt_integer(
            receipt, key, label="V20o signed_continuum provider receipt"
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
        raise ValueError("V20o signed_continuum provider accounting differs")


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
            AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider,
        )
        else provider
    )
    return _sha(
        runtime_provider.artifact_sha256,
        label="V20o runtime provider artifact",
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
                AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider,
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

    outer = _identifier(outer_family_id, label="V20o inner outer family")
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
        raise RuntimeError("V20o inner family geometry differs")

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
            label="V20o inner selected reflection variant",
        )
        held_records = tuple(
            record for record in ordered if record.sequence.family_id == inner
        )
        if len(held_records) != _PROMPTS_PER_FAMILY:
            raise RuntimeError("V20o inner-held prompt geometry differs")

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
        raise RuntimeError("V20o inner provider artifacts are not all distinct")
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
        raise ValueError("V20o response selection requires seven inner OOF families")
    outer_ids = {
        _identifier(
            inner_evidence_by_family[family].get("outer_held_family_id"),
            label="V20o response selection outer family",
        )
        for family in families
    }
    if len(outer_ids) != 1 or next(iter(outer_ids)) in families:
        raise ValueError("V20o response selection outer family geometry differs")
    outer = next(iter(outer_ids))
    all_families = tuple(sorted((*families, outer)))
    objectives_by_response: dict[str, float] = {}
    aggregate_artifacts: dict[str, str] = {}
    objectives_by_family_and_response: dict[str, dict[str, float]] = {}
    for family in families:
        raw = _mapping(
            inner_evidence_by_family[family].get("objective_by_response"),
            label="V20o inner objective ladder",
        )
        if set(raw) != set(_RESPONSE_KEYS):
            raise ValueError("V20o inner objective response geometry differs")
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
    outer = _identifier(outer_family_id, label="V20o inner-response outer family")
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
        raise PermissionError("V20o inner freeze barrier is not satisfied")

    inner_evidence: dict[str, dict[str, object]] = {}
    gradient_evidence = _mapping(
        _mapping(
            authenticated_v20g_fold.get("fit_training_evidence"),
            label="V20o inherited fit evidence",
        ).get("gradient_evidence"),
        label="V20o inherited gradient evidence",
    )
    eta_zero_objectives = _mapping(
        gradient_evidence.get("eta_zero_objectives_by_family"),
        label="V20o inherited eta-zero objectives",
    )
    eta_zero_h4 = _mapping(
        gradient_evidence.get("post_cast_h4_sha256s"),
        label="V20o inherited eta-zero H4 hashes",
    )
    eta_zero_logits = _mapping(
        gradient_evidence.get("supervised_full_vocab_logits_sha256s"),
        label="V20o inherited eta-zero logits hashes",
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
                raise RuntimeError("V20o inner score family geometry differs")
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
            label="V20o inner-held capability",
        )
        zero = evidence_by_response[_response_key((0.0, 0.0, 0.0))]
        zero_objectives = _mapping(
            zero.get("objectives_by_example"),
            label="V20o inner eta-zero objectives",
        )
        zero_h4 = _mapping(
            zero.get("post_cast_h4_sha256s"),
            label="V20o inner eta-zero H4 hashes",
        )
        zero_logits = _mapping(
            zero.get("supervised_full_vocab_logits_sha256s"),
            label="V20o inner eta-zero logits hashes",
        )
        expected_zero_objectives = _mapping(
            eta_zero_objectives.get(inner),
            label="V20o inherited family eta-zero objectives",
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
            raise RuntimeError("V20o inner eta-zero output anchor differs from V20g")
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


def _fit_inner_signed_continuum(
    context: object,
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Reproduce V20m, then fit the exact -1/0/+1 signed continuum."""

    outer = _identifier(
        outer_family_id, label="V20o signed-continuum outer family"
    )
    v20m_inner, response_selection = _fit_inner_response(
        context,
        endpoint,
        source_direction_receipt,
        teacher_vault,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
    )
    inherited_selection = _mapping(
        authenticated_v20m_fold.get("response_selection_receipt"),
        label="V20o authenticated V20m response selection",
    )
    for key in (
        "objectives_by_inner_family_and_response",
        "family_equal_objective_by_response",
        "simplex_response_selection_receipt",
        "selected_response",
    ):
        if _v14._canonical_json_bytes(response_selection.get(key)) != (
            _v14._canonical_json_bytes(inherited_selection.get(key))
        ):
            raise RuntimeError(
                f"V20o live V20m response selection {key} reproduction differs"
            )

    inherited_inner = _mapping(
        _mapping(
            authenticated_v20m_fold.get("inner_receipt"),
            label="V20o authenticated V20m inner receipt",
        ).get("inner_evidence_by_family"),
        label="V20o authenticated V20m inner evidence",
    )
    live_inner = _mapping(
        v20m_inner.get("inner_evidence_by_family"),
        label="V20o live V20m inner evidence",
    )
    if set(live_inner) != set(inherited_inner):
        raise RuntimeError("V20o live V20m inner family geometry differs")
    for family in sorted(live_inner):
        live_responses = _mapping(
            _mapping(live_inner[family], label="V20o live inner family").get(
                "response_evidence"
            ),
            label="V20o live V20m response evidence",
        )
        inherited_responses = _mapping(
            _mapping(
                inherited_inner[family], label="V20o authenticated inner family"
            ).get("response_evidence"),
            label="V20o authenticated V20m response evidence",
        )
        if set(live_responses) != set(inherited_responses):
            raise RuntimeError("V20o live V20m response geometry differs")
        for response_key in sorted(live_responses):
            live = _mapping(
                live_responses[response_key], label="V20o live response evidence"
            )
            inherited = _mapping(
                inherited_responses[response_key],
                label="V20o authenticated response evidence",
            )
            for evidence_key in (
                "objectives_by_example",
                "post_cast_h4_sha256s",
                "supervised_full_vocab_logits_sha256s",
            ):
                if _v14._canonical_json_bytes(live.get(evidence_key)) != (
                    _v14._canonical_json_bytes(inherited.get(evidence_key))
                ):
                    raise RuntimeError(
                        "V20o live V20m inner exact-output reproduction differs"
                    )

    response = _response_tuple(response_selection["selected_response"])
    response_key = _response_key(response)
    training = _v20b._ordered_records(endpoint.training_records)
    families = tuple(sorted(live_inner))
    if len(families) != _INNER_FAMILY_COUNT or outer in families:
        raise RuntimeError("V20o signed-continuum inner family geometry differs")
    fits: dict[str, Mapping[str, object]] = {}
    directions: dict[str, tuple[float, float, float, float]] = {}
    held_by_family: dict[str, tuple[object, ...]] = {}
    for inner in families:
        fit = _mapping(
            _mapping(live_inner[inner], label="V20o inner evidence").get(
                "reflection_fit_receipt"
            ),
            label="V20o inner reflection fit",
        )
        fits[inner] = fit
        directions[inner] = _selected_direction(fit)
        held_by_family[inner] = _v20b._ordered_records(
            tuple(
                record
                for record in training
                if record.sequence.family_id == inner
            )
        )
        if len(held_by_family[inner]) != _PROMPTS_PER_FAMILY:
            raise RuntimeError("V20o inner prompt geometry differs")

    missing_anchor_values = (-1.0, 0.0)

    def missing_anchor_id(value: float) -> str:
        if value == -1.0:
            return "signed_minus_one"
        if value == 0.0:
            return "signed_zero"
        raise ValueError("V20o missing anchor must be -1 or 0")

    def missing_anchor_role(value: float) -> str:
        if value == -1.0:
            return "inner_signed_continuum_mirror_anchor"
        if value == 0.0:
            return "inner_signed_continuum_fixed_plus_anchor"
        raise ValueError("V20o missing anchor must be -1 or 0")

    anchor_providers: dict[
        str,
        dict[
            float,
            AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider,
        ],
    ] = {family: {} for family in families}
    anchor_traces: dict[str, dict[float, dict[str, object]]] = {
        family: {} for family in families
    }
    anchor_receipts: dict[str, dict[float, dict[str, object]]] = {
        family: {} for family in families
    }
    anchor_seeds: dict[str, dict[float, str]] = {
        family: {} for family in families
    }
    for inner in families:
        fit = fits[inner]
        for signed_scalar in missing_anchor_values:
            role = missing_anchor_role(signed_scalar)
            provider, seed = _materialize_signed_continuum_provider(
                endpoint,
                direction=directions[inner],
                direction_artifact_sha256=_sha(
                    fit.get("selected_variant_artifact_sha256"),
                    label="V20o anchor selected direction artifact",
                ),
                reflection_fit_sha256=_sha(
                    fit.get("artifact_sha256"),
                    label="V20o anchor reflection fit artifact",
                ),
                response=response,
                signed_scalar=signed_scalar,
                outer_family_id=outer,
                inner_family_id=inner,
                role=role,
            )
            anchor_providers[inner][signed_scalar] = provider
            anchor_seeds[inner][signed_scalar] = seed
            anchor_traces[inner][signed_scalar] = _provider_trace(
                provider, held_by_family[inner], role=role
            )
            anchor_receipts[inner][signed_scalar] = (
                _signed_continuum_provider_receipt(
                    provider,
                    role=role,
                    response=response,
                    signed_scalar=signed_scalar,
                    direction=directions[inner],
                )
            )
    if len(
        {
            provider.artifact_sha256
            for providers in anchor_providers.values()
            for provider in providers.values()
        }
    ) != 2 * _INNER_FAMILY_COUNT:
        raise RuntimeError("V20o missing-anchor artifacts are not all distinct")
    missing_anchor_manifest = _hashed(
        {
            "stage": "missing_signed_anchors",
            "outer_held_family_id": outer,
            "inner_family_order": families,
            "source_response": response,
            "missing_anchor_values": missing_anchor_values,
            "provider_artifact_sha256s_by_inner_family_and_anchor": {
                family: {
                    missing_anchor_id(value): anchor_providers[family][
                        value
                    ].artifact_sha256
                    for value in missing_anchor_values
                }
                for family in families
            },
            "runtime_provider_artifact_sha256s_by_inner_family_and_anchor": {
                family: {
                    missing_anchor_id(value): _runtime_provider_artifact_sha256(
                        anchor_providers[family][value]
                    )
                    for value in missing_anchor_values
                }
                for family in families
            },
            "provider_transfer_evidence_sha256s_by_inner_family_and_anchor": {
                family: {
                    missing_anchor_id(value): anchor_seeds[family][value]
                    for value in missing_anchor_values
                }
                for family in families
            },
            "provider_receipts_by_inner_family_and_anchor": {
                family: {
                    missing_anchor_id(value): anchor_receipts[family][value]
                    for value in missing_anchor_values
                }
                for family in families
            },
            "trace_sha256s_by_inner_family_and_anchor": {
                family: {
                    missing_anchor_id(value): anchor_traces[family][value][
                        "artifact_sha256"
                    ]
                    for value in missing_anchor_values
                }
                for family in families
            },
            "all_fourteen_missing_anchor_providers_and_traces_frozen_before_"
            "any_anchor_capability": True,
            "anchor_capability_count_at_freeze": 0,
            "anchor_objectives_or_teacher_rows_used_at_freeze": False,
            "outer_held_family_used": False,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=_INNER_MANIFEST_DOMAIN,
    )

    missing_anchor_evidence: dict[str, dict[str, dict[str, object]]] = {}
    missing_anchor_objectives: dict[str, dict[str, float]] = {}
    for inner in families:
        held = held_by_family[inner]
        capability = teacher_vault.capability(
            tuple(record.sequence.example_id for record in held),
            held_family_id=outer,
        )
        runs: dict[
            str,
            tuple[
                float,
                dict[str, float],
                dict[str, str],
                dict[str, str],
                dict[str, str],
            ],
        ] = {}
        for signed_scalar in missing_anchor_values:
            anchor_id = missing_anchor_id(signed_scalar)
            role = missing_anchor_role(signed_scalar)
            provider = anchor_providers[inner][signed_scalar]
            score_seed = _v14._sha256(
                {
                    "missing_anchor_manifest_sha256": (
                        missing_anchor_manifest["artifact_sha256"]
                    ),
                    "outer_held_family_id": outer,
                    "inner_held_family_id": inner,
                    "anchor_id": anchor_id,
                    "signed_scalar": signed_scalar,
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "runtime_provider_artifact_sha256": (
                        _runtime_provider_artifact_sha256(provider)
                    ),
                    "all_fourteen_missing_anchor_providers_frozen": True,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
            objectives, h4_hashes, logits_hashes, execution_hashes = (
                _score_exact_provider(
                    context,
                    held,
                    capability,
                    provider=provider,
                    phase="inner_signed_continuum_missing_anchor_score",
                    outer_family_id=outer,
                    inner_family_id=inner,
                    role=role,
                    evidence_sha256=score_seed,
                    domain=_INNER_EXECUTION_DOMAIN,
                )
            )
            macro, family_scores = _v19._family_equal_mean(objectives, held)
            if set(family_scores) != {inner}:
                raise RuntimeError(
                    "V20o missing-anchor score family geometry differs"
                )
            runs[anchor_id] = (
                macro,
                objectives,
                h4_hashes,
                logits_hashes,
                execution_hashes,
            )
        capability_receipt = capability.receipt()
        _v20b._validate_capability_receipt(
            capability_receipt,
            expected_example_ids=tuple(
                record.sequence.example_id for record in held
            ),
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=2,
            label="V20o missing-anchor capability",
        )
        missing_anchor_evidence[inner] = {}
        missing_anchor_objectives[inner] = {}
        for signed_scalar in missing_anchor_values:
            anchor_id = missing_anchor_id(signed_scalar)
            role = missing_anchor_role(signed_scalar)
            provider = anchor_providers[inner][signed_scalar]
            macro, objectives, h4_hashes, logits_hashes, execution_hashes = (
                runs[anchor_id]
            )
            missing_anchor_objectives[inner][anchor_id] = macro
            missing_anchor_evidence[inner][anchor_id] = _hashed(
                {
                    "stage": "missing_signed_anchor",
                    "outer_held_family_id": outer,
                    "inner_held_family_id": inner,
                    "anchor_id": anchor_id,
                    "role": role,
                    "signed_scalar": signed_scalar,
                    "signed_scalar_hex": signed_scalar.hex(),
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "runtime_provider_artifact_sha256": (
                        _runtime_provider_artifact_sha256(provider)
                    ),
                    "lineage_wrapper_not_inference_executor": True,
                    "manifest_sha256": missing_anchor_manifest[
                        "artifact_sha256"
                    ],
                    "response_trace": anchor_traces[inner][signed_scalar],
                    "objective": macro,
                    "objectives_by_example": dict(sorted(objectives.items())),
                    "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                    "supervised_full_vocab_logits_sha256s": dict(
                        sorted(logits_hashes.items())
                    ),
                    "execution_sha256s": dict(
                        sorted(execution_hashes.items())
                    ),
                    "capability_receipt": capability_receipt,
                    "all_fourteen_missing_anchor_providers_frozen_before_score": True,
                    "outer_family_absent_from_fit_and_score": True,
                    "exact_execution": True,
                    "finite": True,
                    "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )

    anchor_objectives: dict[str, dict[str, float]] = {}
    plus_one_evidence: dict[str, dict[str, object]] = {}
    for family in families:
        response_evidence = _mapping(
            _mapping(live_inner[family], label="V20o live inner family").get(
                "response_evidence"
            ),
            label="V20o live response evidence",
        )
        plus_one = dict(
            _mapping(
                response_evidence.get(response_key),
                label="V20o selected V20m response evidence",
            )
        )
        plus_one_evidence[family] = plus_one
        anchor_objectives[family] = {
            "signed_minus_one": missing_anchor_objectives[family][
                "signed_minus_one"
            ],
            "signed_zero": missing_anchor_objectives[family]["signed_zero"],
            "signed_plus_one": float(plus_one["objective"]),
        }
    all_families = tuple(sorted((*families, outer)))
    anchor_receipt = (
        _signed_continuum_fit.build_soft_polarity_signed_continuum_anchor_receipt(
            all_development_family_ids=all_families,
            outer_held_family_id=outer,
            exact_anchor_objectives_by_family_and_anchor=anchor_objectives,
        )
    )
    proposal_receipt = (
        _signed_continuum_fit
        .build_soft_polarity_signed_continuum_quadratic_proposal_receipt(
            anchor_receipt=anchor_receipt
        )
    )
    proposed_signed_scalar = float(
        proposal_receipt["proposed_signed_scalar"]
    )
    if not -1.0 <= proposed_signed_scalar <= 1.0:
        raise RuntimeError("V20o core fitter proposed a scalar outside [-1,1]")

    vertex_providers: dict[
        str, AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider
    ] = {}
    vertex_traces: dict[str, dict[str, object]] = {}
    vertex_receipts: dict[str, dict[str, object]] = {}
    vertex_seeds: dict[str, str] = {}
    for inner in families:
        fit = fits[inner]
        provider, seed = _materialize_signed_continuum_provider(
            endpoint,
            direction=directions[inner],
            direction_artifact_sha256=_sha(
                fit.get("selected_variant_artifact_sha256"),
                label="V20o vertex direction artifact",
            ),
            reflection_fit_sha256=_sha(
                fit.get("artifact_sha256"),
                label="V20o vertex reflection fit",
            ),
            response=response,
            signed_scalar=proposed_signed_scalar,
            outer_family_id=outer,
            inner_family_id=inner,
            role="inner_signed_continuum_vertex",
        )
        vertex_providers[inner] = provider
        vertex_seeds[inner] = seed
        vertex_traces[inner] = _provider_trace(
            provider,
            held_by_family[inner],
            role="inner_signed_continuum_vertex",
        )
        vertex_receipts[inner] = _signed_continuum_provider_receipt(
            provider,
            role="inner_signed_continuum_vertex",
            response=response,
            signed_scalar=proposed_signed_scalar,
            direction=directions[inner],
        )
    if len(
        {provider.artifact_sha256 for provider in vertex_providers.values()}
    ) != _INNER_FAMILY_COUNT:
        raise RuntimeError("V20o vertex-provider artifacts are not distinct")
    vertex_manifest = _hashed(
        {
            "stage": "quadratic_vertex",
            "outer_held_family_id": outer,
            "inner_family_order": families,
            "source_response": response,
            "signed_scalar": proposed_signed_scalar,
            "signed_scalar_hex": proposed_signed_scalar.hex(),
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
        held = held_by_family[inner]
        capability = teacher_vault.capability(
            tuple(record.sequence.example_id for record in held),
            held_family_id=outer,
        )
        provider = vertex_providers[inner]
        score_seed = _v14._sha256(
            {
                "vertex_manifest_sha256": vertex_manifest["artifact_sha256"],
                "outer_held_family_id": outer,
                "inner_held_family_id": inner,
                "provider_artifact_sha256": provider.artifact_sha256,
                "runtime_provider_artifact_sha256": (
                    _runtime_provider_artifact_sha256(provider)
                ),
                "all_seven_vertex_providers_frozen": True,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
        objectives, h4_hashes, logits_hashes, execution_hashes = (
            _score_exact_provider(
                context,
                held,
                capability,
                provider=provider,
                phase="inner_signed_continuum_vertex_score",
                outer_family_id=outer,
                inner_family_id=inner,
                role="inner_signed_continuum_vertex",
                evidence_sha256=score_seed,
                domain=_INNER_EXECUTION_DOMAIN,
            )
        )
        macro, family_scores = _v19._family_equal_mean(objectives, held)
        if set(family_scores) != {inner}:
            raise RuntimeError("V20o vertex score family geometry differs")
        vertex_objectives[inner] = macro
        endpoint_anchor = True
        if proposed_signed_scalar in _SIGNED_CONTINUUM_ANCHOR_VALUES:
            expected = (
                plus_one_evidence[inner]
                if proposed_signed_scalar == 1.0
                else missing_anchor_evidence[inner][
                    missing_anchor_id(proposed_signed_scalar)
                ]
            )
            endpoint_anchor = (
                macro == float(expected["objective"])
                and dict(sorted(objectives.items()))
                == dict(
                    _mapping(
                        expected.get("objectives_by_example"),
                        label="V20o endpoint anchor objectives",
                    )
                )
                and dict(sorted(h4_hashes.items()))
                == dict(
                    _mapping(
                        expected.get("post_cast_h4_sha256s"),
                        label="V20o endpoint anchor H4 hashes",
                    )
                )
                and dict(sorted(logits_hashes.items()))
                == dict(
                    _mapping(
                        expected.get(
                            "supervised_full_vocab_logits_sha256s"
                        ),
                        label="V20o endpoint anchor logits hashes",
                    )
                )
            )
            if not endpoint_anchor:
                raise RuntimeError(
                    "V20o endpoint vertex failed exact anchor reproduction"
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
            label="V20o vertex-score capability",
        )
        vertex_evidence[inner] = _hashed(
            {
                "stage": "quadratic_vertex",
                "outer_held_family_id": outer,
                "inner_held_family_id": inner,
                "signed_scalar": proposed_signed_scalar,
                "provider_artifact_sha256": provider.artifact_sha256,
                "runtime_provider_artifact_sha256": (
                    _runtime_provider_artifact_sha256(provider)
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
                "endpoint_vertex_exact_anchor": endpoint_anchor,
                "outer_family_absent_from_fit_and_score": True,
                "exact_execution": True,
                "finite": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )

    vertex_score_receipt = (
        _signed_continuum_fit
        .build_soft_polarity_signed_continuum_vertex_score_receipt(
            anchor_receipt=anchor_receipt,
            proposal_receipt=proposal_receipt,
            exact_vertex_objectives_by_family=vertex_objectives,
        )
    )
    selection_receipt = (
        _signed_continuum_fit
        .build_soft_polarity_signed_continuum_selection_receipt(
            anchor_receipt=anchor_receipt,
            proposal_receipt=proposal_receipt,
            vertex_score_receipt=vertex_score_receipt,
        )
    )
    selected_signed_scalar = float(selection_receipt["selected_signed_scalar"])
    signed_selection = _hashed(
        {
            "outer_held_family_id": outer,
            "source_response": response,
            "v20m_response_selection_receipt_sha256": response_selection[
                "artifact_sha256"
            ],
            "missing_anchor_provider_manifest": missing_anchor_manifest,
            "missing_anchor_evidence_by_family_and_anchor": (
                missing_anchor_evidence
            ),
            "reused_v20m_plus_one_evidence_by_family": plus_one_evidence,
            "anchor_objectives_by_family_and_anchor": anchor_objectives,
            "core_anchor_receipt": anchor_receipt,
            "core_quadratic_proposal_receipt": proposal_receipt,
            "vertex_provider_manifest": vertex_manifest,
            "vertex_evidence_by_family": vertex_evidence,
            "core_vertex_score_receipt": vertex_score_receipt,
            "core_selection_receipt": selection_receipt,
            "endpoint_vertex_exact_anchor_by_family": (
                endpoint_vertex_anchor_by_family
            ),
            "all_endpoint_vertex_exact_anchors_passed": all(
                endpoint_vertex_anchor_by_family.values()
            ),
            "selected_signed_scalar": selected_signed_scalar,
            "selected_signed_scalar_hex": selected_signed_scalar.hex(),
            "selected_signed_scalar_interior": (
                (-1.0 < selected_signed_scalar < 0.0)
                or (0.0 < selected_signed_scalar < 1.0)
            ),
            "selected_signed_scalar_negative": selected_signed_scalar < 0.0,
            "selected_signed_scalar_positive": selected_signed_scalar > 0.0,
            "all_fourteen_missing_anchor_providers_frozen_before_any_anchor_"
            "score": True,
            "proposal_frozen_before_any_vertex_provider_or_score": True,
            "all_vertex_providers_frozen_before_any_vertex_score": True,
            "exact_additional_inner_execution_count": (
                3 * _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY
            ),
            "outer_held_family_used_for_fit_or_selection": False,
            "final_refit_or_calibration_b_used": False,
            "raw_provider_prompt_token_logit_h4_or_teacher_tensors_serialized": False,
        },
        domain=_SIGNED_CONTINUUM_SELECTION_DOMAIN,
    )
    return v20m_inner, response_selection, signed_selection


def _matched_v20l_boundary_response(
    authenticated_v20l_fold: Mapping[str, object], *, outer_family_id: str
) -> tuple[float, float, float]:
    """Map the precommitted V20l winner into the simplex boundary exactly."""

    fold = _mapping(
        authenticated_v20l_fold.get("fold_receipt"),
        label="V20o authenticated V20l fold receipt",
    )
    outer = _identifier(outer_family_id, label="V20o V20l boundary family")
    if fold.get("outer_held_family_id") != outer:
        raise ValueError("V20o V20l boundary family differs")
    radius, signed_mix = _v20l._response_pair(fold.get("selected_response"))
    return _simplex_parameters((radius, abs(signed_mix), signed_mix))


def _freeze_outer_providers(
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    outer_reflection_fit: Mapping[str, object],
    held_records: Sequence[object],
    *,
    selected_response: tuple[float, float, float],
    selected_signed_scalar: float,
    outer_family_id: str,
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    outer = _identifier(outer_family_id, label="V20o outer provider family")
    response = _response_tuple(selected_response)
    signed_continuum = float(selected_signed_scalar)
    if not math.isfinite(signed_continuum) or not -1.0 <= signed_continuum <= 1.0:
        raise ValueError("V20o selected signed scalar must be inside [-1,1]")
    linear_response = (response[0], 0.0, 0.0)
    v20l_boundary_response = _matched_v20l_boundary_response(
        authenticated_v20l_fold, outer_family_id=outer
    )
    reflected = _selected_direction(outer_reflection_fit)
    unreflected = _unreflected_direction(source_direction_receipt)
    mirror = tuple(-item for item in reflected)
    selected_variant_artifact = _sha(
        outer_reflection_fit.get("selected_variant_artifact_sha256"),
        label="V20o outer reflection variant",
    )
    source_direction_artifact = _sha(
        source_direction_receipt.get("artifact_sha256"),
        label="V20o outer unreflected direction",
    )
    reflection_fit_artifact = _sha(
        outer_reflection_fit.get("artifact_sha256"),
        label="V20o outer reflection fit",
    )

    reflected_provider, reflected_seed = _materialize_signed_continuum_provider(
        endpoint,
        direction=reflected,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        signed_scalar=signed_continuum,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_signed_continuum_reflected",
    )
    unreflected_provider, unreflected_seed = _materialize_signed_continuum_provider(
        endpoint,
        direction=unreflected,
        direction_artifact_sha256=source_direction_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        signed_scalar=signed_continuum,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_same_simplex_response_unreflected",
    )
    mirror_provider, mirror_seed = _materialize_provider(
        endpoint,
        direction=mirror,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
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
        "signed_continuum_reflected": reflected_provider,
        "simplex_response_reflected_exact_mirror": mirror_provider,
        "matched_v20m_simplex_reflected": matched_v20m_provider,
    }
    if tuple(providers) != _ARMS or len(
        {provider.artifact_sha256 for provider in providers.values()}
    ) != len(_ARMS):
        raise RuntimeError("V20o outer provider arm artifacts are not distinct")

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
        "same_simplex_response_unreflected": _signed_continuum_provider_receipt(
            providers["same_simplex_response_unreflected"],
            role="same_simplex_response_unreflected",
            response=response,
            signed_scalar=signed_continuum,
            direction=unreflected,
        ),
        "signed_continuum_reflected": _signed_continuum_provider_receipt(
            providers["signed_continuum_reflected"],
            role="signed_continuum_reflected",
            response=response,
            signed_scalar=signed_continuum,
            direction=reflected,
        ),
        "simplex_response_reflected_exact_mirror": _provider_receipt(
            providers["simplex_response_reflected_exact_mirror"],
            role="simplex_response_reflected_exact_mirror",
            response=response,
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
            "selected_signed_scalar": signed_continuum,
            "selected_signed_scalar_hex": signed_continuum.hex(),
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
                "signed_continuum_reflected": reflected_seed,
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
    selected_signed_scalar: float,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    outer = _identifier(outer_family_id, label="V20o held outer family")
    held = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id == outer)
    )
    if len(held) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20o outer-held prompt geometry differs")
    providers, manifest, traces = _freeze_outer_providers(
        endpoint,
        source_direction_receipt,
        outer_reflection_fit,
        held,
        selected_response=selected_response,
        selected_signed_scalar=selected_signed_scalar,
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
        raise PermissionError("V20o outer freeze barrier is not satisfied")

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
                    AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider,
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
            raise RuntimeError("V20o outer score family geometry differs")
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
                    AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider,
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
        label="V20o outer-held capability",
    )
    inherited_arms = _mapping(
        _mapping(
            authenticated_v20g_fold.get("held_evidence"),
            label="V20o inherited V20g held evidence",
        ).get("arm_evidence"),
        label="V20o inherited V20g held arms",
    )
    control_anchors: dict[str, bool] = {}
    for arm in ("base", "fixed_plus", "fixed_minus"):
        inherited = _mapping(
            inherited_arms.get(arm), label=f"V20o inherited V20g {arm} arm"
        )
        current = evidence_by_arm[arm]
        control_anchors[arm] = (
            float(current["objective"]) == float(inherited["objective"])
            and dict(
                _mapping(
                    current.get("objectives_by_example"),
                    label=f"V20o current {arm} objectives",
                )
            )
            == dict(
                _mapping(
                    inherited.get("objectives_by_example"),
                    label=f"V20o inherited {arm} objectives",
                )
            )
            and dict(
                _mapping(
                    current.get("post_cast_h4_sha256s"),
                    label=f"V20o current {arm} H4 hashes",
                )
            )
            == dict(
                _mapping(
                    inherited.get("post_cast_h4_sha256s"),
                    label=f"V20o inherited {arm} H4 hashes",
                )
            )
            and dict(
                _mapping(
                    current.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20o current {arm} logits hashes",
                )
            )
            == dict(
                _mapping(
                    inherited.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20o inherited {arm} logits hashes",
                )
            )
        )
    if not all(control_anchors.values()):
        raise RuntimeError("V20o outer control output anchor differs from V20g")
    inherited_v20l_arm = _mapping(
        _mapping(
            _mapping(
                authenticated_v20l_fold.get("held_evidence"),
                label="V20o inherited V20l held evidence",
            ).get("arm_evidence"),
            label="V20o inherited V20l held arms",
        ).get("signed_stack_reflected"),
        label="V20o inherited V20l selected boundary arm",
    )
    current_boundary = evidence_by_arm["matched_v20l_boundary_reflected"]
    v20l_boundary_anchor = (
        float(current_boundary["objective"])
        == float(inherited_v20l_arm["objective"])
        and dict(
            _mapping(
                current_boundary.get("objectives_by_example"),
                label="V20o current V20l boundary objectives",
            )
        )
        == dict(
            _mapping(
                inherited_v20l_arm.get("objectives_by_example"),
                label="V20o inherited V20l boundary objectives",
            )
        )
        and dict(
            _mapping(
                current_boundary.get("post_cast_h4_sha256s"),
                label="V20o current V20l boundary H4 hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20l_arm.get("post_cast_h4_sha256s"),
                label="V20o inherited V20l boundary H4 hashes",
            )
        )
        and dict(
            _mapping(
                current_boundary.get("supervised_full_vocab_logits_sha256s"),
                label="V20o current V20l boundary logits hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20l_arm.get("supervised_full_vocab_logits_sha256s"),
                label="V20o inherited V20l boundary logits hashes",
            )
        )
    )
    if not v20l_boundary_anchor:
        raise RuntimeError(
            "V20o matched boundary failed exact V20l objective/output reproduction"
        )
    inherited_v20m_arm = _mapping(
        _mapping(
            _mapping(
                authenticated_v20m_fold.get("held_evidence"),
                label="V20o inherited V20m held evidence",
            ).get("arm_evidence"),
            label="V20o inherited V20m held arms",
        ).get("simplex_response_reflected"),
        label="V20o inherited V20m selected simplex arm",
    )
    current_v20m = evidence_by_arm["matched_v20m_simplex_reflected"]
    v20m_anchor = (
        float(current_v20m["objective"]) == float(inherited_v20m_arm["objective"])
        and dict(
            _mapping(
                current_v20m.get("objectives_by_example"),
                label="V20o current V20m objectives",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_arm.get("objectives_by_example"),
                label="V20o inherited V20m objectives",
            )
        )
        and dict(
            _mapping(
                current_v20m.get("post_cast_h4_sha256s"),
                label="V20o current V20m H4 hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_arm.get("post_cast_h4_sha256s"),
                label="V20o inherited V20m H4 hashes",
            )
        )
        and dict(
            _mapping(
                current_v20m.get("supervised_full_vocab_logits_sha256s"),
                label="V20o current V20m logits hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_arm.get(
                    "supervised_full_vocab_logits_sha256s"
                ),
                label="V20o inherited V20m logits hashes",
            )
        )
    )
    if not v20m_anchor:
        raise RuntimeError(
            "V20o matched V20m arm failed exact objective/output reproduction"
        )
    inherited_v20m_mirror = _mapping(
        _mapping(
            _mapping(
                authenticated_v20m_fold.get("held_evidence"),
                label="V20o inherited V20m held evidence",
            ).get("arm_evidence"),
            label="V20o inherited V20m held arms",
        ).get("simplex_response_reflected_exact_mirror"),
        label="V20o inherited V20m exact mirror arm",
    )
    current_mirror = evidence_by_arm[
        "simplex_response_reflected_exact_mirror"
    ]
    v20m_mirror_anchor = (
        float(current_mirror["objective"])
        == float(inherited_v20m_mirror["objective"])
        and dict(
            _mapping(
                current_mirror.get("objectives_by_example"),
                label="V20o current V20m mirror objectives",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_mirror.get("objectives_by_example"),
                label="V20o inherited V20m mirror objectives",
            )
        )
        and dict(
            _mapping(
                current_mirror.get("post_cast_h4_sha256s"),
                label="V20o current V20m mirror H4 hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_mirror.get("post_cast_h4_sha256s"),
                label="V20o inherited V20m mirror H4 hashes",
            )
        )
        and dict(
            _mapping(
                current_mirror.get("supervised_full_vocab_logits_sha256s"),
                label="V20o current V20m mirror logits hashes",
            )
        )
        == dict(
            _mapping(
                inherited_v20m_mirror.get(
                    "supervised_full_vocab_logits_sha256s"
                ),
                label="V20o inherited V20m mirror logits hashes",
            )
        )
    )
    if not v20m_mirror_anchor:
        raise RuntimeError(
            "V20o exact mirror failed exact V20m objective/output reproduction"
        )
    base_logits = _mapping(
        evidence_by_arm["base"].get("supervised_full_vocab_logits_sha256s"),
        label="V20o base output hashes",
    )
    candidate_logits = _mapping(
        evidence_by_arm[_PRIMARY_ARM].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20o candidate output hashes",
    )
    candidate_changed = any(
        candidate_logits[example] != base_logits[example] for example in base_logits
    )
    boundary_logits = _mapping(
        evidence_by_arm["matched_v20l_boundary_reflected"].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20o matched V20l boundary output hashes",
    )
    candidate_changed_from_v20l_boundary = any(
        candidate_logits[example] != boundary_logits[example]
        for example in candidate_logits
    )
    v20m_logits = _mapping(
        evidence_by_arm["matched_v20m_simplex_reflected"].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20o matched V20m output hashes",
    )
    linear_logits = _mapping(
        evidence_by_arm["matched_linear_reflected"].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20o matched linear output hashes",
    )
    candidate_changed_from_v20m = any(
        candidate_logits[example] != v20m_logits[example]
        for example in candidate_logits
    )
    candidate_changed_from_linear = any(
        candidate_logits[example] != linear_logits[example]
        for example in candidate_logits
    )
    mirror_logits = _mapping(
        evidence_by_arm["simplex_response_reflected_exact_mirror"].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20o exact mirror output hashes",
    )
    fixed_plus_logits = _mapping(
        evidence_by_arm["fixed_plus"].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20o fixed-plus output hashes",
    )
    candidate_changed_from_mirror = any(
        candidate_logits[example] != mirror_logits[example]
        for example in candidate_logits
    )
    candidate_changed_from_fixed_plus = any(
        candidate_logits[example] != fixed_plus_logits[example]
        for example in candidate_logits
    )
    endpoint_anchor_arm_by_scalar = {
        -1.0: "simplex_response_reflected_exact_mirror",
        0.0: "fixed_plus",
        1.0: "matched_v20m_simplex_reflected",
    }
    endpoint_anchor_id_by_scalar = {
        -1.0: "signed_minus_one",
        0.0: "signed_zero",
        1.0: "signed_plus_one",
    }
    selected_endpoint_anchor_arm = endpoint_anchor_arm_by_scalar.get(
        float(selected_signed_scalar)
    )
    selected_endpoint_anchor_id = endpoint_anchor_id_by_scalar.get(
        float(selected_signed_scalar)
    )
    selected_endpoint_anchor_applicable = selected_endpoint_anchor_arm is not None
    selected_endpoint_exact_anchor = True
    if selected_endpoint_anchor_arm is not None:
        candidate_endpoint = evidence_by_arm[_PRIMARY_ARM]
        expected_endpoint = evidence_by_arm[selected_endpoint_anchor_arm]
        selected_endpoint_exact_anchor = (
            float(candidate_endpoint["objective"])
            == float(expected_endpoint["objective"])
            and dict(
                _mapping(
                    candidate_endpoint.get("objectives_by_example"),
                    label="V20o candidate endpoint objectives",
                )
            )
            == dict(
                _mapping(
                    expected_endpoint.get("objectives_by_example"),
                    label="V20o expected endpoint objectives",
                )
            )
            and dict(
                _mapping(
                    candidate_endpoint.get("post_cast_h4_sha256s"),
                    label="V20o candidate endpoint H4 hashes",
                )
            )
            == dict(
                _mapping(
                    expected_endpoint.get("post_cast_h4_sha256s"),
                    label="V20o expected endpoint H4 hashes",
                )
            )
            and dict(
                _mapping(
                    candidate_endpoint.get(
                        "supervised_full_vocab_logits_sha256s"
                    ),
                    label="V20o candidate endpoint logits hashes",
                )
            )
            == dict(
                _mapping(
                    expected_endpoint.get(
                        "supervised_full_vocab_logits_sha256s"
                    ),
                    label="V20o expected endpoint logits hashes",
                )
            )
        )
        if not selected_endpoint_exact_anchor:
            raise RuntimeError(
                "V20o selected signed endpoint failed exact anchor reproduction"
            )
    selected_signed_scalar_interior = (
        (-1.0 < float(selected_signed_scalar) < 0.0)
        or (0.0 < float(selected_signed_scalar) < 1.0)
    )
    interior_exact_distinct_from_three_anchors = (
        not selected_signed_scalar_interior
        or (
            candidate_changed_from_mirror
            and candidate_changed_from_fixed_plus
            and candidate_changed_from_v20m
        )
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
            "exact_mirror_v20m_exact_output_anchor_passed": (
                v20m_mirror_anchor
            ),
            "selected_endpoint_exact_anchor_applicable": (
                selected_endpoint_anchor_applicable
            ),
            "selected_endpoint_exact_anchor_id": selected_endpoint_anchor_id,
            "selected_endpoint_exact_anchor_passed": (
                selected_endpoint_exact_anchor
            ),
            "raw_prompts_tokens_logits_h4_or_teacher_rows_serialized": False,
        },
        domain=_OUTER_EXECUTION_DOMAIN,
    )
    fold_receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "selected_response": _response_tuple(selected_response),
            "selected_response_key": _response_key(selected_response),
            "selected_signed_scalar": float(selected_signed_scalar),
            "selected_signed_scalar_hex": float(selected_signed_scalar).hex(),
            "selected_signed_scalar_interior": selected_signed_scalar_interior,
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
            "candidate_exact_execution_changed_from_exact_mirror": (
                candidate_changed_from_mirror
            ),
            "candidate_exact_execution_changed_from_fixed_plus": (
                candidate_changed_from_fixed_plus
            ),
            "interior_candidate_exact_distinct_from_all_three_anchors": (
                interior_exact_distinct_from_three_anchors
            ),
            "selected_endpoint_exact_anchor_applicable": (
                selected_endpoint_anchor_applicable
            ),
            "selected_endpoint_exact_anchor_id": selected_endpoint_anchor_id,
            "selected_endpoint_exact_anchor_passed": (
                selected_endpoint_exact_anchor
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
            "exact_mirror_v20m_exact_output_anchor_passed": (
                v20m_mirror_anchor
            ),
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
    authenticated_v20n_fold: Mapping[str, object],
) -> _FoldLive:
    outer = _identifier(outer_family_id, label="V20o outer family")
    if authenticated_v20n_fold.get("fragment_sha256") != _V20N_FOLD_SHA256S.get(
        outer
    ):
        raise RuntimeError("V20o authenticated V20n outer-fold authority differs")
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
        label="V20o inherited endpoint receipt",
    )
    inherited_evidence = _mapping(
        authenticated_v20g_fold.get("endpoint_evidence"),
        label="V20o inherited endpoint evidence",
    )
    if (
        _v14._canonical_json_bytes(endpoint.receipt)
        != _v14._canonical_json_bytes(inherited_endpoint)
        or _v14._canonical_json_bytes(endpoint.evidence)
        != _v14._canonical_json_bytes(inherited_evidence)
    ):
        raise RuntimeError("V20o reconstructed endpoint differs from pinned V20g")
    fit = _mapping(
        authenticated_v20g_fold.get("fit_receipt"),
        label="V20o inherited V20g fit receipt",
    )
    source_direction = _mapping(
        fit.get("direction_receipt"), label="V20o inherited V20g direction"
    )
    _v20g._core.validate_soft_polarity_direction_receipt(source_direction)
    if source_direction.get("held_family_id") != outer:
        raise RuntimeError("V20o inherited direction held family differs")

    inner_receipt, response_selection, signed_continuum_selection = (
        _fit_inner_signed_continuum(
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
    selected_signed_scalar = float(signed_continuum_selection["selected_signed_scalar"])
    provider_manifest, held_evidence, fold_receipt = _score_outer_arms(
        context,
        endpoint,
        records,
        teacher_vault,
        source_direction,
        outer_reflection_fit,
        selected_response=selected_response,
        selected_signed_scalar=selected_signed_scalar,
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
        signed_continuum_selection=signed_continuum_selection,
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
        "signed_continuum_fit_protocol_sha256",
        "signed_continuum_provider_protocol_sha256",
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
        "v20n_fold_fragment_sha256",
        "outer_held_family_id",
        "endpoint_receipt",
        "endpoint_evidence",
        "inner_receipt",
        "outer_reflection_fit_receipt",
        "response_selection_receipt",
        "signed_continuum_selection_receipt",
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
    authenticated_v20n_fold: Mapping[str, object],
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
        "signed_continuum_fit_protocol_sha256": (
            _signed_continuum_fit.SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256
        ),
        "signed_continuum_provider_protocol_sha256": (
            FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256
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
        "v20n_fold_fragment_sha256": authenticated_v20n_fold[
            "fragment_sha256"
        ],
        "outer_held_family_id": outer_family_id,
        "endpoint_receipt": live.endpoint.receipt,
        "endpoint_evidence": live.endpoint.evidence,
        "inner_receipt": live.inner_receipt,
        "outer_reflection_fit_receipt": live.outer_reflection_fit,
        "response_selection_receipt": live.response_selection,
        "signed_continuum_selection_receipt": live.signed_continuum_selection,
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
        value, domain=_INNER_FIT_DOMAIN, label="V20o inner receipt"
    )
    inherited_v20m_inner = _mapping(
        authenticated_v20m_fold.get("inner_receipt"),
        label="V20o authenticated V20m inner receipt",
    )
    inherited_v20m_inner_evidence = _mapping(
        inherited_v20m_inner.get("inner_evidence_by_family"),
        label="V20o authenticated V20m inner evidence",
    )
    inherited_v20m_selection = _mapping(
        authenticated_v20m_fold.get("response_selection_receipt"),
        label="V20o authenticated V20m response selection",
    )
    endpoint_receipt = _mapping(
        authenticated_v20g_fold.get("endpoint_receipt"),
        label="V20o inherited endpoint receipt",
    )
    inherited_bridge = authenticated_v20g_fold.get("bridge_binding_sha256")
    if inherited_bridge is None:
        inherited_bridge = endpoint_receipt.get("bridge_binding_sha256")
    if inherited_bridge is None:
        inherited_bridge = source_direction.get("bridge_binding_sha256")
    expected_bridge_binding = _sha(
        inherited_bridge,
        label="V20o inherited bridge binding",
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
        raise ValueError("V20o inner receipt boundary differs")
    manifest = _validate_hashed(
        _mapping(
            receipt.get("inner_provider_manifest"),
            label="V20o inner provider manifest",
        ),
        domain=_INNER_MANIFEST_DOMAIN,
        label="V20o inner provider manifest",
    )
    families = tuple(
        _identifier(item, label="V20o inner family")
        for item in _sequence(
            manifest.get("inner_family_order"), label="V20o inner family order"
        )
    )
    source_families = tuple(
        _identifier(item, label="V20o source training family")
        for item in _sequence(
            source_direction.get("training_family_ids"),
            label="V20o source training families",
        )
    )
    training_ids_by_family = _mapping(
        source_direction.get("training_example_ids_by_family"),
        label="V20o source training example ids",
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
        raise ValueError("V20o inner manifest freeze geometry differs")
    if tuple(receipt.get("inner_family_order", ())) != families:
        raise ValueError("V20o inner receipt family order differs")
    if set(inherited_v20m_inner_evidence) != set(families):
        raise ValueError("V20o authenticated V20m inner family geometry differs")
    masked_hashes = _mapping(
        manifest.get("masked_direction_receipt_sha256s_by_inner_family"),
        label="V20o inner masked receipt hashes",
    )
    fit_hashes = _mapping(
        manifest.get("reflection_fit_receipt_sha256s_by_inner_family"),
        label="V20o inner reflection fit hashes",
    )
    variant_hashes = _mapping(
        manifest.get("selected_variant_artifact_sha256s_by_inner_family"),
        label="V20o inner selected variant hashes",
    )
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s_by_inner_family_and_response"),
        label="V20o inner provider hashes",
    )
    provider_receipts = _mapping(
        manifest.get("provider_receipts_by_inner_family_and_response"),
        label="V20o inner provider receipts",
    )
    transfer_hashes = _mapping(
        manifest.get(
            "provider_transfer_evidence_sha256s_by_inner_family_and_response"
        ),
        label="V20o inner provider transfer hashes",
    )
    trace_hashes = _mapping(
        manifest.get("response_trace_sha256s_by_inner_family_and_response"),
        label="V20o inner trace hashes",
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
        raise ValueError("V20o inner manifest family bindings differ")
    gradient_evidence = _mapping(
        _mapping(
            authenticated_v20g_fold.get("fit_training_evidence"),
            label="V20o inherited fit evidence",
        ).get("gradient_evidence"),
        label="V20o inherited gradient evidence",
    )
    inherited_zero_objectives = _mapping(
        gradient_evidence.get("eta_zero_objectives_by_family"),
        label="V20o inherited eta-zero objectives",
    )
    inherited_zero_h4 = _mapping(
        gradient_evidence.get("post_cast_h4_sha256s"),
        label="V20o inherited eta-zero H4 hashes",
    )
    inherited_zero_logits = _mapping(
        gradient_evidence.get("supervised_full_vocab_logits_sha256s"),
        label="V20o inherited eta-zero logits hashes",
    )
    if set(inherited_zero_objectives) != family_set:
        raise ValueError("V20o inherited eta-zero family geometry differs")
    raw_inner = _mapping(
        receipt.get("inner_evidence_by_family"),
        label="V20o inner evidence map",
    )
    if set(raw_inner) != set(families):
        raise ValueError("V20o inner evidence family geometry differs")
    validated_inner: dict[str, Mapping[str, object]] = {}
    for family in families:
        evidence = _validate_hashed(
            _mapping(raw_inner[family], label="V20o inner family evidence"),
            domain=_INNER_EXECUTION_DOMAIN,
            label="V20o inner family evidence",
        )
        masked = _mapping(
            evidence.get("masked_direction_receipt"),
            label="V20o masked direction receipt",
        )
        _reflection.validate_soft_polarity_masked_direction_receipt(
            masked,
            source_direction_receipt=source_direction,
            expected_excluded_training_family_id=family,
        )
        reflection_fit = _mapping(
            evidence.get("reflection_fit_receipt"),
            label="V20o inner reflection fit",
        )
        _reflection.validate_soft_polarity_reflection_fit_receipt(
            reflection_fit, direction_receipt=masked
        )
        expected_inner_training = tuple(item for item in families if item != family)
        expected_examples = tuple(
            _identifier(item, label="V20o inner expected example")
            for item in _sequence(
                training_ids_by_family.get(family),
                label="V20o inner expected examples",
            )
        )
        objectives = _mapping(
            evidence.get("objective_by_response"),
            label="V20o inner objectives",
        )
        response_evidence = _mapping(
            evidence.get("response_evidence"),
            label="V20o inner response evidence",
        )
        inherited_v20m_responses = _mapping(
            _mapping(
                inherited_v20m_inner_evidence[family],
                label="V20o authenticated V20m inner family evidence",
            ).get("response_evidence"),
            label="V20o authenticated V20m response evidence",
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
            raise ValueError("V20o inner evidence schedule differs")
        family_provider_hashes = _mapping(
            provider_hashes[family], label="V20o inner family provider hashes"
        )
        family_provider_receipts = _mapping(
            provider_receipts[family], label="V20o inner family provider receipts"
        )
        family_transfer_hashes = _mapping(
            transfer_hashes[family],
            label="V20o inner family provider transfer hashes",
        )
        family_trace_hashes = _mapping(
            trace_hashes[family], label="V20o inner family trace hashes"
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
            raise ValueError("V20o inner manifest response bindings differ")
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
                    response_evidence[key], label="V20o inner response arm evidence"
                ),
                domain=_INNER_EXECUTION_DOMAIN,
                label="V20o inner response arm evidence",
            )
            trace = _validate_hashed(
                _mapping(arm.get("response_trace"), label="V20o inner trace"),
                domain=_TRACE_DOMAIN,
                label="V20o inner trace",
            )
            provider_receipt = _validate_hashed(
                _mapping(
                    family_provider_receipts[key],
                    label="V20o inner provider receipt",
                ),
                domain=_PROVIDER_DOMAIN,
                label="V20o inner provider receipt",
            )
            score_bundle = _validate_exact_score_bundle(
                arm,
                expected_example_ids=expected_examples,
                label=f"V20o inner {family} response {key}",
            )
            arm_objectives, arm_h4, arm_logits, arm_executions = score_bundle
            inherited_v20m_arm = _mapping(
                inherited_v20m_responses[key],
                label="V20o authenticated V20m response arm",
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
                        "V20o inner exact output differs from authenticated V20m"
                    )
            provider_artifact = _sha(
                family_provider_hashes[key], label="V20o inner provider hash"
            )
            transfer_artifact = _sha(
                family_transfer_hashes[key],
                label="V20o inner provider transfer hash",
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
                family_trace_hashes[key], label="V20o inner trace hash"
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
                label="V20o inner response gain hashes",
            )
            for value in response_gain_hashes.values():
                _sha(value, label="V20o inner response gain hash")
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
                        label="V20o simplex_response box certificate",
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
                raise ValueError("V20o inner response evidence differs")
        _v20b._validate_capability_receipt(
            evidence.get("capability_receipt"),
            expected_example_ids=expected_examples,
            expected_family_count=1,
            expected_held_family_id=outer_family_id,
            expected_accesses_per_example=len(_RESPONSES),
            label="V20o resumed inner-held capability",
        )
        zero = _mapping(
            response_evidence[_response_key((0.0, 0.0, 0.0))],
            label="V20o inner zero-response evidence",
        )
        expected_zero_objectives = {
            _identifier(example, label="V20o inherited zero objective example"): float(
                objective
            )
            for example, objective in _mapping(
                inherited_zero_objectives[family],
                label="V20o inherited family zero objectives",
            ).items()
        }
        zero_anchor = (
            dict(
                _mapping(
                    zero.get("objectives_by_example"),
                    label="V20o zero objectives",
                )
            )
            == expected_zero_objectives
            and dict(
                _mapping(
                    zero.get("post_cast_h4_sha256s"),
                    label="V20o zero H4 hashes",
                )
            )
            == {
                example: inherited_zero_h4[example]
                for example in expected_examples
            }
            and dict(
                _mapping(
                    zero.get("supervised_full_vocab_logits_sha256s"),
                    label="V20o zero logits hashes",
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
            raise ValueError("V20o inner zero-response V20g output anchor differs")
        validated_inner[family] = evidence
    selection = _aggregate_response_selection(validated_inner)
    persisted_selection = _mapping(
        receipt.get("response_selection_receipt"),
        label="V20o persisted response selection",
    )
    if _v14._canonical_json_bytes(selection) != _v14._canonical_json_bytes(
        persisted_selection
    ):
        raise ValueError("V20o inner response selection replay differs")
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
                f"V20o response selection {key} differs from authenticated V20m"
            )
    return receipt


def _validate_signed_continuum_selection(
    value: Mapping[str, object],
    *,
    response_selection: Mapping[str, object],
    inner_receipt: Mapping[str, object],
    endpoint_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20i_fold: Mapping[str, object],
) -> Mapping[str, object]:
    """Replay every signed-anchor, vertex, fitter, and execution receipt."""

    receipt = _validate_hashed(
        value,
        domain=_SIGNED_CONTINUUM_SELECTION_DOMAIN,
        label="V20o signed-continuum selection",
    )
    outer = _identifier(
        outer_family_id, label="V20o signed-continuum outer family"
    )
    response = _response_tuple(response_selection.get("selected_response"))
    response_key = _response_key(response)
    inner_evidence = _mapping(
        inner_receipt.get("inner_evidence_by_family"),
        label="V20o signed-continuum V20m inner evidence",
    )
    families = tuple(sorted(inner_evidence))
    if (
        len(families) != _INNER_FAMILY_COUNT
        or outer in families
        or receipt.get("outer_held_family_id") != outer
        or _response_tuple(receipt.get("source_response")) != response
        or receipt.get("v20m_response_selection_receipt_sha256")
        != response_selection.get("artifact_sha256")
        or receipt.get(
            "all_fourteen_missing_anchor_providers_frozen_before_any_anchor_score"
        )
        is not True
        or receipt.get(
            "proposal_frozen_before_any_vertex_provider_or_score"
        )
        is not True
        or receipt.get(
            "all_vertex_providers_frozen_before_any_vertex_score"
        )
        is not True
        or receipt.get("outer_held_family_used_for_fit_or_selection")
        is not False
        or receipt.get("final_refit_or_calibration_b_used") is not False
        or receipt.get(
            "raw_provider_prompt_token_logit_h4_or_teacher_tensors_serialized"
        )
        is not False
        or int(receipt.get("exact_additional_inner_execution_count", -1))
        != 3 * _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY
    ):
        raise ValueError("V20o signed-continuum selection boundary differs")

    missing_values = (-1.0, 0.0)

    def missing_anchor_id(signed_scalar: float) -> str:
        if signed_scalar == -1.0:
            return "signed_minus_one"
        if signed_scalar == 0.0:
            return "signed_zero"
        raise ValueError("V20o missing anchor must be -1 or 0")

    def missing_anchor_role(signed_scalar: float) -> str:
        if signed_scalar == -1.0:
            return "inner_signed_continuum_mirror_anchor"
        if signed_scalar == 0.0:
            return "inner_signed_continuum_fixed_plus_anchor"
        raise ValueError("V20o missing anchor must be -1 or 0")

    expected_examples: dict[str, tuple[str, ...]] = {}
    selected_v20m_evidence: dict[str, Mapping[str, object]] = {}
    for family in families:
        responses = _mapping(
            _mapping(
                inner_evidence[family], label="V20o inner family evidence"
            ).get("response_evidence"),
            label="V20o inner response evidence",
        )
        selected = _mapping(
            responses.get(response_key),
            label="V20o selected V20m inner response evidence",
        )
        selected_v20m_evidence[family] = selected
        expected_examples[family] = tuple(
            sorted(
                _identifier(example, label="V20o inner example")
                for example in _mapping(
                    selected.get("objectives_by_example"),
                    label="V20o selected V20m objectives",
                )
            )
        )
        if len(expected_examples[family]) != _PROMPTS_PER_FAMILY:
            raise ValueError("V20o selected V20m example geometry differs")

    missing_manifest = _validate_hashed(
        _mapping(
            receipt.get("missing_anchor_provider_manifest"),
            label="V20o missing-anchor manifest",
        ),
        domain=_INNER_MANIFEST_DOMAIN,
        label="V20o missing-anchor manifest",
    )
    missing_evidence = _mapping(
        receipt.get("missing_anchor_evidence_by_family_and_anchor"),
        label="V20o missing-anchor evidence",
    )
    provider_hashes = _mapping(
        missing_manifest.get(
            "provider_artifact_sha256s_by_inner_family_and_anchor"
        ),
        label="V20o missing-anchor provider hashes",
    )
    runtime_hashes = _mapping(
        missing_manifest.get(
            "runtime_provider_artifact_sha256s_by_inner_family_and_anchor"
        ),
        label="V20o missing-anchor runtime hashes",
    )
    provider_receipts = _mapping(
        missing_manifest.get(
            "provider_receipts_by_inner_family_and_anchor"
        ),
        label="V20o missing-anchor provider receipts",
    )
    trace_hashes = _mapping(
        missing_manifest.get("trace_sha256s_by_inner_family_and_anchor"),
        label="V20o missing-anchor trace hashes",
    )
    transfer_hashes = _mapping(
        missing_manifest.get(
            "provider_transfer_evidence_sha256s_by_inner_family_and_anchor"
        ),
        label="V20o missing-anchor transfer hashes",
    )
    if (
        missing_manifest.get("stage") != "missing_signed_anchors"
        or missing_manifest.get("outer_held_family_id") != outer
        or tuple(missing_manifest.get("inner_family_order", ())) != families
        or _response_tuple(missing_manifest.get("source_response"))
        != response
        or tuple(missing_manifest.get("missing_anchor_values", ()))
        != missing_values
        or missing_manifest.get(
            "all_fourteen_missing_anchor_providers_and_traces_frozen_before_"
            "any_anchor_capability"
        )
        is not True
        or missing_manifest.get("anchor_capability_count_at_freeze") != 0
        or missing_manifest.get(
            "anchor_objectives_or_teacher_rows_used_at_freeze"
        )
        is not False
        or missing_manifest.get("outer_held_family_used") is not False
        or missing_manifest.get(
            "raw_provider_or_response_tensors_serialized"
        )
        is not False
        or any(
            set(mapping) != set(families)
            for mapping in (
                missing_evidence,
                provider_hashes,
                runtime_hashes,
                provider_receipts,
                trace_hashes,
                transfer_hashes,
            )
        )
    ):
        raise ValueError("V20o missing-anchor freeze manifest differs")

    missing_objectives: dict[str, dict[str, float]] = {}
    for family in families:
        fit = _mapping(
            _mapping(
                inner_evidence[family], label="V20o inner family evidence"
            ).get("reflection_fit_receipt"),
            label="V20o inner reflection fit",
        )
        direction = _selected_direction(fit)
        family_evidence = _mapping(
            missing_evidence[family],
            label="V20o family missing-anchor evidence",
        )
        family_provider_hashes = _mapping(
            provider_hashes[family],
            label="V20o family missing-anchor provider hashes",
        )
        family_runtime_hashes = _mapping(
            runtime_hashes[family],
            label="V20o family missing-anchor runtime hashes",
        )
        family_receipts = _mapping(
            provider_receipts[family],
            label="V20o family missing-anchor provider receipts",
        )
        family_trace_hashes = _mapping(
            trace_hashes[family],
            label="V20o family missing-anchor trace hashes",
        )
        family_transfer_hashes = _mapping(
            transfer_hashes[family],
            label="V20o family missing-anchor transfer hashes",
        )
        anchor_ids = {"signed_minus_one", "signed_zero"}
        if any(
            set(mapping) != anchor_ids
            for mapping in (
                family_evidence,
                family_provider_hashes,
                family_runtime_hashes,
                family_receipts,
                family_trace_hashes,
                family_transfer_hashes,
            )
        ):
            raise ValueError("V20o missing-anchor family geometry differs")
        missing_objectives[family] = {}
        capability_receipt: object | None = None
        for signed_scalar in missing_values:
            anchor_id = missing_anchor_id(signed_scalar)
            role = missing_anchor_role(signed_scalar)
            evidence = _validate_hashed(
                _mapping(
                    family_evidence[anchor_id],
                    label="V20o missing-anchor score evidence",
                ),
                domain=_INNER_EXECUTION_DOMAIN,
                label="V20o missing-anchor score evidence",
            )
            provider_receipt = _validate_hashed(
                _mapping(
                    family_receipts[anchor_id],
                    label="V20o missing-anchor provider receipt",
                ),
                domain=_PROVIDER_DOMAIN,
                label="V20o missing-anchor provider receipt",
            )
            provider_artifact = _sha(
                family_provider_hashes[anchor_id],
                label="V20o missing-anchor provider artifact",
            )
            runtime_artifact = _sha(
                family_runtime_hashes[anchor_id],
                label="V20o missing-anchor runtime artifact",
            )
            expected_transfer = _signed_continuum_provider_seed(
                endpoint_receipt_sha256=str(
                    endpoint_receipt["artifact_sha256"]
                ),
                direction=direction,
                direction_artifact_sha256=str(
                    fit["selected_variant_artifact_sha256"]
                ),
                reflection_fit_sha256=str(fit["artifact_sha256"]),
                response=response,
                signed_scalar=signed_scalar,
                outer_family_id=outer,
                inner_family_id=family,
                role=role,
            )
            _validate_signed_continuum_provider_receipt_evidence(
                provider_receipt,
                expected_role=role,
                expected_provider_artifact_sha256=provider_artifact,
                expected_endpoint_receipt=endpoint_receipt,
                expected_bridge_binding_sha256=bridge_binding_sha256,
                authenticated_v20i_fold=authenticated_v20i_fold,
                expected_response=response,
                expected_signed_scalar=signed_scalar,
                expected_direction=direction,
                expected_transfer_evidence_sha256=expected_transfer,
            )
            objectives, h4_hashes, logits_hashes, execution_hashes = (
                _validate_exact_score_bundle(
                    evidence,
                    expected_example_ids=expected_examples[family],
                    label=f"V20o {family} {anchor_id}",
                )
            )
            objective = math.fsum(objectives.values()) / len(objectives)
            score_seed = _v14._sha256(
                {
                    "missing_anchor_manifest_sha256": (
                        missing_manifest["artifact_sha256"]
                    ),
                    "outer_held_family_id": outer,
                    "inner_held_family_id": family,
                    "anchor_id": anchor_id,
                    "signed_scalar": signed_scalar,
                    "provider_artifact_sha256": provider_artifact,
                    "runtime_provider_artifact_sha256": runtime_artifact,
                    "all_fourteen_missing_anchor_providers_frozen": True,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
            expected_executions = {
                example: _execution_sha256(
                    phase="inner_signed_continuum_missing_anchor_score",
                    outer_family_id=outer,
                    inner_family_id=family,
                    role=role,
                    provider_artifact_sha256=runtime_artifact,
                    example_id=example,
                    family_id=family,
                    objective=objectives[example],
                    h4_sha256=h4_hashes[example],
                    logits_sha256=logits_hashes[example],
                    evidence_sha256=score_seed,
                    domain=_INNER_EXECUTION_DOMAIN,
                )
                for example in expected_examples[family]
            }
            trace = _validate_hashed(
                _mapping(
                    evidence.get("response_trace"),
                    label="V20o missing-anchor trace",
                ),
                domain=_TRACE_DOMAIN,
                label="V20o missing-anchor trace",
            )
            response_gain_hashes = _mapping(
                trace.get("response_gain_sha256s"),
                label="V20o missing-anchor response gain hashes",
            )
            for gain_hash in response_gain_hashes.values():
                _sha(gain_hash, label="V20o response gain hash")
            current_capability = evidence.get("capability_receipt")
            if capability_receipt is None:
                capability_receipt = current_capability
            elif _v14._canonical_json_bytes(current_capability) != (
                _v14._canonical_json_bytes(capability_receipt)
            ):
                raise ValueError(
                    "V20o missing anchors did not share one family capability"
                )
            if (
                evidence.get("stage") != "missing_signed_anchor"
                or evidence.get("outer_held_family_id") != outer
                or evidence.get("inner_held_family_id") != family
                or evidence.get("anchor_id") != anchor_id
                or evidence.get("role") != role
                or float(evidence.get("signed_scalar", math.nan))
                != signed_scalar
                or evidence.get("signed_scalar_hex") != signed_scalar.hex()
                or evidence.get("provider_artifact_sha256")
                != provider_artifact
                or evidence.get("runtime_provider_artifact_sha256")
                != runtime_artifact
                or evidence.get("lineage_wrapper_not_inference_executor")
                is not True
                or evidence.get("manifest_sha256")
                != missing_manifest.get("artifact_sha256")
                or float(evidence.get("objective", math.nan)) != objective
                or execution_hashes != expected_executions
                or provider_receipt.get(
                    "runtime_provider_artifact_sha256"
                )
                != runtime_artifact
                or family_transfer_hashes[anchor_id] != expected_transfer
                or trace.get("artifact_sha256")
                != family_trace_hashes[anchor_id]
                or trace.get("provider_artifact_sha256")
                != provider_artifact
                or trace.get("arm") != role
                or tuple(trace.get("scored_family_ids", ())) != (family,)
                or set(response_gain_hashes) != set(expected_examples[family])
                or evidence.get(
                    "all_fourteen_missing_anchor_providers_frozen_before_score"
                )
                is not True
                or evidence.get("outer_family_absent_from_fit_and_score")
                is not True
                or evidence.get("exact_execution") is not True
                or evidence.get("finite") is not True
                or evidence.get(
                    "raw_logits_h4_teacher_rows_or_tensors_serialized"
                )
                is not False
                or trace.get("finite") is not True
                or trace.get("pointwise_trust_passed") is not True
                or trace.get("endpoint_conditional_ranks_are_16") is not True
                or trace.get("raw_response_or_modal_tensors_serialized")
                is not False
            ):
                raise ValueError("V20o missing-anchor evidence differs")
            missing_objectives[family][anchor_id] = objective
        _v20b._validate_capability_receipt(
            capability_receipt,
            expected_example_ids=expected_examples[family],
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=2,
            label="V20o missing-anchor capability",
        )

    persisted_plus_one = _mapping(
        receipt.get("reused_v20m_plus_one_evidence_by_family"),
        label="V20o reused V20m plus-one evidence",
    )
    if set(persisted_plus_one) != set(families):
        raise ValueError("V20o reused V20m plus-one geometry differs")
    expected_anchors: dict[str, dict[str, float]] = {}
    for family in families:
        if _v14._canonical_json_bytes(persisted_plus_one[family]) != (
            _v14._canonical_json_bytes(selected_v20m_evidence[family])
        ):
            raise ValueError("V20o plus-one evidence differs from live V20m")
        expected_anchors[family] = {
            "signed_minus_one": missing_objectives[family][
                "signed_minus_one"
            ],
            "signed_zero": missing_objectives[family]["signed_zero"],
            "signed_plus_one": float(
                selected_v20m_evidence[family]["objective"]
            ),
        }
    persisted_anchors = _mapping(
        receipt.get("anchor_objectives_by_family_and_anchor"),
        label="V20o persisted signed anchor objectives",
    )
    if _v14._canonical_json_bytes(persisted_anchors) != (
        _v14._canonical_json_bytes(expected_anchors)
    ):
        raise ValueError("V20o anchor objectives differ from exact evidence")

    all_families = tuple(sorted((*families, outer)))
    anchor_receipt = _mapping(
        receipt.get("core_anchor_receipt"),
        label="V20o core anchor receipt",
    )
    _signed_continuum_fit.validate_soft_polarity_signed_continuum_anchor_receipt(
        anchor_receipt,
        all_development_family_ids=all_families,
        outer_held_family_id=outer,
        exact_anchor_objectives_by_family_and_anchor=expected_anchors,
    )
    proposal_receipt = _mapping(
        receipt.get("core_quadratic_proposal_receipt"),
        label="V20o core proposal receipt",
    )
    _signed_continuum_fit.validate_soft_polarity_signed_continuum_quadratic_proposal_receipt(
        proposal_receipt,
        anchor_receipt=anchor_receipt,
    )
    proposed_signed_scalar = float(
        proposal_receipt["proposed_signed_scalar"]
    )
    if not -1.0 <= proposed_signed_scalar <= 1.0:
        raise ValueError("V20o proposed signed scalar is outside [-1,1]")

    vertex_manifest = _validate_hashed(
        _mapping(
            receipt.get("vertex_provider_manifest"),
            label="V20o vertex manifest",
        ),
        domain=_INNER_MANIFEST_DOMAIN,
        label="V20o vertex manifest",
    )
    vertex_evidence = _mapping(
        receipt.get("vertex_evidence_by_family"),
        label="V20o vertex evidence",
    )
    vertex_provider_hashes = _mapping(
        vertex_manifest.get(
            "provider_artifact_sha256s_by_inner_family"
        ),
        label="V20o vertex provider hashes",
    )
    vertex_runtime_hashes = _mapping(
        vertex_manifest.get(
            "runtime_provider_artifact_sha256s_by_inner_family"
        ),
        label="V20o vertex runtime hashes",
    )
    vertex_receipts = _mapping(
        vertex_manifest.get("provider_receipts_by_inner_family"),
        label="V20o vertex provider receipts",
    )
    vertex_trace_hashes = _mapping(
        vertex_manifest.get("trace_sha256s_by_inner_family"),
        label="V20o vertex trace hashes",
    )
    vertex_transfer_hashes = _mapping(
        vertex_manifest.get(
            "provider_transfer_evidence_sha256s_by_inner_family"
        ),
        label="V20o vertex transfer hashes",
    )
    if (
        vertex_manifest.get("stage") != "quadratic_vertex"
        or vertex_manifest.get("outer_held_family_id") != outer
        or tuple(vertex_manifest.get("inner_family_order", ())) != families
        or _response_tuple(vertex_manifest.get("source_response"))
        != response
        or float(vertex_manifest.get("signed_scalar", math.nan))
        != proposed_signed_scalar
        or vertex_manifest.get("signed_scalar_hex")
        != proposed_signed_scalar.hex()
        or vertex_manifest.get("anchor_receipt_sha256")
        != anchor_receipt.get("artifact_sha256")
        or vertex_manifest.get("proposal_receipt_sha256")
        != proposal_receipt.get("artifact_sha256")
        or vertex_manifest.get(
            "all_seven_vertex_providers_and_traces_frozen_before_any_vertex_"
            "capability"
        )
        is not True
        or vertex_manifest.get("vertex_capability_count_at_freeze") != 0
        or vertex_manifest.get(
            "vertex_objectives_or_teacher_rows_used_at_freeze"
        )
        is not False
        or vertex_manifest.get("outer_held_family_used") is not False
        or vertex_manifest.get(
            "raw_provider_or_response_tensors_serialized"
        )
        is not False
        or any(
            set(mapping) != set(families)
            for mapping in (
                vertex_evidence,
                vertex_provider_hashes,
                vertex_runtime_hashes,
                vertex_receipts,
                vertex_trace_hashes,
                vertex_transfer_hashes,
            )
        )
    ):
        raise ValueError("V20o vertex freeze manifest differs")

    vertex_objectives: dict[str, float] = {}
    endpoint_anchors: dict[str, bool] = {}
    for family in families:
        fit = _mapping(
            _mapping(
                inner_evidence[family], label="V20o inner family evidence"
            ).get("reflection_fit_receipt"),
            label="V20o inner reflection fit",
        )
        direction = _selected_direction(fit)
        provider_receipt = _validate_hashed(
            _mapping(
                vertex_receipts[family],
                label="V20o vertex provider receipt",
            ),
            domain=_PROVIDER_DOMAIN,
            label="V20o vertex provider receipt",
        )
        provider_artifact = _sha(
            vertex_provider_hashes[family],
            label="V20o vertex provider artifact",
        )
        runtime_artifact = _sha(
            vertex_runtime_hashes[family],
            label="V20o vertex runtime artifact",
        )
        expected_transfer = _signed_continuum_provider_seed(
            endpoint_receipt_sha256=str(endpoint_receipt["artifact_sha256"]),
            direction=direction,
            direction_artifact_sha256=str(
                fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(fit["artifact_sha256"]),
            response=response,
            signed_scalar=proposed_signed_scalar,
            outer_family_id=outer,
            inner_family_id=family,
            role="inner_signed_continuum_vertex",
        )
        _validate_signed_continuum_provider_receipt_evidence(
            provider_receipt,
            expected_role="inner_signed_continuum_vertex",
            expected_provider_artifact_sha256=provider_artifact,
            expected_endpoint_receipt=endpoint_receipt,
            expected_bridge_binding_sha256=bridge_binding_sha256,
            authenticated_v20i_fold=authenticated_v20i_fold,
            expected_response=response,
            expected_signed_scalar=proposed_signed_scalar,
            expected_direction=direction,
            expected_transfer_evidence_sha256=expected_transfer,
        )
        evidence = _validate_hashed(
            _mapping(
                vertex_evidence[family],
                label="V20o vertex score evidence",
            ),
            domain=_INNER_EXECUTION_DOMAIN,
            label="V20o vertex score evidence",
        )
        objectives, h4_hashes, logits_hashes, execution_hashes = (
            _validate_exact_score_bundle(
                evidence,
                expected_example_ids=expected_examples[family],
                label=f"V20o {family} vertex",
            )
        )
        objective = math.fsum(objectives.values()) / len(objectives)
        score_seed = _v14._sha256(
            {
                "vertex_manifest_sha256": vertex_manifest[
                    "artifact_sha256"
                ],
                "outer_held_family_id": outer,
                "inner_held_family_id": family,
                "provider_artifact_sha256": provider_artifact,
                "runtime_provider_artifact_sha256": runtime_artifact,
                "all_seven_vertex_providers_frozen": True,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
        expected_executions = {
            example: _execution_sha256(
                phase="inner_signed_continuum_vertex_score",
                outer_family_id=outer,
                inner_family_id=family,
                role="inner_signed_continuum_vertex",
                provider_artifact_sha256=runtime_artifact,
                example_id=example,
                family_id=family,
                objective=objectives[example],
                h4_sha256=h4_hashes[example],
                logits_sha256=logits_hashes[example],
                evidence_sha256=score_seed,
                domain=_INNER_EXECUTION_DOMAIN,
            )
            for example in expected_examples[family]
        }
        trace = _validate_hashed(
            _mapping(
                evidence.get("response_trace"),
                label="V20o vertex trace",
            ),
            domain=_TRACE_DOMAIN,
            label="V20o vertex trace",
        )
        response_gain_hashes = _mapping(
            trace.get("response_gain_sha256s"),
            label="V20o vertex response gain hashes",
        )
        for gain_hash in response_gain_hashes.values():
            _sha(gain_hash, label="V20o vertex response gain hash")
        endpoint_anchor = True
        if proposed_signed_scalar in _SIGNED_CONTINUUM_ANCHOR_VALUES:
            expected = (
                selected_v20m_evidence[family]
                if proposed_signed_scalar == 1.0
                else _mapping(
                    _mapping(
                        missing_evidence[family],
                        label="V20o family missing evidence",
                    )[missing_anchor_id(proposed_signed_scalar)],
                    label="V20o endpoint missing evidence",
                )
            )
            endpoint_anchor = (
                objective == float(expected["objective"])
                and dict(objectives)
                == dict(
                    _mapping(
                        expected.get("objectives_by_example"),
                        label="V20o endpoint objectives",
                    )
                )
                and dict(h4_hashes)
                == dict(
                    _mapping(
                        expected.get("post_cast_h4_sha256s"),
                        label="V20o endpoint H4 hashes",
                    )
                )
                and dict(logits_hashes)
                == dict(
                    _mapping(
                        expected.get(
                            "supervised_full_vocab_logits_sha256s"
                        ),
                        label="V20o endpoint logits hashes",
                    )
                )
            )
        endpoint_anchors[family] = endpoint_anchor
        if (
            evidence.get("stage") != "quadratic_vertex"
            or evidence.get("outer_held_family_id") != outer
            or evidence.get("inner_held_family_id") != family
            or float(evidence.get("signed_scalar", math.nan))
            != proposed_signed_scalar
            or evidence.get("provider_artifact_sha256")
            != provider_artifact
            or evidence.get("runtime_provider_artifact_sha256")
            != runtime_artifact
            or evidence.get("lineage_wrapper_not_inference_executor")
            is not True
            or evidence.get("manifest_sha256")
            != vertex_manifest.get("artifact_sha256")
            or float(evidence.get("objective", math.nan)) != objective
            or execution_hashes != expected_executions
            or provider_receipt.get("runtime_provider_artifact_sha256")
            != runtime_artifact
            or vertex_transfer_hashes[family] != expected_transfer
            or trace.get("artifact_sha256")
            != vertex_trace_hashes[family]
            or trace.get("provider_artifact_sha256")
            != provider_artifact
            or trace.get("arm") != "inner_signed_continuum_vertex"
            or tuple(trace.get("scored_family_ids", ())) != (family,)
            or set(response_gain_hashes) != set(expected_examples[family])
            or evidence.get(
                "all_vertex_providers_frozen_before_score"
            )
            is not True
            or evidence.get("endpoint_vertex_exact_anchor")
            is not endpoint_anchor
            or evidence.get("outer_family_absent_from_fit_and_score")
            is not True
            or evidence.get("exact_execution") is not True
            or evidence.get("finite") is not True
            or evidence.get(
                "raw_logits_h4_teacher_rows_or_tensors_serialized"
            )
            is not False
            or trace.get("finite") is not True
            or trace.get("pointwise_trust_passed") is not True
            or trace.get("endpoint_conditional_ranks_are_16") is not True
            or trace.get("raw_response_or_modal_tensors_serialized")
            is not False
        ):
            raise ValueError("V20o vertex evidence differs")
        _v20b._validate_capability_receipt(
            evidence.get("capability_receipt"),
            expected_example_ids=expected_examples[family],
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=1,
            label="V20o vertex capability",
        )
        vertex_objectives[family] = objective

    persisted_endpoint_anchors = _mapping(
        receipt.get("endpoint_vertex_exact_anchor_by_family"),
        label="V20o persisted endpoint anchors",
    )
    if (
        _v14._canonical_json_bytes(persisted_endpoint_anchors)
        != _v14._canonical_json_bytes(endpoint_anchors)
        or not all(endpoint_anchors.values())
        or receipt.get("all_endpoint_vertex_exact_anchors_passed") is not True
    ):
        raise ValueError("V20o endpoint vertex anchor replay differs")
    vertex_receipt = _mapping(
        receipt.get("core_vertex_score_receipt"),
        label="V20o core vertex score receipt",
    )
    _signed_continuum_fit.validate_soft_polarity_signed_continuum_vertex_score_receipt(
        vertex_receipt,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        exact_vertex_objectives_by_family=vertex_objectives,
    )
    selection_receipt = _mapping(
        receipt.get("core_selection_receipt"),
        label="V20o core selection receipt",
    )
    _signed_continuum_fit.validate_soft_polarity_signed_continuum_selection_receipt(
        selection_receipt,
        anchor_receipt=anchor_receipt,
        proposal_receipt=proposal_receipt,
        vertex_score_receipt=vertex_receipt,
    )
    selected_signed_scalar = float(
        selection_receipt["selected_signed_scalar"]
    )
    if (
        not -1.0 <= selected_signed_scalar <= 1.0
        or float(receipt.get("selected_signed_scalar", math.nan))
        != selected_signed_scalar
        or receipt.get("selected_signed_scalar_hex")
        != selected_signed_scalar.hex()
        or receipt.get("selected_signed_scalar_interior")
        is not (
            (-1.0 < selected_signed_scalar < 0.0)
            or (0.0 < selected_signed_scalar < 1.0)
        )
        or receipt.get("selected_signed_scalar_negative")
        is not (selected_signed_scalar < 0.0)
        or receipt.get("selected_signed_scalar_positive")
        is not (selected_signed_scalar > 0.0)
    ):
        raise ValueError("V20o selected signed scalar differs")
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
    authenticated_v20n_fold: Mapping[str, object],
) -> None:
    fragment = _mapping(value, label="V20o fold fragment")
    if set(fragment) != _FOLD_FRAGMENT_KEYS:
        raise ValueError("V20o fold fragment key set differs")
    outer = _identifier(outer_family_id, label="V20o fold outer family")
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
        "signed_continuum_fit_protocol_sha256": (
            _signed_continuum_fit.SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256
        ),
        "signed_continuum_provider_protocol_sha256": (
            FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256
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
        "v20n_fold_fragment_sha256": authenticated_v20n_fold[
            "fragment_sha256"
        ],
        "outer_held_family_id": outer,
    }
    if any(fragment.get(key) != expected for key, expected in expected_header.items()):
        raise ValueError("V20o fold fragment header differs")
    if (
        fragment.get("fixed_schedule_completed") is not True
        or fragment.get("candidate") is not None
        or fragment.get("provider_sidecar") is not None
    ):
        raise ValueError("V20o fold scalar-only boundary differs")
    for key in ("endpoint_receipt", "endpoint_evidence"):
        if _v14._canonical_json_bytes(fragment.get(key)) != _v14._canonical_json_bytes(
            authenticated_v20g_fold.get(key)
        ):
            raise ValueError("V20o fold endpoint lineage differs")

    fit = _mapping(
        authenticated_v20g_fold.get("fit_receipt"),
        label="V20o validation V20g fit",
    )
    source_direction = _mapping(
        fit.get("direction_receipt"), label="V20o validation source direction"
    )
    inner = _validate_inner_receipt(
        _mapping(fragment.get("inner_receipt"), label="V20o inner receipt"),
        source_direction=source_direction,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20i_fold=authenticated_v20i_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
    )
    response_selection = _validate_hashed(
        _mapping(
            fragment.get("response_selection_receipt"),
            label="V20o response selection",
        ),
        domain=_RESPONSE_SELECTION_DOMAIN,
        label="V20o response selection",
    )
    if _v14._canonical_json_bytes(response_selection) != _v14._canonical_json_bytes(
        inner.get("response_selection_receipt")
    ):
        raise ValueError("V20o duplicated response selection differs")
    endpoint_receipt = _mapping(
        authenticated_v20g_fold.get("endpoint_receipt"),
        label="V20o outer inherited endpoint receipt",
    )
    signed_continuum_selection = _validate_signed_continuum_selection(
        _mapping(
            fragment.get("signed_continuum_selection_receipt"),
            label="V20o signed_continuum selection receipt",
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
        label="V20o outer reflection fit",
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
            fragment.get("provider_manifest"), label="V20o outer manifest"
        ),
        domain=_OUTER_MANIFEST_DOMAIN,
        label="V20o outer manifest",
    )
    if (
        tuple(manifest.get("arm_order", ())) != _ARMS
        or manifest.get("outer_held_family_id") != outer
        or _response_tuple(manifest.get("selected_response"))
        != _response_tuple(response_selection["selected_response"])
        or float(manifest.get("selected_signed_scalar", math.nan))
        != float(signed_continuum_selection["selected_signed_scalar"])
        or manifest.get("selected_signed_scalar_hex")
        != float(signed_continuum_selection["selected_signed_scalar"]).hex()
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
        raise ValueError("V20o outer provider manifest differs")
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s"),
        label="V20o outer provider hashes",
    )
    provider_receipts = _mapping(
        manifest.get("provider_receipts"),
        label="V20o outer provider receipts",
    )
    runtime_provider_hashes = _mapping(
        manifest.get("runtime_provider_artifact_sha256s"),
        label="V20o outer runtime provider hashes",
    )
    trace_hashes = _mapping(
        manifest.get("response_trace_sha256s"),
        label="V20o outer trace hashes",
    )
    soft_transfer_hashes = _mapping(
        manifest.get("soft_provider_transfer_evidence_sha256s"),
        label="V20o outer soft provider transfer hashes",
    )
    expected_soft_transfer_arms = (
        "matched_linear_reflected",
        "matched_v20l_boundary_reflected",
        "same_simplex_response_unreflected",
        "signed_continuum_reflected",
        "simplex_response_reflected_exact_mirror",
        "matched_v20m_simplex_reflected",
    )
    if set(soft_transfer_hashes) != set(expected_soft_transfer_arms):
        raise ValueError("V20o outer soft transfer arm bindings differ")
    fixed_control_transfer_hash = _sha(
        manifest.get("fixed_control_transfer_evidence_sha256"),
        label="V20o outer fixed-control transfer hash",
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
        raise ValueError("V20o outer manifest arm bindings differ")
    inherited_arms = _mapping(
        _mapping(
            authenticated_v20g_fold.get("held_evidence"),
            label="V20o inherited V20g held evidence",
        ).get("arm_evidence"),
        label="V20o inherited V20g held arms",
    )
    if not all(arm in inherited_arms for arm in ("base", "fixed_plus", "fixed_minus")):
        raise ValueError("V20o inherited V20g control arms differ")
    expected_examples = tuple(
        sorted(
            _identifier(example, label="V20o outer expected example")
            for example in _mapping(
                _mapping(
                    inherited_arms["base"],
                    label="V20o inherited V20g base arm",
                ).get("objectives_by_example"),
                label="V20o inherited V20g base objectives",
            )
        )
    )
    held = _validate_hashed(
        _mapping(fragment.get("held_evidence"), label="V20o held evidence"),
        domain=_OUTER_EXECUTION_DOMAIN,
        label="V20o held evidence",
    )
    arms = _mapping(held.get("arm_evidence"), label="V20o held arm evidence")
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
        or held.get("exact_mirror_v20m_exact_output_anchor_passed") is not True
        or int(held.get("exact_outer_execution_count", -1))
        != len(_ARMS) * _PROMPTS_PER_FAMILY
        or held.get("raw_prompts_tokens_logits_h4_or_teacher_rows_serialized")
        is not False
    ):
        raise ValueError("V20o outer held schedule differs")
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
    selected_signed_scalar = float(signed_continuum_selection["selected_signed_scalar"])
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
        "same_simplex_response_unreflected": _signed_continuum_provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            signed_scalar=selected_signed_scalar,
            direction_artifact_sha256=str(source_direction["artifact_sha256"]),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            direction=unreflected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_same_simplex_response_unreflected",
        ),
        "signed_continuum_reflected": _signed_continuum_provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            signed_scalar=selected_signed_scalar,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_signed_continuum_reflected",
        ),
        "simplex_response_reflected_exact_mirror": _provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
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
        "signed_continuum_reflected": selected_direction,
        "simplex_response_reflected_exact_mirror": mirror_direction,
        "matched_v20m_simplex_reflected": selected_direction,
    }
    expected_soft_responses = {
        "matched_linear_reflected": matched_linear_response,
        "matched_v20l_boundary_reflected": matched_v20l_boundary_response,
        "same_simplex_response_unreflected": selected_response,
        "signed_continuum_reflected": selected_response,
        "simplex_response_reflected_exact_mirror": selected_response,
        "matched_v20m_simplex_reflected": selected_response,
    }
    signed_continuum_arms = {
        "same_simplex_response_unreflected",
        "signed_continuum_reflected",
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
                label=f"V20o {arm} soft transfer hash",
            )
            for arm in expected_soft_transfer_arms
        }
        != expected_soft_transfers
        or fixed_control_transfer_hash != expected_fixed_control_transfer
    ):
        raise ValueError("V20o outer provider transfer lineage differs")
    health_by_arm: dict[str, bool] = {}
    for arm in _ARMS:
        evidence = _validate_hashed(
            _mapping(arms[arm], label=f"V20o {arm} arm evidence"),
            domain=_OUTER_EXECUTION_DOMAIN,
            label=f"V20o {arm} arm evidence",
        )
        trace = _validate_hashed(
            _mapping(evidence.get("response_trace"), label=f"V20o {arm} trace"),
            domain=_TRACE_DOMAIN,
            label=f"V20o {arm} trace",
        )
        provider_receipt = _validate_hashed(
            _mapping(
                provider_receipts[arm],
                label=f"V20o {arm} provider receipt",
            ),
            domain=_PROVIDER_DOMAIN,
            label=f"V20o {arm} provider receipt",
        )
        score_bundle = _validate_exact_score_bundle(
            evidence,
            expected_example_ids=expected_examples,
            label=f"V20o outer {arm}",
        )
        arm_objectives, arm_h4, arm_logits, arm_executions = score_bundle
        provider_artifact = _sha(
            provider_hashes[arm], label=f"V20o {arm} provider hash"
        )
        if arm in signed_continuum_arms:
            _validate_signed_continuum_provider_receipt_evidence(
                provider_receipt,
                expected_role=arm,
                expected_provider_artifact_sha256=provider_artifact,
                expected_endpoint_receipt=endpoint_receipt,
                expected_bridge_binding_sha256=bridge_binding_sha256,
                authenticated_v20i_fold=authenticated_v20i_fold,
                expected_response=expected_soft_responses[arm],
                expected_signed_scalar=selected_signed_scalar,
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
            trace_hashes[arm], label=f"V20o {arm} trace hash"
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
                in signed_continuum_arms,
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
            label=f"V20o {arm} response gain hashes",
        )
        for value in response_gain_hashes.values():
            _sha(value, label=f"V20o {arm} response gain hash")
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
            is not (arm in signed_continuum_arms)
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
                and arm not in signed_continuum_arms
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
                            label=f"V20o {arm} simplex_response certificate",
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
            raise ValueError("V20o outer arm health differs")
        objectives[arm] = float(evidence["objective"])
        exact_outputs[arm] = (arm_objectives, arm_h4, arm_logits)
    _v20b._validate_capability_receipt(
        held.get("capability_receipt"),
        expected_example_ids=expected_examples,
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ARMS),
        label="V20o resumed outer-held capability",
    )
    control_anchors: dict[str, bool] = {}
    for arm in ("base", "fixed_plus", "fixed_minus"):
        inherited = _mapping(
            inherited_arms[arm], label=f"V20o inherited V20g {arm} arm"
        )
        current_objectives, current_h4, current_logits = exact_outputs[arm]
        control_anchors[arm] = bool(
            objectives[arm] == float(inherited.get("objective", math.nan))
            and current_objectives
            == {
                _identifier(example, label=f"V20o inherited {arm} example"): float(
                    objective
                )
                for example, objective in _mapping(
                    inherited.get("objectives_by_example"),
                    label=f"V20o inherited {arm} objectives",
                ).items()
            }
            and current_h4
            == dict(
                _mapping(
                    inherited.get("post_cast_h4_sha256s"),
                    label=f"V20o inherited {arm} H4 hashes",
                )
            )
            and current_logits
            == dict(
                _mapping(
                    inherited.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20o inherited {arm} logits hashes",
                )
            )
        )
    persisted_control_anchors = _mapping(
        held.get("v20g_control_output_anchors"),
        label="V20o persisted control anchors",
    )
    if (
        not all(control_anchors.values())
        or dict(persisted_control_anchors) != control_anchors
        or held.get("all_v20g_control_output_anchors_passed")
        is not all(control_anchors.values())
    ):
        raise ValueError("V20o outer V20g control output anchor differs")
    inherited_v20l_arm = _mapping(
        _mapping(
            _mapping(
                authenticated_v20l_fold.get("held_evidence"),
                label="V20o resumed V20l held evidence",
            ).get("arm_evidence"),
            label="V20o resumed V20l held arms",
        ).get("signed_stack_reflected"),
        label="V20o resumed V20l selected boundary arm",
    )
    boundary_objectives, boundary_h4, boundary_logits = exact_outputs[
        "matched_v20l_boundary_reflected"
    ]
    v20l_boundary_anchor = bool(
        objectives["matched_v20l_boundary_reflected"]
        == float(inherited_v20l_arm.get("objective", math.nan))
        and boundary_objectives
        == {
            _identifier(example, label="V20o inherited V20l boundary example"): float(
                objective
            )
            for example, objective in _mapping(
                inherited_v20l_arm.get("objectives_by_example"),
                label="V20o inherited V20l boundary objectives",
            ).items()
        }
        and boundary_h4
        == dict(
            _mapping(
                inherited_v20l_arm.get("post_cast_h4_sha256s"),
                label="V20o inherited V20l boundary H4 hashes",
            )
        )
        and boundary_logits
        == dict(
            _mapping(
                inherited_v20l_arm.get("supervised_full_vocab_logits_sha256s"),
                label="V20o inherited V20l boundary logits hashes",
            )
        )
    )
    if (
        not v20l_boundary_anchor
        or held.get("matched_v20l_boundary_exact_output_anchor_passed") is not True
    ):
        raise ValueError("V20o exact V20l boundary output anchor differs")
    inherited_v20m_arm = _mapping(
        _mapping(
            _mapping(
                authenticated_v20m_fold.get("held_evidence"),
                label="V20o resumed V20m held evidence",
            ).get("arm_evidence"),
            label="V20o resumed V20m held arms",
        ).get("simplex_response_reflected"),
        label="V20o resumed V20m selected simplex arm",
    )
    v20m_objectives, v20m_h4, v20m_logits = exact_outputs[
        "matched_v20m_simplex_reflected"
    ]
    v20m_anchor = bool(
        objectives["matched_v20m_simplex_reflected"]
        == float(inherited_v20m_arm.get("objective", math.nan))
        and v20m_objectives
        == {
            _identifier(example, label="V20o inherited V20m example"): float(
                objective
            )
            for example, objective in _mapping(
                inherited_v20m_arm.get("objectives_by_example"),
                label="V20o inherited V20m objectives",
            ).items()
        }
        and v20m_h4
        == dict(
            _mapping(
                inherited_v20m_arm.get("post_cast_h4_sha256s"),
                label="V20o inherited V20m H4 hashes",
            )
        )
        and v20m_logits
        == dict(
            _mapping(
                inherited_v20m_arm.get(
                    "supervised_full_vocab_logits_sha256s"
                ),
                label="V20o inherited V20m logits hashes",
            )
        )
    )
    if (
        not v20m_anchor
        or held.get("matched_v20m_exact_output_anchor_passed") is not True
    ):
        raise ValueError("V20o exact V20m output anchor differs")
    inherited_v20m_mirror = _mapping(
        _mapping(
            _mapping(
                authenticated_v20m_fold.get("held_evidence"),
                label="V20o resumed V20m held evidence",
            ).get("arm_evidence"),
            label="V20o resumed V20m held arms",
        ).get("simplex_response_reflected_exact_mirror"),
        label="V20o resumed V20m mirror arm",
    )
    mirror_objectives, mirror_h4, mirror_logits = exact_outputs[
        "simplex_response_reflected_exact_mirror"
    ]
    v20m_mirror_anchor = bool(
        objectives["simplex_response_reflected_exact_mirror"]
        == float(inherited_v20m_mirror.get("objective", math.nan))
        and mirror_objectives
        == {
            _identifier(example, label="V20o inherited mirror example"): float(
                objective
            )
            for example, objective in _mapping(
                inherited_v20m_mirror.get("objectives_by_example"),
                label="V20o inherited mirror objectives",
            ).items()
        }
        and mirror_h4
        == dict(
            _mapping(
                inherited_v20m_mirror.get("post_cast_h4_sha256s"),
                label="V20o inherited mirror H4 hashes",
            )
        )
        and mirror_logits
        == dict(
            _mapping(
                inherited_v20m_mirror.get(
                    "supervised_full_vocab_logits_sha256s"
                ),
                label="V20o inherited mirror logits hashes",
            )
        )
    )
    if (
        not v20m_mirror_anchor
        or held.get("exact_mirror_v20m_exact_output_anchor_passed") is not True
    ):
        raise ValueError("V20o exact V20m mirror output anchor differs")
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
    candidate_changed_from_mirror = any(
        candidate_logits[example] != mirror_logits[example]
        for example in expected_examples
    )
    fixed_plus_logits = exact_outputs["fixed_plus"][2]
    candidate_changed_from_fixed_plus = any(
        candidate_logits[example] != fixed_plus_logits[example]
        for example in expected_examples
    )
    endpoint_anchor_arm_by_scalar = {
        -1.0: "simplex_response_reflected_exact_mirror",
        0.0: "fixed_plus",
        1.0: "matched_v20m_simplex_reflected",
    }
    endpoint_anchor_id_by_scalar = {
        -1.0: "signed_minus_one",
        0.0: "signed_zero",
        1.0: "signed_plus_one",
    }
    selected_endpoint_anchor_arm = endpoint_anchor_arm_by_scalar.get(
        selected_signed_scalar
    )
    selected_endpoint_anchor_id = endpoint_anchor_id_by_scalar.get(
        selected_signed_scalar
    )
    selected_endpoint_anchor_applicable = selected_endpoint_anchor_arm is not None
    selected_endpoint_exact_anchor = True
    if selected_endpoint_anchor_arm is not None:
        candidate_objectives, candidate_h4, candidate_endpoint_logits = exact_outputs[
            _PRIMARY_ARM
        ]
        endpoint_objectives, endpoint_h4, endpoint_logits = exact_outputs[
            selected_endpoint_anchor_arm
        ]
        selected_endpoint_exact_anchor = (
            objectives[_PRIMARY_ARM] == objectives[selected_endpoint_anchor_arm]
            and candidate_objectives == endpoint_objectives
            and candidate_h4 == endpoint_h4
            and candidate_endpoint_logits == endpoint_logits
        )
        if not selected_endpoint_exact_anchor:
            raise ValueError(
                "V20o selected signed endpoint failed exact anchor reproduction"
            )
    if (
        held.get("selected_endpoint_exact_anchor_applicable")
        is not selected_endpoint_anchor_applicable
        or held.get("selected_endpoint_exact_anchor_id")
        != selected_endpoint_anchor_id
        or held.get("selected_endpoint_exact_anchor_passed")
        is not selected_endpoint_exact_anchor
    ):
        raise ValueError("V20o selected endpoint anchor evidence differs")
    selected_signed_scalar_interior = (
        (-1.0 < selected_signed_scalar < 0.0)
        or (0.0 < selected_signed_scalar < 1.0)
    )
    interior_exact_distinct = (
        not selected_signed_scalar_interior
        or (
            candidate_changed_from_mirror
            and candidate_changed_from_fixed_plus
            and candidate_changed_from_v20m
        )
    )
    candidate_distinct = (
        provider_hashes[_PRIMARY_ARM] != provider_hashes["base"]
    )
    all_healthy = all(health_by_arm.values())
    fold = _validate_hashed(
        _mapping(fragment.get("fold_receipt"), label="V20o fold receipt"),
        domain=_DECISION_DOMAIN,
        label="V20o fold receipt",
    )
    if (
        fold.get("outer_held_family_id") != outer
        or tuple(fold.get("arm_order", ())) != _ARMS
        or dict(fold.get("held_objective_by_arm", {})) != objectives
        or _response_tuple(fold.get("selected_response"))
        != selected_response
        or fold.get("selected_response_key")
        != _response_key(selected_response)
        or float(fold.get("selected_signed_scalar", math.nan)) != selected_signed_scalar
        or fold.get("selected_signed_scalar_hex") != selected_signed_scalar.hex()
        or fold.get("selected_signed_scalar_interior") is not selected_signed_scalar_interior
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
        or fold.get("candidate_exact_execution_changed_from_exact_mirror")
        is not candidate_changed_from_mirror
        or fold.get("candidate_exact_execution_changed_from_fixed_plus")
        is not candidate_changed_from_fixed_plus
        or fold.get("interior_candidate_exact_distinct_from_all_three_anchors")
        is not interior_exact_distinct
        or fold.get("selected_endpoint_exact_anchor_applicable")
        is not selected_endpoint_anchor_applicable
        or fold.get("selected_endpoint_exact_anchor_id")
        != selected_endpoint_anchor_id
        or fold.get("selected_endpoint_exact_anchor_passed")
        is not selected_endpoint_exact_anchor
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
        or fold.get("exact_mirror_v20m_exact_output_anchor_passed")
        is not v20m_mirror_anchor
        or fold.get("selection_frozen_before_outer_score") is not True
        or fold.get("outer_family_used_for_fit_or_selection") is not False
        or fold.get("exact_execution") is not True
    ):
        raise ValueError("V20o fold decision receipt differs")


def _publish_fold_fragment(
    payload: Mapping[str, object], *, output: Path | str, outer_family_id: str
) -> dict[str, object]:
    path = _fold_path(output, outer_family_id)
    _v20b._publish_scalar_fragment(
        payload,
        path=path,
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20o signed-continuum fold fragment",
    )
    return _v20b._load_scalar_fragment(
        path=path,
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20o signed-continuum fold fragment",
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
    authenticated_v20n_fold: Mapping[str, object],
) -> dict[str, object]:
    fragment = _v20b._load_scalar_fragment(
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20o signed-continuum fold fragment",
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
        authenticated_v20n_fold=authenticated_v20n_fold,
    )
    return fragment


def _fold_receipt_map(
    fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    return {
        family: _mapping(
            fragments[family].get("fold_receipt"), label="V20o aggregate fold"
        )
        for family in sorted(fragments)
    }


def _aggregate_decision(
    fold_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    families = tuple(sorted(fold_receipts))
    if len(families) != _FAMILY_COUNT:
        raise ValueError("V20o decision requires all eight outer folds")
    scores: dict[str, dict[str, float]] = {}
    responses: dict[str, tuple[float, float, float]] = {}
    signed_scalars: dict[str, float] = {}
    variants: dict[str, str] = {}
    changed: dict[str, bool] = {}
    changed_from_v20l_boundary: dict[str, bool] = {}
    changed_from_v20m: dict[str, bool] = {}
    changed_from_linear: dict[str, bool] = {}
    interior_exact_distinct: dict[str, bool] = {}
    endpoint_exact_anchor: dict[str, bool] = {}
    health: dict[str, bool] = {}
    v20g_anchors: dict[str, bool] = {}
    v20l_boundary_anchors: dict[str, bool] = {}
    v20m_anchors: dict[str, bool] = {}
    v20m_mirror_anchors: dict[str, bool] = {}
    for family in families:
        fold = fold_receipts[family]
        if tuple(fold.get("arm_order", ())) != _ARMS:
            raise ValueError("V20o aggregate arm order differs")
        raw_scores = _mapping(
            fold.get("held_objective_by_arm"), label="V20o aggregate arm scores"
        )
        if set(raw_scores) != set(_ARMS):
            raise ValueError("V20o aggregate arm geometry differs")
        scores[family] = {arm: float(raw_scores[arm]) for arm in _ARMS}
        if not all(math.isfinite(value) for value in scores[family].values()):
            raise ValueError("V20o aggregate score became nonfinite")
        responses[family] = _response_tuple(fold["selected_response"])
        signed_scalars[family] = float(fold["selected_signed_scalar"])
        if (
            not math.isfinite(signed_scalars[family])
            or not -1.0 <= signed_scalars[family] <= 1.0
        ):
            raise ValueError("V20o aggregate selected signed scalar differs")
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
            "interior_candidate_exact_distinct_from_all_three_anchors"
        ) is True
        endpoint_exact_anchor[family] = fold.get(
            "selected_endpoint_exact_anchor_passed"
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
        v20m_mirror_anchors[family] = (
            fold.get("exact_mirror_v20m_exact_output_anchor_passed") is True
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
    interior_signed_scalar = {
        family: (
            (-1.0 < signed_scalars[family] < 0.0)
            or (0.0 < signed_scalars[family] < 1.0)
        )
        for family in families
    }
    interior_signed_scalar_count = sum(interior_signed_scalar.values())
    negative_signed_scalar = {
        family: -1.0 < signed_scalars[family] < 0.0 for family in families
    }
    positive_signed_scalar = {
        family: 0.0 < signed_scalars[family] < 1.0 for family in families
    }
    negative_signed_scalar_count = sum(negative_signed_scalar.values())
    positive_signed_scalar_count = sum(positive_signed_scalar.values())
    continuous_signed_continuum_evidence = (
        interior_signed_scalar_count >= 5
        and negative_signed_scalar_count >= 1
        and positive_signed_scalar_count >= 1
    )
    all_interior_exact_distinct = all(interior_exact_distinct.values())
    integrity = (
        all(health.values())
        and all(v20g_anchors.values())
        and all(v20l_boundary_anchors.values())
        and all(v20m_anchors.values())
        and all(v20m_mirror_anchors.values())
        and all_interior_exact_distinct
        and all(endpoint_exact_anchor.values())
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
        and continuous_signed_continuum_evidence
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
            "selected_signed_scalar_by_family": signed_scalars,
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
            "interior_candidate_exact_distinct_from_all_three_anchors_by_family": (
                interior_exact_distinct
            ),
            "selected_endpoint_exact_anchor_by_family": endpoint_exact_anchor,
            "runtime_health_by_family": health,
            "v20g_control_output_anchor_by_family": v20g_anchors,
            "matched_v20l_boundary_output_anchor_by_family": (
                v20l_boundary_anchors
            ),
            "matched_v20m_output_anchor_by_family": v20m_anchors,
            "exact_mirror_v20m_output_anchor_by_family": (
                v20m_mirror_anchors
            ),
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
            "selected_interior_signed_scalar_by_family": (
                interior_signed_scalar
            ),
            "selected_interior_signed_scalar_count": (
                interior_signed_scalar_count
            ),
            "selected_negative_interior_signed_scalar_by_family": (
                negative_signed_scalar
            ),
            "selected_negative_interior_signed_scalar_count": (
                negative_signed_scalar_count
            ),
            "selected_positive_interior_signed_scalar_by_family": (
                positive_signed_scalar
            ),
            "selected_positive_interior_signed_scalar_count": (
                positive_signed_scalar_count
            ),
            "continuous_signed_continuum_evidence_gate_passed": (
                continuous_signed_continuum_evidence
            ),
            "all_interior_candidates_exact_distinct_from_all_three_anchors": (
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
    """Return the fixed canonical V20o one-shot schedule.

    Authentication and resume attempts are deliberately excluded.  V20o
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
    inner_missing_anchor_forwards = (
        _FAMILY_COUNT
        * _INNER_FAMILY_COUNT
        * _PROMPTS_PER_FAMILY
        * 2
    )
    inner_vertex_forwards = (
        _FAMILY_COUNT * _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY
    )
    inner_forwards = (
        inner_response_forwards
        + inner_missing_anchor_forwards
        + inner_vertex_forwards
    )
    outer_forwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY * len(_ARMS)
    inner_response_providers = (
        _FAMILY_COUNT * _INNER_FAMILY_COUNT * len(_RESPONSES)
    )
    inner_missing_anchor_providers = (
        _FAMILY_COUNT * _INNER_FAMILY_COUNT * 2
    )
    inner_vertex_providers = _FAMILY_COUNT * _INNER_FAMILY_COUNT
    inner_providers = (
        inner_response_providers
        + inner_missing_anchor_providers
        + inner_vertex_providers
    )
    inner_providers_per_outer_fold = (
        _INNER_FAMILY_COUNT * (len(_RESPONSES) + 3)
    )
    outer_providers = _FAMILY_COUNT * len(_ARMS)
    total_forwards = (
        authority_forwards + endpoint_forwards + inner_forwards + outer_forwards
    )
    authority_backwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY
    total_backwards = authority_backwards + endpoint_forwards
    teacher_accesses = endpoint_forwards + inner_forwards + outer_forwards
    if (
        total_forwards != 2752
        or total_backwards != 128
        or endpoint_forwards != 112
        or inner_forwards != 2464
        or inner_response_forwards != 2128
        or inner_missing_anchor_forwards != 224
        or inner_vertex_forwards != 112
        or outer_forwards != 144
        or teacher_accesses != 2720
        or inner_providers != 1232
        or inner_response_providers != 1064
        or inner_missing_anchor_providers != 112
        or inner_vertex_providers != 56
        or inner_providers_per_outer_fold != 154
        or outer_providers != 72
    ):
        raise RuntimeError("V20o canonical work schedule drifted")
    return {
        "accounting_scope": "canonical_one_shot_schedule",
        "resume_and_authentication_overhead_excluded": True,
        "live_authority_collection_model_forward_count": authority_forwards,
        "endpoint_reconstruction_model_forward_count": endpoint_forwards,
        "inner_conditional_leave_one_family_out_model_forward_count": (
            inner_forwards
        ),
        "inner_original_response_model_forward_count": inner_response_forwards,
        "inner_signed_continuum_missing_anchor_model_forward_count": (
            inner_missing_anchor_forwards
        ),
        "inner_signed_continuum_vertex_model_forward_count": (
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
        "signed_continuum_missing_anchor_candidate_count": (
            inner_missing_anchor_providers
        ),
        "signed_continuum_vertex_candidate_count": (
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
        "inner_signed_continuum_missing_anchor_trace_example_count": (
            inner_missing_anchor_forwards
        ),
        "inner_signed_continuum_vertex_trace_example_count": (
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
    v20n_report: Mapping[str, object],
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
            decision, domain=_DECISION_DOMAIN, label="V20o aggregate decision"
        )
    )
    families = tuple(
        _identifier(item, label="V20o report family")
        for item in _sequence(
            aggregate.get("family_ids"), label="V20o report families"
        )
    )
    if (
        len(families) != _FAMILY_COUNT
        or set(fold_fragments) != set(families)
        or set(folds) != set(families)
    ):
        raise RuntimeError("V20o report requires all eight authenticated folds")
    replayed = _aggregate_decision(folds)
    if _v14._canonical_json_bytes(replayed) != _v14._canonical_json_bytes(
        aggregate
    ):
        raise ValueError("V20o supplied decision differs from fold replay")
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
        "signed_continuum_fit_protocol_sha256": (
            _signed_continuum_fit.SOFT_POLARITY_SIGNED_CONTINUUM_FIT_PROTOCOL_SHA256
        ),
        "signed_continuum_provider_protocol_sha256": (
            FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256
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
        "v20n_authority": {
            "report_sha256": _V20N_LOGICAL_SHA256,
            "file_sha256": _V20N_FILE_SHA256,
            "source_receipt_sha256": _V20N_SOURCE_SHA256,
            "classification": v20n_report.get("classification"),
            "development_oof_passed": v20n_report.get(
                "development_oof_passed"
            ),
            "primary_development_gate_passed": v20n_report.get(
                "primary_development_gate_passed"
            ),
            "mechanism_gate_passed": v20n_report.get(
                "mechanism_gate_passed"
            ),
            "integrity_passed": _mapping(
                v20n_report.get("decision"),
                label="V20o V20n decision authority",
            ).get("integrity_passed"),
            "passed": v20n_report.get("passed"),
            "rollback_to_base": v20n_report.get("rollback_to_base"),
            "final_refit": None,
            "calibration_b_opened": False,
            "fold_fragment_sha256s_by_family": {
                family: source["v20n_fold_fragment_sha256s_by_family"][family]
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
            "soft_polarity_signed_continuum_nested_oof_passed_fresh_shadow_eligible"
            if passed
            else "soft_polarity_signed_continuum_nested_oof_failed_rollback_to_base"
        ),
        "passed": passed,
        "development_oof_passed": passed,
        "primary_development_gate_passed": (
            aggregate.get("primary_development_gate_passed") is True
        ),
        "mechanism_gate_passed": aggregate.get("mechanism_gate_passed") is True,
        "continuous_signed_continuum_evidence_gate_passed": aggregate.get(
            "continuous_signed_continuum_evidence_gate_passed"
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
            else "revise_continuous_signed_continuum_fit_then_repeat_nested_OOF"
        ),
        "work_accounting": _runner_work_accounting(),
        "integrity": {
            "V20g_through_V20n_reports_and_all_fragments_authenticated_before_model_"
            "construction": True,
            "all_56_masked_directions_use_only_six_training_family_summaries": True,
            "inner_response_selection_is_conditional_on_each_fixed_seven_"
            "family_endpoint_not_full_inner_model_cross_validation": True,
            "all_eight_outer_held_families_absent_from_endpoint_direction_"
            "reflection_response_and_signed_scalar_selection": True,
            "all_133_V20m_response_providers_and_traces_staged_within_each_"
            "outer_fold_before_corresponding_response_scoring": True,
            "all_14_missing_anchor_providers_and_traces_staged_within_each_"
            "outer_fold_before_corresponding_missing_anchor_scoring": True,
            "all_7_vertex_providers_and_traces_staged_within_each_outer_fold_"
            "before_corresponding_vertex_scoring": True,
            "all_nine_outer_providers_and_traces_frozen_before_each_outer_"
            "capability": True,
            "all_inner_zero_response_V20g_eta_zero_output_anchors_passed": True,
            "all_outer_base_fixed_plus_fixed_minus_V20g_output_anchors_"
            "passed": True,
            "all_matched_V20l_boundary_exact_output_anchors_passed": True,
            "all_matched_V20m_response_and_outer_output_anchors_passed": True,
            "all_exact_mirror_V20m_outer_output_anchors_passed": True,
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
    v20n_report: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    authenticated_v20g_folds: Mapping[str, Mapping[str, object]],
    authenticated_v20i_folds: Mapping[str, Mapping[str, object]],
    authenticated_v20l_folds: Mapping[str, Mapping[str, object]],
    authenticated_v20m_folds: Mapping[str, Mapping[str, object]],
    authenticated_v20n_folds: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20o signed-continuum nested report",
    )
    families = tuple(sorted(authenticated_v20g_folds))
    if (
        set(authenticated_v20i_folds) != set(families)
        or set(authenticated_v20l_folds) != set(families)
        or set(authenticated_v20m_folds) != set(families)
        or set(authenticated_v20n_folds) != set(families)
    ):
        raise ValueError("V20o report authority family geometry differs")
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
            authenticated_v20n_fold=authenticated_v20n_folds[family],
        )
        for family in families
    }
    rebuilt = _build_report(
        output=output,
        source=source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        v20m_report=v20m_report,
        v20n_report=v20n_report,
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
        raise ValueError("V20o report reconstruction differs")
    return dict(value)


def run_gemma3_l3_l4_complete_h4_soft_polarity_signed_continuum_nested_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or resume the V20o continuous signed-continuum development screen."""

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
        v20n_report,
        authenticated_v20n_folds,
        source,
    ) = _load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"), label="V20o panel receipt"
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20o bridge binding",
    )
    if destination.exists():
        return _load_existing_report(
            destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            v20m_report=v20m_report,
            v20n_report=v20n_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_folds=authenticated_v20g_folds,
            authenticated_v20i_folds=authenticated_v20i_folds,
            authenticated_v20l_folds=authenticated_v20l_folds,
            authenticated_v20m_folds=authenticated_v20m_folds,
            authenticated_v20n_folds=authenticated_v20n_folds,
        )

    family_ids = tuple(sorted(authenticated_v20g_folds))
    if (
        len(family_ids) != _FAMILY_COUNT
        or set(authenticated_v20a_folds) != set(family_ids)
        or set(authenticated_v20i_folds) != set(family_ids)
        or set(authenticated_v20l_folds) != set(family_ids)
        or set(authenticated_v20m_folds) != set(family_ids)
        or set(authenticated_v20n_folds) != set(family_ids)
        or set(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="V20o panel families",
            )
        )
        != set(family_ids)
    ):
        raise RuntimeError("V20o authenticated family geometry differs")

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
                authenticated_v20n_fold=authenticated_v20n_folds[family],
            )
            for family in family_ids
        }
        report = _build_report(
            output=destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            v20m_report=v20m_report,
            v20n_report=v20n_report,
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
                label="V20o signed-continuum nested report",
            )
        except FileExistsError:
            pass
        return _load_existing_report(
            destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            v20m_report=v20m_report,
            v20n_report=v20n_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_folds=authenticated_v20g_folds,
            authenticated_v20i_folds=authenticated_v20i_folds,
            authenticated_v20l_folds=authenticated_v20l_folds,
            authenticated_v20m_folds=authenticated_v20m_folds,
            authenticated_v20n_folds=authenticated_v20n_folds,
        )

    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        if context.bridge.bridge_binding_sha256 != bridge_binding:
            raise RuntimeError("V20o live bridge differs from authenticated authority")
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=prerequisite
        )
        if tuple(live_families) != family_ids:
            raise RuntimeError("V20o live family order differs from authenticated A16")
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
                    authenticated_v20n_fold=authenticated_v20n_folds[family],
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
                authenticated_v20n_fold=authenticated_v20n_folds[family],
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
                authenticated_v20n_fold=authenticated_v20n_folds[family],
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
                authenticated_v20n_fold=authenticated_v20n_folds[family],
            )
        report = _build_report(
            output=destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            v20m_report=v20m_report,
            v20n_report=v20n_report,
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
            label="V20o signed-continuum nested report",
        )
    except FileExistsError:
        pass
    return _load_existing_report(
        destination,
        source=source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        v20m_report=v20m_report,
        v20n_report=v20n_report,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_v20g_folds=authenticated_v20g_folds,
        authenticated_v20i_folds=authenticated_v20i_folds,
        authenticated_v20l_folds=authenticated_v20l_folds,
        authenticated_v20m_folds=authenticated_v20m_folds,
        authenticated_v20n_folds=authenticated_v20n_folds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V20o nested soft-polarity signed-continuum development "
            "screen"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = (
        run_gemma3_l3_l4_complete_h4_soft_polarity_signed_continuum_nested_development(
            output=arguments.output,
            cache_dir=arguments.cache_dir,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
